//! Safe token batches for [`decode`](crate::context::Context::decode).
//!
//! A [`Batch`] owns every array upstream reads during decode (token ids,
//! positions, sequence map, logits flags) and hands `llama_decode` a
//! by-value `llama_batch` view of them. The view borrows only memory the
//! `Batch` owns, and `Batch` offers no mutation after construction, so
//! the view stays valid until the `Batch` is dropped.
//!
//! Phase-1 scope: token batches (no embeddings) on a single implicit
//! sequence (id 0). Multi-sequence and embedding batches extend this
//! module later without changing the [`Context`](crate::context::Context)
//! API.

use crate::error::{Error, Result};
use std::os::raw::c_int;

/// Token id. Always non-negative; [`BatchBuilder`] rejects ids outside
/// the model's vocabulary.
pub type TokenId = u32;

/// Incremental batch constructor. Validates token ids against the
/// model's vocabulary up front so malformed batches fail before any
/// native call.
pub struct BatchBuilder {
    vocab_size: u32,
    tokens: Vec<TokenId>,
    positions: Vec<u32>,
    want_logits: Vec<bool>,
}

impl BatchBuilder {
    /// Start a batch for a model with `vocab_size` tokens.
    pub fn new(vocab_size: u32) -> Self {
        Self {
            vocab_size,
            tokens: Vec::new(),
            positions: Vec::new(),
            want_logits: Vec::new(),
        }
    }

    /// Append one token. `want_logits` selects whether this position
    /// produces an output row (at least one position in a decode should
    /// usually set it; upstream decides the rest).
    pub fn push(&mut self, token: TokenId, pos: u32, want_logits: bool) -> Result<()> {
        if token >= self.vocab_size {
            return Err(Error::batch(format!(
                "token id {token} out of range (vocab size {})",
                self.vocab_size
            )));
        }
        if self.tokens.len() >= c_int::MAX as usize {
            return Err(Error::batch("batch too large"));
        }
        self.tokens.push(token);
        self.positions.push(pos);
        self.want_logits.push(want_logits);
        Ok(())
    }

    /// Number of tokens staged so far.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether no token has been staged yet.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Freeze the staged tokens into an immutable [`Batch`].
    pub fn build(self) -> Result<Batch> {
        if self.tokens.is_empty() {
            return Err(Error::batch("batch must contain at least one token"));
        }
        let mut tokens = Vec::with_capacity(self.tokens.len());
        for token in self.tokens {
            tokens.push(
                c_int::try_from(token)
                    .map_err(|_| Error::batch(format!("token id {token} too large")))?,
            );
        }
        let mut positions = Vec::with_capacity(self.positions.len());
        for pos in self.positions {
            positions.push(
                c_int::try_from(pos)
                    .map_err(|_| Error::batch(format!("position {pos} too large")))?,
            );
        }
        let logits_flags: Vec<i8> = self
            .want_logits
            .iter()
            .map(|want| i8::from(*want))
            .collect();
        // Single implicit sequence: every token maps to one shared,
        // heap-stable sequence id. `seq_ptrs` borrows only this `Box`,
        // which `Batch` never moves or reallocates after construction.
        let mut seq_id = Box::new(0 as c_int);
        let seq_ptr: *mut c_int = &mut *seq_id;
        let n = tokens.len();
        Ok(Batch {
            tokens,
            positions,
            n_seq_ids: vec![1 as c_int; n],
            seq_ptrs: vec![seq_ptr; n],
            seq_id,
            logits_flags,
        })
    }
}

/// An immutable, owned token batch ready for decode.
///
/// `!Send + !Sync` (raw view pointers); dropping frees only Rust memory
/// (upstream allocates nothing for the view itself).
pub struct Batch {
    tokens: Vec<c_int>,
    positions: Vec<c_int>,
    n_seq_ids: Vec<c_int>,
    seq_ptrs: Vec<*mut c_int>,
    // Backing store borrowed by `seq_ptrs`; never read, only kept alive.
    #[allow(dead_code)]
    seq_id: Box<c_int>,
    logits_flags: Vec<i8>,
}

impl Batch {
    /// Number of tokens in the batch (always at least 1).
    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// By-value upstream view over this batch's owned arrays.
    pub(crate) fn as_sys(&self) -> forge_sys::llama_batch {
        forge_sys::llama_batch {
            // SAFETY: `BatchBuilder` capped the length below `c_int::MAX`.
            n_tokens: self.tokens.len() as c_int,
            token: self.tokens.as_ptr().cast_mut(),
            embd: std::ptr::null_mut(),
            pos: self.positions.as_ptr().cast_mut(),
            n_seq_id: self.n_seq_ids.as_ptr().cast_mut(),
            seq_id: self.seq_ptrs.as_ptr().cast_mut(),
            logits: self.logits_flags.as_ptr().cast_mut(),
        }
    }
}

impl std::fmt::Debug for Batch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Batch")
            .field("n_tokens", &self.n_tokens())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_accepts_valid_tokens() {
        let mut builder = BatchBuilder::new(32);
        assert!(builder.is_empty());
        builder.push(1, 0, false).unwrap();
        builder.push(5, 1, true).unwrap();
        assert_eq!(builder.len(), 2);
        let batch = builder.build().unwrap();
        assert_eq!(batch.n_tokens(), 2);
    }

    #[test]
    fn builder_rejects_out_of_vocab_token() {
        let mut builder = BatchBuilder::new(32);
        assert!(builder.push(32, 0, false).is_err());
        assert!(builder.push(0, 0, false).is_ok());
    }

    #[test]
    fn builder_rejects_empty_build() {
        let builder = BatchBuilder::new(32);
        assert!(builder.build().is_err());
    }

    #[test]
    fn builder_rejects_position_beyond_i32() {
        let mut builder = BatchBuilder::new(32);
        builder.push(1, u32::MAX, false).unwrap();
        assert!(builder.build().is_err());
    }
}
