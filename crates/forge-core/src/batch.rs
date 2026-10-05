//! Safe token and embedding batches for [`decode`](crate::context::Context::decode).
//!
//! A [`Batch`] owns every array upstream reads during decode (token ids
//! or embedding rows, positions, sequence map, logits flags) and hands
//! `llama_decode` a by-value `llama_batch` view of them. The view borrows
//! only memory the [`Batch`] owns, and [`Batch`] offers no mutation after
//! construction, so the view stays valid until the [`Batch`] is dropped.
//!
//! Token batches carry vocabulary ids; embedding batches carry one
//! `n_embd`-wide `f32` row per token (see
//! [`Model::n_embd_inp`](crate::model::Model::n_embd_inp)) and decode on
//! any context, producing logits from the supplied rows. Every token
//! belongs to at least one sequence; tokens may be shared across
//! sequences (upstream "coupled" sequences). Upstream validates token
//! ranges, sequence ranges, and position consecutiveness itself and
//! reports violations as decode errors — the builder additionally
//! rejects what upstream cannot take safely (empty sequence lists, mode
//! mixing, dimension mismatches).

use crate::error::{Error, Result};
use std::os::raw::c_int;

/// Token id. Always non-negative; [`BatchBuilder`] rejects ids outside
/// the model's vocabulary.
pub type TokenId = u32;

/// Sequence id. Always non-negative; [`Context::decode`](crate::context::Context::decode)
/// rejects ids at or above the context's effective `n_seq_max`.
pub type SeqId = u32;

/// Maximum sequence ids per token: upstream tracks at most
/// `LLAMA_MAX_SEQ` (256) distinct sequences, so a longer per-token list
/// could only repeat ids.
const MAX_SEQS_PER_TOKEN: usize = 256;

/// Incremental batch constructor. Token mode ([`new`](BatchBuilder::new))
/// stages vocabulary ids; embedding mode
/// ([`new_embeddings`](BatchBuilder::new_embeddings)) stages `f32` rows.
/// The mode is fixed at construction and mixing is rejected.
pub struct BatchBuilder {
    vocab_size: u32,
    tokens: Vec<TokenId>,
    embd: Vec<f32>,
    /// `Some(width)` in embedding mode, `None` in token mode.
    n_embd: Option<usize>,
    positions: Vec<u32>,
    seq_ids: Vec<Vec<SeqId>>,
    want_logits: Vec<bool>,
}

impl BatchBuilder {
    /// Start a token batch for a model with `vocab_size` tokens.
    pub fn new(vocab_size: u32) -> Self {
        Self {
            vocab_size,
            tokens: Vec::new(),
            embd: Vec::new(),
            n_embd: None,
            positions: Vec::new(),
            seq_ids: Vec::new(),
            want_logits: Vec::new(),
        }
    }

    /// Start an embedding batch with `n_embd`-wide rows (see
    /// [`Model::n_embd_inp`](crate::model::Model::n_embd_inp)). A zero
    /// width is rejected: it would hand upstream a dangling pointer
    /// with no backing floats.
    pub fn new_embeddings(n_embd: usize) -> Result<Self> {
        if n_embd == 0 {
            return Err(Error::batch("embedding width must be at least 1"));
        }
        Ok(Self {
            vocab_size: 0,
            tokens: Vec::new(),
            embd: Vec::new(),
            n_embd: Some(n_embd),
            positions: Vec::new(),
            seq_ids: Vec::new(),
            want_logits: Vec::new(),
        })
    }

    /// Append one token on the implicit sequence 0. `want_logits`
    /// selects whether this position produces an output row.
    pub fn push(&mut self, token: TokenId, pos: u32, want_logits: bool) -> Result<()> {
        self.push_on_sequences(token, pos, &[0], want_logits)
    }

    /// Append one token belonging to `seq_ids`. At least one sequence
    /// is required (upstream reads `seq_id[i][0]` unconditionally) and
    /// at most 256 are accepted (upstream tracks at most 256 distinct
    /// sequences, so a longer list could only repeat ids).
    pub fn push_on_sequences(
        &mut self,
        token: TokenId,
        pos: u32,
        seq_ids: &[SeqId],
        want_logits: bool,
    ) -> Result<()> {
        if self.n_embd.is_some() {
            return Err(Error::batch(
                "cannot push a token id into an embedding batch",
            ));
        }
        if token >= self.vocab_size {
            return Err(Error::batch(format!(
                "token id {token} out of range (vocab size {})",
                self.vocab_size
            )));
        }
        Self::check_seq_list(seq_ids)?;
        if self.tokens.len() >= c_int::MAX as usize {
            return Err(Error::batch("batch too large"));
        }
        self.tokens.push(token);
        self.positions.push(pos);
        self.seq_ids.push(seq_ids.to_vec());
        self.want_logits.push(want_logits);
        Ok(())
    }

    /// Append one embedding row on the implicit sequence 0. The row must
    /// hold exactly the width given to [`new_embeddings`](BatchBuilder::new_embeddings).
    pub fn push_embd(&mut self, embd: &[f32], pos: u32, want_logits: bool) -> Result<()> {
        self.push_embd_on_sequences(embd, pos, &[0], want_logits)
    }

    /// Append one embedding row belonging to `seq_ids` (see
    /// [`push_on_sequences`](BatchBuilder::push_on_sequences) for the
    /// sequence rules).
    pub fn push_embd_on_sequences(
        &mut self,
        embd: &[f32],
        pos: u32,
        seq_ids: &[SeqId],
        want_logits: bool,
    ) -> Result<()> {
        let Some(n_embd) = self.n_embd else {
            return Err(Error::batch(
                "cannot push an embedding row into a token batch",
            ));
        };
        if embd.len() != n_embd {
            return Err(Error::batch(format!(
                "embedding row has {} floats, expected {n_embd}",
                embd.len()
            )));
        }
        Self::check_seq_list(seq_ids)?;
        if self.positions.len() >= c_int::MAX as usize {
            return Err(Error::batch("batch too large"));
        }
        self.embd.extend_from_slice(embd);
        self.positions.push(pos);
        self.seq_ids.push(seq_ids.to_vec());
        self.want_logits.push(want_logits);
        Ok(())
    }

    fn check_seq_list(seq_ids: &[SeqId]) -> Result<()> {
        if seq_ids.is_empty() {
            return Err(Error::batch("each token needs at least one sequence id"));
        }
        if seq_ids.len() > MAX_SEQS_PER_TOKEN {
            return Err(Error::batch(format!(
                "too many sequence ids per token ({} > {MAX_SEQS_PER_TOKEN})",
                seq_ids.len()
            )));
        }
        Ok(())
    }

    /// Number of tokens staged so far.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether no token has been staged yet.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Drop all staged tokens, keeping the allocations for reuse.
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.embd.clear();
        self.positions.clear();
        self.seq_ids.clear();
        self.want_logits.clear();
    }

    /// Freeze the staged tokens into an immutable [`Batch`].
    pub fn build(self) -> Result<Batch> {
        if self.positions.is_empty() {
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
        // Flatten the per-token sequence lists into one stable store;
        // `seq_ptrs[i]` borrows the run starting at `seq_offsets[i]`.
        // The store is filled completely before any pointer is taken,
        // and `Batch` never reallocates it afterwards.
        let mut seq_store = Vec::new();
        let mut seq_offsets = Vec::with_capacity(self.seq_ids.len());
        let mut n_seq_ids = Vec::with_capacity(self.seq_ids.len());
        for seq_ids in &self.seq_ids {
            seq_offsets.push(seq_store.len());
            // SAFETY: every list holds 1..=256 entries by construction.
            n_seq_ids.push(seq_ids.len() as c_int);
            for seq in seq_ids {
                seq_store.push(
                    c_int::try_from(*seq)
                        .map_err(|_| Error::batch(format!("seq id {seq} too large")))?,
                );
            }
        }
        let mut batch = Batch {
            tokens,
            embd: self.embd,
            positions,
            n_seq_ids,
            seq_ptrs: Vec::with_capacity(seq_offsets.len()),
            seq_store,
            seq_offsets,
            logits_flags: self
                .want_logits
                .iter()
                .map(|want| i8::from(*want))
                .collect(),
            n_embd: self.n_embd,
        };
        for offset in batch.seq_offsets.iter() {
            // SAFETY: `offset` is the recorded run start for a
            // non-empty run; the store is fully built and never moves
            // again (no `&mut` methods on `Batch`).
            batch
                .seq_ptrs
                .push(unsafe { batch.seq_store.as_mut_ptr().add(*offset) });
        }
        Ok(batch)
    }
}

/// An immutable, owned token or embedding batch ready for decode.
///
/// Exactly one of the token/embedding arrays is populated, so the
/// native view carries exactly one non-NULL input pointer. `!Send +
/// !Sync` (raw view pointers); dropping frees only Rust memory
/// (upstream allocates nothing for the view itself).
pub struct Batch {
    tokens: Vec<c_int>,
    embd: Vec<f32>,
    positions: Vec<c_int>,
    n_seq_ids: Vec<c_int>,
    seq_ptrs: Vec<*mut c_int>,
    // Backing store borrowed by `seq_ptrs` via `seq_offsets`; never
    // reallocated after construction.
    seq_store: Vec<c_int>,
    seq_offsets: Vec<usize>,
    logits_flags: Vec<i8>,
    n_embd: Option<usize>,
}

impl Batch {
    /// Number of tokens in the batch (always at least 1).
    pub fn n_tokens(&self) -> usize {
        self.positions.len()
    }

    /// Whether this is an embedding batch (as opposed to token ids).
    pub fn is_embd(&self) -> bool {
        self.n_embd.is_some()
    }

    /// Embedding row width for embedding batches, `None` for token
    /// batches.
    pub fn n_embd(&self) -> Option<usize> {
        self.n_embd
    }

    /// Number of positions flagged for output.
    pub fn n_outputs(&self) -> usize {
        self.logits_flags.iter().filter(|flag| **flag != 0).count()
    }

    /// Sorted distinct sequence ids participating in the batch.
    pub fn sequences(&self) -> Vec<SeqId> {
        let mut seqs: Vec<SeqId> = self.seq_store.iter().map(|seq| *seq as SeqId).collect();
        seqs.sort_unstable();
        seqs.dedup();
        seqs
    }

    /// Every sequence id in the batch (flat across tokens), for
    /// context-side range validation.
    pub(crate) fn all_seq_ids(&self) -> impl Iterator<Item = SeqId> + '_ {
        self.seq_store.iter().map(|seq| *seq as SeqId)
    }

    /// By-value upstream view over this batch's owned arrays.
    pub(crate) fn as_sys(&self) -> forge_sys::llama_batch {
        forge_sys::llama_batch {
            // SAFETY: builders capped the length below `c_int::MAX`.
            n_tokens: self.positions.len() as c_int,
            token: if self.tokens.is_empty() {
                std::ptr::null_mut()
            } else {
                self.tokens.as_ptr().cast_mut()
            },
            embd: if self.embd.is_empty() {
                std::ptr::null_mut()
            } else {
                self.embd.as_ptr().cast_mut()
            },
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
            .field("is_embd", &self.is_embd())
            .field("n_outputs", &self.n_outputs())
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
        assert!(!batch.is_embd());
        assert_eq!(batch.n_embd(), None);
        assert_eq!(batch.n_outputs(), 1);
        assert_eq!(batch.sequences(), vec![0]);
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

    #[test]
    fn builder_accepts_multiple_sequences() {
        let mut builder = BatchBuilder::new(32);
        builder.push_on_sequences(1, 0, &[0, 1], true).unwrap();
        builder.push_on_sequences(2, 0, &[1], false).unwrap();
        builder.push_on_sequences(3, 1, &[0], true).unwrap();
        let batch = builder.build().unwrap();
        assert_eq!(batch.n_tokens(), 3);
        assert_eq!(batch.sequences(), vec![0, 1]);
        assert_eq!(batch.n_outputs(), 2);
    }

    #[test]
    fn builder_rejects_empty_seq_list() {
        let mut builder = BatchBuilder::new(32);
        assert!(builder.push_on_sequences(1, 0, &[], false).is_err());
    }

    #[test]
    fn builder_rejects_too_many_seqs_per_token() {
        let mut builder = BatchBuilder::new(32);
        let many = vec![0 as SeqId; MAX_SEQS_PER_TOKEN + 1];
        assert!(builder.push_on_sequences(1, 0, &many, false).is_err());
        let max = vec![0 as SeqId; MAX_SEQS_PER_TOKEN];
        assert!(builder.push_on_sequences(1, 0, &max, false).is_ok());
    }

    #[test]
    fn builder_rejects_seq_beyond_i32() {
        let mut builder = BatchBuilder::new(32);
        builder.push_on_sequences(1, 0, &[u32::MAX], false).unwrap();
        assert!(builder.build().is_err());
    }

    #[test]
    fn embedding_builder_round_trips() {
        let mut builder = BatchBuilder::new_embeddings(8).unwrap();
        builder.push_embd(&[0.5; 8], 0, false).unwrap();
        builder
            .push_embd_on_sequences(&[1.5; 8], 1, &[0, 2], true)
            .unwrap();
        assert_eq!(builder.len(), 2);
        let batch = builder.build().unwrap();
        assert!(batch.is_embd());
        assert_eq!(batch.n_embd(), Some(8));
        assert_eq!(batch.sequences(), vec![0, 2]);
    }

    #[test]
    fn embedding_builder_rejects_bad_width() {
        assert!(BatchBuilder::new_embeddings(0).is_err());
        let mut builder = BatchBuilder::new_embeddings(8).unwrap();
        assert!(builder.push_embd(&[0.0; 7], 0, false).is_err());
        assert!(builder.push_embd(&[0.0; 8], 0, false).is_ok());
    }

    #[test]
    fn builder_rejects_mode_mixing() {
        let mut tokens = BatchBuilder::new(32);
        assert!(tokens.push_embd(&[0.0; 8], 0, false).is_err());
        let mut embd = BatchBuilder::new_embeddings(8).unwrap();
        assert!(embd.push(1, 0, false).is_err());
    }

    #[test]
    fn builder_clear_reuses_staging() {
        let mut builder = BatchBuilder::new(32);
        builder.push(1, 0, true).unwrap();
        builder.clear();
        assert!(builder.is_empty());
        assert_eq!(builder.len(), 0);
        builder.push(2, 0, true).unwrap();
        assert_eq!(builder.build().unwrap().n_tokens(), 1);
    }
}
