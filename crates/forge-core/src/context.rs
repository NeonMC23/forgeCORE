//! Safe execution contexts over `llama_context`.
//!
//! A [`Context`] owns one native context plus a share of its [`Model`],
//! so the model always outlives every context built from it. The
//! phase-1 path is single-shot: build a [`Batch`](crate::batch::Batch),
//! [`decode`](Context::decode) it, read [`Logits`].
//!
//! [`ContextOptions`] is `#[non_exhaustive]` and exposes only what the
//! CPU path needs (`n_ctx`, `n_threads`); batching, KV, sequence, and
//! device options extend it later. Upstream owns all numerics, graph
//! construction, and KV-cache behavior — this module only validates
//! arguments, maps error codes, and guards lifetimes.
//!
//! Like all ForgeCore handles, `Context` is `!Send + !Sync` and frees
//! its native context on drop.

use crate::batch::Batch;
use crate::error::{Error, Result};
use crate::model::{Model, ModelInner};
use std::os::raw::c_int;
use std::rc::Rc;

/// Context creation options. `#[non_exhaustive]` so future phases can
/// add batching/KV/sequence/device fields without breaking callers.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ContextOptions {
    /// Context (KV) length. `0` selects the model's training length
    /// (`n_ctx_train`); any other value is used verbatim.
    pub n_ctx: u32,
    /// Worker thread count for generation and batch processing. Must be
    /// at least 1. The default favors determinism over throughput;
    /// RAMforge will set production counts.
    pub n_threads: u32,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            n_ctx: 0,
            n_threads: 1,
        }
    }
}

/// Owned logits for one decoded output position.
///
/// An **owned copy**: `values` holds exactly `n_vocab` `f32`s copied
/// out of the native buffer, so later decodes cannot invalidate it.
#[derive(Debug, Clone)]
pub struct Logits {
    values: Vec<f32>,
    n_vocab: usize,
}

impl Logits {
    /// Logit values, one per vocabulary id.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Vocabulary dimension (always equals `values().len()`).
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }
}

/// An owned libllama execution context bound to one [`Model`].
pub struct Context {
    raw: *mut forge_sys::llama_context,
    // Shared model handle: keeps the native model alive while any
    // context exists. Never read, only kept alive (Drop order frees
    // the context first, then releases this share).
    #[allow(dead_code)]
    model: Rc<ModelInner>,
    n_ctx: u32,
    n_vocab: u32,
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful init, is freed exactly once;
        // the shared model outlives us via the Rc below (Drop runs before
        // fields are released).
        unsafe { forge_sys::llama_free(self.raw) };
    }
}

impl Context {
    /// Create a context for `model` with explicit [`ContextOptions`].
    ///
    /// Returns [`Error`] for invalid options (e.g. `n_threads == 0`) or
    /// when upstream refuses the configuration (e.g. exhausted memory).
    pub fn open(model: &Model, options: &ContextOptions) -> Result<Self> {
        if options.n_threads == 0 {
            return Err(Error::invalid("context n_threads must be at least 1"));
        }
        let n_threads = c_int::try_from(options.n_threads)
            .map_err(|_| Error::invalid("context n_threads too large"))?;
        // SAFETY: default params are valid by construction; the model is
        // live; NULL return (init failure) is checked. The context keeps
        // a share of the model so it cannot outlive it.
        unsafe {
            let mut params = forge_sys::llama_context_default_params();
            params.n_ctx = options.n_ctx;
            params.n_threads = n_threads;
            params.n_threads_batch = n_threads;
            let raw = forge_sys::llama_init_from_model(model.raw(), params);
            if raw.is_null() {
                return Err(Error::context("context creation failed"));
            }
            let n_ctx = if options.n_ctx == 0 {
                model.n_ctx_train()?
            } else {
                options.n_ctx
            };
            let n_vocab = model.vocab_size()?;
            Ok(Self {
                raw,
                model: Rc::clone(model.inner()),
                n_ctx,
                n_vocab,
            })
        }
    }

    /// Effective context length (resolved when `n_ctx: 0` was requested).
    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    /// Vocabulary dimension used for [`Logits`].
    pub fn n_vocab(&self) -> u32 {
        self.n_vocab
    }

    /// Decode one [`Batch`], advancing this context's KV state.
    ///
    /// ForgeCore checks only what upstream cannot take safely (a
    /// non-empty batch; upstream aborts on `n_tokens <= 0`). Positions,
    /// batch sizing, sequence consecutiveness, and KV capacity are
    /// upstream's domain: violations come back as error codes and map to
    /// [`Error::decode`] with the code's meaning — nothing is hidden or
    /// retried.
    pub fn decode(&mut self, batch: &Batch) -> Result<()> {
        if batch.n_tokens() == 0 {
            return Err(Error::invalid("decode: batch is empty"));
        }
        // SAFETY: raw is a live context; the batch view borrows only
        // arrays the `&Batch` owns, which outlive this call; the model
        // is alive via our Rc. Upstream takes the batch by const
        // reference (`llama_context::decode(const llama_batch &)`) and
        // only reads these arrays for token batches.
        let code = unsafe { forge_sys::llama_decode(self.raw, batch.as_sys()) };
        match code {
            0 => Ok(()),
            1 => Err(Error::decode(
                "no KV slot for batch (try a smaller batch or larger context)",
            )),
            2 => Err(Error::decode("execution aborted")),
            -1 => Err(Error::decode("backend rejected the batch as invalid")),
            other => Err(Error::decode(format!(
                "fatal backend failure (code {other})"
            ))),
        }
    }

    /// Copy the logits for one token of the most recently decoded batch.
    ///
    /// `token_index` is the position within that batch (not a token id):
    /// it must be below the batch length and that position must have had
    /// its logits flag set, otherwise upstream returns NULL. Takes `&mut
    /// self` to match upstream (which synchronizes internal state);
    /// returns an owned [`Logits`] copy of `n_vocab` floats, or
    /// [`Error::logits`] when the index has no logits.
    pub fn logits(&mut self, token_index: u32) -> Result<Logits> {
        let index = c_int::try_from(token_index)
            .map_err(|_| Error::invalid("logits token index too large"))?;
        // SAFETY: raw is a live context; NULL (no such output) is
        // checked; on success upstream guarantees a row of exactly
        // n_vocab floats, which is copied before returning.
        unsafe {
            let ptr = forge_sys::llama_get_logits_ith(self.raw, index);
            if ptr.is_null() {
                return Err(Error::logits(format!(
                    "no logits for batch index {token_index}"
                )));
            }
            let values = std::slice::from_raw_parts(ptr, self.n_vocab as usize).to_vec();
            Ok(Logits {
                values,
                n_vocab: self.n_vocab as usize,
            })
        }
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("n_ctx", &self.n_ctx)
            .field("n_vocab", &self.n_vocab)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_options_default_is_model_ctx_single_thread() {
        let options = ContextOptions::default();
        assert_eq!(options.n_ctx, 0);
        assert_eq!(options.n_threads, 1);
    }
}
