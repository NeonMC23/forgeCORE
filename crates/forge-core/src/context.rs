//! Safe execution contexts over `llama_context`.
//!
//! A [`Context`] owns one native context plus a share of its [`Model`],
//! so the model always outlives every context built from it. Build a
//! [`Batch`], [`decode`](Context::decode) it, read
//! [`Logits`] (or [`Embeddings`] from an embeddings context).
//!
//! [`ContextOptions`] is `#[non_exhaustive]` and exposes the low-level
//! context/batch controls: sizing (`n_ctx`, `n_batch`, `n_ubatch`,
//! `n_seq_max`, output caps), threading, KV cache dtypes,
//! embeddings/pooling/attention selection, and the advisory
//! `offload_kqv`/`op_offload` flags. Sequence surgery lives on the
//! borrowed [`Memory`] handle and byte-oriented
//! snapshots on the `export/import_state` methods. Every
//! numeric sanitize upstream performs (context padding, batch
//! clamping, pooling/attention resolution) is queried back after `open`
//! and reported honestly — [`Context`] never trusts the request.
//!
//! [`decode`](Context::decode) refuses up front what upstream cannot
//! take safely: empty batches, batches larger than the effective
//! `n_batch` (upstream aborts), batches larger than the effective
//! `n_ubatch` under non-causal attention (upstream aborts), and
//! sequence ids outside the effective `n_seq_max` (unless unified KV
//! defers sequence validation to upstream). Token ranges,
//! position consecutiveness, and KV capacity stay upstream's domain and
//! map to [`Error::decode`] — nothing is hidden or retried.
//!
//! Like all ForgeCore handles, `Context` is `!Send + !Sync` and frees
//! its native context on drop.

use crate::batch::{Batch, SeqId};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::memory::{Memory, SeqState, State};
use crate::model::{Model, ModelInner};
use std::os::raw::c_int;
use std::rc::Rc;

/// Pooled-embedding mode (upstream `llama_pooling_type`).
///
/// `Unspecified` (the default) resolves to the model default, or
/// `None` when the model has none. Pooling only produces output on
/// embeddings contexts; on other contexts it is accepted and ignored.
/// `Rank` attaches a classifier head whose per-sequence rows hold
/// [`Model::n_cls_out`](crate::model::Model::n_cls_out) floats instead
/// of [`Model::n_embd_out`](crate::model::Model::n_embd_out).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PoolingType {
    /// Resolve from the model (or `None`). The default.
    #[default]
    Unspecified,
    /// No pooling: per-token outputs only.
    None,
    /// Mean-pool token rows per sequence.
    Mean,
    /// First-token (CLS) row per sequence.
    Cls,
    /// Last-token row per sequence.
    Last,
    /// Classifier-head ranks per sequence.
    Rank,
}

impl PoolingType {
    fn as_native(self) -> c_int {
        match self {
            Self::Unspecified => forge_sys::pooling_type::UNSPECIFIED,
            Self::None => forge_sys::pooling_type::NONE,
            Self::Mean => forge_sys::pooling_type::MEAN,
            Self::Cls => forge_sys::pooling_type::CLS,
            Self::Last => forge_sys::pooling_type::LAST,
            Self::Rank => forge_sys::pooling_type::RANK,
        }
    }

    /// Map a resolved upstream value back. Unknown values (only
    /// reachable via a corrupt model default) behave like `None`:
    /// upstream's output switch matches no arm and extracts nothing.
    fn from_native(value: c_int) -> Self {
        match value {
            forge_sys::pooling_type::MEAN => Self::Mean,
            forge_sys::pooling_type::CLS => Self::Cls,
            forge_sys::pooling_type::LAST => Self::Last,
            forge_sys::pooling_type::RANK => Self::Rank,
            _ => Self::None,
        }
    }
}

/// Attention masking (upstream `llama_attention_type`).
///
/// `Unspecified` (the default) resolves to the model default (causal
/// unless the model says otherwise). `NonCausal` additionally requires
/// every decoded batch to fit in one micro-batch (`n_tokens <=
/// n_ubatch`); upstream aborts otherwise, so [`decode`](Context::decode)
/// enforces it up front.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AttentionType {
    /// Resolve from the model. The default.
    #[default]
    Unspecified,
    /// Causal (autoregressive) masking.
    Causal,
    /// Non-causal (bidirectional) masking.
    NonCausal,
}

impl AttentionType {
    fn as_native(self) -> c_int {
        match self {
            Self::Unspecified => forge_sys::attention_type::UNSPECIFIED,
            Self::Causal => forge_sys::attention_type::CAUSAL,
            Self::NonCausal => forge_sys::attention_type::NON_CAUSAL,
        }
    }
}

/// Flash-attention selection (upstream `llama_flash_attn_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FlashAttnType {
    /// Upstream default: enabled except where unsupported.
    #[default]
    Auto,
    /// Never use flash attention.
    Disabled,
    /// Always use flash attention (required for tensor-split models
    /// and quantized V caches; upstream fails context creation
    /// otherwise).
    Enabled,
}

impl FlashAttnType {
    fn as_native(self) -> c_int {
        match self {
            Self::Auto => forge_sys::flash_attn_type::AUTO,
            Self::Disabled => forge_sys::flash_attn_type::DISABLED,
            Self::Enabled => forge_sys::flash_attn_type::ENABLED,
        }
    }
}

/// Context creation options. `#[non_exhaustive]` so future phases can
/// add fields without breaking callers.
///
/// Every field defaults to the pre-P4 behavior: causal decoder
/// context, upstream batch sizing and thread lockstep. Requested
/// values are sanitized by upstream at init (context padding, batch
/// clamping, pooling/attention resolution); the effective values are
/// queried back and exposed on [`Context`].
///
/// Deliberately absent (kept at upstream defaults): RoPE/YaRN tuning,
/// recurrent rollback snapshots, MTP context type, SWA sizing and
/// full-cache selection, performance counters (P7), callbacks, and
/// backend sampler chains. See the P4/P5 engineering reports for the
/// per-field rationale.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ContextOptions {
    /// Context (KV) length. `0` selects the model's training length;
    /// any other value is used verbatim, then padded up to a multiple
    /// of 256 and rounded to a multiple of `n_seq_max` by upstream.
    pub n_ctx: u32,
    /// Worker thread count for generation. Must be at least 1. The
    /// default favors determinism over throughput; RAMforge will set
    /// production counts.
    pub n_threads: u32,
    /// Logical maximum batch size for [`decode`](Context::decode).
    /// Must be at least 1 (upstream aborts while opening a context
    /// with `n_batch == 0`). Under causal attention upstream clamps
    /// this to the context length.
    pub n_batch: u32,
    /// Physical micro-batch size; larger batches are split internally.
    /// `0` selects `n_batch` (upstream rule). Under non-causal
    /// attention every batch must fit in one micro-batch.
    pub n_ubatch: u32,
    /// Maximum distinct sequences per batch (`0` behaves as 1;
    /// upstream refuses values above 256 with a creation failure).
    /// The context length is rounded to a multiple of this value.
    pub n_seq_max: u32,
    /// Worker thread count for batch processing. `0` (the default)
    /// follows `n_threads`, preserving the historical lockstep; any
    /// other value must be at least 1 and is used verbatim.
    pub n_threads_batch: u32,
    /// Graph-sizing hint: maximum outputs per micro-batch (`0`, the
    /// default, selects `n_batch`). Exceeding it only costs a
    /// re-reserve, never an error.
    pub n_outputs_max: u32,
    /// Graph-sizing hint: maximum outputs per sequence (`0` selects
    /// `n_outputs_max`).
    pub n_outputs_max_per_seq: u32,
    /// Advisory: allow the KQV ops (including KV cache buffers) to use
    /// GPU placement when a GPU is available. Upstream default `true`;
    /// a no-op on CPU-only systems (there is nothing to place) and
    /// never a refusal: this flag permits GPU use, it does not request
    /// GPU execution, so the P3 no-silent-fallback rule does not apply.
    pub offload_kqv: bool,
    /// Advisory: allow the scheduler to move host-weight ops onto a
    /// capable device. Same no-op-on-CPU semantics as `offload_kqv`.
    pub op_offload: bool,
    /// Use one unified KV buffer across input sequences instead of a
    /// per-sequence split. Required for coupled batches (one token
    /// shared across sequences); without it upstream fails such
    /// batches with a native error. Also lifts sequence-id validation
    /// to the native 256-sequence bound (enforced by upstream, since
    /// the constant lives in an internal header ForgeCore does not
    /// duplicate) and reports the full context length per sequence.
    pub kv_unified: bool,
    /// Extract embeddings alongside logits (see [`Embeddings`]).
    /// Embedding-*input* batches decode on any context; this flag only
    /// controls embedding *output* extraction.
    pub embeddings: bool,
    /// Pooled-embedding mode for embedding outputs.
    pub pooling: PoolingType,
    /// Attention masking.
    pub attention: AttentionType,
    /// Flash-attention selection.
    pub flash_attn: FlashAttnType,
    /// Data type for the K cache. Upstream default `F16`.
    /// Combinations upstream cannot build fail context creation
    /// (MLA/DeepSeek-V4 models require K and V to match; a quantized V
    /// cache requires flash attention; quantized block sizes must
    /// divide the head dimensions). Recurrent (Mamba/RWKV)
    /// architectures force `F32` regardless of these fields (verified
    /// in the native memory factory). There is no native getter for
    /// the effective types; the applied width shows up in
    /// [`Context::state_size`].
    pub type_k: DType,
    /// Data type for the V cache. See [`ContextOptions::type_k`].
    pub type_v: DType,
}

impl Default for ContextOptions {
    fn default() -> Self {
        Self {
            n_ctx: 0,
            n_threads: 1,
            // Upstream defaults, mirrored so default options keep
            // passing upstream exactly what plain opens always have.
            n_batch: 2048,
            n_ubatch: 512,
            n_seq_max: 1,
            n_threads_batch: 0,
            n_outputs_max: 0,
            n_outputs_max_per_seq: 1,
            offload_kqv: true,
            op_offload: true,
            kv_unified: false,
            embeddings: false,
            pooling: PoolingType::Unspecified,
            attention: AttentionType::Unspecified,
            flash_attn: FlashAttnType::Auto,
            type_k: DType::F16,
            type_v: DType::F16,
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

/// Owned embedding row for one decoded output.
///
/// An **owned copy** like [`Logits`]: per-token rows hold `n_embd_out`
/// floats; pooled per-sequence rows hold `n_embd_out` floats, or
/// `n_cls_out` floats under [`PoolingType::Rank`].
#[derive(Debug, Clone)]
pub struct Embeddings {
    values: Vec<f32>,
    width: usize,
}

impl Embeddings {
    /// Embedding values for the requested token or sequence.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Row width (always equals `values().len()`).
    pub fn width(&self) -> usize {
        self.width
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
    // Effective values, queried back from upstream at `open` (never
    // trusted from the request).
    n_ctx: u32,
    n_ctx_seq: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_vocab: u32,
    n_threads: u32,
    n_threads_batch: u32,
    causal_attn: bool,
    pooling: PoolingType,
    kv_unified: bool,
    n_embd_out: u32,
    n_cls_out: u32,
    // Native parallel-sequence limit (unified seq-id bound) and
    // whether position shifts are legal on this model (refused on
    // multi-position MROPE/IMROPE models, where upstream aborts).
    max_parallel_seqs: u32,
    shift_allowed: bool,
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
    /// Returns [`Error`] for invalid options (`n_threads == 0`,
    /// `n_batch == 0`, unrepresentable thread counts) or when upstream
    /// refuses the configuration (exhausted memory, `n_seq_max > 256`,
    /// tensor-split without flash attention, ...).
    pub fn open(model: &Model, options: &ContextOptions) -> Result<Self> {
        if options.n_threads == 0 {
            return Err(Error::invalid("context n_threads must be at least 1"));
        }
        if options.n_batch == 0 {
            // Not a native failure: upstream aborts inside the
            // constructor (assert on the output sizing), so this must
            // never reach the native call.
            return Err(Error::invalid("context n_batch must be at least 1"));
        }
        let n_threads = c_int::try_from(options.n_threads)
            .map_err(|_| Error::invalid("context n_threads too large"))?;
        let threads_batch = if options.n_threads_batch == 0 {
            options.n_threads
        } else {
            options.n_threads_batch
        };
        let n_threads_batch = c_int::try_from(threads_batch)
            .map_err(|_| Error::invalid("context n_threads_batch too large"))?;
        // SAFETY: default params are valid by construction; the model is
        // live; NULL return (init failure) is checked. The context keeps
        // a share of the model so it cannot outlive it.
        unsafe {
            let mut params = forge_sys::llama_context_default_params();
            params.n_ctx = options.n_ctx;
            params.n_batch = options.n_batch;
            params.n_ubatch = options.n_ubatch;
            params.n_seq_max = options.n_seq_max;
            params.n_outputs_max = options.n_outputs_max;
            params.n_outputs_max_per_seq = options.n_outputs_max_per_seq;
            params.n_threads = n_threads;
            params.n_threads_batch = n_threads_batch;
            params.pooling_type = options.pooling.as_native();
            params.attention_type = options.attention.as_native();
            params.flash_attn_type = options.flash_attn.as_native();
            params.embeddings = options.embeddings;
            params.offload_kqv = options.offload_kqv;
            params.op_offload = options.op_offload;
            params.kv_unified = options.kv_unified;
            params.type_k = options.type_k.ggml_type();
            params.type_v = options.type_v.ggml_type();
            let raw = forge_sys::llama_init_from_model(model.raw(), params);
            if raw.is_null() {
                return Err(Error::context("context creation failed"));
            }
            // Query the effective values back: upstream pads, clamps,
            // and resolves the request, and decode-time validation
            // must use the actuals.
            let n_ctx = forge_sys::llama_n_ctx(raw);
            let n_ctx_seq = forge_sys::llama_n_ctx_seq(raw);
            let n_batch = forge_sys::llama_n_batch(raw);
            let n_ubatch = forge_sys::llama_n_ubatch(raw);
            let n_seq_max = forge_sys::llama_n_seq_max(raw);
            let pooling = PoolingType::from_native(forge_sys::llama_pooling_type(raw));
            let max_parallel_seqs = u32::try_from(forge_sys::llama_max_parallel_sequences())
                .map_err(|_| Error::context("native parallel-sequence limit too large"))?;
            // Multi-position (MROPE/IMROPE) models carry 4 positions per
            // embedding and upstream aborts position shifts there, so
            // `Memory` refuses them up front. The rope family is a pure
            // function of the model weights; anything unrecognized is
            // treated as multi-position only when it matches exactly.
            let rope = forge_sys::llama_model_rope_type(model.raw());
            let shift_allowed =
                rope != forge_sys::rope_type::MROPE && rope != forge_sys::rope_type::IMROPE;
            let n_vocab = model.vocab_size()?;
            let n_embd_out = model.n_embd_out()?;
            let n_cls_out = model.n_cls_out();
            // Upstream exposes no getter for the resolved causal flag,
            // so resolve it from what is known: explicit options are
            // exact; `Unspecified` follows the model default, which is
            // causal for decoder models and non-causal for encoders.
            // (A decoder GGUF explicitly marked non-causal would be
            // misclassified under `Unspecified` — the same exposure as
            // the upstream CLI, which validates nothing.)
            let causal_attn = match options.attention {
                AttentionType::Causal => true,
                AttentionType::NonCausal => false,
                AttentionType::Unspecified => !model.has_encoder(),
            };
            Ok(Self {
                raw,
                model: Rc::clone(model.inner()),
                n_ctx,
                n_ctx_seq,
                n_batch,
                n_ubatch,
                n_seq_max,
                n_vocab,
                n_threads: options.n_threads,
                n_threads_batch: threads_batch,
                causal_attn,
                pooling,
                kv_unified: options.kv_unified,
                n_embd_out,
                n_cls_out,
                max_parallel_seqs,
                shift_allowed,
            })
        }
    }

    /// Effective context length (upstream pads the request up to a
    /// multiple of 256 and rounds to a multiple of `n_seq_max`, so
    /// this generally exceeds the requested value).
    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    /// Effective per-sequence context length (`n_ctx / n_seq_max`,
    /// padded by upstream).
    pub fn n_ctx_seq(&self) -> u32 {
        self.n_ctx_seq
    }

    /// Effective logical batch limit enforced by [`decode`](Context::decode).
    pub fn n_batch(&self) -> u32 {
        self.n_batch
    }

    /// Effective micro-batch size (also the per-decode token limit
    /// under non-causal attention).
    pub fn n_ubatch(&self) -> u32 {
        self.n_ubatch
    }

    /// Effective maximum distinct sequences per batch.
    pub fn n_seq_max(&self) -> u32 {
        self.n_seq_max
    }

    /// Vocabulary dimension used for [`Logits`].
    pub fn n_vocab(&self) -> u32 {
        self.n_vocab
    }

    /// Worker thread count for generation.
    pub fn n_threads(&self) -> u32 {
        self.n_threads
    }

    /// Worker thread count for batch processing.
    pub fn n_threads_batch(&self) -> u32 {
        self.n_threads_batch
    }

    /// Effective causal-attention state (see [`AttentionType`]).
    pub fn causal_attn(&self) -> bool {
        self.causal_attn
    }

    /// Resolved pooling type (see [`PoolingType`]).
    pub fn pooling(&self) -> PoolingType {
        self.pooling
    }

    /// Change the worker thread counts mid-life. Both must be at least
    /// 1 (upstream consumes them verbatim at compute time, so 0 has no
    /// defined meaning). Takes effect on subsequent decodes.
    pub fn set_n_threads(&mut self, n_threads: u32, n_threads_batch: u32) -> Result<()> {
        if n_threads == 0 || n_threads_batch == 0 {
            return Err(Error::invalid("context thread counts must be at least 1"));
        }
        let gen = c_int::try_from(n_threads)
            .map_err(|_| Error::invalid("context n_threads too large"))?;
        let batch = c_int::try_from(n_threads_batch)
            .map_err(|_| Error::invalid("context n_threads_batch too large"))?;
        // SAFETY: raw is a live context; the setter is infallible and
        // only updates validated fields.
        unsafe { forge_sys::llama_set_n_threads(self.raw, gen, batch) };
        self.n_threads = n_threads;
        self.n_threads_batch = n_threads_batch;
        Ok(())
    }

    /// Switch causal attention mid-life (upstream also exposes this
    /// for mixed flows such as image tokens decoded non-causally).
    /// Takes effect on subsequent decodes; retained KV is not cleared.
    /// Prefer [`AttentionType`] at construction when the choice is
    /// known up front.
    pub fn set_causal_attn(&mut self, causal: bool) {
        // SAFETY: raw is a live context; the setter is infallible.
        unsafe { forge_sys::llama_set_causal_attn(self.raw, causal) };
        self.causal_attn = causal;
    }

    /// Wait for in-flight work to complete. A near-no-op on CPU
    /// execution; required before reading device-side state directly
    /// on GPU execution. Safe to call any time.
    pub fn synchronize(&mut self) {
        // SAFETY: raw is a live context; safe on an idle context.
        unsafe { forge_sys::llama_synchronize(self.raw) };
    }

    /// Decode one [`Batch`], advancing this context's KV state.
    ///
    /// ForgeCore refuses up front what upstream cannot take safely: an
    /// empty batch, a batch larger than the effective `n_batch`, a
    /// batch larger than the effective `n_ubatch` under non-causal
    /// attention (both are upstream aborts, verified by probe), and
    /// sequence ids outside the effective `n_seq_max` (unless
    /// [`ContextOptions::kv_unified`] defers sequence validation to
    /// upstream, whose 256-sequence bound fails safely). Token ranges,
    /// position consecutiveness, and KV capacity stay upstream's
    /// domain: violations come back as error codes and map to
    /// [`Error::decode`] with the code's meaning — nothing is hidden
    /// or retried.
    pub fn decode(&mut self, batch: &Batch) -> Result<()> {
        if batch.n_tokens() == 0 {
            return Err(Error::invalid("decode: batch is empty"));
        }
        if batch.n_tokens() > self.n_batch as usize {
            return Err(Error::invalid(format!(
                "decode: batch of {} tokens exceeds context n_batch {}",
                batch.n_tokens(),
                self.n_batch
            )));
        }
        if !self.causal_attn && batch.n_tokens() > self.n_ubatch as usize {
            return Err(Error::invalid(format!(
                "decode: batch of {} tokens exceeds context n_ubatch {} \
                 (non-causal attention requires n_ubatch >= n_tokens)",
                batch.n_tokens(),
                self.n_ubatch
            )));
        }
        // Under unified KV, upstream validates sequence ids against
        // its internal 256-sequence bound (a safe native error, not an
        // abort), so there is nothing to pre-check; otherwise the
        // effective n_seq_max applies.
        if !self.kv_unified {
            if let Some(bad) = batch.all_seq_ids().find(|seq| *seq >= self.n_seq_max) {
                return Err(Error::invalid(format!(
                    "decode: seq id {bad} out of range (context n_seq_max {})",
                    self.n_seq_max
                )));
            }
        }
        // SAFETY: raw is a live context; the batch view borrows only
        // arrays the `&Batch` owns, which outlive this call; the model
        // is alive via our Rc. Upstream takes the batch by const
        // reference (`llama_context::decode(const llama_batch &)`) and
        // retains no pointers after the call returns.
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

    /// Copy the embedding row for one token of the most recently
    /// decoded batch. Requires an embeddings context
    /// ([`ContextOptions::embeddings`]) and a flagged output position,
    /// like [`logits`](Context::logits); returns an owned
    /// [`Embeddings`] row of `n_embd_out` floats, or
    /// [`Error::embeddings`] when the index has no embedding row.
    pub fn embeddings(&mut self, token_index: u32) -> Result<Embeddings> {
        let index = c_int::try_from(token_index)
            .map_err(|_| Error::invalid("embeddings token index too large"))?;
        // SAFETY: raw is a live context; NULL (no such output) is
        // checked; on success upstream guarantees a row of exactly
        // n_embd_out floats, which is copied before returning.
        unsafe {
            let ptr = forge_sys::llama_get_embeddings_ith(self.raw, index);
            if ptr.is_null() {
                return Err(Error::embeddings(format!(
                    "no embedding row for batch index {token_index}"
                )));
            }
            let values = std::slice::from_raw_parts(ptr, self.n_embd_out as usize).to_vec();
            Ok(Embeddings {
                values,
                width: self.n_embd_out as usize,
            })
        }
    }

    /// Copy the pooled embedding row for one sequence of the most
    /// recently decoded batch. Requires an embeddings context with
    /// pooling enabled ([`PoolingType::None`] has no per-sequence rows)
    /// and a sequence that participated in the batch. Rows hold
    /// `n_embd_out` floats, or `n_cls_out` floats under
    /// [`PoolingType::Rank`].
    pub fn embeddings_seq(&mut self, seq: SeqId) -> Result<Embeddings> {
        let id =
            c_int::try_from(seq).map_err(|_| Error::invalid(format!("seq id {seq} too large")))?;
        let width = if self.pooling == PoolingType::Rank {
            self.n_cls_out
        } else {
            self.n_embd_out
        };
        // SAFETY: raw is a live context; NULL (no pooling, or sequence
        // absent) is checked; on success upstream guarantees a row of
        // exactly `width` floats for this pooling mode, which is copied
        // before returning.
        unsafe {
            let ptr = forge_sys::llama_get_embeddings_seq(self.raw, id);
            if ptr.is_null() {
                return Err(Error::embeddings(format!(
                    "no pooled embedding row for sequence {seq}"
                )));
            }
            let values = std::slice::from_raw_parts(ptr, width as usize).to_vec();
            Ok(Embeddings {
                values,
                width: width as usize,
            })
        }
    }

    /// Borrow this context's native memory (KV cache) object for
    /// sequence operations. Returns [`None`] when the model
    /// architecture has no memory object (BERT-family encoders):
    /// sequence surgery is meaningless there, while state snapshots
    /// still work (they carry just the architecture tag).
    pub fn memory(&mut self) -> Option<Memory<'_>> {
        // SAFETY: raw is a live context; the borrowed handle cannot
        // outlive this `&mut` borrow, and NULL (memory-less model) maps
        // to `None` rather than a dangling handle.
        let raw = unsafe { forge_sys::llama_get_memory(self.raw) };
        if raw.is_null() {
            None
        } else {
            Some(Memory::borrow(
                raw,
                self.n_seq_max,
                self.max_parallel_seqs,
                self.kv_unified,
                self.shift_allowed,
            ))
        }
    }

    /// Exact size in bytes of the current whole-context serialized
    /// state (what [`Context::export_state`] would emit).
    ///
    /// This is a *serialization* size — model-arch tag plus live KV
    /// cells — not allocated, resident, or device memory, which the
    /// pinned C API does not report (the breakdown helper is C++-only).
    /// RAMforge must not mistake it for residency.
    pub fn state_size(&mut self) -> Result<u64> {
        // SAFETY: raw is a live context; the native query catches its
        // own failures internally and reports 0, which is never a
        // valid state size (even an empty cache serializes headers).
        let size = unsafe { forge_sys::llama_state_get_size(self.raw) };
        if size == 0 {
            return Err(Error::state("native state size query failed"));
        }
        Ok(size as u64)
    }

    /// Export the whole-context state into caller-owned bytes. The
    /// size query and the copy happen under one `&mut` borrow, so no
    /// decode or sequence op can change the state between them.
    pub fn export_state(&mut self) -> Result<State> {
        let size = usize::try_from(self.state_size()?)
            .map_err(|_| Error::state("native state size exceeds addressable memory"))?;
        let mut bytes = vec![0u8; size];
        // SAFETY: raw is a live context; the buffer is caller-owned
        // with exactly `size` initialized bytes; native writes at most
        // `size` (overrun throws internally and is reported as 0, never
        // overflows) and returns the count written.
        let written =
            unsafe { forge_sys::llama_state_get_data(self.raw, bytes.as_mut_ptr(), size) };
        if written == 0 || written > size {
            return Err(Error::state("native state export failed"));
        }
        bytes.truncate(written);
        Ok(State::from_exported(bytes))
    }

    /// Replace the whole-context state from a snapshot previously
    /// produced by [`Context::export_state`] (possibly on another
    /// context over the same model). Corrupt, truncated, or
    /// architecturally mismatched bytes are rejected with
    /// [`Error::state`]; trailing garbage after a valid prefix is also
    /// rejected (exact consumption). On failure the cache is left
    /// empty (native clears before reporting the error).
    pub fn import_state(&mut self, state: &State) -> Result<()> {
        let bytes = state.as_bytes();
        // SAFETY: raw is a live context; the slice is borrowed for the
        // call; native reads at most `len` (overrun throws internally,
        // reported as 0) and returns the count consumed.
        let read =
            unsafe { forge_sys::llama_state_set_data(self.raw, bytes.as_ptr(), bytes.len()) };
        if read == 0 || read != bytes.len() {
            return Err(Error::state(
                "native state import rejected the bytes \
                 (corrupt, truncated, mismatched, or trailing garbage)",
            ));
        }
        Ok(())
    }

    /// Exact size in bytes of one sequence's serialized state (what
    /// [`Context::export_seq_state`] would emit for `seq`). Same
    /// serialization-not-residency semantics as [`Context::state_size`].
    pub fn seq_state_size(&mut self, seq: SeqId) -> Result<u64> {
        let id = self.check_state_seq(seq, "seq_state_size")?;
        // SAFETY: as in `state_size`; the id satisfies the general
        // bound (export asserts nothing tighter on any implementation).
        let size = unsafe { forge_sys::llama_state_seq_get_size(self.raw, id) };
        if size == 0 {
            return Err(Error::state(format!(
                "native sequence state size query failed for seq {seq}"
            )));
        }
        Ok(size as u64)
    }

    /// Export one sequence's state into caller-owned bytes. The
    /// snapshot restores onto any `seq < n_seq_max` via
    /// [`Context::import_seq_state`], even on another context over the
    /// same model.
    pub fn export_seq_state(&mut self, seq: SeqId) -> Result<SeqState> {
        let id = self.check_state_seq(seq, "export_seq_state")?;
        let size = usize::try_from(self.seq_state_size(seq)?)
            .map_err(|_| Error::state("native sequence state size exceeds addressable memory"))?;
        let mut bytes = vec![0u8; size];
        // SAFETY: as in `export_state`.
        let written =
            unsafe { forge_sys::llama_state_seq_get_data(self.raw, bytes.as_mut_ptr(), size, id) };
        if written == 0 || written > size {
            return Err(Error::state(format!(
                "native sequence state export failed for seq {seq}"
            )));
        }
        bytes.truncate(written);
        Ok(SeqState::from_exported(bytes))
    }

    /// Restore a snapshot produced by [`Context::export_seq_state`]
    /// onto `seq`, replacing that sequence's previous cells. Requires
    /// `seq < n_seq_max` even on unified caches (the DSV4 and
    /// recurrent implementations assert that bound). Same strict
    /// byte-consumption rule as [`Context::import_state`].
    pub fn import_seq_state(&mut self, seq: SeqId, state: &SeqState) -> Result<()> {
        // Tight bound (not the general unified one): restore asserts
        // `n_seq_max` on DSV4/recurrent implementations.
        if seq >= self.n_seq_max {
            return Err(Error::invalid(format!(
                "import_seq_state: seq id {seq} out of range (limit {})",
                self.n_seq_max
            )));
        }
        let id = c_int::try_from(seq)
            .map_err(|_| Error::invalid(format!("import_seq_state: seq id {seq} too large")))?;
        let bytes = state.as_bytes();
        // SAFETY: as in `import_state`; the id satisfies the tightest
        // bound any implementation asserts.
        let read = unsafe {
            forge_sys::llama_state_seq_set_data(self.raw, bytes.as_ptr(), bytes.len(), id)
        };
        if read == 0 || read != bytes.len() {
            return Err(Error::state(format!(
                "native sequence state import rejected the bytes for seq {seq}"
            )));
        }
        Ok(())
    }

    /// Validate a sequence id for state *export*-side calls (general
    /// bound: `n_seq_max`, or the native parallel-sequence limit when
    /// unified). Restore-side calls use the tighter `n_seq_max`
    /// inline; see [`Context::import_seq_state`].
    fn check_state_seq(&self, seq: SeqId, op: &str) -> Result<c_int> {
        let bound = if self.kv_unified {
            self.max_parallel_seqs
        } else {
            self.n_seq_max
        };
        if seq >= bound {
            return Err(Error::invalid(format!(
                "{op}: seq id {seq} out of range (limit {bound})"
            )));
        }
        c_int::try_from(seq).map_err(|_| Error::invalid(format!("{op}: seq id {seq} too large")))
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("n_ctx", &self.n_ctx)
            .field("n_vocab", &self.n_vocab)
            .field("n_batch", &self.n_batch)
            .field("n_seq_max", &self.n_seq_max)
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

    #[test]
    fn context_options_default_matches_native() {
        let options = ContextOptions::default();
        assert_eq!(options.n_ctx, 0);
        assert_eq!(options.n_threads, 1);
        assert_eq!(options.n_batch, 2048);
        assert_eq!(options.n_ubatch, 512);
        assert_eq!(options.n_seq_max, 1);
        assert_eq!(options.n_threads_batch, 0);
        assert_eq!(options.n_outputs_max, 0);
        assert_eq!(options.n_outputs_max_per_seq, 1);
        assert!(options.offload_kqv);
        assert!(options.op_offload);
        assert!(!options.kv_unified);
        assert!(!options.embeddings);
        assert_eq!(options.pooling, PoolingType::Unspecified);
        assert_eq!(options.attention, AttentionType::Unspecified);
        assert_eq!(options.flash_attn, FlashAttnType::Auto);
        assert_eq!(options.type_k, DType::F16);
        assert_eq!(options.type_v, DType::F16);
    }

    #[test]
    fn pooling_type_maps_to_native() {
        assert_eq!(PoolingType::Unspecified.as_native(), -1);
        assert_eq!(PoolingType::None.as_native(), 0);
        assert_eq!(PoolingType::Mean.as_native(), 1);
        assert_eq!(PoolingType::Cls.as_native(), 2);
        assert_eq!(PoolingType::Last.as_native(), 3);
        assert_eq!(PoolingType::Rank.as_native(), 4);
        assert_eq!(PoolingType::from_native(1), PoolingType::Mean);
        assert_eq!(PoolingType::from_native(4), PoolingType::Rank);
        assert_eq!(PoolingType::from_native(0), PoolingType::None);
        assert_eq!(PoolingType::from_native(99), PoolingType::None);
    }

    #[test]
    fn attention_type_maps_to_native() {
        assert_eq!(AttentionType::Unspecified.as_native(), -1);
        assert_eq!(AttentionType::Causal.as_native(), 0);
        assert_eq!(AttentionType::NonCausal.as_native(), 1);
    }

    #[test]
    fn flash_attn_type_maps_to_native() {
        assert_eq!(FlashAttnType::Auto.as_native(), -1);
        assert_eq!(FlashAttnType::Disabled.as_native(), 0);
        assert_eq!(FlashAttnType::Enabled.as_native(), 1);
    }
}
