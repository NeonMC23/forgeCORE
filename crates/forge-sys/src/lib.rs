//! Raw FFI bindings to the pinned upstream llama.cpp/ggml C API.
//!
//! This crate is the **only** place that talks to C. It exposes opaque
//! handle types, `#[repr(C)]` parameter structs, constants mirroring the
//! upstream enums, and `unsafe` `extern "C"` declarations — nothing else.
//! All safety reasoning lives in `forge-core`, which wraps these bindings.
//!
//! Bindings target upstream llama.cpp `v0.5.0`
//! (commit `7fe450e19305b828c199d602c23a8337aaa1f03b`); only the minimum
//! surface `forge-core` needs is bound (see `docs/PIVOT.md`).

#![allow(non_camel_case_types)]

use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_longlong};

// ---------------------------------------------------------------------------
// Opaque handle types (never dereferenced; passed by pointer only)
// ---------------------------------------------------------------------------

macro_rules! opaque {
    ($($name:ident),* $(,)?) => {$(
        #[repr(C)]
        pub struct $name {
            _private: [u8; 0],
        }
    )*};
}

opaque! {
    ggml_context,
    ggml_tensor,
    ggml_cgraph,
    ggml_backend,
    ggml_backend_dev,
    ggml_backend_buffer,
    llama_model,
    llama_vocab,
    llama_context,
    llama_memory_i,
    llama_sampler,
}

/// `ggml_backend_t` — pointer to an opaque backend instance.
pub type ggml_backend_t = *mut ggml_backend;
/// `ggml_backend_dev_t` — pointer to an opaque backend device.
pub type ggml_backend_dev_t = *mut ggml_backend_dev;
/// `llama_memory_t` — the context-owned memory object (`llama.h`).
/// Borrowed from the context via `llama_get_memory`; NULL when the
/// model architecture has no memory (e.g. BERT encoders). Must never
/// outlive its context; `forge-core` models this as a borrow.
pub type llama_memory_t = *mut llama_memory_i;

// ---------------------------------------------------------------------------
// Constants mirroring upstream enums (plain C enums, i32-compatible)
// ---------------------------------------------------------------------------

/// `enum ggml_type` values bound by `forge-core` (`ggml.h`).
pub mod ggml_type {
    pub const F32: i32 = 0;
    pub const F16: i32 = 1;
    pub const Q4_0: i32 = 2;
    pub const Q4_1: i32 = 3;
    pub const Q5_0: i32 = 6;
    pub const Q5_1: i32 = 7;
    pub const Q8_0: i32 = 8;
    pub const Q8_1: i32 = 9;
    pub const Q2_K: i32 = 10;
    pub const Q3_K: i32 = 11;
    pub const Q4_K: i32 = 12;
    pub const Q5_K: i32 = 13;
    pub const Q6_K: i32 = 14;
    pub const Q8_K: i32 = 15;
    pub const I8: i32 = 24;
    pub const I16: i32 = 25;
    pub const I32: i32 = 26;
    pub const F64: i32 = 28;
    pub const BF16: i32 = 30;
}

/// `enum ggml_backend_dev_type` values (`ggml-backend.h`).
pub mod dev_type {
    pub const CPU: i32 = 0;
    pub const GPU: i32 = 1;
    pub const IGPU: i32 = 2;
    pub const ACCEL: i32 = 3;
    pub const META: i32 = 4;
}

/// `enum ggml_status` values (`ggml.h`).
pub mod status {
    pub const ALLOC_FAILED: i32 = -2;
    pub const FAILED: i32 = -1;
    pub const SUCCESS: i32 = 0;
    pub const ABORTED: i32 = 1;
}

/// `enum llama_vocab_type` values (`llama.h`).
pub mod vocab_type {
    pub const NONE: i32 = 0;
    pub const SPM: i32 = 1;
    pub const BPE: i32 = 2;
    pub const WPM: i32 = 3;
    pub const UGM: i32 = 4;
    pub const RWKV: i32 = 5;
    pub const PLAMO2: i32 = 6;
    pub const TEST: i32 = 7;
}

/// `enum llama_token_attr` bitmask values (`llama.h`).
pub mod token_attr {
    pub const UNDEFINED: i32 = 0;
    pub const UNKNOWN: i32 = 1;
    pub const UNUSED: i32 = 2;
    pub const NORMAL: i32 = 4;
    pub const CONTROL: i32 = 8;
    pub const USER_DEFINED: i32 = 16;
    pub const BYTE: i32 = 32;
    pub const NORMALIZED: i32 = 64;
    pub const LSTRIP: i32 = 128;
    pub const RSTRIP: i32 = 256;
    pub const SINGLE_WORD: i32 = 512;
}

/// `enum llama_split_mode` values (`llama.h`).
pub mod split_mode {
    pub const NONE: i32 = 0;
    pub const LAYER: i32 = 1;
    pub const ROW: i32 = 2;
    pub const TENSOR: i32 = 3;
}

/// `enum llama_load_mode` values (`llama.h`).
///
/// `LLAMA_LOAD_MODE_DIRECT_IO` (4) is deliberately unbound: no
/// `forge-core` API selects it.
pub mod load_mode {
    pub const AUTO: i32 = -1;
    pub const NONE: i32 = 0;
    pub const MMAP: i32 = 1;
    pub const MLOCK: i32 = 2;
    pub const MMAP_MLOCK: i32 = 3;
}

/// `enum llama_pooling_type` values (`llama.h`).
pub mod pooling_type {
    pub const UNSPECIFIED: i32 = -1;
    pub const NONE: i32 = 0;
    pub const MEAN: i32 = 1;
    pub const CLS: i32 = 2;
    pub const LAST: i32 = 3;
    pub const RANK: i32 = 4;
}

/// `enum llama_attention_type` values (`llama.h`).
pub mod attention_type {
    pub const UNSPECIFIED: i32 = -1;
    pub const CAUSAL: i32 = 0;
    pub const NON_CAUSAL: i32 = 1;
}

/// `enum llama_flash_attn_type` values (`llama.h`).
pub mod flash_attn_type {
    pub const AUTO: i32 = -1;
    pub const DISABLED: i32 = 0;
    pub const ENABLED: i32 = 1;
}

/// `enum llama_rope_type` values (`llama.h`; non-canonical ids alias
/// the `GGML_ROPE_TYPE_*` defines in `ggml.h`). `forge-core` uses
/// these only to refuse position shifts on multi-position models
/// (`MROPE`/`IMROPE` carry 4 positions per embedding and upstream
/// aborts shifts there).
pub mod rope_type {
    pub const NONE: i32 = -1;
    pub const NORM: i32 = 0;
    pub const NEOX: i32 = 2;
    pub const MROPE: i32 = 8;
    pub const VISION: i32 = 24;
    pub const IMROPE: i32 = 40;
}

/// `LLAMA_TOKEN_NULL` (`llama.h`): sentinel for "no such special token".
pub const LLAMA_TOKEN_NULL: c_int = -1;

/// `LLAMA_DEFAULT_SEED` (`llama.h`): requests a random RNG seed from
/// samplers that take a seed.
pub const LLAMA_DEFAULT_SEED: u32 = 0xFFFF_FFFF;

// ---------------------------------------------------------------------------
// #[repr(C)] parameter structs (field order matches the upstream headers)
// ---------------------------------------------------------------------------

/// `struct ggml_init_params` (`ggml.h`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ggml_init_params {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

/// `struct llama_model_params` (`llama.h`). Construct via
/// [`llama_model_default_params`], never by hand.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_model_params {
    // `ggml_backend_dev_t *`: NULL (default device selection) or a
    // NULL-terminated array of registry-borrowed device handles.
    pub devices: *mut ggml_backend_dev_t,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub load_mode: i32,
    pub lazy_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: Option<unsafe extern "C" fn(f32, *mut c_void) -> bool>,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

/// `ggml_log_callback` (`ggml.h`); also used by `llama_log_set`.
pub type ggml_log_callback =
    Option<unsafe extern "C" fn(level: c_int, text: *const c_char, user_data: *mut c_void)>;

/// `ggml_backend_sched_eval_callback` (`ggml-backend.h`). Only carried
/// inside [`llama_context_params`], never called from Rust.
pub type ggml_sched_eval_callback =
    Option<unsafe extern "C" fn(t: *mut ggml_tensor, ask: bool, user_data: *mut c_void) -> bool>;

/// `ggml_abort_callback` (`ggml.h`). Only carried inside
/// [`llama_context_params`], never installed from Rust.
pub type ggml_abort_callback = Option<unsafe extern "C" fn(data: *mut c_void) -> bool>;

/// `struct llama_batch` (`llama.h`).
///
/// Always built by `forge-core` from owned `Vec`s whose storage
/// outlives every decode call; the struct itself is a by-value view.
/// `embd` stays NULL (token batches only); `seq_id` entries point at
/// `forge-core`-owned sequence ids.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_batch {
    pub n_tokens: c_int,
    pub token: *mut c_int,
    pub embd: *mut f32,
    pub pos: *mut c_int,
    pub n_seq_id: *mut c_int,
    pub seq_id: *mut *mut c_int,
    pub logits: *mut i8,
}

/// `struct llama_context_params` (`llama.h`). Construct via
/// [`llama_context_default_params`], never by hand. Pointer-typed
/// fields the phase-1 API does not use keep their upstream defaults.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: c_int,
    pub n_threads_batch: c_int,
    pub ctx_type: c_int,
    pub rope_scaling_type: c_int,
    pub pooling_type: c_int,
    pub attention_type: c_int,
    pub flash_attn_type: c_int,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: ggml_sched_eval_callback,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: c_int,
    pub type_v: c_int,
    pub abort_callback: ggml_abort_callback,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut c_void,
    pub n_samplers: usize,
    pub ctx_other: *mut llama_context,
}

/// `struct llama_token_data` (`llama.h`): one sampling candidate.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_token_data {
    pub id: c_int,
    pub logit: f32,
    pub p: f32,
}

/// `struct llama_token_data_array` (`llama.h`): candidate set a sampler
/// chain mutates in place (`size` shrinks as filters apply; `selected`
/// is the chosen *index*, not a token id).
///
/// Always built by `forge-core` from an owned `Vec<llama_token_data>`
/// that outlives the `apply` call; `selected` starts at -1.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_token_data_array {
    pub data: *mut llama_token_data,
    pub size: usize,
    pub selected: i64,
    pub sorted: bool,
}

/// `struct llama_sampler_chain_params` (`llama.h`). Construct via
/// [`llama_sampler_chain_default_params`], never by hand.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_sampler_chain_params {
    pub no_perf: bool,
}

// ---------------------------------------------------------------------------
// extern "C" declarations
// ---------------------------------------------------------------------------

unsafe extern "C" {
    // -- ggml.h: context / tensor / graph -----------------------------------
    pub fn ggml_init(params: ggml_init_params) -> *mut ggml_context;
    pub fn ggml_free(ctx: *mut ggml_context);
    pub fn ggml_new_tensor(
        ctx: *mut ggml_context,
        type_: c_int,
        n_dims: c_int,
        ne: *const c_longlong,
    ) -> *mut ggml_tensor;
    pub fn ggml_add(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_mul_mat(
        ctx: *mut ggml_context,
        a: *mut ggml_tensor,
        b: *mut ggml_tensor,
    ) -> *mut ggml_tensor;
    pub fn ggml_new_graph(ctx: *mut ggml_context) -> *mut ggml_cgraph;
    pub fn ggml_build_forward_expand(cgraph: *mut ggml_cgraph, tensor: *mut ggml_tensor);
    pub fn ggml_tensor_overhead() -> usize;
    pub fn ggml_graph_overhead() -> usize;
    pub fn ggml_nelements(tensor: *const ggml_tensor) -> i64;
    pub fn ggml_nbytes(tensor: *const ggml_tensor) -> usize;
    pub fn ggml_status_to_string(status: c_int) -> *const c_char;
    pub fn ggml_log_set(log_callback: ggml_log_callback, user_data: *mut c_void);

    // -- ggml-backend.h: devices / buffers / execution ----------------------
    pub fn ggml_backend_load_all();
    pub fn ggml_backend_dev_count() -> usize;
    pub fn ggml_backend_dev_get(index: usize) -> ggml_backend_dev_t;
    pub fn ggml_backend_dev_name(device: ggml_backend_dev_t) -> *const c_char;
    pub fn ggml_backend_dev_description(device: ggml_backend_dev_t) -> *const c_char;
    pub fn ggml_backend_dev_type(device: ggml_backend_dev_t) -> c_int;
    pub fn ggml_backend_dev_memory(device: ggml_backend_dev_t, free: *mut usize, total: *mut usize);
    pub fn ggml_backend_dev_by_type(type_: c_int) -> ggml_backend_dev_t;
    pub fn ggml_backend_dev_init(
        device: ggml_backend_dev_t,
        params: *const c_char,
    ) -> ggml_backend_t;
    pub fn ggml_backend_init_by_type(type_: c_int, params: *const c_char) -> ggml_backend_t;
    pub fn ggml_backend_free(backend: ggml_backend_t);
    pub fn ggml_backend_buffer_free(buffer: *mut ggml_backend_buffer);
    pub fn ggml_backend_tensor_set(
        tensor: *mut ggml_tensor,
        data: *const c_void,
        offset: usize,
        size: usize,
    );
    pub fn ggml_backend_tensor_get(
        tensor: *const ggml_tensor,
        data: *mut c_void,
        offset: usize,
        size: usize,
    );
    pub fn ggml_backend_graph_compute(backend: ggml_backend_t, cgraph: *mut ggml_cgraph) -> c_int;
    pub fn ggml_backend_synchronize(backend: ggml_backend_t);

    // -- ggml-alloc.h: static graph/tensor allocation ------------------------
    pub fn ggml_backend_alloc_ctx_tensors(
        ctx: *mut ggml_context,
        backend: ggml_backend_t,
    ) -> *mut ggml_backend_buffer;

    // -- ggml-cpu.h: CPU backend controls ------------------------------------
    pub fn ggml_backend_cpu_init() -> ggml_backend_t;
    pub fn ggml_backend_cpu_set_n_threads(backend_cpu: ggml_backend_t, n_threads: c_int);

    // -- llama.h: model loading / metadata ----------------------------------
    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);
    pub fn llama_model_n_params(model: *const llama_model) -> u64;
    // Device/offload capability facts (`llama.h`). All four take no
    // arguments and cannot fail: `max_devices` returns a constant
    // (16); `supports_mmap/mlock` return compile-time platform flags;
    // `supports_gpu_offload` loads the backend registry on first use
    // and reports whether a GPU/IGPU device (or RPC support) exists.
    pub fn llama_max_devices() -> usize;
    pub fn llama_supports_mmap() -> bool;
    pub fn llama_supports_mlock() -> bool;
    pub fn llama_supports_gpu_offload() -> bool;
    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    pub fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;
    pub fn llama_log_set(log_callback: ggml_log_callback, user_data: *mut c_void);

    // -- llama.h: context / batch / decode / logits -------------------------
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);
    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> c_int;
    pub fn llama_get_logits_ith(ctx: *mut llama_context, i: c_int) -> *mut f32;
    // Actual context sizing (`llama.h`): upstream resolves/pads the
    // requested values at init (e.g. `n_ctx` pads up to a multiple of
    // 256), so `forge-core` queries these after `open` instead of
    // trusting the request. Pure getters off a live context.
    pub fn llama_n_ctx(ctx: *const llama_context) -> u32;
    pub fn llama_n_ctx_seq(ctx: *const llama_context) -> u32;
    pub fn llama_n_batch(ctx: *const llama_context) -> u32;
    pub fn llama_n_ubatch(ctx: *const llama_context) -> u32;
    pub fn llama_n_seq_max(ctx: *const llama_context) -> u32;
    // Resolved pooling type (`llama.h`): `UNSPECIFIED` becomes the
    // model default (or `NONE`) at init. Pure getter off a live
    // context; `forge-core` maps it back to `PoolingType`.
    pub fn llama_pooling_type(ctx: *const llama_context) -> c_int;
    // Post-open controls (`llama.h`): plain field updates, infallible.
    // Thread counts are consumed verbatim at compute time (0/negative
    // have no defined meaning), so `forge-core` validates `>= 1`.
    pub fn llama_set_n_threads(ctx: *mut llama_context, n_threads: c_int, n_threads_batch: c_int);
    pub fn llama_set_causal_attn(ctx: *mut llama_context, causal_attn: bool);
    // Wait for in-flight work and fold evaluation stats (`llama.h`).
    // Safe to call any time, including before any decode.
    pub fn llama_synchronize(ctx: *mut llama_context);
    // Embedding output rows (`llama.h`): same failure contract as
    // `llama_get_logits_ith` (NULL for invalid ids in release builds).
    // `_ith` rows hold `n_embd_out` floats; `_seq` rows hold
    // `n_embd_out` floats, or `n_cls_out` under RANK pooling, and are
    // NULL when pooling is NONE or the sequence has no pooled output.
    pub fn llama_get_embeddings_ith(ctx: *mut llama_context, i: c_int) -> *mut f32;
    pub fn llama_get_embeddings_seq(ctx: *mut llama_context, seq_id: c_int) -> *mut f32;
    // Context memory object (`llama.h`): borrowed handle, NULL when
    // the model has no memory (BERT-family encoders). Pure getter.
    pub fn llama_get_memory(ctx: *const llama_context) -> llama_memory_t;
    // Maximum parallel sequence id space (`llama-cparams.h`): returns
    // `LLAMA_MAX_SEQ` (256), the seq-id bound for unified-KV caches.
    pub fn llama_max_parallel_sequences() -> usize;
    // Memory sequence ops (`llama.h`): the C wrappers null-check the
    // handle but hold NO try/catch, and the implementations assert on
    // out-of-range sequence ids (unconditional abort — verified by
    // probe), on partial cross-stream copies, and on shifts for
    // multi-position (MROPE/IMROPE) models; `seq_div` with `d == 0`
    // divides by zero (verified SIGFPE). `forge-core` validates every
    // precondition. `seq_rm` alone reports failure (`false`) instead
    // of aborting. `seq_pos_min/max` return -1 for empty sequences.
    pub fn llama_memory_clear(mem: llama_memory_t, data: bool);
    pub fn llama_memory_seq_rm(mem: llama_memory_t, seq_id: c_int, p0: c_int, p1: c_int) -> bool;
    pub fn llama_memory_seq_cp(
        mem: llama_memory_t,
        seq_id_src: c_int,
        seq_id_dst: c_int,
        p0: c_int,
        p1: c_int,
    );
    pub fn llama_memory_seq_keep(mem: llama_memory_t, seq_id: c_int);
    pub fn llama_memory_seq_add(
        mem: llama_memory_t,
        seq_id: c_int,
        p0: c_int,
        p1: c_int,
        delta: c_int,
    );
    pub fn llama_memory_seq_div(mem: llama_memory_t, seq_id: c_int, p0: c_int, p1: c_int, d: c_int);
    pub fn llama_memory_seq_pos_min(mem: llama_memory_t, seq_id: c_int) -> c_int;
    pub fn llama_memory_seq_pos_max(mem: llama_memory_t, seq_id: c_int) -> c_int;
    pub fn llama_memory_can_shift(mem: llama_memory_t) -> bool;
    // Whole/sequence state serialization (`llama.h`): byte-oriented
    // only (`forge-core` binds no file or `_ext` variants). The
    // implementations catch `std::exception` internally and return 0
    // on any failure (small destination, truncated/corrupt input,
    // architecture or dtype mismatch), so 0 always means failure and
    // the reported byte count always means success. `get_data` writes
    // at most `size` bytes (overrun throws internally, never
    // overflows); `set_data` reads at most `size` bytes. Both
    // synchronize first. State holds the model-arch tag plus live KV
    // cells — no logits, no embeddings (verified by probe).
    pub fn llama_state_get_size(ctx: *mut llama_context) -> usize;
    pub fn llama_state_get_data(ctx: *mut llama_context, dst: *mut u8, size: usize) -> usize;
    pub fn llama_state_set_data(ctx: *mut llama_context, src: *const u8, size: usize) -> usize;
    pub fn llama_state_seq_get_size(ctx: *mut llama_context, seq_id: c_int) -> usize;
    pub fn llama_state_seq_get_data(
        ctx: *mut llama_context,
        dst: *mut u8,
        size: usize,
        seq_id: c_int,
    ) -> usize;
    pub fn llama_state_seq_set_data(
        ctx: *mut llama_context,
        src: *const u8,
        size: usize,
        seq_id: c_int,
    ) -> usize;

    // -- llama.h: model metadata --------------------------------------------
    pub fn llama_model_desc(model: *const llama_model, buf: *mut c_char, buf_size: usize) -> c_int;
    pub fn llama_model_size(model: *const llama_model) -> u64;
    pub fn llama_model_n_ctx_train(model: *const llama_model) -> c_int;
    pub fn llama_model_n_embd(model: *const llama_model) -> c_int;
    // Embedding widths (`llama.h`): input width sizes `llama_batch.embd`
    // rows, output width sizes `llama_get_embeddings_*` rows. Pure
    // getters off a live model; non-negative in practice, checked by
    // `forge-core` like the other dimension getters.
    pub fn llama_model_n_embd_inp(model: *const llama_model) -> c_int;
    pub fn llama_model_n_embd_out(model: *const llama_model) -> c_int;
    // Classifier head width for RANK-pooled sequence embeddings.
    // `uint32_t`, infallible on a live model.
    pub fn llama_model_n_cls_out(model: *const llama_model) -> u32;
    // Whether the model has an encoder (`llama.h`). Pure predicate off
    // a live model; `forge-core` uses it to resolve the effective
    // causal-attention state (which has no direct getter).
    pub fn llama_model_has_encoder(model: *const llama_model) -> bool;
    // RoPE family (`llama.h`): pure function of the model weights.
    // `forge-core` maps it through `rope_type` to refuse position
    // shifts on MROPE/IMROPE models (upstream aborts there).
    pub fn llama_model_rope_type(model: *const llama_model) -> c_int;
    // GGUF metadata string lookup (`llama.h`): copies the value for
    // `key` into `buf` (`snprintf` semantics, returns the would-be
    // length, or -1 when the key is absent). `forge-core` reads
    // `general.architecture` once at context creation to detect
    // DeepSeek-V4, whose memory keeps per-sequence streams even when
    // unified (so `seq < n_seq_max` binds there too).
    pub fn llama_model_meta_val_str(
        model: *const llama_model,
        key: *const c_char,
        buf: *mut c_char,
        buf_size: usize,
    ) -> c_int;
    pub fn llama_model_n_layer(model: *const llama_model) -> c_int;
    pub fn llama_model_n_head(model: *const llama_model) -> c_int;
    pub fn llama_model_n_head_kv(model: *const llama_model) -> c_int;

    // -- llama.h: tokenizer / vocabulary ------------------------------------
    //
    // Every per-token getter below indexes the native id table without a
    // bounds check (`vector::at` throws across the C boundary, which
    // terminates; `is_control` uses unchecked `operator[]`, which is UB
    // out of bounds), and every getter asserts the vocab type is not
    // NONE. `forge-core` therefore validates each token id against
    // `llama_vocab_n_tokens` before every call and refuses NONE vocabs
    // at `Tokenizer` construction. The deprecated `llama_token_*` /
    // `llama_add_bos/eos_token` aliases are deliberately NOT bound.
    pub fn llama_vocab_type(vocab: *const llama_vocab) -> c_int;
    pub fn llama_vocab_get_text(vocab: *const llama_vocab, token: c_int) -> *const c_char;
    pub fn llama_vocab_get_score(vocab: *const llama_vocab, token: c_int) -> f32;
    pub fn llama_vocab_get_attr(vocab: *const llama_vocab, token: c_int) -> c_int;
    pub fn llama_vocab_is_eog(vocab: *const llama_vocab, token: c_int) -> bool;
    pub fn llama_vocab_is_control(vocab: *const llama_vocab, token: c_int) -> bool;
    pub fn llama_vocab_bos(vocab: *const llama_vocab) -> c_int;
    pub fn llama_vocab_eos(vocab: *const llama_vocab) -> c_int;
    pub fn llama_vocab_eot(vocab: *const llama_vocab) -> c_int;
    pub fn llama_vocab_sep(vocab: *const llama_vocab) -> c_int;
    pub fn llama_vocab_nl(vocab: *const llama_vocab) -> c_int;
    pub fn llama_vocab_pad(vocab: *const llama_vocab) -> c_int;
    pub fn llama_vocab_mask(vocab: *const llama_vocab) -> c_int;
    pub fn llama_vocab_get_add_bos(vocab: *const llama_vocab) -> bool;
    pub fn llama_vocab_get_add_eos(vocab: *const llama_vocab) -> bool;
    // Negative return: -(required capacity); INT32_MIN on overflow.
    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: c_int,
        tokens: *mut c_int,
        n_tokens_max: c_int,
        add_special: bool,
        parse_special: bool,
    ) -> c_int;
    pub fn llama_detokenize(
        vocab: *const llama_vocab,
        tokens: *const c_int,
        n_tokens: c_int,
        text: *mut c_char,
        text_len_max: c_int,
        remove_special: bool,
        unparse_special: bool,
    ) -> c_int;

    // -- llama.h: sampler chains ------------------------------------------
    //
    // `forge-core` drives chains exclusively through `apply` over a
    // caller-built `llama_token_data_array` (mirroring the canonical
    // flow inside `llama_sampler_sample`); the context-bound
    // `llama_sampler_sample`, backend samplers, and every sampler
    // outside greedy/dist/top-k/top-p/min-p/temp are deliberately NOT
    // bound. Constructors only fail via C++ OOM throw (process abort,
    // like a Rust allocation failure), never via NULL.
    pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;
    pub fn llama_sampler_chain_init(params: llama_sampler_chain_params) -> *mut llama_sampler;
    pub fn llama_sampler_chain_add(chain: *mut llama_sampler, smpl: *mut llama_sampler);
    pub fn llama_sampler_chain_n(chain: *const llama_sampler) -> c_int;
    pub fn llama_sampler_chain_remove(chain: *mut llama_sampler, i: c_int) -> *mut llama_sampler;
    pub fn llama_sampler_init_greedy() -> *mut llama_sampler;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_k(k: c_int) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_min_p(p: f32, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_temp(t: f32) -> *mut llama_sampler;
    pub fn llama_sampler_apply(smpl: *mut llama_sampler, cur_p: *mut llama_token_data_array);
    pub fn llama_sampler_accept(smpl: *mut llama_sampler, token: c_int);
    pub fn llama_sampler_reset(smpl: *mut llama_sampler);
    pub fn llama_sampler_get_seed(smpl: *const llama_sampler) -> u32;
    pub fn llama_sampler_free(smpl: *mut llama_sampler);
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // Layouts audited against upstream v0.5.0 headers with gcc on x86_64
    // Linux (see docs/PIVOT.md). These tests guard against accidental
    // edits to the #[repr(C)] structs, which are passed to C by value.
    #[test]
    fn ggml_init_params_layout() {
        assert_eq!(size_of::<ggml_init_params>(), 24);
        assert_eq!(offset_of!(ggml_init_params, mem_size), 0);
        assert_eq!(offset_of!(ggml_init_params, mem_buffer), 8);
        assert_eq!(offset_of!(ggml_init_params, no_alloc), 16);
    }

    #[test]
    fn llama_model_params_layout() {
        assert_eq!(size_of::<llama_model_params>(), 80);
        assert_eq!(offset_of!(llama_model_params, devices), 0);
        assert_eq!(offset_of!(llama_model_params, tensor_buft_overrides), 8);
        assert_eq!(offset_of!(llama_model_params, n_gpu_layers), 16);
        assert_eq!(offset_of!(llama_model_params, split_mode), 20);
        assert_eq!(offset_of!(llama_model_params, load_mode), 24);
        assert_eq!(offset_of!(llama_model_params, lazy_mode), 28);
        assert_eq!(offset_of!(llama_model_params, main_gpu), 32);
        assert_eq!(offset_of!(llama_model_params, tensor_split), 40);
        assert_eq!(offset_of!(llama_model_params, progress_callback), 48);
        assert_eq!(
            offset_of!(llama_model_params, progress_callback_user_data),
            56
        );
        assert_eq!(offset_of!(llama_model_params, kv_overrides), 64);
        assert_eq!(offset_of!(llama_model_params, vocab_only), 72);
        assert_eq!(offset_of!(llama_model_params, check_tensors), 73);
        assert_eq!(offset_of!(llama_model_params, use_extra_bufts), 74);
        assert_eq!(offset_of!(llama_model_params, no_host), 75);
        assert_eq!(offset_of!(llama_model_params, no_alloc), 76);
        assert_eq!(offset_of!(llama_model_params, load_mtp), 77);
    }

    #[test]
    fn llama_batch_layout() {
        assert_eq!(size_of::<llama_batch>(), 56);
        assert_eq!(offset_of!(llama_batch, n_tokens), 0);
        assert_eq!(offset_of!(llama_batch, token), 8);
        assert_eq!(offset_of!(llama_batch, embd), 16);
        assert_eq!(offset_of!(llama_batch, pos), 24);
        assert_eq!(offset_of!(llama_batch, n_seq_id), 32);
        assert_eq!(offset_of!(llama_batch, seq_id), 40);
        assert_eq!(offset_of!(llama_batch, logits), 48);
    }

    #[test]
    fn llama_context_params_layout() {
        assert_eq!(size_of::<llama_context_params>(), 160);
        assert_eq!(offset_of!(llama_context_params, n_ctx), 0);
        assert_eq!(offset_of!(llama_context_params, n_batch), 4);
        assert_eq!(offset_of!(llama_context_params, n_ubatch), 8);
        assert_eq!(offset_of!(llama_context_params, n_seq_max), 12);
        assert_eq!(offset_of!(llama_context_params, n_rs_seq), 16);
        assert_eq!(offset_of!(llama_context_params, n_outputs_max), 20);
        assert_eq!(offset_of!(llama_context_params, n_outputs_max_per_seq), 24);
        assert_eq!(offset_of!(llama_context_params, n_threads), 28);
        assert_eq!(offset_of!(llama_context_params, n_threads_batch), 32);
        assert_eq!(offset_of!(llama_context_params, ctx_type), 36);
        assert_eq!(offset_of!(llama_context_params, rope_scaling_type), 40);
        assert_eq!(offset_of!(llama_context_params, pooling_type), 44);
        assert_eq!(offset_of!(llama_context_params, attention_type), 48);
        assert_eq!(offset_of!(llama_context_params, flash_attn_type), 52);
        assert_eq!(offset_of!(llama_context_params, rope_freq_base), 56);
        assert_eq!(offset_of!(llama_context_params, rope_freq_scale), 60);
        assert_eq!(offset_of!(llama_context_params, yarn_ext_factor), 64);
        assert_eq!(offset_of!(llama_context_params, yarn_attn_factor), 68);
        assert_eq!(offset_of!(llama_context_params, yarn_beta_fast), 72);
        assert_eq!(offset_of!(llama_context_params, yarn_beta_slow), 76);
        assert_eq!(offset_of!(llama_context_params, yarn_orig_ctx), 80);
        assert_eq!(offset_of!(llama_context_params, defrag_thold), 84);
        assert_eq!(offset_of!(llama_context_params, cb_eval), 88);
        assert_eq!(offset_of!(llama_context_params, cb_eval_user_data), 96);
        assert_eq!(offset_of!(llama_context_params, type_k), 104);
        assert_eq!(offset_of!(llama_context_params, type_v), 108);
        assert_eq!(offset_of!(llama_context_params, abort_callback), 112);
        assert_eq!(offset_of!(llama_context_params, abort_callback_data), 120);
        assert_eq!(offset_of!(llama_context_params, embeddings), 128);
        assert_eq!(offset_of!(llama_context_params, offload_kqv), 129);
        assert_eq!(offset_of!(llama_context_params, no_perf), 130);
        assert_eq!(offset_of!(llama_context_params, op_offload), 131);
        assert_eq!(offset_of!(llama_context_params, swa_full), 132);
        assert_eq!(offset_of!(llama_context_params, kv_unified), 133);
        assert_eq!(offset_of!(llama_context_params, samplers), 136);
        assert_eq!(offset_of!(llama_context_params, n_samplers), 144);
        assert_eq!(offset_of!(llama_context_params, ctx_other), 152);
    }

    #[test]
    fn llama_token_data_layout() {
        assert_eq!(size_of::<llama_token_data>(), 12);
        assert_eq!(offset_of!(llama_token_data, id), 0);
        assert_eq!(offset_of!(llama_token_data, logit), 4);
        assert_eq!(offset_of!(llama_token_data, p), 8);
    }

    #[test]
    fn llama_token_data_array_layout() {
        assert_eq!(size_of::<llama_token_data_array>(), 32);
        assert_eq!(offset_of!(llama_token_data_array, data), 0);
        assert_eq!(offset_of!(llama_token_data_array, size), 8);
        assert_eq!(offset_of!(llama_token_data_array, selected), 16);
        assert_eq!(offset_of!(llama_token_data_array, sorted), 24);
    }

    #[test]
    fn llama_sampler_chain_params_layout() {
        assert_eq!(size_of::<llama_sampler_chain_params>(), 1);
        assert_eq!(offset_of!(llama_sampler_chain_params, no_perf), 0);
    }
}
