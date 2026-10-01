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
}

/// `ggml_backend_t` — pointer to an opaque backend instance.
pub type ggml_backend_t = *mut ggml_backend;
/// `ggml_backend_dev_t` — pointer to an opaque backend device.
pub type ggml_backend_dev_t = *mut ggml_backend_dev;

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
    pub devices: ggml_backend_dev_t,
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

    // -- llama.h: model metadata --------------------------------------------
    pub fn llama_model_desc(model: *const llama_model, buf: *mut c_char, buf_size: usize) -> c_int;
    pub fn llama_model_size(model: *const llama_model) -> u64;
    pub fn llama_model_n_ctx_train(model: *const llama_model) -> c_int;
    pub fn llama_model_n_embd(model: *const llama_model) -> c_int;
    pub fn llama_model_n_layer(model: *const llama_model) -> c_int;
    pub fn llama_model_n_head(model: *const llama_model) -> c_int;
    pub fn llama_model_n_head_kv(model: *const llama_model) -> c_int;
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
}
