//! Minimal ggml execution: build a graph, run it on a [`Backend`].
//!
//! This milestone executes exactly two ops — [`add`] and [`matmul`] —
//! over F32 tensors. Each call allocates a fresh result tensor, builds a
//! one-node graph in a scratch context, runs
//! `ggml_backend_graph_compute`, and wraps the result. Both inputs must
//! live on the same backend (checked via shared ownership).

use crate::backend::Backend;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{new_ctx, Tensor};
use std::ffi::CStr;
use std::ptr;
use std::rc::Rc;

/// Read a ggml status code into a Rust error.
fn status_error(context: &str, status: std::os::raw::c_int) -> Error {
    // SAFETY: ggml_status_to_string takes an int and returns a static string.
    let detail = unsafe { CStr::from_ptr(forge_sys::ggml_status_to_string(status)) };
    Error::backend(format!(
        "{context} failed: {} (status {status})",
        detail.to_string_lossy()
    ))
}

/// Allocate the result buffer and run a one-node graph producing `out`.
fn finish(
    backend: &Backend,
    context: &str,
    ctx: *mut forge_sys::ggml_context,
    out: *mut forge_sys::ggml_tensor,
    shape: Vec<usize>,
) -> Result<Tensor> {
    // SAFETY: ctx/out are live; NULL returns unwind the context; the
    // scratch graph context is freed on every path below.
    unsafe {
        let buffer = forge_sys::ggml_backend_alloc_ctx_tensors(ctx, backend.raw());
        if buffer.is_null() {
            forge_sys::ggml_free(ctx);
            return Err(Error::backend(format!(
                "{context}: result allocation failed"
            )));
        }
        let scratch = forge_sys::ggml_init(forge_sys::ggml_init_params {
            mem_size: forge_sys::ggml_graph_overhead() + 4 * forge_sys::ggml_tensor_overhead(),
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        });
        if scratch.is_null() {
            forge_sys::ggml_backend_buffer_free(buffer);
            forge_sys::ggml_free(ctx);
            return Err(Error::backend(format!("{context}: scratch context failed")));
        }
        let graph = forge_sys::ggml_new_graph(scratch);
        if graph.is_null() {
            forge_sys::ggml_free(scratch);
            forge_sys::ggml_backend_buffer_free(buffer);
            forge_sys::ggml_free(ctx);
            return Err(Error::backend(format!("{context}: graph creation failed")));
        }
        forge_sys::ggml_build_forward_expand(graph, out);
        let status = forge_sys::ggml_backend_graph_compute(backend.raw(), graph);
        forge_sys::ggml_free(scratch);
        if status != forge_sys::status::SUCCESS {
            forge_sys::ggml_backend_buffer_free(buffer);
            forge_sys::ggml_free(ctx);
            return Err(status_error(context, status));
        }
        Ok(Tensor::wrap_computed(
            backend,
            ctx,
            out,
            buffer,
            DType::F32,
            shape,
        ))
    }
}

fn check_pair(context: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    if !Rc::ptr_eq(a.backend_inner(), b.backend_inner()) {
        return Err(Error::invalid(format!(
            "{context}: tensors live on different backends"
        )));
    }
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        return Err(Error::unsupported(format!(
            "{context} only supports F32, got {} and {}",
            a.dtype().name(),
            b.dtype().name()
        )));
    }
    Ok(())
}

/// Element-wise add of two same-shape F32 tensors on the same backend.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_pair("add", a, b)?;
    if a.shape() != b.shape() {
        return Err(Error::invalid(format!(
            "add: shape mismatch {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let ctx = new_ctx(1)?;
    // SAFETY: ctx/a/b are live; ggml_add with equal shapes cannot fail,
    // but NULL is still checked.
    let out = unsafe { forge_sys::ggml_add(ctx, a.raw(), b.raw()) };
    if out.is_null() {
        unsafe { forge_sys::ggml_free(ctx) };
        return Err(Error::backend("add: ggml graph node creation failed"));
    }
    let backend = a_backend(a);
    finish(&backend, "add", ctx, out, a.shape().to_vec())
}

/// Matrix product of two 2-D F32 tensors on the same backend.
///
/// Shapes are ggml `ne` order: `a` is `[k, m]`, `b` is `[k, n]`, and the
/// result is `[m, n]`. Semantically this is `C = A·Bᵀ` for the row-major
/// `[m, k]` matrix in `a`'s buffer and the row-major `[n, k]` matrix in
/// `b`'s buffer; elementwise, with `a`, `b`, `result` the raw buffers:
///
/// `result[m + n*m] = Σ_k a[m*k + k]·b[n*k + k]`
///
/// (ggml dotes src0 rows against src1 rows; see
/// `ggml_compute_forward_mul_mat_one_chunk`.)
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    check_pair("matmul", a, b)?;
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(Error::invalid(format!(
            "matmul: both inputs must be 2-D, got {:?} and {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let (k, m) = (a.shape()[0], a.shape()[1]);
    let (kb, n) = (b.shape()[0], b.shape()[1]);
    if k != kb {
        return Err(Error::invalid(format!(
            "matmul: inner extents differ ({k} vs {kb})"
        )));
    }
    let ctx = new_ctx(1)?;
    // SAFETY: ctx/a/b are live and inner extents match; NULL checked.
    let out = unsafe { forge_sys::ggml_mul_mat(ctx, a.raw(), b.raw()) };
    if out.is_null() {
        unsafe { forge_sys::ggml_free(ctx) };
        return Err(Error::backend("matmul: ggml graph node creation failed"));
    }
    let backend = a_backend(a);
    finish(&backend, "matmul", ctx, out, vec![m, n])
}

/// Recover the [`Backend`] handle from a tensor's shared owner.
///
/// `Tensor` keeps only the inner `Rc`; rebuilding the public handle is a
/// cheap clone of that `Rc`.
fn a_backend(a: &Tensor) -> Backend {
    Backend::from_inner(Rc::clone(a.backend_inner()))
}
