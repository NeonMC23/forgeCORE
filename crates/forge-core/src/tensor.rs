//! Owned ggml tensor with its storage.
//!
//! [`Tensor`] owns everything ggml needs for one tensor: a metadata
//! context, the tensor itself, and the backend buffer holding its data.
//! Shapes use ggml `ne` order (extent of dimension 0 first, up to 4
//! dimensions); data layout is exactly what ggml uses, with no hidden
//! transposes. Dropping a `Tensor` frees the buffer, then the context.

use crate::backend::{Backend, BackendInner};
use crate::dtype::DType;
use crate::error::{Error, Result};
use std::ffi::c_void;
use std::ptr;
use std::rc::Rc;

/// Maximum tensor rank (ggml supports up to 4 dimensions).
pub const MAX_DIMS: usize = 4;

/// Allocate a metadata-only ggml context sized for `n_tensors` tensors.
pub(crate) fn new_ctx(n_tensors: usize) -> Result<*mut forge_sys::ggml_context> {
    // SAFETY: no arguments.
    let per_tensor = unsafe { forge_sys::ggml_tensor_overhead() };
    let mem_size = per_tensor
        .checked_mul(n_tensors.saturating_add(2))
        .ok_or_else(|| Error::invalid("tensor context size overflow"))?;
    let params = forge_sys::ggml_init_params {
        mem_size,
        mem_buffer: ptr::null_mut(),
        no_alloc: true,
    };
    // SAFETY: params are valid (owned pool, NULL buffer = allocate
    // internally); NULL return (OOM) is checked by the caller.
    let ctx = unsafe { forge_sys::ggml_init(params) };
    if ctx.is_null() {
        return Err(Error::backend("ggml context allocation failed"));
    }
    Ok(ctx)
}

/// Validate a Rust shape into ggml `ne` extents plus the element count.
fn checked_ne(shape: &[usize]) -> Result<(Vec<i64>, usize)> {
    if shape.is_empty() || shape.len() > MAX_DIMS {
        return Err(Error::invalid(format!(
            "rank must be 1..={MAX_DIMS}, got {}",
            shape.len()
        )));
    }
    let mut ne = Vec::with_capacity(shape.len());
    let mut nelements = 1usize;
    for (axis, &extent) in shape.iter().enumerate() {
        if extent == 0 {
            return Err(Error::invalid(format!("axis {axis} has extent 0")));
        }
        let extent_i64 =
            i64::try_from(extent).map_err(|_| Error::invalid("extent exceeds i64 range"))?;
        nelements = nelements
            .checked_mul(extent)
            .ok_or_else(|| Error::invalid("element count overflow"))?;
        ne.push(extent_i64);
    }
    Ok((ne, nelements))
}

/// An owned tensor living on one [`Backend`].
pub struct Tensor {
    backend: Rc<BackendInner>,
    ctx: *mut forge_sys::ggml_context,
    raw: *mut forge_sys::ggml_tensor,
    buffer: *mut forge_sys::ggml_backend_buffer,
    dtype: DType,
    shape: Vec<usize>,
}

impl Drop for Tensor {
    fn drop(&mut self) {
        // SAFETY: buffer and ctx are distinct live allocations owned by
        // this Tensor; buffer first (data), then the metadata context.
        // No other owner exists, so neither can be double-freed.
        unsafe {
            forge_sys::ggml_backend_buffer_free(self.buffer);
            forge_sys::ggml_free(self.ctx);
        }
    }
}

impl Tensor {
    /// Create an F32 tensor on `backend` and upload row bytes from `data`.
    ///
    /// `shape` is ggml `ne` order; `data.len()` must equal the element
    /// count and is uploaded verbatim.
    pub fn from_f32(backend: &Backend, shape: &[usize], data: &[f32]) -> Result<Self> {
        let tensor = Self::empty(backend, DType::F32, shape)?;
        if tensor.nelements() != data.len() {
            return Err(Error::invalid(format!(
                "data has {} elements but shape {shape:?} needs {}",
                data.len(),
                tensor.nelements()
            )));
        }
        let nbytes = tensor.nbytes();
        if nbytes != std::mem::size_of_val(data) {
            return Err(Error::backend("ggml byte size disagrees with F32 layout"));
        }
        // SAFETY: tensor is F32 with nbytes of live backend storage;
        // data points to nbytes of valid host memory.
        unsafe {
            forge_sys::ggml_backend_tensor_set(
                tensor.raw,
                data.as_ptr().cast::<c_void>(),
                0,
                nbytes,
            );
        }
        Ok(tensor)
    }

    /// Create an uninitialized tensor of `dtype`/`shape` on `backend`.
    ///
    /// Contents are undefined until written by an op or (for F32) read
    /// after [`from_f32`](Self::from_f32)-style upload.
    pub fn empty(backend: &Backend, dtype: DType, shape: &[usize]) -> Result<Self> {
        let (ne, _) = checked_ne(shape)?;
        let ctx = new_ctx(1)?;
        // SAFETY: ctx is live; ne points to shape.len() valid i64s; NULL
        // returns are checked and unwind the context.
        unsafe {
            let raw = forge_sys::ggml_new_tensor(
                ctx,
                dtype.ggml_type(),
                i32::try_from(ne.len()).map_err(|_| Error::invalid("rank exceeds i32 range"))?,
                ne.as_ptr(),
            );
            if raw.is_null() {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("ggml tensor creation failed"));
            }
            let buffer = forge_sys::ggml_backend_alloc_ctx_tensors(ctx, backend.raw());
            if buffer.is_null() {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("backend buffer allocation failed"));
            }
            Ok(Self {
                backend: Rc::clone(backend.inner()),
                ctx,
                raw,
                buffer,
                dtype,
                shape: shape.to_vec(),
            })
        }
    }

    /// Wrap a tensor produced by a ggml op plus its owned allocations.
    pub(crate) fn wrap_computed(
        backend: &Backend,
        ctx: *mut forge_sys::ggml_context,
        raw: *mut forge_sys::ggml_tensor,
        buffer: *mut forge_sys::ggml_backend_buffer,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            backend: Rc::clone(backend.inner()),
            ctx,
            raw,
            buffer,
            dtype,
            shape,
        }
    }

    /// Download an F32 tensor to the host.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(Error::unsupported(format!(
                "host download only supports F32, got {}",
                self.dtype.name()
            )));
        }
        let mut out = vec![0.0f32; self.nelements()];
        let nbytes = self.nbytes();
        if nbytes != std::mem::size_of_val(out.as_slice()) {
            return Err(Error::backend("ggml byte size disagrees with F32 layout"));
        }
        // SAFETY: tensor is F32 with nbytes of live backend storage; out
        // points to nbytes of valid host memory.
        unsafe {
            forge_sys::ggml_backend_tensor_get(
                self.raw,
                out.as_mut_ptr().cast::<c_void>(),
                0,
                nbytes,
            );
        }
        Ok(out)
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Shape in ggml `ne` order.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total element count.
    pub fn nelements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Storage size in bytes, as reported by ggml.
    pub fn nbytes(&self) -> usize {
        // SAFETY: raw is a live tensor; takes no aliased pointers.
        unsafe { forge_sys::ggml_nbytes(self.raw) }
    }

    pub(crate) fn raw(&self) -> *mut forge_sys::ggml_tensor {
        self.raw
    }

    pub(crate) fn backend_inner(&self) -> &Rc<BackendInner> {
        &self.backend
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}
