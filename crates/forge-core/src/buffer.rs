//! Buffer types and owned buffers.
//!
//! A [`BufferType`] is a borrowed handle describing how one backend
//! allocates memory (name, alignment, limits, host visibility); a
//! [`Buffer`] is an owned allocation freed on drop. [`Tensor`](crate::tensor::Tensor)
//! storage is managed internally, so most callers only need
//! [`BufferType`] for capacity questions
//! ([`tensor_alloc_size`](BufferType::tensor_alloc_size),
//! [`max_size`](BufferType::max_size)) answered from genuine native
//! facts — never estimates.

use crate::error::{Error, Result};
use crate::tensor::Tensor;
use std::ffi::CStr;
use std::marker::PhantomData;

/// Borrowed buffer-type handle (`ggml_backend_buffer_type_t`).
///
/// Buffer types are owned by the backend or device that produced
/// them; the borrow keeps the owner alive. Every getter only asserts
/// liveness, so all of these are total.
pub struct BufferType<'a> {
    raw: forge_sys::ggml_backend_buffer_type_t,
    _owner: PhantomData<&'a ()>,
}

impl<'a> BufferType<'a> {
    pub(crate) fn from_raw(raw: forge_sys::ggml_backend_buffer_type_t) -> Self {
        Self {
            raw,
            _owner: PhantomData,
        }
    }

    /// Buffer-type name (e.g. `"CPU"`, `"CUDA_Host"`).
    pub fn name(&self) -> String {
        // SAFETY: buft is borrowed live; the name is a static string.
        unsafe {
            CStr::from_ptr(forge_sys::ggml_backend_buft_name(self.raw))
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Required allocation alignment in bytes.
    pub fn alignment(&self) -> usize {
        // SAFETY: buft is borrowed live.
        unsafe { forge_sys::ggml_backend_buft_get_alignment(self.raw) }
    }

    /// Largest single allocation in bytes (0 = no backend limit
    /// beyond address space).
    pub fn max_size(&self) -> usize {
        // SAFETY: buft is borrowed live.
        unsafe { forge_sys::ggml_backend_buft_get_max_size(self.raw) }
    }

    /// Whether memory of this type is directly host-accessible.
    pub fn is_host(&self) -> bool {
        // SAFETY: buft is borrowed live.
        unsafe { forge_sys::ggml_backend_buft_is_host(self.raw) }
    }

    /// Padded bytes this buffer type would reserve for `tensor`
    /// (views report 0 — they reserve nothing).
    pub fn tensor_alloc_size(&self, tensor: &Tensor) -> usize {
        // SAFETY: buft is borrowed live and the tensor is live; the
        // query only reads metadata from both.
        unsafe { forge_sys::ggml_backend_buft_get_alloc_size(self.raw, tensor.raw()) }
    }

    /// Allocate an owned [`Buffer`] of `size` bytes (size 0 yields
    /// the native dummy buffer — total either way; OOM fails).
    pub fn alloc_buffer(&self, size: usize) -> Result<Buffer> {
        // SAFETY: buft is borrowed live; NULL (OOM) is checked.
        let raw = unsafe { forge_sys::ggml_backend_buft_alloc_buffer(self.raw, size) };
        if raw.is_null() {
            return Err(Error::backend(format!(
                "buffer allocation of {size} bytes failed"
            )));
        }
        Ok(Buffer { raw })
    }
}

impl std::fmt::Debug for BufferType<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BufferType")
            .field("name", &self.name())
            .field("alignment", &self.alignment())
            .field("max_size", &self.max_size())
            .field("is_host", &self.is_host())
            .finish_non_exhaustive()
    }
}

/// An owned backend buffer, freed on drop.
pub struct Buffer {
    raw: *mut forge_sys::ggml_backend_buffer,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful allocation and is freed
        // exactly once; no tensor aliases it (buffers never back
        // ForgeCore tensors directly).
        unsafe { forge_sys::ggml_backend_buffer_free(self.raw) };
    }
}

impl Buffer {
    /// Usable size in bytes.
    pub fn size(&self) -> usize {
        // SAFETY: buffer is live; only asserts liveness.
        unsafe { forge_sys::ggml_backend_buffer_get_size(self.raw) }
    }

    /// Allocation alignment in bytes.
    pub fn alignment(&self) -> usize {
        // SAFETY: buffer is live; only asserts liveness.
        unsafe { forge_sys::ggml_backend_buffer_get_alignment(self.raw) }
    }

    /// Largest single allocation in bytes (backend limit).
    pub fn max_size(&self) -> usize {
        // SAFETY: buffer is live; only asserts liveness.
        unsafe { forge_sys::ggml_backend_buffer_get_max_size(self.raw) }
    }

    /// Whether this memory is directly host-accessible.
    pub fn is_host(&self) -> bool {
        // SAFETY: buffer is live; only asserts liveness.
        unsafe { forge_sys::ggml_backend_buffer_is_host(self.raw) }
    }

    /// Buffer name (backend-defined, e.g. `"CPU"`).
    pub fn name(&self) -> String {
        // SAFETY: buffer is live; the name is a static string.
        unsafe {
            CStr::from_ptr(forge_sys::ggml_backend_buffer_name(self.raw))
                .to_string_lossy()
                .into_owned()
        }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Buffer")
            .field("name", &self.name())
            .field("size", &self.size())
            .finish_non_exhaustive()
    }
}
