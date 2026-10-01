//! Owned ggml backend handle.
//!
//! [`Backend`] owns one `ggml_backend_t` (a CPU backend, a CUDA backend,
//! ...) and frees it on drop. Tensors and graphs borrow it; cloning a
//! `Backend` shares the underlying instance. Like all ForgeCore handles,
//! it is `!Send + !Sync` (it contains a raw pointer), so backends stay on
//! the thread that opened them.

use crate::device::{self, DeviceInfo, DeviceType};
use crate::error::{Error, Result};
use std::ptr;
use std::rc::Rc;

/// Reference-counted backend instance behind [`Backend`] and [`Tensor`](crate::tensor::Tensor).
pub(crate) struct BackendInner {
    raw: *mut forge_sys::ggml_backend,
    device_type: DeviceType,
    name: String,
}

impl Drop for BackendInner {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful ggml init call, is freed
        // exactly once (Drop on the shared owner), and no Tensor outlives
        // the Rc, so no use-after-free is possible.
        unsafe { forge_sys::ggml_backend_free(self.raw) };
    }
}

/// An owned ggml backend (compute device context).
#[derive(Clone)]
pub struct Backend {
    inner: Rc<BackendInner>,
}

impl std::fmt::Debug for Backend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Backend")
            .field("name", &self.inner.name)
            .field("device_type", &self.inner.device_type)
            .finish_non_exhaustive()
    }
}

impl Backend {
    /// Open the CPU backend (registry lookup, `ggml_backend_cpu_init` fallback).
    pub fn open_cpu() -> Result<Self> {
        device::ensure_registry();
        // SAFETY: NULL params select defaults; a NULL return means
        // "no backend", which we check.
        let mut raw =
            unsafe { forge_sys::ggml_backend_init_by_type(forge_sys::dev_type::CPU, ptr::null()) };
        if raw.is_null() {
            // SAFETY: no arguments; NULL return checked below.
            raw = unsafe { forge_sys::ggml_backend_cpu_init() };
        }
        if raw.is_null() {
            return Err(Error::backend("CPU backend initialization failed"));
        }
        Ok(Self {
            inner: Rc::new(BackendInner {
                raw,
                device_type: DeviceType::Cpu,
                name: "CPU".to_string(),
            }),
        })
    }

    /// Open the backend for a device previously returned by
    /// [`enumerate_devices`](crate::device::enumerate_devices).
    pub fn open_device(info: &DeviceInfo) -> Result<Self> {
        device::ensure_registry();
        if info.index >= device::device_count() {
            return Err(Error::invalid(format!(
                "stale device index {} (registry holds {})",
                info.index,
                device::device_count()
            )));
        }
        // SAFETY: index is bounds-checked, so the device handle is valid;
        // NULL params select defaults; NULL returns are checked.
        unsafe {
            let dev = forge_sys::ggml_backend_dev_get(info.index);
            if dev.is_null() {
                return Err(Error::backend(format!(
                    "device {} ({}) vanished from the registry",
                    info.index, info.name
                )));
            }
            let raw = forge_sys::ggml_backend_dev_init(dev, ptr::null());
            if raw.is_null() {
                return Err(Error::backend(format!(
                    "backend initialization failed for device {} ({})",
                    info.index, info.name
                )));
            }
            Ok(Self {
                inner: Rc::new(BackendInner {
                    raw,
                    device_type: info.device_type,
                    name: info.name.clone(),
                }),
            })
        }
    }

    /// Set the worker thread count. Only valid on the CPU backend.
    pub fn set_cpu_threads(&self, n_threads: u32) -> Result<()> {
        if self.inner.device_type != DeviceType::Cpu {
            return Err(Error::unsupported(
                "thread count can only be set on the CPU backend",
            ));
        }
        if n_threads == 0 {
            return Err(Error::invalid("thread count must be at least 1"));
        }
        let n_threads = std::os::raw::c_int::try_from(n_threads)
            .map_err(|_| Error::invalid("thread count too large"))?;
        // SAFETY: raw is a live CPU backend (type-checked above) and the
        // call takes no pointers.
        unsafe { forge_sys::ggml_backend_cpu_set_n_threads(self.inner.raw, n_threads) };
        Ok(())
    }

    /// Kind of device this backend runs on.
    pub fn device_type(&self) -> DeviceType {
        self.inner.device_type
    }

    /// Backend display name (from the device snapshot, or `"CPU"`).
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    pub(crate) fn from_inner(inner: Rc<BackendInner>) -> Self {
        Self { inner }
    }

    pub(crate) fn raw(&self) -> *mut forge_sys::ggml_backend {
        self.inner.raw
    }

    pub(crate) fn inner(&self) -> &Rc<BackendInner> {
        &self.inner
    }
}
