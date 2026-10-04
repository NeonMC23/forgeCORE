//! Backend device discovery.
//!
//! [`enumerate_devices`] snapshots the ggml backend registry (CPU, CUDA,
//! Vulkan, ... — whatever the native build registered) into plain Rust
//! values. No raw pointers escape: [`DeviceInfo`] owns its strings and a
//! [`Backend`](crate::backend::Backend) is opened from it by index.

use std::ffi::CStr;
use std::sync::OnceLock;

/// ggml backend device kind, mirroring `enum ggml_backend_dev_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Cpu,
    Gpu,
    Igpu,
    Accel,
    Meta,
    /// Registry reported a kind this ForgeCore does not know yet.
    Unknown(i32),
}

impl DeviceType {
    fn from_ggml(id: i32) -> Self {
        match id {
            forge_sys::dev_type::CPU => Self::Cpu,
            forge_sys::dev_type::GPU => Self::Gpu,
            forge_sys::dev_type::IGPU => Self::Igpu,
            forge_sys::dev_type::ACCEL => Self::Accel,
            forge_sys::dev_type::META => Self::Meta,
            other => Self::Unknown(other),
        }
    }
}

/// Owned snapshot of one ggml backend device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Registry index; use with [`Backend::open_device`](crate::backend::Backend::open_device).
    pub index: usize,
    pub name: String,
    pub description: String,
    pub device_type: DeviceType,
    pub memory_free: usize,
    pub memory_total: usize,
}

/// Load the ggml backend registry exactly once per process.
pub(crate) fn ensure_registry() {
    static LOADED: OnceLock<()> = OnceLock::new();
    LOADED.get_or_init(|| {
        // SAFETY: ggml_backend_load_all takes no arguments and only
        // registers the statically linked backends; idempotent.
        unsafe { forge_sys::ggml_backend_load_all() };
    });
}

/// Number of devices currently in the ggml backend registry.
pub(crate) fn device_count() -> usize {
    ensure_registry();
    // SAFETY: no arguments; valid once the registry is loaded.
    unsafe { forge_sys::ggml_backend_dev_count() }
}

/// Snapshot every device in the ggml backend registry.
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    ensure_registry();
    let count =
        // SAFETY: no arguments; valid once the registry is loaded.
        unsafe { forge_sys::ggml_backend_dev_count() };
    let mut devices = Vec::with_capacity(count);
    for index in 0..count {
        // SAFETY: index < dev_count, so the handle is valid for the
        // registry's lifetime (process-wide); name/description/memory
        // getters take a valid handle and plain out-pointers.
        unsafe {
            let dev = forge_sys::ggml_backend_dev_get(index);
            let name = CStr::from_ptr(forge_sys::ggml_backend_dev_name(dev)).to_string_lossy();
            let description =
                CStr::from_ptr(forge_sys::ggml_backend_dev_description(dev)).to_string_lossy();
            let device_type = DeviceType::from_ggml(forge_sys::ggml_backend_dev_type(dev));
            let mut free = 0usize;
            let mut total = 0usize;
            forge_sys::ggml_backend_dev_memory(dev, &mut free, &mut total);
            devices.push(DeviceInfo {
                index,
                name: name.into_owned(),
                description: description.into_owned(),
                device_type,
                memory_free: free,
                memory_total: total,
            });
        }
    }
    devices
}

/// Maximum device count upstream supports for model offload
/// (`llama_max_devices`).
///
/// The pinned upstream returns the constant 16. A caller-supplied
/// `tensor_split` array is read by index per selected device, so
/// [`ModelOptions`](crate::model::ModelOptions) validates split lengths
/// against the effective device count; this bound sizes such arrays.
pub fn max_devices() -> usize {
    // SAFETY: no arguments; returns a constant, never fails.
    unsafe { forge_sys::llama_max_devices() }
}

/// Whether the platform build supports memory-mapping model files
/// (`llama_supports_mmap`).
///
/// A compile-time platform flag (true on Linux); requesting mmap where
/// it is unsupported makes upstream warn and load without mmap rather
/// than fail.
pub fn supports_mmap() -> bool {
    // SAFETY: no arguments; returns a constant, never fails.
    unsafe { forge_sys::llama_supports_mmap() }
}

/// Whether the platform build supports locking model mappings in RAM
/// (`llama_supports_mlock`).
///
/// A compile-time platform flag (true on Linux), independent of
/// runtime `mlock` limits.
pub fn supports_mlock() -> bool {
    // SAFETY: no arguments; returns a constant, never fails.
    unsafe { forge_sys::llama_supports_mlock() }
}

/// Whether upstream reports GPU offload as available
/// (`llama_supports_gpu_offload`).
///
/// True when the backend registry holds a GPU or IGPU device, or RPC
/// support is compiled in. Self-initializing: upstream loads the
/// registry on first use, so no `ensure_registry()` call is needed.
/// [`Model::load_with_options`](crate::model::Model::load_with_options)
/// refuses GPU layer requests under default device selection while
/// this is false (explicit error, never silent CPU execution).
pub fn supports_gpu_offload() -> bool {
    // SAFETY: no arguments; loads the registry itself when needed.
    unsafe { forge_sys::llama_supports_gpu_offload() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_type_from_ggml_ids() {
        use forge_sys::dev_type;
        assert_eq!(DeviceType::from_ggml(dev_type::CPU), DeviceType::Cpu);
        assert_eq!(DeviceType::from_ggml(dev_type::GPU), DeviceType::Gpu);
        assert_eq!(DeviceType::from_ggml(dev_type::IGPU), DeviceType::Igpu);
        assert_eq!(DeviceType::from_ggml(dev_type::ACCEL), DeviceType::Accel);
        assert_eq!(DeviceType::from_ggml(dev_type::META), DeviceType::Meta);
        assert_eq!(DeviceType::from_ggml(1234), DeviceType::Unknown(1234));
    }
}
