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

/// Snapshot one registry device by index (bounds- and NULL-checked).
pub(crate) fn snapshot(index: usize) -> Result<DeviceInfo, crate::error::Error> {
    ensure_registry();
    if index >= device_count() {
        return Err(crate::error::Error::invalid(format!(
            "stale device index {index} (registry holds {})",
            device_count()
        )));
    }
    // SAFETY: index < dev_count, so the handle is valid for the
    // registry's lifetime (process-wide); NULL (vanishing device) is
    // checked, and every string is NULL-checked before reading.
    unsafe {
        let dev = forge_sys::ggml_backend_dev_get(index);
        if dev.is_null() {
            return Err(crate::error::Error::backend(format!(
                "device {index} vanished from the registry"
            )));
        }
        let name = optional_str(forge_sys::ggml_backend_dev_name(dev));
        let description = optional_str(forge_sys::ggml_backend_dev_description(dev));
        let device_type = DeviceType::from_ggml(forge_sys::ggml_backend_dev_type(dev));
        let mut free = 0usize;
        let mut total = 0usize;
        forge_sys::ggml_backend_dev_memory(dev, &mut free, &mut total);
        Ok(DeviceInfo {
            index,
            name,
            description,
            device_type,
            memory_free: free,
            memory_total: total,
        })
    }
}

/// Copy a native string pointer, tolerating NULL (yields `""`).
unsafe fn optional_str(ptr: *const std::os::raw::c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: non-NULL static native string.
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

/// Snapshot every device in the ggml backend registry (devices that
/// vanish mid-scan are skipped).
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    ensure_registry();
    let count = device_count();
    let mut devices = Vec::with_capacity(count);
    for index in 0..count {
        if let Ok(info) = snapshot(index) {
            devices.push(info);
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

/// Functionality flags for one registry device
/// (`ggml_backend_dev_caps`, copied out by value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCaps {
    /// Device supports asynchronous compute.
    pub async_compute: bool,
    /// Device can allocate host-visible buffers.
    pub host_buffer: bool,
    /// Device can wrap caller-owned host pointers.
    pub buffer_from_host_ptr: bool,
    /// Device supports events.
    pub events: bool,
    /// Backend build supports memory mapping.
    pub mmap: bool,
}

/// Full property block for one registry device
/// (`ggml_backend_dev_props`, copied out by value — no borrowed
/// pointers escape).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceProps {
    pub name: String,
    pub description: String,
    pub memory_free: usize,
    pub memory_total: usize,
    pub device_type: DeviceType,
    /// Backend device id (NULL natively — e.g. on CPU — maps to `None`).
    pub device_id: Option<String>,
    pub caps: DeviceCaps,
}

/// One ForgeCore op family for [`DeviceInfo::supports_op`].
///
/// Dtype-carrying variants exist where device support genuinely
/// varies by dtype (different kernels per conversion/table type);
/// the fixed variants are probed with representative P6 operands
/// (F32 data, I32 positions/indices, F32 masks; RoPE is probed NeoX,
/// full head dim — CPU and CUDA support verdicts are mode-agnostic,
/// verified in source).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpSpec {
    Add,
    Sub,
    Mul,
    Div,
    Matmul,
    Silu,
    Sqr,
    Sqrt,
    Scale,
    RmsNorm,
    Norm,
    Softmax,
    SoftmaxExt,
    Rope,
    Cast {
        from: crate::dtype::DType,
        to: crate::dtype::DType,
    },
    GetRows {
        table: crate::dtype::DType,
    },
    Concat {
        dtype: crate::dtype::DType,
    },
}

impl DeviceInfo {
    /// Read this device's full property block (fails only for stale
    /// indices or vanishing devices).
    pub fn props(&self) -> Result<DeviceProps, crate::error::Error> {
        ensure_registry();
        if self.index >= device_count() {
            return Err(crate::error::Error::invalid(format!(
                "stale device index {} (registry holds {})",
                self.index,
                device_count()
            )));
        }
        // SAFETY: index is bounds-checked; the props struct is a
        // plain out-block (`get_props` memsets it first); every
        // string is copied out immediately with NULL checks.
        unsafe {
            let dev = forge_sys::ggml_backend_dev_get(self.index);
            if dev.is_null() {
                return Err(crate::error::Error::backend(format!(
                    "device {} ({}) vanished from the registry",
                    self.index, self.name
                )));
            }
            let mut raw: forge_sys::ggml_backend_dev_props = std::mem::zeroed();
            forge_sys::ggml_backend_dev_get_props(dev, &mut raw);
            let device_id = if raw.device_id.is_null() {
                None
            } else {
                Some(CStr::from_ptr(raw.device_id).to_string_lossy().into_owned())
            };
            Ok(DeviceProps {
                name: optional_str(raw.name),
                description: optional_str(raw.description),
                memory_free: raw.memory_free,
                memory_total: raw.memory_total,
                device_type: DeviceType::from_ggml(raw.type_),
                device_id,
                caps: DeviceCaps {
                    async_compute: raw.caps.r#async,
                    host_buffer: raw.caps.host_buffer,
                    buffer_from_host_ptr: raw.caps.buffer_from_host_ptr,
                    events: raw.caps.events,
                    mmap: raw.caps.mmap_support,
                },
            })
        }
    }

    /// Query whether this device implements `op`
    /// (`ggml_backend_dev_supports_op`) by building a fully
    /// allocated probe node with representative P6 operands and
    /// asking the device.
    ///
    /// The probe tensors are allocated from the device's own buffer
    /// type (never computed — only queried): CPU support reads
    /// metadata alone, but other devices' queries are only
    /// source-audited, so allocated probes keep every device on its
    /// safe path. Dtype-carrying specs outside the P6-supported
    /// envelope fail instead of probing (a verdict there would be
    /// meaningless — ForgeCore never issues such ops).
    pub fn supports_op(&self, op: OpSpec) -> Result<bool, crate::error::Error> {
        use crate::error::Error;
        match op {
            OpSpec::Cast { from, to } => {
                if !crate::tensor::cast_pair_supported(from, to) {
                    return Err(Error::invalid(format!(
                        "cannot probe unsupported cast {} -> {}",
                        from.name(),
                        to.name()
                    )));
                }
            }
            OpSpec::GetRows { table } if !crate::runtime::get_rows_table_supported(table) => {
                return Err(Error::invalid(format!(
                    "cannot probe unsupported get_rows table {}",
                    table.name()
                )));
            }
            _ => {}
        }
        ensure_registry();
        if self.index >= device_count() {
            return Err(Error::invalid(format!(
                "stale device index {} (registry holds {})",
                self.index,
                device_count()
            )));
        }
        // SAFETY: index is bounds-checked; the probe builder checks
        // every NULL return and frees its context and buffer on all
        // paths; the device query only reads the probe's metadata.
        unsafe {
            let dev = forge_sys::ggml_backend_dev_get(self.index);
            if dev.is_null() {
                return Err(Error::backend(format!(
                    "device {} ({}) vanished from the registry",
                    self.index, self.name
                )));
            }
            let buft = forge_sys::ggml_backend_dev_buffer_type(dev);
            if buft.is_null() {
                return Err(Error::backend(format!(
                    "device {} ({}) has no buffer type",
                    self.index, self.name
                )));
            }
            self.probe_op(dev, buft, op)
        }
    }

    /// Build one allocated probe node for `op` and query the device.
    /// Caller guarantees live `dev`/`buft`; every failure path frees
    /// what it allocated.
    unsafe fn probe_op(
        &self,
        dev: forge_sys::ggml_backend_dev_t,
        buft: forge_sys::ggml_backend_buffer_type_t,
        op: OpSpec,
    ) -> Result<bool, crate::error::Error> {
        use crate::dtype::DType;
        use crate::error::Error;
        // Scratch metadata context (inputs + op node + slack).
        let ctx = crate::tensor::new_ctx(6)
            .map_err(|_| Error::backend("op probe context allocation failed"))?;
        // One probe input; NULL on context OOM.
        unsafe fn input(
            ctx: *mut forge_sys::ggml_context,
            dtype: DType,
            ne: &[i64],
        ) -> *mut forge_sys::ggml_tensor {
            // SAFETY: caller guarantees a live context; `ne` points
            // to `ne.len()` valid extents; dtype ids are mapped.
            forge_sys::ggml_new_tensor(
                ctx,
                dtype.ggml_type(),
                ne.len() as std::os::raw::c_int,
                ne.as_ptr(),
            )
        }
        // Build the probe node; `None` = input/node creation failed.
        let node = match op {
            OpSpec::Add | OpSpec::Sub | OpSpec::Mul | OpSpec::Div => {
                let a = input(ctx, DType::F32, &[4, 3]);
                let b = input(ctx, DType::F32, &[4, 3]);
                if a.is_null() || b.is_null() {
                    None
                } else {
                    Some(match op {
                        OpSpec::Add => forge_sys::ggml_add(ctx, a, b),
                        OpSpec::Sub => forge_sys::ggml_sub(ctx, a, b),
                        OpSpec::Mul => forge_sys::ggml_mul(ctx, a, b),
                        _ => forge_sys::ggml_div(ctx, a, b),
                    })
                }
            }
            OpSpec::Matmul => {
                let a = input(ctx, DType::F32, &[8, 4]);
                let b = input(ctx, DType::F32, &[8, 5]);
                if a.is_null() || b.is_null() {
                    None
                } else {
                    Some(forge_sys::ggml_mul_mat(ctx, a, b))
                }
            }
            OpSpec::Silu
            | OpSpec::Sqr
            | OpSpec::Sqrt
            | OpSpec::Scale
            | OpSpec::RmsNorm
            | OpSpec::Norm
            | OpSpec::Softmax => {
                let a = input(ctx, DType::F32, &[8, 5]);
                if a.is_null() {
                    None
                } else {
                    Some(match op {
                        OpSpec::Silu => forge_sys::ggml_silu(ctx, a),
                        OpSpec::Sqr => forge_sys::ggml_sqr(ctx, a),
                        OpSpec::Sqrt => forge_sys::ggml_sqrt(ctx, a),
                        OpSpec::Scale => forge_sys::ggml_scale(ctx, a, 2.0),
                        OpSpec::RmsNorm => forge_sys::ggml_rms_norm(ctx, a, 1e-5),
                        OpSpec::Norm => forge_sys::ggml_norm(ctx, a, 1e-5),
                        _ => forge_sys::ggml_soft_max(ctx, a),
                    })
                }
            }
            OpSpec::SoftmaxExt => {
                let a = input(ctx, DType::F32, &[8, 5]);
                let mask = input(ctx, DType::F32, &[8, 5]);
                if a.is_null() || mask.is_null() {
                    None
                } else {
                    Some(forge_sys::ggml_soft_max_ext(ctx, a, mask, 1.0, 0.0))
                }
            }
            OpSpec::Rope => {
                let a = input(ctx, DType::F32, &[8, 4, 6]);
                let pos = input(ctx, DType::I32, &[6]);
                if a.is_null() || pos.is_null() {
                    None
                } else {
                    Some(forge_sys::ggml_rope_ext(
                        ctx,
                        a,
                        pos,
                        std::ptr::null_mut(),
                        8,
                        forge_sys::rope_type::NEOX,
                        32,
                        10_000.0,
                        1.0,
                        0.0,
                        1.0,
                        32.0,
                        1.0,
                    ))
                }
            }
            OpSpec::Cast { from, to } => {
                // dim 0 = 256: block-divisible for every quant type on
                // either end, including the 256-wide K super-blocks
                // (creation rule for `from`, quantizer rule for `to`;
                // a ragged probe would carry truncated strides and
                // could misreport on backends that inspect dim 0).
                let a = input(ctx, from, &[256, 4]);
                if a.is_null() {
                    None
                } else {
                    Some(forge_sys::ggml_cast(ctx, a, to.ggml_type()))
                }
            }
            OpSpec::GetRows { table } => {
                // dim 0 = 256: block-divisible for every quant table
                // including K super-blocks (see the cast probe).
                let t = input(ctx, table, &[256, 8]);
                let idx = input(ctx, DType::I32, &[5]);
                if t.is_null() || idx.is_null() {
                    None
                } else {
                    Some(forge_sys::ggml_get_rows(ctx, t, idx))
                }
            }
            OpSpec::Concat { dtype } => {
                // dim 0 = 256: block-divisible for every quant dtype
                // including K super-blocks (see the cast probe).
                let a = input(ctx, dtype, &[256, 3]);
                let b = input(ctx, dtype, &[256, 5]);
                if a.is_null() || b.is_null() {
                    None
                } else {
                    Some(forge_sys::ggml_concat(ctx, a, b, 1))
                }
            }
        };
        let node = match node {
            Some(node) if !node.is_null() => node,
            _ => {
                forge_sys::ggml_free(ctx);
                return Err(Error::backend("op probe node creation failed"));
            }
        };
        // Allocate the probe from the device's own buffer type so
        // the support query runs on the device's safe path.
        let buffer = forge_sys::ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if buffer.is_null() {
            forge_sys::ggml_free(ctx);
            return Err(Error::backend("op probe allocation failed"));
        }
        let verdict = forge_sys::ggml_backend_dev_supports_op(dev, node);
        forge_sys::ggml_backend_buffer_free(buffer);
        forge_sys::ggml_free(ctx);
        Ok(verdict)
    }
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
