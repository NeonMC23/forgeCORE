//! Model loading and metadata via libllama.
//!
//! [`Model`] owns one `llama_model` (weights + metadata for a `.gguf`
//! file) through a shared handle: cloning a `Model` shares the native
//! model (as with [`Backend`](crate::backend::Backend)), and every
//! [`Context`](crate::context::Context) holds a share so the model
//! always outlives its contexts. Loads default to CPU (see
//! [`GpuLayers::Cpu`]); GPU offload is opt-in through [`ModelOptions`]
//! and is refused with an explicit error when no GPU device is
//! available — never silently executed on the CPU.

use crate::device::{self, DeviceInfo};
use crate::error::{Error, Result};
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr;
use std::rc::Rc;

/// Prompt kept from the pre-pivot validation notes so future real-model
/// fixtures can reference one canonical diagnostic input.
pub const QWEN25_DIAGNOSTIC_PROMPT: &str = "hi, what's 2+2=?";

/// Token IDs of [`QWEN25_DIAGNOSTIC_PROMPT`] under the recorded Qwen2.5
/// tokenizer. Valid only for that tokenizer; a fixture placeholder, not
/// a contract.
pub const QWEN25_DIAGNOSTIC_PROMPT_IDS: [u32; 9] = [6023, 11, 1128, 594, 220, 17, 10, 17, 19884];

/// Discard ggml/llama log output (used by tests for expected failures).
unsafe extern "C" fn log_sink(_level: c_int, _text: *const c_char, _user: *mut std::ffi::c_void) {}

/// Silence (`true`) or restore (`false`) ggml/llama log output.
///
/// Process-global, like the upstream log callback itself.
pub fn set_log_quiet(quiet: bool) {
    let callback: forge_sys::ggml_log_callback = quiet
        .then_some(log_sink as unsafe extern "C" fn(c_int, *const c_char, *mut std::ffi::c_void));
    // SAFETY: the callback only ignores its arguments; NULL user data is
    // never dereferenced.
    unsafe { forge_sys::llama_log_set(callback, ptr::null_mut()) };
}

/// Shared ownership of one native model behind [`Model`] and [`Context`](crate::context::Context).
pub(crate) struct ModelInner {
    raw: *mut forge_sys::llama_model,
}

impl Drop for ModelInner {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful load, is freed exactly once
        // (Drop on the shared owner), and no Context outlives the Rc.
        unsafe { forge_sys::llama_model_free(self.raw) };
    }
}

/// How many model layers to offload to GPU devices (upstream
/// `n_gpu_layers`).
///
/// Upstream clamps positive counts to the model's layer count (plus the
/// output layer) and treats any negative value as "all layers"; this
/// enum exposes exactly those three meanings with no other values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GpuLayers {
    /// Keep every layer on the CPU (`n_gpu_layers = 0`). The default:
    /// plain ForgeCore loads never touch a GPU unless asked.
    #[default]
    Cpu,
    /// Offload the first `n` layers (plus output-layer handling
    /// upstream) to the selected devices. `Count(0)` is `Cpu`.
    Count(u32),
    /// Offload all layers (`n_gpu_layers = -1`).
    All,
}

impl GpuLayers {
    fn as_native(self) -> Result<i32> {
        match self {
            Self::Cpu => Ok(0),
            Self::All => Ok(-1),
            Self::Count(n) => i32::try_from(n)
                .map_err(|_| Error::invalid(format!("gpu layer count {n} exceeds i32::MAX"))),
        }
    }
}

/// How to split offloaded layers across multiple GPU devices
/// (upstream `llama_split_mode`).
///
/// The default is [`SplitMode::Layer`], matching the upstream default —
/// not [`SplitMode::None`] — so default ForgeCore loads pass upstream
/// exactly the parameters they always have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SplitMode {
    /// Single GPU: the whole model goes to [`ModelOptions::main_gpu`].
    None,
    /// Split layers (and KV cache) across the selected devices.
    #[default]
    Layer,
    /// Split layers across devices, with tensor parallelism where the
    /// backend supports it.
    Row,
    /// Tensor parallelism across devices. Upstream rejects this mode
    /// for architectures without an implementation (a native load
    /// failure), and requires at least one device even when
    /// [`GpuLayers::Cpu`] is selected.
    Tensor,
}

impl SplitMode {
    fn as_native(self) -> i32 {
        match self {
            Self::None => forge_sys::split_mode::NONE,
            Self::Layer => forge_sys::split_mode::LAYER,
            Self::Row => forge_sys::split_mode::ROW,
            Self::Tensor => forge_sys::split_mode::TENSOR,
        }
    }
}

/// How the model file is mapped into memory (upstream
/// `llama_load_mode`, mmap/mlock subset).
///
/// The default is [`ModelLoadMode::Auto`], matching the upstream
/// default, which additionally lets upstream disable mmap for devices
/// without mmap support. Explicit modes bypass that per-device
/// resolution; requesting mmap where the platform lacks it makes
/// upstream warn and load without mmap rather than fail. Upstream's
/// `LLAMA_LOAD_MODE_DIRECT_IO` has no variant here and cannot be
/// selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModelLoadMode {
    /// Upstream default: mmap where supported, resolved per device.
    #[default]
    Auto,
    /// No memory mapping; read tensor data through plain I/O.
    NoMmap,
    /// Memory-map the model file.
    Mmap,
    /// Lock the model in RAM (without mmap).
    Mlock,
    /// Memory-map the model and lock the mapping in RAM.
    MmapMlock,
}

impl ModelLoadMode {
    fn as_native(self) -> i32 {
        match self {
            Self::Auto => forge_sys::load_mode::AUTO,
            Self::NoMmap => forge_sys::load_mode::NONE,
            Self::Mmap => forge_sys::load_mode::MMAP,
            Self::Mlock => forge_sys::load_mode::MLOCK,
            Self::MmapMlock => forge_sys::load_mode::MMAP_MLOCK,
        }
    }
}

/// Validate a caller-supplied `tensor_split` array's shape and values.
/// Length is checked against the effective device count by the caller.
fn validate_tensor_split_values(split: &[f32]) -> Result<()> {
    if split.is_empty() {
        return Err(Error::invalid(
            "tensor_split must not be empty; use None for the default split",
        ));
    }
    if split.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err(Error::invalid(
            "tensor_split entries must be finite and non-negative",
        ));
    }
    Ok(())
}

/// Model loading options.
///
/// `#[non_exhaustive]` so later phases can add options without breaking
/// callers. Every field defaults to the pre-offload behavior: CPU-only,
/// upstream default split/load modes, default device selection.
///
/// ## Offload model
///
/// - `gpu_layers` selects CPU execution, a layer count, or all layers.
/// - `devices` selects which registry devices take offloaded layers:
///   `None` keeps upstream's default selection (RPC servers first, then
///   GPUs, then integrated GPUs only when no discrete GPU exists; CPU
///   and accelerator devices skipped), while `Some(list)` passes the
///   listed devices to upstream verbatim in the given order — including
///   CPU devices, which upstream accepts. Indices are resolved against
///   the live registry at load time; duplicates are allowed.
/// - `tensor_split` (`None` = split by free memory; all-zero = likewise)
///   gives per-device proportions and must carry at least one entry per
///   effective device.
/// - `main_gpu` indexes the effective device list and is only read by
///   upstream under [`SplitMode::None`]; otherwise it is ignored.
/// - `load_mode` requests mmap/mlock behavior (see [`ModelLoadMode`]).
///
/// ## Failure behavior
///
/// Validation runs before any native call, so misconfiguration never
/// reaches upstream: bad layer counts, empty or ill-valued splits,
/// empty device lists, unknown device indices, and out-of-range
/// `main_gpu` (under [`SplitMode::None`]) are [`Error::invalid`].
/// Requesting GPU layers under default device selection while
/// [`supports_gpu_offload`](crate::device::supports_gpu_offload) is
/// false is [`Error::unsupported`] — upstream would otherwise load the
/// model and silently run it on the CPU. A NULL return from upstream
/// (bad file, tensor-split mode on an unsupported architecture, ...)
/// stays a plain model error.
///
/// What is *not* decided here: which devices to prefer, how many layers
/// fit a memory budget, when to evict or re-split, or any multi-GPU
/// orchestration. That policy belongs to RAMforge; this API only binds
/// the native knobs.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct ModelOptions {
    /// Validate model tensor data while loading (slower, safer).
    pub check_tensors: bool,
    /// How many layers to offload to the selected devices.
    pub gpu_layers: GpuLayers,
    /// How to split offloaded layers across devices.
    pub split_mode: SplitMode,
    /// Index into the effective device list used for the whole model
    /// under [`SplitMode::None`]; ignored by upstream otherwise.
    pub main_gpu: usize,
    /// Per-device offload proportions (`None` = split by free memory).
    /// Must hold at least one entry per effective device: per listed
    /// device for explicit [`ModelOptions::devices`], per registry
    /// device under default selection (a safe over-approximation, since
    /// the default selection is a subset of the registry).
    pub tensor_split: Option<Vec<f32>>,
    /// Devices to offload to, resolved by index at load time (`None` =
    /// upstream default selection).
    pub devices: Option<Vec<DeviceInfo>>,
    /// How the model file is mapped into memory.
    pub load_mode: ModelLoadMode,
}

/// An owned libllama model (`.gguf` weights + metadata).
///
/// Cheap to clone (shares the native model); `!Send + !Sync` like all
/// ForgeCore handles.
#[derive(Clone)]
pub struct Model {
    inner: Rc<ModelInner>,
}

impl Model {
    /// Load a `.gguf` model from `path` onto the CPU with default options.
    ///
    /// Returns [`Error`] (no abort) when the file is missing, unreadable,
    /// or not a valid model.
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_options(path, &ModelOptions::default())
    }

    /// Load a `.gguf` model from `path` with explicit [`ModelOptions`].
    ///
    /// See [`ModelOptions`] for the offload model and failure behavior.
    /// The default options load exactly as before: CPU-only with
    /// upstream's default split mode, load mode, and device selection.
    pub fn load_with_options(path: &Path, options: &ModelOptions) -> Result<Self> {
        device::ensure_registry();
        let path = path
            .to_str()
            .ok_or_else(|| Error::invalid("model path is not UTF-8"))?;
        let path_c = CString::new(path)
            .map_err(|_| Error::invalid("model path contains an interior NUL"))?;
        let registry_count = device::device_count();
        let n_gpu_layers = options.gpu_layers.as_native()?;
        if let Some(split) = options.tensor_split.as_deref() {
            validate_tensor_split_values(split)?;
        }
        // Resolve an explicit device list to NULL-terminated
        // registry-borrowed handles. This runs even for CPU loads: an
        // out-of-range index must never reach upstream as a handle.
        let mut device_handles: Option<Vec<forge_sys::ggml_backend_dev_t>> = match &options.devices
        {
            None => None,
            Some(list) => {
                if list.is_empty() {
                    return Err(Error::invalid(
                        "device list must not be empty; use None for default device selection",
                    ));
                }
                let mut handles = Vec::with_capacity(list.len() + 1);
                for info in list {
                    if info.index >= registry_count {
                        return Err(Error::invalid(format!(
                            "stale device index {} (registry holds {})",
                            info.index, registry_count
                        )));
                    }
                    // SAFETY: index is bounds-checked, so the handle is
                    // valid for the registry's (process-wide) lifetime.
                    handles.push(unsafe { forge_sys::ggml_backend_dev_get(info.index) });
                }
                handles.push(ptr::null_mut());
                Some(handles)
            }
        };
        // Effective device count for the checks below: exact for an
        // explicit list, a safe over-approximation (the whole registry)
        // under default selection.
        let effective_devices = match &device_handles {
            Some(handles) => handles.len() - 1,
            None => registry_count,
        };
        if let Some(split) = options.tensor_split.as_deref() {
            if split.len() < effective_devices {
                return Err(Error::invalid(format!(
                    "tensor_split has {} entries for {} devices",
                    split.len(),
                    effective_devices
                )));
            }
        }
        // Upstream only reads main_gpu under SPLIT_MODE_NONE, so only
        // then is it bounds-checked: anything else must load exactly as
        // upstream would (ignoring the field).
        if options.split_mode == SplitMode::None && options.main_gpu >= effective_devices {
            return Err(Error::invalid(format!(
                "main GPU index {} out of range ({} devices)",
                options.main_gpu, effective_devices
            )));
        }
        let main_gpu = i32::try_from(options.main_gpu).map_err(|_| {
            Error::invalid(format!("main GPU index {} too large", options.main_gpu))
        })?;
        // No silent CPU fallback: with default device selection and no
        // GPU device present, upstream would load successfully and run
        // on the CPU. Refuse instead. An explicit device list is the
        // caller's own selection and passes through verbatim.
        if n_gpu_layers != 0 && device_handles.is_none() && !device::supports_gpu_offload() {
            let what = match options.gpu_layers {
                GpuLayers::All => "all layers".to_string(),
                GpuLayers::Count(n) => format!("{n} layers"),
                GpuLayers::Cpu => unreachable!("n_gpu_layers != 0 implies offload was requested"),
            };
            return Err(Error::unsupported(format!(
                "GPU offload of {what} requested but no GPU device is available \
                 (llama_supports_gpu_offload is false)"
            )));
        }
        // SAFETY: default params are valid by construction; the path
        // pointer is a live NUL-terminated string; device handles are
        // bounds-checked and NULL-terminated; the split slice outlives
        // the call and holds at least one entry per effective device
        // (upstream copies it during load); NULL return (load failure)
        // is checked.
        unsafe {
            let mut params = forge_sys::llama_model_default_params();
            params.n_gpu_layers = n_gpu_layers;
            params.split_mode = options.split_mode.as_native();
            params.load_mode = options.load_mode.as_native();
            params.main_gpu = main_gpu;
            params.check_tensors = options.check_tensors;
            params.tensor_split = options
                .tensor_split
                .as_deref()
                .map_or(ptr::null(), <[f32]>::as_ptr);
            params.devices = device_handles
                .as_mut()
                .map_or(ptr::null_mut(), |handles| handles.as_mut_ptr());
            let raw = forge_sys::llama_model_load_from_file(path_c.as_ptr(), params);
            if raw.is_null() {
                return Err(Error::model(format!("failed to load model from {path}")));
            }
            Ok(Self {
                inner: Rc::new(ModelInner { raw }),
            })
        }
    }

    /// Total parameter count.
    pub fn n_params(&self) -> u64 {
        // SAFETY: raw is a live model; pure getter.
        unsafe { forge_sys::llama_model_n_params(self.inner.raw) }
    }

    /// Vocabulary size (token count).
    pub fn vocab_size(&self) -> Result<u32> {
        // SAFETY: raw is a live model; NULL vocab is checked.
        unsafe {
            let vocab = forge_sys::llama_model_get_vocab(self.inner.raw);
            if vocab.is_null() {
                return Err(Error::model("model has no vocabulary"));
            }
            let n = forge_sys::llama_vocab_n_tokens(vocab);
            u32_from_upstream("vocabulary size", n)
        }
    }

    /// Human-readable model description (architecture, type, file type).
    ///
    /// Owned copy; upstream's buffer is never exposed.
    pub fn description(&self) -> Result<String> {
        let mut buf = vec![0u8; 256];
        loop {
            // SAFETY: buf points to buf.len() live bytes; upstream
            // snprintf semantics return the would-be length.
            let ret = unsafe {
                forge_sys::llama_model_desc(
                    self.inner.raw,
                    buf.as_mut_ptr().cast::<c_char>(),
                    buf.len(),
                )
            };
            if ret < 0 {
                return Err(Error::model("model description failed"));
            }
            let need = ret as usize + 1;
            if need <= buf.len() {
                buf.truncate(ret as usize);
                return String::from_utf8(buf)
                    .map_err(|_| Error::model("model description is not UTF-8"));
            }
            if need > 1 << 20 {
                return Err(Error::model("model description too large"));
            }
            buf.resize(need, 0);
        }
    }

    /// Model weight size in bytes, as accounted by upstream.
    pub fn size_bytes(&self) -> u64 {
        // SAFETY: raw is a live model; pure getter.
        unsafe { forge_sys::llama_model_size(self.inner.raw) }
    }

    /// Training context length the model was built with.
    pub fn n_ctx_train(&self) -> Result<u32> {
        // SAFETY: raw is a live model; pure getter.
        let n = unsafe { forge_sys::llama_model_n_ctx_train(self.inner.raw) };
        u32_from_upstream("n_ctx_train", n)
    }

    /// Embedding width.
    pub fn n_embd(&self) -> Result<u32> {
        // SAFETY: raw is a live model; pure getter.
        let n = unsafe { forge_sys::llama_model_n_embd(self.inner.raw) };
        u32_from_upstream("n_embd", n)
    }

    /// Input embedding width: one row of floats per token in an
    /// embedding batch (see [`BatchBuilder`](crate::batch::BatchBuilder)).
    pub fn n_embd_inp(&self) -> Result<u32> {
        // SAFETY: raw is a live model; pure getter.
        let n = unsafe { forge_sys::llama_model_n_embd_inp(self.inner.raw) };
        u32_from_upstream("n_embd_inp", n)
    }

    /// Output embedding width: one row of floats per
    /// [`Embeddings`](crate::context::Embeddings) output.
    pub fn n_embd_out(&self) -> Result<u32> {
        // SAFETY: raw is a live model; pure getter.
        let n = unsafe { forge_sys::llama_model_n_embd_out(self.inner.raw) };
        u32_from_upstream("n_embd_out", n)
    }

    /// Classifier head width: floats per sequence under RANK pooling.
    pub fn n_cls_out(&self) -> u32 {
        // SAFETY: raw is a live model; pure infallible getter.
        unsafe { forge_sys::llama_model_n_cls_out(self.inner.raw) }
    }

    /// Whether the model has an encoder (used to resolve the effective
    /// causal-attention state, which upstream exposes no getter for).
    pub fn has_encoder(&self) -> bool {
        // SAFETY: raw is a live model; pure predicate.
        unsafe { forge_sys::llama_model_has_encoder(self.inner.raw) }
    }

    /// Number of transformer layers.
    pub fn n_layer(&self) -> Result<u32> {
        // SAFETY: raw is a live model; pure getter.
        let n = unsafe { forge_sys::llama_model_n_layer(self.inner.raw) };
        u32_from_upstream("n_layer", n)
    }

    /// Number of attention heads.
    pub fn n_head(&self) -> Result<u32> {
        // SAFETY: raw is a live model; pure getter.
        let n = unsafe { forge_sys::llama_model_n_head(self.inner.raw) };
        u32_from_upstream("n_head", n)
    }

    /// Number of key/value attention heads.
    pub fn n_head_kv(&self) -> Result<u32> {
        // SAFETY: raw is a live model; pure getter.
        let n = unsafe { forge_sys::llama_model_n_head_kv(self.inner.raw) };
        u32_from_upstream("n_head_kv", n)
    }

    pub(crate) fn raw(&self) -> *mut forge_sys::llama_model {
        self.inner.raw
    }

    pub(crate) fn inner(&self) -> &Rc<ModelInner> {
        &self.inner
    }
}

/// Convert a non-negative upstream `int32` dimension, rejecting negatives.
pub(crate) fn u32_from_upstream(name: &str, value: c_int) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::model(format!("{name} is negative: {value}")))
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("n_params", &self.n_params())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_prompt_ids_are_stable() {
        assert_eq!(QWEN25_DIAGNOSTIC_PROMPT, "hi, what's 2+2=?");
        assert_eq!(QWEN25_DIAGNOSTIC_PROMPT_IDS.len(), 9);
    }

    #[test]
    fn model_options_default_is_cpu_plain() {
        let options = ModelOptions::default();
        assert!(!options.check_tensors);
        assert_eq!(options.gpu_layers, GpuLayers::Cpu);
        // Native defaults (verified against the pinned headers), *not*
        // zero: defaults must pass upstream exactly what plain loads
        // always have.
        assert_eq!(options.split_mode, SplitMode::Layer);
        assert_eq!(options.load_mode, ModelLoadMode::Auto);
        assert_eq!(options.main_gpu, 0);
        assert!(options.tensor_split.is_none());
        assert!(options.devices.is_none());
    }

    #[test]
    fn gpu_layers_map_to_native() {
        assert_eq!(GpuLayers::Cpu.as_native().unwrap(), 0);
        assert_eq!(GpuLayers::All.as_native().unwrap(), -1);
        assert_eq!(GpuLayers::Count(0).as_native().unwrap(), 0);
        assert_eq!(GpuLayers::Count(12).as_native().unwrap(), 12);
        assert!(GpuLayers::Count(u32::MAX).as_native().is_err());
    }

    #[test]
    fn split_mode_maps_to_native() {
        assert_eq!(SplitMode::None.as_native(), 0);
        assert_eq!(SplitMode::Layer.as_native(), 1);
        assert_eq!(SplitMode::Row.as_native(), 2);
        assert_eq!(SplitMode::Tensor.as_native(), 3);
    }

    #[test]
    fn load_mode_maps_to_native() {
        assert_eq!(ModelLoadMode::Auto.as_native(), -1);
        assert_eq!(ModelLoadMode::NoMmap.as_native(), 0);
        assert_eq!(ModelLoadMode::Mmap.as_native(), 1);
        assert_eq!(ModelLoadMode::Mlock.as_native(), 2);
        assert_eq!(ModelLoadMode::MmapMlock.as_native(), 3);
    }

    #[test]
    fn tensor_split_values_are_validated() {
        assert!(validate_tensor_split_values(&[]).is_err());
        assert!(validate_tensor_split_values(&[0.5, 0.5]).is_ok());
        assert!(validate_tensor_split_values(&[0.0, 0.0]).is_ok());
        assert!(validate_tensor_split_values(&[-1.0]).is_err());
        assert!(validate_tensor_split_values(&[f32::NAN]).is_err());
        assert!(validate_tensor_split_values(&[f32::INFINITY]).is_err());
    }
}
