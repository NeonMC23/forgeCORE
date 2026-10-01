//! Model loading and metadata via libllama.
//!
//! [`Model`] owns one `llama_model` (weights + metadata for a `.gguf`
//! file) through a shared handle: cloning a `Model` shares the native
//! model (as with [`Backend`](crate::backend::Backend)), and every
//! [`Context`](crate::context::Context) holds a share so the model
//! always outlives its contexts. Loads default to CPU (`n_gpu_layers
//! = 0`); GPU offload extends [`ModelOptions`] later.

use crate::device;
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

/// Model loading options. `#[non_exhaustive]` so device/offload options
/// (`n_gpu_layers`, split modes, ...) can be added later without
/// breaking callers; only CPU-relevant options exist in this phase.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct ModelOptions {
    /// Validate model tensor data while loading (slower, safer).
    pub check_tensors: bool,
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
    pub fn load_with_options(path: &Path, options: &ModelOptions) -> Result<Self> {
        device::ensure_registry();
        let path = path
            .to_str()
            .ok_or_else(|| Error::invalid("model path is not UTF-8"))?;
        let path_c = CString::new(path)
            .map_err(|_| Error::invalid("model path contains an interior NUL"))?;
        // SAFETY: default params are valid by construction; the path
        // pointer is a live NUL-terminated string; NULL return (load
        // failure) is checked.
        unsafe {
            let mut params = forge_sys::llama_model_default_params();
            params.n_gpu_layers = 0;
            params.check_tensors = options.check_tensors;
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
fn u32_from_upstream(name: &str, value: c_int) -> Result<u32> {
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
        assert!(!ModelOptions::default().check_tensors);
    }
}
