//! Model loading and metadata via libllama.
//!
//! [`Model`] owns one `llama_model` (weights + metadata for a `.gguf`
//! file) and frees it on drop. This milestone covers loading and shape
//! metadata only — no contexts, batching, or generation (see
//! `docs/PIVOT.md` for the roadmap). Loads default to CPU (`n_gpu_layers
//! = 0`); GPU offload reuses the same params struct later.

use crate::device;
use crate::error::{Error, Result};
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr;

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

/// An owned libllama model (`.gguf` weights + metadata).
pub struct Model {
    raw: *mut forge_sys::llama_model,
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: raw came from a successful load, is freed exactly once.
        unsafe { forge_sys::llama_model_free(self.raw) };
    }
}

impl Model {
    /// Load a `.gguf` model from `path` onto the CPU.
    ///
    /// Returns [`Error`] (no abort) when the file is missing, unreadable,
    /// or not a valid model.
    pub fn load(path: &Path) -> Result<Self> {
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
            let raw = forge_sys::llama_model_load_from_file(path_c.as_ptr(), params);
            if raw.is_null() {
                return Err(Error::backend(format!("failed to load model from {path}")));
            }
            Ok(Self { raw })
        }
    }

    /// Total parameter count.
    pub fn n_params(&self) -> u64 {
        // SAFETY: raw is a live model; pure getter.
        unsafe { forge_sys::llama_model_n_params(self.raw) }
    }

    /// Vocabulary size (token count).
    pub fn vocab_size(&self) -> Result<u32> {
        // SAFETY: raw is a live model; NULL vocab is checked.
        unsafe {
            let vocab = forge_sys::llama_model_get_vocab(self.raw);
            if vocab.is_null() {
                return Err(Error::backend("model has no vocabulary"));
            }
            let n = forge_sys::llama_vocab_n_tokens(vocab);
            u32::try_from(n).map_err(|_| Error::backend("negative vocabulary size"))
        }
    }
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
}
