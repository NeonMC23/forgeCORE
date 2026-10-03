//! ForgeCore: thin Rust abstraction over llama.cpp/ggml.
//!
//! ForgeCore is the middle of `RAMforge → ForgeCore → llama.cpp/ggml →
//! hardware`. It owns no compute of its own: [`Backend`] opens ggml
//! backends, [`Tensor`] owns ggml tensors, [`runtime`] executes ggml
//! graphs, [`Model`] loads `.gguf` models through libllama,
//! [`Tokenizer`] encodes text through the model's vocabulary, and
//! [`Context`] runs the minimal CPU inference path (`Model` → `Context`
//! → [`Batch`] → `decode` → [`Logits`]). The only `unsafe` in the crate
//! sits at the documented FFI boundary inside each module; no raw C
//! pointers appear in any public API.
//!
//! The pre-pivot scalar kernels survive frozen under [`reference`] as
//! validation oracles — they cross-check ggml numerics in tests and are
//! never on an execution path.

pub mod backend;
pub mod batch;
pub mod context;
pub mod device;
pub mod dtype;
pub mod error;
pub mod model;
pub mod reference;
pub mod runtime;
pub mod tensor;
pub mod tokenizer;

pub use backend::Backend;
pub use batch::{Batch, BatchBuilder, TokenId};
pub use context::{Context, ContextOptions, Logits};
pub use device::{enumerate_devices, DeviceInfo, DeviceType};
pub use dtype::DType;
pub use error::{Error, Result};
pub use model::{Model, ModelOptions};
pub use runtime::{add, matmul};
pub use tensor::Tensor;
pub use tokenizer::{DecodeOptions, EncodeOptions, SpecialTokens, TokenAttr, Tokenizer, VocabType};
