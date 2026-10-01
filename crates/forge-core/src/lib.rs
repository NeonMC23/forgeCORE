//! ForgeCore: thin Rust abstraction over llama.cpp/ggml.
//!
//! ForgeCore is the middle of `RAMforge → ForgeCore → llama.cpp/ggml →
//! hardware`. It owns no compute of its own: [`Backend`] opens ggml
//! backends, [`Tensor`] owns ggml tensors, [`runtime`] executes ggml
//! graphs, and [`Model`] loads `.gguf` models through libllama. The only
//! `unsafe` in the crate sits at the documented FFI boundary inside each
//! module; no raw C pointers appear in any public API.
//!
//! The pre-pivot scalar kernels survive frozen under [`reference`] as
//! validation oracles — they cross-check ggml numerics in tests and are
//! never on an execution path.

pub mod backend;
pub mod device;
pub mod dtype;
pub mod error;
pub mod model;
pub mod reference;
pub mod runtime;
pub mod tensor;

pub use backend::Backend;
pub use device::{enumerate_devices, DeviceInfo, DeviceType};
pub use dtype::DType;
pub use error::{Error, Result};
pub use model::Model;
pub use runtime::{add, matmul};
pub use tensor::Tensor;
