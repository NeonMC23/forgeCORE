//! ForgeCore: a small, explicit, scalar numerical inference core.
//!
//! ForgeCore rebuilds the transformer inference mathematics from first
//! principles. Correctness comes before performance: every operator here is a
//! deliberately boring scalar reference implementation with no SIMD, no GPU,
//! no threading, and no hidden layout reinterpretation.
//!
//! ## Module layout
//!
//! * [`error`] — the single fallible-result type used by every kernel.
//! * [`shape`] — tensor/matrix shape contracts and the authoritative matrix
//!   convention (`y[o] = sum_i W[o, i] * x[i]`).
//! * [`dtype`] — scalar F16/BF16 bit-pattern conversion.
//! * [`ops`] — scalar reference operators: dot, matvec, RMSNorm,
//!   elementwise add/mul, SiLU, SwiGLU, softmax, RoPE.
//! * [`attention`] — explicit single-token causal attention with GQA mapping.
//! * [`kv`] — explicit `[position][kv_head][head_dim]` KV cache.
//! * [`model`] — reference transformer-layer and token-forward executor plus
//!   model-dimension contracts.
//! * [`checkpoint`] — deterministic numerical summaries for validation.
//! * [`quant`] — quantization interface: block geometry plus scalar
//!   dequantization for the formats implemented so far.

pub mod attention;
pub mod checkpoint;
pub mod dtype;
pub mod error;
pub mod kv;
pub mod model;
pub mod ops;
pub mod quant;
pub mod shape;

pub use error::{Error, Result};
