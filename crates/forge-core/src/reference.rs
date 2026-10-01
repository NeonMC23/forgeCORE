//! Frozen scalar validation oracles.
//!
//! The modules below are the surviving scalar Rust code from before the
//! ggml pivot. They are **not** an inference engine and are **not** on any
//! execution path: all real compute goes through ggml (`crate::runtime`).
//!
//! Their only job is independent cross-validation:
//!
//! - `quant` decodes quantized blocks (Q4_0/Q8_0/Q6_K/...) with plain
//!   scalar Rust so ggml numerics can be checked without trusting ggml.
//! - `ops` implements tiny `matmul`/`add`/RoPE/RMS-norm kernels used as
//!   expected-value oracles in tests.
//! - `convert` holds the exact F16/BF16 bit conversions the oracles need.
//! - `shape` holds the row-major layout helpers the oracles are written
//!   against.
//! - `checkpoint` keeps the legacy tiny-checkpoint format readable so old
//!   fixtures stay loadable.
//!
//! Rules: oracle code must stay dependency-free scalar Rust, must never
//! call into ggml, and must never be used for real execution. New quants
//! or kernels belong upstream in ggml, not here.

pub mod checkpoint;
pub mod convert;
pub mod ops;
pub mod quant;
pub mod shape;
