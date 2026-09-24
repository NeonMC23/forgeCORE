//! Quantization interface and scalar reference decoders.
//!
//! ## Scope of this phase
//!
//! The F32 reference path is authoritative. This module additionally fixes
//! the quantization **interface** every format will implement:
//!
//! 1. exact block geometry ([`block_spec`]);
//! 2. scalar row dequantization ([`dequantize_row`]);
//! 3. reference dot/matvec over dequantized rows ([`matvec_reference`]);
//! 4. deterministic block-level tests (below and in `tests/quant_contract.rs`).
//!
//! Formats without a reviewed scalar decoder return an explicit
//! "unsupported" error; they are never silently approximated. Each future
//! format must arrive with its own block tests before any optimized kernel
//! may target it.
//!
//! ## Implemented formats
//!
//! * [`QuantFormat::F16`] / [`QuantFormat::BF16`]: 2 bytes per value,
//!   decoded through [`crate::dtype`].
//! * [`QuantFormat::Q4_0`]: 32 values per 18-byte block: little-endian F16
//!   scale `d`, then 16 bytes of packed 4-bit nibbles (low nibble first);
//!   `value = (nibble - 8) * d`.
//! * [`QuantFormat::Q8_0`]: 32 values per 34-byte block: little-endian F16
//!   scale `d`, then 32 signed int8 values; `value = q * d`.
//!
//! ## Row geometry
//!
//! A quantized matrix with [`MatrixShape`](crate::shape::MatrixShape)
//! `[input, output]` stores `output` consecutive rows of
//! [`row_bytes`] bytes; row `o` starts at byte `o * row_bytes`. The number
//! of blocks per row is `input / values_per_block`, so `input` must be an
//! exact multiple of the block width.

use crate::dtype::{bf16_bits_to_f32, f16_bits_to_f32};
use crate::error::{Error, Result};
use crate::shape::MatrixShape;

/// Quantized or narrow storage formats in ForgeCore's target set.
///
/// Variant names intentionally mirror the GGML format spellings (`Q4_0`,
/// `Q6_K`, ...) instead of UpperCamelCase so code, errors, and GGUF
/// tooling all name the same format the same way.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// 4-bit, 32 values per block.
    Q4_0,
    /// 4-bit K-quant, 256 values per super-block.
    Q4_K,
    /// 5-bit, 32 values per block.
    Q5_0,
    /// 5-bit K-quant, 256 values per super-block.
    Q5_K,
    /// 6-bit K-quant, 256 values per super-block.
    Q6_K,
    /// 8-bit, 32 values per block.
    Q8_0,
    /// 8-bit K-quant, 256 values per super-block.
    Q8_K,
    /// 16-bit IEEE float.
    F16,
    /// 16-bit brain float.
    BF16,
}

impl QuantFormat {
    /// Short stable format name used in diagnostics and errors.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4_K => "Q4_K",
            Self::Q5_0 => "Q5_0",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_0 => "Q8_0",
            Self::Q8_K => "Q8_K",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
        }
    }
}

/// Exact storage geometry of one block of a format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSpec {
    /// Values decoded from one block.
    pub values_per_block: usize,
    /// Encoded bytes per block.
    pub bytes_per_block: usize,
}

/// Block geometry for a format.
///
/// The K-quant geometries come from the RAMforge source-level format table
/// (verified in-repo values). The `Q5_0` geometry is recorded from the GGML
/// block layout (2-byte scale, 4-byte high bits, 16-byte nibbles) and must
/// be re-verified against GGML headers when its decoder is implemented.
pub const fn block_spec(format: QuantFormat) -> BlockSpec {
    match format {
        QuantFormat::Q4_0 => BlockSpec {
            values_per_block: 32,
            bytes_per_block: 18,
        },
        QuantFormat::Q8_0 => BlockSpec {
            values_per_block: 32,
            bytes_per_block: 34,
        },
        QuantFormat::Q4_K => BlockSpec {
            values_per_block: 256,
            bytes_per_block: 144,
        },
        QuantFormat::Q5_K => BlockSpec {
            values_per_block: 256,
            bytes_per_block: 176,
        },
        QuantFormat::Q6_K => BlockSpec {
            values_per_block: 256,
            bytes_per_block: 210,
        },
        QuantFormat::Q8_K => BlockSpec {
            values_per_block: 256,
            bytes_per_block: 292,
        },
        QuantFormat::Q5_0 => BlockSpec {
            values_per_block: 32,
            bytes_per_block: 22,
        },
        QuantFormat::F16 | QuantFormat::BF16 => BlockSpec {
            values_per_block: 1,
            bytes_per_block: 2,
        },
    }
}

/// Encoded bytes of one matrix row of width `input_dim`.
///
/// `input_dim` must be an exact multiple of the format's block width.
pub fn row_bytes(format: QuantFormat, input_dim: usize) -> Result<usize> {
    let spec = block_spec(format);
    if input_dim == 0 || !input_dim.is_multiple_of(spec.values_per_block) {
        return Err(Error(format!(
            "{} input dimension {input_dim} is not a positive multiple of block width {}",
            format.name(),
            spec.values_per_block
        )));
    }
    (input_dim / spec.values_per_block)
        .checked_mul(spec.bytes_per_block)
        .ok_or_else(|| {
            Error(format!(
                "{} row size overflows for input {input_dim}",
                format.name()
            ))
        })
}

/// Scalar reference dequantization of exactly one matrix row.
///
/// `bytes` must hold exactly [`row_bytes`] bytes and `out` exactly
/// `input_dim` values. Formats without a reviewed decoder return an
/// explicit unsupported error.
pub fn dequantize_row(
    format: QuantFormat,
    bytes: &[u8],
    input_dim: usize,
    out: &mut [f32],
) -> Result<()> {
    let expected_bytes = row_bytes(format, input_dim)?;
    if bytes.len() != expected_bytes {
        return Err(Error(format!(
            "{} row byte mismatch: expected {expected_bytes}, got {}",
            format.name(),
            bytes.len()
        )));
    }
    if out.len() != input_dim {
        return Err(Error(format!(
            "{} row output mismatch: expected {input_dim}, got {}",
            format.name(),
            out.len()
        )));
    }
    match format {
        QuantFormat::Q4_0 => {
            dequantize_row_q4_0(bytes, out);
            Ok(())
        }
        QuantFormat::Q8_0 => {
            dequantize_row_q8_0(bytes, out);
            Ok(())
        }
        QuantFormat::F16 => {
            dequantize_row_f16(bytes, out);
            Ok(())
        }
        QuantFormat::BF16 => {
            dequantize_row_bf16(bytes, out);
            Ok(())
        }
        unsupported => Err(Error(format!(
            "{} has no reference decoder yet; refusing to approximate",
            unsupported.name()
        ))),
    }
}

/// Scalar reference matvec over a quantized matrix.
///
/// Each output row is dequantized by [`dequantize_row`] and dotted with `x`
/// in F32 in input order. `raw` must hold exactly
/// `row_bytes * shape.output` bytes.
pub fn matvec_reference(
    format: QuantFormat,
    raw: &[u8],
    shape: MatrixShape,
    x: &[f32],
    y: &mut [f32],
) -> Result<()> {
    if x.len() != shape.input || y.len() != shape.output {
        return Err(Error(format!(
            "{} matvec shape mismatch: [{}, {}] with x {}, y {}",
            format.name(),
            shape.input,
            shape.output,
            x.len(),
            y.len()
        )));
    }
    let stride = row_bytes(format, shape.input)?;
    let expected = stride
        .checked_mul(shape.output)
        .ok_or_else(|| Error(format!("{} weight size overflows", format.name())))?;
    if raw.len() != expected {
        return Err(Error(format!(
            "{} weight buffer mismatch: expected {expected} bytes, got {}",
            format.name(),
            raw.len()
        )));
    }
    let mut decoded = vec![0.0f32; shape.input];
    for (row_index, output) in y.iter_mut().enumerate() {
        let row = &raw[row_index * stride..(row_index + 1) * stride];
        dequantize_row(format, row, shape.input, &mut decoded)?;
        let mut sum = 0.0f32;
        for index in 0..shape.input {
            sum += decoded[index] * x[index];
        }
        *output = sum;
    }
    Ok(())
}

/// Encoded bytes and decoded values per block for the scalar decoders.
/// These mirror [`block_spec`] for the implemented formats; the `const`
/// assertions below keep them in lockstep.
const Q4_0_BLOCK_BYTES: usize = 18;
const Q4_0_BLOCK_VALUES: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCK_VALUES: usize = 32;

const _: () = {
    assert!(block_spec(QuantFormat::Q4_0).bytes_per_block == Q4_0_BLOCK_BYTES);
    assert!(block_spec(QuantFormat::Q4_0).values_per_block == Q4_0_BLOCK_VALUES);
    assert!(block_spec(QuantFormat::Q8_0).bytes_per_block == Q8_0_BLOCK_BYTES);
    assert!(block_spec(QuantFormat::Q8_0).values_per_block == Q8_0_BLOCK_VALUES);
};

fn dequantize_row_q4_0(bytes: &[u8], out: &mut [f32]) {
    let (blocks, byte_rest) = bytes.as_chunks::<Q4_0_BLOCK_BYTES>();
    let (slots, slot_rest) = out.as_chunks_mut::<Q4_0_BLOCK_VALUES>();
    debug_assert!(byte_rest.is_empty() && slot_rest.is_empty());
    for (block, values) in blocks.iter().zip(slots.iter_mut()) {
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for index in 0..Q4_0_BLOCK_VALUES {
            let packed = block[2 + index / 2];
            let nibble = (packed >> ((index % 2) * 4)) & 0x0F;
            values[index] = f32::from(nibble as i8 - 8) * scale;
        }
    }
}

fn dequantize_row_q8_0(bytes: &[u8], out: &mut [f32]) {
    let (blocks, byte_rest) = bytes.as_chunks::<Q8_0_BLOCK_BYTES>();
    let (slots, slot_rest) = out.as_chunks_mut::<Q8_0_BLOCK_VALUES>();
    debug_assert!(byte_rest.is_empty() && slot_rest.is_empty());
    for (block, values) in blocks.iter().zip(slots.iter_mut()) {
        let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (index, value) in values.iter_mut().enumerate() {
            *value = f32::from(block[2 + index] as i8) * scale;
        }
    }
}

fn dequantize_row_f16(bytes: &[u8], out: &mut [f32]) {
    let (pairs, rest) = bytes.as_chunks::<2>();
    debug_assert!(rest.is_empty());
    for (pair, value) in pairs.iter().zip(out.iter_mut()) {
        *value = f16_bits_to_f32(u16::from_le_bytes(*pair));
    }
}

fn dequantize_row_bf16(bytes: &[u8], out: &mut [f32]) {
    let (pairs, rest) = bytes.as_chunks::<2>();
    debug_assert!(rest.is_empty());
    for (pair, value) in pairs.iter().zip(out.iter_mut()) {
        *value = bf16_bits_to_f32(u16::from_le_bytes(*pair));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_0_block_hand_decoded() {
        // Scale 1.0, first packed byte 0x08: values[0] = 8-8 = 0,
        // values[1] = 0-8 = -8; remaining nibbles 0x9 -> 9-8 = 1.
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        block[2] = 0x08;
        for byte in block.iter_mut().skip(3) {
            *byte = 0x99;
        }
        let mut out = [0.0f32; 32];
        dequantize_row(QuantFormat::Q4_0, &block, 32, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], -8.0);
        for value in out.iter().skip(2) {
            assert_eq!(*value, 1.0);
        }
    }

    #[test]
    fn q4_0_scale_applies() {
        // Scale 2.0 (0x4000), all nibbles 0xA -> (10-8)*2 = 4.
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&0x4000u16.to_le_bytes());
        for byte in block.iter_mut().skip(2) {
            *byte = 0xAA;
        }
        let mut out = [0.0f32; 32];
        dequantize_row(QuantFormat::Q4_0, &block, 32, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 4.0));
    }

    #[test]
    fn q8_0_block_hand_decoded() {
        // Scale 1.0; int8 payload echoed as floats.
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        block[2] = 0;
        block[3] = 1;
        block[4] = 0xFF; // -1
        block[5] = 0x7F; // 127
        block[6] = 0x80; // -128
        let mut out = [0.0f32; 32];
        dequantize_row(QuantFormat::Q8_0, &block, 32, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], -1.0);
        assert_eq!(out[3], 127.0);
        assert_eq!(out[4], -128.0);
    }

    #[test]
    fn f16_bf16_rows_decoded() {
        let mut out = [0.0f32; 2];
        dequantize_row(
            QuantFormat::F16,
            &0x3C00u16.to_le_bytes().repeat(2),
            2,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [1.0, 1.0]);
        let bf16_two = 0x4000u16.to_le_bytes();
        dequantize_row(QuantFormat::BF16, &bf16_two, 1, &mut out[..1]).unwrap();
        assert_eq!(out[0], 2.0);
    }

    #[test]
    fn q4_0_matvec_hand_computed() {
        // One row of all ones (scale 1.0, nibbles 9) dotted with 1..=32.
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        for byte in block.iter_mut().skip(2) {
            *byte = 0x99;
        }
        let x: Vec<f32> = (1..=32).map(|v| v as f32).collect();
        let mut y = [0.0f32; 1];
        matvec_reference(
            QuantFormat::Q4_0,
            &block,
            MatrixShape::new(32, 1),
            &x,
            &mut y,
        )
        .unwrap();
        assert_eq!(y[0], 528.0);
    }

    #[test]
    fn unimplemented_formats_refuse_explicitly() {
        for format in [
            QuantFormat::Q4_K,
            QuantFormat::Q5_0,
            QuantFormat::Q5_K,
            QuantFormat::Q6_K,
            QuantFormat::Q8_K,
        ] {
            let bytes = vec![0u8; row_bytes(format, block_spec(format).values_per_block).unwrap()];
            let mut out = vec![0.0f32; block_spec(format).values_per_block];
            let error = dequantize_row(
                format,
                &bytes,
                block_spec(format).values_per_block,
                &mut out,
            )
            .unwrap_err();
            assert!(
                error.0.contains(format.name()),
                "error must name the format: {error}"
            );
        }
    }

    #[test]
    fn row_geometry_validates_block_alignment() {
        assert_eq!(row_bytes(QuantFormat::Q4_0, 64).unwrap(), 36);
        assert_eq!(row_bytes(QuantFormat::Q8_0, 32).unwrap(), 34);
        assert!(row_bytes(QuantFormat::Q4_0, 33).is_err());
        assert!(row_bytes(QuantFormat::Q6_K, 0).is_err());
    }
}
