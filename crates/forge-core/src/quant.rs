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
//! * [`QuantFormat::Q6_K`]: 256 values per 210-byte block: 128 bytes of
//!   low 4-bit nibbles, 64 bytes of high 2-bit groups, 16 signed int8
//!   scales, then the little-endian F16 super-block scale `d`;
//!   `value = d * scale * (six_bit_quant - 32)`. The byte map is
//!   documented on [`dequantize_block_q6_k`].
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
/// The K-quant geometries were cross-checked against the historical RAMforge
/// source-level format table during the initial audit (documentation
/// provenance only, not a code dependency). The `Q5_0` geometry is recorded from the GGML
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
        QuantFormat::Q6_K => {
            dequantize_row_q6_k(bytes, out);
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
const Q6_K_BLOCK_BYTES: usize = 210;
const Q6_K_BLOCK_VALUES: usize = 256;

const _: () = {
    assert!(block_spec(QuantFormat::Q4_0).bytes_per_block == Q4_0_BLOCK_BYTES);
    assert!(block_spec(QuantFormat::Q4_0).values_per_block == Q4_0_BLOCK_VALUES);
    assert!(block_spec(QuantFormat::Q8_0).bytes_per_block == Q8_0_BLOCK_BYTES);
    assert!(block_spec(QuantFormat::Q8_0).values_per_block == Q8_0_BLOCK_VALUES);
    assert!(block_spec(QuantFormat::Q6_K).bytes_per_block == Q6_K_BLOCK_BYTES);
    assert!(block_spec(QuantFormat::Q6_K).values_per_block == Q6_K_BLOCK_VALUES);
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

/// Scalar reference decode of exactly one Q6_K block into 256 F32 values.
///
/// ## GGML Q6_K byte layout (210 bytes)
///
/// Audited against GGML `block_q6_K` (`ggml-common.h`) and
/// `dequantize_row_q6_K` (`ggml-quants.c`); the loop below mirrors that
/// function branch for branch.
///
/// ```text
/// bytes 0..128    ql:     low 4 bits of each quant (two quants per byte)
/// bytes 128..192  qh:     high 2 bits of each quant (four quants per byte)
/// bytes 192..208  scales: 16 signed int8 per-group scales
/// bytes 208..210  d:      little-endian FP16 super-block scale
/// ```
///
/// ## Reconstruction
///
/// The block holds two 128-value halves. Half `h` uses
/// `ql[h*64 .. h*64+64]`, `qh[h*32 .. h*32+32]`, and
/// `scales[h*8 .. h*8+8]`. Within a half, for `l` in `0..32` with group
/// selector `is = l / 16`, one `qh[l]` byte supplies the high 2 bits of
/// four quants at bit shifts 0, 2, 4, 6:
///
/// ```text
/// q1 = (ql[l] low nibble)    | (qh[l] bits[1:0] << 4) - 32  -> y[l +  0], scale sc[is+0]
/// q2 = (ql[l+32] low nibble) | (qh[l] bits[3:2] << 4) - 32  -> y[l + 32], scale sc[is+2]
/// q3 = (ql[l] high nibble)   | (qh[l] bits[5:4] << 4) - 32  -> y[l + 64], scale sc[is+4]
/// q4 = (ql[l+32] high nibble)| (qh[l] bits[7:6] << 4) - 32  -> y[l + 96], scale sc[is+6]
/// ```
///
/// Each output is `d * scale * (six_bit_quant - 32)` in F32 with
/// left-to-right multiplication, exactly as GGML computes it.
/// Reconstructed quants span `[-32, 31]`.
pub fn dequantize_block_q6_k(block: &[u8], out: &mut [f32]) -> Result<()> {
    if block.len() != Q6_K_BLOCK_BYTES {
        return Err(Error(format!(
            "Q6_K block byte mismatch: expected {Q6_K_BLOCK_BYTES}, got {}",
            block.len()
        )));
    }
    if out.len() != Q6_K_BLOCK_VALUES {
        return Err(Error(format!(
            "Q6_K block output mismatch: expected {Q6_K_BLOCK_VALUES}, got {}",
            out.len()
        )));
    }
    dequantize_block_q6_k_inner(block, out);
    Ok(())
}

/// Reference dot product of one decoded Q6_K block with 256 F32 values.
///
/// Decodes the block with [`dequantize_block_q6_k`], then accumulates
/// `sum(decoded[i] * x[i])` in F32 in index order.
pub fn dot_block_q6_k(block: &[u8], x: &[f32]) -> Result<f32> {
    if x.len() != Q6_K_BLOCK_VALUES {
        return Err(Error(format!(
            "Q6_K block dot input mismatch: expected {Q6_K_BLOCK_VALUES}, got {}",
            x.len()
        )));
    }
    let mut decoded = [0.0f32; Q6_K_BLOCK_VALUES];
    dequantize_block_q6_k(block, &mut decoded)?;
    let mut sum = 0.0f32;
    for (&value, &input) in decoded.iter().zip(x.iter()) {
        sum += value * input;
    }
    Ok(sum)
}

/// Reference dot product over a row of consecutive Q6_K blocks.
///
/// `row` must hold exactly `row_bytes(Q6_K, x.len())` bytes. Accumulation
/// is one F32 running sum in global index order, identical to decoding
/// the whole row and then dotting it.
pub fn dot_row_q6_k(row: &[u8], x: &[f32]) -> Result<f32> {
    let expected = row_bytes(QuantFormat::Q6_K, x.len())?;
    if row.len() != expected {
        return Err(Error(format!(
            "Q6_K row byte mismatch: expected {expected}, got {}",
            row.len()
        )));
    }
    let (blocks, byte_rest) = row.as_chunks::<Q6_K_BLOCK_BYTES>();
    let (chunks, chunk_rest) = x.as_chunks::<Q6_K_BLOCK_VALUES>();
    debug_assert!(byte_rest.is_empty() && chunk_rest.is_empty() && blocks.len() == chunks.len());
    let mut decoded = [0.0f32; Q6_K_BLOCK_VALUES];
    let mut sum = 0.0f32;
    for (block, chunk) in blocks.iter().zip(chunks.iter()) {
        dequantize_block_q6_k_inner(block, &mut decoded);
        for (&value, &input) in decoded.iter().zip(chunk.iter()) {
            sum += value * input;
        }
    }
    Ok(sum)
}

fn dequantize_block_q6_k_inner(block: &[u8], out: &mut [f32]) {
    debug_assert_eq!(block.len(), Q6_K_BLOCK_BYTES);
    debug_assert_eq!(out.len(), Q6_K_BLOCK_VALUES);
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_bits_to_f32(u16::from_le_bytes([block[208], block[209]]));
    for half in 0..2 {
        let base = half * 128;
        let ql = &ql[half * 64..half * 64 + 64];
        let qh = &qh[half * 32..half * 32 + 32];
        let sc = &scales[half * 8..half * 8 + 8];
        for l in 0..32 {
            // Group selector, mirroring GGML's `is = l/16`.
            let is = l / 16;
            let high = qh[l];
            let q1 = i32::from((ql[l] & 0x0F) | ((high & 0x03) << 4)) - 32;
            let q2 = i32::from((ql[l + 32] & 0x0F) | (((high >> 2) & 0x03) << 4)) - 32;
            let q3 = i32::from((ql[l] >> 4) | (((high >> 4) & 0x03) << 4)) - 32;
            let q4 = i32::from((ql[l + 32] >> 4) | (((high >> 6) & 0x03) << 4)) - 32;
            out[base + l] = d * f32::from(sc[is] as i8) * (q1 as f32);
            out[base + l + 32] = d * f32::from(sc[is + 2] as i8) * (q2 as f32);
            out[base + l + 64] = d * f32::from(sc[is + 4] as i8) * (q3 as f32);
            out[base + l + 96] = d * f32::from(sc[is + 6] as i8) * (q4 as f32);
        }
    }
}

fn dequantize_row_q6_k(bytes: &[u8], out: &mut [f32]) {
    let (blocks, byte_rest) = bytes.as_chunks::<Q6_K_BLOCK_BYTES>();
    let (slots, slot_rest) = out.as_chunks_mut::<Q6_K_BLOCK_VALUES>();
    debug_assert!(byte_rest.is_empty() && slot_rest.is_empty() && blocks.len() == slots.len());
    for (block, values) in blocks.iter().zip(slots.iter_mut()) {
        dequantize_block_q6_k_inner(block, values);
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
        // Q6_K has a reviewed reference decoder and is covered by the
        // Q6_K block tests below; the formats below must keep refusing.
        for format in [
            QuantFormat::Q4_K,
            QuantFormat::Q5_0,
            QuantFormat::Q5_K,
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

    /// Uniform Q6_K block: super-scale `d`, one int8 scale everywhere,
    /// one nibble pair everywhere, zero high bits.
    fn uniform_q6_k_block(d_bits: u16, scale: u8, nibble_pair: u8) -> [u8; 210] {
        let mut block = [0u8; 210];
        block[208..210].copy_from_slice(&d_bits.to_le_bytes());
        for byte in block.iter_mut().take(128) {
            *byte = nibble_pair;
        }
        for byte in block.iter_mut().take(208).skip(192) {
            *byte = scale;
        }
        block
    }

    #[test]
    fn q6_k_zero_block_decodes_to_zero() {
        let block = [0u8; 210];
        let mut out = [0.0f32; 256];
        dequantize_block_q6_k(&block, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q6_k_constant_block() {
        // d = 1.0, all scales 1, all nibbles 8, no high bits:
        // every value is 1 * 1 * (8 - 32) = -24.
        let block = uniform_q6_k_block(0x3C00, 1, 0x88);
        let mut out = [0.0f32; 256];
        dequantize_block_q6_k(&block, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == -24.0));
    }

    #[test]
    fn q6_k_nibbles_and_high_bit_positions() {
        // d = 1.0, all scales 1. qh[0] = 0xE4 exercises all four 2-bit
        // fields with distinct values; ql[1] = 0xF0 pins the high-nibble
        // path against the low-nibble path.
        let mut block = uniform_q6_k_block(0x3C00, 1, 0x00);
        block[128] = 0xE4; // qh[0] = 0b11_10_01_00
        block[1] = 0xF0; // ql[1]: low nibble 0, high nibble 15
        let mut out = [0.0f32; 256];
        dequantize_block_q6_k(&block, &mut out).unwrap();
        assert_eq!(out[0], -32.0); // low nibble 0,  high bits 0 -> -32
        assert_eq!(out[32], -16.0); // low nibble 0,  high bits 1 -> -16
        assert_eq!(out[64], 0.0); // high nibble 0, high bits 2 -> 0
        assert_eq!(out[96], 16.0); // high nibble 0, high bits 3 -> 16
        assert_eq!(out[1], -32.0); // low nibble 0 -> -32
        assert_eq!(out[65], -17.0); // high nibble 15 -> 15 - 32
        assert_eq!(out[2], -32.0);
        assert_eq!(out[33], -32.0);
        assert_eq!(out[97], -32.0);
        assert_eq!(out[128], -32.0);
        assert_eq!(out[255], -32.0);
    }

    #[test]
    fn q6_k_scale_groups_and_halves() {
        // d = 1.0, all quants -24 (nibbles 8, no high bits). Half 0 uses
        // scales 1..=8 across its eight 16-value groups; half 1 uses 1s.
        let mut block = uniform_q6_k_block(0x3C00, 1, 0x88);
        block[192..200].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let mut out = [0.0f32; 256];
        dequantize_block_q6_k(&block, &mut out).unwrap();
        let (spans, rest) = out[..128].as_chunks::<16>();
        assert!(rest.is_empty());
        for (group, span) in spans.iter().enumerate() {
            let expected = -24.0 * (group + 1) as f32;
            assert!(
                span.iter().all(|&v| v == expected),
                "group {group} must scale by {}",
                group + 1
            );
        }
        assert!(out[128..].iter().all(|&v| v == -24.0));
    }

    #[test]
    fn q6_k_superblock_scale_and_negative_scales() {
        // d = 0.5, all scales -1 (0xFF), all quants -24:
        // 0.5 * -1 * -24 = 12.
        let block = uniform_q6_k_block(0x3800, 0xFF, 0x88);
        let mut out = [0.0f32; 256];
        dequantize_block_q6_k(&block, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 12.0));
    }

    #[test]
    fn q6_k_second_half_byte_offsets() {
        // Only half-1 base lanes are set: ql[64] low nibble 15, qh[32]
        // bits[1:0] = 3, sc[8] = 1 -> y[128] = 63 - 32 = 31.
        // Everything else decodes to -32, pinning the +64/+32/+8 strides.
        let mut block = uniform_q6_k_block(0x3C00, 1, 0x00);
        block[64] = 0x0F; // half-1 ql base
        block[160] = 0x03; // half-1 qh base (byte 128 + 32)
        let mut out = [0.0f32; 256];
        dequantize_block_q6_k(&block, &mut out).unwrap();
        assert_eq!(out[128], 31.0);
        assert_eq!(out[0], -32.0);
        assert_eq!(out[160], -32.0); // q2: ql[96] = 0, qh[32] bits[3:2] = 0
        assert_eq!(out[192], -32.0); // q3: ql[64] high nibble 0
        assert_eq!(out[224], -32.0); // q4: ql[96] high 0, qh[32] bits[7:6] 0
        assert_eq!(out[129], -32.0);
    }

    #[test]
    fn q6_k_row_of_two_blocks() {
        // Block 0: all -24. Block 1: d = 1, scales 2, nibbles 8 -> -48.
        let block0 = uniform_q6_k_block(0x3C00, 1, 0x88);
        let block1 = uniform_q6_k_block(0x3C00, 2, 0x88);
        let mut row = [0u8; 420];
        row[..210].copy_from_slice(&block0);
        row[210..].copy_from_slice(&block1);
        let mut out = [0.0f32; 512];
        dequantize_row(QuantFormat::Q6_K, &row, 512, &mut out).unwrap();
        assert!(out[..256].iter().all(|&v| v == -24.0));
        assert!(out[256..].iter().all(|&v| v == -48.0));
    }

    #[test]
    fn q6_k_block_dot_products() {
        // All-ones block: quant 1 = 0b100001 needs low nibble 1 and high
        // bits 0b10 in every 2-bit field, i.e. ql = 0x11, qh = 0xAA.
        let mut block = uniform_q6_k_block(0x3C00, 1, 0x11);
        for byte in block.iter_mut().take(192).skip(128) {
            *byte = 0xAA;
        }
        let mut decoded = [0.0f32; 256];
        dequantize_block_q6_k(&block, &mut decoded).unwrap();
        assert!(decoded.iter().all(|&v| v == 1.0));

        assert_eq!(dot_block_q6_k(&block, &[1.0f32; 256]).unwrap(), 256.0);

        let mut ramp = [0.0f32; 256];
        for (index, slot) in ramp.iter_mut().enumerate() {
            *slot = (index + 1) as f32;
        }
        assert_eq!(dot_block_q6_k(&block, &ramp).unwrap(), 32896.0);
    }

    #[test]
    fn q6_k_row_dot_and_matvec() {
        // Row of block 0 (all -24) + block 1 (all -48): 512 values.
        let block0 = uniform_q6_k_block(0x3C00, 1, 0x88);
        let block1 = uniform_q6_k_block(0x3C00, 2, 0x88);
        let mut row = [0u8; 420];
        row[..210].copy_from_slice(&block0);
        row[210..].copy_from_slice(&block1);
        let ones = [1.0f32; 512];
        assert_eq!(dot_row_q6_k(&row, &ones).unwrap(), -18432.0);

        // Same bytes through the generic Q6_K matvec path: two output
        // rows of 512 values against an all-ones input.
        let mut raw = [0u8; 840];
        raw[..210].copy_from_slice(&block0);
        raw[210..420].copy_from_slice(&block0);
        raw[420..630].copy_from_slice(&block1);
        raw[630..].copy_from_slice(&block1);
        let mut y = [0.0f32; 2];
        matvec_reference(
            QuantFormat::Q6_K,
            &raw,
            MatrixShape::new(512, 2),
            &ones,
            &mut y,
        )
        .unwrap();
        assert_eq!(y, [-12288.0, -24576.0]);
    }

    #[test]
    fn q6_k_rejects_bad_lengths() {
        let block = [0u8; 210];
        let mut out = [0.0f32; 256];
        assert!(dequantize_block_q6_k(&block[..209], &mut out).is_err());
        assert!(dequantize_block_q6_k(&[0u8; 211], &mut out).is_err());
        assert!(dequantize_block_q6_k(&block, &mut out[..255]).is_err());
        assert!(dequantize_row(QuantFormat::Q6_K, &[0u8; 419], 512, &mut [0.0f32; 512]).is_err());
        assert!(dot_block_q6_k(&block, &[1.0f32; 255]).is_err());
        assert!(dot_row_q6_k(&[0u8; 210], &[1.0f32; 512]).is_err());
        assert!(dot_row_q6_k(&[0u8; 420], &[1.0f32; 256]).is_err());
    }

    #[test]
    fn q6_k_decode_is_deterministic() {
        // Distinct bytes in every region so a dropped field would show.
        let mut block = [0u8; 210];
        for (index, byte) in block.iter_mut().enumerate() {
            *byte = index as u8;
        }
        let mut first = [0.0f32; 256];
        let mut second = [0.0f32; 256];
        dequantize_block_q6_k(&block, &mut first).unwrap();
        dequantize_block_q6_k(&block, &mut second).unwrap();
        assert_eq!(first, second);
        // The row path decodes the same block identically.
        let mut row_out = [0.0f32; 256];
        dequantize_row(QuantFormat::Q6_K, &block, 256, &mut row_out).unwrap();
        assert_eq!(first, row_out);
    }
}
