//! Quantization contract tests: block geometry and scalar decoders.
use forge_core::quant::{
    block_spec, dequantize_row, matvec_reference, row_bytes, BlockSpec, QuantFormat,
};
use forge_core::shape::MatrixShape;

#[test]
fn block_geometry_table() {
    assert_eq!(
        block_spec(QuantFormat::Q4_0),
        BlockSpec {
            values_per_block: 32,
            bytes_per_block: 18
        }
    );
    assert_eq!(
        block_spec(QuantFormat::Q8_0),
        BlockSpec {
            values_per_block: 32,
            bytes_per_block: 34
        }
    );
    assert_eq!(block_spec(QuantFormat::Q4_K).values_per_block, 256);
    assert_eq!(block_spec(QuantFormat::Q4_K).bytes_per_block, 144);
    assert_eq!(block_spec(QuantFormat::Q5_K).bytes_per_block, 176);
    assert_eq!(block_spec(QuantFormat::Q6_K).bytes_per_block, 210);
    assert_eq!(block_spec(QuantFormat::Q8_K).bytes_per_block, 292);
    assert_eq!(block_spec(QuantFormat::Q5_0).values_per_block, 32);
    assert_eq!(block_spec(QuantFormat::Q5_0).bytes_per_block, 22);
    assert_eq!(block_spec(QuantFormat::F16).bytes_per_block, 2);
    assert_eq!(block_spec(QuantFormat::BF16).bytes_per_block, 2);
}

#[test]
fn qwen25_output_row_geometry_from_handoff() {
    // Recorded VERIFIED geometry: output.weight is Q6_K with 1536-wide
    // rows, 6 blocks per row, 1260 bytes per row, vocab 151936 rows.
    assert_eq!(row_bytes(QuantFormat::Q6_K, 1536).unwrap(), 1260);
    assert_eq!(1260usize * 151_936, 191_439_360);
    // A Q4_0 row of the same width would hold 48 blocks of 18 bytes.
    assert_eq!(row_bytes(QuantFormat::Q4_0, 1536).unwrap(), 48 * 18);
}

#[test]
fn q4_0_multi_block_row_decodes() {
    // Two blocks: block 0 all zeros (nibbles 8, scale 1), block 1 all
    // 0.5 (nibbles 9, scale 0.5 = 0x3800).
    let mut raw = vec![0u8; 36];
    raw[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    for byte in raw.iter_mut().take(18).skip(2) {
        *byte = 0x88;
    }
    raw[18..20].copy_from_slice(&0x3800u16.to_le_bytes());
    for byte in raw.iter_mut().skip(20) {
        *byte = 0x99;
    }
    let mut out = vec![0.0f32; 64];
    dequantize_row(QuantFormat::Q4_0, &raw, 64, &mut out).unwrap();
    assert!(out[..32].iter().all(|&v| v == 0.0));
    assert!(out[32..].iter().all(|&v| v == 0.5));
}

#[test]
fn q8_0_matvec_matches_hand_computed_dot() {
    // Row 0: scale 1.0, payload 0..31. Row 1: scale 0.5, payload all 2.
    let mut raw = vec![0u8; 2 * 34];
    raw[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    for (index, slot) in raw.iter_mut().take(34).skip(2).enumerate() {
        *slot = index as u8;
    }
    raw[34..36].copy_from_slice(&0x3800u16.to_le_bytes());
    for slot in raw.iter_mut().skip(36) {
        *slot = 2;
    }
    let x = vec![1.0f32; 32];
    let mut y = vec![0.0f32; 2];
    matvec_reference(QuantFormat::Q8_0, &raw, MatrixShape::new(32, 2), &x, &mut y).unwrap();
    // sum(0..32) = 496; 32 values of 2*0.5 = 32.
    assert_eq!(y[0], 496.0);
    assert_eq!(y[1], 32.0);
}

#[test]
fn quantized_paths_reject_mismatched_buffers() {
    let x = vec![1.0f32; 32];
    let mut y = vec![0.0f32; 1];
    // Truncated weight buffer.
    assert!(matvec_reference(
        QuantFormat::Q4_0,
        &[0u8; 17],
        MatrixShape::new(32, 1),
        &x,
        &mut y
    )
    .is_err());
    // Wrong input width.
    assert!(matvec_reference(
        QuantFormat::Q4_0,
        &[0u8; 18],
        MatrixShape::new(32, 1),
        &[1.0f32; 31],
        &mut y
    )
    .is_err());
    // Misaligned input dimension.
    assert!(row_bytes(QuantFormat::Q4_0, 100).is_err());
}
