//! Frozen scalar validation oracle. See `crate::reference` for what this means.
//!
//! Scalar floating-point bit-pattern conversion.
//!
//! GGUF tensors may store weights as F16 or BF16. ForgeCore decodes them to
//! F32 through these explicit bit-level conversions before any arithmetic.
//! There is no platform-dependent behavior: the conversions are pure integer
//! bit manipulation plus [`f32::from_bits`].

/// Convert 16 F16 bits (IEEE 754 binary16, little-endian value) to F32.
///
/// Handles zeros, subnormals, normals, infinities, and NaNs. The conversion
/// is exact: every F16 value is representable in F32.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits >> 15) & 1) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x3FF);

    let f32_bits = if exp == 0 {
        if mant == 0 {
            // Signed zero.
            sign
        } else {
            // Subnormal: value = mant * 2^-24. Normalize so bit 10 is set:
            // value = (1 + frac) * 2^(-14 - shift).
            let shift = mant.leading_zeros() - 21;
            let normalized = mant << shift;
            let f32_exp = 113 - shift;
            sign | (f32_exp << 23) | ((normalized & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        // Infinity or NaN; payload preserved in the high mantissa bits.
        sign | (0xFF << 23) | (mant << 13)
    } else {
        // Normal: re-bias exponent from 15 to 127 (written as +112 so
        // small exponents cannot underflow in debug builds).
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Convert 16 BF16 bits (brain float 16, little-endian value) to F32.
///
/// BF16 is the upper 16 bits of an F32; conversion appends 16 zero bits.
/// Exact, including infinities and NaNs.
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_bit_patterns() {
        assert_eq!(f16_bits_to_f32(0x3C00), 1.0);
        assert_eq!(f16_bits_to_f32(0xBC00), -1.0);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        assert_eq!(f16_bits_to_f32(0xC000), -2.0);
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert_eq!(f16_bits_to_f32(0x8000), -0.0);
        assert_eq!(f16_bits_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_bits_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_bits_to_f32(0x7E00).is_nan());
        // 0x3555 = 0 01101 0101010101 -> (1365/1024) * 2^-2
        // = 1365/4096 = 0.333251953125 (exactly representable; the
        // literal is truncated to what F32 distinguishes).
        assert_eq!(f16_bits_to_f32(0x3555), 0.333_251_95);
    }

    #[test]
    fn f16_subnormals_are_exact() {
        // Smallest subnormal: 2^-24.
        assert_eq!(f16_bits_to_f32(0x0001), 2f32.powi(-24));
        // Largest subnormal: 1023 * 2^-24.
        assert_eq!(f16_bits_to_f32(0x03FF), 1023.0 * 2f32.powi(-24));
        // Smallest normal: 2^-14.
        assert_eq!(f16_bits_to_f32(0x0400), 2f32.powi(-14));
        // Largest finite: 65504.
        assert_eq!(f16_bits_to_f32(0x7BFF), 65504.0);
    }

    #[test]
    fn bf16_known_bit_patterns() {
        assert_eq!(bf16_bits_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_bits_to_f32(0xBF80), -1.0);
        assert_eq!(bf16_bits_to_f32(0x4000), 2.0);
        assert_eq!(bf16_bits_to_f32(0x0000), 0.0);
        assert_eq!(bf16_bits_to_f32(0x7F80), f32::INFINITY);
        assert!(bf16_bits_to_f32(0x7FC0).is_nan());
        // 0x3F80_0000 is 1.0f32; upper half 0x3F80 must round-trip exactly.
        assert_eq!(bf16_bits_to_f32(0x3F80).to_bits(), 0x3F80_0000);
        assert_eq!(bf16_bits_to_f32(0xC000).to_bits(), 0xC000_0000);
    }
}
