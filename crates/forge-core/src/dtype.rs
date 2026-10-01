//! Element data types.
//!
//! [`DType`] is the safe-Rust mirror of upstream `enum ggml_type`: each
//! variant's discriminant is the exact ggml type id, so conversion to the
//! C ABI is a plain `as` cast. Types ForgeCore does not map yet surface
//! as [`Error`] through [`DType::from_ggml`], never as silent substitutes.

use crate::error::{Error, Result};

/// Tensor element type, mirroring upstream `enum ggml_type`.
#[allow(non_camel_case_types)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    F64 = 28,
    BF16 = 30,
}

impl DType {
    /// ggml type id for the FFI boundary.
    pub fn ggml_type(self) -> std::os::raw::c_int {
        self as std::os::raw::c_int
    }

    /// Map a ggml type id back, rejecting unmapped ids explicitly.
    pub fn from_ggml(id: i32) -> Result<Self> {
        let dtype = match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            28 => Self::F64,
            30 => Self::BF16,
            other => return Err(Error::unsupported(format!("ggml type id {other}"))),
        };
        Ok(dtype)
    }

    /// Short stable name (matches the ggml spelling).
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::F64 => "F64",
            Self::BF16 => "BF16",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminants_match_sys_constants() {
        assert_eq!(DType::F32.ggml_type(), forge_sys::ggml_type::F32);
        assert_eq!(DType::F16.ggml_type(), forge_sys::ggml_type::F16);
        assert_eq!(DType::Q4_0.ggml_type(), forge_sys::ggml_type::Q4_0);
        assert_eq!(DType::Q4_1.ggml_type(), forge_sys::ggml_type::Q4_1);
        assert_eq!(DType::Q5_0.ggml_type(), forge_sys::ggml_type::Q5_0);
        assert_eq!(DType::Q5_1.ggml_type(), forge_sys::ggml_type::Q5_1);
        assert_eq!(DType::Q8_0.ggml_type(), forge_sys::ggml_type::Q8_0);
        assert_eq!(DType::Q8_1.ggml_type(), forge_sys::ggml_type::Q8_1);
        assert_eq!(DType::Q2_K.ggml_type(), forge_sys::ggml_type::Q2_K);
        assert_eq!(DType::Q3_K.ggml_type(), forge_sys::ggml_type::Q3_K);
        assert_eq!(DType::Q4_K.ggml_type(), forge_sys::ggml_type::Q4_K);
        assert_eq!(DType::Q5_K.ggml_type(), forge_sys::ggml_type::Q5_K);
        assert_eq!(DType::Q6_K.ggml_type(), forge_sys::ggml_type::Q6_K);
        assert_eq!(DType::Q8_K.ggml_type(), forge_sys::ggml_type::Q8_K);
        assert_eq!(DType::I8.ggml_type(), forge_sys::ggml_type::I8);
        assert_eq!(DType::I16.ggml_type(), forge_sys::ggml_type::I16);
        assert_eq!(DType::I32.ggml_type(), forge_sys::ggml_type::I32);
        assert_eq!(DType::F64.ggml_type(), forge_sys::ggml_type::F64);
        assert_eq!(DType::BF16.ggml_type(), forge_sys::ggml_type::BF16);
    }

    #[test]
    fn from_ggml_round_trips_and_rejects_unknown() {
        for dtype in [
            DType::F32,
            DType::F16,
            DType::Q4_0,
            DType::Q4_1,
            DType::Q5_0,
            DType::Q5_1,
            DType::Q8_0,
            DType::Q8_1,
            DType::Q2_K,
            DType::Q3_K,
            DType::Q4_K,
            DType::Q5_K,
            DType::Q6_K,
            DType::Q8_K,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::F64,
            DType::BF16,
        ] {
            assert_eq!(DType::from_ggml(dtype.ggml_type()), Ok(dtype));
        }
        assert!(DType::from_ggml(4).is_err()); // removed Q4_2 slot
        assert!(DType::from_ggml(999).is_err());
    }
}
