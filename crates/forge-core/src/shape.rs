//! Tensor shapes and the authoritative ForgeCore matrix convention.
//!
//! ## Matrix convention
//!
//! A two-dimensional GGUF weight tensor reports logical dimensions
//! `[ne0, ne1]`, where `ne0` is the **input** width and `ne1` is the
//! **output** height. ForgeCore represents this as
//! [`MatrixShape`] `{ input, output }`.
//!
//! Physical F32 storage is contiguous row-major `[output][input]`: row `o`
//! occupies `data[o * input .. (o + 1) * input]`, and the reference
//! operation is:
//!
//! ```text
//! y[o] = sum over i of W[o, i] * x[i]
//! ```
//!
//! There is no transpose guessing anywhere: a buffer whose length does not
//! equal `input * output` is rejected with an explicit error.

use crate::error::{Error, Result};

/// Logical dimensions of a 2-D weight matrix: `[input, output]`.
///
/// `input` corresponds to GGUF `ne[0]` (the input-vector width) and `output`
/// corresponds to GGUF `ne[1]` (the number of output rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    /// Length of the input vector (`x`); GGUF `ne[0]`.
    pub input: usize,
    /// Number of output rows (`y`); GGUF `ne[1]`.
    pub output: usize,
}

impl MatrixShape {
    /// Build a shape directly from logical `[input, output]` dimensions.
    pub const fn new(input: usize, output: usize) -> Self {
        Self { input, output }
    }

    /// Interpret a two-dimensional GGUF `ne` dimension pair.
    ///
    /// `ne[0]` is the input width, `ne[1]` is the output height. Zero
    /// dimensions and values that do not fit in `usize` are rejected.
    pub fn from_gguf_ne(ne: [u64; 2]) -> Result<Self> {
        let input = usize::try_from(ne[0])
            .map_err(|_| Error(format!("GGUF ne[0] value {} exceeds usize", ne[0])))?;
        let output = usize::try_from(ne[1])
            .map_err(|_| Error(format!("GGUF ne[1] value {} exceeds usize", ne[1])))?;
        if input == 0 || output == 0 {
            return Err(Error(format!(
                "GGUF matrix dimensions must be non-zero, got [input={input}, output={output}]"
            )));
        }
        Ok(Self { input, output })
    }

    /// Number of F32 elements in a dense row-major `[output][input]` buffer.
    pub fn element_count(&self) -> Result<usize> {
        self.input.checked_mul(self.output).ok_or_else(|| {
            Error(format!(
                "matrix element count overflows: [input={}, output={}]",
                self.input, self.output
            ))
        })
    }

    /// Validate a dense F32 matvec call: `weights` must hold exactly
    /// `input * output` values, `x` exactly `input`, and `y` exactly `output`.
    pub fn validate_f32(&self, weights_len: usize, x_len: usize, y_len: usize) -> Result<()> {
        let expected = self.element_count()?;
        if weights_len != expected || x_len != self.input || y_len != self.output {
            return Err(Error(format!(
                "matvec arity mismatch: shape [input={}, output={}] requires weights {expected}, x {}, y {}; got weights {weights_len}, x {x_len}, y {y_len}",
                self.input, self.output, self.input, self.output,
            )));
        }
        Ok(())
    }

    /// Element range of output row `o` inside a dense row-major buffer.
    pub fn row_range(&self, output: usize) -> Result<std::ops::Range<usize>> {
        if output >= self.output {
            return Err(Error(format!(
                "output row {output} out of bounds for shape [input={}, output={}]",
                self.input, self.output
            )));
        }
        let start = output
            .checked_mul(self.input)
            .ok_or_else(|| Error("matrix row offset overflows".to_string()))?;
        let end = start
            .checked_add(self.input)
            .ok_or_else(|| Error("matrix row end overflows".to_string()))?;
        Ok(start..end)
    }
}

/// Contiguous row-major F32 matrix with logical shape `[input, output]`.
///
/// Row `o` is `data[o * input .. (o + 1) * input]`; see the module-level
/// matrix convention. The constructor rejects any buffer whose length does
/// not match the shape exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixF32 {
    /// Logical `[input, output]` dimensions.
    pub shape: MatrixShape,
    /// Row-major `[output][input]` values.
    pub data: Vec<f32>,
}

impl MatrixF32 {
    /// Build a matrix, rejecting length/shape mismatches.
    pub fn new(shape: MatrixShape, data: Vec<f32>) -> Result<Self> {
        let expected = shape.element_count()?;
        if data.len() != expected {
            return Err(Error(format!(
                "matrix buffer length mismatch: shape [input={}, output={}] requires {expected} values, got {}",
                shape.input,
                shape.output,
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }

    /// Zero-initialized matrix of the given shape.
    pub fn zeros(shape: MatrixShape) -> Result<Self> {
        let expected = shape.element_count()?;
        Ok(Self {
            shape,
            data: vec![0.0f32; expected],
        })
    }

    /// Borrow output row `o` (`input` consecutive values).
    pub fn row(&self, output: usize) -> Result<&[f32]> {
        let range = self.shape.row_range(output)?;
        Ok(&self.data[range])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_ne_maps_to_input_output() {
        let shape = MatrixShape::from_gguf_ne([1536, 151_936]).unwrap();
        assert_eq!(shape, MatrixShape::new(1536, 151_936));
        assert_eq!(shape.element_count().unwrap(), 1536 * 151_936);
    }

    #[test]
    fn gguf_ne_rejects_zero_dimensions() {
        assert!(MatrixShape::from_gguf_ne([0, 4]).is_err());
        assert!(MatrixShape::from_gguf_ne([4, 0]).is_err());
    }

    #[test]
    fn validate_f32_accepts_exact_arity_only() {
        let shape = MatrixShape::new(3, 2);
        assert!(shape.validate_f32(6, 3, 2).is_ok());
        assert!(shape.validate_f32(6, 2, 2).is_err());
        assert!(shape.validate_f32(6, 3, 3).is_err());
        assert!(shape.validate_f32(5, 3, 2).is_err());
        // Transposed buffer size is silently wrong unless the values match;
        // a 3-output/2-input buffer must not validate as [input=3, output=2].
        assert!(shape.validate_f32(6, 3, 2).is_ok());
        assert!(MatrixShape::new(2, 3).validate_f32(6, 2, 3).is_ok());
    }

    #[test]
    fn row_range_is_row_major_output_major() {
        let shape = MatrixShape::new(4, 3);
        assert_eq!(shape.row_range(0).unwrap(), 0..4);
        assert_eq!(shape.row_range(1).unwrap(), 4..8);
        assert_eq!(shape.row_range(2).unwrap(), 8..12);
        assert!(shape.row_range(3).is_err());
    }

    #[test]
    fn matrix_row_borrows_output_row() {
        let data: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let matrix = MatrixF32::new(MatrixShape::new(2, 3), data).unwrap();
        assert_eq!(matrix.row(0).unwrap(), &[0.0, 1.0]);
        assert_eq!(matrix.row(2).unwrap(), &[4.0, 5.0]);
        assert!(matrix.row(3).is_err());
        assert!(MatrixF32::new(MatrixShape::new(2, 3), vec![0.0; 5]).is_err());
    }
}
