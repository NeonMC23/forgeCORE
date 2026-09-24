//! Scalar reference operators.
//!
//! Every function here is deliberately unoptimized single-threaded scalar
//! code with explicit shapes and F32 accumulation in input order. Optimized
//! kernels added later must prove equivalence against these definitions;
//! they must not redefine the mathematics.
//!
//! Conventions fixed by this module:
//!
//! * [`matvec`]: `y[o] = sum_i W[o, i] * x[i]` over row-major
//!   `[output][input]` weights, F32 accumulation in increasing `i`.
//! * [`rms_norm`]: `y = x / sqrt(mean(x^2) + eps) * w`.
//! * [`silu`]: `x / (1 + exp(-x))`; [`swiglu`]: `SiLU(gate) * up`.
//! * [`softmax_in_place`]: max-subtracted stable softmax.
//! * [`rope`]: half-split pairs `(j, j + head_dim / 2)` with angle
//!   `position * theta^(-2j / head_dim)`; this is the Llama/Qwen2 pairing,
//!   not the interleaved GPT-J pairing.

use crate::error::{Error, Result};
use crate::shape::MatrixShape;

/// Scalar F32 dot product with accumulation in index order.
///
/// The inputs must have equal length. An empty dot product is `0.0`.
pub fn dot(x: &[f32], y: &[f32]) -> Result<f32> {
    if x.len() != y.len() {
        return Err(Error(format!(
            "dot arity mismatch: x {}, y {}",
            x.len(),
            y.len()
        )));
    }
    let mut sum = 0.0f32;
    for (&a, &b) in x.iter().zip(y.iter()) {
        sum += a * b;
    }
    Ok(sum)
}

/// Scalar F32 matrix-vector multiplication.
///
/// `weights` is row-major `[output][input]`; see [`MatrixShape`]. Each row
/// accumulates in F32 in increasing input-index order.
pub fn matvec(weights: &[f32], shape: MatrixShape, x: &[f32], y: &mut [f32]) -> Result<()> {
    shape.validate_f32(weights.len(), x.len(), y.len())?;
    for (output, row_index) in y.iter_mut().zip(0..shape.output) {
        let row = &weights[row_index * shape.input..(row_index + 1) * shape.input];
        let mut sum = 0.0f32;
        for (&weight, &input) in row.iter().zip(x.iter()) {
            sum += weight * input;
        }
        *output = sum;
    }
    Ok(())
}

/// RMSNorm: `y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]`.
///
/// The sum of squares is accumulated in F32 in input order. `eps` must be
/// finite and non-negative. All three slices must have equal non-zero length.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) -> Result<()> {
    if x.is_empty() || x.len() != w.len() || x.len() != y.len() {
        return Err(Error(format!(
            "RMSNorm shape mismatch: x {}, w {}, y {}",
            x.len(),
            w.len(),
            y.len()
        )));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(Error(
            "RMSNorm epsilon must be finite and non-negative".to_string(),
        ));
    }
    let mut sum_squares = 0.0f32;
    for &value in x {
        sum_squares += value * value;
    }
    let rms = (sum_squares / x.len() as f32 + eps).sqrt();
    for index in 0..x.len() {
        y[index] = x[index] / rms * w[index];
    }
    Ok(())
}

/// Elementwise addition: `y[i] = a[i] + b[i]`.
pub fn add(a: &[f32], b: &[f32], y: &mut [f32]) -> Result<()> {
    validate_elementwise("add", a, b, y)?;
    for index in 0..a.len() {
        y[index] = a[index] + b[index];
    }
    Ok(())
}

/// In-place elementwise addition: `a[i] += b[i]`.
pub fn add_assign(a: &mut [f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(Error(format!(
            "add_assign arity mismatch: a {}, b {}",
            a.len(),
            b.len()
        )));
    }
    for index in 0..a.len() {
        a[index] += b[index];
    }
    Ok(())
}

/// Elementwise multiplication: `y[i] = a[i] * b[i]`.
pub fn mul(a: &[f32], b: &[f32], y: &mut [f32]) -> Result<()> {
    validate_elementwise("mul", a, b, y)?;
    for index in 0..a.len() {
        y[index] = a[index] * b[index];
    }
    Ok(())
}

/// SiLU (swish): `y = x / (1 + exp(-x))`.
pub fn silu(x: &[f32], y: &mut [f32]) -> Result<()> {
    if x.len() != y.len() {
        return Err(Error(format!(
            "SiLU shape mismatch: x {}, y {}",
            x.len(),
            y.len()
        )));
    }
    for index in 0..x.len() {
        let value = x[index];
        y[index] = value / (1.0 + (-value).exp());
    }
    Ok(())
}

/// SwiGLU: `y = SiLU(gate) * up`, elementwise.
pub fn swiglu(gate: &[f32], up: &[f32], y: &mut [f32]) -> Result<()> {
    validate_elementwise("swiglu", gate, up, y)?;
    for index in 0..gate.len() {
        let value = gate[index];
        y[index] = value / (1.0 + (-value).exp()) * up[index];
    }
    Ok(())
}

/// In-place numerically stable softmax: subtract the max, exponentiate,
/// normalize by the sum. The input must be non-empty and normalizable
/// (finite, non-zero sum after exponentiation).
pub fn softmax_in_place(values: &mut [f32]) -> Result<()> {
    if values.is_empty() {
        return Err(Error("softmax requires a non-empty vector".to_string()));
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if !sum.is_finite() || sum == 0.0 {
        return Err(Error(
            "softmax normalization is not finite or normalizable".to_string(),
        ));
    }
    for value in values.iter_mut() {
        *value /= sum;
    }
    Ok(())
}

/// Half-split RoPE (Llama/Qwen2 pairing) applied in place to Q and K.
///
/// `q` is laid out `[query_heads, head_dim]`, `k` as `[kv_heads, head_dim]`.
/// Within one head of width `d`, index `j` in `[0, d/2)` rotates the pair
/// `(v[j], v[j + d/2])` by `angle = position * theta^(-2j / d)`:
///
/// ```text
/// v[j]       = v[j] * cos(angle) - v[j + d/2] * sin(angle)
/// v[j + d/2] = v[j] * sin(angle) + v[j + d/2] * cos(angle)
/// ```
///
/// `head_dim` must be non-zero and even, head counts non-zero, and `theta`
/// finite and positive.
pub fn rope(
    q: &mut [f32],
    k: &mut [f32],
    position: usize,
    head_dim: usize,
    query_heads: usize,
    kv_heads: usize,
    theta: f32,
) -> Result<()> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(Error(
            "RoPE head dimension must be non-zero and even".to_string(),
        ));
    }
    if query_heads == 0 || kv_heads == 0 {
        return Err(Error(
            "RoPE requires non-zero query and KV head counts".to_string(),
        ));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error(
            "RoPE frequency base must be finite and positive".to_string(),
        ));
    }
    let q_width = query_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error("RoPE query width overflows".to_string()))?;
    let k_width = kv_heads
        .checked_mul(head_dim)
        .ok_or_else(|| Error("RoPE KV width overflows".to_string()))?;
    if q.len() != q_width || k.len() != k_width {
        return Err(Error(format!(
            "RoPE shape mismatch: q {}, k {}, query_heads {query_heads}, kv_heads {kv_heads}, head_dim {head_dim}",
            q.len(),
            k.len(),
        )));
    }
    for head in 0..query_heads {
        rope_head(
            &mut q[head * head_dim..(head + 1) * head_dim],
            position,
            theta,
        );
    }
    for head in 0..kv_heads {
        rope_head(
            &mut k[head * head_dim..(head + 1) * head_dim],
            position,
            theta,
        );
    }
    Ok(())
}

fn rope_head(values: &mut [f32], position: usize, theta: f32) {
    let half = values.len() / 2;
    let width = values.len() as f32;
    let position = position as f32;
    for index in 0..half {
        let angle = theta.powf(-2.0 * index as f32 / width) * position;
        let (sin, cos) = angle.sin_cos();
        let first = values[index];
        let second = values[index + half];
        values[index] = first * cos - second * sin;
        values[index + half] = first * sin + second * cos;
    }
}

fn validate_elementwise(name: &str, a: &[f32], b: &[f32], y: &[f32]) -> Result<()> {
    if a.len() != b.len() || a.len() != y.len() {
        return Err(Error(format!(
            "{name} shape mismatch: a {}, b {}, y {}",
            a.len(),
            b.len(),
            y.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_accumulates_in_order() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap(), 32.0);
        assert_eq!(dot(&[], &[]).unwrap(), 0.0);
        assert!(dot(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn matvec_uses_input_output_convention() {
        // Shape [input=3, output=2]; rows are [1,2,3] and [4,5,6].
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut y = [0.0; 2];
        matvec(&weights, MatrixShape::new(3, 2), &[1.0, 1.0, 1.0], &mut y).unwrap();
        assert_eq!(y, [6.0, 15.0]);
    }

    #[test]
    fn matvec_nonsquare_hand_computed() {
        // Shape [input=2, output=3]; x = [2, -1].
        let weights = [1.0, 0.5, -1.0, 2.0, 0.0, 3.0];
        let mut y = [0.0; 3];
        matvec(&weights, MatrixShape::new(2, 3), &[2.0, -1.0], &mut y).unwrap();
        assert_eq!(y, [1.5, -4.0, -3.0]);
    }

    #[test]
    fn matvec_rejects_arity_mismatch() {
        let weights = [1.0; 6];
        let mut y = [0.0; 2];
        assert!(matvec(&weights, MatrixShape::new(3, 2), &[1.0, 1.0], &mut y).is_err());
        assert!(matvec(&weights[..5], MatrixShape::new(3, 2), &[1.0; 3], &mut y).is_err());
    }

    #[test]
    fn rms_norm_hand_computed() {
        // x = [3, 4]: mean squares = 25/2, rms = sqrt(12.5).
        let mut y = [0.0; 2];
        rms_norm(&[3.0, 4.0], &[1.0, 1.0], 0.0, &mut y).unwrap();
        let rms = (12.5f32).sqrt();
        assert!((y[0] - 3.0 / rms).abs() < 1e-7);
        assert!((y[1] - 4.0 / rms).abs() < 1e-7);
    }

    #[test]
    fn rms_norm_applies_weight_and_eps() {
        let mut y = [0.0; 1];
        rms_norm(&[2.0], &[3.0], 0.0, &mut y).unwrap();
        assert_eq!(y, [3.0]);
        assert!(rms_norm(&[], &[], 0.0, &mut []).is_err());
        assert!(rms_norm(&[1.0], &[1.0], f32::NAN, &mut [0.0]).is_err());
        assert!(rms_norm(&[1.0], &[1.0], -1.0, &mut [0.0]).is_err());
    }

    #[test]
    fn elementwise_ops() {
        let mut y = [0.0; 2];
        add(&[1.0, 2.0], &[3.0, 4.0], &mut y).unwrap();
        assert_eq!(y, [4.0, 6.0]);
        mul(&[1.0, 2.0], &[3.0, 4.0], &mut y).unwrap();
        assert_eq!(y, [3.0, 8.0]);
        let mut a = [1.0, 2.0];
        add_assign(&mut a, &[3.0, 4.0]).unwrap();
        assert_eq!(a, [4.0, 6.0]);
        assert!(add(&[1.0], &[1.0, 2.0], &mut [0.0]).is_err());
        assert!(add_assign(&mut [0.0], &[0.0, 0.0]).is_err());
    }

    #[test]
    fn silu_and_swiglu_hand_computed() {
        let mut y = [0.0; 3];
        silu(&[0.0, 1.0, -1.0], &mut y).unwrap();
        assert_eq!(y[0], 0.0);
        assert!((y[1] - (1.0 / (1.0 + (-1.0f32).exp()))).abs() < 1e-7);
        assert!((y[2] - (-1.0 / (1.0 + 1.0f32.exp()))).abs() < 1e-7);

        let mut z = [0.0; 2];
        swiglu(&[0.0, 2.0], &[5.0, 4.0], &mut z).unwrap();
        assert_eq!(z[0], 0.0);
        assert!((z[1] - (2.0 / (1.0 + (-2.0f32).exp()) * 4.0)).abs() < 1e-6);
    }

    #[test]
    fn softmax_is_stable_and_normalized() {
        let mut v = [1000.0, 1001.0, 999.0];
        softmax_in_place(&mut v).unwrap();
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Softmax is shift-invariant: same result as [0, 1, -1].
        let e1 = 1.0f32.exp();
        let total = 1.0 + e1 + (-1.0f32).exp();
        assert!((v[0] - 1.0 / total).abs() < 1e-6);
        assert!((v[1] - e1 / total).abs() < 1e-6);
        assert!(softmax_in_place(&mut []).is_err());
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let mut q = [1.0, 2.0, 3.0, 4.0];
        let mut k = [5.0, 6.0, 7.0, 8.0];
        rope(&mut q, &mut k, 0, 4, 1, 1, 10_000.0).unwrap();
        assert_eq!(q, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(k, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn rope_half_split_rotation_hand_computed() {
        // head_dim=2 has one pair (j=0): angle = position * theta^0 = position.
        let mut q = [1.0, 0.0];
        let mut k = [0.0, 1.0];
        rope(&mut q, &mut k, 1, 2, 1, 1, 10_000.0).unwrap();
        assert!((q[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((q[1] - 1.0f32.sin()).abs() < 1e-6);
        assert!((k[0] - -1.0f32.sin()).abs() < 1e-6);
        assert!((k[1] - 1.0f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn rope_second_pair_uses_slower_frequency() {
        // head_dim=4: pair j=1 rotates by position * theta^(-1/2).
        let mut q = [0.0, 1.0, 0.0, 0.0];
        let mut k = [0.0; 4];
        rope(&mut q, &mut k, 2, 4, 1, 1, 10_000.0).unwrap();
        let angle = 2.0 * 10_000f32.powf(-0.5);
        assert!((q[1] - angle.cos()).abs() < 1e-6);
        assert!((q[3] - angle.sin()).abs() < 1e-6);
        assert_eq!(q[0], 0.0);
        assert_eq!(q[2], 0.0);
    }

    #[test]
    fn rope_validates_shapes() {
        let mut q = [0.0; 4];
        let mut k = [0.0; 4];
        assert!(rope(&mut q, &mut k, 0, 3, 1, 1, 10_000.0).is_err());
        assert!(rope(&mut q, &mut k, 0, 0, 1, 1, 10_000.0).is_err());
        assert!(rope(&mut q, &mut k, 0, 4, 0, 1, 10_000.0).is_err());
        assert!(rope(&mut q, &mut k, 0, 4, 1, 1, -1.0).is_err());
        let mut short = [0.0; 2];
        assert!(rope(&mut short, &mut k, 0, 4, 1, 1, 10_000.0).is_err());
    }
}
