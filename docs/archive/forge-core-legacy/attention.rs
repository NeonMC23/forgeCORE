//! Explicit single-token causal attention with grouped-query attention.
//!
//! ## Layout contract
//!
//! * `q`: `[query_head][head_dim]`, the current token's queries.
//! * `k_history` / `v_history`: `[position][kv_head][head_dim]` over exactly
//!   `history_len` committed positions (see [`crate::kv`]).
//! * `k_current` / `v_current`: `[kv_head][head_dim]` for the current token,
//!   read as position `history_len` (i.e. appended to history conceptually,
//!   but supplied separately so the caller controls KV staging).
//! * Output `hidden`: `[query_head][head_dim]`.
//! * `scores` / `probs`: `[query_head][history_len + 1]`; raw scaled dot
//!   products and post-softmax weights respectively, kept for validation
//!   checkpoints.
//!
//! ## GQA mapping
//!
//! `query_heads` must be divisible by `kv_heads`. Query head `h` reads KV
//! head `h * kv_heads / query_heads`. The mapping is validated, never
//! implicit: [`AttentionDims::kv_head_for`] rejects unvalidated
//! configurations and out-of-range heads.
//!
//! ## Attention math
//!
//! Scores are `dot(q_h, k_p) / sqrt(head_dim)` over history positions plus
//! the current token (single-token forward, so every visible position is
//! causal by construction). Probabilities come from the stable softmax in
//! [`crate::ops`]; the head output is the probability-weighted sum of values.

use crate::error::{Error, Result};
use crate::ops::softmax_in_place;

/// Dimensions and history length for one single-token attention call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionDims {
    /// Number of committed history positions in `k_history`/`v_history`.
    pub history_len: usize,
    /// Number of query heads.
    pub query_heads: usize,
    /// Number of key/value heads; must divide `query_heads`.
    pub kv_heads: usize,
    /// Width of one head.
    pub head_dim: usize,
}

impl AttentionDims {
    /// Build attention dimensions directly.
    pub const fn new(
        history_len: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            history_len,
            query_heads,
            kv_heads,
            head_dim,
        }
    }

    /// Validate head counts and dimension; GQA requires
    /// `query_heads % kv_heads == 0`.
    pub fn validate(&self) -> Result<()> {
        if self.query_heads == 0 || self.kv_heads == 0 || self.head_dim == 0 {
            return Err(Error(
                "attention requires non-zero query heads, KV heads, and head dimension".to_string(),
            ));
        }
        if !self.query_heads.is_multiple_of(self.kv_heads) {
            return Err(Error(format!(
                "GQA requires query_heads {} divisible by kv_heads {}",
                self.query_heads, self.kv_heads
            )));
        }
        Ok(())
    }

    /// Total visible positions: committed history plus the current token.
    pub fn total_positions(&self) -> Result<usize> {
        self.history_len
            .checked_add(1)
            .ok_or_else(|| Error("attention position count overflows".to_string()))
    }

    /// Map query head `h` to its KV head: `h * kv_heads / query_heads`.
    ///
    /// Validates the GQA configuration and the head index on every call so
    /// no caller can rely on unchecked integer arithmetic.
    pub fn kv_head_for(&self, query_head: usize) -> Result<usize> {
        self.validate()?;
        if query_head >= self.query_heads {
            return Err(Error(format!(
                "query head {query_head} out of range for {} query heads",
                self.query_heads
            )));
        }
        Ok(query_head * self.kv_heads / self.query_heads)
    }
}

/// Output of [`attention`]: head outputs plus per-head score checkpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionOutput {
    /// `[query_head][head_dim]` weighted value sums.
    pub hidden: Vec<f32>,
    /// `[query_head][history_len + 1]` raw scaled dot products.
    pub scores: Vec<f32>,
    /// `[query_head][history_len + 1]` post-softmax weights.
    pub probs: Vec<f32>,
}

/// Scalar single-token causal attention reference.
///
/// See the module-level documentation for layouts, GQA mapping, and math.
pub fn attention(
    q: &[f32],
    k_history: &[f32],
    v_history: &[f32],
    k_current: &[f32],
    v_current: &[f32],
    dims: AttentionDims,
) -> Result<AttentionOutput> {
    dims.validate()?;
    let total = dims.total_positions()?;
    let kv_width = dims
        .kv_heads
        .checked_mul(dims.head_dim)
        .ok_or_else(|| Error("attention KV width overflows".to_string()))?;
    let q_width = dims
        .query_heads
        .checked_mul(dims.head_dim)
        .ok_or_else(|| Error("attention query width overflows".to_string()))?;
    let history_elems = dims
        .history_len
        .checked_mul(kv_width)
        .ok_or_else(|| Error("attention history size overflows".to_string()))?;

    if q.len() != q_width
        || k_history.len() != history_elems
        || v_history.len() != history_elems
        || k_current.len() != kv_width
        || v_current.len() != kv_width
    {
        return Err(Error(format!(
            "attention tensor shape mismatch: q {}, k_history {}, v_history {}, k_current {}, v_current {} for {:?}",
            q.len(),
            k_history.len(),
            v_history.len(),
            k_current.len(),
            v_current.len(),
            dims,
        )));
    }

    let scale = (dims.head_dim as f32).sqrt();
    let mut hidden = vec![0.0f32; q_width];
    let mut scores = vec![0.0f32; dims.query_heads * total];
    let mut probs = vec![0.0f32; dims.query_heads * total];

    for query_head in 0..dims.query_heads {
        // Validated above; the mapping call re-checks defensively.
        let kv_head = dims.kv_head_for(query_head)?;
        let q_head = &q[query_head * dims.head_dim..(query_head + 1) * dims.head_dim];

        let score_row = &mut scores[query_head * total..(query_head + 1) * total];
        for position in 0..total {
            let k_all = if position < dims.history_len {
                &k_history[position * kv_width..(position + 1) * kv_width]
            } else {
                k_current
            };
            let k_head = &k_all[kv_head * dims.head_dim..(kv_head + 1) * dims.head_dim];
            let mut dot = 0.0f32;
            for index in 0..dims.head_dim {
                dot += q_head[index] * k_head[index];
            }
            score_row[position] = dot / scale;
        }

        let prob_row = &mut probs[query_head * total..(query_head + 1) * total];
        prob_row.copy_from_slice(score_row);
        softmax_in_place(prob_row)?;

        let out_head = &mut hidden[query_head * dims.head_dim..(query_head + 1) * dims.head_dim];
        for position in 0..total {
            let v_all = if position < dims.history_len {
                &v_history[position * kv_width..(position + 1) * kv_width]
            } else {
                v_current
            };
            let v_head = &v_all[kv_head * dims.head_dim..(kv_head + 1) * dims.head_dim];
            let weight = prob_row[position];
            for index in 0..dims.head_dim {
                out_head[index] += weight * v_head[index];
            }
        }
    }

    Ok(AttentionOutput {
        hidden,
        scores,
        probs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_head_mapping_is_explicit() {
        let dims = AttentionDims::new(0, 8, 2, 4);
        let mapped: Vec<usize> = (0..8).map(|h| dims.kv_head_for(h).unwrap()).collect();
        assert_eq!(mapped, [0, 0, 0, 0, 1, 1, 1, 1]);
        assert!(dims.kv_head_for(8).is_err());
        assert!(AttentionDims::new(0, 3, 2, 4).kv_head_for(0).is_err());
        assert!(AttentionDims::new(0, 0, 1, 4).validate().is_err());
    }

    #[test]
    fn single_position_output_is_mapped_value() {
        // With one visible position the softmax weight is 1.0, so each
        // query head must reproduce its mapped KV head's value vector.
        let out = attention(
            &[1.0, 0.0, 0.0, 1.0],
            &[],
            &[],
            &[1.0, 0.0],
            &[2.0, 3.0],
            AttentionDims::new(0, 2, 1, 2),
        )
        .unwrap();
        assert_eq!(out.hidden, [2.0, 3.0, 2.0, 3.0]);
        assert_eq!(out.probs, [1.0, 1.0]);
    }

    #[test]
    fn two_position_scores_hand_computed() {
        // head_dim=1, so scale=1: scores are plain dot products.
        // q=[2], history k=[1], current k=[3] -> scores [2, 6].
        let out = attention(
            &[2.0],
            &[1.0],
            &[10.0],
            &[3.0],
            &[20.0],
            AttentionDims::new(1, 1, 1, 1),
        )
        .unwrap();
        assert_eq!(out.scores, [2.0, 6.0]);
        let w0 = (-4.0f32).exp() / (1.0 + (-4.0f32).exp());
        let w1 = 1.0 / (1.0 + (-4.0f32).exp());
        assert!((out.probs[0] - w0).abs() < 1e-6);
        assert!((out.probs[1] - w1).abs() < 1e-6);
        assert!((out.hidden[0] - (w0 * 10.0 + w1 * 20.0)).abs() < 1e-5);
    }

    #[test]
    fn attention_rejects_shape_mismatch() {
        let dims = AttentionDims::new(1, 1, 1, 2);
        assert!(attention(&[0.0; 2], &[0.0; 2], &[0.0], &[0.0; 2], &[0.0; 2], dims).is_err());
        assert!(attention(&[0.0; 3], &[0.0; 2], &[0.0; 2], &[0.0; 2], &[0.0; 2], dims).is_err());
    }
}
