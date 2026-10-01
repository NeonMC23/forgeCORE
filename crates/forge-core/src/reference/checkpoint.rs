//! Frozen scalar validation oracle. See `crate::reference` for what this means.
//!
//! Deterministic numerical checkpoints for validation.
//!
//! A [`Summary`] reports length, min, max, sum, L2 norm, and the first eight
//! values of a vector without printing the whole vector. Sums accumulate in
//! F64 from each F32 value so long-vector summaries are stable. [`top_k`]
//! reports the highest-logit token ids with token id as the tie-break.
//!
//! This is the vocabulary future real-model comparisons (including external
//! llama.cpp oracle runs) will share: compact, deterministic, printable.

/// Compact numerical summary of a vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// Number of values.
    pub len: usize,
    /// Minimum value (`+inf` when empty).
    pub min: f32,
    /// Maximum value (`-inf` when empty).
    pub max: f32,
    /// Sum accumulated in F64.
    pub sum: f64,
    /// L2 norm accumulated in F64.
    pub l2: f64,
    /// First eight values (fewer when the vector is shorter).
    pub first8: Vec<f32>,
}

/// Summarize a vector. Empty input yields `len 0`, `min +inf`, `max -inf`,
/// zero sum/norm, and no first values.
pub fn summarize(values: &[f32]) -> Summary {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut squares = 0.0f64;
    for &value in values {
        min = min.min(value);
        max = max.max(value);
        let wide = f64::from(value);
        sum += wide;
        squares += wide * wide;
    }
    Summary {
        len: values.len(),
        min,
        max,
        sum,
        l2: squares.sqrt(),
        first8: values.iter().copied().take(8).collect(),
    }
}

/// Format a summary in the stable multi-line diagnostic layout.
pub fn format_summary(name: &str, values: &[f32]) -> String {
    let summary = summarize(values);
    let mut out = String::new();
    out.push_str(&format!("{name}.length = {}\n", summary.len));
    out.push_str(&format!("{name}.min = {:?}\n", summary.min));
    out.push_str(&format!("{name}.max = {:?}\n", summary.max));
    out.push_str(&format!("{name}.sum = {:?}\n", summary.sum));
    out.push_str(&format!("{name}.l2_norm = {:?}\n", summary.l2));
    out.push_str(&format!("{name}.first8 = {:?}\n", summary.first8));
    out
}

/// Read one indexed checkpoint value, or `None` when out of bounds.
pub fn at(values: &[f32], index: usize) -> Option<f32> {
    values.get(index).copied()
}

/// Top-`k` `(token_id, logit)` pairs sorted by descending logit with token
/// id as the tie-break. `k` larger than the input returns the whole input.
pub fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(k);
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_reports_expected_statistics() {
        let summary = summarize(&[3.0, -4.0, 0.0]);
        assert_eq!(summary.len, 3);
        assert_eq!(summary.min, -4.0);
        assert_eq!(summary.max, 3.0);
        assert_eq!(summary.sum, -1.0);
        assert_eq!(summary.l2, 5.0);
        assert_eq!(summary.first8, [3.0, -4.0, 0.0]);
    }

    #[test]
    fn summary_handles_empty_input() {
        let summary = summarize(&[]);
        assert_eq!(summary.len, 0);
        assert_eq!(summary.min, f32::INFINITY);
        assert_eq!(summary.max, f32::NEG_INFINITY);
        assert_eq!(summary.sum, 0.0);
        assert_eq!(summary.l2, 0.0);
        assert!(summary.first8.is_empty());
    }

    #[test]
    fn format_summary_is_stable() {
        let text = format_summary("logits", &[1.0, 2.0]);
        assert!(text.contains("logits.length = 2\n"));
        assert!(text.contains("logits.min = 1.0\n"));
        assert!(text.contains("logits.max = 2.0\n"));
        assert!(text.contains("logits.sum = 3.0\n"));
        assert!(text.contains("logits.first8 = [1.0, 2.0]\n"));
    }

    #[test]
    fn top_k_orders_by_logit_then_id() {
        assert_eq!(
            top_k(&[1.0, 5.0, 3.0, 5.0], 3),
            [(1, 5.0), (3, 5.0), (2, 3.0)]
        );
        assert_eq!(top_k(&[2.0], 8).len(), 1);
        assert!(top_k(&[1.0], 0).is_empty());
    }
}
