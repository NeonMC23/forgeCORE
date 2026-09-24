//! Attention/GQA contract tests: mapping tables, routed values, history.
use forge_core::attention::{attention, AttentionDims};

#[test]
fn gqa_mapping_tables_are_explicit() {
    // 8 query heads / 2 KV heads: groups of four.
    let dims = AttentionDims::new(0, 8, 2, 4);
    let mapped: Vec<usize> = (0..8).map(|h| dims.kv_head_for(h).unwrap()).collect();
    assert_eq!(mapped, [0, 0, 0, 0, 1, 1, 1, 1]);

    // 4 query heads / 2 KV heads: pairs.
    let dims = AttentionDims::new(0, 4, 2, 4);
    let mapped: Vec<usize> = (0..4).map(|h| dims.kv_head_for(h).unwrap()).collect();
    assert_eq!(mapped, [0, 0, 1, 1]);

    // 2 query heads / 1 KV head: everything routes to head 0.
    let dims = AttentionDims::new(0, 2, 1, 4);
    let mapped: Vec<usize> = (0..2).map(|h| dims.kv_head_for(h).unwrap()).collect();
    assert_eq!(mapped, [0, 0]);

    // Qwen2.5-1.5B shape: 12 query heads / 2 KV heads: groups of six.
    let dims = AttentionDims::new(0, 12, 2, 128);
    let mapped: Vec<usize> = (0..12).map(|h| dims.kv_head_for(h).unwrap()).collect();
    assert_eq!(mapped, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
}

#[test]
fn gqa_rejects_indivisible_and_empty_configs() {
    assert!(AttentionDims::new(0, 3, 2, 8).validate().is_err());
    assert!(AttentionDims::new(0, 7, 3, 8).validate().is_err());
    assert!(AttentionDims::new(0, 0, 1, 8).validate().is_err());
    assert!(AttentionDims::new(0, 4, 0, 8).validate().is_err());
    assert!(AttentionDims::new(0, 4, 2, 0).validate().is_err());
}

#[test]
fn each_query_head_reads_its_own_kv_head() {
    // 4 query heads / 2 KV heads, head_dim 2, no history. With a single
    // visible position every head's weight is 1.0, so head h must reproduce
    // the value vector of kv_head_for(h). Distinct values per KV head make
    // any misrouting visible.
    let q = vec![0.5f32; 4 * 2];
    let k_current = vec![1.0, 0.0, 0.0, 1.0];
    let v_current = vec![10.0, 11.0, 20.0, 21.0];
    let out = attention(
        &q,
        &[],
        &[],
        &k_current,
        &v_current,
        AttentionDims::new(0, 4, 2, 2),
    )
    .unwrap();
    assert_eq!(out.hidden, [10.0, 11.0, 10.0, 11.0, 20.0, 21.0, 20.0, 21.0]);
}

#[test]
fn history_and_current_form_one_causal_sequence() {
    // 1 head, head_dim 2, two history positions plus current. Scores are
    // hand-computed scaled dots; probabilities must sum to 1.
    let q = [1.0, 2.0];
    let k_history = [1.0, 0.0, 0.0, 1.0];
    let v_history = [1.0, 1.0, 2.0, 2.0];
    let k_current = [1.0, 1.0];
    let v_current = [3.0, 3.0];
    let out = attention(
        &q,
        &k_history,
        &v_history,
        &k_current,
        &v_current,
        AttentionDims::new(2, 1, 1, 2),
    )
    .unwrap();
    let scale = 2f32.sqrt();
    assert!((out.scores[0] - 1.0 / scale).abs() < 1e-6);
    assert!((out.scores[1] - 2.0 / scale).abs() < 1e-6);
    assert!((out.scores[2] - 3.0 / scale).abs() < 1e-6);
    let prob_sum: f32 = out.probs.iter().sum();
    assert!((prob_sum - 1.0).abs() < 1e-6);
    // Output is the prob-weighted value sum; both lanes are identical here.
    let expected = out.probs[0] * 1.0 + out.probs[1] * 2.0 + out.probs[2] * 3.0;
    assert!((out.hidden[0] - expected).abs() < 1e-5);
    assert!((out.hidden[1] - expected).abs() < 1e-5);
}

#[test]
fn history_order_is_significant() {
    // Swapping the two history positions must change the scores order.
    let q = [1.0, 0.0];
    let v = [0.0, 0.0, 0.0, 0.0];
    let base = attention(
        &q,
        &[1.0, 0.0, 0.0, 1.0],
        &v,
        &[0.0, 0.0],
        &[0.0, 0.0],
        AttentionDims::new(2, 1, 1, 2),
    )
    .unwrap();
    let swapped = attention(
        &q,
        &[0.0, 1.0, 1.0, 0.0],
        &v,
        &[0.0, 0.0],
        &[0.0, 0.0],
        AttentionDims::new(2, 1, 1, 2),
    )
    .unwrap();
    assert_eq!(base.scores.len(), 3);
    assert_eq!(swapped.scores.len(), 3);
    assert!((base.scores[0] - swapped.scores[1]).abs() < 1e-7);
    assert!((base.scores[1] - swapped.scores[0]).abs() < 1e-7);
    assert_ne!(base.scores, swapped.scores);
}
