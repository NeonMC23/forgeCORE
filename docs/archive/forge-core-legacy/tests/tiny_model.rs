//! Tiny end-to-end reference model validated against an independent oracle.
//!
//! The fixture is a 2-layer decoder (hidden 8, vocab 16, 4 query heads,
//! 2 KV heads, head_dim 2, intermediate 12) with deterministic weights.
//! Layer 0 carries Q/K/V biases; layer 1 does not.
//!
//! Expected values come from the F64 oracle below, which re-implements the
//! transformer mathematics with its own plain loops in F64. It shares only
//! the weight values (cast exactly from F32) with the implementation under
//! test — never its functions — so agreement corroborates the F32 engine
//! rather than repeating it.
use forge_core::checkpoint::{format_summary, summarize, top_k};
use forge_core::kv::KvCache;
use forge_core::model::{
    forward_token, prefill, LayerTrace, LayerWeights, ModelDims, ModelWeights, TokenTrace,
};
use forge_core::shape::{MatrixF32, MatrixShape};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

fn fixture_dims() -> ModelDims {
    ModelDims {
        vocab_size: 16,
        hidden_size: 8,
        num_layers: 2,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 2,
        intermediate_size: 12,
        rms_eps: 1e-5,
        rope_theta: 10_000.0,
        context_length: 16,
    }
}

/// Deterministic weight in [-1, 1] from an index and a per-tensor salt.
fn weight_value(index: usize, salt: usize) -> f32 {
    (((index * 37 + salt * 101) % 89) as f32 / 44.0) - 1.0
}

fn fixture_matrix(input: usize, output: usize, salt: usize) -> MatrixF32 {
    MatrixF32::new(
        MatrixShape::new(input, output),
        (0..input * output).map(|i| weight_value(i, salt)).collect(),
    )
    .unwrap()
}

fn fixture_vector(len: usize, salt: usize) -> Vec<f32> {
    (0..len).map(|i| weight_value(i, salt)).collect()
}

fn fixture_layer(with_bias: bool, salt_base: usize) -> LayerWeights {
    let (q_bias, k_bias, v_bias) = if with_bias {
        (
            Some(fixture_vector(8, salt_base + 100)),
            Some(fixture_vector(4, salt_base + 101)),
            Some(fixture_vector(4, salt_base + 102)),
        )
    } else {
        (None, None, None)
    };
    LayerWeights {
        attn_norm: fixture_vector(8, salt_base),
        wq: fixture_matrix(8, 8, salt_base + 1),
        wk: fixture_matrix(8, 4, salt_base + 2),
        wv: fixture_matrix(8, 4, salt_base + 3),
        wo: fixture_matrix(8, 8, salt_base + 4),
        ffn_norm: fixture_vector(8, salt_base + 5),
        wg: fixture_matrix(8, 12, salt_base + 6),
        wu: fixture_matrix(8, 12, salt_base + 7),
        wd: fixture_matrix(12, 8, salt_base + 8),
        q_bias,
        k_bias,
        v_bias,
    }
}

fn fixture_model() -> (ModelDims, ModelWeights) {
    let dims = fixture_dims();
    let weights = ModelWeights {
        token_embd: fixture_matrix(8, 16, 1000),
        layers: vec![fixture_layer(true, 0), fixture_layer(false, 500)],
        output_norm: fixture_vector(8, 2000),
        output: fixture_matrix(8, 16, 3000),
    };
    weights.validate(&dims).unwrap();
    (dims, weights)
}

// ---------------------------------------------------------------------------
// Independent F64 oracle (own loops, own arithmetic, no forge_core calls)
// ---------------------------------------------------------------------------

struct OracleLayer {
    attn_norm: Vec<f64>,
    wq: Vec<f64>,
    wk: Vec<f64>,
    wv: Vec<f64>,
    wo: Vec<f64>,
    ffn_norm: Vec<f64>,
    wg: Vec<f64>,
    wu: Vec<f64>,
    wd: Vec<f64>,
    q_bias: Option<Vec<f64>>,
    k_bias: Option<Vec<f64>>,
    v_bias: Option<Vec<f64>>,
}

struct OracleModel {
    token_embd: Vec<f64>,
    layers: Vec<OracleLayer>,
    output_norm: Vec<f64>,
    output: Vec<f64>,
    k_histories: Vec<Vec<f64>>,
    v_histories: Vec<Vec<f64>>,
}

fn to_f64(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

impl OracleModel {
    fn new(weights: &ModelWeights) -> Self {
        let layers = weights
            .layers
            .iter()
            .map(|layer| OracleLayer {
                attn_norm: to_f64(&layer.attn_norm),
                wq: to_f64(&layer.wq.data),
                wk: to_f64(&layer.wk.data),
                wv: to_f64(&layer.wv.data),
                wo: to_f64(&layer.wo.data),
                ffn_norm: to_f64(&layer.ffn_norm),
                wg: to_f64(&layer.wg.data),
                wu: to_f64(&layer.wu.data),
                wd: to_f64(&layer.wd.data),
                q_bias: layer.q_bias.as_deref().map(to_f64),
                k_bias: layer.k_bias.as_deref().map(to_f64),
                v_bias: layer.v_bias.as_deref().map(to_f64),
            })
            .collect::<Vec<_>>();
        let per_layer = layers.len();
        Self {
            token_embd: to_f64(&weights.token_embd.data),
            layers,
            output_norm: to_f64(&weights.output_norm),
            output: to_f64(&weights.output.data),
            k_histories: vec![Vec::new(); per_layer],
            v_histories: vec![Vec::new(); per_layer],
        }
    }

    fn matvec(matrix: &[f64], input: usize, output: usize, x: &[f64]) -> Vec<f64> {
        assert_eq!(matrix.len(), input * output);
        assert_eq!(x.len(), input);
        let mut y = vec![0.0f64; output];
        for row in 0..output {
            let mut sum = 0.0f64;
            for col in 0..input {
                sum += matrix[row * input + col] * x[col];
            }
            y[row] = sum;
        }
        y
    }

    fn rms_norm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
        let mut squares = 0.0f64;
        for &value in x {
            squares += value * value;
        }
        let rms = (squares / x.len() as f64 + eps).sqrt();
        x.iter().zip(w.iter()).map(|(&a, &b)| a / rms * b).collect()
    }

    fn rope(vector: &mut [f64], position: usize, head_dim: usize, heads: usize, theta: f64) {
        assert_eq!(vector.len(), heads * head_dim);
        for head in 0..heads {
            let base = head * head_dim;
            for pair in 0..head_dim / 2 {
                let angle = theta.powf(-2.0 * pair as f64 / head_dim as f64) * position as f64;
                let (sin, cos) = angle.sin_cos();
                let first = vector[base + pair];
                let second = vector[base + pair + head_dim / 2];
                vector[base + pair] = first * cos - second * sin;
                vector[base + pair + head_dim / 2] = first * sin + second * cos;
            }
        }
    }

    fn forward_token(&mut self, dims: &ModelDims, token: u32, position: usize) -> OracleTokenTrace {
        assert_eq!(
            position,
            self.k_histories[0].len() / (dims.kv_heads * dims.head_dim)
        );
        let hidden_size = dims.hidden_size;
        let q_width = dims.query_heads * dims.head_dim;
        let kv_width = dims.kv_heads * dims.head_dim;
        let inter = dims.intermediate_size;

        let token = token as usize;
        let embedding = self.token_embd[token * hidden_size..(token + 1) * hidden_size].to_vec();

        let mut hidden = embedding.clone();
        let mut layers = Vec::new();
        let mut staged_k = Vec::new();
        let mut staged_v = Vec::new();

        for (index, layer) in self.layers.iter().enumerate() {
            let normed = Self::rms_norm(&hidden, &layer.attn_norm, f64::from(dims.rms_eps));
            let mut q = Self::matvec(&layer.wq, hidden_size, q_width, &normed);
            let mut k = Self::matvec(&layer.wk, hidden_size, kv_width, &normed);
            let mut v = Self::matvec(&layer.wv, hidden_size, kv_width, &normed);
            if let (Some(qb), Some(kb), Some(vb)) = (&layer.q_bias, &layer.k_bias, &layer.v_bias) {
                for (value, bias) in q.iter_mut().zip(qb.iter()) {
                    *value += *bias;
                }
                for (value, bias) in k.iter_mut().zip(kb.iter()) {
                    *value += *bias;
                }
                for (value, bias) in v.iter_mut().zip(vb.iter()) {
                    *value += *bias;
                }
            }
            let mut q_rope = q.clone();
            let mut k_rope = k.clone();
            Self::rope(
                &mut q_rope,
                position,
                dims.head_dim,
                dims.query_heads,
                f64::from(dims.rope_theta),
            );
            Self::rope(
                &mut k_rope,
                position,
                dims.head_dim,
                dims.kv_heads,
                f64::from(dims.rope_theta),
            );

            // Causal attention over oracle history plus current K/V.
            let history_len = position;
            let total = history_len + 1;
            let mut scores = vec![0.0f64; dims.query_heads * total];
            let mut probs = vec![0.0f64; dims.query_heads * total];
            let mut attn_out = vec![0.0f64; q_width];
            let scale = (dims.head_dim as f64).sqrt();
            for query_head in 0..dims.query_heads {
                let kv_head = query_head * dims.kv_heads / dims.query_heads;
                for pos in 0..total {
                    let k_all: &[f64] = if pos < history_len {
                        let start = pos * kv_width;
                        &self.k_histories[index][start..start + kv_width]
                    } else {
                        &k_rope
                    };
                    let mut dot = 0.0f64;
                    for lane in 0..dims.head_dim {
                        dot += q_rope[query_head * dims.head_dim + lane]
                            * k_all[kv_head * dims.head_dim + lane];
                    }
                    scores[query_head * total + pos] = dot / scale;
                }
                let row = &mut probs[query_head * total..(query_head + 1) * total];
                row.copy_from_slice(&scores[query_head * total..(query_head + 1) * total]);
                let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0f64;
                for value in row.iter_mut() {
                    *value = (*value - max).exp();
                    sum += *value;
                }
                for value in row.iter_mut() {
                    *value /= sum;
                }
                for pos in 0..total {
                    let v_all: &[f64] = if pos < history_len {
                        let start = pos * kv_width;
                        &self.v_histories[index][start..start + kv_width]
                    } else {
                        &v
                    };
                    for lane in 0..dims.head_dim {
                        attn_out[query_head * dims.head_dim + lane] +=
                            probs[query_head * total + pos] * v_all[kv_head * dims.head_dim + lane];
                    }
                }
            }

            let proj = Self::matvec(&layer.wo, q_width, hidden_size, &attn_out);
            let residual: Vec<f64> = hidden
                .iter()
                .zip(proj.iter())
                .map(|(&a, &b)| a + b)
                .collect();
            let ffn_normed = Self::rms_norm(&residual, &layer.ffn_norm, f64::from(dims.rms_eps));
            let gate = Self::matvec(&layer.wg, hidden_size, inter, &ffn_normed);
            let up = Self::matvec(&layer.wu, hidden_size, inter, &ffn_normed);
            let swiglu: Vec<f64> = gate
                .iter()
                .zip(up.iter())
                .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let ffn_out = Self::matvec(&layer.wd, inter, hidden_size, &swiglu);
            let output: Vec<f64> = residual
                .iter()
                .zip(ffn_out.iter())
                .map(|(&a, &b)| a + b)
                .collect();

            staged_k.push(k_rope.clone());
            staged_v.push(v.clone());
            layers.push(OracleLayerTrace {
                input: hidden.clone(),
                normed,
                q,
                k,
                v,
                q_rope,
                k_rope,
                scores,
                probs,
                attn_out,
                proj,
                residual,
                ffn_normed,
                gate,
                up,
                swiglu,
                ffn_out,
                output: output.clone(),
            });
            hidden = output;
        }

        for ((history_k, history_v), (k, v)) in self
            .k_histories
            .iter_mut()
            .zip(self.v_histories.iter_mut())
            .zip(staged_k.iter().zip(staged_v.iter()))
        {
            history_k.extend_from_slice(k);
            history_v.extend_from_slice(v);
        }

        let final_norm = Self::rms_norm(&hidden, &self.output_norm, f64::from(dims.rms_eps));
        let logits = Self::matvec(&self.output, hidden_size, dims.vocab_size, &final_norm);

        OracleTokenTrace {
            embedding,
            layers,
            final_norm,
            logits,
        }
    }
}

struct OracleLayerTrace {
    input: Vec<f64>,
    normed: Vec<f64>,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    q_rope: Vec<f64>,
    k_rope: Vec<f64>,
    scores: Vec<f64>,
    probs: Vec<f64>,
    attn_out: Vec<f64>,
    proj: Vec<f64>,
    residual: Vec<f64>,
    ffn_normed: Vec<f64>,
    gate: Vec<f64>,
    up: Vec<f64>,
    swiglu: Vec<f64>,
    ffn_out: Vec<f64>,
    output: Vec<f64>,
}

struct OracleTokenTrace {
    embedding: Vec<f64>,
    layers: Vec<OracleLayerTrace>,
    final_norm: Vec<f64>,
    logits: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

fn assert_close(name: &str, actual: &[f32], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{name}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    let mut max_diff = 0.0f64;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_diff = max_diff.max((f64::from(*a) - *e).abs());
    }
    let magnitude = expected.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let tolerance = 2e-4 + 2e-4 * magnitude;
    assert!(
        max_diff <= tolerance,
        "{name}: max abs diff {max_diff:.3e} exceeds tolerance {tolerance:.3e}"
    );
}

fn assert_layer_close(
    position: usize,
    index: usize,
    actual: &LayerTrace,
    expected: &OracleLayerTrace,
) {
    let tag = format!("pos {position} layer {index}");
    assert_close(&format!("{tag} input"), &actual.input, &expected.input);
    assert_close(&format!("{tag} normed"), &actual.normed, &expected.normed);
    assert_close(&format!("{tag} q"), &actual.q, &expected.q);
    assert_close(&format!("{tag} k"), &actual.k, &expected.k);
    assert_close(&format!("{tag} v"), &actual.v, &expected.v);
    assert_close(&format!("{tag} q_rope"), &actual.q_rope, &expected.q_rope);
    assert_close(&format!("{tag} k_rope"), &actual.k_rope, &expected.k_rope);
    assert_close(&format!("{tag} scores"), &actual.scores, &expected.scores);
    assert_close(&format!("{tag} probs"), &actual.probs, &expected.probs);
    assert_close(
        &format!("{tag} attn_out"),
        &actual.attn_out,
        &expected.attn_out,
    );
    assert_close(&format!("{tag} proj"), &actual.proj, &expected.proj);
    assert_close(
        &format!("{tag} residual"),
        &actual.residual,
        &expected.residual,
    );
    assert_close(
        &format!("{tag} ffn_normed"),
        &actual.ffn_normed,
        &expected.ffn_normed,
    );
    assert_close(&format!("{tag} gate"), &actual.gate, &expected.gate);
    assert_close(&format!("{tag} up"), &actual.up, &expected.up);
    assert_close(&format!("{tag} swiglu"), &actual.swiglu, &expected.swiglu);
    assert_close(
        &format!("{tag} ffn_out"),
        &actual.ffn_out,
        &expected.ffn_out,
    );
    assert_close(&format!("{tag} output"), &actual.output, &expected.output);
}

fn assert_token_close(position: usize, actual: &TokenTrace, expected: &OracleTokenTrace) {
    assert_close(
        &format!("pos {position} embedding"),
        &actual.embedding,
        &expected.embedding,
    );
    assert_eq!(actual.layers.len(), expected.layers.len());
    for (index, (a, e)) in actual.layers.iter().zip(expected.layers.iter()).enumerate() {
        assert_layer_close(position, index, a, e);
    }
    assert_close(
        &format!("pos {position} final_norm"),
        &actual.final_norm,
        &expected.final_norm,
    );
    assert_close(
        &format!("pos {position} logits"),
        &actual.logits,
        &expected.logits,
    );
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, &value) in values.iter().enumerate() {
        if value > values[best] {
            best = index;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn tiny_model_matches_f64_oracle_on_prefill_and_decode() {
    let (dims, weights) = fixture_model();
    let mut kv = KvCache::new(
        dims.num_layers,
        dims.kv_heads,
        dims.head_dim,
        dims.context_length,
    )
    .unwrap();
    let mut oracle = OracleModel::new(&weights);

    // Prefill three tokens; every intermediate of every layer is compared.
    let prompt = [1u32, 5, 3];
    let mut last_actual = None;
    for (position, &token) in prompt.iter().enumerate() {
        let actual = forward_token(&weights, &dims, token, position, &mut kv).unwrap();
        let expected = oracle.forward_token(&dims, token, position);
        assert_token_close(position, &actual, &expected);
        last_actual = Some(actual);
    }
    assert_eq!(kv.seq_len(), 3);

    // A forward at the wrong position must be rejected, not silently run.
    assert!(forward_token(&weights, &dims, prompt[2], 99, &mut kv).is_err());

    // One greedy decode step on the live cache exercises history reads
    // with three committed positions.
    let next = argmax(&last_actual.expect("prefill ran").logits) as u32;
    let actual = forward_token(&weights, &dims, next, 3, &mut kv).unwrap();
    let expected = oracle.forward_token(&dims, next, 3);
    assert_token_close(3, &actual, &expected);
    assert_eq!(kv.seq_len(), 4);
}

#[test]
fn prefill_equivalence_and_checkpoints() {
    let (dims, weights) = fixture_model();
    let prompt = [1u32, 5, 3];

    // prefill() must equal sequential forward_token() calls.
    let mut kv_bulk = KvCache::new(2, 2, 2, 16).unwrap();
    let bulk = prefill(&weights, &dims, &prompt, &mut kv_bulk).unwrap();
    let mut kv_seq = KvCache::new(2, 2, 2, 16).unwrap();
    let mut seq = Vec::new();
    for (position, &token) in prompt.iter().enumerate() {
        seq.push(forward_token(&weights, &dims, token, position, &mut kv_seq).unwrap());
    }
    assert_eq!(bulk, seq);
    assert_eq!(kv_bulk.seq_len(), 3);

    // Checkpoint vocabulary smoke test on real logits.
    let logits = &bulk[2].logits;
    assert_eq!(logits.len(), dims.vocab_size);
    let summary = summarize(logits);
    assert_eq!(summary.len, 16);
    assert!(summary.min <= summary.max);
    let text = format_summary("logits", logits);
    assert!(text.contains("logits.length = 16\n"));
    let ranked = top_k(logits, 3);
    assert_eq!(ranked.len(), 3);
    assert_eq!(ranked[0].0, argmax(logits));
    assert!(ranked[0].1 >= ranked[1].1 && ranked[1].1 >= ranked[2].1);
}
