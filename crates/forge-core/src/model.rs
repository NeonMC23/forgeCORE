//! Reference transformer execution for dense decoder layers.
//!
//! ## Execution order
//!
//! One decoder layer (Qwen2/Qwen2.5-shaped) executes exactly this sequence:
//!
//! ```text
//! input hidden
//! -> attention RMSNorm
//! -> Q/K/V projections
//! -> optional Q/K/V biases (all three present, or none)
//! -> RoPE on Q and K
//! -> KV-cache history read + single-token causal GQA attention
//! -> attention output projection
//! -> attention residual add
//! -> FFN RMSNorm
//! -> gate and up projections
//! -> SwiGLU: SiLU(gate) * up
//! -> down projection
//! -> FFN residual add
//! ```
//!
//! A token forward embeds one token, runs every layer in order (each layer
//! reading the same committed KV history), stages every layer's current K/V,
//! commits the KV history once, then applies the final RMSNorm and the
//! output projection to produce logits.
//!
//! ## Weight layouts
//!
//! All matrices follow [`MatrixShape`](crate::shape::MatrixShape):
//! logical `[input, output]`, physical row-major `[output][input]`.
//! `token_embd` and `output` are `[hidden, vocab]`: row `t` holds the
//! `hidden`-wide vector for vocabulary id `t`.

use crate::attention::{attention, AttentionDims};
use crate::error::{Error, Result};
use crate::kv::KvCache;
use crate::ops::{add, matvec, rms_norm, rope, swiglu};
use crate::shape::{MatrixF32, MatrixShape};

/// Fixed diagnostic prompt recorded for the Qwen2.5 investigation.
///
/// Historical provenance only (documentation, not a code dependency):
/// the RAMforge handoff (`QWEN25_CORRECTNESS_INVESTIGATION.md`).
/// These IDs are valid only for the recorded Qwen2.5 tokenizer; they are
/// kept here so future real-model validation fixtures can reference them.
pub const QWEN25_DIAGNOSTIC_PROMPT: &str = "hi, what's 2+2=?";

/// Token IDs of [`QWEN25_DIAGNOSTIC_PROMPT`] under the recorded tokenizer.
pub const QWEN25_DIAGNOSTIC_PROMPT_IDS: [u32; 9] = [6023, 11, 1128, 594, 220, 17, 10, 17, 19884];

/// Complete dimensional configuration of a dense decoder model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelDims {
    /// Vocabulary size (embedding rows / logit width).
    pub vocab_size: usize,
    /// Hidden/embedding width.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Query heads per layer.
    pub query_heads: usize,
    /// KV heads per layer; must divide `query_heads`.
    pub kv_heads: usize,
    /// Width of one attention head.
    pub head_dim: usize,
    /// Feed-forward intermediate width.
    pub intermediate_size: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// RoPE frequency base (theta).
    pub rope_theta: f32,
    /// Maximum context length (informational for the reference core).
    pub context_length: usize,
}

impl ModelDims {
    /// Query width: `query_heads * head_dim`.
    pub fn q_width(&self) -> Result<usize> {
        self.query_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error("query width overflows".to_string()))
    }

    /// KV width: `kv_heads * head_dim`.
    pub fn kv_width(&self) -> Result<usize> {
        self.kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error("KV width overflows".to_string()))
    }

    /// Validate every dimensional relationship the executor relies on.
    pub fn validate(&self) -> Result<()> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.num_layers == 0
            || self.query_heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || self.intermediate_size == 0
            || self.context_length == 0
        {
            return Err(Error(format!(
                "model dimensions must all be non-zero: {self:?}"
            )));
        }
        if !self.query_heads.is_multiple_of(self.kv_heads) {
            return Err(Error(format!(
                "GQA requires query_heads {} divisible by kv_heads {}",
                self.query_heads, self.kv_heads
            )));
        }
        if self.q_width()? != self.hidden_size {
            return Err(Error(format!(
                "query width {} must equal hidden size {}",
                self.q_width()?,
                self.hidden_size
            )));
        }
        if !self.rms_eps.is_finite() || self.rms_eps < 0.0 {
            return Err(Error(
                "RMSNorm epsilon must be finite and non-negative".to_string(),
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Error(
                "RoPE frequency base must be finite and positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Recorded Qwen2.5-1.5B skeleton dimensions.
    ///
    /// Historical provenance only (documentation, not a code dependency):
    /// the RAMforge handoff. `vocab_size`, `hidden_size`, and
    /// `num_layers` were asserted by the recorded diagnostic; the remaining
    /// values are recorded model configuration that must be re-verified
    /// against the target GGUF's metadata before real-model use.
    pub const fn qwen25_15b() -> Self {
        Self {
            vocab_size: 151_936,
            hidden_size: 1536,
            num_layers: 28,
            query_heads: 12,
            kv_heads: 2,
            head_dim: 128,
            intermediate_size: 8960,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
            context_length: 32_768,
        }
    }
}

/// F32 weights of one decoder layer.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    /// Attention RMSNorm weight, `[hidden]`.
    pub attn_norm: Vec<f32>,
    /// Q projection, `[hidden, query_heads * head_dim]`.
    pub wq: MatrixF32,
    /// K projection, `[hidden, kv_heads * head_dim]`.
    pub wk: MatrixF32,
    /// V projection, `[hidden, kv_heads * head_dim]`.
    pub wv: MatrixF32,
    /// Attention output projection, `[query_heads * head_dim, hidden]`.
    pub wo: MatrixF32,
    /// FFN RMSNorm weight, `[hidden]`.
    pub ffn_norm: Vec<f32>,
    /// FFN gate projection, `[hidden, intermediate]`.
    pub wg: MatrixF32,
    /// FFN up projection, `[hidden, intermediate]`.
    pub wu: MatrixF32,
    /// FFN down projection, `[intermediate, hidden]`.
    pub wd: MatrixF32,
    /// Optional Qwen2-style Q/K/V biases. All three must be present or all
    /// absent; a partial set is rejected. Applied after projection, before
    /// RoPE and KV-cache insertion.
    pub q_bias: Option<Vec<f32>>,
    /// K bias, `[kv_heads * head_dim]`. See [`LayerWeights::q_bias`].
    pub k_bias: Option<Vec<f32>>,
    /// V bias, `[kv_heads * head_dim]`. See [`LayerWeights::q_bias`].
    pub v_bias: Option<Vec<f32>>,
}

impl LayerWeights {
    /// Validate every tensor shape against `dims`.
    pub fn validate(&self, dims: &ModelDims, layer: usize) -> Result<()> {
        dims.validate()?;
        let hidden = dims.hidden_size;
        let q_width = dims.q_width()?;
        let kv_width = dims.kv_width()?;
        let inter = dims.intermediate_size;

        if self.attn_norm.len() != hidden {
            return Err(Error(format!(
                "layer {layer} attn_norm length {} != hidden {hidden}",
                self.attn_norm.len()
            )));
        }
        if self.ffn_norm.len() != hidden {
            return Err(Error(format!(
                "layer {layer} ffn_norm length {} != hidden {hidden}",
                self.ffn_norm.len()
            )));
        }
        let expect = [
            ("wq", &self.wq, MatrixShape::new(hidden, q_width)),
            ("wk", &self.wk, MatrixShape::new(hidden, kv_width)),
            ("wv", &self.wv, MatrixShape::new(hidden, kv_width)),
            ("wo", &self.wo, MatrixShape::new(q_width, hidden)),
            ("wg", &self.wg, MatrixShape::new(hidden, inter)),
            ("wu", &self.wu, MatrixShape::new(hidden, inter)),
            ("wd", &self.wd, MatrixShape::new(inter, hidden)),
        ];
        for (name, matrix, want) in expect {
            if matrix.shape != want {
                return Err(Error(format!(
                    "layer {layer} {name} shape {:?} != expected {want:?}",
                    matrix.shape
                )));
            }
        }
        match (&self.q_bias, &self.k_bias, &self.v_bias) {
            (None, None, None) => Ok(()),
            (Some(q), Some(k), Some(v)) => {
                if q.len() != q_width || k.len() != kv_width || v.len() != kv_width {
                    return Err(Error(format!(
                        "layer {layer} bias lengths q {}, k {}, v {} != q {q_width}, kv {kv_width}",
                        q.len(),
                        k.len(),
                        v.len()
                    )));
                }
                Ok(())
            }
            _ => Err(Error(format!(
                "layer {layer} has a partial Q/K/V bias set; all three or none are required"
            ))),
        }
    }
}

/// F32 weights of a complete dense decoder model.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Token embedding, `[hidden, vocab]`; row `t` is token `t`.
    pub token_embd: MatrixF32,
    /// One weight set per layer.
    pub layers: Vec<LayerWeights>,
    /// Final output RMSNorm weight, `[hidden]`.
    pub output_norm: Vec<f32>,
    /// Output projection, `[hidden, vocab]`; row `t` scores token `t`.
    pub output: MatrixF32,
}

impl ModelWeights {
    /// Validate persistent tensors and every layer against `dims`.
    pub fn validate(&self, dims: &ModelDims) -> Result<()> {
        dims.validate()?;
        let want = MatrixShape::new(dims.hidden_size, dims.vocab_size);
        if self.token_embd.shape != want {
            return Err(Error(format!(
                "token_embd shape {:?} != expected {want:?}",
                self.token_embd.shape
            )));
        }
        if self.output.shape != want {
            return Err(Error(format!(
                "output shape {:?} != expected {want:?}",
                self.output.shape
            )));
        }
        if self.output_norm.len() != dims.hidden_size {
            return Err(Error(format!(
                "output_norm length {} != hidden {}",
                self.output_norm.len(),
                dims.hidden_size
            )));
        }
        if self.layers.len() != dims.num_layers {
            return Err(Error(format!(
                "layer count {} != num_layers {}",
                self.layers.len(),
                dims.num_layers
            )));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            layer.validate(dims, index)?;
        }
        Ok(())
    }
}

/// Every intermediate of one layer execution, for validation checkpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerTrace {
    /// Input hidden state.
    pub input: Vec<f32>,
    /// After attention RMSNorm.
    pub normed: Vec<f32>,
    /// Q/K/V projections before bias and RoPE.
    pub q: Vec<f32>,
    /// K projection before bias and RoPE.
    pub k: Vec<f32>,
    /// V projection before bias.
    pub v: Vec<f32>,
    /// Q after RoPE.
    pub q_rope: Vec<f32>,
    /// K after RoPE (this is the K staged into the KV cache).
    pub k_rope: Vec<f32>,
    /// Raw attention scores, `[query_head][history_len + 1]`.
    pub scores: Vec<f32>,
    /// Post-softmax attention weights, `[query_head][history_len + 1]`.
    pub probs: Vec<f32>,
    /// Attention output before the output projection.
    pub attn_out: Vec<f32>,
    /// Attention output projection.
    pub proj: Vec<f32>,
    /// Hidden state after the attention residual.
    pub residual: Vec<f32>,
    /// After FFN RMSNorm.
    pub ffn_normed: Vec<f32>,
    /// Gate projection.
    pub gate: Vec<f32>,
    /// Up projection.
    pub up: Vec<f32>,
    /// SwiGLU output: `SiLU(gate) * up`.
    pub swiglu: Vec<f32>,
    /// FFN down projection.
    pub ffn_out: Vec<f32>,
    /// Layer output after the FFN residual.
    pub output: Vec<f32>,
}

/// Every intermediate of one token forward, for validation checkpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenTrace {
    /// Token id and position of this forward.
    pub token: u32,
    /// Position id of this forward.
    pub position: usize,
    /// Embedding lookup result.
    pub embedding: Vec<f32>,
    /// One trace per layer, in execution order.
    pub layers: Vec<LayerTrace>,
    /// Final hidden state after the output RMSNorm.
    pub final_norm: Vec<f32>,
    /// Logits over the vocabulary.
    pub logits: Vec<f32>,
}

/// Look up the embedding row for one token.
pub fn embed(weights: &ModelWeights, dims: &ModelDims, token: u32) -> Result<Vec<f32>> {
    dims.validate()?;
    let id =
        usize::try_from(token).map_err(|_| Error(format!("token id {token} exceeds usize")))?;
    if id >= dims.vocab_size {
        return Err(Error(format!(
            "token id {token} out of range for vocab {}",
            dims.vocab_size
        )));
    }
    Ok(weights.token_embd.row(id)?.to_vec())
}

/// Execute one decoder layer over `hidden` at `position`.
///
/// `kv` supplies committed history `[0, position)` for `layer_idx`; the
/// position must equal `kv.seq_len()`. This function only reads the cache:
/// the caller stages the returned `k_rope`/`v` for every layer and commits
/// once (see [`forward_token`]).
pub fn forward_layer(
    weights: &LayerWeights,
    dims: &ModelDims,
    layer_idx: usize,
    position: usize,
    hidden: &[f32],
    kv: &KvCache,
) -> Result<LayerTrace> {
    weights.validate(dims, layer_idx)?;
    if hidden.len() != dims.hidden_size {
        return Err(Error(format!(
            "layer {layer_idx} hidden length {} != hidden {}",
            hidden.len(),
            dims.hidden_size
        )));
    }
    if position != kv.seq_len() {
        return Err(Error(format!(
            "layer {layer_idx} position {position} != committed KV history {}",
            kv.seq_len()
        )));
    }
    let q_width = dims.q_width()?;
    let kv_width = dims.kv_width()?;

    // Attention RMSNorm, then Q/K/V projections.
    let mut normed = vec![0.0f32; dims.hidden_size];
    rms_norm(hidden, &weights.attn_norm, dims.rms_eps, &mut normed)?;
    let mut q = vec![0.0f32; q_width];
    let mut k = vec![0.0f32; kv_width];
    let mut v = vec![0.0f32; kv_width];
    matvec(&weights.wq.data, weights.wq.shape, &normed, &mut q)?;
    matvec(&weights.wk.data, weights.wk.shape, &normed, &mut k)?;
    matvec(&weights.wv.data, weights.wv.shape, &normed, &mut v)?;

    // Optional Qwen2-style biases, before RoPE and KV insertion.
    if let (Some(qb), Some(kb), Some(vb)) = (&weights.q_bias, &weights.k_bias, &weights.v_bias) {
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

    // RoPE on Q and K (V is untouched).
    let mut q_rope = q.clone();
    let mut k_rope = k.clone();
    rope(
        &mut q_rope,
        &mut k_rope,
        position,
        dims.head_dim,
        dims.query_heads,
        dims.kv_heads,
        dims.rope_theta,
    )?;

    // Causal GQA attention over committed history plus current K/V.
    let attn = attention(
        &q_rope,
        kv.k_history(layer_idx)?,
        kv.v_history(layer_idx)?,
        &k_rope,
        &v,
        AttentionDims::new(kv.seq_len(), dims.query_heads, dims.kv_heads, dims.head_dim),
    )?;

    // Output projection and attention residual.
    let mut proj = vec![0.0f32; dims.hidden_size];
    matvec(&weights.wo.data, weights.wo.shape, &attn.hidden, &mut proj)?;
    let mut residual = vec![0.0f32; dims.hidden_size];
    add(hidden, &proj, &mut residual)?;

    // FFN: norm, gate/up, SwiGLU, down, residual.
    let mut ffn_normed = vec![0.0f32; dims.hidden_size];
    rms_norm(&residual, &weights.ffn_norm, dims.rms_eps, &mut ffn_normed)?;
    let mut gate = vec![0.0f32; dims.intermediate_size];
    let mut up = vec![0.0f32; dims.intermediate_size];
    matvec(&weights.wg.data, weights.wg.shape, &ffn_normed, &mut gate)?;
    matvec(&weights.wu.data, weights.wu.shape, &ffn_normed, &mut up)?;
    let mut swiglu_out = vec![0.0f32; dims.intermediate_size];
    swiglu(&gate, &up, &mut swiglu_out)?;
    let mut ffn_out = vec![0.0f32; dims.hidden_size];
    matvec(
        &weights.wd.data,
        weights.wd.shape,
        &swiglu_out,
        &mut ffn_out,
    )?;
    let mut output = vec![0.0f32; dims.hidden_size];
    add(&residual, &ffn_out, &mut output)?;

    Ok(LayerTrace {
        input: hidden.to_vec(),
        normed,
        q,
        k,
        v,
        q_rope,
        k_rope,
        scores: attn.scores,
        probs: attn.probs,
        attn_out: attn.hidden,
        proj,
        residual,
        ffn_normed,
        gate,
        up,
        swiglu: swiglu_out,
        ffn_out,
        output,
    })
}

/// Forward one token at `position`, updating the KV cache.
///
/// Runs every layer against the same committed history, stages each layer's
/// current K/V, commits once, then applies the final RMSNorm and output
/// projection. `position` must equal `kv.seq_len()`.
pub fn forward_token(
    weights: &ModelWeights,
    dims: &ModelDims,
    token: u32,
    position: usize,
    kv: &mut KvCache,
) -> Result<TokenTrace> {
    weights.validate(dims)?;
    if kv.layers() != dims.num_layers
        || kv.kv_heads() != dims.kv_heads
        || kv.head_dim() != dims.head_dim
    {
        return Err(Error(format!(
            "KV cache geometry layers {}/heads {}/dim {} != model layers {}/heads {}/dim {}",
            kv.layers(),
            kv.kv_heads(),
            kv.head_dim(),
            dims.num_layers,
            dims.kv_heads,
            dims.head_dim
        )));
    }

    let embedding = embed(weights, dims, token)?;
    let mut hidden = embedding.clone();
    let mut layers = Vec::with_capacity(dims.num_layers);
    let mut staged: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(dims.num_layers);
    for (index, layer_weights) in weights.layers.iter().enumerate() {
        let trace = forward_layer(layer_weights, dims, index, position, &hidden, kv)?;
        staged.push((trace.k_rope.clone(), trace.v.clone()));
        hidden = trace.output.clone();
        layers.push(trace);
    }
    for (index, (k, v)) in staged.iter().enumerate() {
        kv.append(index, k, v)?;
    }
    kv.commit();

    let mut final_norm = vec![0.0f32; dims.hidden_size];
    rms_norm(&hidden, &weights.output_norm, dims.rms_eps, &mut final_norm)?;
    let mut logits = vec![0.0f32; dims.vocab_size];
    matvec(
        &weights.output.data,
        weights.output.shape,
        &final_norm,
        &mut logits,
    )?;

    Ok(TokenTrace {
        token,
        position,
        embedding,
        layers,
        final_norm,
        logits,
    })
}

/// Forward a prompt: token `i` runs at position `i`, in order.
pub fn prefill(
    weights: &ModelWeights,
    dims: &ModelDims,
    tokens: &[u32],
    kv: &mut KvCache,
) -> Result<Vec<TokenTrace>> {
    let mut traces = Vec::with_capacity(tokens.len());
    for (position, &token) in tokens.iter().enumerate() {
        traces.push(forward_token(weights, dims, token, position, kv)?);
    }
    Ok(traces)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_dims() -> ModelDims {
        ModelDims {
            vocab_size: 4,
            hidden_size: 4,
            num_layers: 1,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            intermediate_size: 8,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            context_length: 16,
        }
    }

    fn tiny_layer() -> LayerWeights {
        let matrix = |input: usize, output: usize| {
            MatrixF32::new(
                MatrixShape::new(input, output),
                vec![0.1f32; input * output],
            )
            .unwrap()
        };
        LayerWeights {
            attn_norm: vec![1.0; 4],
            wq: matrix(4, 4),
            wk: matrix(4, 2),
            wv: matrix(4, 2),
            wo: matrix(4, 4),
            ffn_norm: vec![1.0; 4],
            wg: matrix(4, 8),
            wu: matrix(4, 8),
            wd: matrix(8, 4),
            q_bias: None,
            k_bias: None,
            v_bias: None,
        }
    }

    #[test]
    fn dims_validation_catches_bad_geometry() {
        let good = tiny_dims();
        assert!(good.validate().is_ok());
        let bad_heads = ModelDims {
            query_heads: 3,
            ..good
        };
        assert!(bad_heads.validate().is_err());
        let bad_width = ModelDims {
            hidden_size: 6,
            ..good
        };
        assert!(bad_width.validate().is_err());
        let bad_eps = ModelDims {
            rms_eps: f32::NAN,
            ..good
        };
        assert!(bad_eps.validate().is_err());
    }

    #[test]
    fn qwen25_skeleton_is_self_consistent() {
        let dims = ModelDims::qwen25_15b();
        assert!(dims.validate().is_ok());
        assert_eq!(dims.hidden_size, 1536);
        assert_eq!(dims.vocab_size, 151_936);
        assert_eq!(dims.num_layers, 28);
        assert_eq!(dims.query_heads, 12);
        assert_eq!(dims.kv_heads, 2);
        assert_eq!(dims.head_dim, 128);
        assert_eq!(dims.q_width().unwrap(), dims.hidden_size);
        assert_eq!(QWEN25_DIAGNOSTIC_PROMPT_IDS.len(), 9);
        assert_eq!(QWEN25_DIAGNOSTIC_PROMPT, "hi, what's 2+2=?");
    }

    #[test]
    fn layer_validation_rejects_partial_biases() {
        let dims = tiny_dims();
        assert!(tiny_layer().validate(&dims, 0).is_ok());
        let mut partial = tiny_layer();
        partial.q_bias = Some(vec![0.0; 4]);
        assert!(partial.validate(&dims, 0).is_err());
        let mut bad_len = tiny_layer();
        bad_len.q_bias = Some(vec![0.0; 4]);
        bad_len.k_bias = Some(vec![0.0; 1]);
        bad_len.v_bias = Some(vec![0.0; 2]);
        assert!(bad_len.validate(&dims, 0).is_err());
    }

    #[test]
    fn embed_selects_token_row() {
        let dims = tiny_dims();
        let weights = ModelWeights {
            token_embd: MatrixF32::new(MatrixShape::new(4, 4), (0..16).map(|v| v as f32).collect())
                .unwrap(),
            layers: vec![tiny_layer()],
            output_norm: vec![1.0; 4],
            output: MatrixF32::new(MatrixShape::new(4, 4), vec![0.0; 16]).unwrap(),
        };
        assert_eq!(embed(&weights, &dims, 2).unwrap(), [8.0, 9.0, 10.0, 11.0]);
        assert!(embed(&weights, &dims, 4).is_err());
    }

    #[test]
    fn forward_rejects_position_and_cache_mismatch() {
        let dims = tiny_dims();
        let weights = ModelWeights {
            token_embd: MatrixF32::new(MatrixShape::new(4, 4), vec![0.1; 16]).unwrap(),
            layers: vec![tiny_layer()],
            output_norm: vec![1.0; 4],
            output: MatrixF32::new(MatrixShape::new(4, 4), vec![0.1; 16]).unwrap(),
        };
        let mut kv = KvCache::new(1, 1, 2, 4).unwrap();
        assert!(forward_token(&weights, &dims, 0, 1, &mut kv).is_err());
        let mut bad_kv = KvCache::new(1, 2, 2, 4).unwrap();
        assert!(forward_token(&weights, &dims, 0, 0, &mut bad_kv).is_err());
    }
}
