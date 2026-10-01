# ForgeCore numerical conventions

> Scope note (2026-10-01 ggml pivot): this document now governs the
> **frozen validation oracles** under `crates/forge-core/src/reference/`
> only. All real execution defers to upstream ggml semantics; where ggml
> disagrees with these conventions (e.g. `ggml_mul_mat` layout), ggml
> wins and the difference is documented at the call site. See
> `docs/PIVOT.md`.

This is the authoritative convention record for the ForgeCore reference
core. If code and this document ever disagree, that is a bug in one of
them — file it, do not silently reinterpret.

## 1. Matrices

- A 2-D GGUF tensor reports logical dimensions `[ne0, ne1]` where `ne0`
  is the **input** width and `ne1` is the **output** height
  (`MatrixShape { input, output }`, `shape::MatrixShape::from_gguf_ne`).
- Physical F32 storage is contiguous row-major `[output][input]`:
  row `o` is `data[o*input .. (o+1)*input]`.
- Reference operation: `y[o] = sum over i of W[o, i] * x[i]`,
  F32 accumulation in increasing `i`.
- Arity mismatches are errors. There is no transpose guessing.

## 2. Embeddings and output projection

- `token_embd` and `output` are `[hidden, vocab]` matrices: row `t`
  holds the `hidden`-wide vector for vocabulary id `t`.
- Logits: `logits[t] = dot(output_row[t], final_norm)`.

## 3. RMSNorm

`y = x / sqrt(mean(x^2) + eps) * w`, sum of squares in F32 in input
order, then divide, then elementwise multiply by the weight.

## 4. Activations

- SiLU: `x / (1 + exp(-x))`.
- SwiGLU: `SiLU(gate) * up`, elementwise.

## 5. Softmax

Max-subtracted stable softmax: subtract max, exponentiate, divide by the
sum. Empty input and non-finite/zero sums are errors.

## 6. RoPE

Half-split Llama/Qwen2 pairing (NOT interleaved/GPT-J): within a head of
width `d`, index `j` in `[0, d/2)` rotates the pair `(v[j], v[j+d/2])` by
`angle = position * theta^(-2j/d)`. `head_dim` must be even. Position 0
is the identity.

## 7. Attention

- Layouts: Q `[query_head][head_dim]`; K/V history
  `[position][kv_head][head_dim]` over exactly `history_len` positions;
  current K/V `[kv_head][head_dim]` read as position `history_len`.
- GQA: `query_heads % kv_heads == 0`; query head `h` reads KV head
  `h * kv_heads / query_heads` (validated on every call).
- Scores: `dot(q_h, k_p) / sqrt(head_dim)` over history plus current
  (single-token forward: every visible position is causal by
  construction).
- Head output: softmax-weighted sum of value vectors.
- **Open verification point (ulp-level):** division by `sqrt(d)` vs
  multiplication by a precomputed reciprocal is not pinned by the
  handoff; ForgeCore uses division. Confirm against the external oracle
  during real-model validation.

## 8. KV cache

- Per-layer flattened `[position][kv_head][head_dim]` F32 buffers.
- `seq_len` counts **committed** positions only. A forward at position
  `p` requires `p == seq_len`, reads history `[0, p)`, and returns its
  current K/V separately. The caller appends current K/V for **every**
  layer, then commits once. Staged-but-uncommitted entries are invisible
  to history reads.

## 9. Transformer layer order

```text
input -> attn RMSNorm -> Q/K/V proj -> optional Q/K/V biases
-> RoPE(Q, K) -> causal GQA attention -> output proj -> +residual
-> FFN RMSNorm -> gate/up proj -> SwiGLU -> down proj -> +residual
```

- Biases are all-or-none (Qwen2-style), applied after projection and
  before RoPE/KV insertion.
- RoPE touches Q and K only; V is untouched.

## 10. Token forward order

Embed one token → run layers 0..N against the same committed history →
stage each layer's current K/V → commit once → final RMSNorm → output
projection → logits. Prefill runs token `i` at position `i`, in order.

## 11. Checkpoints

Summaries report length, min, max, F64-accumulated sum, F64-accumulated
L2 norm, and the first eight values. `top_k` sorts by descending logit
with token id as the tie-break.

## 12. Quantization

- Block geometry is fixed per format (`quant::block_spec`); a matrix
  row of width `input` holds `input / values_per_block` consecutive
  blocks, so `input` must be an exact multiple of the block width.
- Q4_0: F16 scale + 16 bytes of nibbles (low nibble first),
  `value = (nibble - 8) * scale`.
- Q8_0: F16 scale + 32 signed bytes, `value = q * scale`.
- F16/BF16: 2 bytes per value, bit-exact decode to F32.
- Q6_K: 210-byte blocks of 256 values: 128 low-nibble bytes, 64 high-bit
  bytes, 16 int8 scales, then the F16 super-scale `d`;
  `value = d * scale * (six_bit_quant - 32)`.
- Formats without a reviewed decoder (Q4_K, Q5_0, Q5_K, Q8_K)
  return explicit unsupported errors — never approximations.
- **Q5_0 geometry `(32 values, 22 bytes)` is recorded from the GGML
  block layout and must be re-verified against GGML headers when its
  decoder is implemented.**
