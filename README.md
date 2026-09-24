# ForgeCore

ForgeCore is the planned numerical inference engine extracted from the
RAMforge project: a small, explicit, testable transformer mathematics
core rebuilt from first principles. Correctness comes before
performance — the core is deliberately boring scalar reference code with
no SIMD, GPU, threading, or hidden layout reinterpretation.

## Status

Initial numerical foundation (reference core only). Implemented:

- Explicit tensor/matrix contracts (`[input, output]`, row-major
  `[output][input]`, `y[o] = sum_i W[o,i] * x[i]`)
- Scalar reference ops: dot, matvec, RMSNorm, add/mul, SiLU, SwiGLU,
  stable softmax, half-split RoPE
- Explicit causal GQA attention with validated head mapping
- Explicit `[position][kv_head][head_dim]` KV cache with stage/commit
  semantics
- Reference decoder-layer and token-forward executor with full
  intermediate traces
- Deterministic numerical checkpoints (length/min/max/sum/L2/first8,
  indexed values, top-k)
- Quantization interface with scalar decoders for F16, BF16, Q4_0, Q8_0
  (K-quants and Q5_0 refuse explicitly until implemented)
- Tiny end-to-end model validated against an independent F64 oracle

Historical context lives in [`docs/handoff/`](docs/handoff/) (imported
verbatim from RAMforge). The audit is [`docs/AUDIT.md`](docs/AUDIT.md);
the convention record is [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md).

## Layout

```text
forgeCORE
├── Cargo.toml                  # workspace
├── crates/forge-core
│   ├── src
│   │   ├── lib.rs              # module map
│   │   ├── error.rs            # single Error/Result type
│   │   ├── shape.rs            # MatrixShape, MatrixF32, GGUF ne mapping
│   │   ├── dtype.rs            # F16/BF16 bit-pattern conversion
│   │   ├── ops.rs              # scalar reference operators
│   │   ├── attention.rs        # causal GQA attention + score checkpoints
│   │   ├── kv.rs               # explicit KV cache
│   │   ├── model.rs            # dims, weights, layer/token forward, traces
│   │   ├── checkpoint.rs       # summaries, top-k
│   │   └── quant.rs            # block specs, scalar decoders, ref matvec
│   └── tests
│       ├── attention_gqa.rs    # GQA mapping/routing/history contracts
│       ├── kv_cache.rs         # staging/layout/ordering contracts
│       ├── quant_contract.rs   # block geometry + hand-decoded blocks
│       └── tiny_model.rs       # 2-layer model vs independent F64 oracle
└── docs
    ├── handoff/                # imported RAMforge context (verbatim)
    ├── AUDIT.md                # Phase 1 repository audit
    └── CONVENTIONS.md          # authoritative convention record
```

## Validation

```text
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

## Known limitations

- F32 reference path only; no optimized kernels yet (by design).
- Quantization decoders exist for F16, BF16, Q4_0, Q8_0 only; Q4_K,
  Q5_K, Q6_K, Q8_K, Q5_0 return explicit unsupported errors.
- No GGUF loading, tokenizer, sampling, or generation loop yet.
- No comparison against llama.cpp has been performed; no compatibility
  claim is made.

## What remains before Qwen2.5 validation

1. Scalar decoders for Q6_K (output projection), Q4_K, Q5_K, Q8_K, Q5_0,
   each with block-level tests.
2. GGUF descriptor parsing + bounded row reader feeding ForgeCore shapes.
3. Tokenizer loading and the fixed nine-token prompt gate.
4. Raw Q6_K row-59/220 byte and row-dot comparisons.
5. Per-layer trace comparison (layers 0–27) and `result_norm`/logits
   checkpoints against the external oracle.
6. Decode-boundary and KV-length validation, then generation.
