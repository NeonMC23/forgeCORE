# ForgeCore Phase 1 — Repository audit

**Date:** 2026-09-24
**Auditor:** initial ForgeCore development session (agentic)

## 1. forgeCORE repository (this repo)

- Remote: `https://github.com/NeonMC23/forgeCORE`
- State at audit time: **completely empty** — no commits, no files, no
  `Cargo.toml`, no handoff documentation.
- Conclusion: a genuine clean restart. There is no inherited implementation
  to preserve, and no accidental RAMforge code to distrust.

## 2. Handoff documentation

The eight handoff files were **not** present in this repo. They were located
in the base repository `https://github.com/NeonMC23/RAMforge`
(`docs/handoff/`, commit `a683513`, "V3.0.2") and imported verbatim into
`docs/handoff/` here (see `docs/handoff/IMPORTED_FROM.md`).

All eight files were read in full before any implementation decision:

| File | Contents used |
|---|---|
| `README.md` | Evidence labels, reading order, stop-ship conclusion |
| `PROJECT_STATE.md` | Workspace layout, support registry, validation snapshot (351 passed / 2 ignored) |
| `ARCHITECTURE.md` | Module map, layer execution order, KV staging contract |
| `CURRENT_INFERENCE_ENGINE.md` | Matrix convention, RoPE pairing, GQA mapping, Q6_K output geometry |
| `QWEN25_CORRECTNESS_INVESTIGATION.md` | Target identity, fixed prompt IDs, trace schema, row 59/220 reference literals, unknowns |
| `REBUILD_PLAN.md` | ForgeCore phase structure and acceptance gates |
| `VALIDATION.md` | Commands that actually ran; diagnostic status (NOT RUN) |
| `REPOSITORY_TRANSFER.md` | Preserve/rebuild split, engineering rules |

## 3. RAMforge source inspection

Checked out alongside this repo for convention cross-checking only. Relevant
observations (all consistent with the handoff):

- Workspace of three crates (`ramforge-core`, `ramforge-runtime`,
  `ramforge-cli`), edition 2021, version 0.1.0.
- `ramforge-core/src/compute.rs`: scalar reference ops with the explicit
  `y[o] = sum_i W[o*input+i] * x[i]` convention, half-split RoPE, GQA
  `kv_head = q * kv_heads / query_heads`, stable softmax.
- `ramforge-runtime/src/model_executor.rs`: the 10-step layer order
  (attn norm → Q/K/V → optional all-or-none biases → RoPE → attention →
  output proj → residual → FFN norm → gate/up → SiLU·up → down → residual).
- `ramforge-runtime/src/kv_cache.rs`: `[position][kv_head][head_dim]`
  layout, append-per-layer then single sequence increment.
- `ramforge-runtime/src/model.rs`: required tensor names
  (`token_embd.weight`, `output_norm.weight`, `blk.{i}.*`), GGUF metadata
  key fallbacks, `head_dim = embedding_length / head_count`.
- `ramforge-core/src/quant.rs`: block-size constants
  (Q4_0: 18, Q8_0: 34, Q2_K: 84, Q3_K: 110, Q4_K: 144, Q5_K: 176,
  Q6_K: 210, Q8_K: 292; QK4_0/QK8_0 = 32, QK_K = 256).

Per the rebuild decision, **none of this code was copied**. ForgeCore's
reference core is written fresh; the RAMforge source served only to confirm
that the conventions adopted here match the documented intent. The
unresolved Qwen2.5 divergence means no RAMforge numerical behavior is
treated as authoritative.

## 4. Toolchain

No Rust toolchain was present in the environment. Installed rustup stable:

- `cargo 1.98.1`, `rustc 1.98.1` — the same versions as the handoff's
  recorded validation toolchain.

## 5. Ambiguities discovered in the handoff

1. **Attention score scaling (ulp-level):** the handoff fixes
   "head-dimension scale" but not division (`dot / sqrt(d)`) versus
   multiplication (`dot * (1/sqrt(d))`). ForgeCore uses division, matching
   the RAMforge reference; recorded in `docs/CONVENTIONS.md` as an open
   verification point for the future llama.cpp comparison.
2. **Q5_0 block geometry:** Q5_0 is in ForgeCore's target list but
   unsupported by RAMforge, so no in-repo geometry exists. The `(32, 22)`
   spec is recorded from the GGML block layout and flagged for
   re-verification when its decoder is implemented.
3. **Qwen2.5 eps/theta/context:** recorded as observed values
   (`1e-6`, `1_000_000`, `32768`) that must be re-verified against the
   target GGUF; carried into `ModelDims::qwen25_15b()` with that caveat.
4. **Layer-27 trace, `result_norm`, completed diagnostic outputs:** UNKNOWN
   in the handoff and remain UNKNOWN; no values were reconstructed.
