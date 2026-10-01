# ForgeCore ggml pivot — architecture record

**Date:** 2026-10-01
**Status:** implemented and validated (CPU)

ForgeCore is no longer an independent inference engine. It is a thin,
high-performance Rust abstraction over upstream llama.cpp/ggml:

```text
RAMforge → ForgeCore → llama.cpp/ggml → CPU / CUDA / HIP / Vulkan / Metal / …
```

All execution, quantization, and model loading are upstream's job.
ForgeCore owns device/backend discovery, safe tensor and graph handles,
model-load plumbing, validation oracles, and the RAMforge-facing API
contracts. Nothing in this tree reimplements ggml kernels.

## 1. Pre-pivot code audit and disposition

Pre-pivot `forge-core` was 3765 lines (10 modules + 4 integration tests).
Module-by-module verdict:

| Module (pre-pivot lines) | Verdict | Now |
|---|---|---|
| `quant.rs` (785) | **Retain as oracle** | `reference/quant.rs` — scalar Q4_0/Q8_0/Q6_K/… decoders for cross-validation |
| `ops.rs` (406) | **Retain as oracle** | `reference/ops.rs` — scalar matvec/add/RoPE/RMS-norm expected-value kernels |
| `dtype.rs` (93) | **Retain as oracle** | `reference/convert.rs` — exact F16/BF16 bit conversions |
| `shape.rs` (191) | **Retain as oracle** | `reference/shape.rs` — row-major layout helpers the oracles are written against |
| `checkpoint.rs` (124) | **Retain as oracle** | `reference/checkpoint.rs` — legacy tiny-checkpoint reader for old fixtures |
| `error.rs` (23) | **Retain, extended** | `error.rs` — same `Error(String)` type plus `backend`/`unsupported`/`invalid` constructors |
| `model.rs` (690) | **Replaced** | `model.rs` — deleted the hand-written executor; now a `llama_model` owner (load/free/metadata) |
| `attention.rs` (265) | **Removed** | Upstream owns attention (ggml ops / libllama) |
| `kv.rs` (279) | **Removed** | Upstream owns KV-cache (libllama contexts) |
| `tests/tiny_model.rs` (555) | **Removed** | Validated our old engine, not ggml; superseded by `tests/ggml_smoke.rs` |
| `tests/attention_gqa.rs` (115) | **Removed** | Covered removed engine code |
| `tests/kv_cache.rs` (98) | **Removed** | Covered removed engine code |
| `tests/quant_contract.rs` (107) | **Retain** | Same contract tests, repathed to `reference::` |

Retained oracle code is frozen scalar Rust: it never calls ggml, is
never on an execution path, and exists only so tests can check ggml
numerics without trusting ggml. No new hand-written quants or kernels
will be added; those belong upstream.

The Qwen2.5 diagnostic prompt/ID fixtures survived as
`model::QWEN25_DIAGNOSTIC_PROMPT{,_IDS}` placeholders for future
real-model fixtures.

## 2. Upstream audit (llama.cpp/ggml)

- **Revision:** tag `v0.5.0`, commit
  `7fe450e19305b828c199d602c23a8337aaa1f03b` (2026-09-23). Pinned by tag
  **and** SHA in `scripts/setup-native.sh`, which aborts on mismatch.
- **License:** MIT, `Copyright (c) 2023-2026 The ggml authors`
  (see `docs/LICENSING.md`). No upstream source is copied into this
  tree; we dynamically link the prebuilt shared libraries.
- **Layout:** public C API in `include/llama.h`, `ggml/include/ggml.h`,
  `ggml/include/ggml-backend.h`, `ggml/include/ggml-alloc.h`,
  `ggml/include/ggml-cpu.h`; backends under `ggml/src/` (cpu, cuda,
  hip, vulkan, metal, cann, musa, sycl, opencl, webgpu, rpc, …).
- **Build:** CMake, `BUILD_SHARED_LIBS=ON`, tests/examples/tools off,
  CPU-only in this environment. Produces `libggml`,
  `libggml-base`, `libggml-cpu`, `libllama` (+ `libllama-common`).

## 3. Architecture decision: ggml **and** libllama

ForgeCore binds **both** C surfaces from the same pinned tree:

- **ggml + ggml-backend** for device discovery, tensor allocation,
  and graph execution. This is the core RAMforge needs: explicit
  backend handles, per-device buffers, and op execution with full
  control over residency — the substrate for streaming, multi-device,
  and larger-than-VRAM work later.
- **libllama** minimally for real model loading (`load`/`free`/
  `n_params`/`vocab`) so ForgeCore never reimplements the `.gguf`
  loader, tokenizer-adjacent metadata, or weight mapping.

Alternatives rejected:

- **ggml only:** would force reimplementing model loading — exactly
  what the pivot forbids.
- **libllama only:** hides backends, buffers, and graphs behind
  contexts; RAMforge needs that control.
- **Vendoring upstream source:** rejected per policy; the native tree
  lives outside the repo at `$FORGE_LLAMA_DIR`.
- **bindgen:** rejected; the bound surface is ~40 functions and two
  small structs, and hand-written declarations keep `unsafe` reviewable
  and the build dependency-free.

## 4. Crate layout and public API

```text
crates/forge-sys    hand-written FFI: opaque handles, repr(C) structs,
                    ggml/llama constants, unsafe extern "C" fns, link script
crates/forge-core   safe API + frozen oracles (depends only on forge-sys)
```

Public API (no raw pointers anywhere):

| Item | Role |
|---|---|
| `enumerate_devices() -> Vec<DeviceInfo>` | Snapshot of the ggml registry (name, kind, memory) |
| `Backend::{open_cpu, open_device}` | Owned backend handle; `set_cpu_threads` on CPU |
| `Tensor::{from_f32, empty, to_vec_f32}` | Owned tensor + storage; ggml `ne` order, ≤4 dims |
| `runtime::{add, matmul}` | One-node ggml graphs on a shared backend (F32) |
| `Model::{load, load_with_options, …}` | Owned `llama_model` (CPU load; `ModelOptions` is `#[non_exhaustive]` for future offload); metadata: `n_params`, `vocab_size`, `description`, `size_bytes`, `n_ctx_train`, `n_embd`, `n_layer`, `n_head`, `n_head_kv` |
| `Context::{open, decode, logits}` | Owned `llama_context` (+ model share); `ContextOptions` is `#[non_exhaustive]` (CPU: `n_ctx`, `n_threads`) |
| `BatchBuilder` / `Batch` | Owned token/position/sequence/logits-flag arrays; immutable view for decode |
| `Logits::{values, n_vocab}` | Owned `n_vocab`-float copy per decoded batch index |
| `DType` | Safe `ggml_type` mirror with explicit unknown-id errors |
| `reference::{quant, ops, convert, shape, checkpoint}` | Frozen validation oracles (never executed) |

`Backend`, `Tensor`, `Model`, `Context`, and `Batch` are `!Send +
!Sync` (raw handles) and free their allocations on drop. All fallible
operations return `Result<T, Error>`, with per-domain constructors
(`backend`/`model`/`context`/`batch`/`decode`/`logits` plus
`unsupported`/`invalid`/`native`).

## 5. FFI boundary

`forge-sys` is the only crate that talks to C. Rules:

- Opaque structs cross as pointers only; the four `#[repr(C)]`
  structs (`ggml_init_params`, `llama_model_params`, `llama_batch`,
  `llama_context_params`) were layout-audited against gcc on x86_64
  (`sizeof` 24/80/56/160; every field offset checked) with
  committed regression tests in `forge-sys`.
- Every `unsafe` block in `forge-core` carries a `// SAFETY:` comment;
  every nullable C return is checked; C strings are copied out
  immediately.
- Enums cross as `c_int` and are mapped to Rust enums with explicit
  `Unknown`/unsupported arms — never silent substitutes.

## 6. Verified ggml semantics

`ggml_mul_mat(a, b)` with `a: [k, m]`, `b: [k, n]` computes `C = A·Bᵀ`
for the row-major `[m, k]` / `[n, k]` buffers and stores the `[m, n]`
result with `result[m + n*m] = Σ_k a[m*k+k]·b[n*k+k]` (verified against
`ggml_compute_forward_mul_mat_one_chunk`, not just the header comment,
whose first draft misled us — see the session notes). `Tensor` and
`runtime::matmul` document this formula; the smoke test checks it
elementwise against the scalar `dot` oracle.

## 7. State, backends, and future room

- Native state (checkout, build, tools, models) lives under
  `$FORGE_LLAMA_DIR` (default `/var/tmp/forge-native`); Rust state
  under `$FORGE_TOOLCHAIN_ROOT` (default `/var/tmp/forge-toolchain`).
  `/var/tmp` is used because `/tmp` is a ~1 GB tmpfs here. Nothing
  toolchain-related may live under `/home/user/` (see
  `docs/TOOLCHAIN.md`, `docs/NATIVE.md`).
- Backend selection is a native-build concern (`-DGGML_CUDA=ON`, …);
  the Rust API already enumerates any registered kind and opens
  accelerators by `DeviceInfo`. Streaming, residency control,
  multi-device graphs, and larger-than-VRAM models build on
  `Backend`/`Tensor`/graph primitives without API breakage.
- RAMforge stays completely out of this tree (code, deps, paths,
  runtime) except historical notes under `docs/handoff/`.

## 8. Validation record (2026-10-01, CPU-only)

Environment: no GPU (`/dev/nvidia*`, `/dev/dri` absent), 2 cores,
Rust 1.98.1, CMake 4.1.2, upstream `v0.5.0` @ `7fe450e1`.

- Native C smoke test (independent of Rust): 1 device
  (`CPU`, Intel Xeon, ~2 GB), `ggml_backend_init_by_type(CPU)` OK.
- `cargo fmt --all -- --check`: clean.
- `cargo test --workspace`: 63 passed —
  47 lib (oracles + mapping/layout units), 9 `ggml_smoke`
  (enumeration, CPU add/matmul vs oracles, validation errors, stale
  index, missing-file load, GPU skip-probe, tiny-fixture load), 5
  `quant_contract`, 2 `forge-sys` layout.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: clean.
- `cargo build --workspace --release`: clean.
- Tiny generated LLAMA GGUF (vocab 32, 1176 params): `Model::load`
  reports `n_params=1176 vocab_size=32`, matching the hand-counted
  parameter total exactly.

Phase-1 CPU inference (same day, Rust 1.99.0, same pin): 81
passed — +7 lib units (batch validation, option defaults), +9
`cpu_decode` (metadata, context open/validation, decode success,
native KV-error propagation, logits copy/determinism/invalid
index), +2 `forge-sys` layout. C probes confirmed: decode codes
(`0`/`1`/`2`/`-1`), out-of-window positions rejected gracefully
(not aborts), `llama_get_logits_ith` NULL on invalid index in
Release while Debug aborts, positive index = batch token position.
`fmt`/`clippy -D warnings`/`release` clean.

## 9. Roadmap (out of this milestone)

Text tokenization, sampling/generation, `n_gpu_layers` offload paths,
KV-cache control, F16/I8 host transfer, quantized matmul
cross-validation against `reference/quant`, streaming execution, and
residency/multi-device policy — all as extensions of the
`Backend`/`Tensor`/`Model`/`Context`/`Batch` handles defined here.
