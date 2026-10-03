# ForgeCore

Thin, high-performance Rust abstraction over upstream llama.cpp/ggml —
the middle of `RAMforge → ForgeCore → llama.cpp/ggml → hardware`.

ForgeCore owns **no compute of its own**: it opens ggml backends,
manages tensors and graphs, loads `.gguf` models through libllama,
runs the minimal CPU inference path (`Model` → `Context` → `Batch` →
`decode` → `Logits`), and keeps frozen scalar oracles for
cross-validation. All execution, quantization, and model loading are
upstream's job.

## Quickstart

```sh
. /home/user/forgeCORE/scripts/env.sh
sh /home/user/forgeCORE/scripts/setup-native.sh   # pinned llama.cpp build (outside the repo)
cd /home/user/forgeCORE
cargo test --workspace
```

See `docs/NATIVE.md` for the native runbook (prerequisites, backend
selection, test fixtures) and `docs/TOOLCHAIN.md` for the rule that no
toolchain or build state may live under `/home/user/`.

## Layout

```text
crates/forge-sys     hand-written FFI to the pinned ggml/libllama C API
crates/forge-core    safe API: devices, backends, tensors, runtime, models, contexts, batches
  src/reference/     frozen scalar validation oracles (never executed)
  tests/ggml_smoke.rs  Rust → ggml → backend → op → Rust integration test
  tests/cpu_decode.rs  Model → Context → Batch → decode → logits integration test
scripts/             env.sh, setup-native.sh, make-tiny-gguf.py
docs/                PIVOT.md, NATIVE.md, LICENSING.md, TOOLCHAIN.md, …
```

Public API at a glance:

```rust
use forge_core::{Backend, Context, ContextOptions, Tensor, add, enumerate_devices, matmul, Model};

let devices = enumerate_devices();          // CPU, CUDA, … — whatever is registered
let backend = Backend::open_cpu()?;         // or Backend::open_device(&devices[i])
let a = Tensor::from_f32(&backend, &[2, 3], &data)?;
let sum = add(&a, &a)?;                     // one-node ggml graph, run on the backend
let prod = matmul(&a, &b)?;                 // C = A·Bᵀ, ggml ne order
let host: Vec<f32> = sum.to_vec_f32()?;
let model = Model::load("model.gguf".as_ref())?;
let mut context = Context::open(&model, &ContextOptions::default())?;
context.decode(&batch)?;                    // batch of token ids + positions
let logits: &[f32] = context.logits(last_index)?.values(); // owned n_vocab copy
```

## Docs

- `docs/PIVOT.md` — why the pivot, what was kept/removed, API and FFI
  design, verified ggml semantics, validation record.
- `docs/NATIVE.md` — native dependency runbook.
- `docs/LICENSING.md` — upstream license terms and attribution.
- `docs/CONVENTIONS.md` — conventions governing the frozen oracles.
- `docs/handoff/` — historical pre-pivot notes (frozen).

## Status

CPU-validated milestone: device discovery, F32 add/matmul execution,
model loading with metadata, context/batch/decode/logits CPU
inference path, and explicit validation errors — 81 tests green with
strict clippy. Out of scope here: text tokenization, sampling and
generation policy, GPU offload, streaming, residency control, and
RAMforge integration (see `docs/PIVOT.md` roadmap).
