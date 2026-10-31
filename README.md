# ForgeCore

Thin, high-performance Rust abstraction over upstream llama.cpp/ggml —
the middle of `RAMforge → ForgeCore → llama.cpp/ggml → hardware`.

ForgeCore owns **no compute of its own**: it opens ggml backends,
manages tensors and graphs, loads `.gguf` models through libllama,
runs the minimal CPU inference path (`Model` → `Context` → `Batch` →
`decode` → `Logits` → `sample`), exposes the model's tokenizer,
binds model GPU-offload options (CPU by default, explicit errors,
never silent fallback), exposes context/batch controls (sizing,
threading, multi-sequence and embedding batches, pooling/attention
selection, embedding outputs), exposes native KV/state primitives
(sequence surgery, position shifts, byte-oriented state snapshots,
exact state sizes, KV cache dtypes), and keeps frozen scalar oracles
for cross-validation. All execution, quantization, and model loading
are upstream's job.

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
crates/forge-core    safe API: devices, backends, tensors, runtime, models, contexts, batches, tokenizer, sampler
  src/reference/     frozen scalar validation oracles (never executed)
  tests/ggml_smoke.rs  Rust → ggml → backend → op → Rust integration test
  tests/cpu_decode.rs  Model → Context → Batch → decode → logits integration test
  tests/tokenizer.rs   Tokenizer encode/decode/vocab integration test
  tests/sampler.rs     SamplerChain over synthetic + real logits integration test
  tests/offload.rs     model offload options + capability facts integration test
  tests/context_batch.rs  context/batch extension integration test
  tests/kv_state.rs       KV/state/memory-facts integration test
scripts/             env.sh, setup-native.sh, make-tiny-gguf.py
docs/                PIVOT.md, NATIVE.md, LICENSING.md, TOOLCHAIN.md, …
```

Public API at a glance:

```rust
use forge_core::{Backend, Context, ContextOptions, GpuLayers, Model, ModelLoadMode, ModelOptions, SampleConfig, Tensor, TokenId, add, enumerate_devices, matmul, supports_gpu_offload};

let devices = enumerate_devices();          // CPU, CUDA, … — whatever is registered
let gpu_ok = supports_gpu_offload();        // capability fact; no placement policy here
let backend = Backend::open_cpu()?;         // or Backend::open_device(&devices[i])
let a = Tensor::from_f32(&backend, &[2, 3], &data)?;
let sum = add(&a, &a)?;                     // one-node ggml graph, run on the backend
let prod = matmul(&a, &b)?;                 // C = A·Bᵀ, ggml ne order
let host: Vec<f32> = sum.to_vec_f32()?;
let model = Model::load("model.gguf".as_ref())?; // CPU unless offload is requested
let mut opts = ModelOptions::default();
opts.gpu_layers = GpuLayers::Count(20);     // explicit unsupported error without a GPU
opts.use_extra_bufts = Some(false);         // disable CPU repack buffers; keep weights file-backed
opts.load_mode = ModelLoadMode::MmapNoPrefetch; // optional: skip whole-file MAP_POPULATE
let gpu_model = Model::load_with_options("model.gguf".as_ref(), &opts)?;
let mut ctx_opts = ContextOptions::default();
ctx_opts.n_batch = 512;                     // + n_ubatch/n_seq_max/threads/embeddings/...
let mut context = Context::open(&model, &ctx_opts)?;
context.decode(&batch)?;                    // token/embedding batch, any sequence layout
context.set_n_threads(4, 4)?;               // post-open threading; synchronize() for GPU
context.memory().unwrap().clear(false);     // sequence surgery on borrowed Memory
let snap = context.export_state()?;         // owned snapshot; import_state() restores
let logits: &[f32] = context.logits(last_index)?.values(); // owned n_vocab copy
let mut sampler = SampleConfig::default().build_chain()?;  // greedy
let token: TokenId = sampler.sample(logits)?;              // back into Batch/Tokenizer
```

`Model::release_mapping_pages(&mut self)` asks the OS to reclaim resident pages
from exactly that model's file-backed mappings. On Linux it uses
`madvise(MADV_DONTNEED)`; it never unmaps or invalidates tensor pointers. Other
platforms return `ResidencyError::Unsupported`.

The native layer now has an execution gate. A context holds a shared lease from
asynchronous graph submission until `Context::synchronize()` (or context drop)
observes completion. Page release takes the exclusive side with a non-blocking
attempt and returns `ResidencyError::Busy` while any context still has
asynchronous work in flight. A synchronized context still logically owns model
weights for future tokens, so this is a safe **between-operation physical page**
boundary, not permission for intra-decode eviction.

`Model::lifetime_state()` exposes a process-local instance id, active/in-flight
context counts, and a point-in-time quiescence observation.
`Context::lifetime_state()` exposes context id, execution generation, and
synchronization state. `Model::residency_contract_version()` reports the native
contract version (currently 1).

`Model::tensor_mappings()` returns owned tensor metadata (name, file-backed
flag, mapping index, offset, size, page bounds). `Model::release_tensor_pages()`
releases only selected tensor ranges, conservatively skipping pages shared with
unselected file-backed tensors. `Model::prefetch_tensor_pages()` is a bounded
Linux `MADV_WILLNEED` hint, not a synchronous read. These are tensor-level
primitives; layer grouping remains caller policy.

`ContextOptions::trace` enables a bounded scheduler eval trace using the
pinned public `llama_context_params.cb_eval`. With tracing enabled, the
scheduler computes observed nodes in groups and synchronizes before the
`ask == false` callback. The trace reports node names, op names, dependencies,
and an explicit `BackendSynchronized` completion phase. It is diagnostic only:
it does not prove that a weight is safe to release for the next graph. Tracing
disabled preserves the original execution path.

`Context::set_dispatch_observer(Some(NodeDispatchOptions { ... }))` installs an
opt-in CPU-only node-dispatch observer in the patched ggml CPU execution loop.
It emits exactly one `NodeDispatchStarted`-semantics event per non-empty,
compute-enabled graph node, from worker 0 and before workers enter the node
kernel. The event carries model/context IDs, execution and event sequences,
node index/count, thread count, op name, node name, and declared source names.
It is diagnostic only: it does not prove a source read, page fault, I/O,
kernel completion, final use, or release safety. The observer is disabled by
default; `set_dispatch_observer(None)` synchronizes before removal. Event storage
is bounded, drops the oldest events, and reports `dropped_events`.

`Model::tensor_mappings()` now includes the mapping id, generation, file path,
file size, and validated page range in addition to the tensor offset/size. The
generation is the live model instance id; a new model/mapping set gets a new
generation. `Model::sample_tensor_pages(name, max_pages)` performs a bounded
Linux `mincore()` sample over only the validated file-backed range. It returns
requested/sampled/resident/nonresident/unknown page counts, mapping identity,
page size, truncation, and an explicit status. It is a point-in-time residency
observation, not evidence of a tensor read, page fault, I/O, or final use.
`NodeDispatchOptions.sample_pages` enables pre-dispatch sampling per declared
source; zero disables it. The sample runs before the dispatch event is stored
and before the worker barrier/kernel entry.

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
inference path, native tokenizer (encode/decode/vocab), native
sampler chains (greedy/dist/temp/top-k/top-p/min-p), model
GPU-offload wiring (layer counts, split modes, device selection,
tensor splits, mmap/mlock and mmap-no-prefetch modes, capability facts), context/batch
extension (effective sizing, multi-sequence and embedding batches,
threading, pooling/attention/flash selection, embedding outputs,
advisory KV/op offload flags), native KV sequence ops and
byte-oriented state snapshots (clear/remove/copy/keep, position
shifts, exact state sizes, KV cache dtypes), and explicit validation
errors — 383 tests green with strict clippy (3 opt-in tests ignored). Out of scope here:
generation policy/loops, allocated/resident memory reporting
(unavailable in the pinned C API), memory estimation, streaming,
residency control, multi-GPU orchestration, and RAMforge integration
(see `docs/handoff/` reports for the phased roadmap state).
