# forgeCORE project state

**As of:** 2026-10-30<br>
**Repository:** `forgeCORE` (`NeonMC23/forgeCORE`)<br>
**Role:** safe Rust abstraction and integration layer between RAMforge and the
pinned llama.cpp/ggml native implementation.

This is the current state document for forgeCORE. The files under
[`docs/handoff/`](handoff/) are a historical RAMforge V3.0.2 handoff imported
verbatim during repository initialization; they are not a description of the
current forgeCORE tree. See [`handoff/README.md`](handoff/README.md).

## Boundary

`RAMforge → forgeCORE → llama.cpp/ggml`

forgeCORE owns safe wrappers, FFI/ABI correctness, native handle ownership,
model/context/batch/graph/backend/device/buffer primitives, and low-level
inspection facts. It does **not** own RAMforge's memory budget, residency,
eviction, cache, placement, scheduling, or prefetch policy. GGUF ownership is
recorded in [`GGUF_OWNERSHIP.md`](GGUF_OWNERSHIP.md).

## Native resource-lifetime contract (2026-10-30)

The native pin is unchanged. The tracked residency patch now adds:

- a model-scoped shared/exclusive execution gate: contexts hold a shared lease
  from asynchronous graph submission through `llama_synchronize`/destruction;
- non-blocking page release (`EBUSY` while any context is in flight);
- model instance ids, active/in-flight context counts, point-in-time
  quiescence, context ids, execution generations, and contract version 1.

This establishes a safe **between-operation physical page reclamation**
boundary. It does not prove that a model weight will not be reused by the next
token/graph and therefore does not authorize intra-decode eviction. KV cache,
graph scratch, backend buffers, and other context-owned allocations have no
independent native page-release API. `ResidencyError::{Busy, Unsupported,
Invalid, Native}` is exposed by forge-core. See `docs/NATIVE.md` and
`README.md`.

## Crates

- `crates/forge-sys` — hand-written FFI to the pinned ggml/libllama C API.
  The native tree is external; this crate does not compile or download it.
- `crates/forge-core` — safe Rust API. Active modules: backend, batch, buffer,
  context, device, dtype, error, gguf, graph, memory, model, quant, reference,
  runtime, sampler, tensor, tokenizer.

## Native baseline

- Upstream: llama.cpp/ggml `v0.5.0`
- Commit: `7fe450e19305b828c199d602c23a8337aaa1f03b`
- ggml version reported by CMake: `0.25.1`
- Shared libraries are resolved through `FORGE_LLAMA_DIR` (default
  `/var/tmp/forge-native`); the build scripts only link against
  `$FORGE_LLAMA_DIR/build/bin` and never build native code.

The current workspace tree is at `native/src` with the matching revision and
prebuilt shared libraries at `native/build/bin`. Point `FORGE_LLAMA_DIR` at the
workspace `native` directory when the default path is absent:

```sh
export FORGE_LLAMA_DIR="/home/neon/Documents/CODE LOCAL/forge/native"
```

## Implemented capabilities

- Model loading with explicit CPU/mmap/mlock modes, tensor checking, static
  GPU-offload configuration (`GpuLayers`, split mode, device list), CPU
  weight-repacking control (`ModelOptions::use_extra_bufts`,
  `without_cpu_weight_repacking()`), and an opt-in
  `ModelLoadMode::MmapNoPrefetch` mode that skips whole-file `MAP_POPULATE`,
  plus model-scoped and tensor-scoped Linux page release:
  `Model::release_mapping_pages()`, `Model::tensor_mappings()`, and
  `Model::release_tensor_pages()`. These are supplied by the documented
  `patches/llama-model-residency.patch`; the default `Auto`/`Mmap` behavior is
  unchanged. Tensor metadata identifies only exact model-owned mapping
  fragments; range release skips pages shared with unselected file-backed
  tensors and never unmaps or invalidates pointers. Unsupported platforms
  return `ENOTSUP`. A bounded Linux `MADV_WILLNEED` prefetch hint and a
  scheduler eval trace (node name/op/dependencies, explicit synchronized
  completion) are exposed through the same patched native tree. The trace is
  diagnostic and does not establish intra-decode eviction safety. A separate
  opt-in CPU node-dispatch observer (`Context::set_dispatch_observer`) emits
  one event from worker 0 before each non-empty compute-enabled node; it is
  dispatch-start evidence only, not a tensor read, page-fault, I/O, completion,
  or release proof. Tensor mappings additionally expose a model-instance
  generation, mapping id, file path/size, and validated page range. A bounded
  Linux `mincore()` sample can be taken per named file-backed tensor; sampling
  is opt-in, per-source, and occurs before dispatch publication/barrier. It is
  still residency observation only. GPU execution is **not** validated on
  hardware in this environment.
- `Model → Context → Batch → decode → Logits` CPU inference path, with
  real-model validation on `llama` (SmolLM2), `qwen2` (Qwen2.5-1.5B), and
  `gptneox` (Pythia-70M).
- Context sizing/threading controls, multi-sequence and embedding batches,
  pooling/attention selection, embedding outputs.
- Native KV/state primitives: sequence surgery, position shifts, byte-oriented
  state snapshots, exact serialization sizes, KV cache dtype selection.
- Tokenizer, sampler chain, tensor/dtype/quantization geometry, graph
  execution, backend/device discovery, and owned buffers.
- Payload-lazy GGUF inspection (header, metadata, tensor descriptors, checked
  file ranges and quantization geometry) through `forge_core::gguf`.
- Quantization storage geometry through `forge_core::quant`
  (`block_spec`, `row_bytes`, `tensor_bytes`, `QuantFormat`).

## Validation (2026-10-24)

Native tree: workspace `native/` at the pinned v0.5.0 commit. Fixtures were
generated with `scripts/make-tiny-gguf.py` into `/tmp` (default, `--tok`,
`--bpe`, `--dsv4`); no fixture is stored in the repository.

```sh
export FORGE_LLAMA_DIR="$PWD/../native"
export FORGE_TEST_MODEL=/tmp/forge-native-models/tiny-llama.gguf
export FORGE_TEST_MODEL_TOK=/tmp/forge-native-models/tiny-tok.gguf
export FORGE_TEST_MODEL_BPE=/tmp/forge-native-models/tiny-bpe.gguf
export FORGE_TEST_MODEL_DSV4=/tmp/forge-native-models/tiny-dsv4.gguf
cargo fmt --all -- --check
cargo check --workspace --locked
cargo test --workspace --locked            # 370 tests with all four fixtures
cargo clippy --workspace --all-targets --locked -- -D warnings
```

The fixtures are not stored in the repository. They were generated with:

```sh
PYTHONPATH=/tmp/forge-audit-py python3 scripts/make-tiny-gguf.py /tmp/forge-native-models/tiny-llama.gguf
PYTHONPATH=/tmp/forge-audit-py python3 scripts/make-tiny-gguf.py --tok /tmp/forge-native-models/tiny-tok.gguf
PYTHONPATH=/tmp/forge-audit-py python3 scripts/make-tiny-gguf.py --bpe /tmp/forge-native-models/tiny-bpe.gguf
PYTHONPATH=/tmp/forge-audit-py python3 scripts/make-tiny-gguf.py --dsv4 /tmp/forge-native-models/tiny-dsv4.gguf
```

`/tmp/forge-audit-py` is a throwaway `pip --target` directory containing
`gguf` and `numpy`; it is not a system installation and is not required to
build or test forgeCORE without fixtures.

Results:

- `cargo fmt --all -- --check`: passed.
- `cargo check --workspace --locked`: passed.
- `cargo test --workspace --locked` with all four fixtures: **370 passed, 0
  failed, 0 ignored, and no `SKIP` lines** (run before the GPT-NeoX test was
  added). Without any fixture environment in the current tree, 370 tests still
  pass, fixture-gated tests report `SKIP`, and the new GPT-NeoX test is the one
  ignored test unless `FORGE_GPTNEOX_MODEL` is set. They must not be counted as
  executed coverage when skipped.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed.

### Mapping residency evidence (2026-10-25)

The native tree is the pinned v0.5.0 checkout plus the documented
`patches/llama-model-residency.patch`; `setup-native.sh` applies it
idempotently and verifies the upstream SHA first. The patch adds
`MmapNoPrefetch` and `llama_model_release_pages`; `llama_model_params` layout
and default `Auto`/`Mmap` behavior are unchanged.

Current standard run: **383 passed, 0 failed, 3 ignored** (GPT-NeoX execution
and the opt-in prefetch/anonymous mapping tests). With `FORGE_TEST_MODEL`
pointing at the tiny fixture, `all_load_modes_load_cpu_fixture` covers
`MmapNoPrefetch`, and `mapping_release` verifies two loaded models:

```text
A 8192->0, B 8192->8192
```

Model A's mapping pages were released; model B's remained resident; a decode
against A after release succeeded. `mapping_ranges` verifies exact tensor
metadata and conservative shared-page handling. With the real Pythia fixture,
the largest file-backed tensor reported:

```text
released_bytes=27,365,376 skipped_shared_bytes=4,096 selected_tensors=1
```

A decode after repeated tensor release succeeded. The opt-in Qwen2.5-1.5B
Q4_0 repack check reported `file_backed=143 anonymous=196`, confirming that
repacked tensors are not misclassified as file-backed. The opt-in disk-backed
prefetch check:

```sh
FORGE_PREFETCH_MODEL=/home/neon/Documents/CODE LOCAL/forge/target/fixtures/pythia-70m.Q8_0.gguf \
  cargo test -p forge-core --test mapping_prefetch --locked -- --ignored --nocapture
```

reported: file `76,722,400 B`; default `Mmap` mapping RSS `74,952,704 B`;
`MmapNoPrefetch` mapping RSS `0 B`. This confirms the option skips
whole-file prefaulting; neither operation implements per-layer residency.

Graph trace and prefetch evidence (2026-10-25):

- Tiny fixture with `ContextTraceOptions { max_events: 4096 }`: 37 requested
  and 37 synchronized-completion events retained; node/op/dependency names
  populated; trace disabled returns no summary/events.
- `max_events=2`: 37 requested/37 completed, 72 events dropped, 2 retained.
- Real Pythia tensor prefetch: `27,369,472 B` requested; repeated prefetch
  succeeded; subsequent tensor release reported `released_bytes=27,365,376`,
  `skipped_shared_bytes=4,096`; decode succeeded.

The eval callback runs synchronously after scheduler backend synchronization
for observed node groups, but it is not proof that a weight can be released for
the next graph/token. Intra-decode eviction remains unsupported.

### GPT-NeoX/Pythia execution (2026-10-25)

The pinned llama.cpp revision already implements `gptneox`
(`native/src/src/models/gptneox.cpp`); no native upgrade or patch was needed.
The missing piece was RAMforge's legacy architecture allowlist, not ForgeCore.
RAMforge's explicit `--backend forgecore` path now exercises this architecture
through the ordinary safe API (`Model → Context → Batch → decode → Logits`).

New opt-in test:
`crates/forge-core/tests/gptneox_execution.rs`, gated by
`FORGE_GPTNEOX_MODEL`. It checks the model description contains `gptneox`,
layer/embedding counts are non-zero, the native tokenizer encodes the prompt,
logits are finite and non-constant, two independent greedy runs produce
identical ids, and the generated ids decode to text.

Fixture: `mradermacher/pythia-70m-deduped-GGUF`,
`pythia-70m-deduped.Q8_0.gguf`, 76,722,400 B, SHA-256
`94788a58bb81431b7058837ec2371c303edb20384b615f0afabfa81539294e0d`,
GGUF v3, 6 layers, 512 embedding, 8 heads, 2048 FFN, 16 rotary dimensions,
parallel residual enabled.

```sh
export FORGE_LLAMA_DIR="$PWD/../native"
export LD_LIBRARY_PATH="$FORGE_LLAMA_DIR/build/bin"
export FORGE_GPTNEOX_MODEL=/path/to/pythia-70m.Q8_0.gguf
cargo test -p forge-core --test gptneox_execution --locked -- --ignored --nocapture
```

Observed: prompt `The capital of France is` encoded as five ids
`[510, 5347, 273, 6181, 310]` (`adds_bos=false`), first four greedy ids
`[253, 5347, 273, 253]`, decoding to ` the capital of`. A native llama.cpp probe
with the same ids produces the same greedy prefix, and RAMforge
`--backend forgecore` outputs ` the capital of the French Empire.` for 8
tokens.

Independent graph check: a NumPy GPT-NeoX forward pass over the same GGUF
(dequantized Q8_0 weights, partial NeoX RoPE, parallel residual, GELU MLP)
reproduces the native final-token logits with top-1 id 253, 9/10 top-10
overlap, cosine similarity 1.000000046, and maximum absolute difference 0.929
on logits of magnitude ~1,579. The residual difference is consistent with
ggml's f16 GELU lookup and reduction order. This is not bit-exact parity.

ForgeCore itself does not enforce RAM budgets, and it exposes no per-layer
weight streaming. At this pin the default `Mmap` mode maps the model and
prefaults the whole mapping with `MAP_POPULATE`; the opt-in
`ModelLoadMode::MmapNoPrefetch` skips that prefault. RAMforge's separate
`--enforce-ram` cgroup v2 supervisor executed the 13,133,230,080 B
GPT-NeoX-20B fixture under an 8 GiB hard cap with `--no-prefetch`: 7.64 GiB of
the 12.23 GiB mapping resident, 7,979 limit events, no OOM kill, one greedy
token ` Paris` in 7.72 s (16.94 s with prefetch). That is OS demand paging
under a kernel cap, not forgeCORE-managed out-of-core streaming. The model
remains impractical as a throughput target; see the RAMforge README for the
enforced matrix and caveats.

## Downstream real-model smoke evidence (2026-10-25)

RAMforge exercised this `forge-core` pin through its explicit
`--backend forgecore` path on real GGUFs. Each run was a separate CPU process;
the large models were never loaded together. Evidence:
`/tmp/ramforge-real-model-smoke/20261025-002055`.

| Model | Backend | Budget | Exit | Wall | Max RSS | Managed peak |
|---|---|---:|---:|---:|---:|---:|
| SmolLM2-135M Q4_K_M | forgecore | 4G | 0 | 0.20 s | 370,388 KiB | not exposed |
| Qwen2.5-1.5B Q4_0 | forgecore | 4G | 0 | 1.31 s | 2,839,744 KiB | not exposed |
| Pythia-70M-deduped Q8_0 | forgecore | 8G | 0 | 0.14 s | 167,616 KiB | not exposed |
| Pythia-70M-deduped Q8_0 | forgecore | 4G | 0 | 0.13 s | 167,576 KiB | not exposed |
| Pythia-70M-deduped Q8_0 | forgecore | 1G | 0 | 0.13 s | 167,584 KiB | not exposed |
| Pythia-70M-deduped Q8_0 | forgecore | 100M | 0 | 0.13 s | 167,512 KiB | not exposed |
| Pythia-70M-deduped Q8_0 | legacy | 1G | 1 | 0.01 s | 23,664 KiB | not exposed |

The earlier GPT-NeoX runs generated byte-identical stdout at 8G, 4G, 1G, and
100M **without** RAMforge's `--enforce-ram` flag. That is expected because
ForgeCore's requested budget is diagnostic only and is **not enforced**; it is
budget-invariance, not out-of-core execution. The legacy row is RAMforge's
intended architecture rejection before weight loading, not a ForgeCore failure.
All cases reported `Swaps: 0`. RSS includes the loader, page-cache effects,
Rust heap, and native allocations; it is not model residency. The native build
is CPU-only, so none of these runs is GPU validation.

RAMforge's enforced matrix uses a cgroup v2 `memory.max` around the same
ForgeCore call. Qwen2.5-1.5B completed at 1G with 857.93 MiB of its 1,011.16 MiB
mapping resident, and at 512M/256M with 334.74/84.36 MiB resident. At 100M the
kernel OOM-killed the process group because the minimum anonymous working set
exceeded the cap. GPT-NeoX-20B completed at 8G with 7.65 GiB of its 12.23 GiB
mapping resident. This is OS reclaim of the file-backed mapping plus
`--no-repack`, not a forgeCORE allocation policy; the pinned loader still
prefaults the whole file.

RAMforge also validated Qwen2.5-7B Q4_K_M as a two-shard set at 4G through its
legacy backend (4 tokens, peak managed 4,285,007,616 B under 4 GiB, 24 cached
layers, 88 evictions). That evidence exercises RAMforge's split discovery and
virtual cross-shard reads, not a new ForgeCore capability; ForgeCore still
receives the primary shard path and the pinned native loader owns the rest.

Qwen2.5 and SmolLM2 above used a raw `Hello` prompt, so their text is not
quality evidence. The GPT-NeoX prompt/continuation evidence is recorded in the
dedicated section above. RAMforge's legacy backend separately produced a
correct chat-templated answer (`4`) for Qwen2.5-1.5B using the model's own
`tokenizer.chat_template`; that formatting and tokenizer work is RAMforge-side
and was not exercised through the ForgeCore backend. Model inputs were hashed
read-only in the RAMforge run inventory.

## Explicit limitations

- No GPU hardware was available. GPU/offload code paths compile and validate
  inputs, but real device execution is unverified.
- Async/multi-device scheduling graphs, events, explicit buffer-type selection,
  and `tensor_buft_overrides` are not implemented.
- KV allocated/resident bytes and per-layer K/V read-back are unavailable in
  the pinned native C API.
- forgeCORE has no per-layer execution scheduler or automatic eviction.
  libllama maps the model and, in the default `Mmap` mode, prefaults the whole
  mapping with `MAP_POPULATE`. The opt-in `MmapNoPrefetch` mode skips that
  prefault; model-scoped and tensor-scoped release exist on Linux, and tensor
  ranges are exact, but layer grouping and release timing are caller policy.
  Release skips pages shared with unselected tensors and cannot guarantee a
  lower load-time peak. A bounded cgroup and `use_extra_bufts=false` can make a
  large model fit under a hard cap (demonstrated downstream on GPT-NeoX-20B at
  8 GiB), but residency is still controlled by the OS, not by forgeCORE
  per-layer scheduling.
- The live BPE fixture covers a small ASCII vocabulary with the `default`
  GPT-2 pre-tokenizer. Byte-level BPE edge cases and other pre-tokenizer
  variants (for example llama3) remain unverified.
- The three pre-ggml pure-Rust integration tests are archived under
  `docs/archive/forge-core-legacy/tests/`; they are intentionally outside
  Cargo's test graph.
