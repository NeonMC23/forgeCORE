# forgeCORE project state

**As of:** 2026-10-24  
**Repository:** `forgeCORE` (`NeonMC23/forgeCORE`)  
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

- Model loading with explicit CPU/mmap/mlock modes, tensor checking, and
  static GPU-offload configuration (`GpuLayers`, split mode, device list).
  GPU execution is **not** validated on hardware in this environment.
- `Model → Context → Batch → decode → Logits` CPU inference path.
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
  failed, 0 ignored, and no `SKIP` lines**. Without any fixture environment,
  113 `SKIP` lines are emitted; they must not be counted as executed coverage.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: passed.

## Downstream real-model smoke evidence (2026-10-24)

RAMforge exercised this `forge-core` pin through its explicit
`--backend forgecore` path on the two real GGUFs in the shared workspace. Each
run was a separate CPU process; the two large models were never loaded
together. Evidence:
`/tmp/ramforge-real-model-smoke/20261024-184439`.

| Model | Exit | Wall | Max RSS | Native compute buffer |
|---|---:|---:|---:|---:|
| SmolLM2-135M Q4_K_M | 0 | 0.30 s | 367,176 KiB | 98.25 MiB |
| Qwen2.5-1.5B Q4_0 | 0 | 1.87 s | 2,815,624 KiB | 302.75 MiB |

Both runs generated 8 tokens and reported `Swaps: 0`. ForgeCore's requested
4 GiB budget is diagnostic only and is not enforced. RSS includes the loader,
page-cache effects, Rust heap, and native allocations; it is not model
residency. The GPU stayed at 40 MiB used / 3731 MiB free before and after every
run, which confirms no GPU execution occurred; it is not GPU validation.

Qwen2.5 received a raw `Hello` prompt without a chat template, so its text is
not quality evidence. SmolLM2 output was coherent. Model inputs were hashed
read-only; the recorded SHA-256 values are in RAMforge's
`scripts/smoke-real-models.sh` run directory.

## Explicit limitations

- No GPU hardware was available. GPU/offload code paths compile and validate
  inputs, but real device execution is unverified.
- Async/multi-device scheduling graphs, events, explicit buffer-type selection,
  and `tensor_buft_overrides` are not implemented.
- KV allocated/resident bytes and per-layer K/V read-back are unavailable in
  the pinned native C API.
- Model-path out-of-core weight streaming is not possible at the pin: libllama
  loads model weights eagerly. This is an architectural constraint, not a
  forgeCORE wrapper gap.
- The live BPE fixture covers a small ASCII vocabulary with the `default`
  GPT-2 pre-tokenizer. Byte-level BPE edge cases and other pre-tokenizer
  variants (for example llama3) remain unverified.
- The three pre-ggml pure-Rust integration tests are archived under
  `docs/archive/forge-core-legacy/tests/`; they are intentionally outside
  Cargo's test graph.
