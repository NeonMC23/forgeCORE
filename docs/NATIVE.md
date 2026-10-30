# Native llama.cpp/ggml dependency — runbook

ForgeCore dynamically links a pinned upstream llama.cpp/ggml build. The
native tree always lives **outside the workspace** under
`$FORGE_LLAMA_DIR` (default `/var/tmp/forge-native`):

```text
$FORGE_LLAMA_DIR/
  src/      pinned upstream checkout (tag v0.5.0) + one documented patch
  build/    CMake tree; shared libs land in build/bin/
  tools/    ephemeral cmake binary (if the system has none)
  models/   optional local test fixtures (never in the repo)
  probes/   ephemeral C harnesses (e.g. the P6 abort battery; never in the repo)
```

`/var/tmp` is the default instead of `/tmp` because `/tmp` is a ~1 GB
tmpfs in this environment while `/var/tmp` sits on the main disk. Both
are outside the persisted workspace, so the native tree is rebuilt per
session. Override `FORGE_LLAMA_DIR` if needed, but never point it under
`/home/user/`.

## Prerequisites

- `git`, `gcc`/`g++`, `make` on `PATH`.
- `cmake` on `PATH`. If the system has none, install the ephemeral
  binary (outside the repo):

```sh
mkdir -p "$FORGE_LLAMA_DIR/tools" && cd "$FORGE_LLAMA_DIR/tools"
curl -sSfL https://github.com/Kitware/CMake/releases/download/v4.1.2/cmake-4.1.2-linux-x86_64.tar.gz -o cmake.tar.gz
tar xzf cmake.tar.gz && rm cmake.tar.gz
```

`sourcing scripts/env.sh` picks this cmake up automatically.

## Quickstart

```sh
. /home/user/forgeCORE/scripts/env.sh
sh /home/user/forgeCORE/scripts/setup-native.sh   # clone (pinned) + CMake build
cd /home/user/forgeCORE && cargo test --workspace
```

`setup-native.sh` clones `--depth 1 --branch v0.5.0`, verifies the
checkout resolves to commit `7fe450e19305b828c199d602c23a8337aaa1f03b`
(aborting otherwise), applies the documented patch under
[`patches/llama-model-residency.patch`](../patches/llama-model-residency.patch),
configures with `BUILD_SHARED_LIBS=ON` and tests/examples/tools off, and
builds with all cores. Re-running it reuses the checkout, skips an
already-applied patch, and rebuilds incrementally.

The patch adds two narrow capabilities:

- `LLAMA_LOAD_MODE_MMAP_NO_PREFETCH = 5`, which passes `prefetch=false` to the
  existing `init_mappings` path.
- `llama_model_release_pages()`, which walks only that model's own
  file-backed mappings and calls Linux `madvise(MADV_DONTNEED)` on their
  mapped fragments. It does not unmap memory or invalidate tensor pointers.
  Other platforms return `ENOTSUP`.
- An execution-quiescence gate. A context acquires a shared lock before
  asynchronous graph submission and releases it only after
  `llama_synchronize` (or context destruction) observes completion. Page
  release takes the exclusive side with `try_to_lock` and returns `EBUSY`
  instead of blocking while any context has work in flight. This protects the
  physical page-release boundary; it is not an intra-decode lifetime proof.
- Model/context lifetime diagnostics: a process-local model instance id,
  active/in-flight context counts, a point-in-time quiescent flag, per-context
  id and execution generation, and residency contract version 1. No native
  pointer is exposed.
- Tensor-to-mapping metadata (`llama_model_tensor_mapping_count` /
  `llama_model_tensor_mapping`) and `llama_model_release_tensor_pages()`,
  which validates tensor indices against exact model-owned mapping fragments.
  Pages shared with unselected file-backed tensors are skipped conservatively.
  Names are borrowed from the model; forgeCORE copies them immediately.
- `llama_model_prefetch_tensor_pages()` as a bounded Linux
  `MADV_WILLNEED` hint over validated tensor pages. It is not a synchronous
  read or a residency guarantee. It may share the execution gate because it
  never removes pages, but it serializes against an exclusive release.
- Read-only tensor graph helpers (`llama_tensor_op_name`,
  `llama_tensor_src_count`, `llama_tensor_src_name`) used by forgeCORE's
  bounded scheduler eval trace. The trace callback itself uses the pinned
  public `llama_context_params.cb_eval`; no native execution behavior changes
  when tracing is disabled.

It does not change `llama_model_params` layout or the default `AUTO`/`MMAP`
behavior. The upstream checkout HEAD remains the pinned SHA; the patch is
tracked in this repository so the native tree is reproducible.

## Pin management

The pin is `LLAMA_TAG`/`LLAMA_SHA` at the top of
`scripts/setup-native.sh` — the single source of truth. To bump:

1. Edit the tag and expected SHA in that script.
2. Refresh the patch against the new checkout and keep it minimal.
3. `rm -rf "$FORGE_LLAMA_DIR/src" "$FORGE_LLAMA_DIR/build"` and re-run
   the script.
4. Re-run the full validation suite; if the C API changed, update
   `forge-sys` and the layout tests, then this doc and `docs/PIVOT.md`.

## Backend selection

The Rust API enumerates whatever the native build registered. This
environment builds CPU-only (no GPU present). To enable accelerators,
reconfigure the native tree with the ggml backend flags, e.g.
`-DGGML_CUDA=ON` (see upstream `ggml/CMakeLists.txt` for the full list),
rebuild, and re-run the suite — `enumerate_devices` and the GPU probe
test pick new backends up with no Rust changes.

## Runtime loader path

`crates/forge-core/build.rs` records an rpath to
`$FORGE_LLAMA_DIR/build/bin`, so `cargo test` binaries find
`libggml`/`libllama` without extra setup. (`cargo:rustc-link-arg` does
not propagate across crates, which is why the rpath lives in
`forge-core`'s build script rather than only in `forge-sys`.)

Downstream crates linking `forge-core` (e.g. RAMforge) must arrange
their own loader path — an rpath on their final binaries or
`LD_LIBRARY_PATH=$FORGE_LLAMA_DIR/build/bin` at runtime.

## Test model fixture (optional)

`tests/ggml_smoke.rs::tiny_fixture_model_loads_when_present` and the
`tests/cpu_decode.rs` suite load a real `.gguf` when `FORGE_TEST_MODEL`
points at one, and skip otherwise. Generate a deterministic 7 KB
fixture with the repo script (needs `pip install gguf numpy` with
`PIP_TARGET` outside the repo):

```sh
. scripts/env.sh
mkdir -p "$FORGE_LLAMA_DIR/models"
PIP_TARGET="$FORGE_LLAMA_DIR/py" pip install gguf numpy
PYTHONPATH="$FORGE_LLAMA_DIR/py" python3 scripts/make-tiny-gguf.py \
    "$FORGE_LLAMA_DIR/models/tiny-llama.gguf"
FORGE_TEST_MODEL="$FORGE_LLAMA_DIR/models/tiny-llama.gguf" cargo test -p forge-core
```

The fixture is a 1-layer LLAMA model (vocab 32, 1176 params); the test
asserts the load reports matching metadata.

## Tokenizer fixture (optional)

`tests/tokenizer.rs` needs a second fixture with a well-formed SPM
vocabulary (the decode fixture has no byte pieces, so upstream byte
fallback throws on any non-empty encode — see the phase-1 report §7).
Generate it with the same script in `--tok` mode (vocab 269:
`<unk> <s> </s>`, ten `tokN` pieces, full `<0xXX>` byte coverage,
`add_bos` + `add_eos`):

```sh
. scripts/env.sh
mkdir -p "$FORGE_LLAMA_DIR/models"
PYTHONPATH="$FORGE_LLAMA_DIR/py" python3 scripts/make-tiny-gguf.py --tok \
    "$FORGE_LLAMA_DIR/models/tiny-tok.gguf"
FORGE_TEST_MODEL_TOK="$FORGE_LLAMA_DIR/models/tiny-tok.gguf" cargo test -p forge-core --test tokenizer
```

## BPE tokenizer fixture (optional)

The same tokenizer suite also covers a live GPT-2-style BPE vocabulary.
The fixture has vocab 11 (`<unk> <s> </s>`, `h e l o`, `he lo hel
hello`), explicit merge ranks, `tokenizer.ggml.pre = "default"`, and no
implicit BOS/EOS. It verifies that native BPE merge ranks produce exact
token ids and that decode reconstructs the pieces.

```sh
. scripts/env.sh
mkdir -p "$FORGE_LLAMA_DIR/models"
PYTHONPATH="$FORGE_LLAMA_DIR/py" python3 scripts/make-tiny-gguf.py --bpe \
    "$FORGE_LLAMA_DIR/models/tiny-bpe.gguf"
FORGE_TEST_MODEL_BPE="$FORGE_LLAMA_DIR/models/tiny-bpe.gguf" cargo test -p forge-core --test tokenizer bpe
```

## DSV4 fixture (optional)

The `tests/kv_state.rs` DeepSeek-V4 regression tests need a fourth
fixture: a minimal loadable `deepseek4`-architecture model (2 layers,
compress ratios 4/128, 2 experts, seed 9) exercising the native
`llama_kv_cache_dsv4` memory class, which asserts `seq < n_seq_max`
even when unified. Generate it with the same script in `--dsv4` mode:

```sh
. scripts/env.sh
mkdir -p "$FORGE_LLAMA_DIR/models"
PYTHONPATH="$FORGE_LLAMA_DIR/py" python3 scripts/make-tiny-gguf.py --dsv4 \
    "$FORGE_LLAMA_DIR/models/tiny-dsv4.gguf"
FORGE_TEST_MODEL_DSV4="$FORGE_LLAMA_DIR/models/tiny-dsv4.gguf" cargo test -p forge-core --test kv_state dsv4
```

## Troubleshooting

- `forge-sys: libggml.so not found …`: run `setup-native.sh` first, or
  export the correct `FORGE_LLAMA_DIR`.
- `SHA mismatch`: the checkout is not the pinned revision — delete
  `src/` and re-run the script (never force it).
- `cmake not found`: install the ephemeral binary (above) and re-source
  `scripts/env.sh`.
- `No space left on device` under `/tmp`: point `FORGE_LLAMA_DIR` and
  `FORGE_TOOLCHAIN_ROOT` at `/var/tmp` (the current defaults).
- GPU present but probe skips: the native build likely lacks that
  backend — reconfigure with the matching `GGML_*` flag.

## Licensing

Upstream is MIT-licensed; see `docs/LICENSING.md` for terms,
attribution, and what is (not) copied into this tree.
