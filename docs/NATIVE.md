# Native llama.cpp/ggml dependency — runbook

ForgeCore dynamically links a pinned upstream llama.cpp/ggml build. The
native tree always lives **outside the workspace** under
`$FORGE_LLAMA_DIR` (default `/var/tmp/forge-native`):

```text
$FORGE_LLAMA_DIR/
  src/      pinned upstream checkout (tag v0.5.0, never edited)
  build/    CMake tree; shared libs land in build/bin/
  tools/    ephemeral cmake binary (if the system has none)
  models/   optional local test fixtures (never in the repo)
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
(aborting otherwise), configures with `BUILD_SHARED_LIBS=ON` and
tests/examples/tools off, and builds with all cores. Re-running it
reuses the checkout and rebuilds incrementally.

## Pin management

The pin is `LLAMA_TAG`/`LLAMA_SHA` at the top of
`scripts/setup-native.sh` — the single source of truth. To bump:

1. Edit the tag and expected SHA in that script.
2. `rm -rf "$FORGE_LLAMA_DIR/src" "$FORGE_LLAMA_DIR/build"` and re-run
   the script.
3. Re-run the full validation suite; if the C API changed, update
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

`tests/ggml_smoke.rs::tiny_fixture_model_loads_when_present` loads a
real `.gguf` when `FORGE_TEST_MODEL` points at one, and skips otherwise.
Generate a deterministic 7 KB fixture with the repo script (needs
`pip install gguf numpy` with `PIP_TARGET` outside the repo):

```sh
. scripts/env.sh
mkdir -p "$FORGE_LLAMA_DIR/models"
PYTHONPATH="$FORGE_LLAMA_DIR/py" python3 scripts/make-tiny-gguf.py \
    "$FORGE_LLAMA_DIR/models/tiny-llama.gguf"
FORGE_TEST_MODEL="$FORGE_LLAMA_DIR/models/tiny-llama.gguf" cargo test -p forge-core
```

The fixture is a 1-layer LLAMA model (vocab 32, 1176 params); the test
asserts the load reports matching metadata.

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
