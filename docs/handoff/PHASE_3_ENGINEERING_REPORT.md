# forgeCORE Phase 3 Engineering Report — Model Offload Wiring

**Verdict: COMPLETE WITH LIMITATIONS** (all P3 acceptance criteria met;
GPU execution itself is unverifiable — the build/validation environment
is CPU-only, so the real-offload path is implemented, validation-gated,
and skip-covered, but has never executed on a GPU; see §13, §17).

## 1. Executive summary

P3 wires the native model-offload surface through ForgeCore with exact
pinned-upstream semantics (llama.cpp v0.5.0,
`7fe450e19305b828c199d602c23a8337aaa1f03b`, re-verified after a full
native rebuild). `ModelOptions` grows six coherent fields (GPU layer
count, split mode, main GPU, tensor split, device list, load mode);
`device.rs` gains four capability facts (`max_devices`,
`supports_mmap/mlock/gpu_offload`); `forge-sys` gains four FFI
functions, two enum-const modules, and a one-line soundness fix to the
`llama_model_params.devices` declaration (missing pointer indirection).

The defining design rule — audited, probed, and tested — is **no
silent CPU fallback**: upstream loads successfully and runs on the CPU
when GPU layers are requested but no GPU device exists (proven with a
C probe, §9), so ForgeCore refuses such requests upfront with an
explicit `unsupported` error. Every other new option is validated at
the Rust boundary before any native call; native NULL returns keep
their existing `model error` mapping. CPU-only behavior is
byte-for-byte unchanged: default options pass upstream exactly the
parameters plain loads always have.

150 tests green in debug and release (125 inherited + 25 new), strict
clippy clean, `cargo doc -D warnings` clean, workspace cleanliness
verified. No P4/P5/P6 work was started; RAMforge was inspected
read-only and classified A/B/C/D (§3).

## 2. Scope and non-goals

In scope (Phase-0 §18 + P3 task): GPU layer count, main GPU, split
mode, tensor split, explicit device lists resolved from `DeviceInfo`,
device capability facts (offload-supported, max devices, mmap/mlock),
mmap/mlock load modes, `ModelOptions` extension with Rust-boundary
validation, explicit errors for every misconfiguration, layout/enum
verification, docs, and this report.

Out of scope and untouched: KV-cache offload and context offload
toggles (`offload_kqv`, `op_offload` — P4), batch extensions and
scheduler surface (P5), ggml surface beyond what exists (P6),
streaming/progress callbacks, `tensor_buft_overrides`, `kv_overrides`,
`load_mtp`, `no_alloc`, lazy modes, Direct I/O, RPC clients, and any
RAMforge-side placement, scheduling, budget, eviction, or
multi-GPU-orchestration policy. No "multi-GPU orchestration" is
claimed anywhere: this phase binds the native knobs, nothing more.

## 3. Audit findings

**Repo state.** P1/P2 intact (103 + 22 = 125 tests). `/var/tmp` had
been wiped again, so the toolchain and native tree were rebuilt from
the workspace scripts (Rust 1.99.0; CMake 4.1.2) and the pin
re-verified by `git rev-parse` (`7fe450e1...`) before any other audit
step.

**`model.rs` pre-state.** `ModelOptions` was `check_tensors`-only and
`load_with_options` hardcoded `n_gpu_layers = 0`: CPU-locked by code,
not just by build. Everything else rode native defaults.

**`device.rs` pre-state.** `enumerate_devices`/`DeviceInfo` existed
with `{index, name, description, device_type, memory_free,
memory_total}`; `Backend::open_device` re-resolves by index.

**RAMforge (public V4.1.0, HEAD `6e67f83`, re-verified unchanged via
the GitHub API, depth-1 clone inspected read-only).** A/B/C/D for P3:

- **A (consumed, frozen):** `enumerate_devices`/`DeviceInfo` are
  consumed *structurally* — RAMforge tests construct
  `ForgeDeviceInfo{index, name, description, device_type,
  memory_free, memory_total}` literals. **P3 must not add fields to
  `DeviceInfo`** (and `#[non_exhaustive]` would break them too). All
  new facts went into new functions/types instead.
- **B (the gap P3 fills):** RAMforge's `execution.rs` names it
  verbatim — "model/context device binding"; non-CPU devices carry
  `selectable=false`; `static_model_offload: Unsupported` everywhere;
  `StrategyId::GpuLayerOffload` plan compilation returns
  `UnsupportedGpuExecution` unconditionally. P3 provides exactly the
  binding those gates wait for; flipping them is RAMforge's change.
- **C (RAMforge-owned, not mirrored):** offload strategy planning,
  `GpuPreference` policy, device-selection syntax, accounting domains,
  capability derivation, CLI presentation, calibration.
- **D (nothing new):** no legacy/dead surface relevant to P3.

## 4. Native offload path semantics (pinned source)

Read in full: `llama_prepare_model_devices` (`src/llama.cpp`),
`llama_model_base::load_tensors` and `llama_model::n_gpu_layers`
(`src/llama-model.cpp`), `llama_model_default_params`, the loader's
`load_mode` handling (`src/llama-model-loader.cpp`), and the
`llama_supports_*` implementations.

- **Device resolution.** Explicit `params.devices` (NULL-terminated)
  is used **verbatim, unfiltered** — even CPU devices. `NULL` selects
  the default scan: skip CPU/ACCEL, order RPC servers first, then
  GPUs (deduplicated by `device_id`), then iGPUs only if no discrete
  GPU exists. Under `SPLIT_MODE_NONE` with a non-empty list,
  `main_gpu < 0` *clears the list* (forced CPU), `main_gpu >= size`
  fails the load, otherwise only `devices[main_gpu]` is kept.
  `main_gpu` is read **nowhere else**.
- **Silent-CPU catalog.** Empty device list ⇒ `act_gpu_layers = 0`
  (`llama-model.cpp`, "devices.empty() ? 0") — load succeeds, runs on
  CPU. `n_gpu_layers`: negative ⇒ all layers + output (`>= 0 ? n :
  n_layer_all + 1`); positive clamped with `min`.
- **`SPLIT_MODE_TENSOR`.** Requires ≥1 device (else load fails, even
  with 0 layers) and throws a load error for architectures without an
  implementation.
- **`supports_gpu_offload`.** Loads the registry itself, then reports
  GPU-or-IGPU-registered **or RPC support compiled in**. ACCEL
  devices count for neither this flag nor default selection.
- **`max_devices`.** Constant `16`; sizes `tensor_split`.
- **`supports_mmap/mlock`.** Compile-time platform flags
  (`_POSIX_MEMLOCK_RANGE`/Windows ⇒ true, incl. Linux). Explicit
  mmap where unsupported ⇒ WARN log + downgrade (load continues);
  mlock runtime failures degrade to warnings inside native code.
- **Load failure contract.** Error (-1) and tensor-load/cancel (-2)
  paths both return NULL; ForgeCore's existing NULL ⇒ `Error::model`
  mapping is preserved unchanged.

## 5. Public API (`crates/forge-core/src/model.rs`, `device.rs`)

```rust
pub enum GpuLayers { Cpu, Count(u32), All }          // default: Cpu
pub enum SplitMode { None, Layer, Row, Tensor }      // default: Layer (native default)
pub enum ModelLoadMode { Auto, NoMmap, Mmap, Mlock, MmapMlock } // default: Auto

#[non_exhaustive]
pub struct ModelOptions {
    pub check_tensors: bool,          // unchanged
    pub gpu_layers: GpuLayers,
    pub split_mode: SplitMode,
    pub main_gpu: usize,              // read only under SplitMode::None
    pub tensor_split: Option<Vec<f32>>,
    pub devices: Option<Vec<DeviceInfo>>,
    pub load_mode: ModelLoadMode,
}

pub fn max_devices() -> usize;
pub fn supports_mmap() -> bool;
pub fn supports_mlock() -> bool;
pub fn supports_gpu_offload() -> bool;
```

Defaults are chosen so `ModelOptions::default()` passes upstream
exactly what plain loads always have: `n_gpu_layers = 0` (forced, as
before), native-default split (`LAYER`, deliberately not `None`),
native-default load mode (`AUTO`, preserving per-device mmap
resolution), `main_gpu = 0`, NULL split, NULL devices. New fields on
the `#[non_exhaustive]` struct are backward compatible. `Model` stays
`!Send + !Sync`; no raw pointers in the safe API.

## 6. FFI surface (`crates/forge-sys/src/lib.rs`)

- **Soundness fix:** `llama_model_params.devices` was declared
  `ggml_backend_dev_t` (one indirection short of upstream's
  `ggml_backend_dev_t *`). Fixed to `*mut ggml_backend_dev_t`.
  Layout-identical (offset 0, 8 bytes); the existing layout test
  passes unchanged; nothing previously wrote the field. Found by the
  P3 header audit, confirmed by probe (`sizeof(devices[0]) == 8`).
- **Const modules:** `split_mode::{NONE,LAYER,ROW,TENSOR}` = 0–3,
  `load_mode::{AUTO,NONE,MMAP,MLOCK,MMAP_MLOCK}` = -1–3
  (`DIRECT_IO` = 4 deliberately unbound — no API selects it).
- **Functions:** `llama_max_devices() -> usize`,
  `llama_supports_mmap/mlock/gpu_offload() -> bool`.
- **Rejected:** `ggml_backend_dev_get_props` (per-device caps) and
  `llama_supports_rpc` — no P3 consumer; binding them would expand
  the FFI surface without a caller. `device_id`-based dedup
  replication was considered and rejected as over-engineering (§11).

Verification: a C probe compiled **and linked** against the pinned
build printed max=16, mmap/mlock=1, offload=0 (this box), all enum
ints, `sizeof(llama_model_params) == 80` with all 17 offsets matching
the Rust layout test exactly, and native defaults
(-1/1/-1/1/0/NULL/NULL/0). No new `repr(C)` structs were needed.

## 7. Ownership and lifetimes

- **Device handles** in an explicit list are resolved from the
  process-global registry at load time (bounds-checked like
  `open_device`), assembled into a NULL-terminated `Vec` owned by
  `load_with_options`, and borrowed by the native call. Upstream
  copies the *handles* into the model; no ownership transfers, no
  post-load lifetime coupling beyond the registry's process-wide
  lifetime. `DeviceInfo` snapshots carry no handles, so staleness
  reduces to the re-checked index bound.
- **`tensor_split`** is borrowed from the caller-owned `Vec` for the
  call duration; upstream copies the entries into
  `tensor_split_owned` during load (`llama-model.cpp`), so no
  post-load borrow exists.
- **Model/Context sharing** unchanged (`Rc<ModelInner>`; contexts
  hold a share). No new `unsafe` outside the documented FFI boundary;
  `unsafe` blocks carry SAFETY comments in the established style.

## 8. Validation and error taxonomy

Validation runs in full before any native call, in this order:
(1) layer-count conversion, (2) split values, (3) device-list
shape/indices, (4) split length vs effective devices, (5) `main_gpu`
bound (only under `SplitMode::None`), (6) `main_gpu` conversion,
(7) offload-availability refusal. Error domains follow existing
conventions (`invalid` for malformed config, `unsupported` for
well-formed-but-unsatisfiable, `model` for native NULL):

| # | Condition | Error |
|---|-----------|-------|
| 1 | `Count(n)`, `n > i32::MAX` | `invalid: gpu layer count …` |
| 2 | split empty / negative / NaN / infinite | `invalid: tensor_split …` |
| 3 | `devices == Some([])` | `invalid: device list …` |
| 4 | device index ≥ registry count | `invalid: stale device index …` (same message as `open_device`) |
| 5 | split shorter than effective devices | `invalid: tensor_split has …` |
| 6 | `main_gpu` out of range (`SplitMode::None`) or unrepresentable | `invalid: main GPU index …` |
| 7 | GPU layers + default selection + `!supports_gpu_offload()` | `unsupported: GPU offload of … but no GPU device …` |
| 8 | native NULL (bad file, TENSOR w/o devices/arch, bad main_gpu vs deduped list, …) | `model error: failed to load …` (message unchanged) |

## 9. No-silent-fallback rule

**Statement.** A request that names GPU execution must either execute
on a GPU or fail explicitly. It must never succeed on the CPU.

**Why it is needed.** The C probe proved upstream's default
(`n_gpu_layers = -1` = ALL, NULL devices) loads and runs on CPU-only
hardware with exit success — silent CPU execution. Three silent
vectors were catalogued: empty-device ⇒ `act_gpu_layers = 0`,
`main_gpu < 0` under `SPLIT_MODE_NONE` ⇒ list cleared, and
mmap-unsupported ⇒ warn-and-downgrade (device-preserving, therefore
passed through and documented rather than refused).

**Enforcement.** Rule (7) above refuses GPU-layer requests under
default selection when no GPU exists. Explicit device lists pass
through verbatim *including CPU devices* — naming a CPU device is an
informed selection, not a fallback, and type-filtering would be
policy (RAMforge's). Negative `main_gpu` (the native force-CPU
encoding) is unrepresentable (`usize`). The refusal message names the
predicate (`llama_supports_gpu_offload is false`) so callers can
branch on it.

## 10. Tensor-split length rule (safety argument)

Upstream reads `tensor_split[0..n_devices)` without a length check —
a short array is an out-of-bounds read. ForgeCore makes that
unreachable: explicit lists require `len >= list.len()` (exact N);
default selection requires `len >= device_count()` (safe
over-approximation — the default selection is always a registry
subset, in both the normal and TENSOR-meta paths). Values must be
finite and non-negative; all-zero keeps its native meaning (split by
free memory). `max_devices()` (16) is exposed so callers can size
arrays; over-long arrays are safe (unread tail).

## 11. `main_gpu` semantics

`usize`, default 0, bounds-checked **only** under `SplitMode::None`
— upstream ignores the field under every other mode, and ForgeCore
refuses nothing upstream would accept (tested: LAYER + `main_gpu =
9999` + bogus path reaches the native call and fails as a plain
model error). Under default selection the check uses the
registry-count over-approximation (exact pre-validation would require
replicating `device_id` dedup — rejected, §6); residuals fall through
to the native exact check and surface as native load failures,
documented in `ModelOptions` rustdoc.

## 12. mmap/mlock semantics

`ModelLoadMode` mirrors the native enum's mmap/mlock subset with
exact values (AUTO=-1 … MMAP_MLOCK=3); default AUTO preserves the
per-device mmap resolution that explicit modes bypass. Requesting
mmap/mlock where unsupported follows the native downgrade path
(warn, load continues) — a device-preserving degradation, so passed
through rather than refused, and documented. All five modes are
tested loading the CPU fixture on Linux (where both flags are true);
the downgrade paths themselves are not executable on this platform
and are documented from source.

## 13. Fixtures and environment

Tiny fixtures rebuilt by `setup-native.sh`
(`tiny-llama.gguf`/`tiny-tok.gguf`); tests use `FORGE_TEST_MODEL`
(+`_TOK`) with the established SKIP convention. Environment: 2-core
Xeon, no `nvidia-smi`, no `/dev/{nvidia,dri,accel}` — CPU-only, so
**GPU execution is unverifiable**: the offload smoke test reports
SKIP here (and asserts nothing fake). What *is* verified without a
GPU: every validation error, the refusal error, default-CPU
regression, all load modes, explicit-device verbatim tolerance
(incl. duplicates), native-failure mapping (TENSOR w/o devices ⇒
model error), and capability-flag consistency
(`supports_gpu_offload() == any(Gpu|Igpu)`, which also covers
RPC-typed devices). If a GPU appears, `gpu_offload_smoke` loads the
fixture with `Count(1)` + default selection and prints the
device/backend/config/result report.

## 14. Test inventory (150 vs 125)

+4 unit tests in `model.rs` (defaults incl. native-default split/load
modes; `GpuLayers`/`SplitMode`/`ModelLoadMode` ⇒ native-int mappings;
split-value validation) and +21 integration tests in the new
`tests/offload.rs` (4 capability, 4 CPU-regression, 10 validation, 1
native-failure mapping, 1 skip-gated smoke, 1 consistency). Pre-fixture
SKIP path verified green (21/21 without `FORGE_TEST_MODEL`).
Per-suite debug+release totals: 72 + 9 + 9 + 21 + 5 + 13 + 14 + 7 =
150, zero failures. No test assumes a GPU; no GPU behavior is faked.

## 15. Validation results

With `scripts/env.sh` sourced and fixtures exported:

- `cargo fmt -p forge-core -p forge-sys -- --check` — clean.
- `cargo check --all-targets` — clean.
- `cargo test --workspace` (debug) — 150 green.
- `cargo test --release --workspace` — 150 green.
- `cargo clippy --all-targets -- -D warnings` — clean.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` — clean, after
  fixing one new private-intra-doc link (mine) and three
  pre-existing rustdoc-1.99 lints in untouched P1/P2 doc comments
  (`mod@reference` disambiguation, two redundant explicit targets;
  comment-only, zero behavior change — see §16).
- Cleanliness `find` under `/home/user` (target/cargo/rustup/so/gguf
  names) — empty; `/home/user` holds only `forgeCORE/` + `RAMforge/`.
- C probes (compile + link + run vs pinned build): FFI
  existence/signatures/enum values/layout/defaults, and the
  silent-CPU + TENSOR-failure behaviors — all as reported above
  (probes lived in `/var/tmp`, outside the persisted workspace).

## 16. Files changed

- `crates/forge-sys/src/lib.rs` — `devices` indirection fix,
  `split_mode`/`load_mode` consts, 4 capability FFI decls + docs.
- `crates/forge-core/src/device.rs` — 4 capability functions + docs.
- `crates/forge-core/src/model.rs` — `GpuLayers`/`SplitMode`/
  `ModelLoadMode`, extended `ModelOptions`, load-time resolution +
  validation, rustdoc (offload model, failure behavior, ownership,
  RAMforge-owned remainder), +4 unit tests.
- `crates/forge-core/src/lib.rs` — exports (7 new names) + one
  rustdoc-link upkeep edit.
- `crates/forge-core/tests/offload.rs` — new, 21 tests.
- `crates/forge-core/src/context.rs`,
  `src/reference/quant.rs` — one redundant-link wording fix each
  (rustdoc-lint upkeep only; no code touched).
- `README.md` — tagline, layout, API glance, status (150 tests,
  offload in scope, KV-offload/orchestration still out).
- `docs/handoff/PHASE_3_ENGINEERING_REPORT.md` — this file.

P1/P2 behavior is unchanged: no sampler/tokenizer/context/batch
logic touched; `DeviceInfo` shape frozen per §3; default loads pass
identical native parameters to before (§5).

## 17. Known limitations and residual risks

1. **Real offload never executed here** (CPU-only environment).
   Residue: run `gpu_offload_smoke` on GPU hardware before claiming
   end-to-end offload. The risk is contained — validation logic is
   fully tested and the native call is the same one probed — but the
   claim is honestly withheld: hence COMPLETE WITH LIMITATIONS.
2. **Refusal predicate follows upstream's definition**
   (GPU/IGPU/RPC). Exotic registries (ACCEL-only + TENSOR +
   default selection) are refused conservatively; explicit device
   selection remains available. Documented, deliberate.
3. **`main_gpu` vs default selection is upper-bound checked**; exact
   enforcement stays native (surfaces as a model error). Requires no
   action unless RAMforge needs pre-flight exactness (would need
   `device_id` binding — a future phase, not P3).
4. **mmap/mlock downgrade paths** are source-documented, not executed
   (unsupported-platform behavior). No action on supported platforms.
5. **Rust 1.99.0 unpinned** (rustup drift,/*.workspace has no
   toolchain pin) — pre-existing condition, unchanged by P3.
6. **Doc-lint upkeep** in P1/P2 comments (§15) reflects rustdoc 1.99
   lints; if the toolchain moves again, `cargo doc -D warnings` may
   need another comment-only pass.

## 18. Recommended next phase

**P4 (context/KV-cache offload + residency controls)** — the natural
continuation: `offload_kqv`/`op_offload` wiring, KV-cache placement
and unified-memory options, and whatever residency introspection the
pinned API exposes, with the same no-silent-fallback discipline
(verify whether context creation degrades silently without a GPU
before designing the refusal rule). No P3 blocker stands in its way.
Do not start batch extensions (P5), ggml surface (P6), streaming, or
any placement/scheduling policy — those remain later phases' or
RAMforge's work.
