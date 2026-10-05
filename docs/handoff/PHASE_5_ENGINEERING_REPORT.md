# forgeCORE Phase 5 Engineering Report — KV/State + Memory Facts

**Verdict: COMPLETE WITH LIMITATIONS** (every required P5 primitive
is exposed, validated, and tested; real GPU execution is unverifiable
— CPU-only environment, GPU smoke SKIP-gates like P3/P4; one narrow
native abort residual is documented, not validatable — see §9, §12,
§14).

## 1. Initial audit

`/var/tmp` was wiped again, so the toolchain (Rust 1.99.0 +
rustfmt/clippy) and the native tree were rebuilt from the workspace
scripts first (the installer again ignored the unexported
`CARGO_HOME`/`RUSTUP_HOME` and landed in `/home/user/.cargo` +
`.rustup`; both were moved into `/var/tmp/forge-toolchain` and the
invariant re-verified). The audit then covered: the post-P4 workspace
(193 tests green, re-verified as the baseline), all prior handoff
reports (P0 §§7/11/13/17/18 as the scope baseline — P5 was specified
there as seq ops + state bytes + `Known/Unknown` facts; P4 §16 as the
deferral list), the pinned native tree (all nine `llama_memory_i`
implementations, the C wrappers, the state serializers, the memory
factory, KV dtype validation), and public RAMforge HEAD `6e67f83`
(V4.1.0, verified unchanged, depth-1 clone read-only:
`forgecore.rs` adapter, `kv_cache.rs` shadow, `resource.rs`,
`inference.rs` usage).

Two corrections to the P4 report came out of the audit (§1, §14):
`ContextOptions` gained 13 fields in P4 (15 total), not 12; and
RAMforge's `kv_cache.rs` is 290 lines, not 289. Both fixed in place.
No P1–P4 defect was found; P1–P4 behavior is unchanged apart from two
additive `ContextOptions` fields (native defaults) and extended
constructor wiring that is a no-op for existing callers.

## 2. Exact native revision

llama.cpp v0.5.0, commit `7fe450e19305b828c199d602c23a8337aaa1f03b`,
re-verified by `git rev-parse` after a full clean rebuild (CPU
backends, shared libs, Release). Every symbol, assert, default, and
behavior below was verified against this tree — no newer-upstream API
was copied. ggml is the in-tree 0.25.1.

## 3. Native APIs inspected

Read in full against the pin:

- C surface (`llama.h`): `llama_get_memory`, all 9 `llama_memory_*`
  ops, `llama_max_parallel_sequences` (= `LLAMA_MAX_SEQ` = 256),
  `llama_model_rope_type`, whole-state `get_size/get_data/set_data`,
  per-sequence `get_size/get_data/set_data`, all four file variants,
  all three `_ext` flag variants, `type_k`/`type_v` params.
- C wrappers (`llama-context.cpp`): the `llama_memory_*` wrappers
  null-check the handle but hold **no try/catch**; the byte-oriented
  state methods catch `std::exception` internally (0 = failure) while
  only the file variants add an outer catch; `get_data`/`set_data`
  synchronize first.
- All 9 memory implementations: plain `llama_kv_cache`, recurrent
  (Mamba/RWKV), hybrid, hybrid-iswa, hybrid-idx, iswa, msa, dsa,
  dsa-iswa, dsv4 (DeepSeek-V4) — every `seq_*`, `clear`,
  `state_write/read` for asserts, throws, and bounds.
- The memory factory (`llama_model::create_memory`): BERT-family
  archs get **NULL memory**; recurrent/hybrid/specialized archs get
  their cache; everything else gets a plain KV cache. Also verified:
  all wrapper entry points delegate symmetrically to every inner
  cache, and `PARTIAL_ONLY` is the only state path that touches a
  subset (never used by ForgeCore — flags are hardcoded 0).
- KV dtype validation (`llama_init_from_model` + constructor):
  F16/F16 defaults; MLA/DSV4 require K == V; quantized V requires
  flash-attn enabled (AUTO silently upgrades); quantized block sizes
  must divide head widths; violations return NULL (throw inside the
  constructor is caught by init). Recurrent archs force F32/F32
  regardless of the params.
- `llama_io_write/read_host`: bounds-checked, throw on overrun —
  small destinations and truncated input fail as 0, never overflow.
- `decode` on memory-less contexts reroutes to `encode()`
  (pre-existing behavior, unchanged).

Probed (compile + link + run): rope type 0 (NORM), parallel limit
256, state sizes (empty 17, 3 cells 181 — exactly arch tag + cells +
per-layer headers, **no logits/embeddings** despite the stale header
comment), bit-exact logits after clear→restore, small-buffer → 0,
data-flip accepted (no checksums) vs header-flip → 0, truncation →
0, empty → 0, failed restore leaves the cache empty, per-seq empty
state 16 bytes, unified copy/keep lifecycle, F32 K/V context works,
Q8_0 K refused (NULL). Abort battery, one process each: unified
`seq_rm(300)` → **SIGABRT** (`:389`), `seq_div(d=0)` → **SIGFPE**,
cross-stream partial `seq_cp` → **SIGABRT** (`:506`), split
`seq_rm(5)` on `n_seq_max = 1` → survives (`r=1`: single-stream
caches size the stream map at 256 — ForgeCore still refuses it for
cross-impl coherence, see §4).

## 4. API design

`Memory<'a>` (new `memory.rs`) borrows the context's native object
(`Context::memory() -> Option<Memory>`; `None` = memory-less
architecture). It carries cached validation facts (`n_seq_max`,
unified mode + limit, shift support) and exposes: `clear(data)`,
`remove_range(seq, start, end: Option<u32>)`, `copy_seq(src, dst)`
(full copies only — partial copies abort on DSV4 and split caches),
`keep_seq`, `shift_positions` (rope-gated, `i32`-guarded, empty
no-op, inversion refusal, ±(2³⁰−1) backstop), `scale_positions`
(divisor ≥ 1), `pos_min`/`pos_max` (`None` = empty), `can_shift`,
`seq_limit`.

Two seq-id bounds (the implementations disagree, and no detection
getter exists at the pin): the *general* bound (`n_seq_max`, or 256
unified) for rm/cp/shift/scale/queries/export; the *tight* bound
(`n_seq_max` always) for `keep_seq` and sequence-state restore
(DSV4/recurrent assert it). Negative ids are unrepresentable (`SeqId`
is `u32`), which sidesteps the native `TAG_LLAMA_SEQ_ID_NEG`
inconsistency (kv treats −1 as match-any, recurrent rejects it).

State snapshots are distinct owned types: `State` (whole) and
`SeqState` (per-sequence), with `from_bytes/as_bytes/len/is_empty/
into_bytes`. `Context` gains `state_size`, `export_state`,
`import_state`, `seq_state_size`, `export_seq_state`,
`import_seq_state`. Imports enforce **exact byte consumption**
(trailing garbage rejected); failures leave the cache empty.
`ContextOptions` gains `type_k`/`type_v: DType` (default F16/F16,
native-verbatim, creation failure on invalid combinations — no
getters exist natively, and caching the request would lie for
recurrent archs, so the applied width is observable via
`state_size`). Deliberately unexposed: file persistence (brief
mandates byte-oriented), `_ext` flags/`ON_DEVICE` (device buffers —
P6), negative-seq match-any, partial copies, `swa_full`, `n_rs_seq`,
rope/YaRN (see §15).

Memory facts are methods, not a struct: the only exact byte fact the
C API reports is the serialization size, so `state_size` (+ per-seq)
is documented as *serialization, never residency*. Allocated/used/
resident/scratch bytes are unavailable at the pin (`breakdown()` is
C++-only) — listed as such, never invented. A one-field struct would
fake more certainty than exists.

## 5. FFI additions

`forge-sys`: opaque `llama_memory_i` + `llama_memory_t` alias,
`rope_type` consts (NONE −1, NORM 0, NEOX 2, MROPE 8, VISION 24,
IMROPE 40 — probed values), and 19 functions: `llama_get_memory`,
`llama_max_parallel_sequences`, `llama_model_rope_type`,
`llama_model_meta_val_str` (corrective pass §18: reads
`general.architecture` to detect DeepSeek-V4), all 9
`llama_memory_*` ops, and the 6 byte-oriented state functions. Each
carries a doc comment with its verified contract. The existing
`llama_context_params` TerZGyr already declared `type_k`/`type_v`
(layout test still green — P5 only writes them). Not bound: file
variants, `_ext` variants, `llama_encode` (still unreachable),
`llama_get_model` (still Rc-covered).

## 6. Ownership/lifetime model

`Model → Context → native memory`, each arrow a borrow-or-share:
context holds its `Rc<ModelInner>` share as before; `Memory<'a>`
holds a `PhantomData<&'a mut Context>` plus plain cached facts, owns
nothing, and implements no `Drop`. The `&mut` borrow serializes
sequence ops against decode/state ops with no interleaving. `State`/
`SeqState` are plain owned `Vec<u8>` wrappers, freely movable.
`!Send + !Sync` preserved throughout (`*mut` + `&mut` phantom).
`Context::open` additionally caches `max_parallel_seqs` (one FFI
call) and `shift_allowed` (rope ∉ {MROPE, IMROPE}); both are pure
queries with no side effects, so default-option opens behave
identically to P4.

## 7. KV/memory semantics

Verified per implementation and encoded in validation: single-stream
caches (unified, or split with `n_seq_max == 1`) accept seq ids below
256 while multi-stream caches assert below `n_seq_max`; KV-style
caches free cells shifted below zero while recurrent caches keep the
shifted tail; `seq_rm` returns `false` (mapped to `Error::memory`)
for recurrent partial-tail and DSV4 bounded-range removals; SWA
caches report the window while the base keeps evicted history (an
observed-empty sequence skips the native shift call — documented);
DSV4 reports the current boundary for *both* position bounds;
`can_shift` is false only for Step-3.5 and multi-position models and
gates nothing (informational). Cross-stream copies move real buffers
via a queued update applied at the next decode; metadata is
synchronous, so the copy test asserts bounds without decoding.

## 8. State serialization semantics

Whole state = model-arch tag + live KV cells + per-layer
type/row-size headers (probed byte budget: 17 empty, +164 for 3
tiny cells). Per-sequence state adds a magic + seq-id header (16
bytes empty on single-stream). Structural validation only, no
checksums: wrong arch/magic/counts/types/row sizes fail as 0, flips
inside KV data rows are accepted (both probed and pinned in tests).
Whole-restore clears first and clears again on failure (cache left
empty — probed). Restore onto a different seq id is supported
(tested). Size-then-write is race-free by `&mut` exclusivity, and
export additionally rejects short/overlong writes defensively.
`get_size` must never size a *restore* (empty cache reports small) —
the API makes this unrepresentable (import takes the snapshot's own
length).

## 9. Error/abort analysis

New domains: `Error::memory` (native seq-op refusal) and
`Error::state` (snapshot failures); all validation stays
`Error::invalid`. Upfront-invalid (never reaching native): bad seq
ids (both bounds), inverted/unrepresentable ranges, `d == 0` or
unrepresentable divisor, shifts on MROPE/IMROPE, shifts beyond
±(2³⁰−1), shifts overflowing the observed `i32` range, shifts on
inverted ranges, restores with trailing bytes. Proven by probe:
unified `seq_rm(300)` SIGABRT, `d = 0` SIGFPE, partial cross-stream
copy SIGABRT — all unreachable through the API. Residuals: (1) DSV4
arch + unified + `seq >= n_seq_max` + rm/cp aborts in
compressed-state helpers — ELIMINATED by the corrective pass (§18):
the architecture is detectable after all
(`general.architecture == "deepseek4"` routes to the DSV4 class in
the native factory, the only such site at the pin), so ForgeCore now
refuses those ids pre-FFI on DSV4 while every other implementation
keeps the relaxed unified bound.
(2) OOM-class `bad_alloc` across the C ABI (same class as P2's
sampler analysis). (3) SWA-family ghosts (evicted base history) do
not shift when the window reports empty — a documented semantic, not
a safety hole (the native call is skipped, so nothing unguarded
executes).

## 10. Tests

+2 unit (`MAX_SHIFT` pin, snapshot accessor roundtrip) and +37
integration in new `tests/kv_state.rs` (44 after the §18 corrective
pass: +7 DSV4-unified strict-bound regressions, gated on the new
`FORGE_TEST_MODEL_DSV4` fixture): handle/limits (2),
pos queries (1), seq-id refusal incl. tight bound (2), clear (2),
rm/cp/keep (6), range validation (2), shifts/scales (7), repeated
ops (1), state size/roundtrip/cross-context/corruption/truncation
(5), per-seq state (3), dtypes (2), decode interaction + lifecycle
(3), skip-gated GPU smoke (1). Notable pins: empty state exactly 17
bytes, empty per-seq state exactly 16, bit-exact logits after
restore, data-flip accepted vs header-flip rejected, F32 state
strictly larger than F16, Q8_0 K refused at open. Untestable at the
pin (no fixtures exist): `memory() == None` (needs a BERT GGUF),
MROPE shift refusal (needs an MROPE GGUF) — both paths are
three-line null/flag checks over audited semantics.

## 11. CPU validation

Full suite on the CPU-only build: **239 green in debug and 239 in
release** (86 lib + 44 kv_state + 31 + 9 + 9 + 21 + 5 + 13 + 14 + 7;
was 193/193, then 232/232 before the §18 corrective pass).
Pre-fixture SKIP path green (239 pass, 44/44 SKIP in
the new suite). `cargo fmt --all --check`, `cargo check --workspace
--all-targets`, `cargo clippy --workspace --all-targets
--all-features -- -D warnings`, and `RUSTDOCFLAGS="-D warnings"
cargo doc --workspace --no-deps` all clean (two doc links + seven
test-only `drop()`s fixed, all mine; no pre-existing warnings).

## 12. GPU validation

Not executable: no GPU in this environment (`supports_gpu_offload`
false), so `gpu_state_smoke` reports SKIP. It performs a real decode
+ shift + export→clear→import roundtrip and reports sizes wherever
hardware exists. Same environmental limitation as P3/P4; no claim is
made beyond it.

## 13. RAMforge compatibility audit

Public RAMforge V4.1.0 (`6e67f83`) re-audited read-only after
implementation; untouched.

- **A (provided, must keep working):** the adapter's entire
  `forge_core` surface (`BatchBuilder/Context/Model/ModelOptions/
  ContextOptions`, `enumerate_devices/DeviceInfo/DeviceType`, all
  used methods) is signature- and behavior-identical; P5 is purely
  additive (two defaulted option fields whose F16 values equal the
  native defaults, plus new items). Default-option opens execute the
  same native calls as P4. The §18 corrective pass re-verified this
  read-only: no public signature changed (only validation bodies, a
  private field, and a `pub(crate)` parameter), RAMforge uses only
  `BatchBuilder/Context/ContextOptions/Model`, never enables unified
  KV, and non-DSV4 behavior is bit-identical — the stricter bound
  only bites DSV4-unified ids `>= n_seq_max`, which RAMforge's slot
  discipline never emits.
- **B (P5 fills, for future RAMforge use):** native clear/rm/cp/keep/
  shift/scale/pos-bounds for session prefix management through the
  adapter (replacing manual position bookkeeping); whole + per-seq
  snapshots for session save/restore and multi-tenant state moves;
  exact state bytes replacing f32-shadow math for native-side
  accounting; KV dtype options for VRAM planning input.
- **C (RAMforge-owned):** budgets, eviction/retention choice,
  residency, prefetch, scheduling, placement, orchestration, and any
  migration of the numerical executor — none implemented here.
- **D (legacy):** `kv_cache.rs` (290-line shadow) is **not yet
  replaceable**. Exact blockers: (1) the numerical
  `model_executor`/`inference` forward path reads per-layer K/V rows
  (`get_k/get_v`), and native exposes no such read-back API (state
  export is opaque bytes) — the shadow dies only when RAMforge
  retires that executor for the adapter path, a RAMforge migration
  decision; (2) allocated/resident KV bytes are unavailable at the
  pin (C++-only `breakdown()`), so the shadow's byte math has no
  native counterpart yet. P5 removes every *other* reason to keep it:
  session management, snapshots, and exact serialization bytes are
  now native. No RAMforge migration performed (out of scope).

## 14. Known limitations

1. **GPU execution unvalidated** (environmental) — §12.
2. **DSV4-unified-highseq abort residual** (§9.1) — RESOLVED by
   the §18 corrective pass (pre-FFI rejection proven by regression;
   negative control: the test aborts without the fix).
3. **`memory() == None` and MROPE-refusal paths untested** (no
   BERT/MROPE fixtures craftable from the tiny generator) — §10.
4. **Causal heuristic, decode-throw-on-OOM, `n_seq_max > 256`
   deferral, RANK width, unpinned Rust** — carried over from P4 §15
   unchanged.
5. **P4 report corrections applied**: +13 fields (15 total), 290-line
   shadow — §1.
6. Bootstrap note: the rustup installer ignores unexported
   `CARGO_HOME`/`RUSTUP_HOME` (lands in `/home/user/`); export first
   or move afterwards. `PIP_CACHE_DIR=/var/tmp` contained pip this
   time.

## 15. Explicit non-goals

Not implemented: RAMforge changes/migration, file persistence,
`_ext`/ON_DEVICE state flags, negative-seq match-any, partial
copies, `swa_full`, `n_rs_seq` rollback snapshots, RoPE/YaRN tuning,
SWA sizing, MTP contexts, perf counters (P7), abort callbacks (P7),
threadpool attach, sampler/tokenizer expansion, P6 ggml surface,
budgets/eviction/prefetch/placement/scheduling/orchestration,
generation loops, profiling. Each was audited and its deferral
recorded in rustdoc or here.

## 16. Exact validation commands/results

With `scripts/env.sh` sourced and
`FORGE_TEST_MODEL=$FORGE_LLAMA_DIR/models/tiny-llama.gguf`,
`FORGE_TEST_MODEL_TOK=$FORGE_LLAMA_DIR/models/tiny-tok.gguf`,
`FORGE_TEST_MODEL_DSV4=$FORGE_LLAMA_DIR/models/tiny-dsv4.gguf`
(`--dsv4` mode of `scripts/make-tiny-gguf.py`, seed 9):

- `cargo fmt --all --check` → clean.
- `cargo check --workspace --all-targets` → clean.
- `cargo test --workspace` → 239 green (86 + 44 + 31 + 9 + 9 + 21 +
  5 + 13 + 14 + 7).
- `cargo test --workspace --release` → 239 green (same split).
- `cargo clippy --workspace --all-targets --all-features -- -D
  warnings` → clean.
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` →
  clean.
- Fixture-less `cargo test --workspace` → 239 pass (44/44 SKIP in
  `kv_state`, verified via `--nocapture`).
- Native probes: `p5-values-probe` exit 0 (all facts as in §3);
  `p5-abort-probe`: `seqrm300` → 134, `seqdiv0` → 136,
  `seqcp-partial` → 134, `seqrm-split` → survived `r=1`,
  `corrupt-struct` → clean 0s; §18 `probe_dsv4` matrix: rm/cp/keep/
  simport abort at both geometries, add/div/sexport/decode abort
  past `n_seq_max` 1 (see §18 table).
- `/home/user/` contains only `RAMforge/` + `forgeCORE/`; no
  `target/`, caches, fixtures, or probe artifacts in the workspace
  (all under `/var/tmp`, outside the snapshot).

## 17. Final verdict

**Verdict: COMPLETE WITH LIMITATIONS** (residual-free after §18:
the §9.1 abort is eliminated, not merely documented)

Implemented: borrowed `Memory` handle with clear/rm/cp/keep/shift/
scale/pos-bounds/can_shift, owned `State`/`SeqState` snapshots with
size/export/import (+ per-sequence), exact serialization-size facts,
KV cache dtype options, two error domains — all validated against
the pin, all tested. Audited but intentionally not exposed: file
persistence, `_ext` flags, match-any seq ids, partial copies, dtype
getters (none exist natively), `swa_full`, `n_rs_seq`.
Deferred to later forgeCORE phases: device-buffer state (P6),
allocated/resident reporting (needs upstream C API or P6 buffer
introspection), observability (P7). RAMforge-owned: everything in
§13-C. Limitations: environmental GPU gap (§12), two
untestable-but-audited paths (§10); the §9.1 abort residual is
eliminated by §18. Recommended next forgeCORE phase: **P6 ggml
execution surface** (buffer introspection would additionally let a
future pass ground the allocated/resident facts P5 had to leave
unknown).

## 18. Corrective pass: DSV4/unified strict sequence bound

### 18.1 Pin re-verification

`/var/tmp` was wiped between sessions, so the toolchain, native
tree, fixtures, and probes were rebuilt from the workspace scripts
(`rustup` with exported `CARGO_HOME`/`RUSTUP_HOME`, CMake 4.1.2,
`setup-native.sh`, `make-tiny-gguf.py`). The rebuilt checkout
re-verifies to `7fe450e19305b828c199d602c23a8337aaa1f03b`
(`git describe` → `v0.5.0`): the same pin as P0–P5, so every
P0–P5 audit conclusion still stands and the audit below targets the
identical source.

### 18.2 Corrected native invariant

The §9.1 residual theory ("unified DSV4 compresses with
`n_stream = 1`") was wrong. The DSV4 constructor
(`llama-kv-cache-dsv4.cpp:1210`) takes the public `unified` flag and
**ignores it**:

```cpp
GGML_UNUSED(unified);
// Keep DSV4 KV/state streams per sequence even when public KV mode is unified.
const bool unified_raw = false;
...
const bool unified_compressed = false;
```

All four inner caches (`kv_raw` iswa, `kv_csa`, `kv_hca`, `kv_lid`)
and all three compressor states are built with `unified = false`,
so every inner `n_stream` equals `n_seq_max` — always. The exact,
state-derivable invariant is therefore uniform and simple:

> **On `llama_kv_cache_dsv4`, every per-sequence entry point
> requires `seq < n_seq_max`, unified or not.**

Verified assert/throw sites at the pin (all reached with
caller-controlled ids, none guarded by a wider check first):

| Entry | Site | Mechanism (`seq >= n_seq_max`) |
|---|---|---|
| `seq_rm` full-tail | `clear_compressed`, dsv4:1734 | `GGML_ASSERT(seq < n_seq_max)` → SIGABRT |
| `seq_rm` partial (`p0 > 0`) | dsv4:1463 | returns `false` (safe; maps to `Error::memory`) |
| `seq_cp` | `comp_state::seq_cp`, dsv4:1026–1027 | `GGML_ASSERT(seq < n_stream == n_seq_max)` → SIGABRT |
| `seq_keep` | dsv4:1530 | `GGML_ASSERT(seq < n_seq_max)` → SIGABRT |
| `seq_add`/`seq_div` | inner cache, kv-cache:576/627 | `GGML_ASSERT(seq < seq_to_stream.size())` → SIGABRT past `n_seq_max` 1 (at `n_seq_max` 1 the inner map stays 256 wide and the op silently aliases stream 0) |
| `seq_pos_min`/`seq_pos_max` | dsv4:1555/1563 | guarded: return −1 (safe) |
| seq export | inner `seq_pos_max`, kv-cache:678 via dsv4:1608 | SIGABRT past `n_seq_max` 1 (at 1, the later `rs_idx` throw is caught by `state_seq_get_data`'s handler → size 0) |
| seq restore | `clear_compressed`, dsv4:1734 via dsv4:1656 | SIGABRT (asserts are uncatchable) |
| decode | `dsv4_stream_offset`, dsv4:51 via the context ctor | `throw runtime_error` uncaught through `llama_decode` → terminate past `n_seq_max` 1 (at 1, `n_stream <= 1` skips the throw and the batch silently aliases stream 0) |

Whole-state export/import (`seq = -1`) never touches a per-stream
index and is safe on every geometry. The public C wrappers
(`llama_memory_seq_*`, `llama_state_seq_*`, `llama_decode`) add no
sequence validation (null checks and the state try/catch only), so
all of the above must be pre-validated in ForgeCore.

No per-implementation native getter exists at the pin
(`llama_memory_is_unified` and friends are absent from the headers
and sources), but the implementation *is* exactly derivable: the
`general.architecture` GGUF string maps to `LLM_ARCH_DEEPSEEK4`
(`llama-arch.cpp:83`; unknown strings fail the load, so a loaded
model always carries a recognized one), and the memory factory
constructs `llama_kv_cache_dsv4` at exactly one site
(`llama-model.cpp:2491`, inside `case LLM_ARCH_DEEPSEEK4`, non-MTP
branch). ForgeCore never sets `ctx_type`, so the non-MTP branch
always applies: **`general.architecture == "deepseek4"` ⟺ DSV4
memory**, with no approximation and no fake limit.

### 18.3 Deterministic repro

`scripts/make-tiny-gguf.py --dsv4` (seed 9) fabricates a minimal
loadable DeepSeek-V4 GGUF: 2 layers (compress ratios 4 + 128, so
every compressor cache owns a layer), 2 experts / 1 shared,
hyper-connection mult 4 (native asserts `hc == 4`), indexer head 64
(only the indexer cache takes the forced Hadamard path, and 64 rows
always divide its `nrot` 64). Geometry lessons (all verified
crashes, not guesses): a layer-less forced-rotation cache divides
by zero (`build_input_k_rot`), `hc != 4` aborts the graph, indexer
head `< n_rot` aborts the compressor build, and Hadamard inputs
whose element count is not a multiple of 64 abort the reshape.

`probe_dsv4` (throwaway C probe under `/var/tmp`, one op per
process, SIGFPE/SIGABRT/SIGSEGV backtrace handler) drives
`SeqId(5)` on a unified `n_seq_max = 1` context, then repeats at
`n_seq_max = 4`:

| Op (seq 5, unified) | `n_seq_max` 1 | `n_seq_max` 4 |
|---|---|---|
| `rm` / `cp` / `keep` | SIGABRT | SIGABRT |
| `add` / `div` | silent stream-0 alias | SIGABRT |
| `pos_min` / `pos_max` | −1 (guarded) | −1 (guarded) |
| seq export size+data | 0 (throw caught) | SIGABRT |
| seq restore | SIGABRT | SIGABRT |
| decode | success (aliases stream 0!) | SIGABRT (uncaught throw) |

Valid-id controls all succeed at both geometries (`rm0`, `cp00`,
`decode0`, `can_shift = 0`). The `n_seq_max = 1` silent-aliasing
rows (shift/div/decode touching stream 0 on behalf of seq 5) are
arguably worse than the aborts: no error, wrong cache.

### 18.4 Fix (safe API, additive validation only)

`Context::open` reads `general.architecture` once via the new
`llama_model_meta_val_str` binding and caches `dsv4_memory: bool`
(exact-match only; anything else keeps prior behavior
bit-for-bit). The flag collapses every general sequence bound to
`n_seq_max` on DSV4:

- `Memory::seq_limit()` → `n_seq_max` (covers `remove_range`,
  `copy_seq`, shifts, scales, position queries);
- `check_state_seq` (export side) → `n_seq_max` (`keep_seq` and
  restore already used it);
- `decode` applies the `n_seq_max` check even when unified.

No public signature changed (validation bodies, one private field,
one `pub(crate)` parameter); no global unified disable; no API
removal; no caller `unsafe`. Non-DSV4 paths execute byte-identical
logic (proven by the unchanged legacy suites in §18.6).

### 18.5 Regression tests

+7 integration tests in `tests/kv_state.rs`, gated on the new
`FORGE_TEST_MODEL_DSV4` fixture (SKIP without it): strict limit
reported on unified DSV4 (`seq_limit() == 1`, `can_shift() ==
false`); all eight memory ops refuse seq 1 and 5 (the exact prior
abort cases); valid seq-0 decode/query/shift/surgery succeeds;
state export/import refuse 5 and roundtrip 0; decode refuses 5 and
accepts 0; split DSV4 matches; and a unified `n_seq_max = 4`
context proves the boundary is exactly `n_seq_max` (seq 3 works
everywhere, seq 5 refuses everywhere). Negative control: with the
`seq_limit` gate temporarily reverted, the memory-ops test dies
with SIGABRT; with the fix, it passes — the suite genuinely guards
the residual.

### 18.6 Validation

239/239 debug and release (86 + 44 + 31 + 9 + 9 + 21 + 5 + 13 + 14 +
7); `fmt --check`, `check --workspace --all-targets`, `clippy
--workspace --all-targets --all-features -- -D warnings`, and
`RUSTDOCFLAGS="-D warnings" cargo doc` all clean; fixture-less run
239 pass with 44/44 SKIP in `kv_state`. §16 commands re-run for
this pass. RAMforge V4.1.0 re-audited read-only after the fix
(§13-A): untouched, and the stricter bound only bites ids
RAMforge's slot discipline never emits. `/home/user/` holds only
`RAMforge/` + `forgeCORE/`; no `target/`, fixtures, or probe
artifacts in the workspace.

### 18.7 Limitations delta and no-P6 statement

§14 item 2 is resolved and struck. Everything else in §14 stands
(GPU environment, two untestable-but-audited paths, P4 carryovers).
This pass adds no P6 surface (no ggml execution, no buffer
introspection, no device state) and no RAMforge changes; the DSV4
fixture generator mode and `FORGE_TEST_MODEL_DSV4` are test-only
scaffolding under the existing `/var/tmp` discipline.
