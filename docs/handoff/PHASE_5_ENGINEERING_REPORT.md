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
IMROPE 40 — probed values), and 18 functions: `llama_get_memory`,
`llama_max_parallel_sequences`, `llama_model_rope_type`, all 9
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
compressed-state helpers — undetectable at the pin (no impl getter),
documented on the methods and here; refusing it would strand
legitimately decoded sequences on the other eight implementations,
and slot-disciplined callers (`seq < n_seq_max`) never reach it.
(2) OOM-class `bad_alloc` across the C ABI (same class as P2's
sampler analysis). (3) SWA-family ghosts (evicted base history) do
not shift when the window reports empty — a documented semantic, not
a safety hole (the native call is skipped, so nothing unguarded
executes).

## 10. Tests

+2 unit (`MAX_SHIFT` pin, snapshot accessor roundtrip) and +37
integration in new `tests/kv_state.rs`: handle/limits (2),
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

Full suite on the CPU-only build: **232 green in debug and 232 in
release** (86 lib + 37 kv_state + 31 + 9 + 9 + 21 + 5 + 13 + 14 + 7;
was 193/193). Pre-fixture SKIP path green (232 pass, 37/37 SKIP in
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
  same native calls as P4.
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
2. **DSV4-unified-highseq abort residual** (§9.1) — undetectable,
   documented, unreachable under slot discipline.
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
`FORGE_TEST_MODEL_TOK=$FORGE_LLAMA_DIR/models/tiny-tok.gguf`:

- `cargo fmt --all --check` → clean.
- `cargo check --workspace --all-targets` → clean.
- `cargo test --workspace` → 232 green (86 + 37 + 31 + 9 + 9 + 21 +
  5 + 13 + 14 + 7).
- `cargo test --workspace --release` → 232 green (same split).
- `cargo clippy --workspace --all-targets --all-features -- -D
  warnings` → clean.
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` →
  clean.
- Fixture-less `cargo test --workspace` → 232 pass (37/37 SKIP in
  `kv_state`, verified via `--nocapture`).
- Native probes: `p5-values-probe` exit 0 (all facts as in §3);
  `p5-abort-probe`: `seqrm300` → 134, `seqdiv0` → 136,
  `seqcp-partial` → 134, `seqrm-split` → survived `r=1`,
  `corrupt-struct` → clean 0s.
- `/home/user/` contains only `RAMforge/` + `forgeCORE/`; no
  `target/`, caches, fixtures, or probe artifacts in the workspace
  (all under `/var/tmp`, outside the snapshot).

## 17. Final verdict

**Verdict: COMPLETE WITH LIMITATIONS**

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
§13-C. Limitations: environmental GPU gap (§12), one documented
native abort residual (§9.1), two untestable-but-audited paths
(§10). Recommended next forgeCORE phase: **P6 ggml execution
surface** (buffer introspection would additionally let a future
pass ground the allocated/resident facts P5 had to leave unknown).
