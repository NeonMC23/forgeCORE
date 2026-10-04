# forgeCORE Phase 2 Engineering Report — Sampling Primitives

**Verdict: COMPLETE**

Implements the Phase-0 §18 **P2 — Sampler primitives** scope against
the unchanged upstream pin (llama.cpp `v0.5.0` @
`7fe450e19305b828c199d602c23a8337aaa1f03b`, re-verified this
session). P1 (tokenizer) was audited first and found intact — no
correctness or safety defect, no rewrite; all 103 baseline tests
still pass untouched. No RAMforge change (local tree and public
`V4.1.0` both read-only for this task).

---

## 1. Initial audit findings

**Repository (before changes).** `crates/forge-sys` (479 lines),
`crates/forge-core` with `backend/batch/context/device/dtype/error/
model/runtime/tensor/tokenizer` modules, 4 integration suites, and
the P0/P1 handoff reports — all matching the P1 report's file
inventory. Baseline re-run green: **103/103** (62 lib, 9
`cpu_decode`, 9 `ggml_smoke`, 5 `quant_contract`, 14 `tokenizer`, 4
`forge-sys`).

**No sampler existed.** `grep` for `sampler` hits only the
`llama_context_params.samplers` passthrough fields (backend-sampler
wiring, untouched) and the roadmap text. Per the roadmap rule, P2
was genuinely unimplemented, so it — and only it — was built.

**Key API facts the design rests on (all re-read, not assumed):**
`Logits` is an owned `Vec<f32>` with `values() -> &[f32]`
(`context.rs`); `TokenId = u32` lives in `batch.rs` (single
definition, reused); `BatchBuilder::new(vocab_size)` validates ids
(a sampled id feeds straight back in); errors are string-backed
with per-domain constructors; every handle is `!Send + !Sync`;
`SampleConfig`-shaped validation follows the `ContextOptions` /
`EncodeOptions` `#[non_exhaustive]` + `Default` pattern.

**Stale docs found and fixed:** `README.md` still claimed 81 tests
with tokenization "out of scope" (stale since P1). Updated to the
125-test status (§10). `docs/PIVOT.md` §9 keeps its historical
wording (it describes that milestone's scope, correctly).

---

## 2. Native sampling APIs inspected

All in `include/llama.h` + `src/llama-sampler.cpp` (+
`src/llama.cpp:78` for chain defaults) at the pin. The sampler
layer is a **vtable + chain** design: `llama_sampler { iface,
ctx }`, chains own element samplers, and each element implements
`apply` (mutate/select over `llama_token_data_array`) with
optional `accept/reset/clone/free`.

**Entry-point decision.** Two native flows exist:
`llama_sampler_sample(smpl, ctx, idx)` (context-bound: pulls
logits itself, applies, asserts `selected` in range, accepts) and
`llama_sampler_apply(smpl, &cur_p)` over a caller-built candidate
array. ForgeCore binds **`apply`, not `sample`**: the owned
`Logits` type is the task's composition point, and the ctx-bound
entry would bypass it while coupling the sampler to `Context`
borrows. The wrapper mirrors `llama_sampler_sample`'s canonical
steps exactly (`selected=-1` init → apply → range-check → read
`data[selected].id` → accept), with the native `GGML_ASSERT`
converted to `Error`.

**Constructors audited** (all `new`-based: only fail via C++ OOM
throw, never NULL — hence infallible Rust constructors):
`init_greedy` (strict-`>` scan, first-max-wins, ties → lowest id);
`init_dist` (explicit seed verbatim; `LLAMA_DEFAULT_SEED` →
random-device-or-clock; stores both original and resolved seed);
`init_top_k` (`k<=0` → internal no-op sampler; impl clamps `k` to
size, so `k>=n` keeps all); `init_top_p` (`p>=1` → no-op; else
softmax + sorted cumsum with `min_keep` floor); `init_min_p`
(`p<=0` → no-op; keeps `logit >= max+log(p)` with `min_keep`
fallback); `init_temp` (`t==1` → no-op; `t<=0` keeps only the
argmax, rest `-inf`; else `l_i/t`).

**Lifecycle audited:** `chain_init/add` (add takes ownership, no
NULL checks — wrapper never passes NULL); `chain_n` (no NULL
check — wrapper guarantees liveness); `chain_remove`
(bounds-checked, NULL on OOB, ownership returns to caller);
`accept/apply/reset/free` (NULL-tolerant on the sampler;
`dist.accept` ignores its token in the CPU path;
`dist.reset` re-seeds from the *original* seed);
`get_seed` (dist → resolved `seed_cur`; chain → reverse search
for first non-default seed; NULL-unsafe — wrapper guarantees).
`chain_default_params` = `{ no_perf: true }`.

**Assessed and deliberately NOT bound** (one line each):
`llama_sampler_sample` (bypasses `Logits`, §above);
`llama_set_sampler` + `context_params.samplers` (backend/graph
sampling — execution-path territory); `penalties` (repetition
policy, explicitly out); `temp_ext/typical/top_n_sigma`
(unlisted truncation variants); `xtc/dry/mirostat*`
(roadmap-later); `grammar*` (policy/stateful parsing);
`adaptive_p/infill/logit_bias` (specialized); `clone/copy/name`
(unneeded surface); custom `llama_sampler_i` (user vtables =
unsafe surface); `init_empty` (internal).

---

## 3. Implementation performed

| File | Change |
|---|---|
| `crates/forge-core/src/sampler.rs` | **new**, 528 lines: API + validators + 6 unit tests |
| `crates/forge-core/tests/sampler.rs` | **new**, 340 lines: 13 integration tests |
| `crates/forge-sys/src/lib.rs` | +86: `llama_sampler` opaque type, 3 `repr(C)` structs + layout tests, `LLAMA_DEFAULT_SEED`, 15 decls |
| `crates/forge-core/src/error.rs` | +7: `Error::sample` + test line |
| `crates/forge-core/src/lib.rs` | +3: module, re-exports, crate-doc line |
| `README.md` | status/layout/API-glance refresh (was stale since P1) |

Not touched: tokenizer, all P0/P1 tests, fixtures, `setup-native.sh`,
`env.sh`, RAMforge, past handoff docs. No `.cargo` changes were
needed (the repo has no `.cargo`; env.sh + prebuilt toolchain reused,
offline after bootstrap).

---

## 4. Public Rust API

```rust
pub struct SampleConfig {  // #[non_exhaustive], Default = pure greedy
    pub temperature: f32,      // 0.0 = greedy; else finite, >= 0
    pub top_k: Option<u32>,    // Some(>=1, <= i32::MAX)
    pub top_p: Option<f32>,    // Some((0, 1] finite)
    pub min_p: Option<f32>,    // Some((0, 1] finite)
    pub seed: Option<u32>,     // None = random; greedy ignores it
}
impl SampleConfig {
    pub fn validate(&self) -> Result<()>;
    pub fn build_chain(&self) -> Result<SamplerChain>;  // top_k → top_p → min_p → temp?/greedy → dist?
}

pub struct SamplerChain { /* owns one native chain; !Send + !Sync */ }
impl SamplerChain {
    pub fn new() -> Self;  // + Default
    pub fn push_greedy(&mut self);
    pub fn push_dist(&mut self, seed: Option<u32>);
    pub fn push_top_k(&mut self, k: u32) -> Result<()>;
    pub fn push_top_p(&mut self, p: f32) -> Result<()>;
    pub fn push_min_p(&mut self, p: f32) -> Result<()>;
    pub fn push_temp(&mut self, temp: f32) -> Result<()>;
    pub fn len(&self) -> usize;  // + is_empty
    pub fn remove(&mut self, index: usize) -> Result<()>;  // frees the element
    pub fn seed(&self) -> Option<u32>;   // resolved seed, None if unseeded
    pub fn reset(&mut self);             // replay explicit-seed streams
    pub fn accept(&mut self, token: TokenId) -> Result<()>;
    pub fn sample(&mut self, logits: &[f32]) -> Result<TokenId>;
}
```

Design points: the chain is the primitive (explicit element order,
`min_keep` fixed at the upstream `common` default of 1 — documented,
keeps chains total); the config is the validated one-shot shape
(RAMforge's stateless call style maps onto `build_chain` + `sample`
directly); `temperature == 0.0` is *modal* (greedy selector, no temp
element, no dist) — the same `temp <= 0 → greedy` semantic
RAMforge's `Sampler` uses, so migration preserves meaning; `None`
seed is random with `seed()` as the testable observable; `sample`
takes `&[f32]` so `Logits::values()` feeds it with zero new API on
`Logits`, and unit/integration tests need no model.

---

## 5. FFI changes (`forge-sys`)

Opaque `llama_sampler`; `repr(C)` `llama_token_data { id: c_int,
logit: f32, p: f32 }` (12 B), `llama_token_data_array { data, size,
selected: i64, sorted: bool }` (32 B), `llama_sampler_chain_params
{ no_perf: bool }` (1 B) — each with offset/size layout tests;
`LLAMA_DEFAULT_SEED = 0xFFFF_FFFF`; 15 declarations
(`chain_default_params/init/add/n/remove`, six `init_*`, `apply`,
`accept`, `reset`, `get_seed`, `free`) with a contract comment
stating the `forge-core` safety obligations. `bool`/`c_int`/`u32`/
`usize`/`f32` mappings verified against the header (`size_t
min_keep`, `int64_t selected`, `uint32_t seed`).

---

## 6. Ownership/lifetime model

`SamplerChain` exclusively owns one native chain (`*mut` + `Drop`
→ `llama_sampler_free`, which frees all added elements; `remove`
frees the detached element exactly once). Nothing is borrowed:
chains sample caller-supplied slices, so — unlike `Tokenizer` —
there are no lifetime parameters and no `Model`/`Context`
coupling. `!Send + !Sync` via a `PhantomData<Rc<()>>` marker
(house uniformity; Phase-0 flagged thread-safety as a P2 risk).
The candidate `Vec<llama_token_data>` is built per `sample` call
and outlives the `apply`; no native pointer or buffer escapes
into public API (selected id copied out as `TokenId`).

---

## 7. Safety analysis

| Native hazard (verified in source) | Guard |
|---|---|
| `chain_add/n` deref without NULL checks | wrapper's `raw` is always live (private field, set once) |
| `chain_remove` OOB → NULL | checked → `Error::sample` naming index + length |
| `get_seed` derefs without NULL check | same liveness guarantee |
| `top_p`'s softmax asserts `size > 0`; greedy on empty yields OOB `selected` | `sample` rejects empty input before any call |
| filters-only/empty chain → `selected == -1` | `selected` range-validated → clean `Error`, never the native assert |
| `selected` is an *index*, not an id | id read from `data[selected].id`, then `TokenId::try_from` (total) |
| `u32` ids/`len` vs `i32` native | `try_from` everywhere (`push_top_k`, `accept`, `remove`, `sample` length cap) |
| `dist` RNG state advances per sample | explicit: documented statefulness + `reset` replay + `seed()` observable |
| constructor OOM → C++ throw (abort) | accepted and documented (identical to Rust alloc-OOM semantics) |
| non-finite logits (comparator UB in theory) | documented GIGO, matching upstream (real decode output is finite; no O(n) hot-path tax) |

`unsafe` audit: every new `unsafe` block carries a `SAFETY`
comment naming the upheld invariant; `cargo clippy -D warnings`
is clean; no `unsafe` in tests. Fuzz surface is narrow: all
caller-controlled numbers are validated before crossing FFI; the
only unvalidated input is logit *values* (see GIGO row).

---

## 8. Tests added (125 vs 103)

+22: 6 `sampler.rs` unit (pure validators, no native calls), 13
`tests/sampler.rs` integration (synthetic logits; 1 fixture-gated),
3 `forge-sys` layout tests.

| Integration test | Covers (task §) |
|---|---|
| `greedy_selects_argmax_first_max_wins` | §1 known logits + tie→lowest-id (RAMforge-shaped case as compat witness) |
| `greedy_config_builds_a_seedless_singleton` | default config = greedy, `seed() == None` |
| `temperature_zero_is_argmax_for_any_seed` | §2 boundary (None/0/1234 seeds) |
| `temperature_one_is_identity_for_fixed_seed` | §2 boundary vs bare dist |
| `invalid_push_parameters_are_errors` | §2/§3/§4/§6 invalid (NaN/±inf/neg temp, k=0, k> i32::MAX, bad p) + failed pushes add nothing |
| `top_k_one_is_argmax_for_any_seed` | §3 k=1 |
| `top_k_restricts_and_clamps` | §3 k=2 membership {1,2} over 50 draws; k=1000 valid + exact pin |
| `top_p_pins_and_nucleus_membership` | §4 exact pins (1.0→2, 0.5→2, 0.3→1), p=1⟺bare-dist, skewed {0}-only sweep |
| `min_p_pins_and_argmax_at_one` | min-p=1⟺argmax any seed; 0.5 membership {0,1} |
| `seeded_dist_pins_and_replays` | §5 exact two-draw pins (seeds 0/1/42), fresh-chain equality, `reset` replay, seed echo |
| `different_seeds_sample_differently` | §5 20 fixed seeds reach all 3 ids (deterministic test of cross-seed variance) + random seed resolves |
| `chain_add_remove_len_and_empty_sample` | add/remove/n, filters-only + empty-chain clean errors, `accept` incl. `u32::MAX` rejection, empty-logits error |
| `integration_sampled_id_…` (fixture-gated) | §7 real tiny-llama logits → greedy/dist ids → `BatchBuilder` + `Tokenizer::decode` |

Task-§6 (no panics): every invalid input above arrives as
`Error::sample`; task-§8 (regression): all 103 baseline tests pass
unchanged. Exact-id pins reproduce the pinned `mt19937` stream and
are labeled re-pin-on-pinning-move (same policy as P1's
exact-id tests).

---

## 9. Validation commands and results

Environment: 2 cores, no GPU, rustc/cargo **1.99.0**, cmake 4.1.2,
gcc 14.2.0, upstream re-verified `7fe450e1`, ggml libs 0.25.1
(fresh `/var/tmp` rebuild this session; toolchain unpinned by
design, inherited).

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo check --workspace --all-targets` | clean, no warnings |
| `cargo test --workspace` (both fixtures) | **125 passed, 0 failed**: 68 lib, 9 `cpu_decode`, 9 `ggml_smoke`, 5 `quant_contract`, 13 `sampler`, 14 `tokenizer`, 7 `forge-sys` |
| `cargo test --workspace --release` (same) | 125 passed, 0 failed (identical split) |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | clean |
| `cargo test --workspace` (fixtures unset) | green (fixture suites SKIP) |
| `cargo doc` new warnings | 0 (3 pre-existing elsewhere untouched) |
| Repo cleanliness `find` (`target/`, `*.gguf`, `*.so`, `__pycache__`, `.cargo`, editor files) | no hits |
| `git status` | n/a — no git repo (owner decision, per P1) |

One test-logic failure occurred mid-task (`reset` replay compared
against a misaligned stream position); it was a test bug, fixed by
comparing to the pinned first draw — implementation untouched.
Fixture/model: existing `tiny-llama.gguf` (+`tiny-tok.gguf` for
the tokenizer suite); no new models or artifacts added.

---

## 10. Known limitations

1. **No penalties/repetition control** — out of scope by brief;
   chains needing history-weighted sampling await a later pass
   (`accept` is already exposed for it).
2. **No grammar/structured samplers** — out of scope; RAMforge uses
   none (verified by repo-wide grep).
3. **`min_keep` fixed at 1** — documented; matches upstream
   `common` and keeps chains total.
4. **Non-finite logits are GIGO** (§7) — documented on the module.
5. **Exact-id pins are pin-sensitive** — labeled in-test; re-pin if
   upstream moves (same as P1).
6. **Only `dist` carries RNG state** — `seed()` reports `None` for
   greedy/filter-only chains (by native design).
7. **Random-seed `reset()` re-randomizes** — documented; only
   explicit seeds replay.
8. **No context-bound sampling** — `llama_sampler_sample` not
   bound by design (§2); equivalent flow via `apply`.

---

## 11. Compatibility considerations for RAMforge

Inspected public RAMforge `HEAD` `6e67f83` (`V4.1.0`; local
`V3.0.2` is stale — inspection used a read-only `/var/tmp`
clone; nothing modified).

| Finding | Class | Note |
|---|---|---|
| `Model/Context/Batch/decode/Logits` adapter use (`forgecore.rs`) | **A** already provided | unchanged by P2 |
| Greedy/temp/top-k/top-p over `&[f32]` → `u32` (`sampling.rs`) | **B** missing primitive → **now provided** | `SamplerChain` + `SampleConfig` cover it natively, seeded (their `thread_rng` is unseeded — a gap P2 fills) |
| `temp <= 0 → greedy` semantic | **B → provided** | preserved exactly (`temperature == 0.0` modal greedy) |
| Stateless-per-call usage (`Sampler` fresh per call) | **B → provided** | maps to `build_chain` + `sample` (chains are cheap; reuse is optional) |
| Generation loops, EOS/stop policy, temp schedules, `tmp:sampling` budgets, device selection, CLI/TUI, prompt handling | **C** RAMforge-owned | untouched, as briefed |
| Pure-Rust `sampling.rs::Sampler` | **D** legacy | replaceable by P2 primitives at RAMforge's pace; not architecture input |
| `tokenizer::Tokenizer::from_gguf` inspection parser | **D** legacy | P1 already supersedes for execution; migration is RAMforge's business |
| No repetition/grammar/mirostat use anywhere in RAMforge | — | confirms P2's out-of-scope list matches actual need |

RAMforge pins public forgeCORE `944bc3c`; consuming P2 is a
RAMforge-side rev bump + adapter extension (their
`ForgeCoreAdapterError` already has a per-domain shape that fits a
future `Sample` variant). No action taken here.

---

## 12. Recommended next roadmap phase

**P3 — Model offload wiring** (Phase-0 §18, unchanged): extend
`ModelOptions` (`n_gpu_layers`, `main_gpu`, `split_mode`,
`tensor_split`, device selection from `DeviceInfo`) with honest
validation (refuse GPU requests with no GPU registered — error,
never silent CPU fallback); bind
`llama_supports_gpu_offload/mmap/mlock`, `llama_max_devices`.
Single phase only — no P4+ work is included or started.
