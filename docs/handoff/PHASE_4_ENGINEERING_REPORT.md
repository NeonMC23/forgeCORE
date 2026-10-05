# forgeCORE Phase 4 Engineering Report — Context/Batch Extension

**Verdict: COMPLETE WITH LIMITATIONS** (every required P4 primitive is
exposed, validated, and tested; real GPU execution of the context
paths is unverifiable — the environment is CPU-only, so the GPU smoke
test SKIP-gates exactly like P3's; see §13, §15).

## 1. Initial audit

`/var/tmp` was wiped again, so the toolchain (Rust 1.99.0, rustfmt,
clippy) and the native tree were rebuilt from the workspace scripts
before any other step. The audit then covered: the P1/P2/P3 workspace
state (150 tests, untouched since P3), all prior handoff reports
(Phase-0 §§7/11/18 as the P4 scope baseline, P3 §18 as the handoff
note), the pinned native tree (`llama_context_params`, init, decode,
batch allocation, offload flags, embeddings, getters), and the public
RAMforge repo (HEAD `6e67f83` V4.1.0, verified unchanged, depth-1
clone inspected read-only).

Two concrete P1 defects were found by the audit and fixed in P4 (see
§3): a latent decode abort on oversized batches and a dishonest
`n_ctx()` getter. Everything else in P1/P2/P3 is byte-identical apart
from two `n_ctx` test pins updated to the corrected values and
comment-only rustdoc upkeep.

## 2. Exact native revision

llama.cpp v0.5.0, commit `7fe450e19305b828c199d602c23a8337aaa1f03b`,
re-verified by `git rev-parse` after a full clean rebuild (CPU
backends, shared libs, Release). Every symbol, enum value, default,
and behavior below was verified against this tree — no newer-upstream
API was copied. ggml is the in-tree 0.25.1 (`7fe450e`).

## 3. Existing Context/Batch architecture

Pre-P4 state: `ContextOptions { n_ctx, n_threads }` (`#[non_exhaustive]`,
manual `Default`); `Context::open` set only `n_ctx`/`n_threads`/
`n_threads_batch` (gen+batch locked); `BatchBuilder` staged single-seq
token batches with a shared seq-id-0 `Box`; `decode` refused only
empty batches and trusted upstream error codes for everything else;
`logits` copied `n_vocab` rows.

**Defect 1 (safety, fixed):** `decode` documented that "batch sizing
is upstream's domain: violations come back as error codes". False:
`llama_context::decode` holds `GGML_ASSERT(n_tokens <= n_batch)` and
`GGML_ASSERT(causal || n_ubatch >= n_tokens)` with no pre-check, and
`GGML_ASSERT` aborts unconditionally (verified in `ggml.h`, proven by
probe: exit 134). Any caller decoding more than `n_batch` tokens
aborted the process. P1 never hit it only because no test oversized a
batch. P4 validates both bounds up front (§10).

**Defect 2 (correctness, fixed):** `n_ctx()` returned the *requested*
length (or `n_ctx_train`), but upstream pads the effective length up
to a multiple of 256 (and rounds to a multiple of `n_seq_max`). The
tiny fixture reported 64 while native used 256. P4 queries all
effective values back after `open` and reports those (§5).

## 4. Native APIs inspected

Read in full against the pin: `llama_context_params` (all 37 fields)
+ `llama_context_default_params`; `llama_init_from_model` (all six
NULL conditions + the `n_seq_max > 256` throw-then-NULL path);
`llama_context` constructor (padding, clamping, resolution rules);
`llama_batch_allocr::init` (token/seq/position/consistency validation,
all safe `-1` paths); `decode` (asserts, KV-slot `1`, output reserve);
ubatch/graph/output processing; `offload_kqv` (KV-buffer placement
hint, KQV-CPU pinning, pipeline-parallel gate) and `op_offload`
(scheduler hint) — both advisory, both no-ops on CPU-only; embeddings
input (no mode gate — decodes anywhere) and output (per-token and
pooled rows, widths, NULL contracts); pooling/attention/flash enums
and resolution; `set_n_threads`/`set_causal_attn`/`set_embeddings`/
`synchronize` (all infallible field updates or scheduler sync);
actuals getters; `llama_batch_init/free/get_one` (`get_one` marked
"avoid using it" upstream); `llama_encode` (unreachable — only for
memory-less contexts, which require `vocab_only`, unexposed).

Probed (compile + link + run, one process per abort scenario):
enum ints (pool −1..4, attn −1..1, flash −1..1); `n_embd*` = 8,
`has_encoder` = 0, `n_cls_out` = 1; effective sizes incl. the 256-pad;
`set_n_threads` roundtrip; **65 > 64 → abort**; **`n_batch = 0` →
abort inside the constructor**; pooling ± embeddings (both fine);
**non-causal 600 > ubatch 512 → abort**; embd-input batch on a plain
context (fine, logits produced).

## 5. Public Rust API changes

`ContextOptions` (+13 fields, 15 total, all defaulting to pre-P4 behavior):
`n_batch` (2048; 0 refused — aborts upstream), `n_ubatch` (512; 0 =
follow `n_batch`, native rule), `n_seq_max` (1), `n_threads_batch` (0
= follow `n_threads`, preserving lockstep), `n_outputs_max` (0),
`n_outputs_max_per_seq` (1), `offload_kqv`/`op_offload` (true,
advisory), `kv_unified` (false; required for coupled batches),
`embeddings` (false), `pooling`/`attention`/`flash_attn` (native
defaults). New enums `PoolingType`, `AttentionType`, `FlashAttnType`
with exact native mappings.

`Context`: effective getters `n_ctx` (now honest), `n_ctx_seq`,
`n_batch`, `n_ubatch`, `n_seq_max`, `n_threads`, `n_threads_batch`,
`causal_attn`, `pooling`; `set_n_threads` (≥1 validated),
`set_causal_attn` (tracked), `synchronize`, `embeddings(i)` and
`embeddings_seq(s)` returning the new owned `Embeddings` type;
`decode` with abort-preventing validation. `Model`: `n_embd_inp/out`,
`n_cls_out`, `has_encoder`.

`batch.rs`: `SeqId` type; `push_on_sequences` /
`push_embd_on_sequences` (non-empty, ≤256 lists); embedding mode via
`new_embeddings` (0 width refused) + `push_embd` (exact-width rows);
homogeneous-mode enforcement; `clear()` for staging reuse; `Batch`
introspection (`is_embd`, `n_embd`, `n_outputs`, `sequences`).
`TokenId` untouched; `push` signature unchanged — P2 sampled ids feed
in exactly as before (existing test still green).

## 6. FFI changes

`forge-sys`: 3 const mods (`pooling_type`, `attention_type`,
`flash_attn_type`, values probed) + 15 functions: `llama_n_ctx`,
`llama_n_ctx_seq`, `llama_n_batch`, `llama_n_ubatch`,
`llama_n_seq_max`, `llama_pooling_type`, `llama_set_n_threads`,
`llama_set_causal_attn`, `llama_synchronize`,
`llama_get_embeddings_ith`, `llama_get_embeddings_seq`,
`llama_model_n_embd_inp`, `llama_model_n_embd_out`,
`llama_model_n_cls_out`, `llama_model_has_encoder`. Each carries a doc
comment with its verified contract (mutability, ownership,
nullability, failure mode). No new `repr(C)` structs (the existing
`llama_context_params` TerZGyr was re-verified sufficient — P4 only
writes already-declared fields).

Deliberately unbound (audited, documented): `llama_n_threads*`
getters (values cached — only our API mutates them),
`llama_set_embeddings`/`set_warmup` (construction covers the need;
niche/bench flows), `llama_batch_init/free/get_one` (manual C
allocation; ours is strictly safer), `llama_encode` (unreachable),
`llama_get_model`/`llama_get_memory` (Rc / P5), threadpool attach
(lifetime complexity), abort callbacks (P7), `has_decoder` (no
behavioral need). Failure semantics: getters are pure/infallible off
live handles; setters/synchronize are infallible; `_ith`/`_seq`
return NULL on missing outputs in Release (abort only in native
debug builds — same contract P1 relies on for logits).

## 7. Ownership/lifetime model

`Model → Context` unchanged (`Rc<ModelInner>` share; context Drop
frees native state before releasing the share; `!Send + !Sync`
preserved — no stronger native guarantee exists). `Batch` owns all
arrays (tokens *or* `f32` rows, positions, flat `c_int` seq store +
offsets, flags) and exposes no mutation, so the by-value `as_sys`
view (exactly one non-NULL input pointer) cannot dangle; seq runs
point into the fully-built store which never reallocates. Upstream
takes the batch by const reference and retains no pointers after
`llama_decode` returns (verified in source). `Logits`/`Embeddings`
are owned copies. No raw pointers in any public API; every `unsafe`
block keeps its SAFETY comment.

## 8. Context offload semantics

`offload_kqv` (default true): KV-cache buffer placement preference +
"pin KQV subgraph to CPU when false" + multi-GPU pipeline-parallel
gate. `op_offload` (default true): scheduler permission to move
host-weight ops onto capable devices. Verified from source that on a
CPU-only registry both are complete no-ops (single backend: the sched
loop body never executes; no GPU buffers exist to prefer). They are
therefore **advisory by native design, not explicit GPU requests**:
P4 exposes them as plain bools with native defaults, applies no
refusal rule (there is no fallback to refuse — setting either false
on CPU changes nothing), and proves it with a 4-combination open +
decode test. `kv_unified` (default false) selects the unified KV
buffer: required for coupled batches (native error otherwise),
defers seq validation to upstream's internal 256-bound, and reports
full length per sequence. No silent-degradation vector exists in any
of the three flags.

## 9. Batch memory/lifetime safety

Token/embedding exclusivity is structural (two `Vec`s, exactly one
populated ⇒ exactly one non-NULL pointer, satisfying
`GGML_ASSERT(token || embd)` with `n_tokens ≥ 1` guaranteed by the
non-empty build rule). Every token carries 1..=256 seq ids (empty
lists refused — upstream reads `seq_id[i][0]` unconditionally; the
256 cap is principled: ids must be distinct within 256 slots).
Positions/token/seq ids convert `u32 → i32` with explicit errors.
Embedding rows are exact-width slices copied into the owned flat
`Vec` (`n_embd ≥ 1` enforced, so the pointer always has backing
floats). Reuse: `clear()` reuses staging allocations; a built batch
decodes on any number of fresh contexts (tested). No caller-reachable
path passes NULL `pos`/`seq_id`/`logits` (native auto-generates those
when NULL — ForgeCore always provides them explicitly).

## 10. Error handling

One new domain: `Error::embeddings` ("embeddings error: …"), mirroring
`logits`, for missing embedding rows (flagged off, embeddings
disabled, pooling NONE, absent sequence). Everything else reuses
existing domains: `invalid` for misconfiguration (thread/batch/seq
violations at open, decode-time batch/limit/seq violations),
`context` for native creation refusal (`n_seq_max > 256`, OOM, …),
`decode` for native decode failures (unchanged codes), `batch` for
builder violations. Upfront-invalid (never reaching native):
`n_threads == 0`, `n_batch == 0`, unrepresentable threads,
`set_n_threads(0,·)`, empty batch, `n_tokens > n_batch`, non-causal
`n_tokens > n_ubatch`, `seq ≥ n_seq_max` (non-unified). Native NULL /
codes map as before. No panics on caller-controlled values; no silent
CPU fallback anywhere (the newly exposed GPU-adjacent flags are
advisory — §8).

## 11. Tests added

+12 unit (enum mappings ×3 incl. pooling roundtrip, extended defaults,
8 batch builder tests) and +31 integration in new
`tests/context_batch.rs`: effective sizing (7 incl. padding, scaling,
follow-rules), option validation (3), advisory flags (1),
multi-seq/coupled/reuse/refusals (7), embeddings input/output/pooling
(7), threads/causal/flash/sync (5), skip-gated GPU smoke (1).
Notable: the two abort-prevention tests decode 65/513-token batches
expecting explicit errors (would abort unguarded); coupled batches are
tested both succeeding (unified) and failing honestly (split).
Pre-existing suites untouched except the two `n_ctx` pins corrected to
effective values (256/256) with padding-rule comments. No test
assumes a GPU; no GPU behavior is faked.

## 12. CPU validation

Full suite on the CPU-only build: **193 green in debug and 193 in
release** (84 lib + 31 context_batch + 9 + 9 + 21 + 5 + 13 + 14 + 7;
was 150/150). Pre-fixture SKIP path green (31/31 + 84/84 without
`FORGE_TEST_MODEL`). `cargo fmt --all --check`, `cargo check
--workspace --all-targets`, `cargo clippy --workspace --all-targets
--all-features -- -D warnings`, and `RUSTDOCFLAGS="-D warnings" cargo
doc --no-deps` all clean (two doc links fixed, both mine; no
pre-existing warnings encountered this phase).

## 13. GPU validation

Not executable: no GPU in this environment (`supports_gpu_offload`
false), so `gpu_context_smoke` reports SKIP. It performs a real open
+ decode + report (sizes, causal, pooling) plus an offload-disabled
counterpart wherever hardware exists. Same environmental limitation
as P3; no claim is made beyond it.

## 14. RAMforge compatibility findings

Public RAMforge V4.1.0 (`6e67f83`) audited read-only; untouched.

- **A (provided, must keep working):** `ContextOptions::default()` +
  `n_ctx`/`n_threads` field assignment, `BatchBuilder::new/push/
  build`, `Context::decode/logits`, `Model::load` — the
  `forgecore.rs` adapter path (single-seq, last-token logits). All
  signatures and behaviors preserved; additive `#[non_exhaustive]`
  extension is compatible. `n_threads` keeps its lockstep meaning via
  the follow-default.
- **B (P4 fills):** batch sizing, micro-batch control, sequence
  capacity, batch-thread split + mid-life rethreading, multi-seq
  batches, embedding batches/outputs, pooling/attention/flash
  selection, causal toggle, synchronize, effective-size facts,
  advisory offload flags. RAMforge has no names for any of these yet
  (no `n_batch`/ubatch/embeddings concepts found) — purely new
  capability.
- **C (RAMforge-owned):** thread-count/batch-packing policy, memory
  budgets, eviction, residency, scheduling, orchestration — none
  implemented in P4.
- **D (legacy):** `kv_cache.rs` (290-line pure-Rust shadow KV) and the
  numerical `model_executor`/`inference` path — temporary until P5's
  native memory ops let RAMforge delegate; the adapter path already
  bypasses them.

P4 will eventually let RAMforge replace its manual batch/position
bookkeeping with native multi-seq batches, and P5's memory ops will
retire the shadow `KvCache`. No RAMforge migration performed (out of
scope).

## 15. Known limitations

1. **GPU execution unvalidated** (environmental) — §13.
2. **Causal resolution heuristic:** `Unspecified` + model-default
   causality is inferred as non-causal iff the model has an encoder
   (no upstream getter exists). Exact for all known models; an
   adversarial decoder GGUF marked non-causal would be misclassified
   — same exposure as the upstream CLI, documented in `open`.
3. **Decode can throw only on allocation failure** (`llama_decode`
   has no catch; the one throw site is an alloc-failure path) —
   same abort-class residual as P2's sampler-OOM analysis, documented.
4. **`n_seq_max > 256` and unified seq bound** defer to native errors
   (the 256 constant lives in an internal header, deliberately not
   duplicated).
5. **RANK pooling width mapping is implemented but decode-untested**
   (no classifier fixture); ordinary pooling is fully tested.
6. **Rust 1.99.0 unpinned** (pre-existing; rustup drift).
7. Bootstrap note: `rustup-init` without `--no-modify-path` wrote
   `/home/user/.profile`, and pip wrote `/home/user/.cache` — both
   removed; future bootstraps should pass `--no-modify-path` and
   `PIP_CACHE_DIR=/var/tmp`.

## 16. Explicit non-goals

Not implemented (per the P5/P6/P7 boundary): KV sequence ops
(`llama_memory_*`), state save/load/serialization, memory
estimation/accounting, RoPE/YaRN tuning, KV cache dtypes, recurrent
rollback, MTP contexts, SWA sizing, perf counters/timings, abort
callbacks, backend sampler chains, threadpool attach, warmup mode,
`ctx_other` sharing, defrag (deprecated upstream), generation loops,
scheduling/eviction/prefetch/placement policy of any kind. Each was
audited and its deferral documented in `ContextOptions` rustdoc or
this report.

## 17. Exact validation commands/results

With `scripts/env.sh` sourced and
`FORGE_TEST_MODEL=$FORGE_LLAMA_DIR/models/tiny-llama.gguf`,
`FORGE_TEST_MODEL_TOK=$FORGE_LLAMA_DIR/models/tiny-tok.gguf`:

- `carg
...[truncated 1198 chars]