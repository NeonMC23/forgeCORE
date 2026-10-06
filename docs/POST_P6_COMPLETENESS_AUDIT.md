# Post-P6 Low-Level Completeness Audit — forgeCORE

**Date:** 2026-10-06
**Auditor:** agentic session (read-only on RAMforge and upstream)
**Upstream pin (re-verified this session):** llama.cpp `v0.5.0` @
`7fe450e19305b828c199d602c23a8337aaa1f03b`, ggml `0.25.1`
(`git rev-parse` + `git describe` on a fresh clone; CMake reports
`ggml version: 0.25.1`)
**Baseline (re-verified this session):** 336/336 debug, 336/336
release, `fmt --check` / `clippy -D warnings` / `cargo doc -D
warnings` clean; fresh native Release build + regenerated
deterministic fixtures (`tiny-llama`, `--tok`, `--dsv4`)

> Scope discipline: this audit proposes **no implementation**. Every
> "missing" item below is classified; only items RAMforge cannot
> safely do without could justify a targeted continuation. None meet
> that bar (§10–§13).

---

## 1. Executive verdict

**forgeCORE low-level substrate is sufficient for RAMforge to resume.**

- The P6 report's claims reconcile with the current tree (§3): every
  guard, table, and count sampled matches the code; the 336 baseline
  re-verifies bit-for-bit on a clean rebuild.
- RAMforge (public V4.1.0, `6e67f83`) consumes forgeCORE rev
  `944bc3c` ("v2.0", P1-era) through a 5-type adapter surface. Every
  capability its architecture currently names as "waiting for
  ForgeCore" — model-offload binding, batch/state/memory control,
  tokenizer, sampler, device facts — **already exists** in the
  current tree. RAMforge's next step is a rev bump + adapter
  extension, not a forgeCORE phase.
- The remaining unexposed native surface divides cleanly into:
  future multi-device/async/out-of-core primitives (no consumer
  design exists yet — building them now would be speculative),
  upstream-limited facts (no C API exists at the pin — no forgeCORE
  work can ground them), RAMforge-owned policy, and genuine
  optionals. **Zero items classify as REQUIRED.**
- Safety posture holds: no panics/unwraps in non-test code, no
  `Send`/`Sync`, no public raw pointers, FFI widths re-verified
  against the pinned headers.

**Recommended next step:** stop forgeCORE development; RAMforge
resumes (§12).

---

## 2. Current forgeCORE capability inventory

Source of truth: the tree at `/home/user/forgeCORE/crates`
(269 `pub` items vs 119 at the consumed rev). Status legend:
**CPU✓** = validated on real CPU execution in tests;
**mapped** = bound + validated structurally, behavior unproven on
that path; **absent** = not exposed.

### A. Model (`model.rs`, 576 lines)

| Capability | State | Notes |
|---|---|---|
| GGUF load (`load`, `load_with_options`) | CPU✓ | Split-name `<name>-%05d-of-%05d.gguf` shards work natively through the same call (header-verified); custom split naming absent |
| `check_tensors` | CPU✓ | |
| mmap/mlock/load modes (`ModelLoadMode` 5 variants) | CPU✓ | All 5 modes load-tested; downgrade paths source-documented |
| GPU layers (`GpuLayers` Cpu/Count/All) | mapped | Validation + refusal fully CPU-tested; real offload never executed (no GPU) |
| Split modes (`SplitMode` None/Layer/Row/Tensor) | mapped | Same as above; native-default Layer preserved |
| `main_gpu` | mapped | Bounds-checked under None only (native reads it nowhere else — source-verified) |
| `tensor_split` | mapped | OOB-read guard (len ≥ devices); finiteness checks |
| `devices` explicit list | mapped | Verbatim incl. CPU devices (informed selection, not fallback); stale-index refused |
| No-silent-fallback refusal | CPU✓ | `unsupported` when GPU requested w/o GPU (probed native silent-CPU catalog) |
| Metadata: n_params, vocab_size, size_bytes, n_ctx_train, n_embd(+_inp/_out), n_layer, n_head(+_kv), n_cls_out, has_encoder, description | CPU✓ | 13 getters; negatives rejected; description regrow loop |
| Arch string / ftype / rope-freq / is_recurrent/hybrid/diffusion / has_decoder / meta enumeration | absent | Only `general.architecture` is read, privately, for the DSV4 bound (P5 §18) |
| `tensor_buft_overrides` placement | absent | (pattern → buft) load-time placement; needs explicit buft handles (§6-B) |
| `kv_overrides`, `vocab_only`, `load_mtp`, `no_alloc`, lazy_mode, progress cb, `use_extra_bufs` | absent | Deliberately out (P3/P4); no consumer |
| `set_log_quiet` | CPU✓ | Global log gate |

### B. Context (`context.rs`, 971 lines)

| Capability | State | Notes |
|---|---|---|
| `ContextOptions` 17 fields (n_ctx/threads/batch/ubatch/seq_max/threads_batch/outputs×2/offload_kqv/op_offload/kv_unified/embeddings/pooling/attention/flash_attn/type_k/type_v) | CPU✓ | All default to pre-P4 behavior; 0-batch refused (native abort), n_seq_max>256 → native error |
| Effective getters (n_ctx honest incl. 256-pad, n_ctx_seq, n_batch, n_ubatch, n_seq_max, n_threads×2, causal, pooling, n_vocab) | CPU✓ | P4 fixed the dishonest `n_ctx()` |
| `decode` (+ oversize/non-causal-ubatch/seq-bound validation) | CPU✓ | P4 fixed the latent oversize abort; codes mapped |
| `logits(i)`, `embeddings(i)`, `embeddings_seq(s)` (owned copies) | CPU✓ | NULL → Error (Release contract; debug-abort carries as documented) |
| `set_n_threads`, `set_causal_attn` (tracked), `synchronize` | CPU✓ | Infallible field updates / sched sync |
| State: `state_size/export/import`, per-seq size/export/import | CPU✓ | Exact-byte-consumption; empty-on-failure; 17/16-byte pins |
| `memory()` borrowed handle / None for memory-less archs | CPU✓ | None-path audited, fixture-untestable (no BERT GGUF) |
| `llama_encode`, MTP contexts, threadpool attach, abort callback, `set_embeddings/warmup`, `n_threads*` getters (cached instead) | absent | Each deliberately deferred with rationale (P4) |

### C. Batch (`batch.rs`)

| Capability | State | Notes |
|---|---|---|
| Token batches, `SeqId`, positions, logits flags | CPU✓ | Single-seq fast path unchanged since P1 |
| Multi-seq (`push_on_sequences`, ≤256 lists, non-empty) | CPU✓ | Coupled-batch unified requirement honestly surfaced |
| Embedding batches (`new_embeddings`, `push_embd`, exact-width rows) | CPU✓ | Homogeneous-mode enforcement; structural token/embd exclusivity |
| `clear()` staging reuse; introspection (n_tokens/is_embd/n_embd/n_outputs/sequences) | CPU✓ | Built batch decodes on N fresh contexts (tested) |
| Native `llama_batch_init/free/get_one` | absent | Deliberately unbound — owned builder is strictly safer |

### D. Tokenizer (`tokenizer.rs`, 732 lines)

| Capability | State | Notes |
|---|---|---|
| `encode`/`decode` + options, vocab text/score/attr, EOG/control, 7 specials, BOS/EOS flags, `VocabType` (all 7 native + Unknown) | CPU✓ | NONE refused; add_special per-type guard table; strict-UTF8 decode |
| `decode_bytes` (streaming) | absent | Optional; strict `decode` errs honestly until then |
| FIM tokens, `cls`, `get_add_sep`, `token_to_piece`, suppress-tokens | absent | Deliberately unbound (covered or behavior-free) |
| Vocab coverage | partial | Only SPM fixture-live; BPE/WPM/UGM/RWKV/PLAMO2 arms unit-pinned from source |

### E. Sampler (`sampler.rs`, 528 lines)

| Capability | State | Notes |
|---|---|---|
| greedy/dist/top-k/top-p/min-p/temp, chains, seed/resolved-seed, reset, accept, remove | CPU✓ | 6 of 20 native inits; seeded pins; `temp==0` modal greedy (matches RAMforge `Sampler`) |
| penalties, temp_ext, typical, top_n_sigma, XTC, DRY, mirostat×2, grammar×3, adaptive_p, infill, logit_bias | absent | **Optional**: repo-wide grep shows RAMforge uses none (its `Sampler` = temp/top-k/top-p only) |
| `llama_sampler_sample` (ctx-bound), `llama_set_sampler` + `get_sampled_*`, custom vtable samplers | absent | Intentionally out: bypass `Logits` / execution-path coupling / unsafe surface |
| `llama_decode_with_sampler` | n/a | **Commented out upstream at the pin** — upstream-limited, not a gap |

### F. KV / state / memory (`memory.rs`, 470 lines)

| Capability | State | Notes |
|---|---|---|
| clear/remove_range/copy_seq(full)/keep_seq/shift/scale/pos_min/pos_max/can_shift/seq_limit | CPU✓ | Dual seq-id bounds; DSV4 strict bound via private arch detect; unified/split probed |
| Whole + per-seq snapshots (`State`/`SeqState`, owned bytes) | CPU✓ | Roundtrip bit-exact logits; corruption/truncation pins |
| Exact serialization sizes | CPU✓ | Documented as *serialization, never residency* |
| Allocated/used/resident/scratch KV bytes | absent | **Upstream-limited**: `breakdown()` is C++-only; must not be invented |
| Per-layer K/V row read-back | absent | **Upstream-limited**: no native read-back API (state export is opaque) |
| File persistence, `_ext` flags incl. `ON_DEVICE`, partial copies, negative match-any, `swa_full`, `n_rs_seq` | absent | Deliberately out (unsafe or device-buffer-coupled) |

### G. Tensor / dtype / quant (`tensor.rs` 1247 lines, `dtype.rs`)

| Capability | State | Notes |
|---|---|---|
| 19 `DType`s + `from_ggml` rejection, `block_len/type_size/name`, quant geometry | CPU✓ | IQ/TQ/MXFP4/NVFP4/Q1_0/Q2_0/I64 unmapped (model path inherits them opaquely) |
| `empty/from_f32/from_i32/from_bytes`, shape/extent validation (≤4D, i64, overflow-checked) | CPU✓ | |
| upload/download F32/I32/bytes; `fill_f32` (gated F32+contig), `fill_bytes` (set-staged, memset avoided — unqueryable iface) | CPU✓ | F16/I8/F64/BF16 transfers via bytes only (optional) |
| `copy_into` (sync, same layout, views refused) | CPU✓ | Asymmetric mechanism proven: src aborts, dst survives (P6 F4) |
| view_1d–4d/reshape/transpose/permute/cont/cast + strict-subset pair table + Q8_1/Q8_K dequant ban | CPU✓ | 31-case abort battery: every guard load-bearing or documented |
| name/op introspection, contiguity/view predicates, nbytes (stride-aware) | CPU✓ | |
| Cross-backend async copy, 2D strided transfer, memset, view buffer init | absent | Later / optional / intentionally-out respectively |

### H. Runtime ops (`runtime.rs`, 877 lines) — all 16 inventoried

| Op | Impl | CPU-tested | Deterministic | Validation-covered | Abort-safe |
|---|---|---|---|---|---|
| add/sub/mul/div | ✓ | exact + div-by-zero arithmetic | ✓ | dtype/shape/backend | ✓ (battery A1) |
| matmul | ✓ | hand-computed + oracle; strided-B accepted | ✓ | inner-dim/geometry | ✓ (battery A2) |
| silu/sqr/sqrt/scale | ✓ | exact + NaN case | ✓ | F32+contig gates | ✓ |
| rms_norm/norm (+eps≥0) | ✓ | exact + oracle | ✓ | eps rules | ✓ (battery A7) |
| soft_max/soft_max_ext | ✓ | exact + oracle; mask/scale/bias | ✓ | geometry incl. mask | ✓ (battery A6) |
| rope (Neox/Normal + YaRN scalars) | ✓ | oracle-crossed | ✓ | even-ne0/n_dims/pos rules | ✓ (battery C1 CORRUPT) |
| get_rows (+dtype list, OOB scan) | ✓ | exact + OOB rejection | ✓ | index geometry + values | ✓ (battery A3/A18/A19/S3) |
| concat (all dims) | ✓ | exact all dims | ✓ | shape/dim | ✓ (battery A8/A11) |
| everything else (~350 ctors) | — | — | — | — | Intentionally unsupported (bind-on-demonstrated-need) |

GPU-tested: none (environmental — no device in any session). No op is "merely wrapped": each has exact-value, oracle-cross, or rejection tests.

### I. Backend / device (`backend.rs`, `device.rs` 527 lines)

| Capability | State | Notes |
|---|---|---|
| `enumerate_devices`, `DeviceInfo` (frozen shape per RAMforge structural use), `DeviceType` + Unknown arm | CPU✓ | Registry `OnceLock`; 1 CPU observed |
| `Backend::{open_cpu (fallback), open_device, set_cpu_threads, synchronize, device, name, is_cpu}` | CPU✓ | Stale-index + vanished-device errors; `!Send+!Sync` |
| `max_devices/supports_mmap/mlock/gpu_offload` | CPU✓ | Consistency-tested (`offload == any(Gpu\|Igpu)`) |
| `props()` (name/desc/mem/type/device_id/caps) + `supports_op(OpSpec 16 variants)` | CPU✓ | Full caps block incl. async/events/host_buffer/from_host_ptr/mmap |
| GPU `open_device` / `Graph::compute` on GPU | mapped | Code path backend-agnostic; never executed (no hardware) |
| by-name init, `init_best`, registry plugin load/unload, META-device use | absent | Optional / out (static build; RAMforge owns selection policy) |

### J. Buffers / allocations (`buffer.rs`)

| Capability | State | Notes |
|---|---|---|
| `BufferType` (default buft of a backend): name/alignment/max_size/is_host/tensor_alloc_size | CPU✓ | Genuine native facts; tensor_alloc_size documents the view-span caveat |
| `Buffer` owned alloc + size/align/max/host/name + Drop-free | CPU✓ | **Unwired**: no tensor-in-buffer attach, no transfer through it — placeholder shape |
| Per-device buft selection, host/pinned bufts, `buffer_from_host_ptr`, single-tensor alloc | absent | **Later bundle** (with tensor_buft_overrides) — no consumer design yet |
| Async copy, events/fences, mapped-memory expose | absent | **Later** (no async consumer yet) |

### K. Graph / execution (`graph.rs`)

| Capability | State | Notes |
|---|---|---|
| `Graph` multi-output, conservative cap bounds, single-backend gate, NodeInfo introspect + find-by-name | CPU✓ | Overflow abort made unreachable; re-add conservative |
| `Graph::compute` single-shot (backend-agnostic) | CPU✓ | GPU path mapped-but-unexecuted |
| `Plan` reusable (CPU-only, source-verified: CUDA/Vulkan plan iface NULL — re-verified this session) | CPU✓ | Deterministic recompute tested |
| Status-code mapping, empty/mixed-backend refusals | CPU✓ | |
| Multi-backend sched graphs, async compute, selective expansion, scratch-size query, dot-dump | absent | Later / optional (P7 diagnostics) |

### L. Resource / memory facts

| Fact | State | Notes |
|---|---|---|
| Tensor logical bytes (`nbytes`, stride-aware) / elements | **exact** | Native-reported |
| Per-tensor padded alloc size (`tensor_alloc_size`) | **exact** | Per-buft native query |
| Buffer size/align/max/host | **exact** | Owned-buffer getters |
| Model bytes (`size_bytes`), params | **exact** | Upstream-accounted |
| State serialization bytes (whole + per-seq) | **exact** | Never residency (documented) |
| Device free/total | **exact snapshot** | Staleness accepted by design; RAMforge treats 0/0 as Unknown client-side |
| Graph scratch/workspace bytes | **unavailable** | Internal to plan/sched reserve; no C getter at pin for the Plan path |
| KV allocated/resident/used bytes | **unavailable** | **Upstream-limited** (C++-only `breakdown()`) |
| Host RAM / RSS / page cache | n/a | RAMforge-owned via `/proc` (already implemented there) |
| Mapped bytes / residency state | **unknown** | No native per-tensor residency query at pin |

Nothing is estimated; unavailable facts are named, never invented.

---

## 3. P0–P6 coverage matrix

| Phase | Promised (prior report) | Delivered in tree | Tests | Reconciled |
|---|---|---|---|---|
| P0 audit | arch + native audit, no code | docs only | 81 baseline | ✓ reports match tree |
| P1 tokenizer | safe native tokenizer | `tokenizer.rs` + 19 FFI | 103 (+22) | ✓ guards match §7 table |
| P2 sampler | 6 primitives + chains | `sampler.rs` + 15 FFI | 125 (+22) | ✓ `apply`-not-`sample` as designed |
| P3 offload | 6 ModelOptions fields + refusal | `model.rs`/`device.rs` + 4 FFI + `devices` indirection fix | 150 (+25) | ✓ fix present; `DeviceInfo` frozen |
| P4 ctx/batch | 13 ctx fields, multi-seq, embeddings, 2 defect fixes | `context.rs`/`batch.rs` + 15 FFI | 193 (+43) | ✓ abort guards + honest `n_ctx` present |
| P5 KV/state | Memory + snapshots + facts + DSV4 pass | `memory.rs` + 19 FFI + arch detect | 239 (+46) | ✓ dual bounds + `deepseek4` detect present |
| P6 ggml | tensor/runtime/quant/backend-graph + proof + battery | `tensor.rs`/`runtime.rs`/`buffer.rs`/`graph.rs`/`device.rs` props | 336 (+97) | ✓ sampled: rope guards, cast table, copy guard, 31-case battery log all match code |

Reconciliation notes (report-vs-code deltas found — all benign):
- P6 "336/336" re-verified exactly on a clean rebuild (17 suites).
- `n_ctx()` behavior change (requested → effective, P4 defect 2) is the
  one behavior delta a RAMforge rev bump must absorb; it is the *honest*
  value RAMforge's length checks want.
- P0 §20 items 4/5 still open: `reference::checkpoint` has zero
  consumers (verified); `QWEN25_DIAGNOSTIC_*` constants remain
  placeholders. Documentation-only; §9.
- `llama_decode_with_sampler` (listed in P2's "not bound" table as a
  design decision) is actually **commented out upstream** — stronger
  than reported: not bindable at the pin at all.

---

## 4. Pinned native capability comparison

Bound: 90 llama + 94 ggml/backend/alloc symbols (~184 of ~750
declarations ≈ 25% — deliberately, per bind-on-need).

| Unbound capability | Classification | Rationale |
|---|---|---|
| sched multi-backend graphs (`sched_*` ~20 fns) | **REQUIRED LATER** | The multi-device graph story; no RAMforge multi-device-graph design exists |
| Events (`event_*`) + async compute/copy/set/get | **REQUIRED LATER** | Async overlap/completion; no async consumer exists |
| Explicit bufts (`dev_buffer_type`, `host_buffer_type`, `buffer_from_host_ptr`, `alloc_buffer`, `tensor_alloc`) | **REQUIRED LATER** | Per-domain placement + staging; bundle with overrides |
| `tensor_buft_overrides` (+ `max_tensor_buft_overrides`) | **REQUIRED LATER** | Load-time per-tensor placement; needs the buft bundle |
| Graph scratch-size query (via sched/gallocr) | **REQUIRED LATER** | Plumb when raw-graph execution gets a consumer |
| `ON_DEVICE` state `_ext` variants | **REQUIRED LATER** | Needs device buffers; host bytes suffice today |
| Abort callback, perf counters/timings, `print_system_info`, `version`, `log_get`, graph dot-dump | **REQUIRED LATER (P7 observability)** | Diagnostics/cancellation; no consumer today |
| penalties/temp_ext/typical/top_n_sigma/XTC/DRY/mirostat×2/grammar×3/adaptive_p/infill/logit_bias, `set_sampler`+`get_sampled_*` | **OPTIONAL** | Zero RAMforge use (grep-verified); add on demonstrated need |
| `decode_bytes`, FIM/cls/suppress tokens, BPE-live fixture | **OPTIONAL** | Streaming/convenience; honest errors until then |
| F16/I8/F64/BF16 native transfer, 2D strided transfer, `validate_row_data`, `rope_set_offset`, `mul_mat_id/add_id`, `set_zero`/`fill` | **OPTIONAL** | Convenience or libllama-internal; bytes path covers transfers |
| `llama_model_load_from_splits`, model quantize/save, chat templates, LoRA adapters, optimizer, threadpool attach, NUMA, `set_embeddings/warmup`, `n_threads*` getters, `get_model`, by-name `init_best`, RPC | **OPTIONAL / OUT** | Custom-split naming, authoring, policy-level, or deliberately-deferred (each with a prior rationale) |
| `format_name` (used internally), `ggml_nelements`… (now used) | — | P0's bound-but-unused list is down to `ggml_log_set` + `dev_by_type` (P7 / trivial) |
| KV allocated/resident (`breakdown()` C++-only), per-layer K/V read-back, `decode_with_sampler` (commented out), device vendor field (no native field) | **UPSTREAM-LIMITED** | No C API at the pin; no forgeCORE work can ground these |
| Deprecated `llama_token_*`/`llama_n_vocab`/`llama_free_model` aliases, `llama_batch_init/free/get_one`, `llama_encode` (unreachable), training/backward/grad APIs, inplace-op variants, custom vtable samplers, `tensor_memset` (unqueryable abort), `view_init` (NULL-buffer invariant) | **INTENTIONALLY OUT** | Safer equivalent exists, unreachable, or unsafe surface |
| All 61 `gguf_*` (zero parsing) | **RAMforge RESPONSIBILITY** | RAMforge owns its GGUF parser; forgeCORE must not duplicate it |
| Device/placement/scheduling/retention/migration policy of any kind | **RAMforge RESPONSIBILITY** | Architecture boundary (§5 of task) |

---

## 5. RAMforge requirement cross-audit

**Consumed revision:** `944bc3ca1ed7e48e2ad897107d9dcda5dd5eedc9`
("v2.0", P1-era) — pinned identically in local worktree and public
V4.1.0 (`6e67f83`).
**Consumed surface (exhaustive):** `Model::{load_with_options,
n_params, vocab_size, size_bytes, n_ctx_train, n_embd, n_layer,
n_head, n_head_kv, description}`, `ModelOptions{check_tensors}`,
`Context::{open, decode, logits, n_ctx, n_vocab}`,
`ContextOptions{n_ctx, n_threads}`, `BatchBuilder::{new, push,
build}`, `enumerate_devices/DeviceInfo/DeviceType`,
`model::set_log_quiet` (tests).
**Compatibility:** zero breaking removals on this surface (the only
removed names are pre-pivot `kv.rs`/`attention.rs` oracle items
RAMforge never touched); `DeviceInfo` shape frozen; `#[non_exhaustive]`
extensions are additive; the single behavior delta is the honest
`n_ctx()` (wanted). A rev bump is source-compatible.

| RAMforge need (from `forgecore.rs`, `execution.rs`, planner, kv shadow) | forgeCORE today | Owner/Action |
|---|---|---|
| Static CPU execution (adapter path) | ✓ provided | — (working at pinned rev) |
| Static GPU execution (`GpuLayerOffload` gate: "model/context device binding") | ✓ provided (P3), needs rev bump | **RAMforge**: bump + flip `selectable`/`static_model_offload` gates + extend adapter |
| Batch/state/memory control for prefix mgmt, save/restore, native accounting bytes | ✓ provided (P4/P5), unused by adapter | **RAMforge**: adopt at its pace |
| Tokenizer/sampler migration off inspection parser + `thread_rng` | ✓ provided (P1/P2) | **RAMforge**: migration decision |
| Exact native state bytes replacing f32-shadow math | ✓ provided | **RAMforge**: adopt |
| Retire `kv_cache.rs` shadow fully | Blocked: needs per-layer K/V read-back (**upstream-limited**) + allocated/resident bytes (**upstream-limited**) | No action possible; shadow stays for the legacy executor only (adapter path already bypasses it) |
| Residency/eviction/prefetch/placement/scheduling/budgets/orchestration/diagnostics | Policy — correctly RAMforge-owned | **RAMforge**: build on existing primitives |
| Vendor field per device | **Upstream-limited** (no native field) | None; `device_id` (P6) is the closest stable key |
| Out-of-core weight streaming through the model path | **Impossible at pin** (libllama loads all weights eagerly; `lazy_mode` is arch-specific) | Future architecture decision, not a primitive gap |
| Async overlap / multi-device graphs / fine placement | Later bundles (§4) | Build only when a RAMforge design consumes them |

No finding moves policy into forgeCORE. No RAMforge change was made
or is proposed here.

---

## 6. Future out-of-core capability analysis

Modeled flow: storage → host staging → device buffer → graph
execution → synchronization → release/retain → eviction/reuse.

| Step | Primitive needed | forgeCORE state |
|---|---|---|
| Storage → host staging | Owned host allocations + GGUF ranges | RAMforge-owned (has parser + `MemoryBudget`); `Buffer` shape exists but unwired |
| Host staging → device buffer | Per-domain buft alloc + sync/async copy | Sync `copy_into` ✓ (same-layout); **async + events missing (Later)**; per-domain buft selection missing (Later) |
| Device buffer residency | Explicit tensor-in-buffer placement + overrides | Missing (Later bundle); default-buft only today |
| Graph execution | Single/multi-backend graphs | Single-backend ✓; sched multi-backend missing (Later) |
| Synchronization / completion | sync + events | `synchronize` (backend + context) ✓; events missing (Later) |
| Release/retain + lifetime | Owned Droponomian buffers/tensors/graphs | ✓ (`Buffer`/`Tensor`/`Graph`/`Plan` all Drop-free; History DAG retains) |
| Eviction/reuse/memory-size queries | Exact sizes for policy input | Partial: tensor/model/state/buffer/device ✓; graph scratch + KV resident **unavailable** (Later / upstream-limited) |
| Capability queries for planning | Device/buft/op facts | ✓ (`props`, caps, `supports_op`, `tensor_alloc_size`, `max_size`) |

**Conclusion:** the *synchronous, single-device, default-placement*
out-of-core loop is expressible today; the *async, multi-device,
fine-placement* loop awaits the Later bundles. Crucially, model-path
weight streaming is impossible at the pin regardless of bindings, so
there is no primitive whose absence blocks a currently-designed
RAMforge flow. No implementation now.

---

## 7. GPU / multi-device gap analysis

| Bucket | Items |
|---|---|
| **A. Exposed, CPU-validated, GPU-unvalidated** | `ModelOptions` offload fields + refusal rule; `open_device`; `Graph::compute` on any backend; per-tensor `supports_op`; `tensor_alloc_size`; `props`/caps; offload/context/KV skip-gated smoke tests (3) |
| **B. Genuinely missing** | sched graphs; events + async compute/copy; explicit buft selection + `tensor_buft_overrides`; per-device buffer alloc; `init_best`/by-name (trivial) |
| **C. Belongs to RAMforge** | Selection syntax/policy (`DeviceSelection` exists), placement strategy, `GpuPreference`, split planning, orchestration, accounting domains |
| **D. Unvalidatable here** | All real GPU execution (no device in any session to date); iGPU/RPC/ACCEL paths; TENSOR-split arch support |
| **E. Upstream-limited at pin** | KV allocated/resident bytes; per-layer read-back; fused decode+sample (commented out) |

GPU support is **not** labeled complete: bucket A is honest
plumbing with validation gates, bucket B is future work, bucket D
is environmental. What *is* complete is the static-offload
*configuration* surface RAMforge's `GpuLayerOffload` strategy waits
for — the remaining GPU risk is execution validation on hardware,
not missing knobs.

---

## 8. Safety gap analysis

Sweep of all non-test `forge-core` + `forge-sys` code (P5/P6
standard: no abort/crash reachable through public API):

| Check | Result |
|---|---|
| `panic!` in non-test code | None |
| `unwrap()`/`expect()` (30 sites) | **All test-only** (every site below its file's `mod tests`; `backend`/`buffer`/`runtime` have zero) |
| `unreachable!` (1: `model.rs:337`) | Intentional safe boundary (logically-impossible `GpuLayers::Cpu` arm) |
| `as` casts on untrusted values | None found: all are widenings (`u32→usize`), `ret<0`-guarded (`model.rs:411`), `row<0`-guarded (`runtime.rs:769`), const bit patterns, or the `blck_size as usize` over a natively-positive table constant (debug-asserted) |
| `Send`/`Sync` impls | None — everything `!Send+!Sync` via raw ptr/`Rc`/phantom (checked: no `unsafe impl`) |
| Public raw pointers / sys-type leaks | None — zero `pub fn` mentions `forge_sys` or `*mut`/`*const`; all `raw` fields private; `raw()`/`inner()`/`graph_size()` are `pub(crate)` |
| FFI layout | 9/9 structs `repr(C)`; widths re-verified (`u32/u64` getters, `c_longlong` blck, `usize` sizes); 13-arg `rope_ext` order matches header |
| Native asserts reachable | None known: 31-case battery + P5 probes cover every guard; `graph/compute/plan` bounds-check both ends; batch/decode/state pre-validate all probed abort sites |
| OOM-throw across C ABI (decode/sampler ctors) | Documented residual (same class as Rust alloc-OOM), carried since P2 — accepted, not a blocker |
| Malformed-vocab throw (P1 §16.1) | Documented residual, well-formedness boundary — accepted |
| Debug-vs-Release NULL/abort contracts (`_ith` getters) | Relied upon per Release contract; documented |

**Findings classification:** zero blockers; zero corrective
hardening; residuals are intentional safe boundaries (documented in
rustdoc + prior reports); unwraps are test-only; the two P0 §20
leftovers are documentation-only (§9).

---

## 9. API quality review

Reviewed as one library (269 `pub` items, 15 modules):

- **Naming:** consistent (`open/load/new/build/compute/plan`,
  `n_*` getters, `is_*/has_*` predicates). One wart:
  `DeviceInfo::props` spells its return `Result<DeviceProps,
  crate::error::Error>` instead of the crate `Result` alias —
  cosmetic, optional.
- **Ownership:** uniform and sound — owned handles + borrows with
  explicit lifetimes (`Tokenizer<'model>`, `Graph<'t>`,
  `Plan<'a>`, `BufferType<'a>`, `Memory<'a>` with `&mut` borrow);
  `Rc`-shared internals never leak; `Drop` frees exactly once.
- **Borrowing:** `&mut` correctly marks native mutation (`decode`,
  `logits`, `memory`, seq ops); pure getters take `&self`.
- **Errors:** single string type + 13 domain prefixes, pinned by
  test. RAMforge demonstrably works around the lack of variants
  with its own adapter enum — no pressure to churn the type.
- **Boundaries:** `reference/` oracles correctly test-only… except
  `reference::checkpoint` (zero consumers) and `QWEN25_DIAGNOSTIC_*`
  placeholders are still **dead public API** (P0 §20.4/5, still
  open). Suggest `pub(crate)`-ing or removing in passing — explicitly
  non-blocking, documentation-only.
- **Surface size:** no duplication found (one discovery path, one
  batch builder, one graph path); unbound upstream surface is
  justified per item (§4), not accreted.
- **Upstream leakage:** none — no ggml/llama types, no error codes,
  no config structs cross the boundary; enums mirror with `Unknown`
  arms where upstream can grow.
- **RAMforge fit:** the adapter's consumed surface is source-stable
  across 5 phases of additive growth — the `#[non_exhaustive]` +
  frozen-`DeviceInfo` discipline works. The one future friction is
  `Buffer`'s unwired shape (allocate/query/free only): either wire
  it in the Later placement bundle or document it as a facts-only
  handle — no action now.

No redesign recommended. No change proposed for style.

---

## 10. Complete gap table

| # | Capability | forgeCORE state | Needed? | Owner | Action |
|---|---|---|---|---|---|
| 1 | Static CPU execution | ✓ complete | — | — | none |
| 2 | Static GPU offload config | ✓ complete (unvalidated on HW) | — | — | none (validate on GPU HW when available) |
| 3 | Multi-backend (sched) graphs | absent | Later | forgeCORE (future) | none now — no consumer design |
| 4 | Events + async compute/copy | absent | Later | forgeCORE (future) | none now |
| 5 | Explicit buft selection + `buffer_from_host_ptr` | absent | Later | forgeCORE (future) | none now; bundle with #6 |
| 6 | `tensor_buft_overrides` placement | absent | Later | forgeCORE (future) | none now; needs #5 |
| 7 | ON_DEVICE state `_ext` | absent | Later | forgeCORE (future) | none now; host bytes suffice |
| 8 | Graph scratch-size query | absent | Later | forgeCORE (future) | none now; needs #3 surface |
| 9 | Abort callback / perf / sysinfo / version / dot-dump | absent | Later (P7) | forgeCORE (future) | none now |
| 10 | KV allocated/resident bytes | impossible (C++-only) | — | upstream | none possible; RAMforge keeps shadow math |
| 11 | Per-layer K/V read-back | impossible (no API) | — | upstream / RAMforge migration | none possible; adapter path bypasses shadow |
| 12 | Fused decode+sample | commented out upstream | — | upstream | none possible |
| 13 | Device vendor field | no native field | — | upstream | none; use `device_id` |
| 14 | Extra samplers (penalties/mirostat/grammar/…) | absent | Optional | forgeCORE iff RAMforge asks | none (zero use) |
| 15 | `decode_bytes`, FIM/cls, BPE-live cover | absent/partial | Optional | forgeCORE iff asked | none |
| 16 | F16/I8/F64/BF16 native transfer, 2D strided xfer | via-bytes / absent | Optional | forgeCORE iff asked | none |
| 17 | IQ/TQ/MXFP4/NVFP4/Q1_0/Q2_0/I64 dtypes | unmapped (model path inherits) | Optional | forgeCORE iff asked | none |
| 18 | Arch/ftype/rope-freq/model predicates | mostly absent | Optional | forgeCORE iff asked | none (RAMforge parses GGUF itself) |
| 19 | Custom-split loading, quantize/save, chat templates, LoRA, threadpool, NUMA | absent | Out | — | none (authoring/policy/deferred) |
| 20 | GGUF parsing/inspection | absent (zero) | — | **RAMforge** | none (RAMforge owns its parser) |
| 21 | Placement/selection/scheduling/budget/eviction/prefetch policy | absent (none!) | — | **RAMforge** | none (correctly unmirrored) |
| 22 | Out-of-core weight streaming (model path) | impossible at pin | — | future arch decision | none (eager libllama loads) |
| 23 | Typed `Error` variants | string+prefix | Optional | forgeCORE iff RAMforge needs matching | none (adapter works around) |
| 24 | `reference::checkpoint`, QWEN25 placeholders | dead public API | — | forgeCORE cleanup | optional `pub(crate)`/removal, non-blocking |
| 25 | `Buffer` unwired shape | facts-only | — | forgeCORE (with #5) | none now; wire in Later bundle or document |

**Required rows: none.**

---

## 11. Required vs optional vs RAMforge-owned capabilities

- **REQUIRED (blocking RAMforge):** *none identified.* Every
  RAMforge-named need exists in the tree behind a rev bump.
- **REQUIRED LATER (future forgeCORE bundles, build only on
  consumer design):** multi-backend sched graphs; events + async;
  explicit bufts + `tensor_buft_overrides`; ON_DEVICE state;
  graph-scratch query; P7 observability (abort callback, perf,
  sysinfo, logging).
- **OPTIONAL (build iff RAMforge demonstrates need):** extra
  samplers; tokenizer streaming/expansion; wider transfers;
  exotic dtypes; arch metadata; typed errors.
- **RAMforge-OWNED (must never move):** GGUF parsing; all
  placement/selection/scheduling/budget/eviction/prefetch/
  residency/orchestration policy; generation loops; diagnostics
  presentation; the shadow-KV retirement migration decision.
- **UPSTREAM-LIMITED (no action possible at pin):** KV
  allocated/resident bytes; per-layer K/V read-back; fused
  decode+sample; device vendor.
- **INTENTIONALLY OUT (do not revisit without new evidence):**
  deprecated aliases; unsafe/unqueryable surface (`memset`,
  custom vtable samplers, `view_init` reliance); training/backward
  APIs; inplace variants; file persistence; partial copies;
  negative match-any.

---

## 12. Recommended next step

**No forgeCORE phase.** Concretely, for RAMforge's return:

1. Bump `forge-core` from `944bc3c` to the current forgeCORE
   revision (source-compatible on the adapter surface; absorb the
   honest `n_ctx()` — it fixes length accounting, not breaks it).
2. Extend the adapter in dependency order: offload fields
   (`GpuLayers`/devices/split → flip `selectable` +
   `static_model_offload` + retire `UnsupportedGpuExecution`),
   then tokenizer/sampler migration, then batch/state/memory
   adoption for prefix mgmt and snapshots.
3. Keep the shadow `KvCache` for the legacy numerical executor
   only; its full retirement awaits upstream C APIs that do not
   exist at the pin — track, do not work around unsafely.
4. GPU execution validation on real hardware is the highest-value
   future validation upgrade (forgeCORE's skip-gated smokes +
   RAMforge's gates are already in place).

forgeCORE should accept only: bug fixes, rev-bump fallout, and the
§11 Later bundles *when a RAMforge design consumes them*.

---

## 13. Explicit decision

**forgeCORE is sufficiently complete and RAMforge can resume.**

*"Can RAMforge now rely on forgeCORE as its complete safe
low-level execution substrate and focus exclusively on resource
attribution, planning, residency, movement orchestration,
scheduling, and out-of-core policy?"*

**Yes** — for every flow RAMforge has designed (static CPU/GPU
model execution, batch/state/memory management, tokenization,
sampling, device discovery and facts, synchronous graph execution).
The async/multi-device/fine-placement future is correctly
unbuilt: no design consumes it, and the pin forbids the one flow
(model-path weight streaming) that could have forced it early.

No targeted forgeCORE continuation is required. No version or
milestone is incremented by this audit. RAMforge was not modified.
RAMforge Phase 4 is not started here — it awaits the owner's
explicit directive.
