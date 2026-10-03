# forgeCORE Phase 0 Engineering Report

**Date:** 2026-10-03
**Scope:** architecture + native API audit only. No implementation, no refactor, no migration.
**Method:** every claim below is either directly observed in the audited tree / pinned
headers / executed commands, or explicitly labeled as a recommendation. No GPU was
present in the audit environment, so every accelerator statement is qualified as such.

Capability labels used throughout: `IMPLEMENTED`, `PARTIALLY_IMPLEMENTED`,
`AVAILABLE_NATIVELY`, `NOT_EXPOSED`, `NOT_SUPPORTED`, `UNKNOWN`.
(§16 uses the multi-device label set required by the audit brief.)

---

## 1. Executive summary

forgeCORE is a small (389 KB, ~5,000 lines incl. docs/tests), zero-dependency Rust
workspace with two crates: `forge-sys` (hand-written FFI, 51 functions) and `forge-core`
(safe wrappers). It dynamically links a pinned upstream llama.cpp build
(`v0.5.0` @ `7fe450e19305b828c199d602c23a8337aaa1f03b`, SHA re-verified this session)
that lives outside the repo. Nothing is vendored, nothing is generated, there is no
CI, no git history, and no platform-specific code.

What actually works today (all re-validated this session: fmt/check/81 tests/
strict clippy/release green, plus an independent C device probe):

- CPU device discovery, CPU backend execution of two F32 ggml ops (`add`, `matmul`).
- CPU model loading from `.gguf` with 9 metadata getters.
- Minimal CPU inference: `Model → Context → Batch → decode → Logits`.
- Explicit validation errors; no raw pointers in any public API; all 37 `unsafe`
  sites carry `// SAFETY:` comments (mechanically verified).

What is natively available but not exposed: the entire tokenizer/vocabulary API,
the sampler chain, KV-cache sequence ops, context state save/load, GPU offload
parameters, multi-GPU split controls, device capabilities, buffer-type control,
multi-backend scheduling, ~98% of the ggml op surface, and all CPU feature queries.

What is genuinely absent: GPU execution cannot happen through the model path
(`n_gpu_layers = 0` is hardcoded); sampling, detokenized generation, KV control,
and memory accounting beyond device free/total + model bytes do not exist.

High-level verdicts:

| Capability | Verdict |
|---|---|
| CPU tensor execution (F32 add/matmul) | `IMPLEMENTED` |
| CPU model load + metadata | `IMPLEMENTED` |
| CPU decode + logits | `IMPLEMENTED` |
| Device discovery (name/kind/memory) | `IMPLEMENTED` |
| GPU tensor execution via registry | `PARTIALLY_IMPLEMENTED` (backend-agnostic code path exists; no accelerator verified here) |
| GPU model offload | `NOT_SUPPORTED` (hardcoded CPU; native supports it) |
| Text tokenization / detokenization | `NOT_EXPOSED` (`AVAILABLE_NATIVELY`) |
| Sampling | `NOT_EXPOSED` (`AVAILABLE_NATIVELY`) |
| KV-cache control / state save-load | `NOT_EXPOSED` (`AVAILABLE_NATIVELY`) |
| Multi-GPU / split execution | `NOT_EXPOSED` (`AVAILABLE_NATIVELY`) |
| Memory accounting (context/KV/alloc) | `PARTIALLY_IMPLEMENTED` (device + model-bytes only) |
| Per-backend quantization limits | `UNKNOWN` (CPU-only environment) |

Small audit fixes applied (§18-allowed): two stale PIVOT claims corrected (FFI
surface size; nonexistent `Error::native`), one README import line completed, the
fixture recipe made reproducible (NATIVE.md), `.gitignore` extended per the
cleanliness rule. No code touched.

---

## 2. Repository structure

Observed tree (no `.git`, no CI config, no `.cargo`, no `target/`):

```text
forgeCORE/                          389 KB total
  .gitignore                        ignore rules (extended this phase, §19)
  Cargo.toml                        workspace: forge-core + forge-sys, resolver 2
  Cargo.lock                        2 packages, 0 external deps
  LICENSE                           MIT (ForgeCore-owned material)
  README.md                         overview, quickstart, API glance, status
  crates/forge-sys/
    Cargo.toml                      no deps, no build-deps
    build.rs                        links prebuilt .so files; panics if absent
    src/lib.rs                      404 lines: all FFI
  crates/forge-core/
    Cargo.toml                      depends only on forge-sys (path)
    build.rs                        re-emits rpath for own targets
    src/                            10 modules (lib, error, device, backend,
                                    dtype, tensor, runtime, model, context, batch)
    src/reference/                  5 frozen oracle modules (never executed)
    tests/                          ggml_smoke, cpu_decode, quant_contract
  scripts/                          env.sh, setup-native.sh, make-tiny-gguf.py
  docs/                             PIVOT, NATIVE, LICENSING, TOOLCHAIN, AUDIT,
                                    CONVENTIONS (+ handoff/, 1510 lines, frozen)
```

Line counts (code + tests, excl. docs): `forge-sys` 404; `forge-core` src ~1,700;
`reference/` ~1,600 (oracles); integration tests ~500.

Answers to the brief's identification questions:

- Rust workspace (2 members), not a standalone crate.
- Wrapper around an **external** llama.cpp checkout (never vendored, never in-repo).
- **Manually maintained FFI** (51 `extern "C"` declarations); no bindgen, no
  generated bindings anywhere (verified: no `bindgen`, `cxx`, `bindgen`-output
  markers in the tree).
- No examples/, no benches/, no feature flags, no platform-specific code
  (`#[cfg]` appears only for `#[cfg(test)]`).
- Build requirements: Rust toolchain + prebuilt native libs (see §4, §19).

---

## 3. Current crate/workspace architecture

**`forge-sys`** — the only crate that talks to C. Contents, all verified by reading:

- 9 opaque handle types (`ggml_context/tensor/cgraph/backend/backend_dev/
  backend_buffer`, `llama_model/vocab/context`) + 2 pointer aliases.
- 4 `#[repr(C)]` structs passed by value (`ggml_init_params` 24 B,
  `llama_model_params` 80 B, `llama_batch` 56 B, `llama_context_params` 160 B),
  each guarded by gcc-audited size+offset layout tests (4 tests).
- 3 constant modules (`ggml_type`, `dev_type`, `status`) + 3 callback typedefs.
- 51 `unsafe extern "C"` declarations: 13 `ggml.h`, 16 `ggml-backend.h`,
  1 `ggml-alloc.h`, 2 `ggml-cpu.h`, 19 `llama.h`.
- 4 bound-but-unused declarations (verified by use-site grep): `ggml_nelements`,
  `ggml_log_set`, `ggml_backend_dev_by_type`, `ggml_backend_synchronize`.
  Candidates for first use in Phases P4–P6, not for removal.
- `build.rs` links `ggml, ggml-base, ggml-cpu, llama` from
  `$FORGE_LLAMA_DIR/build/bin` (default `/var/tmp/forge-native`), emits rpath,
  compiles/downloads nothing.

**`forge-core`** — safe API, depends only on `forge-sys`:

| Module | Public surface | Role |
|---|---|---|
| `device` | `enumerate_devices`, `DeviceInfo`, `DeviceType` | registry snapshot |
| `backend` | `Backend::{open_cpu, open_device, set_cpu_threads, …}` | owned `ggml_backend_t` (Rc-shared) |
| `dtype` | `DType` (19 variants) | `ggml_type` mirror, explicit unknown-id errors |
| `tensor` | `Tensor::{from_f32, empty, to_vec_f32, …}` | owned tensor + backend buffer |
| `runtime` | `add`, `matmul` | one-node F32 graphs, same-backend checked |
| `model` | `Model::{load, load_with_options, …}`, `ModelOptions` | owned `llama_model` (Rc-shared) + 9 getters |
| `context` | `Context::{open, decode, logits}`, `ContextOptions`, `Logits` | owned `llama_context` + model share |
| `batch` | `BatchBuilder`, `Batch`, `TokenId` | owned token/pos/seq/logits arrays, single seq 0 |
| `error` | `Error` (8 prefixed ctors), `Result` | string error with domain prefixes |
| `reference/*` | frozen oracles | test-only cross-validation (see below) |

Invariants (verified, not assumed):

- No raw pointers in any public API (all `pub fn` signatures take/return owned
  Rust values; raw pointers are `pub(crate)` at most).
- `Backend`, `Tensor`, `Model`, `Context`, `Batch` are `!Send + !Sync` (raw
  pointers + `Rc`); all free natively on drop; `Context` holds an `Rc` model
  share so models outlive contexts.
- 37 `unsafe` operation sites in `forge-core`, each with a `// SAFETY:` comment
  in the preceding 6 lines (mechanically verified; the only `unsafe` lines
  without one are the `log_sink` declaration and its cast, not operation sites).
- Every nullable C return observed is NULL-checked; C strings are copied out
  immediately (`to_string_lossy` / owned `String`).
- `reference/` is consumed only by integration tests (`ggml_smoke` → `ops`;
  `quant_contract` → `quant`/`shape`); zero use from non-test `src`. No
  execution path touches it. **`reference::checkpoint` has no consumers
  anywhere** (only its own `mod` declaration) — obsolete-candidate; see §20.
- RAMforge separation: `grep` finds RAMforge mentioned only in doc comments
  (5 spots, all non-functional); zero Cargo/build/runtime references. The
  AUDIT.md separation claim still holds.

---

## 4. llama.cpp / ggml provenance

| Question | Finding (observed) |
|---|---|
| Source location | `https://github.com/ggml-org/llama.cpp`, cloned by `scripts/setup-native.sh` (`--depth 1 --branch v0.5.0`) |
| Pinned version | tag `v0.5.0`, commit `7fe450e19305b828c199d602c23a8337aaa1f03b` — **re-verified this session** via `git rev-parse HEAD` + `git describe --tags` on the fresh clone |
| How obtained | source clone + local CMake build; no binary downloads, no system packages, no submodules |
| Vendored? | No. `src/` + `build/` live under `$FORGE_LLAMA_DIR` (default `/var/tmp/forge-native`), outside the repo |
| System libraries used? | No. Only toolchain `libc`/`libstdc++`-level linking; ggml/llama come from the pinned build |
| Native compilation config | `BUILD_SHARED_LIBS=ON`, tests/examples/tools/server/app OFF, Release, default backend flags (= CPU-only here) |
| Produced libraries | `libggml(.so.0.25.1)`, `libggml-base`, `libggml-cpu`, `libllama(.so.0.5.0)` (+ `libllama-common`, built but **not** linked by forgeCORE) |
| License | MIT, `Copyright (c) 2023-2026 The ggml authors` (root LICENSE; jsonhpp notice also present). Consistent with `docs/LICENSING.md`; that doc needed no changes |
| C API surface size | `llama.h`: 246 `LLAMA_API` declarations · `ggml.h`: 379 `GGML_API` · `gguf.h`: 61 · `ggml-backend.h`: ~90 · `ggml-alloc.h` / `ggml-cpu.h`: smaller sets |
| Wrapped by Rust | 51 functions (~7% of `llama.h`, ~3% of `ggml.h`) across 5 headers; ~20 public headers untouched |
| Bound-but-unused | 4 functions (see §3) |

---

## 5. Native backend matrix

Source tree contains backend dirs: `cpu, cuda, hip, vulkan, metal, cann, musa,
opencl, webgpu, rpc, sycl, blas, et, hexagon, openvino, virtgpu, zdnn, zendnn`
(+ core `ggml-alloc/opt/threading`, which are not backends).

CMake enablement (observed in `ggml/CMakeLists.txt`; `ggml_add_backend(X)` reads
`GGML_X`):

| Backend | In source | CMake flag / default | In forgeCORE build | Runtime discovery | Model execution |
|---|---|---|---|---|---|
| CPU | yes | `GGML_CPU`, ON | yes (`libggml-cpu`) | yes — the only device | yes (tensors + decode) |
| CUDA | yes | `GGML_CUDA`, OFF | no | no | `NOT_SUPPORTED` (build lacks it; load path also CPU-locked) |
| HIP/ROCm | yes | `GGML_HIP`, OFF | no | no | `NOT_SUPPORTED` (same) |
| Vulkan | yes | `GGML_VULKAN`, OFF | no | no | `NOT_SUPPORTED` (same) |
| Metal | yes | `GGML_METAL`, ON on Apple / OFF elsewhere | no (Linux) | no | `NOT_SUPPORTED` here |
| MUSA | yes | `GGML_MUSA`, OFF | no | no | `NOT_SUPPORTED` |
| SYCL | yes | `GGML_SYCL`, OFF | no | no | `NOT_SUPPORTED` |
| OpenCL | yes | `GGML_OPENCL`, OFF | no | no | `NOT_SUPPORTED` |
| WebGPU | yes | `GGML_WEBGPU`, OFF | no | no | `NOT_SUPPORTED` |
| RPC | yes | `GGML_RPC`, OFF | no | no | `NOT_SUPPORTED` (+ needs remote peers) |
| CANN | yes (`ggml-cann/`, header installed) | **no declared option**; `ggml_add_backend(CANN)` reads undeclared `GGML_CANN` (unset = off) | no | no | `NOT_SUPPORTED`; enablement default `UNKNOWN` |
| OpenVINO | yes | `GGML_OPENVINO`, OFF | no | no | `NOT_SUPPORTED` |
| ET | yes | `GGML_ET`, OFF | no | no | `NOT_SUPPORTED` |
| Hexagon | yes | `GGML_HEXAGON`, OFF | no | no | `NOT_SUPPORTED` |
| zDNN | yes | `GGML_ZDNN`, OFF | no | no | `NOT_SUPPORTED` (s390x-only in practice) |
| VirtGPU | yes | `GGML_VIRTGPU`/`_BACKEND`, OFF | no | no | `NOT_SUPPORTED` |
| BLAS / Accelerate / LLAMAFILE / OpenMP / ZenDNN | yes | CPU enhancers (ON/defaults vary) | (folded into CPU backend) | not devices | n/a |

Critical distinctions (the brief's core question):

- **Source support ≠ forgeCORE support.** Seventeen backend dirs exist; one is built.
- **forgeCORE compile-time support is backend-agnostic**: `enumerate_devices` /
  `Backend::open_device` take whatever the registry holds, and the non-mandatory
  GPU probe test would exercise an accelerator with zero Rust changes if one
  appeared. So ggml-*tensor* execution on a future accelerator is
  `PARTIALLY_IMPLEMENTED` (plumbing exists, never verified).
- **libllama model execution is CPU-locked by code, not just by build**:
  `Model::load_with_options` forces `n_gpu_layers = 0` and exposes no device
  selection. Even against a CUDA-enabled native build, the current API could not
  offload. This is the single most important gap for GPU work (§18, P3).
- Environment: no `/dev/nvidia*`, no `/dev/dri`, 2 cores. All accelerator
  behavior is unverifiable here and reported as such.

---

## 6. Device abstraction audit

Existing surface: `enumerate_devices() → Vec<DeviceInfo>`,
`DeviceInfo { index, name, description, device_type, memory_free, memory_total }`,
`DeviceType::{Cpu, Gpu, Igpu, Accel, Meta, Unknown(i32)}`,
`Backend::{open_cpu, open_device, set_cpu_threads, device_type, name}`.

Observed runtime fact (Rust test + independent C probe agree):
`dev_count=1; dev[0] name=CPU type=0 free=2081390592 total=2081390592 (~2 GB),
desc=Intel Xeon @ 2.60GHz`.

| Audit question | Finding |
|---|---|
| Device types / identifiers | Kind enum (+ `Unknown` arm) and registry index; **no PCI `device_id`** (`AVAILABLE_NATIVELY` via props, `NOT_EXPOSED`) |
| Discovery API | `IMPLEMENTED` (`ensure_registry` via `OnceLock`, `load_all` once per process) |
| Memory information | free/total as `usize` (`IMPLEMENTED`, but no Known/Unknown distinction — §13) |
| Capability information | `NOT_EXPOSED` (`ggml_backend_dev_caps`, `get_props`, `supports_op/buft`, `offload_op`, guid all `AVAILABLE_NATIVELY`) |
| Selectable for **tensor** execution | `IMPLEMENTED` (`open_device`, stale-index + vanished-device errors) |
| Selectable for **model** execution | `NOT_SUPPORTED` (load path takes no device; `devices` param untouched) |
| Contexts inherit placement | No — contexts take no device input at all |
| GPU layers / offload exposed | `NOT_SUPPORTED` (`n_gpu_layers` forced 0) |
| Multiple devices representable | List only; no selection-by-id, no pairing, no scheduler use |

Gaps/duplication: no duplication (one discovery path). Gaps are all omissions
listed above; the `Unknown(i32)` arm and stale-index validation are the two
forward-compat strengths to preserve.

---

## 7. Model/context execution path

Traced path (all observed in `model.rs` → `context.rs` → `batch.rs`):

1. **Load:** `Model::load(path)` → `load_with_options` → `ensure_registry` →
   UTF-8/NUL path validation → `llama_model_default_params()` +
   `n_gpu_layers = 0` + `check_tensors` passthrough →
   `llama_model_load_from_file` (NULL → `Error::model`) → `Rc<ModelInner>`.
2. **Metadata:** 9 getters (`n_params`, `vocab_size` via vocab handle,
   `description` with snprintf regrow loop, `size_bytes`, `n_ctx_train`,
   `n_embd`, `n_layer`, `n_head`, `n_head_kv`); negatives rejected.
3. **Context:** `Context::open` validates `n_threads ≥ 1` →
   `llama_context_default_params()` + `n_ctx`, `n_threads`/`n_threads_batch`
   (both = same value) → `llama_init_from_model` (NULL → `Error::context`) →
   resolves `n_ctx == 0` to `n_ctx_train`, caches `n_vocab`; holds `Rc` model share.
4. **Batch:** `BatchBuilder::new(vocab)` validates ids < vocab and count <
   `c_int::MAX`; `build()` freezes token/pos/seq/logits `Vec`s, single shared
   seq id 0; `as_sys()` builds the by-value view (`embd` NULL).
5. **Decode:** non-empty check (upstream aborts on `n_tokens ≤ 0`) →
   `llama_decode` → codes mapped: `0` ok, `1` no-KV-slot, `2` aborted,
   `-1` invalid batch, else fatal. No retry, no fallback.
6. **Logits:** `llama_get_logits_ith` (NULL → `Error::logits`); row of
   `n_vocab` floats copied into owned `Logits`. `&mut self` matches upstream.
7. **Release:** `Drop` frees context, then the `Rc` model share; batch/logits
   are pure Rust memory. Reopen-after-drop covered by test.

Expressiveness checklist:

| Capability | Verdict | Exact reason |
|---|---|---|
| CPU-only execution | `IMPLEMENTED` | validated end-to-end |
| GPU offload | `NOT_SUPPORTED` | `n_gpu_layers = 0` hardcoded; no option field |
| Explicit backend selection (tensors) | `IMPLEMENTED` | `open_device` |
| Explicit backend/device selection (model) | `NOT_SUPPORTED` | `devices`, `main_gpu`, `tensor_split`, `split_mode` unsettable |
| CPU+GPU execution | `NOT_SUPPORTED` | same as above |
| Multi-GPU execution | `NOT_SUPPORTED` | same as above |
| Configurable context size | `IMPLEMENTED` | `n_ctx` (0 = train) |
| Batch / micro-batch size | `NOT_EXPOSED` | `n_batch`/`n_ubatch` left at upstream defaults |
| Thread count | `PARTIALLY_IMPLEMENTED` | gen + batch locked to one value; no post-open adjust |
| KV-cache configuration | `NOT_EXPOSED` | `type_k/v`, `n_seq_max`, `offload_kqv`, `kv_unified`, seq ops all untouched |
| Model loading options | `PARTIALLY_IMPLEMENTED` | only `check_tensors`; 14 params fields unsettable |
| Memory-related configuration | `NOT_SUPPORTED` | no placement/buffer/override knobs |

Points where capability is lost (native → Rust): `llama_model_params`
(devices, buft overrides, split/load/lazy modes, main_gpu, tensor_split,
progress cb, kv overrides, vocab_only, extra bufts, no_host, no_alloc,
load_mtp); `llama_context_params` (~30 fields incl. batching, RoPE, KV types,
offload flags, abort cb, samplers); batch (multi-seq, embeddings);
post-open controls (`llama_set_n_threads`, `llama_synchronize`, abort).

---

## 8. Tensor/data model audit

`Tensor` owns metadata ctx + tensor + backend buffer; shapes in ggml `ne`
order, ≤4 dims, extents validated (nonzero, i64 range, overflow-checked);
`DType` mirrors 19 `ggml_type` ids with explicit rejection of the rest.

| Abstraction | Status | Native equivalent / note |
|---|---|---|
| Name | `NOT_EXPOSED` | tensors are anonymous; `ggml_set_name` `AVAILABLE_NATIVELY` |
| Shape / rank | `IMPLEMENTED` (stored in Rust, ≤4) | trusted copy, **not** re-read from native |
| Element count | `IMPLEMENTED` (Rust `product()`) | `ggml_nelements` is bound but **unused** |
| Byte size | `IMPLEMENTED` | `ggml_nbytes` |
| DType / quant type | `IMPLEMENTED` (19 ids) | `ggml_type`; 24 ids rejected (see §12) |
| Device / backend | `IMPLEMENTED` (via owning `Backend`) | same-`Rc` check in `runtime` |
| Buffer type / placement | `NOT_EXPOSED` | `buft_*` family `AVAILABLE_NATIVELY`; always default alloc |
| Allocation | `IMPLEMENTED` (create-only) | `ggml_backend_alloc_ctx_tensors`; no reserve/plan |
| Transfer host↔device | `PARTIALLY_IMPLEMENTED` | F32 only (`tensor_set/get`); F16/I8/etc. rejected; no 2D/async/memset/copy |
| Cross-backend transfer | `NOT_EXPOSED` | `ggml_backend_tensor_copy[_async]` `AVAILABLE_NATIVELY` |
| Data access (elements) | `NOT_EXPOSED` | `ggml_get/set_f32/i32_*` `AVAILABLE_NATIVELY` |
| Ops | `PARTIALLY_IMPLEMENTED` | `add` + `mul_mat` of ~hundreds of `ggml_*` constructors |
| Graph execution | `PARTIALLY_IMPLEMENTED` | one-node graphs; no plan reuse, no sched, no async, no events |
| Synchronization | `NOT_EXPOSED` | `ggml_backend_synchronize` bound but **unused**; `llama_synchronize` unbound |

`C = A·Bᵀ` semantics for `matmul` are documented and oracle-checked; the one
sharp edge is that `Tensor` trusts its Rust-side shape copy (shape is never
round-tripped through the native tensor).

---

## 9. GGUF metadata boundary

- forgeCORE performs **zero GGUF parsing**: no `gguf.h` bindings (61 native
  APIs untouched), no KV reading, no tensor-descriptor parsing. Verified by
  header-use inventory + source grep.
- Execution-relevant metadata comes from 9 libllama getters (§7.2). Nothing is
  duplicated: forgeCORE never re-derives what libllama reports.
- Planning/inspection metadata (tensor names/shapes/dtypes, KV pairs, file
  layout) is currently RAMforge's job via its own parser (RAMforge's adapter
  docstring confirms the inspection/execution split; RAMforge was otherwise not
  audited this phase).
- Long-term boundary (recommendation): RAMforge keeps file-level planning
  (ranges, layout, budgets); forgeCORE adds execution-relevant model facts it
  does not yet expose (offload capability, `ftype`, encoder/decoder shape,
  rope type, vocab details) and — only if RAMforge needs authoritative
  answers — a minimal read-only GGUF inspection view over `gguf.h`. No
  speculative GGUF API is proposed in §18.

---

## 10. Tokenizer audit

forgeCORE has **no tokenizer**: no `encode`, no `decode`, no vocab-info API
beyond `vocab_size()`, no special-token queries, no BOS/EOS handling. The
`TokenId = u32` alias and `QWEN25_DIAGNOSTIC_PROMPT{,_IDS}` constants are
placeholders, not a tokenizer (the latter is explicitly documented as a
non-contract fixture placeholder).

Natively available and verified in `llama.h` (`NOT_EXPOSED`, all
`AVAILABLE_NATIVELY`): `llama_tokenize`, `llama_detokenize`,
`llama_vocab_type/n_tokens`, `llama_vocab_get_text/score/attr`,
`llama_vocab_is_eog/is_control`, `llama_vocab_bos/eos/eot/sep/nl/pad/mask`,
`llama_vocab_get_add_bos/eos/sep`, suppress-tokens, FIM tokens, plus deprecated
`llama_token_*` / `llama_n_vocab` aliases (which must **not** be bound —
bind the `llama_vocab_*` generation).

Per the brief, the tokenizer is not replaced in this phase (there is nothing to
replace). It is the recommended next phase (§20).

---

## 11. KV cache and batching audit

| Item | Native (`llama.h`) | forgeCORE |
|---|---|---|
| Batch construction | `llama_batch_init/free/get_one` | own builder (equivalent for single-seq); native helpers unbound |
| Token batches, positions, logits flags | struct fields | `IMPLEMENTED` (single seq 0) |
| Multi-sequence batches | `seq_id` arrays, `n_seq_max` | `NOT_EXPOSED` |
| Embedding batches | `embd` pointer, `embeddings` mode | `NOT_EXPOSED` |
| `n_ctx` / `n_batch` / `n_ubatch` | context params | `n_ctx` only; rest at defaults |
| KV sequence ops | `llama_memory_clear/seq_rm/cp/keep/add/div/pos_min/max/can_shift` | `NOT_EXPOSED` |
| Context state | `llama_state_get/set_size/data`, save/load file, per-seq variants | `NOT_EXPOSED` |
| Cache sizing/placement | `type_k/v`, `offload_kqv`, `kv_unified`, `n_outputs_max*` | `NOT_EXPOSED` |
| Abort / cancel | `llama_set_abort_callback`, code 2 mapping | code mapped; callback `NOT_EXPOSED` |

Native-vs-policy split (recommendation): sequence/mask/rollback mechanics, state
bytes, and cache allocation are native execution primitives → forgeCORE should
expose them thinly. Prompt packing, eviction choice, retention priority, and
multi-tenant scheduling stay RAMforge policy. forgeCORE must never invent a
second KV implementation (the pre-pivot `kv.rs` was correctly removed).

---

## 12. Quantization matrix

Upstream `enum ggml_type` has 43 ids (0–42; slots 4–5, 31–33, 36–38 removed).
forgeCORE `DType` maps 19. Scalar oracle decoders exist for F16/BF16/Q4_0/Q8_0/
Q6_K (per module docs); execution through `runtime` is F32-only.

| Format | Native id | `DType` | Oracle decode | Direct-tensor exec | Via model load |
|---|---|---|---|---|---|
| F32 | 0 | yes | n/a (native) | yes | yes |
| F16 | 1 | yes | yes | no (transfer rejects) | yes (libllama-internal) |
| Q4_0 | 2 | yes | yes | no | yes |
| Q4_1 | 3 | yes | no (explicit unsupported) | no | yes |
| Q5_0 | 6 | yes | no — geometry flagged re-verify | no | yes |
| Q5_1 | 7 | yes | no | no | yes |
| Q8_0 | 8 | yes | yes | no | yes |
| Q8_1 | 9 | yes | no | no | yes |
| Q2_K | 10 | yes | geometry only | no | yes |
| Q3_K | 11 | yes | geometry only | no | yes |
| Q4_K | 12 | yes | geometry only | no | yes |
| Q5_K | 13 | yes | geometry only | no | yes |
| Q6_K | 14 | yes | yes | no | yes |
| Q8_K | 15 | yes | geometry only | no | yes |
| I8/I16/I32 | 24/25/26 | yes | no | no | yes |
| F64 | 28 | yes | no | no | yes |
| BF16 | 30 | yes | yes | no | yes |
| IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_S/M, IQ4_NL/XS (16–23, 29) | yes | **no** | no | no | yes (opaque to forgeCORE) |
| I64 (27) | yes | **no** | no | no | yes |
| TQ1_0/TQ2_0 (34/35) | yes | **no** | no | no | `UNKNOWN` (ternary; needs verification) |
| MXFP4/NVFP4 (39/40) | yes | **no** | no | no | `UNKNOWN` (new float microscaling) |
| Q1_0/Q2_0 (41/42) | yes | **no** | no | no | `UNKNOWN` |

Notes: (a) "via model load = yes" means libllama decodes weights internally —
forgeCORE never sees the format, so model-level quant support is inherited, not
implemented. (b) Per-backend kernel support is `UNKNOWN` — CPU-only environment,
no evidence either way, and no claim is made. (c) `DType::from_ggml` correctly
rejects unmapped ids (tested, incl. removed slot 4). (d) The oracle Q5_0
geometry caveat in CONVENTIONS.md is still open.

---

## 13. Memory/resource accounting

Exposed today: `DeviceInfo.memory_free/memory_total: usize` and
`Model::size_bytes() -> u64`. That is the complete inventory — no host-RAM API,
no allocation tracking, no context/KV/graph-scratch reporting, no
Known/Unknown typing (bare integers; a backend reporting 0s would be
indistinguishable from "none").

| Question | Status |
|---|---|
| Host RAM info | `NOT_EXPOSED` (CPU `dev_memory` reports ~2 GB here — observed value, presumably constrained system memory; semantics not verified against backend source) |
| Device memory (free/total) | `IMPLEMENTED` (snapshot at enumerate time; GBM-style staleness accepted by design) |
| Backend/buffer sizes | `NOT_EXPOSED` (`buffer_get_size/max_size/alloc_size`, `buft_get_*` `AVAILABLE_NATIVELY`) |
| Allocation/free events | `NOT_SUPPORTED` (no native callback surfaced; poll-only native facts exist) |
| Model memory | `IMPLEMENTED` (`size_bytes`, upstream-accounted) |
| Context / KV memory | `NOT_EXPOSED` (`llama_state_get_size` = closest native fact, `AVAILABLE_NATIVELY`; no dedicated context-memory getter was found in the audited surface) |
| Graph scratch | `NOT_EXPOSED` (`gallocr_get_buffer_size` `AVAILABLE_NATIVELY`) |
| Known vs Unknown | `NOT_SUPPORTED` — no such type exists; this is a required Phase-1+ direction, never fabricate |

---

## 14. Error handling

`Error(pub String)` + 8 domain-prefixed constructors; `Display` + `std::error`.
Observed behavior: native NULLs → `Error::backend/model/context/logits` with
context; decode codes → `Error::decode` with per-code meanings; validation →
`Error::invalid/unsupported` naming the bad field. No panics on the fallible
path (only `Error::backend` on OOM-shaped NULLs). No silent fallbacks anywhere
(there is nothing to fall back *to* — a strength to preserve, especially for
future device selection: requesting CUDA must error, never quietly run on CPU).

Required direction (not implemented this phase):

- Typed variants (or at least prefix discipline — already present) for:
  unsupported backend, unavailable/mismatched device, invalid device index
  (exists), unsupported model config, native init failure (exists),
  context creation failure (exists), execution failure (exists).
- The gap is narrow: mostly *device/offload-specific* errors for P3, plus a
  decision on whether `Error` stays a string or gains variants before the
  public API freezes. Recommendation: keep the string + prefixes through the
  next 2–3 phases; introduce variants only when RAMforge needs to match on them.

---

## 15. CPU/threading capabilities

| Item | Status |
|---|---|
| Thread count (backend) | `IMPLEMENTED` (`set_cpu_threads`, CPU-only guard, ≥1 validated) |
| Thread count (context) | `PARTIALLY_IMPLEMENTED` (gen+batch locked together; `llama_set_n_threads` post-open adjust `NOT_EXPOSED`; `llama_n_threads*` getters `NOT_EXPOSED`) |
| Topology (cores, NUMA) | `NOT_EXPOSED` (`std::thread::available_parallelism` is caller's job; `ggml_numa_init/is_numa` `AVAILABLE_NATIVELY`) |
| SIMD / ISA features | `NOT_EXPOSED` (`ggml_cpu_has_avx2/fma/avx512/…` — 13+ getters `AVAILABLE_NATIVELY`) |
| Threadpool control | `NOT_EXPOSED` (`ggml_threadpool_*` `AVAILABLE_NATIVELY`) |
| Affinity / pinning | `NOT_SUPPORTED_BY_PINNED_NATIVE_VERSION` as a public C API (no affinity entry found in audited headers; internal threading exists) |
| CPU architecture string | `IMPLEMENTED` indirectly (device `description`: "Intel(R) Xeon(R) Processor @ 2.60GHz") |

Environment facts: 2 logical cores (`nproc`), x86_64, gcc 14.2, `-march=native`
CPU backend build. Determinism note: single-threaded decode of the fixed
fixture is bit-identical across fresh contexts (tested).

---

## 16. Multi-device capabilities

Labels per the brief (`SUPPORTED` = usable through forgeCORE today):

| Capability | Verdict | Evidence |
|---|---|---|
| Enumerate multiple devices | `SUPPORTED` (list) | registry-driven; 1 device observed here |
| Select device for ggml tensors | `SUPPORTED` | `open_device`; untested beyond CPU |
| `n_gpu_layers` offload | `NOT_EXPOSED_BY_FORGECORE` | hardcoded 0; native param exists |
| `devices` list / `main_gpu` | `NOT_EXPOSED_BY_FORGECORE` | native params exist |
| `tensor_split` proportions | `NOT_EXPOSED_BY_FORGECORE` | native param exists (`llama_max_devices` sizing) |
| `split_mode` NONE/LAYER/ROW/TENSOR | `NOT_EXPOSED_BY_FORGECORE` | native enum exists |
| `load_mode` / `lazy_mode` | `NOT_EXPOSED_BY_FORGECORE` | native enums exist |
| Tensor-parallel execution (ROW "if supported") | `UNKNOWN` | support depends on backend + build; no hardware to probe |
| Multi-backend ggml graphs (`ggml_backend_sched_*`) | `NOT_EXPOSED_BY_FORGECORE` | full sched API present in `ggml-backend.h` |
| META device use | `NOT_EXPOSED_BY_FORGECORE` | type mappable; no API consumes it |
| Tensor-split distribution policy | `NOT_EXPOSED_BY_FORGECORE` | native `tensor_split` + sched; policy is RAMforge's |
| RPC-distributed execution | `UNKNOWN` | backend source+flag exist; needs peers; never exercised |
| CPU/GPU hybrid (KQV/ops offload flags) | `NOT_EXPOSED_BY_FORGECORE` | `offload_kqv`, `op_offload` params exist |

No fictional multi-GPU abstraction exists or is proposed for the near term; P3
(§18) wires the real parameters without claiming untested behavior.

---

## 17. forgeCORE vs RAMforge responsibility boundary

Observed current state: RAMforge (`/home/user/RAMforge`, separate repo, not
modified) already depends on `forge-core` via pinned git rev and isolates it
behind a `forgecore_backend` adapter (manifest + module docstring evidence).
RAMforge currently owns tokenization (via its inspection parser) and sampling
policy; forgeCORE owns load/context/batch/decode/logits. Dependency direction
`RAMforge → forgeCORE → llama.cpp/ggml` holds; the reverse does not exist.

Proposed boundary (recommendation; each row justified by the audit):

| Concern | Owner | Rationale |
|---|---|---|
| Device/backend discovery + capabilities | forgeCORE | thin wrap of registry facts (§6 gaps to close) |
| ggml tensor/buffer/graph execution | forgeCORE | native substrate; RAMforge must not touch `ggml_*` |
| Model loading + execution metadata | forgeCORE | already owned; extend offload opts (P3) |
| Text tokenizer (encode/decode/vocab) | **forgeCORE (proposed move)** | native tokenizer is authoritative; RAMforge's is inspection-derived duplication for execution purposes |
| Sampling primitives (chains, seeds) | forgeCORE exposes, RAMforge parameterizes | native sampler is the execution truth; policy (temp schedules, grammars-as-policy) stays RAMforge |
| Batch/KV/state mechanics | forgeCORE exposes thinly | native-owned behavior (§11) |
| Memory *facts* (Known/Unknown) | forgeCORE reports | only native knows true residency |
| Memory *budgets*/policy, storage hierarchy | RAMforge | orchestration by design |
| Layer streaming, residency, eviction, prefetch | RAMforge | policy over forgeCORE primitives |
| Scheduling, placement, serving, CLI | RAMforge | orchestration by design |
| GGUF file planning (ranges, layout) | RAMforge (keep) | no execution need; avoid duplication |

Workspace/integration answers (§16 of the brief): keep two repos with a pinned
git dependency (current state works); RAMforge must depend on forgeCORE, never
the reverse (verified absent); native dependency + build flags owned by
forgeCORE (`setup-native.sh`, `FORGE_LLAMA_DIR`); feature flags (when they
arrive) owned by forgeCORE with RAMforge selecting, never defining, backends;
multi-backend enablement = native rebuild + registry discovery, no Rust coupling;
no circularity risk as long as `forge-sys`/`forge-core` keep zero workspace-external
deps and RAMforge never leaks into them (verified true today).

---

## 18. Proposed phased implementation plan

Common validation for every phase: `fmt --check`, `cargo check`, full test
suite with + without `FORGE_TEST_MODEL`, `clippy -D warnings`, release build,
independent C probe where behavior is new. Common out-of-scope: RAMforge
changes, residency/eviction/prefetch policy, second-engine numerics in Rust.

**P1 — Tokenizer + vocabulary (recommended next; see §20).**
Objective: safe native text tokenization so execution stops depending on
external tokenizers. Scope: bind `llama_tokenize/detokenize` + `llama_vocab_*`
(non-deprecated generation only); add `forge_core::tokenizer::{Tokenizer, VocabInfo,
SpecialTokens}` borrowing `Model`; UTF-8/buffer-growth handling mirroring
`description()`. Files: `forge-sys/lib.rs`, new `forge-core/src/tokenizer.rs`,
`cpu_decode.rs` additions. Tests: fixture round-trip (ids→text→ids), special
tokens match fixture header (bos 1/eos 2/unk 0), overlong-input growth,
invalid-UTF8 errors. Risks: `llama_tokenize` negative-return protocol;
add_bos/sep defaults must be explicit, never guessed. Depends on: nothing.
Out: sampling, chat templates.

**P2 — Sampler primitives.**
Objective: native sampling chains callable from safe Rust. Scope: bind sampler
lifecycle + `greedy/dist/top_k/top_p/min_p/temp` + `sample`; `SamplerChain`
builder owning the chain; seed control; `accept` for multi-step use. Tests:
greedy determinism on fixture logits; distribution shape invariants (seeded);
chain add/remove/n. Risks: sampler/context lifetime coupling (`set_sampler`
semantics); thread-safety stays `!Send`. Depends on: P1 (vocab for bias/infill
later; greedy needs none). Out: grammar samplers, mirostat, DRY/XTC (later pass).

**P3 — Model offload wiring (no untested claims).**
Objective: express GPU placement honestly. Scope: extend `ModelOptions`
(`n_gpu_layers`, `main_gpu`, `split_mode` enum, `tensor_split` array,
`devices` selection from `DeviceInfo`) with validation (e.g. refuse GPU
requests when no GPU device is registered — error, never CPU fallback);
bind `llama_supports_gpu_offload/mmap/mlock`, `llama_max_devices`.
Tests: CPU-path unchanged; GPU-request-without-GPU errors explicitly;
skip-gated offload test for GPU CI. Risks: claiming behavior without hardware
— mitigate by construction (options + validation are testable on CPU; actual
offload stays skip-gated). Depends on: nothing. Out: KV offload flags (P4),
multi-GPU distribution policy (RAMforge).

**P4 — Context/batch extension.**
Objective: remove the Phase-1 batching blind spots. Scope: expose
`n_batch/n_ubatch/n_seq_max` in `ContextOptions`; multi-sequence `BatchBuilder`
+ embedding-batch support; bind/use `llama_set_n_threads` (+ getters) and
`llama_synchronize`; use bound-but-unused `ggml_backend_synchronize` or
document why not. Tests: multi-seq decode on fixture; ubatch<batch behavior;
thread-count change mid-life. Risks: KV consecutiveness rules (already mapped);
embeddings-mode context creation flags. Depends on: P1 (multi-seq prompts).
Out: KV sequence surgery (P5).

**P5 — KV/state primitives + honest memory facts.**
Objective: native cache control + measurable memory. Scope: bind
`llama_memory_*` seq ops + `llama_state_get/set_size/data` + file save/load;
add `MemoryFact { Known(u64) | Unknown }` reporting (device, model, state
bytes); bind `llama_state_seq_*` only if a RAMforge need is demonstrated.
Tests: seq rm/cp round-trip; state save→clear→restore bit-equality on fixture;
every reported number traced to a native call. Risks: state-format coupling
across versions (pin-pinned, documented). Depends on: P4. Out: budgets,
eviction, streaming (RAMforge).

**P6 — ggml execution surface for streaming.**
Objective: the buffer/graph controls RAMforge streaming needs. Scope: bind
buffer types (`buft_*`), `tensor_copy`, graph plan create/compute, `supports_op`,
device props/caps; safe `BufferType`/`GraphPlan` handles; tensor transfer for
F16/I8 (narrow, tested). Tests: cross-buffer copy bytes; plan reuse; caps-gated
op selection errors. Risks: lifetime complexity (buffers vs tensors) — keep the
`Tensor`-owns-buffer pattern. Depends on: P3 (device selection). Out: full op
coverage (add ops only on demonstrated need), sched multi-backend graphs (P7).

**P7 — Observability + error hardening.**
Objective: production diagnostics. Scope: bind `llama_perf_context/sampler`,
`llama_print_system_info`, `llama_log_set` routing (use bound `ggml_log_set`
or remove it); decide typed `Error` variants vs prefixes with RAMforge input.
Tests: perf counters move across decodes; log routing capture. Risks: perf
struct layout drift (layout-test it). Depends on: P2, P5. Out: metrics
pipelines (RAMforge).

---

## 19. Validation results

Environment: 2 cores, no GPU, rustc/cargo **1.99.0**, cmake 4.1.2, gcc 14.2.0,
fresh clone re-verified at `v0.5.0` @ `7fe450e1`, ggml libs 0.25.1. (Toolchain
is unpinned by design — reinstalls track stable; noted, not changed.)

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo check --workspace` | clean, no warnings |
| `cargo test --workspace` (with `FORGE_TEST_MODEL=tiny-llama.gguf`) | **81 passed, 0 failed**: 54 lib, 9 `cpu_decode`, 9 `ggml_smoke`, 5 `quant_contract`, 4 `forge-sys` layout |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | clean |
| `cargo build --workspace --release` | clean |
| Independent C device probe (`dev_probe.c`, no Rust) | `dev_count=1`, CPU type 0, free=total=2081390592 — agrees with Rust output exactly |
| `FORGE_TEST_MODEL` unset path | suite still green (fixture tests SKIP) — verified in prior runs; not re-run this session |

Cleanliness: no `target/`, `*.gguf`, `*.so`, or caches under `/home/user`
(verified by find); `.gitignore` now covers `target/`, `*.gguf`, `*.bin`,
editor files. There is no git repository in forgeCORE (history was removed by
owner decision), so "accidentally committed" is inapplicable — but the ignore
rules are in place for any future init. Validation ran before the doc-only
fixes; no code changed after, so results stand.

---

## 20. Known limitations and unresolved questions

1. **CPU-only evidence.** Every accelerator, offload, split-mode, and
   per-backend quantization statement is structural (headers/CMake) rather than
   behavioral. A GPU CI leg is the single highest-value validation upgrade.
2. **CANN enablement** (`ggml-cann/` + installed header, no declared CMake
   option; `ggml_add_backend(CANN)` reads undeclared `GGML_CANN`): presumed
   `-DGGML_CANN=ON`-able, default effectively off — `UNKNOWN`, needs an
   Ascend machine or upstream docs to confirm.
3. **No dedicated native context-memory getter found** in the audited surface;
   `llama_state_get_size` is the closest verified fact. (The audit covered the
   public headers' declaration lists, not every function body — a deeper
   function-level pass could still find one.)
4. **`reference::checkpoint` is dead public API** (no consumers). Either wire a
   consumer (old-fixture test) or remove it in a cleanup pass — flagged, not
   acted on per phase scope.
5. **`QWEN25_DIAGNOSTIC_PROMPT{,_IDS}`** in `model.rs` are documented
   placeholders, unreferenced by any test. Keep until real-model fixtures exist.
6. **4 bound-but-unused FFI declarations** (§3). Harmless; §18 assigns each a
   first-use phase.
7. **Debug-vs-Release abort behavior** (`logits_ith` NULL vs abort) is carried
   from the PIVOT record, not re-probed this session.
8. **RAMforge's exact tokenizer/sampling contract** was inspected only
   minimally (manifest + adapter docstring) per phase scope; P1/P2 should
   confirm RAMforge's needs before freezing APIs.
9. **Workspace invariant vs session state:** `docs/TOOLCHAIN.md` and
   `scripts/env.sh` require `/home/user/` to contain only `forgeCORE/`, but
   `/home/user/RAMforge/` exists (created by the RAMforge Phase-2 task). This
   is a cross-task session condition, not a forgeCORE defect; the owner should
   confirm whether the invariant means "forgeCORE tooling creates nothing
   else" (holds) or "nothing else may exist" (currently violated).
10. **Unverified PIVOT carry-overs:** the `~2 GB` CPU memory semantics and the
    pre-pivot session notes referenced in §6 are historical; re-verified only
    insofar as current tests assert current behavior.

---

## Recommended next phase

**P1 — Tokenizer + vocabulary API** (§18).

Precise objective: expose the pinned native tokenizer through safe Rust so that
no execution path depends on an external or reimplemented tokenizer.

Scope (and only this): bind `llama_tokenize`, `llama_detokenize`, and the
non-deprecated `llama_vocab_*` getters; add `forge_core::tokenizer` with a
`Model`-borrowing `Tokenizer`, vocab text/score/attribute access, and special
tokens; fixture tests for round-trip fidelity, special-token values, buffer
growth, and UTF-8 errors; validation per §18's common bar. Explicitly out:
sampling (P2), chat templates, tokenizer overrides, any RAMforge change.
