# ForgeCore Phase 6 — Engineering report

**Date:** 2026-10-06
**Upstream pin:** llama.cpp `v0.5.0` /
`7fe450e19305b828c199d602c23a8337aaa1f03b` / ggml `0.25.1`
**Status: COMPLETE.** All suites green, all audits done, no regressions.

> Path note: this report lives at `docs/` root, not `docs/handoff/`.
> `docs/handoff/` is frozen RAMforge history (Phases 0–5); P6 is
> ForgeCore work and stays out of it.

## 1. Scope

P6 is the safe minimal low-level ggml execution surface: tensor,
runtime, quant, and backend-graph suites over the pinned native
build, with a mandatory deterministic real-CPU execution proof
through the public API. Acceptance required, in order: the
integration suites, the CPU proof, an ownership/lifetime audit,
abort-safety validation, error-model and FFI audits, doc updates, a
read-only RAMforge compatibility audit, full regression validation,
and hygiene. P0–P5 behavior had to stay intact throughout.

## 2. Abort battery (abort-safety validation)

**Method.** `/var/tmp/forge-native/probes/p6_abort.c` (outside the
workspace, never persisted) drives *raw ggml* — no ForgeCore — with
one `argv`-selected case per process, mirroring ForgeCore's exact
allocation flow (`init_by_type` CPU with `cpu_init` fallback,
one metadata ctx per tensor, backend alloc for non-views only, no
`view_init` on aliases). Each case feeds native an input ForgeCore
rejects and records the outcome: `134` SIGABRT, `139` SIGSEGV,
exit `0` + verdict, exit `2` survived, exit `3` layout-skip.
Reproduce with:

```sh
cc -O2 -I$FORGE_LLAMA_DIR/src/ggml/include p6_abort.c \
   -L$FORGE_LLAMA_DIR/build/bin -lggml -lggml-cpu -lggml-base -o p6_abort
LD_LIBRARY_PATH=$FORGE_LLAMA_DIR/build/bin ./p6_abort <case>
```

**Results: 31/31 conclusive, 0 skips.** 21 aborts, 2 segfaults,
7 verdicts, 1 documented survival.

| Case | ForgeCore guard | Outcome |
|---|---|---|
| `add-shape` | add shape check | 134 |
| `matmul-inner` | matmul K check | 134 |
| `graph-overflow` | graph cap bound | 134 |
| `smaxext-geom` | soft_max_ext geometry | 134 |
| `concat-shape` | concat shape check | 134 |
| `view-overrun` | view span check | 134 |
| `reshape` | element-count check | 134 |
| `concat-dim4` | concat dim range | 134 |
| `permute` | permute axis range | 134 |
| `rope-poscount` | rope position count | 134 |
| `rope-posvec` | rope positions rank-1 | 134 |
| `getrows-idxgeom` | get_rows index geometry | 134 |
| `getrows-oob` | get_rows index bounds | 134 |
| `cast-i32f16` | cast pair table | 134 |
| `cast-q8q4` | cast pair table | 134 |
| `cast-f32i8` | cast pair table | 134 |
| `norm-negeps` | norm eps ≥ 0 | 134 |
| `setf32-quant` | fill dtype gate | 134 |
| `getrows-q8k` | get_rows dtype list | 134 |
| `getrows-i8` | get_rows dtype list | 134 |
| `copy-view-src` | copy refuses views | 134 (`GGML_ASSERT(buffer)`, ggml-backend.cpp:205) |
| `copy-view-dst` | copy refuses views | SURVIVED — defense in depth (see F4) |
| `cast-q81f32` | Q8_1 dequant ban | 139 (NULL `to_float` call) |
| `getrows-q81` | Q8_1 dequant ban | 139 (NULL `to_float` call) |
| `rope-odd` | rope even-ne0 | CORRUPT — trailing write clobbered the next pool object header |
| `cast-strided-q` | cast contiguity | MISMATCH — max\|diff\| ≈ 4.0 vs logical transpose |
| `cont-strided-q` | cont/quant rule | MISMATCH — max\|diff\| ≈ 4.0 (identical: `cont` does not rescue strided quants) |
| `add-ok` | — (control) | OK, exact values |
| `rope-even-ok` | — (control) | CLEAN at identical zero gap |
| `cont-q-ok` | — (control) | OK, exact zeros |
| `copy-ok` | — (control) | OK, exact copy |

**Findings.**

- **F1 — Odd-ne0 RoPE corrupts the heap.** With ne0=5 the kernel's
  remainder pair writes 4 bytes past the output tensor; the probe
  catches it clobbering the next pool object's 32-byte allocator
  header. The even-ne0 control at the identical zero gap is clean,
  and a ran-check confirms the kernel executed. The even-ne0 /
  even-n_dims guards are load-bearing against silent heap
  corruption, not just wrong values.
- **F2 — Strided quants silently misread.** A transposed Q8_0 cast
  to F32 returns values ~4.0 off the logical transpose, with no
  abort; routing through `cont()` first produces the *identical*
  wrong output. The cast/contiguity and quant-stride guards are
  load-bearing.
- **F3 — NULL dequantizer rows crash.** Q8_1→F32 and Q8_1 get_rows
  segfault on the NULL `to_float` table entry; Q8_K/I8 get_rows hit
  the dispatch abort. The dtype allow-lists are load-bearing.
- **F4 — Copy is asymmetric.** Source views abort (the branch reads
  `src->buffer` unresolved; ForgeCore views carry NULL buffers),
  while destination views survive (the set path resolves them onto
  the parent buffer). The source refusal is load-bearing; the
  destination refusal is kept as defense in depth. The doc comment
  now states the exact mechanism (it previously claimed both ends
  dereference NULL).
- **F5 — The cast envelope is a strict subset.** Native also
  quantizes F16/BF16→Q*; ForgeCore rejects those pairs as
  unaudited. Docs corrected (they previously implied the list was
  the kernel's full conversion set).
- **F6 — No unpredicted outcomes.** All 20 predicted aborts fired
  (12 constructor, 8 compute), both predicted segfaults fired, and
  every verdict matched its predicted direction. F16 fill needed no
  probe: native supports it, so the ForgeCore rejection is API
  design, confirmed by source read.

## 3. Doc corrections applied

- `Tensor::cast` + `cast_pair_supported`: strict-subset wording;
  strided-source mechanism now cites the native contiguity assert
  (quantizing path) and the probe-measured ~4.0 misread
  (dequantizing path).
- `Tensor::copy_into`: exact copy mechanism (unresolved NULL
  source buffer → native assert, NDEBUG dereference; destination
  resolved via the set path but refused symmetrically); error
  string and the `runtime_param` test comment match.

## 4. Audits

- **Ownership/lifetime.** `Allocation`/`CtxAlloc` free exactly once
  via `Drop`-on-`Rc`; the `History` DAG retains ancestor contexts
  and buffers across graph expansion (the `cpu_exec_proof` UAF
  catch is documented inline); views share the parent's `Rc`
  without `view_init` — consistent with battery finding F4. No
  `Send`/`Sync` impls: thread-confined by construction, so the
  `Rc` sharing is data-race-free. No double-free path: each
  `Allocation` owns exactly one fresh buffer; aliases allocate
  nothing.
- **Error model.** Single `Error(String)` with domain prefixes, one
  constructor per domain, `Display` + `std::error::Error`, unit
  test pinning every prefix. No `unwrap`/`expect`/`panic!` in
  non-test code (one logically-impossible `unreachable!` in
  `model.rs`); all `unwrap`s sit inside `mod tests`.
- **FFI.** All 9 `forge-sys` structs carry `repr(C)`; spot-checked
  declarations match the pinned headers (`rope_ext` 13-arg order,
  `soft_max_ext`, `tensor_copy` const-sides, `set_f32` return,
  `graph_compute` status int, `blck_size`/`is_quantized` widths).
  The surface is exercised end to end by the 336-test suite against
  the real native build.
- **RAMforge (read-only).** ForgeCore carries no functional
  RAMforge coupling: the only mentions are doc comments naming the
  downstream consumer; no code, dependency, CLI, path, or
  environment reference. The layering `RAMforge → ForgeCore →
  llama.cpp/ggml` holds. P6 wrote nothing under `/home/user/RAMforge`
  (its uncommitted Phase-2 worktree changes predate P6 and were
  left untouched); `/home/user/` contains only `forgeCORE/` and
  `RAMforge/`.

## 5. Validation record (2026-10-06)

```
cargo fmt --check                                   # clean
cargo clippy --workspace --all-targets -- -D warnings  # clean
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps  # clean
cargo test --workspace                              # 336 passed, 0 failed (17 suites)
cargo test --workspace --release                    # 336 passed, 0 failed
```

Toolchain/state dirs stayed outside the workspace
(`/tmp/forge-toolchain`, `/var/tmp/forge-native`); no `target/`,
caches, or artifacts under `/home/user`. No git workflow (no `.git`
in ForgeCore, per standing instruction).

## 6. Verdict

P6 is complete: the execution surface is proven on real CPU
hardware, every guard is either load-bearing (demonstrated by abort,
crash, corruption, or misread) or documented defense in depth, and
the full suite passes in both profiles with static gates clean.
RAMforge Phase 4 is **not** started automatically; it awaits an
explicit directive.
