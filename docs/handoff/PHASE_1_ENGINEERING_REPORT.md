# forgeCORE Phase 1 Engineering Report — Tokenizer + Vocabulary API

Closes Phase-0 §10 ("forgeCORE has **no tokenizer**") and the Phase-0
*Recommended next phase* (P1). Upstream pin unchanged: llama.cpp
`v0.5.0` @ `7fe450e19305b828c199d602c23a8337aaa1f03b` (re-verified this
session, §13). No RAMforge change, no sampling, no chat templates, no
GPU work, no KV-cache work, no deprecated-alias bindings.

---

## 1. Executive summary

Safe Rust tokenizer coverage now exists over the pinned native
tokenizer. `forge_core::tokenizer` exposes a `Model`-borrowing
`Tokenizer` with `encode` / `decode`, full vocabulary inspection
(text, score, attribute bitmask, EOG/control predicates),
special-token ids, explicit BOS/EOS configuration, and a vocab-type
query — all backed by 19 new `forge-sys` FFI declarations plus two
constant modules. Every native abort/throw/UB hazard found by reading
`src/llama-vocab.cpp` at the pin is either guarded in Rust (bounds
checks, NONE-vocab refusal, `add_special` pre-checks) or documented
as a residual risk with observed evidence (§7, §16).

The headline finding: **native `llama_tokenize` can throw a C++
exception** (`std::out_of_range` from SPM byte fallback) on vocabs
without byte pieces — observed as `fatal runtime error: Rust cannot
catch foreign exceptions, aborting` (SIGABRT) against the Phase-0
fixture. This is a malformed-vocab condition, unguardable in-process,
and the reason Phase 1 adds a second deterministic fixture,
`tiny-tok.gguf` (vocab 269, full `<0xXX>` byte coverage), leaving
`tiny-llama.gguf` byte-identical (proven by re-generation hash, §11).

Validation: `fmt --check`, `check`, full workspace `test` (debug and
release), `clippy -D warnings`, SKIP-path run, and repo-cleanliness
find — all green. Test total **103 vs the 81 baseline** (+8 unit, +14
integration; §12–§13).

---

## 2. Scope and non-goals

In scope (all delivered):

* Bind `llama_tokenize`, `llama_detokenize`, and the non-deprecated
  `llama_vocab_*` getters the API needs.
* `forge_core::tokenizer`: `Tokenizer<'model>` borrowing `Model`;
  `encode` / `decode`; vocab text/score/attribute access;
  special tokens; explicit BOS semantics (§8).
* Fixture tests for round-trip fidelity, special-token values,
  buffer growth, and UTF-8 errors (coverage map in §12).
* Validation per the Phase-0 §18 common bar (§13).

Explicitly out (unchanged from the brief):

* Sampling (P2), chat templates, tokenizer overrides, GPU, KV-cache.
* Any RAMforge change (the tree was not touched; the
  `docs/TOOLCHAIN.md` vs `/home/user/RAMforge/` session condition
  noted in Phase-0 §20.9 is unchanged).
* Deprecated `llama_token_*` / `llama_add_bos/eos_token` /
  `llama_n_vocab` aliases — verified present upstream, deliberately
  not bound (§4).

Binding-scope decisions (audited, not guessed):

* `llama_vocab_get_add_sep` **not bound**: no audited
  `tokenize`/`detokenize` path reads the sep flag (WPM asserts SEP
  *presence* unconditionally instead, §7), so it has no behavioral
  role in this API.
* `llama_token_to_piece` **not bound**: single-piece lookup is
  covered by `llama_vocab_get_text`; the `lstrip`/`special`
  parameters are `detokenize`-internal concerns.
* `llama_vocab_cls` / `llama_vocab_fim_*` **not bound**: outside the
  brief's special-token list (`bos/eos/eot/sep/nl/pad/mask`).

---

## 3. Public API (`crates/forge-core/src/tokenizer.rs`, 732 lines)

Re-exported from the crate root (`Tokenizer`, `VocabType`,
`TokenAttr`, `SpecialTokens`, `EncodeOptions`, `DecodeOptions`).

```rust
pub struct Tokenizer<'model> { /* borrows &'model Model */ }
impl<'model> Tokenizer<'model> {
    pub fn open(model: &'model Model) -> Result<Self>;
    pub fn n_vocab(&self) -> u32;
    pub fn vocab_type(&self) -> VocabType;
    pub fn adds_bos(&self) -> bool;
    pub fn adds_eos(&self) -> bool;
    pub fn special_tokens(&self) -> SpecialTokens;
    pub fn token_text(&self, id: TokenId) -> Result<String>;
    pub fn token_score(&self, id: TokenId) -> Result<f32>;
    pub fn token_attr(&self, id: TokenId) -> Result<TokenAttr>;
    pub fn is_eog(&self, id: TokenId) -> Result<bool>;
    pub fn is_control(&self, id: TokenId) -> Result<bool>;
    pub fn encode(&self, text: &str, options: &EncodeOptions) -> Result<Vec<TokenId>>;
    pub fn decode(&self, tokens: &[TokenId], options: &DecodeOptions) -> Result<String>;
}

pub enum VocabType { Spm, Bpe, Wpm, Ugm, Rwkv, Plamo2, Test, Unknown(i32) }
pub struct TokenAttr(u32); // bitmask mirror of enum llama_token_attr
pub struct SpecialTokens { pub bos/eos/eot/sep/nl/pad/mask: Option<TokenId> }

#[non_exhaustive] pub struct EncodeOptions { pub add_special: bool, pub parse_special: bool }
#[non_exhaustive] pub struct DecodeOptions { pub remove_special: bool, pub unparse_special: bool }
```

Defaults: `EncodeOptions { add_special: true, parse_special: false }`
(upstream `common` convention), `DecodeOptions { remove_special:
false, unparse_special: false }` (plain pieces, §9). Both option
structs are `#[non_exhaustive]` with `Default`, matching the
`ModelOptions` pattern (configure by field assignment, as in
`cpu_decode.rs`). `VocabType` mirrors the `DeviceType` precedent
(exhaustive variants + `Unknown(i32)` fallback).

---

## 4. FFI surface (`crates/forge-sys/src/lib.rs`, +75 lines)

New constant modules mirroring the upstream enums:

* `vocab_type::{NONE, SPM, BPE, WPM, UGM, RWKV, PLAMO2, TEST}` (`0–7`).
* `token_attr::{UNDEFINED, UNKNOWN, UNUSED, NORMAL, CONTROL,
  USER_DEFINED, BYTE, NORMALIZED, LSTRIP, RSTRIP, SINGLE_WORD}`
  (`0, 1, 2, 4, 8, …, 512` — a bitmask, §10).
* `LLAMA_TOKEN_NULL: c_int = -1` (absence sentinel).

19 new `extern "C"` declarations, signatures verified character by
character against `llama.h` (`bool` ↔ Rust `bool`, `llama_token` ↔
`c_int`, enum returns ↔ `c_int`):

| Declaration | Notes |
|---|---|
| `llama_vocab_type` | assert-free accessor; the safe NONE probe (§7) |
| `llama_vocab_n_tokens` | pre-existing, reused |
| `llama_vocab_get_text/score/attr` | per-token; `at()`-throw hazard |
| `llama_vocab_is_eog/is_control` | `is_control` is unchecked-`[]` UB |
| `llama_vocab_bos/eos/eot/sep/nl/pad/mask` | NULL-or-valid-id protocol |
| `llama_vocab_get_add_bos/get_add_eos` | model BOS/EOS configuration |
| `llama_tokenize` | negative = −required, `INT32_MIN` = overflow |
| `llama_detokenize` | same protocol; writes **no NUL** |

The `extern` block carries a comment stating the exact safety
contract `forge-core` upholds (validate-then-call, NONE refusal).
No `#[repr(C)]` structs were needed, so no layout tests were added;
the 4 pre-existing `forge-sys` layout tests still pass.

---

## 5. Ownership and lifetimes

`Tokenizer<'model>` holds `&'model Model` plus a raw
`*const llama_vocab` and cached metadata (`n_vocab`, `vocab_type`,
`add_bos/eos`, `SpecialTokens`). The borrow (mandated by the brief)
keeps the native model — and its post-load-immutable vocabulary —
alive for every call, so the cached metadata and the raw pointer
cannot dangle; this is stated once as the struct invariant in the
module docs and referenced by each `SAFETY` comment. Caching one
metadata round-trip at `open()` keeps hot paths (`decode` id
validation, `encode` guards) FFI-free. `!Send + !Sync` is inherited
from `Model` (`Rc`-based); no marker impls. `Debug` is manual (no
raw pointers; shows model summary + cached metadata).

---

## 6. `TokenId` conversion discipline

Native `llama_token` is `i32`; ForgeCore uses `TokenId = u32`
(`batch.rs`) in all public signatures. Three conversion points,
each total (no `as` casts on untrusted values):

* **In** (`decode`, per-token getters): `checked_token_id(id,
  n_vocab)` rejects `id >= n_vocab`, else narrows — sound because
  `n_vocab` itself came from a non-negative `c_int`.
* **Out** (`encode`): each native id goes through
  `TokenId::try_from`, mapping a (by-construction impossible)
  negative to `Error::tokenizer` rather than trusting upstream.
* **Specials at `open()`**: `special_from_native` maps
  `LLAMA_TOKEN_NULL → None`, accepts `0 <= id < n_vocab`, and fails
  `open()` on any other negative or out-of-range id (fail fast on a
  corrupt model instead of caching a lie).

`text.len()` / `tokens.len()` conversions to `c_int` use `try_from`
with explicit errors; `Capacity::Grow` values are `<= i32::MAX` by
construction of the protocol (`-ret` for `ret > INT32_MIN`).

---

## 7. Native safety audit (pinned `src/llama-vocab.cpp`)

Read in full at the pin; all line references below are
`src/llama-vocab.cpp` @ `7fe450e1` unless noted. Two facts frame
everything: `GGML_ASSERT(x)` is `if (!(x)) GGML_ABORT(...)`
(`ggml/include/ggml.h:288`) — **always active, including release** —
and `id_to_token` is a `std::vector`, so `.at()` throws and
`operator[]` is unchecked.

| # | Native hazard | Location | Rust guard |
|---|---|---|---|
| 1 | `get_text/score/attr` → `id_to_token.at()` throws `std::out_of_range` across the C boundary = terminate | 4045+ | `checked_token_id` before **every** per-token call |
| 2 | `is_control` → `id_to_token[]` = OOB/UB on invalid id | 3211 | same uniform check (this one averts UB, not just a throw) |
| 3 | every special getter `GGML_ASSERT(type != NONE)` | 3079+ | `open()` reads `llama_vocab_type` first and refuses NONE |
| 4 | `tokenize` head `GGML_ASSERT(tokenizer)` (NULL for NONE) | 3417 | same NONE gate (verified `llama_vocab_type` is a trivial assert-free accessor, 4290–4292, so the probe itself cannot abort) |
| 5 | `detokenize` returns `0` for NONE *before* its assert | 3791–3795 | covered by the same gate; no special case needed |
| 6 | `add_special` insert asserts per type (table below) | 3438–3574, 586–602 | `check_add_special` pre-check in `encode` |
| 7 | `byte_to_token` `.at()` throw on vocabs without byte pieces | 3996–4026 | **residual risk** — no in-process guard exists (§16.1) |
| 8 | detokenize-C++-overload self-check `GGML_ASSERT(check == -n_chars)` | 3376 | unreachable after id validation (deterministic native, fixed input); trusted like `description()` trusts `snprintf` |

`add_special` assert map (finding 6), each verified in its session
body — this table *is* `check_add_special`:

| Vocab type | Aborts iff `add_special` and … | Guard |
|---|---|---|
| SPM / BPE / UGM | `add_bos && bos == NULL`, or `add_eos && eos == NULL` | require ids exactly when the matching flag is set |
| WPM | `bos == NULL` **or** `sep == NULL` (unconditional on the flags) | require both, always |
| RWKV / PLAMO2 / TEST | never (paths ignore `add_special`) | always pass |
| `Unknown(i32)` future type | unknowable | refuse with `Error::unsupported` — never risk an abort |

Per-call precondition summary: `open()` establishes non-NULL vocab
+ non-NONE type + `n_vocab`; every later call runs only after id
validation (`decode`, getters), the add-special guard (`encode`),
or no input at all (cached accessors). `NULL/0` sizing calls match
upstream's own callers (`common_tokenize`).

---

## 8. Encode semantics

`encode(text, options)`: `text_len` via `try_from` (length-delimited
upstream — interior NULs are data, no `CString` needed);
`add_special` pre-checked (§7.6); sizing call (`NULL/0`) then a fill
call (`capacity_step` maps the negative/INT32_MIN protocol; native
tokenization is deterministic for fixed input, so one fill always
fits); output ids narrowed with `try_from` (§6).

* **Fast path**: empty text with `add_special == false` returns
  `vec![]` without calling native (every upstream type only
  iterates the empty fragment list then — verified per type).
* **Explicit BOS semantics**: nothing is implicit. `adds_bos()` /
  `adds_eos()` expose the model flags; `add_special` inserts
  exactly what the model configures (fixture truth table: `"" →
  [1]` with bos-only tiny-llama, `"" → [1, 2]` with tiny-tok);
  missing-required-id is `Error::tokenizer`, never an abort.
* **`parse_special`**: parses `<s>`-style spellings into ids via
  upstream's partitioner (observed: `"<s>" → [1]` vs 6 plaintext
  byte ids); never inserts a leading space (upstream doc).

---

## 9. Decode semantics

`decode(tokens, options)`: empty input returns `""` without calling
native; **all** ids validated before the call (fail-before-native —
an abort-risking id never crosses FFI); same sizing→fill protocol
(the buffer holds raw bytes, upstream writes no NUL); bytes via
strict `String::from_utf8` (`bytes_to_text`).

* **Strict UTF-8 is deliberate**: `decode` returns `String`, and a
  lone `0xE2` leader genuinely is not text — observed clean error
  `tokenizer error: decoded text is not valid UTF-8 (1 bytes)`.
  Byte-exact/streaming callers need a future `decode_bytes` (§17).
* **`remove_special`**: strips a leading BOS / trailing EOS exactly
  when the model configures them (observed `[1,5,2] → "tok2"`).
* **`unparse_special`** (corrected during probing — the first doc
  draft had this backwards): `false` **skips** control pieces
  (`[1,5,2] → "tok2"`), `true` renders them
  (`[1,5,2] → "<s>tok2</s>"`), matching the upstream
  `@param special` doc on `llama_token_to_piece`.
* **Round-trips are byte-verbatim**: pieces concatenate with no
  separator, so SPM round-trips yield U+2581 where the encoder put
  word boundaries (`"a b" → "▁a▁b"`); tests pin
  `format!("▁{}", text.replace(' ', "▁"))`, not naive equality.

---

## 10. Vocab and special-token semantics

* `token_text` returns an owned copy (never upstream's buffer, like
  `description()`), lossy (`to_string_lossy`, like `device.rs`):
  byte-level vocabs legitimately hold non-UTF-8 pieces, so strict
  conversion would wrongly fail on real vocabs. (Deliberately
  different from strict `decode`: display vs data.)
* `token_score` passes the `f32` through (0.0 when the model stores
  none — both fixtures).
* `token_attr` is a **bitmask**, not an enum — `UNKNOWN=1,
  UNUSED=2, NORMAL=4, CONTROL=8, …, SINGLE_WORD=512`. `TokenAttr`
  is a `u32` newtype with `bits()/contains()/is_empty()` and
  constants referencing the sys values (single source of truth).
  Fixture truth: `unk=0x1, bos/eos=0x8, words=0x4, bytes=0x20`.
* `is_eog` is bounds-safe upstream (set-membership) but validates
  anyway: one uniform rule, no exceptions, no per-function hazard
  matrix for callers to remember.
* `SpecialTokens`: `None` ⟺ `LLAMA_TOKEN_NULL`. Notable upstream
  behavior pinned by tests: tiny-tok auto-detects `nl = Some(23)`
  from the `<0x0A>` byte piece; tiny-llama reports `nl: None`
  (there is no Universal default-13 — the audit-table default was
  loader-dependent).
* All specials/flags are snapshotted at `open()`: post-load
  vocabulary metadata is immutable, so caching is sound.

---

## 11. Fixtures

`tiny-llama.gguf` (Phase-0, `FORGE_TEST_MODEL`): **unchanged**.
Regeneration after the script edit is byte-identical —
`sha256 7cbfe2a5…5628d` before and after. Tokenizer-relevant
properties: SPM, `add_bos=true`, `add_eos=false` (flag simply
absent), specials bos/eos-only, and **no byte pieces** — any
non-empty `encode` throws `std::out_of_range` from `byte_to_token`
(observed SIGABRT transcript in §16.1). It therefore covers vocab
getters + empty encodes only, by design.

`tiny-tok.gguf` (new, `FORGE_TEST_MODEL_TOK`): same 1-layer LLAMA
recipe, vocab **269** = `<unk> <s> </s>` (UNKNOWN/CONTROL/CONTROL)
+ `tok0`–`tok9` (NORMAL) + `<0x00>`–`<0xFF>` (BYTE, **uppercase**
hex — `byte_to_token` builds `"<0x%02X>"`), `add_bos` + `add_eos`
both explicitly true, `n_params = 4968`, `sha256 221ba2bb…8357`.
Generation (documented in `docs/NATIVE.md`, new "Tokenizer fixture"
section):

```sh
PYTHONPATH="$FORGE_LLAMA_DIR/py" python3 scripts/make-tiny-gguf.py --tok \
    "$FORGE_LLAMA_DIR/models/tiny-tok.gguf"
```

The `--tok` mode shares no code path with `main()` (duplicated
tensor block, own RNG stream): a deliberate ~70-line duplication so
the Phase-0 fixture cannot regress. Deterministic (seed 8).

---

## 12. Test inventory (103 vs 81)

Baseline (Phase-0 §19): 54 lib + 9 `cpu_decode` + 9 `ggml_smoke` +
5 `quant_contract` + 4 `forge-sys` = **81**, all still passing
untouched. New: **+22 = 103**.

Unit (8, in `tokenizer.rs` — pure helpers, no fixture needed):

| Test | Covers |
|---|---|
| `vocab_type_from_llama_ids` | 0→`None`, 1–7 variants, 99→`Unknown` |
| `token_attr_flags_match_upstream_bits` | all 11 bits + `contains`/`is_empty` |
| `capacity_step_maps_the_native_protocol` | `Ready`, `Grow`, `INT32_MIN+1 → Grow(MAX)`, `INT32_MIN → Err` |
| `checked_token_id_enforces_vocab_bounds` | 0/31 ok, 32/`u32::MAX` err with id named |
| `special_from_native_maps_null_and_rejects_garbage` | NULL→`None`, valid→`Some`, −2/32 err |
| `add_special_guard_covers_every_vocab_type` | full §7 table incl. WPM unconditional + `Unknown → unsupported` |
| `bytes_to_text_rejects_invalid_utf8` | ok/empty/`0xFF` |
| `option_defaults_follow_upstream_common` | both `Default` impls |

Integration (14, `tests/tokenizer.rs`, per-fixture SKIP):

| Test | Fixture | Pins |
|---|---|---|
| `tok_fixture_metadata_is_correct` | tok | 269/Spm/flags/specials incl. `nl=23`, `n_params=4968`, `vocab_size` cross-check, `Debug` |
| `tiny_fixture_metadata_is_correct` | tiny | 32/Spm/`adds_eos=false`/`nl=None` (flag difference vs tok) |
| `token_text_score_attr_cover_all_classes` | tok | 8 ids across UNKNOWN/CONTROL/NORMAL/BYTE |
| `eog_and_control_matrix` | tok | eog ⟺ id 2; control ⟺ ids 1,2 |
| `out_of_range_ids_are_clean_errors` | tok | 5 getters × OOB → `tokenizer error`, id named, no abort |
| `encode_empty_matches_spm_convention` | tok | `"" → []`, `"" + special → [1, 2]` |
| `encode_is_byte_exact_and_deterministic` | tok | `"tok5" → [239,163,142,129,124,120,66]` (`13+byte` math asserted), repeat-equal, all `< n_vocab` |
| `encode_add_special_wraps_with_bos_eos` | tok | `[1] + plain + [2]` |
| `parse_special_controls_special_spellings` | tok | plaintext 6 bytes vs parsed `[1]` |
| `encode_maps_inner_spaces_to_word_boundaries` | tok | `"a b"` → `▁ a ▁ b` byte ids |
| `decode_renders_and_skips_specials_by_flag` | tok | default/strip/unparse/empty 5-way table |
| `decode_rejects_bad_ids_and_bad_utf8` | tok | OOB ids, lone-`0xE2` UTF-8 error, full `▁` ok |
| `spm_round_trip_maps_spaces_to_word_boundaries` | tok | 5 inputs incl. punctuation + special/strip round-trip |
| `tiny_encode_empty_only` | tiny | `"" → []`, `"" + special → [1]`; non-empty deliberately untested (cites §7/§16.1) |

Required-coverage map (Phase-0 brief): round-trip fidelity →
`spm_round_trip…`; special-token values → both metadata tests;
buffer growth → `capacity_step…` unit (protocol arms) **plus** the
real sizing→fill two-call sequence executed by every non-empty
integration encode/decode; UTF-8 errors → `decode_rejects…` +
`bytes_to_text…`.

---

## 13. Validation results

Environment: 2 cores, no GPU, rustc/cargo **1.99.0**, cmake 4.1.2,
gcc 14.2.0, upstream re-verified `7fe450e19305b828c199d602c23a8337aaa1f03b`,
ggml libs 0.25.1, gguf-py 0.19.0. (Toolchain unpinned by design —
inherited from Phase-0, not changed.)

| Command | Result |
|---|---|
| `cargo fmt --all -- --check` | clean |
| `cargo check --workspace --all-targets` | clean, no warnings |
| `cargo test --workspace` (`FORGE_TEST_MODEL` + `FORGE_TEST_MODEL_TOK`) | **103 passed, 0 failed**: 62 lib, 9 `cpu_decode`, 9 `ggml_smoke`, 5 `quant_contract`, 14 `tokenizer`, 4 `forge-sys` |
| `cargo test --workspace --release` (same fixtures) | 103 passed, 0 failed (identical split) |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo test --workspace` (both fixtures unset) | green — fixture suites SKIP, nothing fails |
| `cargo doc` new warnings | 0 (3 pre-existing warnings in `lib.rs`/`context.rs`/`quant.rs` left untouched) |
| Repo cleanliness (`find` for `target/`, `*.gguf`, `*.so`, `__pycache__`, `.cargo`) | no hits under `/home/user/forgeCORE` |
| Fixture integrity | tiny-llama regen byte-identical (§11) |

Validation ran on the final code; the only post-test change was
doc-comment link fixes, re-verified by `fmt --check` + `clippy -D`.

---

## 14. Error taxonomy

One new domain, `Error::tokenizer` (`"tokenizer error: …"`), for
everything `tokenizer.rs` reports; two `open()` failures stay in
`Error::model` (corrupt/missing model data, matching the
`vocab_size()` precedent), and unknown-type `add_special` refusal is
`Error::unsupported`. No panics on any path (`try_from` + explicit
errors throughout, including the by-construction-impossible ones).

| Failure | Domain | Example message |
|---|---|---|
| NULL vocab at `open()` | model | `model error: model has no vocabulary` |
| NONE vocab type at `open()` | tokenizer | `tokenizer error: model has no tokenizer (vocabulary type NONE)` |
| negative `n_tokens` | model | via `u32_from_upstream` |
| corrupt special id at `open()` | tokenizer | `… invalid special token BOS: -2 (vocab size 269)` |
| OOB token id (any entry) | tokenizer | `… token id 269 out of range (vocab size 269)` |
| `add_special` requirements unmet | tokenizer | `… cannot add special tokens: model configures BOS but defines no BOS token` |
| `add_special` on unknown type | unsupported | `unsupported: add_special for unknown vocab type 99` |
| text / ids / output exceed `i32` | tokenizer | `… text too long: … bytes`, `… too many tokens: …` |
| native `INT32_MIN` | tokenizer | `… integer overflow (output exceeds i32::MAX)` |
| decoded bytes not UTF-8 | tokenizer | `… decoded text is not valid UTF-8 (1 bytes)` |
| native returns invalid id | tokenizer | `… native tokenizer returned invalid id -1` (defensive; by construction unreachable) |

---

## 15. Files changed

| File | Change |
|---|---|
| `crates/forge-core/src/tokenizer.rs` | **new**, 732 lines: API + guards + 8 unit tests |
| `crates/forge-core/tests/tokenizer.rs` | **new**, 380 lines: 14 integration tests |
| `crates/forge-sys/src/lib.rs` | +75: 2 const modules, `LLAMA_TOKEN_NULL`, 19 decls + contract comment |
| `crates/forge-core/src/error.rs` | +6: `Error::tokenizer` + test line |
| `crates/forge-core/src/lib.rs` | +3: `pub mod tokenizer`, re-exports, crate-doc line |
| `crates/forge-core/src/model.rs` | 1 line: `u32_from_upstream` → `pub(crate)` for reuse |
| `scripts/make-tiny-gguf.py` | +70: `--tok` fixture mode (default path untouched) |
| `docs/NATIVE.md` | +16: "Tokenizer fixture" section |
| `docs/handoff/PHASE_1_ENGINEERING_REPORT.md` | this report |

Not touched: RAMforge tree, `tiny-llama.gguf` bytes, all Phase-0
tests, `setup-native.sh`, `env.sh`, past handoff docs.

---

## 16. Known limitations and residual risks

1. **Malformed-vocab throw (residual, unguardable).** Upstream
   `byte_to_token` (SPM/UGM) and the BPE/WPM/PLAMO2 equivalents
   `.at()`-throw when a vocab lacks byte pieces; Rust cannot catch
   a foreign exception. Observed evidence (tiny-llama,
   `encode("tok5")`):
   `fatal runtime error: Rust cannot catch foreign exceptions, aborting`
   → SIGABRT. ForgeCore therefore supports well-formed vocabs only
   — the same assumption upstream's own tools make. No Rust-side
   probe can detect a missing byte piece short of trial-encoding
   all 256 bytes (rejected: slow, fragile, still racy against
   nothing — it would be deterministic, but it bakes fixture-isms
   into `open()`; reconsider if a real-world malformed vocab
   appears).
2. **NONE-vocab `open()` rejection is wired but fixture-untested**
   (no NONE model exists): the `from_llama(0) → None` mapping is
   unit-tested; the `open()` wiring is reviewed, not executed.
3. **Strict `decode` UTF-8** means streaming decoders (token-at-a-
   time prefixes) need a future byte API; `decode` errs honestly
   until then.
4. **Unknown future vocab types** read fine (getters are
   type-agnostic) but `add_special` is refused (`unsupported`).
5. **`token_text` is lossy** (U+FFFD) by design (§10) — documented
   on the method.
6. **Exact-id encode tests pin upstream's all-zero-score Viterbi
   tie-break.** If upstream re-breaks ties, `encode("tok5")` ids
   change while staying valid — the test would fail correctly
   (behavior changed), and the fix is to re-pin, not to loosen.
7. **Only SPM is fixture-exercised** (both fixtures). BPE/WPM/UGM/
   RWKV/PLAMO2/TEST guard arms are covered by `check_add_special`
   unit tests over the audited table, not by live models. A tiny
   BPE fixture is the highest-value follow-up (§17).
8. Phase-0 §20 items carry over unchanged where still open
   (notably 8: confirm RAMforge's tokenizer needs before freezing
   further API — P2's job; 4/5: dead placeholder API, still
   flagged-not-acted per scope).

---

## 17. Recommended next phase

**P2 — Sampling** (per Phase-0 §18), now unblocked on a real
tokenizer: grammar-free greedy/temperature/top-k/top-p over
`Context` logits, with the same audit-then-guard discipline (the
sampler C API has its own seed/state contracts to verify).

Tokenizer-adjacent follow-ups, in value order:

1. `decode_bytes(&[TokenId]) -> Result<Vec<u8>>` for streaming
   decoders (§16.3) — thin wrapper over the existing fill loop.
2. A tiny BPE fixture (merges + `byte_encode`) to live-cover a
   second `check_add_special` arm and byte-map differences (§16.7).
3. Revisit the 256-byte trial-encode probe for `open()` only if a
   real malformed vocab surfaces (§16.1); until then the honest
   error boundary is "well-formed vocabs".
4. Confirm RAMforge's tokenizer/sampling contract (Phase-0 §20.8)
   before any API freeze — P2 entry criterion.

Acceptance bar for P2: same as this phase — audited FFI deltas,
pure-function unit tests for every guard, fixture-backed
integration tests, 103+X total with the arithmetic shown, and the
§13 gates green in debug **and** release.
