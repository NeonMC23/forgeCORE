# GGUF inspection ownership

**Decision date:** 2026-10-24  
**Status:** current, deliberate split of responsibilities.

## Decision

There are two GGUF readers with different jobs:

1. **forgeCORE `forge_core::gguf`** owns payload-lazy **structural
   inspection** of a GGUF file: magic/version, metadata values, tensor
   descriptors, checked absolute file ranges, alignment, rank/name limits, and
   quantization geometry. Its contract follows the pinned llama.cpp/ggml GGUF
   reader: versions 2 and 3 are accepted; version 1 and newer versions are
   rejected. It never copies or decodes tensor payloads and never calls native
   code.
2. **RAMforge `ramforge_core::gguf`** remains the runtime's richer
   metadata/tokenizer parser. RAMforge also accepts GGUF version 1 and exposes
   model-level metadata used by planning, tokenization, and its file-backed
   data source.

forgeCORE is the lower-level, native-contract-aligned inspector. RAMforge is
the model/runtime interpretation layer. The two readers are not interchangeable
and neither silently delegates to the other.

## Why not consolidate now

- RAMforge's parser supports GGUF version 1 and exposes metadata shapes and
  serde-friendly model information that its planner/tokenizer consume.
- forgeCORE's inspector is deliberately stricter (v2/v3) and payload-lazy; it
  has no RAMforge tokenizer/model layer.
- A migration would need feature parity, tokenizer fixture parity, and a
  validated data-source path. Doing it as a drive-by rewrite would risk the
  legacy runtime and the ForgeCore adapter at the same time.
- No RAMforge code currently consumes `forge_core::gguf`, so there is no
  behavioural conflict today.

## Rules that keep the split safe

- RAMforge must not depend on `forge-sys` or call native GGUF/llama symbols
  directly. Any future adoption of forgeCORE inspection must use the public
  `forge_core::gguf` API.
- The two readers must not be mixed silently for one operation. A parser
  version/feature mismatch is an error, not a fallback.
- Tests on both sides remain authoritative for their own contract. Shared
  fixture parity tests are the prerequisite for any consolidation.
- If one reader becomes the sole source of truth, the other is removed or
  archived in the same reviewed change, with its tests preserved.

## Follow-up

1. **Shared-fixture contract implemented.** Both repositories now have a
   `tiny_fixture_matches_shared_structural_contract` test that loads the same
   `FORGE_TEST_MODEL` file and asserts the same version, alignment,
   architecture, tensor names, payload lengths, and total payload bytes. The
   crates are not linked, so this is a contract test rather than an in-process
   comparison; running both suites against one file catches descriptor
   divergence.
2. Decide whether RAMforge should adopt `forge_core::gguf` for descriptor
   validation while retaining its model-level metadata layer.
3. Consolidate only after parity is demonstrated and the legacy execution path
   remains validated.
