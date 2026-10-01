# Licensing and attribution

Three separate things live under (or next to) this repository, with
three separate statuses:

1. **ForgeCore's own source code and documentation** — MIT licensed
   (see below).
2. **Upstream llama.cpp / ggml libraries** — under their own MIT
   license; dynamically linked, never vendored (see below).
3. **Test GGUF fixture** — no `.gguf` binary is committed in this
   repository; only the generator script, which is ForgeCore-owned and
   therefore MIT licensed like the rest of section 1 (see below).

## 1. This repository (ForgeCore) — MIT

ForgeCore's own Rust source code, documentation, scripts, and
repository-owned tooling are licensed under the MIT License. The
`LICENSE` file at the repository root is the authoritative license
for all ForgeCore-owned material, and each ForgeCore-owned Cargo
manifest declares `license = "MIT"`.

This covers all Rust sources (including the `reference/` validation
oracles and the hand-written `forge-sys` declarations), all
documentation, all scripts, and the GGUF generator script
(`scripts/make-tiny-gguf.py`). Provenance comments in
`reference/quant.rs` cite the upstream files the decoders were
validated against; that code is original Rust written in this
repository and contains no copied upstream source text.

## 2. Upstream llama.cpp / ggml — their own MIT license, not vendored

ForgeCore dynamically links a pinned upstream build:

- repo: `https://github.com/ggml-org/llama.cpp`
- tag `v0.5.0`, commit `7fe450e19305b828c199d602c23a8337aaa1f03b`
  (same pin as `scripts/setup-native.sh` and `docs/PIVOT.md`)

The upstream native sources are **not** vendored or copied into this
repository: they are cloned and built externally under
`$FORGE_LLAMA_DIR` (see `docs/NATIVE.md`), and only the resulting
shared libraries (`libggml`, `libggml-base`, `libggml-cpu`, `libllama`)
are linked at build time.

Upstream's root `LICENSE` file reads (verified against the pinned
checkout):

```text
MIT License

Copyright (c) 2023-2026 The ggml authors
```

followed by the standard MIT permission/conditions text. A further
bundled notice, upstream `licenses/LICENSE-jsonhpp`, reads:

```text
MIT License

Copyright (c) 2013-2025 Niels Lohmann
```

llama.cpp / ggml remain separate upstream projects: ForgeCore's MIT
license applies only to ForgeCore-owned material and does not
relicense, replace, or alter the upstream projects' notices or
copyright. The upstream MIT license text conditions its permission on
including the copyright notice and permission notice with copies of
the software; any distribution that includes the upstream libraries
must therefore reproduce the notices above. If upstream source text
is ever copied into this tree (not currently the case), its notices
must be kept alongside it.

## 3. Test GGUF fixture — no binary in the repo

No `*.gguf` file is committed anywhere in this repository (verified by
filename search). The repository contains only the deterministic
generator, `scripts/make-tiny-gguf.py`, which is ForgeCore-owned and
therefore covered by the repository's MIT license like all other
ForgeCore-owned material.

Generated fixtures are written outside the repository (to
`$FORGE_LLAMA_DIR/models/`, see `docs/NATIVE.md`). A generated fixture
contains only synthetic content — seeded random F32 weights and made-up
token strings — and no upstream code or data. If a fixture binary is
ever committed to the repository, this section must be revisited.
