# Licensing and attribution

## Upstream: llama.cpp / ggml

ForgeCore dynamically links a pinned upstream build (tag `v0.5.0`,
commit `7fe450e19305b828c199d602c23a8337aaa1f03b`, repo
`https://github.com/ggml-org/llama.cpp`). Upstream is MIT-licensed:

```text
MIT License

Copyright (c) 2023-2026 The ggml authors
```

(from the `LICENSE` file at the repository root; the standard MIT
permission/conditions text follows). An additional bundled notice,
`licenses/LICENSE-jsonhpp` (MIT, Copyright (c) 2013-2025 Niels
Lohmann), covers a third-party header used inside upstream's own
sources.

**What this means for ForgeCore:**

- No upstream source is copied into this tree. The `forge-sys`
  declarations are hand-written signatures against the pinned headers,
  and the native tree is built outside the repo (`docs/NATIVE.md`).
- The MIT license permits dynamic linking with no source-disclosure
  duty, but **distributions of ForgeCore (or products embedding it)
  must reproduce the upstream copyright and permission notices** above.
  Keep this file and the exact revision in `docs/PIVOT.md` with any
  distribution.
- If upstream source is ever reused verbatim in this tree (not
  currently the case), its notices must be preserved alongside it.

## This repository (ForgeCore)

ForgeCore's own license is **undecided**: there is currently no
`LICENSE` file at the repository root. Until the owner chooses one,
treat all files under this root as all-rights-reserved to the owner
and do not redistribute. Choosing a license compatible with dynamic
linking to MIT code (e.g. MIT/Apache-2.0, which impose no additional
duties here) is recommended but is the owner's call.
