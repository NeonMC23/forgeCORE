# Archived forge-core legacy implementations

This directory preserves historical, inactive `forge-core` implementations
from before the ggml pivot. They are source material only and are not
supported production APIs.

- `attention.rs` — legacy scalar single-token attention implementation.
- `kv.rs` — legacy pure-Rust per-layer KV-cache implementation.
- `tests/attention_gqa.rs` — integration tests for the archived attention API.
- `tests/kv_cache.rs` — integration tests for the archived KV-cache API.
- `tests/tiny_model.rs` — integration tests for the archived pure-Rust model,
  checkpoint, shape, and KV-cache APIs.

These files are outside the active Rust module graph and outside Cargo's test
target graph. Their source is preserved as historical evidence, but Cargo does
not compile or run it. The current equivalents are the native `Context`,
`Memory`, `State`, and `Batch` APIs and their integration tests
(`context_batch.rs`, `kv_state.rs`, `cpu_decode.rs`).

The files were archived rather than deleted because they document the
pre-ggml numerical contract. They must not be restored, ported, or adapted
without a separately approved task.
