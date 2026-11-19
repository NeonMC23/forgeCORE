# Mapping identity and invalidation

This document records the native lifecycle assumptions behind
`TensorMappingInfo`, `PageSample`, and the adaptive dispatch cache.

## Native lifecycle audit

- `llama_model_loader::init_mappings` creates one `llama_mmap` per model file
  during model construction.
- Mappings are moved into `llama_model::pimpl->mappings` while the model is
  still being constructed.
- `unmap_fragment` is used by the loader and quantization path while tensors
  are being loaded or converted. It is not called after a model has been
  returned by the public load API.
- The model mapping vector and each mapping's address range are therefore
  immutable for the lifetime of a loaded `llama_model` in the pinned native
  revision.
- A context borrows a model. Dropping or recreating contexts cannot change the
  model's mapping vector.
- Replacing a path on disk does not change an existing mapping: the mapping
  still refers to the inode that was opened when the model was loaded.

## Identity model

- `mapping_id` is the index of the file mapping in the model's immutable
  mapping vector.
- `mapping_generation` is the native model-instance id, allocated from a
  process-wide monotonic counter. It changes whenever a model instance is
  created, including reloads of the same file.
- Tensor metadata adds a page-aligned `page_start`/`page_end`, the tensor
  offset/size, and the mapping file path/size.
- Dispatch caches are context-scoped and keyed by
  `(mapping_id, mapping_generation, page_start, page_end)`.
- A cache entry cannot outlive its context. A context cannot outlive its
  model. A stale model/mapping generation is therefore not reachable through
  the safe API; the native sampling entry point additionally rejects a
  mismatched `expected_generation` with `ESTALE`.

## Invalidation behavior

- Context recreation starts with an empty cache.
- Model reload produces a new `mapping_generation`; identities from the old
  model are not reusable.
- Capacity eviction is deterministic and counted.
- Freshness is an event-sequence window, not a wall-clock guarantee.
- The cache never stores a non-complete observation as reusable.

## Verified behavior

The mapping test suite verifies:

- File-backed tensors report a nonzero mapping generation.
- All file-backed mappings in one model share that generation.
- Opening and dropping contexts leaves mapping metadata unchanged.
- A separately loaded model receives a different generation while reusing
  local mapping indices.
- Re-reading the first model after another model load still returns its
  original generation and ranges.

## Remaining limitation

The pinned upstream revision does not mutate model mappings after load, so a
separate per-fragment generation is unnecessary. If a future upstream revision
begins mutating mappings after model construction, the generation must be
extended or the mapping metadata must be re-snapshotted. The current safe API
does not expose a post-load mutation hook, and no such mutation is assumed.
