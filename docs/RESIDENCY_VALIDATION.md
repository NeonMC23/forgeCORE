# Controlled residency validation

`forge_core::residency_experiment` is a reusable, file-scoped validation
harness for page-residency observations. It exists to check the operating
system behavior that the page-sampling APIs rely on, not to benchmark
inference.

## Method

For each configured cycle the harness:

1. Creates a unique temporary file and extends it to a page-aligned size.
2. Maps it with `MAP_SHARED` and writes one byte in every page.
3. Synchronizes dirty pages with `msync(MS_SYNC)`.
4. Samples page residency with `mincore()`.
5. Unmaps the file.
6. Requests a file-scoped cache drop with
   `posix_fadvise(POSIX_FADV_DONTNEED)`.
7. Re-maps the same file and samples residency again.
8. Touches every page again and samples a third time.

The temporary file and mapping are released before the report is returned.
The harness never drops system-wide page cache, never applies memory pressure,
and never targets a file it does not own.

## Interpretation

- `after_touch` reports the pages the experiment touched.
- `after_eviction` reports what `mincore()` observed after the advisory
  file-scoped request.
- `after_retouch` reports the pages after the experiment touched them again.
- `transition_observed` is true only when every requested page was observed
  resident before the request and nonresident afterwards.
- If the kernel keeps the pages resident, the report says so. No transition is
  inferred from the request itself.

The report distinguishes preparation (`controlled-file-mapping`) from
production model mappings. It does not claim that a page was physically read,
that a page fault occurred, that I/O happened, or that a kernel accessed a
page. Residency is not availability, and eviction is not proof of release.

## Storage caveat

`posix_fadvise(DONTNEED)` is advisory and filesystem-specific. On tmpfs it
generally cannot make the file's pages nonresident; the report includes an
explicit `Tmpfs` storage marker and a note. For transition validation, supply a
disk-backed directory:

```rust
let config = ResidencyExperimentConfig {
    pages: 16,
    cycles: 2,
    directory: Some("/var/tmp/forge-residency".into()),
};
let report = run_file_residency_experiment(config)?;
```

`FORGE_RESIDENCY_EXPERIMENT_DIR` can set the default directory without code
changes.

## Verified behavior

- Touched file-backed pages are reported resident.
- A disk-backed file-scoped `madvise`/`fadvise` sequence can produce an
  observed resident-to-nonresident transition; this was verified on the
  development machine's btrfs filesystem.
- Re-touching reports resident pages again.
- Repeated cycles re-observe instead of reusing earlier values.
- Concurrent experiments use distinct files and mappings.
- Invalid or missing mappings return an observation error, not a false
  nonresident count.
- tmpfs remains explicitly non-transitional where the kernel cannot satisfy
  the eviction request.

## Limitations

- Linux only. Other platforms report `Unsupported`.
- The behavior is filesystem- and kernel-dependent.
- The harness validates the OS residency primitive, not the llama.cpp mapping
  metadata or inference kernels.
- Mapping disappearance is not interpreted as nonresidency; callers must not
  sample a released mapping.
