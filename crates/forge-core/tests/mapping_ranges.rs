//! Tensor-scoped mapping metadata and page-release validation.

#![cfg(target_os = "linux")]

use std::path::PathBuf;

use forge_core::{BatchBuilder, Context, ContextOptions, Model, ModelLoadMode, ModelOptions};

fn fixture() -> Option<PathBuf> {
    let path = std::env::var_os("FORGE_TEST_MODEL")?;
    let path = PathBuf::from(path);
    path.is_file().then_some(path)
}

fn options() -> ModelOptions {
    let mut options = ModelOptions::default();
    options.load_mode = ModelLoadMode::Mmap;
    options.use_extra_bufts = Some(false);
    options
}

fn tiny_batch() -> forge_core::Batch {
    let mut builder = BatchBuilder::new(32);
    builder.push(1, 0, false).unwrap();
    builder.push(5, 1, false).unwrap();
    builder.push(6, 2, false).unwrap();
    builder.push(7, 3, true).unwrap();
    builder.build().unwrap()
}

#[test]
fn large_tensor_release_releases_pages_and_keeps_inference() {
    let Some(path) = std::env::var_os("FORGE_PREFETCH_MODEL").map(PathBuf::from) else {
        eprintln!("SKIP: FORGE_PREFETCH_MODEL not set");
        return;
    };
    if !path.is_file() {
        eprintln!("SKIP: FORGE_PREFETCH_MODEL is not a file");
        return;
    }
    let mut model = Model::load_with_options(&path, &options()).expect("real model");
    let mappings = model.tensor_mappings().expect("tensor mappings");
    let selected = mappings
        .iter()
        .enumerate()
        .filter(|(_, mapping)| mapping.file_backed)
        .max_by_key(|(_, mapping)| mapping.size)
        .map(|(index, _)| index as i32)
        .expect("file-backed tensor");
    let prefetched = model
        .prefetch_tensor_pages(&[selected])
        .expect("large tensor prefetch");
    eprintln!("prefetched={prefetched}");
    assert!(prefetched > 0);
    model
        .prefetch_tensor_pages(&[selected])
        .expect("repeated prefetch");
    let result = model
        .release_tensor_pages(&[selected])
        .expect("large tensor release");
    eprintln!("large tensor result={result:?}");
    assert!(result.selected_tensors >= 1);
    assert!(
        result.released_bytes > 0,
        "large tensor should have private pages: {result:?}"
    );
    model
        .release_tensor_pages(&[selected])
        .expect("repeated large tensor release");

    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    context
        .decode(&tiny_batch())
        .expect("decode after large tensor release");
}

#[test]
#[ignore = "requires a Q4_0 real model with FORGE_ANON_MODEL set"]
fn repacked_tensors_are_reported_anonymous() {
    let Some(path) = std::env::var_os("FORGE_ANON_MODEL").map(PathBuf::from) else {
        eprintln!("SKIP: FORGE_ANON_MODEL not set");
        return;
    };
    if !path.is_file() {
        eprintln!("SKIP: FORGE_ANON_MODEL is not a file");
        return;
    }
    let mut options = ModelOptions::default();
    options.load_mode = ModelLoadMode::Mmap;
    options.use_extra_bufts = Some(true);
    let model = Model::load_with_options(&path, &options).expect("repacked model");
    let mappings = model.tensor_mappings().expect("tensor mappings");
    let file_backed = mappings
        .iter()
        .filter(|mapping| mapping.file_backed)
        .count();
    let anonymous = mappings
        .iter()
        .filter(|mapping| !mapping.file_backed)
        .count();
    eprintln!("file_backed={file_backed} anonymous={anonymous}");
    assert!(file_backed > 0);
    assert!(
        anonymous > 0,
        "repacked Q4_0 should expose anonymous tensors"
    );
}

#[test]
fn tensor_mapping_metadata_is_exact_and_release_preserves_inference() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let mut model = Model::load_with_options(&path, &options()).expect("model");
    let mappings = model.tensor_mappings().expect("tensor mappings");
    assert!(!mappings.is_empty(), "model must expose tensor metadata");

    let mut file_backed = Vec::new();
    for (index, mapping) in mappings.iter().enumerate() {
        assert!(!mapping.name.is_empty(), "tensor name must not be empty");
        if mapping.file_backed {
            assert!(mapping.mapping_size > 0);
            assert!(mapping.page_size > 0);
            assert_eq!(mapping.page_start % mapping.page_size, 0);
            assert!(mapping.page_start <= mapping.offset);
            assert!(mapping.page_end >= mapping.offset.saturating_add(mapping.size));
            assert!(
                mapping.offset.saturating_add(mapping.size) <= mapping.mapping_size,
                "tensor range exceeds its mapping: {mapping:?}"
            );
            assert!(mapping.mapping_id.is_some());
            assert!(mapping.mapping_generation.is_some());
            assert!(mapping
                .file_size
                .is_some_and(|size| size == mapping.mapping_size));
            assert!(mapping.file_path.is_some());
            file_backed.push(index);
        } else {
            assert!(mapping.mapping_index.is_none());
        }
    }
    assert!(
        !file_backed.is_empty(),
        "fixture should have file-backed tensors"
    );

    let selected = file_backed[0] as i32;
    let result = model
        .release_tensor_pages(&[selected])
        .expect("tensor release");
    eprintln!(
        "metadata={} file_backed={} result={result:?}",
        mappings.len(),
        file_backed.len()
    );
    assert_eq!(result.selected_tensors, 1);
    assert!(
        result.released_bytes + result.skipped_shared_bytes > 0,
        "selected tensor must have a page interval"
    );

    model
        .release_tensor_pages(&[selected])
        .expect("repeated tensor release");
    assert!(model.release_tensor_pages(&[]).is_err());
    assert!(model.release_tensor_pages(&[-1]).is_err());
    assert!(model.release_tensor_pages(&[i32::MAX]).is_err());

    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    context
        .decode(&tiny_batch())
        .expect("decode after tensor release");
    assert_eq!(model.n_params(), 1176);
}

#[test]
fn tensor_page_sample_is_bounded_and_file_backed() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load_with_options(&path, &options()).expect("model");
    let mappings = model.tensor_mappings().expect("tensor mappings");
    let mapping = mappings
        .iter()
        .max_by_key(|mapping| mapping.size)
        .expect("mapping");
    let sample = model
        .sample_tensor_pages(&mapping.name, 64)
        .expect("sample");
    eprintln!("sample={sample:?}");
    assert!(sample.requested_pages > 0);
    assert!(sample.sampled_pages <= 64);
    assert_eq!(
        sample.resident_pages + sample.nonresident_pages + sample.unknown_pages,
        sample.sampled_pages
    );
    assert!(sample.mapping_generation != 0);
    assert!(matches!(
        sample.status,
        forge_core::PageSampleStatus::Complete | forge_core::PageSampleStatus::Partial
    ));
}

#[test]
fn mapping_identity_is_stable_within_a_model_and_changes_across_instances() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load_with_options(&path, &options()).expect("first model");
    let first = model.tensor_mappings().expect("first mappings");
    let first_file_backed: Vec<_> = first.iter().filter(|mapping| mapping.file_backed).collect();
    assert!(!first_file_backed.is_empty());

    let generation = first_file_backed[0]
        .mapping_generation
        .expect("mapping generation");
    assert!(generation != 0);
    assert!(first_file_backed
        .iter()
        .all(|mapping| mapping.mapping_generation == Some(generation)));
    let identities: Vec<_> = first_file_backed
        .iter()
        .map(|mapping| {
            (
                mapping.name.clone(),
                mapping.mapping_id,
                mapping.page_start,
                mapping.page_end,
            )
        })
        .collect();

    // Opening and dropping contexts must not change model mapping identity.
    for _ in 0..2 {
        let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
        context.decode(&tiny_batch()).expect("decode");
    }
    let second = model.tensor_mappings().expect("second mappings");
    assert_eq!(first, second);

    // A separately loaded model gets a distinct instance generation even
    // though the local mapping indices are reused.
    let other = Model::load_with_options(&path, &options()).expect("second model");
    let third = other.tensor_mappings().expect("third mappings");
    let third_file_backed: Vec<_> = third.iter().filter(|mapping| mapping.file_backed).collect();
    assert!(!third_file_backed.is_empty());
    let other_generation = third_file_backed[0]
        .mapping_generation
        .expect("other generation");
    assert_ne!(generation, other_generation);

    // Re-reading the first model's metadata after another model has loaded
    // still yields the original generation and ranges.
    let fourth = model.tensor_mappings().expect("fourth mappings");
    assert_eq!(first, fourth);
    assert_eq!(
        identities,
        fourth
            .iter()
            .filter(|mapping| mapping.file_backed)
            .map(|mapping| (
                mapping.name.clone(),
                mapping.mapping_id,
                mapping.page_start,
                mapping.page_end
            ))
            .collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL; copies the model to an isolated temporary file"]
fn isolated_model_mapping_transition_is_observed_or_explicitly_not_claimed() {
    use std::os::fd::AsRawFd;

    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let unique = format!(
        "forge-model-residency-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    );
    let copy = std::env::temp_dir().join(unique);
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(copy.clone());
    std::fs::copy(&path, &copy).expect("copy model");
    let model = Model::load_with_options(&copy, &options()).expect("copied model");
    let mappings = model.tensor_mappings().expect("mappings");
    let selected = mappings
        .iter()
        .filter(|mapping| mapping.file_backed)
        .max_by_key(|mapping| mapping.size)
        .map(|mapping| mapping.name.clone())
        .expect("file-backed tensor");
    let before = model
        .sample_tensor_pages(&selected, 64)
        .expect("baseline sample");
    let generation = before.mapping_generation;
    drop(model);
    let file = std::fs::File::open(&copy).expect("open copy");
    // File-scoped advisory request on the isolated copy only, after the
    // model mapping has been released.
    let fadvise = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    let mut no_prefetch = options();
    no_prefetch.load_mode = ModelLoadMode::MmapNoPrefetch;
    let reloaded = Model::load_with_options(&copy, &no_prefetch).expect("reloaded model");
    let after = reloaded
        .sample_tensor_pages(&selected, 64)
        .expect("post-eviction sample");
    assert_ne!(generation, 0);
    assert_ne!(after.mapping_generation, generation);
    assert_eq!(
        before.sampled_pages,
        before.resident_pages
            + before.nonresident_pages
            + before.unknown_pages
            + before.failed_pages
    );
    assert_eq!(
        after.sampled_pages,
        after.resident_pages + after.nonresident_pages + after.unknown_pages + after.failed_pages
    );
    assert!(before.mapping_generation != 0);
    eprintln!(
            "model_mapping before resident={} nonresident={} unknown={} failed={} fadvise_ok={} after resident={} nonresident={} unknown={} failed={}",
            before.resident_pages,
            before.nonresident_pages,
            before.unknown_pages,
            before.failed_pages,
            fadvise == 0,
            after.resident_pages,
            after.nonresident_pages,
            after.unknown_pages,
            after.failed_pages
        );
    // The advisory request is not proof of a physical transition. If the
    // kernel reports nonresident pages, that is an observation, not a
    // performance claim.
    assert!(after.requested_pages > 0);
}
