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
