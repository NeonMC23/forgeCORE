//! Native execution-quiescence and resource-lifetime contract.
//!
//! These tests prove only the boundary the pinned implementation now
//! enforces: release cannot run while a context has asynchronous graph work
//! in flight. They do not claim intra-decode eviction or future-token safety.

use std::path::PathBuf;

use forge_core::{
    Batch, BatchBuilder, Context, ContextOptions, Model, ModelLoadMode, ModelOptions,
    ResidencyError,
};

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

fn context_options() -> ContextOptions {
    let mut options = ContextOptions::default();
    options.n_ctx = 64;
    options.n_batch = 32;
    options.n_ubatch = 32;
    options.n_threads = 2;
    options
}

fn batch() -> Batch {
    batch_at(0)
}

fn batch_at(position: u32) -> Batch {
    let mut builder = BatchBuilder::new(32);
    builder.push(1, position, false).unwrap();
    builder.push(5, position + 1, false).unwrap();
    builder.push(6, position + 2, false).unwrap();
    builder.push(7, position + 3, true).unwrap();
    builder.build().unwrap()
}

#[test]
fn release_is_busy_until_native_synchronization_then_safe() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let mut model = Model::load_with_options(&path, &options()).expect("model");
    assert_eq!(Model::residency_contract_version(), 1);
    assert_ne!(model.instance_id(), 0);

    let initial = model.lifetime_state().expect("initial lifetime");
    assert_eq!(initial.active_contexts, 0);
    assert_eq!(initial.inflight_contexts, 0);
    assert!(initial.quiescent);

    let mut context = Context::open(&model, &context_options()).expect("context");
    let with_context = model.lifetime_state().expect("context lifetime");
    assert_eq!(with_context.active_contexts, 1);
    assert!(with_context.quiescent);

    let context_state = context.lifetime_state().expect("context id");
    assert_ne!(context_state.context_id, 0);
    assert_eq!(context_state.execution_generation, 0);
    assert!(context_state.synchronized);

    context.decode(&batch()).expect("decode queues work");
    let after_submit = model.lifetime_state().expect("submitted lifetime");
    assert_eq!(after_submit.inflight_contexts, 1);
    assert!(!after_submit.quiescent);
    let context_after = context.lifetime_state().expect("submitted context");
    assert!(context_after.inflight);
    assert!(!context_after.synchronized);
    assert_eq!(context_after.execution_generation, 1);

    let selected = model
        .tensor_mappings()
        .expect("mappings")
        .iter()
        .position(|mapping| mapping.file_backed)
        .expect("file-backed tensor") as i32;
    assert_eq!(
        model.release_tensor_pages(&[selected]).unwrap_err(),
        ResidencyError::Busy
    );
    assert_eq!(
        model.release_mapping_pages().unwrap_err(),
        ResidencyError::Busy
    );

    context.synchronize();
    let after_sync = model.lifetime_state().expect("synchronized lifetime");
    assert_eq!(after_sync.inflight_contexts, 0);
    assert!(after_sync.quiescent);
    let context_synced = context.lifetime_state().expect("synchronized context");
    assert!(!context_synced.inflight);
    assert!(context_synced.synchronized);
    assert_eq!(context_synced.execution_generation, 1);

    let release = model
        .release_tensor_pages(&[selected])
        .expect("release after synchronization");
    assert!(release.selected_tensors == 1);
    assert!(release.released_bytes + release.skipped_shared_bytes > 0);

    // Logical ownership survives page eviction; inference refaults pages.
    context.decode(&batch_at(4)).expect("decode after release");
    context.synchronize();
    assert_eq!(
        model
            .lifetime_state()
            .expect("post-release")
            .active_contexts,
        1
    );
}

#[test]
fn two_contexts_must_both_be_synchronized_before_release() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let mut model = Model::load_with_options(&path, &options()).expect("model");
    let selected = model
        .tensor_mappings()
        .expect("mappings")
        .iter()
        .position(|mapping| mapping.file_backed)
        .expect("file-backed tensor") as i32;
    let mut first = Context::open(&model, &context_options()).expect("first");
    let mut second = Context::open(&model, &context_options()).expect("second");
    assert_eq!(model.lifetime_state().expect("state").active_contexts, 2);
    assert_ne!(
        first.lifetime_state().expect("first id").context_id,
        second.lifetime_state().expect("second id").context_id
    );

    first.decode(&batch()).expect("first decode");
    second.decode(&batch()).expect("second decode");
    assert_eq!(
        model.release_tensor_pages(&[selected]).unwrap_err(),
        ResidencyError::Busy
    );
    first.synchronize();
    assert_eq!(
        model
            .release_tensor_pages(&[selected])
            .expect_err("second context is still inflight"),
        ResidencyError::Busy
    );
    second.synchronize();
    model
        .release_tensor_pages(&[selected])
        .expect("both contexts synchronized");
}

#[test]
fn context_drop_synchronizes_and_releases_native_count() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let mut model = Model::load_with_options(&path, &options()).expect("model");
    let selected = model
        .tensor_mappings()
        .expect("mappings")
        .iter()
        .position(|mapping| mapping.file_backed)
        .expect("file-backed tensor") as i32;
    let mut context = Context::open(&model, &context_options()).expect("context");
    context.decode(&batch()).expect("decode");
    drop(context);
    let after_drop = model.lifetime_state().expect("state");
    assert_eq!(after_drop.active_contexts, 0);
    assert_eq!(after_drop.inflight_contexts, 0);
    assert!(after_drop.quiescent);
    model
        .release_tensor_pages(&[selected])
        .expect("release after context drop");
}

#[test]
fn model_instance_identity_is_process_local_and_distinct() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let first = Model::load_with_options(&path, &options()).expect("first model");
    let second = Model::load_with_options(&path, &options()).expect("second model");
    assert_ne!(first.instance_id(), 0);
    assert_ne!(second.instance_id(), 0);
    assert_ne!(first.instance_id(), second.instance_id());
}
