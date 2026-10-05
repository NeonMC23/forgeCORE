//! Phase-1 CPU inference path: Model → Context → Batch → decode → logits.
//!
//! Every test below needs a real `.gguf` via `FORGE_TEST_MODEL` (see
//! `docs/NATIVE.md`) and reports SKIP otherwise. Pure validation logic
//! (batch builder, option defaults) is unit-tested inside the crate.

use forge_core::model::set_log_quiet;
use forge_core::{Batch, BatchBuilder, Context, ContextOptions, Model, ModelOptions};
use std::path::PathBuf;

fn fixture() -> Option<PathBuf> {
    match std::env::var("FORGE_TEST_MODEL") {
        Ok(path) => Some(PathBuf::from(path)),
        Err(_) => {
            println!("SKIP: FORGE_TEST_MODEL not set (see docs/NATIVE.md)");
            None
        }
    }
}

fn load_fixture() -> Option<Model> {
    fixture().map(|path| Model::load(&path).expect("fixture model must load"))
}

fn tiny_batch() -> Batch {
    let mut builder = BatchBuilder::new(32);
    builder.push(1, 0, false).unwrap();
    builder.push(5, 1, false).unwrap();
    builder.push(6, 2, false).unwrap();
    builder.push(7, 3, true).unwrap();
    builder.build().unwrap()
}

// -- A. model ---------------------------------------------------------------

#[test]
fn tiny_model_metadata_is_correct() {
    let Some(model) = load_fixture() else { return };
    assert_eq!(model.n_params(), 1176);
    assert_eq!(model.vocab_size().expect("vocab"), 32);
    let description = model.description().expect("description");
    assert!(
        description.contains("llama"),
        "description names the arch: {description}"
    );
    assert_eq!(model.size_bytes(), 4704);
    assert_eq!(model.n_ctx_train().expect("n_ctx_train"), 64);
    assert_eq!(model.n_embd().expect("n_embd"), 8);
    assert_eq!(model.n_layer().expect("n_layer"), 1);
    assert_eq!(model.n_head().expect("n_head"), 2);
    assert_eq!(model.n_head_kv().expect("n_head_kv"), 2);
}

#[test]
fn tiny_model_load_with_check_tensors() {
    let Some(path) = fixture() else { return };
    let mut options = ModelOptions::default();
    options.check_tensors = true;
    let model = Model::load_with_options(&path, &options).expect("checked load");
    assert_eq!(model.n_params(), 1176);
}

// -- B. context -------------------------------------------------------------

#[test]
fn tiny_context_opens_with_defaults_and_explicit() {
    let Some(model) = load_fixture() else { return };
    let context = Context::open(&model, &ContextOptions::default()).expect("default context");
    // n_ctx resolves to n_ctx_train (64), then upstream pads the
    // effective length up to a multiple of 256: 256, not 64.
    assert_eq!(context.n_ctx(), 256, "n_ctx is the effective length");
    assert_eq!(context.n_vocab(), 32);

    let mut explicit = ContextOptions::default();
    explicit.n_ctx = 32;
    let context = Context::open(&model, &explicit).expect("explicit context");
    assert_eq!(context.n_ctx(), 256, "explicit n_ctx is padded too");
}

#[test]
fn context_rejects_zero_threads() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.n_threads = 0;
    let error = Context::open(&model, &options).expect_err("threads=0 must fail");
    assert!(
        error.to_string().contains("n_threads"),
        "error names the bad field: {error}"
    );
}

#[test]
fn context_drop_and_reopen_is_clean() {
    let Some(model) = load_fixture() else { return };
    {
        let _first = Context::open(&model, &ContextOptions::default()).expect("first");
    }
    let mut second = Context::open(&model, &ContextOptions::default()).expect("second");
    second.decode(&tiny_batch()).expect("decode after reopen");
}

// -- D. decode --------------------------------------------------------------

#[test]
fn tiny_decode_succeeds() {
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    context.decode(&tiny_batch()).expect("decode");
}

#[test]
fn tiny_decode_propagates_native_kv_error() {
    let Some(model) = load_fixture() else { return };
    // n_ctx=1: the second single-token decode re-uses position 0, which
    // upstream rejects (positions must stay consecutive). All inputs are
    // valid, so the failure must arrive as a native decode error.
    let mut options = ContextOptions::default();
    options.n_ctx = 1;
    let mut context = Context::open(&model, &options).expect("tiny context");
    let mut first = BatchBuilder::new(32);
    first.push(1, 0, true).unwrap();
    context.decode(&first.build().unwrap()).expect("first");
    let mut second = BatchBuilder::new(32);
    second.push(2, 0, true).unwrap();
    set_log_quiet(true);
    let error = second
        .build()
        .map_err(|error| error.to_string())
        .and_then(|batch| context.decode(&batch).map_err(|error| error.to_string()));
    set_log_quiet(false);
    let message = error.expect_err("KV exhaustion must fail");
    assert!(
        message.starts_with("decode error"),
        "native failure maps to decode error: {message}"
    );
}

// -- E. logits --------------------------------------------------------------

#[test]
fn tiny_logits_after_decode() {
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    context.decode(&tiny_batch()).expect("decode");
    // Logits flag was set on batch index 3 only.
    let logits = context.logits(3).expect("logits for index 3");
    assert_eq!(logits.n_vocab(), 32);
    assert_eq!(logits.values().len(), 32);
    assert!(
        logits.values().iter().all(|value| value.is_finite()),
        "fixture logits must be finite"
    );
    assert!(
        logits.values().iter().any(|value| *value != 0.0),
        "fixture logits must be non-trivial"
    );
    // Single-threaded CPU decode of a fixed batch is deterministic.
    let mut again = Context::open(&model, &ContextOptions::default()).expect("context");
    again.decode(&tiny_batch()).expect("decode");
    let repeat = again.logits(3).expect("logits");
    assert_eq!(logits.values(), repeat.values());
}

#[test]
fn tiny_logits_invalid_index_is_error() {
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    context.decode(&tiny_batch()).expect("decode");
    set_log_quiet(true);
    let beyond = context.logits(99).expect_err("index 99 must fail");
    let unflagged = context.logits(0).expect_err("unflagged index must fail");
    set_log_quiet(false);
    assert!(
        beyond.to_string().starts_with("logits error"),
        "NULL maps to logits error: {beyond}"
    );
    assert!(
        unflagged.to_string().starts_with("logits error"),
        "unflagged index maps to logits error: {unflagged}"
    );
}
