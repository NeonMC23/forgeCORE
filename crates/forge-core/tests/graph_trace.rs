//! Opt-in graph eval trace tests against the tiny fixture.

use std::path::PathBuf;

use forge_core::{
    BatchBuilder, Context, ContextOptions, ContextTraceOptions, Model, TraceCompletion, TracePhase,
};

fn fixture() -> Option<PathBuf> {
    let path = std::env::var_os("FORGE_TEST_MODEL")?;
    let path = PathBuf::from(path);
    path.is_file().then_some(path)
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
fn trace_is_bounded_and_reports_synchronized_completion() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    let mut options = ContextOptions::default();
    options.trace = Some(ContextTraceOptions { max_events: 4096 });
    let mut context = Context::open(&model, &options).expect("traced context");
    context.decode(&tiny_batch()).expect("decode");
    let summary = context.trace_summary().expect("trace summary");
    eprintln!("{summary:?}");
    assert!(summary.requested_events > 0);
    assert!(summary.completed_events > 0);
    assert!(summary.retained_events > 0);
    let events = context.trace_events();
    assert!(events
        .iter()
        .any(|event| event.phase == TracePhase::Requested));
    assert!(events.iter().any(|event| {
        event.phase == TracePhase::CompletedSynchronized
            && event.completion == TraceCompletion::BackendSynchronized
    }));
    assert!(events
        .iter()
        .any(|event| event.node_name.is_some() || event.op_name.is_some()));

    let mut next = BatchBuilder::new(32);
    next.push(1, 4, true).unwrap();
    context
        .decode(&next.build().unwrap())
        .expect("second decode");
    assert!(context.trace_summary().expect("summary").executions >= 2);

    let mut plain = ContextOptions::default();
    plain.trace = None;
    let plain_context = Context::open(&model, &plain).expect("plain context");
    assert!(plain_context.trace_summary().is_none());
    assert!(plain_context.trace_events().is_empty());
}

#[test]
fn trace_overflow_drops_old_events() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    let mut options = ContextOptions::default();
    options.trace = Some(ContextTraceOptions { max_events: 2 });
    let mut context = Context::open(&model, &options).expect("traced context");
    context.decode(&tiny_batch()).expect("decode");
    let summary = context.trace_summary().expect("summary");
    eprintln!("{summary:?}");
    assert!(summary.retained_events <= 2);
    assert!(summary.dropped_events > 0);
    context.clear_trace();
    assert_eq!(context.trace_summary().expect("summary").retained_events, 0);
}
