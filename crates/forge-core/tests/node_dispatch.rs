//! CPU node-dispatch observer tests against the tiny fixture.

use std::path::PathBuf;

use forge_core::{
    BatchBuilder, Context, ContextOptions, Model, NodeDispatchOptions, MAX_NODE_DISPATCH_EVENTS,
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
fn dispatch_observer_is_opt_in_and_drains_to_none() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    let plain = Context::open(&model, &ContextOptions::default()).expect("plain context");
    assert!(plain.dispatch_summary().is_none());
    assert!(plain.dispatch_events().is_empty());

    let mut context = Context::open(&model, &ContextOptions::default()).expect("traced context");
    context
        .set_dispatch_observer(Some(NodeDispatchOptions { max_events: 4096 }))
        .expect("install observer");
    assert!(context.dispatch_observer_active());
    context.decode(&tiny_batch()).expect("decode");
    let summary = context.dispatch_summary().expect("summary");
    assert!(summary.executions >= 1);
    assert!(summary.events > 0);
    assert_eq!(summary.dropped_events, 0);
    assert!(summary.retained_events > 0);
    let events = context.dispatch_events();
    assert_eq!(events.len(), summary.retained_events);
    assert!(events
        .iter()
        .all(|event| event.backend == forge_core::NodeDispatchBackend::Cpu));
    assert!(events.iter().all(|event| event.node_count > 0));
    assert!(events
        .iter()
        .all(|event| event.node_index < event.node_count));
    assert!(events.iter().any(|event| event.node_name.is_some()
        || event.op_name.is_some()
        || !event.source_names.is_empty()));
    assert!(events
        .windows(2)
        .all(|pair| pair[0].event_sequence < pair[1].event_sequence));

    context
        .set_dispatch_observer(None)
        .expect("remove observer");
    assert!(!context.dispatch_observer_active());
    assert!(context.dispatch_summary().is_none());
    assert!(context.dispatch_events().is_empty());
    context
        .decode(&tiny_batch())
        .expect("decode without observer");
}

#[test]
fn dispatch_is_exactly_once_with_multiple_workers() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    let mut options = ContextOptions::default();
    options.n_threads = 4;
    options.n_threads_batch = 4;
    let mut context = Context::open(&model, &options).expect("four-thread context");
    context
        .set_dispatch_observer(Some(NodeDispatchOptions::default()))
        .expect("observer");
    context.decode(&tiny_batch()).expect("decode");
    let events = context.dispatch_events();
    let execution = events.first().expect("dispatch events").execution_sequence;
    let same_execution: Vec<_> = events
        .iter()
        .filter(|event| event.execution_sequence == execution)
        .collect();
    let node_count = same_execution
        .iter()
        .map(|event| event.node_count)
        .max()
        .expect("node count");
    assert_eq!(same_execution.len(), node_count as usize);
    let mut indices: Vec<u32> = same_execution
        .iter()
        .map(|event| event.node_index)
        .collect();
    indices.sort_unstable();
    assert_eq!(indices, (0..node_count).collect::<Vec<_>>());
}

#[test]
fn dispatch_events_are_bounded_and_overflow_is_reported() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    context
        .set_dispatch_observer(Some(NodeDispatchOptions { max_events: 2 }))
        .expect("observer");
    context.decode(&tiny_batch()).expect("decode");
    let summary = context.dispatch_summary().expect("summary");
    assert!(summary.events > 2);
    assert_eq!(summary.retained_events, 2);
    assert!(summary.dropped_events > 0);
    assert_eq!(context.dispatch_events().len(), 2);
}

#[test]
fn dispatch_identity_is_per_context_and_generation_aware() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    let mut first = Context::open(&model, &ContextOptions::default()).expect("first");
    let mut second = Context::open(&model, &ContextOptions::default()).expect("second");
    first
        .set_dispatch_observer(Some(NodeDispatchOptions::default()))
        .expect("first observer");
    second
        .set_dispatch_observer(Some(NodeDispatchOptions::default()))
        .expect("second observer");
    first.decode(&tiny_batch()).expect("first decode");
    second.decode(&tiny_batch()).expect("second decode");
    let first_events = first.dispatch_events();
    let second_events = second.dispatch_events();
    assert!(!first_events.is_empty());
    assert!(!second_events.is_empty());
    let first_id = first_events[0].context_id;
    let second_id = second_events[0].context_id;
    assert_ne!(first_id, second_id);
    assert!(first_events
        .iter()
        .all(|event| event.context_id == first_id));
    assert!(second_events
        .iter()
        .all(|event| event.context_id == second_id));
    assert!(first_events
        .iter()
        .all(|event| event.model_instance_id != 0));
    let first_execution = first_events[0].execution_sequence;
    first.decode(&tiny_batch()).expect("second execution");
    let later = first.dispatch_events();
    assert!(later
        .iter()
        .any(|event| event.execution_sequence > first_execution));
}

#[test]
fn dispatch_options_are_bounded() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    assert!(context
        .set_dispatch_observer(Some(NodeDispatchOptions { max_events: 0 }))
        .is_err());
    assert!(context
        .set_dispatch_observer(Some(NodeDispatchOptions {
            max_events: MAX_NODE_DISPATCH_EVENTS + 1
        }))
        .is_err());
}
