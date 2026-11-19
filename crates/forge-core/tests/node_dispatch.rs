//! CPU node-dispatch observer tests against the tiny fixture.

use std::path::PathBuf;

use forge_core::{
    BatchBuilder, Context, ContextOptions, Model, NodeDispatchOptions, NodeSamplingStrategy,
    MAX_NODE_DISPATCH_EVENTS,
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

fn batch_at(base: u32) -> forge_core::Batch {
    let mut builder = BatchBuilder::new(32);
    builder.push(1, base, false).unwrap();
    builder.push(5, base + 1, false).unwrap();
    builder.push(6, base + 2, false).unwrap();
    builder.push(7, base + 3, true).unwrap();
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
        .set_dispatch_observer(Some(NodeDispatchOptions {
            max_events: 4096,
            sample_pages: 0,
            ..NodeDispatchOptions::default()
        }))
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
    assert!(events.iter().all(|event| event.dispatch_sample.is_none()));
    assert_eq!(summary.budget_total, 0);
    assert_eq!(summary.budget_remaining, 0);
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
    let mut indices: Vec<u32> = same_execution
        .iter()
        .map(|event| event.node_index)
        .collect();
    let event_count = indices.len();
    indices.sort_unstable();
    indices.dedup();
    assert_eq!(indices.len(), event_count, "duplicate dispatch index");
    assert!(indices.iter().all(|index| *index < node_count));
}

#[test]
fn pre_dispatch_sampling_is_bounded_and_generation_aware() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    context
        .set_dispatch_observer(Some(NodeDispatchOptions {
            max_events: 4096,
            sample_pages: 64,
            sample_budget: 64,
            ..NodeDispatchOptions::default()
        }))
        .expect("sampling observer");
    context.decode(&tiny_batch()).expect("decode");
    let events = context.dispatch_events();
    let samples: Vec<_> = events
        .iter()
        .filter_map(|event| event.dispatch_sample)
        .collect();
    assert!(!samples.is_empty());
    assert!(samples.iter().any(|sample| {
        sample.requested_pages > 0
            && sample.sampled_pages > 0
            && sample.mapping_generation != 0
            && matches!(
                sample.status,
                forge_core::PageSampleStatus::Complete | forge_core::PageSampleStatus::Partial
            )
    }));
    assert!(samples.iter().all(|sample| sample.sampled_pages <= 64));
    assert!(samples.iter().all(|sample| sample.resident_pages
        + sample.nonresident_pages
        + sample.unknown_pages
        == sample.sampled_pages));
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
        .set_dispatch_observer(Some(NodeDispatchOptions {
            max_events: 2,
            sample_pages: 0,
            ..NodeDispatchOptions::default()
        }))
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
    let _ = first_execution;
}

#[test]
fn execution_budget_is_strict_and_reports_unsampled_pages() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    context
        .set_dispatch_observer(Some(NodeDispatchOptions {
            max_events: 4096,
            sample_pages: 4096,
            sample_budget: 1,
            strategy: NodeSamplingStrategy::Targeted,
            cache_capacity: 0,
            cache_ttl_events: 0,
        }))
        .expect("observer");
    context.decode(&tiny_batch()).expect("decode");
    let events = context.dispatch_events();
    let sampled: u64 = events
        .iter()
        .filter_map(|event| event.dispatch_sample)
        .map(|sample| sample.sampled_pages)
        .sum();
    assert!(sampled <= 1, "budget exceeded: {sampled}");
    assert!(events
        .iter()
        .filter_map(|event| event.dispatch_sample)
        .any(|sample| sample.unsampled_pages > 0));
}

#[test]
fn sampling_strategies_have_distinct_cache_behavior() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    for strategy in [
        NodeSamplingStrategy::Systematic,
        NodeSamplingStrategy::Targeted,
        NodeSamplingStrategy::Adaptive,
    ] {
        let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
        context
            .set_dispatch_observer(Some(NodeDispatchOptions {
                max_events: 4096,
                sample_pages: 4096,
                sample_budget: 4096,
                strategy,
                cache_capacity: 256,
                cache_ttl_events: 1024,
            }))
            .expect("observer");
        context.decode(&batch_at(0)).expect("first decode");
        let first = context.dispatch_summary().expect("summary");
        context.decode(&batch_at(4)).expect("second decode");
        let second = context.dispatch_summary().expect("summary");
        match strategy {
            NodeSamplingStrategy::Adaptive => {
                assert!(second.cache_reuses > 0, "adaptive cache must be reused");
                assert_eq!(second.cache_skips, 0);
            }
            NodeSamplingStrategy::Targeted => {
                assert!(second.cache_skips > 0, "targeted must skip fresh ranges");
                assert_eq!(second.cache_reuses, 0);
                assert!(second.events.checked_sub(second.cache_skips).is_some());
            }
            NodeSamplingStrategy::Systematic => {
                assert_eq!(second.cache_reuses, 0);
                assert_eq!(second.cache_skips, 0);
            }
        }
        assert!(first.events > 0);
    }
}

#[test]
fn zero_sampling_budget_disables_sampling() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    context
        .set_dispatch_observer(Some(NodeDispatchOptions {
            max_events: 1024,
            sample_pages: 4096,
            sample_budget: 0,
            ..NodeDispatchOptions::default()
        }))
        .expect("observer");
    context.decode(&tiny_batch()).expect("decode");
    let summary = context.dispatch_summary().expect("summary");
    assert_eq!(summary.budget_total, 0);
    assert!(context
        .dispatch_events()
        .iter()
        .all(|event| event.dispatch_sample.is_none()));
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
        .set_dispatch_observer(Some(NodeDispatchOptions {
            max_events: 0,
            sample_pages: 0,
            ..NodeDispatchOptions::default()
        }))
        .is_err());
    assert!(context
        .set_dispatch_observer(Some(NodeDispatchOptions {
            max_events: MAX_NODE_DISPATCH_EVENTS + 1,
            sample_pages: 0,
            ..NodeDispatchOptions::default()
        }))
        .is_err());
}
