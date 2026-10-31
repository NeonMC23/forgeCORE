//! Manual CPU node-dispatch observer overhead smoke test.
//!
//! Run with an actual model:
//! `FORGE_TEST_MODEL=... cargo test -p forge-core --test node_dispatch_overhead -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use forge_core::{BatchBuilder, Context, ContextOptions, Model, NodeDispatchOptions};

fn fixture() -> Option<PathBuf> {
    let path = std::env::var_os("FORGE_TEST_MODEL")?;
    let path = PathBuf::from(path);
    path.is_file().then_some(path)
}

fn batch_at(position: u32) -> forge_core::Batch {
    let mut builder = BatchBuilder::new(32);
    builder.push(1, position, false).unwrap();
    builder.push(5, position + 1, false).unwrap();
    builder.push(6, position + 2, false).unwrap();
    builder.push(7, position + 3, true).unwrap();
    builder.build().unwrap()
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

#[test]
#[ignore = "manual observer overhead measurement"]
fn node_dispatch_observer_overhead_smoke() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    for threads in [1usize, 4] {
        for (enabled, drain) in [(false, false), (true, true), (true, false)] {
            let mut options = ContextOptions::default();
            options.n_threads = threads as u32;
            options.n_threads_batch = threads as u32;
            let mut context = Context::open(&model, &options).expect("context");
            if enabled {
                context
                    .set_dispatch_observer(Some(NodeDispatchOptions {
                        max_events: if drain { 4096 } else { 65_536 },
                        sample_pages: 0,
                    }))
                    .expect("observer");
            }
            context.decode(&batch_at(0)).expect("warmup");
            if enabled {
                let _ = context.take_dispatch_events();
            }
            let mut samples = Vec::new();
            let mut events = 0usize;
            let mut dropped = 0u64;
            for repetition in 0..5 {
                let position = 4 * (repetition as u32 + 1);
                let started = Instant::now();
                context
                    .decode(&batch_at(position))
                    .expect("measured decode");
                samples.push(started.elapsed().as_nanos());
                if enabled && drain {
                    let summary = context.dispatch_summary().expect("summary");
                    dropped = summary.dropped_events;
                    events += context.take_dispatch_events().len();
                }
            }
            if enabled && !drain {
                let summary = context.dispatch_summary().expect("summary");
                events = summary.events as usize;
                dropped = summary.dropped_events;
            }
            let median = median(&mut samples);
            eprintln!(
                "threads={threads} observer={} drain={drain} median_ns={median} events={events} dropped={dropped}",
                if enabled { "on" } else { "off" }
            );
            if enabled {
                assert!(events > 0);
            }
        }
    }
}

#[test]
#[ignore = "manual sampling overhead measurement"]
fn pre_dispatch_sampling_overhead_smoke() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    for threads in [1usize, 4] {
        for sample_pages in [0u64, 64] {
            let mut options = ContextOptions::default();
            options.n_threads = threads as u32;
            options.n_threads_batch = threads as u32;
            let mut context = Context::open(&model, &options).expect("context");
            context
                .set_dispatch_observer(Some(NodeDispatchOptions {
                    max_events: 4096,
                    sample_pages,
                }))
                .expect("observer");
            context.decode(&batch_at(0)).expect("warmup");
            let _ = context.take_dispatch_events();
            let mut samples = Vec::new();
            let mut observed = 0u64;
            for repetition in 0..5 {
                let position = 4 * (repetition as u32 + 1);
                let started = Instant::now();
                context.decode(&batch_at(position)).expect("decode");
                samples.push(started.elapsed().as_nanos());
                observed += context.dispatch_summary().expect("summary").events;
                let _ = context.take_dispatch_events();
            }
            let median = median(&mut samples);
            eprintln!(
                "threads={threads} sample_pages={sample_pages} median_ns={median} events={observed}"
            );
        }
    }
}
