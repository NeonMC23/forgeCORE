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
                        ..NodeDispatchOptions::default()
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
                    sample_budget: if sample_pages == 0 { 0 } else { 4096 },
                    ..NodeDispatchOptions::default()
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

#[test]
#[ignore = "manual sampling strategy comparison"]
fn sampling_strategy_overhead_smoke() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let model = Model::load(&path).expect("model");
    for strategy in [
        forge_core::NodeSamplingStrategy::Systematic,
        forge_core::NodeSamplingStrategy::Targeted,
        forge_core::NodeSamplingStrategy::Adaptive,
    ] {
        let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
        context
            .set_dispatch_observer(Some(NodeDispatchOptions {
                max_events: 4096,
                sample_pages: 64,
                sample_budget: 4096,
                strategy,
                cache_capacity: 1024,
                cache_ttl_events: 1024,
            }))
            .expect("observer");
        context.decode(&batch_at(0)).expect("warmup");
        let _ = context.take_dispatch_events();
        let mut samples = Vec::new();
        for repetition in 0..5 {
            let position = 4 * (repetition as u32 + 1);
            let started = Instant::now();
            context.decode(&batch_at(position)).expect("decode");
            samples.push(started.elapsed().as_nanos());
            let _ = context.take_dispatch_events();
        }
        let summary = context.dispatch_summary().expect("summary");
        eprintln!(
            "strategy={strategy:?} median_ns={} cache_reuses={} cache_skips={} cache_evictions={}",
            median(&mut samples),
            summary.cache_reuses,
            summary.cache_skips,
            summary.cache_evictions
        );
    }
}

fn quantile(sorted: &[u128], numerator: usize, denominator: usize) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let index = (sorted.len() - 1) * numerator / denominator;
    sorted[index]
}

#[derive(Default)]
struct SamplingTotals {
    requested: u64,
    sampled: u64,
    fresh: u64,
    reused: u64,
    skipped: u64,
    budget_excluded: u64,
    unsampled: u64,
    resident: u64,
    nonresident: u64,
    unknown: u64,
    cache_reuses: u64,
    cache_skips: u64,
    cache_evictions: u64,
    events: u64,
}

#[test]
#[ignore = "manual statistical sampling benchmark"]
fn statistical_sampling_benchmark() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let runs = std::env::var("FORGE_BENCH_RUNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(7)
        .max(3);
    let iterations = std::env::var("FORGE_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(5)
        .max(1);
    let threads = std::env::var("FORGE_BENCH_THREADS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(1);
    let model = Model::load(&path).expect("model");
    let configs: [(&str, Option<forge_core::NodeSamplingStrategy>); 4] = [
        ("disabled", None),
        (
            "systematic",
            Some(forge_core::NodeSamplingStrategy::Systematic),
        ),
        ("targeted", Some(forge_core::NodeSamplingStrategy::Targeted)),
        ("adaptive", Some(forge_core::NodeSamplingStrategy::Adaptive)),
    ];
    for (name, strategy) in configs {
        let mut samples = Vec::with_capacity(runs * iterations);
        let mut totals = SamplingTotals::default();
        for _ in 0..runs {
            let mut options = ContextOptions::default();
            options.n_threads = threads;
            options.n_threads_batch = threads;
            let mut context = Context::open(&model, &options).expect("context");
            let sampling = strategy.is_some();
            context
                .set_dispatch_observer(Some(NodeDispatchOptions {
                    max_events: 4096,
                    sample_pages: if sampling { 64 } else { 0 },
                    sample_budget: if sampling { 4096 } else { 0 },
                    strategy: strategy.unwrap_or_default(),
                    cache_capacity: 1024,
                    cache_ttl_events: 1024,
                }))
                .expect("observer");
            context.decode(&batch_at(0)).expect("warmup");
            let _ = context.take_dispatch_events();
            for iteration in 0..iterations {
                let position = 4 * (iteration as u32 + 1);
                let started = Instant::now();
                context
                    .decode(&batch_at(position))
                    .expect("measured decode");
                samples.push(started.elapsed().as_nanos());
                for event in context.take_dispatch_events() {
                    totals.events += 1;
                    let Some(sample) = event.dispatch_sample else {
                        continue;
                    };
                    totals.requested += sample.requested_pages;
                    totals.sampled += sample.sampled_pages;
                    totals.fresh += sample.fresh_sampled_pages;
                    totals.reused += sample.reused_pages;
                    totals.skipped += sample.skipped_pages;
                    totals.budget_excluded += sample.budget_excluded_pages;
                    totals.unsampled += sample.unsampled_pages;
                    totals.resident += sample.resident_pages;
                    totals.nonresident += sample.nonresident_pages;
                    totals.unknown += sample.unknown_pages;
                }
            }
            let summary = context.dispatch_summary().expect("summary");
            totals.cache_reuses += summary.cache_reuses;
            totals.cache_skips += summary.cache_skips;
            totals.cache_evictions += summary.cache_evictions;
        }
        samples.sort_unstable();
        let median = quantile(&samples, 1, 2);
        let min = *samples.first().unwrap_or(&0);
        let max = *samples.last().unwrap_or(&0);
        let q1 = quantile(&samples, 1, 4);
        let q3 = quantile(&samples, 3, 4);
        eprintln!(
            "{{\"config\":\"{name}\",\"threads\":{threads},\"runs\":{runs},\"iterations\":{iterations},\"measurements\":{},\"sample_pages\":{},\"sample_budget\":{},\"median_ns\":{median},\"min_ns\":{min},\"max_ns\":{max},\"q1_ns\":{q1},\"q3_ns\":{q3},\"events\":{},\"requested_pages\":{},\"sampled_pages\":{},\"fresh_pages\":{},\"reused_pages\":{},\"skipped_pages\":{},\"budget_excluded_pages\":{},\"unsampled_pages\":{},\"resident_pages\":{},\"nonresident_pages\":{},\"unknown_pages\":{},\"cache_reuses\":{},\"cache_skips\":{},\"cache_evictions\":{},\"condition\":\"warm-cache-uncontrolled\"}}",
            samples.len(),
            if strategy.is_some() { 64 } else { 0 },
            if strategy.is_some() { 4096 } else { 0 },
            totals.events,
            totals.requested,
            totals.sampled,
            totals.fresh,
            totals.reused,
            totals.skipped,
            totals.budget_excluded,
            totals.unsampled,
            totals.resident,
            totals.nonresident,
            totals.unknown,
            totals.cache_reuses,
            totals.cache_skips,
            totals.cache_evictions
        );
    }
}
