//! Physical single-layer transfer/ownership prototype.
//!
//! This proves byte transfer, integrity, bounded staging ownership, completion,
//! and model validity. It does **not** prove that inference consumed the staged
//! copy; the graph integration remains unimplemented.

use std::path::PathBuf;

use forge_core::staging::{Direction, Engine, MemoryType, StagingError};
use forge_core::{Context, ContextOptions, Model};

fn fixture() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("FORGE_TEST_MODEL")?);
    path.is_file().then_some(path)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL and a CUDA driver"]
fn single_layer_transfer_ownership_prototype() {
    let Some(path) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let engine = match Engine::new(0) {
        Ok(engine) => engine,
        Err(StagingError::Unsupported(reason)) => {
            eprintln!("SKIP: CUDA unavailable: {reason}");
            return;
        }
        Err(error) => panic!("staging engine: {error}"),
    };
    let model = Model::load(&path).expect("model load");
    let lifecycle = model.lifetime_state().expect("lifetime state");
    assert!(lifecycle.quiescent);
    assert_eq!(lifecycle.active_contexts, 0);

    let plan = model.layer_staging_plan(0).expect("layer plan");
    assert!(plan.total_bytes > 0);
    assert!(!plan.tensors.is_empty());
    eprintln!(
        "layer {}: {} tensors, {} bytes",
        plan.layer,
        plan.tensors.len(),
        plan.total_bytes
    );

    let mut source = vec![0u8; usize::try_from(plan.total_bytes).expect("layer fits usize")];
    let mut offset = 0usize;
    for tensor in &plan.tensors {
        let length = usize::try_from(tensor.tensor_bytes).expect("tensor fits usize");
        model
            .read_tensor_bytes(&tensor.name, 0, &mut source[offset..offset + length])
            .unwrap_or_else(|error| panic!("read {}: {error}", tensor.name));
        offset += length;
    }
    assert_eq!(offset, source.len());
    let source_hash = fnv1a(&source);

    let mut host = engine
        .allocate(MemoryType::PinnedHost, plan.total_bytes)
        .expect("pinned allocation");
    let mut device = engine
        .allocate(MemoryType::Device, plan.total_bytes)
        .expect("device allocation");
    let mut readback = engine
        .allocate(MemoryType::PinnedHost, plan.total_bytes)
        .expect("readback allocation");
    host.write_host(0, &source).expect("host write");
    engine
        .submit(
            &host,
            0,
            &device,
            0,
            plan.total_bytes,
            Direction::HostToDevice,
        )
        .expect("H2D submit");
    let h2d = engine.wait().expect("H2D wait");
    assert_eq!(h2d.bytes, plan.total_bytes);
    assert!(h2d.duration_ns > 0);
    engine
        .submit(
            &device,
            0,
            &readback,
            0,
            plan.total_bytes,
            Direction::DeviceToHost,
        )
        .expect("D2H submit");
    let d2h = engine.wait().expect("D2H wait");
    assert_eq!(d2h.bytes, plan.total_bytes);
    let mut received = vec![0u8; source.len()];
    readback.read_host(0, &mut received).expect("readback");
    assert_eq!(fnv1a(&received), source_hash);
    assert_eq!(received, source);

    let stats = engine.stats().expect("stats");
    assert_eq!(stats.transfer_requests, 2);
    assert_eq!(stats.transfer_completed, 2);
    assert_eq!(stats.transfer_failed, 0);
    assert_eq!(stats.transfer_bytes, plan.total_bytes * 2);
    assert_eq!(stats.in_flight_operations, 0);
    eprintln!(
        "transfer prototype: H2D {:.2} GB/s, D2H {:.2} GB/s, bytes={}, source_hash={:x}",
        h2d.bandwidth_gbps().unwrap_or(0.0),
        d2h.bandwidth_gbps().unwrap_or(0.0),
        plan.total_bytes,
        source_hash
    );

    host.release().expect("release host");
    device.release().expect("release device");
    readback.release().expect("release readback");
    assert_eq!(engine.stats().expect("stats").current_bytes, 0);

    // The transfer/ownership prototype must leave the model usable, but it
    // does not claim that inference consumed the staged copy.
    let mut options = ContextOptions::default();
    options.n_ctx = 128;
    let context = Context::open(&model, &options).expect("model remains usable");
    assert!(context.n_ctx() >= 128);
}
