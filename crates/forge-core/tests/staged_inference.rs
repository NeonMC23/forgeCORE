//! Native graph-integrated single-layer staged inference.
//!
//! The stage API allocates independent CUDA tensors and rewrites graph weight
//! sources before scheduler allocation. This test proves that inference consumes
//! the substituted tensors, not merely that bytes were copied.

use std::path::PathBuf;

use forge_core::staging::{Engine, MemoryType, StagingError};
use forge_core::{
    enumerate_devices, Batch, BatchBuilder, Context, ContextOptions, DeviceType, GpuLayers, Model,
    ModelLoadMode, ModelOptions, SplitMode,
};

fn fixture() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os("FORGE_TEST_MODEL")?);
    path.is_file().then_some(path)
}

fn batch() -> Batch {
    let mut builder = BatchBuilder::new(8);
    builder.push(1, 0, false).unwrap();
    builder.push(2, 1, false).unwrap();
    builder.push(3, 2, false).unwrap();
    builder.push(4, 3, true).unwrap();
    builder.build().unwrap()
}

fn staging_model(path: &std::path::Path) -> Option<Model> {
    let devices = enumerate_devices();
    let cpu = devices
        .iter()
        .find(|device| device.device_type == DeviceType::Cpu)?
        .clone();
    let gpu = devices
        .iter()
        .find(|device| matches!(device.device_type, DeviceType::Gpu | DeviceType::Igpu))?
        .clone();
    let mut options = ModelOptions::default();
    options.gpu_layers = GpuLayers::Cpu;
    options.split_mode = SplitMode::Layer;
    options.devices = Some(vec![cpu, gpu]);
    options.load_mode = ModelLoadMode::Mmap;
    Some(Model::load_with_options(path, &options).expect("staging model load"))
}

fn fill_staged_layer(
    model: &Model,
    context: &Context,
    engine: &Engine,
    layer: u32,
    corrupt_name: Option<&str>,
) -> Result<usize, String> {
    context
        .stage_layer_cuda(layer)
        .map_err(|error| error.to_string())?;
    let count = context
        .staged_tensor_count(layer)
        .map_err(|error| error.to_string())?;
    let mut tensors = Vec::new();
    for index in 0..count {
        tensors.push(
            context
                .staged_tensor_info(layer, index)
                .map_err(|error| error.to_string())?,
        );
    }
    let total: u64 = tensors.iter().map(|tensor| tensor.bytes).sum();
    let mut host = engine
        .allocate(MemoryType::PinnedHost, total)
        .map_err(|error| error.to_string())?;
    let mut offset = 0u64;
    for tensor in &tensors {
        let length = usize::try_from(tensor.bytes).map_err(|error| error.to_string())?;
        if corrupt_name == Some(tensor.name.as_str()) {
            host.write_host(offset, &vec![0u8; length])
                .map_err(|error| error.to_string())?;
        } else {
            let mut bytes = vec![0u8; length];
            model
                .read_tensor_bytes(&tensor.name, 0, &mut bytes)
                .map_err(|error| error.to_string())?;
            host.write_host(offset, &bytes)
                .map_err(|error| error.to_string())?;
        }
        engine
            .submit_to_device_ptr(&host, offset, tensor.device_ptr, tensor.bytes)
            .map_err(|error| error.to_string())?;
        let _ = engine.wait().map_err(|error| error.to_string())?;
        offset += tensor.bytes;
    }
    host.release().map_err(|error| error.to_string())?;
    let info = context
        .staged_layer_info(layer)
        .map_err(|error| error.to_string())?;
    Ok(usize::try_from(info.graph_substitutions).unwrap_or(0))
}

fn logits(context: &mut Context) -> Vec<f32> {
    context.logits(3).expect("logits").values().to_vec()
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn staged_layer_is_consumed_by_inference() {
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

    let reference = staging_model(&path).expect("reference model");
    let mut reference_context =
        Context::open(&reference, &ContextOptions::default()).expect("reference context");
    reference_context
        .decode(&batch())
        .expect("reference decode");
    let reference_logits = logits(&mut reference_context);

    let staged_model = staging_model(&path).expect("CUDA device");
    let mut options = ContextOptions::default();
    options.n_ctx = 128;
    let mut staged_context = Context::open(&staged_model, &options).expect("staged context");
    fill_staged_layer(&staged_model, &staged_context, &engine, 0, None)
        .expect("stage and fill layer 0");
    staged_context.decode(&batch()).expect("staged decode");
    let staged_logits = logits(&mut staged_context);
    let info = staged_context.staged_layer_info(0).expect("staged info");
    assert!(
        info.graph_substitutions > 0,
        "graph did not substitute staged tensors: {info:?}"
    );
    eprintln!(
        "staged layer consumed: substitutions={}, device_bytes={}, max_logit_diff={:.6}",
        info.graph_substitutions,
        info.device_bytes,
        max_abs_diff(&reference_logits, &staged_logits)
    );
    let argmax = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap_or(0)
    };
    let dot: f32 = reference_logits
        .iter()
        .zip(&staged_logits)
        .map(|(a, b)| a * b)
        .sum();
    let norm_a: f32 = reference_logits.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = staged_logits.iter().map(|v| v * v).sum::<f32>().sqrt();
    eprintln!(
        "argmax ref={} staged={} cosine={:.6} range_ref=({:.3},{:.3}) range_staged=({:.3},{:.3})",
        argmax(&reference_logits),
        argmax(&staged_logits),
        dot / (norm_a * norm_b),
        reference_logits
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min),
        reference_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
        staged_logits.iter().cloned().fold(f32::INFINITY, f32::min),
        staged_logits
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
    );
    let difference = max_abs_diff(&reference_logits, &staged_logits);
    assert_eq!(
        argmax(&reference_logits),
        argmax(&staged_logits),
        "staged and reference argmax differ"
    );
    assert!(
        dot / (norm_a * norm_b) > 0.99,
        "staged and reference logits are not correlated"
    );
    assert!(
        difference < 20.0,
        "staged output diverged too much from the CPU reference: {difference}"
    );
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn corrupting_staged_weights_changes_logits_proving_consumption() {
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
    let reference = staging_model(&path).expect("reference model");
    let mut reference_context =
        Context::open(&reference, &ContextOptions::default()).expect("reference context");
    reference_context
        .decode(&batch())
        .expect("reference decode");
    let reference_logits = logits(&mut reference_context);

    let staged_model = staging_model(&path).expect("CUDA device");
    let mut context = Context::open(&staged_model, &ContextOptions::default()).expect("context");
    fill_staged_layer(
        &staged_model,
        &context,
        &engine,
        0,
        Some("blk.0.attn_q.weight"),
    )
    .expect("stage corrupted layer 0");
    context.decode(&batch()).expect("decode");
    let corrupted_logits = logits(&mut context);
    let info = context.staged_layer_info(0).expect("staged info");
    assert!(info.graph_substitutions > 0);
    let difference = max_abs_diff(&reference_logits, &corrupted_logits);
    eprintln!("corrupted staged weights logit difference: {difference}");
    assert!(
        difference > 1.0,
        "corrupting staged weights did not change logits; staged data may be ignored"
    );
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn staged_layer_eviction_requires_invalidation_and_supports_replacement() {
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

    let model = staging_model(&path).expect("CUDA device");
    let mut options = ContextOptions::default();
    options.n_ctx = 128;
    let mut context = Context::open(&model, &options).expect("context");

    fill_staged_layer(&model, &context, &engine, 0, None).expect("stage and fill layer 0");
    let before = context.staging_state().expect("state before");
    assert_eq!(before.resident_layers, 1);
    assert!(before.graph_invalidated);

    context.decode(&batch()).expect("decode with layer 0");

    // A submitted graph still references the staged sources. Eviction must be
    // refused until invalidation has quiesced and detached them.
    let evict_error = context
        .evict_layer_cuda(0)
        .expect_err("eviction before invalidation must fail");
    assert!(
        evict_error.0.contains("busy"),
        "expected typed busy error, got {evict_error}"
    );

    context
        .invalidate_graph()
        .expect("invalidate after execution");
    // Idempotent invalidation.
    context.invalidate_graph().expect("repeat invalidation");

    let invalidated = context.staging_state().expect("state after invalidation");
    assert!(invalidated.graph_invalidated);
    assert!(invalidated.graph_invalidation_count >= 2);
    assert!(invalidated.scheduler_reset_count >= 2);
    assert!(invalidated.quiescence_wait_ns > 0);

    context.evict_layer_cuda(0).expect("evict layer 0");
    let duplicate = context
        .evict_layer_cuda(0)
        .expect_err("duplicate eviction must fail");
    assert!(
        duplicate.0.contains("not_staged"),
        "expected typed not_staged error, got {duplicate}"
    );

    fill_staged_layer(&model, &context, &engine, 1, None).expect("stage and fill layer 1");

    let mut next = BatchBuilder::new(8);
    next.push(5, 4, true).unwrap();
    let next = next.build().unwrap();
    context.decode(&next).expect("decode with layer 1");
    let lifecycle = context.staging_state().expect("state after replacement");
    assert_eq!(lifecycle.resident_layers, 1);
    assert_eq!(lifecycle.release_count, 1);
    assert!(lifecycle.allocation_count >= 2);
    assert!(lifecycle.graph_rebuild_count >= 2);
    assert_eq!(lifecycle.reuse_count, 0);
    assert!(lifecycle.eviction_supported);
    assert!(!lifecycle.allocation_reuse_supported);
    assert!(!lifecycle.transfer_compute_overlap_supported);

    let info = context.staged_layer_info(1).expect("layer 1 info");
    assert!(
        info.graph_substitutions > 0,
        "rebuilt graph did not consume the replacement layer: {info:?}"
    );
    assert!(
        context.staged_layer_info(0).is_err(),
        "evicted layer 0 must not remain resident"
    );
    assert!(!context.logits(0).expect("logits").values().is_empty());
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL, a CUDA build, and a physical GPU"]
fn replacement_layer_is_consumed_corruption_control() {
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

    let model = staging_model(&path).expect("CUDA device");
    let mut options = ContextOptions::default();
    options.n_ctx = 128;
    let mut context = Context::open(&model, &options).expect("context");

    fill_staged_layer(&model, &context, &engine, 0, None).expect("stage layer 0");
    context.decode(&batch()).expect("decode layer 0");
    let layer0_logits = logits(&mut context);

    context.invalidate_graph().expect("invalidate layer 0");
    context.evict_layer_cuda(0).expect("evict layer 0");
    fill_staged_layer(&model, &context, &engine, 1, None).expect("stage layer 1");
    context.memory().expect("kv memory").clear(true);
    context.decode(&batch()).expect("decode layer 1");
    let layer1_logits = logits(&mut context);

    context.invalidate_graph().expect("invalidate layer 1");
    context.evict_layer_cuda(1).expect("evict layer 1");
    fill_staged_layer(&model, &context, &engine, 1, Some("blk.1.attn_q.weight"))
        .expect("stage corrupted layer 1");
    context.memory().expect("kv memory").clear(true);
    context.decode(&batch()).expect("decode corrupted layer 1");
    let corrupted_logits = logits(&mut context);

    let base_difference = max_abs_diff(&layer0_logits, &layer1_logits);
    let corruption_difference = max_abs_diff(&layer1_logits, &corrupted_logits);
    let info = context.staged_layer_info(1).expect("layer 1 info");
    eprintln!(
        "replacement differences: layer0-vs-layer1={base_difference:.6}, corruption={corruption_difference:.6}, substitutions={}",
        info.graph_substitutions
    );
    assert!(
        info.graph_substitutions > 0,
        "replacement graph did not substitute layer 1"
    );
    assert!(
        corruption_difference > 1.0,
        "corrupting the replacement staged layer did not change logits; staged data may be ignored"
    );

    let lifecycle = context.staging_state().expect("lifecycle");
    assert!(lifecycle.graph_invalidation_count >= 2);
    assert_eq!(lifecycle.release_count, 2);
    assert_eq!(lifecycle.reuse_count, 0);
}
