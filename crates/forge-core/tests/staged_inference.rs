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
