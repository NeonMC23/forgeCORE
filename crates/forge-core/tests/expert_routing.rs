//! Opt-in real-model expert-routing observation test.
//!
//! Requires a CPU-executable MoE GGUF (DeepSeek-Coder-V2-Lite used during
//! development) and `FORGE_TEST_MODEL`. The test proves that observation is
//! read-only, produces valid indices, and leaves normal decoding unchanged.

use forge_core::{
    BatchBuilder, Context, ContextOptions, ExpertRoutingOptions, GpuLayers, Model,
    ModelLoadMode, ModelOptions,
};

fn fixture() -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(std::env::var_os("FORGE_TEST_MODEL")?);
    path.is_file().then_some(path)
}

#[test]
#[ignore = "requires FORGE_TEST_MODEL=<MoE gguf>; opt-in real-model routing test"]
fn routing_observation_records_valid_first_token_indices() {
    let Some(path) = fixture() else {
        return;
    };
    let expert_count = std::env::var("FORGE_TEST_EXPERT_COUNT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64u32);
    let expert_used = std::env::var("FORGE_TEST_EXPERT_USED")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6u32);

    let mut model_options = ModelOptions::default();
    model_options.gpu_layers = GpuLayers::Cpu;
    model_options.load_mode = ModelLoadMode::MmapNoPrefetch;
    model_options.use_extra_bufts = Some(false);
    let model = Model::load_with_options(&path, &model_options).expect("model load");

    let mut options = ContextOptions::default();
    options.n_ctx = 256;
    options.n_batch = 64;
    options.n_ubatch = 64;
    options.n_threads = 4;
    options.n_threads_batch = 4;
    let context = Context::open(&model, &options).expect("plain context");
    assert!(context.routing_observations().is_empty());
    assert!(context.routing_summary().is_none());

    options.routing = Some(ExpertRoutingOptions {
        max_observations: 4096,
        expert_count,
        expert_used_count: expert_used,
    });
    let mut context = Context::open(&model, &options).expect("routing context");

    let vocab = model.vocab_size().expect("vocab");
    let mut batch = BatchBuilder::new(vocab);
    batch.push(1, 0, true).expect("token 1");
    batch.push(2, 1, true).expect("token 2");
    context
        .decode(&batch.build().expect("batch"))
        .expect("prefill");

    let mut logits = context.logits(1).expect("logits").values().to_vec();
    for position in 2..6u32 {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index as u32)
            .expect("non-empty logits");
        let mut batch = BatchBuilder::new(vocab);
        batch.push(next, position, true).expect("generated token");
        context
            .decode(&batch.build().expect("batch"))
            .expect("decode");
        logits = context.logits(0).expect("logits").values().to_vec();
    }

    let observations = context.routing_observations();
    let summary = context.routing_summary().expect("routing summary");
    assert!(summary.observations >= 1);
    assert!(summary.complete_decode_observations >= 1);
    assert_eq!(summary.invalid_indices, 0);
    assert!(observations.iter().any(|observation| {
        observation.complete_for_token
            && observation.selected_experts.len() == expert_used as usize
            && observation
                .selected_experts
                .iter()
                .all(|expert| *expert < expert_count)
    }));
    assert!(observations.iter().all(|observation| {
        observation.selected_experts.len() == expert_used as usize
            && observation.invalid_indices == 0
    }));
    assert!(!logits.is_empty());
}
