//! Opt-in GPT-NeoX/Pythia execution regression.
//!
//! The pinned llama.cpp revision implements the `gptneox` architecture, and
//! ForgeCore exposes it through the ordinary Model → Context → Batch → decode
//! path. This test proves that path with a real, small Pythia GGUF instead of
//! inferring support from an architecture string in a dispatch table.
//!
//! The fixture is not stored in the repository. Set `FORGE_GPTNEOX_MODEL` to a
//! GPT-NeoX GGUF (for example a quantized Pythia-70M) and run:
//!
//! ```text
//! FORGE_GPTNEOX_MODEL=/path/to/pythia.gguf \
//!   cargo test -p forge-core --test gptneox_execution -- --ignored --nocapture
//! ```
//!
//! Without the variable the test prints SKIP and returns, so the ordinary
//! workspace suite stays fixture-free.

use std::path::PathBuf;

use forge_core::{BatchBuilder, Context, ContextOptions, EncodeOptions, Model, Tokenizer};

const PROMPT: &str = "The capital of France is";
const GENERATED_TOKENS: usize = 4;

fn fixture() -> Option<PathBuf> {
    match std::env::var("FORGE_GPTNEOX_MODEL") {
        Ok(path) => Some(PathBuf::from(path)),
        Err(_) => {
            println!("SKIP: FORGE_GPTNEOX_MODEL not set (opt-in GPT-NeoX fixture)");
            None
        }
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (index, value) in logits.iter().enumerate().skip(1) {
        if *value > logits[best] {
            best = index;
        }
    }
    u32::try_from(best).expect("logit index fits u32")
}

/// Decode `tokens`, then greedily generate `GENERATED_TOKENS` more tokens.
/// Returns the generated ids so two independent runs can be compared.
fn run_greedy(model: &Model, tokens: &[u32]) -> Vec<u32> {
    let mut context =
        Context::open(model, &ContextOptions::default()).expect("context for GPT-NeoX model");
    let mut builder = BatchBuilder::new(model.vocab_size().expect("vocab size"));
    for (offset, token) in tokens.iter().enumerate() {
        builder
            .push(
                *token,
                u32::try_from(offset).expect("prompt position fits u32"),
                offset + 1 == tokens.len(),
            )
            .expect("prompt batch token");
    }
    context
        .decode(&builder.build().expect("prompt batch"))
        .expect("prompt decode");

    let mut logits = context
        .logits(u32::try_from(tokens.len() - 1).expect("prompt token index fits u32"))
        .expect("logits for final prompt token");
    assert!(
        logits.values().iter().all(|value| value.is_finite()),
        "GPT-NeoX logits must be finite"
    );
    assert!(
        logits
            .values()
            .iter()
            .any(|value| (*value - logits.values()[0]).abs() > f32::EPSILON),
        "GPT-NeoX logits must not be a constant vector"
    );

    let mut generated = Vec::with_capacity(GENERATED_TOKENS);
    let start_position = u32::try_from(tokens.len()).expect("prompt length fits u32");
    for position in start_position..start_position + GENERATED_TOKENS as u32 {
        let next = argmax(logits.values());
        generated.push(next);
        let mut builder = BatchBuilder::new(model.vocab_size().expect("vocab size"));
        builder
            .push(next, position, true)
            .expect("generated token batch");
        context
            .decode(&builder.build().expect("generated batch"))
            .expect("generated token decode");
        logits = context.logits(0).expect("logits for generated token");
    }
    generated
}

#[test]
#[ignore = "requires FORGE_GPTNEOX_MODEL=<gptneox gguf>; opt-in real-model test"]
fn gptneox_fixture_tokenizes_decodes_and_generates_deterministically() {
    let Some(path) = fixture() else {
        return;
    };
    let model = Model::load(&path).expect("GPT-NeoX fixture must load");
    let description = model.description().expect("model description");
    assert!(
        description.to_ascii_lowercase().contains("gptneox"),
        "fixture must be a GPT-NeoX model, got: {description}"
    );
    assert!(model.n_layer().expect("layer count") > 0);
    assert!(model.n_embd().expect("embedding length") > 0);

    let tokenizer = Tokenizer::open(&model).expect("GPT-NeoX tokenizer");
    let mut encode_options = EncodeOptions::default();
    encode_options.add_special = true;
    encode_options.parse_special = true;
    let tokens = tokenizer
        .encode(PROMPT, &encode_options)
        .expect("prompt tokenization");
    assert!(!tokens.is_empty());
    assert!(
        tokens
            .iter()
            .all(|token| *token < model.vocab_size().expect("vocab")),
        "tokenizer ids must stay inside the model vocabulary"
    );

    // Pythia/GPT-NeoX GGUFs do not add BOS. If a future fixture does, the
    // tokenizer reports it and the generated ids below still must repeat.
    println!(
        "gptneox fixture: adds_bos={} prompt_tokens={:?}",
        tokenizer.adds_bos(),
        tokens
    );

    let first = run_greedy(&model, &tokens);
    let second = run_greedy(&model, &tokens);
    assert_eq!(first.len(), GENERATED_TOKENS);
    assert_eq!(
        first, second,
        "greedy CPU decode must be deterministic for identical input"
    );
    let text = tokenizer
        .decode(&first, &Default::default())
        .expect("decode generated tokens");
    println!("gptneox generated ids: {first:?} text={text:?}");
}
