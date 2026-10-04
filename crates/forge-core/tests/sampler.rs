//! Phase-2 sampler chains: synthetic-logits behavior plus one
//! fixture-gated integration loop.
//!
//! All behavior below runs on synthetic `&[f32]` (no model needed —
//! the sampler is decoupled from `Context` by design); only
//! `integration_sampled_id_feeds_execution_and_tokenizer` needs
//! `FORGE_TEST_MODEL` and skips otherwise (see `docs/NATIVE.md`).
//!
//! Exact-id pins reproduce the pinned native `mt19937` stream: they
//! are correct for llama.cpp `v0.5.0` @ `7fe450e1` and must be
//! re-pinned — never loosened — if the pin moves.

use forge_core::{
    BatchBuilder, Context, ContextOptions, DecodeOptions, Model, SampleConfig, SamplerChain,
    Tokenizer,
};
use std::collections::BTreeSet;
use std::path::PathBuf;

fn fixture(var: &str) -> Option<PathBuf> {
    match std::env::var(var) {
        Ok(path) => Some(PathBuf::from(path)),
        Err(_) => {
            println!("SKIP: {var} not set (see docs/NATIVE.md)");
            None
        }
    }
}

fn greedy_chain() -> SamplerChain {
    SampleConfig::default().build_chain().expect("greedy chain")
}

fn dist_chain(seed: u32) -> SamplerChain {
    let mut chain = SamplerChain::new();
    chain.push_dist(Some(seed));
    chain
}

fn full_config() -> SampleConfig {
    let mut config = SampleConfig::default();
    config.temperature = 0.8;
    config.top_k = Some(40);
    config.top_p = Some(0.9);
    config.min_p = Some(0.1);
    config.seed = Some(7);
    config
}

// -- A. greedy ---------------------------------------------------------------

#[test]
fn greedy_selects_argmax_first_max_wins() {
    let mut chain = greedy_chain();
    // Same shape as RAMforge's greedy unit test (compat witness).
    assert_eq!(chain.sample(&[0.1, 0.5, 0.2]).expect("sample"), 1);
    // Ties resolve to the lowest id (native strict-`>` scan).
    assert_eq!(chain.sample(&[0.5, 0.5, 0.2]).expect("ties"), 0);
    assert_eq!(chain.sample(&[0.2, 0.1, 0.9]).expect("sample"), 2);
}

#[test]
fn greedy_config_builds_a_seedless_singleton() {
    let chain = greedy_chain();
    assert_eq!(chain.len(), 1);
    assert_eq!(chain.seed(), None, "greedy holds no RNG");
    let debug = format!("{chain:?}");
    assert!(debug.contains("len"), "Debug reports length: {debug}");
}

// -- B. temperature ----------------------------------------------------------

#[test]
fn temperature_zero_is_argmax_for_any_seed() {
    for seed in [None, Some(0), Some(1234)] {
        let mut chain = SamplerChain::new();
        chain.push_temp(0.0).expect("temp 0");
        chain.push_dist(seed);
        assert_eq!(
            chain.sample(&[0.1, 0.5, 0.2]).expect("sample"),
            1,
            "temp 0 + dist == argmax (seed {seed:?})"
        );
    }
}

#[test]
fn temperature_one_is_identity_for_fixed_seed() {
    let mut plain = dist_chain(42);
    let mut tempered = SamplerChain::new();
    tempered.push_temp(1.0).expect("temp 1");
    tempered.push_dist(Some(42));
    assert_eq!(
        plain.sample(&[0.1, 0.5, 0.2]).expect("plain"),
        tempered.sample(&[0.1, 0.5, 0.2]).expect("tempered"),
        "temp 1.0 must not change the stream"
    );
}

#[test]
fn invalid_push_parameters_are_errors() {
    let mut chain = SamplerChain::new();
    assert_eq!(chain.len(), 0, "nothing added yet");
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
        chain.push_temp(bad).expect_err("bad temp must fail");
    }
    chain.push_top_k(0).expect_err("k=0 must fail");
    chain
        .push_top_k(u32::MAX)
        .expect_err("k > i32::MAX must fail");
    for bad in [0.0, -0.5, 1.5, f32::NAN] {
        chain.push_top_p(bad).expect_err("bad top_p must fail");
        chain.push_min_p(bad).expect_err("bad min_p must fail");
    }
    assert_eq!(chain.len(), 0, "failed pushes add nothing");
    full_config().build_chain().expect("valid config builds");
    let mut bad = full_config();
    bad.top_p = Some(2.0);
    bad.build_chain().expect_err("bad config must fail");
}

// -- C. top-k ----------------------------------------------------------------

#[test]
fn top_k_one_is_argmax_for_any_seed() {
    for seed in [0, 1, 999] {
        let mut chain = SamplerChain::new();
        chain.push_top_k(1).expect("k=1");
        chain.push_dist(Some(seed));
        assert_eq!(chain.sample(&[0.1, 0.5, 0.2]).expect("sample"), 1);
    }
}

#[test]
fn top_k_restricts_and_clamps() {
    // k=2 on [0.1, 0.5, 0.2]: only ids {1, 2} can ever emerge.
    let mut chain = SamplerChain::new();
    chain.push_top_k(2).expect("k=2");
    chain.push_dist(Some(3));
    let mut seen = BTreeSet::new();
    for _ in 0..50 {
        seen.insert(chain.sample(&[0.1, 0.5, 0.2]).expect("sample"));
    }
    assert_eq!(
        seen,
        BTreeSet::from([1, 2]),
        "k=2 keeps exactly the top two"
    );
    // k >= n keeps everything (valid, not an error).
    let mut big = SamplerChain::new();
    big.push_top_k(1000).expect("k=1000");
    big.push_dist(Some(42));
    assert_eq!(big.sample(&[0.1, 0.5, 0.2]).expect("sample"), 0);
}

// -- D. top-p / min-p ----------------------------------------------------------

#[test]
fn top_p_pins_and_nucleus_membership() {
    // Exact stream pins for seed 42 on [0.1, 0.5, 0.2].
    for (p, expected) in [(1.0, 2), (0.5, 2), (0.3, 1)] {
        let mut chain = SamplerChain::new();
        chain.push_top_p(p).expect("top_p");
        chain.push_dist(Some(42));
        assert_eq!(
            chain.sample(&[0.1, 0.5, 0.2]).expect("sample"),
            expected,
            "p={p}"
        );
    }
    // p=1 is a native no-op: identical to bare dist for fixed seeds.
    for seed in [0, 42] {
        let mut bare = dist_chain(seed);
        let mut full = SamplerChain::new();
        full.push_top_p(1.0).expect("p=1");
        full.push_dist(Some(seed));
        assert_eq!(
            bare.sample(&[0.1, 0.5, 0.2]).expect("bare"),
            full.sample(&[0.1, 0.5, 0.2]).expect("full")
        );
    }
    // Skewed logits + p=0.5: the nucleus is {0} alone.
    let mut skewed = SamplerChain::new();
    skewed.push_top_p(0.5).expect("top_p");
    skewed.push_dist(Some(11));
    for _ in 0..50 {
        assert_eq!(skewed.sample(&[5.0, 0.1, 0.1]).expect("sample"), 0);
    }
}

#[test]
fn min_p_pins_and_argmax_at_one() {
    // min_p=1 keeps only the argmax, whatever the seed.
    for seed in [0, 42, 777] {
        let mut chain = SamplerChain::new();
        chain.push_min_p(1.0).expect("min_p=1");
        chain.push_dist(Some(seed));
        assert_eq!(chain.sample(&[1.0, 5.0, 2.0]).expect("sample"), 1);
    }
    // min_p=0.5 on [4.5, 5.0, 2.0]: min logit 5+ln(0.5)≈4.31 keeps {0, 1}.
    let mut chain = SamplerChain::new();
    chain.push_min_p(0.5).expect("min_p");
    chain.push_dist(Some(42));
    let mut seen = BTreeSet::new();
    for _ in 0..50 {
        seen.insert(chain.sample(&[4.5, 5.0, 2.0]).expect("sample"));
    }
    assert_eq!(seen, BTreeSet::from([0, 1]));
}

// -- E. determinism / seeds ----------------------------------------------------

#[test]
fn seeded_dist_pins_and_replays() {
    // Exact first-two-draw pins for fixed seeds.
    for (seed, first, second) in [(0, 1, 2), (1, 2, 2), (42, 2, 0)] {
        let mut chain = dist_chain(seed);
        assert_eq!(
            chain.sample(&[0.1, 0.5, 0.2]).expect("first"),
            first,
            "seed {seed}"
        );
        assert_eq!(
            chain.sample(&[0.1, 0.5, 0.2]).expect("second"),
            second,
            "seed {seed}"
        );
        assert_eq!(chain.seed(), Some(seed), "seed echoes");
    }
    // Fresh chains + same seed = same stream; reset() replays it.
    let mut first = dist_chain(42);
    let mut second = dist_chain(42);
    assert_eq!(
        first.sample(&[0.1, 0.5, 0.2]).expect("a"),
        second.sample(&[0.1, 0.5, 0.2]).expect("b")
    );
    let _ = first.sample(&[0.1, 0.5, 0.2]).expect("advance");
    first.reset();
    assert_eq!(
        first.sample(&[0.1, 0.5, 0.2]).expect("replay"),
        2,
        "reset restores the explicit-seed stream to its first draw"
    );
}

#[test]
fn different_seeds_sample_differently() {
    // Flat logits: 20 fixed seeds must reach all 3 ids (deterministic
    // seeds, so the test itself is deterministic).
    let mut seen = BTreeSet::new();
    for seed in 0..20u32 {
        seen.insert(dist_chain(seed).sample(&[1.0, 1.0, 1.0]).expect("sample"));
    }
    assert_eq!(seen, BTreeSet::from([0, 1, 2]));
    // Random seeding reports a resolved seed (value unpredictable).
    let mut random = SamplerChain::new();
    random.push_dist(None);
    assert!(random.seed().is_some(), "random resolves a seed");
}

// -- F. chain mechanics ----------------------------------------------------------

#[test]
fn chain_add_remove_len_and_empty_sample() {
    let mut chain = SamplerChain::new();
    assert!(chain.is_empty());
    chain
        .sample(&[0.1])
        .expect_err("empty chain selects nothing");
    chain.push_top_k(5).expect("k");
    chain.push_temp(0.8).expect("temp");
    chain.push_dist(Some(1));
    assert_eq!(chain.len(), 3);
    // Filters without a selector also select nothing.
    let mut filters = SamplerChain::new();
    filters.push_top_k(2).expect("k");
    filters
        .sample(&[0.1, 0.5])
        .expect_err("no selector selects nothing");
    // Remove frees the element and compacts the chain.
    chain.remove(1).expect("remove middle");
    assert_eq!(chain.len(), 2);
    let error = chain.remove(9).expect_err("OOB remove must fail");
    assert!(error.to_string().starts_with("sample error"), "{error}");
    // Accept follows the protocol (and validates the id width).
    chain.accept(3).expect("accept");
    chain.accept(u32::MAX).expect_err("id too large must fail");
    chain.sample(&[]).expect_err("empty logits must fail");
}

// -- G. integration --------------------------------------------------------------

#[test]
fn integration_sampled_id_feeds_execution_and_tokenizer() {
    let Some(path) = fixture("FORGE_TEST_MODEL") else {
        return;
    };
    let model = Model::load(&path).expect("fixture model must load");
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    let mut builder = BatchBuilder::new(32);
    builder.push(1, 0, true).unwrap();
    context.decode(&builder.build().unwrap()).expect("decode");
    let logits = context.logits(0).expect("logits");
    assert_eq!(logits.n_vocab(), 32);

    // Greedy over real logits: a valid id, stable across fresh chains.
    let first = greedy_chain().sample(logits.values()).expect("greedy");
    let again = greedy_chain().sample(logits.values()).expect("greedy");
    assert_eq!(first, again, "greedy is deterministic");
    assert!(first < 32, "sampled id is in-vocab: {first}");

    // The id feeds straight back into the execution path ...
    let mut next = BatchBuilder::new(32);
    next.push(first, 1, true).expect("sampled id batches");
    next.build().expect("batch builds");

    // ... and into the tokenizer (decode succeeds even when the id is
    // a control piece that renders empty under default options).
    let tokenizer = Tokenizer::open(&model).expect("tokenizer");
    tokenizer
        .decode(&[first], &DecodeOptions::default())
        .expect("tokenizer decodes the id");

    // Seeded dist over real logits: valid id, reproducible stream.
    let mut config = SampleConfig::default();
    config.temperature = 1.0;
    config.seed = Some(1234);
    let one = config
        .build_chain()
        .expect("chain")
        .sample(logits.values())
        .expect("dist");
    let two = config
        .build_chain()
        .expect("chain")
        .sample(logits.values())
        .expect("dist");
    assert_eq!(one, two, "fixed seed reproduces over real logits");
    assert!(one < 32, "dist id is in-vocab: {one}");
}
