//! Phase-4 context/batch extension: effective sizing, batching options,
//! multi-sequence and embedding batches, embedding outputs, threading,
//! causal/flash controls, and the advisory offload flags.
//!
//! Fixture tests SKIP without `FORGE_TEST_MODEL`; the GPU smoke test
//! additionally SKIPs without `supports_gpu_offload`. Nothing here
//! assumes a GPU.

use forge_core::model::set_log_quiet;
use forge_core::{
    supports_gpu_offload, AttentionType, BatchBuilder, Context, ContextOptions, FlashAttnType,
    Model, PoolingType,
};
use std::path::PathBuf;

fn fixture() -> Option<PathBuf> {
    match std::env::var("FORGE_TEST_MODEL") {
        Ok(path) => Some(PathBuf::from(path)),
        Err(_) => {
            println!("SKIP: FORGE_TEST_MODEL not set (see docs/NATIVE.md)");
            None
        }
    }
}

fn load_fixture() -> Option<Model> {
    fixture().map(|path| Model::load(&path).expect("fixture model must load"))
}

fn tiny_batch() -> forge_core::Batch {
    let mut builder = BatchBuilder::new(32);
    builder.push(1, 0, false).unwrap();
    builder.push(5, 1, false).unwrap();
    builder.push(6, 2, false).unwrap();
    builder.push(7, 3, true).unwrap();
    builder.build().unwrap()
}

/// The tiny batch shifted to start at `pos0` (positions must advance
/// across decodes on one context).
fn tiny_batch_at(pos0: u32) -> forge_core::Batch {
    let mut builder = BatchBuilder::new(32);
    builder.push(1, pos0, false).unwrap();
    builder.push(5, pos0 + 1, false).unwrap();
    builder.push(6, pos0 + 2, false).unwrap();
    builder.push(7, pos0 + 3, true).unwrap();
    builder.build().unwrap()
}

/// `n` consecutive tokens on sequence 0, logits on the last only.
fn long_batch(n_tokens: usize) -> forge_core::Batch {
    let mut builder = BatchBuilder::new(32);
    for i in 0..n_tokens {
        builder
            .push((i * 7 % 32) as u32, i as u32, i + 1 == n_tokens)
            .unwrap();
    }
    builder.build().unwrap()
}

// -- A. effective sizing --------------------------------------------------------

#[test]
fn default_context_reports_effective_sizes() {
    let Some(model) = load_fixture() else { return };
    let context = Context::open(&model, &ContextOptions::default()).expect("default context");
    // Train length 64, padded up to a multiple of 256; causal batch
    // clamped to the unpadded length; single sequence; lockstep threads.
    assert_eq!(context.n_ctx(), 256);
    assert_eq!(context.n_ctx_seq(), 256);
    assert_eq!(context.n_batch(), 64);
    assert_eq!(context.n_ubatch(), 64);
    assert_eq!(context.n_seq_max(), 1);
    assert_eq!(context.n_vocab(), 32);
    assert_eq!(context.n_threads(), 1);
    assert_eq!(context.n_threads_batch(), 1);
    assert!(context.causal_attn());
    assert_eq!(context.pooling(), PoolingType::None);
}

#[test]
fn explicit_context_sizes_are_padded() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.n_ctx = 512;
    let context = Context::open(&model, &options).expect("n_ctx=512");
    assert_eq!(context.n_ctx(), 512);
    assert_eq!(context.n_batch(), 512);

    options.n_ctx = 100;
    let context = Context::open(&model, &options).expect("n_ctx=100");
    assert_eq!(context.n_ctx(), 256);
}

#[test]
fn seq_max_scales_context() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.n_seq_max = 2;
    let context = Context::open(&model, &options).expect("n_seq_max=2");
    assert_eq!(context.n_seq_max(), 2);
    assert_eq!(context.n_ctx_seq(), 256);
    assert_eq!(context.n_ctx(), 512);
}

#[test]
fn batch_sizes_are_honored() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.n_ctx = 512;
    options.n_batch = 128;
    options.n_ubatch = 32;
    let context = Context::open(&model, &options).expect("custom batch sizes");
    assert_eq!(context.n_batch(), 128);
    assert_eq!(context.n_ubatch(), 32);
}

#[test]
fn ubatch_zero_follows_batch() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.n_ctx = 512;
    options.n_batch = 100;
    options.n_ubatch = 0;
    let context = Context::open(&model, &options).expect("n_ubatch=0");
    assert_eq!(context.n_batch(), 100);
    assert_eq!(context.n_ubatch(), 100);
}

#[test]
fn threads_batch_zero_follows_threads() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.n_threads = 4;
    let context = Context::open(&model, &options).expect("follow");
    assert_eq!(context.n_threads(), 4);
    assert_eq!(context.n_threads_batch(), 4);

    options.n_threads_batch = 2;
    let context = Context::open(&model, &options).expect("explicit");
    assert_eq!(context.n_threads(), 4);
    assert_eq!(context.n_threads_batch(), 2);
}

// -- B. option validation -----------------------------------------------------

#[test]
fn zero_batch_rejected() {
    let Some(model) = load_fixture() else { return };
    // Upstream aborts inside the constructor for n_batch == 0, so this
    // must fail before any native call.
    for ubatch in [0, 512] {
        let mut options = ContextOptions::default();
        options.n_batch = 0;
        options.n_ubatch = ubatch;
        let error = Context::open(&model, &options).expect_err("n_batch=0 must fail");
        assert!(
            error.to_string().contains("n_batch"),
            "error names the bad field: {error}"
        );
    }
}

#[test]
fn seq_max_over_limit_fails_natively() {
    let Some(model) = load_fixture() else { return };
    set_log_quiet(true);
    let mut options = ContextOptions::default();
    options.n_seq_max = 257;
    let error = Context::open(&model, &options).expect_err("n_seq_max=257 must fail");
    set_log_quiet(false);
    assert!(
        error.to_string().starts_with("context error"),
        "native refusal maps to a context error: {error}"
    );
}

#[test]
fn outputs_caps_accepted() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.n_outputs_max = 16;
    options.n_outputs_max_per_seq = 4;
    let mut context = Context::open(&model, &options).expect("output caps");
    context.decode(&tiny_batch()).expect("decode");
    context.logits(3).expect("logits");
}

// -- C. advisory offload flags ------------------------------------------------

#[test]
fn offload_flags_are_advisory_on_cpu() {
    let Some(model) = load_fixture() else { return };
    // Both flags merely permit GPU placement; on CPU-only systems every
    // combination behaves identically. No refusal, no fallback: there is
    // nothing to fall back from.
    for (kqv, op) in [(true, true), (true, false), (false, true), (false, false)] {
        let mut options = ContextOptions::default();
        options.offload_kqv = kqv;
        options.op_offload = op;
        let mut context = Context::open(&model, &options)
            .unwrap_or_else(|e| panic!("open kqv={kqv} op={op}: {e}"));
        context
            .decode(&tiny_batch())
            .unwrap_or_else(|e| panic!("decode kqv={kqv} op={op}: {e}"));
        let logits = context.logits(3).expect("logits");
        assert_eq!(logits.n_vocab(), 32);
    }
}

// -- D. multi-sequence batches ------------------------------------------------

#[test]
fn multi_sequence_batch_decodes() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.n_seq_max = 2;
    let mut context = Context::open(&model, &options).expect("2-seq context");
    let mut builder = BatchBuilder::new(32);
    builder.push_on_sequences(1, 0, &[0], false).unwrap();
    builder.push_on_sequences(2, 0, &[1], false).unwrap();
    builder.push_on_sequences(3, 1, &[0], true).unwrap();
    builder.push_on_sequences(4, 1, &[1], true).unwrap();
    let batch = builder.build().unwrap();
    assert_eq!(batch.sequences(), vec![0, 1]);
    context.decode(&batch).expect("2-seq decode");
    assert_eq!(context.logits(2).expect("logits 2").n_vocab(), 32);
    assert_eq!(context.logits(3).expect("logits 3").n_vocab(), 32);
}

#[test]
fn coupled_token_decodes_with_unified_kv() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.n_seq_max = 2;
    options.kv_unified = true;
    let mut context = Context::open(&model, &options).expect("unified context");
    // Unified KV reports the full context length per sequence.
    assert_eq!(context.n_ctx_seq(), context.n_ctx());
    let mut builder = BatchBuilder::new(32);
    builder.push_on_sequences(9, 0, &[0, 1], true).unwrap();
    context
        .decode(&builder.build().unwrap())
        .expect("coupled decode");
    context.logits(0).expect("logits");
}

#[test]
fn coupled_token_fails_natively_without_unified_kv() {
    let Some(model) = load_fixture() else { return };
    // Without unified KV, upstream cannot place a token shared across
    // sequences: an honest native error, not an abort.
    let mut options = ContextOptions::default();
    options.n_seq_max = 2;
    let mut context = Context::open(&model, &options).expect("split context");
    let mut builder = BatchBuilder::new(32);
    builder.push_on_sequences(9, 0, &[0, 1], true).unwrap();
    set_log_quiet(true);
    let error = context
        .decode(&builder.build().unwrap())
        .expect_err("coupled batch must fail without unified KV");
    set_log_quiet(false);
    assert!(
        error.to_string().starts_with("decode error"),
        "native failure maps to a decode error: {error}"
    );
}

#[test]
fn batch_reuse_across_contexts() {
    let Some(model) = load_fixture() else { return };
    let batch = tiny_batch();
    for _ in 0..2 {
        let mut context = Context::open(&model, &ContextOptions::default()).expect("fresh context");
        context.decode(&batch).expect("reused batch decodes");
    }
}

#[test]
fn oversized_batch_rejected_not_aborted() {
    let Some(model) = load_fixture() else { return };
    // Default context: effective n_batch 64. Upstream aborts on the
    // 65th token; ForgeCore refuses first.
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    assert_eq!(context.n_batch(), 64);
    let error = context
        .decode(&long_batch(65))
        .expect_err("65 tokens must exceed n_batch 64");
    assert!(
        error.to_string().contains("n_batch"),
        "error names the limit: {error}"
    );
    // The context survives the refusal.
    context.decode(&tiny_batch()).expect("decode after refusal");
}

#[test]
fn bad_seq_rejected() {
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    assert_eq!(context.n_seq_max(), 1);
    let mut builder = BatchBuilder::new(32);
    builder.push_on_sequences(1, 0, &[1], true).unwrap();
    let error = context
        .decode(&builder.build().unwrap())
        .expect_err("seq 1 must exceed n_seq_max 1");
    let message = error.to_string();
    assert!(
        message.contains("seq id 1"),
        "error names the bad sequence: {message}"
    );
}

#[test]
fn gapped_positions_fail_natively() {
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    let mut builder = BatchBuilder::new(32);
    builder.push(1, 0, false).unwrap();
    builder.push(2, 2, true).unwrap();
    set_log_quiet(true);
    let error = context
        .decode(&builder.build().unwrap())
        .expect_err("position gap must fail");
    set_log_quiet(false);
    assert!(
        error.to_string().starts_with("decode error"),
        "native failure maps to a decode error: {error}"
    );
}

// -- E. embeddings ------------------------------------------------------------

#[test]
fn model_embedding_facts() {
    let Some(model) = load_fixture() else { return };
    assert_eq!(model.n_embd_inp().expect("n_embd_inp"), 8);
    assert_eq!(model.n_embd_out().expect("n_embd_out"), 8);
    assert_eq!(model.n_cls_out(), 1);
    assert!(!model.has_encoder());
}

#[test]
fn embedding_batch_decodes_on_plain_context() {
    let Some(model) = load_fixture() else { return };
    let n_embd = model.n_embd_inp().expect("n_embd_inp") as usize;
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    let mut builder = BatchBuilder::new_embeddings(n_embd).unwrap();
    for i in 0..4 {
        let row = vec![0.01 * i as f32; n_embd];
        builder.push_embd(&row, i, i == 3).unwrap();
    }
    let batch = builder.build().unwrap();
    assert!(batch.is_embd());
    context.decode(&batch).expect("embd batch decodes");
    let logits = context.logits(3).expect("logits from embd batch");
    assert_eq!(logits.n_vocab(), 32);
}

#[test]
fn embeddings_require_embeddings_context() {
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    context.decode(&tiny_batch()).expect("decode");
    let error = context
        .embeddings(3)
        .expect_err("plain context has no rows");
    assert!(
        error.to_string().starts_with("embeddings error"),
        "missing rows map to an embeddings error: {error}"
    );
}

#[test]
fn embeddings_context_yields_rows() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.embeddings = true;
    let mut context = Context::open(&model, &options).expect("embeddings context");
    context.decode(&tiny_batch()).expect("decode");
    let embd = context.embeddings(3).expect("token row");
    assert_eq!(embd.width(), 8);
    assert_eq!(embd.values().len(), 8);
    // Logits are still extracted alongside.
    assert_eq!(context.logits(3).expect("logits").n_vocab(), 32);
}

#[test]
fn pooled_embeddings_per_sequence() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.embeddings = true;
    options.pooling = PoolingType::Mean;
    options.n_seq_max = 2;
    let mut context = Context::open(&model, &options).expect("pooled context");
    assert_eq!(context.pooling(), PoolingType::Mean);
    let mut builder = BatchBuilder::new(32);
    builder.push_on_sequences(1, 0, &[0], true).unwrap();
    builder.push_on_sequences(2, 0, &[1], true).unwrap();
    builder.push_on_sequences(3, 1, &[0], true).unwrap();
    builder.push_on_sequences(4, 1, &[1], true).unwrap();
    context.decode(&builder.build().unwrap()).expect("decode");
    for seq in [0, 1] {
        let pooled = context.embeddings_seq(seq).expect("pooled row");
        assert_eq!(pooled.width(), 8);
    }
}

#[test]
fn pooling_variants_decode() {
    let Some(model) = load_fixture() else { return };
    for pooling in [PoolingType::None, PoolingType::Cls, PoolingType::Last] {
        let mut options = ContextOptions::default();
        options.embeddings = true;
        options.pooling = pooling;
        let mut context =
            Context::open(&model, &options).unwrap_or_else(|e| panic!("open {pooling:?}: {e}"));
        assert_eq!(context.pooling(), pooling);
        context
            .decode(&tiny_batch())
            .unwrap_or_else(|e| panic!("decode {pooling:?}: {e}"));
        if pooling != PoolingType::None {
            context.embeddings_seq(0).expect("pooled row");
        }
    }
}

#[test]
fn seq_embeddings_require_pooling() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.embeddings = true;
    let mut context = Context::open(&model, &options).expect("embeddings context");
    context.decode(&tiny_batch()).expect("decode");
    let error = context
        .embeddings_seq(0)
        .expect_err("unpooled context has no seq rows");
    assert!(
        error.to_string().starts_with("embeddings error"),
        "missing rows map to an embeddings error: {error}"
    );
}

// -- F. threads, causal, flash, synchronize -----------------------------------

#[test]
fn set_n_threads_mid_life() {
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    context.decode(&tiny_batch()).expect("decode");
    context.set_n_threads(2, 3).expect("set threads");
    assert_eq!(context.n_threads(), 2);
    assert_eq!(context.n_threads_batch(), 3);
    context
        .decode(&tiny_batch_at(4))
        .expect("decode after rethread");
    let error = context
        .set_n_threads(0, 1)
        .expect_err("zero threads must fail");
    assert!(error.to_string().contains("thread"));
    let error = context
        .set_n_threads(1, 0)
        .expect_err("zero batch threads must fail");
    assert!(error.to_string().contains("thread"));
}

#[test]
fn noncausal_context_decodes_small_batches() {
    let Some(model) = load_fixture() else { return };
    let mut options = ContextOptions::default();
    options.attention = AttentionType::NonCausal;
    let mut context = Context::open(&model, &options).expect("non-causal context");
    assert!(!context.causal_attn());
    // Non-causal skips the n_ctx batch clamp: verbatim request.
    assert_eq!(context.n_batch(), 2048);
    assert_eq!(context.n_ubatch(), 512);
    context
        .decode(&tiny_batch())
        .expect("small non-causal decode");
}

#[test]
fn noncausal_oversized_vs_ubatch_rejected() {
    let Some(model) = load_fixture() else { return };
    // 513 tokens fits n_batch (2048) but not n_ubatch (512): upstream
    // aborts, so ForgeCore refuses first.
    let mut options = ContextOptions::default();
    options.attention = AttentionType::NonCausal;
    let mut context = Context::open(&model, &options).expect("non-causal context");
    let error = context
        .decode(&long_batch(513))
        .expect_err("513 tokens must exceed n_ubatch 512");
    assert!(
        error.to_string().contains("n_ubatch"),
        "error names the limit: {error}"
    );
}

#[test]
fn set_causal_attn_flips_and_validates() {
    let Some(model) = load_fixture() else { return };
    // n_batch 128 vs n_ubatch 32: the two decode limits are
    // distinguishable, so the flipped ubatch arm is really exercised.
    let mut options = ContextOptions::default();
    options.n_ctx = 512;
    options.n_batch = 128;
    options.n_ubatch = 32;
    let mut context = Context::open(&model, &options).expect("context");
    assert!(context.causal_attn());
    context.decode(&tiny_batch()).expect("decode while causal");
    context.set_causal_attn(false);
    assert!(!context.causal_attn());
    context
        .decode(&tiny_batch_at(4))
        .expect("decode while non-causal");
    // 65 tokens fits n_batch (128) but not n_ubatch (32). The refusal
    // touches no KV state, so positions need no advance.
    let error = context
        .decode(&long_batch(65))
        .expect_err("65 tokens must exceed n_ubatch 32");
    assert!(error.to_string().contains("n_ubatch"));
    context.set_causal_attn(true);
    assert!(context.causal_attn());
    context
        .decode(&tiny_batch_at(8))
        .expect("decode after flip back");
}

#[test]
fn flash_variants_decode() {
    let Some(model) = load_fixture() else { return };
    for flash in [FlashAttnType::Disabled, FlashAttnType::Enabled] {
        let mut options = ContextOptions::default();
        options.flash_attn = flash;
        let mut context =
            Context::open(&model, &options).unwrap_or_else(|e| panic!("open {flash:?}: {e}"));
        context
            .decode(&tiny_batch())
            .unwrap_or_else(|e| panic!("decode {flash:?}: {e}"));
        context.logits(3).expect("logits");
    }
}

#[test]
fn synchronize_is_safe() {
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &ContextOptions::default()).expect("context");
    context.synchronize();
    context.decode(&tiny_batch()).expect("decode");
    context.synchronize();
    context.logits(3).expect("logits");
}

// -- G. skip-gated GPU smoke ---------------------------------------------------

#[test]
fn gpu_context_smoke() {
    if !supports_gpu_offload() {
        println!("SKIP: no GPU device in this environment");
        return;
    }
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &ContextOptions::default()).expect("gpu context");
    context.decode(&tiny_batch()).expect("gpu decode");
    println!(
        "gpu smoke: n_ctx={} n_batch={} n_ubatch={} causal={} pooling={:?}",
        context.n_ctx(),
        context.n_batch(),
        context.n_ubatch(),
        context.causal_attn(),
        context.pooling(),
    );
    let mut options = ContextOptions::default();
    options.offload_kqv = false;
    options.op_offload = false;
    let mut context = Context::open(&model, &options).expect("cpu-pinned context");
    context.decode(&tiny_batch()).expect("cpu-pinned decode");
    println!("gpu smoke: offload flags disabled, decode OK");
}
