//! Phase-5 KV/state primitives: memory handle, sequence surgery,
//! position shifts, state snapshots, and KV cache dtypes.
//!
//! Fixture tests SKIP without `FORGE_TEST_MODEL`; the GPU smoke test
//! additionally SKIPs without `supports_gpu_offload`. Nothing here
//! assumes a GPU.

use forge_core::model::set_log_quiet;
use forge_core::{
    supports_gpu_offload, BatchBuilder, Context, ContextOptions, DType, Model, SeqState, State,
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

fn quiet() {
    set_log_quiet(true);
}

/// Four tokens on sequence 0 at positions 0..4, logits on the last.
fn tiny_batch() -> forge_core::Batch {
    let mut builder = BatchBuilder::new(32);
    builder.push(1, 0, false).unwrap();
    builder.push(5, 1, false).unwrap();
    builder.push(6, 2, false).unwrap();
    builder.push(7, 3, true).unwrap();
    builder.build().unwrap()
}

/// Same tokens shifted to start at `pos0`.
fn tiny_batch_at(pos0: u32) -> forge_core::Batch {
    let mut builder = BatchBuilder::new(32);
    builder.push(1, pos0, false).unwrap();
    builder.push(5, pos0 + 1, false).unwrap();
    builder.push(6, pos0 + 2, false).unwrap();
    builder.push(7, pos0 + 3, true).unwrap();
    builder.build().unwrap()
}

/// Two tokens on `seq` at positions 0..2, logits on the last.
fn seq_batch(seq: u32) -> forge_core::Batch {
    let mut builder = BatchBuilder::new(32);
    builder.push_on_sequences(5, 0, &[seq], false).unwrap();
    builder.push_on_sequences(6, 1, &[seq], true).unwrap();
    builder.build().unwrap()
}

fn two_seq_options() -> ContextOptions {
    let mut options = ContextOptions::default();
    options.n_ctx = 64;
    options.n_seq_max = 2;
    options
}

fn unified_options() -> ContextOptions {
    let mut options = two_seq_options();
    options.kv_unified = true;
    options
}

fn default_options() -> ContextOptions {
    let mut options = ContextOptions::default();
    options.n_ctx = 64;
    options
}

// -- A. handle acquisition + position queries ---------------------------------

#[test]
fn memory_handle_reports_split_seq_limit() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    let memory = context.memory().expect("tiny model has memory");
    assert_eq!(memory.seq_limit(), 1);
    assert!(memory.can_shift());
}

#[test]
fn unified_memory_reports_parallel_seq_limit() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &unified_options()).expect("open");
    let memory = context.memory().expect("tiny model has memory");
    assert_eq!(memory.seq_limit(), 256);
}

#[test]
fn pos_queries_empty_then_filled() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    {
        let memory = context.memory().expect("memory");
        assert_eq!(memory.pos_min(0).unwrap(), None);
        assert_eq!(memory.pos_max(0).unwrap(), None);
    }
    context.decode(&tiny_batch()).expect("decode");
    let memory = context.memory().expect("memory");
    assert_eq!(memory.pos_min(0).unwrap(), Some(0));
    assert_eq!(memory.pos_max(0).unwrap(), Some(3));
}

#[test]
fn invalid_seq_ids_are_refused() {
    quiet();
    let Some(model) = load_fixture() else { return };
    // Split cache (limit 1): every op refuses seq >= 1.
    {
        let mut context = Context::open(&model, &default_options()).expect("open");
        let mut memory = context.memory().expect("memory");
        for seq in [1, 5, u32::MAX] {
            assert!(memory.remove_range(seq, 0, None).is_err(), "rm {seq}");
            assert!(memory.copy_seq(0, seq).is_err(), "cp dst {seq}");
            assert!(memory.copy_seq(seq, 0).is_err(), "cp src {seq}");
            assert!(memory.keep_seq(seq).is_err(), "keep {seq}");
            assert!(
                memory.shift_positions(seq, 0, None, 1).is_err(),
                "shift {seq}"
            );
            assert!(
                memory.scale_positions(seq, 0, None, 2).is_err(),
                "scale {seq}"
            );
            assert!(memory.pos_min(seq).is_err(), "min {seq}");
            assert!(memory.pos_max(seq).is_err(), "max {seq}");
        }
    }
    // Unified cache (limit 256): 300 refuses everywhere (would abort
    // natively — verified SIGABRT).
    let mut context = Context::open(&model, &unified_options()).expect("open");
    {
        let mut memory = context.memory().expect("memory");
        for seq in [256, 300, u32::MAX] {
            assert!(memory.remove_range(seq, 0, None).is_err(), "rm {seq}");
            assert!(memory.pos_max(seq).is_err(), "max {seq}");
        }
    }
    assert!(context.seq_state_size(300).is_err());
    assert!(context.export_seq_state(300).is_err());
}

#[test]
fn tight_bound_applies_to_keep_and_seq_restore() {
    quiet();
    let Some(model) = load_fixture() else { return };
    // Unified with n_seq_max = 1: general ops accept high seqs, but
    // keep/restore require seq < n_seq_max (DSV4/recurrent asserts).
    let mut options = default_options();
    options.kv_unified = true;
    let mut context = Context::open(&model, &options).expect("open");
    context.decode(&seq_batch(0)).expect("decode");
    {
        let mut memory = context.memory().expect("memory");
        assert_eq!(memory.seq_limit(), 256);
        assert_eq!(memory.pos_max(200).unwrap(), None);
        memory.remove_range(200, 0, None).expect("rm high seq");
        assert!(memory.keep_seq(200).is_err(), "keep refuses high seq");
    }
    let snap = context.export_seq_state(0).expect("export seq");
    assert!(context.import_seq_state(200, &snap).is_err());
}

// -- B. clear / remove / copy / keep ------------------------------------------

#[test]
fn clear_empties_cache_and_context_reuses() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    context.memory().expect("memory").clear(false);
    {
        let memory = context.memory().expect("memory");
        assert_eq!(memory.pos_max(0).unwrap(), None);
    }
    // Same positions decode again after the clear.
    context.decode(&tiny_batch()).expect("re-decode");
    let memory = context.memory().expect("memory");
    assert_eq!(memory.pos_max(0).unwrap(), Some(3));
}

#[test]
fn clear_with_data_flag_matches() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    context.memory().expect("memory").clear(true);
    let memory = context.memory().expect("memory");
    assert_eq!(memory.pos_min(0).unwrap(), None);
    assert_eq!(memory.pos_max(0).unwrap(), None);
}

#[test]
fn remove_full_sequence_leaves_other_intact() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &two_seq_options()).expect("open");
    context.decode(&seq_batch(0)).expect("decode 0");
    context.decode(&seq_batch(1)).expect("decode 1");
    let mut memory = context.memory().expect("memory");
    memory.remove_range(0, 0, None).expect("rm seq 0");
    assert_eq!(memory.pos_max(0).unwrap(), None);
    assert_eq!(memory.pos_min(1).unwrap(), Some(0));
    assert_eq!(memory.pos_max(1).unwrap(), Some(1));
}

#[test]
fn remove_partial_range() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    let mut memory = context.memory().expect("memory");
    memory.remove_range(0, 1, Some(3)).expect("rm [1, 3)");
    // Positions 0 and 3 survive.
    assert_eq!(memory.pos_min(0).unwrap(), Some(0));
    assert_eq!(memory.pos_max(0).unwrap(), Some(3));
}

#[test]
fn remove_nonexistent_range_is_ok() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    {
        let mut memory = context.memory().expect("memory");
        memory.remove_range(0, 100, Some(200)).expect("rm on empty");
    }
    context.decode(&tiny_batch()).expect("decode");
    let mut memory = context.memory().expect("memory");
    memory
        .remove_range(0, 100, Some(200))
        .expect("rm beyond end");
    assert_eq!(memory.pos_max(0).unwrap(), Some(3));
}

#[test]
fn copy_seq_duplicates_onto_unified() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &unified_options()).expect("open");
    context.decode(&seq_batch(0)).expect("decode");
    let mut memory = context.memory().expect("memory");
    memory.copy_seq(0, 1).expect("copy");
    assert_eq!(memory.pos_min(1).unwrap(), Some(0));
    assert_eq!(memory.pos_max(1).unwrap(), Some(1));
    // Self-copy is a no-op.
    memory.copy_seq(1, 1).expect("self copy");
    assert_eq!(memory.pos_max(1).unwrap(), Some(1));
}

#[test]
fn copy_seq_full_across_split_streams() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &two_seq_options()).expect("open");
    context.decode(&seq_batch(0)).expect("decode");
    let mut memory = context.memory().expect("memory");
    // Cross-stream copies require the full buffer (partial aborts
    // natively — verified SIGABRT); `copy_seq` is full by design.
    memory.copy_seq(0, 1).expect("cross-stream copy");
    assert_eq!(memory.pos_min(1).unwrap(), Some(0));
    assert_eq!(memory.pos_max(1).unwrap(), Some(1));
}

#[test]
fn keep_seq_isolates_one_sequence() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &unified_options()).expect("open");
    context.decode(&seq_batch(0)).expect("decode 0");
    context.decode(&seq_batch(1)).expect("decode 1");
    let mut memory = context.memory().expect("memory");
    memory.keep_seq(1).expect("keep");
    assert_eq!(memory.pos_max(0).unwrap(), None);
    assert_eq!(memory.pos_min(1).unwrap(), Some(0));
    assert_eq!(memory.pos_max(1).unwrap(), Some(1));
}

#[test]
fn inverted_ranges_are_refused() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    let mut memory = context.memory().expect("memory");
    assert!(memory.remove_range(0, 5, Some(2)).is_err());
    assert!(memory.shift_positions(0, 5, Some(2), 1).is_err());
    assert!(memory.scale_positions(0, 5, Some(2), 2).is_err());
    // Empty (start == end) ranges are honest no-ops, not errors.
    memory.remove_range(0, 2, Some(2)).expect("empty rm");
}

#[test]
fn unrepresentable_positions_are_refused() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    let mut memory = context.memory().expect("memory");
    assert!(memory.remove_range(0, u32::MAX, None).is_err());
    assert!(memory.remove_range(0, 0, Some(u32::MAX)).is_err());
    assert!(memory.shift_positions(0, u32::MAX, None, 1).is_err());
}

// -- C. shifts and scales ------------------------------------------------------

#[test]
fn shift_positions_moves_range() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    let mut memory = context.memory().expect("memory");
    memory.shift_positions(0, 0, None, 10).expect("shift +10");
    assert_eq!(memory.pos_min(0).unwrap(), Some(10));
    assert_eq!(memory.pos_max(0).unwrap(), Some(13));
}

#[test]
fn shift_partial_negative() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    let mut memory = context.memory().expect("memory");
    memory
        .shift_positions(0, 1, Some(3), -1)
        .expect("shift [1,3) by -1");
    // Cells were 0,1,2,3 and are now 0,0,1,3.
    assert_eq!(memory.pos_min(0).unwrap(), Some(0));
    assert_eq!(memory.pos_max(0).unwrap(), Some(3));
}

#[test]
fn shift_zero_and_empty_are_noops() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    let mut memory = context.memory().expect("memory");
    memory
        .shift_positions(0, 0, None, 0)
        .expect("zero shift on empty");
    memory
        .shift_positions(0, 0, None, 100)
        .expect("shift on empty");
    assert_eq!(memory.pos_max(0).unwrap(), None);
}

#[test]
fn shift_overflow_is_refused() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    let mut memory = context.memory().expect("memory");
    // max is 3: 3 + i32::MAX overflows.
    assert!(memory.shift_positions(0, 0, None, i32::MAX).is_err());
}

#[test]
fn shift_backstop_is_refused() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    let mut memory = context.memory().expect("memory");
    assert!(memory.shift_positions(0, 0, None, 1 << 30).is_err());
    assert!(memory.shift_positions(0, 0, None, -(1 << 30)).is_err());
    // Just inside the backstop is fine (and frees nothing here).
    memory
        .shift_positions(0, 0, None, (1 << 30) - 1)
        .expect("max shift");
    assert_eq!(memory.pos_min(0).unwrap(), Some((1 << 30) - 1));
}

#[test]
fn scale_positions_divides() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    let mut memory = context.memory().expect("memory");
    memory.scale_positions(0, 0, None, 1).expect("div by 1");
    assert_eq!(memory.pos_max(0).unwrap(), Some(3));
    memory.scale_positions(0, 0, None, 2).expect("div by 2");
    assert_eq!(memory.pos_min(0).unwrap(), Some(0));
    assert_eq!(memory.pos_max(0).unwrap(), Some(1));
}

#[test]
fn scale_positions_rejects_bad_divisors() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    let mut memory = context.memory().expect("memory");
    // Division by zero is SIGFPE natively (verified) — never called.
    assert!(memory.scale_positions(0, 0, None, 0).is_err());
    assert!(memory.scale_positions(0, 0, None, u32::MAX).is_err());
}

#[test]
fn repeated_ops_are_stable() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &unified_options()).expect("open");
    context.decode(&seq_batch(0)).expect("decode");
    let mut memory = context.memory().expect("memory");
    memory.remove_range(0, 0, Some(1)).expect("rm");
    memory.remove_range(0, 0, Some(1)).expect("rm again");
    memory.copy_seq(0, 1).expect("copy");
    memory.copy_seq(0, 1).expect("copy again");
    memory.shift_positions(1, 0, None, 5).expect("shift");
    memory.shift_positions(1, 0, None, 5).expect("shift again");
    assert_eq!(memory.pos_min(1).unwrap(), Some(11));
    memory.keep_seq(1).expect("keep");
    memory.keep_seq(1).expect("keep again");
    memory.clear(false);
    memory.clear(true);
    assert_eq!(memory.pos_max(1).unwrap(), None);
}

// -- D. state snapshots ----------------------------------------------------------

#[test]
fn state_size_empty_and_filled() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    // Probed: arch tag + stream headers, no cells.
    assert_eq!(context.state_size().unwrap(), 17);
    context.decode(&tiny_batch()).expect("decode");
    let filled = context.state_size().unwrap();
    assert!(filled > 17, "filled state grows: {filled}");
    let snap = context.export_state().expect("export");
    assert_eq!(snap.len() as u64, filled);
}

#[test]
fn state_roundtrip_restores_logits_bit_exact() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    let snap = context.export_state().expect("export");
    context.memory().expect("memory").clear(true);
    context.import_state(&snap).expect("import");
    context.decode(&tiny_batch_at(4)).expect("continue");
    let restored = context.logits(3).expect("logits").values().to_vec();

    let mut control = Context::open(&model, &default_options()).expect("open");
    control.decode(&tiny_batch()).expect("decode");
    control.decode(&tiny_batch_at(4)).expect("continue");
    let expected = control.logits(3).expect("logits").values().to_vec();
    assert_eq!(restored, expected);
}

#[test]
fn state_import_cross_context() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut first = Context::open(&model, &default_options()).expect("open");
    first.decode(&tiny_batch()).expect("decode");
    let snap = first.export_state().expect("export");
    drop(first);
    let mut second = Context::open(&model, &default_options()).expect("open");
    second.import_state(&snap).expect("import");
    let memory = second.memory().expect("memory");
    assert_eq!(memory.pos_min(0).unwrap(), Some(0));
    assert_eq!(memory.pos_max(0).unwrap(), Some(3));
}

#[test]
fn state_corrupt_header_rejected_data_flip_accepted() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    let snap = context.export_state().expect("export");
    // Structural corruption (arch tag) is rejected.
    let mut bad = snap.as_bytes().to_vec();
    bad[2] ^= 0xff;
    assert!(context.import_state(&State::from_bytes(bad)).is_err());
    // Native states carry no checksums: flips inside KV data rows are
    // structurally valid and accepted (verified by probe). The last
    // byte of this snapshot is V-row data.
    let mut soft = snap.as_bytes().to_vec();
    let last = soft.len() - 1;
    soft[last] ^= 0xff;
    context
        .import_state(&State::from_bytes(soft))
        .expect("data flip accepted");
}

#[test]
fn state_truncated_empty_and_trailing_rejected() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    let snap = context.export_state().expect("export");
    let bytes = snap.as_bytes();
    assert!(context
        .import_state(&State::from_bytes(bytes[..bytes.len() / 2].to_vec()))
        .is_err());
    assert!(context.import_state(&State::from_bytes(vec![])).is_err());
    let mut trailing = bytes.to_vec();
    trailing.push(0xAA);
    assert!(
        context.import_state(&State::from_bytes(trailing)).is_err(),
        "trailing garbage rejected"
    );
}

#[test]
fn seq_state_roundtrip_and_retarget() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &two_seq_options()).expect("open");
    context.decode(&seq_batch(0)).expect("decode 0");
    context.decode(&seq_batch(1)).expect("decode 1");
    assert_eq!(
        context.seq_state_size(1).unwrap() as usize,
        context.export_seq_state(1).expect("export").len()
    );
    let snap = context.export_seq_state(1).expect("export seq 1");
    context.memory().expect("memory").clear(true);
    // Restore onto a different sequence id.
    context.import_seq_state(0, &snap).expect("import as seq 0");
    let memory = context.memory().expect("memory");
    assert_eq!(memory.pos_min(0).unwrap(), Some(0));
    assert_eq!(memory.pos_max(0).unwrap(), Some(1));
    assert_eq!(memory.pos_max(1).unwrap(), None);
}

#[test]
fn seq_state_empty_seq_roundtrip() {
    quiet();
    let Some(model) = load_fixture() else { return };
    // Unified (single stream): probed layout is magic + seq id +
    // stream/count headers with no cells = 16 bytes.
    let mut context = Context::open(&model, &unified_options()).expect("open");
    context.decode(&seq_batch(0)).expect("decode");
    assert_eq!(context.seq_state_size(1).unwrap(), 16);
    let snap = context.export_seq_state(1).expect("export empty");
    assert_eq!(snap.len(), 16);
    context.import_seq_state(1, &snap).expect("import empty");
    let memory = context.memory().expect("memory");
    assert_eq!(memory.pos_max(1).unwrap(), None);
}

#[test]
fn seq_state_bad_magic_rejected() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &two_seq_options()).expect("open");
    context.decode(&seq_batch(0)).expect("decode");
    let snap = context.export_seq_state(0).expect("export");
    let mut bad = snap.as_bytes().to_vec();
    bad[0] ^= 0xff;
    assert!(context
        .import_seq_state(0, &SeqState::from_bytes(bad))
        .is_err());
    assert!(context
        .import_seq_state(0, &SeqState::from_bytes(vec![]))
        .is_err());
}

// -- E. KV cache dtypes ------------------------------------------------------------

#[test]
fn f32_kv_context_decodes_and_grows_state() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut options = default_options();
    options.type_k = DType::F32;
    options.type_v = DType::F32;
    let mut context = Context::open(&model, &options).expect("f32 context");
    context.decode(&tiny_batch()).expect("decode");
    let f32_size = context.state_size().unwrap();
    let mut base = Context::open(&model, &default_options()).expect("open");
    base.decode(&tiny_batch()).expect("decode");
    let f16_size = base.state_size().unwrap();
    // Wider rows serialize to strictly more bytes: the option applied.
    assert!(f32_size > f16_size, "{f32_size} > {f16_size}");
}

#[test]
fn invalid_kv_dtype_combination_fails_open() {
    quiet();
    let Some(model) = load_fixture() else { return };
    // Q8_0 blocks (32) do not divide the tiny head width (4): native
    // refuses creation (verified by probe).
    let mut options = default_options();
    options.type_k = DType::Q8_0;
    assert!(Context::open(&model, &options).is_err());
    // Quantized V cache without flash attention is refused natively.
    let mut options = default_options();
    options.type_v = DType::Q8_0;
    options.flash_attn = forge_core::FlashAttnType::Disabled;
    assert!(Context::open(&model, &options).is_err());
}

// -- F. decode interaction + lifecycle ------------------------------------------------

#[test]
fn decode_after_rm_refills_positions() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    context
        .memory()
        .expect("memory")
        .remove_range(0, 0, None)
        .expect("rm all");
    context.decode(&tiny_batch()).expect("refill");
    let memory = context.memory().expect("memory");
    assert_eq!(memory.pos_min(0).unwrap(), Some(0));
    assert_eq!(memory.pos_max(0).unwrap(), Some(3));
}

#[test]
fn decode_continues_after_shift() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("open");
    context.decode(&tiny_batch()).expect("decode");
    context
        .memory()
        .expect("memory")
        .shift_positions(0, 0, None, 100)
        .expect("shift");
    // Fresh positions past the shifted window decode normally.
    context.decode(&tiny_batch_at(104)).expect("continue");
    let memory = context.memory().expect("memory");
    assert_eq!(memory.pos_max(0).unwrap(), Some(107));
}

#[test]
fn context_drop_and_reopen_with_state() {
    quiet();
    let Some(model) = load_fixture() else { return };
    let snap = {
        let mut context = Context::open(&model, &default_options()).expect("open");
        context.decode(&tiny_batch()).expect("decode");
        context.export_state().expect("export")
    };
    let mut reopened = Context::open(&model, &default_options()).expect("reopen");
    reopened.import_state(&snap).expect("import");
    let memory = reopened.memory().expect("memory");
    assert_eq!(memory.pos_max(0).unwrap(), Some(3));
}

#[test]
fn gpu_state_smoke() {
    quiet();
    if !supports_gpu_offload() {
        println!("SKIP: no GPU device in this environment");
        return;
    }
    let Some(model) = load_fixture() else { return };
    let mut context = Context::open(&model, &default_options()).expect("gpu context");
    context.decode(&tiny_batch()).expect("gpu decode");
    {
        let mut memory = context.memory().expect("gpu memory");
        assert_eq!(memory.pos_max(0).unwrap(), Some(3));
        memory.shift_positions(0, 0, None, 4).expect("gpu shift");
    }
    let snap = context.export_state().expect("gpu export");
    context.memory().expect("memory").clear(true);
    context.import_state(&snap).expect("gpu import");
    println!(
        "gpu smoke: state_size={} pos_max={:?}",
        context.state_size().unwrap(),
        context.memory().expect("memory").pos_max(0).unwrap(),
    );
}
