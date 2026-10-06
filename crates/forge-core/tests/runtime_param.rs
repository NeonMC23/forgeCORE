//! Runtime, tensor-init, and model-param surface.
//!
//! Three groups: (A) the CPU runtime substrate every suite stands on
//! (`Backend::open_cpu`, thread-count validation), (B) tensor
//! initialization and host IO (`from_*`, `upload_*`, `to_vec_*`,
//! `fill_*`, `copy_into` — including the stride-aware footprint
//! semantics of the `_bytes` path and the exact rejection rules),
//! (C) the model parameter surface (`n_params`, `size_bytes`, and the
//! metadata getters) pinned against the deterministic tiny-llama
//! fixture (`scripts/make-tiny-gguf.py`: vocab=32, embd=8, heads=2,
//! kv=2, layers=1, ff=16, ctx=64).
//!
//! CPU-only, single-threaded for determinism. Model tests need
//! `FORGE_TEST_MODEL` and report SKIP otherwise.

use forge_core::model::set_log_quiet;
use forge_core::{Backend, DType, DeviceType, Model, Tensor};
use std::path::PathBuf;

fn open_test_cpu() -> Backend {
    let backend = Backend::open_cpu().expect("CPU backend must open");
    backend
        .set_cpu_threads(1)
        .expect("single-threaded CPU for determinism");
    backend
}

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

// -- A. runtime substrate ---------------------------------------------------

#[test]
fn cpu_backend_open_reports_cpu() {
    let backend = open_test_cpu();
    assert!(backend.is_cpu(), "open_cpu must yield a CPU backend");
    assert_eq!(backend.device_type(), DeviceType::Cpu);
    assert!(!backend.name().is_empty(), "backend name must be set");
    let info = backend.device().expect("CPU backend resolves a device");
    assert_eq!(info.device_type, DeviceType::Cpu);
    backend.synchronize(); // total no-op on CPU; must not hang or abort
}

#[test]
fn set_cpu_threads_validates() {
    let backend = open_test_cpu();
    assert!(backend.set_cpu_threads(0).is_err(), "zero threads rejected");
    assert!(
        backend.set_cpu_threads(u32::MAX).is_err(),
        "thread count past c_int range rejected"
    );
    assert!(backend.set_cpu_threads(1).is_ok());
    assert!(backend.set_cpu_threads(4).is_ok());
}

// -- B. tensor init and host IO ---------------------------------------------

#[test]
fn from_f32_round_trip_exact() {
    let backend = open_test_cpu();
    let data = vec![-3.5, -0.0, 0.0, 1.25, 100.0, 65504.0];
    let t = Tensor::from_f32(&backend, &[2, 3], &data).expect("create");
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.nelements(), 6);
    assert_eq!(t.nbytes(), 24);
    assert_eq!(t.to_vec_f32().expect("download"), data);
}

#[test]
fn from_i32_round_trip_exact() {
    let backend = open_test_cpu();
    let data = vec![-7, 0, 1, 31, i32::MIN, i32::MAX];
    let t = Tensor::from_i32(&backend, &[6], &data).expect("create");
    assert_eq!(t.dtype(), DType::I32);
    assert_eq!(t.nelements(), 6);
    assert_eq!(t.nbytes(), 24);
    assert_eq!(t.to_vec_i32().expect("download"), data);
}

#[test]
fn from_bytes_f32_preserves_bits() {
    let backend = open_test_cpu();
    // Bit patterns chosen so any float-domain touching (vs verbatim
    // bytes) would show: -0.0, subnormal min, +inf, signalling NaN bits.
    let words = [0x8000_0000u32, 0x0000_0001, 0x7F80_0000, 0x7FA0_0000];
    let mut bytes = Vec::with_capacity(16);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    let t = Tensor::from_bytes(&backend, DType::F32, &[4], &bytes).expect("create");
    assert_eq!(t.to_bytes().expect("download"), bytes);
    let back = t.to_vec_f32().expect("f32 download");
    for (got, want) in back.iter().zip(words) {
        assert_eq!(got.to_bits(), want, "bit-exact round-trip");
    }
}

#[test]
fn from_bytes_quant_zeroed_blocks_round_trip() {
    let backend = open_test_cpu();
    // Block = 32 elements: Q8_0 [64] is two 34-byte blocks (68
    // bytes); Q4_0 [32] is one 18-byte block. Zeroed blocks are valid
    // input per `from_bytes`.
    for (dtype, ne0, block) in [(DType::Q8_0, 64, 68), (DType::Q4_0, 32, 18)] {
        let bytes = vec![0u8; block];
        let t = Tensor::from_bytes(&backend, dtype, &[ne0], &bytes).expect("create");
        assert_eq!(t.dtype(), dtype);
        assert_eq!(t.nbytes(), block);
        assert_eq!(t.to_bytes().expect("download"), bytes);
    }
}

#[test]
fn from_constructors_reject_bad_lengths() {
    let backend = open_test_cpu();
    assert!(
        Tensor::from_f32(&backend, &[2, 3], &[0.0; 5]).is_err(),
        "short F32"
    );
    assert!(
        Tensor::from_f32(&backend, &[2, 3], &[0.0; 7]).is_err(),
        "long F32"
    );
    assert!(
        Tensor::from_i32(&backend, &[4], &[0; 3]).is_err(),
        "short I32"
    );
    assert!(
        Tensor::from_bytes(&backend, DType::F32, &[4], &[0u8; 15]).is_err(),
        "short bytes"
    );
    assert!(
        Tensor::from_bytes(&backend, DType::F32, &[4], &[0u8; 17]).is_err(),
        "long bytes"
    );
}

#[test]
fn empty_uninit_then_upload_or_fill_is_deterministic() {
    let backend = open_test_cpu();
    // `empty` leaves contents undefined (never asserted here); the
    // first write makes the tensor deterministic.
    let data = vec![1.0f32, -2.0, 3.0, -4.0];
    let t = Tensor::empty(&backend, DType::F32, &[2, 2]).expect("empty");
    t.upload_f32(&data).expect("upload");
    assert_eq!(t.to_vec_f32().expect("download"), data);

    let u = Tensor::empty(&backend, DType::F32, &[3]).expect("empty");
    u.fill_f32(-17.25).expect("fill");
    assert_eq!(u.to_vec_f32().expect("download"), vec![-17.25; 3]);

    let v = Tensor::empty(&backend, DType::I32, &[2]).expect("empty");
    v.upload_i32(&[9, -9]).expect("upload");
    assert_eq!(v.to_vec_i32().expect("download"), vec![9, -9]);
}

#[test]
fn fill_bytes_covers_stride_aware_span() {
    let backend = open_test_cpu();
    let parent = Tensor::from_f32(&backend, &[4, 3], &[0.0; 12]).expect("parent");
    // Rows 0 and 2 with a one-row gap: span 48, packed 32.
    let view = parent.view_2d([4, 2], 32, 0).expect("strided view");
    assert_eq!(view.nbytes(), 48);
    view.fill_bytes(0xA5).expect("span fill");
    assert_eq!(view.to_bytes().expect("view bytes"), vec![0xA5; 48]);
    // The span covers the whole parent, so the parent is all fill too.
    assert_eq!(parent.to_bytes().expect("parent bytes"), vec![0xA5; 48]);
}

#[test]
fn upload_rejects_wrong_dtype_strides_len() {
    let backend = open_test_cpu();
    let f32t = Tensor::from_f32(&backend, &[2, 2], &[0.0; 4]).expect("f32");
    let i32t = Tensor::from_i32(&backend, &[2, 2], &[0; 4]).expect("i32");

    assert!(f32t.upload_i32(&[0; 4]).is_err(), "I32 upload into F32");
    assert!(i32t.upload_f32(&[0.0; 4]).is_err(), "F32 upload into I32");
    assert!(
        f32t.upload_f32(&[0.0; 3]).is_err(),
        "F32 upload length mismatch"
    );
    assert!(
        i32t.upload_i32(&[0; 5]).is_err(),
        "I32 upload length mismatch"
    );

    // Strided views: the typed path refuses (packed order would be a
    // lie); the bytes path takes the span.
    let strided = f32t.transpose().expect("transpose");
    assert!(!strided.is_contiguous());
    assert!(
        strided.upload_f32(&[0.0; 4]).is_err(),
        "typed upload into strided view"
    );
    assert!(
        strided.upload_bytes(&[0u8; 16]).is_ok(),
        "span upload into strided view"
    );
}

#[test]
fn download_rejects_wrong_dtype_strides() {
    let backend = open_test_cpu();
    let f32t = Tensor::from_f32(&backend, &[2, 2], &[1.0; 4]).expect("f32");
    let i32t = Tensor::from_i32(&backend, &[2, 2], &[1; 4]).expect("i32");
    assert!(f32t.to_vec_i32().is_err(), "I32 download of F32");
    assert!(i32t.to_vec_f32().is_err(), "F32 download of I32");

    let strided = f32t.transpose().expect("transpose");
    assert!(
        strided.to_vec_f32().is_err(),
        "typed download of strided view"
    );
    // The bytes path downloads the span regardless of strides.
    assert_eq!(strided.to_bytes().expect("span download").len(), 16);
}

#[test]
fn upload_bytes_span_semantics_on_strided_view() {
    let backend = open_test_cpu();
    let parent = Tensor::from_f32(&backend, &[4, 3], &[0.0; 12]).expect("parent");
    let view = parent.view_2d([4, 2], 32, 0).expect("strided view");
    assert_eq!(view.nbytes(), 48, "span, not packed size");
    assert!(
        view.upload_bytes(&[0u8; 32]).is_err(),
        "packed length is not the span"
    );
    let span: Vec<u8> = (0..48u8).collect();
    view.upload_bytes(&span).expect("span upload");
    // Native view resolution writes parent storage at the covered
    // offset, so a full-span view reads back the parent bytes.
    assert_eq!(view.to_bytes().expect("view bytes"), span);
    assert_eq!(parent.to_bytes().expect("parent bytes"), span);
}

#[test]
fn copy_into_round_trip_and_rejections() {
    let backend = open_test_cpu();
    let src = Tensor::from_f32(&backend, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("src");
    let dst = Tensor::empty(&backend, DType::F32, &[2, 3]).expect("dst");
    src.copy_into(&dst).expect("copy");
    assert_eq!(
        dst.to_vec_f32().expect("download"),
        src.to_vec_f32().unwrap()
    );
    src.copy_into(&src).expect("self-copy is a no-op");

    let other_dtype = Tensor::from_i32(&backend, &[2, 3], &[0; 6]).expect("i32");
    assert!(src.copy_into(&other_dtype).is_err(), "dtype mismatch");
    let other_shape = Tensor::empty(&backend, DType::F32, &[3, 2]).expect("shape");
    assert!(src.copy_into(&other_shape).is_err(), "shape mismatch");
    let strided = src.transpose().expect("transpose");
    let strided_dst = Tensor::empty(&backend, DType::F32, &[3, 2]).expect("dst2");
    assert!(
        strided.copy_into(&strided_dst).is_err(),
        "strided source refused"
    );
    assert!(
        strided_dst.copy_into(&strided).is_err(),
        "strided dest refused"
    );
    // A view can be contiguous (full-coverage slice) yet still be
    // refused: native copy asserts on a view's NULL source buffer.
    let alias = src.view_2d([2, 3], 8, 0).expect("full-coverage view");
    assert!(alias.is_view() && alias.is_contiguous());
    assert!(alias.copy_into(&dst).is_err(), "view source refused");
    assert!(dst.copy_into(&alias).is_err(), "view dest refused");
}

#[test]
fn fill_f32_rejects_non_f32_and_strided() {
    let backend = open_test_cpu();
    let i32t = Tensor::from_i32(&backend, &[4], &[0; 4]).expect("i32");
    assert!(i32t.fill_f32(1.0).is_err(), "F32 fill of I32");
    let f32t = Tensor::from_f32(&backend, &[2, 2], &[0.0; 4]).expect("f32");
    assert!(
        f32t.transpose().expect("transpose").fill_f32(1.0).is_err(),
        "F32 fill of strided view"
    );
}

// -- C. model params ----------------------------------------------------------

#[test]
fn model_param_count_and_size_exact() {
    let Some(model) = load_fixture() else { return };
    // Generator ledger (all F32): embd 256 + attn_norm 8 + q/k/v/out
    // 4x64 + ffn_norm 8 + gate/up/down 3x128 + output_norm 8 +
    // output 256 = 1176 params = 4704 weight bytes.
    assert_eq!(model.n_params(), 1176);
    assert_eq!(model.size_bytes(), 4704);
}

#[test]
fn model_meta_matches_generator() {
    let Some(model) = load_fixture() else { return };
    assert_eq!(model.vocab_size().expect("vocab"), 32);
    assert_eq!(model.n_layer().expect("layers"), 1);
    assert_eq!(model.n_embd().expect("embd"), 8);
    assert_eq!(model.n_embd_inp().expect("embd_in"), 8);
    assert_eq!(model.n_embd_out().expect("embd_out"), 8);
    assert_eq!(model.n_head().expect("heads"), 2);
    assert_eq!(model.n_head_kv().expect("kv heads"), 2);
    assert_eq!(model.n_ctx_train().expect("ctx"), 64);
    assert_eq!(
        model.n_cls_out(),
        1,
        "no classifier labels -> upstream default 1"
    );
    assert!(!model.has_encoder(), "decoder-only");
    let desc = model.description().expect("description");
    assert!(!desc.is_empty());
    assert!(
        desc.to_lowercase().contains("llama"),
        "description names the arch: {desc}"
    );
}

#[test]
fn model_load_rejects_corrupt_content() {
    // Valid path, invalid bytes: a clean Error, never an abort.
    // (Missing-file is covered in ggml_smoke.)
    let mut path = std::env::temp_dir();
    path.push(format!("forge-corrupt-{}.gguf", std::process::id()));
    std::fs::write(&path, b"definitely not a gguf file\x00\x01\x02").expect("write");
    set_log_quiet(true);
    let result = Model::load(&path);
    set_log_quiet(false);
    std::fs::remove_file(&path).ok();
    assert!(result.is_err(), "corrupt bytes must fail cleanly");
}
