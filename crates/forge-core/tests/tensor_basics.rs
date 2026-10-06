//! Tensor basics: creation, upload/download, fill, copy, names, facts.
//!
//! CPU-only, single-threaded for determinism. Every test names the
//! native rule it pins (see the Phase-6 report for the audit).

use forge_core::{Backend, DType, Tensor};

fn open_test_cpu() -> Backend {
    let backend = Backend::open_cpu().expect("CPU backend must open");
    backend
        .set_cpu_threads(1)
        .expect("single-threaded CPU for determinism");
    backend
}

#[test]
fn shapes_normalize_trailing_ones() {
    let backend = open_test_cpu();
    let data = vec![0.0f32; 6];
    for shape in [&[2, 3][..], &[2, 3, 1][..], &[2, 3, 1, 1][..]] {
        let t = Tensor::from_f32(&backend, shape, &data).expect("create");
        assert_eq!(t.shape(), &[2, 3], "shape {shape:?} normalizes");
        assert_eq!(t.nelements(), 6);
    }
    // Leading ones are significant (ggml_n_dims counts them).
    let t = Tensor::from_f32(&backend, &[1, 4], &[0.0; 4]).expect("leading one");
    assert_eq!(t.shape(), &[1, 4]);
    // All-ones collapses to rank 1, never rank 0.
    let t = Tensor::from_f32(&backend, &[1, 1, 1], &[7.0]).expect("ones");
    assert_eq!(t.shape(), &[1]);
    assert_eq!(t.to_vec_f32().expect("download"), vec![7.0]);
}

#[test]
fn creation_rejects_bad_shapes() {
    let backend = open_test_cpu();
    assert!(Tensor::empty(&backend, DType::F32, &[]).is_err(), "rank 0");
    assert!(
        Tensor::empty(&backend, DType::F32, &[2, 2, 2, 2, 2]).is_err(),
        "rank 5"
    );
    assert!(
        Tensor::empty(&backend, DType::F32, &[4, 0]).is_err(),
        "extent 0"
    );
    assert!(
        Tensor::from_f32(&backend, &[2, 2], &[1.0]).is_err(),
        "count mismatch"
    );
    assert!(
        Tensor::from_i32(&backend, &[3], &[1, 2]).is_err(),
        "i32 count mismatch"
    );
}

#[test]
fn creation_enforces_quant_block_geometry() {
    let backend = open_test_cpu();
    // 32-wide family.
    assert!(Tensor::empty(&backend, DType::Q4_0, &[30]).is_err());
    assert!(Tensor::empty(&backend, DType::Q4_0, &[32]).is_ok());
    assert!(Tensor::empty(&backend, DType::Q8_0, &[31, 2]).is_err());
    assert!(Tensor::empty(&backend, DType::Q8_0, &[32, 2]).is_ok());
    // 256-wide K super-blocks.
    assert!(Tensor::empty(&backend, DType::Q4_K, &[32]).is_err());
    assert!(Tensor::empty(&backend, DType::Q4_K, &[256]).is_ok());
    assert!(Tensor::empty(&backend, DType::Q8_K, &[128]).is_err());
    assert!(Tensor::empty(&backend, DType::Q8_K, &[256, 3]).is_ok());
    // Plain types take any positive dim 0.
    assert!(Tensor::empty(&backend, DType::F32, &[7]).is_ok());
    assert!(Tensor::empty(&backend, DType::I8, &[1]).is_ok());
}

#[test]
fn fresh_tensors_are_contiguous_leaves() {
    let backend = open_test_cpu();
    let t = Tensor::from_f32(&backend, &[4, 3], &[1.0; 12]).expect("create");
    assert!(t.is_contiguous());
    assert!(!t.is_view());
    assert_eq!(t.op_name(), "NONE");
    assert_eq!(t.nbytes(), 12 * 4);
    assert_eq!(t.dtype(), DType::F32);
    let q = Tensor::empty(&backend, DType::Q8_0, &[32, 2]).expect("quant");
    assert!(q.is_contiguous());
    // 2 rows x 34 bytes/block.
    assert_eq!(q.nbytes(), 68);
}

#[test]
fn f32_and_i32_round_trip_verbatim() {
    let backend = open_test_cpu();
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 3.0).collect();
    let t = Tensor::from_f32(&backend, &[6, 4], &data).expect("upload");
    assert_eq!(t.to_vec_f32().expect("download"), data);
    let ints: Vec<i32> = vec![-5, 0, 7, 1_000_000, i32::MIN + 1, i32::MAX];
    let ti = Tensor::from_i32(&backend, &[6], &ints).expect("upload i32");
    assert_eq!(ti.to_vec_i32().expect("download i32"), ints);
}

#[test]
fn bytes_round_trip_any_dtype() {
    let backend = open_test_cpu();
    // F16 bytes (opaque to ForgeCore, verbatim through ggml).
    let bytes: Vec<u8> = (0..32u8).collect();
    let t = Tensor::from_bytes(&backend, DType::F16, &[16], &bytes).expect("F16 bytes");
    assert_eq!(t.to_bytes().expect("download bytes"), bytes);
    // Zeroed Q8_0 blocks are valid (decode to zeros — see cast tests).
    let zeros = vec![0u8; 34 * 3];
    let q = Tensor::from_bytes(&backend, DType::Q8_0, &[32, 3], &zeros).expect("Q8_0 zeros");
    assert_eq!(q.to_bytes().expect("download"), zeros);
    assert!(
        Tensor::from_bytes(&backend, DType::F32, &[4], &[0u8; 15]).is_err(),
        "short byte buffer must fail"
    );
    assert!(
        Tensor::from_bytes(&backend, DType::F32, &[4], &[0u8; 17]).is_err(),
        "long byte buffer must fail"
    );
}

#[test]
fn transfers_reject_dtype_and_stride_mismatches() {
    let backend = open_test_cpu();
    let f16 = Tensor::empty(&backend, DType::F16, &[4]).expect("F16");
    assert!(f16.to_vec_f32().is_err(), "F32 download of F16");
    assert!(f16.upload_f32(&[1.0; 4]).is_err(), "F32 upload to F16");
    let i32t = Tensor::empty(&backend, DType::I32, &[4]).expect("I32");
    assert!(i32t.to_vec_f32().is_err(), "F32 download of I32");
    assert!(i32t.upload_f32(&[1.0; 4]).is_err(), "F32 upload to I32");

    // Strided tensors refuse element transfers (packed order would lie).
    let t = Tensor::from_f32(&backend, &[2, 3], &[1.0; 6]).expect("base");
    let tr = t.transpose().expect("transpose");
    assert!(!tr.is_contiguous());
    assert!(tr.to_vec_f32().is_err(), "download of strided");
    assert!(tr.upload_f32(&[1.0; 6]).is_err(), "upload to strided");
    // ... but contiguous copies transfer fine.
    let packed = tr.cont().expect("cont");
    assert_eq!(packed.to_vec_f32().expect("download").len(), 6);
}

#[test]
fn fill_f32_covers_and_validates() {
    let backend = open_test_cpu();
    let t = Tensor::from_f32(&backend, &[3, 2], &[1.0; 6]).expect("base");
    t.fill_f32(2.5).expect("fill");
    assert_eq!(t.to_vec_f32().expect("download"), vec![2.5; 6]);
    let f16 = Tensor::empty(&backend, DType::F16, &[4]).expect("F16");
    assert!(f16.fill_f32(1.0).is_err(), "F32 fill of F16 aborts natively");
    let tr = t.transpose().expect("transpose");
    assert!(tr.fill_f32(1.0).is_err(), "strided fill miswrites natively");
}

#[test]
fn fill_bytes_covers_footprints() {
    let backend = open_test_cpu();
    let t = Tensor::empty(&backend, DType::F32, &[8]).expect("base");
    t.fill_bytes(0xAB).expect("fill");
    assert_eq!(t.to_bytes().expect("download"), vec![0xABu8; 32]);
    // Works on any dtype (staging path, no memset interface needed).
    let q = Tensor::empty(&backend, DType::Q4_0, &[32]).expect("quant");
    q.fill_bytes(0).expect("zero blocks");
    assert_eq!(q.to_bytes().expect("download"), vec![0u8; 18]);
}

#[test]
fn copy_into_moves_bytes_across_backends() {
    let backend = open_test_cpu();
    let other = Backend::open_cpu().expect("second backend");
    let src = Tensor::from_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0]).expect("src");
    let dst = Tensor::empty(&other, DType::F32, &[2, 2]).expect("dst");
    src.copy_into(&dst).expect("cross-backend copy");
    assert_eq!(dst.to_vec_f32().expect("download"), vec![1.0, 2.0, 3.0, 4.0]);
    // Self-copy is a natively handled no-op.
    src.copy_into(&src).expect("self copy");
    assert_eq!(src.to_vec_f32().expect("download"), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn copy_into_rejects_mismatches_and_views() {
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[2, 2], &[1.0; 4]).expect("a");
    let other_dtype = Tensor::empty(&backend, DType::F16, &[2, 2]).expect("f16");
    assert!(a.copy_into(&other_dtype).is_err(), "dtype mismatch");
    let other_shape = Tensor::empty(&backend, DType::F32, &[4]).expect("shape");
    assert!(a.copy_into(&other_shape).is_err(), "shape mismatch");
    // Square transpose keeps shape [2, 2] but is strided + a view.
    let strided = a.transpose().expect("transpose");
    assert_eq!(strided.shape(), &[2, 2]);
    assert!(a.copy_into(&strided).is_err(), "strided/view dst");
    assert!(strided.copy_into(&a).is_err(), "strided/view src");
    let wide = Tensor::from_f32(&backend, &[8], &[1.0; 8]).expect("wide");
    let v = wide.view_1d(4, 0).expect("view");
    assert!(a.copy_into(&v).is_err(), "shape mismatch on views");
    let _ = other_shape;
}

#[test]
fn strided_copy_refused_even_same_shape() {
    let backend = open_test_cpu();
    // Same-shape strided pair via padded views: copy refuses (the native
    // copy asserts identical packed layout). Parent rows are 8 wide
    // (32 bytes); the view reads 4-wide rows on the 32-byte stride.
    let wide = Tensor::from_f32(&backend, &[8, 2], &[2.0; 16]).expect("wide");
    let v = wide.view_2d([4, 2], 32, 0).expect("padded view");
    assert_eq!(v.shape(), &[4, 2]);
    assert!(!v.is_contiguous());
    let dst = Tensor::empty(&backend, DType::F32, &[4, 2]).expect("dst");
    assert!(v.copy_into(&dst).is_err(), "strided src");
    assert!(dst.copy_into(&v).is_err(), "strided dst");
}

#[test]
fn names_round_trip_truncate_and_reject_nul() {
    let backend = open_test_cpu();
    let t = Tensor::from_f32(&backend, &[2], &[1.0, 2.0]).expect("base");
    assert_eq!(t.name(), "", "fresh tensors are anonymous");
    t.set_name("weights.attn").expect("set");
    assert_eq!(t.name(), "weights.attn");
    assert!(t.set_name("has\0nul").is_err(), "NUL rejected");
    // 63-byte truncation.
    let long = "n".repeat(100);
    t.set_name(&long).expect("long name truncates");
    assert_eq!(t.name(), "n".repeat(63));
    // Truncation respects UTF-8 boundaries (é = 2 bytes; 63 bytes = 31 chars).
    let wide = "é".repeat(100);
    t.set_name(&wide).expect("wide name truncates");
    assert_eq!(t.name(), "é".repeat(31));
    assert_eq!(t.name().len(), 62);
}

#[test]
fn graph_add_names_anonymous_tensors() {
    use forge_core::Graph;
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[2], &[1.0, 2.0]).expect("a");
    assert_eq!(a.name(), "", "fresh tensors are anonymous");
    let b = forge_core::add(&a, &a).expect("add");
    // Eager execution visits inputs too, so even the scratch graph
    // names anonymous participants (cosmetic, but observable).
    assert_eq!(a.name(), "leaf_0");
    assert_ne!(b.name(), "", "op results are named at construction");
    let b_name = b.name();
    let mut graph = Graph::new().expect("graph");
    graph.add_output(&b).expect("add");
    // Already-named tensors keep their names on graph add.
    assert_eq!(a.name(), "leaf_0");
    assert_eq!(b.name(), b_name);
}

#[test]
fn views_outlive_parents_via_shared_allocation() {
    let backend = open_test_cpu();
    let view = {
        let parent = Tensor::from_f32(&backend, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("parent");
        parent.transpose().expect("transpose")
        // Parent drops here; the view keeps the allocation alive.
    };
    assert_eq!(view.shape(), &[3, 2]);
    // Transposed [2,3] -> [3,2]: result (i,j) = parent (j,i), packed
    // dim-0-fastest: [p00, p01, p02, p10, p11, p12].
    let packed = view.cont().expect("materialize");
    assert_eq!(
        packed.to_vec_f32().expect("download"),
        vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn debug_format_is_total() {
    let backend = open_test_cpu();
    let t = Tensor::from_f32(&backend, &[2], &[1.0, 2.0]).expect("base");
    let rendered = format!("{t:?}");
    assert!(rendered.contains("F32"), "debug shows dtype: {rendered}");
    assert!(rendered.contains('2'), "debug shows shape: {rendered}");
    let rendered = format!("{:?}", backend);
    assert!(rendered.contains("CPU"), "backend debug: {rendered}");
}
