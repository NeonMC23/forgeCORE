//! Zero-copy shape ops: views, reshape, transpose, permute, cont, cast.
//!
//! CPU-only, single-threaded. Value tests materialize through `cont`
//! (or `to_bytes` for quant) and compare against hand layouts.

use forge_core::{Backend, DType, Tensor};

fn open_test_cpu() -> Backend {
    let backend = Backend::open_cpu().expect("CPU backend must open");
    backend
        .set_cpu_threads(1)
        .expect("single-threaded CPU for determinism");
    backend
}

#[test]
fn view_1d_slices_and_validates() {
    let backend = open_test_cpu();
    let base: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&backend, &[10], &base).expect("base");
    let v = t.view_1d(4, 3 * 4).expect("slice [3..7]");
    assert_eq!(v.shape(), &[4]);
    assert!(v.is_view());
    assert!(v.is_contiguous(), "packed 1-D slice stays contiguous");
    let packed = v.cont().expect("materialize");
    assert_eq!(
        packed.to_vec_f32().expect("download"),
        vec![3.0, 4.0, 5.0, 6.0]
    );

    assert!(t.view_1d(0, 0).is_err(), "empty extent");
    assert!(t.view_1d(8, 3 * 4).is_err(), "packed form overruns");
    assert!(t.view_1d(4, 3).is_err(), "misaligned offset");
    assert!(t.view_1d(4, usize::MAX - 3).is_err(), "offset overflow");
}

#[test]
fn view_2d_strides_rows() {
    let backend = open_test_cpu();
    // 4x3 parent, values = dim0 + 10*dim1.
    let data: Vec<f32> = (0..3)
        .flat_map(|r| (0..4).map(move |c| c as f32 + 10.0 * r as f32))
        .collect();
    let t = Tensor::from_f32(&backend, &[4, 3], &data).expect("base");
    // Every other row: 2 rows on a 2-row stride (32 bytes).
    let v = t.view_2d([4, 2], 32, 0).expect("strided view");
    assert_eq!(v.shape(), &[4, 2]);
    assert!(!v.is_contiguous());
    let packed = v.cont().expect("materialize");
    assert_eq!(
        packed.to_vec_f32().expect("download"),
        // Rows 0 and 2 of the parent.
        vec![0.0, 1.0, 2.0, 3.0, 20.0, 21.0, 22.0, 23.0]
    );

    assert!(t.view_2d([4, 2], 30, 0).is_err(), "misaligned stride");
    assert!(t.view_2d([4, 3], 32, 0).is_err(), "strided span overruns");
    assert!(t.view_2d([5, 3], 16, 0).is_err(), "packed form overruns");
}

#[test]
fn view_3d_and_4d_cover_and_validate() {
    let backend = open_test_cpu();
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&backend, &[2, 3, 4], &data).expect("base");
    let v = t.view_3d([2, 2, 2], [8, 32], 40).expect("3d view");
    assert_eq!(v.shape(), &[2, 2, 2]);
    assert!(v.is_view());
    // Offset 40 = element 10; rows of 2 on 8-byte stride, planes on 32.
    let packed = v.cont().expect("materialize");
    assert_eq!(
        packed.to_vec_f32().expect("download"),
        vec![10.0, 11.0, 12.0, 13.0, 18.0, 19.0, 20.0, 21.0]
    );
    assert!(t.view_3d([2, 3, 4], [8, 24], 8).is_err(), "span overruns");

    let t4 = Tensor::from_f32(&backend, &[2, 2, 2, 2], &[1.0; 16]).expect("4d base");
    let v4 = t4.view_4d([2, 2, 2, 2], [8, 16, 32], 0).expect("4d view");
    assert_eq!(v4.shape(), &[2, 2, 2, 2]);
    assert!(
        t4.view_4d([2, 2, 2, 2], [8, 16, 32], 4).is_err(),
        "offset overruns"
    );
    assert!(
        t4.view_4d([3, 2, 2, 2], [8, 24, 48], 0).is_err(),
        "packed overruns"
    );
}

#[test]
fn nested_views_chain_offsets() {
    let backend = open_test_cpu();
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&backend, &[16], &data).expect("base");
    let v1 = t.view_1d(12, 8).expect("v1 = [2..14]");
    let v2 = v1.view_1d(4, 16).expect("v2 = v1[4..8] = [6..10]");
    assert!(v2.is_view());
    let packed = v2.cont().expect("materialize");
    assert_eq!(
        packed.to_vec_f32().expect("download"),
        vec![6.0, 7.0, 8.0, 9.0]
    );
    // Nested bounds are checked against the immediate parent footprint.
    assert!(v1.view_1d(12, 8).is_err(), "nested overrun refused");
}

#[test]
fn overlapping_views_allowed_but_not_transposable() {
    let backend = open_test_cpu();
    let t = Tensor::from_f32(&backend, &[8], &[1.0; 8]).expect("base");
    // 2 rows of 4 on an 8-byte stride: rows overlap (packed 32 > span 24).
    let v = t
        .view_2d([4, 2], 8, 0)
        .expect("overlapping view is storable");
    assert_eq!(v.nbytes(), 24, "span, not packed size");
    assert!(
        v.transpose().is_err(),
        "transpose of overlapping view would trip the native packed-size assert"
    );
    assert!(
        v.permute([1, 0, 2, 3]).is_err(),
        "permute of overlapping view likewise"
    );
    // Non-overlapping strided views transpose fine.
    let wide = Tensor::from_f32(&backend, &[8, 2], &[1.0; 16]).expect("wide");
    let s = wide.view_2d([4, 2], 32, 0).expect("padded view");
    let tr = s.transpose().expect("transpose of padded view");
    assert_eq!(tr.shape(), &[2, 4]);
}

#[test]
fn reshape_repacks_values() {
    let backend = open_test_cpu();
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&backend, &[12], &data).expect("base");
    let r = t.reshape(&[3, 4]).expect("reshape");
    assert_eq!(r.shape(), &[3, 4]);
    assert!(r.is_view());
    assert!(r.is_contiguous(), "contiguous reshape stays packed");
    assert_eq!(r.to_vec_f32().expect("download"), data, "order preserved");
    let back = r.reshape(&[12]).expect("reshape back");
    assert_eq!(back.to_vec_f32().expect("download"), data);

    assert!(t.reshape(&[3, 5]).is_err(), "count mismatch");
    assert!(t.reshape(&[0, 12]).is_err(), "empty extent");
    let strided = t.reshape(&[3, 4]).expect("r").transpose().expect("tr");
    assert!(strided.reshape(&[12]).is_err(), "strided reshape refused");
}

#[test]
fn reshape_handles_quant_blocks() {
    let backend = open_test_cpu();
    // Q8_0 [64] -> [32, 2]: same 2 blocks, repacked strides.
    let q = Tensor::empty(&backend, DType::Q8_0, &[64]).expect("quant");
    q.fill_bytes(0x11).expect("pattern");
    let r = q.reshape(&[32, 2]).expect("quant reshape");
    assert_eq!(r.shape(), &[32, 2]);
    assert!(r.is_contiguous());
    assert_eq!(r.to_bytes().expect("download").len(), 68);
    // Ragged quant reshapes are refused at the Rust boundary.
    assert!(q.reshape(&[30, 2]).is_err(), "30 is not block-divisible");
}

#[test]
fn transpose_swaps_dims_twice_to_identity() {
    let backend = open_test_cpu();
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&backend, &[2, 3], &data).expect("base");
    let tr = t.transpose().expect("transpose");
    assert_eq!(tr.shape(), &[3, 2]);
    assert!(tr.is_view());
    assert!(!tr.is_contiguous());
    let back = tr.transpose().expect("transpose back");
    assert_eq!(back.shape(), &[2, 3]);
    assert_eq!(back.cont().expect("c").to_vec_f32().expect("d"), data);
    // Higher dims ride along untouched.
    let t3 = Tensor::from_f32(&backend, &[2, 3, 5], &[0.0; 30]).expect("3d");
    assert_eq!(t3.transpose().expect("tr").shape(), &[3, 2, 5]);
}

#[test]
fn permute_reorders_and_validates() {
    let backend = open_test_cpu();
    let t = Tensor::from_f32(&backend, &[2, 3, 5], &[0.0; 30]).expect("base");
    let id = t.permute([0, 1, 2, 3]).expect("identity");
    assert_eq!(id.shape(), &[2, 3, 5]);
    let rev = t.permute([2, 1, 0, 3]).expect("reversed");
    assert_eq!(rev.shape(), &[5, 3, 2]);
    assert!(rev.is_view());
    // Values: result (i,j,k) = base (k,j,i).
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let b = Tensor::from_f32(&backend, &[2, 2, 2], &data).expect("cube");
    let p = b.permute([2, 1, 0, 3]).expect("swap 0/2");
    let got = p.cont().expect("c").to_vec_f32().expect("d");
    let mut expected = vec![0.0f32; 8];
    for (i, slot) in expected.iter_mut().enumerate() {
        let (x, y, z) = (i % 2, (i / 2) % 2, i / 4);
        *slot = data[z + 2 * y + 4 * x];
    }
    assert_eq!(got, expected);

    assert!(t.permute([0, 0, 2, 3]).is_err(), "duplicated axis");
    assert!(t.permute([0, 1, 2, 4]).is_err(), "axis out of range");
}

#[test]
fn cont_packs_strided_and_passes_quant_through() {
    let backend = open_test_cpu();
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&backend, &[3, 4], &data).expect("base");
    let tr = t.transpose().expect("transpose");
    let packed = tr.cont().expect("cont");
    assert!(packed.is_contiguous());
    assert!(!packed.is_view());
    assert_eq!(packed.shape(), &[4, 3]);
    // result (i,j) = base (j,i), packed dim-0-fastest.
    let mut expected = vec![0.0f32; 12];
    for (i, slot) in expected.iter_mut().enumerate() {
        let (x, y) = (i % 4, i / 4);
        *slot = data[y + 3 * x];
    }
    assert_eq!(packed.to_vec_f32().expect("download"), expected);

    // Contiguous quant tensors pass through (fresh copy, same bytes).
    let q = Tensor::empty(&backend, DType::Q4_0, &[32, 2]).expect("quant");
    q.fill_bytes(0x5A).expect("pattern");
    let qc = q.cont().expect("quant cont");
    assert_eq!(qc.to_bytes().expect("download"), vec![0x5Au8; 36]);
    // Strided quant is refused (the native strided copy overruns blocks).
    let qs = q.transpose().expect("quant transpose");
    assert!(qs.cont().is_err(), "strided quant cont refused");
}

#[test]
fn cast_float_int_pairs_round_trip() {
    let backend = open_test_cpu();
    let data: Vec<f32> = vec![-3.5, -0.0, 0.0, 1.25, 100.0, 0.000061, -100.0, 65504.0];
    let t = Tensor::from_f32(&backend, &[8], &data).expect("base");
    for dtype in [DType::F16, DType::BF16] {
        let q = t.cast(dtype).expect("downcast");
        assert_eq!(q.dtype(), dtype);
        let back = q.cast(DType::F32).expect("upcast");
        let got = back.to_vec_f32().expect("download");
        for (g, e) in got.iter().zip(&data) {
            let tol = 0.002 * e.abs().max(1.0);
            assert!((g - e).abs() <= tol, "{dtype:?}: {g} vs {e}");
        }
    }
    // F16 <-> BF16 through the kernel (not through F32).
    let h = t.cast(DType::F16).expect("to F16");
    let b = h.cast(DType::BF16).expect("F16 to BF16");
    assert_eq!(b.dtype(), DType::BF16);
    let h2 = b.cast(DType::F16).expect("BF16 to F16");
    assert_eq!(h2.dtype(), DType::F16);
    // Same-type casts copy.
    let same = t.cast(DType::F32).expect("same type");
    assert_eq!(same.to_vec_f32().expect("download"), data);
}

#[test]
fn cast_f32_to_i32_truncates() {
    let backend = open_test_cpu();
    let t = Tensor::from_f32(&backend, &[6], &[3.9, -3.9, 2.0, -0.5, 0.5, 100.1]).expect("base");
    let i = t.cast(DType::I32).expect("to I32");
    assert_eq!(i.dtype(), DType::I32);
    assert_eq!(i.to_vec_i32().expect("download"), vec![3, -3, 2, 0, 0, 100]);
    let back = i.cast(DType::F32).expect("to F32");
    assert_eq!(
        back.to_vec_f32().expect("download"),
        vec![3.0, -3.0, 2.0, 0.0, 0.0, 100.0]
    );
}

#[test]
fn cast_quant_round_trips_within_error_bounds() {
    let backend = open_test_cpu();
    // Smooth data in [-1, 1]: friendly to every flat-block quantizer.
    let data: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.37).sin()).collect();
    let t = Tensor::from_f32(&backend, &[128], &data).expect("base");
    // (dtype, max abs error bound) — generous over measured error.
    // Bounds carry headroom over errors measured on ggml 0.25.1 for this exact
    // ladder input: Q4_0=0.1232, Q4_1=0.0660, Q5_0=0.0609, Q5_1=0.0320,
    // Q8_0=0.00418. A bound trip means the quantizer changed, not noise.
    for (dtype, bound) in [
        (DType::Q4_0, 0.15),
        (DType::Q4_1, 0.09),
        (DType::Q5_0, 0.08),
        (DType::Q5_1, 0.045),
        (DType::Q8_0, 0.008),
    ] {
        let q = t.cast(dtype).expect("quantize");
        assert_eq!(q.dtype(), dtype);
        assert_eq!(q.shape(), &[128]);
        let back = q.cast(DType::F32).expect("dequantize");
        let got = back.to_vec_f32().expect("download");
        let max_err = got
            .iter()
            .zip(&data)
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err <= bound,
            "{dtype:?} roundtrip error {max_err} exceeds {bound}"
        );
    }
}

#[test]
fn cast_zeroed_k_blocks_decode_to_zeros() {
    let backend = open_test_cpu();
    // K-quant super-blocks are not quantizable into (unaudited layout),
    // but zeroed blocks decode through the kernel to finite zeros.
    for dtype in [
        DType::Q2_K,
        DType::Q3_K,
        DType::Q4_K,
        DType::Q5_K,
        DType::Q6_K,
    ] {
        let row_bytes = dtype.type_size();
        let zeros = vec![0u8; row_bytes];
        let q = Tensor::from_bytes(&backend, dtype, &[256], &zeros).expect("zeroed K blocks");
        let back = q.cast(DType::F32).expect("dequantize");
        let got = back.to_vec_f32().expect("download");
        assert_eq!(got.len(), 256);
        assert!(got.iter().all(|&v| v == 0.0), "{dtype:?} zero blocks");
    }
}

#[test]
fn cast_rejects_unimplemented_and_unsafe_pairs() {
    let backend = open_test_cpu();
    let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let t = Tensor::from_f32(&backend, &[32], &data).expect("base");
    // No-dequantizer traps (NULL to_float natively).
    let q81 = Tensor::from_bytes(&backend, DType::Q8_1, &[32], &[0u8; 36]).expect("Q8_1");
    assert!(q81.cast(DType::F32).is_err(), "Q8_1 has no dequantizer");
    let q8k = Tensor::from_bytes(&backend, DType::Q8_K, &[256], &[0u8; 292]).expect("Q8_K");
    assert!(q8k.cast(DType::F32).is_err(), "Q8_K has no dequantizer");
    // One-way / unaudited quantize targets.
    assert!(
        t.cast(DType::Q8_1).is_err(),
        "F32 to Q8_1 is a one-way trap"
    );
    assert!(t.cast(DType::Q8_K).is_err(), "K super-blocks unaudited");
    // Kernel aborts ("not implemented").
    let h = t.cast(DType::F16).expect("F16");
    assert!(h.cast(DType::I32).is_err(), "F16 to I32 unimplemented");
    assert!(t.cast(DType::F64).is_err(), "F32 to F64 unimplemented");
    // Strided input (aborts the quantizing path, over-reads dequant).
    let tr = t.reshape(&[8, 4]).expect("r").transpose().expect("tr");
    assert!(tr.cast(DType::F16).is_err(), "strided cast refused");
    // Ragged quantize target dim.
    let ragged = Tensor::from_f32(&backend, &[30], &[0.0; 30]).expect("ragged");
    assert!(
        ragged.cast(DType::Q4_0).is_err(),
        "dim 0 must divide the block"
    );
}
