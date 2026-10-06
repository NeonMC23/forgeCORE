//! Mandatory proof: real deterministic CPU execution through the public API.
//!
//! Every op below runs on the CPU backend through the safe API and its
//! output is checked against hand-computed values or the frozen scalar
//! oracles (`forge_core::reference::ops` — dependency-free Rust that
//! never touches ggml). Rejection tests prove the preconditions fire
//! as `Err`, never as aborts. The closing group proves determinism:
//! fresh rebuilds are bit-identical, plans recompute deterministically,
//! and the thread count does not change results.
//!
//! CPU-only. The determinism core runs single-threaded except the one
//! test whose subject is the thread count itself.

use forge_core::reference::ops as oracle;
use forge_core::{
    add, concat, div, get_rows, matmul, mul, norm, rms_norm, rope, scale, silu, soft_max,
    soft_max_ext, sqr, sqrt, sub, Backend, DType, RopeMode, RopeParams, Tensor,
};

fn open_test_cpu() -> Backend {
    let backend = Backend::open_cpu().expect("CPU backend must open");
    backend
        .set_cpu_threads(1)
        .expect("single-threaded CPU for determinism");
    backend
}

/// Max absolute difference between ggml output and the oracle.
fn max_diff(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
}

// -- A. elementwise -------------------------------------------------------------

#[test]
fn add_sub_mul_exact() {
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0]).expect("a");
    let b = Tensor::from_f32(&backend, &[2, 2], &[10.0, 20.0, 30.0, 40.0]).expect("b");
    assert_eq!(
        add(&a, &b).expect("add").to_vec_f32().expect("read"),
        vec![11.0, 22.0, 33.0, 44.0]
    );
    assert_eq!(
        sub(&b, &a).expect("sub").to_vec_f32().expect("read"),
        vec![9.0, 18.0, 27.0, 36.0]
    );
    assert_eq!(
        mul(&a, &b).expect("mul").to_vec_f32().expect("read"),
        vec![10.0, 40.0, 90.0, 160.0]
    );
    // `b` may be strided (same shape, transposed storage): the kernel
    // reads it through strides. b_t holds [[10,30],[20,40]].
    let b_t = b.transpose().expect("transpose");
    assert!(!b_t.is_contiguous());
    assert_eq!(
        add(&a, &b_t)
            .expect("add strided b")
            .to_vec_f32()
            .expect("read"),
        vec![11.0, 32.0, 23.0, 44.0]
    );
    // `a` must be contiguous (the row loop advances it packed).
    assert!(add(&b_t, &a).is_err(), "strided `a` refused");
    assert!(sub(&b_t, &b).is_err(), "strided `a` refused");
    assert!(
        mul(&a, &a.transpose().expect("t")).is_ok(),
        "strided `b` fine"
    );
}

#[test]
fn binary_rejects_dtype_shape_backend_mismatch() {
    let backend = open_test_cpu();
    let other = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[2, 2], &[1.0; 4]).expect("a");
    let _b = Tensor::from_f32(&backend, &[2, 2], &[2.0; 4]).expect("b");
    let i = Tensor::from_i32(&backend, &[2, 2], &[2; 4]).expect("i32");
    let tall = Tensor::from_f32(&backend, &[2, 3], &[2.0; 6]).expect("tall");
    let far = Tensor::from_f32(&other, &[2, 2], &[2.0; 4]).expect("far");
    assert!(add(&a, &i).is_err(), "F32+I32 refused");
    assert!(sub(&i, &a).is_err(), "I32-F32 refused");
    assert!(mul(&a, &tall).is_err(), "shape mismatch refused");
    assert!(div(&a, &far).is_err(), "mixed backends refused");
    assert!(add(&a, &far).is_err(), "mixed backends refused");
}

#[test]
fn div_exact_and_division_by_zero_is_arithmetic() {
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[4], &[7.0, -6.0, 1.0, 0.0]).expect("a");
    let b = Tensor::from_f32(&backend, &[4], &[2.0, 4.0, 8.0, 5.0]).expect("b");
    assert_eq!(
        div(&a, &b).expect("div").to_vec_f32().expect("read"),
        vec![3.5, -1.5, 0.125, 0.0]
    );
    // No domain guard: 1/0 = +inf, 0/0 = NaN, no abort.
    let z = Tensor::from_f32(&backend, &[3], &[1.0, 0.0, -2.0]).expect("z");
    let zero = Tensor::from_f32(&backend, &[3], &[0.0; 3]).expect("zero");
    let q = div(&z, &zero)
        .expect("div by zero runs")
        .to_vec_f32()
        .expect("read");
    assert_eq!(q[0], f32::INFINITY);
    assert!(q[1].is_nan());
    assert_eq!(q[2], f32::NEG_INFINITY);
}

#[test]
fn scale_sqr_sqrt_exact_and_nan() {
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[4], &[1.0, -2.0, 4.0, 0.5]).expect("a");
    assert_eq!(
        scale(&a, 2.5).expect("scale").to_vec_f32().expect("read"),
        vec![2.5, -5.0, 10.0, 1.25]
    );
    assert_eq!(
        sqr(&a).expect("sqr").to_vec_f32().expect("read"),
        vec![1.0, 4.0, 16.0, 0.25]
    );
    let s = Tensor::from_f32(&backend, &[3], &[4.0, 2.0, 0.0]).expect("s");
    let r = sqrt(&s).expect("sqrt").to_vec_f32().expect("read");
    assert_eq!(r[0], 2.0);
    assert!((r[1] - 2.0f32.sqrt()).abs() < 1e-7);
    assert_eq!(r[2], 0.0);
    // sqrt of a negative is NaN (raw sqrtf), not an abort.
    let neg = Tensor::from_f32(&backend, &[1], &[-1.0]).expect("neg");
    let n = sqrt(&neg)
        .expect("sqrt(-1) runs")
        .to_vec_f32()
        .expect("read");
    assert!(n[0].is_nan());
}

#[test]
fn unary_rejects_non_f32_and_strided() {
    let backend = open_test_cpu();
    let i = Tensor::from_i32(&backend, &[2, 2], &[1; 4]).expect("i32");
    assert!(silu(&i).is_err());
    assert!(sqr(&i).is_err());
    assert!(sqrt(&i).is_err());
    assert!(scale(&i, 2.0).is_err());
    assert!(rms_norm(&i, 1e-5).is_err());
    assert!(norm(&i, 1e-5).is_err());
    assert!(soft_max(&i).is_err());
    let f = Tensor::from_f32(&backend, &[2, 2], &[1.0; 4]).expect("f32");
    let t = f.transpose().expect("transpose");
    assert!(silu(&t).is_err(), "strided silu refused");
    assert!(soft_max(&t).is_err(), "strided soft_max refused");
    assert!(rms_norm(&t, 1e-5).is_err(), "strided rms refused");
}

// -- B. matmul ------------------------------------------------------------------

#[test]
fn matmul_hand_computed_and_oracle_crossed() {
    let backend = open_test_cpu();
    // a [K=3, M=2]: rows m=0:[1,2,3], m=1:[4,5,6].
    let a = Tensor::from_f32(&backend, &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("a");
    // b [K=3, N=2]: rows n=0:[1,0,-1], n=1:[2,1,0].
    let b = Tensor::from_f32(&backend, &[3, 2], &[1.0, 0.0, -1.0, 2.0, 1.0, 0.0]).expect("b");
    let c = matmul(&a, &b).expect("matmul");
    assert_eq!(c.shape(), &[2, 2], "[M, N]");
    let got = c.to_vec_f32().expect("read");
    // out[m,n] = sum_k a[m,k]*b[n,k], packed [m + n*M].
    let want = vec![-2.0, -2.0, 4.0, 13.0];
    assert_eq!(got, want);
    // Independent cross-check: every cell equals the oracle dot.
    let av = a.to_vec_f32().expect("a");
    let bv = b.to_vec_f32().expect("b");
    for m in 0..2 {
        for n in 0..2 {
            let row_a = &av[m * 3..m * 3 + 3];
            let row_b = &bv[n * 3..n * 3 + 3];
            assert_eq!(oracle::dot(row_a, row_b).expect("dot"), want[m + n * 2]);
        }
    }
}

#[test]
fn matmul_accepts_strided_b() {
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[2, 2], &[1.0, 0.0, 0.0, 1.0]).expect("eye");
    // Same-shape strided b: [2,2] slice of a [3,2] parent (row gap).
    let parent =
        Tensor::from_f32(&backend, &[3, 2], &[5.0, 6.0, 0.0, 7.0, 8.0, 0.0]).expect("parent");
    let b = parent.view_2d([2, 2], 12, 0).expect("strided b");
    assert!(!b.is_contiguous());
    let c = matmul(&a, &b).expect("matmul strides b explicitly");
    assert_eq!(c.to_vec_f32().expect("read"), vec![5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn matmul_rejects() {
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[3, 2], &[1.0; 6]).expect("a");
    let b = Tensor::from_f32(&backend, &[3, 2], &[1.0; 6]).expect("b");
    let inner = Tensor::from_f32(&backend, &[4, 2], &[1.0; 8]).expect("inner");
    let flat = Tensor::from_f32(&backend, &[6], &[1.0; 6]).expect("rank1");
    let i = Tensor::from_i32(&backend, &[3, 2], &[1; 6]).expect("i32");
    assert!(matmul(&a, &inner).is_err(), "inner mismatch");
    assert!(matmul(&a, &flat).is_err(), "rank != 2");
    assert!(matmul(&flat, &b).is_err(), "rank != 2");
    assert!(matmul(&a, &i).is_err(), "non-F32");
    assert!(
        matmul(&a.transpose().expect("t"), &b).is_err(),
        "strided `a`"
    );
}

// -- C. norms / softmax / silu ----------------------------------------------------

#[test]
fn rms_norm_exact_case_and_oracle() {
    let backend = open_test_cpu();
    // Exact: x=[2,2], eps=0 -> rms=2 -> [1,1].
    let e = Tensor::from_f32(&backend, &[2], &[2.0, 2.0]).expect("e");
    assert_eq!(
        rms_norm(&e, 0.0).expect("rms").to_vec_f32().expect("read"),
        vec![1.0, 1.0]
    );
    // Oracle: per-column rms_norm with unit weights, eps=1e-5.
    let x = vec![3.0, -1.0, 0.5, 2.0, 0.0, -4.0, 1.0, 1.0, 1.0];
    let t = Tensor::from_f32(&backend, &[3, 3], &x).expect("x");
    let got = rms_norm(&t, 1e-5).expect("rms").to_vec_f32().expect("read");
    let mut want = vec![0.0; 9];
    for col in 0..3 {
        oracle::rms_norm(
            &x[col * 3..col * 3 + 3],
            &[1.0; 3],
            1e-5,
            &mut want[col * 3..col * 3 + 3],
        )
        .expect("oracle");
    }
    assert!(
        max_diff(&got, &want) < 1e-6,
        "rms vs oracle: {got:?} {want:?}"
    );
}

#[test]
fn norm_centers_exact_case_and_eps_rules() {
    let backend = open_test_cpu();
    // Exact: [2,4] mean 3 var 1 eps 0 -> [-1,1].
    let t = Tensor::from_f32(&backend, &[2], &[2.0, 4.0]).expect("t");
    assert_eq!(
        norm(&t, 0.0).expect("norm").to_vec_f32().expect("read"),
        vec![-1.0, 1.0]
    );
    // Property: output has mean ~0 on generic input.
    let g = Tensor::from_f32(&backend, &[4], &[1.0, 2.0, 4.0, 9.0]).expect("g");
    let y = norm(&g, 1e-5).expect("norm").to_vec_f32().expect("read");
    let mean = y.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-6, "centered: {y:?}");
    // Eps rules: negative and NaN abort natively -> Err; +inf is arithmetic.
    assert!(norm(&t, -1.0).is_err(), "negative eps");
    assert!(norm(&t, f32::NAN).is_err(), "NaN eps");
    assert!(rms_norm(&t, -0.5).is_err(), "negative eps");
    assert!(norm(&t, f32::INFINITY).is_ok(), "+inf eps is arithmetic");
    // Constant column, eps=0: 0/0 = NaN, no abort.
    let c = Tensor::from_f32(&backend, &[3], &[5.0; 3]).expect("const");
    let n = norm(&c, 0.0).expect("runs").to_vec_f32().expect("read");
    assert!(n.iter().all(|v| v.is_nan()));
}

#[test]
fn soft_max_exact_case_and_oracle() {
    let backend = open_test_cpu();
    // Exact: single-element column -> [1.0].
    let one = Tensor::from_f32(&backend, &[1, 2], &[5.0, -3.0]).expect("one");
    assert_eq!(
        soft_max(&one).expect("softmax").to_vec_f32().expect("read"),
        vec![1.0, 1.0]
    );
    // Oracle: stable softmax per column (includes a large shift).
    let x = vec![1000.0, 1001.0, 999.0, 0.0, 1.0, -1.0];
    let t = Tensor::from_f32(&backend, &[3, 2], &x).expect("x");
    let got = soft_max(&t).expect("softmax").to_vec_f32().expect("read");
    let mut want = x.clone();
    oracle::softmax_in_place(&mut want[0..3]).expect("oracle");
    oracle::softmax_in_place(&mut want[3..6]).expect("oracle");
    assert!(max_diff(&got, &want) < 1e-6, "softmax vs oracle");
    for col in 0..2 {
        let sum: f32 = got[col * 3..col * 3 + 3].iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "column {col} sums to 1");
    }
}

#[test]
fn soft_max_ext_mask_scale_and_rejections() {
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[3, 2], &[1.0, 2.0, 3.0, 0.5, -1.0, 2.0]).expect("a");
    // Zero mask, unit scale, no ALiBi: plain softmax of each column.
    let zeros = Tensor::from_f32(&backend, &[3, 2], &[0.0; 6]).expect("zeros");
    let plain = soft_max(&a).expect("plain").to_vec_f32().expect("read");
    let ext = soft_max_ext(&a, &zeros, 1.0, 0.0)
        .expect("ext")
        .to_vec_f32()
        .expect("read");
    assert!(max_diff(&plain, &ext) < 1e-7, "unmasked ext == softmax");
    // -inf mask entry zeroes that position (mask is active, slope 1).
    let mask = Tensor::from_f32(
        &backend,
        &[3, 2],
        &[f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .expect("mask");
    let m = soft_max_ext(&a, &mask, 1.0, 0.0)
        .expect("masked")
        .to_vec_f32()
        .expect("read");
    assert!(m[0] < 1e-7, "masked position ~0, got {}", m[0]);
    assert!((m[1] + m[2] - 1.0).abs() < 1e-6, "rest renormalizes");
    // Scale applies to `a`: ext(a, 0, 2) == softmax(2a).
    let scaled = soft_max(&scale(&a, 2.0).expect("2a"))
        .expect("softmax")
        .to_vec_f32()
        .expect("read");
    let ext2 = soft_max_ext(&a, &zeros, 2.0, 0.0)
        .expect("ext")
        .to_vec_f32()
        .expect("read");
    assert!(max_diff(&scaled, &ext2) < 1e-7, "scale == softmax(2a)");
    // F16 masks take the dedicated mask path with the same result.
    let mask16 = zeros.cast(DType::F16).expect("f16 mask");
    let e16 = soft_max_ext(&a, &mask16, 1.0, 0.0)
        .expect("f16 mask")
        .to_vec_f32()
        .expect("read");
    assert!(max_diff(&plain, &e16) < 1e-6, "F16 mask matches");
    // ALiBi path executes and biases (max_bias > 0 changes output).
    let ali = soft_max_ext(&a, &zeros, 1.0, 8.0)
        .expect("alibi runs")
        .to_vec_f32()
        .expect("read");
    assert!(ali.iter().all(|v| v.is_finite()), "ALiBi output finite");
    // Rejections: bad mask dtype, ragged geometry, strided inputs.
    let badi = Tensor::from_i32(&backend, &[3, 2], &[0; 6]).expect("i32 mask");
    assert!(soft_max_ext(&a, &badi, 1.0, 0.0).is_err(), "I32 mask");
    let narrow = Tensor::from_f32(&backend, &[2, 2], &[0.0; 4]).expect("narrow");
    assert!(soft_max_ext(&a, &narrow, 1.0, 0.0).is_err(), "ne0 mismatch");
    let short = Tensor::from_f32(&backend, &[3, 1], &[0.0; 3]).expect("short");
    assert!(soft_max_ext(&a, &short, 1.0, 0.0).is_err(), "ne1 shortfall");
    let wide = Tensor::from_f32(&backend, &[4, 2], &[0.0; 8]).expect("wide");
    let strided = wide.view_2d([3, 2], 16, 0).expect("strided mask");
    assert!(!strided.is_contiguous());
    assert!(
        soft_max_ext(&a, &strided, 1.0, 0.0).is_err(),
        "strided mask"
    );
    assert!(
        soft_max_ext(&a.transpose().expect("t"), &zeros, 1.0, 0.0).is_err(),
        "strided input"
    );
}

#[test]
fn silu_and_swiglu_match_oracle() {
    let backend = open_test_cpu();
    // silu(0) == 0 exactly.
    let z = Tensor::from_f32(&backend, &[1], &[0.0]).expect("zero");
    assert_eq!(
        silu(&z).expect("silu").to_vec_f32().expect("read"),
        vec![0.0]
    );
    // Oracle across a range incl. saturation tails.
    let x = vec![-6.0, -1.0, -0.25, 0.5, 2.0, 6.0];
    let t = Tensor::from_f32(&backend, &[6], &x).expect("x");
    let got = silu(&t).expect("silu").to_vec_f32().expect("read");
    let mut want = vec![0.0; 6];
    oracle::silu(&x, &mut want).expect("oracle");
    assert!(max_diff(&got, &want) < 1e-6, "silu vs oracle");
    // Composite: swiglu(gate, up) == silu(gate) * up.
    let gate = Tensor::from_f32(&backend, &[3], &[0.0, 2.0, -3.0]).expect("gate");
    let up = Tensor::from_f32(&backend, &[3], &[5.0, 4.0, 1.0]).expect("up");
    let got = mul(&silu(&gate).expect("silu"), &up)
        .expect("mul")
        .to_vec_f32()
        .expect("read");
    let mut want = vec![0.0; 3];
    oracle::swiglu(&[0.0, 2.0, -3.0], &[5.0, 4.0, 1.0], &mut want).expect("oracle");
    assert!(max_diff(&got, &want) < 1e-6, "swiglu vs oracle");
}

// -- D. rope / get_rows / concat --------------------------------------------------

#[test]
fn rope_neox_matches_oracle() {
    let backend = open_test_cpu();
    // [ne0=4, heads=2, tokens=2], positions [0, 1]; token t head h
    // starts at (t*2+h)*4.
    let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let a = Tensor::from_f32(&backend, &[4, 2, 2], &data).expect("a");
    let pos = Tensor::from_i32(&backend, &[2], &[0, 1]).expect("positions");
    let params = RopeParams {
        n_dims: 4,
        ..RopeParams::default()
    };
    assert_eq!(params.mode, RopeMode::NeoX);
    let got = rope(&a, &pos, &params)
        .expect("rope")
        .to_vec_f32()
        .expect("read");
    // Position 0 is the identity, exactly.
    assert_eq!(got[0..8], data[0..8]);
    // Position 1 matches the half-split oracle per head.
    let mut q = data[8..16].to_vec();
    let mut k = vec![0.0; 4]; // oracle needs a KV side; unused here
    oracle::rope(&mut q, &mut k, 1, 4, 2, 1, 10_000.0).expect("oracle");
    assert!(
        max_diff(&got[8..16], &q) < 1e-5,
        "rope vs oracle: {:?} {:?}",
        &got[8..16],
        q
    );
}

#[test]
fn rope_modes_and_rejections() {
    let backend = open_test_cpu();
    let data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let a = Tensor::from_f32(&backend, &[4, 2, 2], &data).expect("a");
    let pos = Tensor::from_i32(&backend, &[2], &[0, 1]).expect("positions");
    let neox = rope(
        &a,
        &pos,
        &RopeParams {
            n_dims: 4,
            ..RopeParams::default()
        },
    )
    .expect("neox")
    .to_vec_f32()
    .expect("read");
    let normal = rope(
        &a,
        &pos,
        &RopeParams {
            n_dims: 4,
            mode: RopeMode::Normal,
            ..RopeParams::default()
        },
    )
    .expect("normal runs")
    .to_vec_f32()
    .expect("read");
    // Both modes fix position 0 exactly ...
    assert_eq!(normal[0..8], data[0..8]);
    // ... and differ past it (the mode switch is real).
    assert!(
        max_diff(&normal[8..16], &neox[8..16]) > 1e-3,
        "Normal and NeoX differ at pos 1"
    );

    // Rejections (each mirrors a native assert or overrun).
    let odd = Tensor::from_f32(&backend, &[3, 2, 2], &[1.0; 12]).expect("odd");
    assert!(
        rope(
            &odd,
            &pos,
            &RopeParams {
                n_dims: 2,
                ..RopeParams::default()
            }
        )
        .is_err(),
        "odd ne0 overruns"
    );
    let flat = Tensor::from_f32(&backend, &[4, 4], &[1.0; 16]).expect("rank2");
    assert!(
        rope(
            &flat,
            &pos,
            &RopeParams {
                n_dims: 4,
                ..RopeParams::default()
            }
        )
        .is_err(),
        "rank != 3"
    );
    let f16a = Tensor::from_f32(&backend, &[4, 2, 2], &data)
        .expect("a")
        .cast(DType::F16)
        .expect("f16");
    assert!(
        rope(
            &f16a,
            &pos,
            &RopeParams {
                n_dims: 4,
                ..RopeParams::default()
            }
        )
        .is_err(),
        "non-F32 input"
    );
    let fpos = Tensor::from_f32(&backend, &[2], &[0.0, 1.0]).expect("f32 pos");
    assert!(
        rope(
            &a,
            &fpos,
            &RopeParams {
                n_dims: 4,
                ..RopeParams::default()
            }
        )
        .is_err(),
        "non-I32 positions"
    );
    let short = Tensor::from_i32(&backend, &[1], &[0]).expect("short pos");
    assert!(
        rope(
            &a,
            &short,
            &RopeParams {
                n_dims: 4,
                ..RopeParams::default()
            }
        )
        .is_err(),
        "position count"
    );
    let matrix = Tensor::from_i32(&backend, &[1, 2], &[0, 1]).expect("matrix pos");
    assert!(
        rope(
            &a,
            &matrix,
            &RopeParams {
                n_dims: 4,
                ..RopeParams::default()
            }
        )
        .is_err(),
        "position rank"
    );
    assert!(
        rope(&a, &pos, &RopeParams::default()).is_err(),
        "default n_dims=0"
    );
    assert!(
        rope(
            &a,
            &pos,
            &RopeParams {
                n_dims: 3,
                ..RopeParams::default()
            }
        )
        .is_err(),
        "odd n_dims"
    );
    assert!(
        rope(
            &a,
            &pos,
            &RopeParams {
                n_dims: 6,
                ..RopeParams::default()
            }
        )
        .is_err(),
        "n_dims > ne0"
    );
    assert!(
        rope(
            &a,
            &pos,
            &RopeParams {
                n_dims: 4,
                n_ctx_orig: 0,
                ..RopeParams::default()
            }
        )
        .is_err(),
        "n_ctx_orig=0"
    );
}

#[test]
fn get_rows_exact_and_oob_rejected() {
    let backend = open_test_cpu();
    // Table [d=3, rows=4]: row r = [10r, 10r+1, 10r+2].
    let table = Tensor::from_f32(
        &backend,
        &[3, 4],
        &[
            0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0,
        ],
    )
    .expect("table");
    let idx = Tensor::from_i32(&backend, &[3], &[3, 0, 3]).expect("indices");
    let out = get_rows(&table, &idx).expect("gather");
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(
        out.to_vec_f32().expect("read"),
        vec![30.0, 31.0, 32.0, 0.0, 1.0, 2.0, 30.0, 31.0, 32.0]
    );
    // I32 tables keep their dtype.
    let itable = Tensor::from_i32(&backend, &[2, 3], &[7, 8, 9, 10, 11, 12]).expect("itable");
    let iidx = Tensor::from_i32(&backend, &[2], &[2, 1]).expect("iidx");
    let iout = get_rows(&itable, &iidx).expect("i32 gather");
    assert_eq!(iout.dtype(), DType::I32);
    assert_eq!(iout.to_vec_i32().expect("read"), vec![11, 12, 9, 10]);
    // Every index is prevalidated (the kernel asserts per row).
    for bad in [&[4i32][..], &[-1][..], &[0, 4][..], &[2, 2, 99][..]] {
        let idx = Tensor::from_i32(&backend, &[bad.len()], bad).expect("idx");
        assert!(get_rows(&table, &idx).is_err(), "OOB {bad:?} refused");
    }
    // Rank rules: indices [k, t2, t3] with t2 == table.ne2 (1 here).
    let wide = Tensor::from_i32(&backend, &[2, 2], &[0, 1, 0, 1]).expect("wide");
    assert!(get_rows(&table, &wide).is_err(), "t2 mismatch");
    let fidx = Tensor::from_f32(&backend, &[2], &[0.0, 1.0]).expect("f32 idx");
    assert!(get_rows(&table, &fidx).is_err(), "non-I32 indices");
    assert!(
        get_rows(&table.transpose().expect("t"), &iidx).is_err(),
        "strided table"
    );
}

#[test]
fn concat_exact_all_dims_and_rejections() {
    let backend = open_test_cpu();
    let a = Tensor::from_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0]).expect("a");
    let b = Tensor::from_f32(&backend, &[3, 2], &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).expect("b");
    let c0 = concat(&a, &b, 0).expect("dim0");
    assert_eq!(c0.shape(), &[5, 2]);
    assert_eq!(
        c0.to_vec_f32().expect("read"),
        vec![1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0]
    );
    let d = Tensor::from_f32(&backend, &[2, 3], &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).expect("d");
    let c1 = concat(&a, &d, 1).expect("dim1");
    assert_eq!(c1.shape(), &[2, 5]);
    assert_eq!(
        c1.to_vec_f32().expect("read"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    );
    // Non-quant inputs may be strided.
    let at = a.transpose().expect("strided a");
    let e = Tensor::from_f32(&backend, &[2, 2], &[5.0, 6.0, 7.0, 8.0]).expect("e");
    let cs = concat(&at, &e, 0).expect("strided concat");
    assert_eq!(
        cs.to_vec_f32().expect("read"),
        vec![1.0, 3.0, 5.0, 6.0, 2.0, 4.0, 7.0, 8.0]
    );
    // Quant inputs join block rows when contiguous ...
    let qa = Tensor::from_bytes(&backend, DType::Q8_0, &[32], &[0x11u8; 34]).expect("qa");
    let qb = Tensor::from_bytes(&backend, DType::Q8_0, &[32], &[0x22u8; 34]).expect("qb");
    let qc = concat(&qa, &qb, 0).expect("quant concat");
    assert_eq!(qc.shape(), &[64]);
    let mut joined = vec![0x11u8; 34];
    joined.extend_from_slice(&[0x22u8; 34]);
    assert_eq!(qc.to_bytes().expect("read"), joined);
    // ... and are refused when strided. (A *square* quant transpose
    // has uniform strides and reports contiguous — genuinely packed,
    // holding the parent's transposed values. Non-square is strided.)
    let square =
        Tensor::from_bytes(&backend, DType::Q8_0, &[32, 32], &[0u8; 34 * 32]).expect("square");
    let square_t = square.transpose().expect("square transpose");
    assert!(square_t.is_contiguous(), "uniform strides are packed");
    assert!(
        square_t.cont().is_ok(),
        "contiguous square transpose copies"
    );
    let qbig = Tensor::from_bytes(&backend, DType::Q8_0, &[32, 64], &[0u8; 34 * 64]).expect("qbig");
    let qstrided = qbig.transpose().expect("strided quant");
    assert_eq!(qstrided.shape(), &[64, 32]);
    assert!(!qstrided.is_contiguous());
    let qplain = Tensor::empty(&backend, DType::Q8_0, &[64, 32]).expect("qplain");
    assert!(
        concat(&qstrided, &qplain, 1).is_err(),
        "strided quant refused"
    );
    assert!(
        concat(&qplain, &qstrided, 1).is_err(),
        "strided quant refused"
    );
    // Shape/dtype/dim rules.
    assert!(concat(&a, &d, 0).is_err(), "off-axis mismatch");
    assert!(concat(&a, &i32pair(&backend), 0).is_err(), "dtype mismatch");
    assert!(concat(&a, &a, 4).is_err(), "dim >= 4");
}

fn i32pair(backend: &Backend) -> Tensor {
    Tensor::from_i32(backend, &[2, 2], &[1, 2, 3, 4]).expect("i32")
}

// -- E. history retention -----------------------------------------------------------
// A result's ggml node points at its inputs' contexts through raw
// `src`/`view_src` pointers. Statement-scoped inputs drop while the
// result lives on; without retention, later graph expansion over the
// result is heap use-after-free in safe code (pre-fix symptom:
// `munmap_chunk(): invalid pointer` in the attention test below).
// These tests pin the retention; see `History` in `tensor.rs`.

#[test]
fn dropped_intermediates_do_not_dangle_results() {
    let backend = open_test_cpu();
    let s = Tensor::from_f32(&backend, &[3, 2], &[1.0, 3.0, 3.0, 1.0, 2.0, 1.0]).expect("s");
    let v = Tensor::from_f32(
        &backend,
        &[3, 4],
        &[1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0],
    )
    .expect("v");
    // `sc` drops at the semicolon; `w` retains it.
    let w = soft_max(&scale(&s, 0.5).expect("scale")).expect("weights");
    let o = matmul(&w, &v).expect("output over dropped history");
    // Bit-identical to the all-bound version.
    let sc = scale(&s, 0.5).expect("scale");
    let w2 = soft_max(&sc).expect("weights");
    let o2 = matmul(&w2, &v).expect("output");
    assert_eq!(o.shape(), &[2, 4]);
    assert_eq!(o.to_bytes().expect("read"), o2.to_bytes().expect("read"));
}

#[test]
fn dropped_parents_do_not_dangle_views() {
    let backend = open_test_cpu();
    // The parent drops; the strided view retains its storage.
    let v = {
        let t = Tensor::from_f32(
            &backend,
            &[4, 3],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("t");
        t.view_2d([4, 2], 32, 0).expect("view")
    };
    let bytes = v.to_bytes().expect("read over dropped parent");
    assert_eq!(bytes.len(), 48);
    assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1.0);
    // Transpose over a dropped parent, then a packing copy.
    let t2 = {
        let t = Tensor::from_f32(&backend, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("t");
        t.transpose().expect("transpose")
    };
    assert_eq!(
        t2.cont().expect("cont").to_vec_f32().expect("read"),
        vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn graph_recompute_reads_retained_inputs() {
    use forge_core::Graph;
    let backend = open_test_cpu();
    // Both inputs drop; the result retains their buffers.
    let c = {
        let a = Tensor::from_f32(&backend, &[2], &[1.0, 2.0]).expect("a");
        let b = Tensor::from_f32(&backend, &[2], &[10.0, 20.0]).expect("b");
        add(&a, &b).expect("add")
    };
    let mut graph = Graph::new().expect("graph");
    graph.add_output(&c).expect("add");
    graph
        .compute(&backend)
        .expect("recompute over dropped inputs");
    assert_eq!(c.to_vec_f32().expect("read"), vec![11.0, 22.0]);
}

// -- F. determinism ---------------------------------------------------------------

/// One MLP hidden block: silu(W1 x + b) -> W2 -> rms -> softmax.
/// Shapes: W1 [4,6], x [4,2] -> [6,2]; W2 [6,3] -> [3,2].
fn mlp(backend: &Backend) -> Tensor {
    let w1: Vec<f32> = (0..24).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
    let w1 = Tensor::from_f32(backend, &[4, 6], &w1).expect("w1");
    let x =
        Tensor::from_f32(backend, &[4, 2], &[1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]).expect("x");
    let bias = Tensor::from_f32(backend, &[6, 2], &[0.5; 12]).expect("bias");
    let h1 = silu(&add(&matmul(&w1, &x).expect("mm1"), &bias).expect("add")).expect("silu");
    let w2: Vec<f32> = (0..18).map(|i| ((i % 5) as f32 - 2.0) * 0.5).collect();
    let w2 = Tensor::from_f32(backend, &[6, 3], &w2).expect("w2");
    let h2 = matmul(&w2, &h1).expect("mm2");
    soft_max(&rms_norm(&h2, 1e-5).expect("rms")).expect("softmax")
}

#[test]
fn end_to_end_mlp_is_bit_identical_across_runs() {
    let backend = open_test_cpu();
    let first = mlp(&backend).to_bytes().expect("read");
    let second = mlp(&backend).to_bytes().expect("read");
    assert_eq!(first, second, "fresh rebuilds are bit-identical");
    // ... and sane, not identically degenerate: finite columns summing to 1.
    let back = Tensor::from_bytes(&backend, DType::F32, &[3, 2], &first).expect("wrap");
    let values = back.to_vec_f32().expect("read");
    assert!(values.iter().all(|v| v.is_finite()), "all finite");
    for col in 0..2 {
        let sum: f32 = values[col * 3..col * 3 + 3].iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "column {col} sums to 1");
    }
    assert!(
        values.iter().any(|&v| v > 0.01),
        "non-degenerate: {values:?}"
    );
}

#[test]
fn plan_recompute_is_deterministic_and_live() {
    use forge_core::Graph;
    let backend = open_test_cpu();
    let x =
        Tensor::from_f32(&backend, &[4, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).expect("x");
    let w = Tensor::from_f32(&backend, &[4, 4], &[0.5; 16]).expect("w");
    let y = silu(&matmul(&w, &x).expect("mm")).expect("silu");
    let mut graph = Graph::new().expect("graph");
    graph.add_output(&y).expect("add");
    let plan = graph.plan(&backend).expect("plan");

    plan.compute().expect("run 1");
    let run1 = y.to_bytes().expect("read");
    plan.compute().expect("run 2");
    assert_eq!(y.to_bytes().expect("read"), run1, "same inputs, same bytes");
    // Liveness: changed inputs change the output ...
    x.upload_f32(&[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        .expect("upload");
    plan.compute().expect("run 3");
    let run3 = y.to_bytes().expect("read");
    assert_ne!(run3, run1, "recompute actually runs");
    // ... and restoring the inputs restores the bytes.
    x.upload_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .expect("upload");
    plan.compute().expect("run 4");
    assert_eq!(
        y.to_bytes().expect("read"),
        run1,
        "inputs restored, bytes restored"
    );
}

#[test]
fn thread_count_does_not_change_results() {
    let one = open_test_cpu();
    let four = Backend::open_cpu().expect("CPU backend must open");
    four.set_cpu_threads(4).expect("four threads");
    let a = mlp(&one).to_bytes().expect("read");
    let b = mlp(&four).to_bytes().expect("read");
    assert_eq!(a, b, "1 thread == 4 threads, bit-identical");
}

#[test]
fn attention_scores_hand_computed() {
    let backend = open_test_cpu();
    // Single head, d=4, q=2 queries, k=3 keys. Layout note: ggml
    // softmax runs over dim 0, so scores stay [k, q] (keys along
    // dim 0) and values are stored [k, d].
    let k = Tensor::from_f32(
        &backend,
        &[4, 3],
        &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    )
    .expect("K [d,k]");
    let q = Tensor::from_f32(&backend, &[4, 2], &[1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])
        .expect("Q [d,q]");
    // S[k,q] = K[k] . Q[q]: hand dots below.
    let s = matmul(&k, &q).expect("scores");
    assert_eq!(s.shape(), &[3, 2]);
    assert_eq!(
        s.to_vec_f32().expect("read"),
        vec![1.0, 3.0, 3.0, 1.0, 2.0, 1.0],
        "S = [K0.Q0, K1.Q0, K2.Q0, K0.Q1, K1.Q1, K2.Q1]"
    );
    // Weights: softmax(S * 0.5) over keys, checked against an f64 oracle.
    let w = soft_max(&scale(&s, 0.5).expect("scale")).expect("weights");
    let wv = w.to_vec_f32().expect("read");
    let s64 = [1.0f64, 3.0, 3.0, 1.0, 2.0, 1.0];
    for col in 0..2 {
        let col_vals: Vec<f64> = s64[col * 3..col * 3 + 3].iter().map(|v| v * 0.5).collect();
        let max = col_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = col_vals.iter().map(|v| (v - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for row in 0..3 {
            let want = (exps[row] / sum) as f32;
            assert!(
                (wv[col * 3 + row] - want).abs() < 1e-6,
                "weight[{row},{col}] vs f64 oracle"
            );
        }
    }
    // Output: O[q,d] = sum_k W[k,q] V[k,d] via matmul(W [k,q], V [k,d]).
    let v = Tensor::from_f32(
        &backend,
        &[3, 4],
        &[1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0],
    )
    .expect("V [k,d]");
    let o = matmul(&w, &v).expect("output");
    assert_eq!(o.shape(), &[2, 4]);
    let ov = o.to_vec_f32().expect("read");
    let v64 = [
        1.0f64, 0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0,
    ];
    for qq in 0..2 {
        let max = (0..3)
            .map(|kk| s64[qq * 3 + kk] * 0.5)
            .fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = (0..3)
            .map(|kk| (s64[qq * 3 + kk] * 0.5 - max).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        for dd in 0..4 {
            let want: f64 = (0..3).map(|kk| exps[kk] / sum * v64[dd * 3 + kk]).sum();
            assert!(
                (ov[qq + dd * 2] as f64 - want).abs() < 1e-6,
                "O[{qq},{dd}] vs f64 oracle"
            );
        }
    }
    // The whole attention runs bit-identically twice.
    let w2 = soft_max(&scale(&matmul(&k, &q).expect("s"), 0.5).expect("sc")).expect("w");
    let o2 = matmul(&w2, &v).expect("o");
    assert_eq!(o.to_bytes().expect("read"), o2.to_bytes().expect("read"));
}
