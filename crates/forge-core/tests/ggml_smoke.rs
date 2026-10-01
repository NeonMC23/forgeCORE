//! Rust → ForgeCore → ggml → backend → op → Rust result.
//!
//! The tests below must pass on CPU. The GPU probe at the end is
//! non-mandatory: it exercises a detected accelerator when one exists and
//! otherwise reports a skip.

use forge_core::reference::ops;
use forge_core::{add, enumerate_devices, matmul, Backend, DType, DeviceType, Model, Tensor};

fn open_test_cpu() -> Backend {
    let backend = Backend::open_cpu().expect("CPU backend must open");
    backend
        .set_cpu_threads(1)
        .expect("single-threaded CPU for determinism");
    backend
}

#[test]
fn devices_enumerated_with_cpu_present() {
    let devices = enumerate_devices();
    assert!(
        !devices.is_empty(),
        "registry must report at least one device"
    );
    assert!(
        devices.iter().any(|d| d.device_type == DeviceType::Cpu),
        "a CPU device must be present, got {devices:?}"
    );
    for device in &devices {
        println!(
            "device[{}] {} ({:?}) free={} total={} desc={}",
            device.index,
            device.name,
            device.device_type,
            device.memory_free,
            device.memory_total,
            device.description
        );
    }
}

#[test]
fn cpu_backend_opens_from_device_info() {
    let devices = enumerate_devices();
    let cpu = devices
        .iter()
        .find(|d| d.device_type == DeviceType::Cpu)
        .expect("CPU device");
    let backend = Backend::open_device(cpu).expect("open_device on CPU");
    assert_eq!(backend.device_type(), DeviceType::Cpu);
}

#[test]
fn cpu_add_matches_scalar_oracle() {
    let backend = open_test_cpu();
    let a = [1.0f32, -2.5, 3.25, 100.0, 0.5, -0.125];
    let b = [0.5f32, 2.5, -3.25, 0.25, 8.0, 4.0];
    let expected = {
        let mut y = vec![0.0f32; a.len()];
        ops::add(&a, &b, &mut y).expect("oracle add");
        y
    };
    let ta = Tensor::from_f32(&backend, &[a.len()], &a).expect("upload a");
    let tb = Tensor::from_f32(&backend, &[b.len()], &b).expect("upload b");
    let sum = add(&ta, &tb).expect("ggml add");
    assert_eq!(sum.dtype(), DType::F32);
    assert_eq!(sum.shape(), &[a.len()]);
    let got = sum.to_vec_f32().expect("download");
    assert_eq!(got, expected, "element-wise add is exact");

    // 2-D shape is preserved through the round trip.
    let t2 = Tensor::from_f32(&backend, &[2, 3], &a).expect("upload 2-D");
    let sum2 = add(&t2, &t2).expect("ggml add 2-D");
    assert_eq!(sum2.shape(), &[2, 3]);
    let got2 = sum2.to_vec_f32().expect("download 2-D");
    let doubled: Vec<f32> = a.iter().map(|x| x + x).collect();
    assert_eq!(got2, doubled);
}

#[test]
fn cpu_matmul_matches_dot_oracle() {
    // ggml layout: a is [k, m], b is [k, n], result is [m, n] (C = A·B^T).
    let (k, m, n) = (2usize, 3usize, 4usize);
    let backend = open_test_cpu();

    // a[k + m*k] — contiguous rows of length k.
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 1.0).collect();
    // b[k + n*k] — contiguous columns of length k.
    let b: Vec<f32> = (0..n * k)
        .map(|i| ((i * 2654435761) % 97) as f32 / 97.0 - 0.5)
        .collect();

    let ta = Tensor::from_f32(&backend, &[k, m], &a).expect("upload a");
    let tb = Tensor::from_f32(&backend, &[k, n], &b).expect("upload b");
    let prod = matmul(&ta, &tb).expect("ggml matmul");
    assert_eq!(prod.shape(), &[m, n]);
    let got = prod.to_vec_f32().expect("download");

    // Oracle: each output is a scalar dot product of an a-row and a b-column.
    for mm in 0..m {
        for nn in 0..n {
            let row = &a[mm * k..(mm + 1) * k];
            let col: Vec<f32> = (0..k).map(|kk| b[nn * k + kk]).collect();
            let expected = ops::dot(row, &col).expect("oracle dot");
            let actual = got[mm + nn * m];
            assert!(
                (actual - expected).abs() <= 1e-5,
                "c[{mm},{nn}]: ggml={actual} oracle={expected}"
            );
        }
    }
}

#[test]
fn validation_errors_are_explicit() {
    let backend = open_test_cpu();
    let other = Backend::open_cpu().expect("second CPU backend");
    let a = Tensor::from_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0]).expect("a");
    let b = Tensor::from_f32(&backend, &[3], &[1.0, 2.0, 3.0]).expect("b");
    let c = Tensor::from_f32(&other, &[2, 2], &[1.0, 2.0, 3.0, 4.0]).expect("c");

    assert!(add(&a, &b).is_err(), "shape mismatch must fail");
    assert!(matmul(&a, &b).is_err(), "rank/inner mismatch must fail");
    assert!(add(&a, &c).is_err(), "cross-backend add must fail");

    let q = Tensor::empty(&backend, DType::F16, &[4]).expect("F16 tensor");
    assert!(q.to_vec_f32().is_err(), "non-F32 download must fail");

    assert!(
        Tensor::from_f32(&backend, &[2, 2], &[1.0]).is_err(),
        "element-count mismatch must fail"
    );
    assert!(
        Tensor::from_f32(&backend, &[], &[1.0]).is_err(),
        "rank-0 must fail"
    );
    assert!(
        Tensor::from_f32(&backend, &[2, 2, 2, 2, 2], &[0.0; 32]).is_err(),
        "rank-5 must fail"
    );
}

#[test]
fn stale_device_index_is_rejected() {
    let bogus = forge_core::DeviceInfo {
        index: usize::MAX,
        name: "bogus".to_string(),
        description: String::new(),
        device_type: DeviceType::Cpu,
        memory_free: 0,
        memory_total: 0,
    };
    assert!(Backend::open_device(&bogus).is_err());
}

#[test]
fn missing_model_file_is_an_error_not_an_abort() {
    forge_core::model::set_log_quiet(true);
    let result = Model::load(std::path::Path::new(
        "/var/tmp/forge-native/definitely-not-a-model.gguf",
    ));
    forge_core::model::set_log_quiet(false);
    assert!(result.is_err(), "missing file must fail cleanly");
}

#[test]
fn gpu_probe_is_non_mandatory() {
    let devices = enumerate_devices();
    let accelerators: Vec<_> = devices
        .iter()
        .filter(|d| {
            !matches!(
                d.device_type,
                DeviceType::Cpu | DeviceType::Meta | DeviceType::Unknown(_)
            )
        })
        .collect();
    if accelerators.is_empty() {
        println!("SKIP: no accelerator device detected (CPU-only environment)");
        return;
    }
    for device in accelerators {
        match Backend::open_device(device) {
            Ok(backend) => {
                let a = Tensor::from_f32(&backend, &[2], &[1.0, 2.0]).expect("upload");
                let sum = add(&a, &a).expect("accelerator add");
                assert_eq!(sum.to_vec_f32().expect("download"), vec![2.0, 4.0]);
                println!("accelerator '{}' add OK", device.name);
            }
            Err(error) => {
                println!(
                    "SKIP: accelerator '{}' present but unusable: {error}",
                    device.name
                );
            }
        }
    }
}

#[test]
fn tiny_fixture_model_loads_when_present() {
    let path = match std::env::var("FORGE_TEST_MODEL") {
        Ok(path) => path,
        Err(_) => {
            println!("SKIP: FORGE_TEST_MODEL not set (see docs/NATIVE.md)");
            return;
        }
    };
    let model = Model::load(std::path::Path::new(&path)).expect("fixture model must load");
    let vocab = model.vocab_size().expect("vocab size");
    assert!(model.n_params() > 0, "model must report parameters");
    assert!(vocab > 0, "model must report a vocabulary");
    println!(
        "loaded {path}: n_params={} vocab_size={vocab}",
        model.n_params()
    );
}
