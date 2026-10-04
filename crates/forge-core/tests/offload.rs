//! Phase-3 model offload: device capability facts, [`ModelOptions`]
//! offload wiring, explicit validation errors, and a skip-gated GPU
//! smoke test.
//!
//! Nothing here assumes a GPU exists: capability tests assert internal
//! consistency, validation tests use a nonexistent path (validation
//! runs before any native call), and only the final smoke test touches
//! real offload — reporting SKIP when no GPU device is present.

use forge_core::model::set_log_quiet;
use forge_core::{
    enumerate_devices, max_devices, supports_gpu_offload, supports_mlock, supports_mmap,
    DeviceType, GpuLayers, Model, ModelLoadMode, ModelOptions, SplitMode,
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

/// A path that passes UTF-8/NUL checks but is never opened, because
/// option validation runs before the native load call.
fn bogus_path() -> PathBuf {
    PathBuf::from("does-not-exist-for-validation-tests.gguf")
}

// -- A. capability facts ----------------------------------------------------

#[test]
fn max_devices_is_sixteen() {
    // Pinned upstream constant (`return 16`); sizes tensor_split arrays.
    assert_eq!(max_devices(), 16);
}

#[test]
fn mmap_and_mlock_supported_on_linux() {
    // Compile-time platform flags (`_POSIX_MEMLOCK_RANGE`); revisit if
    // the suite ever runs on a platform without them.
    assert!(supports_mmap());
    assert!(supports_mlock());
}

#[test]
fn offload_flag_consistent_with_enumeration() {
    // Upstream defines offload availability as "a GPU/IGPU device is
    // registered (RPC devices report type GPU), or RPC is supported";
    // without RPC compiled in, the flag must equal GPU/IGPU presence.
    let devices = enumerate_devices();
    let has_gpu = devices
        .iter()
        .any(|d| matches!(d.device_type, DeviceType::Gpu | DeviceType::Igpu));
    assert_eq!(
        supports_gpu_offload(),
        has_gpu,
        "offload flag disagrees with enumeration: {devices:?}"
    );
}

#[test]
fn registry_holds_cpu() {
    let devices = enumerate_devices();
    assert!(
        !devices.is_empty() && devices.iter().any(|d| d.device_type == DeviceType::Cpu),
        "expected at least a CPU device: {devices:?}"
    );
}

// -- B. CPU behavior unchanged ----------------------------------------------

#[test]
fn default_options_match_plain_load() {
    let Some(path) = fixture() else { return };
    let plain = Model::load(&path).expect("fixture model must load");
    let explicit =
        Model::load_with_options(&path, &ModelOptions::default()).expect("default options load");
    assert_eq!(plain.n_params(), explicit.n_params());
    assert_eq!(plain.size_bytes(), explicit.size_bytes());
    assert_eq!(plain.n_layer().unwrap(), explicit.n_layer().unwrap());
}

#[test]
fn explicit_cpu_options_load() {
    let Some(path) = fixture() else { return };
    let devices = enumerate_devices();
    let cpu = devices
        .iter()
        .find(|d| d.device_type == DeviceType::Cpu)
        .expect("CPU device")
        .clone();
    let mut options = ModelOptions::default();
    options.gpu_layers = GpuLayers::Cpu;
    options.split_mode = SplitMode::Layer;
    options.main_gpu = 0;
    options.tensor_split = Some(vec![1.0]);
    options.devices = Some(vec![cpu]);
    options.load_mode = ModelLoadMode::Auto;
    let model = Model::load_with_options(&path, &options).expect("explicit CPU options load");
    assert_eq!(model.n_params(), 1176);
}

#[test]
fn all_load_modes_load_cpu_fixture() {
    let Some(path) = fixture() else { return };
    for mode in [
        ModelLoadMode::Auto,
        ModelLoadMode::NoMmap,
        ModelLoadMode::Mmap,
        ModelLoadMode::Mlock,
        ModelLoadMode::MmapMlock,
    ] {
        let mut options = ModelOptions::default();
        options.load_mode = mode;
        let model = Model::load_with_options(&path, &options)
            .unwrap_or_else(|e| panic!("load mode {mode:?} must load the fixture: {e}"));
        assert_eq!(model.n_params(), 1176);
    }
}

#[test]
fn explicit_cpu_device_duplicates_load() {
    // Duplicates pass through verbatim (upstream accepts them); with
    // CPU layers selected nothing is offloaded anywhere.
    let Some(path) = fixture() else { return };
    let devices = enumerate_devices();
    let cpu = devices
        .iter()
        .find(|d| d.device_type == DeviceType::Cpu)
        .expect("CPU device")
        .clone();
    let mut options = ModelOptions::default();
    options.devices = Some(vec![cpu.clone(), cpu]);
    options.tensor_split = Some(vec![0.5, 0.5]);
    let model = Model::load_with_options(&path, &options).expect("duplicate CPU devices load");
    assert_eq!(model.n_params(), 1176);
}

// -- C. validation errors (no fixture needed) -------------------------------

#[test]
fn bad_layer_count_rejected() {
    let mut options = ModelOptions::default();
    options.gpu_layers = GpuLayers::Count(u32::MAX);
    let err = Model::load_with_options(&bogus_path(), &options).unwrap_err();
    assert!(
        err.to_string().starts_with("invalid: "),
        "bad layer count must be invalid: {err}"
    );
}

#[test]
fn empty_device_list_rejected() {
    let mut options = ModelOptions::default();
    options.devices = Some(vec![]);
    let err = Model::load_with_options(&bogus_path(), &options).unwrap_err();
    assert!(
        err.to_string().starts_with("invalid: "),
        "empty device list must be invalid: {err}"
    );
}

#[test]
fn unknown_device_index_rejected() {
    let devices = enumerate_devices();
    let mut bad = devices
        .first()
        .expect("registry holds at least one device")
        .clone();
    bad.index = devices.len() + 99;
    let mut options = ModelOptions::default();
    options.devices = Some(vec![bad]);
    let err = Model::load_with_options(&bogus_path(), &options).unwrap_err();
    let message = err.to_string();
    assert!(
        message.starts_with("invalid: ") && message.contains("stale device index"),
        "unknown device index must be invalid: {message}"
    );
}

#[test]
fn empty_split_rejected() {
    let mut options = ModelOptions::default();
    options.tensor_split = Some(vec![]);
    let err = Model::load_with_options(&bogus_path(), &options).unwrap_err();
    assert!(
        err.to_string().starts_with("invalid: "),
        "empty split must be invalid: {err}"
    );
}

#[test]
fn ill_valued_split_rejected() {
    for split in [vec![-1.0], vec![0.5, f32::NAN], vec![f32::INFINITY]] {
        let mut options = ModelOptions::default();
        options.tensor_split = Some(split);
        let err = Model::load_with_options(&bogus_path(), &options).unwrap_err();
        assert!(
            err.to_string().starts_with("invalid: "),
            "ill-valued split must be invalid: {err}"
        );
    }
}

#[test]
fn short_split_rejected_for_explicit_devices() {
    let devices = enumerate_devices();
    let cpu = devices
        .iter()
        .find(|d| d.device_type == DeviceType::Cpu)
        .expect("CPU device")
        .clone();
    let mut options = ModelOptions::default();
    options.devices = Some(vec![cpu.clone(), cpu]);
    options.tensor_split = Some(vec![1.0]);
    let err = Model::load_with_options(&bogus_path(), &options).unwrap_err();
    assert!(
        err.to_string().starts_with("invalid: "),
        "short split must be invalid: {err}"
    );
}

#[test]
fn short_split_rejected_for_default_selection() {
    let count = enumerate_devices().len();
    if count < 2 {
        println!("SKIP: registry holds {count} device(s); short-split needs >= 2");
        return;
    }
    let mut options = ModelOptions::default();
    options.tensor_split = Some(vec![1.0; count - 1]);
    let err = Model::load_with_options(&bogus_path(), &options).unwrap_err();
    assert!(
        err.to_string().starts_with("invalid: "),
        "short split must be invalid: {err}"
    );
}

#[test]
fn main_gpu_out_of_range_rejected_under_split_none() {
    let devices = enumerate_devices();
    let cpu = devices
        .iter()
        .find(|d| d.device_type == DeviceType::Cpu)
        .expect("CPU device")
        .clone();
    let mut options = ModelOptions::default();
    options.devices = Some(vec![cpu]);
    options.split_mode = SplitMode::None;
    options.main_gpu = 1;
    let err = Model::load_with_options(&bogus_path(), &options).unwrap_err();
    let message = err.to_string();
    assert!(
        message.starts_with("invalid: ") && message.contains("main GPU index 1"),
        "out-of-range main GPU must be invalid: {message}"
    );
}

#[test]
fn main_gpu_huge_index_rejected() {
    let mut options = ModelOptions::default();
    options.main_gpu = usize::MAX;
    let err = Model::load_with_options(&bogus_path(), &options).unwrap_err();
    assert!(
        err.to_string().starts_with("invalid: "),
        "unrepresentable main GPU must be invalid: {err}"
    );
}

#[test]
fn main_gpu_ignored_unless_split_none() {
    // With LAYER splitting upstream never reads main_gpu, so an
    // out-of-range value must NOT be a validation error: the load
    // proceeds to the native call, which fails on the bogus path with
    // a plain model error.
    set_log_quiet(true);
    let mut options = ModelOptions::default();
    options.split_mode = SplitMode::Layer;
    options.main_gpu = 9999;
    let err = Model::load_with_options(&bogus_path(), &options).unwrap_err();
    set_log_quiet(false);
    assert!(
        err.to_string().starts_with("model error: "),
        "ignored main GPU must reach the native call: {err}"
    );
}

#[test]
fn offload_refused_without_gpu() {
    if supports_gpu_offload() {
        println!("SKIP: GPU present; refusal path not applicable (see gpu_offload_smoke)");
        return;
    }
    for layers in [GpuLayers::Count(1), GpuLayers::All] {
        let mut options = ModelOptions::default();
        options.gpu_layers = layers;
        let err = Model::load_with_options(&bogus_path(), &options).unwrap_err();
        let message = err.to_string();
        assert!(
            message.starts_with("unsupported: ") && message.contains("no GPU device"),
            "offload without GPU must be unsupported, never silent CPU: {message}"
        );
    }
}

// -- D. native failure mapping ----------------------------------------------

#[test]
fn tensor_split_without_devices_fails_natively() {
    // Upstream requires >= 1 device to build the TENSOR meta device,
    // even for CPU loads: with no GPU present this is a native load
    // failure (NULL), surfaced as a model error.
    if supports_gpu_offload() {
        println!("SKIP: GPU present; tensor mode may succeed there");
        return;
    }
    let Some(path) = fixture() else { return };
    set_log_quiet(true);
    let mut options = ModelOptions::default();
    options.split_mode = SplitMode::Tensor;
    let err = Model::load_with_options(&path, &options).unwrap_err();
    set_log_quiet(false);
    assert!(
        err.to_string().starts_with("model error: "),
        "tensor mode without devices must be a native failure: {err}"
    );
}

// -- E. skip-gated real offload ---------------------------------------------

#[test]
fn gpu_offload_smoke() {
    if !supports_gpu_offload() {
        println!("SKIP: no GPU device in this environment (llama_supports_gpu_offload is false)");
        return;
    }
    let Some(path) = fixture() else { return };
    let devices = enumerate_devices();
    let gpu = devices
        .iter()
        .find(|d| matches!(d.device_type, DeviceType::Gpu | DeviceType::Igpu))
        .expect("GPU device")
        .clone();
    println!(
        "offload smoke: device #{} {} ({}) [{:?}], config Count(1) + default selection",
        gpu.index, gpu.name, gpu.description, gpu.device_type
    );
    let mut options = ModelOptions::default();
    options.gpu_layers = GpuLayers::Count(1);
    let model = Model::load_with_options(&path, &options).expect("offload smoke load");
    println!(
        "offload smoke: loaded OK ({} params, {} layers)",
        model.n_params(),
        model.n_layer().unwrap()
    );
    assert_eq!(model.n_params(), 1176);
}
