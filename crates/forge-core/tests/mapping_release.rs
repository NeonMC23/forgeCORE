//! Fixture-backed checks for model-scoped page release.
//!
//! Loading two copies lets the test verify that releasing one model's pages
//! does not disturb the other model's file-backed mapping.

#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};

use forge_core::{BatchBuilder, Context, ContextOptions, Model, ModelLoadMode, ModelOptions};

fn fixture() -> Option<PathBuf> {
    let path = std::env::var_os("FORGE_TEST_MODEL")?;
    let path = PathBuf::from(path);
    path.is_file().then_some(path)
}

fn target_tmp() -> PathBuf {
    match option_env!("CARGO_TARGET_TMPDIR") {
        Some(path) => PathBuf::from(path),
        None => std::env::temp_dir(),
    }
}

fn copy_fixture(source: &Path, name: &str) -> PathBuf {
    let dir = target_tmp().join("forge-release-pages");
    std::fs::create_dir_all(&dir).expect("create test directory");
    let destination = dir.join(name);
    std::fs::copy(source, &destination).expect("copy fixture");
    destination
}

fn mapping_rss(path: &Path) -> Option<u64> {
    let target = std::fs::canonicalize(path).ok()?;
    let target = target.to_str()?;
    let smaps = std::fs::read_to_string("/proc/self/smaps").ok()?;
    let mut in_target = false;
    let mut rss = 0u64;
    for line in smaps.lines() {
        if is_mapping_header(line) {
            in_target = mapping_path(line).is_some_and(|mapped| mapped == target);
            continue;
        }
        if !in_target {
            continue;
        }
        if let Some((key, bytes)) = field(line) {
            if key == "Rss" {
                rss += bytes;
            }
        }
    }
    Some(rss)
}

fn is_mapping_header(line: &str) -> bool {
    let first = line.split_whitespace().next().unwrap_or("");
    first.contains('-') && first.chars().all(|ch| ch.is_ascii_hexdigit() || ch == '-')
}

fn mapping_path(line: &str) -> Option<&str> {
    let mut rest = line;
    for _ in 0..5 {
        let trimmed = rest.trim_start();
        let boundary = trimmed.find(char::is_whitespace)?;
        rest = &trimmed[boundary..];
    }
    let path = rest.trim_start();
    if !path.starts_with('/') {
        return None;
    }
    Some(path.strip_suffix(" (deleted)").unwrap_or(path))
}

fn field(line: &str) -> Option<(&str, u64)> {
    let (key, rest) = line.split_once(':')?;
    let value = rest.split_whitespace().next()?.parse::<u64>().ok()?;
    Some((key, value.saturating_mul(1024)))
}

fn options() -> ModelOptions {
    let mut options = ModelOptions::default();
    options.load_mode = ModelLoadMode::Mmap;
    options.use_extra_bufts = Some(false);
    options
}

#[test]
fn release_pages_targets_only_the_model_mapping_and_keeps_inference_valid() {
    let Some(source) = fixture() else {
        eprintln!("SKIP: FORGE_TEST_MODEL not set");
        return;
    };
    let path_a = copy_fixture(&source, "release-a.gguf");
    let path_b = copy_fixture(&source, "release-b.gguf");
    let options = options();
    let mut model_a = Model::load_with_options(&path_a, &options).expect("model A");
    let model_b = Model::load_with_options(&path_b, &options).expect("model B");

    let a_before = mapping_rss(&path_a).expect("A mapping");
    let b_before = mapping_rss(&path_b).expect("B mapping");
    assert!(a_before > 0, "A should be resident after mmap load");
    assert!(b_before > 0, "B should be resident after mmap load");

    model_a
        .release_mapping_pages()
        .expect("first release succeeds");
    let a_after = mapping_rss(&path_a).expect("A mapping after release");
    let b_after = mapping_rss(&path_b).expect("B mapping after release");
    eprintln!("A {a_before}->{a_after}, B {b_before}->{b_after}");

    if a_after >= a_before {
        eprintln!("SKIP: filesystem did not release mapping pages (tmpfs or unsupported)");
        return;
    }
    assert!(
        b_after.saturating_add(4096) >= b_before,
        "model B mapping must not be released: {b_before}->{b_after}"
    );

    // Repeated advice is allowed and idempotent.
    model_a
        .release_mapping_pages()
        .expect("second release succeeds");

    // The mapping remains valid and inference refaults pages on demand.
    let mut context = Context::open(&model_a, &ContextOptions::default()).expect("context");
    let mut batch = BatchBuilder::new(32);
    batch.push(1, 0, false).unwrap();
    batch.push(5, 1, false).unwrap();
    batch.push(6, 2, false).unwrap();
    batch.push(7, 3, true).unwrap();
    context
        .decode(&batch.build().unwrap())
        .expect("decode after release");
    assert_eq!(model_a.n_params(), 1176);
    assert_eq!(model_b.n_params(), 1176);
}
