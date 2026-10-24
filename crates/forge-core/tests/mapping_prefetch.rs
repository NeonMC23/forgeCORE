//! Opt-in check that the no-prefetch load mode leaves model pages
//! nonresident until they are touched.
//!
//! Set `FORGE_PREFETCH_MODEL` to a disk-backed GGUF and run with `--ignored`.

#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};

use forge_core::{Model, ModelLoadMode, ModelOptions};

fn fixture() -> Option<PathBuf> {
    let path = std::env::var_os("FORGE_PREFETCH_MODEL")?;
    let path = PathBuf::from(path);
    path.is_file().then_some(path)
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

#[test]
#[ignore = "requires FORGE_PREFETCH_MODEL and /proc/self/smaps"]
fn no_prefetch_leaves_model_pages_nonresident() {
    let path = match fixture() {
        Some(path) => path,
        None => {
            eprintln!("skipping: set FORGE_PREFETCH_MODEL to a real GGUF");
            return;
        }
    };
    let file_size = std::fs::metadata(&path).expect("model metadata").len();

    let mut prefetch = ModelOptions::default();
    prefetch.load_mode = ModelLoadMode::Mmap;
    prefetch.use_extra_bufts = Some(false);
    let model = Model::load_with_options(&path, &prefetch).expect("prefetch load");
    let prefetch_rss = mapping_rss(&path).expect("smaps for prefetch load");
    drop(model);

    let mut no_prefetch = prefetch.clone();
    no_prefetch.load_mode = ModelLoadMode::MmapNoPrefetch;
    let model = Model::load_with_options(&path, &no_prefetch).expect("no-prefetch load");
    let no_prefetch_rss = mapping_rss(&path).expect("smaps for no-prefetch load");
    drop(model);

    eprintln!("file={file_size} prefetch_rss={prefetch_rss} no_prefetch_rss={no_prefetch_rss}");
    assert!(prefetch_rss > 0, "the prefetch mapping should be resident");
    assert!(
        no_prefetch_rss < prefetch_rss,
        "no-prefetch must retain fewer pages than the prefetch load"
    );
    assert!(
        no_prefetch_rss < file_size / 2,
        "no-prefetch retained {no_prefetch_rss} of {file_size} bytes"
    );
}
