//! Link `forge-sys` against the prebuilt upstream llama.cpp/ggml shared libraries.
//!
//! The native tree is built outside the workspace by `scripts/setup-native.sh`
//! (see `docs/NATIVE.md`). Its location is `$FORGE_LLAMA_DIR` (default
//! `/var/tmp/forge-native`); this script only points the linker at it and
//! records an rpath so test binaries and downstream crates find the `.so`
//! files at runtime. Nothing is compiled or downloaded here.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let root = env::var("FORGE_LLAMA_DIR").unwrap_or_else(|_| "/var/tmp/forge-native".to_string());
    let lib_dir = PathBuf::from(&root).join("build").join("bin");

    for soname in ["libggml.so", "libllama.so"] {
        if !lib_dir.join(soname).exists() {
            panic!(
                "forge-sys: {soname} not found under {}. Run `sh scripts/setup-native.sh` first (see docs/NATIVE.md).",
                lib_dir.display()
            );
        }
    }

    let header = PathBuf::from(&root)
        .join("src")
        .join("include")
        .join("llama.h");
    if header.exists() {
        let text = fs::read_to_string(&header)
            .unwrap_or_else(|error| panic!("forge-sys: cannot read {}: {error}", header.display()));
        if !text.contains("LLAMA_LOAD_MODE_MMAP_NO_PREFETCH")
            || !text.contains("llama_model_residency_contract_version")
            || !text.contains("llama_model_lifetime_state")
            || !text.contains("llama_context_lifetime_state")
            || !text.contains("llama_model_release_pages")
            || !text.contains("llama_model_tensor_mapping_count")
            || !text.contains("llama_model_release_tensor_pages")
            || !text.contains("llama_model_prefetch_tensor_pages")
            || !text.contains("llama_tensor_src_count")
            || !text.contains("llama_tensor_src_id")
        {
            panic!(
                "forge-sys: {} lacks the required native residency patch. Re-run `sh scripts/setup-native.sh` to apply patches/llama-model-residency.patch.",
                header.display()
            );
        }
        println!("cargo:rerun-if-changed={}", header.display());
    }

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    for lib in ["ggml", "ggml-base", "ggml-cpu", "llama"] {
        println!("cargo:rustc-link-lib={lib}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    println!("cargo:rerun-if-env-changed=FORGE_LLAMA_DIR");
}
