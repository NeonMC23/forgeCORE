//! Link `forge-sys` against the prebuilt upstream llama.cpp/ggml shared libraries.
//!
//! The native tree is built outside the workspace by `scripts/setup-native.sh`
//! (see `docs/NATIVE.md`). Its location is `$FORGE_LLAMA_DIR` (default
//! `/var/tmp/forge-native`); this script only points the linker at it and
//! records an rpath so test binaries and downstream crates find the `.so`
//! files at runtime. Nothing is compiled or downloaded here.

use std::env;
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

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    for lib in ["ggml", "ggml-base", "ggml-cpu", "llama"] {
        println!("cargo:rustc-link-lib={lib}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    println!("cargo:rerun-if-env-changed=FORGE_LLAMA_DIR");
}
