//! Record an rpath to the prebuilt native libraries.
//!
//! `cargo:rustc-link-arg` does not propagate across crate boundaries, so
//! the rpath `forge-sys` emits never reaches final binaries. This script
//! re-emits it for `forge-core`'s own targets (tests, examples, bins).
//! Downstream crates (e.g. RAMforge) must arrange their own loader path;
//! see `docs/NATIVE.md`.

use std::env;

fn main() {
    let root = env::var("FORGE_LLAMA_DIR").unwrap_or_else(|_| "/var/tmp/forge-native".to_string());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{root}/build/bin");
    println!("cargo:rerun-if-env-changed=FORGE_LLAMA_DIR");
}
