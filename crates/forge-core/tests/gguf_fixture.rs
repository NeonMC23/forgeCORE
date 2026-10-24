//! Shared-fixture GGUF structural contract.
//!
//! This is the forgeCORE half of the parity contract with RAMforge's
//! independent GGUF reader. Both tests load the same `FORGE_TEST_MODEL`
//! fixture and assert the same descriptor names, payload lengths, version,
//! alignment, and architecture. They do not link the two crates together;
//! divergence is caught by running both suites against the same generated
//! file. See `docs/GGUF_OWNERSHIP.md`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use forge_core::inspect_gguf_path;

fn fixture() -> Option<PathBuf> {
    match std::env::var("FORGE_TEST_MODEL") {
        Ok(path) => Some(PathBuf::from(path)),
        Err(_) => {
            println!("SKIP: FORGE_TEST_MODEL not set (see docs/NATIVE.md)");
            None
        }
    }
}

fn expected_tensor_lengths() -> BTreeMap<String, u64> {
    [
        ("token_embd.weight", 1024),
        ("blk.0.attn_norm.weight", 32),
        ("blk.0.attn_q.weight", 256),
        ("blk.0.attn_k.weight", 256),
        ("blk.0.attn_v.weight", 256),
        ("blk.0.attn_output.weight", 256),
        ("blk.0.ffn_norm.weight", 32),
        ("blk.0.ffn_gate.weight", 512),
        ("blk.0.ffn_up.weight", 512),
        ("blk.0.ffn_down.weight", 512),
        ("output_norm.weight", 32),
        ("output.weight", 1024),
    ]
    .into_iter()
    .map(|(name, bytes)| (name.to_string(), bytes))
    .collect()
}

#[test]
fn tiny_fixture_matches_shared_structural_contract() {
    let Some(path) = fixture() else {
        return;
    };
    let inspection = inspect_gguf_path(&path).expect("inspect fixture");
    assert_eq!(inspection.version, 3);
    assert_eq!(inspection.alignment, 32);
    assert_eq!(
        inspection
            .metadata
            .get("general.architecture")
            .and_then(|value| value.as_str()),
        Some("llama")
    );

    let actual: BTreeMap<String, u64> = inspection
        .tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.byte_length))
        .collect();
    assert_eq!(actual, expected_tensor_lengths());
    assert_eq!(actual.values().sum::<u64>(), 4704);
}
