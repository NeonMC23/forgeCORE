#!/usr/bin/env sh
# ForgeCore toolchain environment.
#
# Workspace invariant: /home/user/ must contain ONLY /home/user/forgeCORE/.
# No .cargo/, .rustup/, registries, caches, build output, native checkouts,
# or CMake trees may live under /home/user/ (including inside the repo).
#
# Source this file before running any cargo/rustc command:
#
#   . /home/user/forgeCORE/scripts/env.sh
#
# All Rust/Cargo state then lives under $FORGE_TOOLCHAIN_ROOT, which
# defaults to /var/tmp/forge-toolchain (outside the workspace; /tmp is a small tmpfs here). Override it
# if /tmp is unsuitable in your environment:
#
#   FORGE_TOOLCHAIN_ROOT=/scratch/forge-toolchain . scripts/env.sh
#
# If no toolchain is installed there yet, install one with:
#
#   curl -sSf https://sh.rustup.rs -o "$FORGE_TOOLCHAIN_ROOT/rustup-init.sh"
#   sh "$FORGE_TOOLCHAIN_ROOT/rustup-init.sh" -y --no-modify-path \
#       --profile minimal --component rustfmt,clippy
#
# Native llama.cpp/ggml state lives under $FORGE_LLAMA_DIR (default
# /var/tmp/forge-native); see scripts/setup-native.sh and docs/NATIVE.md.
#
# See docs/TOOLCHAIN.md for the full policy.

FORGE_TOOLCHAIN_ROOT="${FORGE_TOOLCHAIN_ROOT:-/var/tmp/forge-toolchain}"
export FORGE_TOOLCHAIN_ROOT
export CARGO_HOME="$FORGE_TOOLCHAIN_ROOT/cargo"
export RUSTUP_HOME="$FORGE_TOOLCHAIN_ROOT/rustup"
export CARGO_TARGET_DIR="$FORGE_TOOLCHAIN_ROOT/target/forgeCore"
export PATH="$CARGO_HOME/bin:$PATH"

FORGE_LLAMA_DIR="${FORGE_LLAMA_DIR:-/var/tmp/forge-native}"
export FORGE_LLAMA_DIR

# cmake from a previous native setup, if present (ephemeral by design).
if [ -d "$FORGE_LLAMA_DIR/tools" ]; then
    for _forge_cmake_dir in "$FORGE_LLAMA_DIR"/tools/cmake-*/bin; do
        if [ -d "$_forge_cmake_dir" ]; then
            PATH="$_forge_cmake_dir:$PATH"
        fi
    done
    unset _forge_cmake_dir
fi
