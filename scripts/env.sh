#!/usr/bin/env sh
# ForgeCore toolchain environment.
#
# Workspace invariant: /home/user/ must contain ONLY /home/user/forgeCore/.
# No .cargo/, .rustup/, registries, caches, or build output may live under
# /home/user/ (including inside the repository itself).
#
# Source this file before running any cargo/rustc command:
#
#   . /home/user/forgeCore/scripts/env.sh
#
# All Rust/Cargo state then lives under $FORGE_TOOLCHAIN_ROOT, which
# defaults to /tmp/forge-toolchain (outside the workspace). Override it
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
# See docs/TOOLCHAIN.md for the full policy.

FORGE_TOOLCHAIN_ROOT="${FORGE_TOOLCHAIN_ROOT:-/tmp/forge-toolchain}"
export FORGE_TOOLCHAIN_ROOT
export CARGO_HOME="$FORGE_TOOLCHAIN_ROOT/cargo"
export RUSTUP_HOME="$FORGE_TOOLCHAIN_ROOT/rustup"
export CARGO_TARGET_DIR="$FORGE_TOOLCHAIN_ROOT/target/forgeCore"
export PATH="$CARGO_HOME/bin:$PATH"
