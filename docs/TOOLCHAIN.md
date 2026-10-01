# ForgeCore toolchain and cache policy

## Invariant

`/home/user/` must contain **only** `/home/user/forgeCORE/`.

The repository itself must contain only source, documentation, manifests,
tests, and intentional project files — never toolchain state or caches.
In particular, none of these may exist under `/home/user/` (including
inside the repository):

- `.cargo/`, `.rustup/`
- Cargo registries, git checkouts, or caches
- compiler caches
- `target/` build output
- native source checkouts or CMake build trees
- shell configuration files or installer leftovers

## Configuration

All Rust/Cargo state lives outside the workspace under
`$FORGE_TOOLCHAIN_ROOT` (default `/var/tmp/forge-toolchain`):

| Variable | Default location |
|---|---|
| `CARGO_HOME` | `/var/tmp/forge-toolchain/cargo` |
| `RUSTUP_HOME` | `/var/tmp/forge-toolchain/rustup` |
| `CARGO_TARGET_DIR` | `/var/tmp/forge-toolchain/target/forgeCore` |

`PATH` gains `$CARGO_HOME/bin`.

Native llama.cpp/ggml state lives under `$FORGE_LLAMA_DIR` (default
`/var/tmp/forge-native`); see `scripts/setup-native.sh` and
`docs/NATIVE.md`.

`/var/tmp` is used instead of `/tmp` because `/tmp` is a small tmpfs
(~1 GB) in this environment while `/var/tmp` sits on the main disk.
Both are outside the persisted workspace, so toolchain binaries are
ephemeral by design and are reinstalled per session. Override the roots
if needed, but never point them anywhere under `/home/user/`.

This is configured by `scripts/env.sh`, which is the persistent record
of this policy. **Source it before any cargo/rustc command:**

```sh
. /home/user/forgeCORE/scripts/env.sh
```

## Installing the toolchain

With the environment sourced, if `cargo` is not yet installed:

```sh
curl -sSf https://sh.rustup.rs -o "$FORGE_TOOLCHAIN_ROOT/rustup-init.sh"
sh "$FORGE_TOOLCHAIN_ROOT/rustup-init.sh" -y --no-modify-path \
    --profile minimal --component rustfmt,clippy
```

`--no-modify-path` keeps the installer from touching shell files.

## Verification

After any toolchain work, confirm the invariant:

```sh
ls -la /home/user/
find /home/user -maxdepth 3 \
    \( -name ".cargo" -o -name ".rustup" -o -name "target" \
    -o -name "registry" -o -name ".profile" -o -name ".bashrc" \)
echo "CARGO_HOME=$CARGO_HOME RUSTUP_HOME=$RUSTUP_HOME CARGO_TARGET_DIR=$CARGO_TARGET_DIR FORGE_LLAMA_DIR=$FORGE_LLAMA_DIR"
```

The `find` must print nothing; all locations must point outside
`/home/user/`.
