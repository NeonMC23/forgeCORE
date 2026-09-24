# ForgeCore toolchain and cache policy

## Invariant

`/home/user/` must contain **only** `/home/user/forgeCore/`.

The repository itself must contain only source, documentation, manifests,
tests, and intentional project files — never toolchain state or caches.
In particular, none of these may exist under `/home/user/` (including
inside the repository):

- `.cargo/`, `.rustup/`
- Cargo registries, git checkouts, or caches
- compiler caches
- `target/` build output
- shell configuration files or installer leftovers

## Configuration

All Rust/Cargo state lives outside the workspace under
`$FORGE_TOOLCHAIN_ROOT` (default `/tmp/forge-toolchain`):

| Variable | Default location |
|---|---|
| `CARGO_HOME` | `/tmp/forge-toolchain/cargo` |
| `RUSTUP_HOME` | `/tmp/forge-toolchain/rustup` |
| `CARGO_TARGET_DIR` | `/tmp/forge-toolchain/target/forgeCore` |

`PATH` gains `$CARGO_HOME/bin`.

This is configured by `scripts/env.sh`, which is the persistent record
of this policy. **Source it before any cargo/rustc command:**

```sh
. /home/user/forgeCore/scripts/env.sh
```

To use a different external root (e.g. scratch storage instead of
`/tmp`), set `FORGE_TOOLCHAIN_ROOT` before sourcing:

```sh
FORGE_TOOLCHAIN_ROOT=/scratch/forge-toolchain . scripts/env.sh
```

Never point it anywhere under `/home/user/`.

## Installing the toolchain

With the environment sourced, if `cargo` is not yet installed:

```sh
curl -sSf https://sh.rustup.rs -o "$FORGE_TOOLCHAIN_ROOT/rustup-init.sh"
sh "$FORGE_TOOLCHAIN_ROOT/rustup-init.sh" -y --no-modify-path \
    --profile minimal --component rustfmt,clippy
```

`--no-modify-path` keeps the installer from touching shell files.
The toolchain binaries themselves are ephemeral (they live outside the
persisted workspace); reinstall them the same way in a fresh session.

## Verification

After any toolchain work, confirm the invariant:

```sh
ls -la /home/user/
find /home/user -maxdepth 3 \
    \( -name ".cargo" -o -name ".rustup" -o -name "target" \
    -o -name "registry" -o -name ".profile" -o -name ".bashrc" \)
echo "CARGO_HOME=$CARGO_HOME RUSTUP_HOME=$RUSTUP_HOME CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
```

The `find` must print nothing; the three variables must all point
outside `/home/user/`.
