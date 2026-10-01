#!/usr/bin/env sh
# Set up the pinned upstream llama.cpp/ggml checkout and native build.
#
# Everything lives outside the workspace under $FORGE_LLAMA_DIR:
#   $FORGE_LLAMA_DIR/src    pinned read-only-ish upstream checkout (never edited)
#   $FORGE_LLAMA_DIR/build  CMake build tree with the shared libraries
#
# Usage:
#   . scripts/env.sh
#   sh scripts/setup-native.sh
#
# Prerequisites: git, a C/C++ compiler, make, and cmake on PATH
# (see docs/NATIVE.md for the ephemeral cmake install).
#
# The upstream revision is pinned by tag AND commit SHA below; the script
# aborts if the clone does not resolve to the expected SHA.
set -eu

LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_TAG="${LLAMA_TAG:-v0.5.0}"
LLAMA_SHA="${LLAMA_SHA:-7fe450e19305b828c199d602c23a8337aaa1f03b}"

: "${FORGE_LLAMA_DIR:=/var/tmp/forge-native}"
SRC="$FORGE_LLAMA_DIR/src"
BUILD="$FORGE_LLAMA_DIR/build"

if ! command -v git >/dev/null 2>&1; then
    echo "setup-native: git not found on PATH" >&2
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "setup-native: cmake not found on PATH (see docs/NATIVE.md)" >&2
    exit 1
fi

if [ ! -d "$SRC/.git" ]; then
    echo "setup-native: cloning $LLAMA_REPO@$LLAMA_TAG ..."
    rm -rf "$SRC"
    git clone --depth 1 --branch "$LLAMA_TAG" "$LLAMA_REPO" "$SRC"
else
    echo "setup-native: reusing existing checkout at $SRC"
fi

actual_sha="$(git -C "$SRC" rev-parse HEAD)"
if [ "$actual_sha" != "$LLAMA_SHA" ]; then
    echo "setup-native: SHA mismatch: got $actual_sha, want $LLAMA_SHA" >&2
    exit 1
fi
echo "setup-native: upstream revision $actual_sha verified"

echo "setup-native: configuring CMake (CPU backends, shared libs) ..."
cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF

echo "setup-native: building ..."
cmake --build "$BUILD" -j "$(nproc)"

echo "setup-native: shared libraries:"
find "$BUILD" -name "libggml*.so*" -o -name "libllama*.so*" | sort
echo "setup-native: done. FORGE_LLAMA_DIR=$FORGE_LLAMA_DIR"
