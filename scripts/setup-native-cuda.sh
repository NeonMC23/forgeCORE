#!/usr/bin/env sh
# Configure/build a CUDA-enabled native backend in a separate build tree.
#
# The script never installs packages, reloads modules, or changes the pinned
# source. It requires an already working NVIDIA driver and CUDA toolkit.
#
# Usage:
#   . scripts/env.sh                                   # optional toolchain env
#   sh scripts/setup-native-cuda.sh --check            # preflight only
#   sh scripts/setup-native-cuda.sh --configure-only   # configure, no build
#   sh scripts/setup-native-cuda.sh                    # configure + build
#
# After a successful build:
#   export FORGE_LLAMA_BUILD_DIR=build-cuda
#   export FORGE_LLAMA_DIR=...                         # same tree as before
#   cargo test --workspace
#
# The CPU build in $FORGE_LLAMA_DIR/build is left untouched.
set -eu

LLAMA_SHA="${LLAMA_SHA:-7fe450e19305b828c199d602c23a8337aaa1f03b}"

: "${FORGE_LLAMA_DIR:=/var/tmp/forge-native}"
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
SRC="$FORGE_LLAMA_DIR/src"
BUILD="${FORGE_CUDA_BUILD_DIR:-$FORGE_LLAMA_DIR/build-cuda}"
PATCH="$ROOT/patches/llama-model-residency.patch"

MODE=build
case "${1:-}" in
    --check) MODE=check ;;
    --configure-only) MODE=configure-only ;;
    "") ;;
    *)
        echo "usage: $0 [--check|--configure-only]" >&2
        exit 2
        ;;
esac

fail() {
    echo "setup-native-cuda: $*" >&2
    exit 1
}

if [ ! -d "$SRC/.git" ]; then
    fail "native source not found at $SRC; run scripts/setup-native.sh first"
fi
actual_sha="$(git -C "$SRC" rev-parse HEAD)"
if [ "$actual_sha" != "$LLAMA_SHA" ]; then
    fail "native SHA mismatch: got $actual_sha, want $LLAMA_SHA"
fi
if ! git -C "$SRC" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    fail "required residency/placement patch is not applied; run scripts/setup-native.sh"
fi

echo "setup-native-cuda: pinned source $actual_sha verified"
echo "setup-native-cuda: build tree $BUILD"

# 1. Driver/runtime must be usable, not merely installed.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found; install the NVIDIA driver utilities first"
fi
if ! nvidia_smi="$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>&1)"; then
    echo "$nvidia_smi" >&2
    fail "nvidia-smi cannot initialize; fix the driver/kernel-module mismatch (usually a reboot) before building CUDA"
fi
echo "setup-native-cuda: GPU: $nvidia_smi"

# 2. Compiler and headers.
if ! command -v nvcc >/dev/null 2>&1; then
    fail "nvcc not found; install the CUDA development toolkit (Nobara: sudo dnf install cuda-devel cuda-gcc gcc15-c++)"
fi
echo "setup-native-cuda: $(nvcc --version | tail -1)"

# 3. Architecture. GA107/RTX 3050 is compute capability 8.6; prefer the
#    reported value and fall back to 86 only when the query is unavailable.
CUDA_ARCH="${CUDA_ARCH:-}"
if [ -z "$CUDA_ARCH" ]; then
    compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]')"
    case "$compute_cap" in
        [0-9]*.[0-9]*) CUDA_ARCH="$(printf '%s' "$compute_cap" | tr -d '.')" ;;
        *) CUDA_ARCH=86 ;;
    esac
fi
echo "setup-native-cuda: CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}-real"

HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER:-}"
if [ -z "$HOST_COMPILER" ]; then
    for candidate in g++-15 g++15 g++; do
        if command -v "$candidate" >/dev/null 2>&1; then
            HOST_COMPILER="$(command -v "$candidate")"
            break
        fi
    done
fi
if [ -n "$HOST_COMPILER" ]; then
    echo "setup-native-cuda: CUDA host compiler $HOST_COMPILER"
else
    echo "setup-native-cuda: warning: no explicit CUDA host compiler; nvcc may reject an unsupported system GCC" >&2
fi

if [ "$MODE" = check ]; then
    echo "setup-native-cuda: preflight passed (no configure/build performed)"
    exit 0
fi

if ! command -v cmake >/dev/null 2>&1; then
    fail "cmake not found on PATH"
fi

echo "setup-native-cuda: configuring CUDA backend (CPU retained) ..."
if [ -n "$HOST_COMPILER" ]; then
    cmake -S "$SRC" -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_CUDA=ON \
        -DGGML_CPU=ON \
        -DGGML_CPU_REPACK=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_CUDA_NCCL=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}-real" \
        -DCMAKE_CUDA_HOST_COMPILER="$HOST_COMPILER" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_APP=OFF
else
    cmake -S "$SRC" -B "$BUILD" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_CUDA=ON \
        -DGGML_CPU=ON \
        -DGGML_CPU_REPACK=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_CUDA_NCCL=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}-real" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_APP=OFF
fi

if [ "$MODE" = configure-only ]; then
    echo "setup-native-cuda: configured. Run without --configure-only to build."
    exit 0
fi

echo "setup-native-cuda: building ..."
cmake --build "$BUILD" -j "$(nproc)"

cuda_lib="$(find "$BUILD" -name 'libggml-cuda.so*' -print -quit)"
if [ -z "$cuda_lib" ]; then
    fail "build produced no libggml-cuda.so; inspect the CMake CUDA output"
fi
echo "setup-native-cuda: CUDA backend artifact: $cuda_lib"
if command -v nm >/dev/null 2>&1; then
    if ! nm -D "$cuda_lib" 2>/dev/null | grep -q 'ggml_backend_cuda_reg'; then
        fail "CUDA artifact does not expose ggml_backend_cuda_reg"
    fi
    echo "setup-native-cuda: CUDA backend registration symbol present"
fi

echo "setup-native-cuda: done"
echo "  export FORGE_LLAMA_DIR=$FORGE_LLAMA_DIR"
echo "  export FORGE_LLAMA_BUILD_DIR=$(basename "$BUILD")"
echo "  then rerun the forgeCORE and RAMforge suites"
