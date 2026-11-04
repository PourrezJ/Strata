#!/usr/bin/env bash
# tools/hip/build_rdna2_local.sh - build the engine for one RDNA2 card (gfx1030: RX 6800 / 6900 / 6950 XT) against a ROCm SDK
# already on this PC (a TheRock tarball or wheel tree), with the host code tuned for this CPU.
#
#   ROCM_SDK=/path/to/rocm tools/hip/build_rdna2_local.sh [build-dir] [target ...]
#
# Defaults: ROCM_SDK=~/ai-runtime/rocm/nightly-10.2.0a20261018-gfx1030, build dir build-hip-gfx1030, targets strata strata-device
# and the DFlash2 checks (dflash2_q4_parity dflash2_ring_test dflash2_gemm_bench dflash2_attn_bench). STRATA_GGML_DIR is taken from
# ~/ai-runtime/src/llama-pinned-3cf03257 when it exists (the pinned llama.cpp revision; no network), else CMake fetches it.
# JOBS (default 3) keeps the HIP compiles inside 32 GB of RAM.
set -euo pipefail

here="$(cd "$(dirname "$0")/../.." && pwd)"
R="${ROCM_SDK:-$HOME/ai-runtime/rocm/nightly-10.2.0a20261018-gfx1030}"
B="${1:-$here/build-hip-gfx1030}"
shift || true
targets=("$@")
[ ${#targets[@]} -gt 0 ] || targets=(strata strata-device dflash2_q4_parity dflash2_ring_test dflash2_gemm_bench dflash2_attn_bench)
ARCH="${STRATA_HIP_ARCHS:-gfx1030}"
JOBS="${JOBS:-3}"

if [ ! -x "$R/llvm/bin/clang++" ]; then
    echo "build_rdna2_local: no ROCm SDK at $R (set ROCM_SDK)" >&2
    exit 1
fi
ggml=()
if [ -z "${STRATA_GGML_DIR:-}" ] && [ -d "$HOME/ai-runtime/src/llama-pinned-3cf03257/ggml" ]; then
    STRATA_GGML_DIR="$HOME/ai-runtime/src/llama-pinned-3cf03257"
fi
[ -n "${STRATA_GGML_DIR:-}" ] && ggml=(-DSTRATA_GGML_DIR="$STRATA_GGML_DIR")

export PATH="$R/bin:$R/llvm/bin:$PATH" ROCM_PATH="$R" HIP_PATH="$R" HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr
export LD_LIBRARY_PATH="$R/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S "$here" -B "$B" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES="$ARCH" -DCMAKE_HIP_COMPILER="$R/llvm/bin/clang++" \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="$R" -DCMAKE_PREFIX_PATH="$R" \
    -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_MMQ_KQUANTS=ON \
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON "${ggml[@]}"
cmake --build "$B" --target "${targets[@]}" -j "$JOBS"
"$B/strata" --version
