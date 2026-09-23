#!/usr/bin/env bash
# MI50-QWEN38 stack — reproducible build.
#
# Usage:
#   ./build.sh golden            # Variant A: upstream 42916d83 + patches 0001-0003 (default toggles = alex behaviour)
#   ./build.sh golden-upmmq      # same, but MMQ uses upstream #27841 GCN profile (A/B for prefill)
#   ./build.sh golden-dpp        # golden + DPP reductions (Stage 3 candidate)
#   ./build.sh alex-exact        # Plan B reference: alex4300 gfx906 @ f9616ce, unmodified
#   ./build.sh upstream          # Plan B reference: plain upstream 42916d83, no gfx906 patches
#
# ROCm selection: ROCM_PATH=/opt/rocm-7.1.1 (STABLE, default — same path
# install-rocm.sh writes) or ROCM_PATH=/opt/therock (MODERN)
# Output: $OUT/<variant>/ with bin/, build-info.txt, CMakeCache.txt, patch list, git status.
# After the build, gfx906 ISA verification artifacts are written to
# build-verification/disassembly/ (skip: SKIP_ISA=1).
set -euo pipefail

VARIANT="${1:-golden}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-$HOME/src/llama.cpp-mi50}"
OUT="${OUT:-$HOME/mi50-builds}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.1.1}"
JOBS="${JOBS:-$(nproc)}"

UPSTREAM_URL=https://github.com/ggml-org/llama.cpp.git
UPSTREAM_SHA=42916d83f4a225e56709f873aa8050ac11f5b6a4
ALEX_URL=https://github.com/alex4300/llama.cpp-gfx906-opt.git
ALEX_SHA=f9616ce7212cf6eceaa7cacf33e13fdd7b20d38f
EXPECTED_GOLDEN_SHA=844e42b4b37a6717d03393155e9914332f891c34   # verified: git am of 0001-0003 on 42916d83 is deterministic

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------- toolchain ----------
[ -x "$ROCM_PATH/bin/hipconfig" ] || die "ROCm not found at $ROCM_PATH (set ROCM_PATH)"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$ROCM_PATH/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
HIP_CLANG="$ROCM_PATH/lib/llvm/bin/clang"; [ -x "$HIP_CLANG" ] || HIP_CLANG="$ROCM_PATH/llvm/bin/clang"
[ -x "$HIP_CLANG" ] || die "clang not found in ROCm"
command -v cmake >/dev/null || die "cmake missing"
command -v ninja >/dev/null || die "ninja missing"
unset HSA_OVERRIDE_GFX_VERSION   # gfx906 is a native target; never override

# ---------- source ----------
mkdir -p "$(dirname "$SRC")"
if [ ! -d "$SRC/.git" ]; then
  git clone --filter=blob:none "$UPSTREAM_URL" "$SRC"
fi
cd "$SRC"
git remote get-url alex >/dev/null 2>&1 || git remote add alex "$ALEX_URL"
git fetch --filter=blob:none origin "$UPSTREAM_SHA" 2>/dev/null || git fetch origin
git fetch --filter=blob:none alex gfx906 2>/dev/null || true

git am --abort 2>/dev/null || true
git merge --abort 2>/dev/null || true
git reset -q --hard
git clean -qfdx -e models

CMAKE_EXTRA=()
case "$VARIANT" in
  golden|golden-upmmq|golden-dpp)
    git checkout -q -B "mi50-$VARIANT" "$UPSTREAM_SHA"
    # fixed identity/date => identical commit SHAs on every machine
    GIT_COMMITTER_NAME=mi50 GIT_COMMITTER_EMAIL=mi50@local \
      git am -q --committer-date-is-author-date "$HERE"/patches/000*.patch
    [ "$(git rev-parse HEAD)" = "$EXPECTED_GOLDEN_SHA" ] || die "patched SHA $(git rev-parse HEAD) != $EXPECTED_GOLDEN_SHA"
    [ "$VARIANT" = golden-upmmq ] && CMAKE_EXTRA+=(-DGGML_MMQ_GCN_PROFILE=0) || CMAKE_EXTRA+=(-DGGML_MMQ_GCN_PROFILE=1)
    [ "$VARIANT" = golden-dpp ]   && CMAKE_EXTRA+=(-DGGML_HIP_GCN_DPP=ON)   || CMAKE_EXTRA+=(-DGGML_HIP_GCN_DPP=OFF)
    ;;
  alex-exact)
    git checkout -q -B mi50-alex-exact "$ALEX_SHA"
    ;;
  upstream)
    git checkout -q -B mi50-upstream "$UPSTREAM_SHA"
    ;;
  *) die "unknown variant $VARIANT" ;;
esac

[ -z "$(git status --porcelain --untracked-files=no)" ] || die "tree dirty after checkout"

BUILD="$OUT/$VARIANT/build"
rm -rf "$BUILD"; mkdir -p "$BUILD"

HIPCXX="$HIP_CLANG" HIP_PATH="$ROCM_PATH" cmake -S . -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx906 \
  -DGPU_TARGETS=gfx906 \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  "${CMAKE_EXTRA[@]}"

cmake --build "$BUILD" -j"$JOBS" --target llama-server llama-bench llama-batched-bench test-backend-ops llama-cli

# ---------- provenance ----------
INFO="$OUT/$VARIANT/build-info.txt"
{
  echo "variant:        $VARIANT"
  echo "date:           $(date -Is)"
  echo "host:           $(hostname) $(uname -r)"
  echo "os:             $(. /etc/os-release; echo "$PRETTY_NAME")"
  echo "git HEAD:       $(git rev-parse HEAD)"
  echo "git describe:   $(git describe --always --dirty)"
  echo "upstream base:  $UPSTREAM_SHA"
  echo "patches:";      git log --format='  %h %s' "$UPSTREAM_SHA"..HEAD 2>/dev/null || true
  echo "git status:";   git status --porcelain | sed 's/^/  /'
  echo "ROCM_PATH:      $ROCM_PATH"
  echo "hipconfig:      $(hipconfig --version 2>/dev/null)"
  echo "clang:          $("$HIP_CLANG" --version | head -1)"
  echo "rocm version:   $(cat "$ROCM_PATH"/.info/version* 2>/dev/null | head -1)"
  echo "rocblas gfx906: $(ls "$ROCM_PATH"/lib/rocblas/library 2>/dev/null | grep -c gfx906) files"
  echo "cmake:          $(cmake --version | head -1)"
  echo "host cxx:       $(c++ --version | head -1)"
  echo "cmake extra:    ${CMAKE_EXTRA[*]:-none}"
  echo "sha256:"
  (cd "$BUILD/bin" && sha256sum llama-server libggml-hip.so* 2>/dev/null | sed 's/^/  /')
} > "$INFO"
cp "$BUILD/CMakeCache.txt" "$OUT/$VARIANT/CMakeCache.txt"
cat "$INFO"

# ---------- ISA verification artifacts (gfx906, no GPU needed) ----------
# Disassemble the required breit kernels (q4_0 n=1/4/8, q5_K, q6_K) with
# roc-objdump/llvm-objdump, confirm gfx906 ISA is present and that the default
# instantiations carry no private-segment (scratch). No performance claims are
# made here — that needs the MI50 (T0/T1). Artifacts: build-verification/disassembly/.
if [ "${SKIP_ISA:-0}" != 1 ]; then
  ROCM_PATH="$ROCM_PATH" "$HERE/verify-isa.sh" "$BUILD"
else
  echo "SKIP_ISA=1 — no ISA verification artifacts generated"
fi

# ---------- mandatory correctness gate (needs the GPU) ----------
if [ "${SKIP_TESTS:-0}" != 1 ] && rocminfo 2>/dev/null | grep -q gfx906; then
  for op in MUL_MAT MUL_MAT_ID FLASH_ATTN_EXT GATED_DELTA_NET; do
    "$BUILD/bin/test-backend-ops" test -o "$op" -b ROCm0 2>&1 | tail -2 | tee -a "$OUT/$VARIANT/test-backend-ops.txt"
  done
  # fused GEMV path (gate/up + GLU) used by Q4_0 n=1
  "$BUILD/bin/test-backend-ops" test -o MUL_MAT_VEC_FUSION -b ROCm0 2>&1 | tail -2 | tee -a "$OUT/$VARIANT/test-backend-ops.txt" || true
  grep -q "FAIL" "$OUT/$VARIANT/test-backend-ops.txt" && die "test-backend-ops FAILED — do not benchmark this build"
fi
echo "OK: $OUT/$VARIANT/build/bin/llama-server"
