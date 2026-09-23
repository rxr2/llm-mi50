#!/usr/bin/env bash
# preflight.sh — THE single pre-flight command for the MI50 benchmark stack.
#
#   ./preflight.sh [--build DIR] [--model FILE] [--report DIR] [--soft]
#
# Ends with exactly one of:
#   READY_FOR_MI50_BENCHMARK
#   NOT_READY:
#     - <specific reason>
#     - ...
#
# Checks (strict; every failure is a reason):
#   * MI50 / gfx906 detected (lspci + rocminfo)
#   * /dev/kfd exists
#   * model file exists
#   * model SHA256 (and size) match the exact provenance  [ALLOW_MODEL_MISMATCH=1 overrides]
#   * correct build binaries exist (llama-server, llama-bench, test-backend-ops)
#   * ROCm is identified (path + version)
#   * HSA_OVERRIDE_GFX_VERSION is unset
#   * GPU PCIe link information is recorded (lspci -vv + sysfs link speed/width/numa_node)
#   * power cap is within the expected stock range (default 220-226 W; POWER_CAP_MIN/MAX)
#
# Also collected into the report dir (item 11 - NUMA / PCIe preflight):
#   lscpu -e=CPU,SOCKET,NODE,CORE   numactl --hardware   lspci -tv
#   GPU numa_node, current_link_speed, current_link_width
#   dual-socket: the NUMA node attached to the MI50 (drives NUMA_MODE=auto)
#
# --soft: print the same report but always exit 0 (used by benchmark-mi50.sh
#         in DRY_RUN mode; real benchmarks always run the strict version).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=model-provenance.sh
. "$HERE/model-provenance.sh"

BUILD=""; MODEL_ARG=""; REPORT="$HERE/preflight-report"; SOFT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --build)  BUILD="${2:?--build needs a dir}"; shift 2 ;;
    --model)  MODEL_ARG="${2:?--model needs a file}"; shift 2 ;;
    --report) REPORT="${2:?--report needs a dir}"; shift 2 ;;
    --soft)   SOFT=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1 (see --help)" >&2; exit 1 ;;
  esac
done

ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.1.1}"   # same path install-rocm.sh 'stable' writes
POWER_CAP_MIN="${POWER_CAP_MIN:-220}"
POWER_CAP_MAX="${POWER_CAP_MAX:-226}"
ALLOW_MODEL_MISMATCH="${ALLOW_MODEL_MISMATCH:-0}"
SMI="${SMI:-rocm-smi}"
GPU="${GPU:-0}"
ROCM_PROBES=$(printf '%s\n' "$ROCM_PATH" /opt/rocm-7.1.1 /opt/rocm /opt/therock | awk '!seen[$0]++' | paste -sd' ' -)

FAIL=(); WARN=()
fail() { FAIL+=("$1"); }
warn() { WARN+=("$1"); }

mkdir -p "$REPORT"
SUMMARY="$REPORT/summary.txt"
exec > >(tee "$SUMMARY") 2>&1

echo "=== MI50 preflight — $(date -Is) — report: $REPORT"

# ---------------------------------------------------------------- GPU / gfx906
GPU_LINE=""
if command -v lspci >/dev/null 2>&1; then
  GPU_LINE=$(lspci -D 2>/dev/null | grep -Ei 'vega 20|mi50|mi60|radeon vii|66a[0-7]' | head -1 || true)
  lspci -tv > "$REPORT/lspci-tv.txt" 2>/dev/null || true
  lspci -nn > "$REPORT/lspci-nn.txt" 2>/dev/null || true
else
  fail "lspci not found (pciutils missing) — cannot detect the MI50"
fi

if [ -n "$GPU_LINE" ]; then
  echo "[PASS] MI50 GPU present: $GPU_LINE"
  GPU_BDF=$(awk '{print $1}' <<<"$GPU_LINE")
else
  [ -n "${GPU_LINE+x}" ] && [ -z "$GPU_LINE" ] && fail "MI50 / gfx906 GPU not detected on PCI"
  GPU_BDF=""
fi

# ROCm identification ---------------------------------------------------------
ROCM_FOUND=""
for R in "$ROCM_PATH" /opt/rocm-7.1.1 /opt/rocm /opt/therock; do
  if [ -x "$R/bin/rocminfo" ] || [ -x "$R/bin/hipconfig" ]; then
    ROCM_FOUND="$R"; break
  fi
done
if [ -n "$ROCM_FOUND" ]; then
  ROCM_VER=$(cat "$ROCM_FOUND"/.info/version* 2>/dev/null | head -1 || true)
  [ -z "$ROCM_VER" ] && ROCM_VER=$("$ROCM_FOUND/bin/hipconfig" --version 2>/dev/null | head -1 || true)
  echo "[PASS] ROCm identified: $ROCM_FOUND ${ROCM_VER:+($ROCM_VER)}"
  if [ -x "$ROCM_FOUND/bin/rocminfo" ]; then
    "$ROCM_FOUND/bin/rocminfo" > "$REPORT/rocminfo.txt" 2>/dev/null || true
    if grep -Eq 'gfx906|amdgcn-amd-amdhsa--gfx906' "$REPORT/rocminfo.txt"; then
      echo "[PASS] gfx906 target reported by rocminfo"
    else
      fail "rocminfo does not report gfx906 (no usable MI50 agent / driver problem)"
    fi
  else
    fail "rocminfo missing in $ROCM_FOUND — cannot confirm gfx906"
  fi
else
  fail "ROCm not identified (tried: $ROCM_PROBES) — run ./install-rocm.sh stable"
fi

# /dev/kfd --------------------------------------------------------------------
if [ -e /dev/kfd ]; then
  echo "[PASS] /dev/kfd exists"
else
  fail "/dev/kfd missing (amdgpu kernel driver not loaded?)"
fi

# HSA override ----------------------------------------------------------------
if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
  fail "HSA_OVERRIDE_GFX_VERSION is set ('$HSA_OVERRIDE_GFX_VERSION') — unset it (gfx906 is native)"
else
  echo "[PASS] HSA_OVERRIDE_GFX_VERSION is unset"
fi

# ---------------------------------------------------------------- build binaries
if [ -z "$BUILD" ]; then
  BUILD="$HOME/mi50-builds/golden/build"
  [ -d "$BUILD/bin" ] || [ -d "$BUILD" ] || BUILD=""
fi
BIN=""
if [ -n "$BUILD" ]; then
  if [ -x "$BUILD/bin/llama-server" ]; then BIN="$BUILD/bin";
  elif [ -x "$BUILD/llama-server" ]; then BIN="$BUILD";
  fi
fi
if [ -n "$BIN" ]; then
  echo "[PASS] build binaries dir: $BIN"
  for b in llama-server llama-bench test-backend-ops; do
    if [ -x "$BIN/$b" ]; then echo "        + $b"; else fail "missing binary: $BIN/$b (run ./build.sh golden)"; fi
  done
  for b in llama-cli llama-batched-bench; do
    [ -x "$BIN/$b" ] || warn "optional binary missing: $BIN/$b"
  done
else
  fail "build not found (tried --build '$BUILD', default ~/mi50-builds/golden/build) — run ./build.sh golden or pass --build DIR"
fi

# ---------------------------------------------------------------- model + provenance
MODEL=""
if [ -n "$MODEL_ARG" ]; then
  MODEL="$MODEL_ARG"
else
  for c in "${MODEL:-}" "/models/$MODEL_FILENAME" "$HOME/models/$MODEL_FILENAME" \
           "$HERE/models/$MODEL_FILENAME" "$HERE/$MODEL_FILENAME"; do
    [ -n "$c" ] && [ -f "$c" ] && { MODEL="$c"; break; }
  done
fi
if [ -n "$MODEL" ] && [ -f "$MODEL" ]; then
  echo "[PASS] model exists: $MODEL"
  echo "        provenance: $MODEL_REPO_URL / $MODEL_FILENAME"
  echo "        expected sha256: $EXPECTED_MODEL_SHA"
  echo "        expected size:   $EXPECTED_MODEL_SIZE bytes"
  ACT_SIZE=$(stat -c%s "$MODEL" 2>/dev/null || echo 0)
  ACT_SHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
  if [ "$ACT_SHA" = "$EXPECTED_MODEL_SHA" ] && [ "$ACT_SIZE" = "$EXPECTED_MODEL_SIZE" ]; then
    echo "[PASS] model SHA256 + size match the exact provenance"
  elif [ "$ALLOW_MODEL_MISMATCH" = 1 ]; then
    warn "model mismatch OVERRIDDEN by ALLOW_MODEL_MISMATCH=1 (sha=$ACT_SHA size=$ACT_SIZE)"
  else
    fail "model SHA256/size mismatch: got sha=$ACT_SHA size=$ACT_SIZE — expected $EXPECTED_MODEL_SHA / $EXPECTED_MODEL_SIZE (override: ALLOW_MODEL_MISMATCH=1)"
  fi
  {
    echo "repo_url=$MODEL_REPO_URL"
    echo "filename=$MODEL_FILENAME"
    echo "path=$MODEL"
    echo "expected_sha256=$EXPECTED_MODEL_SHA"
    echo "actual_sha256=$ACT_SHA"
    echo "expected_size=$EXPECTED_MODEL_SIZE"
    echo "actual_size=$ACT_SIZE"
    echo "allow_mismatch=$ALLOW_MODEL_MISMATCH"
  } > "$REPORT/model-provenance.txt"
else
  fail "model not found (tried --model, /models/$MODEL_FILENAME, ~/models/, repo models/)"
fi

# ---------------------------------------------------------------- power cap
if command -v "$SMI" >/dev/null 2>&1 || [ -x "${ROCM_FOUND:-/nonexistent}/bin/$SMI" ]; then
  SMIBIN="$SMI"; [ -x "${ROCM_FOUND:-}/bin/$SMI" ] && SMIBIN="$ROCM_FOUND/bin/$SMI"
  "$SMIBIN" -d "$GPU" --showmaxpower --showpower --showtemp > "$REPORT/power-status.txt" 2>&1 || true
  CAP=$("$SMIBIN" -d "$GPU" --showmaxpower 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1 || true)
  if [ -z "$CAP" ]; then
    fail "could not read power cap via $SMIBIN"
  elif awk -v c="$CAP" -v lo="$POWER_CAP_MIN" -v hi="$POWER_CAP_MAX" 'BEGIN{exit !(c>=lo && c<=hi)}'; then
    echo "[PASS] power cap ${CAP} W within stock range ${POWER_CAP_MIN}-${POWER_CAP_MAX} W"
  else
    fail "power cap ${CAP} W outside expected stock range ${POWER_CAP_MIN}-${POWER_CAP_MAX} W (run: sudo ./power.sh stock)"
  fi
else
  fail "rocm-smi not found — cannot verify the power cap"
fi

# ---------------------------------------------------------------- NUMA / PCIe collection (item 11)
lscpu -e=CPU,SOCKET,NODE,CORE > "$REPORT/lscpu-numa.txt" 2>/dev/null \
  || lscpu > "$REPORT/lscpu-numa.txt" 2>/dev/null \
  || warn "lscpu unavailable"
if command -v numactl >/dev/null 2>&1; then
  numactl --hardware > "$REPORT/numactl-hardware.txt" 2>&1 || true
else
  warn "numactl not installed (NUMA_MODE=auto will fall back to off)"
fi
SOCKETS=$(lscpu 2>/dev/null | awk -F: '/Socket\(s\)/{gsub(/ /,"",$2); print $2; exit}')
GPU_NUMA_NODE=""; LINK_SPEED=""; LINK_WIDTH=""
if [ -n "$GPU_BDF" ]; then
  lspci -vv -D -s "$GPU_BDF" > "$REPORT/pcie-link.txt" 2>&1 || true
  DEV="/sys/bus/pci/devices/$GPU_BDF"
  GPU_NUMA_NODE=$(cat "$DEV/numa_node" 2>/dev/null || echo "")
  LINK_SPEED=$(cat "$DEV/current_link_speed" 2>/dev/null || echo "")
  LINK_WIDTH=$(cat "$DEV/current_link_width" 2>/dev/null || echo "")
  # fall back to LnkSta from lspci -vv when sysfs attrs are absent
  if [ -z "$LINK_SPEED" ] && [ -s "$REPORT/pcie-link.txt" ]; then
    LINK_SPEED=$(grep -oE 'Speed [^,]+' "$REPORT/pcie-link.txt" | head -1 || true)
    LINK_WIDTH=$(grep -oE 'Width x[0-9]+' "$REPORT/pcie-link.txt" | head -1 || true)
  fi
  {
    echo "bdf=$GPU_BDF"
    echo "numa_node=$GPU_NUMA_NODE"
    echo "current_link_speed=$LINK_SPEED"
    echo "current_link_width=$LINK_WIDTH"
    echo "model=$GPU_LINE"
    echo "sockets=${SOCKETS:-?}"
  } > "$REPORT/pcie-sysfs.txt"
  if [ -n "$LINK_SPEED" ] && [ -n "$LINK_WIDTH" ]; then
    echo "[PASS] GPU PCIe link recorded: $GPU_BDF speed='$LINK_SPEED' width='$LINK_WIDTH' numa_node='$GPU_NUMA_NODE'"
  else
    fail "GPU PCIe link information could not be recorded (bdf=$GPU_BDF speed='$LINK_SPEED' width='$LINK_WIDTH')"
  fi
  if [ "${SOCKETS:-1}" -gt 1 ] 2>/dev/null; then
    if [ -n "$GPU_NUMA_NODE" ] && [ "$GPU_NUMA_NODE" -ge 0 ] 2>/dev/null; then
      echo "[PASS] dual-socket machine: MI50 is on NUMA node $GPU_NUMA_NODE (NUMA_MODE=auto will bind CPU+memory there)"
    else
      warn "dual-socket machine but GPU numa_node='$GPU_NUMA_NODE' unknown — NUMA_MODE=auto cannot bind"
    fi
  fi
else
  fail "GPU PCI address unknown — cannot record PCIe link / NUMA information"
fi

# ---------------------------------------------------------------- verdict
echo
if [ ${#WARN[@]} -gt 0 ]; then
  echo "warnings:"
  for w in "${WARN[@]}"; do echo "  - $w"; done
fi
{
  echo "checked_at=$(date -Is)"
  echo "rocm_path=${ROCM_FOUND:-none}"
  echo "gpu_bdf=${GPU_BDF:-}"
  echo "gpu_numa_node=${GPU_NUMA_NODE:-}"
  echo "link_speed=${LINK_SPEED:-}"
  echo "link_width=${LINK_WIDTH:-}"
  echo "power_cap_w=${CAP:-unknown}"
  echo "model=${MODEL:-not-found}"
} > "$REPORT/facts.env"

if [ ${#FAIL[@]} -eq 0 ]; then
  echo "READY_FOR_MI50_BENCHMARK"
  exit 0
else
  echo "NOT_READY:"
  for f in "${FAIL[@]}"; do echo "  - $f"; done
  if [ "$SOFT" = 1 ]; then
    echo "(--soft: reporting only, exit 0)"
    exit 0
  fi
  exit 1
fi
