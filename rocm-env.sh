#!/usr/bin/env bash
# rocm-env.sh — single source of truth for ROCm toolchain resolution.
#
# Sourced by preflight.sh, build.sh and benchmark-mi50.sh so all three resolve
# the SAME ROCM_PATH, PATH, LD_LIBRARY_PATH and the SAME tool paths
# (rocm-smi, rocprofv3, hipconfig, rocminfo). A READY from preflight therefore
# guarantees the benchmark will find the same rocm-smi/rocprofv3 later.
#
# API:
#   rocm_env_resolve        resolve ROCM_PATH; 0 = usable tree found.
#                           Explicit ROCM_PATH (env) must be valid or we fail
#                           — never silently fall back to a different install.
#   rocm_tool NAME          print absolute path of ROCm tool NAME; 1 if missing
#                           (searched only inside ROCM_PATH trees, then PATH)
# After a successful resolve:
#   ROCM_PATH, ROCM_SMI, ROCPROFV3, HIPCONFIG, ROCMINFO  (exported PATH/LD_LIBRARY_PATH)
#
# Candidates (in order): $ROCM_PATH (if set), /opt/rocm-7.1.1, /opt/rocm, /opt/therock

_ROCM_CANDIDATE_LIST=(/opt/rocm-7.1.1 /opt/rocm /opt/therock)

_rocm_has_tools() {
  [ -n "$1" ] || return 1
  [ -x "$1/bin/hipconfig" ] || [ -x "$1/bin/rocminfo" ] || \
  [ -x "$1/bin/rocm-smi" ]  || [ -x "$1/bin/rocprofv3" ]
}

rocm_tool() {
  local n="$1" d
  if [ -n "${ROCM_PATH:-}" ]; then
    for d in "$ROCM_PATH/bin" "$ROCM_PATH/lib/llvm/bin" "$ROCM_PATH/llvm/bin"; do
      [ -x "$d/$n" ] && { echo "$d/$n"; return 0; }
    done
  fi
  command -v "$n" 2>/dev/null || return 1
}

rocm_env_resolve() {
  local explicit="${ROCM_PATH:-}" r found=""
  local candidates=()
  [ -n "$explicit" ] && candidates+=("$explicit")
  candidates+=("${_ROCM_CANDIDATE_LIST[@]}")

  for r in "${candidates[@]}"; do
    if _rocm_has_tools "$r"; then found="$r"; break; fi
  done

  if [ -n "$explicit" ] && [ "$found" != "$explicit" ]; then
    echo "ERROR: ROCM_PATH=$explicit does not contain ROCm tools" >&2
    return 1
  fi
  [ -n "$found" ] || return 1
  export ROCM_PATH="$found"

  case ":$PATH:" in
    *":$ROCM_PATH/bin:"*) : ;;
    *) export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$ROCM_PATH/llvm/bin${PATH:+:$PATH}" ;;
  esac
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$ROCM_PATH/lib:"*) : ;;
    *) export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac

  ROCM_SMI="$(rocm_tool rocm-smi || true)"
  ROCPROFV3="$(rocm_tool rocprofv3 || true)"
  HIPCONFIG="$(rocm_tool hipconfig || true)"
  ROCMINFO="$(rocm_tool rocminfo || true)"
  export ROCM_SMI ROCPROFV3 HIPCONFIG ROCMINFO
  return 0
}
