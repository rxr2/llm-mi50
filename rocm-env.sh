#!/usr/bin/env bash
# rocm-env.sh — single source of truth for ROCm toolchain resolution.
#
# Sourced by preflight.sh, build.sh and benchmark-mi50.sh so all three resolve
# the SAME ROCM_PATH, PATH, LD_LIBRARY_PATH and the SAME tool paths
# (rocm-smi, rocprofv3, hipconfig, rocminfo). A READY from preflight therefore
# guarantees the benchmark will find the same rocm-smi/rocprofv3 later.
#
# STACK PURITY (review 4): once ROCM_PATH is selected, tools are NEVER taken
# from a different ROCm installation. In-tree binaries win; a tool found via
# generic PATH is accepted only if `readlink -f` proves it resolves INSIDE
# the selected ROCM_PATH (system wrapper/symlink into the same stack) —
# anything else is rejected.
#
# API:
#   rocm_env_resolve        resolve ROCM_PATH; 0 = usable tree found.
#                           Explicit ROCM_PATH (env) must be valid or we fail
#                           — never silently fall back to a different install.
#   rocm_tool NAME          print absolute path of ROCm tool NAME; 1 if it
#                           does not belong to the selected stack.
# After a successful resolve:
#   ROCM_PATH, ROCM_SMI, ROCPROFV3, HIPCONFIG, ROCMINFO
#   (exported PATH / LD_LIBRARY_PATH incl. $ROCM_PATH/extra-libs for the
#    Debian libdw1t64 workaround — install-rocm.sh extracts it there)
#
# Candidates (in order): $ROCM_PATH (if set), /opt/rocm-7.1.1, /opt/rocm, /opt/therock

_ROCM_CANDIDATE_LIST=(/opt/rocm-7.1.1 /opt/rocm /opt/therock)

_rocm_has_tools() {
  [ -n "$1" ] || return 1
  [ -x "$1/bin/hipconfig" ] || [ -x "$1/bin/rocminfo" ] || \
  [ -x "$1/bin/rocm-smi" ]  || [ -x "$1/bin/rocprofv3" ]
}

rocm_tool() {
  local n="$1" d p
  # 1) inside the selected stack — never another installation
  if [ -n "${ROCM_PATH:-}" ]; then
    for d in "$ROCM_PATH/bin" "$ROCM_PATH/lib/llvm/bin" "$ROCM_PATH/llvm/bin"; do
      if [ -x "$d/$n" ]; then
        echo "$d/$n"
        return 0
      fi
    done
  fi
  # 2) generic PATH hit accepted ONLY as a wrapper/symlink into this stack
  #    (readlink -f must land inside ROCM_PATH)
  p="$(command -v "$n" 2>/dev/null || true)"
  if [ -n "$p" ] && [ -n "${ROCM_PATH:-}" ]; then
    p="$(readlink -f "$p" 2>/dev/null || printf '%s' "$p")"
    case "$p" in
      "$ROCM_PATH"/*)
        echo "$p"
        return 0
        ;;
      *)
        # belongs to a different stack (or the OS) — REJECT (stack purity)
        ;;
    esac
  fi
  return 1
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
  # LD_LIBRARY_PATH: main libs + llvm libs + Debian extra-libs (libdw1t64
  # extracted by install-rocm.sh to $ROCM_PATH/extra-libs) when present.
  local ldextra=""
  [ -d "$ROCM_PATH/lib" ] && ldextra="$ROCM_PATH/lib"
  [ -d "$ROCM_PATH/lib/llvm/lib" ] && ldextra="$ldextra:$ROCM_PATH/lib/llvm/lib"
  [ -d "$ROCM_PATH/extra-libs" ] && ldextra="$ldextra:$ROCM_PATH/extra-libs"
  if [ -n "$ldextra" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$ROCM_PATH/lib:"*) : ;;
      *) export LD_LIBRARY_PATH="$ldextra${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi

  ROCM_SMI="$(rocm_tool rocm-smi || true)"
  ROCPROFV3="$(rocm_tool rocprofv3 || true)"
  HIPCONFIG="$(rocm_tool hipconfig || true)"
  ROCMINFO="$(rocm_tool rocminfo || true)"
  export ROCM_SMI ROCPROFV3 HIPCONFIG ROCMINFO
  return 0
}
