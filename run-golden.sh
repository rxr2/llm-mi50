#!/usr/bin/env bash
# MI50-QWEN38-GOLDEN-BASELINE  (a reproducible reference point, NOT claimed to be the fastest config)
#
# Build : ./build.sh golden   -> upstream 42916d83 + patches 0001-0003 = 844e42b4b
# Model : exact provenance in model-provenance.sh:
#         Hugging Face repo : https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
#         exact filename    : Qwen3.8-27B-Q4_0.gguf
#         exact SHA256      : ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d
#         expected size     : 16 056 478 688 bytes
# GPU   : 1x MI50 32GB, stock power cap 225 W, no OC, inbox amdgpu, no HSA override
#
# Knobs (env), each maps to one row of the A/B plan:
#   BIN=/path/to/bin    explicit binaries dir (else auto-detected, see below)
#   CTX=65536         context (f16 KV: 2k 0.13 / 8k 0.54 / 32k 2.1 / 64k 4.3 / 128k 8.6 GB)
#   NMAX=3            MTP draft depth (2 if acceptance < 50 %, 3-4 if >= 50 %; start fixed, no adaptive)
#   NGRAM_NMAX=0      0 = n-gram drafter shares NMAX; >0 = separate deep n-gram cap (A/B only)
#   SPEC=draft-mtp,ngram-mod   ("draft-mtp" alone for the MTP-only arm, "none" for native)
#   GRAPHS=1          0 -> GGML_CUDA_DISABLE_GRAPHS=1
#   KV=f16            do NOT switch to q8_0 without the T8 test
#   PORT=8080
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?usage: run-golden.sh /path/Qwen3.8-27B-Q4_0.gguf}"

# ---- binary discovery ------------------------------------------------------
# 1. explicit BIN wins;
# 2. otherwise find the bare-metal golden build at ~/mi50-builds/golden/build/bin;
# 3. otherwise fall back to the repo-local ./bin ONLY if it exists;
# never silently default to ./bin when it does not exist.
if [ -n "${BIN:-}" ]; then
  [ -x "$BIN/llama-server" ] || { echo "ERROR: BIN=$BIN has no llama-server" >&2; exit 1; }
elif [ -x "$HOME/mi50-builds/golden/build/bin/llama-server" ]; then
  BIN="$HOME/mi50-builds/golden/build/bin"
elif [ -x "$HERE/bin/llama-server" ]; then
  BIN="$HERE/bin"
else
  echo "ERROR: llama-server not found." >&2
  echo "  tried: BIN (unset), ~/mi50-builds/golden/build/bin, $HERE/bin" >&2
  echo "  build it with ./build.sh golden, or set BIN=/path/to/bin explicitly." >&2
  exit 1
fi

CTX="${CTX:-65536}"; NMAX="${NMAX:-3}"; NGRAM_NMAX="${NGRAM_NMAX:-0}"
SPEC="${SPEC:-draft-mtp,ngram-mod}"; GRAPHS="${GRAPHS:-1}"; KV="${KV:-f16}"; PORT="${PORT:-8080}"

unset HSA_OVERRIDE_GFX_VERSION
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
[ "$GRAPHS" = 0 ] && export GGML_CUDA_DISABLE_GRAPHS=1
# never enable the alex Q8_1 cache together with graphs (crash)
unset GGML_CUDA_Q8_1_CACHE

ARGS=(-m "$MODEL" -ngl 999 -fa on -c "$CTX" -np 1
      --cache-type-k "$KV" --cache-type-v "$KV"
      --host 0.0.0.0 --port "$PORT" --jinja --metrics
      --temp 0.6 --top-p 0.95 --top-k 20)
if [ "$SPEC" != none ]; then
  ARGS+=(--spec-type "$SPEC" --spec-draft-n-max "$NMAX")
  [ "$NGRAM_NMAX" != 0 ] && ARGS+=(--spec-draft-n-max-ngram "$NGRAM_NMAX")
fi
echo "exec: $BIN/llama-server ${ARGS[*]}" >&2
exec "$BIN/llama-server" "${ARGS[@]}" ${EXTRA_ARGS:-}
