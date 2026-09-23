#!/usr/bin/env bash
# benchmark-mi50.sh — prepared, NOT yet run on the MI50.
#
#   ./benchmark-mi50.sh <run-name> <build-dir> <model.gguf> [tests]
#   tests: comma list of T0..T10 (default: all)       DRY_RUN=1 prints commands only
#
# Output: results/YYYY-MM-DD/<run-name>/
#   meta.json          SHA, ROCm, driver, kernel, clocks, power cap, temps, model sha256, env, cmdlines
#   T*/...             raw logs + CSV per test
#   requests.csv       one row per server request (all T4-T10): workload, seed, ctx, n-max, TPS, acceptance, latency
#   summary.md         generated at the end
#
# Tests
#   T0  provenance + correctness gate (test-backend-ops MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT, GATED_DELTA_NET)
#   T1  kernel micro-bench: MUL_MAT for the model's shapes, n = 1..16 (MMVQ/MMQ cutover table)
#   T2  llama-bench native: pp512, pp2048, tg128 (prefill + native decode)
#   T3  verify-width step time: llama-bench -p 1,2,3,4,5,6,8,12,16 -n 0 (cost of an MTP verify batch)
#   T4  server native (no speculation), 6 workloads
#   T5  server draft-mtp,ngram-mod n-max 2
#   T6  server draft-mtp,ngram-mod n-max 3   (golden)
#   T7  server draft-mtp,ngram-mod n-max 4 ; + draft-mtp only n-max 3 ; + n-max 2 / ngram 64
#   T8  long context 8k / 32k / 64k prompt, f16 KV (and q8_0 KV as a TEST ONLY arm)
#   T9  HIP graphs ON vs OFF (golden config)
#   T10 soak 30 min + greedy determinism hash
set -euo pipefail

RUN="${1:?run-name}"; BUILD="${2:?build dir containing bin/}"; MODEL="${3:?model.gguf}"
TESTS="${4:-T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$BUILD/bin"; [ -d "$BIN" ] || BIN="$BUILD"
OUT="$HERE/results/$(date +%F)/$RUN"
PROMPTS="$HERE/prompts"
WORKLOADS="${WORKLOADS:-CODING-1 CODING-2 EDIT JSON AGENT PROSE}"
REPS="${REPS:-3}"             # repetitions per workload/config (seeds 42, 43, 44)
NPRED="${NPRED:-768}"         # tokens to generate per request
TEMP="${TEMP:-0.6}"           # production sampling (Qwen defaults); T10 uses greedy
CTX="${CTX:-65536}"
PORT="${PORT:-18080}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
SMI="${SMI:-rocm-smi}"
EXPECTED_MODEL_SHA=ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d

export HIP_VISIBLE_DEVICES="$GPU"
unset HSA_OVERRIDE_GFX_VERSION GGML_CUDA_Q8_1_CACHE
mkdir -p "$OUT"
LOG="$OUT/bench.log"
log() { echo "[$(date +%T)] $*" | tee -a "$LOG" >&2; }
run() { log "+ $*"; [ "$DRY_RUN" = 1 ] && return 0; "$@"; }
want() { [[ ",$TESTS," == *",$1,"* ]]; }

# ---------------------------------------------------------------- monitor
MON_PID=""
mon_start() { # $1 = csv file
  [ "$DRY_RUN" = 1 ] && return 0
  ( echo "ts,power_w,sclk_mhz,mclk_mhz,temp_junction_c,use_pct"
    while :; do
      $SMI -d "$GPU" --showpower --showclocks --showtemp --showuse --csv 2>/dev/null | tail -n +2 | \
        awk -v ts="$(date +%s)" -F, '{print ts","$0}' | head -1
      sleep 1
    done ) > "$1" 2>/dev/null &
  MON_PID=$!
}
mon_stop() { [ -n "$MON_PID" ] && kill "$MON_PID" 2>/dev/null || true; MON_PID=""; }

# ---------------------------------------------------------------- server
SRV_PID=""
srv_start() { # $1 = logfile, rest = extra args
  local lf="$1"; shift
  local cmd=("$BIN/llama-server" -m "$MODEL" -ngl 999 -fa on -c "$CTX" -np 1 --host 127.0.0.1 --port "$PORT"
             --jinja --metrics -lv 4 "$@")
  echo "${cmd[*]}" >> "$OUT/cmdlines.txt"
  log "server: ${cmd[*]} (env: ${SRV_ENV:-})"
  [ "$DRY_RUN" = 1 ] && return 0
  env ${SRV_ENV:-} "${cmd[@]}" > "$lf" 2>&1 &
  SRV_PID=$!
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SRV_PID" 2>/dev/null || { log "server died, see $lf"; tail -20 "$lf" >&2; return 1; }
    sleep 1
  done
  log "server did not become healthy"; return 1
}
srv_stop() { [ -n "$SRV_PID" ] && { kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null || true; }; SRV_PID=""; sleep 5; }
trap 'mon_stop; srv_stop' EXIT

# one request -> one CSV row in requests.csv (+ raw json)
REQ_CSV="$OUT/requests.csv"
[ -f "$REQ_CSV" ] || echo "test,config,workload,rep,seed,temp,ctx,n_max,prompt_n,prompt_ms,prefill_tps,predicted_n,predicted_ms,effective_tps,draft_n,draft_accepted,acceptance,verify_steps_est,ms_per_verify_est,wall_s,ttft_ms,output_sha256" > "$REQ_CSV"
request() { # test config workload rep n_max [n_predict] [temp] [prompt_file]
  local t="$1" cfg="$2" wl="$3" rep="$4" nmax="$5" np="${6:-$NPRED}" temp="${7:-$TEMP}" pf="${8:-$PROMPTS/$3.txt}"
  local seed=$((42 + rep)) raw="$OUT/$t/raw/${cfg}_${wl}_r${rep}.json"
  mkdir -p "$(dirname "$raw")"
  [ "$DRY_RUN" = 1 ] && { log "request $t $cfg $wl r$rep seed=$seed n_max=$nmax"; return 0; }
  python3 - "$PORT" "$pf" "$np" "$temp" "$seed" "$raw" "$t" "$cfg" "$wl" "$rep" "$CTX" "$nmax" >> "$REQ_CSV" <<'EOF'
import sys, json, time, hashlib, urllib.request
port, pf, npred, temp, seed, raw, t, cfg, wl, rep, ctx, nmax = sys.argv[1:]
prompt = open(pf, encoding="utf-8").read()
body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": int(npred), "temperature": float(temp),
        "top_p": 0.95, "top_k": 20, "seed": int(seed), "stream": True, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time(); ttft = None; text = []; timings = {}
with urllib.request.urlopen(req, timeout=1800) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        for ch in ev.get("choices", []):
            c = (ch.get("delta") or {}).get("content")
            if c:
                if ttft is None: ttft = (time.time() - t0) * 1000
                text.append(c)
        if "timings" in ev: timings = ev["timings"]
wall = time.time() - t0
out = "".join(text)
json.dump({"timings": timings, "output": out, "wall_s": wall, "ttft_ms": ttft}, open(raw, "w"), indent=1)
g = lambda k: timings.get(k, 0) or 0
pn, pms, dn, da = g("predicted_n"), g("predicted_ms"), g("draft_n"), g("draft_n_accepted")
eff = pn / pms * 1000 if pms else 0
acc = da / dn if dn else 0
nm = int(nmax) if nmax not in ("", "0") else 0
# verify steps: every step drafts up to n_max tokens -> steps ~= draft_n / n_max (exact count is in the server log "mean len")
steps = dn / nm if (nm and dn) else pn
mspv = pms / steps if steps else 0
print(",".join(map(str, [t, cfg, wl, rep, seed, temp, ctx, nmax, g("prompt_n"), round(g("prompt_ms"), 2),
      round(g("prompt_per_second"), 2), pn, round(pms, 2), round(eff, 3), dn, da, round(acc, 4), round(steps, 1),
      round(mspv, 3), round(wall, 3), round(ttft or 0, 1), hashlib.sha256(out.encode()).hexdigest()[:16]])))
EOF
  tail -1 "$REQ_CSV" | awk -F, '{printf "   -> %s %s %s: %s tok/s eff, acc %s, ttft %s ms\n",$1,$2,$3,$14,$17,$21}' | tee -a "$LOG" >&2
}

# per-position acceptance + mean len from the server trace log (-lv 4)
acc_per_pos() { # logfile -> appends to acc_per_pos.csv
  local lf="$1" cfg="$2"
  [ "$DRY_RUN" = 1 ] && return 0
  grep -E "draft acceptance|acc per pos" "$lf" | paste - - 2>/dev/null | \
    sed -E 's/.*draft acceptance = ([0-9.]+).*mean len = *([0-9.]+).*acc per pos = \(([^)]*)\).*/\1;\2;\3/' | \
    awk -v c="$cfg" -F';' '{print c","$1","$2",\""$3"\""}' >> "$OUT/acc_per_pos.csv" || true
}

server_suite() { # test cfg n_max spec_args...
  local t="$1" cfg="$2" nmax="$3"; shift 3
  mkdir -p "$OUT/$t"
  srv_start "$OUT/$t/server_${cfg}.log" "$@" || return 1
  request "$t" "$cfg" "PROSE" 0 "$nmax" 64 >/dev/null 2>&1 || true   # warm-up, discarded
  [ "$DRY_RUN" = 1 ] || sed -i '$d' "$REQ_CSV"
  mon_start "$OUT/$t/power_${cfg}.csv"
  for wl in $WORKLOADS; do for rep in $(seq 0 $((REPS-1))); do request "$t" "$cfg" "$wl" "$rep" "$nmax"; done; done
  mon_stop
  srv_stop
  acc_per_pos "$OUT/$t/server_${cfg}.log" "$cfg"
}

# ================================================================= T0 provenance
if want T0; then
  mkdir -p "$OUT/T0"
  [ "$DRY_RUN" = 1 ] || {
    MODEL_SHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
    [ "$MODEL_SHA" = "$EXPECTED_MODEL_SHA" ] || log "WARNING: model sha256 $MODEL_SHA != expected unsloth Q4_0"
    python3 - "$OUT/meta.json" <<EOF
import json, subprocess as sp, os, sys
def sh(c):
    try: return sp.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"ERR {e}"
m = {
 "run": "$RUN", "date": sh("date -Is"), "host": sh("hostname"), "kernel": sh("uname -r"),
 "os": sh(". /etc/os-release; echo \$PRETTY_NAME"),
 "build_dir": "$BUILD", "build_info": sh("cat '$BUILD/../build-info.txt' 2>/dev/null"),
 "llama_version": sh("'$BIN/llama-server' --version 2>&1 | tail -2"),
 "git_sha": sh("'$BIN/llama-server' --version 2>&1 | grep -o 'commit [0-9a-f]*'"),
 "rocm": sh("cat \${ROCM_PATH:-/opt/rocm}/.info/version* 2>/dev/null | head -1; hipconfig --version 2>/dev/null"),
 "amdgpu_driver": sh("modinfo amdgpu 2>/dev/null | grep -E '^(filename|version|vermagic)'"),
 "vbios": sh("$SMI -d $GPU --showvbios 2>/dev/null | grep -i vbios"),
 "power_cap": sh("$SMI -d $GPU --showmaxpower 2>/dev/null | grep -iE 'max|cap'"),
 "clocks": sh("$SMI -d $GPU --showclocks 2>/dev/null | grep -E 'sclk|mclk'"),
 "temps": sh("$SMI -d $GPU --showtemp 2>/dev/null | grep -i temp"),
 "perf_level": sh("$SMI -d $GPU --showperflevel 2>/dev/null | grep -i level"),
 "model": "$MODEL", "model_sha256": "$MODEL_SHA", "ctx": $CTX, "reps": $REPS, "n_predict": $NPRED, "temp": $TEMP,
 "env": {k: v for k, v in os.environ.items() if k.startswith(("GGML_", "HIP_", "HSA_", "ROCM", "LLAMA_"))},
 "cpu": sh("lscpu | grep 'Model name'"), "mem": sh("free -g | head -2"),
}
json.dump(m, open(sys.argv[1], "w"), indent=1)
EOF
  }
  for op in MUL_MAT MUL_MAT_ID FLASH_ATTN_EXT GATED_DELTA_NET; do
    run bash -c "'$BIN/test-backend-ops' test -o $op -b ROCm0 > '$OUT/T0/test_$op.txt' 2>&1; tail -3 '$OUT/T0/test_$op.txt'"
  done
  [ "$DRY_RUN" = 1 ] || ! grep -l "FAIL" "$OUT"/T0/test_*.txt >/dev/null 2>&1 || { log "CORRECTNESS FAIL — stopping"; exit 2; }
fi

# ================================================================= T1 kernel micro-bench (cutover table)
if want T1; then
  mkdir -p "$OUT/T1"
  # model shapes (k x m): qkv 5120x10240, gate 5120x6144, ffn_up/gate 5120x17408, ffn_down 17408x5120,
  # ssm_out 6144x5120 (Q5_K), attn_q 5120x12288, output 5120x248320 (Q6_K). Types present: q4_0 q4_1 q5_K q6_K q8_0.
  # test-backend-ops perf runs its built-in MUL_MAT grid; we filter to our types and n<=16.
  run bash -c "'$BIN/test-backend-ops' perf -o MUL_MAT -b ROCm0 > '$OUT/T1/mul_mat_perf.txt' 2>&1 || true"
  [ "$DRY_RUN" = 1 ] || grep -E "type_a=(q4_0|q4_1|q4_K|q5_K|q6_K|q8_0|iq4_xs)" "$OUT/T1/mul_mat_perf.txt" | \
      grep -E "n=([1-9]|1[0-6])," > "$OUT/T1/mul_mat_n1-16.txt" || true
  # dispatch A/B inside the same binary: generic vs wide GEMV
  for sw in GGML_MMVQ_Q4_BREIT_GFX906=0 GGML_MMVQ_Q5K_BREIT_GFX906=0 GGML_MMVQ_Q6K_BREIT_GFX906=0 GGML_MMVQ_Q41_BREIT_GFX906=0; do
    run bash -c "env $sw '$BIN/test-backend-ops' perf -o MUL_MAT -b ROCm0 > '$OUT/T1/mul_mat_perf_${sw%%=*}_off.txt' 2>&1 || true"
  done
fi

# ================================================================= T2 native llama-bench
if want T2; then
  mkdir -p "$OUT/T2"; mon_start "$OUT/T2/power.csv"
  run bash -c "'$BIN/llama-bench' -m '$MODEL' -ngl 99 -fa on -p 512,2048 -n 128 -r 5 -o csv > '$OUT/T2/llama-bench.csv' 2> '$OUT/T2/llama-bench.err'"
  mon_stop
fi

# ================================================================= T3 verify width
if want T3; then
  mkdir -p "$OUT/T3"; mon_start "$OUT/T3/power.csv"
  run bash -c "'$BIN/llama-bench' -m '$MODEL' -ngl 99 -fa on -p 1,2,3,4,5,6,7,8,9,12,16 -n 0 -ub 512 -r 10 -o csv > '$OUT/T3/verify-width.csv' 2> '$OUT/T3/verify-width.err'"
  # same at depth: step cost with 16k tokens already in KV
  run bash -c "'$BIN/llama-bench' -m '$MODEL' -ngl 99 -fa on -p 1,4,8 -n 0 -d 16384 -r 5 -o csv > '$OUT/T3/verify-width-d16k.csv' 2>> '$OUT/T3/verify-width.err'"
  mon_stop
fi

# ================================================================= T4-T7 server workloads
want T4 && server_suite T4 native     0
want T5 && server_suite T5 mtp2       2 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2
want T6 && server_suite T6 mtp3       3 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3
if want T7; then
  server_suite T7 mtp4          4 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4
  server_suite T7 mtponly3      3 --spec-type draft-mtp           --spec-draft-n-max 3
  server_suite T7 mtp2_ngram64  2 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2 --spec-draft-n-max-ngram 64
fi

# ================================================================= T8 long context
if want T8; then
  mkdir -p "$OUT/T8"
  # build long prompts by prepending repository-like filler (source code) to CODING-1
  for n in 8000 32000 64000; do
    f="$OUT/T8/prompt_${n}.txt"
    [ "$DRY_RUN" = 1 ] || python3 - "$n" "$PROMPTS/CODING-1.txt" "$f" <<'EOF'
import sys
n, src, dst = int(sys.argv[1]), sys.argv[2], sys.argv[3]
unit = "\n".join(f"def helper_{i}(x: int, y: int) -> int:\n    '''helper {i}'''\n    return (x * {i} + y) % {997 + i}\n" for i in range(50))
words_per_unit = len(unit.split())
reps = max(1, int(n * 0.70 / words_per_unit))   # ~1.4 tokens per word for code
body = "Context: the following module is part of the repository.\n```python\n" + unit * reps + "\n```\n\n"
open(dst, "w").write(body + open(src).read())
EOF
  done
  for kv in f16 q8_0; do
    CTX_SAVE=$CTX; CTX=73728
    srv_start "$OUT/T8/server_kv${kv}.log" --cache-type-k $kv --cache-type-v $kv --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 || continue
    mon_start "$OUT/T8/power_kv${kv}.csv"
    for n in 8000 32000 64000; do request T8 "kv${kv}_${n}" "CODING-1" 0 3 256 "$TEMP" "$OUT/T8/prompt_${n}.txt"; done
    mon_stop; srv_stop; CTX=$CTX_SAVE
    acc_per_pos "$OUT/T8/server_kv${kv}.log" "T8_kv${kv}"
  done
fi

# ================================================================= T9 HIP graphs ON/OFF
if want T9; then
  WORKLOADS_SAVE="$WORKLOADS"; WORKLOADS="CODING-1 PROSE"
  server_suite T9 graphs_on  3 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3
  SRV_ENV="GGML_CUDA_DISABLE_GRAPHS=1" server_suite T9 graphs_off 3 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3
  SRV_ENV=""; WORKLOADS="$WORKLOADS_SAVE"
  # native too: graphs matter most at n=1
  run bash -c "'$BIN/llama-bench' -m '$MODEL' -ngl 99 -fa on -p 0 -n 128 -r 5 -o csv > '$OUT/T9/tg_graphs_on.csv' 2>/dev/null"
  run bash -c "GGML_CUDA_DISABLE_GRAPHS=1 '$BIN/llama-bench' -m '$MODEL' -ngl 99 -fa on -p 0 -n 128 -r 5 -o csv > '$OUT/T9/tg_graphs_off.csv' 2>/dev/null"
fi

# ================================================================= T10 soak + determinism
if want T10; then
  mkdir -p "$OUT/T10"
  srv_start "$OUT/T10/server.log" --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 || exit 1
  mon_start "$OUT/T10/power.csv"
  # greedy determinism: same prompt twice must give the same hash (MTP must not change greedy output)
  for rep in 0 1; do request T10 greedy_mtp3 CODING-1 "$rep" 3 512 0; done
  end=$(( $(date +%s) + ${SOAK_S:-1800} )); i=0
  while [ "$DRY_RUN" != 1 ] && [ "$(date +%s)" -lt "$end" ]; do
    for wl in $WORKLOADS; do request T10 soak "$wl" "$i" 3; done; i=$((i+1))
  done
  mon_stop; srv_stop
  srv_start "$OUT/T10/server_native.log" || exit 1
  for rep in 0 1; do request T10 greedy_native CODING-1 "$rep" 0 512 0; done
  srv_stop
  [ "$DRY_RUN" = 1 ] || grep -cE "error|ERROR|abort|hipError" "$OUT/T10/server.log" > "$OUT/T10/error_count.txt" || true
fi

# ================================================================= summary
[ "$DRY_RUN" = 1 ] && { log "dry run done: $OUT"; exit 0; }
python3 "$HERE/summarize.py" "$OUT" > "$OUT/summary.md" && log "summary: $OUT/summary.md"
