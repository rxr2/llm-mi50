#!/usr/bin/env bash
# benchmark-mi50.sh — prepared, NOT yet run on the MI50.
#
#   ./benchmark-mi50.sh <run-name> <build-dir> <model.gguf> [tests]
#   tests: comma list of T0..T10 (default: all)
#   DRY_RUN=1             print commands only — no hardware needed, finishes fast
#   PROFILE=1             also run a SHORT fixed workload under rocprofv3
#                         (HIP graphs OFF; kernel trace + names/durations/count +
#                          raw profiler files; never profiles the full suite)
#   NUMA_MODE=auto|off    auto (default) binds CPU+memory to the GPU-local NUMA
#                         node; the binding is saved in meta.json
#   ALLOW_MODEL_MISMATCH=1  explicit override of the model SHA256/size preflight check
#   TEMP_MAX_C=95         thermal watchdog: junction > 95 C terminates server AND
#                         benchmark and marks the run INVALID_THERMAL
#   POWER_CAP_MIN/MAX     expected stock power-cap range (default 220..226 W)
#
# A STRICT preflight (./preflight.sh) runs before ANY benchmark and the run
# aborts unless it prints READY_FOR_MI50_BENCHMARK. Under DRY_RUN the same
# checks run in --soft (report-only) mode.
#
# Output: results/YYYY-MM-DD/<run-name>/
#   preflight/        preflight report: PCIe link, NUMA, lscpu, power, provenance
#   meta.json         SHA, ROCm, driver, kernel, clocks, power cap, temps,
#                     model provenance (repo/filename/sha256/size), NUMA binding,
#                     PCIe link info, env, cmdlines
#   power_caps.csv    ACTUAL power cap recorded before every test group —
#                     summarize.py never aggregates runs with different caps
#   INVALID_THERMAL   written only if the thermal watchdog fired (run aborted)
#   T*/...            raw logs + CSV per test
#   requests.csv      one row per server request (T4-T10): workload, seed, ctx,
#                     n-max, TPS, acceptance, latency (derived values labeled ESTIMATE)
#   profile/          PROFILE=1: kernel trace, kernel names/durations/count, raw files
#   summary.md        generated at the end
#
# Tests
#   T0  provenance + correctness gate (test-backend-ops MUL_MAT, MUL_MAT_ID,
#       FLASH_ATTN_EXT, GATED_DELTA_NET, MUL_MAT_VEC_FUSION — same strict ops
#       as the build gate; any FAIL stops the run)
#   T1  kernel micro-bench: MUL_MAT for the model's shapes, n = 1..16 (MMVQ/MMQ cutover table)
#   T2  llama-bench native: pp512, pp2048, tg128 (prefill + native decode)
#   T3-SYNTHETIC-NCOL (test id: T3)
#         Purpose: characterize the n-column GGML/GEMV/MMQ cost for n = 1..16
#         (llama-bench -p N -n 0 measures prompt processing at width N).
#         It is NOT an MTP verification measurement and must never be reported
#         as actual MTP verification latency. Kept because it is useful.
#   T4  server native (no speculation), 6 workloads
#   T5  server draft-mtp,ngram-mod n-max 2
#   T6  server draft-mtp,ngram-mod n-max 3   (golden)
#   T7  server draft-mtp,ngram-mod n-max 4 ; + draft-mtp only n-max 3 ; + n-max 2 / ngram 64
#         -> T5/T6/T7 carry the REAL MTP verification measurements: EXACT
#            BEFORE/AFTER /metrics Prometheus counter deltas per request
#            (verification_steps, draft_tokens, accepted_tokens,
#            mean_*_per_step, mean_target_width = 1 + draft/step,
#            per-position survivals r[i]=P(A>=i+1)), plus API draft_n /
#            draft_n_accepted, predicted_ms, wall time, and the kernel trace
#            when PROFILE=1 was used.
#   T8  long context 8k / 32k / 64k prompts built with the Qwen/llama tokenizer
#       (server /tokenize endpoint — NO word-count estimates); the ACTUAL prompt
#       token count is recorded (prompt_tokens.csv + requests.csv prompt_n);
#       f16 KV (and q8_0 KV as a TEST ONLY arm)
#   T9  HIP graphs ON vs OFF (golden config)
#   T10 soak 30 min + greedy determinism hash
set -euo pipefail

RUN="${1:?run-name}"; BUILD="${2:?build dir containing bin/}"; MODEL="${3:?model.gguf}"
TESTS="${4:-T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=model-provenance.sh
. "$HERE/model-provenance.sh"

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
PROFILE="${PROFILE:-0}"
NUMA_MODE="${NUMA_MODE:-auto}"
TEMP_MAX_C="${TEMP_MAX_C:-95}"
ALLOW_MODEL_MISMATCH="${ALLOW_MODEL_MISMATCH:-0}"
POWER_CAP_MIN="${POWER_CAP_MIN:-220}"
POWER_CAP_MAX="${POWER_CAP_MAX:-226}"
ALLOW_APPEND="${ALLOW_APPEND:-0}"                  # run isolation (5): default FAIL on existing data
ALLOW_POWER_MISMATCH="${ALLOW_POWER_MISMATCH:-0}"  # power override (6): explicit env only

# ---- run isolation (5): an existing run dir WITH benchmark data must FAIL
if [ "$ALLOW_APPEND" != 1 ] && [ -d "$OUT" ]; then
  has_data=0
  for m in requests.csv meta.json power_caps.csv cmdlines.txt summary.md \
           acc_per_pos.csv INVALID_THERMAL bench.log; do
    [ -e "$OUT/$m" ] && has_data=1
  done
  compgen -G "$OUT/T[0-9]*" >/dev/null && has_data=1
  [ -d "$OUT/profile" ] && has_data=1
  if [ "$has_data" = 1 ]; then
    echo "ERROR: run dir $OUT already contains benchmark data — refusing to append a second experiment." >&2
    echo "       Pick a new run name, or explicitly ALLOW_APPEND=1 to continue anyway." >&2
    exit 1
  fi
fi

export HIP_VISIBLE_DEVICES="$GPU"
export GPU            # meta.json collector reads it via os.environ
unset HSA_OVERRIDE_GFX_VERSION GGML_CUDA_Q8_1_CACHE
mkdir -p "$OUT"
LOG="$OUT/bench.log"
log() { echo "[$(date +%T)] $*" | tee -a "$LOG" >&2; }
run() { log "+ $*"; [ "$DRY_RUN" = 1 ] && return 0; "$@"; }
want() { [[ ",$TESTS," == *",$1,"* ]]; }

# Shared ROCm resolution (3) — same rocm-env.sh preflight/build source, so a
# READY from preflight guarantees these tool paths exist for the benchmark.
# shellcheck source=rocm-env.sh
. "$HERE/rocm-env.sh"
rocm_env_resolve || true     # a failure surfaces as preflight NOT_READY reasons below
SMI="${SMI:-rocm-smi}"
[ "$SMI" = rocm-smi ] && SMI="${ROCM_SMI:-rocm-smi}"
ROCPROF="${ROCPROF:-${ROCPROFV3:-rocprofv3}}"

# ---------------------------------------------------------------- preflight (STRICT, before ANY benchmark)
PREFLIGHT_ARGS=(--build "$BUILD" --model "$MODEL" --report "$OUT/preflight")
if [ "$DRY_RUN" = 1 ]; then
  bash "$HERE/preflight.sh" "${PREFLIGHT_ARGS[@]}" --soft || true
else
  bash "$HERE/preflight.sh" "${PREFLIGHT_ARGS[@]}" || {
    log "preflight reported NOT_READY — refusing to start any benchmark"; exit 2; }
fi
echo $$ > "$OUT/.bench_pid"

# ---------------------------------------------------------------- NUMA setup (item 11)
NUMA_NODE=""
NUMA_RUN=()
numa_setup() {
  case "$NUMA_MODE" in
    off) return 0 ;;
    auto) : ;;
    *) log "invalid NUMA_MODE='$NUMA_MODE' (valid: auto|off)"; exit 1 ;;
  esac
  command -v lspci >/dev/null 2>&1 || { log "NUMA_MODE=auto: lspci missing — binding off"; return 0; }
  local bdf node
  bdf=$(lspci -D 2>/dev/null | grep -Ei 'vega 20|mi50|mi60|radeon vii|66a[0-7]' | awk '{print $1}' | head -1 || true)
  [ -n "$bdf" ] || { log "NUMA_MODE=auto: GPU PCI device not found — binding off"; return 0; }
  node=$(cat "/sys/bus/pci/devices/$bdf/numa_node" 2>/dev/null || echo -1)
  case "$node" in ''|*[!0-9-]*) node=-1 ;; esac
  if [ "$node" -lt 0 ] 2>/dev/null; then log "NUMA_MODE=auto: GPU numa_node unknown — binding off"; return 0; fi
  command -v numactl >/dev/null 2>&1 || { log "NUMA_MODE=auto: numactl missing — binding off"; return 0; }
  NUMA_NODE="$node"
  NUMA_RUN=(numactl --cpunodebind="$node" --membind="$node")
  log "NUMA_MODE=auto: binding CPU + memory to GPU-local NUMA node $node (saved in meta.json)"
}
numa_setup
NPFX="${NUMA_RUN[*]:-}"

# ---------------------------------------------------------------- power cap per test group (5/6: FAIL outside stock range)
record_cap() { # $1 = test, $2 = config
  [ "$DRY_RUN" = 1 ] && return 0
  local f="$OUT/power_caps.csv" cap
  [ -f "$f" ] || echo "test,config,power_cap_w,ts" > "$f"
  cap=$($SMI -d "$GPU" --showmaxpower 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1 || true)
  echo "$1,$2,${cap:-unknown},$(date -Is)" >> "$f"
  if [ -z "$cap" ]; then
    if [ "$ALLOW_POWER_MISMATCH" = 1 ]; then
      log "WARNING: test group $1/$2 power cap unreadable — ALLOWED by ALLOW_POWER_MISMATCH=1"
    else
      log "ERROR: test group $1/$2 cannot read the power cap via $SMI — aborting real benchmark"
      log "       (override: ALLOW_POWER_MISMATCH=1 — never the default)"
      exit 1
    fi
  elif ! awk -v c="$cap" -v lo="$POWER_CAP_MIN" -v hi="$POWER_CAP_MAX" 'BEGIN{exit !(c>=lo && c<=hi)}'; then
    if [ "$ALLOW_POWER_MISMATCH" = 1 ]; then
      log "WARNING: test group $1/$2 runs at power cap ${cap} W (stock ${POWER_CAP_MIN}-${POWER_CAP_MAX} W) — ALLOWED by ALLOW_POWER_MISMATCH=1"
    else
      log "ERROR: test group $1/$2 power cap ${cap} W outside stock range ${POWER_CAP_MIN}-${POWER_CAP_MAX} W — aborting"
      log "       (override: ALLOW_POWER_MISMATCH=1 — never the default)"
      exit 1
    fi
  fi
}

# ---------------------------------------------------------------- thermal watchdog (item 5)
check_thermal() {
  [ -f "$OUT/INVALID_THERMAL" ] || return 0
  log "INVALID_THERMAL: $(cat "$OUT/INVALID_THERMAL")"
  log "terminating: run marked INVALID_THERMAL"
  mon_stop; srv_stop
  exit 3
}

# ---------------------------------------------------------------- monitor (1 s samples + thermal abort)
MON_PID=""
mon_start() { # $1 = csv file
  [ "$DRY_RUN" = 1 ] && return 0
  local csv="$1" smi="$SMI" gpu="$GPU" out="$OUT" tmax="$TEMP_MAX_C" srvf="$OUT/.srv_pid" benchf="$OUT/.bench_pid"
  : > "$csv"
  (
    first=1
    while :; do
      ts=$(date +%s)
      raw=$("$smi" -d "$gpu" --showpower --showclocks --showtemp --showuse --csv 2>/dev/null || true)
      hdr=$(printf '%s\n' "$raw" | sed -n '1p')
      row=$(printf '%s\n' "$raw" | sed -n '2p')
      if [ "$first" = 1 ] && [ -n "$hdr" ]; then echo "ts,$hdr" > "$csv"; first=0; fi
      [ -n "$row" ] && echo "$ts,$row" >> "$csv"
      # junction temperature watchdog
      t=$(printf '%s\n%s\n' "$hdr" "$row" | awk -F, '
            NR==1 { for (i = 1; i <= NF; i++) if (tolower($i) ~ /junction/) c = i; next }
            c { v = $c; gsub(/[^0-9.]/, "", v); if (v != "") print v; exit }')
      if [ -n "$t" ] && awk -v t="$t" -v m="$tmax" 'BEGIN { exit !(t > m) }'; then
        {
          echo "junction temp ${t}C > ${tmax}C at ts=${ts} — run marked INVALID_THERMAL"
          echo "terminating server and benchmark"
        } > "$out/INVALID_THERMAL"
        [ -f "$srvf" ] && kill "$(cat "$srvf")" 2>/dev/null || true
        [ -f "$benchf" ] && kill "$(cat "$benchf")" 2>/dev/null || true
        break
      fi
      sleep 1
    done
  ) 2>/dev/null &
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
  log "server: ${cmd[*]} (env: ${SRV_ENV:-}) (numa: ${NPFX:-off})"
  [ "$DRY_RUN" = 1 ] && return 0
  env ${SRV_ENV:-} "${NUMA_RUN[@]}" "${cmd[@]}" > "$lf" 2>&1 &
  SRV_PID=$!
  echo "$SRV_PID" > "$OUT/.srv_pid"
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$SRV_PID" 2>/dev/null || { log "server died, see $lf"; tail -20 "$lf" >&2; return 1; }
    sleep 1
  done
  log "server did not become healthy"; return 1
}
srv_stop() {
  if [ -n "$SRV_PID" ]; then
    kill "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
    SRV_PID=""
  fi
  rm -f "$OUT/.srv_pid"
  [ "$DRY_RUN" = 1 ] || sleep 5   # never sleep 5 s in DRY_RUN (item 17)
}
trap 'mon_stop; srv_stop' EXIT

# one request -> one CSV row in requests.csv (+ raw json).
# A WARM-UP request (9th arg = 1) NEVER writes to requests.csv and takes NO
# /metrics snapshots — the raw json goes to a warmup_-prefixed file instead.
# A MEASURED request snapshots /metrics BEFORE and AFTER and stores counter
# DELTAS (exact verification_steps / draft_tokens / accepted_tokens /
# per-position survivals + raw per_pos_counts with ZERO deltas preserved —
# spec_metrics.py). Snapshot failure ABORTS (set -e).
# No row is ever deleted afterwards (fixes the old `sed -i '$d'` bug).
REQ_CSV="$OUT/requests.csv"
[ -f "$REQ_CSV" ] || echo "test,config,workload,rep,seed,temp,ctx,n_max,prompt_n,prompt_ms,prefill_tps,predicted_n,predicted_ms,effective_tps,draft_n,draft_accepted,acceptance,verification_steps,draft_tokens,accepted_tokens,mean_draft_tokens_per_step,mean_accepted_tokens_per_step,mean_target_width,survival_per_pos,per_pos_counts,ms_per_spec_step_EST,wall_s,ttft_ms,output_sha256,metrics_label" > "$REQ_CSV"
request() { # test config workload rep n_max [n_predict] [temp] [prompt_file] [warmup=0]
  local t="$1" cfg="$2" wl="$3" rep="$4" nmax="$5" np="${6:-$NPRED}" temp="${7:-$TEMP}" pf="${8:-$PROMPTS/$3.txt}" warm="${9:-0}"
  local seed=$((42 + rep)) rawname="${cfg}_${wl}_r${rep}.json"
  [ "$warm" = 1 ] && rawname="warmup_${cfg}_${wl}.json"
  local raw="$OUT/$t/raw/$rawname"
  mkdir -p "$(dirname "$raw")"
  [ "$DRY_RUN" = 1 ] && { log "request $t $cfg $wl r$rep seed=$seed n_max=$nmax warmup=$warm"; return 0; }
  python3 - "$HERE" "$PORT" "$pf" "$np" "$temp" "$seed" "$raw" "$t" "$cfg" "$wl" "$rep" "$CTX" "$nmax" "$warm" >> "$REQ_CSV" <<'EOF'
import sys, json, time, hashlib, urllib.request
here = sys.argv[1]
(port, pf, npred, temp, seed, raw, t, cfg, wl, rep, ctx, nmax, warm) = sys.argv[2:]
sys.path.insert(0, here)
import spec_metrics as sm   # Prometheus parser + survival math (shared, unit-tested)
prompt = open(pf, encoding="utf-8").read()
body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": int(npred), "temperature": float(temp),
        "top_p": 0.95, "top_k": 20, "seed": int(seed), "stream": True, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
# BEFORE snapshot of the exact spec_decode counters (measured requests only;
# warm-up skips snapshots entirely)
mb = None
if warm != "1":
    mb = sm.fetch_metrics(port)
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
if warm == "1":
    sys.exit(0)   # warm-up: raw saved, no snapshots, requests.csv untouched
# AFTER snapshot + exact deltas (this IS the verification measurement)
ma = sm.fetch_metrics(port)
d = sm.counter_delta(mb, ma)
means = sm.means_from_delta(d) or {}
surv = sm.survival_from_delta(d)
g = lambda k: timings.get(k, 0) or 0
pn, pms, dn, da = g("predicted_n"), g("predicted_ms"), g("draft_n"), g("draft_n_accepted")
eff = pn / pms * 1000 if pms else 0
acc = da / dn if dn else 0
steps = int(d["verification_steps"])
dtok = int(d["draft_tokens"])
atok = int(d["accepted_tokens"])
mdraft = means.get("mean_draft_tokens_per_step", "")
macc = means.get("mean_accepted_tokens_per_step", "")
width = means.get("mean_target_width", "")
ms_step = (pms / steps) if steps and pms else ""
surv_field = ";".join(f"{x:.4f}" for x in surv)          # ';'-joined: CSV-safe
counts_field = ";".join(str(x) for x in sm.per_pos_vector(d["per_pos"]))  # raw ints, zeros kept
if steps > 0:
    label = "exact: /metrics spec_decode counter deltas"
elif dn:
    label = "spec configured; spec_decode counters idle"
else:
    label = "native (spec_decode counters idle)"
print(",".join(map(str, [t, cfg, wl, rep, seed, temp, ctx, nmax, g("prompt_n"), round(g("prompt_ms"), 2),
      round(g("prompt_per_second"), 2), pn, round(pms, 2), round(eff, 3), dn, da, round(acc, 4),
      steps, dtok, atok,
      round(mdraft, 3) if mdraft != "" else "",
      round(macc, 3) if macc != "" else "",
      round(width, 4) if width != "" else "",
      surv_field,
      counts_field,
      round(ms_step, 3) if ms_step != "" else "",
      round(wall, 3), round(ttft or 0, 1), hashlib.sha256(out.encode()).hexdigest()[:16], label])))
EOF
  [ "$warm" = 1 ] && return 0
  check_thermal
  tail -1 "$REQ_CSV" | awk -F, '{printf "   -> %s %s %s: %s tok/s eff, acc %s, ttft %s ms\n",$1,$2,$3,$14,$17,$28}' | tee -a "$LOG" >&2
}

# per-position acceptance + mean len from the server trace log (-lv 4)
acc_per_pos() { # logfile config -> appends to acc_per_pos.csv
  local lf="$1" cfg="$2"
  [ "$DRY_RUN" = 1 ] && return 0
  grep -E "draft acceptance|acc per pos" "$lf" | paste - - 2>/dev/null | \
    sed -E 's/.*draft acceptance = ([0-9.]+).*mean len = *([0-9.]+).*acc per pos = \(([^)]*)\).*/\1;\2;\3/' | \
    awk -v c="$cfg" -F';' '{print c","$1","$2",\""$3"\""}' >> "$OUT/acc_per_pos.csv" || true
}

server_suite() { # test cfg n_max spec_args...
  local t="$1" cfg="$2" nmax="$3"; shift 3
  check_thermal
  mkdir -p "$OUT/$t"
  record_cap "$t" "$cfg"
  srv_start "$OUT/$t/server_${cfg}.log" "$@" || return 1
  # warm-up: writes to a raw warmup_*.json only — NEVER to requests.csv (item 6)
  request "$t" "$cfg" "PROSE" 0 "$nmax" 64 "" "" 1 >/dev/null 2>&1 || true
  mon_start "$OUT/$t/power_${cfg}.csv"
  for wl in $WORKLOADS; do for rep in $(seq 0 $((REPS-1))); do request "$t" "$cfg" "$wl" "$rep" "$nmax"; done; done
  mon_stop
  srv_stop
  acc_per_pos "$OUT/$t/server_${cfg}.log" "$cfg"
}

# ================================================================= PROFILE=1 (item 9)
# SHORT fixed workload under rocprofv3 with HIP graphs disabled. Never wraps
# the full benchmark suite.
if [ "$PROFILE" = 1 ]; then
  mkdir -p "$OUT/profile"
  record_cap PROFILE short-workload
  mon_start "$OUT/profile/power.csv"   # (7) PROFILE is a meaningful GPU load — monitor it
  log "PROFILE=1: short fixed workload under rocprofv3, HIP graphs OFF (kernel trace + names/durations/count + raw files)"
  echo "HIP graphs: DISABLED for the profiled workload (rocprofv3 requires it)" > "$OUT/profile/notes.txt"
  echo "workload: llama-bench -p 128,512 -n 64 -r 1 (short, fixed — NOT the full suite)" >> "$OUT/profile/notes.txt"
  rcmd=("$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -p 128,512 -n 64 -r 1 -o csv)
  pcmd=("$ROCPROF" --kernel-trace --output-format csv --output-directory "$OUT/profile/raw"
        --output-name kernel_trace --)
  if [ ${#NUMA_RUN[@]} -gt 0 ]; then pcmd+=("${NUMA_RUN[@]}"); fi
  pcmd+=("${rcmd[@]}")
  echo "${pcmd[*]}" > "$OUT/profile/cmd.txt"
  if [ "$DRY_RUN" = 1 ]; then
    log "+ GGML_CUDA_DISABLE_GRAPHS=1 ${pcmd[*]}"
  else
    command -v "$ROCPROF" >/dev/null 2>&1 || { log "PROFILE=1 but rocprofv3 not found ($ROCPROF)"; mon_stop; exit 1; }
    mkdir -p "$OUT/profile/raw"
    GGML_CUDA_DISABLE_GRAPHS=1 "${pcmd[@]}" > "$OUT/profile/llama-bench.out" 2> "$OUT/profile/llama-bench.err" \
      || log "profile workload exited non-zero (see profile/llama-bench.err)"
    mon_stop
    check_thermal
    # kernel names / durations / count from the raw trace
    python3 - "$OUT/profile" <<'EOF'
import csv, glob, os, sys
from collections import defaultdict
d = sys.argv[1]
agg = {}
raw_files = glob.glob(os.path.join(d, "raw", "**", "*.csv"), recursive=True)
for f in raw_files:
    try:
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                norm = {(k or "").strip().lower(): v for k, v in row.items()}
                name = (norm.get("kernel_name") or norm.get("kernel name")
                        or norm.get("name"))
                if not name:
                    continue
                start = end = dur = None
                for k, v in norm.items():
                    if k.startswith("start"): start = v
                    elif k.startswith("end"): end = v
                    elif "duration" in k and dur is None:
                        try: dur = float(v)
                        except (TypeError, ValueError): pass
                if dur is None and start is not None and end is not None:
                    try: dur = float(end) - float(start)
                    except (TypeError, ValueError): dur = None
                if dur is None:
                    continue
                a = agg.setdefault(name, {"count": 0, "total_ns": 0.0, "max_ns": 0.0})
                a["count"] += 1
                a["total_ns"] += dur
                a["max_ns"] = max(a["max_ns"], dur)
    except Exception as e:
        print(f"warn: cannot parse {f}: {e}")
out = os.path.join(d, "kernel-summary.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kernel_name", "count", "total_ms", "mean_ms", "max_ms"])
    for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["total_ns"]):
        w.writerow([name, a["count"], round(a["total_ns"] / 1e6, 3),
                    round(a["total_ns"] / a["count"] / 1e6, 4), round(a["max_ns"] / 1e6, 3)])
print(f"kernels profiled: {len(agg)}; summary -> {out}; raw profiler files under {d}/raw/")
EOF
    cp -f "$OUT"/profile/raw/*.csv "$OUT/profile/" 2>/dev/null || true
  fi
fi

# ================================================================= T0 provenance
if want T0; then
  mkdir -p "$OUT/T0"
  record_cap T0 provenance
  [ "$DRY_RUN" = 1 ] || {
    MODEL_SHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
    if [ "$MODEL_SHA" != "$EXPECTED_MODEL_SHA" ] && [ "$ALLOW_MODEL_MISMATCH" != 1 ]; then
      log "ERROR: model sha256 $MODEL_SHA != expected $EXPECTED_MODEL_SHA (override: ALLOW_MODEL_MISMATCH=1)"
      exit 2
    fi
    [ "$MODEL_SHA" = "$EXPECTED_MODEL_SHA" ] || log "WARNING: model sha256 mismatch ALLOWED by ALLOW_MODEL_MISMATCH=1"
    # PCIe facts written by preflight
    PCIE_F="$OUT/preflight/pcie-sysfs.txt"
    GPU_BDF=$(grep -m1 '^bdf=' "$PCIE_F" 2>/dev/null | cut -d= -f2- || true)
    GPU_NUMA=$(grep -m1 '^numa_node=' "$PCIE_F" 2>/dev/null | cut -d= -f2- || true)
    PCIE_SPD=$(grep -m1 '^current_link_speed=' "$PCIE_F" 2>/dev/null | cut -d= -f2- || true)
    PCIE_WDT=$(grep -m1 '^current_link_width=' "$PCIE_F" 2>/dev/null | cut -d= -f2- || true)
    CUR_CAP=$($SMI -d "$GPU" --showmaxpower 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1 || true)
    python3 - "$OUT/meta.json" \
      "$RUN" "$BUILD" "$BIN" "$MODEL" "$MODEL_SHA" \
      "$MODEL_REPO_URL" "$MODEL_REPO" "$MODEL_FILENAME" "$EXPECTED_MODEL_SHA" "$EXPECTED_MODEL_SIZE" \
      "$CTX" "$REPS" "$NPRED" "$TEMP" \
      "$NUMA_MODE" "$NUMA_NODE" "${NPFX:-off}" \
      "${CUR_CAP:-unknown}" "${GPU_BDF:-}" "${GPU_NUMA:-}" "${PCIE_SPD:-}" "${PCIE_WDT:-}" <<'EOF'
import json, subprocess as sp, os, sys
(meta, run, build_dir, bin_dir, model, model_sha,
 repo_url, repo, filename, exp_sha, exp_size,
 ctx, reps, npred, temp,
 numa_mode, numa_node, numa_binding,
 cur_cap, bdf, gpu_numa, pcie_spd, pcie_wdt) = sys.argv[1:]
def sh(c):
    try: return sp.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"ERR {e}"
m = {
 "run": run, "date": sh("date -Is"), "host": sh("hostname"), "kernel": sh("uname -r"),
 "os": sh(". /etc/os-release; echo $PRETTY_NAME"),
 "build_dir": build_dir, "build_info": sh(f"cat '{build_dir}/../build-info.txt' 2>/dev/null"),
 "llama_version": sh(f"'{bin_dir}/llama-server' --version 2>&1 | tail -2"),
 "git_sha": sh(f"'{bin_dir}/llama-server' --version 2>&1 | grep -o 'commit [0-9a-f]*'"),
 "rocm": sh("cat ${ROCM_PATH:-/opt/rocm-7.1.1}/.info/version* 2>/dev/null | head -1; hipconfig --version 2>/dev/null"),
 "amdgpu_driver": sh("modinfo amdgpu 2>/dev/null | grep -E '^(filename|version|vermagic)'"),
 "vbios": sh(f"rocm-smi -d {os.environ.get('GPU','0')} --showvbios 2>/dev/null | grep -i vbios"),
 "power_cap_measured": cur_cap,
 "power_cap_range_expected": "220-226 W (stock 225 W); mixed caps are never summarized together",
 "clocks": sh(f"rocm-smi -d {os.environ.get('GPU','0')} --showclocks 2>/dev/null | grep -E 'sclk|mclk'"),
 "temps": sh(f"rocm-smi -d {os.environ.get('GPU','0')} --showtemp 2>/dev/null | grep -i temp"),
 "perf_level": sh(f"rocm-smi -d {os.environ.get('GPU','0')} --showperflevel 2>/dev/null | grep -i level"),
 "model": model,
 "model_provenance": {"repo_url": repo_url, "repo": repo, "filename": filename,
                      "sha256": exp_sha, "size_bytes": int(exp_size),
                      "actual_sha256": model_sha,
                      "actual_size_bytes": os.path.getsize(model) if os.path.exists(model) else None},
 "model_sha256": model_sha,
 "numa": {"mode": numa_mode, "node": int(numa_node) if numa_node not in ("", None) else None,
          "binding": numa_binding},
 "pcie": {"bdf": bdf or None, "gpu_numa_node": gpu_numa or None,
          "current_link_speed": pcie_spd or None, "current_link_width": pcie_wdt or None},
 "ctx": int(ctx), "reps": int(reps), "n_predict": int(npred), "temp": float(temp),
 "hsa_override_gfx_version": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
 "env": {k: v for k, v in os.environ.items() if k.startswith(("GGML_", "HIP_", "HSA_", "ROCM", "LLAMA_", "NUMA_"))},
 "cpu": sh("lscpu | grep 'Model name'"), "mem": sh("free -g | head -2"),
 "preflight_report": os.path.join(os.path.dirname(os.path.abspath(meta)), "preflight"),
}
json.dump(m, open(meta, "w"), indent=1)
EOF
  }
  # thermal monitoring covers EVERY meaningful GPU-load benchmark (7), incl. T0/T1
  mon_start "$OUT/T0/power.csv"
  # Exit-code correctness (review 3): capture test-backend-ops' OWN status —
  # never tail's (the old `cmd > file; tail file` chain returned tail's rc).
  # For EVERY T0 op including MUL_MAT_VEC_FUSION:
  #   nonzero exit -> STOP   AND   text FAIL -> STOP.
  # A segfault (139) with no literal "FAIL" must also stop the run.
  for op in MUL_MAT MUL_MAT_ID FLASH_ATTN_EXT GATED_DELTA_NET MUL_MAT_VEC_FUSION; do
    op_out="$OUT/T0/test_$op.txt"
    if [ "$DRY_RUN" = 1 ]; then
      log "test-backend-ops test -o $op -b ROCm0  (dry)"
      continue
    fi
    set +e
    bash -c "$NPFX '$BIN/test-backend-ops' test -o $op -b ROCm0 > '$op_out' 2>&1"
    op_rc=$?
    set -e
    tail -3 "$op_out" 2>/dev/null | tee -a "$LOG" >&2 || true
    if [ "$op_rc" -ne 0 ]; then
      log "CORRECTNESS FAIL — test-backend-ops $op exited rc=$op_rc (no 'FAIL' text required) — stopping"
      exit 2
    fi
  done
  mon_stop
  check_thermal
  [ "$DRY_RUN" = 1 ] || ! grep -l "FAIL" "$OUT"/T0/test_*.txt >/dev/null 2>&1 || { log "CORRECTNESS FAIL — text FAIL in op output — stopping"; exit 2; }
fi

# ================================================================= T1 kernel micro-bench (cutover table)
if want T1; then
  mkdir -p "$OUT/T1"
  record_cap T1 mul-mat-micro
  mon_start "$OUT/T1/power.csv"
  # model shapes (k x m): qkv 5120x10240, gate 5120x6144, ffn_up/gate 5120x17408, ffn_down 17408x5120,
  # ssm_out 6144x5120 (Q5_K), attn_q 5120x12288, output 5120x248320 (Q6_K). Types present: q4_0 q4_1 q5_K q6_K q8_0.
  # test-backend-ops perf runs its built-in MUL_MAT grid; we filter to our types and n<=16.
  run bash -c "$NPFX '$BIN/test-backend-ops' perf -o MUL_MAT -b ROCm0 > '$OUT/T1/mul_mat_perf.txt' 2>&1 || true"
  [ "$DRY_RUN" = 1 ] || grep -E "type_a=(q4_0|q4_1|q4_K|q5_K|q6_K|q8_0|iq4_xs)" "$OUT/T1/mul_mat_perf.txt" | \
      grep -E "n=([1-9]|1[0-6])," > "$OUT/T1/mul_mat_n1-16.txt" || true
  # dispatch A/B inside the same binary: generic vs wide GEMV
  for sw in GGML_MMVQ_Q4_BREIT_GFX906=0 GGML_MMVQ_Q5K_BREIT_GFX906=0 GGML_MMVQ_Q6K_BREIT_GFX906=0 GGML_MMVQ_Q41_BREIT_GFX906=0; do
    run bash -c "env $sw $NPFX '$BIN/test-backend-ops' perf -o MUL_MAT -b ROCm0 > '$OUT/T1/mul_mat_perf_${sw%%=*}_off.txt' 2>&1 || true"
  done
  mon_stop
  check_thermal
fi

# ================================================================= T2 native llama-bench
if want T2; then
  mkdir -p "$OUT/T2"; record_cap T2 llama-bench; mon_start "$OUT/T2/power.csv"
  run bash -c "$NPFX '$BIN/llama-bench' -m '$MODEL' -ngl 99 -fa on -p 512,2048 -n 128 -r 5 -o csv > '$OUT/T2/llama-bench.csv' 2> '$OUT/T2/llama-bench.err'"
  mon_stop
fi

# ================================================================= T3-SYNTHETIC-NCOL (test id: T3)
# Purpose: characterize the n-column GGML/GEMV/MMQ cost for n = 1..16.
# llama-bench -p N -n 0 measures prompt processing at WIDTH N — it is NOT an
# MTP verification measurement and must not be reported as MTP verify latency.
# Real MTP verification measurements come from T5/T6/T7 (see requests.csv +
# summary.md "MTP verification — REAL"). Kept because the n-column cost curve
# (MMVQ vs MMQ cutover, MMQ valley) is genuinely useful.
if want T3; then
  mkdir -p "$OUT/T3"; record_cap T3 synthetic-ncol; mon_start "$OUT/T3/power.csv"
  run bash -c "$NPFX '$BIN/llama-bench' -m '$MODEL' -ngl 99 -fa on -p 1,2,3,4,5,6,7,8,9,12,16 -n 0 -ub 512 -r 10 -o csv > '$OUT/T3/ncol-cost-1-16.csv' 2> '$OUT/T3/ncol-cost.err'"
  # same at depth: n-column cost with 16k tokens already in KV
  run bash -c "$NPFX '$BIN/llama-bench' -m '$MODEL' -ngl 99 -fa on -p 1,4,8 -n 0 -d 16384 -r 5 -o csv > '$OUT/T3/ncol-cost-d16k.csv' 2>> '$OUT/T3/ncol-cost.err'"
  mon_stop
  [ "$DRY_RUN" = 1 ] || cat > "$OUT/T3/README.txt" <<'TXT'
T3-SYNTHETIC-NCOL — synthetic n-column cost characterization (n = 1..16).
llama-bench -p N -n 0 measures prompt processing at width N.
This is NOT actual MTP verification latency and must never be reported as such.
Real MTP verification measurements: T5/T6/T7 (requests.csv + summary.md,
"MTP verification — REAL"; derived per-step values are labeled ESTIMATE).
TXT
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

# ================================================================= T8 long context (tokenizer-measured, item 10)
# Prompt lengths are built and MEASURED with the Qwen/llama tokenizer via the
# server /tokenize endpoint — never estimated from word counts. The ACTUAL
# token count of every prompt file is recorded in T8/prompt_tokens.csv, and
# requests.csv prompt_n carries the server's actual prompt token count.
gen_long_prompts() { # requires a running server at $PORT; writes prompt_*.txt + prompt_tokens.csv
  python3 - "$PORT" "$OUT/T8" "$PROMPTS/CODING-1.txt" <<'EOF'
import json, sys, urllib.request, os
port, outdir, src = sys.argv[1], sys.argv[2], sys.argv[3]
TARGETS = [8192, 32768, 65536]          # ~8k, ~32k, ~64k tokens

def tokenize(text):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/tokenize",
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    if "tokens" in data:
        return len(data["tokens"])
    if "count" in data:
        return int(data["count"])
    raise RuntimeError(f"unexpected /tokenize response keys: {list(data)}")

def func(i):
    return (f"def helper_{i}(x: int, y: int) -> int:\n"
            f"    '''helper {i}'''\n"
            f"    return (x * {i} + y) % {997 + i}\n")

FUNCS = [func(i) for i in range(6000)]
HEADER = "Context: the following module is part of the repository.\n```python\n"
FOOTER = "\n```\n\n"
src_text = open(src, encoding="utf-8").read()

def compose(prefix):
    return HEADER + prefix + FOOTER + src_text

def build(target):
    # 1) coarse: binary search how many whole functions fit <= target tokens
    lo, hi = 0, 1
    while hi < len(FUNCS) and tokenize(compose("\n".join(FUNCS[:hi]) + "\n")) < target:
        lo = hi
        hi *= 2
    hi = min(hi, len(FUNCS))
    cnt, text = 0, compose("")
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = compose("\n".join(FUNCS[:mid]) + "\n")
        c = tokenize(cand)
        if c <= target:
            cnt, text = mid, cand
            lo = mid + 1
        else:
            hi = mid - 1
    # whole-function fine pass
    while cnt < len(FUNCS):
        cand = compose("\n".join(FUNCS[:cnt + 1]) + "\n")
        c = tokenize(cand)
        if c > target:
            break
        cnt, text = cnt + 1, cand
    # word-level fill up to (never past) the target
    filler_words = ("The repository follows standard style guidelines and keeps "
                    "each helper function small and independently testable. ").split()
    prefix_core = text[len(HEADER):-len(FOOTER + src_text)]
    while filler_words:
        w = filler_words.pop(0)
        cand_prefix = prefix_core + w + " "
        cand = compose(cand_prefix)
        c = tokenize(cand)
        if c <= target:
            prefix_core, text = cand_prefix, cand
        else:
            break
    actual = tokenize(text)
    return text, actual

rows = []
for target in TARGETS:
    text, actual = build(target)
    dst = os.path.join(outdir, f"prompt_{target}.txt")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    rows.append((target, actual, dst))
    print(f"prompt target={target} actual_tokens={actual} -> {dst}", flush=True)

with open(os.path.join(outdir, "prompt_tokens.csv"), "w") as f:
    f.write("target_tokens,actual_tokens,file,counted_via\n")
    for target, actual, dst in rows:
        f.write(f"{target},{actual},{dst},server /tokenize endpoint (Qwen/llama tokenizer)\n")
if any(abs(a - t) / t > 0.02 for t, a, _ in rows):
    print("WARNING: some prompt token counts differ from target by >2%", file=sys.stderr)
EOF
}

if want T8; then
  mkdir -p "$OUT/T8"
  for kv in f16 q8_0; do
    CTX_SAVE=$CTX; CTX=73728
    check_thermal
    record_cap T8 "kv$kv"
    srv_start "$OUT/T8/server_kv${kv}.log" --cache-type-k $kv --cache-type-v $kv --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 || { CTX=$CTX_SAVE; continue; }
    if [ "$DRY_RUN" = 1 ]; then
      log "T8: would measure prompt token counts via /tokenize for 8192/32768/65536 and request kv$kv"
    elif [ ! -f "$OUT/T8/prompt_tokens.csv" ]; then
      if ! gen_long_prompts; then
        log "T8 kv$kv: tokenizer prompt generation FAILED (is /tokenize available?) — skipping this arm"
        mon_stop; srv_stop; CTX=$CTX_SAVE; continue
      fi
    fi
    mon_start "$OUT/T8/power_kv${kv}.csv"
    for n in 8192 32768 65536; do
      request T8 "kv${kv}_${n}" "CODING-1" 0 3 256 "$TEMP" "$OUT/T8/prompt_${n}.txt"
    done
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
  record_cap T9 native-bench
  # native too: graphs matter most at n=1 — and it is a meaningful GPU load,
  # so the thermal monitor must cover it (7): not only suites/T2/T3
  mon_start "$OUT/T9/power_native.csv"
  run bash -c "$NPFX '$BIN/llama-bench' -m '$MODEL' -ngl 99 -fa on -p 0 -n 128 -r 5 -o csv > '$OUT/T9/tg_graphs_on.csv' 2>/dev/null"
  run bash -c "GGML_CUDA_DISABLE_GRAPHS=1 $NPFX '$BIN/llama-bench' -m '$MODEL' -ngl 99 -fa on -p 0 -n 128 -r 5 -o csv > '$OUT/T9/tg_graphs_off.csv' 2>/dev/null"
  mon_stop
  check_thermal
fi

# ================================================================= T10 soak + determinism
if want T10; then
  mkdir -p "$OUT/T10"
  record_cap T10 greedy-mtp3
  srv_start "$OUT/T10/server.log" --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 || exit 1
  mon_start "$OUT/T10/power.csv"
  # greedy determinism: same prompt twice must give the same hash (MTP must not change greedy output)
  for rep in 0 1; do request T10 greedy_mtp3 CODING-1 "$rep" 3 512 0; done
  end=$(( $(date +%s) + ${SOAK_S:-1800} )); i=0
  while [ "$DRY_RUN" != 1 ] && [ "$(date +%s)" -lt "$end" ]; do
    check_thermal
    for wl in $WORKLOADS; do request T10 soak "$wl" "$i" 3; done; i=$((i+1))
  done
  mon_stop; srv_stop
  check_thermal
  record_cap T10 greedy-native
  srv_start "$OUT/T10/server_native.log" || exit 1
  mon_start "$OUT/T10/power_native.csv"   # native arm is a meaningful load too (7)
  for rep in 0 1; do request T10 greedy_native CODING-1 "$rep" 0 512 0; done
  mon_stop
  srv_stop
  check_thermal
  [ "$DRY_RUN" = 1 ] || grep -cE "error|ERROR|abort|hipError" "$OUT/T10/server.log" > "$OUT/T10/error_count.txt" || true
fi

# ================================================================= summary
[ "$DRY_RUN" = 1 ] && { log "dry run done: $OUT"; exit 0; }
if [ -f "$OUT/INVALID_THERMAL" ]; then
  log "run marked INVALID_THERMAL — $(cat "$OUT/INVALID_THERMAL")"
  exit 3
fi
python3 "$HERE/summarize.py" "$OUT" > "$OUT/summary.md" && log "summary: $OUT/summary.md"
