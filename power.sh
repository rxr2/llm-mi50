#!/usr/bin/env bash
# Safe power/clock procedure for 1x MI50 — stock only, NO overclock.
#   ./power.sh status          read-only: cap, max cap, clocks, temps, perf level
#   sudo ./power.sh stock      reset to factory state (225 W cap, auto perf level)
#   ./power.sh monitor FILE    1 s samples during a benchmark (run in background)
# Rules:
#  * Every test group records the ACTUAL power cap before it runs
#    (benchmark-mi50.sh appends to power_caps.csv); results at different caps
#    are never summarized together (summarize.py splits mixed-cap runs).
#  * Cap is set to the card default (225 W) only; never above. No sclk/mclk/voltage changes.
#  * Thermal abort IS implemented: if junction temp > 95 C (TEMP_MAX_C), this
#    monitor writes FILE.INVALID_THERMAL and stops sampling; benchmark-mi50.sh
#    runs its own watchdog that additionally terminates the server and the
#    benchmark and marks the whole run INVALID_THERMAL.
set -euo pipefail
GPU="${GPU:-0}"
SMI="${SMI:-rocm-smi}"
TEMP_MAX_C="${TEMP_MAX_C:-95}"
case "${1:-status}" in
  status)
    $SMI -d "$GPU" --showmaxpower --showpower --showclocks --showtemp --showperflevel --showuse 2>/dev/null
    ;;
  stock)
    [ "$(id -u)" = 0 ] || { echo "needs root"; exit 1; }
    $SMI -d "$GPU" --resetpoweroverdrive     # back to the vbios default (225 W on MI50)
    $SMI -d "$GPU" --setperflevel auto
    $SMI -d "$GPU" --resetclocks
    cap=$($SMI -d "$GPU" --showmaxpower | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1)
    echo "cap now: ${cap} W"
    awk -v c="$cap" 'BEGIN{exit !(c>=220 && c<=226)}' || echo "WARNING: cap ${cap} W is not the stock 225 W"
    ;;
  monitor)
    OUTF="${2:?file}"
    echo "ts,power_w,sclk_mhz,mclk_m_hz,temp_junction_c,use_pct" > "$OUTF"
    while :; do
      j=$($SMI -d "$GPU" --showpower --showclocks --showtemp --showuse --json 2>/dev/null) || true
      rc=0
      python3 - "$j" "$OUTF" "$TEMP_MAX_C" >> "$OUTF" <<'EOF' || rc=$?
import sys,json,re,time
if len(sys.argv) < 4 or not sys.argv[1]:
    sys.exit(0)
d = json.loads(sys.argv[1])
if isinstance(d, list):                      # tolerate array-wrapped output
    d = d[0] if d else {}
if not isinstance(d, dict):
    sys.exit(0)
c = list(d.values())[0] if d else {}
if not isinstance(c, dict):                  # flat {key: value} shape
    c = d
def f(keys):
    stack = [c]
    while stack:                             # shallow two-level walk
        cur = stack.pop()
        for k, v in cur.items():
            if isinstance(v, dict):
                stack.append(v); continue
            if all(x in str(k).lower() for x in keys):
                m = re.search(r'[\d.]+', str(v)); return m.group(0) if m else ''
    return ''
t = f(['power']); s = f(['sclk']); m = f(['mclk']); temp = f(['junction']); u = f(['gpu use'])
print(",".join([str(int(time.time())), t, s, m, temp, u]))
if temp:
    try: tv = float(temp)
    except ValueError: tv = None
    if tv is not None and tv > float(sys.argv[3]):
        with open(sys.argv[2] + ".INVALID_THERMAL", "w") as fh:
            fh.write(f"junction temp {tv}C > {sys.argv[3]}C at ts={int(time.time())} — run marked INVALID_THERMAL\n")
        print(f"THERMAL ABORT: junction {tv}C > {sys.argv[3]}C — marker written, stopping monitor",
              file=sys.stderr)
        sys.exit(95)
EOF
      [ "$rc" = 95 ] && { echo "monitor stopped: INVALID_THERMAL (see $OUTF.INVALID_THERMAL)"; exit 95; }
      sleep 1
    done
    ;;
  *) echo "usage: $0 status|stock|monitor FILE"; exit 1 ;;
esac
