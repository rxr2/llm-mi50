#!/usr/bin/env bash
# Safe power/clock procedure for 1x MI50 — stock only, NO overclock.
#   ./power.sh status          read-only: cap, max cap, clocks, temps, perf level
#   sudo ./power.sh stock      reset to factory state (225 W cap, auto perf level)
#   ./power.sh monitor FILE    1 s samples during a benchmark (run in background)
# Rules:
#  * Every benchmark records `--showmaxpower` BEFORE it runs; results at different caps are never compared.
#  * Cap is set to the card default (225 W) only; never above. No sclk/mclk/voltage changes.
#  * Abort a run if junction temp > 95 C or power throttling flags appear persistently.
set -euo pipefail
GPU="${GPU:-0}"
SMI="${SMI:-rocm-smi}"
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
    echo "ts,power_w,sclk_mhz,mclk_mhz,temp_junction_c,use_pct" > "$OUTF"
    while :; do
      j=$($SMI -d "$GPU" --showpower --showclocks --showtemp --showuse --json 2>/dev/null) || true
      python3 - "$j" >> "$OUTF" <<'EOF' || true
import sys,json,re,time
d=json.loads(sys.argv[1]); c=list(d.values())[0]
def f(keys):
    for k,v in c.items():
        if all(x in k.lower() for x in keys):
            m=re.search(r'[\d.]+',str(v)); return m.group(0) if m else ''
    return ''
print(",".join([str(int(time.time())), f(['power']), f(['sclk']), f(['mclk']), f(['junction']), f(['gpu use'])]))
EOF
      sleep 1
    done
    ;;
  *) echo "usage: $0 status|stock|monitor FILE"; exit 1 ;;
esac
