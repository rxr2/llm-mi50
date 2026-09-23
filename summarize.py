#!/usr/bin/env python3
"""summarize.py results/DATE/RUN  -> markdown on stdout.
Main metric: effective accepted tok/s = predicted_n / predicted_ms (server timings), per workload, median over reps.
Targets: native ~30, MTP 45-55, >55 stretch. Acceptance rule: <0.50 -> n-max 2, >=0.50 -> 3..4.
"""
import csv, sys, statistics as st, os, json
from collections import defaultdict

run = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(run, "requests.csv"))))
meta = {}
if os.path.exists(os.path.join(run, "meta.json")):
    meta = json.load(open(os.path.join(run, "meta.json")))

print(f"# Summary {os.path.basename(run)}\n")
for k in ("git_sha", "rocm", "power_cap", "model_sha256", "kernel"):
    if k in meta: print(f"- **{k}**: `{str(meta[k]).splitlines()[0] if meta[k] else ''}`")
print()

g = defaultdict(list)
for r in rows:
    g[(r["test"], r["config"], r["workload"])].append(r)

def med(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return st.median(xs) if xs else 0.0

configs = sorted({(t, c) for (t, c, _) in g})
workloads = sorted({w for (_, _, w) in g})
print("## Effective tok/s (median over reps) — acceptance in brackets\n")
print("| test | config | " + " | ".join(workloads) + " | coding-median |")
print("|---|---|" + "---|" * (len(workloads) + 1))
for t, c in configs:
    cells, coding = [], []
    for w in workloads:
        rs = g.get((t, c, w))
        if not rs: cells.append("—"); continue
        e, a = med(r["effective_tps"] for r in rs), med(r["acceptance"] for r in rs)
        cells.append(f"{e:.1f} ({a:.2f})" if a else f"{e:.1f}")
        if w.startswith(("CODING", "EDIT", "AGENT", "JSON")): coding.append(e)
    print(f"| {t} | {c} | " + " | ".join(cells) + f" | **{st.median(coding):.1f}** |" if coding else f"| {t} | {c} | " + " | ".join(cells) + " | — |")

print("\n## Verify cost / latency (median)\n")
print("| test | config | ms/verify-step (est.) | TTFT ms | prefill tok/s | spread eff. tok/s (min-max) |")
print("|---|---|---|---|---|---|")
for t, c in configs:
    rs = [r for (tt, cc, _), v in g.items() if (tt, cc) == (t, c) for r in v]
    effs = [float(r["effective_tps"]) for r in rs]
    print(f"| {t} | {c} | {med(r['ms_per_verify_est'] for r in rs):.2f} | {med(r['ttft_ms'] for r in rs):.0f} | "
          f"{med(r['prefill_tps'] for r in rs):.0f} | {min(effs):.1f}-{max(effs):.1f} |")

print("\n## MTP depth decision per workload\n")
for w in workloads:
    acc3 = [float(r["acceptance"]) for r in g.get(("T6", "mtp3", w), [])]
    if not acc3: continue
    a = st.median(acc3)
    rec = "n-max 2" if a < 0.50 else "n-max 3 (try 4 if T7 mtp4 wins by > noise)"
    print(f"- {w}: acceptance@3 = {a:.2f} -> {rec}")

f = os.path.join(run, "acc_per_pos.csv")
if os.path.exists(f):
    print("\n## Per-position acceptance (server trace)\n```")
    print(open(f).read())
    print("```")

det = [r for r in rows if r["test"] == "T10" and r["config"].startswith("greedy")]
if det:
    print("\n## Determinism (greedy)\n")
    by = defaultdict(set)
    for r in det: by[r["config"]].add(r["output_sha256"])
    for k, v in by.items(): print(f"- {k}: {'OK' if len(v) == 1 else 'DIFFERENT OUTPUTS'} ({', '.join(v)})")
    if len(by) == 2 and len(set.union(*by.values())) == 1: print("- MTP greedy == native greedy: OK")
