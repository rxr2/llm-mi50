#!/usr/bin/env python3
"""summarize.py results/DATE/RUN  -> markdown on stdout.

Main metric: effective accepted tok/s = predicted_n / predicted_ms (server
timings), per workload, median over reps. Targets: native ~30, MTP 45-55,
>55 stretch. Acceptance rule: <0.50 -> n-max 2, >=0.50 -> 3..4.

Semantics fixes (final harness pass):
  * T3 is T3-SYNTHETIC-NCOL: synthetic n-column GGML/GEMV/MMQ cost for
    n=1..16 (llama-bench -p N). It is NOT MTP verification latency.
  * The REAL MTP verification measurements come from T5/T6/T7 as exact
    BEFORE/AFTER `/metrics` Prometheus counter deltas (spec_metrics.py):
    verification_steps, draft_tokens, accepted_tokens, mean_*_per_step,
    mean_target_width = 1 + draft_tokens/verification_steps, and
    per-position survivals r[i] = P(A>=i+1) (server acc-per-pos values —
    UNCONDITIONAL survivals, NEVER chain-multiplied again).
    Accepted-prefix distribution from survivals:
    P(A=0)=1-r0, P(A=k)=r[k-1]-r[k], P(A=K)=r[K-1].
    Means are exact per config (sums-based) — no invented width histogram,
    no last-request-wins overwrites of acc_per_pos.csv rows.
  * Derived per-step values are labeled EST —
    predicted_ms/verification_steps (exact step count) is never called
    "measured verify latency".
  * Runs (or test groups) with DIFFERENT power caps are NEVER aggregated
    together — they are rendered in separate sections.
  * INVALID_THERMAL marks the run as aborted/invalid at the top.
"""
import csv, sys, statistics as st, os, json
from collections import defaultdict

import spec_metrics as sm   # Prometheus parsers + survival math (shared, unit-tested)

run = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(run, "requests.csv"))))
meta = {}
if os.path.exists(os.path.join(run, "meta.json")):
    meta = json.load(open(os.path.join(run, "meta.json")))

# ---------------- power caps: never mix different caps together (item 5) ----
caps_by_group = {}
pcap_file = os.path.join(run, "power_caps.csv")
if os.path.exists(pcap_file):
    for r in csv.DictReader(open(pcap_file)):
        caps_by_group[(r["test"], r["config"])] = r["power_cap_w"]

def cap_of(r):
    return caps_by_group.get((r["test"], r["config"])) or "unknown"

distinct_caps = sorted({cap_of(r) for r in rows})

thermal = os.path.exists(os.path.join(run, "INVALID_THERMAL"))
thermal_msg = ""
if thermal:
    thermal_msg = open(os.path.join(run, "INVALID_THERMAL")).read().strip()

print(f"# Summary {os.path.basename(run)}\n")
if thermal:
    print(f"> **INVALID_THERMAL** — run ABORTED by the thermal watchdog: {thermal_msg}")
    print("> The numbers below are PARTIAL and must not be treated as a valid run.\n")
if len(distinct_caps) > 1:
    print(f"> **DIFFERENT POWER CAPS DETECTED: {', '.join(distinct_caps)}**")
    print("> Sections are separated per cap. Runs at different caps are NEVER compared or aggregated.\n")
for k in ("git_sha", "rocm", "model_sha256", "kernel"):
    if k in meta: print(f"- **{k}**: `{str(meta[k]).splitlines()[0] if meta[k] else ''}`")
mp = meta.get("model_provenance") or {}
if mp:
    print(f"- **model provenance**: {mp.get('repo_url','')} / `{mp.get('filename','')}` "
          f"sha256 `{mp.get('sha256','')}` size `{mp.get('size_bytes','')}`")
if distinct_caps:
    print(f"- **power cap(s)**: {', '.join(distinct_caps)}")
numa = meta.get("numa") or {}
if numa:
    print(f"- **NUMA**: mode={numa.get('mode')} node={numa.get('node')} binding=`{numa.get('binding')}`")
print()

g = defaultdict(list)
for r in rows:
    g[(r["test"], r["config"], r["workload"])].append(r)

def med(xs):
    xs = [float(x) for x in xs if x not in ("", None)]
    return st.median(xs) if xs else 0.0

def parse_acc_vec(s):
    try:
        return [float(x) for x in s.strip().strip('"').split(",") if x.strip()]
    except Exception:
        return []

def nmax_of(r):
    try:
        return int(r.get("n_max") or 0)
    except ValueError:
        return 0

# Accepted-prefix distribution + exact means come from spec_metrics.py
# (Prometheus counter deltas + survival math, unit-tested statically).
# Chain-rule width_estimate was REMOVED: llama-server's acc-per-pos values
# (n_accepted_per_pos[i] / n_draft_verif_steps) are already unconditional
# survival probabilities P(A >= i+1) — never chain-multiply them again.

def render(rs, cap_label):
    if not rs:
        return
    print(f"### power cap: {cap_label}\n")
    gg = defaultdict(list)
    for r in rs:
        gg[(r["test"], r["config"], r["workload"])].append(r)
    configs = sorted({(t, c) for (t, c, _) in gg})
    workloads = sorted({w for (_, _, w) in gg})

    print("## Effective tok/s (median over reps) — acceptance in brackets\n")
    print("| test | config | " + " | ".join(workloads) + " | coding-median |")
    print("|---|---|" + "---|" * (len(workloads) + 1))
    for t, c in configs:
        cells, coding = [], []
        for w in workloads:
            rs_w = gg.get((t, c, w))
            if not rs_w: cells.append("—"); continue
            e, a = med(r["effective_tps"] for r in rs_w), med(r["acceptance"] for r in rs_w)
            cells.append(f"{e:.1f} ({a:.2f})" if a else f"{e:.1f}")
            if w.startswith(("CODING", "EDIT", "AGENT", "JSON")): coding.append(e)
        print(f"| {t} | {c} | " + " | ".join(cells) + (f" | **{st.median(coding):.1f}** |" if coding else " | — |"))

    print("\n## Per-step cost — ms/spec-step from EXACT /metrics verification_steps "
          "(wall overhead folded in; NOT a measured verify latency)\n")
    print("| test | config | ms/spec-step EST | TTFT ms | prefill tok/s | spread eff. tok/s (min-max) | metrics |")
    print("|---|---|---|---|---|---|---|")
    for t, c in configs:
        rs_c = [r for (tt, cc, _), v in gg.items() if (tt, cc) == (t, c) for r in v]
        effs = [float(r["effective_tps"]) for r in rs_c]
        labels = sorted({(r.get("metrics_label") or "").split(":")[0].strip() for r in rs_c})
        print(f"| {t} | {c} | {med(r.get('ms_per_spec_step_EST') for r in rs_c):.2f} | "
              f"{med(r['ttft_ms'] for r in rs_c):.0f} | {med(r['prefill_tps'] for r in rs_c):.0f} | "
              f"{min(effs):.1f}-{max(effs):.1f} | {'/'.join(labels) or '—'} |")

    # ---------------- REAL MTP verification (T5/T6/T7) — item 8 -------------
    mtp_rows = [r for r in rs if r["test"] in ("T5", "T6", "T7")
                and nmax_of(r) > 0 and float(r.get("draft_n") or 0) > 0]
    if mtp_rows:
        print("\n## MTP verification — REAL (T5/T6/T7)\n")
        print("draft_n / draft_n_accepted come from the server API timings; "
              "verification_steps / draft_tokens / accepted_tokens / per-position survivals "
              "from BEFORE/AFTER `/metrics` Prometheus counter deltas "
              "(exact — spec_metrics.py); predicted_ms and wall time are measured totals.\n")
        # Exact counters: means recomputed from SUMS per config (equal weight) —
        # never chain-rule-multiplied, never last-request-wins.
        print("| test | config | verif_steps Σ | draft_tokens Σ | accepted_tokens Σ | "
              "mean_draft/step | mean_accepted/step | mean_target_width | acceptance Σ | "
              "predicted_ms sum | wall_s sum | ms/spec-step | metrics |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        by_cfg = defaultdict(list)
        for r in mtp_rows:
            by_cfg[(r["test"], r["config"])].append(r)
        # ALL acc_per_pos rows per config (append — never last-wins overwrite)
        acc_cfg = defaultdict(list)
        ap = os.path.join(run, "acc_per_pos.csv")
        if os.path.exists(ap):
            for line in csv.reader(open(ap)):
                if len(line) >= 4:
                    acc_cfg[line[0]].append(parse_acc_vec(line[3]))
        def _fsum(rows, key):
            tot = 0.0
            for r in rows:
                try:
                    tot += float(r.get(key) or 0)
                except (TypeError, ValueError):
                    pass
            return tot
        for (t, c), rs_c in sorted(by_cfg.items()):
            steps = _fsum(rs_c, "verification_steps")
            draft = _fsum(rs_c, "draft_tokens")
            acc_t = _fsum(rs_c, "accepted_tokens")
            md = draft / steps if steps else 0.0
            ma = acc_t / steps if steps else 0.0
            width = 1.0 + draft / steps if steps else 0.0
            pms = sum(float(r["predicted_ms"]) for r in rs_c)
            acc = (acc_t / draft) if draft else med(r["acceptance"] for r in rs_c)
            msp_step = (pms / steps) if steps else 0.0
            label = next((str(r.get("metrics_label") or "") for r in rs_c
                          if r.get("metrics_label")), "—")
            print(f"| {t} | {c} | {steps:.0f} | {draft:.0f} | {acc_t:.0f} | "
                  f"{md:.2f} | {ma:.2f} | {width:.3f} | {acc:.4f} | "
                  f"{pms:.0f} | {sum(float(r['wall_s']) for r in rs_c):.1f} | "
                  f"{msp_step:.2f} | {label[:40]} |")
        # Accepted-prefix distribution from UNCONDITIONAL survivals —
        # P(A=0)=1-r0, P(A=k)=r[k-1]-r[k], P(A=K)=r[K-1].  NEVER chain-multiply.
        # Aggregation is EXACT and verification_steps-weighted (review 2):
        #   r[i] = sum(counts[i]) / sum(steps)  — a 100-step request weighs
        # 10x a 10-step request; NEVER an equal-weight mean of per-request
        # survival vectors when exact steps are known.
        print("\nAccepted-prefix distribution P(A=k) from per-position survivals "
              "r[i]=P(A>=i+1) (EXACT, verification_steps-weighted per config):\n")
        for (t, c) in sorted(by_cfg):
            rows_c = by_cfg[(t, c)]
            counts_list, steps_list = [], []
            for r in rows_c:
                try:
                    s = int(float(r.get("verification_steps") or 0))
                except (TypeError, ValueError):
                    s = 0
                cnt = sm.parse_counts_field(r.get("per_pos_counts") or "")
                if cnt:
                    counts_list.append(cnt)
                    steps_list.append(s)
            nreq = 0
            if counts_list:
                rvec = sm.weighted_survival(counts_list, steps_list)
                source = "/metrics per_pos_counts (steps-weighted, exact)"
                nreq = len(counts_list)
            else:
                surv_rows, surv_steps = [], []
                for r in rows_c:
                    v = sm.parse_survival_field(r.get("survival_per_pos") or "")
                    if not v:
                        continue
                    try:
                        s = int(float(r.get("verification_steps") or 0))
                    except (TypeError, ValueError):
                        s = 0
                    surv_rows.append(v)
                    surv_steps.append(s)
                if surv_rows:
                    rvec = sm.weighted_survival_vectors(surv_rows, surv_steps)
                    source = "survival_per_pos fallback (steps-weighted)"
                    nreq = len(surv_rows)
                else:
                    trace_rows = [v for v in acc_cfg.get(c, []) if v]
                    rvec = sm.average_survival(trace_rows)
                    source = "server-trace acc-per-pos fallback (equal-weight, no steps)"
                    nreq = len(trace_rows)
            if not rvec:
                print(f"- {t}/{c}: no survival/acceptance data")
                continue
            dist = sm.accepted_prefix_distribution(rvec)
            hist = ", ".join(f"P(A={k})={p:.3f}" for k, p in enumerate(dist))
            print(f"- {t}/{c} (K={len(rvec)}, n={nreq} reqs, {source}): {hist}")
        prof_sum = os.path.join(run, "profile", "kernel-summary.csv")
        if os.path.exists(prof_sum):
            print(f"\nKernel trace (PROFILE=1): `{os.path.relpath(prof_sum, run)}` "
                  f"(kernel names / durations / count; raw files under `profile/`).")

    # ---------------- MTP depth decision ------------------------------------
    print("\n## MTP depth decision per workload\n")
    for w in workloads:
        acc3 = [float(r["acceptance"]) for r in gg.get(("T6", "mtp3", w), [])]
        if not acc3: continue
        a = st.median(acc3)
        rec = "n-max 2" if a < 0.50 else "n-max 3 (try 4 if T7 mtp4 wins by > noise)"
        print(f"- {w}: acceptance@3 = {a:.2f} -> {rec}")

    det = [r for r in rs if r["test"] == "T10" and r["config"].startswith("greedy")]
    if det:
        print("\n## Determinism (greedy)\n")
        by = defaultdict(set)
        for r in det: by[r["config"]].add(r["output_sha256"])
        for k, v in by.items(): print(f"- {k}: {'OK' if len(v) == 1 else 'DIFFERENT OUTPUTS'} ({', '.join(v)})")
        if len(by) == 2 and len(set.union(*by.values())) == 1:
            print("- MTP greedy == native greedy: OK")
    print()

# ---------------- T3-SYNTHETIC-NCOL section (item 7) ------------------------
def render_t3():
    f1 = os.path.join(run, "T3", "ncol-cost-1-16.csv")
    f2 = os.path.join(run, "T3", "ncol-cost-d16k.csv")
    if not (os.path.exists(f1) or os.path.exists(f2)):
        return
    print("## T3-SYNTHETIC-NCOL — synthetic n-column cost (n = 1..16)\n")
    print("Purpose: characterize the n-column GGML/GEMV/MMQ cost. "
          "`llama-bench -p N -n 0` measures prompt processing at width N. "
          "**This is NOT MTP verification latency** and must not be reported as such. "
          "Real MTP verify numbers: section 'MTP verification — REAL'.\n")
    for tag, f in (("cost n=1..16", f1), ("cost n=1..8 @ 16k KV", f2)):
        if not os.path.exists(f):
            continue
        print(f"### {tag}\n")
        print("| n_prompt | avg_ns | prompt tok/s | us/token |")
        print("|---|---|---|---|")
        for r in csv.DictReader(open(f)):
            try:
                n = int(float(r.get("n_prompt") or 0))
                ns = float(r.get("avg_ns") or r.get("avg_ns_1") or 0)
                tps = float(r.get("prompt_per_second") or r.get("tok_per_s")
                            or r.get("tok_s") or r.get("avg_ts") or 0)
                if not tps and ns and n:
                    tps = n / ns * 1e9
            except (TypeError, ValueError):
                continue
            us = (ns / n / 1000) if n and ns else 0
            print(f"| {n} | {ns:.0f} | {tps:.1f} | {us:.1f} |")
        print()

# ---------------- acc_per_pos raw + profile --------------------------------
def render_globals():
    f = os.path.join(run, "acc_per_pos.csv")
    if os.path.exists(f):
        print("\n## Per-position acceptance (server trace `-lv 4`)\n```\n"
              + open(f).read() + "```")
    prof = os.path.join(run, "profile", "kernel-summary.csv")
    if os.path.exists(prof):
        print("\n## PROFILE=1 — kernel names / durations / count\n")
        rdr = list(csv.DictReader(open(prof)))
        print("| kernel | count | total ms | mean ms | max ms |")
        print("|---|---|---|---|---|")
        for r in rdr[:20]:
            kn = r["kernel_name"]
            if len(kn) > 90: kn = kn[:87] + "..."
            print(f"| `{kn}` | {r['count']} | {r['total_ms']} | {r['mean_ms']} | {r['max_ms']} |")
        if len(rdr) > 20:
            print(f"\n_… {len(rdr) - 20} more kernels in `profile/kernel-summary.csv`._")
        print("\nFull raw profiler files: `profile/raw/`.")

# ---------------- render: split by power cap when mixed ---------------------
render_t3()
if len(distinct_caps) > 1:
    for cap in distinct_caps:
        print(f"\n---\n# POWER CAP {cap}\n")
        render([r for r in rows if cap_of(r) == cap], cap)
else:
    render(rows, distinct_caps[0] if distinct_caps else "unknown")
render_globals()
