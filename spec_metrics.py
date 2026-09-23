#!/usr/bin/env python3
"""spec_metrics.py — Prometheus spec-decode counters + MTP survival math.

Shared by benchmark-mi50.sh (per-request counter deltas) and summarize.py
(distributions), and unit-tested directly.

llama-server /metrics exposes exact counters (upstream base 42916d83):

  llamacpp:spec_decode_num_draft_tokens_total
  llamacpp:spec_decode_num_accepted_tokens_total
  llamacpp:spec_decode_num_drafts_total
  llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="N"}

NOTE the real label is **position="N"** (legacy/test builds used `pos`);
the parser accepts both, preferring `position`.

Semantics (llama-server):
  acc_per_pos[i] = n_accepted_per_pos[i] / n_draft_verif_steps
so the per-position values are UNCONDITIONAL SURVIVAL probabilities
  r[i] = P(A >= i+1).
NEVER chain-multiply them again.

Exact per-request metrics come from BEFORE/AFTER counter deltas:
  verification_steps      = delta(spec_decode_num_drafts_total)
  draft_tokens            = delta(spec_decode_num_draft_tokens_total)
  accepted_tokens         = delta(spec_decode_num_accepted_tokens_total)
  mean_draft_tokens_per_step    = draft_tokens / verification_steps
  mean_accepted_tokens_per_step = accepted_tokens / verification_steps
  mean_target_width       = 1 + draft_tokens / verification_steps

Zero deltas for KNOWN per-position counters are preserved (position 2 with
a zero request delta is recorded as 0, never dropped).  requests.csv stores
the raw integer `per_pos_counts` per request so summaries can aggregate
EXACTLY:

  r[i] = sum_j counts_j[i] / sum_j verification_steps_j

(verification_steps-weighted — a 100-step request weighs 10x a 10-step
request; never an equal-weight mean of per-request survival vectors).

A width HISTOGRAM is not derivable from these counters — only means are
exact; the accepted-prefix distribution is derived from survival rates r:
  P(A=0) = 1 - r[0]
  P(A=k) = r[k-1] - r[k]      (0 < k < K)
  P(A=K) = r[K-1]
"""
import re
import urllib.request

SPEC_DRAFT_TOKENS = "llamacpp:spec_decode_num_draft_tokens_total"
SPEC_ACCEPTED_TOKENS = "llamacpp:spec_decode_num_accepted_tokens_total"
SPEC_DRAFTS = "llamacpp:spec_decode_num_drafts_total"
SPEC_PER_POS = "llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
SPEC_COUNTERS = (SPEC_DRAFT_TOKENS, SPEC_ACCEPTED_TOKENS, SPEC_DRAFTS, SPEC_PER_POS)

_LINE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+'
    r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(?:\s+\d+)?\s*$'
)
# Real upstream label is position="N"; legacy/test builds used pos="N".
_POS_LABEL = re.compile(r'(?:^|[,\s])position\s*=\s*"?(\d+)"?')
_POS_LABEL_LEGACY = re.compile(r'(?:^|[,\s])pos\s*=\s*"?(\d+)"?')


def parse_prometheus(text):
    """Parse Prometheus text exposition; keep only the spec_decode counters.

    Returns {name: {"total": float, "by_pos": {int: float}}}.
    Multiple series (slots/instances) are summed into "total"; per-position
    series of SPEC_PER_POS are additionally grouped by their position label
    (`position="N"` upstream, `pos="N"` legacy).
    """
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        if name not in SPEC_COUNTERS:
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        d = out.setdefault(name, {"total": 0.0, "by_pos": {}})
        d["total"] += v
        if name == SPEC_PER_POS:
            pm = _POS_LABEL.search(labels) or _POS_LABEL_LEGACY.search(labels)
            if pm:
                p = int(pm.group(1))
                d["by_pos"][p] = d["by_pos"].get(p, 0.0) + v
    return out


def fetch_metrics(port):
    """GET http://127.0.0.1:<port>/metrics and parse spec_decode counters."""
    with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        return parse_prometheus(r.read().decode("utf-8", "replace"))


def _total(snap, name):
    return float((snap.get(name) or {}).get("total", 0.0))


def counter_delta(before, after):
    """Counter deltas (after - before; counters are monotonic, clamp >= 0).

    Returns {"verification_steps": int, "draft_tokens": int,
             "accepted_tokens": int, "per_pos": {int: int}}.

    ZERO deltas for positions present in the snapshots are PRESERVED —
    a known position whose counter did not move stays in the dict as 0.
    """
    def dtot(name):
        return max(0.0, _total(after, name) - _total(before, name))

    per_pos_after = (after.get(SPEC_PER_POS) or {}).get("by_pos", {}) or {}
    per_pos_before = (before.get(SPEC_PER_POS) or {}).get("by_pos", {}) or {}
    per_pos = {}
    for p in set(per_pos_after) | set(per_pos_before):
        dv = float(per_pos_after.get(p, 0.0)) - float(per_pos_before.get(p, 0.0))
        per_pos[int(p)] = int(round(max(0.0, dv)))   # keep zeros — do not drop
    return {
        "verification_steps": int(round(dtot(SPEC_DRAFTS))),
        "draft_tokens": int(round(dtot(SPEC_DRAFT_TOKENS))),
        "accepted_tokens": int(round(dtot(SPEC_ACCEPTED_TOKENS))),
        "per_pos": per_pos,
    }


def per_pos_vector(per_pos):
    """{0: 8, 1: 5, 2: 0} -> [8, 5, 0] (positions 0..max, gaps/zeros kept)."""
    if not per_pos:
        return []
    kmax = max(per_pos)
    return [int(per_pos.get(i, 0)) for i in range(kmax + 1)]


def survival_from_delta(delta):
    """r[i] = P(A >= i+1) = per_pos_delta[i] / verification_steps.

    These are unconditional survival rates (llama-server semantics).
    Positions must form a contiguous prefix starting at 0; ZERO-count
    trailing positions are kept (they are valid survivals of 0).
    """
    steps = delta.get("verification_steps", 0)
    per_pos = delta.get("per_pos") or {}
    if steps <= 0 or not per_pos:
        return []
    r = []
    for p in sorted(per_pos):
        if p != len(r):          # require contiguous 0..K-1
            break
        r.append(min(1.0, max(0.0, per_pos[p] / steps)))
    return r


def weighted_survival(counts_list, steps_list):
    """EXACT per-position aggregation across requests, weighted by
    verification_steps (never equal-weight):

        r[i] = sum_j counts_j[i] / sum_j verification_steps_j

    over the requests that expose position i (len(counts_j) > i).
    A request with 100 verification steps weighs 10x one with 10.
    """
    if not counts_list or len(counts_list) != len(steps_list):
        return []
    kmax = max(len(c) for c in counts_list)
    r = []
    for i in range(kmax):
        num, den = 0, 0
        for c, s in zip(counts_list, steps_list):
            if i < len(c):
                num += c[i]
                den += s
        r.append(num / den if den else 0.0)
    return r


def weighted_survival_vectors(vectors, steps_list):
    """Legacy fallback when only survival vectors r_ij are available
    (requests.csv without per_pos_counts):

        r[i] = sum_j r_ij*steps_j / sum_j steps_j   over j exposing i."""
    pairs = [(v, s) for v, s in zip(vectors, steps_list)
             if v and s and s > 0]
    if not pairs:
        return []
    kmax = max(len(v) for v, _ in pairs)
    r = []
    for i in range(kmax):
        num, den = 0.0, 0.0
        for v, s in pairs:
            if i < len(v):
                num += v[i] * s
                den += s
        r.append(num / den if den else 0.0)
    return r


def accepted_prefix_distribution(survival):
    """Accepted-prefix distribution from survival rates r (do NOT chain-multiply).

    P(A=0) = 1 - r[0]
    P(A=k) = r[k-1] - r[k]   for 0 < k < K
    P(A=K) = r[K-1]
    Returns [] for empty survival; probabilities clamped to >= 0.
    """
    if not survival:
        return []
    K = len(survival)
    dist = [0.0] * (K + 1)
    dist[0] = 1.0 - survival[0]
    for k in range(1, K):
        dist[k] = survival[k - 1] - survival[k]
    dist[K] = survival[K - 1]
    return [max(0.0, float(x)) for x in dist]


def means_from_delta(delta):
    """Exact per-request means from counter deltas (None if no spec activity)."""
    steps = delta.get("verification_steps", 0)
    if steps <= 0:
        return None
    draft = delta.get("draft_tokens", 0)
    acc = delta.get("accepted_tokens", 0)
    return {
        "verification_steps": steps,
        "draft_tokens": draft,
        "accepted_tokens": acc,
        "mean_draft_tokens_per_step": draft / steps,
        "mean_accepted_tokens_per_step": acc / steps,
        "mean_target_width": 1.0 + draft / steps,
    }


def average_survival(vectors):
    """EQUAL-weight element-wise mean — only for the server-trace fallback
    where per-request verification_steps are unavailable.  Never use when
    exact steps are known (see weighted_survival / weighted_survival_vectors)."""
    vecs = [v for v in vectors if v]
    if not vecs:
        return []
    k = min(len(v) for v in vecs)
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(k)]


def parse_survival_field(s):
    """Parse the requests.csv `survival_per_pos` field (';separated, no commas)."""
    out = []
    for x in (s or "").split(";"):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def parse_counts_field(s):
    """Parse the requests.csv `per_pos_counts` field (';separated integers)."""
    out = []
    for x in (s or "").split(";"):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except ValueError:
            pass
    return out
