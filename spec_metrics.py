#!/usr/bin/env python3
"""spec_metrics.py — Prometheus spec-decode counters + MTP survival math.

Shared by benchmark-mi50.sh (per-request counter deltas) and summarize.py
(distributions), and unit-tested directly.

llama-server /metrics exposes exact counters:

  llamacpp:spec_decode_num_draft_tokens_total
  llamacpp:spec_decode_num_accepted_tokens_total
  llamacpp:spec_decode_num_drafts_total
  llamacpp:spec_decode_num_accepted_tokens_per_pos_total

Semantics (llama-server):
  acc_per_pos[i] = n_accepted_per_pos[i] / n_draft_verif_steps
so the per-position values are UNCONDITIONAL SURVIVAL probabilities
  r[i] = P(A >= i+1).
NEVER chain-multiply them again.

Exact per-request metrics come from BEFORE/AFTER counter deltas:
  verification_steps      = delta(spec_decode_num_drafts_total)
  draft_tokens            = delta(spec_decode_num_draft_tokens_total)
  accepted_tokens         = delta(spec_decode_num_accepted_tokens_total)
  mean_draft_tokens_per_step   = draft_tokens / verification_steps
  mean_accepted_tokens_per_step = accepted_tokens / verification_steps
  mean_target_width       = 1 + draft_tokens / verification_steps
      (target batch = 1 sampled token + proposed draft tokens)

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
_POS_LABEL = re.compile(r'(?:^|[,\s])pos\s*=\s*"?(\d+)"?')


def parse_prometheus(text):
    """Parse Prometheus text exposition; keep only the spec_decode counters.

    Returns {name: {"total": float, "by_pos": {int: float}}}.
    Multiple series (slots/instances) are summed into "total"; per-position
    series of SPEC_PER_POS are additionally grouped by their pos label.
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
            pm = _POS_LABEL.search(labels)
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
    """Counter deltas (after - before, clamped >= 0).

    Returns {"verification_steps": int, "draft_tokens": int,
             "accepted_tokens": int, "per_pos": {int: int}}.
    """
    def dtot(name):
        return max(0.0, _total(after, name) - _total(before, name))

    per_pos_after = (after.get(SPEC_PER_POS) or {}).get("by_pos", {}) or {}
    per_pos_before = (before.get(SPEC_PER_POS) or {}).get("by_pos", {}) or {}
    per_pos = {}
    for p in set(per_pos_after) | set(per_pos_before):
        dv = max(0.0, float(per_pos_after.get(p, 0.0)) - float(per_pos_before.get(p, 0.0)))
        if dv > 0:
            per_pos[int(p)] = int(round(dv))
    return {
        "verification_steps": int(round(dtot(SPEC_DRAFTS))),
        "draft_tokens": int(round(dtot(SPEC_DRAFT_TOKENS))),
        "accepted_tokens": int(round(dtot(SPEC_ACCEPTED_TOKENS))),
        "per_pos": per_pos,
    }


def survival_from_delta(delta):
    """r[i] = P(A >= i+1) = per_pos_delta[i] / verification_steps.

    These are unconditional survival rates (llama-server semantics).
    Enforces 0 <= r <= 1 and non-increasing; positions must form a
    contiguous prefix starting at 0 (gaps truncate the vector).
    """
    steps = delta.get("verification_steps", 0)
    per_pos = delta.get("per_pos") or {}
    if steps <= 0 or not per_pos:
        return []
    r = []
    last = None
    for p in sorted(per_pos):
        if p != len(r):          # require contiguous 0..K-1
            break
        v = min(1.0, max(0.0, per_pos[p] / steps))
        if last is not None:
            v = min(v, last)     # survival must be non-increasing
        r.append(v)
        last = v
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
    """Element-wise mean of survival vectors (equal weight per request),
    truncated to the shortest non-empty vector."""
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
