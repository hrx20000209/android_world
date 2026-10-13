"""Pilot A metrics: coverage@k, MRR, novel coverage, two-hop, by app/history."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.ROOT / "analysis" / "pilot_a"
POL = ["P0", "P0u", "P1", "P2", "P3", "P4", "ORA"]


def cov(rows, p, k, denom):
  base = rows if denom == "all" else [r for r in rows if r["eligible"]]
  hit = sum(1 for r in base if r[f"rank_{p}"] is not None and r[f"rank_{p}"] <= k)
  return hit, len(base)


def novel(rows, p, k):
  """Agent's next action is safe, its outcome was NOT yet in the graph, and the
  policy would probe it within k: the probe yields new, used information."""
  base = [r for r in rows if r["eligible"]]
  hit = sum(1 for r in base if not r["target_known"] and r[f"rank_{p}"] is not None and r[f"rank_{p}"] <= k)
  return hit, len(base), sum(1 for r in base if not r["target_known"])


def mrr(rows, p):
  base = [r for r in rows if r["eligible"]]
  return float(np.mean([1 / r[f"rank_{p}"] for r in base])) if base else 0


def main(key="struct_sig"):
  rows = [json.loads(l) for l in (OUT / f"decisions_{key}.jsonl").read_text().splitlines()]
  S = {"key": key}
  for setting in ("warm", "loto", "loao", "cold"):
    R = [r for r in rows if r["setting"] == setting]
    s = {"n_all": len(R), "n_eligible": sum(r["eligible"] for r in R),
         "n_target_unknown_eligible": sum(1 for r in R if r["eligible"] and not r["target_known"]),
         "mean_cands": float(np.mean([r["n_cands"] for r in R])),
         "matched_context_rate": sum(1 for r in R if r["longest_order"] >= 0) / len(R),
         "longest_order_dist": dict(Counter(r["longest_order"] for r in R)),
         "policies": {}}
    for p in POL:
      d = {}
      for k in (1, 2, 3, 5):
        h, n = cov(R, p, k, "eligible")
        ha, na = cov(R, p, k, "all")
        hn, nn, nu = novel(R, p, k)
        d[f"top{k}_eligible"] = round(h / n, 4)
        d[f"top{k}_all"] = round(ha / na, 4)
        d[f"novel{k}"] = [hn, nn]
      d["mrr_eligible"] = round(mrr(R, p), 4)
      s["policies"][p] = d
    # by app
    by = defaultdict(list)
    for r in R:
      by[r["app"]].append(r)
    s["by_app_top1_eligible"] = {a: {p: round(cov(v, p, 1, "eligible")[0] / max(1, cov(v, p, 1, "eligible")[1]), 3)
                                     for p in ("P0", "P1", "P3")} | {"n": sum(x["eligible"] for x in v)}
                                 for a, v in by.items()}
    # by history length (longest matched order)
    byk = defaultdict(list)
    for r in R:
      byk[r["longest_order"]].append(r)
    s["by_longest_order_top1"] = {k: {p: round(cov(v, p, 1, "eligible")[0] / max(1, cov(v, p, 1, "eligible")[1]), 3)
                                      for p in ("P0", "P1", "P2", "P3")} | {"n": sum(x["eligible"] for x in v)}
                                  for k, v in sorted(byk.items())}
    # two-hop: consecutive eligible clicks in same episode
    eps = defaultdict(list)
    for r in R:
      eps[r["ep"]].append(r)
    two = defaultdict(lambda: [0, 0])
    for ep, lst in eps.items():
      lst.sort(key=lambda r: r["step"])
      for a, b in zip(lst, lst[1:]):
        if b["step"] != a["step"] + 1 or not (a["eligible"] and b["eligible"]):
          continue
        for p in ("P0", "P1", "P3", "ORA"):
          for k1, k2 in ((1, 1), (2, 1), (2, 2), (3, 2)):
            ok = a[f"rank_{p}"] and a[f"rank_{p}"] <= k1 and b[f"rank_{p}"] and b[f"rank_{p}"] <= k2
            two[f"{p}@{k1}x{k2}"][0] += int(bool(ok))
            two[f"{p}@{k1}x{k2}"][1] += 1
    s["two_hop"] = dict(two)
    S[setting] = s
  (OUT / f"summary_{key}.json").write_text(json.dumps(S, indent=1))
  for setting in ("warm", "loto", "loao", "cold"):
    s = S[setting]
    print(f"== {setting}  eligible {s['n_eligible']}/{s['n_all']}  cands {s['mean_cands']:.1f}  matched ctx {s['matched_context_rate']:.2f}")
    for p in POL:
      d = s["policies"][p]
      print(f"  {p:4s} top1 {d['top1_eligible']:.3f} top2 {d['top2_eligible']:.3f} top3 {d['top3_eligible']:.3f} "
            f"(all {d['top1_all']:.3f}/{d['top3_all']:.3f}) mrr {d['mrr_eligible']:.3f} novel@1 {d['novel1']} novel@3 {d['novel3']}")
    print("  two-hop", {k: f"{v[0]}/{v[1]}" for k, v in s["two_hop"].items() if k.endswith("2x1") or k.endswith("1x1")})


if __name__ == "__main__":
  main(sys.argv[1] if len(sys.argv) > 1 else "struct_sig")
