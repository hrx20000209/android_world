"""Phase III closed-loop analysis: success, model calls, latency, prefill stats."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ARMS = ["BASE", "BASE2", "BASE3", "PF", "PFV", "PFVS", "PFVL", "PFVA", "PFVA2"]


def main(raw="raw/phase3", out="analysis/phase3"):
  RAW, OUT = C.ROOT / raw, C.ROOT / out
  OUT.mkdir(parents=True, exist_ok=True)
  eps = []
  for d in sorted(RAW.glob("*/*")):
    if not (d / "result.json").exists() or "_partial_" in d.name:
      continue
    m = json.loads((d / "meta.json").read_text())
    r = json.loads((d / "result.json").read_text())
    steps = C.jl(d / "steps.jsonl")
    eps.append({"task": m["task"], "seed": m["seed"], "arm": m.get("pf_arm"), "success": r.get("is_successful") == 1.0,
                "error": r.get("error"), "steps": r.get("steps"), "model_calls": r.get("model_calls"),
                "prefills": r.get("prefills", 0), "rejected": r.get("prefill_rejected", 0), "wall_s": r.get("wall_s"),
                "model_s": sum(s.get("model_latency_s") or 0 for s in steps),
                "prefill_s": sum(s.get("prefill_s") or 0 for s in steps),
                "verify": [s.get("verify") for s in steps if s.get("verify")],
                "rollback_fail": sum(1 for s in steps if s.get("rollback") and not s["rollback"]["ok"]),
                "prefill_steps": [s["step"] for s in steps if s.get("prefill")], "n_steps": len(steps), "ep": str(d)})
  S = {}
  for a in ARMS:
    E = [e for e in eps if e["arm"] == a]
    if not E:
      continue
    S[a] = {"episodes": len(E), "success": sum(e["success"] for e in E), "errors": sum(bool(e["error"]) for e in E),
            "steps": sum(e["steps"] or 0 for e in E), "model_calls": sum(e["model_calls"] or 0 for e in E),
            "prefills": sum(e["prefills"] for e in E), "rejected": sum(e["rejected"] for e in E),
            "rollback_fail": sum(e["rollback_fail"] for e in E),
            "wall_s_mean": float(np.mean([e["wall_s"] for e in E])),
            "model_s_total": round(sum(e["model_s"] for e in E), 1), "prefill_s_total": round(sum(e["prefill_s"] for e in E), 1),
            "episodes_with_prefill": sum(e["prefills"] > 0 for e in E)}
  # paired vs BASE
  by = defaultdict(dict)
  for e in eps:
    by[(e["task"], e["seed"])][e["arm"]] = e
  for a in ARMS[1:]:
    pr = [(v["BASE"], v[a]) for v in by.values() if "BASE" in v and a in v]
    if not pr:
      continue
    S[a]["paired_vs_base"] = {"pairs": len(pr), "arm_only": sum(y["success"] and not x["success"] for x, y in pr),
                              "base_only": sum(x["success"] and not y["success"] for x, y in pr),
                              "calls_saved": sum((x["model_calls"] or 0) - (y["model_calls"] or 0) for x, y in pr),
                              "steps_diff": sum((y["steps"] or 0) - (x["steps"] or 0) for x, y in pr)}
  per_task = defaultdict(dict)
  for e in eps:
    t = per_task[e["task"]].setdefault(e["arm"], [0, 0, 0, 0])
    t[0] += e["success"]
    t[1] += 1
    t[2] += e["model_calls"] or 0
    t[3] += e["prefills"]
  S["per_task"] = {t: {a: f"{v[0]}/{v[1]} calls={v[2]} pf={v[3]}" for a, v in d.items()} for t, d in per_task.items()}
  (OUT / "episodes.json").write_text(json.dumps(eps, indent=1))
  (OUT / "summary.json").write_text(json.dumps(S, indent=1))
  print(json.dumps(S, indent=1))


if __name__ == "__main__":
  main(*sys.argv[1:])
