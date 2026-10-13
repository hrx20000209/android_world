"""Same-app consecutive-task experiment: success, steps, calls, jumps, by pass."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RAW = C.ROOT / "raw" / "seq"
OUT = C.ROOT / "analysis" / "seq"
ARMS = ["BASE", "G", "GX", "GC"]


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  man = json.loads((C.ROOT / "configs" / "seq_manifest.json").read_text())
  order = {}
  for a in man["apps"]:
    for i, t in enumerate(a["tasks"]):
      order[t["task"]] = (a["app"], i)
  eps = []
  for d in sorted(RAW.glob("*/*/*")):
    if not (d / "result.json").exists() or "_partial_" in d.name:
      continue
    m = json.loads((d / "meta.json").read_text())
    r = json.loads((d / "result.json").read_text())
    steps = C.jl(d / "steps.jsonl")
    p = int(d.name.split("_")[0][1:])
    app, ti = order[m["task"]]
    eps.append({"arm": m.get("seq_arm"), "task": m["task"], "app": app, "task_idx": ti, "pass": p, "seed": m["seed"],
                "success": r.get("is_successful") == 1.0, "error": r.get("error"), "steps": r.get("steps") or 0,
                "calls": r.get("model_calls") or 0, "jumps": r.get("jumps", 0), "jump_steps": r.get("jump_steps", 0),
                "jump_rejected": r.get("jump_rejected", 0), "shortcuts": r.get("shortcuts", 0), "repairs": r.get("repairs", 0),
                "probes": r.get("n_probes", 0), "disabled": bool(r.get("explore_disabled")), "wall_s": r.get("wall_s"),
                "requery": sum(bool(s.get("requery")) for s in steps),
                "jump_hops_shortcut": sum(1 for s in steps if s.get("kind") == "jump" and s.get("shortcut") and s.get("match")),
                "model_s": sum(s.get("model_latency_s") or 0 for s in steps), "ep": str(d)})
  S = {}
  for arm in ARMS:
    for label, f in (("all", lambda e: True), ("pass0 (cross-task only)", lambda e: e["pass"] == 0),
                     ("pass1-2 (+same-task history)", lambda e: e["pass"] > 0)):
      E = [e for e in eps if e["arm"] == arm and f(e)]
      if not E:
        continue
      S.setdefault(arm, {})[label] = {
          "episodes": len(E), "success": sum(e["success"] for e in E), "errors": sum(bool(e["error"]) for e in E),
          "steps": sum(e["steps"] for e in E), "calls": sum(e["calls"] for e in E), "jumps": sum(e["jumps"] for e in E),
          "jump_steps": sum(e["jump_steps"] for e in E), "jump_rejected": sum(e["jump_rejected"] for e in E),
          "shortcut_hops": sum(e["jump_hops_shortcut"] for e in E), "repairs": sum(e["repairs"] for e in E),
          "probes": sum(e["probes"] for e in E), "disabled": sum(e["disabled"] for e in E),
          "requery": sum(e["requery"] for e in E), "wall_s_mean": round(float(np.mean([e["wall_s"] for e in E])), 1),
          "succ_steps_mean": round(float(np.mean([e["steps"] for e in E if e["success"]])), 2) if any(e["success"] for e in E) else None,
          "succ_calls_mean": round(float(np.mean([e["calls"] for e in E if e["success"]])), 2) if any(e["success"] for e in E) else None}
  # paired vs BASE
  by = defaultdict(dict)
  for e in eps:
    by[(e["task"], e["pass"])][e["arm"]] = e
  for arm in ("G", "GX", "GC"):
    pr = [(v["BASE"], v[arm]) for v in by.values() if "BASE" in v and arm in v]
    both = [(x, y) for x, y in pr if x["success"] and y["success"]]
    S.setdefault(arm, {})["paired_vs_BASE"] = {
        "pairs": len(pr), "arm_only": sum(y["success"] and not x["success"] for x, y in pr),
        "base_only": sum(x["success"] and not y["success"] for x, y in pr),
        "both_success_pairs": len(both),
        "both_success_steps": [sum(x["steps"] for x, _ in both), sum(y["steps"] for _, y in both)],
        "both_success_calls": [sum(x["calls"] for x, _ in both), sum(y["calls"] for _, y in both)]}
  per_task = defaultdict(dict)
  for e in eps:
    c = per_task[e["task"]].setdefault(e["arm"], [0, 0, 0, 0])
    c[0] += e["success"]; c[1] += 1; c[2] += e["calls"]; c[3] += e["jump_steps"]
  S["per_task"] = {t: {a: f"{v[0]}/{v[1]} calls={v[2]} jumped={v[3]}" for a, v in d.items()} for t, d in per_task.items()}
  (OUT / "episodes.json").write_text(json.dumps(eps, indent=1))
  (OUT / "summary.json").write_text(json.dumps(S, indent=1))
  print(json.dumps(S, indent=1))


if __name__ == "__main__":
  main()
