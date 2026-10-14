"""Full AndroidWorld analysis: BASE vs FULL per pass, paired, mechanisms."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RAW = C.ROOT / "raw" / "full"
OUT = C.ROOT / "analysis" / "full"


def app_of(task):
  for p in ("Expense", "Markor", "Recipe", "SimpleCalendar", "SimpleSms", "System", "Tasks", "Notes", "Sports",
            "Osm", "Audio", "Browser", "Camera", "Clock", "Contacts", "Files", "Retro", "Vlc", "SimpleDraw",
            "Gallery", "SaveCopy", "Turn", "Open", "Expense"):
    if task.startswith(p):
      return p
  return task[:8]


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  eps = []
  for d in sorted(RAW.glob("*/*/p*")):
    if not (d / "result.json").exists() or "_partial_" in d.name:
      continue
    m = json.loads((d / "meta.json").read_text())
    r = json.loads((d / "result.json").read_text())
    e = {"arm": m.get("full_arm"), "task": m["task"], "app": app_of(m["task"]), "pass": int(d.name[1:]),
         "success": r.get("is_successful") == 1.0, "error": r.get("error"), "steps": r.get("steps") or 0,
         "calls": r.get("model_calls") or 0, "wall_s": r.get("wall_s") or 0}
    for k, v in r.items():
      if k.startswith("m_"):
        e[k] = v
    eps.append(e)
  pre = set((C.ROOT / "configs" / "full_p1_prefix_tasks.txt").read_text().split()) if (C.ROOT / "configs" / "full_p1_prefix_tasks.txt").exists() else set()
  S = {}
  mech_keys = sorted({k for e in eps for k in e if k.startswith("m_")})
  for arm in ("BASE", "FULL", "FULLNE", "FULL2", "FULL3", "FULL4", "FULL5"):
    for label, f in (("all", lambda e: True), ("pass0", lambda e: e["pass"] == 0), ("pass1", lambda e: e["pass"] == 1),
                     ("pass1_postfix", lambda e: e["pass"] == 1 and e["task"] not in pre)):
      E = [e for e in eps if e["arm"] == arm and f(e)]
      if not E:
        continue
      d = {"episodes": len(E), "success": sum(e["success"] for e in E), "errors": sum(bool(e["error"]) for e in E),
           "steps": sum(e["steps"] for e in E), "calls": sum(e["calls"] for e in E),
           "mean_steps_success": round(float(np.mean([e["steps"] for e in E if e["success"]])), 2) if any(e["success"] for e in E) else None,
           "wall_s_mean": round(float(np.mean([e["wall_s"] for e in E])), 1)}
      for k in mech_keys:
        d[k] = sum(e.get(k, 0) for e in E)
      S.setdefault(arm, {})[label] = d
  by = defaultdict(dict)
  for e in eps:
    by[(e["task"], e["pass"])][e["arm"]] = e
  for other in ("FULL", "FULL2", "FULL3", "FULL4", "FULL5"):
    key = "paired" if other == "FULL" else f"paired_{other}"
    for label, f in (("all", lambda t, p: True), ("pass0", lambda t, p: p == 0), ("pass1", lambda t, p: p == 1),
                     ("pass1_postfix", lambda t, p: p == 1 and t not in pre)):
      pr = [(v["BASE"], v[other]) for (t, p), v in by.items() if "BASE" in v and other in v and f(t, p)]
      both = [(x, y) for x, y in pr if x["success"] and y["success"]]
      S.setdefault(key, {})[label] = {
          "pairs": len(pr), "full_only": sum(y["success"] and not x["success"] for x, y in pr),
          "base_only": sum(x["success"] and not y["success"] for x, y in pr),
          "both": len(both), "both_steps": [sum(x["steps"] for x, _ in both), sum(y["steps"] for _, y in both)],
          "both_calls": [sum(x["calls"] for x, _ in both), sum(y["calls"] for _, y in both)],
          "both_wall": [round(sum(x["wall_s"] for x, _ in both)), round(sum(y["wall_s"] for _, y in both))]}
  apps = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
  for e in eps:
    a = apps[e["app"]][e["arm"]]
    a[0] += e["success"]; a[1] += 1; a[2] += e["calls"]
  S["per_app"] = {k: {arm: f"{v[0]}/{v[1]} calls={v[2]}" for arm, v in d.items()} for k, d in sorted(apps.items())}
  (OUT / "episodes.json").write_text(json.dumps(eps, indent=1))
  (OUT / "summary.json").write_text(json.dumps(S, indent=1))
  print(json.dumps({k: v for k, v in S.items() if k != "per_app"}, indent=1))


if __name__ == "__main__":
  main()
