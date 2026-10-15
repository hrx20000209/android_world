"""Exploration read-outs of an arm: probes, rollback, relaunch-replay restoration, graph share."""
import json, glob, sys
from collections import Counter
arm = sys.argv[1]; c = Counter()
for d in glob.glob(f"raw/full/{arm}/*/p0"):
  try:
    for l in open(d + "/windows.jsonl"):
      w = json.loads(l); c["windows"] += 1; c["stop_" + (w.get("stop") or "none")] += 1
      for p in w.get("probes", []):
        c["probes"] += 1; c["rb_fail"] += not p["rollback_ok"]
      if "relaunch_restore" in w: c["relaunch_" + str(w["relaunch_restore"])] += 1
  except FileNotFoundError: pass
g = Counter(json.loads(l)["origin"] for l in open(f"graph_snapshots/full/{arm}/graph.jsonl"))
print(f"[{arm} explore] windows={c['windows']} probes={c['probes']} back_fail={c['rb_fail']} relaunch_ok={c['relaunch_True']} "
      f"relaunch_fail={c['relaunch_False']} transient_skip={c['stop_transient_state']} not_restorable_skip={c['stop_not_restorable']} "
      f"graph_edges agent={g['agent']} explore={g['explore']}")
