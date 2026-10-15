"""Stage B/C offline: applicability on held-out BASE executions (LOO memory)."""
import json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory as M, retrieval as RT
C, U, R = M.C, M.U, M.R
eps = sorted((M.TRV / "raw/full/BASE").glob("*/p*"))
snaps = {}
cnt = Counter(); rows_out = []; lat = []
for ep in eps:
  meta = json.loads((ep / "meta.json").read_text()); res = json.loads((ep / "result.json").read_text())
  task, goal = meta["task"], meta["goal"]; succ = res.get("is_successful") == 1.0
  tk = M.template_key(task)
  if tk not in snaps:
    snaps[tk] = M.Snapshot(task)
  snap = snaps[tk]
  ret = RT.BM25(snap).search(goal)
  cnt["ret_abstain"] += not ret
  ev = RT.Evidence(snap)
  for l in open(ep / "steps.jsonl"):
    r = json.loads(l)
    if r.get("kind") != "model":
      continue
    ui = R.load_ui(ep, r["pre"]["cid"])
    t0 = time.time(); rows, info = RT.classify(ui, goal, ev); lat.append(time.time() - t0)
    a = r["action"]; tok, fam, sel = C.action_token(ui, a)
    model_row = next((x for x in rows if x["fam"] == fam), None) if fam else None
    cnt["steps"] += 1
    cnt[f"model_cls:{model_row['cls'] if model_row else ('nonclick' if not fam else 'unslotted')}"] += 1
    if info["direct"] is not None:
      ok = info["direct"]["fam"] == fam
      cnt[f"direct_{'succ' if succ else 'fail'}_{'match' if ok else 'mismatch'}"] += 1
      rows_out.append(("DIRECT", succ, ok, task, r["step"], info["direct"]["label"], info["direct"]["reach"][:3], tok[:45]))
    elif info["n_anchor1"] == 1 and info["n_unknown"] > 0 and not info["anchors_here"]:
      a1 = next(x for x in rows if x["cls"] == "ANCHOR1")
      cnt[f"blocked_by_unknown_{'succ' if succ else 'fail'}_{'match' if a1['fam'] == fam else 'mismatch'}"] += 1
      rows_out.append(("BLOCKED", succ, a1["fam"] == fam, task, r["step"], a1["label"], a1["reach"][:3], tok[:45]))
    elif info["n_anchor1"] == 0 and info["n_unknown"] > 0 and not info["anchors_here"]:
      cnt["no_anchor_evidence_unknown_exists"] += 1
print(json.dumps(cnt, indent=1, sort_keys=True))
print("classify latency ms p50", round(sorted(lat)[len(lat)//2]*1000, 1), "p95", round(sorted(lat)[int(len(lat)*.95)]*1000, 1))
for x in rows_out: print(x)
