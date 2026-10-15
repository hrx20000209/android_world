"""Offline replay: would a function from OTHER templates of the same app trigger on
the held-out BASE trajectory, and would its actions match what the model did?"""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fgraph as FG, memory as M
C, U, R = M.C, M.U, M.R
ROOTS = [M.TRV / "raw/full"]
print("functions cached:", FG.build_cache(ROOTS), flush=True)
libs = {}
stats = Counter(); ex = []
for ep in sorted((M.TRV / "raw/full/BASE").glob("*/p0")):
  task = ep.parent.name; meta = json.loads((ep / "meta.json").read_text())
  res = json.loads((ep / "result.json").read_text()); succ = res.get("is_successful") == 1.0
  tk = M.template_key(task)
  if tk not in libs:
    libs[tk] = FG.Library(task)
  lib = libs[tk]
  P = FG.parse_goal(meta["goal"]); recs = FG.records(P)
  rows = [json.loads(l) for l in open(ep / "steps.jsonl")]; rows = [r for r in rows if r.get("kind") == "model"]
  i, used_rec = 0, 0
  while i < len(rows):
    ui = R.load_ui(ep, rows[i]["pre"]["cid"])
    cands = lib.candidates(P, ui.activity)
    f = None
    for c0 in cands:
      s0 = c0["steps"][0]
      live = {C.family_of_sel(sl.sel) for sl in U.enumerate_slots(ui)}
      if s0["op"] == "nav" and s0["fam"] in live or s0["op"] != "nav":
        f = c0; break
    if f is None:
      i += 1; continue
    stats["triggers"] += 1; stats[f"trig_{'succ' if succ else 'fail'}"] += 1
    m = 0
    for j, st in enumerate(f["steps"]):
      if i + j >= len(rows):
        break
      r = rows[i + j]; a = r["action"]; u2 = R.load_ui(ep, r["pre"]["cid"])
      tok, fam, _ = C.action_token(u2, a)
      vals = recs.get(st.get("field"), [])
      val = vals[min(used_rec, len(vals) - 1)] if vals else None
      if st["op"] == "nav":
        ok = fam == st["fam"]
      elif st["op"] == "type":
        ok = a.get("action_type") == "input_text" and FG.norm(a.get("text")) == FG.norm(val)
      else:
        n = U.element_at(u2, int(a["x"]), int(a["y"])) if a.get("x") is not None else None
        inner = " ".join(FG.norm(x.text or x.desc) for x in u2.app_nodes() if n is not None and (x.text or x.desc)
                         and n.bounds[0] <= x.center[0] < n.bounds[2] and n.bounds[1] <= x.center[1] < n.bounds[3])
        ok = val is not None and FG.norm(val) in inner
      if not ok:
        break
      m += 1
    stats["steps_predicted"] += len(f["steps"]); stats["steps_matched_prefix"] += m
    stats["full_match"] += m == len(f["steps"])
    if len(ex) < 25:
      ex.append((task, rows[i]["step"], succ, f["src"], [(s["op"], (s.get("field") or s.get("label") or "")[:18]) for s in f["steps"]], m))
    i += max(1, m); used_rec += 1 if m == len(f["steps"]) else 0
print(dict(stats))
for e in ex: print(e)
