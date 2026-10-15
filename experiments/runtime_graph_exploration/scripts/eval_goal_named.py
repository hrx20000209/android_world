"""Offline: how often the model's next click is the unique goal-named control."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory as M, retrieval as RT
C, U, R = M.C, M.U, M.R
cnt = Counter(); ex = []
for ep in sorted((M.TRV / "raw/full/BASE").glob("*/p*")):
  meta = json.loads((ep / "meta.json").read_text()); res = json.loads((ep / "result.json").read_text())
  goal = meta["goal"]; gn = M.norm(goal); succ = res.get("is_successful") == 1.0
  clicked = set()
  for l in open(ep / "steps.jsonl"):
    r = json.loads(l)
    if r.get("kind") != "model":
      continue
    ui = R.load_ui(ep, r["pre"]["cid"])
    tok, fam, sel = C.action_token(ui, r["action"])
    G = []
    for sl in U.enumerate_slots(ui):
      lab = sl.sel.get("label") or sl.sel.get("desc") or ""
      f = C.family_of_sel(sl.sel)
      key = (f, M.norm(lab))
      if lab and RT.goal_named(lab, gn) and key not in clicked:
        G.append((sl, f, key))
    kinds = "safe" if all(g[0].safe for g in G) else "mixed"
    if len(G) == 1:
      sl, f, key = G[0]
      cls = "row" if sl.sel.get("in_list") else ("safe" if sl.safe else sl.unsafe_reason)
      hit = (fam == f) and M.norm(sl.sel.get("label") or sl.sel.get("desc") or "") == key[1]
      cnt[(cls, "succ" if succ else "fail", "hit" if hit else "miss")] += 1
      if not hit and succ and len(ex) < 40:
        ex.append((meta["task"], r["step"], cls, key[1][:30], tok[:40]))
    cnt["steps"] += 1
    if fam:
      n = U.element_at(ui, int(r["action"]["x"]), int(r["action"]["y"])) if r["action"].get("x") is not None else None
      if n is not None:
        clicked.add((fam, M.norm(U.label_of(ui, n))))
for k, v in sorted(cnt.items(), key=str): print(k, v)
for e in ex: print(e)
