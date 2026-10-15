"""Offline check of FULL16's exploration value vs the old frontier order, at every FX probe moment of an arm.
Oracle of a useful control: its family was clicked by the model in a successful episode of the same task (any arm),
or later in the same episode."""
import json, glob, sys, time
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9, explore_value as EV
import full_run as F
R, U, C = full9.R, full9.U, full9.C
arm = sys.argv[1] if len(sys.argv) > 1 else "FULL12"
RAW = Path("raw/full") / arm

class Ep:
  def __init__(s, t): s.args = type("a", (), {"task": t})()

# oracle: families clicked by the model in successful episodes of each task (all FULL arms + BASE)
useful = defaultdict(set)
for f in glob.glob("raw/full/*/*/p0/agent_transitions.jsonl"):
  res = Path(f).parent / "result.json"
  if not res.exists() or json.load(open(res)).get("is_successful") != 1.0:
    continue
  t = f.split("/")[3]
  for l in open(f):
    o = json.loads(l)
    sel = o.get("selector") or {}
    if sel:
      useful[t].add(C.family_of_sel(sel))

graph_lines = [json.loads(l) for l in open(f"graph_snapshots/full/{arm}/graph.jsonl")]
c = Counter()
for d in sorted(glob.glob(str(RAW / "*" / "p0"))):
  t = d.split("/")[3]
  wf = Path(d) / "windows.jsonl"
  if not wf.exists():
    continue
  meta = json.load(open(Path(d) / "meta.json")); goal = meta["goal"]
  fx = full9.FuncExec(Ep(t), goal)
  gvec = EV.goal_vector(goal, fx)
  later = defaultdict(set)
  for l in open(Path(d) / "steps.jsonl"):
    s = json.loads(l)
    if s.get("kind") == "model" and s.get("tok"):
      later[s["step"]].add(s["tok"] if s["tok"].startswith("E|") else "")
  probes = [json.loads(l) for l in open(Path(d) / "probes.jsonl")] if (Path(d) / "probes.jsonl").exists() else []
  for p in probes:
    if p.get("depth", 1) != 1:
      continue
    t0 = p["t"]
    g = R.Graph(None)
    for o in graph_lines:
      if o.get("t", 0) < t0 and o["instance"] != p["instance"]:
        g.add(o)
    try:
      ui = R.load_ui(Path(d), p["src_cid"])
    except Exception:
      continue
    cands = [s for s in U.enumerate_slots(ui) if s.safe and not s.sel.get("in_list") and not full9.effectful(s.sel)]
    if len(cands) < 2:
      continue
    old = full9.frontier_order(ui, cands, fx.P["ops"], F.ELEMENT_APPS)[0]
    gv = EV.GraphValue(g, RAW.parent, gvec, ui.package)
    import os
    EV.BETA = float(os.environ.get("BETA", EV.BETA))
    spec_w = float(os.environ.get("SPEC", 0))
    def sc(s):
      toks = EV.tokens((s.sel.get("label") or s.sel.get("desc") or "") + " " + (s.sel.get("rid") or "").replace("_", " "))
      spec = (sum(gv.w(t) for t in toks) / len(toks) / 8.0) if toks else 0.0
      return gv.score(ui, s)[0] + spec_w * spec
    scored = sorted(((sc(s), i, s) for i, s in enumerate(cands)), key=lambda x: (-x[0], x[1]))
    new = scored[0][2]
    fut = set().union(*[v for k, v in later.items() if k >= p["step"]]) if later else set()
    for name, s in (("old", old), ("new", new)):
      fam = C.family_of_sel(s.sel)
      c[name, "n"] += 1
      c[name, "useful"] += fam in useful[t]
      c[name, "clicked_later"] += fam in fut
    lv = gv.value(ui, new)[1]
    c["new_level", lv] += 1
    c["new_rel>0"] += gv.value(ui, new)[0] > 0
    c["changed"] += C.family_of_sel(old.sel) != C.family_of_sel(new.sel)
for k in sorted(c, key=str):
  print(k, c[k])
