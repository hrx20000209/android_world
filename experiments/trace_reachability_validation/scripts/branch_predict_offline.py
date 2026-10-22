"""Offline bound for speculative branch exploration: at each model click step of a finished arm, rank the on-screen
controls the window could probe (safe, not data-changing, Back expected to undo) by the graph exploration value
(TF-IDF relevance with graph evidence, as the window does), and check whether the model's chosen control is in the
top-K. Also reports how often the chosen control is probe-able at all."""
import glob, json, random, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402
import explore_value as EV  # noqa: E402
R, U, C = full9.R, full9.U, full9.C


def main(arm, n=300, seed=0):
  random.seed(seed)
  raw = Path("raw/full") / arm
  lines = sorted((json.loads(l) for l in open(f"graph_snapshots/full/{arm}/graph.jsonl")), key=lambda o: o.get("t", 0))
  samples = []
  for f in sorted(glob.glob(str(raw / "*" / "p0" / "steps.jsonl"))):
    d = Path(f).parent
    for r in (json.loads(l) for l in open(f)):
      if r.get("kind") == "model" and r.get("pre") and (r.get("action") or {}).get("action_type") == "click" \
          and r.get("tok", "").startswith(("E|", "L|")):
        samples.append((d, r))
  random.shuffle(samples)
  c = Counter()
  for d, r in samples[:n]:
    task = d.parts[-2]
    goal = json.load(open(d / "meta.json"))["goal"]
    ui = R.load_ui(d, r["pre"]["cid"])
    gr = R.Graph(None)
    for o in lines:
      if o.get("t", 0) >= r["pre"]["t"]:
        break
      gr.add(o)
    fx = full9.FuncExec(type("E", (), {"args": type("a", (), {"task": task})()})(), goal)
    gv = EV.GraphValue(gr, raw, EV.goal_vector(goal, fx), ui.package, app_doc_idf=True, instance=f"{task}/x")
    gv.pvec = EV.param_vector(fx)
    evid = full9.container_evidence(gr)
    app_ev = full9.app_kind_evidence(gr, split_rows=True)
    cands = []
    for sl in U.enumerate_slots(ui):
      if not sl.safe or full9.effectful(sl.sel):
        continue
      kind = "discover" + ("_row" if sl.sel.get("in_list") else "")
      if full9.p_reversible_h(ui, sl.sel, evid, app_ev, kind) < 0.7:
        continue
      cands.append((gv.score(ui, sl)[0], C.family_of_sel(sl.sel)))
    cands.sort(key=lambda x: -x[0])
    fams = [f for _, f in cands]
    tok = r["tok"]
    c["steps"] += 1
    c["probeable"] += tok in fams
    for k in (1, 3, 5):
      c[f"top{k}"] += tok in fams[:k]
    c["n_cands"] += len(fams)
  s = c["steps"]
  print(f"[{arm}] element-click model steps {s}: chosen control probe-able {c['probeable']} ({c['probeable']/s:.0%}); "
        f"in top-1 {c['top1']} ({c['top1']/s:.0%}), top-3 {c['top3']} ({c['top3']/s:.0%}), top-5 {c['top5']} "
        f"({c['top5']/s:.0%}); probe-able candidates per screen {c['n_cands']/s:.1f}")


if __name__ == "__main__":
  main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
