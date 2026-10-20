"""Offline bound for goal-directed routing: the graph does not choose the model's action; it only takes the model
to the page where the task's substantive work happens, over inference + exploration edges.

For each episode (run order): the work page W = the page of the first model action that is task-specific (entity
row, typing, commit or answer). Navigation steps the model spent before reaching W (inside the app) are the
saveable inferences. At each of those steps, with the graph as of that moment (all origins, all tasks):
  identifiable : W's activity holds the graph page that ranks first by TF-IDF relevance (goal vector vs. page text,
                 app-as-document IDF) among the app's known pages other than the current one
  reachable    : a path of <= 4 safe / creation-entry edges from the current activity to that page
The bound counts the navigation steps from the first step at which both hold until W is reached.
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402
import explore_value as EV  # noqa: E402
import joint_graph as JG  # noqa: E402
import cross_task_bound as X  # noqa: E402

R, U, C = full9.R, full9.U, full9.C


def run(arm, max_hops=4):
  raw = Path("raw/full") / arm
  lines = sorted((json.loads(l) for l in open(f"graph_snapshots/full/{arm}/graph.jsonl")), key=lambda o: o.get("t", 0))
  c = Counter()
  for f in sorted(glob.glob(str(raw / "*" / "p0" / "steps.jsonl"))):
    d = Path(f).parent
    t = d.parts[-2]
    goal = json.load(open(d / "meta.json"))["goal"]
    g = EV.goal_vector(goal, full9.FuncExec(X.Ep(t), goal))
    rows = [r for r in (json.loads(l) for l in open(f)) if r.get("kind") == "model" and r.get("pre")]
    wi = next((i for i, r in enumerate(rows) if X.kind(r.get("tok") or "") in ("entity_row", "typing", "commit", "answer")), None)
    if wi is None:
      continue
    W = rows[wi]["pre"]["summary"]
    nav = [r for r in rows[:wi] if "launcher" not in (r["pre"]["summary"].get("package") or "")
           and X.kind(r.get("tok") or "") == "navigation"]
    c["episodes"] += 1
    c["nav_before_work"] += len(nav)
    for j, r in enumerate(nav):
      ui = R.load_ui(d, r["pre"]["cid"])
      gr = R.Graph(None)
      for o in lines:
        if o.get("t", 0) >= r["pre"]["t"]:
          break
        gr.add(o)
      jg = JG.JointGraph(gr)
      w = EV.w_app(ui.package)
      here = JG.page_key(ui)
      best = None
      for p, ents in jg.into.items():
        if p[0] != ui.package or p == here:
          continue
        dui = EV.load_cap(raw, ents[-1][0], ents[-1][1])
        if dui is None:
          continue
        s = EV.rel(g, EV.page_tokens(dui), w)
        if best is None or s > best[0]:
          best = (s, p)
      if best is None or best[0] <= 0 or best[1][1] != W.get("activity"):
        continue
      starts = [p for p in jg.out if p[0] == ui.package and p[1] == ui.activity]
      live = {C.family_of_sel(x.sel) for x in U.enumerate_slots(ui)}
      paths = [pa for pa in jg.paths(ui, lambda p: p == best[1], max_hops=max_hops,
                                     safe=lambda s: JG.safe_control(s) or JG.entry_control(s), starts=starts)
               if pa and pa[0][0] in live]
      if paths:
        c["saveable"] += len(nav) - j
        c["episodes_routable"] += 1
        c["explore_hops"] += sum(1 for h in paths[0] if "explore" in h[3] and "agent" not in h[3])
        c["hops"] += len(paths[0])
        break
  n_model = sum(1 for f in glob.glob(str(raw / "*" / "p0" / "steps.jsonl")) for l in open(f) if '"kind": "model"' in l)
  print(f"[{arm}] model inferences {n_model}; episodes {c['episodes']}; navigation steps before the work page "
        f"{c['nav_before_work']}; saveable by goal-directed routing {c['saveable']} ({c['saveable'] / n_model:.1%} of "
        f"inferences) in {c['episodes_routable']} episodes; route hops {c['hops']} of which exploration-only "
        f"{c['explore_hops']}")


if __name__ == "__main__":
  for a in sys.argv[1:]:
    run(a)
