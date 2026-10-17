"""Origin x outcome matrix of the joint graph (agent nodes + exploration nodes), per arm.

For every model action (page, control family) -> landing page, look at the graph as it was just before the
action (observations with t < action time, any task of the run, including this episode):
  rows    : who had observed this edge  -- none / agent only / explore only / both
  columns : the graph's predicted landing page was right (activity+layout) / right activity only / wrong
Also: of the pages the model landed on, which were already in the graph and who discovered them first; and how
graph executions (routes, restorations) used edges of each origin."""
import json, glob, sys, bisect
from collections import Counter, defaultdict


def pk(s):
  return (s.get("package"), s.get("activity"), s.get("layout_sig"))


def fam_of(o):
  sid = o["slot_id"]
  if sid.startswith("E|"):
    return sid
  return "E|" + sid


for arm in sys.argv[1:]:
  edges = defaultdict(list)       # (src page, fam) -> [(t, origin, dst page)]
  first_seen = {}                 # page -> (t, origin) as a destination
  for l in open(f"graph_snapshots/full/{arm}/graph.jsonl"):
    o = json.loads(l)
    if not o.get("dst"):
      continue
    org = "explore" if o.get("origin") == "explore" else "agent"
    edges[(pk(o["src"]), fam_of(o))].append((o.get("t", 0), org, pk(o["dst"])))
    dp = pk(o["dst"])
    if dp not in first_seen or o.get("t", 0) < first_seen[dp][0]:
      first_seen[dp] = (o.get("t", 0), org)
  for v in edges.values():
    v.sort()
  M = Counter(); land = Counter(); usage = Counter()
  for f in glob.glob(f"raw/full/{arm}/*/p0/steps.jsonl"):
    rows = [json.loads(l) for l in open(f)]
    for i, r in enumerate(rows):
      if r.get("kind") in ("function_route",):
        for o in r.get("origins", []):
          usage["route_hop_" + o] += 1
      if r.get("kind") != "model" or not r.get("tok") or not r.get("pre"):
        continue
      nxt = next((x for x in rows[i + 1:] if x.get("pre")), None)
      if nxt is None:
        continue
      t = r["pre"].get("t", 0)
      src, dst = pk(r["pre"]["summary"]), pk(nxt["pre"]["summary"])
      if src == dst:
        continue                                   # in-page actions (typing, toggles) are not navigation
      obs = edges.get((src, r["tok"] if r["tok"].startswith("E|") else "E|" + r["tok"]), [])
      obs = obs[:bisect.bisect_left([x[0] for x in obs], t)]
      origins = {x[1] for x in obs}
      row = "none" if not origins else "both" if len(origins) == 2 else origins.pop() + "_only"
      if obs:
        pred = Counter(x[2] for x in obs).most_common(1)[0][0]
        col = "exact" if pred == dst else "activity" if pred[1] == dst[1] else "wrong"
      else:
        col = "-"
      M[row, col] += 1
      fs = first_seen.get(dst)
      land["already_in_graph" if fs and fs[0] < t else "new"] += 1
      if fs and fs[0] < t:
        land["first_by_" + fs[1]] += 1
  rows_ = ["none", "agent_only", "explore_only", "both"]
  print(f"[{arm}] model navigation actions: {sum(M.values())}")
  print(f"    {'edge known by':14s} {'n':>5s} {'exact':>6s} {'activity':>9s} {'wrong':>6s}  accuracy")
  for r_ in rows_:
    n = sum(M[r_, c] for c in ("exact", "activity", "wrong", "-"))
    good = M[r_, "exact"] + M[r_, "activity"]
    known = n - M[r_, "-"]
    print(f"    {r_:14s} {n:5d} {M[r_, 'exact']:6d} {M[r_, 'activity']:9d} {M[r_, 'wrong']:6d}  "
          + (f"{good / known:.0%}" if known else "-"))
  tot = land["already_in_graph"] + land["new"]
  print(f"    landing pages already in graph: {land['already_in_graph']}/{tot} "
        f"(first discovered by exploration {land['first_by_explore']}, by agent {land['first_by_agent']})")
  if usage:
    print("    graph-execution hops by origin:", dict(usage))
