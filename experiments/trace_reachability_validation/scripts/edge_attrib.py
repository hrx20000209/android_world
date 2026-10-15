"""Who discovered the graph edges that graph executions used? For every form/function entry taken over a graph
edge, find the earliest observation of that edge (src struct + control family) in graph.jsonl (append order =
time) that predates the use, and report its origin (explore / agent)."""
import json, glob, sys
from collections import Counter
arm = sys.argv[1]
first = {}
for i, l in enumerate(open(f"graph_snapshots/full/{arm}/graph.jsonl")):
  o = json.loads(l)
  fam = o["slot_id"] if o["slot_id"].startswith("E|") else "E|" + o["slot_id"]
  key = (o["src"]["struct_sig"], fam)
  if key not in first:
    first[key] = (o["origin"], o.get("t", 0), o["instance"])
c = Counter(); rows = []
for f in sorted(glob.glob(f"raw/full/{arm}/*/p0/steps.jsonl")):
  t = f.split("/")[3]
  res = json.load(open(f.replace("steps.jsonl", "result.json"))) if glob.glob(f.replace("steps.jsonl", "result.json")) else {}
  for l in open(f):
    s = json.loads(l)
    if s.get("kind") not in ("form_entry", "function_entry"):
      continue
    key = (s["pre"]["summary"]["struct_sig"], s["tok"])
    org = first.get(key, ("?", 0, ""))
    c[(s["kind"], org[0])] += 1
    rows.append((t, s["kind"], s["tok"][-30:], "first seen by " + org[0], org[2], "succ" if res.get("is_successful") == 1.0 else "fail"))
for r in rows:
  print(" ", *r)
print(dict(c))
