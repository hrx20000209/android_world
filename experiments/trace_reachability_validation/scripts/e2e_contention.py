"""Mark episodes whose decisions queued behind another client on the shared Orin vLLM (time to first token > 15 s on
>= 2 decisions; our own prefill is ~5 s): append an error row so the next driver start re-runs them."""
import json, sys
from pathlib import Path
tag = sys.argv[1]
ROOT = Path(__file__).resolve().parents[1]
log = ROOT / "reports" / f"{tag}_runs.jsonl"
rows = {}
for l in open(log):
  r = json.loads(l)
  rows[(r["task"], r["arm"])] = r
marked = []
for (t, a), r in rows.items():
  if r.get("error"):
    continue
  g = ROOT / "raw" / tag / a / t / "p0" / "gen.jsonl"
  if not g.exists():
    continue
  slow = [json.loads(l).get("ttft_s", 0) for l in open(g)]
  n = sum(1 for x in slow if x and x > 15)
  if n >= 2:
    marked.append((t, a, n))
with open(log, "a") as f:
  for t, a, n in marked:
    f.write(json.dumps({"task": t, "arm": a, "error": f"orin contention: {n} decisions with ttft > 15 s"}) + "\n")
print("marked", marked)
