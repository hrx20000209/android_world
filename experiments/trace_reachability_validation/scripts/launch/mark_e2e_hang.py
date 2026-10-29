"""Mark e2e episodes hit by an Orin hang (a decision > 30 s, or a stream error) as errors, so that the next
run_orin_e2e.sh with the same ARMS/TAG re-runs them. Backs the runs log up under aborted/ first.
usage: python scripts/launch/mark_e2e_hang.py TAG ARM"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
tag, arm = sys.argv[1], sys.argv[2]
log = ROOT / "reports" / f"{tag}_runs.jsonl"
shutil.copy(log, ROOT / "aborted" / f"{tag}_runs.before_mark_{int(time.time())}.jsonl")
rows = {}
for l in open(log):
  r = json.loads(l)
  if r["arm"] == arm:
    rows[r["task"]] = r
marked = []
for t, r in rows.items():
  if r.get("error"):
    continue
  d = ROOT / "raw" / tag / arm / t / "p0"
  lat = [json.loads(l).get("model_latency_s") or 0 for l in open(d / "steps.jsonl") if json.loads(l)["kind"] == "model"]
  n, se = sum(x > 30 for x in lat), os.path.exists(d / "stream_error.jsonl")
  if n or se:
    marked.append((t, n, se))
with open(log, "a") as f:
  for t, n, se in marked:
    f.write(json.dumps({"task": t, "arm": arm, "error": f"orin hang: {n} decisions > 30 s, stream_error={se}"}) + "\n")
print("marked", marked)
