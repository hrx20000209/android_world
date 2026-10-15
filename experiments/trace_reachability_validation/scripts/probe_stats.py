import json, glob, sys
arm = sys.argv[1]
n = f = tr = 0
for p in glob.glob(f"raw/full/{arm}/*/p0/probes.jsonl"):
  for l in open(p):
    r = json.loads(l); n += 1; f += not r["rollback_ok"]
for p in glob.glob(f"raw/full/{arm}/*/p0/windows.jsonl"):
  for l in open(p):
    tr += json.loads(l).get("stop") == "transient_state"
print(f"probes {n}, rollback failures {f} ({f / max(1, n):.0%}), windows skipped as transient {tr}")
