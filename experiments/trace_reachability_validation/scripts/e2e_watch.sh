#!/bin/sh
# Return when one more task is paired across all arms, when no episode finished for 12 min (stall), or when done.
cd "$(dirname "$0")/.."
count() { python3 -c "
import json
rows={}
try:
  for l in open('reports/${TAG:-e2e10}_runs.jsonl'):
    r=json.loads(l)
    if r.get('error'): rows.pop((r['task'],r['arm']),None)
    else: rows[(r['task'],r['arm'])]=r
except FileNotFoundError: pass
A=__import__('os').environ.get('ARMS','BASE,BASE+S,FULL42').split(',')
print(len({t for t,_ in rows if all((t,a) in rows for a in A)}))"; }
n0=$(count)
lines() { [ -f reports/${TAG:-e2e10}_runs.jsonl ] && wc -l < reports/${TAG:-e2e10}_runs.jsonl || echo 0; }
l0=$(lines); t0=$(date +%s)
while :; do
  sleep 30
  if grep -q "E2E DRIVER DONE" logs/${TAG:-e2e10}_driver.log; then echo "DONE"; break; fi
  if [ "$(count)" -gt "$n0" ]; then echo "PAIRED $(count)"; break; fi
  l=$(lines); if [ "$l" != "$l0" ]; then l0=$l; t0=$(date +%s); fi
  if [ $(( $(date +%s) - t0 )) -gt 720 ]; then echo "STALL: no episode finished for 12 min"; tail -3 logs/${TAG:-e2e10}_driver.log; break; fi
done
grep "E2E " logs/${TAG:-e2e10}_driver.log | tail -${TAILN:-4} | cut -c1-200
