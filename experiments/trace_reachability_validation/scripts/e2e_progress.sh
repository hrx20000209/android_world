#!/bin/sh
# wait until N tasks are paired across all arms of the e2e10 run, then print the report table
N=$1
cd "$(dirname "$0")/.."
until [ "$(python3 -c "
import json
rows={}
try:
  for l in open('reports/e2e10_runs.jsonl'):
    r=json.loads(l)
    if r.get('error'): rows.pop((r['task'],r['arm']),None)
    else: rows[(r['task'],r['arm'])]=r
except FileNotFoundError: pass
A=['BASE','BASE+S','FULL41','FULL42']
print(len({t for t,_ in rows if all((t,a) in rows for a in A)}))
")" -ge "$N" ] || grep -q "E2E DRIVER DONE" logs/e2e10_driver.log; do sleep 30; done
PYTHONPATH=/Users/huangrunxi/Projects/android_world python -c "
import sys; sys.argv=['x','BASE,BASE+S,FULL41,FULL42','e2e10']; sys.path.insert(0,'scripts')
import e2e_driver as E
rows=E.load_rows(); tasks=sorted({t for t,_ in rows})
n=sum(all((t,a) in rows for a in E.ARMS) for t in tasks)
E.REPORT=E.ROOT/'reports'/'e2e10_progress_tmp.md'; E.REPORT.write_text('')
E.report(tasks, rows, n); print(E.REPORT.read_text())
" 2>&1 | grep -v "Warn\|genai\|google\|^$"
