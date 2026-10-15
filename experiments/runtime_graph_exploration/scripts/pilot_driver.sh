#!/bin/bash
# Pair-inference pilot over the 20 pre-registered states; a state whose probe
# rollback fails is redone once without the failing probe (recorded).
cd /Users/huangrunxi/Projects/android_world/experiments/runtime_graph_exploration
for i in $(seq 0 19); do
  [ -f raw/pilot/s${i}_pair/pilot.json ] && grep -q '"resp"' raw/pilot/s${i}_pair/pilot.json && continue
  python scripts/oracle_pilot.py --state $i --out raw/pilot/s${i}_pair > logs/pilot_s${i}.log 2>&1
  if grep -q restore_failed raw/pilot/s${i}_pair/pilot.json 2>/dev/null; then
    bad=$(python -c "import json;d=json.load(open('raw/pilot/s${i}_pair/pilot.json'));print(';;'.join(p['want'] for p in d['probes'] if p.get('status')=='rollback_failed'))")
    python scripts/oracle_pilot.py --state $i --out raw/pilot/s${i}_pair_retry --skip "$bad" > logs/pilot_s${i}_retry.log 2>&1
  fi
  echo "PILOT $i done $(date +%H:%M:%S)"
done
echo "PILOT ALL DONE"
