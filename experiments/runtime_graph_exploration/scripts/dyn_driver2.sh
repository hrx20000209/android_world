#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/runtime_graph_exploration
until grep -q "DYN ALL DONE" logs/dyn_driver.log; do sleep 20; done
run() { o=raw/dyn/$1/$2/$3; [ -f $o/result.json ] && return
  DYN_VAR=$1 DYN_ARM=$2 python scripts/dyn_run.py --task $3 --seed 32 --out $o > logs/dyn_$1_$2_$3.log 2>&1
  echo "DYN $1 $2 $3 $(python -c "import json;r=json.load(open('$o/result.json'));print('succ',r.get('is_successful'),'steps',r.get('steps'),'calls',r.get('m_calls'),'inj',r.get('m_injected'),'stuck_ev',r.get('m_stuck_evidence'),'pstuck',r.get('m_probes_stuck'),'rbf',r.get('m_rollback_fail'),r.get('error','')[:60])" 2>&1 | tail -1)"; }
for t in MarkorCreateFolder MarkorCreateNote MarkorDeleteNote; do run V1 GX2 $t; done
for t in MarkorCreateFolder MarkorCreateNote MarkorDeleteNote; do run V0 GX2 $t; done
echo "DYN2 ALL DONE"
