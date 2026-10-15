#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/runtime_graph_exploration
run() { o=raw/dyn_s33/$1/$2/$3; [ -f $o/result.json ] && return
  DYN_VAR=$1 DYN_ARM=$2 python scripts/dyn_run.py --task $3 --seed 33 --out $o > logs/dyn33_$1_$2_$3.log 2>&1
  echo "DYN33 $1 $2 $3 $(python -c "import json;r=json.load(open('$o/result.json'));print('succ',r.get('is_successful'),'steps',r.get('steps'),'calls',r.get('m_calls'),'inj',r.get('m_injected'),'stuck_ev',r.get('m_stuck_evidence'),'rbf',r.get('m_rollback_fail'),r.get('error','')[:60])" 2>&1 | tail -1)"; }
for t in MarkorCreateFolder MarkorCreateNote MarkorDeleteNote; do for a in BASE TRAJ GX3; do run V1 $a $t; done; done
for t in MarkorCreateFolder MarkorCreateNote MarkorDeleteNote; do run V0 BASE $t; done
echo "DYN4 ALL DONE"
