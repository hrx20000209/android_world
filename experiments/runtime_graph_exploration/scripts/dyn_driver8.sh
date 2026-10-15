#!/bin/bash
# GX5: exploration effects persist across episodes (fresh memory for this experiment, run order fixed).
cd /Users/huangrunxi/Projects/android_world/experiments/runtime_graph_exploration
export DYN_EFFECT_MEM=$PWD/graph/effect_memory_gx5.jsonl
run() { o=raw/dyn_$4/$1/$2/$3; [ -f $o/result.json ] && return
  DYN_VAR=$1 DYN_ARM=$2 python scripts/dyn_run.py --task $3 --seed $4 --out $o > logs/dyn$4_$1_$2_$3.log 2>&1
  echo "DYN8 s$4 $1 $2 $3 $(python -c "import json;r=json.load(open('$o/result.json'));print('succ',r.get('is_successful'),'steps',r.get('steps'),'calls',r.get('m_calls'),'inj',r.get('m_injected'),'stuck_ev',r.get('m_stuck_evidence'),'reused',r.get('m_effect_reused'),'rbf',r.get('m_rollback_fail'),r.get('error','')[:60])" 2>&1 | tail -1) restore=$(cat $o/restore.txt 2>/dev/null | tail -c 30)"; }
for seed in 32 33; do for t in MarkorCreateFolder MarkorCreateNote MarkorDeleteNote; do for v in V1 V0 V5; do run $v GX5 $t $seed; done; done; done
echo "DYN8 ALL DONE"
