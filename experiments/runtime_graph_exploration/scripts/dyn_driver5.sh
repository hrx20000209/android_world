#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/runtime_graph_exploration
run() { o=raw/dyn_$4/$1/$2/$3; [ -f $o/result.json ] && return
  DYN_VAR=$1 DYN_ARM=$2 python scripts/dyn_run.py --task $3 --seed $4 --out $o > logs/dyn$4_$1_$2_$3.log 2>&1
  echo "DYN5 s$4 $1 $2 $3 $(python -c "import json;r=json.load(open('$o/result.json'));print('succ',r.get('is_successful'),'steps',r.get('steps'),'calls',r.get('m_calls'),'inj',r.get('m_injected'),'stuck_ev',r.get('m_stuck_evidence'),'rbf',r.get('m_rollback_fail'),'pref',r.get('m_var_pref','')[-30:],r.get('error','')[:60])" 2>&1 | tail -1) restore=$(cat $o/restore.txt 2>/dev/null | tail -c 40)"; }
for seed in 32 33; do
  for t in MarkorEditNote MarkorChangeNoteContent MarkorAddNoteHeader; do for a in BASE TRAJ GX3; do run V4 $a $t $seed; done; run V0 BASE $t $seed; done
  for t in MarkorCreateFolder MarkorCreateNote MarkorDeleteNote; do for a in BASE TRAJ GX3; do run V5 $a $t $seed; done; done
done
echo "DYN5 ALL DONE"
