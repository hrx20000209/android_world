#!/bin/bash
# Runtime-dynamics matrix: arms x {default, variation}; serialized on emulator-5554.
cd /Users/huangrunxi/Projects/android_world/experiments/runtime_graph_exploration
run() { # var arm task
  o=raw/dyn/$1/$2/$3; [ -f $o/result.json ] && return
  DYN_VAR=$1 DYN_ARM=$2 python scripts/dyn_run.py --task $3 --seed 32 --out $o > logs/dyn_$1_$2_$3.log 2>&1
  echo "DYN $1 $2 $3 $(python -c "import json;r=json.load(open('$o/result.json'));print('succ',r.get('is_successful'),'steps',r.get('steps'),'calls',r.get('m_calls'),'inj',r.get('m_injected'),'drop_e',r.get('m_fn_dropped_entry'),'drop_t',r.get('m_fn_dropped_transition'),'pfn',r.get('m_probes_fn'),'pfr',r.get('m_probes_frontier'),'rbf',r.get('m_rollback_fail'),r.get('error','')[:60])" 2>&1 | tail -1)"
}
for t in SimpleCalendarAddOneEvent SimpleCalendarAddOneEventTomorrow SimpleCalendarDeleteOneEvent SimpleCalendarEventsOnDate; do
  for v in V2 V0; do for a in BASE TRAJ GX; do run $v $a $t; done; done
done
for t in MarkorCreateFolder MarkorCreateNote MarkorDeleteNote; do
  for v in V1 V0; do for a in BASE TRAJ GX; do run $v $a $t; done; done
done
echo "DYN ALL DONE"
