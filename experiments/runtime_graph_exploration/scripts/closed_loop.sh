#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/runtime_graph_exploration
run() { # state cond skip
  [ -f raw/closed/s$1_$2/result.json ] && return
  python scripts/oracle_pilot.py --state $1 --cond $2 --skip "$3" --out raw/closed/s$1_$2 > logs/closed_s$1_$2.log 2>&1
  echo "CL $1 $2 $(grep -o '"is_successful": [0-9.]*' raw/closed/s$1_$2/result.json) $(date +%H:%M:%S)"
}
for c in M0 ME; do run 3 $c ""; run 7 $c "action_preview"; run 12 $c ""; run 11 $c ""; done
echo "CL ALL DONE"
