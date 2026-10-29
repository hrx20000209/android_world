#!/bin/sh
# Seen/unseen evaluation on the LabServer vLLM, all 116 templates at seed 30, arms interleaved per task.
#   FULL45        cold graph (same-session control)
#   FULL45+SEENR  graph + function library copied from the warm-up, same-template functions + commit, trajectory replay
# usage: run_seen_eval_labserver.sh [eval_tag=seenE2] [warm_tag=seenW1] [arms=FULL45,FULL45+SEENR]
cd "$(dirname "$0")/../.." || exit 1
export SEEN_APPS="Expense,Recipe,Markor,SimpleCalendar,SimpleSms,Contacts,SimpleDraw,Tasks,Notes,AudioRecorder,Retro,OsmAnd,System,Turn,Browser,Camera,Clock,Files,Vlc,SaveCopy,Sports,Open"
TAG=${1:-seenE2}; WARM=${2:-seenW1}
export SEEN_ARMS=${3:-FULL45,FULL45+SEENR} SEEN_WARM=$WARM
nohup sh -c "python scripts/seen_driver.py eval $TAG $WARM >> logs/${TAG}_driver.log 2>&1" > /dev/null 2>&1 &
