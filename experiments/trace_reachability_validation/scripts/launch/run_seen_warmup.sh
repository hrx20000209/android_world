#!/bin/sh
# Seen-task warm-up on the LabServer vLLM (no on-device load; step counts only).
#   warm : SEEN templates at seeds 31, 32; FULL45 without graph execution (NO_EXEC), fixed exploration (WARM_EXPLORE)
#   warm2: templates still without a successful demo, seeds 33, 34, 40-call budget
# Output: raw/seenW1/FULL45+WARM/<task>/p0..p3, graph_snapshots/seenW1 (graph, store, function library).
cd "$(dirname "$0")/../.." || exit 1
export SEEN_APPS="Expense,Recipe,Markor,SimpleCalendar,SimpleSms,Contacts,SimpleDraw,Tasks,Notes,AudioRecorder,Retro,OsmAnd,System,Turn,Browser,Camera,Clock,Files,Vlc,SaveCopy,Sports,Open"
TAG=${1:-seenW1}
nohup sh -c "python scripts/seen_driver.py warm $TAG >> logs/${TAG}_driver.log 2>&1 && \
             python scripts/seen_driver.py warm2 $TAG >> logs/${TAG}_driver.log 2>&1" > /dev/null 2>&1 &
