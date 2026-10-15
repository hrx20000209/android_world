#!/bin/bash
# start FULL13 once FULL12's single pass is done and no episode process remains
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
until grep -q "FULL12 SINGLE PASS DONE" logs/full_driver_full12.log; do sleep 30; done
while pgrep -f "full_run.py" > /dev/null || pgrep -f "full_driver.py FULL12" > /dev/null; do sleep 10; done
echo "- $(date '+%m-%d %H:%M:%S') FULL13 single pass started (FULL12 + restorability-guaranteed exploration: probe only while the path since launch is replayable)" >> reports/execution_log.md
nohup python scripts/full_driver.py FULL13 > logs/full_driver_full13.log 2>&1 &
sleep 2
nohup bash scripts/stop_after_pass0_full13.sh > logs/stop_full13.log 2>&1 &
