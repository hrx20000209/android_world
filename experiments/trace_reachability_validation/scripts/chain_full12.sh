#!/bin/bash
# start FULL12 once FULL11's single pass is done and no episode process remains
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
until grep -q "FULL11 SINGLE PASS DONE" logs/full_driver_full11.log; do sleep 30; done
while pgrep -f "full_run.py" > /dev/null || pgrep -f "full_driver.py FULL11" > /dev/null; do sleep 10; done
echo "- $(date '+%m-%d %H:%M:%S') FULL12 single pass started (FULL11 + destination-grounded function entry over graph edges)" >> reports/execution_log.md
nohup python scripts/full_driver.py FULL12 > logs/full_driver_full12.log 2>&1 &
sleep 2
nohup bash scripts/stop_after_pass0_full12.sh > logs/stop_full12.log 2>&1 &
