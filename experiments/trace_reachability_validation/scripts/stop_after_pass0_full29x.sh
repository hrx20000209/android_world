#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
until grep -q "FULLRUN PASS 0 DONE" logs/full_driver_full29x.log; do sleep 15; done
pkill -f "full_driver.py FULL29X"
until ! pgrep -f "full_run.py" >/dev/null; do sleep 3; done
for d in raw/full/FULL29X/*/p1; do [ -d "$d" ] && mv "$d" "${d}_stopped_single_pass"; done
echo "- $(date '+%m-%d %H:%M') FULL29X 只跑一轮（seed 30），第 1 轮结束后停止。" >> reports/execution_log.md
echo "FULL29X SINGLE PASS DONE" >> logs/full_driver_full29x.log
