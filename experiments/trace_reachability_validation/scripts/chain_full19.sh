#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
until grep -q "FULL18 SINGLE PASS DONE" logs/full_driver_full18.log; do sleep 30; done
while pgrep -f "full_run.py" > /dev/null || pgrep -f "full_driver.py FULL18" > /dev/null; do sleep 10; done
echo "- $(date '+%m-%d %H:%M:%S') FULL19 single pass started (FULL18 + 复用只针对离开当前页面的重复转移，页内重复（翻月/数字键/卡住）交还模型)" >> reports/execution_log.md
nohup python scripts/full_driver.py FULL19 > logs/full_driver_full19.log 2>&1 &
sleep 2
nohup bash scripts/stop_after_pass0_full19.sh > logs/stop_full19.log 2>&1 &
