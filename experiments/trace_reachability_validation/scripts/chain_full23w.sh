#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
until grep -q "FULL23 SINGLE PASS DONE" logs/full_driver_full23.log; do sleep 30; done
while pgrep -f "full_run.py" > /dev/null || pgrep -f "full_driver.py FULL23$" > /dev/null; do sleep 10; done
echo "- $(date '+%m-%d %H:%M:%S') FULL23W single pass started (FULL23 + 暖启动：edge_use 由 FULL21/FULL21R 成功任务预置，按任务留一；图与探索从零开始)" >> reports/execution_log.md
nohup python scripts/full_driver.py FULL23W > logs/full_driver_full23w.log 2>&1 &
sleep 2
nohup bash scripts/stop_after_pass0_full23w.sh > logs/stop_full23w.log 2>&1 &
