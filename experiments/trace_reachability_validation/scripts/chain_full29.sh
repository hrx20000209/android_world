#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
until grep -q "FULL28R SINGLE PASS DONE" logs/full_driver_full28r.log; do sleep 30; done
while pgrep -f "full_run.py" > /dev/null || pgrep -f "full_driver.py FULL28R" > /dev/null; do sleep 10; done
echo "- $(date '+%m-%d %H:%M:%S') FULL29 single pass started (FULL28 + 表单 v3：记录的每个字段只填一次；首字段只填 title/name 框。FULL28 两轮 Recipe 两题都把标题填进 Source 框而失败)" >> reports/execution_log.md
nohup python scripts/full_driver.py FULL29 > logs/full_driver_full29.log 2>&1 &
sleep 2
nohup bash scripts/stop_after_pass0_full29.sh > logs/stop_full29.log 2>&1 &
