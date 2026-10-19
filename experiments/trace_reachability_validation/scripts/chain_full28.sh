#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
until grep -q "FULL23R SINGLE PASS DONE" logs/full_driver_full23r.log; do sleep 30; done
while pgrep -f "full_run.py" > /dev/null || pgrep -f "full_driver.py FULL23R" > /dev/null; do sleep 10; done
echo "- $(date '+%m-%d %H:%M:%S') FULL28 single pass started (FULL23 + 表单函数 v2：用 a11y hint 判断空框、字段同义词、单字段记录、number 仅填电话类框、file: 仅新建类任务填 name 框；离线：模型自行输入的 96 个参数值中 69 个可由函数代填）" >> reports/execution_log.md
nohup python scripts/full_driver.py FULL28 > logs/full_driver_full28.log 2>&1 &
sleep 2
nohup bash scripts/stop_after_pass0_full28.sh > logs/stop_full28.log 2>&1 &
