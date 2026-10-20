#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
until grep -q "FULL30 SINGLE PASS DONE" logs/full_driver_full30.log; do sleep 30; done
while pgrep -f "full_run.py" > /dev/null || pgrep -f "full_driver.py FULL30" > /dev/null; do sleep 10; done
echo "- $(date '+%m-%d %H:%M:%S') FULL31 single pass started (FULL30 去掉 repeat 复用：FULL23–30 所有 repeat 复用（每轮 14–27 次）几乎全部发生在失败 episode 的循环里，省下的是失败任务里兜圈子的推理)" >> reports/execution_log.md
nohup python scripts/full_driver.py FULL31 > logs/full_driver_full31.log 2>&1 &
sleep 2
nohup bash scripts/stop_after_pass0_full31.sh > logs/stop_full31.log 2>&1 &
