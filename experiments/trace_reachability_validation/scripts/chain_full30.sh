#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
until grep -q "FULL29X SINGLE PASS DONE" logs/full_driver_full29x.log; do sleep 30; done
while pgrep -f "full_run.py" > /dev/null || pgrep -f "full_driver.py FULL29X" > /dev/null; do sleep 10; done
echo "- $(date '+%m-%d %H:%M:%S') FULL30 single pass started (FULL29 + 可逆性门槛 0.7→0.8、去掉风险区：FULL29 中 P_rev<0.8 的探索回滚失败 33%，≥0.8 为 4%；回滚失败导致 requery 抵消了探索带来的函数步)" >> reports/execution_log.md
nohup python scripts/full_driver.py FULL30 > logs/full_driver_full30.log 2>&1 &
sleep 2
nohup bash scripts/stop_after_pass0_full30.sh > logs/stop_full30.log 2>&1 &
