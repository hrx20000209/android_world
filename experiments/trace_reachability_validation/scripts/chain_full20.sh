#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
until grep -q "FULL19 SINGLE PASS DONE" logs/full_driver_full19.log; do sleep 30; done
while pgrep -f "full_run.py" > /dev/null || pgrep -f "full_driver.py FULL19" > /dev/null; do sleep 10; done
echo "- $(date '+%m-%d %H:%M:%S') FULL20 single pass started (FULL19 + FX 只探 P≥0.7（去掉 0.5–0.7 兜底档）、每窗口 FX≤2、其余由重新观测补足：FX Back 失败 31% vs RV 3%，重新推理 46/48 发生在 Back 失败之后)" >> reports/execution_log.md
nohup python scripts/full_driver.py FULL20 > logs/full_driver_full20.log 2>&1 &
sleep 2
nohup bash scripts/stop_after_pass0_full20.sh > logs/stop_full20.log 2>&1 &
