#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
until grep -q "FULL32 SINGLE PASS DONE" logs/full_driver_full32.log; do sleep 30; done
while pgrep -f "full_run.py" > /dev/null || pgrep -f "full_driver.py FULL32" > /dev/null; do sleep 10; done
echo "- $(date '+%m-%d %H:%M:%S') FULL33 single pass started (FULL32 + 统一图复用：启动、确认、函数、表单、repeat、宏都作为当前页面的出边，以本轮运行时图上的边为触发依据（完全在线：不用 profiling 图，函数库只从本轮已完成且模型自报完成的 episode 抽取）、执行后写回图、被模型撤销即记误用并刹车 + 增强探索（价值信息项：相关性未知的控件（无文本且无图证据、或参数不能判定的实体行）排在相关性已知为 0 的控件之前，λ=κ(1−0.7)；新建类入口 New/Add/+ 在有待填记录时允许探索，门槛 0.7）+ 目标解析修复：冒号列表只按逗号拆分（'Kale and Quinoa Salad' 不再被拆成两条）；时间里的冒号和问句不再当作列表)" >> reports/execution_log.md
nohup python scripts/full_driver.py FULL33 > logs/full_driver_full33.log 2>&1 &
sleep 2
nohup bash scripts/stop_after_pass0_full33.sh > logs/stop_full33.log 2>&1 &
