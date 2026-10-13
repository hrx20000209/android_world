#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
while pgrep -f "seq_driver.py BASE,G,GX" >/dev/null; do sleep 30; done
python scripts/seq_driver.py GC > logs/seq_driver_gc.log 2>&1
echo "- $(date '+%m-%d %H:%M:%S') GC CHAIN DONE" >> reports/execution_log.md
