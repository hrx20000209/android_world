#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
while pgrep -f "full_driver.py$" >/dev/null || pgrep -f "full_driver.py BASE,FULL" >/dev/null; do sleep 60; done
python scripts/full_driver.py FULLNE > logs/full_driver_ne.log 2>&1
