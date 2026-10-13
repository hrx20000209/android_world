#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
while pgrep -f "pf_driver.py" >/dev/null; do sleep 20; done
python scripts/pf_driver.py --manifest configs/phase3_manifest.json --raw raw/phase3 --arms BASE2 > logs/driver_phase3_base2.log 2>&1
