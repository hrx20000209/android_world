#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation
python scripts/tr_driver.py --manifest configs/phase2_manifest.json --raw raw/phase2 --arms TG1C > logs/driver_tg1c.log 2>&1
python scripts/pf_driver.py --manifest configs/phase3_manifest.json --raw raw/phase3 --arms BASE3,PFVA2 > logs/driver_rep.log 2>&1
echo "- $(date '+%m-%d %H:%M:%S') MEET CHAIN DONE" >> reports/execution_log.md
