#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/runtime_graph_study
while pgrep -f "rgs_driver.py --manifest configs/batch1" >/dev/null; do sleep 20; done
mkdir -p raw/old_markorcreate graphs/old_markorcreate
for v in v0 v1 v2 v3 v4; do mv raw/main/MarkorCreateNote/${v}_agent raw/old_markorcreate/ 2>/dev/null; done
mv graphs/main/MarkorCreateNote.explore.jsonl graphs/old_markorcreate/ 2>/dev/null
echo "- $(date '+%m-%d %H:%M:%S') MarkorCreateNote exploration arm rerun (value_selection rule)" >> reports/execution_log.md
python scripts/rgs_driver.py --manifest configs/batch1_manifest.json --raw raw/main --graphs graphs/main --only MarkorCreateNote >> logs/driver_batch1.log 2>&1
echo "- $(date '+%m-%d %H:%M:%S') batch2 start" >> reports/execution_log.md
python scripts/rgs_driver.py --manifest configs/batch2_manifest.json --raw raw/main --graphs graphs/main > logs/driver_batch2.log 2>&1
echo "- $(date '+%m-%d %H:%M:%S') CHAIN DONE" >> reports/execution_log.md
