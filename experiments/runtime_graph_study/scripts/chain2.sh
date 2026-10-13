#!/bin/bash
cd /Users/huangrunxi/Projects/android_world/experiments/runtime_graph_study
while pgrep -f "chain_after_batch1.sh" >/dev/null; do sleep 30; done
mkdir -p raw/old_markordelete graphs/old_markordelete
for v in v0 v1 v2 v3 v4; do mv raw/main/MarkorDeleteNewestNote/${v}_agent raw/old_markordelete/ 2>/dev/null; done
mv graphs/main/MarkorDeleteNewestNote.explore.jsonl graphs/old_markordelete/ 2>/dev/null
echo "- $(date '+%m-%d %H:%M:%S') MarkorDeleteNewestNote exploration arm rerun (tab reselect fix)" >> reports/execution_log.md
python scripts/rgs_driver.py --manifest configs/batch1_manifest.json --raw raw/main --graphs graphs/main --only MarkorDeleteNewestNote >> logs/driver_batch1.log 2>&1
echo "- $(date '+%m-%d %H:%M:%S') CHAIN2 DONE" >> reports/execution_log.md
