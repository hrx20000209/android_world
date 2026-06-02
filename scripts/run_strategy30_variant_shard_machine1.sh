#!/usr/bin/env bash
set -euo pipefail
: "${RUN_GROUP_ID:?Set RUN_GROUP_ID, for example RUN_GROUP_ID=sensys_strategy30_20260602}"
MACHINE_ID="${MACHINE_ID:-1}"
python scripts/run_strategy_30task_split.py \
  --run_group_id "${RUN_GROUP_ID}" \
  --machine_id "${MACHINE_ID}" \
  --task_shard_id ALL \
  --max_cases 30 \
  --variants B0_BASELINE_RERUN,S1_BFS_BUDGET12,S3_BEAM_BUDGET12,S5_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12
