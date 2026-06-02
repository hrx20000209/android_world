#!/usr/bin/env bash
set -euo pipefail
: "${RUN_GROUP_ID:?Set RUN_GROUP_ID to the same value used on machine1}"
MACHINE_ID="${MACHINE_ID:-2}"
python scripts/run_strategy_30task_split.py \
  --run_group_id "${RUN_GROUP_ID}" \
  --machine_id "${MACHINE_ID}" \
  --task_shard_id ALL \
  --max_cases 30 \
  --variants S2_DFS_BUDGET12,S4_MCTS_BUDGET12
