#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# V3 goal:
# - Match the 116-task baseline more fairly by allowing long-but-solvable tasks.
# - Keep fast_provider latency logging.
# - Keep exploration lightweight unless the task gate finds safe query/navigation evidence.
export EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/results/lb_mcts_final_116task_fast_latency_v3_improved}"
export A11Y_METHOD="${A11Y_METHOD:-fast_provider}"
export MAX_STEPS="${MAX_STEPS:-21}"
export MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-300}"

export ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_BUDGET="${ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_BUDGET:-8}"
export ANDROID_WORLD_LIGHT_EXPLORE_MIN_ATTEMPTS_PER_STEP="${ANDROID_WORLD_LIGHT_EXPLORE_MIN_ATTEMPTS_PER_STEP:-8}"
export ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_DEPTH="${ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_DEPTH:-2}"
export ANDROID_WORLD_LB_MCTS_MIN_BUDGET_MS="${ANDROID_WORLD_LB_MCTS_MIN_BUDGET_MS:-1500}"
export ANDROID_WORLD_LB_MCTS_MAX_BUDGET_MS="${ANDROID_WORLD_LB_MCTS_MAX_BUDGET_MS:-6000}"
export ANDROID_WORLD_LB_MCTS_MAX_ROLLOUTS_PER_STEP="${ANDROID_WORLD_LB_MCTS_MAX_ROLLOUTS_PER_STEP:-8}"
export ANDROID_WORLD_LB_MCTS_LAMBDA_LATENCY="${ANDROID_WORLD_LB_MCTS_LAMBDA_LATENCY:-0.8}"

echo "[fast-latency-v3] exp_root=${EXP_ROOT}"
echo "[fast-latency-v3] a11y=${A11Y_METHOD}"
echo "[fast-latency-v3] max_steps=${MAX_STEPS}"
echo "[fast-latency-v3] branch_budget=${ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_BUDGET}"
echo "[fast-latency-v3] min_attempts=${ANDROID_WORLD_LIGHT_EXPLORE_MIN_ATTEMPTS_PER_STEP}"
echo "[fast-latency-v3] lb_mcts_budget=${ANDROID_WORLD_LB_MCTS_MIN_BUDGET_MS}-${ANDROID_WORLD_LB_MCTS_MAX_BUDGET_MS}ms"

exec bash "${ROOT_DIR}/scripts/run_lb_mcts_116task_stable_foreground.sh"
