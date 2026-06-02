#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Usage:
#   bash scripts/run_single_tpes_exploration_task.sh NotesRecipeIngredientCount
#   TASK=SimpleCalendarEventsOnDate DEPTH=4 BRANCH_BUDGET=5 bash scripts/run_single_tpes_exploration_task.sh
#
# This uses the broad TPES diagnostic configuration:
# - every reasoning step runs post-planning exploration
# - branch depth is >2 by default
# - launcher/low-relevance filters are disabled so the report shows what is actually explored
# - transaction-safe rollback remains enabled to avoid committing speculative side effects

TASK="${TASK:-NotesRecipeIngredientCount}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  TASK="$1"
  shift
fi

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-results/tpes_single_task}"
BRANCH_BUDGET="${BRANCH_BUDGET:-4}"
DEPTH="${DEPTH:-3}"
STRATEGY="${STRATEGY:-dfs}"
IMAGE_WIDTH="${IMAGE_WIDTH:-20%}"
MAX_CASES="${MAX_CASES:-12}"
REPORT_MAX_BRANCHES="${REPORT_MAX_BRANCHES:-8}"
REPORT_MAX_CANDIDATES="${REPORT_MAX_CANDIDATES:-10}"

mkdir -p "$EXPERIMENT_ROOT"

python scripts/run_exploration_experiment_report.py \
  --run \
  --agent_name=explore_agent_gelab \
  --suite_family=android_world \
  --tasks="${TASK}" \
  --n_task_combinations=1 \
  --fixed_task_seed \
  --image_downsample_scale=1.0 \
  --a11y_method=uiautomator \
  --experiment_root="${EXPERIMENT_ROOT}" \
  --baseline_table=results/4B_2.txt \
  --explore_force_every_step \
  --no-explore_diagnostic_full \
  --explore_fast_mode \
  --explore_transaction_safe \
  --explore_max_runs=10000 \
  --explore_max_step=10000 \
  --explore_branch_budget="${BRANCH_BUDGET}" \
  --explore_branch_depth="${DEPTH}" \
  --explore_back_limit=4 \
  --explore_replay_max_actions=6 \
  --no-explore_planned_only \
  --explore_fallback_safe_candidates \
  --no-explore_safe_click_only \
  --no-explore_skip_launcher \
  --no-explore_filter_launcher_relevance \
  --no-explore_skip_destructive_goals \
  --explore_strategy="${STRATEGY}" \
  --max_cases="${MAX_CASES}" \
  "$@"

RUN_DIR="$(ls -td "${EXPERIMENT_ROOT}"/run_* | head -n 1)"
REPORT_PATH="$(python scripts/generate_single_tpes_trace_report.py \
  --run_dir "${RUN_DIR}" \
  --image_width "${IMAGE_WIDTH}" \
  --max_branches "${REPORT_MAX_BRANCHES}" \
  --max_candidates "${REPORT_MAX_CANDIDATES}")"

echo "RUN_DIR=${RUN_DIR}"
echo "REPORT=${REPORT_PATH}"
