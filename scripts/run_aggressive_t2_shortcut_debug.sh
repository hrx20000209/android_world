#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/huangrunxi/Projects/android_world"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
RESULT_ROOT="${REPO_ROOT}/results/aggressive_t2_shortcut"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/shortcut_integration_debug_$(date +%Y%m%dT%H%M%S)}"

TASKS="NotesRecipeIngredientCount,NotesTodoItemCount,BrowserMaze,SportsTrackerActivityDuration,SportsTrackerTotalDistanceForCategoryOverInterval,SimpleCalendarNextMeetingWithPerson,TasksHighPriorityTasksDueOnDate,TasksDueNextWeek"

mkdir -p "$RUN_ROOT"
echo "[debug] root=${RUN_ROOT}"
echo "[debug] tasks=${TASKS}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_aggressive_t2_unit_tests.py" --output_dir "$RUN_ROOT"

run_variant() {
  local label="$1"
  local t2_mode="$2"
  local threshold="$3"
  local top1="$4"
  local margin="$5"
  local variant_root="${RUN_ROOT}/${label}"
  mkdir -p "$variant_root"
  echo "[debug] starting ${label}"
  (
    cd "$REPO_ROOT"
    "$PYTHON_BIN" "$REPO_ROOT/scripts/run_exploration_experiment_report.py" \
      --run \
      --agent_name=explore_agent_gelab \
      --suite_family=android_world \
      --tasks="$TASKS" \
      --n_task_combinations=1 \
      --task_random_seed=30 \
      --fixed_task_seed \
      --image_downsample_scale=1.0 \
      --a11y_method=uiautomator \
      --baseline_table=results/4B_2.txt \
      --experiment_root="$variant_root" \
      --max_cases=12 \
      --explore_max_runs=10000 \
      --explore_max_step=10000 \
      --explore_branch_budget=1 \
      --explore_branch_depth=2 \
      --explore_back_limit=4 \
      --explore_planned_only \
      --no-explore_fallback_safe_candidates \
      --no-explore_safe_click_only \
      --explore_skip_launcher \
      --explore_filter_launcher_relevance \
      --explore_skip_destructive_goals \
      --explore_fast_mode \
      --explore_fast_state \
      --explore_transaction_safe \
      --explore_action_settle_s=0.25 \
      --explore_hint_policy=strict \
      --explore_search_policy=task_gate \
      --explore_search_strategy=best_first \
      --explore_rollback_policy=improved \
      --explore_fixed_framework \
      --explore_answer_extractors \
      --explore_slot_complete \
      --explore_slot_policy_switcher \
      --explore_no_action_hint \
      --explore_lightweight_a11y_trace \
      --explore_trace_a11y_limit=32 \
      --explore_variant="$label" \
      --t2_mode="$t2_mode" \
      --t2_state_mode=FULL_A11Y_CONTROL \
      --t2_allow_safe_search_input \
      --t2_confidence_threshold="$threshold" \
      --t2_min_top1_score="$top1" \
      --t2_min_score_margin="$margin" \
      2>&1 | tee "${variant_root}/driver.log"
  )
  echo "[debug] finished ${label}"
}

run_variant "V16_SHADOW_RELAXED_SEARCH_ONLY" "shadow" "0.65" "0.0" "0.0"
run_variant "V18_ACTIVE_SAFE_SEARCH_ONLY" "active_safe" "0.80" "0.0" "0.0"
run_variant "V18_ACTIVE_SAFE_SEARCH_ONLY_RELAXED" "active_safe" "0.70" "0.0" "0.0"

"$PYTHON_BIN" "$REPO_ROOT/scripts/aggregate_aggressive_t2_diagnostic.py" "$RUN_ROOT"
cp "$RUN_ROOT/aggressive_t2_shortcut_diagnostic_cn.md" "$RUN_ROOT/shortcut_integration_debug_cn.md"
cp "$RUN_ROOT/aggressive_t2_shortcut_diagnostic_cn.md" "$RUN_ROOT/aggressive_t2_integration_v2_cn.md"
echo "[debug] all done. root=${RUN_ROOT}"
