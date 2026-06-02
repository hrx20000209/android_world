#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/huangrunxi/Projects/android_world"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
RESULT_ROOT="${REPO_ROOT}/results/aggressive_t2_shortcut"
RUN_ROOT="${RUN_ROOT:-${RESULT_ROOT}/run_$(date +%Y%m%dT%H%M%S)}"
PHASE="${1:-diagnostic}"

TASKS_PILOT="SportsTrackerActivityDuration,SimpleCalendarEventsInTimeRange,NotesRecipeIngredientCount,TasksDueNextWeek,BrowserMaze"
TASKS_DIAGNOSTIC="SportsTrackerActivityDuration,SportsTrackerTotalDistanceForCategoryOverInterval,SportsTrackerActivitiesOnDate,SportsTrackerActivitiesCountForWeek,SportsTrackerTotalDurationForCategoryThisWeek,SimpleCalendarEventsOnDate,SimpleCalendarEventsInTimeRange,SimpleCalendarNextEvent,SimpleCalendarNextMeetingWithPerson,NotesRecipeIngredientCount,NotesTodoItemCount,NotesIsTodo,TasksDueOnDate,TasksDueNextWeek,TasksHighPriorityTasksDueOnDate,TasksHighPriorityTasks,FilesDeleteFile,FilesMoveFile,BrowserMaze,BrowserMultiply,MarkorCreateFolder,MarkorCreateNote,MarkorDeleteNote,MarkorDeleteNewestNote,ExpenseAddSingle,ExpenseDeleteSingle,ExpenseDeleteDuplicates,ClockStopWatchRunning,ClockTimerEntry,AudioRecorderRecordAudio"

case "$PHASE" in
  pilot)
    TASKS="$TASKS_PILOT"
    ;;
  diagnostic|30|40)
    TASKS="$TASKS_DIAGNOSTIC"
    ;;
  *)
    echo "Unknown phase: ${PHASE}; use pilot or diagnostic" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT"
echo "[run] phase=${PHASE}"
echo "[run] root=${RUN_ROOT}"
echo "[run] task_count=$("$PYTHON_BIN" - <<PY
print(len("${TASKS}".split(",")))
PY
)"

run_variant() {
  local label="$1"
  local agent_name="$2"
  local t2_mode="$3"
  local t2_state_mode="$4"
  local safe_search="$5"
  local variant_root="${RUN_ROOT}/${label}"
  local log_file="${variant_root}/driver.log"
  mkdir -p "$variant_root"

  local cmd=(
    "$PYTHON_BIN"
    "$REPO_ROOT/scripts/run_exploration_experiment_report.py"
    --run
    --agent_name="$agent_name"
    --suite_family=android_world
    --tasks="$TASKS"
    --n_task_combinations=1
    --task_random_seed=30
    --fixed_task_seed
    --image_downsample_scale=1.0
    --a11y_method=uiautomator
    --baseline_table=results/4B_2.txt
    --experiment_root="$variant_root"
    --max_cases=20
    --explore_max_runs=10000
    --explore_max_step=10000
    --explore_branch_budget=2
    --explore_branch_depth=2
    --explore_back_limit=4
    --explore_replay_max_actions=6
    --explore_planned_only
    --no-explore_fallback_safe_candidates
    --explore_safe_click_only
    --explore_skip_launcher
    --explore_filter_launcher_relevance
    --explore_skip_destructive_goals
    --explore_fast_mode
    --explore_fast_state
    --explore_transaction_safe
    --explore_action_settle_s=0.25
    --explore_hint_policy=strict
    --explore_search_policy=task_gate
    --explore_search_strategy=best_first
    --explore_rollback_policy=improved
    --explore_fixed_framework
    --explore_answer_extractors
    --explore_slot_complete
    --explore_slot_policy_switcher
    --explore_no_action_hint
    --explore_lightweight_a11y_trace
    --explore_trace_a11y_limit=32
    --explore_variant="$label"
    --t2_mode="$t2_mode"
    --t2_state_mode="$t2_state_mode"
    --t2_confidence_threshold=0.85
    --t2_min_top1_score=7.0
    --t2_min_score_margin=2.0
  )
  if [[ "$safe_search" == "1" ]]; then
    cmd+=(--t2_allow_safe_search_input)
  else
    cmd+=(--no-t2_allow_safe_search_input)
  fi

  echo "[run] starting ${label}"
  (
    cd "$REPO_ROOT"
    "${cmd[@]}" 2>&1 | tee "$log_file"
  )
  echo "[run] finished ${label}"
}

run_variant "V0_BASELINE" "gelab_agent" "off" "FULL_A11Y_CONTROL" "0"
run_variant "V12_NO_SHORTCUT" "explore_agent_gelab" "off" "FULL_A11Y_CONTROL" "0"
run_variant "V16_T2_SHADOW" "explore_agent_gelab" "shadow" "FULL_A11Y_CONTROL" "0"
run_variant "V17_T2_ACTIVE_SAFE_STRICT" "explore_agent_gelab" "active_safe" "FULL_A11Y_CONTROL" "0"
run_variant "V18_T2_ACTIVE_SAFE_WITH_SAFE_SEARCH_INPUT" "explore_agent_gelab" "active_safe" "FULL_A11Y_CONTROL" "1"
run_variant "V19_T2_ACTIVE_SAFE_LIGHT_STATE" "explore_agent_gelab" "active_safe" "HYBRID_LIGHT" "1"

"$PYTHON_BIN" "$REPO_ROOT/scripts/aggregate_aggressive_t2_diagnostic.py" "$RUN_ROOT"
echo "[run] all done. root=${RUN_ROOT}"
