#!/usr/bin/env bash
set -euo pipefail

# Aggressive T2 experiment runner for AndroidWorld.
# Phase:
#   pilot  -> 5-task smoke
#   full   -> 40-task diagnostic set

REPO_ROOT="/Users/huangrunxi/Projects/android_world"
PYTHON_BIN="$(command -v python3)"
RESULT_ROOT="${REPO_ROOT}/results/aggressive_t2_shortcut"
EXPERIMENT_NAME="run_$(date +%Y%m%dT%H%M%S)"
RUN_ROOT="${RESULT_ROOT}/${EXPERIMENT_NAME}"

PHASE="${1:-pilot}"
if [[ -n "${2:-}" ]]; then
  PHASE="${2:-$PHASE}"
fi

TASKS_PILOT="ExpenseDeleteSingle,MarkorDeleteNewestNote,NotesRecipeIngredientCount,SimpleCalendarEventsInTimeRange,SportsTrackerActivityDuration"
TASKS_FULL="AudioRecorderRecordAudio,AudioRecorderRecordAudioWithFileName,BrowserDraw,BrowserMaze,BrowserMultiply,CameraTakePhoto,CameraTakeVideo,ClockStopWatchPausedVerify,ClockStopWatchRunning,ClockTimerEntry,ContactsAddContact,ContactsNewContactDraft,ExpenseAddSingle,ExpenseDeleteSingle,ExpenseDeleteDuplicates,ExpenseDeleteMultiple,FilesDeleteFile,FilesMoveFile,MarkorAddNoteHeader,MarkorChangeNoteContent,MarkorCreateFolder,MarkorCreateNote,MarkorDeleteAllNotes,MarkorDeleteNewestNote,MarkorDeleteNote,MarkorEditNote,MarkorMoveNote,MarkorTranscribeReceipt,OpenAppTaskEval,OsmAndFavorite,NotesIsTodo,NotesTodoItemCount,NotesRecipeIngredientCount,SimpleCalendarEventsOnDate,SimpleCalendarEventsInTimeRange,SimpleCalendarNextEvent,SimpleCalendarNextMeetingWithPerson,SportsTrackerActivitiesCountForWeek,SportsTrackerActivitiesOnDate,SportsTrackerActivityDuration,SportsTrackerTotalDistanceForCategoryOverInterval,SportsTrackerTotalDurationForCategoryThisWeek,TasksDueOnDate,TasksDueNextWeek,TasksHighPriorityTasks,TasksHighPriorityTasksDueOnDate,SystemWifiTurnOffVerify,SystemWifiTurnOnVerify,SystemBluetoothTurnOff,SystemBluetoothTurnOn"

if [[ "$PHASE" == "pilot" ]]; then
  TASKS="$TASKS_PILOT"
elif [[ "$PHASE" == "full" ]]; then
  TASKS="$TASKS_FULL"
else
  echo "Unknown phase: $PHASE (pilot|full)"
  exit 1
fi

mkdir -p "$RUN_ROOT"

echo "[run] phase=${PHASE}"
echo "[run] task_count=$(python3 - <<PY
print(len("${TASKS}".split(',')))
PY)"
echo "[run] root=${RUN_ROOT}"

timestamp() { date +%Y%m%dT%H%M%S; }

run_variant() {
  local label="$1"
  local agent_name="$2"
  local t2_mode="$3"
  local t2_state_mode="$4"

  local variant_root="${RUN_ROOT}/${label}"
  mkdir -p "$variant_root"
  local log_file="${variant_root}/driver.log"

  cat >"${variant_root}/env.sh" <<ENV
export ANDROID_WORLD_EXPLORATION_TRACE_ROOT="${variant_root}/traces"
export ANDROID_WORLD_LIGHT_EXPLORER_SLOT_COMPLETE=1
export ANDROID_WORLD_LIGHT_EXPLORE_ENABLE=1
export ANDROID_WORLD_LIGHT_EXPLORE_MAX_RUNS=10000
export ANDROID_WORLD_LIGHT_EXPLORE_MAX_STEP=10000
export ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_BUDGET=2
export ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_DEPTH=3
export ANDROID_WORLD_LIGHT_EXPLORE_BACK_LIMIT=4
export ANDROID_WORLD_LIGHT_EXPLORE_PLANNED_ONLY=1
export ANDROID_WORLD_LIGHT_EXPLORE_FALLBACK_SAFE_CANDIDATES=1
export ANDROID_WORLD_LIGHT_EXPLORE_SAFE_CLICK_ONLY=1
export ANDROID_WORLD_LIGHT_EXPLORE_SKIP_LAUNCHER=1
export ANDROID_WORLD_LIGHT_EXPLORE_FILTER_LAUNCHER_RELEVANCE=1
export ANDROID_WORLD_LIGHT_EXPLORE_FAST_MODE=1
export ANDROID_WORLD_LIGHT_EXPLORE_FAST_STATE=1
export ANDROID_WORLD_LIGHT_EXPLORE_TRANSACTION_SAFE=1
export ANDROID_WORLD_LIGHT_EXPLORE_ACTION_SETTLE_S=0.25
export ANDROID_WORLD_LIGHT_EXPLORE_TRACE_A11Y_LIMIT=32
export ANDROID_WORLD_LIGHT_EXPLORE_HINT_POLICY=strict
export ANDROID_WORLD_LIGHT_EXPLORE_SEARCH_POLICY=task_gate
export ANDROID_WORLD_LIGHT_EXPLORE_SEARCH_STRATEGY=best_first
export ANDROID_WORLD_LIGHT_EXPLORE_ROLLBACK_POLICY=improved
export ANDROID_WORLD_LIGHT_EXPLORE_SKIP_DESTRUCTIVE_GOALS=1
export ANDROID_WORLD_LIGHT_EXPLORE_DISABLE_PROMPT_INJECTION=0
export ANDROID_WORLD_LIGHT_EXPLORE_SLOT_COMPLETE=1
export ANDROID_WORLD_LIGHT_EXPLORE_NO_ACTION_HINT=1
export ANDROID_WORLD_LIGHT_EXLPLORE_SLOT_POLICY_SWITCHER=0
export ANDROID_WORLD_LIGHT_EXPLORE_LIGHTWEIGHT_A11Y_TRACE=1
export ANDROID_WORLD_BANDIT_T2_BUDGET=3
export ANDROID_WORLD_BANDIT_STALLED_T2_BUDGET=4
export ANDROID_WORLD_BANDIT_NEGATIVE_CONTEXT=1
export ANDROID_WORLD_LIGHT_EXPLORE_VARIANT="${label}"
export ANDROID_WORLD_T2_MODE="${t2_mode}"
export ANDROID_WORLD_T2_STATE_MODE="${t2_state_mode}"
ENV

  local cmd=(
    "$PYTHON_BIN"
    "$REPO_ROOT/scripts/run_exploration_experiment_report.py"
    --run
    --agent_name="$agent_name"
    --suite_family=android_world
    --n_task_combinations=1
    --task_random_seed=30
    --fixed_task_seed
    --image_downsample_scale=1.0
    --a11y_method=uiautomator
    --baseline_table=results/4B_2.txt
    --experiment_root="$variant_root"
    --tasks="$TASKS"
    --max_cases=16
    --explore_slot_complete
    --explore_slot_policy_switcher
    --explore_fast_mode
    --explore_transaction_safe
    --explore_max_runs=10000
    --explore_max_step=10000
    --explore_branch_budget=2
    --explore_branch_depth=3
    --explore_back_limit=4
    --explore_planned_only
    --explore_fallback_safe_candidates
    --explore_safe_click_only
    --explore_skip_launcher
    --explore_filter_launcher_relevance
    --explore_skip_destructive_goals
    --explore_hint_policy=strict
    --explore_search_policy=task_gate
    --explore_search_strategy=best_first
    --explore_rollback_policy=improved
    --explore_fixed_framework
    --explore_answer_extractors
    --explore_no_action_hint
    --explore_lightweight_a11y_trace
    --explore_trace_a11y_limit=32
    --explore_variant="${label}"
  )

  echo "[run] starting ${label}"
  (
    set -a
    source "${variant_root}/env.sh"
    set +a
    cd "$REPO_ROOT"
    "${cmd[@]}" 2>&1 | tee "$log_file"
  )

  local latest_run
  latest_run="$(ls -dt "${variant_root}"/run_* 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_run}" ]]; then
    cp "${latest_run}/report/summary.json" "${variant_root}/summary.json" 2>/dev/null || true
    cp "${latest_run}/report/step_metrics.csv" "${variant_root}/per_step_metrics.csv" 2>/dev/null || true
    cp "${latest_run}/traces/exploration_trace.jsonl" "${variant_root}/exploration_trace.jsonl" 2>/dev/null || true
    cp "${latest_run}/traces/exploration_match_trace.jsonl" "${variant_root}/exploration_match_trace.jsonl" 2>/dev/null || true
    cp "${latest_run}/traces/evidence_capsules.jsonl" "${variant_root}/evidence_capsules.jsonl" 2>/dev/null || true
    cp "${latest_run}/traces/rollback_trace.jsonl" "${variant_root}/rollback_events.jsonl" 2>/dev/null || true

    # 简单生成 per_task_results.csv（用于对齐你要的报告字段）
    python3 - "$latest_run" "${label}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
label = sys.argv[2]
summary = json.loads((run_root / "report" / "summary.json").read_text(encoding="utf-8"))
rows = []
base_rows = {str(r.get("task") or ""): r for r in summary.get("baseline_compare", {}).get("rows", [])}
for row in summary.get("episodes", {}).get("task_episode_rows", []) or []:
    task = str(row.get("task") or "")
    base = base_rows.get(task, {})
    rows.append({
        "task_id": task,
        "app": "",
        "task_mode": "",
        "task_subtype": "",
        "variant": label,
        "baseline_success": base.get("baseline_success_rate", ""),
        "baseline_steps": base.get("baseline_episode_length", ""),
        "variant_success": float(row.get("success") or 0.0),
        "variant_steps": row.get("episode_length", ""),
        "rescued": "",
        "broken": "",
        "explored_pages": "",
        "shortcut_candidates": "",
        "would_fire": "",
        "fired": "",
        "skipped_vlm_calls": "",
        "rollback_success": "",
        "rollback_failure": "",
    })

a = run_root / "per_task_results.csv"
with a.open("w", encoding="utf-8", newline="") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
PY
  fi

  echo "[run] finished ${label}"
}

run_variant "V0_BASELINE" "gelab_agent" "off" ""
run_variant "V12_NO_SHORTCUT" "explore_agent_gelab" "off" ""
run_variant "V16_T2_SHADOW" "explore_agent_gelab" "shadow" "HYBRID_LIGHT"
run_variant "V17_T2_ACTIVE_SAFE" "explore_agent_gelab" "active_safe" "CACHED_FULL_A11Y"
run_variant "V18_T2_ACTIVE_SAFE_LIGHT_STATE" "explore_agent_gelab" "active_safe" "HYBRID_LIGHT"
run_variant "V19_DIRECT_ANSWER_SHADOW" "explore_agent_gelab" "direct_answer_shadow" ""

echo "[run] all done. root=${RUN_ROOT}"
