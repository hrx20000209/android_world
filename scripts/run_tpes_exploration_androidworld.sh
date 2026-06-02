#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Broad TPES verification mode:
# - every reasoning step tries speculative exploration
# - multiple root branches are explored
# - depth is >2 so t+1 can be matched while t+2/t+3 evidence is available
# - transaction-safe remains enabled to avoid irreversible app-state commits
DEFAULT_TASKS="ContactsAddContact,ExpenseDeleteDuplicates,MarkorCreateNote,RecipeDeleteSingleRecipe,SimpleCalendarDeleteOneEvent,AudioRecorderRecordAudio,ClockTimerEntry,FilesDeleteFile,MarkorAddNoteHeader,SimpleCalendarEventsInNextWeek,ExpenseDeleteMultiple,ExpenseDeleteSingle,MarkorDeleteNote,SimpleSmsReplyMostRecent,SystemWifiTurnOffVerify,TasksHighPriorityTasks,CameraTakePhoto,SimpleCalendarEventsOnDate,NotesTodoItemCount,SystemCopyToClipboard"
FULL_SUITE="${FULL_SUITE:-0}"
TASKS="${TASKS:-$DEFAULT_TASKS}"
TASK_ARGS=()
if [[ "$FULL_SUITE" != "1" ]]; then
  TASK_ARGS+=(--tasks="${TASKS}")
fi

python scripts/run_exploration_experiment_report.py \
  --run \
  --agent_name=explore_agent_gelab \
  --suite_family=android_world \
  "${TASK_ARGS[@]}" \
  --n_task_combinations=1 \
  --fixed_task_seed \
  --image_downsample_scale=1.0 \
  --a11y_method=uiautomator \
  --experiment_root=results/tpes_androidworld_broad \
  --baseline_table=results/4B_2.txt \
  --explore_force_every_step \
  --no-explore_diagnostic_full \
  --explore_fast_mode \
  --explore_transaction_safe \
  --explore_max_runs=10000 \
  --explore_max_step=10000 \
  --explore_branch_budget=4 \
  --explore_branch_depth=3 \
  --explore_back_limit=4 \
  --explore_replay_max_actions=6 \
  --no-explore_planned_only \
  --explore_fallback_safe_candidates \
  --no-explore_safe_click_only \
  --no-explore_skip_launcher \
  --no-explore_filter_launcher_relevance \
  --no-explore_skip_destructive_goals \
  --explore_strategy=dfs \
  --max_cases=20 \
  "$@"
