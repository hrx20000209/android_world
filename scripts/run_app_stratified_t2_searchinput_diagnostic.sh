#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/huangrunxi/Projects/android_world"
TS="$(date +%Y%m%dT%H%M%S)"
RUN_ROOT="${ROOT_DIR}/results/aggressive_t2_shortcut/app_stratified_searchinput/run_${TS}"
TASKS="NotesRecipeIngredientCount,NotesTodoItemCount,BrowserMaze,BrowserMultiply,SportsTrackerActivityDuration,SportsTrackerActivitiesOnDate,SimpleCalendarNextMeetingWithPerson,SimpleCalendarEventsOnDate,TasksHighPriorityTasksDueOnDate,TasksDueNextWeek,MarkorCreateFolder,MarkorDeleteNewestNote,ExpenseAddSingle,ExpenseDeleteSingle,ClockStopWatchRunning,SystemWifiTurnOnVerify"

mkdir -p "${RUN_ROOT}"

cat > "${RUN_ROOT}/task_selection.md" <<'EOF'
# App-stratified aggressive t+2 SearchInputShortcut diagnostic task selection

任务清单在运行前冻结。本轮不根据单个 task 结果修改规则后重跑同一任务，不跑 full 116。

| # | task_id | app | role | task_mode | shortcut 预期 |
|---:|---|---|---|---|---|
| 1 | NotesRecipeIngredientCount | Joplin/Notes | shortcut-friendly | RECIPE_INGREDIENT | SearchInputShortcut: Chicken Alfredo |
| 2 | NotesTodoItemCount | Joplin/Notes | no-harm/control | INFO_QUERY_COUNT | 不应 active generic |
| 3 | BrowserMaze | Files/Browser | shortcut-friendly | NAVIGATION_SEARCH | SearchInputShortcut: task.html if search UI exists |
| 4 | BrowserMultiply | Browser | no-harm/control | NAVIGATION_SEARCH | 不应 active generic navigation |
| 5 | SportsTrackerActivityDuration | OpenTracks | shortcut-friendly | ACTIVITY_STATS | SearchInputShortcut: October 12 2023 / skiing if search UI exists |
| 6 | SportsTrackerActivitiesOnDate | OpenTracks | no-harm/control | INFO_QUERY_COUNT | 不应 active generic filter/date click |
| 7 | SimpleCalendarNextMeetingWithPerson | Simple Calendar | shortcut-friendly | EVENT_QUERY | SearchInputShortcut: Ava if search UI exists |
| 8 | SimpleCalendarEventsOnDate | Simple Calendar | no-harm/control | EVENT_QUERY | 不应 active date-cell generic click |
| 9 | TasksHighPriorityTasksDueOnDate | Tasks | shortcut-friendly | INFO_QUERY_COUNT | SearchInputShortcut: October 18 2023 if search UI exists |
| 10 | TasksDueNextWeek | Tasks | no-harm/control | INFO_QUERY_COUNT | 不应 active due-date row generic click |
| 11 | MarkorCreateFolder | Markor | no-harm/control | FORM_CREATE_EDIT | active input 禁用 |
| 12 | MarkorDeleteNewestNote | Markor | no-harm/control | DELETE_COMMIT | active delete/confirm 禁用 |
| 13 | ExpenseAddSingle | Pro Expense | no-harm/control | FORM_CREATE_EDIT | active expense form input 禁用 |
| 14 | ExpenseDeleteSingle | Pro Expense | no-harm/control | DELETE_COMMIT | active delete/confirm 禁用 |
| 15 | ClockStopWatchRunning | Clock | no-harm/control | SIMPLE_VERIFY | active shortcut 禁用 |
| 16 | SystemWifiTurnOnVerify | Settings/System | no-harm/control | SIMPLE_VERIFY | active system mutation 禁用 |

覆盖 app：Joplin/Notes, Files/Browser, Browser, OpenTracks, Simple Calendar, Tasks, Markor, Pro Expense, Clock, Settings/System。
每个 app 最多 2 个 task。
EOF

common_args=(
  --run
  --agent_name=explore_agent_gelab
  --suite_family=android_world
  --tasks="${TASKS}"
  --n_task_combinations=1
  --task_random_seed=31
  --fixed_task_seed
  --image_downsample_scale=1.0
  --a11y_method=uiautomator
  --baseline_table=results/4B_2.txt
  --max_cases=20
)

explore_args=(
  --explore_max_runs=10000
  --explore_max_step=10000
  --explore_branch_budget=1
  --explore_branch_depth=2
  --explore_back_limit=4
  --explore_planned_only
  --no-explore_fallback_safe_candidates
  --no-explore_safe_click_only
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
)

run_variant() {
  local label="$1"
  local t2_mode="$2"
  local threshold="$3"
  local variant_root="${RUN_ROOT}/${label}"
  mkdir -p "${variant_root}"
  echo "[app-stratified] starting ${label}"
  cd "${ROOT_DIR}"
  if [[ "${label}" == "V0_BASELINE" ]]; then
    /Users/huangrunxi/anaconda3/bin/python3 "${ROOT_DIR}/scripts/run_exploration_experiment_report.py" \
      "${common_args[@]}" \
      --experiment_root="${variant_root}" \
      --explore_variant="${label}" \
      --explore_max_runs=0 \
      --explore_max_step=0 \
      --explore_branch_budget=0 \
      --explore_branch_depth=0 \
      --no-explore_force_every_step \
      --explore_disable_prompt_injection \
      --t2_mode=off \
      | tee "${variant_root}/driver.log"
  else
    /Users/huangrunxi/anaconda3/bin/python3 "${ROOT_DIR}/scripts/run_exploration_experiment_report.py" \
      "${common_args[@]}" \
      "${explore_args[@]}" \
      --experiment_root="${variant_root}" \
      --explore_variant="${label}" \
      --t2_mode="${t2_mode}" \
      --t2_state_mode=FULL_A11Y_CONTROL \
      --t2_allow_safe_search_input \
      --t2_confidence_threshold="${threshold}" \
      --t2_min_top1_score=0.0 \
      --t2_min_score_margin=0.0 \
      | tee "${variant_root}/driver.log"
  fi
  echo "[app-stratified] finished ${label}"
}

run_variant V0_BASELINE off 1.00
run_variant V12_NO_SHORTCUT off 1.00
run_variant V16_SHADOW_SEARCHINPUT_ONLY shadow 0.85
run_variant V18_ACTIVE_SEARCHINPUT_ONLY_STRICT active_safe 0.85
run_variant V18_ACTIVE_SEARCHINPUT_ONLY_RELAXED_DEBUG active_safe 0.70

/Users/huangrunxi/anaconda3/bin/python3 "${ROOT_DIR}/scripts/aggregate_app_stratified_shortcut_report.py" "${RUN_ROOT}"

echo "[app-stratified] all done. root=${RUN_ROOT}"
