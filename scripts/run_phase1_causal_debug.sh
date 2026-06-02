#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-NotesRecipeIngredientCount,BrowserMaze,SportsTrackerActivityDuration,SimpleCalendarNextMeetingWithPerson,ClockStopWatchRunning}"
ROOT="${ROOT:-results/phase1_causal_debug/run_$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$ROOT"
echo "$TASKS" > "$ROOT/task_list.txt"
echo "[phase1] root=$ROOT"

run_variant() {
  local variant="$1"; shift
  local dir="$ROOT/$variant"
  mkdir -p "$dir"
  echo "[phase1] preflight variant=$variant"
  curl -fsS --max-time 5 http://localhost:8081/v1/models >/dev/null
  echo "[phase1] variant=$variant"
  python3 -u scripts/run_exploration_experiment_report.py --run \
    --suite_family=android_world \
    --tasks="$TASKS" \
    --n_task_combinations=1 \
    --task_random_seed=43 \
    --fixed_task_seed \
    --image_downsample_scale=1.0 \
    --baseline_table=results/4B_2.txt \
    --experiment_root="$dir" \
    --max_cases=20 \
    --a11y_method=fast_provider \
    --a11y_preflight_timeout=90 \
    --latency_profile \
    --explore_variant="$variant" \
    "$@" 2>&1 | tee "$dir/driver.log"
}

COMMON_EXPLORE=(
  --agent_name=explore_agent_gelab
  --explore_enable
  --explore_max_runs=10000
  --explore_max_step=10000
  --explore_branch_budget=10
  --explore_branch_depth=2
  --explore_min_attempts_per_step=10
  --exploration_timing=parallel_shadow
  --decoupled_exploration
  --no-explore_use_current_action
  --no-explore_use_planning_text
  --explore_back_limit=4
  --explore_replay_max_actions=6
  --no-explore_planned_only
  --explore_decouple_planned
  --explore_parallel_vlm
  --explore_parallel_lookahead
  --explore_fallback_safe_candidates
  --explore_safe_click_only
  --explore_skip_launcher
  --explore_filter_launcher_relevance
  --explore_skip_destructive_goals
  --explore_fast_mode
  --explore_fast_state
  --explore_transaction_safe
  --explore_quality_filters
  --explore_action_settle_s=0.25
  --explore_hint_policy=strict
  --explore_search_policy=task_gate
  --explore_search_strategy=best_first
  --explore_rollback_policy=improved
  --explore_fixed_framework
  --explore_answer_extractors
  --explore_slot_complete
  --explore_slot_policy_switcher
  --explore_enable_t2_lookahead
  --explore_lightweight_a11y_trace
  --explore_trace_a11y_limit=80
  --t2_mode=shadow
)

run_variant B0_BASELINE_RERUN \
  --agent_name=gelab_agent_resize \
  --no-explore_enable \
  --explore_max_runs=0 \
  --explore_max_step=0 \
  --explore_branch_budget=0 \
  --explore_branch_depth=0 \
  --explore_min_attempts_per_step=0 \
  --no-explore_enable_t2_lookahead \
  --t2_mode=off

run_variant D0_EXPLORATION_SHADOW_ONLY \
  "${COMMON_EXPLORE[@]}" \
  --explore_shadow_only \
  --explore_disable_prompt_injection

run_variant D1_REAL_EXPLORATION_NO_INJECTION \
  "${COMMON_EXPLORE[@]}" \
  --no-explore_shadow_only \
  --explore_disable_prompt_injection

run_variant D2_STRICT_INJECTION \
  "${COMMON_EXPLORE[@]}" \
  --no-explore_shadow_only \
  --no-explore_disable_prompt_injection \
  --no-explore_relaxed_diagnostic_injection

run_variant D3_RELAXED_DIAGNOSTIC_INJECTION \
  "${COMMON_EXPLORE[@]}" \
  --no-explore_shadow_only \
  --no-explore_disable_prompt_injection \
  --explore_relaxed_diagnostic_injection

python3 scripts/summarize_phase1_causal_debug.py "$ROOT"
echo "[phase1] complete root=$ROOT"
