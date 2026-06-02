#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/run_exploration_experiment_report.py \
  --run \
  --agent_name=explore_agent_gelab \
  --suite_family=android_world \
  --n_task_combinations=1 \
  --fixed_task_seed \
  --image_downsample_scale=1.0 \
  --a11y_method=uiautomator \
  --experiment_root=results/exploration_full_androidworld_transaction_safe \
  --baseline_table=results/4B_2.txt \
  --explore_force_every_step \
  --no-explore_diagnostic_full \
  --explore_fast_mode \
  --explore_transaction_safe \
  --explore_max_runs=10000 \
  --explore_max_step=10000 \
  --explore_branch_budget=1 \
  --explore_branch_depth=2 \
  --explore_back_limit=2 \
  --explore_replay_max_actions=3 \
  --explore_planned_only \
  --no-explore_safe_click_only \
  --explore_skip_launcher \
  --no-explore_skip_destructive_goals \
  --explore_strategy=dfs \
  --max_cases=20 \
  "$@"
