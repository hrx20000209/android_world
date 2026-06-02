#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASKS_CSV="${TASKS_CSV:-${ROOT_DIR}/configs/frozen_30task_selection.csv}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/results/lb_mcts_final_30task}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results}"
MAX_STEPS="${MAX_STEPS:-16}"
if [[ -z "${ADB_PATH:-}" ]]; then
  if [[ -n "${ANDROID_ADB_PATH:-}" ]]; then
    ADB_PATH="${ANDROID_ADB_PATH}"
  elif command -v adb >/dev/null 2>&1; then
    ADB_PATH="$(command -v adb)"
  elif [[ -x "${HOME}/Library/Android/sdk/platform-tools/adb" ]]; then
    ADB_PATH="${HOME}/Library/Android/sdk/platform-tools/adb"
  elif [[ -x "${HOME}/Android/Sdk/platform-tools/adb" ]]; then
    ADB_PATH="${HOME}/Android/Sdk/platform-tools/adb"
  elif [[ -n "${ANDROID_HOME:-}" && -x "${ANDROID_HOME}/platform-tools/adb" ]]; then
    ADB_PATH="${ANDROID_HOME}/platform-tools/adb"
  else
    echo "adb not found. Set ADB_PATH=/absolute/path/to/adb and rerun." >&2
    exit 2
  fi
fi
CONSOLE_PORT="${CONSOLE_PORT:-5554}"
A11Y_METHOD="${A11Y_METHOD:-fast_provider}"
RUN_MISSING_BASELINE="${RUN_MISSING_BASELINE:-1}"

mkdir -p "${EXP_ROOT}"
echo "[lb-mcts] using adb: ${ADB_PATH}"

TASKS="$(
python3 - "${TASKS_CSV}" <<'PY'
import csv, sys
path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
tasks = [r.get("task") or r.get("task_id") or "" for r in rows]
print(",".join(t for t in tasks if t))
PY
)"

python3 scripts/check_existing_baseline.py \
  --results_root "${RESULTS_ROOT}" \
  --tasks_csv "${TASKS_CSV}" \
  --output_dir "${EXP_ROOT}"

INVENTORY="${EXP_ROOT}/baseline_inventory.json"
BASELINE_PATH="$(
python3 - "${INVENTORY}" <<'PY'
import json, sys
inv = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("matched_selected", "baseline_30", "baseline_50", "baseline_116"):
    item = inv.get(key)
    if isinstance(item, dict) and item.get("path"):
        print(item["path"])
        break
PY
)"

if [[ -z "${BASELINE_PATH}" && "${RUN_MISSING_BASELINE}" == "1" ]]; then
  BASELINE_PATH="${EXP_ROOT}/B0_BASELINE_RERUN"
  echo "[lb-mcts] no reusable matched baseline found; running missing B0 subset at ${BASELINE_PATH}"
  python3 scripts/run_exploration_experiment_report.py \
    --run \
    --suite_family=android_world \
    --tasks="${TASKS}" \
    --n_task_combinations=1 \
    --task_random_seed=43 \
    --fixed_task_seed \
    --agent_name=explore_agent_gelab \
    --image_downsample_scale=1.0 \
    --adb_path="${ADB_PATH}" \
    --console_port="${CONSOLE_PORT}" \
    --a11y_method="${A11Y_METHOD}" \
    --a11y_preflight_timeout=90 \
    --experiment_root="${BASELINE_PATH}" \
    --max_n_steps="${MAX_STEPS}" \
    --no-explore_enable \
    --explore_max_runs=0 \
    --explore_max_step=0 \
    --explore_branch_budget=0 \
    --explore_branch_depth=0 \
    --explore_min_attempts_per_step=0 \
    --t2_mode=off \
    --no-t2_allow_safe_search_input
  python3 scripts/generate_lb_mcts_report.py \
    --experiment_root "${EXP_ROOT}" \
    --variant B0_BASELINE_RERUN \
    --max_steps "${MAX_STEPS}" \
    --report_name baseline_materialize_report_cn.md >/dev/null
fi

export ANDROID_WORLD_MAX_STEPS="${MAX_STEPS}"
export ANDROID_WORLD_MAX_N_STEPS="${MAX_STEPS}"
export ANDROID_WORLD_DECOUPLED_EXPLORATION=1
export ANDROID_WORLD_EXPLORATION_TIMING=parallel_shadow
export ANDROID_WORLD_LIGHT_EXPLORE_USE_CURRENT_ACTION=0
export ANDROID_WORLD_LIGHT_EXPLORE_USE_PLANNING_TEXT=0
export ANDROID_WORLD_LIGHT_EXPLORE_PLANNED_ONLY=0
export ANDROID_WORLD_LIGHT_EXPLORE_LB_MCTS=1
export ANDROID_WORLD_LB_MCTS_C_PUCT="${ANDROID_WORLD_LB_MCTS_C_PUCT:-1.4}"
export ANDROID_WORLD_LB_MCTS_LAMBDA_RISK="${ANDROID_WORLD_LB_MCTS_LAMBDA_RISK:-1.0}"
export ANDROID_WORLD_LB_MCTS_LAMBDA_LATENCY="${ANDROID_WORLD_LB_MCTS_LAMBDA_LATENCY:-0.5}"
export ANDROID_WORLD_LB_MCTS_MIN_BUDGET_MS="${ANDROID_WORLD_LB_MCTS_MIN_BUDGET_MS:-3000}"
export ANDROID_WORLD_LB_MCTS_MAX_BUDGET_MS="${ANDROID_WORLD_LB_MCTS_MAX_BUDGET_MS:-12000}"
export ANDROID_WORLD_LB_MCTS_SLACK_RATIO="${ANDROID_WORLD_LB_MCTS_SLACK_RATIO:-0.8}"
export ANDROID_WORLD_LB_MCTS_MAX_ROLLOUTS_PER_STEP="${ANDROID_WORLD_LB_MCTS_MAX_ROLLOUTS_PER_STEP:-12}"
export ANDROID_WORLD_LB_MCTS_MIN_DEPTH2_ATTEMPTS="${ANDROID_WORLD_LB_MCTS_MIN_DEPTH2_ATTEMPTS:-1}"
export ANDROID_WORLD_T2_MODE=off
export ANDROID_WORLD_T2_ACTIVE=0

LB_ROOT="${EXP_ROOT}/LB_MCTS_FINAL"
python3 scripts/run_exploration_experiment_report.py \
  --run \
  --suite_family=android_world \
  --tasks="${TASKS}" \
  --n_task_combinations=1 \
  --task_random_seed=43 \
  --fixed_task_seed \
  --agent_name=explore_agent_gelab \
  --image_downsample_scale=1.0 \
  --adb_path="${ADB_PATH}" \
  --console_port="${CONSOLE_PORT}" \
  --a11y_method="${A11Y_METHOD}" \
  --a11y_preflight_timeout=90 \
  --experiment_root="${LB_ROOT}" \
  --max_n_steps="${MAX_STEPS}" \
  --explore_enable \
  --explore_max_runs=10000 \
  --explore_max_step=10000 \
  --explore_branch_budget=12 \
  --explore_branch_depth=2 \
  --explore_min_attempts_per_step=0 \
  --decoupled_exploration \
  --exploration_timing=parallel_shadow \
  --no-explore_use_current_action \
  --no-explore_use_planning_text \
  --no-explore_planned_only \
  --explore_decouple_planned \
  --explore_parallel_vlm \
  --no-explore_parallel_lookahead \
  --explore_fallback_safe_candidates \
  --explore_safe_click_only \
  --explore_skip_launcher \
  --explore_filter_launcher_relevance \
  --no-explore_skip_destructive_goals \
  --explore_fast_mode \
  --explore_fast_state \
  --explore_transaction_safe \
  --explore_action_settle_s=0.25 \
  --explore_strategy=dfs \
  --explore_search_policy=task_gate \
  --explore_search_strategy=mcts \
  --explore_hint_policy=strict \
  --explore_rollback_policy=improved \
  --explore_fixed_framework \
  --explore_safe_mcts \
  --explore_answer_extractors \
  --explore_slot_complete \
  --explore_slot_policy_switcher \
  --explore_upper_bound_evidence \
  --explore_enable_t2_lookahead \
  --explore_lightweight_a11y_trace \
  --explore_trace_a11y_limit=120 \
  --trace_screenshot_mode=failure+level2+depth2+injected+sampled \
  --t2_mode=off \
  --t2_allow_safe_search_input \
  --latency_profile \
  --explore_variant=LB_MCTS_FINAL

python3 scripts/generate_lb_mcts_report.py \
  --experiment_root "${EXP_ROOT}" \
  --variant LB_MCTS_FINAL \
  --baseline_inventory "${INVENTORY}" \
  --baseline_path "${BASELINE_PATH}" \
  --max_steps "${MAX_STEPS}" \
  --report_name lb_mcts_30task_report_cn.md

echo "[lb-mcts] report: ${LB_ROOT}/lb_mcts_30task_report_cn.md"
