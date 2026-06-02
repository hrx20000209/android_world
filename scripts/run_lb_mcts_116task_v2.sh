#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASKS_CSV="${TASKS_CSV:-${ROOT_DIR}/configs/frozen_30task_selection.csv}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/results/lb_mcts_final_116task_v2}"
LB_ROOT="${LB_ROOT:-${EXP_ROOT}/LB_MCTS_FINAL}"
MAX_STEPS="${MAX_STEPS:-16}"
CONSOLE_PORT="${CONSOLE_PORT:-5554}"
A11Y_METHOD="${A11Y_METHOD:-fast_provider}"

if [[ -z "${ADB_PATH:-}" ]]; then
  if [[ -n "${ANDROID_ADB_PATH:-}" ]]; then
    ADB_PATH="${ANDROID_ADB_PATH}"
  elif command -v adb >/dev/null 2>&1; then
    ADB_PATH="$(command -v adb)"
  elif [[ -x "/opt/homebrew/bin/adb" ]]; then
    ADB_PATH="/opt/homebrew/bin/adb"
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

mkdir -p "${EXP_ROOT}" "${LB_ROOT}"

TASKS="$(${PYTHON:-python3} - "${TASKS_CSV}" "${EXP_ROOT}/ordered_116task_tasks.txt" "${EXP_ROOT}/ordered_116task_tasks.csv" <<'PY'
import csv
import sys
from pathlib import Path
from android_world import registry

csv_path = Path(sys.argv[1])
out_txt = Path(sys.argv[2])
out_csv = Path(sys.argv[3])
with csv_path.open(newline="", encoding="utf-8") as f:
    frozen_rows = list(csv.DictReader(f))
frozen = []
seen = set()
for row in frozen_rows:
    task = (row.get("task") or row.get("task_id") or "").strip()
    if task and task not in seen:
        frozen.append(task)
        seen.add(task)
reg = registry.TaskRegistry().get_registry(family=registry.TaskRegistry.ANDROID_WORLD_FAMILY)
full = list(reg.keys())
missing = [task for task in frozen if task not in reg]
if missing:
    raise SystemExit("frozen tasks missing from android_world registry: " + ",".join(missing))
ordered = frozen + [task for task in full if task not in seen]
out_txt.write_text("\n".join(ordered) + "\n", encoding="utf-8")
with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["order", "task", "prefix"])
    w.writeheader()
    for i, task in enumerate(ordered, 1):
        w.writerow({"order": i, "task": task, "prefix": "frozen30" if task in seen else "remaining"})
print(",".join(ordered))
PY
)"
TASK_COUNT="$(wc -l < "${EXP_ROOT}/ordered_116task_tasks.txt" | tr -d ' ')"

echo "[lb-mcts-v2] using adb: ${ADB_PATH}"
echo "[lb-mcts-v2] tasks=${TASK_COUNT}; ordered list=${EXP_ROOT}/ordered_116task_tasks.txt"
echo "[lb-mcts-v2] first 30 tasks follow ${TASKS_CSV} order"

export ANDROID_WORLD_MAX_STEPS="${MAX_STEPS}"
export ANDROID_WORLD_MAX_N_STEPS="${MAX_STEPS}"
export PYTHONUNBUFFERED=1
export ANDROID_WORLD_DECOUPLED_EXPLORATION=1
export ANDROID_WORLD_EXPLORATION_TIMING=parallel_shadow
export ANDROID_WORLD_LIGHT_EXPLORE_USE_CURRENT_ACTION=0
export ANDROID_WORLD_LIGHT_EXPLORE_USE_PLANNING_TEXT=0
export ANDROID_WORLD_LIGHT_EXPLORE_PLANNED_ONLY=0
export ANDROID_WORLD_LIGHT_EXPLORE_LB_MCTS=1
export ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_BUDGET=12
export ANDROID_WORLD_LIGHT_EXPLORE_MIN_ATTEMPTS_PER_STEP=12
export ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_DEPTH=2
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
export ANDROID_WORLD_T2_ALLOW_SAFE_SEARCH_INPUT=1
export ANDROID_WORLD_LIGHT_EXPLORE_HOME_REPLAY_TIMEOUT_SEC="${ANDROID_WORLD_LIGHT_EXPLORE_HOME_REPLAY_TIMEOUT_SEC:-30}"
export ANDROID_WORLD_TRACE_SCREENSHOT_MODE=failure+level2+depth2+injected+sampled

${PYTHON:-python3} scripts/run_exploration_experiment_report.py \
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
  --explore_max_runs=100000 \
  --explore_max_step=100000 \
  --explore_branch_budget=12 \
  --explore_branch_depth=2 \
  --explore_min_attempts_per_step=12 \
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
  --explore_variant=LB_MCTS_FINAL_V2
