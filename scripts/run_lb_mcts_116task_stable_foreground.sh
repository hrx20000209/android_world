#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/results/lb_mcts_final_116task_stable}"
LOG_DIR="${EXP_ROOT}/logs"
mkdir -p "${LOG_DIR}"

if ps aux | grep -E 'run_exploration_experiment_report.py|/run.py --agent_name=explore_agent_gelab|run_lb_mcts_116task_v2.sh|run_lb_mcts_116task_stable_foreground.sh' | grep -v grep | grep -v "$$" >/dev/null 2>&1; then
  echo "[stable-116] Existing AndroidWorld experiment process detected. Stop it first, then rerun." >&2
  ps aux | grep -E 'run_exploration_experiment_report.py|/run.py --agent_name=explore_agent_gelab|run_lb_mcts_116task_v2.sh|run_lb_mcts_116task_stable_foreground.sh' | grep -v grep | grep -v "$$" >&2 || true
  exit 2
fi

export EXP_ROOT
export A11Y_METHOD="${A11Y_METHOD:-uiautomator}"
export MAX_STEPS="${MAX_STEPS:-16}"
export PYTHONUNBUFFERED=1

RUN_LOG="${LOG_DIR}/foreground_run_$(date +%Y%m%dT%H%M%S)_${A11Y_METHOD}.log"
MON_LOG="${LOG_DIR}/foreground_monitor_$(date +%Y%m%dT%H%M%S).log"

echo "[stable-116] root=${ROOT_DIR}"
echo "[stable-116] exp_root=${EXP_ROOT}"
echo "[stable-116] a11y=${A11Y_METHOD}"
echo "[stable-116] max_steps=${MAX_STEPS}"
echo "[stable-116] run_log=${RUN_LOG}"
echo "[stable-116] monitor_log=${MON_LOG}"
echo "[stable-116] first 30 tasks follow configs/frozen_30task_selection.csv"

# Generate ordered task list before monitor starts. The v2 runner will regenerate the same list.
python3 - "${ROOT_DIR}/configs/frozen_30task_selection.csv" "${EXP_ROOT}/ordered_116task_tasks.txt" "${EXP_ROOT}/ordered_116task_tasks.csv" <<'PY'
import csv
import sys
from pathlib import Path
from android_world import registry
csv_path = Path(sys.argv[1])
out_txt = Path(sys.argv[2])
out_csv = Path(sys.argv[3])
out_txt.parent.mkdir(parents=True, exist_ok=True)
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
missing = [task for task in frozen if task not in reg]
if missing:
    raise SystemExit("frozen tasks missing from android_world registry: " + ",".join(missing))
ordered = frozen + [task for task in reg.keys() if task not in seen]
out_txt.write_text("\n".join(ordered) + "\n", encoding="utf-8")
with out_csv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["order", "task", "prefix"])
    w.writeheader()
    for i, task in enumerate(ordered, 1):
        w.writerow({"order": i, "task": task, "prefix": "frozen30" if task in seen else "remaining"})
print(f"[stable-116] ordered_tasks={len(ordered)}")
PY

python3 "${ROOT_DIR}/scripts/monitor_lb_mcts_116task_v2.py" \
  --variant-root "${EXP_ROOT}/LB_MCTS_FINAL" \
  --expected-tasks-file "${EXP_ROOT}/ordered_116task_tasks.txt" \
  --loop \
  --interval-sec "${MONITOR_INTERVAL_SEC:-600}" > "${MON_LOG}" 2>&1 &
MON_PID=$!
echo "${MON_PID}" > "${EXP_ROOT}/monitor.pid"

cleanup() {
  status=$?
  if kill -0 "${MON_PID}" >/dev/null 2>&1; then
    kill "${MON_PID}" >/dev/null 2>&1 || true
  fi
  python3 "${ROOT_DIR}/scripts/monitor_lb_mcts_116task_v2.py" \
    --variant-root "${EXP_ROOT}/LB_MCTS_FINAL" \
    --expected-tasks-file "${EXP_ROOT}/ordered_116task_tasks.txt" >/dev/null 2>&1 || true
  if [[ -f "${ROOT_DIR}/scripts/summarize_a11y_latency_breakdown.py" ]]; then
    python3 "${ROOT_DIR}/scripts/summarize_a11y_latency_breakdown.py" \
      --variant-root "${EXP_ROOT}/LB_MCTS_FINAL" >/dev/null 2>&1 || true
  fi
  echo "[stable-116] exit_status=${status}"
  echo "[stable-116] latest_progress=${EXP_ROOT}/LB_MCTS_FINAL/monitor/latest_progress_cn.md"
  echo "[stable-116] latest_latency_breakdown=${EXP_ROOT}/LB_MCTS_FINAL/$(ls -1t "${EXP_ROOT}/LB_MCTS_FINAL" 2>/dev/null | grep '^run_' | head -1)/report/latency_breakdown/latency_breakdown_cn.md"
  exit "${status}"
}
trap cleanup EXIT INT TERM

"${ROOT_DIR}/scripts/run_lb_mcts_116task_v2.sh" 2>&1 | tee "${RUN_LOG}"
