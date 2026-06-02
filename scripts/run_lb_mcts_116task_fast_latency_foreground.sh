#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/results/lb_mcts_final_116task_fast_latency}"
export A11Y_METHOD="${A11Y_METHOD:-fast_provider}"
export MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-300}"
export MAX_STEPS="${MAX_STEPS:-16}"

echo "[fast-latency-116] forcing A11Y_METHOD=${A11Y_METHOD}"
echo "[fast-latency-116] exp_root=${EXP_ROOT}"
echo "[fast-latency-116] latency report will be generated at run/report/latency_breakdown/latency_breakdown_cn.md"

exec bash "${ROOT_DIR}/scripts/run_lb_mcts_116task_stable_foreground.sh"
