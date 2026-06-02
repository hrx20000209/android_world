#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/huangrunxi/Projects/android_world"
cd "$ROOT"

TASK_COUNT="${TASK_COUNT:-50}"
VARIANTS="${VARIANTS:-baseline,main}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-results/decoupled_parallel_50task}"
CHECK_HOST="${CHECK_HOST:-127.0.0.1}"
CHECK_PORT="${CHECK_PORT:-8081}"
CHECK_INTERVAL_S="${CHECK_INTERVAL_S:-60}"
RUN_ID="${RUN_ID:-auto_$(date +%Y%m%dT%H%M%S)}"
LOG_DIR="$ROOT/$EXPERIMENT_ROOT/$RUN_ID"
mkdir -p "$LOG_DIR"
STATUS_LOG="$LOG_DIR/watcher_status.log"
RUN_LOG="$LOG_DIR/runner.log"

printf '[%s] Waiting for OpenAI-compatible LLM server at %s:%s\n' "$(date -Is)" "$CHECK_HOST" "$CHECK_PORT" | tee -a "$STATUS_LOG"

while true; do
  if python3 - "$CHECK_HOST" "$CHECK_PORT" <<'PY' >/dev/null 2>&1
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2.0)
s.connect((host, port))
s.close()
PY
  then
    if curl -sf --max-time 5 "http://${CHECK_HOST}:${CHECK_PORT}/v1/models" > "$LOG_DIR/llm_models.json" 2>>"$STATUS_LOG"; then
      printf '[%s] /v1/models preflight passed. Starting experiment.\n' "$(date -Is)" | tee -a "$STATUS_LOG"
      break
    fi
    printf '[%s] TCP port open but /v1/models not ready; retrying.\n' "$(date -Is)" | tee -a "$STATUS_LOG"
  else
    printf '[%s] LLM server unavailable; retrying in %ss.\n' "$(date -Is)" "$CHECK_INTERVAL_S" | tee -a "$STATUS_LOG"
  fi
  sleep "$CHECK_INTERVAL_S"
done

printf '[%s] Command: python3 scripts/run_50task_decoupled_exploration.py --task_count %s --variants %s --experiment_root %s\n' "$(date -Is)" "$TASK_COUNT" "$VARIANTS" "$EXPERIMENT_ROOT" | tee -a "$STATUS_LOG"
python3 scripts/run_50task_decoupled_exploration.py \
  --task_count "$TASK_COUNT" \
  --variants "$VARIANTS" \
  --experiment_root "$EXPERIMENT_ROOT" \
  > "$RUN_LOG" 2>&1
rc=$?
printf '[%s] Experiment exited with code %s. Runner log: %s\n' "$(date -Is)" "$rc" "$RUN_LOG" | tee -a "$STATUS_LOG"
exit "$rc"
