#!/usr/bin/env bash
# Launch the supervisor on LabServer so SSH disconnects cannot stop it.
set -euo pipefail

run_id="${1:?usage: $0 STUDY_ID}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$ ]]; then
  echo "invalid STUDY_ID" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
studies_root="/data/rxhuang/android_world_server/androidworld_graph_studies"
run_root="${studies_root}/${run_id}"
mkdir -p "$run_root"

if [[ -s "${run_root}/supervisor.pid" ]]; then
  pid="$(<"${run_root}/supervisor.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "already running: pid=${pid} run_root=${run_root}"
    exit 0
  fi
fi

nohup setsid python3 "${repo_root}/scripts/labserver_androidworld_graph_study.py" run \
  --run-id "$run_id" \
  --run-root "$run_root" \
  --repo "$repo_root" \
  --protocol "${repo_root}/experiments/labserver_androidworld_graph_study/protocol.json" \
  --reuse-vllm-port 8085 \
  --max-workers 3 \
  --failure-backoff-s 15 \
  --publish \
  >>"${run_root}/supervisor.log" 2>&1 </dev/null &
supervisor_pid=$!
printf '%s\n' "$supervisor_pid" >"${run_root}/supervisor.pid"
sleep 1
if ! kill -0 "$supervisor_pid" 2>/dev/null; then
  echo "supervisor exited during startup; inspect ${run_root}/supervisor.log" >&2
  exit 1
fi
echo "started: pid=${supervisor_pid} run_root=${run_root}"
