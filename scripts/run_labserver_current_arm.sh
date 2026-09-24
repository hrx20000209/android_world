#!/usr/bin/env bash

# Resumable sequential runner for the current MobileExplorer arm on a worker.
# Each task gets its own AndroidWorld output directory, while executable
# memory and semantic-prefix state are shared across tasks in this arm.
set -u

tasks_csv="${1:?usage: $0 TASKS_CSV OUTPUT_PREFIX [SEED]}"
output_prefix="${2:?usage: $0 TASKS_CSV OUTPUT_PREFIX [SEED]}"
seed="${3:-34}"
max_probes="${ME_MAX_PROBES:-5}"
min_probes="${ME_MIN_PROBES:-5}"
max_exploration_time_s="${ME_MAX_EXPLORATION_TIME_S:-8}"

IFS=',' read -r -a tasks <<< "${tasks_csv}"
shared_memory="${output_prefix}_shared_memory.json"
shared_prefix="${output_prefix}_shared_prefix.json"

for task in "${tasks[@]}"; do
  task="${task//[[:space:]]/}"
  [[ -z "${task}" ]] && continue
  output="${output_prefix}_${task}"
  log="${output}.log"
  checkpoint="${output}/checkpoints/${task}_0.pkl.gz"
  if [[ -f "${checkpoint}" ]]; then
    echo "SKIP ${task} (checkpoint exists)"
    continue
  fi
  echo "START ${task}"
  python3 scripts/run_sensys30_online_task.py \
    --output "${output}" \
    --task "${task}" \
    --max_steps 0 \
    --seed "${seed}" \
    --metrics_url "${ANDROID_WORLD_METRICS_URL:?ANDROID_WORLD_METRICS_URL is required}" \
    --console_port "${ANDROID_WORLD_CONSOLE_PORT:-5554}" \
    --a11y_method grpc \
    --agent_name mobileexplorer_executable \
    --variant full \
    --two_system \
    --executable_memory \
    --executable_memory_path "${shared_memory}" \
    --semantic_prefix_mode assist \
    --semantic_prefix_path "${shared_prefix}" \
    --max_probes "${max_probes}" \
    --min_probes "${min_probes}" \
    --max_exploration_time_s "${max_exploration_time_s}" \
    > "${log}" 2>&1
  status=$?
  echo "FINISH ${task} rc=${status}"
done
