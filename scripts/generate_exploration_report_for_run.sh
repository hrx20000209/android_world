#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir> [extra report args...]" >&2
  exit 2
fi

run_dir="$1"
shift

python scripts/run_exploration_experiment_report.py \
  --no-run \
  --checkpoint_dir="${run_dir}/checkpoints" \
  --trace_root="${run_dir}/traces" \
  --report_dir="${run_dir}/report" \
  --baseline_table=results/4B_2.txt \
  --max_cases=20 \
  "$@"
