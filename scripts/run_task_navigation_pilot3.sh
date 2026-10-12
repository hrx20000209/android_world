#!/usr/bin/env bash
# Module ablations of full_rec (explainability experiment). Standalone copy of
# the pilot2 launcher so that file stays frozen while r6 imports/uses it.
#   usage: scripts/run_task_navigation_pilot3.sh OUT_ROOT SEED ARM TASK [TASK...]
# Every ablation is full_rec with exactly one module removed.
set -u
out_root="${1:?OUT_ROOT}"; seed="${2:?SEED}"; arm="${3:?ARM}"; shift 3
api_url="${API_URL:-http://127.0.0.1:8000/v1/chat/completions}"

common=(--seed "$seed" --max_steps 0 --api_url "$api_url"
        --no_decision_constraints --graph_context off
        --exploration_budget_s auto --max_depth 2 ${STABLE:+--stable_screenshot} ${DRAG:+--deterministic_swipe})
explore=(--probes_per_step 3 --max_depth 2)
skip=(--enable_skip)
context=(--graph_context distill)
replay=(--procedure_replay execute --procedure_records)
aim=(--probe_only_certifiable --probe_entity_rows)
checker=(--state_checker on)
case "$arm" in
  base)               flags=(--probes_per_step 0) ;;
  full_rec)           flags=("${explore[@]}" "${skip[@]}" "${context[@]}" "${replay[@]}" "${checker[@]}" "${aim[@]}") ;;
  no_replay)          flags=("${explore[@]}" "${skip[@]}" "${context[@]}" --procedure_replay off "${checker[@]}" "${aim[@]}") ;;
  no_skip)            flags=("${explore[@]}" "${context[@]}" "${replay[@]}" "${checker[@]}" "${aim[@]}") ;;
  no_context)         flags=("${explore[@]}" "${skip[@]}" "${replay[@]}" "${checker[@]}" "${aim[@]}") ;;
  no_explore)         flags=(--probes_per_step 0 "${skip[@]}" "${context[@]}" "${replay[@]}" "${checker[@]}") ;;
  explore_untargeted) flags=("${explore[@]}" "${skip[@]}" "${context[@]}" "${replay[@]}" "${checker[@]}") ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac

mkdir -p "$out_root/$arm"
for task in "$@"; do
  log="$out_root/$arm/$task.log"
  adb -s emulator-5554 forward tcp:8765 localabstract:androidworld_fast_a11y >/dev/null 2>&1
  if [[ -f "$log" ]] && grep -q "Global mean step latency" "$log"; then
    echo "SKIP $arm $task"; continue
  fi
  rm -rf "$out_root/$arm/$task"
  echo "START $arm $task $(date +%H:%M:%S)"
  python scripts/run_serial_exploration_task.py --task "$task" \
    --output "$out_root/$arm/$task" "${common[@]}" "${flags[@]}" > "$log" 2>&1
  echo "FINISH $arm $task rc=$? $(date +%H:%M:%S)"
done
