#!/usr/bin/env bash
# Naive-alternative ablations of full_rec (rebuttal experiment). Each arm keeps
# full_rec and swaps exactly one design for its naive counterpart:
#   naive_graph   - nodes keyed by the exact screen instance (struct_sig, text and
#                   16px positions included), edges keyed by raw tap coordinates,
#                   procedure hops replayed at the demonstrated coordinates.
#   naive_explore - probe candidates drawn uniformly at random from the same safe
#                   set (same budget, same rollback), no task-entity targeting.
#   naive_use     - graph knowledge only as prompt text, AutoDroid-style (one line
#                   per control with a known destination, every step); no
#                   procedure replay and no graph skip.
#   full_handoff  - full_rec plus the exploration->replay hand-off: when replay is
#                   refused because the next entity is off screen, take the probe
#                   action that exploration saw reveal it.
#   usage: scripts/run_task_navigation_pilot4.sh OUT_ROOT SEED ARM TASK [TASK...]
set -u
out_root="${1:?OUT_ROOT}"; seed="${2:?SEED}"; arm="${3:?ARM}"; shift 3
api_url="${API_URL:-http://127.0.0.1:8000/v1/chat/completions}"

common=(--seed "$seed" --max_steps 0 --api_url "$api_url"
        --no_decision_constraints --graph_context off
        --exploration_budget_s auto --max_depth 2 ${STABLE:+--stable_screenshot})
explore=(--probes_per_step 3 --max_depth 2)
checker=(--state_checker on --probe_only_certifiable)
full=("${explore[@]}" --enable_skip --graph_context distill
      --procedure_replay execute --procedure_records "${checker[@]}" --probe_entity_rows)
naive_env=0
case "$arm" in
  base)          flags=(--probes_per_step 0) ;;
  full_rec)      flags=("${full[@]}") ;;
  naive_graph)   flags=("${full[@]}"); naive_env=1 ;;
  full_handoff)  flags=("${full[@]}" --explore_handoff) ;;
  naive_explore) flags=("${explore[@]}" --exploration_policy random --enable_skip --graph_context distill
                        --procedure_replay execute --procedure_records "${checker[@]}") ;;
  naive_use)     flags=("${explore[@]}" --graph_context onclick --procedure_replay off
                        "${checker[@]}" --probe_entity_rows) ;;
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
  ANDROID_WORLD_NAIVE_GRAPH=$naive_env python scripts/run_serial_exploration_task.py --task "$task" \
    --output "$out_root/$arm/$task" "${common[@]}" "${flags[@]}" > "$log" 2>&1
  echo "FINISH $arm $task rc=$? $(date +%H:%M:%S)"
done
