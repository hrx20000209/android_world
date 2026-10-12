#!/usr/bin/env bash
# Paired serial pilot for task-directed navigation. One arm x task per run;
# every arm shares the backbone, base prompt, bootstrap, seed and step cap.
#   usage: scripts/run_task_navigation_pilot.sh OUT_ROOT SEED ARM TASK [TASK...]
# Resumable: a task whose log already holds the result table is skipped.
set -u
out_root="${1:?OUT_ROOT}"; seed="${2:?SEED}"; arm="${3:?ARM}"; shift 3
api_url="${API_URL:-http://127.0.0.1:8000/v1/chat/completions}"

common=(--seed "$seed" --max_steps 0 --api_url "$api_url"
        --no_decision_constraints --graph_context off
        --exploration_budget_s auto --max_depth 2 ${STABLE:+--stable_screenshot} ${DRAG:+--deterministic_swipe})
explore=(--min_node_visits 1 --probes_per_step 2 --probe_budget 12)
case "$arm" in
  base)            flags=(--probes_per_step 0) ;;
  intent_only)     flags=(--probes_per_step 0 --task_navigation shadow) ;;
  shadow_frontier) flags=("${explore[@]}" --task_navigation shadow --task_navigation_frontier
                          --task_navigation_history --state_checker on) ;;
  history_nav)     flags=(--probes_per_step 0 --task_navigation execute
                          --task_navigation_history --state_checker on) ;;
  online_random)   flags=("${explore[@]}" --exploration_policy random --task_navigation execute
                          --state_checker on) ;;
  online_current)  flags=("${explore[@]}" --task_navigation execute --state_checker on) ;;
  online_frontier) flags=("${explore[@]}" --task_navigation execute --task_navigation_frontier
                          --state_checker on) ;;
  full_strict)     flags=("${explore[@]}" --task_navigation execute --task_navigation_frontier
                          --task_navigation_history --state_checker on) ;;
  full_ui_only)    flags=("${explore[@]}" --task_navigation execute --task_navigation_frontier
                          --task_navigation_history --task_navigation_ui_only) ;;
  # Warm history reuse. corpus writes the store on TRAINING seeds with a
  # checker verdict per executed click; warm_* read a frozen copy (never
  # written back) on held-out seeds. STORE must be set for all three.
  corpus)          flags=(--probes_per_step 0 --app_memory "${STORE:?STORE}" --state_checker on) ;;
  warm_prefill)    flags=(--probes_per_step 0 --app_memory "${STORE:?STORE}" --app_memory_readonly
                          --store_prefill) ;;
  warm_prefill_cert) flags=(--probes_per_step 0 --app_memory "${STORE:?STORE}" --app_memory_readonly
                          --store_prefill --prefill_require_state_cert) ;;
  warm_strict)     flags=(--probes_per_step 0 --app_memory "${STORE:?STORE}" --app_memory_readonly
                          --store_prefill --prefill_require_state_cert --prefill_strict_selector) ;;
  # Online, task-specific procedure graph (episode-local, cold).
  proc_shadow)     flags=(--probes_per_step 0 --procedure_replay shadow --state_checker on) ;;
  proc_execute)    flags=(--probes_per_step 0 --procedure_replay execute --state_checker on) ;;
  proc_ui_only)    flags=(--probes_per_step 0 --procedure_replay execute --procedure_allow_unverified) ;;
  # Original design (v29 best: parallel-window exploration + graph skip +
  # distilled graph context) plus the procedure graph, exploration aimed at it.
  v29)             flags=(--probes_per_step 3 --max_depth 2 --enable_skip --graph_context distill) ;;
  full)            flags=(--probes_per_step 3 --max_depth 2 --enable_skip --graph_context distill
                          --procedure_replay execute --state_checker on) ;;
  full_rec)        flags=(--probes_per_step 3 --max_depth 2 --enable_skip --graph_context distill
                          --procedure_replay execute --state_checker on --procedure_records
                          --probe_only_certifiable --probe_entity_rows) ;;
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
