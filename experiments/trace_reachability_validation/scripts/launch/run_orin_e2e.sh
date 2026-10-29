#!/bin/sh
# End-to-end run on the Orin vLLM (tunnel localhost:18097) with the co-located on-device load (Qwen2-VL-2B
# llama-server inside the emulator, restarted by the driver when missing), 116 tasks in the e2e116n order, arms
# interleaved per task. Re-running the same command resumes: finished (task, arm) pairs are skipped, error rows re-run.
# usage: run_orin_e2e.sh ARMS TAG
#   e2e116n : BASE,FULL45
#   e2e116nA: FULL45+RAND,FULL45+INJ,FULL45+NOSYS        (replacement ablations)
#   e2e116S2: FULL45+SEENR                               (latest: replay + EXPLORE_FIT + NAV_UNAMBIG; warm-up seenW1)
#   e2e116B : BASE,FULL45+SEENR0                         (same-session BASE vs the seenE2 replay rules)
cd "$(dirname "$0")/../.." || exit 1
ARMS=$1; TAG=$2
[ -n "$ARMS" ] && [ -n "$TAG" ] || { echo "usage: $0 ARMS TAG"; exit 1; }
nohup sh -c "E2E_TASKS=\$(cat reports/e2e116n_task_order.txt) E2E_REPORT_EVERY=1000 \
  python scripts/e2e_driver.py $ARMS $TAG >> logs/${TAG}_driver.log 2>&1" > /dev/null 2>&1 &
