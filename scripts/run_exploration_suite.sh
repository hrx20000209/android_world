#!/bin/bash
# Fast validation for exploration-side changes.
#
# Exploration runs a probe on only 51 of 115 AndroidWorld tasks (measured over
# five full runs), so the other 64 cannot register an exploration change and
# cost ~2.2h of device time to confirm that. This runs just the 51. Use the
# full suite for anything that touches the prompt on every step.
#
#   scripts/run_exploration_suite.sh <out_dir> <store_dir> [extra args...]
#
# SUITE= picks the task list:
#   configs/exploration_active_suite.json    (default, 51) - for changes to HOW
#       exploration ranks or probes, which cannot move a task it never visits.
#   configs/exploration_coverage_suite.json  (76) - for changes to WHERE
#       exploration runs; adds the 25 longest tasks it currently never reaches.
# A change to the prompt still needs the full 116-task suite: it touches every
# step, including the 39 short tasks neither list contains.
set -u
OUT="$1"; STORE="$2"; shift 2
cd "$(dirname "$0")/.."
API=${API:-http://localhost:8084/v1/chat/completions}
export ANDROID_WORLD_A11Y_METHOD=fast_provider
export ANDROID_WORLD_FAST_A11Y_SOCKET_PORT=8765
mkdir -p "$OUT" "$STORE"
SUITE=${SUITE:-configs/exploration_active_suite.json}
TASKS=$(python3 -c "
import json,sys; print(' '.join(json.load(open(sys.argv[1]))['tasks']))" "$SUITE")
for T in $TASKS; do
  compgen -G "$OUT/$T/run_*" > /dev/null && continue
  adb -s ${ANDROID_SERIAL:-emulator-5554} forward tcp:8765 localabstract:androidworld_fast_a11y >/dev/null 2>&1
  # The SMS role drifts to Google Messages between runs; when it does every
  # SimpleSms* task launches into RequestRoleActivity and never reaches the app.
  adb -s ${ANDROID_SERIAL:-emulator-5554} shell cmd role add-role-holder android.app.role.SMS com.simplemobiletools.smsmessenger >/dev/null 2>&1
  python scripts/run_serial_exploration_task.py --task "$T" --output "$OUT/$T" \
    --max_steps 15 --app_memory "$STORE" --api_url "$API" "$@" \
    > "$OUT/$T.log" 2>&1
done
echo EXPLORATION_SUITE_DONE
