#!/usr/bin/env python3
"""Run one AndroidWorld arm with a full AVD snapshot before every task.

This is deliberately separate from the legacy app-private-snapshot study.
The named AVD snapshot must already exist and be verified by the operator.
Each task gets its own run.py process so no prior episode can change its
starting device state. Existing checkpoints are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world import registry  # pylint: disable=wrong-import-position,import-error
from android_world import checkpointer  # pylint: disable=wrong-import-position,import-error


def _adb(adb: Path, serial: str, *args: str) -> str:
  result = subprocess.run(
      [str(adb), "-s", serial, *args], check=True, capture_output=True,
      text=True, timeout=90,
  )
  return result.stdout


def _restore(adb: Path, serial: str, snapshot: str) -> float:
  start = time.monotonic()
  output = _adb(adb, serial, "emu", "avd", "snapshot", "load", snapshot)
  if "OK" not in output:
    raise RuntimeError(f"Snapshot load did not confirm success: {output!r}")
  _adb(adb, serial, "wait-for-device")
  for _ in range(30):
    if _adb(adb, serial, "shell", "getprop", "sys.boot_completed").strip() == "1":
      return time.monotonic() - start
    time.sleep(1)
  raise RuntimeError("Emulator did not complete boot after snapshot load")


def _paired_outcomes(
    checkpoint_dir: Path, baseline_dir: Path
) -> tuple[int, int, int]:
  """Count strictly seed/goal-paired arm wins and losses to no-memory."""
  def rows(directory: Path) -> dict[str, dict]:
    if not directory.is_dir():
      return {}
    return {
        str(row.get("task_template")): row
        for row in checkpointer.IncrementalCheckpointer(str(directory)).load()
    }
  baseline = rows(baseline_dir)
  arm = rows(checkpoint_dir)
  paired = wins = losses = 0
  for task, other in arm.items():
    reference = baseline.get(task)
    if reference is None or (
        reference.get("seed") != other.get("seed")
        or reference.get("goal") != other.get("goal")
    ):
      continue
    paired += 1
    a = float(reference.get("is_successful") or 0) > .5
    b = float(other.get("is_successful") or 0) > .5
    wins += int(b and not a)
    losses += int(a and not b)
  return paired, wins, losses


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--arm", required=True)
  parser.add_argument("--agent", required=True)
  parser.add_argument("--snapshot", required=True)
  parser.add_argument("--adb", type=Path, default=Path("adb"))
  parser.add_argument("--console-port", type=int, default=5554)
  parser.add_argument("--seed", type=int, default=4242)
  parser.add_argument("--api-url", default="http://127.0.0.1:18083/v1/chat/completions")
  parser.add_argument("--model", default="GELAB-ZERO-4B")
  parser.add_argument("--memory", type=Path)
  parser.add_argument("--tasks", nargs="*")
  parser.add_argument("--limit", type=int)
  parser.add_argument("--paired-baseline-checkpoints", type=Path)
  parser.add_argument("--min-pairs-before-gate", type=int, default=4)
  parser.add_argument("--max-net-regressions", type=int, default=2)
  args = parser.parse_args()

  if args.memory and not args.memory.is_file():
    raise FileNotFoundError(args.memory)
  serial = f"emulator-{args.console_port}"
  snapshots = _adb(args.adb, serial, "emu", "avd", "snapshot", "list")
  if args.snapshot not in snapshots:
    raise RuntimeError(f"Snapshot {args.snapshot!r} not present on {serial}")
  all_tasks = registry.TaskRegistry().get_registry(family="android_world")
  tasks = args.tasks or list(all_tasks)
  unknown = sorted(set(tasks) - set(all_tasks))
  if unknown:
    raise ValueError(f"Unknown tasks: {unknown}")
  if args.limit is not None:
    tasks = tasks[:args.limit]

  arm_root = args.root.resolve() / args.arm
  checkpoints = arm_root / "checkpoints"
  logs = arm_root / "run_logs"
  checkpoints.mkdir(parents=True, exist_ok=True)
  logs.mkdir(parents=True, exist_ok=True)
  status = arm_root / "snapshot_arm_status.jsonl"
  env = os.environ.copy()
  env["ANDROID_WORLD_LLM_API_URL"] = args.api_url
  env["ANDROID_WORLD_LLAMACPP_MODEL"] = args.model
  env["MOBILEEXPLORER_OUTPUT_PATH"] = str(arm_root / "agent_logs")
  if args.memory:
    env["MOBILEEXPLORER_TRAJECTORY_MEMORY"] = str(args.memory.resolve())
  else:
    env.pop("MOBILEEXPLORER_TRAJECTORY_MEMORY", None)

  if args.paired_baseline_checkpoints:
    paired, wins, losses = _paired_outcomes(
        checkpoints, args.paired_baseline_checkpoints
    )
    if (paired >= args.min_pairs_before_gate
        and losses - wins >= args.max_net_regressions):
      raise SystemExit(
          f"Existing checkpoints already cross the performance gate: "
          f"paired={paired}, arm_only_wins={wins}, baseline_only_wins={losses}. "
          "Inspect implementation before resuming."
      )

  for task in tasks:
    checkpoint = checkpoints / f"{task}_0.pkl.gz"
    if checkpoint.exists():
      print(f"Skipping completed {task}", flush=True)
      continue
    restore_s = _restore(args.adb, serial, args.snapshot)
    event = {
        "time_s": time.time(), "task": task, "arm": args.arm,
        "snapshot": args.snapshot, "snapshot_restore_s": restore_s,
        "status": "started",
    }
    with status.open("a", encoding="utf-8") as stream:
      stream.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)
    command = [
        sys.executable, str(REPO_ROOT / "run.py"),
        "--suite_family=android_world", f"--agent_name={args.agent}",
        "--n_task_combinations=1", f"--task_random_seed={args.seed}",
        f"--console_port={args.console_port}", f"--tasks={task}",
        f"--checkpoint_dir={checkpoints}",
    ]
    with (logs / f"{task}.log").open("w", encoding="utf-8") as output:
      result = subprocess.run(
          command, cwd=REPO_ROOT, env=env, stdout=output,
          stderr=subprocess.STDOUT, check=False,
      )
    event.update({
        "time_s": time.time(), "status": "complete" if result.returncode == 0
        and checkpoint.is_file() else "failed", "returncode": result.returncode,
        "checkpoint": str(checkpoint),
    })
    with status.open("a", encoding="utf-8") as stream:
      stream.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)
    if event["status"] != "complete":
      raise RuntimeError(f"{task} failed; see {logs / (task + '.log')}")
    if args.paired_baseline_checkpoints:
      paired, wins, losses = _paired_outcomes(
          checkpoints, args.paired_baseline_checkpoints
      )
      if (paired >= args.min_pairs_before_gate
          and losses - wins >= args.max_net_regressions):
        gate_event = {
            "time_s": time.time(), "arm": args.arm,
            "status": "performance_gate_paused", "paired": paired,
            "arm_only_wins": wins, "baseline_only_wins": losses,
            "reason": "Inspect implementation before expanding this arm.",
        }
        with status.open("a", encoding="utf-8") as stream:
          stream.write(json.dumps(gate_event) + "\n")
        print(json.dumps(gate_event), flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
  main()
