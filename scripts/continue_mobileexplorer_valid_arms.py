#!/usr/bin/env python3
"""Resume the valid same-primary MobileExplorer arms and finish the study."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world import checkpointer  # pylint: disable=wrong-import-position,import-error
from android_world import registry  # pylint: disable=wrong-import-position,import-error


def _count(path: Path) -> int:
  if not path.is_dir():
    return 0
  return len({
      str(row.get("task_template"))
      for row in checkpointer.IncrementalCheckpointer(str(path)).load()
  })


def _alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  return True


def _run_arm(
    root: Path, arm: str, agent: str, memory: Path, checkpoint: Path
) -> None:
  checkpoint.mkdir(parents=True, exist_ok=True)
  env = os.environ.copy()
  env.update({
      "ANDROID_WORLD_LLM_API_URL": "http://127.0.0.1:18083/v1/chat/completions",
      "ANDROID_WORLD_LLAMACPP_MODEL": "GELAB-ZERO-4B",
      "MOBILEEXPLORER_TRAJECTORY_MEMORY": str(memory),
      "MOBILEEXPLORER_OUTPUT_PATH": str(root / f"{arm}_agent_logs"),
  })
  subprocess.run([
      sys.executable, str(REPO_ROOT / "run.py"),
      "--suite_family=android_world", f"--agent_name={agent}",
      "--n_task_combinations=1", "--task_random_seed=4242",
      "--console_port=5554", f"--checkpoint_dir={checkpoint}",
  ], cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--offline-pid", type=int, required=True)
  args = parser.parse_args()
  root = args.root.resolve()
  expected = len(registry.TaskRegistry().get_registry(family="android_world"))
  memory = root / "trajectory_memory_frozen.jsonl"
  offline = root / "offline_trajectory" / "checkpoints"

  while _alive(args.offline_pid):
    print(f"offline_trajectory={_count(offline)}/{expected}", flush=True)
    time.sleep(60)
  if _count(offline) < expected:
    _run_arm(root, "offline_trajectory", "mobileexplorer_offline", memory, offline)
  if _count(offline) != expected:
    raise RuntimeError("offline trajectory did not reach full coverage")

  replay = root / "offline_replay" / "checkpoints"
  _run_arm(root, "offline_replay", "mobileexplorer_replay", memory, replay)
  if _count(replay) != expected:
    raise RuntimeError("offline replay did not reach full coverage")

  mobile = root / "mobileexplorer"
  subprocess.run([
      sys.executable, str(REPO_ROOT / "scripts/run_mobileexplorer_full_suite.py"),
      "--memory", str(memory), "--root", str(mobile),
      "--task-random-seed", "4242", "--primary-console-port", "5554",
      "--shadow-console-port", "5556", "--shadow-grpc-port", "8556",
      "--primary-api-url", "http://127.0.0.1:18083/v1/chat/completions",
      "--shadow-api-url", "http://127.0.0.1:18084/v1/chat/completions",
  ], cwd=REPO_ROOT, check=True)
  if _count(mobile / "checkpoints") != expected:
    raise RuntimeError("MobileExplorer did not reach full coverage")
  subprocess.run([
      sys.executable, str(REPO_ROOT / "scripts/analyze_mobileexplorer_full.py"),
      "--root", str(root), "--output-json", str(root / "final.json"),
  ], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
  main()
