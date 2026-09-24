#!/usr/bin/env python3
"""Finish the leak-free full AndroidWorld MobileExplorer experiment.

This driver is deliberately resumable.  It adopts the two already-running
source/baseline jobs, tops up successful information-retrieval demonstrations
with independent seeds, builds one frozen trajectory store, runs both offline
conditions, and finally runs the dual-emulator live condition.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world import checkpointer  # pylint: disable=wrong-import-position,import-error
from android_world import registry  # pylint: disable=wrong-import-position,import-error
from android_world.task_evals.information_retrieval import information_retrieval  # pylint: disable=wrong-import-position,import-error


def _episodes(directory: Path) -> list[dict[str, Any]]:
  if not directory.is_dir():
    return []
  return checkpointer.IncrementalCheckpointer(str(directory)).load()


def _coverage(directory: Path) -> int:
  return len({str(row.get("task_template")) for row in _episodes(directory)})


def _alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  return True


def _log(status_path: Path, event: str, **values: Any) -> None:
  row = {"event": event, "time_s": time.time(), **values}
  status_path.parent.mkdir(parents=True, exist_ok=True)
  with status_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
  print(json.dumps(row, ensure_ascii=False), flush=True)


def _env(api_url: str, memory: Path | None, log_dir: Path) -> dict[str, str]:
  result = os.environ.copy()
  result.update({
      "ANDROID_WORLD_LLM_API_URL": api_url,
      "ANDROID_WORLD_LLAMACPP_MODEL": "GELAB-ZERO-4B",
      "MOBILEEXPLORER_OUTPUT_PATH": str(log_dir),
  })
  if memory is not None:
    result["MOBILEEXPLORER_TRAJECTORY_MEMORY"] = str(memory)
  return result


def _run(
    *, agent: str, seed: int, console_port: int, api_url: str,
    checkpoint_dir: Path, log_dir: Path, memory: Path | None = None,
    tasks: list[str] | None = None,
) -> None:
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  command = [
      sys.executable, str(REPO_ROOT / "run.py"),
      "--suite_family=android_world", f"--agent_name={agent}",
      "--n_task_combinations=1", f"--task_random_seed={seed}",
      f"--console_port={console_port}", f"--checkpoint_dir={checkpoint_dir}",
  ]
  if tasks:
    command.append(f"--tasks={','.join(tasks)}")
  subprocess.run(
      command, cwd=REPO_ROOT,
      env=_env(api_url, memory, log_dir), check=True,
  )


def _adopt_or_resume(
    *, pid: int, expected: int, checkpoint_dir: Path, agent: str, seed: int,
    console_port: int, api_url: str, log_dir: Path, status_path: Path,
) -> None:
  while _alive(pid):
    _log(status_path, "adopted_job_progress", pid=pid,
         checkpoint_count=_coverage(checkpoint_dir), expected=expected)
    time.sleep(60)
  observed = _coverage(checkpoint_dir)
  if observed < expected:
    _log(status_path, "resume_incomplete_job", pid=pid,
         checkpoint_count=observed, expected=expected)
    _run(
        agent=agent, seed=seed, console_port=console_port, api_url=api_url,
        checkpoint_dir=checkpoint_dir, log_dir=log_dir,
    )
  final = _coverage(checkpoint_dir)
  if final != expected:
    raise RuntimeError(f"Expected {expected} checkpoints in {checkpoint_dir}, got {final}")


def _successful_templates(directories: list[Path]) -> set[str]:
  result = set()
  for directory in directories:
    for row in _episodes(directory):
      if float(row.get("is_successful") or 0) > .5:
        result.add(str(row.get("task_template")))
  return result


def _compile_memory(
    source_dirs: list[Path], target_dir: Path, output: Path,
    status_path: Path,
) -> None:
  target_seeds = sorted({
      int(row.get("seed", -1)) for row in _episodes(target_dir)
      if int(row.get("seed", -1)) >= 0
  })
  if output.exists():
    _log(status_path, "reuse_frozen_memory", path=str(output))
    return
  command = [
      sys.executable, str(REPO_ROOT / "scripts/build_offline_trajectory_memory.py"),
      "--output-jsonl", str(output),
  ]
  for directory in source_dirs:
    command.extend(("--checkpoint-dir", str(directory)))
  for seed in target_seeds:
    command.extend(("--exclude-actual-seed", str(seed)))
  subprocess.run(command, cwd=REPO_ROOT, check=True)
  _log(status_path, "compiled_memory", path=str(output),
       source_directories=[str(path) for path in source_dirs],
       excluded_target_seeds=len(target_seeds))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--baseline-pid", type=int, required=True)
  parser.add_argument("--source-pid", type=int, required=True)
  parser.add_argument("--baseline-checkpoints", type=Path, required=True)
  parser.add_argument("--source-checkpoints", type=Path, required=True)
  parser.add_argument("--target-seed", type=int, default=4242)
  parser.add_argument("--source-seeds", type=int, nargs="*", default=[4240, 4239, 4238])
  parser.add_argument("--primary-api-url", default="http://127.0.0.1:18083/v1/chat/completions")
  parser.add_argument("--secondary-api-url", default="http://127.0.0.1:18084/v1/chat/completions")
  args = parser.parse_args()

  args.root = args.root.resolve()
  status = args.root / "continuation_status.jsonl"
  task_registry = registry.TaskRegistry().get_registry(family="android_world")
  ir_tasks = sorted(
      name for name, task_type in task_registry.items()
      if issubclass(task_type, information_retrieval.InformationRetrieval)
  )
  expected = len(task_registry)

  # Source owns the secondary emulator.  The target baseline can continue on
  # the primary emulator while source top-ups and offline arms run.
  _adopt_or_resume(
      pid=args.source_pid, expected=expected,
      checkpoint_dir=args.source_checkpoints, agent="mobileexplorer_no_memory",
      seed=4241, console_port=5556, api_url=args.secondary_api_url,
      log_dir=args.root / "memory_source_agent_logs", status_path=status,
  )

  source_dirs = [args.source_checkpoints]
  for seed in args.source_seeds:
    covered = _successful_templates(source_dirs)
    missing = sorted(set(ir_tasks) - covered)
    families = {name.split("Tasks", 1)[0] if name.startswith("Tasks") else
                name.split("Tracker", 1)[0] if name.startswith("SportsTracker") else
                "Notes" if name.startswith("Notes") else
                "Calendar" if name.startswith("SimpleCalendar") else name
                for name in covered & set(ir_tasks)}
    if len(covered & set(ir_tasks)) >= 5 and len(families) >= 2:
      break
    retry_dir = args.root / f"memory_source_ir_seed{seed}" / "checkpoints"
    _log(status, "source_topup_start", seed=seed, missing_ir=missing)
    _run(
        agent="mobileexplorer_no_memory", seed=seed, console_port=5556,
        api_url=args.secondary_api_url, checkpoint_dir=retry_dir,
        log_dir=args.root / f"memory_source_ir_seed{seed}_agent_logs",
        tasks=missing,
    )
    source_dirs.append(retry_dir)
    _log(status, "source_topup_end", seed=seed,
         successful_ir=sorted(_successful_templates(source_dirs) & set(ir_tasks)))

  # Wait for target seed completion before freezing memory so every target
  # episode seed can be rejected explicitly by the compiler.
  _adopt_or_resume(
      pid=args.baseline_pid, expected=expected,
      checkpoint_dir=args.baseline_checkpoints, agent="mobileexplorer_no_memory",
      seed=args.target_seed, console_port=5554, api_url=args.primary_api_url,
      log_dir=args.root / "no_memory_agent_logs", status_path=status,
  )
  memory = args.root / "trajectory_memory_frozen.jsonl"
  _compile_memory(source_dirs, args.baseline_checkpoints, memory, status)

  for arm, agent in (
      ("offline_trajectory", "mobileexplorer_offline"),
      ("offline_replay", "mobileexplorer_replay"),
  ):
    checkpoint_dir = args.root / arm / "checkpoints"
    _log(status, "arm_start", arm=arm, completed=_coverage(checkpoint_dir))
    _run(
        agent=agent, seed=args.target_seed, console_port=5556,
        api_url=args.secondary_api_url, checkpoint_dir=checkpoint_dir,
        log_dir=args.root / f"{arm}_agent_logs", memory=memory,
    )
    if _coverage(checkpoint_dir) != expected:
      raise RuntimeError(f"{arm} did not finish all {expected} templates")
    _log(status, "arm_complete", arm=arm)

  _log(status, "arm_start", arm="mobileexplorer")
  subprocess.run([
      sys.executable, str(REPO_ROOT / "scripts/run_mobileexplorer_full_suite.py"),
      "--memory", str(memory), "--root", str(args.root / "mobileexplorer"),
      "--task-random-seed", str(args.target_seed),
      "--primary-console-port", "5554", "--shadow-console-port", "5556",
      "--shadow-grpc-port", "8556", "--primary-api-url", args.primary_api_url,
      "--shadow-api-url", args.secondary_api_url,
  ], cwd=REPO_ROOT, check=True)
  if _coverage(args.root / "mobileexplorer" / "checkpoints") != expected:
    raise RuntimeError(f"mobileexplorer did not finish all {expected} templates")
  subprocess.run([
      sys.executable, str(REPO_ROOT / "scripts/analyze_mobileexplorer_full.py"),
      "--root", str(args.root), "--output-json", str(args.root / "final.json"),
  ], cwd=REPO_ROOT, check=True)
  _log(status, "experiment_complete", result=str(args.root / "final.json"))


if __name__ == "__main__":
  signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
  main()
