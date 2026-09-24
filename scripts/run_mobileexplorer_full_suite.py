#!/usr/bin/env python3
"""Run all AndroidWorld templates with live probes only on eligible IR tasks.

Each primary task is a separate resumable run.py invocation sharing one
checkpoint directory.  A shadow worker is started only when the task is an
information-retrieval template and an exact successful source trajectory
exists.  Primary completion cancels any unfinished worker; it never waits for
speculation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world import registry  # pylint: disable=wrong-import-position,import-error
from android_world.task_evals.information_retrieval import information_retrieval  # pylint: disable=wrong-import-position,import-error


def _memory_templates(path: Path) -> set[str]:
  result = set()
  with path.open(encoding="utf-8") as stream:
    for line in stream:
      if line.strip():
        row = json.loads(line)
        if row.get("successful") and row.get("task_template"):
          result.add(str(row["task_template"]))
  return result


def _append(path: Path, row: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stop_worker(worker: subprocess.Popen[str]) -> str:
  if worker.poll() is not None:
    if worker.returncode == 0:
      return "completed"
    if worker.returncode == 75:
      return "deadline_missed"
    return "failed"
  worker.terminate()
  try:
    worker.wait(timeout=5)
  except subprocess.TimeoutExpired:
    worker.kill()
    worker.wait(timeout=5)
  return "deadline_cancelled"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--memory", type=Path, required=True)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--task-random-seed", type=int, required=True)
  parser.add_argument("--primary-console-port", type=int, default=5554)
  parser.add_argument("--shadow-console-port", type=int, default=5556)
  parser.add_argument("--shadow-grpc-port", type=int, default=8556)
  parser.add_argument("--primary-api-url", required=True)
  parser.add_argument("--shadow-api-url", required=True)
  parser.add_argument("--model", default="GELAB-ZERO-4B")
  parser.add_argument("--tasks", nargs="*")
  args = parser.parse_args()

  task_registry = registry.TaskRegistry().get_registry(family="android_world")
  selected = args.tasks or list(task_registry)
  unknown = set(selected) - set(task_registry)
  if unknown:
    raise ValueError(f"Unknown tasks: {sorted(unknown)}")
  ir_tasks = {
      name for name, task_type in task_registry.items()
      if issubclass(task_type, information_retrieval.InformationRetrieval)
  }
  memory_tasks = _memory_templates(args.memory)
  checkpoint_dir = args.root / "checkpoints"
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  status_path = args.root / "probe_runs.jsonl"

  for index, task in enumerate(selected, 1):
    if (checkpoint_dir / f"{task}_0.pkl.gz").is_file():
      continue
    task_dir = args.root / "live" / task
    opportunity = task_dir / "opportunities.jsonl"
    evidence = task_dir / "evidence.jsonl"
    eligible = task in ir_tasks and task in memory_tasks
    worker = None
    worker_log = None
    started_s = time.time()
    if eligible:
      task_dir.mkdir(parents=True, exist_ok=True)
      worker_log = (task_dir / "shadow.log").open("a", encoding="utf-8")
      worker = subprocess.Popen(
          [
              sys.executable, str(REPO_ROOT / "scripts/run_shadow_oracle_probe.py"),
              "--task", task,
              "--task-random-seed", str(args.task_random_seed),
              "--trajectory-memory", str(args.memory),
              "--opportunity-jsonl", str(opportunity),
              "--evidence-jsonl", str(evidence),
              "--console-port", str(args.shadow_console_port),
              "--grpc-port", str(args.shadow_grpc_port),
              "--llm-api-url", args.shadow_api_url,
              "--model", args.model,
              "--timeout-s", "180",
          ],
          cwd=REPO_ROOT,
          stdout=worker_log,
          stderr=subprocess.STDOUT,
          text=True,
      )
      time.sleep(.5)

    primary_env = os.environ.copy()
    primary_env.update({
        "ANDROID_WORLD_LLM_API_URL": args.primary_api_url,
        "ANDROID_WORLD_LLAMACPP_MODEL": args.model,
        "MOBILEEXPLORER_TRAJECTORY_MEMORY": str(args.memory),
        "MOBILEEXPLORER_EVIDENCE_INBOX": str(evidence),
        "MOBILEEXPLORER_OPPORTUNITY_PATH": str(opportunity),
        "MOBILEEXPLORER_OUTPUT_PATH": str(args.root / "agent_logs"),
    })
    primary_result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "run.py"),
            "--suite_family=android_world",
            "--agent_name=mobileexplorer_with_replay",
            f"--tasks={task}",
            "--n_task_combinations=1",
            f"--task_random_seed={args.task_random_seed}",
            f"--console_port={args.primary_console_port}",
            f"--checkpoint_dir={checkpoint_dir}",
        ],
        cwd=REPO_ROOT,
        env=primary_env,
        check=False,
    )
    worker_status = "not_eligible"
    if worker is not None:
      worker_status = _stop_worker(worker)
      if worker_log is not None:
        worker_log.close()
    _append(status_path, {
        "task": task,
        "index": index,
        "eligible": eligible,
        "primary_returncode": primary_result.returncode,
        "worker_status": worker_status,
        "evidence_created": evidence.is_file() and evidence.stat().st_size > 0,
        "elapsed_s": time.time() - started_s,
        "finished_at_s": time.time(),
    })
    if primary_result.returncode != 0:
      raise RuntimeError(f"Primary run failed for {task}: {primary_result.returncode}")


if __name__ == "__main__":
  main()
