#!/usr/bin/env python3
"""Run a fixed MobileExplorer task manifest and compare it with 4B_2."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]


def _result(task_root: Path) -> dict[str, Any] | None:
  files = sorted(task_root.glob("androidworld_run/run_*/*.pkl.gz"))
  if not files:
    return None
  with gzip.open(files[-1], "rb") as stream:
    rows = pickle.load(stream)
  return rows[0] if rows else None


def _jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--manifest", type=Path, default=REPO / "configs/two_system_baseline_gate_10tasks.json")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--agent_name", choices=["gelab_agent", "m3a_llamacpp"], default="gelab_agent")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--skip_confidence", type=float, default=0.86)
  parser.add_argument("--max_probes", type=int, default=20)
  parser.add_argument("--max_depth", type=int, default=3)
  parser.add_argument("--inject_evidence", action="store_true")
  args = parser.parse_args()
  manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
  root = args.output.resolve()
  root.mkdir(parents=True, exist_ok=True)
  rows = []
  for spec in manifest["tasks"]:
    task = spec["name"]
    task_root = root / task
    existing = _result(task_root) if args.resume else None
    if existing is None:
      command = [
          sys.executable, str(REPO / "scripts/run_one_parallel_exploration_task.py"),
          "--output", str(task_root), "--task", task,
          "--max_steps", str(spec["max_steps"]), "--seed", str(manifest["seed"]),
          "--two_system", "--enable_skip_inference",
          "--skip_confidence", str(args.skip_confidence),
          "--agent_name", args.agent_name,
          "--a11y_method", "fast_provider", "--max_probes", str(args.max_probes),
          "--min_probes", "2", "--post_inference_grace_s", "0",
          "--max_depth", str(args.max_depth), "--max_exploration_time_s", "8",
      ] + (["--inject_evidence"] if args.inject_evidence else [])
      completed = subprocess.run(command, cwd=REPO, check=False)
      existing = _result(task_root)
      returncode = completed.returncode
    else:
      returncode = 0
    requests = _jsonl(task_root / "request_latency.jsonl")
    windows = _jsonl(task_root / "inference_windows.jsonl")
    skip_events = _jsonl(task_root / "skip_events.jsonl")
    row = {
        "task": task, "baseline_success": int(spec["baseline_success"]),
        "baseline_steps": int(spec["baseline_steps"]),
        "success": float(existing.get("is_successful", 0.0)) if existing else 0.0,
        "episode_length": int(existing.get("episode_length", 0)) if existing else 0,
        "mean_step_latency_s": (existing.get("aux_data") or {}).get("mean_step_latency_sec") if existing else None,
        "requests": len(requests),
        "inference_calls": len(requests),
        "inference_skips": len(skip_events),
        "verified_skips": sum(bool(item.get("successor_matched")) for item in skip_events),
        "prefill_mean_s": sum(x.get("prefill_s", 0.0) for x in requests) / len(requests) if requests else None,
        "decode_mean_s": sum(x.get("decode_s", 0.0) for x in requests) / len(requests) if requests else None,
        "inference_mean_s": sum(x.get("inference_s", 0.0) for x in requests) / len(requests) if requests else None,
        "probes": sum(int(x.get("probes_completed", 0)) for x in windows),
        "rollback_failures": sum(x.get("restore_status") != "RESTORED" for x in windows),
        "trajectory_replays": sum(x.get("abort_recovery_level") == "TRAJECTORY_REPLAY" for x in windows),
        "returncode": returncode,
    }
    rows.append(row)
    (root / "partial_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

  aggregate = {
      "num_tasks": len(rows),
      "baseline_success_rate": sum(x["baseline_success"] for x in rows) / len(rows),
      "success_rate": sum(x["success"] for x in rows) / len(rows),
      "baseline_total_steps": sum(x["baseline_steps"] for x in rows),
      "total_episode_steps": sum(x["episode_length"] for x in rows),
      "total_inference_calls": sum(x["inference_calls"] for x in rows),
      "total_inference_skips": sum(x["inference_skips"] for x in rows),
      "verified_inference_skips": sum(x["verified_skips"] for x in rows),
      "baseline_success_subset_rate": sum(x["success"] for x in rows if x["baseline_success"]) / max(1, sum(x["baseline_success"] for x in rows)),
      "baseline_failure_subset_new_successes": sum(x["success"] for x in rows if not x["baseline_success"]),
      "total_probes": sum(x["probes"] for x in rows),
      "rollback_failures": sum(x["rollback_failures"] for x in rows),
      "trajectory_replays": sum(x["trajectory_replays"] for x in rows),
  }
  (root / "comparison.json").write_text(json.dumps({"aggregate": aggregate, "tasks": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
  print(json.dumps(aggregate, ensure_ascii=False, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
