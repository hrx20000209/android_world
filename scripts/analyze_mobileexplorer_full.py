#!/usr/bin/env python3
"""Coverage and paired metrics for the full MobileExplorer AndroidWorld study."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world import checkpointer  # pylint: disable=wrong-import-position,import-error
from android_world import registry  # pylint: disable=wrong-import-position,import-error
from android_world.task_evals.information_retrieval import information_retrieval  # pylint: disable=wrong-import-position,import-error


ARM_DIRECTORIES = {
    "no_memory": "no_memory",
    # Only the same-primary-emulator source is valid for the final study.  The
    # earlier secondary-emulator source is intentionally quarantined under an
    # ``invalid_secondary`` name and must never be merged here.
    "memory_source": "memory_source_primary_seed4241",
    "offline_trajectory": "offline_trajectory",
    "offline_navigation": "offline_navigation",
    "offline_replay": "offline_replay",
    "mobileexplorer": "mobileexplorer",
    "mobileexplorer_replay": "mobileexplorer_replay",
}
COMPARISON_ARMS = tuple(
    arm for arm in ARM_DIRECTORIES if arm not in {"no_memory", "memory_source"}
)


def _list(value: Any) -> list[Any]:
  return list(value) if isinstance(value, (list, tuple)) else []


def _episodes(directory: Path) -> list[dict[str, Any]]:
  runs = sorted(directory.glob("run_*"))
  direct_checkpoint = directory / "checkpoints"
  if direct_checkpoint.is_dir() and list(direct_checkpoint.glob("*.pkl.gz")):
    runs.append(direct_checkpoint)
  if list(directory.glob("*.pkl.gz")):
    runs.append(directory)
  if not runs:
    return []
  # Resumed runs should use --checkpoint_dir. If multiple independent runs are
  # present, retain the latest result for each task instead of double counting.
  by_task: dict[str, dict[str, Any]] = {}
  for run in runs:
    for episode in checkpointer.IncrementalCheckpointer(str(run)).load():
      by_task[str(episode.get("task_template", ""))] = episode
  return list(by_task.values())


def _row(
    arm: str,
    episode: dict[str, Any],
    ir_tasks: set[str],
    memory_templates: set[str],
) -> dict[str, Any]:
  raw_data = episode.get("episode_data")
  data = raw_data if isinstance(raw_data, dict) else {}
  responses = _list(data.get("response"))
  reasoning = [str(x) for x in _list(data.get("reasoning_mode"))]
  evidence_ids = _list(data.get("evidence_ids"))
  audits: Counter[str] = Counter()
  for audit in _list(data.get("evidence_audit")):
    if isinstance(audit, dict):
      audits.update({str(key): int(value) for key, value in audit.items()})
  latencies = [float(x) for x in _list(data.get("runner_step_latency_sec"))]
  policy_digests = {
      str(value) for value in _list(data.get("policy_digest")) if value
  }
  task = str(episode.get("task_template", ""))
  return {
      "arm": arm,
      "task": task,
      "goal": str(episode.get("goal", "")),
      "policy_digests": sorted(policy_digests),
      "information_retrieval": task in ir_tasks,
      "memory_available": task in memory_templates,
      "actual_seed": int(episode.get("seed", -1)),
      "success": float(episode.get("is_successful") or 0),
      "actions": int(episode.get("episode_length") or 0),
      "primary_vlm_calls": sum(isinstance(x, str) and bool(x.strip()) for x in responses),
      "replays": reasoning.count("replay"),
      "reasoning_skips": reasoning.count("skip"),
      "evidence_consumptions": sum(bool(x) for x in evidence_ids),
      "runtime_s": float(episode.get("run_time") or 0),
      "step_latency_total_s": sum(latencies),
      "exception": bool(episode.get("exception_info")),
      "evidence_audit": dict(audits),
  }


def _mean(values: list[float]) -> float | None:
  return statistics.fmean(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  position = (len(ordered) - 1) * percentile
  lower = math.floor(position)
  upper = math.ceil(position)
  if lower == upper:
    return ordered[lower]
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_mean_ci(
    values: list[float], *, samples: int = 5000, seed: int = 20260916
) -> list[float] | None:
  """Deterministic task-paired percentile bootstrap confidence interval."""
  if not values:
    return None
  if len(values) == 1:
    return [values[0], values[0]]
  rng = random.Random(seed)
  means = []
  for _ in range(samples):
    means.append(statistics.fmean(rng.choice(values) for _ in values))
  return [_percentile(means, .025), _percentile(means, .975)]


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  successes = [float(row["success"]) for row in rows]
  actions = [float(row["actions"]) for row in rows]
  calls = [float(row["primary_vlm_calls"]) for row in rows]
  runtimes = [float(row["runtime_s"]) for row in rows]
  evidence_rows = [row for row in rows if row["evidence_consumptions"]]
  return {
      "episodes": len(rows),
      "successes": sum(value > .5 for value in successes),
      "success_rate": _mean(successes),
      "success_rate_bootstrap_95ci": _bootstrap_mean_ci(successes),
      "mean_actions": _mean(actions),
      "mean_primary_vlm_calls": _mean(calls),
      "mean_runtime_s": _mean(runtimes),
      "runtime_p50_s": _percentile(runtimes, .50),
      "runtime_p95_s": _percentile(runtimes, .95),
      "evidence_consuming_episodes": len(evidence_rows),
      # A failed episode after evidence consumption is a warning, not proof
      # that the evidence itself was false; slot-level labels are separate.
      "evidence_consumed_failed_episodes": sum(
          float(row["success"]) <= .5 for row in evidence_rows
      ),
  }


def _probe_summary(root: Path) -> dict[str, Any]:
  live_root = root / "mobileexplorer"
  status_path = live_root / "probe_runs.jsonl"
  statuses = []
  if status_path.is_file():
    statuses = [
        json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
  evidence = []
  for path in live_root.glob("live/*/evidence.jsonl"):
    evidence.extend(
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
  eligible = [row for row in statuses if row.get("eligible")]
  worker_statuses = Counter(str(row.get("worker_status")) for row in eligible)
  probe_latencies = [float(row["probe_latency_s"]) for row in evidence]
  navigation_latencies = [float(row["navigation_latency_s"]) for row in evidence]
  extraction_latencies = [float(row["extraction_latency_s"]) for row in evidence]
  opportunity_latencies = [float(row["opportunity_to_evidence_s"]) for row in evidence]
  return {
      "tasks_logged": len(statuses),
      "eligible_probe_tasks": len(eligible),
      "evidence_created_tasks": sum(bool(row.get("evidence_created")) for row in eligible),
      "inference_window_hit_rate": (
          sum(bool(row.get("evidence_created")) for row in eligible) / len(eligible)
          if eligible else None
      ),
      "deadline_miss_rate": (
          (worker_statuses["deadline_cancelled"] + worker_statuses["deadline_missed"])
          / len(eligible) if eligible else None
      ),
      "worker_statuses": dict(worker_statuses),
      "probe_latency_p50_s": _percentile(probe_latencies, .50),
      "probe_latency_p95_s": _percentile(probe_latencies, .95),
      "navigation_latency_p50_s": _percentile(navigation_latencies, .50),
      "navigation_latency_p95_s": _percentile(navigation_latencies, .95),
      "extraction_latency_p50_s": _percentile(extraction_latencies, .50),
      "extraction_latency_p95_s": _percentile(extraction_latencies, .95),
      "opportunity_to_evidence_p50_s": _percentile(opportunity_latencies, .50),
      "opportunity_to_evidence_p95_s": _percentile(opportunity_latencies, .95),
  }


def _memory_summary(root: Path, expected: set[str], ir_tasks: set[str]) -> dict[str, Any]:
  path = root / "trajectory_memory_frozen.jsonl"
  rows = []
  if path.is_file():
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
  templates = {str(row.get("task_template")) for row in rows}
  return {
      "trajectories": len(rows),
      "task_templates": len(templates),
      "template_coverage": len(templates & expected) / len(expected),
      "information_retrieval_templates": len(templates & ir_tasks),
      "information_retrieval_coverage": len(templates & ir_tasks) / len(ir_tasks),
  }


def _memory_templates(root: Path) -> set[str]:
  path = root / "trajectory_memory_frozen.jsonl"
  if not path.is_file():
    return set()
  return {
      str(row.get("task_template"))
      for row in (
          json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
          if line.strip()
      )
  }


def analyze(root: Path) -> dict[str, Any]:
  task_registry = registry.TaskRegistry().get_registry(family="android_world")
  expected = set(task_registry)
  ir_tasks = {
      name for name, task_type in task_registry.items()
      if issubclass(task_type, information_retrieval.InformationRetrieval)
  }
  memory_templates = _memory_templates(root)
  rows = []
  coverage = {}
  for arm, dirname in ARM_DIRECTORIES.items():
    arm_rows = [
        _row(arm, value, ir_tasks, memory_templates)
        for value in _episodes(root / dirname)
    ]
    rows.extend(arm_rows)
    observed = {row["task"] for row in arm_rows}
    coverage[arm] = {
        "completed": len(observed),
        "expected": len(expected),
        "missing": sorted(expected - observed),
        "information_retrieval_completed": len(observed & ir_tasks),
        "memory_eligible_completed": len(observed & memory_templates),
    }
  summaries = {}
  for arm in ARM_DIRECTORIES:
    arm_rows = [row for row in rows if row["arm"] == arm]
    audit: Counter[str] = Counter()
    for row in arm_rows:
      audit.update(row["evidence_audit"])
    summaries[arm] = {
        "episodes": len(arm_rows),
        "successes": sum(row["success"] > .5 for row in arm_rows),
        "success_rate": (
            sum(row["success"] for row in arm_rows) / len(arm_rows)
            if arm_rows else None
        ),
        "actions": sum(row["actions"] for row in arm_rows),
        "primary_vlm_calls": sum(row["primary_vlm_calls"] for row in arm_rows),
        "replays": sum(row["replays"] for row in arm_rows),
        "reasoning_skips": sum(row["reasoning_skips"] for row in arm_rows),
        "evidence_consumptions": sum(row["evidence_consumptions"] for row in arm_rows),
        "runtime_s": sum(row["runtime_s"] for row in arm_rows),
        "exceptions": sum(row["exception"] for row in arm_rows),
        "policy_digests": sorted({
            digest for row in arm_rows for digest in row["policy_digests"]
        }),
        "episodes_without_policy_digest": sum(
            not row["policy_digests"] for row in arm_rows
        ),
        "evidence_rejections": dict(audit),
        "all": _group_summary(arm_rows),
        "information_retrieval": _group_summary([
            row for row in arm_rows if row["information_retrieval"]
        ]),
        "negative_controls": _group_summary([
            row for row in arm_rows if not row["information_retrieval"]
        ]),
        "memory_available": _group_summary([
            row for row in arm_rows if row["memory_available"]
        ]),
        "memory_unavailable": _group_summary([
            row for row in arm_rows if not row["memory_available"]
        ]),
        "replay_audit": {
            "eligible_episodes": sum(row["memory_available"] for row in arm_rows),
            "replayed_episodes": sum(row["replays"] > 0 for row in arm_rows),
            "replayed_failures": sum(
                row["replays"] > 0 and row["success"] <= .5 for row in arm_rows
            ),
            "replayed_failure_rate": (
                sum(row["replays"] > 0 and row["success"] <= .5 for row in arm_rows)
                / sum(row["replays"] > 0 for row in arm_rows)
                if any(row["replays"] > 0 for row in arm_rows) else None
            ),
            "ineligible_replay_violations": sum(
                row["replays"] > 0 and not row["memory_available"] for row in arm_rows
            ),
        },
    }
  by_arm_task = {(row["arm"], row["task"]): row for row in rows}
  paired = []
  for task in sorted(expected):
    baseline = by_arm_task.get(("no_memory", task))
    if baseline is None:
      continue
    item: dict[str, Any] = {"task": task, "information_retrieval": task in ir_tasks}
    for arm in COMPARISON_ARMS:
      other = by_arm_task.get((arm, task))
      if other is None:
        continue
      item[arm] = {
          "same_actual_seed": baseline["actual_seed"] == other["actual_seed"],
          "same_goal": baseline["goal"] == other["goal"],
          "success_delta": other["success"] - baseline["success"],
          "actions_delta": other["actions"] - baseline["actions"],
          "primary_vlm_calls_delta": (
              other["primary_vlm_calls"] - baseline["primary_vlm_calls"]
          ),
          "runtime_delta_s": other["runtime_s"] - baseline["runtime_s"],
      }
    if len(item) > 2:
      paired.append(item)
  negative = [
      row for row in rows
      if row["arm"] == "mobileexplorer"
      and not row["information_retrieval"]
      and (row["evidence_consumptions"] or row["reasoning_skips"])
  ]
  paired_summaries = {}
  for arm in COMPARISON_ARMS:
    pairs = []
    for task in sorted(expected):
      baseline = by_arm_task.get(("no_memory", task))
      other = by_arm_task.get((arm, task))
      if baseline is not None and other is not None:
        pairs.append((baseline, other))
    arm_result: dict[str, Any] = {
        "paired_tasks": len(pairs),
        "seed_mismatches": sum(a["actual_seed"] != b["actual_seed"] for a, b in pairs),
        "goal_mismatches": sum(a["goal"] != b["goal"] for a, b in pairs),
    }
    strict_pairs = [
        (a, b) for a, b in pairs
        if a["actual_seed"] == b["actual_seed"] and a["goal"] == b["goal"]
    ]
    arm_result["strictly_paired_tasks"] = len(strict_pairs)
    for group_name, predicate in (
        ("all", lambda row: True),
        ("information_retrieval", lambda row: bool(row["information_retrieval"])),
        ("negative_controls", lambda row: not bool(row["information_retrieval"])),
    ):
      selected = [(a, b) for a, b in strict_pairs if predicate(a)]
      group: dict[str, Any] = {"paired_tasks": len(selected)}
      for metric in ("success", "actions", "primary_vlm_calls", "runtime_s"):
        deltas = [float(b[metric]) - float(a[metric]) for a, b in selected]
        group[f"mean_{metric}_delta"] = _mean(deltas)
        group[f"mean_{metric}_delta_bootstrap_95ci"] = _bootstrap_mean_ci(deltas)
      arm_result[group_name] = group
    paired_summaries[arm] = arm_result
  return {
      "expected_task_templates": len(expected),
      "information_retrieval_templates": len(ir_tasks),
      "coverage": coverage,
      "summaries": summaries,
      "paired": paired,
      "paired_summaries": paired_summaries,
      "frozen_memory": _memory_summary(root, expected, ir_tasks),
      "probe_system": _probe_summary(root),
      "negative_control_violations": negative,
      "rows": rows,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--output-json", type=Path)
  args = parser.parse_args()
  result = analyze(args.root)
  encoded = json.dumps(result, ensure_ascii=False, indent=2)
  if args.output_json:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(encoded + "\n", encoding="utf-8")
  print(encoded)


if __name__ == "__main__":
  main()
