#!/usr/bin/env python3
"""Audit per-task checkpoint integrity and policy homogeneity for one arm."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world import checkpointer  # pylint: disable=wrong-import-position,import-error
from android_world import registry  # pylint: disable=wrong-import-position,import-error


def audit(root: Path, arm: str) -> dict:
  arm_root = root / arm
  checkpoint_dir = arm_root / "checkpoints"
  episodes = list(checkpointer.IncrementalCheckpointer(str(checkpoint_dir)).load())
  expected = set(registry.TaskRegistry().get_registry(family="android_world"))
  observed = {str(row.get("task_template")) for row in episodes}
  status_path = arm_root / "snapshot_arm_status.jsonl"
  statuses = []
  if status_path.is_file():
    statuses = [
        json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
  completed_status = {
      str(row.get("task")) for row in statuses if row.get("status") == "complete"
  }
  digests: Counter[str] = Counter()
  unversioned = []
  incomplete = []
  errors = []
  for episode in episodes:
    task = str(episode.get("task_template"))
    raw_data = episode.get("episode_data")
    data = raw_data if isinstance(raw_data, dict) else {}
    length = int(episode.get("episode_length") or 0)
    for key in ("response", "action_dict", "runner_step_latency_sec"):
      if len(data.get(key) or []) != length:
        incomplete.append({
            "task": task, "field": key, "expected_steps": length,
            "observed_steps": len(data.get(key) or []),
        })
    values = {str(x) for x in data.get("policy_digest", []) if x}
    if not values:
      unversioned.append(task)
    elif len(values) != 1:
      errors.append({"task": task, "mixed_policy_digests": sorted(values)})
    else:
      digests.update(values)
    if episode.get("exception_info"):
      errors.append({"task": task, "exception": str(episode["exception_info"])})
    if not isinstance(raw_data, dict):
      errors.append({"task": task, "episode_data_type": type(raw_data).__name__})
  canonical_digest = digests.most_common(1)[0][0] if digests else None
  return {
      "arm": arm,
      "expected_tasks": len(expected),
      "checkpoint_tasks": len(observed),
      "missing_tasks": sorted(expected - observed),
      "unknown_tasks": sorted(observed - expected),
      "checkpoint_without_complete_status": sorted(observed - completed_status),
      "complete_status_without_checkpoint": sorted(completed_status - observed),
      "policy_digest_counts": dict(digests),
      "canonical_policy_digest": canonical_digest,
      "unversioned_tasks": sorted(unversioned),
      "noncanonical_policy_tasks": sorted(
          str(row.get("task_template")) for row in episodes
          if canonical_digest and isinstance(row.get("episode_data"), dict)
          and row["episode_data"].get("policy_digest")
          and canonical_digest not in row["episode_data"].get("policy_digest", [])
      ),
      "incomplete_telemetry": incomplete,
      "errors": errors,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--arm", required=True)
  args = parser.parse_args()
  print(json.dumps(audit(args.root, args.arm), indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
