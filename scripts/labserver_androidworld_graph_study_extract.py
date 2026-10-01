#!/usr/bin/env python3
"""Emit privacy-minimized evaluator fields from one AndroidWorld checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world import checkpointer  # pylint: disable=wrong-import-position


def extract(checkpoint_dir: Path) -> dict[str, object] | None:
  if not checkpoint_dir.is_dir():
    return None
  try:
    episodes = checkpointer.IncrementalCheckpointer(str(checkpoint_dir)).load()
  except Exception:
    return None
  for episode in episodes:
    task = str(episode.get("task_template") or "").strip()
    if not task:
      task = str(episode.get("task_name") or "").strip()
    try:
      steps_value = float(episode.get("episode_length"))
      steps: float | int | None = int(steps_value) if steps_value.is_integer() else steps_value
      if not math.isfinite(steps_value):
        steps = None
    except (TypeError, ValueError):
      steps = None
    try:
      evaluator_score = float(episode.get("is_successful") or 0.0)
    except (TypeError, ValueError):
      evaluator_score = 0.0
    # Some composite AndroidWorld evaluators return a fractional progress
    # score. Count only a fully satisfied evaluator as successful.
    exception = bool(episode.get("exception_info"))
    success = evaluator_score >= 1.0 and not exception and steps is not None
    return {
        "task": task,
        "success": success,
        "evaluator_score": evaluator_score,
        "episode_steps": steps,
        "evaluator_complete": bool(not exception and steps is not None),
        "episode_exception": exception,
    }
  return None


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint-dir", type=Path, required=True)
  args = parser.parse_args()
  result = extract(args.checkpoint_dir)
  if result is None:
    return 2
  print(json.dumps(result, ensure_ascii=False, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
