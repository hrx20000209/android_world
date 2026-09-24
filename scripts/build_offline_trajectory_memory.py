#!/usr/bin/env python3
"""Compile evaluator-confirmed AndroidWorld checkpoints into trajectory memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world import checkpointer  # pylint: disable=wrong-import-position,import-error


def _value_at(value: Any, index: int) -> Any:
  if isinstance(value, (list, tuple)) and index < len(value):
    return value[index]
  return None


def _navigation_function(action: dict[str, Any]) -> str:
  action_type = str(action.get("action_type") or action.get("action") or "unknown")
  if action_type.lower() == "open_app":
    app = action.get("app_name") or action.get("text")
    return f"open_app({app})" if app else "open_app"
  if action_type.lower() in {"navigate_back", "navigate_home", "scroll"}:
    direction = action.get("direction")
    return f"{action_type}({direction})" if direction else action_type
  # Coordinates, entered values, answers, and task-specific labels are omitted.
  return action_type


def _episode_to_row(episode: dict[str, Any]) -> dict[str, Any]:
  data = episode.get("episode_data") or {}
  length = int(episode.get("episode_length") or 0)
  actions = data.get("action_dict") or data.get("action") or []
  summaries = data.get("summary") or data.get("response") or []
  state_ids = data.get("state_id") or []
  state_schema_ids = data.get("state_schema_id") or []
  source_activities = data.get("source_activity") or []
  source_landmarks = data.get("source_landmarks") or []
  target_descriptors = data.get("target_descriptor") or []
  policy_digests = sorted({str(x) for x in data.get("policy_digest", []) if x})
  steps = []
  navigation = []
  for index in range(length):
    raw_action = _value_at(actions, index)
    action = dict(raw_action) if isinstance(raw_action, dict) else {"action": str(raw_action or "")}
    summary = str(_value_at(summaries, index) or "")
    steps.append({
        "action": action,
        "summary": summary,
        "state_id": str(_value_at(state_ids, index) or ""),
        "state_schema_id": str(_value_at(state_schema_ids, index) or ""),
        "source_activity": str(_value_at(source_activities, index) or ""),
        "source_landmarks": list(_value_at(source_landmarks, index) or []),
        "target_descriptor": _value_at(target_descriptors, index),
    })
    navigation.append(_navigation_function(action))
  identity = (
      f"{episode.get('task_template')}|{episode.get('seed')}|"
      f"{episode.get('instance_id', 0)}"
  )
  return {
      "trajectory_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
      "task_template": str(episode.get("task_template", "")),
      "goal": str(episode.get("goal", "")),
      "seed": int(episode.get("seed", -1)),
      "successful": True,
      "steps": steps,
      "navigation_functions": list(dict.fromkeys(navigation)),
      "source_agent": str(episode.get("agent_name", "")),
      "source_policy_digest": policy_digests[0] if len(policy_digests) == 1 else None,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--checkpoint-dir", type=Path, required=True, action="append",
      help="Checkpoint directory; repeat to merge independent source seeds.",
  )
  parser.add_argument("--output-jsonl", type=Path, required=True)
  parser.add_argument(
      "--exclude-actual-seed", type=int, action="append", default=[],
      help="Reject episodes with these task-instance seeds (leakage guard).",
  )
  parser.add_argument(
      "--exclude-checkpoint-dir", type=Path, action="append", default=[],
      help=("Load every episode seed from this target checkpoint directory "
            "and reject matching source episodes; repeatable."),
  )
  parser.add_argument(
      "--require-replay-fields", action="store_true",
      help="Reject a source episode without homogeneous policy and UI binding telemetry.",
  )
  args = parser.parse_args()
  if args.output_jsonl.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output_jsonl}")
  episodes = []
  for directory in args.checkpoint_dir:
    episodes.extend(checkpointer.IncrementalCheckpointer(str(directory)).load())
  excluded = set(args.exclude_actual_seed)
  for directory in args.exclude_checkpoint_dir:
    excluded.update(
        int(episode.get("seed", -1))
        for episode in checkpointer.IncrementalCheckpointer(str(directory)).load()
        if int(episode.get("seed", -1)) >= 0
    )
  successful = [
      episode for episode in episodes
      if float(episode.get("is_successful") or 0) > 0.5
      and int(episode.get("seed", -1)) not in excluded
  ]
  if args.require_replay_fields:
    invalid = []
    for episode in successful:
      data = episode.get("episode_data") or {}
      actions = data.get("action_dict") or []
      step_count = len(actions)
      terminal_extra = int(
          step_count > 0 and isinstance(actions[-1], dict)
          and actions[-1].get("action_type") in {"status", "answer"}
      )
      digests = {str(x) for x in data.get("policy_digest", []) if x}
      if (len(digests) != 1 or not step_count
          or any(len(data.get(key) or []) not in {
              step_count, step_count - terminal_extra
          } for key in (
              "source_activity", "source_landmarks", "target_descriptor",
          ))):
        invalid.append(str(episode.get("task_template")))
    if invalid:
      raise ValueError(
          "Source episodes lack homogeneous replay-binding telemetry: "
          + ", ".join(sorted(invalid))
      )
  rows_by_id = {
      row["trajectory_id"]: row for row in map(_episode_to_row, successful)
  }
  rows = list(rows_by_id.values())
  args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
  with args.output_jsonl.open("w", encoding="utf-8") as stream:
    for row in rows:
      stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
  print(json.dumps({
      "episodes_seen": len(episodes),
      "successful_trajectories_written": len(rows),
      "task_templates_covered": len({row["task_template"] for row in rows}),
      "excluded_actual_seeds": sorted(excluded),
      "excluded_checkpoint_directories": [
          str(directory) for directory in args.exclude_checkpoint_dir
      ],
      "output": str(args.output_jsonl),
  }, indent=2))


if __name__ == "__main__":
  main()
