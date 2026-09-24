#!/usr/bin/env python3
"""Summarize MobileExplorer checkpoints without relying on run-directory names."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world import checkpointer  # pylint: disable=wrong-import-position


def _as_list(value: Any) -> list[Any]:
  return list(value) if isinstance(value, (list, tuple)) else []


def _method(episode: dict[str, Any], path: Path) -> str:
  modes = _as_list((episode.get("episode_data") or {}).get("memory_mode"))
  if modes and modes[0]:
    return str(modes[0])
  lowered = str(path).lower()
  if "offline" in lowered:
    return "offline_trajectory"
  if "mobile" in lowered:
    return "mobileexplorer"
  return "unknown"


def _answer(actions: list[Any], parsed_actions: list[Any]) -> str:
  for value in reversed(actions):
    if not isinstance(value, dict):
      continue
    action_type = str(value.get("action_type") or value.get("action") or "").lower()
    if action_type == "answer":
      return str(value.get("text") or value.get("value") or "")
  for value in reversed(parsed_actions):
    if not isinstance(value, dict):
      continue
    if str(value.get("action") or "").lower() in {
        "answer", "complete", "finished", "terminate"
    }:
      return str(
          value.get("return") or value.get("return_text")
          or value.get("value") or value.get("text") or ""
      )
  return ""


def summarize(root: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for checkpoint in sorted(root.glob("**/run_*")):
    for episode in checkpointer.IncrementalCheckpointer(str(checkpoint)).load():
      data = episode.get("episode_data") or {}
      responses = _as_list(data.get("response"))
      latencies = [float(x) for x in _as_list(data.get("runner_step_latency_sec"))]
      actions = _as_list(data.get("action_dict"))
      parsed_actions = _as_list(data.get("parsed_action"))
      reasoning = [str(x) for x in _as_list(data.get("reasoning_mode"))]
      evidence = _as_list(data.get("evidence_ids"))
      rows.append({
          "method": _method(episode, checkpoint),
          "task": str(episode.get("task_template", "")),
          "task_seed": int(episode.get("seed", -1)),
          "success": float(episode.get("is_successful") or 0),
          "actions": int(episode.get("episode_length") or len(actions)),
          "primary_vlm_calls": sum(bool(str(x).strip()) for x in responses),
          "reasoning_skips": reasoning.count("skip"),
          "evidence_consumptions": sum(bool(x) for x in evidence),
          "step_latency_total_s": round(sum(latencies), 4),
          "step_latency_mean_s": round(sum(latencies) / len(latencies), 4) if latencies else None,
          "runtime_s": round(float(episode.get("run_time") or 0), 4),
          "answer": _answer(actions, parsed_actions),
          "checkpoint": str(checkpoint),
      })
  return rows


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--output-csv", type=Path)
  args = parser.parse_args()
  rows = summarize(args.root)
  if args.output_csv:
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
      if rows:
        writer.writeheader()
        writer.writerows(rows)
  print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
