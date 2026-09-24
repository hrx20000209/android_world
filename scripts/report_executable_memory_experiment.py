"""Summarize real executable-memory outputs without imputing task outcomes.

The script consumes the directories produced by ``run.py`` or the existing
``run_sensys30_online_task.py``.  It deliberately treats success as unknown
unless an evaluator JSON/JSONL record explicitly contains a success field.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, ValueError):
    return None
  return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  try:
    lines = path.read_text(encoding="utf-8").splitlines()
  except OSError:
    return rows
  for line in lines:
    try:
      row = json.loads(line)
    except ValueError:
      continue
    if isinstance(row, dict):
      rows.append(row)
  return rows


def _success_index(root: Path) -> dict[str, bool]:
  """Find explicit evaluator results, never infer them from action logs."""
  result: dict[str, bool] = {}
  for path in root.rglob("*.jsonl"):
    for row in _read_jsonl(path):
      for key in ("is_successful", "successful", "success"):
        if isinstance(row.get(key), bool):
          task = str(row.get("task") or row.get("task_template") or row.get("goal") or "")
          if task:
            result[task] = row[key]
  for path in root.rglob("*.json"):
    row = _read_json(path)
    if not row:
      continue
    for key in ("is_successful", "successful", "success"):
      if isinstance(row.get(key), bool):
        task = str(row.get("task") or row.get("task_template") or row.get("goal") or path.parent.name)
        result[task] = row[key]
  return result


def _metrics(path: Path) -> dict[str, Any]:
  payload = _read_json(path) or {}
  metrics = payload.get("metrics")
  return dict(metrics) if isinstance(metrics, dict) else {}


def build_report(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  success = _success_index(root)
  per_task: list[dict[str, Any]] = []
  by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
  memory_files = sorted(root.rglob("executable_memory.json"))
  for path in memory_files:
    arm = path.parent.name
    metrics = _metrics(path)
    action_files = sorted(path.parent.rglob("action.jsonl"))
    task_names = [action.parent.name for action in action_files] or [arm]
    for task in task_names:
      row = {
          "arm": arm, "task": task,
          "success": success.get(task),
          "metrics": metrics,
          "source": str(path),
      }
      per_task.append(row)
      by_arm[arm].append(row)
  summary: dict[str, Any] = {"schema_version": 1, "arms": {}}
  for arm, rows in sorted(by_arm.items()):
    values = [row["metrics"] for row in rows]
    known_success = [row["success"] for row in rows if isinstance(row["success"], bool)]
    def mean(name: str) -> float | None:
      numbers = [float(value[name]) for value in values if isinstance(value.get(name), (int, float))]
      return sum(numbers) / len(numbers) if numbers else None
    summary["arms"][arm] = {
        "tasks": len(rows),
        "success_rate": (sum(known_success) / len(known_success) if known_success else None),
        "success_observations": len(known_success),
        "five_probe_completion_rate": mean("five_probe_completion_rate"),
        "extra_time_s": mean("extra_time_s"),
        "prompt_adoptions": mean("prompt_adoptions"),
        "coordinate_corrections": mean("coordinate_corrections"),
        "graph_overrides": mean("graph_overrides"),
        "graph_rejections": mean("graph_rejections"),
        "graph_coverage": mean("graph_coverage"),
        "skip_hits": mean("skip_hits"),
        "jump_hit_rate": mean("jump_hit_rate"),
        "recovery_failures": mean("recovery_failures"),
        "recovery_failure_rate": mean("recovery_failure_rate"),
    }
  return per_task, summary


def _write_plot(summary: dict[str, Any], path: Path) -> None:
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    return
  arms = list(summary.get("arms", {}))
  if not arms:
    return
  labels = [arm.replace("_", "\n") for arm in arms]
  fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
  for axis, key, title in (
      (axes[0], "success_rate", "任务成功率（仅 evaluator 字段）"),
      (axes[1], "five_probe_completion_rate", "5-probe 完成率"),
      (axes[2], "extra_time_s", "额外耗时（秒）"),
  ):
    values = [summary["arms"][arm].get(key) for arm in arms]
    shown = [float(value) if value is not None else 0.0 for value in values]
    bars = axis.bar(range(len(arms)), shown)
    axis.set_title(title)
    axis.set_xticks(range(len(arms)), labels, rotation=30, ha="right")
    if key == "success_rate" and not any(value is not None for value in values):
      axis.text(.5, .5, "unavailable", ha="center", va="center", transform=axis.transAxes)
    for bar, value in zip(bars, values):
      if value is not None:
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                  ha="center", va="bottom", fontsize=8)
  fig.suptitle("Executable Exploration Memory 实验汇总（真实记录）")
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  per_task, summary = build_report(args.root)
  with (args.output / "per_task.jsonl").open("w", encoding="utf-8") as stream:
    for row in per_task:
      stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
  (args.output / "summary.json").write_text(
      json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
  _write_plot(summary, args.output / "executable_memory_summary_zh.png")
  print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
