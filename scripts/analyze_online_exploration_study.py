#!/usr/bin/env python3
"""Validate, summarize, and plot the two online-exploration gate studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from android_world.parallel_exploration import oracle_validation
from android_world.parallel_exploration import shadow_isolation


def _load_probe_outcomes(path: Path) -> list[shadow_isolation.ProbeOutcome]:
  rows = []
  with path.open(encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, 1):
      if not line.strip():
        continue
      try:
        rows.append(shadow_isolation.ProbeOutcome.from_dict(json.loads(line)))
      except Exception as exc:
        raise ValueError(f"{path}:{line_number}: {exc}") from exc
  return rows


def _write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _plot_oracle(summary: dict[str, Any], output: Path) -> None:
  import matplotlib.pyplot as plt  # Imported only for the reporting command.

  arms = list(summary["arms"])
  labels = [name.replace("online_oracle_without_freshness", "online_stale") for name in arms]
  metrics = ("slot_accuracy", "remaining_reasoning_calls", "remaining_actions", "stale_error_rate")
  titles = ("Slot accuracy ↑", "Remaining calls ↓", "Remaining actions ↓", "Stale-error rate ↓")
  figure, axes = plt.subplots(1, 4, figsize=(14, 3.5))
  for axis, metric, title in zip(axes, metrics, titles):
    axis.bar(range(len(arms)), [summary["arms"][arm][metric] for arm in arms])
    axis.set_title(title)
    axis.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
    axis.grid(axis="y", alpha=0.2)
  figure.suptitle(f"Online-information oracle ({summary['paired_cases']} paired cases)")
  figure.tight_layout()
  figure.savefig(output, dpi=180, bbox_inches="tight")
  plt.close(figure)


def _plot_isolation(summary: dict[str, Any], output: Path) -> None:
  import matplotlib.pyplot as plt

  stages = ("probe", "shadow_resync", "evidence_extraction")
  p50 = [summary[name]["p50_s"] for name in stages]
  p95 = [summary[name]["p95_s"] for name in stages]
  figure, axes = plt.subplots(1, 2, figsize=(8, 3.5))
  x = range(len(stages))
  axes[0].bar([n - 0.18 for n in x], p50, width=0.36, label="P50")
  axes[0].bar([n + 0.18 for n in x], p95, width=0.36, label="P95")
  axes[0].set_xticks(list(x), [name.replace("_", "\n") for name in stages])
  axes[0].set_ylabel("seconds")
  axes[0].legend()
  rates = ("accepted_within_window_rate", "deadline_miss_rate", "state_match_rate")
  axes[1].bar(range(3), [summary[name] for name in rates])
  axes[1].set_xticks(range(3), ("within\nwindow", "deadline\nmiss", "state\nmatch"))
  axes[1].set_ylim(0, 1)
  axes[1].set_ylabel("fraction")
  figure.suptitle(f"Shadow-isolation feasibility (n={summary['n']})")
  figure.tight_layout()
  figure.savefig(output, dpi=180, bbox_inches="tight")
  plt.close(figure)


def _markdown(oracle: dict[str, Any], isolation: dict[str, Any] | None) -> str:
  decision = oracle["go_no_go"]
  lines = [
      "# Online Exploration Gate Report",
      "",
      f"**Decision: {decision['decision']}**",
      "",
      f"Paired cases: {oracle['paired_cases']}; passing app families: "
      f"{decision['passing_app_families']}.",
      "",
      "The decision uses paired task×seed comparisons. A positive online result must "
      "save at least one future model call or action versus both offline topology and "
      "deployment history in at least two app families, and fresh evidence must beat stale evidence.",
      "",
      "![Oracle result](oracle_result.png)",
  ]
  if isolation is not None:
    lines.extend([
        "",
        "## Isolation feasibility",
        "",
        f"Accepted within the inference window: {isolation['accepted_within_window_rate']:.1%}; "
        f"deadline misses: {isolation['deadline_miss_rate']:.1%}; "
        f"state matches: {isolation['state_match_rate']:.1%}.",
        "",
        "![Isolation feasibility](isolation_feasibility.png)",
        "",
        "These numbers describe a dual-emulator prototype only. They do not establish "
        "same-phone virtualization or resource neutrality.",
    ])
  return "\n".join(lines) + "\n"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--oracle-jsonl", type=Path, required=True)
  parser.add_argument("--isolation-jsonl", type=Path)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--bootstrap-iterations", type=int, default=4000)
  args = parser.parse_args()

  args.output_dir.mkdir(parents=True, exist_ok=True)
  oracle_rows = oracle_validation.load_results(args.oracle_jsonl)
  oracle = oracle_validation.summarize_results(
      oracle_rows, bootstrap_iterations=args.bootstrap_iterations
  )
  isolation = None
  if args.isolation_jsonl:
    isolation = shadow_isolation.summarize_probe_outcomes(
        _load_probe_outcomes(args.isolation_jsonl)
    )
  _write_json(args.output_dir / "oracle_summary.json", oracle)
  _plot_oracle(oracle, args.output_dir / "oracle_result.png")
  if isolation is not None:
    _write_json(args.output_dir / "isolation_summary.json", isolation)
    _plot_isolation(isolation, args.output_dir / "isolation_feasibility.png")
  (args.output_dir / "study_report.md").write_text(
      _markdown(oracle, isolation), encoding="utf-8"
  )


if __name__ == "__main__":
  main()
