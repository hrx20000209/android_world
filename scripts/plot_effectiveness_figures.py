#!/usr/bin/env python3
# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plot effectiveness-analysis figures from effectiveness_trace.json files.

Figure 1:
- X-axis: page complexity bucket (clickable element count).
- Left axis: rollback level1/level2 trigger (rate or count).
- Right axis: rollback success rate.

Figure 2:
- X-axis: page complexity bucket (clickable element count).
- Left axis: hint-hit rate (final action hits hint bbox).
- Right axis: task success rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _parse_bins(raw: str) -> list[int]:
    out = [_safe_int(x.strip()) for x in str(raw).split(",")]
    out = [x for x in out if x is not None]
    if len(out) < 2:
        raise ValueError("complexity bins must include at least two integers")
    uniq = sorted(set(out))
    if len(uniq) < 2:
        raise ValueError("complexity bins must include at least two distinct integers")
    return uniq


def _bucket_label(value: int | None, bins: list[int]) -> str:
    if value is None:
        return "unknown"
    if value < bins[0]:
        return f"<{bins[0]}"
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if i < len(bins) - 2:
            if lo <= value < hi:
                return f"[{lo},{hi})"
        else:
            if lo <= value <= hi:
                return f"[{lo},{hi}]"
    return f">{bins[-1]}"


def _bucket_order(bins: list[int]) -> list[str]:
    labels = [f"<{bins[0]}"]
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if i < len(bins) - 2:
            labels.append(f"[{lo},{hi})")
        else:
            labels.append(f"[{lo},{hi}]")
    labels.append(f">{bins[-1]}")
    labels.append("unknown")
    return labels


def _load_trace(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("steps"), list):
        return None
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate(trace_files: list[Path], bins: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    bucket_roll: dict[str, dict[str, int]] = {}
    for label in _bucket_order(bins):
        bucket_roll[label] = {
            "steps": 0,
            "rollback_triggered": 0,
            "rollback_success": 0,
            "rollback_level1": 0,
            "rollback_level2": 0,
            "hint_true": 0,
            "hint_false": 0,
            "hint_total": 0,
        }

    task_stats_by_bucket: dict[str, dict[str, int]] = {}
    for label in _bucket_order(bins):
        task_stats_by_bucket[label] = {"tasks": 0, "task_success": 0}

    task_count = 0
    step_count = 0

    for trace_path in trace_files:
        payload = _load_trace(trace_path)
        if payload is None:
            continue
        task_count += 1
        task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
        steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []

        clickables: list[int] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_count += 1
            clickable = _safe_int(step.get("page_clickable_element_count"))
            if clickable is not None:
                clickables.append(clickable)
            bucket = _bucket_label(clickable, bins)
            stats = bucket_roll.setdefault(
                bucket,
                {
                    "steps": 0,
                    "rollback_triggered": 0,
                    "rollback_success": 0,
                    "rollback_level1": 0,
                    "rollback_level2": 0,
                    "hint_true": 0,
                    "hint_false": 0,
                    "hint_total": 0,
                },
            )
            stats["steps"] += 1

            rollback_triggered = bool(step.get("rollback_triggered"))
            if rollback_triggered:
                stats["rollback_triggered"] += 1
                if bool(step.get("rollback_success")):
                    stats["rollback_success"] += 1
                level = str(step.get("rollback_level") or "none")
                if level == "level1":
                    stats["rollback_level1"] += 1
                elif level == "level2":
                    stats["rollback_level2"] += 1

            match = step.get("final_action_bbox_match_hint")
            if match is True:
                stats["hint_true"] += 1
                stats["hint_total"] += 1
            elif match is False:
                stats["hint_false"] += 1
                stats["hint_total"] += 1

        task_clickable_mean = int(round(float(sum(clickables)) / len(clickables))) if clickables else None
        task_bucket = _bucket_label(task_clickable_mean, bins)
        tstats = task_stats_by_bucket.setdefault(task_bucket, {"tasks": 0, "task_success": 0})
        tstats["tasks"] += 1
        if _safe_bool(task.get("episode_success")) is True:
            tstats["task_success"] += 1

    rollback_rows: list[dict[str, Any]] = []
    hint_task_rows: list[dict[str, Any]] = []
    for label in _bucket_order(bins):
        stats = bucket_roll.get(label, {})
        tstats = task_stats_by_bucket.get(label, {})

        steps = int(stats.get("steps", 0))
        rb_triggered = int(stats.get("rollback_triggered", 0))
        rb_success = int(stats.get("rollback_success", 0))
        rb_l1 = int(stats.get("rollback_level1", 0))
        rb_l2 = int(stats.get("rollback_level2", 0))
        hint_true = int(stats.get("hint_true", 0))
        hint_false = int(stats.get("hint_false", 0))
        hint_total = int(stats.get("hint_total", 0))
        task_n = int(tstats.get("tasks", 0))
        task_succ = int(tstats.get("task_success", 0))

        rollback_rows.append(
            {
                "bucket": label,
                "steps": steps,
                "rollback_triggered": rb_triggered,
                "rollback_success": rb_success,
                "rollback_level1": rb_l1,
                "rollback_level2": rb_l2,
                "rollback_trigger_rate": _ratio(rb_triggered, steps),
                "rollback_success_rate": _ratio(rb_success, rb_triggered),
                "rollback_level1_trigger_rate": _ratio(rb_l1, steps),
                "rollback_level2_trigger_rate": _ratio(rb_l2, steps),
            }
        )
        hint_task_rows.append(
            {
                "bucket": label,
                "steps": steps,
                "hint_hit_true": hint_true,
                "hint_hit_false": hint_false,
                "hint_hit_total": hint_total,
                "hint_hit_rate": _ratio(hint_true, hint_total),
                "tasks": task_n,
                "task_success_count": task_succ,
                "task_success_rate": _ratio(task_succ, task_n),
            }
        )

    meta = {"task_count": task_count, "step_count": step_count}
    return rollback_rows, hint_task_rows, meta


def _as_plot_vals(values: list[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def _plot_fig1(
    rollback_rows: list[dict[str, Any]],
    out_dir: Path,
    trigger_metric: str,
) -> None:
    labels = [row["bucket"] for row in rollback_rows]
    x = np.arange(len(labels))

    if trigger_metric == "count":
        l1 = np.array([float(row["rollback_level1"]) for row in rollback_rows], dtype=float)
        l2 = np.array([float(row["rollback_level2"]) for row in rollback_rows], dtype=float)
        left_ylabel = "Rollback Trigger Count"
    else:
        l1 = _as_plot_vals([row["rollback_level1_trigger_rate"] for row in rollback_rows])
        l2 = _as_plot_vals([row["rollback_level2_trigger_rate"] for row in rollback_rows])
        left_ylabel = "Rollback Trigger Rate"

    success = _as_plot_vals([row["rollback_success_rate"] for row in rollback_rows])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bar_w = 0.36
    bars1 = ax1.bar(x - bar_w / 2, l1, width=bar_w, color="#4C78A8", label="Level1 Trigger")
    bars2 = ax1.bar(x + bar_w / 2, l2, width=bar_w, color="#F58518", label="Level2 Trigger")
    ax1.set_xlabel("Page Complexity (Clickable Element Count)")
    ax1.set_ylabel(left_ylabel)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    line = ax2.plot(
        x,
        success,
        color="#2CA02C",
        marker="o",
        linewidth=2.2,
        label="Rollback Success Rate",
    )[0]
    ax2.set_ylabel("Rollback Success Rate")
    ax2.set_ylim(0.0, 1.05)

    h1, l1_labels = ax1.get_legend_handles_labels()
    h2, l2_labels = ax2.get_legend_handles_labels()
    ax1.legend(h1 + [line], l1_labels + l2_labels, loc="upper right")

    ax1.set_title("Rollback Trigger and Success vs Page Complexity")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig1_rollback_vs_complexity.png", dpi=200)
    fig.savefig(out_dir / "fig1_rollback_vs_complexity.pdf")
    plt.close(fig)


def _plot_fig2(hint_task_rows: list[dict[str, Any]], out_dir: Path) -> None:
    labels = [row["bucket"] for row in hint_task_rows]
    x = np.arange(len(labels))

    hint_hit = _as_plot_vals([row["hint_hit_rate"] for row in hint_task_rows])
    task_succ = _as_plot_vals([row["task_success_rate"] for row in hint_task_rows])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(x, hint_hit, width=0.55, color="#4C78A8", label="Hint Hit Rate")
    ax1.set_xlabel("Page Complexity (Clickable Element Count)")
    ax1.set_ylabel("Hint Hit Rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    line = ax2.plot(
        x,
        task_succ,
        color="#D62728",
        marker="s",
        linewidth=2.2,
        label="Task Success Rate",
    )[0]
    ax2.set_ylabel("Task Success Rate")
    ax2.set_ylim(0.0, 1.05)

    h1, l1_labels = ax1.get_legend_handles_labels()
    h2, l2_labels = ax2.get_legend_handles_labels()
    ax1.legend(h1 + [line], l1_labels + l2_labels, loc="upper right")

    ax1.set_title("Hint-Hit and Task Success vs Page Complexity")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig2_hint_tasksuccess_vs_complexity.png", dpi=200)
    fig.savefig(out_dir / "fig2_hint_tasksuccess_vs_complexity.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot effectiveness figures.")
    parser.add_argument(
        "--trace-root",
        type=str,
        default="./output/effectiveness_traces",
        help="Root directory containing effectiveness_trace.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./results/effectiveness_figures",
        help="Output directory for figures and intermediate CSV metrics.",
    )
    parser.add_argument(
        "--complexity-bins",
        type=str,
        default="0,8,16,24,128",
        help="Clickable-count bins, e.g., '0,8,16,24,128'.",
    )
    parser.add_argument(
        "--trigger-metric",
        type=str,
        default="rate",
        choices=["rate", "count"],
        help="Use rollback trigger rate or count in Figure 1.",
    )
    args = parser.parse_args()

    trace_root = Path(args.trace_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    bins = _parse_bins(args.complexity_bins)

    if not trace_root.exists():
        raise FileNotFoundError(f"trace root not found: {trace_root}")

    trace_files = sorted(trace_root.rglob("effectiveness_trace.json"))
    if not trace_files:
        raise FileNotFoundError(f"no effectiveness_trace.json found under: {trace_root}")

    rollback_rows, hint_task_rows, meta = _aggregate(trace_files, bins)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        out_dir / "fig1_bucket_rollback_metrics.csv",
        rollback_rows,
        [
            "bucket",
            "steps",
            "rollback_triggered",
            "rollback_success",
            "rollback_level1",
            "rollback_level2",
            "rollback_trigger_rate",
            "rollback_success_rate",
            "rollback_level1_trigger_rate",
            "rollback_level2_trigger_rate",
        ],
    )
    _write_csv(
        out_dir / "fig2_bucket_hint_task_metrics.csv",
        hint_task_rows,
        [
            "bucket",
            "steps",
            "hint_hit_true",
            "hint_hit_false",
            "hint_hit_total",
            "hint_hit_rate",
            "tasks",
            "task_success_count",
            "task_success_rate",
        ],
    )

    _plot_fig1(rollback_rows, out_dir, trigger_metric=args.trigger_metric)
    _plot_fig2(hint_task_rows, out_dir)

    summary = {
        "trace_root": str(trace_root),
        "trace_file_count": len(trace_files),
        "task_count": int(meta["task_count"]),
        "step_count": int(meta["step_count"]),
        "complexity_bins": bins,
        "trigger_metric": args.trigger_metric,
        "figure1_png": str(out_dir / "fig1_rollback_vs_complexity.png"),
        "figure2_png": str(out_dir / "fig2_hint_tasksuccess_vs_complexity.png"),
        "figure1_csv": str(out_dir / "fig1_bucket_rollback_metrics.csv"),
        "figure2_csv": str(out_dir / "fig2_bucket_hint_task_metrics.csv"),
    }
    (out_dir / "plot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded traces: {len(trace_files)} from {trace_root}")
    print(f"Figure 1: {out_dir / 'fig1_rollback_vs_complexity.png'}")
    print(f"Figure 2: {out_dir / 'fig2_hint_tasksuccess_vs_complexity.png'}")
    print(f"Metrics CSV: {out_dir / 'fig1_bucket_rollback_metrics.csv'}")
    print(f"Metrics CSV: {out_dir / 'fig2_bucket_hint_task_metrics.csv'}")


if __name__ == "__main__":
    main()
