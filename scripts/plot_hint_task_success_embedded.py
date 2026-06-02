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

"""Plot hint-hit and task-success vs complexity using embedded data only.

Rationale for paper reading:
- Hint-hit is step-level and only defined when hint bbox/action alignment is available.
- Task success is task-level and reflects all recovery mechanisms (rollback/exploration),
  not only direct hint following.
The chart therefore visualizes correlation, not a one-to-one causal relation.
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


# Embedded from:
# /Users/huangrunxi/Projects/android_world/results/effectiveness_figures_latest/fig2_bucket_hint_task_metrics.csv
EMBEDDED_BUCKETS: list[dict[str, Any]] = [
    {
        "bucket": "[0,8)",
        "steps": 324,
        "hint_hit_true": 4,
        "hint_hit_false": 59,
        "hint_hit_total": 63,
        "tasks": 19,
        "task_success_count": 10,
    },
    {
        "bucket": "[8,16)",
        "steps": 236,
        "hint_hit_true": 5,
        "hint_hit_false": 19,
        "hint_hit_total": 24,
        "tasks": 33,
        "task_success_count": 18,
    },
    {
        "bucket": "[16,24)",
        "steps": 84,
        "hint_hit_true": 0,
        "hint_hit_false": 10,
        "hint_hit_total": 10,
        "tasks": 9,
        "task_success_count": 4,
    },
    {
        "bucket": "[24,128]",
        "steps": 317,
        "hint_hit_true": 1,
        "hint_hit_false": 27,
        "hint_hit_total": 28,
        "tasks": 43,
        "task_success_count": 30,
    },
]


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = float(k) / float(n)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return center - half, center + half


def _to_numpy(values: list[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def _prepare_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in EMBEDDED_BUCKETS:
        hint_true = int(row["hint_hit_true"])
        hint_total = int(row["hint_hit_total"])
        task_succ = int(row["task_success_count"])
        task_total = int(row["tasks"])

        hint_rate = _ratio(hint_true, hint_total)
        task_rate = _ratio(task_succ, task_total)
        hint_lo, hint_hi = _wilson(hint_true, hint_total)
        task_lo, task_hi = _wilson(task_succ, task_total)

        rows.append(
            {
                **row,
                "hint_hit_rate": hint_rate,
                "task_success_rate": task_rate,
                "hint_ci_low": hint_lo,
                "hint_ci_high": hint_hi,
                "task_ci_low": task_lo,
                "task_ci_high": task_hi,
            }
        )
    return rows


def _plot(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    labels = [str(r["bucket"]) for r in rows]
    x = np.arange(len(rows))

    hint = _to_numpy([r["hint_hit_rate"] for r in rows])
    task = _to_numpy([r["task_success_rate"] for r in rows])

    hint_lo = _to_numpy([r["hint_ci_low"] for r in rows])
    hint_hi = _to_numpy([r["hint_ci_high"] for r in rows])
    task_lo = _to_numpy([r["task_ci_low"] for r in rows])
    task_hi = _to_numpy([r["task_ci_high"] for r in rows])

    hint_err = np.vstack(
        [
            np.maximum(0.0, hint - hint_lo),
            np.maximum(0.0, hint_hi - hint),
        ]
    )
    task_err = np.vstack(
        [
            np.maximum(0.0, task - task_lo),
            np.maximum(0.0, task_hi - task),
        ]
    )

    fig, ax1 = plt.subplots(figsize=(11.4, 6.2))
    bar_w = 0.56
    bars = ax1.bar(
        x,
        hint,
        width=bar_w,
        color="#4C78A8",
        yerr=hint_err,
        capsize=5,
        label="Hint Hit Rate (step-level)",
    )
    ax1.set_ylabel("Hint Hit Rate")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_xlabel("Page Complexity (Clickable Element Count)")

    ax2 = ax1.twinx()
    line = ax2.errorbar(
        x,
        task,
        yerr=task_err,
        color="#D62728",
        marker="s",
        markersize=7,
        linewidth=2.2,
        capsize=4,
        label="Task Success Rate (task-level)",
    )
    ax2.set_ylabel("Task Success Rate")
    ax2.set_ylim(0.0, 1.05)

    for idx, bar in enumerate(bars):
        row = rows[idx]
        hint_total = int(row["hint_hit_total"])
        tasks = int(row["tasks"])
        h = float(bar.get_height())
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.03,
            f"hint n={hint_total}\ntask n={tasks}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = [line.lines[0]], ["Task Success Rate (task-level)"]
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    ax1.set_title("Hint-Hit and Task Success vs Page Complexity (Embedded Data)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "fig2_hint_tasksuccess_embedded.png"
    out_pdf = out_dir / "fig2_hint_tasksuccess_embedded.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


def _write_exports(rows: list[dict[str, Any]], out_dir: Path, png: Path, pdf: Path) -> None:
    csv_path = out_dir / "fig2_hint_task_embedded_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "bucket",
            "steps",
            "hint_hit_true",
            "hint_hit_false",
            "hint_hit_total",
            "hint_hit_rate",
            "hint_ci_low",
            "hint_ci_high",
            "tasks",
            "task_success_count",
            "task_success_rate",
            "task_ci_low",
            "task_ci_high",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "data_source": "embedded",
        "png": str(png),
        "pdf": str(pdf),
        "csv": str(csv_path),
        "note": (
            "Hint-hit is computed on hint-evaluable click steps only; "
            "task success is computed at task level."
        ),
    }
    (out_dir / "plot_summary_embedded.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot hint-hit/task-success with embedded data.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./results/effectiveness_figures_hint_embedded",
        help="Output directory for figure and exported metrics.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    rows = _prepare_rows()
    png, pdf = _plot(rows, out_dir)
    _write_exports(rows, out_dir, png, pdf)

    print(f"Figure PNG: {png}")
    print(f"Figure PDF: {pdf}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
