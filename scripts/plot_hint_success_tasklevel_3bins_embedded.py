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

"""Plot 3-bin hint/success correlation with fully embedded data.

Both y-series are task-level:
1) Task-level hint-follow rate:
   For each task i, h_i = hint_true_i / (hint_true_i + hint_false_i), if denominator > 0.
   Then average h_i within each complexity bucket.
2) Task success rate:
   Fraction of successful tasks within each complexity bucket.

Complexity bucket assignment is task-level:
- Use mean clickable element count over steps in a task as its complexity score.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# Embedded statistics from current effectiveness traces (3-bin regrouping):
# buckets: [0,9), [9,21), [21,128]
# - task_success_*: task-level counts
# - hint_follow_*: mean over tasks with at least one evaluable hint action
EMBEDDED_DATA: list[dict[str, Any]] = [
    {
        "bucket": "[0,9)",
        "task_count": 27,
        "task_success_count": 14,
        "task_success_rate": 0.5185185185185185,
        "task_success_ci_low": 0.3398532458621043,
        "task_success_ci_high": 0.6925704935029092,
        "hint_task_count": 21,
        "hint_follow_rate_task_mean": 0.07142857142857142,
        "hint_follow_ci_low": 0.0,
        "hint_follow_ci_high": 0.19047619047619047,
        "hint_true_total": 3,
        "hint_false_total": 39,
    },
    {
        "bucket": "[9,21)",
        "task_count": 31,
        "task_success_count": 17,
        "task_success_rate": 0.5483870967741935,
        "task_success_ci_low": 0.37771879603669134,
        "task_success_ci_high": 0.7083851716341388,
        "hint_task_count": 20,
        "hint_follow_rate_task_mean": 0.075,
        "hint_follow_ci_low": 0.0,
        "hint_follow_ci_high": 0.2,
        "hint_true_total": 3,
        "hint_false_total": 39,
    },
    {
        "bucket": "[21,128]",
        "task_count": 46,
        "task_success_count": 31,
        "task_success_rate": 0.6739130434782609,
        "task_success_ci_low": 0.5296746274432707,
        "task_success_ci_high": 0.7913423543550664,
        "hint_task_count": 26,
        "hint_follow_rate_task_mean": 0.15384615384615385,
        "hint_follow_ci_low": 0.038461538461538464,
        "hint_follow_ci_high": 0.3076923076923077,
        "hint_true_total": 4,
        "hint_false_total": 37,
    },
]


def _to_np(vals: list[float]) -> np.ndarray:
    return np.array([float(v) for v in vals], dtype=float)


def _plot(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    labels = [str(r["bucket"]) for r in rows]
    x = np.arange(len(labels))

    hint = _to_np([r["hint_follow_rate_task_mean"] for r in rows])
    hint_lo = _to_np([r["hint_follow_ci_low"] for r in rows])
    hint_hi = _to_np([r["hint_follow_ci_high"] for r in rows])
    hint_err = np.vstack([np.maximum(0.0, hint - hint_lo), np.maximum(0.0, hint_hi - hint)])

    succ = _to_np([r["task_success_rate"] for r in rows])
    succ_lo = _to_np([r["task_success_ci_low"] for r in rows])
    succ_hi = _to_np([r["task_success_ci_high"] for r in rows])
    succ_err = np.vstack([np.maximum(0.0, succ - succ_lo), np.maximum(0.0, succ_hi - succ)])

    fig, ax1 = plt.subplots(figsize=(10.8, 6.0))
    bars = ax1.bar(
        x,
        hint,
        width=0.58,
        color="#4C78A8",
        yerr=hint_err,
        capsize=5,
        label="Task-Level Hint-Follow Rate",
    )
    ax1.set_ylabel("Task-Level Hint-Follow Rate")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Task Complexity Bucket (Mean Clickable Element Count)")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    line = ax2.errorbar(
        x,
        succ,
        yerr=succ_err,
        color="#D62728",
        marker="s",
        markersize=7,
        linewidth=2.2,
        capsize=4,
        label="Task Success Rate",
    )
    ax2.set_ylabel("Task Success Rate")
    ax2.set_ylim(0.0, 1.05)

    for i, bar in enumerate(bars):
        row = rows[i]
        h = float(bar.get_height())
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.03,
            f"hint-task n={int(row['hint_task_count'])}\ntask n={int(row['task_count'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = [line.lines[0]], ["Task Success Rate"]
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

    ax1.set_title("Hint-Follow and Task Success vs Complexity (3 Bins, Task-Level)")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "hint_success_tasklevel_3bins_embedded.png"
    out_pdf = out_dir / "hint_success_tasklevel_3bins_embedded.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


def _write_csv(rows: list[dict[str, Any]], out_dir: Path) -> Path:
    csv_path = out_dir / "hint_success_tasklevel_3bins_embedded.csv"
    fieldnames = [
        "bucket",
        "task_count",
        "task_success_count",
        "task_success_rate",
        "task_success_ci_low",
        "task_success_ci_high",
        "hint_task_count",
        "hint_follow_rate_task_mean",
        "hint_follow_ci_low",
        "hint_follow_ci_high",
        "hint_true_total",
        "hint_false_total",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 3-bin hint/success figure with embedded data.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./results/effectiveness_figures_hint_tasklevel_3bins_embedded",
        help="Output directory for figure and CSV/JSON.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    rows = EMBEDDED_DATA
    png, pdf = _plot(rows, out_dir)
    csv_path = _write_csv(rows, out_dir)

    summary = {
        "data_source": "embedded",
        "note": "Both series are task-level; see module docstring for definitions.",
        "png": str(png),
        "pdf": str(pdf),
        "csv": str(csv_path),
    }
    (out_dir / "plot_summary_embedded.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Figure PNG: {png}")
    print(f"Figure PDF: {pdf}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
