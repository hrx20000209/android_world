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

"""Plot rollback-by-complexity figure using embedded data only.

This script does not read trace/CSV inputs. It uses the extracted bucket stats
directly so plotting can be reproduced with one command.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Extracted from:
# /Users/huangrunxi/Projects/android_world/results/effectiveness_figures_rollback_only_3bins/rollback_bucket_metrics.csv
EMBEDDED_BUCKET_DATA = [
    {
        "bucket": "[0,4)",
        "steps": 210,
        "rollback_total": 54,
        "rollback_level1": 38,
        "rollback_level2": 16,
        "rollback_success": 47,
    },
    {
        "bucket": "[4,43)",
        "steps": 473,
        "rollback_total": 105,
        "rollback_level1": 73,
        "rollback_level2": 32,
        "rollback_success": 100,
    },
    {
        "bucket": "[43,128]",
        "steps": 278,
        "rollback_total": 54,
        "rollback_level1": 33,
        "rollback_level2": 21,
        "rollback_success": 47,
    },
]


def _safe_ratio(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num) / float(den)


def _prepare_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for row in EMBEDDED_BUCKET_DATA:
        total = int(row["rollback_total"])
        l1 = int(row["rollback_level1"])
        l2 = int(row["rollback_level2"])
        succ = int(row["rollback_success"])
        rows.append(
            {
                "bucket": str(row["bucket"]),
                "steps": int(row["steps"]),
                "rollback_total": total,
                "rollback_level1": l1,
                "rollback_level2": l2,
                "rollback_success": succ,
                "rollback_level1_ratio": _safe_ratio(l1, total),
                "rollback_level2_ratio": _safe_ratio(l2, total),
                "rollback_success_rate": _safe_ratio(succ, total),
            }
        )
    return rows


def _annotate_segment(
    ax: plt.Axes,
    x: float,
    y_bottom: float,
    height: float,
    text: str,
    *,
    color: str = "white",
) -> None:
    if height <= 0:
        return
    y = y_bottom + height / 2.0
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color=color, fontweight="bold")


def _plot(rows: list[dict[str, float | int | str]], out_dir: Path) -> tuple[Path, Path]:
    labels = [str(r["bucket"]) for r in rows]
    totals = np.array([float(r["rollback_total"]) for r in rows], dtype=float)
    l1 = np.array([float(r["rollback_level1"]) for r in rows], dtype=float)
    l2 = np.array([float(r["rollback_level2"]) for r in rows], dtype=float)
    l1_ratio = np.array([float(r["rollback_level1_ratio"]) for r in rows], dtype=float)
    l2_ratio = np.array([float(r["rollback_level2_ratio"]) for r in rows], dtype=float)
    success = np.array([float(r["rollback_success_rate"]) for r in rows], dtype=float)

    x = np.arange(len(rows))
    width = 0.62

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )
    fig, ax1 = plt.subplots(figsize=(10.8, 6.4))
    ax2 = ax1.twinx()

    bar_l1 = ax1.bar(x, l1, width=width, color="#4C78A8", edgecolor="white", linewidth=1.0, label="Level1 Count")
    bar_l2 = ax1.bar(
        x, l2, width=width, bottom=l1, color="#F58518", edgecolor="white", linewidth=1.0, label="Level2 Count"
    )
    line = ax2.plot(
        x,
        success,
        color="#2CA02C",
        marker="o",
        markersize=7,
        linewidth=2.4,
        label="Rollback Success Rate",
        zorder=5,
    )[0]

    ax1.set_ylabel("Rollback Trigger Count")
    ax2.set_ylabel("Rollback Success Rate")
    ax1.set_xlabel("Page Complexity Bucket (Clickable Element Count)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    ax1.set_axisbelow(True)

    ax1.set_ylim(0, max(totals) * 1.20)
    ax2.set_ylim(0.0, 1.05)

    for i in range(len(rows)):
        total = totals[i]
        if total <= 0:
            continue
        ax1.text(i, total + max(totals) * 0.04, f"n={int(total)}", ha="center", va="bottom", fontsize=10)
        _annotate_segment(ax1, i, 0.0, l1[i], f"{l1_ratio[i] * 100:.1f}%")
        _annotate_segment(ax1, i, l1[i], l2[i], f"{l2_ratio[i] * 100:.1f}%")
        ax2.text(
            i,
            success[i] + 0.02,
            f"{success[i] * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#2CA02C",
        )

    handles = [bar_l1, bar_l2, line]
    labels_legend = [h.get_label() for h in handles]
    ax1.legend(handles, labels_legend, loc="upper left", frameon=True)

    fig.suptitle("Rollback Count, Composition, and Success by Page Complexity", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "rollback_vs_complexity_embedded.png"
    out_pdf = out_dir / "rollback_vs_complexity_embedded.pdf"
    fig.savefig(out_png, dpi=240)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


def _write_exports(rows: list[dict[str, float | int | str]], out_dir: Path, png: Path, pdf: Path) -> None:
    csv_path = out_dir / "rollback_bucket_metrics_embedded.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "bucket",
            "steps",
            "rollback_total",
            "rollback_level1",
            "rollback_level2",
            "rollback_success",
            "rollback_level1_ratio",
            "rollback_level2_ratio",
            "rollback_success_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "data_source": "embedded",
        "bucket_count": len(rows),
        "png": str(png),
        "pdf": str(pdf),
        "csv": str(csv_path),
    }
    (out_dir / "plot_summary_embedded.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rollback complexity figure using embedded data.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./results/effectiveness_figures_rollback_embedded",
        help="Output directory for figure and exported CSV/JSON.",
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
