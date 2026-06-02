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

"""Plot rollback-only effectiveness figure by page complexity buckets.

This script is step-level only:
- X-axis: page complexity bucket by clickable element count.
- Bar axis: rollback type composition (level1/level2 share, sums to 1 per bucket).
- Line axis: overall rollback success rate.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None
    return payload


def _aggregate(trace_files: list[Path], bins: list[int]) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, int]] = {}
    for label in _bucket_order(bins):
        stats[label] = {
            "steps": 0,
            "rollback_triggered": 0,
            "rollback_success": 0,
            "rollback_level1": 0,
            "rollback_level2": 0,
            "rollback_level1_success": 0,
            "rollback_level2_success": 0,
        }

    for trace_file in trace_files:
        payload = _load_trace(trace_file)
        if payload is None:
            continue
        for step in payload.get("steps", []):
            if not isinstance(step, dict):
                continue
            clickable = _safe_int(step.get("page_clickable_element_count"))
            bucket = _bucket_label(clickable, bins)
            bucket_stats = stats.setdefault(
                bucket,
                {
                    "steps": 0,
                    "rollback_triggered": 0,
                    "rollback_success": 0,
                    "rollback_level1": 0,
                    "rollback_level2": 0,
                    "rollback_level1_success": 0,
                    "rollback_level2_success": 0,
                },
            )
            bucket_stats["steps"] += 1

            if bool(step.get("rollback_triggered")):
                bucket_stats["rollback_triggered"] += 1
                rollback_success = bool(step.get("rollback_success"))
                if rollback_success:
                    bucket_stats["rollback_success"] += 1
                level = str(step.get("rollback_level") or "none")
                if level == "level1":
                    bucket_stats["rollback_level1"] += 1
                    if rollback_success:
                        bucket_stats["rollback_level1_success"] += 1
                elif level == "level2":
                    bucket_stats["rollback_level2"] += 1
                    if rollback_success:
                        bucket_stats["rollback_level2_success"] += 1

    rows: list[dict[str, Any]] = []
    for bucket in _bucket_order(bins):
        s = stats[bucket]
        rows.append(
            {
                "bucket": bucket,
                "steps": s["steps"],
                "rollback_triggered": s["rollback_triggered"],
                "rollback_success": s["rollback_success"],
                "rollback_level1": s["rollback_level1"],
                "rollback_level2": s["rollback_level2"],
                "rollback_level1_success": s["rollback_level1_success"],
                "rollback_level2_success": s["rollback_level2_success"],
                "rollback_trigger_rate": _ratio(s["rollback_triggered"], s["steps"]),
                "rollback_success_rate": _ratio(s["rollback_success"], s["rollback_triggered"]),
                "rollback_level1_trigger_rate": _ratio(s["rollback_level1"], s["steps"]),
                "rollback_level2_trigger_rate": _ratio(s["rollback_level2"], s["steps"]),
                "rollback_level1_success_rate": _ratio(s["rollback_level1_success"], s["rollback_level1"]),
                "rollback_level2_success_rate": _ratio(s["rollback_level2_success"], s["rollback_level2"]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "bucket",
        "steps",
        "rollback_triggered",
        "rollback_success",
        "rollback_level1",
        "rollback_level2",
        "rollback_level1_success",
        "rollback_level2_success",
        "rollback_trigger_rate",
        "rollback_success_rate",
        "rollback_level1_trigger_rate",
        "rollback_level2_trigger_rate",
        "rollback_level1_success_rate",
        "rollback_level2_success_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_plot(values: list[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def _plot(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    rows = [row for row in rows if int(row.get("steps") or 0) > 0]
    if not rows:
        raise ValueError("no non-empty buckets to plot")
    labels = [row["bucket"] for row in rows]
    x = np.arange(len(labels))
    rollback_total = np.array(
        [float((row.get("rollback_level1") or 0) + (row.get("rollback_level2") or 0)) for row in rows],
        dtype=float,
    )
    level1_raw = np.array([float(row.get("rollback_level1") or 0) for row in rows], dtype=float)
    level2_raw = np.array([float(row.get("rollback_level2") or 0) for row in rows], dtype=float)
    level1_ratio = np.divide(level1_raw, rollback_total, out=np.full_like(level1_raw, np.nan), where=rollback_total > 0)
    level2_ratio = np.divide(level2_raw, rollback_total, out=np.full_like(level2_raw, np.nan), where=rollback_total > 0)
    success_rate = _to_plot([row["rollback_success_rate"] for row in rows])

    fig, ax1 = plt.subplots(figsize=(12, 6.2))
    ax2 = ax1.twinx()
    width = 0.34
    ax1.bar(x - (width / 2), level1_ratio, width=width, color="#4C78A8", label="Level1 Ratio")
    ax1.bar(x + (width / 2), level2_ratio, width=width, color="#F58518", label="Level2 Ratio")
    ax2.plot(x, success_rate, marker="o", linewidth=2.2, color="#2CA02C", label="Rollback Success Rate")

    ax1.set_ylabel("Rollback Ratio")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(axis="y", alpha=0.25)

    # Annotate rollback sample count per bucket for statistical context.
    for i, total in enumerate(rollback_total):
        if total > 0:
            ax1.text(i, 1.02, f"n={int(total)}", ha="center", va="bottom", fontsize=9)

    ax2.set_ylabel("Rollback Success Rate")
    ax2.set_ylim(0.0, 1.05)
    ax1.set_xlabel("Page Complexity (Clickable Element Count)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", ncol=2, frameon=True)

    fig.suptitle("Rollback Composition and Success by Page Complexity", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "rollback_vs_complexity_only.png"
    out_pdf = out_dir / "rollback_vs_complexity_only.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png, out_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot rollback-only figure by complexity.")
    parser.add_argument(
        "--trace-root",
        type=str,
        default="./output/effectiveness_traces",
        help="Root directory containing effectiveness_trace.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./results/effectiveness_figures_rollback_only",
        help="Output directory for figure and CSV.",
    )
    parser.add_argument(
        "--complexity-bins",
        type=str,
        default="0,8,16,24,128",
        help="Clickable-count bins, e.g. '0,8,16,24,128'.",
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

    rows = _aggregate(trace_files, bins)
    csv_path = out_dir / "rollback_bucket_metrics.csv"
    _write_csv(csv_path, rows)
    png_path, pdf_path = _plot(rows, out_dir)

    summary = {
        "trace_root": str(trace_root),
        "trace_file_count": len(trace_files),
        "complexity_bins": bins,
        "csv": str(csv_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    (out_dir / "plot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded traces: {len(trace_files)} from {trace_root}")
    print(f"CSV: {csv_path}")
    print(f"Figure: {png_path}")


if __name__ == "__main__":
    main()
