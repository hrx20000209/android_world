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

"""Generate candidate paper findings from effectiveness traces.

Focuses on analysis signals beyond the original hint-hit figure:
1) Rollback-assisted recovery after no-effect steps.
2) Prompt-level hint mode effect (all steps and higher-complexity pages).
3) Exploration trigger boundedness and reason distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
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


def _as_bool(value: Any) -> bool:
    return bool(value is True)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _wilson_interval(success: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = success / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return center - half, center + half


def _ztest_two_proportions(x1: int, n1: int, x0: int, n0: int) -> dict[str, float | None]:
    if min(n1, n0) <= 0:
        return {"p1": None, "p0": None, "delta": None, "z": None, "p_value": None}
    p1 = x1 / n1
    p0 = x0 / n0
    pooled = (x1 + x0) / (n1 + n0)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n0))
    if se <= 0:
        return {"p1": p1, "p0": p0, "delta": p1 - p0, "z": None, "p_value": None}
    z = (p1 - p0) / se
    # two-sided p-value from standard normal CDF.
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return {"p1": p1, "p0": p0, "delta": p1 - p0, "z": z, "p_value": p_value}


@dataclass
class StepRef:
    task_idx: int
    step_idx: int
    step: dict[str, Any]


def _load_traces(trace_root: Path) -> list[list[dict[str, Any]]]:
    trace_files = sorted(trace_root.rglob("effectiveness_trace.json"))
    if not trace_files:
        raise FileNotFoundError(f"no effectiveness_trace.json found under: {trace_root}")
    traces: list[list[dict[str, Any]]] = []
    for trace_file in trace_files:
        try:
            payload = json.loads(trace_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        steps = payload.get("steps")
        if isinstance(steps, list):
            traces.append(steps)
    return traces


def _has_valid_within(steps: list[dict[str, Any]], start_idx: int, horizon: int) -> bool:
    end = min(len(steps), start_idx + 1 + horizon)
    for idx in range(start_idx + 1, end):
        if _as_bool(steps[idx].get("valid_transition")):
            return True
    return False


def _aggregate(
    traces: list[list[dict[str, Any]]],
    *,
    high_complexity_threshold: int,
    recovery_horizon: int,
) -> dict[str, Any]:
    # 1) No-effect recovery with and without rollback.
    rec = {
        "with_rollback": {"success": 0, "total": 0},
        "without_rollback": {"success": 0, "total": 0},
    }
    rec_by_bucket: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {
            "with_rollback": {"success": 0, "total": 0},
            "without_rollback": {"success": 0, "total": 0},
        }
    )

    # 2) Prompt hint mode effect.
    hint_all = {
        "baseline": {"valid": 0, "no_effect": 0, "total": 0},
        "baseline_plus_hint": {"valid": 0, "no_effect": 0, "total": 0},
    }
    hint_high = {
        "baseline": {"valid": 0, "no_effect": 0, "total": 0},
        "baseline_plus_hint": {"valid": 0, "no_effect": 0, "total": 0},
    }

    # 3) Exploration boundedness.
    exploration_reason_counts: Counter[str] = Counter()
    exploration_budget_values: list[int] = []
    exploration_budget_negative_count = 0
    exploration_count_per_task: list[int] = []

    for task_idx, steps in enumerate(traces):
        del task_idx
        task_exploration_count = 0
        for i, step in enumerate(steps):
            clickable = _safe_int(step.get("page_clickable_element_count"))
            bucket = "high" if clickable is not None and clickable >= high_complexity_threshold else "low"

            valid = _as_bool(step.get("valid_transition"))
            no_effect = _as_bool(step.get("no_effect"))
            rollback_triggered = _as_bool(step.get("rollback_triggered"))
            prompt_mode = str(step.get("prompt_mode") or "baseline")
            if prompt_mode not in ("baseline", "baseline_plus_hint"):
                prompt_mode = "baseline"

            # Hint-mode step effectiveness.
            hint_all[prompt_mode]["total"] += 1
            if valid:
                hint_all[prompt_mode]["valid"] += 1
            if no_effect:
                hint_all[prompt_mode]["no_effect"] += 1

            if bucket == "high":
                hint_high[prompt_mode]["total"] += 1
                if valid:
                    hint_high[prompt_mode]["valid"] += 1
                if no_effect:
                    hint_high[prompt_mode]["no_effect"] += 1

            # Recovery after no-effect.
            if no_effect:
                outcome = _has_valid_within(steps, i, recovery_horizon)
                rec_key = "with_rollback" if rollback_triggered else "without_rollback"
                rec[rec_key]["total"] += 1
                rec_by_bucket[bucket][rec_key]["total"] += 1
                if outcome:
                    rec[rec_key]["success"] += 1
                    rec_by_bucket[bucket][rec_key]["success"] += 1

            # Exploration boundedness.
            if _as_bool(step.get("exploration_triggered")):
                task_exploration_count += 1
                exploration_reason_counts[str(step.get("exploration_trigger_reason") or "unknown")] += 1
                budget = _safe_int(step.get("exploration_budget_remaining"))
                if budget is not None:
                    exploration_budget_values.append(budget)
                    if budget < 0:
                        exploration_budget_negative_count += 1

        exploration_count_per_task.append(task_exploration_count)

    def _pack_group(metrics: dict[str, int]) -> dict[str, Any]:
        rate = _ratio(metrics["success"], metrics["total"])
        lo, hi = _wilson_interval(metrics["success"], metrics["total"])
        return {
            "success": metrics["success"],
            "total": metrics["total"],
            "rate": rate,
            "wilson_low": lo,
            "wilson_high": hi,
        }

    rec_summary = {
        "with_rollback": _pack_group(rec["with_rollback"]),
        "without_rollback": _pack_group(rec["without_rollback"]),
    }
    rec_summary["ztest"] = _ztest_two_proportions(
        rec["with_rollback"]["success"],
        rec["with_rollback"]["total"],
        rec["without_rollback"]["success"],
        rec["without_rollback"]["total"],
    )

    rec_bucket_summary: dict[str, Any] = {}
    for bucket, groups in rec_by_bucket.items():
        rec_bucket_summary[bucket] = {
            "with_rollback": _pack_group(groups["with_rollback"]),
            "without_rollback": _pack_group(groups["without_rollback"]),
            "ztest": _ztest_two_proportions(
                groups["with_rollback"]["success"],
                groups["with_rollback"]["total"],
                groups["without_rollback"]["success"],
                groups["without_rollback"]["total"],
            ),
        }

    def _pack_hint_group(group: dict[str, dict[str, int]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in group.items():
            vt = _ratio(v["valid"], v["total"])
            ne = _ratio(v["no_effect"], v["total"])
            vt_lo, vt_hi = _wilson_interval(v["valid"], v["total"])
            ne_lo, ne_hi = _wilson_interval(v["no_effect"], v["total"])
            out[k] = {
                "total": v["total"],
                "valid": v["valid"],
                "no_effect": v["no_effect"],
                "valid_transition_rate": vt,
                "valid_transition_wilson_low": vt_lo,
                "valid_transition_wilson_high": vt_hi,
                "no_effect_rate": ne,
                "no_effect_wilson_low": ne_lo,
                "no_effect_wilson_high": ne_hi,
            }
        out["valid_transition_ztest"] = _ztest_two_proportions(
            group["baseline_plus_hint"]["valid"],
            group["baseline_plus_hint"]["total"],
            group["baseline"]["valid"],
            group["baseline"]["total"],
        )
        out["no_effect_ztest"] = _ztest_two_proportions(
            group["baseline_plus_hint"]["no_effect"],
            group["baseline_plus_hint"]["total"],
            group["baseline"]["no_effect"],
            group["baseline"]["total"],
        )
        return out

    hint_summary = {
        "all_steps": _pack_hint_group(hint_all),
        "high_complexity_steps": _pack_hint_group(hint_high),
        "high_complexity_threshold": high_complexity_threshold,
    }

    exploration_summary = {
        "task_count": len(exploration_count_per_task),
        "mean_exploration_triggers_per_task": (
            float(np.mean(exploration_count_per_task)) if exploration_count_per_task else None
        ),
        "median_exploration_triggers_per_task": (
            float(np.median(exploration_count_per_task)) if exploration_count_per_task else None
        ),
        "max_exploration_triggers_per_task": (
            int(max(exploration_count_per_task)) if exploration_count_per_task else None
        ),
        "min_exploration_budget_remaining": (
            int(min(exploration_budget_values)) if exploration_budget_values else None
        ),
        "max_exploration_budget_remaining": (
            int(max(exploration_budget_values)) if exploration_budget_values else None
        ),
        "negative_budget_count": exploration_budget_negative_count,
        "reason_counts": dict(exploration_reason_counts.most_common()),
        "count_per_task": exploration_count_per_task,
    }

    return {
        "rollback_recovery": rec_summary,
        "rollback_recovery_by_complexity": rec_bucket_summary,
        "hint_mode_effect": hint_summary,
        "exploration_boundedness": exploration_summary,
    }


def _plot_rollback_recovery(summary: dict[str, Any], out_dir: Path) -> Path:
    groups = ["without_rollback", "with_rollback"]
    labels = ["No Rollback", "With Rollback"]
    rates = [summary[g]["rate"] for g in groups]
    lows = [summary[g]["wilson_low"] for g in groups]
    highs = [summary[g]["wilson_high"] for g in groups]
    totals = [summary[g]["total"] for g in groups]
    successes = [summary[g]["success"] for g in groups]

    x = np.arange(len(groups))
    y = np.array([np.nan if r is None else float(r) for r in rates], dtype=float)
    yerr_low = np.array([0.0 if l is None or r is None else max(0.0, float(r) - float(l)) for r, l in zip(rates, lows)])
    yerr_high = np.array(
        [0.0 if h is None or r is None else max(0.0, float(h) - float(r)) for r, h in zip(rates, highs)]
    )

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    bars = ax.bar(
        x,
        y,
        width=0.56,
        color=["#BAB0AC", "#4C78A8"],
        yerr=np.vstack([yerr_low, yerr_high]),
        capsize=5,
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(f"P(valid transition within next {2} steps)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Recovery After No-Effect Steps")

    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.03,
            f"{h*100:.1f}%\n({successes[i]}/{totals[i]})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    p_value = summary.get("ztest", {}).get("p_value")
    if p_value is not None:
        ax.text(0.02, 0.98, f"Two-proportion z-test p={p_value:.4f}", transform=ax.transAxes, va="top", fontsize=10)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "figA_noeffect_recovery_rollback.png"
    out_pdf = out_dir / "figA_noeffect_recovery_rollback.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png


def _plot_hint_mode_effect(summary: dict[str, Any], out_dir: Path) -> Path:
    contexts = ["all_steps", "high_complexity_steps"]
    context_labels = ["All Steps", "High Complexity"]

    valid_base = [summary[c]["baseline"]["valid_transition_rate"] for c in contexts]
    valid_hint = [summary[c]["baseline_plus_hint"]["valid_transition_rate"] for c in contexts]
    noeff_base = [summary[c]["baseline"]["no_effect_rate"] for c in contexts]
    noeff_hint = [summary[c]["baseline_plus_hint"]["no_effect_rate"] for c in contexts]
    n_base = [summary[c]["baseline"]["total"] for c in contexts]
    n_hint = [summary[c]["baseline_plus_hint"]["total"] for c in contexts]

    x = np.arange(len(contexts))
    w = 0.18

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 7.0), sharex=True, gridspec_kw={"height_ratios": [1, 1]})

    ax1.bar(x - w / 2, valid_base, width=w, color="#BAB0AC", label="Baseline")
    ax1.bar(x + w / 2, valid_hint, width=w, color="#4C78A8", label="Baseline + Hint")
    ax1.set_ylabel("Valid Transition Rate")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="lower left")

    ax2.bar(x - w / 2, noeff_base, width=w, color="#BAB0AC", label="Baseline")
    ax2.bar(x + w / 2, noeff_hint, width=w, color="#4C78A8", label="Baseline + Hint")
    ax2.set_ylabel("No-Effect Rate")
    ax2.set_ylim(0.0, 0.40)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [
            f"{context_labels[0]}\n(n_base={n_base[0]}, n_hint={n_hint[0]})",
            f"{context_labels[1]}\n(n_base={n_base[1]}, n_hint={n_hint[1]})",
        ]
    )

    p_all = summary["all_steps"]["valid_transition_ztest"]["p_value"]
    p_high = summary["high_complexity_steps"]["valid_transition_ztest"]["p_value"]
    ax1.text(
        0.02,
        0.98,
        f"Valid-transition p-values: all={p_all:.4f}, high={p_high:.4f}",
        transform=ax1.transAxes,
        va="top",
        fontsize=10,
    )

    fig.suptitle("Hint Prompt Mode vs Baseline", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "figB_hint_mode_effect.png"
    out_pdf = out_dir / "figB_hint_mode_effect.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png


def _plot_exploration_boundedness(summary: dict[str, Any], out_dir: Path) -> Path:
    counts = summary["count_per_task"]
    reason_counts = summary["reason_counts"]
    bins = ["0", "1-2", "3-4", "5+"]
    binned = Counter()
    for c in counts:
        if c == 0:
            binned["0"] += 1
        elif c <= 2:
            binned["1-2"] += 1
        elif c <= 4:
            binned["3-4"] += 1
        else:
            binned["5+"] += 1

    top_reason_items = list(reason_counts.items())[:6]
    reason_labels = [k for k, _ in top_reason_items]
    reason_vals = [v for _, v in top_reason_items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8), gridspec_kw={"width_ratios": [1.0, 1.4]})

    x1 = np.arange(len(bins))
    y1 = [binned[b] for b in bins]
    ax1.bar(x1, y1, color="#4C78A8", width=0.62)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(bins)
    ax1.set_ylabel("Task Count")
    ax1.set_xlabel("Exploration Triggers per Task")
    ax1.set_title("Bounded Trigger Frequency")
    ax1.grid(axis="y", alpha=0.25)

    x2 = np.arange(len(reason_labels))
    ax2.barh(x2, reason_vals, color="#72B7B2")
    ax2.set_yticks(x2)
    ax2.set_yticklabels(reason_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("Triggered Steps")
    ax2.set_title("Top Trigger Reasons")
    ax2.grid(axis="x", alpha=0.25)

    ax1.text(
        0.02,
        0.98,
        (
            f"mean={summary['mean_exploration_triggers_per_task']:.2f}, "
            f"median={summary['median_exploration_triggers_per_task']:.1f}, "
            f"max={summary['max_exploration_triggers_per_task']}\n"
            f"budget min={summary['min_exploration_budget_remaining']}, "
            f"negative_budget={summary['negative_budget_count']}"
        ),
        transform=ax1.transAxes,
        va="top",
        fontsize=9,
    )

    fig.suptitle("Exploration Activation is Frequent but Budget-Bounded", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "figC_exploration_boundedness.png"
    out_pdf = out_dir / "figC_exploration_boundedness.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png


def _write_csvs(out_dir: Path, summary: dict[str, Any]) -> None:
    # Figure A table
    figa_csv = out_dir / "figA_noeffect_recovery_table.csv"
    with figa_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "success",
                "total",
                "rate",
                "wilson_low",
                "wilson_high",
            ],
        )
        writer.writeheader()
        for group in ("without_rollback", "with_rollback"):
            row = {"group": group, **summary["rollback_recovery"][group]}
            writer.writerow(row)

    # Figure B table
    figb_csv = out_dir / "figB_hint_mode_effect_table.csv"
    with figb_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "context",
                "mode",
                "total",
                "valid",
                "no_effect",
                "valid_transition_rate",
                "no_effect_rate",
            ],
        )
        writer.writeheader()
        for context in ("all_steps", "high_complexity_steps"):
            for mode in ("baseline", "baseline_plus_hint"):
                source = summary["hint_mode_effect"][context][mode]
                row = {
                    "context": context,
                    "mode": mode,
                    "total": source.get("total"),
                    "valid": source.get("valid"),
                    "no_effect": source.get("no_effect"),
                    "valid_transition_rate": source.get("valid_transition_rate"),
                    "no_effect_rate": source.get("no_effect_rate"),
                }
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate effectiveness findings for paper plots.")
    parser.add_argument(
        "--trace-root",
        type=str,
        default="./output/effectiveness_traces",
        help="Root directory containing effectiveness_trace.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./results/effectiveness_candidate_findings",
        help="Output directory for figures and summary files.",
    )
    parser.add_argument(
        "--high-complexity-threshold",
        type=int,
        default=16,
        help="Clickable-element threshold for high complexity split.",
    )
    parser.add_argument(
        "--recovery-horizon",
        type=int,
        default=2,
        help="Number of subsequent steps for recovery metric.",
    )
    args = parser.parse_args()

    trace_root = Path(args.trace_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not trace_root.exists():
        raise FileNotFoundError(f"trace root not found: {trace_root}")

    traces = _load_traces(trace_root)
    summary = _aggregate(
        traces,
        high_complexity_threshold=int(args.high_complexity_threshold),
        recovery_horizon=int(args.recovery_horizon),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_a = _plot_rollback_recovery(summary["rollback_recovery"], out_dir)
    fig_b = _plot_hint_mode_effect(summary["hint_mode_effect"], out_dir)
    fig_c = _plot_exploration_boundedness(summary["exploration_boundedness"], out_dir)
    _write_csvs(out_dir, summary)

    export = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trace_root": str(trace_root),
        "trace_count": len(traces),
        "config": {
            "high_complexity_threshold": int(args.high_complexity_threshold),
            "recovery_horizon": int(args.recovery_horizon),
        },
        "summary": summary,
        "figures": {
            "figA_noeffect_recovery_rollback": str(fig_a),
            "figB_hint_mode_effect": str(fig_b),
            "figC_exploration_boundedness": str(fig_c),
        },
        "tables": {
            "figA_noeffect_recovery_table": str(out_dir / "figA_noeffect_recovery_table.csv"),
            "figB_hint_mode_effect_table": str(out_dir / "figB_hint_mode_effect_table.csv"),
        },
    }
    summary_path = out_dir / "candidate_findings_summary.json"
    summary_path.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded traces: {len(traces)} from {trace_root}")
    print(f"Summary: {summary_path}")
    print(f"Figure A: {fig_a}")
    print(f"Figure B: {fig_b}")
    print(f"Figure C: {fig_c}")


if __name__ == "__main__":
    main()
