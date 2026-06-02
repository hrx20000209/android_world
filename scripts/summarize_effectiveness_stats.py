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

"""Aggregate effectiveness traces into task/step tables and summary metrics.

Usage:
  python scripts/summarize_effectiveness_stats.py \
      --trace-root ./output/effectiveness_traces \
      --out-dir ./results/effectiveness_stats
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any


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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


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


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in columns})


def _load_trace(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("steps"), list):
        return None
    return payload


def _collect_rows(trace_files: list[Path], bins: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    for trace_path in trace_files:
        payload = _load_trace(trace_path)
        if payload is None:
            continue
        task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
        steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []

        task_id = str(task.get("task_id") or trace_path.parent.name)
        task_name = str(task.get("task_name") or trace_path.parent.name)

        task_explore_steps = 0
        task_rollback_steps = 0
        task_rollback_success_steps = 0
        task_hint_generated_steps = 0
        task_hint_injected_steps = 0
        task_no_effect_steps = 0
        task_valid_transition_steps = 0
        task_screen_diffs: list[float] = []

        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            screen_diff = _safe_float(step.get("screen_diff"))
            if screen_diff is not None:
                task_screen_diffs.append(screen_diff)

            explore_triggered = bool(step.get("exploration_triggered"))
            rollback_triggered = bool(step.get("rollback_triggered"))
            rollback_success = bool(step.get("rollback_success"))
            hint_generated = bool(step.get("hint_generated"))
            hint_injected = bool(step.get("hint_injected_into_prompt"))
            no_effect = bool(step.get("no_effect"))
            valid_transition = bool(step.get("valid_transition"))

            if explore_triggered:
                task_explore_steps += 1
            if rollback_triggered:
                task_rollback_steps += 1
                if rollback_success:
                    task_rollback_success_steps += 1
            if hint_generated:
                task_hint_generated_steps += 1
            if hint_injected:
                task_hint_injected_steps += 1
            if no_effect:
                task_no_effect_steps += 1
            if valid_transition:
                task_valid_transition_steps += 1

            clickable = _safe_int(step.get("page_clickable_element_count"))
            step_rows.append(
                {
                    "trace_path": str(trace_path),
                    "task_id": task_id,
                    "task_name": task_name,
                    "step_index": idx,
                    "page_total_element_count": _safe_int(step.get("page_total_element_count")),
                    "page_clickable_element_count": clickable,
                    "page_editable_element_count": _safe_int(step.get("page_editable_element_count")),
                    "page_complexity_bucket": _bucket_label(clickable, bins),
                    "exploration_triggered": explore_triggered,
                    "exploration_trigger_reason": step.get("exploration_trigger_reason"),
                    "exploration_budget_remaining": _safe_int(step.get("exploration_budget_remaining")),
                    "exploration_candidate_count": _safe_int(step.get("exploration_candidate_count")),
                    "exploration_selected_target_count": _safe_int(step.get("exploration_selected_target_count")),
                    "exploration_branch_count": _safe_int(step.get("exploration_branch_count")),
                    "exploration_best_branch_depth": _safe_int(step.get("exploration_best_branch_depth")),
                    "rollback_triggered": rollback_triggered,
                    "rollback_level": step.get("rollback_level"),
                    "rollback_success": rollback_success,
                    "rollback_latency_ms": _safe_float(step.get("rollback_latency_ms")),
                    "rollback_back_presses": _safe_int(step.get("rollback_back_presses")),
                    "rollback_replayed_actions": _safe_int(step.get("rollback_replayed_actions")),
                    "rollback_matched_by": step.get("rollback_matched_by"),
                    "hint_generated": hint_generated,
                    "hint_text": step.get("hint_text"),
                    "hint_source_depth": _safe_int(step.get("hint_source_depth")),
                    "hint_bbox": step.get("hint_bbox"),
                    "hint_target_element_text": step.get("hint_target_element_text"),
                    "hint_target_element_id": step.get("hint_target_element_id"),
                    "hint_injected_into_prompt": hint_injected,
                    "final_action_type": step.get("final_action_type"),
                    "final_action_coordinate": step.get("final_action_coordinate"),
                    "final_action_bbox_match_hint": step.get("final_action_bbox_match_hint"),
                    "screen_diff": screen_diff,
                    "no_effect": no_effect,
                    "repeated_no_effect_count": _safe_int(step.get("repeated_no_effect_count")),
                    "page_changed": bool(step.get("page_changed")),
                    "activity_changed": bool(step.get("activity_changed")),
                    "valid_transition": valid_transition,
                }
            )

        task_rows.append(
            {
                "trace_path": str(trace_path),
                "task_id": task_id,
                "task_name": task_name,
                "task_goal": task.get("task_goal"),
                "episode_success": _safe_bool(task.get("episode_success")),
                "episode_length": _safe_int(task.get("episode_length")) or len(steps),
                "total_rollbacks": _safe_int(task.get("total_rollbacks")),
                "total_level1_rollbacks": _safe_int(task.get("total_level1_rollbacks")),
                "total_level2_rollbacks": _safe_int(task.get("total_level2_rollbacks")),
                "hint_follow_rate": _safe_float(task.get("hint_follow_rate")),
                "hint_follow_match_count": _safe_int(task.get("hint_follow_match_count")),
                "hint_follow_total_count": _safe_int(task.get("hint_follow_total_count")),
                "episode_started_at": _safe_float(task.get("episode_started_at")),
                "episode_duration_sec": _safe_float(task.get("episode_duration_sec")),
                "exploration_triggered_steps": task_explore_steps,
                "rollback_triggered_steps": task_rollback_steps,
                "rollback_success_steps": task_rollback_success_steps,
                "hint_generated_steps": task_hint_generated_steps,
                "hint_injected_steps": task_hint_injected_steps,
                "no_effect_steps": task_no_effect_steps,
                "valid_transition_steps": task_valid_transition_steps,
                "screen_diff_mean": _mean(task_screen_diffs),
            }
        )

    return task_rows, step_rows


def _build_summary(
    task_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    bins: list[int],
    trace_root: str,
    trace_files: list[str],
) -> dict[str, Any]:
    known_success = [row for row in task_rows if row.get("episode_success") is not None]
    success_count = sum(1 for row in known_success if row.get("episode_success") is True)

    hint_follow_match_total = sum(int(row.get("hint_follow_match_count") or 0) for row in task_rows)
    hint_follow_total_total = sum(int(row.get("hint_follow_total_count") or 0) for row in task_rows)
    per_task_hint_follow = [float(row["hint_follow_rate"]) for row in task_rows if row.get("hint_follow_rate") is not None]

    episode_lengths = [int(row["episode_length"]) for row in task_rows if row.get("episode_length") is not None]
    episode_durations = [
        float(row["episode_duration_sec"]) for row in task_rows if row.get("episode_duration_sec") is not None
    ]

    total_steps = len(step_rows)
    exploration_steps = sum(1 for row in step_rows if row.get("exploration_triggered"))
    rollback_steps = [row for row in step_rows if row.get("rollback_triggered")]
    rollback_success_steps = sum(1 for row in rollback_steps if row.get("rollback_success"))
    rollback_level1 = sum(1 for row in rollback_steps if row.get("rollback_level") == "level1")
    rollback_level2 = sum(1 for row in rollback_steps if row.get("rollback_level") == "level2")
    hint_generated_steps = sum(1 for row in step_rows if row.get("hint_generated"))
    hint_injected_steps = sum(1 for row in step_rows if row.get("hint_injected_into_prompt"))

    bbox_true = sum(1 for row in step_rows if row.get("final_action_bbox_match_hint") is True)
    bbox_false = sum(1 for row in step_rows if row.get("final_action_bbox_match_hint") is False)

    hint_followed = [row for row in step_rows if row.get("final_action_bbox_match_hint") is True]
    hint_not_followed = [row for row in step_rows if row.get("final_action_bbox_match_hint") is False]

    def _rates(rows: list[dict[str, Any]]) -> dict[str, float | None]:
        if not rows:
            return {"valid_transition_rate": None, "no_effect_rate": None}
        valid = sum(1 for row in rows if row.get("valid_transition"))
        no_effect = sum(1 for row in rows if row.get("no_effect"))
        return {
            "valid_transition_rate": _ratio(valid, len(rows)),
            "no_effect_rate": _ratio(no_effect, len(rows)),
        }

    # Page-complexity buckets for rollback/exploration/hint analysis.
    bucket_keys = []
    for i in range(len(bins) - 1):
        lo = bins[i]
        hi = bins[i + 1]
        if i < len(bins) - 2:
            bucket_keys.append(f"[{lo},{hi})")
        else:
            bucket_keys.append(f"[{lo},{hi}]")
    bucket_keys = [f"<{bins[0]}"] + bucket_keys + [f">{bins[-1]}", "unknown"]

    complexity: dict[str, dict[str, Any]] = {}
    for key in bucket_keys:
        complexity[key] = {
            "steps": 0,
            "exploration_triggered_steps": 0,
            "rollback_triggered_steps": 0,
            "rollback_success_steps": 0,
            "hint_match_true_steps": 0,
            "hint_match_false_steps": 0,
            "valid_transition_steps": 0,
            "no_effect_steps": 0,
            "rollback_level1_steps": 0,
            "rollback_level2_steps": 0,
        }

    for row in step_rows:
        key = str(row.get("page_complexity_bucket") or "unknown")
        stats = complexity.setdefault(
            key,
            {
                "steps": 0,
                "exploration_triggered_steps": 0,
                "rollback_triggered_steps": 0,
                "rollback_success_steps": 0,
                "hint_match_true_steps": 0,
                "hint_match_false_steps": 0,
                "valid_transition_steps": 0,
                "no_effect_steps": 0,
                "rollback_level1_steps": 0,
                "rollback_level2_steps": 0,
            },
        )
        stats["steps"] += 1
        if row.get("exploration_triggered"):
            stats["exploration_triggered_steps"] += 1
        if row.get("rollback_triggered"):
            stats["rollback_triggered_steps"] += 1
            if row.get("rollback_success"):
                stats["rollback_success_steps"] += 1
            if row.get("rollback_level") == "level1":
                stats["rollback_level1_steps"] += 1
            elif row.get("rollback_level") == "level2":
                stats["rollback_level2_steps"] += 1
        if row.get("final_action_bbox_match_hint") is True:
            stats["hint_match_true_steps"] += 1
        elif row.get("final_action_bbox_match_hint") is False:
            stats["hint_match_false_steps"] += 1
        if row.get("valid_transition"):
            stats["valid_transition_steps"] += 1
        if row.get("no_effect"):
            stats["no_effect_steps"] += 1

    complexity_with_rates: dict[str, dict[str, Any]] = {}
    for key, stats in complexity.items():
        steps = int(stats["steps"])
        rollback_triggered_steps = int(stats["rollback_triggered_steps"])
        hint_match_total = int(stats["hint_match_true_steps"]) + int(stats["hint_match_false_steps"])
        complexity_with_rates[key] = {
            **stats,
            "exploration_trigger_rate": _ratio(int(stats["exploration_triggered_steps"]), steps),
            "rollback_trigger_rate": _ratio(rollback_triggered_steps, steps),
            "rollback_success_rate": _ratio(int(stats["rollback_success_steps"]), rollback_triggered_steps),
            "hint_follow_rate": _ratio(int(stats["hint_match_true_steps"]), hint_match_total),
            "valid_transition_rate": _ratio(int(stats["valid_transition_steps"]), steps),
            "no_effect_rate": _ratio(int(stats["no_effect_steps"]), steps),
        }

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "trace_root": trace_root,
        "trace_file_count": len(trace_files),
        "trace_files": trace_files,
        "task_level": {
            "task_count": len(task_rows),
            "known_success_task_count": len(known_success),
            "successful_task_count": success_count,
            "success_rate": _ratio(success_count, len(known_success)),
            "mean_episode_length": _mean([float(x) for x in episode_lengths]),
            "mean_episode_duration_sec": _mean(episode_durations),
            "weighted_hint_follow_rate": _ratio(hint_follow_match_total, hint_follow_total_total),
            "unweighted_hint_follow_rate": _mean(per_task_hint_follow),
        },
        "step_level": {
            "step_count": total_steps,
            "exploration_triggered_steps": exploration_steps,
            "exploration_trigger_rate": _ratio(exploration_steps, total_steps),
            "rollback_triggered_steps": len(rollback_steps),
            "rollback_success_steps": rollback_success_steps,
            "rollback_success_rate": _ratio(rollback_success_steps, len(rollback_steps)),
            "rollback_level1_steps": rollback_level1,
            "rollback_level2_steps": rollback_level2,
            "hint_generated_steps": hint_generated_steps,
            "hint_injected_steps": hint_injected_steps,
            "hint_bbox_match_true_steps": bbox_true,
            "hint_bbox_match_false_steps": bbox_false,
            "hint_follow_rate": _ratio(bbox_true, bbox_true + bbox_false),
            "hint_followed_effectiveness": _rates(hint_followed),
            "hint_not_followed_effectiveness": _rates(hint_not_followed),
        },
        "page_complexity_bins": bins,
        "page_complexity_metrics": complexity_with_rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize effectiveness_trace.json files.")
    parser.add_argument(
        "--trace-root",
        type=str,
        default="./output/effectiveness_traces",
        help="Root directory that contains effectiveness_trace.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./results/effectiveness_stats",
        help="Output directory for aggregated CSV/JSON stats.",
    )
    parser.add_argument(
        "--complexity-bins",
        type=str,
        default="0,8,16,24,128",
        help="Comma-separated clickable-count bins for page complexity analysis.",
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

    task_rows, step_rows = _collect_rows(trace_files, bins)
    summary = _build_summary(
        task_rows=task_rows,
        step_rows=step_rows,
        bins=bins,
        trace_root=str(trace_root),
        trace_files=[str(p) for p in trace_files],
    )

    task_columns = [
        "trace_path",
        "task_id",
        "task_name",
        "task_goal",
        "episode_success",
        "episode_length",
        "total_rollbacks",
        "total_level1_rollbacks",
        "total_level2_rollbacks",
        "hint_follow_rate",
        "hint_follow_match_count",
        "hint_follow_total_count",
        "episode_started_at",
        "episode_duration_sec",
        "exploration_triggered_steps",
        "rollback_triggered_steps",
        "rollback_success_steps",
        "hint_generated_steps",
        "hint_injected_steps",
        "no_effect_steps",
        "valid_transition_steps",
        "screen_diff_mean",
    ]
    step_columns = [
        "trace_path",
        "task_id",
        "task_name",
        "step_index",
        "page_total_element_count",
        "page_clickable_element_count",
        "page_editable_element_count",
        "page_complexity_bucket",
        "exploration_triggered",
        "exploration_trigger_reason",
        "exploration_budget_remaining",
        "exploration_candidate_count",
        "exploration_selected_target_count",
        "exploration_branch_count",
        "exploration_best_branch_depth",
        "rollback_triggered",
        "rollback_level",
        "rollback_success",
        "rollback_latency_ms",
        "rollback_back_presses",
        "rollback_replayed_actions",
        "rollback_matched_by",
        "hint_generated",
        "hint_text",
        "hint_source_depth",
        "hint_bbox",
        "hint_target_element_text",
        "hint_target_element_id",
        "hint_injected_into_prompt",
        "final_action_type",
        "final_action_coordinate",
        "final_action_bbox_match_hint",
        "screen_diff",
        "no_effect",
        "repeated_no_effect_count",
        "page_changed",
        "activity_changed",
        "valid_transition",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "task_metrics.csv", task_rows, task_columns)
    _write_csv(out_dir / "step_metrics.csv", step_rows, step_columns)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Loaded {len(trace_files)} trace files from: {trace_root}")
    print(f"Task metrics CSV: {out_dir / 'task_metrics.csv'}")
    print(f"Step metrics CSV: {out_dir / 'step_metrics.csv'}")
    print(f"Summary JSON: {out_dir / 'summary.json'}")
    print(
        "Summary:"
        f" tasks={summary['task_level']['task_count']},"
        f" success_rate={summary['task_level']['success_rate']},"
        f" exploration_trigger_rate={summary['step_level']['exploration_trigger_rate']},"
        f" rollback_success_rate={summary['step_level']['rollback_success_rate']}"
    )


if __name__ == "__main__":
    main()
