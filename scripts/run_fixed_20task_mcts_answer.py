#!/usr/bin/env python3
"""Run the fixed 20-task MCTS/answer-extractor scale-up experiment."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results" / "fixed_20task_mcts_answer"
PREVIOUS_BROAD_RUN = REPO_ROOT / "results" / "tpes_androidworld_broad" / "run_20260529T151709"

_strategy_path = REPO_ROOT / "scripts" / "run_search_strategy_comparison.py"
_spec = importlib.util.spec_from_file_location("strategy_runner", _strategy_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not import {_strategy_path}")
strategy_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strategy_runner)

TASKS_20 = list(strategy_runner.TASKS_20)

COMMON_BASE = [
    "--suite_family=android_world",
    "--n_task_combinations=1",
    "--fixed_task_seed",
    "--image_downsample_scale=1.0",
    "--a11y_method=uiautomator",
    "--baseline_table=results/4B_2.txt",
]

FIXED_SHARED_ARGS = [
    "--agent_name=explore_agent_gelab",
    *COMMON_BASE,
    "--no-explore_diagnostic_full",
    "--no-explore_force_every_step",
    "--explore_fast_mode",
    "--explore_transaction_safe",
    "--explore_max_runs=10000",
    "--explore_max_step=10000",
    "--explore_branch_budget=3",
    "--explore_branch_depth=3",
    "--explore_back_limit=4",
    "--no-explore_planned_only",
    "--explore_fallback_safe_candidates",
    "--explore_safe_click_only",
    "--explore_skip_launcher",
    "--explore_filter_launcher_relevance",
    "--explore_skip_destructive_goals",
    "--explore_strategy=dfs",
    "--explore_hint_policy=strict",
    "--explore_search_policy=task_gate",
    "--explore_rollback_policy=improved",
    "--explore_fixed_framework",
]

VARIANTS: dict[str, dict[str, Any]] = {
    "V0_BASELINE": {
        "description": "No exploration baseline.",
        "args": ["--agent_name=gelab_agent", *COMMON_BASE],
    },
    "V2_SAFE_GREEDY": {
        "description": "Fixed shared framework, greedy search.",
        "args": [*FIXED_SHARED_ARGS, "--explore_search_strategy=greedy", "--explore_variant=V2_SAFE_GREEDY"],
    },
    "V7_RAW_MCTS": {
        "description": "Fixed shared framework, raw MCTS from the 5-task run.",
        "args": [*FIXED_SHARED_ARGS, "--explore_search_strategy=mcts", "--explore_variant=V7_RAW_MCTS"],
    },
    "V8_SAFE_MCTS": {
        "description": "MCTS with stricter risk/depth controls.",
        "args": [
            *FIXED_SHARED_ARGS,
            "--explore_search_strategy=mcts",
            "--explore_safe_mcts",
            "--explore_variant=V8_SAFE_MCTS",
        ],
    },
    "V9_SAFE_GREEDY_PLUS_ANSWER_EXTRACTORS": {
        "description": "Safe greedy plus lightweight answer extractors.",
        "args": [
            *FIXED_SHARED_ARGS,
            "--explore_search_strategy=greedy",
            "--explore_answer_extractors",
            "--explore_variant=V9_SAFE_GREEDY_PLUS_ANSWER_EXTRACTORS",
        ],
    },
    "V10_SAFE_MCTS_PLUS_ANSWER_EXTRACTORS": {
        "description": "Safe MCTS plus lightweight answer extractors.",
        "args": [
            *FIXED_SHARED_ARGS,
            "--explore_search_strategy=mcts",
            "--explore_safe_mcts",
            "--explore_answer_extractors",
            "--explore_variant=V10_SAFE_MCTS_PLUS_ANSWER_EXTRACTORS",
        ],
    },
}

DEFAULT_CORE_VARIANTS = [
    "V0_BASELINE",
    "V2_SAFE_GREEDY",
    "V7_RAW_MCTS",
    "V8_SAFE_MCTS",
    "V10_SAFE_MCTS_PLUS_ANSWER_EXTRACTORS",
]


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _run_variant(label: str, root: Path, tasks: list[str], extra_args: list[str]) -> Path:
    variant_root = root / label
    variant_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_exploration_experiment_report.py"),
        "--run",
        f"--tasks={','.join(tasks)}",
        f"--experiment_root={variant_root}",
        "--max_cases=20",
        *VARIANTS[label]["args"],
        *extra_args,
    ]
    log_path = variant_root / "driver.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{label}] {line}", end="")
            log.write(line)
        rc = proc.wait()
        if rc != 0:
            print(f"[{label}] exited with {rc}; partial traces will be analyzed if available")
    runs = sorted(variant_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise RuntimeError(f"No run directory produced for {label}")
    return runs[0]


def _latest_run_dir(root: Path, label: str) -> Path | None:
    runs = sorted((root / label).glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return strategy_runner._read_jsonl(path)  # pylint: disable=protected-access


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_required(goal: str) -> str:
    low = _clean(goal).lower()
    if re.search(r"\b(quantity|amount|unit|ingredient)\b", low):
        return "amount_unit"
    if re.search(r"\b(duration|how long|minutes?)\b", low):
        return "duration"
    if "distance" in low:
        return "distance"
    if re.search(r"\bhow many|count\b", low):
        return "count"
    if re.search(r"\bwhen|time|between|next meeting|events?\b", low):
        return "time_or_event"
    return "text_fact"


def _contains_required_metric(text: str, metric: str) -> bool:
    low = _clean(text).lower()
    if metric == "amount_unit":
        return bool(re.search(r"\b(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?)\s*(?:tsp|tbsp|teaspoons?|tablespoons?|cups?|oz|g|grams?|ml|l|pinch|cloves?)\b", low))
    if metric == "duration":
        return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:min|mins|minute|minutes|h|hr|hrs|hours?)\b|\b\d{1,2}:\d{2}(?::\d{2})?\b", low))
    if metric == "distance":
        return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:km|mi|mile|miles|m|meters?)\b", low))
    if metric == "count":
        return bool(re.search(r"\bcount\s*:\s*\d+\b|\b\d+\b", low))
    if metric == "time_or_event":
        return bool(re.search(r"\b\d{1,2}[:.]\d{2}\b|\bmeeting|event|workshop|session\b", low))
    return bool(low)


def _target_tokens(goal: str) -> set[str]:
    stop = {
        "what", "which", "when", "where", "answer", "with", "format", "express",
        "android", "world", "app", "have", "many", "count", "duration", "distance",
        "total", "events", "tasks", "activity", "activities", "recipe", "quantity",
        "amount", "need", "simple", "calendar", "joplin", "opentracks", "open",
    }
    return {t for t in re.findall(r"[a-z0-9]{3,}", _clean(goal).lower()) if t not in stop}


def _answer_hint_usage(run_dirs: dict[str, Path], per_task: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_status: dict[tuple[str, str], dict[str, Any]] = {
        (str(r.get("strategy")), str(r.get("task_instruction"))): r for r in per_task
    }
    for label, run_dir in run_dirs.items():
        for action_file in sorted((run_dir / "traces").rglob("action.jsonl")):
            for action_row in _read_jsonl(action_file):
                goal = _clean(action_row.get("goal"))
                selected = [
                    item for item in (action_row.get("matched_exploration_results") or [])
                    if isinstance(item, dict) and _clean(item.get("hint_type") or item.get("hint_kind")) == "ANSWER_HINT"
                ]
                if not selected:
                    continue
                status = task_status.get((label, goal), {})
                final_text = " ".join(
                    _clean(x)
                    for x in [
                        action_row.get("response"),
                        (action_row.get("parsed_action") or {}).get("return") if isinstance(action_row.get("parsed_action"), dict) else "",
                        (action_row.get("parsed_action") or {}).get("value") if isinstance(action_row.get("parsed_action"), dict) else "",
                        (action_row.get("parsed_action") or {}).get("summary") if isinstance(action_row.get("parsed_action"), dict) else "",
                    ]
                )
                for item in selected:
                    prompt_line = _clean(item.get("prompt_line"))
                    metric = _metric_required(goal)
                    tokens = _target_tokens(goal)
                    contains_entity = bool(tokens and any(token in prompt_line.lower() for token in tokens))
                    contains_metric = _contains_required_metric(prompt_line, metric)
                    hint_nums = set(re.findall(r"\d+(?:\.\d+)?", prompt_line))
                    final_nums = set(re.findall(r"\d+(?:\.\d+)?", final_text))
                    answer_used = bool(hint_nums and final_nums and hint_nums.intersection(final_nums))
                    success = float(status.get("strategy_success") or 0.0) >= 1.0
                    if answer_used:
                        correctness = "true"
                    elif success and contains_metric:
                        correctness = "likely_true"
                    elif contains_metric:
                        correctness = "unknown"
                    else:
                        correctness = "false"
                    rows.append(
                        {
                            "variant": label,
                            "task_instruction": goal,
                            "task_id": status.get("task_id", ""),
                            "task_mode": status.get("task_mode", strategy_runner._task_mode(goal)),  # pylint: disable=protected-access
                            "step": action_row.get("step"),
                            "answer_hint": prompt_line,
                            "required_metric": metric,
                            "answer_hint_contains_target_entity": contains_entity,
                            "answer_hint_contains_required_metric": contains_metric,
                            "answer_hint_used_in_final_response": answer_used,
                            "final_status_answer_matches_hint": correctness,
                            "answer_hint_harmful": bool(not success and contains_metric and not answer_used),
                            "trace_file": str(action_file),
                        }
                    )
    return rows


def _extract_events(run_dirs: dict[str, Path], root: Path, answer_usage: list[dict[str, Any]]) -> None:
    rollback_rows: list[dict[str, Any]] = []
    hint_rows: list[dict[str, Any]] = []
    for label, run_dir in run_dirs.items():
        for path in sorted((run_dir / "traces").rglob("rollback_trace.jsonl")):
            for item in _read_jsonl(path):
                item["variant"] = label
                item["trace_file"] = str(path)
                rollback_rows.append(item)
        for path in sorted((run_dir / "traces").rglob("action.jsonl")):
            for item in _read_jsonl(path):
                if item.get("prompt_hint") or item.get("matched_exploration_results"):
                    item["variant"] = label
                    item["trace_file"] = str(path)
                    hint_rows.append(item)
    _write_jsonl(root / "rollback_events.jsonl", rollback_rows)
    _write_jsonl(root / "prompt_hints.jsonl", hint_rows)
    _write_jsonl(root / "answer_hint_usage.jsonl", answer_usage)


def _remap_strategy_outputs(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = json.loads((root / "strategy_results_summary.json").read_text(encoding="utf-8"))
    if "S_baseline" in summary.get("strategies", {}):
        summary["strategies"]["V0_BASELINE"] = summary["strategies"].pop("S_baseline")
    if "S_baseline" in summary.get("run_dirs", {}):
        summary["run_dirs"]["V0_BASELINE"] = summary["run_dirs"].pop("S_baseline")

    per_task: list[dict[str, Any]] = []
    with (root / "per_task_strategy_results.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("strategy") == "S_baseline":
                row["strategy"] = "V0_BASELINE"
            per_task.append(row)

    baseline: dict[str, dict[str, Any]] = {
        row["task_id"]: row for row in per_task if row.get("strategy") == "V0_BASELINE"
    }
    for row in per_task:
        base = baseline.get(row.get("task_id", ""))
        if not base:
            continue
        row["baseline_success"] = base.get("strategy_success")
        row["baseline_steps"] = base.get("strategy_steps")
        try:
            row["step_delta"] = float(row.get("strategy_steps") or 0.0) - float(base.get("strategy_steps") or 0.0)
            row["baseline_failure_rescued"] = bool(
                float(base.get("strategy_success") or 0.0) == 0.0
                and float(row.get("strategy_success") or 0.0) >= 1.0
            )
            row["baseline_success_broken"] = bool(
                float(base.get("strategy_success") or 0.0) >= 1.0
                and float(row.get("strategy_success") or 0.0) < 1.0
            )
        except Exception:
            pass

    # Recompute aggregate fields that depend on the baseline join.
    for label, item in list(summary.get("strategies", {}).items()):
        rows = [r for r in per_task if r.get("strategy") == label]
        if not rows:
            continue
        deltas = [float(r.get("step_delta") or 0.0) for r in rows]
        item["paired_avg_step_delta"] = sum(deltas) / len(deltas)
        item["baseline_failure_rescued"] = sum(int(bool(r.get("baseline_failure_rescued"))) for r in rows)
        item["baseline_success_broken"] = sum(int(bool(r.get("baseline_success_broken"))) for r in rows)

    step_rows: list[dict[str, Any]] = []
    step_path = root / "per_step_strategy_metrics.csv"
    if step_path.exists() and step_path.stat().st_size:
        with step_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("strategy_label") == "S_baseline":
                    row["strategy_label"] = "V0_BASELINE"
                step_rows.append(row)
    (root / "fixed_20task_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(root / "per_task_results.csv", per_task)
    _write_csv(root / "per_step_metrics.csv", step_rows)
    return summary, per_task, step_rows


def _depth3_stats(root: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for item in _read_jsonl(root / "search_tree_traces.jsonl"):
        label = item.get("strategy_label", "")
        if label == "S_baseline":
            label = "V0_BASELINE"
        stats = out.setdefault(str(label), {"depth3_count": 0, "depth3_useful": 0, "depth3_harmful_or_no_gain": 0})
        for obs in item.get("observations") or []:
            if not isinstance(obs, dict) or int(obs.get("depth_reached") or 0) < 3:
                continue
            stats["depth3_count"] += 1
            if float(obs.get("evidence_gain") or 0.0) > 0 and bool((obs.get("rollback") or {}).get("success", True)):
                stats["depth3_useful"] += 1
            else:
                stats["depth3_harmful_or_no_gain"] += 1
    return out


def _write_plots(root: Path, summary: dict[str, Any], answer_usage: list[dict[str, Any]], depth3: dict[str, dict[str, int]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = list(summary.get("strategies", {}).keys())
    short = [x.replace("V0_", "V0\n").replace("V2_", "V2\n").replace("V7_", "V7\n").replace("V8_", "V8\n").replace("V9_", "V9\n").replace("V10_", "V10\n") for x in labels]

    def vals(metric: str, percent: bool = False) -> list[float]:
        data = [float(summary["strategies"][label].get(metric) or 0.0) for label in labels]
        return [x * 100.0 for x in data] if percent else data

    for metric, title, filename, pct in [
        ("success_rate", "Success rate by variant", "success_rate_by_variant.png", True),
        ("avg_episode_length", "Average episode length by variant", "avg_steps_by_variant.png", False),
        ("paired_avg_step_delta", "Paired step delta by variant", "paired_step_delta_by_variant.png", False),
        ("hint_follow_rate", "ACTION_HINT follow rate", "action_hint_follow_rate.png", True),
    ]:
        plt.figure(figsize=(11, 4.8))
        plt.bar(short, vals(metric, pct))
        plt.axhline(0, color="black", linewidth=0.8) if "delta" in metric else None
        plt.title(title)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=170)
        plt.close()

    x = np.arange(len(labels))
    plt.figure(figsize=(11, 4.8))
    plt.bar(x - 0.2, [summary["strategies"][l].get("baseline_failure_rescued", 0) for l in labels], width=0.4, label="rescued")
    plt.bar(x + 0.2, [summary["strategies"][l].get("baseline_success_broken", 0) for l in labels], width=0.4, label="broken")
    plt.xticks(x, short, rotation=30, ha="right")
    plt.legend()
    plt.title("Rescued vs broken tasks")
    plt.tight_layout()
    plt.savefig(plot_dir / "rescued_vs_broken_by_variant.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 4.8))
    plt.bar(x - 0.25, [summary["strategies"][l].get("ACTION_HINT_count", 0) for l in labels], width=0.25, label="ACTION")
    plt.bar(x, [summary["strategies"][l].get("ANSWER_HINT_count", 0) for l in labels], width=0.25, label="ANSWER")
    plt.bar(x + 0.25, [summary["strategies"][l].get("evidence_only_vague_hint_count", 0) for l in labels], width=0.25, label="vague")
    plt.xticks(x, short, rotation=30, ha="right")
    plt.legend()
    plt.title("Hint count by type")
    plt.tight_layout()
    plt.savefig(plot_dir / "hint_count_by_type.png", dpi=170)
    plt.close()

    answer_by_label = {label: {"total": 0, "metric": 0, "likely": 0} for label in labels}
    for row in answer_usage:
        label = str(row.get("variant"))
        if label not in answer_by_label:
            continue
        answer_by_label[label]["total"] += 1
        answer_by_label[label]["metric"] += int(bool(row.get("answer_hint_contains_required_metric")))
        answer_by_label[label]["likely"] += int(str(row.get("final_status_answer_matches_hint")) in {"true", "likely_true"})
    plt.figure(figsize=(11, 4.8))
    plt.bar(x - 0.2, [answer_by_label[l]["metric"] for l in labels], width=0.4, label="contains required metric")
    plt.bar(x + 0.2, [answer_by_label[l]["likely"] for l in labels], width=0.4, label="likely used/correct")
    plt.xticks(x, short, rotation=30, ha="right")
    plt.legend()
    plt.title("ANSWER_HINT correctness proxy")
    plt.tight_layout()
    plt.savefig(plot_dir / "answer_hint_correctness_proxy.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 4.8))
    plt.bar(x - 0.25, [summary["strategies"][l].get("rollback_failures", 0) for l in labels], width=0.25, label="rollback failure")
    plt.bar(x, [summary["strategies"][l].get("rollback_induced_WAIT_count", 0) for l in labels], width=0.25, label="WAIT")
    plt.bar(x + 0.25, [summary["strategies"][l].get("planned_action_suppressions", 0) for l in labels], width=0.25, label="suppress")
    plt.xticks(x, short, rotation=30, ha="right")
    plt.legend()
    plt.title("Rollback failure / WAIT by variant")
    plt.tight_layout()
    plt.savefig(plot_dir / "rollback_wait_by_variant.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 4.8))
    plt.bar(x - 0.2, [depth3.get(l, {}).get("depth3_useful", 0) for l in labels], width=0.4, label="depth3 useful")
    plt.bar(x + 0.2, [depth3.get(l, {}).get("depth3_harmful_or_no_gain", 0) for l in labels], width=0.4, label="depth3 harmful/no-gain")
    plt.xticks(x, short, rotation=30, ha="right")
    plt.legend()
    plt.title("Depth=3 useful vs harmful")
    plt.tight_layout()
    plt.savefig(plot_dir / "depth3_useful_vs_harmful.png", dpi=170)
    plt.close()


def _examples(root: Path) -> dict[str, str]:
    examples = {
        "successful_answer": "暂无",
        "shallow_answer": "暂无",
        "action_hint": "暂无",
        "avoid_hint": "暂无",
        "risk_prevented": "暂无",
        "rollback_failure": "暂无",
        "sports_failure": "暂无",
        "calendar_trace": "暂无",
    }
    for row in _read_jsonl(root / "answer_hint_usage.jsonl"):
        text = _clean(row.get("answer_hint"))
        if examples["successful_answer"] == "暂无" and str(row.get("final_status_answer_matches_hint")) in {"true", "likely_true"}:
            examples["successful_answer"] = f"`{row.get('variant')}` `{row.get('task_id')}` step={row.get('step')}: {text}"
        if examples["shallow_answer"] == "暂无" and not bool(row.get("answer_hint_contains_required_metric")):
            examples["shallow_answer"] = f"`{row.get('variant')}` `{row.get('task_id')}` step={row.get('step')}: {text}"
    for row in _read_jsonl(root / "prompt_hints.jsonl"):
        hint = _clean(row.get("prompt_hint"))
        if "Recommended next action" in hint and examples["action_hint"] == "暂无":
            examples["action_hint"] = f"`{row.get('variant')}` step={row.get('step')}: {hint[:260]}"
        if "Avoid:" in hint and examples["avoid_hint"] == "暂无":
            examples["avoid_hint"] = f"`{row.get('variant')}` step={row.get('step')}: {hint[:260]}"
    for row in _read_jsonl(root / "evidence_capsules.jsonl"):
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        action = row.get("action") if isinstance(row.get("action"), dict) else {}
        if evidence.get("type") == "RISK_HINT" and examples["risk_prevented"] == "暂无":
            examples["risk_prevented"] = f"`{row.get('strategy_label')}` step={row.get('step_id')}: risky `{action.get('label')}` was recorded, not executed."
    for row in _read_jsonl(root / "rollback_events.jsonl"):
        if not row.get("success") and examples["rollback_failure"] == "暂无":
            examples["rollback_failure"] = f"`{row.get('variant')}` {row.get('matched_by')} level={row.get('level')} file={row.get('trace_file')}"
    for row in _read_jsonl(root / "search_tree_traces.jsonl"):
        goal = _clean(row.get("goal")).lower()
        if "opentracks" in goal and "duration" in goal and examples["sports_failure"] == "暂无":
            examples["sports_failure"] = f"`{row.get('strategy_label')}` step={row.get('step')} status={row.get('status')} gate={row.get('gate_reason')}"
        if "simple calendar" in goal and examples["calendar_trace"] == "暂无":
            examples["calendar_trace"] = f"`{row.get('strategy_label')}` step={row.get('step')} status={row.get('status')} gate={row.get('gate_reason')}"
    return examples


def _write_report(root: Path, summary: dict[str, Any], per_task: list[dict[str, Any]], answer_usage: list[dict[str, Any]], depth3: dict[str, dict[str, int]], include_v9: bool) -> None:
    labels = list(summary.get("strategies", {}).keys())
    plot_dir = root / "plots"
    examples = _examples(root)
    answer_total = len(answer_usage)
    answer_metric = sum(1 for row in answer_usage if bool(row.get("answer_hint_contains_required_metric")))
    answer_likely = sum(1 for row in answer_usage if str(row.get("final_status_answer_matches_hint")) in {"true", "likely_true"})

    def fmt(value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except Exception:
            return str(value)

    lines = [
        "# Fixed 20-task MobileExplorer MCTS/Answer Extractor 对比报告",
        "",
        f"- 输出目录: `{root}`",
        f"- previous broad run: `{PREVIOUS_BROAD_RUN}`",
        f"- 任务数: `{len(TASKS_20)}`",
        f"- V9 是否运行: `{include_v9}`",
        "- 未运行 full 116-task。",
        "",
        "## 总览",
        "",
        "| variant | success | avg steps | paired delta | rescued | broken | attempts | branches | ACTION | ANSWER | AVOID | SCHEMA | vague | hint follow | rollback fail | WAIT | suppress | max depth |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in labels:
        row = summary["strategies"][label]
        lines.append(
            f"| `{label}` | {row.get('success_rate', 0)*100:.1f}% | {fmt(row.get('avg_episode_length'))} | "
            f"{float(row.get('paired_avg_step_delta') or 0):+.2f} | {row.get('baseline_failure_rescued', 0)} | "
            f"{row.get('baseline_success_broken', 0)} | {row.get('exploration_attempts', 0)} | "
            f"{row.get('explored_root_branches', 0)} | {row.get('ACTION_HINT_count', 0)} | "
            f"{row.get('ANSWER_HINT_count', 0)} | {row.get('AVOID_HINT_count', 0)} | {row.get('SCHEMA_HINT_count', 0)} | "
            f"{row.get('evidence_only_vague_hint_count', 0)} | {row.get('hint_follow_rate', 0)*100:.1f}% | "
            f"{row.get('rollback_failures', 0)} | {row.get('rollback_induced_WAIT_count', 0)} | "
            f"{row.get('planned_action_suppressions', 0)} | {row.get('max_depth_reached', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 图表",
            "",
            f"![success_rate]({plot_dir / 'success_rate_by_variant.png'})",
            "",
            f"![avg_steps]({plot_dir / 'avg_steps_by_variant.png'})",
            "",
            f"![paired_delta]({plot_dir / 'paired_step_delta_by_variant.png'})",
            "",
            f"![rescued_broken]({plot_dir / 'rescued_vs_broken_by_variant.png'})",
            "",
            f"![hint_count]({plot_dir / 'hint_count_by_type.png'})",
            "",
            f"![answer_proxy]({plot_dir / 'answer_hint_correctness_proxy.png'})",
            "",
            f"![rollback_wait]({plot_dir / 'rollback_wait_by_variant.png'})",
            "",
            f"![depth3]({plot_dir / 'depth3_useful_vs_harmful.png'})",
            "",
            "## ANSWER_HINT 诊断",
            "",
            f"- ANSWER_HINT usage records: `{answer_total}`",
            f"- contains required metric: `{answer_metric}`",
            f"- likely used/correct: `{answer_likely}`",
            "- 说明：如果 AndroidWorld checker 没直接暴露 final answer，本报告使用弱 proxy：final response 与 hint 共享数字/单位，或任务成功且 hint 含 required metric。",
            "",
            "## 必答问题",
            "",
        ]
    )
    baseline = summary["strategies"].get("V0_BASELINE", {})
    v2 = summary["strategies"].get("V2_SAFE_GREEDY", {})
    v7 = summary["strategies"].get("V7_RAW_MCTS", {})
    v8 = summary["strategies"].get("V8_SAFE_MCTS", {})
    v10 = summary["strategies"].get("V10_SAFE_MCTS_PLUS_ANSWER_EXTRACTORS", {})
    best = max(labels, key=lambda l: float(summary["strategies"][l].get("success_rate") or 0.0)) if labels else "NA"
    best_delta = min(labels, key=lambda l: float(summary["strategies"][l].get("paired_avg_step_delta") or 0.0)) if labels else "NA"
    lines.extend(
        [
            f"1. V2 是否仍安全：{'是' if v2 and v2.get('rollback_induced_WAIT_count', 1) == 0 and v2.get('evidence_only_vague_hint_count', 1) == 0 else '否/需检查'}。",
            f"2. V7 是否仍优于 V2：{'是' if v7.get('success_rate', 0) > v2.get('success_rate', 0) else '否'}。",
            f"3. V8 是否降低 V7 rollback/WAIT：rollback {v7.get('rollback_failures', 0)} -> {v8.get('rollback_failures', 0)}, WAIT {v7.get('rollback_induced_WAIT_count', 0)} -> {v8.get('rollback_induced_WAIT_count', 0)}。",
            f"4. V10 是否优于 V2/V7：success V10={v10.get('success_rate', 0)*100:.1f}%, V2={v2.get('success_rate', 0)*100:.1f}%, V7={v7.get('success_rate', 0)*100:.1f}%。",
            f"5. answer extractors 是否改善 INFO_QUERY_COUNT：看 `answer_hint_usage.jsonl`，required metric 命中 {answer_metric}/{answer_total}。",
            "6. SportsTrackerActivityDuration 是否仍失败：见 per-task 明细和 SportsTracker 样例；若仍失败，主要原因是只找到 Search/Filter，未抽到 duration/date/category 组合事实。",
            "7. Calendar query 是否因 zero-action ListInspect 更安全：Calendar 样例和 rollback/WAIT 统计用于判断；可见列表任务应避免 depth=3。",
            "8. DELETE_COMMIT 是否保护：看 DELETE_COMMIT per-task；V2/V8/V10 应无 risky speculative execution、无 broken。",
            f"9. depth=3 是否有帮助：depth3 stats = `{depth3}`。",
            f"10. 最好 rescued-broken tradeoff：`{best}`。",
            f"11. 最好 paired step delta：`{best_delta}`。",
            f"12. 下一步 116-task 候选：若满足 decision rule，优先 `{best}`；否则先修 answer extraction/rollback。",
            "13. Top failure modes: (1) ANSWER_HINT 仍可能缺 required metric；(2) OpenTracks detail/statistics 抽取不足；(3) depth=3 rollback 仍有扰动；(4) state match 不等于 hint 可用；(5) evidence yield 与 success 不充分相关。",
            "",
            "## 具体样例",
            "",
            f"- successful ANSWER_HINT: {examples['successful_answer']}",
            f"- failed/shallow ANSWER_HINT: {examples['shallow_answer']}",
            f"- useful ACTION_HINT: {examples['action_hint']}",
            f"- useful AVOID_HINT: {examples['avoid_hint']}",
            f"- prevented risky action: {examples['risk_prevented']}",
            f"- remaining rollback failure: {examples['rollback_failure']}",
            f"- SportsTracker trace: {examples['sports_failure']}",
            f"- Calendar trace: {examples['calendar_trace']}",
            "",
            "## Per-task 明细",
            "",
            "| variant | task | mode | base | variant | base steps | variant steps | delta | rescued | broken | hints | rollback fail | WAIT | max depth |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in per_task:
        lines.append(
            f"| `{row.get('strategy')}` | `{row.get('task_id')}` | `{row.get('task_mode')}` | "
            f"{row.get('baseline_success')} | {row.get('strategy_success')} | {row.get('baseline_steps')} | "
            f"{row.get('strategy_steps')} | {row.get('step_delta')} | {row.get('baseline_failure_rescued')} | "
            f"{row.get('baseline_success_broken')} | {row.get('prompt_hint_steps', 0)} | "
            f"{row.get('rollback_failures', 0)} | {row.get('rollback_induced_WAIT_count', 0)} | {row.get('max_depth_reached', 0)} |"
        )
    lines.extend(["", "## Decision Rule", ""])
    accepted = []
    for label in labels:
        if label == "V0_BASELINE":
            continue
        row = summary["strategies"][label]
        if (
            row.get("success_rate", 0.0) >= baseline.get("success_rate", 1.0)
            and row.get("baseline_failure_rescued", 0) >= row.get("baseline_success_broken", 0)
            and row.get("rollback_induced_WAIT_count", 99) <= max(1, v2.get("rollback_induced_WAIT_count", 0))
            and row.get("evidence_only_vague_hint_count", 1) == 0
            and row.get("paired_avg_step_delta", 99.0) <= max(1.0, baseline.get("avg_episode_length", 0.0) * 0.1)
        ):
            accepted.append(label)
    if accepted:
        lines.append("- 满足 20-task decision rule 的候选：" + ", ".join(f"`{x}`" for x in accepted))
        lines.append("- 仍不自动运行 full 116-task；建议人工确认 risky execution audit 和 answer_hint_usage 后再跑。")
    else:
        lines.append("- 没有 variant 完全满足 decision rule；不建议跑 full 116-task。")
        lines.append("- 下一步优先修 answer extraction，其次修 depth=3 rollback。")
    (root / "fixed_20task_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_outputs(root: Path) -> None:
    mapping = {
        "per_task_strategy_results.csv": "per_task_results_raw.csv",
        "per_step_strategy_metrics.csv": "per_step_metrics_raw.csv",
        "strategy_comparison.md": "strategy_comparison_base.md",
    }
    for src_name, dst_name in mapping.items():
        src = root / src_name
        if src.exists():
            shutil.copyfile(src, root / dst_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--variants", default="")
    parser.add_argument("--include_v9", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--analyze_only", default="")
    parser.add_argument("--extra_arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.analyze_only:
        root = Path(args.analyze_only).expanduser().resolve()
    else:
        root = Path(args.experiment_root).expanduser().resolve() / f"run_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)
    tasks = list(TASKS_20)
    (root / "tasks.txt").write_text("\n".join(tasks) + "\n", encoding="utf-8")

    if args.variants:
        variant_labels = [x.strip() for x in args.variants.split(",") if x.strip()]
    else:
        variant_labels = list(DEFAULT_CORE_VARIANTS)
        if args.include_v9 and "V9_SAFE_GREEDY_PLUS_ANSWER_EXTRACTORS" not in variant_labels:
            variant_labels.insert(-1, "V9_SAFE_GREEDY_PLUS_ANSWER_EXTRACTORS")

    run_dirs: dict[str, Path] = {}
    if args.analyze_only:
        for label in variant_labels:
            run_dir = _latest_run_dir(root, label)
            if run_dir:
                run_dirs["S_baseline" if label == "V0_BASELINE" else label] = run_dir
    else:
        for label in variant_labels:
            run_dir = _run_variant(label, root, tasks, args.extra_arg)
            run_dirs["S_baseline" if label == "V0_BASELINE" else label] = run_dir

    strategy_runner._analyze(run_dirs, root)  # pylint: disable=protected-access
    _copy_outputs(root)
    summary, per_task, _ = _remap_strategy_outputs(root)
    answer_usage = _answer_hint_usage({"V0_BASELINE" if k == "S_baseline" else k: v for k, v in run_dirs.items()}, per_task)
    _extract_events({"V0_BASELINE" if k == "S_baseline" else k: v for k, v in run_dirs.items()}, root, answer_usage)
    depth3 = _depth3_stats(root)
    _write_plots(root, summary, answer_usage, depth3)
    _write_report(root, summary, per_task, answer_usage, depth3, include_v9=bool("V9_SAFE_GREEDY_PLUS_ANSWER_EXTRACTORS" in variant_labels))
    print(json.dumps(summary.get("strategies", {}), ensure_ascii=False, indent=2))
    print(f"[fixed_20task] root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
