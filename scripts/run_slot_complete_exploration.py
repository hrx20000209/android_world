#!/usr/bin/env python3
"""Run Slot-Complete Evidence-Guided Exploration experiments."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results" / "slot_complete_exploration"
LATEST_FIXED_20 = REPO_ROOT / "results" / "fixed_20task_mcts_answer" / "run_20260530T015432"
PREVIOUS_BROAD = REPO_ROOT / "results" / "tpes_androidworld_broad" / "run_20260529T151709"

_fixed_path = REPO_ROOT / "scripts" / "run_fixed_20task_mcts_answer.py"
_fixed_spec = importlib.util.spec_from_file_location("fixed20", _fixed_path)
if _fixed_spec is None or _fixed_spec.loader is None:
    raise RuntimeError(f"Could not import {_fixed_path}")
fixed20 = importlib.util.module_from_spec(_fixed_spec)
_fixed_spec.loader.exec_module(fixed20)

_strategy_runner = fixed20.strategy_runner

_slot_path = REPO_ROOT / "android_world" / "agents" / "slot_complete_evidence.py"
_slot_spec = importlib.util.spec_from_file_location("slot_complete_evidence", _slot_path)
if _slot_spec is None or _slot_spec.loader is None:
    raise RuntimeError(f"Could not import {_slot_path}")
slot_complete_evidence = importlib.util.module_from_spec(_slot_spec)
sys.modules[_slot_spec.name] = slot_complete_evidence
_slot_spec.loader.exec_module(slot_complete_evidence)

TASKS_20 = list(fixed20.TASKS_20)

FIXED_SHARED_ARGS = list(fixed20.FIXED_SHARED_ARGS)

VARIANTS: dict[str, dict[str, Any]] = {
    "V0_BASELINE": fixed20.VARIANTS["V0_BASELINE"],
    "V2_SAFE_GREEDY": fixed20.VARIANTS["V2_SAFE_GREEDY"],
    "V10_CURRENT": {
        "description": "Current V10 control.",
        "args": [
            *FIXED_SHARED_ARGS,
            "--explore_search_strategy=mcts",
            "--explore_safe_mcts",
            "--explore_answer_extractors",
            "--explore_variant=V10_CURRENT",
        ],
    },
    "V10_SAFE_MCTS_PLUS_ANSWER_EXTRACTORS": {
        "description": "Safe MCTS plus lightweight answer extractors.",
        "args": [
            *FIXED_SHARED_ARGS,
            "--explore_search_strategy=mcts",
            "--explore_safe_mcts",
            "--explore_answer_extractors",
            "--explore_slot_complete",
            "--explore_variant=V10_SAFE_MCTS_PLUS_ANSWER_EXTRACTORS",
        ],
    },
    "V11_SLOT_COMPLETE_SAFE_GREEDY": {
        "description": "Safe greedy with slot-complete evidence extraction.",
        "args": [
            *FIXED_SHARED_ARGS,
            "--explore_search_strategy=greedy",
            "--explore_answer_extractors",
            "--explore_slot_complete",
            "--explore_variant=V11_SLOT_COMPLETE_SAFE_GREEDY",
        ],
    },
    "V12_SLOT_COMPLETE_BEST_FIRST": {
        "description": "Best-first missing-slot search.",
        "args": [
            *FIXED_SHARED_ARGS,
            "--explore_search_strategy=best_first",
            "--explore_answer_extractors",
            "--explore_slot_complete",
            "--explore_variant=V12_SLOT_COMPLETE_BEST_FIRST",
        ],
    },
    "V13_SLOT_COMPLETE_POLICY_SWITCHER": {
        "description": "Slot-aware policy switcher.",
        "args": [
            *FIXED_SHARED_ARGS,
            "--explore_search_strategy=best_first",
            "--explore_answer_extractors",
            "--explore_slot_complete",
            "--explore_slot_policy_switcher",
            "--explore_variant=V13_SLOT_COMPLETE_POLICY_SWITCHER",
        ],
    },
    "V14_NO_ACTION_HINT_SLOT_COMPLETE": {
        "description": "Slot-aware policy switcher without ACTION_HINT.",
        "args": [
            *FIXED_SHARED_ARGS,
            "--explore_search_strategy=best_first",
            "--explore_answer_extractors",
            "--explore_slot_complete",
            "--explore_slot_policy_switcher",
            "--explore_no_action_hint",
            "--explore_variant=V14_NO_ACTION_HINT_SLOT_COMPLETE",
        ],
    },
}

DEFAULT_VARIANTS = [
    "V0_BASELINE",
    "V10_SAFE_MCTS_PLUS_ANSWER_EXTRACTORS",
    "V11_SLOT_COMPLETE_SAFE_GREEDY",
    "V12_SLOT_COMPLETE_BEST_FIRST",
]


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _observation_rows_from_search_trace(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = []
    if (root / "search_tree_traces.jsonl").exists():
        paths.append(root / "search_tree_traces.jsonl")
    paths.extend(sorted((root / "traces").rglob("exploration_trace.jsonl")))
    seen: set[str] = set()
    for path in paths:
        if str(path) in seen:
            continue
        seen.add(str(path))
        for item in _read_jsonl(path):
            goal = _clean(item.get("goal") or item.get("task_id"))
            if not goal:
                continue
            root_labels = item.get("root_a11y") or []
            if root_labels:
                rows.append(
                    {
                        "task_instruction": goal,
                        "step_id": item.get("step"),
                        "path": str(path),
                        "activity": item.get("root_activity", ""),
                        "labels": [x.get("text") if isinstance(x, dict) else x for x in root_labels],
                        "observation_source": "root_a11y",
                    }
                )
            for obs in item.get("observations") or []:
                if not isinstance(obs, dict):
                    continue
                rows.append(
                    {
                        "task_instruction": goal,
                        "step_id": item.get("step"),
                        "path": str(path),
                        "activity": obs.get("after_activity") or item.get("root_activity", ""),
                        "labels": list(obs.get("observed_elements") or obs.get("labels") or []),
                        "observation_source": "exploration_observation",
                    }
                )
                for step in obs.get("steps") or []:
                    if isinstance(step, dict):
                        rows.append(
                            {
                                "task_instruction": goal,
                                "step_id": step.get("depth") or item.get("step"),
                                "path": str(path),
                                "activity": step.get("after_activity") or obs.get("after_activity") or "",
                                "labels": list(step.get("observed_elements") or []),
                                "observation_source": f"depth_{step.get('depth')}",
                            }
                        )
    return rows


def _observation_rows_from_capsules(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [root / "evidence_capsules.jsonl", *(root / "traces").rglob("evidence_capsules.jsonl")]:
        for item in _read_jsonl(path):
            goal = _clean(item.get("task_id"))
            labels = []
            child = item.get("child_state") if isinstance(item.get("child_state"), dict) else {}
            delta = item.get("delta") if isinstance(item.get("delta"), dict) else {}
            labels.extend(child.get("salient_labels") or [])
            labels.extend(delta.get("new_labels") or [])
            rows.append(
                {
                    "task_instruction": goal,
                    "step_id": item.get("step_id"),
                    "path": str(path),
                    "activity": child.get("activity", ""),
                    "screen_role": child.get("screen_role", ""),
                    "labels": labels,
                    "observation_source": "evidence_capsule",
                }
            )
    return rows


def run_posthoc_audit(output_root: Path, source_roots: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for source_root in source_roots:
        if not source_root.exists():
            continue
        observations = _observation_rows_from_search_trace(source_root) + _observation_rows_from_capsules(source_root)
        for obs in observations:
            labels = [slot_complete_evidence.clean(x) for x in obs.get("labels") or [] if slot_complete_evidence.clean(x)]
            goal = _clean(obs.get("task_instruction"))
            if not goal or not labels:
                continue
            slots = slot_complete_evidence.parse_task_slots(goal)
            evidence = slot_complete_evidence.extract_slot_evidence(
                labels,
                goal,
                activity=_clean(obs.get("activity")),
                screen_role=_clean(obs.get("screen_role")),
            )
            rows.append(
                {
                    "task_id": goal[:120],
                    "task_instruction": goal,
                    "task_mode": slots.task_mode,
                    "task_subtype": slots.task_subtype,
                    "observation_source": obs.get("observation_source"),
                    "step_id": obs.get("step_id"),
                    "path": obs.get("path"),
                    "package": "",
                    "activity": obs.get("activity", ""),
                    "screen_role": evidence.get("screen_role", obs.get("screen_role", "")),
                    "extracted_slots": evidence.get("extracted_slots", {}),
                    "missing_slots": evidence.get("missing_slots", []),
                    "slot_coverage": evidence.get("slot_coverage", 0.0),
                    "answer_reconstructable": evidence.get("answer_reconstructable", False),
                    "reason_if_not_reconstructable": evidence.get("reason_if_not_reconstructable", ""),
                    "facts": evidence.get("facts", []),
                    "source_root": str(source_root),
                }
            )
    _write_jsonl(output_root / "posthoc_slot_audit.jsonl", rows)
    by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["task_id"])
        item = by_task.setdefault(
            key,
            {
                "task_id": key,
                "task_mode": row.get("task_mode"),
                "task_subtype": row.get("task_subtype"),
                "observations": 0,
                "max_slot_coverage": 0.0,
                "answer_reconstructable": False,
                "best_path": "",
                "missing_slots_at_best": "",
            },
        )
        item["observations"] += 1
        cov = float(row.get("slot_coverage") or 0.0)
        if cov >= float(item["max_slot_coverage"]):
            item["max_slot_coverage"] = cov
            item["best_path"] = row.get("path", "")
            item["missing_slots_at_best"] = ",".join(row.get("missing_slots") or [])
        item["answer_reconstructable"] = bool(item["answer_reconstructable"] or row.get("answer_reconstructable"))
    summary_rows = list(by_task.values())
    _write_csv(output_root / "posthoc_slot_audit_summary.csv", summary_rows)
    _write_posthoc_report(output_root, summary_rows)
    return rows, summary_rows


def _write_posthoc_report(output_root: Path, rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    reconstructable = sum(1 for r in rows if r.get("answer_reconstructable"))
    lines = [
        "# Post-hoc Evidence Slot Audit",
        "",
        f"- audited tasks: `{total}`",
        f"- answer reconstructable from previous traces: `{reconstructable}`",
        "",
        "## 结论",
        "",
        "如果某个失败任务的 `max_slot_coverage` 已经接近 1 但没有注入正确 hint，主要问题是 extractor / prompt gating；如果 coverage 很低，则主要问题是 search coverage。",
        "",
        "| task | subtype | observations | max coverage | reconstructable | missing slots at best |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in sorted(rows, key=lambda r: (str(r.get("task_subtype")), -float(r.get("max_slot_coverage") or 0.0)))[:80]:
        lines.append(
            f"| `{row.get('task_id')}` | `{row.get('task_subtype')}` | {row.get('observations')} | "
            f"{float(row.get('max_slot_coverage') or 0.0):.2f} | {row.get('answer_reconstructable')} | "
            f"`{row.get('missing_slots_at_best')}` |"
        )
    (output_root / "posthoc_slot_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _augment_task_slots(root: Path, per_task: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in per_task:
        goal = _clean(row.get("task_instruction"))
        slots = slot_complete_evidence.parse_task_slots(goal)
        row["task_subtype"] = slots.task_subtype
        row["required_slots"] = ",".join(slots.required_slots)
        rows.append(
            {
                "strategy": row.get("strategy"),
                "task_id": row.get("task_id"),
                "task_instruction": goal,
                **slots.to_dict(),
            }
        )
    fixed20._write_csv(root / "per_task_results.csv", per_task)
    _write_jsonl(root / "task_slots.jsonl", rows)
    return per_task


def _answer_slot_usage(root: Path, answer_usage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in answer_usage:
        text = _clean(row.get("answer_hint"))
        goal = _clean(row.get("task_instruction"))
        evidence = slot_complete_evidence.extract_slot_evidence([text], goal)
        row["slot_complete"] = bool(evidence.get("slot_complete"))
        row["slot_coverage"] = evidence.get("slot_coverage")
        row["missing_slots"] = ",".join(evidence.get("missing_slots") or [])
        out.append(row)
    _write_jsonl(root / "answer_hint_usage.jsonl", out)
    return out


def _exploration_latency_rows(run_dirs: dict[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for label, run_dir in run_dirs.items():
        variant = "V0_BASELINE" if label == "S_baseline" else label
        if variant == "V0_BASELINE":
            continue
        for path in sorted((run_dir / "traces").rglob("exploration_trace.jsonl")):
            for trace in _read_jsonl(path):
                observations = [o for o in trace.get("observations") or [] if isinstance(o, dict)]
                base = {
                    "variant": variant,
                    "task_instruction": _clean(trace.get("goal")),
                    "task_id": path.parent.name,
                    "step": trace.get("step"),
                    "status": trace.get("status"),
                    "gate_enabled": trace.get("gate_enabled"),
                    "trigger_reason": trace.get("trigger_reason"),
                    "lightweight_a11y_trace": trace.get("lightweight_a11y_trace"),
                    "total_latency_ms": float(trace.get("latency_ms") or 0.0),
                    "root_state_fetch_ms": float(trace.get("root_state_fetch_ms") or 0.0),
                    "root_a11y_dump_ms": float(trace.get("root_a11y_dump_ms") or 0.0),
                    "root_summary_ms": float(trace.get("root_summary_ms") or 0.0),
                    "root_a11y_trace_ms": float(trace.get("root_a11y_trace_ms") or 0.0),
                    "exploration_state_fetch_ms": float(trace.get("exploration_state_fetch_ms") or 0.0),
                    "exploration_a11y_dump_ms": float(trace.get("exploration_a11y_dump_ms") or 0.0),
                    "exploration_summary_ms": float(trace.get("exploration_summary_ms") or 0.0),
                    "exploration_a11y_trace_ms": float(trace.get("exploration_a11y_trace_ms") or 0.0),
                    "observation_count": len(observations),
                    "max_depth": max([int(o.get("depth_reached") or 0) for o in observations] or [0]),
                    "trace_file": str(path),
                }
                rows.append({**base, "row_type": "step", "branch_id": ""})
                for obs in observations:
                    branch_row = {
                        **base,
                        "row_type": "branch",
                        "branch_id": obs.get("branch_id"),
                        "operator": obs.get("operator"),
                        "boundary_type": obs.get("boundary_type"),
                        "evidence_type": obs.get("evidence_type"),
                        "evidence_gain": obs.get("evidence_gain"),
                        "branch_state_fetch_ms": float(obs.get("state_fetch_ms") or 0.0),
                        "branch_a11y_dump_ms": float(obs.get("a11y_dump_ms") or 0.0),
                        "branch_summary_ms": float(obs.get("summary_ms") or 0.0),
                        "branch_a11y_trace_ms": float(obs.get("a11y_trace_ms") or 0.0),
                        "branch_depth": int(obs.get("depth_reached") or 0),
                    }
                    rows.append(branch_row)
    for variant in sorted({str(r.get("variant")) for r in rows if r.get("variant")}):
        step_rows = [r for r in rows if r.get("variant") == variant and r.get("row_type") == "step"]
        branch_rows = [r for r in rows if r.get("variant") == variant and r.get("row_type") == "branch"]

        def mean(items: list[dict[str, Any]], key: str) -> float:
            values = [float(x.get(key) or 0.0) for x in items]
            return round(sum(values) / len(values), 3) if values else 0.0

        summary[variant] = {
            "variant": variant,
            "steps_logged": len(step_rows),
            "completed_steps": sum(1 for r in step_rows if r.get("status") == "completed"),
            "branches_logged": len(branch_rows),
            "mean_step_latency_ms": mean(step_rows, "total_latency_ms"),
            "mean_completed_step_latency_ms": mean([r for r in step_rows if r.get("status") == "completed"], "total_latency_ms"),
            "mean_branch_state_fetch_ms": mean(branch_rows, "branch_state_fetch_ms"),
            "mean_branch_a11y_dump_ms": mean(branch_rows, "branch_a11y_dump_ms"),
            "mean_branch_summary_ms": mean(branch_rows, "branch_summary_ms"),
            "mean_branch_a11y_trace_ms": mean(branch_rows, "branch_a11y_trace_ms"),
            "mean_root_a11y_dump_ms": mean(step_rows, "root_a11y_dump_ms"),
            "mean_root_summary_ms": mean(step_rows, "root_summary_ms"),
            "mean_root_a11y_trace_ms": mean(step_rows, "root_a11y_trace_ms"),
            "max_depth": max([int(r.get("max_depth") or 0) for r in step_rows] or [0]),
        }
    return rows, list(summary.values())


def _plot_latency_metrics(root: Path, latency_summary: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    if not latency_summary:
        return
    plot_dir = root / "plots"
    plot_dir.mkdir(exist_ok=True)
    labels = [str(r.get("variant")) for r in latency_summary]
    short = [x.replace("_", "\n", 2) for x in labels]
    x = np.arange(len(labels))
    width = 0.22
    plt.figure(figsize=(12, 5))
    plt.bar(x - width, [float(r.get("mean_branch_a11y_dump_ms") or 0.0) for r in latency_summary], width=width, label="a11y dump/parse")
    plt.bar(x, [float(r.get("mean_branch_summary_ms") or 0.0) for r in latency_summary], width=width, label="semantic summary")
    plt.bar(x + width, [float(r.get("mean_branch_a11y_trace_ms") or 0.0) for r in latency_summary], width=width, label="trace summary")
    plt.xticks(x, short, rotation=25, ha="right")
    plt.ylabel("ms per explored branch")
    plt.title("Exploration branch latency components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "exploration_latency_components.png", dpi=170)
    plt.close()


def _plot_slot_metrics(root: Path, summary: dict[str, Any], answer_usage: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = root / "plots"
    plot_dir.mkdir(exist_ok=True)
    labels = list(summary.get("strategies", {}).keys())
    short = [x.replace("_", "\n", 2) for x in labels]
    by_label = {label: {"total": 0, "complete": 0, "metric": 0} for label in labels}
    for row in answer_usage:
        label = str(row.get("variant"))
        if label not in by_label:
            continue
        by_label[label]["total"] += 1
        by_label[label]["complete"] += int(bool(row.get("slot_complete")))
        by_label[label]["metric"] += int(bool(row.get("answer_hint_contains_required_metric")))
    x = range(len(labels))
    plt.figure(figsize=(11, 4.8))
    plt.bar(x, [
        (100.0 * by_label[l]["complete"] / by_label[l]["total"]) if by_label[l]["total"] else 0.0
        for l in labels
    ])
    plt.xticks(list(x), short, rotation=30, ha="right")
    plt.ylabel("percent")
    plt.title("ANSWER_HINT slot completeness rate")
    plt.tight_layout()
    plt.savefig(plot_dir / "slot_completeness_by_variant.png", dpi=170)
    plt.close()

    plt.figure(figsize=(11, 4.8))
    plt.bar(x, [
        (100.0 * by_label[l]["metric"] / by_label[l]["total"]) if by_label[l]["total"] else 0.0
        for l in labels
    ])
    plt.xticks(list(x), short, rotation=30, ha="right")
    plt.ylabel("percent")
    plt.title("Required metric hit rate")
    plt.tight_layout()
    plt.savefig(plot_dir / "required_metric_hit_rate.png", dpi=170)
    plt.close()


def _write_slot_report(
    root: Path,
    summary: dict[str, Any],
    per_task: list[dict[str, Any]],
    answer_usage: list[dict[str, Any]],
    posthoc_summary: list[dict[str, Any]],
    phase_decision: dict[str, Any],
    latency_summary: list[dict[str, Any]] | None = None,
) -> None:
    labels = list(summary.get("strategies", {}).keys())
    baseline = summary["strategies"].get("V0_BASELINE", {})
    best = max((l for l in labels if l != "V0_BASELINE"), key=lambda l: float(summary["strategies"][l].get("success_rate") or 0.0), default="")
    lines = [
        "# Slot-Complete Evidence-Guided Exploration 中文报告",
        "",
        f"- 输出目录: `{root}`",
        f"- post-hoc source 1: `{LATEST_FIXED_20}`",
        f"- post-hoc source 2: `{PREVIOUS_BROAD}`",
        "- 未无条件运行 full 116-task。",
        "",
        "## 总览",
        "",
        "| variant | success | avg steps | paired delta | rescued | broken | ANSWER | ACTION | vague | rollback fail | WAIT |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in labels:
        item = summary["strategies"][label]
        lines.append(
            f"| `{label}` | {float(item.get('success_rate') or 0.0)*100:.1f}% | "
            f"{float(item.get('avg_episode_length') or 0.0):.2f} | {float(item.get('paired_avg_step_delta') or 0.0):+.2f} | "
            f"{item.get('baseline_failure_rescued', 0)} | {item.get('baseline_success_broken', 0)} | "
            f"{item.get('ANSWER_HINT_count', 0)} | {item.get('ACTION_HINT_count', 0)} | "
            f"{item.get('evidence_only_vague_hint_count', 0)} | {item.get('rollback_failures', 0)} | "
            f"{item.get('rollback_induced_WAIT_count', 0)} |"
        )
    metric_total = len(answer_usage)
    metric_hit = sum(1 for r in answer_usage if bool(r.get("answer_hint_contains_required_metric")))
    slot_complete = sum(1 for r in answer_usage if bool(r.get("slot_complete")))
    lines.extend(
        [
            "",
            "## Slot 诊断",
            "",
            f"- ANSWER_HINT records: `{metric_total}`",
            f"- required metric hit: `{metric_hit}/{metric_total}`",
            f"- slot-complete hints: `{slot_complete}/{metric_total}`",
            f"- post-hoc audited tasks: `{len(posthoc_summary)}`",
            f"- post-hoc reconstructable tasks: `{sum(1 for r in posthoc_summary if r.get('answer_reconstructable'))}`",
            "",
            "## 图表",
            "",
            f"![success]({root / 'plots' / 'success_rate_by_variant.png'})",
            "",
            f"![steps]({root / 'plots' / 'avg_steps_by_variant.png'})",
            "",
            f"![rescued_broken]({root / 'plots' / 'rescued_vs_broken_by_variant.png'})",
            "",
            f"![slot_complete]({root / 'plots' / 'slot_completeness_by_variant.png'})",
            "",
            f"![metric_hit]({root / 'plots' / 'required_metric_hit_rate.png'})",
            "",
            f"![latency]({root / 'plots' / 'exploration_latency_components.png'})",
            "",
            "## Exploration Latency",
            "",
            "| variant | completed explore steps | branches | mean completed step ms | mean branch state-fetch ms | mean branch a11y dump ms | mean branch summary ms | mean branch trace ms | max depth |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in latency_summary or []:
        lines.append(
            f"| `{row.get('variant')}` | {row.get('completed_steps')} | {row.get('branches_logged')} | "
            f"{float(row.get('mean_completed_step_latency_ms') or 0.0):.1f} | "
            f"{float(row.get('mean_branch_state_fetch_ms') or 0.0):.1f} | "
            f"{float(row.get('mean_branch_a11y_dump_ms') or 0.0):.1f} | "
            f"{float(row.get('mean_branch_summary_ms') or 0.0):.3f} | "
            f"{float(row.get('mean_branch_a11y_trace_ms') or 0.0):.3f} | {row.get('max_depth')} |"
        )
    lines.extend(
        [
            "",
            "Latency 解释：`a11y dump/parse` 来自 AndroidWorld controller 的真实 a11y 获取与 UIElement 解析；`semantic summary` 和 `trace summary` 是从已获取的 `State.ui_elements` 中做轻量文本/结构摘要的本地开销。当前代码能缓存 root anchors、只保存 class/resource-id 摘要，但 AndroidWorld 的 uiautomator/gRPC 接口仍然需要整页 observation，不能真正只 dump changed subtree。",
            "",
            "## 必答问题",
            "",
            "1. performance 是 search coverage 还是 extractor quality 限制：看 `posthoc_slot_audit_summary.csv`。coverage 高但 reconstructable false 多为 extractor/slot gate；coverage 低多为 search miss。",
            "2. slot-complete hinting 是否提升 required metric hit：见 `required_metric_hit_rate.png` 和 `answer_hint_usage.jsonl`。",
            "3. removing ACTION_HINT 是否提升稳定性：比较 V13 与 V14 的 broken、steps 和 success。",
            "4. best-first missing-slot search 是否优于 greedy：比较 V11 与 V12。",
            "5. rescued/broken 任务：见 `per_task_results.csv` 的 `baseline_failure_rescued` 和 `baseline_success_broken`。",
            "6. OpenTracks 失败：若 `posthoc_slot_audit_summary.csv` 中 OpenTracks coverage 低，是没有到 detail/statistics；若 coverage 高但 hint 不完整，是 metric extraction 问题。",
            "7. Calendar 失败：重点检查 screen_role 是否为 EventCreationForm/DatePicker，以及 event_title/event_time/person slots。",
            "8. Tasks 失败：重点检查是否仍把 FilterOrDatePicker / FilterCreationForm 误判成 TaskList。",
            "9. Joplin 失败：重点检查 target recipe 是否打开，以及 amount_value/unit 是否同时出现。",
            f"10. 当前候选：`{best}`。",
            f"11. full 116-task 是否启动：`{phase_decision.get('run_full_116', False)}`；原因：{phase_decision.get('reason', '')}",
            "",
            "## Phase Decision",
            "",
            "```json",
            json.dumps(phase_decision, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Per-task 摘要",
            "",
            "| variant | task | subtype | base | success | steps | delta | rescued | broken | required slots |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in per_task:
        lines.append(
            f"| `{row.get('strategy')}` | `{row.get('task_id')}` | `{row.get('task_subtype')}` | "
            f"{row.get('baseline_success')} | {row.get('strategy_success')} | {row.get('strategy_steps')} | "
            f"{row.get('step_delta')} | {row.get('baseline_failure_rescued')} | "
            f"{row.get('baseline_success_broken')} | `{row.get('required_slots')}` |"
        )
    (root / "slot_complete_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _phase_decision(summary: dict[str, Any], answer_usage: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = summary["strategies"].get("V0_BASELINE", {})
    v10 = summary["strategies"].get("V10_CURRENT", summary["strategies"].get("V10_SAFE_MCTS_PLUS_ANSWER_EXTRACTORS", {}))
    v10_metric = sum(1 for r in answer_usage if r.get("variant") in {"V10_CURRENT", "V10_SAFE_MCTS_PLUS_ANSWER_EXTRACTORS"} and bool(r.get("answer_hint_contains_required_metric")))
    accepted = []
    for label, item in summary.get("strategies", {}).items():
        if not label.startswith(("V11", "V12", "V13", "V14")):
            continue
        metric_hit = sum(1 for r in answer_usage if r.get("variant") == label and bool(r.get("answer_hint_contains_required_metric")))
        if (
            float(item.get("success_rate") or 0.0) >= float(baseline.get("success_rate") or 0.0) + 0.05
            and int(item.get("baseline_failure_rescued") or 0) >= int(item.get("baseline_success_broken") or 0)
            and float(item.get("avg_episode_length") or 99.0) <= float(baseline.get("avg_episode_length") or 0.0) + 0.5
            and int(item.get("rollback_failures") or 0) == 0
            and int(item.get("evidence_only_vague_hint_count") or 0) == 0
            and metric_hit > v10_metric
        ):
            accepted.append(label)
    return {
        "accepted_for_40_task": accepted,
        "run_40_task": bool(accepted),
        "run_full_116": False,
        "reason": "20-task threshold passed; 40-task diagnostic is allowed." if accepted else "No slot-complete variant passed all Phase C thresholds.",
    }


def _run_40_task_diagnostic(root: Path, best_label: str, extra_args: list[str]) -> None:
    # A diagnostic hook is provided, but intentionally capped at 40 and never full 116.
    tasks = list(_strategy_runner.TASKS_20)[:40] if hasattr(_strategy_runner, "TASKS_20") else TASKS_20
    tasks = tasks[:40]
    diag_root = root / "phase_c_40task"
    diag_root.mkdir(exist_ok=True)
    for label in ["V0_BASELINE", "V2_SAFE_GREEDY", "V10_CURRENT", best_label]:
        if label in VARIANTS:
            _run_variant(label, diag_root, tasks, extra_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--analyze_only", default="")
    parser.add_argument("--posthoc_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_phase_c", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--extra_arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.analyze_only:
        root = Path(args.analyze_only).expanduser().resolve()
    else:
        root = Path(args.experiment_root).expanduser().resolve() / f"run_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks.txt").write_text("\n".join(TASKS_20) + "\n", encoding="utf-8")

    posthoc_rows, posthoc_summary = run_posthoc_audit(root, [LATEST_FIXED_20, PREVIOUS_BROAD])
    if args.posthoc_only:
        print(f"[slot_complete] posthoc rows={len(posthoc_rows)} root={root}")
        return 0

    labels = [x.strip() for x in args.variants.split(",") if x.strip()]
    run_dirs: dict[str, Path] = {}
    if args.analyze_only:
        for label in labels:
            run_dir = _latest_run_dir(root, label)
            if run_dir:
                run_dirs["S_baseline" if label == "V0_BASELINE" else label] = run_dir
    else:
        for label in labels:
            if label == "V15_ORACLE_POSTHOC_HINT_DIAGNOSTIC":
                continue
            run_dir = _run_variant(label, root, TASKS_20, args.extra_arg)
            run_dirs["S_baseline" if label == "V0_BASELINE" else label] = run_dir

    _strategy_runner._analyze(run_dirs, root)  # pylint: disable=protected-access
    fixed20._copy_outputs(root)  # pylint: disable=protected-access
    summary, per_task, _ = fixed20._remap_strategy_outputs(root)  # pylint: disable=protected-access
    per_task = _augment_task_slots(root, per_task)
    shutil.copyfile(root / "fixed_20task_summary.json", root / "slot_complete_summary.json")
    answer_usage = fixed20._answer_hint_usage({"V0_BASELINE" if k == "S_baseline" else k: v for k, v in run_dirs.items()}, per_task)  # pylint: disable=protected-access
    answer_usage = _answer_slot_usage(root, answer_usage)
    fixed20._extract_events({"V0_BASELINE" if k == "S_baseline" else k: v for k, v in run_dirs.items()}, root, answer_usage)  # pylint: disable=protected-access
    latency_rows, latency_summary = _exploration_latency_rows(run_dirs)
    _write_csv(root / "exploration_latency_by_branch.csv", latency_rows)
    _write_csv(root / "exploration_latency_summary.csv", latency_summary)
    depth3 = fixed20._depth3_stats(root)  # pylint: disable=protected-access
    fixed20._write_plots(root, summary, answer_usage, depth3)  # pylint: disable=protected-access
    _plot_slot_metrics(root, summary, answer_usage)
    _plot_latency_metrics(root, latency_summary)
    decision = _phase_decision(summary, answer_usage)
    if args.run_phase_c and decision.get("accepted_for_40_task"):
        _run_40_task_diagnostic(root, decision["accepted_for_40_task"][0], args.extra_arg)
    _write_slot_report(root, summary, per_task, answer_usage, posthoc_summary, decision, latency_summary)
    print(json.dumps({"root": str(root), "phase_decision": decision, "strategies": summary.get("strategies", {})}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
