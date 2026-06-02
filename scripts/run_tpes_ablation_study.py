#!/usr/bin/env python3
"""Run TPES ablation variants on a fixed AndroidWorld task set and report metrics."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

TASKS_20 = [
    "ClockStopWatchRunning",
    "ExpenseDeleteSingle",
    "MarkorCreateFolder",
    "MarkorDeleteNewestNote",
    "MarkorDeleteNote",
    "NotesIsTodo",
    "NotesTodoItemCount",
    "OpenAppTaskEval",
    "SimpleCalendarEventsOnDate",
    "TasksDueOnDate",
    "NotesRecipeIngredientCount",
    "SimpleCalendarEventsInTimeRange",
    "SimpleCalendarNextEvent",
    "SimpleCalendarNextMeetingWithPerson",
    "SportsTrackerActivitiesCountForWeek",
    "SportsTrackerActivitiesOnDate",
    "SportsTrackerActivityDuration",
    "SportsTrackerTotalDistanceForCategoryOverInterval",
    "TasksDueNextWeek",
    "TasksHighPriorityTasksDueOnDate",
]

TASKS_SMOKE = [
    "ExpenseDeleteSingle",
    "NotesRecipeIngredientCount",
    "SimpleCalendarEventsInTimeRange",
    "MarkorDeleteNewestNote",
    "SportsTrackerActivityDuration",
]


COMMON_EXPLORE_ARGS = [
    "--agent_name=explore_agent_gelab",
    "--suite_family=android_world",
    "--n_task_combinations=1",
    "--fixed_task_seed",
    "--image_downsample_scale=1.0",
    "--a11y_method=uiautomator",
    "--baseline_table=results/4B_2.txt",
    "--no-explore_diagnostic_full",
    "--explore_fast_mode",
    "--explore_transaction_safe",
    "--explore_max_runs=10000",
    "--explore_max_step=10000",
    "--explore_branch_budget=3",
    "--explore_branch_depth=3",
    "--explore_back_limit=4",
    "--no-explore_planned_only",
    "--explore_fallback_safe_candidates",
    "--no-explore_safe_click_only",
    "--no-explore_skip_launcher",
    "--no-explore_filter_launcher_relevance",
    "--no-explore_skip_destructive_goals",
    "--explore_strategy=dfs",
]

VARIANTS: dict[str, dict[str, Any]] = {
    "V0_baseline": {
        "description": "Original GELAB baseline without exploration.",
        "args": ["--agent_name=gelab_agent"],
    },
    "V1_current_tpes": {
        "description": "Current broad TPES control.",
        "args": [*COMMON_EXPLORE_ARGS, "--explore_force_every_step", "--explore_variant=V1_current_tpes"],
    },
    "V2_no_prompt_injection": {
        "description": "Exploration runs but matched evidence is not injected into prompts.",
        "args": [
            *COMMON_EXPLORE_ARGS,
            "--explore_force_every_step",
            "--explore_disable_prompt_injection",
            "--explore_variant=V2_no_prompt_injection",
        ],
    },
    "V3_strict_evidence": {
        "description": "Strict ACTION/ANSWER/AVOID/SCHEMA hints only; no vague evidence-only text.",
        "args": [
            *COMMON_EXPLORE_ARGS,
            "--explore_force_every_step",
            "--explore_hint_policy=strict",
            "--explore_variant=V3_strict_evidence",
        ],
    },
    "V4_operator_boundary": {
        "description": "Operator-aware boundary-stopped search; no depth 3.",
        "args": [
            *COMMON_EXPLORE_ARGS,
            "--explore_force_every_step",
            "--explore_search_policy=operator",
            "--explore_branch_depth=2",
            "--explore_variant=V4_operator_boundary",
        ],
    },
    "V5_improved_rollback": {
        "description": "Current TPES plus improved rollback suppression rule.",
        "args": [
            *COMMON_EXPLORE_ARGS,
            "--explore_force_every_step",
            "--explore_rollback_policy=improved",
            "--explore_variant=V5_improved_rollback",
        ],
    },
    "V6_task_stage_gate": {
        "description": "Task/stage-aware exploration gate with operator logs.",
        "args": [
            *COMMON_EXPLORE_ARGS,
            "--no-explore_force_every_step",
            "--explore_search_policy=task_gate",
            "--explore_branch_depth=2",
            "--explore_variant=V6_task_stage_gate",
        ],
    },
    "V7_full_fix": {
        "description": "Strict hints + operator boundary search + improved rollback + task gate.",
        "args": [
            *COMMON_EXPLORE_ARGS,
            "--no-explore_force_every_step",
            "--explore_hint_policy=strict",
            "--explore_search_policy=task_gate",
            "--explore_rollback_policy=improved",
            "--explore_branch_depth=2",
            "--explore_variant=V7_full_fix",
        ],
    },
}


def _clean(text: Any) -> str:
    return str(text or "").replace("\n", " ").strip()


def _task_mode(goal: str) -> str:
    low = _clean(goal).lower()
    if re.search(r"\b(delete|remove|trash|discard|clear all|erase)\b", low):
        return "DELETE_COMMIT"
    if re.search(r"\b(create|add|edit|change|enter|fill|record audio with file name|named|draft)\b", low):
        return "FORM_CREATE_EDIT"
    if re.search(r"\b(open .* app|run the stopwatch|verify|turn on|turn off|toggle)\b", low):
        return "SIMPLE_VERIFY_OPEN"
    if re.search(r"\b(find|search|open .*file|open .*note|recipe named|note titled|named)\b", low):
        return "NAVIGATION_SEARCH"
    if re.search(
        r"\b(how many|what|which|who|where|when|answer with|count|duration|total|longest|next upcoming|is the|do i have)\b",
        low,
    ) or low.rstrip().endswith("?"):
        return "INFO_QUERY_COUNT"
    return "NAVIGATION_SEARCH"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _load_checkpoints(checkpoint_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not checkpoint_dir.exists():
        return rows
    for path in sorted(checkpoint_dir.glob("*.pkl.gz")):
        try:
            with gzip.open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue
        if isinstance(data, list):
            items = [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            items = [data]
        else:
            items = []
        for item in items:
            row = dict(item)
            row["_checkpoint_file"] = str(path)
            rows.append(row)
    return rows


def _run_variant(
    variant: str,
    spec: dict[str, Any],
    tasks: list[str],
    root: Path,
    extra_args: list[str],
) -> Path:
    variant_root = root / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_exploration_experiment_report.py"),
        "--run",
        f"--tasks={','.join(tasks)}",
        f"--experiment_root={variant_root}",
        "--max_cases=12",
        *spec["args"],
        *extra_args,
    ]
    log_path = variant_root / "variant_driver.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{variant}] {line}", end="")
            log.write(line)
        rc = process.wait()
        if rc != 0:
            print(f"[{variant}] exited with {rc}; continuing so partial traces can be analyzed")
    runs = sorted(variant_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise RuntimeError(f"No run directory produced for {variant}")
    return runs[0]


def _action_sequence(episode: dict[str, Any]) -> list[str]:
    data = episode.get("episode_data") if isinstance(episode.get("episode_data"), dict) else {}
    seq = data.get("action_dict") or data.get("action") or []
    out: list[str] = []
    for item in seq:
        if isinstance(item, dict):
            typ = item.get("action_type")
            if typ == "click":
                out.append(f"click@({item.get('x')},{item.get('y')})")
            elif typ == "input_text":
                out.append(f"type:{_clean(item.get('text'))[:40]}")
            elif typ == "open_app":
                out.append(f"open_app:{_clean(item.get('app_name'))}")
            elif typ == "status":
                out.append(f"status:{_clean(item.get('goal_status'))}")
            else:
                out.append(_clean(typ))
        else:
            out.append(_clean(item)[:100])
    return out


def _trace_metrics(trace_root: Path) -> dict[str, dict[str, Any]]:
    by_goal: dict[str, dict[str, Any]] = {}
    for task_dir in sorted(trace_root.iterdir()) if trace_root.exists() else []:
        if not task_dir.is_dir():
            continue
        actions = _read_jsonl(task_dir / "action.jsonl")
        explorations = _read_jsonl(task_dir / "exploration_trace.jsonl")
        matches = _read_jsonl(task_dir / "exploration_match_trace.jsonl")
        if not actions and not explorations:
            continue
        goal = _clean((actions[0] if actions else explorations[0]).get("goal"))
        if not goal:
            goal = task_dir.name
        hint_steps = sum(1 for a in actions if a.get("prompt_hint"))
        evidence_only = sum(
            1
            for a in actions
            if "Evidence only" in _clean(a.get("prompt_hint"))
            or any((r.get("hint_kind") == "evidence_only") for r in (a.get("matched_exploration_results") or []) if isinstance(r, dict))
        )
        action_hint_steps = 0
        answer_hint_steps = 0
        avoid_hint_steps = 0
        schema_hint_steps = 0
        hint_followed = 0
        harmful_hint_count = 0
        for a in actions:
            results = [r for r in (a.get("matched_exploration_results") or []) if isinstance(r, dict)]
            types = {r.get("hint_type") or r.get("hint_kind") for r in results}
            action_hint_steps += int("ACTION_HINT" in types or "actionable" in types)
            answer_hint_steps += int("ANSWER_HINT" in types)
            avoid_hint_steps += int("AVOID_HINT" in types)
            schema_hint_steps += int("SCHEMA_HINT" in types)
            action = a.get("action_dict") if isinstance(a.get("action_dict"), dict) else {}
            if action.get("action_type") == "click":
                try:
                    x, y = float(action.get("x")), float(action.get("y"))
                except Exception:
                    x, y = -9999.0, -9999.0
                for r in results:
                    cand = r.get("next_candidate") if isinstance(r.get("next_candidate"), dict) else {}
                    center = cand.get("center")
                    if isinstance(center, list) and len(center) >= 2:
                        if ((x - float(center[0])) ** 2 + (y - float(center[1])) ** 2) ** 0.5 <= 120:
                            hint_followed += 1
                            break
        rollback_failures = 0
        level_counts: dict[str, int] = {"level0": 0, "level1": 0, "level2": 0, "abort": 0}
        max_depth = 0
        operator_dist: dict[str, int] = {}
        boundary_dist: dict[str, int] = {}
        for e in explorations:
            if e.get("status") == "rollback_failed":
                rollback_failures += 1
            for rb in e.get("rollbacks") or []:
                if not isinstance(rb, dict):
                    continue
                level = _clean(rb.get("level")) or "abort"
                if level not in level_counts:
                    level_counts[level] = 0
                level_counts[level] += 1
                if not rb.get("success"):
                    rollback_failures += 1
            for obs in e.get("observations") or []:
                if not isinstance(obs, dict):
                    continue
                max_depth = max(max_depth, int(obs.get("depth_reached") or 0))
                op = _clean(obs.get("operator")) or "unknown"
                bd = _clean(obs.get("boundary_type")) or "none"
                operator_dist[op] = operator_dist.get(op, 0) + 1
                boundary_dist[bd] = boundary_dist.get(bd, 0) + 1
        planned_supp = sum(1 for a in actions if a.get("planned_action_suppressed"))
        wait_caused = sum(
            1
            for a in actions
            if a.get("planned_action_suppressed")
            or (isinstance(a.get("parsed_action"), dict) and a["parsed_action"].get("planned_action_suppressed_by_exploration_rollback"))
        )
        by_goal[goal] = {
            "trace_dir": str(task_dir),
            "exploration_attempts": len(explorations),
            "prompt_hint_steps": hint_steps,
            "evidence_only_hint_count": evidence_only,
            "action_hint_steps": action_hint_steps,
            "answer_hint_steps": answer_hint_steps,
            "avoid_hint_steps": avoid_hint_steps,
            "schema_hint_steps": schema_hint_steps,
            "hint_followed": hint_followed,
            "harmful_hint_count": harmful_hint_count,
            "rollback_failures": rollback_failures,
            "level_counts": level_counts,
            "planned_action_suppressions": planned_supp,
            "wait_caused_by_exploration": wait_caused,
            "max_exploration_depth": max_depth,
            "operator_distribution": operator_dist,
            "boundary_stop_distribution": boundary_dist,
        }
    return by_goal


def _analyze_runs(run_dirs: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    episodes_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    trace_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, run_dir in run_dirs.items():
        episodes = _load_checkpoints(run_dir / "checkpoints")
        episodes_by_variant[variant] = {
            _clean(e.get("task_template")): e for e in episodes if _clean(e.get("task_template"))
        }
        trace_by_variant[variant] = _trace_metrics(run_dir / "traces")

    baseline = episodes_by_variant.get("V0_baseline", {})
    task_names = sorted({name for rows in episodes_by_variant.values() for name in rows})
    per_task: list[dict[str, Any]] = []
    for variant, episodes in episodes_by_variant.items():
        for task in task_names:
            ep = episodes.get(task)
            base = baseline.get(task)
            if not ep:
                continue
            goal = _clean(ep.get("goal"))
            trace = trace_by_variant.get(variant, {}).get(goal, {})
            baseline_success = float(base.get("is_successful") or 0.0) if base else None
            variant_success = float(ep.get("is_successful") or 0.0)
            baseline_steps = float(base.get("episode_length") or 0.0) if base else None
            variant_steps = float(ep.get("episode_length") or 0.0)
            row = {
                "variant": variant,
                "task_id": task,
                "task_instruction": goal,
                "task_mode": _task_mode(goal),
                "baseline_success": baseline_success,
                "variant_success": variant_success,
                "baseline_steps": baseline_steps,
                "variant_steps": variant_steps,
                "step_delta": (variant_steps - baseline_steps) if baseline_steps is not None else "",
                "baseline_success_broken": bool(baseline_success == 1.0 and variant_success < 1.0),
                "baseline_failure_rescued": bool(baseline_success == 0.0 and variant_success >= 1.0),
                "final_status": "success" if variant_success >= 1.0 else "fail",
                "action_sequence": " -> ".join(_action_sequence(ep)),
                **trace,
            }
            if row["baseline_success_broken"] and row.get("prompt_hint_steps", 0):
                row["harmful_hint_count"] = int(row.get("prompt_hint_steps") or 0)
            per_task.append(row)

    summary: dict[str, Any] = {"variants": {}, "run_dirs": {k: str(v) for k, v in run_dirs.items()}}
    for variant in run_dirs:
        rows = [r for r in per_task if r["variant"] == variant]
        if not rows:
            continue
        success = sum(float(r["variant_success"]) for r in rows)
        steps = [float(r["variant_steps"]) for r in rows]
        succ_steps = [float(r["variant_steps"]) for r in rows if float(r["variant_success"]) >= 1.0]
        action_hints = sum(int(r.get("action_hint_steps") or 0) for r in rows)
        hint_follow = sum(int(r.get("hint_followed") or 0) for r in rows)
        summary["variants"][variant] = {
            "description": VARIANTS.get(variant, {}).get("description", ""),
            "num_tasks": len(rows),
            "success_rate": success / len(rows),
            "avg_episode_length": sum(steps) / len(steps),
            "avg_success_episode_length": (sum(succ_steps) / len(succ_steps)) if succ_steps else None,
            "baseline_success_broken": sum(int(r["baseline_success_broken"]) for r in rows),
            "baseline_failure_rescued": sum(int(r["baseline_failure_rescued"]) for r in rows),
            "exploration_attempts": sum(int(r.get("exploration_attempts") or 0) for r in rows),
            "prompt_hint_steps": sum(int(r.get("prompt_hint_steps") or 0) for r in rows),
            "action_hint_steps": action_hints,
            "answer_hint_steps": sum(int(r.get("answer_hint_steps") or 0) for r in rows),
            "avoid_hint_steps": sum(int(r.get("avoid_hint_steps") or 0) for r in rows),
            "schema_hint_steps": sum(int(r.get("schema_hint_steps") or 0) for r in rows),
            "evidence_only_hint_count": sum(int(r.get("evidence_only_hint_count") or 0) for r in rows),
            "hint_follow_rate": (hint_follow / action_hints) if action_hints else 0.0,
            "harmful_hint_count": sum(int(r.get("harmful_hint_count") or 0) for r in rows),
            "rollback_failures": sum(int(r.get("rollback_failures") or 0) for r in rows),
            "planned_action_suppressions": sum(int(r.get("planned_action_suppressions") or 0) for r in rows),
            "wait_caused_by_exploration": sum(int(r.get("wait_caused_by_exploration") or 0) for r in rows),
            "max_exploration_depth": max(int(r.get("max_exploration_depth") or 0) for r in rows),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in per_task for key in row.keys()})
    with (out_dir / "per_task_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_task:
            writer.writerow(row)
    _write_plots(summary, per_task, out_dir / "plots")
    _write_markdown(summary, per_task, out_dir)
    return summary


def _write_plots(summary: dict[str, Any], per_task: list[dict[str, Any]], plot_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    variants = list(summary.get("variants", {}).keys())
    if not variants:
        return
    def bar(metric: str, title: str, filename: str) -> None:
        vals = [summary["variants"][v].get(metric) or 0 for v in variants]
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(variants)), vals)
        plt.xticks(range(len(variants)), variants, rotation=35, ha="right")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=160)
        plt.close()
    bar("success_rate", "Success rate by variant", "success_rate_by_variant.png")
    bar("avg_episode_length", "Average episode length by variant", "avg_steps_by_variant.png")
    bar("hint_follow_rate", "Hint follow rate by variant", "hint_follow_rate_by_variant.png")
    bar("wait_caused_by_exploration", "Exploration-induced WAIT by variant", "rollback_wait_by_variant.png")
    broken = [summary["variants"][v].get("baseline_success_broken") or 0 for v in variants]
    rescued = [summary["variants"][v].get("baseline_failure_rescued") or 0 for v in variants]
    plt.figure(figsize=(10, 4))
    xs = range(len(variants))
    plt.bar([x - 0.2 for x in xs], broken, width=0.4, label="broken")
    plt.bar([x + 0.2 for x in xs], rescued, width=0.4, label="rescued")
    plt.xticks(list(xs), variants, rotation=35, ha="right")
    plt.legend()
    plt.title("Broken vs rescued tasks")
    plt.tight_layout()
    plt.savefig(plot_dir / "broken_vs_rescued_by_variant.png", dpi=160)
    plt.close()

    for variant in variants:
        rows = [r for r in per_task if r["variant"] == variant and r.get("step_delta") != ""]
        if not rows:
            continue
        plt.figure(figsize=(12, 5))
        plt.bar(range(len(rows)), [float(r["step_delta"]) for r in rows])
        plt.xticks(range(len(rows)), [r["task_id"] for r in rows], rotation=70, ha="right", fontsize=7)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title(f"Step delta by task: {variant}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"step_delta_{variant}.png", dpi=160)
        plt.close()


def _write_markdown(summary: dict[str, Any], per_task: list[dict[str, Any]], out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    lines: list[str] = ["# TPES 20-task Ablation Study 中文报告", ""]
    lines.append(f"- 输出目录: `{out_dir}`")
    lines.append(f"- 任务数: `{len({r['task_id'] for r in per_task})}`")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append("| variant | success | avg steps | success steps | broken | rescued | hints | action hints | answer hints | evidence-only | hint follow | rollback failures | WAIT by exploration |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for variant, row in summary.get("variants", {}).items():
        succ_steps = row.get("avg_success_episode_length")
        succ_steps_text = f"{succ_steps:.2f}" if isinstance(succ_steps, (int, float)) else "NA"
        lines.append(
            f"| `{variant}` | {row['success_rate']*100:.1f}% | {row['avg_episode_length']:.2f} | "
            f"{succ_steps_text} | "
            f"{row['baseline_success_broken']} | {row['baseline_failure_rescued']} | {row['prompt_hint_steps']} | "
            f"{row['action_hint_steps']} | {row['answer_hint_steps']} | {row['evidence_only_hint_count']} | "
            f"{row['hint_follow_rate']*100:.1f}% | {row['rollback_failures']} | {row['wait_caused_by_exploration']} |"
        )
    lines.append("")
    lines.append("## 图表")
    lines.append("")
    for name in [
        "success_rate_by_variant.png",
        "avg_steps_by_variant.png",
        "broken_vs_rescued_by_variant.png",
        "hint_follow_rate_by_variant.png",
        "rollback_wait_by_variant.png",
    ]:
        path = plot_dir / name
        if path.exists():
            lines.append(f"![{name}]({path})")
            lines.append("")
    lines.append("## 关键判断")
    lines.append("")
    current = summary.get("variants", {}).get("V1_current_tpes", {})
    baseline = summary.get("variants", {}).get("V0_baseline", {})
    if current and baseline:
        lines.append(f"- Baseline: success={baseline['success_rate']*100:.1f}%, avg steps={baseline['avg_episode_length']:.2f}.")
        lines.append(f"- Current TPES: success={current['success_rate']*100:.1f}%, avg steps={current['avg_episode_length']:.2f}.")
    improved = [
        (v, r)
        for v, r in summary.get("variants", {}).items()
        if v not in {"V0_baseline", "V1_current_tpes"}
        and current
        and (r["success_rate"] > current.get("success_rate", 0) or r["avg_episode_length"] < current.get("avg_episode_length", 1e9))
    ]
    if improved:
        lines.append("- 相对 current TPES 有改进的 variant: " + ", ".join(f"`{v}`" for v, _ in improved))
    else:
        lines.append("- 没有 variant 同时明显改善 current TPES 的成功率或步数。")
    lines.append("")
    lines.append("## Per-task 明细")
    lines.append("")
    lines.append("| variant | task | mode | baseline success | variant success | baseline steps | variant steps | delta | broken | rescued | hints | rollback fail | WAIT | max depth |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for r in per_task:
        lines.append(
            f"| `{r['variant']}` | `{r['task_id']}` | `{r['task_mode']}` | {r.get('baseline_success')} | "
            f"{r.get('variant_success')} | {r.get('baseline_steps')} | {r.get('variant_steps')} | {r.get('step_delta')} | "
            f"{r.get('baseline_success_broken')} | {r.get('baseline_failure_rescued')} | {r.get('prompt_hint_steps',0)} | "
            f"{r.get('rollback_failures',0)} | {r.get('wait_caused_by_exploration',0)} | {r.get('max_exploration_depth',0)} |"
        )
    (out_dir / "variant_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--experiment_root", default="results/tpes_ablation_study")
    parser.add_argument("--variants", default=",".join(VARIANTS.keys()))
    parser.add_argument("--tasks", default="")
    parser.add_argument("--analyze_only", default="")
    parser.add_argument("--extra_arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.analyze_only:
        root = Path(args.analyze_only).expanduser().resolve()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        root = (REPO_ROOT / args.experiment_root / f"{args.phase}_{stamp}").resolve()
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()] if args.tasks else (TASKS_SMOKE if args.phase == "smoke" else TASKS_20)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    run_dirs: dict[str, Path] = {}
    if args.analyze_only:
        for variant in variants:
            candidates = sorted((root / variant).glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                run_dirs[variant] = candidates[0]
    else:
        root.mkdir(parents=True, exist_ok=True)
        (root / "tasks.txt").write_text("\n".join(tasks) + "\n", encoding="utf-8")
        for variant in variants:
            if variant not in VARIANTS:
                raise ValueError(f"Unknown variant: {variant}")
            run_dirs[variant] = _run_variant(variant, VARIANTS[variant], tasks, root, args.extra_arg)
    summary = _analyze_runs(run_dirs, root)
    print(json.dumps(summary.get("variants", {}), ensure_ascii=False, indent=2))
    print(f"[ablation] root={root}")
    print(f"[ablation] report={root / 'variant_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
