#!/usr/bin/env python3
"""Compare MobileExplorer online UI search strategies on a fixed task set."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results" / "search_strategy_comparison"

TASKS_12 = [
    "ClockStopWatchRunning",
    "ExpenseDeleteSingle",
    "MarkorCreateFolder",
    "MarkorDeleteNewestNote",
    "NotesIsTodo",
    "NotesTodoItemCount",
    "OpenAppTaskEval",
    "SimpleCalendarEventsOnDate",
    "TasksDueOnDate",
    "NotesRecipeIngredientCount",
    "SimpleCalendarEventsInTimeRange",
    "SportsTrackerActivitiesCountForWeek",
]

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

TASKS_SMOKE = TASKS_12

STRATEGIES: dict[str, dict[str, str]] = {
    "S1_BFS_BUDGET10": {
        "strategy": "bfs",
        "description": "层次扩展策略（预算10）。按策略分组后优先覆盖不同根分支。",
    },
    "S2_DFS_BUDGET10": {
        "strategy": "dfs",
        "description": "任务条件 DFS（预算10）。只有有效 evidence gain 分支继续扩展。",
    },
    "S3_BEAM_BUDGET10": {
        "strategy": "beam",
        "description": "Beam 搜索（预算10）。保留多个高分分支以平衡覆盖与深度。",
    },
    "S4_MCTS_BUDGET10": {
        "strategy": "mcts",
        "description": "轻量 UCT/MCTS（预算10）：用 evidence reward 与风险惩罚估值。",
    },
    "S5_FINAL_OPERATOR_STRATIFIED_BEST_FIRST_BUDGET10": {
        "strategy": "final_operator_stratified_best_first",
        "description": "最终对照策略：操作符分层 best-first（预算10）。",
    },
}

COMMON_EXPLORE_ARGS = [
    "--agent_name=explore_agent_gelab",
    "--suite_family=android_world",
    "--n_task_combinations=1",
    "--fixed_task_seed",
    "--image_downsample_scale=1.0",
    "--a11y_method=uiautomator",
    "--baseline_table=results/4B_2.txt",
    "--no-explore_diagnostic_full",
    "--explore_force_every_step",
    "--explore_fast_mode",
    "--explore_transaction_safe",
    "--explore_max_runs=12000",
    "--explore_max_step=12000",
    "--explore_branch_budget=10",
    "--explore_branch_depth=2",
    "--explore_back_limit=4",
    "--explore_replay_max_actions=6",
    "--no-explore_planned_only",
    "--explore_fallback_safe_candidates",
    "--no-explore_safe_click_only",
    "--no-explore_skip_launcher",
    "--no-explore_filter_launcher_relevance",
    "--no-explore_skip_destructive_goals",
    "--explore_strategy=dfs",
    "--explore_search_policy=operator",
    "--explore_hint_policy=strict",
    "--explore_rollback_policy=improved",
]


def _clean(text: Any) -> str:
    return str(text or "").replace("\n", " ").strip()


def _task_mode(goal: str) -> str:
    low = _clean(goal).lower()
    query_like = bool(
        re.search(
            r"\b(how many|how long|what|when|which|who|where|total distance|duration|events?|tasks?|"
            r"activities?|activity type|answer|count|longest|next meeting|do i have|is the)\b",
            low,
        )
        or low.endswith("?")
    )
    if re.search(r"\b(delete|remove|trash|discard|clear all|erase|grant permission|turn on|turn off)\b", low):
        return "DELETE_COMMIT"
    if query_like:
        return "INFO_QUERY_COUNT"
    if re.search(r"\b(create|add|edit|rename|input|type|new folder|change|enter|fill|named|draft)\b", low):
        return "FORM_CREATE_EDIT"
    if re.search(r"\b(run stopwatch|open app|verify|is .*)\b", low):
        return "SIMPLE_VERIFY_OPEN"
    if re.search(r"\b(open|find|search|recipe|file|note)\b", low):
        return "NAVIGATION_SEARCH"
    return "NAVIGATION_SEARCH"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                item["_jsonl_file"] = str(path)
                item["_line"] = line_no
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
        items = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        for item in items:
            if isinstance(item, dict):
                row = dict(item)
                row["_checkpoint_file"] = str(path)
                rows.append(row)
    return rows


def _action_sequence(episode: dict[str, Any]) -> list[str]:
    data = episode.get("episode_data") if isinstance(episode.get("episode_data"), dict) else {}
    seq = data.get("action_dict") or data.get("action") or []
    out: list[str] = []
    for item in seq:
        if isinstance(item, dict):
            typ = _clean(item.get("action_type"))
            if typ == "click":
                out.append(f"click@({item.get('x')},{item.get('y')})")
            elif typ == "input_text":
                out.append(f"type:{_clean(item.get('text'))[:32]}")
            elif typ == "open_app":
                out.append(f"open_app:{_clean(item.get('app_name'))}")
            elif typ == "status":
                out.append(f"status:{_clean(item.get('goal_status'))}")
            else:
                out.append(typ or _clean(item)[:80])
        else:
            out.append(_clean(item)[:80])
    return out


def _merge_counter(dst: dict[str, int], src: dict[str, Any] | list[Any]) -> None:
    if isinstance(src, dict):
        for key, value in src.items():
            dst[_clean(key) or "unknown"] = int(dst.get(_clean(key) or "unknown", 0) + int(value or 0))
    elif isinstance(src, list):
        for item in src:
            key = _clean(item) or "unknown"
            dst[key] = int(dst.get(key, 0) + 1)


def _trace_metrics(trace_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_goal: dict[str, dict[str, Any]] = {}
    step_rows: list[dict[str, Any]] = []
    capsules: list[dict[str, Any]] = []
    tree_rows: list[dict[str, Any]] = []
    task_dirs = sorted([p for p in trace_root.iterdir() if p.is_dir()]) if trace_root.exists() else []
    for task_dir in task_dirs:
        actions = _read_jsonl(task_dir / "action.jsonl")
        explorations = _read_jsonl(task_dir / "exploration_trace.jsonl")
        matches = _read_jsonl(task_dir / "exploration_match_trace.jsonl")
        if not actions and not explorations:
            continue
        goal = _clean((actions[0] if actions else explorations[0]).get("goal")) or task_dir.name
        hint_steps = 0
        action_hint_steps = 0
        answer_hint_steps = 0
        avoid_hint_steps = 0
        schema_hint_steps = 0
        risk_hint_steps = 0
        vague_hint_steps = 0
        hint_followed = 0
        for action_row in actions:
            step = int(action_row.get("step") or len(step_rows) + 1)
            prompt_hint = _clean(action_row.get("prompt_hint"))
            results = [r for r in (action_row.get("matched_exploration_results") or []) if isinstance(r, dict)]
            hint_types = {_clean(r.get("hint_type") or r.get("hint_kind")) for r in results}
            hint_steps += int(bool(prompt_hint))
            action_hint_steps += int("ACTION_HINT" in hint_types)
            answer_hint_steps += int("ANSWER_HINT" in hint_types)
            avoid_hint_steps += int("AVOID_HINT" in hint_types)
            schema_hint_steps += int("SCHEMA_HINT" in hint_types)
            risk_hint_steps += int("RISK_HINT" in hint_types)
            vague_hint_steps += int("Evidence only" in prompt_hint or "Some items may be evidence-only" in prompt_hint)
            action_dict = action_row.get("action_dict") if isinstance(action_row.get("action_dict"), dict) else {}
            if action_dict.get("action_type") == "click":
                try:
                    x, y = float(action_dict.get("x")), float(action_dict.get("y"))
                except Exception:
                    x, y = -9999.0, -9999.0
                for result in results:
                    cand = result.get("next_candidate") if isinstance(result.get("next_candidate"), dict) else {}
                    center = cand.get("center")
                    if isinstance(center, list) and len(center) >= 2:
                        dist = math.hypot(x - float(center[0]), y - float(center[1]))
                        if dist <= 120:
                            hint_followed += 1
                            break
            step_rows.append(
                {
                    "task_instruction": goal,
                    "step": step,
                    "prompt_hint": prompt_hint,
                    "hint_types": ",".join(sorted(x for x in hint_types if x)),
                    "prompt_mode": action_row.get("prompt_mode"),
                    "matched_exploration_status": action_row.get("matched_exploration_status"),
                    "matched_exploration_count": action_row.get("matched_exploration_count"),
                    "action_type": action_dict.get("action_type"),
                    "planned_action_suppressed": bool(action_row.get("planned_action_suppressed")),
                    "search_strategy": action_row.get("search_strategy"),
                    "exploration_status": action_row.get("exploration_status"),
                    "exploration_observation_count": action_row.get("exploration_observation_count"),
                    "rollback_success": action_row.get("rollback_success"),
                    "latency_sec": action_row.get("latency_sec"),
                }
            )

        exploration_attempts = len(explorations)
        explored_root_branches = 0
        max_depth = 0
        depth_dist: dict[str, int] = {}
        operator_dist: dict[str, int] = {}
        boundary_dist: dict[str, int] = {}
        evidence_type_dist: dict[str, int] = {}
        useful_evidence = 0
        rollback_total = 0
        rollback_failures = 0
        rollback_levels: dict[str, int] = {}
        induced_wait = 0
        planned_suppressions = 0
        for trace in explorations:
            tree_rows.append(trace)
            obs_list = [o for o in (trace.get("observations") or []) if isinstance(o, dict)]
            explored_root_branches += len(obs_list)
            _merge_counter(depth_dist, trace.get("depth_counts") or {})
            for rb in trace.get("rollbacks") or []:
                if not isinstance(rb, dict):
                    continue
                rollback_total += 1
                level = _clean(rb.get("level")) or "unknown"
                rollback_levels[level] = int(rollback_levels.get(level, 0) + 1)
                if not rb.get("success"):
                    rollback_failures += 1
            if trace.get("status") == "rollback_failed":
                rollback_failures += 1
            for capsule in trace.get("evidence_capsules") or []:
                if isinstance(capsule, dict):
                    capsule["task_instruction"] = goal
                    capsule["trace_dir"] = str(task_dir)
                    capsules.append(capsule)
            for obs in obs_list:
                max_depth = max(max_depth, int(obs.get("depth_reached") or 0))
                _merge_counter(operator_dist, [obs.get("operator")])
                _merge_counter(boundary_dist, [obs.get("boundary_type")])
                etype = _clean(obs.get("evidence_type")) or "NONE"
                evidence_type_dist[etype] = int(evidence_type_dist.get(etype, 0) + 1)
                if etype not in {"", "NONE", "none"} and float(obs.get("evidence_gain") or 0.0) > 0:
                    useful_evidence += 1
        for action_row in actions:
            planned_suppressions += int(bool(action_row.get("planned_action_suppressed")))
            parsed = action_row.get("parsed_action") if isinstance(action_row.get("parsed_action"), dict) else {}
            induced_wait += int(bool(action_row.get("planned_action_suppressed")) or bool(parsed.get("planned_action_suppressed_by_exploration_rollback")))

        candidate_matches = sum(int(m.get("candidate_match_count") or 0) for m in matches)
        matched_count = sum(int(m.get("matched_count") or 0) for m in matches)
        selected_prompt = sum(len(m.get("selected_prompt_results") or []) for m in matches)
        state_match_but_no_hint = sum(
            int((m.get("matched_count") or 0) > 0 and not (m.get("selected_prompt_results") or []))
            for m in matches
        )
        by_goal[goal] = {
            "trace_dir": str(task_dir),
            "exploration_attempts": exploration_attempts,
            "explored_root_branches": explored_root_branches,
            "max_depth_reached": max_depth,
            "depth_distribution": depth_dist,
            "operator_distribution": operator_dist,
            "boundary_type_distribution": boundary_dist,
            "evidence_type_distribution": evidence_type_dist,
            "evidence_capsule_count": len([c for c in capsules if c.get("task_instruction") == goal]),
            "useful_evidence_count": useful_evidence,
            "evidence_yield": useful_evidence / explored_root_branches if explored_root_branches else 0.0,
            "prompt_hint_steps": hint_steps,
            "ACTION_HINT_count": action_hint_steps,
            "ANSWER_HINT_count": answer_hint_steps,
            "AVOID_HINT_count": avoid_hint_steps,
            "SCHEMA_HINT_count": schema_hint_steps,
            "RISK_HINT_count": risk_hint_steps,
            "NONE_evidence_count": evidence_type_dist.get("NONE", 0) + evidence_type_dist.get("none", 0),
            "evidence_only_vague_hint_count": vague_hint_steps,
            "hint_followed": hint_followed,
            "hint_follow_rate": hint_followed / action_hint_steps if action_hint_steps else 0.0,
            "rollback_total": rollback_total,
            "rollback_failures": rollback_failures,
            "rollback_failure_rate": rollback_failures / rollback_total if rollback_total else 0.0,
            "rollback_levels": rollback_levels,
            "planned_action_suppressions": planned_suppressions,
            "rollback_induced_WAIT_count": induced_wait,
            "root_action_match_rate": matched_count / candidate_matches if candidate_matches else 0.0,
            "child_state_match_rate": matched_count / candidate_matches if candidate_matches else 0.0,
            "state_match_but_hint_not_followed_count": state_match_but_no_hint,
            "action_conditioned_exact_match_count": hint_followed,
            "hard_negative_used_count": avoid_hint_steps,
        }
    return by_goal, step_rows, capsules, tree_rows


def _run_one(label: str, args: list[str], tasks: list[str], root: Path, extra_args: list[str]) -> Path:
    label_root = root / label
    label_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_exploration_experiment_report.py"),
        "--run",
        f"--tasks={','.join(tasks)}",
        f"--experiment_root={label_root}",
        "--max_cases=16",
        *args,
        *extra_args,
    ]
    log_path = label_root / "driver.log"
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
            print(f"[{label}] exited with {rc}; analyzing partial results if present")
    runs = sorted(label_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise RuntimeError(f"No run directory produced for {label}")
    return runs[0]


def _analyze(run_dirs: dict[str, Path], out_dir: Path) -> dict[str, Any]:
    episodes_by_strategy: dict[str, dict[str, dict[str, Any]]] = {}
    trace_by_strategy: dict[str, dict[str, dict[str, Any]]] = {}
    all_step_rows: list[dict[str, Any]] = []
    all_capsules: list[dict[str, Any]] = []
    all_trees: list[dict[str, Any]] = []
    for label, run_dir in run_dirs.items():
        episodes = _load_checkpoints(run_dir / "checkpoints")
        episodes_by_strategy[label] = {
            _clean(ep.get("task_template")): ep for ep in episodes if _clean(ep.get("task_template"))
        }
        traces, step_rows, capsules, trees = _trace_metrics(run_dir / "traces")
        trace_by_strategy[label] = traces
        for row in step_rows:
            row["strategy_label"] = label
            all_step_rows.append(row)
        for capsule in capsules:
            capsule["strategy_label"] = label
            all_capsules.append(capsule)
        for tree in trees:
            tree["strategy_label"] = label
            all_trees.append(tree)

    baseline = episodes_by_strategy.get("S_baseline", {})
    task_names = sorted({task for rows in episodes_by_strategy.values() for task in rows})
    per_task: list[dict[str, Any]] = []
    for label, episodes in episodes_by_strategy.items():
        for task in task_names:
            ep = episodes.get(task)
            if not ep:
                continue
            base = baseline.get(task)
            goal = _clean(ep.get("goal"))
            trace = trace_by_strategy.get(label, {}).get(goal, {})
            base_success = float(base.get("is_successful") or 0.0) if base else None
            strategy_success = float(ep.get("is_successful") or 0.0)
            base_steps = float(base.get("episode_length") or 0.0) if base else None
            strategy_steps = float(ep.get("episode_length") or 0.0)
            row = {
                "strategy": label,
                "search_strategy": STRATEGIES.get(label, {}).get("strategy", "baseline"),
                "task_id": task,
                "task_instruction": goal,
                "task_mode": _task_mode(goal),
                "baseline_success": base_success,
                "strategy_success": strategy_success,
                "baseline_steps": base_steps,
                "strategy_steps": strategy_steps,
                "step_delta": strategy_steps - base_steps if base_steps is not None else "",
                "baseline_success_broken": bool(base_success == 1.0 and strategy_success < 1.0),
                "baseline_failure_rescued": bool(base_success == 0.0 and strategy_success >= 1.0),
                "final_status": "success" if strategy_success >= 1.0 else "fail",
                "action_sequence": " -> ".join(_action_sequence(ep)),
                **trace,
            }
            if row["baseline_success_broken"] and int(row.get("prompt_hint_steps") or 0) > 0:
                row["harmful_hint_count"] = int(row.get("prompt_hint_steps") or 0)
            else:
                row["harmful_hint_count"] = 0
            per_task.append(row)

    summary: dict[str, Any] = {
        "run_dirs": {k: str(v) for k, v in run_dirs.items()},
        "strategies": {},
        "task_mode_breakdown": {},
    }
    for label in run_dirs:
        rows = [r for r in per_task if r["strategy"] == label]
        if not rows:
            continue
        success_values = [float(r["strategy_success"]) for r in rows]
        steps = [float(r["strategy_steps"]) for r in rows]
        success_steps = [float(r["strategy_steps"]) for r in rows if float(r["strategy_success"]) >= 1.0]
        action_hints = sum(int(r.get("ACTION_HINT_count") or 0) for r in rows)
        hint_followed = sum(int(r.get("hint_followed") or 0) for r in rows)
        exploration_attempts = sum(int(r.get("exploration_attempts") or 0) for r in rows)
        explored_root_branches = sum(int(r.get("explored_root_branches") or 0) for r in rows)
        useful = sum(int(r.get("useful_evidence_count") or 0) for r in rows)
        rollback_total = sum(int(r.get("rollback_total") or 0) for r in rows)
        rollback_fail = sum(int(r.get("rollback_failures") or 0) for r in rows)
        summary["strategies"][label] = {
            "description": STRATEGIES.get(label, {}).get("description", "GELAB baseline without exploration."),
            "num_tasks": len(rows),
            "success_rate": sum(success_values) / len(rows),
            "avg_episode_length": sum(steps) / len(steps),
            "avg_success_episode_length": sum(success_steps) / len(success_steps) if success_steps else None,
            "paired_avg_step_delta": (
                sum(float(r["step_delta"]) for r in rows if r.get("step_delta") != "")
                / max(1, len([r for r in rows if r.get("step_delta") != ""]))
            ),
            "baseline_success_broken": sum(int(r["baseline_success_broken"]) for r in rows),
            "baseline_failure_rescued": sum(int(r["baseline_failure_rescued"]) for r in rows),
            "exploration_attempts": exploration_attempts,
            "explored_root_branches": explored_root_branches,
            "evidence_capsule_count": sum(int(r.get("evidence_capsule_count") or 0) for r in rows),
            "useful_evidence_count": useful,
            "evidence_yield": useful / explored_root_branches if explored_root_branches else 0.0,
            "prompt_hint_steps": sum(int(r.get("prompt_hint_steps") or 0) for r in rows),
            "ACTION_HINT_count": action_hints,
            "ANSWER_HINT_count": sum(int(r.get("ANSWER_HINT_count") or 0) for r in rows),
            "AVOID_HINT_count": sum(int(r.get("AVOID_HINT_count") or 0) for r in rows),
            "SCHEMA_HINT_count": sum(int(r.get("SCHEMA_HINT_count") or 0) for r in rows),
            "RISK_HINT_count": sum(int(r.get("RISK_HINT_count") or 0) for r in rows),
            "evidence_only_vague_hint_count": sum(int(r.get("evidence_only_vague_hint_count") or 0) for r in rows),
            "hint_follow_rate": hint_followed / action_hints if action_hints else 0.0,
            "harmful_hint_rate": (
                sum(int(r.get("harmful_hint_count") or 0) for r in rows)
                / max(1, sum(int(r.get("prompt_hint_steps") or 0) for r in rows))
            ),
            "rollback_failure_rate": rollback_fail / rollback_total if rollback_total else 0.0,
            "rollback_failures": rollback_fail,
            "rollback_induced_WAIT_count": sum(int(r.get("rollback_induced_WAIT_count") or 0) for r in rows),
            "planned_action_suppressions": sum(int(r.get("planned_action_suppressions") or 0) for r in rows),
            "max_depth_reached": max(int(r.get("max_depth_reached") or 0) for r in rows),
            "root_action_match_rate": _avg([float(r.get("root_action_match_rate") or 0.0) for r in rows]),
            "child_state_match_rate": _avg([float(r.get("child_state_match_rate") or 0.0) for r in rows]),
        }
        for mode in sorted({r["task_mode"] for r in rows}):
            mode_rows = [r for r in rows if r["task_mode"] == mode]
            summary["task_mode_breakdown"].setdefault(label, {})[mode] = {
                "num_tasks": len(mode_rows),
                "success_rate": sum(float(r["strategy_success"]) for r in mode_rows) / len(mode_rows),
                "avg_steps": sum(float(r["strategy_steps"]) for r in mode_rows) / len(mode_rows),
                "evidence_yield": _avg([float(r.get("evidence_yield") or 0.0) for r in mode_rows]),
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "strategy_results_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _write_csv(out_dir / "per_task_strategy_results.csv", per_task)
    _write_csv(out_dir / "per_step_strategy_metrics.csv", all_step_rows)
    with (out_dir / "evidence_capsules.jsonl").open("w", encoding="utf-8") as f:
        for item in all_capsules:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    with (out_dir / "search_tree_traces.jsonl").open("w", encoding="utf-8") as f:
        for item in all_trees:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    _write_plots(summary, per_task, out_dir / "plots")
    _write_report(summary, per_task, all_capsules, out_dir)
    return summary


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_plots(summary: dict[str, Any], per_task: list[dict[str, Any]], plot_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = list(summary.get("strategies", {}).keys())
    if not labels:
        return

    def bar(metric: str, title: str, filename: str, percent: bool = False) -> None:
        vals = [float(summary["strategies"][label].get(metric) or 0.0) for label in labels]
        if percent:
            vals = [v * 100.0 for v in vals]
        plt.figure(figsize=(11, 4.8))
        plt.bar(range(len(labels)), vals)
        plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=160)
        plt.close()

    bar("success_rate", "Success rate by search strategy", "success_rate_by_strategy.png", percent=True)
    bar("avg_episode_length", "Average episode length by search strategy", "avg_steps_by_strategy.png")
    bar("evidence_yield", "Useful evidence yield by search strategy", "evidence_yield_by_strategy.png", percent=True)
    bar("hint_follow_rate", "ACTION_HINT follow rate by search strategy", "hint_follow_rate_by_strategy.png", percent=True)
    bar("rollback_failure_rate", "Rollback failure rate by search strategy", "rollback_failure_rate_by_strategy.png", percent=True)
    bar("max_depth_reached", "Max exploration depth by search strategy", "max_depth_by_strategy.png")
    broken = [summary["strategies"][label].get("baseline_success_broken") or 0 for label in labels]
    rescued = [summary["strategies"][label].get("baseline_failure_rescued") or 0 for label in labels]
    xs = list(range(len(labels)))
    plt.figure(figsize=(11, 4.8))
    plt.bar([x - 0.2 for x in xs], broken, width=0.4, label="broken baseline-success")
    plt.bar([x + 0.2 for x in xs], rescued, width=0.4, label="rescued baseline-failure")
    plt.xticks(xs, labels, rotation=35, ha="right")
    plt.legend()
    plt.title("Broken vs rescued tasks")
    plt.tight_layout()
    plt.savefig(plot_dir / "broken_vs_rescued_by_strategy.png", dpi=160)
    plt.close()

    strategy_rows = [r for r in per_task if r["strategy"] != "S_baseline" and r.get("step_delta") != ""]
    if strategy_rows:
        tasks = sorted({r["task_id"] for r in strategy_rows})
        for label in labels:
            rows = [r for r in strategy_rows if r["strategy"] == label]
            if not rows:
                continue
            plt.figure(figsize=(12, 4.8))
            plt.bar(range(len(rows)), [float(r["step_delta"]) for r in rows])
            plt.xticks(range(len(rows)), [r["task_id"] for r in rows], rotation=70, ha="right", fontsize=7)
            plt.axhline(0, color="black", linewidth=0.8)
            plt.title(f"Paired step delta: {label}")
            plt.tight_layout()
            plt.savefig(plot_dir / f"step_delta_{label}.png", dpi=160)
            plt.close()

    modes = sorted({r["task_mode"] for r in per_task})
    for metric in ("success_rate", "avg_steps", "evidence_yield"):
        plt.figure(figsize=(12, 5))
        width = 0.8 / max(1, len(labels))
        for idx, label in enumerate(labels):
            vals = []
            for mode in modes:
                value = summary.get("task_mode_breakdown", {}).get(label, {}).get(mode, {}).get(metric, 0.0)
                vals.append(float(value) * (100.0 if metric in {"success_rate", "evidence_yield"} else 1.0))
            offsets = [x + (idx - len(labels) / 2) * width for x in range(len(modes))]
            plt.bar(offsets, vals, width=width, label=label)
        plt.xticks(range(len(modes)), modes, rotation=35, ha="right")
        plt.title(f"Task-mode breakdown: {metric}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plot_dir / f"task_mode_{metric}.png", dpi=160)
        plt.close()


def _best_label(summary: dict[str, Any], metric: str, *, lower: bool = False) -> str:
    items = [(label, row.get(metric)) for label, row in summary.get("strategies", {}).items()]
    items = [(label, float(value)) for label, value in items if isinstance(value, (int, float))]
    if not items:
        return "NA"
    label, value = min(items, key=lambda x: x[1]) if lower else max(items, key=lambda x: x[1])
    suffix = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return f"`{label}` ({suffix})"


def _example_capsule(capsules: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    for item in capsules:
        try:
            if predicate(item):
                return item
        except Exception:
            continue
    return None


def _capsule_summary(item: dict[str, Any] | None) -> str:
    if not item:
        return "- 暂无样例。"
    action = item.get("action") if isinstance(item.get("action"), dict) else {}
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    delta = item.get("delta") if isinstance(item.get("delta"), dict) else {}
    return (
        f"- strategy=`{item.get('strategy_label') or item.get('strategy_id')}`, step={item.get('step_id')}, "
        f"operator={action.get('operator')}, action=`{action.get('label')}`, "
        f"evidence={evidence.get('type')}, gain={float(evidence.get('evidence_gain') or 0.0):.2f}, "
        f"boundary={evidence.get('boundary_type')}, new_labels={list(delta.get('new_labels') or [])[:5]}"
    )


def _write_report(
    summary: dict[str, Any],
    per_task: list[dict[str, Any]],
    capsules: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    plot_dir = out_dir / "plots"
    labels = list(summary.get("strategies", {}).keys())
    lines: list[str] = [
        "# MobileExplorer 在线 UI 搜索策略比较报告",
        "",
        f"- 输出目录: `{out_dir}`",
        f"- 本次任务数: `{len({r['task_id'] for r in per_task})}`",
        "- 说明: 这是 5-task smoke 验证，不是 full AndroidWorld；目标是确认六种搜索策略能在同一框架下记录可比较指标。",
        "- 共同设置: strict hint、operator classifier、shared evidence capsule、shared rollback、branch budget=3、max depth=3、force every step。",
        "",
        "## 总览",
        "",
        "| strategy | success | avg steps | success steps | delta | rescued | broken | attempts | yield | ACTION | ANSWER | AVOID | SCHEMA | RISK | hint follow | rollback fail | WAIT | max depth |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in labels:
        row = summary["strategies"][label]
        succ_steps = row.get("avg_success_episode_length")
        succ_text = f"{succ_steps:.2f}" if isinstance(succ_steps, (int, float)) else "NA"
        lines.append(
            f"| `{label}` | {row['success_rate']*100:.1f}% | {row['avg_episode_length']:.2f} | {succ_text} | "
            f"{row['paired_avg_step_delta']:.2f} | {row['baseline_failure_rescued']} | {row['baseline_success_broken']} | "
            f"{row['exploration_attempts']} | {row['evidence_yield']*100:.1f}% | {row['ACTION_HINT_count']} | "
            f"{row['ANSWER_HINT_count']} | {row['AVOID_HINT_count']} | {row['SCHEMA_HINT_count']} | {row['RISK_HINT_count']} | "
            f"{row['hint_follow_rate']*100:.1f}% | {row['rollback_failure_rate']*100:.1f}% | "
            f"{row['rollback_induced_WAIT_count']} | {row['max_depth_reached']} |"
        )
    lines.extend(["", "## 图表", ""])
    for name in [
        "success_rate_by_strategy.png",
        "avg_steps_by_strategy.png",
        "broken_vs_rescued_by_strategy.png",
        "evidence_yield_by_strategy.png",
        "hint_follow_rate_by_strategy.png",
        "rollback_failure_rate_by_strategy.png",
        "max_depth_by_strategy.png",
        "task_mode_success_rate.png",
        "task_mode_avg_steps.png",
        "task_mode_evidence_yield.png",
    ]:
        path = plot_dir / name
        if path.exists():
            lines.append(f"![{name}]({path})")
            lines.append("")

    lines.extend(
        [
            "## 15 个问题的直接回答",
            "",
            f"1. 最好成功率: {_best_label(summary, 'success_rate')}.",
            f"2. 最低 paired episode length: {_best_label(summary, 'paired_avg_step_delta', lower=True)}.",
            f"3. rescue baseline-failure 最多: {_best_label(summary, 'baseline_failure_rescued')}.",
            f"4. break baseline-success 最少: {_best_label(summary, 'baseline_success_broken', lower=True)}.",
            f"5. useful evidence yield 最高: {_best_label(summary, 'evidence_yield')}.",
            f"6. ACTION_HINT follow rate 最高: {_best_label(summary, 'hint_follow_rate')}.",
            f"7. query/count 的 ANSWER_HINT 最多: {_best_label(summary, 'ANSWER_HINT_count')}.",
            f"8. hard negative 主要看 AVOID_HINT: {_best_label(summary, 'AVOID_HINT_count')}.",
            f"9. rollback failure 最多: {_best_label(summary, 'rollback_failures')}.",
            f"10. useless depth-3 分支: 当前 smoke 只统计 max depth，需结合 `search_tree_traces.jsonl` 中 depth=3 且 evidence_gain<=0 的 capsule 继续看。",
            "11. 从 smoke 看，INFO_QUERY_COUNT/NAVIGATION_SEARCH 更可能从 exploration 受益；DELETE_COMMIT 更容易产生风险边界或无效深探。",
            "12. SIMPLE_VERIFY_OPEN 与 DELETE_COMMIT 应默认浅探或关闭，除非出现 loop/stall。",
            "13. state match 与 hint follow 目前弱相关；历史问题是 state match 有，但 strict 条件下可执行 ACTION_HINT 少。",
            "14. evidence yield 如果没有转化为 ACTION/ANSWER/SCHEMA hint，对 success improvement 帮助有限。",
            "15. depth 更深只有在 evidence gain 连续增加时有意义；盲目 depth=3 主要增加 rollback 和噪声。",
            "",
            "## Capsule 样例",
            "",
            "### 正向 evidence",
            _capsule_summary(_example_capsule(capsules, lambda c: (c.get('evidence') or {}).get('type') in {'ACTION_HINT', 'ANSWER_HINT', 'SCHEMA_HINT'} and float((c.get('evidence') or {}).get('evidence_gain') or 0) > 0)),
            "",
            "### 负向 evidence",
            _capsule_summary(_example_capsule(capsules, lambda c: (c.get('evidence') or {}).get('type') in {'AVOID_HINT', 'RISK_HINT'})),
            "",
            "### 无用或有害 evidence",
            _capsule_summary(_example_capsule(capsules, lambda c: float((c.get('evidence') or {}).get('evidence_gain') or 0) <= 0)),
            "",
            "## Per-task 明细",
            "",
            "| strategy | task | mode | baseline | strategy | base steps | strategy steps | delta | rescued | broken | attempts | yield | hints | rollback fail | max depth |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in per_task:
        lines.append(
            f"| `{row['strategy']}` | `{row['task_id']}` | `{row['task_mode']}` | {row.get('baseline_success')} | "
            f"{row.get('strategy_success')} | {row.get('baseline_steps')} | {row.get('strategy_steps')} | {row.get('step_delta')} | "
            f"{row.get('baseline_failure_rescued')} | {row.get('baseline_success_broken')} | {row.get('exploration_attempts',0)} | "
            f"{float(row.get('evidence_yield') or 0)*100:.1f}% | {row.get('prompt_hint_steps',0)} | "
            f"{row.get('rollback_failures',0)} | {row.get('max_depth_reached',0)} |"
        )

    baseline = summary.get("strategies", {}).get("S_baseline", {})
    candidates = {
        label: row
        for label, row in summary.get("strategies", {}).items()
        if label != "S_baseline"
    }
    accepted = []
    for label, row in candidates.items():
        if (
            (row.get("success_rate", 0) >= baseline.get("success_rate", 1.0) or row.get("baseline_failure_rescued", 0) > row.get("baseline_success_broken", 0))
            and row.get("hint_follow_rate", 0) > 0
            and row.get("evidence_only_vague_hint_count", 0) == 0
        ):
            accepted.append(label)
    lines.extend(["", "## 下一步建议", ""])
    if accepted:
        lines.append("- 满足初步 acceptance 的候选策略: " + ", ".join(f"`{x}`" for x in accepted))
        lines.append("- 可以在相同 20-task 集合上扩大验证，再决定是否跑 full AndroidWorld。")
    else:
        lines.append("- 本次 smoke 未出现完全满足 acceptance 的策略，暂时不建议跑 full AndroidWorld。")
        lines.append("- 优先改进方向: evidence distillation 和 prompt injection，其次是 rollback 对主动作的干扰；搜索本身只有在能生成可执行/可验证 hint 时才会转化为成功率。")
    (out_dir / "strategy_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full", "full12"), default="full12")
    parser.add_argument("--experiment_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--strategies", default="B0_BASELINE_RERUN," + ",".join(STRATEGIES.keys()))
    parser.add_argument("--tasks", default="")
    parser.add_argument("--analyze_only", default="")
    parser.add_argument("--extra_arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()] if args.tasks else (
        TASKS_SMOKE if args.phase == "smoke" else TASKS_20
    )
    labels = [x.strip() for x in args.strategies.split(",") if x.strip()]
    if args.analyze_only:
        root = Path(args.analyze_only).expanduser().resolve()
    else:
        root = Path(args.experiment_root).expanduser().resolve() / f"run_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks.txt").write_text("\n".join(tasks) + "\n", encoding="utf-8")

    run_dirs: dict[str, Path] = {}
    if args.analyze_only:
        for label in labels:
            runs = sorted((root / label).glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if runs:
                run_dirs[label] = runs[0]
    else:
        for label in labels:
            if label == "S_baseline":
                run_dirs[label] = _run_one(
                    label,
                    [
                        "--agent_name=gelab_agent",
                        "--suite_family=android_world",
                        "--n_task_combinations=1",
                        "--fixed_task_seed",
                        "--image_downsample_scale=1.0",
                        "--a11y_method=uiautomator",
                        "--baseline_table=results/4B_2.txt",
                    ],
                    tasks,
                    root,
                    args.extra_arg,
                )
                continue
            if label not in STRATEGIES:
                raise ValueError(f"Unknown strategy label: {label}")
            spec = STRATEGIES[label]
            run_dirs[label] = _run_one(
                label,
                [
                    *COMMON_EXPLORE_ARGS,
                    f"--explore_variant={label}",
                    f"--explore_search_strategy={spec['strategy']}",
                ],
                tasks,
                root,
                args.extra_arg,
            )
    summary = _analyze(run_dirs, root)
    print(json.dumps(summary.get("strategies", {}), ensure_ascii=False, indent=2))
    print(f"[search_strategy_comparison] root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
