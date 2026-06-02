#!/usr/bin/env python3
"""Run frozen 30-task AndroidWorld strategy comparison by variant shard.

Example:
  python scripts/run_strategy_30task_split.py \
    --run_group_id sensys_strategy_30task_20260602 \
    --machine_id machine1 \
    --variants B0_BASELINE_RERUN,S1_BFS_BUDGET12,S3_BEAM_BUDGET12,S5_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12

Run the same script on another computer with the same run_group_id and a
different --variants list. Each variant runs the full frozen 30-task set.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = REPO_ROOT / "scripts" / "run_exploration_experiment_report.py"
PYTHON_BIN = sys.executable

FROZEN_30_TASKS: list[dict[str, str]] = [
    {"task_id": "1", "task": "NotesRecipeIngredientCount", "app": "Joplin", "task_mode": "answer_entity", "shard": "A"},
    {"task_id": "2", "task": "NotesIsTodo", "app": "Joplin", "task_mode": "answer_boolean", "shard": "A"},
    {"task_id": "3", "task": "FilesDeleteFile", "app": "Files", "task_mode": "delete_boundary", "shard": "A"},
    {"task_id": "4", "task": "FilesMoveFile", "app": "Files", "task_mode": "navigation_mutation", "shard": "A"},
    {"task_id": "5", "task": "SportsTrackerActivityDuration", "app": "OpenTracks", "task_mode": "answer_stats", "shard": "A"},
    {"task_id": "6", "task": "SportsTrackerTotalDistanceForCategoryOverInterval", "app": "OpenTracks", "task_mode": "answer_stats", "shard": "A"},
    {"task_id": "7", "task": "SimpleCalendarNextMeetingWithPerson", "app": "Calendar", "task_mode": "answer_entity", "shard": "A"},
    {"task_id": "8", "task": "SimpleCalendarEventsInTimeRange", "app": "Calendar", "task_mode": "answer_entity", "shard": "A"},
    {"task_id": "9", "task": "ClockStopWatchRunning", "app": "Clock", "task_mode": "state_verify", "shard": "A"},
    {"task_id": "10", "task": "ClockTimerEntry", "app": "Clock", "task_mode": "state_verify", "shard": "A"},
    {"task_id": "11", "task": "MarkorCreateNote", "app": "Markor", "task_mode": "create_note", "shard": "A"},
    {"task_id": "12", "task": "MarkorEditNote", "app": "Markor", "task_mode": "edit_note", "shard": "A"},
    {"task_id": "13", "task": "MarkorCreateFolder", "app": "Markor", "task_mode": "create_folder", "shard": "A"},
    {"task_id": "14", "task": "ExpenseDeleteSingle", "app": "Expense", "task_mode": "delete_boundary", "shard": "A"},
    {"task_id": "15", "task": "ExpenseDeleteMultiple2", "app": "Expense", "task_mode": "delete_boundary", "shard": "A"},
    {"task_id": "16", "task": "ExpenseAddSingle", "app": "Expense", "task_mode": "create_record", "shard": "B"},
    {"task_id": "17", "task": "BrowserMaze", "app": "Browser", "task_mode": "web_interaction", "shard": "B"},
    {"task_id": "18", "task": "BrowserMultiply", "app": "Browser", "task_mode": "web_interaction", "shard": "B"},
    {"task_id": "19", "task": "BrowserDraw", "app": "Browser", "task_mode": "web_interaction", "shard": "B"},
    {"task_id": "20", "task": "ContactsNewContactDraft", "app": "Contacts", "task_mode": "draft_data_entry", "shard": "B"},
    {"task_id": "21", "task": "ContactsAddContact", "app": "Contacts", "task_mode": "data_entry", "shard": "B"},
    {"task_id": "22", "task": "AudioRecorderRecordAudio", "app": "AudioRecorder", "task_mode": "record_audio", "shard": "B"},
    {"task_id": "23", "task": "AudioRecorderRecordAudioWithFileName", "app": "AudioRecorder", "task_mode": "record_audio", "shard": "B"},
    {"task_id": "24", "task": "SystemWifiTurnOnVerify", "app": "System", "task_mode": "state_verify", "shard": "B"},
    {"task_id": "25", "task": "SystemWifiTurnOffVerify", "app": "System", "task_mode": "state_verify", "shard": "B"},
    {"task_id": "26", "task": "SystemBrightnessMinVerify", "app": "System", "task_mode": "state_verify", "shard": "B"},
    {"task_id": "27", "task": "SimpleDrawProCreateDrawing", "app": "SimpleDraw", "task_mode": "create_drawing", "shard": "B"},
    {"task_id": "28", "task": "CameraTakePhoto", "app": "Camera", "task_mode": "capture_photo", "shard": "B"},
    {"task_id": "29", "task": "OsmAndFavorite", "app": "OsmAnd", "task_mode": "map_marker", "shard": "B"},
    {"task_id": "30", "task": "OsmAndMarker", "app": "OsmAnd", "task_mode": "map_marker", "shard": "B"},
]

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "B0_BASELINE_RERUN": {"enabled": False},
    "S1_BFS_BUDGET12": {"enabled": True, "explore_strategy": "bfs", "search_strategy": "stratified_bfs", "safe_mcts": False},
    "S2_DFS_BUDGET12": {"enabled": True, "explore_strategy": "dfs", "search_strategy": "iddfs", "safe_mcts": False},
    "S3_BEAM_BUDGET12": {"enabled": True, "explore_strategy": "dfs", "search_strategy": "beam", "safe_mcts": False},
    "S4_MCTS_BUDGET12": {"enabled": True, "explore_strategy": "dfs", "search_strategy": "mcts", "safe_mcts": True},
    "S5_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12": {"enabled": True, "explore_strategy": "dfs", "search_strategy": "best_first", "safe_mcts": False, "pattern_aware": True},
}

REQUIRED_FILES = [
    "runtime_config.yaml",
    "frozen_task_selection.md",
    "task_shard.csv",
    "per_task_results.csv",
    "per_step_metrics.csv",
    "candidate_scores.jsonl",
    "candidate_filter_stats.jsonl",
    "exploration_step_summary.jsonl",
    "exploration_branch_trace.jsonl",
    "exploration_page_trace.jsonl",
    "exploration_latency.jsonl",
    "state_alignment.jsonl",
    "evidence_decisions.jsonl",
    "prompt_hints.jsonl",
    "prompt_traces.jsonl",
    "hint_hit_follow.jsonl",
    "rollback_events.jsonl",
    "rollback_gate_decisions.jsonl",
    "rollback_level2_cases.md",
    "rollback_failure_cases.md",
    "shortcut_plans.jsonl",
    "shortcut_shadow_eval.jsonl",
    "state_acquisition_metrics.csv",
    "step_decoupling_status.jsonl",
]

SCREENSHOT_DIRS = [
    "screenshots/rollback_failures",
    "screenshots/depth1_branches",
    "screenshots/depth2_branches",
    "screenshots/injected_hints",
]


def _task_rows_for_shard(shard: str, max_cases: int | None = None) -> list[dict[str, str]]:
    if shard.upper() in {"ALL", "FULL", "30", "FULL30"}:
        rows = [dict(row) for row in FROZEN_30_TASKS]
    else:
        rows = [dict(row) for row in FROZEN_30_TASKS if row["shard"] == shard]
    if max_cases is not None:
        rows = rows[:max_cases]
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task_id", "task", "app", "task_mode", "shard"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_frozen_selection(root: Path, shard_rows: list[dict[str, str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root / "frozen_30task_selection.csv", FROZEN_30_TASKS)
    _write_csv(root / "machine_A_task_shard.csv", _task_rows_for_shard("A"))
    _write_csv(root / "machine_B_task_shard.csv", _task_rows_for_shard("B"))
    _write_csv(root / "task_shard.csv", shard_rows)
    lines = [
        "# Frozen 30-task strategy selection",
        "",
        "- benchmark: AndroidWorld",
        "- split: variant-sharded; every assigned variant runs the same frozen 30 tasks",
        "- variants: B0/S1/S2/S3/S4/S5 budget12",
        "- depth: 2",
        "- per-step exploration budget: 12 safe candidates",
        "- active t+2 shortcut: disabled; shadow traces only",
        "- note: the `shard` column below is only a task bucket label, not the machine split",
        "",
        "| task_id | task | app | task_mode | shard |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in FROZEN_30_TASKS:
        lines.append(f"| {row['task_id']} | `{row['task']}` | {row['app']} | {row['task_mode']} | {row['shard']} |")
    (root / "frozen_30task_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latest_run(variant_root: Path) -> Path | None:
    runs = sorted(variant_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _placeholder(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".md":
        path.write_text("# No records\n\nThis artifact was not emitted for this variant.\n", encoding="utf-8")
    elif path.suffix == ".csv":
        path.write_text("\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")


def _materialize_variant_outputs(variant_root: Path, run_dir: Path | None, shard_rows: list[dict[str, str]]) -> None:
    variant_root.mkdir(parents=True, exist_ok=True)
    if run_dir:
        for filename in REQUIRED_FILES:
            src = run_dir / filename
            dst = variant_root / filename
            if src.exists() and src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        for rel_dir in SCREENSHOT_DIRS:
            src_dir = run_dir / rel_dir
            dst_dir = variant_root / rel_dir
            if src_dir.exists() and src_dir.is_dir():
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.copytree(src_dir, dst_dir)
    _write_csv(variant_root / "task_shard.csv", shard_rows)
    for filename in REQUIRED_FILES:
        _placeholder(variant_root / filename)
    for rel_dir in SCREENSHOT_DIRS:
        (variant_root / rel_dir).mkdir(parents=True, exist_ok=True)


def _common_args(variant_name: str, tasks: list[str], variant_root: Path, max_cases: int) -> list[str]:
    return [
        PYTHON_BIN,
        str(REPORT_SCRIPT),
        "--run",
        "--suite_family=android_world",
        "--tasks=" + ",".join(tasks),
        "--n_task_combinations=1",
        "--task_random_seed=43",
        "--fixed_task_seed",
        "--image_downsample_scale=1.0",
        "--baseline_table=results/4B_2.txt",
        "--experiment_root=" + str(variant_root),
        f"--max_cases={max_cases}",
        "--a11y_method=fast_provider",
        "--a11y_preflight_timeout=90",
        "--latency_profile",
        "--agent_name=explore_agent_gelab",
        "--explore_variant=" + variant_name,
        "--decoupled_exploration",
        "--exploration_timing=parallel_shadow",
        "--no-explore_use_current_action",
        "--no-explore_use_planning_text",
        "--no-explore_planned_only",
        "--explore_decouple_planned",
        "--explore_parallel_vlm",
        "--explore_parallel_lookahead",
        "--t2_mode=shadow",
        "--no-t2_allow_safe_search_input",
        "--no-explore_enable_t2_lookahead",
    ]


def _variant_args(variant_name: str, tasks: list[str], variant_root: Path, max_cases: int) -> list[str]:
    cfg = VARIANT_CONFIGS[variant_name]
    cmd = _common_args(variant_name, tasks, variant_root, max_cases)
    if not cfg.get("enabled"):
        cmd.extend([
            "--no-explore_enable",
            "--explore_max_runs=0",
            "--explore_max_step=0",
            "--explore_branch_budget=0",
            "--explore_min_attempts_per_step=0",
            "--explore_branch_depth=0",
        ])
        return cmd
    cmd.extend([
        "--explore_enable",
        "--explore_max_runs=10000",
        "--explore_max_step=10000",
        "--explore_branch_budget=12",
        "--explore_min_attempts_per_step=12",
        "--explore_branch_depth=2",
        "--explore_back_limit=4",
        "--explore_fallback_safe_candidates",
        "--explore_safe_click_only",
        "--explore_skip_launcher",
        "--explore_filter_launcher_relevance",
        "--no-explore_skip_destructive_goals",
        "--explore_fast_mode",
        "--explore_fast_state",
        "--explore_transaction_safe",
        "--explore_action_settle_s=0.25",
        "--explore_hint_policy=strict",
        "--explore_search_policy=task_gate",
        f"--explore_strategy={cfg['explore_strategy']}",
        f"--explore_search_strategy={cfg['search_strategy']}",
        "--explore_rollback_policy=improved",
        "--explore_fixed_framework",
        "--explore_answer_extractors",
        "--explore_slot_complete",
        "--explore_slot_policy_switcher",
        "--explore_lightweight_a11y_trace",
        "--explore_trace_a11y_limit=120",
        "--explore_upper_bound_evidence",
        "--trace_screenshot_mode=failure+level2+depth2+injected+sampled",
    ])
    if cfg.get("safe_mcts"):
        cmd.append("--explore_safe_mcts")
    return cmd


def _variant_env(variant_name: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["ANDROID_WORLD_T2_ACTIVE"] = "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_UNSAFE_SPECULATIVE_EXECUTION"] = "0"
    if VARIANT_CONFIGS[variant_name].get("pattern_aware"):
        env["ANDROID_WORLD_LIGHT_EXPLORE_PATTERN_AWARE_OPERATOR_BEST_FIRST"] = "1"
    return env


def _run_variant(run_group_root: Path, variant_name: str, shard_rows: list[dict[str, str]], max_cases: int, dry_run: bool) -> int:
    machine_id = run_group_root.name
    variant_root = run_group_root / variant_name
    variant_root.mkdir(parents=True, exist_ok=True)
    tasks = [row["task"] for row in shard_rows]
    cmd = _variant_args(variant_name, tasks, variant_root, max_cases)
    manifest = {
        "machine_id": machine_id,
        "task_scope": "full_30task" if len(shard_rows) == len(FROZEN_30_TASKS) else "task_subset",
        "task_shards": sorted({row["shard"] for row in shard_rows}),
        "variant": variant_name,
        "budget": 12 if VARIANT_CONFIGS[variant_name].get("enabled") else 0,
        "depth": 2 if VARIANT_CONFIGS[variant_name].get("enabled") else 0,
        "tasks": tasks,
        "command": cmd,
        "dry_run": dry_run,
    }
    (variant_root / "variant_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(variant_root / "task_shard.csv", shard_rows)
    if dry_run:
        (variant_root / "driver.log").write_text("DRY RUN\n" + " ".join(cmd) + "\n", encoding="utf-8")
        _materialize_variant_outputs(variant_root, None, shard_rows)
        return 0
    log_path = variant_root / "driver.log"
    print(f"[strategy30] machine={machine_id} variant={variant_name} tasks={len(tasks)}")
    print("[strategy30] cmd=" + " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=_variant_env(variant_name),
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    run_dir = _latest_run(variant_root)
    manifest["returncode"] = proc.returncode
    manifest["run_dir"] = str(run_dir) if run_dir else ""
    manifest["log"] = str(log_path)
    (variant_root / "variant_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _materialize_variant_outputs(variant_root, run_dir, shard_rows)
    print(f"[strategy30] machine={machine_id} variant={variant_name} returncode={proc.returncode} run_dir={run_dir}")
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group_id", default="strategy30_" + dt.datetime.now().strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--machine_id", required=True)
    parser.add_argument("--task_shard_id", default="ALL", choices=["ALL", "A", "B"])
    parser.add_argument("--experiment_root", default=str(REPO_ROOT / "results" / "strategy_30task_split"))
    parser.add_argument("--variants", default=",".join(VARIANT_CONFIGS.keys()))
    parser.add_argument("--max_cases", type=int, default=30)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    selected_variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [item for item in selected_variants if item not in VARIANT_CONFIGS]
    if unknown:
        raise SystemExit("Unknown variants: " + ", ".join(unknown))

    shard_rows = _task_rows_for_shard(args.task_shard_id, args.max_cases)
    if not shard_rows:
        raise SystemExit(f"No tasks for shard {args.task_shard_id}")

    root = Path(args.experiment_root).resolve() / args.run_group_id
    machine_dir_name = args.machine_id if args.machine_id.startswith("machine_") else f"machine_{args.machine_id}"
    machine_root = root / machine_dir_name
    machine_root.mkdir(parents=True, exist_ok=True)
    _write_frozen_selection(root, shard_rows)
    _write_frozen_selection(machine_root, shard_rows)
    (machine_root / "machine_id.txt").write_text(args.machine_id + "\n", encoding="utf-8")
    (machine_root / "run_group_id.txt").write_text(args.run_group_id + "\n", encoding="utf-8")

    manifest = {
        "run_group_id": args.run_group_id,
        "machine_id": args.machine_id,
        "task_shard_id": args.task_shard_id,
        "variant_shard": selected_variants,
        "experiment_root": str(root),
        "variants": selected_variants,
        "max_cases": args.max_cases,
        "tasks": [row["task"] for row in shard_rows],
        "dry_run": args.dry_run,
    }
    (machine_root / "machine_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    failures: list[str] = []
    for variant_name in selected_variants:
        rc = _run_variant(machine_root, variant_name, shard_rows, len(shard_rows), args.dry_run)
        if rc != 0:
            failures.append(f"{variant_name}:{rc}")
    if failures:
        print("[strategy30] failures=" + ",".join(failures))
        return 1
    print(f"[strategy30] done root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
