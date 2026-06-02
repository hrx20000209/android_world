#!/usr/bin/env python3
"""Run the frozen conservative 40-task MobileExplorer diagnostic."""

from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = REPO_ROOT / "scripts" / "run_exploration_experiment_report.py"
PYTHON_BIN = "python3"
BASELINE_TABLE = REPO_ROOT / "results" / "4B_2.txt"
FROZEN_CONFIG = {
    "ANDROID_WORLD_LIGHT_EXPLORE_ENABLE": "1",
    "ANDROID_WORLD_LIGHT_EXPLORE_FIXED_FRAMEWORK": "1",
    "ANDROID_WORLD_LIGHT_EXPLORE_SLOT_COMPLETE": "1",
    "ANDROID_WORLD_LIGHT_EXPLORE_SEARCH_POLICY": "task_gate",
    "ANDROID_WORLD_LIGHT_EXPLORE_SEARCH_STRATEGY": "best_first",
    "ANDROID_WORLD_LIGHT_EXPLORE_HINT_POLICY": "strict",
    "ANDROID_WORLD_LIGHT_EXPLORE_ROLLBACK_POLICY": "improved",
    "ANDROID_WORLD_LIGHT_EXPLORE_PLANNED_ONLY": "1",
    "ANDROID_WORLD_LIGHT_EXPLORE_FORCE_EVERY_STEP": "0",
    "ANDROID_WORLD_LIGHT_EXPLORE_ENABLE_T2_LOOKAHEAD": "1",
    "ANDROID_WORLD_T2_MODE": "off",
}


TASKS = [
    # OpenTracks / SportsTracker (5)
    {
        "task_id": "SportsTrackerActivityDuration",
        "app": "OpenTracks",
        "task_family": "OpenTracks",
        "task_mode": "ACTIVITY_STATS",
        "task_subtype": "activity duration",
    },
    {
        "task_id": "SportsTrackerTotalDistanceForCategoryOverInterval",
        "app": "OpenTracks",
        "task_family": "OpenTracks",
        "task_mode": "ACTIVITY_STATS",
        "task_subtype": "total distance",
    },
    {
        "task_id": "SportsTrackerActivitiesOnDate",
        "app": "OpenTracks",
        "task_family": "OpenTracks",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "activities on date",
    },
    {
        "task_id": "SportsTrackerActivitiesCountForWeek",
        "app": "OpenTracks",
        "task_family": "OpenTracks",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "activities count",
    },
    {
        "task_id": "SportsTrackerLongestDistanceActivity",
        "app": "OpenTracks",
        "task_family": "OpenTracks",
        "task_mode": "ACTIVITY_STATS",
        "task_subtype": "longest distance",
    },
    # Simple Calendar (5)
    {
        "task_id": "SimpleCalendarEventsOnDate",
        "app": "Simple Calendar",
        "task_family": "Simple Calendar",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "events on date",
    },
    {
        "task_id": "SimpleCalendarEventsInTimeRange",
        "app": "Simple Calendar",
        "task_family": "Simple Calendar",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "events in time range",
    },
    {
        "task_id": "SimpleCalendarNextEvent",
        "app": "Simple Calendar",
        "task_family": "Simple Calendar",
        "task_mode": "EVENT_QUERY",
        "task_subtype": "next event",
    },
    {
        "task_id": "SimpleCalendarNextMeetingWithPerson",
        "app": "Simple Calendar",
        "task_family": "Simple Calendar",
        "task_mode": "EVENT_QUERY",
        "task_subtype": "meeting with person",
    },
    {
        "task_id": "SimpleCalendarAnyEventsOnDate",
        "app": "Simple Calendar",
        "task_family": "Simple Calendar",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "events on date",
    },
    # Joplin / Notes (4)
    {
        "task_id": "NotesRecipeIngredientCount",
        "app": "Joplin/Notes",
        "task_family": "Joplin/Notes",
        "task_mode": "RECIPE_INGREDIENT",
        "task_subtype": "recipe ingredient amount",
    },
    {
        "task_id": "NotesTodoItemCount",
        "app": "Joplin/Notes",
        "task_family": "Joplin/Notes",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "todo count",
    },
    {
        "task_id": "NotesMeetingAttendeeCount",
        "app": "Joplin/Notes",
        "task_family": "Joplin/Notes",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "todo status",
    },
    {
        "task_id": "NotesIsTodo",
        "app": "Joplin/Notes",
        "task_family": "Joplin/Notes",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "todo status",
    },
    # Tasks (4)
    {
        "task_id": "TasksDueOnDate",
        "app": "Tasks",
        "task_family": "Tasks",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "due date",
    },
    {
        "task_id": "TasksDueNextWeek",
        "app": "Tasks",
        "task_family": "Tasks",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "next week count",
    },
    {
        "task_id": "TasksHighPriorityTasksDueOnDate",
        "app": "Tasks",
        "task_family": "Tasks",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "high-priority due date",
    },
    {
        "task_id": "TasksHighPriorityTasks",
        "app": "Tasks",
        "task_family": "Tasks",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "high-priority tasks",
    },
    # Files / Browser (5)
    {
        "task_id": "FilesDeleteFile",
        "app": "Files",
        "task_family": "Files",
        "task_mode": "DELETE_COMMIT",
        "task_subtype": "delete file",
    },
    {
        "task_id": "FilesMoveFile",
        "app": "Files",
        "task_family": "Files",
        "task_mode": "NAVIGATION_SEARCH",
        "task_subtype": "move file",
    },
    {
        "task_id": "BrowserMaze",
        "app": "Browser",
        "task_family": "Browser",
        "task_mode": "NAVIGATION_SEARCH",
        "task_subtype": "browser maze",
    },
    {
        "task_id": "BrowserMultiply",
        "app": "Browser",
        "task_family": "Browser",
        "task_mode": "NAVIGATION_SEARCH",
        "task_subtype": "browser multiply",
    },
    {
        "task_id": "BrowserDraw",
        "app": "Browser",
        "task_family": "Browser",
        "task_mode": "NAVIGATION_SEARCH",
        "task_subtype": "browser draw",
    },
    # Markor (5)
    {
        "task_id": "MarkorCreateFolder",
        "app": "Markor",
        "task_family": "Markor",
        "task_mode": "FORM_CREATE_EDIT",
        "task_subtype": "create folder",
    },
    {
        "task_id": "MarkorCreateNote",
        "app": "Markor",
        "task_family": "Markor",
        "task_mode": "FORM_CREATE_EDIT",
        "task_subtype": "create note",
    },
    {
        "task_id": "MarkorDeleteNewestNote",
        "app": "Markor",
        "task_family": "Markor",
        "task_mode": "DELETE_COMMIT",
        "task_subtype": "delete note",
    },
    {
        "task_id": "MarkorDeleteNote",
        "app": "Markor",
        "task_family": "Markor",
        "task_mode": "DELETE_COMMIT",
        "task_subtype": "delete note",
    },
    {
        "task_id": "MarkorEditNote",
        "app": "Markor",
        "task_family": "Markor",
        "task_mode": "FORM_CREATE_EDIT",
        "task_subtype": "edit note",
    },
    # Pro Expense (4)
    {
        "task_id": "ExpenseAddSingle",
        "app": "Pro Expense",
        "task_family": "Pro Expense",
        "task_mode": "FORM_CREATE_EDIT",
        "task_subtype": "add expense",
    },
    {
        "task_id": "ExpenseDeleteSingle",
        "app": "Pro Expense",
        "task_family": "Pro Expense",
        "task_mode": "DELETE_COMMIT",
        "task_subtype": "delete expense",
    },
    {
        "task_id": "ExpenseAddMultiple",
        "app": "Pro Expense",
        "task_family": "Pro Expense",
        "task_mode": "FORM_CREATE_EDIT",
        "task_subtype": "add multiple expense",
    },
    {
        "task_id": "ExpenseDeleteDuplicates",
        "app": "Pro Expense",
        "task_family": "Pro Expense",
        "task_mode": "DELETE_COMMIT",
        "task_subtype": "delete expense",
    },
    # Clock / Contacts / Settings (9 total)
    {
        "task_id": "ClockStopWatchRunning",
        "app": "Clock",
        "task_family": "Clock",
        "task_mode": "SIMPLE_VERIFY_OPEN",
        "task_subtype": "stopwatch",
    },
    {
        "task_id": "ClockTimerEntry",
        "app": "Clock",
        "task_family": "Clock",
        "task_mode": "SIMPLE_VERIFY_OPEN",
        "task_subtype": "timer entry",
    },
    {
        "task_id": "ClockStopWatchPausedVerify",
        "app": "Clock",
        "task_family": "Clock",
        "task_mode": "SIMPLE_VERIFY_OPEN",
        "task_subtype": "stopwatch verify",
    },
    {
        "task_id": "ContactsAddContact",
        "app": "Contacts",
        "task_family": "Contacts",
        "task_mode": "FORM_CREATE_EDIT",
        "task_subtype": "add contact",
    },
    {
        "task_id": "ContactsNewContactDraft",
        "app": "Contacts",
        "task_family": "Contacts",
        "task_mode": "FORM_CREATE_EDIT",
        "task_subtype": "new draft contact",
    },
    {
        "task_id": "AudioRecorderRecordAudioWithFileName",
        "app": "Audio Recorder",
        "task_family": "Audio Recorder",
        "task_mode": "FORM_CREATE_EDIT",
        "task_subtype": "record audio",
    },
    {
        "task_id": "SystemWifiTurnOnVerify",
        "app": "Settings",
        "task_family": "Settings",
        "task_mode": "SIMPLE_VERIFY_OPEN",
        "task_subtype": "system setting verify",
    },
    {
        "task_id": "SystemBluetoothTurnOnVerify",
        "app": "Settings",
        "task_family": "Settings",
        "task_mode": "SIMPLE_VERIFY_OPEN",
        "task_subtype": "system setting verify",
    },
]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(str(value).strip())
    except ValueError:
        return None
    return out


def _read_baseline(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("task_num") or s.startswith("task ") or "=" in s:
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        task = parts[0]
        rows[task] = {
            "baseline_success": _float_or_none(parts[3]),
            "baseline_steps": _float_or_none(parts[4]),
        }
    return rows


def _write_task_selection(run_root: Path, tasks: list[dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> None:
    output_json = run_root / "task_selection.json"
    for item in tasks:
        b = baseline.get(item["task_id"], {})
        item["baseline_success"] = b.get("baseline_success")
        item["baseline_steps"] = b.get("baseline_steps")
    output_json.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    output_md = run_root / "task_selection.md"
    lines = [
        "# aggressive_t2_shortcut 40-task frozen selection",
        "",
        "| # | task_id | app | task_family | task_mode | task_subtype | baseline_success | baseline_steps |",
        "|---:|---|---|---|---|---|---:|---:|",
    ]
    for idx, item in enumerate(tasks, start=1):
        lines.append(
            f"| {idx} | {item['task_id']} | {item['app']} | {item['task_family']} | "
            f"{item['task_mode']} | {item['task_subtype']} | {item.get('baseline_success') or ''} | "
            f"{item.get('baseline_steps') or ''} |"
        )
    output_md.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    (run_root / "frozen_final_task_selection.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")

    output_csv = run_root / "task_selection.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "task_id",
                "app",
                "task_family",
                "task_mode",
                "task_subtype",
                "baseline_success",
                "baseline_steps",
            ],
        )
        writer.writeheader()
        for idx, item in enumerate(tasks, start=1):
            writer.writerow({
                "idx": idx,
                **{k: item.get(k, "") for k in [
                    "task_id",
                    "app",
                    "task_family",
                    "task_mode",
                    "task_subtype",
                ]},
                "baseline_success": item.get("baseline_success", ""),
                "baseline_steps": item.get("baseline_steps", ""),
            })
    (run_root / "frozen_final_task_selection.csv").write_text(output_csv.read_text(encoding="utf-8"), encoding="utf-8")
    (run_root / "frozen_final_task_selection.json").write_text(output_json.read_text(encoding="utf-8"), encoding="utf-8")


def _write_frozen_config(run_root: Path) -> None:
    lines = [
        "# Frozen conservative MobileExplorer configuration.",
        "# Active t+2 is OFF for the main paper candidate.",
    ]
    for key, value in FROZEN_CONFIG.items():
        lines.append(f"{key}: {value}")
    (run_root / "final_conservative_design_config.yaml").write_text("\\n".join(lines) + "\\n", encoding="utf-8")


def _run_variant(
    variant_root: Path,
    variant_name: str,
    experiment_root: Path,
    tasks_str: str,
    extra_args: list[str],
) -> None:
    variant_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN,
        str(REPORT_SCRIPT),
        "--run",
        "--agent_name=explore_agent_gelab",
        "--suite_family=android_world",
        "--tasks=" + tasks_str,
        "--n_task_combinations=1",
        "--task_random_seed=31",
        "--fixed_task_seed",
        "--image_downsample_scale=1.0",
        "--baseline_table=results/4B_2.txt",
        "--experiment_root=" + str(variant_root),
        "--max_cases=40",
        "--explore_variant=" + variant_name,
    ]
    cmd.extend(extra_args)
    log = variant_root / "driver.log"
    print(f"[run] variant={variant_name}")
    print(f"[run] cmd={' '.join(cmd)}")
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"variant {variant_name} failed, see {log}")

    run_dirs = sorted(variant_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise RuntimeError(f"variant {variant_name}: no run_* child directory created")
    selected = run_dirs[0]
    manifest = {
        "variant": variant_name,
        "command": " ".join(cmd),
        "run_dir": str(selected),
    }
    (variant_root / "variant_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    missing = sorted(item["task_id"] for item in TASKS if item["task_id"] in {"", None})
    assert not missing
    task_ids = [x["task_id"] for x in TASKS]
    if len(set(task_ids)) != len(task_ids):
        raise RuntimeError("TASKS contains duplicated task_id")
    if len(TASKS) != 40:
        raise RuntimeError(f"TASKS length = {len(TASKS)}; expected 40")

    app_counts: dict[str, int] = {}
    for item in TASKS:
        app_counts[item["app"]] = app_counts.get(item["app"], 0) + 1
    max_per_app = max(app_counts.values())
    if max_per_app > 5:
        raise RuntimeError(f"max per-app tasks is {max_per_app}, exceeds 5")
    if len(app_counts) < 8:
        raise RuntimeError(f"only {len(app_counts)} apps covered; expected >= 8")

    baseline = _read_baseline(BASELINE_TABLE)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_root = REPO_ROOT / "results" / "aggressive_t2_shortcut" / f"run_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    _write_task_selection(run_root, [dict(item) for item in TASKS], baseline)
    _write_frozen_config(run_root)
    (run_root / "run_info.json").write_text(
        json.dumps(
            {
                "run_root": str(run_root),
                "timestamp": dt.datetime.now().isoformat(),
                "task_count": len(TASKS),
                "unique_apps": sorted(app_counts),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tasks_str = ",".join(task_ids)

    base_args = [
        "--explore_enable",
        "--explore_max_runs=10000",
        "--explore_max_step=10000",
        "--explore_branch_budget=1",
        "--explore_branch_depth=2",
        "--explore_back_limit=4",
        "--explore_replay_max_actions=6",
        "--explore_planned_only",
        "--explore_fallback_safe_candidates",
        "--explore_safe_click_only",
        "--explore_skip_launcher",
        "--explore_filter_launcher_relevance",
        "--explore_skip_destructive_goals",
        "--explore_fast_mode",
        "--explore_fast_state",
        "--explore_transaction_safe",
        "--explore_action_settle_s=0.25",
        "--explore_hint_policy=strict",
        "--explore_search_policy=task_gate",
        "--explore_search_strategy=best_first",
        "--explore_rollback_policy=improved",
        "--explore_fixed_framework",
        "--explore_answer_extractors",
        "--explore_slot_complete",
        "--explore_slot_policy_switcher",
        "--explore_enable_t2_lookahead",
        "--explore_lightweight_a11y_trace",
        "--explore_trace_a11y_limit=32",
    ]

    baseline_args = [
        "--no-explore_enable",
        "--explore_max_runs=0",
        "--explore_max_step=0",
        "--explore_branch_budget=0",
        "--explore_branch_depth=0",
        "--no-explore_slot_complete",
        "--no-explore_slot_policy_switcher",
        "--no-explore_answer_extractors",
        "--explore_disable_prompt_injection",
        "--no-explore_enable_t2_lookahead",
        "--t2_mode=off",
    ]
    main_args = base_args + ["--t2_mode=off"]
    no_prompt_args = base_args + ["--explore_disable_prompt_injection", "--t2_mode=off"]
    shadow_args = base_args + [
        "--t2_mode=shadow",
        "--t2_state_mode=FULL_A11Y_CONTROL",
        "--t2_allow_safe_search_input",
        "--t2_confidence_threshold=0.85",
        "--t2_min_top1_score=7.0",
        "--t2_min_score_margin=2.0",
    ]
    light_state_args = base_args + [
        "--t2_mode=off",
        "--t2_state_mode=HYBRID_LIGHT",
    ]

    variants = [
        ("V0_BASELINE", baseline_args),
        ("V_MAIN_CERTAINTY_EXPLORATION", main_args),
        ("V_NO_PROMPT_INJECTION", no_prompt_args),
        ("V_T2_SHADOW_ONLY", shadow_args),
        ("V_LIGHT_STATE_DIAG", light_state_args),
    ]

    for name, args in variants:
        variant_root = run_root / name
        _run_variant(variant_root, name, run_root, tasks_str, args)

    # Aggregate outputs.
    aggregate_cmd = [
        PYTHON_BIN,
        str(REPO_ROOT / "scripts" / "aggregate_aggressive_t2_shortcut_frozen.py"),
        str(run_root),
    ]
    subprocess.run(
        aggregate_cmd,
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(f"[run] finished. root={run_root}")


if __name__ == "__main__":
    main()
