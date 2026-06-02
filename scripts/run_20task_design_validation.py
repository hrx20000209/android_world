#!/usr/bin/env python3
"""Run a 20-task MobileExplorer baseline/main validation."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = REPO_ROOT / "scripts" / "run_exploration_experiment_report.py"
AGGREGATE_SCRIPT = REPO_ROOT / "scripts" / "aggregate_aggressive_t2_shortcut_frozen.py"
BASELINE_TABLE = REPO_ROOT / "results" / "4B_2.txt"
PYTHON_BIN = "python3"


TASKS: list[dict[str, Any]] = [
    {"task_id": "SportsTrackerActivityDuration", "app": "OpenTracks", "task_family": "OpenTracks", "task_mode": "ACTIVITY_STATS", "task_subtype": "activity duration"},
    {"task_id": "SportsTrackerActivitiesOnDate", "app": "OpenTracks", "task_family": "OpenTracks", "task_mode": "INFO_QUERY_COUNT", "task_subtype": "activities on date"},
    {"task_id": "SimpleCalendarNextMeetingWithPerson", "app": "Simple Calendar", "task_family": "Simple Calendar", "task_mode": "EVENT_QUERY", "task_subtype": "meeting with person"},
    {"task_id": "SimpleCalendarEventsOnDate", "app": "Simple Calendar", "task_family": "Simple Calendar", "task_mode": "INFO_QUERY_COUNT", "task_subtype": "events on date"},
    {"task_id": "NotesRecipeIngredientCount", "app": "Joplin/Notes", "task_family": "Joplin/Notes", "task_mode": "RECIPE_INGREDIENT", "task_subtype": "recipe ingredient amount"},
    {"task_id": "NotesTodoItemCount", "app": "Joplin/Notes", "task_family": "Joplin/Notes", "task_mode": "INFO_QUERY_COUNT", "task_subtype": "todo count"},
    {"task_id": "TasksDueNextWeek", "app": "Tasks", "task_family": "Tasks", "task_mode": "INFO_QUERY_COUNT", "task_subtype": "next week count"},
    {"task_id": "TasksHighPriorityTasksDueOnDate", "app": "Tasks", "task_family": "Tasks", "task_mode": "INFO_QUERY_COUNT", "task_subtype": "high priority due date"},
    {"task_id": "MarkorDeleteNote", "app": "Markor", "task_family": "Markor", "task_mode": "DELETE_COMMIT", "task_subtype": "delete note"},
    {"task_id": "MarkorCreateNote", "app": "Markor", "task_family": "Markor", "task_mode": "FORM_CREATE_EDIT", "task_subtype": "create note"},
    {"task_id": "ExpenseDeleteSingle", "app": "Pro Expense", "task_family": "Pro Expense", "task_mode": "DELETE_COMMIT", "task_subtype": "delete expense"},
    {"task_id": "ExpenseAddSingle", "app": "Pro Expense", "task_family": "Pro Expense", "task_mode": "FORM_CREATE_EDIT", "task_subtype": "add expense"},
    {"task_id": "ClockTimerEntry", "app": "Clock", "task_family": "Clock", "task_mode": "SIMPLE_VERIFY_OPEN", "task_subtype": "timer entry"},
    {"task_id": "ClockStopWatchRunning", "app": "Clock", "task_family": "Clock", "task_mode": "SIMPLE_VERIFY_OPEN", "task_subtype": "stopwatch running"},
    {"task_id": "SystemWifiTurnOnVerify", "app": "Settings", "task_family": "Settings", "task_mode": "SIMPLE_VERIFY_OPEN", "task_subtype": "wifi verify"},
    {"task_id": "SystemBluetoothTurnOnVerify", "app": "Settings", "task_family": "Settings", "task_mode": "SIMPLE_VERIFY_OPEN", "task_subtype": "bluetooth verify"},
    {"task_id": "ContactsAddContact", "app": "Contacts", "task_family": "Contacts", "task_mode": "FORM_CREATE_EDIT", "task_subtype": "add contact"},
    {"task_id": "FilesMoveFile", "app": "Files", "task_family": "Files", "task_mode": "NAVIGATION_SEARCH", "task_subtype": "move file"},
    {"task_id": "BrowserMaze", "app": "Browser", "task_family": "Browser", "task_mode": "NAVIGATION_SEARCH", "task_subtype": "browser maze"},
    {"task_id": "AudioRecorderRecordAudioWithFileName", "app": "Audio Recorder", "task_family": "Audio Recorder", "task_mode": "FORM_CREATE_EDIT", "task_subtype": "record audio"},
]


FROZEN_CONFIG: dict[str, str] = {
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


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


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
        rows[parts[0]] = {"baseline_success": _float_or_none(parts[3]), "baseline_steps": _float_or_none(parts[4])}
    return rows


def _write_selection(run_root: Path) -> None:
    baseline = _read_baseline(BASELINE_TABLE)
    rows = [dict(item) for item in TASKS]
    for item in rows:
        item.update(baseline.get(str(item["task_id"]), {}))
    (run_root / "task_selection.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# 20-task MobileExplorer baseline/main validation task selection",
        "",
        "| # | task_id | app | task_mode | task_subtype | baseline_success | baseline_steps |",
        "|---:|---|---|---|---|---:|---:|",
    ]
    for i, item in enumerate(rows, 1):
        md.append(
            f"| {i} | {item['task_id']} | {item['app']} | {item['task_mode']} | {item['task_subtype']} | "
            f"{item.get('baseline_success') or ''} | {item.get('baseline_steps') or ''} |"
        )
    (run_root / "frozen_final_task_selection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (run_root / "task_selection.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    config_lines = ["# 20-task baseline/main conservative config"]
    config_lines.extend(f"{k}: {v}" for k, v in FROZEN_CONFIG.items())
    (run_root / "final_conservative_design_config.yaml").write_text("\n".join(config_lines) + "\n", encoding="utf-8")


def _run_variant(variant_root: Path, variant_name: str, tasks_str: str, extra_args: list[str]) -> None:
    variant_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN,
        str(REPORT_SCRIPT),
        "--run",
        "--agent_name=explore_agent_gelab",
        "--suite_family=android_world",
        "--tasks=" + tasks_str,
        "--n_task_combinations=1",
        "--task_random_seed=43",
        "--fixed_task_seed",
        "--image_downsample_scale=1.0",
        "--baseline_table=results/4B_2.txt",
        "--experiment_root=" + str(variant_root),
        "--max_cases=20",
        "--a11y_method=uiautomator",
        "--explore_variant=" + variant_name,
    ]
    cmd.extend(extra_args)
    log = variant_root / "driver.log"
    print(f"[20-run] variant={variant_name}")
    print(f"[20-run] cmd={' '.join(cmd)}")
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"variant {variant_name} failed, see {log}")
    runs = sorted(variant_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if runs:
        (variant_root / "variant_manifest.json").write_text(
            json.dumps({"variant": variant_name, "run_dir": str(runs[0]), "command": " ".join(cmd)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="baseline,main", help="Only baseline,main is supported for this 20-task runner.")
    args = parser.parse_args()
    requested = {x.strip().lower() for x in args.variants.split(",") if x.strip()}
    if not requested <= {"baseline", "main", "all"}:
        raise RuntimeError("--variants supports baseline,main,all")
    if "all" in requested:
        requested = {"baseline", "main"}
    if len(TASKS) != 20:
        raise RuntimeError(f"expected 20 tasks, got {len(TASKS)}")
    apps = sorted({str(t["app"]) for t in TASKS})
    if len(apps) < 8:
        raise RuntimeError(f"expected >=8 apps, got {len(apps)}")
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_root = REPO_ROOT / "results" / "design_validation_meeting" / f"TWENTY_TASK_BASELINE_MAIN_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    _write_selection(run_root)
    (run_root / "run_info.json").write_text(
        json.dumps({"run_root": str(run_root), "task_count": 20, "app_count": len(apps), "apps": apps, "variants": sorted(requested)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tasks_str = ",".join(str(t["task_id"]) for t in TASKS)
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
    main_args = [
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
        "--t2_mode=off",
    ]
    if "baseline" in requested:
        _run_variant(run_root / "V0_BASELINE", "V0_BASELINE", tasks_str, baseline_args)
    if "main" in requested:
        _run_variant(run_root / "V_MAIN_CERTAINTY_EXPLORATION", "V_MAIN_CERTAINTY_EXPLORATION", tasks_str, main_args)
    subprocess.run([PYTHON_BIN, str(AGGREGATE_SCRIPT), str(run_root)], cwd=str(REPO_ROOT), check=True)
    print(f"[20-run] finished. root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
