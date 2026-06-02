#!/usr/bin/env python3
"""Run the final MobileExplorer AndroidWorld configuration.

This entry point is intentionally thin: it pins the current best design to a
single reproducible command and delegates execution/reporting to
run_exploration_experiment_report.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results" / "final_mobileexplorer_androidworld"

# Fast sanity check: covers commit safety, note/ingredient extraction, and
# OpenTracks-style metric search without taking an overnight run.
QUICK_TASKS = [
    "ExpenseDeleteSingle",
    "NotesRecipeIngredientCount",
    "SportsTrackerActivityDuration",
]

# One representative per major AndroidWorld app family. This is meant for
# app-adapter validation before running the full 116-task benchmark.
APP_SMOKE_TASKS = [
    "AudioRecorderRecordAudio",
    "BrowserMultiply",
    "CameraTakeVideo",
    "ClockStopWatchRunning",
    "ContactsAddContact",
    "ExpenseDeleteSingle",
    "FilesDeleteFile",
    "MarkorDeleteNewestNote",
    "NotesRecipeIngredientCount",
    "OpenAppTaskEval",
    "OsmAndFavorite",
    "RecipeAddSingleRecipe",
    "RetroPlaylistDuration",
    "SaveCopyOfReceiptTaskEval",
    "SimpleCalendarEventsInTimeRange",
    "SimpleDrawProCreateDrawing",
    "SimpleSmsSend",
    "SportsTrackerActivityDuration",
    "SystemWifiTurnOnVerify",
    "TasksDueOnDate",
    "VlcCreatePlaylist",
]

# The same 20-task diagnostic set used in previous MobileExplorer comparisons.
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


FINAL_EXPLORATION_ARGS = [
    "--agent_name=explore_agent_gelab",
    "--suite_family=android_world",
    "--n_task_combinations=1",
    "--fixed_task_seed",
    "--image_downsample_scale=1.0",
    "--a11y_method=uiautomator",
    "--baseline_table=results/4B_2.txt",
    "--no-explore_diagnostic_full",
    "--no-explore_force_every_step",
    "--explore_fast_mode",
    "--explore_transaction_safe",
    "--explore_max_runs=10000",
    "--explore_max_step=10000",
    "--explore_branch_budget=3",
    "--explore_branch_depth=3",
    "--explore_back_limit=4",
    "--explore_replay_max_actions=6",
    "--no-explore_planned_only",
    "--explore_fallback_safe_candidates",
    "--explore_safe_click_only",
    "--explore_skip_launcher",
    "--explore_filter_launcher_relevance",
    "--explore_skip_destructive_goals",
    "--explore_strategy=dfs",
    "--explore_hint_policy=strict",
    "--explore_search_policy=task_gate",
    "--explore_search_strategy=best_first",
    "--explore_rollback_policy=improved",
    "--explore_fixed_framework",
    "--explore_answer_extractors",
    "--explore_slot_complete",
    "--explore_slot_policy_switcher",
    "--explore_no_action_hint",
    "--explore_lightweight_a11y_trace",
    "--explore_trace_a11y_limit=32",
    "--explore_variant=FINAL_APP_ADAPTER_SLOT_BEST_FIRST",
]


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def _tasks_for_phase(phase: str, custom_tasks: str) -> list[str]:
    if custom_tasks.strip():
        return [x.strip() for x in custom_tasks.split(",") if x.strip()]
    if phase == "quick":
        return QUICK_TASKS
    if phase == "app-smoke":
        return APP_SMOKE_TASKS
    if phase == "20-task":
        return TASKS_20
    if phase == "full":
        return []
    raise ValueError(f"Unknown phase: {phase}")


def _write_design_note(run_root: Path, phase: str, tasks: list[str], command: list[str]) -> None:
    task_text = "full AndroidWorld suite" if not tasks else "\n".join(f"- {task}" for task in tasks)
    note = f"""# Final MobileExplorer Run Design

## Design
- Search: slot-complete evidence-guided best-first.
- Evidence objective: fill missing task slots before injecting any hint.
- App adapters: lightweight app-family extractors are enabled when a task/screen matches a known app family; otherwise the generic slot extractor still runs.
- Prompt injection: strict only. This run disables ACTION_HINT by default and only injects ANSWER_HINT / AVOID_HINT / SCHEMA_HINT / RISK_HINT.
- Safety: risky actions are never executed speculatively; rollback uses the improved verifier and does not suppress the main action unless the planned precondition is absent.
- Logging: lightweight a11y trace is enabled, with latency and slot evidence written into traces and the generated Chinese report.

## Phase
`{phase}`

## Tasks
{task_text}

## Command
```bash
{' '.join(command)}
```
"""
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "final_design.md").write_text(note, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final MobileExplorer AndroidWorld design.")
    parser.add_argument(
        "--phase",
        choices=("quick", "app-smoke", "20-task", "full"),
        default="quick",
        help="quick verifies the command; app-smoke runs one task per app family; full runs the whole suite.",
    )
    parser.add_argument("--tasks", default="", help="Comma-separated custom task names. Overrides --phase task set.")
    parser.add_argument("--experiment_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--report_cases", type=int, default=20)
    parser.add_argument("--print_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--extra_arg", action="append", default=[], help="Extra arg passed to run_exploration_experiment_report.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.experiment_root).expanduser().resolve() / f"run_{_timestamp()}_{args.phase}"
    tasks = _tasks_for_phase(args.phase, args.tasks)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_exploration_experiment_report.py"),
        "--run",
        f"--experiment_root={run_root}",
        f"--max_cases={args.report_cases}",
        *FINAL_EXPLORATION_ARGS,
        *args.extra_arg,
    ]
    if tasks:
        command.append(f"--tasks={','.join(tasks)}")
    _write_design_note(run_root, args.phase, tasks, command)
    print(f"[final] run_root={run_root}")
    print(f"[final] design={run_root / 'final_design.md'}")
    print("[final] command=" + " ".join(command))
    if args.print_only:
        return 0
    return subprocess.call(command, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
