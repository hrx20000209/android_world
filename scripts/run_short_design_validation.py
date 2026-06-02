#!/usr/bin/env python3
"""Run the short frozen MobileExplorer design validation for the meeting."""

from __future__ import annotations

import csv
import datetime as dt
import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = REPO_ROOT / "scripts" / "run_exploration_experiment_report.py"
AGGREGATE_SCRIPT = REPO_ROOT / "scripts" / "aggregate_aggressive_t2_shortcut_frozen.py"
BASELINE_TABLE = REPO_ROOT / "results" / "4B_2.txt"
PYTHON_BIN = "python3"
VARIANT_ALIASES = {
    "baseline": "V0_BASELINE",
    "v0": "V0_BASELINE",
    "main": "V_MAIN_CERTAINTY_EXPLORATION",
    "v_main": "V_MAIN_CERTAINTY_EXPLORATION",
    "no_prompt": "V_NO_PROMPT_INJECTION",
    "noprompt": "V_NO_PROMPT_INJECTION",
    "t2_shadow": "V_T2_SHADOW_ONLY",
    "shadow": "V_T2_SHADOW_ONLY",
}


TASKS: list[dict[str, Any]] = [
    {
        "task_id": "SportsTrackerActivityDuration",
        "app": "OpenTracks",
        "task_family": "OpenTracks",
        "task_mode": "ACTIVITY_STATS",
        "task_subtype": "activity duration",
        "role": "shortcut-friendly",
    },
    {
        "task_id": "SimpleCalendarNextMeetingWithPerson",
        "app": "Simple Calendar",
        "task_family": "Simple Calendar",
        "task_mode": "EVENT_QUERY",
        "task_subtype": "meeting with person",
        "role": "shortcut-friendly",
    },
    {
        "task_id": "NotesRecipeIngredientCount",
        "app": "Joplin/Notes",
        "task_family": "Joplin/Notes",
        "task_mode": "RECIPE_INGREDIENT",
        "task_subtype": "recipe ingredient amount",
        "role": "shortcut-friendly",
    },
    {
        "task_id": "TasksDueNextWeek",
        "app": "Tasks",
        "task_family": "Tasks",
        "task_mode": "INFO_QUERY_COUNT",
        "task_subtype": "next week count",
        "role": "shortcut-friendly",
    },
    {
        "task_id": "MarkorDeleteNote",
        "app": "Markor",
        "task_family": "Markor",
        "task_mode": "DELETE_COMMIT",
        "task_subtype": "delete note",
        "role": "no-harm/control",
    },
    {
        "task_id": "ExpenseDeleteSingle",
        "app": "Pro Expense",
        "task_family": "Pro Expense",
        "task_mode": "DELETE_COMMIT",
        "task_subtype": "delete expense",
        "role": "no-harm/control",
    },
    {
        "task_id": "ClockTimerEntry",
        "app": "Clock",
        "task_family": "Clock",
        "task_mode": "SIMPLE_VERIFY_OPEN",
        "task_subtype": "timer entry",
        "role": "no-harm/control",
    },
    {
        "task_id": "SystemWifiTurnOnVerify",
        "app": "Settings",
        "task_family": "Settings",
        "task_mode": "SIMPLE_VERIFY_OPEN",
        "task_subtype": "system setting verify",
        "role": "no-harm/control",
    },
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
        rows[parts[0]] = {
            "baseline_success": _float_or_none(parts[3]),
            "baseline_steps": _float_or_none(parts[4]),
        }
    return rows


def _write_task_selection(run_root: Path, baseline: dict[str, dict[str, Any]]) -> None:
    rows = [dict(item) for item in TASKS]
    for item in rows:
        b = baseline.get(str(item["task_id"]), {})
        item["baseline_success"] = b.get("baseline_success")
        item["baseline_steps"] = b.get("baseline_steps")
    (run_root / "task_selection.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "# Short frozen MobileExplorer design validation task selection",
        "",
        "| # | task_id | app | task_family | task_mode | task_subtype | role | baseline_success | baseline_steps |",
        "|---:|---|---|---|---|---|---|---:|---:|",
    ]
    for idx, item in enumerate(rows, start=1):
        md.append(
            f"| {idx} | {item['task_id']} | {item['app']} | {item['task_family']} | "
            f"{item['task_mode']} | {item['task_subtype']} | {item['role']} | "
            f"{item.get('baseline_success') or ''} | {item.get('baseline_steps') or ''} |"
        )
    text = "\n".join(md) + "\n"
    (run_root / "task_selection.md").write_text(text, encoding="utf-8")
    (run_root / "frozen_final_task_selection.md").write_text(text, encoding="utf-8")
    with (run_root / "task_selection.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_config(run_root: Path) -> None:
    lines = [
        "# Short frozen conservative MobileExplorer configuration.",
        "# Active t+2 is OFF; V_T2_SHADOW_ONLY logs shortcut feasibility only.",
    ]
    for key, value in FROZEN_CONFIG.items():
        lines.append(f"{key}: {value}")
    (run_root / "final_conservative_design_config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "--task_random_seed=41",
        "--fixed_task_seed",
        "--image_downsample_scale=1.0",
        "--baseline_table=results/4B_2.txt",
        "--experiment_root=" + str(variant_root),
        "--max_cases=12",
        "--a11y_method=uiautomator",
        "--explore_variant=" + variant_name,
    ]
    cmd.extend(extra_args)
    log = variant_root / "driver.log"
    print(f"[short-run] variant={variant_name}")
    print(f"[short-run] cmd={' '.join(cmd)}")
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, text=True, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"variant {variant_name} failed, see {log}")
    run_dirs = sorted(variant_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise RuntimeError(f"variant {variant_name}: no run_* child directory created")
    (variant_root / "variant_manifest.json").write_text(
        json.dumps(
            {"variant": variant_name, "command": " ".join(cmd), "run_dir": str(run_dirs[0])},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short frozen MobileExplorer design validation."
    )
    parser.add_argument(
        "--variants",
        default="all",
        help=(
            "Comma-separated variants to run. Use all, baseline, main, "
            "no_prompt, t2_shadow. Example: --variants baseline,main"
        ),
    )
    return parser.parse_args()


def _resolve_variants(value: str) -> set[str]:
    raw = [part.strip().lower() for part in str(value or "all").split(",") if part.strip()]
    if not raw or "all" in raw:
        return {
            "V0_BASELINE",
            "V_MAIN_CERTAINTY_EXPLORATION",
            "V_NO_PROMPT_INJECTION",
            "V_T2_SHADOW_ONLY",
        }
    selected: set[str] = set()
    unknown: list[str] = []
    for name in raw:
        resolved = VARIANT_ALIASES.get(name)
        if resolved is None:
            unknown.append(name)
        else:
            selected.add(resolved)
    if unknown:
        raise RuntimeError(f"unknown --variants entries: {', '.join(unknown)}")
    if not selected:
        raise RuntimeError("--variants resolved to an empty set")
    return selected


def main() -> int:
    args = _parse_args()
    selected_variants = _resolve_variants(args.variants)
    if len(TASKS) < 8 or len(TASKS) > 12:
        raise RuntimeError(f"short validation requires 8-12 tasks, got {len(TASKS)}")
    apps = {item["app"] for item in TASKS}
    if len(apps) < 8:
        raise RuntimeError(f"short validation requires >=8 apps, got {len(apps)}")

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_root = REPO_ROOT / "results" / "design_validation_meeting" / f"SHORT_VALIDATION_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    _write_config(run_root)
    _write_task_selection(run_root, _read_baseline(BASELINE_TABLE))
    (run_root / "run_info.json").write_text(
        json.dumps(
            {
                "run_root": str(run_root),
                "timestamp": dt.datetime.now().isoformat(),
                "task_count": len(TASKS),
                "app_count": len(apps),
                "apps": sorted(apps),
                "selected_variants": sorted(selected_variants),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    tasks_str = ",".join(str(item["task_id"]) for item in TASKS)
    base_args = [
        "--explore_enable",
        "--explore_max_runs=10000",
        "--explore_max_step=10000",
        "--explore_branch_budget=1",
        "--explore_branch_depth=2",
        "--explore_back_limit=4",
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
    variants = [
        (
            "V0_BASELINE",
            [
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
            ],
        ),
        ("V_MAIN_CERTAINTY_EXPLORATION", base_args + ["--t2_mode=off"]),
        (
            "V_NO_PROMPT_INJECTION",
            base_args + ["--explore_disable_prompt_injection", "--t2_mode=off"],
        ),
        (
            "V_T2_SHADOW_ONLY",
            base_args
            + [
                "--t2_mode=shadow",
                "--t2_state_mode=FULL_A11Y_CONTROL",
                "--t2_allow_safe_search_input",
                "--t2_confidence_threshold=0.85",
                "--t2_min_top1_score=7.0",
                "--t2_min_score_margin=2.0",
            ],
        ),
    ]
    variants = [(name, args) for name, args in variants if name in selected_variants]

    for name, args in variants:
        _run_variant(run_root / name, name, tasks_str, args)

    subprocess.run([PYTHON_BIN, str(AGGREGATE_SCRIPT), str(run_root)], cwd=str(REPO_ROOT), check=True)
    report = run_root / "final_design_decision_report_cn.md"
    if report.exists():
        (run_root / "active_safe_diverse_design_validation_short_cn.md").write_text(
            report.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    print(f"[short-run] finished. root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
