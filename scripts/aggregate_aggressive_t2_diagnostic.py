#!/usr/bin/env python3
"""Aggregate aggressive t+2 shortcut diagnostic runs into Chinese reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VARIANT_ORDER = [
    "V0_BASELINE",
    "V12_NO_SHORTCUT",
    "V16_T2_SHADOW",
    "V16_T2_SHADOW_DEBUG",
    "V16_SHADOW_RELAXED_SEARCH_ONLY",
    "V17_T2_ACTIVE_SAFE_STRICT",
    "V18_T2_ACTIVE_SAFE_WITH_SAFE_SEARCH_INPUT",
    "V18_T2_ACTIVE_SAFE_WITH_SAFE_SEARCH_INPUT_DEBUG",
    "V18_ACTIVE_SAFE_SEARCH_ONLY",
    "V18_ACTIVE_SAFE_SEARCH_ONLY_RELAXED",
    "V19_T2_ACTIVE_SAFE_LIGHT_STATE",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            rows.append({"_raw": line})
    return rows


def _read_jsonl_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.extend(_read_jsonl(path))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, str) and value.lower() == "nan":
            return default
        return float(value)
    except Exception:
        return default


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return None
    return float(statistics.mean(values))


def _task_meta(task: str) -> dict[str, str]:
    if task.startswith("SportsTracker"):
        family = "OpenTracks/SportsTracker"
        mode = "ACTIVITY_STATS"
    elif task.startswith("SimpleCalendar"):
        family = "Simple Calendar"
        mode = "INFO_QUERY_COUNT"
    elif task.startswith("Notes"):
        family = "Joplin/Notes"
        mode = "RECIPE_INGREDIENT" if "Recipe" in task else "INFO_QUERY_COUNT"
    elif task.startswith("Tasks"):
        family = "Tasks"
        mode = "INFO_QUERY_COUNT"
    elif task.startswith("Files"):
        family = "Files"
        mode = "NAVIGATION_SEARCH" if "Open" in task or "Move" in task else "DELETE_COMMIT"
    elif task.startswith("Browser"):
        family = "Browser"
        mode = "NAVIGATION_SEARCH"
    elif task.startswith("Markor"):
        family = "Markor"
        mode = "DELETE_COMMIT" if "Delete" in task else "FORM_CREATE_EDIT"
    elif task.startswith("Expense"):
        family = "Pro Expense"
        mode = "DELETE_COMMIT" if "Delete" in task else "FORM_CREATE_EDIT"
    elif task.startswith("Clock"):
        family = "Clock"
        mode = "SIMPLE_VERIFY_OPEN" if "Verify" in task else "NAVIGATION_SEARCH"
    elif task.startswith("AudioRecorder"):
        family = "Audio Recorder"
        mode = "FORM_CREATE_EDIT"
    elif task.startswith("Contacts"):
        family = "Contacts"
        mode = "FORM_CREATE_EDIT"
    elif task.startswith("System"):
        family = "Settings"
        mode = "SIMPLE_VERIFY_OPEN"
    else:
        family = "Other"
        mode = "UNKNOWN"
    return {"app": family, "task_family": family, "task_mode": mode, "task_subtype": task}


def _canonical_task_key(text: str) -> str:
    low = str(text or "").lower()
    if not low:
        return ""
    if "chicken alfredo" in low or "spirulina" in low:
        return "NotesRecipeIngredientCount"
    if "ideas" in low and "todo" in low:
        return "NotesTodoItemCount"
    if "task.html" in low and "bottom-right" in low:
        return "BrowserMaze"
    if "task.html" in low:
        return "BrowserMaze"
    if "skiing" in low and "october 12" in low:
        return "SportsTrackerActivityDuration"
    if "kayaking" in low and "distance" in low:
        return "SportsTrackerTotalDistanceForCategoryOverInterval"
    if "meeting with ava" in low:
        return "SimpleCalendarNextMeetingWithPerson"
    if "high priority" in low and "due" in low:
        return "TasksHighPriorityTasksDueOnDate"
    if "due next week" in low:
        return "TasksDueNextWeek"
    if "due october 24" in low:
        return "TasksDueOnDate"
    return str(text or "")


def _latest_run_dir(variant_dir: Path) -> Path | None:
    runs = sorted([p for p in variant_dir.glob("run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _load_variant(root: Path, variant: str) -> dict[str, Any]:
    variant_dir = root / variant
    run_dir = _latest_run_dir(variant_dir)
    if run_dir is None:
        return {"variant": variant, "variant_dir": variant_dir, "run_dir": None, "summary": {}}
    summary = _read_json(run_dir / "report" / "summary.json")
    rollup = _read_json(run_dir / "exploration_rollup.json")
    if rollup:
        summary["agent_rollup"] = rollup
    root_plan_path = run_dir / "shortcut_plans.jsonl"
    root_event_path = run_dir / "shortcut_events.jsonl"
    root_shadow_path = run_dir / "shortcut_shadow_eval.jsonl"
    root_direct_path = run_dir / "direct_answer_candidates.jsonl"
    trace_plan_path = run_dir / "traces" / "shortcut_plans.jsonl"
    trace_event_path = run_dir / "traces" / "shortcut_events.jsonl"
    trace_shadow_path = run_dir / "traces" / "shortcut_shadow_eval.jsonl"
    trace_direct_path = run_dir / "traces" / "direct_answer_candidates.jsonl"
    plan_paths = [root_plan_path] if root_plan_path.exists() else ([trace_plan_path] if trace_plan_path.exists() else list(run_dir.rglob("shortcut_plans.jsonl")))
    event_paths = [root_event_path] if root_event_path.exists() else ([trace_event_path] if trace_event_path.exists() else list(run_dir.rglob("shortcut_events.jsonl")))
    shadow_paths = [root_shadow_path] if root_shadow_path.exists() else ([trace_shadow_path] if trace_shadow_path.exists() else list(run_dir.rglob("shortcut_shadow_eval.jsonl")))
    direct_paths = [root_direct_path] if root_direct_path.exists() else ([trace_direct_path] if trace_direct_path.exists() else list(run_dir.rglob("direct_answer_candidates.jsonl")))
    state_path = run_dir / "state_acquisition_metrics.csv"
    if not state_path.exists() and (run_dir / "traces" / "state_acquisition_metrics.csv").exists():
        state_path = run_dir / "traces" / "state_acquisition_metrics.csv"
    return {
        "variant": variant,
        "variant_dir": variant_dir,
        "run_dir": run_dir,
        "summary": summary,
        "step_metrics": _read_csv(run_dir / "report" / "step_metrics.csv"),
        "plans": _read_jsonl_many(plan_paths),
        "events": _read_jsonl_many(event_paths),
        "shadow": _read_jsonl_many(shadow_paths),
        "direct": _read_jsonl_many(direct_paths),
        "exploration": _read_jsonl(run_dir / "traces" / "exploration_trace.jsonl"),
        "rollback": _read_jsonl(run_dir / "traces" / "rollback_trace.jsonl"),
        "state_metrics": _read_csv(state_path),
        "shortcut_plans_exists": any(path.exists() for path in plan_paths),
        "shortcut_events_exists": any(path.exists() for path in event_paths),
        "shortcut_shadow_exists": any(path.exists() for path in shadow_paths),
        "state_metrics_exists": state_path.exists(),
    }


def _episode_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((summary.get("episodes") or {}).get("task_episode_rows") or []))


def _variant_summary(row: dict[str, Any]) -> dict[str, Any]:
    summary = row.get("summary") or {}
    compare = summary.get("baseline_compare") or {}
    episodes = _episode_rows(summary)
    success_values = [1.0 if item.get("success") else 0.0 for item in episodes]
    steps = [_float(item.get("episode_length"), math.nan) for item in episodes]
    exception_count = sum(1 for item in episodes if bool(item.get("exception")))
    shortcut = ((summary.get("agent_rollup") or {}).get("shortcut") or {})
    state = ((summary.get("agent_rollup") or {}).get("state_acquisition") or {})
    if not state and row.get("state_metrics"):
        state = {
            str(item.get("metric")): _float(item.get("value"))
            for item in row.get("state_metrics") or []
            if item.get("metric")
        }
    if not shortcut:
        shortcut = summary.get("shortcut") or {}
    plans_missing = not bool(row.get("shortcut_plans_exists"))
    events_missing = not bool(row.get("shortcut_events_exists"))
    shadow_missing = not bool(row.get("shortcut_shadow_exists"))
    state_missing = not bool(row.get("state_metrics_exists")) and not bool(state)
    plans = row.get("plans") or []
    events = row.get("events") or []
    shadow = row.get("shadow") or []
    plan_with_action = [item for item in plans if not str(item.get("no_plan_reason") or "").strip()]
    search_input_plans = [
        item for item in plan_with_action
        if bool(((item.get("shortcut_t2_action") or {}).get("is_safe_search_input")))
    ]
    exact_click_plans = [
        item for item in plan_with_action
        if bool(((item.get("shortcut_t2_action") or {}).get("is_exact_result_click")))
    ]
    would_fire = [item for item in events if bool(item.get("would_fire"))]
    fired = [item for item in events if bool(item.get("fired"))]
    shadow_would_fire = [item for item in shadow if bool(item.get("would_fire"))]
    shadow_matches = [item for item in shadow_would_fire if bool(item.get("action_match"))]
    return {
        "variant": row["variant"],
        "task_count": len(episodes),
        "success_rate": _mean(success_values),
        "avg_steps": _mean(steps),
        "exception_count": exception_count,
        "baseline_table_success_rate": compare.get("avg_baseline_success_rate"),
        "baseline_table_avg_steps": compare.get("avg_baseline_episode_length"),
        "shortcut_plan_count": "MISSING" if plans_missing else len(plans),
        "shortcut_candidate_count": "MISSING" if plans_missing else len(plan_with_action),
        "eligible_plan_count": "MISSING" if events_missing else len(would_fire),
        "search_input_plan_count": "MISSING" if plans_missing else len(search_input_plans),
        "exact_result_click_plan_count": "MISSING" if plans_missing else len(exact_click_plans),
        "shortcut_would_fire_count": "MISSING" if events_missing else len(would_fire),
        "shadow_eval_count": "MISSING" if shadow_missing else len(shadow_would_fire),
        "shadow_action_match_count": "MISSING" if shadow_missing else len(shadow_matches),
        "shadow_precision": (
            "MISSING"
            if shadow_missing
            else (float(len(shadow_matches)) / len(shadow_would_fire) if shadow_would_fire else None)
        ),
        "active_shortcut_fired_count": "MISSING" if events_missing else len(fired),
        "skipped_vlm_calls": "MISSING" if events_missing else sum(1 for item in fired if bool(item.get("skipped_vlm_reasoning"))),
        "direct_answer_candidate_count": int(_float(shortcut.get("direct_answer_candidate_count"))),
        "full_a11y_calls": "MISSING" if state_missing else _float(state.get("full_a11y_calls")),
        "screenshot_calls": "MISSING" if state_missing else _float(state.get("screenshot_calls")),
        "activity_calls": "MISSING" if state_missing else _float(state.get("activity_calls")),
        "cached_state_hits": "MISSING" if state_missing else _float(state.get("cached_state_hits")),
        "rollback_full_a11y_fallbacks": "MISSING" if state_missing else _float(state.get("rollback_full_a11y_fallbacks")),
        "branch_full_a11y_calls": "MISSING" if state_missing else _float(state.get("branch_full_a11y_calls")),
        "shortcut_plans_file": "MISSING" if plans_missing else "present",
        "shortcut_events_file": "MISSING" if events_missing else "present",
        "shortcut_shadow_file": "MISSING" if shadow_missing else "present",
        "state_metrics_file": "MISSING" if state_missing else "present",
    }


def _task_rows(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_task: dict[str, dict[str, Any]] = {}
    for variant in variants:
        if variant["variant"] != "V0_BASELINE":
            continue
        for item in _episode_rows(variant.get("summary") or {}):
            task = str(item.get("task") or "")
            baseline_by_task[task] = {
                "baseline_success": 1.0 if item.get("success") else 0.0,
                "baseline_steps": _float(item.get("episode_length"), math.nan),
            }
    rows: list[dict[str, Any]] = []
    for variant in variants:
        events_by_task = Counter(_canonical_task_key(str(item.get("task_id") or item.get("goal") or "")) for item in variant.get("events") or [])
        would_by_task = Counter(
            _canonical_task_key(str(item.get("task_id") or item.get("goal") or ""))
            for item in variant.get("events") or []
            if item.get("would_fire")
        )
        fired_by_task = Counter(
            _canonical_task_key(str(item.get("task_id") or item.get("goal") or ""))
            for item in variant.get("events") or []
            if item.get("fired")
        )
        skipped_by_task = Counter(
            _canonical_task_key(str(item.get("task_id") or item.get("goal") or ""))
            for item in variant.get("events") or []
            if item.get("skipped_vlm_reasoning")
        )
        plans_by_task = Counter(_canonical_task_key(str(item.get("task_id") or item.get("goal") or "")) for item in variant.get("plans") or [])
        candidate_by_task = Counter(
            _canonical_task_key(str(item.get("task_id") or item.get("goal") or ""))
            for item in variant.get("plans") or []
            if not str(item.get("no_plan_reason") or "").strip()
        )
        for item in _episode_rows(variant.get("summary") or {}):
            task = str(item.get("task") or "")
            meta = _task_meta(task)
            base = baseline_by_task.get(task, {})
            variant_success = 1.0 if item.get("success") else 0.0
            baseline_success = base.get("baseline_success")
            rescued = bool(baseline_success == 0.0 and variant_success == 1.0)
            broken = bool(baseline_success == 1.0 and variant_success == 0.0)
            rows.append(
                {
                    "variant": variant["variant"],
                    "task_id": task,
                    **meta,
                    "baseline_success": baseline_success,
                    "baseline_steps": base.get("baseline_steps"),
                    "variant_success": variant_success,
                    "variant_steps": _float(item.get("episode_length"), math.nan),
                    "exception": bool(item.get("exception")),
                    "rescued": rescued,
                    "broken": broken,
                    "shortcut_plan_count": plans_by_task.get(task, 0),
                    "candidate_plan_count": candidate_by_task.get(task, 0),
                    "would_fire_count": would_by_task.get(task, 0),
                    "shortcut_event_count": events_by_task.get(task, 0),
                    "shortcut_fired_count": fired_by_task.get(task, 0),
                    "skipped_vlm_calls": skipped_by_task.get(task, 0),
                }
            )
    return rows


def _state_metric_rows(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        summary = _variant_summary(variant)
        rows.append(
            {
                "variant": variant["variant"],
                "full_a11y_calls": summary["full_a11y_calls"],
                "screenshot_calls": summary["screenshot_calls"],
                "activity_calls": summary["activity_calls"],
                "cached_state_hits": summary["cached_state_hits"],
                "rollback_full_a11y_fallbacks": summary["rollback_full_a11y_fallbacks"],
                "branch_full_a11y_calls": summary["branch_full_a11y_calls"],
                "success_rate": summary["success_rate"],
                "avg_steps": summary["avg_steps"],
                "shadow_precision": summary["shadow_precision"],
            }
        )
    return rows


def _plot(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 4.5))
    ax.bar(labels, values, color="#2F6F7E")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_figures(root: Path, summaries: list[dict[str, Any]]) -> dict[str, Path]:
    fig_dir = root / "figures"
    labels = [row["variant"].replace("V", "V\n") for row in summaries]
    figures: dict[str, Path] = {}
    specs = [
        ("success_rate", "Success rate by variant", [100.0 * _float(row.get("success_rate")) for row in summaries], "%"),
        ("avg_steps", "Average steps by variant", [_float(row.get("avg_steps")) for row in summaries], "steps"),
        ("shortcut_fired", "Active shortcut fired", [_float(row.get("active_shortcut_fired_count")) for row in summaries], "count"),
        ("shortcut_plans", "Shortcut plans generated", [_float(row.get("shortcut_plan_count")) for row in summaries], "count"),
        ("full_a11y", "Full a11y calls", [_float(row.get("full_a11y_calls")) for row in summaries], "calls"),
    ]
    for name, title, values, ylabel in specs:
        path = fig_dir / f"{name}.png"
        _plot(path, title, labels, values, ylabel)
        if path.exists():
            figures[name] = path
    return figures


def _fmt_pct(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if value == "MISSING":
        return "MISSING"
    return f"{100.0 * _float(value):.1f}%"


def _fmt_num(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if value == "MISSING":
        return "MISSING"
    return f"{_float(value):.2f}"


def _fmt_count(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if value == "MISSING":
        return "MISSING"
    return str(int(_float(value)))


def _write_report(root: Path, summaries: list[dict[str, Any]], task_rows: list[dict[str, Any]], figures: dict[str, Path]) -> None:
    by_variant = {row["variant"]: row for row in summaries}
    baseline = by_variant.get("V0_BASELINE", {})
    best = max(summaries, key=lambda r: (_float(r.get("success_rate")), -_float(r.get("avg_steps"), 9999)), default={})
    shadow = by_variant.get("V16_T2_SHADOW", {})
    v18 = by_variant.get("V18_T2_ACTIVE_SAFE_WITH_SAFE_SEARCH_INPUT", {})
    v19 = by_variant.get("V19_T2_ACTIVE_SAFE_LIGHT_STATE", {})
    active_candidates = [row for row in summaries if row["variant"].startswith(("V17", "V18", "V19"))]
    best_active = max(active_candidates, key=lambda r: (_float(r.get("success_rate")), _float(r.get("active_shortcut_fired_count"))), default={})

    task_by_variant = defaultdict(list)
    for row in task_rows:
        task_by_variant[row["variant"]].append(row)
    broken_by_active = [
        row for row in task_by_variant.get(best_active.get("variant", ""), [])
        if row.get("broken")
    ]
    rescued_by_active = [
        row for row in task_by_variant.get(best_active.get("variant", ""), [])
        if row.get("rescued")
    ]

    full_a11y_drop = None
    if baseline and v19:
        base_calls = _float(baseline.get("full_a11y_calls"))
        light_calls = _float(v19.get("full_a11y_calls"))
        if base_calls > 0:
            full_a11y_drop = (base_calls - light_calls) / base_calls

    acceptance = {
        "shortcut_plan_count > 0": any(_float(row.get("shortcut_plan_count")) > 0 for row in summaries),
        "shadow precision >= 80%": _float(shadow.get("shadow_precision"), -1.0) >= 0.8,
        "active_shortcut_fired_count >= 5": _float(best_active.get("active_shortcut_fired_count")) >= 5,
        "skipped_vlm_calls >= 5": _float(best_active.get("skipped_vlm_calls")) >= 5,
        "harmful_shortcut_count = 0": len(broken_by_active) == 0,
        "success >= baseline + 3pp": (
            _float(best_active.get("success_rate"), -1.0) >= _float(baseline.get("success_rate")) + 0.03
            if baseline and best_active
            else False
        ),
        "avg steps <= baseline + 0.3": (
            _float(best_active.get("avg_steps"), 9999.0) <= _float(baseline.get("avg_steps"), 0.0) + 0.3
            if baseline and best_active
            else False
        ),
        "rescued >= broken": len(rescued_by_active) >= len(broken_by_active),
        "HYBRID_LIGHT a11y -30% and no success loss": (
            bool(full_a11y_drop is not None and full_a11y_drop >= 0.30)
            and _float(v19.get("success_rate"), -1.0) >= _float(v18.get("success_rate"), -1.0)
        ),
    }

    app_stats = defaultdict(lambda: {"tasks": 0, "success": 0.0, "steps": []})
    for row in task_by_variant.get(best.get("variant", ""), []):
        app = str(row.get("app") or "Other")
        app_stats[app]["tasks"] += 1
        app_stats[app]["success"] += _float(row.get("variant_success"))
        app_stats[app]["steps"].append(_float(row.get("variant_steps"), math.nan))

    lines: list[str] = [
        "# aggressive t+2 shortcut 诊断总报告",
        "",
        f"- run root: `{root}`",
        f"- 任务规模: {int(_float(best.get('task_count')))} tasks" if best else "- 任务规模: NA",
        f"- 最佳总体变体: `{best.get('variant', 'NA')}`，success={_fmt_pct(best.get('success_rate'))}，avg_steps={_fmt_num(best.get('avg_steps'))}",
        f"- 最佳 active shortcut 变体: `{best_active.get('variant', 'NA')}`，fired={int(_float(best_active.get('active_shortcut_fired_count')))}，skipped_vlm_calls={int(_float(best_active.get('skipped_vlm_calls')))}",
        "",
        "## 变体总览",
        "",
        "| variant | success | avg steps | exceptions | plans | would_fire | shadow precision | active fired | skipped VLM | full a11y |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {_fmt_pct(row.get('success_rate'))} | {_fmt_num(row.get('avg_steps'))} | "
            f"{int(_float(row.get('exception_count')))} | "
            f"{_fmt_count(row.get('shortcut_plan_count'))} | {_fmt_count(row.get('shortcut_would_fire_count'))} | "
            f"{_fmt_pct(row.get('shadow_precision'))} | {_fmt_count(row.get('active_shortcut_fired_count'))} | "
            f"{_fmt_count(row.get('skipped_vlm_calls'))} | {_fmt_count(row.get('full_a11y_calls'))} |"
        )
    lines += ["", "## 图表", ""]
    for name, path in figures.items():
        lines.append(f"![{name}]({path})")
        lines.append("")
    missing_shortcut_variants = [
        row["variant"]
        for row in summaries
        if row.get("shortcut_plan_count") == "MISSING"
        or row.get("shortcut_would_fire_count") == "MISSING"
        or row.get("shadow_eval_count") == "MISSING"
    ]
    if missing_shortcut_variants:
        lines += [
            "## Shortcut instrumentation 状态",
            "",
            "Shortcut instrumentation missing or no plans generated.",
            f"- missing variants: {', '.join(missing_shortcut_variants)}",
            "- `MISSING` 表示对应 jsonl/csv 文件没有写出；不能解释为 0。",
            "",
        ]

    lines += [
        "## 关键问题回答",
        "",
        f"0. 本轮是否有效: {'否' if any(_float(r.get('exception_count')) > 0 for r in summaries if r.get('variant') != 'V0_BASELINE') else '是'}。V12-V19 的 exception_count 用来判断是否能作为 shortcut 结论。",
        f"1. aggressive t+2 是否真的运行: {'是' if any(_float(r.get('shortcut_plan_count')) > 0 for r in summaries) else '否'}。判据是 `shortcut_plans.jsonl` 非空，而不是 success rate。",
        f"2. ShortcutPlan 数量: {sum(int(_float(r.get('shortcut_plan_count'))) for r in summaries)}，其中 V16={int(_float(shadow.get('shortcut_plan_count')))}。",
        f"3. shadow would_fire 数量: {int(_float(shadow.get('shortcut_would_fire_count')))}。",
        f"4. shadow precision: {_fmt_pct(shadow.get('shadow_precision'))}。",
        f"5. active fired 数量: {sum(int(_float(r.get('active_shortcut_fired_count'))) for r in active_candidates)}。",
        f"6. skipped VLM calls: {sum(int(_float(r.get('skipped_vlm_calls'))) for r in active_candidates)}。",
        f"7. active shortcut 是否提高 success: 最佳 active success={_fmt_pct(best_active.get('success_rate'))}，V0={_fmt_pct(baseline.get('success_rate'))}。",
        f"8. active shortcut 是否降低 steps: 最佳 active avg_steps={_fmt_num(best_active.get('avg_steps'))}，V0={_fmt_num(baseline.get('avg_steps'))}。",
        f"9. 是否破坏 baseline-success task: broken={len(broken_by_active)}，rescued={len(rescued_by_active)}。",
        "10. 适合 t+2 的 app/task: 优先看 `per_task_results.csv` 中有 shortcut fired 且 success 的 INFO_QUERY_COUNT/NAVIGATION_SEARCH/ACTIVITY_STATS。",
        "11. 应禁用 shortcut 的 task: DELETE_COMMIT、FORM_CREATE_EDIT、SIMPLE_VERIFY_OPEN 默认禁用或只允许非提交导航。",
        f"12. SafeSearchInputShortcut 是否有效: V18 fired={int(_float(v18.get('active_shortcut_fired_count')))}，success={_fmt_pct(v18.get('success_rate'))}；需要结合 `shortcut_events.jsonl` 看 TYPE 是否命中。",
        f"13. full a11y calls 是否下降: V19 full_a11y={int(_float(v19.get('full_a11y_calls')))}，相对 V0 drop={_fmt_pct(full_a11y_drop) if full_a11y_drop is not None else 'NA'}。",
        f"14. lightweight state 是否影响安全性: V19 success={_fmt_pct(v19.get('success_rate'))}，V18 success={_fmt_pct(v18.get('success_rate'))}，若 V19 更低则 state matching/light verify 有风险。",
        "15. 当前瓶颈判断: 如果 plans=0 是 shortcut selection/coverage；如果 would_fire 低是 state matching/gate；如果 shadow precision 低是 extractor/selector；如果 active fired 低但 shadow 高是 active gate 过严；如果 a11y 不降是 state acquisition 优化没有真正生效。",
        "",
        "## 验收标准",
        "",
        "| condition | pass |",
        "|---|---:|",
    ]
    for key, ok in acceptance.items():
        lines.append(f"| {key} | {'PASS' if ok else 'FAIL'} |")
    if not all(acceptance.values()):
        failed = [key for key, ok in acceptance.items() if not ok]
        lines += [
            "",
            "结论: 不建议跑 full 116。",
            "失败条件: " + "；".join(failed),
            "优先修复建议: 先看 `shortcut_plans.jsonl` 是否为空；若为空修 selection/branch depth，若有 plan 但 would_fire 低修 state matching，若 shadow precision 低修 action matcher/extractor，若 V19 a11y 不降修 lightweight acquisition。",
        ]
    else:
        lines += ["", "结论: 30/40-task 诊断满足验收，可以考虑 full 116。"]

    lines += ["", "## app/task-mode 统计（最佳总体变体）", ""]
    lines.append("| app | tasks | success | avg steps |")
    lines.append("|---|---:|---:|---:|")
    for app, stat in sorted(app_stats.items()):
        lines.append(
            f"| {app} | {stat['tasks']} | {_fmt_pct(stat['success'] / max(1, stat['tasks']))} | {_fmt_num(_mean(stat['steps']))} |"
        )

    rollback_rows = _read_jsonl(root / "rollback_events.jsonl")
    rollback_total = len(rollback_rows)
    rollback_fail = sum(1 for item in rollback_rows if not bool(item.get("success")))
    wait_count = sum(1 for row in (root / "per_step_metrics.csv",) for _ in [])
    lines += [
        "",
        "## rollback / WAIT / suppress",
        "",
        f"- rollback events: {rollback_total}",
        f"- rollback failures: {rollback_fail}",
        "- suppress/WAIT 细节请看 `per_step_metrics.csv` 的 action 与 `rollback_events.jsonl` 的 failure_analysis。",
    ]

    (root / "aggressive_t2_shortcut_diagnostic_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Run root containing V0/V12/V16/V17/V18/V19 variant directories.")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    variants = [_load_variant(root, variant) for variant in VARIANT_ORDER if (root / variant).exists()]
    summaries = [_variant_summary(row) for row in variants]
    task_rows = _task_rows(variants)
    state_rows = _state_metric_rows(variants)

    merged_step_rows: list[dict[str, Any]] = []
    merged_plans: list[dict[str, Any]] = []
    merged_events: list[dict[str, Any]] = []
    merged_shadow: list[dict[str, Any]] = []
    merged_direct: list[dict[str, Any]] = []
    merged_exploration: list[dict[str, Any]] = []
    merged_rollback: list[dict[str, Any]] = []
    for variant in variants:
        name = variant["variant"]
        for row in variant.get("step_metrics") or []:
            merged_step_rows.append({"variant": name, **row})
        for target, rows in (
            (merged_plans, variant.get("plans") or []),
            (merged_events, variant.get("events") or []),
            (merged_shadow, variant.get("shadow") or []),
            (merged_direct, variant.get("direct") or []),
            (merged_exploration, variant.get("exploration") or []),
            (merged_rollback, variant.get("rollback") or []),
        ):
            for row in rows:
                target.append({"variant": name, **row})

    _write_csv(root / "variant_summary.csv", summaries)
    _write_csv(root / "per_task_results.csv", task_rows)
    _write_csv(root / "per_step_metrics.csv", merged_step_rows)
    _write_csv(root / "state_acquisition_metrics.csv", state_rows)
    _write_jsonl(root / "shortcut_plans.jsonl", merged_plans)
    _write_jsonl(root / "shortcut_events.jsonl", merged_events)
    _write_jsonl(root / "shortcut_shadow_eval.jsonl", merged_shadow)
    _write_jsonl(root / "direct_answer_candidates.jsonl", merged_direct)
    _write_jsonl(root / "exploration_page_trace.jsonl", merged_exploration)
    _write_jsonl(root / "rollback_events.jsonl", merged_rollback)
    figures = _make_figures(root, summaries)
    _write_report(root, summaries, task_rows, figures)
    print(f"[aggregate] wrote {root / 'aggressive_t2_shortcut_diagnostic_cn.md'}")


if __name__ == "__main__":
    main()
