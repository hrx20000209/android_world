#!/usr/bin/env python3
"""Aggregate frozen aggressive t+2 shortcut experiments and generate CN report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VARIANTS = [
    "V0_BASELINE",
    "V_MAIN_CERTAINTY_EXPLORATION",
    "V_NO_PROMPT_INJECTION",
    "V_T2_SHADOW_ONLY",
    "V_LIGHT_STATE_DIAG",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\\n")


def _safe_float(v: Any, default: float | None = 0.0) -> float | None:
    if v is None:
        return default
    if isinstance(v, str) and v.strip() in {"", "nan", "None", "NA"}:
        return default
    try:
        out = float(v)
    except Exception:
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _safe_mean(values: list[float]) -> float | None:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None
    return float(statistics.mean(finite))


def _canonical_task_key(text: str) -> str:
    low = str(text or "").lower()
    if "sailing activity" in low and "october" in low:
        return "SportsTrackerActivityDuration"
    for key in [
        "sportstrackeractivityduration",
        "sportstrackerlongestdistanceactivity",
        "sportstrackertotaldistanceforcategoryoverinterval",
        "sportstrackeractivitiesondate",
        "sportstrackeractivitiescountforweek",
    ]:
        if key in low:
            return "".join(w.capitalize() for w in key.split("_")) if "_" in key else "".join(p.title() for p in key.split())
    for key in [
        "simplecalendar", "calendar", "next meeting", "events on", "events in time", "next event",
    ]:
        if key in low:
            if "next meeting" in low:
                return "SimpleCalendarNextMeetingWithPerson"
            if "next event" in low:
                return "SimpleCalendarNextEvent"
            if "in time" in low or "time range" in low:
                return "SimpleCalendarEventsInTimeRange"
            if "any events on" in low:
                return "SimpleCalendarAnyEventsOnDate"
            return "SimpleCalendarEventsOnDate"
    if "notes recipe ingredient" in low or "recipe ingredient" in low:
        return "NotesRecipeIngredientCount"
    if "meeting attendee" in low:
        return "NotesMeetingAttendeeCount"
    if "todo" in low and ("count" in low or "many" in low):
        return "NotesTodoItemCount"
    if low.startswith("notes") and "todo" in low:
        return "NotesIsTodo"
    if "opentracks" in low or "activity" in low:
        for t in [
            "SportsTrackerActivityDuration",
            "SportsTrackerTotalDistanceForCategoryOverInterval",
            "SportsTrackerActivitiesOnDate",
            "SportsTrackerActivitiesCountForWeek",
            "SportsTrackerLongestDistanceActivity",
        ]:
            if t.lower() in low:
                return t
    if "task" in low and "due next week" in low:
        return "TasksDueNextWeek"
    if "due on" in low and ("october" in low or "date" in low):
        return "TasksDueOnDate"
    if "high priority" in low and "due" in low:
        return "TasksHighPriorityTasksDueOnDate"
    if "high priority" in low:
        return "TasksHighPriorityTasks"
    for app_prefix, task_id in [
        ("browser maze", "BrowserMaze"),
        ("browser multiply", "BrowserMultiply"),
        ("browser draw", "BrowserDraw"),
        ("delete file", "FilesDeleteFile"),
        ("move file", "FilesMoveFile"),
        ("create folder", "MarkorCreateFolder"),
        ("create note", "MarkorCreateNote"),
        ("delete newest note", "MarkorDeleteNewestNote"),
        ("delete note", "MarkorDeleteNote"),
        ("edit note", "MarkorEditNote"),
        ("add single expense", "ExpenseAddSingle"),
        ("expense add", "ExpenseAddMultiple"),
        ("delete expense", "ExpenseDeleteSingle"),
        ("delete duplicate", "ExpenseDeleteDuplicates"),
        ("stopwatch", "ClockStopWatchRunning"),
        ("timer", "ClockTimerEntry"),
        ("add contact", "ContactsAddContact"),
        ("contact draft", "ContactsNewContactDraft"),
        ("audio recorder", "AudioRecorderRecordAudioWithFileName"),
        ("wifi", "SystemWifiTurnOnVerify"),
        ("bluetooth", "SystemBluetoothTurnOnVerify"),
    ]:
        if app_prefix in low:
            return task_id
    return str(text or "").strip()


def _latest_run_dir(variant_dir: Path) -> Path | None:
    runs = [p for p in variant_dir.glob("run_*") if p.is_dir()]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _collect_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        norm = str(path.resolve())
        if norm in seen:
            continue
        seen.add(norm)
        rows.extend(_read_jsonl(path))
    return rows


def _load_trace_files(run_dir: Path) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    trace_root = run_dir / "traces"
    plan_paths = []
    event_paths = []
    shadow_paths = []
    direct_paths = []
    state_paths = []
    if not trace_root.exists():
        return plan_paths, event_paths, shadow_paths, direct_paths
    for path in sorted(trace_root.rglob("*.jsonl")):
        name = path.name.lower()
        if "shortcut_plans" in name or "plan" in name and "shortcut" in name:
            plan_paths.append(path)
        if "shortcut_events" in name:
            event_paths.append(path)
        if "shortcut_shadow" in name:
            shadow_paths.append(path)
        if "direct_answer" in name:
            direct_paths.append(path)
        if "state_acquisition" in name:
            state_paths.append(path)
    return plan_paths, event_paths, shadow_paths, direct_paths


def _load_variant(root: Path, variant: str) -> dict[str, Any]:
    variant_dir = root / variant
    run_dir = _latest_run_dir(variant_dir)
    if run_dir is None:
        return {
            "variant": variant,
            "variant_dir": variant_dir,
            "run_dir": None,
            "summary": {},
        }
    summary = _read_json(run_dir / "report" / "summary.json")
    step_metrics = _read_csv(run_dir / "report" / "step_metrics.csv")
    for row in step_metrics:
        row["variant"] = variant
    state_file = run_dir / "traces" / "state_acquisition_metrics.csv"
    if not state_file.exists():
        fallback = sorted((run_dir / "traces").glob("*state_acquisition_metrics.csv"))
        if fallback:
            state_file = fallback[0]
    state_rows = _read_csv(state_file)
    for row in state_rows:
        row["variant"] = variant
        row["task"] = variant
    plan_paths, event_paths, shadow_paths, direct_paths = _load_trace_files(run_dir)
    plans = _collect_jsonl(plan_paths)
    events = _collect_jsonl(event_paths)
    shadow = _collect_jsonl(shadow_paths)
    direct = _collect_jsonl(direct_paths)
    rollbacks = []
    for rb in summary.get("_rollbacks") or []:
        if isinstance(rb, dict):
            row = dict(rb)
            row["variant"] = variant
            rollbacks.append(row)
    # Some runs keep rollback in jsonl
    if not rollbacks:
        for path in sorted(run_dir.rglob("rollback_trace.jsonl")):
            for row in _read_jsonl(path):
                row["variant"] = variant
                rollbacks.append(row)
    match_paths = [p for p in (run_dir / "traces").rglob("exploration_match_trace.jsonl")]
    matches = _collect_jsonl(sorted(match_paths))
    action_paths = [p for p in (run_dir / "traces").rglob("action.jsonl")]
    actions = _collect_jsonl(sorted(action_paths))
    return {
        "variant": variant,
        "variant_dir": variant_dir,
        "run_dir": run_dir,
        "summary": summary,
        "step_metrics": step_metrics,
        "plans": plans,
        "events": events,
        "shadow": shadow,
        "direct": direct,
        "matches": matches,
        "actions": actions,
        "rollbacks": rollbacks,
        "state": state_rows,
        "plan_paths": plan_paths,
        "event_paths": event_paths,
        "shadow_paths": shadow_paths,
        "direct_paths": direct_paths,
    }


def _build_summary_rows(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in variants:
        summary = item.get("summary") or {}
        base = summary.get("baseline_compare") or {}
        episodes = summary.get("episodes") or {}
        actions = summary.get("actions") or {}
        exploration = summary.get("exploration") or {}
        rollback = summary.get("rollback") or {}
        shortcut = summary.get("shortcut") or {}
        state_acq = summary.get("state_acquisition") or {}
        plans = item.get("plans") or []
        events = item.get("events") or []
        shadow = item.get("shadow") or []
        step_rows = item.get("step_metrics") or []
        would_fire = [x for x in events if x.get("would_fire") or x.get("shortcut_would_fire")]
        fired = [x for x in events if x.get("fired") or x.get("shortcut_fired")]
        rows.append(
            {
                "variant": item["variant"],
                "task_count": len(episodes.get("task_episode_rows") or []),
                "success_rate": episodes.get("success_rate_all_trials"),
                "avg_steps": episodes.get("avg_episode_length_complete"),
                "avg_episode_length_complete": episodes.get("avg_episode_length_complete"),
                "avg_episode_length_success": episodes.get("avg_episode_length_success"),
                "baseline_success_rate": base.get("avg_baseline_success_rate"),
                "baseline_steps": base.get("avg_baseline_episode_length"),
                "exploration_traces": exploration.get("total_traces"),
                "exploration_completed": exploration.get("completed"),
                "available_for_next_prompt": exploration.get("available_for_next_prompt"),
                "available_rate": exploration.get("available_rate"),
                "candidate_count": shortcut.get("shortcut_plan_count", len(plans)),
                "candidate_with_action": shortcut.get(
                    "shortcut_candidate_count",
                    len([p for p in plans if not str(p.get("no_plan_reason") or "").strip()]),
                ),
                "would_fire_count": shortcut.get("shortcut_would_fire_count", len(would_fire)),
                "fired_count": shortcut.get("active_shortcut_fired_count", len(fired)),
                "shadow_count": shortcut.get("shortcut_shadow_eval_count", len(shadow)),
                "shadow_action_match_count": shortcut.get("shortcut_shadow_action_match_count"),
                "shadow_precision": shortcut.get("shortcut_shadow_precision"),
                "skipped_vlm_calls": shortcut.get(
                    "skipped_vlm_calls",
                    len([r for r in step_rows if str(r.get("skipped_vlm_call") or "").lower() in {"true", "1"}]),
                ),
                "rollback_total": rollback.get("total") or len(item.get("rollbacks")),
                "rollback_success": rollback.get("success") if rollback else sum(1 for r in item.get("rollbacks") or [] if r.get("success")),
                "rollback_failures": rollback.get("failures") if rollback else sum(1 for r in item.get("rollbacks") or [] if not r.get("success")),
                "full_a11y_calls": state_acq.get("full_a11y_calls"),
                "screenshot_calls": state_acq.get("screenshot_calls"),
                "activity_calls": state_acq.get("activity_calls"),
                "root_state_reuse_count": state_acq.get("root_state_reuse_count"),
                "passive_no_get_state_count": state_acq.get("passive_no_get_state_count"),
                "rollback_light_verify_count": state_acq.get("rollback_light_verify_count"),
                "rollback_full_a11y_fallback_count": state_acq.get("rollback_full_a11y_fallback_count"),
                "cached_state_hits": state_acq.get("cached_state_hits"),
            }
        )
    return rows


def _load_task_meta(run_root: Path) -> dict[str, dict[str, Any]]:
    task_meta_file = run_root / "task_selection.json"
    rows: dict[str, dict[str, Any]] = {}
    if task_meta_file.exists():
        for item in _read_json(task_meta_file):
            rows[item.get("task_id")] = dict(item)
    return rows


def _build_task_rows(variants: list[dict[str, Any]], task_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, float]] = defaultdict(dict)
    base_rows_by_task: dict[tuple[str, str], dict[str, Any]] = {}
    for var in variants:
        summary = var.get("summary") or {}
        baseline_rows = {str(x.get("task") or ""): x for x in summary.get("baseline_compare", {}).get("rows", [])}
        for episode in summary.get("episodes", {}).get("task_episode_rows", []):
            task = str(episode.get("task") or "")
            key = (var["variant"], task)
            baseline = baseline_rows.get(task, {})
            base = task_meta.get(task, {})
            by_task[key] = {
                "variant": var["variant"],
                "task_id": task,
                "app": base.get("app", "UNKNOWN"),
                "task_family": base.get("task_family", base.get("app", "UNKNOWN")),
                "task_mode": base.get("task_mode", ""),
                "task_subtype": base.get("task_subtype", ""),
                "baseline_success": baseline.get("baseline_success_rate"),
                "baseline_steps": baseline.get("baseline_episode_length"),
                "variant_success": 1.0 if episode.get("success") else 0.0,
                "variant_steps": episode.get("episode_length"),
                "baseline_success_bool": base.get("baseline_success", 0.0) or baseline.get("baseline_success_rate", 0.0) or 0.0,
                "variant_success_bool": 1.0 if episode.get("success") else 0.0,
                "exception": episode.get("exception"),
            }

    # Add rescued/broken flags and aggregate candidate / shortcut counters per task with weak key matching.
    plan_events_map: dict[str, int] = defaultdict(int)
    would_map: dict[str, int] = defaultdict(int)
    fired_map: dict[str, int] = defaultdict(int)
    for var in variants:
        for plan in var.get("plans") or []:
            k = _canonical_task_key(plan.get("task_id") or "")
            plan_events_map[f"{var['variant']}::{k}"] += 1
        for e in var.get("events") or []:
            if str(e.get("event_type") or e.get("type") or "").lower() == "shortcut_plan":
                k = _canonical_task_key(e.get("task_id") or e.get("goal") or "")
                plan_events_map[f"{var['variant']}::{k}"] += 1
            if e.get("would_fire") or e.get("shortcut_would_fire"):
                k = _canonical_task_key(e.get("task_id") or e.get("goal") or "")
                would_map[f"{var['variant']}::{k}"] += 1
            if e.get("fired") or e.get("shortcut_fired"):
                k = _canonical_task_key(e.get("task_id") or e.get("goal") or "")
                fired_map[f"{var['variant']}::{k}"] += 1

    rows: list[dict[str, Any]] = []
    for key, row in by_task.items():
        variant, task = key
        row = dict(row)
        row["shortcut_plan_count"] = plan_events_map.get(f"{variant}::{task}", 0)
        row["would_fire_count"] = would_map.get(f"{variant}::{task}", 0)
        row["shortcut_fired_count"] = fired_map.get(f"{variant}::{task}", 0)
        baseline_success = _safe_float(row.get("baseline_success"), 0.0) or 0.0
        variant_success = _safe_float(row.get("variant_success"), 0.0) or 0.0
        row["rescued"] = int(1 if (baseline_success < 0.5 and variant_success >= 0.5) else 0)
        row["broken"] = int(1 if (baseline_success >= 0.5 and variant_success < 0.5) else 0)
        row["explored_pages"] = 0
        row["selected_pages"] = 0
        row["slot_complete_count"] = 0
        rows.append(row)
    return rows


def _build_exploration_trace(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for var in variants:
        run_dir = var.get("run_dir")
        if run_dir is None:
            continue
        for p in sorted((run_dir / "traces").rglob("exploration_trace.jsonl")):
            for row in _read_jsonl(p):
                record = dict(row)
                record["variant"] = var["variant"]
                record["_trace_file"] = str(p)
                rows.append(record)
    return rows


def _collect_action_artifacts(variants: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    per_step_rows: list[dict[str, Any]] = []
    for var in variants:
        for row in var.get("actions") or []:
            variant = var["variant"]
            step = row.get("step")
            task = row.get("goal") or row.get("_goal") or row.get("task_id") or ""
            for cap in row.get("exploration_selected_prompt_results") or []:
                if not isinstance(cap, dict):
                    continue
                payload = {
                    "variant": variant,
                    "task": task,
                    "step": step,
                    "hint_type": cap.get("hint_type") or cap.get("hint_kind"),
                    "evidence_type": cap.get("evidence_type"),
                    "score": cap.get("score"),
                    "confidence": cap.get("confidence"),
                    "evidence_depth": cap.get("evidence_depth"),
                    "source_step": cap.get("source_step"),
                    "depth_reached": cap.get("depth_reached"),
                    "slot_coverage": ((cap.get("slot_evidence") or {}).get("slot_coverage")),
                    "slot_complete": ((cap.get("slot_evidence") or {}).get("slot_complete")),
                    "missing_slots": ((cap.get("slot_evidence") or {}).get("missing_slots")),
                    "facts": (cap.get("slot_evidence") or {}).get("facts"),
                }
                evidence_rows.append(payload)
                if (cap.get("hint_type") or "").upper() == "ANSWER_HINT":
                    payload = dict(payload)
                    payload["prompt_line"] = cap.get("prompt_line")
                    answer_rows.append(payload)
            if row.get("shortcut_event"):
                row = dict(row)
                row["variant"] = variant
                per_step_rows.append(row)
    # Keep step metrics as light output for report compatibility
    return evidence_rows, answer_rows, per_step_rows


def _fmt_rate(v: float | None) -> str:
    if v is None:
        return "n/a"
    if math.isnan(v):
        return "n/a"
    return f"{v:.1%}"


def _fmt_num(v: Any, digits: int = 2) -> str:
    value = _safe_float(v, None)
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _write_report(root: Path, summary_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]]) -> None:
    by_variant = {r["variant"]: r for r in summary_rows}
    baseline = by_variant.get("V0_BASELINE", {})
    baseline_success = _safe_float(baseline.get("success_rate"), None)

    best_active_name = None
    best_active_val = -1.0
    for name in ("V_NO_PROMPT_INJECTION", "V_T2_SHADOW_ONLY", "V_MAIN_CERTAINTY_EXPLORATION"):
        if name in by_variant:
            val = _safe_float(by_variant[name].get("success_rate"), 0.0) or 0.0
            if val > best_active_val:
                best_active_val = val
                best_active_name = name

    best_active = by_variant.get(best_active_name, {})
    shadow = by_variant.get("V_T2_SHADOW_ONLY", {})
    would_fire = _safe_float(shadow.get("would_fire_count"), 0.0) or 0.0
    shadow_matches = _safe_float(shadow.get("shadow_action_match_count"), 0.0) or 0.0
    shadow_precision = _safe_float(shadow.get("shadow_precision"), None)
    if shadow_precision is None and would_fire:
        shadow_precision = shadow_matches / would_fire
    main = by_variant.get("V_MAIN_CERTAINTY_EXPLORATION", {})
    no_prompt = by_variant.get("V_NO_PROMPT_INJECTION", {})
    light = by_variant.get("V_LIGHT_STATE_DIAG", {})
    main_success = _safe_float(main.get("success_rate"), 0.0) or 0.0
    no_prompt_success = _safe_float(no_prompt.get("success_rate"), 0.0) or 0.0
    main_rollbacks = _safe_float(main.get("rollback_total"), 0.0) or 0.0
    main_rb_fail = _safe_float(main.get("rollback_failures"), 0.0) or 0.0
    main_a11y = _safe_float(main.get("full_a11y_calls"), None)
    light_a11y = _safe_float(light.get("full_a11y_calls"), None)
    if main_a11y and light_a11y is not None:
        a11y_reduction = (main_a11y - light_a11y) / main_a11y
    else:
        a11y_reduction = None

    lines: list[str] = [
        "# MobileExplorer 最终冻结设计诊断报告",
        "",
        f"- 任务行数: {len(task_rows)} (按 task×variant 展开)",
        f"- 运行目录: `{root}`",
        "- 主设计 active t+2: OFF",
        "",
        "## 冻结配置",
        "",
        "`final_conservative_design_config.yaml` 是本次有效性判断的配置源；如果实际 log 与该文件不一致，应视为 invalid run。",
        "",
        "## 变体汇总",
        "",
        "| variant | success_rate | avg_steps | baseline_success | ShortcutPlan | would_fire | shadow_precision | active_fired | skipped_vlm | rollback_total | rollback_failures | full_a11y_calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        precision = _safe_float(row.get("shadow_precision"), None)
        would_fire_value = _safe_float(row.get("would_fire_count"), 0.0) or 0.0
        fired_value = _safe_float(row.get("fired_count"), 0.0) or 0.0
        skipped_value = _safe_float(row.get("skipped_vlm_calls"), 0.0) or 0.0
        lines.append(
            f"| {row['variant']} | {_fmt_rate(_safe_float(row.get('success_rate')))} | "
            f"{_fmt_num(row.get('avg_steps'))} | "
            f"{_fmt_rate(_safe_float(row.get('baseline_success_rate')))} | "
            f"{int(_safe_float(row.get('candidate_count')) or 0)} | {would_fire_value:.0f} | "
            f"{_fmt_rate(precision)} | "
            f"{fired_value:.0f} | "
            f"{skipped_value:.0f} | "
            f"{int(_safe_float(row.get('rollback_total')) or 0)} | {_safe_float(row.get('rollback_failures')) or 0} | "
            f"{_fmt_num(row.get('full_a11y_calls'), 0)} |"
        )

    lines.extend(
        [
            "",
            "## 关键问题答案",
            f"1. 是否应该冻结保守主设计：{'是' if main_success >= (baseline_success or 0.0) and main_rb_fail <= max(1.0, main_rollbacks * 0.05) else '否，需要继续修'}。",
            f"2. aggressive t+2 定位：主实验 active OFF；shadow would_fire={int(would_fire)}，precision={_fmt_rate(shadow_precision)}，因此只作为 optional/future work 诊断。",
            f"3. V_MAIN 是否超过或持平 baseline：baseline={_fmt_rate(baseline_success)}，V_MAIN={_fmt_rate(main_success)}，Δ={main_success - (baseline_success or 0.0):.3f}。",
            f"4. V_NO_PROMPT_INJECTION 与 V_MAIN 差异：V_NO_PROMPT={_fmt_rate(no_prompt_success)}，V_MAIN={_fmt_rate(main_success)}，Δ={main_success - no_prompt_success:.3f}。",
            "5. prompt injection 是否真正发生：见 `evidence_capsules.jsonl` 和 `answer_hint_usage.jsonl`；若 V_MAIN 与 V_NO_PROMPT 接近，说明收益主要来自探索副作用或样本噪声。",
            f"6. t+2 evidence 是否产生：V_T2_SHADOW_ONLY ShortcutPlan={int(_safe_float(shadow.get('candidate_count')) or 0)}，would_fire={int(would_fire)}。",
            f"7. rollback 是否安全：V_MAIN rollback_total={int(main_rollbacks)}，failures={int(main_rb_fail)}。",
            "8. level2 rollback 何时发生：通常是一次 Back 无法回到 root，或 activity/pHash/anchor 未匹配时进入 replay/back 多步恢复；具体见 `rollback_events.jsonl`。",
            "9. latency bottleneck：仍应优先看 full_a11y_calls 和 state_acquisition_metrics；此前单任务 profile 已显示 get_state/a11y 是主瓶颈。",
            f"10. V_LIGHT_STATE_DIAG 是否减少 full a11y：V_MAIN={_fmt_num(main_a11y, 0)}，V_LIGHT={_fmt_num(light_a11y, 0)}，reduction={_fmt_rate(a11y_reduction)}。",
            "11. 哪些 app/task family 受益：见下方 task-level rows 和 `per_task_results.csv`。",
            "12. 哪些 app/task family 被伤害：看 `broken=1` 的行，尤其 DELETE/FORM/SIMPLE_VERIFY。",
            "13. 是否建议跑 full 116：只有当 V_MAIN >= baseline、rescued >= broken、rollback WAIT 低、light-state 不降成功率时才建议。",
            "14. 如果不跑 full 116：论文应表述为 frozen diagnostic，强调安全 rollback/evidence injection 的机制验证，同时把 aggressive t+2 放入 optional optimization/limitation。",
            "",
            "## 任务级示例（前 30 条）",
            "| variant | task_id | app | task_mode | baseline_success | variant_success | rescued | broken | shortcut_plan_count | would_fire_count | fired_count |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in task_rows[:30]:
        lines.append(
            f"| {row['variant']} | {row['task_id']} | {row['app']} | {row.get('task_mode')} | "
            f"{_fmt_rate(_safe_float(row.get('baseline_success')))} | {_fmt_rate(_safe_float(row.get('variant_success')))} | "
            f"{int(row.get('rescued') or 0)} | {int(row.get('broken') or 0)} | "
            f"{int(row.get('shortcut_plan_count') or 0)} | {int(row.get('would_fire_count') or 0)} | {int(row.get('shortcut_fired_count') or 0)} |"
        )
    report = "\\n".join(lines) + "\\n"
    (root / "final_design_decision_report_cn.md").write_text(report, encoding="utf-8")
    (root / "aggressive_t2_shortcut_diagnostic_cn.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    variants = [_load_variant(root, v) for v in VARIANTS]
    variants = [v for v in variants if v.get("run_dir") is not None]
    if not variants:
        raise RuntimeError(f"No variant runs found under {root}")

    task_meta = _load_task_meta(root)
    summary_rows = _build_summary_rows(variants)
    task_rows = _build_task_rows(variants, task_meta)
    per_step_rows: list[dict[str, Any]] = []
    for var in variants:
        per_step_rows.extend(var.get("step_metrics") or [])
        for rb in var.get("rollbacks") or []:
            rb["variant"] = var["variant"]
    evidence_rows, answer_rows, _ = _collect_action_artifacts(variants)
    exploration_rows = _build_exploration_trace(variants)

    _write_csv(root / "variant_summary.csv", summary_rows)
    _write_csv(root / "per_task_results.csv", task_rows)
    _write_csv(root / "per_step_metrics.csv", per_step_rows)
    state_rows: list[dict[str, Any]] = []
    for var in variants:
        for row in var.get("state") or []:
            row = dict(row)
            state_rows.append(row)
    _write_csv(root / "state_acquisition_metrics.csv", state_rows)

    all_plans: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    all_shadow: list[dict[str, Any]] = []
    all_direct: list[dict[str, Any]] = []
    all_rollbacks: list[dict[str, Any]] = []
    for var in variants:
        all_plans.extend([{**r, "variant": var["variant"]} for r in (var.get("plans") or [])])
        all_events.extend([{**r, "variant": var["variant"]} for r in (var.get("events") or [])])
        all_shadow.extend([{**r, "variant": var["variant"]} for r in (var.get("shadow") or [])])
        all_direct.extend([{**r, "variant": var["variant"]} for r in (var.get("direct") or [])])
        all_rollbacks.extend([{**r, "variant": var["variant"]} for r in (var.get("rollbacks") or [])])

    _write_jsonl(root / "shortcut_plans.jsonl", all_plans)
    _write_jsonl(root / "shortcut_events.jsonl", all_events)
    _write_jsonl(root / "shortcut_shadow_eval.jsonl", all_shadow)
    _write_jsonl(root / "direct_answer_candidates.jsonl", all_direct)
    _write_jsonl(root / "exploration_page_trace.jsonl", exploration_rows)
    _write_jsonl(root / "rollback_events.jsonl", all_rollbacks)
    _write_jsonl(root / "evidence_capsules.jsonl", evidence_rows)
    _write_jsonl(root / "answer_hint_usage.jsonl", answer_rows)

    _write_report(root, summary_rows, task_rows)


if __name__ == "__main__":
    main()
