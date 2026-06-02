#!/usr/bin/env python3
"""Merge two-machine frozen 30-task strategy comparison results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

VARIANTS = [
    "B0_BASELINE_RERUN",
    "S1_BFS_BUDGET12",
    "S2_DFS_BUDGET12",
    "S3_BEAM_BUDGET12",
    "S4_MCTS_BUDGET12",
    "S5_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12",
]

JSONL_FILES = [
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
    "shortcut_plans.jsonl",
    "shortcut_shadow_eval.jsonl",
    "step_decoupling_status.jsonl",
]

CSV_FILES = [
    "per_task_results.csv",
    "per_step_metrics.csv",
    "state_acquisition_metrics.csv",
    "candidate_scores.csv",
]

TASK_META: dict[str, dict[str, str]] = {
    "NotesRecipeIngredientCount": {"app": "Joplin", "task_mode": "answer_entity", "shard": "A"},
    "NotesIsTodo": {"app": "Joplin", "task_mode": "answer_boolean", "shard": "A"},
    "FilesDeleteFile": {"app": "Files", "task_mode": "delete_boundary", "shard": "A"},
    "FilesMoveFile": {"app": "Files", "task_mode": "navigation_mutation", "shard": "A"},
    "SportsTrackerActivityDuration": {"app": "OpenTracks", "task_mode": "answer_stats", "shard": "A"},
    "SportsTrackerTotalDistanceForCategoryOverInterval": {"app": "OpenTracks", "task_mode": "answer_stats", "shard": "A"},
    "SimpleCalendarNextMeetingWithPerson": {"app": "Calendar", "task_mode": "answer_entity", "shard": "A"},
    "SimpleCalendarEventsInTimeRange": {"app": "Calendar", "task_mode": "answer_entity", "shard": "A"},
    "ClockStopWatchRunning": {"app": "Clock", "task_mode": "state_verify", "shard": "A"},
    "ClockTimerEntry": {"app": "Clock", "task_mode": "state_verify", "shard": "A"},
    "MarkorCreateNote": {"app": "Markor", "task_mode": "create_note", "shard": "A"},
    "MarkorEditNote": {"app": "Markor", "task_mode": "edit_note", "shard": "A"},
    "MarkorCreateFolder": {"app": "Markor", "task_mode": "create_folder", "shard": "A"},
    "ExpenseDeleteSingle": {"app": "Expense", "task_mode": "delete_boundary", "shard": "A"},
    "ExpenseDeleteMultiple2": {"app": "Expense", "task_mode": "delete_boundary", "shard": "A"},
    "ExpenseAddSingle": {"app": "Expense", "task_mode": "create_record", "shard": "B"},
    "BrowserMaze": {"app": "Browser", "task_mode": "web_interaction", "shard": "B"},
    "BrowserMultiply": {"app": "Browser", "task_mode": "web_interaction", "shard": "B"},
    "BrowserDraw": {"app": "Browser", "task_mode": "web_interaction", "shard": "B"},
    "ContactsNewContactDraft": {"app": "Contacts", "task_mode": "draft_data_entry", "shard": "B"},
    "ContactsAddContact": {"app": "Contacts", "task_mode": "data_entry", "shard": "B"},
    "AudioRecorderRecordAudio": {"app": "AudioRecorder", "task_mode": "record_audio", "shard": "B"},
    "AudioRecorderRecordAudioWithFileName": {"app": "AudioRecorder", "task_mode": "record_audio", "shard": "B"},
    "SystemWifiTurnOnVerify": {"app": "System", "task_mode": "state_verify", "shard": "B"},
    "SystemWifiTurnOffVerify": {"app": "System", "task_mode": "state_verify", "shard": "B"},
    "SystemBrightnessMinVerify": {"app": "System", "task_mode": "state_verify", "shard": "B"},
    "SimpleDrawProCreateDrawing": {"app": "SimpleDraw", "task_mode": "create_drawing", "shard": "B"},
    "CameraTakePhoto": {"app": "Camera", "task_mode": "capture_photo", "shard": "B"},
    "OsmAndFavorite": {"app": "OsmAnd", "task_mode": "map_marker", "shard": "B"},
    "OsmAndMarker": {"app": "OsmAnd", "task_mode": "map_marker", "shard": "B"},
}

STRATEGY_TEXT = {
    "B0_BASELINE_RERUN": "不做 exploration；每一步只使用主循环 VLM 的当前 UI state 和历史。",
    "S1_BFS_BUDGET12": "BFS 按 depth 分层展开：先覆盖 depth1 的 safe candidate，再考虑 depth2；目标是最大化浅层页面覆盖，不追单一路径。每 step 至多/至少尝试 12 个 safe candidates，遇到 answer/target/schema/shortcut/negative/risk/dead-end/budget_exhausted/rollback_failed 停止 branch。",
    "S2_DFS_BUDGET12": "DFS 选择一个 safe candidate 后尽快沿该分支深入；只有 depth1 semantic changed 且 rollback 可控时进入 depth2。目标是快速验证一条可能通往信息页的路径，预算耗尽后回溯换分支。",
    "S3_BEAM_BUDGET12": "Beam 保留有限 frontier；每轮只展开当前 frontier 中最可能产生新 UI/目标实体的若干分支。目标是在 BFS 覆盖和 DFS 深入之间折中，避免早期单路径误导。",
    "S4_MCTS_BUDGET12": "Safe-MCTS 在 operator/candidate frontier 上做受限试探；只允许 safe click/search 类动作，风险/破坏性动作不进入 rollout。目标是把预算集中到历史上更可能产生 semantic change 的操作族。",
    "S5_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12": "Pattern-aware operator best-first 先判断任务模式和 app，再偏向 SearchPeek、FilterPeek、ListInspect、DetailPeek、StatsPeek 等操作族；候选选择目标是补齐任务缺失 slot、找到 exact target/entity、避免 rollback 高风险，而不是单纯追分。预算 12，depth2 只在 semantic changed、safe、rollback controllable 时展开。",
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        try:
            return [dict(row) for row in csv.DictReader(f) if any((v or "").strip() for v in row.values())]
        except csv.Error:
            return []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "success", "succeeded", "passed", "pass"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _task_name(row: dict[str, Any]) -> str:
    for key in ("task", "task_name", "task_id", "goal"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return "unknown"


def _success(row: dict[str, Any]) -> bool:
    for key in ("success", "succeeded", "is_success", "pass", "passed"):
        if key in row:
            return _truthy(row.get(key))
    return False


def _steps(row: dict[str, Any]) -> float:
    for key in ("steps", "step_count", "num_steps", "total_steps"):
        if key in row:
            return _num(row.get(key))
    return 0.0


def _latency(row: dict[str, Any]) -> float:
    for key in ("latency_s", "total_latency_s", "elapsed_s", "duration_s"):
        if key in row:
            return _num(row.get(key))
    for key in ("latency_ms", "total_latency_ms", "elapsed_ms"):
        if key in row:
            return _num(row.get(key)) / 1000.0
    return 0.0


def _variant_dirs(run_group: Path) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    for machine_dir in sorted(run_group.glob("machine_*")):
        if not machine_dir.is_dir():
            continue
        machine_id = machine_dir.name.replace("machine_", "")
        for variant in VARIANTS:
            variant_dir = machine_dir / variant
            if variant_dir.exists():
                out.append((machine_id, variant, variant_dir))
    return out


def _enrich(row: dict[str, Any], machine_id: str, variant: str) -> dict[str, Any]:
    row = dict(row)
    row.setdefault("machine_id", machine_id)
    row.setdefault("variant", variant)
    task = _task_name(row)
    meta = TASK_META.get(task, {})
    row.setdefault("task", task)
    row.setdefault("app", meta.get("app", ""))
    row.setdefault("task_mode", meta.get("task_mode", ""))
    row.setdefault("shard", meta.get("shard", machine_id))
    return row


def _summarize_variant(variant: str, per_task: list[dict[str, Any]], step_rows: list[dict[str, Any]], json_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    successes = sum(1 for row in per_task if _success(row))
    task_count = len(per_task)
    step_values = [_steps(row) for row in per_task if _steps(row) > 0]
    lat_values = [_latency(row) for row in per_task if _latency(row) > 0]
    branch_rows = json_rows.get("exploration_branch_trace.jsonl", [])
    step_summary = json_rows.get("exploration_step_summary.jsonl", [])
    evidence = json_rows.get("evidence_decisions.jsonl", [])
    hints = json_rows.get("prompt_hints.jsonl", [])
    hit_follow = json_rows.get("hint_hit_follow.jsonl", [])
    rollback = json_rows.get("rollback_events.jsonl", [])
    gate = json_rows.get("rollback_gate_decisions.jsonl", [])
    candidate_scores = json_rows.get("candidate_scores.jsonl", [])
    filters = json_rows.get("candidate_filter_stats.jsonl", [])
    decoupling = json_rows.get("step_decoupling_status.jsonl", [])

    attempts = [_num(row.get("attempt_count") or row.get("exploration_attempts") or row.get("attempts")) for row in step_summary]
    depth2 = sum(1 for row in branch_rows if _num(row.get("depth_reached")) >= 2 or row.get("depth2_action"))
    semantic_changed = sum(1 for row in branch_rows if _truthy(row.get("semantic_changed")))
    rollback_failures = [row for row in rollback if not _truthy(row.get("success", True))]
    rollback_level2 = [row for row in rollback if _truthy(row.get("level2_triggered")) or row.get("rollback_level") == "level2" or row.get("level") == "level2"]
    injected = [row for row in evidence if _truthy(row.get("injected"))]
    rejected = [row for row in evidence if not _truthy(row.get("injected"))]
    action_answer = [row for row in hit_follow if str(row.get("hint_type") or row.get("type") or "").upper() in {"ACTION_HINT", "ANSWER_HINT", "ACTION", "ANSWER"}]
    hit_count = sum(1 for row in hit_follow if _truthy(row.get("hit") or row.get("hint_hit") or row.get("is_hit")))
    follow_count = sum(1 for row in hit_follow if _truthy(row.get("follow") or row.get("followed") or row.get("is_followed")))
    action_answer_follow = sum(1 for row in action_answer if _truthy(row.get("follow") or row.get("followed") or row.get("is_followed")))
    decoupled_invalid = sum(1 for row in decoupling if row.get("decoupled_valid") is not None and not _truthy(row.get("decoupled_valid")))
    contaminated = sum(1 for row in gate + rollback if _truthy(row.get("prompt_history_contaminated")))
    suppressed = sum(1 for row in gate + rollback if _truthy(row.get("main_action_suppressed") or row.get("planned_action_suppressed")))
    active_t2 = sum(1 for row in json_rows.get("shortcut_plans.jsonl", []) if _truthy(row.get("active") or row.get("executed")))

    operator_counter = Counter(str(row.get("operator") or row.get("root_operator") or "unknown") for row in branch_rows if row)
    stop_counter = Counter(str(row.get("stop_reason") or row.get("stop_condition") or "unknown") for row in branch_rows if row)
    selected_reason_counter = Counter(str(row.get("selected_reason") or "unknown") for row in branch_rows if row)
    evidence_counter = Counter(str(row.get("final_evidence_type") or row.get("evidence_type") or "NONE") for row in evidence)
    rejection_counter = Counter(str(row.get("rejected_reason") or "") for row in rejected if row.get("rejected_reason"))
    filter_totals = Counter()
    for row in filters:
        for key in ("filtered_duplicate_count", "filtered_wrong_task_family_count", "filtered_risky_count"):
            filter_totals[key] += int(_num(row.get(key)))

    return {
        "variant": variant,
        "tasks": task_count,
        "successes": successes,
        "success_rate": successes / task_count if task_count else 0.0,
        "avg_steps": sum(step_values) / len(step_values) if step_values else 0.0,
        "avg_task_latency_s": sum(lat_values) / len(lat_values) if lat_values else 0.0,
        "step_rows": len(step_rows),
        "exploration_steps": len(step_summary),
        "avg_attempts_per_exploration_step": sum(attempts) / len(attempts) if attempts else 0.0,
        "branches": len(branch_rows),
        "depth2_branches": depth2,
        "semantic_changed_branches": semantic_changed,
        "candidate_scores": len(candidate_scores),
        "prompt_hints": len(hints),
        "evidence_decisions": len(evidence),
        "evidence_injected": len(injected),
        "evidence_rejected": len(rejected),
        "hint_hit_follow_rows": len(hit_follow),
        "hint_hit_rate": hit_count / len(hit_follow) if hit_follow else 0.0,
        "hint_follow_rate": follow_count / len(hit_follow) if hit_follow else 0.0,
        "action_answer_follow_rate": action_answer_follow / len(action_answer) if action_answer else 0.0,
        "rollback_events": len(rollback),
        "rollback_failures": len(rollback_failures),
        "rollback_level2": len(rollback_level2),
        "rollback_gate_decisions": len(gate),
        "decoupled_invalid_steps": decoupled_invalid,
        "prompt_history_contaminated": contaminated,
        "main_action_suppressed_by_exploration": suppressed,
        "active_t2_events": active_t2,
        "top_operators": "; ".join(f"{k}:{v}" for k, v in operator_counter.most_common(8)),
        "top_stop_reasons": "; ".join(f"{k}:{v}" for k, v in stop_counter.most_common(8)),
        "top_selected_reasons": "; ".join(f"{k}:{v}" for k, v in selected_reason_counter.most_common(8)),
        "evidence_types": "; ".join(f"{k}:{v}" for k, v in evidence_counter.most_common(8)),
        "top_rejection_reasons": "; ".join(f"{k}:{v}" for k, v in rejection_counter.most_common(8)),
        "filtered_duplicate_count": filter_totals["filtered_duplicate_count"],
        "filtered_wrong_task_family_count": filter_totals["filtered_wrong_task_family_count"],
        "filtered_risky_count": filter_totals["filtered_risky_count"],
    }


def _per_app(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("app") or "unknown"), str(row.get("task_mode") or "unknown"))].append(row)
    out: list[dict[str, Any]] = []
    for (app, mode), items in sorted(grouped.items()):
        successes = sum(1 for row in items if _success(row))
        step_values = [_steps(row) for row in items if _steps(row) > 0]
        out.append({
            "variant": variant,
            "app": app,
            "task_mode": mode,
            "tasks": len(items),
            "successes": successes,
            "success_rate": successes / len(items) if items else 0.0,
            "avg_steps": sum(step_values) / len(step_values) if step_values else 0.0,
        })
    return out


def _rollback_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if _truthy(row.get("success", True)):
            continue
        out.append({
            "variant": row.get("variant", ""),
            "machine_id": row.get("machine_id", ""),
            "task": row.get("task") or row.get("task_id") or "",
            "step": row.get("step", ""),
            "branch_id": row.get("branch_id") or row.get("candidate_id") or "",
            "rollback_level": row.get("rollback_level") or row.get("level") or "",
            "rollback_mode": row.get("rollback_mode") or row.get("mode") or "",
            "failure_reasons": row.get("failure_reasons") or row.get("failure_reason") or "",
            "level2_trigger_reason": row.get("level2_trigger_reason") or "",
            "screenshot_path": row.get("screenshot_path") or row.get("screenshot") or "",
            "rollback_timeline_image": row.get("rollback_timeline_image") or "",
            "branch_evidence_discarded": row.get("branch_evidence_discarded", ""),
            "prompt_history_contaminated": row.get("prompt_history_contaminated", ""),
            "main_action_suppressed": row.get("main_action_suppressed", ""),
        })
    return out


def _strategy_selection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("variant", "")), str(row.get("root_operator") or row.get("operator") or "unknown"), str(row.get("stop_reason") or row.get("stop_condition") or "unknown"))].append(row)
    out: list[dict[str, Any]] = []
    for (variant, operator, stop), items in sorted(grouped.items()):
        out.append({
            "variant": variant,
            "operator": operator,
            "stop_reason": stop,
            "branches": len(items),
            "semantic_changed": sum(1 for row in items if _truthy(row.get("semantic_changed"))),
            "depth2": sum(1 for row in items if _num(row.get("depth_reached")) >= 2 or row.get("depth2_action")),
            "rollback_success": sum(1 for row in items if _truthy(row.get("rollback_success"))),
        })
    return out


def _hint_follow(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("variant", "")), str(row.get("hint_type") or row.get("type") or "unknown"))].append(row)
    out: list[dict[str, Any]] = []
    for (variant, hint_type), items in sorted(grouped.items()):
        hits = sum(1 for row in items if _truthy(row.get("hit") or row.get("hint_hit") or row.get("is_hit")))
        follows = sum(1 for row in items if _truthy(row.get("follow") or row.get("followed") or row.get("is_followed")))
        out.append({
            "variant": variant,
            "hint_type": hint_type,
            "rows": len(items),
            "hits": hits,
            "hit_rate": hits / len(items) if items else 0.0,
            "follows": follows,
            "follow_rate": follows / len(items) if items else 0.0,
        })
    return out


def _write_report(output: Path, summaries: list[dict[str, Any]], per_task: list[dict[str, Any]], rollback_failures: list[dict[str, Any]], strategy_rows: list[dict[str, Any]], hint_rows: list[dict[str, Any]], invalids: list[str]) -> None:
    baseline = next((row for row in summaries if row["variant"] == "B0_BASELINE_RERUN"), None)
    lines: list[str] = []
    lines.extend([
        "# 30-task AndroidWorld 搜索策略对比报告",
        "",
        "## 1. 实验设置",
        "",
        "- 任务：固定 30-task app-stratified selection；按 variant 分片，多台机器分别跑不同策略，每个策略都跑完整 30 题。",
        "- Variant：B0 baseline + S1/S2/S3/S4/S5 budget12。",
        "- Exploration：与当前 t-step VLM planned action 解耦；只使用 UI state/task info；证据只允许 t+1 prompt 使用。",
        "- Safety：active t+2 禁止；depth2 仅在 semantic changed、safe、rollback controllable 时展开。",
        "- Evidence gate：只注入 state-aligned + rollback-verified + target/entity exact match；低置信、system/status-bar、弱相关 schema/avoid hint 拒绝。",
        "",
        "## 2. 每种策略实际搜索目标",
        "",
    ])
    for variant in VARIANTS:
        lines.append(f"- `{variant}`: {STRATEGY_TEXT[variant]}")
    lines.extend([
        "",
        "## 3. 总体结果",
        "",
        "| variant | success | success_rate | avg_steps | avg_task_latency_s | exploration_steps | avg_attempts | branches | depth2 | rollback_failures | injected | hint_follow_rate | action_answer_follow_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summaries:
        lines.append(
            f"| `{row['variant']}` | {row['successes']}/{row['tasks']} | {row['success_rate']:.3f} | {row['avg_steps']:.2f} | {row['avg_task_latency_s']:.2f} | {row['exploration_steps']} | {row['avg_attempts_per_exploration_step']:.2f} | {row['branches']} | {row['depth2_branches']} | {row['rollback_failures']} | {row['evidence_injected']} | {row['hint_follow_rate']:.3f} | {row['action_answer_follow_rate']:.3f} |"
        )
    lines.extend(["", "## 4. 相对 baseline 的 rescued / broken", ""])
    if baseline:
        base_success = {str(row.get("task")): _success(row) for row in per_task if row.get("variant") == "B0_BASELINE_RERUN"}
        for variant in [item for item in VARIANTS if item != "B0_BASELINE_RERUN"]:
            variant_rows = [row for row in per_task if row.get("variant") == variant]
            rescued = [str(row.get("task")) for row in variant_rows if _success(row) and not base_success.get(str(row.get("task")), False)]
            broken = [str(row.get("task")) for row in variant_rows if not _success(row) and base_success.get(str(row.get("task")), False)]
            lines.append(f"- `{variant}` rescued={len(rescued)}: {', '.join(rescued) if rescued else 'none'}")
            lines.append(f"- `{variant}` broken={len(broken)}: {', '.join(broken) if broken else 'none'}")
    lines.extend(["", "## 5. Exploration 中间结果：选择了什么、为什么停", ""])
    lines.append("| variant | operator | stop_reason | branches | semantic_changed | depth2 | rollback_success |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in strategy_rows[:120]:
        lines.append(f"| `{row['variant']}` | {row['operator']} | {row['stop_reason']} | {row['branches']} | {row['semantic_changed']} | {row['depth2']} | {row['rollback_success']} |")
    lines.extend(["", "## 6. Evidence 注入和 hint follow 计算", ""])
    lines.append("- `hit_rate`：`hint_hit_follow.jsonl` 中 hit/hint_hit/is_hit 为真时计为 hit，分母为该 hint type 的记录数。")
    lines.append("- `follow_rate`：下一步 action 或回答行为被记录为 follow/followed/is_followed 时计为 follow。AVOID_HINT 的 follow 表示模型没有违反 avoid；ACTION/ANSWER 的 follow 才表示正向采用。")
    lines.append("- `action_answer_follow_rate`：只在 ACTION/ANSWER 类 hint 上计算，避免被 AVOID_HINT 的不违反行为虚高。")
    lines.append("")
    lines.append("| variant | hint_type | rows | hits | hit_rate | follows | follow_rate |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in hint_rows:
        lines.append(f"| `{row['variant']}` | {row['hint_type']} | {row['rows']} | {row['hits']} | {row['hit_rate']:.3f} | {row['follows']} | {row['follow_rate']:.3f} |")
    lines.extend(["", "## 7. Rollback 失败分析", ""])
    if rollback_failures:
        lines.append("| variant | task | step | branch | level | mode | reason | screenshot | discarded | contaminated | suppressed |")
        lines.append("| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rollback_failures[:80]:
            reason = str(row.get("failure_reasons") or row.get("level2_trigger_reason") or "").replace("\n", " ")[:160]
            lines.append(f"| `{row['variant']}` | {row['task']} | {row['step']} | {row['branch_id']} | {row['rollback_level']} | {row['rollback_mode']} | {reason} | {row['screenshot_path']} | {row['branch_evidence_discarded']} | {row['prompt_history_contaminated']} | {row['main_action_suppressed']} |")
    else:
        lines.append("- 当前合并结果没有 rollback failure 记录，或者相关 JSONL 尚未产生。")
    lines.extend(["", "## 8. Invalid-rule 检查", ""])
    if invalids:
        for item in invalids:
            lines.append(f"- INVALID: {item}")
    else:
        lines.append("- 未发现 decoupled invalid、prompt history contamination、active t+2 或缺失关键 artifact。")
    lines.extend(["", "## 9. 当前结果说明", ""])
    lines.append("- 如果 S1-S5 success rate 没有超过 B0，但 latency 和 rollback failure 增加，说明当前 exploration 更多是在消耗预算而不是产生可执行的 exact evidence。")
    lines.append("- 如果 `action_answer_follow_rate` 接近 0，而 AVOID_HINT follow 高，说明 prompt 注入主要起到约束/避坑作用，没有真正驱动主 action。")
    lines.append("- 如果 depth2 很少，主要瓶颈是 semantic_changed/rollback controllable gate 过严或 safe candidate 可用性不足；如果 depth2 多但 success 不升，说明深层观测没有被 summarizer 转成 exact target evidence。")
    lines.append("- 如果 broken 任务集中在 Joplin/OpenTracks/Files，优先检查 rollback_level2_cases 和 rollback_gate_decisions，确认 branch 证据是否已被丢弃且没有污染 prompt/history。")
    (output / "merged_strategy_30task_report_cn.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    run_group = Path(args.run_group).resolve()
    output = Path(args.output).resolve() if args.output else run_group / "merged"
    output.mkdir(parents=True, exist_ok=True)

    all_per_task: list[dict[str, Any]] = []
    all_per_step: list[dict[str, Any]] = []
    all_json: dict[str, list[dict[str, Any]]] = {name: [] for name in JSONL_FILES}
    all_csv: dict[str, list[dict[str, Any]]] = {name: [] for name in CSV_FILES}
    missing_required: list[str] = []

    for machine_id, variant, variant_dir in _variant_dirs(run_group):
        for csv_name in CSV_FILES:
            rows = [_enrich(row, machine_id, variant) for row in _read_csv(variant_dir / csv_name)]
            all_csv[csv_name].extend(rows)
            if csv_name == "per_task_results.csv":
                all_per_task.extend(rows)
            elif csv_name == "per_step_metrics.csv":
                all_per_step.extend(rows)
        for jsonl_name in JSONL_FILES:
            path = variant_dir / jsonl_name
            if not path.exists():
                missing_required.append(f"{machine_id}/{variant}/{jsonl_name}")
            rows = [_enrich(row, machine_id, variant) for row in _read_jsonl(path)]
            all_json[jsonl_name].extend(rows)

    _write_csv(output / "merged_per_task_results.csv", all_per_task)
    _write_csv(output / "merged_per_step_metrics.csv", all_per_step)
    for csv_name, rows in all_csv.items():
        _write_csv(output / f"merged_{csv_name}", rows)
    for jsonl_name, rows in all_json.items():
        out_path = output / f"merged_{jsonl_name}"
        if out_path.exists():
            out_path.unlink()
        _append_jsonl(out_path, rows)

    summaries: list[dict[str, Any]] = []
    per_app_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_task = [row for row in all_per_task if row.get("variant") == variant]
        variant_step = [row for row in all_per_step if row.get("variant") == variant]
        variant_json = {name: [row for row in rows if row.get("variant") == variant] for name, rows in all_json.items()}
        summaries.append(_summarize_variant(variant, variant_task, variant_step, variant_json))
        per_app_rows.extend(_per_app(variant_task, variant))

    rollback_rows = _rollback_failures(all_json["rollback_events.jsonl"])
    strategy_rows = _strategy_selection(all_json["exploration_branch_trace.jsonl"])
    hint_rows = _hint_follow(all_json["hint_hit_follow.jsonl"])

    _write_csv(output / "merged_variant_summary.csv", summaries)
    _write_csv(output / "merged_per_app_task_mode_summary.csv", per_app_rows)
    _write_csv(output / "merged_rollback_failures.csv", rollback_rows)
    _write_csv(output / "merged_strategy_selection_summary.csv", strategy_rows)
    _write_csv(output / "merged_hint_follow_summary.csv", hint_rows)

    invalids: list[str] = []
    invalids.extend(f"missing artifact: {item}" for item in missing_required)
    for row in summaries:
        if row["tasks"] != 30:
            invalids.append(f"{row['variant']} task_count={row['tasks']} expected=30")
        if row["decoupled_invalid_steps"]:
            invalids.append(f"{row['variant']} decoupled_invalid_steps={row['decoupled_invalid_steps']}")
        if row["prompt_history_contaminated"]:
            invalids.append(f"{row['variant']} prompt_history_contaminated={row['prompt_history_contaminated']}")
        if row["main_action_suppressed_by_exploration"]:
            invalids.append(f"{row['variant']} main_action_suppressed_by_exploration={row['main_action_suppressed_by_exploration']}")
        if row["active_t2_events"]:
            invalids.append(f"{row['variant']} active_t2_events={row['active_t2_events']}")
    _write_report(output, summaries, all_per_task, rollback_rows, strategy_rows, hint_rows, invalids)
    (output / "merge_manifest.json").write_text(json.dumps({"run_group": str(run_group), "output": str(output), "invalids": invalids}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge_strategy30] output={output}")
    print(f"[merge_strategy30] invalid_count={len(invalids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
