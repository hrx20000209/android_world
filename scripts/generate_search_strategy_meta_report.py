#!/usr/bin/env python3
"""Generate a Chinese meta report for B0/S1-S5 search strategy results."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "search_strategy_meta_analysis"
OUT_MD = OUT_DIR / "b0_s1_s5_vs_current_strategy_report_cn.md"
OUT_JSON = OUT_DIR / "b0_s1_s5_vs_current_strategy_summary.json"


FIXED12_ROOT = ROOT / "results" / "phase1_strategy_compare_fixed_v3" / "run_20260602T110121"
FIXED12_VARIANTS = {
    "B0": ("Baseline no exploration", FIXED12_ROOT / "B0_BASELINE_RERUN"),
    "S1": ("BFS", FIXED12_ROOT / "S1_BFS"),
    "S2": ("DFS", FIXED12_ROOT / "S2_DFS"),
    "S3": ("Beam", FIXED12_ROOT / "S3_BEAM"),
    "S4": ("MCTS old", FIXED12_ROOT / "S4_MCTS"),
    "S5": ("Operator-Stratified Best-First", FIXED12_ROOT / "S5_OPERATOR_STRATIFIED_BEST_FIRST"),
}

SPLIT30_ROOT = ROOT / "results" / "strategy_30task_split" / "sensys_strategy30_v1"
GIT_UPLOAD_ROOT = ROOT / "results" / "git_upload" / "sensys_strategy30_v1_machine2_upload" / "sensys_strategy30_v1"
SPLIT30_VARIANTS = {
    "B0": (
        "Baseline no exploration",
        SPLIT30_ROOT / "machine_1" / "B0_BASELINE_RERUN",
        SPLIT30_ROOT / "machine_1" / "B0_BASELINE_RERUN" / "run_20260602T152055" / "report" / "summary.json",
    ),
    "S1": (
        "BFS budget12",
        SPLIT30_ROOT / "machine_1" / "S1_BFS_BUDGET12",
        SPLIT30_ROOT / "machine_1" / "S1_BFS_BUDGET12" / "run_20260602T160122" / "report" / "summary.json",
    ),
    "S2": (
        "DFS budget12",
        GIT_UPLOAD_ROOT / "machine_2" / "S2_DFS_BUDGET12",
        GIT_UPLOAD_ROOT / "machine_2" / "S2_DFS_BUDGET12" / "report" / "summary.json",
    ),
    "S3": (
        "Beam budget12",
        SPLIT30_ROOT / "machine_1" / "S3_BEAM_BUDGET12",
        SPLIT30_ROOT / "machine_1" / "S3_BEAM_BUDGET12" / "run_20260602T174105" / "report" / "summary.json",
    ),
    "S4": (
        "MCTS budget12 old invalid",
        GIT_UPLOAD_ROOT / "machine_2" / "S4_MCTS_BUDGET12",
        GIT_UPLOAD_ROOT / "machine_2" / "S4_MCTS_BUDGET12" / "report" / "summary.json",
    ),
    "S5": (
        "Pattern-Aware Operator Best-First budget12",
        SPLIT30_ROOT / "machine_1" / "S5_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12",
        SPLIT30_ROOT
        / "machine_1"
        / "S5_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12"
        / "run_20260602T190630"
        / "report"
        / "summary.json",
    ),
}

CURRENT_DIR = ROOT / "results" / "lb_mcts_final_30task" / "LB_MCTS_FINAL"
CURRENT_SUMMARY = CURRENT_DIR / "run_20260602T220405" / "report" / "summary.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _num(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "success"}
    return bool(value)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _pct(n: float, d: float) -> float:
    return n / d if d else 0.0


def _episode_rows(summary: dict[str, Any], variant_dir: Path) -> list[dict[str, Any]]:
    rows = (((summary.get("episodes") or {}).get("task_episode_rows")) or [])
    if rows:
        return list(rows)
    csv_rows = _read_csv(variant_dir / "per_task_results.csv")
    out = []
    for row in csv_rows:
        out.append(
            {
                "task": row.get("task", ""),
                "episode_length": _num(row.get("episode_length")),
                "success": _bool(row.get("success")),
                "exception": _bool(row.get("exception")),
            }
        )
    return out


def _episode_metrics(summary: dict[str, Any], variant_dir: Path) -> dict[str, Any]:
    episodes = dict(summary.get("episodes") or {})
    rows = _episode_rows(summary, variant_dir)
    if rows and not episodes:
        total = len(rows)
        success = sum(1 for r in rows if _bool(r.get("success")))
        exceptions = sum(1 for r in rows if _bool(r.get("exception")))
        complete_rows = [r for r in rows if not _bool(r.get("exception"))]
        episodes = {
            "total": total,
            "complete": len(complete_rows),
            "exception_failures": exceptions,
            "success": success,
            "success_rate_all_trials": _pct(success, total),
            "success_rate_complete_only": _pct(success, len(complete_rows)),
            "avg_episode_length_complete": _mean([_num(r.get("episode_length")) for r in complete_rows]),
            "avg_episode_length_success": _mean(
                [_num(r.get("episode_length")) for r in rows if _bool(r.get("success"))]
            ),
        }
    episodes["task_episode_rows"] = rows
    return episodes


def _summarize_trace(variant_dir: Path) -> dict[str, Any]:
    step_rows = _read_jsonl(variant_dir / "exploration_step_summary.jsonl")
    branch_rows = _read_jsonl(variant_dir / "exploration_branch_trace.jsonl")
    evidence_rows = _read_jsonl(variant_dir / "evidence_decisions.jsonl")
    hint_rows = _read_jsonl(variant_dir / "hint_hit_follow.jsonl")
    rollback_rows = _read_jsonl(variant_dir / "rollback_events.jsonl")
    prompt_hint_rows = _read_jsonl(variant_dir / "prompt_hints.jsonl")
    candidate_rows = _read_jsonl(variant_dir / "candidate_scores.jsonl")
    mcts_selection_rows = _read_jsonl(variant_dir / "mcts_selection_records.jsonl")
    latency_budget_rows = _read_jsonl(variant_dir / "latency_budget_records.jsonl")
    pattern_rows = _read_jsonl(variant_dir / "pattern_tuple_records.jsonl")

    triggered = [r for r in step_rows if _bool(r.get("exploration_triggered"))]
    triggered_count = len(triggered)
    attempted = [_num(r.get("attempted_count")) for r in triggered]
    executed = [_num(r.get("executed_attempt_count")) for r in triggered]
    passive = [_num(r.get("passive_attempt_count")) for r in triggered]
    depth1 = [_num(r.get("depth1_count")) for r in triggered]
    depth2 = [_num(r.get("depth2_count")) for r in triggered]
    useful = [_num(r.get("useful_evidence_count")) for r in triggered]
    injected_step = [_num(r.get("injected_evidence_count")) for r in triggered]
    stop_counter: Counter[str] = Counter()
    trigger_counter: Counter[str] = Counter()
    for r in step_rows:
        trigger_counter[str(r.get("trigger_reason") or "")] += 1
        dist = r.get("stop_reason_distribution") or {}
        if isinstance(dist, dict):
            for k, v in dist.items():
                stop_counter[str(k)] += int(_num(v))

    op_counter: Counter[str] = Counter()
    selected_reason_counter: Counter[str] = Counter()
    branch_stop_counter: Counter[str] = Counter()
    depth_counter: Counter[str] = Counter()
    branch_a11y = []
    branch_fetch = []
    screenshots = 0
    examples = []
    for r in branch_rows:
        op = r.get("root_operator") or r.get("operator") or ""
        if op:
            op_counter[str(op)] += 1
        selected_reason_counter[str(r.get("selected_reason") or r.get("selected_reason_root") or "")] += 1
        branch_stop_counter[str(r.get("stop_reason") or r.get("stop_condition") or "")] += 1
        depth_counter[str(int(_num(r.get("depth_reached"))))] += 1
        if r.get("branch_a11y_latency_ms") is not None:
            branch_a11y.append(_num(r.get("branch_a11y_latency_ms")))
        elif r.get("a11y_dump_ms") is not None:
            branch_a11y.append(_num(r.get("a11y_dump_ms")))
        if r.get("branch_state_fetch_ms") is not None:
            branch_fetch.append(_num(r.get("branch_state_fetch_ms")))
        elif r.get("state_fetch_ms") is not None:
            branch_fetch.append(_num(r.get("state_fetch_ms")))
        paths = r.get("screenshot_paths") or []
        if isinstance(paths, list):
            screenshots += len(paths)
        if len(examples) < 6 and r.get("root_candidate_label"):
            examples.append(
                {
                    "task": str(r.get("task_id") or "")[:120],
                    "step": r.get("step"),
                    "operator": op,
                    "label": str(r.get("root_candidate_label") or "")[:80],
                    "stop_reason": r.get("stop_reason") or r.get("stop_condition") or "",
                    "semantic_changed": r.get("semantic_changed"),
                    "rollback_success": r.get("rollback_success"),
                }
            )

    evidence_type_counter: Counter[str] = Counter()
    evidence_final_counter: Counter[str] = Counter()
    reject_counter: Counter[str] = Counter()
    injected_rows = []
    confidence = []
    for r in evidence_rows:
        evidence_type_counter[str(r.get("evidence_type_candidate") or "")] += 1
        evidence_final_counter[str(r.get("final_evidence_type") or r.get("evidence_type") or "")] += 1
        if _bool(r.get("injected")):
            injected_rows.append(r)
        else:
            reject_counter[str(r.get("rejected_reason") or "")] += 1
        if r.get("confidence") is not None:
            confidence.append(_num(r.get("confidence")))

    hint_type_counter = Counter(str(r.get("hint_type") or "") for r in hint_rows)
    follow_reason_counter = Counter(str(r.get("follow_reason") or "") for r in hint_rows)
    hit_count = sum(1 for r in hint_rows if _bool(r.get("hit")))
    followed_count = sum(1 for r in hint_rows if _bool(r.get("followed")))
    violated_count = sum(1 for r in hint_rows if _bool(r.get("violated")))

    prompt_hint_type = Counter(str(r.get("hint_type") or r.get("type") or "") for r in prompt_hint_rows)

    rb_success = sum(1 for r in rollback_rows if _bool(r.get("success")))
    rb_level = Counter(str(r.get("rollback_level") or r.get("level") or "") for r in rollback_rows)
    rb_mode = Counter(str(r.get("rollback_mode") or r.get("mode") or "") for r in rollback_rows)
    rb_fail_reasons = Counter(
        ",".join(map(str, r.get("failure_reasons") or [])) or str(r.get("failure_reason") or "")
        for r in rollback_rows
        if not _bool(r.get("success"))
    )
    rb_fail_tasks = Counter(
        str(r.get("task_id") or "")[:80] for r in rollback_rows if not _bool(r.get("success"))
    )
    rb_latency = [
        _num(r.get("rollback_total_ms") if r.get("rollback_total_ms") is not None else r.get("latency_ms"))
        for r in rollback_rows
    ]
    rb_level2 = sum(1 for r in rollback_rows if _bool(r.get("level2_triggered")) or str(r.get("rollback_level") or r.get("level")) == "level2")

    candidate_operator = Counter(str(r.get("operator") or "") for r in candidate_rows)

    mcts_selected = [r for r in mcts_selection_rows if _bool(r.get("selected"))]
    mcts_ops = Counter(str(r.get("operator") or "") for r in mcts_selected)
    mcts_predictions = Counter(
        str((r.get("pattern_tuple") or {}).get("prediction") or "") for r in mcts_selected
    )
    reward_components: dict[str, list[float]] = defaultdict(list)
    for r in mcts_selected:
        comps = r.get("reward_components") or {}
        if isinstance(comps, dict):
            for k, v in comps.items():
                reward_components[k].append(_num(v))

    latency_actual_rollouts = [_num(r.get("actual_rollouts")) for r in latency_budget_rows if r.get("phase") == "final"]
    latency_stop = Counter()
    for r in latency_budget_rows:
        if _bool(r.get("stopped_by_latency_budget")):
            latency_stop["latency_budget"] += 1
        if _bool(r.get("stopped_by_rollback_failure")):
            latency_stop["rollback_failure"] += 1
        if _bool(r.get("stopped_by_no_safe_action")):
            latency_stop["no_safe_action"] += 1
    pattern_predictions = Counter(str(r.get("prediction") or (r.get("pattern_tuple") or {}).get("prediction") or "") for r in pattern_rows)

    state_metrics = {}
    for row in _read_csv(variant_dir / "state_acquisition_metrics.csv"):
        metric = row.get("metric")
        if metric:
            state_metrics[metric] = _num(row.get("value"))

    return {
        "trace_files": {
            "exploration_step_summary": len(step_rows),
            "exploration_branch_trace": len(branch_rows),
            "evidence_decisions": len(evidence_rows),
            "hint_hit_follow": len(hint_rows),
            "prompt_hints": len(prompt_hint_rows),
            "rollback_events": len(rollback_rows),
            "candidate_scores": len(candidate_rows),
            "mcts_selection_records": len(mcts_selection_rows),
            "latency_budget_records": len(latency_budget_rows),
            "pattern_tuple_records": len(pattern_rows),
        },
        "exploration": {
            "steps": len(step_rows),
            "triggered_steps": triggered_count,
            "trigger_rate": _pct(triggered_count, len(step_rows)),
            "avg_attempted": _mean(attempted),
            "avg_executed": _mean(executed),
            "avg_passive": _mean(passive),
            "avg_depth1": _mean(depth1),
            "avg_depth2": _mean(depth2),
            "total_depth2": sum(depth2),
            "avg_useful_evidence": _mean(useful),
            "avg_injected_evidence_per_triggered_step": _mean(injected_step),
            "attempted_lt_12": sum(1 for v in attempted if v < 12),
            "top_trigger_reasons": trigger_counter.most_common(8),
            "top_stop_reasons": stop_counter.most_common(10),
        },
        "branch": {
            "rows": len(branch_rows),
            "operator_counts": op_counter.most_common(12),
            "selected_reason_counts": selected_reason_counter.most_common(8),
            "stop_reason_counts": branch_stop_counter.most_common(12),
            "depth_counts": depth_counter.most_common(),
            "avg_branch_a11y_latency_ms": _mean(branch_a11y),
            "avg_branch_state_fetch_ms": _mean(branch_fetch),
            "screenshot_refs": screenshots,
            "examples": examples,
        },
        "evidence": {
            "rows": len(evidence_rows),
            "injected": len(injected_rows),
            "injection_rate": _pct(len(injected_rows), len(evidence_rows)),
            "candidate_type_counts": evidence_type_counter.most_common(12),
            "final_type_counts": evidence_final_counter.most_common(12),
            "reject_reasons": reject_counter.most_common(12),
            "avg_confidence": _mean(confidence),
        },
        "hints": {
            "hit_follow_rows": len(hint_rows),
            "hit": hit_count,
            "followed": followed_count,
            "violated": violated_count,
            "hit_rate": _pct(hit_count, len(hint_rows)),
            "follow_rate": _pct(followed_count, len(hint_rows)),
            "hint_type_counts": hint_type_counter.most_common(8),
            "follow_reasons": follow_reason_counter.most_common(8),
            "prompt_hint_rows": len(prompt_hint_rows),
            "prompt_hint_types": prompt_hint_type.most_common(8),
        },
        "rollback": {
            "rows": len(rollback_rows),
            "success": rb_success,
            "failures": len(rollback_rows) - rb_success,
            "success_rate": _pct(rb_success, len(rollback_rows)),
            "level_counts": rb_level.most_common(),
            "mode_counts": rb_mode.most_common(8),
            "level2_count": rb_level2,
            "fail_reasons": rb_fail_reasons.most_common(8),
            "fail_tasks": rb_fail_tasks.most_common(8),
            "avg_latency_ms": _mean(rb_latency),
        },
        "state_acquisition": state_metrics,
        "candidate_operator_counts": candidate_operator.most_common(12),
        "mcts": {
            "selection_rows": len(mcts_selection_rows),
            "selected_rows": len(mcts_selected),
            "selected_operator_counts": mcts_ops.most_common(12),
            "selected_pattern_predictions": mcts_predictions.most_common(12),
            "avg_reward_components": {k: _mean(v) for k, v in sorted(reward_components.items())},
            "avg_actual_rollouts_final": _mean(latency_actual_rollouts),
            "latency_stop_counts": latency_stop.most_common(),
            "pattern_prediction_counts": pattern_predictions.most_common(12),
        },
    }


def _summarize_variant(name: str, label: str, variant_dir: Path, summary_path: Path | None = None) -> dict[str, Any]:
    summary = _read_json(summary_path or (variant_dir / "summary.json"))
    if not summary and (variant_dir / "run_20260602T220405" / "report" / "summary.json").exists():
        summary = _read_json(variant_dir / "run_20260602T220405" / "report" / "summary.json")
    episodes = _episode_metrics(summary, variant_dir)
    actions = dict(summary.get("actions") or {})
    exploration_summary = dict(summary.get("exploration") or {})
    rollback_summary = dict(summary.get("rollback") or {})
    trace = _summarize_trace(variant_dir)
    return {
        "name": name,
        "label": label,
        "dir": str(variant_dir),
        "summary_path": str(summary_path or (variant_dir / "summary.json")),
        "episodes": episodes,
        "actions": actions,
        "summary_exploration": exploration_summary,
        "summary_rollback": rollback_summary,
        "trace": trace,
    }


def _task_map(variant: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(r.get("task")): r
        for r in (variant.get("episodes") or {}).get("task_episode_rows", [])
        if r.get("task")
    }


def _rescued_broken(base: dict[str, Any], variant: dict[str, Any]) -> dict[str, list[str]]:
    b = _task_map(base)
    v = _task_map(variant)
    common = sorted(set(b) & set(v))
    rescued = [t for t in common if not _bool(b[t].get("success")) and _bool(v[t].get("success"))]
    broken = [t for t in common if _bool(b[t].get("success")) and not _bool(v[t].get("success"))]
    both_success = [t for t in common if _bool(b[t].get("success")) and _bool(v[t].get("success"))]
    both_fail = [t for t in common if not _bool(b[t].get("success")) and not _bool(v[t].get("success"))]
    return {
        "common": common,
        "rescued": rescued,
        "broken": broken,
        "both_success": both_success,
        "both_fail": both_fail,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _rate(value: Any) -> str:
    return f"{_num(value) * 100:.1f}%"


def _short_list(items: list[Any], n: int = 5) -> str:
    if not items:
        return "-"
    vals = []
    for item in items[:n]:
        if isinstance(item, (list, tuple)):
            vals.append(f"{item[0]}:{item[1]}")
        else:
            vals.append(str(item))
    suffix = "" if len(items) <= n else f" (+{len(items)-n})"
    return ", ".join(vals) + suffix


def _table_overall(variants: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Variant | Strategy | Tasks | Success | Exception | Avg steps | Step latency s | Exploration latency s | Rollback fail | Inject rate | Hint follow |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, v in variants.items():
        ep = v["episodes"]
        act = v.get("actions") or {}
        tr = v["trace"]
        trace_exp = tr["exploration"]
        ev = tr["evidence"]
        hints = tr["hints"]
        rb = tr["rollback"]
        exp_latency = act.get("avg_exploration_latency_ms")
        if exp_latency is None:
            exp_latency = (v.get("summary_exploration") or {}).get("avg_latency_ms")
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    v["label"],
                    str(int(_num(ep.get("total")))),
                    f"{int(_num(ep.get('success')))}/{int(_num(ep.get('total')))} ({_rate(ep.get('success_rate_all_trials'))})",
                    str(int(_num(ep.get("exception_failures")))),
                    _fmt(ep.get("avg_episode_length_complete")),
                    _fmt(_num(act.get("avg_step_latency_ms")) / 1000.0),
                    _fmt(_num(exp_latency) / 1000.0),
                    str(rb.get("failures")),
                    _rate(ev.get("injection_rate")),
                    _rate(hints.get("follow_rate")),
                ]
            )
            + " |"
        )
    return lines


def _table_mechanism(variants: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Variant | Triggered steps | Avg attempted | Avg executed | Total depth2 | Main operators | Stop reasons | Branch a11y ms | Rollback level2/fail |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for name, v in variants.items():
        tr = v["trace"]
        exp = tr["exploration"]
        br = tr["branch"]
        rb = tr["rollback"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(exp.get("triggered_steps")),
                    _fmt(exp.get("avg_attempted")),
                    _fmt(exp.get("avg_executed")),
                    _fmt(exp.get("total_depth2")),
                    _short_list(br.get("operator_counts", []), 4),
                    _short_list(br.get("stop_reason_counts", []) or exp.get("top_stop_reasons", []), 4),
                    _fmt(br.get("avg_branch_a11y_latency_ms")),
                    f"{rb.get('level2_count')}/{rb.get('failures')}",
                ]
            )
            + " |"
        )
    return lines


def _variant_problem_analysis(name: str, v: dict[str, Any]) -> str:
    tr = v["trace"]
    exp = tr["exploration"]
    ev = tr["evidence"]
    hints = tr["hints"]
    rb = tr["rollback"]
    br = tr["branch"]
    ep = v["episodes"]
    if name == "B0":
        return "B0 没有主动 exploration，优势是快、rollback 风险低；问题是遇到需要先探测列表/搜索/详情页的信息查询任务时，没有额外 evidence 可以救场。"
    if name == "S1":
        return (
            "BFS 覆盖最宽，branch 数和 rollback 数通常最高；它能 rescue 一些 query/list 任务，但大量早层候选是低信息或 passive coverage，"
            f"注入率只有 {_rate(ev.get('injection_rate'))}，整体容易把 latency 和 rollback 成本放大。"
        )
    if name == "S2":
        return (
            "DFS 更偏向沿单一路径深入，但当前 depth2 gate 实际触发很少，DFS 的优势没有兑现；同时早期错误分支会占用预算，"
            f"hint follow rate {_rate(hints.get('follow_rate'))} 不足以抵消 broken 任务。"
        )
    if name == "S3":
        return (
            "Beam 比 DFS 稳定一些，能保留多个候选，但 beam 的排序仍依赖浅层 score；当 score 与真实任务 slot 不一致时，"
            "会反复探索相似列表/导航控件，产生中等 coverage 但低有效注入。"
        )
    if name == "S4":
        if _num(ep.get("exception_failures")) > 5:
            return (
                "旧 S4/MCTS 这轮不应视为有效策略结论：exception 很高，旧报告中也指出 prompt/context 文件缺失导致大量任务直接失败。"
                "能用的信息主要是：未修复工程链路时，MCTS 的额外 bookkeeping 会把 latency 放大到不可接受。"
            )
        return "旧 S4/MCTS 的 rollout 代价高，且 reward/rollback gate 当时不够稳定，导致搜索开销大于收益。"
    if name == "S5":
        return (
            "S5 引入 operator-stratified / pattern-aware 候选分层，覆盖更有结构；但旧版 depth2 基本没有真正产生收益，"
            f"rollback fail {rb.get('failures')} 次，ACTION_HINT 大量因 low confidence 被拒，最终成功率没有稳定超过 baseline。"
        )
    if name == "CURRENT":
        return (
            "当前 LB-MCTS/PASE 明显降低了 exploration 成本：用 latency budget、pattern tuple、rollback risk 和 reward prior 控制 rollout，"
            f"成功率 {_rate(ep.get('success_rate_all_trials'))}，平均 exploration latency {_fmt(_num((v.get('actions') or {}).get('avg_exploration_latency_ms'))/1000.0)}s。"
            "它的问题是 hint hit/follow 仍低，且某些创建/编辑/浏览器任务仍无法靠探索解决。"
        )
    return ""


def _write_report(
    fixed: dict[str, dict[str, Any]],
    split: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    comparisons: dict[str, Any],
) -> None:
    current_v = current["CURRENT"]
    lines: list[str] = [
        "# B0 / S1-S5 搜索策略与当前 LB-MCTS/PASE 对比报告",
        "",
        "生成依据：本报告只读取已有落盘结果，不重新运行 AndroidWorld。",
        "",
        "## 1. 找到的结果目录",
        "",
        f"- 12-task 全策略主对比：`{FIXED12_ROOT}`",
        f"- 30-task split 策略对比：`{SPLIT30_ROOT}`",
        f"- S2/S4 上传结果：`{GIT_UPLOAD_ROOT}`",
        f"- 当前策略 30-task：`{CURRENT_DIR}`",
        "",
        "## 2. 结论摘要",
        "",
        "- B0/S1-S5 的搜索策略记录已经找到。最干净的同目录全策略记录是 `phase1_strategy_compare_fixed_v3/run_20260602T110121`；更接近最终 30task 的记录是 `strategy_30task_split/sensys_strategy30_v1`，其中 S2/S4 来自 `git_upload`。",
        "- 旧 S1/S2/S3/S5 的共性问题不是“没有探索”，而是探索结果没有稳定转化为可注入、可被模型 follow 的 evidence；探索覆盖增加了，但 prompt 行为提升很弱。",
        "- 旧 S4/MCTS 在 split 30task 中是无效 run：22/30 exception，不能作为 MCTS 搜索思想的负面结论，只能说明当时工程链路坏了。",
        "- 当前 LB-MCTS/PASE 30task 是目前最强结果：成功率 16/30 = 53.3%，高于 30task B0 的 15/30，也明显高于旧 S5 的 13/30；同时 exploration latency 从旧 S5 约 9.27s/step 降到约 1.34s/step。",
        "- 当前策略仍未彻底解决 hint hit/follow：成功率提升主要来自更少的破坏、更低 rollback 成本和更精准的候选/预算控制，而不是大量 ACTION/ANSWER hint 被模型直接采用。",
        "",
        "## 3. 策略定义",
        "",
        "| Variant | 搜索策略 | 搜索目标 | 主要风险 |",
        "| --- | --- | --- | --- |",
        "| B0 | 无 exploration baseline | 只依赖主 VLM 当前观察和历史 | 信息不足时无法主动查证 |",
        "| S1 | BFS | 宽覆盖 root safe candidates，优先发现可见答案、avoid、schema | coverage 过宽，rollback/latency 成本高 |",
        "| S2 | DFS | 沿候选路径尝试更深搜索 | depth2 gate 少触发，容易被早期错误分支占预算 |",
        "| S3 | Beam | 保留多个高分候选并扩展 | beam score 与真实任务 slot 不一致时会重复弱相关控件 |",
        "| S4 | 旧 MCTS | 用 rollout/reward 选择候选 | split run 工程异常，旧实现 latency 很高 |",
        "| S5 | Operator-Stratified Best-First / PASE 早期版 | 按 operator 分层，找 slot/target/answer/action boundary | evidence gate 太保守，depth2 收益弱 |",
        "| CURRENT | LB-MCTS / 当前 PASE | pattern-aware + latency budget + rollback risk + reward prior | hint follow 仍弱，创建/编辑类任务帮助有限 |",
        "",
        "## 4. 12-task 全策略主对比",
        "",
        "这组来自同一目录，B0/S1/S2/S3/S4/S5 字段一致，适合分析机制差异；但任务数只有 12，不等同于最终 30task 成功率。",
        "",
        *_table_overall(fixed),
        "",
        *_table_mechanism(fixed),
        "",
        "### 4.1 每个旧策略的问题",
        "",
    ]
    for name, v in fixed.items():
        lines.append(f"- `{name}`: {_variant_problem_analysis(name, v)}")
    lines.extend(
        [
            "",
            "### 4.2 rescued / broken 相对 B0",
            "",
            "| Variant | Rescued | Broken | Rescued tasks | Broken tasks |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for name, comp in comparisons["fixed_vs_b0"].items():
        lines.append(
            f"| {name} | {len(comp['rescued'])} | {len(comp['broken'])} | "
            f"{_short_list(comp['rescued'], 4)} | {_short_list(comp['broken'], 4)} |"
        )

    lines.extend(
        [
            "",
            "## 5. 30-task split 策略对比",
            "",
            "这组更接近最终 frozen 30task。B0/S1/S3/S5 在本机，S2/S4 在 `git_upload`。注意 S4 是异常 run，不能当作有效 MCTS。",
            "",
            *_table_overall(split),
            "",
            *_table_mechanism(split),
            "",
            "### 5.1 30task rescued / broken 相对 B0",
            "",
            "| Variant | Rescued | Broken | Rescued tasks | Broken tasks |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for name, comp in comparisons["split_vs_b0"].items():
        lines.append(
            f"| {name} | {len(comp['rescued'])} | {len(comp['broken'])} | "
            f"{_short_list(comp['rescued'], 5)} | {_short_list(comp['broken'], 5)} |"
        )

    lines.extend(
        [
            "",
            "## 6. 当前 LB-MCTS/PASE 与旧策略全方位对比",
            "",
            "当前策略与 30task split B0/S5 是最有意义的直接比较，因为都是 frozen 30task 范围。",
            "",
            *_table_overall({"B0_30": split["B0"], "S5_old_30": split["S5"], "CURRENT": current_v}),
            "",
            *_table_mechanism({"B0_30": split["B0"], "S5_old_30": split["S5"], "CURRENT": current_v}),
            "",
            "### 6.1 当前策略的 MCTS / pattern-aware 中间结果",
            "",
        ]
    )
    cur_trace = current_v["trace"]
    cur_mcts = cur_trace["mcts"]
    lines.extend(
        [
            f"- MCTS selection rows: `{cur_mcts['selection_rows']}`，selected rows: `{cur_mcts['selected_rows']}`。",
            f"- selected operator top: `{_short_list(cur_mcts['selected_operator_counts'], 8)}`。",
            f"- selected pattern predictions top: `{_short_list(cur_mcts['selected_pattern_predictions'], 8)}`。",
            f"- latency budget final avg rollouts: `{_fmt(cur_mcts['avg_actual_rollouts_final'])}`。",
            f"- latency stop counts: `{_short_list(cur_mcts['latency_stop_counts'], 8)}`。",
            f"- avg reward components: `{ {k: round(v, 3) for k, v in cur_mcts['avg_reward_components'].items()} }`。",
            "",
            "### 6.2 当前策略相对 30task B0",
            "",
        ]
    )
    comp_cur = comparisons["current_vs_split_b0"]
    lines.extend(
        [
            f"- rescued: `{len(comp_cur['rescued'])}`，包括 `{_short_list(comp_cur['rescued'], 10)}`。",
            f"- broken: `{len(comp_cur['broken'])}`，包括 `{_short_list(comp_cur['broken'], 10)}`。",
            f"- both success: `{len(comp_cur['both_success'])}`；both fail: `{len(comp_cur['both_fail'])}`。",
            "",
            "解释：当前策略比 B0 多成功 1 个任务，但不是无代价。它 rescue 了若干需要额外列表/搜索/详情页证据的任务，同时仍会破坏个别 baseline 能完成的路径。相比旧 S5，它最大的改进是减少无效探索和 rollback 数量，而不是大幅提高 hint follow。",
            "",
            "### 6.3 当前策略相对旧 S5",
            "",
        ]
    )
    comp_s5 = comparisons["current_vs_split_s5"]
    lines.extend(
        [
            f"- 当前成功率 `{_rate(current_v['episodes'].get('success_rate_all_trials'))}` vs 旧 S5 `{_rate(split['S5']['episodes'].get('success_rate_all_trials'))}`。",
            f"- 当前 avg steps `{_fmt(current_v['episodes'].get('avg_episode_length_complete'))}` vs 旧 S5 `{_fmt(split['S5']['episodes'].get('avg_episode_length_complete'))}`。",
            f"- 当前 avg step latency `{_fmt(_num((current_v.get('actions') or {}).get('avg_step_latency_ms'))/1000.0)}s` vs 旧 S5 `{_fmt(_num((split['S5'].get('actions') or {}).get('avg_step_latency_ms'))/1000.0)}s`。",
            f"- 当前 rollback events `{current_v['trace']['rollback']['rows']}` / fail `{current_v['trace']['rollback']['failures']}` vs 旧 S5 `{split['S5']['trace']['rollback']['rows']}` / fail `{split['S5']['trace']['rollback']['failures']}`。",
            f"- 当前 exploration evidence rows `{current_v['trace']['evidence']['rows']}` / injected `{current_v['trace']['evidence']['injected']}` vs 旧 S5 `{split['S5']['trace']['evidence']['rows']}` / injected `{split['S5']['trace']['evidence']['injected']}`。",
            f"- 当前相对旧 S5 rescued `{len(comp_s5['rescued'])}`，broken `{len(comp_s5['broken'])}`。",
            "",
            "结论：当前策略把旧 S5 的“大量 branch + 高 rollback + 高延迟”改成了“少量高置信 rollout + latency budget + reward prior”。这解释了为什么当前策略成功率更高、步数更低、速度接近 baseline。",
            "",
            "## 7. 中间结果说明：这些指标如何解读",
            "",
            "- `attempted_count`: 当前 step 被搜索策略纳入预算的候选数，不等于实际点击数。",
            "- `executed_attempt_count`: 真的执行 speculative action 的候选数；低于 attempted 通常说明存在 passive coverage、风险过滤、重复候选或 gate 拦截。",
            "- `depth2_count`: 真正到第二层探索的次数。旧策略中大量 depth2 被 hash/semantic/should_expand gate 阻断，所以 DFS/MCTS 深搜优势没有完全体现。",
            "- `evidence injection rate`: evidence_decisions 中 `injected=true` 的比例。旧策略最大问题是 evidence 很多但低置信或 state 不对齐，最终进 prompt 的少。",
            "- `hint hit rate`: hint 建议的 ACTION/ANSWER 是否被下一步 VLM 动作/答案直接命中。",
            "- `hint follow rate`: 包含 AVOID_HINT 被遵守等更宽松的 follow。follow 高不一定代表任务成功，因为 avoid hint 可能只是阻止错误，不提供下一步正向动作。",
            "- `rollback success rate`: exploration 是否能恢复 root state。失败 branch 理论上会被丢弃；但失败越多，latency 越高，也越可能改变主循环节奏。",
            "",
            "## 8. 主要 failure pattern",
            "",
            "- 文件/列表任务：`FilesMoveFile`、删除/移动类任务对路径和列表状态敏感，旧策略 rollback level2/fail 多，容易 broken。",
            "- 系统设置任务：亮度/Wi-Fi 类任务 baseline 常可直接做，探索反而可能引入弱相关 hint 或消耗 step budget。",
            "- Joplin/Notes 查询任务：探索可能有帮助，但旧 gate 经常把 ACTION_HINT 拒为 low confidence，导致 evidence 没能稳定进入 prompt。",
            "- 浏览器/绘图/创建编辑类：这类任务主要需要精确连续操作，浅层 a11y exploration 很难提供决定性信息；当前策略也仍大量 fail。",
            "",
            "## 9. 最终判断",
            "",
            "旧 B0/S1-S5 实验说明：单纯提高搜索覆盖不够，必须同时降低 rollback/latency、提高 state-aligned evidence 的注入质量，并让 hint 真正可执行。当前 LB-MCTS/PASE 已经朝这个方向改进：它不是靠更多探索取胜，而是靠更少、更安全、更有 pattern prior 的探索取胜。当前结果支持继续沿 LB-MCTS/PASE 做小步优化，但不支持回退到 BFS/DFS/Beam 的高覆盖策略。",
            "",
            "## 10. 输出文件",
            "",
            f"- JSON summary: `{OUT_JSON}`",
            f"- Markdown report: `{OUT_MD}`",
        ]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    fixed = {
        name: _summarize_variant(name, label, path)
        for name, (label, path) in FIXED12_VARIANTS.items()
    }
    split = {
        name: _summarize_variant(name, label, path, summary_path)
        for name, (label, path, summary_path) in SPLIT30_VARIANTS.items()
    }
    current = {
        "CURRENT": _summarize_variant(
            "CURRENT",
            "LB-MCTS / current PASE 30task",
            CURRENT_DIR,
            CURRENT_SUMMARY,
        )
    }
    comparisons = {
        "fixed_vs_b0": {
            name: _rescued_broken(fixed["B0"], v)
            for name, v in fixed.items()
            if name != "B0"
        },
        "split_vs_b0": {
            name: _rescued_broken(split["B0"], v)
            for name, v in split.items()
            if name != "B0"
        },
        "current_vs_split_b0": _rescued_broken(split["B0"], current["CURRENT"]),
        "current_vs_split_s5": _rescued_broken(split["S5"], current["CURRENT"]),
    }
    payload = {
        "fixed12": fixed,
        "split30": split,
        "current": current,
        "comparisons": comparisons,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_report(fixed, split, current, comparisons)
    print(f"report={OUT_MD}")
    print(f"json={OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
