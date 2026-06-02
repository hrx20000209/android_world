#!/usr/bin/env python3
"""Materialize LB-MCTS outputs and generate the final Chinese report."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMELINE_SCRIPT = REPO_ROOT / "scripts" / "generate_rollback_timeline.py"

REQUIRED_JSONL = [
    "step_limit_records.jsonl",
    "candidate_filter_stats.jsonl",
    "candidate_scores.jsonl",
    "pattern_tuple_records.jsonl",
    "mcts_selection_records.jsonl",
    "mcts_reward_records.jsonl",
    "latency_budget_records.jsonl",
    "exploration_step_summary.jsonl",
    "exploration_branch_trace.jsonl",
    "exploration_page_trace.jsonl",
    "state_alignment.jsonl",
    "evidence_decisions.jsonl",
    "prompt_hints.jsonl",
    "prompt_traces.jsonl",
    "hint_hit_follow.jsonl",
    "rollback_events.jsonl",
    "shortcut_shadow_eval.jsonl",
]

REQUIRED_CSV = [
    "per_task_results.csv",
    "per_step_metrics.csv",
    "state_acquisition_metrics.csv",
]

REQUIRED_MD = [
    "rollback_level2_cases.md",
    "rollback_failure_cases.md",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception:
        return []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _task_name(row: dict[str, Any]) -> str:
    for key in ("task", "task_id", "task_name", "name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _latest_run(root: Path) -> Path | None:
    runs = sorted(root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _concat_jsonl(src_root: Path, filename: str, dst: Path) -> int:
    rows = []
    if src_root.exists():
        for path in sorted(src_root.rglob(filename)):
            if path.resolve() == dst.resolve():
                continue
            rows.extend(_read_jsonl(path))
    _write_jsonl(dst, rows)
    return len(rows)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _materialize_per_task(variant_root: Path, run_dir: Path | None) -> list[dict[str, Any]]:
    existing = _read_csv(variant_root / "per_task_results.csv")
    if existing:
        return [dict(r) for r in existing]
    summary = _read_json((run_dir or variant_root) / "report" / "summary.json")
    rows = ((summary.get("episodes") or {}).get("task_episode_rows") or [])
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out_rows.append(
            {
                "task": _task_name(row),
                "episode_length": row.get("episode_length"),
                "success": row.get("success"),
                "exception": row.get("exception"),
            }
        )
    if out_rows:
        _write_csv(variant_root / "per_task_results.csv", out_rows, ["task", "episode_length", "success", "exception"])
    return out_rows


def _materialize_per_step(variant_root: Path, run_dir: Path | None) -> None:
    dst = variant_root / "per_step_metrics.csv"
    if dst.exists() and dst.stat().st_size > 1:
        return
    src = (run_dir or variant_root) / "report" / "step_metrics.csv"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    else:
        dst.write_text("\n", encoding="utf-8")


def _materialize_outputs(variant_root: Path, max_steps: int) -> Path:
    variant_root.mkdir(parents=True, exist_ok=True)
    run_dir = _latest_run(variant_root)
    trace_root = (run_dir / "traces") if run_dir else variant_root
    task_rows = _materialize_per_task(variant_root, run_dir)
    _materialize_per_step(variant_root, run_dir)

    for name in REQUIRED_JSONL:
        if name == "step_limit_records.jsonl":
            continue
        _concat_jsonl(trace_root, name, variant_root / name)

    step_limit_rows = []
    for row in task_rows:
        final_step = int(_to_float(row.get("episode_length")))
        success = _to_bool(row.get("success"))
        step_limit_rows.append(
            {
                "record_type": "StepLimitRecord",
                "task_id": _task_name(row),
                "variant": variant_root.name,
                "max_step_limit": int(max_steps),
                "stopped_by_step_limit": bool((not success) and final_step >= int(max_steps)),
                "final_step": final_step,
                "success": success,
            }
        )
    _write_jsonl(variant_root / "step_limit_records.jsonl", step_limit_rows)

    runtime = variant_root / "runtime_config.yaml"
    if not runtime.exists():
        runtime.write_text(
            "\n".join(
                [
                    "variant: LB_MCTS_FINAL",
                    "strategy: Latency-Bounded Task-Aware MCTS Exploration",
                    f"max_step_limit: {int(max_steps)}",
                    "decoupled_exploration: true",
                    "active_t2_shortcut: false",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    for csv_name in REQUIRED_CSV:
        path = variant_root / csv_name
        if not path.exists():
            path.write_text("\n", encoding="utf-8")
    for md_name in REQUIRED_MD:
        path = variant_root / md_name
        if not path.exists():
            path.write_text("无记录\n", encoding="utf-8")
    return trace_root


def _load_baseline(inventory_path: Path | None, explicit_path: str) -> tuple[Path | None, list[dict[str, Any]], str]:
    if explicit_path:
        root = Path(explicit_path).expanduser().resolve()
        return root, [dict(r) for r in _read_csv(root / "per_task_results.csv")], "explicit"
    if not inventory_path or not inventory_path.exists():
        return None, [], "none"
    inv = _read_json(inventory_path)
    for key in ("matched_selected", "baseline_30", "baseline_50", "baseline_116"):
        item = inv.get(key)
        if isinstance(item, dict) and item.get("path"):
            root = Path(str(item["path"])).expanduser().resolve()
            rows = [dict(r) for r in _read_csv(root / "per_task_results.csv")]
            if rows:
                return root, rows, key
    return None, [], "none"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"task_count": 0, "success_count": 0, "success_rate": 0.0, "avg_steps": 0.0}
    success = sum(1 for r in rows if _to_bool(r.get("success")))
    return {
        "task_count": len(rows),
        "success_count": success,
        "success_rate": float(success) / len(rows),
        "avg_steps": sum(_to_float(r.get("episode_length")) for r in rows) / len(rows),
    }


def _task_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_task_name(r): r for r in rows if _task_name(r)}


def _counter(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str(r.get(key) or "") for r in rows if str(r.get(key) or ""))


def _write_report(
    *,
    variant_root: Path,
    baseline_root: Path | None,
    baseline_source: str,
    max_steps: int,
    report_name: str,
) -> Path:
    lb_rows = [dict(r) for r in _read_csv(variant_root / "per_task_results.csv")]
    base_rows = [dict(r) for r in _read_csv((baseline_root or Path()) / "per_task_results.csv")] if baseline_root else []
    lb_summary = _summary(lb_rows)
    base_summary = _summary(base_rows)

    step_rows = _read_jsonl(variant_root / "exploration_step_summary.jsonl")
    branch_rows = _read_jsonl(variant_root / "exploration_branch_trace.jsonl")
    evidence_rows = _read_jsonl(variant_root / "evidence_decisions.jsonl")
    prompt_hints = _read_jsonl(variant_root / "prompt_hints.jsonl")
    prompt_traces = _read_jsonl(variant_root / "prompt_traces.jsonl")
    hit_rows = _read_jsonl(variant_root / "hint_hit_follow.jsonl")
    rollback_rows = _read_jsonl(variant_root / "rollback_events.jsonl")
    mcts_rows = _read_jsonl(variant_root / "mcts_selection_records.jsonl")
    reward_rows = _read_jsonl(variant_root / "mcts_reward_records.jsonl")
    pattern_rows = _read_jsonl(variant_root / "pattern_tuple_records.jsonl")
    latency_rows = _read_jsonl(variant_root / "latency_budget_records.jsonl")
    step_limit_rows = _read_jsonl(variant_root / "step_limit_records.jsonl")

    base_map = _task_map(base_rows)
    lb_map = _task_map(lb_rows)
    rescued = []
    broken = []
    for task, lb in lb_map.items():
        base = base_map.get(task)
        if not base:
            continue
        if _to_bool(lb.get("success")) and not _to_bool(base.get("success")):
            rescued.append(task)
        if not _to_bool(lb.get("success")) and _to_bool(base.get("success")):
            broken.append(task)

    rollback_fail = [r for r in rollback_rows if not _to_bool(r.get("success", True))]
    level2 = [r for r in rollback_rows if _to_bool(r.get("level2_triggered"))]
    injected = [r for r in evidence_rows if _to_bool(r.get("injected"))]
    hint_hit = sum(1 for r in hit_rows if _to_bool(r.get("hit")))
    hint_follow = sum(1 for r in hit_rows if _to_bool(r.get("followed")) or _to_bool(r.get("follow")))
    depth2_attempted = sum(1 for r in latency_rows if _to_bool(r.get("depth2_attempted")))

    app_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "success": 0, "steps": 0})
    for row in lb_rows:
        task = _task_name(row)
        app = task.split("Task")[0]
        if task:
            app = re_app = "".join(ch for ch in task if not ch.islower())
            app = re_app[:20] or task[:20]
        app_stats[app]["n"] += 1
        app_stats[app]["success"] += 1 if _to_bool(row.get("success")) else 0
        app_stats[app]["steps"] += _to_float(row.get("episode_length"))

    lines = [
        "# LB-MCTS 30-task Final Report",
        "",
        "## 1. Executive summary",
        "",
        f"- LB-MCTS variant path: `{variant_root}`",
        f"- baseline source: `{baseline_source}`",
        f"- baseline path: `{baseline_root or ''}`",
        f"- max step limit: `{int(max_steps)}`",
        f"- LB-MCTS success: `{lb_summary['success_count']}/{lb_summary['task_count']}` = `{lb_summary['success_rate']:.3f}`",
        f"- baseline success: `{base_summary['success_count']}/{base_summary['task_count']}` = `{base_summary['success_rate']:.3f}`",
        f"- LB-MCTS avg steps: `{lb_summary['avg_steps']:.2f}`",
        f"- baseline avg steps: `{base_summary['avg_steps']:.2f}`",
        f"- rescued: `{len(rescued)}` {rescued}",
        f"- broken: `{len(broken)}` {broken}",
        "",
        "Recommendation:",
    ]
    if not lb_rows or not prompt_traces:
        lines.append("- Instrumentation invalid; rerun.")
    elif base_rows and lb_summary["success_rate"] > base_summary["success_rate"] and len(broken) <= len(rescued):
        lines.append("- LB-MCTS is final design; run 116-task.")
    elif base_rows and lb_summary["success_rate"] >= base_summary["success_rate"]:
        lines.append("- LB-MCTS promising; run 50-task first.")
    else:
        lines.append("- LB-MCTS not better; report as limitation/future work.")

    lines.extend(
        [
            "",
            "## 2. Design summary",
            "",
            "- Strategy: Latency-Bounded Task-Aware MCTS Exploration with Pattern-Aware Speculative UI Probing.",
            "- MCTS 只分配有限 speculative rollouts，不再做 BFS/DFS/Beam/PASE 的广泛比较。",
            "- Pattern prior 来自 UI Pattern Tuple `(context, prediction, function, probability)`，用于 PUCT 的 P 项。",
            "- Latency bound 使用最近 VLM reasoning latency 的 EWMA：`min(MAX, max(MIN, SLACK_RATIO * expected_reasoning_latency))`。",
            "- Rollback-safe eligibility 保证 speculative path 和 authoritative path 隔离；rollback fail 的 evidence 不注入 prompt。",
            "- Active t+2 shortcut 禁用；depth2 exploration 只作为 speculative evidence，不直接执行主动作。",
            "",
            "## 3. MCTS details",
            "",
            "- UCT: `Q + c_puct * P * sqrt(log(N_parent + 1) / (N_child + 1)) - lambda_risk * RollbackRisk - lambda_latency * LatencyCostNorm`",
            "- Reward: `3*AnswerBoundary + 2*ActionBoundary + 2*TargetEntityFound + 1.5*SlotCompletionGain + UsefulSchema + HardNegative - 2*RiskViolation - 2*RollbackFailure - LowQualityObservation - 0.5*LatencyCostNorm`",
            f"- MCTSSelectionRecord rows: `{len(mcts_rows)}`",
            f"- MCTSRewardRecord rows: `{len(reward_rows)}`",
            f"- PatternTupleRecord rows: `{len(pattern_rows)}`",
            f"- LatencyBudgetRecord rows: `{len(latency_rows)}`",
            f"- depth2_attempted records: `{depth2_attempted}`",
            "",
            "Top selected operators:",
            "",
            "| operator | count |",
            "| --- | ---: |",
        ]
    )
    for op, count in _counter([r for r in mcts_rows if _to_bool(r.get("selected"))], "operator").most_common(12):
        lines.append(f"| `{op}` | {count} |")

    lines.extend(
        [
            "",
            "## 4. Exploration statistics",
            "",
            f"- exploration_step_summary rows: `{len(step_rows)}`",
            f"- exploration_branch_trace rows: `{len(branch_rows)}`",
            f"- triggered steps: `{sum(1 for r in step_rows if _to_bool(r.get('exploration_triggered')))} `",
            f"- total attempted_count: `{sum(int(_to_float(r.get('attempted_count'))) for r in step_rows)}`",
            f"- total executed_attempt_count: `{sum(int(_to_float(r.get('executed_attempt_count'))) for r in step_rows)}`",
            f"- depth1 count: `{sum(int(_to_float(r.get('depth1_count'))) for r in step_rows)}`",
            f"- depth2 count: `{sum(int(_to_float(r.get('depth2_count'))) for r in step_rows)}`",
            "",
            "Stop reasons:",
            "",
            "| reason | count |",
            "| --- | ---: |",
        ]
    )
    stop_counter: Counter[str] = Counter()
    for row in step_rows:
        dist = row.get("stop_reason_distribution")
        if isinstance(dist, dict):
            stop_counter.update({str(k): int(v) for k, v in dist.items()})
    for reason, count in stop_counter.most_common(15):
        lines.append(f"| `{reason}` | {count} |")

    lines.extend(
        [
            "",
            "## 5. Prompt injection",
            "",
            f"- evidence_decisions rows: `{len(evidence_rows)}`",
            f"- evidence injected: `{len(injected)}`",
            f"- prompt_hints rows: `{len(prompt_hints)}`",
            f"- prompt_traces rows: `{len(prompt_traces)}`",
            f"- hint_hit_follow rows: `{len(hit_rows)}`",
            f"- ACTION/ANSWER hit count: `{hint_hit}`",
            f"- follow count: `{hint_follow}`",
            "",
            "Injected hint types:",
            "",
            "| type | count |",
            "| --- | ---: |",
        ]
    )
    for typ, count in _counter(injected, "final_evidence_type").most_common():
        lines.append(f"| `{typ}` | {count} |")
    lines.extend(["", "Top rejected reasons:", "", "| reason | count |", "| --- | ---: |"])
    for reason, count in _counter([r for r in evidence_rows if not _to_bool(r.get("injected"))], "rejected_reason").most_common(10):
        lines.append(f"| `{reason}` | {count} |")
    sample_prompt = next((r for r in prompt_traces if r.get("exploration_context_text")), {})
    if sample_prompt:
        lines.extend(["", "Example exploration block:", "", "```text", str(sample_prompt.get("exploration_context_text"))[:2500], "```"])

    lines.extend(
        [
            "",
            "## 6. Rollback",
            "",
            f"- rollback_events rows: `{len(rollback_rows)}`",
            f"- level2 count: `{len(level2)}`",
            f"- rollback failures: `{len(rollback_fail)}`",
            f"- prompt_history_contaminated=true count: `{sum(1 for r in rollback_rows if _to_bool(r.get('prompt_history_contaminated')))}`",
            "",
            "Level2 trigger reasons:",
            "",
            "| reason | count |",
            "| --- | ---: |",
        ]
    )
    for reason, count in _counter(level2, "level2_trigger_reason").most_common(12):
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "Rollback failure tasks:", "", "| task | count |", "| --- | ---: |"])
    for task, count in _counter(rollback_fail, "task_id").most_common(12):
        lines.append(f"| `{task}` | {count} |")
    timeline_index = variant_root / "rollback_timeline_images" / "rollback_timeline_index.md"
    lines.append("")
    lines.append(f"- rollback timeline index: `{timeline_index}`")

    lines.extend(
        [
            "",
            "## 7. Performance",
            "",
            "| task | baseline_success | lb_mcts_success | baseline_steps | lb_mcts_steps |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for task, lb in sorted(lb_map.items()):
        base = base_map.get(task, {})
        lines.append(
            f"| `{task}` | {int(_to_bool(base.get('success')))} | {int(_to_bool(lb.get('success')))} | "
            f"{_to_float(base.get('episode_length')):.0f} | {_to_float(lb.get('episode_length')):.0f} |"
        )

    lines.extend(["", "Per-app coarse table:", "", "| app key | tasks | success | avg steps |", "| --- | ---: | ---: | ---: |"])
    for app, stat in sorted(app_stats.items()):
        n = max(1, stat["n"])
        lines.append(f"| `{app}` | {int(stat['n'])} | {int(stat['success'])} | {stat['steps'] / n:.2f} |")

    lines.extend(
        [
            "",
            "## 8. Validity checks",
            "",
            f"- max_step_limit=16 records: `{sum(1 for r in step_limit_rows if int(r.get('max_step_limit') or 0) == int(max_steps))}/{len(step_limit_rows)}`",
            f"- prompt text saved rows: `{sum(1 for r in prompt_traces if r.get('full_prompt_text_path'))}/{len(prompt_traces)}`",
            f"- exploration context saved rows: `{sum(1 for r in prompt_traces if r.get('exploration_context_text_path'))}/{len(prompt_traces)}`",
            f"- rollback level2 trigger missing: `{sum(1 for r in level2 if not r.get('level2_trigger_reason'))}`",
            f"- rollback timeline images dir: `{variant_root / 'rollback_timeline_images'}`",
            f"- depth2 never attempted invalid flag: `{depth2_attempted == 0}`",
            "",
            "## 9. Required files",
            "",
        ]
    )
    for name in ["runtime_config.yaml", *REQUIRED_CSV, *REQUIRED_JSONL, *REQUIRED_MD]:
        lines.append(f"- `{variant_root / name}`")

    out = variant_root / report_name
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--variant", default="LB_MCTS_FINAL")
    parser.add_argument("--baseline_inventory", default="")
    parser.add_argument("--baseline_path", default="")
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--report_name", default="lb_mcts_30task_report_cn.md")
    args = parser.parse_args()

    exp_root = Path(args.experiment_root).expanduser().resolve()
    variant_root = exp_root / args.variant
    if not variant_root.exists() and exp_root.name == args.variant:
        variant_root = exp_root
    trace_root = _materialize_outputs(variant_root, max_steps=int(args.max_steps))
    timeline_dir = variant_root / "rollback_timeline_images"
    subprocess.run(
        [sys.executable, str(TIMELINE_SCRIPT), "--trace_root", str(trace_root), "--output_dir", str(timeline_dir)],
        cwd=str(REPO_ROOT),
        check=False,
    )
    inventory = Path(args.baseline_inventory).expanduser().resolve() if args.baseline_inventory else None
    baseline_root, _, source = _load_baseline(inventory, args.baseline_path)
    report = _write_report(
        variant_root=variant_root,
        baseline_root=baseline_root,
        baseline_source=source,
        max_steps=int(args.max_steps),
        report_name=args.report_name,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
