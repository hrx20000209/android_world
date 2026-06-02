#!/usr/bin/env python3
"""Create a Chinese cross-variant summary for Phase 1 causal debug runs."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


TASK_ORDER = [
    "BrowserMaze",
    "ClockStopWatchRunning",
    "NotesRecipeIngredientCount",
    "SimpleCalendarNextMeetingWithPerson",
    "SportsTrackerActivityDuration",
]

SUMMARY_VARIANTS = [
    "B0_BASELINE_RERUN",
    "D0_EXPLORATION_SHADOW_ONLY",
    "D1_REAL_EXPLORATION_NO_INJECTION",
    "D2_STRICT_INJECTION",
]

D3_MAIN = "D3_RELAXED_DIAGNOSTIC_INJECTION"
D3_REMAINDER = "D3_RELAXED_DIAGNOSTIC_INJECTION_REMAINDER"
D3_MERGED = "D3_RELAXED_DIAGNOSTIC_INJECTION_MERGED"


def _load_json(path: Path) -> dict[str, Any]:
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
            continue
    return rows


def _latest_summary(variant_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    reports = sorted(
        variant_dir.glob("run_*/report/summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        return None, {}
    return reports[0], _load_json(reports[0])


def _parse_driver(driver: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    if not driver.exists():
        return tasks
    current = ""
    pending_latency: tuple[float, int, float] | None = None
    for line in driver.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"^Running task:\s*(\S+)", line)
        if m:
            current = m.group(1)
            pending_latency = None
            continue
        m = re.search(
            r"Task latency: avg_step=([0-9.]+)s over ([0-9]+) steps \(total=([0-9.]+)s\)",
            line,
        )
        if m and current:
            pending_latency = (float(m.group(1)), int(m.group(2)), float(m.group(3)))
            continue
        if current and ("Task Successful" in line or "Task Failed" in line):
            avg_s, steps, total_s = pending_latency or (0.0, 0, 0.0)
            tasks[current] = {
                "result": "pass" if "Task Successful" in line else "fail",
                "success": "Task Successful" in line,
                "steps": steps,
                "avg_step_s": avg_s,
                "total_s": total_s,
            }
            current = ""
            pending_latency = None
    return tasks


def _driver_latency(tasks: dict[str, dict[str, Any]]) -> float | None:
    total_steps = sum(int(t.get("steps") or 0) for t in tasks.values())
    total_s = sum(float(t.get("total_s") or 0.0) for t in tasks.values())
    if not total_steps:
        return None
    return 1000.0 * total_s / total_steps


def _summary_tasks(summary: dict[str, Any], driver_tasks: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = (
        summary.get("episodes", {}).get("task_episode_rows", [])
        if isinstance(summary.get("episodes"), dict)
        else []
    )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        task = row.get("task")
        if not task:
            continue
        out[task] = {
            "result": "pass" if row.get("success") else "fail",
            "success": bool(row.get("success")),
            "steps": int(row.get("episode_length") or 0),
            "avg_step_s": driver_tasks.get(task, {}).get("avg_step_s", 0.0),
            "total_s": driver_tasks.get(task, {}).get("total_s", 0.0),
        }
    for task, data in driver_tasks.items():
        out.setdefault(task, data)
    return out


def _aggregate_trace_roots(trace_roots: list[Path]) -> dict[str, Any]:
    prompt_rows = 0
    injected_true = 0
    rendered_nonempty = 0
    evidence_rows = 0
    candidate_rows = 0
    candidate_raw = 0
    candidate_after_filter = 0
    exploration_rows = 0
    rollback_events = 0
    rollback_failures = 0
    prompt_reject: Counter[str] = Counter()
    evidence_reject: Counter[str] = Counter()
    evidence_types: Counter[str] = Counter()
    operators: Counter[str] = Counter()
    level2: Counter[str] = Counter()
    exploration_total_ms: list[float] = []
    root_state_fetch_ms: list[float] = []
    root_a11y_ms: list[float] = []
    branch_a11y_ms: list[float] = []
    rendered_examples: list[dict[str, Any]] = []

    for root in trace_roots:
        if not root.exists():
            continue
        for path in root.glob("*/prompt_hints.jsonl"):
            for row in _read_jsonl(path):
                prompt_rows += 1
                if row.get("injected"):
                    injected_true += 1
                text = str(row.get("rendered_prompt_text") or "").strip()
                if text:
                    rendered_nonempty += 1
                    if len(rendered_examples) < 4:
                        rendered_examples.append(
                            {
                                "task_dir": path.parent.name,
                                "step": row.get("step"),
                                "text": text[:260],
                            }
                        )
                reason = row.get("rejected_reason")
                if reason:
                    prompt_reject[str(reason)] += 1
        for path in root.glob("*/evidence_decisions.jsonl"):
            for row in _read_jsonl(path):
                evidence_rows += 1
                reason = row.get("rejection_reason")
                if reason:
                    evidence_reject[str(reason)] += 1
                evidence_type = row.get("final_evidence_type") or row.get("evidence_type")
                if evidence_type:
                    evidence_types[str(evidence_type)] += 1
        for path in root.glob("*/candidate_scores.jsonl"):
            for row in _read_jsonl(path):
                candidate_rows += 1
                operators[str(row.get("operator") or "unknown")] += 1
        for path in root.glob("*/exploration_latency.jsonl"):
            for row in _read_jsonl(path):
                exploration_rows += 1
                if isinstance(row.get("exploration_total_ms"), (int, float)):
                    exploration_total_ms.append(float(row["exploration_total_ms"]))
                if isinstance(row.get("root_state_fetch_ms"), (int, float)):
                    root_state_fetch_ms.append(float(row["root_state_fetch_ms"]))
                if isinstance(row.get("root_a11y_ms"), (int, float)):
                    root_a11y_ms.append(float(row["root_a11y_ms"]))
                candidate_raw += int(row.get("candidate_count_raw") or 0)
                candidate_after_filter += int(row.get("candidate_count_after_filter") or 0)
                for branch in row.get("branches") or row.get("branch_latencies") or []:
                    if isinstance(branch.get("depth1_a11y_ms"), (int, float)):
                        branch_a11y_ms.append(float(branch["depth1_a11y_ms"]))
                    if isinstance(branch.get("depth2_a11y_ms"), (int, float)):
                        branch_a11y_ms.append(float(branch["depth2_a11y_ms"]))
        for path in root.glob("*/rollback_events.jsonl"):
            for row in _read_jsonl(path):
                rollback_events += 1
                if not row.get("success", True):
                    rollback_failures += 1
                reason = row.get("level2_trigger_reason")
                if reason:
                    level2[str(reason)] += 1

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "prompt_hint_decisions": prompt_rows,
        "prompt_hints_injected": injected_true,
        "prompt_hints_rendered": rendered_nonempty,
        "prompt_hint_rejection_reasons": dict(prompt_reject),
        "evidence_decisions": evidence_rows,
        "evidence_rejection_reasons": dict(evidence_reject),
        "evidence_type_counts": dict(evidence_types),
        "candidate_scores": candidate_rows,
        "candidate_raw_total": candidate_raw,
        "candidate_filtered_total": candidate_raw - candidate_after_filter,
        "candidate_operator_counts": dict(operators),
        "exploration_latency_rows": exploration_rows,
        "avg_exploration_latency_ms": mean(exploration_total_ms),
        "avg_root_state_fetch_ms": mean(root_state_fetch_ms),
        "avg_root_a11y_ms": mean(root_a11y_ms),
        "avg_branch_a11y_ms": mean(branch_a11y_ms),
        "rollback_events": rollback_events,
        "rollback_event_failures": rollback_failures,
        "rollback_level2_triggers": dict(level2),
        "rendered_examples": rendered_examples,
    }


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except Exception:
        return "n/a"


def _ms(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.0f}"
    except Exception:
        return "n/a"


def _make_summary_row(root: Path, variant: str) -> dict[str, Any]:
    variant_dir = root / variant
    summary_path, summary = _latest_summary(variant_dir)
    driver_tasks = _parse_driver(variant_dir / "driver.log")
    tasks = _summary_tasks(summary, driver_tasks)
    episodes = summary.get("episodes") if isinstance(summary.get("episodes"), dict) else {}
    actions = summary.get("actions") if isinstance(summary.get("actions"), dict) else {}
    diagnostics = summary.get("diagnostics") if isinstance(summary.get("diagnostics"), dict) else {}
    success = sum(1 for value in tasks.values() if value.get("success"))
    total = len(tasks)
    return {
        "variant": variant,
        "summary_path": str(summary_path) if summary_path else "",
        "trace_roots": [str(p.parent.parent / "traces") for p in [summary_path] if p],
        "tasks": tasks,
        "success": episodes.get("success", success),
        "total": episodes.get("total", total),
        "success_rate": episodes.get("success_rate_all_trials")
        if episodes.get("success_rate_all_trials") is not None
        else (success / total if total else None),
        "avg_steps": episodes.get("avg_episode_length_complete")
        if episodes.get("avg_episode_length_complete") is not None
        else (sum(int(t.get("steps") or 0) for t in tasks.values()) / total if total else None),
        "avg_step_latency_ms": _driver_latency(driver_tasks) or actions.get("avg_step_latency_ms"),
        "avg_exploration_latency_ms": actions.get("avg_exploration_latency_ms"),
        "diagnostics": diagnostics,
    }


def _make_d3_row(root: Path) -> dict[str, Any]:
    main_dir = root / D3_MAIN
    remainder_dir = root / D3_REMAINDER
    main_tasks = _parse_driver(main_dir / "driver.log")
    remainder_tasks = _parse_driver(remainder_dir / "driver.log")
    tasks = dict(main_tasks)
    tasks.update(remainder_tasks)
    trace_roots = [
        main_dir / "run_20260601T210044" / "traces",
        remainder_dir / "run_20260601T211111" / "traces",
    ]
    diagnostics = _aggregate_trace_roots(trace_roots)
    success = sum(1 for value in tasks.values() if value.get("success"))
    total = len(tasks)
    total_steps = sum(int(t.get("steps") or 0) for t in tasks.values())
    return {
        "variant": D3_MERGED,
        "summary_path": "",
        "trace_roots": [str(p) for p in trace_roots],
        "ignored_invalid_run": str(main_dir / "run_20260601T204148"),
        "tasks": tasks,
        "success": success,
        "total": total,
        "success_rate": success / total if total else None,
        "avg_steps": total_steps / total if total else None,
        "avg_step_latency_ms": _driver_latency(tasks),
        "avg_exploration_latency_ms": diagnostics.get("avg_exploration_latency_ms"),
        "diagnostics": diagnostics,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_phase1_causal_debug.py <phase1_root>")
        return 2
    root = Path(sys.argv[1]).expanduser().resolve()
    rows = [_make_summary_row(root, variant) for variant in SUMMARY_VARIANTS]
    rows.append(_make_d3_row(root))

    task_union = [task for task in TASK_ORDER if any(task in row["tasks"] for row in rows)]
    lines = [
        "# Phase 1 causal debug report",
        "",
        "## 结论",
        "",
        "- 目前不建议进入 50-task 正式实验。探索版没有稳定收益，主要问题是候选质量和 rollback/timing side effects。",
        "- `D2_STRICT_INJECTION` 的 prompt 注入次数为 0，因此 D2 相对 baseline 的退化不能归因于 prompt augmentation。",
        "- `D3_RELAXED_DIAGNOSTIC_INJECTION_MERGED` 有 2 次实际渲染/注入，但注入内容来自 Joplin 状态栏/容器类噪声，不是有用证据。",
        "- `D0_EXPLORATION_SHADOW_ONLY` 已经从 3/5 掉到 2/5，说明即使不执行真实探索、不注入 prompt，wrapper/timing/VLM 随机性也足以改变结果。",
        "- `D1_REAL_EXPLORATION_NO_INJECTION` 成功率回到 3/5，但平均 step latency 从 baseline 的约 6.36s 增加到约 10.29s。",
        "",
        "## Cross-variant summary",
        "",
        "| variant | success | avg steps | avg step latency ms | avg exploration ms | prompt rows | injected | rendered | evidence rows | rollback events | rollback failures | level2 triggers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        diagnostics = row["diagnostics"]
        lines.append(
            f"| `{row['variant']}` | {row['success']}/{row['total']} ({_pct(row['success_rate'])}) | "
            f"{row['avg_steps']:.2f} | {_ms(row['avg_step_latency_ms'])} | {_ms(row.get('avg_exploration_latency_ms'))} | "
            f"{diagnostics.get('prompt_hint_decisions', 0)} | {diagnostics.get('prompt_hints_injected', 0)} | "
            f"{diagnostics.get('prompt_hints_rendered', diagnostics.get('prompt_hints_injected', 0))} | "
            f"{diagnostics.get('evidence_decisions', 0)} | {diagnostics.get('rollback_events', 0)} | "
            f"{diagnostics.get('rollback_event_failures', 0)} | `{diagnostics.get('rollback_level2_triggers', {})}` |"
        )

    lines.extend(
        [
            "",
            "## Per-task result matrix",
            "",
            "| task | " + " | ".join(f"`{row['variant']}`" for row in rows) + " |",
            "| --- | " + " | ".join("---:" for _ in rows) + " |",
        ]
    )
    for task in task_union:
        lines.append(
            "| `"
            + task
            + "` | "
            + " | ".join(row["tasks"].get(task, {}).get("result", "-") for row in rows)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Baseline 成功但探索失败的任务",
            "",
            "- `SportsTrackerActivityDuration`: baseline 成功，D0 失败，D1 成功，D2 失败，D3 失败。这说明该任务对 timing/探索/rollback 扰动都敏感，不能只看注入。",
            "- D3 remainder 中 SportsTracker 的失败路径是搜索 `October 15` 后误判没有记录，并返回自然语言失败解释；目标要求单个整数分钟。",
            "- `SimpleCalendarNextMeetingWithPerson`: baseline 本身失败，不属于 baseline success -> exploration fail，但 D3 中探索仍大量点 `simple` 月历网格候选，候选过滤还没解决根因。",
            "",
            "## Prompt injection diagnosis",
            "",
        ]
    )
    for row in rows:
        diagnostics = row["diagnostics"]
        lines.append(f"### {row['variant']}")
        lines.append("")
        lines.append(f"- prompt rows: `{diagnostics.get('prompt_hint_decisions', 0)}`")
        lines.append(f"- injected flag count: `{diagnostics.get('prompt_hints_injected', 0)}`")
        lines.append(
            f"- rendered prompt count: `{diagnostics.get('prompt_hints_rendered', diagnostics.get('prompt_hints_injected', 0))}`"
        )
        lines.append(f"- prompt rejection reasons: `{diagnostics.get('prompt_hint_rejection_reasons', {})}`")
        lines.append(f"- evidence type counts: `{diagnostics.get('evidence_type_counts', {})}`")
        lines.append(f"- evidence rejection reasons: `{diagnostics.get('evidence_rejection_reasons', {})}`")
        examples = diagnostics.get("rendered_examples") or diagnostics.get("injected_examples") or []
        if examples:
            lines.append(f"- rendered examples: `{examples}`")
        if row.get("summary_path"):
            lines.append(f"- summary: `{row['summary_path']}`")
        if row.get("trace_roots"):
            lines.append(f"- trace roots: `{row['trace_roots']}`")
        if row.get("ignored_invalid_run"):
            lines.append(f"- ignored invalid run: `{row['ignored_invalid_run']}`")
        lines.append("")

    lines.extend(
        [
            "## Latency diagnosis",
            "",
            "- Baseline driver 平均 step latency: about `6361 ms`。",
            "- D1 real exploration without injection: about `10291 ms`, exploration itself平均约 `4064 ms`。",
            "- D2 strict injection: about `9818 ms`, exploration itself平均约 `3889 ms`。",
            "- D3 merged: about `12563 ms`; D3 trace 中 root a11y/state fetch 往往在 1s 级，Calendar/Joplin 分支放大后会到 20s+ 每步。",
            "",
            "## Causal attribution",
            "",
            "- `D0 != B0`: baseline vs shadow-only 已经不同，说明实验必须多 seed 或多轮重复，否则 5-task 结论不稳。",
            "- `D1 ~= B0` success but slower: 真实探索主要带来 latency 成本，没有可靠改善。",
            "- `D2 < D1` while injected=0: 不是 prompt 证据造成，而是 run variance、rollback/探索 side effect 或同批运行状态差异。",
            "- `D3` 有实际注入但内容低质：放宽注入只证明 prompt 通路能工作，没证明 evidence capsule 有价值。",
            "",
            "## 下一步建议",
            "",
            "- 先不要增加 search attempts；当前瓶颈不是 a11y tree 速度，而是候选目标错误和每个分支的 rollback/状态获取放大。",
            "- 先做候选去重和语义过滤：去掉 app-name-only、statusbar/container、月历空白 grid cell；Calendar 需要优先 Search/FAB/filter 而不是日期空格。",
            "- 对 SportsTracker 单独加 negative evidence：搜索无结果不是任务完成证据，不能注入或强化“无记录”。",
            "- 修好候选质量后，先跑同一 5-task 的 3 次重复，再决定是否跑 50-task。",
            "",
        ]
    )

    out = root / "phase1_causal_debug_report_cn.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    machine = root / "phase1_causal_debug_report.json"
    machine.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    print(machine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
