#!/usr/bin/env python3
"""Run Phase A high-budget exploration upper-bound validation."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = REPO_ROOT / "scripts" / "run_exploration_experiment_report.py"
PYTHON_BIN = sys.executable

PHASE_A_TASKS = [
    "NotesRecipeIngredientCount",
    "NotesIsTodo",
    "FilesDeleteFile",
    "SportsTrackerActivityDuration",
    "SportsTrackerTotalDistanceForCategoryOverInterval",
    "SimpleCalendarNextMeetingWithPerson",
    "SimpleCalendarEventsInTimeRange",
    "ClockStopWatchRunning",
    "MarkorCreateNote",
    "ExpenseDeleteMultiple2",
    "BrowserMaze",
    "ContactsNewContactDraft",
]

SEARCH_VARIANT_CONFIGS = {
    "S1_BFS": {"explore_strategy": "bfs", "search_strategy": "stratified_bfs", "safe_mcts": False},
    "S2_DFS": {"explore_strategy": "dfs", "search_strategy": "iddfs", "safe_mcts": False},
    "S3_BEAM": {"explore_strategy": "dfs", "search_strategy": "beam", "safe_mcts": False},
    "S4_MCTS": {"explore_strategy": "dfs", "search_strategy": "mcts", "safe_mcts": True},
    "S5_OPERATOR_STRATIFIED_BEST_FIRST": {
        "explore_strategy": "dfs",
        "search_strategy": "best_first",
        "safe_mcts": False,
    },
}

REQUIRED_FILES = [
    "runtime_config.yaml",
    "frozen_task_selection.md",
    "per_task_results.csv",
    "per_step_metrics.csv",
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
    "rollback_level2_cases.md",
    "rollback_failure_cases.md",
    "shortcut_plans.jsonl",
    "shortcut_shadow_eval.jsonl",
    "state_acquisition_metrics.csv",
    "step_decoupling_status.jsonl",
]


def _latest_run(variant_root: Path) -> Path | None:
    runs = sorted(variant_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _write_task_selection(run_root: Path, tasks: list[str]) -> None:
    lines = [
        "# Phase A Frozen Task Selection",
        "",
        "- task_count: 12",
        "- apps: Notes/Joplin, Files, SportsTracker, Calendar, Clock, Markor, Expense, Browser, Contacts",
        "- rule: max 2 tasks per app",
        "",
        "| # | task | role |",
        "| ---: | --- | --- |",
    ]
    roles = [
        "Joplin recipe query",
        "Joplin boolean/status query",
        "Files navigation/delete boundary",
        "SportsTracker duration query",
        "SportsTracker distance query",
        "Calendar person/event query",
        "Calendar time-range query",
        "Clock stopwatch/timer",
        "Markor create/edit/open note",
        "Expense delete/query boundary",
        "Browser maze/draw/multiply",
        "Contacts/settings/audio style data entry",
    ]
    for idx, task in enumerate(tasks, start=1):
        lines.append(f"| {idx} | `{task}` | {roles[idx - 1]} |")
    text = "\n".join(lines) + "\n"
    (run_root / "frozen_task_selection.md").write_text(text, encoding="utf-8")
    (run_root / "task_selection.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def _common_args(variant_name: str, tasks_str: str, variant_root: Path, max_cases: int) -> list[str]:
    return [
        PYTHON_BIN,
        str(REPORT_SCRIPT),
        "--run",
        "--suite_family=android_world",
        "--tasks=" + tasks_str,
        "--n_task_combinations=1",
        "--task_random_seed=43",
        "--fixed_task_seed",
        "--image_downsample_scale=1.0",
        "--baseline_table=results/4B_2.txt",
        "--experiment_root=" + str(variant_root),
        f"--max_cases={max_cases}",
        "--a11y_method=fast_provider",
        "--a11y_preflight_timeout=90",
        "--latency_profile",
        "--agent_name=explore_agent_gelab",
        "--explore_variant=" + variant_name,
        "--decoupled_exploration",
        "--exploration_timing=parallel_shadow",
        "--no-explore_use_current_action",
        "--no-explore_use_planning_text",
        "--no-explore_planned_only",
        "--explore_decouple_planned",
        "--explore_parallel_vlm",
        "--explore_parallel_lookahead",
        "--t2_mode=shadow",
        "--no-t2_allow_safe_search_input",
    ]


def _variant_args(variant_name: str, tasks_str: str, variant_root: Path, max_cases: int) -> list[str]:
    cmd = _common_args(variant_name, tasks_str, variant_root, max_cases)
    if variant_name == "B0_BASELINE_RERUN":
        cmd.extend(
            [
                "--no-explore_enable",
                "--explore_max_runs=0",
                "--explore_max_step=0",
                "--explore_branch_budget=0",
                "--explore_min_attempts_per_step=0",
                "--explore_branch_depth=0",
                "--no-explore_enable_t2_lookahead",
            ]
        )
        return cmd
    strategy_config = SEARCH_VARIANT_CONFIGS.get(
        variant_name,
        {"explore_strategy": "dfs", "search_strategy": "best_first", "safe_mcts": False},
    )
    cmd.extend(
        [
            "--explore_enable",
            "--explore_max_runs=10000",
            "--explore_max_step=10000",
            "--explore_branch_budget=30",
            "--explore_min_attempts_per_step=30",
            "--explore_branch_depth=2",
            "--explore_back_limit=4",
            "--explore_replay_max_actions=6",
            "--explore_fallback_safe_candidates",
            "--explore_safe_click_only",
            "--explore_skip_launcher",
            "--explore_filter_launcher_relevance",
            "--no-explore_skip_destructive_goals",
            "--explore_fast_mode",
            "--explore_fast_state",
            "--explore_transaction_safe",
            "--explore_action_settle_s=0.25",
            "--explore_hint_policy=strict",
            "--explore_search_policy=task_gate",
            f"--explore_strategy={strategy_config['explore_strategy']}",
            f"--explore_search_strategy={strategy_config['search_strategy']}",
            "--explore_rollback_policy=improved",
            "--explore_fixed_framework",
            "--explore_answer_extractors",
            "--explore_slot_complete",
            "--explore_slot_policy_switcher",
            "--explore_enable_t2_lookahead",
            "--explore_lightweight_a11y_trace",
            "--explore_trace_a11y_limit=120",
            "--explore_upper_bound_evidence",
            "--trace_screenshot_mode=failure+level2+depth2+injected+sampled",
        ]
    )
    if strategy_config.get("safe_mcts"):
        cmd.append("--explore_safe_mcts")
    if variant_name == "U2_HIGH_BUDGET30_SUMMARY_ALL_DEBUG":
        cmd.extend(["--explore_summary_all_debug", "--explore_relaxed_diagnostic_injection"])
    return cmd


def _run_variant(run_root: Path, variant_name: str, tasks_str: str, max_cases: int) -> int:
    variant_root = run_root / variant_name
    variant_root.mkdir(parents=True, exist_ok=True)
    cmd = _variant_args(variant_name, tasks_str, variant_root, max_cases)
    log_path = variant_root / "driver.log"
    print(f"[phaseA] variant={variant_name}")
    print("[phaseA] cmd=" + " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=log, stderr=subprocess.STDOUT, check=False)
    run_dir = _latest_run(variant_root)
    manifest = {
        "variant": variant_name,
        "returncode": proc.returncode,
        "run_dir": str(run_dir) if run_dir else "",
        "command": cmd,
        "log": str(log_path),
    }
    (variant_root / "variant_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if run_dir:
        _materialize_variant_outputs(variant_root, run_dir, variant_name)
    print(f"[phaseA] variant={variant_name} returncode={proc.returncode} run_dir={run_dir}")
    return int(proc.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _concat_jsonl(run_dir: Path, filename: str, out_path: Path) -> int:
    rows = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for path in sorted((run_dir / "traces").rglob(filename)):
            if path == out_path:
                continue
            if path.parent == (run_dir / "traces"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text and not text.endswith("\n"):
                text += "\n"
            out.write(text)
            rows += sum(1 for line in text.splitlines() if line.strip())
    return rows


def _concat_markdown(run_dir: Path, filename: str, out_path: Path) -> None:
    chunks = []
    for path in sorted((run_dir / "traces").rglob(filename)):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            chunks.append(f"<!-- {path} -->\n\n{text}")
    out_path.write_text("\n\n".join(chunks) + ("\n" if chunks else "无。\n"), encoding="utf-8")


def _write_per_task_results(summary: dict[str, Any], out_path: Path) -> None:
    rows = list((summary.get("episodes") or {}).get("task_episode_rows") or [])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "episode_length", "success", "exception"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _materialize_variant_outputs(variant_root: Path, run_dir: Path, variant_name: str) -> None:
    report_dir = run_dir / "report"
    summary = _read_json(report_dir / "summary.json")
    if summary:
        shutil.copy2(report_dir / "summary.json", variant_root / "summary.json")
    if (report_dir / "step_metrics.csv").exists():
        shutil.copy2(report_dir / "step_metrics.csv", variant_root / "per_step_metrics.csv")
    else:
        (variant_root / "per_step_metrics.csv").write_text("", encoding="utf-8")
    if (report_dir / "exploration_experiment_report_cn.md").exists():
        shutil.copy2(report_dir / "exploration_experiment_report_cn.md", variant_root / "exploration_experiment_report_cn.md")
    _write_per_task_results(summary, variant_root / "per_task_results.csv")

    first_runtime = next(iter(sorted((run_dir / "traces").rglob("runtime_config.yaml"))), None)
    if first_runtime and first_runtime.exists():
        shutil.copy2(first_runtime, variant_root / "runtime_config.yaml")
    else:
        (variant_root / "runtime_config.yaml").write_text(
            f"variant: {json.dumps(variant_name)}\nrun_dir: {json.dumps(str(run_dir))}\n",
            encoding="utf-8",
        )
    for filename in REQUIRED_FILES:
        out_path = variant_root / filename
        if filename.endswith(".jsonl"):
            _concat_jsonl(run_dir, filename, out_path)
        elif filename in {"rollback_level2_cases.md", "rollback_failure_cases.md"}:
            _concat_markdown(run_dir, filename, out_path)
    state_metrics = next(iter(sorted((run_dir / "traces").rglob("state_acquisition_metrics.csv"))), None)
    if state_metrics and state_metrics.exists():
        shutil.copy2(state_metrics, variant_root / "state_acquisition_metrics.csv")


def _variant_summary(variant_root: Path) -> dict[str, Any]:
    data = _read_json(variant_root / "summary.json")
    data["_variant_root"] = str(variant_root)
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "<empty>")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


def _offline_replay(run_root: Path, variants: list[str]) -> Path:
    lines = [
        "# Offline Search Strategy Replay 中文报告",
        "",
        "高预算 trace 被当作 candidate universe；这里只做诊断性 replay，不重新执行 Android UI。",
        "",
        "| variant | strategy | K | useful_recall@K | action_recall@K | answer_recall@K | avoid_recall@K | risk_avoidance_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in variants:
        root = run_root / variant
        candidates = _read_jsonl(root / "candidate_scores.jsonl")
        useful = [
            row for row in _read_jsonl(root / "evidence_decisions.jsonl")
            if str(row.get("final_evidence_type") or "") in {"ANSWER_HINT", "ACTION_HINT", "AVOID_HINT", "SCHEMA_HINT", "RISK_HINT"}
            and str(row.get("rejected_reason") or "") not in {"no_supported_evidence_type", "rollback_not_verified"}
        ]
        if not useful:
            for strategy in ["BFS", "DFS", "Beam", "MCTS", "Operator-Stratified Best-First"]:
                for k in [5, 10, 15]:
                    lines.append(f"| {variant} | {strategy} | {k} | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |")
            continue
        by_step: dict[Any, list[dict[str, Any]]] = {}
        for cand in candidates:
            by_step.setdefault(cand.get("step"), []).append(cand)

        def rank(strategy: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if strategy == "BFS":
                return sorted(rows, key=lambda r: (int(r.get("selected_rank") or 9999), -float(r.get("final_score") or 0.0)))
            if strategy == "DFS":
                return sorted(rows, key=lambda r: (str(r.get("operator") or ""), int(r.get("selected_rank") or 9999)), reverse=True)
            if strategy == "Beam":
                return sorted(rows, key=lambda r: float(r.get("final_score") or 0.0), reverse=True)
            if strategy == "MCTS":
                return sorted(rows, key=lambda r: float(r.get("final_score") or 0.0) - 2.0 * float(r.get("RiskPenalty") or 0.0), reverse=True)
            groups: dict[str, list[dict[str, Any]]] = {}
            for row in sorted(rows, key=lambda r: float(r.get("final_score") or 0.0), reverse=True):
                groups.setdefault(str(row.get("operator") or "Other"), []).append(row)
            ordered = []
            while any(groups.values()):
                for op in sorted(groups):
                    if groups[op]:
                        ordered.append(groups[op].pop(0))
            return ordered

        useful_by_step: dict[Any, list[dict[str, Any]]] = {}
        for row in useful:
            useful_by_step.setdefault(row.get("step"), []).append(row)

        for strategy in ["BFS", "DFS", "Beam", "MCTS", "Operator-Stratified Best-First"]:
            for k in [5, 10, 15]:
                denom = max(1, len(useful))
                hit = action_hit = answer_hit = avoid_hit = 0
                risky_selected = 0
                selected_total = 0
                for step, useful_rows in useful_by_step.items():
                    selected = rank(strategy, by_step.get(step) or [])[:k]
                    selected_total += len(selected)
                    risky_selected += sum(1 for row in selected if float(row.get("RiskPenalty") or 0.0) > 0.0)
                    selected_labels = {str(row.get("label") or "").lower() for row in selected}
                    for item in useful_rows:
                        label = str(item.get("candidate_label") or "").lower()
                        if label and any(label in selected_label or selected_label in label for selected_label in selected_labels):
                            hit += 1
                            et = str(item.get("final_evidence_type") or "")
                            action_hit += int(et == "ACTION_HINT")
                            answer_hit += int(et == "ANSWER_HINT")
                            avoid_hit += int(et == "AVOID_HINT")
                risk_avoidance = 1.0 - (risky_selected / float(max(1, selected_total)))
                lines.append(
                    f"| {variant} | {strategy} | {k} | {hit / denom:.3f} | {action_hit / denom:.3f} | "
                    f"{answer_hit / denom:.3f} | {avoid_hit / denom:.3f} | {risk_avoidance:.3f} |"
                )
    out = run_root / "offline_strategy_replay_report_cn.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _write_phase_report(run_root: Path, variants: list[str]) -> Path:
    summaries = {variant: _variant_summary(run_root / variant) for variant in variants}
    rows = []
    for variant, summary in summaries.items():
        episodes = summary.get("episodes") or {}
        actions = summary.get("actions") or {}
        diagnostics = summary.get("diagnostics") or {}
        rows.append(
            {
                "variant": variant,
                "success": episodes.get("success"),
                "complete": episodes.get("complete"),
                "success_rate": episodes.get("success_rate_all_trials"),
                "avg_steps": episodes.get("avg_episode_length_complete"),
                "reasoning_steps": actions.get("total_steps"),
                "hint_steps": actions.get("prompt_hint_steps"),
                "hint_follow_rate": actions.get("prompt_hint_follow_rate"),
                "diagnostic_injected": diagnostics.get("prompt_hints_injected"),
                "decoupled_invalid": diagnostics.get("decoupled_invalid_steps"),
            }
        )

    baseline_success = float((summaries.get("B0_BASELINE_RERUN", {}).get("episodes") or {}).get("success_rate_all_trials") or 0.0)
    best_explore = max(
        float((summaries.get(v, {}).get("episodes") or {}).get("success_rate_all_trials") or 0.0)
        for v in variants
        if v != "B0_BASELINE_RERUN"
    )
    injected = sum(
        int(((summaries.get(v, {}).get("diagnostics") or {}).get("prompt_hints_injected") or 0))
        for v in variants
        if v != "B0_BASELINE_RERUN"
    )
    if injected <= 0:
        recommendation = "Instrumentation still broken; fix injection/alignment first."
    elif best_explore > baseline_success:
        recommendation = "High-budget evidence helps; proceed to strategy optimization."
    else:
        recommendation = "High-budget evidence does not help; redesign evidence summarization."

    lines = [
        "# Phase A 高预算探索 Upper-bound 验证报告",
        "",
        "## Executive Summary",
        "",
        f"- baseline success rate: `{baseline_success:.3f}`",
        f"- best high-budget success rate: `{best_explore:.3f}`",
        f"- injected hint count across U1/U2: `{injected}`",
        f"- recommendation: `{recommendation}`",
        "",
        "## Variant Comparison",
        "",
        "| variant | success | complete | success rate | avg steps | reasoning steps | hint steps | hint follow rate | diagnostic injected | decoupled invalid |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['success']} | {row['complete']} | {row['success_rate']} | "
            f"{row['avg_steps']} | {row['reasoning_steps']} | {row['hint_steps']} | {row['hint_follow_rate']} | "
            f"{row['diagnostic_injected']} | {row['decoupled_invalid']} |"
        )
    lines.extend(["", "## Prompt Evidence Examples", ""])
    for variant in variants:
        if variant == "B0_BASELINE_RERUN":
            continue
        root = run_root / variant
        hints = [row for row in _read_jsonl(root / "prompt_hints.jsonl") if row.get("injected")]
        rejected = [row for row in _read_jsonl(root / "evidence_decisions.jsonl") if not row.get("injected")]
        lines.extend([f"### {variant}", ""])
        if hints:
            for idx, hint in enumerate(hints[:3], start=1):
                lines.extend(
                    [
                        f"Injected example {idx}:",
                        "",
                        "```text",
                        str(hint.get("rendered_text") or hint.get("rendered_prompt_text") or "")[:1600],
                        "```",
                        "",
                    ]
                )
        else:
            lines.append("没有注入 hint；下面列出被拒绝证据样例。")
            lines.append("")
            for item in rejected[:5]:
                lines.append(
                    f"- step={item.get('step')} branch={item.get('branch_id')} type={item.get('final_evidence_type')} "
                    f"reason={item.get('rejected_reason')} candidate=`{item.get('candidate_label')}`"
                )
            lines.append("")

    lines.extend(["## Rollback", ""])
    for variant in variants:
        root = run_root / variant
        events = _read_jsonl(root / "rollback_events.jsonl")
        level2 = [row for row in events if row.get("level2_triggered")]
        failures = [row for row in events if not bool(row.get("success", True))]
        lines.append(
            f"- `{variant}` rollback_events={len(events)}, level2={len(level2)}, failures={len(failures)}, "
            f"level2_reasons={_counts(level2, 'level2_trigger_reason')}"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- frozen task selection: `{run_root / 'frozen_task_selection.md'}`",
            f"- offline replay: `{run_root / 'offline_strategy_replay_report_cn.md'}`",
        ]
    )
    for variant in variants:
        lines.append(f"- `{variant}` artifacts: `{run_root / variant}`")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            recommendation,
            "",
        ]
    )
    out = run_root / "phaseA_high_budget_upper_bound_report_cn.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", default=str(REPO_ROOT / "results" / "phaseA_high_budget_upper_bound"))
    parser.add_argument(
        "--variants",
        default="B0_BASELINE_RERUN,S1_BFS,S2_DFS,S3_BEAM,S4_MCTS,S5_OPERATOR_STRATIFIED_BEST_FIRST",
    )
    parser.add_argument("--max_cases", type=int, default=12)
    args = parser.parse_args()

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_root = Path(args.experiment_root).expanduser().resolve() / f"run_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    _write_task_selection(run_root, PHASE_A_TASKS)
    tasks_str = ",".join(PHASE_A_TASKS)
    rc = 0
    for variant in variants:
        rc |= _run_variant(run_root, variant, tasks_str, int(args.max_cases))
    _offline_replay(run_root, [v for v in variants if v != "B0_BASELINE_RERUN"])
    report = _write_phase_report(run_root, variants)
    print(f"[phaseA] report={report}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
