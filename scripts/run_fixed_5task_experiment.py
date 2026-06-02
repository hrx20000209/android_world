#!/usr/bin/env python3
"""Run the fixed 5-task MobileExplorer framework validation experiment."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "results" / "fixed_5task_search"
TASKS = [
    "ExpenseDeleteSingle",
    "MarkorDeleteNewestNote",
    "NotesRecipeIngredientCount",
    "SimpleCalendarEventsInTimeRange",
    "SportsTrackerActivityDuration",
]

_strategy_path = REPO_ROOT / "scripts" / "run_search_strategy_comparison.py"
_spec = importlib.util.spec_from_file_location("strategy_runner", _strategy_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not import {_strategy_path}")
strategy_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strategy_runner)


COMMON_BASE = [
    "--suite_family=android_world",
    "--n_task_combinations=1",
    "--fixed_task_seed",
    "--image_downsample_scale=1.0",
    "--a11y_method=uiautomator",
    "--baseline_table=results/4B_2.txt",
]

CURRENT_CONTROL_ARGS = [
    "--agent_name=explore_agent_gelab",
    *COMMON_BASE,
    "--no-explore_diagnostic_full",
    "--explore_force_every_step",
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
    "--no-explore_safe_click_only",
    "--no-explore_skip_launcher",
    "--no-explore_filter_launcher_relevance",
    "--no-explore_skip_destructive_goals",
    "--explore_strategy=dfs",
    "--explore_hint_policy=current",
    "--explore_search_policy=current",
    "--explore_search_strategy=greedy",
    "--explore_rollback_policy=current",
    "--no-explore_fixed_framework",
]

FIXED_SHARED_ARGS = [
    "--agent_name=explore_agent_gelab",
    *COMMON_BASE,
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
    "--explore_rollback_policy=improved",
    "--explore_fixed_framework",
]

VARIANTS: dict[str, dict[str, Any]] = {
    "V0_BASELINE": {
        "description": "No exploration baseline.",
        "args": ["--agent_name=gelab_agent", *COMMON_BASE],
    },
    "V1_CURRENT_CONTROL": {
        "description": "Current exploration control without shared fixes.",
        "args": [*CURRENT_CONTROL_ARGS, "--explore_variant=V1_CURRENT_CONTROL"],
    },
    "V2_FIXED_SHARED_FRAMEWORK_GREEDY": {
        "description": "Fixed shared framework with greedy search.",
        "args": [
            *FIXED_SHARED_ARGS,
            "--explore_search_strategy=greedy",
            "--explore_variant=V2_FIXED_SHARED_FRAMEWORK_GREEDY",
        ],
    },
    "V3_STRATIFIED_BFS": {
        "description": "Fixed shared framework + stratified BFS.",
        "args": [*FIXED_SHARED_ARGS, "--explore_search_strategy=stratified_bfs", "--explore_variant=V3_STRATIFIED_BFS"],
    },
    "V4_ITERATIVE_DEEPENING_DFS": {
        "description": "Fixed shared framework + IDDFS.",
        "args": [*FIXED_SHARED_ARGS, "--explore_search_strategy=iddfs", "--explore_variant=V4_ITERATIVE_DEEPENING_DFS"],
    },
    "V5_EVIDENCE_BEST_FIRST": {
        "description": "Fixed shared framework + evidence best-first.",
        "args": [*FIXED_SHARED_ARGS, "--explore_search_strategy=best_first", "--explore_variant=V5_EVIDENCE_BEST_FIRST"],
    },
    "V6_BEAM_SEARCH": {
        "description": "Fixed shared framework + beam search.",
        "args": [*FIXED_SHARED_ARGS, "--explore_search_strategy=beam", "--explore_variant=V6_BEAM_SEARCH"],
    },
    "V7_LIGHTWEIGHT_MCTS_UCT": {
        "description": "Fixed shared framework + lightweight MCTS/UCT.",
        "args": [*FIXED_SHARED_ARGS, "--explore_search_strategy=mcts", "--explore_variant=V7_LIGHTWEIGHT_MCTS_UCT"],
    },
}


def _run_variant(label: str, root: Path, extra_args: list[str]) -> Path:
    variant_root = root / label
    variant_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_exploration_experiment_report.py"),
        "--run",
        f"--tasks={','.join(TASKS)}",
        f"--experiment_root={variant_root}",
        "--max_cases=16",
        *VARIANTS[label]["args"],
        *extra_args,
    ]
    log_path = variant_root / "driver.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[{label}] {line}", end="")
            log.write(line)
        rc = proc.wait()
        if rc != 0:
            print(f"[{label}] exited with {rc}; partial traces will be analyzed if available")
    runs = sorted(variant_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise RuntimeError(f"No run directory produced for {label}")
    return runs[0]


def _extract_events(run_dirs: dict[str, Path], out_dir: Path) -> None:
    rollback_out = out_dir / "rollback_events.jsonl"
    hints_out = out_dir / "prompt_hints.jsonl"
    with rollback_out.open("w", encoding="utf-8") as rb_f, hints_out.open("w", encoding="utf-8") as hint_f:
        for label, run_dir in run_dirs.items():
            trace_root = run_dir / "traces"
            for path in sorted(trace_root.rglob("rollback_trace.jsonl")):
                for item in strategy_runner._read_jsonl(path):  # pylint: disable=protected-access
                    item["variant"] = label
                    item["trace_file"] = str(path)
                    rb_f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            for path in sorted(trace_root.rglob("action.jsonl")):
                for item in strategy_runner._read_jsonl(path):  # pylint: disable=protected-access
                    if item.get("prompt_hint") or item.get("matched_exploration_results"):
                        item["variant"] = label
                        item["trace_file"] = str(path)
                        hint_f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def _copy_outputs(root: Path) -> None:
    mapping = {
        "strategy_results_summary.json": "fixed_5task_summary.json",
        "per_task_strategy_results.csv": "per_task_results.csv",
        "per_step_strategy_metrics.csv": "per_step_metrics.csv",
        "strategy_comparison.md": "fixed_5task_comparison_base.md",
    }
    for src_name, dst_name in mapping.items():
        src = root / src_name
        if src.exists():
            shutil.copyfile(src, root / dst_name)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _variant(summary: dict[str, Any], label: str) -> dict[str, Any]:
    return dict(summary.get("strategies", {}).get(label, {}))


def _safety_ok(summary: dict[str, Any], per_task: list[dict[str, str]]) -> tuple[bool, list[str]]:
    v1 = _variant(summary, "V1_CURRENT_CONTROL")
    v2 = _variant(summary, "V2_FIXED_SHARED_FRAMEWORK_GREEDY")
    reasons: list[str] = []
    if not v2:
        return False, ["V2 result missing"]
    if int(v2.get("evidence_only_vague_hint_count") or 0) != 0:
        reasons.append("vague evidence-only hint count is not zero")
    hint_count = sum(int(v2.get(k) or 0) for k in ("ACTION_HINT_count", "ANSWER_HINT_count", "AVOID_HINT_count", "SCHEMA_HINT_count", "RISK_HINT_count"))
    if hint_count <= 0:
        reasons.append("no strict hint was injected")
    if v1 and int(v2.get("rollback_induced_WAIT_count") or 0) >= int(v1.get("rollback_induced_WAIT_count") or 0):
        reasons.append("rollback-induced WAIT was not lower than V1")
    for row in per_task:
        if row.get("strategy") == "V2_FIXED_SHARED_FRAMEWORK_GREEDY" and row.get("task_mode") == "DELETE_COMMIT":
            if row.get("baseline_success") == "1.0" and row.get("strategy_success") != "1.0":
                reasons.append(f"DELETE_COMMIT baseline-success task broken: {row.get('task_id')}")
    return not reasons, reasons


def _write_cn_report(root: Path, summary: dict[str, Any], safety_reasons: list[str], ran_all: bool) -> None:
    rows = _read_csv(root / "per_task_results.csv")
    lines: list[str] = [
        "# Fixed 5-task MobileExplorer 搜索框架验证报告",
        "",
        f"- 输出目录: `{root}`",
        f"- 是否运行 S1-S5: `{ran_all}`",
        "- 本报告用于判断 fixed shared framework 是否足够安全，是否值得进入 20-task。",
        "",
        "## 总览",
        "",
        "| variant | success | avg steps | delta | rescued | broken | hints | action | answer | avoid | schema | risk | hint follow | rollback fail | WAIT | suppress | max depth |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, item in summary.get("strategies", {}).items():
        hints = sum(int(item.get(k) or 0) for k in ("ACTION_HINT_count", "ANSWER_HINT_count", "AVOID_HINT_count", "SCHEMA_HINT_count", "RISK_HINT_count"))
        lines.append(
            f"| `{label}` | {float(item.get('success_rate') or 0)*100:.1f}% | {float(item.get('avg_episode_length') or 0):.2f} | "
            f"{float(item.get('paired_avg_step_delta') or 0):.2f} | {item.get('baseline_failure_rescued', 0)} | "
            f"{item.get('baseline_success_broken', 0)} | {hints} | {item.get('ACTION_HINT_count', 0)} | "
            f"{item.get('ANSWER_HINT_count', 0)} | {item.get('AVOID_HINT_count', 0)} | {item.get('SCHEMA_HINT_count', 0)} | "
            f"{item.get('RISK_HINT_count', 0)} | {float(item.get('hint_follow_rate') or 0)*100:.1f}% | "
            f"{item.get('rollback_failures', 0)} | {item.get('rollback_induced_WAIT_count', 0)} | "
            f"{item.get('planned_action_suppressions', 0)} | {item.get('max_depth_reached', 0)} |"
        )
    lines.extend(["", "## 图表", ""])
    for name in [
        "success_rate_by_strategy.png",
        "avg_steps_by_strategy.png",
        "step_delta_V2_FIXED_SHARED_FRAMEWORK_GREEDY.png",
        "broken_vs_rescued_by_strategy.png",
        "evidence_yield_by_strategy.png",
        "hint_follow_rate_by_strategy.png",
        "rollback_failure_rate_by_strategy.png",
        "max_depth_by_strategy.png",
    ]:
        path = root / "plots" / name
        if path.exists():
            lines.append(f"![{name}]({path})")
            lines.append("")
    v1 = _variant(summary, "V1_CURRENT_CONTROL")
    v2 = _variant(summary, "V2_FIXED_SHARED_FRAMEWORK_GREEDY")
    lines.extend(
        [
            "## 必答问题",
            "",
            f"1. V2 是否降低 rollback-induced WAIT: `{int(v2.get('rollback_induced_WAIT_count') or 0) < int(v1.get('rollback_induced_WAIT_count') or 0) if v1 and v2 else 'NA'}`。",
            f"2. 是否消除 vague evidence-only hints: `{int(v2.get('evidence_only_vague_hint_count') or 0) == 0 if v2 else 'NA'}`。",
            f"3. 是否产生严格 hint: `{sum(int(v2.get(k) or 0) for k in ('ACTION_HINT_count','ANSWER_HINT_count','AVOID_HINT_count','SCHEMA_HINT_count','RISK_HINT_count')) if v2 else 0}`。",
            f"4. 是否有 hint 被 follow: `{float(v2.get('hint_follow_rate') or 0) > 0 if v2 else 'NA'}`。",
            f"5. 是否避免破坏 DELETE_COMMIT: `{not any(r.get('strategy') == 'V2_FIXED_SHARED_FRAMEWORK_GREEDY' and r.get('task_mode') == 'DELETE_COMMIT' and r.get('baseline_success') == '1.0' and r.get('strategy_success') != '1.0' for r in rows)}`。",
            "6. rescued task: " + ", ".join(r.get("task_id", "") for r in rows if r.get("strategy") == "V2_FIXED_SHARED_FRAMEWORK_GREEDY" and r.get("baseline_failure_rescued") == "True") or "无。",
            "7. broken baseline-success task: " + ", ".join(r.get("task_id", "") for r in rows if r.get("strategy") == "V2_FIXED_SHARED_FRAMEWORK_GREEDY" and r.get("baseline_success_broken") == "True") or "无。",
            "8. gate 是否禁用 obvious delete/commit step: 看 `search_tree_traces.jsonl` 中 fixed_gate_disabled:DELETE_COMMIT；若 DELETE 任务 exploration_attempts 为 0 或仅早期安全探测，则满足。",
            "9. RiskBoundary 是否阻止风险 speculative execution: fixed framework 中 transaction unsafe candidate 不执行，只记录 risk capsule；详见 `evidence_capsules.jsonl`。",
            f"10. 现在是否值得比较 S1-S5: `{ran_all}`。",
            "11-16. 若未运行 S1-S5，则暂不回答策略排名；原因见安全检查。",
            "",
            "## 安全检查",
            "",
        ]
    )
    if safety_reasons:
        lines.append("- V2 未通过安全门槛：")
        for reason in safety_reasons:
            lines.append(f"  - {reason}")
    else:
        lines.append("- V2 通过安全门槛，已继续运行 S1-S5。")
    lines.extend(
        [
            "",
            "## Per-task 明细",
            "",
            "| variant | task | mode | baseline | variant | base steps | variant steps | delta | rescued | broken | attempts | hints | rollback fail | WAIT | max depth |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        hints = sum(int(row.get(k) or 0) for k in ("ACTION_HINT_count", "ANSWER_HINT_count", "AVOID_HINT_count", "SCHEMA_HINT_count", "RISK_HINT_count"))
        lines.append(
            f"| `{row.get('strategy')}` | `{row.get('task_id')}` | `{row.get('task_mode')}` | {row.get('baseline_success')} | "
            f"{row.get('strategy_success')} | {row.get('baseline_steps')} | {row.get('strategy_steps')} | {row.get('step_delta')} | "
            f"{row.get('baseline_failure_rescued')} | {row.get('baseline_success_broken')} | {row.get('exploration_attempts',0)} | "
            f"{hints} | {row.get('rollback_failures',0)} | {row.get('rollback_induced_WAIT_count',0)} | {row.get('max_depth_reached',0)} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
        ]
    )
    if safety_reasons:
        lines.append("- 不建议进入 20-task。本轮应先修复上面列出的安全门槛失败项。")
    else:
        lines.append("- 可以进入 20-task，但仍需重点观察 hint follow 和 rollback failure。")
    (root / "fixed_5task_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--analyze_only", default="")
    parser.add_argument("--variants", default="")
    parser.add_argument("--extra_arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.analyze_only:
        root = Path(args.analyze_only).expanduser().resolve()
    else:
        root = Path(args.experiment_root).expanduser().resolve() / f"run_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "tasks.txt").write_text("\n".join(TASKS) + "\n", encoding="utf-8")

    labels = [x.strip() for x in args.variants.split(",") if x.strip()]
    if not labels:
        labels = ["V0_BASELINE", "V1_CURRENT_CONTROL", "V2_FIXED_SHARED_FRAMEWORK_GREEDY"]

    run_dirs: dict[str, Path] = {}
    if args.analyze_only:
        for label in labels:
            runs = sorted((root / label).glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if runs:
                run_dirs[label] = runs[0]
    else:
        for label in labels:
            run_dirs[label] = _run_variant(label, root, args.extra_arg)
            if label == "V2_FIXED_SHARED_FRAMEWORK_GREEDY":
                summary = strategy_runner._analyze(run_dirs, root)  # pylint: disable=protected-access
                _copy_outputs(root)
                per_task = _read_csv(root / "per_task_results.csv")
                ok, reasons = _safety_ok(summary, per_task)
                if not ok:
                    _extract_events(run_dirs, root)
                    _write_cn_report(root, summary, reasons, ran_all=False)
                    print("[fixed_5task] V2 safety failed; stopping before S1-S5")
                    print(json.dumps(reasons, ensure_ascii=False, indent=2))
                    print(f"[fixed_5task] root={root}")
                    return 0
                for next_label in [
                    "V3_STRATIFIED_BFS",
                    "V4_ITERATIVE_DEEPENING_DFS",
                    "V5_EVIDENCE_BEST_FIRST",
                    "V6_BEAM_SEARCH",
                    "V7_LIGHTWEIGHT_MCTS_UCT",
                ]:
                    run_dirs[next_label] = _run_variant(next_label, root, args.extra_arg)
                break

    summary = strategy_runner._analyze(run_dirs, root)  # pylint: disable=protected-access
    _copy_outputs(root)
    _extract_events(run_dirs, root)
    per_task = _read_csv(root / "per_task_results.csv")
    ok, reasons = _safety_ok(summary, per_task)
    ran_all = all(label in run_dirs for label in ["V3_STRATIFIED_BFS", "V7_LIGHTWEIGHT_MCTS_UCT"])
    _write_cn_report(root, summary, [] if ok else reasons, ran_all=ran_all)
    print(json.dumps(summary.get("strategies", {}), ensure_ascii=False, indent=2))
    print(f"[fixed_5task] root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
