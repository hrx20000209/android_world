#!/usr/bin/env python3
"""Run 30/50-task final strategy validation for MobileExplorer."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = REPO_ROOT / "scripts" / "run_exploration_experiment_report.py"
BASELINE_TABLE = REPO_ROOT / "results" / "4B_2.txt"
PYTHON_BIN = sys.executable


def _parse_baseline_tasks(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("task_num") or line == "task":
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            task_num = int(float(parts[1]))
        except ValueError:
            continue
        rows.append(
            {
                "task_id": parts[0],
                "task_num": task_num,
                "baseline_success_rate_prior": float(parts[3]) if parts[3] != "NaN" else None,
                "baseline_episode_length_prior": float(parts[4]) if parts[4] != "NaN" else None,
                "baseline_step_latency_s_prior": float(parts[6]) if parts[6] != "NaN" else None,
            }
        )
    rows.sort(key=lambda item: int(item["task_num"]))
    return rows[:limit]


def _latest_run(variant_root: Path) -> Path | None:
    runs = sorted(variant_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _run_variant(variant_root: Path, variant_name: str, tasks_str: str, args: list[str]) -> int:
    variant_root.mkdir(parents=True, exist_ok=True)
    cmd = [
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
        "--max_cases=50",
        "--a11y_method=fast_provider",
        "--a11y_preflight_timeout=90",
        "--latency_profile",
        "--explore_variant=" + variant_name,
    ]
    cmd.extend(args)
    log_path = variant_root / "driver.log"
    print(f"[50-run] variant={variant_name}")
    print("[50-run] cmd=" + " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    run_dir = _latest_run(variant_root)
    manifest = {
        "variant": variant_name,
        "returncode": proc.returncode,
        "run_dir": str(run_dir) if run_dir else "",
        "command": cmd,
        "log": str(log_path),
    }
    (variant_root / "variant_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[50-run] variant={variant_name} returncode={proc.returncode} run_dir={run_dir}")
    return int(proc.returncode)


def _load_summary(variant_root: Path) -> dict[str, Any]:
    run_dir = _latest_run(variant_root)
    if run_dir is None:
        return {}
    summary_path = run_dir / "report" / "summary.json"
    if not summary_path.exists():
        return {"run_dir": str(run_dir), "missing_summary": True}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    data["_run_dir"] = str(run_dir)
    data["_report"] = str(
        run_dir / "report" / "phaseC_final_strategy_30_or_50task_report_cn.md"
    )
    return data


def _write_comparison(run_root: Path, tasks: list[dict[str, Any]]) -> Path:
    baseline = _load_summary(run_root / "B0_BASELINE_RERUN")
    main = _load_summary(run_root / "FINAL_STRATEGY")
    comparison = {
        "run_root": str(run_root),
        "tasks": tasks,
        "baseline": baseline,
        "exploration": main,
    }
    (run_root / "comparison_summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def ep(summary: dict[str, Any], key: str, default: Any = None) -> Any:
        return ((summary.get("episodes") or {}).get(key, default) if isinstance(summary, dict) else default)

    def act(summary: dict[str, Any], key: str, default: Any = None) -> Any:
        return ((summary.get("actions") or {}).get(key, default) if isinstance(summary, dict) else default)

    def exp(summary: dict[str, Any], key: str, default: Any = None) -> Any:
        return ((summary.get("exploration") or {}).get(key, default) if isinstance(summary, dict) else default)

    report = run_root / "phaseC_final_strategy_50task_comparison_report_zh.md"
    lines = [
        "# 50-task 阶段C：基线与最终策略对比",
        "",
        f"- run root: `{run_root}`",
        f"- baseline report: `{baseline.get('_report', '')}`",
        f"- final strategy report: `{main.get('_report', '')}`",
        f"- task count: `{len(tasks)}`",
        "",
        "## Task Selection",
        "",
        "| # | task | prior baseline success | prior baseline steps |",
        "|---:|---|---:|---:|",
    ]
    for idx, task in enumerate(tasks, start=1):
        lines.append(
            f"| {idx} | `{task['task_id']}` | {task.get('baseline_success_rate_prior')} | "
            f"{task.get('baseline_episode_length_prior')} |"
        )
    lines.extend(
        [
            "",
            "## Overall Comparison",
            "",
            "| metric | B0_BASELINE_RERUN | FINAL_STRATEGY |",
            "|---|---:|---:|",
            f"| complete episodes | {ep(baseline, 'complete')} | {ep(main, 'complete')} |",
            f"| success rate all trials | {ep(baseline, 'success_rate_all_trials')} | {ep(main, 'success_rate_all_trials')} |",
            f"| avg episode length complete | {ep(baseline, 'avg_episode_length_complete')} | {ep(main, 'avg_episode_length_complete')} |",
            f"| total reasoning steps | {act(baseline, 'total_steps')} | {act(main, 'total_steps')} |",
            f"| avg step latency ms | {act(baseline, 'avg_step_latency_ms')} | {act(main, 'avg_step_latency_ms')} |",
            f"| avg VLM latency ms | {act(baseline, 'avg_vlm_latency_ms')} | {act(main, 'avg_vlm_latency_ms')} |",
            f"| avg exploration latency ms | {act(baseline, 'avg_exploration_latency_ms')} | {act(main, 'avg_exploration_latency_ms')} |",
            f"| avg wait after VLM ms | {act(baseline, 'avg_exploration_wait_after_vlm_ms')} | {act(main, 'avg_exploration_wait_after_vlm_ms')} |",
            f"| exploration traces | {exp(baseline, 'total_traces')} | {exp(main, 'total_traces')} |",
            f"| depth2 observation rate | {exp(baseline, 'depth2_rate')} | {exp(main, 'depth2_rate')} |",
            f"| avg branch a11y latency ms | {exp(baseline, 'avg_branch_a11y_latency_ms')} | {exp(main, 'avg_branch_a11y_latency_ms')} |",
            "",
            "## Trace Artifacts",
            "",
            "- Step traces are under each task directory as `exploration_step_summary.jsonl` / `exploration_branch_trace.jsonl` / `exploration_page_trace.jsonl`。",
            "- Prompt traces are under `prompt_traces.jsonl`，提示记录在 `prompt_hints.jsonl`。",
            "- Rollback traces are under `rollback_events.jsonl`，并保存在 `rollback_level2_cases.md` / `rollback_failure_cases.md`。",
            "- 证据与回放字段优先使用 `evidence_decisions.jsonl` / `hint_hit_follow.jsonl`。",
            "- Per-step latency CSV is in each run report directory as `step_metrics.csv`.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_count", type=int, default=50)
    parser.add_argument("--variants", default="baseline,final")
    parser.add_argument("--experiment_root", default=str(REPO_ROOT / "results" / "decoupled_parallel_50task"))
    args = parser.parse_args()

    tasks = _parse_baseline_tasks(BASELINE_TABLE, int(args.task_count))
    if len(tasks) != int(args.task_count):
        raise RuntimeError(f"expected {args.task_count} tasks, got {len(tasks)}")
    tasks_str = ",".join(str(task["task_id"]) for task in tasks)
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_root = Path(args.experiment_root).expanduser().resolve() / f"run_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "task_selection.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (run_root / "task_selection.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tasks[0].keys()))
        writer.writeheader()
        writer.writerows(tasks)

    requested = {x.strip().lower() for x in args.variants.split(",") if x.strip()}
    rc = 0
    if "baseline" in requested or "b0_baseline_rerun" in requested:
        rc |= _run_variant(
            run_root / "B0_BASELINE_RERUN",
            "B0_BASELINE_RERUN",
            tasks_str,
            [
                "--agent_name=gelab_agent_resize",
                "--no-explore_enable",
                "--explore_max_runs=0",
                "--explore_max_step=0",
                "--explore_branch_budget=0",
                "--explore_branch_depth=0",
                "--no-explore_enable_t2_lookahead",
                "--t2_mode=off",
            ],
        )
    if "final" in requested or "final_strategy" in requested:
        rc |= _run_variant(
            run_root / "FINAL_STRATEGY",
            "FINAL_STRATEGY",
            tasks_str,
            [
                "--agent_name=explore_agent_gelab",
                "--explore_enable",
                "--explore_max_runs=10000",
                "--explore_max_step=10000",
                "--explore_branch_budget=10",
                "--explore_branch_depth=2",
                "--explore_back_limit=4",
                "--no-explore_planned_only",
                "--explore_decouple_planned",
                "--explore_parallel_vlm",
                "--explore_parallel_lookahead",
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
                "--explore_trace_a11y_limit=80",
                "--t2_mode=off",
            ],
        )
    report = _write_comparison(run_root, tasks)
    print(f"[50-run] comparison_report={report}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
