#!/usr/bin/env python3
"""Run 30-task AndroidWorld validation: B0 baseline vs PASE on one Linux host.

This script launches two variants in sequence on a single machine:
- B0_BASELINE_RERUN (no exploration)
- PASE_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12 (PASE config)

It also materializes all requested diagnostics (jsonl/csv/artifacts) under each
variant directory and generates a merged Chinese report.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from collections import Counter
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = REPO_ROOT / "scripts" / "run_exploration_experiment_report.py"
PYTHON_BIN = sys.executable

DEFAULT_TASKS_CSV = REPO_ROOT / "configs" / "frozen_30task_selection.csv"
DEFAULT_ROOT = REPO_ROOT / "results" / "pase_30task_final_single"
DEFAULT_MAX_STEPS = 16
DEFAULT_BRANCH_BUDGET = 12
DEFAULT_MIN_ATTEMPTS = 12
DEFAULT_PILOT_TASKS = 12

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "B0_BASELINE_RERUN": {
        "enabled": False,
    },
    "PASE_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12": {
        "enabled": True,
        "explore_strategy": "dfs",
        "explore_search_strategy": "best_first",
        "safe_mcts": False,
        "pattern_aware": True,
    },
}

REQUIRED_FILES = [
    "runtime_config.yaml",
    "runtime_config.json",
    "frozen_task_selection.md",
    "task_shard.csv",
    "per_task_results.csv",
    "per_step_metrics.csv",
    "exploration_step_summary.jsonl",
    "exploration_branch_trace.jsonl",
    "exploration_page_trace.jsonl",
    "exploration_latency.jsonl",
    "candidate_scores.jsonl",
    "candidate_filter_stats.jsonl",
    "evidence_decisions.jsonl",
    "prompt_hint_decisions.jsonl",
    "prompt_hints.jsonl",
    "prompt_traces.jsonl",
    "hint_hit_follow.jsonl",
    "rollback_events.jsonl",
    "rollback_gate_decisions.jsonl",
    "rollback_level2_cases.md",
    "rollback_failure_cases.md",
    "shortcut_plans.jsonl",
    "shortcut_shadow_eval.jsonl",
    "state_acquisition_metrics.csv",
    "latency_profile.jsonl",
    "slot_state_records.jsonl",
    "adaptive_budget_records.jsonl",
    "depth_decision_records.jsonl",
    "evidence_memory.jsonl",
    "promotion_events.jsonl",
    "state_alignment.jsonl",
    "step_decoupling_status.jsonl",
    "checkpoint_rows.jsonl",
]


def _base_variant_name(variant_name: str) -> str:
    """Strip staged suffixes like '_P12' from staged variant names."""

    return re.sub(r"_P\d+$", "", variant_name)

SCREENSHOT_DIRS = [
    Path("rollback_timeline_images"),
    Path("screenshots/rollback_failures"),
    Path("screenshots/depth1_branches"),
    Path("screenshots/depth2_branches"),
    Path("screenshots/injected_hints"),
    Path("screenshots/sample"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
    for line in path.read_text(encoding="utf-8").splitlines():
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


def _to_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _load_task_rows(tasks_csv: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    with tasks_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
    tasks = [str(r.get("task") or "").strip() for r in rows if str(r.get("task") or "").strip()]
    tasks = [t for t in tasks if t]
    return rows, tasks


def _write_task_files(variant_root: Path, rows: list[dict[str, str]], csv_path: Path) -> None:
    variant_root.mkdir(parents=True, exist_ok=True)
    rows = rows[:]
    (variant_root / "frozen_task_selection.csv").write_text(
        "task_id,task,app,task_mode,shard\n"
        + "\n".join(
            ",".join([str(r.get("task_id", "")), r.get("task", ""), r.get("app", ""), r.get("task_mode", ""), r.get("shard", "")])
            for r in rows
        )
        + "\n",
        encoding="utf-8",
    )
    (variant_root / "frozen_30task_selection.csv").write_text(
        "task_id,task,app,task_mode,shard\n"
        + "\n".join(
            ",".join([str(r.get("task_id", "")), r.get("task", ""), r.get("app", ""), r.get("task_mode", ""), r.get("shard", "")])
            for r in rows
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Task selection for final PASE run",
        "",
        "| # | task_id | task | app | task_mode | shard |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {_to_int(r.get('task_id'))} | `{r.get('task_id', '')}` | `{r.get('task', '')}` | "
            f"{r.get('app', '')} | {r.get('task_mode', '')} | {r.get('shard', '')} |"
        )
    (variant_root / "frozen_task_selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (variant_root / "task_shard.csv").write_text(
        csv_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def _write_placeholder(path: Path, is_csv: bool = False) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_csv:
        path.write_text("\n", encoding="utf-8")
    else:
        path.write_text("无可用数据\n", encoding="utf-8")


def _latest_run(variant_root: Path) -> Path | None:
    runs = sorted(variant_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _concat_jsonl(run_dir: Path, filename: str, out_path: Path) -> int:
    count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "traces").exists():
        _write_placeholder(out_path)
        return 0
    with out_path.open("w", encoding="utf-8") as out:
        for p in sorted((run_dir / "traces").rglob(filename)):
            if p == out_path:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            if not text:
                continue
            if not text.endswith("\n"):
                text = text + "\n"
            out.write(text)
            count += sum(1 for line in text.splitlines() if line.strip())
    return count


def _read_trace_rows(run_dir: Path, filename: str) -> list[dict[str, Any]]:
    traces_dir = run_dir / "traces"
    if not traces_dir.exists():
        return []
    for p in sorted(traces_dir.rglob(filename)):
        if p.is_file():
            rows = _read_jsonl(p)
            if rows:
                return rows
    return []


def _write_jsonl_rows(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _derive_slot_state_records(run_dir: Path, out_path: Path) -> None:
    rows = _read_trace_rows(run_dir, "evidence_decisions.jsonl")
    derived: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        derived.append(
            {
                "task_id": row.get("task_id"),
                "step": row.get("step"),
                "branch_id": row.get("branch_id"),
                "evidence_type": row.get("evidence_type"),
                "evidence_depth": row.get("evidence_depth"),
                "candidate_label": row.get("candidate_label"),
                "operator": row.get("operator"),
                "state_match_score": row.get("state_match_score"),
                "slot_or_target_score": row.get("slot_or_target_score"),
                "rollback_verified": row.get("rollback_verified"),
                "target_visible": row.get("target_visible"),
                "injected": row.get("injected"),
                "rejected_reason": row.get("rejected_reason"),
                "injected_reason": row.get("injected_reason"),
            }
        )
    _write_jsonl_rows(derived, out_path)


def _derive_adaptive_budget_records(run_dir: Path, out_path: Path) -> None:
    rows = _read_trace_rows(run_dir, "latency_budget_records.jsonl")
    if not rows:
        # Fallback: state acquisition metrics can act as budget proxies for quick diagnostics.
        metric_rows: list[dict[str, Any]] = []
        metric_path = run_dir / "state_acquisition_metrics.csv"
        if not metric_path.exists():
            fallback = sorted(run_dir.rglob("state_acquisition_metrics.csv"))
            metric_path = fallback[0] if fallback else metric_path
        if metric_path.exists():
            with metric_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if row:
                        metric_rows.append(
                            {
                                "step": row.get("step"),
                                "episode_id": row.get("episode_id"),
                                "task": row.get("task"),
                                "candidate_count": row.get("candidate_count"),
                                "attempted_count": row.get("attempted_count"),
                                "latency_budget_ms": row.get("a11y_ms")
                                or row.get("state_fetch_ms")
                                or row.get("latency_ms"),
                                "source": "state_acquisition_metrics",
                            }
                        )
        rows = metric_rows
    _write_jsonl_rows(rows, out_path)


def _derive_depth_decision_records(run_dir: Path, out_path: Path) -> None:
    rows = _read_trace_rows(run_dir, "exploration_step_summary.jsonl")
    if not rows:
        rows = _read_trace_rows(run_dir, "exploration_branch_trace.jsonl")
    derived = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        derived.append(
            {
                "task_id": row.get("task_id"),
                "step": row.get("step"),
                "branch_id": row.get("branch_id") or row.get("branch_key"),
                "depth_reached": row.get("depth_reached"),
                "stop_reason": row.get("stop_reason"),
                "candidate_label": row.get("selected_candidate", {}).get("label")
                if isinstance(row.get("selected_candidate"), dict)
                else row.get("selected_candidate"),
                "score": row.get("score"),
                "operator": row.get("operator"),
                "state_match": row.get("state_match"),
            }
        )
    _write_jsonl_rows(derived, out_path)


def _derive_evidence_memory(run_dir: Path, out_path: Path) -> None:
    rows = _read_trace_rows(run_dir, "evidence_decisions.jsonl")
    if not rows:
        rows = _read_trace_rows(run_dir, "prompt_hints.jsonl")
    memory = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        memory.append(row)
    _write_jsonl_rows(memory, out_path)


def _derive_promotion_events(run_dir: Path, out_path: Path) -> None:
    rows = _read_trace_rows(run_dir, "evidence_decisions.jsonl")
    events = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        events.append(
            {
                "task_id": row.get("task_id"),
                "step": row.get("step"),
                "branch_id": row.get("branch_id"),
                "event_type": "promote" if row.get("injected") else "reject",
                "injected": row.get("injected"),
                "evidence_type": row.get("evidence_type"),
                "candidate_label": row.get("candidate_label"),
                "score": row.get("score"),
                "rejected_reason": row.get("rejected_reason"),
                "state_match_score": row.get("state_match_score"),
                "target_visible": row.get("target_visible"),
                "rollback_verified": row.get("rollback_verified"),
            }
        )
    _write_jsonl_rows(events, out_path)


def _concat_markdown(run_dir: Path, filename: str, out_path: Path) -> None:
    chunks: list[str] = []
    for p in sorted((run_dir / "traces").rglob(filename)):
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if txt:
            chunks.append(f"<!-- {p.as_posix()} -->\n\n{txt}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(chunks) + ("\n" if chunks else "无可用记录\n"), encoding="utf-8")


def _copy_screenshots(run_dir: Path, variant_root: Path) -> None:
    src_root = run_dir / "screenshots"
    if not src_root.exists():
        for d in SCREENSHOT_DIRS:
            (variant_root / d).mkdir(parents=True, exist_ok=True)
        return
    for sub in SCREENSHOT_DIRS:
        s = src_root / sub
        dst = variant_root / sub
        if dst.exists():
            shutil.rmtree(dst)
        if s.exists() and s.is_dir():
            shutil.copytree(s, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)


def _write_per_task_results(summary: dict[str, Any], out_path: Path) -> None:
    rows = (summary.get("episodes") or {}).get("task_episode_rows")
    if not isinstance(rows, list) or not rows:
        _write_placeholder(out_path, is_csv=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "episode_length", "success", "exception"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, dict):
                continue
            writer.writerow({
                "task": row.get("task"),
                "episode_length": row.get("episode_length"),
                "success": row.get("success"),
                "exception": row.get("exception"),
            })


def _materialize_variant_outputs(variant_root: Path, run_dir: Path, variant_name: str, tasks_csv: Path, task_rows: list[dict[str, str]]) -> None:
    report_dir = run_dir / "report"
    traces_root = run_dir / "traces"
    summary = _read_json(report_dir / "summary.json")
    if summary:
        (variant_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if (report_dir / "step_metrics.csv").exists():
        shutil.copy2(report_dir / "step_metrics.csv", variant_root / "per_step_metrics.csv")
    else:
        _write_placeholder(variant_root / "per_step_metrics.csv", is_csv=True)

    _write_per_task_results(summary, variant_root / "per_task_results.csv")
    _write_task_files(variant_root, task_rows, tasks_csv)

    # runtime config: prefer root trace, else report runtime config
    runtime_candidates = sorted(traces_root.rglob("runtime_config.yaml"))
    if runtime_candidates:
        shutil.copy2(runtime_candidates[0], variant_root / "runtime_config.yaml")
    elif (report_dir / "runtime_config.yaml").exists():
        shutil.copy2(report_dir / "runtime_config.yaml", variant_root / "runtime_config.yaml")
    else:
        (variant_root / "runtime_config.yaml").write_text(f"variant: {variant_name}\n", encoding="utf-8")

    # concat all requested diagnostics
    for name in REQUIRED_FILES:
        if name == "runtime_config.yaml":
            continue
        if name == "frozen_task_selection.md" or name == "task_shard.csv":
            continue
        out = variant_root / name
        if name.endswith(".jsonl"):
            _concat_jsonl(run_dir, name, out)
        elif name.endswith(".csv"):
            csv_candidates = sorted(traces_root.rglob(name)) if traces_root.exists() else []
            src = csv_candidates[0] if csv_candidates else None
            if src and src.exists():
                shutil.copy2(src, out)
            else:
                _write_placeholder(out, is_csv=True)
        elif name.endswith(".md"):
            _concat_markdown(run_dir, name, out)
        else:
            # unknown extension, keep best-effort passthrough from report if possible
            if (report_dir / name).exists():
                shutil.copy2(report_dir / name, out)
            else:
                _write_placeholder(out)

    # Derived diagnostics for design-specific files.
    derived_targets = {
        "slot_state_records.jsonl": _derive_slot_state_records,
        "adaptive_budget_records.jsonl": _derive_adaptive_budget_records,
        "depth_decision_records.jsonl": _derive_depth_decision_records,
        "evidence_memory.jsonl": _derive_evidence_memory,
        "promotion_events.jsonl": _derive_promotion_events,
    }
    for file_name, producer in derived_targets.items():
        path = variant_root / file_name
        if path.exists():
            if path.stat().st_size == 0:
                producer(run_dir, path)
            else:
                try:
                    txt = path.read_text(encoding="utf-8").strip()
                    if not txt or txt == "无可用数据":
                        producer(run_dir, path)
                except Exception:
                    producer(run_dir, path)
        else:
            producer(run_dir, path)

    for file_like in ["candidate_scores.jsonl", "prompt_traces.jsonl"]:
        out = variant_root / file_like
        if out.exists():
            continue
        _write_placeholder(out)

    # screenshot artifacts
    _copy_screenshots(run_dir, variant_root)

    # manifest for reproducibility
    manifest = {
        "variant": variant_name,
        "run_dir": str(run_dir),
        "generated_at": dt.datetime.now().isoformat(),
        "run_root": str(variant_root),
        "required_files": [str(variant_root / x) for x in REQUIRED_FILES],
    }
    (variant_root / "variant_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _common_args(
    variant_name: str,
    tasks: list[str],
    variant_root: Path,
    max_cases: int,
    max_steps: int,
) -> list[str]:
    return [
        PYTHON_BIN,
        str(REPORT_SCRIPT),
        "--run",
        "--suite_family=android_world",
        f"--tasks={','.join(tasks)}",
        "--n_task_combinations=1",
        "--task_random_seed=43",
        "--fixed_task_seed",
        "--image_downsample_scale=1.0",
        "--baseline_table=results/baseline_4b_full_report/task_results_4B_2.txt",
        f"--experiment_root={variant_root}",
        f"--max_cases={max_cases}",
        f"--max_n_steps={max_steps}",
        "--a11y_method=fast_provider",
        "--a11y_preflight_timeout=90",
        "--latency_profile",
        "--agent_name=explore_agent_gelab",
        f"--explore_variant={variant_name}",
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


def _variant_args(
    variant_name: str,
    tasks: list[str],
    variant_root: Path,
    max_cases: int,
    max_steps: int,
    branch_budget: int,
    min_attempts: int,
) -> list[str]:
    base_variant = _base_variant_name(variant_name)
    cfg = VARIANT_CONFIGS[base_variant]
    cmd = _common_args(base_variant, tasks, variant_root, max_cases, max_steps)
    if not cfg.get("enabled"):
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

    cmd.extend(
        [
            "--explore_enable",
            "--explore_max_runs=10000",
            "--explore_max_step=10000",
            f"--explore_branch_budget={branch_budget}",
            f"--explore_min_attempts_per_step={min_attempts}",
            "--explore_branch_depth=2",
            "--explore_back_limit=0",
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
            f"--explore_strategy={cfg['explore_strategy']}",
            f"--explore_search_strategy={cfg['explore_search_strategy']}",
            "--explore_rollback_policy=improved",
            "--explore_fixed_framework",
            "--explore_answer_extractors",
            "--explore_slot_complete",
            "--explore_slot_policy_switcher",
            "--no-explore_enable_t2_lookahead",
            "--explore_lightweight_a11y_trace",
            "--explore_trace_a11y_limit=120",
            "--trace_screenshot_mode=failure+level2+depth2+injected+sampled",
        ]
    )
    if cfg.get("safe_mcts"):
        cmd.append("--explore_safe_mcts")
    return cmd


def _run_variant(
    run_root: Path,
    variant_name: str,
    tasks: list[str],
    task_rows: list[dict[str, str]],
    tasks_csv: Path,
    max_cases: int,
    max_steps: int,
    branch_budget: int,
    min_attempts: int,
    force: bool,
    env_overrides: dict[str, str],
) -> int:
    variant_root = run_root / variant_name
    variant_root.mkdir(parents=True, exist_ok=True)
    manifest_path = variant_root / "variant_manifest.json"

    if manifest_path.exists() and not force:
        manifest = _read_json(manifest_path)
        run_dir = manifest.get("run_dir", "")
        if run_dir and Path(run_dir).exists() and manifest.get("returncode") in {0, "0", None}:
            print(f"[pase] skip {variant_name} (manifest exists, use --force to rerun).")
            return 0

    cmd = _variant_args(
        variant_name,
        tasks,
        variant_root,
        max_cases,
        max_steps,
        branch_budget,
        min_attempts,
    )
    log_path = variant_root / "driver.log"

    env = os.environ.copy()
    env.update(env_overrides)
    # enforce required env for this benchmark
    base_variant = _base_variant_name(variant_name)
    env.update(
        {
            "ANDROID_WORLD_DECOUPLED_EXPLORATION": "1",
            "ANDROID_WORLD_EXPLORATION_TIMING": "parallel_shadow",
            "ANDROID_WORLD_LIGHT_EXPLORE_USE_CURRENT_ACTION": "0",
            "ANDROID_WORLD_LIGHT_EXPLORE_USE_PLANNING_TEXT": "0",
            "ANDROID_WORLD_LIGHT_EXPLORE_PLANNED_ONLY": "0",
            "ANDROID_WORLD_T2_MODE": "shadow",
            "ANDROID_WORLD_T2_ACTIVE": "0",
            "ANDROID_WORLD_MAX_N_STEPS": str(max_steps),
            "ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_BUDGET": str(branch_budget),
            "ANDROID_WORLD_LIGHT_EXPLORE_MIN_ATTEMPTS_PER_STEP": str(min_attempts),
            "ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_DEPTH": "2",
            "ANDROID_WORLD_TRACE_SCREENSHOT_MODE": "failure+level2+depth2+injected+sampled",
            "ANDROID_WORLD_LIGHT_EXPLORE_T2_LOOKAHEAD": "0",
            "ANDROID_WORLD_LIGHT_EXPLORE_ENABLE_T2_LOOKAHEAD": "0",
            "ANDROID_WORLD_T2_ALLOW_SAFE_SEARCH_INPUT": "0",
            "ANDROID_WORLD_LIGHT_EXPLORE_PATTERN_AWARE_OPERATOR_BEST_FIRST": "1" if "PASE_" in base_variant else "0",
        }
    )

    with log_path.open("w", encoding="utf-8") as log:
        log.write("python "+" ".join(cmd)+"\n\n")
        log.write("ENV:\n")
        for k, v in sorted(env.items()):
            if k.startswith("ANDROID_WORLD"):
                log.write(f"{k}={v}\n")

    print(f"[pase] run variant={variant_name}")
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    run_dir = _latest_run(variant_root)
    manifest = {
        "variant": variant_name,
        "command": cmd,
        "returncode": proc.returncode,
        "run_dir": str(run_dir) if run_dir else "",
        "log": str(log_path),
        "env": env_overrides,
        "task_count": len(tasks),
        "tasks": tasks,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if not run_dir:
        return int(proc.returncode)

    _materialize_variant_outputs(variant_root, run_dir, variant_name, tasks_csv, task_rows)
    return int(proc.returncode)


def _merge_compare_report(
    run_root: Path,
    variants: list[str],
    tasks: list[str],
    baseline_variant: str = "B0_BASELINE_RERUN",
    pase_variant: str = "PASE_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12",
    report_name: str = "merged_pase_30task_report_cn.md",
) -> Path:
    def _baseline_rows_from_summary(rows: list[dict[str, Any]], task_order: list[str]) -> list[dict[str, str]]:
        if not rows:
            return []
        task_lookup = {r.get("task"): r for r in rows}
        synthetic: list[dict[str, str]] = []
        for task in task_order:
            r = task_lookup.get(task, {})
            success = _to_bool(r.get("baseline_success_rate"))
            episode_length = r.get("baseline_episode_length")
            synthetic.append(
                {
                    "task": task,
                    "success": str(bool(success)),
                    "episode_length": str(_to_float(episode_length)) if episode_length is not None else "0",
                    "exception": "",
                }
            )
        return synthetic
    fallback_baseline_rows: list[dict[str, str]] = []
    for variant in variants:
        root = run_root / variant
        variant_summary = _read_json(root / "summary.json")
        if not fallback_baseline_rows:
            baseline_compare = variant_summary.get("baseline_compare", {})
            if isinstance(baseline_compare, dict):
                rows = baseline_compare.get("rows", [])
                if rows:
                    fallback_baseline_rows = list(rows)

    variant_rows: dict[str, dict[str, Any]] = {}
    for variant in variants:
        root = run_root / variant
        task_rows = _read_csv(root / "per_task_results.csv")
        step_rows = _read_csv(root / "per_step_metrics.csv")
        hint_rows = _read_jsonl(root / "prompt_hints.jsonl")
        hint_hit_rows = _read_jsonl(root / "hint_hit_follow.jsonl")
        evidence_rows = _read_jsonl(root / "evidence_decisions.jsonl")
        rollback_rows = _read_jsonl(root / "rollback_events.jsonl")
        variant_summary = _read_json(root / "summary.json")
        if not fallback_baseline_rows and variant_summary.get("baseline_compare", {}).get("rows"):
            fallback_baseline_rows = list(variant_summary["baseline_compare"]["rows"])

        success_rows = [r for r in task_rows if _to_bool(r.get("success"))]
        if variant == baseline_variant and not success_rows and fallback_baseline_rows:
            # 如果该变体没有 per_task 结果，尝试从 baseline_compare 回退（例如只存在 B0_P12 阶段）。
            task_rows = _baseline_rows_from_summary(fallback_baseline_rows, tasks)
            success_rows = [r for r in task_rows if _to_bool(r.get("success"))]
        success_rate = _to_float(len(success_rows)) / max(1, len(task_rows))
        avg_steps = 0.0
        if task_rows:
            avg_steps = sum(_to_float(r.get("episode_length")) for r in task_rows) / len(task_rows)

        injected = [r for r in hint_rows if _to_bool(r.get("injected"))]
        prompt_injected = len(injected)
        hint_hit = sum(1 for r in hint_rows if _to_bool(r.get("hit")) or _to_bool(r.get("hit_hint")))
        hint_follow = sum(1 for r in hint_rows if _to_bool(r.get("follow")) or _to_bool(r.get("follow_hint")))

        evidence_rejected = [r for r in evidence_rows if not _to_bool(r.get("injected"))]
        evidence_injected = [r for r in evidence_rows if _to_bool(r.get("injected"))]

        rollback_total = len(rollback_rows)
        rollback_fail = sum(1 for r in rollback_rows if not _to_bool(r.get("success", True)))
        level2 = sum(1 for r in rollback_rows if r.get("level2_triggered"))
        variant_rows[variant] = {
            "per_task": task_rows,
            "per_step": step_rows,
            "summary": {
                "task_count": len(tasks),
                "success_count": len(success_rows),
                "success_rate": success_rate,
                "avg_steps": avg_steps,
                "prompt_injected": prompt_injected,
                "hint_hit": hint_hit,
                "hint_follow": hint_follow,
                "evidence_injected": len(evidence_injected),
                "evidence_rejected": len(evidence_rejected),
                "rollback_total": rollback_total,
                "rollback_fail": rollback_fail,
                "rollback_level2": level2,
            },
            "hint_hit_follow_rows": hint_hit_rows,
            "evidence_rows": evidence_rows,
            "rollback_rows": rollback_rows,
        }

        # write combined per-task file for merging
        _write_checkpoint_rows(root / "checkpoint_rows.jsonl", task_rows, hint_rows, evidence_rows, rollback_rows)

    base = variant_rows.get(baseline_variant, {}).get("summary", {})
    pase = variant_rows.get(pase_variant, {}).get("summary", {})

    title = "# merged_pase_30task_report_cn.md"
    if not report_name.startswith("merged_pase_30task"):
        title = "# " + report_name.replace(".md", "")
    lines = [
        title,
        "",
        "## 总体对比",
        "",
        f"- 任务总数：`{len(tasks)}`",
        "- 任务 step 上限：`16`",
        "- Active t+2 shortcut：`off`（shadow metadata only）",
        f"- B0 success_rate: `{base.get('success_rate', 0):.4f}`",
        f"- PASE success_rate: `{pase.get('success_rate', 0):.4f}`",
        f"- B0 成功任务数：`{base.get('success_count', 0)}`",
        f"- PASE 成功任务数：`{pase.get('success_count', 0)}`",
        f"- B0 平均步数：`{base.get('avg_steps', 0):.2f}`",
        f"- PASE 平均步数：`{pase.get('avg_steps', 0):.2f}`",
        f"- PASE 注入 hint 数：`{pase.get('prompt_injected', 0)}`",
        f"- PASE hint hit/follow：`{pase.get('hint_hit', 0)} / {pase.get('hint_follow', 0)}`",
        f"- PASE evidence injected/rejected：`{pase.get('evidence_injected', 0)} / {pase.get('evidence_rejected', 0)}`",
        f"- PASE rollback events / failures：`{pase.get('rollback_total', 0)} / {pase.get('rollback_fail', 0)}`",
        "",
        "## 任务级结果（按 baseline 成功率排序）",
        "",
        "| task | B0_success | PASE_success | B0_episode_length | PASE_episode_length |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    base_tasks = {
        r.get("task") or r.get("task_id"): r
        for r in variant_rows.get(baseline_variant, {}).get("per_task", [])
    }
    pase_tasks = {
        r.get("task") or r.get("task_id"): r
        for r in variant_rows.get(pase_variant, {}).get("per_task", [])
    }
    for t in tasks:
        b = base_tasks.get(t, {})
        p = pase_tasks.get(t, {})
        lines.append(
            f"| `{t}` | {int(_to_bool(b.get('success')))} | {int(_to_bool(p.get('success')))} | "
            f"{_to_float(b.get('episode_length')):.2f} | {_to_float(p.get('episode_length')):.2f} |"
        )

    lines.extend(["", "## 每步探索与 rollback 简表", ""])

    # per-step counts
    for variant, payload in variant_rows.items():
        step_rows = payload.get("per_step", [])
        if not step_rows:
            lines.append(f"- `{variant}`：无 `per_step_metrics.csv` 记录")
            continue
        attempts = sum(_to_int(r.get("attempted_count") or r.get("candidate_count")) for r in step_rows)
        executed = sum(_to_int(r.get("executed_attempt_count") or r.get("selected_candidates")) for r in step_rows)
        passive = sum(_to_int(r.get("passive_attempt_count") or 0) for r in step_rows)
        depth1 = sum(_to_int(r.get("depth1_count") or r.get("depth1_candidates") or 0) for r in step_rows)
        depth2 = sum(_to_int(r.get("depth2_count") or r.get("depth2_candidates") or 0) for r in step_rows)
        stop_reasons = Counter(r.get("stop_reason") for r in step_rows if r.get("stop_reason"))
        lines.extend(
            [
                f"- `{variant}`",
                f"  - attempted_count={attempts}",
                f"  - executed_attempt_count={executed}",
                f"  - passive_attempt_count={passive}",
                f"  - risk_classified_attempt_count={sum(_to_int(r.get('risk_classified_attempt_count') or 0) for r in step_rows)}",
                f"  - depth1_count={depth1} / depth2_count={depth2}",
                f"  - rollback_events={payload['summary'].get('rollback_total', 0)}",
                f"  - rollback_level2={payload['summary'].get('rollback_level2', 0)}",
                f"  - rollback_failed={payload['summary'].get('rollback_fail', 0)}",
                f"  - stop reason top3: {dict(stop_reasons.most_common(3))}",
            ]
        )

    lines.extend(["", "## Evidence 与 Hint",
                  "", "- 下列文件可直接用于后续审计：", "",
                  f"  - {run_root}/{baseline_variant}/evidence_decisions.jsonl",
                  f"  - {run_root}/{pase_variant}/evidence_decisions.jsonl",
                  f"  - {run_root}/{pase_variant}/prompt_hints.jsonl",
                  f"  - {run_root}/{pase_variant}/hint_hit_follow.jsonl",
                  "", "## 输出路径", "",
                  f"- `per_task_results.csv`: 各变体分别位于 `{baseline_variant}/` 与 `{pase_variant}/`", ])

    out = run_root / report_name
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _write_checkpoint_rows(out_path: Path, task_rows: list[dict[str, str]], hint_rows: list[dict[str, Any]], evidence_rows: list[dict[str, Any]], rollback_rows: list[dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in task_rows:
            row = dict(row)
            row.update({"_type": "task"})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in hint_rows:
            row = dict(row)
            row.update({"_type": "hint"})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in evidence_rows:
            row = dict(row)
            row.update({"_type": "evidence"})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in rollback_rows:
            row = dict(row)
            row.update({"_type": "rollback"})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stage_variant_name(variant_name: str, task_count: int, full_task_count: int) -> str:
    if task_count >= full_task_count:
        return variant_name
    return f"{variant_name}_P{task_count}"


def _collect_tasks_for_stage(all_tasks: list[str], stage_size: int) -> list[str]:
    return all_tasks[:stage_size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_csv", default=str(DEFAULT_TASKS_CSV))
    parser.add_argument("--experiment_root", default=str(DEFAULT_ROOT))
    parser.add_argument("--variants", default=",".join(VARIANT_CONFIGS.keys()))
    parser.add_argument("--max_cases", type=int, default=30)
    parser.add_argument("--pilot_cases", type=int, default=DEFAULT_PILOT_TASKS)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--branch_budget", type=int, default=DEFAULT_BRANCH_BUDGET)
    parser.add_argument("--min_attempts_per_step", type=int, default=DEFAULT_MIN_ATTEMPTS)
    parser.add_argument(
        "--run_stages",
        default="pilot,full",
        help="Comma-separated execution stages, e.g. pilot,full",
    )
    parser.add_argument(
        "--continue_on_pilot_fail",
        action="store_true",
        help="Continue to full stage even if pilot stage has non-zero error return.",
    )
    parser.add_argument("--force", action="store_true", help="rerun variants even if existing manifest is present")
    parser.add_argument("--report_only", action="store_true", help="skip running, only regenerate merged report")
    args = parser.parse_args()

    tasks_csv = Path(args.tasks_csv).expanduser().resolve()
    if not tasks_csv.exists():
        raise SystemExit(f"tasks_csv not found: {tasks_csv}")

    run_root = Path(args.experiment_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    task_rows, tasks = _load_task_rows(tasks_csv)
    if len(tasks) < args.max_cases:
        raise SystemExit(f"tasks in csv={len(tasks)} < max_cases={args.max_cases}")
    if args.pilot_cases < 1:
        raise SystemExit(f"--pilot_cases must be >= 1, got {args.pilot_cases}")
    if args.pilot_cases > args.max_cases:
        args.pilot_cases = args.max_cases
    tasks = tasks[: args.max_cases]

    raw_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    variants = []
    for raw in raw_variants:
        base = _base_variant_name(raw)
        if base not in VARIANT_CONFIGS:
            continue
        if base not in variants:
            variants.append(base)
    unknown = [v for v in raw_variants if _base_variant_name(v) not in VARIANT_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")

    stages = [s.strip() for s in args.run_stages.split(",") if s.strip()]
    if not stages:
        stages = ["pilot", "full"]
    for s in stages:
        if s not in {"pilot", "full", "none"}:
            raise SystemExit(f"invalid stage '{s}', expect pilot/full/none")

    print(f"[pase] root={run_root}")
    print(f"[pase] tasks={len(tasks)} variants={','.join(variants)}")
    print(f"[pase] stages={','.join(stages)} pilot={args.pilot_cases} full={args.max_cases}")

    if not args.report_only:
        stage_task_counts = {}
        for s in stages:
            if s == "pilot":
                stage_task_counts[s] = args.pilot_cases
            elif s == "full":
                stage_task_counts[s] = args.max_cases

        pilot_failed = False
        for stage in stages:
            if stage == "none":
                continue
            stage_tasks = _collect_tasks_for_stage(tasks, stage_task_counts[stage])
            stage_task_set = set(stage_tasks)
            stage_rows = [r for r in task_rows if str(r.get("task", "")) in stage_task_set]
            print(f"[pase] stage={stage} task_count={len(stage_tasks)}")
            for variant in variants:
                stage_variant = _stage_variant_name(variant, len(stage_tasks), args.max_cases)
                rc = _run_variant(
                    run_root=run_root,
                    variant_name=stage_variant,
                    tasks=stage_tasks,
                    task_rows=stage_rows,
                    tasks_csv=tasks_csv,
                    max_cases=len(stage_tasks),
                    max_steps=args.max_steps,
                    branch_budget=args.branch_budget,
                    min_attempts=args.min_attempts_per_step,
                    force=args.force,
                    env_overrides={},
                )
                if rc != 0:
                    print(f"[pase] stage={stage} variant={variant} exit={rc}")
                    pilot_failed = True
            if stage == "pilot" and pilot_failed and not args.continue_on_pilot_fail:
                print("[pase] stop at pilot due to failure; use --continue_on_pilot_fail to continue.")
                return int(pilot_failed)

    report_size = args.max_cases if "full" in stages else args.pilot_cases
    report_tasks = _collect_tasks_for_stage(tasks, report_size)
    report_variants = [
        _stage_variant_name(variant, report_size, args.max_cases)
        for variant in variants
    ]
    baseline_variant = _stage_variant_name("B0_BASELINE_RERUN", report_size, args.max_cases) if "B0_BASELINE_RERUN" in variants else ""
    pase_variant = _stage_variant_name(
        "PASE_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12",
        report_size,
        args.max_cases,
    ) if "PASE_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12" in variants else ""

    report = _merge_compare_report(
        run_root=run_root,
        variants=report_variants,
        tasks=report_tasks,
        baseline_variant=baseline_variant or "B0_BASELINE_RERUN",
        pase_variant=pase_variant or "PASE_PATTERN_AWARE_OPERATOR_BEST_FIRST_BUDGET12",
        report_name="merged_pase_30task_report_cn.md" if "full" in stages else "merged_pase_pilot_report_cn.md",
    )
    print(f"[pase] merged report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
