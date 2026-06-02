#!/usr/bin/env python3
"""Periodic progress monitor for LB-MCTS V2 AndroidWorld runs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception:
        return []


def _read_jsonl_many(root: Path, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob(filename)):
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
        except Exception:
            continue
    return rows


def _latest_run(variant_root: Path) -> Path | None:
    runs = sorted(variant_root.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _task_name(row: dict[str, Any]) -> str:
    for key in ("task", "task_template", "task_id", "task_name", "name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _checkpoint_rows(run_dir: Path | None) -> list[dict[str, Any]]:
    if not run_dir:
        return []
    ckpt_dir = run_dir / "checkpoints"
    rows: list[dict[str, Any]] = []
    for path in sorted(ckpt_dir.glob("*.pkl.gz"), key=lambda p: p.stat().st_mtime):
        try:
            with gzip.open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                row = dict(obj[0])
            elif isinstance(obj, dict):
                row = dict(obj)
            else:
                row = {"task_template": path.name.rsplit("_", 1)[0]}
            task = _task_name(row) or path.name.rsplit("_", 1)[0]
            rows.append(
                {
                    "task": task,
                    "success": _bool(row.get("is_successful", row.get("success", False))),
                    "episode_length": int(_float(row.get("episode_length"))),
                    "exception": str(row.get("exception_info") or row.get("exception") or ""),
                    "checkpoint": str(path),
                    "mtime": path.stat().st_mtime,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "task": path.name.rsplit("_", 1)[0],
                    "success": False,
                    "episode_length": 0,
                    "exception": f"checkpoint_load_error:{exc}",
                    "checkpoint": str(path),
                    "mtime": path.stat().st_mtime,
                }
            )
    return rows


def _rows_from_root(root: Path) -> list[dict[str, Any]]:
    csv_rows = _read_csv(root / "per_task_results.csv")
    if csv_rows:
        out: list[dict[str, Any]] = []
        for row in csv_rows:
            out.append(
                {
                    "task": _task_name(row),
                    "success": _bool(row.get("success")),
                    "episode_length": int(_float(row.get("episode_length"))),
                    "exception": str(row.get("exception") or ""),
                }
            )
        return out
    return _checkpoint_rows(_latest_run(root))


def _load_expected_tasks(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def _rate(rows: list[dict[str, Any]]) -> tuple[int, int, float, float]:
    n = len(rows)
    succ = sum(1 for row in rows if _bool(row.get("success")))
    avg_steps = sum(int(_float(row.get("episode_length"))) for row in rows) / n if n else 0.0
    return succ, n, (succ / n if n else 0.0), avg_steps


def _by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("task") or ""): row for row in rows if str(row.get("task") or "")}


def _compare(current: dict[str, dict[str, Any]], ref: dict[str, dict[str, Any]], tasks: list[str]) -> dict[str, Any]:
    overlap = [task for task in tasks if task in current and task in ref]
    rescued = [task for task in overlap if _bool(current[task].get("success")) and not _bool(ref[task].get("success"))]
    broken = [task for task in overlap if not _bool(current[task].get("success")) and _bool(ref[task].get("success"))]
    curr_succ = sum(1 for task in overlap if _bool(current[task].get("success")))
    ref_succ = sum(1 for task in overlap if _bool(ref[task].get("success")))
    return {
        "overlap": overlap,
        "current_success": curr_succ,
        "reference_success": ref_succ,
        "rescued": rescued,
        "broken": broken,
    }


def _trace_stats(run_dir: Path | None) -> dict[str, Any]:
    trace_root = run_dir / "traces" if run_dir else Path("/nonexistent")
    steps = _read_jsonl_many(trace_root, "exploration_step_summary.jsonl")
    rollbacks = _read_jsonl_many(trace_root, "rollback_events.jsonl")
    evidence = _read_jsonl_many(trace_root, "evidence_decisions.jsonl")
    hints = _read_jsonl_many(trace_root, "prompt_hints.jsonl")
    hit_follow = _read_jsonl_many(trace_root, "hint_hit_follow.jsonl")
    attempted = sum(int(_float(row.get("attempted_count", row.get("attempt_count", 0)))) for row in steps)
    executed = sum(int(_float(row.get("executed_attempt_count", row.get("executed_count", 0)))) for row in steps)
    depth2 = sum(int(_float(row.get("depth2_count", row.get("rooted_depth2_count", 0)))) for row in steps)
    injected = sum(1 for row in evidence if _bool(row.get("injected")))
    rollback_fail = sum(1 for row in rollbacks if row.get("success") is not None and not _bool(row.get("success")))
    rollback_level2 = sum(1 for row in rollbacks if str(row.get("level") or "") == "level2" or _bool(row.get("level2_triggered")))
    hint_hit = sum(1 for row in hit_follow if _bool(row.get("hit")) or _bool(row.get("hint_hit")))
    hint_follow = sum(1 for row in hit_follow if _bool(row.get("follow")) or _bool(row.get("hint_follow")))
    return {
        "step_rows": len(steps),
        "attempted": attempted,
        "executed": executed,
        "depth2": depth2,
        "rollback_events": len(rollbacks),
        "rollback_fail": rollback_fail,
        "rollback_level2": rollback_level2,
        "evidence_decisions": len(evidence),
        "evidence_injected": injected,
        "prompt_hints": len(hints),
        "hint_rows": len(hit_follow),
        "hint_hit": hint_hit,
        "hint_follow": hint_follow,
    }


def _experiment_process_alive() -> bool:
    try:
        proc = subprocess.run(
            ["pgrep", "-f", "run_exploration_experiment_report.py|/run.py --agent_name=explore_agent_gelab"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return bool(proc.stdout.strip())
    except Exception:
        return True


def _write_monitor(args: argparse.Namespace) -> tuple[int, int]:
    variant_root = Path(args.variant_root).expanduser().resolve()
    expected = _load_expected_tasks(Path(args.expected_tasks_file).expanduser().resolve())
    frozen30 = expected[:30]
    run_dir = _latest_run(variant_root)
    current_rows = _checkpoint_rows(run_dir)
    current_by_task = _by_task(current_rows)
    lb30_rows = _rows_from_root(Path(args.lb30_root).expanduser().resolve()) if args.lb30_root else []
    base_rows = _rows_from_root(Path(args.baseline_root).expanduser().resolve()) if args.baseline_root else []
    lb30_by_task = _by_task(lb30_rows)
    base_by_task = _by_task(base_rows)
    succ, n, rate, avg_steps = _rate(current_rows)
    prefix_rows = [current_by_task[t] for t in frozen30 if t in current_by_task]
    psucc, pn, prate, pavg = _rate(prefix_rows)
    cmp_lb30 = _compare(current_by_task, lb30_by_task, frozen30)
    cmp_base = _compare(current_by_task, base_by_task, frozen30)
    trace = _trace_stats(run_dir)
    monitor_dir = variant_root / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    recent = sorted(current_rows, key=lambda r: float(r.get("mtime") or 0.0), reverse=True)[:10]
    lines = [
        "# LB-MCTS V2 运行监控",
        "",
        f"- 更新时间：{now}",
        f"- run_dir：`{run_dir}`",
        f"- 进度：{n}/{len(expected) if expected else '?'} tasks",
        f"- 当前成功率：{succ}/{n} = {rate:.1%}",
        f"- 当前平均步数：{avg_steps:.2f}",
        f"- frozen30 prefix 已完成：{psucc}/{pn} = {prate:.1%}，平均步数 {pavg:.2f}",
        "",
        "## frozen30 overlap 对比",
        "",
        f"- 对旧 LB-MCTS 30task：overlap {len(cmp_lb30['overlap'])}，当前 {cmp_lb30['current_success']}，旧版 {cmp_lb30['reference_success']}，rescued {len(cmp_lb30['rescued'])}，broken {len(cmp_lb30['broken'])}",
        f"- 对 B0 baseline 30task：overlap {len(cmp_base['overlap'])}，当前 {cmp_base['current_success']}，baseline {cmp_base['reference_success']}，rescued {len(cmp_base['rescued'])}，broken {len(cmp_base['broken'])}",
        "",
        "### 相对旧 LB-MCTS 30task broken",
        "",
        *(f"- {task}" for task in cmp_lb30["broken"][:20]),
        "",
        "### 相对 baseline broken",
        "",
        *(f"- {task}" for task in cmp_base["broken"][:20]),
        "",
        "## exploration / rollback / evidence 指标",
        "",
        f"- exploration step rows：{trace['step_rows']}",
        f"- attempts：attempted={trace['attempted']}，executed={trace['executed']}，depth2={trace['depth2']}",
        f"- rollback：events={trace['rollback_events']}，level2={trace['rollback_level2']}，fail={trace['rollback_fail']}",
        f"- evidence：decisions={trace['evidence_decisions']}，injected={trace['evidence_injected']}，prompt_hints={trace['prompt_hints']}",
        f"- hint hit/follow：rows={trace['hint_rows']}，hit={trace['hint_hit']}，follow={trace['hint_follow']}",
        "",
        "## 最近完成任务",
        "",
    ]
    for row in recent:
        lines.append(
            f"- {row.get('task')}: success={bool(row.get('success'))}, steps={row.get('episode_length')}, exception={row.get('exception') or '-'}"
        )
    text = "\n".join(lines) + "\n"
    (monitor_dir / "latest_progress_cn.md").write_text(text, encoding="utf-8")
    snapshot = monitor_dir / f"progress_{datetime.now().strftime('%Y%m%dT%H%M%S')}.md"
    snapshot.write_text(text, encoding="utf-8")
    summary = {
        "updated_at": now,
        "run_dir": str(run_dir),
        "completed": n,
        "expected": len(expected),
        "success": succ,
        "success_rate": rate,
        "avg_steps": avg_steps,
        "frozen30_completed": pn,
        "frozen30_success": psucc,
        "trace": trace,
        "lb30_broken": cmp_lb30["broken"],
        "baseline_broken": cmp_base["broken"],
        "lb30_rescued": cmp_lb30["rescued"],
        "baseline_rescued": cmp_base["rescued"],
    }
    (monitor_dir / "latest_progress.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[monitor] {now} completed={n}/{len(expected) if expected else '?'} success={succ}/{n} rate={rate:.1%}", flush=True)
    return n, len(expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-root", default="/Users/huangrunxi/Projects/android_world/results/lb_mcts_final_116task_v2/LB_MCTS_FINAL")
    parser.add_argument("--expected-tasks-file", default="/Users/huangrunxi/Projects/android_world/results/lb_mcts_final_116task_v2/ordered_116task_tasks.txt")
    parser.add_argument("--lb30-root", default="/Users/huangrunxi/Projects/android_world/results/lb_mcts_final_30task/LB_MCTS_FINAL")
    parser.add_argument("--baseline-root", default="/Users/huangrunxi/Projects/android_world/results/strategy_30task_split/sensys_strategy30_v1/machine_1/B0_BASELINE_RERUN")
    parser.add_argument("--interval-sec", type=int, default=600)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    while True:
        completed, expected = _write_monitor(args)
        if not args.loop:
            return 0
        if expected and completed >= expected:
            return 0
        if completed == 0 and not _experiment_process_alive():
            return 1
        time.sleep(max(30, int(args.interval_sec)))


if __name__ == "__main__":
    raise SystemExit(main())
