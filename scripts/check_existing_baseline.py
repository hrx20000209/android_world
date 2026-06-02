#!/usr/bin/env python3
"""Inventory existing AndroidWorld baseline runs.

The script is intentionally conservative: it prefers reusing a baseline only
when it can see per-task rows and the path/runtime indicates no exploration.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception:
        return []


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def _read_text(path: Path, limit: int = 200000) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def _task_name(row: dict[str, Any]) -> str:
    for key in ("task", "task_id", "task_name", "name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _load_selected_tasks(tasks_csv: Path | None) -> list[str]:
    if not tasks_csv or not tasks_csv.exists():
        return []
    rows = _read_csv(tasks_csv)
    tasks = [_task_name(r) for r in rows]
    return [t for t in tasks if t]


def _extract_task_rows(root: Path) -> list[dict[str, Any]]:
    csv_rows = _read_csv(root / "per_task_results.csv")
    if csv_rows:
        return [dict(r) for r in csv_rows if _task_name(r)]
    summary = _read_json(root / "summary.json")
    rows = ((summary.get("episodes") or {}).get("task_episode_rows") or [])
    if isinstance(rows, list) and rows:
        return [dict(r) for r in rows if isinstance(r, dict) and _task_name(r)]
    report_summary = _read_json(root / "report" / "summary.json")
    rows = ((report_summary.get("episodes") or {}).get("task_episode_rows") or [])
    if isinstance(rows, list) and rows:
        return [dict(r) for r in rows if isinstance(r, dict) and _task_name(r)]
    return []


def _find_runtime_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in ("runtime_config.yaml", "runtime_config.json", "androidworld_run.log", "driver.log"):
        direct = root / name
        if direct.exists():
            files.append(direct)
    for pattern in ("run_*/androidworld_run.log", "run_*/traces/*/runtime_config.yaml", "run_*/traces/*/runtime_config.json"):
        files.extend(root.glob(pattern))
    return files[:20]


def _runtime_text(root: Path) -> str:
    return "\n".join(_read_text(p, limit=80000) for p in _find_runtime_files(root))


def _infer_max_steps(text: str) -> str:
    patterns = [
        r"--max_n_steps=(\d+)",
        r"ANDROID_WORLD_MAX_STEPS=(\d+)",
        r"ANDROID_WORLD_MAX_N_STEPS=(\d+)",
        r"max_step_limit:\s*(\d+)",
        r'"max_step_limit"\s*:\s*(\d+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return "unknown"


def _infer_evaluator(text: str) -> dict[str, str]:
    def grab(pattern: str) -> str:
        m = re.search(pattern, text)
        return m.group(1) if m else ""

    return {
        "suite_family": grab(r"--suite_family=([^\s]+)") or "android_world",
        "task_random_seed": grab(r"--task_random_seed=(\d+)") or "",
        "fixed_task_seed": "true" if "--fixed_task_seed" in text else ("false" if "--nofixed_task_seed" in text else ""),
        "answer_normalization": "AndroidWorld evaluator default; no custom normalizer found",
    }


def _is_baseline_path(root: Path, text: str) -> bool:
    joined = (str(root) + "\n" + text[:5000]).lower()
    if any(token in joined for token in ("b0_baseline", "baseline_rerun", "baseline", "gelabresizeagent")):
        return True
    if "android_world_light_explore_enable=0" in joined or "explore_enable=false" in joined:
        return True
    if '"exploration_triggered": false' in joined and "light_explore" not in joined:
        return True
    return False


def _candidate_roots(results_root: Path) -> list[Path]:
    roots: set[Path] = set()
    names = {
        "per_task_results.csv",
        "variant_summary.csv",
        "summary.json",
        "androidworld_run.log",
        "driver.log",
    }
    for name in names:
        for path in results_root.rglob(name):
            roots.add(path.parent)
            if path.parent.name == "report":
                roots.add(path.parent.parent)
    return sorted(roots, key=lambda p: str(p))


def build_inventory(results_roots: list[Path], selected_tasks: list[str]) -> dict[str, Any]:
    selected_set = set(selected_tasks)
    candidates: list[dict[str, Any]] = []
    for results_root in results_roots:
        if not results_root.exists():
            continue
        for root in _candidate_roots(results_root):
            text = _runtime_text(root)
            task_rows = _extract_task_rows(root)
            if not task_rows and not text:
                continue
            is_baseline = _is_baseline_path(root, text)
            if not is_baseline:
                continue
            tasks = [_task_name(r) for r in task_rows if _task_name(r)]
            task_count = len(set(tasks)) if tasks else 0
            success_count = sum(1 for r in task_rows if str(r.get("success", "")).lower() in {"1", "true", "yes"})
            max_steps = _infer_max_steps(text)
            evaluator = _infer_evaluator(text)
            matched_selected = bool(selected_set and selected_set.issubset(set(tasks)))
            candidates.append(
                {
                    "path": str(root.resolve()),
                    "task_count": task_count,
                    "success_count": success_count,
                    "success_rate": (float(success_count) / task_count if task_count else None),
                    "max_steps": max_steps,
                    "evaluator": evaluator,
                    "tasks": sorted(set(tasks)),
                    "matched_selected_tasks": matched_selected,
                    "has_per_task_results": bool(task_rows),
                    "has_androidworld_log": bool((root / "androidworld_run.log").exists() or list(root.glob("run_*/androidworld_run.log"))),
                    "reusable": bool(task_rows and task_count > 0),
                }
            )
    candidates.sort(key=lambda x: (bool(x.get("matched_selected_tasks")), int(x.get("task_count") or 0)), reverse=True)
    by_size = {
        "baseline_116": next((c for c in candidates if int(c.get("task_count") or 0) >= 116), None),
        "baseline_50": next((c for c in candidates if int(c.get("task_count") or 0) >= 50), None),
        "baseline_30": next((c for c in candidates if int(c.get("task_count") or 0) >= 30), None),
        "matched_selected": next((c for c in candidates if c.get("matched_selected_tasks")), None),
    }
    return {
        "selected_tasks": selected_tasks,
        "candidate_count": len(candidates),
        "candidates": candidates,
        **by_size,
    }


def write_report(inventory: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "baseline_inventory_report_cn.md"
    inv_json = out_dir / "baseline_inventory.json"
    inv_json.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    b116 = inventory.get("baseline_116") or {}
    b50 = inventory.get("baseline_50") or {}
    b30 = inventory.get("baseline_30") or {}
    matched = inventory.get("matched_selected") or {}

    def yes(item: dict[str, Any]) -> str:
        return "是" if item else "否"

    lines = [
        "# Baseline Inventory Report",
        "",
        "## 结论",
        "",
        f"1. 是否存在有效 116-task baseline：{yes(b116)}",
        f"2. 116-task baseline 路径：`{b116.get('path', '')}`",
        f"3. 116-task task list 数量：`{len(b116.get('tasks', []) or [])}`",
        f"4. 116-task max step：`{b116.get('max_steps', 'unknown')}`",
        f"5. evaluator / answer normalization：`{(b116.get('evaluator') or {}).get('answer_normalization', '')}`",
        f"6. 是否可复用做最终比较：{yes(matched or b116)}",
        f"7. 不可复用原因：`{'无 matched selected baseline' if not (matched or b116) else ''}`",
        f"8. 是否存在 matched 30-task baseline：{yes(matched if int((matched or {}).get('task_count') or 0) >= 30 else b30)}",
        f"9. 是否存在 matched 50-task baseline：{yes(matched if int((matched or {}).get('task_count') or 0) >= 50 else b50)}",
        "",
        "## 可复用优先级",
        "",
        f"- matched selected baseline: `{matched.get('path', '')}`",
        f"- 116-task baseline: `{b116.get('path', '')}`",
        f"- 50-task baseline: `{b50.get('path', '')}`",
        f"- 30-task baseline: `{b30.get('path', '')}`",
        "",
        "## Candidates",
        "",
        "| path | task_count | success | max_steps | matched_selected | reusable |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for c in inventory.get("candidates", [])[:40]:
        lines.append(
            f"| `{c.get('path')}` | {c.get('task_count')} | {c.get('success_count')} | "
            f"{c.get('max_steps')} | {c.get('matched_selected_tasks')} | {c.get('reusable')} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", action="append", default=[str(DEFAULT_RESULTS_ROOT)])
    parser.add_argument("--tasks_csv", default="")
    parser.add_argument("--output_dir", default=str(DEFAULT_RESULTS_ROOT / "lb_mcts_final"))
    args = parser.parse_args()

    tasks_csv = Path(args.tasks_csv).expanduser().resolve() if args.tasks_csv else None
    selected_tasks = _load_selected_tasks(tasks_csv)
    roots = [Path(p).expanduser().resolve() for p in args.results_root]
    inventory = build_inventory(roots, selected_tasks)
    report = write_report(inventory, Path(args.output_dir).expanduser().resolve())
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
