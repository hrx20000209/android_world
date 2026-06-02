#!/usr/bin/env python3
"""Resume the Phase1 high-budget strategy comparison experiment.

Usage examples:
  python scripts/resume_phase1_strategy_compare.py
  python scripts/resume_phase1_strategy_compare.py --run-root results/phase1_strategy_compare_fixed_v3/run_20260602T094817
  python scripts/resume_phase1_strategy_compare.py --variants S1_BFS,S2_DFS
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_phaseA_high_budget_upper_bound as phase  # pylint: disable=wrong-import-position


DEFAULT_BASE = REPO_ROOT / "results" / "phase1_strategy_compare_fixed_v3"
DEFAULT_VARIANTS = [
    "B0_BASELINE_RERUN",
    "S1_BFS",
    "S2_DFS",
    "S3_BEAM",
    "S4_MCTS",
    "S5_OPERATOR_STRATIFIED_BEST_FIRST",
]


def _latest_root() -> Path:
    roots = sorted(DEFAULT_BASE.glob("run_*"))
    if roots:
        return roots[-1]
    return DEFAULT_BASE / ("run_" + datetime.now().strftime("%Y%m%dT%H%M%S"))


def _latest_variant_run(variant_dir: Path) -> Path | None:
    runs = sorted(variant_dir.glob("run_*"))
    return runs[-1] if runs else None


def _csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open(newline="") as f:
            return sum(1 for _ in csv.DictReader(f))
    except Exception:
        return 0


def _variant_complete(run_root: Path, variant: str, expected_tasks: int) -> bool:
    variant_dir = run_root / variant
    latest = _latest_variant_run(variant_dir)
    candidates = [variant_dir / "per_task_results.csv"]
    if latest is not None:
        candidates.append(latest / "per_task_results.csv")
    return any(_csv_rows(path) >= expected_tasks for path in candidates)


def _active_experiment_processes() -> list[str]:
    pattern = "run_phaseA_high_budget|run_exploration_experiment|android_world.*run.py|explore_gelab_agent_light"
    proc = subprocess.run(
        ["pgrep", "-af", pattern],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    current_pid = str(os.getpid())
    lines = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        pid = line.split(maxsplit=1)[0]
        if pid == current_pid:
            continue
        lines.append(line)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--new-root", action="store_true")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--expected-tasks", type=int, default=12)
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore active experiment process guard.")
    args = parser.parse_args()

    if args.new_root:
        run_root = DEFAULT_BASE / ("run_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    else:
        run_root = args.run_root or _latest_root()
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    active = _active_experiment_processes()
    if active and not args.force:
        print("[resume] active experiment process detected; not starting another run.")
        for line in active:
            print("[resume] " + line)
        print("[resume] Wait for it to finish, kill it intentionally, or rerun with --force.")
        return 2

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    tasks_str = ",".join(phase.PHASE_A_TASKS)
    rc = 0

    print(f"[resume] run_root={run_root}", flush=True)
    for variant in variants:
        if not args.rerun_complete and _variant_complete(run_root, variant, args.expected_tasks):
            print(f"[resume] skip complete variant={variant}", flush=True)
            continue
        print(f"[resume] starting variant={variant}", flush=True)
        rc |= phase._run_variant(run_root, variant, tasks_str, args.expected_tasks)  # pylint: disable=protected-access
        print(f"[resume] finished variant={variant} cumulative_rc={rc}", flush=True)

    explore_variants = [v for v in variants if v != "B0_BASELINE_RERUN"]
    print("[resume] offline replay/report", flush=True)
    phase._offline_replay(run_root, explore_variants)  # pylint: disable=protected-access
    report = phase._write_phase_report(run_root, variants)  # pylint: disable=protected-access
    print(f"[resume] report={report}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
