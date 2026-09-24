#!/usr/bin/env python3
"""Run the complete MobileExplorer + Semantic Prefix experiment matrix.

This is an orchestration/reporting script only.  Each task is still executed
by ``run_sensys30_online_task.py`` and therefore uses the normal AndroidWorld
evaluator, action parser, and five-probe shadow runner.  It intentionally runs
tasks serially because they share the Android device.

The default task list is imported from the existing frozen 40-task selection,
so no new task sampling is introduced here.  ``--limit`` is useful for the
smoke pass; omit it for the full matrix.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_sensys30_online_task.py"
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


def _tasks() -> list[str]:
  scripts_dir = str(REPO_ROOT / "scripts")
  if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
  from run_aggressive_t2_shortcut_frozen_40task import TASKS
  return [str(item["task_id"]) for item in TASKS]


ARMS: dict[str, dict[str, Any]] = {
    # Focused-study control: identical reasoning/evaluator path without the
    # parallel explorer. It is selected explicitly, not used as an Ex5 arm.
    "Z_REASONING_ONLY": {
        "semantic_prefix_mode": "off", "executable_memory": False,
        "inject_evidence": False, "ranker": "InformationNeedRanker",
        "variant": "offline", "two_system": False,
    },
    # A is the unchanged Ex5 shadow-exploration baseline.
    "A_EX5_BASELINE": {
        "semantic_prefix_mode": "off", "executable_memory": False,
        "inject_evidence": False, "ranker": "InformationNeedRanker",
    },
    # B is the previous exploration + executable graph design, before Prefix.
    "B_EXPLORATION_GRAPH": {
        "semantic_prefix_mode": "off", "executable_memory": True,
        "inject_evidence": True, "ranker": "InformationNeedRanker",
        "high_confidence_skip_enabled": False,
    },
    # C/D add the compact Prefix sidecar while retaining the complete graph.
    "C_GRAPH_PREFIX_LOGGING": {
        "semantic_prefix_mode": "logging", "executable_memory": True,
        "inject_evidence": True, "ranker": "InformationNeedRanker",
        "high_confidence_skip_enabled": False,
    },
    "D_GRAPH_PREFIX_PROMPT": {
        "semantic_prefix_mode": "prompt", "executable_memory": True,
        "inject_evidence": True, "ranker": "InformationNeedRanker",
        "high_confidence_skip_enabled": False,
    },
    # E is the complete MobileExplorer on one canonical persisted graph.
    # Ex5 still owns its transient safety/backtracking controller state; all
    # reusable state/action evidence, prompt retrieval, fusion and route assist
    # live in executable_memory.json (no SemanticPrefix sidecar).
    "E_FULL_MOBILEEXPLORER": {
        "semantic_prefix_mode": "off", "executable_memory": True,
        "inject_evidence": True, "ranker": "InformationNeedRanker",
        "high_confidence_skip_enabled": True,
    },
    # Explicit negative control; coverage is the primary exploration ranker.
    "F_COVERAGE_ONLY_NEGATIVE": {
        "semantic_prefix_mode": "coverage_only", "executable_memory": True,
        "inject_evidence": True, "ranker": "CoverageRanker",
        "high_confidence_skip_enabled": False,
    },
}


def _percentile(values: list[float], percentile: float) -> float | None:
  if not values:
    return None
  values = sorted(values)
  rank = (len(values) - 1) * percentile / 100.0
  lower, upper = math.floor(rank), math.ceil(rank)
  if lower == upper:
    return float(values[lower])
  return float(values[lower] + (values[upper] - values[lower]) * (rank - lower))


def _read_json(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}
  except (OSError, ValueError, json.JSONDecodeError):
    return {}


def _task_metrics(task_root: Path, log_text: str, task: str) -> dict[str, Any]:
  steps_path = task_root / "step_latency.jsonl"
  probes_path = task_root / "probe_trace.jsonl"
  windows_path = task_root / "inference_windows.jsonl"
  steps: list[dict[str, Any]] = []
  probes: list[dict[str, Any]] = []
  windows: list[dict[str, Any]] = []
  for path, target in ((steps_path, steps), (probes_path, probes), (windows_path, windows)):
    if not path.is_file():
      continue
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
      try:
        value = json.loads(line)
      except json.JSONDecodeError:
        continue
      if isinstance(value, dict):
        target.append(value)
  agent_actions: list[dict[str, Any]] = []
  for action_path in task_root.glob("agent_traces/*/action.jsonl"):
    for line in action_path.read_text(encoding="utf-8", errors="replace").splitlines():
      try:
        value = json.loads(line)
      except json.JSONDecodeError:
        continue
      if isinstance(value, dict):
        agent_actions.append(value)
  prefix = _read_json(task_root / "semantic_prefix_summary.json")
  prefix_metrics = prefix.get("delta_metrics") if isinstance(prefix.get("delta_metrics"), dict) else None
  if not isinstance(prefix_metrics, dict):
    prefix_metrics = prefix.get("metrics") if isinstance(prefix.get("metrics"), dict) else {}
  executable = _read_json(task_root / "executable_memory_summary.json")
  executable_metrics = (
      executable.get("metrics") if isinstance(executable.get("metrics"), dict) else {}
  )
  guidance_windows = [
      row.get("executable_memory_guidance")
      for row in windows
      if isinstance(row.get("executable_memory_guidance"), dict)
  ]
  discovered_elements = 0
  discovered_texts: set[str] = set()
  discovered_pages: set[tuple[str, str]] = set()
  for row in probes:
    discovered = row.get("discovered")
    if isinstance(discovered, dict):
      discovered_elements += int(discovered.get("new_element_count") or 0)
      discovered_texts.update(
          str(value).strip() for value in (discovered.get("new_texts") or [])
          if str(value).strip()
      )
    graph = row.get("graph") if isinstance(row.get("graph"), dict) else {}
    destination = (
        row.get("destination") or row.get("dst") or row.get("after_state")
        or graph.get("dst")
    )
    if isinstance(destination, dict):
      activity = str(destination.get("activity") or destination.get("activity_name") or "")
      layout = str(
          destination.get("layout_signature") or destination.get("layout_sig")
          or destination.get("struct_signature") or destination.get("state_id") or ""
      )
      if activity or layout:
        discovered_pages.add((activity, layout))
  # run.py prints the evaluator table.  A missing line is unknown, never a
  # fabricated failure; this is important when a runner aborts before eval.
  success: float | None = None
  # Prefer the evaluator's per-task row.  The aggregate mean_success_rate is
  # absent in some short/early-stop runs, so use it only as a fallback.
  row_matches = re.findall(
      rf"^\s*{re.escape(task)}\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+",
      log_text,
      re.MULTILINE,
  )
  if row_matches:
    try:
      success = float(row_matches[-1])
    except ValueError:
      success = None
  if success is None:
    matches = re.findall(r"mean_success_rate\s+([0-9.]+)", log_text)
    if matches:
      try:
        success = float(matches[-1])
      except ValueError:
        success = None
  evaluator_error = None
  if f"Logging exception and skipping task. Task: {task}:" in log_text:
    evaluator_error = "runner exception skipped the task before evaluator completion"
    # AndroidWorld prints a zero-success aggregate for an exception that
    # prevented evaluation. Such a row is infrastructure-unknown, not a
    # measured task failure, and must not enter the success-rate denominator.
    success = None
  if "Must use a11y_grpc_wrapper.A11yGrpcWrapper" in log_text:
    evaluator_error = "task evaluator requires AndroidWorld gRPC accessibility wrapper"
    # The suite prints a zero-valued aggregate row when the evaluator raises
    # before any trial is completed. That is infrastructure-unknown, not a
    # task failure and must not enter the success-rate denominator.
    success = None
  if f"SKIPPING {task}." in log_text and "Could not get a11y tree" in log_text:
    evaluator_error = "AndroidWorld accessibility tree unavailable before task actions"
    success = None
  return {
      "task": task,
      "success": success,
      "evaluator_error": evaluator_error,
      "evaluation_status": "infrastructure_unknown" if evaluator_error else "evaluated",
      "steps": len(steps),
      "step_latency_s": [float(row.get("step_total_s") or 0.0) for row in steps],
      "probe_count": len(probes),
      "exploration_new_element_count": discovered_elements,
      "exploration_unique_text_count": len(discovered_texts),
      "exploration_unique_page_count": len(discovered_pages),
      "reasoning_count": len(steps),
      "open_app_reasoning_skips": sum(
          bool((row.get("pre_reasoning_bootstrap") or {}).get("package_match"))
          for row in steps
      ),
      "probe_windows": len(windows),
      "five_probe_complete_windows": sum(bool(row.get("five_probe_complete")) for row in windows),
      "graph_guidance": {
          "windows": len(guidance_windows),
          "state_matched_windows": sum(bool(row.get("state_matched")) for row in guidance_windows),
          "candidates_scored": sum(int(row.get("candidate_count") or 0) for row in guidance_windows),
          "seen_candidate_scores": sum(int(row.get("seen_count") or 0) for row in guidance_windows),
          "unseen_candidate_scores": sum(int(row.get("unseen_count") or 0) for row in guidance_windows),
          "mature_candidate_scores": sum(int(row.get("mature_count") or 0) for row in guidance_windows),
          "task_relevant_candidate_scores": sum(int(row.get("relevant_count") or 0) for row in guidance_windows),
          "revalidation_candidate_scores": sum(int(row.get("revalidation_count") or 0) for row in guidance_windows),
          "repeat_suppressed_candidates": sum(int(row.get("repeat_suppressed_count") or 0) for row in guidance_windows),
          "known_noop_suppressed_candidates": sum(int(row.get("known_noop_suppressed_count") or 0) for row in guidance_windows),
          "side_effect_veto_candidates": sum(int(row.get("side_effect_veto_count") or 0) for row in guidance_windows),
          "trap_suppressed_candidates": sum(int(row.get("trap_suppressed_count") or 0) for row in guidance_windows),
      },
      "stop_reasons": {
          str(row.get("stop_reason") or "unknown"): sum(
              1 for item in windows if str(item.get("stop_reason") or "unknown") == str(row.get("stop_reason") or "unknown")
          ) for row in windows
      },
      "prefix": prefix_metrics,
      "prefix_node_count": int(prefix.get("node_count") or 0),
      "prefix_edge_count": int(prefix.get("edge_count") or 0),
      "prefix_average_depth": float(prefix.get("average_depth") or 0.0),
      "parse_error_count": max(
          int(prefix_metrics.get("parse_error_count") or 0),
          sum(bool(row.get("parse_error")) for row in agent_actions),
      ),
      "executable": executable_metrics,
      "graph_node_count": int(executable.get("node_count") or 0),
      "graph_edge_count": int(executable.get("edge_count") or 0),
      "graph_sample_count": int(executable.get("sample_count") or 0),
      "graph_skill_count": int(executable.get("skill_count") or 0),
  }


def _aggregate(arm: str, task_rows: list[dict[str, Any]]) -> dict[str, Any]:
  evaluated_rows = [row for row in task_rows if row.get("success") is not None]
  known_success = [float(row["success"]) for row in evaluated_rows]
  steps = [float(row["steps"]) for row in evaluated_rows]
  latencies = [float(value) for row in evaluated_rows for value in row.get("step_latency_s", [])]
  prefix_rows = [row.get("prefix") or {} for row in task_rows]
  executable_rows = [row.get("executable") or {} for row in task_rows]
  guidance_rows = [row.get("graph_guidance") or {} for row in task_rows]
  def total(key: str) -> int:
    return sum(int(row.get(key) or 0) for row in prefix_rows)
  def graph_max(key: str) -> int:
    return max((int(row.get(key) or 0) for row in executable_rows), default=0)
  return {
      "arm": arm,
      "task_count": len(task_rows),
      "evaluated_task_count": len(known_success),
      "success_rate": (sum(known_success) / len(known_success) if known_success else None),
      "step_mean": statistics.mean(steps) if steps else None,
      "step_median": statistics.median(steps) if steps else None,
      "step_p95": _percentile(steps, 95),
      "step_latency_mean_s": statistics.mean(latencies) if latencies else None,
      "exploration_probe_count": sum(int(row.get("probe_count") or 0) for row in task_rows),
      "exploration_new_element_count": sum(
          int(row.get("exploration_new_element_count") or 0) for row in task_rows),
      "exploration_unique_text_count": sum(
          int(row.get("exploration_unique_text_count") or 0) for row in task_rows),
      "exploration_unique_page_count": sum(
          int(row.get("exploration_unique_page_count") or 0) for row in task_rows),
      "reasoning_count": sum(int(row.get("reasoning_count") or 0) for row in task_rows),
      "five_probe_complete_windows": sum(int(row.get("five_probe_complete_windows") or 0) for row in task_rows),
      "prefix_route_candidate_count": total("route_candidates"),
      "prefix_route_hit_count": total("route_hits"),
      "prefix_route_miss_count": total("route_misses"),
      "graph_route_candidate_count": graph_max("retrieval_path_candidates"),
      "graph_route_hit_count": graph_max("route_hit_count"),
      "graph_route_miss_count": graph_max("route_miss_count"),
      "parse_error_count": sum(int(row.get("parse_error_count") or 0) for row in task_rows),
      "prefix_node_count": max((int(row.get("prefix_node_count") or 0) for row in task_rows), default=0),
      "prefix_edge_count": max((int(row.get("prefix_edge_count") or 0) for row in task_rows), default=0),
      "prefix_average_depth_mean": statistics.mean(
          [float(row.get("prefix_average_depth") or 0.0) for row in task_rows]
      ) if task_rows else 0.0,
      "graph_node_count": max((int(row.get("graph_node_count") or 0) for row in task_rows), default=0),
      "graph_edge_count": max((int(row.get("graph_edge_count") or 0) for row in task_rows), default=0),
      "graph_sample_count": max((int(row.get("graph_sample_count") or 0) for row in task_rows), default=0),
      "graph_skill_count": max((int(row.get("graph_skill_count") or 0) for row in task_rows), default=0),
      "graph_guidance": {
          key: sum(int(row.get(key) or 0) for row in guidance_rows)
          for key in ("windows", "state_matched_windows", "candidates_scored",
                      "seen_candidate_scores", "mature_candidate_scores",
                      "unseen_candidate_scores", "task_relevant_candidate_scores",
                      "revalidation_candidate_scores", "repeat_suppressed_candidates",
                      "known_noop_suppressed_candidates",
                      "side_effect_veto_candidates",
                      "trap_suppressed_candidates")
      },
      "graph_prompt_adoptions": graph_max("prompt_adoptions"),
      "coordinate_corrections": graph_max("coordinate_corrections"),
      "graph_overrides": graph_max("graph_overrides"),
      "graph_rejections": graph_max("graph_rejections"),
      "graph_skip_attempts": graph_max("skip_attempts"),
      "graph_skip_hits": graph_max("skip_hits"),
      "unknown_evaluation_count": len(task_rows) - len(known_success),
  }


def _preflight(api_url: str, metrics_url: str) -> None:
  model_url = api_url.rsplit("/v1/", 1)[0] + "/v1/models" if "/v1/" in api_url else api_url.rstrip("/") + "/models"
  for label, url in (("models", model_url), ("metrics", metrics_url)):
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    print(f"[preflight] {label}: {url} -> {response.status_code}", flush=True)


def _restart_grpc_accessibility_forwarder(adb_path: str, serial: str) -> None:
  """Re-arm AndroidWorld's evaluator service after restoring an AVD snapshot."""
  service = (
      "com.google.androidenv.accessibilityforwarder/"
      "com.google.androidenv.accessibilityforwarder.AccessibilityForwarder"
  )
  commands = (
      ["wait-for-device"],
      ["shell", "input", "keyevent", "3"],
      ["shell", "settings", "put", "secure", "accessibility_enabled", "1"],
      ["shell", "settings", "put", "secure", "enabled_accessibility_services", service],
      ["shell", "am", "force-stop", "com.google.androidenv.accessibilityforwarder"],
      ["shell", "input", "keyevent", "3"],
  )
  for command in commands:
    result = subprocess.run(
        [adb_path, "-s", serial, *command], capture_output=True,
        text=True, timeout=30.0, check=False,
    )
    if result.returncode:
      raise RuntimeError(
          f"Could not restart gRPC accessibility forwarder ({command}): "
          f"{(result.stdout + result.stderr)[-1200:]}"
      )
  time.sleep(3.0)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_root", type=Path, required=True)
  parser.add_argument("--api_url", default="http://127.0.0.1:18084/v1/chat/completions")
  parser.add_argument("--metrics_url", default="http://127.0.0.1:18084/metrics")
  parser.add_argument("--model", default="GELAB-ZERO-4B")
  parser.add_argument("--seed", type=int, default=30)
  parser.add_argument("--max_steps", type=int, default=0)
  parser.add_argument("--console_port", type=int, default=5554)
  parser.add_argument(
      "--reset_snapshot", default="",
      help="Restore this emulator snapshot before every task, identically for each arm.",
  )
  parser.add_argument("--max_probes", type=int, default=5)
  parser.add_argument("--min_probes", type=int, default=5)
  parser.add_argument(
      "--memory_min_task_relevance", type=float, default=0.30,
      help="Minimum semantic relevance for fusion and executable route use.",
  )
  parser.add_argument(
      "--memory_min_prompt_relevance", type=float, default=0.30,
      help="Minimum semantic relevance for advisory graph context in the prompt.",
  )
  parser.add_argument(
      "--a11y_method", choices=("grpc", "fast_provider", "uiautomator"),
      default="grpc",
      help="Use gRPC for AndroidWorld task evaluators that request its wrapped accessibility tree.",
  )
  parser.add_argument(
      "--skip_open_app_reasoning", action=argparse.BooleanOptionalAction,
      default=True,
      help="Use the same deterministic, package-verified app bootstrap in each selected arm.",
  )
  parser.add_argument("--limit", type=int, default=0)
  parser.add_argument(
      "--tasks", nargs="*",
      help="Optional exact task list for a focused study; defaults to frozen 40.",
  )
  parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
  parser.add_argument("--skip_preflight", action="store_true")
  parser.add_argument(
      "--resume", action="store_true",
      help="Reuse completed task_metrics.json files under output_root and continue incomplete arms.",
  )
  args = parser.parse_args()
  if not args.skip_preflight:
    _preflight(args.api_url, args.metrics_url)
  frozen_tasks = _tasks()
  tasks = list(args.tasks) if args.tasks else frozen_tasks
  if args.tasks:
    from android_world import registry
    available = registry.TaskRegistry().get_registry(family="android_world")
    invalid = sorted(set(tasks) - set(available))
    if invalid:
      raise ValueError(f"Unknown AndroidWorld tasks: {invalid}")
  if args.limit:
    tasks = tasks[:args.limit]
  output_root = args.output_root.resolve()
  output_root.mkdir(parents=True, exist_ok=True)
  (output_root / "experiment_meta.json").write_text(json.dumps({
      "created_at": dt.datetime.now().isoformat(), "task_count": len(tasks),
      "tasks": tasks, "arms": args.arms, "seed": args.seed,
      "arm_configs": {name: ARMS[name] for name in args.arms},
      "model": args.model, "api_url": args.api_url, "metrics_url": args.metrics_url,
      "max_steps": args.max_steps, "max_probes": args.max_probes, "min_probes": args.min_probes,
      "memory_min_task_relevance": args.memory_min_task_relevance,
      "memory_min_prompt_relevance": args.memory_min_prompt_relevance,
      "a11y_method": args.a11y_method,
      "console_port": args.console_port,
      "reset_snapshot": args.reset_snapshot or None,
      "skip_open_app_reasoning": args.skip_open_app_reasoning,
      "ex5_mapping": "run_sensys30_online_task.py conservative parallel shadow path",
  }, ensure_ascii=False, indent=2), encoding="utf-8")

  all_summaries: list[dict[str, Any]] = []
  for arm in args.arms:
    arm_root = output_root / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    arm_config = ARMS[arm]
    executable_config_path = arm_root / "executable_memory_config.json"
    if arm_config.get("executable_memory"):
      executable_config_path.write_text(json.dumps({
          "enabled": True,
          "exploration_enabled": True,
          "exploration_guidance_enabled": True,
          "graph_enabled": True,
          "graph_prompt_enabled": True,
          "post_fusion_enabled": True,
          "high_confidence_skip_enabled": bool(
              arm_config.get("high_confidence_skip_enabled", False)),
          "k_step_memory_enabled": True,
          "hard_negatives_enabled": True,
          "action_groups_enabled": True,
          "repeated_validation_enabled": True,
          "compact_graph_enabled": arm != "E_FULL_MOBILEEXPLORER",
          "max_probes": 5,
          "max_depth": 4,
          "override_confidence": 0.82,
          "min_task_relevance": max(0.0, min(1.0, args.memory_min_task_relevance)),
          "prompt_min_task_relevance": max(
              0.0, min(1.0, args.memory_min_prompt_relevance)),
      }, ensure_ascii=False, indent=2), encoding="utf-8")
    task_rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
      task_root = arm_root / f"{index:02d}_{task}"
      existing_metrics = task_root / "task_metrics.json"
      if args.resume and existing_metrics.is_file():
        try:
          existing_row = json.loads(existing_metrics.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
          existing_row = None
        if (
            isinstance(existing_row, dict)
            and existing_row.get("task") == task
            and int(existing_row.get("returncode", 1)) == 0
        ):
          task_rows.append(existing_row)
          print(
              f"[resume-skip {index}/{len(tasks)}] arm={arm} task={task} "
              f"returncode={existing_row.get('returncode')} success={existing_row.get('success')}",
              flush=True,
          )
          continue
      if task_root.exists():
        # Preserve partial checkpoints/logs from an interrupted attempt and
        # give the single-task runner the fresh directory it requires.
        suffix = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        archived = task_root.with_name(f"{task_root.name}.interrupted-{suffix}")
        task_root.rename(archived)
      # The single-task runner intentionally requires a fresh output root
      # (exist_ok=False).  Keep the live log beside that root so it can be
      # opened before the runner creates its artifacts.
      log_path = arm_root / f"{index:02d}_{task}.runner.log"
      if log_path.exists():
        suffix = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path.rename(log_path.with_name(f"{log_path.stem}.interrupted-{suffix}{log_path.suffix}"))
      if args.reset_snapshot:
        adb_path = (
            shutil.which("adb")
            or os.environ.get("ADB_PATH")
            or "/Users/huangrunxi/Library/Android/sdk/platform-tools/adb"
        )
        serial = f"emulator-{args.console_port}"
        restored = subprocess.run(
            [adb_path, "-s", serial, "emu", "avd", "snapshot", "load", args.reset_snapshot],
            capture_output=True, text=True, timeout=180.0, check=False,
        )
        if restored.returncode:
          raise RuntimeError(
              f"Snapshot restore failed for {serial}/{args.reset_snapshot}: "
              f"{(restored.stdout + restored.stderr)[-2000:]}"
          )
        ready = subprocess.run(
            [adb_path, "-s", serial, "wait-for-device"],
            capture_output=True, text=True, timeout=120.0, check=False,
        )
        boot = subprocess.run(
            [adb_path, "-s", serial, "shell", "getprop", "sys.boot_completed"],
            capture_output=True, text=True, timeout=30.0, check=False,
        )
        if ready.returncode or boot.returncode or boot.stdout.strip() != "1":
          raise RuntimeError(
              f"Emulator did not become ready after snapshot restore: "
              f"{(ready.stdout + ready.stderr + boot.stdout + boot.stderr)[-2000:]}"
          )
        _restart_grpc_accessibility_forwarder(adb_path, serial)
        (arm_root / f"{index:02d}_{task}.snapshot_restore.json").write_text(
            json.dumps({
                "serial": serial, "snapshot": args.reset_snapshot,
                "restored_at": dt.datetime.now().isoformat(),
                "stdout": restored.stdout.strip(),
                "grpc_accessibility_forwarder_restarted": True,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
      command = [
          sys.executable, str(RUNNER), "--output", str(task_root), "--task", task,
          "--seed", str(args.seed), "--max_steps", str(args.max_steps),
          "--console_port", str(args.console_port),
          "--metrics_url", args.metrics_url,
          "--max_probes", str(args.max_probes), "--min_probes", str(args.min_probes),
          "--agent_name", (
              "mobileexplorer_executable"
              if arm_config.get("executable_memory") else "mobileexplorer_no_memory"),
          "--ranker", str(arm_config["ranker"]),
          "--semantic_prefix_mode", str(arm_config["semantic_prefix_mode"]),
          "--variant", str(arm_config.get("variant", "full")),
          "--a11y_method", args.a11y_method,
      ]
      if arm_config.get("two_system", True):
        command.append("--two_system")
      if args.skip_open_app_reasoning:
        command.append("--skip_open_app_reasoning")
      if arm_config.get("executable_memory"):
        command.extend([
            "--executable_memory",
            "--executable_memory_config", str(executable_config_path),
            "--executable_memory_path", str(arm_root / "executable_memory.json"),
        ])
      if arm_config.get("inject_evidence"):
        command.append("--inject_evidence")
      if arm_config["semantic_prefix_mode"] != "off":
        # Keep one versioned sidecar per arm so routes can accumulate across
        # the fixed task sequence, while never leaking evidence between
        # ablations.
        command.extend(["--semantic_prefix_path", str(arm_root / "semantic_prefix_memory.json")])
      env = dict(os.environ)
      env["ANDROID_WORLD_LLM_API_URL"] = args.api_url
      env["ANDROID_WORLD_LLAMACPP_MODEL"] = args.model
      env["ANDROID_WORLD_EXPERIMENT_ARM"] = arm
      # Keep per-reasoning-step screenshots and raw action traces with the
      # task artifacts so every failed case can be reproduced and reviewed.
      env["MOBILEEXPLORER_OUTPUT_PATH"] = str(task_root / "agent_traces")
      print(f"[run {index}/{len(tasks)}] arm={arm} task={task}", flush=True)
      started = time.time()
      with log_path.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND " + " ".join(command) + "\n")
        stream.flush()
        proc = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
      log_text = log_path.read_text(encoding="utf-8", errors="replace")
      row = _task_metrics(task_root, log_text, task)
      row.update({"arm": arm, "returncode": proc.returncode, "wall_time_s": time.time() - started})
      task_rows.append(row)
      # A runner-side argument/device failure may happen before it creates its
      # artifact directory. Preserve the arm-level result and log instead of
      # turning that diagnostic into an orchestrator exception.
      task_root.mkdir(parents=True, exist_ok=True)
      (task_root / "task_metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
      print(f"[done] arm={arm} task={task} returncode={proc.returncode} success={row['success']} probes={row['probe_count']}", flush=True)
    summary = _aggregate(arm, task_rows)
    summary["tasks"] = task_rows
    all_summaries.append(summary)
    (arm_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

  result = {"experiment_root": str(output_root), "arms": all_summaries}
  (output_root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
  lines = ["# MobileExplorer 单一权威图实验汇总", "", "各 arm 使用同一任务顺序、seed、模型和 step budget；E 固定 Ex5 probe 参数。E 的可持久化/检索/执行记忆统一位于 executable_memory.json；Ex5 安全控制图仅保留为任务内运行状态。", "", "| Arm | Success | Steps mean/median/p95 | Probes | New elements/pages | Reasoning | Graph nodes/edges/skills | EAM guidance matched/seen | Graph route cand/hit/miss | Parse errors |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
  for summary in all_summaries:
    success = "unknown" if summary["success_rate"] is None else f"{summary['success_rate']:.3f}"
    lines.append(
        f"| {summary['arm']} | {success} | {summary['step_mean']} / {summary['step_median']} / {summary['step_p95']} | "
        f"{summary['exploration_probe_count']} | {summary['exploration_new_element_count']} / {summary['exploration_unique_page_count']} | {summary['reasoning_count']} | "
        f"{summary['graph_node_count']} / {summary['graph_edge_count']} / {summary['graph_skill_count']} | "
        f"{summary['graph_guidance']['state_matched_windows']} / {summary['graph_guidance']['seen_candidate_scores']} | "
        f"{summary['graph_route_candidate_count']} / {summary['graph_route_hit_count']} / {summary['graph_route_miss_count']} | "
        f"{summary['parse_error_count']} |"
    )
  (output_root / "summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
