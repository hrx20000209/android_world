#!/usr/bin/env python3
"""Run one GELAB AndroidWorld task with live probing during each VLM call."""

from __future__ import annotations

import argparse
import contextvars
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import re
import runpy
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world.agents import mobileexplorer
from android_world.parallel_exploration.online_evidence_index import OnlineEvidenceIndex
from android_world.agents import gelab_agent
from android_world.agents import explore_agent_text
from android_world.agents import infer
from android_world.agents import m3a
from android_world.parallel_exploration.live_probe import prepare_explorer
from android_world.parallel_exploration.live_probe import run_prepared_explorer
from android_world.parallel_exploration.live_probe import stop_prepared_explorer
from android_world.parallel_exploration.live_probe import element_identity_from_dict
from android_world.parallel_exploration.belief_graph import GraphNode
from android_world.parallel_exploration.belief_graph import EdgeStatus
from android_world.parallel_exploration.belief_graph import NodeStatus
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
from android_world.parallel_exploration.belief_graph import control_key_from_identity
from android_world.parallel_exploration.belief_graph import stable_control_key_from_identity
from android_world.parallel_exploration.config import TwoSystemConfig
from android_world.parallel_exploration.executable_memory import ExecutableExplorationMemory
from android_world.parallel_exploration.executable_memory import ExecutableMemoryConfig
from android_world.parallel_exploration.executable_memory import JsonlMemoryLogger
from android_world.parallel_exploration.executable_memory import PageObservation
from android_world.parallel_exploration.executable_memory import goal_semantic_tokens
from android_world.parallel_exploration.executable_memory import is_task_specific_value
from android_world.parallel_exploration.executable_memory import is_dynamic_ui_text
from android_world.parallel_exploration.logging import InferenceRoundRecord
from android_world.parallel_exploration.logging import RoundLogger
from android_world.parallel_exploration.information import parse_information_need
from android_world.parallel_exploration.resources import ResourceAdaptiveBudgetController
from android_world.parallel_exploration.resources import ResourceMonitor
from android_world.parallel_exploration.runtime import SkipInferenceGate
from android_world.parallel_exploration.semantic_prefix_memory import SemanticPrefixMemory
from android_world.parallel_exploration.semantic_prefix_memory import state_id_from_signature
from android_world.parallel_exploration.state import create_optimized_state_capture
from android_world.env import json_action
from android_world.agents import base_agent


METRICS = {
    "ttft_s": "vllm:time_to_first_token_seconds_sum",
    "queue_s": "vllm:request_queue_time_seconds_sum",
    "inference_s": "vllm:request_inference_time_seconds_sum",
    "prefill_s": "vllm:request_prefill_time_seconds_sum",
    "decode_s": "vllm:request_decode_time_seconds_sum",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
}


def _append(path: Path, row: dict[str, Any]) -> None:
  with path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _instrument_graph_updates(path: Path) -> None:
  """Write per-task wall timings for graph mutations without nested double counts."""
  depth_var = contextvars.ContextVar("androidworld_graph_update_depth", default=0)
  for method_name in ("observe_page", "record_transition", "ingest_probe_row"):
    original = getattr(ExecutableExplorationMemory, method_name)
    if getattr(original, "_androidworld_graph_timed", False):
      continue

    def timed(self, *args, _original=original, _method_name=method_name, **kwargs):
      config = getattr(self, "config", None)
      enabled = bool(getattr(config, "enabled", False) and
                     getattr(config, "graph_enabled", False))
      depth = depth_var.get()
      token = depth_var.set(depth + 1)
      started = time.perf_counter()
      try:
        return _original(self, *args, **kwargs)
      finally:
        elapsed = max(0.0, time.perf_counter() - started)
        depth_var.reset(token)
        if enabled and depth == 0:
          _append(path, {"operation": _method_name, "elapsed_s": elapsed})

    timed._androidworld_graph_timed = True
    setattr(ExecutableExplorationMemory, method_name, timed)


def _action_signature(action_dict: dict[str, Any]) -> tuple[Any, ...]:
  action_type = str(action_dict.get("action_type", "")).lower()
  if action_type in {"click", "tap"}:
    return (action_type, action_dict.get("x"), action_dict.get("y"))
  return (action_type, action_dict.get("text"), action_dict.get("app_name"))


def _actions_near_duplicate(a: dict[str, Any], b: dict[str, Any], radius: float = 80.0) -> bool:
  """True if two real (authoritative) actions look like the same move.

  Used to detect that the model just repeated something it already did a
  few steps ago - a general, observed sign that whatever ran in between
  (typically a skip-executed graph edge) did not actually move the task
  forward, rather than a rule about any specific widget or app.
  """
  sig_a, sig_b = _action_signature(a), _action_signature(b)
  if sig_a[0] != sig_b[0]:
    return False
  if sig_a[0] in {"click", "tap"}:
    ax, ay, bx, by = sig_a[1], sig_a[2], sig_b[1], sig_b[2]
    if ax is None or ay is None or bx is None or by is None:
      return False
    return (float(ax) - float(bx)) ** 2 + (float(ay) - float(by)) ** 2 <= radius ** 2
  return sig_a == sig_b


def _stated_next_intent(summary: str) -> str:
  """The forward-looking half of a step summary, or "" if it has none.

  Summaries the model writes look like "I have successfully opened the
  Contacts app. The next step is to fill in the name field." Only the second
  sentence is about the decision the explorer must anticipate; the first is a
  recap of what already happened and would drag ranking toward elements that
  are already behind us.
  """
  text = str(summary or "").strip()
  if not text:
    return ""
  match = re.search(
      r"(?:the\s+next\s+step\s+is|next,|now\s+i\s+(?:need|will)|i\s+(?:need|will)\s+to)\b(.*)",
      text, flags=re.IGNORECASE | re.DOTALL,
  )
  return match.group(0).strip() if match else ""


def _deterministic_bootstrap_target(goal: str, activity: str, enabled: bool) -> str | None:
  """A conservative target-app lookup; unknown goals remain model-driven."""
  if not enabled or "nexuslauncher" not in str(activity or "").casefold():
    return None
  target = explore_agent_text._infer_goal_target_app(goal)  # pylint: disable=protected-access
  if not target or not explore_agent_text._APP_PACKAGE_HINTS.get(target):  # pylint: disable=protected-access
    return None
  return target


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--task", default="ClockStopWatchRunning")
  parser.add_argument("--max_steps", type=int, default=0)
  parser.add_argument("--seed", type=int, default=30)
  parser.add_argument("--ranker", choices=[
      "RandomRanker", "SimpleRelevanceRanker", "InformationNeedRanker",
      "LightweightLinearRanker", "CoverageRanker",
  ], default="InformationNeedRanker")
  parser.add_argument(
      "--selector_weights", type=Path,
      default=Path(__file__).resolve().parents[1] / "gui_exploration_selector" / "weights.txt",
      help="manual linear-selector weights used by LightweightLinearRanker",
  )
  parser.add_argument("--metrics_url", default="http://127.0.0.1:8081/metrics")
  parser.add_argument("--console_port", type=int, default=5554)
  parser.add_argument("--two_system", action="store_true")
  parser.add_argument("--enable_skip_inference", action="store_true")
  parser.add_argument(
      "--skip_open_app_reasoning", action="store_true",
      help=("On the launcher, use the known task target to open its app before "
            "the first reasoning call; fall back when target/package is unclear."),
  )
  parser.add_argument("--skip_confidence", type=float, default=0.90)
  parser.add_argument("--max_probes", type=int, default=5)
  parser.add_argument("--min_probes", type=int, default=5)
  parser.add_argument("--force_min_probes", action=argparse.BooleanOptionalAction,
                      default=False)
  parser.add_argument("--post_inference_grace_s", type=float, default=4.0)
  parser.add_argument("--max_depth", type=int, default=3)
  parser.add_argument("--max_exploration_time_s", type=float, default=8.0)
  parser.add_argument("--a11y_method", choices=["fast_provider", "uiautomator", "grpc"], default="fast_provider")
  parser.add_argument("--agent_name", choices=["mobileexplorer_no_memory", "mobileexplorer_executable"], default="mobileexplorer_no_memory")
  parser.add_argument("--executable_memory", action="store_true",
                      help="Enable the versioned executable-memory layer on the existing runner.")
  parser.add_argument("--executable_memory_config", type=Path,
                      help="Optional JSON config for executable-memory ablations.")
  parser.add_argument(
      "--executable_memory_path", type=Path,
      help="Optional persistent executable graph shared by tasks in one experiment arm.",
  )
  parser.add_argument("--m3a_summary", action="store_true", help="Enable M3A's second model call for summaries")
  parser.add_argument("--inject_evidence", action="store_true", help="Summarize exploration findings into the next prompt")
  parser.add_argument("--variant", choices=["full", "random", "flat", "naive", "offline"], default="full")
  parser.add_argument("--offline_graph", type=Path)
  parser.add_argument(
      "--semantic_prefix_mode",
      choices=["off", "logging", "prompt", "assist", "coverage_only"],
      default="off",
      help=("Semantic Prefix sidecar. logging persists only; prompt adds bounded "
            "top-K context; assist enables conservative verified routes; "
            "coverage_only is a negative-control logger."),
  )
  parser.add_argument("--semantic_prefix_path", type=Path,
                      help="Optional persistent Semantic Prefix JSON path.")
  parser.add_argument("--semantic_prefix_top_k", type=int, default=3)
  args = parser.parse_args()
  if args.semantic_prefix_mode != "off":
    # Ex5's intended budget is five.  Do not silently run a Prefix arm with a
    # smaller probe budget just because a stale command line was reused.
    args.max_probes = max(5, args.max_probes)
    args.min_probes = max(5, args.min_probes)
    args.two_system = True
  if args.executable_memory and args.agent_name == "mobileexplorer_no_memory":
    args.agent_name = "mobileexplorer_executable"
  if args.executable_memory:
    args.two_system = True
  if args.a11y_method == "grpc":
    os.environ.pop("ANDROID_WORLD_A11Y_METHOD", None)
  else:
    os.environ["ANDROID_WORLD_A11Y_METHOD"] = args.a11y_method
  adb_path = os.environ.get(
      "ANDROID_WORLD_ADB_PATH", "/opt/android/platform-tools/adb"
  )
  serial = os.environ.get("ANDROID_WORLD_SERIAL", f"emulator-{args.console_port}")
  a11y_local_port = int(os.environ.get("ANDROID_WORLD_A11Y_LOCAL_PORT", "8765"))
  if args.a11y_method == "fast_provider":
    subprocess.run(
        [adb_path, "-s", serial, "forward", f"tcp:{a11y_local_port}", "localabstract:androidworld_fast_a11y"],
        check=True, timeout=10,
    )
    os.environ["ANDROID_WORLD_FAST_A11Y_SOCKET_PORT"] = str(a11y_local_port)
  if args.agent_name == "m3a_llamacpp" and not args.m3a_summary:
    os.environ["ANDROID_WORLD_M3A_DISABLE_SUMMARY"] = "1"
  root = args.output.resolve()
  root.mkdir(parents=True, exist_ok=False)
  trace = root / "probe_trace.jsonl"
  filtered = root / "filtered_elements.jsonl"
  windows = root / "inference_windows.jsonl"
  requests_path = root / "request_latency.jsonl"
  steps_path = root / "step_latency.jsonl"
  executable_memory_path = args.executable_memory_path or (root / "executable_memory.json")
  graph_perf_path = root / "graph_construction_perf.jsonl"
  _instrument_graph_updates(graph_perf_path)
  executable_memory_summary_path = root / "executable_memory_summary.json"
  semantic_prefix_path = args.semantic_prefix_path or (root / "semantic_prefix_memory.json")
  semantic_prefix_events_path = root / "semantic_prefix_events.jsonl"
  semantic_prefix_summary_path = root / "semantic_prefix_summary.json"
  semantic_prefix = None
  semantic_prefix_event_cursor = 0
  semantic_prefix_initial_metrics: dict[str, int] = {}
  if args.semantic_prefix_mode != "off":
    semantic_prefix = SemanticPrefixMemory(
        semantic_prefix_path, mode=args.semantic_prefix_mode)
    semantic_prefix_initial_metrics = semantic_prefix.metrics.to_dict()
  executable_memory = None
  if args.executable_memory:
    os.environ["MOBILEEXPLORER_EXECUTABLE_MEMORY"] = "1"
    os.environ["MOBILEEXPLORER_EXECUTABLE_MEMORY_PATH"] = str(executable_memory_path)
    os.environ["MOBILEEXPLORER_EXECUTABLE_MEMORY_LOG"] = str(root / "executable_memory_events.jsonl")
    if args.executable_memory_config:
      os.environ["MOBILEEXPLORER_EXECUTABLE_MEMORY_CONFIG"] = str(args.executable_memory_config)
    memory_config = ExecutableMemoryConfig.from_json(args.executable_memory_config)
    if not memory_config.exploration_enabled:
      args.two_system = False
    executable_memory = ExecutableExplorationMemory(
        dataclasses.replace(memory_config, enabled=True, max_probes=args.max_probes),
        path=executable_memory_path,
        logger=JsonlMemoryLogger(root / "executable_memory_events.jsonl"),
    )
  http = requests.Session()
  http.trust_env = False
  two_system_config = TwoSystemConfig(
      enabled=args.two_system,
      skip_inference_enabled=args.enable_skip_inference,
      reusable_confidence_threshold=args.skip_confidence,
      max_probes=args.max_probes,
      max_depth=args.max_depth,
      max_exploration_time_s=args.max_exploration_time_s,
  )
  graph = ProgressiveBeliefGraph(args.task)
  evidence_index = OnlineEvidenceIndex()
  evidence_event_cursor = 0
  current_source_version = ""
  current_query = args.task
  if args.offline_graph:
    payload = json.loads(args.offline_graph.read_text())
    if payload.get("source_seed") == args.seed:
      raise ValueError("Offline graph source seed must differ from target")
    # Only value-free topology is imported. Dynamic content is never an answer cache.
    for row in payload.get("observations", []):
      evidence_index.add(**row)
    evidence_index.commit()
  skip_gate = SkipInferenceGate(graph, two_system_config)
  resource_monitor = ResourceMonitor()
  budget_controller = ResourceAdaptiveBudgetController(two_system_config)
  round_logger = RoundLogger(root / "two_system_rounds.jsonl")
  graph_path = root / "progressive_belief_graph.json"
  skip_events_path = root / "skip_events.jsonl"
  bootstrap_events_path = root / "bootstrap_events.jsonl"
  last_budget = budget_controller.allocate(resource_monitor.sample(), predicted_utility=1.0)
  trace_cursor = 0
  trial_edges: dict[str, list[str]] = {}
  pending_rounds: dict[int, InferenceRoundRecord] = {}
  pending_prefix_edge_ids: list[str] = []
  # The authoritative model routinely states its own next intent in the
  # summary it just wrote ("The next step is to open the 'my_expenses.txt'
  # file..."). That is free, already-paid-for predictive information about
  # the decision the explorer is trying to anticipate, so it is what the
  # explorer ranks candidates against, instead of the whole task goal which
  # stays constant for the entire episode and cannot discriminate between
  # steps. Measured on 2026-08-29: 20% of steps state it explicitly, and on
  # the rest this falls back to the task goal, i.e. the previous behaviour.
  last_intent: str = ""
  blocked_element_identities: set[str] = set()
  # Progressive recovery memory: "<activity>|<probe_type>" combinations
  # whose rollback has already been observed to fail this episode. The
  # per-element blocklist above cannot catch these, because a screen full
  # of similar controls presents a new element identity every time while
  # being exactly as unrecoverable (2026-08-29: TAP_NAV at depth 2 failed
  # to roll back 29 times in 145 probes, and all 87 failures that run were
  # genuine - the device really was left on a different screen, so this
  # has to be prevented rather than tolerated). Generalizing one observed
  # failure to the screen it happened on is what lets memory make
  # exploration safer instead of only recording that it was unsafe.
  blocked_recovery_contexts: set[str] = set()
  node_candidate_hits: dict[str, dict[str, int]] = {}

  def explored_control_evidence_by_node() -> dict[str, dict[str, dict[str, Any]]]:
    """Roll up probe evidence by source page and stable control identity."""
    evidence: dict[str, dict[str, dict[str, Any]]] = {}
    for edge in graph.edges.values():
      action = edge.action or {}
      identity = str(action.get("element_identity") or "")
      key = str(action.get("stable_control_key") or
                stable_control_key_from_identity(identity))
      if not key:
        continue
      slot = evidence.setdefault(edge.src_node, {}).setdefault(key, {
          "support_count": 0, "probe_count": 0, "execution_count": 0,
          "destinations": [], "recovery_failures": 0,
          "recovery_successes": 0,
      })
      slot["probe_count"] += int(edge.probe_count or 0)
      slot["execution_count"] += int(edge.execution_hit_count or 0)
      slot["support_count"] += int(edge.probe_count or 0) + int(edge.execution_hit_count or 0)
      slot["recovery_failures"] += int(edge.rollback_failure_count or 0)
      slot["recovery_successes"] += int(edge.rollback_success_count or 0)
      slot["destinations"] = sorted(set(slot["destinations"]) | set(
          edge.observed_destinations or ((edge.dst_node,) if edge.dst_node else ())))
    return evidence

  def explored_element_identities_by_node() -> dict[str, list[str]]:
    """Derived on demand from the graph - the single persistent per-task
    record of what has already been explored - rather than a parallel
    tracking structure that could drift out of sync with it."""
    by_node: dict[str, set[str]] = {}
    for edge in graph.edges.values():
      identity = edge.action.get("element_identity")
      if identity:
        by_node.setdefault(edge.src_node, set()).add(str(identity))
    return {node: sorted(identities) for node, identities in by_node.items()}

  def ingest_probe_trace(trial_id: str) -> list[dict[str, Any]]:
    nonlocal trace_cursor, semantic_prefix_event_cursor
    graph_ingest_started = time.perf_counter()
    if not trace.exists():
      return []
    rows = trace.read_text(encoding="utf-8").splitlines()
    new_rows = [json.loads(line) for line in rows[trace_cursor:] if line.strip()]
    trace_cursor = len(rows)
    ingested = []
    edge_ids = []
    for row in new_rows:
      # Prefix is a sidecar and observes the same probe rows as Ex5.  A
      # malformed sidecar record must never abort the authoritative task.
      if semantic_prefix is not None:
        try:
          prefix_row = dict(row)
          prefix_element = dict(row.get("element") or {})
          dynamic_control = any(
              is_task_specific_value(row.get("task", args.task), value)
              or is_dynamic_ui_text(value)
              for value in (prefix_element.get("text"), prefix_element.get("content_desc"))
          )
          if dynamic_control:
            prefix_element["text"] = ""
            prefix_element["content_desc"] = ""
            prefix_row["dynamic_content"] = True
            prefix_row["element"] = prefix_element
            prefix_graph = dict(row.get("graph") or {})
            prefix_action = dict(prefix_graph.get("action") or {})
            identity = str(prefix_action.get("element_identity") or "")
            if identity:
              parts = identity.split("|")
              if len(parts) >= 4:
                parts[1] = ""
                parts[2] = ""
                prefix_action["element_identity"] = "|".join(parts)
              else:
                prefix_action["element_identity"] = "[dynamic control label redacted]"
            prefix_graph["action"] = prefix_action
            prefix_row["graph"] = prefix_graph
          semantic_prefix.record_probe_row(prefix_row)
        except Exception as exc:  # pragma: no cover - defensive live guard
          _append(semantic_prefix_events_path, {
              "event": "prefix_record_error", "error": f"{type(exc).__name__}: {exc}",
          })
      if row.get("notes") in {"RESTORE_FAILED", "NESTED_RESTORE_FAILED"}:
        # This element's rollback could not be verified by any recovery
        # level (see recovery.py's ladder). Learn from the observed outcome
        # rather than re-deriving safety from a priori rules: never probe
        # this exact element again for the rest of the task.
        # Depth-independent on purpose: an element that cannot be rolled
        # back is equally unsafe whichever hop reached it, and deeper hops
        # report the failure under a different note. Restricting this to
        # depth-1/"RESTORE_FAILED" let the same unrecoverable recipe card be
        # re-probed and fail three more times at depth 2 within one task
        # (2026-08-29, RecipeDeleteMultipleRecipes: 4 of its 8 failures were
        # one element the blocklist should already have excluded).
        blocked_element_identities.add(element_identity_from_dict(row.get("element", {})))
        source_activity = str(((row.get("graph") or {}).get("src") or {}).get("activity") or "")
        if source_activity:
          blocked_recovery_contexts.add(f"{source_activity}|{row.get('probe_type')}")
      reached_activity = str((row.get("discovered") or {}).get("reached_activity") or "")
      reached_package = reached_activity.split("/", 1)[0]
      probed_app_package = str(row.get("app_package") or "")
      if reached_package and probed_app_package and reached_package != probed_app_package:
        # This probe left the app's own package entirely (e.g. a "Share"
        # action landing on Android's system ChooserActivity - 2026-08-29,
        # RecipeDeleteMultipleRecipes). Package identity is a structural
        # fact, not a per-app rule. Even when the recovery ladder happens to
        # get back this time (Back usually dismisses a system chooser),
        # leaving the task's own app is inherently not worth the risk to
        # re-attempt, so blocklist it regardless of whether this specific
        # attempt's rollback succeeded.
        blocked_element_identities.add(element_identity_from_dict(row.get("element", {})))
      if row.get("trial_id") != trial_id or "graph" not in row:
        continue
      item = row["graph"]
      src, dst = item["src"], item["dst"]
      src_id = graph.make_node_id(src["activity"], src.get("layout_signature", src["structural_signature"]))
      dst_id = graph.make_node_id(dst["activity"], dst.get("layout_signature", dst["structural_signature"]))
      for node_id, value, status in ((src_id, src, NodeStatus.COMMITTED), (dst_id, dst, NodeStatus.SPECULATIVE)):
        graph.upsert_node(GraphNode(
            node_id=node_id, activity=value["activity"],
            package=value["activity"].split("/", 1)[0],
            visual_signature=value["visual_signature"],
            structural_signature=value["structural_signature"],
            layout_signature=value.get("layout_signature", value["structural_signature"]),
            status=status,
        ))
      element = row.get("element", {})
      task_text = row.get("task", args.task)
      safe_labels = tuple(
          str(label) for label in (row.get("discovered") or {}).get("new_texts", [])[:8]
          if not is_task_specific_value(task_text, label) and not is_dynamic_ui_text(label)
      )
      dynamic_control = any(
          is_task_specific_value(task_text, value) or is_dynamic_ui_text(value)
          for value in (element.get("text"), element.get("content_desc"),
                        element.get("content-description"))
      )
      role = str(element.get("class", "")).rsplit(".", 1)[-1].casefold()
      depth = int(row.get("depth", 1))
      risk = (
          "LOW" if depth >= 2 and bool(row.get("recovery_ok"))
          else ("LOW" if role in {"imagebutton", "tabwidget"} else "UNKNOWN")
      )
      probe_action = dict(item["action"])
      probe_action["probe_type"] = row.get("probe_type", "")
      probe_action["role"] = role
      identity = str(probe_action.get("element_identity") or "")
      if identity:
        parts = identity.split("|")
        if dynamic_control or any(
            is_task_specific_value(task_text, part) or is_dynamic_ui_text(part)
            for part in parts[1:3]
        ):
          identity = "|".join([parts[0], "", "", *parts[3:]])
          probe_action["element_identity"] = identity
        probe_action["control_key"] = control_key_from_identity(identity)
        probe_action["stable_control_key"] = stable_control_key_from_identity(identity)
      edge = graph.add_speculative_transition(
          src_id, probe_action, dst_id,
          path_probability=max(0.05, float(element.get("score", 0.0))),
          confidence=0.20, expected_information_gain=min(1.0, row.get("discovered", {}).get("new_element_count", 0) / 10.0),
          risk_level=risk,
          exploration_cost=float(row.get("timings_ms", {}).get("total", 0.0)) / 1000.0,
          rollback_success=bool(row.get("recovery_ok")),
          discovered_labels=safe_labels,
          dynamic_content=dynamic_control,
      )
      # The transition itself is not enough: the task graph must also retain
      # how many times it was measured and whether returning to the anchor
      # actually worked. Without this update every edge looked permanently
      # low-support to the scorer, even though probe_trace.jsonl had outcomes.
      graph.record_probe(
          edge.edge_id, rollback_ok=bool(row.get("recovery_ok")),
          cost_s=float((row.get("timings_ms") or {}).get("total", 0.0)) / 1000.0,
          realized_ig=min(1.0, float((row.get("discovered") or {}).get(
              "new_element_count", 0) or 0) / 10.0),
          generation=request_no,
      )
      if not row.get("recovery_ok"):
        edge.status = EdgeStatus.INVALID
        edge.confidence = 0.0
      evidence_index.add(
          source=src_id, source_version=str(src.get("structural_signature", "")),
          destination=dst_id, action=item["action"],
          labels=[label for label in (row.get("discovered") or {}).get("new_texts", [])[:12]
                  if not is_task_specific_value(task_text, label)
                  and not is_dynamic_ui_text(label)],
          recovered=bool(row.get("recovery_ok")),
          cost_s=float(row.get("timings_ms", {}).get("total", 0)) / 1000,
      )
      edge_ids.append(edge.edge_id)
      ingested.append({
          "edge_id": edge.edge_id, "depth": depth,
          "probe_count": edge.probe_count,
          "predicted_IG": edge.expected_information_gain,
          "realized_IG": edge.realized_information_gain,
          "risk_level": risk,
      })
      if executable_memory is not None:
        executable_memory.ingest_probe_row(row)
    trial_edges[trial_id] = edge_ids
    # The Ex5 graph remains the in-process rollback/safety controller. When
    # EAM is enabled, its canonical graph is the only persisted reusable graph.
    if executable_memory is None:
      graph_path.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if executable_memory is not None:
      executable_memory.metrics.node_count = len(executable_memory.states)
      executable_memory.metrics.edge_count = len(executable_memory.edges)
      executable_memory.save()
    if semantic_prefix is not None:
      semantic_prefix.save()
      for event in semantic_prefix.events[semantic_prefix_event_cursor:]:
        _append(semantic_prefix_events_path, event)
      semantic_prefix_event_cursor = len(semantic_prefix.events)
      semantic_prefix_summary_path.write_text(
          json.dumps(semantic_prefix.summary(), ensure_ascii=False, indent=2),
          encoding="utf-8")
    _append(graph_perf_path, {
        "operation": "probe_trace_ingest_wall",
        "elapsed_s": max(0.0, time.perf_counter() - graph_ingest_started),
    })
    return ingested

  def snapshot() -> dict[str, float]:
    last_error: Exception | None = None
    for attempt in range(3):
      try:
        response = http.get(args.metrics_url, timeout=10)
        response.raise_for_status()
        text = response.text
        break
      except requests.RequestException as exc:
        last_error = exc
        http.close()
        time.sleep(0.25 * (attempt + 1))
    else:
      raise RuntimeError(f"metrics snapshot failed after 3 attempts: {last_error}")
    values = {}
    for key, metric in METRICS.items():
      match = re.search(r"^" + re.escape(metric) + r"\{[^\n]*\}\s+([0-9.eE+-]+)$", text, re.MULTILINE)
      values[key] = float(match.group(1)) if match else 0.0
    return values

  profile_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("parallel_profile", default=None)
  request_no = 0
  original_predict = infer.LlamaCppWrapper.predict_mm
  prepared = None
  device_dirty = False
  dirty_events_path = root / "dirty_events.jsonl"

  def explorer_config(number: int, step: int, task_text: str) -> dict[str, Any]:
    information_need = parse_information_need(None, last_intent or task_text)
    budget = last_budget
    allow_exploration = bool(args.two_system and budget.allow_exploration)
    effective_max_probes = (
        min(args.max_probes, budget.max_probes) if allow_exploration else 0)
    effective_max_depth = (
        min(args.max_depth, max(1, budget.max_depth)) if allow_exploration else 1)
    effective_exploration_time_s = (
        min(args.max_exploration_time_s, budget.max_exploration_time_s)
        if allow_exploration else 0.0)
    config = {
        "trial_id": f"{args.task}-step{step}-request{number}",
        "trace_path": str(trace), "filtered_path": str(filtered),
        "save_probe_diagnostics": os.environ.get(
            "ANDROID_WORLD_SAVE_PROBE_DIAGNOSTICS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"},
        "probe_diagnostics_root": str(root / "probe_diagnostics"),
        "task": task_text, "step_idx": step,
        "ranker": "RandomRanker" if args.variant == "random" else args.ranker,
        "seed": args.seed + number,
        # Coverage/frontier ranking is retained only as an explicit negative
        # control.  The complete system must be driven by the structured
        # information need plus measured relevance/recovery/cost; silently
        # enabling posterior_frontier here made transition entropy the main
        # objective and bypassed InformationNeedRanker entirely.
        "posterior_frontier": args.semantic_prefix_mode == "coverage_only",
        "predictive_scorer": (
            args.variant not in {"random", "flat"}
            and args.ranker not in {"LightweightLinearRanker", "CoverageRanker"}
        ),
        "graph_snapshot": graph.to_dict() if args.variant != "flat" else {},
        "serial": serial,
        "console_port": args.console_port, "a11y_local_port": a11y_local_port,
        "restore_timeout_s": 30.0,
        "max_probes": 0 if args.variant == "offline" else effective_max_probes,
        "min_probes": min(args.min_probes, effective_max_probes),
        "force_min_probes": bool(args.force_min_probes and effective_max_probes),
        "post_inference_grace_s": args.post_inference_grace_s,
        "max_depth": effective_max_depth,
        # Depth-2 caused the majority of real restore failures in the first
        # smoke (keyboard keys, dialogs, and edit modes). Descend only after
        # this exact root probe type has already demonstrated an exact inverse
        # on this screen in an earlier round.
        "depth2_needs_known_inverse": True,
        "max_exploration_time_s": effective_exploration_time_s,
        "resource_budget_reason": budget.reason,
        "information_need": information_need.to_dict(),
        "lightweight_selector_weights": str(args.selector_weights),
        "blocked_element_identities": sorted(blocked_element_identities),
        "blocked_recovery_contexts": sorted(blocked_recovery_contexts),
        "node_candidate_hits": {node: dict(hits) for node, hits in node_candidate_hits.items()},
        "explored_element_identities": explored_element_identities_by_node(),
        "explored_control_evidence": explored_control_evidence_by_node(),
        "goal_relevance_threshold": two_system_config.goal_relevance_threshold,
        "max_stack_depth_increase": two_system_config.max_stack_depth_increase,
    }
    # Filled immediately before INFERENCE_START, after MobileExplorer has
    # committed this page's pending authoritative transition. The explorer
    # process is already waiting, so its page anchor and memory generation
    # remain from the same reasoning step.
    config["executable_memory_guidance"] = {
        "enabled": False, "state_matched": False, "controls": {},
    }
    return config

  def prepare_explorer_for_step(number: int, step: int, task_text: str):
    config = explorer_config(number, step, task_text)
    if config["max_probes"] <= 0:
      return None
    return prepare_explorer(config)

  def executable_page_from_capture(current_state):
    """Mirror live_probe's compact graph-element view for cross-store matching."""
    elements = [element for element in (current_state.elements or ())
                if (getattr(element, "resource_id", "")
                    or getattr(element, "clickable", False)
                    or getattr(element, "scrollable", False))][:256]
    bounds = [getattr(element, "bounds", (0, 0, 0, 0)) for element in elements]
    screen_size = (
        max((float(box[2]) for box in bounds if len(box) == 4), default=1.0),
        max((float(box[3]) for box in bounds if len(box) == 4), default=1.0),
    )
    return PageObservation.from_ui(
        elements,
        package=current_state.activity.component.split("/", 1)[0],
        activity=current_state.activity.component,
        visual_signature=current_state.phash,
        screen_size=screen_size,
    )

  def parallel_predict(self, text_prompt, images, messages=None):
    nonlocal request_no, prepared, last_budget, device_dirty
    request_no += 1
    profile = profile_var.get() or {}
    if device_dirty:
      # A prior round's rollback left the device in an unverified state
      # (StateVerifier could not confirm recovery). Do not assume rollback
      # always succeeds and do not abort the episode over it: fall back to
      # inference-only for the remainder of this task, same as severe
      # resource pressure zeroing the exploration budget. The next step's
      # own perception naturally re-observes the real screen.
      _append(dirty_events_path, {
          "trial_id": f"{args.task}-step{profile.get('step')}-request{request_no}",
          "reason": "device_dirty_fallback_skip_exploration",
      })
      return original_predict(self, text_prompt, images, messages)
    if args.variant == "offline" or prepared is None:
      return original_predict(self, text_prompt, images, messages)
    if isinstance(prepared.config, dict):
      prepared.config["committed_actions"] = list(profile.get("committed_actions") or [])
      prepared.config["graph_snapshot"] = graph.to_dict() if args.variant != "flat" else {}
    trial_id = str(prepared.config["trial_id"])
    used_budget = last_budget
    before = snapshot()
    resources_before = resource_monitor.sample()
    process_memory_before = resources_before.process_memory_mb
    client_started = time.perf_counter()
    if executable_memory is not None:
      # The authoritative MobileExplorer instance and this shadow-ingestion
      # instance share one file. Pull in authoritative edges from the prior
      # step before appending this round's probes so a stale shadow snapshot
      # cannot overwrite them.
      executable_memory.refresh()
      if (isinstance(prepared.config, dict)
          and executable_memory.config.exploration_guidance_enabled):
        try:
          page = profile.get("executable_memory_page")
          if isinstance(page, PageObservation):
            prepared.config["executable_memory_guidance"] = (
                executable_memory.exploration_guidance(
                    page, str(profile.get("goal") or args.task)))
        except Exception as exc:  # graph guidance is advisory only
          prepared.config["executable_memory_guidance"] = {
              "enabled": False, "state_matched": False, "controls": {},
              "error": f"{type(exc).__name__}: {exc}",
          }
          _append(root / "executable_memory_guidance_errors.jsonl", {
              "task": args.task, "step": profile.get("step"),
              "error": f"{type(exc).__name__}: {exc}",
          })
      executable_memory.begin_round()
    result, window = run_prepared_explorer(
        prepared, lambda: original_predict(self, text_prompt, images, messages)
    )
    client_total_s = time.perf_counter() - client_started
    rollback_ok = window["restore_status"] == EventKind.RESTORED.value
    if rollback_ok:
      # The next authoritative action changes the screen. Reusing a worker
      # prewarmed on this page would give the next reasoning/exploration pair
      # different anchors and can make rollback restore a stale page.
      prepared = None
    else:
      failed_window = dict(window)
      # Do not assume rollback always succeeds. Abandon the prewarmed next
      # explorer, stop probing for the rest of this episode, and let the
      # authoritative inference result execute against the real (unverified)
      # screen rather than aborting the whole task.
      device_dirty = True
      stop_prepared_explorer(prepared)
      prepared = None
      _append(dirty_events_path, {"trial_id": trial_id, "reason": "discard_stale_action_and_reobserve"})
      # Re-observe with a fresh authoritative request. The exploratory result
      # is discarded; recovery failure is counted as overhead rather than
      # converted into a task exception.
      result = original_predict(self, text_prompt, images, messages)
      # Preserve the explorer's real terminal diagnosis. Replacing it with
      # an empty dict made later rounds look like uninstrumented failures and
      # hid the reason the safe fallback was taken.
      window = {**failed_window, "fallback_inference_only": True}
    after = snapshot()
    resources_after = resource_monitor.sample()
    delta = {key: after[key] - before[key] for key in before}
    server_e2e = delta["ttft_s"] + delta["decode_s"]
    row = {
        "request_number": request_no, "trial_id": trial_id,
        "task": args.task, "goal": profile.get("goal", ""), "step": profile.get("step"),
        "client_total_s": client_total_s,
        "queue_s": delta["queue_s"],
        "prefill_s": delta["prefill_s"], "decode_s": delta["decode_s"],
        "inference_s": delta["inference_s"], "ttft_s": delta["ttft_s"],
        "server_e2e_s": server_e2e,
        "prompt_tokens": int(round(delta["prompt_tokens"])),
        "generation_tokens": int(round(delta["generation_tokens"])),
        # A rollback failure can intentionally fall back to a fresh
        # authoritative inference, for which there is no second explorer
        # window.  Treat absent post-processing fields as instrumentation
        # defaults, never as a task exception.
        "critical_path_extension_ms": window.get("critical_path_extension_ms", 0.0),
        "restore_status": window.get("restore_status", "NOT_CAPTURED"),
    }
    _append(requests_path, row)
    _append(windows, {**window, "task": args.task, "goal": profile.get("goal", ""), "step": profile.get("step")})
    explored_edges = ingest_probe_trace(trial_id)
    if executable_memory is not None:
      executable_memory.metrics.probes_completed += len(explored_edges)
      executable_memory.metrics.extra_time_s += max(
          0.0, float(window.get("critical_path_extension_ms", 0.0)) / 1000.0)
      if len(explored_edges) >= min(5, args.max_probes):
        executable_memory.metrics.probe_rounds_complete += 1
      else:
        executable_memory.metrics.stop_reasons[str(window.get("stop_reason", "unknown"))] += 1
      executable_memory.save()
    if not rollback_ok:
      for edge_id in trial_edges.get(trial_id, ()):
        edge = graph.edges.get(edge_id)
        if edge is not None:
          graph.invalidate_subtree(edge.src_node)
      _append(dirty_events_path, {
          "trial_id": trial_id, "window": failed_window,
          "reason": "rollback_failed_falling_back_to_inference_only",
      })
    last_budget = budget_controller.allocate(
        resources_after, predicted_utility=max((edge["predicted_IG"] for edge in explored_edges), default=0.1),
        recent_inference_s=client_total_s,
        recent_exploration_s=float(window.get("exploration_elapsed_ms", 0.0)) / 1000.0,
    )
    pending_rounds[request_no] = InferenceRoundRecord(
        round_id=request_no, task_id=args.task,
        current_state_id=next(
            (graph.edges[item["edge_id"]].src_node for item in explored_edges if item.get("depth") == 1),
            "",
        ),
        inference_latency_s=client_total_s,
        inference_memory_mb=resources_after.process_memory_mb,
        exploration_budget={
            "max_probes": used_budget.max_probes, "max_depth": used_budget.max_depth,
            "max_extra_memory_mb": used_budget.max_extra_memory_mb,
            "max_exploration_time_s": used_budget.max_exploration_time_s,
            "allow_exploration": used_budget.allow_exploration, "reason": used_budget.reason,
        },
        number_of_exploration_probes=int(window.get("probes_completed", 0)),
        max_exploration_depth=int(window.get("max_depth_reached", 0)),
        explored_edges=explored_edges,
        rollback_latency_ms=float(window.get("critical_path_extension_ms", 0.0)),
        rollback_success=rollback_ok,
        preemption_count=int(bool(window.get("preempted"))),
        unfinished_exploration=bool(window.get("preempted")),
        concurrent_memory_overhead_mb=max(0.0, resources_after.process_memory_mb - process_memory_before),
        interference_estimate=budget_controller.interference_ema,
        cpu_utilization=resources_after.cpu_used_ratio,
        gpu_utilization=resources_after.gpu_utilization,
        power_w=resources_after.power_w, temperature_c=resources_after.temperature_c,
        memory_confidence=graph.get_local_confidence(graph.edges[trial_edges[trial_id][0]].src_node) if trial_edges.get(trial_id) else 0.0,
        explorer_reliability=budget_controller.explorer_reliability,
    )
    return result

  from android_world.parallel_exploration.protocol import EventKind
  infer.LlamaCppWrapper.predict_mm = parallel_predict
  agent_class = mobileexplorer.MobileExplorer
  original_step = agent_class.step
  skip_capture = None
  fast_a11y_ready = False

  def ensure_fast_a11y_companion():
    """Enable the low-latency sidecar without replacing AndroidWorld gRPC."""
    nonlocal fast_a11y_ready
    if fast_a11y_ready or args.a11y_method != "grpc":
      return
    apk_path = REPO_ROOT / "tools" / "fast_a11y_dumper" / "build" / "fast-a11y.apk"
    build_script = REPO_ROOT / "tools" / "fast_a11y_dumper" / "build_apk.sh"

    def adb(*parts: str, timeout: float = 20.0) -> str:
      completed = subprocess.run(
          [adb_path, "-s", serial, *parts], capture_output=True, text=True,
          timeout=timeout, check=False,
      )
      output = (completed.stdout or "") + (completed.stderr or "")
      if completed.returncode:
        raise RuntimeError(
            f"adb {' '.join(parts)} failed ({completed.returncode}): {output[-1000:]}"
        )
      return output.strip()

    if not apk_path.exists():
      built = subprocess.run(
          [str(build_script)], cwd=REPO_ROOT, capture_output=True, text=True,
          timeout=120.0, check=False,
      )
      if built.returncode or not apk_path.exists():
        raise RuntimeError(
            "FastA11y sidecar APK is unavailable: "
            + (built.stdout + built.stderr)[-1500:]
        )
    adb("install", "-r", str(apk_path), timeout=60.0)
    enabled = adb("shell", "settings", "get", "secure", "enabled_accessibility_services")
    services = [item for item in enabled.split(":") if item and item != "null"]
    fast_service = (
        "com.androidworld.fasta11y/"
        "com.androidworld.fasta11y.FastA11yService"
    )
    if fast_service not in services:
      services.append(fast_service)
    adb("shell", "settings", "put", "secure", "accessibility_enabled", "1")
    adb("shell", "settings", "put", "secure", "enabled_accessibility_services", ":".join(services))
    time.sleep(0.5)
    smoke = adb(
        "shell", "content", "read", "--uri",
        "content://com.androidworld.fasta11y.provider/flat?compact=1",
    )
    if '"ok":true' not in smoke:
      raise RuntimeError(f"FastA11y sidecar did not become ready: {smoke[:500]}")
    adb(
        "forward", f"tcp:{a11y_local_port}",
        "localabstract:androidworld_fast_a11y",
    )
    with socket.create_connection(("127.0.0.1", a11y_local_port), timeout=5.0) as connection:
      connection.settimeout(5.0)
      connection.sendall(b"flat 0 10000\n")

      def read_exact(size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
          chunk = connection.recv(size - len(chunks))
          if not chunk:
            raise ConnectionError("FastA11y socket closed during startup probe")
          chunks.extend(chunk)
        return bytes(chunks)

      payload_size = struct.unpack(">I", read_exact(4))[0]
      if payload_size <= 0 or payload_size > 32 * 1024 * 1024:
        raise RuntimeError(f"FastA11y socket returned invalid payload size: {payload_size}")
      socket_smoke = json.loads(read_exact(payload_size).decode("utf-8"))
    if not socket_smoke.get("ok") or socket_smoke.get("nodeCount", 0) <= 0:
      raise RuntimeError(f"FastA11y socket returned an empty/unready tree: {socket_smoke}")
    fast_a11y_ready = True

  def ensure_skip_capture():
    nonlocal skip_capture
    if skip_capture is None:
      # AndroidWorld's gRPC wrapper owns the primary accessibility service.
      # Install/enable FastA11y alongside it only after env setup; enabling it
      # earlier lets the wrapper replace the service list and leaves the
      # skip-app observer with a dead socket.
      ensure_fast_a11y_companion()
      skip_capture = create_optimized_state_capture(
          serial=serial, console_port=args.console_port,
          adb_path=adb_path, a11y_local_port=a11y_local_port,
          use_fast_a11y_socket=True,
      )
    return skip_capture

  # Exploration evidence -> next prompt. Even when an explored edge is not
  # trustworthy enough to skip inference with, it still holds real
  # predictive information ("this control leads to a screen showing X")
  # that the authoritative model would otherwise have to spend a step
  # discovering. Injected by wrapping the agent's prompt builder rather
  # than editing gelab_agent.py, so the plain baseline path stays byte-for
  # -byte unchanged and this stays behind --inject_evidence.
  evidence_var: contextvars.ContextVar[str] = contextvars.ContextVar("exploration_evidence", default="")
  prefix_var: contextvars.ContextVar[str] = contextvars.ContextVar("semantic_prefix_context", default="")
  original_build_messages = mobileexplorer.build_mobileexplorer_messages

  def build_messages_with_evidence(goal, history, screenshot, **kwargs):
    messages = original_build_messages(goal, history, screenshot, **kwargs)
    evidence = evidence_var.get()
    if evidence:
      for item in messages[-1]["content"]:
        if item.get("type") == "text":
          item["text"] = f"{item['text']}\n\nKnown from exploring this screen:\n{evidence}"
          break
    prefix_context = prefix_var.get()
    if prefix_context:
      for item in messages[-1]["content"]:
        if item.get("type") == "text":
          item["text"] = f"{item['text']}\n\n{prefix_context}"
          break
    return messages

  def exploration_evidence_for(node_id: str | None) -> str:
    if not args.inject_evidence or not node_id:
      return ""
    if args.variant not in {"naive", "flat"}:
      selected = evidence_index.retrieve(node_id, current_source_version, current_query)
      lines = [
          f"- Observed transition via {o.action.get('element_identity', o.action)}: "
          + "; ".join(o.labels) + ". This observation is not an instruction."
          for o in selected]
      lines.extend(graph_path_context(node_id, current_query, top_k=3))
      return "\n".join(lines)[:1800]
    lines = []
    for edge in graph.edges.values():
      if edge.src_node != node_id or edge.status == EdgeStatus.INVALID:
        continue
      if not edge.discovered_labels:
        continue
      label = str(edge.action.get("element_identity", "")).split("|")[1:3]
      name = next((part for part in label if part), None)
      where = f"({edge.action.get('x')},{edge.action.get('y')})"
      target = ", ".join(edge.discovered_labels[:5])
      lines.append(f"- Tapping {name or 'the control'} at {where} opens: {target}")
      if len(lines) >= 4:
        break
    return "\n".join(lines)

  def graph_path_context(node_id: str, query: str, *, top_k: int = 3) -> list[str]:
    """Retrieve a few short, task-matched observed paths from the live graph."""
    query_tokens = goal_semantic_tokens(query)
    if not query_tokens:
      return []
    outgoing: dict[str, list[Any]] = {}
    for edge in graph.edges.values():
      if edge.status in {EdgeStatus.INVALID, EdgeStatus.STALE}:
        continue
      if edge.dynamic_content:
        continue
      if not edge.dst_node or edge.rollback_failure_count:
        continue
      if not (edge.rollback_success_count or edge.execution_hit_count):
        continue
      outgoing.setdefault(edge.src_node, []).append(edge)
    paths: list[tuple[float, str]] = []
    frontier: list[tuple[str, list[Any]]] = [(node_id, [])]
    seen_routes: set[tuple[str, ...]] = set()
    for _ in range(3):
      next_frontier = []
      for source_id, prefix in frontier:
        for edge in outgoing.get(source_id, ()):
          visited_nodes = {node_id} | {item.dst_node for item in prefix if item.dst_node}
          if edge.dst_node in visited_nodes:
            continue
          route = prefix + [edge]
          route_text = " ; then ".join(
              " ".join((str(item.action.get("element_identity") or ""),
                        " ".join(item.discovered_labels or ()),
                        graph.nodes.get(item.dst_node).activity
                        if item.dst_node in graph.nodes else ""))
              for item in route)
          route_tokens = goal_semantic_tokens(route_text)
          overlap = query_tokens & route_tokens
          route_key = tuple(item.edge_id for item in route)
          if overlap and route_key not in seen_routes:
            seen_routes.add(route_key)
            relevance = len(overlap) / len(query_tokens)
            if relevance >= 0.30:
              support = sum(item.probe_count + item.execution_hit_count for item in route)
              stability = sum(
                  (item.rollback_success_count + item.execution_hit_count) /
                  max(1, item.probe_count + item.execution_hit_count)
                  for item in route) / len(route)
              cost = sum(item.exploration_cost for item in route) / len(route)
              score = (0.65 * relevance + 0.20 * min(1.0, support / 3.0)
                       + 0.15 * stability) / (1.0 + max(0.0, cost)) / len(route)
              labels = []
              for item in route:
                identity = str(item.action.get("element_identity") or "")
                parts = identity.split("|")
                selector = next((part for part in parts[1:4] if part), "control")
                dest = graph.nodes.get(item.dst_node)
                landing = (dest.activity.rsplit("/", 1)[-1] if dest else "known page")
                delta = ", ".join(item.discovered_labels[:3]) or landing
                labels.append(
                    f"{selector} -> {delta} (support={item.probe_count + item.execution_hit_count}, "
                    f"reversible={item.rollback_success_count > 0 or item.execution_hit_count > 0})")
              paths.append((score, " ; then ".join(labels)))
          if edge.dst_node != source_id:
            next_frontier.append((edge.dst_node, route))
      frontier = next_frontier
    paths.sort(key=lambda item: (-item[0], item[1]))
    return [f"- Observed graph path (score={score:.2f}): {text}. Verify on the live page."
            for score, text in paths[:top_k]]

  def semantic_prefix_context_for(signature, goal: str) -> str:
    if semantic_prefix is None or args.semantic_prefix_mode not in {"prompt", "assist"}:
      return ""
    try:
      node_id = state_id_from_signature(signature)
      return semantic_prefix.prompt_context(
          node_id, goal, top_k=max(1, args.semantic_prefix_top_k))
    except Exception as exc:  # advisory context must never break parsing
      _append(semantic_prefix_events_path, {
          "event": "prefix_prompt_error", "error": f"{type(exc).__name__}: {exc}",
      })
      return ""

  def resolve_graph_node(signature) -> str | None:
    # Node identity is now the tolerant activity+layout signature (see
    # ProgressiveBeliefGraph.make_node_id), so a direct lookup already
    # recognizes the same screen showing different data. The previous
    # struct/pHash fuzzy fallback existed only to paper over the old strict
    # identity and is no longer needed.
    node_id = graph.make_node_id(signature.activity.component, signature.layout_sig)
    node = graph.nodes.get(node_id)
    if node is None or node.status in {NodeStatus.INVALID, NodeStatus.STALE}:
      return None
    return node_id

  def replayable_here(edge, signature) -> bool:  # pylint: disable=unused-variable
    """UNUSED. Kept with its measurement; see the note at the end.

    May this recorded action be replayed on the screen in front of us?

    Only the screen's fixed chrome qualifies. Replaying a tap on a list row
    means "tap whatever now sits in that slot", which on a repetitive task is
    a different item every time - measured as 2 of 9 landing as predicted
    (2026-08-30). The anchor check on top confirms the control really is
    still there, since chrome can also move when a screen reflows.

    Measured and rejected on 2026-08-30. Gating reuse on the anchor alone
    gave 6/15 with 0 of 5 skips landing as predicted; additionally requiring
    chrome gave 5/15 with 0 of 2. Both scored below the unfiltered rule
    (7/15, 2 of 9), because they suppressed skips without making the
    surviving ones more accurate - so the mispredictions are not explained by
    "the control moved" or "it was a list row", and this filter costs
    coverage for nothing. Left here so the next idea in this direction starts
    from the evidence rather than re-running it.
    """
    role = edge.action.get("element_role")
    anchor = edge.action.get("element_anchor")
    if not anchor:
      return True  # Nothing positional to replay (e.g. open_app).
    if role != "chrome":
      return False
    x, y = edge.action.get("x"), edge.action.get("y")
    if x is None or y is None:
      return True
    for element in signature.elements:
      left, top, right, bottom = element.bounds
      if left <= float(x) <= right and top <= float(y) <= bottom:
        return anchor == "|".join(
            (element.class_name or "", element.resource_id or ""))
    return False

  def expected_successor_matches(expected_node_id: str | None, signature) -> bool:
    if not expected_node_id or expected_node_id not in graph.nodes:
      return False
    expected = graph.nodes[expected_node_id]
    if expected.activity != signature.activity.component:
      return False
    # "Did this land on the same screen the graph predicted?" is a
    # screen-identity question, so it uses the same tolerant layout
    # signature as node identity. Rollback verification keeps using the
    # strict struct/pHash comparison (see live_probe._strictly_restored) -
    # there, "exactly the state I left" really is the question.
    return expected.layout_signature == signature.layout_sig

  def try_semantic_prefix_route(self, goal: str):
    """Attempt only a fully gated Prefix route; None means normal Ex5.

    This function is intentionally a sidecar ahead of ``try_skip_step``.  A
    rejection, relocation ambiguity, landing mismatch, or unexpected error
    records a miss and returns None, so the existing Ex5 reasoning path gets
    the next chance with no changed action payload.
    """
    if semantic_prefix is None or args.semantic_prefix_mode != "assist":
      return None
    if args.agent_name not in {"mobileexplorer_no_memory", "mobileexplorer_executable"}:
      return None
    try:
      before = ensure_skip_capture().capture()
      source_id = state_id_from_signature(before)
      routes = semantic_prefix.candidate_routes(
          source_id, goal, top_k=max(1, args.semantic_prefix_top_k))
      # Three repeated shadow landings are useful prompt evidence, but are not
      # sufficient authority to mutate the live task. At least one edge-wise
      # route must also have been confirmed by the real reasoner's action.
      routes = [route for route in routes
                if all(edge.route_hit_count > 0 for edge in route.edges)]
      if not routes:
        return None
      route = routes[0]
      executed: list[dict[str, Any]] = []
      started = time.perf_counter()
      for edge in route.edges:
        action_dict = semantic_prefix.relocate(edge, before.elements if not executed else after.elements)
        if action_dict is None:
          semantic_prefix.record_route_result(route, hit=False, reason="selector_unrelocatable")
          _append(semantic_prefix_events_path, {
              "event": "prefix_route_miss", "route_id": route.route_id,
              "reason": "selector_unrelocatable",
          })
          return None
        action_type = str(action_dict.get("action_type") or "").lower()
        # V0 route assist only executes reversible clicks.  Back/scroll can be
        # learned and shown in prompt context but require a more explicit
        # inverse policy than this prototype has.
        if action_type not in {json_action.CLICK, json_action.NAVIGATE_BACK}:
          semantic_prefix.record_route_result(route, hit=False, reason="unsupported_safe_action")
          return None
        try:
          action = json_action.JSONAction(**{
              key: value for key, value in action_dict.items()
              if key in {"action_type", "x", "y", "text", "direction", "app_name", "index"}
          })
        except (TypeError, ValueError) as exc:
          semantic_prefix.record_route_result(route, hit=False, reason="invalid_relocated_action")
          semantic_prefix.record_parse_error(exc, action_dict)
          return None
        self._execute_action(action, {})
        time.sleep(0.20)
        after = ensure_skip_capture().capture()
        if state_id_from_signature(after) != edge.target_state_id:
          semantic_prefix.record_route_result(route, hit=False, reason="wrong_landing")
          _append(semantic_prefix_events_path, {
              "event": "prefix_route_miss", "route_id": route.route_id,
              "edge_id": edge.edge_id, "reason": "wrong_landing",
          })
          # Best-effort safe restoration.  Failure is observable and does not
          # become a task failure; Ex5 will re-observe before its action.
          for _ in range(max(1, len(executed) + 1)):
            try:
              self._execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK), {})
              time.sleep(0.12)
            except Exception:
              break
          return None
        executed.append({"action": action, "edge_id": edge.edge_id})
      semantic_prefix.record_route_result(route, hit=True, reason="verified_landing")
      latency_s = time.perf_counter() - started
      action_dict = executed[-1]["action"].__dict__ if executed else {"action_type": json_action.WAIT}
      summary = f"Verified Semantic Prefix route ({route.length} hop(s))."
      record = {
          "goal": goal, "response": "[PREFIX_ROUTE_ASSIST]",
          "parsed_action": {"action": "PREFIX_ROUTE", "summary": summary},
          "tool_call": {"name": "semantic_prefix", "arguments": action_dict},
          "action_dict": action_dict, "summary": summary,
          "latency_sec": latency_s, "inference_skipped": True,
          "prefix_route_assist": True, "prefix_route_hit": True,
          "prefix_route_id": route.route_id, "successor_matched": True,
      }
      self._actions.append(record)
      self._summaries.append(summary)
      self._responses.append("[PREFIX_ROUTE_ASSIST]")
      self._write_action_log(goal)
      _append(semantic_prefix_events_path, {
          "event": "prefix_route_hit", "route_id": route.route_id,
          "length": route.length, "latency_s": latency_s,
      })
      semantic_prefix.save()
      semantic_prefix_summary_path.write_text(
          json.dumps(semantic_prefix.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
      return base_agent.AgentInteractionResult(False, record)
    except Exception as exc:  # Prefix must never turn into a task failure.
      if semantic_prefix is not None:
        semantic_prefix.record_route_result(route, hit=False, reason="exception") if "route" in locals() else None
        semantic_prefix.events.append({
            "event": "prefix_route_exception", "error": f"{type(exc).__name__}: {exc}",
        })
        semantic_prefix.save()
      _append(semantic_prefix_events_path, {
          "event": "prefix_route_exception", "error": f"{type(exc).__name__}: {exc}",
      })
      return None

  def bootstrap_target_app(self, goal: str) -> dict[str, Any] | None:
    """Open an unambiguous task app before the model's first reasoning call.

    This runs inside the existing evaluator step: after launch, the same step
    captures the app page, starts Ex5 from that anchor, and asks the model for
    the first task action. Unknown target apps and non-launcher pages are a
    strict no-op so the original GELAB reasoning path remains the fallback.
    """
    if not args.skip_open_app_reasoning or getattr(self, "_actions", []):
      return None
    try:
      before = ensure_skip_capture().capture()
    except Exception as exc:  # Observation failure must leave normal inference available.
      error = f"{type(exc).__name__}: {exc}"
      _append(bootstrap_events_path, {
          "task": args.task, "goal": goal, "package_match": False,
          "reasoning_step_skipped": False, "error": error,
      })
      return {"package_match": False, "reasoning_step_skipped": False,
              "error": error}
    target_app = _deterministic_bootstrap_target(
        goal, before.activity.component, enabled=True)
    if not target_app:
      return None
    started = time.perf_counter()
    try:
      action = json_action.JSONAction(
          action_type=json_action.OPEN_APP, app_name=target_app)
      self._execute_action(action, {})
      time.sleep(0.30)
      after = ensure_skip_capture().capture()
      package = after.activity.component.split("/", 1)[0]
      package_hints = explore_agent_text._APP_PACKAGE_HINTS.get(  # pylint: disable=protected-access
          target_app, ())
      matched = any(package.startswith(hint) for hint in package_hints)
      event = {
          "task": args.task, "goal": goal, "target_app": target_app,
          "source_activity": before.activity.component,
          "destination_activity": after.activity.component,
          "package_match": matched,
          "latency_s": time.perf_counter() - started,
          "reasoning_step_skipped": matched,
          "same_evaluator_step_continues": True,
      }
      _append(bootstrap_events_path, event)
      return event
    except Exception as exc:  # App bootstrap is advisory; inference still runs.
      error = f"{type(exc).__name__}: {exc}"
      _append(bootstrap_events_path, {
          "task": args.task, "goal": goal, "target_app": target_app,
          "package_match": False, "reasoning_step_skipped": False,
          "error": error, "latency_s": time.perf_counter() - started,
      })
      return {
          "target_app": target_app, "package_match": False,
          "reasoning_step_skipped": False, "error": error,
      }

  def try_skip_step(self, goal: str):
    nonlocal pending_prefix_edge_ids, prepared
    prefix_result = try_semantic_prefix_route(self, goal)
    if prefix_result is not None:
      return prefix_result
    if not (args.enable_skip_inference and args.agent_name == "mobileexplorer_no_memory"):
      return None
    # Note: device_dirty only disables further speculative exploration (see
    # parallel_predict). Whether a stored edge may still be reused here is an
    # independent question, answered below by resolve_graph_node matching the
    # freshly captured real state against a known committed node, plus
    # SkipInferenceGate's own confidence/entropy/freshness/risk gate. A
    # different exploration probe going dirty does not retroactively make an
    # already-VERIFIED/REUSABLE edge untrustworthy.
    history = getattr(self, "_actions", [])
    target_app = (explore_agent_text._infer_goal_target_app(goal)  # pylint: disable=protected-access
                  if not history else None)
    may_bootstrap = False  # All arms use the same model-controlled app launch.
    # Both licences the graph can offer (see get_reusable_action): a promoted
    # lookahead child, or an authoritative edge out of a screen the agent has
    # really stood on more than once. Checking only for REUSABLE meant that
    # with probing disabled this returned before the gate was ever queried,
    # so 20 revisited nodes across 15 tasks produced zero skips even though
    # four of them had a fresh, SAFE, zero-entropy verified edge waiting
    # (2026-08-30).
    has_graph_work = bool(pending_prefix_edge_ids) or any(
        edge.status == EdgeStatus.REUSABLE
        or (
            edge.status == EdgeStatus.VERIFIED
            and edge.dst_node not in (None, edge.src_node)
            and (graph.nodes[edge.src_node].visit_count > 1
                 if edge.src_node in graph.nodes else False)
        )
        for edge in graph.edges.values()
    )
    if not (may_bootstrap or has_graph_work):
      return None
    before = ensure_skip_capture().capture()
    if may_bootstrap and "nexuslauncher" in before.activity.component:
      started = time.perf_counter()
      action = json_action.JSONAction(
          action_type=json_action.OPEN_APP, app_name=target_app
      )
      self._execute_action(action, {})
      time.sleep(0.30)
      after = skip_capture.capture()
      after_package = after.activity.component.split("/", 1)[0]
      package_hints = explore_agent_text._APP_PACKAGE_HINTS.get(target_app, ())  # pylint: disable=protected-access
      matched = any(after_package.startswith(hint) for hint in package_hints)
      src_id = graph.make_node_id(before.activity.component, before.layout_sig)
      dst_id = graph.make_node_id(after.activity.component, after.layout_sig)
      for node_id, signature, status in (
          (src_id, before, NodeStatus.COMMITTED),
          (dst_id, after, NodeStatus.COMMITTED),
      ):
        graph.upsert_node(GraphNode(
            node_id=node_id, activity=signature.activity.component,
            package=signature.activity.component.split("/", 1)[0],
            visual_signature=signature.phash,
            structural_signature=signature.struct_sig.digest,
            layout_signature=signature.layout_sig,
            status=status, decision_entropy=0.0,
        ), visited=node_id == dst_id)
      edge = graph.add_speculative_transition(
          src_id, action.__dict__, dst_id,
          path_probability=1.0, confidence=1.0,
          expected_information_gain=0.0, risk_level="SAFE",
          exploration_cost=time.perf_counter() - started,
          rollback_success=True,
      )
      graph.record_execution_verification(edge.edge_id, matched)
      latency_s = time.perf_counter() - started
      # Model-visible history must read like any other action (see
      # apply_history_fix.py): our internal reason for taking it belongs in
      # skip_detail, not in the prompt.
      summary = str({"action": "open_app", "text": target_app})
      skip_detail = (
          f'Deterministic bootstrap skipped inference and opened "{target_app}"; '
          f"package_match={matched}."
      )
      step_record = {
          "goal": goal, "response": "[INFERENCE_SKIPPED_BOOTSTRAP]",
          "parsed_action": {"action": "AWAKE", "value": target_app, "summary": summary},
          "tool_call": {"name": "mobile_use", "arguments": {"action": "open_app", "text": target_app}},
          "action_dict": action.__dict__, "summary": summary, "skip_detail": skip_detail,
          "latency_sec": latency_s, "inference_skipped": True,
          "skip_kind": "deterministic_bootstrap",
          "graph_edge_id": edge.edge_id, "successor_matched": matched,
      }
      self._actions.append(step_record)
      self._summaries.append(summary)
      self._responses.append("[INFERENCE_SKIPPED_BOOTSTRAP]")
      self._write_action_log(goal)
      _append(skip_events_path, {
          "task": args.task, "goal": goal, "step": 0,
          "skip_kind": "deterministic_bootstrap",
          "edge_id": edge.edge_id, "src_node": src_id, "dst_node": dst_id,
          "confidence": edge.confidence, "action": action.__dict__,
          "successor_matched": matched, "latency_s": latency_s,
      })
      if executable_memory is None:
        graph_path.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
      # The process prepared for inference round zero captured the Launcher
      # state. Replace it so the next actual inference explores the opened app.
      stop_prepared_explorer(prepared)
      prepared = prepare_explorer_for_step(request_no + 1, 1, goal)
      return base_agent.AgentInteractionResult(False, step_record)
    # Coordinate comparison is brittle across normalized VLM coordinates,
    # nested clickable containers, and icon hit boxes.  At the next step the
    # actual successor is authoritative: if it equals an explored root
    # successor, the corresponding branch prefix is aligned even when the tap
    # coordinates differed.
    aligned_prefix_edge_ids: list[str] = []
    for edge_id in pending_prefix_edge_ids:
      edge = graph.edges.get(edge_id)
      if (
          edge is None or edge.dst_node is None or edge.dst_node == edge.src_node
          or edge.rollback_success is not True
          or not expected_successor_matches(edge.dst_node, before)
      ):
        continue
      graph.record_inference_alignment(edge.src_node, edge.action, True)
      graph.promote_children_of_aligned_prefix(edge.edge_id)
      graph.record_execution_verification(edge.edge_id, True)
      aligned_prefix_edge_ids.append(edge.edge_id)
      element_identity = edge.action.get("element_identity")
      if element_identity:
        # Ground truth, not a guess: probing this element led where the
        # authoritative model's own action then led, so it is what the model
        # effectively chose at this node. Prioritize it over text similarity
        # the next time this node is visited (InformationNeedRanker). This
        # bookkeeping moved here from the deleted coordinate comparison.
        node_hits = node_candidate_hits.setdefault(edge.src_node, {})
        node_hits[element_identity] = node_hits.get(element_identity, 0) + 1
    pending_prefix_edge_ids = []
    current_node_id = resolve_graph_node(before)
    if current_node_id is None:
      return None
    decision = skip_gate.query(current_node_id)
    if not decision.skip or decision.edge is None:
      return None
    edge = decision.edge
    # Conservative reuse: exact content state, repeatedly executed same-task action,
    # no writes/input/answers and no prior skip failure. This is evidence sufficiency,
    # not a calibrated semantic-correctness guarantee.
    source = graph.nodes.get(current_node_id)
    if (source is None or source.structural_signature != before.struct_sig.digest
        or edge.execution_hit_count < 3 or edge.execution_miss_count
        or edge.skip_attempt_count > edge.skip_success_count
        or edge.action.get("action_type", "").lower() not in {"click", "navigate_back"}
        or edge.risk_level != "SAFE"):
      return None
    # Captured before execute/verify: record_execution_verification below can
    # zero this out on a mismatch, which previously made skip_events.jsonl
    # misleadingly log "confidence=0.0" for a skip that was actually offered
    # (and taken) at a high, threshold-crossing confidence that just turned
    # out to be wrong (2026-08-29, ExpenseDeleteMultiple).
    decision_confidence = edge.confidence
    started = time.perf_counter()
    action_dict = {
        key: value for key, value in edge.action.items()
        if key in {"action_type", "x", "y", "text", "direction", "app_name"}
    }
    action_dict["action_type"] = str(action_dict.get("action_type", "")).lower()
    action = json_action.JSONAction(**action_dict)
    self._execute_action(action, {})
    time.sleep(0.20)
    after = skip_capture.capture()
    matched = expected_successor_matches(edge.dst_node, after)
    graph.record_execution_verification(edge.edge_id, matched)
    budget_controller.observe_explorer_result(matched)
    if matched:
      # This hop just earned real trust (VERIFIED), so its own already
      # -explored children (probed speculatively at depth>=2 while the
      # authoritative model computed some earlier step) can now extend the
      # reusable chain one more hop, without waiting for the model to walk
      # this edge again.
      graph.promote_children_of_aligned_prefix(edge.edge_id)
    if not matched and edge.dst_node:
      graph.invalidate_subtree(edge.dst_node)
    latency_s = time.perf_counter() - started
    # Same reasoning as the bootstrap case above: the history the model reads
    # describes the action, never the fact that a cached edge chose it.
    summary = str(
        {"action": action_dict.get("action_type"),
         "coordinate": [action_dict.get("x"), action_dict.get("y")]}
        if action_dict.get("x") is not None
        else {k: v for k, v in action_dict.items() if v is not None}
    )
    skip_detail = (
        f"Skipped model inference using verified lookahead edge {edge.edge_id}; "
        f"successor_match={matched}."
    )
    step_record = {
        "goal": goal, "response": "[INFERENCE_SKIPPED]",
        "parsed_action": {"action": action.action_type, "summary": summary},
        "tool_call": {"name": "mobile_use", "arguments": action_dict},
        "action_dict": action.__dict__, "summary": summary, "skip_detail": skip_detail,
        "latency_sec": latency_s, "inference_skipped": True,
        "graph_edge_id": edge.edge_id, "successor_matched": matched,
    }
    self._actions.append(step_record)
    self._summaries.append(summary)
    self._responses.append("[INFERENCE_SKIPPED]")
    self._write_action_log(goal)
    _append(skip_events_path, {
        "task": args.task, "goal": goal, "step": len(self._actions) - 1,
        "edge_id": edge.edge_id, "src_node": current_node_id,
        "dst_node": edge.dst_node, "confidence": decision_confidence,
        "action": action_dict, "successor_matched": matched,
        "latency_s": latency_s,
    })
    if executable_memory is None:
      graph_path.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return base_agent.AgentInteractionResult(False, step_record)

  def timed_step(self, goal: str):
    nonlocal pending_prefix_edge_ids, device_dirty, prepared, last_intent
    nonlocal current_source_version, current_query, evidence_event_cursor
    nonlocal semantic_prefix_event_cursor
    step_started = time.perf_counter()
    bootstrap_event = bootstrap_target_app(self, goal)
    skipped_result = try_skip_step(self, goal)
    if skipped_result is not None:
      _append(steps_path, {
          "task": args.task, "goal": goal,
          "step": len(getattr(self, "_actions", [])) - 1,
          "step_total_s": skipped_result.data.get("latency_sec", 0.0),
          "done": False, "action": skipped_result.data.get("action_dict", {}),
          "inference_skipped": True,
          "successor_matched": skipped_result.data.get("successor_matched"),
          "pre_reasoning_bootstrap": bootstrap_event,
      })
      return skipped_result
    history = getattr(self, "_actions", getattr(self, "history", []))
    committed_actions = []
    for item in history:
      if not isinstance(item, dict):
        continue
      raw_action = item.get("action_dict") or item.get("action_output_json")
      action = dict(getattr(raw_action, "__dict__", raw_action) or {})
      committed_actions.append({
          "action_dict": action,
          "tool_call": item.get("tool_call") or {},
      })
    profile = {
        "task": args.task, "goal": goal, "step": len(history),
        "committed_actions": committed_actions,
        "pre_reasoning_bootstrap": bootstrap_event,
    }
    token = profile_var.set(profile)
    # One capture serves two purposes: the evidence summary for this prompt,
    # and the "before" endpoint of the authoritative transition recorded
    # after the step completes.
    before_state = ensure_skip_capture().capture() if (
        args.two_system or args.semantic_prefix_mode in {"prompt", "assist"}
    ) else None
    profile["executable_memory_page"] = (
        executable_page_from_capture(before_state)
        if executable_memory is not None and before_state is not None else None
    )
    current_source_version = before_state.struct_sig.digest if before_state else ""
    current_query = goal + " " + last_intent
    evidence_token = evidence_var.set(
        exploration_evidence_for(resolve_graph_node(before_state))
        if (args.inject_evidence and before_state is not None) else ""
    )
    prefix_token = prefix_var.set(
        semantic_prefix_context_for(before_state, goal)
        if before_state is not None else "")
    # Bind both systems to this step's live page. Preparing the next worker
    # during the previous inference captured a stale pre-action page, so from
    # step 2 onward exploration and reasoning could start from different
    # states. A fresh worker here preserves the required parallel semantics.
    if args.two_system and args.variant != "offline":
      stop_prepared_explorer(prepared)
      prepared = prepare_explorer_for_step(
          request_no + 1, profile["step"], goal)
    started = step_started
    try:
      result = original_step(self, goal)
    finally:
      profile_var.reset(token)
      evidence_var.reset(evidence_token)
      prefix_var.reset(prefix_token)
    # The raw page is process-local guidance input, not an event-log field.
    # It contains the full accessibility tree and can be large or dynamic.
    profile.pop("executable_memory_page", None)
    if device_dirty:
      # A prior round's rollback failure only tells us that PAST screen was
      # left unverified, not that the device is dirty forever. The real step
      # that just completed lands on a fresh state chosen by the
      # authoritative model, so it is safe to try exploring again from here.
      # Element-level and subtree invalidations already recorded in the
      # graph/blocklist persist regardless. KNOWN RISK (2026-08-28,
      # ExpenseDeleteMultiple): this let exploration cascade through an
      # expense's edit view into an Android share sheet exposing the real
      # logged-in Google account before recovery caught up. Kept enabled
      # deliberately per user direction; not yet mitigated.
      device_dirty = False
    summaries = getattr(self, "_summaries", None) or []
    if summaries:
      # Only overwrite when this step actually stated an intent; otherwise
      # keep the last one that did, which is still the most recent forward
      # -looking thing the model said.
      stated = _stated_next_intent(summaries[-1])
      if stated:
        last_intent = stated
    raw_action = result.data.get("action_dict") or result.data.get("action_output_json")
    action_dict = dict(getattr(raw_action, "__dict__", raw_action) or {})
    if semantic_prefix is not None and result.data.get("parse_error"):
      semantic_prefix.record_parse_error(
          result.data.get("parse_error"), action_dict or {"action_type": json_action.WAIT})
      semantic_prefix.save()
      _append(semantic_prefix_events_path, semantic_prefix.events[-1])
      semantic_prefix_event_cursor = len(semantic_prefix.events)
      semantic_prefix_summary_path.write_text(
          json.dumps(semantic_prefix.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
    # Loop/regression check: if the model's real action here nearly repeats a
    # real action from a few steps back, that is a general, observed sign
    # that no real progress happened in between - whether or not a
    # skip-executed edge is the direct cause. Originally this only fired
    # when a skip sat between the two repeats (2026-08-28,
    # ClockStopWatchRunning: a cached edge kept re-toggling the stopwatch
    # right after the model (re-)started it). But most looping failures
    # observed on 2026-08-29's 15-task batch (RecipeDeleteMultipleRecipes,
    # RecipeAddMultipleRecipes, MarkorEditNote, ExpenseDeleteMultiple) had
    # NO skip in between at all - the model got stuck purely on its own,
    # often shortly after an unrelated skip mismatch earlier in the episode
    # confused its sense of what state it was actually in. Restricting the
    # trigger to "a skip must be the direct cause" was missing most of the
    # real occurrences, so it now fires on any repeat.
    actions_before_this_step = getattr(self, "_actions", getattr(self, "history", []))[:-1]
    skip_edge_ids_since_prior_real: list[str] = []
    prior_real_action_dict: dict[str, Any] | None = None
    for item in reversed(actions_before_this_step):
      if not isinstance(item, dict):
        continue
      if item.get("inference_skipped"):
        edge_id = item.get("graph_edge_id")
        if edge_id:
          skip_edge_ids_since_prior_real.append(edge_id)
        continue
      prior_raw = item.get("action_dict") or item.get("action_output_json")
      prior_real_action_dict = dict(getattr(prior_raw, "__dict__", prior_raw) or {})
      break
    if prior_real_action_dict is not None and _actions_near_duplicate(action_dict, prior_real_action_dict):
      for stale_edge_id in skip_edge_ids_since_prior_real:
        stale_edge = graph.edges.get(stale_edge_id)
        if stale_edge is None:
          continue
        stale_edge.status = EdgeStatus.INVALID
        stale_edge.confidence = 0.0
        if stale_edge.dst_node:
          graph.invalidate_subtree(stale_edge.dst_node)
      # Whether or not a skip was directly implicated, a loop is negative
      # evidence for how much the rest of this episode should trust
      # speculative reuse: feed it into the same reliability signal that
      # already scales ResourceAdaptiveBudgetController's aggressiveness
      # (see resources.py), rather than inventing a parallel mechanism.
      budget_controller.observe_explorer_result(False)
      _append(root / "loop_detected.jsonl", {
          "task": args.task, "goal": goal, "step": len(actions_before_this_step),
          "repeated_action": action_dict, "prior_action": prior_real_action_dict,
          "invalidated_edge_ids": skip_edge_ids_since_prior_real,
          "skip_directly_implicated": bool(skip_edge_ids_since_prior_real),
      })
    current_round = pending_rounds.pop(request_no, None)
    if current_round is not None:
      current_round.inference_output_action = action_dict
      aligned = 0
      # Alignment used to be decided here by comparing tap coordinates within
      # an 80px radius, which was both an arbitrary constant and structurally
      # blind: it only ran for action_type=="click", and clicks are just 63%
      # of what the model actually emits (measured over 191 rounds on
      # 2026-08-29 - open_app 17%, long_press 7%, swipe 4%, input_text 4%).
      # Center-to-center distance is also unreliable across normalized VLM
      # coordinates, nested clickable containers and icon hit boxes, so it
      # judged only 14 of 329 depth-1 probes aligned (4.3%).
      #
      # Two actions are equivalent when they leave the device in the same
      # state, so alignment is now decided by the landing state alone, in
      # try_skip_step at the next step, where the real successor is known
      # (see expected_successor_matches). That criterion is an observation
      # rather than a threshold, and it applies to every action type.
      # pending_prefix_edge_ids carries this round's depth-1 edges there.
      current_round.memory_confidence = max(
          (graph.get_local_confidence(graph.edges[item["edge_id"]].src_node) for item in current_round.explored_edges if item["edge_id"] in graph.edges),
          default=0.0,
      )
      round_logger.append(current_round)
      pending_prefix_edge_ids = [
          item["edge_id"] for item in current_round.explored_edges
          if item.get("depth") == 1 and item.get("edge_id") in graph.edges
      ]
      if executable_memory is None:
        graph_path.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if before_state is not None and not result.done:
      # Progressive memory from the authoritative path itself: the model
      # just executed this action for real and the device landed somewhere
      # observable. Recording it costs one extra state capture and carries
      # no risk at all - unlike a speculative probe, nothing needs to be
      # rolled back because this transition was supposed to happen. It is
      # marked VERIFIED because it is a real, observed execution, which is
      # strictly stronger evidence than any speculative probe can provide.
      # Without this the graph only ever learned from probes, throwing away
      # the one perfectly reliable source of state-action transitions the
      # system already has.
      try:
        after_state = ensure_skip_capture().capture()
        if semantic_prefix is not None:
          ax, ay = action_dict.get("x"), action_dict.get("y")
          hit_elements = []
          if ax is not None and ay is not None:
            for candidate in before_state.elements:
              left, top, right, bottom = candidate.bounds
              if (right > left and bottom > top
                  and left <= float(ax) <= right
                  and top <= float(ay) <= bottom):
                hit_elements.append(((right - left) * (bottom - top), candidate))
          if hit_elements:
            _, live_element = min(hit_elements, key=lambda item: item[0])
            confirmed = semantic_prefix.confirm_authoritative_transition(
                source_state_id=state_id_from_signature(before_state),
                target_state_id=state_id_from_signature(after_state),
                raw_element=live_element, action=action_dict,
            )
            if confirmed is not None:
              _append(semantic_prefix_events_path, semantic_prefix.events[-1])
              semantic_prefix.save()
        src_id = graph.make_node_id(before_state.activity.component, before_state.layout_sig)
        dst_id = graph.make_node_id(after_state.activity.component, after_state.layout_sig)
        if src_id != dst_id:
          for node_id, sig in ((src_id, before_state), (dst_id, after_state)):
            graph.upsert_node(GraphNode(
                node_id=node_id, activity=sig.activity.component,
                package=sig.activity.component.split("/", 1)[0],
                visual_signature=sig.phash,
                structural_signature=sig.struct_sig.digest,
                layout_signature=sig.layout_sig,
                status=NodeStatus.COMMITTED, decision_entropy=0.0,
            # Only the arrival counts as a visit: the agent is standing here
            # now. src was counted when it was some earlier step's arrival, so
            # counting it again would make every node look revisited after one
            # pass and hand out reuse rights that were never earned.
            ), visited=node_id == dst_id)
          # Same predictive payload a speculative probe would have produced
          # ("this action leads to a screen showing X"), obtained here at
          # zero risk because the action was going to be taken anyway. This
          # is what lets evidence injection work even with probing fully
          # disabled, where discovered_labels would otherwise never be
          # populated.
          destination_labels = tuple(dict.fromkeys(
              label for element in after_state.elements
              for label in (element.text, element.content_desc)
              if label and len(label) < 40
          ))[:8]
          # Remember WHICH control the model tapped, not just where. On a
          # repetitive task the same screen recurs with different content, so
          # raw coordinates alone can land on a different row next time round
          # (2026-08-30: authoritative reuse landed as predicted only 2 of 9
          # times). Recorded from the pre-action state by hit-testing the
          # action's own coordinates, so it costs nothing extra.
          # Record whether the model tapped the screen's fixed chrome or one
          # of its content rows. A control kind that occurs once on the
          # screen (a FAB, a toolbar button, Save) is part of what the screen
          # IS, and replaying a tap on it later means the same thing. A
          # control kind that repeats is a list row, and "the first row" is
          # a different thing on every visit - replaying it after the item
          # was deleted or a new one was added taps whatever moved into that
          # slot. This is the same singleton-vs-repeated distinction that
          # layout_signature uses for screen identity (see state.py), applied
          # to the question of which actions may be replayed at all.
          anchored_action = dict(action_dict)
          ax, ay = anchored_action.get("x"), anchored_action.get("y")
          if ax is not None and ay is not None:
            kinds: dict[tuple[str, str], int] = {}
            for element in before_state.elements:
              key = (element.class_name or "", element.resource_id or "")
              kinds[key] = kinds.get(key, 0) + 1
            for element in before_state.elements:
              left, top, right, bottom = element.bounds
              if left <= float(ax) <= right and top <= float(ay) <= bottom:
                key = (element.class_name or "", element.resource_id or "")
                anchored_action["element_anchor"] = "|".join(key)
                anchored_action["element_role"] = (
                    "content" if kinds[key] > 1 else "chrome")
                if element.identity:
                  anchored_action["control_key"] = control_key_from_identity(element.identity)
                  anchored_action["stable_control_key"] = stable_control_key_from_identity(
                      element.identity)
                break
          observed = graph.add_speculative_transition(
              src_id, anchored_action, dst_id,
              path_probability=1.0, confidence=0.0,
              expected_information_gain=0.0, risk_level="SAFE",
              exploration_cost=0.0, rollback_success=True,
              discovered_labels=destination_labels,
          )
          graph.record_execution_verification(observed.edge_id, True)
          _append(root / "authoritative_edges.jsonl", {
              "task": args.task, "step": profile["step"], "edge_id": observed.edge_id,
              "src_node": src_id, "dst_node": dst_id, "action": action_dict,
          })
      except Exception as exc:
        _append(root / "authoritative_edges.jsonl", {
            "task": args.task, "step": profile["step"],
            "error": f"{type(exc).__name__}: {exc}",
        })
    evidence_index.commit()
    for event in evidence_index.events[evidence_event_cursor:]:
      _append(root / "evidence_events.jsonl", event)
    evidence_event_cursor = len(evidence_index.events)
    _append(steps_path, {
        **profile, "step_total_s": time.perf_counter() - started,
        "done": bool(result.done), "action": action_dict,
    })
    return result

  agent_class.step = timed_step
  if args.inject_evidence or args.semantic_prefix_mode in {"prompt", "assist"}:
    mobileexplorer.build_mobileexplorer_messages = build_messages_with_evidence
  (root / "meta.json").write_text(json.dumps({
      "started_at": dt.datetime.now().isoformat(), "task": args.task,
      "max_steps": args.max_steps, "seed": args.seed, "ranker": args.ranker,
      "agent": args.agent_name, "summary_enabled": bool(args.m3a_summary),
      "exploration": "parallel_live_probe", "variant": args.variant,
      "two_system": args.two_system,
      "semantic_prefix": {
          "mode": args.semantic_prefix_mode,
          "path": str(semantic_prefix_path) if semantic_prefix is not None else None,
          "top_k": args.semantic_prefix_top_k,
          "shortcut_confidence": 0.82,
          "max_route_length": 3,
      },
      "executable_memory": {
          "enabled": executable_memory is not None,
          "path": str(executable_memory_path) if executable_memory is not None else None,
          "config": str(args.executable_memory_config) if args.executable_memory_config else None,
      },
      "skip_inference_enabled": args.enable_skip_inference,
      "two_system_skip_inference_enabled": args.enable_skip_inference,
      "executable_memory_high_confidence_skip_enabled": bool(
          executable_memory is not None
          and executable_memory.config.high_confidence_skip_enabled),
      "two_system_config": dataclasses.asdict(two_system_config),
      "exploration_execution": {
          "min_probes": args.min_probes,
          "force_min_probes": args.force_min_probes,
          "post_inference_grace_s": args.post_inference_grace_s,
      },
  }, ensure_ascii=False, indent=2), encoding="utf-8")

  # ``timed_step`` prepares from the task's actual current page immediately
  # before the model call. Do not capture an emulator setup/previous page here.
  prepared = None

  sys.argv = [
      "run.py", "--suite_family=android_world", f"--agent_name={args.agent_name}",
      f"--tasks={args.task}", "--n_task_combinations=1", "--fixed_task_seed",
      f"--task_random_seed={args.seed}", f"--max_n_steps={args.max_steps}",
      f"--console_port={args.console_port}", f"--checkpoint_dir={root / 'checkpoints'}",
      f"--adb_path={adb_path}",
  ]
  try:
    runpy.run_path("run.py", run_name="__main__")
  except SystemExit as exc:
    return int(exc.code or 0)
  finally:
    stop_prepared_explorer(prepared)
    if skip_capture is not None:
      skip_capture.close()
    if semantic_prefix is not None:
      semantic_prefix.save()
      prefix_summary = semantic_prefix.summary()
      cumulative_metrics = prefix_summary.get("metrics") or {}
      prefix_summary["cumulative_metrics"] = dict(cumulative_metrics)
      prefix_summary["delta_metrics"] = {
          key: max(0, int(value) - int(semantic_prefix_initial_metrics.get(key, 0)))
          for key, value in cumulative_metrics.items()
          if key not in {"nodes", "edges"}
      }
      semantic_prefix_summary_path.write_text(
          json.dumps(prefix_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if executable_memory is not None:
      # Read the final shared graph because the agent and shadow ingester both
      # contribute. Evaluator output remains the only source of task success.
      final_memory = ExecutableExplorationMemory(
          dataclasses.replace(memory_config, enabled=True, max_probes=args.max_probes),
          path=executable_memory_path,
      )
      executable_memory_summary_path.write_text(json.dumps({
          "schema_version": 1,
          "path": str(executable_memory_path),
          "node_count": len(final_memory.states),
          "edge_count": len(final_memory.edges),
          "sample_count": len(final_memory.samples),
          "skill_count": len(final_memory.skills),
          "metrics": final_memory.metrics.to_dict(),
      }, ensure_ascii=False, indent=2), encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
