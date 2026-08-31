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
import subprocess
import sys
import threading
import time
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world.agents import gelab_agent
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
from android_world.parallel_exploration.config import TwoSystemConfig
from android_world.parallel_exploration.logging import InferenceRoundRecord
from android_world.parallel_exploration.logging import RoundLogger
from android_world.parallel_exploration.information import parse_information_need
from android_world.parallel_exploration.resources import ResourceAdaptiveBudgetController
from android_world.parallel_exploration.resources import ResourceMonitor
from android_world.parallel_exploration.runtime import SkipInferenceGate
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


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--task", default="ClockStopWatchRunning")
  parser.add_argument("--max_steps", type=int, default=6)
  parser.add_argument("--seed", type=int, default=30)
  parser.add_argument("--ranker", choices=["RandomRanker", "SimpleRelevanceRanker", "InformationNeedRanker"], default="InformationNeedRanker")
  parser.add_argument("--metrics_url", default="http://127.0.0.1:8081/metrics")
  parser.add_argument("--two_system", action="store_true")
  parser.add_argument("--enable_skip_inference", action="store_true")
  parser.add_argument("--skip_confidence", type=float, default=0.90)
  parser.add_argument("--max_probes", type=int, default=12)
  parser.add_argument("--min_probes", type=int, default=4)
  parser.add_argument("--post_inference_grace_s", type=float, default=4.0)
  parser.add_argument("--max_depth", type=int, default=3)
  parser.add_argument("--max_exploration_time_s", type=float, default=8.0)
  parser.add_argument("--a11y_method", choices=["fast_provider", "uiautomator", "grpc"], default="fast_provider")
  parser.add_argument("--agent_name", choices=["gelab_agent", "m3a_llamacpp"], default="m3a_llamacpp")
  parser.add_argument("--m3a_summary", action="store_true", help="Enable M3A's second model call for summaries")
  parser.add_argument("--inject_evidence", action="store_true", help="Summarize exploration findings into the next prompt")
  args = parser.parse_args()
  if args.a11y_method == "grpc":
    os.environ.pop("ANDROID_WORLD_A11Y_METHOD", None)
  else:
    os.environ["ANDROID_WORLD_A11Y_METHOD"] = args.a11y_method
  if args.a11y_method == "fast_provider":
    adb_path = "/Users/huangrunxi/Library/Android/sdk/platform-tools/adb"
    subprocess.run(
        [adb_path, "-s", "emulator-5554", "forward", "tcp:8765", "localabstract:androidworld_fast_a11y"],
        check=True, timeout=10,
    )
    os.environ["ANDROID_WORLD_FAST_A11Y_SOCKET_PORT"] = "8765"
  if args.agent_name == "m3a_llamacpp" and not args.m3a_summary:
    os.environ["ANDROID_WORLD_M3A_DISABLE_SUMMARY"] = "1"
  root = args.output.resolve()
  root.mkdir(parents=True, exist_ok=False)
  trace = root / "probe_trace.jsonl"
  filtered = root / "filtered_elements.jsonl"
  windows = root / "inference_windows.jsonl"
  requests_path = root / "request_latency.jsonl"
  steps_path = root / "step_latency.jsonl"
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
  skip_gate = SkipInferenceGate(graph, two_system_config)
  resource_monitor = ResourceMonitor()
  budget_controller = ResourceAdaptiveBudgetController(two_system_config)
  round_logger = RoundLogger(root / "two_system_rounds.jsonl")
  graph_path = root / "progressive_belief_graph.json"
  skip_events_path = root / "skip_events.jsonl"
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
    nonlocal trace_cursor
    if not trace.exists():
      return []
    rows = trace.read_text(encoding="utf-8").splitlines()
    new_rows = [json.loads(line) for line in rows[trace_cursor:] if line.strip()]
    trace_cursor = len(rows)
    ingested = []
    edge_ids = []
    for row in new_rows:
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
      role = str(element.get("class", "")).rsplit(".", 1)[-1].casefold()
      depth = int(row.get("depth", 1))
      risk = (
          "LOW" if depth >= 2 and bool(row.get("recovery_ok"))
          else ("LOW" if role in {"imagebutton", "tabwidget"} else "UNKNOWN")
      )
      edge = graph.add_speculative_transition(
          src_id, item["action"], dst_id,
          path_probability=max(0.05, float(element.get("score", 0.0))),
          confidence=0.20, expected_information_gain=min(1.0, row.get("discovered", {}).get("new_element_count", 0) / 10.0),
          risk_level=risk,
          exploration_cost=float(row.get("timings_ms", {}).get("total", 0.0)) / 1000.0,
          rollback_success=bool(row.get("recovery_ok")),
          discovered_labels=tuple(
              str(label) for label in (row.get("discovered") or {}).get("new_texts", [])[:8]
          ),
      )
      edge_ids.append(edge.edge_id)
      ingested.append({
          "edge_id": edge.edge_id, "depth": depth,
          "predicted_IG": edge.expected_information_gain,
          "realized_IG": edge.realized_information_gain,
          "risk_level": risk,
      })
    trial_edges[trial_id] = edge_ids
    graph_path.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
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
    return {
        "trial_id": f"{args.task}-step{step}-request{number}",
        "trace_path": str(trace), "filtered_path": str(filtered),
        "task": task_text, "step_idx": step,
        "ranker": args.ranker, "seed": args.seed + number,
        "serial": "emulator-5554", "console_port": 5554, "a11y_local_port": 8765,
        "restore_timeout_s": 30.0,
        "max_probes": last_budget.max_probes if args.two_system else args.max_probes,
        "min_probes": args.min_probes,
        "post_inference_grace_s": args.post_inference_grace_s,
        "max_depth": last_budget.max_depth if args.two_system else 1,
        "max_exploration_time_s": last_budget.max_exploration_time_s if args.two_system else args.max_exploration_time_s,
        "information_need": information_need.to_dict(),
        "blocked_element_identities": sorted(blocked_element_identities),
        "blocked_recovery_contexts": sorted(blocked_recovery_contexts),
        "node_candidate_hits": {node: dict(hits) for node, hits in node_candidate_hits.items()},
        "explored_element_identities": explored_element_identities_by_node(),
        "goal_relevance_threshold": two_system_config.goal_relevance_threshold,
        "max_stack_depth_increase": two_system_config.max_stack_depth_increase,
    }

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
    if isinstance(prepared.config, dict):
      prepared.config["committed_actions"] = list(profile.get("committed_actions") or [])
    trial_id = str(prepared.config["trial_id"])
    next_holder: dict[str, Any] = {}
    def prepare_next() -> None:
      try:
        next_holder["value"] = prepare_explorer(explorer_config(
            request_no + 1, int(profile.get("step", -1)) + 1,
            str(profile.get("goal", args.task)),
        ))
      except Exception as exc:
        next_holder["error"] = exc
    next_thread = threading.Thread(target=prepare_next, daemon=True)
    next_thread.start()
    used_budget = last_budget
    before = snapshot()
    resources_before = resource_monitor.sample()
    process_memory_before = resources_before.process_memory_mb
    client_started = time.perf_counter()
    result, window = run_prepared_explorer(
        prepared, lambda: original_predict(self, text_prompt, images, messages)
    )
    client_total_s = time.perf_counter() - client_started
    next_thread.join(timeout=15)
    if "error" in next_holder:
      raise RuntimeError(f"Failed to prewarm next explorer: {next_holder['error']}")
    rollback_ok = window["restore_status"] == EventKind.RESTORED.value
    if rollback_ok:
      prepared = next_holder.get("value")
    else:
      # Do not assume rollback always succeeds. Abandon the prewarmed next
      # explorer, stop probing for the rest of this episode, and let the
      # authoritative inference result execute against the real (unverified)
      # screen rather than aborting the whole task.
      device_dirty = True
      stop_prepared_explorer(next_holder.get("value"))
      prepared = None
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
        "critical_path_extension_ms": window["critical_path_extension_ms"],
        "restore_status": window["restore_status"],
    }
    _append(requests_path, row)
    _append(windows, {**window, "task": args.task, "goal": profile.get("goal", ""), "step": profile.get("step")})
    explored_edges = ingest_probe_trace(trial_id)
    if not rollback_ok:
      for edge_id in trial_edges.get(trial_id, ()):
        edge = graph.edges.get(edge_id)
        if edge is not None:
          graph.invalidate_subtree(edge.src_node)
      _append(dirty_events_path, {
          "trial_id": trial_id, "window": window,
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
  agent_class = gelab_agent.GELABAgent if args.agent_name == "gelab_agent" else m3a.M3A
  original_step = agent_class.step
  skip_capture = None

  def ensure_skip_capture():
    nonlocal skip_capture
    if skip_capture is None:
      skip_capture = create_optimized_state_capture(
          serial="emulator-5554", console_port=5554,
          adb_path="/Users/huangrunxi/Library/Android/sdk/platform-tools/adb",
          a11y_local_port=8765,
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
  original_build_messages = gelab_agent.build_gelab_messages

  def build_messages_with_evidence(goal, history, screenshot):
    messages = original_build_messages(goal, history, screenshot)
    evidence = evidence_var.get()
    if evidence:
      for item in messages[-1]["content"]:
        if item.get("type") == "text":
          item["text"] = f"{item['text']}\n\nKnown from exploring this screen:\n{evidence}"
          break
    return messages

  def exploration_evidence_for(node_id: str | None) -> str:
    if not args.inject_evidence or not node_id:
      return ""
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

  def try_skip_step(self, goal: str):
    nonlocal pending_prefix_edge_ids, prepared
    if not (args.enable_skip_inference and args.agent_name == "gelab_agent"):
      return None
    # Note: device_dirty only disables further speculative exploration (see
    # parallel_predict). Whether a stored edge may still be reused here is an
    # independent question, answered below by resolve_graph_node matching the
    # freshly captured real state against a known committed node, plus
    # SkipInferenceGate's own confidence/entropy/freshness/risk gate. A
    # different exploration probe going dirty does not retroactively make an
    # already-VERIFIED/REUSABLE edge untrustworthy.
    history = getattr(self, "_actions", [])
    target_app = gelab_agent._infer_goal_target_app(goal) if not history else None
    may_bootstrap = bool(target_app)
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
      package_hints = gelab_agent._APP_PACKAGE_HINTS.get(target_app, ())
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
      graph_path.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
      # The process prepared for inference round zero captured the Launcher
      # state. Replace it so the next actual inference explores the opened app.
      stop_prepared_explorer(prepared)
      prepared = prepare_explorer(explorer_config(request_no + 1, 1, goal))
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
    graph_path.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return base_agent.AgentInteractionResult(False, step_record)

  def timed_step(self, goal: str):
    nonlocal pending_prefix_edge_ids, device_dirty, prepared, last_intent
    skipped_result = try_skip_step(self, goal)
    if skipped_result is not None:
      _append(steps_path, {
          "task": args.task, "goal": goal,
          "step": len(getattr(self, "_actions", [])) - 1,
          "step_total_s": skipped_result.data.get("latency_sec", 0.0),
          "done": False, "action": skipped_result.data.get("action_dict", {}),
          "inference_skipped": True,
          "successor_matched": skipped_result.data.get("successor_matched"),
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
    }
    token = profile_var.set(profile)
    # One capture serves two purposes: the evidence summary for this prompt,
    # and the "before" endpoint of the authoritative transition recorded
    # after the step completes.
    before_state = ensure_skip_capture().capture() if args.two_system else None
    evidence_token = evidence_var.set(
        exploration_evidence_for(resolve_graph_node(before_state))
        if (args.inject_evidence and before_state is not None) else ""
    )
    started = time.perf_counter()
    try:
      result = original_step(self, goal)
    finally:
      profile_var.reset(token)
      evidence_var.reset(evidence_token)
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
      prepared = prepare_explorer(explorer_config(request_no + 1, profile["step"] + 1, goal))
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
    _append(steps_path, {
        **profile, "step_total_s": time.perf_counter() - started,
        "done": bool(result.done), "action": action_dict,
    })
    return result

  agent_class.step = timed_step
  if args.inject_evidence:
    gelab_agent.build_gelab_messages = build_messages_with_evidence
  (root / "meta.json").write_text(json.dumps({
      "started_at": dt.datetime.now().isoformat(), "task": args.task,
      "max_steps": args.max_steps, "seed": args.seed, "ranker": args.ranker,
      "agent": args.agent_name, "summary_enabled": bool(args.m3a_summary),
      "exploration": "parallel_live_probe",
      "two_system": args.two_system,
      "skip_inference_enabled": args.enable_skip_inference,
      "two_system_config": dataclasses.asdict(two_system_config),
      "exploration_execution": {
          "min_probes": args.min_probes,
          "post_inference_grace_s": args.post_inference_grace_s,
      },
  }, ensure_ascii=False, indent=2), encoding="utf-8")

  prepared = prepare_explorer(explorer_config(1, 0, args.task))

  sys.argv = [
      "run.py", "--suite_family=android_world", f"--agent_name={args.agent_name}",
      f"--tasks={args.task}", "--n_task_combinations=1", "--fixed_task_seed",
      f"--task_random_seed={args.seed}", f"--max_n_steps={args.max_steps}",
      "--console_port=5554", f"--output_path={root / 'androidworld_run'}",
  ]
  try:
    runpy.run_path("run.py", run_name="__main__")
  except SystemExit as exc:
    return int(exc.code or 0)
  finally:
    stop_prepared_explorer(prepared)
    if skip_capture is not None:
      skip_capture.close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
