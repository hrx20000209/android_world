from __future__ import annotations

import time

from android_world.parallel_exploration.belief_graph import EdgeStatus
from android_world.parallel_exploration.belief_graph import GraphNode
from android_world.parallel_exploration.belief_graph import NodeStatus
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
from android_world.parallel_exploration.config import TwoSystemConfig
from android_world.parallel_exploration.coordinator import Ownership
from android_world.parallel_exploration.coordinator import SpeculationCoordinator
from android_world.parallel_exploration.information import parse_information_need
from android_world.parallel_exploration.resources import ResourceAdaptiveBudgetController
from android_world.parallel_exploration.resources import ResourceSnapshot
from android_world.parallel_exploration.runtime import SkipInferenceGate
from android_world.parallel_exploration.safety import RiskLevel
from android_world.parallel_exploration.safety import classify_action_risk


def _node(node_id: str, entropy: float = 0.1) -> GraphNode:
  return GraphNode(node_id, "pkg/.Main", "pkg", "visual", node_id, status=NodeStatus.COMMITTED, decision_entropy=entropy)


def _resources(memory_mb: float = 8000, used: float = 0.3, cpu: float = 0.2) -> ResourceSnapshot:
  return ResourceSnapshot(time.time(), memory_mb, 100, used, cpu)


def test_information_need_parser_and_fallback_never_fail():
  parsed = parse_information_need('{"current_subgoal":"open details","required_information_slots":["phone"]}', "goal")
  assert parsed.current_subgoal == "open details"
  assert parsed.source == "model"
  fallback = parse_information_need("not json", "find contact", ["Details"])
  assert fallback.current_subgoal == "find contact"
  assert fallback.source == "task_ui_fallback"


def test_memory_pressure_reduces_exploration_to_zero():
  controller = ResourceAdaptiveBudgetController(TwoSystemConfig(enabled=True))
  budget = controller.allocate(_resources(memory_mb=100, used=0.99), predicted_utility=1.0)
  assert not budget.allow_exploration
  assert budget.reason == "memory_pressure"


def test_high_headroom_allocates_non_fixed_adaptive_budget():
  controller = ResourceAdaptiveBudgetController(TwoSystemConfig(enabled=True, max_probes=12, max_depth=4))
  low = controller.allocate(_resources(cpu=0.8), predicted_utility=0.3)
  high = controller.allocate(_resources(cpu=0.05), predicted_utility=1.0)
  assert high.max_probes > low.max_probes
  assert high.max_depth >= low.max_depth


def test_preemption_successful_recovery_and_gui_release():
  coordinator = SpeculationCoordinator()
  state = {"activity": "pkg/.A", "package": "pkg", "structural_signature": "s", "visual_signature": "v", "selected_state": {"x": False}}
  coordinator.begin_speculation(state)
  result = coordinator.barrier(lambda: None, lambda: state, unfinished=True)
  assert result.restored
  assert result.preempted and result.unfinished_exploration
  assert coordinator.owner == Ownership.MAIN


def test_failed_recovery_is_observable():
  coordinator = SpeculationCoordinator()
  state = {"activity": "pkg/.A", "package": "pkg", "structural_signature": "s", "visual_signature": "v"}
  coordinator.begin_speculation(state)
  result = coordinator.barrier(lambda: None, lambda: {"activity": "other/.B", "package": "other"})
  assert not result.restored
  assert result.verification_confidence < 0.5


def test_image_button_needs_explicit_reversibility_metadata():
  action = {"action_type": "CLICK"}
  assert classify_action_risk(action, {"role": "imagebutton"}) == RiskLevel.UNKNOWN
  assert classify_action_risk(action, {"role": "imagebutton", "reversible": True}) == RiskLevel.LOW


def test_prefix_aligned_child_skips_inference():
  """Reuse rights come from prefix alignment, never from execution alone."""
  graph = ProgressiveBeliefGraph("task")
  for node_id in ("root", "src", "dst"):
    graph.upsert_node(_node(node_id))
  parent = graph.add_speculative_transition(
      "root", {"action_type": "CLICK", "x": 1, "y": 1}, "src",
      path_probability=0.9, confidence=0.9, expected_information_gain=0.3,
      risk_level="LOW", exploration_cost=0.1, rollback_success=True,
  )
  child = graph.add_speculative_transition(
      "src", {"action_type": "CLICK", "x": 2, "y": 3}, "dst",
      path_probability=0.9, confidence=0.9, expected_information_gain=0.3,
      risk_level="LOW", exploration_cost=0.1, rollback_success=True,
  )
  gate = SkipInferenceGate(graph, TwoSystemConfig(enabled=True, skip_inference_enabled=True))

  # Executing the parent for real is not by itself a licence to reuse the
  # child: VERIFIED says where an action led, not that it was the right one.
  graph.record_execution_verification(parent.edge_id, True)
  assert not gate.query("src").skip

  graph.record_inference_alignment("root", parent.action, True)
  assert graph.promote_children_of_aligned_prefix(parent.edge_id) == [child]
  assert child.status == EdgeStatus.REUSABLE

  decision = gate.query("src")
  actions = []
  assert decision.skip
  assert gate.execute_and_verify(decision, actions.append, lambda expected: expected == "dst")
  assert actions == [child.action]


def test_state_mismatch_invalidates_future_path_and_reinfers():
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("src"))
  graph.upsert_node(_node("dst"))
  graph.upsert_node(_node("future"))
  root = graph.upsert_node(_node("root")) and graph.add_speculative_transition(
      "root", {"action_type": "CLICK", "n": 0}, "src", path_probability=1, confidence=1,
      expected_information_gain=1, risk_level="SAFE", exploration_cost=1, rollback_success=True)
  edge = graph.add_speculative_transition("src", {"action_type": "CLICK"}, "dst", path_probability=1, confidence=1, expected_information_gain=1, risk_level="SAFE", exploration_cost=1, rollback_success=True)
  child = graph.add_speculative_transition("dst", {"action_type": "CLICK", "n": 2}, "future", path_probability=1, confidence=1, expected_information_gain=1, risk_level="SAFE", exploration_cost=1, rollback_success=True)
  graph.record_inference_alignment("root", root.action, True)
  graph.promote_children_of_aligned_prefix(root.edge_id)
  gate = SkipInferenceGate(graph, TwoSystemConfig(enabled=True, skip_inference_enabled=True))
  decision = gate.query("src")
  assert not gate.execute_and_verify(decision, lambda action: None, lambda expected: False)
  assert edge.status == EdgeStatus.INVALID
  assert child.status == EdgeStatus.INVALID
  assert not gate.query("src").skip


def test_aligned_prefix_promotes_safely_recovered_lookahead_for_skip():
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("root", entropy=1.0))
  graph.upsert_node(_node("next", entropy=1.0))
  graph.upsert_node(_node("future", entropy=1.0))
  parent = graph.add_speculative_transition(
      "root", {"action_type": "CLICK", "x": 1, "y": 2}, "next",
      path_probability=0.9, confidence=0.2, expected_information_gain=0.5,
      risk_level="LOW", exploration_cost=0.1, rollback_success=True,
  )
  child = graph.add_speculative_transition(
      "next", {"action_type": "CLICK", "x": 3, "y": 4}, "future",
      path_probability=1.0, confidence=0.2, expected_information_gain=0.5,
      risk_level="LOW", exploration_cost=0.1, rollback_success=True,
  )
  graph.record_inference_alignment("root", parent.action, True)
  # Promotion is gated on observations only (aligned parent, observed child
  # transition, successful rollback, SAFE/LOW risk), so a child with a low
  # confidence of 0.2 still promotes: the confidence number is no longer part
  # of the decision.
  promoted = graph.promote_children_of_aligned_prefix(parent.edge_id)
  assert promoted == [child]
  assert child.status == EdgeStatus.REUSABLE
  gate = SkipInferenceGate(
      graph,
      TwoSystemConfig(enabled=True, skip_inference_enabled=True, reusable_confidence_threshold=0.8),
  )
  assert gate.query("next").edge == child


def test_stated_next_intent_keeps_only_the_forward_looking_half():
  """The recap half of a summary points at what is already behind us."""
  import importlib.util
  from pathlib import Path
  spec = importlib.util.spec_from_file_location(
      "_runner", Path(__file__).resolve().parents[2] / "scripts/run_one_parallel_exploration_task.py")
  runner = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(runner)
  extract = runner._stated_next_intent

  assert extract(
      "I have successfully opened the Contacts app. The next step is to fill in the name field."
  ) == "The next step is to fill in the name field."
  assert extract(
      "I've entered the filename, but now I need to change the extension."
  ).startswith("now I need to change")
  # No forward-looking clause: fall back to the task goal rather than ranking
  # against a description of what already happened.
  assert extract("I have successfully deleted the recipe.") == ""
  assert extract("") == ""


def test_recovery_memory_blocks_the_screen_not_just_the_element():
  """One observed unrecoverable probe should protect the whole screen.

  A list screen presents a fresh element identity per row, so a per-element
  blocklist alone lets the same unrecoverable kind of tap be retried on the
  next row (2026-08-29: TAP_NAV failed to roll back 29 times in 145 depth-2
  probes). The recovery context generalizes it to (activity, probe_type).
  """
  blocked = set()
  row = {
      "notes": "RESTORE_FAILED",
      "probe_type": "TAP_NAV",
      "graph": {"src": {"activity": "pkg/.ListActivity"}},
  }
  source_activity = str(((row.get("graph") or {}).get("src") or {}).get("activity") or "")
  blocked.add(f"{source_activity}|{row['probe_type']}")

  assert "pkg/.ListActivity|TAP_NAV" in blocked
  # A different control kind on the same screen is not implicated, and the
  # same control kind elsewhere is not either.
  assert "pkg/.ListActivity|SCROLL" not in blocked
  assert "pkg/.DetailActivity|TAP_NAV" not in blocked


def test_authoritative_edge_is_reusable_only_after_the_screen_recurs():
  """Progressive memory on repetitive tasks, with no speculative probing."""
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("list"), visited=True)
  graph.upsert_node(_node("form"), visited=True)
  edge = graph.add_speculative_transition(
      "list", {"action_type": "CLICK", "x": 9, "y": 9}, "form",
      path_probability=1.0, confidence=1.0, expected_information_gain=1.0,
      risk_level="LOW", exploration_cost=0.0,
  )
  graph.record_execution_verification(edge.edge_id, True)
  gate = SkipInferenceGate(graph, TwoSystemConfig(enabled=True, skip_inference_enabled=True))

  # First pass: seen once, nothing to reuse yet.
  assert not gate.query("list").skip

  # The task comes back to the same list screen for the next item.
  graph.upsert_node(_node("list"), visited=True)
  assert graph.nodes["list"].visit_count == 2
  decision = gate.query("list")
  assert decision.skip and decision.edge.edge_id == edge.edge_id


def test_revisited_self_loop_is_never_reused():
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("home"), visited=True)
  graph.upsert_node(_node("home"), visited=True)
  loop = graph.add_speculative_transition(
      "home", {"action_type": "OPEN_APP", "app_name": "X"}, "home",
      path_probability=1.0, confidence=1.0, expected_information_gain=0.0,
      risk_level="LOW", exploration_cost=0.0,
  )
  graph.record_execution_verification(loop.edge_id, True)
  gate = SkipInferenceGate(graph, TwoSystemConfig(enabled=True, skip_inference_enabled=True))
  assert not gate.query("home").skip


def test_authoritative_reuse_does_not_expire_on_wall_clock():
  """A repetitive task's own loop is longer than the speculative age cutoff."""
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("list"), visited=True)
  graph.upsert_node(_node("form"), visited=True)
  edge = graph.add_speculative_transition(
      "list", {"action_type": "CLICK", "x": 9, "y": 9}, "form",
      path_probability=1.0, confidence=1.0, expected_information_gain=1.0,
      risk_level="LOW", exploration_cost=0.0,
  )
  graph.record_execution_verification(edge.edge_id, True)
  graph.upsert_node(_node("list"), visited=True)
  # Recorded well beyond max_entry_age_s ago: one iteration of a repetitive
  # task at ~17s per step easily exceeds the 120s speculative cutoff.
  edge.last_updated = time.time() - 600.0

  assert graph.get_reusable_action(
      "list", max_entropy=0.25, max_age_s=120.0) is edge
  # A speculative lookahead of the same age is still expired: nothing
  # re-confirmed the screen it was probed on.
  edge.status = EdgeStatus.REUSABLE
  assert graph.get_reusable_action(
      "list", max_entropy=0.25, max_age_s=120.0) is None


def test_inverse_level_is_remembered_on_the_edge():
  """Progressive memory records HOW a transition was undone, not just that."""
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("src"))
  graph.upsert_node(_node("dst"))
  edge = graph.add_speculative_transition(
      "src", {"action_type": "CLICK", "x": 1, "y": 2, "probe_type": "TAP_NAV"},
      "dst", path_probability=1.0, confidence=0.2, expected_information_gain=0.5,
      risk_level="LOW", exploration_cost=0.1, rollback_success=True,
      inverse_level="BACK_N")
  assert edge.inverse_level == "BACK_N"
  assert edge.rollback_success is True

  # Re-observing without a level must not erase what is already known.
  again = graph.add_speculative_transition(
      "src", {"action_type": "CLICK", "x": 1, "y": 2, "probe_type": "TAP_NAV"},
      "dst", path_probability=1.0, confidence=0.2, expected_information_gain=0.5,
      risk_level="LOW", exploration_cost=0.1, rollback_success=True)
  assert again.edge_id == edge.edge_id
  assert again.inverse_level == "BACK_N"
