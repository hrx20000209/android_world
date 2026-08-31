"""The step boundary that keeps the serial runner honest."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from android_world.parallel_exploration.belief_graph import EdgeStatus
from android_world.parallel_exploration.belief_graph import GraphNode
from android_world.parallel_exploration.belief_graph import NodeStatus
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
from scripts.run_serial_exploration_task import GenerationError
from scripts.run_serial_exploration_task import GenerationGuardedGraph


def _node(node_id: str) -> GraphNode:
  return GraphNode(node_id, "pkg/.Main", "pkg", "v", node_id,
                   layout_signature=node_id, status=NodeStatus.COMMITTED,
                   decision_entropy=0.0)


def _graph_with_verified_edge():
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("list"), visited=True)
  graph.upsert_node(_node("form"), visited=True)
  edge = graph.add_speculative_transition(
      "list", {"action_type": "CLICK", "x": 1, "y": 2}, "form",
      path_probability=1.0, confidence=1.0, expected_information_gain=1.0,
      risk_level="LOW", exploration_cost=0.0)
  graph.record_execution_verification(edge.edge_id, True)
  return graph, edge


def test_exploration_from_this_step_is_invisible_to_this_step():
  """The whole point: step i cannot see what step i explored."""
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("list"), visited=True)
  graph.upsert_node(_node("list"), visited=True)
  guarded = GenerationGuardedGraph(graph)

  # Exploration during step 0 writes to the live graph.
  edge = graph.add_speculative_transition(
      "list", {"action_type": "CLICK", "x": 1, "y": 2}, "form",
      path_probability=1.0, confidence=1.0, expected_information_gain=1.0,
      risk_level="LOW", exploration_cost=0.0)
  graph.record_execution_verification(edge.edge_id, True)

  # Step 0 still sees nothing: in the parallel design this probe would not
  # even have finished when step 0's inference started.
  assert guarded.snapshot(for_step=0).reusable_edge("list") is None

  guarded.commit_step()
  assert guarded.snapshot(for_step=1).reusable_edge("list") is not None


def test_reading_a_newer_generation_raises_instead_of_leaking():
  graph, _ = _graph_with_verified_edge()
  guarded = GenerationGuardedGraph(graph)
  guarded.commit_step()
  with pytest.raises(GenerationError):
    guarded.snapshot(for_step=0)


def test_snapshot_is_not_a_live_view_of_the_graph():
  """A snapshot handed out earlier must not gain rows later."""
  graph, _ = _graph_with_verified_edge()
  guarded = GenerationGuardedGraph(graph)
  guarded.commit_step()
  snap = guarded.snapshot(for_step=1)
  before = len(snap.edges)
  graph.add_speculative_transition(
      "list", {"action_type": "CLICK", "x": 9, "y": 9}, "other",
      path_probability=1.0, confidence=1.0, expected_information_gain=1.0,
      risk_level="LOW", exploration_cost=0.0)
  assert len(snap.edges) == before


def test_unvisited_screen_offers_nothing_even_with_a_verified_edge():
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("list"), visited=True)   # seen once only
  graph.upsert_node(_node("form"), visited=True)
  edge = graph.add_speculative_transition(
      "list", {"action_type": "CLICK", "x": 1, "y": 2}, "form",
      path_probability=1.0, confidence=1.0, expected_information_gain=1.0,
      risk_level="LOW", exploration_cost=0.0)
  graph.record_execution_verification(edge.edge_id, True)
  guarded = GenerationGuardedGraph(graph)
  guarded.commit_step()
  assert guarded.snapshot(for_step=1).reusable_edge("list") is None


def _explored_edge(graph, src, dst, labels, x=100, y=200, ident="cls|id|Add|"):
  edge = graph.add_speculative_transition(
      src, {"action_type": "CLICK", "x": x, "y": y, "element_identity": ident},
      dst, path_probability=1.0, confidence=0.5, expected_information_gain=1.0,
      risk_level="LOW", exploration_cost=0.1, rollback_success=True,
      discovered_labels=labels)
  return edge


def test_briefing_states_what_was_seen_and_separates_what_was_used():
  """A briefing must read as observation, not as a suggestion to re-click.

  Phrasing everything as "tapping X opens Y" made the model tap the same
  control again and loop (MarkorCreateFolder, 2026-08-30), so controls
  already used are called out as such instead of being offered again.
  """
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("home"), visited=True)
  graph.upsert_node(_node("folder"), visited=True)
  graph.upsert_node(_node("file"), visited=True)
  used = _explored_edge(graph, "home", "folder", ("New folder", "Cancel"),
                        ident="cls|id/add|Add|")
  fresh = _explored_edge(graph, "home", "file", ("New file", "Save"),
                         x=300, ident="cls|id/new|New|")
  guarded = GenerationGuardedGraph(graph)
  guarded.commit_step()
  snap = guarded.snapshot(for_step=1)

  text = snap.screen_briefing("home", taken={used.edge_id})
  assert "New file" in text and "Save" in text
  assert "Already used on this screen" in text
  # The used control appears only under the "already used" heading.
  head, _, tail = text.partition("Already used on this screen")
  assert "id/add" not in head
  assert "id/add" in tail


def test_briefing_is_empty_without_explored_edges():
  """Authoritative-only graphs carry no lookahead labels to report."""
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("home"), visited=True)
  guarded = GenerationGuardedGraph(graph)
  guarded.commit_step()
  assert guarded.snapshot(for_step=1).screen_briefing("home", set()) == ""


def test_briefing_never_shows_the_current_steps_exploration():
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("home"), visited=True)
  graph.upsert_node(_node("folder"), visited=True)
  guarded = GenerationGuardedGraph(graph)
  # Explored during step 0, so step 0's own prompt must not mention it.
  _explored_edge(graph, "home", "folder", ("New folder",))
  assert guarded.snapshot(for_step=0).screen_briefing("home", set()) == ""
  guarded.commit_step()
  assert "New folder" in guarded.snapshot(for_step=1).screen_briefing("home", set())


def test_aligned_lookahead_is_reusable_on_a_screen_never_visited_twice():
  """The case that gets ahead of the authoritative path.

  A probe guessed action a at S; the model's real action then landed on the
  same successor, so the explorer read the intent correctly there. The hop it
  explored BEYOND that successor may now be executed without inference - on a
  screen the agent is standing on for the first time, which is exactly what
  replaying the model's own past decisions can never do.
  """
  graph = ProgressiveBeliefGraph("task")
  for name in ("home", "list", "detail"):
    graph.upsert_node(_node(name), visited=True)
  parent = graph.add_speculative_transition(
      "home", {"action_type": "CLICK", "x": 1, "y": 1}, "list",
      path_probability=0.9, confidence=0.2, expected_information_gain=0.5,
      risk_level="LOW", exploration_cost=0.1, rollback_success=True)
  child = graph.add_speculative_transition(
      "list", {"action_type": "CLICK", "x": 2, "y": 2}, "detail",
      path_probability=0.9, confidence=0.2, expected_information_gain=0.5,
      risk_level="LOW", exploration_cost=0.1, rollback_success=True)
  guarded = GenerationGuardedGraph(graph)
  guarded.commit_step()

  # "list" has been stood on once, so replay alone offers nothing.
  assert graph.nodes["list"].visit_count == 1
  assert guarded.snapshot(for_step=1).reusable_edge("list") is None

  # The model's real action landed on "list", confirming the probe's guess.
  graph.record_inference_alignment("home", parent.action, True)
  assert graph.promote_children_of_aligned_prefix(parent.edge_id) == [child]
  guarded.commit_step()

  offered = guarded.snapshot(for_step=2).reusable_edge("list")
  assert offered is not None and offered["edge_id"] == child.edge_id


def test_promoted_lookahead_is_preferred_over_a_replay():
  """Only the lookahead can save an inference the model has not made."""
  graph = ProgressiveBeliefGraph("task")
  for name in ("list", "old", "new"):
    graph.upsert_node(_node(name), visited=True)
  graph.upsert_node(_node("list"), visited=True)   # revisited
  replay = graph.add_speculative_transition(
      "list", {"action_type": "CLICK", "x": 5, "y": 5}, "old",
      path_probability=1.0, confidence=1.0, expected_information_gain=0.0,
      risk_level="LOW", exploration_cost=0.0)
  graph.record_execution_verification(replay.edge_id, True)
  lookahead = graph.add_speculative_transition(
      "list", {"action_type": "CLICK", "x": 7, "y": 7}, "new",
      path_probability=0.5, confidence=0.2, expected_information_gain=1.0,
      risk_level="LOW", exploration_cost=0.1, rollback_success=True)
  lookahead.status = EdgeStatus.REUSABLE
  guarded = GenerationGuardedGraph(graph)
  guarded.commit_step()

  chosen = guarded.snapshot(for_step=1).reusable_edge("list")
  assert chosen["edge_id"] == lookahead.edge_id


def test_low_entropy_skip_needs_a_revisit_not_just_one_mapped_edge():
  """H = 0 from ignorance must not look like H = 0 from determinism.

  A screen with fifteen controls where exploration managed to map exactly one
  has the same measured entropy as a confirmation dialog that really offers
  one continuation. Only repeated visits, during which the explorer probes
  controls it has not tried here before, separate the two.
  """
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(_node("screen"), visited=True)
  graph.upsert_node(_node("next"), visited=True)
  only = graph.add_speculative_transition(
      "screen", {"action_type": "CLICK", "x": 1, "y": 1}, "next",
      path_probability=1.0, confidence=0.2, expected_information_gain=0.5,
      risk_level="LOW", exploration_cost=0.1, rollback_success=True)
  guarded = GenerationGuardedGraph(graph)
  guarded.commit_step()
  snap = guarded.snapshot(for_step=1)
  assert snap.node_entropy["screen"] == 0.0        # measured as decided...
  assert snap.reusable_edge("screen") is None      # ...but seen only once

  graph.upsert_node(_node("screen"), visited=True)  # second visit
  guarded.commit_step()
  offered = guarded.snapshot(for_step=2).reusable_edge("screen")
  assert offered is not None and offered["edge_id"] == only.edge_id


def test_a_screen_with_alternatives_is_never_low_entropy_skipped():
  graph = ProgressiveBeliefGraph("task")
  for name in ("screen", "a", "b"):
    graph.upsert_node(_node(name), visited=True)
  graph.upsert_node(_node("screen"), visited=True)   # revisited
  for i, dst in enumerate(("a", "b")):
    graph.add_speculative_transition(
        "screen", {"action_type": "CLICK", "x": i, "y": i}, dst,
        path_probability=1.0, confidence=0.2, expected_information_gain=0.5,
        risk_level="LOW", exploration_cost=0.1, rollback_success=True)
  guarded = GenerationGuardedGraph(graph)
  guarded.commit_step()
  snap = guarded.snapshot(for_step=1)
  assert snap.node_entropy["screen"] > 0.6          # ~ln 2, a real choice
  assert snap.reusable_edge("screen") is None       # so the model must decide


def test_skip_refuses_to_walk_back_into_the_recent_path():
  """A remembered edge is only progress if it leads somewhere new.

  Every hop of a cycle verifies correctly on its own, so nothing downstream
  notices the trajectory is going round: CameraTakeVideo took 13 skips on
  2026-08-31, all landing exactly where predicted, and still exhausted its
  budget without progressing.
  """
  graph = ProgressiveBeliefGraph("task")
  for name in ("a", "b"):
    graph.upsert_node(_node(name), visited=True)
  graph.upsert_node(_node("a"), visited=True)   # revisited
  back = graph.add_speculative_transition(
      "a", {"action_type": "CLICK", "x": 1, "y": 1}, "b",
      path_probability=1.0, confidence=1.0, expected_information_gain=0.0,
      risk_level="LOW", exploration_cost=0.0)
  graph.record_execution_verification(back.edge_id, True)
  guarded = GenerationGuardedGraph(graph)
  guarded.commit_step()
  snap = guarded.snapshot(for_step=1)

  assert snap.reusable_edge("a") is not None            # nothing known yet
  assert snap.reusable_edge("a", recent=("b",)) is None  # b is where we came from
