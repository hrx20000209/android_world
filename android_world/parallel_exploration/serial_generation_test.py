"""The step boundary that keeps the serial runner honest."""

from __future__ import annotations

import sys
from pathlib import Path

import json
import math
import re
import statistics

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from android_world.parallel_exploration.belief_graph import EdgeStatus
from android_world.parallel_exploration.belief_graph import GraphNode
from android_world.parallel_exploration.belief_graph import NodeStatus
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
from scripts.run_serial_exploration_task import GenerationError
from scripts.run_serial_exploration_task import GenerationGuardedGraph
import pathlib


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
  # Twice: one execution says the transition worked here, two say it keeps
  # being the answer here, and only the second licenses a replay.
  graph.record_execution_verification(edge.edge_id, True)
  return graph, edge


def test_exploration_from_this_step_is_invisible_to_this_step():
  """The whole point: step i cannot see what step i explored."""
  graph = ProgressiveBeliefGraph("task")
  for _ in range(3):
    graph.upsert_node(_node("list"), visited=True)
  guarded = GenerationGuardedGraph(graph)

  # Exploration during step 0 writes to the live graph.
  edge = graph.add_speculative_transition(
      "list", {"action_type": "CLICK", "x": 1, "y": 2}, "form",
      path_probability=1.0, confidence=1.0, expected_information_gain=1.0,
      risk_level="LOW", exploration_cost=0.0)
  graph.record_execution_verification(edge.edge_id, True)
  # Twice: one execution says the transition worked here, two say it keeps
  # being the answer here, and only the second licenses a replay.
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
  # Twice: one execution says the transition worked here, two say it keeps
  # being the answer here, and only the second licenses a replay.
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
  graph.upsert_node(_node("screen"), visited=True)  # third
  guarded.commit_step()
  # Still refused: a revisit shows the screen was mapped, not that the single
  # mapped transition is the one to take. Separating skip outcomes by source
  # on 2026-08-31 measured graph-based skips at 3/17 against 80/80 for the
  # launch collapse, and every step regression against the control was a
  # mis-skip of this kind.
  assert guarded.snapshot(for_step=2).reusable_edge("screen") is None

  # The model itself having taken it is the evidence that was missing.
  graph.record_execution_verification(only.edge_id, True)
  graph.record_execution_verification(only.edge_id, True)
  guarded.commit_step()
  offered = guarded.snapshot(for_step=3).reusable_edge("screen")
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
  for _ in range(2):
    graph.upsert_node(_node("a"), visited=True)   # revisited, twice
  back = graph.add_speculative_transition(
      "a", {"action_type": "CLICK", "x": 1, "y": 1}, "b",
      path_probability=1.0, confidence=1.0, expected_information_gain=0.0,
      risk_level="LOW", exploration_cost=0.0)
  graph.record_execution_verification(back.edge_id, True)
  graph.record_execution_verification(back.edge_id, True)  # twice: replayable
  guarded = GenerationGuardedGraph(graph)
  guarded.commit_step()
  snap = guarded.snapshot(for_step=1)

  assert snap.reusable_edge("a") is not None            # nothing known yet
  # A promoted lookahead edge is still refused when it points back into the
  # recent path. The exemption is only for transitions the model itself has
  # executed repeatedly - see reusable_edge - so make this one speculative.
  lookahead = graph.add_speculative_transition(
      "a", {"action_type": "CLICK", "x": 9, "y": 9}, "b",
      path_probability=1.0, confidence=1.0, expected_information_gain=0.0,
      risk_level="LOW", exploration_cost=0.0)
  graph.edges[lookahead.edge_id].status = EdgeStatus.REUSABLE
  guarded.commit_step()
  assert guarded.snapshot(for_step=2).reusable_edge(
      "a", recent=("b",)) is None  # b is where we came from


def test_skip_needs_two_executions_not_one():
  """One execution says it worked here; two say it keeps being the answer.

  Separating 28 graph-based skips by their edge's execution history on
  2026-08-31: all 21 misses replayed an edge the model had executed exactly
  once, while every hit came from an edge it had executed at least twice.
  """
  graph = ProgressiveBeliefGraph("task")
  for name in ("screen", "next"):
    graph.upsert_node(_node(name), visited=True)
  graph.upsert_node(_node("screen"), visited=True)   # revisited
  graph.upsert_node(_node("screen"), visited=True)   # and again
  edge = graph.add_speculative_transition(
      "screen", {"action_type": "CLICK", "x": 1, "y": 1}, "next",
      path_probability=1.0, confidence=1.0, expected_information_gain=0.0,
      risk_level="LOW", exploration_cost=0.0)
  guarded = GenerationGuardedGraph(graph)

  graph.record_execution_verification(edge.edge_id, True)
  guarded.commit_step()
  assert guarded.snapshot(for_step=1).reusable_edge("screen") is None

  graph.record_execution_verification(edge.edge_id, True)
  guarded.commit_step()
  assert guarded.snapshot(for_step=2).reusable_edge("screen") is not None


def test_same_control_at_wobbling_coordinates_is_one_edge():
  """The model's tap coordinate moves between visits; the control does not.

  Measured on ExpenseAddMultiple (2026-09-01): the same button was pressed 11
  times as (540,1063) six times and (540,1068) five times, and the graph
  recorded two edges of six and five. The node was visited 33 times and its
  only recorded transition still read one execution.
  """
  graph = ProgressiveBeliefGraph("task")
  for name in ("screen", "next"):
    graph.upsert_node(_node(name), visited=True)
  key = "com.app:id/save|Save||android.widget.Button"
  first = graph.add_speculative_transition(
      "screen", {"action_type": "click", "x": 540, "y": 1063, "control_key": key},
      "next", path_probability=1.0, confidence=1.0,
      expected_information_gain=0.0, risk_level="LOW", exploration_cost=0.0)
  second = graph.add_speculative_transition(
      "screen", {"action_type": "click", "x": 540, "y": 1068, "control_key": key},
      "next", path_probability=1.0, confidence=1.0,
      expected_information_gain=0.0, risk_level="LOW", exploration_cost=0.0)
  assert first.edge_id == second.edge_id
  assert len(graph.edges) == 1


def test_a_probe_and_the_model_taking_it_are_the_same_edge():
  """Probe records CLICK, the agent records click; one control, one edge."""
  graph = ProgressiveBeliefGraph("task")
  for name in ("screen", "next"):
    graph.upsert_node(_node(name), visited=True)
  key = "com.app:id/details|Details||android.widget.TextView"
  probed = graph.add_speculative_transition(
      "screen", {"action_type": "CLICK", "x": 100, "y": 200, "control_key": key,
                 "probe_type": "TAP_NAV"},
      "next", path_probability=0.5, confidence=0.2,
      expected_information_gain=0.5, risk_level="LOW", exploration_cost=1.0)
  taken = graph.add_speculative_transition(
      "screen", {"action_type": "click", "x": 103, "y": 198, "control_key": key},
      "next", path_probability=1.0, confidence=1.0,
      expected_information_gain=0.0, risk_level="LOW", exploration_cost=0.0)
  assert probed.edge_id == taken.edge_id


def test_actions_without_a_control_keep_their_own_identity():
  """No control resolved: fall back to the whole action, as before."""
  graph = ProgressiveBeliefGraph("task")
  for name in ("screen", "next"):
    graph.upsert_node(_node(name), visited=True)
  a = graph.add_speculative_transition(
      "screen", {"action_type": "input_text", "text": "hello"}, "next",
      path_probability=1.0, confidence=1.0, expected_information_gain=0.0,
      risk_level="LOW", exploration_cost=0.0)
  b = graph.add_speculative_transition(
      "screen", {"action_type": "input_text", "text": "world"}, "next",
      path_probability=1.0, confidence=1.0, expected_information_gain=0.0,
      risk_level="LOW", exploration_cost=0.0)
  assert a.edge_id != b.edge_id


def test_tap_resolves_to_the_smallest_control_containing_it():
  """Containers enclose their children; the child is what was pressed."""
  import types
  from scripts.run_serial_exploration_task import _control_key_at
  from android_world.parallel_exploration.rankers import UiElement
  container = UiElement(resource_id="com.app:id/row", class_name="android.widget.LinearLayout",
                        bounds=(0, 1000, 1080, 1200))
  button = UiElement(resource_id="com.app:id/save", text="Save",
                     class_name="android.widget.Button", bounds=(500, 1040, 600, 1090))
  state = types.SimpleNamespace(elements=(container, button))
  assert _control_key_at(state, {"x": 540, "y": 1063}) == \
      _control_key_at(state, {"x": 540, "y": 1068})
  assert "Save" in _control_key_at(state, {"x": 540, "y": 1063})
  # Outside every element, and actions with no coordinate, resolve to nothing.
  assert _control_key_at(state, {"x": 5, "y": 5}) == ""
  assert _control_key_at(state, {"action_type": "input_text", "text": "x"}) == ""


def test_unnamed_controls_are_separated_by_position():
  """A bare widget class is not an identity; two of them must not merge."""
  import types
  from scripts.run_serial_exploration_task import _control_key_at
  from android_world.parallel_exploration.rankers import UiElement
  left_box = UiElement(class_name="android.widget.RelativeLayout", bounds=(0, 0, 200, 200))
  right_box = UiElement(class_name="android.widget.RelativeLayout", bounds=(800, 0, 1000, 200))
  state = types.SimpleNamespace(elements=(left_box, right_box))
  a = _control_key_at(state, {"x": 100, "y": 100})
  b = _control_key_at(state, {"x": 900, "y": 100})
  assert a and b and a != b
  # ...while a few pixels of wobble on the same one still resolves the same.
  assert a == _control_key_at(state, {"x": 106, "y": 94})


def test_an_edge_is_not_replayed_more_than_twice():
  """A learned loop replays perfectly and still goes nowhere.

  SimpleSmsReplyMostRecent (2026-09-01): the model looped between two screens,
  the graph learned the loop, and skipping replayed it twelve times with every
  hop matching its predicted destination while the episode made no progress.
  """
  graph = ProgressiveBeliefGraph("task")
  for name in ("a", "b"):
    graph.upsert_node(_node(name), visited=True)
  graph.upsert_node(_node("a"), visited=True)
  edge = graph.add_speculative_transition(
      "a", {"action_type": "CLICK", "x": 1, "y": 1}, "b",
      path_probability=1.0, confidence=1.0, expected_information_gain=0.0,
      risk_level="LOW", exploration_cost=0.0)
  for _ in range(2):
    graph.record_execution_verification(edge.edge_id, True)
  guarded = GenerationGuardedGraph(graph)
  guarded.commit_step()
  assert guarded.snapshot(for_step=1).reusable_edge("a") is not None

  graph.record_skip_result(edge.edge_id, True)
  guarded.commit_step()
  assert guarded.snapshot(for_step=2).reusable_edge("a") is not None

  graph.record_skip_result(edge.edge_id, True)
  guarded.commit_step()
  assert guarded.snapshot(for_step=3).reusable_edge("a") is None


def test_replay_requires_the_task_to_have_moved():
  """Repeating an action is only evidence when the task iterated meanwhile.

  Two full runs of identical code (2026-09-01): on tasks where skipping fired
  in both, this arm scored 6 and 6 against a control's 13, and every task that
  saw a mis-skip was lost twice over. Iterating discovers screens - a list with
  one fewer item is a different screen - while spinning discovers nothing.
  """
  graph = ProgressiveBeliefGraph("task")
  for name in ("a", "b"):
    graph.upsert_node(_node(name), visited=True)
  graph.upsert_node(_node("a"), visited=True)
  edge = graph.add_speculative_transition(
      "a", {"action_type": "CLICK", "x": 1, "y": 1}, "b",
      path_probability=1.0, confidence=1.0, expected_information_gain=0.0,
      risk_level="LOW", exploration_cost=0.0)
  for _ in range(2):
    graph.record_execution_verification(edge.edge_id, True)
  guarded = GenerationGuardedGraph(graph)

  # Spinning: the graph knew these two screens when the edge last ran, and
  # still knows only those two.
  edge.nodes_at_last_execution = len(graph.nodes)
  guarded.commit_step()
  assert guarded.snapshot(for_step=1).reusable_edge("a") is None

  # Iterating: the trajectory has since reached a screen never seen before.
  graph.upsert_node(_node("c"), visited=True)
  guarded.commit_step()
  assert guarded.snapshot(for_step=2).reusable_edge("a") is not None


def test_skipping_stops_after_two_and_after_any_miss():
  """Volume is the only signal that survived: 5/68 at one or two skips,
  1/32 at three or four, 0/9 at five or more (five runs, 2026-09-01)."""
  def allowed(attempted, correct, consecutive):
    return consecutive < 3 and correct == attempted and attempted < 2
  assert allowed(0, 0, 0)          # first is free
  assert allowed(1, 1, 1)          # second, still perfect
  assert not allowed(2, 2, 2)      # third refused even when both landed
  assert not allowed(1, 0, 1)      # a single miss ends it


def test_settings_screens_are_never_probed():
  """A flipped switch is not navigation, so no rung of the ladder puts it back.

  Settings is also where the tasks that read those switches live, so a probe
  there can invalidate the very state being checked rather than merely risking
  the episode.
  """
  from scripts.run_serial_exploration_task import _unprobeable
  for package in ("com.android.settings", "com.google.android.settings",
                  "com.android.systemui"):
    assert _unprobeable(package)
  # The launcher is not an app the task works in, and it is the surface the
  # model navigates by swiping - a scroll probe there moves the icon it is
  # hunting for.
  for package in ("com.google.android.apps.nexuslauncher",
                  "com.android.launcher3"):
    assert _unprobeable(package)
  for package in ("com.flauschcode.broccoli", "net.gsantner.markor",
                  "com.arduia.expense"):
    assert not _unprobeable(package)


def _runner_source() -> str:
  return pathlib.Path(
      "scripts/run_serial_exploration_task.py").read_text(encoding="utf-8")


def test_explore_block_comes_before_inference_call():
  """Exploration must start from the screen the window opened on.

  GELABAgent.step() executes the action it chooses, so calling it before the
  explorer put every probe on the NEXT screen. The prefix check asks whether a
  probe's destination equals the screen the model then landed on, which for
  probes taken from that next screen can never hold - prefix alignment was 0
  in all five full runs (2026-09-04). Locked structurally because the failure
  was silent: everything ran, nothing aligned.
  """
  src = _runner_source()
  explore = src.index("# (3) Exploration from S_i, BEFORE inference")
  run_probe = src.index("outcome = run_serial_exploration(")
  inference = src.index("result = original_step(self, goal)")
  record = src.index("# (5) Record the authoritative transition")
  assert explore < run_probe < inference < record


def test_explorer_is_not_given_this_step_output():
  """The only need handed to the explorer comes from the PREVIOUS window."""
  src = _runner_source()
  block = src[src.index("# (3) Exploration from S_i"):
              src.index("result = original_step(self, goal)")]
  assert "fresh_need=current_need.to_dict()" in block
  assert "_responses" not in block
  assert "result.data" not in block


def _load_runner_helpers():
  """Exec just the pure helpers, so the test needs no device or agent."""
  src = pathlib.Path(
      "scripts/run_serial_exploration_task.py").read_text(encoding="utf-8")
  ns = {"Any": object, "Mapping": dict}
  ns.update(__import__("typing").__dict__)
  # typing exports a deprecated `typing.re` alias that shadows the real module,
  # so any helper in this slice calling re.compile got
  # "type object 'typing.re' has no attribute 'compile'". Restore the modules
  # the runner actually imports, after the typing splat.
  ns.update({"re": re, "math": math, "json": json, "statistics": statistics})
  start = src.index("def _clean_label")
  end = src.index("def _control_key_at")
  exec(compile(src[start:end], "helpers", "exec"), ns)  # pylint: disable=exec-used
  return ns


class _Snap:
  """Minimal stand-in for GraphSnapshot."""

  def __init__(self, edges, outgoing, visits):
    self.edges, self.outgoing, self.node_visits = edges, outgoing, visits


def test_screen_history_names_controls_the_model_can_find():
  """An unnamed control must be left out, not named by its widget class.

  The first version fell back to the class name and produced lines like
  "ImageButton -> a screen showing: Share, Rename" (2026-09-04) - true, and
  unusable, because there is no "ImageButton" to find on a screenshot.
  """
  ns = _load_runner_helpers()
  edges = {
      "e1": {"status": "VERIFIED", "dst_node": "n2",
             "discovered_labels": ["   Share  ", "Recording: %s"],
             "action": {"action_type": "click",
                        "control_key": "app:id/x|||android.widget.ImageButton"}},
      "e2": {"status": "VERIFIED", "dst_node": "n2",
             "discovered_labels": ["Sent", "Inbox"],
             "action": {"action_type": "click",
                        "control_key": "app:id/send|Send||android.widget.Button"}},
  }
  snap = _Snap(edges, {"n1": ["e1", "e2"]}, {"n1": 3, "n2": 2})
  text = ns["screen_history_context"](snap, "n1", {"e2"})
  assert "ImageButton" not in text
  assert "%s" not in text
  assert "Send -> a screen showing: Sent, Inbox" in text
  assert text.startswith("[Screen memory]")


def test_screen_history_separates_done_from_available():
  """Already-used controls are named as history, never as a suggestion."""
  ns = _load_runner_helpers()
  edges = {
      "e1": {"status": "VERIFIED", "dst_node": "n2",
             "discovered_labels": ["Sent"],
             "action": {"action_type": "click",
                        "control_key": "app:id/send|Send||android.widget.Button"}},
      "e2": {"status": "SPECULATIVE", "dst_node": "n9",
             "discovered_labels": ["Camera"],
             "action": {"action_type": "click",
                        "control_key": "app:id/at|Attach||android.widget.Button"}},
  }
  snap = _Snap(edges, {"n1": ["e1", "e2"]}, {"n1": 3})
  text = ns["screen_history_context"](snap, "n1", {"e1"})
  assert "Already used on this screen this visit: Send" in text
  assert "Also known to lead somewhere from this screen: Attach" in text


def test_screen_history_is_silent_when_the_graph_knows_nothing():
  ns = _load_runner_helpers()
  snap = _Snap({}, {}, {})
  assert ns["screen_history_context"](snap, "nX", set()) == ""


def _canon():
  src = _runner_source()
  ns = {"Mapping": dict, "Any": object}
  start = src.index("def _canonical_scroll")
  end = src.index("def _control_key_at")
  exec(compile(src[start:end], "canon", "exec"), ns)  # pylint: disable=exec-used
  return ns["_canonical_control"]


def test_same_control_matches_despite_coordinate_wobble():
  """Two records of one decision must compare equal.

  The model predicts coordinates in a normalised space and they move between
  identical choices - ExpenseAddMultiple pressed one button as (540,1063) six
  times and (540,1068) five times - which is why edge identity stopped using
  them and why alignment must not either.
  """
  canon = _canon()
  probe = {"action_type": "click", "control_key": "app:id/save|SAVE||Button",
           "x": 540, "y": 1063}
  model = {"action_type": "click", "control_key": "app:id/save|SAVE||Button",
           "x": 540, "y": 1068}
  assert canon(probe) == canon(model) != ""


def test_different_controls_do_not_match():
  canon = _canon()
  a = {"action_type": "click", "control_key": "app:id/save|SAVE||Button"}
  b = {"action_type": "click", "control_key": "app:id/cancel|Cancel||Button"}
  assert canon(a) != canon(b)


def test_action_without_control_key_falls_back_to_its_own_shape():
  canon = _canon()
  assert canon({"action_type": "input_text", "text": "hello"}) == "input_text|text=hello"
  assert canon({"action_type": "open_app", "app_name": "Files"}) == "open_app|app_name=Files"
  # A vertical swipe no longer keeps its raw direction: `swipe` and `scroll`
  # are inverses in AndroidWorld, so the raw string compared two opposite
  # gestures as equal. It normalises onto the content direction instead - see
  # the scroll/swipe tests below.
  assert canon({"action_type": "swipe", "direction": "up"}) == "scroll|forward"
  assert canon({}) == ""


def test_coordinates_alone_are_never_an_identity():
  """A tap with nothing but a coordinate names no control at all."""
  canon = _canon()
  assert canon({"action_type": "click", "x": 100, "y": 200}) == ""


# --- prefix alignment compares a probe edge against the model's own action ---
#
# Regression for 2026-09-09: the probe edge carries a control_key, the model
# reports only a coordinate, and _canonical_control reads control_key first.
# Every click therefore compared a real key against "" and aligned_by_action
# was structurally false - a broken comparator that read as a finding.


class _Elem:

  def __init__(self, bounds, identity, resource_id="", text="", content_desc=""):
    self.bounds = bounds
    self.identity = identity
    self.resource_id = resource_id
    self.text = text
    self.content_desc = content_desc


class _Screen:

  def __init__(self, elements):
    self.elements = elements


_DELETE = _Elem((900, 100, 1100, 300),
                "android.widget.TextView|app:id/action_delete|Delete||[900,100][1100,300]",
                resource_id="app:id/action_delete", text="Delete")
_SHARE = _Elem((600, 100, 800, 300),
               "android.widget.TextView|app:id/action_share|Share||[600,100][800,300]",
               resource_id="app:id/action_share", text="Share")


def test_bare_model_click_has_no_canonical_control():
  from scripts.run_serial_exploration_task import _canonical_control
  assert _canonical_control({"action_type": "click", "x": 1000, "y": 200}) == ""


def test_resolved_model_click_matches_the_probe_that_took_the_same_control():
  from scripts.run_serial_exploration_task import _canonical_control
  from scripts.run_serial_exploration_task import _control_key_at
  screen = _Screen([_DELETE, _SHARE])
  model = {"action_type": "click", "x": 1000, "y": 200}
  model["control_key"] = _control_key_at(screen, model)
  probe = {"action_type": "click",
           "control_key": _control_key_at(screen, {"action_type": "click",
                                                   "x": 1010, "y": 250})}
  assert model["control_key"]
  assert _canonical_control(model) == _canonical_control(probe)


def test_a_different_control_on_the_same_screen_does_not_match():
  from scripts.run_serial_exploration_task import _canonical_control
  from scripts.run_serial_exploration_task import _control_key_at
  screen = _Screen([_DELETE, _SHARE])
  model = {"action_type": "click", "x": 1000, "y": 200}
  model["control_key"] = _control_key_at(screen, model)
  probe = {"action_type": "click",
           "control_key": _control_key_at(screen, {"action_type": "click",
                                                   "x": 700, "y": 200})}
  assert _canonical_control(model) != _canonical_control(probe)


def test_tap_on_nothing_resolves_to_empty_rather_than_a_wrong_control():
  from scripts.run_serial_exploration_task import _control_key_at
  screen = _Screen([_DELETE])
  assert _control_key_at(screen, {"action_type": "click", "x": 10, "y": 10}) == ""


# --- scroll and swipe are inverses in AndroidWorld (actuation.py:153,173) ---
#
# Regression for 2026-09-09: a SCROLL probe records the scrollable container as
# its control_key and no direction; the model records a direction and no
# container. Comparing the raw strings never matched, and comparing the raw
# `direction` across the two verbs would have matched OPPOSITE gestures.


def test_probe_scroll_and_model_scroll_down_are_the_same_gesture():
  from scripts.run_serial_exploration_task import _canonical_control
  probe = {"action_type": "SWIPE", "probe_type": "SCROLL",
           "control_key": "android.widget.ScrollView", "x": 540, "y": 1306}
  model = {"action_type": "scroll", "direction": "down"}
  assert _canonical_control(probe) == _canonical_control(model)


def test_probe_scroll_and_model_swipe_down_are_opposite_gestures():
  from scripts.run_serial_exploration_task import _canonical_control
  probe = {"action_type": "SWIPE", "probe_type": "SCROLL",
           "control_key": "android.widget.ScrollView", "x": 540, "y": 1306}
  model = {"action_type": "swipe", "direction": "down"}
  assert _canonical_control(probe) != _canonical_control(model)


def test_swipe_up_reveals_the_same_content_as_scroll_down():
  from scripts.run_serial_exploration_task import _canonical_control
  assert (_canonical_control({"action_type": "swipe", "direction": "up"})
          == _canonical_control({"action_type": "scroll", "direction": "down"}))


def test_horizontal_scroll_cannot_collide_with_any_probe_scroll():
  from scripts.run_serial_exploration_task import _canonical_control
  # Horizontal is not normalised - the two verbs agree there and no probe ever
  # scrolls sideways, so it keeps the generic direction fallback. What must
  # hold is that it can never equal a probe's key, and that the two verbs stay
  # distinguishable rather than being silently merged.
  probe = _canonical_control({"action_type": "SWIPE", "probe_type": "SCROLL",
                              "control_key": "android.widget.ScrollView"})
  left_scroll = _canonical_control({"action_type": "scroll", "direction": "left"})
  left_swipe = _canonical_control({"action_type": "swipe", "direction": "left"})
  assert left_scroll and left_swipe
  assert left_scroll != probe and left_swipe != probe
  assert left_scroll != left_swipe


def test_a_scroll_never_matches_a_tap():
  from scripts.run_serial_exploration_task import _canonical_control
  probe = {"action_type": "SWIPE", "probe_type": "SCROLL",
           "control_key": "android.widget.ScrollView"}
  tap = {"action_type": "click", "control_key": "android.widget.ScrollView"}
  assert _canonical_control(probe) != _canonical_control(tap)


# --- nodes carry a readable description, paid for by the model already ------
#
# semantic_summary was populated on 0 of 1175 nodes and salient_ui_labels only
# by cross-task seeding, so a node was a hash and nothing else. At step i the
# model looks at S_i and its <THINK> opens by saying what it sees; reading that
# costs no extra call.


def test_screen_sentence_takes_the_models_first_observation():
  from scripts.run_serial_exploration_task import _screen_sentence
  out = ("<THINK> I see a 'Confirm Delete' dialog on the screen, asking if I "
         "really want to delete the document. I should press OK. </THINK>\n"
         "explain:press OK\taction:CLICK")
  assert _screen_sentence(out) == (
      "I see a 'Confirm Delete' dialog on the screen, asking if I really want "
      "to delete the document.")


def test_screen_sentence_falls_back_to_explain_when_there_is_no_think():
  from scripts.run_serial_exploration_task import _screen_sentence
  assert "file list" in _screen_sentence(
      "explain:The file list is showing. I will tap the first note.\taction:CLICK")


def test_screen_sentence_is_empty_when_the_model_emitted_no_prose():
  """A bare tool call must leave the field empty rather than store noise."""
  from scripts.run_serial_exploration_task import _screen_sentence
  assert _screen_sentence('{"action":"CLICK","point":"500,815"}') == ""
  assert _screen_sentence("") == ""
  assert _screen_sentence(None) == ""


class _El:

  def __init__(self, text="", desc="", clickable=True, scrollable=False):
    self.text, self.content_desc = text, desc
    self.clickable, self.scrollable, self.checked = clickable, scrollable, None


class _State:

  def __init__(self, elements):
    self.elements = elements


def test_actionable_labels_keeps_screen_order_and_drops_unnamed():
  from scripts.run_serial_exploration_task import _actionable_labels
  s = _State([_El("Files"), _El(desc="Search"), _El(), _El("Recent")])
  assert _actionable_labels(s) == ("Files", "Search", "Recent")


def test_actionable_labels_skips_non_interactive_and_overlong_text():
  from scripts.run_serial_exploration_task import _actionable_labels
  s = _State([_El("Heading", clickable=False), _El("x" * 60), _El("OK")])
  assert _actionable_labels(s) == ("OK",)


def test_actionable_labels_deduplicates_case_insensitively_and_caps():
  from scripts.run_serial_exploration_task import _actionable_labels
  s = _State([_El("Delete"), _El("delete"), _El("Share")])
  assert _actionable_labels(s) == ("Delete", "Share")
  assert len(_actionable_labels(_State([_El(f"row {i}") for i in range(30)]))) == 10


def test_progress_recaps_are_rejected_rather_than_stored_as_descriptions():
  """35% of <THINK> openings are progress or intent, not observation. Stored
  as a screen description they mislead every later task that lands there."""
  from scripts.run_serial_exploration_task import _screen_sentence
  for recap in (
      "<THINK> I have successfully navigated to the correct date, October 29th. "
      "Now I will tap the button. </THINK>",
      "<THINK> I need to turn on Bluetooth. The switch is on the right. </THINK>",
      "<THINK> The task is to delete the note. I will select it first. </THINK>"):
    assert _screen_sentence(recap) == ""


def test_observations_are_kept():
  from scripts.run_serial_exploration_task import _screen_sentence
  for good in (
      "<THINK> I see a 'Confirm Delete' dialog on the screen. Press OK. </THINK>",
      "<THINK> I am looking at the file list in the Markor app. Tap it. </THINK>",
      "<THINK> The main settings page is showing. Scroll down. </THINK>"):
    assert _screen_sentence(good) != ""


# --- naming the screens the episode left blank --------------------------------
#
# Measured on desc2 (2026-09-10): 16 of 18 sampled undescribed nodes could be
# described perfectly well from the labels already stored on them. They were
# blank because the online write point - after an inference on the screen the
# model reasoned about - is never reached for a node that is only landed on,
# skipped over, seeded, or reached during bootstrap. Replaying this pass over
# desc2's own graphs took coverage 64% -> 87% at 4.3 s per task, all of it
# after run.py has returned.

from android_world.parallel_exploration.belief_graph import GraphEdge
from android_world.parallel_exploration.belief_graph import GraphNode
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
from scripts.run_serial_exploration_task import _backfill_descriptions
from scripts.run_serial_exploration_task import _incoming_labels


class _Describer:
  """Stands in for the 0.6B: echoes the labels it was actually given."""

  def __init__(self, refuse=()):
    self.calls = []
    self._refuse = set(refuse)

  def __call__(self, port, task, activity, labels, timeout_s=8.0):
    self.calls.append((activity, tuple(labels)))
    if activity in self._refuse or not labels:
      return ""
    return "A screen with " + ", ".join(labels) + "."


def _graph_with(*nodes, edges=()):
  graph = ProgressiveBeliefGraph("task")
  for node in nodes:
    graph.upsert_node(node, visited=node.visit_count > 0)
  for edge in edges:
    graph.edges[edge.edge_id] = edge
  return graph


def _node(node_id, labels=(), summary="", visits=1):
  return GraphNode(node_id=node_id, activity=f"com.app/.{node_id}",
                   package="com.app", visual_signature="",
                   structural_signature="", layout_signature=f"L-{node_id}",
                   salient_ui_labels=tuple(labels), semantic_summary=summary,
                   visit_count=visits)


def _install(monkeypatch, describer):
  from android_world.parallel_exploration import semantic_service
  monkeypatch.setattr(semantic_service, "query_describe", describer)


def test_a_node_left_blank_online_is_named_after_the_episode(monkeypatch):
  describer = _Describer()
  _install(monkeypatch, describer)
  graph = _graph_with(_node("Landing", labels=["Save", "Cancel"]))
  assert _backfill_descriptions(graph, None, 8766) == 1
  assert (graph.nodes["Landing"].semantic_summary
          == "A screen with Save, Cancel.")


def test_a_description_the_model_already_wrote_is_never_overwritten(monkeypatch):
  """The agent's own sentence is better written and already paid for."""
  describer = _Describer()
  _install(monkeypatch, describer)
  graph = _graph_with(_node("Main", labels=["Save"],
                            summary="I see the note editor."))
  assert _backfill_descriptions(graph, None, 8766) == 0
  assert describer.calls == []
  assert graph.nodes["Main"].semantic_summary == "I see the note editor."


def test_a_seeded_node_is_named_from_what_reached_it(monkeypatch):
  """visit_count 0 means no capture of its own - only an incoming edge."""
  describer = _Describer()
  _install(monkeypatch, describer)
  edge = GraphEdge(edge_id="e1", src_node="Main", action={}, dst_node="Detail",
                   discovered_labels=("Phone", "Address"))
  graph = _graph_with(_node("Main", labels=["Details"], summary="seen"),
                      _node("Detail", labels=(), visits=0), edges=[edge])
  assert _backfill_descriptions(graph, None, 8766) == 1
  assert (graph.nodes["Detail"].semantic_summary
          == "A screen with Phone, Address.")


def test_a_node_nothing_ever_saw_stays_blank_rather_than_invented(monkeypatch):
  describer = _Describer()
  _install(monkeypatch, describer)
  graph = _graph_with(_node("Ghost", labels=(), visits=0))
  assert _backfill_descriptions(graph, None, 8766) == 0
  assert describer.calls == []


def test_a_refused_description_leaves_the_node_blank(monkeypatch):
  """The grounding guard returns "" - blank beats a confident wrong name."""
  _install(monkeypatch, _Describer(refuse={"com.app/.Keys"}))
  graph = _graph_with(_node("Keys", labels=["q", "w", "e"]))
  assert _backfill_descriptions(graph, None, 8766) == 0
  assert graph.nodes["Keys"].semantic_summary == ""


def test_backfilled_names_reach_app_memory_for_the_next_task(monkeypatch):
  """A screen this task only passed through is one the next may reason on."""
  from android_world.parallel_exploration.app_memory import AppMemoryStore
  _install(monkeypatch, _Describer())
  store = AppMemoryStore("/tmp/does-not-need-to-exist")
  graph = _graph_with(_node("Landing", labels=["Save", "Cancel"]))
  _backfill_descriptions(graph, store, 8766)
  assert (store.get("com.app").recall_description("L-Landing")
          == "A screen with Save, Cancel.")


def test_incoming_labels_dedupe_and_ignore_other_destinations():
  edges = [
      GraphEdge(edge_id="a", src_node="s", action={}, dst_node="d",
                discovered_labels=("Phone", "phone")),
      GraphEdge(edge_id="b", src_node="t", action={}, dst_node="d",
                discovered_labels=("Address",)),
      GraphEdge(edge_id="c", src_node="s", action={}, dst_node="other",
                discovered_labels=("Elsewhere",)),
  ]
  graph = _graph_with(_node("d", visits=0), edges=edges)
  assert _incoming_labels(graph, "d") == ["Phone", "Address"]


# --- the walked path as an injection form -------------------------------------
#
# Every earlier form said "control X leads to Y", which needs the current node
# to have outgoing edges. Measured on desc2 (2026-09-10) 68% of steps stand on
# a node that has none, so 76 steps produced 0 injections. This form needs only
# node descriptions.

from scripts.run_serial_exploration_task import _walked_path_context


class _PathSnap:

  def __init__(self, visits, summaries):
    self.node_visits = visits
    self.node_summaries = summaries


def test_the_block_names_visited_screens_most_recent_last():
  text = _walked_path_context(
      _PathSnap({"a": 1, "b": 2, "cur": 1},
            {"a": "A file list.", "b": "A folder view.", "cur": "A dialog."}),
      "cur")
  assert text == ("[Memory] Screens already visited on this task:\n"
                  "- A file list.\n- A folder view.")


def test_the_current_screen_is_not_repeated_back_to_the_model():
  text = _walked_path_context(_PathSnap({"cur": 3}, {"cur": "A dialog."}), "cur")
  assert text == ""


def test_a_screen_visited_three_times_is_named_once():
  text = _walked_path_context(
      _PathSnap({"a": 3, "b": 1}, {"a": "A file list.", "b": "a file list."}), "cur")
  assert text.count("file list") == 1


def test_a_node_only_heard_about_is_not_claimed_as_visited():
  """visit_count 0 is seeded or probe-discovered - the episode never stood there."""
  text = _walked_path_context(
      _PathSnap({"a": 0, "b": 1}, {"a": "A settings page.", "b": "A file list."}),
      "cur")
  assert "settings" not in text


def test_only_the_most_recent_screens_survive_the_cap():
  text = _walked_path_context(
      _PathSnap({f"n{i}": 1 for i in range(9)},
            {f"n{i}": f"Screen {i}." for i in range(9)}), "cur", limit=3)
  assert text.endswith("- Screen 6.\n- Screen 7.\n- Screen 8.")


def test_an_empty_or_undescribed_graph_injects_nothing():
  assert _walked_path_context(None, "cur") == ""
  assert _walked_path_context(_PathSnap({"a": 1}, {}), "cur") == ""
  assert _walked_path_context(_PathSnap({}, {}), "") == ""


def test_the_block_states_a_record_and_never_an_instruction():
  """The one block that told the model what to do cost 12 tasks (v29)."""
  text = _walked_path_context(_PathSnap({"a": 1}, {"a": "A file list."}), "cur")
  lowered = text.lower()
  for word in ("should", "must", "do not", "avoid", "prefer", "instead",
               "you need", "try "):
    assert word not in lowered


def test_a_long_reasoning_sentence_is_trimmed_to_its_identifying_clause():
  """Space spent on the block is space not spent on the screenshot."""
  long = ("I see that the audio recorder is currently recording, as indicated "
          "by the 'Recording...' text and the timer at 00:05.")
  text = _walked_path_context(_PathSnap({"a": 1}, {"a": long}), "cur")
  assert len(text.splitlines()[1].split()) <= 15
  assert "audio recorder" in text


def test_the_whole_block_stays_inside_a_sixty_token_budget():
  five = {f"n{i}": "I see a very long sentence " * 6 for i in range(5)}
  text = _walked_path_context(_PathSnap({k: 1 for k in five}, five), "cur")
  assert len(text.split()) <= 90


def test_a_trimmed_line_ends_at_a_clause_not_mid_phrase():
  long = ("I see the 'New name' dialog on the screen, which is the final step "
          "to save the audio recording.")
  text = _walked_path_context(_PathSnap({"a": 1}, {"a": long}), "cur")
  line = text.splitlines()[1]
  assert line == "- I see the 'New name' dialog on the screen."


def test_a_sentence_with_no_clause_boundary_is_simply_capped():
  long = "A screen listing " + " ".join(f"item{i}" for i in range(20))
  text = _walked_path_context(_PathSnap({"a": 1}, {"a": long}), "cur")
  assert len(text.splitlines()[1].split()) <= 14


def test_a_short_sentence_is_left_exactly_as_written():
  text = _walked_path_context(_PathSnap({"a": 1}, {"a": "A file list."}), "cur")
  assert text.splitlines()[1] == "- A file list."


# --- the screen is what the app shows, not what the keyboard shows -----------
#
# Measured 2026-09-11 over a 453-screen store: 48 screens across 8 apps were
# described as "A screen with More features, Sticker Keyboard, GIF Keyboard..."
# - one such sentence stood on 23 different screens - and 32% of remembered
# transitions pointed at a screen so described, making the fact simply wrong.

from scripts.run_serial_exploration_task import _actionable_labels


class _PkgEl:

  def __init__(self, text="", content_desc="", package="", clickable=True):
    self.text = text
    self.content_desc = content_desc
    self.package = package
    self.clickable = clickable
    self.scrollable = False
    self.checked = None


class _PkgAct:

  def __init__(self, component):
    self.component = component


class _PkgState:

  def __init__(self, component, elements):
    self.activity = _PkgAct(component)
    self.elements = elements


def test_keyboard_controls_are_not_part_of_the_screen():
  state = _PkgState("net.gsantner.markor/.MainActivity", [
      _PkgEl(text="New note", package="net.gsantner.markor"),
      _PkgEl(text="Sticker Keyboard",
          package="com.google.android.inputmethod.latin"),
      _PkgEl(text="GIF Keyboard",
          package="com.google.android.inputmethod.latin"),
  ])
  assert _actionable_labels(state) == ("New note",)


def test_the_status_bar_is_not_part_of_the_screen():
  state = _PkgState("com.app/.MainActivity", [
      _PkgEl(text="Save", package="com.app"),
      _PkgEl(text="Battery 100 percent.", package="com.android.systemui"),
  ])
  assert _actionable_labels(state) == ("Save",)


def test_an_element_with_no_package_is_kept():
  """Older captures and synthetic states carry no package; dropping them would
  silently empty the label set rather than clean it."""
  state = _PkgState("com.app/.MainActivity", [_PkgEl(text="Save", package="")])
  assert _actionable_labels(state) == ("Save",)


def test_labels_are_kept_when_the_activity_is_unknown():
  state = _PkgState("", [_PkgEl(text="Save", package="com.whatever")])
  assert _actionable_labels(state) == ("Save",)


# --- prefill 的风险门：只放行 Back 能撤销的控件 --------------------------------

from scripts.run_serial_exploration_task import _risk_of, _element_for_control


class _RiskEl:

  def __init__(self, text="", content_desc="", class_name="android.widget.TextView",
               checked=None, identity="", bounds=(0, 0, 10, 10)):
    self.text = text
    self.content_desc = content_desc
    self.class_name = class_name
    self.checked = checked
    self.identity = identity
    self.bounds = bounds


def test_a_named_navigation_control_is_safe():
  assert _risk_of(_RiskEl(text="Details")) == "SAFE"


def test_a_button_is_not_safe_because_it_acts():
  assert _risk_of(_RiskEl(text="Delete", class_name="android.widget.Button")) == "UNKNOWN"


def test_a_toggle_is_not_safe_because_it_changes_state():
  assert _risk_of(_RiskEl(text="Wi-Fi", checked=False)) == "UNKNOWN"


def test_an_unnamed_control_is_not_safe():
  """没有名字就无从判断它做什么。"""
  assert _risk_of(_RiskEl()) == "UNKNOWN"


def test_an_image_button_with_a_description_is_safe():
  assert _risk_of(_RiskEl(content_desc="Navigate up",
                          class_name="android.widget.ImageButton")) == "SAFE"


class _RiskState:

  def __init__(self, elements): self.elements = elements


def test_the_control_key_resolves_back_to_the_element_on_screen():
  el = _RiskEl(text="Details", identity="id/row|Details||TextView|[0,0][10,10]")
  assert _element_for_control(_RiskState([_RiskEl(text="Other"), el]),
                              "id/row|Details||TextView") is el


def test_a_control_that_left_the_screen_resolves_to_nothing():
  assert _element_for_control(_RiskState([_RiskEl(text="Other")]),
                              "id/gone|Details||TextView") is None


# --- episode 内预填：本集把这屏"走定了"吗 --------------------------------------
#
# 本集在某屏只按过一个控件时，下次回到这屏还按同一个的比例：执行过 1 次 83%
# （96/115），2 次 93%（42/45）。skip 用不了 1 次这个门控（它替代推理，猜错丢一步，
# 2026-09-01 的一对跑测里每个误跳的任务两次都输）；预填执行后仍然推理，猜错只多
# 一个动作，所以取机会多的那个。

from scripts.run_serial_exploration_task import _episode_decided


class _PfView:

  def __init__(self, outgoing, edges):
    self.outgoing = outgoing
    self.edges = edges


def test_one_executed_control_settles_the_screen():
  v = _PfView({"n": ("e1",)},
              {"e1": {"execution_hit_count": 1, "dst_node": "m",
                      "action": {"control_key": "id/fab|Add||Button"}}})
  assert _episode_decided(v, "n", 1) == ("id/fab|Add||Button", "m")


def test_two_executed_controls_do_not_settle_it():
  v = _PfView({"n": ("e1", "e2")},
              {"e1": {"execution_hit_count": 1, "dst_node": "m",
                      "action": {"control_key": "a|A||B"}},
               "e2": {"execution_hit_count": 1, "dst_node": "p",
                      "action": {"control_key": "b|B||B"}}})
  assert _episode_decided(v, "n", 1) is None


def test_a_speculative_edge_does_not_settle_it():
  """探测说"这里有这么个动作"，从不说"这是该走的路"。"""
  v = _PfView({"n": ("e1",)},
              {"e1": {"execution_hit_count": 0, "dst_node": "m",
                      "action": {"control_key": "a|A||B"}}})
  assert _episode_decided(v, "n", 1) is None


def test_the_stricter_gate_needs_two_executions():
  v = _PfView({"n": ("e1",)},
              {"e1": {"execution_hit_count": 1, "dst_node": "m",
                      "action": {"control_key": "a|A||B"}}})
  assert _episode_decided(v, "n", 2) is None


def test_a_self_loop_is_never_prefilled():
  """回到同一屏的动作，预填它只会原地打转。"""
  v = _PfView({"n": ("e1",)},
              {"e1": {"execution_hit_count": 3, "dst_node": "n",
                      "action": {"control_key": "a|A||B"}}})
  assert _episode_decided(v, "n", 1) is None


def test_a_screen_with_no_outgoing_edge_is_not_settled():
  assert _episode_decided(_PfView({}, {}), "n", 1) is None
  assert _episode_decided(None, "n", 1) is None


from scripts.run_serial_exploration_task import _control_display_label


def test_the_prefilled_control_is_named_by_its_text():
  assert _control_display_label("id/fab|Add recipe||Button") == "Add recipe"


def test_a_content_description_names_a_control_with_no_text():
  assert _control_display_label("id/fab||Navigate up|ImageButton") == "Navigate up"


def test_a_resource_id_only_control_still_gets_a_readable_name():
  assert _control_display_label("com.app:id/btn_record_stop|||View") == "btn record stop"


def test_an_empty_key_still_names_something():
  """模型必须知道被代按了什么；空字符串会让历史读起来像什么都没发生。"""
  assert _control_display_label("") == "a control"


# --- 元素清单注入 --------------------------------------------------------------
#
# 每一种早先的注入形式都以边为单位，而一个节点只有 0.85 条边、对照屏上 9-14 个
# 可命名控件，所以 37% 的已访问屏无话可说。清单把整屏变成可用面。

from scripts.run_serial_exploration_task import _elements_context


class _ElView:

  def __init__(self, elements, outgoing=None, edges=None, summaries=None):
    self.node_elements = elements
    self.outgoing = outgoing or {}
    self.edges = edges or {}
    self.node_summaries = summaries or {}


def test_a_pressed_control_is_stated_with_its_count():
  v = _ElView({"n": ({"label": "Details", "control_key": "k", "clicks": 3},)})
  assert _elements_context(v, "n", {}) == (
      "[Screen] What has been pressed here before:\n- Details (3x)")


def test_a_control_never_pressed_here_is_left_out():
  """它是探索要的覆盖信息，不是模型要的——模型能从截图上读到它。"""
  v = _ElView({"n": ({"label": "Search", "control_key": "k", "clicks": 0},)})
  assert _elements_context(v, "n", {}) == ""


def test_the_destination_is_named_when_the_graph_knows_it():
  v = _ElView({"n": ({"label": "Details", "control_key": "k", "clicks": 1},)},
              outgoing={"n": ("e1",)},
              edges={"e1": {"action": {"control_key": "k"}, "dst_node": "m"}},
              summaries={"m": "A contact detail page with Phone."})
  assert _elements_context(v, "n", {"m": "A contact detail page with Phone."}) == (
      "[Screen] What has been pressed here before:\n"
      "- Details (1x) led to A contact detail page with Phone")


def test_the_most_pressed_control_comes_first():
  v = _ElView({"n": ({"label": "A", "control_key": "a", "clicks": 1},
                     {"label": "B", "control_key": "b", "clicks": 5})})
  assert _elements_context(v, "n", {}).splitlines()[1].startswith("- B (5x)")


def test_the_block_stays_inside_its_token_budget():
  many = tuple({"label": f"Control number {i}", "control_key": str(i),
                "clicks": 1} for i in range(14))
  text = _elements_context(_ElView({"n": many}), "n", {}, budget=30)
  assert len(text.split()) <= 34


def test_a_screen_with_no_inventory_says_nothing():
  assert _elements_context(_ElView({}), "n", {}) == ""
  assert _elements_context(None, "n", {}) == ""


# --- 破环提示 ------------------------------------------------------------------
#
# 94 个打满上限的 episode 里 11 个（12%）含有被执行 >=4 次的边，各浪费 7.9 步。
# 模型自己看不到：历史窗口只留 8 条，而环比这更长（SystemCopyToClipboard 上同一个
# 控件被按了 7 次）。

from scripts.run_serial_exploration_task import _loop_context


class _LoopView:

  def __init__(self, outgoing, edges, nodes=3):
    self.outgoing = outgoing
    self.edges = edges
    self.node_visits = {f"n{i}": 1 for i in range(nodes)}


def _edge(hits, dst, key="id/copy|Copy||Button", nodes_then=3):
  # nodes_then = how large the graph was when this edge was last taken; the
  # graph having grown since means the task is iterating, not spinning.
  return {"execution_hit_count": hits, "dst_node": dst,
          "nodes_at_last_execution": nodes_then,
          "action": {"control_key": key}}


def test_a_control_that_keeps_returning_here_is_reported():
  v = _LoopView({"n": ("e1",)}, {"e1": _edge(4, "n")})
  assert _loop_context(v, "n") == (
      '[Loop] "Copy" has been used here 4 times and no new screen has been '
      "reached since.")


def test_a_task_that_keeps_finding_new_screens_is_iterating_not_spinning():
  """删一个菜谱、回到少一项的列表、再删下一个——图在长大，那是迭代。"""
  v = _LoopView({"n": ("e1",)}, {"e1": _edge(6, "m", nodes_then=1)}, nodes=5)
  assert _loop_context(v, "n") == ""


def test_a_cycle_across_several_screens_is_still_a_loop():
  """真实形态是多节点环，不是自环——任何一次跑测里 dst==src 的边都是 0。"""
  v = _LoopView({"n": ("e1",)}, {"e1": _edge(4, "other")})
  assert '4 times' in _loop_context(v, "n")


def test_two_uses_are_not_yet_a_loop():
  v = _LoopView({"n": ("e1",)}, {"e1": _edge(2, "n")})
  assert _loop_context(v, "n") == ""


def test_the_worst_offender_is_the_one_reported():
  v = _LoopView({"n": ("e1", "e2")},
                {"e1": _edge(3, "n", "id/a|Alpha||Button"),
                 "e2": _edge(7, "n", "id/b|Beta||Button")})
  assert '"Beta" has been used here 7 times' in _loop_context(v, "n")


def test_the_threshold_is_configurable():
  v = _LoopView({"n": ("e1",)}, {"e1": _edge(2, "n")})
  assert _loop_context(v, "n", min_repeats=2) != ""


def test_it_states_a_record_and_never_an_instruction():
  text = _loop_context(_LoopView({"n": ("e1",)}, {"e1": _edge(5, "n")}), "n").lower()
  for word in ("should", "must", "try ", "instead", "avoid", "do not", "stop"):
    assert word not in text


def test_a_screen_with_nothing_repeated_says_nothing():
  assert _loop_context(_LoopView({}, {}), "n") == ""
  assert _loop_context(None, "n") == ""
