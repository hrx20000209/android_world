"""Adversarial invariants for episode-local, task-directed navigation."""

import json
import copy
from types import SimpleNamespace as NS

import pytest

from android_world.parallel_exploration import task_navigation as tn
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
from scripts.run_serial_exploration_task import GenerationGuardedGraph


TASK = "Find Alice's event time in Calendar"


def state(name, labels=()):
  return {"activity": "app.example/." + name, "package": "app.example",
          "layout_signature": name, "structural_signature": "strict-" + name,
          "labels": list(labels)}


def element(label, bounds=(0, 0, 100, 100)):
  return NS(text=label, content_desc="", resource_id="app:id/" + label,
            class_name="android.widget.ImageButton", bounds=bounds,
            clickable=True, checked=None, package="app.example",
            identity=f"app:id/{label}|{label}||android.widget.ImageButton|{bounds}")


def signature(value, elements=()):
  elems = list(elements) + [NS(text=x, content_desc="", package="app.example") for x in value["labels"]]
  return NS(activity=NS(component=value["activity"]), layout_sig=value["layout_signature"],
            struct_sig=NS(digest=value["structural_signature"]), elements=elems)


def intent(output=None, generation=0):
  value = {"navigation_only": True, "stop_before_decision": True,
           "package": "app.example", "target_labels": ["Event time"], "entity_labels": ["Alice"]}
  if output is not None:
    value.update(output)
  text = "<THINK>Continue navigation. <NAV_INTENT>" + json.dumps(value) + "</NAV_INTENT></THINK>"
  return tn.parse_navigation_intent(text, TASK, generation)


def snapshot():
  a, b, c = state("Home"), state("Calendar"), state("Details", ["Event time", "Alice"])
  edges = {}
  for i, (src, dst, label) in enumerate(((a, b, "Calendar"), (b, c, "Details"))):
    edges[str(i)] = {
        "edge_id": str(i), "src_node": src["layout_signature"], "dst_node": dst["layout_signature"],
        "status": "SPECULATIVE", "probe_count": 1, "discovered_labels": dst["labels"],
        "risk_level": "LOW", "action": {"element_identity": element(label).identity},
        "navigation_observations": [{
            "generation": 0, "source": src, "target": dst,
            "selector": {"resource_id": "app:id/" + label, "text": label,
                         "class_name": "android.widget.ImageButton"},
            "action_type": "click", "probe_type": "TAP_NAV", "ui_restored": True,
            "inverse_level": "INVERSE", "state_effect_verified": True, "cost_s": .1,
        }],
    }
  return {"generation": 1, "edges": edges}, a, b, c


def compile_route(snap=None, need=None, current=None, **config):
  base, a, _, _ = snapshot()
  return tn.TaskNavigationPlanner(**config).compile(
      snap or base, current or a, need or intent(), task=TASK, inference_cost_s=10)


def test_cold_route_needs_no_revisits_no_entropy_and_stops_at_task_terminal():
  route = compile_route()
  assert len(route.hops) == 2
  assert route.terminal["labels"] == ["Event time", "Alice"]
  assert route.expected_net_saving_s == pytest.approx(19.3)


@pytest.mark.parametrize("mutation", [
    {"generation": 1}, {"generation": -6}, {"ui_restored": False},
    {"inverse_level": "DEEPLINK"}, {"state_effect_verified": False},
    {"action_type": "input_text"}, {"checked": False}, {"dynamic_content": True},
])
def test_uncommitted_stale_mutating_or_unverified_witness_cannot_execute(mutation):
  snap, *_ = snapshot()
  snap["edges"]["0"]["navigation_observations"][0].update(mutation)
  assert compile_route(snap) is None


@pytest.mark.parametrize("field,value", [
    ("status", "INVALID"), ("status", "STALE"), ("risk_level", "HIGH"),
    ("rollback_failure_count", 1), ("execution_miss_count", 1), ("dynamic_content", True),
])
def test_contradictory_edge_history_and_risk_cannot_be_outvoted(field, value):
  snap, *_ = snapshot()
  snap["edges"]["0"][field] = value
  assert compile_route(snap) is None


def test_screen_aliasing_wrong_entity_wrong_task_and_expired_intent_fail_closed():
  snap, a, _, c = snapshot()
  assert compile_route(current={**a, "structural_signature": "another-list-instance"}) is None
  c["labels"] = ["Event time", "Bob"]
  assert compile_route(snap) is None
  snap, a, *_ = snapshot()
  planner = tn.TaskNavigationPlanner()
  assert planner.compile(snap, a, intent(), task="Delete Alice", inference_cost_s=10) is None
  assert planner.compile({**snap, "generation": 5}, a, intent(), task=TASK) is None


def test_ui_only_is_explicit_ablation_not_a_safety_certificate():
  snap, *_ = snapshot()
  for edge in snap["edges"].values():
    edge["navigation_observations"][0]["state_effect_verified"] = False
  assert compile_route(snap) is None
  route = compile_route(snap, allow_ui_only=True)
  assert all(h.evidence_level == "ui_only" for h in route.hops)


def test_cross_package_and_weak_selector_are_rejected():
  snap, *_ = snapshot()
  snap["edges"]["0"]["navigation_observations"][0]["target"]["package"] = "other.app"
  assert compile_route(snap) is None
  snap, *_ = snapshot()
  snap["edges"]["0"]["navigation_observations"][0]["selector"]["resource_id"] = ""
  assert compile_route(snap) is None


def test_macro_relocates_pixels_verifies_each_hop_and_returns_before_decision():
  _, a, b, c = snapshot()
  states = iter([signature(a, [element("Calendar", (100, 200, 300, 400))]),
                 signature(b, [element("Details")]), signature(c)])
  actions, history = [], []
  result = tn.execute_route(compile_route(), intent(), task=TASK, capture=lambda: next(states),
                            execute=actions.append, on_action=lambda action, hop: history.append(hop.edge_id))
  assert result.reached_terminal and result.completed_hops == 2
  assert actions[0] == {"action_type": "click", "x": 200, "y": 300}
  assert len(actions) == len(history) == 2  # No terminal save/edit/answer.


def test_macro_wrong_landing_aborts_without_second_action_and_preserves_history():
  _, a, _, _ = snapshot()
  states = iter([signature(a, [element("Calendar")]), signature(state("Unexpected"))])
  actions, history = [], []
  result = tn.execute_route(compile_route(), intent(), task=TASK, capture=lambda: next(states),
                            execute=actions.append, on_action=lambda action, hop: history.append(hop.edge_id))
  assert result.reason == "landing_contract_mismatch" and result.attempted_hops == 1
  assert result.completed_hops == 0 and result.failed_edge_id == "0"
  assert len(actions) == len(history) == 1


def test_duplicate_resources_and_changed_screen_stop_before_first_action():
  _, a, *_ = snapshot()
  actions = []
  result = tn.execute_route(compile_route(), intent(), task=TASK,
                            capture=lambda: signature(a, [element("Calendar"), element("Calendar")]),
                            execute=actions.append)
  assert result.reason == "selector_missing_or_ambiguous" and not actions
  result = tn.execute_route(compile_route(), intent(), task=TASK,
                            capture=lambda: signature({**a, "structural_signature": "changed"}),
                            execute=actions.append)
  assert result.reason == "source_contract_mismatch" and not actions


def test_explicit_contract_parser_rejects_heuristic_or_bad_outputs():
  assert tn.parse_navigation_intent("I need to find Alice", TASK, 0) is None
  assert intent({"navigation_only": "true"}) is None
  assert intent({"stop_before_decision": False}) is None
  assert intent({"target_labels": "Calendar"}) is None
  assert intent({"package": "launcher"}) is None
  assert intent({"target_activity": "other.app/.Main"}) is None


def test_graph_witness_is_bounded_copied_and_absent_from_current_generation():
  graph = ProgressiveBeliefGraph("task")
  edge = graph.add_speculative_transition(
      "a", {"action_type": "click"}, "b", path_probability=.5,
      confidence=.2, expected_information_gain=0, risk_level="LOW", exploration_cost=0)
  guarded = GenerationGuardedGraph(graph)
  observation = {"source": {"labels": ["Original"]}, "generation": 0}
  for _ in range(6):
    graph.record_navigation_observation(edge.edge_id, observation)
  observation["source"]["labels"].append("Mutated")
  assert len(edge.navigation_observations) == 4
  assert edge.navigation_observations[0]["source"]["labels"] == ["Original"]
  assert guarded.snapshot(for_step=0).edges[edge.edge_id]["navigation_observations"] == ()
  guarded.commit_step()
  frozen = guarded.snapshot(for_step=1)
  graph.record_navigation_observation(edge.edge_id, {"generation": 1})
  assert frozen.edges[edge.edge_id]["navigation_observations"][-1]["generation"] == 0


def test_frontier_propagates_task_target_beyond_local_button_text():
  snap, *_ = snapshot()
  elem = element("Calendar")
  scores = tn.frontier_scores(snap, "Home", [elem], ["Event time", "Alice"])
  assert scores[elem.identity]["target_coverage"] == 0
  assert scores[elem.identity]["downstream_coverage"] > 0
  assert scores[elem.identity]["probe_count"] == 1
  assert tn.frontier_scores(snap, "Home", [elem], [])[elem.identity]["score"] == 0


def test_no_route_if_terminal_already_visible_and_search_budget_is_bounded():
  snap, _, _, c = snapshot()
  assert compile_route(snap, current=c) is None
  assert compile_route(max_expansions=1) is None


def test_macro_actions_remain_in_recovery_prefix_without_becoming_model_calls():
  model = [{"action_dict": {"action_type": "open_app"}},
           {"action_dict": {"action_type": "click", "x": 10}}]
  macro = [{"before_action_index": 1, "action_dict": {"action_type": "click", "x": 20}},
           {"before_action_index": 2, "action_dict": {"action_type": "click", "x": 30}}]
  merged = tn.merge_action_prefix(model, macro)
  assert [x["action_dict"].get("x") for x in merged] == [None, 20, 10, 30]
  assert len(model) == 2


def test_same_exact_source_and_action_with_conflicting_landings_is_not_a_certificate():
  snap, *_ = snapshot()
  extra = copy.deepcopy(snap["edges"]["0"]["navigation_observations"][0])
  extra["target"] = state("Different")
  snap["edges"]["0"]["navigation_observations"].append(extra)
  assert compile_route(snap) is None


def test_model_action_parser_is_compatible_with_optional_think_contract():
  from android_world.agents.gelab_agent import parse_gelab_response
  contract = {"navigation_only": True, "stop_before_decision": True,
              "package": "app.example", "target_labels": ["Event time"]}
  response = ("<THINK>Navigate <NAV_INTENT>" + json.dumps(contract)
              + "</NAV_INTENT></THINK>\nexplain:open calendar\taction:CLICK\tpoint:200,300\tsummary:opened calendar")
  parsed = parse_gelab_response(response)
  assert parsed["action"] == "CLICK"
  assert parsed["summary"] == "opened calendar"
  assert tn.parse_navigation_intent(response, TASK, 0) is not None


def test_capture_or_execution_exception_returns_control_without_following_suffix():
  _, a, *_ = snapshot()
  def fail():
    raise RuntimeError("capture unavailable")
  result = tn.execute_route(compile_route(), intent(), task=TASK, capture=fail, execute=lambda _: None)
  assert result.reason == "capture_failed" and result.attempted_hops == 0
  history = []
  def fail_execution(_):
    raise RuntimeError("uncertain device action")
  result = tn.execute_route(compile_route(), intent(), task=TASK,
                            capture=lambda: signature(a, [element("Calendar")]),
                            execute=fail_execution,
                            on_action=lambda action, hop: history.append(hop.edge_id))
  assert result.reason == "execution_failed" and result.attempted_hops == 1
  assert history == ["0"]


def _execution_snapshot(**overrides):
  snap, a, b, c = snapshot()
  for edge in snap["edges"].values():
    witness = edge["navigation_observations"][0]
    witness.update({"origin": "execution", "probe_type": "EXEC_NAV",
                    "ui_restored": False, "inverse_level": ""})
    witness.update(overrides)
  return snap, a


def test_execution_witness_needs_state_certificate_not_inverse():
  snap, a = _execution_snapshot()
  route = tn.TaskNavigationPlanner().compile(snap, a, intent(), task=TASK, inference_cost_s=10)
  assert route is not None and len(route.hops) == 2
  snap, a = _execution_snapshot(state_effect_verified=False)
  planner = tn.TaskNavigationPlanner()
  assert planner.compile(snap, a, intent(), task=TASK, inference_cost_s=10) is None
  assert planner.last_rejections == {"unverified_app_state_effect": 2}


def test_execution_origin_cannot_borrow_probe_types():
  # A probe row relabelled as execution must still look like an executed click.
  snap, a = _execution_snapshot(probe_type="TAP_NAV")
  planner = tn.TaskNavigationPlanner()
  assert planner.compile(snap, a, intent(), task=TASK, inference_cost_s=10) is None
  assert planner.last_rejections == {"not_navigation_probe": 2}
