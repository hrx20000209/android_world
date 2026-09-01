"""Tests for the candidate information matrix, scorer, distiller and gate."""

from __future__ import annotations

import dataclasses
import math

from absl.testing import absltest

from android_world.parallel_exploration import graph_distiller
from android_world.parallel_exploration import state_graph_information as sgi
from android_world.parallel_exploration.rankers import UiElement


@dataclasses.dataclass
class _Activity:
  component: str = "com.app/.MainActivity"


@dataclasses.dataclass
class _State:
  activity: _Activity = dataclasses.field(default_factory=_Activity)
  screen_width: int = 1000
  screen_height: int = 2000


@dataclasses.dataclass
class _Snapshot:
  generation: int = 5
  node_visits: dict = dataclasses.field(default_factory=dict)
  node_entropy: dict = dataclasses.field(default_factory=dict)
  edges: dict = dataclasses.field(default_factory=dict)
  outgoing: dict = dataclasses.field(default_factory=dict)


def _element(text="Details", cls="android.widget.Button", **kw):
  return UiElement(text=text, class_name=cls, bounds=(0, 0, 100, 100),
                   clickable=True, **kw)


def _edge(edge_id, src, identity, **kw):
  edge = {
      "edge_id": edge_id, "src_node": src, "dst_node": kw.pop("dst", "n2"),
      "status": kw.pop("status", "SPECULATIVE"), "confidence": 0.5,
      "risk_level": "LOW",
      "action": {"element_identity": identity, "probe_type": "TAP_NAV",
                 "role": "button"},
      "discovered_labels": kw.pop("labels", ("Phone", "Address")),
      "last_updated_generation": kw.pop("gen", 5),
  }
  edge.update(kw)
  return edge


class MatrixTest(absltest.TestCase):

  def _build(self, elements, snapshot=None, need=None, **kw):
    return sgi.StateGraphInformationMatrix().build(
        _State(), "n1", elements, snapshot, need or {}, **kw)

  def test_unknown_history_is_none_not_zero(self):
    """A never-probed candidate must not read as a measured failure."""
    row, = self._build([_element()], _Snapshot())
    self.assertFalse(row.has_exact_history)
    self.assertIsNone(row.alignment_rate)
    self.assertIsNone(row.rollback_success_rate)
    self.assertIsNone(row.mean_realized_ig)

  def test_exact_history_matched_by_action_not_edge_id(self):
    element = _element()
    snap = _Snapshot(
        outgoing={"n1": ["e1"]},
        edges={"e1": _edge("e1", "n1", element.identity, probe_count=2,
                           inference_alignment_count=2,
                           execution_hit_count=2, execution_miss_count=0,
                           cumulative_realized_ig=1.0,
                           cumulative_exploration_cost=4.0)})
    row, = self._build([element], snap)
    self.assertTrue(row.has_exact_history)
    self.assertEqual(row.alignment_rate, 1.0)
    self.assertEqual(row.mean_realized_ig, 0.5)
    self.assertEqual(row.mean_exploration_cost, 2.0)
    self.assertEqual(row.destination_node_id, "n2")

  def test_node_maturity_features(self):
    element = _element()
    snap = _Snapshot(node_visits={"n1": 3}, node_entropy={"n1": 0.9},
                     outgoing={"n1": ["e1", "e2"]},
                     edges={"e1": _edge("e1", "n1", element.identity),
                            "e2": _edge("e2", "n1", "other", status="INVALID")})
    row, _ = self._build([element, _element(text="Other")], snap)
    self.assertEqual(row.node_visit_count, 3)
    self.assertEqual(row.outgoing_edge_count, 2)
    self.assertEqual(row.valid_outgoing_edge_count, 1)
    self.assertEqual(row.candidate_element_count, 2)
    self.assertEqual(row.exploration_coverage, 1.0)

  def test_destination_features_and_recent_path(self):
    element = _element()
    snap = _Snapshot(node_visits={"n2": 2}, node_entropy={"n2": 0.3},
                     outgoing={"n1": ["e1"], "n2": ["e2"]},
                     edges={"e1": _edge("e1", "n1", element.identity),
                            "e2": _edge("e2", "n2", "x", dst="n3")})
    row, = self._build([element], snap, recent_nodes=["n2"])
    self.assertEqual(row.destination_visit_count, 2)
    self.assertEqual(row.destination_out_degree, 1)
    self.assertEqual(row.destination_subtree_size, 1)
    self.assertTrue(row.destination_in_recent_path)

  def test_subtree_size_terminates_on_cycle(self):
    snap = _Snapshot(outgoing={"a": ["e1"], "b": ["e2"]},
                     edges={"e1": _edge("e1", "a", "x", dst="b"),
                            "e2": _edge("e2", "b", "y", dst="a")})
    self.assertEqual(sgi._subtree_size(snap, "a"), 1)

  def test_need_features_stay_separate(self):
    row, = self._build([_element(text="Phone number")], _Snapshot(),
                       need={"target_entity": "phone",
                             "required_information_slots": ["number"]})
    self.assertEqual(row.target_match, 1.0)
    self.assertEqual(row.unresolved_information_match, 1.0)
    self.assertEqual(row.expected_affordance_match, 0.0)

  def test_cost_varies_by_probe_type(self):
    matrix = sgi.StateGraphInformationMatrix()
    rows = matrix.build(_State(), "n1",
                        [_element(), _element(text="List", cls="x.ListView")],
                        _Snapshot(), {},
                        probe_type_of=lambda e: "EXPAND" if e.text == "List" else "TAP_NAV")
    tap, expand = rows
    self.assertLess(tap.expected_total_exploration_cost,
                    expand.expected_total_exploration_cost)


class SafetyGateTest(absltest.TestCase):

  def _row(self, **kw):
    row = sgi.CandidateInformationRow(
        element=_element(), element_identity="i", text="t", content_desc="",
        role="button", probe_type="TAP_NAV", norm_x=0.5, norm_y=0.5,
        clickable=True, scrollable=False, estimated_recoverability=0.9)
    for k, v in kw.items():
      setattr(row, k, v)
    return row

  def test_allows_recoverable(self):
    self.assertTrue(sgi.SafetyGate().evaluate(self._row()).allowed)

  def test_blocks_each_hard_condition(self):
    gate = sgi.SafetyGate()
    for field in ("blocked_element", "blocked_recovery_context",
                  "cross_package_history"):
      self.assertFalse(gate.evaluate(self._row(**{field: True})).allowed, field)
    self.assertFalse(gate.evaluate(self._row(risk_level="HIGH")).allowed)
    self.assertFalse(
        gate.evaluate(self._row(estimated_recoverability=0.2)).allowed)


class ScorerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.scorer = sgi.PredictiveElementScorer()
    self.matrix = sgi.StateGraphInformationMatrix()

  def _rows(self, elements, snapshot=None, need=None):
    return self.matrix.build(_State(), "n1", elements, snapshot or _Snapshot(),
                             need or {})

  def test_alignment_history_raises_path_probability(self):
    """A5 case 2: probes that kept matching the model become preferred."""
    aligned, plain = _element(text="Details"), _element(text="Plain")
    snap = _Snapshot(outgoing={"n1": ["e1"]},
                     edges={"e1": _edge("e1", "n1", aligned.identity,
                                        probe_count=4,
                                        inference_alignment_count=4,
                                        execution_miss_count=0)})
    scored = {s.row.text: s for s in self.scorer.rank(self._rows([aligned, plain], snap))}
    self.assertGreater(scored["Details"].components["path_probability"],
                       scored["Plain"].components["path_probability"])

  def test_known_destination_is_discounted(self):
    """A5 case 6: a covered node pushes exploration to untried candidates."""
    known, fresh = _element(text="Known"), _element(text="Fresh")
    snap = _Snapshot(outgoing={"n1": ["e1"]},
                     edges={"e1": _edge("e1", "n1", known.identity,
                                        probe_count=1, status="VERIFIED")})
    scored = {s.row.text: s for s in self.scorer.rank(self._rows([known, fresh], snap))}
    self.assertLess(scored["Known"].components["expected_information_gain"],
                    scored["Fresh"].components["expected_information_gain"])

  def test_expensive_recovery_lowers_utility(self):
    rows = self.matrix.build(
        _State(), "n1", [_element(), _element(text="B")], _Snapshot(), {},
        probe_type_of=lambda e: "EXPAND" if e.text == "B" else "TAP_NAV")
    scored = {s.row.text: s.utility for s in self.scorer.rank(rows)}
    self.assertGreater(scored["Details"], scored["B"])

  def test_need_match_beats_novelty(self):
    """A2: a screen full of new labels is not the same as task relevance."""
    relevant = _element(text="Phone")
    noisy = _element(text="Unrelated")
    snap = _Snapshot(
        outgoing={"n1": ["e2"]},
        edges={"e2": _edge("e2", "n1", noisy.identity, probe_count=1,
                           labels=("a", "b", "c", "d", "e", "f", "g", "h"))})
    scored = {s.row.text: s.utility for s in self.scorer.rank(
        self._rows([relevant, noisy], snap,
                   need={"target_entity": "phone"}))}
    self.assertGreater(scored["Phone"], scored["Unrelated"])

  def test_components_logged_separately(self):
    scored, = self.scorer.rank(self._rows([_element()]))
    self.assertContainsSubset(
        ["path_probability", "expected_information_gain", "predictive_value",
         "expected_cost", "recoverability", "ui_novelty_score"],
        scored.components)


class ContextualHistoryTest(absltest.TestCase):

  def test_aggregates_by_kind_and_transfers(self):
    table = sgi.ContextualHistoryTable()
    key = sgi.ContextualHistoryTable.key("TAP_MENU", "menu_item", "entity")
    for ok in (True, True, False, True):
      table.record_probe(key, rollback_ok=ok, realized_ig=0.5)
    stats = table.lookup(key)
    self.assertEqual(stats["contextual_probe_count"], 4)
    self.assertEqual(stats["contextual_rollback_success_rate"], 0.75)
    self.assertEqual(stats["contextual_mean_realized_ig"], 0.5)

  def test_unseen_key_is_empty_not_zero_rates(self):
    stats = sgi.ContextualHistoryTable().lookup(
        sgi.ContextualHistoryTable.key("SCROLL", "container", "slot"))
    self.assertEqual(stats, {"contextual_probe_count": 0})

  def test_contextual_recoverability_feeds_the_gate(self):
    table = sgi.ContextualHistoryTable()
    key = sgi.ContextualHistoryTable.key("TAP_NAV", "button", "none")
    for _ in range(20):
      table.record_probe(key, rollback_ok=False)
    matrix = sgi.StateGraphInformationMatrix(contextual_history=table)
    row, = matrix.build(_State(), "n1", [_element()], _Snapshot(), {})
    self.assertLess(row.estimated_recoverability, 0.55)
    self.assertFalse(sgi.SafetyGate().evaluate(row).allowed)


class DistillerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.distiller = graph_distiller.GraphDistiller()

  def _snap(self, **edges):
    return _Snapshot(outgoing={"n1": list(edges)}, edges=dict(edges))

  def test_empty_graph_produces_no_context(self):
    self.assertEqual(self.distiller.distill("n1", self._snap()), "")

  def test_observed_fact_is_factual_not_advice(self):
    snap = self._snap(e1=_edge("e1", "n1", "rid|Details|||", status="VERIFIED"))
    out = self.distiller.distill("n1", snap, {"target_entity": "phone"})
    self.assertIn("Details -> {Phone, Address}.", out)
    for word in ("should", "click", "recommend", "try"):
      self.assertNotIn(word, out.lower())

  def test_taken_edge_becomes_done(self):
    snap = self._snap(e1=_edge("e1", "n1", "rid|Details|||", status="VERIFIED"),
                      e2=_edge("e2", "n1", "rid|Reviews|||", status="VERIFIED"))
    out = self.distiller.distill("n1", snap, {}, taken_edges=["e1"])
    self.assertIn("Done:", out)
    self.assertIn("already used", out)

  def test_absence_is_phrased_as_unobserved(self):
    snap = self._snap(e1=_edge("e1", "n1", "rid|Reviews|||", labels=(),
                               probe_count=1),
                      e2=_edge("e2", "n1", "rid|Details|||", status="VERIFIED"))
    out = self.distiller.distill("n1", snap, {"target_entity": "phone"})
    self.assertIn("no relevant evidence observed under Reviews", out)
    self.assertNotIn("does not contain", out)

  def test_unprobed_edge_makes_no_absence_claim(self):
    """Coverage, not silence, is what licenses "nothing was observed"."""
    snap = self._snap(e1=_edge("e1", "n1", "rid|Reviews|||", labels=()))
    self.assertEqual(self.distiller.distill("n1", snap, {}), "")

  def test_no_raw_confidence_numbers(self):
    snap = self._snap(e1=_edge("e1", "n1", "rid|Details|||", status="VERIFIED"))
    out = self.distiller.distill("n1", snap, {"target_entity": "phone"})
    self.assertNotIn("confidence", out.lower())
    self.assertNotIn("0.", out)

  def test_budget_caps_facts_and_tokens(self):
    edges = {f"e{i}": _edge(f"e{i}", "n1", f"rid|Item{i}|||", status="VERIFIED")
             for i in range(10)}
    snap = _Snapshot(outgoing={"n1": list(edges)}, edges=edges)
    out = self.distiller.distill("n1", snap, {"target_entity": "item"})
    self.assertLessEqual(len(out.split()), 64)
    self.assertLessEqual(sum(line.count("->") for line in out.splitlines()), 3)

  def test_stale_generation_lowers_utility(self):
    fresh = graph_distiller.GraphDistiller()._to_fact(
        _edge("e1", "n1", "rid|A|||", status="VERIFIED", gen=5), set(), set(), 5)
    old = graph_distiller.GraphDistiller()._to_fact(
        _edge("e2", "n1", "rid|A|||", status="VERIFIED", gen=0), set(), set(), 20)
    self.assertGreater(fresh.utility_score, old.utility_score)

  def test_invalid_edges_never_reach_the_prompt(self):
    snap = self._snap(e1=_edge("e1", "n1", "rid|Gone|||", status="INVALID"))
    self.assertEqual(self.distiller.distill("n1", snap, {}), "")


class ReasoningGateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.gate = graph_distiller.ReasoningGate()
    self.reusable = _edge("e1", "n1", "rid|Save|||", status="REUSABLE")

  def test_skips_when_edge_reusable_and_node_certain(self):
    snap = _Snapshot(node_entropy={"n1": 0.0}, outgoing={"n1": ["e1"]},
                     edges={"e1": self.reusable})
    decision = self.gate.decide("n1", snap, {}, self.reusable)
    self.assertEqual(decision.mode, graph_distiller.SKIP_INFERENCE)

  def test_cycle_blocks_skip_but_context_still_offered(self):
    snap = _Snapshot(node_entropy={"n1": 0.0}, outgoing={"n1": ["e1"]},
                     edges={"e1": self.reusable})
    decision = self.gate.decide("n1", snap, {"target_entity": "save"},
                                self.reusable, recent_nodes=["n2"])
    self.assertEqual(decision.mode, graph_distiller.GRAPH_ENHANCED_INFERENCE)
    self.assertIn("cycle", decision.reason)
    self.assertIn("[Memory]", decision.graph_context)

  def test_high_entropy_blocks_skip(self):
    snap = _Snapshot(node_entropy={"n1": 1.2}, outgoing={"n1": ["e1", "e2"]},
                     edges={"e1": self.reusable,
                            "e2": _edge("e2", "n1", "rid|Other|||")})
    decision = self.gate.decide("n1", snap, {}, self.reusable)
    self.assertNotEqual(decision.mode, graph_distiller.SKIP_INFERENCE)

  def test_zero_entropy_single_edge_is_not_vacuously_certain(self):
    """decision_entropy is identically 0 with one edge; degree must matter."""
    snap = _Snapshot(node_entropy={"n1": 0.0}, outgoing={"n1": ["e1"]},
                     edges={"e1": self.reusable})
    self.assertEqual(len(snap.outgoing["n1"]), 1)
    self.assertEqual(self.gate.decide("n1", snap, {}, self.reusable).mode,
                     graph_distiller.SKIP_INFERENCE)

  def test_consecutive_skip_cap(self):
    snap = _Snapshot(node_entropy={"n1": 0.0}, outgoing={"n1": ["e1"]},
                     edges={"e1": self.reusable})
    decision = self.gate.decide("n1", snap, {}, self.reusable,
                                consecutive_skips=3)
    self.assertNotEqual(decision.mode, graph_distiller.SKIP_INFERENCE)

  def test_no_facts_means_plain_inference(self):
    snap = _Snapshot(outgoing={"n1": []}, edges={})
    decision = self.gate.decide("n1", snap, {}, None)
    self.assertEqual(decision.mode, graph_distiller.NORMAL_INFERENCE)
    self.assertEqual(decision.graph_context, "")

  def test_ablation_flags_disable_each_path(self):
    snap = _Snapshot(node_entropy={"n1": 0.0}, outgoing={"n1": ["e1"]},
                     edges={"e1": self.reusable})
    no_skip = graph_distiller.ReasoningGate(
        config=graph_distiller.GateConfig(enable_skip=False))
    self.assertNotEqual(no_skip.decide("n1", snap, {}, self.reusable).mode,
                        graph_distiller.SKIP_INFERENCE)
    neither = graph_distiller.ReasoningGate(
        config=graph_distiller.GateConfig(enable_skip=False,
                                          enable_graph_context=False))
    self.assertEqual(neither.decide("n1", snap, {}, self.reusable).mode,
                     graph_distiller.NORMAL_INFERENCE)


if __name__ == "__main__":
  absltest.main()


class NeedLineTest(absltest.TestCase):

  def test_deduplicates_overgenerated_need_items(self):
    line, = graph_distiller._need_line({
        "target_entity": "Zucchini Noodles with Pesto",
        "required_information_slots": [
            "Zucchini Noodles with Pesto", "Zucchini", "Noodles", "Pesto"],
    })
    self.assertEqual(line, "Need: Zucchini Noodles with Pesto.")

  def test_caps_item_count_and_length(self):
    line, = graph_distiller._need_line({
        "target_entity": "",
        "required_information_slots": ["a" * 80, "b", "c", "d", "e", "f"],
    })
    self.assertNotIn("a" * 80, line)
    self.assertLessEqual(line.count(","), 3)

  def test_no_need_line_when_nothing_stated(self):
    self.assertEqual(graph_distiller._need_line({}), [])


class OpenAppFactTest(absltest.TestCase):

  def test_open_app_makes_no_absence_claim(self):
    edge = _edge("e1", "n1", "rid|Broccoli|||", labels=(), probe_count=1)
    edge["action"]["action_type"] = "open_app"
    snap = _Snapshot(outgoing={"n1": ["e1"]}, edges={"e1": edge})
    self.assertEqual(graph_distiller.GraphDistiller().distill("n1", snap, {}), "")


class NegativeFactBudgetTest(absltest.TestCase):

  def test_unnamed_control_makes_no_absence_claim(self):
    edge = _edge("e1", "n1", "|||", labels=(), probe_count=1)
    edge["action"]["x"], edge["action"]["y"] = 540, 1168
    snap = _Snapshot(outgoing={"n1": ["e1"]}, edges={"e1": edge})
    self.assertEqual(graph_distiller.GraphDistiller().distill("n1", snap, {}), "")

  def test_at_most_one_negative_fact(self):
    edges = {f"e{i}": _edge(f"e{i}", "n1", f"rid|Dead{i}|||", labels=(),
                            probe_count=1) for i in range(4)}
    edges["good"] = _edge("good", "n1", "rid|Details|||", status="VERIFIED")
    snap = _Snapshot(outgoing={"n1": list(edges)}, edges=edges)
    out = graph_distiller.GraphDistiller().distill(
        "n1", snap, {"target_entity": "dead"})
    self.assertLessEqual(out.count("no relevant evidence"), 1)

  def test_positive_fact_is_not_crowded_out_by_negatives(self):
    edges = {f"e{i}": _edge(f"e{i}", "n1", f"rid|Dead{i}|||", labels=(),
                            probe_count=1) for i in range(4)}
    edges["good"] = _edge("good", "n1", "rid|Details|||", status="VERIFIED")
    snap = _Snapshot(outgoing={"n1": list(edges)}, edges=edges)
    out = graph_distiller.GraphDistiller().distill("n1", snap,
                                                   {"target_entity": "phone"})
    self.assertIn("Details -> {Phone, Address}.", out)

  def test_model_prose_is_not_echoed_back_as_need(self):
    line = graph_distiller._need_line({
        "target_entity": "Lentil Soup",
        "required_information_slots": [
            "s detail page. I will click on the", "card to proceed",
            "Therefore", "Garlic Butter Shrimp"],
    })
    self.assertEqual(line, ["Need: Lentil Soup, Garlic Butter Shrimp."])


class InjectionThresholdTest(absltest.TestCase):

  def test_no_injection_without_a_positive_fact(self):
    """Negatives and Done facts alone are not worth prompt space."""
    edges = {
        "e1": _edge("e1", "n1", "rid|Dead|||", labels=(), probe_count=1),
        "e2": _edge("e2", "n1", "rid|Used|||", status="VERIFIED"),
    }
    snap = _Snapshot(outgoing={"n1": list(edges)}, edges=edges)
    out = graph_distiller.GraphDistiller().distill(
        "n1", snap, {"target_entity": "phone"}, taken_edges=["e2"])
    self.assertEqual(out, "")

  def test_positive_fact_licenses_the_supporting_ones(self):
    edges = {
        "e1": _edge("e1", "n1", "rid|Dead|||", labels=(), probe_count=1),
        "e2": _edge("e2", "n1", "rid|Details|||", status="VERIFIED"),
    }
    snap = _Snapshot(outgoing={"n1": list(edges)}, edges=edges)
    out = graph_distiller.GraphDistiller().distill(
        "n1", snap, {"target_entity": "phone"})
    self.assertIn("Details -> {Phone, Address}.", out)

  def test_coordinate_named_controls_never_appear(self):
    edge = _edge("e1", "n1", "|||", status="VERIFIED")
    edge["action"]["x"], edge["action"]["y"] = 540, 605
    snap = _Snapshot(outgoing={"n1": ["e1"]}, edges={"e1": edge})
    out = graph_distiller.GraphDistiller().distill("n1", snap, {},
                                                   taken_edges=["e1"])
    self.assertEqual(out, "")


class AuthoritativeEdgeLabelTest(absltest.TestCase):
  """Edges the model itself took must be nameable, or nothing can be injected."""

  def test_control_key_names_an_authoritative_edge(self):
    edge = _edge("e1", "n1", "", status="VERIFIED")
    edge["action"] = {"action_type": "click", "x": 540, "y": 1063,
                      "control_key": "com.app:id/save|Save||android.widget.Button"}
    snap = _Snapshot(outgoing={"n1": ["e1"]}, edges={"e1": edge})
    out = graph_distiller.GraphDistiller().distill("n1", snap, {"target_entity": "phone"})
    self.assertIn("Save -> {Phone, Address}.", out)

  def test_coordinates_alone_still_yield_nothing(self):
    edge = _edge("e1", "n1", "", status="VERIFIED")
    edge["action"] = {"action_type": "click", "x": 540, "y": 1063}
    snap = _Snapshot(outgoing={"n1": ["e1"]}, edges={"e1": edge})
    self.assertEqual(graph_distiller.GraphDistiller().distill("n1", snap, {}), "")
