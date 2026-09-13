

def test_reasoning_prior_reads_what_the_model_said_it_wants():
  """Eq.(4): K_target / A_expect / K_risk / U_miss from the model's own text."""
  from android_world.parallel_exploration.information import parse_reasoning_prior
  output = (
      "<THINK> I see the list of files in Markor. The task is to delete the "
      "note named 'bold_king_edited'. I need to long-press the file to reveal "
      "the delete option in the menu. </THINK>\n"
      "explain:I need to long-press the 'bold_king_edited' note to select it"
  )
  prior = parse_reasoning_prior(output, "some unrelated global goal")
  assert prior.source == "reasoning_prior"
  assert "bold_king_edited" in prior.required_information_slots
  assert "menu" in prior.expected_affordances
  assert "delete" in prior.risk_constraints
  # The subgoal is this step's intent, not the episode-wide task text.
  assert "long-press" in prior.current_subgoal
  assert "unrelated global goal" not in prior.current_subgoal


def test_reasoning_prior_falls_back_when_there_is_no_reasoning_text():
  """A model emitting a bare tool call still has to work, just without P."""
  from android_world.parallel_exploration.information import parse_reasoning_prior
  bare = '<tool_call>{"name":"mobile_use","arguments":{"action":"click"}}</tool_call>'
  prior = parse_reasoning_prior(bare, "find contact")
  assert prior.source != "reasoning_prior"
  assert prior.current_subgoal == "find contact"


def test_reasoning_prior_distinguishes_consecutive_steps():
  """The point of P: the task goal is constant, the prior is not."""
  from android_world.parallel_exploration.information import parse_reasoning_prior
  goal = "Delete the note named bold_king_edited in Markor"
  step1 = parse_reasoning_prior("<THINK>I should open the Markor app first.</THINK>", goal)
  step2 = parse_reasoning_prior(
      "<THINK>Markor is open. I need to find the search field.</THINK>", goal)
  assert step1.current_subgoal != step2.current_subgoal
  assert "search" in step2.expected_affordances
  assert "search" not in step1.expected_affordances


# --- onclick form: AutoDroid's per-control annotation (MobiCom'24 S3.2.2) ----
#
# The block form must contain at least one VERIFIED/OBSERVED fact or it emits
# nothing; measured 2026-09-09 it fired 10 times per 1010 steps while 31% of
# steps had a candidate fact available. The onclick form drops the block-level
# gate and keeps every per-fact filter.

from android_world.parallel_exploration.graph_distiller import GraphDistiller
from android_world.parallel_exploration.graph_distiller import GraphFact


def _fact(kind, label="Delete", labels=("Trash",)):
  return GraphFact(fact_type=kind, action_label=label, certainty="Verified",
                   utility_score=1.0, evidence_labels=tuple(labels),
                   source_edge_id="e1")


def test_onclick_form_is_bare_lines_while_the_block_form_is_a_block():
  """The two forms differ in framing as well as in gating: the block reads as
  a memo the model may weigh, a bare per-control line reads as a property of
  that control - which is what AutoDroid attaches to the element itself."""
  facts = [_fact("VERIFIED", "Delete"), _fact("OBSERVED", "Share")]
  block = GraphDistiller().render(facts, {})
  assert block.startswith("[Memory]")
  onclick = "\n".join(f.render() for f in facts
                       if f.fact_type in ("VERIFIED", "OBSERVED"))
  assert not onclick.startswith("[Memory]")
  assert all(f.render() in onclick for f in facts)


def test_onclick_form_drops_negative_only_facts():
  """Negatives are supporting detail for a positive finding, never the whole
  message - a list of dead ends read as a menu and cost four tasks."""
  facts = [_fact("VERIFIED", "Delete"), _fact("NO_RELEVANT_EVIDENCE", "Sync")]
  kept = [f for f in facts if f.fact_type in ("VERIFIED", "OBSERVED")]
  assert len(kept) == 1 and kept[0].action_label == "Delete"


# --- decision entropy is a claim about the model, not about the app ---------
#
# H is read by the skip gate ("the answer here has always been the same").
# Probe edges and edges seeded from cross-task memory say a control exists and
# leads somewhere, with no model having chosen it; counting them made every
# probe and every remembered screen push the gate shut. Measured 2026-09-10:
# mean finite H ordered every arm's skip count (0.224->14, 0.246->13,
# 0.295->7, 0.324->2).

from android_world.parallel_exploration import belief_graph as _bg
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
import math


def _graph_with(executed: int, speculative: int):
  g = ProgressiveBeliefGraph("t")
  g.upsert_node(_bg.GraphNode(node_id="n1", activity="a", package="p",
                              visual_signature="", structural_signature="",
                              layout_signature="L1"))
  for i in range(executed + speculative):
    dst = f"d{i}"
    g.upsert_node(_bg.GraphNode(node_id=dst, activity="a", package="p",
                                visual_signature="", structural_signature="",
                                layout_signature=f"L{i}"))
    e = g.add_speculative_transition(
        "n1", {"action_type": "click", "control_key": f"c{i}"}, dst,
        path_probability=0.5, confidence=0.1, expected_information_gain=0.0,
        risk_level="LOW", exploration_cost=0.0, rollback_success=True)
    if i < executed:
      g.record_execution_verification(e.edge_id, True)
  return g


def test_seeded_and_probe_edges_no_longer_raise_entropy(monkeypatch):
  monkeypatch.setattr(_bg, "ENTROPY_OVER_EXECUTED_ONLY", True)
  one_executed = _graph_with(executed=1, speculative=4)
  assert one_executed.recompute_decision_entropy("n1") == 0.0


def test_historical_behaviour_is_preserved_by_default(monkeypatch):
  monkeypatch.setattr(_bg, "ENTROPY_OVER_EXECUTED_ONLY", False)
  one_executed = _graph_with(executed=1, speculative=4)
  assert one_executed.recompute_decision_entropy("n1") > 0.5


def test_a_screen_with_nothing_executed_still_looks_undecided(monkeypatch):
  """An unmapped screen must never look decided, or skip fires on no evidence."""
  monkeypatch.setattr(_bg, "ENTROPY_OVER_EXECUTED_ONLY", True)
  none_executed = _graph_with(executed=0, speculative=3)
  h = none_executed.recompute_decision_entropy("n1")
  assert h > 0.5 and h != math.inf   # falls back to all viable edges


def test_two_different_executed_actions_still_look_undecided(monkeypatch):
  monkeypatch.setattr(_bg, "ENTROPY_OVER_EXECUTED_ONLY", True)
  two = _graph_with(executed=2, speculative=3)
  assert two.recompute_decision_entropy("n1") > 0.5


# --- semantic consistency check on the skip gate -----------------------------
#
# Every existing skip condition is structural (revisited screen, one action
# ever chosen there, executed twice); none asks whether the remembered action
# is what the task needs now. Measured skip hit rate is 57-85%.

from android_world.parallel_exploration import graph_distiller as _gd


def test_skip_is_not_blocked_when_the_control_cannot_be_named(monkeypatch):
    """An unnameable control leaves the structural licence exactly as it was -
    that licence is the one mechanism with measured value (48/48 skips)."""
    monkeypatch.setattr(_gd, "SKIP_SEMANTIC_PORT", 8766)
    edge = {"action": {"x": 5, "y": 6}}
    assert _gd._semantic_skip_block(edge, {"target_entity": "delete note"}) is None


def test_skip_is_not_blocked_when_there_is_no_stated_need(monkeypatch):
    monkeypatch.setattr(_gd, "SKIP_SEMANTIC_PORT", 8766)
    edge = {"action": {"control_key": "id/del||Delete|Button"}}
    assert _gd._semantic_skip_block(edge, {}) is None
    assert _gd._semantic_skip_block(edge, None) is None


def test_skip_is_not_blocked_when_the_encoder_is_unreachable(monkeypatch):
    """An optional model must never be able to change a decision by dying."""
    monkeypatch.setattr(_gd, "SKIP_SEMANTIC_PORT", 9)   # nothing listens there
    edge = {"action": {"control_key": "id/del||Delete|Button"}}
    assert _gd._semantic_skip_block(
        edge, {"target_entity": "delete the note"}) is None


# --- a fact names where a control leads, not what is written there ----------
#
# "Pause -> {05, Start}" lists labels the model reads off the next screenshot
# anyway. Which screen the control leads to is the one thing the current
# screenshot cannot show, so a destination description replaces the labels.

def test_a_destination_description_replaces_the_label_list():
  f = GraphFact(fact_type="VERIFIED", action_label="Delete", source_edge_id="e",
                evidence_labels=("CANCEL", "OK"),
                destination_summary="A confirmation dialog asking to delete.")
  assert f.render() == "Delete -> A confirmation dialog asking to delete."


def test_labels_are_still_used_when_the_destination_has_no_description():
  f = GraphFact(fact_type="VERIFIED", action_label="Delete", source_edge_id="e",
                evidence_labels=("CANCEL", "OK"))
  assert f.render() == "Delete -> {CANCEL, OK}."


def test_a_destination_nobody_described_still_reads_as_a_transition():
  f = GraphFact(fact_type="VERIFIED", action_label="Delete", source_edge_id="e")
  assert f.render() == "Delete leads to another screen."
