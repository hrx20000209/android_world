"""Tests for candidate rankers."""

from android_world.parallel_exploration.rankers import RandomRanker
from android_world.parallel_exploration.rankers import SimpleRelevanceRanker
from android_world.parallel_exploration.rankers import UiElement


def _elements():
  return [
      UiElement(text="Delete", class_name="android.widget.Button"),
      UiElement(text="Flight details", class_name="android.widget.TextView"),
      UiElement(content_desc="Open trip", class_name="android.widget.ImageView"),
  ]


def test_random_ranker_is_reproducible():
  first = RandomRanker(19).rank(_elements(), "flight")
  second = RandomRanker(19).rank(_elements(), "flight")
  assert [item.element.identity for item in first] == [
      item.element.identity for item in second
  ]


def test_relevance_prefers_token_overlap_then_role():
  ranked = SimpleRelevanceRanker().rank(_elements(), "show flight details")
  assert ranked[0].element.text == "Flight details"
  assert ranked[0].score > ranked[1].score


def test_relevance_penalizes_previously_probed_element():
  elements = [
      UiElement(text="Alpha", class_name="android.widget.TextView"),
      UiElement(text="Alpha", resource_id="second",
                class_name="android.widget.TextView"),
  ]
  ranked = SimpleRelevanceRanker(already_probed_penalty=0.2).rank(
      elements, "alpha", [elements[0].identity]
  )
  assert ranked[0].element.resource_id == "second"


# --- CoverageRanker: rank by what the graph does not know -------------------
#
# Prediction as an objective is falsified four ways (probe budget doubled,
# safety filter relaxed, cross-task revisit, and uniform random ranking
# scoring 57/115). Coverage is answerable from the graph itself: mean
# out-degree is 0.77, so most nodes cannot say what is on them at all.

from android_world.parallel_exploration.rankers import CoverageRanker


def _el(text="", cls="android.widget.Button", rid="", desc="", bounds=(0, 0, 10, 10)):
  return UiElement(text=text, content_desc=desc, resource_id=rid,
                   class_name=cls, bounds=bounds, clickable=True)


def test_unexplored_control_outranks_one_the_graph_already_has_an_edge_for():
  known = _el(text="Delete", rid="app:id/del")
  fresh = _el(text="Share", rid="app:id/share")
  ranked = CoverageRanker(
      known_control_keys={CoverageRanker.control_key(known)}).rank(
          [known, fresh], "")
  assert ranked[0].element is fresh


def test_a_control_that_cannot_be_named_still_ranks_below_one_that_can():
  """An edge on an unnameable control can never become a fact.

  _action_label renders it as "the control at (540,605)" and the distiller
  rejects that, so probing it adds out-degree the distiller cannot use.
  """
  unnameable = _el(cls="android.widget.ImageButton")
  named = _el(text="Share", rid="app:id/share")
  ranked = CoverageRanker().rank([unnameable, named], "")
  assert ranked[0].element is named


def test_already_probed_this_round_is_pushed_down():
  a = _el(text="Share", rid="app:id/share")
  b = _el(text="Rename", rid="app:id/rename")
  ranked = CoverageRanker().rank([a, b], "", probed_ids=[a.identity])
  assert ranked[0].element is b


def test_control_key_drops_bounds_so_it_matches_the_graph_edge_key():
  """The graph keys edges on identity minus bounds; a moved control is one
  control, and comparing with bounds would call every scroll a new element."""
  a = _el(text="Share", rid="app:id/share", bounds=(0, 0, 10, 10))
  moved = _el(text="Share", rid="app:id/share", bounds=(0, 40, 10, 50))
  assert CoverageRanker.control_key(a) == CoverageRanker.control_key(moved)
  ranked = CoverageRanker(
      known_control_keys={CoverageRanker.control_key(a)}).rank([moved], "")
  assert ranked[0].score < CoverageRanker().rank([moved], "")[0].score


def test_ranking_is_deterministic_for_equal_scores():
  a, b = _el(text="A", rid="x"), _el(text="B", rid="y")
  first = [r.element.text for r in CoverageRanker().rank([a, b], "")]
  again = [r.element.text for r in CoverageRanker().rank([a, b], "")]
  assert first == again == ["A", "B"]


# --- the walked-path term ----------------------------------------------------
#
# The labels-only ranker is a measured result (probes -37..-45% over two
# suites), so the walked-path term defaults to weight 0 and every test below
# either leaves it off or turns it on explicitly.

from android_world.parallel_exploration import semantic_service
from android_world.parallel_exploration.rankers import GraphKeywordRanker


class _FakeEncoder:
  """Returns a fixed score per (query, candidate text) pair."""

  def __init__(self, table):
    self.table = table
    self.calls = []

  def __call__(self, port, queries, candidates):
    self.calls.append(list(queries))
    return [[self.table.get((q, c), 0.0) for c in candidates] for q in queries]


def _install(monkeypatch, table):
  fake = _FakeEncoder(table)
  monkeypatch.setattr(semantic_service, "query_many", fake)
  return fake


def _two():
  return [UiElement(text="Onward", class_name="android.widget.Button"),
          UiElement(text="Homeward", class_name="android.widget.Button")]


def test_a_candidate_that_reads_like_a_visited_screen_is_pushed_down(monkeypatch):
  _install(monkeypatch, {
      ("find the file", "Onward"): 0.5,
      ("find the file", "Homeward"): 0.6,
      ("A file list with Documents.", "Homeward"): 0.9,
      ("A file list with Documents.", "Onward"): 0.0,
  })
  ranked = GraphKeywordRanker(
      base=SimpleRelevanceRanker(), need_text="find the file",
      visited_summaries=["A file list with Documents."],
      visited_discount=0.5).rank(_two(), "find the file")
  assert ranked[0].element.text == "Onward"    # 0.5 vs 0.6 - 0.45


def test_the_walked_path_costs_nothing_at_the_default_weight(monkeypatch):
  """Weight 0 must leave the measured labels-only ranker bit-for-bit alone."""
  fake = _install(monkeypatch, {("find the file", "Homeward"): 0.6,
                                ("find the file", "Onward"): 0.5})
  ranked = GraphKeywordRanker(
      base=SimpleRelevanceRanker(), need_text="find the file",
      visited_summaries=["A file list with Documents."]).rank(
          _two(), "find the file")
  assert ranked[0].element.text == "Homeward"
  # And the summaries are not even sent - the encoder is paid per string.
  assert fake.calls == [["find the file"]]


def test_known_labels_and_the_walked_path_are_discounted_separately(monkeypatch):
  """One round trip, two penalties, each with its own weight."""
  fake = _install(monkeypatch, {
      ("need", "Onward"): 1.0, ("need", "Homeward"): 1.0,
      ("Rename", "Onward"): 1.0,                       # known-label penalty
      ("A visited screen.", "Homeward"): 1.0,          # walked-path penalty
  })
  ranker = GraphKeywordRanker(
      base=SimpleRelevanceRanker(), need_text="need",
      known_labels=["Rename"], discount=0.9,
      visited_summaries=["A visited screen."], visited_discount=0.2)
  ranked = ranker.rank(_two(), "need")
  assert fake.calls == [["need", "Rename", "A visited screen."]]
  assert ranked[0].element.text == "Homeward"          # 1 - 0.2 beats 1 - 0.9
  assert [round(s[-1], 2) for s in ranker.last_scores] == [0.8, 0.1]


def test_the_score_breakdown_is_recorded_for_the_event_log(monkeypatch):
  """"Why was this element probed" was unanswerable while scores were dropped."""
  _install(monkeypatch, {("need", "Onward"): 0.7, ("need", "Homeward"): 0.2,
                         ("Rename", "Onward"): 0.4})
  ranker = GraphKeywordRanker(base=SimpleRelevanceRanker(), need_text="need",
                              known_labels=["Rename"], discount=0.5)
  ranker.rank(_two(), "need")
  top = ranker.last_scores[0]
  assert top[0] == "Onward"
  assert (round(top[1], 2), round(top[2], 2), round(top[3], 2)) == (0.7, 0.4, 0.0)
  assert round(top[4], 2) == 0.5


# --- letting the small model pick the control --------------------------------
#
# The previous attempt at this failed on positional bias (2/11 against uniform
# random's 10/11). These pin the properties that make the failure detectable
# rather than silent: a refusal must leave the base order untouched, and the
# reply must resolve to a control actually on screen.

from android_world.parallel_exploration.rankers import LlmChoiceRanker
from android_world.parallel_exploration.rankers import a11y_label


def _els(*specs):
  out = []
  for spec in specs:
    if isinstance(spec, str):
      out.append(UiElement(text=spec, clickable=True))
    else:
      out.append(UiElement(clickable=True, **spec))
  return out


def test_a11y_label_joins_the_description_and_the_value():
  assert a11y_label(UiElement(content_desc="Departure airport/city",
                              text="Hong Kong")) == "Departure airport/city Hong Kong"


def test_a11y_label_does_not_repeat_an_identical_text_and_description():
  assert a11y_label(UiElement(content_desc="Search", text="Search")) == "Search"


def test_a11y_label_falls_back_to_the_resource_id_for_an_icon():
  assert a11y_label(UiElement(resource_id="com.app:id/btn_record_stop")) == (
      "btn record stop")


def test_a_refusal_leaves_the_base_order_bit_for_bit(monkeypatch):
  """An optional model must never be able to change or abort a probe round."""
  from android_world.parallel_exploration import semantic_service
  monkeypatch.setattr(semantic_service, "query_choose",
                      lambda *a, **k: ("", ""))
  base = SimpleRelevanceRanker()
  elements = _els("Search", "Cancel", "Settings")
  ranker = LlmChoiceRanker(base=base)
  assert ([r.element for r in ranker.rank(elements, "find flights")]
          == [r.element for r in base.rank(elements, "find flights")])
  assert ranker.last_choice == ""


def test_a_chosen_label_is_promoted_to_rank_zero(monkeypatch):
  from android_world.parallel_exploration import semantic_service
  monkeypatch.setattr(semantic_service, "query_choose",
                      lambda *a, **k: ("Settings", "Settings"))
  elements = _els("Search", "Cancel", "Settings")
  ranked = LlmChoiceRanker(base=SimpleRelevanceRanker()).rank(elements, "open settings")
  assert ranked[0].element.text == "Settings"
  assert ranked[0].rank == 0
  assert [r.rank for r in ranked] == list(range(len(ranked)))


def test_a_label_the_screen_does_not_carry_is_ignored(monkeypatch):
  """The model can generate a plausible control that is simply not there."""
  from android_world.parallel_exploration import semantic_service
  monkeypatch.setattr(semantic_service, "query_choose",
                      lambda *a, **k: ("Submit", "Submit"))
  base = SimpleRelevanceRanker()
  elements = _els("Search", "Cancel")
  ranker = LlmChoiceRanker(base=base)
  assert ([r.element for r in ranker.rank(elements, "go")]
          == [r.element for r in base.rank(elements, "go")])


def test_the_candidate_order_sent_to_the_model_is_shuffled(monkeypatch):
  """Shuffling is what makes a position-biased model measurably no better
  than random, rather than accidentally aligned with the base ranker."""
  from android_world.parallel_exploration import semantic_service
  sent = []
  monkeypatch.setattr(semantic_service, "query_choose",
                      lambda port, task, prog, labels, **k: (sent.append(list(labels)), ("", ""))[1])
  elements = _els(*[f"Item {i}" for i in range(12)])
  LlmChoiceRanker(base=SimpleRelevanceRanker()).rank(elements, "go")
  base_order = [a11y_label(r.element)
                for r in SimpleRelevanceRanker().rank(elements, "go")][:12]
  assert sorted(sent[0]) == sorted(base_order)
  assert sent[0] != base_order


def test_duplicate_labels_are_offered_once(monkeypatch):
  """A name that means two elements cannot be resolved back to one."""
  from android_world.parallel_exploration import semantic_service
  sent = []
  monkeypatch.setattr(semantic_service, "query_choose",
                      lambda port, task, prog, labels, **k: (sent.append(list(labels)), ("", ""))[1])
  elements = _els("Delete", "Delete", "Keep")
  LlmChoiceRanker(base=SimpleRelevanceRanker()).rank(elements, "go")
  assert sorted(sent[0]) == ["Delete", "Keep"]


def test_a_screen_with_one_nameable_control_is_not_worth_a_call(monkeypatch):
  from android_world.parallel_exploration import semantic_service
  calls = []
  monkeypatch.setattr(semantic_service, "query_choose",
                      lambda *a, **k: (calls.append(1), ("", ""))[1])
  LlmChoiceRanker(base=SimpleRelevanceRanker()).rank(
      _els("Search", {"class_name": "android.view.View"}), "go")
  assert not calls


def test_the_same_screen_is_only_paid_for_once(monkeypatch):
  """`_pick` re-ranks for every probe the round tries; the screen has not
  changed between those calls, and paying again ate the whole window."""
  from android_world.parallel_exploration import semantic_service
  calls = []
  monkeypatch.setattr(
      semantic_service, "query_choose",
      lambda *a, **k: (calls.append(1), ("Settings", "Settings"))[1])
  ranker = LlmChoiceRanker(base=SimpleRelevanceRanker())
  elements = _els("Search", "Cancel", "Settings")
  ranker.rank(elements, "go")
  ranker.rank(elements, "go")
  # `_pick` re-ranks with the rejected candidate removed; that is the same
  # screen, and paying again for it emptied the whole exploration budget.
  ranker.rank(elements[1:], "go")
  assert len(calls) == 1
  assert ranker.last_choice == "Settings"


def test_a_different_screen_is_a_different_question(monkeypatch):
  from android_world.parallel_exploration import semantic_service
  calls = []
  monkeypatch.setattr(
      semantic_service, "query_choose",
      lambda port, task, prog, labels, **k: (calls.append(list(labels)), (labels[0], labels[0]))[1])
  # One ranker instance is one round on one screen; a different screen is a
  # different round and therefore a different instance.
  LlmChoiceRanker(base=SimpleRelevanceRanker()).rank(_els("Search", "Cancel"), "go")
  LlmChoiceRanker(base=SimpleRelevanceRanker()).rank(_els("Save", "Discard"), "go")
  assert len(calls) == 2


def test_a_refusal_is_cached_too(monkeypatch):
  """Re-asking a screen the model already declined is the same wasted call."""
  from android_world.parallel_exploration import semantic_service
  calls = []
  monkeypatch.setattr(semantic_service, "query_choose",
                      lambda *a, **k: (calls.append(1), ("", ""))[1])
  ranker = LlmChoiceRanker(base=SimpleRelevanceRanker())
  elements = _els("Search", "Cancel")
  ranker.rank(elements, "go")
  ranker.rank(elements, "go")
  assert len(calls) == 1


def test_the_round_record_survives_a_depleted_final_rerank(monkeypatch):
  """One instance is one round. `_pick` re-ranks with rejected candidates
  removed, and the last call often has fewer than two nameable controls left -
  so per-call resets made a round that DID ask read as one that never did."""
  from android_world.parallel_exploration import semantic_service
  monkeypatch.setattr(semantic_service, "query_choose",
                      lambda *a, **k: ("Settings", "Settings"))
  ranker = LlmChoiceRanker(base=SimpleRelevanceRanker())
  ranker.rank(_els("Search", "Cancel", "Settings"), "go")
  assert ranker.last_choice == "Settings"
  assert len(ranker.last_candidates) == 3
  ranker.rank(_els("Search"), "go")          # everything else rejected
  assert ranker.last_choice == "Settings"
  assert len(ranker.last_candidates) == 3
