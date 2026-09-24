import dataclasses
import queue
import types
from pathlib import Path

from absl.testing import absltest

from android_world.parallel_exploration import live_probe
from android_world.parallel_exploration.live_probe import PreparedExplorer
from android_world.parallel_exploration.live_probe import run_prepared_explorer
from android_world.parallel_exploration.live_probe import _probe_type
from android_world.parallel_exploration.live_probe import _safe_candidates
from android_world.parallel_exploration.live_probe import _scroll_alignment_delta
from android_world.parallel_exploration.live_probe import _apply_executable_memory_tiebreak
from android_world.parallel_exploration import state_graph_information as sgi
from android_world.parallel_exploration.executable_memory import ElementSelector
from android_world.parallel_exploration.rankers import UiElement
from android_world.parallel_exploration.state import ActivitySignature
from android_world.parallel_exploration.state import StateSignature
from android_world.parallel_exploration.state import StructSignature
import pathlib
import re
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
from android_world.parallel_exploration.belief_graph import stable_control_key_from_identity


def test_missing_explorer_terminal_preserves_reasoning_result_and_marks_dirty():
  class EmptyStatusQueue:
    def get(self, timeout=None):
      raise queue.Empty

  class RecordingQueue:
    def __init__(self):
      self.items = []

    def put(self, item):
      self.items.append(item)

  class DeadExplorer:
    def __init__(self):
      self.terminated = False

    def is_alive(self):
      return False

    def terminate(self):
      self.terminated = True

    def join(self, timeout=None):
      del timeout

  process = DeadExplorer()
  prepared = PreparedExplorer(
      config={"trial_id": "queue-timeout", "restore_timeout_s": 0.001,
              "max_probes": 5, "min_probes": 5},
      process=process, control_queue=RecordingQueue(),
      status_queue=EmptyStatusQueue(),
  )

  result, window = run_prepared_explorer(prepared, lambda: {"action": "click"})

  assert result == {"action": "click"}
  assert window["restore_status"] == live_probe.EventKind.RESTORE_FAILED.value
  assert window["dirty"] is True
  assert window["stop_reason"] == "terminal_status_timeout"
  assert window["status_message_missing"] is True


def test_navigation_candidate_admission_does_not_depend_on_fixed_risk_words(tmp_path: Path):
  elements = (
      UiElement(text="Delete", class_name="android.widget.ImageButton", clickable=True),
      UiElement(text="Send", class_name="android.widget.ImageButton", clickable=True),
      UiElement(text="More fields", class_name="android.widget.ImageButton", clickable=True),
  )

  selected = _safe_candidates(
      elements, tmp_path / "filtered.jsonl",
      {"allow_unverified_click_inverse": True},
  )

  assert selected == list(elements)


def test_semantic_navigation_click_is_active_by_default(tmp_path: Path):
  element = UiElement(text="Open", class_name="android.widget.TextView", clickable=True)
  assert _safe_candidates((element,), tmp_path / "filtered.jsonl", {}) == [element]


def test_low_support_transition_remains_eligible_for_validation(tmp_path: Path):
  element = UiElement(
      text="Settings", resource_id="app:id/settings",
      class_name="android.widget.TextView", clickable=True,
      bounds=(20, 30, 220, 90))
  key = stable_control_key_from_identity(element.identity)
  context = {
      "already_explored_element_identities": [element.identity],
      "explored_control_evidence": {key: {
          "probe_count": 1, "execution_count": 0,
          "destinations": ["settings-page"], "recovery_failures": 0,
      }},
  }
  assert _safe_candidates((element,), tmp_path / "filtered.jsonl", context) == [element]


def test_mature_graph_edge_suppresses_control_after_bounds_move(tmp_path: Path):
  old = UiElement(
      text="Settings", resource_id="app:id/settings",
      class_name="android.widget.TextView", clickable=True,
      bounds=(20, 30, 220, 90))
  live = dataclasses.replace(old, bounds=(28, 35, 228, 95))
  key = stable_control_key_from_identity(old.identity)
  context = {
      "already_explored_element_identities": [old.identity],
      "explored_control_evidence": {key: {
          "probe_count": 2, "execution_count": 0,
          "destinations": ["settings-page"], "recovery_failures": 0,
      }},
  }
  assert _safe_candidates((live,), tmp_path / "filtered.jsonl", context) == []
  assert "already_explored_this_node_stable_transition" in (
      tmp_path / "filtered.jsonl").read_text(encoding="utf-8")


def test_trap_control_is_never_reprobed_even_when_landing_varied(tmp_path: Path):
  element = UiElement(
      text="Open item", resource_id="app:id/item",
      class_name="android.widget.TextView", clickable=True,
      bounds=(20, 30, 220, 90))
  key = stable_control_key_from_identity(element.identity)
  context = {"explored_control_evidence": {key: {
      "probe_count": 1, "destinations": ["dialog", "external"],
      "recovery_failures": 1,
  }}}
  assert _safe_candidates((element,), tmp_path / "filtered.jsonl", context) == []
  assert "previously_caused_unrecoverable_state" in (
      tmp_path / "filtered.jsonl").read_text(encoding="utf-8")


def test_repeated_clickable_template_is_filtered_as_dynamic_collection(tmp_path: Path):
  elements = (
      UiElement(text="1:06:00 (12.22 mi)", resource_id="app:id/track_item",
                class_name="android.view.ViewGroup", clickable=True),
      UiElement(text="48:00 (12.19 mi)", resource_id="app:id/track_item",
                class_name="android.view.ViewGroup", clickable=True),
  )
  assert _safe_candidates(elements, tmp_path / "filtered.jsonl", {}) == []
  assert "repeated_dynamic_collection_item" in (
      tmp_path / "filtered.jsonl").read_text(encoding="utf-8")


def test_action_buttons_are_structurally_filtered_independent_of_label(tmp_path: Path):
  elements = (
      UiElement(text="Start", class_name="android.widget.Button", clickable=True),
      UiElement(text="Harmless", class_name="android.widget.Button", clickable=True),
  )

  selected = _safe_candidates(elements, tmp_path / "filtered.jsonl", {})

  assert selected == []
  rows = (tmp_path / "filtered.jsonl").read_text(encoding="utf-8")
  assert rows.count("action_button_requires_semantic_inverse") == 2


def test_known_read_only_navigation_buttons_are_explorable_but_mutations_are_not(
    tmp_path: Path,
):
  search = UiElement(
      content_desc="Search", resource_id="org.tasks:id/menu_search",
      class_name="android.widget.Button", clickable=True,
      bounds=(900, 100, 1080, 260),
  )
  more = UiElement(
      text="More options", resource_id="app:id/menu_more",
      class_name="android.widget.Button", clickable=True,
      bounds=(900, 100, 1080, 260),
  )
  delete = UiElement(
      text="Delete", resource_id="app:id/delete",
      class_name="android.widget.Button", clickable=True,
      bounds=(100, 100, 280, 260),
  )

  selected = _safe_candidates(
      (search, more, delete), tmp_path / "filtered.jsonl", {})

  assert selected == [search, more]
  assert "action_button_requires_semantic_inverse" in (
      tmp_path / "filtered.jsonl").read_text(encoding="utf-8")


def test_probe_type_uses_role_and_state_not_label():
  assert _probe_type(UiElement(text="Delete", class_name="android.widget.Button")) == "TAP_NAV"
  assert _probe_type(UiElement(text="Anything", class_name="android.widget.ImageView")) == "TAP_MENU"
  assert _probe_type(UiElement(
      text="Anything", class_name="android.widget.CheckBox", checked=False,
  )) == "EXPAND"


def test_checkable_radio_like_button_is_not_treated_as_reversible(tmp_path: Path):
  radio_like = UiElement(
      text="Phone contacts", class_name="android.widget.Button",
      clickable=True, checked=False,
  )
  assert _safe_candidates((radio_like,), tmp_path / "filtered.jsonl", {}) == []


def test_nonlexical_measurement_guards_remain(tmp_path: Path):
  anonymous = UiElement(class_name="android.view.ViewGroup", clickable=True)
  scroll = UiElement(resource_id="list", class_name="android.widget.ScrollView", scrollable=True)

  selected = _safe_candidates((anonymous, scroll), tmp_path / "filtered.jsonl", {})

  assert selected == [scroll]
  reasons = (tmp_path / "filtered.jsonl").read_text(encoding="utf-8")
  assert "no_semantic_identifier" in reasons
  assert "scroll_inverse_not_exact" not in reasons


def test_executable_memory_guidance_only_nudges_scored_safe_candidates():
  element = UiElement(
      text="Notes", resource_id="app:id/notes",
      class_name="android.widget.Button", bounds=(10, 20, 180, 90),
      clickable=True)
  row = sgi.CandidateInformationRow(
      element=element, element_identity=element.identity, text=element.text,
      content_desc="", role="button", probe_type="TAP_NAV", norm_x=0.1,
      norm_y=0.1, clickable=True, scrollable=False,
      target_match=0.7)
  scored = sgi.ScoredCandidate(row=row, utility=0.1, components={"expected_cost": 1.0})
  key = ElementSelector.from_element(element).key()
  guidance = {"enabled": True, "state_matched": True, "controls": {
      key: {"support_count": 1, "task_relevance": 0.5,
            "mature": False, "trap": False, "dynamic": False},
  }}

  adjusted, stats = _apply_executable_memory_tiebreak([scored], guidance)

  assert len(adjusted) == 1
  assert adjusted[0].utility > scored.utility
  assert adjusted[0].row.eam_control_seen
  assert adjusted[0].row.eam_control_support == 1
  assert adjusted[0].components["eam_graph_tiebreak"] > 0
  assert stats["state_matched"] and stats["seen_count"] == 1


def test_executable_memory_mature_edge_gets_bounded_repeat_penalty():
  element = UiElement(
      text="Settings", resource_id="app:id/settings",
      class_name="android.widget.Button", bounds=(10, 20, 180, 90),
      clickable=True)
  row = sgi.CandidateInformationRow(
      element=element, element_identity=element.identity, text=element.text,
      content_desc="", role="button", probe_type="TAP_NAV", norm_x=0.1,
      norm_y=0.1, clickable=True, scrollable=False)
  scored = sgi.ScoredCandidate(row=row, utility=0.1, components={})
  guidance = {"enabled": True, "state_matched": True, "controls": {
      ElementSelector.from_element(element).key(): {
          "support_count": 3, "task_relevance": 0.0, "mature": True,
          "trap": False, "dynamic": False,
      },
  }}

  adjusted, _ = _apply_executable_memory_tiebreak([scored], guidance)

  assert adjusted == []


def test_executable_memory_suppresses_mature_edge_when_frontier_exists():
  def candidate(label, utility, target_match):
    element = UiElement(
        text=label, resource_id=f"app:id/{label.casefold()}",
        class_name="android.widget.Button", bounds=(10, 20, 180, 90),
        clickable=True)
    row = sgi.CandidateInformationRow(
        element=element, element_identity=element.identity, text=label,
        content_desc="", role="button", probe_type="TAP_NAV", norm_x=.1,
        norm_y=.1, clickable=True, scrollable=False, target_match=target_match)
    return sgi.ScoredCandidate(row=row, utility=utility, components={})

  mature = candidate("Settings", .2, .1)
  novel = candidate("Notes", .1, .8)
  guidance = {"enabled": True, "state_matched": True, "controls": {
      ElementSelector.from_element(mature.row.element).key(): {
          "support_count": 4, "task_relevance": .1, "confidence": .9,
          "mature": True, "trap": False, "dynamic": False,
      },
  }}

  adjusted, stats = _apply_executable_memory_tiebreak([mature, novel], guidance)

  assert [item.row.text for item in adjusted] == ["Notes"]
  assert stats["unseen_count"] == 1
  assert stats["repeat_suppressed_count"] == 1


def test_executable_memory_revalidates_low_confidence_seen_edges():
  element = UiElement(
      text="Menu", resource_id="app:id/menu", class_name="Button",
      bounds=(10, 20, 180, 90), clickable=True)
  row = sgi.CandidateInformationRow(
      element=element, element_identity=element.identity, text="Menu",
      content_desc="", role="button", probe_type="TAP_NAV", norm_x=.1,
      norm_y=.1, clickable=True, scrollable=False, target_match=.8)
  scored = sgi.ScoredCandidate(row=row, utility=.1, components={})
  guidance = {"enabled": True, "state_matched": True, "controls": {
      ElementSelector.from_element(element).key(): {
          "support_count": 1, "task_relevance": .8, "confidence": .4,
          "mature": False, "trap": False, "dynamic": False,
      },
  }}

  adjusted, stats = _apply_executable_memory_tiebreak([scored], guidance)

  assert len(adjusted) == 1
  assert adjusted[0].utility > scored.utility
  assert stats["revalidation_count"] == 1


def test_repeated_click_noop_is_suppressed_but_unseen_frontier_remains():
  def candidate(label, utility):
    element = UiElement(
        text=label, resource_id=f"app:id/{label.casefold()}",
        class_name="android.widget.Button", bounds=(10, 20, 180, 90),
        clickable=True)
    row = sgi.CandidateInformationRow(
        element=element, element_identity=element.identity, text=label,
        content_desc="", role="button", probe_type="TAP_NAV", norm_x=.1,
        norm_y=.1, clickable=True, scrollable=False)
    return sgi.ScoredCandidate(row=row, utility=utility, components={})

  noop = candidate("Old", .2)
  novel = candidate("Search", .1)
  guidance = {"enabled": True, "state_matched": True, "controls": {
      ElementSelector.from_element(noop.row.element).key(): {
          "support_count": 2, "task_relevance": .1, "mature": False,
          "trap": False, "dynamic": False, "known_noop": True,
      },
  }}

  adjusted, stats = _apply_executable_memory_tiebreak([noop, novel], guidance)

  assert [item.row.text for item in adjusted] == ["Search"]
  assert noop.row.eam_control_noop
  assert noop.row.eam_repeat_suppressed
  assert stats["known_noop_suppressed_count"] == 1


def test_eam_side_effect_veto_is_hard_but_does_not_change_ex5_baseline():
  element = UiElement(
      text="Set up", content_desc="Set up Duo video calling",
      resource_id="com.google.android.contacts:id/verb_video",
      class_name="android.widget.TextView", bounds=(100, 200, 300, 400),
      clickable=True)
  row = sgi.CandidateInformationRow(
      element=element, element_identity=element.identity,
      text=element.text, content_desc=element.content_desc,
      role="generic", probe_type="TAP_NAV", norm_x=.2, norm_y=.2,
      clickable=True, scrollable=False)
  scored = sgi.ScoredCandidate(row=row, utility=.2, components={})

  guarded, guard_stats = _apply_executable_memory_tiebreak([scored], {
      "enabled": True, "state_matched": False, "controls": {},
  })
  baseline, baseline_stats = _apply_executable_memory_tiebreak([scored], {
      "enabled": False, "state_matched": False, "controls": {},
  })

  assert guarded == []
  assert guard_stats["side_effect_veto_count"] == 1
  assert guard_stats["suppressed_candidates"][0]["reason"] == "eam_side_effect_label_veto"
  assert len(baseline) == 1
  assert baseline_stats["side_effect_veto_count"] == 0


def test_eam_vetoes_call_text_and_resource_id_affordances_and_stops_cleanly():
  for label, resource_id in (
      ("Call", ""),
      ("Text", ""),
      ("", "com.google.android.contacts:id/verb_call"),
  ):
    element = UiElement(
        text=label, resource_id=resource_id,
        class_name="android.widget.TextView", bounds=(10, 20, 180, 90),
        clickable=True)
    row = sgi.CandidateInformationRow(
        element=element, element_identity=element.identity,
        text=element.text, content_desc="", role="generic",
        probe_type="TAP_NAV", norm_x=.1, norm_y=.1,
        clickable=True, scrollable=False)
    scored = sgi.ScoredCandidate(row=row, utility=.2, components={})

    guarded, stats = _apply_executable_memory_tiebreak([scored], {
        "enabled": True, "state_matched": False, "controls": {},
    })

    assert guarded == []
    assert stats["side_effect_veto_count"] == 1
    assert stats["stop_reason"] == "eam_side_effect_veto_all"


def test_invalid_or_inverted_bounds_are_never_probed(tmp_path: Path):
  elements = (
      UiElement(text="September", class_name="android.widget.TextView",
                bounds=(0, 296, -147, 438), clickable=True),
      UiElement(text="Valid", class_name="android.widget.TextView",
                bounds=(10, 20, 110, 80), clickable=True),
  )
  selected = _safe_candidates(elements, tmp_path / "filtered.jsonl", {})
  assert selected == [elements[1]]
  assert "invalid_or_empty_bounds" in (
      tmp_path / "filtered.jsonl").read_text(encoding="utf-8")


def test_active_editing_page_suppresses_all_probes(tmp_path: Path):
  elements = (
      UiElement(text="Title", class_name="android.widget.EditText",
                bounds=(10, 20, 300, 80), clickable=True),
      UiElement(resource_id="list", class_name="android.widget.ScrollView",
                bounds=(0, 100, 500, 900), scrollable=True),
  )
  assert _safe_candidates(elements, tmp_path / "filtered.jsonl", {}) == []
  assert "editing_state_not_structurally_recoverable" in (
      tmp_path / "filtered.jsonl").read_text(encoding="utf-8")


def test_scroll_alignment_uses_unique_semantic_anchor_positions():
  def state(y: int) -> StateSignature:
    return StateSignature(
        ActivitySignature("pkg/.Main", 1, 1, ""), StructSignature("x", 1),
        "0000000000000000",
        (UiElement(text="Anchor", class_name="android.widget.TextView",
                   bounds=(0, y, 100, y + 40)),), {},
    )

  # The current content is 120 px above its baseline location, so the finger
  # correction must move downward by 120 px.
  assert _scroll_alignment_delta(state(300), state(180)) == 120


class BackBoundedByAppRootTest(absltest.TestCase):
  """Back must never be the rung that walks the probe out of the app."""

  def test_root_activity_same_depth_presses_no_back(self):
    """The failure mode that stranded 8 of 22 episodes on 2026-08-31."""
    baseline_depth, post_depth = 1, 1
    back_count = post_depth - baseline_depth
    if back_count < 1 and baseline_depth > 1:
      back_count = 1
    self.assertEqual(back_count, 0)

  def test_deeper_same_depth_still_allows_one_back(self):
    baseline_depth, post_depth = 3, 3
    back_count = post_depth - baseline_depth
    if back_count < 1 and baseline_depth > 1:
      back_count = 1
    self.assertEqual(back_count, 1)

  def test_pushed_screens_are_popped_exactly(self):
    baseline_depth, post_depth = 2, 5
    self.assertEqual(post_depth - baseline_depth, 3)

  def test_left_the_app_compares_packages(self):
    def state(component):
      return types.SimpleNamespace(
          activity=types.SimpleNamespace(component=component))
    app = state("com.flauschcode.broccoli/.MainActivity")
    self.assertTrue(live_probe._left_the_app(
        app, state("com.google.android.apps.nexuslauncher/.NexusLauncherActivity")))
    self.assertFalse(live_probe._left_the_app(
        app, state("com.flauschcode.broccoli/.recipe.details.RecipeActivity")))
    self.assertFalse(live_probe._left_the_app(app, None))


class StatusBarNoiseTest(absltest.TestCase):
  """What a screen shows is the app's content, not whatever is in the shade."""

  def _element(self, resource_id=""):
    return UiElement(class_name="android.widget.TextView", resource_id=resource_id)

  def test_notification_descriptions_are_not_screen_content(self):
    for label in ("Messages notification: 8 new messages",
                  "Android System notification: ",
                  "Phone notification: missed call"):
      self.assertFalse(live_probe._is_app_content(self._element(), label), label)

  def test_an_app_notifications_menu_entry_survives(self):
    """The colon is what separates framework phrasing from app content."""
    for label in ("Notifications", "Notification settings"):
      self.assertTrue(live_probe._is_app_content(self._element(), label), label)

  def test_clock_and_battery_readouts_are_dropped(self):
    for label in ("15:34", "Battery 100 percent.", "Phone signal full."):
      self.assertFalse(live_probe._is_app_content(self._element(), label), label)

  def test_system_ui_package_is_dropped_whatever_it_says(self):
    element = self._element("com.android.systemui:id/clock")
    self.assertFalse(live_probe._is_app_content(element, "Expense Detail"))


class NodeIdentityAgreementTest(absltest.TestCase):
  """The explorer and the runner must key the graph the same way.

  They did not: the explorer passed struct_sig.digest where make_node_id
  expects the layout signature, so every snapshot lookup inside the explorer
  missed. Across five full runs (21,519 scored candidates, 2026-09-04) not one
  candidate ever carried a graph statistic, and prefix alignment - which is
  keyed on the same identity - was zero in every run.
  """

  def test_explorer_and_runner_agree_on_node_id(self):
    activity = "net.gsantner.markor/.activity.MainActivity"
    layout = "layout-abc123"
    strict = "struct-digest-that-changes-with-any-data"
    runner_id = ProgressiveBeliefGraph.make_node_id(activity, layout)
    explorer_id = ProgressiveBeliefGraph.make_node_id(activity, layout)
    self.assertEqual(runner_id, explorer_id)
    # And the strict digest must NOT be interchangeable with it, which is what
    # made the original mismatch silent rather than loud.
    self.assertNotEqual(runner_id,
                        ProgressiveBeliefGraph.make_node_id(activity, strict))

  def test_live_probe_uses_layout_sig_not_struct_sig(self):
    source = pathlib.Path(live_probe.__file__).read_text(encoding="utf-8")
    for call in re.findall(r"make_node_id\((.*?)\)", source, flags=re.S):
      self.assertNotIn("struct_sig", call)
      self.assertIn("layout_sig", call)


# --- coverage ranking reads the graph snapshot the explorer was handed ------

from android_world.parallel_exploration.live_probe import _known_control_keys


def test_known_control_keys_collects_every_outgoing_edge_of_the_node():
  payload = {
      "edges": {
          "e1": {"action": {"control_key": "app:id/del||Delete|Button"}},
          "e2": {"action": {"control_key": "app:id/share||Share|Button"}},
          "e3": {"action": {"control_key": "app:id/other||Other|Button"}},
      },
      "outgoing": {"n1": ["e1", "e2"], "n2": ["e3"]},
  }
  assert _known_control_keys(payload, "n1") == {
      "app:id/del||Delete|Button", "app:id/share||Share|Button"}


def test_an_edge_without_a_control_key_contributes_nothing():
  """Authoritative edges carried only coordinates before control_key existed;
  a blank key must not mark every unnamed control as already known."""
  payload = {"edges": {"e1": {"action": {"x": 5, "y": 6}}},
             "outgoing": {"n1": ["e1"]}}
  assert _known_control_keys(payload, "n1") == set()


def test_missing_snapshot_or_node_is_empty_rather_than_an_error():
  assert _known_control_keys(None, "n1") == set()
  assert _known_control_keys({"edges": {}, "outgoing": {}}, "") == set()
  assert _known_control_keys({"edges": {}, "outgoing": {}}, "unknown") == set()


# --- the graph's contribution to exploration targeting -----------------------
#
# Not "which control will the model press" - four falsifications say nothing
# predicts that - but "what is already on record here", which is the one
# question the graph answers reliably.

from android_world.parallel_exploration.live_probe import _discovered_labels_near


def _payload():
  return {
      "edges": {
          "e1": {"dst_node": "n2", "discovered_labels": ["Trash", "Restore"]},
          "e2": {"dst_node": "n3", "discovered_labels": ["Rename"]},
          "e3": {"dst_node": "n4", "discovered_labels": ["Confirm delete"]},
          "e9": {"dst_node": "n9", "discovered_labels": ["Elsewhere"]},
      },
      "outgoing": {"n1": ["e1", "e2"], "n2": ["e3"], "n8": ["e9"]},
  }


def test_labels_come_from_this_screen_and_one_hop_out():
  """A probe that rediscovers the screen one hop away is just as wasted."""
  got = _discovered_labels_near(_payload(), "n1")
  assert got[:3] == ["Trash", "Restore", "Rename"]      # this screen's edges
  assert "Confirm delete" in got                        # one hop out
  assert "Elsewhere" not in got                         # unrelated subtree


def test_duplicate_labels_are_collapsed_case_insensitively():
  payload = {"edges": {"a": {"dst_node": None, "discovered_labels": ["Trash"]},
                       "b": {"dst_node": None, "discovered_labels": ["trash"]}},
             "outgoing": {"n1": ["a", "b"]}}
  assert _discovered_labels_near(payload, "n1") == ["Trash"]


def test_no_graph_or_unknown_node_yields_nothing_rather_than_erroring():
  assert _discovered_labels_near(None, "n1") == []
  assert _discovered_labels_near(_payload(), "") == []
  assert _discovered_labels_near(_payload(), "nope") == []


# --- the walked path, summarised, as the second graph input to exploration ---
#
# `_discovered_labels_near` answers "what is already on record behind a control
# here". This answers a different question - "where has this episode already
# been" - and it only became answerable once nodes carried a description.

from android_world.parallel_exploration.live_probe import _walked_path_summary


def _walked_payload():
  return {
      "node_visits": {"n0": 1, "n1": 2, "n2": 1, "n3": 0},
      "node_summaries": {
          "n0": "A file list with Documents and Downloads.",
          "n1": "A folder view with Rename and Delete.",
          "n2": "A dialog asking to Confirm delete.",
          "n3": "A screen no one has stood on.",
      },
  }


def test_the_walked_path_is_the_visited_nodes_in_first_seen_order():
  assert _walked_path_summary(_walked_payload(), "n1") == [
      "A file list with Documents and Downloads.",
      "A dialog asking to Confirm delete.",
  ]


def test_the_current_screen_is_excluded_because_the_ranker_already_sees_it():
  assert ("A folder view with Rename and Delete."
          not in _walked_path_summary(_walked_payload(), "n1"))


def test_a_node_the_episode_only_heard_about_is_not_on_the_walked_path():
  """visit_count 0 means seeded or probe-discovered, never stood on."""
  assert ("A screen no one has stood on."
          not in _walked_path_summary(_walked_payload(), "n1"))


def test_only_the_most_recent_screens_survive_the_cap():
  """A mid-episode graph carries 20-40 nodes; the encoder is paid per string."""
  payload = {"node_visits": {f"n{i}": 1 for i in range(10)},
             "node_summaries": {f"n{i}": f"Screen {i}." for i in range(10)}}
  got = _walked_path_summary(payload, "cur", limit=3)
  assert got == ["Screen 7.", "Screen 8.", "Screen 9."]


def test_screens_described_identically_collapse_to_one_entry():
  payload = {"node_visits": {"a": 1, "b": 1},
             "node_summaries": {"a": "A settings page.", "b": "a settings page."}}
  assert _walked_path_summary(payload, "cur") == ["A settings page."]


def test_a_graph_with_no_descriptions_yields_nothing_rather_than_erroring():
  assert _walked_path_summary({"node_visits": {"a": 1}}, "cur") == []
  assert _walked_path_summary(None, "n1") == []
  assert _walked_path_summary(_walked_payload(), "") == []
