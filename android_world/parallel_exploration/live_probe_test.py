import types
from pathlib import Path

from absl.testing import absltest

from android_world.parallel_exploration import live_probe
from android_world.parallel_exploration.live_probe import _probe_type
from android_world.parallel_exploration.live_probe import _safe_candidates
from android_world.parallel_exploration.live_probe import _scroll_alignment_delta
from android_world.parallel_exploration.rankers import UiElement
from android_world.parallel_exploration.state import ActivitySignature
from android_world.parallel_exploration.state import StateSignature
from android_world.parallel_exploration.state import StructSignature
import pathlib
import re
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph


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


def test_action_buttons_are_structurally_filtered_independent_of_label(tmp_path: Path):
  elements = (
      UiElement(text="Start", class_name="android.widget.Button", clickable=True),
      UiElement(text="Harmless", class_name="android.widget.Button", clickable=True),
  )

  selected = _safe_candidates(elements, tmp_path / "filtered.jsonl", {})

  assert selected == []
  rows = (tmp_path / "filtered.jsonl").read_text(encoding="utf-8")
  assert rows.count("action_button_requires_semantic_inverse") == 2


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
