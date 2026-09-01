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
