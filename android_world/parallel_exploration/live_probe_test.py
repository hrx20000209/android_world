from pathlib import Path

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
