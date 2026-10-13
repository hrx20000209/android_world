from android_world.parallel_exploration.action_binding import resolve_action_target
from android_world.parallel_exploration.rankers import UiElement


def _element(*, resource_id="", text="", content_desc="", class_name="",
             bounds=(0, 0, 100, 100), clickable=False, scrollable=False):
  return UiElement(
      resource_id=resource_id,
      text=text,
      content_desc=content_desc,
      class_name=class_name,
      bounds=bounds,
      clickable=clickable,
      scrollable=scrollable,
  )


def test_click_binds_smallest_compatible_interactive_node_not_outer_frame():
  screen = _element(class_name="android.widget.FrameLayout", bounds=(0, 0, 500, 900))
  button = _element(
      resource_id="app:id/save", class_name="android.widget.Button",
      bounds=(20, 40, 180, 100), clickable=True,
  )

  result = resolve_action_target({"action_type": "click", "x": 50, "y": 60},
                                 [screen, button])

  assert result.matchable
  assert result.scope == "stable_control"
  assert result.element is button


def test_click_resolves_clickable_row_when_point_lands_on_nonclickable_label():
  label = _element(
      resource_id="android:id/title", text="Documents",
      class_name="android.widget.TextView", bounds=(40, 70, 220, 110),
  )
  row = _element(
      class_name="android.widget.LinearLayout", bounds=(0, 40, 300, 140),
      clickable=True,
  )

  result = resolve_action_target({"action_type": "click", "x": 80, "y": 90},
                                 [label, row])

  assert result.matchable
  assert result.element is row
  assert result.scope == "runtime_instance"


def test_input_text_prefers_editable_target():
  parent = _element(class_name="android.widget.FrameLayout", bounds=(0, 0, 500, 900))
  field = _element(
      resource_id="app:id/search", class_name="android.widget.EditText",
      bounds=(20, 40, 450, 100),
  )

  result = resolve_action_target({"action_type": "input_text", "x": 80, "y": 70},
                                 [parent, field])

  assert result.matchable
  assert result.element is field
  assert result.scope == "stable_control"


def test_unique_scroll_container_can_be_bound_for_same_instance_verification():
  scroll = _element(
      class_name="android.widget.ScrollView", bounds=(0, 128, 1080, 2400),
      scrollable=True,
  )

  result = resolve_action_target({"action_type": "swipe", "direction": "up"},
                                 [scroll])

  assert result.matchable
  assert result.scope == "runtime_instance"
  assert result.element is scroll


def test_ambiguous_scroll_container_is_not_guessed():
  first = _element(class_name="android.widget.ScrollView", bounds=(0, 128, 1080, 900),
                   scrollable=True)
  second = _element(class_name="android.widget.ScrollView", bounds=(0, 900, 1080, 2200),
                    scrollable=True)

  result = resolve_action_target({"action_type": "swipe", "direction": "up"},
                                 [first, second])

  assert not result.matchable
  assert result.element is None
  assert result.reason == "ambiguous_scroll_container"
  assert result.candidate_count == 2


def test_action_outside_accessibility_bounds_is_unmatchable():
  button = _element(class_name="android.widget.Button", bounds=(0, 0, 100, 100),
                    clickable=True)

  result = resolve_action_target({"action_type": "click", "x": 120, "y": 120},
                                 [button])

  assert not result.matchable
  assert result.reason == "no_clickable_target_at_coordinate"
