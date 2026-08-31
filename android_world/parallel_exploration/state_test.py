"""Pure tests for state parsing and independent comparisons."""

import cv2
import numpy as np

from android_world.parallel_exploration.state import parse_activity_dump
from android_world.parallel_exploration.state import parse_elements
from android_world.parallel_exploration.state import parse_fast_elements
from android_world.parallel_exploration.state import ActivitySignature
from android_world.parallel_exploration.state import StateSignature
from android_world.parallel_exploration.state import StructSignature
from android_world.parallel_exploration.state import layout_signature
from android_world.parallel_exploration.state import perceptual_hash
from android_world.parallel_exploration.state import structural_signature
from android_world.parallel_exploration.rankers import UiElement


_XML = """<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<hierarchy rotation="0"><node text="Trips" resource-id="app:id/title"
 class="android.widget.TextView" content-desc="" clickable="true"
 scrollable="false" checked="false" bounds="[17,33][199,81]" /></hierarchy>"""


def test_structural_signature_quantizes_bounds_and_counts_nodes():
  first = parse_elements(_XML)
  shifted = parse_elements(_XML.replace("[17,33][199,81]", "[18,34][198,82]"))
  assert structural_signature(first).element_count == 1
  assert structural_signature(first).digest == structural_signature(shifted).digest


def test_structural_signature_sorts_mixed_empty_and_nonempty_text():
  common = {
      "content_desc": "", "resource_id": "same", "class_name": "same",
      "bounds": (0, 0, 16, 16), "clickable": False, "scrollable": False,
      "checked": None,
  }
  signature = structural_signature((
      UiElement(text="", **common), UiElement(text="visible", **common),
  ))
  assert signature.element_count == 2


def test_activity_parser_extracts_component_and_task():
  parsed = parse_activity_dump(
      "mResumedActivity: ActivityRecord{abc u0 com.example/.MainActivity t42}\n"
      "  ActivityRecord{def u0 com.example/.Other t42}"
  )
  assert parsed.component == "com.example/.MainActivity"
  assert parsed.task_id == 42
  assert parsed.stack_depth == 2


def test_activity_parser_accepts_component_before_closing_brace():
  parsed = parse_activity_dump(
      "topResumedActivity=ActivityRecord{abc u0 com.example/.MainActivity} t6}"
  )
  assert parsed.component == "com.example/.MainActivity"
  assert parsed.task_id == 6


def test_phash_is_stable_for_identical_image():
  image = np.zeros((64, 64, 3), dtype=np.uint8)
  image[16:48, 16:48] = 255
  ok, encoded = cv2.imencode(".png", image)
  assert ok
  assert perceptual_hash(encoded.tobytes()) == perceptual_hash(encoded.tobytes())


def test_fast_provider_json_maps_to_ui_elements():
  elements = parse_fast_elements({
      "nodes": [{
          "text": "Trips", "contentDescription": "Open trips",
          "resourceId": "app:id/trips", "class": "android.widget.Button",
          "bounds": [16, 32, 160, 80], "clickable": True,
          "scrollable": False, "checkable": False,
      }]
  })
  assert len(elements) == 1
  assert elements[0].text == "Trips"
  assert elements[0].bounds == (16, 32, 160, 80)
  assert elements[0].clickable


def test_phash_match_uses_hamming_distance_for_dynamic_pixels():
  common = {
      "activity": ActivitySignature("pkg/.Main", 1, 1, ""),
      "struct_sig": StructSignature("same", 1),
      "elements": (), "timings_ms": {},
  }
  baseline = StateSignature(phash="0000000000000000", **common)
  changed = StateSignature(phash="000000000000000f", **common)
  assert baseline.phash_distance(changed) == 4
  assert baseline.matches(changed)["phash"]


def test_layout_signature_ignores_volatile_text_but_keeps_skeleton():
  """Same screen with different data must resolve to the same identity.

  Regression guard for the 2026-08-29 finding that struct_sig/pHash-based
  node identity fragmented one Activity into dozens of nodes (a ticking
  stopwatch, a list losing a row), so the belief graph never recognized a
  revisit and progressive memory could not accumulate.
  """
  def screen(timer_text: str, extra_row: bool) -> tuple[UiElement, ...]:
    elements = [
        UiElement(text=timer_text, class_name="android.widget.TextView",
                  bounds=(0, 0, 200, 50)),
        UiElement(text="Start", resource_id="app:id/fab", clickable=True,
                  class_name="android.widget.Button", bounds=(10, 100, 110, 200)),
    ]
    if extra_row:
      elements.append(UiElement(text="Lap 1", class_name="android.widget.TextView",
                                bounds=(0, 300, 200, 350)))
    return tuple(elements)

  # Volatile content differs (timer value, one extra non-interactive row) but
  # the interaction skeleton is identical.
  assert layout_signature(screen("00:03", False)) == layout_signature(screen("00:47", True))
  # A genuinely different screen - a new interactive control - must differ.
  changed = screen("00:03", False) + (
      UiElement(text="Reset", resource_id="app:id/reset", clickable=True,
                class_name="android.widget.Button", bounds=(300, 100, 400, 200)),
  )
  assert layout_signature(screen("00:03", False)) != layout_signature(changed)


def _row(index: int, resource_id: str = "pkg:id/row"):
  from android_world.parallel_exploration.rankers import UiElement
  return UiElement(
      class_name="android.widget.LinearLayout", resource_id=resource_id,
      text=f"item {index}", bounds=(0, 200 + index * 150, 1080, 340 + index * 150),
      clickable=True,
  )


def _fab():
  from android_world.parallel_exploration.rankers import UiElement
  return UiElement(
      class_name="android.widget.ImageButton", resource_id="pkg:id/fab",
      content_desc="Add", bounds=(900, 2100, 1020, 2220), clickable=True,
  )


def test_layout_signature_ignores_how_many_rows_a_list_holds():
  """A repetitive task revisits the same list screen with one more row."""
  from android_world.parallel_exploration.state import layout_signature
  three = layout_signature([_fab()] + [_row(i) for i in range(3)])
  four = layout_signature([_fab()] + [_row(i) for i in range(4)])
  assert three == four


def test_layout_signature_still_separates_different_screens():
  """Row-count tolerance must not merge genuinely different screens."""
  from android_world.parallel_exploration.state import layout_signature
  list_screen = layout_signature([_fab()] + [_row(i) for i in range(3)])
  # Same rows, no floating action button: a different screen.
  without_fab = layout_signature([_row(i) for i in range(3)])
  # A screen whose repeated control is a different kind entirely.
  other_rows = layout_signature([_fab()] + [_row(i, "pkg:id/cell") for i in range(3)])
  assert list_screen != without_fab
  assert list_screen != other_rows
