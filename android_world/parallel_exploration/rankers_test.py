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
