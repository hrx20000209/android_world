"""Exploration may probe the list row of an undone task entity, nothing more."""

import json
import tempfile
from pathlib import Path

from android_world.parallel_exploration.live_probe import _safe_candidates
from android_world.parallel_exploration.rankers import UiElement


def rows(names):
  out = []
  for i, name in enumerate(names):
    y = 300 + 200 * i
    out.append(UiElement(resource_id="app:id/row", class_name="android.widget.LinearLayout",
                         bounds=(0, y, 1080, y + 180), clickable=True, package="app.x"))
    out.append(UiElement(text=name, class_name="android.widget.TextView",
                         bounds=(20, y + 20, 800, y + 90), package="app.x"))
  return tuple(out)


def run(targets):
  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "filtered.jsonl"
    kept = _safe_candidates(rows(["Pesto", "Garlic Shrimp", "Lentil Soup"]), path,
                            {"trial_id": "t", "entity_row_targets": targets})
    reasons = [json.loads(l)["reason"] for l in path.read_text().splitlines()] if path.exists() else []
  return kept, reasons


def test_without_targets_every_repeated_row_is_filtered():
  kept, reasons = run([])
  assert not [e for e in kept if e.resource_id == "app:id/row"]
  # Each filter reason is logged once per candidate pass; three rows, none kept.
  assert reasons.count("repeated_dynamic_collection_item") % 3 == 0 and reasons


def test_only_the_row_naming_an_undone_entity_is_admitted():
  kept, reasons = run(["Garlic Shrimp"])
  admitted = [e for e in kept if e.resource_id == "app:id/row"]
  assert [e.bounds[1] for e in admitted] == [500]
  assert "icon_only_no_accessible_label" not in reasons


def test_scroll_parent_replays_as_swipe_not_tap():
  from android_world.parallel_exploration import live_probe
  from android_world.parallel_exploration.rankers import UiElement
  import inspect
  fields = inspect.signature(UiElement).parameters
  kwargs = {name: (p.default if p.default is not inspect.Parameter.empty else "")
            for name, p in fields.items()}
  kwargs.update(bounds=(120, 836, 960, 1724))
  pager = UiElement(**kwargs)
  swipe = live_probe._replay_record("SCROLL", pager)["action_dict"]
  assert swipe["action_type"] == "swipe"
  assert swipe["start_coordinate"][1] > swipe["end_coordinate"][1]
  assert live_probe._replay_record("TAP_NAV", pager)["action_dict"]["action_type"] == "click"
