"""Bind runtime actions to the live accessibility target they address.

Coordinates are execution-time geometry, not reusable control identity. This
module resolves a coordinate against the current UI tree and returns the
smallest compatible interactive node. For gestures without a target coordinate,
it only binds when the UI tree exposes one unambiguous scroll container.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from android_world.parallel_exploration.belief_graph import (
    stable_control_key_from_identity,
)
from android_world.parallel_exploration.rankers import UiElement


@dataclass(frozen=True)
class ActionTargetResolution:
  element: UiElement | None
  matchable: bool
  scope: str
  reason: str
  candidate_count: int = 0


def _valid_bounds(element: UiElement) -> bool:
  left, top, right, bottom = element.bounds
  return right > left and bottom > top


def _contains(element: UiElement, x: float, y: float) -> bool:
  left, top, right, bottom = element.bounds
  return left <= x < right and top <= y < bottom


def _is_editable(element: UiElement) -> bool:
  if bool(getattr(element, "editable", False)):
    return True
  class_name = (element.class_name or "").casefold()
  return any(token in class_name for token in (
      "edittext", "autocompletetextview", "textinputedittext",
  ))


def resolve_action_target(
    action: dict[str, Any], elements: Sequence[UiElement],
) -> ActionTargetResolution:
  """Resolve a UI action to a unique live target, without guessing.

  A semantic control key supports relocation across runtime layouts. When a
  control has no stable label or resource ID, its exact accessibility identity
  can still be used for same-instance verification; callers must retain the
  source runtime fingerprint in that case.
  """
  action_type = str(action.get("action_type") or "").casefold()
  x, y = action.get("x"), action.get("y")

  if action_type in {"click", "tap", "long_press", "input_text", "type"}:
    if x is None or y is None:
      return ActionTargetResolution(None, False, "unmatched", "missing_coordinates")
    try:
      point = float(x), float(y)
    except (TypeError, ValueError):
      return ActionTargetResolution(None, False, "unmatched", "invalid_coordinates")

    if action_type in {"input_text", "type"}:
      compatible = [element for element in elements
                    if _valid_bounds(element) and _contains(element, *point)
                    and _is_editable(element)]
      empty_reason = "no_editable_target_at_coordinate"
    else:
      compatible = [element for element in elements
                    if _valid_bounds(element) and _contains(element, *point)
                    and bool(element.clickable)]
      empty_reason = "no_clickable_target_at_coordinate"
    if not compatible:
      return ActionTargetResolution(None, False, "unmatched", empty_reason)

    # Nested Android views often include the screen root, app content, row,
    # and actual button at the same point. Select the smallest compatible
    # interactive node, not the first enclosing container in tree order.
    areas = [((e.bounds[2] - e.bounds[0]) * (e.bounds[3] - e.bounds[1]))
             for e in compatible]
    min_area = min(areas)
    best = [e for e, area in zip(compatible, areas) if area == min_area]
    identities = {element.identity for element in best}
    if len(identities) != 1:
      return ActionTargetResolution(
          None, False, "unmatched", "ambiguous_interactive_targets", len(best))
    target = best[0]
  elif action_type in {"swipe", "scroll"}:
    candidates = [element for element in elements
                  if bool(element.scrollable) and _valid_bounds(element)]
    if x is not None and y is not None:
      try:
        point = float(x), float(y)
      except (TypeError, ValueError):
        return ActionTargetResolution(None, False, "unmatched", "invalid_coordinates")
      candidates = [element for element in candidates
                    if _contains(element, *point)]
    if not candidates:
      return ActionTargetResolution(None, False, "unmatched", "no_scroll_container")
    if len(candidates) != 1:
      return ActionTargetResolution(
          None, False, "unmatched", "ambiguous_scroll_container", len(candidates))
    target = candidates[0]
  else:
    return ActionTargetResolution(None, False, "unmatched", "action_has_no_ui_target")

  scope = (
      "stable_control" if stable_control_key_from_identity(target.identity)
      else "runtime_instance"
  )
  return ActionTargetResolution(target, True, scope, "", 1)
