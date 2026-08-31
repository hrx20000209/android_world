"""Live, safety-filtered UI probing during one inference window."""

from __future__ import annotations

import json
import multiprocessing
import queue
import statistics
import dataclasses
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from android_world.parallel_exploration.adb import AdbClient
from android_world.parallel_exploration.protocol import Event
from android_world.parallel_exploration.protocol import EventKind
from android_world.parallel_exploration.protocol import ProcessRole
from android_world.parallel_exploration.information import InformationNeed
from android_world.parallel_exploration.information import parse_information_need
from android_world.parallel_exploration.rankers import InformationNeedRanker
from android_world.parallel_exploration.rankers import Ranker
from android_world.parallel_exploration.rankers import RandomRanker
from android_world.parallel_exploration.rankers import SimpleRelevanceRanker
from android_world.parallel_exploration import state_graph_information as sgi
from android_world.parallel_exploration.rankers import UiElement
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
from android_world.parallel_exploration.state import StateSignature
from android_world.parallel_exploration.state import create_optimized_state_capture
from android_world.parallel_exploration.safety import RiskLevel
from android_world.parallel_exploration.recovery import InverseActionPlanner
from android_world.parallel_exploration.recovery import execute_inverse
from android_world.parallel_exploration.recovery import replay_navigation_trajectory

import re


def _tokens(value: str) -> set[str]:
  return set(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def _goal_tokens_from_information_need(information_need: Mapping[str, Any]) -> set[str]:
  fields = (
      information_need.get("target_entity"),
      *(information_need.get("required_information_slots") or ()),
      *(information_need.get("expected_affordances") or ()),
      information_need.get("current_subgoal"),
  )
  tokens: set[str] = set()
  for field in fields:
    if field:
      tokens |= _tokens(str(field))
  return tokens


def _goal_relevance_score(new_texts: Iterable[str] | None, goal_tokens: set[str]) -> float:
  """Fraction of a hop's newly discovered labels that overlap the task's need.

  Used to decide whether a speculative branch is converging on what the
  task actually needs (InformationNeed) before spending further probe
  budget and recovery risk on it - not just whether the next candidate
  still looks generically clickable.
  """
  if not goal_tokens or not new_texts:
    return 0.0
  discovered: set[str] = set()
  for text in new_texts:
    discovered |= _tokens(text)
  if not discovered:
    return 0.0
  return len(discovered & goal_tokens) / len(discovered)


def _looks_unfamiliar(
    baseline_elements: Iterable[UiElement], post_elements: Iterable[UiElement],
) -> bool:
  """Structural "landed somewhere unfamiliar" fingerprint - no text involved.

  Two complementary patterns, neither of which a back-stack depth check
  reliably catches (some apps swap in a detail/edit fragment within the
  same Activity, so stack_depth does not move):

  - Overlay/dialog: most of the background persists and only a handful of
    new elements appear on top (e.g. a confirmation prompt).
  - Content replaced: most of what was on screen is simply gone, meaning a
    real navigation happened even though the Activity did not change
    (2026-08-29, ExpenseDeleteMultiple's "Expense Detail" / "will be
    deleted!" cases both discovered 59 new elements from a same-activity
    fragment swap into the edit view - too many to be a small overlay, but
    with next to nothing of the original list screen surviving either).
  """
  baseline_ids = {e.identity for e in baseline_elements}
  post_ids = {e.identity for e in post_elements}
  if not baseline_ids:
    return False
  persisted_ratio = len(baseline_ids & post_ids) / len(baseline_ids)
  new_count = len(post_ids - baseline_ids)
  overlay_like = persisted_ratio >= 0.6 and 0 < new_count <= 6
  content_replaced = persisted_ratio < 0.5
  return overlay_like or content_replaced


# Resource ids from Material Components' own design_navigation_menu_item.xml
# (and its icon/action-area siblings) - library-authored, identical across
# every app that uses the standard NavigationView drawer, not app-specific.
_MATERIAL_NAV_MENU_ITEM_IDS = {
    "design_menu_item_text", "design_menu_item_icon", "design_menu_item_action_area",
}


def _write_jsonl(path: Path, row: Mapping[str, Any]) -> None:
  with path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def _element_dict(element: UiElement, rank: int, score: float) -> dict[str, Any]:
  return {
      "text": element.text,
      "content_desc": element.content_desc,
      "resource_id": element.resource_id,
      "class": element.class_name,
      "bounds": list(element.bounds),
      "selected": element.selected,
      "rank": rank,
      "score": score,
  }


def element_identity_from_dict(element: Mapping[str, Any]) -> str:
  """Mirror UiElement.identity for a serialized probe_trace element record."""
  bounds = tuple(element.get("bounds") or ())
  return "|".join((
      str(element.get("resource_id") or ""), str(element.get("text") or ""),
      str(element.get("content_desc") or ""), str(element.get("class") or ""),
      repr(bounds),
  ))


_SYSTEM_UI_PACKAGES = ("com.android.systemui", "com.google.android.inputmethod",
                       "com.android.inputmethod")


@dataclasses.dataclass(frozen=True)
class _SnapshotView:
  """The step i-1 graph, as handed to the explorer process.

  The explorer runs in its own process and cannot touch the live graph, which
  is what makes the temporal rule structural here too: it sees a serialised
  copy of the previous generation and has no way to reach the current one.
  """

  generation: int
  node_visits: dict
  node_entropy: dict
  edges: dict
  outgoing: dict

  @classmethod
  def from_payload(cls, payload) -> "_SnapshotView | None":
    if not payload:
      return None
    return cls(
        generation=int(payload.get("generation", 0)),
        node_visits=dict(payload.get("node_visits") or {}),
        node_entropy={k: float(v) for k, v in (payload.get("node_entropy") or {}).items()},
        edges=dict(payload.get("edges") or {}),
        outgoing={k: list(v) for k, v in (payload.get("outgoing") or {}).items()},
    )


def _record_filtered(path: Path, row, reason: str) -> None:
  """PART E: every candidate the safety gate removed, and why."""
  _write_jsonl(path, {
      "stage": "safety_gate", "reason": reason,
      "element_identity": row.element_identity, "probe_type": row.probe_type,
      "role": row.role, "text": row.text,
      "estimated_recoverability": row.estimated_recoverability,
      "risk_level": row.risk_level,
  })


def _record_scored(path: Path, scored) -> None:
  """PART E: the chosen candidate with every scoring term kept separate.

  Logged component-wise on purpose - a regression has to be attributable to
  path probability, information gain or cost, not just to "the score moved".
  """
  row = scored.row
  _write_jsonl(path, {
      "stage": "predictive_scorer", "utility": scored.utility,
      "element_identity": row.element_identity, "probe_type": row.probe_type,
      "role": row.role, "text": row.text,
      "has_exact_history": row.has_exact_history,
      "session_alignment_hits": row.session_alignment_hits,
      "exploration_coverage": row.exploration_coverage,
      **scored.components,
  })


def _is_app_content(element: UiElement, label: str) -> bool:
  """Is this label something the app is showing, or chrome around it?

  What a probe "discovered" is supposed to describe the screen it reached, but
  the raw diff picks up the status bar and IME as well: real captures on
  2026-08-31 produced discovered_labels of ["15:34", "Do Not Disturb",
  "Battery 100 percent.", "Phone signal full.", "No internet"] - a clock that
  ticks, and notification icons that come and go - with no app content at all.
  Anything computed from those labels (expected information gain, the prompt
  briefing) was therefore built on noise, which is the likeliest reason
  injecting the briefing measured as harmful.

  Filters by structure rather than by a phrase list: system packages, and
  clock/battery/signal readouts that match a shape rather than a wording.
  """
  resource = (element.resource_id or "").lower()
  if any(pkg in resource for pkg in _SYSTEM_UI_PACKAGES):
    return False
  text = label.strip()
  if not text or len(text) > 60:
    return False
  if re.fullmatch(r"\d{1,2}:\d{2}(\s*[APap][Mm])?", text):
    return False           # clock
  if re.fullmatch(r".*\b\d{1,3}\s*percent\b.*", text, flags=re.I):
    return False           # battery readout
  if re.fullmatch(r"(phone|wifi|wi-fi)\s+signal.*", text, flags=re.I):
    return False
  if text.lower() in {"no internet", "do not disturb", ":"}:
    return False
  return True


def _safe_candidates(
    elements: tuple[UiElement, ...], filtered_path: Path, context: Mapping[str, Any]
) -> list[UiElement]:
  blocked = set(context.get("blocked_element_identities") or ())
  already_explored = set(context.get("already_explored_element_identities") or ())
  # Accessibility often places the label/icon on a non-clickable child and
  # the click handler on an unlabeled parent.  Transfer the child's semantics
  # to the smallest containing clickable parent, but keep the parent's actual
  # bounds and clickability.  This improves coverage without tapping arbitrary
  # non-clickable status/summary text.
  augmented = list(elements)
  # Proxy for screen extent: the largest right/bottom edge seen on this
  # screen, since the accessibility dump does not carry the raw display
  # size. Used only to recognize a near-full-screen unlabeled clickable
  # region below.
  screen_w = max((e.bounds[2] for e in elements), default=0)
  screen_h = max((e.bounds[3] for e in elements), default=0)
  for parent in elements:
    if not parent.clickable or parent.text or parent.content_desc:
      continue
    pl, pt, pr, pb = parent.bounds
    if pr <= pl or pb <= pt:
      continue
    if screen_w > 0 and screen_h > 0 and (pr - pl) >= 0.85 * screen_w and (pb - pt) >= 0.75 * screen_h:
      # A clickable region with no label of its own that also covers nearly
      # the whole screen is structurally a dismiss backdrop/scrim (Android's
      # "touch outside to close" pattern), not meaningful content. Borrowing
      # whatever long caption happens to sit underneath it as a synthetic
      # label is actively misleading - it makes a generic dismiss action
      # look like a specific, informative control (2026-08-29,
      # ExpenseDeleteMultiple: a full-screen "touch_outside" view kept
      # borrowing captions like "Expense Detail" this way). Leave it
      # unlabeled so the later no_semantic_identifier check excludes it.
      continue
    descendants = [
        child for child in elements
        if child is not parent
        and not child.clickable
        and bool(child.text or child.content_desc)
        and not child.resource_id.startswith(("com.android.systemui:", "com.example.androidworld:"))
        and pl <= child.bounds[0] <= child.bounds[2] <= pr
        and pt <= child.bounds[1] <= child.bounds[3] <= pb
    ]
    if descendants:
      label = max(descendants, key=lambda item: len(item.text or item.content_desc))
      # If the SPECIFIC descendant chosen as this candidate's borrowed label
      # carries one of Material Components' own internal navigation-menu
      # -item ids (design_navigation_menu_item.xml, compiled into every
      # app's resource namespace but authored by the library, not the app),
      # this clickable region is a NavigationView drawer item - true
      # regardless of app package, and independent of the accessibility
      # class name check above (custom views frequently do not override
      # getAccessibilityClassName(), so NavigationView itself often reports
      # as plain FrameLayout - observed 2026-08-29 on
      # RecipeDeleteMultipleRecipes's nav_view). Checked on `label`
      # specifically, not on any bounds-contained descendant: while a
      # drawer is open the underlying screen's own elements keep their
      # original coordinates and can spatially overlap real drawer items,
      # so "any descendant" over-fired on unrelated content sharing that
      # screen region (also observed 2026-08-29, a recipe card's own
      # description mislabeled this way while exploring with the drawer
      # open).
      in_drawer_item = label.resource_id.rsplit("/", 1)[-1] in _MATERIAL_NAV_MENU_ITEM_IDS
      augmented.append(dataclasses.replace(
          parent,
          text=label.text,
          content_desc=label.content_desc,
          resource_id=parent.resource_id or label.resource_id,
          in_navigation_drawer=parent.in_navigation_drawer or in_drawer_item,
      ))
  safe = []
  scroll_regions: list[tuple[int, int, int, int]] = []
  for element in augmented:
    if element.resource_id.startswith(("com.android.systemui:", "com.example.androidworld:")):
      continue
    if element.in_navigation_drawer:
      # Structural: this element sits under a standard AndroidX/Material
      # DrawerLayout or NavigationView (the widget classes essentially every
      # app builds its nav-drawer menu from - see FastA11yService.java's
      # isDrawerContainerClass), not a per-app naming convention. Drawer
      # items routinely lead to about/share/support screens that take the
      # task outside its own flow (2026-08-29, RecipeDeleteMultipleRecipes:
      # nav_recipes/nav_support/nav_about/Share all failed to roll back).
      _write_jsonl(filtered_path, {
          **context, "element": _element_dict(element, 0, 0.0),
          "reason": "navigation_drawer_item",
      })
      continue
    if element.identity in blocked:
      # This exact element already produced an unrecoverable state transition
      # earlier in this task (see dirty_events / RESTORE_FAILED). Learned from
      # observed rollback outcome, not a keyword/label rule: never probe it
      # again for the rest of the task.
      _write_jsonl(filtered_path, {
          **context, "element": _element_dict(element, 0, 0.0),
          "reason": "previously_caused_unrecoverable_state",
      })
      continue
    if element.identity in already_explored and not element.scrollable:
      # Already on record in the graph from a prior round that revisited
      # this same node (progressive per-task memory - see
      # ProgressiveBeliefGraph), whatever the outcome was. Re-probing it
      # would spend fresh budget and fresh rollback risk to relearn
      # something already known; let the ranker move on to a genuinely
      # unexplored candidate instead. Scrollable containers are exempt:
      # re-scrolling the same container can still reveal new list content,
      # unlike re-tapping a static control.
      _write_jsonl(filtered_path, {
          **context, "element": _element_dict(element, 0, 0.0),
          "reason": "already_explored_this_node",
      })
      continue
    role = element.class_name.rsplit(".", 1)[-1].casefold()
    left, top, right, bottom = element.bounds
    if not (element.clickable or element.scrollable):
      continue
    if element.scrollable:
      # Accessibility trees commonly expose both a coordinator/ScrollView and
      # its nested RecyclerView as independently scrollable.  Probing both in
      # one inference window compounds small, non-linear fling offsets and can
      # leave the UI between rows even after applying the inverse gesture.
      # Keep only one representative for substantially overlapping regions.
      area = max(1, (right - left) * (bottom - top))
      duplicate = False
      for other_left, other_top, other_right, other_bottom in scroll_regions:
        intersection = max(0, min(right, other_right) - max(left, other_left)) * max(
            0, min(bottom, other_bottom) - max(top, other_top)
        )
        other_area = max(
            1, (other_right - other_left) * (other_bottom - other_top)
        )
        if intersection / min(area, other_area) >= 0.80:
          duplicate = True
          break
      if duplicate:
        _write_jsonl(filtered_path, {
            **context, "element": _element_dict(element, 0, 0.0),
            "reason": "overlapping_scroll_container",
        })
        continue
      scroll_regions.append(element.bounds)
      safe.append(element)
      continue
    if not element.text and not element.content_desc and not element.resource_id:
      _write_jsonl(filtered_path, {
          **context, "element": _element_dict(element, 0, 0.0),
          "reason": "no_semantic_identifier",
      })
      continue
    # Exploration is click-first: semantic clickable controls are admitted and
    # ordinary navigation clicks use Back plus full state verification.  The
    # decision is structural rather than label/keyword based. Text-entry fields
    # are excluded because merely focusing them adds little successor-state
    # information and introduces keyboard state.
    reversible_toggle = element.checked is not None and role in {
        "switch", "checkbox", "togglebutton",
    }
    if element.checked is not None and not reversible_toggle:
      _write_jsonl(filtered_path, {
          **context, "element": _element_dict(element, 0, 0.0),
          "reason": "non_reversible_selection_control",
          "risk_level": RiskLevel.UNKNOWN.value,
      })
      continue
    if role == "button" and not reversible_toggle:
      _write_jsonl(filtered_path, {
          **context, "element": _element_dict(element, 0, 0.0),
          "reason": "action_button_requires_semantic_inverse",
          "risk_level": RiskLevel.UNKNOWN.value,
      })
      continue
    if not element.text and not element.content_desc and not reversible_toggle:
      # A resource_id alone gives an element a stable identity for tracking,
      # but says nothing about what it does. 89% of observed RESTORE_FAILED
      # probes (2026-08-28, ExpenseDeleteMultiple/RecipeAddMultipleRecipes
      # etc.) were icon-only elements (ImageView/ImageButton/blank TextView)
      # that had a resource_id and therefore passed the earlier
      # no_semantic_identifier check, then got structurally admitted as
      # "navigation" - two of them cascaded through an expense's edit view
      # into an Android share sheet. depth-2 probing already excludes
      # button/imagebutton/imageview/edittext roles; without an accessible
      # label there is no way to tell an icon-only nav control (back arrow,
      # tab) from an icon-only action trigger (sync, camera, swipe-delete
      # handle), so require the same semantic-inverse evidence as buttons.
      _write_jsonl(filtered_path, {
          **context, "element": _element_dict(element, 0, 0.0),
          "reason": "icon_only_no_accessible_label",
          "risk_level": RiskLevel.UNKNOWN.value,
      })
      continue
    navigation_click = bool(context.get("allow_navigation_click", True)) and role != "edittext"
    structurally_invertible = reversible_toggle or navigation_click
    if not structurally_invertible:
      _write_jsonl(filtered_path, {
          **context, "element": _element_dict(element, 0, 0.0),
          "reason": "risk_not_structurally_proven_safe",
          "risk_level": RiskLevel.UNKNOWN.value,
      })
      continue
    safe.append(element)
  return safe


def _probe_type(element: UiElement) -> str:
  """Which kind of probe this element affords.

  The set here bounds what exploration can ever anticipate: a probe type the
  explorer cannot perform is an action the model may take that no amount of
  ranking or budget will ever guess. Measured over 106 real actions on
  2026-08-31, 43% were of types the explorer had no probe for at all -
  input_text 9%, swipe 9%, long_press 2%, plus open_app/wait/status which are
  not UI interactions and are deliberately left out. Adding text entry and
  long press closes most of that gap.

  Restricted to taps and scrolls on purpose. Those are the two whose inverse
  is easy to establish - a scroll reverses by scrolling back, a tap that
  navigated reverses with Back - so recovery stays cheap and reliable. Text
  entry and long press were tried on 2026-08-31 and removed: long press opens
  context menus whose dismissal leaves residue, and every one of five tasks
  ran 3-4 steps longer than baseline with them enabled (MarkorDeleteNote
  5 -> 9, MarkorCreateFolder 5 -> 8), which is the opposite of the goal.
  Their share of model actions (11%) is not worth a recovery path that cannot
  be trusted.
  """
  if element.scrollable:
    return "SCROLL"
  role = element.class_name.rsplit(".", 1)[-1].casefold()
  if element.checked is not None and role in {"switch", "checkbox", "togglebutton"}:
    return "EXPAND"
  if role in {"imagebutton", "imageview"}:
    return "TAP_MENU"
  return "TAP_NAV"


def _center(element: UiElement) -> tuple[int, int]:
  left, top, right, bottom = element.bounds
  return (left + right) // 2, (top + bottom) // 2


def _adb_action(adb: AdbClient, probe_type: str, element: UiElement, inverse: bool = False) -> float:
  started = time.monotonic_ns()
  if probe_type == "SCROLL":
    left, top, right, bottom = element.bounds
    x = (left + right) // 2
    y1 = top + int((bottom - top) * (0.75 if not inverse else 0.25))
    y2 = top + int((bottom - top) * (0.25 if not inverse else 0.75))
    adb.run(["shell", "input", "swipe", str(x), str(y1), str(x), str(y2), "180"], timeout_s=2.0)
  elif inverse and probe_type in {"TAP_NAV", "TAP_MENU"}:
    adb.run(["shell", "input", "keyevent", "BACK"], timeout_s=2.0)
  else:
    x, y = _center(element)
    adb.run(["shell", "input", "tap", str(x), str(y)], timeout_s=2.0)
  return (time.monotonic_ns() - started) / 1e6


def _strictly_restored(
    baseline: StateSignature, current: StateSignature
) -> tuple[bool, dict[str, bool], dict[str, Any]]:
  matches = baseline.matches(current)
  def stable_records(state: StateSignature) -> list[tuple[Any, ...]]:
    return sorted(
        (
            item.class_name, item.resource_id,
            tuple((coordinate // 16) * 16 for coordinate in item.bounds),
            item.clickable, item.scrollable,
            -1 if item.checked is None else int(item.checked),
            -1 if item.selected is None else int(item.selected),
        )
        for item in state.elements
    )
  stable_structure = stable_records(baseline) == stable_records(current)
  phash_distance = baseline.phash_distance(current)
  # Dynamic text (clocks, timers, network counters) legitimately changes the
  # structural hash. Require the same activity plus agreement from at least one
  # visual/structural oracle, while preserving all three independent results.
  ok = matches["activity"] and (
      matches["struct"]
      or phash_distance <= 6
      or (matches["phash"] and stable_structure)
  )
  return ok, matches, {
      "phash_hamming_distance": phash_distance,
      "baseline_element_count": baseline.struct_sig.element_count,
      "current_element_count": current.struct_sig.element_count,
      "stable_structure": stable_structure,
  }


def _scroll_alignment_delta(
    baseline: StateSignature, current: StateSignature
) -> float | None:
  """Estimate the content offset using stable, uniquely visible UI anchors."""
  def anchors(state: StateSignature) -> dict[tuple[str, str, str, str], list[float]]:
    result: dict[tuple[str, str, str, str], list[float]] = {}
    for item in state.elements:
      semantic = (item.resource_id, item.text, item.content_desc, item.class_name)
      if not any(semantic[:3]):
        continue
      result.setdefault(semantic, []).append((item.bounds[1] + item.bounds[3]) / 2)
    return result

  before = anchors(baseline)
  after = anchors(current)
  deltas = [
      before[key][0] - after[key][0]
      for key in before.keys() & after.keys()
      if len(before[key]) == 1 and len(after[key]) == 1
  ]
  return statistics.median(deltas) if deltas else None


def _correct_scroll_with_anchors(
    adb: AdbClient, capture, baseline: StateSignature,
    current: StateSignature, element: UiElement, journal,
    abort_event: threading.Event,
) -> tuple[StateSignature, bool, dict[str, bool], dict[str, Any], float]:
  """Closed-loop correction for non-linear swipe/inverse displacement."""
  verify_ms = 0.0
  left, top, right, bottom = element.bounds
  x = (left + right) // 2
  center_y = (top + bottom) // 2
  matches: dict[str, bool] = {}
  details: dict[str, Any] = {}
  corrections: list[dict[str, Any]] = []
  for _ in range(3):
    delta = _scroll_alignment_delta(baseline, current)
    if delta is None:
      execute_inverse(adb, journal)
      applied = None
    else:
      applied = max(-500.0, min(500.0, delta))
      if abs(applied) < 8:
        break
      target_y = max(top + 24, min(bottom - 24, center_y + int(applied)))
      adb.run([
          "shell", "input", "swipe", str(x), str(center_y), str(x),
          str(target_y), "160",
      ], timeout_s=2.0)
    # Once a probe has mutated the device, cancellation must not bypass UI
    # settling.  The inference may finish during recovery, but restoration is
    # now on the critical path and has to complete deterministically.
    time.sleep(0.08)
    started = time.monotonic_ns()
    current = capture.capture()
    verify_ms += (time.monotonic_ns() - started) / 1e6
    ok, matches, details = _strictly_restored(baseline, current)
    corrections.append({
        "estimated_delta_y": delta, "applied_delta_y": applied,
        "phash_hamming_distance": details["phash_hamming_distance"],
    })
    if ok:
      details["anchor_corrections"] = corrections
      return current, True, matches, details, verify_ms
  ok, matches, details = _strictly_restored(baseline, current)
  details["anchor_corrections"] = corrections
  return current, ok, matches, details, verify_ms


def _recover(
    adb: AdbClient,
    capture,
    baseline: StateSignature,
    post: StateSignature | None,
    probe_type: str,
    element: UiElement,
    abort_event: threading.Event,
    committed_actions: list[Mapping[str, Any]] | None = None,
    known_inverse: str = "",
) -> tuple[str, bool, dict[str, bool], dict[str, Any], float, float]:
  """Undo a probe. known_inverse names the rung that worked here before.

  The ladder is ordered cheapest-first, so walking it from the top is right
  the first time a screen is probed and wasteful every time after: a screen
  that needs BACK_N pays for a failed INVERSE attempt on every single probe,
  and each of those attempts is itself a real interaction with the device.
  Starting from the remembered rung skips that, and the result is still
  verified the same way - a remembered rung that stops working falls through
  to the rest of the ladder exactly as an unknown one would.
  """
  recovery_started = time.monotonic_ns()
  verify_ms = 0.0
  if known_inverse in {"BACK_N", "BACK_OVERLAY", "DEEPLINK", "TRAJECTORY_REPLAY"}:
    # These live below INVERSE on the ladder; jumping to them avoids sending an
    # inverse gesture that has already been shown not to undo this transition.
    post = None
  # Some containers advertise scrollability even when they are already at an
  # edge (or while an overlay/search field owns the gesture).  If the observed
  # post-state is structurally identical, the probe was a semantic no-op.  Do
  # not apply an "inverse" gesture: that would create the first real mutation.
  if post is not None:
    post_ok, post_matches, post_details = _strictly_restored(baseline, post)
    # The structural signature is blind to how far a list is scrolled: it
    # records which controls exist and their coarse positions, so a scrolled
    # list still hashes the same. Declaring NOOP on that basis skips recovery
    # entirely and leaves the screen genuinely changed under the model - which
    # is how 55 of 178 successful recoveries were classified on 2026-08-31 (16
    # SCROLL, 39 TAP_NAV), and the arm's failures were twice as likely as the
    # baseline's to be the model announcing completion early, on a screen that
    # was no longer what it had been.
    #
    # Comparing exact bounds closes that: identical structure AND identical
    # element geometry is a real no-op, identical structure with shifted
    # geometry is a scroll that has to be undone.
    def geometry(state) -> tuple[tuple[Any, ...], ...]:
      return tuple(sorted(
          (e.class_name or "", e.resource_id or "", tuple(e.bounds))
          for e in state.elements))
    unmoved = geometry(baseline) == geometry(post)
    if post_matches["activity"] and post_matches["struct"] and unmoved:
      return (
          "NOOP", True, post_matches, post_details,
          (time.monotonic_ns() - recovery_started) / 1e6, verify_ms,
      )
  x, y = _center(element)
  action: dict[str, Any] = {"action_type": "CLICK", "x": x, "y": y}
  if probe_type == "SCROLL":
    left, top, right, bottom = element.bounds
    sx = (left + right) // 2
    action = {
        "action_type": "SWIPE",
        "start_coordinate": [sx, top + int((bottom - top) * 0.75)],
        "end_coordinate": [sx, top + int((bottom - top) * 0.25)],
    }
  post_match = None
  if post is not None:
    post_match = next((candidate for candidate in post.elements if (
        (element.resource_id and candidate.resource_id == element.resource_id)
        or (
            not element.resource_id
            and candidate.class_name == element.class_name
            and candidate.text == element.text
            and candidate.content_desc == element.content_desc
        )
    )), None)
  selected_candidates = [
      candidate for candidate in baseline.elements
      if candidate.selected is True
      and (candidate.resource_id or candidate.text or candidate.content_desc)
  ]
  # Material bottom-navigation marks the selected item and its descendants as
  # non-clickable. Tapping the outer selected region still selects it, so use
  # the largest selected semantic node rather than requiring clickable=True.
  selected_before = max(
      selected_candidates,
      key=lambda candidate: max(0, candidate.bounds[2] - candidate.bounds[0])
      * max(0, candidate.bounds[3] - candidate.bounds[1]),
      default=None,
  )
  selection_inverse = None
  if (
      selected_before is not None
      and element.selected is False
      and post_match is not None
      and post_match.selected is True
  ):
    selected_x, selected_y = _center(selected_before)
    selection_inverse = {
        "action_type": "CLICK", "x": selected_x, "y": selected_y,
    }
  elif (
      post is not None
      and post.activity.component == baseline.activity.component
      and post_match is not None
      and probe_type != "SCROLL"
  ):
    # Persistent same-activity controls are commonly mode/menu toggles. Retap
    # the identical semantic control first; verification decides whether that
    # was a true inverse before any Back/trajectory fallback is attempted.
    selection_inverse = action
  journal = InverseActionPlanner().plan(
      action, pre_checked=element.checked,
      post_checked=post_match.checked if post_match else None,
      selection_inverse_action=selection_inverse,
  )
  execute_inverse(adb, journal)
  time.sleep(0.10)
  verify_started = time.monotonic_ns()
  current = capture.capture()
  verify_ms += (time.monotonic_ns() - verify_started) / 1e6
  ok, matches, details = _strictly_restored(baseline, current)
  if ok:
    return "INVERSE", True, matches, details, (time.monotonic_ns() - recovery_started) / 1e6, verify_ms

  if probe_type == "SCROLL":
    current, ok, matches, details, anchor_verify_ms = _correct_scroll_with_anchors(
        adb, capture, baseline, current, element, journal, abort_event,
    )
    verify_ms += anchor_verify_ms
    if ok:
      return "INVERSE_ANCHOR", True, matches, details, (time.monotonic_ns() - recovery_started) / 1e6, verify_ms

  baseline_depth = baseline.activity.stack_depth or 1
  post_depth = (post.activity.stack_depth if post else None) or baseline_depth
  back_count = max(1, post_depth - baseline_depth)
  for _ in range(back_count):
    adb.run(["shell", "input", "keyevent", "BACK"], timeout_s=2.0)
  time.sleep(0.10)
  verify_started = time.monotonic_ns()
  current = capture.capture()
  verify_ms += (time.monotonic_ns() - verify_started) / 1e6
  ok, matches, details = _strictly_restored(baseline, current)
  if ok:
    return "BACK_N", True, matches, details, (time.monotonic_ns() - recovery_started) / 1e6, verify_ms

  # Same-activity overlays can consume one Back for the IME and another for
  # the overlay itself. Apply one additional, verified Back before escalating
  # to relaunch/trajectory reconstruction.
  if post is not None and post.activity.component == baseline.activity.component:
    adb.run(["shell", "input", "keyevent", "BACK"], timeout_s=2.0)
    time.sleep(0.10)
    verify_started = time.monotonic_ns()
    current = capture.capture()
    verify_ms += (time.monotonic_ns() - verify_started) / 1e6
    ok, matches, details = _strictly_restored(baseline, current)
    if ok:
      return "BACK_OVERLAY", True, matches, details, (time.monotonic_ns() - recovery_started) / 1e6, verify_ms

  component = baseline.activity.component
  if component:
    adb.run(["shell", "am", "start", "-n", component], timeout_s=3.0)
    time.sleep(0.15)
    verify_started = time.monotonic_ns()
    current = capture.capture()
    verify_ms += (time.monotonic_ns() - verify_started) / 1e6
    ok, matches, details = _strictly_restored(baseline, current)
    if ok:
      return "DEEPLINK", True, matches, details, (time.monotonic_ns() - recovery_started) / 1e6, verify_ms
  if committed_actions:
    replay_details = replay_navigation_trajectory(adb, committed_actions)
    time.sleep(0.20)
    verify_started = time.monotonic_ns()
    current = capture.capture()
    verify_ms += (time.monotonic_ns() - verify_started) / 1e6
    ok, matches, details = _strictly_restored(baseline, current)
    details["trajectory_replay"] = replay_details
    if ok:
      return "TRAJECTORY_REPLAY", True, matches, details, (time.monotonic_ns() - recovery_started) / 1e6, verify_ms
  return "FAILED", False, matches, details, (time.monotonic_ns() - recovery_started) / 1e6, verify_ms


_DEEP_EXCLUDED_ROLES = {"button", "imagebutton", "imageview", "edittext"}


def _explore_deeper(
    adb: AdbClient,
    capture,
    ranker: Ranker,
    ranker_name: str,
    context: Mapping[str, Any],
    filtered_path: Path,
    trace_path: Path,
    parent_state: StateSignature,
    parent_element_identity: str,
    depth: int,
    max_depth: int,
    goal_tokens: set[str],
    goal_relevance_threshold: float,
    root_baseline: StateSignature,
    max_stack_depth_increase: int,
    inference_done_event: threading.Event,
    exploration_started: float,
    max_exploration_time_s: float,
    probe_budget: list[int],
    inference_start_ns: int,
    explored_element_identities: Mapping[str, set[str]],
    replay_prefix: list[Mapping[str, Any]],
) -> tuple[int, bool]:
  """Probe one more speculative hop from parent_state; DFS with post-order rollback.

  Returns (deepest depth actually reached, whether any hop in this branch
  failed to roll back). The caller is responsible for rolling parent_state's
  own action back to ITS parent; this function only unwinds the hops it
  itself takes, deepest first, which is what makes an N-hop excursion safe:
  nothing is left mutated once this call returns.
  """
  probe_idx, max_probes = probe_budget[0], probe_budget[1]
  if (
      depth > max_depth
      or probe_idx >= max_probes
      or inference_done_event.is_set()
      or time.monotonic() - exploration_started >= max_exploration_time_s
      or "nexuslauncher" in context["app_package"]
  ):
    return depth - 1, False
  parent_node_id = ProgressiveBeliefGraph.make_node_id(
      parent_state.activity.component, parent_state.struct_sig.digest, parent_state.phash,
  )
  child_context = {
      **context, "depth": depth, "parent_element_identity": parent_element_identity,
      "already_explored_element_identities": sorted(explored_element_identities.get(parent_node_id, ())),
  }
  child_candidates = [
      child for child in _safe_candidates(parent_state.elements, filtered_path, child_context)
      if not child.scrollable and child.checked is None
      and child.identity != parent_element_identity
      and child.class_name.rsplit(".", 1)[-1].casefold() not in _DEEP_EXCLUDED_ROLES
  ]
  # Depth 2 stays on the InformationNeed ranker: the child screen was reached
  # for the first time moments ago, so it has no graph node and none of the
  # history the matrix exists to read. Routing it through the scorer would
  # only apply the fallback prior under a more expensive code path.
  child_ranked = ranker.rank(child_candidates, context["task"], ())
  child_candidate = child_ranked[0] if child_ranked else None
  if child_candidate is None:
    return depth - 1, False
  child = child_candidate.element
  child_started = time.monotonic_ns()
  child_action_ms = _adb_action(adb, _probe_type(child), child)
  time.sleep(0.10)
  child_post_started = time.monotonic_ns()
  child_post = capture.capture()
  child_post_capture_ms = (time.monotonic_ns() - child_post_started) / 1e6
  discovered_ids = {e.identity for e in child_post.elements} - {e.identity for e in parent_state.elements}
  new_texts = sorted({
      e.text or e.content_desc for e in child_post.elements
      if e.identity in discovered_ids and (e.text or e.content_desc)
  })
  probe_budget[0] += 1
  deepest = depth

  # Safety: a large single-hop jump in the Android back-stack usually means
  # this probe landed on an unfamiliar surface (detail/dialog/share flow),
  # not a simple navigation step; do not compound recovery risk by
  # continuing deeper from there (2026-08-28, ExpenseDeleteMultiple: exactly
  # this pattern cascaded into an Android share sheet).
  child_stack_depth = child_post.activity.stack_depth or 0
  root_stack_depth = root_baseline.activity.stack_depth or 0
  stack_ok = child_stack_depth <= root_stack_depth + max_stack_depth_increase
  # Goal-directed continuation: only keep expanding this branch while it is
  # still converging on what the task's InformationNeed says it needs, not
  # merely "still looks clickable" - the exact failure mode a plain
  # relevance ranker has for multi-hop lookahead.
  relevance = _goal_relevance_score(new_texts, goal_tokens)
  relevant = (not goal_tokens) or relevance > goal_relevance_threshold

  nested_failed = False
  if depth < max_depth and stack_ok and relevant:
    deepest, nested_failed = _explore_deeper(
        adb, capture, ranker, ranker_name, context, filtered_path, trace_path,
        child_post, child.identity, depth + 1, max_depth, goal_tokens,
        goal_relevance_threshold, root_baseline, max_stack_depth_increase,
        inference_done_event, exploration_started, max_exploration_time_s,
        probe_budget, inference_start_ns, explored_element_identities,
        list(replay_prefix) + [{"action_dict": {
            "action_type": "click", "x": _center(child)[0], "y": _center(child)[1],
        }}],
    )

  # parent_state is reconstructible as "committed prefix + every ancestor
  # probe action", so nested probes get the same full recovery ladder as
  # depth-1 ones instead of stopping short of TRAJECTORY_REPLAY. Passing
  # None here meant 12 of 23 observed rollback failures (2026-08-29) were
  # nested probes that never had access to the deepest recovery level.
  level, ok, matches, details, recovery_ms, verify_ms = _recover(
      adb, capture, parent_state, child_post, _probe_type(child), child,
      inference_done_event, replay_prefix or None,
  )
  total_ms = (time.monotonic_ns() - child_started) / 1e6
  _write_jsonl(trace_path, {
      **context, "probe_idx": probe_budget[0] - 1, "depth": depth,
      "parent_element_identity": parent_element_identity,
      "ranker": ranker_name, "probe_type": _probe_type(child),
      "element": _element_dict(child, child_candidate.rank, child_candidate.score),
      "t_offset_from_inference_start_ms": (child_started - inference_start_ns) / 1e6,
      "timings_ms": {
          "action_exec": child_action_ms, "post_state_capture": child_post_capture_ms,
          "recovery": recovery_ms, "recovery_verify": verify_ms, "total": total_ms,
      },
      "recovery_level": level, "recovery_ok": ok,
      "sig_match": matches, "sig_details": details,
      "goal_relevance": relevance, "stack_depth_ok": stack_ok,
      "discovered": {
          "new_element_count": len(discovered_ids), "new_texts": new_texts[:30],
          "reached_activity": child_post.activity.component,
      },
      "graph": {
          "src": {
              "activity": parent_state.activity.component,
              "structural_signature": parent_state.struct_sig.digest,
              "visual_signature": parent_state.phash,
              "layout_signature": parent_state.layout_sig,
          },
          "action": {
              "action_type": "CLICK", "x": _center(child)[0], "y": _center(child)[1],
              "element_identity": child.identity,
          },
          "dst": {
              "activity": child_post.activity.component,
              "structural_signature": child_post.struct_sig.digest,
              "visual_signature": child_post.phash,
              "layout_signature": child_post.layout_sig,
          },
      },
      "aborted_midway": inference_done_event.is_set(),
      "notes": "" if ok else "NESTED_RESTORE_FAILED",
  })
  return deepest, (nested_failed or not ok)


def explorer_window_process(
    config: Mapping[str, Any], control_queue: Any, status_queue: Any
) -> None:
  """Process B for one inference window."""
  trial_id = str(config["trial_id"])
  trace_path = Path(config["trace_path"])
  filtered_path = Path(config["filtered_path"])
  trace_path.parent.mkdir(parents=True, exist_ok=True)
  inference_done_event = threading.Event()
  inference_done_at: list[float] = []
  inference_start_ns = 0

  def listen_abort() -> None:
    while True:
      message = control_queue.get()
      if message.kind == EventKind.ABORT:
        inference_done_at.append(time.monotonic())
        inference_done_event.set()
        return

  adb = AdbClient(str(config.get("serial", "emulator-5554")))
  capture = create_optimized_state_capture(
      serial=str(config.get("serial", "emulator-5554")),
      console_port=int(config.get("console_port", 5554)),
      adb_path=str(config.get("adb_path", "/Users/huangrunxi/Library/Android/sdk/platform-tools/adb")),
      a11y_local_port=int(config.get("a11y_local_port", 8765)),
  )
  status_queue.put(Event.now(trial_id, EventKind.EXPLORER_READY, ProcessRole.EXPLORER))
  # A prepared explorer may intentionally sit idle while AndroidWorld resets the
  # task and captures the first observation.  That setup is not part of the
  # inference window, so do not expire the worker merely because it takes more
  # than an arbitrary number of seconds.
  first = control_queue.get()
  if first.kind != EventKind.INFERENCE_START:
    raise RuntimeError(f"Expected INFERENCE_START, got {first.kind}")
  inference_start_ns = first.monotonic_ns
  committed_actions = list(first.payload.get("committed_actions") or [])
  threading.Thread(target=listen_abort, daemon=True).start()
  dirty = False
  in_flight = False
  last_recovery = "NONE"
  # Pre-initialized so the except-clause below can always report a terminal
  # status, even if the failure happens before these are computed for real
  # (e.g. capture.capture() itself hangs).
  probe_idx = 0
  max_depth_reached = 0
  exploration_started = time.monotonic()
  min_probes = 0
  post_inference_grace_s = 0.0
  try:
    baseline = capture.capture()
    baseline_node_id = ProgressiveBeliefGraph.make_node_id(
        baseline.activity.component, baseline.struct_sig.digest, baseline.phash,
    )
    explored_element_identities = {
        node: set(identities)
        for node, identities in (config.get("explored_element_identities") or {}).items()
    }
    context = {
        "trial_id": trial_id, "step_idx": int(config.get("step_idx", 0)),
        "app_package": baseline.activity.component.split("/", 1)[0],
        "task": str(config.get("task", "")),
        "allow_navigation_click": bool(config.get("allow_navigation_click", True)),
        "blocked_element_identities": list(config.get("blocked_element_identities") or ()),
        "blocked_recovery_contexts": list(config.get("blocked_recovery_contexts") or ()),
        "allowed_probe_types": list(config.get("allowed_probe_types") or ("TAP_NAV", "SCROLL")),
        "known_inverse_levels": dict(config.get("known_inverse_levels") or {}),
        "recent_nodes": list(config.get("recent_nodes") or ()),
        # Already on record from a prior round revisiting this same node
        # (outcome known either way): re-probing it spends budget and fresh
        # rollback risk to relearn something already known, so the ranker
        # should move on to a still-unexplored candidate instead.
        "already_explored_element_identities": sorted(explored_element_identities.get(baseline_node_id, ())),
    }
    candidates = _safe_candidates(baseline.elements, filtered_path, context)
    ranker_name = str(config.get("ranker", "InformationNeedRanker"))
    # Populated only for InformationNeedRanker; RandomRanker/SimpleRelevanceRanker
    # stay unchanged (no goal-directed depth gating) so they remain valid,
    # unmodified baselines for comparison.
    goal_tokens: set[str] = set()
    if ranker_name == "RandomRanker":
      ranker: Ranker = RandomRanker(int(config.get("seed", 0)))
    elif ranker_name == "SimpleRelevanceRanker":
      ranker = SimpleRelevanceRanker()
    else:
      node_candidate_hits = dict(config.get("node_candidate_hits") or {})
      # Computed here, not passed in from the main process: only the explorer
      # subprocess has the freshly captured real screen, so this is the only
      # place expected_affordances can reflect what is actually on-screen
      # right now rather than the previous round's stale context.
      ui_labels = [
          label for element in baseline.elements
          for label in (element.text, element.content_desc) if label
      ]
      # Prefer the reasoning prior the main process extracted from the model's
      # previous output (paper Eq. 4). Falling back to the task goal is what
      # the explorer did unconditionally before, and the task goal cannot
      # discriminate between steps - it is the same string all episode.
      supplied = config.get("information_need")
      if supplied and supplied.get("source") == "reasoning_prior":
        information_need = InformationNeed(**supplied)
      else:
        information_need = parse_information_need(None, context["task"], ui_labels)
      ranker = InformationNeedRanker(
          information_need=information_need.to_dict(),
          known_good_identities=node_candidate_hits.get(baseline_node_id) or {},
      )
      goal_tokens = _goal_tokens_from_information_need(information_need.to_dict())
    probed: set[str] = set()
    probe_idx = 0
    max_depth_reached = 0
    scroll_probed = False
    deep_yield = False
    blocked_recovery_contexts = set(context.get("blocked_recovery_contexts") or ())
    allowed_probe_types = set(context.get("allowed_probe_types")
                              or ("TAP_NAV", "SCROLL"))
    known_inverse_levels = dict(context.get("known_inverse_levels") or {})
    exploration_started = time.monotonic()
    max_probes = max(0, int(config.get("max_probes", 12)))
    min_probes = min(max_probes, max(0, int(config.get("min_probes", 4))))
    post_inference_grace_s = max(
        0.0, float(config.get("post_inference_grace_s", 4.0))
    )
    max_exploration_time_s = max(0.0, float(config.get("max_exploration_time_s", 8.0)))
    max_depth = max(1, int(config.get("max_depth", 1)))
    goal_relevance_threshold = float(config.get("goal_relevance_threshold", 0.0))
    max_stack_depth_increase = int(config.get("max_stack_depth_increase", 1))

    graph_view = _SnapshotView.from_payload(config.get("graph_snapshot"))
    scored_path = Path(config.get("scored_path")
                       or filtered_path.parent / "scored_candidates.jsonl")
    use_scorer = bool(config.get("predictive_scorer", True)) and graph_view is not None
    matrix = sgi.StateGraphInformationMatrix()
    safety = sgi.SafetyGate()
    scorer = sgi.PredictiveElementScorer()

    # Where the chosen candidate's rank and score are handed back for the
    # trace row, so the probe record still says how it was picked regardless
    # of which selector picked it.
    selection = {"rank": 0, "score": 0.0}

    def select_candidate(pool, already_probed, state, node_id, out):
      """Which element to probe next: graph history first, labels second.

      Ranking by InformationNeed alone answers "which control looks related to
      the task", which is the only question the current screen can answer. The
      matrix adds the three the graph can: has probing this before ever
      matched what the model then did, does it come back, and what does it
      cost. Safety is applied as a filter before scoring, not as a low score,
      because an unrecoverable probe is not a worse choice - it ends the
      episode's ability to continue.

      Falls back to the ranker whenever there is no graph yet (the first steps
      of every task) or the scorer is ablated off, so the mechanism can be
      switched out without changing anything else.
      """
      remaining = [e for e in pool if e.identity not in already_probed]
      if not remaining:
        return None
      if not use_scorer:
        ranked = ranker.rank(remaining, context["task"], already_probed)
        if not ranked:
          return None
        out["rank"], out["score"] = ranked[0].rank, ranked[0].score
        return ranked[0].element
      rows = matrix.build(
          current_state=state, current_node_id=node_id,
          candidate_elements=remaining, graph_snapshot=graph_view,
          information_need=information_need.to_dict(),
          recovery_history={
              "blocked_recovery_contexts": blocked_recovery_contexts,
              "blocked_element_identities": context.get("blocked_element_identities") or (),
          },
          recent_nodes=config.get("recent_nodes") or (),
          probe_type_of=_probe_type,
          known_good_identities=node_candidate_hits.get(node_id) or {},
      )
      safe = []
      for row in rows:
        verdict = safety.evaluate(row)
        if verdict.allowed:
          safe.append(row)
        else:
          _record_filtered(filtered_path, row, verdict.reason)
      if not safe:
        return None
      best = scorer.rank(safe)[0]
      _record_scored(scored_path, best)
      out["rank"], out["score"] = 1, best.utility
      return best.row.element

    def should_stop_for_inference() -> bool:
      if not inference_done_event.is_set():
        return False
      if probe_idx >= min_probes:
        return True
      ended_at = inference_done_at[0] if inference_done_at else time.monotonic()
      return time.monotonic() - ended_at >= post_inference_grace_s

    while (
        not should_stop_for_inference()
        and candidates
        and probe_idx < max_probes
        and time.monotonic() - exploration_started < max_exploration_time_s
    ):
      element = select_candidate(candidates, probed, baseline, baseline_node_id,
                                 selection)
      if element is None:
        break
      root_probe_idx = probe_idx
      probed.add(element.identity)
      probe_type = _probe_type(element)
      if probe_type not in allowed_probe_types:
        # Recovery reliability differs sharply by probe type, and it is
        # measured, not assumed: over 452 probes on 2026-08-31, TAP_NAV at
        # depth 1 failed to roll back 7% of the time and SCROLL 10%, while
        # TAP_MENU was 16% and EXPAND 23%. Those two carried much of the 21%
        # overall failure rate that cost the arm ten tasks, so exploration
        # keeps the reliable kinds and drops the rest rather than stopping
        # altogether - the coverage loss is small because most candidates on
        # a screen are ordinary navigation targets.
        continue
      if f"{baseline.activity.component}|{probe_type}" in blocked_recovery_contexts:
        # This screen has already swallowed a probe of this kind this episode
        # (see blocked_recovery_contexts in the runner). Trying again spends a
        # fresh unrecoverable-state risk to relearn the same thing.
        continue
      if probe_type == "SCROLL" and scroll_probed:
        continue
      if probe_type == "SCROLL":
        scroll_probed = True
      row_started = time.monotonic_ns()
      in_flight = True
      action_ms = _adb_action(adb, probe_type, element)
      # Even if inference just completed, capture a settled post-state before
      # planning the inverse.  Skipping this delay caused intermittent dirty
      # states on long RecyclerViews.
      time.sleep(0.10)
      post_started = time.monotonic_ns()
      post = capture.capture()
      post_capture_ms = (time.monotonic_ns() - post_started) / 1e6
      discovered_ids = {e.identity for e in post.elements} - {e.identity for e in baseline.elements}
      new_texts = sorted({
          label for e in post.elements
          if e.identity in discovered_ids and (label := (e.text or e.content_desc))
          and _is_app_content(e, label)
      })
      max_depth_reached = max(max_depth_reached, 1)

      # Two structural (not text/keyword) signals that this root hop already
      # landed somewhere unfamiliar enough that drilling further would
      # compound risk rather than gather more signal: a back-stack jump
      # (real screen transitions, e.g. into an item's edit/detail view), or
      # a dialog/overlay fingerprint (most of the background persists, only
      # a few new elements appear - e.g. a delete-confirmation prompt,
      # which usually does not move the stack). Either one stops further
      # depth expansion from here; the root hop's own recovery still runs
      # either way (see below) and its outcome still feeds the learned
      # per-element blocklist regardless.
      root_stack_ok = (post.activity.stack_depth or 0) <= (baseline.activity.stack_depth or 0) + max_stack_depth_increase
      root_unfamiliar = _looks_unfamiliar(baseline.elements, post.elements)

      # Speculatively deepen this branch, goal-directed and safety-bounded
      # (see _explore_deeper): each hop is executed and independently rolled
      # back, deepest first, before the root successor itself is recovered.
      # The graph may later reuse a deeper hop once its ancestors have
      # earned real trust (INFERENCE_ALIGNED / VERIFIED via
      # promote_children_of_aligned_prefix).
      nested_recovery_failed = False
      depth2_blocked = f"{baseline.activity.component}|DEPTH2" in blocked_recovery_contexts
      if max_depth >= 2 and root_stack_ok and not root_unfamiliar and not depth2_blocked:
        probe_budget = [probe_idx, max_probes]
        reached_depth, nested_recovery_failed = _explore_deeper(
            adb, capture, ranker, ranker_name, context, filtered_path, trace_path,
            post, element.identity, 2, max_depth, goal_tokens,
            goal_relevance_threshold, baseline, max_stack_depth_increase,
            inference_done_event, exploration_started, max_exploration_time_s,
            probe_budget, inference_start_ns, explored_element_identities,
            list(committed_actions) + [{"action_dict": {
                "action_type": "click", "x": _center(element)[0], "y": _center(element)[1],
            }}],
        )
        probe_idx = probe_budget[0]
        max_depth_reached = max(max_depth_reached, reached_depth)
        deep_yield = deep_yield or reached_depth >= 2

      level, ok, matches, signature_details, recovery_ms, verify_ms = _recover(
          adb, capture, baseline, post, probe_type, element, inference_done_event,
          committed_actions,
          known_inverse=str(known_inverse_levels.get(
              f"{baseline.activity.component}|{probe_type}", "")),
      )
      last_recovery = level
      total_ms = (time.monotonic_ns() - row_started) / 1e6
      row = {
          **context, "probe_idx": root_probe_idx, "depth": 1,
          "ranker": ranker_name,
          "probe_type": probe_type,
          "element": _element_dict(element, selection["rank"], selection["score"]),
          "t_offset_from_inference_start_ms": (row_started - inference_start_ns) / 1e6,
          "timings_ms": {
              "action_exec": action_ms, "post_state_capture": post_capture_ms,
              "recovery": recovery_ms, "recovery_verify": verify_ms,
              "total": total_ms,
              **{f"baseline_{k}": v for k, v in baseline.timings_ms.items()},
              **{f"post_{k}": v for k, v in post.timings_ms.items()},
          },
          "recovery_level": level, "recovery_ok": ok,
          "sig_match": matches, "sig_details": signature_details,
          "root_stack_ok": root_stack_ok, "root_unfamiliar": root_unfamiliar,
          "discovered": {
              "new_element_count": len(discovered_ids),
              "new_texts": new_texts[:30],
              "reached_activity": post.activity.component,
          },
          "graph": {
              "src": {
                  "activity": baseline.activity.component,
                  "structural_signature": baseline.struct_sig.digest,
                  "visual_signature": baseline.phash,
                  "layout_signature": baseline.layout_sig,
              },
              "action": {
                  "action_type": "SWIPE" if probe_type == "SCROLL" else "CLICK",
                  "x": _center(element)[0], "y": _center(element)[1],
                  "element_identity": element.identity,
              },
              "dst": {
                  "activity": post.activity.component,
                  "structural_signature": post.struct_sig.digest,
                  "visual_signature": post.phash,
                  "layout_signature": post.layout_sig,
              },
          },
          "aborted_midway": inference_done_event.is_set(),
          "notes": "" if ok else "RESTORE_FAILED",
      }
      _write_jsonl(trace_path, row)
      in_flight = False
      probe_idx += 1
      if not ok:
        dirty = True
        break
      if nested_recovery_failed:
        break
      if deep_yield:
        # Stop widening once this window has a depth-2 child in hand.
        #
        # Only a child of the branch the model actually takes can save an
        # inference: step i's own decision is already being computed by the
        # authoritative model while we probe, so probing further depth-1
        # alternatives has no information gain for the objective (which is
        # the decision at step i+1). Measured on 2026-08-29: 537 of 682
        # probes were extra depth-1 breadth, they produced 58 of the 87
        # unrecoverable states, and the whole run bought 4 graph skips. The
        # breadth was paying the device-perturbation cost of the mechanism
        # without contributing to its benefit.
        break

    terminal = EventKind.RESTORE_FAILED if dirty else EventKind.RESTORED
    status_queue.put(Event.now(
        trial_id, terminal, ProcessRole.EXPLORER,
        {"restore_elapsed_ms": 0.0, "abort_during_work": in_flight,
         "abort_recovery_level": last_recovery, "dirty": dirty,
         "probes_completed": probe_idx, "max_depth_reached": max_depth_reached,
         "exploration_elapsed_ms": (time.monotonic() - exploration_started) * 1000,
         "preempted": inference_done_event.is_set(),
         "minimum_probes": min_probes,
         "post_inference_grace_s": post_inference_grace_s},
    ))
  except Exception as exc:
    # An unexpected failure anywhere in this window (e.g. one adb call
    # hanging past its own timeout, observed 2026-08-29 on
    # RecipeDeleteMultipleRecipes) must never leave this process silently
    # dead: the parent blocks on status_queue.get() until ITS OWN timeout,
    # then crashes the whole task. Report failure honestly instead - the
    # caller already treats RESTORE_FAILED as "assume dirty, fall back to
    # inference-only for this round" (see device_dirty handling in
    # run_one_parallel_exploration_task.py), the same as any other
    # unrecoverable probe.
    status_queue.put(Event.now(
        trial_id, EventKind.RESTORE_FAILED, ProcessRole.EXPLORER,
        {"restore_elapsed_ms": 0.0, "abort_during_work": in_flight,
         "abort_recovery_level": last_recovery, "dirty": True,
         "probes_completed": probe_idx, "max_depth_reached": max_depth_reached,
         "exploration_elapsed_ms": (time.monotonic() - exploration_started) * 1000,
         "preempted": inference_done_event.is_set(),
         "minimum_probes": min_probes,
         "post_inference_grace_s": post_inference_grace_s,
         "unexpected_error": f"{type(exc).__name__}: {exc}"},
    ))
  finally:
    capture.close()


def run_explorer_for_inference(
    config: Mapping[str, Any], inference_call
) -> tuple[Any, dict[str, Any]]:
  """Start Process B, execute inference in A, then abort and await restore."""
  context = multiprocessing.get_context("spawn")
  control_queue = context.Queue()
  status_queue = context.Queue()
  process = context.Process(
      target=explorer_window_process,
      args=(dict(config), control_queue, status_queue),
      name="live-ui-explorer",
  )
  process.start()
  ready = status_queue.get(timeout=15)
  if ready.kind != EventKind.EXPLORER_READY:
    raise RuntimeError(f"Explorer failed to become ready: {ready}")
  start = Event.now(str(config["trial_id"]), EventKind.INFERENCE_START, ProcessRole.INFERENCE)
  control_queue.put(start)
  inference_started = time.monotonic_ns()
  result = inference_call()
  inference_ms = (time.monotonic_ns() - inference_started) / 1e6
  abort = Event.now(str(config["trial_id"]), EventKind.ABORT, ProcessRole.INFERENCE)
  control_queue.put(abort)
  extension_started = time.monotonic_ns()
  terminal = status_queue.get(timeout=float(config.get("restore_timeout_s", 10.0)))
  extension_ms = (time.monotonic_ns() - extension_started) / 1e6
  process.join(timeout=2)
  if process.is_alive():
    process.terminate()
    process.join()
  return result, {
      "trial_id": config["trial_id"], "inference_ms": inference_ms,
      "critical_path_extension_ms": extension_ms,
      "restore_status": terminal.kind.value, **dict(terminal.payload),
  }


@dataclasses.dataclass
class PreparedExplorer:
  config: Mapping[str, Any]
  process: Any
  control_queue: Any
  status_queue: Any


def spawn_explorer(config: Mapping[str, Any]) -> PreparedExplorer:
  """Start Process B without waiting for it to finish warming up.

  Spawning costs a Python interpreter plus the state-capture setup, and none
  of that is exploration - the parallel design deliberately keeps it outside
  the measured window. The serial runner needs the same exclusion for a
  different reason: paying it inline made a 2-probe step take ~6.7s when the
  probes themselves are ~0.6s each (2026-08-30). Callers spawn before their
  inference call and await_ready() after it, so the startup overlaps the HTTP
  wait instead of adding to it.
  """
  context = multiprocessing.get_context("spawn")
  control_queue = context.Queue()
  status_queue = context.Queue()
  process = context.Process(
      target=explorer_window_process,
      args=(dict(config), control_queue, status_queue),
      name="live-ui-explorer",
  )
  process.start()
  return PreparedExplorer(dict(config), process, control_queue, status_queue)


def await_explorer_ready(prepared: PreparedExplorer, timeout_s: float = 30.0) -> PreparedExplorer:
  """Block until a spawned explorer reports EXPLORER_READY."""
  ready = prepared.status_queue.get(timeout=timeout_s)
  if ready.kind != EventKind.EXPLORER_READY:
    prepared.process.terminate()
    prepared.process.join()
    raise RuntimeError(f"Explorer failed to become ready: {ready}")
  return prepared


def prepare_explorer(config: Mapping[str, Any]) -> PreparedExplorer:
  """Initialize Process B before it enters the measured inference path."""
  return await_explorer_ready(spawn_explorer(config), timeout_s=15.0)


def run_serial_exploration(prepared: PreparedExplorer, timeout_s: float = 60.0) -> dict[str, Any]:
  """Run a prepared explorer to the end of its own probe budget, then wait.

  The parallel driver below races exploration against an inference call and
  aborts whatever is unfinished when the model answers. Serially there is no
  race to win: the explorer is given a fixed probe budget and allowed to spend
  all of it, because the point of the serial runner is to measure what
  exploration finds, not how much of it fits in a window. The explorer already
  posts its terminal event when the budget runs out (see the end of the probe
  loop), so no ABORT is sent - sending one would cut the budget short and make
  "probes_per_step" mean something different from what it says.

  Exploration time here lands on the critical path. That is expected and is
  why serial latency numbers are not comparable to the parallel design's.
  """
  trial_id = str(prepared.config["trial_id"])
  started = time.monotonic_ns()
  prepared.control_queue.put(Event.now(
      trial_id, EventKind.INFERENCE_START, ProcessRole.INFERENCE,
      {"committed_actions": list(prepared.config.get("committed_actions") or [])},
  ))
  try:
    terminal = prepared.status_queue.get(timeout=timeout_s)
  except Exception:  # pylint: disable=broad-exception-caught
    # Budget overrun or a hung probe: stop the explorer rather than letting a
    # stuck subprocess hold the device for the rest of the episode.
    prepared.control_queue.put(Event.now(trial_id, EventKind.ABORT, ProcessRole.INFERENCE))
    try:
      terminal = prepared.status_queue.get(timeout=float(
          prepared.config.get("restore_timeout_s", 10.0)))
    except Exception:  # pylint: disable=broad-exception-caught
      terminal = Event.now(trial_id, EventKind.RESTORE_FAILED, ProcessRole.EXPLORER,
                           {"dirty": True, "notes": "serial_timeout"})
  elapsed_ms = (time.monotonic_ns() - started) / 1e6
  prepared.process.join(timeout=2)
  if prepared.process.is_alive():
    prepared.process.terminate()
    prepared.process.join()
  return {
      "trial_id": trial_id,
      "exploration_ms": elapsed_ms,
      "restore_status": terminal.kind.value,
      **dict(terminal.payload),
  }


def run_prepared_explorer(prepared: PreparedExplorer, inference_call) -> tuple[Any, dict[str, Any]]:
  trial_id = str(prepared.config["trial_id"])
  start = Event.now(
      trial_id, EventKind.INFERENCE_START, ProcessRole.INFERENCE,
      {"committed_actions": list(prepared.config.get("committed_actions") or [])},
  )
  prepared.control_queue.put(start)
  inference_started = time.monotonic_ns()
  result = inference_call()
  inference_ms = (time.monotonic_ns() - inference_started) / 1e6
  prepared.control_queue.put(Event.now(trial_id, EventKind.ABORT, ProcessRole.INFERENCE))
  extension_started = time.monotonic_ns()
  terminal = prepared.status_queue.get(
      timeout=float(prepared.config.get("restore_timeout_s", 10.0))
  )
  extension_ms = (time.monotonic_ns() - extension_started) / 1e6
  prepared.process.join(timeout=2)
  if prepared.process.is_alive():
    prepared.process.terminate()
    prepared.process.join()
  return result, {
      "trial_id": trial_id, "inference_ms": inference_ms,
      "critical_path_extension_ms": extension_ms,
      "restore_status": terminal.kind.value, **dict(terminal.payload),
  }


def stop_prepared_explorer(prepared: PreparedExplorer | None) -> None:
  if prepared is not None and prepared.process.is_alive():
    prepared.process.terminate()
    prepared.process.join(timeout=2)
