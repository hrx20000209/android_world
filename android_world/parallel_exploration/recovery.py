"""Inverse-action planning and conservative committed-trajectory replay."""

from __future__ import annotations

import dataclasses
import enum
import time
from collections.abc import Mapping, Sequence
from typing import Any

from android_world.parallel_exploration.adb import AdbClient


class InverseKind(str, enum.Enum):
  BACK = "BACK"
  REVERSE_SCROLL = "REVERSE_SCROLL"
  RETAP_TOGGLE = "RETAP_TOGGLE"
  RETAP_SELECTION = "RETAP_SELECTION"
  NONE = "NONE"


@dataclasses.dataclass(frozen=True)
class ActionJournalEntry:
  action: dict[str, Any]
  inverse_kind: InverseKind
  inverse_action: dict[str, Any]
  src_state_id: str = ""
  dst_state_id: str = ""
  semantic_reversible: bool = False
  created_at: float = dataclasses.field(default_factory=time.time)


class InverseActionPlanner:
  """Plans an inverse from action/state metadata, never from task keywords."""

  def plan(
      self, action: Mapping[str, Any], *, pre_checked: bool | None = None,
      post_checked: bool | None = None,
      selection_inverse_action: Mapping[str, Any] | None = None,
  ) -> ActionJournalEntry:
    item = dict(action)
    kind = str(item.get("action_type") or "").upper()
    if kind in {"SCROLL", "SWIPE"}:
      start = item.get("start_coordinate") or item.get("start")
      end = item.get("end_coordinate") or item.get("end")
      if start and end:
        return ActionJournalEntry(
            item, InverseKind.REVERSE_SCROLL,
            {"action_type": "SWIPE", "start_coordinate": list(end), "end_coordinate": list(start)},
            semantic_reversible=False,
        )
    if kind in {"CLICK", "TAP", "EXPAND"}:
      if selection_inverse_action:
        return ActionJournalEntry(
            item, InverseKind.RETAP_SELECTION,
            dict(selection_inverse_action), semantic_reversible=True,
        )
      if pre_checked is not None and post_checked is not None and pre_checked != post_checked:
        return ActionJournalEntry(
            item, InverseKind.RETAP_TOGGLE, item.copy(), semantic_reversible=True,
        )
      return ActionJournalEntry(
          item, InverseKind.BACK, {"action_type": "NAVIGATE_BACK"},
          semantic_reversible=False,
      )
    return ActionJournalEntry(item, InverseKind.NONE, {}, semantic_reversible=False)


def execute_inverse(adb: AdbClient, entry: ActionJournalEntry) -> None:
  inverse = entry.inverse_action
  kind = entry.inverse_kind
  if kind == InverseKind.BACK:
    adb.run(["shell", "input", "keyevent", "BACK"], timeout_s=2.0)
  elif kind == InverseKind.REVERSE_SCROLL:
    start, end = inverse["start_coordinate"], inverse["end_coordinate"]
    adb.run(["shell", "input", "swipe", str(int(start[0])), str(int(start[1])), str(int(end[0])), str(int(end[1])), "180"], timeout_s=2.0)
  elif kind in {InverseKind.RETAP_TOGGLE, InverseKind.RETAP_SELECTION}:
    adb.run(["shell", "input", "tap", str(int(inverse["x"])), str(int(inverse["y"]))], timeout_s=2.0)


_APP_PACKAGES = {
    "clock": "com.google.android.deskclock",
    "joplin": "net.cozic.joplin",
    "markor": "net.gsantner.markor",
    "settings": "com.android.settings",
    "tasks": "org.tasks",
    "opentracks": "de.dennisguse.opentracks",
}


def replay_navigation_trajectory(adb: AdbClient, actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
  """Reconstruct the committed main-agent state from its action queue.

  These are actions already committed by the main agent, not speculative
  explorer actions. Replaying click/input is required to reconstruct states
  such as ``open app -> click Search -> enter query``. The caller always
  verifies the resulting full state signature before accepting recovery.
  """
  # Only start from HOME when the trajectory itself contains a launch that can
  # be resolved to a package. Going HOME without one leaves every following
  # coordinate to be tapped on the launcher, which is how a recovery for an
  # expense-app probe twice ended up in YouTube (2026-08-31); replaying in
  # place is both safe and closer to the state being reconstructed.
  from_home = any(
      _APP_PACKAGES.get(str(
          (dict(r.get("action_dict") or r.get("action") or r)).get("app_name")
          or (dict(r.get("tool_call") or {}).get("arguments") or {}).get("text")
          or "").casefold())
      for r in actions
      if str((dict(r.get("action_dict") or r.get("action") or r)).get(
          "action_type") or "").lower() == "open_app")
  if from_home:
    adb.run(["shell", "input", "keyevent", "HOME"], timeout_s=2.0)
    time.sleep(0.20)
  replayed = 0
  skipped = 0
  launched = not from_home
  for record in actions:
    action = dict(record.get("action_dict") or record.get("action") or record)
    tool = dict(record.get("tool_call") or {})
    arguments = dict(tool.get("arguments") or {})
    kind = str(action.get("action_type") or "").lower()
    if kind == "open_app":
      app_name = str(action.get("app_name") or arguments.get("text") or "").casefold()
      package = _APP_PACKAGES.get(app_name)
      if package:
        adb.run(["shell", "monkey", "-p", package, "1"], timeout_s=4.0)
        # Do not replay the next coordinate against the launcher or an app
        # transition frame. This is especially important for toolbar actions.
        time.sleep(0.35)
        replayed += 1
        launched = True
      else:
        # Every action after this one is a coordinate whose meaning depends on
        # the app being open. Skipping the launch and replaying them anyway
        # taps those coordinates on the launcher - which is how a recovery for
        # an expense-app probe ended up in YouTube, twice, on 2026-08-31.
        # Nothing further can be reconstructed, so stop.
        skipped += 1
        return {"replayed": replayed, "skipped_non_idempotent": skipped,
                "aborted": "unresolved_open_app", "app_name": app_name}
    elif kind == "click":
      coordinate = arguments.get("coordinate")
      x = action.get("x")
      y = action.get("y")
      if coordinate and len(coordinate) >= 2:
        x, y = coordinate[0], coordinate[1]
      if not launched:
        # A coordinate replayed before any app has been launched lands on the
        # launcher. There is nothing to reconstruct from here.
        skipped += 1
        return {"replayed": replayed, "skipped_non_idempotent": skipped,
                "aborted": "coordinate_before_launch"}
      if x is not None and y is not None:
        adb.run(["shell", "input", "tap", str(int(x)), str(int(y))], timeout_s=2.0)
        time.sleep(0.12)
        replayed += 1
      else:
        skipped += 1
    elif kind in {"input_text", "type"}:
      coordinate = arguments.get("coordinate")
      x = action.get("x")
      y = action.get("y")
      if coordinate and len(coordinate) >= 2:
        x, y = coordinate[0], coordinate[1]
      value = str(action.get("text") or arguments.get("text") or "")
      if x is not None and y is not None:
        adb.run(["shell", "input", "tap", str(int(x)), str(int(y))], timeout_s=2.0)
      if value:
        adb.run(["shell", "input", "text", value.replace(" ", "%s")], timeout_s=3.0)
        time.sleep(0.12)
        replayed += 1
      else:
        skipped += 1
    elif kind in {"swipe", "scroll"}:
      start = arguments.get("start_coordinate") or action.get("start_coordinate")
      end = arguments.get("end_coordinate") or action.get("end_coordinate")
      if start and end:
        adb.run(["shell", "input", "swipe", str(int(start[0])), str(int(start[1])), str(int(end[0])), str(int(end[1])), "180"], timeout_s=2.0)
        time.sleep(0.15)
        replayed += 1
      else:
        skipped += 1
    elif kind == "navigate_back":
      adb.run(["shell", "input", "keyevent", "BACK"], timeout_s=2.0)
      time.sleep(0.12)
      replayed += 1
    elif kind == "navigate_home":
      adb.run(["shell", "input", "keyevent", "HOME"], timeout_s=2.0)
      time.sleep(0.20)
      replayed += 1
    else:
      skipped += 1
  return {"replayed": replayed, "skipped_non_idempotent": skipped}
