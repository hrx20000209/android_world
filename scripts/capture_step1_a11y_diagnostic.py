#!/usr/bin/env python3
"""Capture the clickable A11y elements before and after Clock's overflow menu."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from android_world.parallel_exploration.adb import AdbClient
from android_world.parallel_exploration.live_probe import _safe_candidates
from android_world.parallel_exploration.rankers import UiElement
from android_world.parallel_exploration.state import create_optimized_state_capture


def _element(element: UiElement) -> dict:
  return dataclasses.asdict(element)


def _write(path: Path, value) -> None:
  path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _capture_page(capture, output: Path, name: str) -> dict:
  state = capture.capture()
  context = {
      "trial_id": "ClockStopWatchRunning-step1-diagnostic",
      "step_idx": 1,
      "app_package": state.activity.component.split("/", 1)[0],
      "task": "Run the stopwatch.",
      "page": name,
  }
  rejection_path = output / f"{name}_rejections.jsonl"
  safe = _safe_candidates(state.elements, rejection_path, context)
  safe_ids = {item.identity for item in safe}
  rejection_by_id = {}
  if rejection_path.exists():
    for line in rejection_path.read_text(encoding="utf-8").splitlines():
      row = json.loads(line)
      item = row["element"]
      identity = "|".join((
          item["resource_id"], item["text"], item["content_desc"],
          item["class"], repr(tuple(item["bounds"])),
      ))
      rejection_by_id[identity] = row["reason"]

  clickable = []
  for element in state.elements:
    if not element.clickable:
      continue
    clickable.append({
        **_element(element),
        "exploration_decision": "allowed" if element.identity in safe_ids else "blocked",
        "reason": "explicitly_reversible" if element.identity in safe_ids
                  else rejection_by_id.get(element.identity, "not_a_candidate"),
    })
  _write(output / f"{name}_full_a11y.json", {
      "activity": state.activity.component,
      "timings_ms": state.timings_ms,
      "element_count": len(state.elements),
      "elements": [_element(item) for item in state.elements],
  })
  _write(output / f"{name}_clickable.json", {
      "activity": state.activity.component,
      "clickable_count": len(clickable),
      "allowed_count": sum(x["exploration_decision"] == "allowed" for x in clickable),
      "blocked_count": sum(x["exploration_decision"] == "blocked" for x in clickable),
      "elements": clickable,
  })
  return {"state": state, "clickable": clickable}


def _center(element: UiElement) -> tuple[int, int]:
  left, top, right, bottom = element.bounds
  return (left + right) // 2, (top + bottom) // 2


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  output = Path(args.output).resolve()
  output.mkdir(parents=True, exist_ok=True)
  adb = AdbClient("emulator-5554")
  capture = create_optimized_state_capture(
      serial="emulator-5554", console_port=5554, a11y_local_port=8765
  )
  menu_open = False
  try:
    adb.run(["shell", "am", "start", "-n",
             "com.google.android.deskclock/com.android.deskclock.DeskClock"])
    time.sleep(0.5)
    initial = capture.capture()
    stopwatch = next(
        item for item in initial.elements
        if item.content_desc == "Stopwatch" and item.clickable
    )
    adb.run(["shell", "input", "tap", *map(str, _center(stopwatch))])
    time.sleep(0.4)

    before = _capture_page(capture, output, "before_menu")
    more = next(
        item for item in before["state"].elements
        if item.content_desc == "More options" and item.clickable
    )
    adb.run(["shell", "input", "tap", *map(str, _center(more))])
    menu_open = True
    time.sleep(0.4)
    after = _capture_page(capture, output, "after_menu")

    _write(output / "summary.json", {
        "before": {
            "clickable": len(before["clickable"]),
            "allowed": sum(x["exploration_decision"] == "allowed" for x in before["clickable"]),
            "blocked": sum(x["exploration_decision"] == "blocked" for x in before["clickable"]),
        },
        "after": {
            "clickable": len(after["clickable"]),
            "allowed": sum(x["exploration_decision"] == "allowed" for x in after["clickable"]),
            "blocked": sum(x["exploration_decision"] == "blocked" for x in after["clickable"]),
        },
        "implementation_note": (
            "The current explorer computes candidates once from before_menu, "
            "then restores immediately; it does not recursively rank after_menu."
        ),
    })
  finally:
    if menu_open:
      adb.run(["shell", "input", "keyevent", "BACK"])
    adb.run(["shell", "input", "keyevent", "HOME"])
    capture.close()


if __name__ == "__main__":
  main()
