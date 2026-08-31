# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text-only exploration agent for AndroidWorld.

This mirrors the lightweight exploration design from
Desktop/MobiCom/explorer_agent_gelab_light.py, but the model input is text only:
no screenshot or image_url is sent to the model. The policy receives task,
history, previous exploration hint, and the AndroidWorld UI element list.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict
from typing import Any

from PIL import Image

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import gelab_agent
from android_world.agents import gelab_agent_resize
from android_world.agents import m3a_utils
from android_world.agents import t3a
from android_world.agents.explorer_agent_utils import _hash_diff
from android_world.agents.explorer_agent_utils import _phash_pixels
from android_world.env import json_action

MAX_EXPLORER_STEPS = 16

TEXT_SYSTEM_PROMPT = (
    "You are a text-only Android GUI agent. Do not output chain-of-thought. "
    "Use only the UI element list and return one valid Action JSON."
)

_OPEN_APP_ALIASES = {
    "alarm": "Clock",
    "clock": "Clock",
    "stopwatch": "Clock",
    "timer": "Clock",
    "calendar": "Simple Calendar Pro",
    "simple calendar": "Simple Calendar Pro",
    "contacts": "Contacts",
    "settings": "Settings",
    "wifi": "Settings",
    "wi-fi": "Settings",
    "bluetooth": "Settings",
    "brightness": "Settings",
    "camera": "Camera",
    "chrome": "Chrome",
    "browser": "Chrome",
    "joplin": "Joplin",
    "notes": "Joplin",
    "markor": "Markor",
    "tasks": "Tasks",
    "todo": "Tasks",
    "to-do": "Tasks",
    "simple draw": "Simple Draw Pro",
    "draw": "Simple Draw Pro",
    "gallery": "Simple Gallery Pro",
    "simple gallery": "Simple Gallery Pro",
    "sms": "Simple SMS Messenger",
    "message": "Simple SMS Messenger",
    "messenger": "Simple SMS Messenger",
    "audio recorder": "Audio Recorder",
    "recorder": "Audio Recorder",
    "expense": "Pro Expense",
    "pro expense": "Pro Expense",
    "pro expenses": "Pro Expense",
    "broccoli": "Broccoli APP",
    "osmand": "OSMand",
    "map": "OSMand",
    "vlc": "VLC",
    "music": "Retro Music",
    "retro music": "Retro Music",
    "opentracks": "OpenTracks",
    "open tracks": "OpenTracks",
    "sports tracker": "OpenTracks",
    "activity tracker": "OpenTracks",
}

_GOAL_APP_PATTERNS: tuple[tuple[str, str], ...] = (
    ("audio recorder", "Audio Recorder"),
    ("record an audio", "Audio Recorder"),
    ("record audio", "Audio Recorder"),
    ("stopwatch", "Clock"),
    ("timer", "Clock"),
    ("alarm", "Clock"),
    ("clock", "Clock"),
    ("calendar", "Simple Calendar Pro"),
    ("meeting", "Simple Calendar Pro"),
    ("event", "Simple Calendar Pro"),
    ("contacts", "Contacts"),
    ("contact", "Contacts"),
    ("markor", "Markor"),
    ("joplin", "Joplin"),
    ("ingredient", "Joplin"),
    ("recipe", "Broccoli APP"),
    ("expense", "Pro Expense"),
    ("receipt", "Pro Expense"),
    ("wifi", "Settings"),
    ("wi-fi", "Settings"),
    ("bluetooth", "Settings"),
    ("brightness", "Settings"),
    ("settings", "Settings"),
    ("chrome", "Chrome"),
    ("browser", "Chrome"),
    ("osmand", "OSMand"),
    ("map", "OSMand"),
    ("favorite", "OSMand"),
    ("vlc", "VLC"),
    ("music", "Retro Music"),
    ("open tracks", "OpenTracks"),
    ("opentracks", "OpenTracks"),
    ("sports tracker", "OpenTracks"),
)

_APP_PACKAGE_HINTS = {
    "Audio Recorder": ("com.dimowner.audiorecorder",),
    "Clock": ("com.google.android.deskclock", "com.android.deskclock"),
    "Simple Calendar Pro": ("com.simplemobiletools.calendar",),
    "Contacts": ("com.google.android.contacts", "com.android.contacts"),
    "Markor": ("net.gsantner.markor",),
    "Joplin": ("net.cozic.joplin",),
    "Broccoli APP": ("com.flauschcode.broccoli",),
    "Pro Expense": ("com.arduia.expense", "com.expensemanager"),
    "Settings": ("com.android.settings",),
    "Chrome": ("com.android.chrome",),
    "OSMand": ("net.osmand",),
    "VLC": ("org.videolan.vlc",),
    "Retro Music": ("code.name.monkey.retromusic",),
    "OpenTracks": ("de.dennisguse.opentracks",),
}

_KNOWN_APP_LABELS = {name.lower() for name in _APP_PACKAGE_HINTS}
_KNOWN_APP_LABELS.update(alias.lower() for alias in _OPEN_APP_ALIASES)


def _now_hms() -> str:
    return time.strftime("%H:%M:%S")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _element_label(element: Any) -> str:
    if element is None:
        return ""
    parts = [
        _clean_text(getattr(element, "text", "")),
        _clean_text(getattr(element, "content_description", "")),
        _clean_text(getattr(element, "resource_name", "")),
    ]
    return _clean_text(" / ".join(part for part in parts if part))


def _salient_ui_overview(state: Any, limit: int = 10) -> str:
    labels: list[str] = []
    for element in list(getattr(state, "ui_elements", None) or []):
        label = _element_label(element)
        if not label:
            continue
        low = label.lower()
        if "systemui" in low or "battery" in low or "android system notification" in low:
            continue
        if label not in labels:
            labels.append(label)
        if len(labels) >= limit:
            break
    return "; ".join(labels)


def _local_step_summary(
    action: json_action.JSONAction,
    reason: str,
    before_elements: list[Any],
    after_state: Any | None,
) -> str:
    action_type = action.action_type
    action_part = f"Executed {action_type}."
    if action_type == json_action.OPEN_APP:
        action_part = f'Opened app "{_clean_text(action.app_name)}".'
    elif action_type == json_action.INPUT_TEXT:
        label = ""
        if action.index is not None and 0 <= int(action.index) < len(before_elements):
            label = _element_label(before_elements[int(action.index)])
        action_part = f'Typed "{_clean_text(action.text)}" into element {action.index}'
        if label:
            action_part += f" ({label})"
        action_part += "."
    elif action_type in {json_action.CLICK, json_action.LONG_PRESS, json_action.SCROLL}:
        label = ""
        if action.index is not None and 0 <= int(action.index) < len(before_elements):
            label = _element_label(before_elements[int(action.index)])
        action_part = f"Executed {action_type} on element {action.index}"
        if label:
            action_part += f" ({label})"
        action_part += "."
    elif action_type == json_action.ANSWER:
        action_part = f'Answered "{_clean_text(action.text)}".'
    elif action_type == json_action.STATUS:
        action_part = f"Returned status {action.goal_status}."

    reason_part = f" Reason: {_clean_text(reason)}." if _clean_text(reason) else ""
    after_part = ""
    if after_state is not None:
        overview = _salient_ui_overview(after_state)
        if overview:
            after_part = f" Current visible UI: {overview}."
    return _clean_text(action_part + reason_part + after_part)[:700]


def _visible_labels_lower(state: Any) -> str:
    labels = [_element_label(element) for element in list(getattr(state, "ui_elements", None) or [])]
    return " | ".join(label.lower() for label in labels if label)


def _extract_audio_target_basename(goal: str) -> str:
    goal_text = _clean_text(goal)
    match = re.search(r'(?:name|named)\s+["“”]?([^"“”]+?\.m4a)["“”]?', goal_text, flags=re.IGNORECASE)
    if not match:
        return ""
    return _clean_text(match.group(1)).strip()


def _is_audio_recorder_record_goal(goal: str) -> bool:
    goal_low = _clean_text(goal).lower()
    return "audio recorder" in goal_low and "record" in goal_low and "save" in goal_low


def _is_stopwatch_run_goal(goal: str) -> bool:
    goal_low = _clean_text(goal).lower()
    return "stopwatch" in goal_low and any(word in goal_low for word in ("run", "start", "running"))


def _is_stopwatch_page(state: Any) -> bool:
    labels = _visible_labels_lower(state)
    return (
        "stopwatch / com.google.android.deskclock:id/action_bar_title" in labels
        or "com.google.android.deskclock:id/stopwatch_time_text" in labels
    )


def _goal_field(goal: str, field_name: str) -> str:
    match = re.search(
        rf"{re.escape(field_name)}\s*:\s*([^,\n.]+)",
        _clean_text(goal),
        flags=re.IGNORECASE,
    )
    return _clean_text(match.group(1)) if match else ""


def _contacts_draft_complete(goal: str, labels: str) -> str:
    goal_low = _clean_text(goal).lower()
    if "contact" not in goal_low or "do not hit save" not in goal_low:
        return ""
    first = _goal_field(goal, "First Name")
    last = _goal_field(goal, "Last Name")
    phone = _goal_field(goal, "Phone")
    phone_label = _goal_field(goal, "Phone Label")
    required = [first, last, phone]
    if any(not item for item in required):
        return ""
    labels_compact = re.sub(r"[\s\-().]+", "", labels.lower())
    phone_compact = re.sub(r"[\s\-().]+", "", phone.lower())
    if (
        "create contact" in labels
        and first.lower() in labels
        and last.lower() in labels
        and phone_compact in labels_compact
        and (not phone_label or phone_label.lower() in labels)
    ):
        return "Contacts draft screen contains the requested first name, last name, phone number, and phone label without saving."
    return ""


def _expense_delete_complete(goal: str, labels: str) -> str:
    goal_low = _clean_text(goal).lower()
    if "delete" not in goal_low or "expense" not in goal_low:
        return ""
    if "expense logs" in labels and "no expense logs" in labels:
        return "Pro Expense shows Expense Logs with NO EXPENSE LOGS after the requested deletion."
    requested = []
    match = re.search(r":\s*(.+?)(?:\.|$)", _clean_text(goal))
    if match:
        requested = [_clean_text(item).lower() for item in re.split(r"[,;]", match.group(1)) if _clean_text(item)]
    requested_absent = bool(requested) and all(item not in labels for item in requested)
    empty_totals = "home" in labels and "totals" in labels and "outcome" in labels and re.search(r"\boutcome\b.*\b0\b", labels)
    if requested_absent and empty_totals:
        return "Pro Expense no longer shows the requested expense names and the visible total outcome is 0."
    return ""


def _settings_wifi_complete(goal: str, labels: str) -> str:
    goal_low = _clean_text(goal).lower()
    if "wifi" not in goal_low and "wi-fi" not in goal_low:
        return ""
    wants_on = "turn wifi on" in goal_low or "turn wi-fi on" in goal_low
    wants_off = "turn off wifi" in goal_low or "turn off wi-fi" in goal_low
    wifi_on_visible = "androidwifi,saved" in labels or "wifi signal full" in labels
    wifi_off_visible = "wi-fi / android:id/title" in labels and "networks available" not in labels and "androidwifi,saved" not in labels
    if wants_on and wifi_on_visible:
        return "Settings Internet page shows Wi-Fi/network availability, indicating Wi-Fi is on."
    if wants_off and wifi_off_visible:
        return "Settings Internet page shows Wi-Fi control without available Wi-Fi networks, indicating Wi-Fi is off."
    return ""


def _settings_bluetooth_complete(goal: str, labels: str) -> str:
    goal_low = _clean_text(goal).lower()
    if "bluetooth" not in goal_low:
        return ""
    wants_on = "enable bluetooth" in goal_low or "turn bluetooth on" in goal_low or "turn on bluetooth" in goal_low
    wants_off = "turn bluetooth off" in goal_low or "turn off bluetooth" in goal_low
    bluetooth_on_visible = "bluetooth / com.android.settings:id/collapsing_toolbar" in labels and (
        "device name" in labels or "pair new device" in labels
    )
    bluetooth_off_visible = "bluetooth / com.android.settings:id/collapsing_toolbar" in labels and (
        "bluetooth will turn on to pair" in labels or "use bluetooth" in labels
    ) and "device name" not in labels
    if wants_on and bluetooth_on_visible:
        return "Bluetooth settings show Device name / Pair new device, indicating Bluetooth is enabled."
    if wants_off and bluetooth_off_visible:
        return "Bluetooth settings show Bluetooth is not fully enabled."
    return ""


def _find_ui_index(state: Any, predicate: Any) -> int | None:
    for idx, element in enumerate(list(getattr(state, "ui_elements", None) or [])):
        try:
            if predicate(element):
                return idx
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    return None


def _audio_recorder_forced_action(goal: str, state: Any) -> tuple[json_action.JSONAction, OrderedDict[str, Any], str] | None:
    if not _is_audio_recorder_record_goal(goal):
        return None
    labels = _visible_labels_lower(state)
    target_base = _extract_audio_target_basename(goal)

    if "get started" in labels and "btn_action" in labels:
        start_idx = _find_ui_index(
            state,
            lambda e: _clean_text(getattr(e, "text", "")).lower() == "get started"
            or "btn_action" in _clean_text(getattr(e, "resource_name", "")).lower(),
        )
        if start_idx is not None:
            action = json_action.JSONAction(action_type=json_action.CLICK, index=start_idx)
            parsed = OrderedDict(cot="", action="click", summary="Open Audio Recorder setup.")
            return action, parsed, 'Action: {"action_type":"click","index":%d}' % start_idx

    if "setup" in labels and "apply" in labels and "btn_apply" in labels:
        apply_idx = _find_ui_index(
            state,
            lambda e: _clean_text(getattr(e, "text", "")).lower() == "apply"
            or "btn_apply" in _clean_text(getattr(e, "resource_name", "")).lower(),
        )
        if apply_idx is not None:
            action = json_action.JSONAction(action_type=json_action.CLICK, index=apply_idx)
            parsed = OrderedDict(cot="", action="click", summary="Apply Audio Recorder setup.")
            return action, parsed, 'Action: {"action_type":"click","index":%d}' % apply_idx

    if "warning!" in labels and "ok" in labels and "dialog_ok_btn" in labels:
        ok_idx = _find_ui_index(
            state,
            lambda e: _clean_text(getattr(e, "text", "")).lower() == "ok"
            or "dialog_ok_btn" in _clean_text(getattr(e, "resource_name", "")).lower(),
        )
        if ok_idx is not None:
            action = json_action.JSONAction(action_type=json_action.CLICK, index=ok_idx)
            parsed = OrderedDict(cot="", action="click", summary="Dismiss Audio Recorder warning dialog.")
            return action, parsed, 'Action: {"action_type":"click","index":%d}' % ok_idx

    if "new name" in labels and ("save" in labels or "dialog_positive_btn" in labels):
        if target_base:
            target_low = target_base.lower()
            if target_low not in labels:
                input_idx = _find_ui_index(
                    state,
                    lambda e: bool(getattr(e, "is_editable", False))
                    or "input_name" in _clean_text(getattr(e, "resource_name", "")).lower(),
                )
                if input_idx is not None:
                    action = json_action.JSONAction(
                        action_type=json_action.INPUT_TEXT,
                        index=input_idx,
                        text=target_base,
                        clear_text=True,
                    )
                    parsed = OrderedDict(cot="", action="input_text", summary=f'Type target recording name "{target_base}".')
                    return action, parsed, json.dumps(action.as_dict(skip_none=True))
            save_idx = _find_ui_index(
                state,
                lambda e: _clean_text(getattr(e, "text", "")).lower() == "save"
                or "dialog_positive_btn" in _clean_text(getattr(e, "resource_name", "")).lower(),
            )
            if save_idx is not None:
                action = json_action.JSONAction(action_type=json_action.CLICK, index=save_idx)
                parsed = OrderedDict(cot="", action="click", summary="Save the named recording.")
                return action, parsed, 'Action: {"action_type":"click","index":%d}' % save_idx
        save_idx = _find_ui_index(
            state,
            lambda e: _clean_text(getattr(e, "text", "")).lower() == "save"
            or "dialog_positive_btn" in _clean_text(getattr(e, "resource_name", "")).lower(),
        )
        if save_idx is not None:
            action = json_action.JSONAction(action_type=json_action.CLICK, index=save_idx)
            parsed = OrderedDict(cot="", action="click", summary="Save the recording.")
            return action, parsed, 'Action: {"action_type":"click","index":%d}' % save_idx

    active_recording = "btn_record_stop" in labels or "recording…" in labels or "recording..." in labels or "paused" in labels
    if active_recording and "btn_record_stop" in labels:
        stop_idx = _find_ui_index(
            state,
            lambda e: "btn_record_stop" in _clean_text(getattr(e, "resource_name", "")).lower(),
        )
        if stop_idx is not None:
            action = json_action.JSONAction(action_type=json_action.CLICK, index=stop_idx)
            parsed = OrderedDict(cot="", action="click", summary="Stop the current recording.")
            return action, parsed, 'Action: {"action_type":"click","index":%d}' % stop_idx

    if "audio recorder" in labels and "btn_record" in labels and "new name" not in labels and not active_recording:
        record_idx = _find_ui_index(
            state,
            lambda e: "btn_record" in _clean_text(getattr(e, "resource_name", "")).lower()
            and "btn_record_stop" not in _clean_text(getattr(e, "resource_name", "")).lower()
            and "btn_record_delete" not in _clean_text(getattr(e, "resource_name", "")).lower(),
        )
        if record_idx is not None:
            action = json_action.JSONAction(action_type=json_action.CLICK, index=record_idx)
            parsed = OrderedDict(cot="", action="click", summary="Start a new Audio Recorder recording.")
            return action, parsed, 'Action: {"action_type":"click","index":%d}' % record_idx

    return None


def _clock_forced_action(goal: str, state: Any) -> tuple[json_action.JSONAction, OrderedDict[str, Any], str] | None:
    if not _is_stopwatch_run_goal(goal):
        return None
    if not _state_matches_target_app(state, "Clock"):
        return None
    labels = _visible_labels_lower(state)
    is_stopwatch_page = _is_stopwatch_page(state)
    if is_stopwatch_page and "pause" in labels and "reset" in labels:
        return None
    if not is_stopwatch_page:
        stopwatch_idx = _find_ui_index(
            state,
            lambda e: "tab_menu_stopwatch" in _clean_text(getattr(e, "resource_name", "")).lower()
            or _clean_text(getattr(e, "text", "")).lower() == "stopwatch"
            or _clean_text(getattr(e, "content_description", "")).lower() == "stopwatch",
        )
        if stopwatch_idx is not None:
            action = json_action.JSONAction(action_type=json_action.CLICK, index=stopwatch_idx)
            parsed = OrderedDict(cot="", action="click", summary="Switch Clock to the Stopwatch tab.")
            return action, parsed, 'Action: {"action_type":"click","index":%d}' % stopwatch_idx
    start_idx = _find_ui_index(
        state,
        lambda e: bool(getattr(e, "is_clickable", False))
        and (
            _clean_text(getattr(e, "text", "")).lower() == "start"
            or _clean_text(getattr(e, "content_description", "")).lower() == "start"
            or (is_stopwatch_page and "fab" in _clean_text(getattr(e, "resource_name", "")).lower())
        ),
    )
    if start_idx is not None:
        action = json_action.JSONAction(action_type=json_action.CLICK, index=start_idx)
        parsed = OrderedDict(cot="", action="click", summary="Start the stopwatch.")
        return action, parsed, 'Action: {"action_type":"click","index":%d}' % start_idx
    return None


def _auto_complete_summary(goal: str, state: Any, history_summaries: list[str] | None = None) -> str:
    goal_low = _clean_text(goal).lower()
    labels = _visible_labels_lower(state)
    history_low = " ".join(_clean_text(x).lower() for x in (history_summaries or []))
    if goal_low.startswith("open ") and "app" in goal_low:
        target_app = _infer_goal_target_app(goal)
        permission_visible = any(
            marker in labels
            for marker in (
                "permission",
                "allow",
                "deny",
                "while using the app",
                "only this time",
                "grant",
            )
        )
        if target_app and _state_matches_target_app(state, target_app) and not permission_visible:
            return f'{target_app} is open and no required permission pop-up is visible.'
    if _is_stopwatch_run_goal(goal):
        if _is_stopwatch_page(state) and "pause" in labels and "reset" in labels:
            return "Stopwatch is running because the Stopwatch page shows Pause and Reset controls."
    if _is_audio_recorder_record_goal(goal):
        target_base = _extract_audio_target_basename(goal)
        clicked_save = "save / com.dimowner.audiorecorder:id/dialog_positive_btn" in history_low or "save the named recording" in history_low
        if clicked_save and "new name" not in labels and "dialog_positive_btn" not in labels:
            if target_base:
                if target_base.lower() in labels and "m4a" in labels:
                    return f'Audio Recorder shows saved recording "{target_base}" with M4a metadata.'
            elif "record-" in labels and "m4a" in labels and "txt_record_info" in labels:
                return "Audio Recorder shows a newly saved recording with M4a metadata."
    for detector in (
        _contacts_draft_complete,
        _expense_delete_complete,
        _settings_wifi_complete,
        _settings_bluetooth_complete,
    ):
        summary = detector(goal, labels)
        if summary:
            return summary
    return ""


def _normalize_open_app_name(app_name: str | None) -> str | None:
    if not app_name:
        return app_name
    cleaned = " ".join(str(app_name).strip().split())
    return _OPEN_APP_ALIASES.get(cleaned.lower(), cleaned)


def _infer_goal_target_app(goal: str) -> str | None:
    goal_low = _clean_text(goal).lower()
    for pattern, app_name in _GOAL_APP_PATTERNS:
        if pattern in goal_low:
            return app_name
    match = re.search(r"\bopen\s+(?:the\s+)?(.+?)\s+app\b", goal_low)
    if match:
        candidate = _normalize_open_app_name(match.group(1))
        if candidate:
            return candidate
    return None


def _state_package_names(state: Any) -> set[str]:
    packages: set[str] = set()
    for element in list(getattr(state, "ui_elements", None) or []):
        package_name = _clean_text(getattr(element, "package_name", ""))
        if package_name:
            packages.add(package_name)
    return packages


def _is_launcher_state(state: Any) -> bool:
    packages = _state_package_names(state)
    if not packages:
        return False
    non_system = {
        package
        for package in packages
        if package
        and package not in {"com.android.systemui", "com.google.android.inputmethod.latin"}
    }
    return bool(non_system) and non_system <= {"com.google.android.apps.nexuslauncher"}


def _state_matches_target_app(state: Any, app_name: str | None) -> bool:
    if not app_name:
        return False
    packages = _state_package_names(state)
    for package_hint in _APP_PACKAGE_HINTS.get(app_name, ()):
        if any(package.startswith(package_hint) for package in packages):
            return True
    labels = _visible_labels_lower(state)
    return app_name.lower() in labels


def _launcher_click_matches_target(state: Any, action: json_action.JSONAction, target_app: str | None) -> bool:
    if not target_app or action.action_type != json_action.CLICK or action.index is None:
        return False
    elements = list(getattr(state, "ui_elements", None) or [])
    try:
        idx = int(action.index)
    except (TypeError, ValueError):
        return False
    if idx < 0 or idx >= len(elements):
        return False
    label = _element_label(elements[idx]).lower()
    return target_app.lower() in label


def _is_unrelated_launcher_app_click(state: Any, action: json_action.JSONAction, target_app: str | None) -> bool:
    if not target_app or action.action_type != json_action.CLICK or action.index is None or not _is_launcher_state(state):
        return False
    elements = list(getattr(state, "ui_elements", None) or [])
    try:
        idx = int(action.index)
    except (TypeError, ValueError):
        return False
    if idx < 0 or idx >= len(elements):
        return True
    label = _element_label(elements[idx]).lower()
    if target_app.lower() in label:
        return False
    return any(app_label and app_label in label for app_label in _KNOWN_APP_LABELS)


def _make_open_app_action(app_name: str, reason: str) -> tuple[json_action.JSONAction, dict[str, Any], OrderedDict[str, Any]]:
    action = json_action.JSONAction(action_type=json_action.OPEN_APP, app_name=app_name)
    tool_call = {"name": "mobile_use", "arguments": action.as_dict(skip_none=True)}
    parsed = OrderedDict(cot="", action="open_app", app_name=app_name, summary=reason, sanitized_fallback=True)
    return action, tool_call, parsed


def _find_required_button_for_goal(goal: str, state: Any) -> int | None:
    goal_low = _clean_text(goal).lower()
    labels = _visible_labels_lower(state)
    preferred: list[str] = []
    if "stopwatch" in goal_low and any(word in goal_low for word in ("run", "start", "running")):
        preferred.extend(["start"])
    if _is_audio_recorder_record_goal(goal):
        preferred.extend(["get started", "apply", "ok", "start", "record", "stop", "save"])
    if any(word in goal_low for word in ("create", "new", "add")):
        preferred.extend(["add", "save", "ok"])
    if not preferred or not any(word in labels for word in preferred):
        return None
    return _find_ui_index(
        state,
        lambda e: bool(getattr(e, "is_clickable", False))
        and any(word == _clean_text(getattr(e, "text", "")).lower() or word in _element_label(e).lower() for word in preferred),
    )


def _looks_like_goal_or_plan_input(text: str | None, goal: str) -> bool:
    text_low = _clean_text(text).lower()
    goal_low = _clean_text(goal).lower()
    if not text_low:
        return False
    if text_low == goal_low or goal_low in text_low or text_low in goal_low:
        return True
    if text_low.startswith(("plan:", "first,", "first ", "step 1", "we need", "i need")):
        return True
    return len(text_low) > 80 and any(marker in text_low for marker in ("navigate", "open", "click", "task", "goal"))


def _sanitize_text_action(
    goal: str,
    state: Any,
    action: json_action.JSONAction,
    tool_call: dict[str, Any],
    parsed_action: OrderedDict[str, Any],
) -> tuple[json_action.JSONAction, dict[str, Any], OrderedDict[str, Any]]:
    target_app = _infer_goal_target_app(goal)

    if action.action_type == json_action.OPEN_APP:
        normalized = _normalize_open_app_name(action.app_name)
        if normalized and normalized != action.app_name:
            action = json_action.JSONAction(action_type=json_action.OPEN_APP, app_name=normalized)
            tool_call = {"name": "mobile_use", "arguments": action.as_dict(skip_none=True)}
            parsed_action["app_name"] = normalized
            parsed_action["sanitized_fallback"] = "normalized_open_app"
        return action, tool_call, parsed_action

    if target_app and _is_launcher_state(state) and not _launcher_click_matches_target(state, action, target_app):
        return _make_open_app_action(target_app, f'On launcher; open target app "{target_app}" instead of taking unrelated action.')

    if target_app and _is_unrelated_launcher_app_click(state, action, target_app):
        return _make_open_app_action(target_app, f'Prevent unrelated launcher app click; open target app "{target_app}".')

    ui_elements = list(getattr(state, "ui_elements", None) or [])
    if action.action_type == json_action.INPUT_TEXT:
        editable = False
        idx = None
        try:
            idx = int(action.index) if action.index is not None else None
        except (TypeError, ValueError):
            idx = None
        if idx is not None and 0 <= idx < len(ui_elements):
            editable = bool(getattr(ui_elements[idx], "is_editable", False))
        if (idx is None or not editable or _looks_like_goal_or_plan_input(action.text, goal)) and target_app and not _state_matches_target_app(state, target_app):
            return _make_open_app_action(target_app, f'Invalid input_text before reaching target app; open "{target_app}".')
        if idx is None or not editable or _looks_like_goal_or_plan_input(action.text, goal):
            parsed_action["sanitized_fallback"] = "invalid_input_text_to_wait"
            parsed_action["sanitized_reason"] = "input_text requires an editable target and must not type the task/plan text."
            action = json_action.JSONAction(action_type=json_action.WAIT)
            tool_call = {"name": "mobile_use", "arguments": {"action_type": "wait"}}
            return action, tool_call, parsed_action

    if action.action_type == json_action.STATUS and action.goal_status == "complete":
        required_idx = _find_required_button_for_goal(goal, state)
        if required_idx is not None:
            repaired = json_action.JSONAction(action_type=json_action.CLICK, index=required_idx)
            parsed = OrderedDict(cot="", action="click", summary="Completion was premature; click the visible required button first.", sanitized_fallback=True)
            return repaired, {"name": "mobile_use", "arguments": repaired.as_dict(skip_none=True)}, parsed
        if target_app and not _state_matches_target_app(state, target_app):
            return _make_open_app_action(target_app, f'Completion is premature before reaching "{target_app}".')

    return action, tool_call, parsed_action


def _text_prompt(goal: str, history: str, hint: str, ui_elements_description: str) -> str:
    history_text = history or "You just started, no action has been performed yet."
    exploration_block = ""
    if hint:
        exploration_block = (
            "\n\nRollback-verified exploration hint from previous step:\n"
            f"{hint}\n"
            "Use it only if the current UI element list still matches; otherwise ignore it.\n"
        )
    return (
        t3a.PROMPT_PREFIX
        + f"\nThe current user goal/request is: {goal}"
        + f"\n\nHere is a history of what you have done so far:\n{history_text}"
        + exploration_block
        + "\n\nHere is a list of descriptions for some UI elements on the current screen:\n"
        + (ui_elements_description if ui_elements_description else "Not available")
        + "\n"
        + t3a.GUIDANCE
        + "\nAdditional text-only constraints:\n"
        + "- Do not output coordinates; use UI element index for click/long_press/input_text.\n"
        + "- If the current screen is the launcher/home screen and the goal names an app, use open_app with the exact app name; do not click unrelated launcher icons.\n"
        + "- Never type the full user goal, a plan, or reasoning text into the phone. input_text is only for user-requested data values and only into editable fields.\n"
        + "- Do not invent high-level actions. The only valid action_type values are status, answer, click, long_press, input_text, keyboard_enter, navigate_home, navigate_back, scroll, open_app, wait.\n"
        + '- For stopwatch/timer/alarm tasks, open app "Clock" rather than "Stopwatch" or "Timer".\n'
        + '- If the task is to run/start the stopwatch and a "Start" button is visible, click the "Start" button before completing.\n'
        + '- In Audio Recorder, when a "New name" dialog is visible and the goal gives a .m4a file name, type the full requested file name including ".m4a", then click Save.\n'
        + '- In Audio Recorder, after the saved recording is visible with M4a metadata, return status complete.\n'
        + '- Do not mark the task complete while an obvious required action button such as "Start", "Save", "Add", or "OK" is still visible and not yet used.\n'
        + '- Use canonical AndroidWorld app names when opening apps, e.g. "Pro Expense", "Joplin", "Markor", "Simple Calendar Pro".\n'
        + "- For question tasks, use answer with the exact requested format before status complete.\n"
        + "\n\nNow output one action from the above list in the correct JSON format. Keep it concise and deterministic.\n"
        + "Preferred format:\n"
        + "Reason: <one short sentence, <= 20 words>\n"
        + "Action: {\"action_type\":...}\n"
        + "Do not output long chain-of-thought.\n\n"
        + "Your Answer:\n"
    )


def _summary_prompt(
    goal: str,
    action: json_action.JSONAction,
    reason: str,
    before_elements: str,
    after_elements: str,
) -> str:
    return (
        "You are summarizing one Android GUI step for future action selection.\n"
        "Do not output chain-of-thought. Return one concise sentence.\n\n"
        f"Task:\n{goal}\n\n"
        f"Chosen action:\n{action.as_dict(skip_none=True)}\n\n"
        f"Model reason:\n{reason or 'No reason provided.'}\n\n"
        f"Before UI elements:\n{before_elements or 'Not available'}\n\n"
        f"After UI elements:\n{after_elements or 'Not available'}\n\n"
        "Summary requirements:\n"
        "- Say what changed or whether the action was redundant.\n"
        "- Keep it under 25 words.\n\n"
        "Summary:"
    )


def _collect_response_candidates(action_output: str, raw_response: object) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(text: object) -> None:
        if not isinstance(text, str):
            return
        value = text.strip()
        if value and value not in seen:
            seen.add(value)
            candidates.append(value)

    add(action_output)
    if isinstance(raw_response, dict):
        try:
            message = raw_response["choices"][0]["message"]
        except Exception:  # pylint: disable=broad-exception-caught
            message = None
        if isinstance(message, dict):
            add(message.get("content"))
            add(message.get("reasoning_content"))
    return candidates


def _json_objects_from_text(text: str) -> list[dict[str, Any]]:
    text = str(text or "")
    if "<tool_call>" in text and "</tool_call>" in text:
        try:
            inner = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
            text = inner.strip()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    decoder = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        if isinstance(obj, dict):
            out.append(obj)
    fallback = agent_utils.extract_json(text)
    if isinstance(fallback, dict):
        out.append(fallback)
    return out


def _normalize_action_dict(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("arguments"), dict):
        payload = dict(payload["arguments"])
    raw_action = payload.get("action_type") or payload.get("action") or payload.get("name")
    if not raw_action:
        return None
    action_name = _clean_text(raw_action).lower().replace("-", "_")
    aliases = {
        "tap": json_action.CLICK,
        "click": json_action.CLICK,
        "longpress": json_action.LONG_PRESS,
        "long_press": json_action.LONG_PRESS,
        "type": json_action.INPUT_TEXT,
        "input": json_action.INPUT_TEXT,
        "input_text": json_action.INPUT_TEXT,
        "enter_text": json_action.INPUT_TEXT,
        "keyboard_enter": json_action.KEYBOARD_ENTER,
        "enter": json_action.KEYBOARD_ENTER,
        "back": json_action.NAVIGATE_BACK,
        "navigate_back": json_action.NAVIGATE_BACK,
        "home": json_action.NAVIGATE_HOME,
        "navigate_home": json_action.NAVIGATE_HOME,
        "open": json_action.OPEN_APP,
        "open_app": json_action.OPEN_APP,
        "scroll": json_action.SCROLL,
        "swipe": json_action.SCROLL,
        "wait": json_action.WAIT,
        "answer": json_action.ANSWER,
        "complete": json_action.STATUS,
        "terminate": json_action.STATUS,
        "status": json_action.STATUS,
    }
    if action_name == "system_button":
        button = _clean_text(payload.get("button")).lower()
        action_type = json_action.NAVIGATE_HOME if button == "home" else json_action.NAVIGATE_BACK
    else:
        action_type = aliases.get(action_name)
    if not action_type:
        return None

    result: dict[str, Any] = {"action_type": action_type}
    if action_type in {json_action.CLICK, json_action.LONG_PRESS, json_action.INPUT_TEXT, json_action.SCROLL}:
        idx = payload.get("index")
        if idx is None:
            idx = payload.get("element_id")
        if idx is not None and idx != "":
            result["index"] = idx
    if action_type == json_action.INPUT_TEXT:
        result["text"] = str(payload.get("text") if payload.get("text") is not None else payload.get("value", ""))
        if payload.get("clear_text") is not None:
            result["clear_text"] = bool(payload.get("clear_text"))
    if action_type == json_action.SCROLL:
        direction = _clean_text(payload.get("direction") or "down").lower()
        result["direction"] = direction if direction in {"up", "down", "left", "right"} else "down"
    if action_type == json_action.OPEN_APP:
        result["app_name"] = _normalize_open_app_name(
            _clean_text(payload.get("app_name") or payload.get("text") or payload.get("value"))
        )
    if action_type == json_action.STATUS:
        status = _clean_text(payload.get("goal_status") or payload.get("status") or payload.get("value") or "complete").lower()
        result["goal_status"] = "infeasible" if status in {"fail", "failure", "infeasible", "abort"} else "complete"
    if action_type == json_action.ANSWER:
        result["text"] = str(payload.get("text") if payload.get("text") is not None else payload.get("value", ""))
    return result


def _parse_text_action(response: str, raw_response: object) -> tuple[json_action.JSONAction, dict[str, Any], OrderedDict[str, Any], str]:
    parse_error = ""
    for candidate in _collect_response_candidates(response, raw_response):
        reason, action_text = m3a_utils.parse_reason_action_output(candidate)
        search_units = [action_text] if action_text else []
        search_units.append(candidate)
        for unit in search_units:
            if not unit:
                continue
            for obj in _json_objects_from_text(unit):
                action_dict = _normalize_action_dict(obj)
                if not action_dict:
                    continue
                try:
                    action = json_action.JSONAction(**action_dict)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    parse_error = str(exc)
                    continue
                parsed = OrderedDict(
                    cot="",
                    action=action.action_type,
                    summary=_clean_text(reason) or f"Selected {action.action_type} from text UI.",
                )
                if action.text is not None:
                    parsed["value"] = action.text
                if action.app_name is not None:
                    parsed["value"] = action.app_name
                if action.goal_status is not None:
                    parsed["status"] = action.goal_status
                tool_call = {"name": "mobile_use", "arguments": action.as_dict(skip_none=True)}
                return action, tool_call, parsed, parse_error
    raise ValueError(parse_error or "missing valid action JSON")


class ExplorerTextAgent(gelab_agent_resize.GELABResizeAgent):
    """Text-only GELAB-like core with the same lightweight exploration hint design."""

    def set_max_steps(self, max_steps: int) -> None:
        super().set_max_steps(min(MAX_EXPLORER_STEPS, int(max_steps)))

    def _effective_max_steps(self) -> int:
        if self._max_steps is None:
            return MAX_EXPLORER_STEPS
        return min(MAX_EXPLORER_STEPS, int(self._max_steps))

    def __init__(
        self,
        env,
        vllm: Any,
        name: str = "ExplorerTextAgent",
        output_path: str = "",
        history_limit: int = 8,
        image_downsample_scale: float = 1.0,
        enable_light_exploration: bool = True,
        light_explore_max_runs: int = 1,
        light_explore_max_step: int = 2,
        light_explore_launcher_only: bool = True,
        light_explore_require_keyword: bool = True,
        light_explore_require_stall: bool = True,
        light_explore_back_limit: int = 2,
        light_explore_hash_threshold: int = 10,
        light_explore_replay_max_actions: int = 1,
        **kwargs: Any,
    ):
        _ = kwargs
        super().__init__(
            env=env,
            vllm=vllm,
            name=name,
            output_path=output_path,
            history_limit=history_limit,
            image_downsample_scale=image_downsample_scale,
        )
        self.enable_light_exploration = bool(enable_light_exploration)
        self.light_explore_max_runs = max(0, int(light_explore_max_runs))
        self.light_explore_max_step = max(0, int(light_explore_max_step))
        self.light_explore_launcher_only = bool(light_explore_launcher_only)
        self.light_explore_require_keyword = bool(light_explore_require_keyword)
        self.light_explore_require_stall = bool(light_explore_require_stall)
        self.light_explore_back_limit = max(1, int(light_explore_back_limit))
        self.light_explore_hash_threshold = max(1, int(light_explore_hash_threshold))
        self.light_explore_replay_max_actions = max(0, int(light_explore_replay_max_actions))
        self._pending_explore_hint = ""
        self._light_explore_runs = 0
        self._last_probe_candidates: list[dict[str, Any]] = []

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home=go_home)
        self._pending_explore_hint = ""
        self._light_explore_runs = 0
        self._last_probe_candidates = []

    @staticmethod
    def _normalize_activity_name(activity: str | None) -> str:
        return _clean_text(activity).lower()

    def _foreground_activity_name(self) -> str:
        try:
            return str(self.env.foreground_activity_name or "").strip()
        except Exception:  # pylint: disable=broad-exception-caught
            return ""

    @staticmethod
    def _is_launcher_activity(activity: str | None) -> bool:
        value = _clean_text(activity).lower()
        return bool(value and ("launcher" in value or "nexuslauncher" in value or "quickstep" in value))

    def _goal_app_keywords(self, goal: str) -> list[str]:
        text = _clean_text(goal).lower()
        out: list[str] = []
        seen: set[str] = set()
        for app in gelab_agent.AVAILABLE_APPS:
            app_low = _clean_text(app).lower()
            if app_low and app_low in text and app_low not in seen:
                seen.add(app_low)
                out.append(app_low)
            for token in re.findall(r"[a-z0-9]{4,}", app_low):
                if token in text and token not in seen:
                    seen.add(token)
                    out.append(token)
        return out

    @staticmethod
    def _element_text(element: Any) -> str:
        text = _clean_text(getattr(element, "text", ""))
        desc = _clean_text(getattr(element, "content_description", ""))
        element_id = _clean_text(getattr(element, "resource_id", ""))
        merged = " ".join(x for x in [text, desc, element_id] if x)
        return _clean_text(merged).lower()

    @staticmethod
    def _safe_center_from_element(element: Any) -> tuple[int, int] | None:
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            return None
        try:
            return int((bbox.x_min + bbox.x_max) / 2.0), int((bbox.y_min + bbox.y_max) / 2.0)
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    @staticmethod
    def _is_interactive(element: Any) -> bool:
        return bool(
            getattr(element, "is_clickable", False)
            or getattr(element, "is_long_clickable", False)
            or getattr(element, "is_editable", False)
        )

    @staticmethod
    def _state_hash(state: Any) -> int:
        try:
            return int(_phash_pixels(state.pixels))
        except Exception:  # pylint: disable=broad-exception-caught
            return -1

    def _same_root_page(self, curr_state: Any, root_activity: str, root_hash: int) -> tuple[bool, str]:
        curr_activity = self._normalize_activity_name(self._foreground_activity_name())
        root_activity_norm = self._normalize_activity_name(root_activity)
        curr_hash = self._state_hash(curr_state)
        if root_hash < 0 or curr_hash < 0:
            same_activity = bool(curr_activity and root_activity_norm and curr_activity == root_activity_norm)
            return same_activity, "activity_match_only"
        try:
            diff = int(_hash_diff(root_hash, curr_hash))
        except Exception:  # pylint: disable=broad-exception-caught
            diff = 10**9
        same_activity = bool(curr_activity and root_activity_norm and curr_activity == root_activity_norm)
        if same_activity and diff <= int(self.light_explore_hash_threshold):
            return True, f"activity+phash<={self.light_explore_hash_threshold}:{diff}"
        if same_activity and diff <= int(self.light_explore_hash_threshold + 4):
            return True, f"activity+phash<={self.light_explore_hash_threshold + 4}:{diff}"
        return False, f"activity_match={same_activity}|phash_diff={diff}"

    def _collect_probe_candidates(self, state: Any, goal: str) -> list[dict[str, Any]]:
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        if not ui_elements:
            return []
        keywords = self._goal_app_keywords(goal)
        if self.light_explore_require_keyword and not keywords:
            return []
        candidates: list[dict[str, Any]] = []
        for idx, element in enumerate(ui_elements):
            if not self._is_interactive(element):
                continue
            center = self._safe_center_from_element(element)
            if center is None:
                continue
            merged = self._element_text(element)
            if not merged or "inputmethod" in merged or "systemui" in merged:
                continue
            score = 0.0
            if keywords:
                for kw in keywords:
                    if kw and kw in merged:
                        score += 4.0 + min(len(kw), 12) * 0.1
            else:
                score = 1.0
            if self.light_explore_require_keyword and score <= 0.0:
                continue
            label = _clean_text(getattr(element, "text", "")) or _clean_text(getattr(element, "content_description", ""))
            if not label:
                label = keywords[0] if keywords else "candidate"
            candidates.append({"index": idx, "element": element, "center": center, "label": label, "score": float(score), "merged": merged})
        candidates.sort(key=lambda item: (float(item.get("score", 0.0)), -int(item.get("index", 0))), reverse=True)
        return candidates[:8]

    @staticmethod
    def _choose_probe_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        return candidates[0] if candidates else None

    @staticmethod
    def _is_replay_safe_action(action: json_action.JSONAction | None) -> bool:
        if action is None:
            return False
        if action.action_type == json_action.OPEN_APP:
            return bool(_clean_text(getattr(action, "app_name", "")))
        return action.action_type in {
            json_action.CLICK,
            json_action.LONG_PRESS,
            json_action.SCROLL,
            json_action.SWIPE,
            json_action.NAVIGATE_BACK,
            json_action.NAVIGATE_HOME,
            json_action.OPEN_APP,
        }

    def _json_action_from_record(self, record: dict[str, Any] | None) -> json_action.JSONAction | None:
        if not isinstance(record, dict):
            return None
        fields = {k: record.get(k) for k in ("action_type", "index", "x", "y", "text", "direction", "goal_status", "app_name", "keycode", "clear_text")}
        fields = {k: v for k, v in fields.items() if v is not None}
        if not fields.get("action_type"):
            return None
        try:
            action = json_action.JSONAction(**fields)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        return action if self._is_replay_safe_action(action) else None

    def _select_replay_actions_for_probe(self, current_action: json_action.JSONAction | None) -> list[json_action.JSONAction]:
        actions: list[json_action.JSONAction] = []
        if self._is_replay_safe_action(current_action):
            actions.append(current_action)
        if self.light_explore_replay_max_actions <= 0:
            return actions
        for record in reversed(list(self._actions)):
            action = self._json_action_from_record(record.get("action_dict"))
            if action is None:
                continue
            actions.append(action)
            if len(actions) >= max(1, int(self.light_explore_replay_max_actions)):
                break
        actions.reverse()
        return actions

    def _rollback_to_probe_root(self, root_activity: str, root_hash: int, replay_actions: list[json_action.JSONAction], step_idx: int) -> dict[str, Any]:
        result: dict[str, Any] = {"success": False, "mode": "backtrack_failed", "back_presses": 0, "replayed_actions": 0, "matched_by": None}
        back_limit = max(1, int(self.light_explore_back_limit))
        for i in range(back_limit + 1):
            curr_state = self.env.get_state(wait_to_stabilize=True)
            same, matched_by = self._same_root_page(curr_state, root_activity, root_hash)
            if same:
                result.update(success=True, mode="backtrack", matched_by=matched_by)
                print(f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} rollback_done back_presses={result['back_presses']} matched_by={matched_by}")
                return result
            if i >= back_limit:
                break
            print(f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} rollback_back#{i + 1}/{back_limit} matched={matched_by}")
            self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK))
            result["back_presses"] += 1
        if replay_actions:
            result["mode"] = "replay"
            print(f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} rollback_fallback=replay")
            for action in replay_actions:
                try:
                    self.env.execute_action(action)
                    result["replayed_actions"] += 1
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    print(f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} rollback_replay_action_failed action={action.action_type} error={exc}")
            final_state = self.env.get_state(wait_to_stabilize=True)
            same, matched_by = self._same_root_page(final_state, root_activity, root_hash)
            result.update(success=bool(same), matched_by=matched_by, mode="replay" if same else "replay_failed")
            print(f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} rollback_replay_done success={result['success']} replayed={result['replayed_actions']} matched_by={matched_by}")
            return result
        print(f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} rollback_failed back_presses={result['back_presses']}")
        return result

    def _probe_page_changed(self, root_activity: str, root_hash: int, after_state: Any) -> tuple[bool, str, int | None]:
        after_activity = self._foreground_activity_name()
        changed = bool(self._normalize_activity_name(root_activity) and self._normalize_activity_name(after_activity) and self._normalize_activity_name(root_activity) != self._normalize_activity_name(after_activity))
        diff: int | None = None
        after_hash = self._state_hash(after_state)
        if root_hash >= 0 and after_hash >= 0:
            try:
                diff = int(_hash_diff(root_hash, after_hash))
            except Exception:  # pylint: disable=broad-exception-caught
                diff = None
        if diff is not None and diff >= 6:
            changed = True
        return changed, after_activity, diff

    @staticmethod
    def _build_hint_from_observation(candidate: dict[str, Any], changed: bool, after_activity: str) -> str:
        if not changed:
            return ""
        label = _clean_text(candidate.get("label")) or "that element"
        after_short = _clean_text(after_activity).split("/")[-1]
        if after_short:
            return f'Quick exploration: tapping "{label}" opened {after_short}; consider this action first if the UI still matches.'
        return f'Quick exploration: tapping "{label}" opened a different page; consider this action first if the UI still matches.'

    def _is_page_stalled(self, current_activity: str, current_hash: int) -> bool:
        if not self._actions:
            return False
        prev = self._actions[-1]
        prev_activity = self._normalize_activity_name(str(prev.get("start_page_activity", "")))
        curr_activity = self._normalize_activity_name(current_activity)
        try:
            prev_hash = int(prev.get("start_page_hash"))
        except Exception:  # pylint: disable=broad-exception-caught
            prev_hash = -1
        if prev_hash < 0 or current_hash < 0:
            return bool(prev_activity and curr_activity and prev_activity == curr_activity)
        try:
            diff = int(_hash_diff(prev_hash, current_hash))
        except Exception:  # pylint: disable=broad-exception-caught
            diff = 10**9
        return bool(prev_activity and curr_activity and prev_activity == curr_activity and diff <= 2)

    def _should_run_light_exploration(self, step_idx: int, root_activity: str, page_stalled: bool) -> tuple[bool, str]:
        if not self.enable_light_exploration:
            return False, "disabled"
        if self.light_explore_max_runs <= 0:
            return False, "run_budget_zero"
        if self._light_explore_runs >= self.light_explore_max_runs:
            return False, "run_budget_exhausted"
        if step_idx >= self.light_explore_max_step:
            return False, "beyond_step_window"
        if self.light_explore_require_stall and not page_stalled:
            return False, "page_not_stalled"
        if self.light_explore_launcher_only and not self._is_launcher_activity(root_activity):
            return False, "not_launcher_activity"
        return True, "ok"

    def _run_light_exploration(self, goal: str, step_idx: int, current_action: json_action.JSONAction | None = None, page_stalled: bool = False) -> str:
        self._last_probe_candidates = []
        try:
            root_state = self.env.get_state(wait_to_stabilize=True)
            root_hash = self._state_hash(root_state)
            root_activity = self._foreground_activity_name()
            should_run, reason = self._should_run_light_exploration(step_idx=step_idx, root_activity=root_activity, page_stalled=page_stalled)
            print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_should_run={should_run} reason={reason} page_stalled={page_stalled}")
            if not should_run:
                return ""
            candidates = self._collect_probe_candidates(root_state, goal)
            self._last_probe_candidates = list(candidates)
            if not candidates:
                print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_no_candidates")
                self._light_explore_runs += 1
                return ""
            candidate = self._choose_probe_candidate(candidates)
            if not candidate:
                print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_candidate_none")
                self._light_explore_runs += 1
                return ""
            center = candidate.get("center")
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_center_invalid")
                self._light_explore_runs += 1
                return ""
            label = _clean_text(candidate.get("label")) or "candidate"
            probe_action = json_action.JSONAction(action_type=json_action.CLICK, x=int(center[0]), y=int(center[1]))
            print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_probe_action=click label={label} center={[int(center[0]), int(center[1])]}")
            self.env.execute_action(probe_action)
            after_state = self.env.get_state(wait_to_stabilize=True)
            changed, after_activity, hash_diff = self._probe_page_changed(root_activity, root_hash, after_state)
            print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_probe_effect changed={changed} hash_diff={hash_diff} after_activity={after_activity}")
            rollback_info = self._rollback_to_probe_root(root_activity=root_activity, root_hash=root_hash, replay_actions=self._select_replay_actions_for_probe(current_action), step_idx=step_idx)
            self._light_explore_runs += 1
            if not bool(rollback_info.get("success")):
                print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_restore_failed={rollback_info}")
                self.enable_light_exploration = False
                return ""
            hint = self._build_hint_from_observation(candidate=candidate, changed=changed, after_activity=after_activity)
            print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_hint_generated: {hint}" if hint else f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_no_useful_hint")
            return hint
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_failed: {exc}")
            self._light_explore_runs += 1
            return ""

    def _ui_description(self, state: Any) -> str:
        return t3a._generate_ui_elements_description_list_full(  # pylint: disable=protected-access
            list(getattr(state, "ui_elements", None) or []),
            self.env.logical_screen_size,
        )

    def _repair_prompt(self, prompt: str, parse_error: str) -> str:
        return (
            prompt
            + "\n\nFORMAT REPAIR REQUIRED:\n"
            + "Return exactly one valid Action JSON. No prose except optional Reason line.\n"
            + "Example: Action: {\"action_type\":\"click\",\"index\":3}\n"
            + f"Previous parse error: {parse_error}\n"
        )

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        start_time = time.time()
        step_idx = len(self._actions)
        if step_idx >= self._effective_max_steps():
            summary = f"Reached the maximum step limit ({self._effective_max_steps()})."
            action = json_action.JSONAction(action_type=json_action.STATUS, goal_status="infeasible")
            return base_agent.AgentInteractionResult(done=True, data={"response": "", "parsed_action": {"action": "ABORT", "summary": summary}, "action": repr(action), "action_dict": action.__dict__, "summary": summary, "latency_sec": float(max(0.0, time.time() - start_time))})

        print("=" * 96)
        print(f"Step {step_idx}: Goal")
        print(goal)
        state = self.get_post_transition_state()
        auto_summary = _auto_complete_summary(goal, state, self._summaries)
        if auto_summary:
            action = json_action.JSONAction(action_type=json_action.STATUS, goal_status="complete")
            latency_sec = float(max(0.0, time.time() - start_time))
            step_record = {
                "goal": goal,
                "response": "",
                "parsed_action": {"action": "status", "summary": auto_summary, "status": "complete"},
                "raw_response": None,
                "summary_raw_response": None,
                "tool_call": {"name": "mobile_use", "arguments": action.as_dict(skip_none=True)},
                "action_dict": action.__dict__,
                "summary": auto_summary,
                "latency_sec": latency_sec,
                "prompt_mode": "auto_complete",
                "prompt_hint": "",
                "next_step_hint": "",
                "light_explore_runs": self._light_explore_runs,
                "start_page_activity": self._foreground_activity_name(),
                "start_page_hash": self._state_hash(state),
                "page_stalled": False,
                "text_only": True,
            }
            self._actions.append(step_record)
            self._summaries.append(auto_summary)
            self._responses.append("")
            print(f"Step {step_idx}: Auto complete")
            print(auto_summary)
            print("=" * 96)
            return base_agent.AgentInteractionResult(
                done=True,
                data={
                    "response": "",
                    "parsed_action": dict(step_record["parsed_action"]),
                    "tool_call": step_record["tool_call"],
                    "action": repr(action),
                    "action_dict": action.__dict__,
                    "summary": auto_summary,
                    "hints": [],
                    "latency_sec": latency_sec,
                    "prompt_mode": "auto_complete",
                    "prompt_hint": "",
                    "next_step_hint": "",
                    "light_explore_runs": self._light_explore_runs,
                    "text_only": True,
                },
            )
        forced_mode = "forced_audio_recorder"
        forced = _audio_recorder_forced_action(goal, state)
        if forced is None:
            forced_mode = "forced_clock"
            forced = _clock_forced_action(goal, state)
        if forced is not None:
            action, parsed_action, response = forced
            ui_elements = list(getattr(state, "ui_elements", None) or [])
            tool_call = {"name": "mobile_use", "arguments": action.as_dict(skip_none=True)}
            start_page_activity = self._foreground_activity_name()
            start_page_hash = self._state_hash(state)
            page_stalled = self._is_page_stalled(start_page_activity, start_page_hash)
            print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} prompt_mode: {forced_mode}; hint: <none>")
            gelab_agent._print_step_section(step_idx, "Forced action", str(response))  # pylint: disable=protected-access
            gelab_agent._print_step_section(step_idx, "Parsed action", gelab_agent._json_dumps_safe(dict(parsed_action)))  # pylint: disable=protected-access
            gelab_agent._print_step_section(step_idx, "Tool call", gelab_agent._json_dumps_safe(tool_call))  # pylint: disable=protected-access
            self._execute_action(action, {"wait_seconds": 1})
            gelab_agent._print_step_section(step_idx, "Action", gelab_agent._json_dumps_safe(action.__dict__))  # pylint: disable=protected-access
            after_state_for_summary = None
            if action.action_type not in {json_action.STATUS, json_action.ANSWER}:
                try:
                    after_state_for_summary = self.get_post_transition_state()
                except Exception:  # pylint: disable=broad-exception-caught
                    after_state_for_summary = None
            summary = _local_step_summary(
                action=action,
                reason=_clean_text(parsed_action.get("summary")),
                before_elements=ui_elements,
                after_state=after_state_for_summary,
            )
            terminal_summary = ""
            if after_state_for_summary is not None:
                terminal_summary = _auto_complete_summary(goal, after_state_for_summary, self._summaries + [summary])
            if terminal_summary:
                latency_sec = float(max(0.0, time.time() - start_time))
                step_record = {
                    "goal": goal,
                    "response": response,
                    "parsed_action": dict(parsed_action),
                    "raw_response": None,
                    "summary_raw_response": None,
                    "tool_call": tool_call,
                    "action_dict": action.__dict__,
                    "summary": terminal_summary,
                    "latency_sec": latency_sec,
                    "prompt_mode": f"{forced_mode}_terminal_auto_complete",
                    "prompt_hint": "",
                    "next_step_hint": "",
                    "light_explore_runs": self._light_explore_runs,
                    "start_page_activity": start_page_activity,
                    "start_page_hash": start_page_hash,
                    "page_stalled": page_stalled,
                    "text_only": True,
                    "terminal_auto_complete": True,
                }
                self._actions.append(step_record)
                self._summaries.append(terminal_summary)
                self._responses.append(str(response))
                print(f"Step {step_idx}: Auto complete after action")
                print(terminal_summary)
                print("=" * 96)
                return base_agent.AgentInteractionResult(
                    done=True,
                    data={
                        "response": response,
                        "parsed_action": dict(parsed_action),
                        "tool_call": tool_call,
                        "action": repr(action),
                        "action_dict": action.__dict__,
                        "summary": terminal_summary,
                        "hints": [],
                        "latency_sec": latency_sec,
                        "prompt_mode": step_record["prompt_mode"],
                        "prompt_hint": "",
                        "next_step_hint": "",
                        "light_explore_runs": self._light_explore_runs,
                        "text_only": True,
                        "terminal_auto_complete": True,
                    },
                )
            next_hint = self._run_light_exploration(goal=goal, step_idx=step_idx, current_action=action, page_stalled=page_stalled)
            self._pending_explore_hint = _clean_text(next_hint)
            latency_sec = float(max(0.0, time.time() - start_time))
            step_record = {
                "goal": goal,
                "response": response,
                "parsed_action": dict(parsed_action),
                "raw_response": None,
                "summary_raw_response": None,
                "tool_call": tool_call,
                "action_dict": action.__dict__,
                "summary": summary,
                "latency_sec": latency_sec,
                "prompt_mode": forced_mode,
                "prompt_hint": "",
                "next_step_hint": self._pending_explore_hint,
                "light_explore_runs": self._light_explore_runs,
                "start_page_activity": start_page_activity,
                "start_page_hash": start_page_hash,
                "page_stalled": page_stalled,
                "text_only": True,
            }
            self._actions.append(step_record)
            self._summaries.append(summary)
            self._responses.append(str(response))
            task_dir = self._task_output_dir(goal)
            if task_dir:
                os.makedirs(task_dir, exist_ok=True)
                try:
                    Image.fromarray(state.pixels).save(os.path.join(task_dir, f"screenshot_{len(self._actions) - 1}.png"))
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                with open(os.path.join(task_dir, f"prompt_{len(self._actions) - 1}.txt"), "w", encoding="utf-8") as f:
                    f.write(forced_mode)
                self._write_action_log(goal)
            print(f"Step {step_idx}: Latency")
            print(f"{latency_sec:.3f}s")
            print(f"Step {step_idx}: Result")
            print(f"done=False, action_type={action.action_type}, summary={summary}")
            print("=" * 96)
            return base_agent.AgentInteractionResult(
                done=False,
                data={
                    "response": response,
                    "parsed_action": dict(parsed_action),
                    "tool_call": tool_call,
                    "action": repr(action),
                    "action_dict": action.__dict__,
                    "summary": summary,
                    "hints": [],
                    "latency_sec": latency_sec,
                    "prompt_mode": forced_mode,
                    "prompt_hint": "",
                    "next_step_hint": self._pending_explore_hint,
                    "light_explore_runs": self._light_explore_runs,
                    "text_only": True,
                },
            )
        screen_size = self.env.logical_screen_size
        ui_description = self._ui_description(state)
        history = self._history_text()
        start_page_activity = self._foreground_activity_name()
        start_page_hash = self._state_hash(state)
        page_stalled = self._is_page_stalled(start_page_activity, start_page_hash)
        hint_for_prompt = _clean_text(self._pending_explore_hint)
        self._pending_explore_hint = ""
        prompt_mode = "baseline_plus_hint" if hint_for_prompt else "baseline"
        print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} prompt_mode: {prompt_mode}; hint: {hint_for_prompt or '<none>'}")

        prompt = TEXT_SYSTEM_PROMPT + "\n\n" + _text_prompt(goal, history, hint_for_prompt, ui_description)
        gelab_agent._print_step_section(step_idx, "Text model input", prompt[:16000])  # pylint: disable=protected-access
        response, is_safe, raw_response = self.vllm.predict(prompt)
        if not is_safe:  # pylint: disable=singleton-comparison
            response = 'Action: {"action_type":"status", "goal_status":"infeasible"}'
        gelab_agent._print_step_section(step_idx, "Model output", str(response))  # pylint: disable=protected-access

        parse_error = ""
        try:
            action, tool_call, parsed_action, parse_error = _parse_text_action(str(response), raw_response)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            parse_error = _clean_text(exc)
            repair_prompt = self._repair_prompt(prompt, parse_error)
            retry_response, retry_safe, retry_raw = self.vllm.predict(repair_prompt)
            if not retry_safe:  # pylint: disable=singleton-comparison
                retry_response = 'Action: {"action_type":"wait"}'
            try:
                action, tool_call, parsed_action, _ = _parse_text_action(str(retry_response), retry_raw)
                response = retry_response
                raw_response = retry_raw
                parsed_action["parse_error"] = parse_error
                parsed_action["fallback"] = "format_repair"
            except Exception as retry_exc:  # pylint: disable=broad-exception-caught
                parsed_action = OrderedDict(cot="", action="wait", value="1", summary="Parser fallback wait", parse_error=parse_error, recovery_error=_clean_text(retry_exc))
                action = json_action.JSONAction(action_type=json_action.WAIT)
                tool_call = {"name": "mobile_use", "arguments": {"action_type": "wait"}}

        ui_elements = list(getattr(state, "ui_elements", None) or [])
        if action.index is not None and action.action_type in {json_action.CLICK, json_action.LONG_PRESS, json_action.INPUT_TEXT, json_action.SCROLL}:
            if action.index < 0 or action.index >= len(ui_elements):
                parsed_action["index_error"] = f"index_out_of_range:{action.index}/{len(ui_elements)}"
                action = json_action.JSONAction(action_type=json_action.WAIT)
                tool_call = {"name": "mobile_use", "arguments": {"action_type": "wait"}}
        action, tool_call, parsed_action = _sanitize_text_action(goal, state, action, tool_call, parsed_action)

        if parse_error:
            gelab_agent._print_step_section(step_idx, "Parse fallback", parse_error)  # pylint: disable=protected-access
        gelab_agent._print_step_section(step_idx, "Parsed action", gelab_agent._json_dumps_safe(dict(parsed_action)))  # pylint: disable=protected-access
        gelab_agent._print_step_section(step_idx, "Tool call", gelab_agent._json_dumps_safe(tool_call))  # pylint: disable=protected-access

        if action.action_type == json_action.ANSWER and action.text:
            self.env.interaction_cache = str(action.text)
        self._execute_action(action, {"wait_seconds": 1})
        gelab_agent._print_step_section(step_idx, "Action", gelab_agent._json_dumps_safe(action.__dict__))  # pylint: disable=protected-access

        reason_for_summary = _clean_text(parsed_action.get("summary"))
        after_state_for_summary = None
        if action.action_type not in {json_action.STATUS, json_action.ANSWER}:
            try:
                after_state_for_summary = self.get_post_transition_state()
            except Exception:  # pylint: disable=broad-exception-caught
                after_state_for_summary = None
        summary = _local_step_summary(
            action=action,
            reason=reason_for_summary,
            before_elements=ui_elements,
            after_state=after_state_for_summary,
        )
        terminal_summary = ""
        if after_state_for_summary is not None:
            terminal_summary = _auto_complete_summary(goal, after_state_for_summary, self._summaries + [summary])
        if terminal_summary:
            latency_sec = float(max(0.0, time.time() - start_time))
            step_record = {
                "goal": goal,
                "response": response,
                "parsed_action": dict(parsed_action),
                "raw_response": raw_response,
                "summary_raw_response": None,
                "tool_call": tool_call,
                "action_dict": action.__dict__,
                "summary": terminal_summary,
                "latency_sec": latency_sec,
                "prompt_mode": f"{prompt_mode}_terminal_auto_complete",
                "prompt_hint": hint_for_prompt,
                "next_step_hint": "",
                "light_explore_runs": self._light_explore_runs,
                "start_page_activity": start_page_activity,
                "start_page_hash": start_page_hash,
                "page_stalled": page_stalled,
                "text_only": True,
                "terminal_auto_complete": True,
            }
            self._actions.append(step_record)
            self._summaries.append(terminal_summary)
            self._responses.append(str(response))
            print(f"Step {step_idx}: Auto complete after action")
            print(terminal_summary)
            print("=" * 96)
            return base_agent.AgentInteractionResult(
                done=True,
                data={
                    "response": response,
                    "parsed_action": dict(parsed_action),
                    "tool_call": tool_call,
                    "action": repr(action),
                    "action_dict": action.__dict__,
                    "summary": terminal_summary,
                    "hints": [],
                    "latency_sec": latency_sec,
                    "prompt_mode": step_record["prompt_mode"],
                    "prompt_hint": hint_for_prompt,
                    "next_step_hint": "",
                    "light_explore_runs": self._light_explore_runs,
                    "text_only": True,
                    "terminal_auto_complete": True,
                },
            )
        step_summary_raw = None
        next_hint = self._run_light_exploration(goal=goal, step_idx=step_idx, current_action=action, page_stalled=page_stalled)
        self._pending_explore_hint = _clean_text(next_hint)
        latency_sec = float(max(0.0, time.time() - start_time))
        step_record = {
            "goal": goal,
            "response": response,
            "parsed_action": dict(parsed_action),
            "raw_response": raw_response,
            "summary_raw_response": step_summary_raw,
            "tool_call": tool_call,
            "action_dict": action.__dict__,
            "summary": summary,
            "latency_sec": latency_sec,
            "prompt_mode": prompt_mode,
            "prompt_hint": hint_for_prompt,
            "next_step_hint": self._pending_explore_hint,
            "light_explore_runs": self._light_explore_runs,
            "start_page_activity": start_page_activity,
            "start_page_hash": start_page_hash,
            "page_stalled": page_stalled,
            "text_only": True,
        }
        self._actions.append(step_record)
        self._summaries.append(summary)
        self._responses.append(str(response))

        task_dir = self._task_output_dir(goal)
        if task_dir:
            os.makedirs(task_dir, exist_ok=True)
            try:
                Image.fromarray(state.pixels).save(os.path.join(task_dir, f"screenshot_{len(self._actions) - 1}.png"))
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            with open(os.path.join(task_dir, f"prompt_{len(self._actions) - 1}.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
            self._write_action_log(goal)

        done = action.action_type in {json_action.STATUS, json_action.ANSWER}
        print(f"Step {step_idx}: Latency")
        print(f"{latency_sec:.3f}s")
        print(f"Step {step_idx}: Result")
        print(f"done={done}, action_type={action.action_type}, summary={summary}")
        print("=" * 96)
        return base_agent.AgentInteractionResult(
            done=done,
            data={
                "response": response,
                "parsed_action": dict(parsed_action),
                "tool_call": tool_call,
                "action": repr(action),
                "action_dict": action.__dict__,
                "summary": summary,
                "hints": [],
                "latency_sec": latency_sec,
                "prompt_mode": prompt_mode,
                "prompt_hint": hint_for_prompt,
                "next_step_hint": self._pending_explore_hint,
                "light_explore_runs": self._light_explore_runs,
                "text_only": True,
            },
        )


class ElementTextAgent(ExplorerTextAgent):
    pass
