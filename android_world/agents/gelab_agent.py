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

"""GELAB-style agent adapted for AndroidWorld."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from collections import OrderedDict
from typing import Any

from PIL import Image

from android_world.agents import base_agent
from android_world.agents import seeact_utils
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import json_action


AVAILABLE_APPS = [
    "Camera",
    "Chrome",
    "Clock",
    "Contacts",
    "Dialer",
    "Files",
    "Settings",
    "Markor",
    "Tasks",
    "Simple Draw Pro",
    "Simple Gallery Pro",
    "Simple SMS Messenger",
    "Audio Recorder",
    "Pro Expense",
    "Broccoli APP",
    "OSMand",
    "VLC",
    "Joplin",
    "Retro Music",
    "OpenTracks",
    "Simple Calendar Pro",
]

MAX_AGENT_STEPS = 20


GELAB_SYSTEM_PROMPT = """You are a mobile GUI agent for AndroidWorld.
You receive the user task, the current screenshot, concise operation history, and optional UI hints.
You must choose exactly one next action to move the task forward.

Coordinate system:
- The screenshot coordinate origin is the top-left corner.
- Use normalized coordinates in the range 0-1000 for both x and y.

Action space:
1. CLICK: click one point.
   Format: action:CLICK\tpoint:x,y
2. TYPE: type text into a field.
   Format: action:TYPE\tvalue:text to type\tpoint:x,y
3. COMPLETE: finish the task and report the result.
   Format: action:COMPLETE\treturn:final result for the user
5. AWAKE: open an app quickly.
   Format: action:AWAKE\tvalue:app name
6. INFO: ask the user for missing information.
   Format: action:INFO\tvalue:question for the user
7. ABORT: stop only if the task is impossible or unsafe.
   Format: action:ABORT\tvalue:reason
8. SLIDE: swipe from one point to another.
   Format: action:SLIDE\tpoint1:x1,y1\tpoint2:x2,y2
9. LONGPRESS: long-press one point.
   Format: action:LONGPRESS\tpoint:x,y

Output format:
<THINK>brief reasoning</THINK>
explain:short purpose of this action\taction:ACTION_NAME\t...parameters...\tsummary:updated one-line task progress summary

Rules:
- Track your previous action mentally. Do not slide in the same way more than 5 times in a row.
- Follow the latest user instruction strictly.
- Prefer AWAKE when opening an app is the fastest valid move.
- Prefer COMPLETE when the task is done. Use ABORT only when the task truly cannot continue.
- Use INFO only when progress is blocked by missing user information.
- Return only the required format. Do not wrap it in Markdown.
- Available apps: """ + json.dumps(AVAILABLE_APPS, ensure_ascii=True) + """
""".strip()


def _normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _now_hms() -> str:
    return time.strftime("%H:%M:%S")


def _print_step_section(step_idx: int, title: str, text: str) -> None:
    print(f"Step {step_idx}: {title}")
    print(text)


def _messages_text_for_logging(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        content = message.get("content")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text_parts.append(str(item.get("text") or ""))
            if text_parts:
                lines.append(f"[GELAB text input | {role}]")
                lines.append("\n".join(text_parts).strip())
        elif isinstance(content, str) and content.strip():
            lines.append(f"[GELAB text input | {role}]")
            lines.append(content.strip())
    return "\n".join(lines).strip()


def _json_dumps_safe(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:  # pylint: disable=broad-exception-caught
        return str(value)


def _extract_tool_call_payload(text: str) -> str:
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(text or "").strip()


def _strip_code_fences(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
    return stripped


def _safe_json_loads(text: str) -> Any:
    raw = _strip_code_fences(_extract_tool_call_payload(text))
    for candidate in [raw]:
        try:
            return json.loads(candidate)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = raw[first:last + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    raise seeact_utils.ParseActionError("cannot parse tool-call JSON")


def _extract_coordinate(arguments: dict[str, Any], *keys: str) -> list[int] | None:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return [int(round(float(value[0]))), int(round(float(value[1])))]
            except Exception:  # pylint: disable=broad-exception-caught
                continue
    return None


def _normalize_tool_call(tool_obj: Any) -> dict[str, Any]:
    if not isinstance(tool_obj, dict):
        raise seeact_utils.ParseActionError("tool-call payload is not a JSON object")

    if isinstance(tool_obj.get("arguments"), dict):
        args = dict(tool_obj["arguments"])
        name = str(tool_obj.get("name") or "mobile_use")
    else:
        args = dict(tool_obj)
        name = "mobile_use"

    action_name = str(args.get("action") or tool_obj.get("name") or "").strip().lower()
    if not action_name:
        raise seeact_utils.ParseActionError("tool-call JSON missing action")

    normalized_args: dict[str, Any] = {"action": action_name}

    if action_name in ("click", "tap"):
        coord = _extract_coordinate(args, "coordinate", "point", "coordinates")
        if coord is None:
            raise seeact_utils.ParseActionError("tool-call click missing coordinate")
        normalized_args["action"] = "click"
        normalized_args["coordinate"] = coord
    elif action_name in ("long_press", "longpress"):
        coord = _extract_coordinate(args, "coordinate", "point", "coordinates")
        if coord is None:
            raise seeact_utils.ParseActionError("tool-call long_press missing coordinate")
        normalized_args["action"] = "long_press"
        normalized_args["coordinate"] = coord
    elif action_name == "type":
        normalized_args["action"] = "type"
        normalized_args["text"] = str(args.get("text") or args.get("value") or "")
        coord = _extract_coordinate(args, "coordinate", "point", "coordinates")
        if coord is not None:
            normalized_args["coordinate"] = coord
    elif action_name in ("swipe", "slide"):
        normalized_args["action"] = "swipe"
        direction = _normalize_space(args.get("direction")).lower()
        start_coordinate = _extract_coordinate(args, "start_coordinate", "point1")
        end_coordinate = _extract_coordinate(args, "end_coordinate", "point2")
        if not direction and start_coordinate is not None and end_coordinate is not None:
            direction = _infer_direction(start_coordinate, end_coordinate)
        if not direction:
            raise seeact_utils.ParseActionError("tool-call swipe missing direction")
        normalized_args["direction"] = direction
        if start_coordinate is not None and end_coordinate is not None:
            normalized_args["start_coordinate"] = start_coordinate
            normalized_args["end_coordinate"] = end_coordinate
    elif action_name in ("open_app", "open", "awake"):
        normalized_args["action"] = "open_app"
        normalized_args["text"] = _normalize_space(args.get("text") or args.get("value") or args.get("app_name"))
    elif action_name == "system_button":
        normalized_args["action"] = "system_button"
        normalized_args["button"] = _normalize_space(args.get("button")).lower() or "back"
    elif action_name in ("back", "home", "enter"):
        normalized_args["action"] = "system_button"
        normalized_args["button"] = action_name
    elif action_name == "wait":
        normalized_args["action"] = "wait"
        normalized_args["value"] = max(1, _safe_int(args.get("value")) or 1)
    elif action_name in ("answer", "info"):
        normalized_args["action"] = "answer"
        normalized_args["text"] = _normalize_space(args.get("text") or args.get("value"))
    elif action_name in ("terminate", "status", "complete", "abort"):
        normalized_args["action"] = "terminate"
        goal_status = _normalize_space(args.get("goal_status") or args.get("status")).lower()
        if action_name == "abort" or goal_status in ("fail", "failed", "infeasible"):
            normalized_args["status"] = "fail"
        else:
            normalized_args["status"] = "success"
        result_text = _normalize_space(args.get("text") or args.get("value") or args.get("return"))
        if result_text:
            normalized_args["text"] = result_text
    else:
        raise seeact_utils.ParseActionError(f"unsupported tool-call action: {action_name}")

    return {"name": name, "arguments": normalized_args}


def _safe_task_name(goal: str, max_len: int = 72) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(goal or "").strip())
    value = re.sub(r"_+", "_", value).strip("._")
    return (value or "task")[:max_len]


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _clip_norm_point(point: list[int]) -> list[int]:
    return [max(0, min(1000, int(point[0]))), max(0, min(1000, int(point[1])))]


def _parse_point(text: str) -> list[int]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text or ""))
    if len(nums) < 2:
        raise seeact_utils.ParseActionError(f"invalid point: {text}")
    return _clip_norm_point([int(round(float(nums[0]))), int(round(float(nums[1])))])


def _norm_to_abs(point: list[int], screen_size: tuple[int, int]) -> list[int]:
    width, height = screen_size
    width = max(1, int(width))
    height = max(1, int(height))
    x_norm, y_norm = _clip_norm_point(point)
    x = int(round((x_norm / 1000.0) * max(0, width - 1)))
    y = int(round((y_norm / 1000.0) * max(0, height - 1)))
    return [x, y]


def _infer_direction(point1: list[int], point2: list[int]) -> str:
    dx = int(point2[0]) - int(point1[0])
    dy = int(point2[1]) - int(point1[1])
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _normalize_think_tags(text: str) -> str:
    out = str(text or "").strip()
    out = out.replace("<TINK>", "<THINK>").replace("</TINK>", "</THINK>")
    out = out.replace("<think>", "<THINK>").replace("</think>", "</THINK>")
    return re.sub(
        r"<\s*/?\s*think\s*>",
        lambda m: "</THINK>" if "/" in m.group() else "<THINK>",
        out,
        flags=re.IGNORECASE,
    )


def _extract_think_and_kv(text: str) -> tuple[str, str]:
    normalized = _normalize_think_tags(text)
    think_match = re.search(r"<THINK>(.*?)</THINK>", normalized, flags=re.IGNORECASE | re.DOTALL)
    if think_match:
        cot = _normalize_space(think_match.group(1))
        kv_text = normalized[think_match.end() :].strip()
        return cot, kv_text
    return "", normalized


def _parse_key_values(kv_text: str) -> OrderedDict[str, Any]:
    fields = [
        "explain",
        "action",
        "action_type",
        "summary",
        "value",
        "return",
        "point",
        "point1",
        "point2",
    ]
    pattern = re.compile(
        r"(^|[\t\n])\s*(%s)\s*:" % "|".join(re.escape(field) for field in fields),
        flags=re.IGNORECASE,
    )
    matches = list(pattern.finditer(kv_text))
    data: OrderedDict[str, Any] = OrderedDict()
    if not matches:
        raise seeact_utils.ParseActionError("cannot find GELAB key-value fields")
    for idx, match in enumerate(matches):
        key = match.group(2).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(kv_text)
        value = kv_text[start:end].strip(" \t\r\n")
        data[key] = value
    return data


def _parse_tool_call_response(response_text: str, cot: str = "") -> OrderedDict[str, Any]:
    tool_call = _normalize_tool_call(_safe_json_loads(response_text))
    parsed = OrderedDict()
    parsed["cot"] = cot
    parsed["action"] = "__TOOL_CALL__"
    parsed["source_format"] = "tool_call_json"
    parsed["tool_call"] = tool_call
    args = tool_call["arguments"]
    if "text" in args:
        parsed["value"] = args["text"]
    if "coordinate" in args:
        parsed["coordinate"] = list(args["coordinate"])
    if "start_coordinate" in args:
        parsed["start_coordinate"] = list(args["start_coordinate"])
    if "end_coordinate" in args:
        parsed["end_coordinate"] = list(args["end_coordinate"])
    if "direction" in args:
        parsed["direction"] = args["direction"]
    if "button" in args:
        parsed["button"] = args["button"]
    if "status" in args:
        parsed["status"] = args["status"]
    return parsed


def parse_gelab_response(response_text: str) -> OrderedDict[str, Any]:
    if not response_text:
        raise seeact_utils.ParseActionError("empty GELAB response")
    cot, kv_text = _extract_think_and_kv(response_text)
    try:
        action = _parse_key_values(kv_text)
        action["cot"] = cot
        if "action" not in action and "action_type" in action:
            action["action"] = action["action_type"]
        if "action" not in action:
            raise seeact_utils.ParseActionError("missing GELAB action field")
        action["action"] = str(action["action"]).strip().upper()
        if "point" in action:
            action["point"] = _parse_point(action["point"])
        if "point1" in action:
            action["point1"] = _parse_point(action["point1"])
        if "point2" in action:
            action["point2"] = _parse_point(action["point2"])
        return action
    except seeact_utils.ParseActionError as kv_error:
        try:
            return _parse_tool_call_response(response_text, cot=cot)
        except seeact_utils.ParseActionError as tool_error:
            raise seeact_utils.ParseActionError(
                f"{kv_error}; tool-call fallback failed: {tool_error}"
            ) from tool_error


def gelab_action_to_json_action(
    parsed_action: OrderedDict[str, Any],
    screen_size: tuple[int, int],
) -> tuple[json_action.JSONAction, dict[str, Any], dict[str, Any]]:
    action_type = str(parsed_action.get("action") or "").strip().upper()
    extras: dict[str, Any] = {}
    tool_call: dict[str, Any] = {"name": "mobile_use", "arguments": {}}

    if action_type == "__TOOL_CALL__":
        tool_call = dict(parsed_action.get("tool_call") or tool_call)
        arguments = dict(tool_call.get("arguments") or {})
        tool_action = str(arguments.get("action") or "").strip().lower()

        if tool_action == "click":
            coordinate = _extract_coordinate(arguments, "coordinate")
            if coordinate is None:
                raise seeact_utils.ParseActionError("tool-call click missing coordinate")
            return (
                json_action.JSONAction(action_type=json_action.CLICK, x=coordinate[0], y=coordinate[1]),
                tool_call,
                extras,
            )

        if tool_action == "long_press":
            coordinate = _extract_coordinate(arguments, "coordinate")
            if coordinate is None:
                raise seeact_utils.ParseActionError("tool-call long_press missing coordinate")
            return (
                json_action.JSONAction(action_type=json_action.LONG_PRESS, x=coordinate[0], y=coordinate[1]),
                tool_call,
                extras,
            )

        if tool_action == "type":
            text = str(arguments.get("text") or "")
            coordinate = _extract_coordinate(arguments, "coordinate")
            if coordinate is None:
                return json_action.JSONAction(action_type=json_action.INPUT_TEXT, text=text), tool_call, extras
            return (
                json_action.JSONAction(
                    action_type=json_action.INPUT_TEXT,
                    x=coordinate[0],
                    y=coordinate[1],
                    text=text,
                ),
                tool_call,
                extras,
            )

        if tool_action == "swipe":
            direction = _normalize_space(arguments.get("direction")).lower()
            start_coordinate = _extract_coordinate(arguments, "start_coordinate")
            end_coordinate = _extract_coordinate(arguments, "end_coordinate")
            if not direction and start_coordinate is not None and end_coordinate is not None:
                direction = _infer_direction(start_coordinate, end_coordinate)
            if not direction:
                raise seeact_utils.ParseActionError("tool-call swipe missing direction")
            if start_coordinate is not None and end_coordinate is not None:
                extras["start_coordinate"] = start_coordinate
                extras["end_coordinate"] = end_coordinate
            extras["direction"] = direction
            return json_action.JSONAction(action_type=json_action.SWIPE, direction=direction), tool_call, extras

        if tool_action == "open_app":
            app_name = _normalize_space(arguments.get("text"))
            if not app_name:
                raise seeact_utils.ParseActionError("tool-call open_app missing text")
            return (
                json_action.JSONAction(action_type=json_action.OPEN_APP, app_name=app_name),
                tool_call,
                extras,
            )

        if tool_action == "system_button":
            button = _normalize_space(arguments.get("button")).lower()
            if button == "back":
                return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK), tool_call, extras
            if button == "home":
                return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME), tool_call, extras
            if button == "enter":
                return json_action.JSONAction(action_type=json_action.KEYBOARD_ENTER), tool_call, extras
            raise seeact_utils.ParseActionError(f"unsupported system button: {button}")

        if tool_action == "wait":
            wait_value = max(1, _safe_int(arguments.get("value")) or 1)
            extras["wait_seconds"] = wait_value
            return json_action.JSONAction(action_type=json_action.WAIT), tool_call, extras

        if tool_action == "answer":
            text = _normalize_space(arguments.get("text"))
            return json_action.JSONAction(action_type=json_action.ANSWER, text=text), tool_call, extras

        if tool_action == "terminate":
            status = _normalize_space(arguments.get("status")).lower()
            result_text = _normalize_space(arguments.get("text"))
            if result_text:
                extras["return_text"] = result_text
            goal_status = "infeasible" if status in ("fail", "failed", "infeasible") else "task_complete"
            return (
                json_action.JSONAction(action_type=json_action.STATUS, goal_status=goal_status),
                tool_call,
                extras,
            )

        raise seeact_utils.ParseActionError(f"unsupported tool-call action: {tool_action}")

    if action_type == "CLICK":
        point = parsed_action.get("point")
        if point is None:
            raise seeact_utils.ParseActionError("CLICK requires point")
        abs_point = _norm_to_abs(point, screen_size)
        tool_call["arguments"] = {"action": "click", "coordinate": abs_point}
        return (
            json_action.JSONAction(action_type=json_action.CLICK, x=abs_point[0], y=abs_point[1]),
            tool_call,
            extras,
        )

    if action_type == "LONGPRESS":
        point = parsed_action.get("point")
        if point is None:
            raise seeact_utils.ParseActionError("LONGPRESS requires point")
        abs_point = _norm_to_abs(point, screen_size)
        tool_call["arguments"] = {"action": "long_press", "coordinate": abs_point}
        return (
            json_action.JSONAction(action_type=json_action.LONG_PRESS, x=abs_point[0], y=abs_point[1]),
            tool_call,
            extras,
        )

    if action_type == "TYPE":
        value = str(parsed_action.get("value") or "")
        if not value:
            raise seeact_utils.ParseActionError("TYPE requires value")
        point = parsed_action.get("point")
        if point is None:
            raise seeact_utils.ParseActionError("TYPE requires point")
        abs_point = _norm_to_abs(point, screen_size)
        tool_call["arguments"] = {"action": "type", "text": value, "coordinate": abs_point}
        return (
            json_action.JSONAction(
                action_type=json_action.INPUT_TEXT,
                x=abs_point[0],
                y=abs_point[1],
                text=value,
            ),
            tool_call,
            extras,
        )

    if action_type == "SLIDE":
        point1 = parsed_action.get("point1")
        point2 = parsed_action.get("point2")
        if point1 is None or point2 is None:
            raise seeact_utils.ParseActionError("SLIDE requires point1 and point2")
        start_abs = _norm_to_abs(point1, screen_size)
        end_abs = _norm_to_abs(point2, screen_size)
        direction = _infer_direction(point1, point2)
        tool_call["arguments"] = {
            "action": "swipe",
            "direction": direction,
            "start_coordinate": start_abs,
            "end_coordinate": end_abs,
        }
        extras["start_coordinate"] = start_abs
        extras["end_coordinate"] = end_abs
        extras["direction"] = direction
        return (
            json_action.JSONAction(action_type=json_action.SWIPE, direction=direction),
            tool_call,
            extras,
        )

    if action_type == "AWAKE":
        app_name = _normalize_space(parsed_action.get("value"))
        if not app_name:
            raise seeact_utils.ParseActionError("AWAKE requires value")
        tool_call["arguments"] = {"action": "open_app", "text": app_name}
        return (
            json_action.JSONAction(action_type=json_action.OPEN_APP, app_name=app_name),
            tool_call,
            extras,
        )

    if action_type == "WAIT":
        wait_value = max(1, _safe_int(parsed_action.get("value")) or 1)
        extras["wait_seconds"] = wait_value
        tool_call["arguments"] = {"action": "wait", "value": wait_value}
        return json_action.JSONAction(action_type=json_action.WAIT), tool_call, extras

    if action_type == "COMPLETE":
        result_text = _normalize_space(parsed_action.get("return") or parsed_action.get("value"))
        if result_text:
            extras["return_text"] = result_text
        tool_call["arguments"] = {"action": "terminate", "status": "success"}
        return (
            json_action.JSONAction(action_type=json_action.STATUS, goal_status="task_complete"),
            tool_call,
            extras,
        )

    if action_type == "ABORT":
        reason = _normalize_space(parsed_action.get("value"))
        if reason:
            extras["abort_reason"] = reason
        tool_call["arguments"] = {"action": "terminate", "status": "fail"}
        return (
            json_action.JSONAction(action_type=json_action.STATUS, goal_status="infeasible"),
            tool_call,
            extras,
        )

    if action_type == "INFO":
        question = _normalize_space(parsed_action.get("value"))
        if not question:
            raise seeact_utils.ParseActionError("INFO requires value")
        tool_call["arguments"] = {"action": "answer", "text": question}
        return (
            json_action.JSONAction(action_type=json_action.ANSWER, text=question),
            tool_call,
            extras,
        )

    raise seeact_utils.ParseActionError(f"unsupported GELAB action: {action_type}")


def _element_hints(
    ui_elements: list[Any],
    screen_size: tuple[int, int],
    limit: int = 8,
) -> list[str]:
    hints: list[str] = []
    width, height = max(1, int(screen_size[0])), max(1, int(screen_size[1]))
    for idx, element in enumerate(ui_elements):
        if len(hints) >= max(1, int(limit)):
            break
        clickable = bool(getattr(element, "is_clickable", False))
        editable = bool(getattr(element, "is_editable", False))
        scrollable = bool(getattr(element, "is_scrollable", False))
        long_clickable = bool(getattr(element, "is_long_clickable", False))
        if not any([clickable, editable, scrollable, long_clickable]):
            continue
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            continue
        text = _normalize_space(getattr(element, "text", ""))
        desc = _normalize_space(getattr(element, "content_description", ""))
        rid = _normalize_space(getattr(element, "resource_id", "") or getattr(element, "resource_name", ""))
        center_x = int(round((((bbox.x_min + bbox.x_max) / 2.0) / width) * 1000))
        center_y = int(round((((bbox.y_min + bbox.y_max) / 2.0) / height) * 1000))
        flags = []
        if clickable:
            flags.append("click")
        if editable:
            flags.append("edit")
        if scrollable:
            flags.append("scroll")
        if long_clickable:
            flags.append("long")
        label = text or desc or rid or f"element_{idx}"
        hints.append(
            f"element_id={idx}, label='{label}', center=[{center_x},{center_y}], flags={flags}"
        )
    return hints


def build_gelab_messages(
    goal: str,
    history: str,
    screenshot: Image.Image,
    hints: list[str],
) -> list[dict[str, Any]]:
    hint_text = "\n".join(f"- {hint}" for hint in hints) if hints else "None."
    user_text = (
        f"User task:\n{goal}\n\n"
        f"Action history:\n{history or 'None yet.'}\n\n"
        f"Helpful UI hints:\n{hint_text}\n\n"
        "Current screenshot is attached below.\n"
        "Choose the next single action."
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": GELAB_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": _image_to_data_url(screenshot)}},
            ],
        },
    ]


class GELABAgent(base_agent.EnvironmentInteractingAgent):
    """GELAB-style AndroidWorld agent."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        vllm: Any,
        name: str = "GELABAgent",
        output_path: str = "",
        history_limit: int = 8,
    ):
        super().__init__(env, name)
        self.vllm = vllm
        self.output_path = str(output_path or "").strip()
        self.history_limit = max(1, int(history_limit))
        self._actions: list[dict[str, Any]] = []
        self._summaries: list[str] = []
        self._responses: list[str] = []
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self.env.hide_automation_ui()
        self._actions.clear()
        self._summaries.clear()
        self._responses.clear()

    def set_max_steps(self, max_steps: int) -> None:
        super().set_max_steps(min(MAX_AGENT_STEPS, int(max_steps)))

    def _effective_max_steps(self) -> int:
        if self._max_steps is None:
            return MAX_AGENT_STEPS
        return min(MAX_AGENT_STEPS, int(self._max_steps))

    def _task_output_dir(self, goal: str) -> str:
        if not self.output_path:
            return ""
        task_dir = os.path.join(self.output_path, _safe_task_name(goal))
        os.makedirs(task_dir, exist_ok=True)
        return task_dir

    def _history_text(self) -> str:
        if not self._summaries:
            return "None yet."
        start = max(0, len(self._summaries) - self.history_limit)
        lines = []
        for idx, summary in enumerate(self._summaries[start:], start=start + 1):
            lines.append(f"{idx}. {summary}")
        return "\n".join(lines)

    def _write_action_log(self, goal: str) -> None:
        task_dir = self._task_output_dir(goal)
        if not task_dir:
            return
        out_path = os.path.join(task_dir, "action.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in self._actions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _execute_action(self, action: json_action.JSONAction, extras: dict[str, Any]) -> None:
        if action.action_type == json_action.WAIT:
            time.sleep(float(max(1, int(extras.get("wait_seconds", 1)))))
            return
        if action.action_type == json_action.SWIPE and extras.get("start_coordinate") and extras.get("end_coordinate"):
            start_xy = extras["start_coordinate"]
            end_xy = extras["end_coordinate"]
            command = adb_utils.generate_swipe_command(
                int(start_xy[0]),
                int(start_xy[1]),
                int(end_xy[0]),
                int(end_xy[1]),
                500,
            )
            adb_utils.issue_generic_request(command, self.env.controller)
            return
        self.env.execute_action(action)

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        start_time = time.time()
        step_idx = len(self._actions)
        if step_idx >= self._effective_max_steps():
            summary = f"Reached the maximum step limit ({self._effective_max_steps()})."
            action = json_action.JSONAction(action_type=json_action.STATUS, goal_status="infeasible")
            tool_call = {"name": "mobile_use", "arguments": {"action": "terminate", "status": "fail"}}
            print("=" * 96)
            print(f"Step {step_idx}: Result")
            print(summary)
            print("=" * 96)
            return base_agent.AgentInteractionResult(
                done=True,
                data={
                    "response": "",
                    "parsed_action": {
                        "action": "ABORT",
                        "summary": summary,
                        "value": summary,
                    },
                    "tool_call": tool_call,
                    "action": repr(action),
                    "action_dict": action.__dict__,
                    "summary": summary,
                    "hints": [],
                    "latency_sec": float(max(0.0, time.time() - start_time)),
                },
            )
        print("=" * 96)
        print(f"Step {step_idx}: Goal")
        print(goal)
        state = self.get_post_transition_state()
        screenshot = Image.fromarray(state.pixels)
        screen_size = self.env.logical_screen_size
        hints = _element_hints(state.ui_elements, screen_size)
        history = self._history_text()
        messages = build_gelab_messages(goal, history, screenshot, hints)
        message_text = _messages_text_for_logging(messages)
        if message_text:
            _print_step_section(step_idx, "Model input", message_text)

        response, _, _ = self.vllm.predict_mm("", [], messages=messages)
        _print_step_section(step_idx, "Model output", str(response))
        parse_error = None
        try:
            parsed_action = parse_gelab_response(response)
            action, tool_call, extras = gelab_action_to_json_action(parsed_action, screen_size)
        except seeact_utils.ParseActionError as error:
            parse_error = str(error)
            parsed_action = OrderedDict(
                cot="",
                action="WAIT",
                value="1",
                summary="Parser fallback wait",
                parse_error=parse_error,
            )
            action = json_action.JSONAction(action_type=json_action.WAIT)
            tool_call = {"name": "mobile_use", "arguments": {"action": "wait", "value": 1}}
            extras = {"wait_seconds": 1, "parse_error": parse_error, "fallback": "parse_error_wait"}

        if parse_error:
            _print_step_section(step_idx, "Parse fallback", parse_error)
        _print_step_section(step_idx, "Parsed action", _json_dumps_safe(dict(parsed_action)))
        _print_step_section(step_idx, "Tool call", _json_dumps_safe(tool_call))

        if extras.get("return_text"):
            self.env.interaction_cache = str(extras["return_text"])

        self._execute_action(action, extras)
        _print_step_section(step_idx, "Action", _json_dumps_safe(action.__dict__))
        if extras:
            _print_step_section(step_idx, "Action extras", _json_dumps_safe(extras))

        summary = _normalize_space(parsed_action.get("summary")) or _normalize_space(parsed_action.get("explain"))
        if not summary:
            summary = str(tool_call.get("arguments") or tool_call)

        latency_sec = float(max(0.0, time.time() - start_time))
        step_record = {
            "goal": goal,
            "response": response,
            "parsed_action": dict(parsed_action),
            "tool_call": tool_call,
            "action_dict": action.__dict__,
            "summary": summary,
            "latency_sec": latency_sec,
        }
        self._actions.append(step_record)
        self._summaries.append(summary)
        self._responses.append(str(response))

        task_dir = self._task_output_dir(goal)
        if task_dir:
            screenshot.save(os.path.join(task_dir, f"screenshot_{len(self._actions) - 1}.png"))
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
                "hints": hints,
                "latency_sec": latency_sec,
            },
        )
