# Copyright 2025 The android_world Authors.
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

"""MAI-UI agent adapted for AndroidWorld."""

from __future__ import annotations

import base64
import copy
import json
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from PIL import Image

from android_world.agents import base_agent
from android_world.agents import json_action
from android_world.agents import mobile_agent_utils
from android_world.agents import seeact_utils
from android_world.agents.coordinate_resize import convert_point_format
from android_world.agents.coordinate_resize import update_image_size_
from android_world.env import actuation
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import tools


SCALE_FACTOR = 999
MAX_AGENT_STEPS = 20

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


MAI_MOBILE_SYS_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<thinking>
...
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": <args-json-object>}
</tool_call>
```

## Action Space

{"action": "click", "coordinate": [x, y]}
{"action": "long_press", "coordinate": [x, y]}
{"action": "type", "text": ""}
{"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]} # "coordinate" is optional. Use the "coordinate" if you want to swipe a specific UI element.
{"action": "open", "text": "app_name"}
{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}
{"action": "system_button", "button": "button_name"} # Options: back, home, menu, enter
{"action": "wait"}
{"action": "terminate", "status": "success or fail"}
{"action": "answer", "text": "xxx"} # Use escape characters \\\', \\", and \\n in text part to ensure we can parse the text in normal python string format.

## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part.
- Available Apps: `""" + json.dumps(AVAILABLE_APPS, ensure_ascii=True) + """`.
You should use the `open` action to open the app as possible as you can, because it is the fast way to open the app.
- You must follow the Action Space strictly, and return the correct json object within <thinking> </thinking> and <tool_call></tool_call> XML tags.
""".strip()


@dataclass
class TrajStep:
    """Single MAI trajectory step."""

    step_index: int
    screenshot: Image.Image
    prediction: str
    thought: str
    tool_call: dict[str, Any]
    action_dict: dict[str, Any]
    summary: str


@dataclass
class TrajMemory:
    """Simple trajectory memory."""

    task_goal: str = ""
    task_id: str = ""
    steps: list[TrajStep] = field(default_factory=list)


def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def mask_image_urls_for_logging(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages_masked = copy.deepcopy(messages)
    for message in messages_masked:
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "image_url" in item:
                    item["image_url"]["url"] = "[IMAGE_DATA]"
    return messages_masked


def _extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise seeact_utils.ParseActionError("Cannot find tool_call JSON in response.")
    return text[start : end + 1]


def _normalize_downsample_scale(scale: int | float | str) -> int:
    try:
        value = float(scale)
    except Exception:  # pylint: disable=broad-exception-caught
        value = 1.0
    return max(1, int(round(value)))


def _parse_coord_like(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return [int(round(float(value[0]))), int(round(float(value[1])))]
        except Exception:  # pylint: disable=broad-exception-caught
            return None
    if isinstance(value, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(nums) >= 2:
            return [int(round(float(nums[0]))), int(round(float(nums[1])))]
    return None


def _extract_inline_coord(text: str) -> list[int] | None:
    patterns = [
        r'["\']?coordinate["\']?\s*(?:[:=：]|[\(\[\{])\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)',
        r'["\']?coordinates["\']?\s*(?:[:=：]|[\(\[\{])\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)',
        r'["\']?point["\']?\s*(?:[:=：]|[\(\[\{])\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)',
        r"\bpoint\s*\(\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*\)",
        r"\[\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*\]",
    ]
    for pattern in patterns:
        m = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
        if m:
            return [int(round(float(m.group(1)))), int(round(float(m.group(2))))]
    return None


def _infer_action_from_fields(args: dict[str, Any], raw_text: str = "") -> str:
    if not isinstance(args, dict):
        return ""
    answer_keys = ("text", "content", "value", "return", "answer", "respond", "response", "reply", "final_answer", "read")
    if any(args.get(key) for key in ("answer", "respond", "response", "reply", "final_answer", "read")):
        return "answer"
    if any(args.get(key) for key in ("status", "goal_status")):
        return "terminate"
    if any(args.get(key) is not None for key in ("point", "coordinate", "coordinates")):
        return "click"
    if any(args.get(key) is not None for key in ("point1", "point2", "start_coordinate", "end_coordinate", "direction")):
        return "swipe"
    if any(args.get(key) for key in ("app", "app_name")):
        return "open"
    if any(args.get(key) for key in answer_keys) and re.search(r"(?i)\b(answer|respond|response|reply|final_answer|read)\b", str(raw_text or "")):
        return "answer"
    if re.search(r"(?i)\b(status|goal_status)\b", str(raw_text or "")):
        return "terminate"
    return ""


def _clean_action_name(action_value: Any) -> str:
    action_text = str(action_value or "").strip().strip('"').strip("'")
    if not action_text:
        return ""
    action_text = action_text.replace("\\t", "\t")
    if action_text.lower().startswith("action:"):
        action_text = action_text.split(":", 1)[1].strip()
    action_text = action_text.split("\t", 1)[0].split("\n", 1)[0].strip()
    if " " in action_text:
        action_text = action_text.split(" ", 1)[0].strip()
    return action_text


def _normalize_tool_call_obj(obj: dict[str, Any], raw_text: str = "") -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise seeact_utils.ParseActionError("tool_call is not a dict")

    name = str(obj.get("name") or "mobile_use").strip() or "mobile_use"
    args_raw = obj.get("arguments")
    if isinstance(args_raw, str):
        try:
            args_raw = json.loads(args_raw)
        except Exception:  # pylint: disable=broad-exception-caught
            args_raw = {}

    if isinstance(args_raw, dict):
        args = copy.deepcopy(args_raw)
    else:
        args = {k: v for k, v in obj.items() if k != "name"}

    if "action" not in args and "action_type" in args:
        args["action"] = args.get("action_type")
    if "action" not in args and isinstance(args.get("name"), str):
        args["action"] = args.get("name")
    if "action" not in args and name.lower() != "mobile_use":
        args["action"] = name

    raw_action = args.get("action")
    action_name = _clean_action_name(raw_action)
    if not action_name:
        action_name = _infer_action_from_fields(args, raw_text=raw_text) or "wait"
    action_low = action_name.lower()

    if action_low == "tap":
        action_name = "click"
        action_low = "click"
    elif action_low == "longpress":
        action_name = "long_press"
        action_low = "long_press"
    elif action_low in ("input_text", "input", "text"):
        action_name = "type"
        action_low = "type"
    elif action_low in ("slide", "scroll"):
        action_name = "swipe"
        action_low = "swipe"
    elif action_low in ("open_app", "awake"):
        action_name = "open"
        action_low = "open"
    elif action_low in ("respond", "response", "reply", "read", "info"):
        action_name = "answer"
        action_low = "answer"
    elif action_low in ("final_answer", "call_user", "calluser"):
        action_name = "answer"
        action_low = "answer"
    elif action_low in ("status", "complete", "abort"):
        action_name = "terminate"
        action_low = "terminate"
    elif action_low in ("hot_key", "hotkey"):
        action_name = "system_button"
        action_low = "system_button"
    elif action_low in ("double_tap", "double_click", "doubleclick"):
        action_name = "click"
        action_low = "click"

    args["action"] = action_name

    coordinate = None
    for key in ("coordinate", "point", "coordinates"):
        coordinate = _parse_coord_like(args.get(key))
        if coordinate is not None:
            break
    if coordinate is None:
        coordinate = _extract_inline_coord(str(raw_action or "")) or _extract_inline_coord(raw_text)

    if action_low in ("click", "long_press", "type") and coordinate is not None:
        args["coordinate"] = coordinate
    if action_low == "type":
        if args.get("text") is None:
            args["text"] = str(args.get("value") or args.get("content") or args.get("return") or "")
    if action_low == "swipe":
        if args.get("direction") is None:
            direction_match = re.search(
                r'"direction"\s*:\s*"([^"]+)"|\bdirection\s*:\s*([a-zA-Z_]+)',
                raw_text,
                flags=re.IGNORECASE,
            )
            if direction_match:
                args["direction"] = (direction_match.group(1) or direction_match.group(2) or "").strip()
        if coordinate is not None and "coordinate" not in args:
            args["coordinate"] = coordinate
        if args.get("direction") is None:
            if _parse_coord_like(args.get("point1")) and _parse_coord_like(args.get("point2")):
                args["point1"] = _parse_coord_like(args.get("point1"))
                args["point2"] = _parse_coord_like(args.get("point2"))
            elif _parse_coord_like(args.get("start_coordinate")) and _parse_coord_like(args.get("end_coordinate")):
                args["start_coordinate"] = _parse_coord_like(args.get("start_coordinate"))
                args["end_coordinate"] = _parse_coord_like(args.get("end_coordinate"))
    if action_low == "drag":
        start_coordinate = _parse_coord_like(args.get("start_coordinate")) or _parse_coord_like(args.get("point1"))
        end_coordinate = _parse_coord_like(args.get("end_coordinate")) or _parse_coord_like(args.get("point2"))
        if start_coordinate is not None:
            args["start_coordinate"] = start_coordinate
        if end_coordinate is not None:
            args["end_coordinate"] = end_coordinate
    if action_low in ("answer", "terminate"):
        if args.get("text") is None:
            args["text"] = str(
                args.get("content")
                or args.get("value")
                or args.get("return")
                or args.get("answer")
                or args.get("respond")
                or args.get("response")
                or args.get("reply")
                or args.get("final_answer")
                or args.get("read")
                or ""
            )
    if action_low == "terminate":
        status = str(args.get("status") or args.get("goal_status") or "").strip().lower()
        if not status:
            status = "success" if str(args.get("text") or "").strip() else "fail"
        if status in ("completed", "complete", "done"):
            status = "success"
        if status in ("failed", "failure", "infeasible", "error", "task_failed"):
            status = "fail"
        if status in ("task_complete", "ok"):
            status = "success"
        args["status"] = status
    if action_low in ("back", "home", "menu", "enter"):
        args["action"] = "system_button"
        args["button"] = action_low
    if action_low == "system_button":
        button = str(args.get("button") or args.get("key") or args.get("value") or "").strip().lower()
        if button in ("", "none"):
            button = "back"
        if button == "return":
            button = "back"
        args["button"] = button
    if action_low == "open":
        app_name = _normalize_space(
            args.get("text")
            or args.get("app_name")
            or args.get("app")
            or args.get("value")
            or ""
        )
        if not app_name:
            raw_action_text = _normalize_space(str(raw_action or ""))
            m = re.search(
                r"(?i)^(?:open_app|open|awake)\s*[:=]?\s*([A-Za-z][A-Za-z0-9 ._\\-]{1,48})$",
                raw_action_text,
            )
            if m:
                app_name = _normalize_space(m.group(1))
        if not app_name:
            m = re.search(r'"app_name"\s*:\s*"([^"]+)"', str(raw_text or ""), flags=re.IGNORECASE)
            if m:
                app_name = _normalize_space(m.group(1))
        if app_name:
            args["text"] = app_name
            args["app_name"] = app_name

    return {"name": "mobile_use", "arguments": args}


def _parse_legacy_kv_tool_call(text: str) -> dict[str, Any] | None:
    raw = str(text or "")
    if not raw.strip():
        return None
    no_thinking = re.sub(
        r"<\s*think(?:ing)?\s*>.*?<\s*/\s*think(?:ing)?\s*>",
        " ",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    segments = [seg.strip() for seg in re.split(r"[\t\r\n]+", no_thinking) if seg.strip()]
    kv: dict[str, str] = {}
    for seg in segments:
        if ":" not in seg:
            continue
        key, value = seg.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in ("explain", "summary", "reason"):
            continue
        kv[key] = value

    action_value = kv.get("action") or kv.get("action_type")
    if not action_value:
        action_match = re.search(r"\baction\s*:\s*([^\t\r\n]+)", no_thinking, flags=re.IGNORECASE)
        if action_match:
            action_value = action_match.group(1).strip()
    if not action_value:
        for key in ("answer", "respond", "response", "reply", "final_answer", "read"):
            if str(kv.get(key) or "").strip():
                action_value = "answer"
                break
    if not action_value and any(k in kv for k in ("status", "goal_status")):
        action_value = "terminate"
    if not action_value:
        return None

    args: dict[str, Any] = {"action": action_value}
    point_value = kv.get("point") or kv.get("coordinate") or kv.get("coordinates")
    if point_value:
        point = _parse_coord_like(point_value)
        if point is not None:
            args["coordinate"] = point
    if "coordinate" not in args:
        inline_coord = _extract_inline_coord(no_thinking)
        if inline_coord is not None:
            args["coordinate"] = inline_coord
    if "point1" in kv:
        point1 = _parse_coord_like(kv.get("point1"))
        if point1 is not None:
            args["point1"] = point1
    if "point2" in kv:
        point2 = _parse_coord_like(kv.get("point2"))
        if point2 is not None:
            args["point2"] = point2

    if "value" in kv:
        args["value"] = kv["value"]
    if "text" in kv:
        args["text"] = kv["text"]
    if "content" in kv:
        args["content"] = kv["content"]
    if "return" in kv:
        args["return"] = kv["return"]
    if "answer" in kv:
        args["answer"] = kv["answer"]
    if "respond" in kv:
        args["respond"] = kv["respond"]
    if "response" in kv:
        args["response"] = kv["response"]
    if "reply" in kv:
        args["reply"] = kv["reply"]
    if "final_answer" in kv:
        args["final_answer"] = kv["final_answer"]
    if "read" in kv:
        args["read"] = kv["read"]
    if "status" in kv:
        args["status"] = kv["status"]
    if "goal_status" in kv:
        args["goal_status"] = kv["goal_status"]
    if "key" in kv:
        args["key"] = kv["key"]
    if "button" in kv:
        args["button"] = kv["button"]
    if "direction" in kv:
        args["direction"] = kv["direction"]

    return _normalize_tool_call_obj({"name": "mobile_use", "arguments": args}, raw)


def safe_json_loads(s: str) -> dict[str, Any]:
    """Robust JSON parsing for model tool-call output."""
    s = str(s or "").strip()

    try:
        return _normalize_tool_call_obj(json.loads(s), s)
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()

    try:
        return _normalize_tool_call_obj(json.loads(s), s)
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    s = re.sub(r"\}\s*\}\s*\]", "]]", s)
    s = re.sub(r"\]\s*\}+", "]", s)

    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = s[first:last + 1]
        try:
            return _normalize_tool_call_obj(json.loads(candidate), candidate)
        except Exception:  # pylint: disable=broad-exception-caught
            s = candidate

    legacy_tool_call = _parse_legacy_kv_tool_call(s)
    if legacy_tool_call is not None:
        return legacy_tool_call

    def extract(pattern: str, default: str | None = None, flags: int = 0) -> str | None:
        m = re.search(pattern, s)
        if m is None and flags:
            m = re.search(pattern, s, flags=flags)
        return m.group(1) if m else default

    def extract_coord(key: str | None = None) -> list[int]:
        if key is not None:
            m = re.search(rf'"{key}"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)[^\]]*\]', s)
            if m:
                return [int(round(float(m.group(1)))), int(round(float(m.group(2))))]
            m = re.search(rf'"{key}"\s*:\s*"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"', s)
            if m:
                return [int(round(float(m.group(1)))), int(round(float(m.group(2))))]
        m = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", s)
        if m:
            return [int(round(float(m.group(1)))), int(round(float(m.group(2))))]
        inline = _extract_inline_coord(s)
        if inline is not None:
            return inline
        return [-1, -1]

    name = extract(r'"name"\s*:\s*"([^"]+)"', "mobile_use")
    action = extract(r'"action"\s*:\s*"([^"]+)"')
    if action is None:
        action = extract(r'"action_type"\s*:\s*"([^"]+)"')
    if action is None:
        action = extract(r"\baction\s*:\s*([^\t\r\n,}]+)", flags=re.IGNORECASE)
    if action is None:
        if re.search(r"(?i)\b(answer|respond|response|reply|final_answer|read)\s*[:=]", s):
            action = "answer"
        elif re.search(r"(?i)\b(status|goal_status)\s*[:=]", s):
            action = "terminate"
        elif _extract_inline_coord(s) is not None:
            action = "click"
    action_low = str(action or "click").strip().lower()
    text_value = extract(r'"text"\s*:\s*"([^"]*)"') or extract(r"\btext\s*:\s*([^\t\r\n]+)", flags=re.IGNORECASE)
    content = extract(r'"content"\s*:\s*"([^"]*)"')
    value = extract(r'"value"\s*:\s*"([^"]*)"') or extract(r"\bvalue\s*:\s*([^\t\r\n]+)", flags=re.IGNORECASE)
    return_text = extract(r'"return"\s*:\s*"([^"]*)"') or extract(r"\breturn\s*:\s*([^\t\r\n]+)", flags=re.IGNORECASE)
    button = extract(r'"button"\s*:\s*"([^"]+)"') or extract(r"\bbutton\s*:\s*([^\t\r\n]+)", flags=re.IGNORECASE)
    key = extract(r'"key"\s*:\s*"([^"]+)"') or extract(r"\bkey\s*:\s*([^\t\r\n]+)", flags=re.IGNORECASE)
    direction = extract(r'"direction"\s*:\s*"([^"]+)"') or extract(r"\bdirection\s*:\s*([^\t\r\n]+)", flags=re.IGNORECASE)

    args: dict[str, Any] = {"action": action}

    if action_low in ("click", "long_press", "tap"):
        coord = extract_coord("coordinate")
        if coord != [-1, -1]:
            args["coordinate"] = coord
    elif action_low == "type":
        args["text"] = text_value or value or ""
        coord = extract_coord("coordinate")
        if coord != [-1, -1]:
            args["coordinate"] = coord
    elif action_low == "swipe":
        args["direction"] = direction or "down"
        coord = extract_coord("coordinate")
        if coord != [-1, -1]:
            args["coordinate"] = coord
    elif action_low == "drag":
        args["start_coordinate"] = extract_coord("start_coordinate")
        args["end_coordinate"] = extract_coord("end_coordinate")
    elif action_low in ("open", "open_app", "awake"):
        args["action"] = "open"
        app_name = (
            extract(r'"app_name"\s*:\s*"([^"]+)"')
            or extract(r"\bapp_name\s*:\s*([^\t\r\n]+)", flags=re.IGNORECASE)
            or text_value
            or value
        )
        if app_name:
            args["text"] = str(app_name).strip().strip('"').strip("'")
    elif action_low == "system_button":
        args["button"] = button or "back"
    elif action_low == "back":
        args["action"] = "system_button"
        args["button"] = "back"
    elif action_low in ("answer", "respond", "response", "reply", "read", "final_answer", "call_user", "calluser", "info"):
        args["action"] = "answer"
        args["text"] = text_value or content or value or return_text or ""
    elif action_low in ("terminate", "status", "complete", "abort"):
        args["action"] = "terminate"
        status = extract(r'"status"\s*:\s*"([^"]+)"') or extract(r"\bstatus\s*:\s*([^\t\r\n]+)", "fail", flags=re.IGNORECASE)
        args["status"] = status
        if not args.get("text"):
            args["text"] = text_value or content or value or return_text or ""
    elif action_low in ("hot_key", "hotkey", "system_button"):
        args["action"] = "system_button"
        args["button"] = (button or key or value or "back")
    else:
        args["action"] = "wait"

    fixed_obj = _normalize_tool_call_obj({"name": name, "arguments": args}, s)
    return fixed_obj


def parse_tagged_text(text: str) -> dict[str, Any]:
    """Parse <thinking> and <tool_call> tags."""
    text = str(text or "")

    text = text.replace("<TINK>", "<THINK>").replace("</TINK>", "</THINK>")
    text = text.replace("<think>", "<thinking>").replace("</think>", "</thinking>")
    text = text.replace("<THINK>", "<thinking>").replace("</THINK>", "</thinking>")

    result: dict[str, Any] = {"thinking": "", "tool_call": None}

    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, flags=re.DOTALL | re.IGNORECASE)
    if thinking_match:
        result["thinking"] = thinking_match.group(1).strip().strip('"')

    tool_match = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL | re.IGNORECASE)
    if tool_match:
        tool_text = tool_match.group(1).strip().strip('"')
        result["tool_call"] = safe_json_loads(tool_text)
        return result

    try:
        result["tool_call"] = safe_json_loads(_extract_json_block(text))
    except Exception:  # pylint: disable=broad-exception-caught
        result["tool_call"] = _parse_legacy_kv_tool_call(text)

    return result


def parse_mai_tool_call(response_text: str) -> dict[str, Any]:
    parsed = parse_tagged_text(response_text)
    tool_call = parsed.get("tool_call")
    if tool_call is None:
        raise seeact_utils.ParseActionError("Cannot find tool_call JSON in response.")
    return tool_call


def parse_action_to_structure_output(text: str) -> dict[str, Any]:
    """Parse MAI output into structured dict (official style)."""
    parsed = parse_tagged_text(text)
    tool_call = parsed.get("tool_call") or {}
    action = copy.deepcopy(tool_call.get("arguments") or {})

    for key in ("coordinate", "start_coordinate", "end_coordinate"):
        if key not in action:
            continue
        coords = action.get(key) or []
        if not isinstance(coords, list):
            continue
        if len(coords) == 2:
            point_x, point_y = float(coords[0]), float(coords[1])
        elif len(coords) == 4:
            x1, y1, x2, y2 = [float(v) for v in coords]
            point_x = (x1 + x2) / 2
            point_y = (y1 + y2) / 2
        else:
            continue
        action[key] = [point_x / SCALE_FACTOR, point_y / SCALE_FACTOR]

    return {
        "thinking": parsed.get("thinking") or "",
        "action_json": action,
        "tool_call": tool_call,
    }


def fetch_resized_image(screenshot_file: str, scale: int = 1) -> tuple[Image.Image, int, int, dict[str, Any]]:
    screenshot = Image.open(screenshot_file)
    width, height = screenshot.size
    scale_divisor = _normalize_downsample_scale(scale)
    image_ele = update_image_size_(
        {
            "image": screenshot_file,
            "width": max(1, int(round(width / scale_divisor))),
            "height": max(1, int(round(height / scale_divisor))),
        }
    )
    resized_width = int(image_ele["resized_width"])
    resized_height = int(image_ele["resized_height"])
    screenshot = screenshot.resize((resized_width, resized_height))
    return screenshot, resized_width, resized_height, image_ele


def _summarize_action(arguments: dict[str, Any]) -> str:
    action = str(arguments.get("action") or "").strip().lower()
    if action == "click":
        if "coordinate" in arguments:
            return f"click at {arguments.get('coordinate')}"
        if "element_id" in arguments:
            return f"click element_id={arguments.get('element_id')}"
    if action == "long_press":
        if "coordinate" in arguments:
            return f"long_press at {arguments.get('coordinate')}"
        if "element_id" in arguments:
            return f"long_press element_id={arguments.get('element_id')}"
    if action == "type":
        text = str(arguments.get("text") or "")
        if "coordinate" in arguments:
            return f"type \"{text[:24]}\" at {arguments.get('coordinate')}"
        if "element_id" in arguments:
            return f"type \"{text[:24]}\" element_id={arguments.get('element_id')}"
        return f"type \"{text[:24]}\""
    if action == "swipe":
        return f"swipe {arguments.get('direction', '')}".strip()
    if action == "open":
        return f"open \"{arguments.get('text', '')}\""
    if action == "system_button":
        return f"system_button {arguments.get('button', '')}".strip()
    if action == "terminate":
        return f"terminate({arguments.get('status', '')})"
    if action == "answer":
        return f"answer \"{str(arguments.get('text') or '')[:32]}\""
    if action == "wait":
        return "wait"
    return str(arguments)


def _thinking_to_summary(thinking: str) -> str:
    text = re.sub(r"\s+", " ", str(thinking or "")).strip()
    if not text:
        return ""
    chunks = [chunk.strip() for chunk in re.split(r"[。！？.!?]", text) if chunk.strip()]
    if not chunks:
        return text[:96]
    last = chunks[-1]
    if len(last) > 96:
        last = last[:96].rstrip(" ,.;:") + "..."
    return last


def _action_to_dict(action_obj: json_action.JSONAction) -> dict[str, Any]:
    return {k: v for k, v in action_obj.__dict__.items() if v is not None}


def _format_ui_element_list(
    ui_elements: list[Any],
    image_ele: dict[str, Any],
    coordinate_format: str,
    limit: int = 12,
) -> str:
    lines = []
    for idx, element in enumerate(ui_elements or []):
        if len(lines) >= max(1, int(limit)):
            break
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            continue
        clickable = bool(getattr(element, "is_clickable", False))
        editable = bool(getattr(element, "is_editable", False))
        long_clickable = bool(getattr(element, "is_long_clickable", False))
        if not any([clickable, editable, long_clickable]):
            continue

        center_abs = [int((bbox.x_min + bbox.x_max) / 2), int((bbox.y_min + bbox.y_max) / 2)]
        center = convert_point_format(
            center_abs,
            image_ele,
            src_format="abs_origin",
            tgt_format=coordinate_format,
            scale=1,
        )
        text = str(getattr(element, "text", "") or "").strip()
        desc = str(getattr(element, "content_description", "") or "").strip()
        rid = str(getattr(element, "resource_id", "") or getattr(element, "resource_name", "") or "").strip()
        label = text or desc or rid or f"element_{idx}"
        flags = []
        if clickable:
            flags.append("click")
        if editable:
            flags.append("edit")
        if long_clickable:
            flags.append("long")
        lines.append(
            f"- element_id={idx}, label='{label[:80]}', center=[{int(center[0])}, {int(center[1])}], flags={flags}"
        )
    return "\n".join(lines) if lines else "None."


def build_mai_messages(goal: str, history: str, screenshot_path: str, ui_element_text: str = "None.") -> list[dict[str, Any]]:
    _ = ui_element_text
    system_msg = {
        "role": "system",
        "content": [{"type": "text", "text": MAI_MOBILE_SYS_PROMPT}],
    }
    user_text = (
        f"Task:\n{goal}\n\n"
        f"Action History:\n{history if history else 'None yet.'}\n\n"
        "Current screenshot is attached below.\n"
        "Choose the next single action."
    )
    user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_to_data_url(screenshot_path)}},
        ],
    }
    return [system_msg, user_msg]


class MAIUIAgent(base_agent.EnvironmentInteractingAgent):
    """MAI-UI style AndroidWorld agent."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        vllm: Any,
        src_format: str,
        api_key: str | None = None,
        url: str | None = None,
        name: str = "MAIUIAgent",
        output_path: str = "",
        image_downsample_scale: int | float = 1,
        history_limit: int = 8,
    ):
        super().__init__(env, name)
        self.vllm = vllm
        self.src_format = str(src_format or "qwen-vl")
        self.api_key = api_key
        self.url = url
        self.output_path = str(output_path or "").strip()
        self.image_downsample_scale = _normalize_downsample_scale(image_downsample_scale)
        self.history_limit = max(1, int(history_limit))

        self._actions: list[dict[str, Any]] = []
        self._screenshots: list[Image.Image] = []
        self._summarys: list[str] = []
        self._thoughts: list[str | None] = []
        self._response: list[str] = []
        self._text_actions: list[str] = []
        self.output_result: dict[str, Any] = {}
        self.output_list: list[Any] = []
        self.task_name: dict[str, str] = {}

        self.traj_memory = TrajMemory()

        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)

    def set_max_steps(self, max_steps: int) -> None:
        super().set_max_steps(min(MAX_AGENT_STEPS, int(max_steps)))

    def _effective_max_steps(self) -> int:
        if self._max_steps is None:
            return MAX_AGENT_STEPS
        return min(MAX_AGENT_STEPS, int(self._max_steps))

    def set_image_downsample_scale(self, scale: int | float) -> None:
        self.image_downsample_scale = _normalize_downsample_scale(scale)

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self.env.hide_automation_ui()
        self._actions.clear()
        self._screenshots.clear()
        self._summarys.clear()
        self._thoughts.clear()
        self._response.clear()
        self._text_actions.clear()
        self.traj_memory = TrajMemory()

    def get_task_name(self, suite: dict[str, Any]) -> None:
        for name, instances in suite.items():
            self.task_name[instances[0].goal] = name

    def initialize_chrome(self) -> None:
        print("Running additional chrome initialization...")
        adb_utils.launch_app("chrome", self.env.controller)
        time.sleep(5)

        tool_controller = tools.AndroidToolController(env=self.env.controller)
        time.sleep(2)

        first_op = False
        try:
            tool_controller.click_element("Use without an account")
            time.sleep(5.0)
            first_op = True
        except Exception:  # pylint: disable=broad-exception-caught
            print("Failed to click 'Use without an account' button.")

        if not first_op:
            try:
                tool_controller.click_element("Accept & continue")
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            time.sleep(3.0)
            try:
                tool_controller.click_element("No thanks")
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            time.sleep(5.0)

        adb_utils.press_home_button(self.env.controller)
        time.sleep(2.0)
        print("Done additional chrome initialization")

    def _task_output_dir(self, goal: str) -> str:
        if not self.output_path:
            return ""
        if goal in self.task_name:
            task_dir = os.path.join(self.output_path, self.task_name[goal])
        else:
            task_dir = os.path.join(self.output_path, goal.replace(" ", "_")[:50])
        os.makedirs(task_dir, exist_ok=True)
        return task_dir

    def _history_text(self) -> str:
        if not self._summarys:
            return "None yet."
        start = max(0, len(self._summarys) - self.history_limit)
        lines = []
        for idx, summary in enumerate(self._summarys[start:], start=start + 1):
            lines.append(f"{idx}. {summary}")
        return "\n".join(lines)

    def _infer_app_from_goal(self, goal: str) -> str:
        goal_text = str(goal or "").strip().lower()
        if not goal_text:
            return ""
        alias_map = {
            "audio recorder": "Audio Recorder",
            "pro expense": "Pro Expense",
            "simple gallery pro": "Simple Gallery Pro",
            "simple gallery": "Simple Gallery Pro",
            "simple calendar pro": "Simple Calendar Pro",
            "simple calendar": "Simple Calendar Pro",
            "broccoli": "Broccoli APP",
            "joplin": "Joplin",
            "markor": "Markor",
            "tasks": "Tasks",
            "chrome": "Chrome",
            "camera": "Camera",
            "files": "Files",
            "settings": "Settings",
            "contacts": "Contacts",
            "clock": "Clock",
            "dialer": "Dialer",
        }
        for marker, app_name in alias_map.items():
            if marker in goal_text:
                return app_name
        apps_sorted = sorted(AVAILABLE_APPS, key=lambda x: len(str(x)), reverse=True)
        for app_name in apps_sorted:
            app_text = str(app_name or "").strip()
            if app_text and app_text.lower() in goal_text:
                return app_text
        return ""

    def _recent_open_app_name(self) -> str:
        for item in reversed(self._actions):
            if not isinstance(item, dict):
                continue
            args = item.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            action_name = str(args.get("action") or "").strip().lower()
            if action_name not in {"open", "open_app", "awake"}:
                continue
            app_name = str(
                args.get("text")
                or args.get("app_name")
                or args.get("value")
                or ""
            ).strip()
            if app_name:
                return app_name
        return ""

    def _resolve_open_app_name(self, goal: str, tool_call: dict[str, Any]) -> str:
        args = (tool_call or {}).get("arguments") if isinstance(tool_call, dict) else {}
        if not isinstance(args, dict):
            args = {}
        app_name = str(
            args.get("text")
            or args.get("app_name")
            or args.get("value")
            or ""
        ).strip()
        if app_name:
            return app_name
        inferred = self._infer_app_from_goal(goal)
        if inferred:
            return inferred
        return self._recent_open_app_name()

    def _write_action_log(self, task_output_dir: str) -> None:
        if not task_output_dir:
            return
        out_path = os.path.join(task_output_dir, "action.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in self._actions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _record_traj_step(
        self,
        step_index: int,
        screenshot: Image.Image,
        prediction: str,
        thought: str,
        tool_call: dict[str, Any],
        action_dict: dict[str, Any],
        summary: str,
        goal: str,
    ) -> None:
        if not self.traj_memory.task_goal:
            self.traj_memory.task_goal = goal
        self.traj_memory.steps.append(
            TrajStep(
                step_index=step_index,
                screenshot=screenshot,
                prediction=prediction,
                thought=thought,
                tool_call=tool_call,
                action_dict=action_dict,
                summary=summary,
            )
        )

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        result: dict[str, Any] = {
            "ui_elements": None,
            "screenshot": None,
            "action_response": None,
            "dummy_action": None,
            "dummy_action_translated": None,
            "action": None,
            "summary": None,
        }
        start_time = time.time()

        if len(self._actions) >= self._effective_max_steps():
            action = json_action.JSONAction(action_type=json_action.STATUS, goal_status="infeasible")
            summary = f"Reached the maximum step limit ({self._effective_max_steps()})."
            result["action"] = action
            result["summary"] = summary
            print(f"[INFO] {summary}")
            return base_agent.AgentInteractionResult(done=True, data=result)

        step_idx = len(self._actions)
        state = self.get_post_transition_state()
        result["ui_elements"] = state.ui_elements
        result["screenshot"] = state.pixels
        screenshot = Image.fromarray(state.pixels)
        self._screenshots.append(screenshot)

        task_output_dir = self._task_output_dir(goal)
        screenshot_file = os.path.join(task_output_dir, f"screenshot_{step_idx}.png") if task_output_dir else f"screenshot_{step_idx}.png"
        screenshot.save(screenshot_file)

        scale = self.image_downsample_scale
        resized_image, resized_width, resized_height, image_ele = fetch_resized_image(screenshot_file, scale)
        resized_screenshot_file = re.sub(r"screenshot_(\d+)\.png$", r"screenshot_resized_\1.png", screenshot_file)
        if resized_screenshot_file == screenshot_file:
            resized_screenshot_file = screenshot_file.replace(".png", f"_resized_scale{scale}.png")
        resized_image.save(resized_screenshot_file)
        with Image.open(resized_screenshot_file) as resized_file_img:
            resized_file_size = resized_file_img.size

        history_text = self._history_text()
        messages = build_mai_messages(goal, history_text, resized_screenshot_file)

        action_response, _, _ = self.vllm.predict_mm("", [], messages=messages)
        result["action_response"] = action_response
        print("========== MAI action_response ==========")
        print(action_response)

        summary = "wait"
        thought = ""
        parse_error = None

        try:
            parsed = parse_tagged_text(action_response)
            thought = parsed.get("thinking") or ""
            tool_call = parsed.get("tool_call") or parse_mai_tool_call(action_response)
            dummy_action = tool_call
            print("========== MAI parsed_tool_call ==========")
            print(json.dumps(tool_call, ensure_ascii=False, indent=2))

            action_args = (dummy_action or {}).get("arguments", {})
            if isinstance(action_args, dict):
                action_name = str(action_args.get("action") or "").strip().lower()
                if action_name in {"open", "open_app", "awake"}:
                    resolved_app = self._resolve_open_app_name(goal, dummy_action)
                    if resolved_app:
                        action_args["action"] = "open"
                        action_args["text"] = resolved_app
                        action_args["app_name"] = resolved_app
                        print(f"[CHECK] MAI open_app resolved to: {resolved_app}")
                    else:
                        print("[CHECK] MAI open_app unresolved: missing app name")

            action, dummy_action_translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
                dummy_action,
                image_ele,
                src_format=self.src_format,
                tgt_format="abs_origin",
                scale=scale,
                ui_elements=state.ui_elements,
            )
            print("========== MAI normalized_action ==========")
            print(repr(action))
            print("========== MAI translated_tool_call ==========")
            print(json.dumps(dummy_action_translated, ensure_ascii=False, indent=2))

            if action.action_type == json_action.ANSWER and action.text:
                self.env.interaction_cache = action.text
            else:
                action_args = (dummy_action or {}).get("arguments", {})
                if isinstance(action_args, dict):
                    act = str(action_args.get("action") or "").strip().lower()
                    if act in ("terminate", "status", "complete"):
                        terminate_text = str(
                            action_args.get("text")
                            or action_args.get("value")
                            or action_args.get("return")
                            or ""
                        ).strip()
                        if terminate_text:
                            self.env.interaction_cache = terminate_text

            actuation.execute_adb_action(
                action,
                state.ui_elements,
                self.env.logical_screen_size,
                self.env.controller,
            )

            result["dummy_action"] = dummy_action
            result["dummy_action_translated"] = dummy_action_translated
            result["action"] = action
            action_summary = _summarize_action(dummy_action.get("arguments", {}))
            thinking_summary = _thinking_to_summary(thought)
            summary = thinking_summary or action_summary
            if len(summary) > 140:
                summary = summary[:140].rstrip(" ,.;:") + "..."

        except (
            seeact_utils.ParseActionError,
            ValueError,
            KeyError,
            NotImplementedError,
            json.JSONDecodeError,
        ) as exc:
            print("Failed to parse/normalize MAI tool_call:", exc)
            parse_error = str(exc)
            dummy_action = {"name": "mobile_use", "arguments": {"action": "wait"}}
            action = json_action.JSONAction(action_type=json_action.WAIT)

            actuation.execute_adb_action(
                action,
                state.ui_elements,
                self.env.logical_screen_size,
                self.env.controller,
            )

            result["dummy_action"] = dummy_action
            result["dummy_action_translated"] = dummy_action
            result["action"] = action
            summary = "Parser fallback wait"

        except Exception:  # pylint: disable=broad-exception-caught
            traceback.print_exc()
            print(action_response)
            raise

        print("========== MAI final_action ==========")
        print(repr(result["action"]))

        self._text_actions.append(summary)
        self._summarys.append(summary)
        self._thoughts.append(thought)
        self._response.append(str(action_response))
        self._actions.append(
            {
                "name": "mobile_use",
                "arguments": (result["dummy_action"] or {}).get("arguments", {}),
                "summary": summary,
                "parse_error": parse_error,
            }
        )

        action_obj = result["action"] if isinstance(result["action"], json_action.JSONAction) else json_action.JSONAction(action_type=json_action.WAIT)
        self._record_traj_step(
            step_index=step_idx,
            screenshot=screenshot,
            prediction=str(action_response),
            thought=thought,
            tool_call=result["dummy_action"] or {},
            action_dict=_action_to_dict(action_obj),
            summary=summary,
            goal=goal,
        )

        self._write_action_log(task_output_dir)

        latency = time.time() - start_time
        result["summary"] = summary
        result["latency_sec"] = float(max(0.0, latency))
        print(f"[INFO] Step Latency: {latency:.2f} seconds")

        done = bool(
            isinstance(action_obj, json_action.JSONAction)
            and action_obj.action_type in {json_action.STATUS, json_action.ANSWER}
        )
        return base_agent.AgentInteractionResult(done=done, data=result)
