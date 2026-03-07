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


GELAB_SYSTEM_PROMPT = """
你是一个手机 GUI-Agent 操作专家。你会收到：任务目标、历史动作、当前截图。
请输出“下一步唯一动作”，坐标使用 0-1000 归一化空间（左上角原点，x 向右，y 向下）。

动作空间（GELAB）：
1. CLICK：action:CLICK\tpoint:x,y
2. TYPE：action:TYPE\tvalue:输入文本\tpoint:x,y
3. COMPLETE：action:COMPLETE\treturn:最终回复
4. AWAKE：action:AWAKE\tvalue:应用名
5. INFO：action:INFO\tvalue:提问内容
6. ABORT：action:ABORT\tvalue:原因
7. SLIDE：action:SLIDE\tpoint1:x1,y1\tpoint2:x2,y2
8. LONGPRESS：action:LONGPRESS\tpoint:x,y
9. ANSWER：action:ANSWER\tvalue:最终答案（问答类任务推荐）

首选输出格式（推荐）：
<THINK>简短思考</THINK>
explain:本步目的\taction:动作名\t...参数...\tsummary:本步后简短进展

官方 action_tool 格式（强兼容，优先推荐）：
{"action_type":"CLICK","point":[x,y]}
{"action_type":"TYPE","value":"文本","point":[x,y]}
{"action_type":"HOT_KEY","key":"ENTER|BACK|HOME"}
{"action_type":"SLIDE","point1":[x1,y1],"point2":[x2,y2]}
{"action_type":"ANSWER","value":"最终答案"}
{"action_type":"COMPLETE","status":"SUCCESS|FAILURE","value":"可选结果"}

兼容格式（可选）：
<tool_call>
{"name":"mobile_use","arguments":{"action":"click","coordinate":[x,y]}}
</tool_call>

强约束：
- 只输出一个动作，不要输出多个候选。
- 不要把 point/value/summary 拼进 action 字段。
- action 字段只能是纯动作名（如 CLICK 或 click）。
- 若使用 tool_call JSON，坐标必须放在 coordinate 数组，不要写成 "action":"CLICK\\tpoint:..."
- 优先推动任务完成，避免无效重复动作。
- 不要输出 Wait 这个action
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
        if isinstance(value, str):
            nums = re.findall(r"-?\d+(?:\.\d+)?", value)
            if len(nums) >= 2:
                try:
                    return [int(round(float(nums[0]))), int(round(float(nums[1])))]
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
    return None


def _extract_direction(value: Any) -> str:
    text = _normalize_space(value).lower()
    for direction in ("up", "down", "left", "right"):
        if re.search(rf"\b{direction}\b", text):
            return direction
    return ""


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
        direction = _extract_direction(args.get("direction"))
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
        "respond",
        "answer",
        "response",
        "reply",
        "final_answer",
        "text",
        "content",
        "app_name",
        "return",
        "status",
        "key",
        "point",
        "coordinate",
        "coordinates",
        "point1",
        "point2",
        "start_coordinate",
        "end_coordinate",
        "direction",
        "button",
    ]
    pattern = re.compile(
        r'(^|[\t\n])\s*["\']?(%s)["\']?\s*(?:[:=：]|[\(\[\{])'
        % "|".join(re.escape(field) for field in fields),
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


_ACTION_ALIASES = {
    "TAP": "CLICK",
    "PRESS": "CLICK",
    "LONG_CLICK": "LONGPRESS",
    "LONGCLICK": "LONGPRESS",
    "LONG_PRESS": "LONGPRESS",
    "LONGPRESS": "LONGPRESS",
    "DOUBLE_TAP": "DOUBLECLICK",
    "DOUBLE_CLICK": "DOUBLECLICK",
    "DOUBLECLICK": "DOUBLECLICK",
    "HOT_KEY": "HOTKEY",
    "HOTKEY": "HOTKEY",
    "CALL_USER": "CALLUSER",
    "CALLUSER": "CALLUSER",
    "SWIPE": "SLIDE",
    "OPEN_APP": "AWAKE",
    "OPEN": "AWAKE",
    "ANSWER": "ANSWER",
    "RESPOND": "ANSWER",
    "RESPONSE": "ANSWER",
    "REPLY": "ANSWER",
    "FINAL_ANSWER": "ANSWER",
    "READ": "ANSWER",
    "TERMINATE": "COMPLETE",
    "STATUS": "COMPLETE",
}


def _canonical_action_name(raw_action: Any) -> str:
    text = _normalize_space(raw_action).upper().replace("-", "_")
    if not text:
        raise seeact_utils.ParseActionError("missing GELAB action field")
    match = re.search(r"[A-Z_]+", text)
    if not match:
        raise seeact_utils.ParseActionError(f"invalid GELAB action field: {raw_action}")
    token = match.group(0)
    return _ACTION_ALIASES.get(token, token)


def _extract_norm_point_from_text(text: str, keys: tuple[str, ...]) -> list[int] | None:
    for key in keys:
        match = re.search(
            rf'(?i)\b["\']?{re.escape(key)}["\']?\s*(?:[:=：]|[\(\[\{{])\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)',
            text,
        )
        if match:
            return _clip_norm_point([int(round(float(match.group(1)))), int(round(float(match.group(2))))])
    return None


def _extract_unlabeled_points_from_text(text: str, max_points: int = 3) -> list[list[int]]:
    points: list[list[int]] = []
    for match in re.finditer(
        r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*[\]\)]",
        str(text or ""),
    ):
        points.append(
            _clip_norm_point([int(round(float(match.group(1)))), int(round(float(match.group(2))))])
        )
        if len(points) >= max(1, int(max_points)):
            break
    return points


def _extract_text_field_from_text(text: str, keys: tuple[str, ...]) -> str:
    for key in keys:
        match = re.search(
            rf'(?is)\b["\']?{re.escape(key)}["\']?\s*(?:[:=：]|[\(\[\{{])\s*(.+?)(?:[\t\r\n]|$)',
            text,
        )
        if match:
            value = _normalize_space(match.group(1).strip(" \t\r\n\"'}]"))
            if value:
                return value
    return ""


def _apply_text_fallback_fields(action: OrderedDict[str, Any], kv_text: str) -> None:
    action_name = str(action.get("action") or "").upper()
    unlabeled_points = _extract_unlabeled_points_from_text(kv_text, max_points=4)
    if action_name in {"CLICK", "LONGPRESS", "DOUBLECLICK", "TYPE"}:
        if not any(key in action for key in ("point", "coordinate", "coordinates")):
            point = _extract_norm_point_from_text(kv_text, ("point", "coordinate", "coordinates"))
            if point is None and unlabeled_points:
                point = unlabeled_points[0]
            if point is not None:
                action["point"] = point

    if action_name == "SLIDE":
        if "point1" not in action and "start_coordinate" not in action:
            point1 = _extract_norm_point_from_text(kv_text, ("point1", "start_coordinate"))
            if point1 is None and len(unlabeled_points) >= 1:
                point1 = unlabeled_points[0]
            if point1 is not None:
                action["point1"] = point1
        if "point2" not in action and "end_coordinate" not in action:
            point2 = _extract_norm_point_from_text(kv_text, ("point2", "end_coordinate"))
            if point2 is None and len(unlabeled_points) >= 2:
                point2 = unlabeled_points[1]
            if point2 is not None:
                action["point2"] = point2
        if "point" not in action and "direction" in action:
            point = _extract_norm_point_from_text(kv_text, ("point", "coordinate", "coordinates"))
            if point is None and unlabeled_points:
                point = unlabeled_points[0]
            if point is not None:
                action["point"] = point
        if "direction" not in action:
            direction = _extract_direction(kv_text)
            if direction:
                action["direction"] = direction

    if action_name in {"TYPE", "AWAKE", "INFO", "ABORT", "CALLUSER", "ANSWER"} and not action.get("value"):
        value = _extract_text_field_from_text(
            kv_text,
            (
                "value",
                "text",
                "app_name",
                "return",
                "answer",
                "respond",
                "response",
                "reply",
                "final_answer",
                "content",
            ),
        )
        if value:
            action["value"] = value

    if action_name == "HOTKEY" and not action.get("key"):
        key = _extract_text_field_from_text(kv_text, ("key", "value"))
        if key:
            action["key"] = key

    if action_name == "COMPLETE" and not action.get("status"):
        status = _extract_text_field_from_text(kv_text, ("status",))
        if status:
            action["status"] = status


def _parse_action_tool_json(response_text: str, cot: str = "") -> OrderedDict[str, Any]:
    payload = _safe_json_loads(response_text)
    if not isinstance(payload, dict):
        raise seeact_utils.ParseActionError("action_tool payload is not a JSON object")

    if isinstance(payload.get("arguments"), dict):
        payload = dict(payload["arguments"])

    raw_action = payload.get("action_type")
    if not raw_action:
        raise seeact_utils.ParseActionError("action_tool JSON missing action_type")

    parsed = OrderedDict()
    parsed["cot"] = cot
    parsed["action"] = _canonical_action_name(raw_action)

    for src, dst in (("summary", "summary"), ("explain", "explain"), ("status", "status"), ("key", "key")):
        value = _normalize_space(payload.get(src))
        if value:
            parsed[dst] = value

    for key in ("point", "coordinate", "coordinates"):
        point = _extract_coordinate(payload, key)
        if point is not None:
            parsed["point"] = _clip_norm_point(point)
            break

    point1 = _extract_coordinate(payload, "point1", "start_coordinate")
    point2 = _extract_coordinate(payload, "point2", "end_coordinate")
    if point1 is not None:
        parsed["point1"] = _clip_norm_point(point1)
    if point2 is not None:
        parsed["point2"] = _clip_norm_point(point2)

    if "point" not in parsed:
        x = _safe_int(payload.get("x"))
        y = _safe_int(payload.get("y"))
        if x is not None and y is not None:
            parsed["point"] = _clip_norm_point([x, y])

    direction = _extract_direction(payload.get("direction"))
    if direction:
        parsed["direction"] = direction

    value = payload.get("value")
    if value is None:
        for key in ("text", "app_name", "return"):
            if payload.get(key) is not None:
                value = payload.get(key)
                break
    if value is not None:
        parsed["value"] = str(value)
    if payload.get("return") is not None:
        parsed["return"] = str(payload.get("return"))

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
            answer_text = _normalize_space(
                action.get("value")
                or action.get("answer")
                or action.get("respond")
                or action.get("response")
                or action.get("reply")
                or action.get("final_answer")
                or action.get("content")
            )
            if answer_text:
                action["action"] = "ANSWER"
                action["value"] = answer_text
        if "action" not in action:
            raise seeact_utils.ParseActionError("missing GELAB action field")
        action["action"] = _canonical_action_name(action["action"])
        for key in (
            "point",
            "coordinate",
            "coordinates",
            "point1",
            "point2",
            "start_coordinate",
            "end_coordinate",
        ):
            if key in action:
                action[key] = _parse_point(action[key])
        _apply_text_fallback_fields(action, kv_text)
        return action
    except seeact_utils.ParseActionError as kv_error:
        try:
            return _parse_action_tool_json(response_text, cot=cot)
        except seeact_utils.ParseActionError as action_tool_error:
            try:
                return _parse_tool_call_response(response_text, cot=cot)
            except seeact_utils.ParseActionError as tool_error:
                raise seeact_utils.ParseActionError(
                    f"{kv_error}; action-tool fallback failed: {action_tool_error}; tool-call fallback failed: {tool_error}"
                ) from tool_error
        except Exception as action_tool_error:  # pylint: disable=broad-exception-caught
            raise seeact_utils.ParseActionError(
                f"{kv_error}; action-tool fallback failed: {action_tool_error}"
            ) from action_tool_error


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

        if tool_action in {"double_tap", "double_click"}:
            coordinate = _extract_coordinate(arguments, "coordinate")
            if coordinate is None:
                raise seeact_utils.ParseActionError("tool-call double_tap missing coordinate")
            return (
                json_action.JSONAction(action_type=json_action.DOUBLE_TAP, x=coordinate[0], y=coordinate[1]),
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
            direction = _extract_direction(arguments.get("direction"))
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

        if tool_action in {"hot_key", "hotkey"}:
            key = _normalize_space(arguments.get("key") or arguments.get("value")).lower()
            if key == "back":
                return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK), tool_call, extras
            if key == "home":
                return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME), tool_call, extras
            if key == "enter":
                return json_action.JSONAction(action_type=json_action.KEYBOARD_ENTER), tool_call, extras
            raise seeact_utils.ParseActionError(f"unsupported HOT_KEY: {key}")

        if tool_action == "wait":
            wait_value = max(1, _safe_int(arguments.get("value")) or 1)
            extras["wait_seconds"] = wait_value
            return json_action.JSONAction(action_type=json_action.WAIT), tool_call, extras

        if tool_action == "answer":
            text = _normalize_space(arguments.get("text"))
            return json_action.JSONAction(action_type=json_action.ANSWER, text=text), tool_call, extras

        if tool_action == "call_user":
            text = _normalize_space(arguments.get("text") or arguments.get("value"))
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
        point = parsed_action.get("point") or parsed_action.get("coordinate") or parsed_action.get("coordinates")
        if point is None:
            raise seeact_utils.ParseActionError("CLICK requires point/coordinate")
        abs_point = _norm_to_abs(point, screen_size)
        tool_call["arguments"] = {"action": "click", "coordinate": abs_point}
        return (
            json_action.JSONAction(action_type=json_action.CLICK, x=abs_point[0], y=abs_point[1]),
            tool_call,
            extras,
        )

    if action_type == "LONGPRESS":
        point = parsed_action.get("point") or parsed_action.get("coordinate") or parsed_action.get("coordinates")
        if point is None:
            raise seeact_utils.ParseActionError("LONGPRESS requires point/coordinate")
        abs_point = _norm_to_abs(point, screen_size)
        tool_call["arguments"] = {"action": "long_press", "coordinate": abs_point}
        return (
            json_action.JSONAction(action_type=json_action.LONG_PRESS, x=abs_point[0], y=abs_point[1]),
            tool_call,
            extras,
        )

    if action_type == "DOUBLECLICK":
        point = parsed_action.get("point") or parsed_action.get("coordinate") or parsed_action.get("coordinates")
        if point is None:
            raise seeact_utils.ParseActionError("DOUBLECLICK requires point/coordinate")
        abs_point = _norm_to_abs(point, screen_size)
        tool_call["arguments"] = {"action": "double_tap", "coordinate": abs_point}
        return (
            json_action.JSONAction(action_type=json_action.DOUBLE_TAP, x=abs_point[0], y=abs_point[1]),
            tool_call,
            extras,
        )

    if action_type == "TYPE":
        value = str(parsed_action.get("value") or "")
        if not value:
            raise seeact_utils.ParseActionError("TYPE requires value")
        point = parsed_action.get("point") or parsed_action.get("coordinate") or parsed_action.get("coordinates")
        if point is None:
            tool_call["arguments"] = {"action": "type", "text": value}
            return json_action.JSONAction(action_type=json_action.INPUT_TEXT, text=value), tool_call, extras
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
        point1 = parsed_action.get("point1") or parsed_action.get("start_coordinate")
        point2 = parsed_action.get("point2") or parsed_action.get("end_coordinate")
        direction = _extract_direction(parsed_action.get("direction") or parsed_action.get("value"))
        if point1 is not None and point2 is not None:
            start_abs = _norm_to_abs(point1, screen_size)
            end_abs = _norm_to_abs(point2, screen_size)
            if not direction:
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
        if not direction:
            raise seeact_utils.ParseActionError("SLIDE requires point1/point2 or direction")
        tool_call["arguments"] = {"action": "swipe", "direction": direction}
        extras["direction"] = direction
        return json_action.JSONAction(action_type=json_action.SWIPE, direction=direction), tool_call, extras

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

    if action_type == "HOTKEY":
        key = _normalize_space(parsed_action.get("key") or parsed_action.get("value")).lower()
        if key == "back":
            tool_call["arguments"] = {"action": "system_button", "button": "back"}
            return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK), tool_call, extras
        if key == "home":
            tool_call["arguments"] = {"action": "system_button", "button": "home"}
            return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME), tool_call, extras
        if key == "enter":
            tool_call["arguments"] = {"action": "system_button", "button": "enter"}
            return json_action.JSONAction(action_type=json_action.KEYBOARD_ENTER), tool_call, extras
        raise seeact_utils.ParseActionError("HOTKEY requires key in {ENTER,BACK,HOME}")

    if action_type == "COMPLETE":
        status = _normalize_space(parsed_action.get("status")).lower()
        failed = status in {"failure", "fail", "failed", "infeasible"}
        result_text = _normalize_space(parsed_action.get("return") or parsed_action.get("value"))
        if result_text.lower() in {"success", "failure", "fail", "failed"}:
            result_text = ""
        if result_text:
            extras["return_text"] = result_text
            # Embed return_text in tool_call so callers can pick it up and set
            # interaction_cache without needing to track extras separately.
            tool_call["arguments"] = {
                "action": "terminate",
                "status": "fail" if failed else "success",
                "return_text": result_text,
            }
        else:
            tool_call["arguments"] = {"action": "terminate", "status": "fail" if failed else "success"}
        return (
            json_action.JSONAction(
                action_type=json_action.STATUS,
                goal_status="infeasible" if failed else "task_complete",
            ),
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

    if action_type == "ANSWER":
        answer = _normalize_space(parsed_action.get("value") or parsed_action.get("return"))
        if not answer:
            raise seeact_utils.ParseActionError("ANSWER requires value")
        tool_call["arguments"] = {"action": "answer", "text": answer}
        return (
            json_action.JSONAction(action_type=json_action.ANSWER, text=answer),
            tool_call,
            extras,
        )

    if action_type == "CALLUSER":
        question = _normalize_space(parsed_action.get("value") or parsed_action.get("return"))
        if not question:
            question = "Need user input to continue."
        tool_call["arguments"] = {"action": "answer", "text": question}
        return (
            json_action.JSONAction(action_type=json_action.ANSWER, text=question),
            tool_call,
            extras,
        )

    raise seeact_utils.ParseActionError(f"unsupported GELAB action: {action_type}")


def build_gelab_messages(
    goal: str,
    history: str,
    screenshot: Image.Image,
) -> list[dict[str, Any]]:
    user_text = (
        f"Task:\n{goal}\n\n"
        f"History actions:\n{history or 'None yet.'}\n\n"
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
        hints: list[str] = []
        history = self._history_text()
        messages = build_gelab_messages(goal, history, screenshot)
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
