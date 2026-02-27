"""Utility functions for ExplorerElementAgent parsing and normalization."""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

import numpy as np
from PIL import Image

from android_world.agents import seeact_utils
from android_world.env import json_action
from android_world.agents.explorer_agent_constants import _DIR_SET, _STOP_TOKENS


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_\\-]{2,}", text.lower()) if t not in _STOP_TOKENS]


def _to_data_url(image_array: np.ndarray) -> str:
    image = Image.fromarray(image_array)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{payload}"


def _extract_first_json_block(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _safe_json_loads(payload: str) -> dict[str, Any]:
    s = payload.strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.startswith("json"):
            s = s[4:].lstrip()

    s = re.sub(r"\}\s*\}\s*\]", "]]", s)
    s = re.sub(r"\]\s*\}+", "]", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)

    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        s = s[first : last + 1]

    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise seeact_utils.ParseActionError("tool_call is not a dict")
        return obj
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    def _extract(pattern: str, default: str | None = None) -> str | None:
        m = re.search(pattern, s)
        return m.group(1) if m else default

    def _extract_coord(key: str | None = None) -> list[int] | None:
        if key is not None:
            m = re.search(
                rf'"{key}"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)[^\]]*\]',
                s,
            )
            if m:
                return [int(round(float(m.group(1)))), int(round(float(m.group(2))))]
            m = re.search(
                rf'"{key}"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)',
                s,
            )
            if m:
                return [int(round(float(m.group(1)))), int(round(float(m.group(2))))]
            m = re.search(
                rf'"{key}"\s*:\s*"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"',
                s,
            )
            if m:
                return [int(round(float(m.group(1)))), int(round(float(m.group(2))))]
        m = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", s)
        if m:
            return [int(round(float(m.group(1)))), int(round(float(m.group(2))))]
        return None

    name = _extract(r'"name"\s*:\s*"([^"]+)"', "mobile_use")
    act = _extract(r'"action"\s*:\s*"([^"]+)"')
    if not act:
        act = _extract(r'"action_type"\s*:\s*"([^"]+)"')
    act = (act or "wait").strip()

    text = _extract(r'"text"\s*:\s*"([^"]*)"', "")
    button = _extract(r'"button"\s*:\s*"([^"]+)"')
    direction = _extract(r'"direction"\s*:\s*"([^"]+)"')
    status = _extract(r'"status"\s*:\s*"([^"]+)"', "fail")
    goal_status = _extract(r'"goal_status"\s*:\s*"([^"]+)"')
    app_name = _extract(r'"app_name"\s*:\s*"([^"]+)"')
    open_text = _extract(r'"text"\s*:\s*"([^"]+)"')

    args: dict[str, Any] = {"action": act}
    act_low = act.lower()
    if act_low in {"click", "long_press", "tap"}:
        coord = _extract_coord("coordinate")
        if coord is not None:
            args["coordinate"] = coord
    elif act_low in {"type", "input_text"}:
        args["text"] = text or ""
        coord = _extract_coord("coordinate")
        if coord is not None:
            args["coordinate"] = coord
    elif act_low in {"swipe", "scroll"}:
        args["direction"] = direction or "down"
        coord = _extract_coord("coordinate")
        if coord is not None:
            args["coordinate"] = coord
        start_coord = _extract_coord("start_coordinate")
        end_coord = _extract_coord("end_coordinate")
        if start_coord is not None and end_coord is not None:
            args["start_coordinate"] = start_coord
            args["end_coordinate"] = end_coord
    elif act_low in {"open", "open_app"}:
        args["text"] = open_text or app_name or ""
        if app_name:
            args["app_name"] = app_name
    elif act_low in {"system_button", "back"}:
        args["action"] = "system_button"
        args["button"] = button or ("back" if act_low == "back" else "back")
    elif act_low == "answer":
        args["text"] = text or ""
    elif act_low in {"terminate", "status"}:
        args["status"] = status
        if goal_status:
            args["goal_status"] = goal_status
    else:
        args["action"] = "wait"

    fixed_obj = {"name": name or "mobile_use", "arguments": args}
    print("[safe_json_loads] recovered ->", fixed_obj)
    return fixed_obj


def parse_tool_call(response_text: str) -> dict[str, Any]:
    if not response_text:
        raise seeact_utils.ParseActionError("empty response")
    text = str(response_text)

    block = None
    if "<tool_call>" in text and "</tool_call>" in text:
        try:
            block = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise seeact_utils.ParseActionError(f"malformed tool_call: {exc}")
    if block is None:
        block = _extract_first_json_block(text)
    if block is None:
        raise seeact_utils.ParseActionError("cannot find action JSON")

    obj = _safe_json_loads(block)

    if "arguments" in obj:
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = _safe_json_loads(args)
            except Exception:  # pylint: disable=broad-exception-caught
                args = {"text": args}
        if (
            isinstance(args, dict)
            and "arguments" in args
            and "action" not in args
            and "action_type" not in args
            and isinstance(args.get("arguments"), dict)
        ):
            args = args.get("arguments")
        if not isinstance(args, dict):
            args = {}
        name = str(obj.get("name") or "mobile_use").strip().lower()
        if "action" not in args and "action_type" in args:
            args["action"] = args.get("action_type")
        if "action" not in args and name != "mobile_use":
            args["action"] = name
        return {"name": "mobile_use", "arguments": args}

    if isinstance(obj.get("action"), str):
        return {"name": "mobile_use", "arguments": obj}

    if isinstance(obj.get("action_type"), str):
        args = dict(obj)
        args["action"] = args.get("action_type")
        return {"name": "mobile_use", "arguments": args}

    if isinstance(obj.get("name"), str):
        name = str(obj["name"]).strip().lower()
        if name != "mobile_use":
            args = {k: v for k, v in obj.items() if k != "name"}
            args["action"] = name
            return {"name": "mobile_use", "arguments": args}

    raise seeact_utils.ParseActionError("unsupported tool_call schema")


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _target_index(args: dict[str, Any]) -> int | None:
    idx = _safe_int(args.get("element_id"))
    if idx is None:
        idx = _safe_int(args.get("index"))
    return idx


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _as_xy(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _safe_float(value[0])
        y = _safe_float(value[1])
        if x is not None and y is not None:
            return int(round(x)), int(round(y))
        return None
    if isinstance(value, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", value)
        if len(nums) >= 2:
            return int(round(float(nums[0]))), int(round(float(nums[1])))
    return None


def _extract_coordinate(args: dict[str, Any]) -> tuple[int, int] | None:
    for key in ("coordinate", "point", "xy", "tap_point"):
        if key in args:
            xy = _as_xy(args.get(key))
            if xy is not None:
                if xy[0] < 0 or xy[1] < 0:
                    return None
                return xy
    point_obj = args.get("coordinate")
    if isinstance(point_obj, dict):
        x = _safe_float(point_obj.get("x"))
        y = _safe_float(point_obj.get("y"))
        if x is not None and y is not None and x >= 0 and y >= 0:
            return int(round(x)), int(round(y))
    x = _safe_float(args.get("x"))
    y = _safe_float(args.get("y"))
    if x is not None and y is not None and x >= 0 and y >= 0:
        return int(round(x)), int(round(y))
    return None


def _clamp_coordinate_to_screen(
    coordinate: tuple[int, int],
    screen_size: tuple[int, int] | None,
) -> tuple[int, int]:
    x, y = int(coordinate[0]), int(coordinate[1])
    if not screen_size:
        return x, y
    width, height = int(screen_size[0]), int(screen_size[1])
    if width > 0:
        x = min(max(0, x), width - 1)
    if height > 0:
        y = min(max(0, y), height - 1)
    return x, y


def _scale_coordinate_by_mode(
    coordinate: tuple[int, int],
    screen_size: tuple[int, int] | None,
    mode: str,
) -> tuple[int, int]:
    x, y = float(coordinate[0]), float(coordinate[1])
    m = (mode or "auto").strip().lower()
    if not screen_size:
        return int(round(x)), int(round(y))
    width, height = float(screen_size[0]), float(screen_size[1])
    if width <= 1 or height <= 1:
        return int(round(x)), int(round(y))

    if m == "absolute":
        return _clamp_coordinate_to_screen((int(round(x)), int(round(y))), screen_size)

    if m in {"ratio_1", "normalized_1", "1"}:
        return _clamp_coordinate_to_screen((int(round(x * width)), int(round(y * height))), screen_size)

    if m in {"ratio_1000", "normalized_1000", "1000"}:
        return _clamp_coordinate_to_screen(
            (int(round(x / 1000.0 * width)), int(round(y / 1000.0 * height))),
            screen_size,
        )

    # Auto mode: prefer absolute pixels whenever the point already fits the
    # current screen, and only fall back to normalized scaling when clearly needed.
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return _clamp_coordinate_to_screen((int(round(x * width)), int(round(y * height))), screen_size)
    if 0.0 <= x <= width and 0.0 <= y <= height:
        return _clamp_coordinate_to_screen((int(round(x)), int(round(y))), screen_size)
    if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
        return _clamp_coordinate_to_screen(
            (int(round(x / 1000.0 * width)), int(round(y / 1000.0 * height))),
            screen_size,
        )
    return _clamp_coordinate_to_screen((int(round(x)), int(round(y))), screen_size)


def _nearest_interactive_distance(
    point: tuple[int, int],
    ui_elements: list[Any],
) -> float | None:
    best: float | None = None
    for element in ui_elements:
        if not bool(
            getattr(element, "is_clickable", False)
            or getattr(element, "is_long_clickable", False)
            or getattr(element, "is_editable", False)
            or getattr(element, "is_scrollable", False)
        ):
            continue
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            continue
        center_x = float(bbox.x_min + bbox.x_max) / 2.0
        center_y = float(bbox.y_min + bbox.y_max) / 2.0
        dist = ((float(point[0]) - center_x) ** 2 + (float(point[1]) - center_y) ** 2) ** 0.5
        if best is None or dist < best:
            best = dist
    return best


def _resolve_coordinate_by_mode(
    coordinate: tuple[int, int],
    screen_size: tuple[int, int] | None,
    mode: str,
    ui_elements: list[Any],
) -> tuple[int, int]:
    m = (mode or "auto").strip().lower()
    if m != "auto":
        return _scale_coordinate_by_mode(coordinate, screen_size, m)
    if not screen_size:
        return _scale_coordinate_by_mode(coordinate, screen_size, "auto")

    x = float(coordinate[0])
    y = float(coordinate[1])
    width = float(screen_size[0])
    height = float(screen_size[1])
    if width <= 1.0 or height <= 1.0:
        return _scale_coordinate_by_mode(coordinate, screen_size, "auto")

    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return _scale_coordinate_by_mode(coordinate, screen_size, "1")

    in_absolute = 0.0 <= x <= width and 0.0 <= y <= height
    in_1000 = 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0
    if not in_absolute:
        return _scale_coordinate_by_mode(coordinate, screen_size, "auto")
    if not in_1000:
        return _clamp_coordinate_to_screen((int(round(x)), int(round(y))), screen_size)
    if width <= 1000.0 and height <= 1000.0:
        return _clamp_coordinate_to_screen((int(round(x)), int(round(y))), screen_size)

    absolute_xy = _clamp_coordinate_to_screen((int(round(x)), int(round(y))), screen_size)
    scaled_1000_xy = _scale_coordinate_by_mode(coordinate, screen_size, "1000")
    if scaled_1000_xy == absolute_xy:
        return absolute_xy

    abs_dist = _nearest_interactive_distance(absolute_xy, ui_elements)
    scaled_dist = _nearest_interactive_distance(scaled_1000_xy, ui_elements)
    if abs_dist is None or scaled_dist is None:
        return absolute_xy

    # Prefer 1000-space mapping when it is clearly closer to interactive targets.
    if scaled_dist + 24.0 < abs_dist:
        return scaled_1000_xy
    diag = (width**2 + height**2) ** 0.5
    if abs_dist > 0.35 * diag and scaled_dist < 0.2 * diag:
        return scaled_1000_xy
    return absolute_xy


def _infer_swipe_direction_from_coordinates(
    start_xy: tuple[int, int] | None,
    end_xy: tuple[int, int] | None,
    fallback: str = "down",
) -> str:
    if start_xy is None or end_xy is None:
        return fallback
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _normalize_dir(value: Any, fallback: str = "down") -> str:
    direction = str(value or fallback).strip().lower()
    if direction in _DIR_SET:
        return direction
    return fallback


def _goal_status(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in {"success", "complete", "done", "completed", "task_complete"}:
        return "complete"
    return "infeasible"


def _looks_like_back_intent(text: str) -> bool:
    low = str(text or "").lower()
    patterns = [
        '"action":"navigate_back"',
        '"action":"back"',
        '"action_type":"navigate_back"',
        '"button":"back"',
        "'button': 'back'",
        "navigate_back",
        "system_button",
    ]
    if any(p in low for p in patterns):
        return "back" in low
    return False


def _to_json_action(
    tool_call: dict[str, Any],
    ui_elements: list[Any],
    fallback_index: int | None = None,
    logical_screen_size: tuple[int, int] | None = None,
    coordinate_mode: str = "1000",
) -> json_action.JSONAction:
    args = tool_call.get("arguments") or {}
    action = str(args.get("action") or args.get("action_type") or "").strip().lower()

    def pick_index(allow_fallback: bool = False) -> int | None:
        idx = _target_index(args)
        if idx is not None and 0 <= idx < len(ui_elements):
            return idx
        if allow_fallback and fallback_index is not None and 0 <= fallback_index < len(ui_elements):
            return fallback_index
        return None

    def pick_coordinate() -> tuple[int, int] | None:
        return _extract_coordinate(args)

    if action == "click":
        coordinate = pick_coordinate()
        if "coordinate" in args and coordinate is None:
            return json_action.JSONAction(action_type=json_action.UNKNOWN)
        if coordinate is not None:
            x, y = _resolve_coordinate_by_mode(
                coordinate,
                logical_screen_size,
                coordinate_mode,
                ui_elements,
            )
            return json_action.JSONAction(action_type=json_action.CLICK, x=x, y=y)
        idx = pick_index(allow_fallback=False)
        if idx is None:
            return json_action.JSONAction(action_type=json_action.UNKNOWN)
        return json_action.JSONAction(action_type=json_action.CLICK, index=idx)

    if action == "long_press":
        coordinate = pick_coordinate()
        if "coordinate" in args and coordinate is None:
            return json_action.JSONAction(action_type=json_action.UNKNOWN)
        if coordinate is not None:
            x, y = _resolve_coordinate_by_mode(
                coordinate,
                logical_screen_size,
                coordinate_mode,
                ui_elements,
            )
            return json_action.JSONAction(action_type=json_action.LONG_PRESS, x=x, y=y)
        idx = pick_index(allow_fallback=False)
        if idx is None:
            return json_action.JSONAction(action_type=json_action.UNKNOWN)
        return json_action.JSONAction(action_type=json_action.LONG_PRESS, index=idx)

    if action in {"type", "input_text"}:
        coordinate = pick_coordinate()
        if "coordinate" in args and coordinate is None:
            return json_action.JSONAction(action_type=json_action.UNKNOWN)
        if coordinate is not None:
            x, y = _resolve_coordinate_by_mode(
                coordinate,
                logical_screen_size,
                coordinate_mode,
                ui_elements,
            )
            return json_action.JSONAction(
                action_type=json_action.INPUT_TEXT,
                x=x,
                y=y,
                text=str(args.get("text", "")),
            )
        return json_action.JSONAction(
            action_type=json_action.INPUT_TEXT,
            index=pick_index(allow_fallback=False),
            text=str(args.get("text", "")),
        )

    if action in {"swipe", "scroll"}:
        start_xy = _as_xy(args.get("start_coordinate"))
        end_xy = _as_xy(args.get("end_coordinate"))
        direction = _normalize_dir(
            args.get("direction"),
            fallback=_infer_swipe_direction_from_coordinates(start_xy, end_xy, fallback="down"),
        )
        idx = pick_index()
        if action == "scroll" or idx is not None:
            return json_action.JSONAction(
                action_type=json_action.SCROLL,
                direction=direction,
                index=idx,
            )
        return json_action.JSONAction(action_type=json_action.SWIPE, direction=direction)

    if action in {"open", "open_app"}:
        return json_action.JSONAction(
            action_type=json_action.OPEN_APP,
            app_name=str(args.get("text") or args.get("app_name") or ""),
        )

    if action == "system_button":
        btn = str(args.get("button", "back")).strip().lower()
        if btn == "back":
            return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)
        if btn == "home":
            return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME)
        if btn == "enter":
            return json_action.JSONAction(action_type=json_action.KEYBOARD_ENTER)
        return json_action.JSONAction(action_type=json_action.WAIT)

    if action == "wait":
        return json_action.JSONAction(action_type=json_action.WAIT)

    if action == "answer":
        return json_action.JSONAction(
            action_type=json_action.STATUS,
            goal_status="complete",
        )

    if action in {"terminate", "status", "finish", "done"}:
        return json_action.JSONAction(
            action_type=json_action.STATUS,
            goal_status=_goal_status(args.get("status") or args.get("goal_status")),
        )

    if action in {"back", "navigate_back"}:
        return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)

    if action in {"home", "navigate_home"}:
        return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME)

    if action in {"enter", "keyboard_enter"}:
        return json_action.JSONAction(action_type=json_action.KEYBOARD_ENTER)

    return json_action.JSONAction(action_type=json_action.UNKNOWN)


def _pixel_delta(prev: np.ndarray | None, curr: np.ndarray | None) -> float | None:
    if prev is None or curr is None:
        return None
    if prev.shape != curr.shape:
        return None
    return float(np.abs(prev.astype(np.int16) - curr.astype(np.int16)).mean())


def _element_hint_compact_label(element: Any) -> str:
    text = _normalize_space(getattr(element, "text", ""))
    desc = _normalize_space(getattr(element, "content_description", ""))
    rid = _normalize_space(getattr(element, "resource_name", "") or getattr(element, "resource_id", ""))
    bbox = getattr(element, "bbox_pixels", None)
    if bbox is not None:
        center = [int((bbox.x_min + bbox.x_max) / 2.0), int((bbox.y_min + bbox.y_max) / 2.0)]
    else:
        center = None
    flags = []
    if bool(getattr(element, "is_clickable", False)):
        flags.append("click")
    if bool(getattr(element, "is_editable", False)):
        flags.append("edit")
    if bool(getattr(element, "is_scrollable", False)):
        flags.append("scroll")
    if bool(getattr(element, "is_long_clickable", False)):
        flags.append("long")
    key_text = text or desc or rid or "unnamed"
    return f"text='{text}', desc='{desc}', key='{key_text}', id='{rid}', center={center}, flags={flags}"


def _extract_task_queries(text: str) -> list[str]:
    value = _normalize_space(text)
    if not value:
        return []
    chunks = re.split(r"[.;,\n]|\bthen\b|\band then\b", value, flags=re.IGNORECASE)
    out = [value]
    seen = {value.lower()}
    for chunk in chunks:
        chunk = _normalize_space(chunk)
        key = chunk.lower()
        if len(chunk) < 4 or key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def _phash_pixels(pixels: np.ndarray) -> int:
    img = Image.fromarray(pixels).convert("L").resize((9, 8), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = 0
    for flag in diff.flatten():
        bits = (bits << 1) | int(bool(flag))
    return bits


def _hash_diff(h1: int, h2: int) -> int:
    return int((int(h1) ^ int(h2)).bit_count())


def _mae_small(a_pixels: np.ndarray, b_pixels: np.ndarray) -> float:
    a = np.asarray(Image.fromarray(a_pixels).convert("L").resize((18, 40), Image.BILINEAR), dtype=np.float32)
    b = np.asarray(Image.fromarray(b_pixels).convert("L").resize((18, 40), Image.BILINEAR), dtype=np.float32)
    return float(np.mean(np.abs(a - b)))
