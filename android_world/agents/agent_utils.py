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

"""Utilities for agents."""

import ast
import json
import re
from typing import Any, Dict, Optional, Tuple


def extract_json(s: str) -> dict[str, Any] | None:
    """Extracts JSON from string.

  Tries conversion with ast and json modules.

  Args:
    s: A string with a JSON in it. E.g., "{'hello': 'world'}" or from CoT:
      "let's think step-by-step, ..., {'hello': 'world'}".

  Returns:
    JSON object.
  """
    pattern = r'\{.*?\}'
    match = re.search(pattern, s)
    # print(f"[DEBUG] Match: {match} ")
    if match:
        try:
            return ast.literal_eval(match.group())
        except (SyntaxError, ValueError) as error:
            try:
                # Try conversion with json module.
                return json.loads(match.group())
            except (SyntaxError, ValueError) as error2:
                print('Cannot extract JSON, skipping due to errors %s and %s', error, error2,)
                return None
    else:
        return None


_DIR_WORDS = {"up", "down", "left", "right"}


def _to_xy(v: Any) -> Optional[Tuple[float, float]]:
    """
    Accepts:
      - tuple/list: (x, y) or [x, y]
      - string: "(x,y)" or "x,y" or "(x, y)"
    Returns (x, y) as floats, or None.
    """
    if isinstance(v, (tuple, list)) and len(v) == 2:
        try:
            return float(v[0]), float(v[1])
        except Exception:
            return None

    if isinstance(v, str):
        s = v.strip()
        # remove surrounding quotes are already handled by ast in many cases,
        # but keep it safe.
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1].strip()

        # strip parentheses
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()

        # now should look like "x,y"
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except Exception:
            return None

    return None


def _infer_scroll_direction_from_swipe(start_xy: Tuple[float, float], end_xy: Tuple[float, float]) -> str:
    """
    Android screen coords: origin top-left, y increases downward.

    We infer direction in terms of *scroll direction* (as in AndroidWorld action space):
      - finger swipes up (end_y < start_y) => scroll down (content moves up) => direction='down'
      - finger swipes down (end_y > start_y) => scroll up  => direction='up'
      - finger swipes left (end_x < start_x) => scroll right => direction='right'
      - finger swipes right (end_x > start_x) => scroll left  => direction='left'

    If ambiguous (very small movement), default to 'down'.
    """
    sx, sy = start_xy
    ex, ey = end_xy
    dx = ex - sx
    dy = ey - sy

    # choose dominant axis
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "down"

    if abs(dy) >= abs(dx):
        # vertical swipe dominates
        if ey < sy:
            return "down"   # swipe up => scroll down
        else:
            return "up"     # swipe down => scroll up
    else:
        # horizontal swipe dominates
        if ex < sx:
            return "right"  # swipe left => scroll right
        else:
            return "left"   # swipe right => scroll left


def parse_function_action(action: str) -> Optional[Dict[str, Any]]:
    """
    Parse various action styles into JSONAction-compatible dict.

    Supported:
      1) function style:
         open_app(app_name='Markor')
         scroll(start_box='(511,611)', end_box='(511,369)')
         scroll(direction='up')
         scroll('up')  # positional
      2) natural language short form:
         scroll up
         scroll down
         click 123 456   (optional: basic support)
    """
    if action is None:
        return None

    raw = action.strip()

    # Some models may output "Action: xxx" or include extra tokens; strip simple prefix.
    raw = re.sub(r"^\s*Action\s*:\s*", "", raw).strip()

    # ---- Case A: "scroll up/down/left/right" ----
    m = re.match(r"^scroll\s+(up|down|left|right)\s*$", raw, flags=re.IGNORECASE)
    if m:
        direction = m.group(1).lower()
        return {"action_type": "scroll", "direction": direction}

    # ---- (Optional) Case A2: "scroll" alone -> default down ----
    if re.fullmatch(r"scroll", raw, flags=re.IGNORECASE):
        return {"action_type": "scroll", "direction": "down"}

    # ---- (Optional) basic "click x y" support (helps with sloppy outputs) ----
    m = re.match(r"^click\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$", raw, flags=re.IGNORECASE)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        # match AndroidWorld style: click(point='<point>x y</point>') is common, but JSONAction
        # may accept x/y or point; adapt if your JSONAction expects different keys.
        return {"action_type": "click", "x": x, "y": y}

    # ---- Case B: function call style via AST (robust to commas in "(x,y)") ----
    # Strict pattern check first (fast reject)
    m = re.match(r"^([A-Za-z_]\w*)\((.*)\)\s*$", raw, flags=re.S)
    if not m:
        return None

    func_name = m.group(1)
    # Parse as a Python expression call: func_name(...)
    try:
        expr = ast.parse(raw, mode="eval")
        if not isinstance(expr.body, ast.Call):
            return None
        call: ast.Call = expr.body
    except Exception:
        return None

    action_type = func_name

    # Extract keyword args safely
    args: Dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            # **kwargs is not supported here
            continue
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            # fallback: try to keep string form
            try:
                args[kw.arg] = ast.unparse(kw.value)
            except Exception:
                args[kw.arg] = None

    # Extract positional args (rare but happens)
    pos_args = []
    for a in call.args:
        try:
            pos_args.append(ast.literal_eval(a))
        except Exception:
            try:
                pos_args.append(ast.unparse(a))
            except Exception:
                pos_args.append(None)

    # ---- Normalize typing / input actions ----
    if action_type.lower() in {"type", "input", "enter_text", "input_text"}:

        # rename "content" -> "text"
        if "text" not in args and "content" in args:
            args["text"] = args.pop("content")

        # support positional form: type("Grace")
        if "text" not in args and len(pos_args) == 1 and isinstance(pos_args[0], str):
            args["text"] = pos_args[0]

        # ensure final action name
        action_type = "input_text"

        # 默认先不指定 index，后续再补
        if "index" not in args:
            args["index"] = None

        return {"action_type": action_type, **args}

    # Normalize some common sloppy variants
    # Example: scroll('up') or scroll("down")
    if action_type.lower() == "scroll":
        if "direction" not in args and len(pos_args) == 1 and isinstance(pos_args[0], str):
            d = pos_args[0].strip().lower()
            if d in _DIR_WORDS:
                args["direction"] = d

        # If the model outputs start_box/end_box, infer direction if missing
        if "direction" not in args and ("start_box" in args or "start_point" in args) and ("end_box" in args or "end_point" in args):
            start_val = args.get("start_box", args.get("start_point"))
            end_val = args.get("end_box",   args.get("end_point"))

            start_xy = _to_xy(start_val)
            end_xy = _to_xy(end_val)

            if start_xy is not None and end_xy is not None:
                args["direction"] = _infer_scroll_direction_from_swipe(start_xy, end_xy)

        # If still missing direction, try to infer from y1/y2 keys (some models do this)
        if "direction" not in args:
            # common pattern: start_x/start_y/end_x/end_y
            if {"start_x", "start_y", "end_x", "end_y"}.issubset(args.keys()):
                try:
                    start_xy = (float(args["start_x"]), float(args["start_y"]))
                    end_xy = (float(args["end_x"]), float(args["end_y"]))
                    args["direction"] = _infer_scroll_direction_from_swipe(start_xy, end_xy)
                except Exception:
                    pass

        # final fallback
        if "direction" not in args:
            args["direction"] = "down"

        return {"action_type": "scroll", **args}

    # Default: return function name + args
    # This keeps open_app(app_name='...') etc.
    return {"action_type": action_type, **args}


COORD_KEYS = {
    "start_box", "end_box",
    "start_point", "end_point",
    "point",
    "x", "y",
    "x1", "y1", "x2", "y2",
}


COORDINATE_ACTIONS = {
    "click",
    "long_press",
    "drag",
    "swipe",
}


def sanitize_coordinate_actions(action_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove all coordinate-style arguments from actions like click / long_press.
    Keeps only semantic fields (e.g., index if already present).
    """
    action_type = action_dict.get("action_type", "").lower()

    if action_type in COORDINATE_ACTIONS:
        clean = {"action_type": action_type}

        # index 是唯一允许保留的定位方式
        if "index" in action_dict:
            clean["index"] = action_dict["index"]

        return clean

    # 非坐标型 action，原样返回
    return action_dict


def element_text(el):
    parts = []
    if el.get("text"):
        parts.append(el["text"])
    if el.get("content_description"):
        parts.append(el["content_description"])
    return " ".join(parts).lower()


def extract_keywords(thought):
    thought = thought.lower()
    tokens = re.findall(r"[a-z0-9\+]+", thought)

    stop = {
        "the", "a", "an", "to", "of", "and", "is", "are",
        "i", "need", "want", "will", "should", "can",
        "button", "screen", "corner"
    }
    return [t for t in tokens if t not in stop]


def parse_ui_dump_to_list(ui_dump: str):
    out = []
    for line in ui_dump.splitlines():
        m = re.search(r"(\{.*\})", line)
        if not m:
            continue
        try:
            obj = ast.literal_eval(m.group(1))
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            pass
    return out


def match_click_element(thought, ui_elements):
    keywords = extract_keywords(thought)

    best_score = 0
    best_index = None

    ui_elements = parse_ui_dump_to_list(ui_elements)
    # print(f"[DEBUG] UI elements list: {ui_elements}")

    for el in ui_elements:
        # print(f"[DEBUG] UI element: {el}")
        if not el.get("is_clickable", False):
            # print(f"[DEBUG] UI element is not clickable: {el}")
            continue
        # print(f"[DEBUG] UI element is clickable: {el}")

        text = element_text(el)
        if not text:
            continue

        score = 0
        for kw in keywords:
            if kw in text:
                score += 1

        # 重要：避免误点 account / header
        if "signed in" in text or "account" in text:
            score -= 2

        if score > best_score:
            best_score = score
            best_index = el["index"]

    if best_score == 0:
        return None

    print(f"[DEBUG] Thought: {thought}")
    print(f"[DEBUG] Selected Element: {ui_elements[best_index]}")
    return best_index
