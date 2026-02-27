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
MAX_AGENT_STEPS = 40

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
{"action": "type", "text": "", "coordinate": [x, y]}
{"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]} # "coordinate" is optional. Use the "coordinate" if you want to swipe a specific UI element.
{"action": "open", "text": "app_name"}
{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}
{"action": "system_button", "button": "button_name"} # Options: back, home, menu, enter
{"action": "wait"}
{"action": "terminate", "status": "success or fail"}
{"action": "answer", "text": "xxx"} # Use escape characters \\\', \\", and \\n in text part to ensure we can parse the text in normal python string format.

Fallback only when exact coordinates are unavailable:
{"action": "click", "element_id": 3}
{"action": "long_press", "element_id": 3}
{"action": "type", "text": "xxx", "element_id": 3}

## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part.
- For click/long_press/type, prefer coordinate-based actions. Do not invent keys like button/label/target for these actions.
- If a coordinate is not reliable, use one of the provided element_id values.
- Available Apps: `""" + json.dumps(AVAILABLE_APPS, ensure_ascii=True) + """`.
You should use the `open` action to open the app as possible as you can, because it is the fast way to open the app.
- Use the `answer` action for question-answer tasks that only require returning a final value.
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


def safe_json_loads(s: str) -> dict[str, Any]:
    """Robust JSON parsing for model tool-call output."""
    s = str(s or "").strip()

    try:
        return json.loads(s)
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()

    try:
        return json.loads(s)
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    s = re.sub(r"\}\s*\}\s*\]", "]]", s)
    s = re.sub(r"\]\s*\}+", "]", s)

    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = s[first:last + 1]
        try:
            return json.loads(candidate)
        except Exception:  # pylint: disable=broad-exception-caught
            s = candidate

    def extract(pattern: str, default: str | None = None) -> str | None:
        m = re.search(pattern, s)
        return m.group(1) if m else default

    def extract_coord(key: str | None = None) -> list[int]:
        if key is not None:
            m = re.search(rf'"{key}"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)[^\]]*\]', s)
            if m:
                return [int(m.group(1)), int(m.group(2))]
        m = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)", s)
        if m:
            return [int(m.group(1)), int(m.group(2))]
        return [-1, -1]

    name = extract(r'"name"\s*:\s*"([^"]+)"', "mobile_use")
    action = extract(r'"action"\s*:\s*"([^"]+)"', "click")
    action_low = str(action or "click").strip().lower()
    text = extract(r'"text"\s*:\s*"([^"]*)"')
    content = extract(r'"content"\s*:\s*"([^"]*)"')
    value = extract(r'"value"\s*:\s*"([^"]*)"')
    return_text = extract(r'"return"\s*:\s*"([^"]*)"')
    button = extract(r'"button"\s*:\s*"([^"]+)"')
    direction = extract(r'"direction"\s*:\s*"([^"]+)"')

    args: dict[str, Any] = {"action": action}

    if action_low in ("click", "long_press", "tap"):
        coord = extract_coord("coordinate")
        if coord != [-1, -1]:
            args["coordinate"] = coord
    elif action_low == "type":
        args["text"] = text or value or ""
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
    elif action_low == "system_button":
        args["button"] = button or "back"
    elif action_low == "back":
        args["action"] = "system_button"
        args["button"] = "back"
    elif action_low in ("answer", "respond", "response", "reply", "read"):
        args["action"] = "answer"
        args["text"] = text or content or value or return_text or ""
    elif action_low == "terminate":
        status = extract(r'"status"\s*:\s*"([^"]+)"', "fail")
        args["status"] = status
    else:
        args["action"] = "wait"

    fixed_obj = {"name": name, "arguments": args}
    print("[safe_json_loads] recovered ->", fixed_obj)
    return fixed_obj


def parse_tagged_text(text: str) -> dict[str, Any]:
    """Parse <thinking> and <tool_call> tags."""
    text = str(text or "")

    if "</think>" in text and "</thinking>" not in text:
        text = text.replace("</think>", "</thinking>")
        if "<thinking>" not in text:
            text = "<thinking>" + text

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
        result["tool_call"] = None

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
    image_ele = update_image_size_(
        {
            "image": screenshot_file,
            "width": max(1, int(width // scale)),
            "height": max(1, int(height // scale)),
        }
    )
    resized_width = int(image_ele["resized_width"])
    resized_height = int(image_ele["resized_height"])
    screenshot = screenshot.resize((resized_width, resized_height))
    return screenshot, resized_width, resized_height, image_ele


def _safe_get_action_type(action_dict: dict[str, Any] | None) -> str | None:
    if not isinstance(action_dict, dict):
        return None
    arguments = action_dict.get("arguments")
    if isinstance(arguments, dict) and "action" in arguments:
        return str(arguments.get("action"))
    if "name" in action_dict:
        return str(action_dict.get("name"))
    return None


def _safe_get_text(action_dict: dict[str, Any] | None) -> str:
    if not isinstance(action_dict, dict):
        return ""
    arguments = action_dict.get("arguments")
    if not isinstance(arguments, dict):
        return ""
    return str(arguments.get("text") or arguments.get("content") or "")


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
    system_msg = {
        "role": "system",
        "content": [{"type": "text", "text": MAI_MOBILE_SYS_PROMPT}],
    }
    user_text = (
        f"Task:\n{goal}\n\n"
        f"Action History:\n{history if history else 'None yet.'}\n\n"
        f"Visible UI elements:\n{ui_element_text}\n\n"
        "Now output the next action as a JSON function call for `mobile_use` inside <tool_call></tool_call>."
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
    ):
        super().__init__(env, name)
        self.vllm = vllm
        self.src_format = str(src_format or "qwen-vl")
        self.api_key = api_key
        self.url = url
        self.output_path = str(output_path or "").strip()

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
        lines = []
        for idx, summary in enumerate(self._summarys, start=1):
            lines.append(f"Step {idx}: {summary};")
        return " ".join(lines)

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

        scale = 1
        resized_image, resized_width, resized_height, image_ele = fetch_resized_image(screenshot_file, scale)
        resized_screenshot_file = re.sub(r"screenshot_(\d+)\.png$", r"screenshot_resized_\1.png", screenshot_file)
        if resized_screenshot_file == screenshot_file:
            resized_screenshot_file = screenshot_file.replace(".png", f"_resized_scale{scale}.png")
        resized_image.save(resized_screenshot_file)

        history_text = self._history_text()
        ui_element_text = _format_ui_element_list(
            state.ui_elements,
            image_ele,
            coordinate_format=self.src_format,
            limit=12,
        )
        messages = build_mai_messages(goal, history_text, resized_screenshot_file, ui_element_text)

        print(f"[DEBUG] resized screenshot saved to: {resized_screenshot_file}, size: {(resized_width, resized_height)}")
        print(f"[DEBUG] Messages: {mask_image_urls_for_logging(messages)}")

        action_response, _, _ = self.vllm.predict_mm("", [], messages=messages)
        result["action_response"] = action_response
        print("========== MAI action_response ==========")
        print(action_response)

        summary = "wait"
        thought = ""

        try:
            parsed = parse_tagged_text(action_response)
            thought = parsed.get("thinking") or ""
            tool_call = parsed.get("tool_call") or parse_mai_tool_call(action_response)
            dummy_action = tool_call

            last_action = self._actions[-1] if self._actions else None
            if _safe_get_action_type(last_action) == "answer":
                dummy_action = {
                    "name": "mobile_use",
                    "arguments": {"action": "terminate", "status": "success"},
                }
                answer_text = _safe_get_text(last_action)
                if answer_text:
                    self.env.interaction_cache = answer_text

            action, dummy_action_translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
                dummy_action,
                image_ele,
                src_format=self.src_format,
                tgt_format="abs_origin",
                scale=scale,
                ui_elements=state.ui_elements,
            )

            if action.action_type == json_action.ANSWER and action.text:
                self.env.interaction_cache = action.text

            actuation.execute_adb_action(
                action,
                state.ui_elements,
                self.env.logical_screen_size,
                self.env.controller,
            )

            result["dummy_action"] = dummy_action
            result["dummy_action_translated"] = dummy_action_translated
            result["action"] = action
            summary = _summarize_action(dummy_action.get("arguments", {}))

        except (
            seeact_utils.ParseActionError,
            ValueError,
            KeyError,
            NotImplementedError,
            json.JSONDecodeError,
        ) as exc:
            print("Failed to parse/normalize MAI tool_call:", exc)
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
            summary = "wait"

        except Exception:  # pylint: disable=broad-exception-caught
            traceback.print_exc()
            print(action_response)
            raise

        self._text_actions.append(summary)
        self._summarys.append(summary)
        self._thoughts.append(thought)
        self._response.append(str(action_response))
        self._actions.append(
            {
                "name": "mobile_use",
                "arguments": (result["dummy_action"] or {}).get("arguments", {}),
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

        done = bool(isinstance(action_obj, json_action.JSONAction) and action_obj.action_type == json_action.STATUS)
        return base_agent.AgentInteractionResult(done=done, data=result)
