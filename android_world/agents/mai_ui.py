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

"""MAI-UI-2B agent for AndroidWorld."""
from android_world.agents import base_agent
from android_world.agents import seeact_utils
from android_world.agents import mobile_agent_utils
from android_world.env import actuation
from android_world.env import interface
from android_world.env import adb_utils
from android_world.env import tools
from android_world.agents import json_action

from PIL import Image
from io import BytesIO
import base64
import json
import pprint
import os
import re
import time
import traceback

from android_world.agents.coordinate_resize import update_image_size_


###############################################################################
# Helper and prompt
###############################################################################
# ---------- helper: safe extraction ----------
def _safe_get_last_action(self):
    """Return (last_action_dict or None)."""
    if not getattr(self, "_actions", None):
        return None
    if len(self._actions) == 0:
        return None
    last = self._actions[-1]
    return last if isinstance(last, dict) else None


def _safe_get_action_type(action_dict):
    """
    Try to get action type from different schemas:
    1) {"name":"mobile_use","arguments":{"action":"click"...}}
    2) {"arguments":{"action":"click"...}}
    3) {"name":"click", "arguments": {...}}   (non-mai schema)
    """
    if not isinstance(action_dict, dict):
        return None

    args = action_dict.get("arguments", {})
    if isinstance(args, dict) and "action" in args:
        return args.get("action")

    # fallback: sometimes action type is stored in name
    # e.g. {"name":"open","arguments":{"app_name":"Clock"}}
    return action_dict.get("name")


def _safe_get_coordinate(action_dict):
    """Return coordinate list [x, y] if exists and valid else None."""
    if not isinstance(action_dict, dict):
        return None
    args = action_dict.get("arguments", {})
    if not isinstance(args, dict):
        return None
    xy = args.get("coordinate")
    if (
        isinstance(xy, (list, tuple))
        and len(xy) == 2
        and isinstance(xy[0], (int, float))
        and isinstance(xy[1], (int, float))
    ):
        # reject [-1,-1] or other invalid sentinel
        if xy[0] < 0 or xy[1] < 0:
            return None
        return [int(xy[0]), int(xy[1])]
    return None


def _safe_get_text(action_dict):
    """Return text if exists."""
    if not isinstance(action_dict, dict):
        return ""
    args = action_dict.get("arguments", {})
    if isinstance(args, dict):
        return args.get("text", "") or ""
    return ""


def safe_json_loads(s: str):
    import json, re

    s = s.strip()

    # ---------- ① 去掉 ```json 包裹 ----------
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.startswith("json"):
            s = s[len("json"):].lstrip()

    # ---------- ② 常见畸形修复 ----------
    # 例如 ]}} , }}] , 多余右括号
    s = re.sub(r'\}\s*\}\s*\]', ']]', s)
    s = re.sub(r'\]\s*\}+', ']', s)

    # ---------- ③ 截取最外层 { ... } ----------
    first = s.find("{")
    last  = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        s = s[first:last+1]

    # ---------- ④ 先尝试正常解析 ----------
    try:
        return json.loads(s)
    except Exception:
        pass  # 进入抢救模式

    # ===================================================
    #   进入“容错恢复模式”（即使 JSON 坏也尽量提取字段）
    # ===================================================

    def extract(pattern, default=None):
        m = re.search(pattern, s)
        return m.group(1) if m else default

    def extract_coord(key=None):
        # 优先匹配 action 字段对应坐标
        if key is not None:
            m = re.search(
                rf'"{key}"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)[^\]]*\]',
                s
            )
            if m:
                return [int(m.group(1)), int(m.group(2))]

        # 宽松兜底：抓前两个数字即可
        m = re.search(r'\[\s*(-?\d+)\s*,\s*(-?\d+)', s)
        if m:
            return [int(m.group(1)), int(m.group(2))]

        return [-1, -1]  # fallback，不算有效坐标

    name  = extract(r'"name"\s*:\s*"([^"]+)"', "mobile_use")
    act   = extract(r'"action"\s*:\s*"([^"]+)"', "click")
    text  = extract(r'"text"\s*:\s*"([^"]*)"')
    button = extract(r'"button"\s*:\s*"([^"]+)"')
    direction = extract(r'"direction"\s*:\s*"([^"]+)"')

    args = {"action": act}

    # ---------- click / long_press ----------
    if act in ("click", "long_press"):
        args["coordinate"] = extract_coord("coordinate")

    # ---------- type ----------
    elif act == "type":
        args["text"] = text or ""

    # ---------- swipe ----------
    elif act == "swipe":
        args["direction"] = direction or "down"
        coord = extract_coord("coordinate")
        if coord != [-1, -1]:
            args["coordinate"] = coord   # coordinate 是可选

    # ---------- drag ----------
    elif act == "drag":
        args["start_coordinate"] = extract_coord("start_coordinate")
        args["end_coordinate"] = extract_coord("end_coordinate")

    # ---------- system button ----------
    elif act == "system_button":
        args["button"] = button or "back"

    # ---------- answer ----------
    elif act == "answer":
        args["text"] = text or ""

    # ---------- terminate ----------
    elif act == "terminate":
        status = extract(r'"status"\s*:\s*"([^"]+)"', "fail")
        args["status"] = status

    # ---------- wait / fallback ----------
    else:
        # 保底 action
        args.setdefault("action", "wait")

    fixed_obj = {"name": name, "arguments": args}

    print("[safe_json_loads] recovered →", fixed_obj)
    return fixed_obj


def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def fetch_resized_image(screenshot_file):
    screenshot = Image.open(screenshot_file)
    width, height = screenshot.size
    current_image_ele = update_image_size_(
        {"image": screenshot_file, "width": width, "height": height}
    )
    resized_width = current_image_ele["resized_width"]
    resized_height = current_image_ele["resized_height"]
    screenshot = screenshot.resize((resized_width, resized_height))
    return screenshot, resized_width, resized_height, current_image_ele


MAI_MOBILE_SYS_PROMPT_NO_THINKING = """You are a GUI agent. You are given a task and your action history, 
with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "mobile_use", "arguments": <args-json-object>}
</tool_call>

## Action Space

{"action": "click", "coordinate": [x, y]} {"action": "long_press", "coordinate": [x, y]} 
{"action": "type", "text": ""} 
{"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]} # "coordinate" is 
optional. Use the "coordinate" if you want to swipe a specific UI element. 
{"action": "open", "text": "app_name"} 
{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]} 
{"action": "system_button", "button": "button_name"} # Options: back, home, menu, enter {"action": "wait"} 
{"action": "terminate", "status": "success or fail"} 
{"action": "answer", "text": "xxx"} 
# Use escape characters \\\', \\", and \\n in text part to ensure we can parse the text in normal python string format.


## Note
- Available Apps: `["Camera","Chrome","Clock","Contacts","Dialer","Files","Settings","Markor","Tasks","Simple Draw Pro","Simple Gallery Pro","Simple SMS Messenger","Audio Recorder","Pro Expense","Broccoli APP","OSMand","VLC","Joplin","Retro Music","OpenTracks","Simple Calendar Pro"]`.
You should use the `open` action to open the app as possible as you can, because it is the fast way to open the app.
- You must follow the Action Space strictly, and return the correct json object within <tool_call></tool_call> XML tags.
""".strip()


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


## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part.
- Available Apps: `["Camera","Chrome","Clock","Contacts","Dialer","Files","Settings","Markor","Tasks","Simple Draw Pro","Simple Gallery Pro","Simple SMS Messenger","Audio Recorder","Pro Expense","Broccoli APP","OSMand","VLC","Joplin","Retro Music","OpenTracks","Simple Calendar Pro"]`.
You should use the `open` action to open the app as possible as you can, because it is the fast way to open the app.
- You must follow the Action Space strictly, and return the correct json object within <thinking> </thinking> and <tool_call></tool_call> XML tags.
""".strip()


def image_to_data_url(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def build_mai_messages(goal, history, screenshot_path):
    """Build messages for MAI-UI-2B."""
    system_msg = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": MAI_MOBILE_SYS_PROMPT,
            }
        ],
    }

    user_text = (
        f"Task:\n{goal}\n\n"
        f"Action History:\n{history if history else 'None yet.'}\n\n"
        "Now output the next action as a JSON function call for `mobile_use` "
        "inside <tool_call></tool_call>."
    )

    print(f"[DEBUG] System Message: {system_msg}")
    print(f"[DEBUG] User Text: {user_text}")

    user_msg = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_text,
            },
            # Many vLLM wrappers accept this simple image spec; adjust if needed.
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(screenshot_path)
                },
            },
        ],
    }
    return [system_msg, user_msg]


def parse_mai_tool_call(response_text: str) -> dict:
    """Parse <tool_call>...</tool_call> and return the JSON dict."""
    if "<tool_call" not in response_text:
        # Fallback: try to find first {...} that looks like the call
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise seeact_utils.ParseActionError("Cannot find tool_call JSON in response.")
        return json.loads(response_text[start : end + 1])

    try:
        tool_block = response_text.split("<tool_call>")[1].split("</tool_call>")[0]
    except Exception as e:  # noqa: BLE001
        raise seeact_utils.ParseActionError(f"Malformed tool_call block: {e}")

    tool_block = tool_block.strip()
    # Some models wrap JSON in code fences
    if tool_block.startswith("```"):
        tool_block = tool_block.strip("`").strip()
        # Remove possible `json` language tag
        if tool_block.startswith("json"):
            tool_block = tool_block[len("json"):].lstrip()
    return safe_json_loads(tool_block)


###############################################################################
# Agent
###############################################################################


class MAIUIAgent(base_agent.EnvironmentInteractingAgent):
    """MAI-UI-2B based mobile agent for Android."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        vllm,
        src_format,
        api_key=None,
        url=None,
        name: str = "MAIUIAgent",
        output_path: str = "",
    ):
        super().__init__(env, name)
        self._actions = []
        self._screenshots = []
        self._summarys = []
        self._thoughts = []  # kept for compatibility, not used
        self._response = []
        self._text_actions = []
        self.output_result = {}
        self.output_path = output_path
        if self.output_path and not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.vllm = vllm
        self.src_format = src_format
        self.url = url
        self.api_key = api_key
        self.output_list = []
        self.task_name = {}

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self.env.hide_automation_ui()
        self._actions.clear()
        self._text_actions.clear()
        self._screenshots.clear()
        self._summarys.clear()
        self._thoughts.clear()
        self._response.clear()

    def initialize_chrome(self):
        print("Running additional chrome initialization...")
        adb_utils.launch_app("chrome", self.env.controller)
        time.sleep(5)

        tool_controller = tools.AndroidToolController(env=self.env.controller)
        time.sleep(2)

        first_op = False
        try:
            print("try first variant...")
            tool_controller.click_element("Use without an account")
            time.sleep(5.0)
            first_op = True
        except Exception:
            print("Failed to click 'Use without an account' button.")

        if not first_op:
            print("try second variant...")
            try:
                tool_controller.click_element("Accept & continue")
            except Exception:
                pass
            time.sleep(3.0)
            try:
                tool_controller.click_element("No thanks")
            except Exception:
                pass
            time.sleep(5.0)

        adb_utils.press_home_button(self.env.controller)
        time.sleep(2.0)
        print("Done additional chrome initialization")

    def get_task_name(self, suite):
        for name, instances in suite.items():
            self.task_name[instances[0].goal] = name

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        result = {
            "ui_elements": None,
            "screenshot": None,
            "actionable_elements": None,
            "action_gen_payload": None,
            "action_gen_response": None,
            "action_ground_payload": None,
            "action_ground_response": None,
            "seeact_action": None,
            "action": None,
            "action_description": None,
        }

        step_idx = len(self._screenshots)
        state = self.get_post_transition_state()
        result["ui_elements"] = state.ui_elements

        xml_tree = result["ui_elements"]

        result["screenshot"] = state.pixels
        screenshot = Image.fromarray(state.pixels)
        screenshot_file = f"screenshot_{step_idx}.png"

        if self.output_path:
            if goal not in self.task_name:
                task_output_dir = os.path.join(
                    self.output_path, goal.replace(" ", "_")[:50]
                )
            else:
                task_output_dir = os.path.join(self.output_path, self.task_name[goal])
            screenshot_file = os.path.join(
                task_output_dir, f"screenshot_{step_idx}.png"
            )
            if not os.path.exists(task_output_dir):
                os.mkdir(task_output_dir)
            screenshot.save(screenshot_file)
            with open(
                os.path.join(task_output_dir, "action.jsonl"),
                "w",
                encoding="utf-8",
            ) as f:
                for item in self._actions:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + "\n")
        self._screenshots.append(screenshot)

        # Build concise textual history
        stage2_history = ""
        for idx, his in enumerate(self._summarys):
            if his is not None:
                stage2_history += (
                    "Step "
                    + str(idx + 1)
                    + ": "
                    + str(his.replace("\n", "").replace('"', ""))
                    + "; "
                )

        screenshot, resized_width, resized_height, current_image_ele = fetch_resized_image(
            screenshot_file
        )

        print("[DEBUG] resize→abs mapping test", resized_width, resized_height, current_image_ele)

        # Build messages for MAI-UI-2B
        messages = build_mai_messages(goal, stage2_history, screenshot_file)

        # Call model
        action_response, _, _ = self.vllm.predict_mm(
            "",
            [],
            messages=messages,
        )

        result["action_response"] = action_response
        print("========== MAI action_response ==========")
        pprint.pprint(action_response)

        action = None
        dummy_action = None
        summary = None  # MAI prompt没有显式summary，这里留空

        try:
            # Parse <tool_call> JSON
            tool_call = parse_mai_tool_call(action_response)

            # Expect {"name": "mobile_use", "arguments": {...}}
            dummy_action = tool_call

            last_action = _safe_get_last_action(self)
            last_type = _safe_get_action_type(last_action)

            if last_type == "answer":
                dummy_action = {
                    "name": "mobile_use",
                    "arguments": {"action": "terminate", "status": "success"},
                }
                # cache answer text safely
                self.env.interaction_cache = _safe_get_text(last_action)

            # 将 MAI 的坐标空间转换成 json_action 所需格式
            action, dummy_action_translated = (
                mobile_agent_utils.convert_mobile_agent_action_to_json_action(
                    dummy_action,
                    current_image_ele,
                    src_format=self.src_format,
                    tgt_format="abs_origin",
                )
            )

            # === Detect repeated CLICK at the same coordinate ===
            # last_action = _safe_get_last_action(self)
            # prev_type = _safe_get_action_type(last_action)
            # curr_type = _safe_get_action_type(dummy_action)

            # if prev_type == "click" and curr_type == "click":
            #     prev_xy = _safe_get_coordinate(last_action)
            #     curr_xy = _safe_get_coordinate(dummy_action)

                # if prev_xy is not None and curr_xy is not None and prev_xy == curr_xy:
                #     print(f"[STOP] Repeated CLICK at {curr_xy}, terminating task.")
                #     dummy_action = {
                #         "name": "mobile_use",
                #         "arguments": {"action": "status", "goal_status": "complete"},
                #     }
                #     action = json_action.JSONAction(json_action.STATUS)

            result["dummy_action"] = dummy_action
            result["dummy_action_translated"] = dummy_action_translated
            result["action"] = action

        except seeact_utils.ParseActionError as e:
            print("Failed to parse MAI tool_call:", e)
            action = json_action.JSONAction(action_type=json_action.UNKNOWN)
            result["seeact_action"] = None
            result["action"] = action
        except Exception:
            traceback.print_exc()
            print(action_response)
            raise
        else:
            # 真正执行 adb 动作
            actuation.execute_adb_action(
                action,
                [],
                self.env.logical_screen_size,
                self.env.controller,
            )

            # MAI 没有显式 thought/summary，这里把 action_response 当成粗略 summary
            try:
                args = dummy_action.get("arguments", {})
                act = args.get("action", "")

                if act == "click":
                    summary = f'click at {args.get("coordinate")}'
                elif act == "long_press":
                    summary = f'long_press at {args.get("coordinate")}'
                elif act == "type":
                    summary = f'type "{args.get("text", "")[:20]}"'
                elif act == "swipe":
                    summary = f'swipe {args.get("direction", "")}'
                elif act == "open":
                    summary = f'open "{args.get("text", "")}"'
                elif act == "system_button":
                    summary = f'system_button {args.get("button", "")}'
                elif act == "terminate":
                    summary = f'terminate({args.get("status", "")})'
                else:
                    summary = str(dummy_action)
            except Exception:
                summary = str(dummy_action)

            self._text_actions.append(summary)
            self._actions.append(dummy_action)
            self._summarys.append(summary)
            self._thoughts.append(None)
            self._response.append(action_response)

        if self.output_path:
            if goal not in self.task_name:
                task_output_dir = os.path.join(
                    self.output_path, goal.replace(" ", "_")[:50]
                )
            else:
                task_output_dir = os.path.join(self.output_path, self.task_name[goal])
            if not os.path.exists(task_output_dir):
                os.mkdir(task_output_dir)
            screenshot.save(screenshot_file)
            with open(
                os.path.join(task_output_dir, "action.jsonl"),
                "w",
                encoding="utf-8",
            ) as f:
                for item in self._actions:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + "\n")

        return base_agent.AgentInteractionResult(
            done=action.action_type == json_action.STATUS,
            data=result,
        )
