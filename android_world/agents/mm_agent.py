"""
Element-based Text GUI Agent for AndroidWorld

Key idea:
- LLM does NOT output coordinates
- LLM selects UI element_id
- We directly map element_id → JSONAction(index=element_id)

This follows AndroidWorld text-agent design.
"""

from android_world.agents import base_agent
from android_world.agents import json_action
from android_world.agents import seeact_utils
from android_world.agents import t3a
from android_world.env import actuation, interface

from PIL import Image
import base64
import json
import os
import pprint
import traceback
import time


ELEMENT_SYS_PROMPT = """You are a GUI agent. You are given a task and your action history, 
with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<thinking>
...
</thinking>
<tool_call>
{"action": "mobile_use", "arguments": <args-json-object>}
</tool_call>

## Action Space

{"action": "click", "element_id": int} {"action": "long_press", "element_id": int} 
{"action": "type", "text": ""} 
{"action": "swipe", "direction": "up or down or left or right", "element_id": int} # "coordinate" is 
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

IMPORTANT:
- Do NOT output coordinates.
- You MUST choose element_id from the provided UI list.
"""


def format_ui_elements(ui_elements):
    """
    Convert AndroidWorld UI nodes into readable text for LLM.
    """
    lines = []
    for i, e in enumerate(ui_elements):
        text = getattr(e, "text", "") or ""
        desc = getattr(e, "content_desc", "") or ""
        rid = getattr(e, "resource_id", "") or ""
        cls = getattr(e, "class_name", "") or ""
        clickable = getattr(e, "clickable", False)

        lines.append(
            f"Element {i}: "
            f"text='{text}', desc='{desc}', class='{cls}', id='{rid}', clickable={clickable}"
        )
    return "\n".join(lines)


def image_to_data_url(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def parse_tool_call(response_text: str) -> dict:
    """
    Robust parser for model output.

    Accepts:
    1) <tool_call>{"name":"mobile_use","arguments":{...}}</tool_call>
    2) <tool_call>{"action":"click","element_id":3}</tool_call>   (shorthand)
    3) If tool_call tags are missing, fallback to the first {...} JSON block.

    Returns a normalized dict:
      {"name":"mobile_use","arguments":{...}}
    """
    import json
    import re
    from android_world.agents import seeact_utils

    if response_text is None:
        raise seeact_utils.ParseActionError("Empty response")

    s = str(response_text)

    # ------------- 1) Extract inside <tool_call> ... </tool_call> -------------
    block = None
    if "<tool_call>" in s and "</tool_call>" in s:
        try:
            block = s.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        except Exception:
            block = None

    # ------------- 2) Fallback: grab outermost JSON { ... } -------------
    if block is None:
        # Try to locate the first JSON object in the whole response
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            block = s[first:last + 1].strip()

    if block is None:
        raise seeact_utils.ParseActionError("tool_call block missing")

    # Remove ```json fences if any
    if block.startswith("```"):
        block = block.strip("`").strip()
        if block.startswith("json"):
            block = block[len("json"):].lstrip()

    # ------------- 3) json.loads with small repairs -------------
    def loads_with_repair(x: str) -> dict:
        x = x.strip()
        x = re.sub(r",\s*}", "}", x)
        x = re.sub(r",\s*]", "]", x)
        try:
            return json.loads(x)
        except Exception:
            # try to keep outermost braces
            a = x.find("{")
            b = x.rfind("}")
            if a != -1 and b != -1 and b > a:
                return json.loads(x[a:b + 1])
            raise

    try:
        obj = loads_with_repair(block)
    except Exception as e:
        raise seeact_utils.ParseActionError(f"Malformed tool_call JSON: {e}")

    # ------------- 4) Normalize schema -------------
    # Case A: already {"name":..., "arguments":...}
    if isinstance(obj, dict) and "arguments" in obj:
        # Some models omit name; set default
        if "name" not in obj:
            obj["name"] = "mobile_use"
        return obj

    # Case B: shorthand {"action": "...", ...}
    if isinstance(obj, dict) and "action" in obj:
        return {"name": "mobile_use", "arguments": obj}

    if isinstance(obj, dict) and "name" in obj and "element_id" in obj:
        return {
            "name": "mobile_use",
            "arguments": {
                "action": obj["name"],
                "element_id": obj["element_id"]
            }
        }

    raise seeact_utils.ParseActionError("Parsed JSON does not look like a tool_call")



def element_action_to_json_action(tool_call):
    """
    Convert element-based LLM output → AndroidWorld JSONAction
    """
    args = tool_call["arguments"]
    act = args["action"]

    # -------- CLICK --------
    if act == "click":
        return json_action.JSONAction(
            action_type=json_action.CLICK,
            index=args["element_id"]
        )

    # -------- LONG PRESS --------
    if act == "long_press":
        return json_action.JSONAction(
            action_type=json_action.LONG_PRESS,
            index=args["element_id"]
        )

    # -------- TYPE --------
    if act == "type":
        # Click field → then type
        return [
            json_action.JSONAction(
                action_type=json_action.CLICK,
                index=args["element_id"]
            ),
            json_action.JSONAction(
                action_type=json_action.TYPE,
                text=args.get("text", "")
            )
        ]

    # -------- SWIPE --------
    if act == "swipe":
        return json_action.JSONAction(
            action_type=json_action.SWIPE,
            direction=args.get("direction", "down"),
            index=args["element_id"]
        )

    # -------- OPEN APP --------
    if act == "open":
        return json_action.JSONAction(
            action_type=json_action.OPEN_APP,
            app_name=args.get("text", "")
        )

    # -------- SYSTEM BUTTON --------
    if act == "system_button":
        btn = args.get("button", "back")
        if btn == "back":
            return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)
        if btn == "home":
            return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME)

    # -------- WAIT --------
    if act == "wait":
        return json_action.JSONAction(action_type=json_action.WAIT)

    # -------- TERMINATE --------
    if act == "terminate":
        return json_action.JSONAction(action_type=json_action.STATUS)

    return json_action.JSONAction(action_type=json_action.UNKNOWN)


class ElementTextAgent(base_agent.EnvironmentInteractingAgent):
    """
    Element-based LLM GUI Agent for AndroidWorld.
    """

    def __init__(self, env: interface.AsyncEnv, vllm, name="ElementTextAgent"):
        super().__init__(env, name)
        self.vllm = vllm
        self.actions = []
        self.history = []

    def reset(self, go_home=False):
        super().reset(go_home)
        self.actions.clear()
        self.history.clear()

    def step(self, goal):

        state = self.get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size

        ui_elements = state.ui_elements
        ui_element_list = t3a._generate_ui_elements_description_list_full(
            ui_elements,
            logical_screen_size,
        )

        # print(f"[DEBUG] ui_elements {ui_element_list}")

        # Save screenshot
        screenshot = Image.fromarray(state.pixels)
        screenshot_path = "tmp_screen.png"
        screenshot.save(screenshot_path)

        # Build prompt
        history_text = "; ".join(self.history) if self.history else "None yet."
        # ui_text = format_ui_elements(ui_element_list)
        ui_text = str(ui_element_list)

        # print(f"[DEBUG] UI elements: {ui_text}")

        user_text = (
            f"Task: {goal}\n\n"
            f"History: {history_text}\n\n"
            f"UI Elements:\n{ui_text}"
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": ELEMENT_SYS_PROMPT}]},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_to_data_url(screenshot_path)}}
            ]}
        ]

        # Call model
        # ---------- LLM CALL ----------
        response, _, _ = self.vllm.predict_mm("", [], messages=messages)

        print("===== LLM RESPONSE =====")
        pprint.pprint(response)

        tool_call = None
        action = json_action.JSONAction(action_type=json_action.UNKNOWN)

        # ---------- PARSE ----------
        try:
            tool_call = parse_tool_call(response)
            action = element_action_to_json_action(tool_call)
        except Exception as e:
            print("Parse error:", str(e))

        # ---------- EXECUTE ----------
        try:
            if isinstance(action, list):
                for a in action:
                    self.env.execute_action(a)
            else:
                self.env.execute_action(action)
        except Exception as e:
            print("Execution error:", str(e))

        # ---------- RECORD HISTORY ONLY IF PARSED ----------
        if tool_call is not None:
            self.actions.append(tool_call)
            self.history.append(str(tool_call))

        # ---------- DONE CHECK ----------
        done = (
            action.action_type == json_action.STATUS
            if isinstance(action, json_action.JSONAction)
            else False
        )

        return base_agent.AgentInteractionResult(
            done=done,
            data={"action": str(action)}
        )




