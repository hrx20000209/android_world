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

"""GELAB agent variant with configurable screenshot resize for model input."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import Any

from PIL import Image

from android_world.agents import base_agent
from android_world.agents import gelab_agent
from android_world.agents import seeact_utils
from android_world.env import interface
from android_world.env import json_action


def _resize_for_model_input(image: Image.Image, scale: float) -> Image.Image:
  """Resize screenshot by a divisor scale for model input."""
  safe_scale = max(1.0, float(scale))
  if safe_scale <= 1.0:
    return image
  width, height = image.size
  target_width = max(1, int(round(width / safe_scale)))
  target_height = max(1, int(round(height / safe_scale)))
  if target_width == width and target_height == height:
    return image
  if hasattr(Image, "Resampling"):
    resample = Image.Resampling.BILINEAR
  else:
    resample = Image.BILINEAR
  return image.resize((target_width, target_height), resample=resample)


class GELABResizeAgent(gelab_agent.GELABAgent):
  """GELAB agent with image downsample support and per-step resolution logs."""

  def __init__(
      self,
      env: interface.AsyncEnv,
      vllm: Any,
      name: str = "GELABResizeAgent",
      output_path: str = "",
      history_limit: int = 8,
      image_downsample_scale: float = 2.0,
  ):
    super().__init__(
        env=env,
        vllm=vllm,
        name=name,
        output_path=output_path,
        history_limit=history_limit,
    )
    self.image_downsample_scale = max(1.0, float(image_downsample_scale))

  def _build_model_screenshot(self, screenshot: Image.Image) -> tuple[Image.Image, tuple[int, int], tuple[int, int]]:
    original_size = screenshot.size
    model_screenshot = _resize_for_model_input(screenshot, self.image_downsample_scale)
    resized_size = model_screenshot.size
    return model_screenshot, original_size, resized_size

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
    model_screenshot, original_size, resized_size = self._build_model_screenshot(screenshot)
    screen_size = self.env.logical_screen_size
    hints: list[str] = []
    history = self._history_text()
    gelab_agent._print_step_section(  # pylint: disable=protected-access
        step_idx,
        "Resolution",
        (
            f"original={original_size[0]}x{original_size[1]}, "
            f"model={resized_size[0]}x{resized_size[1]}, "
            f"image_downsample_scale={self.image_downsample_scale:.3f}"
        ),
    )
    messages = gelab_agent.build_gelab_messages(goal, history, model_screenshot)
    message_text = gelab_agent._messages_text_for_logging(messages)  # pylint: disable=protected-access
    if message_text:
      gelab_agent._print_step_section(step_idx, "Model input", message_text)  # pylint: disable=protected-access

    response, _, _ = self.vllm.predict_mm("", [], messages=messages)
    gelab_agent._print_step_section(step_idx, "Model output", str(response))  # pylint: disable=protected-access
    parse_error = None
    try:
      parsed_action = gelab_agent.parse_gelab_response(response)
      action, tool_call, extras = gelab_agent.gelab_action_to_json_action(parsed_action, screen_size)
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
      gelab_agent._print_step_section(step_idx, "Parse fallback", parse_error)  # pylint: disable=protected-access
    gelab_agent._print_step_section(step_idx, "Parsed action", gelab_agent._json_dumps_safe(dict(parsed_action)))  # pylint: disable=protected-access
    gelab_agent._print_step_section(step_idx, "Tool call", gelab_agent._json_dumps_safe(tool_call))  # pylint: disable=protected-access

    if extras.get("return_text"):
      self.env.interaction_cache = str(extras["return_text"])

    self._execute_action(action, extras)
    gelab_agent._print_step_section(step_idx, "Action", gelab_agent._json_dumps_safe(action.__dict__))  # pylint: disable=protected-access
    if extras:
      gelab_agent._print_step_section(step_idx, "Action extras", gelab_agent._json_dumps_safe(extras))  # pylint: disable=protected-access

    summary = gelab_agent._normalize_space(parsed_action.get("summary")) or gelab_agent._normalize_space(parsed_action.get("explain"))  # pylint: disable=protected-access
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
        "model_input_resolution": {"width": resized_size[0], "height": resized_size[1]},
        "original_resolution": {"width": original_size[0], "height": original_size[1]},
        "image_downsample_scale": self.image_downsample_scale,
    }
    self._actions.append(step_record)
    self._summaries.append(summary)
    self._responses.append(str(response))

    task_dir = self._task_output_dir(goal)
    if task_dir:
      screenshot.save(os.path.join(task_dir, f"screenshot_{len(self._actions) - 1}.png"))
      model_screenshot.save(os.path.join(task_dir, f"screenshot_model_input_{len(self._actions) - 1}.png"))
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
            "model_input_resolution": {"width": resized_size[0], "height": resized_size[1]},
            "original_resolution": {"width": original_size[0], "height": original_size[1]},
            "image_downsample_scale": self.image_downsample_scale,
        },
    )


class GELABAgent(GELABResizeAgent):
  """Compatibility alias so run.py can instantiate GELABAgent from this module."""

  pass
