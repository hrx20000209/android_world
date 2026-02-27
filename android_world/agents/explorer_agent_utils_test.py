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

from absl.testing import absltest

from android_world.agents import explorer_agent_utils
from android_world.env import json_action


class _BBox:

  def __init__(self, x_min: int, y_min: int, x_max: int, y_max: int):
    self.x_min = x_min
    self.y_min = y_min
    self.x_max = x_max
    self.y_max = y_max


class _Element:

  def __init__(self, *, bbox: _BBox):
    self.is_clickable = True
    self.is_long_clickable = False
    self.is_editable = False
    self.is_scrollable = False
    self.bbox_pixels = bbox


class CoordinateModeTest(absltest.TestCase):

  def test_parse_tool_call_accepts_open_app_shorthand(self):
    tool_call = explorer_agent_utils.parse_tool_call(
        '<tool_call>\n{"action":"open_app","text":"Audio Recorder"}\n</tool_call>'
    )

    self.assertEqual(
        tool_call,
        {
            "name": "mobile_use",
            "arguments": {"action": "open_app", "text": "Audio Recorder"},
        },
    )

    action = explorer_agent_utils._to_json_action(
        tool_call=tool_call,
        ui_elements=[],
        logical_screen_size=(1080, 2400),
        coordinate_mode="auto",
    )
    self.assertEqual(action.action_type, json_action.OPEN_APP)
    self.assertEqual(action.app_name, "Audio Recorder")

  def test_auto_prefers_absolute_coordinate_when_in_screen(self):
    result = explorer_agent_utils._scale_coordinate_by_mode(
        coordinate=(500, 902),
        screen_size=(1080, 2400),
        mode="auto",
    )
    self.assertEqual(result, (500, 902))

  def test_auto_uses_ratio_1_for_normalized_values(self):
    result = explorer_agent_utils._scale_coordinate_by_mode(
        coordinate=(0.5, 0.5),
        screen_size=(1080, 2400),
        mode="auto",
    )
    self.assertEqual(result, (540, 1200))

  def test_auto_uses_ratio_1000_when_absolute_is_out_of_bounds(self):
    result = explorer_agent_utils._scale_coordinate_by_mode(
        coordinate=(900, 900),
        screen_size=(720, 1280),
        mode="auto",
    )
    self.assertEqual(result, (648, 1152))

  def test_explicit_1000_mode_alias(self):
    result = explorer_agent_utils._scale_coordinate_by_mode(
        coordinate=(500, 900),
        screen_size=(1080, 2400),
        mode="1000",
    )
    self.assertEqual(result, (540, 2160))

  def test_to_json_action_auto_prefers_1000_if_closer_to_interactive_target(self):
    action = explorer_agent_utils._to_json_action(
        tool_call={"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 902]}},
        ui_elements=[_Element(bbox=_BBox(460, 2090, 620, 2230))],
        logical_screen_size=(1080, 2400),
        coordinate_mode="auto",
    )
    self.assertEqual(action.action_type, json_action.CLICK)
    self.assertEqual((action.x, action.y), (540, 2165))

  def test_to_json_action_auto_keeps_absolute_if_closer_to_interactive_target(self):
    action = explorer_agent_utils._to_json_action(
        tool_call={"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 902]}},
        ui_elements=[_Element(bbox=_BBox(430, 840, 600, 980))],
        logical_screen_size=(1080, 2400),
        coordinate_mode="auto",
    )
    self.assertEqual(action.action_type, json_action.CLICK)
    self.assertEqual((action.x, action.y), (500, 902))


if __name__ == "__main__":
  absltest.main()
