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

from android_world.agents import mobile_agent_utils
from android_world.env import json_action


class _BBox:

  def __init__(self, x_min, y_min, x_max, y_max):
    self.x_min = x_min
    self.y_min = y_min
    self.x_max = x_max
    self.y_max = y_max


class _Element:

  def __init__(self, bbox):
    self.bbox_pixels = bbox


class MobileAgentUtilsConversionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.img_ele = {
        "width": 1000,
        "height": 2000,
        "resized_width": 1000,
        "resized_height": 2000,
    }

  def test_click_uses_point_alias(self):
    action, translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
        dummy_action={
            "name": "mobile_use",
            "arguments": {"action": "CLICK", "point": [500, 500]},
        },
        img_ele=self.img_ele,
        src_format="abs_origin",
        tgt_format="abs_origin",
    )

    self.assertEqual(action.action_type, json_action.CLICK)
    self.assertEqual((action.x, action.y), (500, 500))
    self.assertEqual(translated["arguments"]["coordinate"], [500, 500])

  def test_longpress_uses_uppercase_action(self):
    action, translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
        dummy_action={
            "name": "mobile_use",
            "arguments": {"action": "LONGPRESS", "coordinate": [200, 300]},
        },
        img_ele=self.img_ele,
        src_format="abs_origin",
        tgt_format="abs_origin",
    )

    self.assertEqual(action.action_type, json_action.LONG_PRESS)
    self.assertEqual((action.x, action.y), (200, 300))
    self.assertEqual(translated["arguments"]["coordinate"], [200, 300])

  def test_type_accepts_value_and_point(self):
    action, translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
        dummy_action={
            "name": "mobile_use",
            "arguments": {"action": "TYPE", "value": "hello", "point": [100, 400]},
        },
        img_ele=self.img_ele,
        src_format="abs_origin",
        tgt_format="abs_origin",
    )

    self.assertEqual(action.action_type, json_action.INPUT_TEXT)
    self.assertEqual(action.text, "hello")
    self.assertEqual((action.x, action.y), (100, 400))
    self.assertEqual(translated["arguments"]["coordinate"], [100, 400])

  def test_awake_maps_to_open_app(self):
    action, _ = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
        dummy_action={
            "name": "mobile_use",
            "arguments": {"action": "AWAKE", "value": "Audio Recorder"},
        },
        img_ele=self.img_ele,
        src_format="abs_origin",
        tgt_format="abs_origin",
    )

    self.assertEqual(action.action_type, json_action.OPEN_APP)
    self.assertEqual(action.app_name, "Audio Recorder")

  def test_complete_fail_maps_to_infeasible(self):
    action, _ = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
        dummy_action={
            "name": "mobile_use",
            "arguments": {"action": "terminate", "status": "fail"},
        },
        img_ele=self.img_ele,
        src_format="abs_origin",
        tgt_format="abs_origin",
    )

    self.assertEqual(action.action_type, json_action.STATUS)
    self.assertEqual(action.goal_status, "infeasible")

  def test_answer_accepts_content_alias(self):
    action, translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
        dummy_action={
            "name": "mobile_use",
            "arguments": {"action": "answer", "content": "True"},
        },
        img_ele=self.img_ele,
        src_format="abs_origin",
        tgt_format="abs_origin",
    )

    self.assertEqual(action.action_type, json_action.ANSWER)
    self.assertEqual(action.text, "True")
    self.assertEqual(translated["arguments"]["text"], "True")

  def test_read_alias_maps_to_answer(self):
    action, translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
        dummy_action={
            "name": "mobile_use",
            "arguments": {"action": "read", "value": "5"},
        },
        img_ele=self.img_ele,
        src_format="abs_origin",
        tgt_format="abs_origin",
    )

    self.assertEqual(action.action_type, json_action.ANSWER)
    self.assertEqual(action.text, "5")
    self.assertEqual(translated["arguments"]["action"], "answer")

  def test_top_level_name_answer_is_supported(self):
    action, translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
        dummy_action={
            "name": "answer",
            "arguments": {"text": "I have taken one photo."},
        },
        img_ele=self.img_ele,
        src_format="abs_origin",
        tgt_format="abs_origin",
    )

    self.assertEqual(action.action_type, json_action.ANSWER)
    self.assertEqual(action.text, "I have taken one photo.")
    self.assertEqual(translated["arguments"]["text"], "I have taken one photo.")

  def test_top_level_name_click_missing_coordinate_raises_clean_value_error(self):
    with self.assertRaisesRegex(ValueError, "missing coordinate"):
      mobile_agent_utils.convert_mobile_agent_action_to_json_action(
          dummy_action={
              "name": "click",
              "arguments": {"button": "play"},
          },
          img_ele=self.img_ele,
          src_format="abs_origin",
          tgt_format="abs_origin",
      )

  def test_click_accepts_element_id_fallback(self):
    action, translated = mobile_agent_utils.convert_mobile_agent_action_to_json_action(
        dummy_action={
            "name": "mobile_use",
            "arguments": {"action": "click", "element_id": 2},
        },
        img_ele=self.img_ele,
        src_format="abs_origin",
        tgt_format="abs_origin",
        ui_elements=[
            _Element(_BBox(0, 0, 10, 10)),
            _Element(_BBox(10, 10, 20, 20)),
            _Element(_BBox(20, 20, 40, 40)),
        ],
    )

    self.assertEqual(action.action_type, json_action.CLICK)
    self.assertEqual(action.index, 2)
    self.assertNotIn("coordinate", translated["arguments"])


if __name__ == "__main__":
  absltest.main()
