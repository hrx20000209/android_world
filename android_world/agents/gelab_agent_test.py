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
from unittest import mock

from android_world.agents import gelab_agent
from android_world.env import json_action
from android_world.utils import test_utils


class GELABParserTest(absltest.TestCase):

  def test_parse_click_response(self):
    response = (
        "<THINK>Open the Audio Recorder app from the launcher.</THINK>\n"
        "explain:Open the requested app\t"
        "action:AWAKE\t"
        "value:Audio Recorder\t"
        "summary:Opened Audio Recorder"
    )

    parsed = gelab_agent.parse_gelab_response(response)

    self.assertEqual(parsed["cot"], "Open the Audio Recorder app from the launcher.")
    self.assertEqual(parsed["action"], "AWAKE")
    self.assertEqual(parsed["value"], "Audio Recorder")
    self.assertEqual(parsed["summary"], "Opened Audio Recorder")

  def test_parse_slide_points(self):
    response = (
        "<THINK>Swipe up to reveal more content.</THINK>\n"
        "explain:Scroll the page upward\t"
        "action:SLIDE\t"
        "point1:500,800\t"
        "point2:500,200\t"
        "summary:Scrolled upward once"
    )

    parsed = gelab_agent.parse_gelab_response(response)

    self.assertEqual(parsed["action"], "SLIDE")
    self.assertEqual(parsed["point1"], [500, 800])
    self.assertEqual(parsed["point2"], [500, 200])

  def test_parse_tool_call_json_fallback(self):
    response = """
<tool_call>
{"name":"mobile_use","arguments":{"action":"open_app","text":"Audio Recorder"}}
</tool_call>
"""

    parsed = gelab_agent.parse_gelab_response(response)

    self.assertEqual(parsed["action"], "__TOOL_CALL__")
    self.assertEqual(parsed["source_format"], "tool_call_json")
    self.assertEqual(parsed["tool_call"]["arguments"]["action"], "open_app")
    self.assertEqual(parsed["tool_call"]["arguments"]["text"], "Audio Recorder")


class GELABActionAdapterTest(absltest.TestCase):

  def test_awake_maps_to_open_app(self):
    action, tool_call, extras = gelab_agent.gelab_action_to_json_action(
        parsed_action=gelab_agent.parse_gelab_response(
            "<THINK>x</THINK>\nexplain:y\taction:AWAKE\tvalue:Audio Recorder\tsummary:z"
        ),
        screen_size=(1080, 2400),
    )

    self.assertEqual(action.action_type, json_action.OPEN_APP)
    self.assertEqual(action.app_name, "Audio Recorder")
    self.assertEqual(tool_call["arguments"]["action"], "open_app")
    self.assertEmpty(extras)

  def test_click_maps_normalized_point_to_absolute(self):
    action, tool_call, _ = gelab_agent.gelab_action_to_json_action(
        parsed_action=gelab_agent.parse_gelab_response(
            "<THINK>x</THINK>\nexplain:y\taction:CLICK\tpoint:500,500\tsummary:z"
        ),
        screen_size=(1080, 2400),
    )

    self.assertEqual(action.action_type, json_action.CLICK)
    self.assertEqual((action.x, action.y), (540, 1200))
    self.assertEqual(tool_call["arguments"]["coordinate"], [540, 1200])

  def test_type_preserves_text_and_point(self):
    action, tool_call, _ = gelab_agent.gelab_action_to_json_action(
        parsed_action=gelab_agent.parse_gelab_response(
            "<THINK>x</THINK>\nexplain:y\taction:TYPE\tvalue:hello world\tpoint:250,750\tsummary:z"
        ),
        screen_size=(1000, 2000),
    )

    self.assertEqual(action.action_type, json_action.INPUT_TEXT)
    self.assertEqual(action.text, "hello world")
    self.assertEqual((action.x, action.y), (250, 1499))
    self.assertEqual(tool_call["arguments"]["text"], "hello world")

  def test_slide_keeps_start_end_coordinates(self):
    action, tool_call, extras = gelab_agent.gelab_action_to_json_action(
        parsed_action=gelab_agent.parse_gelab_response(
            "<THINK>x</THINK>\nexplain:y\taction:SLIDE\tpoint1:100,500\tpoint2:900,500\tsummary:z"
        ),
        screen_size=(1080, 2400),
    )

    self.assertEqual(action.action_type, json_action.SWIPE)
    self.assertEqual(action.direction, "right")
    self.assertEqual(extras["start_coordinate"], [108, 1200])
    self.assertEqual(extras["end_coordinate"], [971, 1200])
    self.assertEqual(tool_call["arguments"]["direction"], "right")

  def test_complete_maps_to_status_complete(self):
    action, _, extras = gelab_agent.gelab_action_to_json_action(
        parsed_action=gelab_agent.parse_gelab_response(
            "<THINK>x</THINK>\nexplain:y\taction:COMPLETE\treturn:done\tsummary:z"
        ),
        screen_size=(1080, 2400),
    )

    self.assertEqual(action.action_type, json_action.STATUS)
    self.assertEqual(action.goal_status, "task_complete")
    self.assertEqual(extras["return_text"], "done")

  def test_tool_call_click_keeps_absolute_coordinate(self):
    action, tool_call, _ = gelab_agent.gelab_action_to_json_action(
        parsed_action=gelab_agent.parse_gelab_response(
            '<tool_call>{"name":"mobile_use","arguments":{"action":"click","coordinate":[500,500]}}</tool_call>'
        ),
        screen_size=(1080, 2400),
    )

    self.assertEqual(action.action_type, json_action.CLICK)
    self.assertEqual((action.x, action.y), (500, 500))
    self.assertEqual(tool_call["arguments"]["coordinate"], [500, 500])

  def test_abort_maps_to_status_infeasible(self):
    action, _, extras = gelab_agent.gelab_action_to_json_action(
        parsed_action=gelab_agent.parse_gelab_response(
            "<THINK>x</THINK>\nexplain:y\taction:ABORT\tvalue:not possible\tsummary:z"
        ),
        screen_size=(1080, 2400),
    )

    self.assertEqual(action.action_type, json_action.STATUS)
    self.assertEqual(action.goal_status, "infeasible")
    self.assertEqual(extras["abort_reason"], "not possible")

  def test_info_maps_to_answer(self):
    action, tool_call, _ = gelab_agent.gelab_action_to_json_action(
        parsed_action=gelab_agent.parse_gelab_response(
            "<THINK>x</THINK>\nexplain:y\taction:INFO\tvalue:Which file should I open?\tsummary:z"
        ),
        screen_size=(1080, 2400),
    )

    self.assertEqual(action.action_type, json_action.ANSWER)
    self.assertEqual(action.text, "Which file should I open?")
    self.assertEqual(tool_call["arguments"]["action"], "answer")


class GELABStepLimitTest(absltest.TestCase):

  def test_step_limit_is_capped_at_20(self):
    env = test_utils.FakeAsyncEnv()
    vllm = mock.Mock()
    agent = gelab_agent.GELABAgent(env=env, vllm=vllm)
    agent.set_max_steps(200)
    agent._actions = [{} for _ in range(20)]

    result = agent.step("do something")

    self.assertTrue(result.done)
    self.assertEqual(result.data["action_dict"]["goal_status"], "infeasible")
    vllm.predict_mm.assert_not_called()


if __name__ == "__main__":
  absltest.main()
