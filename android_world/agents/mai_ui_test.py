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

from android_world.agents import mai_ui
from android_world.utils import test_utils


class MAIParserTest(absltest.TestCase):

  def test_parse_tool_call_keeps_coordinate_for_valid_json(self):
    response = (
        '<tool_call>\n'
        '{"name": "mobile_use", "arguments": {"action": "CLICK", "coordinate": [500, 500]}}\n'
        '</tool_call>'
    )

    parsed = mai_ui.parse_mai_tool_call(response)

    self.assertEqual(
        parsed,
        {
            "name": "mobile_use",
            "arguments": {"action": "CLICK", "coordinate": [500, 500]},
        },
    )

  def test_safe_json_loads_recovers_uppercase_click_coordinate(self):
    malformed = '{"name":"mobile_use","arguments":{"action":"CLICK","coordinate":[500,500]'

    parsed = mai_ui.safe_json_loads(malformed)

    self.assertEqual(parsed["name"], "mobile_use")
    self.assertEqual(parsed["arguments"]["action"], "CLICK")
    self.assertEqual(parsed["arguments"]["coordinate"], [500, 500])

  def test_safe_json_loads_recovers_answer_content_alias(self):
    malformed = '{"name":"mobile_use","arguments":{"action":"answer","content":"True"}'

    parsed = mai_ui.safe_json_loads(malformed)

    self.assertEqual(parsed["name"], "mobile_use")
    self.assertEqual(parsed["arguments"]["action"], "answer")
    self.assertEqual(parsed["arguments"]["text"], "True")


class MAIStepLimitTest(absltest.TestCase):

  def test_step_limit_is_capped_at_20(self):
    env = test_utils.FakeAsyncEnv()
    vllm = mock.Mock()
    agent = mai_ui.MAIUIAgent(env=env, vllm=vllm, src_format="abs_origin")
    agent.set_max_steps(200)
    agent._actions = [{} for _ in range(20)]

    result = agent.step("do something")

    self.assertTrue(result.done)
    self.assertEqual(result.data["action"].goal_status, "infeasible")
    vllm.predict_mm.assert_not_called()


if __name__ == "__main__":
  absltest.main()
