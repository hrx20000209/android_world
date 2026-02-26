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


class CoordinateModeTest(absltest.TestCase):

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


if __name__ == "__main__":
  absltest.main()
