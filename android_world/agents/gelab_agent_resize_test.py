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

import base64
import contextlib
import io
from unittest import mock

from absl.testing import absltest
import numpy as np
from PIL import Image

from android_world.agents import gelab_agent_resize
from android_world.env import interface
from android_world.utils import test_utils


class _FixedStateEnv(test_utils.FakeAsyncEnv):
  """Fake env with fixed screenshot size for resize testing."""

  def __init__(self, width: int, height: int):
    super().__init__()
    self._pixels = np.zeros((height, width, 3), dtype=np.uint8)

  def reset(self, go_home: bool = False) -> interface.State:
    del go_home
    return interface.State(
        pixels=np.array(self._pixels, copy=True),
        forest=None,
        ui_elements=[],
    )

  def get_state(self, wait_to_stabilize: bool = False) -> interface.State:
    del wait_to_stabilize
    return interface.State(
        pixels=np.array(self._pixels, copy=True),
        forest=mock.MagicMock(),
        ui_elements=[],
    )

  @property
  def screen_size(self) -> tuple[int, int]:
    return (int(self._pixels.shape[1]), int(self._pixels.shape[0]))

  @property
  def logical_screen_size(self) -> tuple[int, int]:
    return (int(self._pixels.shape[1]), int(self._pixels.shape[0]))


def _decode_data_url_to_image(data_url: str) -> Image.Image:
  self_prefix = "data:image/png;base64,"
  if not data_url.startswith(self_prefix):
    raise ValueError("not a png data URL")
  payload = data_url[len(self_prefix):]
  image_bytes = base64.b64decode(payload)
  return Image.open(io.BytesIO(image_bytes))


class GELABResizeAgentTest(absltest.TestCase):

  def test_step_resizes_model_input_and_logs_resolution(self):
    env = _FixedStateEnv(width=240, height=120)
    vllm = mock.Mock()
    vllm.predict_mm.return_value = (
        '{"action_type":"COMPLETE","status":"SUCCESS","value":"done"}',
        None,
        None,
    )
    agent = gelab_agent_resize.GELABAgent(
        env=env,
        vllm=vllm,
        image_downsample_scale=2.0,
    )

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      result = agent.step("dummy task")

    self.assertTrue(result.done)
    output_text = stdout.getvalue()
    self.assertIn("Step 0: Resolution", output_text)
    self.assertIn("original=240x120", output_text)
    self.assertIn("model=120x60", output_text)

    call_kwargs = vllm.predict_mm.call_args.kwargs
    messages = call_kwargs["messages"]
    image_data_url = messages[1]["content"][1]["image_url"]["url"]
    model_input_image = _decode_data_url_to_image(image_data_url)
    self.assertEqual(model_input_image.size, (120, 60))


if __name__ == "__main__":
  absltest.main()

