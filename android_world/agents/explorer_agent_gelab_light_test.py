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

import os
import tempfile

from absl.testing import absltest

from android_world.agents.explorer_agent_gelab_light import ExplorerElementAgent
from android_world.env import json_action


class HomeReplayAnchorTest(absltest.TestCase):

  def _agent(self) -> ExplorerElementAgent:
    return object.__new__(ExplorerElementAgent)

  def test_root_activity_maps_to_app_name(self):
    agent = self._agent()

    self.assertEqual(
        agent._app_name_for_root_activity(
            "de.dennisguse.opentracks/de.dennisguse.opentracks.TrackListActivity"
        ),
        "OpenTracks",
    )
    self.assertEqual(
        agent._app_name_for_root_activity(
            "com.android.settings/com.android.settings.homepage.SettingsHomepageActivity"
        ),
        "Settings",
    )
    self.assertEqual(
        agent._app_name_for_root_activity(
            "com.google.android.apps.nexuslauncher/com.google.android.apps.nexuslauncher.NexusLauncherActivity"
        ),
        "",
    )

  def test_prepare_home_replay_inserts_missing_open_app_anchor(self):
    agent = self._agent()

    replay_actions, info = agent._prepare_home_replay_actions(
        [
            json_action.JSONAction(action_type=json_action.SWIPE, direction="up"),
            json_action.JSONAction(action_type=json_action.CLICK, x=100, y=200),
        ],
        "com.android.settings/com.android.settings.homepage.SettingsHomepageActivity",
    )

    self.assertTrue(info["inserted_open_app_anchor"])
    self.assertEqual(replay_actions[0].action_type, json_action.OPEN_APP)
    self.assertEqual(replay_actions[0].app_name, "Settings")
    self.assertEqual(
        [action.action_type for action in replay_actions],
        [json_action.OPEN_APP, json_action.SWIPE, json_action.CLICK],
    )

  def test_prepare_home_replay_drops_actions_before_existing_root_anchor(self):
    agent = self._agent()

    replay_actions, info = agent._prepare_home_replay_actions(
        [
            json_action.JSONAction(action_type=json_action.INPUT_TEXT, text="bad prefix"),
            json_action.JSONAction(action_type=json_action.OPEN_APP, app_name="OpenTracks"),
            json_action.JSONAction(action_type=json_action.CLICK, x=10, y=20),
            json_action.JSONAction(action_type=json_action.OPEN_APP, app_name="Clock"),
        ],
        "de.dennisguse.opentracks/de.dennisguse.opentracks.TrackListActivity",
    )

    self.assertFalse(info["inserted_open_app_anchor"])
    self.assertEqual(info["dropped_actions_before_anchor"], 1)
    self.assertEqual(info["dropped_foreign_open_app_actions"], 1)
    self.assertEqual(
        [action.action_type for action in replay_actions],
        [json_action.OPEN_APP, json_action.CLICK],
    )
    self.assertEqual(replay_actions[0].app_name, "OpenTracks")

  def test_select_replay_actions_keeps_full_safe_history(self):
    agent = self._agent()
    agent._actions = [
        {"action_dict": {"action_type": json_action.OPEN_APP, "app_name": "Settings"}},
        {"action_dict": {"action_type": json_action.SWIPE, "direction": "up"}},
        {"action_dict": {"action_type": json_action.CLICK, "x": 10, "y": 20}},
        {"action_dict": {"action_type": json_action.CLICK, "x": 30, "y": 40}},
        {"action_dict": {"action_type": json_action.WAIT}},
    ]

    replay_actions = agent._select_replay_actions_for_probe(
        json_action.JSONAction(action_type=json_action.CLICK, x=50, y=60)
    )

    self.assertEqual(
        [action.action_type for action in replay_actions],
        [
            json_action.OPEN_APP,
            json_action.SWIPE,
            json_action.CLICK,
            json_action.CLICK,
            json_action.WAIT,
            json_action.CLICK,
        ],
    )
    self.assertEqual(replay_actions[0].app_name, "Settings")

  def test_prompt_trace_files_use_short_path_when_windows_path_is_long(self):
    if os.name != "nt":
      self.skipTest("Windows path-length fallback is only used on Windows.")
    agent = self._agent()
    agent.light_explore_variant = "S4_MCTS_BUDGET12"

    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
      long_task_dir = os.path.join(tmp, "task_" + ("x" * 155))
      agent._task_output_dir = lambda goal: long_task_dir

      full_path, context_path = agent._write_prompt_trace_files(
          goal="long path prompt trace test",
          step_idx=0,
          messages=[],
          exploration_context="ctx",
          message_text="msg",
          hint_for_prompt="hint",
      )

      self.assertIn(os.path.join("p", "S4MCTSBU"), context_path)
      self.assertTrue(os.path.exists(full_path))
      self.assertTrue(os.path.exists(context_path))
      self.assertLess(len(os.path.abspath(context_path)), 240)


if __name__ == "__main__":
  absltest.main()
