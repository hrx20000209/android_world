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


if __name__ == "__main__":
  absltest.main()
