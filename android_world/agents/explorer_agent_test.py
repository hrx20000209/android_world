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
import numpy as np

from android_world.agents import explorer_agent
from android_world.agents.explorer_agent import ExplorerElementAgent
from android_world.utils import test_utils


class ExplorerHistoryFormattingTest(absltest.TestCase):

  def _agent(self) -> ExplorerElementAgent:
    return object.__new__(ExplorerElementAgent)

  def test_history_prompt_keeps_open_app_swipe_and_system_button(self):
    agent = self._agent()
    agent.actions = [
        {
            "tool_call": {"arguments": {"action": "open_app", "text": "Audio Recorder"}},
            "action_dict": {"action_type": "open_app", "app_name": "Audio Recorder"},
        },
        {
            "tool_call": {"arguments": {"action": "swipe", "direction": "up"}},
            "action_dict": {"action_type": "swipe", "direction": "up"},
        },
        {
            "tool_call": {"arguments": {"action": "system_button", "button": "back"}},
            "action_dict": {"action_type": "navigate_back"},
        },
    ]
    agent.history = ["legacy1", "legacy2", "legacy3"]

    text = agent._history_prompt_text(max_items=8)

    self.assertIn("1. open_app \"Audio Recorder\"", text)
    self.assertIn("2. swipe up", text)
    self.assertIn("3. system_button back", text)

  def test_simplify_history_item_supports_swipe_coordinate_form(self):
    item = "[llm] action=swipe, start_coordinate=[10, 20], end_coordinate=[30, 40]"
    self.assertEqual(
        ExplorerElementAgent._simplify_history_item(item),
        "swipe [10, 20]->[30, 40]",
    )

  def test_task_wants_back_navigation(self):
    self.assertTrue(
        ExplorerElementAgent._task_wants_back_navigation(
            ["Go back to home screen"],
            [],
        )
    )
    self.assertFalse(
        ExplorerElementAgent._task_wants_back_navigation(
            ["Record audio and save it"],
            ["open_app"],
        )
    )


class _BBox:

  def __init__(self, x_min: int, y_min: int, x_max: int, y_max: int):
    self.x_min = x_min
    self.y_min = y_min
    self.x_max = x_max
    self.y_max = y_max


class _Element:

  def __init__(
      self,
      *,
      text: str = "",
      desc: str = "",
      rid: str = "",
      class_name: str = "android.widget.Button",
      clickable: bool = True,
      editable: bool = False,
      long_clickable: bool = False,
      scrollable: bool = False,
      checkable: bool = False,
      visible: bool = True,
      enabled: bool = True,
      bbox: _BBox | None = None,
  ):
    self.text = text
    self.content_description = desc
    self.resource_id = rid
    self.resource_name = rid
    self.class_name = class_name
    self.is_clickable = clickable
    self.is_editable = editable
    self.is_long_clickable = long_clickable
    self.is_scrollable = scrollable
    self.is_checkable = checkable
    self.is_visible = visible
    self.is_enabled = enabled
    self.bbox_pixels = bbox


class ExplorerFilteringRegressionTest(absltest.TestCase):

  def _agent(self) -> ExplorerElementAgent:
    agent = object.__new__(ExplorerElementAgent)
    agent._recent_clicked_bounds = []
    agent._recent_clicked_window = 20
    agent._bound_effect_ema = {}
    agent._bound_effect_count = {}
    agent._bound_visit_count = {}
    agent._embed_ready = True
    agent._embed_model = None
    agent._emb_cache = {}
    return agent

  def test_collect_candidates_keeps_singleton_even_if_recent_and_low_effect(self):
    agent = self._agent()
    key = "0:0:100:100"
    agent._recent_clicked_bounds = [key]
    agent._bound_effect_ema = {key: 0.0}
    agent._bound_effect_count = {key: 3}
    agent._bound_visit_count = {key: 5.0}
    element = _Element(text="Get started", rid="com.dimowner.audiorecorder:id/btn_action", bbox=_BBox(0, 0, 100, 100))

    candidates, stats = agent._collect_candidates(
        [element],
        safe=True,
        intent_flags={"input": False, "select": False, "nav": False},
        query_keywords=[],
        avoid_keys={key},
        hard_avoid=True,
        allow_back_navigation=False,
    )

    self.assertLen(candidates, 1)
    self.assertEqual(stats["removed_visited"], 0)
    self.assertEqual(stats.get("removed_low_effect_repeat", 0), 0)

  def test_collect_candidates_keeps_singleton_back(self):
    agent = self._agent()
    back = _Element(desc="Navigate up", rid="com.dimowner.audiorecorder:id/btn_back", bbox=_BBox(0, 0, 120, 120))

    candidates, stats = agent._collect_candidates(
        [back],
        safe=True,
        intent_flags={"input": False, "select": False, "nav": False},
        query_keywords=[],
        avoid_keys=set(),
        hard_avoid=False,
        allow_back_navigation=False,
    )

    self.assertLen(candidates, 1)
    self.assertEqual(stats["kept_back_escape"], 0)

  def test_collect_candidates_keeps_keyboard_in_similarity_only_mode(self):
    agent = self._agent()
    keyboard = _Element(
        text="A",
        rid="com.google.android.inputmethod.latin:id/key_pos_0_0",
        class_name="android.widget.Button",
        bbox=_BBox(0, 0, 80, 80),
    )
    normal = _Element(
        text="Get started",
        rid="com.dimowner.audiorecorder:id/btn_action",
        bbox=_BBox(100, 100, 220, 180),
    )

    candidates, stats = agent._collect_candidates(
        [keyboard, normal],
        safe=True,
        intent_flags={"input": False, "select": False, "nav": False},
        query_keywords=[],
        avoid_keys=set(),
        hard_avoid=False,
        allow_back_navigation=False,
    )
    self.assertLen(candidates, 2)
    self.assertEqual(stats["removed_meaningless"], 0)

  def test_collect_candidates_uses_noninteractive_text_fallback(self):
    agent = self._agent()
    share = _Element(
        text="Share",
        rid="android:id/title",
        clickable=False,
        bbox=_BBox(700, 260, 950, 350),
    )
    rename = _Element(
        text="Rename",
        rid="android:id/title",
        clickable=False,
        bbox=_BBox(700, 520, 950, 620),
    )
    clock = _Element(
        text="15:51",
        desc="15:51",
        rid="com.android.systemui:id/clock",
        clickable=False,
        bbox=_BBox(20, 20, 180, 90),
    )

    candidates, stats = agent._collect_candidates(
        [share, rename, clock],
        safe=True,
        intent_flags={"input": False, "select": False, "nav": False},
        query_keywords=[],
        avoid_keys=set(),
        hard_avoid=False,
        allow_back_navigation=False,
    )

    self.assertLen(candidates, 2)
    self.assertListEqual([idx for idx, _ in candidates], [0, 1])
    self.assertEqual(stats["interactive_total"], 0)
    self.assertEqual(stats["filter_level"], "similarity_with_noninteractive_fallback")
    self.assertEqual(stats["removed_meaningless"], 1)

  def test_score_is_similarity_only(self):
    agent = self._agent()
    element = _Element(
        text="Audio Recorder",
        rid="com.google.android.apps.nexuslauncher:id/icon",
        bbox=_BBox(0, 0, 100, 100),
    )

    scored = agent._score_candidate(
        index=0,
        element=element,
        goal_queries=["Open Audio Recorder app"],
        runtime_queries=[],
        intent_flags={"input": False, "select": False, "nav": False},
        query_keywords=[],
    )
    self.assertGreaterEqual(scored.score, scored.similarity)

  def test_collect_candidates_hard_avoid_filters_when_multiple(self):
    agent = self._agent()
    element_a = _Element(text="A", rid="rid_a", bbox=_BBox(0, 0, 100, 100))
    element_b = _Element(text="B", rid="rid_b", bbox=_BBox(120, 0, 220, 100))
    avoid_key = agent._element_key(0, element_a)

    candidates, stats = agent._collect_candidates(
        [element_a, element_b],
        safe=True,
        intent_flags={"input": False, "select": False, "nav": False},
        query_keywords=[],
        avoid_keys={avoid_key},
        hard_avoid=True,
        allow_back_navigation=False,
    )

    self.assertLen(candidates, 1)
    self.assertEqual(candidates[0][0], 1)
    self.assertEqual(stats["removed_visited"], 1)

  def test_select_depth_candidate_respects_hard_avoid(self):
    agent = self._agent()
    agent.no_effect_delta_threshold = 1.2
    cand_a = explorer_agent.CandidateScore(
        index=0,
        key="key_a",
        text="A",
        score=0.90,
        task_similarity=0.90,
        runtime_similarity=0.0,
        similarity=0.90,
        visits=0.0,
        is_clickable=True,
    )
    cand_b = explorer_agent.CandidateScore(
        index=1,
        key="key_b",
        text="B",
        score=0.60,
        task_similarity=0.60,
        runtime_similarity=0.0,
        similarity=0.60,
        visits=0.0,
        is_clickable=True,
    )

    picked, skipped = agent._select_depth_candidate(
        [cand_a, cand_b],
        semantic_low=0.2,
        intent_flags={"input": False, "select": False, "nav": False},
        avoid_keys={"key_a"},
        hard_avoid=True,
    )

    self.assertIsNotNone(picked)
    self.assertEqual(picked.key, "key_b")
    self.assertEqual(skipped, 1)

  def test_select_depth_candidate_prefers_semantic_floor(self):
    agent = self._agent()
    agent.no_effect_delta_threshold = 1.2
    cand_low_sim = explorer_agent.CandidateScore(
        index=0,
        key="key_low",
        text="low",
        score=0.95,
        task_similarity=0.10,
        runtime_similarity=0.0,
        similarity=0.10,
        visits=0.0,
        is_clickable=True,
    )
    cand_high_sim = explorer_agent.CandidateScore(
        index=1,
        key="key_high",
        text="high",
        score=0.70,
        task_similarity=0.70,
        runtime_similarity=0.0,
        similarity=0.70,
        visits=0.0,
        is_clickable=True,
    )

    picked, skipped = agent._select_depth_candidate(
        [cand_low_sim, cand_high_sim],
        semantic_low=0.35,
        intent_flags={"input": False, "select": False, "nav": False},
        avoid_keys=set(),
        hard_avoid=False,
    )

    self.assertIsNotNone(picked)
    self.assertEqual(picked.key, "key_high")
    self.assertEqual(skipped, 0)

  def test_same_root_page_rejects_activity_mismatch_when_mae_is_large(self):
    agent = self._agent()
    root_pixels = np.zeros((24, 24, 3), dtype=np.uint8)
    agent._explore_root_pixels = root_pixels
    agent._explore_root_hash = explorer_agent._phash_pixels(root_pixels)
    agent._explore_root_activity = (
        "com.dimowner.audiorecorder/com.dimowner.audiorecorder.app.main.MainActivity"
    )

    same, reason = agent._same_root_page(
        np.full((24, 24, 3), 255, dtype=np.uint8),
        curr_activity=(
            "com.dimowner.audiorecorder/com.dimowner.audiorecorder.app.trash.TrashActivity"
        ),
    )
    self.assertFalse(same)
    self.assertIn("activity_mismatch", reason)

    same_main, _ = agent._same_root_page(
        np.array(root_pixels, copy=True),
        curr_activity=(
            "com.dimowner.audiorecorder/com.dimowner.audiorecorder.app.main.MainActivity"
        ),
    )
    self.assertTrue(same_main)


class ExplorerBudgetPolicyTest(absltest.TestCase):

  def _agent(self) -> ExplorerElementAgent:
    agent = object.__new__(ExplorerElementAgent)
    agent.explore_max_depth = 3
    agent.explore_max_steps = 14
    agent.explore_leaf_width = 4
    return agent

  def test_compute_explore_budget_default_no_boost(self):
    agent = self._agent()
    max_depth, topk, max_steps, boost = agent._compute_explore_budget()
    self.assertEqual(max_depth, 3)
    self.assertEqual(topk, 7)
    self.assertEqual(max_steps, 14)
    self.assertEqual(boost, 0)

  def test_compute_explore_budget_ignores_boost_state(self):
    agent = self._agent()
    max_depth, topk, max_steps, boost = agent._compute_explore_budget()
    self.assertEqual(max_depth, 3)
    self.assertEqual(topk, 7)
    self.assertEqual(boost, 0)
    self.assertEqual(max_steps, 14)


class ExplorerTargetedPolicyTest(absltest.TestCase):

  def _agent(self) -> ExplorerElementAgent:
    agent = object.__new__(ExplorerElementAgent)
    agent.enable_parallel_exploration = True
    agent.targeted_exploration = True
    agent._explore_cooldown_steps = 0
    agent.explore_bootstrap_steps = 1
    agent.explore_read_task_step_cap = 2
    agent.explore_periodic_interval = 0
    agent.explore_stuck_action_repeat = 2
    agent._no_effect_repeat = 0
    agent.actions = []
    agent._reasoning_page_records = []
    return agent

  def test_should_start_exploration_skips_late_read_only_goal(self):
    agent = self._agent()
    should_start, reason = agent._should_start_exploration(
        step_no=3,
        goal="What events do I have on October 24 in Simple Calendar Pro? Answer with the event names.",
        current_activity="com.simplemobiletools.calendar.pro/com.simplemobiletools.calendar.pro.activities.MainActivity",
    )
    self.assertFalse(should_start)
    self.assertEqual(reason, "read_only_skip")

  def test_should_start_exploration_triggers_on_no_effect(self):
    agent = self._agent()
    agent._no_effect_repeat = 1
    should_start, reason = agent._should_start_exploration(
        step_no=4,
        goal="Delete the selected recipe from Broccoli app.",
    )
    self.assertTrue(should_start)
    self.assertEqual(reason, "no_effect_repeat")

  def test_force_open_target_app_on_launcher(self):
    agent = self._agent()
    action = explorer_agent.json_action.JSONAction(
        action_type=explorer_agent.json_action.SWIPE,
        direction="up",
    )
    force_open, app_name = agent._should_force_open_target_app(
        step_no=1,
        goal="Is the note titled To-Do List in the Joplin app marked as a todo item?",
        current_activity="com.google.android.apps.nexuslauncher/com.google.android.apps.nexuslauncher.NexusLauncherActivity",
        action=action,
    )
    self.assertTrue(force_open)
    self.assertEqual(app_name, "Joplin")

  def test_force_back_from_chooser_for_delete_goal(self):
    agent = self._agent()
    action = explorer_agent.json_action.JSONAction(
        action_type=explorer_agent.json_action.CLICK,
        x=100,
        y=100,
    )
    self.assertTrue(
        agent._should_force_back_from_chooser(
            goal="Delete the selected recipe from Broccoli app.",
            current_activity="android/com.android.internal.app.ChooserActivity",
            action=action,
        )
    )


class ExplorerStatusPolicyTest(absltest.TestCase):

  def test_ensure_root_before_reasoning_skips_without_root_baseline(self):
    agent = object.__new__(ExplorerElementAgent)
    agent._explore_root_hash = None
    agent._explore_root_pixels = None
    logs = []
    agent._emit_log = lambda message, tag="": logs.append((message, tag))

    result = agent._ensure_root_before_reasoning(step_no=9, max_attempts=2)

    self.assertTrue(result["success"])
    self.assertTrue(result["verified"])
    self.assertEqual(result["attempts"], 0)
    self.assertEqual(result["reason"], "missing_root_baseline")
    self.assertTrue(any("rollback_guard_skipped" in msg for msg, _ in logs))

  def test_is_complete_status_action(self):
    complete_action = explorer_agent.json_action.JSONAction(
        action_type=explorer_agent.json_action.STATUS,
        goal_status="complete",
    )
    infeasible_action = explorer_agent.json_action.JSONAction(
        action_type=explorer_agent.json_action.STATUS,
        goal_status="infeasible",
    )

    self.assertTrue(ExplorerElementAgent._is_complete_status_action(complete_action))
    self.assertFalse(ExplorerElementAgent._is_complete_status_action(infeasible_action))

  def test_task_status_from_action(self):
    complete_action = explorer_agent.json_action.JSONAction(
        action_type=explorer_agent.json_action.STATUS,
        goal_status="task_complete",
    )
    infeasible_action = explorer_agent.json_action.JSONAction(
        action_type=explorer_agent.json_action.STATUS,
        goal_status="infeasible",
    )
    click_action = explorer_agent.json_action.JSONAction(
        action_type=explorer_agent.json_action.CLICK,
        x=120,
        y=320,
    )

    self.assertEqual(ExplorerElementAgent._task_status_from_action(complete_action), "completed")
    self.assertEqual(ExplorerElementAgent._task_status_from_action(infeasible_action), "infeasible")
    self.assertIsNone(ExplorerElementAgent._task_status_from_action(click_action))


class ExplorerScreenshotScaleTest(absltest.TestCase):

  def _agent(self, scale: int) -> ExplorerElementAgent:
    agent = object.__new__(ExplorerElementAgent)
    agent.image_downsample_scale = scale
    return agent

  def test_normalize_downsample_scale_clamps_to_positive_int(self):
    self.assertEqual(explorer_agent._normalize_downsample_scale(2.4), 2)
    self.assertEqual(explorer_agent._normalize_downsample_scale("3"), 3)
    self.assertEqual(explorer_agent._normalize_downsample_scale(0), 1)

  def test_prepare_reasoning_pixels_downsamples_by_scale(self):
    agent = self._agent(scale=2)
    pixels = np.zeros((120, 80, 3), dtype=np.uint8)

    resized = agent._prepare_reasoning_pixels(pixels)

    self.assertEqual(resized.shape, (60, 40, 3))

  def test_prepare_reasoning_pixels_is_noop_when_scale_is_one(self):
    agent = self._agent(scale=1)
    pixels = np.zeros((33, 19, 3), dtype=np.uint8)

    resized = agent._prepare_reasoning_pixels(pixels)

    self.assertIs(resized, pixels)


class ExplorerStepLimitTest(absltest.TestCase):

  def test_step_limit_is_capped_at_20(self):
    env = test_utils.FakeAsyncEnv()
    vllm = mock.Mock()
    agent = explorer_agent.ExplorerElementAgent(
        env=env,
        vllm=vllm,
        verbose_step_logs=False,
    )
    agent.set_max_steps(200)
    agent.actions = [{} for _ in range(20)]

    result = agent.step("do something")

    self.assertTrue(result.done)
    self.assertEqual(result.data["goal_status"], "infeasible")
    self.assertEqual(result.data["source"], "max_step_guard")
    vllm.predict_mm.assert_not_called()


if __name__ == "__main__":
  absltest.main()
