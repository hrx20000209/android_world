# Copyright 2026 The android_world Authors.

import json
from pathlib import Path
import tempfile
import time

from absl.testing import absltest
from PIL import Image

from android_world.agents import mobileexplorer
from android_world.env import json_action
from android_world.env import representation_utils
from scripts import run_shadow_oracle_probe


def _trajectory(goal: str, seed: int = 1):
  return mobileexplorer.SuccessfulTrajectory(
      trajectory_id=f"trajectory-{seed}",
      task_template="SimpleCalendarNextEvent",
      goal=goal,
      seed=seed,
      steps=(mobileexplorer.TrajectoryStep(
          action={"action_type": "click", "index": 4},
          summary="Opened event Alice at 14:30",
          state_id="0000000000000000",
          state_schema_id="schema",
      ),),
      navigation_functions=("open calendar", "open next event details"),
  )


class TrajectoryMemoryTest(absltest.TestCase):

  def test_retrieval_generalizes_values_and_excludes_current_seed(self):
    memory = mobileexplorer.TrajectoryMemory([
        _trajectory("What is my next event after 14:30 on October 20?", seed=1),
        _trajectory("Delete a recipe", seed=2),
    ])

    result = memory.retrieve(
        "What is my next event after 09:15 on October 21?", exclude_seed=2
    )

    self.assertIsNotNone(result)
    self.assertEqual(result.trajectory.seed, 1)

  def test_exact_template_retrieval_rejects_similar_other_task(self):
    memory = mobileexplorer.TrajectoryMemory((
        mobileexplorer.SuccessfulTrajectory(
            trajectory_id="named",
            task_template="RecordWithFileName",
            goal="Record audio and save with name Alice",
            seed=1,
            steps=(),
            navigation_functions=(),
        ),
    ))

    self.assertIsNone(memory.retrieve(
        "Record audio and save it",
        task_template="RecordAudio",
        exact_template=True,
    ))

  def test_offline_arm_sees_concrete_history_mobile_arm_only_navigation(self):
    retrieved = mobileexplorer.RetrievedTrajectory(
        _trajectory("What is my next event?"), .9
    )
    image = Image.new("RGB", (8, 8), "white")
    offline = mobileexplorer.build_mobileexplorer_messages(
        "What is my next event?", "", image,
        mode=mobileexplorer.MemoryMode.OFFLINE_TRAJECTORY,
        retrieved=retrieved,
    )
    online = mobileexplorer.build_mobileexplorer_messages(
        "What is my next event?", "", image,
        mode=mobileexplorer.MemoryMode.MOBILEEXPLORER,
        retrieved=retrieved,
    )

    offline_text = offline[1]["content"][0]["text"]
    online_text = online[1]["content"][0]["text"]
    self.assertIn("Alice at 14:30", offline_text)
    self.assertNotIn("Alice at 14:30", online_text)
    self.assertIn("open next event details", online_text)

  def test_safe_offline_navigation_excludes_source_values_and_coordinates(self):
    retrieved = mobileexplorer.RetrievedTrajectory(
        _trajectory("What is my next event?"), .9
    )
    image = Image.new("RGB", (8, 8), "white")
    messages = mobileexplorer.build_mobileexplorer_messages(
        "What is my next event?", "", image,
        mode=mobileexplorer.MemoryMode.OFFLINE_NAVIGATION,
        retrieved=retrieved,
    )
    prompt = messages[1]["content"][0]["text"]
    self.assertIn("action types only", prompt)
    self.assertIn("click", prompt)
    self.assertNotIn("Alice", prompt)
    self.assertNotIn("14:30", prompt)
    self.assertNotIn("index", prompt)

  def test_empty_memory_prompt_matches_gelab_light_prompt(self):
    image = Image.new("RGB", (8, 8), "white")
    messages = mobileexplorer.build_mobileexplorer_messages(
        "Do the task", "None yet.", image,
        mode=mobileexplorer.MemoryMode.MOBILEEXPLORER,
        retrieved=None,
    )
    from android_world.agents.explorer_agent_gelab_light import _to_user_text
    self.assertEqual(
        messages[1]["content"][0]["text"],
        _to_user_text("Do the task", "None yet.", ""),
    )

  def test_terminal_answer_is_reduced_to_requested_exact_format(self):
    action = json_action.JSONAction(
        action_type=json_action.STATUS, goal_status="task_complete"
    )
    normalized = mobileexplorer._normalize_terminal_answer(
        "Return the quantity as <amount> <unit>", action,
        mobileexplorer.OrderedDict((
            ("action", "COMPLETE"), ("return", "It is 3/4 cup."),
        )),
        {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}},
        {"return_text": "It is 3/4 cup."},
    )
    self.assertEqual(normalized[3]["return_text"], "3/4 cup")
    self.assertEqual(
        normalized[1]["arguments"]["return_text"], "3/4 cup"
    )

  def test_replay_points_are_ordered_and_exclude_value_or_terminal_steps(self):
    trajectory = mobileexplorer.SuccessfulTrajectory(
        trajectory_id="ordered", task_template="Task", goal="Do task", seed=1,
        steps=tuple(
            mobileexplorer.TrajectoryStep(
                action={"action_type": action_type}, summary="",
                state_schema_id=f"state-{index}",
                source_activity="Main", source_landmarks=("a", "b"),
            )
            for index, action_type in enumerate((
                json_action.OPEN_APP, json_action.INPUT_TEXT,
                json_action.CLICK, json_action.ANSWER,
            ))
        ),
        navigation_functions=(),
    )

    self.assertEqual(mobileexplorer._replayable_step_indices(trajectory), (0,))

  def test_coordinate_replay_requires_a_live_target_node(self):
    element = representation_utils.UIElement(
        bbox_pixels=representation_utils.BoundingBox(0, 100, 0, 100),
        is_visible=True, is_enabled=True, is_clickable=True,
    )
    state = type("State", (), {"ui_elements": [element]})()

    self.assertTrue(mobileexplorer._coordinate_target_is_live(
        json_action.JSONAction(action_type=json_action.CLICK, x=50, y=50),
        state,
    ))
    self.assertFalse(mobileexplorer._coordinate_target_is_live(
        json_action.JSONAction(action_type=json_action.CLICK, x=150, y=150),
        state,
    ))

  def test_non_coordinate_reuse_point_is_executable(self):
    state = type("State", (), {"ui_elements": []})()

    self.assertTrue(mobileexplorer._coordinate_target_is_live(
        json_action.JSONAction(
            action_type=json_action.OPEN_APP, app_name="Tasks"
        ),
        state,
    ))

  def test_validated_replay_rejects_legacy_trace_without_binding(self):
    state = type("State", (), {"ui_elements": []})()
    step = mobileexplorer.TrajectoryStep(
        action={"action_type": json_action.OPEN_APP, "app_name": "Tasks"},
        summary="", state_schema_id="old-only",
    )
    self.assertIsNone(mobileexplorer._validated_replay_action(step, state, "Main"))
    trajectory = mobileexplorer.SuccessfulTrajectory(
        trajectory_id="legacy", task_template="Task", goal="Do task", seed=1,
        steps=(step,), navigation_functions=(),
    )
    self.assertEmpty(mobileexplorer._replayable_step_indices(trajectory))

  def test_validated_replay_remaps_unique_semantic_target(self):
    def element(resource, left, right):
      return representation_utils.UIElement(
          package_name="pkg", resource_name=resource,
          class_name="Button", is_visible=True, is_enabled=True,
          is_clickable=True,
          bbox_pixels=representation_utils.BoundingBox(left, right, 0, 100),
      )
    state = type("State", (), {
        "ui_elements": [element("record", 100, 200), element("settings", 0, 80)]
    })()
    step = mobileexplorer.TrajectoryStep(
        action={"action_type": json_action.CLICK, "x": 5, "y": 5}, summary="",
        source_activity="Main",
        source_landmarks=("pkg|record|Button", "pkg|settings|Button"),
        target_descriptor={
            "package_name": "pkg", "resource_name": "record",
            "class_name": "Button", "content_description": "",
        },
    )
    action = mobileexplorer._validated_replay_action(step, state, "Main")
    self.assertEqual((action.x, action.y), (150, 50))
    self.assertIsNone(mobileexplorer._validated_replay_action(step, state, "Other"))

  def test_container_click_is_not_recorded_as_replayable_button(self):
    state = type("State", (), {"ui_elements": [
        representation_utils.UIElement(
            package_name="calendar", resource_name="month_view_background",
            class_name="android.view.View", is_visible=True,
            is_enabled=True, is_clickable=True,
            bbox_pixels=representation_utils.BoundingBox(0, 1000, 0, 800),
        )
    ]})()
    action = json_action.JSONAction(action_type=json_action.CLICK, x=100, y=100)
    self.assertIsNone(mobileexplorer._target_descriptor(action, state))


class LiveEvidenceTest(absltest.TestCase):

  def test_evidence_requires_next_generation_and_exact_binding(self):
    now = time.time()
    evidence = mobileexplorer.LiveEvidence(
        evidence_id="e1",
        episode_id="episode",
        state_id="0",
        state_schema_id="schema",
        task_need="need",
        slot="answer",
        value="14:30",
        observed_generation=1,
        observed_at_s=now,
        ttl_s=30,
        confidence=.99,
        isolation_id="shadow",
        state_match=True,
        extraction_verified=True,
    )

    self.assertFalse(evidence.usable(
        episode_id="episode", state_id="0", state_schema_id="schema",
        task_need="need",
        generation=1, now_s=now,
    ))
    self.assertTrue(evidence.usable(
        episode_id="episode", state_id="1", state_schema_id="schema",
        task_need="need",
        generation=2, now_s=now + 1,
    ))
    self.assertFalse(evidence.usable(
        episode_id="episode", state_id="ff", state_schema_id="schema",
        task_need="need",
        generation=2, now_s=now + 1,
    ))

  def test_state_binding_allows_small_visual_distance(self):
    self.assertEqual(mobileexplorer._state_distance("0", "3"), 2)
    self.assertGreater(mobileexplorer._state_distance("0", "ff"), 6)

  def test_stale_evidence_has_auditable_rejection_reason(self):
    now = time.time()
    evidence = mobileexplorer.LiveEvidence(
        evidence_id="stale", episode_id="episode", state_id="0",
        state_schema_id="schema", task_need="need", slot="answer", value="x",
        observed_generation=1, observed_at_s=now - 31, ttl_s=30,
        confidence=.99, isolation_id="shadow", state_match=True,
        extraction_verified=True,
    )

    self.assertEqual(evidence.rejection_reason(
        episode_id="episode", state_id="0", state_schema_id="schema",
        task_need="need", generation=2, now_s=now,
    ), "stale")

  def test_worker_evidence_fields_parse_and_bind_to_primary(self):
    now = time.time()
    row = {
        "evidence_id": "e", "episode_id": "episode", "state_id": "0",
        "state_schema_id": "primary", "task_need": "need", "slot": "answer",
        "value": "Meeting room", "observed_generation": 1,
        "observed_at_s": now, "ttl_s": 60, "confidence": 1,
        "isolation_id": "emulator-5556", "state_match": True,
        "extraction_verified": True, "shadow_observation_state_id": "ff",
        "shadow_observation_schema_id": "shadow", "deadline_s": 100,
        "deadline_miss": False,
    }
    evidence = mobileexplorer.LiveEvidence(**row)
    self.assertTrue(evidence.usable(
        episode_id="episode", state_id="0", state_schema_id="primary",
        task_need="need", generation=2, now_s=now + 1,
    ))
    self.assertEqual(evidence.shadow_observation_schema_id, "shadow")


class ShadowTrajectorySelectionTest(absltest.TestCase):

  def test_selects_matching_template_not_first_row(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "memory.jsonl"
      rows = []
      for task in ("WrongTask", "WantedTask"):
        rows.append({
            "trajectory_id": task,
            "task_template": task,
            "goal": "What is the answer?",
            "seed": 1,
            "successful": True,
            "steps": [],
            "navigation_functions": [],
        })
      path.write_text(
          "".join(json.dumps(row) + "\n" for row in rows),
          encoding="utf-8",
      )

      selected = run_shadow_oracle_probe._load_trajectory(
          path, task_template="WantedTask", goal="What is the answer?"
      )

      self.assertEqual(selected["trajectory_id"], "WantedTask")

  def test_probe_deadline_is_a_hard_admission_gate(self):
    run_shadow_oracle_probe._check_deadline(time.monotonic() + 1, "test")
    with self.assertRaises(run_shadow_oracle_probe.ProbeDeadlineMiss):
      run_shadow_oracle_probe._check_deadline(time.monotonic() - 1, "test")

  def test_shadow_answer_must_be_visible_in_ui(self):
    element = representation_utils.UIElement(
        text="Meeting room: East Wing", is_visible=True
    )
    state = type("State", (), {"ui_elements": [element]})()
    self.assertTrue(run_shadow_oracle_probe._answer_is_visible("East Wing", state))
    self.assertFalse(run_shadow_oracle_probe._answer_is_visible("West Wing", state))


if __name__ == "__main__":
  absltest.main()
