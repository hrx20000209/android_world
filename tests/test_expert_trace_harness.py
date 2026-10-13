"""Offline contract tests: no Android dependencies, emulator, or model service."""

import dataclasses
import datetime
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "expert_trace_harness.py"
SPEC = importlib.util.spec_from_file_location("expert_trace_harness", SOURCE)
harness = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harness)


@dataclasses.dataclass
class FakeElement:
    text: str = "PRIVATE_VALUE"


class LazyForest:
    def __getattr__(self, name):
        raise AssertionError("lazy forest must not perform a gRPC lookup")


class FakeEnv:
    foreground_activity_name = "fake.activity"
    logical_screen_size = (1080, 2400)
    interaction_cache = ""

    def __init__(self, fail_action=False):
        self.executed = []
        self.fail_action = fail_action

    def reset(self, go_home=False):
        pass

    def get_state(self, wait_to_stabilize=False):
        return types.SimpleNamespace(pixels=None, forest=LazyForest(), ui_elements=[FakeElement()])

    def execute_action(self, action):
        if self.fail_action:
            raise RuntimeError("PRIVATE_VALUE")
        self.executed.append(action.action_type)
        if action.action_type == "answer":
            self.interaction_cache = action.text

    def close(self):
        pass


class FakeAction:
    def __init__(self, **values):
        self.values = values
        self.action_type = values["action_type"]
        self.text = values.get("text")
        self.goal_status = values.get("goal_status")

    def as_dict(self):
        return self.values


class FakeTask:
    params = {
        "seed": 123, "private_field": "PRIVATE_VALUE",
        "rows": [FakeElement()], "binary_id": b"\x00\xff",
        "date": datetime.date(2026, 10, 12),
    }
    goal = "PRIVATE_VALUE"
    complexity = 1

    def initialize_task(self, env):
        pass

    def is_successful(self, env):
        return 1.0


class FakeImage:
    @staticmethod
    def fromarray(pixels):
        return FakeImage()

    def save(self, path):
        path.write_bytes(b"fake screenshot")


class ExpertHarnessTests(unittest.TestCase):
    def execute(self, commands, fail_action=False, origin="manual_expert", scores=None):
        env = FakeEnv(fail_action)
        registry = types.SimpleNamespace(TaskRegistry=lambda: types.SimpleNamespace(
            get_registry=lambda family: {"ExampleTask": FakeTask}
        ))
        suite = types.SimpleNamespace(create_suite=lambda *a, **kw: {"ExampleTask": [FakeTask()]})
        if scores is not None:
            task = FakeTask()
            outcomes = iter(scores)
            def evaluate(env):
                result = next(outcomes)
                if isinstance(result, Exception):
                    raise result
                return result
            task.is_successful = evaluate
            suite.create_suite = lambda *a, **kw: {"ExampleTask": [task]}
        android = types.ModuleType("android_world")
        android.registry, android.suite_utils = registry, suite
        android_env = types.ModuleType("android_world.env")
        android_env.env_launcher = types.SimpleNamespace(load_and_setup_env=lambda **kw: env)
        android_env.json_action = types.SimpleNamespace(JSONAction=FakeAction)
        pil = types.ModuleType("PIL")
        pil.Image = FakeImage
        modules = {"android_world": android, "android_world.env": android_env, "PIL": pil}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "trace"
            args = types.SimpleNamespace(output_dir=root, task="ExampleTask", seed=34,
                                         console_port=5554, grpc_port=8554, adb_path="fake-adb",
                                         action_origin=origin, actions_jsonl=None)
            with mock.patch.dict("sys.modules", modules):
                code = harness.run(args, commands)
            summary_text = (root / "summary.json").read_text()
            summary = json.loads(summary_text)
            self.assertNotIn("PRIVATE_VALUE", summary_text)
            self.assertEqual((root / "private").stat().st_mode & 0o777, 0o700)
            for f in (root / "private").iterdir():
                self.assertEqual(f.stat().st_mode & 0o777, 0o600)
            self.assertIn("PRIVATE_VALUE", (root / "private" / "task.json").read_text())
            recorded_task = json.loads((root / "private" / "task.json").read_text())
            self.assertEqual(recorded_task["params"]["rows"], [{"text": "PRIVATE_VALUE"}])
            self.assertEqual(recorded_task["params"]["binary_id"], {"bytes_hex": "00ff"})
            self.assertEqual(recorded_task["params"]["date"], "2026-10-12")
            if summary["actions"]:
                self.assertTrue((root / "private" / "step_0000_before.png").exists())
                self.assertTrue((root / "private" / "step_0000_after.png").exists())
            return code, summary, env

    def test_answer_executes_cache_and_validates(self):
        code, result, env = self.execute([
            {"action": {"action_type": "answer", "text": "PRIVATE_VALUE"}},
        ])
        self.assertEqual(code, 0)
        self.assertEqual(env.interaction_cache, "PRIVATE_VALUE")
        self.assertTrue(result["validated_reference"])
        self.assertEqual(result["model_calls"], 0)

    def test_old_evaluation_is_invalid_after_another_action(self):
        _, result, _ = self.execute([
            {"command": "evaluate"},
            {"action": {"action_type": "navigate_back"}},
        ])
        self.assertFalse(result["done"])
        self.assertFalse(result["evaluator_complete"])
        self.assertIsNone(result["evaluator_score"])
        self.assertFalse(result["validated_reference"])

    def test_evaluate_without_completion_does_not_validate(self):
        _, result, _ = self.execute([{"command": "evaluate"}])
        self.assertTrue(result["evaluator_complete"])
        self.assertFalse(result["validated_reference"])

    def test_failed_re_evaluation_clears_old_score(self):
        code, result, _ = self.execute([
            {"command": "evaluate"}, {"command": "finish"},
        ], scores=[1.0, RuntimeError("PRIVATE_VALUE")])
        self.assertEqual(code, 1)
        self.assertFalse(result["evaluator_complete"])
        self.assertIsNone(result["evaluator_score"])
        self.assertFalse(result["validated_reference"])

    def test_unverified_input_is_never_an_expert_reference(self):
        _, result, _ = self.execute([{"command": "finish"}], origin="unverified")
        self.assertTrue(result["validated_execution"])
        self.assertFalse(result["validated_reference"])

    def test_finish_validates_and_error_does_not(self):
        _, result, _ = self.execute([{"command": "finish"}])
        self.assertTrue(result["validated_reference"])
        code, result, _ = self.execute([
            {"action": {"action_type": "navigate_back"}},
        ], fail_action=True)
        self.assertEqual(code, 1)
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertFalse(result["validated_reference"])

    def test_parser_rejects_ambiguous_or_missing_targets(self):
        for row in (
            {"action_type": "click"},
            {"action_type": "click", "index": 2, "x": 10, "y": 20},
            {"action_type": "swipe"},
            {"action_type": "input_text", "text": ""},
            {"action_type": "navigate_back", "private_field": "PRIVATE_VALUE"},
        ):
            with self.assertRaises(ValueError) as caught:
                harness.parse_command(json.dumps(row))
            self.assertNotIn("PRIVATE_VALUE", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
