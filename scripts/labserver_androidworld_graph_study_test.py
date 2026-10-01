#!/usr/bin/env python3
"""Offline tests for the LabServer study protocol and aggregate analysis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("labserver_androidworld_graph_study.py")
SPEC = importlib.util.spec_from_file_location("aw_graph_study", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class StudyTests(unittest.TestCase):

  def setUp(self) -> None:
    self.protocol = json.loads(study.DEFAULT_PROTOCOL.read_text(encoding="utf-8"))

  def test_protocol_freezes_decoding_and_has_full_factorial_arms(self) -> None:
    study._validate_protocol(self.protocol)
    self.assertEqual(self.protocol["controls"]["client_temperature"], 0.0)
    self.assertEqual(self.protocol["controls"]["client_top_p"], 1.0)
    self.assertEqual(self.protocol["controls"]["a11y_method"], "grpc")
    self.assertEqual(len(self.protocol["tasks"]), 30)
    self.assertEqual(
        [row["name"] for row in self.protocol["arms"]],
        ["baseline", "probe_only", "graph_build_only",
         "graph_guided_exploration", "graph_prompt_inference",
         "graph_post_fusion", "graph_verified_skip",
         "graph_skip_allow_unknown_reversibility", "graph_skip_activity_landing"],
    )

  def test_paired_metrics_use_common_successes_for_success_steps(self) -> None:
    protocol = {
        "tasks": ["A", "B", "C"],
        "arms": [{"name": "baseline"}, {"name": "graph"}],
    }
    rows = [
        {"status": "complete", "task": "A", "arm": "baseline", "success": True,
         "evaluator_complete": True, "episode_steps": 10},
        {"status": "complete", "task": "B", "arm": "baseline", "success": False,
         "evaluator_complete": True, "episode_steps": 20},
        {"status": "complete", "task": "C", "arm": "baseline", "success": True,
         "evaluator_complete": True, "episode_steps": 14},
        {"status": "complete", "task": "A", "arm": "graph", "success": True,
         "evaluator_complete": True, "episode_steps": 7, "skip_hits": 1},
        {"status": "complete", "task": "B", "arm": "graph", "success": True,
         "evaluator_complete": True, "episode_steps": 13, "graph_delta_nodes": 2},
        {"status": "complete", "task": "C", "arm": "graph", "success": False,
         "evaluator_complete": True, "episode_steps": 15},
    ]
    summary = study._summarize(rows, protocol)
    graph = summary["arms"]["graph"]
    self.assertAlmostEqual(graph["success_rate"], 2 / 3)
    self.assertEqual(graph["total_steps_all_completed"], 35)
    contrast = summary["paired_vs_baseline"]["graph"]
    self.assertEqual(contrast["paired_n"], 3)
    self.assertEqual(contrast["common_success_n"], 1)
    self.assertEqual(contrast["paired_common_success_steps_delta_variant_minus_baseline"], -3.0)

  def test_checkpointed_evaluator_exceptions_count_as_failures(self) -> None:
    protocol = {
        "tasks": ["A", "B"],
        "arms": [{"name": "baseline"}, {"name": "graph"}],
    }
    rows = [
        {"status": "complete", "task": "A", "arm": "baseline", "success": True,
         "evaluator_complete": True, "episode_steps": 8, "episode_exception": False},
        {"status": "complete", "task": "B", "arm": "baseline", "success": False,
         "evaluator_complete": False, "episode_steps": None, "episode_exception": True},
        {"status": "complete", "task": "A", "arm": "graph", "success": True,
         "evaluator_complete": True, "episode_steps": 6, "episode_exception": False},
        {"status": "complete", "task": "B", "arm": "graph", "success": False,
         "evaluator_complete": False, "episode_steps": None, "episode_exception": True},
    ]
    summary = study._summarize(rows, protocol)
    self.assertEqual(summary["arms"]["baseline"]["success_rate"], 0.5)
    self.assertEqual(summary["arms"]["baseline"]["n_evaluator_complete"], 1)
    self.assertEqual(summary["arms"]["baseline"]["n_episode_exceptions"], 1)
    self.assertEqual(summary["paired_vs_baseline"]["graph"]["paired_n"], 2)
    self.assertEqual(summary["paired_vs_baseline"]["graph"]["paired_step_n"], 1)

  def test_pre_action_exception_without_model_request_is_retryable_infrastructure(self) -> None:
    infra = {"episode_exception": True, "episode_steps": 0}
    self.assertTrue(study._is_pre_action_infra_failure(infra, 0, True))
    self.assertFalse(study._is_pre_action_infra_failure(infra, 1, True))
    self.assertFalse(study._is_pre_action_infra_failure(infra, 0, False))
    self.assertFalse(study._is_pre_action_infra_failure(
        {"episode_exception": False, "episode_steps": 0}, 0, True))
    self.assertFalse(study._is_pre_action_infra_failure(
        {"episode_exception": True, "episode_steps": 1}, 0, True))

  def test_trace_counter_separates_mutation_and_overlapping_ingest_wall_time(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      study._append_jsonl(root / "graph_construction_perf.jsonl", {
          "operation": "observe_page", "elapsed_s": 0.002,
      })
      study._append_jsonl(root / "graph_construction_perf.jsonl", {
          "operation": "record_transition", "elapsed_s": 0.003,
      })
      study._append_jsonl(root / "graph_construction_perf.jsonl", {
          "operation": "probe_trace_ingest_wall", "elapsed_s": 0.010,
      })
      counts = study._trace_counts(root)
      self.assertEqual(counts["graph_update_calls"], 2)
      self.assertAlmostEqual(counts["graph_construction_time_s"], 0.005)
      self.assertAlmostEqual(counts["probe_trace_ingest_wall_s"], 0.010)

  def test_online_runner_uses_private_child_output_and_checkpoint_path(self) -> None:
    self.assertEqual(
        study._checkpoint_container_path("TaskA", "probe_only", "attempt-02", "run.py"),
        "/study/episodes/TaskA/probe_only/attempt-02/checkpoints",
    )
    self.assertEqual(
        study._checkpoint_container_path(
            "TaskA", "probe_only", "attempt-02", "run_sensys30_online_task.py"),
        "/study/episodes/TaskA/probe_only/attempt-02/runner_output/checkpoints",
    )

  def test_analyzer_writes_aggregate_only_visual_artifacts(self) -> None:
    protocol = {
        "study": "unit test", "tasks": ["A"], "primary_outcomes": [],
        "controls": {"task_seed": 34},
        "arms": [{"name": "baseline"}, {"name": "graph_build_only"}],
    }
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      study._append_jsonl(root / "records.jsonl", {
          "status": "complete", "task": "A", "arm": "baseline", "seed": 34,
          "success": True, "evaluator_complete": True, "episode_steps": 9,
      })
      study._append_jsonl(root / "records.jsonl", {
          "status": "complete", "task": "A", "arm": "graph_build_only", "seed": 34,
          "success": True, "evaluator_complete": True, "episode_steps": 7,
      })
      study.analyze(root, protocol)
      for name in ("report.md", "aggregate_summary.json", "per_task_metrics.csv",
                   "success_rate.svg", "mean_steps.svg", "success_steps.svg"):
        self.assertTrue((root / name).is_file(), name)
      report = (root / "report.md").read_text(encoding="utf-8")
      self.assertIn("jointly successful", report)
      self.assertNotIn("buckwheat groats", report)


if __name__ == "__main__":
  unittest.main()
