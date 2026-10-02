#!/usr/bin/env python3
"""Offline tests for the LabServer study protocol and aggregate analysis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


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
    self.assertEqual(
        study._a11y_method_for_task(self.protocol, "SystemWifiTurnOffVerify"),
        "uiautomator",
    )
    self.assertEqual(
        study._a11y_method_for_task(self.protocol, "FilesDeleteFile"), "grpc"
    )
    self.assertEqual(len(self.protocol["tasks"]), 30)
    self.assertEqual(
        [row["name"] for row in self.protocol["arms"]],
        ["baseline", "probe_only", "graph_build_only",
         "graph_guided_exploration", "graph_prompt_inference",
         "graph_post_fusion", "graph_verified_skip",
         "graph_skip_allow_unknown_reversibility", "graph_skip_activity_landing"],
    )

  def test_a11y_overrides_must_be_in_cohort_and_supported(self) -> None:
    protocol = json.loads(json.dumps(self.protocol))
    protocol["controls"]["a11y_method_overrides"] = {"NotInCohort": "uiautomator"}
    with self.assertRaisesRegex(ValueError, "outside this cohort"):
      study._validate_protocol(protocol)

    protocol["controls"]["a11y_method_overrides"] = {
        "SystemWifiTurnOffVerify": "unknown"
    }
    with self.assertRaisesRegex(ValueError, "unsupported a11y method"):
      study._validate_protocol(protocol)

  def test_wifi_uia_protocol_is_a_matched_infrastructure_pilot(self) -> None:
    path = study.DEFAULT_PROTOCOL.with_name("protocol_wifi_uia_infra_pilot.json")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    study._validate_protocol(protocol)
    self.assertEqual(protocol["tasks"], ["SystemWifiTurnOffVerify"])
    self.assertEqual(
        [arm["name"] for arm in protocol["arms"]],
        ["baseline", "graph_guided_exploration"],
    )
    self.assertEqual(
        protocol["controls"]["a11y_method_overrides"],
        {"SystemWifiTurnOffVerify": "uiautomator"},
    )

  def test_worker_preflight_uses_network_independent_uiautomator(self) -> None:
    result = mock.Mock(
        returncode=0, stdout="androidworld-worker-preflight-ok 33"
    )
    with mock.patch.object(study.subprocess, "run", return_value=result) as run:
      study._preflight_worker("worker-test")

    code = run.call_args.args[0][-1]
    self.assertIn("A11yMethod.UIAUTOMATOR", code)
    self.assertNotIn("A11yMethod.A11Y_FORWARDER_APP", code)

  def test_shared_vllm_idle_requires_zero_running_and_waiting_requests(self) -> None:
    idle_metrics = {
        "vllm:request_success_total": 4.0,
        "vllm:prompt_tokens_total": 120.0,
        "vllm:generation_tokens_total": 64.0,
        "vllm:num_requests_running": 0.0,
        "vllm:num_requests_waiting": 0.0,
    }
    with mock.patch.object(
        study, "_metrics", side_effect=[(idle_metrics, ""), (idle_metrics, "")]
    ), mock.patch.object(study, "_endpoint_ok", return_value=True), mock.patch.object(
        study.time, "sleep"
    ):
      self.assertTrue(study._endpoint_idle(8095, 20))

    running_metrics = {**idle_metrics, "vllm:num_requests_running": 1.0}
    with mock.patch.object(
        study, "_metrics", return_value=(running_metrics, "")
    ), mock.patch.object(study, "_endpoint_ok", return_value=True), mock.patch.object(
        study.time, "sleep"
    ) as sleep:
      self.assertFalse(study._endpoint_idle(8095, 20))
      sleep.assert_not_called()

    with mock.patch.object(
        study, "_metrics", return_value=({"vllm:request_success_total": 0.0}, "")
    ), mock.patch.object(study, "_endpoint_ok", return_value=True):
      self.assertFalse(study._endpoint_idle(8095, 20))

  def test_shared_vllm_idle_rejects_requests_that_arrive_during_window(self) -> None:
    before = {
        "vllm:request_success_total": 4.0,
        "vllm:prompt_tokens_total": 120.0,
        "vllm:generation_tokens_total": 64.0,
        "vllm:num_requests_running": 0.0,
        "vllm:num_requests_waiting": 0.0,
    }
    after = {**before, "vllm:num_requests_waiting": 1.0}
    with mock.patch.object(
        study, "_metrics", side_effect=[(before, ""), (after, "")]
    ), mock.patch.object(study, "_endpoint_ok", return_value=True), mock.patch.object(
        study.time, "sleep"
    ):
      self.assertFalse(study._endpoint_idle(8095, 20))

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

  def test_trace_counter_separates_route_gate_reasons_and_skip_mechanisms(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      attempt = Path(tmp) / "attempt-01"
      output = attempt / "runner_output"
      output.mkdir(parents=True)
      study._append_jsonl(output / "step_latency.jsonl", {"inference_skipped": True})
      study._append_jsonl(output / "skip_events.jsonl", {
          "skip_kind": "graph_route", "successor_matched": True,
      })
      study._append_jsonl(output / "executable_memory_events.jsonl", {
          "event": "high_confidence_route_gate", "candidates": 2,
          "eligible_routes": 1,
          "blocks": {"passed": 1, "task_object_mismatch": 1},
      })
      action_dir = output / "private-task-label"
      action_dir.mkdir()
      study._append_jsonl(action_dir / "action.jsonl", {
          "reasoning_mode": "high_confidence_skip", "inference_skipped": True,
          "memory_route_length": 2,
      })
      counts = study._trace_counts(attempt)
      self.assertEqual(counts["inference_skipped_steps"], 2)
      self.assertEqual(counts["two_system_skip_route_attempts"], 1)
      self.assertEqual(counts["two_system_skip_route_hits"], 1)
      self.assertEqual(counts["executable_memory_skip_action_records"], 1)
      self.assertEqual(counts["high_confidence_route_gate_queries"], 1)
      self.assertEqual(counts["high_confidence_route_gate_candidates"], 2)
      self.assertEqual(counts["high_confidence_route_gate_eligible"], 1)
      self.assertEqual(counts["high_confidence_route_gate_blocks"], {
          "passed": 1, "task_object_mismatch": 1,
      })

  def test_supervisor_reexec_preserves_publish_policy(self) -> None:
    for flag, expected in (("--publish", True), ("--no-publish", False)):
      args = study.build_parser().parse_args([
          "run", "--run-root", "/tmp/study", "--run-id", "test", flag,
      ])
      resumed = study.build_parser().parse_args(study._resume_argv(args)[2:])
      self.assertEqual(resumed.publish, expected)

  def test_publication_is_allowlisted_and_idempotent(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      repo = root / "repo"
      remote = root / "origin.git"
      repo.mkdir()
      study._run(["git", "init", "--quiet", "--bare", str(remote)])
      study._run(["git", "init", "--quiet", "-b", "main", str(repo)])
      study._run(["git", "-C", str(repo), "config", "user.name", "AndroidWorld Test"])
      study._run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"])
      (repo / "README.md").write_text("test\n", encoding="utf-8")
      study._run(["git", "-C", str(repo), "add", "README.md"])
      study._run(["git", "-C", str(repo), "commit", "--quiet", "-m", "Initialize"])
      study._run(["git", "-C", str(repo), "remote", "add", "origin", str(remote)])
      study._run(["git", "-C", str(repo), "push", "--quiet", "-u", "origin", "main"])

      run_root = root / "study"
      run_root.mkdir()
      (run_root / "report.md").write_text("aggregate report\n", encoding="utf-8")
      (run_root / "aggregate_summary.json").write_text("{}\n", encoding="utf-8")
      (run_root / "private_trace.json").write_text("secret task data\n", encoding="utf-8")
      protocol = {
          "study": "test", "suite": "android_world", "tasks": ["TaskA"],
          "controls": {"task_seed": 34}, "arms": [{"name": "baseline"}],
          "primary_outcomes": ["success_rate"],
      }
      study.publish_report(repo, run_root, protocol, "test-run")
      published_commit = study._run([
          "git", "-C", str(repo), "rev-parse", "HEAD"], timeout=30).stdout.strip()
      study.publish_report(repo, run_root, protocol, "test-run")
      self.assertEqual(
          study._run(["git", "-C", str(repo), "rev-parse", "HEAD"], timeout=30).stdout.strip(),
          published_commit,
      )
      published = study._run([
          "git", "--git-dir", str(remote), "ls-tree", "-r", "--name-only", "main"],
          timeout=30).stdout.splitlines()
      prefix = "reports/labserver_androidworld_graph_study/test-run/"
      self.assertTrue(published)
      self.assertTrue(all(path.startswith(prefix) or path == "README.md" for path in published))
      self.assertFalse(any("private_trace" in path for path in published))

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
