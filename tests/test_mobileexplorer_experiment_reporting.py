from scripts.run_semantic_prefix_ex5_experiment import _aggregate
from scripts.run_semantic_prefix_ex5_experiment import _task_metrics


def test_a11y_startup_skip_is_infrastructure_unknown_not_task_failure(tmp_path):
  row = _task_metrics(
      tmp_path,
      "Could not get a11y tree.\n"
      "SKIPPING SampleTask.\n",
      "SampleTask",
  )

  assert row["success"] is None
  assert row["evaluation_status"] == "infrastructure_unknown"
  assert "accessibility tree unavailable" in row["evaluator_error"]


def test_unknown_infrastructure_rows_do_not_enter_step_or_success_denominators():
  summary = _aggregate("A", [
      {"task": "ok", "success": 1.0, "steps": 4, "step_latency_s": [2.0, 3.0]},
      {"task": "infra", "success": None, "steps": 0, "step_latency_s": []},
  ])

  assert summary["task_count"] == 2
  assert summary["evaluated_task_count"] == 1
  assert summary["unknown_evaluation_count"] == 1
  assert summary["success_rate"] == 1.0
  assert summary["step_mean"] == 4.0
  assert summary["step_median"] == 4.0
  assert summary["step_latency_mean_s"] == 2.5
