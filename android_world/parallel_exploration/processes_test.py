"""Tests for phase-1 process lifecycle and timing authority."""

from android_world.parallel_exploration import processes
from android_world.parallel_exploration.protocol import EventKind


def _config() -> dict:
  return {
      "trial_id": "test-trial",
      "seed": 3,
      "prompt": "test",
      "trial_timeout_s": 5.0,
      "restore_timeout_s": 2.0,
      "inference": {
          "backend": "mock",
          "duration_s": 0.18,
          "token_count": 6,
          "first_token_fraction": 0.25,
          "decode_progress_every_n_tokens": 2,
      },
      "explorer": {"poll_interval_ms": 10, "mock_restore_ms": 40},
  }


def test_mock_trial_has_ordered_lifecycle_and_measures_restore_delay():
  trial = processes.run_trial(_config())
  kinds = [event.kind for event in trial.events]
  for required in (
      EventKind.EXPLORER_READY,
      EventKind.INFERENCE_START,
      EventKind.FIRST_TOKEN,
      EventKind.DECODE_PROGRESS,
      EventKind.INFERENCE_END,
      EventKind.ABORT,
      EventKind.RESTORED,
      EventKind.TRIAL_COMPLETE,
  ):
    assert required in kinds
  assert kinds.index(EventKind.INFERENCE_START) < kinds.index(EventKind.FIRST_TOKEN)
  assert kinds.index(EventKind.FIRST_TOKEN) < kinds.index(EventKind.INFERENCE_END)
  assert kinds.index(EventKind.INFERENCE_END) < kinds.index(EventKind.ABORT)
  assert kinds.index(EventKind.ABORT) < kinds.index(EventKind.TRIAL_COMPLETE)
  assert trial.result["restore_status"] == "RESTORED"
  assert trial.result["critical_path_extension_ms"] >= 35
  assert trial.result["critical_path_extension_ms"] < 500
  assert trial.result["abort_during_work"] is True


def test_restore_failure_is_reported_without_being_hidden():
  config = _config()
  config["explorer"]["mock_restore_failure"] = True
  trial = processes.run_trial(config)
  assert trial.result["restore_status"] == "RESTORE_FAILED"
  assert any(event.kind == EventKind.RESTORE_FAILED for event in trial.events)
