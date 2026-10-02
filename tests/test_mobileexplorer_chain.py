from scripts.run_sensys30_online_task import (
    _deterministic_bootstrap_target,
    _use_fast_a11y_socket,
)


def test_open_app_reasoning_skip_is_only_for_known_target_on_launcher():
  goal = "Start the stopwatch in the Clock app"
  assert _deterministic_bootstrap_target(
      goal, "com.google.android.apps.nexuslauncher/.NexusLauncherActivity", True
  ) == "Clock"
  assert _deterministic_bootstrap_target(
      goal, "com.google.android.deskclock/.ClockActivity", True
  ) is None
  assert _deterministic_bootstrap_target(
      "Do something in an unknown application",
      "com.google.android.apps.nexuslauncher/.NexusLauncherActivity", True
  ) is None
  assert _deterministic_bootstrap_target(
      goal, "com.google.android.apps.nexuslauncher/.NexusLauncherActivity", False
  ) is None


def test_uiautomator_override_also_controls_graph_skip_capture():
  assert not _use_fast_a11y_socket("uiautomator")
  assert _use_fast_a11y_socket("grpc")
  assert _use_fast_a11y_socket("fast_provider")
