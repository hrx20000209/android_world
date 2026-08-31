from android_world.parallel_exploration.recovery import InverseActionPlanner
from android_world.parallel_exploration.recovery import InverseKind
from android_world.parallel_exploration.recovery import execute_inverse
from android_world.parallel_exploration.recovery import replay_navigation_trajectory


class FakeAdb:
  def __init__(self):
    self.commands = []

  def run(self, command, timeout_s=0):
    self.commands.append((command, timeout_s))


def test_scroll_inverse_reverses_exact_gesture():
  entry = InverseActionPlanner().plan({
      "action_type": "SWIPE",
      "start_coordinate": [10, 90], "end_coordinate": [10, 20],
  })
  assert entry.inverse_kind == InverseKind.REVERSE_SCROLL
  assert entry.inverse_action["start_coordinate"] == [10, 20]
  assert entry.inverse_action["end_coordinate"] == [10, 90]
  adb = FakeAdb()
  execute_inverse(adb, entry)
  assert adb.commands[0][0][2:8] == ["swipe", "10", "20", "10", "90", "180"]


def test_checkable_click_inverse_is_retap():
  entry = InverseActionPlanner().plan(
      {"action_type": "CLICK", "x": 5, "y": 6},
      pre_checked=False, post_checked=True,
  )
  assert entry.inverse_kind == InverseKind.RETAP_TOGGLE
  assert entry.semantic_reversible


def test_plain_click_inverse_is_back_but_not_semantic_certificate():
  entry = InverseActionPlanner().plan({"action_type": "CLICK", "x": 5, "y": 6})
  assert entry.inverse_kind == InverseKind.BACK
  assert not entry.semantic_reversible


def test_selected_tab_inverse_retaps_previous_selection():
  entry = InverseActionPlanner().plan(
      {"action_type": "CLICK", "x": 50, "y": 60},
      selection_inverse_action={"action_type": "CLICK", "x": 10, "y": 20},
  )
  assert entry.inverse_kind == InverseKind.RETAP_SELECTION
  assert entry.inverse_action["x"] == 10
  assert entry.semantic_reversible


def test_replay_reconstructs_committed_click_and_input_actions():
  adb = FakeAdb()
  result = replay_navigation_trajectory(adb, [
      {"action_dict": {"action_type": "swipe"}, "tool_call": {"arguments": {"start_coordinate": [1, 9], "end_coordinate": [1, 2]}}},
      {"action_dict": {"action_type": "click", "x": 3, "y": 4}},
      {"action_dict": {"action_type": "input_text", "text": "secret"}},
  ])
  assert result == {"replayed": 3, "skipped_non_idempotent": 0}
  assert any("secret" in command for command, _ in adb.commands)
