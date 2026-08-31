"""One definition of the per-task step budget, shared by every arm.

Rule:
  - A task that some baseline actually finished but needed more than 20 steps
    for gets 20. Capping those lower would measure the cap rather than the
    method.
  - Everything else gets 15, which is the budget the method is meant to beat.
  - Never above the task's own AndroidWorld default: a task whose native
    budget is 10 steps does not become easier by being handed 15.

Both arms import this, so a budget mismatch cannot quietly favour one of them
the way it did on 2026-08-31, when the baseline had been run under an older,
wider rule and took 16 steps on a task the other arm was allowed 15.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_DEFAULTS = json.loads((REPO / "configs/task_default_steps.json").read_text())


def budget(task: str, ref_success: int, ref_steps: int,
           measured_success: int = 0, measured_steps: int = 0) -> int:
  """Steps allowed for `task`. ref_* is 4B_2; measured_* is our own baseline."""
  long_success = (
      (int(ref_success) == 1 and int(ref_steps) > 20)
      or (int(measured_success) == 1 and int(measured_steps) > 20)
  )
  cap = 20 if long_success else 15
  return min(cap, int(_DEFAULTS.get(task, cap)))
