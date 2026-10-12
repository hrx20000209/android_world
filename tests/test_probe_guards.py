"""Exploration guards added after r5 (ISSUES #9, #10)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_serial_exploration_task.py"


def _runner():
  spec = importlib.util.spec_from_file_location("run_serial_exploration_task", _PATH)
  module = importlib.util.module_from_spec(spec)
  sys.modules.setdefault("run_serial_exploration_task", module)
  spec.loader.exec_module(module)
  return module


def el(rid="", cls="android.widget.TextView", text=""):
  return NS(resource_id=rid, class_name=cls, text=text)


def test_date_and_time_pickers_block_probing():
  runner = _runner()
  date_dialog = NS(elements=(el("android:id/date_picker_header_date", text="Thu, Oct 19"),
                             el("android:id/day_picker_view_pager",
                                "com.android.internal.widget.ViewPager")))
  time_dialog = NS(elements=(el("android:id/radial_picker", "android.widget.RadialTimePickerView"),))
  number = NS(elements=(el("", "android.widget.NumberPicker"),))
  form = NS(elements=(el("com.app:id/edt_name", "android.widget.EditText"),
                      el("com.app:id/button_save")))
  assert runner._has_value_picker(date_dialog)
  assert runner._has_value_picker(time_dialog)
  assert runner._has_value_picker(number)
  assert not runner._has_value_picker(form)
