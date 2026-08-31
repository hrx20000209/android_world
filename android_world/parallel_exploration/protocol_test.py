"""Unit tests for cross-process protocol messages."""

from android_world.parallel_exploration.protocol import Event
from android_world.parallel_exploration.protocol import EventKind
from android_world.parallel_exploration.protocol import ProcessRole
from android_world.parallel_exploration.protocol import require_event


def test_event_serializes_with_monotonic_timestamp():
  event = Event.now(
      "trial-1", EventKind.DECODE_PROGRESS, ProcessRole.INFERENCE,
      {"token_count": 5, "partial_text": "hello"}
  )
  assert event.monotonic_ns > 0
  assert event.to_dict()["event"] == "DECODE_PROGRESS"
  assert require_event(event, EventKind.DECODE_PROGRESS) is event
