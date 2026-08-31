"""Typed cross-process event protocol for the exploration harness."""

from __future__ import annotations

import dataclasses
import enum
import time
from typing import Any, Mapping


class EventKind(str, enum.Enum):
  """All protocol messages exchanged or recorded in phase 1."""

  INFERENCE_START = "INFERENCE_START"
  FIRST_TOKEN = "FIRST_TOKEN"
  DECODE_PROGRESS = "DECODE_PROGRESS"
  INFERENCE_END = "INFERENCE_END"
  ABORT = "ABORT"
  EXPLORER_READY = "EXPLORER_READY"
  RESTORED = "RESTORED"
  RESTORE_FAILED = "RESTORE_FAILED"
  TRIAL_COMPLETE = "TRIAL_COMPLETE"
  ERROR = "ERROR"


class ProcessRole(str, enum.Enum):
  INFERENCE = "inference"
  EXPLORER = "explorer"
  RUNNER = "runner"


@dataclasses.dataclass(frozen=True)
class Event:
  """A pickle-safe message carrying a sender-monotonic timestamp."""

  trial_id: str
  kind: EventKind
  sender: ProcessRole
  monotonic_ns: int
  payload: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  @classmethod
  def now(
      cls,
      trial_id: str,
      kind: EventKind,
      sender: ProcessRole,
      payload: Mapping[str, Any] | None = None,
  ) -> "Event":
    return cls(
        trial_id=trial_id,
        kind=kind,
        sender=sender,
        monotonic_ns=time.monotonic_ns(),
        payload=dict(payload or {}),
    )

  def to_dict(self) -> dict[str, Any]:
    return {
        "trial_id": self.trial_id,
        "event": self.kind.value,
        "sender": self.sender.value,
        "monotonic_ns": self.monotonic_ns,
        "payload": dict(self.payload),
    }


def require_event(value: object, *allowed: EventKind) -> Event:
  """Validate an object received from a multiprocessing queue."""
  if not isinstance(value, Event):
    raise TypeError(f"Expected Event, got {type(value).__name__}")
  if allowed and value.kind not in allowed:
    names = ", ".join(event.value for event in allowed)
    raise ValueError(f"Expected one of [{names}], got {value.kind.value}")
  return value
