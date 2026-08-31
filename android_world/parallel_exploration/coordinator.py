"""GUI ownership, speculation barrier, and recovery records."""

from __future__ import annotations

import dataclasses
import enum
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from android_world.parallel_exploration.safety import StateVerifier


class Ownership(str, enum.Enum):
  MAIN = "MAIN"
  EXPLORER = "EXPLORER"
  RECOVERING = "RECOVERING"


@dataclasses.dataclass(frozen=True)
class BarrierResult:
  restored: bool
  verification_confidence: float
  recovery_latency_ms: float
  preempted: bool
  unfinished_exploration: bool


class SpeculationCoordinator:
  def __init__(self, verifier: StateVerifier | None = None):
    self._verifier = verifier or StateVerifier()
    self._lock = threading.RLock()
    self._stop = threading.Event()
    self.owner = Ownership.MAIN
    self.committed_state: dict[str, Any] | None = None
    self.preemption_count = 0

  @property
  def stop_requested(self) -> bool:
    return self._stop.is_set()

  def begin_speculation(self, committed_state: Mapping[str, Any]) -> None:
    with self._lock:
      if self.owner != Ownership.MAIN:
        raise RuntimeError(f"GUI is owned by {self.owner.value}")
      self.committed_state = dict(committed_state)
      self._stop.clear()
      self.owner = Ownership.EXPLORER

  def request_preemption(self) -> None:
    self.preemption_count += 1
    self._stop.set()

  def barrier(self, recover: Callable[[], None], capture: Callable[[], Mapping[str, Any]], *, unfinished: bool = False) -> BarrierResult:
    self.request_preemption()
    started = time.monotonic_ns()
    with self._lock:
      self.owner = Ownership.RECOVERING
      recover()
      actual = dict(capture())
      expected = self.committed_state or {}
      result = self._verifier.verify(expected, actual)
      self.owner = Ownership.MAIN
    return BarrierResult(
        restored=result.semantic_recovered,
        verification_confidence=result.confidence,
        recovery_latency_ms=(time.monotonic_ns() - started) / 1e6,
        preempted=True,
        unfinished_exploration=unfinished,
    )

  def commit_main_action(self, execute: Callable[[], Any]) -> Any:
    with self._lock:
      if self.owner != Ownership.MAIN:
        raise RuntimeError("Main action cannot commit before speculation barrier")
      return execute()


class SpeculationBarrier:
  def __init__(self, coordinator: SpeculationCoordinator):
    self.coordinator = coordinator

  def stop_recover_verify(self, recover: Callable[[], None], capture: Callable[[], Mapping[str, Any]], *, unfinished: bool = False) -> BarrierResult:
    return self.coordinator.barrier(recover, capture, unfinished=unfinished)

