"""Dual-instance isolation and deadline-aware admission for GUI probes.

This is an emulator prototype boundary, not a claim of phone-level VM
isolation.  The controller never receives a primary action executor: probes
can only mutate the shadow instance and evidence is accepted only after
prefix/state checks.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any


class InferencePhase(str, enum.Enum):
  IDLE = "idle"
  PREFILL = "prefill"
  DECODE = "decode"
  COMPLETE = "complete"


class ProbeStatus(str, enum.Enum):
  ACCEPTED = "accepted"
  REJECTED_STATE_MISMATCH = "rejected_state_mismatch"
  REJECTED_DEADLINE = "rejected_deadline"
  REJECTED_POLICY = "rejected_policy"
  CANCELLED = "cancelled"
  PROBE_FAILED = "probe_failed"
  PRIMARY_CHANGED = "primary_changed"


@dataclasses.dataclass(frozen=True)
class InstanceState:
  activity: str
  ui_structure_hash: str
  evaluator_state_hash: str
  action_prefix_hash: str

  def matches(self, other: "InstanceState") -> bool:
    return dataclasses.astuple(self) == dataclasses.astuple(other)


@dataclasses.dataclass(frozen=True)
class ProbeTransaction:
  primary_state_id: str
  shadow_state_id: str
  deadline_s: float
  isolation_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)


@dataclasses.dataclass(frozen=True)
class ProbeOutcome:
  status: ProbeStatus
  probe_latency_s: float
  rebuild_latency_s: float
  extraction_latency_s: float
  state_match: bool
  deadline_miss: bool
  isolation_id: str
  observed_fact: Mapping[str, Any] | None = None
  detail: str = ""

  def to_dict(self) -> dict[str, Any]:
    row = dataclasses.asdict(self)
    row["status"] = self.status.value
    return row

  @classmethod
  def from_dict(cls, value: Mapping[str, Any]) -> "ProbeOutcome":
    return cls(
        status=ProbeStatus(str(value["status"])),
        probe_latency_s=float(value["probe_latency_s"]),
        rebuild_latency_s=float(value["rebuild_latency_s"]),
        extraction_latency_s=float(value["extraction_latency_s"]),
        state_match=bool(value["state_match"]),
        deadline_miss=bool(value["deadline_miss"]),
        isolation_id=str(value["isolation_id"]),
        observed_fact=value.get("observed_fact"),
        detail=str(value.get("detail", "")),
    )


@dataclasses.dataclass(frozen=True)
class ResourceSnapshot:
  monotonic_s: float
  cpu_psi_some_percent: float
  memory_psi_some_percent: float
  thermal_level: int
  inference_phase: InferencePhase


@dataclasses.dataclass(frozen=True)
class AdmissionDecision:
  admitted: bool
  reason: str
  deadline_s: float


class PhaseAwareProbeScheduler:
  """Admission controller informed by measured phone contention.

  Prior measurements in ``~/Projects/agent`` show GUI exploration increases
  prefill by roughly 39% while graph maintenance is a much smaller marginal
  cost.  Consequently prefill is a hard admission stop; decode admits only
  probes predicted to finish before the drain margin and under CPU pressure.
  """

  def __init__(
      self,
      *,
      cpu_psi_limit_percent: float = 18.0,
      thermal_limit: int = 3,
      drain_margin_s: float = 0.25,
  ) -> None:
    self.cpu_psi_limit_percent = cpu_psi_limit_percent
    self.thermal_limit = thermal_limit
    self.drain_margin_s = drain_margin_s
    self._cost_multiplier = 1.0

  def admit(
      self,
      snapshot: ResourceSnapshot,
      *,
      inference_deadline_s: float,
      predicted_probe_s: float,
      now_s: float | None = None,
  ) -> AdmissionDecision:
    now = time.monotonic() if now_s is None else now_s
    if snapshot.inference_phase != InferencePhase.DECODE:
      return AdmissionDecision(False, f"phase_{snapshot.inference_phase.value}", inference_deadline_s)
    if snapshot.cpu_psi_some_percent > self.cpu_psi_limit_percent:
      return AdmissionDecision(False, "cpu_pressure", inference_deadline_s)
    if snapshot.thermal_level >= self.thermal_limit:
      return AdmissionDecision(False, "thermal_pressure", inference_deadline_s)
    predicted_end = now + predicted_probe_s * self._cost_multiplier
    if predicted_end + self.drain_margin_s > inference_deadline_s:
      return AdmissionDecision(False, "deadline_drain", inference_deadline_s)
    return AdmissionDecision(True, "decode_slack", inference_deadline_s)

  def observe(self, *, predicted_s: float, actual_s: float, deadline_miss: bool) -> None:
    """AIMD-style adaptation: miss quickly, recover cautiously."""
    if deadline_miss or actual_s > predicted_s * self._cost_multiplier:
      self._cost_multiplier = min(4.0, self._cost_multiplier * 1.25)
    else:
      self._cost_multiplier = max(1.0, self._cost_multiplier - 0.05)


class ShadowIsolationController:
  """Executes one speculative transaction exclusively on the shadow."""

  def __init__(
      self,
      *,
      capture_primary: Callable[[], InstanceState],
      capture_shadow: Callable[[], InstanceState],
      rebuild_shadow: Callable[[Sequence[Mapping[str, Any]]], None],
  ) -> None:
    self._capture_primary = capture_primary
    self._capture_shadow = capture_shadow
    self._rebuild_shadow = rebuild_shadow

  def run_probe(
      self,
      transaction: ProbeTransaction,
      *,
      authoritative_prefix: Sequence[Mapping[str, Any]],
      execute_shadow_probe: Callable[[threading.Event], Any],
      extract_evidence: Callable[[Any], Mapping[str, Any]],
      cancel: threading.Event | None = None,
      now: Callable[[], float] = time.monotonic,
  ) -> ProbeOutcome:
    cancel_event = cancel or threading.Event()
    initial_primary = self._capture_primary()
    initial_shadow = self._capture_shadow()
    if not initial_primary.matches(initial_shadow):
      return self._outcome(
          ProbeStatus.REJECTED_STATE_MISMATCH, transaction, detail="prefix/state mismatch before probe"
      )
    if cancel_event.is_set() or now() >= transaction.deadline_s:
      return self._outcome(ProbeStatus.CANCELLED, transaction, deadline_miss=True)

    probe_started = now()
    try:
      observation = execute_shadow_probe(cancel_event)
    except Exception as exc:  # A shadow crash is data, not a primary failure.
      probe_latency = now() - probe_started
      rebuild_latency = self._rebuild(authoritative_prefix, now)
      return ProbeOutcome(
          status=ProbeStatus.PROBE_FAILED,
          probe_latency_s=probe_latency,
          rebuild_latency_s=rebuild_latency,
          extraction_latency_s=0.0,
          state_match=False,
          deadline_miss=now() >= transaction.deadline_s,
          isolation_id=transaction.isolation_id,
          detail=f"{type(exc).__name__}: {exc}",
      )

    probe_latency = now() - probe_started
    if cancel_event.is_set() or now() >= transaction.deadline_s:
      rebuild_latency = self._rebuild(authoritative_prefix, now)
      return ProbeOutcome(
          status=ProbeStatus.CANCELLED,
          probe_latency_s=probe_latency,
          rebuild_latency_s=rebuild_latency,
          extraction_latency_s=0.0,
          state_match=False,
          deadline_miss=now() >= transaction.deadline_s,
          isolation_id=transaction.isolation_id,
      )

    extraction_started = now()
    fact = dict(extract_evidence(observation))
    extraction_latency = now() - extraction_started
    primary_after = self._capture_primary()
    primary_unchanged = initial_primary.matches(primary_after)
    rebuild_latency = self._rebuild(authoritative_prefix, now)
    rebuilt_match = initial_primary.matches(self._capture_shadow())
    deadline_miss = now() >= transaction.deadline_s
    if not primary_unchanged:
      status = ProbeStatus.PRIMARY_CHANGED
    elif not rebuilt_match:
      status = ProbeStatus.REJECTED_STATE_MISMATCH
    elif deadline_miss:
      status = ProbeStatus.REJECTED_DEADLINE
    else:
      status = ProbeStatus.ACCEPTED
    return ProbeOutcome(
        status=status,
        probe_latency_s=probe_latency,
        rebuild_latency_s=rebuild_latency,
        extraction_latency_s=extraction_latency,
        state_match=primary_unchanged and rebuilt_match,
        deadline_miss=deadline_miss,
        isolation_id=transaction.isolation_id,
        observed_fact=fact if status == ProbeStatus.ACCEPTED else None,
    )

  def _rebuild(
      self, prefix: Sequence[Mapping[str, Any]], now: Callable[[], float]
  ) -> float:
    started = now()
    self._rebuild_shadow(prefix)
    return now() - started

  @staticmethod
  def _outcome(
      status: ProbeStatus,
      transaction: ProbeTransaction,
      *,
      deadline_miss: bool = False,
      detail: str = "",
  ) -> ProbeOutcome:
    return ProbeOutcome(
        status=status,
        probe_latency_s=0.0,
        rebuild_latency_s=0.0,
        extraction_latency_s=0.0,
        state_match=False,
        deadline_miss=deadline_miss,
        isolation_id=transaction.isolation_id,
        detail=detail,
    )


def summarize_probe_outcomes(outcomes: Sequence[ProbeOutcome]) -> dict[str, Any]:
  """Produce the feasibility panel required by the isolation experiment."""
  if not outcomes:
    raise ValueError("No probe outcomes were provided")

  def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
      return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

  def distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50_s": percentile(values, 0.5),
        "p95_s": percentile(values, 0.95),
    }

  accepted = sum(x.status == ProbeStatus.ACCEPTED for x in outcomes)
  return {
      "n": len(outcomes),
      "status_counts": {
          status.value: sum(x.status == status for x in outcomes)
          for status in ProbeStatus
      },
      "probe": distribution([x.probe_latency_s for x in outcomes]),
      "shadow_resync": distribution([x.rebuild_latency_s for x in outcomes]),
      "evidence_extraction": distribution(
          [x.extraction_latency_s for x in outcomes]
      ),
      "accepted_within_window_rate": accepted / len(outcomes),
      "deadline_miss_rate": sum(x.deadline_miss for x in outcomes) / len(outcomes),
      "state_match_rate": sum(x.state_match for x in outcomes) / len(outcomes),
  }
