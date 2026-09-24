"""Lifecycle joining evidence graph, scheduler, and shadow isolation."""

from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from android_world.parallel_exploration.agent_evidence_graph import AgentEvidenceGraph
from android_world.parallel_exploration.agent_evidence_graph import DecisionContext
from android_world.parallel_exploration.agent_evidence_graph import EvidenceRecord
from android_world.parallel_exploration.agent_evidence_graph import EvidenceScope
from android_world.parallel_exploration.agent_evidence_graph import ProbeCandidate
from android_world.parallel_exploration.shadow_isolation import AdmissionDecision
from android_world.parallel_exploration.shadow_isolation import PhaseAwareProbeScheduler
from android_world.parallel_exploration.shadow_isolation import ProbeOutcome
from android_world.parallel_exploration.shadow_isolation import ProbeStatus
from android_world.parallel_exploration.shadow_isolation import ProbeTransaction
from android_world.parallel_exploration.shadow_isolation import ResourceSnapshot
from android_world.parallel_exploration.shadow_isolation import ShadowIsolationController


@dataclasses.dataclass(frozen=True)
class ExtractedFact:
  slot: str
  value: str
  confidence: float
  ttl_s: float
  extraction_verified: bool

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, value: Mapping[str, Any]) -> "ExtractedFact":
    return cls(
        slot=str(value["slot"]),
        value=str(value["value"]),
        confidence=float(value["confidence"]),
        ttl_s=float(value["ttl_s"]),
        extraction_verified=bool(value["extraction_verified"]),
    )


@dataclasses.dataclass(frozen=True)
class ProbeAttempt:
  admission: AdmissionDecision
  candidate: ProbeCandidate | None
  outcome: ProbeOutcome | None
  evidence_id: str | None


class OnlineExplorationRuntime:
  """One-probe runtime with explicit generation and isolation barriers."""

  def __init__(
      self,
      *,
      graph: AgentEvidenceGraph,
      scheduler: PhaseAwareProbeScheduler,
      isolation: ShadowIsolationController,
  ) -> None:
    self.graph = graph
    self.scheduler = scheduler
    self.isolation = isolation

  def try_probe(
      self,
      context: DecisionContext,
      candidates: Sequence[ProbeCandidate],
      *,
      resource_snapshot: ResourceSnapshot,
      inference_deadline_s: float,
      authoritative_prefix: Sequence[Mapping[str, Any]],
      shadow_state_id: str,
      execute_shadow: Callable[[ProbeCandidate, threading.Event], Any],
      extract_fact: Callable[[Any], ExtractedFact],
      cancel: threading.Event | None = None,
  ) -> ProbeAttempt:
    slack_s = max(0.0, inference_deadline_s - time.monotonic())
    summary = self.graph.summarize_for_exploration(
        context, candidates, slack_s=slack_s, maximum_candidates=1
    )
    if not summary.candidates:
      rejection = next(iter(summary.rejected.values()), "no_unresolved_slot")
      return ProbeAttempt(
          admission=AdmissionDecision(False, rejection, inference_deadline_s),
          candidate=None,
          outcome=None,
          evidence_id=None,
      )
    candidate = summary.candidates[0]
    predicted_s = (
        candidate.probe_latency_s
        + candidate.extraction_latency_s
        + candidate.resync_latency_s
    )
    admission = self.scheduler.admit(
        resource_snapshot,
        inference_deadline_s=inference_deadline_s,
        predicted_probe_s=predicted_s,
    )
    if not admission.admitted:
      return ProbeAttempt(admission, candidate, None, None)

    transaction = ProbeTransaction(
        primary_state_id=context.state_id,
        shadow_state_id=shadow_state_id,
        deadline_s=inference_deadline_s,
    )
    outcome = self.isolation.run_probe(
        transaction,
        authoritative_prefix=authoritative_prefix,
        execute_shadow_probe=lambda event: execute_shadow(candidate, event),
        extract_evidence=lambda observation: extract_fact(observation).to_dict(),
        cancel=cancel,
    )
    actual_s = (
        outcome.probe_latency_s
        + outcome.extraction_latency_s
        + outcome.rebuild_latency_s
    )
    self.scheduler.observe(
        predicted_s=predicted_s,
        actual_s=actual_s,
        deadline_miss=outcome.deadline_miss,
    )
    self.graph.record_probe_outcome(
        context.state_id,
        candidate.action,
        resolved_slot=outcome.status == ProbeStatus.ACCEPTED,
        calls_saved=0.0,
        actions_saved=0.0,
        probe_latency_s=outcome.probe_latency_s,
        interference_cost_s=candidate.interference_penalty_s,
        safe=outcome.state_match,
    )
    if outcome.status != ProbeStatus.ACCEPTED or outcome.observed_fact is None:
      return ProbeAttempt(admission, candidate, outcome, None)

    fact = ExtractedFact.from_dict(outcome.observed_fact)
    if fact.slot not in context.required_slots or fact.slot not in candidate.target_slots:
      rejected = dataclasses.replace(
          outcome,
          status=ProbeStatus.REJECTED_POLICY,
          observed_fact=None,
          detail="extractor returned a slot outside the admitted task need",
      )
      return ProbeAttempt(admission, candidate, rejected, None)
    evidence_id = uuid.uuid4().hex
    self.graph.add_evidence(EvidenceRecord(
        evidence_id=evidence_id,
        slot=fact.slot,
        value=fact.value,
        source_action=candidate.action,
        source_state_id=context.state_id,
        progress_signature=context.progress_signature,
        task_need=context.task_need,
        observed_at_s=time.time(),
        ttl_s=fact.ttl_s,
        confidence=fact.confidence,
        scope=EvidenceScope.CURRENT_EPISODE,
        episode_id=context.episode_id,
        isolation_id=outcome.isolation_id,
        state_match=outcome.state_match,
        extraction_verified=fact.extraction_verified,
        observed_generation=context.generation,
    ))
    return ProbeAttempt(admission, candidate, outcome, evidence_id)
