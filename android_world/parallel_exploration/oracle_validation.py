"""Protocol and analysis primitives for the online-information oracle study.

This module deliberately models *current-episode evidence* separately from
navigation memory.  It contains no graph-ranking policy: the first experiment
is an oracle upper bound whose purpose is to decide whether a learned online
explorer is worth building at all.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ExperimentArm(str, enum.Enum):
  NO_MEMORY = "no_memory"
  OFFLINE_TOPOLOGY = "offline_topology"
  DEPLOYMENT_HISTORY = "deployment_history"
  ONLINE_ORACLE = "online_oracle"
  ONLINE_STALE = "online_oracle_without_freshness"


REQUIRED_ARMS = tuple(ExperimentArm)


@dataclasses.dataclass(frozen=True)
class ExplorationOpportunity:
  """A decision point at which current hidden information may be useful."""

  task_id: str
  app_family: str
  episode_seed: int
  decision_id: str
  state_id: str
  task_need: str
  freshness_requirement_s: float | None
  candidate_actions: tuple[str, ...]
  online_opportunity: bool = True

  @property
  def case_id(self) -> str:
    return f"{self.task_id}|{self.episode_seed}|{self.decision_id}"

  @classmethod
  def from_dict(cls, value: Mapping[str, Any]) -> "ExplorationOpportunity":
    return cls(
        task_id=str(value["task_id"]),
        app_family=str(value["app_family"]),
        episode_seed=int(value["episode_seed"]),
        decision_id=str(value["decision_id"]),
        state_id=str(value["state_id"]),
        task_need=str(value["task_need"]),
        freshness_requirement_s=(
            None if value.get("freshness_requirement_s") is None
            else float(value["freshness_requirement_s"])
        ),
        candidate_actions=tuple(str(x) for x in value.get("candidate_actions", ())),
        online_opportunity=bool(value.get("online_opportunity", True)),
    )


@dataclasses.dataclass(frozen=True)
class FreshEvidence:
  """Task-local observation bound to the state from which it was obtained."""

  source_action: str
  observed_fact: str
  state_binding: str
  observed_at_s: float
  confidence: float
  is_fresh: bool
  isolation_id: str

  @classmethod
  def from_dict(cls, value: Mapping[str, Any]) -> "FreshEvidence":
    return cls(
        source_action=str(value["source_action"]),
        observed_fact=str(value["observed_fact"]),
        state_binding=str(value["state_binding"]),
        observed_at_s=float(value["observed_at_s"]),
        confidence=float(value["confidence"]),
        is_fresh=bool(value["is_fresh"]),
        isolation_id=str(value["isolation_id"]),
    )


@dataclasses.dataclass(frozen=True)
class OracleTrialResult:
  """One arm result at one paired AndroidWorld decision point."""

  trial_id: str
  arm: ExperimentArm
  opportunity: ExplorationOpportunity
  memory_seeds: tuple[int, ...]
  slot_correct: bool
  next_inference_avoided: bool
  remaining_reasoning_calls: int
  remaining_actions: int
  stale_evidence_error: bool
  evidence_eligible: bool
  task_success: bool | None = None
  e2e_s: float | None = None
  inference_window_s: float | None = None
  evidence: FreshEvidence | None = None
  metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

  def validate(self) -> None:
    if self.remaining_reasoning_calls < 0 or self.remaining_actions < 0:
      raise ValueError(f"{self.trial_id}: remaining costs must be non-negative")
    if self.opportunity.episode_seed in self.memory_seeds:
      raise ValueError(
          f"{self.trial_id}: episode seed leaks into offline/history memory"
      )
    if self.arm == ExperimentArm.ONLINE_ORACLE:
      if self.evidence is None or not self.evidence.is_fresh:
        raise ValueError(f"{self.trial_id}: online oracle requires fresh evidence")
    elif self.arm == ExperimentArm.ONLINE_STALE:
      if self.evidence is None or self.evidence.is_fresh:
        raise ValueError(f"{self.trial_id}: stale arm requires stale evidence")
    elif self.evidence is not None:
      raise ValueError(
          f"{self.trial_id}: non-online arm must not receive probe evidence"
      )
    if self.evidence is not None:
      if self.evidence.state_binding != self.opportunity.state_id:
        raise ValueError(f"{self.trial_id}: evidence is bound to another state")
      if not 0.0 <= self.evidence.confidence <= 1.0:
        raise ValueError(f"{self.trial_id}: confidence must be in [0, 1]")

  def to_dict(self) -> dict[str, Any]:
    row = dataclasses.asdict(self)
    row["arm"] = self.arm.value
    row["memory_seeds"] = list(self.memory_seeds)
    return row

  @classmethod
  def from_dict(cls, value: Mapping[str, Any]) -> "OracleTrialResult":
    result = cls(
        trial_id=str(value["trial_id"]),
        arm=ExperimentArm(str(value["arm"])),
        opportunity=ExplorationOpportunity.from_dict(value["opportunity"]),
        memory_seeds=tuple(int(x) for x in value.get("memory_seeds", ())),
        slot_correct=bool(value["slot_correct"]),
        next_inference_avoided=bool(value["next_inference_avoided"]),
        remaining_reasoning_calls=int(value["remaining_reasoning_calls"]),
        remaining_actions=int(value["remaining_actions"]),
        stale_evidence_error=bool(value["stale_evidence_error"]),
        evidence_eligible=bool(value["evidence_eligible"]),
        task_success=(
            None if value.get("task_success") is None else bool(value["task_success"])
        ),
        e2e_s=None if value.get("e2e_s") is None else float(value["e2e_s"]),
        inference_window_s=(
            None if value.get("inference_window_s") is None
            else float(value["inference_window_s"])
        ),
        evidence=(
            None if value.get("evidence") is None
            else FreshEvidence.from_dict(value["evidence"])
        ),
        metadata=dict(value.get("metadata", {})),
    )
    result.validate()
    return result


def load_results(path: Path) -> list[OracleTrialResult]:
  results: list[OracleTrialResult] = []
  with Path(path).open(encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, 1):
      if not line.strip():
        continue
      try:
        results.append(OracleTrialResult.from_dict(json.loads(line)))
      except Exception as exc:
        raise ValueError(f"{path}:{line_number}: {exc}") from exc
  validate_paired_results(results)
  return results


def write_results(path: Path, results: Iterable[OracleTrialResult]) -> None:
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  rows = list(results)
  validate_paired_results(rows)
  with output.open("w", encoding="utf-8") as stream:
    for result in rows:
      stream.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def validate_paired_results(results: Sequence[OracleTrialResult]) -> None:
  if not results:
    raise ValueError("No oracle results were provided")
  by_case: dict[str, dict[ExperimentArm, OracleTrialResult]] = defaultdict(dict)
  for result in results:
    result.validate()
    arms = by_case[result.opportunity.case_id]
    if result.arm in arms:
      raise ValueError(
          f"{result.opportunity.case_id}: duplicate arm {result.arm.value}"
      )
    arms[result.arm] = result
  expected = set(REQUIRED_ARMS)
  for case_id, arms in by_case.items():
    missing = expected - set(arms)
    if missing:
      names = ", ".join(sorted(x.value for x in missing))
      raise ValueError(f"{case_id}: missing paired arms: {names}")


_METRICS = {
    "slot_accuracy": lambda r: float(r.slot_correct),
    "next_inference_avoid_rate": lambda r: float(r.next_inference_avoided),
    "remaining_reasoning_calls": lambda r: float(r.remaining_reasoning_calls),
    "remaining_actions": lambda r: float(r.remaining_actions),
    "stale_error_rate": lambda r: float(r.stale_evidence_error),
    "evidence_coverage": lambda r: float(r.evidence_eligible),
}


def _mean(values: Sequence[float]) -> float:
  return sum(values) / len(values) if values else math.nan


def _percentile(values: Sequence[float], q: float) -> float:
  if not values:
    return math.nan
  ordered = sorted(values)
  position = (len(ordered) - 1) * q
  low = math.floor(position)
  high = math.ceil(position)
  if low == high:
    return ordered[low]
  return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _paired_differences(
    results: Sequence[OracleTrialResult],
    treatment: ExperimentArm,
    control: ExperimentArm,
    metric: str,
) -> dict[str, float]:
  by_case: dict[str, dict[ExperimentArm, OracleTrialResult]] = defaultdict(dict)
  for result in results:
    by_case[result.opportunity.case_id][result.arm] = result
  fn = _METRICS[metric]
  differences = [
      fn(arms[treatment]) - fn(arms[control])
      for arms in by_case.values()
      if treatment in arms and control in arms
  ]
  return {"n": len(differences), "mean": _mean(differences)}


def paired_bootstrap_ci(
    results: Sequence[OracleTrialResult],
    treatment: ExperimentArm,
    control: ExperimentArm,
    metric: str,
    *,
    iterations: int = 4000,
    seed: int = 20270915,
) -> dict[str, float]:
  """Case-level paired bootstrap; arms from a case are never split."""
  if iterations < 100:
    raise ValueError("iterations must be at least 100")
  by_case: dict[str, dict[ExperimentArm, OracleTrialResult]] = defaultdict(dict)
  for result in results:
    by_case[result.opportunity.case_id][result.arm] = result
  fn = _METRICS[metric]
  differences = [
      fn(arms[treatment]) - fn(arms[control])
      for arms in by_case.values()
      if treatment in arms and control in arms
  ]
  if not differences:
    return {"n": 0, "mean": math.nan, "low": math.nan, "high": math.nan}
  rng = random.Random(seed)
  samples = [
      _mean([differences[rng.randrange(len(differences))] for _ in differences])
      for _ in range(iterations)
  ]
  return {
      "n": len(differences),
      "mean": _mean(differences),
      "low": _percentile(samples, 0.025),
      "high": _percentile(samples, 0.975),
  }


def summarize_results(
    results: Sequence[OracleTrialResult], *, bootstrap_iterations: int = 4000
) -> dict[str, Any]:
  validate_paired_results(results)
  by_arm: dict[ExperimentArm, list[OracleTrialResult]] = defaultdict(list)
  for result in results:
    by_arm[result.arm].append(result)
  arms: dict[str, Any] = {}
  for arm in REQUIRED_ARMS:
    rows = by_arm[arm]
    arms[arm.value] = {
        "n": len(rows),
        **{
            name: _mean([fn(row) for row in rows])
            for name, fn in _METRICS.items()
        },
        "task_success_rate": _mean(
            [float(row.task_success) for row in rows if row.task_success is not None]
        ),
        "mean_e2e_s": _mean([row.e2e_s for row in rows if row.e2e_s is not None]),
    }

  comparisons: dict[str, Any] = {}
  for control in (ExperimentArm.OFFLINE_TOPOLOGY, ExperimentArm.DEPLOYMENT_HISTORY):
    key = f"online_oracle_vs_{control.value}"
    comparisons[key] = {
        metric: paired_bootstrap_ci(
            results,
            ExperimentArm.ONLINE_ORACLE,
            control,
            metric,
            iterations=bootstrap_iterations,
        )
        for metric in _METRICS
    }
  comparisons["online_oracle_vs_stale"] = {
      metric: paired_bootstrap_ci(
          results,
          ExperimentArm.ONLINE_ORACLE,
          ExperimentArm.ONLINE_STALE,
          metric,
          iterations=bootstrap_iterations,
      )
      for metric in _METRICS
  }

  by_family: dict[str, list[OracleTrialResult]] = defaultdict(list)
  for result in results:
    if result.opportunity.online_opportunity:
      by_family[result.opportunity.app_family].append(result)
  family_reductions: dict[str, Any] = {}
  passing_families = 0
  for family, rows in sorted(by_family.items()):
    vs_offline_calls = _paired_differences(
        rows, ExperimentArm.OFFLINE_TOPOLOGY, ExperimentArm.ONLINE_ORACLE,
        "remaining_reasoning_calls")
    vs_history_calls = _paired_differences(
        rows, ExperimentArm.DEPLOYMENT_HISTORY, ExperimentArm.ONLINE_ORACLE,
        "remaining_reasoning_calls")
    vs_offline_actions = _paired_differences(
        rows, ExperimentArm.OFFLINE_TOPOLOGY, ExperimentArm.ONLINE_ORACLE,
        "remaining_actions")
    vs_history_actions = _paired_differences(
        rows, ExperimentArm.DEPLOYMENT_HISTORY, ExperimentArm.ONLINE_ORACLE,
        "remaining_actions")
    stable = (
        max(vs_offline_calls["mean"], vs_offline_actions["mean"]) >= 1.0
        and max(vs_history_calls["mean"], vs_history_actions["mean"]) >= 1.0
    )
    passing_families += int(stable)
    family_reductions[family] = {
        "saved_vs_offline_calls": vs_offline_calls["mean"],
        "saved_vs_offline_actions": vs_offline_actions["mean"],
        "saved_vs_history_calls": vs_history_calls["mean"],
        "saved_vs_history_actions": vs_history_actions["mean"],
        "passes_one_unit_rule": stable,
    }

  freshness_slot = comparisons["online_oracle_vs_stale"]["slot_accuracy"]
  freshness_errors = comparisons["online_oracle_vs_stale"]["stale_error_rate"]
  freshness_pass = freshness_slot["low"] > 0 or freshness_errors["high"] < 0
  history_vs_none = paired_bootstrap_ci(
      results,
      ExperimentArm.DEPLOYMENT_HISTORY,
      ExperimentArm.NO_MEMORY,
      "remaining_reasoning_calls",
      iterations=bootstrap_iterations,
  )
  online_pass = passing_families >= 2
  if online_pass and freshness_pass:
    decision = "CONTINUE_ONLINE"
  elif history_vs_none["high"] < 0:
    decision = "PIVOT_TO_HISTORY_REUSE"
  else:
    decision = "STOP_ONLINE_EXPLORATION"
  return {
      "schema_version": 1,
      "paired_cases": len({r.opportunity.case_id for r in results}),
      "arms": arms,
      "comparisons": comparisons,
      "app_families": family_reductions,
      "go_no_go": {
          "decision": decision,
          "online_pass": online_pass,
          "freshness_pass": freshness_pass,
          "passing_app_families": passing_families,
          "rule": (
              "Continue only if fresh online evidence saves >=1 future reasoning "
              "call or action versus both offline topology and deployment history "
              "in at least two app families, and fresh evidence beats stale evidence."
          ),
      },
  }
