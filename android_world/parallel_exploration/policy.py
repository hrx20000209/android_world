"""Replaceable exploration policy and information-gain estimator."""

from __future__ import annotations

import abc
import dataclasses
import math
from collections.abc import Iterable, Mapping
from typing import Any

from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
from android_world.parallel_exploration.information import InformationNeed
from android_world.parallel_exploration.resources import ExplorationBudget


@dataclasses.dataclass(frozen=True)
class CandidateEdge:
  action: dict[str, Any]
  path_probability: float
  predicted_information_gain: float
  predicted_future_value: float
  exploration_cost: float
  risk_level: str = "UNKNOWN"


class InformationGainPredictor(abc.ABC):
  @abc.abstractmethod
  def estimate(self, task: str, labels: Iterable[str], need: InformationNeed, action: Mapping[str, Any]) -> tuple[float, float, float]:
    raise NotImplementedError


class HeuristicInformationGainPredictor(InformationGainPredictor):
  def estimate(self, task: str, labels: Iterable[str], need: InformationNeed, action: Mapping[str, Any]) -> tuple[float, float, float]:
    haystack = " ".join([task, need.current_subgoal, need.target_entity, *need.required_information_slots, *need.expected_affordances]).casefold()
    action_text = " ".join(str(v) for v in action.values()).casefold()
    goal_tokens = {token for token in haystack.replace("_", " ").split() if len(token) > 1}
    action_tokens = {token for token in action_text.replace("_", " ").split() if len(token) > 1}
    overlap = len(goal_tokens & action_tokens) / max(1, len(goal_tokens))
    novelty = min(1.0, len(tuple(labels)) / 12.0)
    probability = min(0.95, 0.15 + 1.8 * overlap)
    information_gain = min(1.0, 0.2 + 0.5 * novelty + 0.5 * overlap)
    future_value = min(1.0, probability * (0.5 + novelty))
    return probability, information_gain, future_value


class ExplorationPolicy(abc.ABC):
  @abc.abstractmethod
  def choose(self, task: str, need: InformationNeed, graph: ProgressiveBeliefGraph, budget: ExplorationBudget, candidates: Iterable[CandidateEdge]) -> CandidateEdge | None:
    raise NotImplementedError


class DeterministicExplorationPolicy(ExplorationPolicy):
  def __init__(self, gamma: float = 0.5):
    self.gamma = gamma

  def choose(self, task: str, need: InformationNeed, graph: ProgressiveBeliefGraph, budget: ExplorationBudget, candidates: Iterable[CandidateEdge]) -> CandidateEdge | None:
    del task, need, graph
    if not budget.allow_exploration:
      return None
    def score(edge: CandidateEdge) -> float:
      value = edge.predicted_information_gain + self.gamma * edge.predicted_future_value
      return value / max(edge.exploration_cost, 1e-6)
    allowed = [edge for edge in candidates if edge.risk_level in {"SAFE", "LOW", "UNKNOWN"} and math.isfinite(score(edge))]
    return max(allowed, key=score, default=None)

