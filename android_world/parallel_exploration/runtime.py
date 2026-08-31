"""High-level progressive memory and skip-inference gate."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from android_world.parallel_exploration.belief_graph import GraphEdge
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
from android_world.parallel_exploration.config import TwoSystemConfig


@dataclasses.dataclass(frozen=True)
class SkipDecision:
  skip: bool
  reason: str
  edge: GraphEdge | None = None


class SkipInferenceGate:
  def __init__(self, graph: ProgressiveBeliefGraph, config: TwoSystemConfig):
    self.graph = graph
    self.config = config
    self.consecutive_skip_count = 0

  def query(self, current_state_id: str) -> SkipDecision:
    if not self.config.enabled or not self.config.skip_inference_enabled:
      return SkipDecision(False, "feature_disabled")
    edge = self.graph.get_reusable_action(
        current_state_id,
        max_entropy=self.config.max_reusable_entropy,
        max_age_s=self.config.max_entry_age_s,
    )
    if edge is None:
      return SkipDecision(False, "no_fresh_safe_verified_edge")
    return SkipDecision(True, "verified_graph_transition", edge)

  def execute_and_verify(
      self, decision: SkipDecision, execute: Callable[[Mapping[str, Any]], Any],
      successor_matches: Callable[[str | None], bool],
  ) -> bool:
    if not decision.skip or decision.edge is None:
      return False
    execute(decision.edge.action)
    matched = successor_matches(decision.edge.dst_node)
    self.graph.record_execution_verification(decision.edge.edge_id, matched)
    if matched:
      self.consecutive_skip_count += 1
      return True
    self.consecutive_skip_count = 0
    if decision.edge.dst_node:
      self.graph.invalidate_subtree(decision.edge.dst_node)
    return False

