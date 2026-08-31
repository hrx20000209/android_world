"""Central configuration for the two-system MobileExplorer runtime."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class TwoSystemConfig:
  enabled: bool = False
  graph_enabled: bool = True
  exploration_enabled: bool = True
  skip_inference_enabled: bool = False
  structured_logging_enabled: bool = True
  # Framed as a per-round exploration-step budget, not a latency budget: how
  # many candidate transitions can be tried and how deep can the search go
  # before the round ends, independent of wall-clock target. 12/3 left most
  # rounds throttled by ResourceAdaptiveBudgetController's scale factor down
  # to 1-4 probes at depth 1, which was empirically too narrow to ever cover
  # the element the authoritative model actually picks (see
  # docs/mobileexplorer_implementation_audit_zh.md and the 2026-08-28
  # single-task probe-coverage experiment).
  max_probes: int = 20
  max_depth: int = 3
  # Goal-directed depth continuation: keep expanding a speculative branch past
  # depth 1 only while what it reveals still overlaps the task's
  # InformationNeed (not merely "still looks clickable" - see
  # rankers.InformationNeedRanker and live_probe._goal_relevance_score).
  goal_relevance_threshold: float = 0.0
  # Stop deepening a branch once its Android back-stack has grown more than
  # this many levels past the round's root baseline: a big single-hop jump
  # usually means the probe landed on an unfamiliar surface (detail/dialog/
  # share flow) rather than a simple navigation step, and continuing from
  # there compounds recovery risk instead of gathering more signal.
  max_stack_depth_increase: int = 1
  max_exploration_time_s: float = 8.0
  max_extra_memory_mb: float = 512.0
  min_available_memory_mb: float = 1024.0
  memory_pressure_ratio: float = 0.85
  cpu_pressure_ratio: float = 0.90
  interference_threshold: float = 0.15
  interference_ema_alpha: float = 0.25
  min_predicted_utility: float = 0.05
  reusable_confidence_threshold: float = 0.90
  max_reusable_entropy: float = 0.25
  max_entry_age_s: float = 120.0
  state_verify_threshold: float = 0.72
  recovery_timeout_s: float = 12.0
  abort_poll_interval_s: float = 0.05

