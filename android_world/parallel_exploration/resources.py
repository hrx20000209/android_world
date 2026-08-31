"""Runtime resource sampling and adaptive exploration budgets."""

from __future__ import annotations

import dataclasses
import os
import time
from collections.abc import Callable
from typing import Any

import psutil

from android_world.parallel_exploration.config import TwoSystemConfig


@dataclasses.dataclass(frozen=True)
class ResourceSnapshot:
  timestamp: float
  available_memory_mb: float
  process_memory_mb: float
  memory_used_ratio: float
  cpu_used_ratio: float
  gpu_utilization: float | None = None
  gpu_memory_used_mb: float | None = None
  power_w: float | None = None
  temperature_c: float | None = None


@dataclasses.dataclass(frozen=True)
class ExplorationBudget:
  max_probes: int
  max_depth: int
  max_extra_memory_mb: float
  max_exploration_time_s: float
  allow_exploration: bool
  reason: str


class ResourceMonitor:
  def __init__(self, platform_sampler: Callable[[], dict[str, float | None]] | None = None):
    self._process = psutil.Process(os.getpid())
    self._platform_sampler = platform_sampler

  def sample(self) -> ResourceSnapshot:
    virtual = psutil.virtual_memory()
    extra: dict[str, Any] = self._platform_sampler() if self._platform_sampler else {}
    return ResourceSnapshot(
        timestamp=time.time(),
        available_memory_mb=virtual.available / 1024**2,
        process_memory_mb=self._process.memory_info().rss / 1024**2,
        memory_used_ratio=virtual.percent / 100.0,
        cpu_used_ratio=psutil.cpu_percent(interval=None) / 100.0,
        gpu_utilization=extra.get("gpu_utilization"),
        gpu_memory_used_mb=extra.get("gpu_memory_used_mb"),
        power_w=extra.get("power_w"),
        temperature_c=extra.get("temperature_c"),
    )


class ResourceAdaptiveBudgetController:
  def __init__(self, config: TwoSystemConfig):
    self.config = config
    self.interference_ema = 0.0
    self.explorer_reliability = 0.5

  def observe_interference(self, infer_solo_s: float | None, infer_concurrent_s: float | None) -> float:
    if not infer_solo_s or infer_solo_s <= 0 or infer_concurrent_s is None:
      return self.interference_ema
    observed = max(-1.0, (infer_concurrent_s - infer_solo_s) / infer_solo_s)
    alpha = self.config.interference_ema_alpha
    self.interference_ema = alpha * observed + (1.0 - alpha) * self.interference_ema
    return self.interference_ema

  def observe_explorer_result(self, hit: bool) -> float:
    self.explorer_reliability = 0.8 * self.explorer_reliability + 0.2 * float(hit)
    return self.explorer_reliability

  def allocate(self, resources: ResourceSnapshot, *, predicted_utility: float, recent_inference_s: float = 0.0, recent_exploration_s: float = 0.0) -> ExplorationBudget:
    del recent_inference_s, recent_exploration_s
    c = self.config
    if not c.enabled or not c.exploration_enabled:
      return ExplorationBudget(0, 0, 0.0, 0.0, False, "feature_disabled")
    if resources.available_memory_mb < c.min_available_memory_mb or resources.memory_used_ratio >= c.memory_pressure_ratio:
      return ExplorationBudget(0, 0, 0.0, 0.0, False, "memory_pressure")
    if predicted_utility < c.min_predicted_utility:
      return ExplorationBudget(0, 0, 0.0, 0.0, False, "low_predicted_utility")
    if self.interference_ema > c.interference_threshold * 2:
      return ExplorationBudget(0, 0, 0.0, 0.0, False, "severe_inference_interference")
    if c.max_probes <= 0:
      # An explicitly zero probe budget is a real configuration, not a
      # degenerate one to be clamped away: it is how exploration is ablated
      # while every other part of the two-system runtime (graph, skip gate,
      # evidence) stays active. The scaling below has a max(1, ...) floor
      # that would otherwise silently turn "no exploration" into "one probe
      # per round", making that ablation impossible to actually run.
      return ExplorationBudget(0, 0, 0.0, 0.0, False, "zero_probe_budget_configured")
    headroom = min(1.0, resources.available_memory_mb / max(c.min_available_memory_mb * 4, 1.0))
    cpu_factor = max(0.15, 1.0 - resources.cpu_used_ratio)
    interference_factor = max(0.15, 1.0 - max(0.0, self.interference_ema))
    utility_factor = min(1.0, max(0.15, predicted_utility))
    reliability_factor = 0.5 + 0.5 * self.explorer_reliability
    scale = min(1.0, headroom * 1.5) * cpu_factor * interference_factor * utility_factor * reliability_factor
    probes = max(1, round(c.max_probes * scale))
    depth = max(1, min(c.max_depth, 1 + int(scale * c.max_depth)))
    return ExplorationBudget(
        probes, depth, c.max_extra_memory_mb * max(0.1, scale),
        c.max_exploration_time_s * max(0.1, scale), True, "adaptive",
    )

