"""Machine-readable per-round experiment logging."""

from __future__ import annotations

import csv
import dataclasses
import json
import threading
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class InferenceRoundRecord:
  round_id: int
  task_id: str
  current_state_id: str
  inference_latency_s: float = 0.0
  inference_memory_mb: float = 0.0
  inference_output_action: dict[str, Any] = dataclasses.field(default_factory=dict)
  exploration_budget: dict[str, Any] = dataclasses.field(default_factory=dict)
  number_of_exploration_probes: int = 0
  max_exploration_depth: int = 0
  explored_edges: list[dict[str, Any]] = dataclasses.field(default_factory=list)
  rollback_latency_ms: float = 0.0
  rollback_success: bool = False
  preemption_count: int = 0
  unfinished_exploration: bool = False
  concurrent_memory_overhead_mb: float = 0.0
  interference_estimate: float = 0.0
  cpu_utilization: float | None = None
  gpu_utilization: float | None = None
  power_w: float | None = None
  temperature_c: float | None = None
  memory_confidence: float = 0.0
  explorer_reliability: float = 0.0
  inference_skipped: bool = False
  consecutive_skip_count: int = 0
  verified_lookahead_horizon: int = 0


class RoundLogger:
  def __init__(self, jsonl_path: Path, csv_path: Path | None = None):
    self.jsonl_path = Path(jsonl_path)
    self.csv_path = Path(csv_path) if csv_path else self.jsonl_path.with_suffix(".csv")
    self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    self._lock = threading.Lock()

  def append(self, record: InferenceRoundRecord) -> None:
    row = dataclasses.asdict(record)
    with self._lock:
      with self.jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
      flat = {key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value) for key, value in row.items()}
      exists = self.csv_path.exists()
      with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        if not exists:
          writer.writeheader()
        writer.writerow(flat)

