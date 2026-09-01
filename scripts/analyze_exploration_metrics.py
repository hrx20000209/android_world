"""PART F metrics: does the graph improve exploration, and does it improve reasoning?

Reads what a run already wrote - ``scored_candidates.jsonl`` (every candidate,
ranked, per step) and ``serial_events.jsonl`` (gate decisions, skips, the real
action each step committed) - and joins them.

The central measure is ``rank_of_future_real_action``: after the model picks
its action, where did that element sit in the ranking the explorer built
*before* seeing it. It separates the two failure modes a hit rate collapses
together - a selector that ranked the right element second is a very different
thing from one that never had it in the list at all - and it is the only
metric here that speaks directly to whether accumulated history makes
exploration more predictive.

Usage:
  python scripts/analyze_exploration_metrics.py <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import collections
import json
import pathlib
import statistics
import sys
from typing import Any, Iterable


def _rows(path: pathlib.Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  out = []
  for line in path.open(encoding="utf-8"):
    line = line.strip()
    if line:
      try:
        out.append(json.loads(line))
      except json.JSONDecodeError:
        continue
  return out


def _element_key(action: dict[str, Any]) -> tuple[int, int] | None:
  """A committed action's position, for matching against candidate bounds.

  Matching on coordinates rather than on element identity because the agent's
  action carries a tap point, while the candidate row carries the element it
  came from; identity strings are not comparable across the two.
  """
  x, y = action.get("x"), action.get("y")
  if x is None or y is None:
    return None
  return int(x), int(y)


def _covers(row: dict[str, Any], point: tuple[int, int], width: int = 1440,
            height: int = 2560) -> bool:
  cx, cy = row.get("norm_x"), row.get("norm_y")
  if cx is None or cy is None:
    return False
  # Rows keep the element's centre, normalised. A tap within 6% of the centre
  # in both axes is the same control at typical Android control sizes.
  return (abs(cx * width - point[0]) <= 0.06 * width
          and abs(cy * height - point[1]) <= 0.06 * height)


def exploration_metrics(run_dir: pathlib.Path) -> dict[str, Any]:
  scored = _rows(run_dir / "scored_candidates.jsonl")
  events = _rows(run_dir / "serial_events.jsonl")
  by_step: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
  for row in scored:
    by_step[row.get("step")].append(row)
  for rows in by_step.values():
    rows.sort(key=lambda r: r.get("rank", 0))

  ranks: list[int] = []
  missed = 0
  for event in events:
    if event.get("kind") != "inference":
      continue
    action = event.get("action") or {}
    point = _element_key(action if isinstance(action, dict) else {})
    rows = by_step.get(event.get("step"))
    if point is None or not rows:
      continue
    hit = next((r for r in rows if _covers(r, point)), None)
    if hit is None:
      missed += 1
    else:
      ranks.append(int(hit.get("rank", 0)))

  considered = len(ranks) + missed
  selected = [r for r in scored if r.get("selected_for_probe")]
  with_history = [r for r in selected if r.get("has_exact_history")]
  return {
      "steps_with_candidates": len(by_step),
      "real_action_considered": considered,
      "real_action_in_candidate_set": len(ranks),
      "top1_future_action_hit_rate":
          sum(1 for r in ranks if r == 1) / considered if considered else None,
      "top3_future_action_coverage":
          sum(1 for r in ranks if r <= 3) / considered if considered else None,
      "mean_rank_of_future_real_action":
          statistics.mean(ranks) if ranks else None,
      "median_rank_of_future_real_action":
          statistics.median(ranks) if ranks else None,
      "graph_history_usage_rate":
          len(with_history) / len(selected) if selected else None,
      "mean_selected_utility":
          statistics.mean([r.get("final_exploration_score", 0.0) for r in selected])
          if selected else None,
      "mean_selected_recoverability":
          statistics.mean([r.get("estimated_recoverability", 0.0) for r in selected])
          if selected else None,
  }


def reasoning_metrics(run_dir: pathlib.Path) -> dict[str, Any]:
  events = _rows(run_dir / "serial_events.jsonl")
  modes = collections.Counter()
  skips = [e for e in events if e.get("kind") == "skip"]
  contexts = [e for e in events if e.get("kind") == "graph_context"]
  probes = rounds = unrestored = 0
  for event in events:
    if event.get("kind") == "gate":
      modes[event.get("mode")] += 1
    elif event.get("kind") == "explore":
      rounds += 1
      probes += event.get("probes_completed", 0) or 0
      if event.get("restore_status") != "RESTORED":
        unrestored += 1
  inferences = sum(1 for e in events if e.get("kind") == "inference")
  injected = [c for c in contexts if c.get("injected")]
  actions = [json.dumps(e.get("action"), sort_keys=True)
             for e in events if e.get("kind") == "inference" and e.get("action")]
  repeated = sum(1 for a, b in zip(actions, actions[1:]) if a == b)
  return {
      "inference_calls": inferences,
      "gate_modes": dict(modes),
      "inference_skip_rate":
          modes.get("SKIP_INFERENCE", 0) / sum(modes.values()) if modes else None,
      "skips": len(skips),
      "skip_hit_rate":
          sum(1 for s in skips if s.get("matched")) / len(skips) if skips else None,
      "context_injected": len(injected),
      "mean_context_tokens":
          statistics.mean([c.get("tokens", 0) for c in injected]) if injected else 0,
      "mean_selected_fact_count":
          statistics.mean([c.get("selected_fact_count", 0) for c in contexts])
          if contexts else None,
      "repeated_action_rate": repeated / len(actions) if actions else None,
      "probe_rounds": rounds,
      "probes": probes,
      "unrestored_rounds": unrestored,
  }


def _merge(dicts: Iterable[dict[str, Any]]) -> dict[str, Any]:
  """Aggregate per-task metrics: sum counts, average rates."""
  merged: dict[str, Any] = {}
  collected: dict[str, list[Any]] = collections.defaultdict(list)
  for d in dicts:
    for key, value in d.items():
      if value is not None:
        collected[key].append(value)
  for key, values in collected.items():
    if isinstance(values[0], dict):
      total = collections.Counter()
      for value in values:
        total.update(value)
      merged[key] = dict(total)
    elif isinstance(values[0], bool):
      merged[key] = sum(values)
    elif isinstance(values[0], int) and not key.endswith(("rate", "coverage")):
      merged[key] = sum(values)
    else:
      merged[key] = round(statistics.mean(values), 3)
  return merged


def main(argv: list[str]) -> int:
  if len(argv) < 2:
    print(__doc__)
    return 2
  for arg in argv[1:]:
    root = pathlib.Path(arg)
    tasks = sorted(d for d in root.iterdir()
                   if d.is_dir() and (d / "serial_events.jsonl").exists())
    if not tasks:
      tasks = [root]
    print(f"\n===== {root.name}  ({len(tasks)} tasks) =====")
    print("-- F1 does the graph improve exploration --")
    for key, value in _merge(exploration_metrics(t) for t in tasks).items():
      print(f"  {key}: {value}")
    print("-- F2 does the graph improve reasoning --")
    for key, value in _merge(reasoning_metrics(t) for t in tasks).items():
      print(f"  {key}: {value}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
