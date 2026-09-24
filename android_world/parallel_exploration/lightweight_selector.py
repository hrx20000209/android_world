"""Dependency-free linear GUI selector and compact graph view.

The C++ implementation in ``gui_exploration_selector`` is the device-side
reference. This module mirrors the exact feature map in Python so the existing
AndroidWorld shadow explorer can use the same model without starting Python on
the phone. It also contains a terse graph representation used for persistence
and bounded reasoning prompts.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


COMPACT_GRAPH_SCHEMA_VERSION = 1
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(*values: object) -> set[str]:
  tokens: set[str] = set()
  for value in values:
    tokens.update(TOKEN_RE.findall(str(value or "").lower()))
  return tokens


def _field(element: object, *names: str) -> str:
  if isinstance(element, Mapping):
    for name in names:
      if name in element and element[name] is not None:
        return str(element[name])
    return ""
  for name in names:
    value = getattr(element, name, None)
    if value is not None:
      return str(value)
  return ""


@dataclasses.dataclass(frozen=True)
class LinearModel:
  bias: float
  weights: Mapping[str, float]

  @classmethod
  def from_text(cls, path: str | Path) -> "LinearModel":
    bias = 0.0
    weights: dict[str, float] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
      parts = raw.strip().split()
      if len(parts) < 2 or parts[0].startswith("#"):
        continue
      key = parts[0].lower()
      try:
        value = float(parts[1])
      except ValueError:
        continue
      if key == "bias":
        bias = value
      elif key not in {"version", "vocab_size"}:
        weights[key] = value
    return cls(bias=bias, weights=weights)

  def score(self, element: object) -> float:
    tokens = tokenize(
        _field(element, "text"),
        _field(element, "content_desc", "contentDescription"),
        _field(element, "resource_id", "resourceId"),
    )
    return self.bias + sum(self.weights.get(token, 0.0) for token in tokens)


@dataclasses.dataclass(frozen=True)
class ScoredElement:
  element: object
  score: float
  original_index: int


def rank_elements(elements: Iterable[object], model: LinearModel) -> list[ScoredElement]:
  ranked = [
      ScoredElement(element, model.score(element), index)
      for index, element in enumerate(elements)
  ]
  ranked.sort(key=lambda item: (-item.score, item.original_index))
  return ranked


class LightweightLinearRanker:
  """Adapter for AndroidWorld's existing safe candidate pipeline."""

  def __init__(self, weights_path: str | Path):
    self.model = LinearModel.from_text(weights_path)

  def rank(self, elements, task: str = "", probed_ids=()):
    del task, probed_ids
    from android_world.parallel_exploration.rankers import RankedElement

    return [
        RankedElement(element=item.element, rank=rank, score=item.score)
        for rank, item in enumerate(rank_elements(elements, self.model), start=1)
    ]


def _short_label(selector: Mapping[str, Any]) -> str:
  for key in ("text", "content_desc", "resource_id", "class_name"):
    value = str(selector.get(key) or "").strip()
    if value:
      if key == "resource_id" and "/" in value:
        value = value.rsplit("/", 1)[-1]
      return re.sub(r"\s+", " ", value)[:48]
  return "?"


def _short_activity(activity: str) -> str:
  activity = str(activity or "")
  return activity.rsplit("/", 1)[-1][-48:]


def _edge_score(edge: Any, goal_tokens: set[str]) -> float:
  function = str(getattr(edge, "function", "") or "")
  relevance = len(goal_tokens & tokenize(function)) / max(1, len(goal_tokens))
  q_value = getattr(edge, "q_value", None)
  if callable(q_value):
    try:
      return float(q_value(task_relevance=relevance))
    except TypeError:
      return float(q_value())
  return float(getattr(edge, "confidence", 0.0) or 0.0)


@dataclasses.dataclass(frozen=True)
class CompactGraphSummary:
  """A terse row-oriented graph representation.

  Full memory remains the source of truth. This representation intentionally
  omits raw element arrays, coordinates, duplicate target observations, and
  dynamic text. State IDs are local S0/S1 labels, making it cheap to put a
  small bounded slice in a prompt or inspect it on a device.
  """

  states: tuple[Mapping[str, Any], ...]
  edges: tuple[Mapping[str, Any], ...]
  skills: tuple[Mapping[str, Any], ...] = ()

  @classmethod
  def from_memory(
      cls, memory: Any, *, max_nodes: int = 128, max_edges: int = 256,
      max_skills: int = 32,
  ) -> "CompactGraphSummary":
    state_values = sorted(
        memory.states.values(),
        key=lambda node: (-int(getattr(node, "visit_count", 0)), str(node.node_id)),
    )[:max_nodes]
    state_ids = {node.node_id: f"S{index}" for index, node in enumerate(state_values)}
    states = []
    for node in state_values:
      landmarks = [str(value)[:32] for value in tuple(node.landmarks)[:8]]
      states.append({
          "i": state_ids[node.node_id],
          "n": str(node.node_id),
          "a": _short_activity(node.activity),
          "l": landmarks,
          "v": int(node.visit_count),
      })
    edges = []
    ordered_edges = sorted(
        memory.edges.values(),
        key=lambda edge: (-_edge_score(edge, set()), str(edge.edge_id)),
    )
    for edge in ordered_edges:
      source = state_ids.get(edge.source_node)
      if source is None:
        continue
      targets = [
          {"i": state_ids[target], "n": int(count)}
          for target, count in edge.target_states.items()
          if target in state_ids
      ]
      if not targets:
        continue
      targets.sort(key=lambda item: (-item["n"], item["i"]))
      edges.append({
          "s": source,
          "a": _short_label(dataclasses.asdict(edge.selector)),
          "t": targets[:3],
          "q": round(float(edge.q_value()), 4),
          "p": int(edge.support_count),
          "r": round(float(edge.reversible_rate), 3),
          "x": int(bool(edge.trap_count or edge.external_count)),
      })
      if len(edges) >= max_edges:
        break
    skills = []
    for skill in sorted(
        memory.skills.values(),
        key=lambda item: (-int(item.support_count), str(item.skill_id)),
    )[:max_skills]:
      skills.append({
          "a": list(skill.action_sequence),
          "p": int(skill.support_count),
          "v": int(skill.validation_count),
          "x": int(bool(skill.executable)),
      })
    return cls(tuple(states), tuple(edges), tuple(skills))

  def to_dict(self) -> dict[str, Any]:
    return {
        "schema_version": COMPACT_GRAPH_SCHEMA_VERSION,
        "encoding": "gui-memory-rows-v1",
        "states": list(self.states),
        "edges": list(self.edges),
        "skills": list(self.skills),
    }

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

  def prompt(self, *, max_chars: int = 1200, state_id: str = "") -> str:
    lines = ["[GUI-MEMORY-COMPACT v1]"]
    selected_alias = next(
        (row.get("i") for row in self.states if row.get("n") == state_id),
        state_id,
    )
    selected_states = [
        row for row in self.states
        if not selected_alias or row.get("i") == selected_alias
    ]
    for row in selected_states[:2]:
      landmarks = ",".join(str(item) for item in row.get("l") or ())
      lines.append(f"{row.get('i')}: {_short_activity(row.get('a', ''))} [{landmarks}]")
    selected_edges = [
        edge for edge in self.edges
        if not selected_alias or edge.get("s") == selected_alias
    ]
    for edge in selected_edges[:5]:
      targets = ",".join(
          f"{item.get('i')}({item.get('n')})" for item in edge.get("t") or ()
      )
      lines.append(
          f"{edge.get('s')} -{edge.get('a')}-> {targets}; "
          f"q={float(edge.get('q', 0.0)):.2f},n={edge.get('p', 0)},"
          f"rev={float(edge.get('r', 0.0)):.2f},trap={edge.get('x', 0)}"
      )
    executable = [skill for skill in self.skills if skill.get("x")]
    if executable:
      lines.append(f"validated_groups={len(executable)}")
    return "\n".join(lines)[:max_chars]


def write_compact_graph(path: str | Path, memory: Any) -> Path:
  target = Path(path)
  target.parent.mkdir(parents=True, exist_ok=True)
  summary = CompactGraphSummary.from_memory(memory)
  target.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
  return target


__all__ = [
    "COMPACT_GRAPH_SCHEMA_VERSION", "CompactGraphSummary", "LinearModel",
    "LightweightLinearRanker", "ScoredElement", "rank_elements", "tokenize",
    "write_compact_graph",
]
