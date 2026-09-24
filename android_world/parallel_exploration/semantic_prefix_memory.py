"""Small, executable Semantic Prefix Memory sidecar.

This module deliberately does not define a second UI-state identity.  It is a
bounded, value-free execution memory attached to the existing Ex5 runner.  The
authoritative graph and the reasoning model remain the source of truth; this
sidecar is only allowed to offer a route when all conservative checks pass.

There are no embeddings, neural models, or LLM calls here.  Semantic matching
is a small explicit alias table plus token overlap, which keeps the file easy
to inspect, serialize, and replay on a failed experiment.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
import tempfile
import time
from typing import Any


SCHEMA_VERSION = 1
TOKEN_RE = re.compile(r"[a-z0-9]+")
SHORTCUT_CONFIDENCE = 0.82
MAX_ROUTE_LENGTH = 3

# Explicit and intentionally tiny.  Values are canonical terms, so a route
# recorded as "preferences" can match a task that says "settings" without
# introducing a learned semantic representation.
SEMANTIC_ALIASES: dict[str, str] = {
    "preferences": "settings", "preference": "settings", "option": "settings",
    "options": "settings", "configuration": "settings", "configure": "settings",
    "magnify": "search", "find": "search", "lookup": "search",
    "previous": "back", "return": "back", "up": "back",
    "nextstep": "next", "forward": "next", "proceed": "continue",
    "person": "profile", "user": "profile", "account": "account",
    "remove": "delete", "trash": "delete", "bin": "delete",
    "compose": "create", "new": "create", "plus": "add",
    "favourite": "favorite", "starred": "favorite",
    "preferencescreen": "settings", "hamburger": "menu",
}


def _tokens(*values: object) -> set[str]:
  result: set[str] = set()
  for value in values:
    for token in TOKEN_RE.findall(str(value or "").lower()):
      result.add(SEMANTIC_ALIASES.get(token, token))
  return result


def _text(value: object, *names: str) -> str:
  if isinstance(value, Mapping):
    for name in names:
      if value.get(name) is not None:
        return str(value.get(name) or "")
  else:
    for name in names:
      result = getattr(value, name, None)
      if result is not None:
        return str(result or "")
  return ""


def _bounds(value: object) -> tuple[float, float, float, float] | None:
  raw = value.get("bounds") if isinstance(value, Mapping) else getattr(value, "bounds", None)
  if raw is None:
    return None
  try:
    if isinstance(raw, str):
      raw = json.loads(raw)
    if len(raw) != 4:
      return None
    return tuple(float(x) for x in raw)  # type: ignore[return-value]
  except (TypeError, ValueError, json.JSONDecodeError):
    return None


def _element_dict(value: object) -> dict[str, Any]:
  if isinstance(value, Mapping):
    source = dict(value)
    return {
        "text": str(source.get("text") or ""),
        "content_desc": str(source.get("content_desc", source.get("contentDescription", "")) or ""),
        "resource_id": str(source.get("resource_id", source.get("resourceId", "")) or ""),
        "class": str(source.get("class", source.get("class_name", "")) or ""),
        "bounds": list(source.get("bounds") or []) if source.get("bounds") is not None else [],
    }
  return {
      "text": _text(value, "text"),
      "content_desc": _text(value, "content_desc", "content_description", "contentDescription"),
      "resource_id": _text(value, "resource_id", "resource_name", "resourceId"),
      "class": _text(value, "class", "class_name", "className"),
      "bounds": list(_bounds(value) or ()),
  }


def _state_id(activity: str, layout_signature: str) -> str:
  raw = f"{activity}\0{layout_signature}".encode("utf-8")
  return "pfx-" + hashlib.sha256(raw).hexdigest()[:20]


def state_id_from_signature(signature: object) -> str:
  """Return the sidecar key for an existing AndroidWorld state capture."""
  activity_obj = getattr(signature, "activity", None)
  if activity_obj is not None:
    activity = _text(activity_obj, "component")
  elif isinstance(signature, Mapping) and isinstance(signature.get("activity"), Mapping):
    activity = _text(signature.get("activity"), "component", "activity")
  else:
    activity = _text(signature, "activity")
  layout = _text(signature, "layout_signature", "layout_sig")
  return _state_id(activity, layout)


@dataclasses.dataclass
class PrefixState:
  state_id: str
  parent_state_id: str | None = None
  incoming_action_token: str = ""
  depth: int = 0
  visit_count: int = 0
  semantic_aliases: tuple[str, ...] = ()
  reversible: bool = True
  return_cost: float = 0.0
  dynamic_content: bool = False
  activity: str = ""
  layout_signature: str = ""

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, value: Mapping[str, Any]) -> "PrefixState":
    fields = {field.name for field in dataclasses.fields(cls)}
    return cls(**{key: value[key] for key in fields if key in value})


@dataclasses.dataclass
class PrefixEdge:
  edge_id: str
  source_state_id: str
  raw_element: dict[str, Any]
  normalized_semantic_token: str
  support_count: int = 0
  reversible: bool = False
  return_cost: float = 0.0
  landing_distribution: dict[str, int] = dataclasses.field(default_factory=dict)
  route_hit_count: int = 0
  route_miss_count: int = 0
  confidence: float = 0.0
  dynamic_content: bool = False
  ambiguous_selector: bool = False
  trap: bool = False
  target_activity: str = ""
  action: dict[str, Any] = dataclasses.field(default_factory=dict)
  last_updated: float = dataclasses.field(default_factory=time.time)

  @property
  def consistent_landing_count(self) -> int:
    return max(self.landing_distribution.values(), default=0)

  @property
  def target_state_id(self) -> str | None:
    if not self.landing_distribution:
      return None
    return max(self.landing_distribution, key=self.landing_distribution.get)

  @property
  def route_success_count(self) -> int:
    return self.route_hit_count

  def recompute_confidence(self) -> None:
    if self.trap or self.ambiguous_selector or self.dynamic_content:
      self.confidence = 0.0
      return
    support = max(0, self.support_count)
    consistent = self.consistent_landing_count / max(1, support)
    # Three identical, reversible observations pass the .82 gate.  A prior
    # verified route hit can independently seed confidence, but never bypasses
    # selector/reversibility/dynamic/trap checks.
    evidence = min(1.0, 0.18 * min(support, 5) + 0.34 * consistent)
    rollback = 0.14 if self.reversible else 0.0
    verified_hit = 0.36 if self.route_hit_count else 0.0
    miss_penalty = min(0.35, 0.12 * self.route_miss_count)
    self.confidence = max(0.0, min(0.99, 0.20 + evidence + rollback + verified_hit - miss_penalty))

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, value: Mapping[str, Any]) -> "PrefixEdge":
    fields = {field.name for field in dataclasses.fields(cls)}
    edge = cls(**{key: value[key] for key in fields if key in value})
    edge.landing_distribution = {
        str(key): int(item) for key, item in (edge.landing_distribution or {}).items()
    }
    edge.recompute_confidence()
    return edge


@dataclasses.dataclass(frozen=True)
class PrefixRoute:
  route_id: str
  source_state_id: str
  edges: tuple[PrefixEdge, ...]
  relevance: float
  score: float

  @property
  def length(self) -> int:
    return len(self.edges)

  def to_dict(self) -> dict[str, Any]:
    return {
        "route_id": self.route_id,
        "source_state_id": self.source_state_id,
        "edge_ids": [edge.edge_id for edge in self.edges],
        "length": self.length,
        "relevance": self.relevance,
        "score": self.score,
    }


@dataclasses.dataclass
class PrefixMetrics:
  probe_records: int = 0
  nodes: int = 0
  edges: int = 0
  prompt_context_count: int = 0
  route_candidates: int = 0
  route_attempts: int = 0
  route_hits: int = 0
  route_misses: int = 0
  parse_error_count: int = 0
  selector_relocations: int = 0
  selector_ambiguities: int = 0
  repeated_verifications: int = 0

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


class SemanticPrefixMemory:
  """Versioned, task-independent sidecar memory for safe prefix routes."""

  def __init__(self, path: str | Path | None = None, *, mode: str = "off") -> None:
    self.path = Path(path) if path else None
    self.mode = mode
    self.states: dict[str, PrefixState] = {}
    self.edges: dict[str, PrefixEdge] = {}
    self._outgoing: dict[str, set[str]] = collections.defaultdict(set)
    self.metrics = PrefixMetrics()
    self.events: list[dict[str, Any]] = []
    if self.path and self.path.is_file():
      self.load(self.path)

  @staticmethod
  def _edge_key(source: str, element: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    selector = _element_dict(element)
    raw = json.dumps({
        "source": source,
        "resource_id": selector.get("resource_id"),
        "text": selector.get("text"),
        "content_desc": selector.get("content_desc"),
        "class": selector.get("class"),
        "action_type": str(action.get("action_type") or "").lower(),
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "pe-" + hashlib.sha256(raw.encode()).hexdigest()[:20]

  @staticmethod
  def _state_from_row(value: Mapping[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    activity = str(value.get("activity") or "")
    layout = str(value.get("layout_signature") or value.get("structural_signature") or "")
    aliases = tuple(sorted(_tokens(activity.rsplit("/", 1)[-1], activity.split(".")[-1])))
    return _state_id(activity, layout), activity, layout, aliases

  def _upsert_state(
      self, state_id: str, *, parent_state_id: str | None, incoming_action_token: str,
      depth: int, aliases: Sequence[str], reversible: bool, return_cost: float,
      dynamic_content: bool, activity: str, layout_signature: str,
      visited: bool = True,
  ) -> PrefixState:
    old = self.states.get(state_id)
    if old is None:
      old = PrefixState(
          state_id=state_id, parent_state_id=parent_state_id,
          incoming_action_token=incoming_action_token, depth=depth,
          visit_count=0, semantic_aliases=tuple(sorted(set(aliases))),
          reversible=reversible, return_cost=return_cost,
          dynamic_content=dynamic_content, activity=activity,
          layout_signature=layout_signature,
      )
      self.states[state_id] = old
    else:
      old.semantic_aliases = tuple(sorted(set(old.semantic_aliases) | set(aliases)))
      old.dynamic_content = old.dynamic_content or dynamic_content
      if parent_state_id and not old.parent_state_id:
        old.parent_state_id = parent_state_id
      if incoming_action_token and not old.incoming_action_token:
        old.incoming_action_token = incoming_action_token
      old.reversible = old.reversible and reversible
      old.return_cost = max(old.return_cost, return_cost)
    if visited:
      old.visit_count += 1
    self.metrics.nodes = len(self.states)
    return old

  def record_transition(
      self, *, source_state_id: str, target_state_id: str,
      raw_element: Mapping[str, Any] | object, action: Mapping[str, Any],
      depth: int = 1, reversible: bool = False, return_cost: float = 0.0,
      dynamic_content: bool = False, ambiguous_selector: bool = False,
      trap: bool = False, target_activity: str = "", route_verified: bool = False,
      source_aliases: Sequence[str] = (), target_aliases: Sequence[str] = (),
  ) -> PrefixEdge:
    element = _element_dict(raw_element)
    semantic = " ".join(sorted(_tokens(
        element.get("text"), element.get("content_desc"),
        element.get("resource_id"), element.get("class"),
        action.get("action_type"),
    )))
    edge_id = self._edge_key(source_state_id, element, action)
    edge = self.edges.get(edge_id)
    if edge is None:
      edge = PrefixEdge(
          edge_id=edge_id, source_state_id=source_state_id,
          raw_element=element, normalized_semantic_token=semantic,
          reversible=bool(reversible), return_cost=float(return_cost),
          dynamic_content=bool(dynamic_content), ambiguous_selector=bool(ambiguous_selector),
          trap=bool(trap), target_activity=target_activity, action=dict(action),
      )
      self.edges[edge_id] = edge
      self._outgoing[source_state_id].add(edge_id)
    edge.support_count += 1
    edge.landing_distribution[target_state_id] = edge.landing_distribution.get(target_state_id, 0) + 1
    edge.reversible = edge.reversible and bool(reversible)
    edge.return_cost = max(edge.return_cost, float(return_cost))
    edge.dynamic_content = edge.dynamic_content or bool(dynamic_content)
    edge.ambiguous_selector = edge.ambiguous_selector or bool(ambiguous_selector)
    edge.trap = edge.trap or bool(trap)
    edge.target_activity = edge.target_activity or target_activity
    edge.last_updated = time.time()
    if route_verified:
      edge.route_hit_count += 1
    edge.recompute_confidence()
    self.metrics.edges = len(self.edges)
    self.metrics.repeated_verifications += int(edge.support_count > 1)
    return edge

  def record_probe_row(self, row: Mapping[str, Any]) -> PrefixEdge | None:
    graph = row.get("graph")
    if not isinstance(graph, Mapping):
      return None
    source = graph.get("src")
    target = graph.get("dst")
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
      return None
    source_id, source_activity, source_layout, source_aliases = self._state_from_row(source)
    target_id, target_activity, target_layout, target_aliases = self._state_from_row(target)
    element = row.get("element") if isinstance(row.get("element"), Mapping) else {}
    action = graph.get("action") if isinstance(graph.get("action"), Mapping) else {}
    notes = str(row.get("notes") or "")
    reached_activity = str((row.get("discovered") or {}).get("reached_activity") or target_activity)
    app_package = str(row.get("app_package") or "")
    reached_package = reached_activity.split("/", 1)[0]
    external = bool(app_package and reached_package and app_package != reached_package)
    failed_restore = notes in {"RESTORE_FAILED", "NESTED_RESTORE_FAILED"} or not bool(row.get("recovery_ok"))
    dynamic = bool(row.get("dynamic_content")) or not bool((row.get("sig_details") or {}).get("stable_structure", True))
    selector = _element_dict(element)
    ambiguous = not bool(selector.get("resource_id") or selector.get("text") or selector.get("content_desc"))
    edge = self.record_transition(
        source_state_id=source_id, target_state_id=target_id,
        raw_element=element, action=action, depth=int(row.get("depth") or 1),
        reversible=bool(row.get("recovery_ok")),
        return_cost=float((row.get("timings_ms") or {}).get("recovery", 0.0) or 0.0) / 1000.0,
        dynamic_content=dynamic, ambiguous_selector=ambiguous,
        trap=failed_restore or external, target_activity=target_activity,
        source_aliases=source_aliases, target_aliases=target_aliases,
    )
    self._upsert_state(
        source_id, parent_state_id=None, incoming_action_token="", depth=max(0, int(row.get("depth") or 1) - 1),
        aliases=source_aliases, reversible=True, return_cost=0.0,
        dynamic_content=False, activity=source_activity, layout_signature=source_layout,
        visited=False,
    )
    self._upsert_state(
        target_id, parent_state_id=source_id,
        incoming_action_token=edge.normalized_semantic_token,
        depth=int(row.get("depth") or 1), aliases=target_aliases,
        reversible=bool(row.get("recovery_ok")),
        return_cost=edge.return_cost, dynamic_content=dynamic,
        activity=target_activity, layout_signature=target_layout,
        visited=True,
    )
    self.metrics.probe_records += 1
    self.events.append({
        "event": "probe_recorded", "source_state_id": source_id,
        "target_state_id": target_id, "edge_id": edge.edge_id,
        "reversible": bool(row.get("recovery_ok")), "trap": edge.trap,
        "dynamic_content": edge.dynamic_content,
    })
    return edge

  def record_authoritative_transition(
      self, *, source_state_id: str, target_state_id: str,
      raw_element: Mapping[str, Any] | object, action: Mapping[str, Any],
      reversible: bool = True, target_activity: str = "",
  ) -> PrefixEdge:
    return self.record_transition(
        source_state_id=source_state_id, target_state_id=target_state_id,
        raw_element=raw_element, action=action, depth=1,
        reversible=reversible, return_cost=0.0, route_verified=False,
        target_activity=target_activity,
    )

  def confirm_authoritative_transition(
      self, *, source_state_id: str, target_state_id: str,
      raw_element: Mapping[str, Any] | object, action: Mapping[str, Any],
  ) -> PrefixEdge | None:
    """Promote an explored edge only when the real reasoner confirms it.

    This does not create a shortcut from coordinates alone.  The stored
    exploration selector must match the live control and its predicted
    landing must equal the authoritative action's observed landing.
    """
    edge_id = self._edge_key(source_state_id, _element_dict(raw_element), action)
    edge = self.edges.get(edge_id)
    if edge is None or edge.trap or edge.target_state_id != target_state_id:
      return None
    edge.route_hit_count += 1
    edge.recompute_confidence()
    self.events.append({
        "event": "authoritative_route_confirmation", "edge_id": edge.edge_id,
        "source_state_id": source_state_id, "target_state_id": target_state_id,
    })
    return edge

  def record_route_result(
      self, route: PrefixRoute, *, hit: bool, reason: str = "",
  ) -> None:
    self.metrics.route_attempts += 1
    if hit:
      self.metrics.route_hits += 1
    else:
      self.metrics.route_misses += 1
    for edge in route.edges:
      if hit:
        edge.route_hit_count += 1
      else:
        edge.route_miss_count += 1
      edge.recompute_confidence()
    self.events.append({
        "event": "prefix_route_hit" if hit else "prefix_route_miss",
        "route_id": route.route_id, "edge_ids": [edge.edge_id for edge in route.edges],
        "reason": reason,
    })

  def record_parse_error(self, error: object, fallback_action: Mapping[str, Any] | None = None) -> None:
    self.metrics.parse_error_count += 1
    self.events.append({
        "event": "parse_error", "error": str(error),
        "fallback_action": dict(fallback_action or {}),
    })

  @staticmethod
  def _edge_relevance(edge: PrefixEdge, goal_tokens: set[str], target: PrefixState | None) -> float:
    edge_tokens = _tokens(edge.normalized_semantic_token, edge.raw_element.get("text"),
                          edge.raw_element.get("content_desc"), edge.raw_element.get("resource_id"))
    target_tokens = set(target.semantic_aliases) if target else set()
    union = goal_tokens | edge_tokens
    direct = len(goal_tokens & edge_tokens) / max(1, len(goal_tokens))
    destination = len(goal_tokens & target_tokens) / max(1, len(goal_tokens))
    return min(1.0, max(direct, destination, len(goal_tokens & edge_tokens) / max(1, len(union))))

  def _eligible(self, edge: PrefixEdge) -> bool:
    return bool(
        edge.reversible and not edge.dynamic_content and not edge.ambiguous_selector
        and not edge.trap and edge.confidence >= SHORTCUT_CONFIDENCE
        and (edge.consistent_landing_count >= 3 or edge.route_hit_count > 0)
        and edge.target_state_id in self.states
    )

  def candidate_routes(self, current_state_id: str, goal: str, *, top_k: int = 3) -> list[PrefixRoute]:
    goal_tokens = _tokens(goal)
    if not goal_tokens or current_state_id not in self.states:
      return []
    candidates: list[PrefixRoute] = []

    def visit(source: str, path: tuple[PrefixEdge, ...], seen: set[str]) -> None:
      if len(path) >= MAX_ROUTE_LENGTH:
        return
      for edge_id in sorted(self._outgoing.get(source, ())):
        edge = self.edges[edge_id]
        if not self._eligible(edge):
          continue
        target_id = edge.target_state_id
        if not target_id or target_id in seen:
          continue
        next_path = path + (edge,)
        target = self.states.get(target_id)
        relevance = max(self._edge_relevance(item, goal_tokens, target) for item in next_path)
        # An explicit semantic match is required.  Confidence and path length
        # only rank already-matching routes; they never create a match.
        if relevance > 0.0:
          score = 0.62 * relevance + 0.30 * min(item.confidence for item in next_path) - 0.04 * (len(next_path) - 1)
          route_id = "route-" + hashlib.sha256(
              (current_state_id + ":" + ":".join(item.edge_id for item in next_path)).encode()
          ).hexdigest()[:20]
          candidates.append(PrefixRoute(route_id, current_state_id, next_path, relevance, score))
        visit(target_id, next_path, seen | {target_id})

    visit(current_state_id, (), {current_state_id})
    candidates.sort(key=lambda item: (-item.score, item.length, item.route_id))
    self.metrics.route_candidates += len(candidates)
    return candidates[:max(0, top_k)]

  def prompt_context(self, current_state_id: str, goal: str, *, top_k: int = 3, max_chars: int = 1400) -> str:
    routes = self.candidate_routes(current_state_id, goal, top_k=top_k)
    self.metrics.prompt_context_count += int(bool(routes))
    if not routes:
      return ""
    lines = ["[SEMANTIC_PREFIX_MEMORY v1 | advisory only]"]
    for route in routes:
      hops = []
      for edge in route.edges:
        label = edge.raw_element.get("text") or edge.raw_element.get("content_desc") or edge.raw_element.get("resource_id") or edge.normalized_semantic_token
        hops.append(f"{label} (support={edge.support_count}, conf={edge.confidence:.2f}, reversible=yes)")
      lines.append(f"- route {route.length} hop(s), relevance={route.relevance:.2f}: " + " -> ".join(hops))
    lines.append("Use only if the live selector is uniquely relocatable; otherwise follow Ex5.")
    return "\n".join(lines)[:max_chars]

  def relocate(self, edge: PrefixEdge, live_elements: Sequence[object]) -> dict[str, Any] | None:
    """Relocate a stored selector to a unique live element, never by coords alone."""
    raw = edge.raw_element
    matches = list(live_elements)
    resource_id = str(raw.get("resource_id") or "")
    if resource_id:
      matches = [item for item in matches if _text(item, "resource_id", "resource_name", "resourceId") == resource_id]
    if not matches and raw.get("content_desc"):
      matches = [item for item in live_elements if _text(item, "content_desc", "content_description", "contentDescription") == raw["content_desc"]]
    if not matches and raw.get("text"):
      matches = [item for item in live_elements if _text(item, "text") == raw["text"]]
    if raw.get("class"):
      typed = [item for item in matches if _text(item, "class", "class_name", "className") == raw["class"]]
      if typed:
        matches = typed
    if len(matches) != 1:
      self.metrics.selector_ambiguities += int(len(matches) != 1)
      return None
    bounds = _bounds(matches[0])
    if bounds is None:
      return None
    left, top, right, bottom = bounds
    action = dict(edge.action)
    action["action_type"] = str(action.get("action_type") or "CLICK").lower()
    action["x"] = round((left + right) / 2)
    action["y"] = round((top + bottom) / 2)
    self.metrics.selector_relocations += 1
    return action

  def summary(self) -> dict[str, Any]:
    self.metrics.nodes = len(self.states)
    self.metrics.edges = len(self.edges)
    depths = [state.depth for state in self.states.values()]
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": self.mode,
        "metrics": self.metrics.to_dict(),
        "node_count": len(self.states), "edge_count": len(self.edges),
        "average_depth": sum(depths) / len(depths) if depths else 0.0,
        "max_depth": max(depths, default=0),
    }

  def to_dict(self) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "encoding": "semantic-prefix-memory-v1",
        "mode": self.mode,
        "states": [state.to_dict() for state in self.states.values()],
        "edges": [edge.to_dict() for edge in self.edges.values()],
        "metrics": self.metrics.to_dict(),
    }

  def save(self, path: str | Path | None = None) -> Path | None:
    target = Path(path) if path else self.path
    if target is None:
      return None
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
    try:
      with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
      os.replace(temp_name, target)
    finally:
      if os.path.exists(temp_name):
        os.unlink(temp_name)
    self.path = target
    return target

  def load(self, path: str | Path) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = int(payload.get("schema_version", 0))
    if version > SCHEMA_VERSION:
      raise ValueError(f"unsupported semantic prefix schema version: {version}")
    self.states = {}
    self.edges = {}
    self._outgoing.clear()
    for value in payload.get("states", ()):
      if isinstance(value, Mapping) and value.get("state_id"):
        state = PrefixState.from_dict(value)
        self.states[state.state_id] = state
    for value in payload.get("edges", ()):
      if isinstance(value, Mapping) and value.get("edge_id") and value.get("source_state_id"):
        edge = PrefixEdge.from_dict(value)
        self.edges[edge.edge_id] = edge
        self._outgoing[edge.source_state_id].add(edge.edge_id)
    raw_metrics = payload.get("metrics") or {}
    fields = {field.name for field in dataclasses.fields(PrefixMetrics)}
    self.metrics = PrefixMetrics(**{key: int(raw_metrics[key]) for key in fields if key in raw_metrics})
    self.metrics.nodes = len(self.states)
    self.metrics.edges = len(self.edges)
