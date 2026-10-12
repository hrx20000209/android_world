"""Episode-local navigation to an explicit decision boundary.

This is an experimental controller, not a safety oracle. UI restoration and
task applicability are separate evidence. By default execution requires an
independent app-state effect check; ``allow_ui_only`` is a measured ablation.
No visits, model-choice entropy, or lexical similarity license execution.
"""

from __future__ import annotations

import dataclasses
import hashlib
import heapq
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _norm(value: Any) -> str:
  return " ".join(str(value or "").casefold().split())


def _digest(value: Any) -> str:
  return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def _get(value: Any, *keys: str, default: Any = None) -> Any:
  for key in keys:
    result = value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)
    if result is not None:
      return result
  return default


def _strings(value: Any) -> tuple[str, ...]:
  if not isinstance(value, (list, tuple)) or not all(isinstance(x, str) and x.strip() for x in value):
    return ()
  return tuple(dict.fromkeys(_norm(x) for x in value))


@dataclasses.dataclass(frozen=True)
class NavigationIntent:
  task_key: str
  intent_id: str
  package: str
  target_labels: tuple[str, ...]
  entity_labels: tuple[str, ...] = ()
  target_activity: str = ""
  issued_generation: int = 0
  valid_for_generations: int = 4
  max_hops: int = 4

  def valid(self, task: str, generation: int) -> bool:
    return (self.task_key == _digest(task)
            and 0 <= generation - self.issued_generation <= self.valid_for_generations)

  def matches(self, state: Mapping[str, Any]) -> bool:
    labels = {_norm(x) for x in state.get("labels", ())}
    return (state.get("package") == self.package
            and (not self.target_activity or state.get("activity") == self.target_activity)
            and bool(self.target_labels)
            and set(self.target_labels + self.entity_labels).issubset(labels))


def parse_navigation_intent(output: str, task: str, generation: int) -> NavigationIntent | None:
  """Accept only an explicit model contract, bound locally to this task/epoch.

  Put the optional NAV_INTENT tag inside the existing THINK block to preserve
  GELAB's action parser. Missing/malformed contracts fail closed. A task-goal
  token overlap, a guessed entity or a confidence score is never a contract.
  """
  tags = re.findall(r"<NAV_INTENT>\s*(.*?)\s*</NAV_INTENT>", output or "", re.S | re.I)
  if len(tags) != 1:
    return None
  try:
    value = json.loads(tags[0])
    if not isinstance(value, dict) or value.get("navigation_only") is not True:
      return None
    if value.get("stop_before_decision") is not True:
      return None
    labels = _strings(value.get("target_labels"))
    entities = _strings(value.get("entity_labels", []))
    if "entity_labels" in value and value["entity_labels"] and not entities:
      return None
    package = value.get("package")
    if not labels or not isinstance(package, str) or not re.fullmatch(r"[\w]+(?:\.[\w]+)+", package):
      return None
    activity = value.get("target_activity", "")
    if not isinstance(activity, str) or (activity and not activity.startswith(package + "/")):
      return None
    if len(labels) + len(entities) > 12:
      return None
    return NavigationIntent(
        task_key=_digest(task), intent_id=_digest([task, generation, value]),
        package=package, target_labels=labels, entity_labels=entities,
        target_activity=activity, issued_generation=generation,
    )
  except (ValueError, TypeError):
    return None


INTENT_PROMPT = """[Optional navigation contract]
Inside your existing <THINK> block, optionally include exactly one
<NAV_INTENT>{"navigation_only":true,"stop_before_decision":true,
"package":"actual.android.package","target_labels":["exact landing cue"],
"entity_labels":[],"target_activity":""}</NAV_INTENT>.
The contract describes where navigation should continue AFTER your chosen
action, before the next task decision. Only provide it when you know that
destination is needed. The executor stops on arrival and asks you to reason.
Use exact GUI labels, bind an entity when its identity matters, and omit the
tag if uncertain, already at the destination, or ready to edit/submit/answer.
The contract never authorizes a data mutation or declaring task completion.
"""


def state_from_signature(signature: Any) -> dict[str, Any]:
  activity = _get(_get(signature, "activity"), "component", default="")
  elements = _get(signature, "elements", default=())
  return {
      "activity": activity, "package": activity.split("/", 1)[0],
      "layout_signature": _get(signature, "layout_sig", default=""),
      "structural_signature": _get(_get(signature, "struct_sig"), "digest", default=""),
      "labels": list(dict.fromkeys(
          _norm(_get(e, key, default="")) for e in elements
          if _get(e, "package", default="") in ("", activity.split("/", 1)[0])
          for key in ("text", "content_desc") if _get(e, key, default=""))),
  }


def _state_from_row(value: Mapping[str, Any]) -> dict[str, Any]:
  activity = str(value.get("activity") or "")
  labels = list(value.get("labels") or ())
  labels.extend(_get(e, key, default="") for e in value.get("elements", ())
                for key in ("text", "content_desc") if _get(e, key, default=""))
  return {
      "activity": activity, "package": activity.split("/", 1)[0],
      "layout_signature": str(value.get("layout_signature") or ""),
      "structural_signature": str(value.get("structural_signature") or ""),
      "labels": list(dict.fromkeys(_norm(x) for x in labels if x)),
  }


def observation_from_probe(row: Mapping[str, Any], generation: int) -> dict[str, Any] | None:
  """A bounded witness tied to the exact pre/post screens, not latest nodes.

  ``state_effect_verified`` must come from an independent app-state checker.
  Ordinary live-probe rows lack it and remain explicitly UI-only evidence.
  """
  graph = row.get("graph") or {}
  source, target = graph.get("src") or {}, graph.get("dst") or {}
  if not source or not target:
    return None
  element = row.get("element") or {}
  identity = str((graph.get("action") or {}).get("element_identity") or "")
  parts = identity.split("|") + [""] * 4
  selector = {
      "resource_id": str(_get(element, "resource_id", default=parts[0])),
      "text": str(_get(element, "text", default=parts[1])),
      "content_desc": str(_get(element, "content_desc", default=parts[2])),
      "class_name": str(_get(element, "class_name", "class", default=parts[3])),
  }
  return {
      "generation": generation, "source": _state_from_row(source),
      "target": _state_from_row(target), "selector": selector,
      "action_type": str((graph.get("action") or {}).get("action_type") or "").lower(),
      "probe_type": str(row.get("probe_type") or ""),
      "ui_restored": row.get("recovery_ok") is True,
      "inverse_level": str(row.get("recovery_level") or ""),
      "state_effect_verified": row.get("state_effect_verified") is True,
      "checked": element.get("checked"),
      "dynamic_content": bool(row.get("dynamic_content", False)),
      "cost_s": max(0.0, float((row.get("timings_ms") or {}).get("total", 0)) / 1000),
      "trial_id": str(row.get("trial_id") or ""),
  }


def snapshot_edges(snapshot: Any) -> dict[str, dict[str, Any]]:
  values = _get(snapshot, "edges", default={})
  if isinstance(values, Mapping):
    return {str(k): dict(v) for k, v in values.items()}
  return {str(v["edge_id"]): dict(v) for v in values}


def merge_action_prefix(model_actions: Sequence[Mapping[str, Any]],
                        navigation_actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
  """Interleave macro actions without inflating the model's decision counter.

  Each macro record has the model-action index it preceded. Include current
  step's macro even though its terminal model action has not run yet.
  """
  by_index: dict[int, list[dict[str, Any]]] = {}
  for item in navigation_actions:
    by_index.setdefault(int(item["before_action_index"]), []).append(
        {"action_dict": dict(item["action_dict"]), "tool_call": {}})
  merged = []
  for index in range(len(model_actions) + 1):
    merged.extend(by_index.get(index, ()))
    if index < len(model_actions):
      merged.append(dict(model_actions[index]))
  return merged


def _state_key(state: Mapping[str, Any]) -> str:
  # The coarse layout is only a retrieval key; never an execution guard.
  return _digest([state.get("activity"), state.get("layout_signature")])


def _same_state(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
  return (bool(expected.get("structural_signature"))
          and expected.get("activity") == observed.get("activity")
          and expected.get("structural_signature") == observed.get("structural_signature"))


@dataclasses.dataclass(frozen=True)
class NavigationHop:
  edge_id: str
  source: Mapping[str, Any]
  target: Mapping[str, Any]
  selector: Mapping[str, Any]
  evidence_generation: int
  cost_s: float
  evidence_level: str


@dataclasses.dataclass(frozen=True)
class NavigationRoute:
  intent_id: str
  snapshot_generation: int
  hops: tuple[NavigationHop, ...]
  terminal: Mapping[str, Any]
  expected_net_saving_s: float

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


class TaskNavigationPlanner:
  """Bounded search over witnessed routes to a task-licensed terminal."""

  def __init__(self, *, allow_ui_only: bool = False, max_age_generations: int = 6,
               max_expansions: int = 128, verification_cost_s: float = .25):
    self.allow_ui_only = allow_ui_only
    self.max_age_generations = max_age_generations
    self.max_expansions = max_expansions
    self.verification_cost_s = verification_cost_s
    self.last_rejections: dict[str, int] = {}

  def _witnesses(self, snapshot: Any, intent: NavigationIntent) -> list[NavigationHop]:
    generation = int(_get(snapshot, "generation", default=0))
    self.last_rejections = {}
    hops = []
    for eid, edge in snapshot_edges(snapshot).items():
      for witness in edge.get("navigation_observations", ()):
        reason = ""
        source, target = witness["source"], witness["target"]
        execution = witness.get("origin") == "execution"
        age = generation - int(witness["generation"])
        selector = witness["selector"]
        if str(edge.get("status")) in {"INVALID", "STALE"} or edge.get("dynamic_content"):
          reason = "invalid_or_dynamic_edge"
        elif str(edge.get("risk_level", "UNKNOWN")).upper() in {"HIGH", "IRREVERSIBLE"}:
          reason = "edge_risk"
        elif not 0 < age <= self.max_age_generations:
          reason = "uncommitted_or_stale"
        elif source.get("package") != intent.package or target.get("package") != intent.package:
          reason = "package_mismatch"
        elif not source.get("structural_signature") or not target.get("structural_signature"):
          reason = "missing_screen_contract"
        elif witness.get("action_type") != "click":
          reason = "not_navigation_probe"
        elif execution and witness.get("probe_type") != "EXEC_NAV":
          reason = "not_navigation_probe"
        elif not execution and witness.get("probe_type") not in {"TAP_NAV", "TAP_MENU"}:
          reason = "not_navigation_probe"
        elif not execution and (not witness.get("ui_restored")
                                or witness.get("inverse_level") != "INVERSE"):
          # A probe that could not be undone is evidence of a side effect. An
          # executed click is never undone; its certificate is the checker's.
          reason = "no_exact_inverse"
        elif edge.get("rollback_failure_count", 0) or edge.get("execution_miss_count", 0):
          reason = "contradictory_history"
        elif witness.get("checked") is not None or witness.get("dynamic_content"):
          reason = "stateful_or_dynamic_control"
        elif not selector.get("resource_id") or not (selector.get("text") or selector.get("content_desc")):
          reason = "weak_selector"
        elif selector.get("class_name", "").rsplit(".", 1)[-1].casefold() == "edittext":
          reason = "editable_control"
        elif not witness.get("state_effect_verified") and not self.allow_ui_only:
          reason = "unverified_app_state_effect"
        elif _same_state(source, target):
          reason = "no_state_change"
        if reason:
          self.last_rejections[reason] = self.last_rejections.get(reason, 0) + 1
          continue
        hops.append(NavigationHop(
            eid, dict(source), dict(target), dict(selector), int(witness["generation"]),
            float(witness.get("cost_s", 0)),
            "app_state_verified" if witness.get("state_effect_verified") else "ui_only",
        ))
    outcomes: dict[str, set[str]] = {}
    for hop in hops:
      source_action = _digest([hop.source.get("activity"), hop.source.get("structural_signature"), hop.selector])
      outcomes.setdefault(source_action, set()).add(
          _digest([hop.target.get("activity"), hop.target.get("structural_signature")]))
    unambiguous = []
    for hop in hops:
      key = _digest([hop.source.get("activity"), hop.source.get("structural_signature"), hop.selector])
      if len(outcomes[key]) == 1:
        unambiguous.append(hop)
      else:
        self.last_rejections["ambiguous_transition"] = self.last_rejections.get("ambiguous_transition", 0) + 1
    return unambiguous

  def compile(self, snapshot: Any, current: Mapping[str, Any], intent: NavigationIntent | None,
              *, task: str, inference_cost_s: float = 0.0) -> NavigationRoute | None:
    self.last_rejections = {}
    generation = int(_get(snapshot, "generation", default=0))
    if intent is None or not intent.valid(task, generation) or intent.matches(current):
      return None
    if current.get("package") != intent.package:
      return None
    outgoing: dict[str, list[NavigationHop]] = {}
    for hop in self._witnesses(snapshot, intent):
      outgoing.setdefault(_state_key(hop.source), []).append(hop)
    queue = [(0.0, 0, current, (), frozenset())]
    serial = 0
    for _ in range(self.max_expansions):
      if not queue:
        break
      cost, _, state, path, seen = heapq.heappop(queue)
      if path and intent.matches(state):
        # Each hop removes an intermediate call; terminal reasoning remains.
        saving = len(path) * max(0.0, inference_cost_s) - cost
        return NavigationRoute(intent.intent_id, generation, path, dict(state), saving)
      if len(path) >= intent.max_hops:
        continue
      for hop in sorted(outgoing.get(_state_key(state), ()), key=lambda x: x.edge_id):
        if not _same_state(hop.source, state) or hop.edge_id in seen:
          continue
        if _same_state(hop.target, current) or any(_same_state(hop.target, prior.source) for prior in path):
          continue
        serial += 1
        heapq.heappush(queue, (
            cost + hop.cost_s + self.verification_cost_s, serial, hop.target,
            path + (hop,), seen | {hop.edge_id},
        ))
    return None


def relocate(selector: Mapping[str, Any], elements: Sequence[Any]) -> dict[str, Any] | None:
  """Exact named resource + class + label; refuse recycled-row ambiguity."""
  matches = []
  for element in elements:
    if not _get(element, "clickable", "is_clickable", default=False):
      continue
    if _get(element, "checked") is not None or _get(element, "is_editable", default=False):
      continue
    if _get(element, "enabled", "is_enabled", default=True) is False:
      continue
    if _get(element, "visible", "is_visible", default=True) is False:
      continue
    if not all(_norm(_get(element, key, default="")) == _norm(value)
               for key, value in selector.items() if value):
      continue
    bounds = _get(element, "bounds", default=())
    if len(bounds) != 4 or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
      continue
    matches.append(bounds)
  if len(matches) != 1:
    return None
  left, top, right, bottom = matches[0]
  return {"action_type": "click", "x": round((left + right) / 2), "y": round((top + bottom) / 2)}


@dataclasses.dataclass(frozen=True)
class RouteExecution:
  completed_hops: int
  attempted_hops: int
  reached_terminal: bool
  reason: str
  actions: tuple[Mapping[str, Any], ...]
  verified_edge_ids: tuple[str, ...]
  failed_edge_id: str = ""


def execute_route(route: NavigationRoute, intent: NavigationIntent, *, task: str,
                  capture: Callable[[], Any], execute: Callable[[Mapping[str, Any]], None],
                  on_action: Callable[[Mapping[str, Any], NavigationHop], None] | None = None
                  ) -> RouteExecution:
  """Check each precondition/selector and landing; stop before terminal reasoning.

  A mismatched hop is already a real action: preserve it in model history and
  return control from the actual screen. Never pretend to undo it with Back.
  """
  actions: list[Mapping[str, Any]] = []
  verified = []
  def result(reason: str, reached: bool = False, failed: str = "") -> RouteExecution:
    return RouteExecution(len(verified), len(actions), reached, reason,
                          tuple(actions), tuple(verified), failed)
  if route.intent_id != intent.intent_id or not intent.valid(task, route.snapshot_generation):
    return result("intent_mismatch")
  try:
    state = capture()
  except Exception:
    return result("capture_failed")
  for hop in route.hops:
    observed = state_from_signature(state)
    if intent.matches(observed):
      return result("terminal_reached", True)
    if not _same_state(hop.source, observed):
      return result("source_contract_mismatch", failed=hop.edge_id)
    action = relocate(hop.selector, _get(state, "elements", default=()))
    if action is None:
      return result("selector_missing_or_ambiguous", failed=hop.edge_id)
    actions.append(action)
    if on_action is not None:
      on_action(action, hop)
    try:
      execute(action)
    except Exception:
      return result("execution_failed", failed=hop.edge_id)
    try:
      state = capture()
    except Exception:
      return result("capture_failed_after_action", failed=hop.edge_id)
    if not _same_state(hop.target, state_from_signature(state)):
      return result("landing_contract_mismatch", failed=hop.edge_id)
    verified.append(hop.edge_id)
  reached = intent.matches(state_from_signature(state))
  return result("terminal_reached" if reached else "terminal_contract_mismatch", reached)


def frontier_scores(snapshot: Any, node_id: str, elements: Sequence[Any],
                    target_labels: Sequence[str], *, max_hops: int = 4,
                    screen_elements: Sequence[Any] | None = None) -> dict[str, dict[str, Any]]:
  """A task-specific structural heuristic, never called information gain.

  Back-propagate observed target coverage, expose unexplored exits, suppress
  repeated no-ops. This ranks only candidates already admitted by SafetyGate.
  Scores do not authorize probe side effects or navigation execution.
  """
  targets = {_norm(x) for x in target_labels if x}
  edges = snapshot_edges(snapshot)
  value: dict[str, float] = {}
  for edge in edges.values():
    if str(edge.get("status")) in {"INVALID", "STALE"}:
      continue
    labels = {_norm(x) for x in edge.get("discovered_labels", ())}
    for observation in edge.get("navigation_observations", ()):
      labels.update(observation["target"].get("labels", ()))
    dst = edge.get("dst_node")
    if dst and targets:
      value[dst] = max(value.get(dst, 0), len(targets & labels) / len(targets))
  for _ in range(max_hops):
    updated = dict(value)
    for edge in edges.values():
      if str(edge.get("status")) in {"INVALID", "STALE"}:
        continue
      updated[edge["src_node"]] = max(updated.get(edge["src_node"], 0),
                                     .8 * value.get(edge.get("dst_node"), 0))
    value = updated
  scores = {}
  local = [e for e in edges.values() if e.get("src_node") == node_id]
  for element in elements:
    identity = str(_get(element, "identity", default=""))
    control = "|".join(identity.split("|")[:4])
    known = [e for e in local if (e.get("action") or {}).get("element_identity") == identity
             or (control and (e.get("action") or {}).get("control_key") == control)]
    evidence = sum(int(e.get("probe_count", 0)) for e in known)
    labels = {_norm(_get(element, key, default="")) for key in ("text", "content_desc")}
    if screen_elements is not None:
      # A list row names its entity in a child label, not its own text.
      from android_world.parallel_exploration.procedure_replay import inner_labels
      labels |= {_norm(x) for x in inner_labels(element, screen_elements)}
    direct = len(targets & labels) / len(targets) if targets else 0
    downstream = max((value.get(e.get("dst_node"), 0) for e in known), default=0)
    no_op = any(evidence and e.get("src_node") == e.get("dst_node") for e in known)
    unknown = 1 / (1 + evidence)
    value_score = direct + downstream + .25 * unknown if targets else 0.0
    if no_op:
      value_score = 0.0
    scores[identity] = {"score": value_score, "target_coverage": direct,
                        "downstream_coverage": downstream, "probe_count": evidence,
                        "unknown_bonus": unknown, "known_noop": no_op}
  return scores
