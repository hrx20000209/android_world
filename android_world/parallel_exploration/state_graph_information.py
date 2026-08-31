"""Candidate-level information table over the progressive belief graph.

Exploration used to pick what to probe from one signal: how well a control's
label overlaps the task's InformationNeed (``InformationNeedRanker``). That is
a statement about the *current screen only* - it cannot express "this control
was probed twice already and both times the model went elsewhere", or "probes
of this kind on this screen have failed to roll back before". The graph knows
both, and paid real steps to learn them.

This module turns the graph into a table: one row per candidate element,
carrying current-UI features, node maturity, exact edge history, destination
structure, contextual (aggregated) history, InformationNeed match, safety, and
cost. ``PredictiveElementScorer`` then reduces a row to a scalar utility.

Two properties matter more than the individual features:

* **Unknown is not zero.** A candidate never probed has ``None`` for its
  history statistics, not 0.0. Storing "0% alignment" for something never
  tried would rank it below a candidate measured to be useless, which is
  backwards - the untried one is the only one that can still teach anything.
* **Safety is a gate, not a penalty.** ``SafetyGate`` removes candidates
  before scoring. A low score still gets probed when nothing better exists;
  an unrecoverable probe costs the episode.

Nothing here is app-specific or benchmark-specific: every feature is computed
from graph statistics and framework-level element attributes.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Iterable, Mapping, Protocol, Sequence

from android_world.parallel_exploration.rankers import UiElement


class GraphView(Protocol):
  """The read-only graph surface a row builder needs.

  Deliberately structural rather than a concrete class: the serial runner's
  generation-guarded ``GraphSnapshot`` and a plain adapter over
  ``ProgressiveBeliefGraph`` both satisfy it, so the temporal guarantee (step
  i reads only step i-1) is enforced by which object is passed in, and cannot
  be bypassed from inside this module.
  """

  generation: int
  node_visits: Mapping[str, int]
  node_entropy: Mapping[str, float]
  edges: Mapping[str, Mapping[str, Any]]
  outgoing: Mapping[str, Sequence[str]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ScoringConfig:
  """Every threshold and weight the scorer uses, in one place.

  Priors are measured, not guessed: the per-probe-type rollback rates come
  from 452 recorded probes on 2026-08-31 (TAP_NAV 93% recovered, SCROLL 90%,
  TAP_MENU 84%, EXPAND 77%), and the latencies from the same traces.
  """

  # Beta-Binomial shrinkage. With n observations of a rate, the estimate is
  # pulled toward the fallback prior with weight `prior_strength`; two
  # successes out of two should not read as certainty.
  prior_strength: float = 3.0

  # Recoverability hard gate (A3): below this a candidate is not probed.
  min_recoverability: float = 0.55
  # Per-probe-type rollback priors, used when history is absent.
  rollback_prior: Mapping[str, float] = dataclasses.field(
      default_factory=lambda: {
          "TAP_NAV": 0.93, "SCROLL": 0.90, "TAP_MENU": 0.84, "EXPAND": 0.77,
      })
  generic_rollback_prior: float = 0.80

  # Cost model, seconds (A1.8).
  probe_latency_prior: Mapping[str, float] = dataclasses.field(
      default_factory=lambda: {
          "TAP_NAV": 1.6, "SCROLL": 1.2, "TAP_MENU": 1.7, "EXPAND": 1.5,
      })
  generic_probe_latency: float = 1.5
  rollback_latency: float = 1.4
  # A failed rollback does not merely cost its own latency: it runs the
  # recovery ladder and can leave the device dirty for the next real step.
  # Priced well above a probe so the cost term, not just the gate, prefers
  # reliably reversible candidates.
  rollback_failure_penalty_s: float = 12.0

  # Path probability (A5 cases 2 and 5).
  path_prior_unexplored: float = 0.30
  need_match_weight: float = 0.45

  # Information gain (A2). Re-probing a transition whose destination is
  # already recorded teaches almost nothing; this is what makes a mature node
  # push exploration toward its unexplored candidates (case 6).
  known_destination_discount: float = 0.25
  novelty_scale: float = 8.0
  ig_prior_unexplored: float = 0.5

  # A candidate whose probes keep landing on screens with no task-relevant
  # content is down-weighted rather than banned - the evidence is weak until
  # repeated.
  irrelevant_branch_penalty: float = 0.5


DEFAULT_SCORING = ScoringConfig()


def _shrink(hits: float | None, total: float | None, prior: float,
            strength: float) -> float:
  """Posterior rate, pulled toward `prior` when observations are few."""
  if not total:
    return prior
  return (hits + strength * prior) / (total + strength)


# ---------------------------------------------------------------------------
# A1 - the row
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CandidateInformationRow:
  """One candidate element, described from every source available."""

  # --- A1.1 current UI -----------------------------------------------------
  element: UiElement
  element_identity: str
  text: str
  content_desc: str
  role: str
  probe_type: str
  norm_x: float
  norm_y: float
  clickable: bool
  scrollable: bool
  enabled: bool = True
  selected: bool | None = None
  checked: bool | None = None
  nearby_text: str = ""

  # --- A1.2 node / graph maturity -----------------------------------------
  node_visit_count: int = 0
  node_decision_entropy: float = math.inf
  outgoing_edge_count: int = 0
  valid_outgoing_edge_count: int = 0
  explored_element_count: int = 0
  candidate_element_count: int = 0
  exploration_coverage: float = 0.0
  graph_generation: int = 0

  # --- A1.3 exact edge history (None == never observed, not zero) ---------
  has_exact_history: bool = False
  edge_id: str | None = None
  edge_status: str | None = None
  confidence: float | None = None
  probe_count: int | None = None
  inference_alignment_count: int | None = None
  execution_hit_count: int | None = None
  execution_miss_count: int | None = None
  alignment_rate: float | None = None
  execution_hit_rate: float | None = None
  skip_attempt_count: int | None = None
  skip_success_count: int | None = None
  skip_success_rate: float | None = None
  mean_realized_ig: float | None = None
  rollback_success_count: int | None = None
  rollback_failure_count: int | None = None
  rollback_success_rate: float | None = None
  mean_exploration_cost: float | None = None
  known_inverse_level: str | None = None
  destination_known: bool = False
  destination_node_id: str | None = None

  # --- A1.4 destination ----------------------------------------------------
  destination_visit_count: int | None = None
  destination_entropy: float | None = None
  destination_out_degree: int | None = None
  destination_valid_out_degree: int | None = None
  destination_subtree_size: int | None = None
  destination_known_label_count: int | None = None
  destination_in_recent_path: bool = False
  discovered_labels: tuple[str, ...] = ()

  # How often, this episode, the authoritative model turned out to pick this
  # exact element at this exact node. Kept separate from the graph's
  # inference_alignment_count because it is observed live by the explorer
  # rather than read from the committed graph, and the two must not be summed
  # into one number whose provenance is then unrecoverable.
  session_alignment_hits: int = 0

  # --- A1.5 contextual history --------------------------------------------
  contextual_probe_count: int = 0
  contextual_alignment_rate: float | None = None
  contextual_execution_hit_rate: float | None = None
  contextual_mean_realized_ig: float | None = None
  contextual_rollback_success_rate: float | None = None

  # --- A1.6 information need ----------------------------------------------
  target_match: float = 0.0
  expected_affordance_match: float = 0.0
  unresolved_information_match: float = 0.0
  candidate_action_type_match: float = 0.0
  risk_conflict: float = 0.0

  # --- A1.7 safety / recoverability ---------------------------------------
  blocked_element: bool = False
  blocked_recovery_context: bool = False
  historical_rollback_success_rate: float | None = None
  historical_deep_recovery_rate: float | None = None
  cross_package_history: bool = False
  risk_level: str = "LOW"
  estimated_recoverability: float = 0.0

  # --- A1.8 cost -----------------------------------------------------------
  expected_probe_latency: float = 0.0
  expected_rollback_latency: float = 0.0
  expected_total_exploration_cost: float = 0.0

  # --- derived (filled by the scorer, logged separately) -------------------
  ui_novelty_score: float = 0.0
  path_probability: float = 0.0
  expected_information_gain: float = 0.0
  predictive_value: float = 0.0
  utility: float = 0.0

  def as_log_record(self) -> dict[str, Any]:
    record = {k: v for k, v in dataclasses.asdict(self).items()
              if k != "element"}
    record["discovered_labels"] = list(self.discovered_labels)
    return record


# ---------------------------------------------------------------------------
# A1.5 - contextual history
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Bucket:
  probes: int = 0
  aligned: int = 0
  alignment_seen: int = 0
  hits: int = 0
  hit_seen: int = 0
  rollback_ok: int = 0
  rollback_seen: int = 0
  cumulative_ig: float = 0.0


class ContextualHistoryTable:
  """History aggregated over element *kinds*, not individual elements.

  Exact-node history is almost always empty on the screen that matters: the
  agent reaches most screens once. Aggregating by (probe type, role, need
  type) gives a candidate a prior the first time it is seen - "menu probes on
  list screens have rolled back 84% of the time" transfers across apps
  because it is a statement about the interaction kind, not about any app.

  Never keyed on raw element text, which would make it app-specific and would
  not generalise past the exact wording.
  """

  def __init__(self) -> None:
    self._buckets: dict[tuple[str, str, str], _Bucket] = {}

  @staticmethod
  def key(probe_type: str, role: str, need_type: str) -> tuple[str, str, str]:
    return (probe_type, role, need_type)

  def _bucket(self, key: tuple[str, str, str]) -> _Bucket:
    return self._buckets.setdefault(key, _Bucket())

  def record_probe(self, key, *, rollback_ok: bool, realized_ig: float = 0.0) -> None:
    b = self._bucket(key)
    b.probes += 1
    b.rollback_seen += 1
    b.rollback_ok += int(rollback_ok)
    b.cumulative_ig += realized_ig

  def record_alignment(self, key, aligned: bool) -> None:
    b = self._bucket(key)
    b.alignment_seen += 1
    b.aligned += int(aligned)

  def record_execution(self, key, matched: bool) -> None:
    b = self._bucket(key)
    b.hit_seen += 1
    b.hits += int(matched)

  def lookup(self, key) -> dict[str, Any]:
    b = self._buckets.get(key)
    if b is None:
      return {"contextual_probe_count": 0}
    return {
        "contextual_probe_count": b.probes,
        "contextual_alignment_rate":
            b.aligned / b.alignment_seen if b.alignment_seen else None,
        "contextual_execution_hit_rate":
            b.hits / b.hit_seen if b.hit_seen else None,
        "contextual_mean_realized_ig":
            b.cumulative_ig / b.probes if b.probes else None,
        "contextual_rollback_success_rate":
            b.rollback_ok / b.rollback_seen if b.rollback_seen else None,
    }


# ---------------------------------------------------------------------------
# A1.7 - safety gate
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SafetyVerdict:
  allowed: bool
  reason: str = ""


class SafetyGate:
  """Hard feasibility filter, applied before any ranking.

  Separate from the scorer on purpose. Ranking is relative - when every
  candidate scores badly the least-bad one still gets probed, which is
  correct for usefulness and wrong for safety. A probe that cannot be undone
  does not cost a worse choice, it costs the rest of the episode.
  """

  def __init__(self, config: ScoringConfig = DEFAULT_SCORING):
    self._config = config

  def evaluate(self, row: CandidateInformationRow) -> SafetyVerdict:
    if row.blocked_element:
      return SafetyVerdict(False, "element blocked after an earlier failure")
    if row.blocked_recovery_context:
      return SafetyVerdict(False, f"{row.probe_type} already unrecoverable here")
    if row.cross_package_history:
      return SafetyVerdict(False, "probe historically left the app package")
    if row.risk_level.upper() in ("HIGH", "IRREVERSIBLE"):
      return SafetyVerdict(False, f"risk_level={row.risk_level}")
    if row.estimated_recoverability < self._config.min_recoverability:
      return SafetyVerdict(
          False,
          f"recoverability {row.estimated_recoverability:.2f} < "
          f"{self._config.min_recoverability:.2f}")
    return SafetyVerdict(True)


# ---------------------------------------------------------------------------
# A1 - matrix builder
# ---------------------------------------------------------------------------


def _role_of_class(class_name: str, *, scrollable: bool = False,
                   in_navigation_drawer: bool = False) -> str:
  """Coarse interaction role, from the framework widget class.

  Split out from `_role_of` so callers that only have a serialised class name
  (the probe trace records strings, not UiElements) key the contextual history
  table exactly the way the matrix does - the two must agree or the statistics
  are written under one key and read under another.
  """
  cls = (class_name or "").rsplit(".", 1)[-1].lower()
  if scrollable or "recycler" in cls or "listview" in cls or "scroll" in cls:
    return "container"
  if "edit" in cls:
    return "input"
  if "checkbox" in cls or "switch" in cls or "radio" in cls or "toggle" in cls:
    return "toggle"
  if "image" in cls:
    return "icon"
  if "button" in cls:
    return "button"
  if in_navigation_drawer:
    return "menu_item"
  return "generic"


def _role_of(element: UiElement) -> str:
  return _role_of_class(element.class_name, scrollable=element.scrollable,
                        in_navigation_drawer=element.in_navigation_drawer)


def _need_type(information_need: Mapping[str, Any]) -> str:
  """Coarse bucket of what the model currently lacks, for A1.5 keying."""
  slots = information_need.get("required_information_slots") or ()
  if information_need.get("target_entity") and slots:
    return "entity_slot"
  if slots:
    return "slot"
  if information_need.get("target_entity"):
    return "entity"
  return "none"


def _tokens(text: str) -> set[str]:
  return {t for t in "".join(
      ch.lower() if ch.isalnum() else " " for ch in text).split() if len(t) > 1}


def _overlap(a: set[str], b: set[str]) -> float:
  return len(a & b) / len(a) if a else 0.0


class StateGraphInformationMatrix:
  """Builds ``CandidateInformationRow``s for the candidates on one screen."""

  def __init__(self,
               config: ScoringConfig = DEFAULT_SCORING,
               contextual_history: ContextualHistoryTable | None = None):
    self._config = config
    self.contextual_history = contextual_history or ContextualHistoryTable()

  def build(
      self,
      current_state: Any,
      current_node_id: str,
      candidate_elements: Sequence[UiElement],
      graph_snapshot: GraphView | None,
      information_need: Mapping[str, Any],
      recovery_history: Mapping[str, Any] | None = None,
      recent_nodes: Sequence[str] = (),
      probe_type_of=None,
      known_good_identities: Mapping[str, int] | None = None,
  ) -> list[CandidateInformationRow]:
    recovery_history = recovery_history or {}
    blocked_elements = set(recovery_history.get("blocked_element_identities") or ())
    blocked_contexts = set(recovery_history.get("blocked_recovery_contexts") or ())
    cross_package = set(recovery_history.get("cross_package_identities") or ())
    activity = getattr(getattr(current_state, "activity", None), "component", "") or ""
    screen_w = max(1, int(getattr(current_state, "screen_width", 0) or 1440))
    screen_h = max(1, int(getattr(current_state, "screen_height", 0) or 2560))

    node_features = self._node_features(
        current_node_id, graph_snapshot, len(candidate_elements))
    by_identity = self._history_index(current_node_id, graph_snapshot)
    need_tokens = self._need_tokens(information_need)
    need_type = _need_type(information_need)

    rows: list[CandidateInformationRow] = []
    for element in candidate_elements:
      probe_type = probe_type_of(element) if probe_type_of else (
          "SCROLL" if element.scrollable else "TAP_NAV")
      role = _role_of(element)
      row = CandidateInformationRow(
          element=element,
          element_identity=element.identity,
          text=element.text,
          content_desc=element.content_desc,
          role=role,
          probe_type=probe_type,
          norm_x=(element.bounds[0] + element.bounds[2]) / 2 / screen_w,
          norm_y=(element.bounds[1] + element.bounds[3]) / 2 / screen_h,
          clickable=element.clickable,
          scrollable=element.scrollable,
          selected=element.selected,
          checked=element.checked,
          **node_features,
      )
      row.session_alignment_hits = int((known_good_identities or {}).get(
          element.identity, 0))
      self._fill_exact_history(row, by_identity.get(
          _match_key(element.identity, probe_type, role, current_node_id)))
      self._fill_destination(row, graph_snapshot, recent_nodes)
      for key, value in self.contextual_history.lookup(
          ContextualHistoryTable.key(probe_type, role, need_type)).items():
        setattr(row, key, value)
      self._fill_need(row, element, need_tokens, information_need)
      self._fill_safety(row, element, activity, blocked_elements,
                        blocked_contexts, cross_package)
      self._fill_cost(row)
      rows.append(row)
    return rows

  # -- A1.2 ----------------------------------------------------------------
  def _node_features(self, node_id, snapshot, candidate_count: int) -> dict[str, Any]:
    if snapshot is None:
      return {"candidate_element_count": candidate_count}
    edge_ids = list(snapshot.outgoing.get(node_id, ()))
    valid = [e for e in edge_ids
             if snapshot.edges[e].get("status") not in ("INVALID", "STALE")]
    explored = len({snapshot.edges[e]["action"].get("element_identity")
                    for e in edge_ids} - {None, ""})
    return {
        "node_visit_count": snapshot.node_visits.get(node_id, 0),
        "node_decision_entropy": snapshot.node_entropy.get(node_id, math.inf),
        "outgoing_edge_count": len(edge_ids),
        "valid_outgoing_edge_count": len(valid),
        "explored_element_count": explored,
        "candidate_element_count": candidate_count,
        "exploration_coverage": explored / candidate_count if candidate_count else 0.0,
        "graph_generation": snapshot.generation,
    }

  # -- A1.3 ----------------------------------------------------------------
  def _history_index(self, node_id, snapshot) -> dict[tuple, Mapping[str, Any]]:
    """Index this node's outgoing edges by what identifies the *action*.

    Keyed on element identity + action type + role + source node rather than
    on the stored edge id, because the edge id includes the destination: the
    same control leading to two different screens (a list row whose target
    depends on scroll position) splits into two edges, and looking the
    candidate up by edge id would then find neither.
    """
    if snapshot is None:
      return {}
    index: dict[tuple, Mapping[str, Any]] = {}
    for edge_id in snapshot.outgoing.get(node_id, ()):
      edge = snapshot.edges[edge_id]
      action = edge.get("action", {})
      identity = str(action.get("element_identity", ""))
      if not identity:
        continue
      key = _match_key(identity, str(action.get("probe_type", "")),
                       str(action.get("role", "")), node_id)
      previous = index.get(key)
      # Same action, several recorded destinations: keep the better-evidenced
      # one rather than whichever hashed last.
      if previous is None or (edge.get("probe_count") or 0) > (previous.get("probe_count") or 0):
        index[key] = edge
    return index

  def _fill_exact_history(self, row, edge: Mapping[str, Any] | None) -> None:
    if edge is None:
      row.has_exact_history = False   # statistics stay None: unknown, not 0
      return
    row.has_exact_history = True
    row.edge_id = edge.get("edge_id")
    row.edge_status = edge.get("status")
    row.confidence = edge.get("confidence")
    row.probe_count = edge.get("probe_count") or 0
    row.inference_alignment_count = edge.get("inference_alignment_count") or 0
    row.execution_hit_count = edge.get("execution_hit_count") or 0
    row.execution_miss_count = edge.get("execution_miss_count") or 0
    row.skip_attempt_count = edge.get("skip_attempt_count") or 0
    row.skip_success_count = edge.get("skip_success_count") or 0
    row.rollback_success_count = edge.get("rollback_success_count") or 0
    row.rollback_failure_count = edge.get("rollback_failure_count") or 0
    row.known_inverse_level = edge.get("inverse_level") or None
    row.discovered_labels = tuple(edge.get("discovered_labels") or ())
    row.destination_node_id = edge.get("dst_node")
    row.destination_known = bool(edge.get("dst_node"))

    aligned_seen = row.inference_alignment_count + row.execution_miss_count
    row.alignment_rate = (row.inference_alignment_count / aligned_seen
                          if aligned_seen else None)
    hit_seen = row.execution_hit_count + row.execution_miss_count
    row.execution_hit_rate = (row.execution_hit_count / hit_seen
                              if hit_seen else None)
    row.skip_success_rate = (row.skip_success_count / row.skip_attempt_count
                             if row.skip_attempt_count else None)
    roll_seen = row.rollback_success_count + row.rollback_failure_count
    row.rollback_success_rate = (row.rollback_success_count / roll_seen
                                 if roll_seen else None)
    row.mean_realized_ig = ((edge.get("cumulative_realized_ig") or 0.0) / row.probe_count
                            if row.probe_count else None)
    row.mean_exploration_cost = ((edge.get("cumulative_exploration_cost") or 0.0)
                                 / row.probe_count if row.probe_count else None)

  # -- A1.4 ----------------------------------------------------------------
  def _fill_destination(self, row, snapshot, recent_nodes) -> None:
    dst = row.destination_node_id
    if snapshot is None or not dst:
      return
    out = list(snapshot.outgoing.get(dst, ()))
    row.destination_visit_count = snapshot.node_visits.get(dst, 0)
    row.destination_entropy = snapshot.node_entropy.get(dst)
    row.destination_out_degree = len(out)
    row.destination_valid_out_degree = sum(
        1 for e in out
        if snapshot.edges[e].get("status") not in ("INVALID", "STALE"))
    row.destination_subtree_size = _subtree_size(snapshot, dst)
    row.destination_known_label_count = len(row.discovered_labels)
    row.destination_in_recent_path = dst in set(recent_nodes)

  # -- A1.6 ----------------------------------------------------------------
  @staticmethod
  def _need_tokens(need: Mapping[str, Any]) -> dict[str, set[str]]:
    return {
        "target": _tokens(str(need.get("target_entity") or "")),
        "slots": _tokens(" ".join(need.get("required_information_slots") or ())),
        "affordance": _tokens(" ".join(need.get("expected_affordances") or ())),
        "risk": _tokens(" ".join(need.get("risk_keywords") or ())),
    }

  def _fill_need(self, row, element, need_tokens, need) -> None:
    label = _tokens(f"{element.text} {element.content_desc}")
    row.target_match = _overlap(need_tokens["target"], label)
    row.expected_affordance_match = _overlap(need_tokens["affordance"], label)
    row.unresolved_information_match = _overlap(need_tokens["slots"], label)
    row.risk_conflict = _overlap(need_tokens["risk"], label)
    expected = {str(a).lower() for a in (need.get("expected_action_types") or ())}
    row.candidate_action_type_match = float(
        not expected or row.probe_type.lower() in expected
        or ("scroll" in expected) == row.scrollable)

  # -- A1.7 ----------------------------------------------------------------
  def _fill_safety(self, row, element, activity, blocked_elements,
                   blocked_contexts, cross_package) -> None:
    row.blocked_element = element.identity in blocked_elements
    row.blocked_recovery_context = f"{activity}|{row.probe_type}" in blocked_contexts
    row.cross_package_history = element.identity in cross_package
    row.historical_rollback_success_rate = row.rollback_success_rate
    if row.risk_conflict > 0:
      row.risk_level = "HIGH"
    prior = self._config.rollback_prior.get(
        row.probe_type, self._config.generic_rollback_prior)
    contextual = row.contextual_rollback_success_rate
    if contextual is not None:
      prior = contextual
    roll_seen = ((row.rollback_success_count or 0) + (row.rollback_failure_count or 0)
                 if row.has_exact_history else 0)
    row.estimated_recoverability = _shrink(
        row.rollback_success_count if row.has_exact_history else None,
        roll_seen, prior, self._config.prior_strength)
    if row.known_inverse_level in ("NOOP", "INVERSE"):
      # A recorded cheap inverse is direct evidence this transition is
      # reversible without the recovery ladder.
      row.estimated_recoverability = max(row.estimated_recoverability, 0.9)

  # -- A1.8 ----------------------------------------------------------------
  def _fill_cost(self, row) -> None:
    cfg = self._config
    if row.mean_exploration_cost:
      row.expected_probe_latency = row.mean_exploration_cost
    else:
      row.expected_probe_latency = cfg.probe_latency_prior.get(
          row.probe_type, cfg.generic_probe_latency)
    fail_p = 1.0 - row.estimated_recoverability
    row.expected_rollback_latency = (
        cfg.rollback_latency + fail_p * cfg.rollback_failure_penalty_s)
    row.expected_total_exploration_cost = (
        row.expected_probe_latency + row.expected_rollback_latency)


def _match_key(identity: str, probe_type: str, role: str, node_id: str) -> tuple:
  return (identity, probe_type, role, node_id)


def _subtree_size(snapshot: GraphView, node_id: str, max_nodes: int = 32) -> int:
  """Reachable nodes below `node_id`, bounded so a cyclic graph terminates."""
  seen = {node_id}
  frontier = [node_id]
  while frontier and len(seen) < max_nodes:
    current = frontier.pop()
    for edge_id in snapshot.outgoing.get(current, ()):
      dst = snapshot.edges[edge_id].get("dst_node")
      if dst and dst not in seen:
        seen.add(dst)
        frontier.append(dst)
  return len(seen) - 1


# ---------------------------------------------------------------------------
# A3 - the scorer
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ScoredCandidate:
  row: CandidateInformationRow
  utility: float
  components: dict[str, float]


class PredictiveElementScorer:
  """Reduces a row to ``path_probability * expected_IG / expected_cost``.

  Interpretable and untrained by design: every term is a rate the graph
  actually measured, combined by an explicit formula, and each component is
  logged separately so a regression can be attributed to one term rather than
  to "the score went down". The interface is what matters for later work -
  a learned ``SmallModelPredictiveEstimator`` can replace this class without
  touching the pipeline around it.
  """

  def __init__(self, config: ScoringConfig = DEFAULT_SCORING):
    self._config = config

  def path_probability(self, row: CandidateInformationRow) -> float:
    """How likely this candidate is on the path the model will really take.

    Grounded first in alignment history - the number of times a probe of this
    transition later matched the authoritative model's own choice - because
    that is the only signal that directly measures the thing being predicted.
    """
    cfg = self._config
    need = max(row.target_match, row.unresolved_information_match,
               row.expected_affordance_match)
    prior = cfg.path_prior_unexplored + cfg.need_match_weight * need
    prior = min(0.95, prior)
    if row.contextual_alignment_rate is not None:
      prior = 0.5 * prior + 0.5 * row.contextual_alignment_rate
    if row.session_alignment_hits:
      # Directly observed, this episode, on this screen: the model really did
      # choose this control here. Nothing else in the row is evidence of the
      # same kind, so it dominates rather than being averaged in.
      prior = max(prior, min(0.95, 0.6 + 0.1 * row.session_alignment_hits))
    if not row.has_exact_history:
      return prior
    seen = (row.inference_alignment_count or 0) + (row.execution_miss_count or 0)
    return _shrink(row.inference_alignment_count, seen, prior, cfg.prior_strength)

  def expected_information_gain(self, row: CandidateInformationRow) -> float:
    """What probing this is expected to add to the graph, not to the screen.

    ``new_element_count`` is kept only as ``ui_novelty_score``: a screen full
    of unfamiliar labels is not the same as an answer to what the model is
    stuck on. The dominant term is task relevance, and a transition whose
    destination is already recorded is discounted hard - re-probing it can
    only confirm what the graph holds, which is why a well-covered node
    pushes exploration onto its untried candidates.
    """
    cfg = self._config
    row.ui_novelty_score = min(
        1.0, (row.destination_known_label_count or 0) / cfg.novelty_scale)
    relevance = max(row.target_match, row.unresolved_information_match)
    if row.has_exact_history and row.mean_realized_ig is not None:
      base = row.mean_realized_ig
    elif row.contextual_mean_realized_ig is not None:
      base = 0.5 * row.contextual_mean_realized_ig + 0.5 * cfg.ig_prior_unexplored
    else:
      base = cfg.ig_prior_unexplored
    gain = base * (0.5 + relevance) + 0.25 * row.ui_novelty_score
    if row.destination_known:
      gain *= cfg.known_destination_discount
    if (row.probe_count or 0) >= 2 and relevance == 0.0 and not row.discovered_labels:
      # Probed repeatedly, never surfaced anything the task needs (A5 case 5).
      gain *= cfg.irrelevant_branch_penalty
    if row.destination_in_recent_path:
      gain *= 0.5
    return max(0.0, gain)

  def score(self, row: CandidateInformationRow) -> ScoredCandidate:
    p = self.path_probability(row)
    ig = self.expected_information_gain(row)
    cost = max(0.5, row.expected_total_exploration_cost)
    row.path_probability = p
    row.expected_information_gain = ig
    row.predictive_value = p * ig
    row.utility = row.predictive_value / cost
    return ScoredCandidate(row, row.utility, {
        "path_probability": p,
        "expected_information_gain": ig,
        "predictive_value": row.predictive_value,
        "expected_cost": cost,
        "recoverability": row.estimated_recoverability,
        "ui_novelty_score": row.ui_novelty_score,
    })

  def rank(self, rows: Iterable[CandidateInformationRow]) -> list[ScoredCandidate]:
    return sorted((self.score(r) for r in rows),
                  key=lambda s: s.utility, reverse=True)
