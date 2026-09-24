"""Task-local progressive GUI belief graph."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Any


class EdgeStatus(str, enum.Enum):
  UNEXPLORED = "UNEXPLORED"
  SPECULATIVE = "SPECULATIVE"
  INFERENCE_ALIGNED = "INFERENCE_ALIGNED"
  VERIFIED = "VERIFIED"
  REUSABLE = "REUSABLE"
  STALE = "STALE"
  INVALID = "INVALID"


class NodeStatus(str, enum.Enum):
  COMMITTED = "COMMITTED"
  SPECULATIVE = "SPECULATIVE"
  STALE = "STALE"
  INVALID = "INVALID"


def canonical_action(action: Mapping[str, Any]) -> str:
  """A stable key for "the same decision, taken again".

  Keyed on the control when one is known, not on the tap coordinate. The model
  predicts coordinates in a normalised space and they wobble: on
  ExpenseAddMultiple (2026-09-01) it pressed the same button 11 times as
  (540,1063) six times and (540,1068) five times, and the graph recorded two
  edges of six and five instead of one of eleven. The node was visited 33
  times and its only recorded transition still read one execution, so the
  "has the model done this here before" test could never pass. The same shape
  of bug as putting the destination in the edge id, one level down.

  `control_key` deliberately excludes the element's bounds, which move with
  scroll position and layout, and excludes the coordinate entirely. Action
  type is lower-cased so a probe's CLICK and the agent's click are one thing.
  """
  a = dict(action)
  stable_control = a.get("stable_control_key")
  if stable_control:
    return json.dumps({
        "action_type": str(a.get("action_type", "")).lower(),
        "stable_control_key": stable_control,
        "direction": a.get("direction"),
        "app_name": a.get("app_name"),
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
  control = a.get("control_key")
  if control:
    return json.dumps({
        "action_type": str(a.get("action_type", "")).lower(),
        "control_key": control,
        "text": a.get("text"),
        "direction": a.get("direction"),
        "app_name": a.get("app_name"),
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
  return json.dumps(a, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def control_key_from_identity(identity: str) -> str:
  """resource_id|text|content_desc|class from a UiElement.identity, dropping bounds."""
  parts = str(identity or "").split("|")
  key = "|".join(parts[:4]).strip("|")
  return key if key.strip("|") else ""


def stable_control_key_from_identity(identity: str) -> str:
  """Return a bounds- and dynamic-label-tolerant control key.

  Resource id + class is the preferred identity. For controls without ids,
  stable content-description/text plus class is used. This key is for
  suppressing duplicate exploration and merging repeated observations; the
  richer ``control_key_from_identity`` remains available for relocation and
  diagnostics.
  """
  parts = [part.strip().casefold() for part in str(identity or "").split("|")]
  parts += [""] * max(0, 4 - len(parts))
  resource_id, text, content_desc, class_name = parts[:4]
  if resource_id:
    return f"resource:{resource_id}|class:{class_name}"
  if content_desc:
    return f"desc:{content_desc}|class:{class_name}"
  if text:
    return f"text:{text}|class:{class_name}"
  return ""


@dataclasses.dataclass
class GraphNode:
  node_id: str
  activity: str
  package: str
  visual_signature: str
  structural_signature: str
  # Tolerant screen identity this node is keyed by; the two signatures
  # above are retained as strict, point-in-time verification evidence.
  layout_signature: str = ""
  salient_ui_labels: tuple[str, ...] = ()
  # Every named actionable control on this screen, with how often it has been
  # pressed and probed HERE. The graph used to know only the controls it had an
  # edge for - 0.85 per node, and 37% of visited nodes had none at all -
  # while a screen carries 9 to 14 named controls. Everything downstream read
  # that 0.85, which is why ranking candidates by embedding scored 40% top-1
  # against a 38% random baseline: on one or two candidates there is nothing to
  # rank. With the inventory the candidate set is 9 and random falls to 11%.
  #
  # Each entry: {"label", "control_key", "role", "clicks", "probes"}.
  ui_elements: tuple[Mapping[str, Any], ...] = ()
  semantic_summary: str = ""
  timestamp: float = dataclasses.field(default_factory=time.time)
  status: NodeStatus = NodeStatus.SPECULATIVE
  decision_entropy: float = math.inf
  task_progress: float = 0.0
  freshness: float = 1.0
  # How many times the agent has really stood on this screen. A screen seen
  # again is what makes an authoritative edge out of it reusable: the model
  # already answered "what do I do here" the last time, in this same
  # situation, and that answer cost a real inference we do not want to pay
  # twice. Repetitive tasks revisit constantly (add three expenses, delete
  # several recipes), which is exactly where progressive memory should pay.
  visit_count: int = 0


# Whether decision entropy counts only edges the model has executed. False is
# the historical behaviour: every viable outgoing edge counts, so probe and
# seeded edges raise H and close the skip gate. Set from the runner.
ENTROPY_OVER_EXECUTED_ONLY = False


@dataclasses.dataclass
class GraphEdge:
  edge_id: str
  src_node: str
  action: dict[str, Any]
  dst_node: str | None
  status: EdgeStatus = EdgeStatus.UNEXPLORED
  path_probability: float = 0.0
  confidence: float = 0.0
  expected_information_gain: float = 0.0
  realized_information_gain: float = 0.0
  inference_alignment_count: int = 0
  execution_hit_count: int = 0
  execution_miss_count: int = 0
  rollback_success: bool | None = None
  # Which rung of the recovery ladder actually undid this transition, once it
  # is known. Recovery currently rediscovers this every time: the ladder is
  # walked from the top on every probe, so a screen where NOOP suffices still
  # pays for an INVERSE attempt first, and a screen that genuinely needs
  # BACK_N pays for everything above it. Storing it turns a repeated search
  # into a lookup, and it is also the honest place to record that a
  # transition has a known undo at all - the property that makes a branch
  # safe to explore a second time.
  inverse_level: str = ""
  risk_level: str = "UNKNOWN"
  # A probe selected a control whose visible label is runtime/user data.
  # Keep the observation for diagnostics, but never use it as reusable route
  # guidance. Older serialized graphs default this field to False.
  dynamic_content: bool = False
  exploration_cost: float = 0.0
  # Cumulative history. The single-value fields above record the latest
  # observation; these keep the record, which is what a ranking or a
  # distillation decision actually needs - "rolled back once, failed twice" is
  # a different situation from "rolled back once" even though both leave
  # rollback_success True.
  # Every node this transition has been seen to reach, most recent last.
  observed_destinations: tuple[str, ...] = ()
  probe_count: int = 0
  skip_attempt_count: int = 0
  skip_success_count: int = 0
  rollback_success_count: int = 0
  rollback_failure_count: int = 0
  cumulative_realized_ig: float = 0.0
  cumulative_exploration_cost: float = 0.0
  last_updated_generation: int = 0
  # How many screens the graph knew about the last time the agent itself
  # executed this transition. A replay is licensed by the task iterating, not
  # merely by the action recurring, and the two look identical from the edge
  # alone - see nodes_known in the runner's reusable_edge.
  nodes_at_last_execution: int = 0

  @property
  def alignment_rate(self) -> float | None:
    seen = self.inference_alignment_count + self.execution_miss_count
    return self.inference_alignment_count / seen if seen else None

  @property
  def execution_hit_rate(self) -> float | None:
    seen = self.execution_hit_count + self.execution_miss_count
    return self.execution_hit_count / seen if seen else None

  @property
  def skip_success_rate(self) -> float | None:
    return (self.skip_success_count / self.skip_attempt_count
            if self.skip_attempt_count else None)

  @property
  def rollback_success_rate(self) -> float | None:
    seen = self.rollback_success_count + self.rollback_failure_count
    return self.rollback_success_count / seen if seen else None

  @property
  def mean_realized_ig(self) -> float | None:
    return self.cumulative_realized_ig / self.probe_count if self.probe_count else None

  @property
  def mean_exploration_cost(self) -> float | None:
    return (self.cumulative_exploration_cost / self.probe_count
            if self.probe_count else None)
  # What taking this action actually revealed, as on-screen labels. This is
  # the payload that makes an explored edge useful even when it is not
  # confident enough to reuse for an inference skip: it can still be
  # summarized back into the next prompt so the authoritative model knows
  # what is behind an option without having to spend a step finding out.
  discovered_labels: tuple[str, ...] = ()
  last_updated: float = dataclasses.field(default_factory=time.time)


class ProgressiveBeliefGraph:
  """Persistent within one task; all mutating APIs are thread-safe."""

  def __init__(self, task_id: str):
    self.task_id = task_id
    self.nodes: dict[str, GraphNode] = {}
    self.edges: dict[str, GraphEdge] = {}
    self._outgoing: dict[str, set[str]] = defaultdict(set)
    self._lock = threading.RLock()

  @staticmethod
  def make_node_id(activity: str, layout_signature: str, _unused_visual_signature: str = "") -> str:
    """Graph node identity: activity + interaction-skeleton layout signature.

    Deliberately excludes the strict structural digest and the pHash: those
    change with any on-screen data change, which fragmented a single
    Activity into dozens of never-revisited nodes and prevented progressive
    memory from accumulating (see StateSignature.layout_sig). They remain in
    each GraphNode's fields, and remain the basis of rollback verification,
    where strictness is exactly what is wanted.
    """
    del _unused_visual_signature
    raw = f"{activity}\0{layout_signature}".encode()
    return hashlib.sha256(raw).hexdigest()[:20]

  def upsert_node(self, node: GraphNode, *, visited: bool = False) -> GraphNode:
    with self._lock:
      old = self.nodes.get(node.node_id)
      if old is None:
        node.visit_count = int(visited)
        self.nodes[node.node_id] = node
      else:
        node.timestamp = max(old.timestamp, node.timestamp)
        node.status = old.status if old.status == NodeStatus.COMMITTED else node.status
        node.visit_count = old.visit_count + int(visited)
        self.nodes[node.node_id] = node
      return self.nodes[node.node_id]

  def add_speculative_transition(
      self, src_node: str, action: Mapping[str, Any], dst_node: str | None,
      *, path_probability: float, confidence: float,
      expected_information_gain: float, risk_level: str,
      exploration_cost: float, rollback_success: bool | None = None,
      discovered_labels: tuple[str, ...] = (), inverse_level: str = "",
      dynamic_content: bool = False,
  ) -> GraphEdge:
    with self._lock:
      # Identity is (source screen, action) - the destination is something
      # this transition was observed to do, not part of what it is.
      #
      # Including dst_node split one repeated decision into several edges.
      # RecipeDeleteMultipleRecipes (2026-08-31) is the clean example: on the
      # recipe-detail screen the model chose the same control on all three
      # passes, but each pass left a list with one fewer recipe, so the three
      # landings hashed to three nodes and the graph recorded three separate
      # edges of one execution each instead of one edge of three. Every test
      # of "has the model done this here before" then read 1, and the
      # repetitive tasks that progressive memory exists to serve were exactly
      # the ones it could never fire on.
      key = f"{src_node}\0{canonical_action(action)}"
      edge_id = hashlib.sha256(key.encode()).hexdigest()[:24]
      edge = self.edges.get(edge_id)
      if edge is None:
        edge = GraphEdge(edge_id, src_node, dict(action), dst_node)
        self.edges[edge_id] = edge
        self._outgoing[src_node].add(edge_id)
      if dst_node:
        # Keep the whole set. One destination means the transition is
        # deterministic and a replay can be verified against it; several mean
        # the action is still the right one to take but where it lands depends
        # on state the graph does not model, and the verification has to be
        # correspondingly weaker.
        edge.observed_destinations = tuple(
            dict.fromkeys(edge.observed_destinations + (dst_node,)))
        edge.dst_node = dst_node
      edge.status = EdgeStatus.SPECULATIVE
      edge.path_probability = max(0.0, min(1.0, path_probability))
      edge.confidence = max(edge.confidence, max(0.0, min(1.0, confidence)))
      edge.expected_information_gain = expected_information_gain
      edge.risk_level = risk_level
      edge.dynamic_content = bool(edge.dynamic_content or dynamic_content)
      edge.exploration_cost = exploration_cost
      edge.rollback_success = rollback_success
      if inverse_level:
        edge.inverse_level = inverse_level
      if discovered_labels:
        edge.discovered_labels = tuple(discovered_labels)
      edge.last_updated = time.time()
      return edge

  def record_inference_alignment(self, src_node: str, action: Mapping[str, Any], aligned: bool) -> list[GraphEdge]:
    target = canonical_action(action)
    changed = []
    with self._lock:
      for edge_id in self._outgoing.get(src_node, ()):
        edge = self.edges[edge_id]
        if canonical_action(edge.action) != target:
          continue
        edge.inference_alignment_count += int(aligned)
        if aligned:
          edge.status = EdgeStatus.INFERENCE_ALIGNED
          edge.confidence = min(1.0, 0.55 + 0.15 * edge.inference_alignment_count)
        else:
          edge.confidence *= 0.5
        edge.last_updated = time.time()
        changed.append(edge)
    return changed

  def record_probe(self, edge_id: str, *, rollback_ok: bool, cost_s: float,
                   realized_ig: float = 0.0, generation: int = 0) -> GraphEdge:
    """One speculative probe of this transition, with what it cost and yielded."""
    with self._lock:
      edge = self.edges[edge_id]
      edge.probe_count += 1
      if rollback_ok:
        edge.rollback_success_count += 1
      else:
        edge.rollback_failure_count += 1
      edge.cumulative_exploration_cost += max(0.0, cost_s)
      edge.cumulative_realized_ig += realized_ig
      edge.last_updated_generation = generation
      edge.last_updated = time.time()
      return edge

  def record_skip_result(self, edge_id: str, matched: bool, generation: int = 0) -> GraphEdge:
    """This edge was replayed in place of an inference; did it land as stored?

    Kept separate from record_execution_verification because the two answer
    different questions: that one asks whether a transition behaves as
    recorded, this one asks whether trusting it instead of the model worked
    out. An edge can be a faithful description of the UI and still be the
    wrong thing to do at this point in the task.
    """
    with self._lock:
      edge = self.edges[edge_id]
      edge.skip_attempt_count += 1
      edge.skip_success_count += int(matched)
      edge.last_updated_generation = generation
      edge.last_updated = time.time()
      return edge

  def record_rollback_result(self, edge_id: str, ok: bool, level: str = "") -> GraphEdge:
    with self._lock:
      edge = self.edges[edge_id]
      if ok:
        edge.rollback_success_count += 1
        if level:
          edge.inverse_level = level
      else:
        edge.rollback_failure_count += 1
      edge.rollback_success = ok
      edge.last_updated = time.time()
      return edge

  def record_execution_verification(self, edge_id: str, matched: bool, realized_information_gain: float = 0.0) -> GraphEdge:
    with self._lock:
      edge = self.edges[edge_id]
      if matched:
        edge.execution_hit_count += 1
        edge.status = EdgeStatus.VERIFIED
        edge.confidence = min(1.0, max(edge.confidence, 0.75) + 0.10)
      else:
        edge.execution_miss_count += 1
        edge.status = EdgeStatus.INVALID
        edge.confidence = 0.0
      edge.realized_information_gain = realized_information_gain
      edge.last_updated = time.time()
      return edge

  def promote_children_of_aligned_prefix(self, edge_id: str) -> list[GraphEdge]:
    """Promote safely observed lookahead after its parent matches inference.

    A child is reusable only when the parent action was trusted by real
    evidence - either selected by the authoritative model this round
    (INFERENCE_ALIGNED), or itself a previously promoted child that was
    later skip-executed and its successor verified (VERIFIED) - and the
    speculative child transition was observed with rollback succeeding.
    Accepting VERIFIED parents (not just INFERENCE_ALIGNED) is what lets a
    multi-hop speculative chain extend across several rounds: hop 2 earns
    trust once it has actually been skip-executed and confirmed once, at
    which point hop 3's already-explored children can be promoted too,
    without needing the authoritative model to walk hop 2 again.
    """
    promoted: list[GraphEdge] = []
    with self._lock:
      parent = self.edges[edge_id]
      if parent.status not in {EdgeStatus.INFERENCE_ALIGNED, EdgeStatus.VERIFIED} or not parent.dst_node:
        return promoted
      children = [self.edges[item] for item in self._outgoing.get(parent.dst_node, ())]
      eligible = [
          child for child in children
          if child.status == EdgeStatus.SPECULATIVE
          and child.dst_node is not None
          and child.rollback_success is True
          and child.risk_level in {"SAFE", "LOW"}
      ]
      if not eligible:
        return promoted
      probabilities = [max(1e-6, child.path_probability) for child in eligible]
      self.update_decision_distribution(parent.dst_node, probabilities)
      for child in eligible:
        # Eligibility above is entirely observational, so there is no extra
        # numeric bar here: the parent action was confirmed by the
        # authoritative model (INFERENCE_ALIGNED) or by a verified
        # skip-execution, the child transition was actually performed and its
        # landing state actually seen, the device actually rolled back, and
        # the action is SAFE/LOW risk. "S --a--> S'" is a measurement, not a
        # guess, and the only open question - whether a is the action we want
        # next - is what prefix alignment answers.
        #
        # A confidence cap/threshold used to sit here (2026-08-29). It was
        # meant to stop a zero-evidence shortcut, but it scored an observed
        # transition by the ranker's a priori path_probability, and it capped
        # at 0.85 against a 0.86 gate, which made REUSABLE - the only status
        # the skip path accepts - unreachable in every run. The real defect in
        # the ExpenseDeleteMultiple regression was an edge that never passed
        # prefix alignment at all, which the eligibility filter above is what
        # excludes. Wrong reuse is caught after the fact by
        # SkipInferenceGate.execute_and_verify, which compares the real
        # landing state against dst_node and invalidates the subtree on a
        # mismatch - an observation rather than a prior.
        child.status = EdgeStatus.REUSABLE
        child.last_updated = time.time()
        promoted.append(child)
    return promoted

  def invalidate_subtree(self, node_id: str) -> None:
    with self._lock:
      todo = deque([node_id])
      seen = set()
      while todo:
        current = todo.popleft()
        if current in seen:
          continue
        seen.add(current)
        node = self.nodes.get(current)
        if node:
          node.status = NodeStatus.INVALID
        for edge_id in self._outgoing.get(current, ()):
          edge = self.edges[edge_id]
          edge.status = EdgeStatus.INVALID
          edge.confidence = 0.0
          if edge.dst_node:
            todo.append(edge.dst_node)

  def mark_stale(self, now: float | None = None, max_age_s: float = 120.0) -> None:
    now = time.time() if now is None else now
    with self._lock:
      for node in self.nodes.values():
        age = max(0.0, now - node.timestamp)
        node.freshness = max(0.0, 1.0 - age / max(max_age_s, 1e-6))
        if age > max_age_s and node.status != NodeStatus.INVALID:
          node.status = NodeStatus.STALE
      for edge in self.edges.values():
        if now - edge.last_updated > max_age_s and edge.status != EdgeStatus.INVALID:
          edge.status = EdgeStatus.STALE

  def get_reusable_action(self, node_id: str, *, max_entropy: float, max_age_s: float) -> GraphEdge | None:
    """Best reusable edge out of node_id, or None.

    Reaching REUSABLE already required prefix alignment plus an observed,
    rolled-back transition (see promote_children_of_aligned_prefix), so no
    confidence threshold is applied on top of it. What is still checked here
    is observational: the node must not be stale or invalidated, the screen
    must not present an ambiguous choice (decision entropy), the action must
    be SAFE/LOW risk, and the observation must be recent enough that the UI
    is unlikely to have moved underneath it.
    """
    now = time.time()
    with self._lock:
      node = self.nodes.get(node_id)
      if not node or node.status in {NodeStatus.STALE, NodeStatus.INVALID} or node.decision_entropy > max_entropy:
        return None
      # REUSABLE only. A VERIFIED edge records that an action was executed
      # from this node and where it landed; it does not record that the action
      # was the right one to take here. Accepting VERIFIED gave the skip path
      # edges that had never passed prefix alignment at all, and they were
      # wrong most of the time (2026-08-29: 24 graph skips, 33% landed as
      # predicted; the worst were re-issuing open_app at 25% while the app was
      # already open, from self-loop-ish edges with inference_alignment_count
      # of 0). REUSABLE is reached only through
      # promote_children_of_aligned_prefix, i.e. only after the explorer's
      # guess at this point matched what the authoritative model actually did.
      # Two independent licences to reuse, both observational:
      #
      # 1. REUSABLE - the explorer's guess at this point matched what the
      #    authoritative model then did (prefix alignment), so its lookahead
      #    child is trusted. Costs speculative probing to obtain.
      # 2. VERIFIED on a REVISITED node - the agent has really stood here
      #    before and the model itself chose this action in this same
      #    situation, and the transition was observed. Costs nothing at all:
      #    no probe, no rollback, no device risk. This is what makes
      #    progressive memory pay on repetitive tasks, where the same screen
      #    recurs once per item.
      #
      # A self-loop is excluded from (2) regardless: an action that lands on
      # the same node makes no progress, so replaying it can only stall. That
      # is what re-issuing open_app while the app was already open did, at a
      # 25% landing-match rate (2026-08-29).
      revisited = node.visit_count > 1

      def usable(edge: GraphEdge) -> bool:
        if edge.risk_level not in {"SAFE", "LOW"}:
          return False
        if edge.status == EdgeStatus.REUSABLE:
          # Speculative lookahead: the screen may have moved on since the
          # probe observed it, and nothing re-checks that, so it expires.
          return now - edge.last_updated <= max_age_s
        # Authoritative edge out of a revisited screen. No age limit: the
        # caller resolved node_id from a freshly captured real state, so the
        # screen has just been re-confirmed to be this node by the same
        # layout identity the edge was recorded under. A wall-clock cutoff
        # would only measure how long the task's own loop takes - at ~17s per
        # step the 120s default expires after seven steps, which is shorter
        # than one iteration of most repetitive tasks.
        return (
            revisited
            and edge.status == EdgeStatus.VERIFIED
            and edge.dst_node not in (None, node_id)
        )

      candidates = [
          self.edges[eid] for eid in self._outgoing.get(node_id, ())
          if usable(self.edges[eid])
      ]
      return max(candidates, key=lambda edge: edge.confidence, default=None)

  def get_frontier(self) -> list[GraphEdge]:
    with self._lock:
      return [e for e in self.edges.values() if e.status in {EdgeStatus.UNEXPLORED, EdgeStatus.SPECULATIVE}]

  def get_local_confidence(self, node_id: str) -> float:
    with self._lock:
      edges = [self.edges[eid] for eid in self._outgoing.get(node_id, ()) if self.edges[eid].status != EdgeStatus.INVALID]
      return max((edge.confidence for edge in edges), default=0.0)

  def recompute_decision_entropy(self, node_id: str) -> float:
    """H over the continuations this node is actually known to offer.

    H_i = -sum p_j log p_j over the viable outgoing edges, with p_j from each
    edge's path probability. The point is that H answers a question the
    explorer CAN settle without predicting anything: how much is there left
    to decide here. A screen whose map shows a single viable continuation - a
    confirmation dialog, a one-button step in a wizard - has H = 0, and an
    inference spent there buys nothing, because there is no alternative for
    it to choose between.

    That is a different bet from prefix alignment. Alignment needs the ranker
    to guess which action the model will take, measured at 0-3% (2026-08-31),
    and no amount of probing fixes a predictor that weak. Entropy needs only
    coverage of the screen, which accumulates over repeated visits.

    inf when nothing viable is known yet, so an unmapped screen never looks
    decided.
    """
    with self._lock:
      node = self.nodes.get(node_id)
      if node is None:
        return math.inf
      viable = [
          self.edges[eid] for eid in self._outgoing.get(node_id, ())
          if self.edges[eid].status != EdgeStatus.INVALID
          and self.edges[eid].dst_node not in (None, node_id)
      ]
      if not viable:
        node.decision_entropy = math.inf
        return math.inf
      if ENTROPY_OVER_EXECUTED_ONLY:
        # H is read by the skip gate, which is a claim about the MODEL's
        # behaviour - "the answer here has always been the same" - not about
        # what the app makes possible. Probe edges and edges seeded from
        # cross-task memory are the latter: they say a control exists and
        # leads somewhere, with no model having chosen it. Counting them makes
        # every probe and every remembered screen push the gate shut.
        #
        # Measured 2026-09-10, mean finite H against skips fired, four arms:
        #   semantic 0.224 -> 14   probes-off 0.246 -> 13
        #   probes-on 0.295 ->  7   warm memory 0.324 ->  2
        # H was the only variable that ordered every arm correctly, and both
        # things that raise it - probing and seeding - are exactly the edges
        # with no execution behind them.
        executed = [e for e in viable
                    if (e.execution_hit_count + e.execution_miss_count) > 0]
        if executed:
          viable = executed
      weights = [max(1e-6, edge.path_probability) for edge in viable]
      total = sum(weights)
      entropy = -sum((w / total) * math.log(w / total) for w in weights)
      node.decision_entropy = entropy
      return entropy

  def update_decision_distribution(self, node_id: str, probabilities: Iterable[float]) -> float:
    probs = [max(0.0, float(p)) for p in probabilities]
    total = sum(probs)
    entropy = math.inf if total <= 0 else -sum((p / total) * math.log(p / total) for p in probs if p > 0)
    with self._lock:
      self.nodes[node_id].decision_entropy = entropy
    return entropy

  def to_dict(self) -> dict[str, Any]:
    with self._lock:
      return {
          "task_id": self.task_id,
          "nodes": [{**dataclasses.asdict(n), "status": n.status.value} for n in self.nodes.values()],
          "edges": [{**dataclasses.asdict(e), "status": e.status.value} for e in self.edges.values()],
      }
