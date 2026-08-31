"""Turns local graph structure into a few sentences of prompt context.

This is the third consumption path for exploration results. Skipping
inference requires near-certainty and therefore fires rarely; without this
module everything the graph learned in every other step was thrown away.

The pipeline is retrieve -> filter -> score -> compress -> template, all
deterministic. No model is called: a summarisation call would cost roughly
what the inference it is meant to help costs, which would defeat the point.

Two hard-won constraints shape the output format:

* **Facts, never recommendations.** An earlier briefing wrote "Tapping X
  opens Y", which reads as advice; the model obeyed it and re-tapped
  controls it had already used (MarkorCreateFolder looped between "+" and the
  filename field for ten steps, 2026-08-30). Everything here is phrased as an
  observation, and what has already been done is stated explicitly.
* **No absence claims without coverage.** "Reviews has no phone number" is a
  claim about the whole subtree; exploration has seen one probe of it. The
  distiller says "no phone-related evidence was observed under Reviews",
  which is what the graph can actually support.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Iterable, Mapping, Sequence

from android_world.parallel_exploration.state_graph_information import GraphView


@dataclasses.dataclass(frozen=True)
class DistillerConfig:
  max_graph_facts: int = 3
  max_graph_context_tokens: int = 64
  # Below this utility a fact is weaker than the prompt space it costs.
  min_fact_utility: float = 0.15
  # Steps after which an observation is treated as possibly stale.
  freshness_half_life_generations: float = 6.0


DEFAULT_DISTILLER = DistillerConfig()

# Certainty tiers, in words rather than numbers: exposing "confidence=0.63"
# invites the model to do arithmetic on a calibration it cannot see.
_STATUS_CERTAINTY = {
    "REUSABLE": ("Verified", 1.0),
    "VERIFIED": ("Verified", 0.95),
    "INFERENCE_ALIGNED": ("Observed", 0.75),
    "SPECULATIVE": ("Tentative", 0.45),
    "UNEXPLORED": ("Tentative", 0.3),
}


@dataclasses.dataclass
class GraphFact:
  fact_type: str            # VERIFIED | OBSERVED | DONE | NO_RELEVANT_EVIDENCE
  action_label: str
  source_edge_id: str
  evidence_labels: tuple[str, ...] = ()
  edge_status: str = ""
  certainty: str = "Tentative"
  certainty_weight: float = 0.0
  need_match: float = 0.0
  historical_utility: float = 0.0
  freshness: float = 1.0
  already_taken: bool = False
  risk_level: str = "LOW"
  utility_score: float = 0.0

  def render(self) -> str:
    if self.fact_type == "DONE":
      return f"{self.action_label} already used."
    if self.fact_type == "NO_RELEVANT_EVIDENCE":
      return f"no relevant evidence observed under {self.action_label}."
    if self.evidence_labels:
      return f"{self.action_label} -> {{{', '.join(self.evidence_labels)}}}."
    return f"{self.action_label} leads to another screen."


def _tokens(text: str) -> set[str]:
  return {t for t in "".join(
      ch.lower() if ch.isalnum() else " " for ch in text).split() if len(t) > 1}


# Sentence connectives the model starts clauses with. A capitalised single
# word otherwise looks exactly like an entity name, and "Need: Therefore."
# is the result. This is English grammar, not anything about an app or a
# benchmark, so it stays a fixed list.
_DISCOURSE_WORDS = frozenset({
    "therefore", "however", "then", "next", "so", "thus", "also", "first",
    "finally", "since", "because", "now", "here", "this", "that", "it",
})

_MAX_NEED_ITEMS = 4
_MAX_NEED_ITEM_CHARS = 40


def _is_entity_like(item: str) -> bool:
  """Is this a thing the model is looking for, or a piece of its own prose?

  parse_reasoning_prior mines the model's <THINK> text, and clause fragments
  come out alongside real entities - observed 2026-08-31 on
  RecipeDeleteMultipleRecipes: "s detail page. I will click on the",
  "card to proceed", "Therefore". Restating the model's own half-sentences
  back to it is worse than saying nothing. Filtered by shape rather than by a
  word list: an entity is short, has no internal sentence punctuation, and
  does not begin mid-clause.
  """
  if not item or len(item) > _MAX_NEED_ITEM_CHARS:
    return False
  words = item.split()
  if not 1 <= len(words) <= 4:
    return False
  if any(mark in item for mark in ".,;:!?"):
    return False
  if item.lower() in _DISCOURSE_WORDS:
    return False
  return item[0].isupper() or item[0].isdigit() or len(words) == 1


def _need_line(need: Mapping[str, Any]) -> list[str]:
  """The Need: line, deduplicated and capped.

  parse_reasoning_prior deliberately over-generates: it emits the target
  entity, the same entity again as a slot, and its individual words, so that
  the exploration ranker has several chances to match. Rendered verbatim that
  produced lines like "Need: Zucchini Noodles with Pesto, Zucchini Noodles
  with Pesto, Zucchini, Noodles, Pesto." - 20 tokens of the budget spent
  restating the goal the model can already see. Substrings of an item already
  listed add nothing, so they are dropped.
  """
  items: list[str] = []
  raw = [str(need.get("target_entity") or "")]
  raw += [str(s) for s in (need.get("required_information_slots") or ())]
  for item in raw:
    item = item.strip().rstrip(".")
    if not _is_entity_like(item):
      continue
    lowered = item.lower()
    if any(lowered in kept.lower() for kept in items):
      continue
    items = [kept for kept in items if kept.lower() not in lowered]
    items.append(item)
    if len(items) >= _MAX_NEED_ITEMS:
      break
  return [f"Need: {', '.join(items)}."] if items else []


def _action_label(edge: Mapping[str, Any]) -> str:
  """A name for the control this edge acts on, as the user would see it."""
  action = edge.get("action", {})
  anchor = str(action.get("element_identity", "")).split("|")
  # identity is resource_id|text|content_desc|class|bounds - prefer the two
  # human-readable middle fields over the resource id.
  name = next((part for part in anchor[1:3] if part.strip()), "")
  if name:
    return name.strip()
  if action.get("x") is not None:
    return f"the control at ({action['x']},{action['y']})"
  return "an unlabeled control"


class GraphDistiller:
  """Deterministic local-graph -> compact prompt context."""

  def __init__(self, config: DistillerConfig = DEFAULT_DISTILLER):
    self._config = config

  # -- retrieve -----------------------------------------------------------
  def _retrieve(self, node_id: str, snapshot: GraphView) -> list[Mapping[str, Any]]:
    """Outgoing edges of the current node only (B2).

    One hop, not the task graph. Everything beyond the screen in front of the
    model is context it cannot act on this step, and prompt space spent on it
    is space not spent on the screenshot.
    """
    return [snapshot.edges[e] for e in snapshot.outgoing.get(node_id, ())]

  # -- filter + score -----------------------------------------------------
  def _to_fact(self, edge, need_tokens, taken_edges, generation) -> GraphFact | None:
    status = str(edge.get("status", ""))
    if status in ("INVALID", "STALE"):
      return None
    label = _action_label(edge)
    if not label or label.startswith(("an unlabeled", "the control at (")):
      # A control the graph can only name by coordinates cannot be found in a
      # screenshot by the model, so no fact about it is actionable - "the
      # control at (540,605) already used" was a whole injection's content on
      # MarkorDeleteAllNotes (2026-08-31).
      return None
    certainty, weight = _STATUS_CERTAINTY.get(status, ("Tentative", 0.3))
    labels = tuple(str(l) for l in (edge.get("discovered_labels") or ()) if l)[:4]
    already = edge.get("edge_id") in taken_edges

    need_match = 0.0
    if need_tokens:
      pool = _tokens(" ".join(labels) + " " + label)
      need_match = len(need_tokens & pool) / len(need_tokens)

    if already:
      fact_type = "DONE"
    elif not labels:
      # A transition was recorded but nothing was learned about where it goes.
      # Reported as absence of evidence, never as absence of the answer (B7),
      # and only when a probe actually ran: an edge that exists without ever
      # having been probed supports no claim about its destination at all.
      if not (edge.get("probe_count") or 0):
        return None
      if str(edge.get("action", {}).get("action_type", "")).lower() == "open_app":
        # "No relevant evidence observed under Broccoli" - where Broccoli is
        # the app itself - is true and completely useless, and it recurred at
        # every step because launching the app is an edge like any other.
        # A statement about a whole app is not local graph knowledge.
        return None
      fact_type = "NO_RELEVANT_EVIDENCE"
    elif status in ("REUSABLE", "VERIFIED"):
      fact_type = "VERIFIED"
    else:
      fact_type = "OBSERVED"

    age = max(0, generation - int(edge.get("last_updated_generation") or 0))
    freshness = math.exp(-age / max(1e-6, self._config.freshness_half_life_generations))

    hits = edge.get("execution_hit_count") or 0
    misses = edge.get("execution_miss_count") or 0
    skips_ok = edge.get("skip_success_count") or 0
    skips = edge.get("skip_attempt_count") or 0
    historical = 0.5
    if hits + misses:
      historical = hits / (hits + misses)
    elif skips:
      historical = skips_ok / skips

    fact = GraphFact(
        fact_type=fact_type, action_label=label,
        source_edge_id=str(edge.get("edge_id", "")),
        evidence_labels=labels, edge_status=status,
        certainty=certainty, certainty_weight=weight,
        need_match=need_match, historical_utility=historical,
        freshness=freshness, already_taken=already,
        risk_level=str(edge.get("risk_level", "LOW")),
    )
    # B5. A DONE fact earns its place by preventing a repeat, not by matching
    # the need, so it is not multiplied down to nothing by need_match.
    if fact_type == "DONE":
      fact.utility_score = 0.4 * freshness
    elif fact_type == "NO_RELEVANT_EVIDENCE":
      # Worth stating - it is the only thing that stops the same dead branch
      # being re-explored - but it answers nothing, so it ranks below any
      # positive fact and enters only when the budget has room left.
      fact.utility_score = 0.3 * freshness
    else:
      # The floor term is what a fact is worth before any need match: a
      # verified observation of where a control leads is useful even on a step
      # where parse_reasoning_prior extracted nothing, and 0.2 put exactly
      # that case just under min_fact_utility.
      fact.utility_score = (
          (0.25 + need_match) * weight * (0.5 + 0.5 * historical) * freshness)
    return fact

  # -- compress + template ------------------------------------------------
  def distill(
      self,
      current_node_id: str,
      graph_snapshot: GraphView | None,
      information_need: Mapping[str, Any] | None = None,
      taken_edges: Iterable[str] = (),
      recent_nodes: Sequence[str] = (),
      token_budget: int | None = None,
  ) -> str:
    del recent_nodes  # reserved: cycle context is handled by the skip gate.
    if graph_snapshot is None:
      return ""
    need = information_need or {}
    need_tokens = _tokens(
        f"{need.get('target_entity') or ''} "
        f"{' '.join(need.get('required_information_slots') or ())}")
    taken = set(taken_edges)
    budget = token_budget or self._config.max_graph_context_tokens

    facts = [f for f in (
        self._to_fact(e, need_tokens, taken, graph_snapshot.generation)
        for e in self._retrieve(current_node_id, graph_snapshot)) if f]
    facts = [f for f in facts if f.utility_score >= self._config.min_fact_utility]
    facts.sort(key=lambda f: f.utility_score, reverse=True)

    chosen: list[GraphFact] = []
    used = 4  # the "[Memory]" header and the Need line's own overhead
    negatives = 0
    for fact in facts:
      if len(chosen) >= self._config.max_graph_facts:
        break
      if fact.fact_type == "NO_RELEVANT_EVIDENCE":
        # At most one. Several of these read as a list of dead ends and
        # crowded out every positive fact on RecipeDeleteMultipleRecipes
        # (2026-08-31), where six of seven injections carried nothing else.
        if negatives:
          continue
        negatives += 1
      cost = len(fact.render().split()) + 2
      if used + cost > budget:
        continue
      chosen.append(fact)
      used += cost
    if not any(f.fact_type in ("VERIFIED", "OBSERVED") for f in chosen):
      # B9.3, sharpened by measurement. A positive fact - "X was observed to
      # lead to {A, B}" - is the only kind that tells the model something the
      # screenshot does not. Without one, an injection is a Need: line
      # restating the model's own words plus a dead end it cannot use; across
      # 12 tasks on 2026-08-31 every one of the 20 injections was exactly
      # that, and the arm lost four tasks. Negative and Done facts are
      # supporting detail for a positive finding, never a reason to speak.
      return ""
    return self.render(chosen, need)

  @staticmethod
  def render(facts: Sequence[GraphFact], need: Mapping[str, Any]) -> str:
    lines = ["[Memory]"]
    lines.extend(_need_line(need))
    by_type: dict[str, list[GraphFact]] = {}
    for fact in facts:
      by_type.setdefault(fact.fact_type, []).append(fact)
    for label, key in (("Verified", "VERIFIED"), ("Observed", "OBSERVED"),
                       ("Done", "DONE"), ("Observed", "NO_RELEVANT_EVIDENCE")):
      group = by_type.get(key)
      if group:
        lines.append(f"{label}: " + " ".join(f.render() for f in group))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# B9 - the three-way gate
# ---------------------------------------------------------------------------


NORMAL_INFERENCE = "NORMAL_INFERENCE"
GRAPH_ENHANCED_INFERENCE = "GRAPH_ENHANCED_INFERENCE"
SKIP_INFERENCE = "SKIP_INFERENCE"


@dataclasses.dataclass(frozen=True)
class GateDecision:
  mode: str
  reason: str
  graph_context: str = ""
  reusable_edge: Mapping[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class GateConfig:
  enable_skip: bool = True
  enable_graph_context: bool = True
  max_consecutive_skips: int = 3
  # A node with several plausible next actions is exactly where the model
  # earns its cost; skipping there trades a cheap step for a wrong one.
  # Compared against outgoing-edge count as well, because decision entropy is
  # identically 0 while a node has a single recorded edge and would otherwise
  # pass this test vacuously.
  max_skip_entropy: float = 0.4


class ReasoningGate:
  """Chooses NORMAL / GRAPH_ENHANCED / SKIP for one step, from one snapshot."""

  def __init__(self, distiller: GraphDistiller | None = None,
               config: GateConfig = GateConfig()):
    self._distiller = distiller or GraphDistiller()
    self._config = config

  def decide(
      self,
      current_node_id: str,
      graph_snapshot: GraphView | None,
      information_need: Mapping[str, Any] | None,
      reusable_edge: Mapping[str, Any] | None,
      *,
      taken_edges: Iterable[str] = (),
      recent_nodes: Sequence[str] = (),
      consecutive_skips: int = 0,
  ) -> GateDecision:
    if graph_snapshot is None:
      return GateDecision(NORMAL_INFERENCE, "no graph")

    if self._config.enable_skip and reusable_edge is not None:
      blocked = self._skip_blocked(reusable_edge, current_node_id,
                                   graph_snapshot, recent_nodes,
                                   consecutive_skips)
      if blocked is None:
        return GateDecision(SKIP_INFERENCE, "reusable edge",
                            reusable_edge=reusable_edge)
      reason = blocked
    else:
      reason = "no reusable edge"

    if not self._config.enable_graph_context:
      return GateDecision(NORMAL_INFERENCE, reason)
    context = self._distiller.distill(
        current_node_id, graph_snapshot, information_need,
        taken_edges=taken_edges, recent_nodes=recent_nodes)
    if not context:
      return GateDecision(NORMAL_INFERENCE, f"{reason}; no useful facts")
    return GateDecision(GRAPH_ENHANCED_INFERENCE, reason, graph_context=context)

  def _skip_blocked(self, edge, node_id, snapshot, recent_nodes,
                    consecutive_skips) -> str | None:
    """Why this reusable edge must not be replayed, or None if it may be."""
    if consecutive_skips >= self._config.max_consecutive_skips:
      return "consecutive skip cap reached"
    dst = edge.get("dst_node")
    if dst and dst in set(recent_nodes):
      # Replaying into a screen just visited is how a graph walk turns into a
      # loop; the model is the only thing that can break out of one.
      return "destination forms a recent cycle"
    if str(edge.get("risk_level", "LOW")).upper() in ("HIGH", "IRREVERSIBLE"):
      return "edge risk"
    entropy = snapshot.node_entropy.get(node_id, math.inf)
    out_degree = len(snapshot.outgoing.get(node_id, ()))
    if out_degree > 1 and entropy > self._config.max_skip_entropy:
      return f"node entropy {entropy:.2f} too high"
    return None
