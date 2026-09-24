"""Candidate ranking strategies for safe UI probing."""

from __future__ import annotations

import abc
import dataclasses
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


@dataclasses.dataclass(frozen=True)
class UiElement:
  text: str = ""
  content_desc: str = ""
  resource_id: str = ""
  class_name: str = ""
  bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
  clickable: bool = False
  scrollable: bool = False
  checked: bool | None = None
  selected: bool | None = None
  # Structural: any ancestor is a standard AndroidX/Material DrawerLayout or
  # NavigationView (the framework widgets essentially every app's nav-drawer
  # menu is built from), computed from the widget class hierarchy - not a
  # label, resource-id naming convention, or per-app rule.
  in_navigation_drawer: bool = False
  # Which app drew this node. The accessibility dump has always carried it;
  # without it the soft keyboard could only be recognised by resource-id
  # prefix, and its unnamed keys have no id to match (2026-09-08).
  package: str = ""

  @property
  def identity(self) -> str:
    return "|".join((self.resource_id, self.text, self.content_desc,
                     self.class_name, repr(self.bounds)))


@dataclasses.dataclass(frozen=True)
class RankedElement:
  element: UiElement
  rank: int
  score: float


class Ranker(abc.ABC):

  @abc.abstractmethod
  def rank(
      self,
      elements: Sequence[UiElement],
      task: str,
      probed_ids: Iterable[str] = (),
  ) -> list[RankedElement]:
    raise NotImplementedError


class RandomRanker(Ranker):

  def __init__(self, seed: int):
    self._seed = seed

  def rank(
      self,
      elements: Sequence[UiElement],
      task: str,
      probed_ids: Iterable[str] = (),
  ) -> list[RankedElement]:
    del task, probed_ids
    ordered = list(elements)
    random.Random(self._seed).shuffle(ordered)
    return [
        RankedElement(element=element, rank=index, score=0.0)
        for index, element in enumerate(ordered, start=1)
    ]


def _tokens(value: str) -> set[str]:
  return set(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def _weighted_overlap(element_tokens: set[str], weighted_fields: Sequence[tuple[set[str], float]]) -> float:
  if not element_tokens:
    return 0.0
  total = 0.0
  for field_tokens, weight in weighted_fields:
    if not field_tokens:
      continue
    total += weight * len(element_tokens & field_tokens) / len(element_tokens)
  return total


class SimpleRelevanceRanker(Ranker):
  """Token relevance + role prior - repeat penalty, with stable tie-breaking."""

  _ROLE_BONUS = {
      "button": 0.30,
      "edittext": 0.30,
      "textview": 0.15,
      "imagebutton": 0.10,
      "imageview": 0.05,
  }

  def __init__(self, already_probed_penalty: float = 0.10):
    self._already_probed_penalty = already_probed_penalty

  def rank(
      self,
      elements: Sequence[UiElement],
      task: str,
      probed_ids: Iterable[str] = (),
  ) -> list[RankedElement]:
    task_tokens = _tokens(task)
    probed = set(probed_ids)
    scored: list[tuple[float, int, UiElement]] = []
    for original_index, element in enumerate(elements):
      element_tokens = _tokens(f"{element.text} {element.content_desc}")
      overlap = (
          len(element_tokens & task_tokens) / len(element_tokens)
          if element_tokens else 0.0
      )
      role = element.class_name.rsplit(".", 1)[-1].casefold()
      score = overlap + self._ROLE_BONUS.get(role, 0.0)
      if element.identity in probed:
        score -= self._already_probed_penalty
      scored.append((score, original_index, element))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        RankedElement(element=element, rank=index, score=score)
        for index, (score, _, element) in enumerate(scored, start=1)
    ]


class LightweightLinearRanker(Ranker):
  """Fixed multi-hot linear selector used by the V0 feasibility experiment.

  Safety filtering happens before this ranker in ``live_probe``. This class
  only applies ``w^T x + b`` to text, content description, and resource id;
  it deliberately does not add task semantics, embeddings, or graph features.
  The weights file can therefore be the manually initialized ground-truth-like
  prototype shipped under ``gui_exploration_selector/``.
  """

  def __init__(self, weights_path: str):
    from android_world.parallel_exploration.lightweight_selector import LinearModel
    self.model = LinearModel.from_text(weights_path)

  def rank(
      self,
      elements: Sequence[UiElement],
      task: str,
      probed_ids: Iterable[str] = (),
  ) -> list[RankedElement]:
    del task, probed_ids
    scored = [
        (self.model.score(element), position, element)
        for position, element in enumerate(elements)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        RankedElement(element=element, rank=rank, score=score)
        for rank, (score, _, element) in enumerate(scored, start=1)
    ]


class InformationNeedRanker(Ranker):
  """Ranks by structured InformationNeed fields plus verified session memory.

  A single failing probe can burn an entire exploration window on recovery
  (see live_probe.py's recovery ladder), so raising the probe budget does not
  help - the candidate tried first has to actually be the predictive one.
  Two signals drive that choice, in priority order:

  1. ``known_good_identities``: elements previously observed, at this exact
     graph node, to be the action the authoritative model actually picked.
     This is grounded in a verified outcome rather than text similarity, so
     it always wins once available (a form of session-local, cross-round
     exploration memory).
  2. Weighted overlap against InformationNeed's structured fields:
     target_entity / required_information_slots outrank the looser
     current_subgoal sentence, unlike whole-goal-string token overlap.
  """

  _ROLE_BONUS = SimpleRelevanceRanker._ROLE_BONUS

  def __init__(
      self,
      information_need: Mapping[str, Any],
      known_good_identities: Mapping[str, int] | None = None,
      already_probed_penalty: float = 0.10,
  ):
    self._target_tokens = _tokens(str(information_need.get("target_entity") or ""))
    self._slot_tokens = _tokens(" ".join(information_need.get("required_information_slots") or ()))
    self._affordance_tokens = _tokens(" ".join(information_need.get("expected_affordances") or ()))
    self._subgoal_tokens = _tokens(str(information_need.get("current_subgoal") or ""))
    self._known_good = dict(known_good_identities or {})
    self._already_probed_penalty = already_probed_penalty

  def rank(
      self,
      elements: Sequence[UiElement],
      task: str,
      probed_ids: Iterable[str] = (),
  ) -> list[RankedElement]:
    del task  # superseded by the structured InformationNeed fields above.
    probed = set(probed_ids)
    weighted_fields = (
        (self._target_tokens, 3.0),
        (self._slot_tokens, 3.0),
        (self._affordance_tokens, 2.0),
        (self._subgoal_tokens, 1.0),
    )
    scored: list[tuple[float, int, UiElement]] = []
    for original_index, element in enumerate(elements):
      element_tokens = _tokens(f"{element.text} {element.content_desc}")
      score = _weighted_overlap(element_tokens, weighted_fields)
      role = element.class_name.rsplit(".", 1)[-1].casefold()
      score += self._ROLE_BONUS.get(role, 0.0)
      hits = self._known_good.get(element.identity, 0)
      if hits:
        score += 10.0 + hits
      if element.identity in probed:
        score -= self._already_probed_penalty
      scored.append((score, original_index, element))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        RankedElement(element=element, rank=index, score=score)
        for index, (score, _, element) in enumerate(scored, start=1)
    ]


class CoverageRanker(Ranker):
  """Rank by what the graph does NOT know about this screen.

  Every earlier ranker optimised the same objective: guess the control the
  model is about to press, so the probe's landing state can be reused. That
  objective is falsified four ways - doubling the probe budget, relaxing the
  safety filter, cross-task revisiting, and finally uniform random ranking,
  which scored 57/115, among the best arms ever measured. A ranker cannot beat
  chance at predicting the model, so ranking for prediction buys nothing.

  Coverage is a different objective and the graph can answer it reliably:
  which controls on this screen have no outgoing edge yet? The measured
  problem is that the graph's mean out-degree is 0.77 (2026-09-09, 473 screens
  / 362 transitions across 29 packages) - most nodes have no outgoing edge at
  all, so "what can I do here" is unanswerable, which is why injection finds no
  positive fact, skip has no reusable edge, and lookahead never aligns.

  Two signals, in order:

  1. Does this control already have an outgoing edge from this node? Probing
     it again re-learns something known and spends fresh rollback risk.
  2. Can the control be named? An edge whose control has no text and no
     content_desc renders as "the control at (540,605)", and _action_label
     rejects that, so it can never become a fact no matter where it leads -
     55.7% of edges are in that state. Probing an unnameable control adds
     out-degree that the distiller cannot use.
  """

  _ROLE_BONUS = {
      "button": 1.0, "imagebutton": 1.0, "textview": 0.5,
      "imageview": 0.5, "checkbox": 0.5, "switch": 0.5,
  }

  def __init__(
      self,
      known_control_keys: Iterable[str] = (),
      already_probed_penalty: float = 8.0,
      unexplored_bonus: float = 10.0,
      nameable_bonus: float = 3.0,
  ):
    self._known = {k for k in known_control_keys if k}
    self._already_probed_penalty = already_probed_penalty
    self._unexplored_bonus = unexplored_bonus
    self._nameable_bonus = nameable_bonus

  @staticmethod
  def control_key(element: UiElement) -> str:
    """The key the graph stores an edge under: identity minus the bounds."""
    key = "|".join(element.identity.split("|")[:4]).strip("|")
    return key if key.strip("|") else ""

  @staticmethod
  def is_nameable(element: UiElement) -> bool:
    """Would an edge on this control render as something the model can find?"""
    return bool((element.text or "").strip()
                or (element.content_desc or "").strip())

  def rank(
      self,
      elements: Sequence[UiElement],
      task: str,
      probed_ids: Iterable[str] = (),
  ) -> list[RankedElement]:
    del task  # coverage is a property of the graph, not of the goal
    probed = set(probed_ids)
    scored: list[tuple[float, int, UiElement]] = []
    for original_index, element in enumerate(elements):
      score = 0.0
      if self.control_key(element) not in self._known:
        score += self._unexplored_bonus
      if self.is_nameable(element):
        score += self._nameable_bonus
      role = element.class_name.rsplit(".", 1)[-1].casefold()
      score += self._ROLE_BONUS.get(role, 0.0)
      if element.identity in probed:
        score -= self._already_probed_penalty
      scored.append((score, original_index, element))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        RankedElement(element=element, rank=index, score=score)
        for index, (score, _, element) in enumerate(scored, start=1)
    ]


class GraphKeywordRanker(Ranker):
  """Rank by need-relevance MINUS what the graph has already discovered here.

  This is the first ranker that uses the graph at all. Every earlier one
  matched candidates against the model's stated information need and nothing
  else, so the graph contributed exactly zero to exploration targeting - which
  is why "how does graph history improve exploration" had no answer.

  The graph cannot say which control the model is about to press (four
  independent falsifications, uniform random ranking among the best arms ever).
  It can reliably say what is already known: `discovered_labels` records the
  text a probe or an authoritative step found behind a control. Probing toward
  content the graph has already characterised buys nothing, so relevance to the
  need is discounted by similarity to what is already on record:

      score = sim(candidate, need) - discount * max_j sim(candidate, known_j)

  Scores come from a 22M sentence encoder over a socket (`semantic_service`);
  108 candidates cost ~52 ms to encode but jitter to ~190 ms under load, so the
  caller narrows to `rerank_top_k` with a cheap ranker first - 20 candidates
  measure a steady 20 ms, inside the 150 ms an exploration round can afford.

  Falls back to the supplied base ordering whenever the service is unreachable
  or nothing on screen carries a label: an optional model must never be able to
  abort a probe window.
  """

  def __init__(
      self,
      base: Ranker,
      need_text: str,
      known_labels: Iterable[str] = (),
      port: int = 8766,
      rerank_top_k: int = 32,
      discount: float = 0.5,
      visited_summaries: Iterable[str] = (),
      visited_discount: float = 0.0,
  ):
    self._base = base
    self._need = (need_text or "").strip()
    self._known = [str(l).strip() for l in known_labels if str(l).strip()][:24]
    self._port = port
    self._top_k = max(1, rerank_top_k)
    self._discount = discount
    # One sentence per screen this episode has already stood on, most recent
    # last - the walked path, not the whole graph. known_labels says what is
    # on record behind a control; this says where the episode has already
    # been, which is a different kind of waste: a control whose destination
    # reads like a screen already visited leads back, not forward. Default
    # weight 0.0 leaves the measured labels-only ranker bit-for-bit unchanged,
    # so turning it on is a controlled arm rather than a redefinition.
    self._visited = [str(t).strip() for t in visited_summaries
                     if str(t).strip()][:6]
    self._visited_discount = visited_discount
    self.last_reranked = 0  # instrumentation: 0 means the fallback was used
    # Per-candidate breakdown of the most recent rank(), for the event log:
    # (text, need, known_penalty, visited_penalty, final). The scores were
    # invisible before, which made "why was this element probed" unanswerable.
    self.last_scores: list[tuple[str, float, float, float, float]] = []

  @staticmethod
  def _text(element: UiElement) -> str:
    return " ".join(p for p in (element.text, element.content_desc) if p).strip()

  def rank(
      self,
      elements: Sequence[UiElement],
      task: str,
      probed_ids: Iterable[str] = (),
  ) -> list[RankedElement]:
    base = self._base.rank(elements, task, probed_ids)
    self.last_reranked = 0
    if not self._need or len(base) < 2:
      return base
    head, tail = base[:self._top_k], base[self._top_k:]
    texts = [self._text(r.element) for r in head]
    if not any(texts):
      return base

    from android_world.parallel_exploration import semantic_service
    # One round trip for the need and every known label together. A call per
    # label took a probe round to 6.9 s (2026-09-10): the encoder is dominated
    # by how many strings it sees, and the candidates repeat across queries, so
    # they have to be encoded once.
    visited = self._visited if self._visited_discount else []
    rows = semantic_service.query_many(
        self._port, [self._need] + self._known + visited, texts)
    if not rows:
      return base
    need_scores = rows[0]
    split = 1 + len(self._known)
    penalty = [0.0] * len(texts)
    for row in rows[1:split]:
      penalty = [max(p, g) for p, g in zip(penalty, row)]
    been = [0.0] * len(texts)
    for row in rows[split:]:
      been = [max(p, g) for p, g in zip(been, row)]

    scored = [
        (need_scores[i] - self._discount * penalty[i]
         - self._visited_discount * been[i], i, head[i].element)
        for i in range(len(head))
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    self.last_reranked = len(scored)
    self.last_scores = [
        (texts[i], need_scores[i], penalty[i], been[i], total)
        for total, i, _ in scored
    ]
    out = [RankedElement(element=e, rank=n, score=s)
           for n, (s, _, e) in enumerate(scored, start=1)]
    out.extend(RankedElement(element=r.element, rank=len(out) + i + 1,
                             score=r.score)
               for i, r in enumerate(tail))
    return out


def a11y_label(element: UiElement) -> str:
  """What this control is called, for a human or a small model reading a list.

  Text and content-description together, because a control often carries the
  value in one and the field name in the other ("Departure airport/city" as
  the description, "Hong Kong" as the text) and either alone is ambiguous.
  Falls back to the resource-id's last segment with separators opened up, so
  an icon-only control is still nameable instead of dropping out of the list.
  """
  parts = [p.strip() for p in (element.content_desc, element.text) if p and p.strip()]
  seen: list[str] = []
  for part in parts:
    if part.casefold() not in [s.casefold() for s in seen]:
      seen.append(part)
  if seen:
    return " ".join(seen)[:70]
  rid = (element.resource_id or "").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
  rid = re.sub(r"[_\-.]+", " ", rid).strip()
  return rid[:70]


class LlmChoiceRanker(Ranker):
  """Let the small model pick the control to probe, from the labels on screen.

  The graph rankers score candidates one at a time against a need string. This
  asks a different question, in one call: given the task, what is already
  settled, and every named control on this screen, which one would you try
  next? That is the question a person answers instantly and a cosine similarity
  cannot - "Search" is not lexically closer to "submit the search" than
  "Departure date" is.

  An earlier attempt to use this model for element ranking failed on positional
  bias, scoring 2/11 against uniform random's 10/11. Three things are different
  here and each is checkable: the prompt carries the task and the progress
  statement rather than a bare need phrase; the shots put the answer in four
  different positions; and the candidate order is shuffled per call, so a model
  that has learned "pick the first" is measurably no better than random rather
  than accidentally aligned with the base ranker.

  The chosen element is promoted to rank 0 and everything else keeps the base
  ordering, so a refusal (an empty reply, a label that is not on screen, an
  unreachable service) leaves the measured base ranker bit-for-bit unchanged.
  """

  def __init__(self, base: Ranker, need_text: str = "", port: int = 8766,
               max_labels: int = 20, seed: int = 0):
    self._base = base
    self._progress = (need_text or "").strip()
    self._port = port
    self._max_labels = max(2, max_labels)
    self._seed = seed
    # Instrumentation: "why was this element probed" was unanswerable before.
    self.last_choice = ""
    self.last_candidates: list[str] = []
    self.last_promoted_from = -1
    # The model's raw first line, kept only so a refusal is attributable: an
    # empty choice cannot distinguish a bad label set from a model that
    # answered with a sentence.
    self.last_raw = ""
    # One answer per screen, not per probe attempt. `_pick` calls rank() once
    # for every probe the round tries, with `already_probed` growing, and a
    # fresh 0.6-0.8 s model call each time ate the whole exploration window:
    # measured on the first task of the `llm` arm, rounds ran 2.9-4.0 s
    # against an `auto` budget of ~3.4 s and completed ZERO probes. The
    # screen has not changed between those calls, so neither has the answer.
    # One ranker instance is one exploration round on one screen.
    self._answer: tuple[str, str] | None = None

  def rank(
      self,
      elements: Sequence[UiElement],
      task: str,
      probed_ids: Iterable[str] = (),
  ) -> list[RankedElement]:
    base = self._base.rank(elements, task, probed_ids)
    # The last_* fields are deliberately NOT reset here. One instance is one
    # exploration round, and `_pick` calls rank() again for every candidate
    # the round rejects - the final call sees the most depleted set, often
    # with fewer than two nameable controls left. Resetting per call made the
    # round-level record read "the chooser was never asked" on 13 of 13 rounds
    # that had in fact asked (measured 2026-09-11, third `llm` launch).
    if len(base) < 2:
      return base
    head = base[:self._max_labels]
    labels = [a11y_label(r.element) for r in head]
    # Keep the first occurrence of each label: a duplicate cannot be resolved
    # back to one element, and offering it twice only invites the model to
    # pick a name that means two things.
    # Keyed on the whitespace-normalised label, because that is the form the
    # service echoes back: a label carrying a newline would otherwise be sent,
    # chosen, and then fail the lookup, reading as a refusal.
    by_label: dict[str, int] = {}
    for i, label in enumerate(labels):
      norm = " ".join(label.split())
      if norm and norm not in by_label:
        by_label[norm] = i
    if len(by_label) < 2:
      return base
    order = list(by_label)
    # Shuffled per call, seeded by the screen's own labels so a rerun of the
    # same step shuffles the same way and the arm stays reproducible.
    random.Random(f"{self._seed}\0{'|'.join(order)}").shuffle(order)
    if not self.last_candidates:
      self.last_candidates = order
    # Asked once per SCREEN, not once per candidate set. `_pick` calls rank()
    # again for every candidate the round rejects downstream, each time with a
    # smaller set - so a cache keyed on the current set never hits, and the
    # round pays another 0.5 s model call per retry. Measured on the second
    # `llm` launch: every round ran 2.7 s (against 369 ms for a candidate-less
    # round on `wide`) and completed ZERO probes. The screen has not changed
    # between those calls; only what is still allowed on it has.
    if self._answer is None:
      from android_world.parallel_exploration import semantic_service
      self._answer = semantic_service.query_choose(
          self._port, task, self._progress, order[:self._max_labels])
    chosen, raw = self._answer
    self.last_raw = raw
    if not chosen or chosen not in by_label:
      return base
    index = by_label[chosen]
    self.last_choice = chosen
    if self.last_promoted_from < 0:
      self.last_promoted_from = base[index].rank
    picked = base[index]
    rest = [r for i, r in enumerate(base) if i != index]
    return [RankedElement(picked.element, 0, picked.score)] + [
        RankedElement(r.element, i + 1, r.score) for i, r in enumerate(rest)]
