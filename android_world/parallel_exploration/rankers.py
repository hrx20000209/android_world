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
