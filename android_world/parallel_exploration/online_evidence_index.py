"""Incremental, generation-versioned evidence retrieval for live GUI probes.

Transition outcomes, recovery outcomes and authoritative choices are distinct
observations. A symmetric Dirichlet posterior measures transition uncertainty;
it does not certify task correctness. Retrieval is a budgeted weighted coverage
problem over query terms, with exact-source binding and observed recovery gates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any


def terms(text: str) -> set[str]:
  return set(re.findall(r"[\w]+", text.casefold()))


@dataclass
class Observation:
  key: str
  source: str
  source_version: str
  destination: str
  action: dict[str, Any]
  labels: tuple[str, ...]
  generation: int
  recovered: bool
  cost_s: float
  origin: str = "probe"
  destination_family: str | None = None


def _action_key(action: dict[str, Any]) -> str:
  """Identify an action slot without confusing moved controls or gestures."""
  action_type = str(action.get("action_type", "")).casefold()
  selector = (
      action.get("stable_control_key")
      or action.get("control_key")
      or action.get("element_identity")
  )
  if selector:
    payload = {
        "action_type": action_type,
        "selector": str(selector),
        "direction": action.get("direction"),
        "text": action.get("text") if action_type in {"input_text", "type"} else None,
        "app_name": action.get("app_name"),
    }
  else:
    payload = dict(action)
    payload["action_type"] = action_type
  return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class OnlineEvidenceIndex:
  def __init__(self):
    self.generation = 0
    self.records: dict[str, Observation] = {}
    self.by_source: dict[str, set[str]] = defaultdict(set)
    self.postings: dict[str, set[str]] = defaultdict(set)
    self.outcomes: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    self.destination_families: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    self.events: list[dict[str, Any]] = []

  def add(self, *, source: str, source_version: str, destination: str,
          action: dict, labels, recovered: bool, cost_s: float,
          origin: str = "probe", destination_family: str | None = None) -> str:
    action_key = _action_key(action)
    # Destination is part of the observation identity. A control can have
    # several valid successors as app data, permissions, or system state vary.
    # Keeping each branch is necessary for later verification.
    key = hashlib.sha256(
        f"{source}|{source_version}|{action_key}|{destination}".encode()
    ).hexdigest()[:24]
    old = self.records.get(key)
    if old and old.recovered and not recovered and origin != "authoritative":
      # A later failed attempt must not revoke an earlier verified observation
      # of the same branch. Keep the failure in the event log, not in retrieval.
      self.events.append({
          "event": "rejected_duplicate",
          "generation": self.generation,
          "key": key,
          "source": source,
          "destination": destination,
          "destination_family": destination_family,
          "origin": origin,
          "recovered": False,
          "nodes": len(self.by_source),
          "records": len(self.records),
      })
      return key
    if old:
      for token in terms(" ".join(old.labels) + " " + str(old.action)):
        self.postings[token].discard(key)
    obs = Observation(key, source, source_version, destination, dict(action),
                      tuple(str(x) for x in labels), self.generation,
                      recovered, cost_s, origin, destination_family)
    self.records[key] = obs
    self.by_source[source].add(key)
    for token in terms(" ".join(obs.labels) + " " + str(action)):
      self.postings[token].add(key)
    # A failed speculative recovery is retained in the audit log but cannot
    # become a trusted transition. Authoritative executions are already on
    # the primary instance and therefore do not need speculative recovery.
    if recovered or origin == "authoritative":
      self.outcomes[(source, source_version, action_key)][destination] += 1
      if destination_family:
        self.destination_families[(source, source_version, action_key)][
            destination_family
        ] += 1
    self.events.append({"event": "insert" if old is None else "replace",
                        "generation": self.generation, "key": key,
                        "source": source, "destination": destination,
                        "destination_family": destination_family,
                        "origin": origin, "recovered": recovered,
                        "nodes": len(self.by_source), "records": len(self.records)})
    return key

  def classify_transition(self, *, source: str, source_version: str,
                          action: dict[str, Any], destination: str | None,
                          destination_family: str | None = None,
                          source_match: bool = True,
                          action_match: bool = True) -> dict[str, Any]:
    """Classify an authoritative successor against verified edge outcomes.

    ``source_match`` and ``action_match`` are supplied by the caller because
    state-family and selector matching are environment-specific. This method
    only compares exact destination fingerprints after those gates pass.
    """
    action_key = _action_key(action)
    known = self.outcomes.get((source, source_version, action_key), Counter()).copy()
    known_families = self.destination_families.get(
        (source, source_version, action_key), Counter()).copy()
    same_family_prior_destinations = []
    if destination_family:
      for prior_destination in known:
        prior_key = hashlib.sha256(
            f"{source}|{source_version}|{action_key}|{prior_destination}".encode()
        ).hexdigest()[:24]
        prior_observation = self.records.get(prior_key)
        if prior_observation and prior_observation.destination_family == destination_family:
          same_family_prior_destinations.append(prior_destination)
    if not source_match or not action_match or not destination or not known:
      verdict = "UNMATCHABLE"
    elif destination not in known:
      verdict = "NEW_BRANCH"
    else:
      max_support = max(known.values(), default=0)
      verdict = (
          "KNOWN_OUTCOME" if known[destination] == max_support
          else "KNOWN_ALTERNATIVE"
      )
    return {
        "verdict": verdict,
        "source": source,
        "source_version": source_version,
        "action_key": action_key,
        "destination": destination,
        "destination_family": destination_family,
        "known_destinations": dict(known),
        "same_family_prior_destinations": sorted(same_family_prior_destinations),
        "known_destination_families": dict(known_families),
        "reason": (
            (
                "source_or_action_not_matched" if not source_match or not action_match
                else ("missing_destination" if not destination else "no_verified_prior_edge")
            ) if verdict == "UNMATCHABLE"
            else ("no_prior_destination_match" if verdict == "NEW_BRANCH" else "")
        ),
    }

  def commit(self):
    self.generation += 1

  def retrieve(self, source: str, source_version: str, query: str,
               budget_words: int = 160, max_records: int = 4) -> list[Observation]:
    query_terms = terms(query)
    matched = set().union(*(self.postings.get(t, set()) for t in query_terms)) if query_terms else set()
    candidates = [self.records[k] for k in sorted(matched & self.by_source.get(source, set()))
                  if self.records[k].source_version == source_version
                  and self.records[k].recovered
                  and self.records[k].generation < self.generation]
    covered: set[str] = set()
    result = []
    used = 0
    while candidates and len(result) < max_records:
      scored = []
      for obs in candidates:
        tokens = terms(" ".join(obs.labels) + " " + str(obs.action))
        newly_covered = (tokens & query_terms) - covered
        gain = sum(math.log1p(len(self.records) / max(1, len(self.postings[t])))
                   for t in newly_covered)
        size = max(1, len((" ".join(obs.labels) + " " + str(obs.action)).split()))
        if used + size <= budget_words and gain > 0:
          scored.append((gain / size, obs.key, size, obs, tokens))
      if not scored:
        break
      _, _, size, chosen, tokens = max(scored, key=lambda x: (x[0], x[1]))
      result.append(chosen)
      used += size
      covered |= tokens & query_terms
      candidates.remove(chosen)
    self.events.append({"event": "retrieve", "generation": self.generation,
                        "source": source, "eligible": len(matched & self.by_source.get(source, set())),
                        "selected": [x.key for x in result], "words": used})
    return result


def posterior_frontier_scores(elements, task: str, graph_payload: dict,
                              current_node: str) -> dict[str, float]:
  """Expected transition entropy after Dirichlet smoothing, per measured second.

  Relevance is a normalized query likelihood from current labels, not a claim
  of calibrated decision utility. Unknown destinations are an explicit category.
  No task success label or evaluator state is used.
  """
  edges = graph_payload.get("edges", [])
  if isinstance(edges, dict):
    edges = list(edges.values())
  local = defaultdict(list)
  for edge in edges:
    if edge.get("src_node") == current_node and edge.get("status") != "INVALID":
      a = edge.get("action", {})
      identity = a.get("element_identity", "")
      local[identity].append(edge)
  query = terms(task)
  likelihoods = [1 + len(query & terms(e.text + " " + e.content_desc)) for e in elements]
  denominator = sum(likelihoods) or 1
  costs = [float(e.get("exploration_cost", 0)) for e in edges if float(e.get("exploration_cost", 0)) > 0]
  default_cost = sum(costs) / len(costs) if costs else 1.0
  scores = {}
  for element, likelihood in zip(elements, likelihoods):
    records = local[element.identity]
    counts = Counter()
    for edge in records:
      counts.update(edge.get("observed_destinations", []) or [edge.get("dst_node", "unknown")])
    # One unobserved-successor category remains even for a deterministic edge.
    alpha = [v + 0.5 for v in counts.values()] + [0.5]
    if not counts:
      alpha.append(0.5)
    total = sum(alpha)
    entropy = -sum((a / total) * math.log(a / total) for a in alpha)
    successes = sum(int(x.get("rollback_success_count", 0)) for x in records)
    failures = sum(int(x.get("rollback_failure_count", 0)) for x in records)
    recovery_p = (successes + 1) / (successes + failures + 2)
    cost_samples = [float(x.get("exploration_cost", 0)) for x in records if float(x.get("exploration_cost", 0)) > 0]
    cost = sum(cost_samples) / len(cost_samples) if cost_samples else default_cost
    scores[element.identity] = likelihood / denominator * entropy * recovery_p / max(cost, 1e-6)
  return scores
