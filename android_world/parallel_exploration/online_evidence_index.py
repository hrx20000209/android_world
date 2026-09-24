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


class OnlineEvidenceIndex:
  def __init__(self):
    self.generation = 0
    self.records: dict[str, Observation] = {}
    self.by_source: dict[str, set[str]] = defaultdict(set)
    self.postings: dict[str, set[str]] = defaultdict(set)
    self.outcomes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    self.events: list[dict[str, Any]] = []

  def add(self, *, source: str, source_version: str, destination: str,
          action: dict, labels, recovered: bool, cost_s: float,
          origin: str = "probe") -> str:
    selector = action.get("control_key") or action.get("element_identity")
    action_key = str(selector or json.dumps(action, sort_keys=True))
    key = hashlib.sha256(f"{source}|{source_version}|{action_key}".encode()).hexdigest()[:24]
    old = self.records.get(key)
    if old:
      for token in terms(" ".join(old.labels) + " " + str(old.action)):
        self.postings[token].discard(key)
    obs = Observation(key, source, source_version, destination, dict(action),
                      tuple(str(x) for x in labels), self.generation,
                      recovered, cost_s, origin)
    self.records[key] = obs
    self.by_source[source].add(key)
    for token in terms(" ".join(obs.labels) + " " + str(action)):
      self.postings[token].add(key)
    self.outcomes[(source, action_key)][destination] += 1
    self.events.append({"event": "insert" if old is None else "replace",
                        "generation": self.generation, "key": key,
                        "source": source, "destination": destination,
                        "origin": origin, "recovered": recovered,
                        "nodes": len(self.by_source), "records": len(self.records)})
    return key

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
