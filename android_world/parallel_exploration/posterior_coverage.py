"""Posterior-coverage exploration: probe what the full agent will most likely do.

At a screen s for task q the tap candidates are A = {a_1..a_n}. A cheap
posterior p_i = P(A_full = a_i | q, s) comes from the same agent model by
constrained candidate log-probability scoring: every candidate is rendered as
the agent's own canonical action string ("action:CLICK\\tpoint:x,y" at the
candidate's center, in the model's 0-1000 coordinates), appended to the exact
full-reasoning prompt as the start of the assistant turn, and scored with
vLLM prompt_logprobs. The shared prompt (screenshot included) is a cached
prefix, so after the first request each candidate costs a few tokens. The
scores are the log-probabilities of the coordinate tokens only - everything
before them is identical across candidates and cancels in the softmax.

Selection (all over the unverified safe candidates U):
  current            the existing MobileExplorer ranker (GraphKeywordRanker)
  random             uniform over U
  probability        descending p_i, no budget reasoning
  posterior_coverage max sum p_i s.t. sum c_i <= B, exact 0/1 knapsack,
                     recomputed after every probe with the remaining budget

Costs c_i and the budget B are online measurements (see CostModel and
ReasoningClock); nothing here is a hand-set weight. Safety filtering is the
existing hard filter and never enters a score.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import math
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests

from android_world.parallel_exploration import live_probe
from android_world.parallel_exploration.rankers import UiElement

METHODS = ("none", "current", "random", "probability", "posterior_coverage")
# Probe types the posterior can describe: a tap on an element's center.
# Scrolls, text entry and toggles are out of scope for this experiment.
TAP_TYPES = ("TAP_NAV", "TAP_MENU")
# The existing fusion code's definition of a full-page container: an element
# covering more than a quarter of the screen reacts to where it is tapped
# (month grids, canvases, maps), so "tap its center" is not one action.
CONTAINER_AREA_FRACTION = 0.25
# Knapsack time resolution. Probe costs are ~0.5-3 s, so 50 ms keeps the
# rounding error under a few percent of any single item.
KNAPSACK_RESOLUTION_S = 0.05


@dataclasses.dataclass(frozen=True)
class Candidate:
  key: str                      # element identity (resource|text|desc|class|bounds)
  element: UiElement
  probe_type: str
  center: tuple[int, int]       # device pixels
  model_point: tuple[int, int]  # the agent's 0-1000 coordinates
  label: str

  @property
  def action_string(self) -> str:
    return f"action:CLICK\tpoint:{self.model_point[0]},{self.model_point[1]}"


def abs_to_model_point(x: int, y: int, screen_size: tuple[int, int]) -> tuple[int, int]:
  """Inverse of gelab_agent._norm_to_abs."""
  width, height = (max(1, int(v)) for v in screen_size)
  return (int(round(x * 1000.0 / max(1, width - 1))),
          int(round(y * 1000.0 / max(1, height - 1))))


def tap_candidates(elements: Sequence[UiElement], filtered_path: Path,
                   context: Mapping[str, Any],
                   screen_size: tuple[int, int]) -> list[Candidate]:
  """Safe, tappable, non-container elements, one per distinct model point."""
  safe = live_probe._safe_candidates(tuple(elements), filtered_path, context)  # pylint: disable=protected-access
  width, height = screen_size
  screen_area = max(1, width * height)
  out: list[Candidate] = []
  seen_points: set[tuple[int, int]] = set()
  for element in safe:
    probe_type = live_probe._probe_type(element)  # pylint: disable=protected-access
    if probe_type not in TAP_TYPES:
      continue
    left, top, right, bottom = element.bounds
    area = max(0, right - left) * max(0, bottom - top)
    if area <= 0 or area > CONTAINER_AREA_FRACTION * screen_area:
      continue
    center = ((left + right) // 2, (top + bottom) // 2)
    point = abs_to_model_point(center[0], center[1], screen_size)
    if point in seen_points:
      continue
    seen_points.add(point)
    label = " ".join(p for p in (element.text, element.content_desc) if p) or (
        element.resource_id.rsplit("/", 1)[-1] or element.class_name.rsplit(".", 1)[-1])
    out.append(Candidate(element.identity, element, probe_type, center, point, label[:60]))
  return out


def final_action_key(action: Mapping[str, Any], elements: Sequence[UiElement],
                     candidates: Sequence[Candidate]) -> str:
  """Which candidate the full agent's action is, "" if it is not a tap on one.

  The smallest clickable element containing the tap is what was pressed; the
  action counts as candidate a_i only if that element is a_i. A tap resolved
  to a clickable element the safety filter removed is reported as such, so
  coverage is measured against the whole action space, not the probeable part.
  """
  if str(action.get("action_type") or "") not in {"click", "long_press"}:
    return f"non_tap:{action.get('action_type')}"
  x, y = action.get("x"), action.get("y")
  if x is None or y is None:
    return "non_tap:no_point"
  best, best_area = None, None
  for element in elements:
    if not element.clickable:
      continue
    left, top, right, bottom = element.bounds
    if not (left <= x <= right and top <= y <= bottom):
      continue
    area = max(0, right - left) * max(0, bottom - top)
    if area > 0 and (best_area is None or area < best_area):
      best, best_area = element, area
  if best is None:
    return "tap:no_clickable"
  keys = {c.key for c in candidates}
  return best.identity if best.identity in keys else "tap:not_candidate"


@dataclasses.dataclass
class Posterior:
  keys: list[str]
  logprobs: list[float]
  probs: list[float]
  latency_s: float
  first_request_s: float
  errors: int

  def as_dict(self) -> dict[str, float]:
    return dict(zip(self.keys, self.probs))


def score_candidates(url: str, model: str, messages: list[dict[str, Any]],
                     candidates: Sequence[Candidate],
                     max_workers: int = 8, timeout_s: float = 60.0) -> Posterior:
  """Constrained candidate log-probabilities from the agent model itself.

  The first request goes alone so the screenshot prefix is computed once and
  cached; the rest then run concurrently against the cached prefix.
  """
  started = time.monotonic()
  if not candidates:
    return Posterior([], [], [], 0.0, 0.0, 0)

  def one(candidate: Candidate) -> float | None:
    body = {
        "model": model,
        "messages": list(messages) + [{"role": "assistant",
                                       "content": candidate.action_string}],
        "continue_final_message": True, "add_generation_prompt": False,
        "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0,
    }
    try:
      reply = requests.post(url, json=body, timeout=timeout_s).json()
      rows = reply.get("prompt_logprobs") or []
    except Exception:  # pylint: disable=broad-exception-caught
      return None
    # Walk back over the prompt tokens until the coordinate digits
    # "x,y" are consumed; the text before them is common to all candidates.
    suffix = f"{candidate.model_point[0]},{candidate.model_point[1]}"
    total, consumed = 0.0, ""
    for row in reversed(rows):
      if not row:
        return None
      info = next(iter(row.values()))
      token = str(info.get("decoded_token") or "")
      total += float(info.get("logprob") or 0.0)
      consumed = token + consumed
      if len(consumed) >= len(suffix):
        return total if consumed.endswith(suffix) else None
    return None

  first_started = time.monotonic()
  results: list[float | None] = [one(candidates[0])]
  first_s = time.monotonic() - first_started
  if len(candidates) > 1:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
      results.extend(pool.map(one, candidates[1:]))
  keys, logprobs = [], []
  errors = 0
  for candidate, value in zip(candidates, results):
    if value is None:
      errors += 1
      continue
    keys.append(candidate.key)
    logprobs.append(value)
  probs = softmax(logprobs)
  return Posterior(keys, logprobs, probs, time.monotonic() - started, first_s, errors)


def softmax(values: Sequence[float]) -> list[float]:
  if not values:
    return []
  top = max(values)
  weights = [math.exp(v - top) for v in values]
  total = sum(weights)
  return [w / total for w in weights]


def knapsack(items: Sequence[tuple[str, float, float]], budget_s: float,
             resolution_s: float = KNAPSACK_RESOLUTION_S) -> list[str]:
  """Exact 0/1 knapsack: keys maximizing sum p with sum cost <= budget.

  Costs are rounded UP to the resolution, so a chosen set never exceeds the
  real budget because of discretization.
  """
  capacity = int(math.floor(max(0.0, budget_s) / resolution_s + 1e-9))
  if capacity <= 0 or not items:
    return []
  weights = [max(1, int(math.ceil(cost / resolution_s - 1e-9))) for _, _, cost in items]
  best = [0.0] * (capacity + 1)
  keep = [[False] * (capacity + 1) for _ in items]
  for index, (_, value, _) in enumerate(items):
    weight = weights[index]
    for cap in range(capacity, weight - 1, -1):
      candidate = best[cap - weight] + value
      if candidate > best[cap] + 1e-15:
        best[cap] = candidate
        keep[index][cap] = True
  chosen, cap = [], capacity
  for index in range(len(items) - 1, -1, -1):
    if keep[index][cap]:
      chosen.append(items[index][0])
      cap -= weights[index]
  return chosen[::-1]


class CostModel:
  """Online probe cost c_i: forward transition + observation + recovery.

  Hierarchy, most specific measured first: this candidate earlier in this
  episode, this app and probe type in this method's session, this probe type,
  then the prior measured before the evaluation. Only latency crosses task
  boundaries - never which control leads where.
  """

  def __init__(self, session: dict[str, list[float]], prior_s: float):
    self.session = session
    self.prior_s = prior_s
    self.episode: dict[str, list[float]] = {}

  def estimate(self, candidate: Candidate, package: str) -> tuple[float, str]:
    for key, source in ((candidate.key, "episode_candidate"),):
      if self.episode.get(key):
        return statistics.median(self.episode[key]), source
    for key, source in ((f"{package}|{candidate.probe_type}", "session_app_type"),
                        (candidate.probe_type, "session_type")):
      values = self.session.get(key) or []
      if values:
        return statistics.median(values[-50:]), source
    return self.prior_s, "prior"

  def record(self, candidate: Candidate, package: str, cost_s: float) -> None:
    self.episode.setdefault(candidate.key, []).append(cost_s)
    for key in (f"{package}|{candidate.probe_type}", candidate.probe_type):
      self.session.setdefault(key, []).append(cost_s)


class ReasoningClock:
  """B: time left before the in-flight full reasoning call returns.

  Predicted duration = median of this method's last 20 measured reasoning
  calls (prior before any are measured), minus the time already elapsed.
  """

  def __init__(self, history: list[float], prior_s: float, started: float):
    self.predicted_s = statistics.median(history[-20:]) if history else prior_s
    self.source = "session_median" if history else "prior"
    self.started = started

  def remaining(self) -> float:
    return self.predicted_s - (time.monotonic() - self.started)


@dataclasses.dataclass
class ProbeRecord:
  key: str
  label: str
  p: float
  predicted_cost_s: float
  cost_source: str
  budget_before_s: float
  forward_s: float = 0.0
  settle_s: float = 0.0
  rollback_s: float = 0.0
  rollback_ok: bool | None = None
  rollback_level: str = ""
  successor_state: str = ""
  successor_activity: str = ""
  moved: bool = False
  started_after_reasoning: bool = False
  finished_after_reasoning: bool = False

  @property
  def actual_cost_s(self) -> float:
    return self.forward_s + self.settle_s + self.rollback_s


@dataclasses.dataclass
class WindowResult:
  probes: list[ProbeRecord] = dataclasses.field(default_factory=list)
  committed: bool = False
  parked_key: str = ""
  dirty: bool = False
  exposed_s: float = 0.0        # reasoning return -> device ready (or committed)
  final_rollback_s: float = 0.0
  stop_reason: str = ""
  error: str = ""


def state_id(state) -> str:
  return f"{state.activity.component}#{state.layout_sig}"


class ProbeWindow(threading.Thread):
  """Explores while the full reasoning call is in flight, then commits or recovers.

  The window never learns the final action until reasoning has returned; it
  is told through finish(final_key) and uses it only for the commit/recover
  decision, never to choose a probe.
  """

  def __init__(self, *, method: str, adb, capture, baseline, candidates: Sequence[Candidate],
               probs: Mapping[str, float], verified: set[str], cost_model: CostModel,
               clock: ReasoningClock, package: str, current_order: Sequence[str],
               rng: random.Random, max_probes: int):
    super().__init__(daemon=True)
    self.method = method
    self.adb, self.capture, self.baseline = adb, capture, baseline
    self.candidates = {c.key: c for c in candidates}
    self.probs = dict(probs)
    self.verified = verified
    self.cost_model = cost_model
    self.clock = clock
    self.package = package
    self.current_order = list(current_order)
    self.rng = rng
    self.max_probes = max_probes
    self.result = WindowResult()
    self.plans: list[dict[str, Any]] = []
    self._done = threading.Event()
    self._final_key: list[str] = []
    self._at_baseline = True
    self._parked: Candidate | None = None
    self._last_post = None
    self.new_edges: list[tuple[str, str, str]] = []  # (key, successor_state, activity)

  # --- selection -------------------------------------------------------
  def _unverified(self, tried: set[str]) -> list[Candidate]:
    return [c for key, c in self.candidates.items()
            if key not in tried and key not in self.verified]

  def _next(self, tried: set[str]) -> Candidate | None:
    pool = self._unverified(tried)
    if not pool:
      return None
    if self.method == "random":
      return self.rng.choice(pool)
    if self.method == "current":
      pool_keys = {c.key for c in pool}
      for key in self.current_order:
        if key in pool_keys:
          return self.candidates[key]
      return None
    if self.method == "probability":
      return max(pool, key=lambda c: (self.probs.get(c.key, 0.0), c.key))
    if self.method == "posterior_coverage":
      budget = self.clock.remaining()
      if self._parked is not None:
        # The parked probe's rollback has not been paid yet; it has to be
        # before the next probe can start, so it comes out of the budget.
        record = self.result.probes[-1]
        estimate, _ = self.cost_model.estimate(self._parked, self.package)
        budget -= max(0.0, estimate - record.forward_s - record.settle_s)
      items = []
      for c in pool:
        cost, _ = self.cost_model.estimate(c, self.package)
        items.append((c.key, self.probs.get(c.key, 0.0), cost))
      chosen = knapsack(items, budget)
      self.plans.append({"t_remaining_s": round(budget, 3), "pool": len(pool),
                         "chosen": chosen,
                         "chosen_mass": round(sum(self.probs.get(k, 0.0) for k in chosen), 4)})
      if not chosen:
        return None
      return max((self.candidates[k] for k in chosen),
                 key=lambda c: (self.probs.get(c.key, 0.0), c.key))
    raise ValueError(self.method)

  # --- device ----------------------------------------------------------
  def _recover(self, candidate: Candidate) -> tuple[bool, str, float]:
    level, ok, _, _, recovery_ms, _ = live_probe._recover(  # pylint: disable=protected-access
        self.adb, self.capture, self.baseline, self._last_post, candidate.probe_type,
        candidate.element, threading.Event(), None)
    return ok, level, recovery_ms / 1000.0

  def run(self) -> None:
    tried: set[str] = set()
    try:
      while not self._done.is_set() and len(self.result.probes) < self.max_probes:
        candidate = self._next(tried)
        if candidate is None:
          self.result.stop_reason = "selector_stopped"
          break
        if self._parked is not None:
          if self._done.is_set():
            # Reasoning returned while the next probe was being chosen: stay
            # parked, the parked action may be the one to commit.
            break
          record = self.result.probes[-1]
          ok, level, seconds = self._recover(self._parked)
          record.rollback_s, record.rollback_ok, record.rollback_level = seconds, ok, level
          self.cost_model.record(self._parked, self.package, record.actual_cost_s)
          self._parked = None
          self._at_baseline = ok
          if not ok:
            self.result.dirty = True
            self.result.stop_reason = "recovery_failed"
            break
          if self._done.is_set():
            break
        tried.add(candidate.key)
        cost, source = self.cost_model.estimate(candidate, self.package)
        record = ProbeRecord(candidate.key, candidate.label,
                             self.probs.get(candidate.key, 0.0), cost, source,
                             self.clock.remaining(),
                             started_after_reasoning=self._done.is_set())
        self.result.probes.append(record)
        started = time.monotonic()
        live_probe._adb_action(self.adb, candidate.probe_type, candidate.element)  # pylint: disable=protected-access
        record.forward_s = time.monotonic() - started
        time.sleep(0.10)
        settle_started = time.monotonic()
        post = self.capture.capture()
        record.settle_s = time.monotonic() - settle_started + 0.10
        record.finished_after_reasoning = self._done.is_set()
        record.successor_state = state_id(post)
        record.successor_activity = post.activity.component
        record.moved = record.successor_state != state_id(self.baseline)
        self._last_post = post
        self._parked = candidate
        self._at_baseline = False
        self.new_edges.append((candidate.key, record.successor_state,
                               record.successor_activity))
      else:
        if not self.result.stop_reason:
          self.result.stop_reason = ("reasoning_finished" if self._done.is_set()
                                     else "max_probes")
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self.result.error = f"{type(exc).__name__}: {exc}"
      self.result.stop_reason = "error"
    # Park until the full agent answers: the device stays wherever the last
    # probe left it, which is what makes a commit possible.
    self._done.wait()
    decided = time.monotonic()
    final_key = self._final_key[0] if self._final_key else ""
    if self._parked is not None:
      if final_key == self._parked.key and self._parked_still_there():
        # No rollback happened, so this probe's full cost c_i (which
        # includes recovery) was not observed and is not recorded.
        self.result.committed = True
        self.result.parked_key = self._parked.key
      else:
        record = self.result.probes[-1]
        ok, level, seconds = self._recover(self._parked)
        record.rollback_s, record.rollback_ok, record.rollback_level = seconds, ok, level
        self.cost_model.record(self._parked, self.package, record.actual_cost_s)
        self.result.final_rollback_s = seconds
        self.result.dirty = self.result.dirty or not ok
    self.result.exposed_s = time.monotonic() - decided

  def _parked_still_there(self) -> bool:
    try:
      now = self.capture.capture()
    except Exception:  # pylint: disable=broad-exception-caught
      return False
    return (now.activity.component == self._last_post.activity.component
            and now.layout_sig == self._last_post.layout_sig)

  def finish(self, final_key: str) -> None:
    """Called once reasoning has returned; the only place the final action enters."""
    self._final_key.append(final_key)
    self._done.set()
