import itertools
import random
import threading
import time
import types

from android_world.parallel_exploration import posterior_coverage as pc
from android_world.parallel_exploration.rankers import UiElement


def test_knapsack_is_exact():
  rng = random.Random(0)
  for _ in range(200):
    items = [(f"k{i}", rng.random(), rng.uniform(0.2, 3.0)) for i in range(rng.randint(1, 7))]
    budget = rng.uniform(0, 6)
    chosen = pc.knapsack(items, budget)
    best = 0.0
    for r in range(len(items) + 1):
      for combo in itertools.combinations(items, r):
        # Brute force under the same round-up discretization.
        if sum(pc.math.ceil(c / pc.KNAPSACK_RESOLUTION_S - 1e-9) for _, _, c in combo) \
            <= int(budget / pc.KNAPSACK_RESOLUTION_S + 1e-9):
          best = max(best, sum(p for _, p, _ in combo))
    by_key = {k: (p, c) for k, p, c in items}
    assert abs(sum(by_key[k][0] for k in chosen) - best) < 1e-9
    assert sum(by_key[k][1] for k in chosen) <= budget + 1e-9


def test_knapsack_prefers_mass_over_count_and_respects_budget():
  items = [("a", 0.6, 2.0), ("b", 0.25, 1.0), ("c", 0.2, 1.0)]
  assert set(pc.knapsack(items, 2.0)) == {"a"}
  assert set(pc.knapsack(items, 3.0)) == {"a", "b"}
  assert pc.knapsack(items, 0.5) == []


def test_softmax_sums_to_one():
  probs = pc.softmax([-3.0, -1.0, -10.0])
  assert abs(sum(probs) - 1.0) < 1e-12 and probs[1] > probs[0] > probs[2]


def test_model_point_round_trips_through_gelab_conversion():
  from android_world.agents.gelab_agent import _norm_to_abs
  for x, y in [(0, 0), (540, 1200), (1079, 2399), (963, 2071)]:
    point = pc.abs_to_model_point(x, y, (1080, 2400))
    back = _norm_to_abs(list(point), (1080, 2400))
    assert abs(back[0] - x) <= 2 and abs(back[1] - y) <= 3


def _el(text, bounds, clickable=True, cls="android.widget.Button"):
  return UiElement(text=text, class_name=cls, bounds=bounds, clickable=clickable)


def _cand(element):
  center = ((element.bounds[0] + element.bounds[2]) // 2,
            (element.bounds[1] + element.bounds[3]) // 2)
  return pc.Candidate(element.identity, element, "TAP_NAV", center,
                      pc.abs_to_model_point(*center, (1080, 2400)), element.text)


def test_final_action_maps_to_smallest_clickable_candidate():
  row = _el("row", (0, 100, 1080, 300))
  button = _el("OK", (800, 150, 1000, 250))
  cands = [_cand(row), _cand(button)]
  elements = [row, button]
  assert pc.final_action_key({"action_type": "click", "x": 900, "y": 200}, elements, cands) == button.identity
  assert pc.final_action_key({"action_type": "click", "x": 100, "y": 200}, elements, cands) == row.identity
  assert pc.final_action_key({"action_type": "input_text", "text": "a"}, elements, cands).startswith("non_tap")
  hidden = _el("Delete", (0, 500, 200, 600))
  assert pc.final_action_key({"action_type": "click", "x": 50, "y": 550},
                             elements + [hidden], cands) == "tap:not_candidate"


class _State:
  def __init__(self, name):
    self.activity = types.SimpleNamespace(component=f"app/.{name}")
    self.layout_sig = name
    self.elements = ()


class _FakeDevice:
  """Screen = the last tapped candidate's name, or 'home'."""

  def __init__(self, names_by_point, tap_s=0.05):
    self.screen = "home"
    self.names = names_by_point
    self.taps = []
    self.tap_s = tap_s

  def run(self, argv, timeout_s=2.0):
    if argv[:3] == ["shell", "input", "tap"]:
      time.sleep(self.tap_s)
      point = (int(argv[3]), int(argv[4]))
      self.taps.append(point)
      self.screen = self.names[point]
    return types.SimpleNamespace(stdout="", returncode=0)

  def capture(self):
    return _State(self.screen)

  def close(self):
    pass


def _window(method, probs, remaining_s, verified=(), tap_s=0.05):
  elements = [_el(n, (100 * i, 100, 100 * i + 80, 180)) for i, n in enumerate("abcd", start=1)]
  cands = [_cand(e) for e in elements]
  device = _FakeDevice({c.center: c.label for c in cands}, tap_s)
  clock = pc.ReasoningClock([], remaining_s, time.monotonic())
  cost = pc.CostModel({}, prior_s=0.5)
  window = pc.ProbeWindow(
      method=method, adb=device, capture=device, baseline=_State("home"),
      candidates=cands, probs={c.key: probs[c.label] for c in cands},
      verified={c.key for c in cands if c.label in verified}, cost_model=cost,
      clock=clock, package="app", current_order=[c.key for c in cands],
      rng=random.Random(0), max_probes=10)
  recoveries = []

  def fake_recover(candidate):
    recoveries.append(candidate.label)
    device.screen = "home"
    return True, "INVERSE", 0.05
  window._recover = fake_recover
  return window, cands, device, recoveries


def test_commit_when_parked_on_the_final_action_and_no_rollback():
  window, cands, device, recoveries = _window(
      "posterior_coverage", {"a": 0.05, "b": 0.8, "c": 0.1, "d": 0.05}, remaining_s=0.8)
  window.start()
  time.sleep(0.5)
  final = next(c for c in cands if c.label == "b")
  window.finish(final.key)
  window.join(5)
  res = window.result
  assert [p.label for p in res.probes] == ["b"]      # budget fits one probe: the top one
  assert res.committed and res.parked_key == final.key
  assert recoveries == []                             # never rolled back
  assert device.screen == "b" and len(device.taps) == 1  # action executed exactly once


def test_mismatch_recovers_to_baseline():
  window, cands, device, recoveries = _window(
      "posterior_coverage", {"a": 0.05, "b": 0.8, "c": 0.1, "d": 0.05}, remaining_s=0.8)
  window.start()
  time.sleep(0.5)
  other = next(c for c in cands if c.label == "c")
  window.finish(other.key)
  window.join(5)
  assert not window.result.committed
  assert recoveries == ["b"] and device.screen == "home"


def test_coverage_only_starts_probes_that_fit_probability_ignores_budget():
  probs = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
  cov, _, _, _ = _window("posterior_coverage", probs, remaining_s=1.2, tap_s=0.3)
  prob, _, _, _ = _window("probability", probs, remaining_s=1.2, tap_s=0.3)
  for w in (cov, prob):
    w.start()
  time.sleep(2.5)
  for w in (cov, prob):
    w.finish("none")
    w.join(5)
  assert cov.result.probes and prob.result.probes
  for p in cov.result.probes:
    assert p.budget_before_s >= p.predicted_cost_s - pc.KNAPSACK_RESOLUTION_S
  # Probability keeps going after the predicted window has closed.
  assert any(p.budget_before_s < p.predicted_cost_s for p in prob.result.probes)
  assert len(prob.result.probes) > len(cov.result.probes)
  # Both visit candidates in descending p.
  labels = [p.label for p in prob.result.probes]
  assert labels == sorted(labels, key=lambda l: -probs[l])


def test_verified_actions_are_not_probed_again():
  window, _, _, _ = _window("probability", {"a": 0.7, "b": 0.2, "c": 0.05, "d": 0.05},
                            remaining_s=5, verified=("a",))
  window.start()
  time.sleep(0.3)
  window.finish("none")
  window.join(5)
  assert "a" not in [p.label for p in window.result.probes]
