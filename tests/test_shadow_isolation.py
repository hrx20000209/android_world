import threading
import time

from android_world.parallel_exploration.shadow_isolation import InferencePhase
from android_world.parallel_exploration.shadow_isolation import InstanceState
from android_world.parallel_exploration.shadow_isolation import PhaseAwareProbeScheduler
from android_world.parallel_exploration.shadow_isolation import ProbeStatus
from android_world.parallel_exploration.shadow_isolation import ProbeTransaction
from android_world.parallel_exploration.shadow_isolation import ResourceSnapshot
from android_world.parallel_exploration.shadow_isolation import ShadowIsolationController


def _state(prefix="prefix"):
  return InstanceState("activity", "ui", "evaluator", prefix)


def test_shadow_mutation_is_discarded_and_primary_remains_authoritative():
  primary = [_state()]
  shadow = [_state()]

  def rebuild(_prefix):
    shadow[0] = primary[0]

  controller = ShadowIsolationController(
      capture_primary=lambda: primary[0],
      capture_shadow=lambda: shadow[0],
      rebuild_shadow=rebuild,
  )

  def mutate_shadow(_cancel: threading.Event):
    shadow[0] = InstanceState("other", "mutated", "db-changed", "bad")
    return {"text": "14:30"}

  outcome = controller.run_probe(
      ProbeTransaction("p", "s", time.monotonic() + 10),
      authoritative_prefix=({"action": "open calendar"},),
      execute_shadow_probe=mutate_shadow,
      extract_evidence=lambda observation: {"event_time": observation["text"]},
  )

  assert outcome.status == ProbeStatus.ACCEPTED
  assert outcome.observed_fact == {"event_time": "14:30"}
  assert primary[0] == _state()
  assert shadow[0] == primary[0]


def test_pre_probe_mismatch_rejects_evidence_without_executing_probe():
  called = []
  controller = ShadowIsolationController(
      capture_primary=lambda: _state("primary"),
      capture_shadow=lambda: _state("shadow"),
      rebuild_shadow=lambda _prefix: None,
  )
  outcome = controller.run_probe(
      ProbeTransaction("p", "s", time.monotonic() + 10),
      authoritative_prefix=(),
      execute_shadow_probe=lambda _cancel: called.append(True),
      extract_evidence=lambda _: {},
  )

  assert outcome.status == ProbeStatus.REJECTED_STATE_MISMATCH
  assert not called
  assert outcome.observed_fact is None


def test_scheduler_blocks_prefill_pressure_and_deadline_tail():
  scheduler = PhaseAwareProbeScheduler()
  base = ResourceSnapshot(10.0, 5.0, 0.0, 0, InferencePhase.PREFILL)
  assert not scheduler.admit(base, inference_deadline_s=20, predicted_probe_s=1, now_s=10).admitted

  decode = ResourceSnapshot(10.0, 5.0, 0.0, 0, InferencePhase.DECODE)
  assert scheduler.admit(decode, inference_deadline_s=20, predicted_probe_s=1, now_s=10).admitted
  assert not scheduler.admit(decode, inference_deadline_s=11, predicted_probe_s=1, now_s=10).admitted

  pressure = ResourceSnapshot(10.0, 30.0, 0.0, 0, InferencePhase.DECODE)
  assert not scheduler.admit(pressure, inference_deadline_s=20, predicted_probe_s=1, now_s=10).admitted
