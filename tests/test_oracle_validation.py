import dataclasses

import pytest

from android_world.parallel_exploration.oracle_validation import ExperimentArm
from android_world.parallel_exploration.oracle_validation import ExplorationOpportunity
from android_world.parallel_exploration.oracle_validation import FreshEvidence
from android_world.parallel_exploration.oracle_validation import OracleTrialResult
from android_world.parallel_exploration.oracle_validation import summarize_results


def _paired_case(family: str, seed: int):
  opportunity = ExplorationOpportunity(
      task_id=f"{family}-task",
      app_family=family,
      episode_seed=seed,
      decision_id="decision-1",
      state_id="state-1",
      task_need="current value",
      freshness_requirement_s=30,
      candidate_actions=("open details",),
  )
  rows = []
  for arm in ExperimentArm:
    online = arm == ExperimentArm.ONLINE_ORACLE
    stale = arm == ExperimentArm.ONLINE_STALE
    evidence = None
    if online or stale:
      evidence = FreshEvidence(
          source_action="open details",
          observed_fact="value=now" if online else "value=old",
          state_binding="state-1",
          observed_at_s=1.0,
          confidence=1.0,
          is_fresh=online,
          isolation_id="shadow",
      )
    rows.append(OracleTrialResult(
        trial_id=f"{family}-{seed}-{arm.value}",
        arm=arm,
        opportunity=opportunity,
        memory_seeds=(seed + 100,),
        slot_correct=online,
        next_inference_avoided=online,
        remaining_reasoning_calls=0 if online else 1,
        remaining_actions=0 if online else 1,
        stale_evidence_error=stale,
        evidence_eligible=True,
        evidence=evidence,
    ))
  return rows


def test_go_when_fresh_oracle_saves_work_in_two_families():
  rows = _paired_case("calendar", 1) + _paired_case("tasks", 2)

  summary = summarize_results(rows, bootstrap_iterations=100)

  assert summary["go_no_go"]["decision"] == "CONTINUE_ONLINE"
  assert summary["go_no_go"]["passing_app_families"] == 2


def test_current_seed_leak_is_rejected():
  row = _paired_case("calendar", 1)[0]
  leaked = dataclasses.replace(row, memory_seeds=(1,))

  with pytest.raises(ValueError, match="leaks"):
    leaked.validate()
