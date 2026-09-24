import dataclasses

from android_world.parallel_exploration.agent_evidence_graph import AgentEvidenceGraph
from android_world.parallel_exploration.agent_evidence_graph import DecisionContext
from android_world.parallel_exploration.agent_evidence_graph import EvidenceRecord
from android_world.parallel_exploration.agent_evidence_graph import EvidenceScope
from android_world.parallel_exploration.agent_evidence_graph import ProbeCandidate
from android_world.parallel_exploration.agent_evidence_graph import ReasoningMode


def _context(now_s: float = 100.0) -> DecisionContext:
  return DecisionContext(
      episode_id="episode-current",
      state_id="calendar-list",
      progress_signature="opened-calendar",
      task_need="find next event time",
      required_slots=("event_time",),
      generation=2,
      now_s=now_s,
  )


def _graph() -> AgentEvidenceGraph:
  return AgentEvidenceGraph(agent_id="agent-a", policy_fingerprint="model+prompt-v1")


def _evidence(**overrides) -> EvidenceRecord:
  values = dict(
      evidence_id="ev-1",
      slot="event_time",
      value="14:30",
      source_action="open event details",
      source_state_id="calendar-list",
      progress_signature="opened-calendar",
      task_need="find next event time",
      observed_at_s=99.0,
      ttl_s=30.0,
      confidence=0.97,
      scope=EvidenceScope.CURRENT_EPISODE,
      episode_id="episode-current",
      isolation_id="shadow-1",
      state_match=True,
      extraction_verified=True,
      observed_generation=1,
  )
  values.update(overrides)
  return EvidenceRecord(**values)


def test_fresh_state_bound_fact_can_certify_low_risk_skip():
  graph = _graph()
  graph.add_evidence(_evidence())

  plan = graph.plan_reasoning(
      _context(),
      action_by_slot_value={("event_time", "14:30"): "answer(14:30)"},
      consequence_safe_to_skip=True,
  )

  assert plan.mode == ReasoningMode.SKIP
  assert plan.deterministic_action == "answer(14:30)"
  assert "Do not repeat the probe action" in plan.prompt_block


def test_stale_or_cross_episode_value_is_not_injected():
  graph = _graph()
  graph.add_evidence(_evidence(observed_at_s=60.0))
  graph.add_evidence(_evidence(evidence_id="other", episode_id="other"))

  plan = graph.plan_reasoning(_context())

  assert plan.mode == ReasoningMode.NORMAL
  assert not plan.prompt_block


def test_same_generation_evidence_cannot_change_inflight_reasoning():
  graph = _graph()
  graph.add_evidence(_evidence(observed_generation=2))

  assert graph.plan_reasoning(_context()).mode == ReasoningMode.NORMAL


def test_exploration_summary_filters_by_need_safety_deadline_and_voi():
  graph = _graph()
  useful = ProbeCandidate(
      action="open details",
      target_slots=("event_time",),
      reveal_probability=1.0,
      future_calls_avoided=1.0,
      future_actions_avoided=1.0,
      probe_latency_s=0.2,
      extraction_latency_s=0.05,
      resync_latency_s=0.05,
      interference_penalty_s=0.05,
  )
  irrelevant = ProbeCandidate(
      action="open settings",
      target_slots=("theme",),
      reveal_probability=1.0,
      future_calls_avoided=3.0,
      future_actions_avoided=3.0,
      probe_latency_s=0.1,
      extraction_latency_s=0.1,
      resync_latency_s=0.1,
      interference_penalty_s=0.0,
  )
  unsafe = dataclasses.replace(useful, action="delete event", destructive=True)
  too_slow = dataclasses.replace(useful, action="slow", probe_latency_s=2.0)

  summary = graph.summarize_for_exploration(
      _context(), [irrelevant, unsafe, too_slow, useful], slack_s=1.0
  )

  assert [x.action for x in summary.candidates] == ["open details"]
  assert summary.rejected == {
      "open settings": "does_not_resolve_current_need",
      "delete event": "unsafe_side_effect",
      "slow": "misses_inference_deadline",
  }
