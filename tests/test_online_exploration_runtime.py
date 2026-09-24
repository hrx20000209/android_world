import dataclasses
import time

from android_world.parallel_exploration.agent_evidence_graph import AgentEvidenceGraph
from android_world.parallel_exploration.agent_evidence_graph import DecisionContext
from android_world.parallel_exploration.agent_evidence_graph import ProbeCandidate
from android_world.parallel_exploration.agent_evidence_graph import ReasoningMode
from android_world.parallel_exploration.online_exploration_runtime import ExtractedFact
from android_world.parallel_exploration.online_exploration_runtime import OnlineExplorationRuntime
from android_world.parallel_exploration.shadow_isolation import InferencePhase
from android_world.parallel_exploration.shadow_isolation import InstanceState
from android_world.parallel_exploration.shadow_isolation import PhaseAwareProbeScheduler
from android_world.parallel_exploration.shadow_isolation import ResourceSnapshot
from android_world.parallel_exploration.shadow_isolation import ShadowIsolationController
from android_world.parallel_exploration.live_probe import _SnapshotView
from android_world.parallel_exploration.live_probe import _known_control_keys


def test_accepted_probe_becomes_visible_only_after_generation_boundary():
  graph = AgentEvidenceGraph(agent_id="agent", policy_fingerprint="policy-v1")
  state = InstanceState("calendar", "ui", "eval", "prefix")
  shadow = [state]
  isolation = ShadowIsolationController(
      capture_primary=lambda: state,
      capture_shadow=lambda: shadow[0],
      rebuild_shadow=lambda _prefix: shadow.__setitem__(0, state),
  )
  runtime = OnlineExplorationRuntime(
      graph=graph,
      scheduler=PhaseAwareProbeScheduler(),
      isolation=isolation,
  )
  context = DecisionContext(
      episode_id="episode",
      state_id="state",
      progress_signature="progress",
      task_need="find time",
      required_slots=("event_time",),
      generation=1,
  )
  candidate = ProbeCandidate(
      action="open details",
      target_slots=("event_time",),
      reveal_probability=1,
      future_calls_avoided=2,
      future_actions_avoided=1,
      probe_latency_s=.01,
      extraction_latency_s=.01,
      resync_latency_s=.01,
      interference_penalty_s=0,
  )
  resource = ResourceSnapshot(
      monotonic_s=time.monotonic(),
      cpu_psi_some_percent=1,
      memory_psi_some_percent=0,
      thermal_level=0,
      inference_phase=InferencePhase.DECODE,
  )

  attempt = runtime.try_probe(
      context,
      [candidate],
      resource_snapshot=resource,
      inference_deadline_s=time.monotonic() + 10,
      authoritative_prefix=(),
      shadow_state_id="state",
      execute_shadow=lambda _candidate, _cancel: "14:30",
      extract_fact=lambda value: ExtractedFact(
          slot="event_time",
          value=value,
          confidence=.99,
          ttl_s=30,
          extraction_verified=True,
      ),
  )

  assert attempt.evidence_id
  assert graph.plan_reasoning(context).mode == ReasoningMode.NORMAL
  next_generation = dataclasses.replace(context, generation=2, now_s=time.time())
  assert graph.plan_reasoning(next_generation).mode == ReasoningMode.ENHANCED


def test_live_probe_accepts_progressive_graph_list_serialization():
  payload = {
      "nodes": [{"node_id": "s0", "visit_count": 2, "decision_entropy": .3}],
      "edges": [{
          "edge_id": "e0", "src_node": "s0", "dst_node": "s1",
          "action": {"control_key": "settings"},
          "discovered_labels": ["Settings"],
      }],
  }

  snapshot = _SnapshotView.from_payload(payload)

  assert snapshot is not None
  assert snapshot.node_visits == {"s0": 2}
  assert snapshot.outgoing == {"s0": ["e0"]}
  assert _known_control_keys(payload, "s0") == {"settings"}
