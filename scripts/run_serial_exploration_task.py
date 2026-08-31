#!/usr/bin/env python3
"""Serial (non-parallel) MobileExplorer runner for fast experiment turnaround.

The real design is the parallel one: exploration rides inside the inference
window, which is free because the UI is static while the model thinks. This
runner exists only because a fast server (~1-2s per inference) leaves no
window to hide exploration in, and waiting on a slow one costs hours per
experiment. Everything else keeps the parallel semantics, in particular the
one that is easy to break by accident:

    Exploration performed while deciding step i must not influence step i.

In the parallel design, step i's exploration runs *concurrently* with step i's
inference, so by construction the model cannot see it, and the skip decision
for step i is taken before that inference even starts. Running the same work
serially puts the exploration results in memory before the inference call,
where nothing but discipline stops them leaking into the prompt - and
discipline is exactly what gets lost in a later edit. So the graph carries an
explicit generation counter: reads for step i are served from the snapshot
taken at the end of step i-1, and a read that would have seen newer data
raises instead of silently returning it.

Per-step order, mirroring the parallel timeline:

  1. capture S_i
  2. skip decision, against the step i-1 snapshot only
  3. if not skipping: explore from S_i (fixed probe count), results go to the
     graph's live generation, invisible to this step
  4. infer on S_i, prompt built from the step i-1 snapshot
  5. execute, record the authoritative transition, bump the generation

Latency here is NOT comparable to the parallel design: serial exploration is
on the critical path, so its cost is real and the inference it saves on a fast
server is only 1-2s. Use this runner to measure skip rate, skip accuracy and
success rate; measure latency benefit with the parallel runner.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


class GenerationError(RuntimeError):
  """A read crossed the step boundary the parallel design guarantees."""


@dataclasses.dataclass
class GraphSnapshot:
  """Read-only view of the belief graph as of the end of one step.

  Holding a snapshot rather than the graph itself is what makes the leak
  impossible instead of merely discouraged: there is no path from here to
  anything exploration wrote during the current step.
  """

  generation: int
  node_visits: dict[str, int]
  node_entropy: dict[str, float]
  edges: dict[str, dict[str, Any]]
  outgoing: dict[str, list[str]]

  def screen_briefing(self, node_id: str, taken: set[str]) -> str:
    """What the graph knows about the screen in front of the model.

    Speculative edges are poor at deciding (37-48% of skips built on them
    landed as predicted, against 75% for edges the model itself had taken),
    because a probe only establishes where an action leads, never that it is
    the action worth taking. That same knowledge is still worth stating: it
    is exactly what the model would otherwise burn a step discovering.

    Controls already used on this screen are listed separately rather than
    omitted. An earlier version phrased everything as "Tapping X opens Y",
    which reads as a suggestion, and the model responded by tapping the same
    control again - MarkorCreateFolder looped between "+" and the filename
    field for ten steps (2026-08-30). Naming what is already done is what
    turns the briefing from a prompt into an observation.
    """
    lines_new: list[str] = []
    lines_done: list[str] = []
    for edge_id in self.outgoing.get(node_id, ()):
      edge = self.edges[edge_id]
      if edge["status"] == "INVALID" or not edge["discovered_labels"]:
        continue
      anchor = str(edge["action"].get("element_identity", "")).split("|")
      name = next((part for part in anchor[1:3] if part), None)
      where = ""
      if edge["action"].get("x") is not None:
        where = f" at ({edge['action']['x']},{edge['action']['y']})"
      leads_to = ", ".join(edge["discovered_labels"][:4])
      entry = f"- {name or 'a control'}{where} leads to a screen showing: {leads_to}"
      (lines_done if edge_id in taken else lines_new).append(entry)
    parts = []
    if lines_new:
      parts.append("Known from having looked ahead on this screen:\n"
                   + "\n".join(lines_new[:4]))
    if lines_done:
      parts.append("Already used on this screen (do not repeat unless the "
                   "task needs it again):\n" + "\n".join(lines_done[:3]))
    return "\n\n".join(parts)

  def reusable_path(self, node_id: str, recent: tuple[str, ...] = (),
                    max_len: int = 3) -> list[dict[str, Any]]:
    """A run of reusable edges leading away from node_id.

    Every skip so far replayed exactly one edge and then handed control back,
    which saves an inference but never a step - AndroidWorld counts one step
    per agent.step() regardless of how much happened inside it. The graph
    stores paths, not just edges, so a stretch the agent has already walked
    (list -> tap FAB -> form -> tap Save) can be replayed as a unit inside a
    single counted step. That is the only mechanism here that shortens the
    trajectory rather than just making it cheaper.

    Stops as soon as the chain leaves what is known: an unseen destination,
    a node already on the recent path (which would be a cycle), or a hop with
    no reusable continuation. Each hop is still verified against its recorded
    destination at execution time, so a path that stops matching is abandoned
    mid-way rather than followed blindly.
    """
    path: list[dict[str, Any]] = []
    seen = set(recent)
    current = node_id
    while len(path) < max_len:
      edge = self.reusable_edge(current, tuple(seen))
      if edge is None:
        break
      path.append(edge)
      seen.add(current)
      current = edge["dst_node"]
      if current is None or current in seen:
        break
    return path

  def reusable_edge(self, node_id: str,
                    recent: tuple[str, ...] = ()) -> dict[str, Any] | None:
    """Best edge out of this screen that may be executed without inference.

    Two sources, and the difference matters:

    REUSABLE - a lookahead hop explored beyond a probe whose guess the model
    then confirmed (prefix alignment). This is the one that gets AHEAD of the
    authoritative path: nothing has ever executed it, and skipping with it
    saves an inference the model would otherwise have had to run. No revisit
    is required, because the licence comes from the alignment, not from
    having been here before.

    VERIFIED on a revisited screen - the model's own past decision in this
    same situation. Cannot lead anywhere new, but is the most reliable thing
    available (75% landed as predicted, against 37-48% for skips built on
    unaligned speculative edges).
    """
    # Nothing left to decide here: the map of this screen shows a single
    # viable continuation, so H = 0 and an inference would be choosing
    # between one option. Requires the screen to have been mapped by
    # exploration - an unmapped screen has H = inf and never qualifies - which
    # is what makes this a use for exploration that does not depend on
    # predicting the model, only on covering the screen.
    # Revisit required, because H = 0 has two very different causes: the
    # screen genuinely offers one continuation, or exploration mapped one
    # element out of fifteen and stopped. A single visit cannot tell them
    # apart. Across repeated visits the explorer probes elements it has not
    # tried here before (already_explored_element_identities), so a screen
    # that still shows one viable continuation after several passes is
    # decided rather than merely unexamined - which is also why accumulating
    # coverage over visits, not guessing well on one visit, is what this
    # mechanism is built on.
    entropy = self.node_entropy.get(node_id, float("inf"))
    if entropy <= 1e-9 and self.node_visits.get(node_id, 0) > 1:
      settled = [
          self.edges[eid] for eid in self.outgoing.get(node_id, ())
          if self.edges[eid]["status"] not in ("INVALID",)
          and self.edges[eid]["dst_node"] not in (None, node_id)
          and self.edges[eid]["dst_node"] not in recent
          and self.edges[eid]["risk_level"] in {"SAFE", "LOW"}
      ]
      if len(settled) == 1:
        return settled[0]

    revisited = self.node_visits.get(node_id, 0) > 1
    candidates = [
        self.edges[edge_id] for edge_id in self.outgoing.get(node_id, ())
        if (self.edges[edge_id]["status"] == "REUSABLE"
            or (revisited and self.edges[edge_id]["status"] == "VERIFIED"))
        and self.edges[edge_id]["dst_node"] not in (None, node_id)
        # Never skip back onto a screen the trajectory just came from. Reusing
        # a remembered edge is only progress if it leads somewhere new; an
        # edge pointing into the recent path closes a cycle, and the graph has
        # no way to notice it is going round because each individual hop still
        # verifies correctly. CameraTakeVideo spent 13 skips this way on
        # 2026-08-31 - every one of them landed exactly where predicted, and
        # the task still ran out of steps without progressing.
        and self.edges[edge_id]["dst_node"] not in recent
        and self.edges[edge_id]["risk_level"] in {"SAFE", "LOW"}
    ]
    if not candidates:
      return None
    # A promoted lookahead beats a replay: it is the one that can save an
    # inference on a step the model has not already answered.
    return max(candidates, key=lambda e: (e["status"] == "REUSABLE", e["confidence"]))


class GenerationGuardedGraph:
  """Belief graph that only ever serves last step's view.

  Writes land in the live generation; reads go through snapshot(), which
  returns the previous generation. Asking for the live generation raises,
  so a future edit that tries to use this step's exploration fails loudly in
  the first test run rather than quietly inflating the skip rate.
  """

  def __init__(self, graph):
    self._graph = graph
    self._generation = 0
    self._snapshot = self._capture(0)

  @property
  def live(self):
    """The mutable graph. Writers only."""
    return self._graph

  @property
  def generation(self) -> int:
    return self._generation

  def _capture(self, generation: int) -> GraphSnapshot:
    edges = {
        edge_id: {
            "edge_id": edge_id,
            "src_node": edge.src_node,
            "dst_node": edge.dst_node,
            "status": edge.status.value,
            "confidence": edge.confidence,
            "risk_level": edge.risk_level,
            "action": dict(edge.action),
            "discovered_labels": tuple(edge.discovered_labels),
            "inverse_level": edge.inverse_level,
            "probe_count": edge.probe_count,
            "inference_alignment_count": edge.inference_alignment_count,
            "execution_hit_count": edge.execution_hit_count,
            "execution_miss_count": edge.execution_miss_count,
            "skip_attempt_count": edge.skip_attempt_count,
            "skip_success_count": edge.skip_success_count,
            "rollback_success_count": edge.rollback_success_count,
            "rollback_failure_count": edge.rollback_failure_count,
            "cumulative_realized_ig": edge.cumulative_realized_ig,
            "cumulative_exploration_cost": edge.cumulative_exploration_cost,
            "last_updated_generation": edge.last_updated_generation,
        }
        for edge_id, edge in self._graph.edges.items()
    }
    outgoing: dict[str, list[str]] = {}
    for edge_id, edge in edges.items():
      outgoing.setdefault(edge["src_node"], []).append(edge_id)
    for node_id in list(self._graph.nodes):
      self._graph.recompute_decision_entropy(node_id)
    return GraphSnapshot(
        generation=generation,
        node_visits={n: node.visit_count for n, node in self._graph.nodes.items()},
        node_entropy={n: node.decision_entropy for n, node in self._graph.nodes.items()},
        edges=edges,
        outgoing=outgoing,
    )

  def snapshot(self, for_step: int) -> GraphSnapshot:
    """The view step `for_step` is allowed to see."""
    if self._snapshot.generation > for_step:
      raise GenerationError(
          f"step {for_step} asked for generation {self._snapshot.generation}: "
          "exploration from the current step must not be visible to it")
    return self._snapshot

  def commit_step(self) -> None:
    """End of step: this step's writes become visible to the next one."""
    self._generation += 1
    self._snapshot = self._capture(self._generation)


def _has_unsaved_input(state) -> bool:
  """Is there typed text on screen that no rollback could put back?

  The whole exploration mechanism rests on probes being reversible, and the
  recovery ladder reverses navigation: Back closes a screen, and the text the
  user typed into it goes with it. Nothing in the ladder can retype it.

  Measured on 2026-08-31: of the tasks exploration lost, the ones that stayed
  lost even after the stranded-device repair landed correctly were the ones
  probed mid-form - ContactsAddContact, probed at step 8 while the contact
  editor held entered fields, came back to the right screen with the form
  cleared. Detected from the widget class and its content, so it needs no
  per-app knowledge: a text-entry widget that currently holds text.
  """
  for element in getattr(state, "elements", ()):
    cls = (element.class_name or "").rsplit(".", 1)[-1].casefold()
    if "edittext" in cls or "autocomplete" in cls:
      if (element.text or "").strip():
        return True
  return False


def _target_app_from_goal(goal: str, installed: list[str]) -> str | None:
  """Which installed app the task names, or None.

  The first action of almost every task is open_app on the app the goal names,
  and it is the single most predictable decision in a trajectory - yet it is
  also one exploration can never anticipate, because launching an app is not a
  UI probe. Measured on 2026-08-31, open_app was 11% of all model actions and
  fell entirely into the "no probe type can express this" bucket.

  Matching against the device's actual app list rather than a hand-written
  alias table is what keeps this from being the kind of keyword heuristic the
  reviews objected to: it needs no per-app rule and adapts to whatever is
  installed. Longest match wins so that "Simple Gallery Pro" is not shadowed
  by a shorter name it contains.
  """
  def normalize(text: str) -> str:
    # Trailing -s only: "create a new contact" must reach the Contacts app,
    # and no app-specific rule should be needed to say so.
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    return " " + " ".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words) + " "

  low = normalize(goal)
  hits = [name for name in installed if normalize(name).strip() in low]
  return max(hits, key=len) if hits else None


def _committed_actions(history: Any) -> list[dict[str, Any]]:
  """The agent's own executed actions, in the shape the replay expects.

  The agent stores whole step records, so the action has to be lifted out of
  action_dict (and the tool_call kept, since replay reads click coordinates
  from it). Handing over the raw records instead silently replays nothing.
  """
  out: list[dict[str, Any]] = []
  for item in history or []:
    if not isinstance(item, dict):
      continue
    raw = item.get("action_dict") or item.get("action_output_json") or item
    out.append({
        "action_dict": dict(getattr(raw, "__dict__", raw) or {}),
        "tool_call": item.get("tool_call") or {},
    })
  return out


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--task", required=True)
  parser.add_argument("--max_steps", type=int, default=20)
  parser.add_argument("--seed", type=int, default=30)
  parser.add_argument("--probes_per_step", type=int, default=3,
                      help="Fixed probe count; serial exploration is on the "
                           "critical path so this is a real cost, not a free one.")
  parser.add_argument("--max_depth", type=int, default=2)
  parser.add_argument("--enable_skip", action="store_true")
  parser.add_argument("--probe_budget", type=int, default=12,
                      help="Probes allowed across the whole episode.")
  parser.add_argument("--max_path", type=int, default=3,
                      help="Hops replayed inside one counted step.")
  parser.add_argument("--graph_context", choices=("off", "briefing", "distill"),
                      default="distill",
                      help="how graph knowledge reaches the prompt (PART G ablation): "
                           "off = never, briefing = the old screen_briefing, "
                           "distill = GraphDistiller facts under a token budget")
  parser.add_argument("--predictive_scorer", action="store_true", default=True,
                      help="rank probe candidates with PredictiveElementScorer")
  parser.add_argument("--no_predictive_scorer", dest="predictive_scorer",
                      action="store_false")
  parser.add_argument("--inject_briefing", action="store_true",
                      help="Summarize the graph's knowledge of the current "
                           "screen into the next prompt (uses the step i-1 "
                           "snapshot, never the current step's exploration).")
  parser.add_argument("--api_url",
                      default=os.environ.get(
                          "ANDROID_WORLD_LLM_API_URL",
                          "http://localhost:8084/v1/chat/completions"))
  parser.add_argument("--a11y_socket_port", type=int, default=8765)
  args = parser.parse_args()

  os.environ["ANDROID_WORLD_A11Y_METHOD"] = "fast_provider"
  os.environ["ANDROID_WORLD_FAST_A11Y_SOCKET_PORT"] = str(args.a11y_socket_port)
  os.environ["ANDROID_WORLD_LLM_API_URL"] = args.api_url
  import runpy
  import subprocess
  subprocess.run(
      ["adb", "-s", "emulator-5554", "forward", f"tcp:{args.a11y_socket_port}",
       "localabstract:androidworld_fast_a11y"], check=True, timeout=10)

  from android_world.agents import base_agent
  from android_world.agents import gelab_agent
  from android_world.env import json_action
  from android_world.parallel_exploration.belief_graph import EdgeStatus
  from android_world.parallel_exploration.belief_graph import GraphNode
  from android_world.parallel_exploration.belief_graph import NodeStatus
  from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
  from android_world.parallel_exploration.information import parse_reasoning_prior
  from android_world.parallel_exploration import graph_distiller as gd
  from android_world.parallel_exploration.live_probe import _is_app_content
  from android_world.parallel_exploration import state_graph_information as sgi
  from android_world.parallel_exploration.live_probe import element_identity_from_dict
  from android_world.parallel_exploration.live_probe import await_explorer_ready
  from android_world.parallel_exploration.live_probe import spawn_explorer
  from android_world.parallel_exploration.live_probe import run_serial_exploration
  from android_world.parallel_exploration.live_probe import stop_prepared_explorer
  from android_world.parallel_exploration.state import create_optimized_state_capture

  root = args.output.resolve()
  root.mkdir(parents=True, exist_ok=True)
  trace_path = root / "probe_trace.jsonl"
  graph = ProgressiveBeliefGraph(args.task)
  guarded = GenerationGuardedGraph(graph)
  capture = create_optimized_state_capture(
      serial="emulator-5554", console_port=5554,
      adb_path="/Users/huangrunxi/Library/Android/sdk/platform-tools/adb",
      a11y_local_port=args.a11y_socket_port,
  )
  events: list[dict[str, Any]] = []
  installed_apps = list(gelab_agent.AVAILABLE_APPS)
  # Session-local, shared by exploration ranking and the reasoning gate: both
  # read the same graph statistics, which is what makes "exploration improves
  # exploration" and "exploration improves reasoning" one mechanism rather
  # than two.
  contextual_history = sgi.ContextualHistoryTable()
  distiller = gd.GraphDistiller()
  reasoning_gate = gd.ReasoningGate(
      distiller,
      gd.GateConfig(enable_skip=args.enable_skip,
                    enable_graph_context=args.graph_context != "off"))

  def need_type_of(model_text: str, task_goal: str) -> str:
    return sgi._need_type(parse_reasoning_prior(model_text, task_goal).to_dict())  # pylint: disable=protected-access

  def log(kind: str, **fields: Any) -> None:
    row = {"kind": kind, "step": fields.pop("step", None), **fields}
    events.append(row)
    with (root / "serial_events.jsonl").open("a", encoding="utf-8") as stream:
      stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

  def node_id_of(signature) -> str:
    return graph.make_node_id(signature.activity.component, signature.layout_sig)

  def upsert(signature, node_id: str, visited: bool) -> None:
    graph.upsert_node(GraphNode(
        node_id=node_id, activity=signature.activity.component,
        package=signature.activity.component.split("/", 1)[0],
        visual_signature=signature.phash,
        structural_signature=signature.struct_sig.digest,
        layout_signature=signature.layout_sig,
        status=NodeStatus.COMMITTED,
    ), visited=visited)

  trace_cursor = {"n": 0}
  episode_goal = {"text": ""}
  depth1_edge_ids: list[str] = []
  # (activity, probe_type) combinations that have already swallowed a probe on
  # this episode. Learned online: the first attempt on any screen is allowed,
  # and a screen that cannot be rolled back stops being probed that way. This
  # is what lets exploration cover every step without the blanket
  # "unexplored nodes only" rule that removed it from 82% of them.
  blocked_recovery_contexts: set[str] = set()
  blocked_element_identities: set[str] = set()
  # Set the first time a probe leaves the app and has to be repaired. After
  # that this episode stops probing entirely - see may_probe.
  stranded_once = {"value": False}
  # This step's depth-1 probes, carried to the next step where the real
  # landing state can confirm or refute them.
  pending_prefix_edge_ids: list[str] = []
  last_real_action: dict[str, Any] = {}
  # The model's own reasoning text from the previous step. This is the
  # reasoning -> exploration link (paper Eq. 4): the explorer ranks candidates
  # against what the model just said it is looking for, instead of against the
  # task goal, which never changes within an episode and therefore ranks every
  # screen the same way.
  last_model_output = {"text": ""}
  # The nodes the trajectory has stood on lately, newest last. Used to refuse
  # skips that would walk back into them.
  recent_nodes: list[str] = []
  consecutive_skips = {"n": 0}
  skip_record = {"n": 0, "ok": 0}
  probe_total = {"n": 0}

  def known_inverse_levels() -> dict[str, str]:
    """What undid each kind of transition, per screen, so far this episode."""
    out: dict[str, str] = {}
    for edge in graph.edges.values():
      if not edge.inverse_level or edge.rollback_success is not True:
        continue
      node = graph.nodes.get(edge.src_node)
      if node is None:
        continue
      probe = str(edge.action.get("probe_type") or "")
      if probe:
        out.setdefault(f"{node.activity}|{probe}", edge.inverse_level)
    return out

  def ingest_probe_trace(trial_id: str) -> None:
    """Write what the probes found into the graph.

    Without this the explorer runs, pays every rollback risk, and its findings
    are thrown away: on 2026-08-30 the graph held 8 edges for
    MarkorCreateFolder and all 8 were authoritative, while probe_trace.jsonl
    held 4 unread probe rows. That run therefore measured the full cost of
    exploration against none of its benefit.

    Probe edges enter as SPECULATIVE. They are not skip candidates on their
    own - a probe establishes where an action leads, never that it is the
    action worth taking - and become reusable only through prefix alignment
    (see promote_children_of_aligned_prefix), which is what supplies the
    missing intent.
    """
    nonlocal_ids: list[str] = []
    if not trace_path.exists():
      depth1_edge_ids.clear()
      return
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    fresh = [json.loads(l) for l in lines[trace_cursor["n"]:] if l.strip()]
    trace_cursor["n"] = len(lines)
    for row in fresh:
      if row.get("trial_id") != trial_id or "graph" not in row:
        continue
      # Near-miss: the probe did come back, but only by walking the deep end
      # of the recovery ladder. NOOP and INVERSE mean the action was
      # structurally reversible; BACK_N, BACK_OVERLAY, DEEPLINK and
      # TRAJECTORY_REPLAY all mean a real navigation happened and the device
      # had to be rebuilt rather than undone. Treating that as a warning is
      # how this stops paying for the first failure on every new screen:
      # blocklisting only after an outright failure means each new
      # (screen, control type) pair costs one unrecoverable state to learn,
      # which was exactly the floor observed on 2026-08-31 - 15 failures, one
      # per new combination, none of them repeats.
      deep_recovery = str(row.get("recovery_level") or "") in {
          "BACK_N", "BACK_OVERLAY", "DEEPLINK", "TRAJECTORY_REPLAY"}
      if deep_recovery and row.get("recovery_ok"):
        src_act = str(((row.get("graph") or {}).get("src") or {}).get("activity") or "")
        if src_act:
          blocked_recovery_contexts.add(f"{src_act}|{row.get('probe_type')}")
          if int(row.get("depth", 1)) >= 2:
            blocked_recovery_contexts.add(f"{src_act}|DEPTH2")

      # A probe that left the task's own package is not worth repeating even
      # when recovery happened to work. Those screens are account pages,
      # system settings, feedback forms and share sheets - they can raise
      # system dialogs that then sit on top of the app (AudioRecorderRecordAudio
      # on 2026-08-31: a probe reached "Signed in as <account>", and the model
      # spent the next six steps fighting a mobile-data notification and
      # re-launching the app to escape it). Package identity is a structural
      # fact, so this needs no per-app rule and no keyword list.
      reached = str((row.get("discovered") or {}).get("reached_activity") or "")
      reached_pkg = reached.split("/", 1)[0]
      probed_pkg = str(row.get("app_package") or "")
      if reached_pkg and probed_pkg and reached_pkg != probed_pkg:
        src_act = str(((row.get("graph") or {}).get("src") or {}).get("activity") or "")
        if src_act:
          blocked_recovery_contexts.add(f"{src_act}|{row.get('probe_type')}")
        blocked_element_identities.add(
            element_identity_from_dict(row.get("element", {})))
        continue

      if row.get("notes") in {"RESTORE_FAILED", "NESTED_RESTORE_FAILED"}:
        # Rollback could not be verified, so the recorded destination is not
        # trustworthy evidence about anything - and this screen just showed it
        # cannot take this kind of probe.
        src_act = str(((row.get("graph") or {}).get("src") or {}).get("activity") or "")
        if src_act:
          blocked_recovery_contexts.add(f"{src_act}|{row.get('probe_type')}")
          # A depth-2 failure closes depth 2 on this screen for every control
          # type: the deeper hop is where the recovery ladder runs out (20% of
          # TAP_NAV probes at depth 2 could not be rolled back, against 7% at
          # depth 1), so one demonstration is enough.
          if int(row.get("depth", 1)) >= 2:
            blocked_recovery_contexts.add(f"{src_act}|DEPTH2")
        continue
      item = row["graph"]
      src, dst = item["src"], item["dst"]
      sid = graph.make_node_id(src["activity"], src.get("layout_signature", src["structural_signature"]))
      did = graph.make_node_id(dst["activity"], dst.get("layout_signature", dst["structural_signature"]))
      for nid, value, status in ((sid, src, NodeStatus.COMMITTED),
                                 (did, dst, NodeStatus.SPECULATIVE)):
        graph.upsert_node(GraphNode(
            node_id=nid, activity=value["activity"],
            package=value["activity"].split("/", 1)[0],
            visual_signature=value["visual_signature"],
            structural_signature=value["structural_signature"],
            layout_signature=value.get("layout_signature", value["structural_signature"]),
            status=status,
        ))
      element = row.get("element", {})
      role = str(element.get("class", "")).rsplit(".", 1)[-1].casefold()
      depth = int(row.get("depth", 1))
      risk = ("LOW" if depth >= 2 and bool(row.get("recovery_ok"))
              else ("LOW" if role in {"imagebutton", "tabwidget"} else "UNKNOWN"))
      action_with_kind = dict(item["action"])
      action_with_kind["probe_type"] = row.get("probe_type")
      edge = graph.add_speculative_transition(
          sid, action_with_kind, did,
          path_probability=max(0.05, float(element.get("score", 0.0))),
          confidence=0.20,
          expected_information_gain=min(
              1.0, (row.get("discovered") or {}).get("new_element_count", 0) / 10.0),
          risk_level=risk,
          exploration_cost=float((row.get("timings_ms") or {}).get("total", 0.0)) / 1000.0,
          rollback_success=bool(row.get("recovery_ok")),
          discovered_labels=tuple(
              str(x) for x in (row.get("discovered") or {}).get("new_texts", [])[:8]),
          inverse_level=str(row.get("recovery_level") or "") if row.get("recovery_ok") else "",
      )
      # PART C: the probe itself is a measurement, not only the edge it
      # created. What it cost and whether it came back are what later decide
      # whether this kind of probe is worth running again.
      cost_s = float((row.get("timings_ms") or {}).get("total", 0.0)) / 1000.0
      realized_ig = min(
          1.0, (row.get("discovered") or {}).get("new_element_count", 0) / 10.0)
      graph.record_probe(edge.edge_id, rollback_ok=bool(row.get("recovery_ok")),
                         cost_s=cost_s, realized_ig=realized_ig,
                         generation=guarded.generation)
      contextual_history.record_probe(
          sgi.ContextualHistoryTable.key(
              str(row.get("probe_type") or ""),
              sgi._role_of_class(str(element.get("class") or "")),
              need_type_of(last_model_output["text"], episode_goal["text"])),
          rollback_ok=bool(row.get("recovery_ok")), realized_ig=realized_ig)
      if depth == 1:
        nonlocal_ids.append(edge.edge_id)
    depth1_edge_ids.clear()
    depth1_edge_ids.extend(nonlocal_ids)

  def repair_after_exploration(agent, before, outcome, step: int) -> None:
    """Put the app back in front before the next step reasons about it.

    When the explorer's recovery ladder bottoms out it reports the failure and
    stops; until now the runner only recorded that and carried on, so the next
    inference read whatever screen the probe had stranded the device on.
    Measured on 2026-08-31 by running the same six tasks three ways: with
    every mechanism off, 6/6 passed; with skipping alone, 6/6; with
    exploration alone, 3/6 - and two of the three losses follow a step-2
    RESTORE_FAILED, one of them stranded in Chrome on a "Privacy error" page
    after a probe left the app entirely.

    Repairs navigation only: HOME, then relaunch the app the episode was in.
    The recovery ladder's TRAJECTORY_REPLAY reconstructs state by replaying
    committed actions, which is right inside the explorer where the actions
    are known to be safe to repeat; doing that here would re-fire the agent's
    own clicks and typing against live data, and on a task whose committed
    actions include deletions that is worse than the state it is fixing.
    Getting back into the right app at its own entry screen costs nothing and
    is what the stranded cases actually needed.

    Executed outside the agent's action history, like the bootstrap launch, so
    the repair does not consume a step of the episode's budget.
    """
    stranded = capture.capture()
    expected_pkg = before.activity.component.split("/", 1)[0]
    actual_pkg = stranded.activity.component.split("/", 1)[0]
    if outcome.get("restore_status") == "RESTORED" and actual_pkg == expected_pkg:
      return
    target = next((name for name in installed_apps
                   if name.casefold().replace(" ", "") in expected_pkg.casefold()
                   or expected_pkg.casefold().endswith(
                       name.casefold().replace(" ", ""))), None)
    if target is None:
      target = _target_app_from_goal(episode_goal["text"], installed_apps)
    if target is None:
      log("repair_skipped", step=step, reason="no app name for package",
          package=expected_pkg, stranded=stranded.activity.component)
      return
    try:
      # Back first. Relaunching resets the app to its entry screen, which
      # throws away scroll position, an open dialog, a half-filled form - real
      # progress the episode paid steps for. Most strandings are one screen
      # deep (a chooser, a browser the probe opened), and Back returns from
      # those with the app's own state intact. HOME + relaunch is the fallback
      # for the cases Back cannot reach, notably the launcher itself.
      after_repair = stranded
      via = ""
      for attempt in range(3):
        agent._execute_action(json_action.JSONAction(action_type="navigate_back"), {})
        time.sleep(0.3)
        after_repair = capture.capture()
        if after_repair.activity.component.split("/", 1)[0] == expected_pkg:
          via = f"back x{attempt + 1}"
          break
      if not via:
        agent._execute_action(json_action.JSONAction(action_type="navigate_home"), {})
        time.sleep(0.3)
        agent._execute_action(
            json_action.JSONAction(action_type="open_app", app_name=target), {})
        time.sleep(0.6)
        after_repair = capture.capture()
        via = "relaunch"
      ok = after_repair.activity.component.split("/", 1)[0] == expected_pkg
      dirty_from_last_step["value"] = not ok
      # One escape ends probing for this episode. Measured on the 22 tasks
      # completed 2026-08-31: every one of the 4 losses had been stranded at
      # least once, against 39% of the 18 wins. Repair puts the app back on
      # screen but cannot put back what was on it, and the per-(activity,
      # probe_type) blocklist only rules out the exact combination that just
      # failed - which is why episodes kept escaping again through a
      # different control. Exploration still runs on every step up to that
      # point, and on episodes that never escape it runs throughout.
      stranded_once["value"] = True
      log("repair", step=step, ok=ok, app=target, via=via,
          stranded=stranded.activity.component,
          landed=after_repair.activity.component)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      log("repair_failed", step=step, error=str(exc)[:300])

  original_step = gelab_agent.GELABAgent.step
  original_build = gelab_agent.build_gelab_messages
  step_counter = {"i": 0}
  dirty_from_last_step = {"value": False}
  taken_edge_ids: set[str] = set()
  briefing_now = {"text": ""}
  gate_mode = {"value": gd.NORMAL_INFERENCE, "context": ""}

  def build_with_briefing(goal_text, history, screenshot):
    """Append the graph's briefing for the current screen to the prompt.

    briefing_now is filled from the step i-1 snapshot before inference runs,
    so this cannot smuggle in anything the current step explored - the
    generation guard is what makes that a structural guarantee rather than a
    convention.
    """
    messages = original_build(goal_text, history, screenshot)
    text = briefing_now["text"]
    if not text:
      return messages
    for message in messages:
      content = message.get("content")
      if not isinstance(content, list):
        continue
      for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
          item["text"] = f"{item['text']}\n\n{text}"
          return messages
    return messages

  def serial_step(self, goal: str):
    episode_goal["text"] = goal
    step = step_counter["i"]
    before = capture.capture()
    src_id = node_id_of(before)
    upsert(before, src_id, visited=True)

    # Prefix alignment: did last step's probe guess land where the model's
    # real action then landed? If so the explorer read the intent correctly
    # at that point, and the deeper hops it explored from there earn the
    # right to be reused - which is the only way a skip can get AHEAD of the
    # authoritative path instead of merely replaying it. Judged by the
    # landing state rather than by comparing coordinates, so it holds for
    # every action type.
    aligned_now = 0
    for edge_id in list(pending_prefix_edge_ids):
      edge = graph.edges.get(edge_id)
      if (edge is None or edge.dst_node is None or edge.dst_node == edge.src_node
          or edge.rollback_success is not True or edge.dst_node != src_id):
        continue
      graph.record_inference_alignment(edge.src_node, edge.action, True)
      promoted = graph.promote_children_of_aligned_prefix(edge.edge_id)
      graph.record_execution_verification(edge.edge_id, True)
      aligned_now += 1
      if promoted:
        log("prefix_aligned", step=step, edge_id=edge_id,
            promoted=[c.edge_id for c in promoted])
    if pending_prefix_edge_ids:
      # Also record whether the model's action was among the probed
      # candidates at all, separately from whether the landing state then
      # matched. Alignment can fail for two different reasons - the ranker
      # never guessed this action, or it guessed it but the destination was
      # recorded differently - and only the first is fixed by widening the
      # probe budget. Conflating them would send the next experiment after
      # the wrong one.
      guessed = 0
      for edge_id in pending_prefix_edge_ids:
        edge = graph.edges.get(edge_id)
        if edge is None:
          continue
        ex, ey = edge.action.get("x"), edge.action.get("y")
        ax, ay = last_real_action.get("x"), last_real_action.get("y")
        same_type = (str(edge.action.get("action_type", "")).lower()
                     == str(last_real_action.get("action_type", "")).lower())
        if same_type and None not in (ex, ey, ax, ay):
          # Same control if the model's tap falls inside the probed element's
          # neighbourhood; the element bounds are not carried on the edge, so
          # this uses the probe's own click point as the anchor.
          if (float(ex) - float(ax)) ** 2 + (float(ey) - float(ay)) ** 2 <= 120 ** 2:
            guessed += 1
        elif same_type and str(edge.action.get("app_name") or "") == str(
            last_real_action.get("app_name") or "") and edge.action.get("app_name"):
          guessed += 1
      log("prefix_check", step=step, candidates=len(pending_prefix_edge_ids),
          aligned=aligned_now, action_was_guessed=guessed,
          real_action=last_real_action.get("action_type"))
    pending_prefix_edge_ids.clear()

    # Step 0 is open_app on the app the goal names. That decision is fixed by
    # the task text, so spending an inference on it buys nothing - and unlike
    # every other action it is one exploration can never anticipate, since
    # launching an app is not a UI probe (open_app was 11% of all model
    # actions on 2026-08-31, entirely inside the "no probe expresses this"
    # bucket). Only from the launcher, and only before any action has been
    # taken, so it cannot fire mid-task.
    if (args.enable_skip and step == 0 and not getattr(self, "_actions", [])
        and "nexuslauncher" in before.activity.component):
      target = _target_app_from_goal(goal, installed_apps)
      if target:
        started = time.perf_counter()
        action = json_action.JSONAction(action_type=json_action.OPEN_APP, app_name=target)
        self._execute_action(action, {})
        time.sleep(0.4)
        # Package check via adb, deliberately not capture.capture(). The
        # runner holds a persistent fast-a11y socket and AndroidWorld's
        # controller holds another; a dump issued from here while the app is
        # still binding its accessibility service blocked the run outright on
        # 2026-08-31 (the launch logged fine, then nothing - no further step
        # ever ran). Reading the resumed component costs one shell call and
        # touches neither socket. The graph is left to the next step, which
        # captures the post-launch state anyway.
        opened = False
        try:
          out = subprocess.run(
              ["adb", "-s", "emulator-5554", "shell", "dumpsys", "activity", "activities"],
              capture_output=True, text=True, timeout=6).stdout
          top = re.search(r"topResumedActivity=.*?\{[^}]*?\s(\S+)/", out)
          opened = bool(top) and "nexuslauncher" not in top.group(1)
        except (OSError, subprocess.SubprocessError):
          pass
        summary = str({"action": "open_app", "text": target})
        self._summaries.append(summary)
        log("skip", step=step, kind_detail="app_bootstrap", matched=opened,
            latency_s=time.perf_counter() - started, action=action.__dict__)
        # No return, and self._actions is left alone on purpose. Launching the
        # app is fixed by the task text, so it is setup rather than a decision
        # worth a step of the episode's budget: falling through runs this same
        # step's inference on the screen the app just opened, so the launch and
        # the first real decision collapse into one counted step and the
        # trajectory is genuinely one step shorter. The agent derives its step
        # index from len(self._actions), so appending here would push the
        # episode forward and undo exactly that; the action stays visible to
        # the model through _summaries.
        #
        # An earlier attempt at this looked like it deadlocked. It did not -
        # the smoke command had not exported ANDROID_WORLD_LLM_API_URL, so the
        # runner fell back to its old default on 8083, a port still listening
        # from a dead VSCode forward, and every inference hung there.
        before = capture.capture()
        src_id = node_id_of(before)
        upsert(before, src_id, visited=True)

    # P_{i-1}: what the model said it was looking for in its LAST output.
    # Computed once per step and shared by exploration ranking and the
    # distiller, so both are conditioned on the same statement of need.
    current_need = parse_reasoning_prior(last_model_output["text"], goal)
    # Reset before the gate runs: leaving the previous step's verdict in place
    # would let a mode decided on a screen we have since left govern this one.
    gate_mode["value"], gate_mode["context"] = gd.NORMAL_INFERENCE, ""

    # (2) Skip decision, against last step's snapshot only.
    if args.enable_skip:
      # A run of skips means several moves in a row that no model looked at.
      # Each one is verified against the state it predicted, but nothing
      # checks that the sequence is heading anywhere, so a wrong turn early on
      # can be followed for the rest of the episode. Handing the next decision
      # back to the model after a few of them bounds that.
      # Stop skipping once this episode's own skips have been mostly wrong.
      # The consecutive-skip cap bounds how long a bad run can last but says
      # nothing about whether the runs are any good: the six tasks v4 lost
      # outright all had skip accuracy under 50% (MarkorCreateFolder 3 wrong
      # of 5, RecipeDeleteDuplicateRecipes 4 of 5), while the tasks it kept
      # were mostly at or above it. Reading back the episode's own record is
      # self-correction from observation, not another tuned knob.
      attempted, correct = skip_record["n"], skip_record["ok"]
      accurate = attempted < 3 or correct * 2 >= attempted
      allowed = consecutive_skips["n"] < 3 and accurate
      # Earn the right to chain. A multi-hop replay commits several moves on
      # one prediction, so it is only worth the exposure once this episode's
      # skips have actually been landing; before that, take one hop at a time
      # and let each be verified.
      hops = args.max_path if (skip_record["n"] >= 2 and
                               skip_record["ok"] == skip_record["n"]) else 1
      candidate_path = (guarded.snapshot(for_step=step).reusable_path(
                            src_id, tuple(recent_nodes[-4:]), max_len=hops)
                        if allowed else [])
      # B9: one gate decides all three consumption modes from the same belief
      # state, so a step that is not certain enough to skip still gets the
      # benefit of what exploration found instead of falling back to a plain
      # inference. The gate judges the FIRST hop - the only one that starts
      # from a state actually observed - and the walk below re-verifies each
      # subsequent hop against the state it predicted.
      gate_decision = reasoning_gate.decide(
          current_node_id=src_id,
          graph_snapshot=guarded.snapshot(for_step=step),
          information_need=current_need.to_dict(),
          reusable_edge=candidate_path[0] if candidate_path else None,
          taken_edges=taken_edge_ids,
          recent_nodes=tuple(recent_nodes[-4:]),
          consecutive_skips=consecutive_skips["n"])
      gate_mode["value"] = gate_decision.mode
      gate_mode["context"] = gate_decision.graph_context
      log("gate", step=step, mode=gate_decision.mode, reason=gate_decision.reason,
          had_reusable=bool(candidate_path),
          context_tokens=len(gate_decision.graph_context.split()))
      path = candidate_path if gate_decision.mode == gd.SKIP_INFERENCE else []
      if path:
        started = time.perf_counter()
        walked = 0
        after = None
        for edge in path:
          action_dict = {k: v for k, v in edge["action"].items()
                         if k in {"action_type", "x", "y", "text", "direction", "app_name"}}
          action_dict["action_type"] = str(action_dict.get("action_type", "")).lower()
          self._execute_action(json_action.JSONAction(**action_dict), {})
          time.sleep(0.2)
          after = capture.capture()
          landed = node_id_of(after)
          matched = landed == edge["dst_node"]
          graph.record_execution_verification(edge["edge_id"], matched)
          graph.record_skip_result(edge["edge_id"], matched,
                                   generation=guarded.generation)
          upsert(after, landed, visited=True)
          summary = str(action_dict)
          self._actions.append(action_dict)
          self._summaries.append(summary)
          recent_nodes.append(landed)
          walked += 1
          skip_record["n"] += 1
          skip_record["ok"] += int(matched)
          log("skip", step=step, edge_id=edge["edge_id"], matched=matched,
              latency_s=time.perf_counter() - started, action=action_dict,
              consecutive=consecutive_skips["n"] + walked, hop=walked,
              episode_accuracy=f'{skip_record["ok"]}/{skip_record["n"]}')
          if not matched:
            # The chain has left the map. Everything after this hop was
            # predicted from a state we are no longer in, so stop and let the
            # model look at the screen.
            break
          if skip_record["n"] >= 3 and skip_record["ok"] * 2 < skip_record["n"]:
            # Accuracy check inside the walk, not only before it. Checking
            # once at the top let a three-hop path run to completion on an
            # episode that was already mostly wrong, and since each hop counts
            # as a skip the path form tripled how fast a bad episode
            # accumulated wrong moves - ExpenseAddSingle took 10 skips with 4
            # wrong, MarkorCreateFolder 6 with 4, both under the 50% bar the
            # gate was supposed to enforce.
            break
        consecutive_skips["n"] += walked
        step_counter["i"] += 1
        guarded.commit_step()
        # One counted step for the whole run of hops: AndroidWorld charges per
        # agent.step(), so replaying a known stretch as a unit is what actually
        # shortens the trajectory. Every earlier form of skipping saved an
        # inference and left the step count untouched.
        summary = str(path[walked - 1]["action"].get("action_type"))
        return base_agent.AgentInteractionResult(False, {
            "response": f"[SKIPPED x{walked}]",
            "parsed_action": {"action": summary, "summary": summary},
            "action_dict": dict(path[walked - 1]["action"]), "summary": summary,
            "hints": [], "inference_skipped": True, "hops": walked,
        })

    # Spawn the explorer before inference so its startup (a Python
    # interpreter plus state-capture setup, ~5s) overlaps the blocking HTTP
    # call instead of adding to the step. No probe runs until await + START
    # below, so this does not touch the device while the model is looking at
    # it, and step i still cannot see anything it finds.
    # Probe only where the graph still has nothing to say about this screen.
    #
    # The objective is to reduce the entropy of the NEXT decision. If this
    # node already has outgoing edges, the graph already carries a
    # distribution over what happens here, so another probe buys
    # H(Z|C,O) ~= H(Z|C) - close to zero gain - while still paying a real
    # rollback risk. Only a screen with no recorded successors at all has
    # unbounded entropy for the graph, which is where a probe is worth its
    # cost. This is a two-valued structural fact ("does this node have any
    # outgoing edge"), not a tuned threshold.
    #
    # Measured on the same 12 tasks (2026-08-30): probing unconditionally at
    # 2 per step gave 217 probes, 74 rollback failures and 0/12, because it
    # doubled the skip count (21 -> 46) while dropping skip accuracy
    # (52% -> 37%), i.e. it injected ~29 wrong actions across 12 episodes.
    snapshot = guarded.snapshot(for_step=step)
    # A failed rollback leaves the device in a state nothing has verified.
    # Stacking a fresh probe on top of that compounds the damage instead of
    # measuring anything, so sit the next step out. The parallel runner has
    # this as device_dirty; it was missing here.
    # Gate on recoverability, learned per screen, not on whether the graph
    # has seen this node before.
    #
    # "Only probe unexplored nodes" was chosen because a node with outgoing
    # edges has little entropy left for the graph. That reasoning is sound
    # about information but says nothing about risk, and it removed
    # exploration from 82% of steps (measured 2026-08-31: of 145 real clicks,
    # 119 happened on a step where no probe ran at all). Prefix alignment
    # needs a guess to exist before it can be right, so alignment sat at 1.7%
    # for want of guesses rather than want of accuracy.
    #
    # Rollback failure is strongly type- and screen-specific (TAP_NAV at
    # depth 1 fails 7% of the time, TAP_NAV at depth 2 20%, EXPAND 23%), so
    # the explorer now probes anywhere except (activity, probe_type)
    # combinations that have already eaten a probe on this episode - see
    # blocked_recovery_contexts. A screen that cannot be rolled back stops
    # being probed after one attempt; screens that recover cleanly stay open.
    # Nothing to explore from yet, and the most to lose. In the opening steps
    # the graph is empty, so a probe cannot connect to anything already known,
    # while the disturbance it leaves lands on the screen the model is about
    # to read. Short tasks are the extreme case: NotesIsTodo and
    # SportsTrackerActivitiesCountForWeek take the baseline 4 and 2 steps -
    # look-and-answer questions with nothing worth probing - and both were
    # lost after a single early probe.
    #
    # The per-episode probe ceiling bounds the outlier case: ExpenseAddSingle
    # spent 30 probes where every other lost task spent 1-7, and lost.
    may_probe = (args.probes_per_step > 0
                 and not dirty_from_last_step["value"]
                 and step >= 2
                 and probe_total["n"] < args.probe_budget
                 and not stranded_once["value"]
                 and not _has_unsaved_input(before))
    pending_explorer = None
    explorer_config = {
        "trial_id": f"{args.task}-step{step}", "task": goal, "step_idx": step,
        "trace_path": str(root / "probe_trace.jsonl"),
        "filtered_path": str(root / "filtered_elements.jsonl"),
        "ranker": "InformationNeedRanker", "seed": args.seed + step,
        "information_need": current_need.to_dict(),
        "serial": "emulator-5554", "console_port": 5554,
        "a11y_local_port": args.a11y_socket_port, "restore_timeout_s": 30.0,
        "max_probes": args.probes_per_step, "min_probes": 0,
        "post_inference_grace_s": 0.0, "max_depth": args.max_depth,
        "max_exploration_time_s": 120.0,
        # Required for TRAJECTORY_REPLAY, the deepest and most reliable rung
        # of the recovery ladder: it reconstructs the committed state by
        # replaying the agent's own actions from HOME, and without them the
        # ladder simply ends. Omitting this list made every one of 107
        # rollback failures bottom out at FAILED, a 49% failure rate against
        # the parallel runner's 13% (2026-08-30).
        "committed_actions": _committed_actions(getattr(self, "_actions", [])),
        "blocked_recovery_contexts": sorted(blocked_recovery_contexts),
        # The step i-1 graph, serialised. The explorer runs in a separate
        # process and gets a copy of the previous generation only, so
        # "exploration uses the graph" cannot become "exploration uses its own
        # current-step findings" even by accident.
        "graph_snapshot": dataclasses.asdict(snapshot),
        "predictive_scorer": args.predictive_scorer,
        "recent_nodes": list(recent_nodes[-4:]),
        "scored_path": str(root / "scored_candidates.jsonl"),
        "blocked_element_identities": sorted(blocked_element_identities),
        # (activity, probe_type) -> the ladder rung that actually undid this
        # kind of transition here before. The explorer starts from that rung
        # instead of walking the ladder from the top, which is the difference
        # between remembering an inverse and rediscovering it.
        "known_inverse_levels": known_inverse_levels(),
    }
    if may_probe:
      try:
        pending_explorer = spawn_explorer(explorer_config)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        log("explore_spawn_failed", step=step, error=str(exc)[:300])

    # B9.2/B9.3: what the graph knows reaches the prompt only when it holds
    # facts worth the space. Read from the step i-1 snapshot, like the skip
    # decision, so this step's own probes cannot influence this step.
    if args.graph_context == "distill":
      # Already computed by the gate when it decided this step was not certain
      # enough to skip; recomputing it would just repeat the same deterministic
      # pass over the same snapshot.
      briefing_now["text"] = gate_mode["context"] if gate_mode["value"] == (
          gd.GRAPH_ENHANCED_INFERENCE) else distiller.distill(
              current_node_id=src_id, graph_snapshot=snapshot,
              information_need=current_need.to_dict(),
              taken_edges=taken_edge_ids, recent_nodes=tuple(recent_nodes[-4:]))
    elif args.graph_context == "briefing":
      briefing_now["text"] = snapshot.screen_briefing(src_id, taken_edge_ids)
    else:
      briefing_now["text"] = ""
    if briefing_now["text"]:
      log("graph_context", step=step, mode=args.graph_context,
          chars=len(briefing_now["text"]), tokens=len(briefing_now["text"].split()),
          text=briefing_now["text"])

    # (4) Inference on the clean pre-exploration state.
    inference_started = time.perf_counter()
    result = original_step(self, goal)
    inference_s = time.perf_counter() - inference_started

    # (3) Exploration from S_i, AFTER inference on purpose: in the parallel
    # design the model's screenshot is taken at window start, so it always
    # sees the clean state even while probes run. Exploring first serially
    # would show it a post-restore - possibly dirty - screen, a failure mode
    # the parallel design does not have. No action has executed yet, so the
    # probes still start from exactly S_i.
    exploration_s = 0.0
    if pending_explorer is not None:
      started = time.perf_counter()
      try:
        if result.done:
          # The episode ends here; probing would only risk the final state.
          stop_prepared_explorer(pending_explorer)
        else:
          outcome = run_serial_exploration(await_explorer_ready(pending_explorer))
          dirty_from_last_step["value"] = outcome.get("restore_status") != "RESTORED"
          probe_total["n"] += int(outcome.get("probes_completed", 0) or 0)
          ingest_probe_trace(explorer_config["trial_id"])
          log("explore", step=step, depth1_edges=len(depth1_edge_ids), **outcome)
          repair_after_exploration(self, before, outcome, step)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        log("explore_failed", step=step, error=str(exc)[:300])
      exploration_s = time.perf_counter() - started

    # (5) Record the authoritative transition, then make this step visible.
    raw = result.data.get("action_dict") or {}
    action_dict = dict(getattr(raw, "__dict__", raw) or {})
    try:
      after = capture.capture()
      dst_id = node_id_of(after)
      upsert(after, dst_id, visited=True)
      if dst_id != src_id:
        # Same system-UI filter the probe path uses. Without it the
        # authoritative edges recorded the status bar - "Do Not Disturb",
        # "Battery 100 percent.", "Phone signal full." - as their evidence,
        # and those labels then fed both the distilled prompt context and the
        # destination features the scorer reads (observed on
        # ClockStopWatchRunning, 2026-08-31: 6 of 8 labels were status bar).
        labels = tuple(dict.fromkeys(
            label for element in after.elements
            for label in (element.text, element.content_desc)
            if label and len(label) < 40 and _is_app_content(element, label)))[:8]
        observed = graph.add_speculative_transition(
            src_id, action_dict, dst_id, path_probability=1.0, confidence=0.0,
            expected_information_gain=0.0, risk_level="SAFE",
            exploration_cost=0.0, rollback_success=True, discovered_labels=labels)
        graph.record_execution_verification(observed.edge_id, True)
        taken_edge_ids.add(observed.edge_id)
        # Any explored edge out of this node whose action the model just took
        # counts as used too: the briefing should stop offering it.
        for eid in list(graph._outgoing.get(src_id, ())):  # pylint: disable=protected-access
          other = graph.edges[eid]
          if other.dst_node == dst_id:
            taken_edge_ids.add(eid)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      log("authoritative_failed", step=step, error=str(exc)[:300])

    if pending_explorer is None:
      # The flag means "the PREVIOUS step's exploration left the device in a
      # state it could not restore". No exploration ran this step, and a full
      # inference has since executed a real action from a freshly captured
      # screen, so whatever was dirty is now either resolved or part of the
      # committed trajectory. Leaving it latched turned one rollback failure
      # into exploration being off for the rest of the episode - measured on
      # ClockStopWatchRunning (2026-08-31): one failed probe at step 2, then
      # zero probes across steps 3-6.
      dirty_from_last_step["value"] = False

    responses = getattr(self, "_responses", None) or []
    if responses:
      last_model_output["text"] = str(responses[-1])
    consecutive_skips["n"] = 0
    recent_nodes.append(src_id)
    pending_prefix_edge_ids.extend(depth1_edge_ids)
    last_real_action.clear()
    last_real_action.update(action_dict)
    log("inference", step=step, inference_s=inference_s,
        exploration_s=exploration_s, action=action_dict, done=result.done)
    step_counter["i"] += 1
    guarded.commit_step()
    return result

  gelab_agent.GELABAgent.step = serial_step
  if args.inject_briefing:
    gelab_agent.build_gelab_messages = build_with_briefing
  sys.argv = [
      "run.py", "--suite_family=android_world", "--agent_name=gelab_agent",
      f"--tasks={args.task}", "--n_task_combinations=1", "--fixed_task_seed",
      f"--task_random_seed={args.seed}", f"--max_n_steps={args.max_steps}",
      "--console_port=5554", f"--output_path={root}",
  ]
  try:
    runpy.run_path(str(REPO_ROOT / "run.py"), run_name="__main__")
  finally:
    (root / "progressive_belief_graph.json").write_text(
        json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
  return 0


if __name__ == "__main__":
  main()
