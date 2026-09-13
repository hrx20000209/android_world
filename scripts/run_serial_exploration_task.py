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
import datetime
import json
import pathlib
import os
from pathlib import Path
import re
import statistics
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
  node_summaries: dict[str, str]
  node_elements: dict[str, tuple] = dataclasses.field(default_factory=dict)

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

    A settled screen - one whose map, across repeated visits, still shows a
    single viable continuation, and whose one edge the model itself took.
    This does not get ahead of the model, but it does not need to guess
    either: there is nothing left to decide.

    Replaying the model's past choice on any revisited screen used to qualify
    as a third source. It was removed on 2026-08-31 after separating skip
    outcomes by kind: the launch collapse was 80/80 correct while graph-based
    skips were 3/17, and every task where this arm cost steps against its
    control was a mis-skip of that third kind. A screen looking the same is
    not the situation being the same.
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
      # No recent-path exclusion here, unlike the promoted-lookahead branch
      # below. That guard exists because a speculative walk cannot tell it is
      # going in circles - every hop verifies on its own - and CameraTakeVideo
      # spent 13 skips that way. It does not apply to a transition the model
      # itself has taken repeatedly: returning to the recipe list after each
      # deletion IS the task, and excluding it blocked the mechanism at
      # exactly the moment it was finally able to fire (measured 2026-08-31,
      # RecipeDeleteMultipleRecipes: two nodes reached hits=3 with a single
      # outgoing action and were refused on this rule alone).
      settled = [
          self.edges[eid] for eid in self.outgoing.get(node_id, ())
          if self.edges[eid]["status"] in ("VERIFIED", "REUSABLE")
          and self.edges[eid]["dst_node"] not in (None, node_id)
          and self.edges[eid]["risk_level"] in {"SAFE", "LOW"}
      ]
      # Only the model's own choices count towards "this screen is decided".
      # A speculative edge says an action is possible here, never that it is
      # the one to take.
      settled = [e for e in settled if e.get("execution_hit_count", 0) >= 1]
      # Twice per episode, then hand it back. The cycle exemption above lets a
      # transition the model repeats be replayed even though it returns to a
      # screen just left, because in a repetitive task that IS the task. It
      # cannot tell that apart from the model being stuck, and on
      # SimpleSmsReplyMostRecent (2026-09-01) the model itself looped between
      # two screens, the graph learned the loop, and replayed it twelve times
      # with every single hop landing exactly where predicted while the
      # episode went nowhere. A cap bounds the damage to one extra lap without
      # blocking the three-iteration tasks this mechanism exists for, which
      # never need a third replay of the same edge.
      settled = [e for e in settled if e.get("skip_replays", 0) < 2]
      # The task has to have moved between the two executions this edge is
      # trusted for. Repeating an action from a screen looks the same whether
      # the task is iterating - delete a recipe, come back to a list that now
      # has one fewer - or the agent is stuck repeating something that is not
      # working. The graph tells them apart: iterating discovers screens
      # (a shorter list is a different screen), spinning does not.
      #
      # Measured over two full runs of the same code (2026-09-01): on the 39
      # tasks where the mechanism fired in both, this arm scored 6 and 6
      # against the control's 13, and every one of the 13 tasks that saw a
      # mis-skip was lost in both runs while the control won 6 of them. Even
      # the 16 tasks whose skips all landed exactly as predicted yielded only
      # 1-2 wins - landing where the graph said proves the dynamics, not that
      # the action was the right one to take now.
      settled = [e for e in settled
                 if len(self.node_visits) > e.get("nodes_at_last_execution", 0)]
      if (len(settled) == 1
          and settled[0].get("execution_hit_count", 0) >= 2
          and self.node_visits.get(node_id, 0) >= 2):
        # The model itself must have taken this transition at least twice
        # before it may be taken without the model. Separating 28 graph-based
        # skips by their edge's execution history on 2026-08-31: every one of
        # the 21 misses replayed an edge the model had executed exactly once,
        # while the hits came from edges it had executed twice or more. Once
        # is "this worked here"; twice is "this keeps being the answer here",
        # which is the property a replay actually depends on and the thing
        # that makes repetitive tasks - delete three recipes, add three
        # expenses - the case progressive memory can serve.
        #
        # Two executions of one action, and no other action ever chosen here.
        # That conjunction is what the repeated suffix of a repetitive task
        # looks like from inside the graph: RecipeDeleteMultipleRecipes runs
        # (1025,197) -> (652,602) -> (860,1295) three times over, differing
        # only in which recipe the cycle starts on, and ExpenseDeleteMultiple2
        # repeats (967,1658) -> (540,2221) the same way. Requiring a third
        # visit as well was tried and is redundant once edges merge by action:
        # it only delayed the same decision by one cycle.
        return settled[0]

    # Only prefix-aligned lookahead qualifies. Replaying the model's own past
    # choice on a revisited screen sounded like the safe source and measured
    # as the opposite: separating the two kinds of skip across 107 episodes on
    # 2026-08-31 gave the launch collapse 80 of 80 correct and graph-based
    # skips 3 of 17 (18%), and every task where this arm cost steps against
    # its control - MarkorCreateFolder +8, SaveCopyOfReceipt +8,
    # ExpenseDeleteDuplicates2 +7 - was a mis-skip of exactly this kind. The
    # earlier 88% figure was the two pooled, with the launch collapse
    # supplying the volume.
    #
    # A revisit means the screen looks the same, not that the situation is:
    # standing on the recipe list having deleted one recipe is not standing on
    # it having deleted none, and the action that was right the first time is
    # the reason the second visit exists. Prefix alignment carries evidence
    # about the current step - the explorer's guess matched what the model
    # then actually did - which is the property that was missing.
    candidates = [
        self.edges[edge_id] for edge_id in self.outgoing.get(node_id, ())
        if self.edges[edge_id]["status"] == "REUSABLE"
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
            "observed_destinations": tuple(edge.observed_destinations),
            "probe_count": edge.probe_count,
            "inference_alignment_count": edge.inference_alignment_count,
            "execution_hit_count": edge.execution_hit_count,
            "execution_miss_count": edge.execution_miss_count,
            "skip_attempt_count": edge.skip_attempt_count,
            "skip_success_count": edge.skip_success_count,
            "rollback_success_count": edge.rollback_success_count,
            "rollback_failure_count": edge.rollback_failure_count,
            "skip_replays": edge.skip_attempt_count,
            "nodes_at_last_execution": edge.nodes_at_last_execution,
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
        # What each screen IS, so a fact can name where a control leads instead
        # of listing labels the model can already read off the screenshot.
        node_summaries={n: node.semantic_summary
                        for n, node in self._graph.nodes.items()
                        if (node.semantic_summary or "").strip()},
        # The screen's whole named-control inventory, with how often each was
        # pressed and probed here. Consumers used to see only the 0.85 controls
        # per node the graph had an edge for.
        node_elements={n: node.ui_elements
                       for n, node in self._graph.nodes.items()
                       if node.ui_elements},
    )

  def refresh_before_exploration(self) -> None:
    """Re-capture the current generation's view, without advancing it.

    Prefix alignment is resolved at the top of a step: the probes taken during
    the previous window are compared against the screen the model actually
    landed on, and the children of a probe that guessed right are promoted to
    REUSABLE. That promotion is exactly the thing the skip gate exists to
    consume, and it is derived only from information this step already had -
    last window's probes plus an observed landing - so withholding it until
    the next step would delay a decision by one window for no reason.

    Safe because it is called before this step's explorer runs: there is
    nothing from the current window in the graph yet, so the generation number
    still describes the view honestly and the guard in snapshot() is
    unaffected.
    """
    self._snapshot = self._capture(self._generation)

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


def _settled_capture(capture, budget_s: float = 2.0):
  """Capture once the screen has stopped changing.

  Collapsing the launch into this step means the model reads the post-launch
  screen in the same step that launches it - so that screen has to actually
  be there. A fixed 0.4s wait was enough for a list to appear but not for a
  month grid: SimpleCalendarFirstEventAfterStartTime, which 4B_2 answers in
  four steps, ran to the 20-step cap with no probe, no injection and no skip
  but the launch collapse (2026-08-31), and its sibling
  SimpleCalendarAnyEventsOnDate answered wrong on step one.

  Polls the layout signature instead of waiting longer unconditionally, so an
  app that is ready immediately still costs one capture.
  """
  state = capture.capture()
  deadline = time.monotonic() + budget_s
  while time.monotonic() < deadline:
    time.sleep(0.25)
    try:
      following = capture.capture()
    except Exception:  # pylint: disable=broad-exception-caught
      return state
    if following.layout_sig == state.layout_sig:
      return following
    state = following
  return state


# Packages that put a system decision in front of the app rather than being
# an app the task could be in: role requests ("make this your default SMS
# app"), install prompts, and the "open with" chooser.
_SYSTEM_DIALOG_PACKAGES = frozenset({
    "com.google.android.permissioncontroller",
    "com.android.permissioncontroller",
    "com.android.packageinstaller",
    "com.google.android.packageinstaller",
    "android",
})


# Packages whose controls change device state that no rollback can undo. A
# toggle is not put back by Back, by relaunching the activity, or by replaying
# the trajectory - the recovery ladder reverses navigation, and a flipped
# switch is not navigation. Settings is also where the tasks that read those
# switches live, so a probe here does not merely risk the episode, it can
# invalidate the very thing being checked. Structural, not a task list: this
# says which surfaces are unprobeable, the same kind of statement as
# "TAP_MENU rolls back 84% of the time".
_UNPROBEABLE_PACKAGES = frozenset({
    "com.android.settings",
    "com.google.android.settings",
    "com.android.systemui",
})

# The launcher is not an app the task works in, so nothing learned there
# transfers to anything - and it is the surface the model navigates by
# swiping. A SCROLL probe moves the app drawer's page; the probe reports
# INVERSE success because layout_signature is deliberately blind to scroll
# offset, while the model is left hunting for an icon that moved.
#
# Measured 2026-09-02: 14 of 65 probes in a full run happened on the
# launcher. On SystemBrightnessMin and SystemBrightnessMaxVerify the model
# swiped the home screen for all 15 steps and never opened Settings, while
# the control opened it at step 2 and finished in 6-8.
def _unprobeable(package: str) -> bool:
  return package in _UNPROBEABLE_PACKAGES or "launcher" in package.casefold()


def _decision_constraints(goal: str, variant: str = "full") -> str:
  """Guardrails against the two failure modes that dominate this agent.

  Ported verbatim in substance from explorer_agent_gelab_light, which reaches
  58/116 against this base agent's 53 on the same suite. That gap turned out
  to be prompt, not exploration: the light agent injects these constraints on
  every step while the plain gelab_agent prompt has none, so every comparison
  between the two designs was also a comparison between two prompts.

  Both clauses name failures we measured independently. Premature completion:
  SimpleCalendarAnyEventsOnDate declared task_complete on step 0 with no
  action taken. Repeating a done action: MarkorCreateFolder looped between "+"
  and the filename field for ten steps. The last clause is the one that
  matters for this design specifically - it tells the model that the injected
  graph context loses to the screenshot whenever the two disagree, which is
  exactly the right precedence for a memory that can be stale.

  The two goal-triggered clauses are matched on verbs, not on task names.
  """
  lowered = (goal or "").lower()
  capture_note = ""
  if re.search(r"\b(record (?:an? )?(?:audio|video|clip)|take (?:a |the )?"
               r"(?:photo|picture|video)|capture (?:a |the )?(?:photo|picture|video))\b",
               lowered):
    capture_note = ("- For recording/photo/video capture tasks, once the current screen shows "
                    "the clip/photo/video was captured, saved, or appears in the "
                    "media/recording list, choose COMPLETE instead of repeatedly "
                    "starting/stopping/capturing again.\n")
  destructive_note = ""
  if re.search(r"\b(delete|remove|trash|discard|clear all|erase)\b", lowered):
    destructive_note = ("- For delete/remove tasks involving files, expenses, recipes, notes, "
                        "events, tasks, contacts, playlists, or list rows, do not COMPLETE "
                        "immediately after a dialog confirmation; first verify on the current "
                        "screen that each exact target item is absent from the relevant "
                        "list/search/folder.\n")
  # The completion gate is the single clause measured to cost tasks whose end
  # state is not legible on screen: with it on, seven System* toggles/sliders
  # burned all 15 steps; with it off they finished in 4-7 (2026-09-02, v24 vs
  # v29). It is separated out rather than deleted because it is also the clause
  # that stops premature ANSWERs, so which way it nets out is a measurement.
  completion_gate = (
      "- Do not choose COMPLETE, ANSWER, or task_complete unless the current screen "
      "visibly proves the requested final state.\n"
      if variant == "full" else "")
  return (
      "Decision constraints:\n"
      f"{completion_gate}"
      "- For pure operation tasks, if the current screen already visibly proves the "
      "requested action is done, choose COMPLETE rather than repeating the same "
      "click/type action.\n"
      f"{capture_note}"
      "- For file delete or file move tasks, do not complete immediately after a "
      "destructive dialog or one list observation; first verify the folder/path and "
      "the exact source absence or destination presence on the current screen.\n"
      f"{destructive_note}"
      "- If exploration context conflicts with the current screen, ignore exploration "
      "context and act only on the current screen.\n"
  )


def _clean_label(text: str) -> str:
  """A label as it would read on screen: whitespace collapsed, junk dropped.

  Accessibility text arrives padded and sometimes unformatted - the graph held
  "      Share,       Information" and "Recording: %s", a resource format
  string the app never renders (2026-09-04).
  """
  cleaned = " ".join(str(text or "").split())
  if not cleaned or "%s" in cleaned or "%d" in cleaned or "%1$" in cleaned:
    return ""
  return cleaned


def _replay_label(edge: Mapping[str, Any]) -> str:
  """Name the replayed control the way the screen names it."""
  action = edge.get("action") or {}
  control = str(action.get("control_key") or "")
  if control:
    # control_key is "<resource-id>|<text>|<desc>|<class>"-ish; the readable
    # part is whichever of text/desc is present.
    parts = [p for p in control.split("|") if p and not p.startswith("(")]
    for part in parts[1:]:
      if part and not part.startswith("android.") and "." not in part:
        return _clean_label(part)
    # Falling back to the widget class produced lines like "ImageButton ->
    # a screen showing: Share, Rename": true, and useless, because the model
    # cannot find "ImageButton" on a screenshot. A control with no name is
    # better left out than named by its type.
    return ""
  kind = str(action.get("action_type") or "the control").lower()
  if action.get("text"):
    return f'{kind} "{action["text"]}"'
  return kind


def screen_history_context(snapshot, node_id: str, taken_here: set[str],
                           max_items: int = 4, max_tokens: int = 120,
                           include_available: bool = True) -> str:
  """What this screen has already done, stated as history rather than advice.

  The distilled context speaks about 9 times in a 1000-step run because it
  demands a labelled positive fact, and an authoritative edge only carries
  labels when the landing screen showed something new. Yet the graph knows
  something far more often than that, and the cheapest useful thing it knows
  is which controls on this screen have already been pressed and where they
  went: repeated_action_rate was 6-10% across every measured run, so the model
  is re-deriving that by hand.

  Phrased as observation, never as suggestion. An earlier briefing said
  "Tapping X opens Y" and the model read it as an instruction, looping between
  "+" and the filename field for ten steps (MarkorCreateFolder, 2026-08-30) -
  which is why what is already done is named first and separately.

  Reads only the step i-1 snapshot, so it carries nothing from this window's
  exploration or inference.
  """
  used: list[str] = []
  unused: list[str] = []
  for edge_id in snapshot.outgoing.get(node_id, ()):
    edge = snapshot.edges[edge_id]
    if edge.get("status") == "INVALID":
      continue
    name = _replay_label(edge)
    if not name or name in ("click", "the control") or name.startswith("android."):
      # A control the model cannot find from its name is not worth a line.
      continue
    labels = [c for c in (_clean_label(x)
                          for x in (edge.get("discovered_labels") or ())) if c][:3]
    dst = edge.get("dst_node")
    if labels:
      where = "a screen showing: " + ", ".join(labels)
    elif dst and dst != node_id and dst in snapshot.node_visits:
      where = "a screen this episode has already been on"
    elif dst and dst == node_id:
      where = "this same screen"
    else:
      continue
    entry = f"{name} -> {where}"
    (used if edge_id in taken_here else unused).append(entry)
  parts: list[str] = []
  if used:
    parts.append("Already used on this screen this visit: "
                 + "; ".join(used[:max_items]) + ".")
  if unused and include_available:
    # Measured 2026-09-04: listing the exits the model has NOT taken reads as
    # a menu, and it takes them. Injection rose to 17.6% of steps and the
    # repeated-action rate rose with it, 19.2% -> 22.9%, while success fell
    # 2-4 - the same failure the old screen_briefing recorded in 2026-08-30.
    # Only the half that names what is already done is an observation.
    parts.append("Also known to lead somewhere from this screen: "
                 + "; ".join(unused[:max_items]) + ".")
  if not parts:
    return ""
  text = "[Screen memory]\n" + "\n".join(parts)
  words = text.split()
  if len(words) > max_tokens:
    text = " ".join(words[:max_tokens])
  return text


def _git_head() -> str:
  """Which commit this run was made from; uncommitted work is flagged."""
  import subprocess
  try:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, timeout=5).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True, timeout=10).stdout.strip()
  except Exception:
    return "unknown"
  return f"{head}{'+dirty' if dirty else ''}"


def _screen_sentence(model_output: str) -> str:
  """The model's own one-line description of the screen it is looking at.

  A node currently carries a hash and nothing readable: `semantic_summary` was
  never populated on any of 1175 nodes, and `salient_ui_labels` only by
  cross-task seeding. The description is already paid for - at step i the model
  looks at S_i and its <THINK> opens by saying what it sees ("I see a 'Confirm
  Delete' dialog on the screen", "I am currently viewing the root directory of
  the 'sdk_gphone64_arm64' storage") - so reading it costs no model call.

  Attached after inference, which is also what keeps the temporal rule intact:
  exploration for step i has already run by then and cannot see this.
  """
  text = str(model_output or "")
  think = re.findall(r"<THINK>(.*?)</THINK>", text, flags=re.S | re.I)
  body = think[0] if think else " ".join(re.findall(r"explain:([^\t\n]*)", text))
  body = re.sub(r"\s+", " ", body).strip()
  if not body:
    return ""
  match = re.match(r"(.{10,140}?[.!?])\s", body + " ")
  sentence = (match.group(1) if match else body[:140]).strip()
  # A <THINK> does not always open by saying what is on screen; 35% of the
  # sentences it yields are progress or intent ("I have successfully navigated
  # to the correct date", "I need to turn on Bluetooth"), measured over 438
  # descriptions. Stored as a screen description those mislead every later task
  # that lands there - "New Event -> I need to create an event two weeks from
  # today" says nothing about the screen. Only observational openings are kept.
  if not _OBSERVATION.match(sentence):
    return ""
  return sentence


# "I see the file list", "The main settings page", "A confirmation dialog".
_OBSERVATION = re.compile(
    r"^(i see\b|i am (on|looking|currently|in|viewing)\b|"
    r"the (screen|current|app|main|note|file|list|page|dialog)\b|"
    r"a |an )", re.I)


def _actionable_labels(state, limit: int = 10) -> tuple[str, ...]:
  """What a person would say is on this screen, in on-screen order.

  Only actionable controls, only ones that carry a name. Two screens of the
  same Activity that differ solely in list content share a layout_signature by
  design; this is the record of what actually differed, which is what a later
  audit of that merge needs.
  """
  # Only what the app under test is showing. The a11y tree of a screen with
  # the keyboard up also carries the IME's own controls, and they are what a
  # describer sees: measured 2026-09-11 over a 453-screen store, 48 screens
  # across 8 different apps were described as "A screen with More features,
  # Sticker Keyboard, GIF Keyboard..." - one of those sentences stood on 23
  # different screens - and 32% of remembered transitions pointed at a screen
  # so described, making the fact simply wrong. Structural, not a phrase list:
  # an element belonging to another package is the keyboard, the status bar,
  # or a system overlay, none of which are what this screen is.
  app_package = str(getattr(getattr(state, "activity", None), "component", "")
                    or "").split("/", 1)[0]
  out: list[str] = []
  seen: set[str] = set()
  for element in getattr(state, "elements", ()) or ():
    if not (element.clickable or element.scrollable or element.checked is not None):
      continue
    if app_package and getattr(element, "package", "") and (
        element.package != app_package):
      continue
    label = (element.text or element.content_desc or "").strip()
    if not label or len(label) > 40:
      continue
    key = label.casefold()
    if key in seen:
      continue
    seen.add(key)
    out.append(label)
    if len(out) >= limit:
      break
  return tuple(out)


def _canonical_scroll(action: Mapping[str, Any]) -> str:
  """Every way of expressing a vertical scroll, on one key.

  AndroidWorld's `scroll` and `swipe` are inverses of each other, and its own
  source says so (`actuation.py:172`, "Inverse of scroll."): `scroll down`
  ends at `y_min`, dragging the finger UP to reveal content below, while
  `swipe down` starts at `0` and ends at `height//2`, dragging the finger DOWN
  to reveal content above. A probe's SCROLL always drags 75% -> 25% of the
  element, finger up, so it is a `scroll down`.

  Comparing the raw `direction` string across the two verbs would therefore
  call two OPPOSITE gestures identical - measured 2026-09-09 on
  FilesDeleteFile step 4, where a SCROLL probe and the model's `swipe down`
  landed on the same node by taking opposite actions.

  Horizontal is deliberately not normalised: the two verbs agree there (both
  end up dragging the same way), and a probe never scrolls horizontally, so
  there is nothing to compare against.
  """
  kind = str(action.get("action_type") or "").lower()
  direction = str(action.get("direction") or "").lower()
  if direction in ("down", "up"):
    if kind == "scroll":
      return "scroll|forward" if direction == "down" else "scroll|back"
    if kind == "swipe":
      return "scroll|back" if direction == "down" else "scroll|forward"
    return ""
  if direction:
    return ""
  # A probe edge names no direction; its probe_type does. Non-inverse SCROLL
  # is the only scroll a probe records - the inverse one is the rollback.
  if str(action.get("probe_type") or "").upper() == "SCROLL":
    return "scroll|forward"
  return ""


def _canonical_control(action: Mapping[str, Any]) -> str:
  """The control an action names, for comparing two actions as the same choice.

  control_key when the action carries one, otherwise the action's own shape -
  text for typing, direction for a swipe, app for a launch. Never coordinates:
  the model predicts them in a normalised space and they wobble between
  identical decisions, which is why edge identity stopped using them.
  """
  if not action:
    return ""
  # Scrolls first: a probe records one as a control_key naming the scrollable
  # container, the model records one as a direction, and the two verbs invert
  # each other. Falling through to control_key would compare a container name
  # against a direction and never match, whichever gesture was actually taken.
  scroll = _canonical_scroll(action)
  if scroll:
    return scroll
  control = str(action.get("control_key") or "")
  kind = str(action.get("action_type") or "").lower()
  if control:
    return f"{kind}|{control}"
  for field in ("text", "direction", "app_name", "keycode"):
    value = action.get(field)
    if value not in (None, ""):
      return f"{kind}|{field}={value}"
  return ""


def _control_key_at(state, action: dict[str, Any]) -> str:
  """Which control the agent's tap landed on, as a stable key.

  The agent reports a coordinate; the graph needs to know which control that
  was, because the coordinate wobbles between visits while the control does
  not. Resolves against the screen captured immediately before the action, and
  picks the SMALLEST element containing the point - containers enclose their
  children, and the child is the thing that was pressed.
  """
  # Imported here rather than at module scope: this file defers every
  # parallel_exploration import into main() to keep import order controlled.
  from android_world.parallel_exploration.belief_graph import control_key_from_identity
  x, y = action.get("x"), action.get("y")
  if x is None or y is None:
    return ""
  best, best_area = None, None
  for element in getattr(state, "elements", ()) or ():
    left, top, right, bottom = element.bounds
    if not (left <= x <= right and top <= y <= bottom):
      continue
    area = max(0, right - left) * max(0, bottom - top)
    if area <= 0:
      continue
    if best_area is None or area < best_area:
      best, best_area = element, area
  if best is None:
    return ""
  key = control_key_from_identity(best.identity)
  resource_id, text, content_desc = best.resource_id, best.text, best.content_desc
  if not (resource_id or text or content_desc):
    # Nothing names this control but its widget class, and a bare class name
    # is not an identity - two RelativeLayouts on one screen would merge into
    # one edge. Anchor it on the element's OWN centre, coarsely: the element's
    # bounds are stable across visits of the same screen (that is what the
    # layout signature asserts), while the tap coordinate is not, so this
    # still absorbs the model's wobble without colliding across the screen.
    left, top, right, bottom = best.bounds
    cell = 96
    key = f"{key}@{int((left + right) / 2) // cell},{int((top + bottom) / 2) // cell}"
  return key


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


def _backfill_descriptions(graph, app_store, port: int, log=None) -> int:
  """Name every screen the episode left blank, after the episode is over.

  Measured on desc2 (2026-09-10): of the nodes with no description, 16 of 18
  sampled could be described perfectly well from the labels already stored on
  them. They were blank not because the describer failed but because the write
  point is never reached for them - the description is attached after an
  inference on the screen the model reasoned about, and a node that is only
  ever landed on, skipped over, or reached during bootstrap is never that
  screen. A seeded node is never even visited.

  So it runs here instead: after run.py has returned, on no critical path at
  all, where a 1.15 s call per unnamed screen costs the episode nothing. The
  descriptions go into app memory as well as the graph, which is the point -
  a screen this task only passed through is a screen the next task may have to
  reason on, and it will now arrive named.
  """
  from android_world.parallel_exploration import semantic_service
  filled = 0
  for node in graph.nodes.values():
    if (node.semantic_summary or "").strip():
      continue
    labels = list(node.salient_ui_labels or ())
    if not labels:
      # Never stood on, so no capture of its own - but whatever probe or step
      # reached it recorded what it saw there.
      labels = _incoming_labels(graph, node.node_id)
    if not labels:
      continue
    sentence = semantic_service.query_describe(
        port, "", node.activity, labels, timeout_s=20.0)
    if not sentence:
      continue
    node.semantic_summary = sentence
    filled += 1
    if app_store is not None:
      memory = app_store.get(node.package or node.activity.split("/", 1)[0])
      if memory is not None:
        memory.observe_screen(node.activity, node.layout_signature,
                              labels, description=sentence)
  if log is not None and filled:
    log("description_backfill", step=None, filled=filled,
        nodes=len(graph.nodes))
  return filled


def _incoming_labels(graph, node_id: str) -> list[str]:
  """What the transitions that reach this node saw there."""
  out: list[str] = []
  seen: set[str] = set()
  for edge in graph.edges.values():
    if edge.dst_node != node_id:
      continue
    for label in edge.discovered_labels or ():
      text = str(label).strip()
      if text and text.casefold() not in seen:
        seen.add(text.casefold())
        out.append(text)
  return out[:12]


def _clip(sentence: str, words: int = 13) -> str:
  """At most `words` words, cut at a clause boundary rather than mid-phrase.

  A hard cap leaves dangling fragments - "I see the 'New name' dialog on the
  screen, which is the final" - and the identifying half of a reasoning
  sentence is almost always the first clause anyway.
  """
  parts = sentence.split()
  if len(parts) <= words:
    return sentence
  head = parts[:words]
  for i in range(len(head) - 1, 2, -1):
    if head[i].endswith(","):
      return " ".join(head[:i + 1]).rstrip(",") + "."
  return " ".join(head)


def _store_routes_context(memory, layout_signature: str,
                          budget: int = 64) -> str:
  """What earlier tasks found leading out of THIS screen, from the store.

  Every previous injection form read the episode's own graph, which is the one
  place that has nothing when it matters: 68% of steps stand on a node with no
  outgoing edge, and the funnel then died on a utility threshold an OBSERVED
  edge cannot mathematically clear. The persistent store is the opposite -
  the current screen is in it on 86% of steps and has a known transition on
  55% - because it was filled by every task that ran before this one.

  Stated destination-first. The control names are weak (38% of remembered
  transitions carry only a resource id and are dropped upstream) while the
  destinations carry real sentences (92% of stored screens have a
  description), so the description is the part worth the tokens.
  """
  if memory is None or not layout_signature:
    return ""
  routes = memory.known_routes(layout_signature)
  if not routes:
    return ""
  lines = ["[Known] Earlier tasks on this screen found:"]
  for name, description in routes:
    short = " ".join(description.split()[:12]).rstrip(".")
    line = f'- "{name}" led to {short}.'
    if len(" ".join(lines + [line]).split()) > budget:
      break
    lines.append(line)
  return "\n".join(lines) if len(lines) > 1 else ""


def _loop_context(snapshot, node_id: str, min_repeats: int = 3) -> str:
  """A control this episode keeps pressing here while nothing new is found.

  12% of the episodes that run out of steps (11 of 94) contain an edge executed
  four or more times and waste 7.9 steps each. They are NOT self-loops - zero
  edges in any run have dst == src, because a screen that "looks unchanged"
  still re-hashes (a clock, a list redraw). FilesMoveFile is the real shape:
  6e6c38 -> 23ce90 -> 2bf0cc -> 1a909d, each edge taken four times, the model
  walking one cycle four times over.

  So the test is repetition plus stagnation, which is the distinction the skip
  gate already draws: iterating discovers screens - delete a recipe, land on a
  list with one fewer - while spinning does not. `nodes_at_last_execution`
  records how large the graph was when the edge was last taken.

  The model cannot see this for itself: its history keeps eight entries and the
  cycle is longer. Stated as a record, never as an instruction; the one block
  that told the model what to do cost 12 tasks.
  """
  if snapshot is None or not node_id:
    return ""
  seen_now = len(getattr(snapshot, "node_visits", {}) or {})
  worst = None
  for eid in (getattr(snapshot, "outgoing", {}) or {}).get(node_id, ()):
    edge = (getattr(snapshot, "edges", {}) or {}).get(eid) or {}
    hits = int(edge.get("execution_hit_count") or 0)
    if hits < min_repeats:
      continue
    if seen_now > int(edge.get("nodes_at_last_execution") or 0):
      continue          # the task found something new since; it is iterating
    if worst is None or hits > worst[1]:
      worst = (edge, hits)
  if worst is None:
    return ""
  label = _control_display_label((worst[0].get("action") or {}).get("control_key") or "")
  return (f'[Loop] "{label}" has been used here {worst[1]} times and no new '
          "screen has been reached since.")


def _elements_context(snapshot, node_id: str, summaries, budget: int = 64) -> str:
  """What this screen's controls have done, from the node's own inventory.

  Every earlier injection form spoke in terms of edges, and a node carries
  0.85 of those against 9 to 14 named controls - so 37% of visited screens had
  nothing to say. The inventory turns the whole screen into the surface: a
  control that has been pressed here carries its count, and one that leads
  somewhere named carries the destination.

  Controls never pressed here are deliberately left out. They are what
  exploration needs (coverage), not what the model needs: it can read them off
  the screenshot, and listing nine of them would spend the budget on nothing.
  """
  if snapshot is None or not node_id:
    return ""
  inventory = (getattr(snapshot, "node_elements", {}) or {}).get(node_id) or ()
  used = [e for e in inventory if int(e.get("clicks") or 0) > 0]
  if not used:
    return ""
  used.sort(key=lambda e: -int(e.get("clicks") or 0))
  lines = ["[Screen] What has been pressed here before:"]
  for entry in used:
    label = str(entry.get("label") or "").strip()
    if not label:
      continue
    times = int(entry.get("clicks") or 0)
    dst = _destination_of(snapshot, node_id, str(entry.get("control_key") or ""),
                          summaries)
    line = f"- {label} ({times}x)" + (f" led to {dst}" if dst else "")
    if len(" ".join(lines + [line]).split()) > budget:
      break
    lines.append(line)
  return "\n".join(lines) if len(lines) > 1 else ""


def _destination_of(snapshot, node_id: str, control_key: str, summaries) -> str:
  """The described screen this control led to from here, or ""."""
  if not control_key:
    return ""
  for eid in (getattr(snapshot, "outgoing", {}) or {}).get(node_id, ()):
    edge = (getattr(snapshot, "edges", {}) or {}).get(eid) or {}
    if ((edge.get("action") or {}).get("control_key") or "") != control_key:
      continue
    dst = edge.get("dst_node") or ""
    text = str((summaries or {}).get(dst) or "").strip()
    if text:
      return " ".join(text.split()[:11]).rstrip(".")
  return ""


def _ui_inventory(state, graph, node_id: str, limit: int = 14):
  """Every named actionable control on this screen, with its history HERE.

  The graph used to know only the controls it had an edge for: 0.85 per node,
  and 37% of visited nodes had none, against 9 to 14 named controls on a
  screen. Every consumer read that 0.85 - which is why ranking candidates by
  embedding measured 40% top-1 against a 38% random baseline, and why a
  coverage ranker could not tell "probed one of ten here" from "probed all
  ten".

  Same package rule as the labels: an element belonging to another package is
  the keyboard or the status bar, not this screen.
  """
  from android_world.parallel_exploration.belief_graph import control_key_from_identity
  app_package = str(getattr(getattr(state, "activity", None), "component", "")
                    or "").split("/", 1)[0]
  clicks: dict[str, int] = {}
  probes: dict[str, int] = {}
  for edge in graph.edges.values():
    if edge.src_node != node_id:
      continue
    key = (edge.action or {}).get("control_key") or ""
    if not key:
      continue
    clicks[key] = clicks.get(key, 0) + int(edge.execution_hit_count or 0)
    probes[key] = probes.get(key, 0) + int(edge.probe_count or 0)
  out = []
  seen: set[str] = set()
  for element in getattr(state, "elements", ()) or ():
    if not (element.clickable or element.scrollable or element.checked is not None):
      continue
    if app_package and getattr(element, "package", "") and (
        element.package != app_package):
      continue
    label = (element.text or element.content_desc or "").strip()
    if not label or len(label) > 40:
      continue
    identity = getattr(element, "identity", "")
    key = control_key_from_identity(identity) if identity else ""
    dedupe = (key or label).casefold()
    if dedupe in seen:
      continue
    seen.add(dedupe)
    out.append({"label": label[:40], "control_key": key,
                "role": (element.class_name or "").rsplit(".", 1)[-1],
                "clicks": clicks.get(key, 0), "probes": probes.get(key, 0)})
    if len(out) >= limit:
      break
  return tuple(out)


def _control_display_label(control_key: str) -> str:
  """What to call this control when telling the model it was pressed."""
  parts = (control_key or "").split("|")
  for index in (1, 2):
    if len(parts) > index and parts[index].strip():
      return " ".join(parts[index].split())[:40]
  tail = parts[0].rsplit("/", 1)[-1] if parts else ""
  return re.sub(r"[_\-.]+", " ", tail).strip()[:40] or "a control"


def _episode_decided(graph, node_id: str, min_exec: int) -> tuple[str, str] | None:
  """(control_key, dst_node) when THIS episode has settled this screen.

  The screen counts as settled when the model has executed exactly one control
  on it and nothing else. Measured 2026-09-12 over 116 tasks: with one prior
  execution the next visit repeats that control **83%** of the time (96/115),
  with two it is 93% (42/45) - but the one-execution gate has 115 chances
  against 45, which is why it is the default here.

  The skip gate cannot use one execution: it replaces the inference, so a miss
  costs a step and every task that saw a mis-skip was lost in both runs of a
  2026-09-01 pair. Prefill executes and then still runs the inference, so a
  miss costs one extra action.

  Reads only what earlier steps executed - never this step's inference - so it
  sits on the same side of the generation guard as every other graph read.
  """
  if graph is None or not node_id:
    return None
  executed = [graph.edges[eid] for eid in graph.outgoing.get(node_id, ())
              if (graph.edges[eid].get("execution_hit_count") or 0) > 0]
  if len(executed) != 1:
    return None
  edge = executed[0]
  if (edge.get("execution_hit_count") or 0) < min_exec:
    return None
  control = (edge.get("action") or {}).get("control_key") or ""
  dst = edge.get("dst_node") or ""
  if not control or dst == node_id:
    return None
  return control, dst


def _element_for_control(state, control_key: str, by_resource_id: bool = False):
  """The element on this screen whose control_key matches, or None.

  A control key is `resource_id|text|content_desc|class`, so a button whose
  label carries a count or a date fails to match itself across screens - 43
  prefill attempts were refused as "control absent" over 50 tasks (2026-09-13)
  against 29 that fired. `by_resource_id` falls back to the resource id alone,
  but only when exactly one element on the screen carries it: ids repeat inside
  list rows, and picking an arbitrary row is worse than not prefilling.
  """
  from android_world.parallel_exploration.belief_graph import control_key_from_identity
  elements = list(getattr(state, "elements", ()) or ())
  for element in elements:
    identity = getattr(element, "identity", "")
    if identity and control_key_from_identity(identity) == control_key:
      return element
  if not by_resource_id:
    return None
  wanted = control_key.split("|")[0]
  if not wanted:
    return None
  same = [element for element in elements
          if (getattr(element, "resource_id", "") or "") == wanted]
  return same[0] if len(same) == 1 else None


def _click_on(element):
  """A click at the element's centre."""
  from android_world.env import json_action
  left, top, right, bottom = element.bounds
  return json_action.JSONAction(action_type=json_action.CLICK,
                                x=int((left + right) / 2),
                                y=int((top + bottom) / 2))


def _risk_of(element) -> str:
  """SAFE/LOW only for controls a Back press can undo.

  Same structural test the probe filter uses: a role of "button" acts and may
  not be undoable, a checked control toggles state, and anything without a
  name cannot be reasoned about at all.
  """
  cls = (getattr(element, "class_name", "") or "").lower()
  if getattr(element, "checked", None) is not None:
    return "UNKNOWN"
  if "button" in cls and "imagebutton" not in cls:
    return "UNKNOWN"
  if not ((element.text or "").strip() or (element.content_desc or "").strip()):
    return "UNKNOWN"
  return "SAFE"


def _prefill_risk_of(element, mode: str) -> str:
  """Risk gate for a prefilled control, which is not the probe gate.

  `_risk_of` exists for probing: a probe is a speculative press taken purely
  to learn, so it may only touch controls a Back press can undo. Applied to
  prefill it throws away 59% of everything a prefill could ever fire on -
  measured over the 419 executed edges of a 115-task run (2026-09-13): 42%
  rejected for having no text or content-desc (they have a resource id, which
  is what the control key matches on), 17% for being an android.widget.Button.

  A prefilled control is a different object. It is not speculative: the store
  only holds controls a real task pressed, `observe_execution` drops any
  transition that left the app, the recorded destination is checked after the
  hop, and a mismatch is rolled back. "loose" keeps the one structural
  rejection that still applies - a control carrying checked state toggles
  something rather than navigating - and lets the destination check do the
  rest.
  """
  if mode == "probe":
    return _risk_of(element)
  if getattr(element, "checked", None) is not None:
    return "UNKNOWN"
  return "SAFE"


def _walked_path_lines(snapshot, current_node_id: str,
                       limit: int = 4) -> list[str]:
  """Descriptions of the screens this episode stood on, most recent last."""
  if snapshot is None or not current_node_id:
    return []
  visits = getattr(snapshot, "node_visits", {}) or {}
  summaries = getattr(snapshot, "node_summaries", {}) or {}
  seen: set[str] = set()
  walked: list[str] = []
  for node_id, count in visits.items():
    if node_id == current_node_id or not count:
      continue
    text = str(summaries.get(node_id) or "").strip()
    if text and text.casefold() not in seen:
      seen.add(text.casefold())
      walked.append(text)
  return walked[-limit:]


def _walked_path_context(snapshot, current_node_id: str,
                         limit: int = 4) -> str:
  """The screens this episode has already stood on, named, most recent last.

  Every earlier injection form said "control X leads to Y", which needs the
  current node to have outgoing edges - and measured on desc2 (2026-09-10),
  68% of steps stand on a node that has none, so 76 steps produced 0
  injections with the funnel dying before the utility threshold was even
  consulted. This form needs no edges at all, only node descriptions, so it
  has material on every step where the episode has been anywhere before.

  It is also the one thing the model provably cannot recover for itself: its
  own history window keeps 8 entries and 48% of episodes exceed it, the
  entries are actions rather than screen identities, and a screen visited
  three times appears three times there and once here. Descriptions written
  by earlier tasks count too, so a screen this episode never described can
  still be named.

  Read from the step i-1 snapshot like every other graph read, so this step's
  own probes cannot reach it.
  """
  walked = _walked_path_lines(snapshot, current_node_id, limit)
  if not walked:
    return ""
  lines = "\n".join("- " + _clip(t) for t in walked)
  # Stated as record, never as instruction. The one prompt block that told the
  # model what to do with what it was given cost 12 tasks (v29 vs the ported
  # decision-constraints prompt), so this says only what was visited.
  return f"[Memory] Screens already visited on this task:\n{lines}"


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--task", required=True)
  parser.add_argument("--max_steps", type=int, default=20,
                      help="per-task step cap. 0 hands the decision back to "
                           "AndroidWorld, whose own budget is 10 x the task's "
                           "complexity (suite_utils.py:525). A flat 15 was used "
                           "for every arm up to 2026-09-13 and it is what made "
                           "43 tasks look impossible: 72%% of those have an "
                           "official budget above 15 and 42%% have 30 or more "
                           "(OsmAndTrack gets 120, MarkorMergeNotes 78, nine "
                           "tasks 50-78). Those 43 were not unwinnable, they "
                           "were starved.")
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
  parser.add_argument("--downsample", type=float, default=1.0,
                      help="send the model a screenshot divided by this factor. "
                           "The reference arm this is measured against runs on "
                           "GELABResizeAgent, whose step() the plain agent's "
                           "monkeypatch cannot reach - so any value above 1.0 "
                           "switches which class is patched as well")
  parser.add_argument("--align_requires_action_match", action="store_true",
                      help="Promote a prefix only when the probe took the same "
                           "control the model then chose, not merely when it "
                           "landed on the same screen. Node identity is loose "
                           "on purpose, so landing-only alignment accepts a "
                           "different action that happened to reach the same "
                           "screen.")
  parser.add_argument("--no_align_promotes_current_step",
                      dest="align_promotes_current_step", action="store_false",
                      help="Make a prefix-alignment promotion wait a step "
                           "before the skip gate can use it.")
  parser.add_argument("--depth2_needs_known_inverse", action="store_true",
                      help="Descend to depth 2 only from a root whose inverse "
                           "is known here (20%% of depth-2 tap probes could not "
                           "be rolled back, against 7%% at depth 1).")
  parser.add_argument("--min_node_visits", type=int, default=2,
                      help="How many times this episode must have stood on a "
                           "screen before it may be probed. 2 restricts "
                           "exploration to revisits, which is where it was "
                           "measured to pay - but that measurement predates "
                           "the node-id and timing fixes that made a probe "
                           "able to pay at all. 1 allows first visits.")
  parser.add_argument("--explore_interval", type=int, default=1,
                      help="Minimum number of steps between exploration "
                           "rounds. 1 explores whenever eligible; larger "
                           "values spread a smaller amount of exploration "
                           "across more of the episode.")
  parser.add_argument("--cross_task_revisit", action="store_true",
                      help="Treat a screen this app met in an earlier task as a "
                           "revisit for the purpose of allowing exploration. "
                           "Exploration only; the skip gate stays episode-local.")
  parser.add_argument("--no_deep_rollback_blocks", dest="deep_rollback_blocks",
                      action="store_false",
                      help="Stop treating a probe that recovered via a deep "
                           "recovery rung as proof that this (activity, probe "
                           "type) can never be probed again. The working rung "
                           "is remembered either way.")
  parser.add_argument("--constraints_variant", default="full",
                      choices=("full", "no_completion_gate"),
                      help="'no_completion_gate' drops the one clause that forbids "
                           "COMPLETE without on-screen proof; every other clause "
                           "stays. Ignored when --no_decision_constraints is set.")
  parser.add_argument("--no_decision_constraints", dest="decision_constraints",
                      action="store_false",
                      help="drop the per-step decision constraints. They are a "
                           "base-agent prompt improvement, not part of the "
                           "exploration design, and the reference arm this is "
                           "measured against already has them")
  parser.add_argument("--app_memory", default=None,
                      help="directory holding per-package structural memory "
                           "kept across tasks. Carries what the APP is - "
                           "transitions, labels, which ladder rung undoes a "
                           "probe here, which probes proved unrecoverable - "
                           "and deliberately not what the MODEL decided, "
                           "which is task-specific. Unset = current behaviour")
  parser.add_argument("--no_bootstrap", dest="bootstrap", action="store_false",
                      help="disable the launch collapse. Separate from "
                           "--enable_skip: collapsing open_app into the first "
                           "decision is fixed by the task text and measured at "
                           "96/96, while replaying a remembered edge is a "
                           "prediction - an ablation of one must not silently "
                           "disable the other")
  parser.add_argument("--dump_graph_steps", action="store_true",
                      help="write the belief graph after every step, for "
                           "reconstructing how it grew")
  parser.add_argument("--allow_icon_only_probes", action="store_true",
                      help="PART G: let the explorer probe clickable controls "
                           "that carry no accessible label. These are 55%% of "
                           "everything the safety filter removes and much of "
                           "what the model actually clicks")
  parser.add_argument("--skip_semantic_check", action="store_true",
                      help="before taking a structurally-licensed skip, require the remembered action's label to match the current information need above a cosine floor. Every existing skip condition is structural (revisited screen, one action ever chosen, executed twice) and none asks whether that action is what the task needs now; measured skip hit rate is 57-85%%. Blocks nothing it cannot judge - an unnameable control or an unreachable encoder leaves the skip exactly as it was.")
  parser.add_argument("--entropy_over", choices=("all", "executed"),
                      default="all",
                      help="which outgoing edges decision entropy counts. 'all' is the historical behaviour and lets probe and memory-seeded edges raise H, which closes the skip gate: mean finite H ordered every arm's skip count on 2026-09-10 (semantic 0.224->14, probes-off 0.246->13, probes-on 0.295->7, warm memory 0.324->2). 'executed' counts only edges the model has actually taken, which is what the skip gate's claim is about.")
  parser.add_argument("--node_summary", choices=("off", "reasoning", "model"),
                      default="reasoning",
                      help="fill each node's semantic_summary with the model's own one-line description of that screen, taken from the <THINK> it wrote while standing on it. Free - no extra model call - and attached after inference, so exploration for that step cannot see it. 'off' restores the historical behaviour, where the field was never populated on any of 1175 nodes. 'model' additionally asks a 0.6B running beside the encoder to describe the 42.5%% of steps that emit a bare tool call with no reasoning at all.")
  parser.add_argument("--semantic_port", type=int, default=8766,
                      help="port of the 22M encoder service used by "
                      "--exploration_policy semantic; exploration falls back to "
                      "its cheap ranker if the service is unreachable")
  parser.add_argument("--visited_discount", type=float, default=0.0,
                      help="weight on the walked-path term of the semantic "
                      "exploration ranker: a candidate whose text reads like a "
                      "screen this episode has already stood on is discounted "
                      "by this much. Reads the node descriptions written by "
                      "earlier steps and by earlier tasks, never this step's "
                      "inference. 0.0 (default) is the measured labels-only "
                      "ranker, unchanged.")
  parser.add_argument("--min_fact_utility", type=float, default=0.15,
                      help="a graph fact is only eligible for injection "
                      "above this utility score. 0.15 is the historical "
                      "value and discards 93%% of the steps that have a "
                      "fact available (measured 2026-09-09).")
  parser.add_argument("--exploration_budget_s", default="120.0",
                      help="seconds a single exploration round may spend "
                      "before it stops starting new probes; 'auto' uses "
                      "the median inference latency observed so far in "
                      "this episode, which is the window exploration has "
                      "to stay inside to remain free. Default 120.0 is "
                      "the historical value, i.e. effectively no budget.")
  parser.add_argument("--exploration_policy",
                      choices=("information_need", "graph_matrix", "random",
                               "coverage", "semantic", "llm_choice"),
                      default="graph_matrix",
                      help="PART G: how probe candidates are ranked. 'random' "
                           "is the control the exploration claim needs: same "
                           "safety filter, same budget, same rollback, but the "
                           "candidate is drawn uniformly instead of ranked. "
                           "'coverage' drops the prediction objective the other "
                           "three share - random ranks as well as any of them, "
                           "so prediction buys nothing - and ranks by what the "
                           "graph does not know: controls with no outgoing edge "
                           "yet, preferring ones that can be named at all.")
  parser.add_argument("--graph_reasoning",
                      choices=("off", "briefing", "distill", "skip_only",
                               "distill_and_skip"),
                      default=None,
                      help="PART G: how the graph reaches reasoning. Sets "
                           "--graph_context and --enable_skip together; "
                           "overrides both when given")
  for group in ("exact_history", "contextual_history", "information_need",
                "cost", "recovery_history"):
    parser.add_argument(f"--disable_{group}", action="store_true",
                        help=f"PART G: drop the {group} feature group from the "
                             "candidate matrix")
  parser.add_argument("--loop_notice", action="store_true",
                      help="append a one-line record when a control has been "
                           "used on this screen >= --loop_repeats times without "
                           "the screen changing. 12%% of the episodes that run "
                           "out of steps contain such an edge and waste 7.9 "
                           "steps each; the model cannot see it because its "
                           "history keeps eight entries and the loop is longer")
  parser.add_argument("--loop_repeats", type=int, default=3,
                      help="how many uses of one control on one screen, with no "
                           "change, count as a loop")
  parser.add_argument("--store_prefill", action="store_true",
                      help="before each inference, if the persistent store has "
                           "seen exactly one control executed on this screen by "
                           ">=2 different tasks, execute it and then run the "
                           "step's inference on the resulting screen. Measured "
                           "87%% accurate (39/45 over 116 tasks); safe at that "
                           "accuracy only because it falls through rather than "
                           "replacing the inference, so a miss costs one extra "
                           "action and not a step of the 15-step budget.")
  parser.add_argument("--prefill_by_resource_id", action="store_true",
                      help="when the full control key does not match any "
                           "element, fall back to the resource id alone, but "
                           "only if exactly one element on the screen carries "
                           "it (97%% of ids are unique on their screen). A "
                           "label that carries a count or a date otherwise "
                           "stops a control from matching itself.")
  parser.add_argument("--prefill_sources", default="store",
                      help="comma-separated: 'store' (what other tasks pressed "
                           "here, via uniqueness and goal retrieval) and/or "
                           "'episode' (what this episode already pressed here). "
                           "Measured live over 50 tasks (2026-09-13) the store "
                           "sources land 26/29 = 90%% and episode 9/30 = 30%%, "
                           "and episode supplied half the hops - the arm lost 8 "
                           "successes against no-graph on precisely the tasks "
                           "where prefill fired. Default is store only.")
  parser.add_argument("--prefill_retrieval", default="both",
                      choices=("uniq", "goal", "both", "off"),
                      help="how the store picks the control to prefill. "
                           "'uniq' is the screen-uniqueness gate (leave-one-out "
                           "over 419 executed steps, 2026-09-13: covers 17%% of "
                           "them at 90%%); 'goal' retrieves what the most "
                           "goal-similar earlier tasks pressed here (42%% at "
                           "84%%); 'both' runs uniqueness first and the goal "
                           "vote on the screens it refuses (44%% at 84%%). For "
                           "contrast a plain screen-majority vote is 45%% at "
                           "66%%, which is 65 wrong hops against this 30.")
  parser.add_argument("--prefill_k", type=int, default=3,
                      help="neighbours in the goal vote (k=1 gives 76%%, k=3 "
                           "84%%, k=5 80%%)")
  parser.add_argument("--prefill_sim_thr", type=float, default=0.5,
                      help="goal similarity the nearest neighbour must clear "
                           "(0.0 gives 76%%, 0.5 gives 84%%)")
  parser.add_argument("--prefill_margin", type=float, default=0.6,
                      help="weighted-vote margin the winning control must hold "
                           "over the runner-up. A near-tie is exactly the case "
                           "a screen vote gets wrong: 0.0 gives 76%%, 0.3 82%%, "
                           "0.6 84%%, 1.0 86%% at falling coverage.")
  parser.add_argument("--prefill_after_bootstrap", action="store_true",
                      help="also prefill immediately after the step-0 app "
                           "launch. The app home screen is the best-covered "
                           "screen in the store (47%% of first in-app screens "
                           "answered at 75%% cold-start, against 34%%/76%% "
                           "averaged over all screens) and no other call site "
                           "reaches it: without this, step 0 spends its "
                           "inference there and the first navigation waits for "
                           "step 1")
  parser.add_argument("--prefill_risk", default="loose",
                      choices=("probe", "loose"),
                      help="which risk gate a prefilled control must clear. "
                           "'probe' reuses the probe filter, which rejects 59%% "
                           "of all executed edges (42%% unnamed, 17%% Button); "
                           "'loose' rejects only controls carrying checked "
                           "state, on the grounds that a prefill is not "
                           "speculative - the store only holds controls a real "
                           "task pressed inside this app, and the hop is "
                           "verified against the recorded destination")
  parser.add_argument("--prefill_rollback", action="store_true",
                      help="when a prefilled hop does not land where the store "
                           "recorded, press Back and drop the summary, so a "
                           "miss costs wall clock instead of handing the model "
                           "a screen it never navigated to")
  parser.add_argument("--prefill_min_exec", type=int, default=1,
                      help="how many times THIS episode must have executed the "
                           "screen's only control before it may be prefilled. "
                           "1 gives 83%% over 115 chances, 2 gives 93%% over 45 "
                           "- the looser gate wins because a miss costs an "
                           "action, not a step")
  parser.add_argument("--prefill_min_tasks", type=int, default=2,
                      help="how many different tasks must have executed the "
                           "control before it may be prefilled (2 gives 87%%, "
                           "1 gives 71%%)")
  parser.add_argument("--max_prefill", type=int, default=3,
                      help="most prefill hops in one step. 59%% of decided "
                           "screens lead to another decided screen, which takes "
                           "the collapsible total from 39 to 64 steps over 116 "
                           "tasks; each hop is verified against the recorded "
                           "destination before the next fires, and a screen is "
                           "prefilled at most once per episode because a "
                           "learned two-screen loop was once replayed twelve "
                           "times")
  parser.add_argument("--graph_context",
                      choices=("off", "briefing", "distill", "history",
                               "history_done", "onclick", "path",
                               "distill_path", "llm", "store",
                               "store_path", "elements", "elements_store"),
                      default="distill",
                      help="how graph knowledge reaches the prompt (PART G ablation): "
                           "store = what earlier TASKS found leading out of "
                           "this screen, read from the persistent app memory "
                           "rather than the episode graph - the screen is in "
                           "the store on 86%% of steps and has a known "
                           "transition on 55%%, against 32%% for the episode "
                           "graph; store_path = that plus the walked path; "
                           "llm = the same walked path, compressed to "
                           "one line by the 0.6B beside the encoder; "
                           "path = the screens this episode already stood "
                           "on, named, most recent last - the only form that "
                           "needs no outgoing edges, and 68%% of steps stand "
                           "on a node that has none; distill_path = both, "
                           "off = never, briefing = the old screen_briefing, "
                           "distill = GraphDistiller facts under a token budget, "
                           "history = what this screen has already done plus "
                           "its known exits, history_done = only the half that "
                           "names what is already done (the exits read as a "
                           "menu and the model takes them), onclick = "
                           "AutoDroid's form (MobiCom'24): one line per control "
                           "the graph knows a destination for, with no "
                           "block-level positive-fact gate - that gate fires 10 "
                           "times per 1010 steps while 31%% of steps have a "
                           "candidate fact, so the block form discards 97%% of "
                           "what the graph knows")
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
  if args.graph_reasoning is not None:
    # One flag for the reasoning-side ablation grid, expanded into the two
    # switches the runner already reads, so the older flags keep working and
    # every combination stays expressible.
    args.graph_context = {
        "off": "off", "briefing": "briefing", "distill": "distill",
        "skip_only": "off", "distill_and_skip": "distill",
    }[args.graph_reasoning]
    args.enable_skip = args.graph_reasoning in ("skip_only", "distill_and_skip")
  args.predictive_scorer = args.exploration_policy == "graph_matrix"
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
  from android_world.parallel_exploration import belief_graph as _bg
  _bg.ENTROPY_OVER_EXECUTED_ONLY = args.entropy_over == "executed"
  from android_world.parallel_exploration.information import parse_reasoning_prior
  from android_world.parallel_exploration import graph_distiller as gd
  gd.SKIP_SEMANTIC_PORT = args.semantic_port if args.skip_semantic_check else 0
  from android_world.parallel_exploration.live_probe import _is_app_content
  from android_world.parallel_exploration.belief_graph import control_key_from_identity
  from android_world.parallel_exploration import state_graph_information as sgi
  from android_world.parallel_exploration.live_probe import element_identity_from_dict
  from android_world.parallel_exploration.live_probe import await_explorer_ready
  from android_world.parallel_exploration.live_probe import spawn_explorer
  from android_world.parallel_exploration.live_probe import run_serial_exploration
  from android_world.parallel_exploration.state import create_optimized_state_capture

  root = args.output.resolve()
  root.mkdir(parents=True, exist_ok=True)
  # Every run records the configuration it ran under. Eleven full 116-task runs
  # were made without this, and the question "was this arm configured like
  # v48?" then had no answer on disk - one success-rate difference (2026-09-09,
  # -4.8 against six references) stayed permanently unattributable because
  # config, build and device state could not be told apart after the fact.
  (root / "run_args.json").write_text(json.dumps(
      {"argv": sys.argv[1:],
       "args": {k: (str(v) if isinstance(v, pathlib.Path) else v)
                for k, v in sorted(vars(args).items())},
       "git_head": _git_head(),
       "started_at": datetime.datetime.now().isoformat(timespec="seconds")},
      indent=2, ensure_ascii=False))
  trace_path = root / "probe_trace.jsonl"
  graph = ProgressiveBeliefGraph(args.task)
  from android_world.parallel_exploration.app_memory import AppMemoryStore
  from android_world.parallel_exploration.app_memory import _is_app_destination
  app_store = AppMemoryStore(args.app_memory) if args.app_memory else None
  seeded_nodes: set[str] = set()

  def memory_for(activity: str):
    """Per-package memory for whatever app this screen belongs to."""
    if app_store is None:
      return None
    package = (activity or "").split("/", 1)[0]
    return app_store.get(package) if package else None
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
  scoring_config = sgi.ScoringConfig(
      use_exact_history=not args.disable_exact_history,
      use_contextual_history=not args.disable_contextual_history,
      use_information_need=not args.disable_information_need,
      use_cost=not args.disable_cost,
      use_recovery_history=not args.disable_recovery_history,
  )
  # One number decides whether the graph speaks at all. Measured
  # 2026-09-09 over 1010 steps: 207 steps had candidate facts, 15
  # survived this threshold, 10 were injected - it alone discards 93%
  # of what the graph knows, against 5 for the positive-fact gate and
  # 55% for facts that cannot be named at all.
  distiller = gd.GraphDistiller(dataclasses.replace(
      gd.DEFAULT_DISTILLER, min_fact_utility=args.min_fact_utility))
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

  def _remembered_description(signature) -> str:
    memory = memory_for(signature.activity.component)
    if memory is None:
      return ""
    return memory.recall_description(signature.layout_sig)

  def upsert(signature, node_id: str, visited: bool) -> None:
    existing = graph.nodes.get(node_id)
    graph.upsert_node(GraphNode(
        node_id=node_id, activity=signature.activity.component,
        package=signature.activity.component.split("/", 1)[0],
        visual_signature=signature.phash,
        structural_signature=signature.struct_sig.digest,
        layout_signature=signature.layout_sig,
        status=NodeStatus.COMMITTED,
        salient_ui_labels=_actionable_labels(signature),
        ui_elements=_ui_inventory(signature, graph, node_id),
        # A description already written for this node survives a revisit: the
        # first time the model described this screen is as good as the second,
        # and overwriting would lose it on steps that emit no reasoning.
        # Falling back to app memory is what makes a description readable
        # *before* this step's inference - the episode graph resets every
        # task, the store does not, and exploration runs first.
        semantic_summary=((existing.semantic_summary if existing else "")
                          or _remembered_description(signature)),
    ), visited=visited)
    memory = memory_for(signature.activity.component)
    if memory is not None:
      # Same package rule as _actionable_labels: what the store remembers about
      # a screen must be what the app showed, not what the IME did.
      app_package = signature.activity.component.split("/", 1)[0]
      memory.observe_screen(
          signature.activity.component, signature.layout_sig,
          [label for element in signature.elements
           if not (getattr(element, "package", "") and
                   element.package != app_package)
           for label in (element.text, element.content_desc)
           if label and len(label) < 40 and _is_app_content(element, label)][:12])
      if node_id not in seeded_nodes:
        seeded_nodes.add(node_id)
        n = memory.seed_graph(graph, node_id, signature.activity.component,
                              signature.layout_sig)
        if n:
          log("app_memory_seed", step=None, node=node_id[:8],
              activity=signature.activity.component, edges=n)

  prefilled_screens: set[str] = set()
  # Goal-to-goal similarity for the store's retrieval, memoised per screen's
  # candidate list because the same screen is asked about repeatedly and the
  # encoder round trip is the only cost the retrieval has.
  _goal_sim_cache: dict[tuple[str, tuple[str, ...]], list[float] | None] = {}

  def _goal_similarity(goal: str, candidates: list[str]) -> list[float] | None:
    """Scores in [0,1], or None when the encoder cannot answer."""
    if not candidates:
      return []
    key = (goal, tuple(candidates))
    if key in _goal_sim_cache:
      return _goal_sim_cache[key]
    scores = None
    try:
      from android_world.parallel_exploration import semantic_service
      scores = semantic_service.query(
          args.semantic_port, goal, list(candidates), timeout_s=3.0)
    except Exception:  # pylint: disable=broad-exception-caught
      scores = None
    _goal_sim_cache[key] = scores
    return scores

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
  # Layout signatures this app was already standing on in some EARLIER task,
  # snapshotted before this episode writes anything back. Exploration is
  # otherwise limited to screens revisited inside one episode, which measured
  # here blocks 62.9% of all steps - and that limit was set when memory was
  # episode-local, so "a screen seen once" really did mean "nothing to gain".
  # With per-package memory it no longer does: 40.6% of the 478 screen nodes
  # in a full run recur across tasks, and the apps recur harder still
  # (Calendar 17 tasks, Markor 15, Broccoli 11), so what a probe learns on a
  # first visit here is read by the next task that opens the same screen.
  # This gates EXPLORATION only. The skip gate keeps reading episode-local
  # visit and execution counts, so cross-task knowledge still cannot license
  # replaying an action without the model.
  preknown_layouts: set[str] = set()
  # Step index of the most recent exploration round, for --explore_interval.
  last_explored_step = {"n": -10**6}
  # Observed inference latencies this episode. Exploration is only free
  # while it fits inside the window inference is already occupying, and
  # that window is the one thing the runner can measure directly. The
  # budget defaulted to 120s, i.e. no budget at all: a coverage round
  # with 8 probes ran 5.45s against a 3.42s median window (2026-09-09).
  inference_latencies: list[float] = []

  def seed_safety_from_memory() -> None:
    """Start the episode already knowing what has hurt before.

    The blocklist is otherwise relearned from scratch every task, and
    relearning costs one unrecoverable probe each time. Across three runs
    (2026-09-01) nine (activity, probe_type) combinations failed to roll back
    in more than one task - fourteen accidents that this seeding skips
    outright, on episodes whose success rate drops from 89% to 67% the moment
    one of them lands.
    """
    if app_store is None:
      return
    for path in sorted(pathlib.Path(app_store.root).glob("*.json")) if pathlib.Path(app_store.root).exists() else []:
      try:
        package = json.loads(path.read_text(encoding="utf-8")).get("package")
      except (OSError, ValueError):
        continue
      if package:
        app_store.get(package)
    for memory in app_store._loaded.values():  # pylint: disable=protected-access
      blocked_recovery_contexts.update(memory.blocked_contexts)
      blocked_element_identities.update(memory.blocked_elements)
      preknown_layouts.update(memory.screens)
    if blocked_recovery_contexts or blocked_element_identities:
      log("app_memory_load", step=None,
          seeded_blocked_contexts=len(blocked_recovery_contexts),
          seeded_blocked_elements=len(blocked_element_identities),
          store=app_store.stats())
  # Set the first time an exploration round cannot restore the screen it
  # started from. After that this episode stops probing entirely - see
  # may_probe. Measured over 35 tasks on 2026-08-31, bucketed by what
  # exploration did:
  #
  #   no probing at all              16 win / 2 loss   89%
  #   probed, every round restored    8 win / 1 loss   89%
  #   probed, some round did not      6 win / 3 loss   67%
  #
  # Exploration that comes back is free; exploration that does not costs a
  # fifth of the success rate. The episode cannot know in advance which kind
  # it will get, but it knows the moment it has had the second kind, and that
  # is the point to stop - the per-(activity, probe_type) blocklist only rules
  # out the exact combination that just failed, so episodes kept failing again
  # through a different control on a different screen.
  unrecovered_once = {"value": False}
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
    """This episode's evidence first, then what earlier tasks learned."""
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
    if app_store is not None:
      for memory in app_store._loaded.values():  # pylint: disable=protected-access
        for key, level in memory.inverse_levels.items():
          out.setdefault(key, level)
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
      if row.get("recovery_ok") and row.get("recovery_level"):
        src_act0 = str(((row.get("graph") or {}).get("src") or {}).get("activity") or "")
        m0 = memory_for(src_act0)
        if m0 is not None and src_act0:
          m0.observe_rollback(src_act0, str(row.get("probe_type") or ""),
                              ok=True, level=str(row.get("recovery_level")))
      if deep_recovery and row.get("recovery_ok") and args.deep_rollback_blocks:
        # A rollback that needed a deep rung but DID come back is expensive,
        # not impossible - and the rung that worked is already remembered in
        # known_inverse_levels, so the next probe here starts from it instead
        # of walking the ladder again. Blocking the context outright made a
        # successful recovery indistinguishable from a failed one: the same
        # probe was recorded ok=True with its level a few lines above and
        # ok=False here. Measured 2026-09-02 on 103 probe rows: 6 of the 24
        # blocking events came from probes that recovered, and because a block
        # is keyed on (activity, probe_type) and is carried across tasks by app
        # memory, those 24 events removed ~886 of 1842 candidates - the single
        # largest reason the candidate set misses what the model goes on to
        # click. Genuine RESTORE_FAILED still blocks on one demonstration.
        src_act = str(((row.get("graph") or {}).get("src") or {}).get("activity") or "")
        if src_act:
          blocked_recovery_contexts.add(f"{src_act}|{row.get('probe_type')}")
          m = memory_for(src_act)
          if m is not None:
            m.observe_rollback(src_act, str(row.get("probe_type") or ""), ok=False)
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
          m = memory_for(src_act)
          if m is not None:
            m.observe_rollback(src_act, str(row.get("probe_type") or ""), ok=False)
        identity = element_identity_from_dict(row.get("element", {}))
        blocked_element_identities.add(identity)
        m = memory_for(src_act)
        if m is not None:
          m.observe_escape(identity)
        continue

      if row.get("notes") in {"RESTORE_FAILED", "NESTED_RESTORE_FAILED"}:
        # Rollback could not be verified, so the recorded destination is not
        # trustworthy evidence about anything - and this screen just showed it
        # cannot take this kind of probe.
        src_act = str(((row.get("graph") or {}).get("src") or {}).get("activity") or "")
        if src_act:
          blocked_recovery_contexts.add(f"{src_act}|{row.get('probe_type')}")
          m = memory_for(src_act)
          if m is not None:
            m.observe_rollback(src_act, str(row.get("probe_type") or ""), ok=False)
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
      probe_control = control_key_from_identity(
          str(action_with_kind.get("element_identity", "")))
      if probe_control:
        action_with_kind["control_key"] = probe_control
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
      memory = memory_for(src["activity"])
      if memory is not None:
        memory.observe_probe(
            activity=src["activity"],
            layout_signature=src.get("layout_signature", ""),
            control_key=str(action_with_kind.get("control_key", "")),
            probe_type=str(row.get("probe_type") or ""),
            action={k: v for k, v in action_with_kind.items()
                    if k in {"action_type", "x", "y", "direction"}},
            dst_activity=dst["activity"],
            dst_layout_signature=dst.get("layout_signature", ""),
            discovered_labels=(row.get("discovered") or {}).get("new_texts", [])[:8],
            rollback_ok=bool(row.get("recovery_ok")),
            inverse_level=str(row.get("recovery_level") or ""))
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

  def episode_app_package(before) -> str:
    """The package this episode is supposed to be working in.

    Deliberately not "whatever was on screen before exploration ran". Two
    attempts at this fix failed on that definition (2026-09-01, SimpleSmsSend):
    the episode was already stranded in the role-request dialog, so repairing
    back to the dialog scored as success; then it was stranded on the
    launcher, so returning to the launcher scored as success. Neither is a
    screen a task wants to be on.

    Taken instead from the graph: the package of the screen this episode has
    stood on most, excluding the launcher and system dialogs. That is the app
    the task has actually been working in, whatever the current screen says.
    """
    counts: dict[str, int] = {}
    for node in graph.nodes.values():
      package = node.package or ""
      if not package or package in _SYSTEM_DIALOG_PACKAGES or "nexuslauncher" in package:
        continue
      counts[package] = counts.get(package, 0) + max(1, node.visit_count)
    if counts:
      return max(counts, key=counts.get)
    # Nothing in the graph yet: fall back to the screen in front of us. Not
    # returning at all was tried and is worse - it skipped the repair
    # entirely 18 times in one episode (2026-09-01), leaving the device
    # wherever exploration had put it.
    return before.activity.component.split("/", 1)[0]

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
    if outcome.get("restore_status") != "RESTORED":
      unrecovered_once["value"] = True
    stranded = capture.capture()
    expected_pkg = episode_app_package(before)
    actual_pkg = stranded.activity.component.split("/", 1)[0]
    if actual_pkg == expected_pkg:
      # Still in the right app. The exploration round may not have restored
      # the exact screen, and dirty_from_last_step already suppresses the next
      # round for that; relaunching here would additionally throw away
      # whatever in-app progress is on screen. Measured 2026-08-31: repairing
      # on any unrestored round rather than only on an escape pushed the
      # successful-task mean from 7.2 to 10.4 steps while three of the four
      # "strandings" were the app's own main activity.
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
        # Relaunching an app can put the system in front of it. Launching an
        # SMS app fresh makes Android ask to be made the default handler
        # (permissioncontroller/RequestRoleActivity); a chooser or an install
        # prompt behaves the same. The repair used to notice only that the
        # package was wrong, record dirty, and hand the dialog to the model -
        # which then spent the rest of the episode pressing the same button.
        # All five SMS tasks were lost that way in v21 (2026-09-01), while the
        # control, never going HOME mid-task, never saw the dialog at all.
        # Dismissing it is a framework-level fact about system dialogs, not a
        # rule about any app.
        # Checked independently of expected_pkg. That expectation is taken
        # from whatever was on screen before exploration ran, and if the
        # episode was already sitting in the role-request dialog then
        # "returning" to it scores as a successful repair - which is exactly
        # what happened on the first attempt at this fix (2026-09-01,
        # SimpleSmsSend: repaired to permissioncontroller, ok=True, task
        # lost). A system dialog is never a screen the task wants to be on.
        for _ in range(2):
          package = after_repair.activity.component.split("/", 1)[0]
          if package not in _SYSTEM_DIALOG_PACKAGES:
            break
          agent._execute_action(json_action.JSONAction(action_type="navigate_back"), {})
          time.sleep(0.35)
          after_repair = capture.capture()
          via = "relaunch+dismiss"
      ok = after_repair.activity.component.split("/", 1)[0] == expected_pkg
      dirty_from_last_step["value"] = not ok
      # Leaving the app ends probing for this episode, exactly as a failed
      # restore does. The latch was keyed only on restore_status, and the
      # explorer reports RESTORED whenever its own ladder believes it
      # succeeded - which it can, while the device is nonetheless sitting in
      # another app. On 2026-09-01 that let five SMS tasks bounce in and out
      # of the app 10 to 17 times each (76 repairs across the run) with the
      # latch never arming, and all five were lost.
      unrecovered_once["value"] = True

      log("repair", step=step, ok=ok, app=target, via=via,
          stranded=stranded.activity.component,
          landed=after_repair.activity.component)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      log("repair_failed", step=step, error=str(exc)[:300])

  def dump_node_screenshot(agent, node_id: str) -> None:
    """One screenshot per screen the graph knows about, the first time it is seen.

    Keyed by node rather than by step on purpose: the node IS the screen, and
    a reader looking at a node wants to see what the agent was looking at.
    Repeated visits reuse the first capture, which is also the honest thing to
    show - the node exists precisely because those visits looked the same.
    """
    if not args.dump_graph_steps:
      return
    folder = root / "screens"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{node_id[:12]}.png"
    if target.exists():
      return
    try:
      from PIL import Image
      pixels = agent.env.get_state(wait_to_stabilize=False).pixels
      image = Image.fromarray(pixels)
      image.thumbnail((300, 300))
      image.save(target)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      log("screenshot_failed", step=None, node=node_id[:12], error=str(exc)[:200])

  def dump_graph_snapshot(step: int) -> None:
    """One graph file per step, for reconstructing how the belief grew.

    The end-of-episode dump shows what was learned but not when, and the
    when is the whole claim: a node has to be revisited and an action
    repeated before a skip becomes available, so the counts at the moment of
    the decision are what a reader needs to see.
    """
    if not args.dump_graph_steps:
      return
    folder = root / "graph_steps"
    folder.mkdir(parents=True, exist_ok=True)
    snap = guarded.snapshot(for_step=step + 1)
    (folder / f"step{step:02d}.json").write_text(json.dumps({
        "step": step,
        "generation": snap.generation,
        "nodes": [
            {"node_id": nid, "activity": node.activity,
             "visit_count": snap.node_visits.get(nid, 0),
             "decision_entropy": snap.node_entropy.get(nid),
             "labels": list(node.salient_ui_labels)[:6]}
            for nid, node in graph.nodes.items()],
        "edges": list(snap.edges.values()),
    }, ensure_ascii=False, default=str, indent=1), encoding="utf-8")

  seed_safety_from_memory()

  if args.downsample > 1.0:
    from android_world.agents import gelab_agent_resize as _resize
    original_step = _resize.GELABResizeAgent.step
  else:
    original_step = gelab_agent.GELABAgent.step
  original_build = gelab_agent.build_gelab_messages
  step_counter = {"i": 0}
  dirty_from_last_step = {"value": False}
  taken_edge_ids: set[str] = set()
  # Edges used during the current uninterrupted stay on one screen.
  taken_here: set[str] = set()
  last_context_node = {"id": None}
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
    extra = "\n\n".join(x for x in (briefing_now["text"],
                                     _decision_constraints(
                                         goal_text, args.constraints_variant)
                                     if args.decision_constraints else "") if x)
    if not extra:
      return messages
    for message in messages:
      content = message.get("content")
      if not isinstance(content, list):
        continue
      for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
          item["text"] = f"{item['text']}\n\n{extra}"
          if args.dump_graph_steps:
            folder = root / "prompts"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"step{step_counter['i']:02d}.txt").write_text(
                item["text"], encoding="utf-8")
          return messages
    return messages

  def serial_step(self, goal: str):
    episode_goal["text"] = goal
    step = step_counter["i"]
    before = capture.capture()
    src_id = node_id_of(before)
    upsert(before, src_id, visited=True)
    dump_node_screenshot(self, src_id)

    # Prefix alignment: did last step's probe guess land where the model's
    # real action then landed? If so the explorer read the intent correctly
    # at that point, and the deeper hops it explored from there earn the
    # right to be reused - which is the only way a skip can get AHEAD of the
    # authoritative path instead of merely replaying it. Judged by the
    # landing state rather than by comparing coordinates, so it holds for
    # every action type.
    aligned_now = 0
    aligned_by_action_now = 0
    aligned_unresolved_now = 0
    for edge_id in list(pending_prefix_edge_ids):
      edge = graph.edges.get(edge_id)
      if (edge is None or edge.dst_node is None or edge.dst_node == edge.src_node
          or edge.rollback_success is not True or edge.dst_node != src_id):
        continue
      # Two different claims, logged apart because they have very different
      # strength. by_node says the probe LANDED where the model then landed;
      # node identity is deliberately loose, so a different action that
      # happens to reach the same screen satisfies it. by_action says the
      # probe took the SAME control the model then chose, compared on
      # control_key - the key edge identity already uses - rather than on
      # coordinates, which wobble.
      probe_key = _canonical_control(edge.action)
      model_key = _canonical_control(last_real_action)
      by_action = bool(probe_key) and probe_key == model_key
      # A blank model_key means the tap resolved to no named control, not
      # that the two actions differ; counting it as a mismatch would report a
      # measurement gap as a finding. Logged apart, with the raw points, so
      # the three outcomes stay separable offline.
      log("prefix_candidate", step=step, edge_id=edge_id,
          aligned_by_node=True, aligned_by_action=by_action,
          model_key_resolved=bool(model_key),
          probe_canonical_action=probe_key, model_canonical_action=model_key,
          probe_xy=[edge.action.get("x"), edge.action.get("y")],
          model_xy=[last_real_action.get("x"), last_real_action.get("y")],
          node_id=src_id, probe_dst_node=edge.dst_node, model_dst_node=src_id)
      aligned_now += 1
      aligned_by_action_now += int(by_action)
      aligned_unresolved_now += int(not model_key)
      # Rejects an unresolved model_key too: promoting on evidence that could
      # not be verified is the very failure this arm exists to remove.
      if args.align_requires_action_match and not by_action:
        continue
      graph.record_inference_alignment(edge.src_node, edge.action, True)
      promoted = graph.promote_children_of_aligned_prefix(edge.edge_id)
      graph.record_execution_verification(edge.edge_id, True)
      if promoted:
        log("prefix_aligned", step=step, edge_id=edge_id, by_action=by_action,
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
          aligned=aligned_now, aligned_by_action=aligned_by_action_now,
          aligned_action_unresolved=aligned_unresolved_now,
          action_was_guessed=guessed,
          real_action=last_real_action.get("action_type"))
    pending_prefix_edge_ids.clear()

    # Store prefill. Same shape as the app launch below, and safe for the same
    # reason: it EXECUTES and then falls through to this step's inference on
    # the resulting screen, so it never spends a step of the 15-step budget.
    # That is what makes an 87%-accurate predictor usable where the skip gate -
    # which replaces the inference - could not be: a miss costs one extra safe
    # action instead of a step, and 65% of all failures are episodes that ran
    # out of steps.
    #
    # Fires only where the persistent store has seen exactly ONE control
    # executed on this screen, by at least two different tasks. Uniqueness is
    # the whole signal (2026-09-12, 116 tasks): 39/45 = 87% under that gate,
    # against 54-58% for "the most frequently executed control", which is
    # barely above chance on a screen with two or three candidates.
    #
    # Once per screen per episode, and at most three per episode: the graph
    # learned a two-screen loop and replayed it twelve times on
    # SimpleSmsReplyMostRecent (2026-09-01), every hop landing exactly where
    # predicted while the episode went nowhere.
    prefill_sources = {x.strip() for x in args.prefill_sources.split(",") if x.strip()}

    def _prefill_from(state_start, node_start, step):
      """Execute the hops the store is confident about, then fall through.

      Called twice per step: once before the inference, and once right
      after the step-0 app launch, because the app's own home screen is
      the single best-covered screen in the store - every task in that app
      stands on it. Measured cold-start over 112 tasks (2026-09-13) the
      store answers there for 47% of tasks at 75%, against 34%/76% averaged
      over all screens, and those 40 correct first hops are steps no other
      call site can reach: before this, step 0 spent its inference on the
      home screen and the first navigation waited for step 1.
      """
      # Generation-guarded like every other graph read: this sees what
      # earlier steps executed, never this step's inference.
      prefill_view = guarded.snapshot(for_step=step)
      memory = memory_for(state_start.activity.component)
      state_now, sig_now = state_start, state_start.layout_sig
      node_now = node_start
      for _ in range(args.max_prefill):
        if sig_now in prefilled_screens:
          break
        # Two sources, episode-local first because it has three times the
        # chances: 115 against 39 over 116 tasks, at 83% against 87%. What
        # this episode just did on a screen is a stronger and far more
        # available signal than what other tasks did on it.
        source = "episode"
        # Off by default. This source - "this episode already executed exactly
        # one control on this node" - carried an 83% figure measured under the
        # old signature rule on a different population. Live over 50 tasks
        # (2026-09-13) it lands 9/30 = 30%, against 23/26 = 88% for the goal
        # retrieval and 3/3 for uniqueness, and it supplied half of all hops:
        # the arm lost 8 successes against no-graph on exactly the tasks where
        # prefill fired. Part of the reason is visible right here - it sets no
        # expected activity, so its landing check falls back to node id, which
        # is the layout signature this whole change moved off.
        local = (_episode_decided(prefill_view, node_now, args.prefill_min_exec)
                 if "episode" in prefill_sources else None)
        expected_activity = ""
        if local is not None:
          control, expected_dst = local[0], ""
          expected_node = local[1]
        else:
          expected_node = ""
          if memory is None:
            log("prefill_refused", step=step, why="no_memory")
            break
          # Uniqueness first, then the goal vote for the screens it refuses.
          # Measured leave-one-out over 419 executed steps (2026-09-13):
          # uniqueness alone covers 17% at 90%, the union covers 44% at 84%,
          # and a plain screen vote covers 45% at 66% - the difference between
          # the last two is 30 wrong hops against 65.
          decided = None
          if "store" not in prefill_sources:
            log("prefill_refused", step=step, why="store_source_off")
            break
          if args.prefill_retrieval in ("uniq", "both"):
            got = memory.prefill_control(sig_now, args.prefill_min_tasks)
            if got is not None:
              decided, source = got, "unique"
          if decided is None and args.prefill_retrieval in ("goal", "both"):
            got = memory.retrieve_control(
                sig_now, goal, _goal_similarity,
                k=args.prefill_k, threshold=args.prefill_sim_thr,
                margin=args.prefill_margin,
                min_tasks=args.prefill_min_tasks)
            if got is not None:
              decided, source = got[:4], got[4]
          if decided is None:
            log("prefill_refused", step=step, why="no_candidate",
                screen_known=sig_now in getattr(memory, "screens", {}))
            break
          control, _remembered, expected_dst, expected_activity = decided
        element = _element_for_control(state_now, control)
        loosened = False
        if element is None and args.prefill_by_resource_id:
          element = _element_for_control(state_now, control, by_resource_id=True)
          loosened = element is not None
        if element is None:
          log("prefill_refused", step=step, why="control_absent",
              source=source, control=control[:80],
              resource_id_seen=any(
                  (getattr(el, "resource_id", "") or "") == control.split("|")[0]
                  for el in (getattr(state_now, "elements", ()) or ())))
          break
        if _prefill_risk_of(element, args.prefill_risk) not in ("SAFE", "LOW"):
          log("prefill_refused", step=step, why="risky", source=source,
              control=control[:80],
              risk=_prefill_risk_of(element, args.prefill_risk))
          break
        prefilled_screens.add(sig_now)
        started = time.perf_counter()
        try:
          self._execute_action(_click_on(element), {})
          time.sleep(0.35)
          state_now = capture.capture()
        except Exception as exc:  # pylint: disable=broad-exception-caught
          log("store_prefill_failed", step=step, error=str(exc)[:200])
          break
        landed = state_now.layout_sig
        landed_activity = state_now.activity.component
        # Activity, not layout signature. The same control on the same screen
        # re-lands on the same signature only 66% of the time and on the same
        # activity 96% (480 repeat observations over two 116-task runs,
        # 2026-09-13) - a signature check calls a third of the working hops
        # wrong, which cuts the chain short and, with rollback on, undoes
        # actions that were right.
        if landed == sig_now:
          # Nothing moved. Matching on activity is what makes the chain
          # usable, but it also makes a press that did nothing look like a
          # hit, because the activity is then trivially unchanged - and a
          # chain that keeps "succeeding" on one screen is the replay loop
          # this guard exists to stop.
          matched = False
        elif expected_activity:
          matched = landed_activity == expected_activity
        else:
          matched = bool(expected_dst) and landed == expected_dst
        # The model must be told what was pressed on its behalf. The step-0
        # app launch appends a summary for exactly this reason: without it the
        # model reads a screen it never navigated to, and its own history says
        # it is still on the previous one.
        name = _control_display_label(control)
        self._summaries.append(str({"action": "click", "text": name}))
        node_now = node_id_of(state_now)
        if expected_node:
          matched = node_now == expected_node
        log("skip", step=step, kind_detail="store_prefill", matched=matched,
            source=source, control=control[:80],
            expected_dst=(expected_dst or expected_node)[:12],
            expected_activity=expected_activity[-40:],
            landed=landed[:12], landed_activity=landed_activity[-40:],
            sig_matched=bool(expected_dst) and landed == expected_dst,
            by_resource_id=loosened,
            latency_s=time.perf_counter() - started)
        # Chaining is verified hop by hop, the way reusable_path is: 59% of
        # decided screens lead to another decided screen (23 of 39), which
        # takes the collapsible total from 39 steps to 64 over 116 tasks - but
        # 87% per hop compounds to 76% over two, so a hop that does not land
        # where the store recorded ends the chain and hands the screen to
        # inference.
        if not matched:
          # A hop that did not land where the store recorded used to hand the
          # model a screen it never asked for, and the model then had to climb
          # back - the expensive half of a 84%-precise predictor. Pressing
          # Back restores the screen the step started on, so a miss costs
          # ~1.5s of wall clock and no step at all. Verified, not assumed: if
          # Back does not return us to sig_now we really have moved, and the
          # summary has to stand.
          if args.prefill_rollback:
            try:
              self._execute_action(
                  json_action.JSONAction(action_type=json_action.NAVIGATE_BACK), {})
              time.sleep(0.35)
              state_back = capture.capture()
            except Exception as exc:  # pylint: disable=broad-exception-caught
              log("prefill_rollback_failed", step=step, error=str(exc)[:200])
              break
            restored = state_back.layout_sig == sig_now
            if restored:
              # The model must not be told about an action that no longer
              # happened; leaving the summary in is what made a miss cost two
              # wrong beliefs instead of none.
              self._summaries.pop()
            log("prefill_rollback", step=step, restored=restored,
                landed=state_back.layout_sig[:12])
            if restored:
              break
          break
        sig_now = landed

    if args.store_prefill and step > 0:
      _prefill_from(before, src_id, step)

    # Step 0 is open_app on the app the goal names. That decision is fixed by
    # the task text, so spending an inference on it buys nothing - and unlike
    # every other action it is one exploration can never anticipate, since
    # launching an app is not a UI probe (open_app was 11% of all model
    # actions on 2026-08-31, entirely inside the "no probe expresses this"
    # bucket). Only from the launcher, and only before any action has been
    # taken, so it cannot fire mid-task.
    if (args.bootstrap and step == 0 and not getattr(self, "_actions", [])
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
        # The app is now open, and its home screen is the best-covered screen
        # the store has: every task in this app stands on it. Prefilling from
        # here is the only way to reach that screen at all - step 0 used to
        # spend its inference on it and leave the first navigation to step 1.
        # Cold-start over 112 tasks (2026-09-13): the store answers on 47% of
        # first in-app screens at 75%, against 34%/76% averaged over all
        # screens, worth 40 further collapsed steps.
        if opened and args.store_prefill and args.prefill_after_bootstrap:
          try:
            launched = capture.capture()
            _prefill_from(launched, node_id_of(launched), step)
          except Exception as exc:  # pylint: disable=broad-exception-caught
            log("prefill_after_bootstrap_failed", step=step,
                error=str(exc)[:200])
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
        before = _settled_capture(capture)
        src_id = node_id_of(before)
        upsert(before, src_id, visited=True)

    # P_{i-1}: what the model said it was looking for in its LAST output.
    # Computed once per step and shared by exploration ranking and the
    # distiller, so both are conditioned on the same statement of need.
    current_need = parse_reasoning_prior(last_model_output["text"], goal)
    # Reset before the gate runs: leaving the previous step's verdict in place
    # would let a mode decided on a screen we have since left govern this one.
    gate_mode["value"], gate_mode["context"] = gd.NORMAL_INFERENCE, ""

    # Fold this step's prefix-alignment promotions into the view the gate
    # reads. Nothing from this window's exploration exists yet - the explorer
    # has not been started - so this cannot leak forward.
    if args.align_promotes_current_step:
      guarded.refresh_before_exploration()

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
      # Two per episode, and the first miss ends it. Pooling five full runs
      # (2026-09-01): of 109 episodes where graph skipping fired, six
      # succeeded. Much of that gap is the tasks themselves - skipping needs
      # revisits, revisits happen in the long episodes, and the control scores
      # 28% on exactly those tasks against 59% on the rest - but on that same
      # 40-task set the control still took 11 while the best arm took 10 and
      # the worst 3, so the mechanism is not carrying its own weight either.
      #
      # Within the episodes that did succeed, the only signal that survives is
      # volume: 5/68 at one or two skips, 1/32 at three or four, 0/9 at five
      # or more. Evidence strength runs the wrong way - every episode whose
      # replayed edge had three or more executions behind it failed (0/65) -
      # so a stronger confidence bar would make this worse, not better. What
      # is left is to let it fire where it has ever helped and stop it the
      # moment it is wrong.
      accurate = correct == attempted
      allowed = (consecutive_skips["n"] < 3 and accurate
                 and skip_record["n"] < 2)
      # Earn the right to chain. A multi-hop replay commits several moves on
      # one prediction, so it is only worth the exposure once this episode's
      # skips have actually been landing; before that, take one hop at a time
      # and let each be verified.
      hops = args.max_path if (skip_record["n"] >= 2 and
                               skip_record["ok"] == skip_record["n"]) else 1
      gate_snapshot = guarded.snapshot(for_step=step)
      candidate_path = (gate_snapshot.reusable_path(
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
          graph_snapshot=gate_snapshot,
          information_need=current_need.to_dict(),
          reusable_edge=candidate_path[0] if candidate_path else None,
          taken_edges=taken_here,
          recent_nodes=tuple(recent_nodes[-4:]),
          consecutive_skips=consecutive_skips["n"])
      gate_mode["value"] = gate_decision.mode
      gate_mode["context"] = gate_decision.graph_context
      first_hop = candidate_path[0] if candidate_path else {}
      log("gate", step=step, mode=gate_decision.mode, reason=gate_decision.reason,
          had_reusable=bool(candidate_path),
          context_tokens=len(gate_decision.graph_context.split()),
          # Decision-time edge statistics, not the end-of-episode ones. The
          # graph dump only holds final counts, and a skip's own outcome is
          # written into them, so post-hoc analysis cannot recover what the
          # gate actually saw.
          edge_status=first_hop.get("status"),
          edge_execution_hits=first_hop.get("execution_hit_count"),
          edge_execution_misses=first_hop.get("execution_miss_count"),
          edge_probe_count=first_hop.get("probe_count"),
          node_visits=gate_snapshot.node_visits.get(src_id, 0),
          node_entropy=gate_snapshot.node_entropy.get(src_id))
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
          # An action the model repeats does not have to land in the same
          # place each time - deleting the second recipe leaves a different
          # list than deleting the first - so a replay is judged against every
          # destination this transition has been seen to reach, and against
          # having moved at all. Demanding the single most recent destination
          # would score the mechanism's best case as a miss.
          seen_destinations = set(edge.get("observed_destinations") or ())
          seen_destinations.add(edge["dst_node"])
          matched = landed in seen_destinations or (
              len(seen_destinations) > 1 and landed != edge["src_node"])
          graph.record_execution_verification(edge["edge_id"], matched)
          graph.record_skip_result(edge["edge_id"], matched,
                                   generation=guarded.generation)
          upsert(after, landed, visited=True)
          # The agent's prompt is built from this history, so what goes in it
          # has to read like what the model itself writes. Putting the raw
          # action dict there - "{'action_type': 'click', 'x': 540, 'y': 600}"
          # - hands the next inference a step with no task-level meaning, and
          # on multi-step stateful tasks that is enough to derail it: the five
          # SMS tasks v37 lost to skipping (2026-09-02) all had 100% action
          # match and still went from 6-7 steps to 13-15. A skip that is
          # action-correct can still be history-destructive, which is why
          # match rate alone was never the right measure of it.
          label = _replay_label(edge)
          leads_to = ", ".join(str(x) for x in
                               (edge.get("discovered_labels") or ())[:3])
          summary = (f"Selected {label} again, repeating a step this episode "
                     f"has already taken here"
                     + (f"; it leads to {leads_to}." if leads_to else "."))
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
                 and not _unprobeable(before.activity.component.split("/", 1)[0])
                 and not dirty_from_last_step["value"]
                 and step >= 2
                 and probe_total["n"] < args.probe_budget
                 and not unrecovered_once["value"]
                 # Only screens the trajectory has actually come back to.
                 # What exploration learns about a screen is worth something
                 # exactly when that screen is seen again - and measured on
                 # 2026-08-31, every edge any skip ever replayed had
                 # probe_count 0, so exploration had never once supplied the
                 # thing skipping runs on. Probing screens the episode passes
                 # through once therefore paid nothing while carrying the full
                 # rollback risk that cost 22 points of success rate on the
                 # episodes where a round failed to restore. Exploration still
                 # runs, on the revisits where progressive memory can pay.
                 and (snapshot.node_visits.get(src_id, 0) >= args.min_node_visits
                      or (args.cross_task_revisit
                          and before.layout_sig in preknown_layouts))
                 # Spread a fixed amount of exploration over the whole episode
                 # instead of concentrating it on whichever screens happen to
                 # satisfy the revisit rule. The revisit rule reaches only 51
                 # of 115 tasks (measured over five full runs); an interval
                 # reaches every task that lasts long enough, at a cost the
                 # episode chooses rather than the app's shape.
                 and (args.explore_interval <= 1
                      or step - last_explored_step["n"] >= args.explore_interval)
                 and not _has_unsaved_input(before))
    pending_explorer = None
    if may_probe:
      last_explored_step["n"] = step
    def _exploration_budget() -> float:
      """How long this round may run before it stops starting new probes.

      'auto' tracks the window inference is actually taking in this episode -
      the median of what has been observed so far, so it adapts per app and
      per model without a hand-set constant. A probe already in flight still
      finishes, so a round can overrun by at most one probe (~0.6s median).

      The pre-existing default (120s) is kept as the explicit numeric option:
      it is no budget at all, which is what every run before 2026-09-09 used.
      """
      if args.exploration_budget_s != "auto":
        return float(args.exploration_budget_s)
      if not inference_latencies:
        return 3.0  # before the first observation, near the measured median
      return statistics.median(inference_latencies)

    explorer_config = {
        "trial_id": f"{args.task}-step{step}", "task": goal, "step_idx": step,
        "trace_path": str(root / "probe_trace.jsonl"),
        "filtered_path": str(root / "filtered_elements.jsonl"),
        "ranker": {"random": "RandomRanker",
                   "coverage": "CoverageRanker",
                   "semantic": "GraphKeywordRanker",
                   "llm_choice": "LlmChoiceRanker"}.get(
                       args.exploration_policy, "InformationNeedRanker"),
        "semantic_port": args.semantic_port,
        "visited_discount": args.visited_discount,
        "seed": args.seed + step,
        "information_need": current_need.to_dict(),
        "serial": "emulator-5554", "console_port": 5554,
        "a11y_local_port": args.a11y_socket_port, "restore_timeout_s": 30.0,
        "max_probes": args.probes_per_step, "min_probes": 0,
        "post_inference_grace_s": 0.0, "max_depth": args.max_depth,
        "max_exploration_time_s": _exploration_budget(),
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
        "allow_icon_only_probes": args.allow_icon_only_probes,
        "recent_nodes": list(recent_nodes[-4:]),
        "scored_path": str(root / "scored_candidates.jsonl"),
        "scoring_config": dataclasses.asdict(scoring_config),
        "blocked_element_identities": sorted(blocked_element_identities),
        # (activity, probe_type) -> the ladder rung that actually undid this
        # kind of transition here before. The explorer starts from that rung
        # instead of walking the ladder from the top, which is the difference
        # between remembering an inverse and rediscovering it.
        "known_inverse_levels": known_inverse_levels(),
        "depth2_needs_known_inverse": args.depth2_needs_known_inverse,
    }
    if may_probe:
      try:
        pending_explorer = spawn_explorer(explorer_config)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        log("explore_spawn_failed", step=step, error=str(exc)[:300])

    # "Already used" has to mean "used on this visit". Scoped to the current
    # contiguous stay on this screen: an edge taken two visits ago is not a
    # repeat to warn against, it is the most useful thing the graph holds
    # about a screen the task keeps coming back to - "last time you were here
    # you pressed X and it led to {...}". Keeping it episode-global made every
    # edge at a revisited node a Done fact, which is why nothing positive was
    # ever left to inject.
    if last_context_node["id"] != src_id:
      taken_here.clear()
      last_context_node["id"] = src_id

    # B9.2/B9.3: what the graph knows reaches the prompt only when it holds
    # facts worth the space. Read from the step i-1 snapshot, like the skip
    # decision, so this step's own probes cannot influence this step.
    if args.graph_context == "onclick":
      # AutoDroid's form: annotate every control the graph knows a destination
      # for, instead of a block that must first earn the right to speak. The
      # gate's own context is not reused here - it was rendered in block form.
      briefing_now["text"] = distiller.distill(
          current_node_id=src_id, graph_snapshot=snapshot,
          information_need=current_need.to_dict(), taken_edges=taken_here,
          recent_nodes=tuple(recent_nodes[-4:]), onclick_form=True)
    elif args.graph_context == "distill":
      # Already computed by the gate when it decided this step was not certain
      # enough to skip; recomputing it would just repeat the same deterministic
      # pass over the same snapshot.
      briefing_now["text"] = gate_mode["context"] if gate_mode["value"] == (
          gd.GRAPH_ENHANCED_INFERENCE) else distiller.distill(
              current_node_id=src_id, graph_snapshot=snapshot,
              information_need=current_need.to_dict(),
              taken_edges=taken_here, recent_nodes=tuple(recent_nodes[-4:]))
    elif args.graph_context == "briefing":
      briefing_now["text"] = snapshot.screen_briefing(src_id, taken_here)
    elif args.graph_context == "llm":
      # One call, one line. The deterministic forms are both true and both
      # long: the walked path renders at ~50 tokens and the distiller at up to
      # 64, and prompt space spent on the graph is space not spent on the
      # screenshot. Reads the same step i-1 snapshot as every other graph
      # read, so this step's own probes cannot reach it.
      lines = _walked_path_lines(snapshot, src_id, limit=8)
      from android_world.parallel_exploration import semantic_service
      one = semantic_service.query_summarize(args.semantic_port, goal, lines)
      briefing_now["text"] = (
          f"[Memory] {one}" if one else _walked_path_context(snapshot, src_id))
    elif args.graph_context in ("elements", "elements_store"):
      summaries = getattr(snapshot, "node_summaries", {}) or {}
      parts = [_elements_context(snapshot, src_id, summaries)]
      if args.graph_context == "elements_store":
        parts.append(_store_routes_context(
            memory_for(before.activity.component), before.layout_sig))
      briefing_now["text"] = "\n\n".join(p for p in parts if p)
    elif args.graph_context in ("store", "store_path"):
      parts = [_store_routes_context(memory_for(before.activity.component),
                                     before.layout_sig)]
      if args.graph_context == "store_path":
        parts.append(_walked_path_context(snapshot, src_id))
      briefing_now["text"] = "\n\n".join(p for p in parts if p)
    elif args.graph_context in ("path", "distill_path"):
      parts = []
      if args.graph_context == "distill_path":
        parts.append(gate_mode["context"] if gate_mode["value"] == (
            gd.GRAPH_ENHANCED_INFERENCE) else distiller.distill(
                current_node_id=src_id, graph_snapshot=snapshot,
                information_need=current_need.to_dict(),
                taken_edges=taken_here, recent_nodes=tuple(recent_nodes[-4:])))
      parts.append(_walked_path_context(snapshot, src_id))
      briefing_now["text"] = "\n\n".join(p for p in parts if p)
    elif args.graph_context in ("history", "history_done"):
      briefing_now["text"] = screen_history_context(
          snapshot, src_id, taken_here,
          include_available=args.graph_context == "history")
    else:
      briefing_now["text"] = ""
    if args.loop_notice:
      loop = _loop_context(snapshot, src_id, args.loop_repeats)
      if loop:
        briefing_now["text"] = (briefing_now["text"] + "\n\n" + loop
                                if briefing_now["text"] else loop)
    distiller_stats: dict[str, Any] = {}
    if args.graph_context in ("distill", "onclick"):
      distiller.distill(
          current_node_id=src_id, graph_snapshot=snapshot,
          information_need=current_need.to_dict(), taken_edges=taken_here,
          recent_nodes=tuple(recent_nodes[-4:]), stats=distiller_stats,
          onclick_form=args.graph_context == "onclick")
    log("graph_context", step=step, mode=args.graph_context,
        graph_mode=gate_mode["value"], injected=bool(briefing_now["text"]),
        chars=len(briefing_now["text"]),
        tokens=len(briefing_now["text"].split()),
        text=briefing_now["text"], **distiller_stats)

    # (3) Exploration from S_i, BEFORE inference. Both start from the screen
    # this window opened on, which is what "parallel" means here: the explorer
    # cannot see this step's inference, and this step's decision may use what
    # the explorer found. Running it after inference - as this did until
    # 2026-09-04 - was not that. GELABAgent.step() executes the action it
    # chooses (gelab_agent.py:1215), so by the time it returned the device was
    # already on S_{i+1} and every probe explored the NEXT screen. The prefix
    # check is written for the other timing - it asks whether a probe's
    # destination equals the screen the model then landed on - which for
    # probes taken from S_{i+1} can never hold. Prefix alignment was 0 in all
    # five full runs, and with it the SPECULATIVE -> REUSABLE promotion the
    # whole lookahead depends on.
    exploration_s = 0.0
    if pending_explorer is not None:
      started = time.perf_counter()
      try:
        outcome = run_serial_exploration(
            await_explorer_ready(pending_explorer),
            fresh_need=current_need.to_dict())
        dirty_from_last_step["value"] = outcome.get("restore_status") != "RESTORED"
        probe_total["n"] += int(outcome.get("probes_completed", 0) or 0)
        ingest_probe_trace(explorer_config["trial_id"])
        log("explore", step=step, depth1_edges=len(depth1_edge_ids), **outcome)
        # Repair before the model reads the screen, not after. The parallel
        # design hands inference a screenshot taken at window start, so a
        # stranding probe costs it nothing; serially the model reads whatever
        # the probe left behind, so the repair has to come first or the
        # disturbance lands on this step instead of the next.
        repair_after_exploration(self, before, outcome, step)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        log("explore_failed", step=step, error=str(exc)[:300])
      exploration_s = time.perf_counter() - started

    # (4) Inference. The explorer has finished and the app has been put
    # back, so the model reads the screen this window opened on.
    inference_started = time.perf_counter()
    result = original_step(self, goal)
    inference_s = time.perf_counter() - inference_started

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
        # Nothing is learned about an app by leaving it. A transition that
        # lands on the launcher or a system dialog still gets recorded - the
        # graph needs to know the agent went there - but it carries no
        # evidence labels, because the labels would be the home screen's.
        # Without this the distiller injected "Verified: Cancel -> {Sun, Oct
        # 15, 0, Gmail, Photos}" into every SMS task (2026-09-02), a true and
        # useless fact learned from an earlier episode escaping a system
        # dialog. Filtering it out of cross-task memory alone was not enough:
        # the edge is rebuilt inside each episode, and the distiller reads the
        # episode's own graph.
        # Only what the transition ADDED. Taking every label on the landing
        # screen makes persistent chrome the content of the fact: on
        # MarkorCreateFolder (2026-09-02) the graph injected "Verified: FOLDER
        # -> {Markor, Go to, Sort by, Search}" - the app's own toolbar, present
        # before and after the tap - three times, and the task went from 5
        # steps to 15. A destination described entirely by what was already on
        # screen carries no information, so it is not worth a claim; the probe
        # path has always used new_texts for exactly this reason.
        before_labels = frozenset(
            label.strip().casefold()
            for element in before.elements
            for label in (element.text, element.content_desc) if label)
        labels = tuple(dict.fromkeys(
            label for element in after.elements
            for label in (element.text, element.content_desc)
            if label and len(label) < 40 and _is_app_content(element, label)
            and label.strip().casefold() not in before_labels
        ))[:8] if _is_app_destination(after.activity.component) else ()
        # Record the control, not the pixel. A copy, so the agent's own action
        # (returned to the harness, replayed by recovery) is untouched.
        edge_action = dict(action_dict)
        control = _control_key_at(before, action_dict)
        if control:
          edge_action["control_key"] = control
        observed = graph.add_speculative_transition(
            src_id, edge_action, dst_id, path_probability=1.0, confidence=0.0,
            expected_information_gain=0.0, risk_level="SAFE",
            exploration_cost=0.0, rollback_success=True, discovered_labels=labels)
        graph.record_execution_verification(observed.edge_id, True)
        observed.nodes_at_last_execution = len(graph.nodes)
        # The dynamics of what the agent just did belong to the app and outlive
        # the task; the fact that the model chose it does not. Only probes fed
        # app memory at first, and probes are rare - 19 tasks produced 7
        # remembered transitions against 70-odd screens (2026-09-01), so almost
        # nothing could be seeded back. Every real step is also an observation
        # of where a control leads. What is deliberately NOT carried across is
        # the execution count: seed_graph writes those as zero, so a remembered
        # transition can describe the app without ever licensing a replay.
        memory = memory_for(before.activity.component)
        if memory is not None and control:
          memory.observe_execution(
              activity=before.activity.component,
              layout_signature=before.layout_sig,
              control_key=control,
              action={k: v for k, v in action_dict.items()
                      if k in {"action_type", "x", "y", "direction"}},
              dst_activity=after.activity.component,
              dst_layout_signature=after.layout_sig,
              labels=labels,
              goal=goal)
        taken_edge_ids.add(observed.edge_id)
        taken_here.add(observed.edge_id)
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
      if args.node_summary != "off":
        node = graph.nodes.get(src_id)
        if node is not None and not (node.semantic_summary or "").strip():
          sentence = _screen_sentence(last_model_output["text"])
          if not sentence and args.node_summary == "model":
            # 42.5% of steps emit a bare tool call with no <THINK>, leaving the
            # screen blank. Only those reach the small model: a description the
            # agent wrote itself is better and already paid for.
            from android_world.parallel_exploration import semantic_service
            sentence = semantic_service.query_describe(
                args.semantic_port, goal, node.activity,
                list(node.salient_ui_labels or ()))
          node.semantic_summary = sentence
          # Cross-task too: the model's per-episode history resets, this does
          # not, and 39% of screens are seen by more than one task.
          memory = memory_for(node.activity)
          if memory is not None and sentence:
            memory.observe_screen(node.activity, node.layout_signature,
                                  description=sentence)
    consecutive_skips["n"] = 0
    recent_nodes.append(src_id)
    pending_prefix_edge_ids.extend(depth1_edge_ids)
    last_real_action.clear()
    last_real_action.update(action_dict)
    # The model reports a coordinate, a probe edge carries a control_key, and
    # _canonical_control reads control_key first. Without resolving the tap to
    # the control it landed on, every click compares a real key against "" and
    # the by_action arm of prefix alignment is structurally false - measured
    # 2026-09-09, every prefix_candidate logged model_canonical_action="".
    # Same resolution the authoritative edge above uses, against the same
    # pre-action screen, so the two keys are directly comparable.
    real_control = _control_key_at(before, action_dict)
    if real_control:
      last_real_action["control_key"] = real_control
    if inference_s > 0:
      inference_latencies.append(inference_s)
    log("inference", step=step, inference_s=inference_s,
        exploration_s=exploration_s, action=action_dict, done=result.done)
    step_counter["i"] += 1
    guarded.commit_step()
    dump_graph_snapshot(step)
    return result

  # Which class carries step() depends on whether the screenshot is
  # downsampled: GELABResizeAgent overrides step() rather than extending it, so
  # patching gelab_agent.GELABAgent leaves the resize path untouched and the
  # whole design would silently not run.
  if args.downsample > 1.0:
    from android_world.agents import gelab_agent_resize
    original_step = gelab_agent_resize.GELABResizeAgent.step
    gelab_agent_resize.GELABResizeAgent.step = serial_step
  else:
    gelab_agent.GELABAgent.step = serial_step
  # Install whenever anything wants to reach the prompt. This used to be gated
  # on --inject_briefing alone, a flag no batch ever passed, so the distiller
  # computed its context on every step and the result was thrown away - module
  # three never ran end to end in any full run, and the "injection is neutral"
  # measurement was measuring nothing (2026-09-02).
  if args.inject_briefing or args.graph_context != "off" or args.decision_constraints:
    gelab_agent.build_gelab_messages = build_with_briefing
  sys.argv = [
      "run.py", "--suite_family=android_world",
      f"--agent_name={'gelab_agent_resize' if args.downsample > 1.0 else 'gelab_agent'}",
      f"--image_downsample_scale={args.downsample}",
      f"--tasks={args.task}", "--n_task_combinations=1", "--fixed_task_seed",
      f"--task_random_seed={args.seed}", f"--max_n_steps={args.max_steps}",
      "--console_port=5554", f"--output_path={root}",
  ]
  try:
    runpy.run_path(str(REPO_ROOT / "run.py"), run_name="__main__")
  finally:
    if args.node_summary == "model":
      try:
        _backfill_descriptions(graph, app_store, args.semantic_port, log)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"description backfill skipped: {exc}", flush=True)
    (root / "progressive_belief_graph.json").write_text(
        json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    if app_store is not None:
      written = app_store.save_all()
      (root / "app_memory_stats.json").write_text(
          json.dumps({**app_store.stats(), "files": [str(x) for x in written]},
                     ensure_ascii=False, indent=1), encoding="utf-8")
  return 0


if __name__ == "__main__":
  main()
