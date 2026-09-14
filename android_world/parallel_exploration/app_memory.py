"""Structural memory of an app, kept across tasks.

The belief graph is rebuilt from nothing at the start of every episode, which
throws away two very different kinds of knowledge:

* **What the app is** - which control leads to which screen, what is found
  there, which rung of the recovery ladder undoes a tap here, which probes
  have proven unrecoverable. None of this depends on the task. Measured over
  three full runs (2026-09-01), 33% of screens recur across tasks, each seen
  by 4.0 tasks on average, and nine (activity, probe_type) combinations failed
  to roll back in more than one task - fourteen probe accidents that a
  remembered blocklist would have skipped outright. An episode that suffers
  one unrecovered round wins 67% of the time against 89% for one that does
  not, so those fourteen are not free.

* **What the model decided** - that it chose this control on this screen, and
  how often. This is entirely task-specific: having pressed Delete on the
  recipe list last task says nothing about whether to press it now. Replaying
  intent across tasks would feed the one mechanism already measured as
  harmful (tasks where skipping fired: 1/17, against a control's 6/17).

So only the first kind persists. This module is that first kind, keyed by
package, and it deliberately carries no execution counts, visit counts or
skip statistics - the fields the skip gate reads stay episode-local by
construction rather than by discipline.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import threading
import time
from typing import Any, Callable, Iterable, Mapping

from android_world.parallel_exploration.belief_graph import EdgeStatus
from android_world.parallel_exploration.belief_graph import GraphNode
from android_world.parallel_exploration.belief_graph import NodeStatus
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph

SCHEMA_VERSION = 1

# Destinations that are not facts about the app. A file picker or a share
# target is a real place the app can send you and worth remembering; the home
# screen and a system dialog are where the agent ends up when it has left, and
# recording them teaches the wrong thing.
#
# Measured on 2026-09-02: every SMS task carried the injected fact
# "Verified: Cancel -> {Sun, Oct 15, 0, Gmail, Photos}" - launcher content.
# It was learned from an earlier episode that had got stuck in the default-SMS
# role dialog and pressed Cancel to escape, which the graph recorded as an
# authoritative edge and cross-task memory then fixed in place and replayed
# into the prompt of every later SMS task. The transition was true and useless:
# leaving the app is not knowledge about the app.
_NON_DESTINATIONS = ("nexuslauncher", "launcher",
                     "permissioncontroller", "packageinstaller")


def _is_app_destination(activity: str) -> bool:
  """Is this somewhere the app sent us, or somewhere we escaped to?"""
  lowered = (activity or "").casefold()
  return bool(lowered) and not any(x in lowered for x in _NON_DESTINATIONS)


@dataclasses.dataclass
class RememberedTransition:
  """One thing an action does on one screen, learned by probing it."""

  control_key: str
  probe_type: str
  action: dict[str, Any]
  dst_layout_signature: str = ""
  dst_activity: str = ""
  discovered_labels: tuple[str, ...] = ()
  probe_count: int = 0
  # 模型自己执行过这条转移多少次，以及有多少个不同的任务执行过它。probe_count
  # 记的是探测，与"任务真的走过这条路"是两回事，而预填只能依据后者。
  executed_count: int = 0
  tasks_executed: int = 0
  # 执行过这条转移的任务的目标原文，最多 8 条。按屏幕投票只有 66% 的命中率
  # （2026-09-13，419 个真实执行步的留一法复盘），因为同一块屏幕上不同任务要
  # 去不同的地方。把目标存下来，检索时就能问"目标和我最像的任务在这儿点了
  # 什么"，同样的覆盖率下命中率 66% → 84%。
  goals: tuple[str, ...] = ()
  rollback_success_count: int = 0
  rollback_failure_count: int = 0
  inverse_level: str = ""
  last_seen: float = dataclasses.field(default_factory=time.time)

  @property
  def rollback_success_rate(self) -> float | None:
    seen = self.rollback_success_count + self.rollback_failure_count
    return self.rollback_success_count / seen if seen else None


@dataclasses.dataclass
class RememberedScreen:
  activity: str
  layout_signature: str
  labels: tuple[str, ...] = ()
  transitions: dict[str, RememberedTransition] = dataclasses.field(default_factory=dict)
  tasks_seen: int = 0
  last_seen: float = dataclasses.field(default_factory=time.time)
  # One line saying what this screen is, in the model's own words, written the
  # first time a task stood on it. The model's per-episode history resets; this
  # does not, and 39% of screens are visited by more than one task (461 screens
  # over 115 tasks, the busiest appearing in 149 of them). Free: the sentence
  # comes from the <THINK> the model already wrote while looking at the screen.
  description: str = ""


def _control_display_name(control_key: str) -> str:
  """A name a person would use for this control, or "" if it has none.

  `control_key` is resource_id|text|content_desc|class. The text and the
  content-description are what the control is called; the resource id is how
  the app refers to it and the class is not a name at all. 38% of remembered
  transitions carry only a resource id (measured 2026-09-11 over 321), and
  rendering those produced facts that read "android.widget.TextView led to a
  sidebar" - true, and useless to a reader.
  """
  parts = (control_key or "").split("|")
  for index in (1, 2):
    if len(parts) > index and parts[index].strip():
      return " ".join(parts[index].split())[:36]
  return ""


class AppMemory:
  """Everything remembered about one package, across every task run on it."""

  def __init__(self, package: str):
    self.package = package
    self.screens: dict[str, RememberedScreen] = {}
    # Screens this process has already counted; see observe_screen.
    self._seen_this_task: set[str] = set()
    # Transitions this process has already counted towards tasks_executed.
    self._executed_this_task: set[str] = set()
    # (activity, probe_type) that has ever failed to roll back. Remembered
    # rather than relearned: relearning costs one unrecoverable probe each
    # time, and the same nine combinations failed in more than one task.
    self.blocked_contexts: set[str] = set()
    # Elements that took the device out of the package entirely.
    self.blocked_elements: set[str] = set()
    # (activity, probe_type) -> the ladder rung that actually undid it.
    self.inverse_levels: dict[str, str] = {}
    self.tasks_recorded: int = 0
    self._lock = threading.Lock()

  # -- observation --------------------------------------------------------

  def observe_screen(self, activity: str, layout_signature: str,
                     labels: Iterable[str] = (),
                     description: str = "") -> RememberedScreen:
    with self._lock:
      screen = self.screens.get(layout_signature)
      if screen is None:
        screen = RememberedScreen(activity=activity, layout_signature=layout_signature)
        self.screens[layout_signature] = screen
      # One process is one task, so the first touch of a screen in this process
      # is this task seeing it. The field was declared, serialised and
      # deserialised but never incremented, so every screen in a 274-screen
      # store read tasks_seen=0 (2026-09-10) - which made "is this description
      # on a screen many tasks land on, or one nobody revisits" unanswerable.
      if layout_signature not in self._seen_this_task:
        self._seen_this_task.add(layout_signature)
        screen.tasks_seen += 1
      merged = tuple(dict.fromkeys(tuple(screen.labels) + tuple(labels)))[:12]
      screen.labels = merged
      # First description wins. A later task's phrasing is not better, and
      # overwriting would let a step that emitted no reasoning blank it.
      if description and not screen.description:
        screen.description = description[:200]
      screen.last_seen = time.time()
      return screen

  def observe_probe(self, *, activity: str, layout_signature: str,
                    control_key: str, probe_type: str, action: Mapping[str, Any],
                    dst_activity: str = "", dst_layout_signature: str = "",
                    discovered_labels: Iterable[str] = (),
                    rollback_ok: bool = True, inverse_level: str = "") -> None:
    """A probe of this transition happened; remember what it does."""
    if not control_key:
      return
    screen = self.observe_screen(activity, layout_signature)
    key = f"{control_key}\0{probe_type}"
    with self._lock:
      known = screen.transitions.get(key)
      if known is None:
        known = RememberedTransition(control_key=control_key, probe_type=probe_type,
                                     action=dict(action))
        screen.transitions[key] = known
      known.probe_count += 1
      known.rollback_success_count += int(rollback_ok)
      known.rollback_failure_count += int(not rollback_ok)
      if inverse_level:
        known.inverse_level = inverse_level
      if dst_layout_signature and _is_app_destination(dst_activity):
        known.dst_layout_signature = dst_layout_signature
        known.dst_activity = dst_activity
      labels = (tuple(str(x) for x in discovered_labels if x)
                if _is_app_destination(dst_activity) else ())
      if labels:
        known.discovered_labels = tuple(
            dict.fromkeys(known.discovered_labels + labels))[:8]
      known.last_seen = time.time()

  def observe_execution(self, *, activity: str, layout_signature: str,
                        control_key: str, action: Mapping[str, Any],
                        dst_activity: str, dst_layout_signature: str,
                        labels: Iterable[str] = (), goal: str = "") -> None:
    """The agent really took this transition; remember where it goes.

    Separate from observe_probe because the rollback statistics must not be
    touched: a real execution is never undone, so it says nothing about
    whether *probing* this control could be recovered from, and folding it in
    would inflate the recoverability that seed_graph reads.

    This is where most remembered dynamics come from. Probes are rare - 19
    tasks produced 7 remembered transitions against 70-odd screens
    (2026-09-01) - while every step the agent takes is an observation of what
    a control does.
    """
    if not control_key or not dst_layout_signature:
      return
    if not _is_app_destination(dst_activity):
      # Left the app. Nothing about that belongs in this app's memory.
      return
    screen = self.observe_screen(activity, layout_signature)
    key = f"{control_key}\0EXECUTED"
    with self._lock:
      known = screen.transitions.get(key)
      if known is None:
        known = RememberedTransition(control_key=control_key,
                                     probe_type="EXECUTED", action=dict(action))
        screen.transitions[key] = known
      known.dst_layout_signature = dst_layout_signature
      known.dst_activity = dst_activity
      known.executed_count += 1
      # One process is one task, so the first execution of this transition in
      # this process is one more task having taken it. Counting tasks rather
      # than executions is what the prefill gate needs: a task that presses
      # the same control six times is one task's worth of evidence.
      if key not in self._executed_this_task:
        self._executed_this_task.add(key)
        known.tasks_executed += 1
      merged = tuple(str(x) for x in labels if x)
      if merged:
        known.discovered_labels = tuple(
            dict.fromkeys(known.discovered_labels + merged))[:8]
      goal_text = (goal or "").strip()
      if goal_text:
        known.goals = tuple(dict.fromkeys(known.goals + (goal_text,)))[:8]
      known.last_seen = time.time()

  def observe_rollback(self, activity: str, probe_type: str, *, ok: bool,
                       level: str = "") -> None:
    key = f"{activity}|{probe_type}"
    with self._lock:
      if ok:
        if level:
          self.inverse_levels[key] = level
      else:
        self.blocked_contexts.add(key)

  def observe_escape(self, element_identity: str) -> None:
    if element_identity:
      with self._lock:
        self.blocked_elements.add(element_identity)

  # -- use ----------------------------------------------------------------

  def undescribed_screens(self) -> list[tuple[str, str, tuple[str, ...]]]:
    """(activity, layout_signature, labels) for screens still without a name.

    Only screens that have something to name them by. The caller decides what
    to spend describing them; this just says which ones are still blank.
    """
    return [(s.activity, sig, s.labels)
            for sig, s in self.screens.items()
            if not (s.description or "").strip() and s.labels]

  def settled_control(self, layout_signature: str,
                      min_tasks: int = 2) -> tuple[str, int] | None:
    """(control_key, how many tasks) when this screen is known for exactly one.

    Measured 2026-09-10 over three arms: on a screen where the store has ever
    seen exactly one control executed, the control the next task presses is
    that same one **82% of the time** (81-86%, and flat no matter how many
    prior tasks are demanded - more evidence shrinks the pool without making
    the prediction better).

    82% is far too low for a skip, which commits the action unverified and
    where one miss in five is ruinous against the 15-step cap - that sweep is
    in the design doc and the arm was cancelled on it. It is high for a
    *statement*: prefix alignment, the only other forward-looking signal this
    system has, runs at 2.5%. So this returns evidence, and the caller may
    only ever say it, never act on it.

    Counted per task, not per execution: a screen one task pressed six times
    is one task's worth of evidence.
    """
    screen = self.screens.get(layout_signature)
    if screen is None:
      return None
    by_control: dict[str, int] = {}
    for known in screen.transitions.values():
      if not known.control_key:
        continue
      by_control[known.control_key] = (by_control.get(known.control_key, 0)
                                       + max(1, known.tasks_seen))
    if len(by_control) != 1:
      return None
    control, tasks = next(iter(by_control.items()))
    return (control, tasks) if tasks >= min_tasks else None

  def prefill_control(self, layout_signature: str,
                      min_tasks: int = 2) -> tuple[str, dict, str, str] | None:
    """(control_key, action, dst_layout_signature, dst_activity) when decided.

    "Decided" means the store has seen exactly ONE control executed here, by at
    least `min_tasks` different tasks. Uniqueness is what carries the signal:
    measured 2026-09-12 over 116 tasks, the next task presses that same control
    **87%** of the time (39/45), while relaxing to "the most frequently
    executed control" drops to 54-58% - barely above chance on a screen with
    two or three candidates.

    The caller must execute this and then still run the step's inference on
    the resulting screen, the way the step-0 app launch does. That is what
    makes 87% usable: a miss costs one extra safe action, not a step of the
    15-step budget, and 65% of all failures are episodes that ran out of steps.
    """
    screen = self.screens.get(layout_signature)
    if screen is None:
      return None
    by_control: dict[str, RememberedTransition] = {}
    tasks: dict[str, int] = {}
    for known in screen.transitions.values():
      if not known.control_key or not known.executed_count:
        continue
      by_control[known.control_key] = known
      tasks[known.control_key] = tasks.get(known.control_key, 0) + max(
          1, known.tasks_executed)
    if len(by_control) != 1:
      return None
    control, known = next(iter(by_control.items()))
    if tasks[control] < min_tasks:
      return None
    return (control, dict(known.action), known.dst_layout_signature,
            known.dst_activity)

  _ASKS = re.compile(
      r"(answer with|answer the following|what (is|are|do|does)|how many"
      r"|do i have|which |tell me)", re.I)

  @classmethod
  def _is_question(cls, goal: str) -> bool:
    """Does this goal ask for an answer rather than an action?

    The store is built exclusively from tasks that *did* something -
    observe_execution only ever records a control someone pressed. A goal that
    asks a question has no prior action to replay, and sentence embeddings put
    it right next to the action goals that share its vocabulary: "Do I have any
    events October 28" sits at high similarity to "Create an event on
    October 28", so the retrieval hands back the new-event FAB.

    Measured over two full 116-task runs (2026-09-14), splitting on this
    predicate: on the 94-96 action tasks the design is level with no-graph
    (47 v 47, 45 v 46); on the 18 question tasks it loses 4 and 2. **The whole
    deficit is here**, and prefill fired on 13 and 8 of those 18.

    This is a property of the goal string, not of any app.
    """
    return bool(cls._ASKS.search(goal or ""))

  def retrieve_control(
      self, layout_signature: str, goal: str,
      similarity: Callable[[str, list[str]], list[float] | None],
      *, k: int = 3, threshold: float = 0.5, margin: float = 0.6,
      min_tasks: int = 2, min_pass_rate: float = 0.7,
      goal_vote: bool = True, same_kind_only: bool = True,
      trace: dict[str, Any] | None = None) -> tuple[str, dict, str, str, str] | None:
    """(control, action, dst_layout_signature, dst_activity, source).

    Goal-conditioned. The destination activity is returned alongside the
    layout signature because that is the field a landing check must use: the
    same control on the same screen re-lands on the same layout signature only
    66% of the time but on the same activity 96% (480 repeat observations over
    two 116-task runs, 2026-09-13). Checking the signature calls a third of
    the correct hops wrong, which ends the chain early and - once a miss is
    rolled back - undoes actions that worked.

    `prefill_control` fires only on screens where the store has seen exactly
    one control executed. That gate is precise - 90% over the 419 executed
    steps of a 115-task run (2026-09-13, leave-one-out) - but it covers only
    17% of them, because most screens are shared by tasks that go different
    ways from there. Voting over the whole screen instead lifts coverage to
    45% and drops precision to 66%: 125 free hops bought with 65 wrong ones,
    which is why measured end to end it was worth nothing.

    What separates the two cases is not the screen, it is the *task*. So this
    asks the narrower question: of the tasks that stood here before, what did
    the ones whose goal reads like mine press? Same leave-one-out, k=3,
    similarity >= 0.5, weighted margin >= 0.6:

        唯一控件 min_tasks=2     覆盖 17%   命中 90%    66 对 /  7 错
        按屏幕多数票             覆盖 45%   命中 66%   125 对 / 65 错
        目标 kNN                覆盖 42%   命中 84%   147 对 / 29 错
        两者并集（本方法）        覆盖 44%   命中 84%   154 对 / 30 错

    Uniqueness runs first because where it applies it is the better of the
    two; the goal vote picks up the screens it refuses.

    `similarity(goal, candidates) -> scores` is injected rather than imported
    so the store keeps working when the encoder is down; returning None there
    simply falls back to the uniqueness answer.
    """
    def refuse(gate: str) -> None:
      if trace is not None:
        trace["gate"] = gate

    screen = self.screens.get(layout_signature)
    asking = self._is_question(goal or "")
    unique = self.prefill_control(layout_signature, min_tasks)
    if unique is not None and screen is not None:
      known = screen.transitions.get(f"{unique[0]}\0EXECUTED")
      # 唯一性门本身不看目标，所以同类要求得在这里加：问答类任务只能复用
      # 同样是问答类的先例。store 绝大部分是执行类记忆，不加这一条，
      # 「有没有 X」会拿到「新建 X」按过的控件。
      kind_ok = (not same_kind_only or not known
                 or any(self._is_question(g) == asking for g in known.goals)
                 or not known.goals)
      if not kind_ok:
        refuse("唯一性门：先例与本任务不同类")
        return None
      if known is None or self._passes_through(screen, known, min_pass_rate):
        return unique[0], unique[1], unique[2], unique[3], "unique"
    text = (goal or "").strip()
    if not goal_vote or not text or screen is None:
      refuse("无屏幕记录" if screen is None else "目标投票关闭")
      return None
    # One flat list of (goal, control) pairs, the way the offline replay
    # scored it: a control executed by four tasks gets four chances to be the
    # nearest neighbour, which is the frequency prior doing its work.
    pairs: list[tuple[str, str]] = []
    known_by_control: dict[str, RememberedTransition] = {}
    for known in screen.transitions.values():
      if not known.control_key or not known.executed_count:
        continue
      known_by_control.setdefault(known.control_key, known)
      for remembered_goal in known.goals:
        if not remembered_goal or remembered_goal == text:
          continue
        # Same kind only. A question goal must not inherit an action goal's
        # control just because they share nouns.
        if same_kind_only and self._is_question(remembered_goal) != asking:
          continue
        pairs.append((remembered_goal, known.control_key))
    if not pairs:
      refuse("该屏无带目标的执行转移")
      return None
    scores = similarity(text, [g for g, _ in pairs])
    if not scores or len(scores) != len(pairs):
      refuse("编码器无应答")
      return None
    ranked = sorted(zip(scores, (c for _, c in pairs)), key=lambda sc: -sc[0])
    if ranked[0][0] < threshold:
      refuse("目标相似度不足")
      if trace is not None:
        trace["best_sim"] = round(float(ranked[0][0]), 3)
      return None
    weight: dict[str, float] = {}
    for score, control in ranked[:k]:
      weight[control] = weight.get(control, 0.0) + float(score)
    order = sorted(weight.items(), key=lambda kv: -kv[1])
    # A near-tie between two controls is the case the screen vote gets wrong.
    # Refusing it is what buys the 84%: dropping the margin to 0.3 puts
    # precision back at 82%, to 0.0 at 76%.
    if len(order) > 1 and order[0][1] - order[1][1] < margin:
      refuse("冠亚军边际不足")
      return None
    control = order[0][0]
    known = known_by_control[control]
    if not self._passes_through(screen, known, min_pass_rate):
      refuse("通过率不足")
      if trace is not None:
        trace["pass_seen"] = screen.tasks_seen
        trace["pass_ran"] = known.tasks_executed
      return None
    return (control, dict(known.action), known.dst_layout_signature,
            known.dst_activity, "goal")

  def _passes_through(self, screen: "RememberedScreen",
                      known: RememberedTransition,
                      min_pass_rate: float) -> bool:
    """Did the tasks that stood on this screen mostly press this control?

    The gate the retrieval was missing, and the reason a 90%-accurate landing
    predictor was picking the right *action* only 36-43% of the time. Scored
    over every screen visit rather than only the ones that produced a click -
    927 visits over 112 tasks, of which just **402 (43%) involved a click
    navigation at all**. On the other 57% the task typed, scrolled or
    terminated, so any prefill there is wrong by construction, and an earlier
    evaluation that scored only the click steps could not see it.

    `tasks_seen` counts tasks that stood here; `tasks_executed` counts tasks
    that pressed this control. Their ratio says whether this is a pass-through
    screen or a fork. Same 927-visit replay:

        无门控        开火 291   正确 36%   104 对 / 187 错   净 -83
        通过率 >=30%  开火 188   正确 52%    97 对 /  91 错   净  +6
        通过率 >=50%  开火 148   正确 59%    87 对 /  61 错   净 +26
        通过率 >=70%  开火 127   正确 66%    84 对 /  43 错   净 +41
        通过率 >=90%  开火 121   正确 65%    79 对 /  42 错   净 +37
    """
    if min_pass_rate <= 0:
      return True
    # This task's own visit must come out of both counts. The question is
    # what *other* tasks did here, the same exclusion `retrieve_control`
    # already applies to goals. Leaving self in dilutes the ratio by one on
    # both sides and, at the counts a cold-start store actually holds,
    # suppressed firing sevenfold: 0.16 hops per task live against 1.13
    # projected (2026-09-13). Of 53 transitions in a live store, 39 cleared
    # 0.7 with self counted and 52 with it removed.
    mine_seen = 1 if screen.layout_signature in self._seen_this_task else 0
    mine_ran = 1 if f"{known.control_key}\0EXECUTED" in self._executed_this_task else 0
    seen = max(screen.tasks_seen - mine_seen, 0)
    ran = max(known.tasks_executed - mine_ran, 0)
    seen = max(seen, ran)
    if seen <= 1:
      # A single prior visit says nothing about whether this is a fork.
      return ran >= 1
    return (ran / seen) >= min_pass_rate

  def known_routes(self, layout_signature: str,
                   limit: int = 4) -> list[tuple[str, str]]:
    """(control name, destination description) pairs remembered for this screen.

    Retrieval from the store rather than from the episode graph, because the
    episode graph is empty when it matters: measured 2026-09-11 over 2327 real
    steps, the current screen was in the store on **86%** of them and had at
    least one known transition on **55%**, against 32% for the within-episode
    graph - and that 32% was computed on the finished graph, not the snapshot
    the step actually saw.

    Deliberately unranked. Ranking the candidates by embedding similarity to
    the task scored 40% top-1 against a 38% random baseline, because a screen
    carries 1.3 known transitions on average - there is almost nothing to rank,
    and the encoder call buys two points.

    Two filters, both because a wrong fact is worse than no fact:
    a control with no text or content-description is rendered from its
    resource id and reads as "android.widget.TextView", and a destination with
    no description cannot be stated at all.
    """
    screen = self.screens.get(layout_signature)
    if screen is None:
      return []
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for known in screen.transitions.values():
      name = _control_display_name(known.control_key)
      if not name:
        continue
      dst = self.screens.get(known.dst_layout_signature or "")
      description = (dst.description if dst is not None else "").strip()
      if not description:
        continue
      key = (name.casefold(), description.casefold())
      if key in seen:
        continue
      seen.add(key)
      out.append((name, description))
    return out[:limit]

  def recall_description(self, layout_signature: str) -> str:
    """What a previous task said this screen was, or "".

    The episode graph starts empty every task, so a screen 149 tasks have
    already stood on still arrives nameless and stays that way until this
    episode's model writes a <THINK> on it - which 42.5% of steps do not.
    Reading the description back at upsert time makes it available *before*
    the step's inference, which is the only point at which exploration can
    use it.
    """
    screen = self.screens.get(layout_signature)
    return screen.description if screen is not None else ""

  def seed_graph(self, graph: ProgressiveBeliefGraph, node_id: str,
                 activity: str, layout_signature: str) -> int:
    """Give the episode's graph what is already known about this screen.

    Adds the remembered transitions as SPECULATIVE edges - the status that
    means "an action is possible here", never "this is the action to take".
    Nothing that the skip gate reads is seeded: execution counts stay zero and
    the node's visit_count is untouched, so a screen known from a previous
    task still has to be revisited *in this episode* before anything may be
    replayed without the model.

    What this does buy is the distiller: a screen the agent is standing on for
    the first time can already carry "Details was observed to lead to {Phone,
    Address}", which is exactly the kind of fact that was almost never
    available inside a single episode.
    """
    screen = self.screens.get(layout_signature)
    if screen is None:
      return 0
    seeded = 0
    for known in screen.transitions.values():
      if not known.dst_layout_signature:
        continue
      if f"{activity}|{known.probe_type}" in self.blocked_contexts:
        continue
      dst_id = ProgressiveBeliefGraph.make_node_id(
          known.dst_activity or activity, known.dst_layout_signature)
      # A seeded destination was, until now, a bare hash: no description and
      # only whatever labels the probe happened to see on the way past. Those
      # were the one bucket of nodes at 0% description coverage (7/7 blank,
      # 2026-09-10) and they are exactly the nodes the distiller injects, so
      # the briefing could only ever say "leads to <hash>". The remembered
      # screen has both, written by whichever task last stood on it.
      remembered = self.screens.get(known.dst_layout_signature)
      dst_labels = known.discovered_labels
      if remembered is not None and not dst_labels:
        dst_labels = remembered.labels
      # upsert_node replaces the record wholesale, so a node this episode has
      # already described must keep that description rather than take "".
      existing = graph.nodes.get(dst_id)
      summary = (existing.semantic_summary if existing else "") or (
          remembered.description if remembered is not None else "")
      graph.upsert_node(GraphNode(
          node_id=dst_id, activity=known.dst_activity or activity,
          package=self.package,
          visual_signature="", structural_signature="",
          layout_signature=known.dst_layout_signature,
          salient_ui_labels=dst_labels,
          semantic_summary=summary,
          status=NodeStatus.SPECULATIVE,
      ))
      action = dict(known.action)
      action["control_key"] = known.control_key
      action["probe_type"] = known.probe_type
      edge = graph.add_speculative_transition(
          node_id, action, dst_id,
          path_probability=0.3, confidence=0.15,
          expected_information_gain=0.0,
          risk_level="LOW" if (known.rollback_success_rate or 0) >= 0.8 else "UNKNOWN",
          exploration_cost=0.0,
          rollback_success=(known.rollback_failure_count == 0),
          discovered_labels=known.discovered_labels,
          inverse_level=known.inverse_level,
      )
      # Remembered, not observed this episode. The probe counter records that
      # somebody paid for this knowledge; the execution counters stay at zero
      # because no model in this episode has endorsed it.
      edge.status = EdgeStatus.SPECULATIVE
      edge.probe_count = known.probe_count
      seeded += 1
    return seeded

  def already_probed(self, layout_signature: str) -> set[str]:
    """control_keys on this screen whose destination is already known."""
    screen = self.screens.get(layout_signature)
    if screen is None:
      return set()
    return {t.control_key for t in screen.transitions.values()
            if t.dst_layout_signature}

  # -- persistence --------------------------------------------------------

  def to_dict(self) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "package": self.package,
        "tasks_recorded": self.tasks_recorded,
        "blocked_contexts": sorted(self.blocked_contexts),
        "blocked_elements": sorted(self.blocked_elements),
        "inverse_levels": dict(self.inverse_levels),
        "screens": [
            {
                "activity": s.activity,
                "layout_signature": s.layout_signature,
                "labels": list(s.labels),
                "description": s.description,
                "tasks_seen": s.tasks_seen,
                "last_seen": s.last_seen,
                "transitions": [dataclasses.asdict(t) for t in s.transitions.values()],
            }
            for s in self.screens.values()
        ],
    }

  @classmethod
  def from_dict(cls, payload: Mapping[str, Any]) -> "AppMemory":
    memory = cls(str(payload.get("package", "")))
    memory.tasks_recorded = int(payload.get("tasks_recorded", 0))
    memory.blocked_contexts = set(payload.get("blocked_contexts") or ())
    memory.blocked_elements = set(payload.get("blocked_elements") or ())
    memory.inverse_levels = dict(payload.get("inverse_levels") or {})
    for raw in payload.get("screens") or ():
      screen = RememberedScreen(
          activity=str(raw.get("activity", "")),
          layout_signature=str(raw.get("layout_signature", "")),
          labels=tuple(raw.get("labels") or ()),
          description=str(raw.get("description") or ""),
          tasks_seen=int(raw.get("tasks_seen", 0)),
          last_seen=float(raw.get("last_seen", 0.0)),
      )
      for t in raw.get("transitions") or ():
        transition = RememberedTransition(
            control_key=str(t.get("control_key", "")),
            probe_type=str(t.get("probe_type", "")),
            action=dict(t.get("action") or {}),
            dst_layout_signature=str(t.get("dst_layout_signature", "")),
            dst_activity=str(t.get("dst_activity", "")),
            discovered_labels=tuple(t.get("discovered_labels") or ()),
            probe_count=int(t.get("probe_count", 0)),
            executed_count=int(t.get("executed_count", 0)),
            tasks_executed=int(t.get("tasks_executed", 0)),
            goals=tuple(str(g) for g in (t.get("goals") or ()) if g),
            rollback_success_count=int(t.get("rollback_success_count", 0)),
            rollback_failure_count=int(t.get("rollback_failure_count", 0)),
            inverse_level=str(t.get("inverse_level", "")),
            last_seen=float(t.get("last_seen", 0.0)),
        )
        screen.transitions[f"{transition.control_key}\0{transition.probe_type}"] = transition
      memory.screens[screen.layout_signature] = screen
    return memory


class AppMemoryStore:
  """One JSON file per package, under a directory the caller controls.

  Per package rather than one big file so that a corrupted or stale memory
  for one app cannot take the others down with it, and so an ablation can
  delete a single app's history.
  """

  def __init__(self, root: str | pathlib.Path):
    self.root = pathlib.Path(root)
    self._loaded: dict[str, AppMemory] = {}

  @staticmethod
  def _safe(package: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in package)[:120]

  def get(self, package: str) -> AppMemory:
    if package in self._loaded:
      return self._loaded[package]
    path = self.root / f"{self._safe(package)}.json"
    memory = AppMemory(package)
    if path.exists():
      try:
        memory = AppMemory.from_dict(json.loads(path.read_text(encoding="utf-8")))
      except (OSError, ValueError, KeyError):
        # A memory that will not parse is worth less than the run it would
        # break; start this package over rather than fail the episode.
        memory = AppMemory(package)
    self._loaded[package] = memory
    return memory

  def save_all(self) -> list[pathlib.Path]:
    self.root.mkdir(parents=True, exist_ok=True)
    written = []
    for package, memory in self._loaded.items():
      memory.tasks_recorded += 1
      path = self.root / f"{self._safe(package)}.json"
      path.write_text(json.dumps(memory.to_dict(), ensure_ascii=False, indent=1),
                      encoding="utf-8")
      written.append(path)
    return written

  def stats(self) -> dict[str, Any]:
    return {
        "packages": len(self._loaded),
        "screens": sum(len(m.screens) for m in self._loaded.values()),
        "transitions": sum(len(s.transitions) for m in self._loaded.values()
                           for s in m.screens.values()),
        "blocked_contexts": sum(len(m.blocked_contexts) for m in self._loaded.values()),
        "blocked_elements": sum(len(m.blocked_elements) for m in self._loaded.values()),
        "inverse_levels": sum(len(m.inverse_levels) for m in self._loaded.values()),
    }
