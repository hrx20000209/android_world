"""Tests for the cross-task structural memory of an app."""

from __future__ import annotations

import tempfile

from absl.testing import absltest

from android_world.parallel_exploration.app_memory import AppMemory
from android_world.parallel_exploration.app_memory import AppMemoryStore
from android_world.parallel_exploration.app_memory import RememberedScreen
from android_world.parallel_exploration.belief_graph import EdgeStatus
from android_world.parallel_exploration.belief_graph import GraphNode
from android_world.parallel_exploration.belief_graph import NodeStatus
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph

ACT = "com.app/.MainActivity"
SIG = "layout-abc"
CONTROL = "com.app:id/details|Details||android.widget.TextView"


def _memory() -> AppMemory:
  memory = AppMemory("com.app")
  memory.observe_probe(
      activity=ACT, layout_signature=SIG, control_key=CONTROL,
      probe_type="TAP_NAV", action={"action_type": "CLICK", "x": 10, "y": 20},
      dst_activity="com.app/.DetailActivity", dst_layout_signature="layout-detail",
      discovered_labels=("Phone", "Address"), rollback_ok=True,
      inverse_level="INVERSE")
  return memory


def _graph_on(node_id: str) -> ProgressiveBeliefGraph:
  graph = ProgressiveBeliefGraph("task")
  graph.upsert_node(GraphNode(
      node_id=node_id, activity=ACT, package="com.app",
      visual_signature="", structural_signature="", layout_signature=SIG,
      status=NodeStatus.COMMITTED), visited=True)
  return graph


class SeedingTest(absltest.TestCase):

  def test_a_remembered_transition_becomes_a_speculative_edge(self):
    memory = _memory()
    node_id = ProgressiveBeliefGraph.make_node_id(ACT, SIG)
    graph = _graph_on(node_id)
    self.assertEqual(memory.seed_graph(graph, node_id, ACT, SIG), 1)
    edge, = [e for e in graph.edges.values() if e.src_node == node_id]
    self.assertEqual(edge.status, EdgeStatus.SPECULATIVE)
    self.assertEqual(edge.discovered_labels, ("Phone", "Address"))
    self.assertEqual(edge.inverse_level, "INVERSE")

  def test_seeding_never_licenses_a_skip(self):
    """What the app is may cross tasks; what the model decided may not.

    Replaying intent across tasks would feed the one mechanism measured as
    harmful - tasks where skipping fired scored 1/17 against a control's 6/17
    (2026-09-01) - so the fields the skip gate reads must arrive at zero no
    matter how much an earlier task learned.
    """
    memory = _memory()
    for _ in range(10):
      memory.observe_probe(
          activity=ACT, layout_signature=SIG, control_key=CONTROL,
          probe_type="TAP_NAV", action={"action_type": "CLICK", "x": 10, "y": 20},
          dst_activity="com.app/.DetailActivity",
          dst_layout_signature="layout-detail", rollback_ok=True)
    node_id = ProgressiveBeliefGraph.make_node_id(ACT, SIG)
    graph = _graph_on(node_id)
    memory.seed_graph(graph, node_id, ACT, SIG)
    edge, = [e for e in graph.edges.values() if e.src_node == node_id]
    self.assertEqual(edge.execution_hit_count, 0)
    self.assertEqual(edge.skip_attempt_count, 0)
    self.assertEqual(edge.nodes_at_last_execution, 0)
    self.assertEqual(graph.nodes[node_id].visit_count, 1)  # this episode only

  def test_blocked_context_is_not_seeded_back_in(self):
    memory = _memory()
    memory.observe_rollback(ACT, "TAP_NAV", ok=False)
    node_id = ProgressiveBeliefGraph.make_node_id(ACT, SIG)
    graph = _graph_on(node_id)
    self.assertEqual(memory.seed_graph(graph, node_id, ACT, SIG), 0)

  def test_a_screen_never_seen_seeds_nothing(self):
    node_id = ProgressiveBeliefGraph.make_node_id(ACT, "other-layout")
    graph = _graph_on(node_id)
    self.assertEqual(_memory().seed_graph(graph, node_id, ACT, "other-layout"), 0)


class SafetyMemoryTest(absltest.TestCase):

  def test_rollback_failures_and_escapes_accumulate(self):
    memory = AppMemory("com.app")
    memory.observe_rollback(ACT, "TAP_MENU", ok=False)
    memory.observe_rollback(ACT, "TAP_NAV", ok=True, level="BACK_N")
    memory.observe_escape("rid|Share||android.widget.TextView|(0, 0, 1, 1)")
    self.assertIn(f"{ACT}|TAP_MENU", memory.blocked_contexts)
    self.assertEqual(memory.inverse_levels[f"{ACT}|TAP_NAV"], "BACK_N")
    self.assertEqual(len(memory.blocked_elements), 1)

  def test_rollback_rate_tracks_both_outcomes(self):
    memory = _memory()
    memory.observe_probe(
        activity=ACT, layout_signature=SIG, control_key=CONTROL,
        probe_type="TAP_NAV", action={}, rollback_ok=False)
    transition, = memory.screens[SIG].transitions.values()
    self.assertEqual(transition.probe_count, 2)
    self.assertEqual(transition.rollback_success_rate, 0.5)


class PersistenceTest(absltest.TestCase):

  def test_round_trip_through_disk(self):
    with tempfile.TemporaryDirectory() as folder:
      store = AppMemoryStore(folder)
      memory = store.get("com.app")
      memory.observe_probe(
          activity=ACT, layout_signature=SIG, control_key=CONTROL,
          probe_type="TAP_NAV", action={"action_type": "CLICK", "x": 10, "y": 20},
          dst_activity="com.app/.DetailActivity",
          dst_layout_signature="layout-detail",
          discovered_labels=("Phone",), rollback_ok=True, inverse_level="INVERSE")
      memory.observe_rollback(ACT, "TAP_MENU", ok=False)
      store.save_all()

      reloaded = AppMemoryStore(folder).get("com.app")
      self.assertIn(f"{ACT}|TAP_MENU", reloaded.blocked_contexts)
      transition, = reloaded.screens[SIG].transitions.values()
      self.assertEqual(transition.control_key, CONTROL)
      self.assertEqual(transition.discovered_labels, ("Phone",))
      self.assertEqual(transition.inverse_level, "INVERSE")

  def test_a_corrupt_file_starts_that_package_over_rather_than_failing(self):
    with tempfile.TemporaryDirectory() as folder:
      store = AppMemoryStore(folder)
      store.get("com.app")
      store.save_all()
      path = next(iter(store.root.glob("*.json")))
      path.write_text("{not json", encoding="utf-8")
      fresh = AppMemoryStore(folder).get("com.app")
      self.assertEqual(fresh.screens, {})

  def test_packages_are_stored_separately(self):
    with tempfile.TemporaryDirectory() as folder:
      store = AppMemoryStore(folder)
      store.get("com.a").observe_rollback("com.a/.M", "TAP_NAV", ok=False)
      store.get("com.b").observe_rollback("com.b/.M", "SCROLL", ok=False)
      store.save_all()
      self.assertEqual(len(list(store.root.glob("*.json"))), 2)
      reloaded = AppMemoryStore(folder)
      self.assertEqual(reloaded.get("com.a").blocked_contexts, {"com.a/.M|TAP_NAV"})
      self.assertEqual(reloaded.get("com.b").blocked_contexts, {"com.b/.M|SCROLL"})



class ExecutedTransitionTest(absltest.TestCase):
  """Every real step is an observation of what a control does."""

  def test_execution_records_dynamics_without_rollback_statistics(self):
    memory = AppMemory("com.app")
    memory.observe_execution(
        activity=ACT, layout_signature=SIG, control_key=CONTROL,
        action={"action_type": "click", "x": 1, "y": 2},
        dst_activity="com.app/.DetailActivity",
        dst_layout_signature="layout-detail", labels=("Phone",))
    transition, = memory.screens[SIG].transitions.values()
    self.assertEqual(transition.dst_layout_signature, "layout-detail")
    self.assertEqual(transition.discovered_labels, ("Phone",))
    # A real execution is never undone, so it says nothing about whether
    # probing this control could be recovered from.
    self.assertEqual(transition.probe_count, 0)
    self.assertEqual(transition.rollback_success_count, 0)
    self.assertIsNone(transition.rollback_success_rate)

  def test_executed_transitions_seed_the_next_task(self):
    memory = AppMemory("com.app")
    memory.observe_execution(
        activity=ACT, layout_signature=SIG, control_key=CONTROL,
        action={"action_type": "click", "x": 1, "y": 2},
        dst_activity="com.app/.DetailActivity",
        dst_layout_signature="layout-detail", labels=("Phone", "Address"))
    node_id = ProgressiveBeliefGraph.make_node_id(ACT, SIG)
    graph = _graph_on(node_id)
    self.assertEqual(memory.seed_graph(graph, node_id, ACT, SIG), 1)
    edge, = [e for e in graph.edges.values() if e.src_node == node_id]
    self.assertEqual(edge.discovered_labels, ("Phone", "Address"))
    self.assertEqual(edge.execution_hit_count, 0)   # dynamics, never intent

  def test_a_transition_that_changed_nothing_is_not_remembered(self):
    memory = AppMemory("com.app")
    memory.observe_execution(
        activity=ACT, layout_signature=SIG, control_key=CONTROL,
        action={}, dst_activity=ACT, dst_layout_signature="")
    self.assertEqual(memory.screens.get(SIG, RememberedScreen(ACT, SIG)).transitions, {})


class EscapeDestinationTest(absltest.TestCase):
  """Leaving the app is not knowledge about the app."""

  def _escape(self, memory, dst):
    memory.observe_execution(
        activity=ACT, layout_signature=SIG, control_key="rid|Cancel||Button",
        action={"action_type": "click", "x": 1, "y": 2},
        dst_activity=dst, dst_layout_signature="layout-home",
        labels=("Gmail", "Photos", "Sun, Oct 15"))

  def test_a_transition_to_the_launcher_is_not_remembered(self):
    """The fact that made every SMS task carry launcher content in its prompt.

    Learned from an episode stuck in the default-SMS role dialog that pressed
    Cancel to escape; cross-task memory then replayed
    "Verified: Cancel -> {Sun, Oct 15, 0, Gmail, Photos}" into every later SMS
    task (2026-09-02). True, and useless.
    """
    memory = AppMemory("com.app")
    self._escape(memory, "com.google.android.apps.nexuslauncher/.NexusLauncherActivity")
    self.assertEqual(memory.screens.get(SIG, RememberedScreen(ACT, SIG)).transitions, {})

  def test_a_transition_into_a_system_dialog_is_not_remembered(self):
    memory = AppMemory("com.app")
    self._escape(memory, "com.google.android.permissioncontroller/.RequestRoleActivity")
    self.assertEqual(memory.screens.get(SIG, RememberedScreen(ACT, SIG)).transitions, {})

  def test_a_real_cross_app_destination_is_still_remembered(self):
    """A file picker is somewhere the app sends you, and worth knowing."""
    memory = AppMemory("com.app")
    memory.observe_execution(
        activity=ACT, layout_signature=SIG, control_key="rid|Attach||Button",
        action={"action_type": "click", "x": 1, "y": 2},
        dst_activity="com.android.documentsui/.files.FilesActivity",
        dst_layout_signature="layout-picker", labels=("Downloads",))
    transition, = memory.screens[SIG].transitions.values()
    self.assertEqual(transition.dst_layout_signature, "layout-picker")
    self.assertEqual(transition.discovered_labels, ("Downloads",))

  def test_a_probe_that_escaped_keeps_its_rollback_record_but_not_its_labels(self):
    """The probe still happened - what it cost is worth keeping; where it
    landed is not."""
    memory = AppMemory("com.app")
    memory.observe_probe(
        activity=ACT, layout_signature=SIG, control_key=CONTROL,
        probe_type="TAP_NAV", action={}, rollback_ok=False,
        dst_activity="com.google.android.apps.nexuslauncher/.NexusLauncherActivity",
        dst_layout_signature="layout-home", discovered_labels=("Gmail",))
    transition, = memory.screens[SIG].transitions.values()
    self.assertEqual(transition.rollback_failure_count, 1)
    self.assertEqual(transition.discovered_labels, ())
    self.assertEqual(transition.dst_layout_signature, "")

if __name__ == "__main__":
  absltest.main()


# --- screens remember what they are, across tasks ---------------------------
#
# The model's per-episode history resets and drops everything past
# history_limit=8; 48% of episodes exceed it. This does not reset, and 39% of
# screens are visited by more than one task.


def test_screen_description_is_written_once_and_survives_later_visits():
  from android_world.parallel_exploration.app_memory import AppMemory
  m = AppMemory("net.gsantner.markor")
  m.observe_screen("a/A", "L1", description="I see the Markor file list.")
  m.observe_screen("a/A", "L1", description="A later, worse phrasing.")
  assert m.screens["L1"].description == "I see the Markor file list."


def test_a_step_without_reasoning_does_not_blank_an_existing_description():
  from android_world.parallel_exploration.app_memory import AppMemory
  m = AppMemory("net.gsantner.markor")
  m.observe_screen("a/A", "L1", description="I see the Markor file list.")
  m.observe_screen("a/A", "L1")            # bare tool call, no prose
  assert m.screens["L1"].description == "I see the Markor file list."


def test_description_survives_a_save_load_round_trip(tmp_path):
  """It is only worth writing if the next task can still read it."""
  from android_world.parallel_exploration.app_memory import AppMemoryStore
  store = AppMemoryStore(str(tmp_path))
  mem = store.get("net.gsantner.markor")
  mem.observe_screen("a/A", "L1", description="I see a Confirm Delete dialog.")
  store.save_all()
  again = AppMemoryStore(str(tmp_path)).get("net.gsantner.markor")
  assert again.screens["L1"].description == "I see a Confirm Delete dialog."


# Seeded destination nodes were the one bucket at 0% description coverage
# (7/7 blank, measured 2026-09-10 over the first 14 tasks of model116) - and
# they are exactly the nodes the distiller names in an injected briefing.


def test_a_seeded_destination_carries_the_description_a_previous_task_wrote():
  memory = _memory()
  memory.observe_screen("com.app/.DetailActivity", "layout-detail",
                        description="A contact detail page with Phone and Address.")
  graph = _graph_on("src")
  memory.seed_graph(graph, "src", ACT, SIG)
  dst = ProgressiveBeliefGraph.make_node_id("com.app/.DetailActivity",
                                            "layout-detail")
  assert (graph.nodes[dst].semantic_summary
          == "A contact detail page with Phone and Address.")


def test_seeding_never_blanks_a_description_this_episode_already_wrote():
  """upsert_node replaces the record wholesale; the summary must survive."""
  memory = _memory()                      # nothing remembered about the dst
  graph = _graph_on("src")
  dst = ProgressiveBeliefGraph.make_node_id("com.app/.DetailActivity",
                                            "layout-detail")
  graph.upsert_node(GraphNode(
      node_id=dst, activity="com.app/.DetailActivity", package="com.app",
      visual_signature="", structural_signature="",
      layout_signature="layout-detail", status=NodeStatus.COMMITTED,
      semantic_summary="I see the contact's phone number."), visited=True)
  memory.seed_graph(graph, "src", ACT, SIG)
  assert graph.nodes[dst].semantic_summary == "I see the contact's phone number."


def test_a_seeded_destination_falls_back_to_remembered_labels():
  memory = AppMemory("com.app")
  memory.observe_probe(
      activity=ACT, layout_signature=SIG, control_key=CONTROL,
      probe_type="TAP_NAV", action={"action_type": "CLICK", "x": 10, "y": 20},
      dst_activity="com.app/.DetailActivity",
      dst_layout_signature="layout-detail",
      discovered_labels=(), rollback_ok=True, inverse_level="INVERSE")
  memory.observe_screen("com.app/.DetailActivity", "layout-detail",
                        ["Phone", "Address", "Email"])
  graph = _graph_on("src")
  memory.seed_graph(graph, "src", ACT, SIG)
  dst = ProgressiveBeliefGraph.make_node_id("com.app/.DetailActivity",
                                            "layout-detail")
  assert "Phone" in graph.nodes[dst].salient_ui_labels


def test_recall_description_is_empty_for_a_screen_never_seen():
  assert AppMemory("com.app").recall_description("layout-unknown") == ""


def test_recall_description_reads_back_what_a_previous_task_wrote():
  memory = AppMemory("com.app")
  memory.observe_screen(ACT, SIG, description="A file list with Save and Cancel.")
  assert memory.recall_description(SIG) == "A file list with Save and Cancel."


def test_tasks_seen_counts_tasks_not_visits():
  """It was declared, serialised and deserialised but never incremented."""
  from android_world.parallel_exploration.app_memory import AppMemory
  m = AppMemory("com.app")
  m.observe_screen("a/A", "L1")
  m.observe_screen("a/A", "L1")          # same task, second visit
  m.observe_screen("a/A", "L2")
  assert m.screens["L1"].tasks_seen == 1
  assert m.screens["L2"].tasks_seen == 1


def test_a_second_task_increments_the_count(tmp_path):
  """One process is one task, so a reload is the next task starting."""
  from android_world.parallel_exploration.app_memory import AppMemoryStore
  store = AppMemoryStore(str(tmp_path))
  store.get("com.app").observe_screen("a/A", "L1")
  store.save_all()
  again = AppMemoryStore(str(tmp_path))
  again.get("com.app").observe_screen("a/A", "L1")
  assert again.get("com.app").screens["L1"].tasks_seen == 2


# --- retrieval from the store, not from the episode graph --------------------
#
# The episode graph is empty when it matters: 68% of steps stand on a node with
# no outgoing edge. The store had the current screen on 86% of 2327 real steps
# and a known transition on 55%.


def test_a_known_route_is_named_by_its_control_text_and_destination():
  memory = AppMemory("com.app")
  memory.observe_execution(
      activity=ACT, layout_signature=SIG, control_key="id/btn|Details||Button",
      action={"action_type": "click"}, dst_activity="com.app/.DetailActivity",
      dst_layout_signature="layout-detail", labels=("Phone",))
  memory.observe_screen("com.app/.DetailActivity", "layout-detail",
                        description="A contact detail page with Phone.")
  assert memory.known_routes(SIG) == [("Details", "A contact detail page with Phone.")]


def test_a_control_with_only_a_resource_id_is_not_stated():
  """It renders as "android.widget.TextView led to ...": true and useless."""
  memory = AppMemory("com.app")
  memory.observe_execution(
      activity=ACT, layout_signature=SIG,
      control_key="id/list_item|||android.widget.TextView",
      action={"action_type": "click"}, dst_activity="com.app/.DetailActivity",
      dst_layout_signature="layout-detail")
  memory.observe_screen("com.app/.DetailActivity", "layout-detail",
                        description="A detail page.")
  assert memory.known_routes(SIG) == []


def test_a_destination_with_no_description_is_not_stated():
  memory = AppMemory("com.app")
  memory.observe_execution(
      activity=ACT, layout_signature=SIG, control_key="id/btn|Details||Button",
      action={"action_type": "click"}, dst_activity="com.app/.DetailActivity",
      dst_layout_signature="layout-detail")
  assert memory.known_routes(SIG) == []


def test_the_content_description_names_a_control_with_no_text():
  memory = AppMemory("com.app")
  memory.observe_execution(
      activity=ACT, layout_signature=SIG,
      control_key="id/fab||Add recipe|ImageButton",
      action={"action_type": "click"}, dst_activity="com.app/.NewActivity",
      dst_layout_signature="layout-new")
  memory.observe_screen("com.app/.NewActivity", "layout-new",
                        description="A new-recipe form.")
  assert memory.known_routes(SIG)[0][0] == "Add recipe"


def test_the_same_route_is_stated_once():
  memory = AppMemory("com.app")
  for sig in ("layout-a", "layout-b"):
    memory.observe_execution(
        activity=ACT, layout_signature=SIG, control_key=f"id/{sig}|Open||Button",
        action={"action_type": "click"}, dst_activity="com.app/.D",
        dst_layout_signature=sig)
    memory.observe_screen("com.app/.D", sig, description="The same page.")
  assert len(memory.known_routes(SIG)) == 1


def test_a_screen_the_store_has_never_seen_yields_nothing():
  assert AppMemory("com.app").known_routes("layout-unknown") == []


# --- prefill: 这一屏是不是"已决定"的 ------------------------------------------
#
# 唯一性是信号的全部来源：同一屏只有一个被执行过的控件、且 >=2 个任务执行过它时，
# 下一个任务按同一个的比例是 87%（39/45，2026-09-12 实测 116 个任务）；放宽成
# "最常执行的控件"掉到 54-58%，对 2-3 个候选的屏几乎等于瞎猜。


def _executed(memory, control, dst, action=None):
  memory.observe_execution(
      activity=ACT, layout_signature=SIG, control_key=control,
      action=action or {"action_type": "click"},
      dst_activity="com.app/.D", dst_layout_signature=dst)


def test_a_screen_two_tasks_agreed_on_is_decided(tmp_path):
  from android_world.parallel_exploration.app_memory import AppMemoryStore
  store = AppMemoryStore(str(tmp_path))
  _executed(store.get("com.app"), "id/fab|Add||Button", "layout-new")
  store.save_all()
  store = AppMemoryStore(str(tmp_path))          # 下一个任务
  _executed(store.get("com.app"), "id/fab|Add||Button", "layout-new")
  got = store.get("com.app").prefill_control(SIG)
  assert got is not None and got[0] == "id/fab|Add||Button"
  assert got[2] == "layout-new"


def test_one_task_alone_is_not_enough():
  memory = AppMemory("com.app")
  _executed(memory, "id/fab|Add||Button", "layout-new")
  _executed(memory, "id/fab|Add||Button", "layout-new")   # 同一任务按了两次
  assert memory.prefill_control(SIG) is None


def test_a_screen_with_two_known_controls_is_not_decided(tmp_path):
  from android_world.parallel_exploration.app_memory import AppMemoryStore
  store = AppMemoryStore(str(tmp_path))
  _executed(store.get("com.app"), "id/a|Save||Button", "layout-x")
  store.save_all()
  store = AppMemoryStore(str(tmp_path))
  _executed(store.get("com.app"), "id/b|Cancel||Button", "layout-y")
  assert store.get("com.app").prefill_control(SIG) is None


def test_a_probed_but_never_executed_transition_does_not_decide():
  """探测说"这里有这么个动作"，从不说"这是该走的路"。"""
  memory = _memory()          # 只有 observe_probe
  assert memory.prefill_control(SIG) is None


def test_tasks_executed_counts_tasks_not_presses(tmp_path):
  from android_world.parallel_exploration.app_memory import AppMemoryStore
  store = AppMemoryStore(str(tmp_path))
  m = store.get("com.app")
  for _ in range(6):
    _executed(m, "id/fab|Add||Button", "layout-new")
  t, = [x for x in m.screens[SIG].transitions.values() if x.control_key.startswith("id/fab")]
  assert t.executed_count == 6
  assert t.tasks_executed == 1


def test_the_counters_survive_a_save_load_round_trip(tmp_path):
  from android_world.parallel_exploration.app_memory import AppMemoryStore
  store = AppMemoryStore(str(tmp_path))
  _executed(store.get("com.app"), "id/fab|Add||Button", "layout-new")
  store.save_all()
  again = AppMemoryStore(str(tmp_path)).get("com.app")
  t, = [x for x in again.screens[SIG].transitions.values()
        if x.control_key.startswith("id/fab")]
  assert (t.executed_count, t.tasks_executed) == (1, 1)


def _goal_executed(memory, sig, control, goal, dst_sig, dst_activity="com.app/.B"):
  memory.observe_execution(activity="com.app/.A", layout_signature=sig,
                           control_key=control, action={"action_type": "click"},
                           dst_activity=dst_activity, dst_layout_signature=dst_sig,
                           goal=goal)
  # 一个进程就是一个任务，测试里用清空这两个集合来模拟"换了一个任务"
  memory._executed_this_task.clear()  # pylint: disable=protected-access
  memory._seen_this_task.clear()  # pylint: disable=protected-access


def _overlap(goal, candidates):
  words = set(goal.lower().split())
  return [len(words & set(c.lower().split())) / max(len(words), 1)
          for c in candidates]


def test_retrieve_control_separates_tasks_that_share_a_screen():
  """The case a screen-majority vote cannot express.

  Two tasks pressed Add here and one pressed Search. A vote over the screen
  answers "Add" to both kinds of goal; conditioning on the goal answers each
  correctly. Measured over 419 executed steps this is 66% against 84%.
  """
  memory = AppMemory("com.app")
  _goal_executed(memory, "S", "id|Add||B", "Create a new event tomorrow", "D1")
  _goal_executed(memory, "S", "id|Add||B", "Add a calendar event next week", "D1")
  _goal_executed(memory, "S", "id|Search||B", "Find the note about groceries", "D2")
  # 通过率门控在这里关掉：这是一块岔路口（三个任务站过、两种走法），生产配置下
  # 它本来就该被拒——见 test_retrieve_control_refuses_a_fork_screen。这个用例
  # 单独检验的是"以目标为条件能不能区分开"，那是另一道门。
  add = memory.retrieve_control("S", "Add an event on Friday", _overlap,
                                threshold=0.1, margin=0.05, min_pass_rate=0.0)
  find = memory.retrieve_control("S", "Find my note", _overlap,
                                 threshold=0.1, margin=0.05, min_pass_rate=0.0)
  assert add is not None and add[0] == "id|Add||B" and add[4] == "goal"
  assert find is not None and find[0] == "id|Search||B"


def test_retrieve_control_carries_the_destination_activity():
  """The landing check needs the activity, not the layout signature."""
  memory = AppMemory("com.app")
  _goal_executed(memory, "S", "id|Add||B", "Create an event", "D1",
            dst_activity="com.app/.EditActivity")
  _goal_executed(memory, "S", "id|Add||B", "Add an event", "D2",
            dst_activity="com.app/.EditActivity")
  got = memory.retrieve_control("S", "Add a new event", _overlap,
                                threshold=0.1, margin=0.05)
  assert got is not None and got[3] == "com.app/.EditActivity"


def test_retrieve_control_refuses_a_near_tie():
  """A near-tie between two controls is what a screen vote gets wrong."""
  memory = AppMemory("com.app")
  _goal_executed(memory, "S", "id|Add||B", "Create an event now", "D1")
  _goal_executed(memory, "S", "id|New||B", "Create an event soon", "D2")
  assert memory.retrieve_control("S", "Create an event", _overlap,
                                 threshold=0.1, margin=0.9) is None


def test_retrieve_control_survives_a_dead_encoder():
  memory = AppMemory("com.app")
  _goal_executed(memory, "S", "id|Add||B", "Create an event", "D1")
  _goal_executed(memory, "S", "id|New||B", "Find a note", "D2")
  assert memory.retrieve_control("S", "Add one", lambda g, c: None) is None


def test_retrieve_control_never_reads_its_own_goal():
  """The seed corpus is the same 116 tasks; a task must not retrieve itself."""
  memory = AppMemory("com.app")
  _goal_executed(memory, "S", "id|Add||B", "Create an event", "D1")
  assert memory.retrieve_control("S", "Create an event", _overlap,
                                 threshold=0.1, margin=0.05) is None


def test_retrieve_control_refuses_a_fork_screen():
  """A screen many tasks pass through differently is not prefillable.

  Only 43% of screen visits involve a click navigation at all (927 visits over
  112 tasks), so without this gate the retrieval fires on forks and lands the
  right *action* 36% of the time despite a 90% landing-match rate.
  """
  memory = AppMemory("com.app")
  # Six tasks stood on this screen; two of them pressed Add.
  for _ in range(6):
    memory.observe_screen("com.app/.A", "S")
    memory._seen_this_task.clear()  # pylint: disable=protected-access
  memory._seen_this_task.clear()  # pylint: disable=protected-access
  _goal_executed(memory, "S", "id|Add||B", "Create an event tomorrow", "D1")
  _goal_executed(memory, "S", "id|Add||B", "Add an event next week", "D1")
  assert memory.screens["S"].tasks_seen >= 6
  assert memory.retrieve_control("S", "Add an event Friday", _overlap,
                                 threshold=0.1, margin=0.05,
                                 min_pass_rate=0.7) is None
  # Same evidence, gate off: the retrieval itself would have answered.
  got = memory.retrieve_control("S", "Add an event Friday", _overlap,
                                threshold=0.1, margin=0.05, min_pass_rate=0.0)
  assert got is not None and got[0] == "id|Add||B"


def test_retrieve_control_allows_a_pass_through_screen():
  """Every task that stood here pressed the same control."""
  memory = AppMemory("com.app")
  _goal_executed(memory, "S", "id|Next||B", "Create an event tomorrow", "D1")
  _goal_executed(memory, "S", "id|Next||B", "Add an event next week", "D1")
  _goal_executed(memory, "S", "id|Next||B", "Schedule a meeting", "D1")
  got = memory.retrieve_control("S", "Add an event Friday", _overlap,
                                threshold=0.1, margin=0.05, min_pass_rate=0.7)
  assert got is not None and got[0] == "id|Next||B"


def test_pass_rate_excludes_the_asking_task():
  """The gate asks what *other* tasks did here, so this task's own visit is out.

  Two prior tasks stood here and both pressed Next; this task is standing here
  now and has not. Counting its visit would read 2/3 = 67% and refuse.
  """
  memory = AppMemory("com.app")
  _goal_executed(memory, "S", "id|Next||B", "Create an event tomorrow", "D1")
  _goal_executed(memory, "S", "id|Next||B", "Add an event next week", "D1")
  memory.observe_screen("com.app/.A", "S")   # 现在轮到本任务站在这里
  assert memory.screens["S"].tasks_seen == 3
  got = memory.retrieve_control("S", "Add an event Friday", _overlap,
                                threshold=0.1, margin=0.05, min_pass_rate=0.7)
  assert got is not None and got[0] == "id|Next||B"


def test_a_question_goal_gets_no_prefill():
  """store 里只有"做事"的先例，问答类任务没有可回放的动作。

  两轮全量 116 的拆分：执行类任务上本设计与不建图打平（47 对 47、45 对 46），
  问答类 18 个任务上输 4 和 2 —— 全部亏损都在这里。句向量把
  "Do I have any events October 28" 判得紧挨着 "Create an event on October 28"，
  于是检索把新建事件的 FAB 交了回去。
  """
  memory = AppMemory("com.app")
  _goal_executed(memory, "S", "id|New||B", "Create an event on October 28", "D1")
  _goal_executed(memory, "S", "id|New||B", "Add an event next week", "D1")
  asked = "Do I have any events October 28? Answer with the titles only."
  assert memory.retrieve_control("S", asked, _overlap, threshold=0.1,
                                 margin=0.05) is None
  # 同类的执行目标仍然拿得到
  got = memory.retrieve_control("S", "Create an event on Friday", _overlap,
                                threshold=0.1, margin=0.05)
  assert got is not None and got[0] == "id|New||B"


def test_a_question_goal_may_reuse_another_question():
  """门挡的是跨类，不是问答类本身。"""
  memory = AppMemory("com.app")
  _goal_executed(memory, "S", "id|List||B",
                 "What events do I have next week? Answer with the titles.", "D1")
  _goal_executed(memory, "S", "id|List||B",
                 "How many events are there today? Answer with a number.", "D1")
  got = memory.retrieve_control(
      "S", "Do I have any events October 28? Answer with the titles only.",
      _overlap, threshold=0.05, margin=0.02)
  assert got is not None and got[0] == "id|List||B"
