import dataclasses
import json

from android_world.parallel_exploration.executable_memory import ActionFusion
from android_world.parallel_exploration.executable_memory import EdgeEvidence
from android_world.parallel_exploration.executable_memory import ElementSelector
from android_world.parallel_exploration.executable_memory import ExecutableExplorationMemory
from android_world.parallel_exploration.executable_memory import ExecutableMemoryConfig
from android_world.parallel_exploration.executable_memory import PageObservation
from android_world.parallel_exploration.executable_memory import ProbeCandidate
from android_world.parallel_exploration.executable_memory import SafeProbePlanner
from android_world.parallel_exploration.executable_memory import TransitionRecord
from android_world.parallel_exploration.executable_memory import extract_k_step_samples
from android_world.parallel_exploration.executable_memory import goal_semantic_tokens
from android_world.parallel_exploration.executable_memory import selector_for_action
from android_world.parallel_exploration.executable_memory import _tokens
from android_world.parallel_exploration.executable_memory import _route_relevance_tokens


def _elements(text="Inbox"):
  return [{
      "text": text,
      "content_description": "",
      "resource_id": "app:id/inbox",
      "class_name": "android.widget.Button",
      "bbox": (10, 20, 110, 80),
      "is_clickable": True,
      "is_enabled": True,
      "is_visible": True,
  }]


def _page(text="Inbox"):
  return PageObservation.from_ui(
      _elements(text), package="app", activity="app/.Main", screen_size=(1000, 1000))


def test_dynamic_content_does_not_create_graph_node_explosion():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  first = memory.observe_page(_page("12:01"))
  second = memory.observe_page(_page("12:02"))
  assert first.node_id == second.node_id
  assert len(memory.states) == 1


def test_stable_heading_separates_same_control_skeleton_without_dynamic_text():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, graph_prompt_enabled=True))

  def page(title):
    return PageObservation.from_ui([
        {"text": title, "resource_id": "app:id/title",
         "class_name": "android.widget.TextView", "bbox": (20, 80, 900, 180)},
        {"text": "Search", "content_description": "Search",
         "resource_id": "app:id/search", "class_name": "android.widget.Button",
         "bbox": (800, 80, 960, 180), "is_clickable": True},
        {"text": "Sort", "resource_id": "app:id/sort",
         "class_name": "android.widget.Button",
         "bbox": (600, 80, 760, 180), "is_clickable": True},
        {"text": "", "resource_id": "app:id/list",
         "class_name": "android.widget.ScrollView", "bbox": (0, 180, 1080, 2200),
         "is_scrollable": True},
    ], package="app", activity="app/.Main", screen_size=(1080, 2400))

  home = page("My Tasks")
  search_results = page("Matching October 15 2023")
  next_day_results = page("Matching October 16 2023")
  assert home.structural_signature == search_results.structural_signature
  assert "Matching October 15 2023" not in json.dumps(search_results.to_dict())

  home_node = memory.observe_page(home)
  result_node = memory.observe_page(search_results)
  assert home_node.node_id != result_node.node_id
  assert result_node.node_id == memory.observe_page(next_day_results).node_id
  assert len(memory.states) == 2

  memory.record_transition(TransitionRecord(
      source=home, action={"action_type": "click"},
      destination=search_results,
      selector=ElementSelector(resource_id="app:id/search", text="Search"),
      function="click Search reveals search field", meaningful=True,
      recovered=True, latency_s=0.2,
  ))
  assert memory.retrieve_paths(search_results, "what tasks are due") == []
  assert memory.prompt_context(search_results, "what tasks are due") == ""


def test_exploration_guidance_matches_page_and_scores_seen_task_relevant_control():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, override_confidence=0.70))
  source = PageObservation.from_ui([
      {"text": "Notes", "resource_id": "app:id/notes",
       "class_name": "android.widget.Button", "bbox": (10, 20, 180, 90),
       "is_clickable": True},
      {"text": "Settings", "resource_id": "app:id/settings",
       "class_name": "android.widget.Button", "bbox": (200, 20, 370, 90),
       "is_clickable": True},
  ], package="app", activity="app/.Main", screen_size=(1000, 2000))
  destination = PageObservation.from_ui([
      {"text": "New note", "resource_id": "app:id/new_note",
       "class_name": "android.widget.Button", "bbox": (10, 20, 180, 90),
       "is_clickable": True},
  ], package="app", activity="app/.Notes", screen_size=(1000, 2000))
  selector = ElementSelector(
      resource_id="app:id/notes", text="Notes",
      class_name="android.widget.Button")
  for _ in range(2):
    edge = memory.record_transition(TransitionRecord(
        source=source, action={"action_type": "click"},
        destination=destination, selector=selector, meaningful=True,
        recovered=True, no_op=False,
    ))

  # Runtime activity shorthand and live dynamic text do not split the node.
  live_page = PageObservation.from_ui([
      {"text": "Notes", "resource_id": "app:id/notes",
       "class_name": "android.widget.Button", "bbox": (12, 21, 182, 91),
       "is_clickable": True},
      {"text": "Settings", "resource_id": "app:id/settings",
       "class_name": "android.widget.Button", "bbox": (201, 21, 371, 91),
       "is_clickable": True},
      {"text": "Today 09:41", "resource_id": "app:id/clock",
       "class_name": "android.widget.TextView", "bbox": (0, 0, 100, 30)},
  ], package="app", activity="app/app.Main", screen_size=(1000, 2000))
  before_visits = {node_id: node.visit_count for node_id, node in memory.states.items()}
  guidance = memory.exploration_guidance(live_page, "Open Notes")

  assert guidance["state_matched"]
  assert guidance["controls"][selector.key()]["support_count"] == 2
  assert guidance["controls"][selector.key()]["mature"]
  assert guidance["controls"][selector.key()]["task_relevance"] > 0
  assert before_visits == {node_id: node.visit_count for node_id, node in memory.states.items()}
  assert memory.metrics.exploration_guidance_state_matches == 1


def test_exploration_guidance_matches_same_structure_after_labels_drift():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))

  def tab_page(labels):
    return PageObservation.from_ui([
        {"text": label, "resource_id": f"app:id/tab_{index}",
         "class_name": "android.widget.Button",
         "bbox": (index * 150, 20, index * 150 + 140, 90),
         "is_clickable": True}
        for index, label in enumerate(labels)
    ], package="app", activity="app/.Tabs", screen_size=(900, 1800))

  source = tab_page(("Notes", "Tasks", "Settings"))
  destination = PageObservation.from_ui([
      {"text": "New note", "resource_id": "app:id/new_note",
       "class_name": "android.widget.Button", "bbox": (20, 100, 200, 170),
       "is_clickable": True}
  ], package="app", activity="app/.Notes", screen_size=(900, 1800))
  selector = ElementSelector(
      resource_id="app:id/tab_0", text="Notes",
      class_name="android.widget.Button")
  memory.record_transition(TransitionRecord(
      source=source, action={"action_type": "click"}, destination=destination,
      selector=selector, meaningful=True, recovered=True, no_op=False,
  ))

  # The app may localize or rename labels while keeping the exact same
  # resource-id interaction skeleton. Retrieval should keep the stable route.
  live_page = tab_page(("Journal", "Reminders", "Preferences"))
  guidance = memory.exploration_guidance(live_page, "Open a note")

  assert guidance["state_matched"]
  assert selector.key() in guidance["controls"]


def test_state_match_ignores_accessibility_provider_wrapper_noise():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  reasoner_page = PageObservation.from_ui([
      {"text": "Create contact", "resource_id": "app:id/create",
       "content_description": "Create contact", "class_name": "ImageButton",
       "bbox": (800, 1600, 940, 1740), "is_clickable": True},
      {"text": "Contacts", "resource_id": "app:id/contacts",
       "class_name": "FrameLayout", "bbox": (0, 1700, 330, 1850),
       "is_clickable": True},
  ], package="app", activity="app/.People", screen_size=(1000, 1900))
  # The second provider exposes Android/framework wrappers and extra
  # non-interactive descendants, but the live actionable selectors agree.
  shadow_page = PageObservation.from_ui([
      {"text": "status bar", "resource_id": "com.android.systemui:id/status_bar",
       "class_name": "FrameLayout", "bbox": (0, 0, 1000, 60)},
      {"text": "Create", "resource_id": "app:id/create",
       "content_description": "Create contact", "class_name": "ImageButton",
       "bbox": (802, 1603, 942, 1742), "is_clickable": True},
      {"text": "Contacts", "resource_id": "app:id/contacts",
       "class_name": "FrameLayout", "bbox": (0, 1702, 330, 1852),
       "is_clickable": True},
      {"text": "decorative wrapper", "resource_id": "app:id/wrapper",
       "class_name": "ViewGroup", "bbox": (0, 60, 1000, 1850)},
  ], package="app", activity="app/app.People", screen_size=(1000, 1900))
  memory.observe_page(reasoner_page)

  guidance = memory.exploration_guidance(shadow_page, "Create a contact")

  assert reasoner_page.structural_signature != shadow_page.structural_signature
  assert guidance["state_matched"]


def test_arbitrary_app_data_text_is_dynamic_not_a_stable_landmark():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  def page(text):
    return PageObservation.from_ui([{
        "text": text, "resource_id": "app:id/row_title",
        "class_name": "android.widget.TextView", "bbox": (10, 20, 300, 80),
        "is_clickable": True,
    }], package="app", activity="app/.Main")
  quinoa = page("Quinoa Salad")
  salad = page("Cheesy Veggie Scramble")
  assert "text|Quinoa Salad" not in quinoa.landmarks
  assert quinoa.dynamic_signature != salad.dynamic_signature
  assert memory.observe_page(quinoa).node_id == memory.observe_page(salad).node_id
  assert len(memory.states) == 1


def test_route_relevance_separates_create_intent_from_save_destination():
  goal = "Go to the new contact screen and enter the following details"
  query = _route_relevance_tokens(goal)
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  source = PageObservation.from_ui([
      {"text": "Create contact", "resource_id": "app:id/create",
       "class_name": "Button", "bbox": (0, 0, 100, 100), "is_clickable": True},
      {"text": "Save", "resource_id": "app:id/save",
       "class_name": "Button", "bbox": (100, 0, 200, 100), "is_clickable": True},
  ], package="app", activity="app/.Main")
  editor = PageObservation.from_ui(_elements("First name"),
                                    package="app", activity="app/.Editor")
  details = PageObservation.from_ui(_elements("Contact details"),
                                     package="app", activity="app/.Details")
  create_selector = ElementSelector(
      resource_id="app:id/create", text="Create contact", class_name="Button")
  save_selector = ElementSelector(
      resource_id="app:id/save", text="Save", class_name="Button")
  for selector, destination, function in (
      (create_selector, editor, "click Create contact reveals contact editor"),
      (save_selector, details, "click Save reveals Contact details"),
  ):
    memory.record_transition(TransitionRecord(
        source, {"action_type": "click"}, destination, selector,
        function=function, meaningful=True, recovered=True,
    ))
  guidance = memory.exploration_guidance(source, goal)
  create_relevance = guidance["controls"][create_selector.key()]["task_relevance"]
  save_relevance = guidance["controls"][save_selector.key()]["task_relevance"]

  assert {"contact", "create", "navigate", "enter"} <= query
  assert create_relevance > save_relevance
  assert save_relevance < ExecutableMemoryConfig().min_task_relevance


def test_question_style_information_goal_retrieves_search_route():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  source = PageObservation.from_ui([{
      "text": "Search", "content_description": "Search",
      "resource_id": "app:id/search", "class_name": "Button",
      "bbox": (10, 10, 100, 100), "is_clickable": True,
  }], package="app", activity="app/.Main")
  destination = PageObservation.from_ui([{
      "text": "Search results", "resource_id": "app:id/results",
      "class_name": "TextView", "bbox": (10, 10, 300, 80),
  }, {
      "text": "Delete", "resource_id": "app:id/delete",
      "class_name": "Button", "bbox": (300, 10, 390, 80),
      "is_clickable": True,
  }], package="app", activity="app/.Search")
  selector = ElementSelector(resource_id="app:id/search", text="Search",
                             class_name="Button")
  memory.record_transition(TransitionRecord(
      source, {"action_type": "click"}, destination, selector,
      function="click Search reveals Delete", meaningful=True,
      recovered=True,
  ))

  goal = "What quantity of buckwheat groats do I need for the recipe Lasagna?"
  assert "search" in _route_relevance_tokens(goal)
  context = memory.prompt_context(source, goal)
  assert "Search" in context
  assert memory.metrics.prompt_context_count == 1


def test_clicking_scroll_container_is_not_retrieved_as_navigation_route():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  source = PageObservation.from_ui([{
      "resource_id": "app:id/contact_editor_scroller",
      "class_name": "android.widget.ScrollView",
      "bbox": (0, 0, 400, 800), "is_clickable": True,
      "is_scrollable": True,
  }], package="app", activity="app/.Editor")
  destination = PageObservation.from_ui(_elements("Contact editor"),
                                         package="app", activity="app/.Editor")
  memory.record_transition(TransitionRecord(
      source, {"action_type": "click"}, destination,
      ElementSelector(resource_id="app:id/contact_editor_scroller",
                      class_name="android.widget.ScrollView"),
      function="click contact_editor_scroller reveals contact editor",
      meaningful=True, recovered=True,
  ))

  assert memory.retrieve_paths(source, "Create a contact draft") == []
  assert memory.prompt_context(source, "Create a contact draft") == ""


def test_authoritative_action_is_not_mislabeled_as_verified_recovery():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  source = PageObservation.from_ui([{
      "text": "Create contact", "content_description": "Create contact",
      "resource_id": "app:id/create", "class_name": "Button",
      "bbox": (10, 10, 100, 100), "is_clickable": True,
  }, {
      "resource_id": "app:id/content", "class_name": "ViewGroup",
      "bbox": (0, 0, 400, 800),
  }], package="app", activity="app/.Main")
  destination = PageObservation.from_ui(_elements("First name"),
                                         package="app", activity="app/.Edit")
  edge = memory.record_authoritative(
      source, {"action_type": "click", "x": 50, "y": 50}, destination)

  assert edge.support_count == 1
  assert edge.recovery_success_count == 0
  assert edge.reversible_rate == 0.0
  memory.record_task_outcome(success=True, steps=2)
  assert edge.task_success_count == 1
  assert edge.reversible_rate == 0.0
  assert memory.metrics.recovery_failures == 0


def test_probe_budget_and_completion_are_local_to_each_round():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, max_probes=2))

  def candidate(name):
    selector = ElementSelector(resource_id=f"app:id/{name}", text=name,
                               class_name="Button")
    return ProbeCandidate(
        selector=selector, action={"action_type": "click"}, safe=True)

  # One probe in each of two pages must not look like a completed two-probe
  # page merely because the global total happens to be divisible by the cap.
  memory.begin_round()
  assert memory.plan_probes([candidate("one")]).candidate is not None
  memory.begin_round()
  assert memory.plan_probes([candidate("two")]).candidate is not None
  assert memory.metrics.probe_rounds_complete == 0

  memory.begin_round()
  assert memory.plan_probes([candidate("three")]).candidate is not None
  assert memory.plan_probes([candidate("four")]).candidate is not None
  assert memory.metrics.probe_rounds_complete == 1
  assert memory.plan_probes([candidate("five")]).reason == "budget_exhausted"


def test_recovery_failure_metric_counts_only_attempted_probe_recovery():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  source = PageObservation.from_ui([{
      "text": "Menu", "resource_id": "app:id/menu", "class_name": "Button",
      "bbox": (10, 10, 100, 100), "is_clickable": True,
  }], package="app", activity="app/.Main")
  destination = PageObservation.from_ui(_elements("Menu page"),
                                         package="app", activity="app/.Menu")
  memory.record_transition(TransitionRecord(
      source, {"action_type": "click"}, destination,
      ElementSelector(resource_id="app:id/menu", text="Menu", class_name="Button"),
      meaningful=True, recovered=False, recovery_attempted=True, trap=True,
  ))
  assert memory.metrics.recovery_failures == 1


def test_two_click_noops_are_negative_evidence_but_scroll_noops_are_direction_ambiguous():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  source = PageObservation.from_ui([{
      "text": "Filter", "resource_id": "app:id/filter", "class_name": "Button",
      "bbox": (10, 10, 100, 100), "is_clickable": True,
  }, {
      "resource_id": "app:id/list", "class_name": "ScrollView",
      "bbox": (0, 100, 400, 800), "is_scrollable": True,
  }], package="app", activity="app/.Main")
  selectors = (
      (ElementSelector(resource_id="app:id/filter", text="Filter", class_name="Button"), "click"),
      (ElementSelector(resource_id="app:id/list", class_name="ScrollView"), "swipe"),
  )
  for selector, action_type in selectors:
    for _ in range(2):
      memory.record_transition(TransitionRecord(
          source, {"action_type": action_type}, source, selector,
          meaningful=False, recovered=True, no_op=True,
      ))

  guidance = memory.exploration_guidance(source, "Find a note")
  click_row = guidance["controls"][selectors[0][0].key()]
  scroll_row = guidance["controls"][selectors[1][0].key()]
  assert click_row["known_noop"]
  assert not scroll_row["known_noop"]


def test_high_confidence_skip_requires_recovery_and_rejects_unsafe_edges():
  config = ExecutableMemoryConfig(
      enabled=True, graph_enabled=True, high_confidence_skip_enabled=True,
      override_confidence=0.82)
  memory = ExecutableExplorationMemory(config)
  source = PageObservation.from_ui([{
      "text": "Notes", "resource_id": "app:id/notes",
      "class_name": "Button", "bbox": (10, 10, 100, 100),
      "is_clickable": True,
  }], package="app", activity="app/.Main")
  destination = PageObservation.from_ui(_elements("Note list"),
                                         package="app", activity="app/.Notes")
  selector = ElementSelector(resource_id="app:id/notes", text="Notes",
                             class_name="Button")
  for _ in range(4):
    edge = memory.record_transition(TransitionRecord(
        source=source, action={"action_type": "click"}, destination=destination,
        selector=selector, function="click Notes opens Notes", meaningful=True,
        recovered=True, recovery_attempted=True, no_op=False,
    ))
  assert edge.confidence >= 0.82
  assert memory.high_confidence_edge(source, "Open Notes") is edge

  unsafe = None
  for _ in range(4):
    unsafe = memory.record_transition(TransitionRecord(
        source=source, action={"action_type": "click"}, destination=destination,
        selector=ElementSelector(resource_id="app:id/delete", text="Delete",
                                 class_name="Button"),
        function="click Delete removes contact", meaningful=True,
        recovered=True, recovery_attempted=True, no_op=False,
    ))
  assert unsafe is not None and unsafe.confidence >= 0.82
  # Even when its evidence is numerically mature, an unsafe control is never
  # offered as a graph shortcut.
  assert memory.high_confidence_edge(source, "Open Notes") is edge


def test_selector_relocates_after_layout_shift_and_rejects_ambiguity():
  selector = ElementSelector(resource_id="app:id/save", class_name="Button")
  live = [{
      "resource_id": "app:id/save", "class_name": "Button", "text": "Save",
      "bbox": (100, 200, 220, 260), "is_enabled": True, "is_visible": True,
  }]
  result = selector.relocate(live)
  assert result.center == (160, 230)
  assert not result.ambiguous

  ambiguous = dataclasses.replace(selector, resource_id="", text="Save")
  result = ambiguous.relocate(live + [dict(live[0], bbox=(400, 200, 520, 260))])
  assert result.ambiguous
  assert result.center is None


def test_selector_rejects_duplicate_resource_id_even_when_id_matches():
  selector = ElementSelector(resource_id="app:id/item", class_name="TextView")
  live = [
      {"resource_id": "app:id/item", "class_name": "TextView", "text": "A",
       "bbox": (10, 10, 110, 60), "is_enabled": True, "is_visible": True},
      {"resource_id": "app:id/item", "class_name": "TextView", "text": "B",
       "bbox": (10, 80, 110, 130), "is_enabled": True, "is_visible": True},
  ]
  result = selector.relocate(live)
  assert result.ambiguous and result.center is None


def test_transition_selector_picks_smallest_clickable_not_full_page_parent():
  page = PageObservation.from_ui([
      {"resource_id": "app:id/root", "class_name": "FrameLayout",
       "bbox": (0, 0, 1000, 2000), "is_clickable": True},
      {"resource_id": "app:id/bottom_nav", "class_name": "FrameLayout",
       "bbox": (0, 1800, 1000, 2000), "is_clickable": True},
      {"resource_id": "app:id/tab_stopwatch", "text": "Stopwatch",
       "class_name": "TextView", "bbox": (700, 1800, 1000, 2000),
       "is_clickable": True},
  ], package="app", activity="app/.Main", screen_size=(1000, 2000))
  selector = selector_for_action({"x": 800, "y": 1900}, page)
  assert selector.resource_id == "app:id/tab_stopwatch"


def test_resource_id_tokenization_splits_identifier_words():
  assert "stopwatch" in _tokens("com.example:id/stopwatch_time_text")
  assert "time" in _tokens("stopwatch_time_text")


def test_failed_recovery_is_a_permanent_trap_and_lowers_value():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  source, destination = _page("Open"), _page("Detail")
  selector = ElementSelector(resource_id="app:id/inbox", class_name="android.widget.Button")
  edge = memory.record_transition(TransitionRecord(
      source=source, action={"action_type": "click"}, destination=destination,
      selector=selector, meaningful=True, recovered=False, trap=True,
  ))
  assert edge.trap_count == 1
  assert edge.confidence == 0.0
  candidate = ProbeCandidate(selector, {"action_type": "click"}, trap=True)
  decision = SafeProbePlanner(ExecutableMemoryConfig(enabled=True)).choose([candidate])
  assert decision.candidate is None
  assert decision.reason == "no_safe_candidate"


def test_low_support_and_high_entropy_edges_can_be_reexplored_but_same_round_cannot_repeat():
  config = ExecutableMemoryConfig(enabled=True, max_probes=5)
  planner = SafeProbePlanner(config)
  candidate = ProbeCandidate(
      ElementSelector(resource_id="app:id/menu"), {"action_type": "click"},
      task_relevance=.8, support_count=1, destination_entropy=.8,
      reversible_known=True,
  )
  first = planner.choose([candidate], used_budget=0)
  second = planner.choose([candidate], already_selected=[candidate.selector.key()], used_budget=1)
  assert first.candidate is candidate
  assert second.candidate is None
  assert second.rejected[candidate.selector.key()] == "duplicate_selector_this_round"


def test_k_step_slicing_keeps_hard_negative_and_evidence():
  first, second, third = _page("A"), _page("B"), _page("C")
  selector = ElementSelector(resource_id="app:id/next")
  trace = [
      TransitionRecord(first, {"action_type": "click"}, second, selector,
                       recovered=True, meaningful=True, latency_s=.2),
      TransitionRecord(second, {"action_type": "click"}, third, selector,
                       recovered=False, trap=True, meaningful=False, latency_s=.3),
  ]
  samples = extract_k_step_samples(trace, k_max=2)
  assert [sample.k for sample in samples] == [1, 2, 1]
  assert samples[0].meaningful
  assert samples[1].hard_negative
  assert samples[1].evidence["trap"]


def test_probe_ingestion_populates_k_step_memory():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  row = {
      "trial_id": "t-step0", "probe_idx": 0, "depth": 1,
      "recovery_ok": True, "notes": "", "timings_ms": {"total": 10},
      "element": {"text": "Settings", "resource_id": "app:id/settings",
                  "class": "android.widget.TextView", "bounds": [1, 2, 30, 40]},
      "discovered": {"new_element_count": 1, "new_texts": ["Preferences"]},
      "graph": {
          "src": {"activity": "app/.Main", "layout_signature": "main"},
          "dst": {"activity": "app/.Settings", "layout_signature": "settings"},
          "action": {"action_type": "CLICK"},
      },
  }
  memory.ingest_probe_row(row)
  assert len(memory.samples) == 1
  assert memory.samples[0].k == 1


def test_probe_state_with_control_snapshot_matches_live_reasoning_page():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  controls = [
      {"text": "Settings", "resource_id": "app:id/settings",
       "class_name": "android.widget.TextView", "bounds": [10, 20, 180, 80],
       "is_clickable": True},
      {"text": "Search", "resource_id": "app:id/search",
       "class_name": "android.widget.Button", "bounds": [200, 20, 350, 80],
       "is_clickable": True},
  ]
  row = {
      "trial_id": "settings-step0", "probe_idx": 0, "depth": 1,
      "recovery_ok": True, "timings_ms": {"total": 10},
      "element": {"text": "Settings", "resource_id": "app:id/settings",
                  "class": "android.widget.TextView", "bounds": [10, 20, 180, 80]},
      "discovered": {"new_texts": ["Preferences"]},
      "graph": {
          "src": {"activity": "app/.Main", "layout_signature": "probe-layout",
                  "elements": controls},
          "dst": {"activity": "app/.Settings", "layout_signature": "settings-layout",
                  "elements": controls},
          "action": {"action_type": "CLICK"},
      },
  }
  edge = memory.ingest_probe_row(row)
  live_page = PageObservation.from_ui(
      controls, package="app", activity="app/.Main", screen_size=(400, 800))
  assert memory.observe_page(live_page, visited=False).node_id == edge.source_node
  assert memory.retrieve_paths(live_page, "open Preferences")


def test_probe_ingestion_keeps_selector_and_semantic_landing_delta():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  row = {
      "trial_id": "t-step0", "probe_idx": 0, "depth": 1,
      "recovery_ok": True, "timings_ms": {"total": 20},
      "element": {"text": "Search", "resource_id": "app:id/search",
                  "class": "android.widget.TextView", "bounds": [1, 2, 30, 40]},
      "discovered": {"new_element_count": 2, "new_texts": ["Contacts", "15:32"]},
      "graph": {
          "src": {"activity": "app/.Main", "layout_signature": "main"},
          "dst": {"activity": "app/.Contacts", "layout_signature": "contacts"},
          "action": {"action_type": "CLICK"},
      },
  }
  edge = memory.ingest_probe_row(row)
  assert edge is not None
  assert "search" in edge.function.casefold()
  assert "contacts" in edge.function.casefold()
  assert "15:32" not in edge.function


def test_prompt_retrieves_bounded_task_relevant_paths():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, graph_prompt_enabled=True, max_depth=3))
  source = PageObservation.from_ui(
      _elements("Search"), package="app", activity="app/.Main")
  destination = PageObservation.from_ui(
      _elements("Contacts"), package="app", activity="app/.Contacts")
  memory.record_transition(TransitionRecord(
      source=source, action={"action_type": "click"}, destination=destination,
      selector=ElementSelector(resource_id="app:id/search", text="Search"),
      function="click Search opens Contacts", meaningful=True, recovered=True,
      latency_s=0.2,
  ))
  context = memory.prompt_context(source, "open Contacts")
  assert "Route 1" in context
  assert "Search" in context and "Contacts" in context
  assert memory.metrics.prompt_context_count == 1
  assert memory.metrics.prompt_context_edges == 1


def test_advisory_prompt_threshold_is_separate_from_execution_threshold():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, graph_prompt_enabled=True,
      min_task_relevance=0.70, prompt_min_task_relevance=0.30))
  source = PageObservation.from_ui(
      _elements("Privacy dashboard"), package="app", activity="app/.Main")
  destination = PageObservation.from_ui(
      _elements("Privacy dashboard"), package="app", activity="app/.Privacy")
  memory.record_transition(TransitionRecord(
      source=source, action={"action_type": "click"}, destination=destination,
      selector=ElementSelector(resource_id="app:id/inbox", text="Privacy dashboard"),
      function="open privacy dashboard", meaningful=True, recovered=True,
      latency_s=0.2,
  ))

  # Goal overlap is enough to provide weak advisory context, but not enough to
  # make this edge eligible through the execution/fusion retrieval path.
  assert memory.retrieve_paths(source, "open security dashboard") == []
  context = memory.prompt_context(source, "open security dashboard")
  assert "Route 1" in context
  assert "Privacy dashboard" in context
  assert memory.high_confidence_edge(source, "open security dashboard") is None


def test_authoritative_text_input_is_redacted_and_never_a_route():
  field_before = [{
      "text": "", "resource_id": "app:id/first_name",
      "class_name": "android.widget.EditText", "bbox": (10, 20, 300, 80),
      "is_editable": True,
  }]
  field_after = [{
      **field_before[0], "text": "Hugo",
  }]
  source = PageObservation.from_ui(
      field_before, package="app", activity="app/.Contact")
  destination = PageObservation.from_ui(
      field_after, package="app", activity="app/.Contact")
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  edge = memory.record_authoritative(
      source, {"action_type": "input_text", "text": "Hugo"}, destination)
  serialized = json.dumps(memory.to_dict(), ensure_ascii=False)
  assert "Hugo" not in serialized
  assert edge.dynamic
  assert edge.selector.dynamic_text
  assert memory.prompt_context(source, "enter Hugo") == ""
  assert source.landmarks == destination.landmarks


def test_task_values_are_redacted_from_later_non_editable_landing_pages():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  source = PageObservation.from_ui([
      {"resource_id": "app:id/name", "class_name": "EditText",
       "is_editable": True, "is_clickable": True, "bbox": (0, 0, 200, 60)},
  ], package="app", activity="app/.Contact")
  memory.record_authoritative(
      source, {"action_type": "input_text", "text": "Hugo"}, source)
  landing = PageObservation.from_ui([
      {"resource_id": "app:id/contact_name", "text": "Hugo Pereira",
       "class_name": "TextView", "is_clickable": True, "bbox": (0, 0, 200, 60)},
      {"resource_id": "app:id/phone", "text": "+1 392-074-1751",
       "class_name": "TextView", "is_clickable": True, "bbox": (0, 80, 200, 140)},
  ], package="app", activity="app/.Contact")
  memory.observe_page(landing)
  serialized = json.dumps(memory.to_dict(), ensure_ascii=False)
  assert "Hugo" not in serialized
  assert "Pereira" not in serialized
  assert "13920741751" not in serialized


def test_probe_ingestion_redacts_task_literals_from_deltas():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  row = {
      "task": "Create a new contact for Hugo Pereira. Their number is +13920741751.",
      "trial_id": "contact-step1", "probe_idx": 0, "depth": 1,
      "recovery_ok": True, "timings_ms": {"total": 12},
      "element": {"text": "Contacts", "resource_id": "app:id/contacts",
                  "class": "TextView", "bounds": [1, 2, 30, 40]},
      "discovered": {"new_texts": ["Hugo Pereira", "+1 392-074-1751", "Contacts"]},
      "graph": {
          "src": {"activity": "app/.Main", "layout_signature": "main"},
          "dst": {"activity": "app/.Contacts", "layout_signature": "contacts"},
          "action": {"action_type": "CLICK"},
      },
  }
  edge = memory.ingest_probe_row(row)
  serialized = json.dumps(memory.to_dict(), ensure_ascii=False)
  assert "Hugo" not in serialized
  assert "Pereira" not in serialized
  assert "13920741751" not in serialized
  assert "contacts" in edge.function.casefold()


def test_probe_ingestion_redacts_unclassified_dynamic_control_from_all_memory():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  row = {
      "task": "Mark a note as todo",
      "trial_id": "dynamic-note-list", "probe_idx": 0, "depth": 1,
      "recovery_ok": True, "timings_ms": {"total": 12},
      "element": {"text": "Emergency Fund Progress",
                  "class": "android.view.ViewGroup",
                  "bounds": [100, 200, 300, 260]},
      "discovered": {"new_texts": ["Emergency Fund Progress", "New note"]},
      "graph": {
          "src": {"activity": "app/.Notes", "layout_signature": "notes"},
          "dst": {"activity": "app/.Notes", "layout_signature": "notes"},
          "action": {"action_type": "CLICK",
                     "element_identity":
                         "|Emergency Fund Progress||android.view.ViewGroup|"
                         "(100, 200, 300, 260)"},
      },
  }

  edge = memory.ingest_probe_row(row)

  serialized = json.dumps(memory.to_dict(), ensure_ascii=False)
  assert edge is not None and edge.dynamic and edge.selector.dynamic_text
  assert "Emergency Fund Progress" not in serialized
  # Stable Android/app navigation concepts survive the conservative filter.
  assert "New note" in serialized


def test_goal_tokens_focus_on_task_object_not_personal_values_or_verbs():
  tokens = goal_semantic_tokens(
      "Create a new contact for Hugo Pereira. Their number is +13920741751.")
  assert tokens == {"contact"}
  assert goal_semantic_tokens("Mark the Notes item as todo") == {"note", "task"}
  assert goal_semantic_tokens(
      "Is the note titled To-Do List marked as a todo item") == {"note", "task"}
  assert goal_semantic_tokens(
      "Go to the new contact screen and enter the following details: "
      "First Name: Grace, Last Name: Adams, Phone: 784-622-3532, "
      "Phone Label: Work. Do NOT hit save.") == {"contact"}


def test_contact_creation_route_is_retrieved_for_field_entry_task():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, graph_prompt_enabled=True, max_depth=3))
  source = PageObservation.from_ui(
      _elements("Contacts"), package="app", activity="app/.Main")
  destination = PageObservation.from_ui(
      _elements("New contact"), package="app", activity="app/.Editor")
  memory.record_transition(TransitionRecord(
      source=source, action={"action_type": "click"}, destination=destination,
      selector=ElementSelector(resource_id="app:id/add_contact", text="Create contact"),
      function="click Create contact opens the new contact editor",
      meaningful=True, recovered=True,
  ))

  context = memory.prompt_context(
      source,
      "Go to the new contact screen and enter the following details: "
      "First Name: Grace, Last Name: Adams, Phone: 784-622-3532, "
      "Phone Label: Work. Do NOT hit save.")

  assert "Route 1" in context
  assert "Create contact" in context


def test_same_activity_tabs_are_not_merged_by_shared_toolbar_only():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  def page(activity_labels):
    return PageObservation.from_ui([
        {"resource_id": resource_id, "text": text,
         "class_name": "android.widget.Button", "bbox": (i * 120, 10, i * 120 + 100, 60),
         "is_clickable": True}
        for i, (resource_id, text) in enumerate(activity_labels)
    ], package="app", activity="app/.Main")
  notes = page([
      ("app:id/toolbar", "Notes"), ("app:id/search", "Search"),
      ("app:id/new_note", "New note"),
  ])
  tasks = page([
      ("app:id/toolbar", "Tasks"), ("app:id/task_filter", "Filter"),
      ("app:id/task_checkbox", "Grocery Trip"),
  ])
  edge = memory.record_transition(TransitionRecord(
      source=notes, action={"action_type": "click"}, destination=tasks,
      selector=ElementSelector(resource_id="app:id/tasks", text="Tasks", class_name="Button"),
      function="click Tasks opens todo items", meaningful=True, recovered=True,
  ))
  target = next(iter(edge.target_states))
  assert target != edge.source_node
  assert memory.retrieve_paths(notes, "open tasks")


def test_android_component_shorthand_matches_live_activity_for_graph_retrieval():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  controls = [{
      "resource_id": "app:id/create", "text": "Create contact",
      "class_name": "android.widget.Button", "bbox": (10, 20, 200, 90),
      "is_clickable": True,
  }]
  probe_page = PageObservation.from_ui(
      controls, package="app", activity="app/.MainActivity", screen_size=(400, 800))
  live_page = PageObservation.from_ui(
      controls, package="app", activity="app/app.MainActivity", screen_size=(400, 800))
  destination = PageObservation.from_ui(
      [{"resource_id": "app:id/editor", "text": "Contact editor",
        "class_name": "android.widget.EditText", "is_editable": True,
        "bbox": (10, 100, 390, 170)}],
      package="app", activity="app/.EditorActivity", screen_size=(400, 800))
  memory.record_transition(TransitionRecord(
      source=probe_page, action={"action_type": "click"}, destination=destination,
      selector=ElementSelector(resource_id="app:id/create", text="Create contact",
                               class_name="android.widget.Button"),
      function="click Create contact opens contact editor",
      meaningful=True, recovered=True,
  ))

  assert probe_page.activity == live_page.activity == "app/app.MainActivity"
  assert memory.observe_page(live_page, visited=False).node_id == memory.observe_page(
      probe_page, visited=False).node_id
  assert memory.retrieve_paths(live_page, "create contact")


def test_legacy_short_activity_is_migrated_without_changing_edge_node_ids(tmp_path):
  path = tmp_path / "legacy-memory.json"
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  page = PageObservation.from_ui(
      [{"resource_id": "app:id/settings", "text": "Settings",
        "class_name": "android.widget.Button", "bbox": (0, 0, 100, 50),
        "is_clickable": True}],
      package="app", activity="app/app.MainActivity", screen_size=(400, 800))
  old_node_id = memory.observe_page(page).node_id
  payload = memory.to_dict()
  payload["states"][0]["activity"] = "app/.MainActivity"
  payload["states"][0]["node_id"] = "legacy-node-id"
  path.write_text(json.dumps(payload), encoding="utf-8")

  restored = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True), path=path)

  assert restored.states["legacy-node-id"].activity == "app/app.MainActivity"
  assert restored.observe_page(page, visited=False).node_id == "legacy-node-id"
  assert old_node_id != "legacy-node-id"


def test_retrieval_filters_task_relevance_before_top_k():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, top_k_paths=1, min_task_relevance=0.30))
  def page(text, resource_id):
    return PageObservation.from_ui([{
        "resource_id": resource_id, "text": text, "class_name": "Button",
        "bbox": (10, 20, 110, 80), "is_clickable": True,
    }], package="app", activity="app/.Main")
  source = page("Home", "app:id/home")
  irrelevant = memory.record_transition(TransitionRecord(
      source=source, action={"action_type": "click"}, destination=page("Privacy", "app:id/privacy"),
      selector=ElementSelector(resource_id="app:id/settings", text="Settings"),
      function="click Settings opens privacy dashboard", meaningful=True, recovered=True,
  ))
  relevant = memory.record_transition(TransitionRecord(
      source=source, action={"action_type": "click"}, destination=page("Todo", "app:id/todo"),
      selector=ElementSelector(resource_id="app:id/notes", text="Notes"),
      function="click Notes opens Todo list", meaningful=True, recovered=True,
  ))
  irrelevant.support_count = 20
  irrelevant.recovery_success_count = 20
  irrelevant.target_states = {"privacy": 20}
  relevant.support_count = 1
  relevant.recovery_success_count = 1
  relevant.target_states = {"todo": 1}
  paths = memory.retrieve_paths(source, "Open Notes todo", top_k=1)
  assert len(paths) == 1
  assert paths[0][0].edge_id == relevant.edge_id


def test_prompt_adoption_counts_only_a_live_selector_hit():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  source = PageObservation.from_ui([{
      "resource_id": "app:id/notes", "text": "Notes", "class_name": "Button",
      "bbox": (10, 20, 210, 100), "is_clickable": True,
  }], package="app", activity="app/.Main", screen_size=(400, 800))
  destination = PageObservation.from_ui([{
      "resource_id": "app:id/todo", "text": "Todo", "class_name": "TextView",
      "bbox": (10, 20, 210, 100), "is_clickable": True,
  }], package="app", activity="app/.Todo")
  memory.record_transition(TransitionRecord(
      source=source, action={"action_type": "click"}, destination=destination,
      selector=ElementSelector(resource_id="app:id/notes", text="Notes", class_name="Button"),
      function="click Notes opens Todo", meaningful=True, recovered=True,
  ))
  assert "Route 1" in memory.prompt_context(source, "Open Notes Todo")
  assert memory.record_prompt_action({"action_type": "click", "x": 80, "y": 60}, source)
  assert not memory.record_prompt_action({"action_type": "click", "x": 350, "y": 750}, source)
  assert memory.metrics.prompt_adoptions == 1


def test_refresh_merges_monotonic_metrics_from_shared_graph_writers(tmp_path):
  path = tmp_path / "shared-memory.json"
  explorer = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True), path=path)
  explorer.metrics.probes_completed = 3
  explorer.save()
  agent = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True), path=path)
  agent.metrics.prompt_adoptions = 2
  agent.save()
  explorer.refresh()
  assert explorer.metrics.probes_completed == 3
  assert explorer.metrics.prompt_adoptions == 2


def test_route_retrieval_does_not_repeat_self_loop_edges():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  page = _page("Settings")
  memory.record_transition(TransitionRecord(
      source=page, action={"action_type": "click"}, destination=page,
      selector=ElementSelector(resource_id="app:id/settings", text="Settings"),
      function="click Settings opens Settings", meaningful=True, recovered=True,
  ))
  assert memory.retrieve_paths(page, "open Settings") == []


def test_unsafe_action_edges_are_not_retrieved_as_routes():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True))
  source = PageObservation.from_ui([{
      "resource_id": "app:id/call", "text": "Call", "class_name": "Button",
      "bbox": (10, 20, 110, 80), "is_clickable": True,
  }], package="app", activity="app/.Main")
  destination = PageObservation.from_ui([{
      "resource_id": "app:id/details", "text": "Contact info",
      "class_name": "TextView", "bbox": (10, 20, 210, 100),
  }], package="app", activity="app/.Contact")
  memory.record_transition(TransitionRecord(
      source=source, action={"action_type": "click"}, destination=destination,
      selector=ElementSelector(resource_id="app:id/call", text="Call", class_name="Button"),
      function="click Call starts a telephone call", meaningful=True, recovered=True,
  ))
  assert memory.retrieve_paths(source, "call contact") == []


def test_persistence_is_versioned_and_restores_edges(tmp_path):
  path = tmp_path / "memory.json"
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True), path=path)
  memory.record_transition(TransitionRecord(
      _page("A"), {"action_type": "click"}, _page("B"),
      ElementSelector(resource_id="app:id/inbox"), meaningful=True,
  ))
  memory.save()
  restored = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True), path=path)
  json_version = restored.to_dict()["schema_version"]
  assert json_version == 3
  assert len(restored.edges) == 1
  loaded_edge = next(iter(restored.edges.values()))
  assert loaded_edge.normalized_action_token
  loaded_node = restored.states[loaded_edge.target_states.most_common(1)[0][0]]
  assert loaded_node.parent_state_ids == (loaded_edge.source_node,)
  assert loaded_node.incoming_action_tokens


def test_schema_v1_migration_removes_inference_from_recovery_evidence(tmp_path):
  path = tmp_path / "legacy_memory.json"
  payload = {
      "schema_version": 1,
      "states": [],
      "edges": [{
          "edge_id": "legacy", "source_node": "source", "selector": {},
          "action_type": "click", "function": "click Notes",
          "target_states": {"destination": 2}, "support_count": 2,
          "alpha": 3, "beta": 1, "meaningful_count": 2,
          "recovery_success_count": 2, "recovery_failure_count": 0,
          "provenance": {"inference": 2},
      }],
  }
  path.write_text(json.dumps(payload), encoding="utf-8")

  restored = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True), path=path)

  assert restored.edges["legacy"].recovery_success_count == 0
  assert restored.edges["legacy"].reversible_rate == 0.0


def test_schema_v2_migration_defaults_unified_route_fields(tmp_path):
  path = tmp_path / "legacy_v2.json"
  path.write_text(json.dumps({
      "schema_version": 2,
      "states": [{
          "node_id": "old", "package": "app", "activity": "app/.Main",
          "structural_signature": "layout", "semantic_signature": "semantic",
      }],
      "edges": [{
          "edge_id": "old-edge", "source_node": "old", "selector": {},
          "action_type": "click", "target_states": {"old": 1},
      }],
  }), encoding="utf-8")

  restored = ExecutableExplorationMemory(ExecutableMemoryConfig(enabled=True), path=path)

  node = restored.states["old"]
  edge = restored.edges["old-edge"]
  assert node.parent_state_ids == () and node.depth == 0
  assert edge.route_hit_count == 0 and edge.route_miss_count == 0


def test_unified_graph_route_requires_repeated_reversible_unambiguous_evidence():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, high_confidence_skip_enabled=True,
      override_confidence=0.82))
  source = PageObservation.from_ui([{
      "text": "Notes", "resource_id": "app:id/notes",
      "class_name": "Button", "bbox": (10, 10, 100, 100),
      "is_clickable": True,
  }], package="app", activity="app/.Main")
  destination = PageObservation.from_ui(_elements("Note list"),
                                         package="app", activity="app/.Notes")
  selector = ElementSelector(resource_id="app:id/notes", text="Notes",
                             class_name="Button")
  for _ in range(4):
    edge = memory.record_transition(TransitionRecord(
        source=source, action={"action_type": "click"}, destination=destination,
        selector=selector, function="click Notes opens Notes", meaningful=True,
        recovered=True, recovery_attempted=True, no_op=False,
    ))
  assert memory.high_confidence_path(source, "Open Notes") == [edge]
  memory.record_route_result([edge], hit=True, reason="verified")
  assert edge.route_hit_count == 1
  assert memory.metrics.route_hit_count == 1

  memory.record_route_result([edge], hit=False, failed_edge=edge,
                             reason="wrong_landing")
  assert edge.route_miss_count == 1
  assert memory.metrics.route_miss_count == 1
  assert memory.high_confidence_path(source, "Open Notes") is None


def test_beta_value_and_fusion_override_gate():
  selector = ElementSelector(resource_id="app:id/menu", class_name="Button",
                             bbox=(10, 20, 110, 80))
  edge = EdgeEvidence(
      edge_id="e", source_node="s", selector=selector, action_type="click",
      function="open menu", support_count=3, alpha=4, beta=1,
      target_states={"d": 3}, recovery_success_count=3,
  )
  assert edge.confidence > .5
  config = ExecutableMemoryConfig(enabled=True, post_fusion_enabled=True,
                                  override_confidence=.5)
  page = PageObservation.from_ui([{
      "resource_id": "app:id/menu", "class_name": "Button", "text": "Menu",
      "bbox": (10, 20, 110, 80), "is_enabled": True, "is_visible": True,
  }], package="app", activity="app/.Main")
  decision = ActionFusion(config).fuse(
      {"action_type": "swipe", "x": 900, "y": 900}, page, [edge],
      task_relevance=lambda _: 1.0)
  assert decision.graph_overrode
  assert decision.action["action_type"] == "click"


def test_post_fusion_keeps_a_point_already_inside_the_graph_target():
  """The tap at (205, 305) is inside the live Menu button, so it already
  reaches the control the graph knows. Snapping it to the center moved taps
  on position-sensitive views (a calendar month grid) onto other days."""
  old = ElementSelector(resource_id="app:id/menu", class_name="Button", bbox=(10, 20, 110, 80))
  edge = EdgeEvidence(
      edge_id="e", source_node="s", selector=old, action_type="click",
      function="open menu", support_count=2, alpha=3, beta=1,
      target_states={"d": 2}, recovery_success_count=2,
  )
  page = PageObservation.from_ui([{
      "resource_id": "app:id/menu", "class_name": "Button", "text": "Menu",
      "bbox": (200, 300, 320, 380), "is_clickable": True,
      "is_enabled": True, "is_visible": True,
  }], package="app", activity="app/.Main")
  decision = ActionFusion(ExecutableMemoryConfig(enabled=True)).fuse(
      {"action_type": "click", "x": 205, "y": 305}, page, [edge])
  assert not decision.coordinate_corrected
  assert decision.action == {"action_type": "click", "x": 205, "y": 305}
  assert decision.source == "graph_consistent"


def test_post_fusion_does_not_snap_to_full_page_container():
  page = PageObservation.from_ui([{
      "resource_id": "app:id/workspace", "class_name": "ScrollView",
      "bbox": (0, 0, 1080, 2400), "is_clickable": True,
      "is_enabled": True, "is_visible": True,
  }], package="app", activity="app/.Main", screen_size=(1080, 2400))
  action = {"action_type": "click", "x": 839, "y": 1763}

  decision = ActionFusion(ExecutableMemoryConfig(enabled=True)).fuse(
      action, page, [])

  assert not decision.coordinate_corrected
  assert decision.action == action


def test_post_fusion_accepts_actions_without_coordinates():
  page = _page("Inbox")
  action = {"action_type": "wait", "x": None, "y": None}
  decision = ActionFusion(ExecutableMemoryConfig(enabled=True)).fuse(
      action, page, [])
  assert decision.source == "reasoning"
  assert decision.action == action


def _notes_route_pages():
  source = PageObservation.from_ui([{
      "text": "Notes", "resource_id": "app:id/notes",
      "class_name": "Button", "bbox": (10, 10, 100, 100),
      "is_clickable": True,
  }], package="app", activity="app/.Main")
  destination = PageObservation.from_ui(_elements("Note list"),
                                         package="app", activity="app/.Notes")
  return source, destination


def _walk_notes(memory, source, destination, times):
  """The agent itself taking the edge: no recovery is ever attempted."""
  edge = None
  for _ in range(times):
    edge = memory.record_transition(TransitionRecord(
        source=source, action={"action_type": "click"}, destination=destination,
        selector=ElementSelector(resource_id="app:id/notes", text="Notes",
                                 class_name="Button"),
        function="click Notes opens Notes", meaningful=True,
        recovered=False, recovery_attempted=False, no_op=False,
        provenance="inference"))
  return edge


def test_execution_edge_is_blocked_by_reversibility_under_the_strict_gate():
  """Why the full run logged 2893 route candidates and 0 shortcut attempts:
  an edge the agent walked has no recovery record, so the strict gate reads
  it as irreversible no matter how many times it landed consistently."""
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, high_confidence_skip_enabled=True,
      override_confidence=0.82))
  source, destination = _notes_route_pages()
  edge = _walk_notes(memory, source, destination, 4)
  assert memory.route_block_reason([edge]) == "reversibility_unproven"
  assert memory.high_confidence_path(source, "Open Notes") is None
  report = memory.route_gate_report(source, "Open Notes")
  assert report["candidates"] >= 1
  assert report["blocks"].get("reversibility_unproven", 0) >= 1


def test_observed_back_press_is_reversibility_evidence():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, high_confidence_skip_enabled=True,
      override_confidence=0.82))
  source, destination = _notes_route_pages()
  edge = _walk_notes(memory, source, destination, 4)
  assert memory.record_observed_reversal(edge.edge_id)
  assert edge.reversible_rate == 1.0
  assert memory.route_block_reason([edge]) == ""
  assert memory.high_confidence_path(source, "Open Notes") == [edge]
  assert not memory.record_observed_reversal("no-such-edge")


def test_unknown_reversibility_switch_admits_unattempted_but_not_failed_edges():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, high_confidence_skip_enabled=True,
      override_confidence=0.82, route_allow_unknown_reversibility=True))
  source, destination = _notes_route_pages()
  edge = _walk_notes(memory, source, destination, 4)
  assert memory.route_block_reason([edge]) == ""
  # One observed recovery failure is disqualifying even under the switch.
  edge.recovery_failure_count += 1
  assert memory.route_block_reason([edge]) == "reversibility_unproven"


def test_unknown_reversibility_switch_survives_config_round_trip():
  config = ExecutableMemoryConfig.from_mapping(
      {"enabled": True, "route_allow_unknown_reversibility": 1})
  assert config.route_allow_unknown_reversibility is True
  assert ExecutableMemoryConfig().route_allow_unknown_reversibility is False


def test_activity_landing_level_tolerates_a_split_dialog_node():
  """One dialog recorded as two nodes (dialog-only vs dialog-over-activity
  dumps) halves node-level stability; the activity is the same both times."""
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, high_confidence_skip_enabled=True, override_confidence=0.82,
      route_allow_unknown_reversibility=True, route_landing_level="activity"))
  source, _ = _notes_route_pages()
  selector = ElementSelector(resource_id="app:id/notes", text="Notes", class_name="Button")
  dialog = [{"text": "OK", "resource_id": "app:id/ok", "class_name": "Button",
             "bbox": (300, 900, 500, 1000), "is_clickable": True},
            {"text": "Cancel", "resource_id": "app:id/cancel", "class_name": "Button",
             "bbox": (600, 900, 800, 1000), "is_clickable": True}]
  behind = [{"text": "", "resource_id": f"app:id/row{i}",
             "class_name": "TextView", "bbox": (0, 100 * i, 1000, 100 * i + 90),
             "is_clickable": True} for i in range(1, 9)]
  pages = {"dialog_only": dialog, "dialog_over_list": dialog + behind}
  edge = None
  for kind in ("dialog_only",) * 4 + ("dialog_over_list",) * 2:
    destination = PageObservation.from_ui(pages[kind], package="app",
                                           activity="app/.Main2")
    edge = memory.record_transition(TransitionRecord(
        source=source, action={"action_type": "click"}, destination=destination,
        selector=selector, function="click Notes opens Notes", meaningful=True,
        recovered=False, recovery_attempted=False, no_op=False, provenance="inference"))
  assert len(edge.target_states) == 2 and edge.ambiguous       # node level: split
  assert not memory.route_edge_ambiguous(edge)                 # activity level: one
  assert memory.route_edge_confidence(edge) >= 0.82
  assert memory.route_block_reason([edge]) == ""
  source_node = memory.observe_page(source, visited=False)
  landed = memory.observe_page(PageObservation.from_ui(
      dialog, package="app", activity="app/.Main2"), visited=False)
  verdict = memory.route_landed(edge, source_node.node_id, landed)
  assert verdict["ok"] and verdict["activity_ok"] and verdict["moved"]
  # A press that changed nothing never counts, even at the activity level.
  stuck = memory.route_landed(edge, landed.node_id, landed)
  assert not stuck["ok"]


def test_node_landing_level_is_the_default_and_unchanged():
  assert ExecutableMemoryConfig().route_landing_level == "node"
  assert ExecutableMemoryConfig.from_mapping(
      {"route_landing_level": "bogus"}).route_landing_level == "node"


def test_one_dynamic_title_does_not_make_a_route_target_dynamic():
  """Spec: dynamic is element-level. Markor's new-file dialog carries the
  file-type label and hint text, which read as task data; under the old
  any-element rule it was a dynamic target and no route into it could pass."""
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, high_confidence_skip_enabled=True, override_confidence=0.82,
      route_allow_unknown_reversibility=True))
  source, _ = _notes_route_pages()
  dialog = PageObservation.from_ui([
      {"text": "Create new file", "resource_id": "", "class_name": "TextView",
       "bbox": (100, 300, 900, 380)},
      {"text": "Markdown", "resource_id": "app:id/type", "class_name": "TextView",
       "bbox": (100, 400, 900, 480)},
      {"text": "OK", "resource_id": "app:id/ok", "class_name": "Button",
       "bbox": (600, 900, 800, 1000), "is_clickable": True},
  ], package="app", activity="app/.Main")
  edge = None
  for _ in range(4):
    edge = memory.record_transition(TransitionRecord(
        source=source, action={"action_type": "click"}, destination=dialog,
        selector=ElementSelector(resource_id="app:id/notes", text="Notes",
                                 class_name="Button"),
        function="click Notes opens dialog", meaningful=True, recovered=False,
        recovery_attempted=False, no_op=False, provenance="inference"))
  target = memory.states[next(iter(edge.target_states))]
  assert any(r.startswith("dynamic_elements:") for r in target.dynamic_reasons)
  assert not target.dynamic_content
  assert memory.route_block_reason([edge]) == ""


def test_a_page_with_no_stable_anchor_is_a_dynamic_target():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, high_confidence_skip_enabled=True, override_confidence=0.82,
      route_allow_unknown_reversibility=True))
  source, _ = _notes_route_pages()
  only_data = PageObservation.from_ui([
      {"text": "John Smith 555-1234", "resource_id": "", "class_name": "TextView",
       "bbox": (100, 300, 900, 380)},
  ], package="app", activity="app/.Main")
  edge = None
  for _ in range(4):
    edge = memory.record_transition(TransitionRecord(
        source=source, action={"action_type": "click"}, destination=only_data,
        selector=ElementSelector(resource_id="app:id/notes", text="Notes",
                                 class_name="Button"),
        function="click Notes", meaningful=True, recovered=False,
        recovery_attempted=False, no_op=False, provenance="inference"))
  target = memory.states[next(iter(edge.target_states))]
  assert not target.landmarks
  assert target.dynamic_content and "no_stable_landmark" in target.dynamic_reasons
  assert memory.route_block_reason([edge]) == "dynamic_target_state"


def _markor_like_memory():
  memory = ExecutableExplorationMemory(ExecutableMemoryConfig(
      enabled=True, graph_prompt_enabled=True, high_confidence_skip_enabled=True,
      override_confidence=0.82, route_allow_unknown_reversibility=True))
  home = PageObservation.from_ui([
      {"text": "", "content_desc": "Create a new file or folder",
       "resource_id": "net.gsantner.markor:id/fab_add_new_item",
       "class_name": "android.widget.ImageButton", "bbox": (900, 2000, 1040, 2140),
       "is_clickable": True},
      {"text": "Files", "resource_id": "net.gsantner.markor:id/nav_files",
       "class_name": "android.widget.TextView", "bbox": (0, 2200, 270, 2300)},
  ], package="net.gsantner.markor", activity="net.gsantner.markor/.activity.MainActivity")
  dialog = PageObservation.from_ui([
      {"text": "Name", "resource_id": "net.gsantner.markor:id/label",
       "class_name": "android.widget.TextView", "bbox": (100, 800, 400, 860)},
      {"text": "OK", "resource_id": "android:id/button1",
       "class_name": "android.widget.Button", "bbox": (700, 1300, 900, 1400),
       "is_clickable": True},
  ], package="net.gsantner.markor", activity="net.gsantner.markor/.activity.MainActivity")
  edge = None
  for _ in range(6):
    edge = memory.record_transition(TransitionRecord(
        source=home, action={"action_type": "click"}, destination=dialog,
        selector=ElementSelector(resource_id="net.gsantner.markor:id/fab_add_new_item",
                                 content_desc="Create a new file or folder",
                                 class_name="android.widget.ImageButton"),
        function="click Create a new file or folder reveals android.widget.Button|android:id/button1",
        meaningful=True, recovered=False, recovery_attempted=False, no_op=False,
        provenance="inference"))
  return memory, home, edge


def test_delete_task_is_not_related_to_a_create_control():
  memory, home, edge = _markor_like_memory()
  query = _route_relevance_tokens("Delete all my notes in Markor.")
  assert memory._edge_task_relevance(edge, query) == 0.0
  assert memory.prompt_context(home, "Delete all my notes in Markor.") == ""


def test_app_name_alone_does_not_make_a_route_relevant():
  memory, _, edge = _markor_like_memory()
  query = _route_relevance_tokens("Open Markor.")
  query.discard("navigate")
  assert memory._edge_task_relevance(edge, query) == 0.0


def test_shortcut_needs_an_object_the_task_names():
  """Intent-only overlap ("add" -> create) may advise the model but must not
  act for it: the recipe task needs recipes.txt opened, not a new file."""
  memory, home, edge = _markor_like_memory()
  goal = "Add the recipes from recipes.txt in Markor to the Broccoli recipe app."
  assert memory.route_block_reason([edge]) == ""
  assert memory.route_task_block([edge], goal) == "task_object_mismatch"
  assert memory.high_confidence_path(home, goal) is None
  report = memory.route_gate_report(home, goal)
  assert report["blocks"].get("task_object_mismatch", 0) >= 1


def test_shortcut_still_passes_when_the_task_names_the_object():
  memory, home, _ = _markor_like_memory()
  path = memory.high_confidence_path(home, "Create a new file in Markor named a.md.")
  assert path is not None
