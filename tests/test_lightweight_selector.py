import json

from android_world.parallel_exploration.executable_memory import ElementSelector
from android_world.parallel_exploration.executable_memory import ExecutableExplorationMemory
from android_world.parallel_exploration.executable_memory import ExecutableMemoryConfig
from android_world.parallel_exploration.executable_memory import PageObservation
from android_world.parallel_exploration.executable_memory import TransitionRecord
from android_world.parallel_exploration.lightweight_selector import CompactGraphSummary
from android_world.parallel_exploration.lightweight_selector import LinearModel
from android_world.parallel_exploration.lightweight_selector import rank_elements


def _page(text):
  return PageObservation.from_ui([{
      "text": text,
      "content_desc": "",
      "resource_id": "app:id/" + text.lower(),
      "class_name": "android.widget.Button",
      "bbox": (10, 20, 110, 80),
      "is_clickable": True,
  }], package="app", activity="app/.Main")


def test_fixed_linear_selector_uses_resource_id_and_multihot(tmp_path):
  weights = tmp_path / "weights.txt"
  weights.write_text("bias 0.1\nsearch 2.0\nsettings 0.5\n", encoding="utf-8")
  model = LinearModel.from_text(weights)
  ranked = rank_elements([
      {"index": 0, "text": "Settings", "resource_id": "app:id/settings"},
      {"index": 1, "text": "", "content_desc": "Search", "resource_id": "app:id/search"},
  ], model)
  assert ranked[0].original_index == 1
  assert ranked[0].score == 2.1


def test_compact_graph_has_bounded_rows_and_current_state_prompt(tmp_path):
  memory = ExecutableExplorationMemory(
      ExecutableMemoryConfig(enabled=True), path=tmp_path / "memory.json")
  source, target = _page("Search"), _page("Results")
  memory.record_transition(TransitionRecord(
      source=source, action={"action_type": "click"}, destination=target,
      selector=ElementSelector(text="Search", resource_id="app:id/search"),
      function="open results", meaningful=True, recovered=True,
  ))
  summary = CompactGraphSummary.from_memory(memory, max_nodes=2, max_edges=2)
  payload = summary.to_dict()
  assert payload["schema_version"] == 1
  assert len(payload["states"]) == 2
  assert len(payload["edges"]) == 1
  source_id = memory.observe_page(source, visited=False).node_id
  prompt = summary.prompt(state_id=source_id)
  assert "GUI-MEMORY-COMPACT" in prompt
  assert "Search" in prompt


def test_save_writes_compact_sidecar(tmp_path):
  path = tmp_path / "memory.json"
  memory = ExecutableExplorationMemory(
      ExecutableMemoryConfig(enabled=True), path=path)
  memory.observe_page(_page("Home"))
  memory.save()
  compact = path.with_name("memory.compact.json")
  assert compact.is_file()
  assert json.loads(compact.read_text(encoding="utf-8"))["encoding"] == "gui-memory-rows-v1"
