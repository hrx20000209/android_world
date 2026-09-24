import json

from android_world.parallel_exploration.semantic_prefix_memory import SemanticPrefixMemory


def _row(*, source="Main", target="Settings", element=None, recovery=True, notes="", stable=True):
  element = element or {
      "text": "Settings", "content_desc": "", "resource_id": "app:id/settings",
      "class": "android.widget.Button", "bounds": [10, 20, 110, 80],
  }
  return {
      "depth": 1, "app_package": "com.example", "notes": notes,
      "recovery_ok": recovery, "recovery_level": "BACK_N" if recovery else "",
      "timings_ms": {"recovery": 10, "total": 20},
      "sig_details": {"stable_structure": stable}, "element": element,
      "discovered": {"reached_activity": f"com.example/.{target}", "new_texts": [target]},
      "graph": {
          "src": {"activity": f"com.example/.{source}", "layout_signature": f"layout-{source}"},
          "dst": {"activity": f"com.example/.{target}", "layout_signature": f"layout-{target}"},
          "action": {"action_type": "CLICK"},
      },
  }


def test_three_consistent_landings_are_executable_and_persist(tmp_path):
  path = tmp_path / "prefix.json"
  memory = SemanticPrefixMemory(path, mode="assist")
  for _ in range(3):
    memory.record_probe_row(_row())
  memory.save()
  restored = SemanticPrefixMemory(path, mode="assist")
  source = next(iter(restored.states))
  routes = restored.candidate_routes(source, "open settings")
  assert routes
  assert routes[0].edges[0].confidence >= 0.82
  assert routes[0].edges[0].consistent_landing_count == 3
  assert restored.summary()["node_count"] == 2


def test_selector_relocation_and_ambiguous_selector():
  memory = SemanticPrefixMemory(mode="assist")
  for _ in range(3):
    memory.record_probe_row(_row())
  source = next(state_id for state_id, state in memory.states.items() if state.activity.endswith(".Main"))
  edge = memory.candidate_routes(source, "settings")[0].edges[0]
  relocated = memory.relocate(edge, [{
      "text": "Settings", "content_desc": "", "resource_id": "app:id/settings",
      "class": "android.widget.Button", "bounds": [100, 200, 300, 300],
  }])
  assert relocated["x"] == 200 and relocated["y"] == 250
  assert memory.relocate(edge, [
      {"text": "Settings", "resource_id": "app:id/settings", "class": "android.widget.Button", "bounds": [0, 0, 10, 10]},
      {"text": "Settings", "resource_id": "app:id/settings", "class": "android.widget.Button", "bounds": [20, 20, 30, 30]},
  ]) is None
  assert memory.metrics.selector_ambiguities


def test_trap_dynamic_and_low_confidence_are_rejected():
  memory = SemanticPrefixMemory(mode="assist")
  memory.record_probe_row(_row(recovery=False, notes="RESTORE_FAILED"))
  memory.record_probe_row(_row(element={"text": "Settings", "resource_id": "app:id/dynamic", "class": "android.widget.Button", "bounds": [0, 0, 1, 1]}, stable=False))
  source = next(state_id for state_id, state in memory.states.items() if state.activity.endswith(".Main"))
  assert memory.candidate_routes(source, "settings") == []


def test_route_hit_miss_parse_error_and_backward_compatible_payload(tmp_path):
  memory = SemanticPrefixMemory(mode="assist")
  for _ in range(3):
    memory.record_probe_row(_row())
  source = next(state_id for state_id, state in memory.states.items() if state.activity.endswith(".Main"))
  route = memory.candidate_routes(source, "settings")[0]
  memory.record_route_result(route, hit=True, reason="verified")
  memory.record_route_result(route, hit=False, reason="wrong_landing")
  memory.record_parse_error("missing action", {"action_type": "wait"})
  assert memory.metrics.route_hits == 1
  assert memory.metrics.route_misses == 1
  assert memory.metrics.parse_error_count == 1
  path = tmp_path / "v0.json"
  payload = memory.to_dict()
  payload.pop("mode", None)
  path.write_text(json.dumps(payload), encoding="utf-8")
  restored = SemanticPrefixMemory(path)
  assert len(restored.edges) == 1


def test_authoritative_transition_confirms_only_matching_explored_landing():
  memory = SemanticPrefixMemory(mode="assist")
  for _ in range(3):
    memory.record_probe_row(_row())
  source = next(state_id for state_id, state in memory.states.items()
                if state.activity.endswith(".Main"))
  target = next(state_id for state_id, state in memory.states.items()
                if state.activity.endswith(".Settings"))
  element = _row()["element"]
  edge = memory.confirm_authoritative_transition(
      source_state_id=source, target_state_id=target,
      raw_element=element, action={"action_type": "click"})
  assert edge is not None
  assert edge.route_hit_count == 1
  assert memory.confirm_authoritative_transition(
      source_state_id=source, target_state_id="wrong",
      raw_element=element, action={"action_type": "click"}) is None
