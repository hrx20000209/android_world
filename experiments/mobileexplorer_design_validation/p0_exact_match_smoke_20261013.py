"""Real-device, model-free exact probe/replay smoke for MobileExplorer P0.

The actions are manually selected, read-only navigation affordances.  This is
an engineering smoke test only; it does not estimate task success or evidence
utility.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

from android_world.parallel_exploration.action_slot import action_slot_for_action
from android_world.parallel_exploration.action_slot import runtime_observation_id
from android_world.parallel_exploration.action_slot import source_state_family
from android_world.parallel_exploration.adb import AdbClient
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
from android_world.parallel_exploration.belief_graph import canonical_action
from android_world.parallel_exploration.online_evidence_index import OnlineEvidenceIndex
from android_world.parallel_exploration.state import StateCapture


OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/me-p0-smoke")
OUT.mkdir(parents=True, exist_ok=True)
ADB = AdbClient("emulator-5554")
CAPTURE = StateCapture(ADB)

SCENARIOS = [
    {
        "name": "markor_todo_tab",
        "package": "net.gsantner.markor",
        "target": "To-Do",
        "recovery": "Files",
        "recovery_mode": "inverse_click",
    },
    {
        "name": "simple_calendar_settings",
        "package": "com.simplemobiletools.calendar.pro",
        "target": "Settings",
        "recovery": "BACK",
        "recovery_mode": "back",
    },
    {
        "name": "simple_gallery_overflow",
        "package": "com.simplemobiletools.gallery.pro",
        "target": "More options",
        "recovery": "BACK",
        "recovery_mode": "back",
    },
]


def center(bounds):
  return ((int(bounds[0]) + int(bounds[2])) // 2,
          (int(bounds[1]) + int(bounds[3])) // 2)


def visible_label(element):
  return element.text.strip() or element.content_desc.strip()


def find_clickable(state, label):
  matches = [e for e in state.elements
             if e.clickable and visible_label(e) == label
             and e.class_name.rsplit(".", 1)[-1].casefold() != "edittext"]
  if len(matches) != 1:
    raise RuntimeError(f"expected one clickable {label!r}; found {len(matches)}")
  return matches[0]


def strict_state_match(left, right):
  return {
      "activity": left.activity.component == right.activity.component,
      "task_id": left.activity.task_id == right.activity.task_id,
      "stack_depth": left.activity.stack_depth == right.activity.stack_depth,
      "struct": left.struct_sig.digest == right.struct_sig.digest,
      "phash": left.phash == right.phash,
  }


def write_state(prefix, state):
  path = OUT / f"{prefix}.json"
  payload = {
      "activity": dataclasses.asdict(state.activity),
      "structural_signature": state.struct_sig.digest,
      "layout_signature": state.layout_sig,
      "visual_signature": state.phash,
      "timings_ms": state.timings_ms,
      "elements": [dataclasses.asdict(e) for e in state.elements],
  }
  path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
  shot = ADB.run(["exec-out", "screencap", "-p"], timeout_s=8.0)
  (OUT / f"{prefix}.png").write_bytes(shot.stdout)
  return path.name, f"{prefix}.png"


def recover(scenario):
  time.sleep(0.8)
  if scenario["recovery_mode"] == "back":
    ADB.run(["shell", "input", "keyevent", "4"])
  else:
    state = CAPTURE.capture()
    target = find_clickable(state, scenario["recovery"])
    x, y = center(target.bounds)
    ADB.run(["shell", "input", "tap", str(x), str(y)])
  time.sleep(1.0)
  return CAPTURE.capture()


def run_scenario(scenario):
  global OUT
  app_out = OUT / scenario["name"]
  app_out.mkdir(parents=True, exist_ok=True)
  original_out = OUT
  OUT = app_out
  result = {"scenario": scenario, "status": "started", "model_calls": 0}
  try:
    ADB.run(["shell", "monkey", "-p", scenario["package"], "1"], timeout_s=10.0)
    time.sleep(1.8)
    source = CAPTURE.capture()
    source_json, source_png = write_state("source", source)
    target = find_clickable(source, scenario["target"])
    x, y = center(target.bounds)
    action = {"action_type": "click", "coordinate": [x, y]}
    family = source_state_family(source.activity.component, source.layout_sig)
    runtime_id = runtime_observation_id(source.activity.component,
                                        source.struct_sig.digest, source.phash)
    slot = action_slot_for_action(action, source.elements,
                                  source_family=family, runtime_id=runtime_id)
    if slot is None or not slot.matchable:
      raise RuntimeError(f"target could not be bound to a unique action slot: {slot}")
    action["action_slot_key"] = slot.key
    action["action_slot"] = dataclasses.asdict(slot)
    result["source"] = {
        "json": source_json, "screenshot": source_png,
        "activity": source.activity.component,
        "structural_signature": source.struct_sig.digest,
        "layout_signature": source.layout_sig,
        "runtime_observation": runtime_id,
    }
    result["action_slot"] = dataclasses.asdict(slot)

    ADB.run(["shell", "input", "tap", str(x), str(y)])
    time.sleep(1.0)
    probed = CAPTURE.capture()
    probe_json, probe_png = write_state("probe_successor", probed)
    graph = ProgressiveBeliefGraph(scenario["name"])
    src_id = graph.make_node_id(source.activity.component, source.layout_sig)
    dst_id = graph.make_node_id(probed.activity.component, probed.layout_sig)

    restored = recover(scenario)
    restored_json, restored_png = write_state("restored_source", restored)
    recovery_checks = strict_state_match(source, restored)
    recovery_ok = all(recovery_checks.values())
    result["probe"] = {
        "successor": {"json": probe_json, "screenshot": probe_png,
                      "activity": probed.activity.component,
                      "structural_signature": probed.struct_sig.digest,
                      "layout_signature": probed.layout_sig,
                      "graph_node": dst_id},
        "rollback": {"json": restored_json, "screenshot": restored_png,
                     "checks": recovery_checks, "verified": recovery_ok},
    }
    if not recovery_ok:
      result["status"] = "rollback_failed_evidence_invalidated"
      return result

    edge = graph.add_speculative_transition(
        src_id, action, dst_id, path_probability=1.0, confidence=0.2,
        expected_information_gain=0.0, risk_level="LOW",
        exploration_cost=0.0, rollback_success=True,
        discovered_labels=tuple(visible_label(e) for e in probed.elements
                                if visible_label(e))[:40],
    )
    graph.record_probe(edge.edge_id, rollback_ok=True, cost_s=0.0)
    index = OnlineEvidenceIndex()
    evidence_id = index.add(
        source=src_id, source_version=source.struct_sig.digest,
        destination=dst_id, action=action,
        labels=[visible_label(e) for e in probed.elements if visible_label(e)],
        recovered=True, cost_s=0.0,
    )
    index.commit()
    retrieved = index.retrieve(src_id, source.struct_sig.digest,
                               scenario["target"], max_records=4)
    negative_source = index.retrieve(src_id + "-wrong", source.struct_sig.digest,
                                     scenario["target"], max_records=4)
    negative_version = index.retrieve(src_id, "wrong-source-version",
                                      scenario["target"], max_records=4)
    wrong_action = dict(action, action_slot_key=slot.key + "-wrong")
    negative_action_rejected = canonical_action(action) != canonical_action(wrong_action)
    result["graph"] = {
        "edge_id": edge.edge_id,
        "evidence_id": evidence_id,
        "retrieved_evidence_ids": [item.key for item in retrieved],
        "positive_retrieval": any(item.key == evidence_id for item in retrieved),
        "negative_wrong_source_count": len(negative_source),
        "negative_wrong_source_version_count": len(negative_version),
        "negative_wrong_action_rejected": negative_action_rejected,
    }

    # Replay only after exact restoration. Re-bind in the restored observation;
    # coordinates are reused only because the source runtime fingerprint matches.
    replay_source = CAPTURE.capture()
    replay_precheck = strict_state_match(source, replay_source)
    replay_slot = action_slot_for_action(
        action, replay_source.elements,
        source_family=source_state_family(replay_source.activity.component,
                                          replay_source.layout_sig),
        runtime_id=runtime_observation_id(replay_source.activity.component,
                                          replay_source.struct_sig.digest,
                                          replay_source.phash),
    )
    replay_slot_match = bool(replay_slot and replay_slot.matchable
                             and replay_slot.key == slot.key)
    if not all(replay_precheck.values()) or not replay_slot_match:
      result["status"] = "replay_precondition_failed"
      return result
    ADB.run(["shell", "input", "tap", str(x), str(y)])
    time.sleep(1.0)
    actual = CAPTURE.capture()
    actual_json, actual_png = write_state("authoritative_successor", actual)
    actual_dst = graph.make_node_id(actual.activity.component, actual.layout_sig)
    family_match = actual_dst == dst_id
    exact_match = (actual.activity.component == probed.activity.component
                   and actual.struct_sig.digest == probed.struct_sig.digest
                   and actual.phash == probed.phash)
    graph.record_inference_alignment(src_id, action, True)
    graph.record_execution_verification(edge.edge_id, family_match)
    result["authoritative_replay"] = {
        "source_checks": replay_precheck,
        "action_slot_match": replay_slot_match,
        "successor": {"json": actual_json, "screenshot": actual_png,
                      "activity": actual.activity.component,
                      "structural_signature": actual.struct_sig.digest,
                      "layout_signature": actual.layout_sig,
                      "graph_node": actual_dst},
        "successor_family_match": family_match,
        "successor_exact_runtime_match": exact_match,
        "verdict": "matched" if family_match else "known_action_new_successor_branch",
    }
    final_source = recover(scenario)
    result["final_cleanup"] = {
        "verified": all(strict_state_match(source, final_source).values()),
        "checks": strict_state_match(source, final_source),
    }
    result["status"] = "passed" if family_match else "successor_branch_mismatch"
    return result
  except Exception as exc:  # preserve failure details for the audit
    result["status"] = "error"
    result["error"] = f"{type(exc).__name__}: {exc}"
    return result
  finally:
    OUT = original_out


def main():
  results = []
  for scenario in SCENARIOS:
    row = run_scenario(scenario)
    results.append(row)
    print(json.dumps({"scenario": row.get("scenario", {}).get("name"),
                      "status": row.get("status"),
                      "error": row.get("error"),
                      "rollback": row.get("probe", {}).get("rollback", {}).get("verified"),
                      "retrieval": row.get("graph", {}).get("positive_retrieval"),
                      "action_match": row.get("authoritative_replay", {}).get("action_slot_match"),
                      "successor_match": row.get("authoritative_replay", {}).get("successor_family_match")},
                     ensure_ascii=False), flush=True)
  (OUT / "audit.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
  print(f"AUDIT={OUT / 'audit.json'}", flush=True)


if __name__ == "__main__":
  main()
