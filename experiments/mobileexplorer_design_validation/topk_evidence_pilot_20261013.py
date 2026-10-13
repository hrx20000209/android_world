#!/usr/bin/env python3
"""Paired offline test of real, recovered one-hop probe evidence.

All decision screenshots and actual probe outcomes come from the completed
116-task GELAB-ZERO-4B run. No Android device is controlled by this script.
The probe selector in that run was random; this tests evidence utility at K=1,
not the value of a Top-K selection policy.
"""
from __future__ import annotations

import ast
import base64
import dataclasses
import hashlib
import itertools
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from android_world.parallel_exploration.action_slot import action_slot_for_action
from android_world.parallel_exploration.action_slot import runtime_observation_id
from android_world.parallel_exploration.action_slot import source_state_family
from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph
from android_world.parallel_exploration.online_evidence_index import OnlineEvidenceIndex


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EXP = Path(os.environ.get(
    "MOBILEEXPLORER_PROFILE_ROOT",
    str(REPO_ROOT / "experiments/info_value_profile_20261012/private_voi8-live-20261012-fd89ba"),
)).expanduser()
ROOTS = [EXP / "cohort40_run_20261013",
         EXP / "androidworld116_remaining76_run_20261013"]
SUMMARY = ROOTS[1] / "aggregate_summary_116.json"
SOURCE = Path(os.environ.get(
    "MOBILEEXPLORER_GELAB_SOURCE",
    str(REPO_ROOT / "android_world/agents/gelab_agent.py"),
)).expanduser()
PORTS = (8095, 8096)
MODEL_HINT = "GELAB-ZERO-4B"
SEED = 20261013
N_STATES = 24
CONDITIONS = ("E0_no_evidence", "E1_fresh_outcome", "E2_fresh_child_affordances",
              "D_wrong_source_control")
RUN_DIR = Path(os.environ.get(
    "MOBILEEXPLORER_PILOT_OUTPUT_DIR", str(HERE / "topk_evidence_pilot_20261013")
)).expanduser()
RUN_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
os.chmod(RUN_DIR, 0o700)


def _static_system_prompt() -> str:
  tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
  for node in tree.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "GELAB_SYSTEM_PROMPT"
        for t in node.targets
    ):
      call = node.value
      if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "strip":
        return ast.literal_eval(call.func.value)
      return ast.literal_eval(call)
  raise RuntimeError("GELAB_SYSTEM_PROMPT not found")


SYSTEM = _static_system_prompt()


class ParseActionError(Exception):
  pass


class SeeactUtils:
  ParseActionError = ParseActionError


def _official_parser():
  """Load the official action parser and its pure helper functions by AST."""
  tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
  funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
  needed: set[str] = set()
  stack = ["parse_gelab_response"]
  while stack:
    name = stack.pop()
    if name in needed or name not in funcs:
      continue
    needed.add(name)
    for node in ast.walk(funcs[name]):
      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
        stack.append(node.func.id)
  constants = [
      node for node in tree.body if isinstance(node, ast.Assign)
      and any(isinstance(t, ast.Name) and t.id in {"_ACTION_ALIASES", "_STATUS_ALIASES"}
              for t in node.targets)
  ]
  body = [
      ast.Import(names=[ast.alias(name="json"), ast.alias(name="re")]),
      ast.ImportFrom(module="collections", names=[ast.alias(name="OrderedDict")], level=0),
      ast.ImportFrom(module="typing", names=[ast.alias(name="Any")], level=0),
  ] + constants + [funcs[name] for name in funcs if name in needed]
  module = ast.Module(body=body, type_ignores=[])
  ast.fix_missing_locations(module)
  namespace: dict[str, Any] = {"seeact_utils": SeeactUtils}
  exec(compile(module, str(SOURCE), "exec"), namespace)  # pylint: disable=exec-used
  return namespace["parse_gelab_response"]


PARSE = _official_parser()


def _http(port: int, method: str, path: str, body=None, timeout=8):
  url = f"http://127.0.0.1:{port}{path}"
  response = requests.request(method, url, json=body, timeout=timeout)
  response.raise_for_status()
  if "json" in response.headers.get("content-type", ""):
    return response.json()
  return response.text


def _model_id(port: int) -> str:
  models = _http(port, "GET", "/v1/models")
  rows = models.get("data", [])
  match = next((str(x.get("id", "")) for x in rows
                if MODEL_HINT.casefold() in str(x.get("id", "")).casefold()), "")
  if not match:
    raise RuntimeError(f"{port} model mismatch: {[x.get('id') for x in rows]}")
  return match


def _queue(port: int) -> dict[str, Any]:
  metrics = _http(port, "GET", "/metrics")
  out = {}
  for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
    match = re.search(r"^" + re.escape(name) + r"(?:\{[^\n]*\})?\s+([0-9.]+)",
                      metrics, flags=re.M)
    out[name] = int(float(match.group(1))) if match else None
  return out


def _token_count(text: str) -> int:
  response = _http(8095, "POST", "/tokenize",
                   {"model": _model_id(8095), "prompt": text})
  return int(response["count"])


def _safe_file_name(path: Path) -> str:
  return str(path.relative_to(EXP))


def _exact_probe_row(probe: dict[str, Any], tv: dict[str, Any], action_row: dict[str, Any],
                     screenshot: Path) -> bool:
  graph = probe.get("graph") or {}
  src, dst = graph.get("src") or {}, graph.get("dst") or {}
  sig = probe.get("sig_match") or {}
  step = probe.get("step_idx")
  source_package = str(src.get("activity", "")).split("/", 1)[0]
  action_package = str(action_row.get("source_activity", "")).split("/", 1)[0]
  return bool(
      probe.get("recovery_ok")
      and sig.get("activity") and sig.get("struct") and sig.get("phash")
      and source_package == action_package
      and src.get("activity") == tv.get("source_activity", src.get("activity"))
      and src.get("structural_signature") == tv.get("source_structure_signature")
      and src.get("layout_signature") == tv.get("source_layout_signature")
      and dst and isinstance(step, int) and screenshot.exists()
  )


def _load_candidate_events():
  aggregate = json.loads(SUMMARY.read_text(encoding="utf-8"))
  task_meta = {row["task"]: row for row in aggregate["tasks"]}
  events = []
  for root in ROOTS:
    for action_file in (root / "agent").glob("*/*/action.jsonl"):
      task = action_file.parents[1].name
      task_dir = root / task
      if not (task_dir / "probe_trace.jsonl").exists():
        continue
      action_rows = [json.loads(line) for line in action_file.read_text(encoding="utf-8").splitlines()]
      tv_rows = {}
      tv_path = task_dir / "transition_verification.jsonl"
      if tv_path.exists():
        for line in tv_path.read_text(encoding="utf-8").splitlines():
          item = json.loads(line)
          tv_rows[int(item["step"])] = item
      loop_steps = set()
      loop_path = task_dir / "loop_detected.jsonl"
      if loop_path.exists():
        for line in loop_path.read_text(encoding="utf-8").splitlines():
          try:
            item = json.loads(line)
            if "step" in item:
              loop_steps.add(int(item["step"]))
          except (TypeError, ValueError, json.JSONDecodeError):
            continue
      for line in (task_dir / "probe_trace.jsonl").read_text(encoding="utf-8").splitlines():
        probe = json.loads(line)
        step = probe.get("step_idx")
        if not isinstance(step, int) or step < 0 or step >= len(action_rows):
          continue
        agent = action_rows[step]
        source = (probe.get("graph") or {}).get("src") or {}
        screenshot = action_file.parent / f"screenshot_model_input_{step}.png"
        tv = tv_rows.get(step) or {}
        if not _exact_probe_row(probe, tv, agent, screenshot):
          continue
        actual_action = agent.get("action_dict") or {}
        if str(actual_action.get("action_type", "")).upper() != "CLICK":
          continue
        if actual_action.get("x") is None or actual_action.get("y") is None:
          continue
        src_elements = source.get("elements") or []
        graph_action = (probe.get("graph") or {}).get("action") or {}
        runtime = runtime_observation_id(source.get("activity", ""),
                                         source.get("structural_signature", ""),
                                         source.get("visual_signature", ""))
        family = source_state_family(source.get("activity", ""),
                                     source.get("layout_signature", ""))
        slot = action_slot_for_action(graph_action, src_elements,
                                      source_family=family, runtime_id=runtime)
        if slot is None or not slot.matchable:
          continue
        meta = task_meta.get(task, {})
        dst = (probe.get("graph") or {}).get("dst") or {}
        events.append({
            "task": task,
            "step": step,
            "app": meta.get("app", "unknown"),
            "complexity_band": meta.get("complexity_band", "unknown"),
            "complexity": meta.get("complexity"),
            "historical_baseline_outcome": meta.get("baseline_outcome", "unknown"),
            "gelab_pilot_success": meta.get("pilot_success"),
            "goal": agent.get("goal", ""),
            "history": _history_text(action_rows, step),
            "actual_response": agent.get("response", ""),
            "actual_action": actual_action,
            "actual_summary": agent.get("summary", ""),
            "source_activity": source.get("activity", ""),
            "source_layout_signature": source.get("layout_signature", ""),
            "source_structural_signature": source.get("structural_signature", ""),
            "source_visual_signature": source.get("visual_signature", ""),
            "source_node": ProgressiveBeliefGraph.make_node_id(
                source.get("activity", ""), source.get("layout_signature", "")),
            "source_elements": src_elements,
            "probe_action_slot_key": slot.key,
            "probe_action": graph_action,
            "probe_action_slot": dataclasses.asdict(slot),
            "probe_edge_id": probe.get("graph", {}).get("edge_id", ""),
            "probe_type": probe.get("probe_type", ""),
            "probe_step_idx": step,
            "probe_recovery_verified": True,
            "probe_discovered": probe.get("discovered") or {},
            "destination_activity": dst.get("activity", ""),
            "destination_layout_signature": dst.get("layout_signature", ""),
            "destination_structural_signature": dst.get("structural_signature", ""),
            "destination_elements": dst.get("elements") or [],
            "source_signature_match": dict(probe.get("sig_match") or {}),
            "screenshot_path": str(screenshot),
            "screenshot_relpath": _safe_file_name(screenshot),
            "screenshot_sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest(),
            "source_state_version": source.get("structural_signature", ""),
            "loop_detected_step": step in loop_steps,
            "task_dir": str(task_dir),
            "probe_trace_path": str(task_dir / "probe_trace.jsonl"),
            "graph_snapshot_path": str(task_dir / "progressive_belief_graph.json"),
            "graph_snapshot_sha256": (hashlib.sha256((task_dir / "progressive_belief_graph.json").read_bytes()).hexdigest()
                                       if (task_dir / "progressive_belief_graph.json").exists() else ""),
            "authoritative_verdict": tv.get("verdict", ""),
            "authoritative_destination_family": tv.get("destination_family", ""),
        })
  return events, aggregate


def _history_text(rows: list[dict[str, Any]], step: int) -> str:
  prior = rows[max(0, step - 8):step]
  if not prior:
    return "None yet."
  return "\n".join(f"{idx + 1}. {item.get('summary') or ''}"
                   for idx, item in enumerate(prior, start=max(0, step - len(prior))))


def _select_one_per_task(events: list[dict[str, Any]], n: int = N_STATES) -> list[dict[str, Any]]:
  by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for row in events:
    by_task[row["task"]].append(row)
  reps = []
  for task, rows in by_task.items():
    # Prefer an action screen where the recovered probe caused a visible
    # structural change and the child has readable affordances. Ties are seeded.
    rows.sort(key=lambda r: (
        bool(r["probe_discovered"].get("new_element_count", 0)),
        len(r["destination_elements"]),
        int(r["step"] > 0),
        -r["step"],
    ), reverse=True)
    reps.append(rows[0])
  rng = random.Random(SEED)
  chosen: list[dict[str, Any]] = []
  for outcome in ("success", "failure"):
    pool = [r for r in reps if r["historical_baseline_outcome"] == outcome]
    selected_apps = {r["app"] for r in chosen}
    selected_complexities = {r["complexity_band"] for r in chosen}
    selected_pilot = {r["gelab_pilot_success"] for r in chosen}
    while pool and sum(r["historical_baseline_outcome"] == outcome for r in chosen) < n // 2:
      scored = []
      for row in pool:
        score = (4 * (row["app"] not in selected_apps)
                 + 2 * (row["complexity_band"] not in selected_complexities)
                 + (row["gelab_pilot_success"] not in selected_pilot)
                 + 0.15 * min(8, int(row["probe_discovered"].get("new_element_count", 0))))
        scored.append((score, rng.random(), row))
      _, _, row = max(scored, key=lambda item: (item[0], item[1]))
      chosen.append(row)
      pool.remove(row)
      selected_apps.add(row["app"])
      selected_complexities.add(row["complexity_band"])
      selected_pilot.add(row["gelab_pilot_success"])
  return sorted(chosen, key=lambda r: (r["task"], r["step"]))


def _box_contains(box, x, y):
  try:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]
  except (TypeError, IndexError):
    return False


def _target_for_reference(row: dict[str, Any]) -> dict[str, Any] | None:
  action = row["actual_action"]
  x, y = float(action["x"]), float(action["y"])
  hits = []
  for element in row["source_elements"]:
    bounds = element.get("bounds")
    if not element.get("is_clickable") or not _box_contains(bounds, x, y):
      continue
    area = max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])
    hits.append((area, element))
  if not hits:
    return None
  return min(hits, key=lambda item: item[0])[1]


def _short_label(element: dict[str, Any], goal: str) -> str:
  label = str(element.get("content_desc") or element.get("text") or "").strip()
  label = re.sub(r"\s+", " ", label)
  if (not label or len(label) > 56 or re.search(r"\d{3,}", label)
      or label.casefold() in goal.casefold()):
    label = ""
  rid = str(element.get("resource_id") or "").rsplit("/", 1)[-1]
  klass = str(element.get("class_name") or element.get("class") or "control").rsplit(".", 1)[-1]
  role = "scrollable" if element.get("is_scrollable") else "clickable"
  visible = f"{label} " if label else ""
  resource = f"id={rid} " if rid and len(rid) < 64 else ""
  return f"{visible}{resource}({klass}, {role})".strip()


def _affordances(row: dict[str, Any]) -> list[str]:
  labels = []
  seen = set()
  for element in row["destination_elements"]:
    if not (element.get("is_clickable") or element.get("is_scrollable")):
      continue
    label = _short_label(element, row["goal"])
    if label and label not in seen:
      seen.add(label)
      labels.append(label)
    if len(labels) >= 12:
      break
  return labels


def _event_text(event: dict[str, Any]) -> str:
  slot = event["probe_action_slot"]
  action = event["probe_action"]
  return (f"Action slot: {slot['action_type']} on {slot['target_class']}"
          f"{(' direction=' + str(dict(slot.get('action_parameters') or {}).get('direction'))) if slot.get('action_parameters') else ''}. "
          f"Observed outcome: activity={event['destination_activity']}; "
          f"screen-family={event['destination_layout_signature']}. "
          "The source was restored and verified after this observation.")


def _fresh_evidence(row: dict[str, Any], with_affordances: bool) -> str:
  text = ("Fresh, recovered observation from this exact current source state:\n"
          + _event_text(row))
  if with_affordances:
    labels = _affordances(row)
    text += ("\nAccessible controls observed on that child screen: "
             + ("; ".join(labels) if labels else "no safe named clickable controls captured"))
  text += "\nThis describes a UI outcome only; it does not say whether the branch completes the task."
  return text


def _stale_control(row: dict[str, Any], pool: list[dict[str, Any]]) -> dict[str, Any]:
  candidates = [x for x in pool if x["app"] == row["app"] and x["task"] != row["task"]
                and (x["source_activity"], x["source_layout_signature"])
                != (row["source_activity"], row["source_layout_signature"])]
  if not candidates:
    candidates = [x for x in pool if x["task"] != row["task"]
                  and x["app"] != row["app"]]
  return min(candidates, key=lambda x: (x["app"] != row["app"],
                                        x["task"], x["step"]))


def _user_text(row: dict[str, Any], evidence: str | None = None, pad: str | None = None) -> str:
  body = (f"Task:\n{row['goal']}\n\nHistory actions:\n{row['history']}\n\n")
  if evidence:
    body += "Additional exploration evidence (use only if it helps choose the next action):\n"
    body += evidence + "\n\n"
    if pad is not None:
      body += "Evidence-length control (contains no additional UI facts): " + pad + "\n\n"
  return body + "Current screenshot is attached below.\nChoose the next single action."


def _align_evidence_lengths(row: dict[str, Any], users: dict[str, str],
                            evidence: dict[str, str]) -> tuple[dict[str, str], dict[str, int]]:
  """Roughly word-match evidence arms; exact prompt tokens are logged later."""
  names = ("E1_fresh_outcome", "E2_fresh_child_affordances", "D_wrong_source_control")
  base_counts = {name: len(users[name].split()) for name in names}
  marker_words = len("Evidence-length control (contains no additional UI facts):".split())
  target = max(base_counts.values()) + marker_words + 12
  aligned = {}
  for name in names:
    gap = max(0, target - base_counts[name] - marker_words)
    pad = "x " * gap
    aligned[name] = _user_text(row, evidence[name], pad)
  result = {**users, **aligned}
  return result, {name: len(value.split()) for name, value in result.items()}


def _messages(row: dict[str, Any], user_text: str) -> list[dict[str, Any]]:
  image = Path(row["screenshot_path"]).read_bytes()
  data = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
  return [
      {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
      {"role": "user", "content": [
          {"type": "text", "text": user_text},
          {"type": "image_url", "image_url": {"url": data}},
      ]},
  ]


def _official_action(text: str) -> dict[str, Any] | None:
  parsed = PARSE(text)
  typ = str(parsed.get("action") or "").upper()
  point = parsed.get("point") or parsed.get("coordinate") or parsed.get("coordinates")
  return {"type": typ, "point": point,
          "direction": str(parsed.get("direction") or "").casefold(),
          "summary": str(parsed.get("summary") or parsed.get("explain") or "")}


def _score_response(text: str, row: dict[str, Any]) -> dict[str, Any]:
  try:
    parsed = _official_action(text)
  except Exception as exc:  # parser errors are data, not a reason to drop a cell
    return {"parse_ok": False, "parse_error_type": type(exc).__name__,
            "action_changed": None, "target_grounded": False,
            "reference_path_consistent": False, "predicted_action_type": None}
  if parsed is None:
    return {"parse_ok": False, "parse_error_type": "empty_parse",
            "action_changed": None, "target_grounded": False,
            "reference_path_consistent": False, "predicted_action_type": None}
  typ = parsed["type"].replace("-", "_")
  typ = {"TAP": "CLICK", "PRESS": "CLICK", "SWIPE": "SLIDE",
         "LONG_PRESS": "LONGPRESS", "LONGCLICK": "LONGPRESS"}.get(typ, typ)
  ref_type = str(row["actual_action"].get("action_type", "")).upper()
  image = Image.open(row["screenshot_path"])
  width, height = image.size
  coord = parsed.get("point")
  x = y = None
  if isinstance(coord, (tuple, list)) and len(coord) >= 2:
    x = int(round(max(0, min(1000, float(coord[0]))) / 1000 * (width - 1)))
    y = int(round(max(0, min(1000, float(coord[1]))) / 1000 * (height - 1)))
  reference = row.get("reference_target")
  target_grounded = False
  reference_path_consistent = False
  predicted_descriptor = None
  if x is not None and y is not None and typ == "CLICK":
    # UI hierarchy bounds and screenshots share the emulator logical size.
    for element in row["source_elements"]:
      if element.get("is_clickable") and _box_contains(element.get("bounds"), x, y):
        target_grounded = True
        predicted_descriptor = {
            "resource_id": element.get("resource_id", ""),
            "text": element.get("text", ""),
            "content_desc": element.get("content_desc", ""),
            "class": element.get("class_name") or element.get("class", ""),
        }
        break
    if reference and _box_contains(reference.get("bounds"), x, y):
      reference_path_consistent = typ == ref_type
  else:
    target_grounded = (typ == "SLIDE" and bool(parsed.get("direction")))
    reference_path_consistent = bool(typ == ref_type
                                     and parsed.get("direction")
                                     and parsed.get("direction") == str(row["actual_action"].get("direction") or "").casefold())
  return {
      "parse_ok": True,
      "predicted_action_type": typ,
      "predicted_point_normalized": coord,
      "predicted_point_pixels": [x, y] if x is not None else None,
      "predicted_target": predicted_descriptor,
      "target_grounded": target_grounded,
      "reference_path_consistent": reference_path_consistent,
      "action_changed": not reference_path_consistent,
      "historical_task_outcome": row["historical_baseline_outcome"],
      "gelab_pilot_success": row["gelab_pilot_success"],
      "metric_note": "Reference-path agreement is not task-action correctness; grounded alternatives remain unresolved without execution/adjudication.",
  }


def _status(port: int) -> dict[str, Any]:
  health = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
  health.raise_for_status()
  return {"port": port, "health": health.status_code == 200,
          "model_id": _model_id(port), "queue": _queue(port)}


def _call(cell: dict[str, Any]) -> dict[str, Any]:
  port = cell["endpoint_port"]
  model_id = _model_id(port)
  image = Path(cell["screenshot_path"]).read_bytes()
  data = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
  messages = [
      {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
      {"role": "user", "content": [
          {"type": "text", "text": cell["user_text"]},
          {"type": "image_url", "image_url": {"url": data}},
      ]},
  ]
  payload = {"model": model_id, "messages": messages,
             "temperature": 0, "top_p": 1, "seed": SEED,
             "max_tokens": 256, "stream": False}
  started = time.perf_counter()
  response = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                           json=payload, timeout=90)
  latency = time.perf_counter() - started
  response.raise_for_status()
  data = response.json()
  raw = str((data.get("choices") or [{}])[0].get("message", {}).get("content") or "")
  score = _score_response(raw, cell["decision"])
  usage = data.get("usage") or {}
  return {
      "decision_id": cell["decision_id"],
      "task": cell["task"], "step": cell["step"],
      "condition": cell["condition"], "endpoint_port": port,
      "model_id": model_id, "latency_s": round(latency, 4),
      "prompt_tokens": usage.get("prompt_tokens"),
      "completion_tokens": usage.get("completion_tokens"),
      "response": raw, "score": score,
      "evidence_text": cell["evidence_text"],
      "screenshot_sha256": cell["screenshot_sha256"],
      "source_structural_signature": cell["source_structural_signature"],
      "probe_edge_id": cell["probe_edge_id"],
      "probe_action_slot_key": cell["probe_action_slot_key"],
      "stale_source_signature": cell.get("stale_source_signature"),
  }


def _make_cells(selected, all_events):
  rng = random.Random(SEED)
  cells = []
  # Every decision receives two conditions on each endpoint. Use all six
  # 2-of-4 assignments equally, shuffled across states, so each treatment is
  # exposed 12 times to each service in the 24-state panel.
  condition_pairs = list(itertools.combinations(range(len(CONDITIONS)), 2))
  pair_schedule = condition_pairs * (len(selected) // len(condition_pairs))
  if len(pair_schedule) != len(selected):
    raise RuntimeError("balanced endpoint schedule requires a multiple of six states")
  rng.shuffle(pair_schedule)
  for index, row in enumerate(selected):
    row["reference_target"] = _target_for_reference(row)
    row["decision_id"] = f"D{index + 1:02d}_{row['task']}_s{row['step']}"
    indexer = OnlineEvidenceIndex()
    evidence_labels = _affordances(row)
    key = indexer.add(
        source=row["source_node"], source_version=row["source_state_version"],
        destination=(row["destination_activity"] + "|" + row["destination_layout_signature"]),
        action={"action_slot_key": row["probe_action_slot_key"], **row["probe_action"]},
        labels=evidence_labels,
        recovered=True, cost_s=float((row["probe_discovered"].get("timings_ms") or {}).get("total", 0)) / 1000,
    )
    indexer.commit()
    retrieved = indexer.retrieve(row["source_node"], row["source_state_version"],
                                 "scroll click screen", max_records=4)
    if not any(record.key == key for record in retrieved):
      # Query based on the exact serialization tokens when generic terms don't match.
      retrieved = indexer.retrieve(row["source_node"], row["source_state_version"],
                                   row["probe_action_slot_key"], max_records=4)
    if not any(record.key == key for record in retrieved):
      raise RuntimeError(f"fresh edge failed exact graph retrieval for {row['decision_id']}")
    row["retrieved_evidence_id"] = key
    stale = _stale_control(row, all_events)
    row["stale_source_signature"] = stale["source_structural_signature"]
    evidence = {
        "E1_fresh_outcome": _fresh_evidence(row, False),
        "E2_fresh_child_affordances": _fresh_evidence(row, True),
        "D_wrong_source_control": (
            "Historical observation from another runtime source in the same app; "
            "it is not matched to the current source and may be stale:\n"
            + _event_text(stale)
            + "\nDo not assume this outcome transfers to the current screen."),
    }
    users = {"E0_no_evidence": _user_text(row)}
    users.update({name: _user_text(row, body) for name, body in evidence.items()})
    users, text_counts = _align_evidence_lengths(row, users, evidence)
    pair_for_first_endpoint = set(pair_schedule[index])
    for condition_index, condition in enumerate(CONDITIONS):
      port = PORTS[0] if condition_index in pair_for_first_endpoint else PORTS[1]
      cells.append({
          "decision_id": row["decision_id"], "task": row["task"],
          "step": row["step"], "app": row["app"],
          "condition": condition, "endpoint_port": port,
          "user_text": users[condition], "input_text_word_count": text_counts[condition],
          "evidence_text": evidence.get(condition, ""),
          "screenshot_path": row["screenshot_path"],
          "screenshot_sha256": row["screenshot_sha256"],
          "source_structural_signature": row["source_structural_signature"],
          "probe_edge_id": row["probe_edge_id"],
          "probe_action_slot_key": row["probe_action_slot_key"],
          "stale_source_signature": row["stale_source_signature"],
          "decision": row,
      })
  # Build concurrent pairs with one request per endpoint without mutating the
  # randomized treatment assignment to repair accidental endpoint collisions.
  queues = {port: [c for c in cells if c["endpoint_port"] == port] for port in PORTS}
  for queue in queues.values():
    rng.shuffle(queue)
  cells = [cell for pair in zip(queues[PORTS[0]], queues[PORTS[1]]) for cell in pair]
  counts = Counter((c["condition"], c["endpoint_port"]) for c in cells)
  expected = len(selected) // 2
  if any(counts[(condition, port)] != expected
         for condition in CONDITIONS for port in PORTS):
    raise RuntimeError(f"unbalanced treatment-to-endpoint assignment: {counts}")
  return cells


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--prepare-only", action="store_true")
  parser.add_argument("--run", action="store_true")
  args = parser.parse_args()
  events, aggregate = _load_candidate_events()
  selected = _select_one_per_task(events)
  if len(selected) != N_STATES:
    raise RuntimeError(f"wanted {N_STATES} distinct-task states; found {len(selected)}")
  apps = {x["app"] for x in selected}
  hist = Counter(x["historical_baseline_outcome"] for x in selected)
  pilot = Counter(str(x["gelab_pilot_success"]) for x in selected)
  by_complexity = Counter(x["complexity_band"] for x in selected)
  if min(hist.get(k, 0) for k in ("success", "failure")) < N_STATES // 2:
    raise RuntimeError(f"cohort not balanced by historical baseline: {hist}")
  if len(apps) < 10:
    raise RuntimeError(f"too few apps in frozen panel: {apps}")
  for row in selected:
    row["reference_target"] = _target_for_reference(row)
  cells = _make_cells(selected, events)
  manifest = {
      "label": "GELAB-ZERO-4B offline pilot; exact-source recovered real probe evidence; K=1",
      "sample_seed": SEED,
      "model": MODEL_HINT,
      "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
      "dataset_roots": [str(x) for x in ROOTS],
      "source_event_count": len(events),
      "n_decision_states": len(selected),
      "n_unique_tasks": len({x["task"] for x in selected}),
      "n_apps": len(apps), "apps": sorted(apps),
      "historical_baseline_strata": dict(hist),
      "gelab_pilot_outcome_strata": dict(pilot),
      "complexity_bands": dict(by_complexity),
      "conditions": list(CONDITIONS),
      "endpoint_allocation": {
          "method": "For each decision state, assign two of four conditions to each endpoint using all six pairings equally; independently shuffle endpoint queues and execute one request per endpoint per batch.",
          "counts_by_condition_and_endpoint": {
              condition: {str(port): sum(1 for cell in cells
                                         if cell["condition"] == condition
                                         and cell["endpoint_port"] == port)
                          for port in PORTS}
              for condition in CONDITIONS
          },
      },
      "k": 1,
      "k2_k3_note": "The 116-task corpus has one recovered probe per decision window; exact-source distinct candidate sets of size 2 or 3 were unavailable for a balanced evaluation.",
      "selection_note": "One decision per task; exact source/activity/layout/structural match; probe recovery and stack checks passed; screenshot_model_input is from that source; baseline action restricted to CLICK for coordinate-grounded scoring.",
      "evidence_policy": "E1/E2 are retrieved from an OnlineEvidenceIndex using exact source node and strict structural version. D is a same-app but wrong-source historical trace. All evidence comes from observed UI transitions; no model is used for Explorer selection. E1/E2/D are whitespace-word-count aligned; exact vLLM prompt-token counts are logged per request.",
      "not_a_ground_truth_note": "Recorded action agreement is only successful-run-path consistency when the whole GELAB pilot task passed; no alternative click is called correct without execution or expert adjudication.",
      "decisions": [{k: v for k, v in row.items() if k not in {"source_elements", "destination_elements"}}
                    for row in selected],
      "cells": [{k: v for k, v in cell.items() if k != "decision"} for cell in cells],
      "aggregate_input": {"task_count": aggregate["registry_size"],
                          "pilot_success_counts": aggregate["pilot_success_counts"]},
  }
  manifest_path = RUN_DIR / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
  os.chmod(manifest_path, 0o600)
  print(json.dumps({"manifest": str(manifest_path), "states": len(selected),
                    "apps": len(apps), "baseline": dict(hist),
                    "pilot": dict(pilot), "complexity": dict(by_complexity),
                    "source_events": len(events), "cells": len(cells),
                    "reference_boxes": sum(bool(x["reference_target"]) for x in selected),
                    "endpoint": {p: _status(p) for p in PORTS}}, ensure_ascii=False))
  if args.prepare_only or not args.run:
    return
  results_path = RUN_DIR / "raw_results.jsonl"
  if results_path.exists():
    raise RuntimeError(f"results already exist: {results_path}; refusing accidental rerun")
  for cell in cells:
    ready_deadline = time.monotonic() + 90.0
    while True:
      try:
        states = [_status(p) for p in PORTS]
        if all(s["health"] and s["queue"].get("vllm:num_requests_running") == 0
               and s["queue"].get("vllm:num_requests_waiting") == 0 for s in states):
          break
      except Exception:
        pass
      if time.monotonic() >= ready_deadline:
        raise TimeoutError("vLLM endpoints did not become idle within 90 seconds")
      time.sleep(1.0)
  # Each service gets half of every treatment, with exactly one concurrent
  # request per endpoint in each batch.
  with ThreadPoolExecutor(max_workers=2) as pool:
    for offset in range(0, len(cells), 2):
      batch = cells[offset:offset + 2]
      if len(batch) != 2 or {c["endpoint_port"] for c in batch} != set(PORTS):
        raise RuntimeError(f"invalid endpoint batch: {[c['endpoint_port'] for c in batch]}")
      futures = [pool.submit(_call, cell) for cell in batch]
      for future in futures:
        try:
          record = future.result()
        except Exception as exc:
          record = {"request_status": "error", "error_type": type(exc).__name__,
                    "error": str(exc)}
        with results_path.open("a", encoding="utf-8") as stream:
          stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.chmod(results_path, 0o600)
        print(json.dumps({"task": record.get("task"), "step": record.get("step"),
                          "condition": record.get("condition"), "endpoint": record.get("endpoint_port"),
                          "status": record.get("request_status", "ok"),
                          "parse": record.get("score", {}).get("parse_ok"),
                          "grounded": record.get("score", {}).get("target_grounded"),
                          "path_consistent": record.get("score", {}).get("reference_path_consistent"),
                          "latency_s": record.get("latency_s")}, ensure_ascii=False), flush=True)
  _write_summary(selected, results_path)


def _write_summary(selected, results_path):
  rows = []
  for line in results_path.read_text(encoding="utf-8").splitlines():
    try:
      rows.append(json.loads(line))
    except json.JSONDecodeError:
      continue
  by_condition = {}
  for condition in CONDITIONS:
    rr = [r for r in rows if r.get("condition") == condition]
    valid = [r for r in rr if r.get("score", {}).get("parse_ok")]
    by_condition[condition] = {
        "n": len(rr),
        "parse_valid": len(valid),
        "target_grounded": sum(bool(r["score"].get("target_grounded")) for r in valid),
        "reference_path_consistent": sum(bool(r["score"].get("reference_path_consistent")) for r in valid),
        "mean_latency_s": round(sum(float(r.get("latency_s") or 0) for r in rr) / len(rr), 4) if rr else None,
        "mean_input_tokens": round(sum(float(r.get("prompt_tokens") or 0) for r in rr) / len(rr), 1) if rr else None,
        "mean_output_tokens": round(sum(float(r.get("completion_tokens") or 0) for r in rr) / len(rr), 1) if rr else None,
    }
  baseline = {r.get("decision_id"): r for r in rows if r.get("condition") == "E0_no_evidence"}
  paired = {}
  paired_reference_direction = {}
  for condition in CONDITIONS[1:]:
    diffs = Counter()
    direction = Counter()
    for row in rows:
      if row.get("condition") != condition or row.get("decision_id") not in baseline:
        continue
      a = baseline[row["decision_id"]].get("score", {})
      b = row.get("score", {})
      if not b.get("parse_ok"):
        diffs["invalid_output"] += 1
      elif not a.get("parse_ok"):
        diffs["baseline_invalid_output"] += 1
        direction["baseline_invalid_excluded"] += 1
      elif b.get("reference_path_consistent") == a.get("reference_path_consistent"):
        diffs["same_reference_path_relation"] += 1
        direction["same"] += 1
      elif b.get("target_grounded"):
        diffs["grounded_but_unresolved_change"] += 1
        direction["toward_reference"] += int(bool(b.get("reference_path_consistent")))
        direction["away_from_reference"] += int(not bool(b.get("reference_path_consistent")))
      else:
        diffs["ungrounded_or_unresolved_change"] += 1
        direction["toward_reference"] += int(bool(b.get("reference_path_consistent")))
        direction["away_from_reference"] += int(not bool(b.get("reference_path_consistent")))
    paired[condition] = dict(diffs)
    paired_reference_direction[condition] = dict(direction)
  summary = {
      "label": "24-state descriptive offline evidence-utility pilot; no device execution; not a causal task-success estimate",
      "n_decision_states": len(selected), "n_cells": len(rows),
      "by_condition": by_condition, "paired_change_categories": paired,
      "paired_reference_path_direction_vs_E0": paired_reference_direction,
      "task_success_effect_measured": False,
      "action_correctness_claim": "none; path agreement and UI grounding are proxies, not ground truth",
      "limitations": [
          "K=1 only; no balanced exact-source K=2/3 candidate set in archived traces.",
          "Depth-2 verified traces were only available for 3 exact same-task joins and were not powered as an arm.",
          "Current 116-task pilot's original selector was random; evidence utility here does not validate a Top-K selection policy.",
          "No action was executed after offline inference; evaluator completion, recovery after the chosen action, and true improvement are unknown.",
      ],
  }
  path = RUN_DIR / "summary.json"
  path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
  os.chmod(path, 0o600)
  print(json.dumps({"summary": str(path), "cells": len(rows),
                    "by_condition": by_condition, "paired": paired,
                    "paired_reference_path_direction_vs_E0": paired_reference_direction}, ensure_ascii=False))


if __name__ == "__main__":
  main()
