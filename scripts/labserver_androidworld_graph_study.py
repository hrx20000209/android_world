#!/usr/bin/env python3
"""Resumable AndroidWorld graph-design study for an SSH-independent LabServer run.

The supervisor is intended to run on the LabServer itself (normally under
nohup/setsid). It only starts vLLM on GPUs that are demonstrably idle, or
reuses an explicitly selected endpoint after its request counters stay flat.
Each task/arm has an immutable attempt directory and an AndroidWorld
IncrementalCheckpointer; completed checkpoints are adopted after a supervisor
restart rather than repeated. Raw logs/checkpoints stay on the server.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
from collections import Counter
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO_ROOT / "experiments/labserver_androidworld_graph_study/protocol.json"
DEFAULT_IMAGE = "android-world:labserver-ready-isolated6-20260924"
DEFAULT_MODEL_DIR = "/home/rxhuang/Projects/models/gelab_zero_4B"
DEFAULT_VLLM_PYTHON = "/home/rxhuang/anaconda3/envs/agent/bin/python"
CONTAINER_PYTHON = "/usr/local/bin/python3"
CONTAINER_SITE_OVERLAY = "/study/python-site-packages"
MODEL_NAME = "GELAB-ZERO-4B"
PUBLISH_PREFIX = Path("reports/labserver_androidworld_graph_study")

def _now() -> str:
  return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, value: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex[:8]}")
  with tmp.open("w", encoding="utf-8") as stream:
    json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
  os.replace(tmp, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
  fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o640)
  try:
    os.write(fd, encoded)
    os.fsync(fd)
  finally:
    os.close(fd)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  if not path.is_file():
    return rows
  with path.open(encoding="utf-8") as stream:
    for line in stream:
      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        continue
      if isinstance(row, dict):
        rows.append(row)
  return rows


def _run(command: list[str], *, timeout: int = 60, check: bool = True,
         text: bool = True) -> subprocess.CompletedProcess:
  result = subprocess.run(command, capture_output=True, text=text, timeout=timeout)
  if check and result.returncode:
    stdout = result.stdout if text else "<binary>"
    stderr = result.stderr if text else "<binary>"
    raise RuntimeError(f"command failed ({result.returncode}): {command!r}\n{stdout[-2000:]}\n{stderr[-2000:]}")
  return result


def _http_json(url: str, timeout: float = 4.0) -> dict[str, Any]:
  request = urllib.request.Request(url, headers={"Accept": "application/json"})
  with urllib.request.urlopen(request, timeout=timeout) as response:
    return json.loads(response.read().decode("utf-8"))


def _metrics(url: str) -> tuple[dict[str, float], str]:
  request = urllib.request.Request(url, headers={"Accept": "text/plain"})
  with urllib.request.urlopen(request, timeout=5) as response:
    body = response.read().decode("utf-8", errors="replace")
  counters: dict[str, float] = {}
  for line in body.splitlines():
    if line.startswith("#") or not line.strip():
      continue
    match = re.match(r"([^\s{]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
    if match and any(token in match.group(1) for token in (
        "request_success_total", "prompt_tokens_total", "generation_tokens_total",
        "num_requests_running", "num_requests_waiting")):
      counters[match.group(1)] = counters.get(match.group(1), 0.0) + float(match.group(2))
  return counters, body


def _endpoint_ok(port: int) -> bool:
  try:
    data = _http_json(f"http://127.0.0.1:{port}/v1/models")
    return any(row.get("id") == MODEL_NAME for row in data.get("data", []))
  except Exception:
    return False


def _endpoint_idle(port: int, seconds: int, status: Path | None = None) -> bool:
  url = f"http://127.0.0.1:{port}/metrics"
  try:
    before, _ = _metrics(url)
    active_metrics = ("vllm:num_requests_running", "vllm:num_requests_waiting")
    # Fail closed when the server does not expose live queue gauges. Cumulative
    # token/request counters alone cannot prove a long-running request is idle.
    if any(name not in before or before[name] > 0 for name in active_metrics):
      return False
    if not _endpoint_ok(port):
      return False
    if status:
      _atomic_json(status, {"state": "checking_shared_vllm_idle", "port": port,
                            "checked_at": _now(), "idle_window_s": seconds})
    time.sleep(seconds)
    after, _ = _metrics(url)
    return (
        all(name in after and after[name] == 0 for name in active_metrics)
        and before == after
    )
  except Exception:
    return False


def _free_gpus(limit: int) -> list[int]:
  try:
    result = _run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                   "--format=csv,noheader,nounits"], timeout=10)
  except Exception:
    return []
  candidates: list[int] = []
  for line in result.stdout.splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 3:
      continue
    try:
      gpu, memory_mb, utilization = map(int, fields)
    except ValueError:
      continue
    if memory_mb <= 512 and utilization <= 3:
      candidates.append(gpu)
  return candidates[:max(0, limit)]


def _gpu_idle(gpu_index: int) -> bool:
  try:
    result = _run([
        "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], timeout=10)
  except Exception:
    return False
  for line in result.stdout.splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 3:
      continue
    try:
      gpu, memory_mb, utilization = map(int, fields)
    except ValueError:
      continue
    if gpu == gpu_index:
      return memory_mb <= 512 and utilization <= 3
  return False


def _port_free(port: int) -> bool:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1)
    return sock.connect_ex(("127.0.0.1", port)) != 0


def _load_services(root: Path) -> list[dict[str, Any]]:
  path = root / "vllm_services.json"
  if not path.is_file():
    return []
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in data if isinstance(row, dict)]
  except (json.JSONDecodeError, TypeError):
    return []


def _save_services(root: Path, services: list[dict[str, Any]]) -> None:
  _atomic_json(root / "vllm_services.json", services)


def _owned_process_matches(item: dict[str, Any]) -> bool:
  pid = item.get("pid")
  if not pid:
    return False
  cmdline_path = Path("/proc") / str(pid) / "cmdline"
  try:
    argv = cmdline_path.read_bytes().decode(errors="replace").split("\0")
  except OSError:
    return False
  joined = " ".join(part for part in argv if part)
  return all(token in joined for token in (
      "vllm.entrypoints.openai.api_server", "--port", str(item.get("port")),
      "--model", str(item.get("model_dir")), "--seed", "0"))


def _server_options(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
  """Return one or more endpoints; never reconfigure a foreign service."""
  own = _load_services(root)
  healthy: list[dict[str, Any]] = []
  stale: list[dict[str, Any]] = []
  loading: list[dict[str, Any]] = []
  for item in own:
    if not _owned_process_matches(item):
      stale.append(item)
      continue
    port = int(item["port"])
    if _endpoint_ok(port):
      healthy.append(item)
    else:
      loading.append(item)
  if stale:
    _save_services(root, [item for item in own if item not in stale])
  if healthy:
    return healthy[:args.max_workers]
  if loading:
    # A study-owned server may still be loading weights. Do not switch to or
    # start another service while it holds its dedicated GPU.
    return []

  free = [] if getattr(args, "reuse_vllm_only", False) else _free_gpus(args.max_workers)
  if free:
    port = args.port_start
    for gpu in free:
      # The initial snapshot is only a candidate list; revalidate immediately
      # before launch to avoid racing another user's scheduler.
      if not _gpu_idle(gpu):
        continue
      while not _port_free(port):
        port += 1
      log = root / "vllm" / f"gpu{gpu}-port{port}.log"
      log.parent.mkdir(parents=True, exist_ok=True)
      env = os.environ.copy()
      env["CUDA_VISIBLE_DEVICES"] = str(gpu)
      with log.open("ab", buffering=0) as stream:
        process = subprocess.Popen([
            args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", args.model_dir, "--served-model-name", MODEL_NAME,
            "--host", "0.0.0.0", "--port", str(port),
            "--tensor-parallel-size", "1", "--max-model-len", "65536",
            "--seed", "0",
        ], cwd=args.vllm_cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
      item = {"port": port, "gpu": gpu, "pid": process.pid, "owned": True,
              "started_at": _now(), "model_dir": args.model_dir,
              "sampling": {"temperature": 0.0, "top_p": 1.0, "vllm_seed": 0}}
      own.append(item)
      _save_services(root, own)
      healthy.append(item)
      port += 1
    deadline = time.monotonic() + args.vllm_start_timeout_s
    while time.monotonic() < deadline and not all(_endpoint_ok(int(x["port"])) for x in healthy):
      time.sleep(5)
    ready = [item for item in healthy if _endpoint_ok(int(item["port"]))]
    if ready:
      return ready
    raise RuntimeError("new vLLM process(es) did not become healthy; see run_root/vllm logs")

  if args.reuse_vllm_port is not None:
    port = int(args.reuse_vllm_port)
    if _endpoint_idle(port, args.idle_window_s, root / "status.json"):
      return [{"port": port, "gpu": None, "owned": False,
               "sampling": {"temperature": 0.0, "top_p": 1.0,
                            "vllm_seed": "endpoint default; client decoding is deterministic"}}]
  return []


def _docker_container_name(run_id: str, worker_index: int) -> str:
  safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", run_id)[:40]
  return f"aw-graph-{safe_id}-w{worker_index}"


def _ensure_container(args: argparse.Namespace, root: Path, worker_index: int) -> str:
  name = _docker_container_name(args.run_id, worker_index)
  inspect = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
  if inspect.returncode == 0:
    payload = json.loads(inspect.stdout)[0]
    labels = payload.get("Config", {}).get("Labels") or {}
    if labels.get("org.androidworld.study") != args.run_id:
      raise RuntimeError(f"container name collision; refusing to touch unowned container {name}")
    command = " ".join(payload.get("Config", {}).get("Cmd") or [])
    if "start_emu_headless.sh" not in command:
      raise RuntimeError(f"study container {name} has no Android emulator startup command; refusing reuse")
    expected_mounts = {str(args.repo.resolve()), str(root.resolve())}
    actual_mounts = {str(Path(mount.get("Source", "")).resolve())
                     for mount in payload.get("Mounts", [])}
    if not expected_mounts.issubset(actual_mounts):
      raise RuntimeError(f"owned container {name} has unexpected bind mounts; refusing reuse")
    running = _run(["docker", "inspect", "-f", "{{.State.Running}}", name]).stdout.strip()
    if running.lower() == "true":
      return name
    _run(["docker", "start", name], timeout=90)
    return name
  image = _run(["docker", "image", "inspect", args.image], timeout=15)
  del image
  command = [
      "docker", "run", "-d", "--name", name,
      "--label", "org.androidworld.study=" + args.run_id,
      "--mount", f"type=bind,src={args.repo},dst=/androidworld",
      "--mount", f"type=bind,src={root},dst=/study",
      "--device", "/dev/kvm", "--shm-size=2g",
      "--add-host=host.docker.internal:host-gateway",
      "--network", "bridge", args.image,
      "-lc", "/bin/bash /androidworld/docker_setup/start_emu_headless.sh && adb root && tail -f /dev/null",
  ]
  _run(command, timeout=120)
  return name


def _wait_for_emulator(container: str, timeout_s: int = 300) -> None:
  adb = "/opt/android/platform-tools/adb"
  deadline = time.monotonic() + timeout_s
  last = "not started"
  while time.monotonic() < deadline:
    result = subprocess.run([
        "docker", "exec", container, adb, "-s", "emulator-5554",
        "shell", "getprop", "sys.boot_completed"], capture_output=True,
        text=True, timeout=15)
    last = (result.stdout + result.stderr).strip()
    if result.returncode == 0 and result.stdout.strip() == "1":
      return
    time.sleep(5)
  raise RuntimeError(f"emulator did not boot in {timeout_s}s ({last[-500:]})")


def _preflight_worker(container: str) -> None:
  """Verify runner imports and a local Android UI tree before allocation.

  The preflight must not depend on guest networking: a previous benchmark task
  may intentionally have disabled Wi-Fi, while the next task can still be
  evaluated through UIAutomator.
  """
  code = (
      "import urllib.request; "
      "import openai, scipy, matplotlib; "
      "from android_world.agents import mobileexplorer, gelab_agent; "
      "from android_world import checkpointer; "
      "from android_world.env.android_world_controller import A11yMethod, get_controller; "
      "controller=get_controller(console_port=5554, "
      "adb_path='/opt/android/platform-tools/adb', grpc_port=8554, "
      "a11y_method=A11yMethod.UIAUTOMATOR); "
      "timestep=controller.reset(); "
      "elements=timestep.observation.get('ui_elements') or []; "
      "controller.close(); "
      "assert elements, 'AndroidWorld gRPC preflight returned no UI elements'; "
      "print('androidworld-worker-preflight-ok', len(elements))"
  )
  result = subprocess.run([
      "docker", "exec", "-e", f"PYTHONPATH={CONTAINER_SITE_OVERLAY}",
      "-w", "/androidworld", container, CONTAINER_PYTHON, "-c", code,
  ], capture_output=True, text=True, timeout=180)
  if result.returncode != 0 or "androidworld-worker-preflight-ok" not in result.stdout:
    raise RuntimeError(
        f"worker dependency preflight failed for {container}: "
        f"{(result.stdout + result.stderr)[-2500:]}"
    )


def _extract_episode(container: str, checkpoint_dir: str) -> dict[str, Any] | None:
  result = subprocess.run([
      "docker", "exec", "-e", f"PYTHONPATH={CONTAINER_SITE_OVERLAY}",
      "-w", "/androidworld", container,
      CONTAINER_PYTHON, "scripts/labserver_androidworld_graph_study_extract.py",
      "--checkpoint-dir", checkpoint_dir,
  ], capture_output=True, text=True, timeout=90)
  if result.returncode != 0:
    return None
  try:
    value = json.loads(result.stdout.strip().splitlines()[-1])
  except (json.JSONDecodeError, IndexError):
    return None
  return value if isinstance(value, dict) and value.get("task") else None


def _checkpoint_container_path(task: str, arm: str, attempt: str,
                               runner: str) -> str:
  root = f"/study/episodes/{task}/{arm}/{attempt}"
  if runner != "run.py":
    root += "/runner_output"
  return root + "/checkpoints"


def _is_pre_action_infra_failure(episode: dict[str, Any], request_delta: float,
                                request_counter_observed: bool) -> bool:
  """Identify a checkpointed reset failure that never reached the method."""
  return bool(
      _is_pre_action_infra_checkpoint(episode)
      and request_counter_observed and request_delta == 0
  )


def _is_pre_action_infra_checkpoint(episode: dict[str, Any]) -> bool:
  """Such a checkpoint is diagnostic evidence, never a completed trial."""
  steps = episode.get("episode_steps")
  no_steps = steps is None or steps == 0
  return bool(episode.get("episode_exception") and no_steps)


def _container_attempt_active(container: str, task: str, arm: str,
                              attempt_name: str) -> bool:
  marker = f"/study/episodes/{task}/{arm}/{attempt_name}"
  result = subprocess.run(["docker", "top", container, "-eo", "args"],
                          capture_output=True, text=True, timeout=20)
  return result.returncode == 0 and marker in result.stdout


def _graph_counts(path: Path) -> dict[str, int]:
  if not path.is_file():
    return {"nodes": 0, "edges": 0, "samples": 0, "skills": 0}
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return {"nodes": 0, "edges": 0, "samples": 0, "skills": 0}
  return {
      "nodes": int(data.get("node_count", len(data.get("states", [])))),
      "edges": int(data.get("edge_count", len(data.get("edges", [])))),
      "samples": int(data.get("sample_count", len(data.get("samples", [])))),
      "skills": int(data.get("skill_count", len(data.get("skills", [])))),
  }


def _trace_counts(attempt: Path) -> dict[str, Any]:
  runner_output = attempt / "runner_output"
  metrics_root = runner_output if runner_output.is_dir() else attempt
  probes = _read_jsonl(metrics_root / "probe_trace.jsonl")
  requests = _read_jsonl(metrics_root / "request_latency.jsonl")
  steps = _read_jsonl(metrics_root / "step_latency.jsonl")
  skip_events = _read_jsonl(metrics_root / "skip_events.jsonl")
  memory_events = _read_jsonl(metrics_root / "executable_memory_events.jsonl")
  action_rows: list[dict[str, Any]] = []
  for action_path in metrics_root.rglob("action.jsonl"):
    action_rows.extend(_read_jsonl(action_path))
  rollbacks = 0
  observations = 0
  for row in probes:
    rollbacks += len(row.get("rollbacks") or [])
    nested_observations = row.get("observations")
    if isinstance(nested_observations, list) and nested_observations:
      observations += len(nested_observations)
    elif isinstance(row.get("graph"), dict) and isinstance(row.get("discovered"), dict):
      # Current probe_trace.jsonl schema writes one row per probe outcome;
      # that row itself contains the resulting screen/graph observation.
      observations += 1
  if not requests and (attempt / "runner.log").is_file():
    # Raw gelab_agent uses the standard run.py logger rather than the online
    # wrapper's per-request JSONL.
    text = (attempt / "runner.log").read_text(encoding="utf-8", errors="ignore")
    primary_requests = len(re.findall(r"^Step\s+\d+: Model input$", text, re.MULTILINE))
  else:
    primary_requests = len(requests)
  graph_events = _read_jsonl(metrics_root / "graph_construction_perf.jsonl")
  graph_mutations = [row for row in graph_events
                     if row.get("operation") != "probe_trace_ingest_wall"]
  graph_construction_time_s = sum(
      max(0.0, float(row.get("elapsed_s") or 0.0)) for row in graph_mutations)
  probe_trace_ingest_wall_s = sum(
      max(0.0, float(row.get("elapsed_s") or 0.0)) for row in graph_events
      if row.get("operation") == "probe_trace_ingest_wall")
  route_gate_events = [row for row in memory_events
                       if row.get("event") == "high_confidence_route_gate"]
  route_gate_blocks: Counter[str] = Counter()
  route_gate_candidates = 0
  route_gate_eligible = 0
  for row in route_gate_events:
    route_gate_candidates += int(row.get("candidates", 0) or 0)
    route_gate_eligible += int(row.get("eligible_routes", 0) or 0)
    route_gate_blocks.update(row.get("blocks") or {})
  executable_memory_skip_steps = sum(
      1 for row in action_rows
      if row.get("reasoning_mode") == "high_confidence_skip")
  executable_memory_route_rollback_failures = sum(
      1 for row in action_rows
      if row.get("reasoning_mode") == "route_rollback_failed")
  two_system_route_attempts = sum(
      1 for row in skip_events if row.get("skip_kind") != "deterministic_bootstrap")
  two_system_route_hits = sum(
      1 for row in skip_events
      if row.get("skip_kind") != "deterministic_bootstrap"
      and row.get("successor_matched") is True)
  two_system_bootstrap_skips = sum(
      1 for row in skip_events if row.get("skip_kind") == "deterministic_bootstrap")
  two_system_skip_steps = sum(1 for row in steps if row.get("inference_skipped"))
  return {
      "probe_events": len(probes), "probe_observations": observations,
      "rollbacks": rollbacks, "primary_vlm_requests": primary_requests,
      "primary_prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in requests),
      "primary_generation_tokens": sum(int(row.get("generation_tokens") or 0) for row in requests),
      "inference_skipped_steps": two_system_skip_steps + executable_memory_skip_steps,
      "two_system_skip_route_attempts": two_system_route_attempts,
      "two_system_skip_route_hits": two_system_route_hits,
      "two_system_bootstrap_skips": two_system_bootstrap_skips,
      "executable_memory_skip_action_records": executable_memory_skip_steps,
      "executable_memory_route_rollback_failures": executable_memory_route_rollback_failures,
      "high_confidence_route_gate_queries": len(route_gate_events),
      "high_confidence_route_gate_candidates": route_gate_candidates,
      "high_confidence_route_gate_eligible": route_gate_eligible,
      "high_confidence_route_gate_blocks": dict(sorted(route_gate_blocks.items())),
      "graph_update_calls": len(graph_mutations),
      "graph_construction_time_s": graph_construction_time_s,
      "probe_trace_ingest_wall_s": probe_trace_ingest_wall_s,
  }


def _memory_metrics(attempt: Path) -> dict[str, Any]:
  runner_output = attempt / "runner_output"
  metrics_root = runner_output if runner_output.is_dir() else attempt
  path = metrics_root / "executable_memory_summary.json"
  if not path.is_file():
    return {}
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return {}
  metrics = data.get("metrics") or {}
  keys = ("exploration_guidance_queries", "exploration_guidance_state_matches",
          "exploration_guidance_controls_seen", "exploration_guidance_controls_relevant",
          "exploration_guidance_controls_mature", "prompt_context_queries",
          "prompt_context_count", "prompt_context_edges", "skip_attempts",
          "skip_hits", "route_hit_count", "route_miss_count", "recovery_failures",
          "state_merges", "graph_overrides", "graph_rejections", "probe_rounds",
          "probe_rounds_complete", "probes_target", "probes_completed",
          "retrieval_path_candidates")
  result = {key: metrics.get(key) for key in keys if key in metrics}
  if isinstance(metrics.get("stop_reasons"), dict):
    result["probe_stop_reasons"] = metrics["stop_reasons"]
  if "extra_time_s" in metrics:
    result["probe_critical_path_extension_s"] = metrics["extra_time_s"]
  result.update({key: data[key] for key in
                 ("node_count", "edge_count", "sample_count", "skill_count") if key in data})
  return result


def _attach_memory_summary_snapshots(
    root: Path, records: list[dict[str, Any]], protocol: dict[str, Any],
) -> list[dict[str, Any]]:
  """Reload safe per-attempt metric snapshots omitted by older supervisors."""
  tasks = set(protocol.get("tasks") or [])
  arms = {str(row.get("name")) for row in protocol.get("arms") or []}
  enriched = []
  for source in records:
    row = dict(source)
    task, arm, attempt_id = row.get("task"), row.get("arm"), row.get("attempt_id")
    if (task in tasks and arm in arms and isinstance(attempt_id, str)
        and re.fullmatch(r"attempt-\d{2,}", attempt_id)):
      snapshot = root / "episodes" / str(task) / str(arm) / attempt_id
      metrics = _memory_metrics(snapshot)
      row.update(metrics)
    # Older analyzer versions counted only a nested `observations` array, while
    # this trace schema records one observed result per probe row. Preserve
    # the original trace-row count as the canonical fallback for old records.
    if not row.get("probe_observations") and row.get("probe_events"):
      row["probe_observations"] = int(row["probe_events"])
    enriched.append(row)
  return enriched


def _run_one(args: argparse.Namespace, root: Path, protocol: dict[str, Any],
             container: str, endpoint: dict[str, Any], task: str,
             arm: dict[str, Any]) -> dict[str, Any]:
  arm_name = str(arm["name"])
  controls = protocol["controls"]
  a11y_method = _a11y_method_for_task(protocol, task)
  task_root = root / "episodes" / task / arm_name
  task_root.mkdir(parents=True, exist_ok=True)
  existing_rows = _read_jsonl(root / "records.jsonl")
  for row in reversed(existing_rows):
    if row.get("task") == task and row.get("arm") == arm_name and row.get("status") == "complete":
      return row

  for candidate in sorted(task_root.glob("attempt-*")):
    deadline = time.monotonic() + args.episode_timeout_s + 60
    while _container_attempt_active(container, task, arm_name, candidate.name):
      if time.monotonic() >= deadline:
        raise RuntimeError(f"previous attempt still active for {task}/{arm_name}; refusing duplicate execution")
      time.sleep(5)
    extracted = _extract_episode(
        container, _checkpoint_container_path(task, arm_name, candidate.name, arm["runner"]))
    if extracted and extracted["task"] == task:
      if _is_pre_action_infra_checkpoint(extracted):
        # Preserve this attempt and its checkpoint as failure evidence, but
        # never let a zero-action exception satisfy the task/arm completion set.
        continue
      record = _make_record(task, arm_name, extracted, candidate,
                            protocol, _trace_counts(candidate), _graph_counts(root / "memory" / arm_name / "executable_memory.json"))
      record.update(_memory_metrics(candidate))
      record.update({"status": "complete", "recovered_from_checkpoint": True,
                     "graph_delta_nodes": 0, "graph_delta_edges": 0,
                     "worker": container, "port": int(endpoint["port"]), "finished_at": _now()})
      _append_jsonl(root / "records.jsonl", record)
      return record

  attempt_n = 1 + max((int(p.name.split("-")[-1]) for p in task_root.glob("attempt-*")
                       if p.name.split("-")[-1].isdigit()), default=0)
  attempt = task_root / f"attempt-{attempt_n:02d}"
  attempt.mkdir(parents=True, exist_ok=False)
  checkpoint_dir = attempt / "checkpoints"
  graph_path = root / "memory" / arm_name / "executable_memory.json"
  graph_path.parent.mkdir(parents=True, exist_ok=True)
  before = _graph_counts(graph_path)
  config_path: Path | None = None
  if arm.get("config") is not None:
    config_path = attempt / "memory_config.json"
    _atomic_json(config_path, arm["config"])
  meta = {
      "task": task, "arm": arm_name, "attempt": attempt_n,
      "seed": protocol["controls"]["task_seed"],
      "started_at": _now(), "worker": container,
      "sampling": protocol["controls"],
      "vllm_endpoint_port": int(endpoint["port"]),
      "runner": arm["runner"],
      "config": arm.get("config"),
      "state": "running",
  }
  _atomic_json(attempt / "attempt.json", meta)
  _atomic_json(root / "status.json", {
      "state": "running", "run_id": args.run_id, "task": task,
      "arm": arm_name, "attempt": attempt_n, "worker": container,
      "port": int(endpoint["port"]), "updated_at": _now(),
  })

  api_base = f"http://host.docker.internal:{endpoint['port']}"
  try:
    server_before, _ = _metrics(f"http://127.0.0.1:{endpoint['port']}/metrics")
  except Exception:
    server_before = {}
  common_env = [
      "-e", f"ANDROID_WORLD_LLM_API_URL={api_base}/v1/chat/completions",
      "-e", f"ANDROID_WORLD_LLAMACPP_MODEL={MODEL_NAME}",
      "-e", f"PYTHONPATH={CONTAINER_SITE_OVERLAY}",
      "-e", "ANDROID_WORLD_ADB_PATH=/opt/android/platform-tools/adb",
      "-e", "ANDROID_WORLD_SERIAL=emulator-5554",
      "-e", f"ANDROID_WORLD_A11Y_METHOD={a11y_method}",
  ]
  if arm["runner"] == "run.py":
    command = [
        "docker", "exec", *common_env, "-w",
        f"/study/episodes/{task}/{arm_name}/{attempt.name}", container,
        CONTAINER_PYTHON, "/androidworld/run.py", "--suite_family=android_world",
        f"--agent_name={arm['agent']}", f"--tasks={task}",
        "--n_task_combinations=1", "--fixed_task_seed",
        f"--task_random_seed={protocol['controls']['task_seed']}",
        f"--max_n_steps={protocol['controls']['max_n_steps']}",
        "--console_port=5554", "--adb_path=/opt/android/platform-tools/adb",
        f"--checkpoint_dir=/study/episodes/{task}/{arm_name}/{attempt.name}/checkpoints",
    ]
  else:
    online_output = f"/study/episodes/{task}/{arm_name}/{attempt.name}/runner_output"
    common_env += ["-e", f"MOBILEEXPLORER_OUTPUT_PATH={online_output}"]
    command = [
        "docker", "exec", *common_env, "-w", "/androidworld", container,
        CONTAINER_PYTHON, "scripts/run_sensys30_online_task.py",
        f"--output={online_output}",
        f"--task={task}", f"--seed={protocol['controls']['task_seed']}",
        f"--max_steps={protocol['controls']['max_n_steps']}",
        f"--ranker={protocol['controls']['ranker']}",
        f"--metrics_url={api_base}/metrics", "--console_port=5554",
        f"--max_probes={protocol['controls']['max_probes']}",
        f"--min_probes={protocol['controls']['min_probes']}",
        f"--max_depth={protocol['controls']['max_depth']}",
        f"--max_exploration_time_s={protocol['controls']['max_exploration_time_s']}",
        f"--agent_name={arm['agent']}",
        f"--a11y_method={a11y_method}",
    ]
    if arm.get("two_system"):
      command.append("--two_system")
    if arm.get("config") is not None:
      command += ["--executable_memory", f"--executable_memory_config=/study/episodes/{task}/{arm_name}/{attempt.name}/memory_config.json",
                  f"--executable_memory_path=/study/memory/{arm_name}/executable_memory.json"]
    command.append("--force_min_probes" if controls.get("force_min_probes", False)
                   else "--no-force_min_probes")
    command.append(
        f"--post_inference_grace_s={controls.get('post_inference_grace_s', 4.0)}")
  _atomic_json(attempt / "command_metadata.json", {
      "command_program": command[0:8], "runner": arm["runner"],
      "task": task, "arm": arm_name,
      "a11y_method": a11y_method,
      "effective_features": {
          "two_system_enabled": bool(arm.get("two_system")),
          "executable_memory_enabled": arm.get("config") is not None,
          "executable_memory_high_confidence_skip_enabled": bool(
              (arm.get("config") or {}).get("high_confidence_skip_enabled", False)),
          "two_system_skip_inference_enabled": bool(
              controls.get("two_system_skip_inference_enabled", False)),
      },
      "exploration_controls": {
          "max_probes": controls["max_probes"],
          "min_probes": controls["min_probes"],
          "force_min_probes": bool(controls.get("force_min_probes", False)),
          "post_inference_grace_s": float(
              controls.get("post_inference_grace_s", 4.0)),
      },
      "fixed_settings": {"temperature": 0.0, "top_p": 1.0,
                         "task_seed": protocol["controls"]["task_seed"],
                         "vllm_seed": endpoint.get("sampling", {}).get("vllm_seed", "unknown")},
  })
  timeout = args.episode_timeout_s
  started = time.monotonic()
  log_path = attempt / "runner.log"
  with log_path.open("ab", buffering=0) as log_stream:
    process = subprocess.Popen(command, cwd=args.repo, stdout=log_stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
      rc = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
      process.terminate()
      try:
        process.wait(timeout=20)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
      rc = 124
  elapsed = time.monotonic() - started
  try:
    server_after, _ = _metrics(f"http://127.0.0.1:{endpoint['port']}/metrics")
  except Exception:
    server_after = {}
  server_request_delta = max(
      0.0, server_after.get("vllm:request_success_total", 0.0)
      - server_before.get("vllm:request_success_total", 0.0))
  extracted = _extract_episode(
      container, _checkpoint_container_path(task, arm_name, attempt.name, arm["runner"]))
  if not extracted or extracted.get("task") != task:
    meta.update({"state": "infra_failed", "finished_at": _now(), "return_code": rc,
                 "elapsed_s": round(elapsed, 3),
                 "reason": "no valid evaluator checkpoint; see private runner.log"})
    _atomic_json(attempt / "attempt.json", meta)
    event = {"status": "infra_failed", "task": task, "arm": arm_name,
             "attempt": attempt_n, "return_code": rc, "elapsed_s": round(elapsed, 3),
             "recorded_at": _now()}
    _append_jsonl(root / "events.jsonl", event)
    raise RuntimeError(f"{task}/{arm_name} ended rc={rc} without an evaluator checkpoint")
  request_counter_observed = (
      "vllm:request_success_total" in server_before
      and "vllm:request_success_total" in server_after
  )
  if _is_pre_action_infra_failure(extracted, server_request_delta,
                                  request_counter_observed):
    reason = "evaluator exception before the first action with zero observed VLM requests"
    meta.update({"state": "infra_failed", "finished_at": _now(),
                 "return_code": rc, "elapsed_s": round(elapsed, 3),
                 "reason": reason})
    _atomic_json(attempt / "attempt.json", meta)
    _append_jsonl(root / "events.jsonl", {
        "status": "infra_failed", "task": task, "arm": arm_name,
        "attempt": attempt_n, "return_code": rc,
        "elapsed_s": round(elapsed, 3), "reason": "pre_action_exception_no_vlm_request",
        "recorded_at": _now(),
    })
    raise RuntimeError(f"{task}/{arm_name} had a pre-action infrastructure exception; retrying in a fresh attempt")
  after = _graph_counts(graph_path)
  record = _make_record(task, arm_name, extracted, attempt, protocol,
                        _trace_counts(attempt), after)
  record.update(_memory_metrics(attempt))
  record.update({
      "status": "complete", "attempt": attempt_n,
      "runner_return_code": rc, "elapsed_s": round(elapsed, 3),
      "graph_delta_nodes": max(0, after["nodes"] - before["nodes"]),
      "graph_delta_edges": max(0, after["edges"] - before["edges"]),
      "worker": container, "port": int(endpoint["port"]),
      "vllm_requests_endpoint_delta": int(server_request_delta),
      "endpoint_owned_by_study": bool(endpoint.get("owned")),
      "finished_at": _now(),
  })
  meta.update({"state": "complete", "finished_at": record["finished_at"],
               "return_code": rc, "elapsed_s": record["elapsed_s"]})
  _atomic_json(attempt / "attempt.json", meta)
  _append_jsonl(root / "records.jsonl", record)
  _append_jsonl(root / "events.jsonl", {"status": "complete", "task": task,
      "arm": arm_name, "attempt": attempt_n, "success": record["success"],
      "episode_steps": record["episode_steps"], "elapsed_s": record["elapsed_s"],
      "recorded_at": _now()})
  return record


def _resume_argv(args: argparse.Namespace) -> list[str]:
  """Rebuild the supervisor command without losing explicit publish policy."""
  argv = [
      sys.executable, str(Path(__file__).resolve()), "run",
      "--run-root", str(args.run_root), "--run-id", args.run_id,
      "--repo", str(args.repo), "--protocol", str(args.protocol),
      "--image", args.image, "--model-dir", args.model_dir,
      "--vllm-python", args.vllm_python, "--vllm-cwd", args.vllm_cwd,
      "--max-workers", str(args.max_workers), "--port-start", str(args.port_start),
      "--reuse-vllm-port", str(args.reuse_vllm_port) if args.reuse_vllm_port is not None else "-1",
      "--idle-window-s", str(args.idle_window_s), "--resource-poll-s", str(args.resource_poll_s),
      "--vllm-start-timeout-s", str(args.vllm_start_timeout_s),
      "--episode-timeout-s", str(args.episode_timeout_s),
      "--failure-backoff-s", str(args.failure_backoff_s),
      "--publish" if args.publish else "--no-publish",
  ]
  if getattr(args, "reuse_vllm_only", False):
    argv.append("--reuse-vllm-only")
  return argv


def _make_record(task: str, arm: str, episode: dict[str, Any], attempt: Path,
                 protocol: dict[str, Any], trace: dict[str, Any],
                 graph: dict[str, int]) -> dict[str, Any]:
  return {
      "task": task, "arm": arm, "seed": int(protocol["controls"]["task_seed"]),
      "success": bool(episode.get("success")),
      "evaluator_score": episode.get("evaluator_score"),
      "episode_steps": episode.get("episode_steps"),
      "evaluator_complete": bool(episode.get("evaluator_complete")),
      "episode_exception": bool(episode.get("episode_exception")),
      **trace,
      "graph_nodes": graph["nodes"], "graph_edges": graph["edges"],
      "graph_samples": graph["samples"], "graph_skills": graph["skills"],
      "attempt_id": attempt.name,
  }


def _write_status(root: Path, state: str, message: str, **data: Any) -> None:
  _atomic_json(root / "status.json", {"state": state, "message": message,
                                      "updated_at": _now(), **data})


def _validate_protocol(protocol: dict[str, Any]) -> None:
  if protocol.get("schema_version") != 1:
    raise ValueError("unsupported protocol schema")
  controls = protocol.get("controls") or {}
  if controls.get("client_temperature") != 0.0 or controls.get("client_top_p") != 1.0:
    raise ValueError("protocol must preserve temperature=0 and top_p=1")
  if controls.get("a11y_method") != "grpc":
    raise ValueError("matched study arms must use the frozen AndroidWorld gRPC accessibility path")
  if not controls.get("fixed_task_seed") or not isinstance(controls.get("task_seed"), int):
    raise ValueError("fixed integer task seed is required")
  if "force_min_probes" in controls and not isinstance(controls["force_min_probes"], bool):
    raise ValueError("force_min_probes must be a boolean")
  grace_s = controls.get("post_inference_grace_s", 4.0)
  if (not isinstance(grace_s, (int, float)) or isinstance(grace_s, bool)
      or not math.isfinite(grace_s) or grace_s < 0 or grace_s > 60):
    raise ValueError("post_inference_grace_s must be finite and between 0 and 60 seconds")
  if not protocol.get("tasks") or not protocol.get("arms"):
    raise ValueError("protocol tasks/arms must not be empty")
  a11y_overrides = controls.get("a11y_method_overrides", {})
  if not isinstance(a11y_overrides, dict):
    raise ValueError("a11y_method_overrides must be a task-to-method mapping")
  unknown_override_tasks = set(a11y_overrides) - set(protocol["tasks"])
  if unknown_override_tasks:
    raise ValueError(
        "a11y method overrides reference tasks outside this cohort: "
        + ", ".join(sorted(unknown_override_tasks))
    )
  allowed_a11y_methods = {"grpc", "uiautomator", "fast_provider"}
  invalid_override_methods = [
      method for method in a11y_overrides.values()
      if not isinstance(method, str) or method not in allowed_a11y_methods
  ]
  if invalid_override_methods:
    raise ValueError(
        "unsupported a11y method override(s): "
        + ", ".join(sorted(map(str, invalid_override_methods)))
    )
  names = [row.get("name") for row in protocol["arms"]]
  if len(names) != len(set(names)):
    raise ValueError("arm names must be unique")
  task_names = protocol["tasks"]
  if len(task_names) != len(set(task_names)):
    raise ValueError("task IDs must be unique")
  if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in task_names):
    raise ValueError("task IDs must be registry identifiers, not paths")
  if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in names):
    raise ValueError("arm names must be safe identifiers")


def _a11y_method_for_task(protocol: dict[str, Any], task: str) -> str:
  controls = protocol["controls"]
  return controls.get("a11y_method_overrides", {}).get(
      task, controls["a11y_method"]
  )


def run_study(args: argparse.Namespace) -> int:
  root = args.run_root.resolve()
  studies_base = Path("/data/rxhuang/android_world_server/androidworld_graph_studies").resolve()
  if root == Path("/") or root.parent != studies_base:
    raise ValueError(f"unsafe run root: {root}")
  root.mkdir(parents=True, exist_ok=True)
  lock_path = root / ".study.lock"
  lock_stream = lock_path.open("a+")
  try:
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
  except BlockingIOError as exc:
    raise RuntimeError(f"another supervisor holds {lock_path}") from exc

  protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
  _validate_protocol(protocol)
  args.repo = args.repo.resolve()
  if not (args.repo / "run.py").is_file():
    raise FileNotFoundError(f"AndroidWorld repo not found: {args.repo}")
  manifest = {
      "schema_version": 1, "run_id": args.run_id,
      "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
      "source_commit": _run(["git", "-C", str(args.repo), "rev-parse", "HEAD"]).stdout.strip(),
      "started_at": _now(), "repo_branch": _run(["git", "-C", str(args.repo), "branch", "--show-current"]).stdout.strip(),
      "protocol": protocol, "host": socket.gethostname(),
      "privacy": "private traces/checkpoints/logs remain under run_root; only aggregate report is published",
  }
  if not (root / "study_manifest.json").exists():
    _atomic_json(root / "study_manifest.json", manifest)
  else:
    old = json.loads((root / "study_manifest.json").read_text(encoding="utf-8"))
    if old.get("protocol_sha256") != manifest["protocol_sha256"] or old.get("source_commit") != manifest["source_commit"]:
      raise RuntimeError("resume refused: protocol or source commit differs from the existing manifest")

  _run(["docker", "image", "inspect", args.image], timeout=20)
  try:
    while True:
      protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
      complete_target = len(protocol["arms"]) * len(protocol["tasks"])
      completed = {(row.get("task"), row.get("arm"))
                   for row in _read_jsonl(root / "records.jsonl")
                   if row.get("status") == "complete"}
      if len(completed) >= complete_target:
        break
      services = _server_options(args, root)
      if not services:
        _write_status(root, "waiting_for_resources",
                      "No verified idle vLLM endpoint or empty GPU; retrying without touching foreign jobs.",
                      gpu_snapshot=_run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], check=False).stdout.strip(),
                      checked_at=_now())
        time.sleep(args.resource_poll_s)
        continue
      containers = [_ensure_container(args, root, i) for i in range(len(services))]
      for container in containers:
        _wait_for_emulator(container)
        _preflight_worker(container)
      _write_status(root, "running", "AndroidWorld matrix is running.",
                    workers=containers, ports=[s["port"] for s in services], updated_at=_now())
      rows = _read_jsonl(root / "records.jsonl")
      completed = {(row.get("task"), row.get("arm")) for row in rows if row.get("status") == "complete"}
      arms = protocol["arms"]
      tasks = protocol["tasks"]
      complete_target = len(arms) * len(tasks)
      completed_count = len(completed)
      if completed_count >= complete_target:
        break
      rng = random.Random(int(protocol["controls"]["randomization_seed"]))
      task_order = list(tasks)
      rng.shuffle(task_order)
      for task_idx, task in enumerate(task_order):
        remaining = [arm for arm in arms if (task, arm["name"]) not in completed]
        rng.shuffle(remaining)
        for offset in range(0, len(remaining), len(services)):
          batch = remaining[offset:offset + len(services)]
          jobs: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
          for j, arm in enumerate(batch):
            worker_idx = (task_idx + offset + j) % len(services)
            jobs.append((arm, containers[worker_idx], services[worker_idx]))
          failure: BaseException | None = None
          with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(_run_one, args, root, protocol, container,
                                   endpoint, task, arm)
                       for arm, container, endpoint in jobs]
            for future in concurrent.futures.as_completed(futures):
              try:
                future.result()
              except BaseException as exc:  # persist and leave checkpoints intact
                failure = exc
          records_now = _read_jsonl(root / "records.jsonl")
          completed = {(row.get("task"), row.get("arm")) for row in records_now if row.get("status") == "complete"}
          _write_status(root, "running", "Progress checkpoint saved.",
                        completed_episodes=len(completed), total_episodes=complete_target,
                        last_task=task, last_error=str(failure)[:500] if failure else None,
                        updated_at=_now())
          if failure:
            _write_status(root, "recovering", "A task attempt failed or was classified as a pre-action infrastructure error; preserving artifacts and retrying safely.",
                          error=str(failure)[:1000], completed_episodes=len(completed),
                          total_episodes=complete_target, updated_at=_now())
            time.sleep(args.failure_backoff_s)
            # A fresh endpoint/container check occurs at the next outer pass.
            raise RuntimeError(str(failure))
        if len(completed) >= complete_target:
          break
      if len(completed) < complete_target:
        continue
  except KeyboardInterrupt:
    _write_status(root, "interrupted", "Supervisor received an explicit interrupt; checkpoints are preserved.", updated_at=_now())
    return 130
  except Exception as exc:
    _append_jsonl(root / "events.jsonl", {"status": "supervisor_error", "error": str(exc)[:2000], "time": _now()})
    _write_status(root, "recovering", "Supervisor recorded an error; checkpoints are preserved and it will retry.",
                  error=str(exc)[:2000], updated_at=_now())
    time.sleep(args.failure_backoff_s)
    # Re-exec the same script to keep the lock/pid lifecycle simple. A stale
    # process cannot overwrite attempts; task-level checkpoints are adopted.
    lock_stream.close()
    os.execv(sys.executable, _resume_argv(args))
  _write_status(root, "analyzing", "All planned task/arm episodes have checkpoints; building aggregate report.", updated_at=_now())
  analyze(root, protocol)
  if args.publish:
    while True:
      try:
        publish_report(args.repo, root, protocol, args.run_id)
        break
      except Exception as exc:
        _append_jsonl(root / "events.jsonl", {"status": "publish_retry", "error": str(exc)[:1200], "time": _now()})
        _write_status(root, "publish_pending", "Experiment is complete; waiting to safely publish aggregate results to GitHub.",
                      error=str(exc)[:1200], updated_at=_now())
        time.sleep(args.failure_backoff_s)
  _write_status(root, "complete", "Study and analysis completed.",
                completed_episodes=complete_target, total_episodes=complete_target,
                report=str(root / "report.md"), completed_at=_now())
  # Keep explicitly started vLLM services available for later work; stop only
  # the Docker containers created by this study, without deleting their data.
  for container in locals().get("containers", []):
    subprocess.run(["docker", "stop", "-t", "20", container], capture_output=True, timeout=45)
  return 0


def _mean(values: list[float]) -> float | None:
  return sum(values) / len(values) if values else None


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
  if n <= 0:
    return None, None
  p = successes / n
  den = 1 + z * z / n
  center = (p + z * z / (2 * n)) / den
  radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
  return max(0.0, center - radius), min(1.0, center + radius)


_CUMULATIVE_MEMORY_COUNTERS = (
    "exploration_guidance_queries", "exploration_guidance_state_matches",
    "exploration_guidance_controls_seen", "exploration_guidance_controls_relevant",
    "exploration_guidance_controls_mature", "prompt_context_queries",
    "prompt_context_count", "prompt_context_edges", "skip_attempts",
    "skip_hits", "route_hit_count", "route_miss_count", "recovery_failures",
    "state_merges", "graph_overrides", "graph_rejections", "probe_rounds",
    "probe_rounds_complete", "probes_target", "probes_completed",
    "retrieval_path_candidates",
)


def _reconstruct_snapshot_deltas(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Recover per-trial graph growth and cumulative memory-counter deltas.

  The shared graph JSON is container-owned and may be unreadable to the host
  supervisor. Per-attempt executable-memory summaries are readable but store
  cumulative graph counts and counters. Difference those snapshots in task
  completion order before computing arm aggregates; otherwise every trial
  repeats all previous graph, prompt, probe, and route metrics.
  """
  normalized = [dict(row) for row in records]
  by_arm: dict[str, list[tuple[int, dict[str, Any]]]] = {}
  for index, row in enumerate(normalized):
    by_arm.setdefault(str(row.get("arm", "")), []).append((index, row))
  for rows in by_arm.values():
    rows.sort(key=lambda pair: (str(pair[1].get("finished_at") or ""), pair[0]))
    previous_nodes = 0
    previous_edges = 0
    previous_metrics: dict[str, float] = {}
    previous_stop_reasons: dict[str, int] = {}
    for _, row in rows:
      node_count = row.get("node_count")
      edge_count = row.get("edge_count")
      has_snapshot = (
          isinstance(node_count, (int, float)) and not isinstance(node_count, bool)
          and isinstance(edge_count, (int, float)) and not isinstance(edge_count, bool)
      )
      if has_snapshot:
        current_nodes = max(0, int(node_count))
        current_edges = max(0, int(edge_count))
        row["graph_nodes"] = current_nodes
        row["graph_edges"] = current_edges
        row["graph_delta_nodes"] = max(0, current_nodes - previous_nodes)
        row["graph_delta_edges"] = max(0, current_edges - previous_edges)
        row["graph_delta_source"] = "successive_arm_snapshots"
        previous_nodes, previous_edges = current_nodes, current_edges
        metric_fields = [key for key in _CUMULATIVE_MEMORY_COUNTERS
                         if isinstance(row.get(key), (int, float))
                         and not isinstance(row.get(key), bool)]
        for key in metric_fields:
          cumulative = float(row[key])
          row[key] = max(0, cumulative - previous_metrics.get(key, 0.0))
          previous_metrics[key] = cumulative
        cumulative_extra_time = row.get("probe_critical_path_extension_s")
        if (isinstance(cumulative_extra_time, (int, float))
            and not isinstance(cumulative_extra_time, bool)):
          row["probe_critical_path_extension_s"] = max(
              0.0, float(cumulative_extra_time)
              - previous_metrics.get("probe_critical_path_extension_s", 0.0))
          previous_metrics["probe_critical_path_extension_s"] = float(cumulative_extra_time)
        stop_reasons = row.get("probe_stop_reasons")
        if isinstance(stop_reasons, dict):
          current_stop_reasons = {
              str(key): max(0, int(value)) for key, value in stop_reasons.items()
              if isinstance(value, (int, float)) and not isinstance(value, bool)
          }
          row["probe_stop_reasons"] = {
              key: max(0, value - previous_stop_reasons.get(key, 0))
              for key, value in current_stop_reasons.items()
              if value - previous_stop_reasons.get(key, 0) > 0
          }
          previous_stop_reasons = current_stop_reasons
        row["memory_metric_delta_source"] = "successive_arm_snapshots"
      else:
        row.setdefault("graph_nodes", 0)
        row.setdefault("graph_edges", 0)
        row.setdefault("graph_delta_nodes", 0)
        row.setdefault("graph_delta_edges", 0)
        row["graph_delta_source"] = "no_executable_memory_snapshot"
  return normalized


def _bootstrap_mean_ci(values: list[float], seed: int, samples: int = 4000) -> list[float | None]:
  if not values:
    return [None, None]
  if len(values) == 1:
    return [values[0], values[0]]
  rng = random.Random(seed)
  means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples))
  return [means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]]


def _svg_bars(title: str, labels: list[str], values: list[float], ylabel: str,
              path: Path, baseline_zero: bool = False) -> None:
  width, height = 1200, 600
  left, right, top, bottom = 100, 35, 80, 150
  plot_h, plot_w = height - top - bottom, width - left - right
  low = min(0.0, min(values, default=0.0)) if baseline_zero else 0.0
  high = max(values, default=1.0)
  high = high if high > low else low + 1.0
  pad = (high - low) * 0.12
  low -= pad if baseline_zero else 0.0
  high += pad
  y = lambda v: top + (high - v) / (high - low) * plot_h
  gap = plot_w / max(1, len(values))
  bar_w = gap * 0.62
  elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              f'<text x="{width/2:.1f}" y="38" text-anchor="middle" font-family="Arial" font-size="24" font-weight="bold">{title}</text>',
              f'<text x="24" y="{top + plot_h/2:.1f}" transform="rotate(-90 24 {top + plot_h/2:.1f})" text-anchor="middle" font-family="Arial" font-size="15">{ylabel}</text>']
  colors = ["#526579" if i == 0 else "#3478b8" for i in range(len(values))]
  for tick in range(6):
    value = low + (high - low) * tick / 5
    yy = y(value)
    elements.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#d9e0e7"/>')
    elements.append(f'<text x="{left-10}" y="{yy+5:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.2f}</text>')
  base_y = y(max(0.0, low))
  elements.append(f'<line x1="{left}" y1="{base_y:.1f}" x2="{width-right}" y2="{base_y:.1f}" stroke="#334155" stroke-width="1.5"/>')
  for i, (label, value) in enumerate(zip(labels, values)):
    x = left + i * gap + (gap - bar_w) / 2
    yy = y(value)
    bar_y = min(yy, base_y)
    bar_h = max(1, abs(base_y - yy))
    elements.append(f'<rect x="{x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" fill="{colors[i]}"/>')
    elements.append(f'<text x="{x+bar_w/2:.1f}" y="{bar_y-8:.1f}" text-anchor="middle" font-family="Arial" font-size="13">{value:.2f}</text>')
    words = label.replace("_", " ").split()
    for j in range(0, len(words), 2):
      row = " ".join(words[j:j+2])
      elements.append(f'<text x="{x+bar_w/2:.1f}" y="{height-bottom+20+(j//2)*16}" text-anchor="middle" font-family="Arial" font-size="11">{row}</text>')
  elements.append("</svg>")
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _summarize(records: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
  records = _reconstruct_snapshot_deltas(records)
  by_arm: dict[str, list[dict[str, Any]]] = {arm["name"]: [] for arm in protocol["arms"]}
  for row in records:
    if row.get("status") == "complete" and row.get("arm") in by_arm:
      by_arm[row["arm"]].append(row)
  arms_summary: dict[str, Any] = {}
  for arm, rows in by_arm.items():
    valid = [r for r in rows if r.get("evaluator_complete") and isinstance(r.get("episode_steps"), (int, float))]
    success = [r for r in rows if r.get("success")]
    route_gate_blocks: Counter[str] = Counter()
    probe_stop_reasons: Counter[str] = Counter()
    for row in rows:
      route_gate_blocks.update(row.get("high_confidence_route_gate_blocks") or {})
      probe_stop_reasons.update(row.get("probe_stop_reasons") or {})
    n_success = len(success)
    ci = _wilson(n_success, len(rows))
    all_steps = [float(r["episode_steps"]) for r in valid]
    success_steps = [float(r["episode_steps"]) for r in success]
    arms_summary[arm] = {
        "n_recorded": len(rows), "n_trials": len(rows),
        "n_evaluator_complete": len(valid),
        "n_episode_exceptions": sum(1 for r in rows if r.get("episode_exception")),
        "n_success": n_success,
        "success_rate": n_success / len(rows) if rows else None,
        "success_rate_wilson_95": list(ci),
        "mean_steps_all_completed": _mean(all_steps),
        "mean_steps_successes": _mean(success_steps),
        "total_steps_all_completed": sum(all_steps),
        "total_steps_successes": sum(success_steps),
        "probe_events": sum(int(r.get("probe_events", 0)) for r in rows),
        "probe_observations": sum(int(r.get("probe_observations", 0)) for r in rows),
        "probe_rounds": sum(int(r.get("probe_rounds", 0) or 0) for r in rows),
        "probe_rounds_complete": sum(int(r.get("probe_rounds_complete", 0) or 0) for r in rows),
        "probes_target": sum(int(r.get("probes_target", 0) or 0) for r in rows),
        "probes_completed": sum(int(r.get("probes_completed", 0) or 0) for r in rows),
        "retrieval_path_candidates": sum(int(r.get("retrieval_path_candidates", 0) or 0) for r in rows),
        "probe_round_completion_rate": (
            sum(int(r.get("probe_rounds_complete", 0) or 0) for r in rows)
            / sum(int(r.get("probe_rounds", 0) or 0) for r in rows)
            if sum(int(r.get("probe_rounds", 0) or 0) for r in rows) else None),
        "probe_budget_utilization": (
            sum(int(r.get("probes_completed", 0) or 0) for r in rows)
            / sum(int(r.get("probes_target", 0) or 0) for r in rows)
            if sum(int(r.get("probes_target", 0) or 0) for r in rows) else None),
        "probe_stop_reasons": dict(sorted(probe_stop_reasons.items())),
        "exploration_guidance_queries": sum(
            int(r.get("exploration_guidance_queries", 0) or 0) for r in rows),
        "exploration_guidance_state_matches": sum(
            int(r.get("exploration_guidance_state_matches", 0) or 0) for r in rows),
        "exploration_guidance_controls_seen": sum(
            int(r.get("exploration_guidance_controls_seen", 0) or 0) for r in rows),
        "exploration_guidance_controls_relevant": sum(
            int(r.get("exploration_guidance_controls_relevant", 0) or 0) for r in rows),
        "exploration_guidance_controls_mature": sum(
            int(r.get("exploration_guidance_controls_mature", 0) or 0) for r in rows),
        "rollbacks": sum(int(r.get("rollbacks", 0)) for r in rows),
        "primary_vlm_requests": sum(int(r.get("primary_vlm_requests", 0)) for r in rows),
        "vllm_requests_endpoint_delta": sum(
            int(r.get("vllm_requests_endpoint_delta", 0)) for r in rows),
        "prompt_context_edges": sum(int(r.get("prompt_context_edges", 0) or 0) for r in rows),
        "prompt_context_queries": sum(int(r.get("prompt_context_queries", 0) or 0) for r in rows),
        "inference_skipped_steps": sum(int(r.get("inference_skipped_steps", 0)) for r in rows),
        "two_system_skip_route_attempts": sum(int(r.get("two_system_skip_route_attempts", 0)) for r in rows),
        "two_system_skip_route_hits": sum(int(r.get("two_system_skip_route_hits", 0)) for r in rows),
        "two_system_bootstrap_skips": sum(int(r.get("two_system_bootstrap_skips", 0)) for r in rows),
        "executable_memory_skip_action_records": sum(int(r.get("executable_memory_skip_action_records", 0)) for r in rows),
        "executable_memory_route_rollback_failures": sum(int(r.get("executable_memory_route_rollback_failures", 0)) for r in rows),
        "high_confidence_route_gate_queries": sum(int(r.get("high_confidence_route_gate_queries", 0)) for r in rows),
        "high_confidence_route_gate_candidates": sum(int(r.get("high_confidence_route_gate_candidates", 0)) for r in rows),
        "high_confidence_route_gate_eligible": sum(int(r.get("high_confidence_route_gate_eligible", 0)) for r in rows),
        "high_confidence_route_gate_blocks": dict(sorted(route_gate_blocks.items())),
        "graph_update_calls": sum(int(r.get("graph_update_calls", 0) or 0) for r in rows),
        "graph_construction_time_s": sum(float(r.get("graph_construction_time_s", 0.0) or 0.0) for r in rows),
        "probe_trace_ingest_wall_s": sum(float(r.get("probe_trace_ingest_wall_s", 0.0) or 0.0) for r in rows),
        "probe_critical_path_extension_s": sum(float(r.get("probe_critical_path_extension_s", 0.0) or 0.0) for r in rows),
        "verified_route_hits": sum(int(r.get("route_hit_count", 0) or 0) for r in rows),
        "verified_route_misses": sum(int(r.get("route_miss_count", 0) or 0) for r in rows),
        "skip_attempts": sum(int(r.get("skip_attempts", 0) or 0) for r in rows),
        "skip_hits": sum(int(r.get("skip_hits", 0) or 0) for r in rows),
        "graph_nodes_final_max": max((int(r.get("graph_nodes", 0)) for r in rows), default=0),
        "graph_edges_final_max": max((int(r.get("graph_edges", 0)) for r in rows), default=0),
        "graph_new_nodes": sum(int(r.get("graph_delta_nodes", 0)) for r in rows),
        "graph_new_edges": sum(int(r.get("graph_delta_edges", 0)) for r in rows),
        "episode_ids": sorted(r["task"] for r in rows),
    }
  base = {r["task"]: r for r in by_arm.get("baseline", [])}
  comparisons: dict[str, Any] = {}
  for arm, rows in by_arm.items():
    if arm == "baseline":
      continue
    paired = [(base[r["task"]], r) for r in rows if r["task"] in base]
    success_diffs = [float(b["success"]) - float(a["success"]) for a, b in paired]
    valid_step_pairs = [(a, b) for a, b in paired
                        if a.get("evaluator_complete") and b.get("evaluator_complete")
                        and isinstance(a.get("episode_steps"), (int, float))
                        and isinstance(b.get("episode_steps"), (int, float))]
    all_step_diffs = [float(b["episode_steps"]) - float(a["episode_steps"])
                      for a, b in valid_step_pairs]
    common_success = [(a, b) for a, b in paired if a.get("success") and b.get("success")]
    success_step_diffs = [float(b["episode_steps"]) - float(a["episode_steps"]) for a, b in common_success]
    comparisons[arm] = {
        "paired_n": len(paired),
        "paired_step_n": len(valid_step_pairs),
        "paired_success_rate_delta": _mean(success_diffs),
        "paired_success_rate_delta_bootstrap_95": _bootstrap_mean_ci(success_diffs, 34030 + len(arm)),
        "paired_all_steps_delta_variant_minus_baseline": _mean(all_step_diffs),
        "paired_all_steps_delta_bootstrap_95": _bootstrap_mean_ci(all_step_diffs, 34031 + len(arm)),
        "common_success_n": len(common_success),
        "paired_common_success_steps_delta_variant_minus_baseline": _mean(success_step_diffs),
        "paired_common_success_steps_delta_bootstrap_95": _bootstrap_mean_ci(success_step_diffs, 34032 + len(arm)),
        "point_estimate_meets_success_noninferiority": bool(
            paired and _mean(success_diffs) is not None and _mean(success_diffs) >= 0.0),
        "paired_success_ci_lower_ge_zero": bool(
            paired and _bootstrap_mean_ci(success_diffs, 34030 + len(arm))[0] is not None
            and _bootstrap_mean_ci(success_diffs, 34030 + len(arm))[0] >= 0.0),
        "point_estimate_reduces_all_steps": bool(
            all_step_diffs and _mean(all_step_diffs) is not None and _mean(all_step_diffs) < 0.0),
        "point_estimate_reduces_steps_on_shared_successes": bool(
            success_step_diffs and _mean(success_step_diffs) is not None and _mean(success_step_diffs) < 0.0),
    }
  return {"arms": arms_summary, "paired_vs_baseline": comparisons,
          "n_planned_tasks": len(protocol["tasks"]),
          "n_planned_arms": len(protocol["arms"]),
          "n_complete_task_arm_pairs": sum(len(rows) for rows in by_arm.values()),
          "interpretation": "Success-rate non-inferiority is not established by point estimates alone; inspect paired confidence intervals and sample size.",
          "records": records}


def _accessibility_report_text(protocol: dict[str, Any]) -> str:
  controls = protocol.get("controls") or {}
  default_method = str(controls.get("a11y_method") or "unspecified")
  overrides = controls.get("a11y_method_overrides") or {}
  if not overrides:
    return ("Accessibility capture is frozen per task across all arms and uses the "
            f"configured default `{default_method}`; no task-specific overrides are set.")
  details = ", ".join(
      f"`{task}`=`{method}`" for task, method in sorted(overrides.items()))
  return ("Accessibility capture is frozen per task across all arms. "
          f"Default `{default_method}`; task-specific overrides: {details}.")


def analyze(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
  records = _read_jsonl(root / "records.jsonl")
  records = _attach_memory_summary_snapshots(root, records, protocol)
  summary = _summarize(records, protocol)
  analysis_records = summary["records"]
  _atomic_json(root / "aggregate_summary.json", {key: value for key, value in summary.items() if key != "records"})
  with (root / "per_task_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
    fields = ["task", "arm", "seed", "success", "evaluator_score", "episode_steps", "evaluator_complete",
              "episode_exception", "probe_events", "probe_observations", "rollbacks",
      "primary_vlm_requests", "primary_prompt_tokens", "primary_generation_tokens",
      "inference_skipped_steps", "graph_nodes", "graph_edges", "graph_delta_nodes",
              "two_system_skip_route_attempts", "two_system_skip_route_hits",
              "two_system_bootstrap_skips", "executable_memory_skip_action_records",
              "executable_memory_route_rollback_failures",
              "high_confidence_route_gate_queries", "high_confidence_route_gate_candidates",
              "high_confidence_route_gate_eligible", "high_confidence_route_gate_blocks",
              "probe_rounds", "probe_rounds_complete", "probes_target", "probes_completed",
              "retrieval_path_candidates", "probe_stop_reasons",
              "graph_delta_edges", "extra_time_s", "exploration_guidance_queries",
              "exploration_guidance_state_matches", "exploration_guidance_controls_seen",
              "exploration_guidance_controls_relevant", "exploration_guidance_controls_mature",
              "graph_update_calls", "graph_construction_time_s", "probe_trace_ingest_wall_s",
              "probe_critical_path_extension_s",
      "prompt_context_queries", "prompt_context_edges", "skip_attempts", "skip_hits", "route_hit_count",
              "route_miss_count", "vllm_requests_endpoint_delta", "graph_delta_source"]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows({
        **row,
        "high_confidence_route_gate_blocks": json.dumps(
            row.get("high_confidence_route_gate_blocks") or {}, sort_keys=True),
        "probe_stop_reasons": json.dumps(
            row.get("probe_stop_reasons") or {}, sort_keys=True),
    } for row in analysis_records)

  arm_names = [arm["name"] for arm in protocol["arms"]]
  summaries = summary["arms"]
  success_vals = [float(summaries[n]["success_rate"] or 0.0) for n in arm_names]
  step_vals = [float(summaries[n]["mean_steps_all_completed"] or 0.0) for n in arm_names]
  _svg_bars("AndroidWorld Success Rate by Design Arm", arm_names, success_vals,
            "success rate", root / "success_rate.svg")
  _svg_bars("Mean Action Steps (all evaluator-complete episodes)", arm_names,
            step_vals, "environment actions / task", root / "mean_steps.svg")
  success_step_vals = [float(summaries[n]["mean_steps_successes"] or 0.0) for n in arm_names]
  _svg_bars("Mean Action Steps on Successful Episodes", arm_names,
            success_step_vals, "environment actions / successful task", root / "success_steps.svg")

  baseline = summaries.get("baseline", {})
  lines = [
      "# AndroidWorld Task-Conditioned Graph Study",
      "",
      f"Study ID: `{root.name}`  ",
      f"Tasks planned: {len(protocol['tasks'])}; arms: {len(protocol['arms'])}  ",
      f"Controls: task seed {protocol['controls']['task_seed']} (fixed), client temperature 0, top_p 1, vLLM seed 0 for services started by this runner.",
      _accessibility_report_text(protocol),
      "",
      "This report uses AndroidWorld evaluator checkpoint metadata for success and episode steps. It does not infer success from agent text. Only aggregate metrics, public task IDs, protocol details, and SVG charts are intended for publication; raw prompts, goals, screenshots, logs, and checkpoints remain server-side.",
      "VLM request totals use the per-episode vLLM successful-request counter delta. Local primary request traces remain in the per-task CSV for diagnostics; they can be absent for some runner modes. When reusing a shared endpoint, concurrent external clients could affect endpoint deltas, so interpret them with service ownership and the idle-check record.",
      "Per-task graph deltas are reconstructed in completion order from cumulative attempt-level executable-memory summaries. The shared graph JSON may be container-owned and unreadable to the host; raw records remain preserved, while CSV and aggregates mark the derived delta source.",
      "",
      "## Main outcomes",
      "",
      "| Arm | Trials | Evaluated steps | Exceptions | Success | Success rate (Wilson 95% CI) | Total observed actions | Mean actions | Mean actions, success only | VLLM requests (endpoint delta) | Probe observations | Rollbacks | Prompt graph edges | Skip hits / attempts | New graph nodes / edges |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for name in arm_names:
    row = summaries[name]
    lo, hi = row["success_rate_wilson_95"]
    pct = "n/a" if row["success_rate"] is None else f"{row['success_rate']:.1%}"
    ci = "n/a" if lo is None else f"[{lo:.1%}, {hi:.1%}]"
    def fmt(value: Any) -> str:
      return "n/a" if value is None else f"{float(value):.2f}"
    lines.append(f"| {name} | {row['n_trials']} | {row['n_evaluator_complete']} | {row['n_episode_exceptions']} | {row['n_success']} | {pct} ({ci}) | {row['total_steps_all_completed']} | {fmt(row['mean_steps_all_completed'])} | {fmt(row['mean_steps_successes'])} | {row['vllm_requests_endpoint_delta']} | {row['probe_observations']} | {row['rollbacks']} | {row['prompt_context_edges']} | {row['skip_hits']} / {row['skip_attempts']} | {row['graph_new_nodes']} / {row['graph_new_edges']} |")
  lines += ["", "## Graph construction overhead", "",
            "Graph update time is the sum of top-level executable-memory state/edge mutation calls; nested mutations are not counted twice. Probe-trace ingestion wall time includes parsing and surrounding ingestion work and overlaps graph update time, so the two columns must not be added. Neither measure includes graph JSON serialization or all agent-side CPU work.",
            "", "| Arm | Graph update calls | Graph mutation wall time (s) | Probe-trace ingestion wall time (s) |", "|---|---:|---:|---:|"]
  for name in arm_names:
    row = summaries[name]
    lines.append(f"| {name} | {row['graph_update_calls']} | {row['graph_construction_time_s']:.4f} | {row['probe_trace_ingest_wall_s']:.4f} |")
  lines += ["", "## Exploration budget and route retrieval", "",
            "Probe target is the per-round maximum, not a mandatory minimum. `Complete rounds` means the full configured target was finished; utilization is diagnostic rather than a success objective. `retrieval_path_candidates` counts graph paths before task-relevance filtering, while executable route candidates count paths surviving the task-relevance filter. The two counts separate missing graph coverage from overly strict matching. Probe observations count one resulting UI observation per probe-trace row (or explicit nested observations in legacy rows); legacy records fall back to the probe-event count.",
            "", "| Arm | Probe rounds | Full-budget rounds | Probes completed / max target | Full-budget rate | Budget utilization | Paths before relevance filter | Prompt route edges | Executable route candidates | Probe stop reasons |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
  for name in arm_names:
    row = summaries[name]
    round_rate = "n/a" if row["probe_round_completion_rate"] is None else f"{row['probe_round_completion_rate']:.1%}"
    budget_rate = "n/a" if row["probe_budget_utilization"] is None else f"{row['probe_budget_utilization']:.1%}"
    stop_reasons = ", ".join(
        f"{key}:{value}" for key, value in row["probe_stop_reasons"].items()) or "none"
    lines.append(
        f"| {name} | {row['probe_rounds']} | {row['probe_rounds_complete']} | "
        f"{row['probes_completed']} / {row['probes_target']} | {round_rate} | "
        f"{budget_rate} | {row['retrieval_path_candidates']} | "
        f"{row['prompt_context_edges']} | {row['high_confidence_route_gate_candidates']} | {stop_reasons} |")
  lines += ["", "## Graph-guidance candidate funnel", "",
            "These counters are differenced from cumulative per-arm memory snapshots. `State matches` counts queries mapped to a stored graph state; `observed controls` counts outgoing controls backed by observed transitions; `task-relevant controls` also includes frontier controls and is not a strict subset of observed controls; `mature controls` are observed transitions passing the repeated-support, confidence, stability, reversibility, recovery, and dynamic-content checks.",
            "", "| Arm | Guidance queries | State matches | Observed controls | Task-relevant controls | Mature controls |", "|---|---:|---:|---:|---:|---:|"]
  for name in arm_names:
    row = summaries[name]
    lines.append(
        f"| {name} | {row['exploration_guidance_queries']} | "
        f"{row['exploration_guidance_state_matches']} | "
        f"{row['exploration_guidance_controls_seen']} | "
        f"{row['exploration_guidance_controls_relevant']} | "
        f"{row['exploration_guidance_controls_mature']} |")
  lines += ["", "## Verified inference-skip diagnostics", "",
            "Executable-memory route gates and the independent two-system SkipInferenceGate are reported separately. A gate candidate is not a skip hit; only a live-validated route/action is counted as a hit. Bootstrap app-open skips are separated because the evaluator action still continues.",
            "", "| Arm | Route-gate queries | Retrieved candidates | Eligible routes | Gate blocks | Executable-memory skip hits / attempts | Two-system route hits / attempts | Bootstrap skips | Route rollback failures |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
  for name in arm_names:
    row = summaries[name]
    blocks = ", ".join(f"{key}:{value}" for key, value in row["high_confidence_route_gate_blocks"].items()) or "none"
    lines.append(
        f"| {name} | {row['high_confidence_route_gate_queries']} | "
        f"{row['high_confidence_route_gate_candidates']} | "
        f"{row['high_confidence_route_gate_eligible']} | {blocks} | "
        f"{row['skip_hits']} / {row['skip_attempts']} | "
        f"{row['two_system_skip_route_hits']} / {row['two_system_skip_route_attempts']} | "
        f"{row['two_system_bootstrap_skips']} | "
        f"{row['executable_memory_route_rollback_failures']} |")
  lines += ["", "## Paired contrasts against raw baseline", "",
            "Negative step deltas favor the graph arm. Success-rate pairs include every task with a checkpoint; episode-step pairs exclude evaluator exceptions/missing steps. Shared-success steps compare only tasks where both methods passed.",
            "", "| Arm | Paired trials | Step pairs | Success-rate delta (95% bootstrap CI) | All-step delta (95% bootstrap CI) | Shared-success step delta (n; 95% CI) |",
            "|---|---:|---:|---:|---:|---:|"]
  for name, row in summary["paired_vs_baseline"].items():
    def ci_text(value: Any) -> str:
      if value[0] is None:
        return "n/a"
      return f"[{value[0]:.3f}, {value[1]:.3f}]"
    d = row["paired_success_rate_delta"]
    success_delta = "n/a" if d is None else f"{d:+.3f}"
    all_step_delta = row["paired_all_steps_delta_variant_minus_baseline"]
    all_step_delta = "n/a" if all_step_delta is None else f"{all_step_delta:+.2f}"
    shared_delta = row["paired_common_success_steps_delta_variant_minus_baseline"]
    shared_delta = "n/a" if shared_delta is None else f"{shared_delta:+.2f}"
    lines.append(f"| {name} | {row['paired_n']} | {row['paired_step_n']} | {success_delta} {ci_text(row['paired_success_rate_delta_bootstrap_95'])} | {all_step_delta} {ci_text(row['paired_all_steps_delta_bootstrap_95'])} | {row['common_success_n']}; {shared_delta} {ci_text(row['paired_common_success_steps_delta_bootstrap_95'])} |")
  lines += [
      "",
      "## What this matrix identifies",
      "",
      "- Graph construction: `graph_build_only` records task-conditioned states/transitions without using them; node/edge growth and task steps are compared with `baseline`.",
      "- Exploration: the primary contrast is `graph_guided_exploration` versus `probe_only` (matched probe budget); `graph_build_only` versus `baseline` measures graph recording without graph steering.",
      "- Inference: `graph_prompt_inference` versus `graph_guided_exploration` isolates prompt retrieval; `graph_post_fusion` tests structured evidence fusion; `graph_verified_skip` tests skipping repeated VLM reasoning only through the configured confidence/verification gate.",
      "- Task specificity: each graph arm uses a separate persistent graph and the method's goal-semantic retrieval; no graph is shared across study arms.",
      "- Inference efficiency: primary VLM calls, prompt tokens, executable-memory skip actions, two-system skip hits, route-gate rejection reasons, and verified-route hit/miss counts are recorded separately from environment action steps.",
      "",
      "## Decision rule and limitations",
      "",
      "The user requirement is success rate no lower than baseline, while reducing total episode steps and steps among jointly successful tasks. A positive point estimate is only a screening signal; small cohorts and wide paired intervals do not establish non-inferiority. If an arm misses either objective, use per-task deltas, probe/rollback traces, and graph coverage to decide which single design factor to revise, then rerun the same frozen task/seed cohort. Do not change temperature, top_p, task seed, model, evaluator, or task list during a comparison.",
      "",
      "![Success rates](success_rate.svg)",
      "",
      "![Mean action steps](mean_steps.svg)",
      "",
      "![Mean steps on successful episodes](success_steps.svg)",
      "",
      "Baseline complete episodes: {}; baseline success rate: {}; baseline mean steps: {}.".format(
          baseline.get("n_trials", 0),
          "n/a" if baseline.get("success_rate") is None else "{:.1%}".format(baseline["success_rate"]),
          baseline.get("mean_steps_all_completed")),
      "",
      "Raw artifact directory is private to the LabServer and is not part of this report.",
      "",
  ]
  (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
  return summary


def publish_report(repo: Path, root: Path, protocol: dict[str, Any], run_id: str) -> None:
  """Publish an explicit allowlist after scanning for local/server paths."""
  destination = repo / PUBLISH_PREFIX / run_id
  destination.mkdir(parents=True, exist_ok=True)
  allow = ["report.md", "aggregate_summary.json", "per_task_metrics.csv",
           "success_rate.svg", "mean_steps.svg", "success_steps.svg"]
  for name in allow:
    source = root / name
    if source.is_file():
      data = source.read_bytes()
      text = data.decode("utf-8", errors="ignore")
      if re.search(r"/(?:Users|home|data|runs)/[^\s\"']+", text):
        raise RuntimeError(f"publication safety scan refused local/server path in {name}")
      if re.search(r"(?i)(?:api[_-]?key|password|token)\s*[:=]\s*\S+", text):
        raise RuntimeError(f"publication safety scan refused credential-like text in {name}")
      (destination / name).write_bytes(data)
  _atomic_json(destination / "protocol_public.json", {
      "study": protocol["study"], "suite": protocol["suite"],
      "tasks": protocol["tasks"], "controls": protocol["controls"],
      "arms": protocol["arms"], "primary_outcomes": protocol["primary_outcomes"],
      "privacy": "No prompts, goal values, screenshots, raw logs, or checkpoints are included.",
  })
  status = _run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], check=True).stdout.splitlines()
  allowed_rel = str((PUBLISH_PREFIX / run_id).as_posix()) + "/"
  unexpected = [line for line in status if not line[3:].startswith(allowed_rel)]
  if unexpected:
    raise RuntimeError("refusing report commit while unrelated changes exist: " + "; ".join(unexpected[:10]))
  allowed_names = [*allow, "protocol_public.json"]
  allowed_paths = [f"{allowed_rel}{name}" for name in allowed_names
                   if (destination / name).is_file()]
  if not allowed_paths:
    raise RuntimeError("no allowlisted aggregate outputs exist to publish")
  _run(["git", "-C", str(repo), "add", "--", *allowed_paths], timeout=30)
  staged = _run(["git", "-C", str(repo), "diff", "--cached", "--name-only"], timeout=30).stdout.splitlines()
  if any(path not in allowed_paths for path in staged):
    raise RuntimeError("staged publication paths are outside the report allowlist")
  if _run(["git", "-C", str(repo), "diff", "--cached", "--quiet"], check=False).returncode == 0:
    missing = []
    for path in allowed_paths:
      result = _run(["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", path],
                    check=False, timeout=30)
      if result.returncode != 0:
        missing.append(path)
    if missing:
      raise RuntimeError("allowlisted report files were not staged or tracked: "
                         + "; ".join(missing))
    latest_subject = _run([
        "git", "-C", str(repo), "log", "-1", "--format=%s"], timeout=30).stdout.strip()
    if latest_subject == f"Publish AndroidWorld graph study {run_id}":
      # A previous iteration may have committed locally but lost its SSH
      # connection during push. Retry that exact publication, not a new commit.
      _run(["git", "-C", str(repo), "push", "origin", "HEAD"], timeout=180)
    return
  _run(["git", "-C", str(repo), "commit", "-m", f"Publish AndroidWorld graph study {run_id}"], timeout=120)
  _run(["git", "-C", str(repo), "push", "origin", "HEAD"], timeout=180)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="command", required=True)
  run_parser = sub.add_parser("run", help="run or safely resume the study")
  run_parser.add_argument("--run-root", type=Path, required=True)
  run_parser.add_argument("--run-id", required=True)
  run_parser.add_argument("--repo", type=Path, default=REPO_ROOT)
  run_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
  run_parser.add_argument("--image", default=DEFAULT_IMAGE)
  run_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
  run_parser.add_argument("--vllm-python", default=DEFAULT_VLLM_PYTHON)
  run_parser.add_argument("--vllm-cwd", default="/home/rxhuang/Projects")
  run_parser.add_argument("--max-workers", type=int, default=3)
  run_parser.add_argument("--port-start", type=int, default=8093)
  run_parser.add_argument("--reuse-vllm-port", type=int)
  run_parser.add_argument(
      "--reuse-vllm-only", action="store_true",
      help="never start a vLLM process; wait until the selected existing endpoint is idle",
  )
  run_parser.add_argument("--idle-window-s", type=int, default=20)
  run_parser.add_argument("--resource-poll-s", type=int, default=300)
  run_parser.add_argument("--vllm-start-timeout-s", type=int, default=900)
  run_parser.add_argument("--episode-timeout-s", type=int, default=5400)
  run_parser.add_argument("--failure-backoff-s", type=int, default=300)
  run_parser.add_argument("--publish", action=argparse.BooleanOptionalAction, default=True)
  status_parser = sub.add_parser("status", help="read study status and completed episode count")
  status_parser.add_argument("--run-root", type=Path, required=True)
  analyze_parser = sub.add_parser("analyze", help="regenerate aggregate report from checkpoints/records")
  analyze_parser.add_argument("--run-root", type=Path, required=True)
  analyze_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
  publish_parser = sub.add_parser("publish", help="publish only sanitized aggregate outputs")
  publish_parser.add_argument("--run-root", type=Path, required=True)
  publish_parser.add_argument("--repo", type=Path, default=REPO_ROOT)
  publish_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
  publish_parser.add_argument("--run-id", required=True)
  return parser


def main() -> int:
  args = build_parser().parse_args()
  if args.command == "run":
    if args.max_workers < 1 or args.max_workers > 3:
      raise SystemExit("--max-workers must be between 1 and 3")
    if args.reuse_vllm_port == -1:
      args.reuse_vllm_port = None
    if args.reuse_vllm_only and args.reuse_vllm_port is None:
      raise SystemExit("--reuse-vllm-only requires --reuse-vllm-port")
    return run_study(args)
  if args.command == "status":
    root = args.run_root.resolve()
    state_path = root / "status.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"state": "not_started"}
    records = [row for row in _read_jsonl(root / "records.jsonl") if row.get("status") == "complete"]
    state["complete_episodes"] = len({(row.get("task"), row.get("arm")) for row in records})
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0
  protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
  _validate_protocol(protocol)
  summary = analyze(args.run_root.resolve(), protocol)
  if args.command == "publish":
    publish_report(args.repo.resolve(), args.run_root.resolve(), protocol, args.run_id)
  print(json.dumps({"state": "analyzed", "records": summary["n_complete_task_arm_pairs"],
                    "report": str(args.run_root.resolve() / "report.md")}, ensure_ascii=False))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
