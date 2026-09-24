#!/usr/bin/env python3
"""One-shot dual-emulator oracle probe for an AndroidWorld information task.

The worker initializes the same task seed on a shadow emulator, waits for a
MobileExplorer opportunity, replays a value-free navigation prefix from a
successful trajectory, and asks the VLM to extract the current answer.  It
writes evidence only when the initial primary/shadow screenshot binding and
the extraction both validate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world import registry  # pylint: disable=wrong-import-position,import-error
from android_world import suite_utils  # pylint: disable=wrong-import-position,import-error
from android_world.agents import gelab_agent  # pylint: disable=wrong-import-position,import-error
from android_world.agents import infer  # pylint: disable=wrong-import-position,import-error
from android_world.agents import mobileexplorer  # pylint: disable=wrong-import-position,import-error
from android_world.env import env_launcher  # pylint: disable=wrong-import-position,import-error


class ProbeDeadlineMiss(TimeoutError):
  """The speculative result did not fit in the advertised inference slack."""


def _check_deadline(deadline_s: float, stage: str) -> None:
  if time.monotonic() > deadline_s:
    raise ProbeDeadlineMiss(f"Probe exceeded inference deadline after {stage}")


def _adb_path() -> str:
  candidates = (
      Path.home() / "Library/Android/sdk/platform-tools/adb",
      Path.home() / "Android/Sdk/platform-tools/adb",
  )
  for candidate in candidates:
    if candidate.is_file():
      return str(candidate)
  raise FileNotFoundError("adb was not found in the supported SDK locations")


def _wait_for_opportunity(path: Path, timeout_s: float) -> dict[str, Any]:
  deadline = time.monotonic() + timeout_s
  while time.monotonic() < deadline:
    if path.is_file():
      lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
      if lines:
        return json.loads(lines[-1])
    time.sleep(.05)
  raise TimeoutError(f"No exploration opportunity appeared in {path}")


def _load_trajectory(
    path: Path, *, task_template: str, goal: str
) -> dict[str, Any]:
  rows = []
  with path.open(encoding="utf-8") as stream:
    for line in stream:
      if line.strip():
        row = json.loads(line)
        if row.get("successful"):
          rows.append(row)
  exact = [row for row in rows if row.get("task_template") == task_template]
  if exact:
    return exact[0]
  raise ValueError(
      f"No exact-template successful trajectory for {task_template} in {path}"
  )


def _execute_from_bound_state(
    env: Any,
    trajectory: dict[str, Any],
    opportunity: dict[str, Any],
) -> None:
  """Validate the primary/shadow binding, then probe via safe source actions."""
  deadline_s = float(opportunity["deadline_s"])
  source = mobileexplorer.SuccessfulTrajectory.from_dict(trajectory)
  reusable = mobileexplorer._replayable_step_indices(source)  # pylint: disable=protected-access
  if not reusable:
    raise ValueError("Source trajectory has no safely replayable prefix")
  if not opportunity.get("source_activity") or not opportunity.get("source_landmarks"):
    raise ValueError("Primary opportunity lacks state-binding descriptors")
  _check_deadline(deadline_s, "task initialization")
  current = env.get_state(wait_to_stabilize=True)
  _check_deadline(deadline_s, "initial state capture")
  bound_position = None
  for position in range(len(reusable) + 1):
    schema_id = mobileexplorer._state_schema_id(  # pylint: disable=protected-access
        current, env.foreground_activity_name
    )
    if (schema_id == opportunity["state_schema_id"]
        and env.foreground_activity_name == opportunity["source_activity"]
        and mobileexplorer._state_distance(  # pylint: disable=protected-access
            mobileexplorer._state_id(Image.fromarray(current.pixels)),  # pylint: disable=protected-access
            opportunity["state_id"],
        ) <= 6):
      bound_position = position
      break
    if position == len(reusable):
      break
    step = source.steps[reusable[position]]
    action = mobileexplorer._validated_replay_action(  # pylint: disable=protected-access
        step, current, env.foreground_activity_name
    )
    if action is None:
      raise ValueError(f"Shadow prefix action {position} failed semantic validation")
    env.execute_action(action)
    _check_deadline(deadline_s, f"prefix action {position}")
    current = env.get_state(wait_to_stabilize=True)
    _check_deadline(deadline_s, f"prefix stabilization {position}")
  if bound_position is None:
    raise ValueError("Shadow could not replay to primary opportunity state")
  probes = 0
  for position in range(bound_position, len(reusable)):
    step = source.steps[reusable[position]]
    action = mobileexplorer._validated_replay_action(  # pylint: disable=protected-access
        step, current, env.foreground_activity_name
    )
    if action is None:
      break
    env.execute_action(action)
    probes += 1
    _check_deadline(deadline_s, "probe action")
    current = env.get_state(wait_to_stabilize=True)
    _check_deadline(deadline_s, "probe stabilization")
  if probes == 0:
    raise ValueError("No safe shadow probe action beyond the bound state")


def _extract_answer(vllm: Any, goal: str, image: Image.Image, screen_size: tuple[int, int]) -> str:
  messages = gelab_agent.build_gelab_messages(
      goal,
      "Shadow observation only. The navigation prefix is complete; answer now if visible.",
      image,
  )
  response, _, _ = vllm.predict_mm("", [], messages=messages)
  parsed = gelab_agent.parse_gelab_response(response)
  _, _, extras = gelab_agent.gelab_action_to_json_action(parsed, screen_size)
  answer = str(extras.get("return_text") or "").strip()
  if not answer:
    raise ValueError(f"Shadow VLM did not return an answer: {response}")
  return answer


def _answer_is_visible(answer: str, state: Any) -> bool:
  """Admit only answers grounded in the probed accessibility observation."""
  normalized_answer = " ".join(answer.casefold().split())
  if len(normalized_answer) < 2:
    return False
  visible = [
      str(value) for element in state.ui_elements
      if element.is_visible is not False
      for value in (element.text, element.content_description)
      if value
  ]
  return any(normalized_answer in " ".join(text.casefold().split()) for text in visible)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", required=True)
  parser.add_argument("--task-random-seed", type=int, required=True)
  parser.add_argument("--trajectory-memory", type=Path, required=True)
  parser.add_argument("--opportunity-jsonl", type=Path, required=True)
  parser.add_argument("--evidence-jsonl", type=Path, required=True)
  parser.add_argument("--console-port", type=int, default=5556)
  parser.add_argument("--grpc-port", type=int, default=8556)
  parser.add_argument("--llm-api-url", required=True)
  parser.add_argument("--model", default="GELAB-ZERO-4B")
  parser.add_argument("--timeout-s", type=float, default=120)
  args = parser.parse_args()

  env = env_launcher.load_and_setup_env(
      console_port=args.console_port,
      grpc_port=args.grpc_port,
      adb_path=_adb_path(),
  )
  task_registry = registry.TaskRegistry()
  suite = suite_utils.create_suite(
      task_registry.get_registry(family="android_world"),
      n_task_combinations=1,
      seed=args.task_random_seed,
      tasks=[args.task],
      use_identical_params=True,
  )
  task = suite[args.task][0]
  env.reset(go_home=True)
  env.hide_automation_ui()
  task.initialize_task(env)
  try:
    opportunity = _wait_for_opportunity(args.opportunity_jsonl, args.timeout_s)
    trajectory = _load_trajectory(
        args.trajectory_memory,
        task_template=args.task,
        goal=opportunity["goal"],
    )
    probe_started_s = time.monotonic()
    _execute_from_bound_state(env, trajectory, opportunity)
    navigation_finished_s = time.monotonic()
    _check_deadline(float(opportunity["deadline_s"]), "navigation")
    final_state = env.get_state(wait_to_stabilize=True)
    final_image = Image.fromarray(final_state.pixels)
    _check_deadline(float(opportunity["deadline_s"]), "final state capture")
    vllm = infer.LlamaCppWrapper(
        api_url=args.llm_api_url,
        temperature=0.0,
        model_name=args.model,
    )
    answer = _extract_answer(vllm, opportunity["goal"], final_image, env.logical_screen_size)
    if not _answer_is_visible(answer, final_state):
      raise ValueError("Shadow answer is not grounded in visible UI text")
    extraction_finished_s = time.monotonic()
    _check_deadline(float(opportunity["deadline_s"]), "evidence extraction")
    row = {
        "evidence_id": f"shadow-{time.time_ns()}",
        "episode_id": opportunity["episode_id"],
        # Bind evidence to the primary decision state that authorized this
        # counterfactual probe, not the shadow screen reached by the probe.
        "state_id": opportunity["state_id"],
        "state_schema_id": opportunity["state_schema_id"],
        "shadow_observation_state_id": mobileexplorer._state_id(final_image),  # pylint: disable=protected-access
        "shadow_observation_schema_id": mobileexplorer._state_schema_id(  # pylint: disable=protected-access
            env.get_state(wait_to_stabilize=False), env.foreground_activity_name
        ),
        "task_need": opportunity["task_need"],
        "slot": "answer",
        "value": answer,
        "observed_generation": int(opportunity["generation"]),
        "observed_at_s": time.time(),
        "ttl_s": 60.0,
        "confidence": 1.0,
        "isolation_id": f"emulator-{args.console_port}",
        "state_match": True,
        "extraction_verified": True,
        "probe_latency_s": extraction_finished_s - probe_started_s,
        "navigation_latency_s": navigation_finished_s - probe_started_s,
        "extraction_latency_s": extraction_finished_s - navigation_finished_s,
        "opportunity_to_evidence_s": time.time() - float(opportunity["created_at_s"]),
        "deadline_s": float(opportunity["deadline_s"]),
        "deadline_miss": False,
    }
    args.evidence_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.evidence_jsonl.open("a", encoding="utf-8") as stream:
      stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False, indent=2))
  finally:
    task.tear_down(env)
    env.close()


if __name__ == "__main__":
  try:
    main()
  except ProbeDeadlineMiss as error:
    print(json.dumps({"status": "deadline_missed", "error": str(error)}))
    sys.exit(75)
