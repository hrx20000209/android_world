#!/usr/bin/env python3
"""Run the fixed 20-task GELAB baseline with per-request latency telemetry."""

from __future__ import annotations

import argparse
import contextvars
import datetime as dt
import json
from pathlib import Path
import re
import runpy
import sys
import time
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from android_world.agents import base_agent
from android_world.agents import gelab_agent
from android_world.agents import infer


TASKS_20 = [
    "ClockStopWatchRunning",
    "ExpenseDeleteSingle",
    "MarkorCreateFolder",
    "MarkorDeleteNewestNote",
    "MarkorDeleteNote",
    "NotesIsTodo",
    "NotesTodoItemCount",
    "OpenAppTaskEval",
    "SimpleCalendarEventsOnDate",
    "TasksDueOnDate",
    "NotesRecipeIngredientCount",
    "SimpleCalendarEventsInTimeRange",
    "SimpleCalendarNextEvent",
    "SimpleCalendarNextMeetingWithPerson",
    "SportsTrackerActivitiesCountForWeek",
    "SportsTrackerActivitiesOnDate",
    "SportsTrackerActivityDuration",
    "SportsTrackerTotalDistanceForCategoryOverInterval",
    "TasksDueNextWeek",
    "TasksHighPriorityTasksDueOnDate",
]

METRIC_NAMES = {
    "ttft_s": "vllm:time_to_first_token_seconds_sum",
    "queue_s": "vllm:request_queue_time_seconds_sum",
    "inference_s": "vllm:request_inference_time_seconds_sum",
    "prefill_s": "vllm:request_prefill_time_seconds_sum",
    "decode_s": "vllm:request_decode_time_seconds_sum",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
}


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", type=Path, required=True)
    parser.add_argument("--metrics_url", default="http://127.0.0.1:8081/metrics")
    parser.add_argument("--console_port", type=int, default=5554)
    parser.add_argument("--max_n_steps", type=int, default=6)
    parser.add_argument("--task_random_seed", type=int, default=30)
    args = parser.parse_args()

    root = args.benchmark_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "request_latency.jsonl"
    step_path = root / "step_latency.jsonl"
    meta_path = root / "benchmark_meta.json"
    for path in (request_path, step_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing telemetry: {path}")

    http = requests.Session()
    http.trust_env = False

    def metric_snapshot() -> dict[str, float]:
        text = http.get(args.metrics_url, timeout=10).text
        values: dict[str, float] = {}
        for key, metric_name in METRIC_NAMES.items():
            match = re.search(
                r"^" + re.escape(metric_name) + r"\{[^\n]*\}\s+([0-9.eE+-]+)$",
                text,
                flags=re.MULTILINE,
            )
            values[key] = float(match.group(1)) if match else 0.0
        return values

    current_profile: contextvars.ContextVar[dict[str, Any] | None] = (
        contextvars.ContextVar("gelab_latency_profile", default=None)
    )
    request_number = 0

    original_predict_mm = infer.LlamaCppWrapper.predict_mm

    def timed_predict_mm(self, text_prompt, images, messages=None):
        nonlocal request_number
        request_number += 1
        before = metric_snapshot()
        started = time.perf_counter()
        result = original_predict_mm(self, text_prompt, images, messages)
        client_total = time.perf_counter() - started
        after = metric_snapshot()
        delta = {key: after[key] - before[key] for key in before}
        server_e2e = delta["ttft_s"] + delta["decode_s"]
        preprocess = max(
            0.0,
            delta["ttft_s"] - delta["queue_s"] - delta["prefill_s"],
        )
        transport = max(0.0, client_total - server_e2e)
        profile = current_profile.get()
        raw = result[2] if isinstance(result, tuple) and len(result) > 2 else None
        usage = raw.get("usage") if isinstance(raw, dict) else None
        row = {
            "record_type": "request_latency",
            "request_number": request_number,
            "task": profile.get("task") if profile else "",
            "goal": profile.get("goal") if profile else "",
            "step": profile.get("step") if profile else None,
            "client_total_s": client_total,
            "queue_s": delta["queue_s"],
            "preprocess_s": preprocess,
            "prefill_s": delta["prefill_s"],
            "decode_s": delta["decode_s"],
            "inference_s": delta["inference_s"],
            "ttft_s": delta["ttft_s"],
            "server_e2e_s": server_e2e,
            "transport_s": transport,
            "prompt_tokens": int(round(delta["prompt_tokens"])),
            "generation_tokens": int(round(delta["generation_tokens"])),
            "response_usage": usage,
            "timestamp": time.time(),
        }
        _append_jsonl(request_path, row)
        if profile is not None:
            profile["request_count"] += 1
            profile["llm_client_s"] += client_total
        print("PER_REQUEST_LATENCY " + json.dumps(row, ensure_ascii=False), flush=True)
        return result

    infer.LlamaCppWrapper.predict_mm = timed_predict_mm

    original_post_state = base_agent.EnvironmentInteractingAgent.get_post_transition_state

    def timed_post_state(self, *call_args, **call_kwargs):
        started = time.perf_counter()
        try:
            return original_post_state(self, *call_args, **call_kwargs)
        finally:
            profile = current_profile.get()
            if profile is not None:
                profile["state_capture_s"] += time.perf_counter() - started

    base_agent.EnvironmentInteractingAgent.get_post_transition_state = timed_post_state

    original_execute = gelab_agent.GELABAgent._execute_action

    def timed_execute(self, action, extras):
        started = time.perf_counter()
        try:
            return original_execute(self, action, extras)
        finally:
            profile = current_profile.get()
            if profile is not None:
                profile["action_execute_s"] += time.perf_counter() - started

    gelab_agent.GELABAgent._execute_action = timed_execute

    original_step = gelab_agent.GELABAgent.step

    def timed_step(self, goal: str):
        profile: dict[str, Any] = {
            "record_type": "step_latency",
            "task": type(getattr(self, "_task", None)).__name__ if getattr(self, "_task", None) else "",
            "goal": goal,
            "step": len(self._actions),
            "state_capture_s": 0.0,
            "llm_client_s": 0.0,
            "action_execute_s": 0.0,
            "request_count": 0,
        }
        token = current_profile.set(profile)
        started = time.perf_counter()
        try:
            result = original_step(self, goal)
        finally:
            step_total = time.perf_counter() - started
            current_profile.reset(token)
        profile["step_total_s"] = step_total
        profile["local_overhead_s"] = max(
            0.0,
            step_total
            - profile["state_capture_s"]
            - profile["llm_client_s"]
            - profile["action_execute_s"],
        )
        profile["done"] = bool(result.done)
        profile["action"] = result.data.get("action_dict")
        profile["parse_error"] = (result.data.get("parsed_action") or {}).get("parse_error")
        profile["timestamp"] = time.time()
        _append_jsonl(step_path, profile)
        print("PER_STEP_LATENCY " + json.dumps(profile, ensure_ascii=False), flush=True)
        return result

    gelab_agent.GELABAgent.step = timed_step

    run_output = root / "androidworld_run"
    meta = {
        "started_at": dt.datetime.now().isoformat(),
        "tasks": TASKS_20,
        "task_count": len(TASKS_20),
        "max_n_steps": args.max_n_steps,
        "task_random_seed": args.task_random_seed,
        "agent": "gelab_agent",
        "summary_request_enabled": False,
        "exploration_enabled": False,
        "metrics_url": args.metrics_url,
        "request_latency_file": str(request_path),
        "step_latency_file": str(step_path),
        "run_output": str(run_output),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    sys.argv = [
        "run.py",
        "--suite_family=android_world",
        "--agent_name=gelab_agent",
        "--tasks=" + ",".join(TASKS_20),
        "--n_task_combinations=1",
        "--fixed_task_seed",
        f"--task_random_seed={args.task_random_seed}",
        f"--max_n_steps={args.max_n_steps}",
        f"--console_port={args.console_port}",
        f"--output_path={run_output}",
    ]
    try:
        runpy.run_path("run.py", run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
