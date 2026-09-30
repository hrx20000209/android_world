"""One AndroidWorld task, one exploration method, for the posterior-coverage study.

The agent is the unmodified GELAB agent (same prompt, parser, executor and
official step budget as the pure-VLM baseline). This script only wraps two
points of its step:

  predict_mm      the full reasoning call. Exploration methods start it,
                  capture the screen, score the cheap posterior, and explore
                  while it is in flight. When it returns, the final action is
                  handed to the probe window, which commits or recovers.
  _execute_action skipped when the probe window already executed exactly this
                  action and is parked at its successor (speculative_commit).

Everything is logged per step to <output>/steps.jsonl.

Usage:
  python scripts/run_posterior_coverage_task.py --task MarkorCreateNote \\
      --method posterior_coverage --output DIR --stats STATS.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import runpy
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", required=True)
  parser.add_argument("--method", required=True,
                      choices=("none", "current", "random", "probability",
                               "posterior_coverage"))
  parser.add_argument("--output", required=True)
  parser.add_argument("--stats", required=True,
                      help="per-method session latency stats (JSON, read+written)")
  parser.add_argument("--priors", default="",
                      help="JSON with reasoning_s and probe_cost_s priors")
  parser.add_argument("--seed", type=int, default=30)
  parser.add_argument("--api_url", default="http://localhost:8084/v1/chat/completions")
  parser.add_argument("--model", default="GELAB-ZERO-4B")
  parser.add_argument("--semantic_port", type=int, default=8766)
  parser.add_argument("--max_probes", type=int, default=10,
                      help="safety cap per window; the budget normally binds first")
  parser.add_argument("--score_workers", type=int, default=8)
  args = parser.parse_args()

  root = Path(args.output)
  root.mkdir(parents=True, exist_ok=True)
  os.environ["ANDROID_WORLD_LLM_API_URL"] = args.api_url
  os.environ.setdefault("ANDROID_WORLD_LLAMACPP_MAX_TOKENS", "2048")
  os.environ.setdefault("ANDROID_WORLD_A11Y_METHOD", "fast_provider")
  os.environ.setdefault("ANDROID_WORLD_FAST_A11Y_SOCKET_PORT", "8765")

  import psutil
  import requests
  from android_world.agents import gelab_agent
  from android_world.agents import infer
  from android_world.parallel_exploration import posterior_coverage as pc
  from android_world.parallel_exploration.information import parse_reasoning_prior
  from android_world.parallel_exploration.rankers import GraphKeywordRanker
  from android_world.parallel_exploration.rankers import SimpleRelevanceRanker
  from android_world.parallel_exploration.state import AdbClient
  from android_world.parallel_exploration.state import create_optimized_state_capture

  priors = {"reasoning_s": 4.0, "probe_cost_s": 1.5}
  if args.priors and Path(args.priors).exists():
    priors.update(json.loads(Path(args.priors).read_text()))
  stats_path = Path(args.stats)
  stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
  stats.setdefault("reasoning_s", [])
  stats.setdefault("probe_cost_s", {})
  cost_model = pc.CostModel(stats["probe_cost_s"], float(priors["probe_cost_s"]))

  steps_path = root / "steps.jsonl"
  exploring = args.method != "none"
  adb = AdbClient("emulator-5554") if exploring else None
  capture = create_optimized_state_capture(
      serial="emulator-5554", console_port=5554,
      adb_path=os.environ.get("ANDROID_WORLD_ADB_PATH",
                              "/Users/huangrunxi/Library/Android/sdk/platform-tools/adb"),
      a11y_local_port=8765) if exploring else None
  verified: dict[str, set[str]] = {}   # this episode's graph: state -> verified keys
  ctx: dict[str, Any] = {"agent": None, "goal": "", "step": 0, "last_output": "",
                         "row": None, "reasoning_calls": 0}
  psutil.cpu_percent(None)
  qemu = [p for p in psutil.process_iter(["name", "cmdline"])
          if "qemu-system" in (p.info["name"] or "")
          and "AndroidWorldAvd" in " ".join(p.info["cmdline"] or [])]

  def resources() -> dict[str, Any]:
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    out = {"host_cpu_pct": psutil.cpu_percent(None), "host_mem_pct": vm.percent,
           "swap_used_gb": round(sw.used / 2**30, 2),
           "load_1m": round(os.getloadavg()[0], 2)}
    if qemu:
      try:
        out["emulator_rss_gb"] = round(qemu[0].memory_info().rss / 2**30, 2)
        out["emulator_cpu_pct"] = qemu[0].cpu_percent(None)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    return out

  def stream_reasoning(wrapper, messages) -> tuple[str, dict[str, Any]]:
    """The full reasoning call, streamed only to time it; same sampling params."""
    payload = {"messages": messages, "temperature": wrapper.temperature,
               "top_p": 1.0, "max_tokens": wrapper.max_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    if wrapper.model_name:
      payload["model"] = wrapper.model_name
    started = time.monotonic()
    first_token = None
    text, usage = [], {}
    for attempt in range(3):
      try:
        with requests.post(wrapper.api_url, json=payload, stream=True, timeout=600) as resp:
          resp.raise_for_status()
          for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
              continue
            data = line[6:]
            if data == b"[DONE]":
              break
            chunk = json.loads(data)
            if chunk.get("usage"):
              usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
              delta = (choice.get("delta") or {}).get("content")
              if delta:
                if first_token is None:
                  first_token = time.monotonic()
                text.append(delta)
        break
      except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[posterior_coverage] reasoning stream error {attempt}: {exc}", flush=True)
        text, usage, first_token = [], {}, None
        time.sleep(2 * (attempt + 1))
    ended = time.monotonic()
    ttft = (first_token - started) if first_token else None
    tokens = int(usage.get("completion_tokens") or 0)
    decode = (ended - first_token) if first_token else None
    return "".join(text), {
        "reasoning_s": ended - started, "ttft_s": ttft, "decode_s": decode,
        "prompt_tokens": usage.get("prompt_tokens"), "output_tokens": tokens,
        "tokens_per_s": (tokens / decode) if decode and tokens else None,
        "t_start": started, "t_end": ended}

  original_predict = infer.LlamaCppWrapper.predict_mm
  original_step = gelab_agent.GELABAgent.step
  original_execute = gelab_agent.GELABAgent._execute_action

  def predict_mm(self, text_prompt, images, messages=None):
    if messages is None:
      return original_predict(self, text_prompt, images, messages)
    agent, step = ctx["agent"], ctx["step"]
    row: dict[str, Any] = {"task_id": args.task, "method": args.method, "step": step,
                           "seed": args.seed, "resources_before": resources()}
    ctx["row"] = row
    ctx["reasoning_calls"] += 1
    holder: dict[str, Any] = {}
    reasoning = threading.Thread(
        target=lambda: holder.update(zip(("text", "timing"),
                                         stream_reasoning(self, messages))),
        daemon=True)
    reasoning_started = time.monotonic()
    reasoning.start()
    window = None
    baseline = None
    candidates: list = []
    if exploring:
      t0 = time.monotonic()
      baseline = capture.capture()
      row["capture_s"] = round(time.monotonic() - t0, 3)
      screen = tuple(agent.env.logical_screen_size)
      package = baseline.activity.component.split("/", 1)[0]
      sid = pc.state_id(baseline)
      row.update({"app": package, "current_state_id": sid, "screen_size": screen})
      candidates = pc.tap_candidates(
          baseline.elements, root / "filtered_elements.jsonl",
          {"task": ctx["goal"], "trial_id": f"{args.task}-{step}"}, screen)
      posterior = pc.score_candidates(args.api_url, args.model, messages, candidates,
                                      max_workers=args.score_workers)
      probs = posterior.as_dict()
      known = verified.setdefault(sid, set())
      need = parse_reasoning_prior(ctx["last_output"], ctx["goal"]).to_dict()
      need_text = " ".join(str(x) for x in (
          need.get("target_entity") or "",
          " ".join(need.get("required_information_slots") or ()),
          need.get("current_subgoal") or "") if x).strip()
      current_order: list[str] = []
      if args.method == "current" and candidates:
        ranker = GraphKeywordRanker(base=SimpleRelevanceRanker(), need_text=need_text,
                                    port=args.semantic_port)
        current_order = [r.element.identity for r in
                         ranker.rank([c.element for c in candidates], ctx["goal"], ())]
      # B counts from when the reasoning request was sent, before the capture
      # and the posterior, so their time is already spent from the window.
      clock = pc.ReasoningClock(stats["reasoning_s"], float(priors["reasoning_s"]),
                                reasoning_started)
      rng = random.Random(f"{args.seed}|{args.task}|{step}")
      window = pc.ProbeWindow(
          method=args.method, adb=adb, capture=capture, baseline=baseline,
          candidates=candidates, probs=probs, verified=set(known),
          cost_model=cost_model, clock=clock, package=package,
          current_order=current_order, rng=rng, max_probes=args.max_probes)
      row.update({
          "candidates": [{"key": c.key, "label": c.label, "action": c.action_string,
                          "probe_type": c.probe_type, "verified": c.key in known,
                          "p": probs.get(c.key), "logprob":
                          (posterior.logprobs[posterior.keys.index(c.key)]
                           if c.key in posterior.keys else None),
                          "predicted_cost_s": round(cost_model.estimate(c, package)[0], 3),
                          "cost_source": cost_model.estimate(c, package)[1]}
                         for c in candidates],
          "n_candidates": len(candidates),
          "posterior_latency_s": round(posterior.latency_s, 3),
          "posterior_first_request_s": round(posterior.first_request_s, 3),
          "posterior_errors": posterior.errors,
          "posterior_sum": round(sum(posterior.probs), 6),
          "current_need": need_text,
          "current_order": current_order[:20],
          "predicted_reasoning_s": round(clock.predicted_s, 3),
          "reasoning_prediction_source": clock.source,
          "budget_at_window_start_s": round(clock.remaining(), 3),
      })
      window.start()
    reasoning.join()
    text, timing = holder.get("text", ""), holder.get("timing", {})
    row["reasoning"] = {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in timing.items() if k not in ("t_start", "t_end")}
    if timing.get("reasoning_s"):
      stats["reasoning_s"].append(timing["reasoning_s"])
    final_action: dict[str, Any] = {}
    try:
      parsed = gelab_agent.parse_gelab_response(text)
      action, _, _ = gelab_agent.gelab_action_to_json_action(
          parsed, tuple(agent.env.logical_screen_size))
      final_action = {k: v for k, v in action.__dict__.items() if v is not None}
    except Exception as exc:  # pylint: disable=broad-exception-caught
      final_action = {"parse_error": str(exc)[:200]}
    row["final_action"] = final_action
    if window is not None:
      final_key = pc.final_action_key(final_action, baseline.elements, candidates)
      row["final_key"] = final_key
      row["final_was_verified_before"] = final_key in verified.get(row["current_state_id"], set())
      window.finish(final_key)
      window.join(timeout=90)
      res = window.result
      row["window"] = {
          "probes": [dict(vars(p), actual_cost_s=round(p.actual_cost_s, 3)) for p in res.probes],
          "plans": window.plans, "committed": res.committed, "parked_key": res.parked_key,
          "dirty": res.dirty, "exposed_s": round(res.exposed_s, 3),
          "final_rollback_s": round(res.final_rollback_s, 3),
          "stop_reason": res.stop_reason, "error": res.error,
          "thread_alive": window.is_alive(),
      }
      probed = {p.key for p in res.probes}
      row["final_was_probed"] = final_key in probed
      row["speculative_commit"] = res.committed
      row["rollback_after_reasoning"] = bool(res.probes) and not res.committed
      for key, succ, _ in window.new_edges:
        verified.setdefault(row["current_state_id"], set()).add(key)
      stats_path.write_text(json.dumps(stats))
    else:
      stats_path.write_text(json.dumps(stats))
    ctx["commit"] = bool(window is not None and window.result.committed)
    ctx["last_output"] = text
    return text, True, {"posterior_coverage": True}

  def step(self, goal):
    ctx["agent"], ctx["goal"] = self, goal
    ctx["commit"], ctx["row"] = False, None
    ctx["exec_s"] = None
    step_started = time.monotonic()
    result = original_step(self, goal)
    row = ctx["row"] or {"task_id": args.task, "method": args.method,
                         "step": ctx["step"], "note": "no_reasoning_call"}
    row["action_execution_s"] = ctx["exec_s"]
    row["step_total_s"] = round(time.monotonic() - step_started, 3)
    row["reasoning_calls_so_far"] = ctx["reasoning_calls"]
    row["resources_after"] = resources()
    row["done"] = bool(result.done)
    with steps_path.open("a", encoding="utf-8") as out:
      out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    ctx["step"] += 1
    return result

  def execute_action(self, action, extras):
    if ctx.get("commit"):
      ctx["exec_s"] = 0.0
      if ctx["row"] is not None:
        ctx["row"]["execution_skipped_by_commit"] = True
      return None
    started = time.monotonic()
    out = original_execute(self, action, extras)
    ctx["exec_s"] = round(time.monotonic() - started, 3)
    return out

  infer.LlamaCppWrapper.predict_mm = predict_mm
  gelab_agent.GELABAgent.step = step
  gelab_agent.GELABAgent._execute_action = execute_action

  (root / "run_args.json").write_text(json.dumps({"argv": sys.argv, "priors": priors}, indent=1))
  sys.argv = [
      "run.py", "--suite_family=android_world", "--agent_name=gelab_agent",
      f"--tasks={args.task}", "--n_task_combinations=1", "--fixed_task_seed",
      f"--task_random_seed={args.seed}", "--console_port=5554",
      f"--output_path={root}",
  ]
  try:
    runpy.run_path(str(REPO_ROOT / "run.py"), run_name="__main__")
  finally:
    if capture is not None:
      try:
        capture.close()
      except Exception:  # pylint: disable=broad-exception-caught
        pass


if __name__ == "__main__":
  main()
