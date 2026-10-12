#!/usr/bin/env python3
"""Per-episode and paired read-outs for the task-navigation pilot.

  python scripts/analyze_task_navigation_pilot.py results/sensys27_task_navigation/<ts>

Reads <root>/<arm>/<task>.log (official evaluator table), serial_events.jsonl
and api_calls.jsonl. Writes episodes.csv, summary.json and paired.json into
<root>. Model calls come from the HTTP log, not from agent.step counts; macro,
probe and model actions are counted apart.
"""

from __future__ import annotations

import collections
import csv
import json
import random
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
  if not path.exists():
    return []
  rows = []
  for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
    try:
      rows.append(json.loads(line))
    except ValueError:
      pass
  return rows


def official(log: Path) -> tuple[float | None, float | None, float | None]:
  """(success, episode_length, runtime_s) from run.py's result table."""
  if not log.exists():
    return None, None, None
  text = log.read_text(encoding="utf-8", errors="ignore")
  m = re.search(r"\n(\w+)\s+\d+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(?:[\d.]+|NaN)\s+[\d.]+\s*\n"
                r"=+ Average", text)
  if not m:
    return None, None, None
  return float(m.group(2)), float(m.group(3)), float(m.group(4))


def baseline_4b2() -> dict[str, tuple[float, float]]:
  out = {}
  path = REPO / "results" / "4B_2.txt"
  for line in path.read_text().splitlines()[2:]:
    parts = line.split()
    if len(parts) >= 5 and parts[0][0].isupper():
      try:
        out[parts[0]] = (float(parts[3]), float(parts[4]))
      except ValueError:
        pass
  return out


def episode(root: Path, arm: str, task: str) -> dict:
  run = root / arm / task
  success, length, runtime = official(root / arm / f"{task}.log")
  events = read_jsonl(run / "serial_events.jsonl")
  calls = read_jsonl(run / "api_calls.jsonl")
  kinds = collections.Counter(e.get("kind") for e in events)
  plans = [e for e in events if e.get("kind") == "task_navigation_plan"]
  execs = [e for e in events if e.get("kind") == "task_navigation_execute"]
  checks = [e for e in events if e.get("kind") == "state_check"]
  explores = [e for e in events if e.get("kind") == "explore"]
  hist = [e for e in events if e.get("kind") == "history_witness"]
  rejections = collections.Counter()
  rejections_ui = collections.Counter()
  for p in plans:
    rejections.update(p.get("rejections") or {})
    rejections_ui.update(p.get("rejections_ui_only") or {})
  intents = [p for p in plans if p.get("intent")]
  first_route = next((p["step"] for p in plans if p.get("route")), None)
  first_route_ui = next((p["step"] for p in plans if p.get("route_ui_only")), None)
  return {
      "arm": arm, "task": task, "success": success, "episode_length": length,
      "runtime_s": runtime,
      "model_calls": len(calls),
      "failed_calls": sum(1 for c in calls if not c.get("ok")),
      "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
      "completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls),
      "outputs_with_nav_intent": sum(1 for c in calls if c.get("has_nav_intent")),
      "parsed_intents": len(intents),
      "plans": len(plans),
      "already_at_target": sum(1 for p in plans if p.get("already_at_target")),
      "routes_strict": sum(1 for p in plans if p.get("route")),
      "routes_ui_only": sum(1 for p in plans if p.get("route_ui_only")),
      "first_route_step": first_route, "first_route_ui_step": first_route_ui,
      "max_witnesses": max((p.get("witnesses") or 0 for p in plans), default=0),
      "nav_executions": len(execs),
      "nav_hops_done": sum(e.get("completed_hops", 0) for e in execs),
      "nav_actions": sum(e.get("attempted_hops", 0) for e in execs),
      "nav_terminal_reached": sum(1 for e in execs if e.get("reached_terminal")),
      "nav_fail_reasons": dict(collections.Counter(e.get("reason") for e in execs
                                                   if not e.get("reached_terminal"))),
      "explore_rounds": len(explores),
      "probes": sum(int(e.get("probes_completed", 0) or 0) for e in explores),
      "restore_failures": sum(1 for e in explores if e.get("restore_status") not in (None, "RESTORED")),
      "state_checks": len(checks),
      "state_verified": sum(1 for e in checks if e.get("verified")),
      "state_reasons": dict(collections.Counter(e.get("reason") for e in checks)),
      "checker_cost_s": sum(e.get("checker_cost_s", 0) for e in checks),
      "history_witnesses": len(hist),
      "history_verified": sum(1 for e in hist if e.get("state_effect_verified")),
      "bootstrap_skips": sum(1 for e in events if e.get("kind") == "skip"
                             and e.get("kind_detail") == "app_bootstrap"),
      "rejections": dict(rejections), "rejections_ui_only": dict(rejections_ui),
      "event_kinds": dict(kinds),
  }


def paired(rows: list[dict], a: str, b: str, key: str, n_boot: int = 2000) -> dict:
  by = {(r["arm"], r["task"]): r for r in rows}
  tasks = sorted(t for (arm, t) in by if arm == a and (b, t) in by
                 and by[(a, t)][key] is not None and by[(b, t)][key] is not None)
  diffs = [by[(b, t)][key] - by[(a, t)][key] for t in tasks]
  if not diffs:
    return {"n": 0}
  rng = random.Random(0)
  boots = sorted(statistics.mean(rng.choice(diffs) for _ in diffs) for _ in range(n_boot))
  return {"n": len(diffs), "mean_diff": statistics.mean(diffs),
          "ci95": [boots[int(.025 * n_boot)], boots[int(.975 * n_boot)]],
          "per_task": dict(zip(tasks, diffs))}


def main() -> int:
  root = Path(sys.argv[1])
  rows = []
  for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    for log in sorted(arm_dir.glob("*.log")):
      rows.append(episode(root, arm_dir.name, log.stem))
  ref = baseline_4b2()
  for r in rows:
    r["4B_2_success"], r["4B_2_length"] = ref.get(r["task"], (None, None))
  flat = [k for k in rows[0] if not isinstance(rows[0][k], dict)] if rows else []
  with (root / "episodes.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=flat, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
  summary = {}
  for arm in sorted({r["arm"] for r in rows}):
    rs = [r for r in rows if r["arm"] == arm and r["success"] is not None]
    if not rs:
      continue
    ok = [r for r in rs if r["success"] > 0]
    agg = collections.Counter()
    agg_ui = collections.Counter()
    for r in rs:
      agg.update(r["rejections"])
      agg_ui.update(r["rejections_ui_only"])
    summary[arm] = {
        "episodes": len(rs), "successes": len(ok),
        "4B_2_successes_same_tasks": sum(1 for r in rs if (r["4B_2_success"] or 0) > 0),
        "mean_steps_all": statistics.mean(r["episode_length"] for r in rs),
        "mean_steps_success": statistics.mean(r["episode_length"] for r in ok) if ok else None,
        "model_calls": sum(r["model_calls"] for r in rs),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rs),
        "completion_tokens": sum(r["completion_tokens"] for r in rs),
        "runtime_s_p50": statistics.median(r["runtime_s"] for r in rs),
        "outputs_with_nav_intent": sum(r["outputs_with_nav_intent"] for r in rs),
        "parsed_intents": sum(r["parsed_intents"] for r in rs),
        "plans": sum(r["plans"] for r in rs),
        "routes_strict": sum(r["routes_strict"] for r in rs),
        "routes_ui_only": sum(r["routes_ui_only"] for r in rs),
        "nav_executions": sum(r["nav_executions"] for r in rs),
        "nav_hops_done": sum(r["nav_hops_done"] for r in rs),
        "probes": sum(r["probes"] for r in rs),
        "state_checks": sum(r["state_checks"] for r in rs),
        "state_verified": sum(r["state_verified"] for r in rs),
        "history_witnesses": sum(r["history_witnesses"] for r in rs),
        "history_verified": sum(r["history_verified"] for r in rs),
        "rejection_waterfall": dict(agg.most_common()),
        "rejection_waterfall_ui_only": dict(agg_ui.most_common()),
    }
  (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
  pairs = {}
  arms = sorted(summary)
  if "base" in arms:
    for arm in arms:
      if arm != "base":
        pairs[arm] = {k: paired(rows, "base", arm, k)
                      for k in ("success", "episode_length", "model_calls", "runtime_s")}
  (root / "paired.json").write_text(json.dumps(pairs, indent=2, ensure_ascii=False))
  print(json.dumps({a: {k: v for k, v in s.items() if not isinstance(v, dict)}
                    for a, s in summary.items()}, indent=1, ensure_ascii=False))
  return 0


if __name__ == "__main__":
  sys.exit(main())
