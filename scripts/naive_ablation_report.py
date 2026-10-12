#!/usr/bin/env python3
"""Intermediate read-outs per arm of a paired ablation run.

  python scripts/naive_ablation_report.py <root> [arm,arm,...]

For every arm: outcome (score, mean steps, success steps, inference calls) and
the mechanism read-outs that answer the three questions:
  graph      nodes per episode, revisit share at decision time
  explore    rounds, probes, restore failures, new screens and new controls
             discovered by probes, probes whose landing shows a task entity,
             probe-discovered edges the agent later used
  reasoning  graph skips (count, landed), replays (count, hops completed,
             hops planned, stopped-before-commit), context injections
Writes <root>/naive_ablation_report.json and prints a table.
"""

from __future__ import annotations

import collections
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from pathlib import Path


def load(path: Path) -> list[dict]:
  if not path.exists():
    return []
  out = []
  for line in path.read_text(errors="ignore").splitlines():
    try:
      out.append(json.loads(line))
    except ValueError:
      pass
  return out


def norm(x) -> str:
  return " ".join(str(x).casefold().split())


def episode(run: Path) -> collections.Counter:
  c = collections.Counter()
  ev = load(run / "serial_events.jsonl")
  steps = {e["step"] for e in ev if e.get("kind") == "step_begin"}
  c["steps_logged"] = len(steps)
  gates = [e for e in ev if e.get("kind") == "gate"]
  c["decision_steps"] = len(gates)
  c["revisit_steps"] = sum(1 for e in gates if (e.get("node_visits") or 0) >= 2)
  graph = {}
  gp = run / "progressive_belief_graph.json"
  if gp.exists():
    graph = json.loads(gp.read_text())
  agent_nodes = set()
  for e in graph.get("edges", []):
    if (e.get("execution_hit_count") or 0) > 0:
      agent_nodes.add(e.get("src_node"))
      agent_nodes.add(e.get("dst_node"))
  c["nodes"] = len(graph.get("nodes", []))
  c["edges"] = len(graph.get("edges", []))
  c["probe_edges_used"] = sum(1 for e in graph.get("edges", [])
                              if (e.get("probe_count") or 0) > 0
                              and ((e.get("execution_hit_count") or 0) > 0
                                   or (e.get("skip_success_count") or 0) > 0))
  c["probe_edges"] = sum(1 for e in graph.get("edges", []) if (e.get("probe_count") or 0) > 0)
  # Exploration.
  entities = set()
  for e in ev:
    if e.get("kind") == "obligations":
      entities = {norm(x) for x in (e.get("entities") or [])}
  for e in ev:
    if e.get("kind") == "explore":
      c["explore_rounds"] += 1
      c["probes"] += int(e.get("probes_completed") or 0)
      c["restore_failed_rounds"] += e.get("restore_status") != "RESTORED"
  new_screens = set()
  # Screens the agent itself stood on at decision time (gate events carry the
  # node it decided on); a probe that previewed one of them looked where the
  # episode then went.
  from android_world.parallel_exploration.belief_graph import ProgressiveBeliefGraph as _G
  visited = {e.get("node_id") for e in gates if e.get("node_id")} | agent_nodes
  for r in load(run / "probe_trace.jsonl"):
    disc = r.get("discovered") or {}
    c["probe_rows"] += 1
    c["new_controls"] += int(disc.get("new_element_count") or 0)
    dst = ((r.get("graph") or {}).get("dst") or {})
    key = (dst.get("activity"), dst.get("layout_signature"))
    if key[0]:
      new_screens.add(key)
      if key[1] and _G.make_node_id(key[0], key[1]) in visited:
        c["probes_previewing_agent_path"] += 1
    labels = {norm(x) for x in (dst.get("labels") or [])}
    if entities and labels & entities:
      c["probes_showing_entity"] += 1
    c["probe_restore_fail"] += not r.get("recovery_ok", True)
  c["probe_screens"] = len(new_screens)
  # Reasoning.
  for e in ev:
    k = e.get("kind")
    if k == "skip" and e.get("kind_detail") not in ("app_bootstrap", "store_prefill"):
      c["skips"] += 1
      c["skips_landed"] += bool(e.get("matched", e.get("successor_matched")))
    elif k == "procedure_execute":
      c["replays"] += 1
      c["replay_hops"] += int(e.get("completed") or 0)
      c["replay_planned"] += int(e.get("planned_hops") or 0)
      c["replay_to_commit"] += e.get("reason") == "stopped_before_commit"
    elif k == "graph_context" and e.get("injected"):
      c["context_steps"] += 1
      c["context_tokens"] += int(e.get("tokens") or 0)
    elif k == "explore_handoff":
      c["explore_handoffs"] += 1
      c["explore_handoff_replanned"] += bool(e.get("replanned"))
    elif k == "skip_refused_commit":
      c["skip_refused_commit"] += 1
  return c


def main() -> int:
  root = Path(sys.argv[1])
  rows = {}
  for line in (root / "pairs.jsonl").read_text().splitlines():
    r = json.loads(line)
    rows[r["task"]] = r
  arms = sys.argv[2].split(",") if len(sys.argv) > 2 else [a for a in next(iter(rows.values())) if a != "task"]
  report = {}
  for arm in arms:
    agg = collections.Counter()
    oks, steps, succ, calls = [], [], [], 0
    for t, r in rows.items():
      if arm not in r or r[arm].get("crash"):
        continue
      oks.append(r[arm]["ok"]); steps.append(r[arm]["steps"]); calls += r[arm].get("calls", 0)
      if r[arm]["ok"] > 0:
        succ.append(r[arm]["steps"])
      run = root / arm / t
      if run.exists():
        agg.update(episode(run))
    n = len(oks)
    report[arm] = {
        "tasks": n, "score": sum(oks), "mean_steps": sum(steps) / max(1, n),
        "success_steps": sum(succ) / max(1, len(succ)), "inference_calls": calls,
        "nodes_per_episode": agg["nodes"] / max(1, n),
        "revisit_share": agg["revisit_steps"] / max(1, agg["decision_steps"]),
        "explore_rounds": agg["explore_rounds"], "probes": agg["probes"],
        "restore_failed_rounds": agg["restore_failed_rounds"],
        "probe_new_screens": agg["probe_screens"], "probe_new_controls": agg["new_controls"],
        "probes_showing_task_entity": agg["probes_showing_entity"],
        "probes_previewing_agent_path": agg["probes_previewing_agent_path"],
        "probe_edges": agg["probe_edges"], "probe_edges_later_used": agg["probe_edges_used"],
        "explore_handoffs": agg["explore_handoffs"],
        "explore_handoff_replanned": agg["explore_handoff_replanned"],
        "skips": agg["skips"], "skips_landed": agg["skips_landed"],
        "skip_refused_commit": agg["skip_refused_commit"],
        "replays": agg["replays"], "replay_hops": agg["replay_hops"],
        "replay_planned_hops": agg["replay_planned"], "replay_to_commit": agg["replay_to_commit"],
        "context_steps": agg["context_steps"], "context_tokens": agg["context_tokens"],
        "steps_without_model": agg["replay_hops"] + agg["skips"],
    }
  (root / "naive_ablation_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
  keys = list(next(iter(report.values())).keys())
  print("metric".ljust(28) + "".join(a[:13].rjust(14) for a in arms))
  for k in keys:
    vals = []
    for a in arms:
      v = report[a][k]
      vals.append((f"{v:.2f}" if isinstance(v, float) else str(v)).rjust(14))
    print(k.ljust(28) + "".join(vals))
  return 0


if __name__ == "__main__":
  sys.exit(main())
