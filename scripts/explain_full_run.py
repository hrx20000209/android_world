#!/usr/bin/env python3
"""Interpretable read-outs of one arm of a full run, per question.

  python scripts/explain_full_run.py <root> <arm>

Writes <root>/<arm>_explain.json (aggregates) and <root>/<arm>_traces/<task>.md
(one row per step: which graph mechanism acted and why). Read-outs:
  Q1 graph:      nodes, revisit share, edges, certified hops, contract entities
  Q2 exploration: rounds, probes, rounds aimed at task entities, entity hits
  Q3 reasoning:  per-step path (replay / graph skip / graph context / plain),
                 skip landing accuracy, replay stop reasons, facts injected
"""

from __future__ import annotations

import collections
import json
import sys
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


def episode(run: Path) -> tuple[dict, list[str]]:
  ev = load(run / "serial_events.jsonl")
  graph = {}
  gp = run / "progressive_belief_graph.json"
  if gp.exists():
    graph = json.loads(gp.read_text())
  steps = sorted({e["step"] for e in ev if e.get("kind") == "step_begin"})
  per_step = collections.defaultdict(lambda: {"path": "plain"})
  visits = collections.Counter()
  revisit_steps = 0
  for e in ev:
    k, s = e.get("kind"), e.get("step")
    if k == "procedure_execute":
      per_step[s]["path"] = "replay"
      per_step[s]["detail"] = f"{e['entity']}: {e['completed']}/{e['planned_hops']} hops, {e['reason']}"
    elif k == "skip" and e.get("kind_detail") not in ("app_bootstrap", "store_prefill"):
      if per_step[s]["path"] == "plain":
        per_step[s]["path"] = "graph_skip"
      per_step[s]["skip_matched"] = e.get("matched", e.get("successor_matched"))
    elif k == "skip" and e.get("kind_detail") == "app_bootstrap":
      per_step[s]["bootstrap"] = True
    elif k == "graph_context" and e.get("injected"):
      if per_step[s]["path"] == "plain":
        per_step[s]["path"] = "graph_context"
      per_step[s]["facts"] = (e.get("text") or "")[:160]
    elif k == "explore":
      per_step[s]["probes"] = int(e.get("probes_completed") or 0)
    elif k == "procedure_plan" and e.get("plan") is None:
      per_step[s].setdefault("replay_refusal", e.get("rejection"))
  for e in ev:
    if e.get("kind") == "task_navigation_plan" and e.get("current"):
      key = (e["current"]["activity"], e["current"]["structural_signature"])
      visits[key] += 1
  # A step is a revisit when, at decision time, its node (activity + layout
  # signature) had already been stood on earlier in the episode.
  nodes = graph.get("nodes", [])
  revisit_steps = sum(1 for e in ev if e.get("kind") == "gate" and (e.get("node_visits") or 0) >= 2)
  hops = [e for e in ev if e.get("kind") == "procedure_hop" and not e.get("barrier")]
  explores = [e for e in ev if e.get("kind") == "explore"]
  scored = []
  for f in run.rglob("scored_candidates.jsonl"):
    scored.extend(r for r in load(f) if r.get("policy") == "task_navigation_frontier")
  aimed_rounds = len({r.get("trial_id") for r in scored})
  entity_hits = sum(1 for r in scored if (r.get("target_coverage") or 0) > 0)
  ents = next((e.get("entities") for e in ev if e.get("kind") == "obligations"), [])
  # Certifiable autonomy: a step in an app the state checker does not cover
  # logs skip_refused_uncertifiable; there the graph only records, never acts.
  uncovered = {e.get("step") for e in ev if e.get("kind") == "skip_refused_uncertifiable"}
  reps = [e for e in ev if e.get("kind") == "procedure_execute"]
  summary = {
      "steps": len(steps),
      "nodes": len(nodes), "edges": len(graph.get("edges", [])),
      "revisit_steps": revisit_steps,
      "hops_recorded": len(hops),
      "hops_certified_unchanged": sum(1 for h in hops if h.get("state_ok") is True),
      "hops_state_changed": sum(1 for h in hops if h.get("state_ok") is False),
      "contract_entities": ents,
      "explore_rounds": len(explores),
      "probes": sum(int(e.get("probes_completed") or 0) for e in explores),
      "rounds_aimed_at_entities": aimed_rounds,
      "candidates_matching_entity": entity_hits,
      "paths": dict(collections.Counter(per_step[s]["path"] for s in steps)),
      "skips": sum(1 for s in steps if per_step[s]["path"] == "graph_skip"),
      "skips_landed": sum(1 for s in steps if per_step[s].get("skip_matched")),
      "replays": [per_step[s]["detail"] for s in steps if per_step[s]["path"] == "replay"],
      "covered": len(uncovered) < max(1, len(steps)),
      # Inferences the graph removed: every replayed hop past the first and
      # every graph skip is an action executed without a model call; the app
      # launch fold removes the first one.
      "replayed_hops": sum(int(e.get("completed") or 0) for e in reps),
      "bootstrap_folds": sum(1 for s in steps if per_step[s].get("bootstrap")),
      "skip_refusals": collections.Counter(
          e.get("kind") for e in ev if str(e.get("kind", "")).startswith("skip_refused")),
  }
  lines = ["| step | path | detail |", "|---|---|---|"]
  for s in steps:
    d = per_step[s]
    detail = d.get("detail") or d.get("facts") or d.get("replay_refusal") or ""
    extra = []
    if d.get("bootstrap"):
      extra.append("app launch folded")
    if d.get("probes"):
      extra.append(f"{d['probes']} probes")
    lines.append(f"| {s} | {d['path']} | {'; '.join(extra + [str(detail)])[:200]} |")
  return summary, lines


def main() -> int:
  root, arm = Path(sys.argv[1]), sys.argv[2]
  out_dir = root / f"{arm}_traces"
  out_dir.mkdir(exist_ok=True)
  agg = collections.Counter()
  by_source = {True: collections.Counter(), False: collections.Counter()}
  refusals = collections.Counter()
  paths = collections.Counter()
  per_task = {}
  for log in sorted((root / arm).glob("*.log")):
    task = log.stem
    run = root / arm / task
    if not run.exists():
      continue
    summary, lines = episode(run)
    per_task[task] = summary
    (out_dir / f"{task}.md").write_text(f"# {task}\n\n" + "\n".join(lines) + "\n")
    for k in ("steps", "nodes", "edges", "revisit_steps", "hops_recorded",
              "hops_certified_unchanged", "hops_state_changed", "explore_rounds", "probes",
              "rounds_aimed_at_entities", "candidates_matching_entity", "skips", "skips_landed"):
      agg[k] += summary[k]
    agg["replays"] += len(summary["replays"])
    for k in ("replayed_hops", "bootstrap_folds"):
      agg[k] += summary[k]
    refusals.update(summary["skip_refusals"])
    src = by_source[summary["covered"]]
    src["tasks"] += 1
    for k in ("steps", "revisit_steps", "explore_rounds", "probes", "skips", "skips_landed",
              "replayed_hops", "hops_certified_unchanged", "hops_state_changed"):
      src[k] += summary[k]
    src["replays"] += len(summary["replays"])
    agg["tasks_with_contract"] += bool(summary["contract_entities"])
    paths.update(summary["paths"])
  sources = {("checker_covered" if k else "uncovered"): dict(v) for k, v in by_source.items()}
  out = {"aggregate": dict(agg), "paths": dict(paths), "by_source": sources,
         "skip_refusals": dict(refusals), "per_task": per_task}
  (root / f"{arm}_explain.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
  print(json.dumps({"aggregate": dict(agg), "paths": dict(paths), "by_source": sources,
                    "skip_refusals": dict(refusals)}, ensure_ascii=False, indent=1))
  return 0


if __name__ == "__main__":
  sys.exit(main())
