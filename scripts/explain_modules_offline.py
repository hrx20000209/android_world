#!/usr/bin/env python3
"""Offline module attribution for full_rec runs (no emulator needed).

  python scripts/explain_modules_offline.py <run_root> [<run_root> ...]

For every run root (e.g. results/sensys27_task_navigation/20261004_full_r5)
reads base/ and full_rec/ episodes and writes <run_root>/modules_offline.json
plus a markdown summary printed to stdout.

  A1 graph:        revisit share at decision time under two node keys that the
                   logs support offline: layout_sig (gate.node_visits) and the
                   exact screenshot sent to the model (api_calls.image_md5).
  A2 contribution: per episode, inferences removed by each module (replayed
                   hops, graph skips, app-launch fold) and the cost added
                   (contract calls / tokens), by task category and by whether
                   the state checker covers the app; paired step deltas.
  A3 provenance:   for every graph skip and every injected fact, whether the
                   edge it used was first produced by exploration (probe) or by
                   the model's own execution.
"""

from __future__ import annotations

import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
META = {m["task_name"]: m for m in json.loads((REPO / "android_world/task_metadata.json").read_text())}
MULTI_NAME = re.compile(r"Multiple|Duplicates|AllNotes|MergeNotes")


def category(task: str) -> str:
  tags = set(META.get(task, {}).get("tags", []))
  if "repetition" in tags or MULTI_NAME.search(task):
    return "multi-entity"
  if tags & {"information_retrieval", "screen_reading"} and not tags & {"data_entry", "data_edit"}:
    return "query"
  return "single-entity"


def jsonl(path: Path) -> list[dict]:
  if not path.exists():
    return []
  out = []
  for line in path.read_text(errors="ignore").splitlines():
    try:
      out.append(json.loads(line))
    except ValueError:
      pass
  return out


def revisit(run: Path) -> dict:
  ev = jsonl(run / "serial_events.jsonl")
  gates = [e for e in ev if e.get("kind") == "gate"]
  calls = [c for c in jsonl(run / "api_calls.jsonl") if c.get("purpose", "step") == "step"]
  seen, pixel_revisits = set(), 0
  for c in calls:
    key = tuple(c.get("image_md5") or ())
    if key and key in seen:
      pixel_revisits += 1
    seen.add(key)
  return {"decisions": len(gates) or len(calls),
          "layout_revisits": sum(1 for g in gates if (g.get("node_visits") or 0) >= 2),
          "pixel_decisions": len(calls), "pixel_revisits": pixel_revisits}


def graph_edges(run: Path) -> dict[str, dict]:
  path = run / "progressive_belief_graph.json"
  if not path.exists():
    return {}
  return {e["edge_id"]: e for e in json.loads(path.read_text()).get("edges", [])}


def origin(edge: dict | None) -> str:
  """Where an edge's evidence first came from, judged from its counters."""
  if edge is None:
    return "unknown"
  probed = (edge.get("probe_count") or 0) > 0
  executed = (edge.get("execution_hit_count") or 0) + (edge.get("execution_miss_count") or 0) > 0
  if probed and executed:
    return "probe+execution"
  if probed:
    return "probe"
  if executed:
    return "execution"
  return "seeded/other"


def episode(run: Path) -> dict:
  ev = jsonl(run / "serial_events.jsonl")
  calls = jsonl(run / "api_calls.jsonl")
  edges = graph_edges(run)
  reps = [e for e in ev if e.get("kind") == "procedure_execute"]
  skips = [e for e in ev if e.get("kind") == "skip" and e.get("kind_detail") not in ("app_bootstrap", "store_prefill")]
  facts = [e for e in ev if e.get("kind") == "graph_context" and e.get("injected")]
  explores = [e for e in ev if e.get("kind") == "explore"]
  return {
      "gui_calls": sum(1 for c in calls if c.get("purpose", "step") == "step"),
      "contract_calls": sum(1 for c in calls if c.get("purpose") == "obligation"),
      "contract_prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls if c.get("purpose") == "obligation"),
      "replays": len(reps),
      "replayed_hops": sum(int(e.get("completed") or 0) for e in reps),
      "replay_reasons": [e.get("reason") for e in reps],
      "skips": len(skips),
      "skips_landed": sum(1 for e in skips if e.get("matched")),
      "skip_origins": [origin(edges.get(e.get("edge_id"))) for e in skips],
      "bootstrap": sum(1 for e in ev if e.get("kind") == "skip" and e.get("kind_detail") == "app_bootstrap"),
      "context_steps": len(facts),
      "fact_origins": [origin(edges.get(eid)) for e in facts for eid in (e.get("selected_edge_ids") or [])],
      "explore_rounds": len(explores),
      "probes": sum(int(e.get("probes_completed") or 0) for e in explores),
      "restore_failures": sum(1 for e in explores if e.get("restore_status") not in (None, "RESTORED")),
      "covered": any(e.get("kind") == "state_check" and e.get("covered") for e in ev)
                 or any(e.get("kind") == "procedure_hop" and e.get("state_ok") is not None
                        and e.get("state_reason") != "no_spec_for_package" for e in ev),
  }


def mean(xs):
  xs = list(xs)
  return round(st.mean(xs), 2) if xs else None


def analyse(root: Path, ref: dict[str, tuple[float, float]]) -> dict:
  pairs = {}
  for p in jsonl(root / "pairs.jsonl"):
    if "base" in p and "full_rec" in p:
      pairs[p["task"]] = p  # a rerun line supersedes the quarantined one
  rows = []
  for task, p in sorted(pairs.items()):
    if p["base"].get("crash") or p["full_rec"].get("crash"):
      continue
    b = (p["base"]["ok"], p["base"]["steps"])
    f = (p["full_rec"]["ok"], p["full_rec"]["steps"])
    ep = episode(root / "full_rec" / task)
    rows.append({"task": task, "category": category(task), "base": b, "full_rec": f,
                 "ref_4b2": ref.get(task), "rev_full": revisit(root / "full_rec" / task),
                 "rev_base": revisit(root / "base" / task), **ep})

  def block(sel: list[dict]) -> dict:
    both = [r for r in sel if r["base"][0] > 0 and r["full_rec"][0] > 0]
    ok = lambda arm: [r for r in sel if r[arm][0] > 0]
    out = {"tasks": len(sel)}
    for arm in ("base", "full_rec"):
      out[arm] = {"success": sum(r[arm][0] for r in sel), "steps": mean(r[arm][1] for r in sel),
                  "success_steps": mean(r[arm][1] for r in ok(arm))}
    refs = [r for r in sel if r["ref_4b2"]]
    out["4B_2"] = {"success": sum(r["ref_4b2"][0] for r in refs), "steps": mean(r["ref_4b2"][1] for r in refs),
                   "success_steps": mean(r["ref_4b2"][1] for r in refs if r["ref_4b2"][0] > 0)}
    out["both_success"] = {"n": len(both), "base": mean(r["base"][1] for r in both),
                           "full_rec": mean(r["full_rec"][1] for r in both)}
    out["removed_inferences"] = {"replayed_hops": sum(r["replayed_hops"] for r in sel),
                                 "graph_skips": sum(r["skips"] for r in sel),
                                 "app_launch_folds": sum(r["bootstrap"] for r in sel)}
    out["added_cost"] = {"contract_calls": sum(r["contract_calls"] for r in sel),
                         "contract_prompt_tokens": sum(r["contract_prompt_tokens"] for r in sel),
                         "gui_calls_base_vs_full": None}
    out["mechanism"] = {"replays": sum(r["replays"] for r in sel),
                        "replay_reasons": dict(collections.Counter(x for r in sel for x in r["replay_reasons"])),
                        "skips": sum(r["skips"] for r in sel), "skips_landed": sum(r["skips_landed"] for r in sel),
                        "context_steps": sum(r["context_steps"] for r in sel),
                        "explore_rounds": sum(r["explore_rounds"] for r in sel), "probes": sum(r["probes"] for r in sel),
                        "restore_failures": sum(r["restore_failures"] for r in sel)}
    # Paired step delta among both-success tasks, split by which module acted.
    deltas = collections.defaultdict(list)
    for r in both:
      who = "replay" if r["replays"] else "skip" if r["skips"] else "context" if r["context_steps"] else "none"
      deltas[who].append(r["full_rec"][1] - r["base"][1])
    out["paired_delta_by_module"] = {k: {"n": len(v), "mean": mean(v), "values": v} for k, v in deltas.items()}
    return out

  report = {"all": block(rows)}
  for cat in ("multi-entity", "single-entity", "query"):
    report[cat] = block([r for r in rows if r["category"] == cat])
  for cov in (True, False):
    report["checker_covered" if cov else "checker_uncovered"] = block([r for r in rows if r["covered"] == cov])
  report["A1_revisit"] = {
      arm: {k: sum(r[f"rev_{arm}"][k] for r in rows) for k in ("decisions", "layout_revisits", "pixel_decisions", "pixel_revisits")}
      for arm in ("base", "full")}
  report["A3_provenance"] = {
      "skip_edge_origin": dict(collections.Counter(o for r in rows for o in r["skip_origins"])),
      "fact_edge_origin": dict(collections.Counter(o for r in rows for o in r["fact_origins"])),
      "replay_template_origin": "execution (by construction: templates are built only from executed hops)",
  }
  report["per_task"] = rows
  return report


def main() -> int:
  ref = {}
  for line in (REPO / "results/4B_2.txt").read_text().splitlines()[2:]:
    p = line.split()
    if len(p) >= 5 and p[0][:1].isupper():
      try:
        ref[p[0]] = (float(p[3]), float(p[4]))
      except ValueError:
        pass
  for arg in sys.argv[1:]:
    root = Path(arg)
    rep = analyse(root, ref)
    (root / "modules_offline.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=list))
    print(f"\n######## {root.name} ########")
    for key in ("all", "multi-entity", "single-entity", "query", "checker_covered", "checker_uncovered"):
      b = rep[key]
      print(f"\n[{key}] n={b['tasks']}")
      for arm in ("4B_2", "base", "full_rec"):
        a = b[arm]
        print(f"  {arm:8s} 成功 {a['success']:5.1f}  平均步数 {a['steps']}  成功任务步数 {a['success_steps']}")
      print(f"  两边都成功 {b['both_success']}")
      print(f"  省掉的推理 {b['removed_inferences']}  新增成本 {dict((k,v) for k,v in b['added_cost'].items() if v is not None)}")
      print(f"  机制读数 {b['mechanism']}")
      deltas = ", ".join("%s: n=%d mean=%s" % (k, v["n"], v["mean"]) for k, v in b["paired_delta_by_module"].items())
      print(f"  配对步数差(按起作用的模块) {{{deltas}}}")
    print(f"\nA1 重访占比 {rep['A1_revisit']}")
    print(f"A3 证据来源 {rep['A3_provenance']}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
