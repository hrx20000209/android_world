"""Evidence for the exploration claim: does ranking beat chance, and what does
the graph look like as it is built?

Writes machine-readable intermediate results next to the figures so the
numbers in a paper can be traced back to the run that produced them.

  python scripts/exploration_evidence.py OUT_DIR RUN_DIR [RUN_DIR ...]
"""
import collections
import json
import pathlib
import re
import statistics
import sys

BOUNDS = re.compile(r"\((\d+), (\d+), (\d+), (\d+)\)")
SCREEN_AREA = 1080 * 2400
CONTAINER_FRACTION = 0.20   # above this a "hit" is a container, not the control


def _inside(x, y, b):
  return b[0] <= x <= b[2] and b[1] <= y <= b[3]


def _area(b):
  return max(1, (b[2] - b[0]) * (b[3] - b[1]))


def task_order(root: pathlib.Path) -> list[str]:
  """Tasks in the order the batch ran them, by first-event mtime."""
  rows = []
  for events in root.glob("*/serial_events.jsonl"):
    rows.append((events.stat().st_mtime, events.parent.name))
  return [name for _, name in sorted(rows)]


def scan_run(root: pathlib.Path) -> dict:
  """Everything the figures need from one run, in a single pass."""
  out = {
      "run": root.name,
      "tasks": {},
      "graph_growth": [],       # cumulative graph size, in run order
      "per_step_growth": {},    # task -> [(step, nodes, edges)]
      "probe_rank_hits": [],    # rank of the model's next control when probed
      "probe_outcomes": collections.Counter(),
      "filter_reasons": collections.Counter(),
  }
  order = task_order(root)
  seen_nodes: set[str] = set()
  seen_edges: set[str] = set()
  for index, task in enumerate(order, start=1):
    d = root / task
    log = root / f"{task}.log"
    text = log.read_text(errors="ignore") if log.exists() else ""
    ok = int("Task Successful" in text)
    events = d / "serial_events.jsonl"
    steps = rounds = probes = unrestored = aligned = guessed = skips = 0
    if events.exists():
      for line in events.open():
        try:
          e = json.loads(line)
        except ValueError:
          continue
        kind = e.get("kind")
        if kind == "inference":
          steps += 1
        elif kind == "explore":
          rounds += 1
          probes += int(e.get("probes_completed") or 0)
          if e.get("restore_status") != "RESTORED":
            unrestored += 1
          out["probe_outcomes"][e.get("restore_status") or "?"] += 1
        elif kind == "prefix_check":
          aligned += int(e.get("aligned") or 0)
          guessed += int(e.get("action_was_guessed") or 0)
        elif kind == "skip" and e.get("kind_detail") != "app_bootstrap":
          skips += 1
    out["tasks"][task] = {
        "order": index, "success": ok, "steps": steps, "rounds": rounds,
        "probes": probes, "unrestored": unrestored, "aligned": aligned,
        "guessed": guessed, "skips": skips,
    }
    graph = d / "progressive_belief_graph.json"
    if graph.exists():
      try:
        g = json.load(graph.open())
      except ValueError:
        g = {}
      for node in g.get("nodes", []):
        seen_nodes.add(node.get("node_id", ""))
      for edge in g.get("edges", []):
        seen_edges.add(edge.get("edge_id", ""))
      out["graph_growth"].append({
          "order": index, "task": task,
          "task_nodes": len(g.get("nodes", [])),
          "task_edges": len(g.get("edges", [])),
          "cumulative_nodes": len(seen_nodes),
          "cumulative_edges": len(seen_edges),
      })
    filtered = d / "filtered_elements.jsonl"
    if filtered.exists():
      for line in filtered.open():
        try:
          out["filter_reasons"][json.loads(line).get("reason", "?")] += 1
        except ValueError:
          pass
    out["probe_rank_hits"].extend(_rank_hits(d))
  return out


def _rank_hits(task_dir: pathlib.Path) -> list[dict]:
  """Where the model's NEXT control sat in this round's ranking.

  The exploration claim reduces to this number: a ranker that is no better
  than chance puts the control the model goes on to press at the middle of
  the list, and one that works puts it near the top.
  """
  scored = task_dir / "scored_candidates.jsonl"
  events = task_dir / "serial_events.jsonl"
  if not (scored.exists() and events.exists()):
    return []
  actions = {}
  for line in events.open():
    try:
      e = json.loads(line)
    except ValueError:
      continue
    if e.get("kind") == "inference":
      a = e.get("action") or {}
      if a.get("x") is not None:
        actions[e["step"]] = (a["x"], a["y"])
  rounds = collections.defaultdict(list)
  for line in scored.open():
    try:
      r = json.loads(line)
    except ValueError:
      continue
    m = BOUNDS.search(r.get("element_identity", "") or "")
    if m:
      rounds[r.get("step")].append((tuple(map(int, m.groups())), r))
  hits = []
  for step, candidates in rounds.items():
    nxt = actions.get((step or 0) + 1)
    if not nxt or len(candidates) < 2:
      continue
    inside = [c for c in candidates
              if _inside(nxt[0], nxt[1], c[0])
              and _area(c[0]) / SCREEN_AREA < CONTAINER_FRACTION]
    if not inside:
      continue
    best = min(inside, key=lambda c: _area(c[0]))
    hits.append({
        "task": task_dir.name, "step": step,
        "rank": best[1].get("rank"),
        "candidates": len(candidates),
        "selected_for_probe": bool(best[1].get("selected_for_probe")),
    })
  return hits


def summarise(scan: dict) -> dict:
  tasks = scan["tasks"]
  wins = sum(t["success"] for t in tasks.values())
  ok_steps = [t["steps"] for t in tasks.values() if t["success"] and t["steps"]]
  all_steps = [t["steps"] for t in tasks.values() if t["steps"]]
  ranks = [h["rank"] for h in scan["probe_rank_hits"] if h["rank"]]
  cands = [h["candidates"] for h in scan["probe_rank_hits"] if h["candidates"]]
  # A uniform ranker puts the target at the middle of the list; this is the
  # number the measured mean rank has to beat for the ranking to have done
  # anything at all.
  chance = statistics.mean([(c + 1) / 2 for c in cands]) if cands else 0.0
  return {
      "run": scan["run"],
      "tasks": len(tasks),
      "success": wins,
      "success_rate": wins / len(tasks) if tasks else 0.0,
      "mean_steps_all": statistics.mean(all_steps) if all_steps else 0.0,
      "mean_steps_success": statistics.mean(ok_steps) if ok_steps else 0.0,
      "explore_rounds": sum(t["rounds"] for t in tasks.values()),
      "probes": sum(t["probes"] for t in tasks.values()),
      "unrestored": sum(t["unrestored"] for t in tasks.values()),
      "prefix_aligned": sum(t["aligned"] for t in tasks.values()),
      "prefix_guessed": sum(t["guessed"] for t in tasks.values()),
      "graph_skips": sum(t["skips"] for t in tasks.values()),
      "tasks_with_probes": sum(1 for t in tasks.values() if t["probes"]),
      "rank_hits": len(ranks),
      "mean_rank_of_next_control": statistics.mean(ranks) if ranks else 0.0,
      "median_rank_of_next_control": statistics.median(ranks) if ranks else 0.0,
      "rank1_hits": sum(1 for r in ranks if r == 1),
      "mean_candidates_per_round": statistics.mean(cands) if cands else 0.0,
      "chance_rank": chance,
      "rank_lift_vs_chance": (chance - statistics.mean(ranks)) if ranks else 0.0,
      "final_cumulative_nodes": (scan["graph_growth"][-1]["cumulative_nodes"]
                                 if scan["graph_growth"] else 0),
      "final_cumulative_edges": (scan["graph_growth"][-1]["cumulative_edges"]
                                 if scan["graph_growth"] else 0),
  }


def main(argv: list[str]) -> int:
  if len(argv) < 3:
    print(__doc__)
    return 2
  out_dir = pathlib.Path(argv[1])
  out_dir.mkdir(parents=True, exist_ok=True)
  summaries = []
  for run in argv[2:]:
    root = pathlib.Path(run)
    if not root.exists():
      print(f"skip (missing): {run}")
      continue
    scan = scan_run(root)
    (out_dir / f"{scan['run']}_scan.json").write_text(
        json.dumps(scan, indent=1, default=list), encoding="utf-8")
    s = summarise(scan)
    summaries.append(s)
    print(f"{s['run']}: {s['success']}/{s['tasks']} 成功  "
          f"成功任务均步 {s['mean_steps_success']:.2f}  "
          f"探测 {s['probes']}  对齐 {s['prefix_aligned']}  "
          f"命中排名 {s['mean_rank_of_next_control']:.2f} vs 随机 {s['chance_rank']:.2f}"
          f" (提升 {s['rank_lift_vs_chance']:+.2f})")
  (out_dir / "summary.json").write_text(
      json.dumps(summaries, indent=1), encoding="utf-8")
  print(f"\n中间结果写入 {out_dir}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
