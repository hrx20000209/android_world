"""What the belief graph looks like, and which parts of it get used.

  python scripts/graph_evidence.py RUN_DIR OUT_DIR [BASELINE_TXT]

Cumulative screens per step is reconstructed exactly rather than estimated:
every step logs a gate event carrying the visit count the current screen had
as of the previous step, so a zero there marks the first time the episode has
stood on that screen.
"""
import collections
import json
import math
import pathlib
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.axisbelow": True,
})
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
     "#999999", "#F0E442"]
SYSTEM_HINTS = ("launcher", "systemui", "permissioncontroller",
                "packageinstaller", "nexuslauncher")
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def baseline_steps(path):
  out = {}
  for line in open(path):
    q = line.split()
    if len(q) >= 5 and q[0][0].isalpha() and q[1].isdigit():
      try:
        out[q[0]] = float(q[4])
      except ValueError:
        pass
  return out


def app_of(task_dir):
  graph = task_dir / "progressive_belief_graph.json"
  if not graph.exists():
    return "unknown"
  try:
    j = json.load(graph.open())
  except ValueError:
    return "unknown"
  counts = collections.Counter()
  for node in j.get("nodes", []):
    pkg = node.get("package", "")
    if pkg and not any(h in pkg for h in SYSTEM_HINTS):
      counts[pkg] += int(node.get("visit_count") or 1)
  return counts.most_common(1)[0][0].split(".")[-1] if counts else "unknown"


def collect(run, base):
  rows = []
  for log in sorted(run.glob("*.log")):
    task = log.stem
    if task == "batch":
      continue
    text = log.read_text(errors="ignore")
    if "Task Successful" not in text and "Task Failed" not in text:
      continue
    d = run / task
    events = d / "serial_events.jsonl"
    if not events.exists():
      continue
    first_visit, used_edges, skips = [], set(), 0
    for line in events.open():
      try:
        e = json.loads(line)
      except ValueError:
        continue
      if e.get("kind") == "gate":
        # The snapshot the gate reads is refreshed after this step's upsert,
        # so the current screen already counts its own visit: 1 means the
        # episode is standing here for the first time, not zero.
        first_visit.append(int((e.get("node_visits") or 0) <= 1))
      elif e.get("kind") == "graph_context" and e.get("injected"):
        used_edges.update(e.get("selected_edge_ids") or ())
      elif e.get("kind") == "skip" and e.get("kind_detail") != "app_bootstrap":
        skips += 1
    cumulative = []
    total = 0
    for new in first_visit:
      total += new
      cumulative.append(total)
    graph = d / "progressive_belief_graph.json"
    nodes, edges = [], []
    if graph.exists():
      try:
        j = json.load(graph.open())
        nodes, edges = j.get("nodes", []), j.get("edges", [])
      except ValueError:
        pass
    b = base.get(task)
    rows.append({
        "task": task, "app": app_of(d), "success": int("Task Successful" in text),
        "steps": len(first_visit), "cumulative_screens": cumulative,
        "nodes": nodes, "edges": edges, "used_edge_ids": sorted(used_edges),
        "skips": skips,
        "difficulty": ("easy" if b is not None and b <= 4 else
                       "medium" if b is not None and b <= 8 else "hard"),
    })
  return rows


def fig_growth_by(rows, key, out, fname, title, min_n=4, max_step=15):
  groups = collections.defaultdict(list)
  for r in rows:
    if r["cumulative_screens"]:
      groups[r[key]].append(r)
  groups = {k: v for k, v in groups.items() if len(v) >= min_n}
  if not groups:
    return
  names = ([n for n in DIFFICULTY_ORDER if n in groups] if key == "difficulty"
           else sorted(groups, key=lambda k: -len(groups[k]))[:6])
  fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
  for ax, only_ok in zip(axes, (False, True)):
    for i, name in enumerate(names):
      series = [r["cumulative_screens"] for r in groups[name]
                if r["success"] or not only_ok]
      if not series:
        continue
      xs, ys = [], []
      for step in range(max_step):
        vals = [s[step] for s in series if len(s) > step]
        if len(vals) < 2:
          break
        xs.append(step + 1)
        ys.append(statistics.mean(vals))
      if xs:
        ax.plot(xs, ys, color=C[i % len(C)], lw=1.3, marker="o", ms=2.5,
                label=f"{name} (n={len(series)})")
    ax.set_xlabel("step")
    ax.set_ylabel("distinct screens seen so far")
    ax.set_title("successful tasks" if only_ok else "all tasks")
    ax.legend(frameon=False, fontsize=6.5)
  fig.suptitle(title, y=1.02)
  fig.savefig(out / f"{fname}.pdf")
  fig.savefig(out / f"{fname}.png")
  plt.close(fig)


def fig_revisit(rows, out):
  """How concentrated revisiting is: most screens are seen once and never again."""
  fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
  for ax, only_ok in zip(axes, (False, True)):
    counts = collections.Counter()
    for r in rows:
      if only_ok and not r["success"]:
        continue
      for n in r["nodes"]:
        counts[min(int(n.get("visit_count") or 0), 6)] += 1
    if not counts:
      continue
    total = sum(counts.values())
    ks = sorted(counts)
    ax.bar([str(k) if k < 6 else "6+" for k in ks],
           [counts[k] / total * 100 for k in ks],
           color=C[0], edgecolor="black", lw=0.4)
    ax.set_xlabel("times the episode stood on that screen")
    ax.set_ylabel("share of screens (%)")
    ax.set_title("successful tasks" if only_ok else "all tasks")
    once = counts.get(1, 0) / total * 100
    ax.text(0.97, 0.92, f"seen once: {once:.0f}%", transform=ax.transAxes,
            ha="right", fontsize=7)
  fig.suptitle("Screen revisits within one episode", y=1.02)
  fig.savefig(out / "graph_revisit_distribution.pdf")
  fig.savefig(out / "graph_revisit_distribution.png")
  plt.close(fig)


def fig_used_nodes(rows, out):
  """What separates a screen the graph could speak about from one it could not.

  A node counts as used when an injected fact was drawn from one of its
  outgoing edges. Everything else on the same screens is the control.
  """
  used, unused = collections.defaultdict(list), collections.defaultdict(list)
  for r in rows:
    by_id = {n.get("node_id"): n for n in r["nodes"]}
    out_deg = collections.Counter(e.get("src_node") for e in r["edges"])
    labelled = collections.Counter(
        e.get("src_node") for e in r["edges"] if e.get("discovered_labels"))
    used_srcs = {e.get("src_node") for e in r["edges"]
                 if e.get("edge_id") in set(r["used_edge_ids"])}
    for node_id, node in by_id.items():
      bucket = used if node_id in used_srcs else unused
      bucket["visits"].append(int(node.get("visit_count") or 0))
      bucket["out_degree"].append(out_deg.get(node_id, 0))
      bucket["labelled_edges"].append(labelled.get(node_id, 0))
      ent = node.get("decision_entropy")
      if isinstance(ent, (int, float)) and math.isfinite(ent):
        bucket["entropy"].append(ent)
      bucket["labels"].append(len(node.get("salient_ui_labels") or ()))
  fields = [("visits", "times revisited"), ("out_degree", "known exits"),
            ("labelled_edges", "exits with known destination"),
            ("entropy", "decision entropy"), ("labels", "salient UI labels")]
  fig, ax = plt.subplots(figsize=(4.6, 2.6))
  x = range(len(fields))
  u = [statistics.mean(used[f]) if used[f] else 0 for f, _ in fields]
  n = [statistics.mean(unused[f]) if unused[f] else 0 for f, _ in fields]
  ax.bar([i - 0.2 for i in x], n, width=0.38, color="#BBBBBB",
         edgecolor="black", lw=0.4,
         label=f"never used (n={len(unused['visits'])})")
  ax.bar([i + 0.2 for i in x], u, width=0.38, color=C[0], edgecolor="black",
         lw=0.4, label=f"supplied an injected fact (n={len(used['visits'])})")
  ax.set_xticks(list(x))
  ax.set_xticklabels([lab for _, lab in fields], rotation=25, ha="right",
                     rotation_mode="anchor")
  ax.set_ylabel("mean per screen")
  ax.legend(frameon=False, fontsize=6.5)
  fig.suptitle("What distinguishes a screen the graph could speak about", y=1.02)
  fig.savefig(out / "graph_used_node_profile.pdf")
  fig.savefig(out / "graph_used_node_profile.png")
  plt.close(fig)
  return {"used": {k: (statistics.mean(v) if v else 0) for k, v in used.items()},
          "unused": {k: (statistics.mean(v) if v else 0) for k, v in unused.items()},
          "used_nodes": len(used["visits"]), "unused_nodes": len(unused["visits"])}


def main(argv):
  if len(argv) < 3:
    print(__doc__)
    return 2
  run, out = pathlib.Path(argv[1]), pathlib.Path(argv[2])
  out.mkdir(parents=True, exist_ok=True)
  base = baseline_steps(pathlib.Path(argv[3])) if len(argv) > 3 else {}
  rows = collect(run, base)
  fig_growth_by(rows, "difficulty", out, "graph_growth_by_difficulty",
                "Screens discovered per step, by baseline difficulty")
  fig_growth_by(rows, "app", out, "graph_growth_by_app",
                "Screens discovered per step, by app")
  fig_revisit(rows, out)
  profile = fig_used_nodes(rows, out)
  (out / "graph_stats.json").write_text(json.dumps({
      "tasks": len(rows),
      "used_node_profile": profile,
      "per_task": [{k: v for k, v in r.items()
                    if k not in ("nodes", "edges")} for r in rows],
  }, indent=1, ensure_ascii=False), encoding="utf-8")
  print(f"tasks {len(rows)}  ->  {out}")
  for p in sorted(out.glob("graph_*.png")):
    print("  ", p.name)
  print("\n被利用 vs 未被利用的节点画像:")
  for k in ("visits", "out_degree", "labelled_edges", "entropy", "labels"):
    print(f"  {k:<20} 被利用 {profile['used'].get(k,0):>6.2f}   "
          f"未被利用 {profile['unused'].get(k,0):>6.2f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
