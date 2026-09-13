"""Which screens the graph can actually speak about, and what marks them out.

  python scripts/graph_node_profile.py OUT_DIR RUN_DIR [RUN_DIR ...]

A screen counts as USED when a fact drawn from one of its outgoing edges was
injected into the prompt, or when an edge leaving it was replayed by a skip.
Everything else the episode stood on is the control. Runs passed together must
be the same configuration - this pools them for sample size, it does not
compare them.

salient_ui_labels is deliberately not plotted: it is only filled in on nodes
seeded from cross-task memory, so the difference it shows is a field-population
artefact rather than a property of the screens.
"""
import collections
import json
import math
import pathlib
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.axisbelow": True,
})
USED_C, UNUSED_C = "#0072B2", "#999999"

FIELDS = [
    ("visits", "times the episode returned to it"),
    ("out_degree", "known exits"),
    ("labelled_edges", "exits with a known destination"),
    ("entropy", "decision entropy"),
    ("probe_edges", "exits discovered by probing"),
]


def collect(runs):
  used = collections.defaultdict(list)
  unused = collections.defaultdict(list)
  for run in runs:
    for graph_path in pathlib.Path(run).glob("*/progressive_belief_graph.json"):
      task = graph_path.parent
      try:
        g = json.load(graph_path.open())
      except ValueError:
        continue
      events = task / "serial_events.jsonl"
      touched = set()
      if events.exists():
        for line in events.open():
          try:
            e = json.loads(line)
          except ValueError:
            continue
          if e.get("kind") == "graph_context" and e.get("injected"):
            touched.update(e.get("selected_edge_ids") or ())
      edges = g.get("edges", [])
      out_deg = collections.Counter(e.get("src_node") for e in edges)
      labelled = collections.Counter(
          e.get("src_node") for e in edges if e.get("discovered_labels"))
      probed = collections.Counter(
          e.get("src_node") for e in edges if (e.get("probe_count") or 0))
      used_srcs = {e.get("src_node") for e in edges
                   if e.get("edge_id") in touched}
      for node in g.get("nodes", []):
        nid = node.get("node_id")
        bucket = used if nid in used_srcs else unused
        bucket["visits"].append(int(node.get("visit_count") or 0))
        bucket["out_degree"].append(out_deg.get(nid, 0))
        bucket["labelled_edges"].append(labelled.get(nid, 0))
        bucket["probe_edges"].append(probed.get(nid, 0))
        ent = node.get("decision_entropy")
        bucket["entropy"].append(
            ent if isinstance(ent, (int, float)) and math.isfinite(ent) else 0.0)
  return used, unused


def figure(used, unused, out, label):
  rows = list(reversed(FIELDS))
  y = range(len(rows))
  u = [statistics.mean(used[f]) if used[f] else 0.0 for f, _ in rows]
  n = [statistics.mean(unused[f]) if unused[f] else 0.0 for f, _ in rows]
  fig, ax = plt.subplots(figsize=(5.2, 2.8))
  for i, (a, b) in enumerate(zip(n, u)):
    ax.plot([a, b], [i, i], color="#CCCCCC", lw=2.0, zorder=1,
            solid_capstyle="round")
  ax.scatter(n, list(y), s=42, color=UNUSED_C, edgecolor="black",
             linewidth=0.5, zorder=3,
             label=f"never used ({len(unused['visits'])} screens)")
  ax.scatter(u, list(y), s=42, color=USED_C, edgecolor="black",
             linewidth=0.5, zorder=3,
             label=f"supplied a fact to the prompt ({len(used['visits'])})")
  span = max(u + n) or 1.0
  for i, (a, b) in enumerate(zip(n, u)):
    if a > 0:
      ax.text(b + span * 0.035, i, f"{b/a:.1f}×", va="center", fontsize=7.5,
              color=USED_C, fontweight="bold")
  ax.set_yticks(list(y))
  ax.set_yticklabels([lab for _, lab in rows])
  ax.set_xlim(-span * 0.03, span * 1.22)
  ax.set_xlabel("mean per screen")
  ax.set_title(f"What marks out a screen the graph can speak about ({label})")
  ax.legend(frameon=False, loc="lower right")
  ax.grid(axis="y", visible=False)
  fig.savefig(out / "graph_used_node_profile.pdf")
  fig.savefig(out / "graph_used_node_profile.png")
  plt.close(fig)


def main(argv):
  if len(argv) < 3:
    print(__doc__)
    return 2
  out = pathlib.Path(argv[1])
  out.mkdir(parents=True, exist_ok=True)
  runs = argv[2:]
  used, unused = collect(runs)
  label = "+".join(pathlib.Path(r).name for r in runs)
  figure(used, unused, out, label)
  stats = {
      "runs": [pathlib.Path(r).name for r in runs],
      "used_screens": len(used["visits"]),
      "unused_screens": len(unused["visits"]),
      "means": {f: {"used": (statistics.mean(used[f]) if used[f] else 0.0),
                    "unused": (statistics.mean(unused[f]) if unused[f] else 0.0)}
                for f, _ in FIELDS},
  }
  (out / "graph_used_node_profile.json").write_text(
      json.dumps(stats, indent=1), encoding="utf-8")
  print(f"used {stats['used_screens']}  unused {stats['unused_screens']}")
  for f, lab in FIELDS:
    a, b = stats["means"][f]["unused"], stats["means"][f]["used"]
    print(f"  {lab:<34} 被利用 {b:>6.2f}  未被利用 {a:>6.2f}  "
          f"{(b/a if a else float('nan')):>5.1f}×")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
