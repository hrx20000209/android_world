"""The busiest screens of an episode, shown as the screens they actually were.

  python scripts/graph_hub_screens.py TASK_DIR OUT_DIR [TOP]

Needs a run made with --dump_graph_steps, which writes one capture per graph
node the first time the episode stands on it.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 7.5, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def main(argv):
  if len(argv) < 3:
    print(__doc__)
    return 2
  task_dir = pathlib.Path(argv[1])
  out = pathlib.Path(argv[2])
  out.mkdir(parents=True, exist_ok=True)
  top = int(argv[3]) if len(argv) > 3 else 5
  graph = json.load((task_dir / "progressive_belief_graph.json").open())
  shots = task_dir / "screens"
  edges = graph.get("edges", [])
  out_degree = {}
  for e in edges:
    out_degree[e.get("src_node")] = out_degree.get(e.get("src_node"), 0) + 1
  nodes = sorted(graph.get("nodes", []),
                 key=lambda n: -int(n.get("visit_count") or 0))
  panels = []
  for node in nodes:
    path = shots / f"{node.get('node_id','')[:12]}.png"
    if path.exists():
      panels.append((node, path))
    if len(panels) >= top:
      break
  if not panels:
    print(f"no captures under {shots}")
    return 1
  fig, axes = plt.subplots(1, len(panels), figsize=(1.5 * len(panels), 3.4))
  if len(panels) == 1:
    axes = [axes]
  for ax, (node, path) in zip(axes, panels):
    ax.imshow(mpimg.imread(path))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
      side.set_linewidth(0.6)
    activity = str(node.get("activity", "")).split("/")[-1].lstrip(".")[:20]
    ax.set_title(f"{node.get('visit_count')} visits · {out_degree.get(node.get('node_id'),0)} exits\n{activity}",
                 fontsize=6.5)
  fig.suptitle(f"{task_dir.name}: the screens the episode kept returning to",
               y=1.0, fontsize=8.5)
  fig.tight_layout()
  stem = out / f"hub_screens_{task_dir.name}"
  fig.savefig(f"{stem}.pdf")
  fig.savefig(f"{stem}.png")
  plt.close(fig)
  print(f"{stem}.png")
  for node, _ in panels:
    print(f"   {node.get('visit_count'):>3} visits  "
          f"{out_degree.get(node.get('node_id'),0):>2} exits  {node.get('activity')}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
