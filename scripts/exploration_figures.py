"""Figures for the exploration claim, from exploration_evidence.py's scans.

  python scripts/exploration_figures.py EVIDENCE_DIR [RUN_DIR_FOR_PER_STEP]

Sized and styled for a two-column ACM/IEEE paper: 3.3in single column, 7in
double, vector output, no colour-only encodings.
"""
import json
import pathlib
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
})
# Okabe-Ito: distinguishable in greyscale and for colour-vision deficiency.
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MARKS = ["o", "s", "^", "D", "v", "P"]


def fig_graph_growth(scans, out):
  """How much of the app space the graph has seen, task by task."""
  fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
  for i, s in enumerate(scans):
    g = s["graph_growth"]
    if not g:
      continue
    x = [r["order"] for r in g]
    axes[0].plot(x, [r["cumulative_nodes"] for r in g], color=C[i % len(C)],
                 lw=1.2, label=s["run"])
    axes[1].plot(x, [r["cumulative_edges"] for r in g], color=C[i % len(C)],
                 lw=1.2, label=s["run"])
  for ax, name in zip(axes, ("distinct screens", "distinct transitions")):
    ax.set_xlabel("task index (order run)")
    ax.set_ylabel(f"cumulative {name}")
  axes[0].legend(frameon=False, loc="upper left")
  fig.savefig(out / "fig_graph_growth_across_tasks.pdf")
  fig.savefig(out / "fig_graph_growth_across_tasks.png")
  plt.close(fig)


def fig_per_task_size(scans, out):
  """Per-task graph size: what one episode alone can build."""
  fig, ax = plt.subplots(figsize=(3.3, 2.2))
  for i, s in enumerate(scans):
    g = s["graph_growth"]
    if not g:
      continue
    nodes = sorted(r["task_nodes"] for r in g)
    ax.plot(nodes, [k / len(nodes) for k in range(1, len(nodes) + 1)],
            color=C[i % len(C)], lw=1.2, label=s["run"])
  ax.set_xlabel("screens in one episode's graph")
  ax.set_ylabel("CDF over tasks")
  ax.legend(frameon=False, loc="lower right")
  fig.savefig(out / "fig_per_task_graph_size_cdf.pdf")
  fig.savefig(out / "fig_per_task_graph_size_cdf.png")
  plt.close(fig)


def fig_rank_vs_chance(scans, out):
  """The exploration claim in one figure.

  For every round where the control the model went on to press was among the
  candidates, where did the ranking put it? A ranker no better than chance
  lands at (n+1)/2.
  """
  fig, ax = plt.subplots(figsize=(3.3, 2.4))
  labels, ranked, chance = [], [], []
  for s in scans:
    hits = [h for h in s["probe_rank_hits"] if h.get("rank")]
    if not hits:
      continue
    labels.append(s["run"])
    ranked.append(statistics.mean(h["rank"] for h in hits))
    chance.append(statistics.mean((h["candidates"] + 1) / 2 for h in hits))
  x = range(len(labels))
  ax.bar([i - 0.2 for i in x], chance, width=0.38, color="#BBBBBB",
         edgecolor="black", lw=0.4, label="uniform ranking")
  ax.bar([i + 0.2 for i in x], ranked, width=0.38, color=C[0],
         edgecolor="black", lw=0.4, label="graph-informed ranking")
  for i, (r, c) in enumerate(zip(ranked, chance)):
    ax.text(i + 0.2, r, f"{c/r:.0f}×", ha="center", va="bottom", fontsize=7)
  ax.set_xticks(list(x))
  ax.set_xticklabels(labels)
  ax.set_ylabel("rank of the model's next control\n(lower is better)")
  ax.legend(frameon=False)
  fig.savefig(out / "fig_rank_vs_chance.pdf")
  fig.savefig(out / "fig_rank_vs_chance.png")
  plt.close(fig)


def fig_outcome(summaries, out):
  """Success and step cost side by side, which is the trade the design makes."""
  fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
  names = [s["run"] for s in summaries]
  x = range(len(names))
  axes[0].bar(x, [s["success_rate"] * 100 for s in summaries], color=C[0],
              edgecolor="black", lw=0.4)
  axes[0].set_ylabel("success rate (%)")
  axes[1].bar(x, [s["mean_steps_success"] for s in summaries], color=C[1],
              edgecolor="black", lw=0.4)
  axes[1].set_ylabel("steps per successful task")
  for ax in axes:
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right")
  fig.savefig(out / "fig_outcome.pdf")
  fig.savefig(out / "fig_outcome.png")
  plt.close(fig)


def fig_per_episode_growth(run_dir, out, top=5):
  """Inside one episode: how fast the graph saturates.

  Node records carry the wall-clock time they were first upserted, so the
  build-up can be recovered from a finished run without instrumenting the
  loop: normalise each node's timestamp to the episode's own span and the
  curve is "fraction of the episode elapsed" against "screens known". A curve
  that flattens early is an app the agent finished mapping; one that climbs to
  the end is an episode that never stopped meeting new screens - which is
  exactly the regime where a probe cannot pay for itself.
  """
  rows = []
  for graph in pathlib.Path(run_dir).glob("*/progressive_belief_graph.json"):
    try:
      g = json.load(graph.open())
    except ValueError:
      continue
    stamps = sorted(float(n["timestamp"]) for n in g.get("nodes", [])
                    if isinstance(n.get("timestamp"), (int, float)))
    if len(stamps) >= 6:
      rows.append((len(stamps), graph.parent.name, stamps))
  rows.sort(reverse=True)
  if not rows:
    return
  fig, ax = plt.subplots(figsize=(3.3, 2.3))
  for i, (_, name, stamps) in enumerate(rows[:top]):
    span = stamps[-1] - stamps[0]
    if span <= 0:
      continue
    x = [(t - stamps[0]) / span for t in stamps]
    ax.step(x, range(1, len(stamps) + 1), where="post",
            color=C[i % len(C)], lw=1.2, label=name[:24])
  ax.set_xlabel("fraction of the episode elapsed")
  ax.set_ylabel("screens in the graph")
  ax.legend(frameon=False, fontsize=6, loc="upper left")
  fig.savefig(out / "fig_per_episode_growth.pdf")
  fig.savefig(out / "fig_per_episode_growth.png")
  plt.close(fig)


def fig_probe_cost(scans, out):
  """What exploration costs: rounds that came back, and rounds that did not."""
  fig, ax = plt.subplots(figsize=(3.3, 2.3))
  names, restored, failed = [], [], []
  for s in scans:
    outcomes = s["probe_outcomes"]
    if isinstance(outcomes, list):        # json round-trip of a Counter
      outcomes = dict(outcomes)
    total = sum(outcomes.values())
    if not total:
      continue
    names.append(s["run"])
    ok = outcomes.get("RESTORED", 0)
    restored.append(ok / total * 100)
    failed.append((total - ok) / total * 100)
  x = range(len(names))
  ax.bar(x, restored, color=C[2], edgecolor="black", lw=0.4, label="restored")
  ax.bar(x, failed, bottom=restored, color=C[1], edgecolor="black", lw=0.4,
         label="not restored")
  ax.set_xticks(list(x))
  ax.set_xticklabels(names)
  ax.set_ylabel("exploration rounds (%)")
  ax.set_ylim(0, 100)
  ax.legend(frameon=False, loc="lower right")
  fig.savefig(out / "fig_probe_recovery.pdf")
  fig.savefig(out / "fig_probe_recovery.png")
  plt.close(fig)


def main(argv):
  if len(argv) < 2:
    print(__doc__)
    return 2
  ev = pathlib.Path(argv[1])
  out = ev / "figures"
  out.mkdir(parents=True, exist_ok=True)
  scans = [json.load(p.open()) for p in sorted(ev.glob("*_scan.json"))]
  summaries = json.load((ev / "summary.json").open())
  fig_graph_growth(scans, out)
  fig_per_task_size(scans, out)
  fig_rank_vs_chance(scans, out)
  fig_outcome(summaries, out)
  fig_probe_cost(scans, out)
  if len(argv) > 2:
    fig_per_episode_growth(argv[2], out)
  print(f"figures -> {out}")
  for p in sorted(out.glob("*.png")):
    print("  ", p.name)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
