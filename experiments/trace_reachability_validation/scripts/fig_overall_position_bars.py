"""New graph states per task (bars) and E2E latency speedup (line) by task position within its app.

Positions: 1st, 2nd, 3rd, 4th+ (the 3rd task has its own bar). Values are pooled over the three task orders of the
order microbenchmark, with the same definitions as order_microbench_REPORT:
  position  = number of earlier tasks of the same app in that order
  speedup   = mean(BASE e2e) / mean(FULL45 e2e) over the task instances both systems solved, pooled over orders
  new nodes = states first seen in the FULL45 graph during the task, averaged over all task instances, pooled

Inputs per order tag (ROOT = experiments/trace_reachability_validation):
  reports/<tag>_runs.jsonl, reports/<tag>_task_order.txt, graph_snapshots/<tag>/FULL45/graph.jsonl
The script also prints the 3-bucket values (1st, 2nd-3rd, 4th+): they should reproduce 5.6 / 2.6 / 1.5 and
1.11 / 1.19 / 1.24 from the report before the 4-bucket figure is used.
"""
import collections
import json
import os
import re
import statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  # noqa: E402

# ==========================================
# Configuration
# ==========================================
ROOT = Path(__file__).resolve().parents[1]
TAGS = ["e2e116n", "e2e116P", "e2e116V"]      # default, random, reversed within app
OUT = "./new_figure"
os.makedirs(OUT, exist_ok=True)
SHOW_N = False                                 # True: add the number of pooled tasks under each label

OUR_COLOR = "#66c2a5"
BASE_COLOR = "black"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 20,
    "axes.labelsize": 28,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 22,
    "axes.linewidth": 1.5,
})


# ==========================================
# Data: pooled over the task orders
# ==========================================
def app(t):
    m = re.match(r"(Simple[A-Z][a-z]+|System|Expense|Markor|Recipe|Files|Contacts|Notes|Tasks|Retro|Vlc|OsmAnd|"
                 r"Browser|Camera|Clock|AudioRecorder|OpenApp|SaveCopy|Sports|Turn)", t)
    return m.group(1) if m else t


def load_order(tag):
    runs, order_f = ROOT / "reports" / f"{tag}_runs.jsonl", ROOT / "reports" / f"{tag}_task_order.txt"
    if not runs.exists() or not order_f.exists():
        return None
    rows = {}
    for line in open(runs):
        r = json.loads(line)
        if r.get("error"):
            rows.pop((r["task"], r["arm"]), None)
        else:
            rows[(r["task"], r["arm"])] = r            # a re-run replaces the earlier record
    order = open(order_f).read().strip().split(",")
    pos, seen = {}, collections.Counter()
    for t in order:
        pos[t] = seen[app(t)]
        seen[app(t)] += 1
    new_nodes = None
    graph = ROOT / "graph_snapshots" / tag / "FULL45" / "graph.jsonl"
    if graph.exists():
        nodes, new_nodes = set(), collections.Counter()
        for line in open(graph):
            e = json.loads(line)
            t = e.get("instance", "").split("/")[0]
            for side in ("src", "dst"):
                n = ((e.get(side) or {}).get("activity"), (e.get(side) or {}).get("layout_sig"))
                if n not in nodes:
                    nodes.add(n)
                    new_nodes[t] += 1
    tasks = [t for t in order if (t, "BASE") in rows and (t, "FULL45") in rows]
    return {"rows": rows, "tasks": tasks, "pos": pos, "new_nodes": new_nodes}


def position_stats(orders, buckets, bucket_of):
    e2e = {b: ([], []) for b in buckets}
    nodes = {b: [] for b in buckets}
    for o in orders:
        for t in o["tasks"]:
            b = bucket_of(o["pos"][t])
            base, ours = o["rows"][(t, "BASE")], o["rows"][(t, "FULL45")]
            if base["success"] and ours["success"]:
                e2e[b][0].append(base["e2e_s"])
                e2e[b][1].append(ours["e2e_s"])
            if o["new_nodes"] is not None:
                nodes[b].append(o["new_nodes"].get(t, 0))
    speedup = np.array([st.mean(e2e[b][0]) / st.mean(e2e[b][1]) for b in buckets])
    new_states = np.array([st.mean(nodes[b]) if nodes[b] else np.nan for b in buckets])
    n_both = [len(e2e[b][0]) for b in buckets]
    return new_states, speedup, n_both


orders = {tag: load_order(tag) for tag in TAGS}
missing = [tag for tag, o in orders.items() if o is None]
orders = [o for o in orders.values() if o is not None]
if missing:
    print(f"WARNING: no runs for {missing}; pooling over {len(orders)} order(s) only")
if not orders:
    raise SystemExit("no order runs found under " + str(ROOT / "reports"))

check = position_stats(orders, ["1st", "2nd-3rd", "4th+"], lambda p: "1st" if p == 0 else ("2nd-3rd" if p <= 2 else "4th+"))
print("3-bucket check  new states", np.round(check[0], 1), " speedup", np.round(check[1], 2), " both-solved", check[2])

categories = ["1st", "2nd", "3rd", "4th+"]
x = np.arange(len(categories))
new_states, speedup, n_both = position_stats(orders, categories, lambda p: categories[min(p, 3)])
print("4-bucket        new states", np.round(new_states, 1), " speedup", np.round(speedup, 2), " both-solved", n_both)
if np.isnan(new_states).any():
    raise SystemExit("graph_snapshots/<tag>/FULL45/graph.jsonl missing: cannot compute new states per task")

# ==========================================
# Figure
# ==========================================
fig, ax1 = plt.subplots(
    figsize=(8, 6),
    dpi=300
)

# ==========================================
# Bars: New Nodes per Task
# ==========================================
bars = ax1.bar(
    x,
    new_states,
    width=0.48,
    color=OUR_COLOR,
    edgecolor="none",
    label="New Nodes per Task",
    zorder=3
)

ax1.set_ylabel(
    "New Nodes per Task",
    labelpad=10
)

ax1.set_ylim(0, 8)
ax1.yaxis.set_major_locator(
    MultipleLocator(1)
)

ax1.set_xticks(x)
ax1.set_xticklabels(
    [f"{c}\n(n={n})" for c, n in zip(categories, n_both)] if SHOW_N else categories
)

ax1.set_xlabel(
    "Task Position within Its App",
    labelpad=10
)

ax1.tick_params(
    axis="both",
    width=1.5,
    length=6
)

# ==========================================
# Line: End-to-End Speedup
# ==========================================
ax2 = ax1.twinx()

line, = ax2.plot(
    x,
    speedup,
    color=BASE_COLOR,
    marker="o",
    markersize=9,
    linewidth=2.8,
    markeredgecolor="white",
    markeredgewidth=1.2,
    label="End-to-End Latency Speedup",
    zorder=5
)

ax2.set_ylabel(
    "Latency Speedup (×)",
    labelpad=12
)

top = max(1.5, np.ceil(speedup.max() * 5) / 5)   # keep 1.5 unless a value is higher
ax2.set_ylim(0, top)
ax2.set_yticks(
    np.arange(0, top + 0.01, 0.2)
)

ax2.yaxis.set_major_formatter(
    FormatStrFormatter("%.1f")
)

ax2.tick_params(
    axis="y",
    width=1.5,
    length=6
)

# ==========================================
# Axes styling
# ==========================================
for ax in (ax1, ax2):
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

# ==========================================
# Legend
# ==========================================
fig.legend(
    [bars, line],
    [
        "New Nodes per Task",
        "End-to-End Latency Speedup"
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=1,
    frameon=True,
    fancybox=True,
    edgecolor="#d9d9d9",
    handlelength=2.5,
    columnspacing=2.0
)

# ==========================================
# Layout
# ==========================================
fig.subplots_adjust(
    left=0.13,
    right=0.84,
    bottom=0.21 if SHOW_N else 0.17,
    top=0.8
)

# ==========================================
# Save PNG only
# ==========================================
output_path = os.path.join(
    OUT,
    "overall_graph.png"
)

fig.savefig(
    output_path,
    dpi=300,
    facecolor="white"
)

plt.close(fig)

print(f"Saved: {output_path}")
