"""New graph states per task (bars) and E2E latency gain (line) by task position within its app.

Positions: 1st, 2nd-3rd, 4th+ (SPLIT_2ND_3RD = True gives the 3rd task its own bar). Values are pooled over the three
task orders of the order microbenchmark, with the same definitions as order_microbench_REPORT:
  position  = number of earlier tasks of the same app in that order
  speedup   = mean(BASE e2e) / mean(FULL45 e2e) over the task instances both systems solved, pooled over orders
  reduction = 1 - mean(FULL45 e2e) / mean(BASE e2e) = 1 - 1 / speedup (same tasks, same data)
  new nodes = states first seen in the FULL45 graph during the task, averaged over all task instances, pooled

RIGHT_METRIC selects the line: "reduction" (E2E latency reduction, %, axis from 0) or "speedup" (x, axis from 1.0,
the no-speedup level).

Inputs per order tag (ROOT = experiments/trace_reachability_validation):
  reports/<tag>_runs.jsonl, reports/<tag>_task_order.txt, graph_snapshots/<tag>/FULL45/graph.jsonl
The script also prints the 3-bucket values (1st, 2nd-3rd, 4th+): they should reproduce 5.6 / 2.6 / 1.5 and
1.11 / 1.19 / 1.24 (10% / 16% / 20%) from the report.
Error bars: 95% bootstrap confidence intervals over task instances (pooled over orders); for the line the
(BASE, FULL45) pairs are resampled together and the ratio of means is recomputed.
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
SHOW_N = False                                 # True: add the number of pooled tasks under each label
SPLIT_2ND_3RD = False                          # True: 1st, 2nd, 3rd, 4th+
ERROR_BARS = True                              # 95% bootstrap CIs over tasks
RIGHT_METRIC = "reduction"                     # "reduction" (%) or "speedup" (x, axis from 1.0)
LABEL_POINTS = True                            # print the value next to each point of the line
BOOT, SEED = 5000, 0

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


def _cluster_ci(groups, rng):
    """95% CI of sum(num) / sum(den), resampling whole tasks: groups = {task: (num, den)} summed over its instances.
    A task pooled from several orders is one cluster, since its instances share the task's difficulty."""
    num = np.array([v[0] for v in groups.values()], dtype=float)
    den = np.array([v[1] for v in groups.values()], dtype=float)
    idx = rng.integers(0, len(num), (BOOT, len(num)))
    return np.percentile(num[idx].sum(1) / den[idx].sum(1), [2.5, 97.5])


def position_stats(orders, buckets, bucket_of):
    e2e = {b: ([], []) for b in buckets}
    nodes = {b: [] for b in buckets}
    e2e_by_task = {b: collections.defaultdict(lambda: [0.0, 0.0]) for b in buckets}    # task -> [sum BASE, sum ours]
    nodes_by_task = {b: collections.defaultdict(lambda: [0.0, 0]) for b in buckets}    # task -> [sum nodes, count]
    for o in orders:
        for t in o["tasks"]:
            b = bucket_of(o["pos"][t])
            base, ours = o["rows"][(t, "BASE")], o["rows"][(t, "FULL45")]
            if base["success"] and ours["success"]:
                e2e[b][0].append(base["e2e_s"])
                e2e[b][1].append(ours["e2e_s"])
                e2e_by_task[b][t][0] += base["e2e_s"]
                e2e_by_task[b][t][1] += ours["e2e_s"]
            if o["new_nodes"] is not None:
                nodes[b].append(o["new_nodes"].get(t, 0))
                nodes_by_task[b][t][0] += o["new_nodes"].get(t, 0)
                nodes_by_task[b][t][1] += 1
    speedup = np.array([st.mean(e2e[b][0]) / st.mean(e2e[b][1]) for b in buckets])
    new_states = np.array([st.mean(nodes[b]) if nodes[b] else np.nan for b in buckets])
    n_both = [len(e2e[b][0]) for b in buckets]
    rng = np.random.default_rng(SEED)
    sp_ci, ns_ci = [], []
    for b in buckets:
        sp_ci.append(_cluster_ci(e2e_by_task[b], rng))                       # ratio of means, tasks resampled
        if nodes[b]:
            ns_ci.append(_cluster_ci(nodes_by_task[b], rng))                 # mean new nodes, tasks resampled
        else:
            ns_ci.append([np.nan, np.nan])
    return new_states, speedup, n_both, np.array(ns_ci), np.array(sp_ci), [len(nodes[b]) for b in buckets]


# ==========================================
# Figure
# ==========================================
def plot_figure(categories, new_states, ns_ci, speedup, sp_ci, n_both=None, path=None):
    """ns_ci / sp_ci: (len(categories), 2) arrays of CI bounds, or None to draw no error bars."""
    x = np.arange(len(categories))
    if RIGHT_METRIC == "reduction":
        line_vals = 100 * (1 - 1 / np.asarray(speedup))
        line_ci = None if sp_ci is None else 100 * (1 - 1 / np.asarray(sp_ci))   # monotone map of the bounds
        line_label, axis_label = "End-to-End Latency Reduction", "Latency Reduction (%)"
    else:
        line_vals, line_ci = np.asarray(speedup), None if sp_ci is None else np.asarray(sp_ci)
        line_label, axis_label = "End-to-End Latency Speedup", "Latency Speedup (×)"

    fig, ax1 = plt.subplots(
        figsize=(8, 6),
        dpi=300
    )

    # Bars: New Nodes per Task
    bars = ax1.bar(
        x,
        new_states,
        width=0.48,
        color=OUR_COLOR,
        edgecolor="none",
        label="New Nodes per Task",
        yerr=np.abs(np.asarray(ns_ci).T - new_states) if ns_ci is not None else None,
        capsize=8,
        error_kw={"elinewidth": 1.8, "capthick": 1.8, "ecolor": "#333333", "zorder": 4},
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
        [f"{c}\n(n={n})" for c, n in zip(categories, n_both)] if SHOW_N and n_both else categories
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

    # Line: End-to-End latency gain
    ax2 = ax1.twinx()
    line, = ax2.plot(
        x,
        line_vals,
        color=BASE_COLOR,
        marker="o",
        markersize=9,
        linewidth=2.8,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label=line_label,
        zorder=5
    )
    if line_ci is not None:
        ax2.errorbar(x, line_vals, yerr=np.abs(line_ci.T - line_vals), fmt="none", ecolor=BASE_COLOR,
                     elinewidth=1.8, capsize=8, capthick=1.8, zorder=6)
    hi = (line_ci[:, 1].max() if line_ci is not None else line_vals.max())
    if RIGHT_METRIC == "reduction":
        top = max(25, 5 * np.ceil((hi + 3) / 5))
        ax2.set_ylim(0, top)
        ax2.set_yticks(np.arange(0, top + 0.1, 5))
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        fmt = "{:.0f}%"
    else:
        lo = (line_ci[:, 0].min() if line_ci is not None else line_vals.min())
        bottom = 1.0 if lo >= 1.0 else np.floor(lo * 10) / 10     # 1.0 = no speedup
        top = max(bottom + 0.3, np.ceil((hi + 0.03) * 10) / 10)
        ax2.set_ylim(bottom, top)
        ax2.set_yticks(np.arange(bottom, top + 0.001, 0.1))
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        fmt = "{:.2f}×"
    if LABEL_POINTS:
        for xi, v in zip(x, line_vals):
            ax2.annotate(fmt.format(v), (xi, v), textcoords="offset points", xytext=(0, 14), ha="center",
                         va="bottom", fontsize=20, zorder=7,
                         bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})
    ax2.set_ylabel(
        axis_label,
        labelpad=12
    )
    ax2.tick_params(
        axis="y",
        width=1.5,
        length=6
    )

    # Axes styling
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Legend
    fig.legend(
        [bars, line],
        [
            "New Nodes per Task",
            line_label
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

    # Layout
    fig.subplots_adjust(
        left=0.13,
        right=0.84,
        bottom=0.21 if SHOW_N else 0.17,
        top=0.8
    )

    # Save PNG only
    os.makedirs(OUT, exist_ok=True)
    output_path = path or os.path.join(OUT, "overall_graph.png")
    fig.savefig(
        output_path,
        dpi=300,
        facecolor="white"
    )
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    orders = {tag: load_order(tag) for tag in TAGS}
    missing = [tag for tag, o in orders.items() if o is None]
    orders = [o for o in orders.values() if o is not None]
    if missing:
        print(f"WARNING: no runs for {missing}; pooling over {len(orders)} order(s) only")
    if not orders:
        raise SystemExit("no order runs found under " + str(ROOT / "reports"))

    check = position_stats(orders, ["1st", "2nd-3rd", "4th+"],
                           lambda p: "1st" if p == 0 else ("2nd-3rd" if p <= 2 else "4th+"))
    print("3-bucket check  new states", np.round(check[0], 1), " speedup", np.round(check[1], 2),
          " reduction %", np.round(100 * (1 - 1 / check[1])), " both-solved", check[2])

    if SPLIT_2ND_3RD:
        categories = ["1st", "2nd", "3rd", "4th+"]
        bucket_of = lambda p: categories[min(p, 3)]  # noqa: E731
    else:
        categories = ["1st", "2nd–3rd", "4th+"]
        bucket_of = lambda p: categories[0] if p == 0 else (categories[1] if p <= 2 else categories[2])  # noqa: E731
    new_states, speedup, n_both, ns_ci, sp_ci, n_tasks = position_stats(orders, categories, bucket_of)
    for c, v, lo_hi, n in zip(categories, speedup, sp_ci, n_both):
        print(f"  {c:8s} speedup {v:.2f}x  95% CI [{lo_hi[0]:.2f}, {lo_hi[1]:.2f}]   "
              f"reduction {100 * (1 - 1 / v):.0f}%  [{100 * (1 - 1 / lo_hi[0]):.0f}%, {100 * (1 - 1 / lo_hi[1]):.0f}%]"
              f"   both-solved {n}")
    for c, v, lo_hi, n in zip(categories, new_states, ns_ci, n_tasks):
        print(f"  {c:8s} new states {v:.2f}  95% CI [{lo_hi[0]:.2f}, {lo_hi[1]:.2f}]  tasks {n}")
    if np.isnan(new_states).any():
        raise SystemExit("graph_snapshots/<tag>/FULL45/graph.jsonl missing: cannot compute new states per task")
    plot_figure(categories, new_states, ns_ci if ERROR_BARS else None, speedup, sp_ci if ERROR_BARS else None,
                n_both)


if __name__ == "__main__":
    main()
