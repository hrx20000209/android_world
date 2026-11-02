"""Overall-performance figure: graph growth, and graph additions vs. E2E speedup by task position (default order).

(a) Mix-Trajectory Graph size over the 116 AndroidWorld tasks in the default order (e2e116n FULL45): states,
    edges, and edges first observed by exploration.
(b) Tasks grouped by their position within their app (how many earlier tasks of the same app ran before them):
    bars  = new states added to the graph per task (left axis),
    line  = E2E speedup vs. the baseline, mean(BASE e2e) / mean(FULL45 e2e) over all tasks in the group (right axis),
    text  = success rate of the baseline / MobileExplorer in the group.

Speedup uses all tasks of a group, like the 151.9 s -> 135.2 s headline, so a group is not reduced to the few tasks
both systems happened to solve. The app grouping is the one used by understand_perf.py.

Inputs: reports/e2e116n_runs.jsonl, reports/e2e116n_task_order.txt, graph_snapshots/e2e116n/FULL45/graph.jsonl.
Without the graph snapshot, panel (a) and the bars fall back to figures/overall/preview_readoff.json (values read
off the order-microbenchmark figure) and the output is marked PREVIEW.

Output: figures/overall/graph_position.{pdf,png}
"""
import collections
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "overall"
TAG = "e2e116n"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
C_BASE, C_ME, C3, C4 = "#8a8f96", "#1d6f78", "#7fb8bd", "#d9a441"
BUCKETS = ("1st", "2nd–3rd", "4th+")


def app(t):
  m = re.match(r"(Simple[A-Z][a-z]+|System|Expense|Markor|Recipe|Files|Contacts|Notes|Tasks|Retro|Vlc|OsmAnd|Browser|"
               r"Camera|Clock|AudioRecorder|OpenApp|SaveCopy|Sports|Turn)", t)
  return m.group(1) if m else t


def bucket(p):
  return BUCKETS[0] if p == 0 else (BUCKETS[1] if p <= 2 else BUCKETS[2])


def load_runs():
  rows = {}
  for line in open(ROOT / "reports" / f"{TAG}_runs.jsonl"):
    r = json.loads(line)
    if r.get("error"):
      rows.pop((r["task"], r["arm"]), None)
    else:
      rows[(r["task"], r["arm"])] = r          # a re-run replaces the earlier record
  order = open(ROOT / "reports" / f"{TAG}_task_order.txt").read().strip().split(",")
  tasks = [t for t in order if (t, "BASE") in rows and (t, "FULL45") in rows]
  pos, seen = {}, collections.Counter()
  for t in order:
    pos[t] = seen[app(t)]
    seen[app(t)] += 1
  return rows, order, tasks, pos


def graph_stats(order, pos):
  """Graph size after each task and states added per task, from the FULL45 graph snapshot."""
  G = ROOT / "graph_snapshots" / TAG / "FULL45" / "graph.jsonl"
  if not G.exists():
    return None
  nodes, edges, explore_edges = set(), set(), set()
  added = collections.Counter()
  after = {}
  for line in open(G):
    e = json.loads(line)
    t = e.get("instance", "").split("/")[0]
    src = ((e.get("src") or {}).get("activity"), (e.get("src") or {}).get("layout_sig"))
    dst = ((e.get("dst") or {}).get("activity"), (e.get("dst") or {}).get("layout_sig"))
    for n in (src, dst):
      if n not in nodes:
        nodes.add(n)
        added[t] += 1
    key = (src, e.get("slot_id"), dst)
    if key not in edges:
      edges.add(key)
      if e.get("origin") == "explore":           # first observed by an exploration probe
        explore_edges.add(key)
    after[t] = (len(nodes), len(edges), len(explore_edges))
  xs, states, edg, exp = [0], [0], [0], [0]
  last = (0, 0, 0)
  for i, t in enumerate(order):
    last = after.get(t, last)
    xs.append(i + 1)
    states.append(last[0])
    edg.append(last[1])
    exp.append(last[2])
  per_bucket = collections.defaultdict(list)
  for t in order:
    per_bucket[bucket(pos[t])].append(added.get(t, 0))
  return {"x": xs, "states": states, "edges": edg, "explore": exp,
          "new_states_by_pos": [st.mean(per_bucket[b]) for b in BUCKETS], "preview": False}


def preview_stats():
  d = json.load(open(OUT / "preview_readoff.json"))
  return {"x": list(range(len(d["states"]))), "states": d["states"], "edges": d["edges"], "explore": d["explore"],
          "new_states_by_pos": d["new_states_by_pos"], "preview": True}


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  rows, order, tasks, pos = load_runs()
  g = graph_stats(order, pos) or preview_stats()

  by = {}
  for b in BUCKETS:
    T = [t for t in tasks if bucket(pos[t]) == b]
    eb = st.mean(rows[(t, "BASE")]["e2e_s"] for t in T)
    ef = st.mean(rows[(t, "FULL45")]["e2e_s"] for t in T)
    by[b] = {"n": len(T), "speedup": eb / ef,
             "sr_base": sum(rows[(t, "BASE")]["success"] for t in T) / len(T),
             "sr_me": sum(rows[(t, "FULL45")]["success"] for t in T) / len(T)}
  overall = {"sr_base": sum(rows[(t, "BASE")]["success"] for t in tasks) / len(tasks),
             "sr_me": sum(rows[(t, "FULL45")]["success"] for t in tasks) / len(tasks),
             "speedup": st.mean(rows[(t, "BASE")]["e2e_s"] for t in tasks) / st.mean(rows[(t, "FULL45")]["e2e_s"] for t in tasks)}

  fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.15), gridspec_kw={"width_ratios": [1.05, 1]})

  # (a) graph growth
  a.plot(g["x"], g["edges"], color=C_ME, lw=1.6, ls=(0, (4, 2)))
  a.plot(g["x"], g["states"], color=C_ME, lw=1.8)
  a.plot(g["x"], g["explore"], color=C4, lw=1.8, ls=(0, (1, 1.4)))
  n = len(order)
  for y, label in ((g["edges"][-1], "edges"), (g["states"][-1], "states"), (g["explore"][-1], "edges from\nexploration")):
    a.text(n + 3, y, label, va="center", fontsize=7, color="#444")
  a.set_xlim(0, n + 30)
  a.set_xticks([0, 29, 58, 87, 116])
  a.set_ylim(bottom=0)
  a.set_xlabel("Tasks executed")
  a.set_ylabel("Graph size")
  a.set_title("(a) Graph growth", fontsize=8, loc="left")

  # (b) graph additions (bars) and E2E speedup (line) by task position
  xs = range(len(BUCKETS))
  ns = g["new_states_by_pos"]
  a2 = b.twinx()
  b.bar(xs, ns, width=0.55, color=C3, edgecolor="white", linewidth=1, zorder=2)
  for x, v in zip(xs, ns):
    b.text(x, v / 2, f"{v:.1f}", ha="center", va="center", fontsize=7, color="#1f3f42", zorder=3)
  b.set_ylim(0, 10)                                   # bars in the lower half, speedup line above them
  b.set_ylabel("New states per task", color="#2c6f75")
  b.tick_params(axis="y", colors="#2c6f75")
  sp = [by[k]["speedup"] for k in BUCKETS]
  a2.axhline(1.0, color="#9aa0a6", lw=0.6, ls=(0, (3, 2)), zorder=1)
  a2.plot(xs, sp, color="#333333", marker="D", ms=4.5, lw=1.4, zorder=4)
  for x, v in zip(xs, sp):
    a2.text(x, v + 0.02, f"{v:.2f}×", ha="center", va="bottom", fontsize=7, color="#333333")
  a2.set_ylim(0.7, 1.25)
  a2.spines["right"].set_visible(True)
  a2.set_ylabel("E2E speedup vs. baseline (×)")
  b.set_xticks(list(xs))
  b.set_xticklabels([f"{k} ({by[k]['n']})\nSR {by[k]['sr_base']:.0%} | {by[k]['sr_me']:.0%}" for k in BUCKETS],
                    fontsize=7)
  b.set_xlabel("Task position within its app")
  b.set_title("(b) Graph additions and speedup", fontsize=8, loc="left")
  handles = [plt.Rectangle((0, 0), 1, 1, color=C3), plt.Line2D([], [], color="#333333", marker="D", ms=4, lw=1.4)]
  b.legend(handles, ["New states per task", "E2E speedup"], fontsize=6.5, frameon=False, loc="lower right",
           bbox_to_anchor=(1.0, 0.24))
  if g["preview"]:
    fig.text(0.5, 0.005, "PREVIEW: panel (a) and the bars are read off the order-microbenchmark figure; "
             "rerun with graph_snapshots/e2e116n for exact values", ha="center", va="bottom", fontsize=6, color="#c0392b")
  fig.tight_layout(rect=(0, 0.04 if g["preview"] else 0, 1, 1))
  for ext in ("pdf", "png"):
    fig.savefig(OUT / f"graph_position.{ext}", bbox_inches="tight", dpi=300)
  summary = {"by_position": by, "overall": overall, "new_states_by_pos": dict(zip(BUCKETS, ns)),
             "graph_final": {"states": g["states"][-1], "edges": g["edges"][-1], "explore_edges": g["explore"][-1]},
             "preview": g["preview"]}
  json.dump(summary, open(OUT / "graph_position_summary.json", "w"), indent=1, ensure_ascii=False)
  print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
  main()
