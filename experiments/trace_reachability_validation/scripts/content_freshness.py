"""Does online exploration keep the stored UI observations close to the live UI? (motivation, Fig. 4c)

For every model step of e2e116n FULL45 whose layout (package, activity, layout_sig) was already recorded by an
earlier task, compare the live screen's content with the stored observations of the same layout:
  history       captures from earlier tasks (model steps and probes)
  + exploration history plus this task's probe destinations captured before the step
Similarity = max content Jaccard over the stored captures. Content = list-row texts, editable-field values and
checkbox states (rgs_ui.UIState.content, the part excluded from the state key).
Also counts, over all probes, how often they reach a layout no earlier task recorded, and how often a probe of an
already recorded layout shows content different from every stored capture (earlier tasks vs. same task).

Inputs: graph_snapshots/e2e116n/FULL45/graph.jsonl, raw/e2e116n/FULL45/<task>/p0/a11y/*.json.gz,
        reports/e2e116n_task_order.txt
Output: figures/motivation/content_freshness.{pdf,png}, figures/motivation/content_freshness.json
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

ROOT = HERE.parent
TAG, ARM = "e2e116n", "FULL45"
RAW = ROOT / "raw" / TAG / ARM
OUT = ROOT / "figures" / "motivation"
_CACHE = {}


def content(task, cid):
  key = (task, cid)
  if key not in _CACHE:
    try:
      c = C.load_ui(str(RAW / task / "p0"), cid).content()
      _CACHE[key] = frozenset(["L:" + x for x in c["list"]] + ["E:" + x for x in c["edits"]] +
                              ["K:" + x for x in c["checks"]])
    except FileNotFoundError:
      _CACHE[key] = None
  return _CACHE[key]


def jac(a, b):
  if not a and not b:
    return 1.0
  return len(a & b) / len(a | b)


def lkey(s):
  return (s["package"], s["activity"], s["layout_sig"])


def main():
  order = open(ROOT / "reports" / f"{TAG}_task_order.txt").read().strip().split(",")
  rank = {t: i for i, t in enumerate(order)}
  obs = [json.loads(l) for l in open(ROOT / "graph_snapshots" / TAG / ARM / "graph.jsonl")]
  for o in obs:
    o["task"] = o["instance"].split("/")[0]
  obs.sort(key=lambda o: (rank.get(o["task"], 1e9), o["t"]))

  # stored captures: (layout, task rank, time, origin, content)
  stored = []
  for o in obs:
    for side in ("src", "dst"):
      s, cid = o.get(side), o.get(side + "_cid")
      if not s or not cid or "launcher" in (s.get("package") or ""):
        continue
      if side == "src" and o["origin"] == "explore":
        continue                                  # a probe's start screen is the model's screen, counted already
      c = content(o["task"], cid)
      if c is not None:
        stored.append((lkey(s), rank[o["task"]], o["t"], o["origin"] if side == "dst" else "agent", c))
  by_layout = collections.defaultdict(list)
  for rec in stored:
    by_layout[rec[0]].append(rec)

  # (1) model steps on layouts recorded by earlier tasks
  rows = []
  for o in obs:
    if o["origin"] != "agent" or not o.get("src_cid") or "launcher" in o["src"]["package"]:
      continue
    live = content(o["task"], o["src_cid"])
    if live is None:
      continue
    L, r, t = lkey(o["src"]), rank[o["task"]], o["t"]
    hist = [rec[4] for rec in by_layout[L] if rec[1] < r]
    if not hist:
      continue
    expl = [rec[4] for rec in by_layout[L] if rec[1] == r and rec[3] == "explore" and rec[2] < t]
    own = [rec[4] for rec in by_layout[L] if rec[1] == r and rec[3] == "agent" and rec[2] < t]
    h = max(jac(live, c) for c in hist)
    ho = max([h] + [jac(live, c) for c in own])           # + this task's earlier model steps
    hoe = max([ho] + [jac(live, c) for c in expl])        # + this task's earlier probes
    rows.append({"task": o["task"], "step": o["step"], "hist": h, "hist_own": ho, "hist_own_expl": hoe,
                 "hist_expl": max([h] + [jac(live, c) for c in expl]), "explored_before": bool(expl),
                 "first_visit": not own})

  # (2) what probes bring back
  probes = [o for o in obs if o["origin"] == "explore" and o.get("dst") and o.get("dst_cid")]
  new_layout, known_prev, diff_prev, known_same, diff_same = 0, 0, 0, 0, 0
  for o in probes:
    c = content(o["task"], o["dst_cid"])
    if c is None:
      continue
    L, r, t = lkey(o["dst"]), rank[o["task"]], o["t"]
    prev = [rec[4] for rec in by_layout[L] if rec[1] < r]
    same = [rec[4] for rec in by_layout[L] if rec[1] == r and rec[2] < t]
    if not prev:
      new_layout += 1
    else:
      known_prev += 1
      diff_prev += all(x != c for x in prev)
    if same:
      known_same += 1
      diff_same += all(x != c for x in same)
  n_probe = len(probes)

  H = np.array([x["hist"] for x in rows])
  HE = np.array([x["hist_expl"] for x in rows])
  sub = [x for x in rows if x["explored_before"]]
  first = [x for x in rows if x["first_visit"]]                     # model's first visit of the layout in its task
  fsub = [x for x in first if x["explored_before"]]
  summary = {
      "model_steps_on_recorded_layouts": len(rows),
      "steps_whose_layout_was_probed_earlier_in_task": len(sub),
      "mean_sim_history": round(float(H.mean()), 3), "mean_sim_history_plus_exploration": round(float(HE.mean()), 3),
      "subset_mean_sim_history": round(float(np.mean([x["hist"] for x in sub])), 3) if sub else None,
      "subset_mean_sim_history_plus_exploration": round(float(np.mean([x["hist_expl"] for x in sub])), 3) if sub else None,
      "subset_mean_sim_history_own": round(float(np.mean([x["hist_own"] for x in sub])), 3) if sub else None,
      "subset_mean_sim_history_own_expl": round(float(np.mean([x["hist_own_expl"] for x in sub])), 3) if sub else None,
      "subset_exact_match_history_own": sum(x["hist_own"] == 1.0 for x in sub),
      "subset_exact_match_history_own_expl": sum(x["hist_own_expl"] == 1.0 for x in sub),
      "subset_tasks": len({x["task"] for x in sub}),
      "subset_exact_match_history": sum(x["hist"] == 1.0 for x in sub),
      "subset_exact_match_history_plus_exploration": sum(x["hist_expl"] == 1.0 for x in sub),
      "first_visits": len(first), "first_visits_probed_before": len(fsub),
      "first_probed_mean_hist": round(float(np.mean([x["hist"] for x in fsub])), 3) if fsub else None,
      "first_probed_mean_hist_expl": round(float(np.mean([x["hist_expl"] for x in fsub])), 3) if fsub else None,
      "first_probed_exact_hist": sum(x["hist"] == 1.0 for x in fsub),
      "first_probed_exact_hist_expl": sum(x["hist_expl"] == 1.0 for x in fsub),
      "first_probed_tasks": len({x["task"] for x in fsub}),
      "first_unprobed_mean_hist": round(float(np.mean([x["hist"] for x in first if not x["explored_before"]])), 3),
      "probes": n_probe, "probes_new_layout": new_layout,
      "probes_known_layout_earlier_tasks": known_prev, "of_which_new_content": diff_prev,
      "probes_known_layout_same_task": known_same, "of_which_new_content_same_task": diff_same,
  }
  OUT.mkdir(parents=True, exist_ok=True)
  json.dump({"summary": summary, "steps": rows}, open(OUT / "content_freshness.json", "w"), indent=1)
  print(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()


def plot():
  """Motivation figure: (c) live content vs. stored records of the same layout; (d) what probes bring back."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  d = json.load(open(OUT / "content_freshness.json"))
  S, rows = d["summary"], d["steps"]
  plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                       "axes.spines.right": False, "pdf.fonttype": 42})
  GRAY, PINK = "#b8bec6", "#d97fb4"
  fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.1), gridspec_kw={"width_ratios": [1.15, 1]})
  v = np.sort([x["hist"] for x in rows])
  a.step(np.concatenate([[0], v]), np.concatenate([[0], np.arange(1, len(v) + 1) / len(v)]), where="post",
         color=PINK, lw=2)
  a.axvline(1.0, color=GRAY, lw=2)
  exact = float(np.mean(v == 1.0))
  a.text(0.985, 1.0, f"structure: same layout (all {len(v)} steps)", ha="right", va="top", fontsize=8, color="#555")
  a.annotate(f"content identical to a stored\nrecord: {exact:.0%} of steps", xy=(0.995, 1 - exact),
             xytext=(0.12, 0.62), fontsize=8, arrowprops={"arrowstyle": "->", "color": "#555"})
  a.set_xlim(0, 1.03)
  a.set_ylim(0, 1.02)
  a.set_xlabel("Content Jaccard similarity to the most\nsimilar record from earlier tasks")
  a.set_ylabel("CDF of model steps")
  a.set_title("(c) Live UI vs. stored UI (same layout)", fontsize=9, loc="left")
  labels = ["layout recorded by\nearlier tasks", "layout recorded earlier\nin the same task"]
  n1, k1 = S["probes_known_layout_earlier_tasks"], S["of_which_new_content"]
  n2, k2 = S["probes_known_layout_same_task"], S["of_which_new_content_same_task"]
  vals = [k1 / n1, k2 / n2]
  b.bar([0, 1], vals, width=0.5, color=[PINK, GRAY])
  for x, (val, k, n) in enumerate(zip(vals, (k1, k2), (n1, n2))):
    b.text(x, val + 0.02, f"{val:.0%}\n({k}/{n})", ha="center", va="bottom", fontsize=8)
  b.set_xticks([0, 1])
  b.set_xticklabels(labels, fontsize=8)
  b.set_ylim(0, 0.8)
  b.set_ylabel("Probes showing content\nnot in any stored record")
  b.set_title(f"(d) What probes bring back ({S['probes']} probes)", fontsize=9, loc="left")
  b.text(1.45, 0.72, f"{S['probes_new_layout'] / S['probes']:.0%} reach a layout\nno earlier task recorded",
         ha="right", va="top", fontsize=8, color="#555")
  fig.tight_layout()
  for ext in ("pdf", "png"):
    fig.savefig(OUT / f"content_freshness.{ext}", dpi=300, bbox_inches="tight")
  print("saved", OUT / "content_freshness.png")


if __name__ == "__main__" and "--plot" in sys.argv:
  plot()
