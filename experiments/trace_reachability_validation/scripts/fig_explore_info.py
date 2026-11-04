"""Motivation (Fig. 4c/d), e2e116n FULL45: how much UI knowledge exploration adds.

(c) distinct transitions (source layout, control, destination layout) recorded over the 116 tasks, from model
    execution only vs. model execution + exploration probes.
(d) per UI action: probability that the action reaches a layout / records a transition not yet in memory,
    model step vs. exploration probe (memory = everything recorded before the action, either source).
Layout = (package, activity, layout_sig); launcher screens and in-page actions excluded.
"""
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "motivation"
order = open(ROOT / "reports" / "e2e116n_task_order.txt").read().strip().split(",")
rank = {t: i for i, t in enumerate(order)}
obs = [json.loads(l) for l in open(ROOT / "graph_snapshots" / "e2e116n" / "FULL45" / "graph.jsonl")]
for o in obs:
  o["task"] = o["instance"].split("/")[0]
obs.sort(key=lambda o: (rank[o["task"]], o["t"]))
key = lambda s: (s["package"], s["activity"], s["layout_sig"])  # noqa: E731
app = lambda s: bool(s) and "launcher" not in s["package"]  # noqa: E731

E_agent, E_all, curve = set(), set(), {}
seenL, seenE, cnt = set(), set(), collections.Counter()
for o in obs:
  if not app(o.get("src")):
    continue
  if o["origin"] == "agent" and o["step"] == 0:
    seenL.add(key(o["src"]))
  org = o["origin"]
  cnt[org] += 1
  if o.get("dst") and app(o["dst"]):
    L, S = key(o["dst"]), key(o["src"])
    if L not in seenL:
      cnt[org + "_newL"] += 1
      seenL.add(L)
    if L != S:
      e = (S, o["slot_id"], L)
      if e not in seenE:
        cnt[org + "_newE"] += 1
        seenE.add(e)
      E_all.add(e)
      if org == "agent":
        E_agent.add(e)
  curve[rank[o["task"]]] = (len(E_agent), len(E_all))

summary = {"transitions_model_only": len(E_agent), "transitions_with_exploration": len(E_all),
           "model_steps": cnt["agent"], "probes": cnt["explore"],
           "new_layout_per_model_step": cnt["agent_newL"] / cnt["agent"],
           "new_layout_per_probe": cnt["explore_newL"] / cnt["explore"],
           "new_transition_per_model_step": cnt["agent_newE"] / cnt["agent"],
           "new_transition_per_probe": cnt["explore_newE"] / cnt["explore"]}
json.dump(summary, open(OUT / "explore_info.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
GRAY, PINK = "#8a8f98", "#d97fb4"
fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.15), gridspec_kw={"width_ratios": [1.15, 1]})
xs = sorted(curve)
X = [x + 1 for x in xs]
ya, yb = [curve[x][0] for x in xs], [curve[x][1] for x in xs]
a.fill_between(X, ya, yb, color=PINK, alpha=0.25, lw=0)
a.plot(X, yb, color=PINK, lw=2, label="model + exploration")
a.plot(X, ya, color=GRAY, lw=2, ls=(0, (4, 2)), label="model only")
a.text(X[-1] + 2, yb[-1], f"+{yb[-1] / ya[-1] - 1:.0%}", color=PINK, va="center", fontsize=9.5, weight="bold")
share = cnt["explore"] / (cnt["explore"] + cnt["agent"])
a.text(3, max(yb) * 1.2, f"probes: {share:.0%} of UI actions", fontsize=7.5, color="#555", va="top")
a.set_xlim(0, 128)
a.set_xticks([0, 29, 58, 87, 116])
a.set_ylim(0, max(yb) * 1.25)
a.set_xlabel("Tasks executed")
a.set_ylabel("UI transitions recorded")
a.set_title("(c) UI knowledge accumulated", fontsize=9, loc="left")
a.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.9))

groups = [("new layout", summary["new_layout_per_model_step"], summary["new_layout_per_probe"]),
          ("new transition", summary["new_transition_per_model_step"], summary["new_transition_per_probe"])]
w = 0.36
for i, (lab, m, p) in enumerate(groups):
  b.bar(i - w / 2, m, w, color=GRAY, label="model step" if i == 0 else None)
  b.bar(i + w / 2, p, w, color=PINK, label="exploration probe" if i == 0 else None)
  b.text(i - w / 2, m + 0.015, f"{m:.2f}", ha="center", fontsize=8)
  b.text(i + w / 2, p + 0.015, f"{p:.2f}", ha="center", fontsize=8)
  b.text(i + w / 2, p + 0.075, f"{p / m:.1f}×", ha="center", fontsize=8.5, color=PINK, weight="bold")
b.set_xticks([0, 1])
b.set_xticklabels([g[0] for g in groups])
b.set_ylim(0, 0.8)
b.set_ylabel("Discoveries per UI action")
b.set_title("(d) Information gained per action", fontsize=9, loc="left")
b.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
for ext in ("pdf", "png"):
  fig.savefig(OUT / f"explore_info.{ext}", dpi=300, bbox_inches="tight")
print("saved", OUT / "explore_info.png")
