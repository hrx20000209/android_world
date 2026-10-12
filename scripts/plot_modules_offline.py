#!/usr/bin/env python3
"""Figures for the module explainability section from modules_offline.json.

  python scripts/plot_modules_offline.py OUT_DIR RUN_ROOT [RUN_ROOT ...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
runs = {Path(r).name.split("_")[-1]: json.loads((Path(r) / "modules_offline.json").read_text()) for r in sys.argv[2:]}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# 1. Removed vs added inference calls per task category.
cats = ["multi-entity", "single-entity", "query"]
fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 2.6), sharey=True)
for ax, (name, rep) in zip(axes if len(runs) > 1 else [axes], runs.items()):
  hops = [rep[c]["removed_inferences"]["replayed_hops"] for c in cats]
  skips = [rep[c]["removed_inferences"]["graph_skips"] for c in cats]
  cost = [-rep[c]["added_cost"]["contract_calls"] for c in cats]
  ax.bar(cats, hops, color="#3b6ea5", label="procedure replay hops")
  ax.bar(cats, skips, bottom=hops, color="#e69f00", label="graph skips")
  ax.bar(cats, cost, color="#999999", label="contract calls (cost)")
  ax.axhline(0, color="k", lw=0.6)
  ax.set_title(name); ax.set_ylabel("calls removed (+) / added (−)")
  ax.tick_params(axis="x", labelsize=8)
axes[0].legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(out / "fig_removed_by_module.png", dpi=200)

# 2. Paired per-task steps on both-success tasks, coloured by acting module.
name, rep = list(runs.items())[-1]
rows = [r for r in rep["per_task"] if r["base"][0] > 0 and r["full_rec"][0] > 0]
rows.sort(key=lambda r: r["full_rec"][1] - r["base"][1])
colour = lambda r: "#3b6ea5" if r["replays"] else "#e69f00" if r["skips"] else "#56b4e9" if r["context_steps"] else "#bbbbbb"
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.bar(range(len(rows)), [r["full_rec"][1] - r["base"][1] for r in rows], color=[colour(r) for r in rows])
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks([]); ax.set_xlabel(f"{len(rows)} tasks both arms solve ({name}), sorted")
ax.set_ylabel("Δ steps (full_rec − base)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=l) for l, c in (("replay fired", "#3b6ea5"), ("skip fired", "#e69f00"),
                                                      ("context only", "#56b4e9"), ("no graph action", "#bbbbbb"))],
          fontsize=7, frameon=False, ncol=4, loc="lower right")
fig.tight_layout(); fig.savefig(out / "fig_paired_steps.png", dpi=200)

# 3. Node key: revisit share under layout key vs exact screenshot.
fig, ax = plt.subplots(figsize=(3.2, 2.4))
xs, labels = [], []
for i, (name, rep) in enumerate(runs.items()):
  a = rep["A1_revisit"]["full"]
  ax.bar([i * 3, i * 3 + 1], [a["layout_revisits"] / a["decisions"], a["pixel_revisits"] / a["pixel_decisions"]],
         color=["#3b6ea5", "#bbbbbb"])
  xs += [i * 3, i * 3 + 1]; labels += [f"{name}\nlayout", f"{name}\npixels"]
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("decisions on a revisited node")
fig.tight_layout(); fig.savefig(out / "fig_node_key_revisit.png", dpi=200)
print("wrote", sorted(p.name for p in out.glob("*.png")))
