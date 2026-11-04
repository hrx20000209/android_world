"""Motivation (Fig. 4c/d): UI information in memory with vs. without exploration records (e2e116n FULL45).

Memory = every screen captured so far (model steps and probe destinations), launcher excluded.
  model only        captures from model execution
  + exploration     plus probe destinations
(c) distinct layouts (package, activity, layout_sig); (d) distinct content items (list-row texts, field values,
checkbox states, rgs_ui content) per layout. Data: figures/motivation/memory_growth.json (content_freshness.py
helpers); probes / model steps from reports/e2e116n_runs.jsonl.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "motivation"
d = {int(k): v for k, v in json.load(open(OUT / "memory_growth.json")).items()}
xs = sorted(d)
rows = {}
for l in open(ROOT / "reports" / "e2e116n_runs.jsonl"):
  r = json.loads(l)
  if r.get("arm") == "FULL45" and not r.get("error"):
    rows[r["task"]] = r
probes, steps = sum(r["probes"] for r in rows.values()), sum(r["steps"] for r in rows.values())

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
GRAY, PINK = "#8a8f98", "#d97fb4"
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.1))
for ax, (i_a, i_b), title, ylab in ((axs[0], (0, 1), "(c) UI layouts in memory", "Distinct layouts"),
                                    (axs[1], (2, 3), "(d) UI contents in memory", "Distinct content items")):
  a = [d[x][i_a] for x in xs]
  b = [d[x][i_b] for x in xs]
  X = [x + 1 for x in xs]
  ax.fill_between(X, a, b, color=PINK, alpha=0.25, lw=0)
  ax.plot(X, b, color=PINK, lw=2, label="model + exploration")
  ax.plot(X, a, color=GRAY, lw=2, ls=(0, (4, 2)), label="model only")
  ax.text(X[-1] + 2, b[-1], f"+{(b[-1] - a[-1]) / a[-1]:.0%}", color=PINK, va="center", fontsize=9, weight="bold")
  ax.set_xlim(0, 126)
  ax.set_xticks([0, 29, 58, 87, 116])
  ax.set_ylim(0, max(b) * 1.22)
  ax.set_xlabel("Tasks executed")
  ax.set_ylabel(ylab)
  ax.set_title(title, fontsize=9, loc="left")
axs[0].legend(fontsize=8, frameon=False, loc="upper left")
axs[1].text(4, max(d[x][3] for x in xs) * 1.14,
            f"probes = {probes / (probes + steps):.0%} of UI actions ({probes} of {probes + steps})",
            fontsize=7.5, color="#555", va="top")
fig.tight_layout()
for ext in ("pdf", "png"):
  fig.savefig(OUT / f"memory_growth.{ext}", dpi=300, bbox_inches="tight")
print("saved", OUT / "memory_growth.png")
