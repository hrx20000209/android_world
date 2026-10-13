import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figs import COL, INK2, save, C
S = json.loads((C.ROOT / "analysis/xt_bridge/summary.json").read_text())
pols = [("TRACE", "Cross-task traces"), ("BR_full", "+ graph (edges seen\nbefore decision)"),
        ("BR_oracleGraph", "+ graph (exploration\ncovers the app)"), ("GOAL+BR_oracleGraph", "+ goal-text match\n+ covered graph")]
groups = [("gap (no cross-task trace at page)", "Gap pages\n(no other-task trace)"), ("covered by cross-task trace", "Pages with\nother-task traces")]
fig, ax = plt.subplots(figsize=(8.5, 3.4))
w = 0.2
for j, (p, lab) in enumerate(pols):
  vals = [S[g][p]["top1"] for g, _ in groups]
  bars = ax.bar(np.arange(2) + (j - 1.5) * w, vals, w * 0.92, color=COL[j], label=lab)
  for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=7)
ax.set_xticks(range(2))
ax.set_xticklabels([f"{l}\n(n={S[g]['n']})" for g, l in groups], fontsize=8)
ax.set_ylabel("Top-1: agent's next click ranked first")
ax.set_ylim(0, 1)
ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.set_title("Same app, other task's knowledge only: exploration fills the gap pages (offline, 4 apps x 2 tasks)", fontsize=8.5, loc="left")
save(fig, "fig12_cross_task_gap_bridging")
print("ok")
