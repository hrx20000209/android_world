"""UI memory similarity in Fig. 4(a) terms (AndroidWorld, e2e116n, MobileExplorer FULL45).
For the first screen of each layout the model meets in a task (501 screens, 116 tasks): maximum Jaccard
similarity to same-app screens recorded by earlier tasks (model execution). 95% CI: bootstrap over tasks.
Data embedded; computed by scripts/fig_memory_similarity.py. Only matplotlib is needed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- data: mean and 95% CI ---------------------------------------------------------------------------------
SIM = {
    "Structure": (0.739, 0.665, 0.801),
    "Content": (0.628, 0.556, 0.692),
}
# with exploration records added to the memory (not plotted; gain is +0.01 to +0.02)
SIM_WITH_EXPLORATION = {
    "Structure": (0.752, 0.683, 0.811),
    "Content": (0.642, 0.574, 0.707),
}

# ---- plot ----------------------------------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
COLORS = {"Structure": "#b8bec6", "Content": "#d97fb4"}
fig, ax = plt.subplots(figsize=(2.6, 2.2))
for i, (lab, (v, lo, hi)) in enumerate(SIM.items()):
  ax.bar(i, v, 0.5, color=COLORS[lab], yerr=[[v - lo], [hi - v]], capsize=3)
  ax.text(i, hi + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(range(len(SIM)))
ax.set_xticklabels(list(SIM))
ax.set_ylim(0, 1.05)
ax.set_ylabel("Jaccard similarity to\nscreens of earlier tasks")
fig.tight_layout()
for ext in ("pdf", "png"):
  fig.savefig(f"memory_similarity.{ext}", dpi=300, bbox_inches="tight")
print("saved memory_similarity.pdf / .png")
