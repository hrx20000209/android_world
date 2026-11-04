"""Fig. 4(c)(d): UI knowledge added by exploration (AndroidWorld, 116 tasks, e2e116n, MobileExplorer FULL45).
Data embedded; computed by scripts/fig_explore_info.py from graph_snapshots/e2e116n/FULL45/graph.jsonl.
Only matplotlib is needed.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- data ----------------------------------------------------------------------------------------------------
# tasks executed (1..116) and distinct UI transitions recorded after each task
TASKS = list(range(1, 117))
TRANSITIONS_MODEL_ONLY = [9, 12, 12, 18, 25, 30, 33, 37, 37, 39, 42, 45, 49, 53, 63, 64, 69, 74, 77, 77, 79, 84, 87, 89, 92, 92, 94, 97, 99, 109, 114, 116, 118, 124, 132, 143, 157, 159, 162, 163, 164, 166, 175, 178, 185, 187, 191, 193, 195, 197, 198, 200, 201, 209, 215, 218, 218, 220, 226, 228, 236, 237, 242, 247, 248, 248, 248, 248, 249, 251, 253, 255, 255, 255, 255, 266, 274, 279, 281, 285, 291, 297, 299, 303, 304, 309, 313, 313, 313, 313, 315, 315, 318, 319, 320, 321, 321, 321, 321, 323, 323, 324, 324, 324, 324, 327, 327, 327, 327, 327, 330, 334, 334, 337, 348, 351]
TRANSITIONS_MODEL_PLUS_EXPLORATION = [9, 12, 12, 19, 26, 32, 35, 40, 40, 42, 46, 49, 55, 59, 76, 77, 82, 90, 93, 93, 97, 102, 105, 108, 112, 112, 114, 117, 119, 131, 139, 142, 144, 150, 158, 169, 184, 190, 193, 194, 195, 199, 208, 211, 218, 220, 226, 228, 230, 232, 233, 235, 236, 245, 251, 254, 254, 256, 267, 270, 278, 279, 285, 292, 293, 293, 293, 293, 294, 299, 301, 304, 305, 305, 307, 322, 333, 339, 341, 347, 355, 363, 367, 372, 373, 383, 387, 388, 390, 390, 393, 393, 396, 397, 399, 401, 401, 401, 402, 404, 405, 406, 407, 407, 408, 411, 411, 411, 411, 411, 415, 422, 422, 425, 437, 442]
# discoveries per UI action (memory = everything recorded before the action)
MODEL_STEPS, PROBES = 1008, 164
NEW_LAYOUT = {"model step": 0.2024, "exploration probe": 0.3963}
NEW_TRANSITION = {"model step": 0.3482, "exploration probe": 0.5549}

# ---- plot ----------------------------------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
GRAY, PINK = "#8a8f98", "#d97fb4"
fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.15), gridspec_kw={"width_ratios": [1.15, 1]})
ya, yb = TRANSITIONS_MODEL_ONLY, TRANSITIONS_MODEL_PLUS_EXPLORATION
a.fill_between(TASKS, ya, yb, color=PINK, alpha=0.25, lw=0)
a.plot(TASKS, yb, color=PINK, lw=2, label="model + exploration")
a.plot(TASKS, ya, color=GRAY, lw=2, ls=(0, (4, 2)), label="model only")
a.text(TASKS[-1] + 2, yb[-1], f"+{yb[-1] / ya[-1] - 1:.0%}", color=PINK, va="center", fontsize=9.5, weight="bold")
a.text(3, max(yb) * 1.2, f"probes: {PROBES / (PROBES + MODEL_STEPS):.0%} of UI actions", fontsize=7.5, color="#555", va="top")
a.set_xlim(0, 128)
a.set_xticks([0, 29, 58, 87, 116])
a.set_ylim(0, max(yb) * 1.25)
a.set_xlabel("Tasks executed")
a.set_ylabel("UI transitions recorded")
a.set_title("(c) UI knowledge accumulated", fontsize=9, loc="left")
a.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.9))
w = 0.36
for i, (lab, d) in enumerate((("new layout", NEW_LAYOUT), ("new transition", NEW_TRANSITION))):
  m, p = d["model step"], d["exploration probe"]
  b.bar(i - w / 2, m, w, color=GRAY, label="model step" if i == 0 else None)
  b.bar(i + w / 2, p, w, color=PINK, label="exploration probe" if i == 0 else None)
  b.text(i - w / 2, m + 0.015, f"{m:.2f}", ha="center", fontsize=8)
  b.text(i + w / 2, p + 0.015, f"{p:.2f}", ha="center", fontsize=8)
  b.text(i + w / 2, p + 0.075, f"{p / m:.1f}×", ha="center", fontsize=8.5, color=PINK, weight="bold")
b.set_xticks([0, 1])
b.set_xticklabels(["new layout", "new transition"])
b.set_ylim(0, 0.8)
b.set_ylabel("Discoveries per UI action")
b.set_title("(d) Information gained per action", fontsize=9, loc="left")
b.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
for ext in ("pdf", "png"):
  fig.savefig(f"explore_info.{ext}", dpi=300, bbox_inches="tight")
print("saved explore_info.pdf / .png")
