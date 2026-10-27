"""Ablation figures for 'Understanding MobileExplorer's performance' (reads figures/understand/ablation_summary.json)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = json.load(open("figures/understand/ablation_summary.json"))
OUT = "figures/understand"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
C_BASE, C_ME, C3, C4, C5 = "#8a8f96", "#1d6f78", "#7fb8bd", "#d9a441", "#c9d2d6"
A = ["BASE", "FULL45", "FULL45+RAND", "FULL45+INJ", "FULL45+NOSYS"]
N = ["Baseline", "Mobile-\nExplorer", "w/ random\nsearch", "w/ prompt\ninjection", "w/o pipelined\nexecution"]
COL = [C_BASE, C_ME, C3, C4, C5]
n5 = S["all5"]
# (g) calls and e2e on tasks all five arms solved -- two panels, one axis each
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.0))
for ax, key, lab in ((axs[0], "calls", "Inference calls / task"), (axs[1], "e2e", "End-to-end latency / task (s)")):
  v = [S["arms"][a]["all5"][key] for a in A]
  b = ax.bar(range(5), v, color=COL, width=0.62, edgecolor="white", linewidth=1)
  for i, x in enumerate(v): ax.text(i, x, f"{x:.1f}" if key == "calls" else f"{x:.0f}", ha="center", va="bottom", fontsize=7, color="#333")
  ax.axhline(v[1], color=C_ME, lw=0.6, ls=":")
  ax.set_xticks(range(5)); ax.set_xticklabels(N, fontsize=6.5); ax.set_ylabel(lab); ax.grid(axis="y", lw=0.3, color="#ddd"); ax.set_axisbelow(True)
fig.suptitle(f"Replacement ablations on the {n5} tasks all five variants solve (Orin, paired)", fontsize=8, y=1.02)
fig.tight_layout(); [fig.savefig(f"{OUT}/g_ablation.{e}", bbox_inches="tight", dpi=300) for e in ("pdf", "png")]
# (h) per-task latency decomposition by arm
fig, ax = plt.subplots(figsize=(3.33, 2.1))
segs = [("decide", "Model decision", C_ME), ("after", "Action exec. & UI wait", C3), ("graph", "Graph execution", C4), ("other", "Setup & finish", C5)]
for i, a in enumerate(A):
  d, x = S["arms"][a]["decomp_all5"], 0
  for k, lab, c in segs:
    w = d.get(k, 0); ax.barh(i, w, left=x, color=c, height=0.6, edgecolor="white", linewidth=1, label=lab if i == 0 else None); x += w
  ax.text(x + 1.5, i, f"{x:.0f}s · {S['arms'][a]['all5']['calls']:.1f} calls", va="center", fontsize=6.5, color="#333")
ax.set_yticks(range(5)); ax.set_yticklabels([n.replace("-\n", "").replace("\n", " ") for n in N], fontsize=7); ax.invert_yaxis()
ax.set_xlabel("Seconds per task"); ax.set_xlim(0, 150); ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
fig.tight_layout(); [fig.savefig(f"{OUT}/h_ablation_breakdown.{e}", bbox_inches="tight", dpi=300) for e in ("pdf", "png")]
