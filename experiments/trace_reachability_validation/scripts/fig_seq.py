import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figs import COL, INK2, save, C, wilson
S = json.loads((C.ROOT / "analysis/seq/summary.json").read_text())
arms = [a for a in ("BASE", "G", "GX", "GC") if a in S and "all" in S[a]]
lab = {"BASE": "Agent only", "G": "Graph jump\n(agent edges)", "GX": "Graph jump\n+ exploration", "GC": "+ commit jump\n(same-task)"}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
x = np.arange(len(arms))
ax = axes[0]
for i, a in enumerate(arms):
  s = S[a]["all"]; p = s["success"] / s["episodes"]; lo, hi = wilson(s["success"], s["episodes"])
  ax.bar(i, p, color=COL[0]); ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color=INK2, capsize=2, lw=0.8)
  ax.text(i, 0.05, f"{s['success']}/{s['episodes']}", ha="center", color="white", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([lab[a] for a in arms], fontsize=7); ax.set_ylim(0, 1.1)
ax.set_title("Task success", fontsize=9, loc="left")
ax = axes[1]
vals = []
for a in arms:
  if a == "BASE":
    continue
  pv = S[a].get("paired_vs_BASE", {})
  if pv.get("both_success_pairs"):
    b, g = pv["both_success_steps"]; vals.append((a, b, g, pv["both_success_pairs"]))
for i, (a, b, g, n) in enumerate(vals):
  ax.bar(i - 0.2, b / n, 0.38, color=COL[3], label="Agent only" if i == 0 else None)
  ax.bar(i + 0.2, g / n, 0.38, color=COL[0], label="graph arm" if i == 0 else None)
  ax.text(i, max(b, g) / n + 0.2, f"{(g - b) / b:+.0%}\n(n={n})", ha="center", fontsize=7)
ax.set_xticks(range(len(vals))); ax.set_xticklabels([lab[v[0]] for v in vals], fontsize=7)
ax.set_ylabel("Steps per episode"); ax.set_ylim(0, 13); ax.legend(fontsize=7, loc="lower right")
ax.set_title("Steps on pairs where both succeed", fontsize=9, loc="left")
ax = axes[2]
ga = [a for a in arms if a != "BASE"]
ax.bar(np.arange(len(ga)) - 0.2, [S[a]["all"]["jumps"] for a in ga], 0.38, color=COL[1], label="jumps planned")
ax.bar(np.arange(len(ga)) + 0.2, [S[a]["all"]["jump_steps"] for a in ga], 0.38, color=COL[2], label="hops executed+verified")
ax.set_xticks(range(len(ga))); ax.set_xticklabels([lab[a] for a in ga], fontsize=7); ax.legend(fontsize=7)
ax.set_title("Graph jumps (commits block most)", fontsize=9, loc="left")
ax = axes[3]
ax.bar(x, [S[a]["all"]["wall_s_mean"] for a in arms], color=COL[6])
for i, a in enumerate(arms):
  s = S[a]["all"]; ax.text(i, s["wall_s_mean"], f"probes {s['probes']}\nstopped {s['disabled']}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([lab[a] for a in arms], fontsize=7); ax.set_ylim(0, 70)
ax.set_title("Wall time per episode (s)", fontsize=9, loc="left")
fig.tight_layout(); save(fig, "fig11_same_app_sequence")
print("ok")
