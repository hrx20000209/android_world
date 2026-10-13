import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figs import COL, INK2, save, C, wilson
S = json.loads((C.ROOT / "analysis/full/summary.json").read_text())
passes = [p for p in ("pass0", "pass1") if p in S.get("FULL", {}) and p in S.get("BASE", {})]
arms = [a for a in ("BASE", "FULL", "FULL2", "FULL3") if a in S and any(S[a].get(p, {}).get("episodes", 0) >= 116 for p in ("pass0", "pass1"))]
lab = {"BASE": "Agent only", "FULL": "Full design v1", "FULLNE": "Full w/o E2", "FULL2": "Full design v2", "FULL3": "Full design v3"}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
ax = axes[0]
w = 0.8 / len(arms)
for j, a in enumerate(arms):
  for i, p in enumerate(passes):
    if p not in S[a] or S[a][p]["episodes"] < 116:
      continue   # only plot complete passes
    s = S[a][p]; v = s["success"] / s["episodes"]; lo, hi = wilson(s["success"], s["episodes"])
    ax.bar(i + (j - (len(arms) - 1) / 2) * w, v, w * 0.92, color=COL[j], label=lab[a] if i == 0 else None)
    ax.errorbar(i + (j - (len(arms) - 1) / 2) * w, v, yerr=[[v - lo], [hi - v]], color=INK2, capsize=2, lw=0.8)
    ax.text(i + (j - (len(arms) - 1) / 2) * w, 0.03, f"{s['success']}/{s['episodes']}", ha="center", color="white", fontsize=7, rotation=90)
ax.set_xticks(range(len(passes)))
ax.set_xticklabels(["Pass 1: cross-task\nknowledge only" if p == "pass0" else "Pass 2: + same-task\nhistory (new seeds)" for p in passes], fontsize=7.5)
ax.set_ylim(0, 0.8); ax.set_ylabel("Task success (116 AndroidWorld tasks)")
ax.legend(fontsize=7); ax.set_title("Success rate", fontsize=9, loc="left")
ax = axes[1]
for i, p in enumerate(passes):
  pv = S["paired"][p]
  b, f = pv["both_calls"]; n = pv["both"]
  ax.bar(i - 0.2, b / n, 0.38, color=COL[0], label="Agent only" if i == 0 else None)
  ax.bar(i + 0.2, f / n, 0.38, color=COL[1], label="Full design" if i == 0 else None)
  ax.text(i, max(b, f) / n + 0.2, f"{(f - b) / b:+.0%}\n(n={n} pairs)", ha="center", fontsize=7)
ax.set_xticks(range(len(passes))); ax.set_xticklabels(["Pass 1" if p == "pass0" else "Pass 2" for p in passes])
ax.set_ylabel("Model calls per episode"); ax.set_ylim(0, 12); ax.legend(fontsize=7)
ax.set_title("Model calls where both arms succeed", fontsize=9, loc="left")
ax = axes[2]
s = S["FULL"]["all"]
items = [("E1 probes", s.get("m_probes_e1", 0), s.get("m_e1_hit", 0)), ("E2 probes", s.get("m_probes_e2", 0), s.get("m_e2_hit", 0)),
         ("Jump plans -> steps", s.get("m_jumps", 0), s.get("m_jump_steps", 0))]
for i, (nm, tot, used) in enumerate(items):
  ax.bar(i - 0.2, tot, 0.38, color=COL[3], label="attempted" if i == 0 else None)
  ax.bar(i + 0.2, used, 0.38, color=COL[2], label="used by agent / executed" if i == 0 else None)
  ax.text(i, max(tot, used) + 3, f"{used}/{tot}", ha="center", fontsize=7)
ax.bar(3, s.get("m_requery", 0), 0.38, color=COL[7], label="re-queries (rollback failed)")
ax.text(3, s.get("m_requery", 0) + 3, str(s.get("m_requery", 0)), ha="center", fontsize=7)
ax.set_xticks(range(4)); ax.set_xticklabels(["E1 predicted\nnext", "E2 gap\ncoverage", "R1 jumps\n(steps)", "Re-queries"], fontsize=7.5)
ax.legend(fontsize=6.5); ax.set_title("Full design: mechanism funnel (all passes so far)", fontsize=9, loc="left")
fig.tight_layout(); save(fig, "fig13_full_androidworld")
print("ok")
