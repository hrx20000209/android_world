import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figs import COL, INK2, save, C, wilson
S = json.loads((C.ROOT / "analysis/full/summary.json").read_text())
passes = [p for p in ("pass0", "pass1") if p in S.get("FULL", {}) and p in S.get("BASE", {})]
arms = [a for a in ("BASE", "FULL", "FULL3", "FULL4", "FULL5") if a in S and any(S[a].get(p, {}).get("episodes", 0) >= 116 for p in ("pass0", "pass1"))]
lab = {"BASE": "Agent only", "FULL": "Full design v1", "FULLNE": "Full w/o E2", "FULL2": "Full design v2", "FULL3": "Full design v3", "FULL4": "Full design v4", "FULL5": "Full design v5"}
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
others = [a for a in arms if a != "BASE"]
w2 = 0.8 / len(others)
for j, a in enumerate(others):
  key = "paired" if a == "FULL" else f"paired_{a}"
  for i, p in enumerate(passes):
    pv = S[key][p]; bb, f = pv["both_calls"]; d = (f - bb) / bb
    x = i + (j - (len(others) - 1) / 2) * w2
    ax.bar(x, d * 100, w2 * 0.92, color=COL[arms.index(a)], label=lab[a] if i == 0 else None)
    ax.text(x, d * 100 + (0.6 if d >= 0 else -1.6), f"{d:+.0%}", ha="center", fontsize=7)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(len(passes))); ax.set_xticklabels(["Pass 1" if p == "pass0" else "Pass 2" for p in passes])
ax.set_ylabel("Model calls vs agent only (%)\n(tasks where both succeed)"); ax.legend(fontsize=7)
ax.set_title("Model-call change on paired successes", fontsize=9, loc="left")
ax = axes[2]
names = [("m_requery", "Re-queries\n(rollback failed)"), ("m_jump_steps", "Jump steps\nexecuted"), ("m_replay_steps", "Replay steps\n(v5 only)"), ("m_e1_hit", "E1 probes used\nby agent")]
for j, a in enumerate(others):
  for i, (k, nm) in enumerate(names):
    v = S[a]["all"].get(k, 0); x = i + (j - (len(others) - 1) / 2) * w2
    ax.bar(x, v, w2 * 0.92, color=COL[arms.index(a)], label=lab[a] if i == 0 else None)
    ax.text(x, v + 2, str(v), ha="center", fontsize=6.5)
ax.set_xticks(range(len(names))); ax.set_xticklabels([n for _, n in names], fontsize=7.5)
ax.set_ylabel("Count (232 episodes)"); ax.set_ylim(0, 150); ax.legend(fontsize=7, ncol=2, loc="upper right")
ax.set_title("Mechanism counts per design version", fontsize=9, loc="left")
fig.tight_layout(); save(fig, "fig13_full_androidworld")
print("ok")
