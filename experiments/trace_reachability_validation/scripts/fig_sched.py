"""Figures for the sched1 validation (base parallel vs measured-lookup system layer), held-out episodes only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "sched1"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6})
C_FIX, C_LK, C_EX, C_OT = "#8a8f96", "#1d6f78", "#d9a441", "#c9d2d6"
S = json.load(open(OUT / "formal_summary.json"))
R = {(json.loads(l)["phase"], json.loads(l)["cond"], json.loads(l)["task"], json.loads(l)["seed"]): json.loads(l)
     for l in open(ROOT / "reports" / "sched1_runs.jsonl")}


def save(fig, name):
  fig.tight_layout()
  for e in ("pdf", "png"):
    fig.savefig(OUT / f"{name}.{e}", bbox_inches="tight", dpi=300)
  plt.close(fig)


# 1. paired E2E per (task, seed); failures drawn hollow
fig, ax = plt.subplots(figsize=(3.6, 2.2))
P = S["pairs"]
for i, p in enumerate(P):
  for j, (m, c) in enumerate((("fixed", C_FIX), ("lookup", C_LK))):
    ok, t, calls = p[m]
    ax.bar(i + (j - 0.5) * 0.38, t, width=0.38, color=c if ok else "white", edgecolor=c, linewidth=1.2,
           label=(("Base parallel" if m == "fixed" else "Full design") if i == 0 else None))
    ax.text(i + (j - 0.5) * 0.38, t + 2, f"{calls}", ha="center", fontsize=6)
  if p["ratio"]:
    ax.text(i, max(p["fixed"][1], p["lookup"][1]) + 12, f"x{p['ratio']:.2f}", ha="center", fontsize=6.5, color="#333")
ax.set_xticks(range(len(P)))
ax.set_xticklabels([("Cal" if p["task"].startswith("Simple") else "Exp") + f"\ns{p['seed']}" for p in P], fontsize=7)
ax.set_ylabel("Task E2E (s), excl. preparation")
ax.set_ylim(0, 160)
ax.set_title("Held-out pairs (n=6, all temperature-matched, same initial screen)\nhollow = task failed; number = model calls; xR = full/base, both succeeded", fontsize=6.5)
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
save(fig, "fig1_paired_e2e")

# 2. timeline of one matched pair (Calendar s203: both succeeded)
fig, axs = plt.subplots(2, 1, figsize=(6.9, 2.2), sharex=True)
for ax, m, lab in ((axs[0], "fixed", "Base parallel (K=1, FULL45 ranking)"), (axs[1], "lookup", "Full design (selected K=0)")):
  r = R[("formal", m, "SimpleCalendarLocationOfEvent", 203)]
  steps = [json.loads(l) for l in open(ROOT / r["raw"] / "steps.jsonl")]
  wins = {json.loads(l)["step"]: json.loads(l) for l in open(ROOT / r["raw"] / "windows.jsonl")}
  t0 = next(s["pre"]["t"] for s in steps if s.get("pre"))
  for s in steps:
    if s["kind"] != "model" or not s.get("t_req"):
      continue
    ax.barh(1, s["t_decision"] - s["t_req"], left=s["t_req"] - t0, height=0.5, color=C_LK if m == "lookup" else C_FIX)
    w = wins.get(s["step"], {})
    t = w.get("t_start")
    for p in w.get("probes", []):
      ax.barh(0, p.get("t") or 0, left=t - t0, height=0.5, color=C_EX if p.get("rollback_ok") else "#b5452c")
      t += p.get("t") or 0
    if s.get("requery"):
      ax.text(s["t_decision"] - t0, 1.35, "re-query", fontsize=6, color="#b5452c")
  ax.set_yticks([0, 1])
  ax.set_yticklabels(["Exploration", "Inference"], fontsize=7)
  ax.set_title(f"{lab}: {r['calls']} calls, {r['task_s']:.0f} s", fontsize=7, loc="left")
axs[1].set_xlabel("Time since the first observation (s); red = probe whose return failed")
save(fig, "fig2_timeline_cal_s203")

# 3. phase composition per method (held-out mean)
fig, ax = plt.subplots(figsize=(3.6, 2.1))
for i, m in enumerate(("fixed", "lookup")):
  x = S["methods"][m]
  ax.barh(i, x["inference_s"], color=C_LK, height=0.55, label="Inference (request -> action)" if i == 0 else None)
  ax.barh(i, x["other_s"], left=x["inference_s"], color=C_OT, height=0.55, label="Action, verify, UI wait, graph steps" if i == 0 else None)
  ax.barh(i - 0.32, x["exploration_s"], color=C_EX, height=0.12, label="Exploration (overlaps inference)" if i == 0 else None)
  ax.text(x["inference_s"] + x["other_s"] + 1, i, f"{x['task_s_all']:.0f} s, {x['calls']:.1f} calls", va="center", fontsize=6.5)
ax.set_yticks([0, 1])
ax.set_yticklabels([f"Base parallel ({S['methods']['fixed']['success']}/6)", f"Full design ({S['methods']['lookup']['success']}/6)"], fontsize=7)
ax.invert_yaxis()
ax.set_xlim(0, 150)
ax.set_xlabel("Mean per episode (s), 6 held-out each")
fig.legend(fontsize=6, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.3, right=0.98, top=0.95, bottom=0.45)
for e in ("pdf", "png"):
  fig.savefig(OUT / f"fig3_phases.{e}", dpi=300)
plt.close(fig)
print("ok")
