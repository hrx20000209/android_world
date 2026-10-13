"""Figures for the trace/reachability study (real data only)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FIG = C.ROOT / "figures"
FIG.mkdir(exist_ok=True)
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "savefig.bbox": "tight"})
PN = {"P0": "Round-robin", "P0u": "Unknown-first (old)", "P1": "State freq.", "P2": "Longest suffix",
      "P3": "Var-order (sup>=2)", "P4": "Trace+frontier", "ORA": "Oracle"}


def save(fig, name):
  fig.savefig(FIG / f"{name}.png", dpi=200)
  fig.savefig(FIG / f"{name}.pdf")
  plt.close(fig)


def wilson(k, n, z=1.96):
  if not n:
    return 0, 0
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return c - h, c + h


def fig1():
  rows = [json.loads(l) for l in (C.ROOT / "analysis/pilot_a/decisions_struct_sig.jsonl").read_text().splitlines()]
  R = [r for r in rows if r["setting"] == "warm"]
  pols = ["P0", "P0u", "P1", "P2", "P3", "P4"]
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
  for ax, denom in zip(axes, ("eligible", "all")):
    base = [r for r in R if r["eligible"]] if denom == "eligible" else R
    for j, k in enumerate((1, 2, 3)):
      vals, errs = [], []
      for p in pols:
        h = sum(1 for r in base if r[f"rank_{p}"] and r[f"rank_{p}"] <= k)
        lo, hi = wilson(h, len(base))
        vals.append(h / len(base))
        errs.append((h / len(base) - lo, hi - h / len(base)))
      ax.bar(np.arange(len(pols)) + (j - 1) * 0.27, vals, 0.25, color=COL[j], label=f"Top-{k}",
             yerr=np.array(errs).T, error_kw=dict(lw=0.6, capsize=1.5, ecolor=INK2))
    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels([PN[p] for p in pols], rotation=25, ha="right", fontsize=7.5)
    ttl = (f"Safe-eligible clicks (n={len(base)})" if denom == "eligible"
           else f"All agent clicks (n={len(base)}; {len(base) - sum(r['eligible'] for r in base)} unsafe targets count as misses)")
    ax.set_title(ttl, fontsize=8.5, loc="left")
  axes[0].set_ylabel("Next-action coverage (warm history)")
  axes[0].legend(fontsize=7)
  save(fig, "fig1_trace_action_coverage")


def fig2():
  rows = [json.loads(l) for l in (C.ROOT / "analysis/pilot_a/decisions_struct_sig.jsonl").read_text().splitlines()]
  settings = [("warm", "Warm (earlier runs,\nany task)"), ("loto", "Cross-task\n(leave task out)"),
              ("loao", "Cross-app\n(leave app out)"), ("cold", "Cold\n(own episode only)")]
  pols = ["P0", "P1", "P2", "P3", "P4"]
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
  ax = axes[0]
  for j, p in enumerate(pols):
    vals = []
    for s, _ in settings:
      base = [r for r in rows if r["setting"] == s and r["eligible"]]
      vals.append(sum(1 for r in base if r[f"rank_{p}"] == 1) / len(base))
    ax.bar(np.arange(4) + (j - 2) * 0.16, vals, 0.15, color=COL[j], label=PN[p])
  ax.set_xticks(range(4))
  ax.set_xticklabels([l for _, l in settings], fontsize=7.5)
  ax.set_ylabel("Top-1 coverage (safe-eligible, n=276)")
  ax.legend(fontsize=6.5, ncol=2)
  ax.set_title("Where the predictive signal comes from", fontsize=9, loc="left")
  ax = axes[1]
  # novel coverage: agent's target safe AND outcome unknown before the decision
  for j, (s, lab) in enumerate(settings[:2]):
    base = [r for r in rows if r["setting"] == s and r["eligible"]]
    unk = [r for r in base if not r["target_known"]]
    for i, p in enumerate(["P0u", "P1", "P4", "ORA"]):
      v = sum(1 for r in unk if r[f"rank_{p}"] == 1)
      ax.bar(j * 5 + i, v, color=COL[i])
      ax.text(j * 5 + i, v, str(v), ha="center", va="bottom", fontsize=7)
  ax.set_xticks([1.5, 6.5])
  ax.set_xticklabels(["Warm", "Cross-task"])
  for i, p in enumerate(["P0u", "P1", "P4", "ORA"]):
    ax.bar(0, 0, color=COL[i], label=PN[p])
  ax.legend(fontsize=7)
  ax.set_ylabel("Top-1 probes that reveal the agent's\nnext action outcome for the first time")
  ax.set_title("New, used information per window (budget = 1 probe)", fontsize=9, loc="left")
  fig.tight_layout()
  save(fig, "fig2_trace_generalization")


def fig3():
  rows = [json.loads(l) for l in (C.ROOT / "analysis/pilot_a/decisions_struct_sig.jsonl").read_text().splitlines()]
  base = [r for r in rows if r["setting"] == "warm" and r["eligible"]]
  ks = [1, 2, 3, 4, 5, 6]
  fig, ax = plt.subplots(figsize=(5.2, 3.2))
  for j, p in enumerate(["P0", "P0u", "P1", "P4"]):
    ax.plot(ks, [sum(1 for r in base if r[f"rank_{p}"] and r[f"rank_{p}"] <= k) / len(base) for k in ks],
            marker="o", ms=4, color=COL[j], lw=1.6, label=PN[p])
  med = 1.75
  ax2 = ax.secondary_xaxis("top", functions=(lambda k: k * med, lambda t: t / med))
  ax2.set_xlabel("Projected exploration time at measured median 1.75 s/probe (s)", fontsize=7.5)
  ax.axvline(2.03 / med, color=COL[4], ls=":", lw=1.2)
  ax.text(2.03 / med + 0.05, 0.05, "LabServer\nlatency", fontsize=6.5, color=INK2)
  ax.set_xlabel("Probe budget per window (first-hop probes)")
  ax.set_ylabel("Agent's next action probed (n=276)")
  ax.legend(fontsize=7, loc="lower right")
  ax.set_ylim(0, 1)
  save(fig, "fig3_coverage_vs_budget")


def fig4():
  sc = json.loads((C.ROOT / "analysis/pilot_b/score.json").read_text())
  conds = ["M1", "M2", "M2P", "M2pre", "M3", "M4"]
  names = {"M1": "One-hop UTG", "M2": "+Child affordances", "M2P": "Placebo (shuffled)", "M2pre": "Child aff. first",
           "M3": "Depth-2 paths", "M4": "Full app graph"}
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
  ax = axes[0]
  x = np.arange(len(conds))
  ax.bar(x - 0.2, [sc[c]["fixed"] for c in conds], 0.38, color=COL[2], label="fixed (wrong -> valid)")
  ax.bar(x + 0.2, [sc[c]["broken"] for c in conds], 0.38, color=COL[7], label="broken (valid -> wrong)")
  for i, c in enumerate(conds):
    ax.text(i, max(sc[c]["fixed"], sc[c]["broken"]) + 0.2, f"chg {sc[c]['changed_vs_M0']}/{sc[c]['n']}\neff {sc[c]['N_efficient']}/{sc[c]['N_n']}",
            ha="center", fontsize=6.5, color=INK2)
  ax.set_xticks(x)
  ax.set_xticklabels([names[c] for c in conds], rotation=25, ha="right", fontsize=7.5)
  ax.set_ylabel("Decision states")
  ax.set_ylim(0, 7)
  ax.legend(fontsize=7, loc="upper left")
  ax.set_title("Effect vs. no memory (M0 repeat: 0/47 changed)", fontsize=9, loc="left")
  ax = axes[1]
  ax.bar(x, [sc[c]["extra_tokens_mean"] for c in conds], color=COL[0])
  for i, c in enumerate(conds):
    ax.text(i, sc[c]["extra_tokens_mean"], f"{sc[c]['extra_tokens_mean']:.0f}", ha="center", va="bottom", fontsize=7)
  ax.set_xticks(x)
  ax.set_xticklabels([names[c] for c in conds], rotation=25, ha="right", fontsize=7.5)
  ax.set_ylabel("Extra prompt tokens (mean)")
  ax.set_title("Prompt overhead", fontsize=9, loc="left")
  fig.tight_layout()
  save(fig, "fig4_memory_formats")


def _thumb(ep, cid, w=250):
  from PIL import Image
  im = Image.open(Path(ep) / "img" / f"{cid}.jpg").convert("RGB")
  return im.resize((w, int(w * im.height / im.width)))


def fig5():
  """Real depth-2 reachability example: parent -> child -> second-hop destination."""
  cands = []
  for d in sorted((C.ROOT / "raw/phase2").glob("*/*")):
    for p in C.jl(d / "probes.jsonl"):
      if p.get("depth") == 1 and p.get("second_hop") and p["rollback_ok"] and p["second_hop"]["outcome"] == "new_screen":
        cands.append((d, p))
  if not cands:
    return
  picks = cands[:: max(1, len(cands) // 2)][:2]
  fig, axes = plt.subplots(len(picks), 4, figsize=(11, 5.0 * len(picks)), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
  axes = np.atleast_2d(axes)
  for r, (d, p) in enumerate(picks):
    sh = p["second_hop"]
    for j, (cid, t) in enumerate([(p["src_cid"], "Parent A (agent's screen)"),
                                  (p["dst_cid"], f"A --tap '{(p['selector'].get('label') or p['selector'].get('rid'))[:18]}'--> B"),
                                  (sh["dst_cid"], f"B --'{sh['family'].split('|')[-1][:16]}'--> C")]):
      axes[r][j].imshow(_thumb(d, cid))
      axes[r][j].set_title(t, fontsize=7.5, loc="left")
      axes[r][j].axis("off")
      axes[r][j].grid(False)
    txt = (f"{d.parent.name} {d.name} step {p['step']}\n\nReachability record attached to A:\n"
           f"  first hop: {p['family'][:40]}\n  child affordances (raw a11y labels):\n    "
           + "\n    ".join(x[:30] for x in p.get("child_affordances", [])[:10]) +
           f"\n  verified second hop: {sh['family'][:36]}\n  rollback B->A ok: {p['rollback_ok']}\n"
           f"  probe+rollback: {p['t_total_s']:.1f}s (incl. 2nd hop {sh['t_total_s']:.1f}s)")
    axes[r][3].text(0, 1, txt, va="top", fontsize=7, family="DejaVu Sans Mono")
    axes[r][3].axis("off")
  save(fig, "fig5_reachability_examples")


def fig6_7():
  S = json.loads((C.ROOT / "analysis/phase2/summary.json").read_text())
  arms = [a for a in ("RR", "TG", "RR1", "TG1") if a in S and S[a]["episodes"]]
  lab = {"RR": "RR d2", "TG": "TG d2", "RR1": "RR d1", "TG1": "TG d1"}
  fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
  ax = axes[0]
  x = np.arange(len(arms))
  w = 0.2
  ax.bar(x - 1.5 * w, [S[a]["eligible_with_active_window"] for a in arms], w, color=COL[0], label="eligible clicks, explorer active")
  ax.bar(x - 0.5 * w, [S[a]["hit_eligible_active"] for a in arms], w, color=COL[2], label="agent's action probed in window")
  ax.bar(x + 0.5 * w, [S[a]["reuse_exact"] for a in arms], w, color=COL[3], label="... outcome == real (exact)")
  ax.bar(x + 1.5 * w, [S[a]["probe_done_before_model"] for a in arms], w, color=COL[6], label="... finished before model returned")
  ax.set_xticks(x)
  ax.set_xticklabels([lab[a] for a in arms], fontsize=7.5)
  ax.legend(fontsize=6.5)
  ax.set_title(f"Online coverage of the agent's next action\n(n clicks = {', '.join(str(S[a]['clicks']) for a in arms)})", fontsize=8.5, loc="left")
  ax = axes[1]
  ax.bar(x - 0.2, [S[a]["episodes_disabled"] for a in arms], 0.38, color=COL[7], label="episodes w/ rollback failure (explorer stopped)")
  ax.bar(x + 0.2, [S[a]["requery"] for a in arms], 0.38, color=COL[1], label="extra model re-queries")
  for i, a in enumerate(arms):
    ax.text(i, max(S[a]["episodes_disabled"], S[a]["requery"]) + 0.3, f"{S[a]['rollback_fail']}/{S[a]['probes']} probes\nfailed rollback", ha="center", fontsize=6.5)
  ax.set_xticks(x)
  ax.set_xticklabels([lab[a] for a in arms], fontsize=7.5)
  ax.set_ylim(0, 22)
  ax.legend(fontsize=6.5, loc="upper left")
  ax.set_title("Exploration safety (18 episodes per arm)", fontsize=8.5, loc="left")
  ax = axes[2]
  ax.bar(x, [S[a]["success"] for a in arms], color=COL[0])
  for i, a in enumerate(arms):
    ax.text(i, S[a]["success"], f"{S[a]['success']}/{S[a]['episodes']}\ncalls {S[a]['model_calls']}", ha="center", va="bottom", fontsize=7)
  ax.set_xticks(x)
  ax.set_xticklabels([lab[a] for a in arms], fontsize=7.5)
  ax.set_ylim(0, 21)
  ax.set_title("Closed-loop task success (same 18 variants)", fontsize=8.5, loc="left")
  fig.tight_layout()
  save(fig, "fig6_7_online_feasibility_closed_loop")


def fig8():
  """Representative failures: second-hop rollback failure, stale pre-observation."""
  fails = []
  for d in sorted((C.ROOT / "raw/phase2").glob("*/*")):
    for p in C.jl(d / "probes.jsonl"):
      if not p["rollback_ok"] and p.get("rollback", {}).get("check"):
        fails.append((d, p))
  if not fails:
    return
  picks = fails[:: max(1, len(fails) // 3)][:3]
  fig, axes = plt.subplots(len(picks), 3, figsize=(8.5, 4.6 * len(picks)))
  axes = np.atleast_2d(axes)
  for r, (d, p) in enumerate(picks):
    for j, (cid, t) in enumerate([(p["src_cid"], f"source ({d.parent.name[:16]})"),
                                  (p["dst_cid"], f"after tap '{(p['selector'].get('label') or p['selector'].get('rid'))[:16]}' (depth {p['depth']})"),
                                  (p["rollback"]["check"], f"after rollback: {p['rollback']['method']} (FAILED)")]):
      try:
        axes[r][j].imshow(_thumb(d, cid))
      except Exception:  # pylint: disable=broad-except
        pass
      axes[r][j].set_title(t, fontsize=7, loc="left")
      axes[r][j].axis("off")
  save(fig, "fig8_failure_cases")


def fig9_stage():
  S = json.loads((C.ROOT / "analysis/phase3_offline/summary.json").read_text())
  b = ["s1", "s2-3", "s4+"]
  fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
  ax = axes[0]
  for j, st in enumerate(["warm", "loto", "cold"]):
    vals = [S["q2"][st][x]["top1_rate"] or 0 for x in b]
    ax.plot(range(3), vals, marker="o", color=COL[j], lw=1.6, label={"warm": "same-task history", "loto": "other tasks only", "cold": "own episode only"}[st])
  ns = [S["q2"]["warm"][x]["eligible"] for x in b]
  ax.set_xticks(range(3))
  ax.set_xticklabels([f"{x}\n(n={n})" for x, n in zip(["2nd step", "steps 3-4", "step 5+"], ns)], fontsize=7.5)
  ax.set_ylim(0, 1)
  ax.set_ylabel("Top-1 next-action coverage (P1)")
  ax.legend(fontsize=7)
  ax.set_title("Predictability by task stage", fontsize=9, loc="left")
  ax = axes[1]
  ax.bar(range(3), [S["q2"]["warm"][x]["eligible_rate"] for x in b], color=COL[3])
  ax.set_xticks(range(3))
  ax.set_xticklabels(["2nd step", "steps 3-4", "step 5+"], fontsize=7.5)
  ax.set_ylim(0, 1)
  ax.set_ylabel("Agent clicks that are probe-safe")
  ax.set_title("Late steps are mostly commits", fontsize=9, loc="left")
  ax = axes[2]
  q3 = S["q3"]
  cats = ["agent_only", "both", "explore_only", "unknown"]
  names = {"agent_only": "prior agent runs only", "both": "agent + explorer", "explore_only": "explorer only", "unknown": "unknown"}
  left = np.zeros(3)
  for j, c in enumerate(cats):
    v = np.array([q3[st].get(c, 0) / q3[st]["n"] for st in ("warm", "loto", "cold")])
    ax.barh(range(3), v, left=left, color=[COL[0], COL[2], COL[1], GRID][j], label=names[c], edgecolor="white")
    left += v
  ax.set_yticks(range(3))
  ax.set_yticklabels(["same-task hist.", "other tasks", "own episode"], fontsize=7.5)
  ax.set_xlabel("Agent clicks (n=533): outcome known before decision from ...")
  ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2)
  ax.set_title("Where outcome knowledge comes from", fontsize=9, loc="left")
  fig.tight_layout()
  save(fig, "fig9_stage_and_knowledge_source")


def fig10_closed_loop():
  S = json.loads((C.ROOT / "analysis/phase3/summary.json").read_text())
  arms = [a for a in ("BASE", "BASE2", "PF", "PFV", "PFVS", "PFVL", "PFVA") if a in S]
  lab = {"BASE": "Agent\nrun 1", "BASE2": "Agent\nrun 2", "PF": "Prefill\nno verify", "PFV": "Prefill\n+verify", "PFVS": "+verify\nstrict", "PFVL": "+verify\nother tasks", "PFVA": "+verify\nstage-aware"}
  fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
  x = np.arange(len(arms))
  ax = axes[0]
  ax.bar(x, [S[a]["success"] / S[a]["episodes"] for a in arms], color=COL[0])
  for i, a in enumerate(arms):
    lo, hi = wilson(S[a]["success"], S[a]["episodes"])
    ax.errorbar(i, S[a]["success"] / S[a]["episodes"], yerr=[[S[a]["success"] / S[a]["episodes"] - lo], [hi - S[a]["success"] / S[a]["episodes"]]], color=INK2, capsize=2, lw=0.8)
    ax.text(i, 0.04, f"{S[a]['success']}/{S[a]['episodes']}", ha="center", color="white", fontsize=7.5)
  ax.set_xticks(x)
  ax.set_xticklabels([lab[a] for a in arms], fontsize=7)
  ax.set_ylim(0, 1.05)
  ax.set_title("Task success (same variants)", fontsize=9, loc="left")
  ax = axes[1]
  ax.bar(x - 0.2, [S[a]["model_calls"] for a in arms], 0.38, color=COL[1], label="model calls")
  ax.bar(x + 0.2, [S[a]["prefills"] for a in arms], 0.38, color=COL[2], label="prefilled steps")
  ax.set_xticks(x)
  ax.set_xticklabels([lab[a] for a in arms], fontsize=7)
  ax.legend(fontsize=7)
  ax.set_title("Model calls vs. prefilled steps", fontsize=9, loc="left")
  ax = axes[2]
  ax.bar(x, [S[a]["rejected"] for a in arms], color=COL[7])
  ax.set_xticks(x)
  ax.set_xticklabels([lab[a] for a in arms], fontsize=7)
  ax.set_title("Prefills rejected by verification", fontsize=9, loc="left")
  fig.tight_layout()
  save(fig, "fig10_closed_loop_prefill")


if __name__ == "__main__":
  for f in (fig1, fig2, fig3, fig4, fig5, fig6_7, fig8, fig9_stage, fig10_closed_loop):
    try:
      f()
      print("ok", f.__name__)
    except Exception as e:  # pylint: disable=broad-except
      import traceback
      traceback.print_exc()
      print("FAIL", f.__name__, e)

