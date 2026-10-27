"""Figures for the paper's 'Understanding MobileExplorer's performance' section (e2e116n: BASE vs FULL45, paired)."""
import json, collections, statistics as st, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.argv = ["x", "e2e116n"]
exec(open("scripts/understand_perf.py").read().split("out = {}")[0])
OUT = "figures/understand"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
C_BASE, C_ME, C3, C4, C5 = "#8a8f96", "#1d6f78", "#7fb8bd", "#d9a441", "#c9d2d6"
W, H = 3.33, 1.9

def decomp(arm, t):
  S = [s for s in steps(arm, t) if s.get("pre")]
  D = collections.Counter()
  for i, s in enumerate(S):
    nxt = S[i + 1]["pre"]["t"] if i + 1 < len(S) else None
    if s["kind"] == "model":
      D["decide"] += s["t_decision"] - s["pre"]["t"]
      if nxt: D["after"] += nxt - s["t_decision"]
    elif nxt:
      D["graph"] += nxt - s["pre"]["t"]
  D["other"] = rows[(t, arm)]["e2e_s"] - sum(D.values())
  return D
M = {}
for arm in ("BASE", "FULL45"):
  A = collections.Counter()
  for t in both: A.update(decomp(arm, t))
  M[arm] = {k: v / len(both) for k, v in A.items()}
  M[arm]["calls"] = st.mean(sum(s["kind"] == "model" for s in steps(arm, t)) for t in both)
summary = {"n_both": len(both), "decomp": M}

# (a) latency breakdown
fig, ax = plt.subplots(figsize=(W, H))
segs = [("decide", "Model decision", C_ME), ("after", "Action execution & UI wait", C3), ("graph", "Graph execution", C4),
        ("other", "Setup & finish", C5)]
names = ["Baseline", "MobileExplorer"]
for i, arm in enumerate(("BASE", "FULL45")):
  left = 0
  for k, lab, c in segs:
    v = M[arm].get(k, 0)
    ax.barh(i, v, left=left, color=c, height=0.55,
            edgecolor="white", linewidth=0.8, label=lab if i == 1 else None)
    if v > 8: ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", color="white" if k == "decide" else "#1b2129", fontsize=7)
    left += v
  ax.text(left + 2, i, f"{left:.1f} s", va="center", fontsize=7.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(names); ax.invert_yaxis()
ax.set_xlabel("End-to-end latency per task (s)"); ax.set_xlim(0, 150)
ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.32))
fig.tight_layout(); fig.savefig(f"{OUT}/a_latency_breakdown.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/a_latency_breakdown.png", dpi=250, bbox_inches="tight")

# (b) attribution waterfall
b, f = M["BASE"], M["FULL45"]
loop_b = (b["decide"] + b["after"]) / b["calls"]; loop_f = (f["decide"] + f["after"]) / f["calls"]
fewer = (b["calls"] - f["calls"]) * loop_b - f.get("graph", 0)
faster = f["calls"] * (loop_b - loop_f)
tot_b, tot_f = sum(b[k] for k in ("decide", "after", "graph", "other") if k in b), sum(f[k] for k in ("decide", "after", "graph", "other") if k in f)
resid = tot_b - fewer - faster - tot_f
summary["attribution"] = {"loop_base": loop_b, "loop_me": loop_f, "fewer_calls_net": fewer, "shorter_loop": faster, "residual": resid,
                          "total_base": tot_b, "total_me": tot_f}
fig, ax = plt.subplots(figsize=(W, H))
labels = ["Baseline", "Fewer\nmodel calls", "Shorter\nsteps", "Mobile-\nExplorer"]
bot = [0, tot_b - fewer, tot_b - fewer - faster, 0]; hh = [tot_b, fewer, faster, tot_f]; cols = [C_BASE, C_ME, C3, C_ME]
txts = [f"{tot_b:.1f}", f"−{fewer:.1f}", f"−{faster:.1f}", f"{tot_f:.1f}"]
for i in range(4):
  ax.bar(i, hh[i], bottom=bot[i], color=cols[i], width=0.6, edgecolor="white")
  ax.text(i, bot[i] + hh[i] + 2, txts[i], ha="center", fontsize=7)
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7); ax.set_ylabel("Latency per task (s)"); ax.set_ylim(0, 150)
fig.tight_layout(); fig.savefig(f"{OUT}/b_attribution.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/b_attribution.png", dpi=250, bbox_inches="tight")

# (c) graph-executed steps by type (all 116 tasks)
kinds = collections.Counter(s["kind"] for t in tasks for s in steps("FULL45", t) if s["kind"] != "model")
names_k = {"bootstrap": "App launch", "form_function": "Form fill", "graph_reuse": "Navigation edge", "function": "Param. procedure",
           "confirm": "Confirm after commit", "unstick": "Unstick (Back)", "graph_macro": "Navigation macro"}
items = sorted(kinds.items(), key=lambda x: -x[1])
summary["graph_kinds"] = dict(items)
fig, ax = plt.subplots(figsize=(W, H))
ax.barh([names_k.get(k, k) for k, _ in items], [v for _, v in items], color=C_ME, height=0.6)
for i, (_, v) in enumerate(items): ax.text(v + 1.5, i, str(v), va="center", fontsize=7)
ax.invert_yaxis(); ax.set_xlabel(f"Graph-executed steps (total {sum(kinds.values())})")
fig.tight_layout(); fig.savefig(f"{OUT}/c_graph_types.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/c_graph_types.png", dpi=250, bbox_inches="tight")

# (d) exploration window vs decision time
Wn, Dn = [], []
for t in tasks:
  S = {s["step"]: s for s in steps("FULL45", t) if s["kind"] == "model"}
  for w in jl("FULL45", t, "windows"):
    s = S.get(w["step"])
    if s is None or not w.get("probes"): continue
    Wn.append(w["window_s"]); Dn.append(s.get("model_latency_s") or 0)
Wn, Dn = np.array(Wn), np.array(Dn)
summary["explore_window"] = {"n": int(len(Wn)), "window_mean": float(Wn.mean()), "decision_mean": float(Dn.mean()),
                             "overrun": int((Wn > Dn).sum()), "util_median": float(np.median(Wn / Dn))}
fig, ax = plt.subplots(figsize=(W, H))
for arr, lab, c in ((Wn, "Exploration window (with probes)", C_ME), (Dn, "Model decision time", C_BASE)):
  x = np.sort(arr); ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=1.4, label=lab)
ax.set_xlabel("Time within one reasoning step (s)"); ax.set_ylabel("CDF"); ax.set_xlim(0, 30)
ax.legend(frameon=False, fontsize=6.5, loc="lower right")
ax.text(18.5, 0.55, f"window > decision: {(Wn > Dn).sum()}/{len(Wn)}", fontsize=6.8)
fig.tight_layout(); fig.savefig(f"{OUT}/d_explore_window.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/d_explore_window.png", dpi=250, bbox_inches="tight")

# (e) savings vs position of the task within its app group (both succeeded)
import re
def app(t):
  m = re.match(r"(Simple[A-Z][a-z]+|System|Expense|Markor|Recipe|Files|Contacts|Notes|Tasks|Retro|Vlc|OsmAnd|Browser|Camera|Clock|AudioRecorder|OpenApp|SaveCopy|Sports|Turn)", t)
  return m.group(1) if m else t
pos, seen = {}, collections.Counter()
for t in order:
  pos[t] = seen[app(t)]; seen[app(t)] += 1
grp = collections.defaultdict(list)
for t in both:
  g = "1st" if pos[t] == 0 else ("2nd–3rd" if pos[t] <= 2 else "4th+")
  grp[g].append((rows[(t, "BASE")]["e2e_s"], rows[(t, "FULL45")]["e2e_s"], rows[(t, "BASE")]["calls"], rows[(t, "FULL45")]["calls"]))
G = ["1st", "2nd–3rd", "4th+"]
summary["by_position"] = {g: {"n": len(grp[g]), "e2e": [st.mean(x[0] for x in grp[g]), st.mean(x[1] for x in grp[g])],
                              "calls": [st.mean(x[2] for x in grp[g]), st.mean(x[3] for x in grp[g])]} for g in G}
fig, ax = plt.subplots(figsize=(W, H))
x = np.arange(3)
bb = [summary["by_position"][g]["e2e"][0] for g in G]; mm = [summary["by_position"][g]["e2e"][1] for g in G]
ax.bar(x - 0.18, bb, 0.34, color=C_BASE, label="Baseline"); ax.bar(x + 0.18, mm, 0.34, color=C_ME, label="MobileExplorer")
for i in range(3): ax.text(x[i] + 0.18, mm[i] + 3, f"−{(1 - mm[i] / bb[i]) * 100:.0f}%", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f"{g} task\n(n={len(grp[g])})" for g in G]); ax.set_ylabel("E2E latency (s)")
ax.set_xlabel("Position of the task among tasks of the same app"); ax.legend(frameon=False, fontsize=6.5); ax.set_ylim(0, 170)
fig.tight_layout(); fig.savefig(f"{OUT}/e_by_position.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/e_by_position.png", dpi=250, bbox_inches="tight")

# (f) rollback outcomes per probe
P = [p for t in tasks for p in jl("FULL45", t, "probes")]
oc = collections.defaultdict(lambda: [0, 0])
for p in P: oc[p.get("outcome")][0 if p.get("rollback_ok") else 1] += 1
names_o = {"new_screen": "New screen", "no_change": "No change", "left_app": "Left app", "minor_change": "Minor change",
           "in_place_content_change": "In-place change"}
items = sorted(oc.items(), key=lambda x: -sum(x[1]))
summary["rollback"] = {k: v for k, v in items}
fig, ax = plt.subplots(figsize=(W, H))
yl = [names_o.get(k, k) for k, _ in items]
ok = [v[0] for _, v in items]; bad = [v[1] for _, v in items]
ax.barh(yl, ok, color=C_ME, height=0.6, label="Recovered"); ax.barh(yl, bad, left=ok, color=C4, height=0.6, label="Recovery failed")
for i in range(len(ok)): ax.text(ok[i] + bad[i] + 1.5, i, f"{ok[i]}/{ok[i] + bad[i]}", va="center", fontsize=7)
ax.invert_yaxis(); ax.set_xlabel(f"Probes by outcome (recovered {sum(ok)}/{len(P)})"); ax.legend(frameon=False, fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/f_rollback.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/f_rollback.png", dpi=250, bbox_inches="tight")

json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps(summary, indent=1, ensure_ascii=False, default=float))
