"""Paper figures for the remaining 'Understanding MobileExplorer' results (same style as fig_understand.py).

i_graph_precision   graph-executed actions that the baseline's own successful trajectory also contains
j_budget            exploration window / decision time, Orin vs a fast server
k_prompt            prompt tokens and time to first token per call (no evidence is injected)
l_safety            candidate filtering and probe recovery
m_graph_growth      Mix-Trajectory Graph size over the task sequence, saved calls vs graph size
n_system_layers     streamed early action (e2e) and decoding configurations (decision latency)
o_seen_split        seen-task split: calls per task and the graph executions behind them
"""
import collections
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "understand"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
C_BASE, C_ME, C3, C4, C5 = "#8a8f96", "#1d6f78", "#7fb8bd", "#d9a441", "#c9d2d6"
W, H = 3.33, 1.9
X = json.load(open(OUT / "extra_summary.json"))


def save(fig, name):
  fig.tight_layout()
  for e in ("pdf", "png"):
    fig.savefig(OUT / f"{name}.{e}", bbox_inches="tight", dpi=300)
  plt.close(fig)


def rows_of(tag, arms):
  rows = {}
  for l in open(ROOT / "reports" / f"{tag}_runs.jsonl"):
    r = json.loads(l)
    if r["arm"] not in arms:
      continue
    if r.get("error"):
      rows.pop((r["task"], r["arm"]), None)
    else:
      rows[(r["task"], r["arm"])] = r
  return rows


def steps(tag, arm, t):
  return [json.loads(l) for l in open(ROOT / "raw" / tag / arm / t / "p0" / "steps.jsonl")]


# ------------------------------------------------------------------ i: graph-execution precision
rows = rows_of("e2e116n", ["BASE", "FULL45"])
agree = collections.Counter()
for (t, a), r in rows.items():
  if a != "FULL45" or not rows.get((t, "BASE"), {}).get("success"):
    continue
  base = steps("e2e116n", "BASE", t)
  btok = {s.get("tok") for s in base if s["kind"] == "model"}
  btype = any((s.get("action") or {}).get("action_type") == "input_text" for s in base)
  for s in steps("e2e116n", "FULL45", t):
    if s["kind"] == "model":
      continue
    tok = s.get("tok") or ""
    ok = btype if tok.startswith("TYPE") else any(str(x).startswith("OPEN_APP") for x in btok) if tok.startswith("OPEN_APP") else tok in btok
    agree[(s["kind"], "n")] += 1
    agree[(s["kind"], "ok")] += ok
names = {"bootstrap": "App launch", "form_function": "Form fill", "function": "Param. procedure", "confirm": "Confirm after commit",
         "graph_reuse": "Navigation edge"}
ks = [k for k in ("bootstrap", "form_function", "function", "confirm") if agree[(k, "n")]]
fig, ax = plt.subplots(figsize=(W, 1.6))
for i, k in enumerate(ks):
  n, ok = agree[(k, "n")], agree[(k, "ok")]
  ax.barh(i, ok / n * 100, color=C_ME, height=0.6)
  ax.barh(i, 100 - ok / n * 100, left=ok / n * 100, color=C5, height=0.6)
  ax.text(101.5, i, f"{ok}/{n}", va="center", fontsize=7, color="#333")
tot_n = sum(agree[(k, "n")] for k in ks)
tot_ok = sum(agree[(k, "ok")] for k in ks)
ax.set_yticks(range(len(ks)))
ax.set_yticklabels([names[k] for k in ks])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Actions also taken by the baseline (%)")
ax.set_title(f"Graph-executed actions: {tot_ok}/{tot_n} = {tot_ok / tot_n:.0%} match", fontsize=7.5)
save(fig, "i_graph_precision")


# ------------------------------------------------------------------ j: budget adaptation
def ratios(tag):
  R = []
  for d in glob.glob(str(ROOT / "raw" / tag / "FULL45" / "*" / "p0")):
    S = {}
    for l in open(d + "/steps.jsonl"):
      s = json.loads(l)
      if s["kind"] == "model":
        S[s["step"]] = s
    for l in open(d + "/windows.jsonl"):
      w = json.loads(l)
      s = S.get(w["step"])
      if s and w.get("probes") and s.get("model_latency_s"):
        R.append(w.get("window_s", 0) / s["model_latency_s"])
  return sorted(R)


fig, ax = plt.subplots(figsize=(W, H))
for tag, lab, c, dec in (("e2e116n", "Jetson AGX Orin", C_ME, X["N2_budget"]["orin_e2e116n"]),
                         ("seenE2", "RTX 4090 server", C4, X["N2_budget"]["labserver_seenE2"])):
  R = ratios(tag)
  over = sum(r > 1 for r in R)
  ax.plot(R, [(i + 1) / len(R) for i in range(len(R))], color=c, lw=2,
          label=f"{lab}: decision {dec['decision_s_mean']:.1f} s, window > decision {over}/{len(R)}")
ax.axvline(1.0, color="#555", lw=0.8, ls=":")
ax.text(1.03, 0.05, "window = decision", fontsize=6.5, color="#555")
ax.set_xscale("log")
ax.set_xlabel("Exploration window / decision time")
ax.set_ylabel("CDF")
ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.38), ncol=1)
save(fig, "j_budget")

# ------------------------------------------------------------------ k: prompt length
P = X["N3_prompt"]
fig, axs = plt.subplots(1, 2, figsize=(W, 1.6))
for ax, key, lab, fmt in ((axs[0], "prompt_tokens_mean", "Prompt tokens / call", "{:.0f}"),
                          (axs[1], "ttft_s_mean", "Time to first token (s)", "{:.2f}")):
  v = [P["BASE"][key], P["FULL45"][key]]
  ax.bar([0, 1], v, color=[C_BASE, C_ME], width=0.6)
  for i, x in enumerate(v):
    ax.text(i, x, fmt.format(x), ha="center", va="bottom", fontsize=7)
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Baseline", "Mobile-\nExplorer"], fontsize=7)
  ax.set_ylabel(lab)
save(fig, "k_prompt")

# ------------------------------------------------------------------ l: safety filtering
S4 = X["N4_safety"]
cd = S4["candidates_dropped"]
fig, ax = plt.subplots(figsize=(W, 1.5))
bars = [("Filtered: low predicted\nreversibility", cd["p_rev_dropped"], C5),
        ("Filtered: risky control\n(toggle / mode / data)", cd["risky_dropped"], C5)]
for i, (lab, v, c) in enumerate(bars):
  ax.barh(i, v, color=c, height=0.6)
  ax.text(v + 20, i, f"{v}", va="center", fontsize=7)
rec = S4["probes"] - S4["rollback_fail"]
ax.barh(2, rec, color=C_ME, height=0.6, label="Recovered")
ax.barh(2, S4["rollback_fail"], left=rec, color=C4, height=0.6, label="Recovery failed")
ax.text(S4["probes"] + 20, 2, f"{S4['probes']} ({rec} recovered)", va="center", fontsize=7)
ax.set_yticks(range(3))
ax.set_yticklabels([b[0] for b in bars] + ["Probed"], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Candidate controls (116 tasks)")
ax.legend(fontsize=6, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.45))
save(fig, "l_safety")

# ------------------------------------------------------------------ m: graph growth
order = open(ROOT / "reports" / "e2e116n_task_order.txt").read().strip().split(",")
G = ROOT / "graph_snapshots" / "e2e116n" / "FULL45"
first, nodes, edges, cur = {}, set(), set(), []
for l in open(G / "graph.jsonl"):
  e = json.loads(l)
  t = e.get("instance", "").split("/")[0]
  if t not in first:
    first[t] = (len(nodes), len(edges))
  src = ((e.get("src") or {}).get("activity"), (e.get("src") or {}).get("layout_sig"))
  dst = ((e.get("dst") or {}).get("activity"), (e.get("dst") or {}).get("layout_sig"))
  nodes.update([src, dst])
  edges.add((src, e.get("slot_id"), dst))
xs, yn, ye = [], [], []
for i, t in enumerate(order):
  if t in first:
    xs.append(i + 1)
    yn.append(first[t][0])
    ye.append(first[t][1])
xs.append(len(order))
yn.append(len(nodes))
ye.append(len(edges))
fig, axs = plt.subplots(1, 2, figsize=(6.9, 1.9))
axs[0].plot(xs, ye, color=C_ME, lw=2)
axs[0].plot(xs, yn, color=C4, lw=2)
axs[0].text(xs[-1], ye[-1], f" {ye[-1]} edges", va="center", fontsize=7, color="#333")
axs[0].text(xs[-1], yn[-1], f" {yn[-1]} nodes", va="center", fontsize=7, color="#333")
axs[0].set_xlabel("Task index in the run")
axs[0].set_ylabel("Graph size")
axs[0].set_xlim(0, len(order) + 22)
# right: calls on successful tasks by the task's position within its app group (what earlier tasks of the same app
# could have put into the graph). Each arm averages over its own successful tasks.
pkg_of = collections.defaultdict(collections.Counter)
for l in open(G / "graph.jsonl"):
  e = json.loads(l)
  p = (e.get("src") or {}).get("package")
  if p and "launcher" not in p:
    pkg_of[e.get("instance", "").split("/")[0]][p] += 1
app = {t: c.most_common(1)[0][0] for t, c in pkg_of.items()}
pos, seen_n = {}, collections.Counter()
for t in order:
  if t in app:
    pos[t] = seen_n[app[t]]
    seen_n[app[t]] += 1
last = {}
for l in open(ROOT / "reports" / "e2e116n_runs.jsonl"):
  r = json.loads(l)
  if r.get("arm") in ("BASE", "FULL45"):
    last[(r["task"], r["arm"])] = r
Tm = [t for t in order if t in pos and (t, "BASE") in last and (t, "FULL45") in last]
bins_m = [("1st", lambda v: v == 0), ("2nd-3rd", lambda v: 1 <= v <= 2), ("4th-6th", lambda v: 3 <= v <= 5), ("7th+", lambda v: v >= 6)]
wm = 0.38
ns_m = {}
for j, (arm, col, lab) in enumerate((("BASE", C_BASE, "Agent only"), ("FULL45", C_ME, "MobileExplorer"))):
  vals, ns = [], []
  for _, f in bins_m:
    v = [last[(t, arm)]["calls"] for t in Tm if f(pos[t]) and last[(t, arm)]["success"]]
    vals.append(st.mean(v) if v else 0.0)
    ns.append(len(v))
  xs_ = [i + (j - 0.5) * wm for i in range(len(bins_m))]
  axs[1].bar(xs_, vals, width=wm, color=col, label=lab, edgecolor="white", linewidth=1)
  ns_m[arm] = ns
  for x_, v_ in zip(xs_, vals):
    axs[1].text(x_, v_, f"{v_:.1f}", ha="center", va="bottom", fontsize=6)
axs[1].set_xticks(range(len(bins_m)))
axs[1].set_xticklabels([f"{b[0]}\nin app\n(n={ns_m['BASE'][i]}/{ns_m['FULL45'][i]})" for i, b in enumerate(bins_m)],
                       fontsize=6.5)
axs[1].set_xlabel("Task position within its app group", fontsize=7)
axs[1].set_ylabel("Model calls per\nsuccessful task")
axs[1].set_ylim(0, 11.5)
axs[1].legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2)
save(fig, "m_graph_growth")

# ------------------------------------------------------------------ n: system layers
r10 = [json.loads(l) for l in open(ROOT / "reports" / "e2e10_runs.jsonl")]
e10 = {a: st.mean(r["e2e_s"] for r in r10 if r["arm"] == a and not r.get("error")) for a in ("BASE", "BASE+S")}
sd = {}
for f, lab in (("specdec_baseline", "Default"), ("specdec_ngram5", "n-gram 5"), ("specdec_ngram8", "n-gram 8"),
               ("specdec_ngram8_graph", "n-gram 8\n+ CUDA graph")):
  R = [json.loads(l) for l in open(ROOT / "reports" / f"{f}.jsonl")]
  sd[lab] = (st.mean(r["t"] for r in R), st.mean(r["tokens"] / max(1e-6, r["t"] - r["ttft"]) for r in R))
fig, axs = plt.subplots(1, 2, figsize=(6.9, 1.9))
v = [e10["BASE"], e10["BASE+S"]]
axs[0].bar([0, 1], v, color=[C_BASE, C_ME], width=0.6)
for i, x in enumerate(v):
  axs[0].text(i, x, f"{x:.0f}", ha="center", va="bottom", fontsize=7)
axs[0].text(1, v[1] / 2, f"{(v[1] / v[0] - 1):+.0%}", ha="center", color="white", fontsize=7)
axs[0].set_xticks([0, 1])
axs[0].set_xticklabels(["Wait for full\ndecoding", "Streamed early\naction"], fontsize=7)
axs[0].set_ylabel("E2E latency per task (s)\n10 tasks, same calls")
labs = list(sd)
axs[1].bar(range(len(labs)), [sd[k][0] for k in labs], color=[C_BASE, C3, C3, C_ME], width=0.6)
for i, k in enumerate(labs):
  axs[1].text(i, sd[k][0], f"{sd[k][0]:.1f} s\n{sd[k][1]:.1f} tok/s", ha="center", va="bottom", fontsize=6.5)
axs[1].set_xticks(range(len(labs)))
axs[1].set_xticklabels(labs, fontsize=7)
axs[1].set_ylabel("Decision latency (s)")
axs[1].set_ylim(0, max(sd[k][0] for k in labs) * 1.35)
save(fig, "n_system_layers")

# ------------------------------------------------------------------ o: seen-task split
sp = json.load(open(ROOT / "reports" / "seenW1_split.json"))
rs = rows_of("seenE2", ["FULL45", "FULL45+SEENR"])
A = ["FULL45", "FULL45+SEENR"]
groups = []          # (label, tasks, succeeded_only): succeeded groups average each arm over its own successes
seen_both = []
for name, g in (("Seen", sp["seen"]), ("Unseen", sp["unseen"])):
  T = [t for t in g if all((t, a) in rs for a in A)]
  B = [t for t in T if all(rs[(t, a)]["success"] for a in A)]
  if name == "Seen":
    seen_both = B
  ns = [sum(1 for t in T if rs[(t, a)]["success"]) for a in A]
  groups.append((f"{name}\nall\n(n={len(T)})", T, False))
  groups.append((f"{name}\nsucceeded\n(n={ns[0]} / {ns[1]})", T, True))


def calls_mean(T, a, succ_only):
  vals = [rs[(t, a)]["calls"] for t in T if not succ_only or rs[(t, a)]["success"]]
  return st.mean(vals) if vals else 0.0


fig, ax = plt.subplots(figsize=(W, 2.1))
w = 0.38
for j, (a, c, lab) in enumerate(((A[0], C_BASE, "Cold graph"), (A[1], C_ME, "+ seen-task replay"))):
  v = [calls_mean(T, a, so) for _, T, so in groups]
  ax.bar([i + (j - 0.5) * w for i in range(len(groups))], v, width=w, color=c, label=lab, edgecolor="white", linewidth=1)
  for i, x in enumerate(v):
    ax.text(i + (j - 0.5) * w, x, f"{x:.1f}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=6)
ax.set_ylabel("Model calls per task")
ax.set_ylim(0, 13)
ax.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=2)
save(fig, "o_seen_split")
# graph executions behind the seen both-succeed savings
T = seen_both
kinds = [("tpl_replay", "Trajectory replay", C4), ("jump", "Recorded-path jump", C3), ("function", "Procedure / form", C_ME),
         ("other", "Launch / confirm / other", C5)]


def execs(a, t):
  c, seen = collections.Counter(), set()
  for s in steps("seenE2", a, t):
    k = s["kind"]
    if k == "model":
      continue
    k = "function" if k in ("function", "form_function") else k if k in ("tpl_replay", "jump") else "other"
    if (k, s["step"]) not in seen:
      seen.add((k, s["step"]))
      c[k] += 1
  return c


fig, ax = plt.subplots(figsize=(W, 2.0))
for i, a in enumerate(A):
  c = collections.Counter()
  for t in T:
    c.update(execs(a, t))
  left = 0
  for k, lab, col in kinds:
    v = c[k] / len(T)
    ax.barh(i, v, left=left, color=col, height=0.55, label=lab if i == 0 else None, edgecolor="white", linewidth=1)
    left += v
  calls = st.mean(rs[(t, a)]["calls"] for t in T)
  ax.text(left + 0.05, i, f"{left:.2f} execs, {calls:.1f} calls", va="center", fontsize=6.5)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Cold graph", "+ seen-task\nreplay"], fontsize=7)
ax.invert_yaxis()
ax.set_xlim(0, 4.2)
ax.set_xlabel("Graph executions per task")
ax.set_title(f"Seen tasks both variants solve (n={len(T)})", fontsize=7.5)
fig.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=2)
fig.subplots_adjust(left=0.24, right=0.97, top=0.86, bottom=0.42)
for e in ("pdf", "png"):
  fig.savefig(OUT / f"p_seen_mechanism.{e}", dpi=300)
plt.close(fig)
print("ok")
