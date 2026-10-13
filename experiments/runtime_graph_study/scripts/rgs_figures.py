"""Figures from analysis/*.csv (+ raw evidence for screenshot panels)."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rgs_ui as U  # noqa: E402
import rgs_analyze as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "analysis")
FIG = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "figures")
FIG.mkdir(parents=True, exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def rd(name):
  p = AN / name
  if not p.exists() or p.stat().st_size == 0:
    return []
  return list(csv.DictReader(p.open()))


def b(x):
  return str(x) == "True"


def f(x, d=0.0):
  try:
    return float(x)
  except (TypeError, ValueError):
    return d


def wilson(k, n, z=1.96):
  if n == 0:
    return (0, 0)
  p = k / n
  den = 1 + z * z / n
  c = (p + z * z / (2 * n)) / den
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return (max(0, c - h), min(1, c + h))


def save(fig, name):
  fig.savefig(FIG / f"{name}.png", dpi=200)
  fig.savefig(FIG / f"{name}.pdf")
  plt.close(fig)


APP = {}
for m in ("configs/batch1_manifest.json", "configs/batch2_manifest.json"):
  p = ROOT / m
  if p.exists():
    for t in json.loads(p.read_text())["tasks"]:
      APP[t["task"]] = t["app"]


# ---------------------------------------------------------------- Fig 1
def fig1(eps):
  tasks = sorted({e["task"] for e in eps})
  nvar = len({(e["task"], e["variant"]) for e in eps if e["mode"] == "agent" and not e["variant"].startswith("v0_ref")})
  fig, ax = plt.subplots(figsize=(10, 3.3))
  ax.axis("off")
  ax.grid(False)
  boxes = [
      (0.0, "AndroidWorld task\n× seeded variants", f"{len(tasks)} tasks\n{nvar} runtime instances"),
      (0.205, "Clean reference run", "GELAB-ZERO-4B (LabServer)\nno exploration"),
      (0.41, "Agent run + inference-\nwindow exploration", "model call in background;\nprobe → observe → rollback"),
      (0.615, "Replay reference\non other variants", "strict (coordinates)\nstructure-aware (selectors)"),
      (0.82, "Offline analysis", "step alignment, screenshot /\nA11y / transition diffs,\ngraph rebuild + ablations"),
  ]
  for x, t, s in boxes:
    ax.add_patch(FancyBboxPatch((x, 0.35), 0.18, 0.5, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#f3f2ef", ec=INK2, lw=0.8))
    ax.text(x + 0.09, 0.73, t, ha="center", va="center", fontsize=9, weight="bold", color=INK)
    ax.text(x + 0.09, 0.5, s, ha="center", va="center", fontsize=7.5, color=INK2)
  for x in (0.18, 0.385, 0.59, 0.795):
    ax.annotate("", xy=(x + 0.025, 0.6), xytext=(x, 0.6), arrowprops=dict(arrowstyle="->", color=INK2))
  ax.add_patch(FancyBboxPatch((0.41, 0.0), 0.385, 0.22, boxstyle="round,pad=0.01,rounding_size=0.02",
                              fc="white", ec=C[0], lw=0.9))
  ax.text(0.6025, 0.11, "Persistent observation graph (append-only raw transitions; rebuilt offline\n"
          "under exact / structural / layout matching, shared vs per-run memory)",
          ha="center", va="center", fontsize=7.5, color=INK)
  ax.annotate("", xy=(0.5, 0.22), xytext=(0.5, 0.35), arrowprops=dict(arrowstyle="<->", color=C[0]))
  ax.annotate("", xy=(0.7, 0.22), xytext=(0.7, 0.35), arrowprops=dict(arrowstyle="->", color=C[0]))
  ax.set_xlim(-0.01, 1.01)
  ax.set_ylim(-0.02, 0.9)
  save(fig, "fig1_overview")


# ---------------------------------------------------------------- Fig 2
def fig2(pairs, rsteps):
  """Variation rates at four levels, per app, over corresponding steps.

  Corresponding steps = replay decisions (ref pre-state vs live pre-state at
  the same reference step, state matched at struct or layout level)."""
  rows = [r for r in rsteps if r["mode"] == "replay_struct" and r.get("state_match") in ("struct", "layout")
          and r.get("pix_diff_ratio") not in (None, "")]
  by = defaultdict(list)
  for r in rows:
    by[APP.get(r["task"], r["task"])].append(r)
  apps = sorted(by)
  levels = [
      ("Screenshot\n(>0.1% pixels)", lambda r: b(r["pix_changed"])),
      ("A11y attributes\n(exact sig)", lambda r: not b(r["a11y_exact_equal"])),
      ("Actionable set\n(clickables+labels)", lambda r: b(r["a11y_clickable_set_changed"])),
      ("List / field content\n(content sig)", lambda r: b(r.get("content_changed"))),
  ]
  outs = rd("replay_outcomes.csv")
  ob = defaultdict(list)
  for o in outs:
    if o["mode"] == "replay_struct":
      ob[APP.get(o["task"], o["task"])].append(o)
  fig, ax = plt.subplots(figsize=(10, 3.6))
  w = 0.17
  x = np.arange(len(apps) + 1)
  labels = apps + ["Pooled"]
  for i, (name, fn) in enumerate(levels + [("Transition outcome\n(≠ reference struct)", None)]):
    vals, errs, ns = [], [], []
    for a in labels:
      if fn is None:
        src = sum((ob[k] for k in apps), []) if a == "Pooled" else ob[a]
        k = sum(1 for o in src if o["category"] not in ("same_exact", "same_struct_dynamic_content"))
      else:
        src = rows if a == "Pooled" else by[a]
        k = sum(1 for r in src if fn(r))
      n = len(src)
      p = k / n if n else 0
      lo, hi = wilson(k, n)
      vals.append(p)
      errs.append((p - lo, hi - p))
      ns.append(n)
    ax.bar(x + (i - 2) * w, vals, w * 0.92, color=C[i], label=name.replace("\n", " "),
           yerr=np.array(errs).T, error_kw=dict(lw=0.6, capsize=1.5, ecolor=INK2))
  for j, a in enumerate(labels):
    n = len(rows if a == "Pooled" else by[a])
    ax.text(x[j], 1.0, f"n={n}", ha="center", fontsize=7, color=INK2, transform=ax.get_xaxis_transform())
  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
  ax.set_ylabel("Fraction of corresponding steps")
  ax.set_ylim(0, 1.25)
  ax.legend(ncol=3, fontsize=7, loc="upper left")
  ax.set_title("Runtime variation at corresponding steps (reference vs. variant, structure-aware replay)",
               fontsize=9, loc="left")
  save(fig, "fig2_variation_breakdown")


# ---------------------------------------------------------------- Fig 4
CAUSE_ORDER = ["completed", "expected_parameter_divergence", "coordinate_drift", "target_moved",
               "target_not_on_screen", "duplicate_targets", "same_target_other_context", "missing_target",
               "ambiguous_target", "changed_app_data", "unexpected_dialog_or_overlay",
               "different_destination_screen", "changed_ui_hierarchy", "keyboard_state",
               "different_app_or_screen", "unknown"]


def fig4(eps, rsteps):
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.1, 1]})
  ax = axes[0]
  modes = [("replay_strict", "Strict replay"), ("replay_struct", "Structure-aware replay")]
  refined = {(r["task"], r["variant"], r["mode"]): (r.get("cause_refined") or r.get("cause") or "unknown")
             for r in rsteps if r["execute"] != "True"}
  cnt = {}
  for m, _ in modes:
    c = Counter()
    for e in eps:
      if e["mode"] != m:
        continue
      c[refined.get((e["task"], e["variant"], m), "completed")] += 1
    cnt[m] = c
  extra = sorted({k for m, _ in modes for k in cnt[m]} - set(CAUSE_ORDER))
  causes = [k for k in CAUSE_ORDER + extra if any(cnt[m][k] for m, _ in modes)]
  left = np.zeros(2)
  for i, k in enumerate(causes):
    v = np.array([cnt[m][k] for m, _ in modes], dtype=float)
    ax.barh([0, 1], v, left=left, color=(C + C)[i], edgecolor="white", lw=1, label=k.replace("_", " "))
    left += v
  ax.set_yticks([0, 1])
  ax.set_yticklabels([l for _, l in modes])
  ax.set_xlabel("Replay episodes (variants)")
  ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
  ax.set_title("Replay outcome / first-divergence cause", fontsize=9, loc="left")
  ax.grid(axis="y", visible=False)
  # success
  for i, (m, _) in enumerate(modes):
    es = [e for e in eps if e["mode"] == m]
    k = sum(1 for e in es if f(e.get("success_before_recovery") if (m == "replay_struct" and e.get("recovery")) else e["success"]) == 1.0)
    ax.text(left[i] + 0.3, i, f"task success w/o recovery {k}/{len(es)}", va="center", fontsize=7, color=INK2)
  ax = axes[1]
  for i, (m, lab) in enumerate(modes):
    pos = []
    for e in eps:
      if e["mode"] != m:
        continue
      dv = json.loads(e["divergence"]) if e.get("divergence") not in (None, "", "null") else None
      if dv and e.get("ref_steps"):
        pos.append(dv["ref_step"] / max(1, int(f(e["ref_steps"]))))
    if pos:
      ax.hist(pos, bins=np.linspace(0, 1, 11), alpha=0.75, color=C[i], label=f"{lab} (n={len(pos)})",
              edgecolor="white", lw=1)
  ax.set_xlabel("Divergence position (fraction of reference steps)")
  ax.set_ylabel("Episodes")
  ax.legend(fontsize=7)
  ax.set_title("Where replay diverges", fontsize=9, loc="left")
  fig.tight_layout()
  save(fig, "fig4_replay_robustness")


# ---------------------------------------------------------------- Fig 5
def fig5():
  outs = rd("replay_outcomes.csv")
  rev = rd("revalidations.csv")
  cats = ["same_exact", "same_struct_dynamic_content", "same_layout_ambiguous", "different_state",
          "no_observable_transition"]
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
  ax = axes[0]
  by = defaultdict(Counter)
  for o in outs:
    if o["mode"] == "replay_struct":
      by[APP.get(o["task"], o["task"])][o["category"]] += 1
  apps = sorted(by)
  left = np.zeros(len(apps))
  for i, c in enumerate(cats):
    v = np.array([by[a][c] for a in apps], dtype=float)
    tot = np.array([sum(by[a].values()) for a in apps], dtype=float)
    ax.barh(range(len(apps)), v / np.maximum(tot, 1), left=left, color=C[i], edgecolor="white", lw=1,
            label=c.replace("_", " "))
    left += v / np.maximum(tot, 1)
  ax.set_yticks(range(len(apps)))
  ax.set_yticklabels([f"{a} (n={sum(by[a].values())})" for a in apps], fontsize=7.5)
  ax.set_xlabel("Fraction of replayed (state, action) pairs")
  ax.set_title("Replayed (state, action) → outcome vs. reference", fontsize=9, loc="left")
  ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
  ax.grid(axis="y", visible=False)
  ax = axes[1]
  r = [x for x in rev if x["rule"] == "struct" and x["origin"] == "explore"]
  by2 = defaultdict(lambda: [0, 0])
  for x in r:
    a = APP.get(x["task"], x["task"])
    by2[a][0] += int(b(x["same_outcome"]))
    by2[a][1] += 1
  apps2 = sorted(by2)
  same = [by2[a][0] for a in apps2]
  diff = [by2[a][1] - by2[a][0] for a in apps2]
  ax.barh(range(len(apps2)), same, color=C[2], label="same outcome as prior run", edgecolor="white")
  ax.barh(range(len(apps2)), diff, left=same, color=C[1], label="new runtime outcome", edgecolor="white")
  ax.set_yticks(range(len(apps2)))
  ax.set_yticklabels(apps2, fontsize=7.5)
  ax.set_xlabel("Re-probed slots (explorer revalidation, struct matching)")
  ax.set_title("Cross-variant revalidation of known (state, slot)", fontsize=9, loc="left")
  ax.legend(fontsize=7)
  ax.grid(axis="y", visible=False)
  fig.tight_layout()
  save(fig, "fig5_transition_variation")


# ---------------------------------------------------------------- Fig 6
def fig6():
  g = [r for r in rd("graph_growth.csv") if r["shared"] == "True"]
  if not g:
    return
  tasks = sorted({r["task"] for r in g})
  n = len(tasks)
  cols = min(4, n)
  rows = int(np.ceil(n / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.1 * rows), squeeze=False)
  for k, t in enumerate(tasks):
    ax = axes[k // cols][k % cols]
    for i, rule in enumerate(["exact", "struct", "layout"]):
      rr = [r for r in g if r["task"] == t and r["rule"] == rule]
      ax.plot([int(r["i"]) for r in rr], [int(r["nodes"]) for r in rr], color=C[i], lw=1.4,
              label=f"nodes ({rule})")
    rr = [r for r in g if r["task"] == t and r["rule"] == "struct"]
    ax.plot([int(r["i"]) for r in rr], [int(r["observed_slots"]) for r in rr], color=C[3], lw=1.4, ls="--",
            label="observed slots (struct)")
    # variant boundaries
    prev = None
    for r in rr:
      if r["instance"] != prev:
        ax.axvline(int(r["i"]), color=GRID, lw=0.8, zorder=0)
        prev = r["instance"]
    ax.set_title(t, fontsize=7.5, loc="left")
    ax.tick_params(labelsize=6.5)
  for k in range(n, rows * cols):
    axes[k // cols][k % cols].axis("off")
  axes[0][0].legend(fontsize=6, loc="upper left")
  fig.supxlabel("Observations ingested (agent + exploration, run order; gray = new variant)", fontsize=8)
  fig.tight_layout()
  save(fig, "fig6_graph_growth")


# ---------------------------------------------------------------- Fig 8
def fig8():
  pr = rd("probes.csv")
  wi = rd("windows.csv")
  if not pr:
    return
  fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
  ax = axes[0]
  tt = np.array([f(p["t_total_s"]) for p in pr])
  ta = np.array([f(p["t_act_s"]) + f(p["t_observe_s"]) for p in pr])
  tr = np.array([f(p["t_rollback_s"]) for p in pr])
  ax.hist(tt, bins=30, color=C[0], alpha=0.8, label=f"probe total (median {np.median(tt):.2f}s)")
  ax.hist(tr[tr > 0], bins=30, color=C[1], alpha=0.7, label=f"rollback (median {np.median(tr[tr > 0]) if (tr > 0).any() else 0:.2f}s)")
  ax.set_xlabel("Seconds")
  ax.set_ylabel("Probes")
  ax.legend(fontsize=7)
  ax.set_title(f"Probe cost (n={len(pr)})", fontsize=9, loc="left")
  ax = axes[1]
  # yield vs window budget: probes completed within budget B, per window,
  # using measured per-probe durations in execution order.
  byw = defaultdict(list)
  for p in pr:
    byw[(p["ep"], p["step"])].append(p)
  budgets = [0.5, 1, 2, 3, 4, 6, 8, 10]
  for i, (label, keyf) in enumerate([("probes completed", None),
                                     ("new (state, slot) transitions", "discover")]):
    vals = []
    for B in budgets:
      tot = 0
      for k, lst in byw.items():
        lst = sorted(lst, key=lambda p: f(p["t"]))
        el = 0
        for p in lst:
          el += f(p["t_total_s"])
          if el > B:
            break
          if keyf is None or p["priority"] == keyf:
            tot += 1
      vals.append(tot / max(1, len([w for w in wi if int(f(w["n_probes"])) >= 0])))
    ax.plot(budgets, vals, marker="o", ms=4, color=C[i], lw=1.6, label=label)
  lat = [f(w["model_latency_s"]) for w in wi if w.get("model_latency_s")]
  if lat:
    ax.axvline(np.median(lat), color=C[4], ls=":", lw=1.2)
    ax.text(np.median(lat), ax.get_ylim()[1] * 0.97, f" LabServer latency\n (median {np.median(lat):.1f}s)", va="top",
            fontsize=6.5, color=INK2)
  ax.axvspan(8, 10, color=GRID, alpha=0.5, lw=0)
  ax.text(9, ax.get_ylim()[1] * 0.5, "projected\non-device\nwindow", fontsize=6.5, ha="center", color=INK2)
  ax.set_xlabel("Exploration budget per inference window (s)")
  ax.set_ylabel("Per window (mean)")
  ax.legend(fontsize=7, loc="lower right")
  ax.set_title("Exploration yield vs. budget (measured probe timings)", fontsize=9, loc="left")
  ax = axes[2]
  oc = Counter(p["outcome"] for p in pr)
  rb = Counter((p["rollback_method"], p["rollback_ok"]) for p in pr)
  labs = list(oc)
  ax.barh(range(len(labs)), [oc[k] for k in labs], color=C[0])
  ax.set_yticks(range(len(labs)))
  ax.set_yticklabels([k.replace("_", " ") for k in labs], fontsize=7.5)
  ok = sum(1 for p in pr if b(p["rollback_ok"]))
  lo, hi = wilson(ok, len(pr))
  ax.set_title(f"Probe outcomes; rollback verified {ok}/{len(pr)} ({ok/len(pr):.0%}, CI {lo:.0%}-{hi:.0%})",
               fontsize=8, loc="left")
  ax.set_xlabel("Probes")
  ax.grid(axis="y", visible=False)
  fig.tight_layout()
  save(fig, "fig8_exploration_budget")


# ---------------------------------------------------------------- Fig 9
ABBR = {"ContactsAddContact": "Contacts.Add", "ExpenseAddSingle": "Expense.Add", "ExpenseDeleteSingle": "Expense.Del",
        "MarkorCreateNote": "Markor.Create", "MarkorDeleteNewestNote": "Markor.DelNewest",
        "NotesTodoItemCount": "Joplin.TodoCnt", "OsmAndFavorite": "OsmAnd.Fav",
        "RecipeDeleteMultipleRecipesWithConstraint": "Recipe.DelMulti", "RecipeDeleteSingleRecipe": "Recipe.Del",
        "SimpleCalendarDeleteOneEvent": "Cal.DelEvent", "SimpleCalendarEventsOnDate": "Cal.EventsOn",
        "SportsTrackerLongestDistanceActivity": "Tracks.Longest", "SystemBluetoothTurnOn": "BT.On (ctrl)",
        "TasksDueOnDate": "Tasks.DueOn"}


def fig9():
  gm = json.loads((AN / "graph_metrics.json").read_text()) if (AN / "graph_metrics.json").exists() else {}
  if not gm:
    return
  tasks = sorted(gm)
  fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
  ax = axes[0]
  # node reuse of agent-visited states from prior runs, by rule (variants after the first)
  for i, rule in enumerate(["exact", "struct", "layout"]):
    vals = []
    for t in tasks:
      pi = gm[t][f"{rule}_shared"]["per_instance"]
      insts = gm[t]["instances"][1:]
      k = sum(pi.get(x, {}).get("agent_states_known_from_prior_runs", 0) for x in insts)
      n = sum(pi.get(x, {}).get("agent_states", 0) for x in insts)
      vals.append(k / n if n else 0)
    ax.bar(np.arange(len(tasks)) + (i - 1) * 0.27, vals, 0.25, color=C[i], label=rule)
  ax.set_xticks(range(len(tasks)))
  ax.set_xticklabels([ABBR.get(t, t) for t in tasks], rotation=60, ha="right", fontsize=6.5)
  ax.set_ylabel("Visited states already in graph")
  ax.set_ylim(0, 1.3)
  ax.set_title("Node reuse in later variants", fontsize=9, loc="left")
  ax.legend(fontsize=7, ncol=3, loc="upper left", title="matching rule", title_fontsize=7)
  ax = axes[1]
  sh = [sum(gm[t]["struct_shared"]["per_instance"].get(x, {}).get("new_nodes", 0) for x in gm[t]["instances"]) for t in tasks]
  pr_ = [sum(gm[t]["struct_per_run"]["per_instance"].get(x, {}).get("new_nodes", 0) for x in gm[t]["instances"]) for t in tasks]
  ax.bar(np.arange(len(tasks)) - 0.2, pr_, 0.38, color=C[1], label="per-run graphs (sum)")
  ax.bar(np.arange(len(tasks)) + 0.2, sh, 0.38, color=C[0], label="shared incremental graph")
  ax.set_xticks(range(len(tasks)))
  ax.set_xticklabels([ABBR.get(t, t) for t in tasks], rotation=60, ha="right", fontsize=6.5)
  ax.set_ylim(0, max(pr_) * 1.35)
  ax.set_ylabel("Node creations (struct rule)")
  ax.set_title("Node creations: per-run vs shared", fontsize=9, loc="left")
  ax.legend(fontsize=7, loc="upper right")
  ax = axes[2]
  wi = rd("windows.csv")
  disc = sum(int(f(w.get("plan_discover"))) for w in wi)
  reva = sum(int(f(w.get("plan_revalidate"))) for w in wi)
  skip = sum(int(f(w.get("plan_skip_known_same_instance"))) for w in wi)
  unr = sum(int(f(w.get("plan_skip_known_unrestorable"))) + int(f(w.get("plan_skip_family_unrestorable")))
            + int(f(w.get("plan_skip_app_unrestorable"))) for w in wi)
  vals = [disc, reva, skip, unr]
  labs = ["unexplored\n(discover)", "other variant\n(revalidate)", "this run\n(skip)",
          "unrestorable\n(skip)"]
  ax.bar(range(4), vals, color=[C[0], C[3], C[2], C[7]])
  for i, v in enumerate(vals):
    ax.text(i, v, str(v), ha="center", va="bottom", fontsize=7)
  ax.set_xticks(range(4))
  ax.set_xticklabels(labs, fontsize=6.5)
  ax.set_ylabel("Safe slots, summed over windows")
  ax.set_title("What memory says about each frontier slot", fontsize=9, loc="left")
  fig.tight_layout()
  save(fig, "fig9_cross_variant_reuse")


# ---------------------------------------------------------------- screenshot panels
def thumb(ep, cid, w=270):
  im = Image.open(Path(ep) / "img" / f"{cid}.jpg").convert("RGB")
  return im.resize((w, int(w * im.height / im.width)))


def fig3(rsteps, n=4):
  """Representative real examples: ref vs variant screenshot, diff overlay, changed elements."""
  rows = [r for r in rsteps if r["mode"] == "replay_struct" and r.get("pix_diff_ratio") not in ("", None)]
  # pick distinct tasks with the largest a11y change among layout/struct matches, plus one failure
  picked, seen = [], set()
  for r in sorted(rows, key=lambda r: -(f(r["a11y_added"]) + f(r["a11y_removed"]) + 50 * f(r["pix_diff_ratio"]))):
    if r["task"] in seen:
      continue
    seen.add(r["task"])
    picked.append(r)
    if len(picked) == n:
      break
  if not picked:
    return
  fig, axes = plt.subplots(len(picked), 4, figsize=(11, 4.6 * len(picked)),
                           gridspec_kw={"width_ratios": [1, 1, 1, 1.4]})
  if len(picked) == 1:
    axes = [axes]
  for k, r in enumerate(picked):
    a = thumb(r["ref_ep"], r["ref_cid"])
    bimg = thumb(r["live_ep"], r["live_cid"])
    da = np.asarray(a.convert("L"), dtype=float)
    db = np.asarray(bimg.convert("L"), dtype=float)
    mask = np.abs(da - db) > 25
    ov = np.asarray(bimg).copy()
    ov[mask] = (0.4 * ov[mask] + 0.6 * np.array([227, 73, 72])).astype(np.uint8)
    for j, (im, t) in enumerate([(a, "Reference"), (bimg, f"Variant {r['variant']}"), (Image.fromarray(ov), "Pixel diff (red)")]):
      axes[k][j].imshow(im)
      axes[k][j].set_title(t, fontsize=8, loc="left")
      axes[k][j].axis("off")
    ua = A.load_ui(r["ref_ep"], r["ref_cid"])
    ub = A.load_ui(r["live_ep"], r["live_cid"])
    sa = Counter((n.short_rid or n.short_cls, U.label_of(ua, n)[:28]) for n in ua.app_nodes() if n.clickable and n.visible)
    sb = Counter((n.short_rid or n.short_cls, U.label_of(ub, n)[:28]) for n in ub.app_nodes() if n.clickable and n.visible)
    rem = list((sa - sb).elements())[:7]
    add = list((sb - sa).elements())[:7]
    txt = (f"{r['task']} · ref step {r['ref_step']} · state match: {r['state_match']}\n"
           f"pixels changed {f(r['pix_diff_ratio']):.1%}, SSIM {f(r['ssim']):.3f}\n"
           f"A11y: +{r['a11y_added']} / −{r['a11y_removed']} elements, {r['a11y_moved']} moved, "
           f"{r['a11y_text_changed']} text changes\n"
           f"replay decision: {'execute' if b(r['execute']) else 'stop'} {r.get('reason') or ''} "
           f"[{r.get('match_category') or r.get('action_type')}]\n\nClickables only in reference:\n")
    txt += "\n".join(f"  − {x[0]} '{x[1]}'" for x in rem) or "  (none)"
    txt += "\nClickables only in variant:\n" + ("\n".join(f"  + {x[0]} '{x[1]}'" for x in add) or "  (none)")
    axes[k][3].text(0, 1, txt, va="top", ha="left", fontsize=7, family="DejaVu Sans Mono", color=INK)
    axes[k][3].axis("off")
  fig.tight_layout()
  save(fig, "fig3_dynamic_ui_examples")


def fig7():
  conv = rd("convergence_events.csv")
  conv = [c for c in conv if c["rule"] == "struct" and c["origin"] == "explore" and c.get("src_cid")]
  if not conv:
    return
  c = max(conv, key=lambda c: f(c["dst_known_out_slots"]))
  # find episode for this instance
  inst = c["instance"]
  task, var = inst.split("/")
  ep = ROOT / "raw" / "main" / task / f"{var}_agent"
  gx = ROOT / "graphs" / "main" / f"{task}.explore.jsonl"
  obs = [json.loads(l) for l in gx.read_text().splitlines() if l.strip()]
  dst_full = [o for o in obs if o["dst"]["struct_sig"].startswith(c["dst"])]
  first = min(dst_full, key=lambda o: o["t"])
  fig, axes = plt.subplots(1, 4, figsize=(11, 4.6))
  ims = [(ep, c["src_cid"], f"New path: source ({var})"), (ep, c["dst_cid"], f"probe '{c['slot_id'].split('|')[-1][:20]}' →"),
         (ROOT / "raw" / "main" / task / (first["instance"].split("/")[1] + "_agent"), first["dst_cid"],
          f"Existing node (first seen {first['instance'].split('/')[1]}, {first['origin']})")]
  for j, (e, cid, t) in enumerate(ims):
    try:
      axes[j].imshow(thumb(e, cid))
    except FileNotFoundError:
      pass
    axes[j].set_title(t, fontsize=7.5, loc="left")
    axes[j].axis("off")
  axes[3].axis("off")
  axes[3].text(0, 1, f"Graph convergence (struct matching)\n\ntask: {task}\nnew edge: {c['slot_id'][:40]}\n"
               f"destination already had {c['incoming_before']} incoming edge(s)\n"
               f"and {c['dst_known_out_slots']} observed outgoing slot(s),\n"
               "which need not be re-explored.\n\nThis establishes an environmental state\n"
               "correspondence, not task-semantic progress.", va="top", fontsize=7.5, color=INK)
  save(fig, "fig7_graph_convergence")


def main():
  eps = rd("episodes.csv")
  rsteps = rd("replay_steps.csv")
  pairs = rd("aligned_pairs.csv")
  for fn, args in [(fig1, (eps,)), (fig2, (pairs, rsteps)), (fig3, (rsteps,)), (fig4, (eps, rsteps)),
                   (fig5, ()), (fig6, ()), (fig7, ()), (fig8, ()), (fig9, ())]:
    try:
      fn(*args)
      print("ok", fn.__name__)
    except Exception as e:  # pylint: disable=broad-except
      import traceback
      traceback.print_exc()
      print("FAIL", fn.__name__, e)


if __name__ == "__main__":
  main()


def fig10():
  ann_p = ROOT / "analysis" / "utility" / "annotations.json"
  resp_p = ROOT / "analysis" / "utility" / "responses.jsonl"
  if not ann_p.exists():
    return
  ann = json.loads(ann_p.read_text())
  resp = [json.loads(l) for l in resp_p.read_text().splitlines() if l.strip()]
  n = len({(r["ep"], r["step"]) for r in resp})
  cats = ["none", "neutral", "beneficial (minor)", "uncertain", "harmful"]
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})
  ax = axes[0]
  for i, cond in enumerate(["B", "C"]):
    c = Counter()
    for k, v in ann["changed"].items():
      c[v[f"{cond}_vs_A"]] += 1
    c["none"] += len(ann["unchanged_states"])
    left = 0
    for j, cat in enumerate(cats):
      ax.barh(i, c[cat], left=left, color=[INK2, C[0], C[2], C[3], C[7]][j], edgecolor="white", lw=1,
              label=cat if i == 0 else None)
      if c[cat]:
        ax.text(left + c[cat] / 2, i, str(c[cat]), ha="center", va="center", fontsize=7, color="white")
      left += c[cat]
  ax.set_yticks([0, 1])
  ax.set_yticklabels(["B: local evidence", "C: whole task graph"])
  ax.set_xlabel(f"Decision states (n={n}); effect of evidence on next action vs. A (no evidence)")
  ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.32))
  ax.grid(axis="y", visible=False)
  ax.set_title("One-step utility (manual judgement)", fontsize=9, loc="left")
  ax = axes[1]
  toks = {c: [r["prompt_tokens"] for r in resp if r["cond"] == c] for c in "ABC"}
  ax.bar(range(3), [np.mean(toks[c]) for c in "ABC"], color=[INK2, C[0], C[1]])
  for i, c in enumerate("ABC"):
    ax.text(i, np.mean(toks[c]), f"{np.mean(toks[c]):.0f}", ha="center", va="bottom", fontsize=7)
  ax.set_xticks(range(3))
  ax.set_xticklabels(["A", "B", "C"])
  ax.set_ylabel("Prompt tokens (mean)")
  ax.set_title("Prompt cost", fontsize=9, loc="left")
  fig.tight_layout()
  save(fig, "fig10_reasoning_utility")


if __name__ == "__main__" and len(sys.argv) > 0:
  try:
    fig10()
    print("ok fig10")
  except Exception as e:  # pylint: disable=broad-except
    print("FAIL fig10", e)
