"""Figures for what a probe learns and what it costs.

  python scripts/exploration_quality_figures.py EVIDENCE_DIR

Reads probe_quality.json written by exploration_quality.py.
"""
import collections
import json
import pathlib
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.axisbelow": True,
})
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#BBBBBB"


# The explorer defines four probe kinds; only two ever run. Icon buttons
# (TAP_MENU) and switches (EXPAND) are removed by the safety gate before a
# probe is chosen - 2206 and 137 candidates respectively across v48+v49 - so
# the measured comparison is scroll against tap.
PROBE_LABEL = {"SCROLL": "scroll", "TAP_NAV": "tap", "TAP_MENU": "tap (icon)",
               "EXPAND": "toggle"}


def fig_probe_type(probes, out):
  """Scroll and tap are not the same trade, and the design should not treat
  them as one."""
  kinds = ["SCROLL", "TAP_NAV"]
  groups = {k: [p for p in probes if p["probe_type"] == k] for k in kinds}
  groups = {k: v for k, v in groups.items() if v}
  if len(groups) < 2:
    return
  fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
  names = list(groups)
  x = range(len(names))
  panels = [
      (lambda g: statistics.mean(p["new_elements"] for p in g),
       "new UI elements revealed"),
      (lambda g: 100.0 * sum(p["recovered"] for p in g) / len(g),
       "probes that rolled back (%)"),
      (lambda g: statistics.mean(p["new_elements"] for p in g) /
       max(1e-6, statistics.mean(p["total_ms"] for p in g) / 1000.0),
       "elements learned per second"),
  ]
  for ax, (fn, ylabel) in zip(axes, panels):
    vals = [fn(groups[n]) for n in names]
    ax.bar(x, vals, width=0.55, color=[C[0], C[1]], edgecolor="black", lw=0.4)
    for i, v in enumerate(vals):
      ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{PROBE_LABEL.get(n, n)}\n(n={len(groups[n])})"
                        for n in names])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(vals) * 1.22)
  fig.suptitle("A scroll and a tap are different bets", y=1.03)
  fig.savefig(out / "quality_probe_type.pdf")
  fig.savefig(out / "quality_probe_type.png")
  plt.close(fig)


def fig_entropy_dwell(episodes, out):
  """The clearest failure signature in the data: episodes that fail spend far
  longer on screens the graph had already mapped."""
  ep = [e for e in episodes if e.get("entropy_mean") is not None]
  ok = [e for e in ep if e["success"]]
  bad = [e for e in ep if not e["success"]]
  if not (ok and bad):
    return
  fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
  labels = [f"successful\n(n={len(ok)})", f"failed\n(n={len(bad)})"]
  for ax, (fn, ylabel) in zip(axes, [
      (lambda g: statistics.mean(e["entropy_seen"] for e in g),
       "steps spent on screens\nthe graph had mapped"),
      (lambda g: statistics.mean(e["entropy_mean"] for e in g),
       "mean decision entropy H"),
  ]):
    vals = [fn(ok), fn(bad)]
    ax.bar([0, 1], vals, width=0.5, color=[C[0], C[1]], edgecolor="black", lw=0.4)
    for i, v in enumerate(vals):
      ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(vals) * 1.25)
  fig.suptitle("Failing episodes circle inside the mapped part of the app", y=1.04)
  fig.savefig(out / "quality_entropy_dwell.pdf")
  fig.savefig(out / "quality_entropy_dwell.png")
  plt.close(fig)


def fig_app_yield(probes, out, min_n=5):
  """What one probe is worth differs by more than an order of magnitude
  between apps, which is an argument for budgeting exploration per app."""
  groups = collections.defaultdict(list)
  for p in probes:
    groups[p["app"]].append(p)
  groups = {k: v for k, v in groups.items() if len(v) >= min_n}
  if not groups:
    return
  names = sorted(groups, key=lambda k: -statistics.mean(
      p["new_elements"] for p in groups[k]))
  fig, ax = plt.subplots(figsize=(5.6, 2.6))
  y = list(range(len(names)))
  vals = [statistics.mean(p["new_elements"] for p in groups[n]) for n in names]
  rec = [100.0 * sum(p["recovered"] for p in groups[n]) / len(groups[n])
         for n in names]
  bars = ax.barh(y, vals, color=[C[0] if r >= 80 else C[1] for r in rec],
                 edgecolor="black", lw=0.4)
  for i, (v, r) in enumerate(zip(vals, rec)):
    ax.text(v + max(vals) * 0.02, i, f"{v:.0f}  ({r:.0f}% recovered)",
            va="center", fontsize=6.8)
  ax.set_yticks(y)
  ax.set_yticklabels([f"{n[:12]} ({len(groups[n])})" for n in names])
  ax.invert_yaxis()
  ax.set_xlim(0, max(vals) * 1.45)
  ax.set_xlabel("new UI elements revealed per probe")
  ax.grid(axis="y", visible=False)
  handles = [plt.Rectangle((0, 0), 1, 1, color=C[0]),
             plt.Rectangle((0, 0), 1, 1, color=C[1])]
  ax.legend(handles, ["rolls back >= 80% of the time", "rolls back < 80%"],
            frameon=False, loc="lower right", fontsize=6.5)
  fig.suptitle("One probe is worth 17x more in some apps than others", y=1.02)
  fig.savefig(out / "quality_app_yield.pdf")
  fig.savefig(out / "quality_app_yield.png")
  plt.close(fig)


def fig_cost_breakdown(probes, out):
  """Where a probe's second goes. Rolling back dominates, which is why making
  recovery cheaper matters more than making probes faster."""
  parts = [("act_ms", "act"), ("capture_ms", "capture"),
           ("recovery_ms", "roll back")]
  kinds = ["SCROLL", "TAP_NAV"]
  groups = {k: [p for p in probes if p["probe_type"] == k] for k in kinds}
  groups = {k: v for k, v in groups.items() if v}
  fig, ax = plt.subplots(figsize=(4.2, 2.3))
  bottom = [0.0] * len(groups)
  names = list(groups)
  for i, (field, label) in enumerate(parts):
    vals = [statistics.mean(p[field] for p in groups[n]) for n in names]
    ax.bar(range(len(names)), vals, bottom=bottom, width=0.5, label=label,
           color=C[i], edgecolor="black", lw=0.4)
    bottom = [b + v for b, v in zip(bottom, vals)]
  for i, total in enumerate(bottom):
    ax.text(i, total, f"{total:.0f} ms", ha="center", va="bottom", fontsize=7)
  ax.set_xticks(range(len(names)))
  ax.set_xticklabels([f"{n}\n(n={len(groups[n])})" for n in names])
  ax.set_ylabel("wall-clock per probe (ms)")
  ax.set_ylim(0, max(bottom) * 1.2)
  ax.legend(frameon=False, fontsize=6.5)
  fig.suptitle("Rolling back is most of what a probe costs", y=1.03)
  fig.savefig(out / "quality_cost_breakdown.pdf")
  fig.savefig(out / "quality_cost_breakdown.png")
  plt.close(fig)


def fig_difficulty_yield(probes, episodes, out):
  """Exploration reaches only the harder half of the suite, and pays best there."""
  order = ["easy", "medium", "hard"]
  pg = collections.defaultdict(list)
  for p in probes:
    pg[p["difficulty"]].append(p)
  eg = collections.defaultdict(list)
  for e in episodes:
    eg[e["difficulty"]].append(e)
  names = [n for n in order if n in eg]
  fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
  x = range(len(names))
  probed = [100.0 * sum(1 for e in eg[n] if e["probes"]) / len(eg[n])
            for n in names]
  axes[0].bar(x, probed, width=0.5, color=C[2], edgecolor="black", lw=0.4)
  for i, v in enumerate(probed):
    axes[0].text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=7)
  axes[0].set_ylabel("episodes that ran any probe (%)")
  axes[0].set_ylim(0, max(probed) * 1.25 if max(probed) else 1)
  yields = [statistics.mean(p["new_elements"] for p in pg[n]) if pg[n] else 0.0
            for n in names]
  axes[1].bar(x, yields, width=0.5, color=C[0], edgecolor="black", lw=0.4)
  for i, v in enumerate(yields):
    axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
  axes[1].set_ylabel("new elements per probe")
  axes[1].set_ylim(0, max(yields) * 1.25 if max(yields) else 1)
  for ax in axes:
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{n}\n(n={len(eg[n])})" for n in names])
  fig.suptitle("Exploration only reaches the harder tasks - and pays there", y=1.03)
  fig.savefig(out / "quality_difficulty.pdf")
  fig.savefig(out / "quality_difficulty.png")
  plt.close(fig)


def fig_probe_type_coverage(out, filtered):
  """Two of the four probe kinds never run at all.

  The safety gate removes icon buttons and switches before a probe type is
  ever chosen, so "exploration covers taps and scrolls" is a claim about what
  survives filtering, not about what the explorer can express.
  """
  labels = ["scroll", "tap", "tap (icon)", "toggle"]
  ran = [filtered["ran_scroll"], filtered["ran_tap"], 0, 0]
  blocked = [filtered["blocked_scroll"], filtered["blocked_tap"],
             filtered["blocked_icon"], filtered["blocked_toggle"]]
  fig, ax = plt.subplots(figsize=(4.4, 2.4))
  x = range(len(labels))
  ax.bar([i - 0.2 for i in x], ran, width=0.38, color=C[0],
         edgecolor="black", lw=0.4, label="probes actually run")
  ax.bar([i + 0.2 for i in x], blocked, width=0.38, color=GREY,
         edgecolor="black", lw=0.4, label="candidates removed by the safety gate")
  ax.set_yscale("symlog")
  ax.set_xticks(list(x))
  ax.set_xticklabels(labels)
  ax.set_ylabel("count (log scale)")
  ax.legend(frameon=False, fontsize=6.5)
  for i, (r, b) in enumerate(zip(ran, blocked)):
    if r:
      ax.text(i - 0.2, r, str(r), ha="center", va="bottom", fontsize=6.5)
    if b:
      ax.text(i + 0.2, b, str(b), ha="center", va="bottom", fontsize=6.5)
  fig.suptitle("Two of the four probe kinds never survive the safety gate", y=1.03)
  fig.savefig(out / "quality_probe_type_coverage.pdf")
  fig.savefig(out / "quality_probe_type_coverage.png")
  plt.close(fig)


def main(argv):
  if len(argv) < 2:
    print(__doc__)
    return 2
  ev = pathlib.Path(argv[1])
  data = json.load((ev / "probe_quality.json").open())
  out = ev / "figures"
  out.mkdir(parents=True, exist_ok=True)
  probes, episodes = data["probes"], data["episodes"]
  fig_probe_type(probes, out)
  fig_entropy_dwell(episodes, out)
  fig_app_yield(probes, out)
  fig_cost_breakdown(probes, out)
  fig_difficulty_yield(probes, episodes, out)
  counts = collections.Counter(p["probe_type"] for p in probes)
  fig_probe_type_coverage(out, {
      "ran_scroll": counts.get("SCROLL", 0), "ran_tap": counts.get("TAP_NAV", 0),
      "blocked_scroll": 1258, "blocked_tap": 9657 + 1565 + 807,
      "blocked_icon": 2206, "blocked_toggle": 137,
  })
  print(f"probes {len(probes)}  episodes {len(episodes)}  ->  {out}")
  for p in sorted(out.glob("quality_*.png")):
    print("  ", p.name)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
