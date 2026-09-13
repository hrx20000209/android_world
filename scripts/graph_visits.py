"""How unevenly an episode's attention lands on the screens it visits.

  python scripts/graph_visits.py OUT_DIR BASELINE_TXT RUN_DIR [RUN_DIR ...]

Runs passed together must be the same configuration; this pools them for
sample size rather than comparing them.
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
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.axisbelow": True,
})
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
SYSTEM_HINTS = ("launcher", "systemui", "permissioncontroller",
                "packageinstaller", "nexuslauncher")
DIFF_ORDER = ["easy", "medium", "hard"]


def baseline_steps(path):
  out = {}
  for line in open(path):
    q = line.split()
    if len(q) >= 5 and q[0][0].isalpha() and q[1].isdigit():
      try:
        out[q[0]] = float(q[4])
      except ValueError:
        pass
  return out


def collect(runs, base):
  rows = []
  for run in runs:
    run = pathlib.Path(run)
    for graph_path in run.glob("*/progressive_belief_graph.json"):
      task_dir = graph_path.parent
      task = task_dir.name
      log = run / f"{task}.log"
      if not log.exists():
        continue
      text = log.read_text(errors="ignore")
      if "Task Successful" not in text and "Task Failed" not in text:
        continue
      try:
        g = json.load(graph_path.open())
      except ValueError:
        continue
      nodes = g.get("nodes", [])
      if not nodes:
        continue
      visits = sorted((int(n.get("visit_count") or 0) for n in nodes),
                      reverse=True)
      apps = collections.Counter()
      for n in nodes:
        pkg = n.get("package", "")
        if pkg and not any(h in pkg for h in SYSTEM_HINTS):
          apps[pkg] += int(n.get("visit_count") or 1)
      b = base.get(task)
      rows.append({
          "task": task, "run": run.name,
          "app": apps.most_common(1)[0][0].split(".")[-1] if apps else "unknown",
          "success": int("Task Successful" in text),
          "difficulty": ("easy" if b is not None and b <= 4 else
                         "medium" if b is not None and b <= 8 else "hard"),
          "visits": visits,
          "screens": len(visits),
          "total_visits": sum(visits),
          "top1_share": visits[0] / sum(visits) if sum(visits) else 0.0,
          "top3_share": sum(visits[:3]) / sum(visits) if sum(visits) else 0.0,
      })
  return rows


def fig_rank(rows, key, out, fname, title, min_n=4, max_rank=12):
  groups = collections.defaultdict(list)
  for r in rows:
    groups[r[key]].append(r)
  groups = {k: v for k, v in groups.items() if len(v) >= min_n}
  names = ([n for n in DIFF_ORDER if n in groups] if key == "difficulty"
           else sorted(groups, key=lambda k: -len(groups[k]))[:6])
  if not names:
    return
  fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
  for ax, only_ok in zip(axes, (False, True)):
    for i, name in enumerate(names):
      series = [r["visits"] for r in groups[name] if r["success"] or not only_ok]
      if not series:
        continue
      xs, ys = [], []
      for rank in range(max_rank):
        vals = [s[rank] for s in series if len(s) > rank]
        if len(vals) < 2:
          break
        xs.append(rank + 1)
        ys.append(statistics.mean(vals))
      if xs:
        ax.plot(xs, ys, color=C[i % len(C)], lw=1.3, marker="o", ms=3,
                label=f"{name} (n={len(series)})")
    ax.set_xlabel("screen, ranked by visits within its episode")
    ax.set_ylabel("mean visits")
    ax.set_yscale("log")
    ax.set_title("successful tasks" if only_ok else "all tasks")
    ax.legend(frameon=False, fontsize=6.5)
  fig.suptitle(title, y=1.02)
  fig.savefig(out / f"{fname}.pdf")
  fig.savefig(out / f"{fname}.png")
  plt.close(fig)


def fig_concentration(rows, out):
  """One number per group: how much of the episode is spent on its busiest screen."""
  fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
  for ax, key in zip(axes, ("difficulty", "app")):
    groups = collections.defaultdict(list)
    for r in rows:
      groups[r[key]].append(r)
    groups = {k: v for k, v in groups.items() if len(v) >= (1 if key == "difficulty" else 8)}
    names = ([n for n in DIFF_ORDER if n in groups] if key == "difficulty"
             else sorted(groups, key=lambda k: -len(groups[k]))[:7])
    x = range(len(names))
    allv = [statistics.mean(r["top1_share"] for r in groups[n]) * 100 for n in names]
    okv = []
    for n in names:
      ok = [r["top1_share"] for r in groups[n] if r["success"]]
      okv.append(statistics.mean(ok) * 100 if ok else 0.0)
    ax.bar([i - 0.2 for i in x], allv, width=0.38, color="#BBBBBB",
           edgecolor="black", lw=0.4, label="all tasks")
    ax.bar([i + 0.2 for i in x], okv, width=0.38, color=C[0],
           edgecolor="black", lw=0.4, label="successful tasks")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{n[:12]} ({len(groups[n])})" for n in names],
                       rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("share of all visits\non the busiest screen (%)")
  axes[0].legend(frameon=False, fontsize=6.5)
  fig.suptitle("Attention concentrates on one screen per episode", y=1.02)
  fig.savefig(out / "graph_visit_concentration.pdf")
  fig.savefig(out / "graph_visit_concentration.png")
  plt.close(fig)


def main(argv):
  if len(argv) < 4:
    print(__doc__)
    return 2
  out = pathlib.Path(argv[1])
  out.mkdir(parents=True, exist_ok=True)
  base = baseline_steps(pathlib.Path(argv[2]))
  rows = collect(argv[3:], base)
  fig_rank(rows, "difficulty", out, "graph_visit_rank_by_difficulty",
           "Visits per screen, ranked within each episode (by difficulty)")
  fig_rank(rows, "app", out, "graph_visit_rank_by_app",
           "Visits per screen, ranked within each episode (by app)")
  fig_concentration(rows, out)
  (out / "graph_visits.json").write_text(
      json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
  print(f"episodes {len(rows)}")
  for key in ("difficulty", "app"):
    g = collections.defaultdict(list)
    for r in rows:
      g[r[key]].append(r)
    print(f"-- by {key} --")
    order = ([n for n in DIFF_ORDER if n in g] if key == "difficulty"
             else sorted(g, key=lambda k: -len(g[k]))[:6])
    for n in order:
      v = g[n]
      print(f"  {n:<14} n={len(v):<4} 屏幕 {statistics.mean(r['screens'] for r in v):>5.1f} "
            f" 最忙屏幕占比 {statistics.mean(r['top1_share'] for r in v)*100:>5.1f}% "
            f" 前三占比 {statistics.mean(r['top3_share'] for r in v)*100:>5.1f}%")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
