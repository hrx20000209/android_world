"""Per-app / per-category / per-difficulty breakdown of one run.

Every panel is drawn twice: over all tasks in the group, and over only the
tasks that succeeded. The gap between the two is the interesting part - a
mechanism that fires equally on both is not what separates a success from a
failure.

  python scripts/exploration_breakdown.py RUN_DIR OUT_DIR [BASELINE_TXT]
"""
import collections
import json
import pathlib
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.axisbelow": True,
})
ALL_C, OK_C = "#BBBBBB", "#0072B2"
SYSTEM_HINTS = ("launcher", "systemui", "permissioncontroller",
                "packageinstaller", "nexuslauncher")


def app_of(task_dir: pathlib.Path) -> str:
  """The app the episode actually worked in: most-visited non-system package."""
  graph = task_dir / "progressive_belief_graph.json"
  if not graph.exists():
    return "unknown"
  try:
    j = json.load(graph.open())
  except ValueError:
    return "unknown"
  counts = collections.Counter()
  for node in j.get("nodes", []):
    pkg = node.get("package", "")
    if pkg and not any(h in pkg for h in SYSTEM_HINTS):
      counts[pkg] += int(node.get("visit_count") or 1)
  return counts.most_common(1)[0][0].split(".")[-1] if counts else "unknown"


def category_of(task: str) -> str:
  m = re.match(r"^([A-Z][a-z]+)", task)
  return m.group(1) if m else task[:8]


def baseline_steps(path: pathlib.Path) -> dict:
  out = {}
  for line in open(path):
    q = line.split()
    if len(q) >= 5 and q[0][0].isalpha() and q[1].isdigit():
      try:
        out[q[0]] = float(q[4])
      except ValueError:
        pass
  return out


def collect(run: pathlib.Path, baseline: dict) -> list[dict]:
  rows = []
  for log in sorted(run.glob("*.log")):
    task = log.stem
    if task == "batch":
      continue
    text = log.read_text(errors="ignore")
    if "Task Successful" not in text and "Task Failed" not in text:
      continue
    d = run / task
    steps = rounds = probes = unrestored = aligned = guessed = skips = inj = 0
    events = d / "serial_events.jsonl"
    if events.exists():
      for line in events.open():
        if '"injected": true' in line:
          inj += 1
        try:
          e = json.loads(line)
        except ValueError:
          continue
        kind = e.get("kind")
        if kind == "inference":
          steps += 1
        elif kind == "explore":
          rounds += 1
          probes += int(e.get("probes_completed") or 0)
          if e.get("restore_status") != "RESTORED":
            unrestored += 1
        elif kind == "prefix_check":
          aligned += int(e.get("aligned") or 0)
          guessed += int(e.get("action_was_guessed") or 0)
        elif kind == "skip" and e.get("kind_detail") != "app_bootstrap":
          skips += 1
    nodes = edges = 0
    graph = d / "progressive_belief_graph.json"
    if graph.exists():
      try:
        j = json.load(graph.open())
        nodes, edges = len(j.get("nodes", [])), len(j.get("edges", []))
      except ValueError:
        pass
    base = baseline.get(task)
    rows.append({
        "task": task, "app": app_of(d), "category": category_of(task),
        "success": int("Task Successful" in text), "steps": steps,
        "rounds": rounds, "probes": probes, "unrestored": unrestored,
        "aligned": aligned, "guessed": guessed, "skips": skips,
        "injections": inj, "nodes": nodes, "edges": edges,
        "baseline_steps": base,
        "difficulty": ("easy" if base is not None and base <= 4 else
                       "medium" if base is not None and base <= 8 else
                       "hard" if base is not None else "unsolved by baseline"),
    })
  return rows


METRICS = [
    ("probes", "probes per task", False),
    ("rounds", "exploration rounds per task", False),
    ("nodes", "screens in the episode's graph", False),
    ("edges", "transitions in the episode's graph", False),
    ("skips", "graph skips per task", False),
    ("injections", "graph-context injections per task", False),
    ("aligned", "prefix alignments per task", False),
    ("unrestored", "unrestored probe rounds per task", False),
]


DIFFICULTY_ORDER = ["easy", "medium", "hard", "unsolved by baseline"]


def _order(names, key):
  """Difficulty reads as a scale, so it must not be sorted alphabetically."""
  if key == "difficulty":
    return sorted(names, key=lambda n: DIFFICULTY_ORDER.index(n)
                  if n in DIFFICULTY_ORDER else len(DIFFICULTY_ORDER))
  return sorted(names)


def _grouped(rows, key, min_n=3):
  groups = collections.defaultdict(list)
  for r in rows:
    groups[r[key]].append(r)
  return {k: v for k, v in groups.items() if len(v) >= min_n}


def panel(ax, groups, field, ylabel, key="app"):
  names = (_order(groups, key) if key == "difficulty" else
           sorted(groups, key=lambda k: -statistics.mean(
               r[field] for r in groups[k])))
  x = range(len(names))
  allv = [statistics.mean(r[field] for r in groups[n]) for n in names]
  okv = []
  for n in names:
    ok = [r[field] for r in groups[n] if r["success"]]
    okv.append(statistics.mean(ok) if ok else 0.0)
  ax.bar([i - 0.2 for i in x], allv, width=0.38, color=ALL_C,
         edgecolor="black", lw=0.4, label="all tasks")
  ax.bar([i + 0.2 for i in x], okv, width=0.38, color=OK_C,
         edgecolor="black", lw=0.4, label="successful tasks")
  ax.set_xticks(list(x))
  ax.set_xticklabels([f"{n[:12]} ({len(groups[n])})" for n in names],
                     rotation=45, ha="right", rotation_mode="anchor")
  ax.set_ylabel(ylabel)


def sheet(rows, key, title, out, fname, min_n=3):
  groups = _grouped(rows, key, min_n)
  if not groups:
    return
  fig, axes = plt.subplots(4, 2, figsize=(7.2, 10.5))
  for ax, (field, ylabel, _) in zip(axes.ravel(), METRICS):
    panel(ax, groups, field, ylabel, key)
  axes[0][0].legend(frameon=False, loc="upper right")
  fig.suptitle(title, y=0.995)
  fig.tight_layout(rect=(0, 0, 1, 0.985))
  fig.savefig(out / f"{fname}.pdf")
  fig.savefig(out / f"{fname}.png")
  plt.close(fig)


def rates_sheet(rows, key, title, out, fname, min_n=3):
  """Rates rather than counts: per-step and per-probe efficiency."""
  groups = _grouped(rows, key, min_n)
  if not groups:
    return
  defs = [
      (lambda g: sum(r["probes"] for r in g) / max(1, sum(r["steps"] for r in g)),
       "probes per step"),
      (lambda g: sum(r["aligned"] for r in g) / max(1, sum(r["probes"] for r in g)),
       "prefix alignments per probe"),
      (lambda g: sum(r["skips"] for r in g) / max(1, sum(r["steps"] for r in g)),
       "graph skips per step"),
      (lambda g: sum(r["injections"] for r in g) / max(1, sum(r["steps"] for r in g)),
       "injections per step"),
      (lambda g: sum(r["unrestored"] for r in g) / max(1, sum(r["rounds"] for r in g)),
       "unrestored share of rounds"),
      (lambda g: sum(r["nodes"] for r in g) / max(1, sum(r["steps"] for r in g)),
       "new screens per step"),
  ]
  names = _order(groups, key)
  fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.0))
  for ax, (fn, ylabel) in zip(axes.ravel(), defs):
    x = range(len(names))
    allv = [fn(groups[n]) for n in names]
    okv = [fn([r for r in groups[n] if r["success"]]) or 0.0 for n in names]
    ax.bar([i - 0.2 for i in x], allv, width=0.38, color=ALL_C,
           edgecolor="black", lw=0.4, label="all tasks")
    ax.bar([i + 0.2 for i in x], okv, width=0.38, color=OK_C,
           edgecolor="black", lw=0.4, label="successful tasks")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{n[:12]} ({len(groups[n])})" for n in names],
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)
  axes[0][0].legend(frameon=False, loc="upper right")
  fig.suptitle(title, y=0.995)
  fig.tight_layout(rect=(0, 0, 1, 0.985))
  fig.savefig(out / f"{fname}.pdf")
  fig.savefig(out / f"{fname}.png")
  plt.close(fig)


def main(argv):
  if len(argv) < 3:
    print(__doc__)
    return 2
  run = pathlib.Path(argv[1])
  out = pathlib.Path(argv[2])
  out.mkdir(parents=True, exist_ok=True)
  base = baseline_steps(pathlib.Path(argv[3])) if len(argv) > 3 else {}
  rows = collect(run, base)
  (out / "per_task.json").write_text(
      json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
  name = run.name
  sheet(rows, "app", f"{name}: exploration by app", out, f"{name}_by_app", 4)
  sheet(rows, "category", f"{name}: exploration by task family", out,
        f"{name}_by_category", 4)
  sheet(rows, "difficulty", f"{name}: exploration by baseline difficulty", out,
        f"{name}_by_difficulty", 1)
  rates_sheet(rows, "app", f"{name}: exploration rates by app", out,
              f"{name}_rates_by_app", 4)
  rates_sheet(rows, "difficulty",
              f"{name}: exploration rates by baseline difficulty", out,
              f"{name}_rates_by_difficulty", 1)
  print(f"tasks {len(rows)}  ->  {out}")
  for p in sorted(out.glob(f"{name}_*.png")):
    print("  ", p.name)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
