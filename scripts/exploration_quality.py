"""How much a probe actually learns, and what it costs to learn it.

  python scripts/exploration_quality.py OUT_DIR BASELINE_TXT RUN_DIR [RUN_DIR ...]

Three quantities, all measured rather than assumed:

  yield     new UI elements and new labels the probe revealed
  entropy   H over a screen's viable continuations, from the graph itself
            (belief_graph.recompute_decision_entropy): H = -sum p log p over
            outgoing edges weighted by path probability. inf while nothing is
            known, 0 once a screen offers one continuation.
  cost      wall-clock, split into acting, capturing and rolling back
"""
import collections
import json
import math
import pathlib
import statistics
import sys

SYSTEM_HINTS = ("launcher", "systemui", "permissioncontroller",
                "packageinstaller", "nexuslauncher")


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


def app_of(task_dir):
  graph = task_dir / "progressive_belief_graph.json"
  if not graph.exists():
    return "unknown"
  try:
    j = json.load(graph.open())
  except ValueError:
    return "unknown"
  counts = collections.Counter()
  for n in j.get("nodes", []):
    pkg = n.get("package", "")
    if pkg and not any(h in pkg for h in SYSTEM_HINTS):
      counts[pkg] += int(n.get("visit_count") or 1)
  return counts.most_common(1)[0][0].split(".")[-1] if counts else "unknown"


def collect(runs, base):
  probes, episodes = [], []
  for run in runs:
    run = pathlib.Path(run)
    for log in sorted(run.glob("*.log")):
      task = log.stem
      if task == "batch":
        continue
      text = log.read_text(errors="ignore")
      if "Task Successful" not in text and "Task Failed" not in text:
        continue
      d = run / task
      b = base.get(task)
      meta = {
          "run": run.name, "task": task, "app": app_of(d),
          "success": int("Task Successful" in text),
          "difficulty": ("easy" if b is not None and b <= 4 else
                         "medium" if b is not None and b <= 8 else "hard"),
      }
      # entropy trajectory: every step logs the current screen's H
      ent_seq, steps = [], 0
      events = d / "serial_events.jsonl"
      if events.exists():
        for line in events.open():
          try:
            e = json.loads(line)
          except ValueError:
            continue
          if e.get("kind") == "inference":
            steps += 1
          elif e.get("kind") == "gate":
            h = e.get("node_entropy")
            if isinstance(h, (int, float)) and math.isfinite(h):
              ent_seq.append(h)
      trace = d / "probe_trace.jsonl"
      task_probes = []
      if trace.exists():
        for line in trace.open():
          try:
            r = json.loads(line)
          except ValueError:
            continue
          disc = r.get("discovered") or {}
          t = r.get("timings_ms") or {}
          task_probes.append({
              **meta,
              "probe_type": r.get("probe_type"),
              "depth": int(r.get("depth") or 1),
              "new_elements": int(disc.get("new_element_count") or 0),
              "new_labels": len(disc.get("new_texts") or ()),
              "left_app": bool(
                  str(disc.get("reached_activity") or "").split("/")[0]
                  not in ("", str(r.get("app_package") or ""))),
              "recovered": bool(r.get("recovery_ok")),
              "recovery_level": r.get("recovery_level") or "",
              "total_ms": float(t.get("total") or 0.0),
              "act_ms": float(t.get("action_exec") or 0.0),
              "capture_ms": float(t.get("post_state_capture") or 0.0),
              "recovery_ms": float(t.get("recovery") or 0.0)
                             + float(t.get("recovery_verify") or 0.0),
          })
      probes.extend(task_probes)
      episodes.append({
          **meta, "steps": steps, "probes": len(task_probes),
          "entropy_seen": len(ent_seq),
          "entropy_mean": statistics.mean(ent_seq) if ent_seq else None,
          "entropy_first": ent_seq[0] if ent_seq else None,
          "entropy_last": ent_seq[-1] if ent_seq else None,
      })
  return probes, episodes


def table(rows, key, fields, title, order=None):
  groups = collections.defaultdict(list)
  for r in rows:
    groups[r[key]].append(r)
  names = order or sorted(groups, key=lambda k: -len(groups[k]))
  names = [n for n in names if n in groups and len(groups[n]) >= 3]
  print(f"\n{title}")
  head = f"  {key:<14}{'n':>5}" + "".join(f"{lab:>16}" for _, lab in fields)
  print(head)
  print("  " + "-" * (len(head) - 2))
  for n in names:
    g = groups[n]
    cells = "".join(f"{fn(g):>16.2f}" for fn, _ in fields)
    print(f"  {n[:13]:<14}{len(g):>5}{cells}")


def main(argv):
  if len(argv) < 4:
    print(__doc__)
    return 2
  out = pathlib.Path(argv[1])
  out.mkdir(parents=True, exist_ok=True)
  base = baseline_steps(pathlib.Path(argv[2]))
  probes, episodes = collect(argv[3:], base)
  (out / "probe_quality.json").write_text(
      json.dumps({"probes": probes, "episodes": episodes}, indent=1,
                 ensure_ascii=False), encoding="utf-8")
  print(f"probes {len(probes)}   episodes {len(episodes)}")

  mean = lambda f: (lambda g: statistics.mean(x[f] for x in g))
  share = lambda f: (lambda g: 100.0 * sum(bool(x[f]) for x in g) / len(g))
  yield_fields = [
      (mean("new_elements"), "new elements"),
      (mean("new_labels"), "new labels"),
      (share("left_app"), "left app %"),
      (share("recovered"), "recovered %"),
      (mean("total_ms"), "cost ms"),
      (lambda g: statistics.mean(x["new_elements"] for x in g) /
       max(1e-6, statistics.mean(x["total_ms"] for x in g) / 1000.0),
       "elements / s"),
  ]
  table(probes, "difficulty", yield_fields, "每次探测的信息产出与代价 — 按难度",
        order=["easy", "medium", "hard"])
  table(probes, "app", yield_fields, "每次探测的信息产出与代价 — 按应用")
  table(probes, "probe_type", yield_fields, "每次探测的信息产出与代价 — 按探测类型")
  table(probes, "run", yield_fields, "每次探测的信息产出与代价 — 按配置")

  ep = [e for e in episodes if e["entropy_mean"] is not None]
  ent_fields = [
      (mean("entropy_mean"), "mean H"),
      (mean("entropy_first"), "H first step"),
      (mean("entropy_last"), "H last step"),
      (mean("entropy_seen"), "steps with H"),
      (mean("steps"), "steps"),
  ]
  table(ep, "difficulty", ent_fields, "决策熵 H — 按难度", order=["easy", "medium", "hard"])
  table(ep, "app", ent_fields, "决策熵 H — 按应用")
  ok = [e for e in ep if e["success"]]
  bad = [e for e in ep if not e["success"]]
  if ok and bad:
    print("\n决策熵 H — 成功 vs 失败")
    for label, g in (("成功", ok), ("失败", bad)):
      print(f"  {label}  n={len(g):<4} 平均 H {statistics.mean(x['entropy_mean'] for x in g):.3f}"
            f"  首步 {statistics.mean(x['entropy_first'] for x in g):.3f}"
            f"  末步 {statistics.mean(x['entropy_last'] for x in g):.3f}"
            f"  有 H 的步数 {statistics.mean(x['entropy_seen'] for x in g):.1f}")
  print(f"\n中间结果 -> {out/'probe_quality.json'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
