"""Score an exploration-suite run: outcome plus the mechanism counters.

On 51 tasks the success rate carries less signal than it does on 116, so the
mechanism counters are what a fast iteration should be read on - they measure
the thing being changed directly instead of through task outcome.
"""
import collections
import json
import pathlib
import statistics
import sys

CAP = 15


def scan(root: pathlib.Path) -> dict:
  out = {"tasks": {}, "m": collections.Counter()}
  for log in sorted(root.glob("*.log")):
    task = log.stem
    if task == "batch":
      continue
    text = log.read_text(errors="ignore")
    if "Task Successful" not in text and "Task Failed" not in text:
      continue
    events = root / task / "serial_events.jsonl"
    steps = 0
    if events.exists():
      for line in events.open():
        if '"injected": true' in line:
          out["m"]["injections"] += 1
        try:
          e = json.loads(line)
        except ValueError:
          continue
        kind = e.get("kind")
        if kind == "inference":
          steps += 1
          out["m"]["steps"] += 1
        elif kind == "explore":
          out["m"]["rounds"] += 1
          out["m"]["probes"] += int(e.get("probes_completed") or 0)
          if e.get("restore_status") != "RESTORED":
            out["m"]["unrestored"] += 1
        elif kind == "prefix_check":
          out["m"]["prefix_checks"] += 1
          out["m"]["aligned"] += int(e.get("aligned") or 0)
          out["m"]["guessed"] += int(e.get("action_was_guessed") or 0)
        elif kind == "skip" and e.get("kind_detail") != "app_bootstrap":
          out["m"]["graph_skips"] += 1
    ok = int("Task Successful" in text)
    if ok and steps > CAP:
      ok, steps = 0, CAP
    out["tasks"][task] = (ok, steps)
  return out


def report(name: str, data: dict) -> None:
  tasks = data["tasks"]
  wins = sum(v[0] for v in tasks.values())
  ok_steps = [v[1] for v in tasks.values() if v[0] and v[1]]
  m = data["m"]
  print(f"{name}: {wins}/{len(tasks)} 成功"
        + (f", 成功任务 {statistics.mean(ok_steps):.2f} 步" if ok_steps else ""))
  if m["steps"]:
    print(f"   探索轮 {m['rounds']} ({m['rounds']/m['steps']:.1%} 的步骤)"
          f"  探测 {m['probes']}  未复原 {m['unrestored']}")
    print(f"   前缀检查 {m['prefix_checks']}  对齐 {m['aligned']}  猜中 {m['guessed']}"
          f"  图跳过 {m['graph_skips']}  注入 {m['injections']}"
          f" ({m['injections']/m['steps']:.1%})")


def main(argv: list[str]) -> int:
  if len(argv) < 2:
    print(__doc__)
    return 2
  runs = [pathlib.Path(p) for p in argv[1:]]
  scans = [(p.name, scan(p)) for p in runs]
  for name, data in scans:
    report(name, data)
  if len(scans) >= 2:
    base_name, base = scans[0]
    for name, data in scans[1:]:
      common = sorted(set(base["tasks"]) & set(data["tasks"]))
      if not common:
        continue
      a = sum(base["tasks"][t][0] for t in common)
      b = sum(data["tasks"][t][0] for t in common)
      both = [(base["tasks"][t][1], data["tasks"][t][1]) for t in common
              if base["tasks"][t][0] and data["tasks"][t][0]]
      faster = sum(1 for x, y in both if x < y)
      slower = sum(1 for x, y in both if x > y)
      delta = statistics.mean([x - y for x, y in both]) if both else 0.0
      print(f"\n  {base_name} vs {name}: 共同 {len(common)}  成功 {a} : {b} ({a-b:+d})"
            f" | 都成功 {len(both)}: 快 {faster} 慢 {slower}, 均步差 {delta:+.2f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
