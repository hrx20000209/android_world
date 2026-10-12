#!/usr/bin/env python3
"""Full AndroidWorld paired run (base vs proc_execute) with pre-registered stops.

  python scripts/run_full_paired.py OUT_ROOT [--seed 30] [--tasks FILE]

For every task, base then proc_execute run back to back (same session, same
seed). After each pair the stop rules are evaluated; the first rule that
fires writes STOP.json and exits non-zero. One line per pair is printed for a
live monitor. Resumable: finished episodes are skipped by the pilot script.

Stop rules (fixed before the run):
  1. mechanism loss: proc failed, base succeeded, and a replay executed in
     that proc episode -> stop at 2
  2. overall drift: after >= 30 pairs, proc successes < base successes - 6
  3. environment: base successes < 4B_2 successes (same tasks) - 12
  4. crash: 2 consecutive episodes without an evaluator result
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def result(log: Path) -> dict | None:
  if not log.exists():
    return None
  text = log.read_text(errors="ignore")
  if "Average" not in text:
    return None
  m = re.search(r"\n(\w+)\s+\d+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                text.split("========= Average")[0][-600:])
  return {"ok": float(m.group(2)), "steps": float(m.group(3)), "runtime": float(m.group(4))}


def replays(run: Path) -> list[dict]:
  path = run / "serial_events.jsonl"
  if not path.exists():
    return []
  out = []
  for line in path.read_text(errors="ignore").splitlines():
    if '"procedure_execute"' in line:
      out.append(json.loads(line))
  return out


def calls(run: Path) -> int:
  path = run / "api_calls.jsonl"
  return len(path.read_text().splitlines()) if path.exists() else 0


def ref_4b2() -> dict[str, float]:
  out = {}
  for line in (REPO / "results" / "4B_2.txt").read_text().splitlines()[2:]:
    parts = line.split()
    if len(parts) >= 5 and parts[0][:1].isupper():
      try:
        out[parts[0]] = float(parts[3])
      except ValueError:
        pass
  return out


def llm_up(api: str = "http://127.0.0.1:8000/health") -> bool:
  import urllib.request
  try:
    with urllib.request.urlopen(api, timeout=5) as resp:
      return resp.status == 200
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def wait_for_llm() -> None:
  """Block until the model server answers; an episode run without it is void."""
  if llm_up():
    return
  print(f"[{time.strftime('%H:%M:%S')}] ⚠ vLLM 不可达（隧道断开？），等待恢复…", flush=True)
  while not llm_up():
    time.sleep(60)
  print(f"[{time.strftime('%H:%M:%S')}] vLLM 已恢复，继续", flush=True)


def llm_failed(log: Path) -> bool:
  return log.exists() and "Error calling LLM" in log.read_text(errors="ignore")


def healthy() -> bool:
  subprocess.run(["adb", "-s", "emulator-5554", "forward", "tcp:8765",
                  "localabstract:androidworld_fast_a11y"], capture_output=True)
  try:
    with socket.create_connection(("127.0.0.1", 8765), timeout=3):
      return True
  except OSError:
    return False


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("out_root")
  parser.add_argument("--seed", default="30")
  parser.add_argument("--tasks", default=str(REPO / "results/mobileexplorer_runs/tasks116.txt"))
  parser.add_argument("--arms", default="base,proc_execute")
  parser.add_argument("--pilot", default=str(REPO / "scripts/run_task_navigation_pilot.sh"),
                      help="Per-episode launcher; never edit one while a run uses it.")
  args = parser.parse_args()
  root = Path(args.out_root)
  root.mkdir(parents=True, exist_ok=True)
  tasks = [t.strip() for t in Path(args.tasks).read_text().splitlines() if t.strip()]
  ref = ref_4b2()
  ledger = root / "pairs.jsonl"
  pairs = [json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
  seen = {p["task"] for p in pairs}
  consecutive_crash = 0
  for task in tasks:
    if task in seen:
      continue
    if not healthy():
      print(f"[{time.strftime('%H:%M:%S')}] ⚠ a11y socket 不可用，停止", flush=True)
      (root / "STOP.json").write_text(json.dumps({"rule": "a11y_unavailable", "task": task}))
      return 3
    # Freeze the status bar (clock, battery, signal, notifications): identical
    # screens must give identical pixels, or temperature-0 decoding still
    # diverges between replicates. Re-applied per task; SystemUI may restart.
    for demo in (["command", "enter"], ["command", "clock", "-e", "hhmm", "1000"],
                 ["command", "battery", "-e", "level", "100", "-e", "plugged", "false"],
                 ["command", "network", "-e", "wifi", "show", "-e", "level", "4",
                  "-e", "mobile", "show", "-e", "datatype", "none", "-e", "level", "4"],
                 ["command", "notifications", "-e", "visible", "false"]):
      subprocess.run(["adb", "-s", "emulator-5554", "shell", "am", "broadcast", "-a",
                      "com.android.systemui.demo", "-e", *demo], capture_output=True)
    subprocess.run(["adb", "-s", "emulator-5554", "shell", "cmd", "role", "add-role-holder",
                    "android.app.role.SMS", "com.simplemobiletools.smsmessenger"],
                   capture_output=True)
    row = {"task": task}
    for arm in args.arms.split(","):
      for _attempt in range(3):
        wait_for_llm()
        subprocess.run(["bash", args.pilot,
                        str(root), args.seed, arm, task], cwd=REPO, capture_output=True)
        log = root / arm / f"{task}.log"
        if not llm_failed(log):
          break
        # The model server dropped mid-episode: void it and run it again.
        print(f"[{time.strftime('%H:%M:%S')}] ⚠ {arm}/{task} 期间 LLM 调用失败，作废重跑", flush=True)
        void = root / "_invalid_llm_down"
        void.mkdir(exist_ok=True)
        log.rename(void / f"{arm}_{task}_{int(time.time())}.log.txt")
        if (root / arm / task).exists():
          (root / arm / task).rename(void / f"{arm}_{task}_{int(time.time())}")
      res = result(root / arm / f"{task}.log")
      run = root / arm / task
      row[arm] = {**(res or {}), "crash": res is None, "calls": calls(run)}
      if arm != "base":
        reps = replays(run)
        row[arm].update(macros=len(reps), hops=sum(r["completed"] for r in reps),
                        reasons=[r["reason"] for r in reps])
      consecutive_crash = consecutive_crash + 1 if res is None else 0
    pairs.append(row)
    with ledger.open("a") as stream:
      stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    test = args.arms.split(",")[1]
    b, p = row["base"], row[test]
    done = [x for x in pairs if not x["base"]["crash"] and not x[test]["crash"]]
    sb = sum(x["base"]["ok"] for x in done)
    sp = sum(x[test]["ok"] for x in done)
    s4 = sum(ref.get(x["task"], 0) for x in done)
    mech = [x["task"] for x in done if x[test]["ok"] == 0 and x["base"]["ok"] > 0
            and x[test].get("macros", 0) > 0]
    fmt = lambda r: "崩溃" if r["crash"] else f"{'成功' if r['ok'] else '失败'}/{int(r['steps'])}步"
    print(f"[{time.strftime('%H:%M:%S')}] {len(pairs)}/{len(tasks)} {task}：base {fmt(b)}，"
          f"{test} {fmt(p)}，宏 {p.get('macros', 0)} 次/{p.get('hops', 0)} 跳"
          f"｜累计分数 base {sb:g} {test} {sp:g} 4B_2 {s4:g}（n={len(done)}）"
          f"｜机制失败 {len(mech)}", flush=True)
    rule = None
    if len(mech) >= 2:
      rule = {"rule": "mechanism_loss", "tasks": mech}
    elif len(done) >= 30 and sp < sb - 6:
      rule = {"rule": "overall_drift", "base": sb, "proc": sp}
    elif sb < s4 - 12:
      rule = {"rule": "environment_vs_4B_2", "base": sb, "4B_2": s4}
    elif consecutive_crash >= 2:
      rule = {"rule": "consecutive_crash", "task": task}
    if rule:
      rule["n"] = len(done)
      (root / "STOP.json").write_text(json.dumps(rule, ensure_ascii=False))
      print(f"== 触发停止规则：{json.dumps(rule, ensure_ascii=False)} ==", flush=True)
      return 2
  print("== FULL_DONE ==", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
