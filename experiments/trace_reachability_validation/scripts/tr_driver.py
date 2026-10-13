"""Paired Phase II driver: for each task variant run TG and RR (alternating order)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT.parent / "runtime_graph_study"


def note(msg):
  line = f"- {time.strftime('%m-%d %H:%M:%S')} {msg}"
  print(line, flush=True)
  with (ROOT / "reports" / "execution_log.md").open("a") as f:
    f.write(line + "\n")


def tunnel_ok():
  r = subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "8", "http://127.0.0.1:18095/v1/models"],
                     capture_output=True, text=True)
  return "GELAB" in r.stdout


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--manifest", required=True)
  ap.add_argument("--raw", default="raw/phase2")
  ap.add_argument("--budget", type=float, default=6.0)
  ap.add_argument("--arms", default="TG,RR")
  a = ap.parse_args()
  busy = subprocess.run(["pgrep", "-fl", "rgs_run.py|tr_run.py|rgs_driver.py"], capture_output=True, text=True).stdout
  if busy.strip():
    print("refusing; busy:\n" + busy)
    return 2
  man = json.loads((ROOT / a.manifest).read_text())
  raw = ROOT / a.raw
  gdir = ROOT / "graph_snapshots" / Path(a.raw).name
  gdir.mkdir(parents=True, exist_ok=True)
  arms = a.arms.split(",")
  for t in man["tasks"]:
    task = t["task"]
    for arm in arms:
      g = gdir / f"{task}.{arm}.explore.jsonl"
      if not g.exists():
        src = PREV / "graphs" / "main" / f"{task}.explore.jsonl"
        shutil.copy(src, g) if src.exists() else g.write_text("")
      m = gdir / f"_app_unrestorable.{arm}.jsonl"
      if not m.exists():
        shutil.copy(PREV / "graphs" / "main" / "_app_unrestorable.jsonl", m)
    for i, seed in enumerate(t["seeds"]):
      order = arms if i % 2 == 0 else arms[::-1]
      for arm in order:
        out = raw / task / f"s{seed}_{arm}"
        if (out / "result.json").exists():
          continue
        if out.exists():
          out.rename(out.with_name(out.name + f"_partial_{int(time.time())}"))
        while not tunnel_ok():
          note("tunnel down, waiting")
          time.sleep(30)
        env = dict(os.environ, TR_POLICY=arm, TR_RAW=str(raw), TR_DEPTH="1" if "1" in arm else "2")
        cmd = [sys.executable, str(ROOT / "scripts" / "tr_run.py"), "--task", task, "--seed", str(seed),
               "--variant", f"s{seed}_{arm}", "--mode", "agent", "--out", str(out), "--explore",
               "--budget", str(a.budget), "--graph", str(gdir / f"{task}.{arm}.explore.jsonl"),
               "--app_memory", str(gdir / f"_app_unrestorable.{arm}.jsonl")]
        t0 = time.time()
        with (ROOT / "logs" / f"{task}.log").open("a") as f:
          rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env, timeout=1800).returncode
        r = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
        note(f"{task} s{seed} {arm} rc={rc} succ={r.get('is_successful')} steps={r.get('steps')} "
             f"probes={r.get('n_probes')} err={r.get('error')} {time.time() - t0:.0f}s")
  note("phase2 driver finished")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
