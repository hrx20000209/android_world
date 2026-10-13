"""Batch driver: per task, reference (with retries) -> variants with exploration
-> strict replay -> structure-aware replay (+ separately labelled recovery).

Resumable: an episode with result.json is skipped. Refuses to start if another
driver or runner is alive (two runs on one emulator corrupt both).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "scripts" / "rgs_run.py"


def sh(cmd: list[str], log: Path, timeout: int = 1800) -> int:
  with log.open("a") as f:
    f.write(f"\n### {time.strftime('%H:%M:%S')} {' '.join(cmd)}\n")
    f.flush()
    try:
      return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, cwd=ROOT).returncode
    except subprocess.TimeoutExpired:
      f.write("TIMEOUT\n")
      return -9


def result(ep: Path) -> dict | None:
  p = ep / "result.json"
  return json.loads(p.read_text()) if p.exists() else None


def tunnel_ok() -> bool:
  r = subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "8", "http://127.0.0.1:18095/v1/models"],
                     capture_output=True, text=True)
  return "GELAB" in r.stdout


def wait_tunnel(note):
  while not tunnel_ok():
    note("tunnel down; waiting 30s")
    time.sleep(30)


def episode(args, task, seed, variant, mode, out: Path, extra: list[str], note) -> dict | None:
  if result(out) is not None:
    return result(out)
  if out.exists():
    # partial episode from an interrupted run: keep it for inspection, rerun fresh
    out.rename(out.with_name(out.name + f"_partial_{int(time.time())}"))
  wait_tunnel(note)
  cmd = [sys.executable, str(RUN), "--task", task, "--seed", str(seed), "--variant", variant,
         "--mode", mode, "--out", str(out)] + extra
  t0 = time.time()
  rc = sh(cmd, ROOT / "logs" / f"{task}.log")
  r = result(out)
  note(f"{task} {variant} {mode} rc={rc} succ={r and r.get('is_successful')} steps={r and r.get('steps')} "
       f"err={r and r.get('error')} {time.time() - t0:.0f}s")
  if r and r.get("error") and "LlamaCpp" in str(r.get("error")):
    # inference failure voids the episode; retry once after the tunnel is back
    out.rename(out.with_name(out.name + f"_void_{int(time.time())}"))
    wait_tunnel(note)
    sh(cmd, ROOT / "logs" / f"{task}.log")
    r = result(out)
  return r


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--manifest", required=True)
  ap.add_argument("--raw", default="raw/main")
  ap.add_argument("--graphs", default="graphs/main")
  ap.add_argument("--budget", type=float, default=6.0)
  ap.add_argument("--ref_attempts", type=int, default=3)
  ap.add_argument("--only")
  args = ap.parse_args()
  others = subprocess.run(["pgrep", "-fl", "rgs_driver.py|rgs_run.py"], capture_output=True, text=True).stdout
  mine = str(__import__("os").getpid())
  others = [l for l in others.splitlines() if l and not l.startswith(mine + " ")]
  if others:
    print("refusing to start; already running:\n" + "\n".join(others))
    return 2
  manifest = json.loads((ROOT / args.manifest).read_text())
  raw, graphs = ROOT / args.raw, ROOT / args.graphs
  raw.mkdir(parents=True, exist_ok=True)
  graphs.mkdir(parents=True, exist_ok=True)
  exec_log = ROOT / "reports" / "execution_log.md"

  def note(msg):
    line = f"- {time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with exec_log.open("a") as f:
      f.write(line + "\n")

  for t in manifest["tasks"]:
    task, seeds = t["task"], t["seeds"]
    if args.only and task not in args.only.split(","):
      continue
    tdir = raw / task
    gx = graphs / f"{task}.explore.jsonl"
    gr = graphs / f"{task}.replay.jsonl"
    xflags = ["--explore", "--budget", str(args.budget), "--graph", str(gx),
              "--app_memory", str(graphs / "_app_unrestorable.jsonl")]
    # 1. clean reference (no exploration): first successful attempt
    ref = None
    for a in range(args.ref_attempts):
      ep = tdir / f"v0_ref{a}"
      r = episode(args, task, seeds[0], f"v0_ref{a}", "agent", ep, [], note)
      if r and r.get("is_successful") == 1.0 and not r.get("error"):
        ref = ep
        break
    if ref is None:
      cands = [tdir / f"v0_ref{a}" for a in range(args.ref_attempts)
               if result(tdir / f"v0_ref{a}") and not result(tdir / f"v0_ref{a}").get("error")]
      ref = cands[0] if cands else None
      note(f"{task}: no successful reference; using {ref} (flagged)")
    (tdir / "reference.json").write_text(json.dumps({"ref": str(ref) if ref else None}))
    # 2. agent + exploration on every variant, v0 first (shared incremental graph)
    for i, s in enumerate(seeds):
      episode(args, task, s, f"v{i}", "agent", tdir / f"v{i}_agent", xflags, note)
    # 2b. no-exploration control on the other variants (paired with 2)
    for i, s in enumerate(seeds[1:], start=1):
      episode(args, task, s, f"v{i}_noexp", "agent", tdir / f"v{i}_noexp", [], note)
    if ref is None:
      continue
    # 3. replays of the reference on the other variants
    for i, s in enumerate(seeds[1:], start=1):
      episode(args, task, s, f"v{i}", "replay_strict", tdir / f"v{i}_strict",
              ["--ref", str(ref), "--graph", str(gr)], note)
      episode(args, task, s, f"v{i}", "replay_struct", tdir / f"v{i}_struct",
              ["--ref", str(ref), "--graph", str(gr), "--recover"], note)
  note("driver finished")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
