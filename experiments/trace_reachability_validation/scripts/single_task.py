"""Single-task end-to-end latency study (system-level): repeat one stable task under BASE and FULLnn with a chosen
inference backend, and record the real completion time with a per-step breakdown.

The FULL arm starts every repetition from the graph state its full run had just before this task (graph, edge
statistics, online function library restricted to episodes that ended earlier), so repetitions are comparable and
no information from the task itself is used. Other tasks' captures are made reachable by symlinks.

usage: single_task.py TASK REPS ARM1[,ARM2...] [--backend orin|lab] [--load] [--source FULL40]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {"orin": ("http://127.0.0.1:18097/v1/chat/completions", "GELab-Zero-4B-preview"),
            "lab": ("http://127.0.0.1:18095/v1/chat/completions", "GELAB-ZERO-4B")}


def task_start(source, task):
  rows = [json.loads(l) for l in open(ROOT / "raw" / "full" / source / task / "p0" / "steps.jsonl")]
  return min(r["pre"]["t"] for r in rows if r.get("pre") and r["pre"].get("t"))


def warm_graph(source, task, gdir):
  """Copy the source run's graph dir, keeping only records written before the task started."""
  t0 = task_start(source, task)
  src = ROOT / "graph_snapshots" / "full" / source
  if gdir.exists():
    shutil.rmtree(gdir)
  gdir.mkdir(parents=True)
  for f in src.iterdir():
    if f.suffix == ".jsonl":
      with f.open() as fi, (gdir / f.name).open("w") as fo:
        for line in fi:
          try:
            o = json.loads(line)
          except ValueError:
            continue
          if o.get("t", 0) < t0 and not str(o.get("instance", "")).startswith(task + "/"):
            fo.write(line)
    elif f.name in ("edge_wait.json", "known_states.txt"):
      shutil.copy(f, gdir / f.name)
  # online function library: only episodes of the source run that ended before this task started
  done = []
  for res in (ROOT / "raw" / "full" / source).glob("*/p0/result.json"):
    t = res.parts[-3]
    steps = res.parent / "steps.jsonl"
    try:
      ts = [json.loads(l)["pre"]["t"] for l in open(steps) if '"pre"' in l and json.loads(l).get("pre")]
    except Exception:  # pylint: disable=broad-except
      continue
    if t != task and ts and max(ts) < t0:
      done.append(t)
  return t0, sorted(done)


def link_raw(source, task, raw_dir, ended_before):
  raw_dir.mkdir(parents=True, exist_ok=True)
  for d in (ROOT / "raw" / "full" / source).iterdir():
    if d.is_dir() and d.name != task and not (raw_dir / d.name).exists():
      (raw_dir / d.name).symlink_to(d.resolve())
  return ended_before


def run_one(task, arm, rep, backend, load, source, seed=30):
  url, model = BACKENDS[backend]
  label = f"{task}__{arm}__{backend}{'_load' if load else ''}"
  raw_dir = ROOT / "raw" / "single" / label
  appdir = ROOT / "graph_snapshots" / "single" / label
  arm_name = arm if not arm.startswith("FULL") else arm + "R"      # replicate suffix: same behaviour, own dir
  gdir = appdir / arm_name
  if arm.startswith("FULL"):
    t0, ended = warm_graph(source, task, gdir)
    link_raw(source, task, raw_dir, ended)
    # pre-build the online library from the ended episodes only (build_online scans raw_dir; mark others done)
    done_marker = gdir / "function_cache_online.done"
    others = [d.name for d in raw_dir.iterdir() if d.is_symlink() and d.name not in ended]
    done_marker.write_text("\n".join(others + [task]))
  else:
    raw_dir.mkdir(parents=True, exist_ok=True)
    gdir.mkdir(parents=True, exist_ok=True)
  out = raw_dir / task / f"r{rep}"
  if out.exists():
    shutil.rmtree(out)
  env = dict(os.environ, FULL_ARM=arm_name, SEQ_APPDIR=str(appdir), ONDEVICE_LOAD="1" if load else "0")
  cmd = [sys.executable, str(ROOT / "scripts" / "full_run.py"), "--task", task, "--seed", str(seed),
         "--variant", f"r{rep}_{arm_name}", "--mode", "agent", "--out", str(out), "--api_url", url, "--model", model]
  t_start = time.time()
  with (ROOT / "logs" / "single.log").open("a") as f:
    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env, timeout=3600).returncode
  wall = time.time() - t_start
  res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
  return {"task": task, "arm": arm, "rep": rep, "backend": backend, "load": load, "rc": rc, "wall_s": round(wall, 1),
          "success": res.get("is_successful"), "calls": res.get("model_calls"), "steps": res.get("steps"),
          "probes": res.get("n_probes"), "out": str(out)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("task")
  ap.add_argument("reps", type=int)
  ap.add_argument("arms")
  ap.add_argument("--backend", default="orin")
  ap.add_argument("--load", action="store_true")
  ap.add_argument("--source", default="FULL40")
  a = ap.parse_args()
  log = ROOT / "reports" / "single_task_runs.jsonl"
  for rep in range(1, a.reps + 1):
    for arm in a.arms.split(","):
      r = run_one(a.task, arm, rep, a.backend, a.load, a.source)
      print(json.dumps(r, ensure_ascii=False), flush=True)
      with log.open("a") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
  main()
