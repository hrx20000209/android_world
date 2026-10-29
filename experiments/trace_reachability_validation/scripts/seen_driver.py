"""Seen-task split (step-count study on the LabServer vLLM, no on-device load).

Templates of the dev apps are split per app into SEEN / UNSEEN (alternating in name order, by template key).
  warm : SEEN templates at a different seed (31), arm FULL45+WARM: no graph execution (every action is the model's,
         so whole procedures are recorded), fixed exploration budget (the window is not cut by the decision).
  eval : all dev templates at seed 30, arms interleaved per task:
         FULL45       cold, empty graph (same-session control)
         FULL45+SEEN  graph / store / function library copied from the warm phase; same-template functions are
                      allowed (SAME_TPL) and also tap the commit they recorded (AGGR_COMMIT)

usage: seen_driver.py split | warm TAG | eval TAG
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = os.environ.get("SEEN_API", "http://127.0.0.1:18126/v1/chat/completions")   # LabServer vLLM tunnel
MODEL = os.environ.get("SEEN_MODEL", "GELAB-ZERO-4B")
APPS = os.environ.get("SEEN_APPS", "Expense,Recipe,Markor,SimpleCalendar,SimpleSms,Contacts").split(",")
ORDER = open(ROOT / "reports" / "e2e116n_task_order.txt").read().strip().split(",")


def tkey(t):
  return re.sub(r"\d+$", "", t)


def split():
  dev = [t for t in ORDER if any(t.startswith(a) for a in APPS)]
  seen, unseen = [], []
  for a in APPS:
    keys = sorted({tkey(t) for t in dev if t.startswith(a)})
    for i, k in enumerate(keys):
      (seen if i % 2 == 0 else unseen).extend(t for t in dev if tkey(t) == k)
  return dev, seen, unseen


def note(m):
  line = f"- {time.strftime('%m-%d %H:%M:%S')} {m}"
  print(line, flush=True)
  with (ROOT / "reports" / "execution_log.md").open("a") as f:
    f.write(line + "\n")


def api_ok():
  r = subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "8", API.replace("/chat/completions", "/models")],
                     capture_output=True, text=True)
  return "gelab" in r.stdout.lower()


def emulator_ok():
  """The emulator died once (10-28 18:50) and every later episode failed at setup: restart it and wait for boot."""
  st = subprocess.run(["adb", "-s", "emulator-5554", "shell", "getprop", "sys.boot_completed"], capture_output=True, text=True)
  if st.stdout.strip() == "1":
    return
  note("SEEN emulator down: restarting AndroidWorldAvd")
  subprocess.Popen([str(Path.home() / "Library/Android/sdk/emulator/emulator"), "-avd", "AndroidWorldAvd", "-no-snapshot",
                    "-grpc", "8554"], stdout=open(ROOT / "logs" / "emulator_restart.log", "a"), stderr=subprocess.STDOUT,
                   start_new_session=True)
  for _ in range(60):
    time.sleep(10)
    st = subprocess.run(["adb", "-s", "emulator-5554", "shell", "getprop", "sys.boot_completed"], capture_output=True, text=True)
    if st.stdout.strip() == "1":
      time.sleep(20)
      note("SEEN emulator back")
      return
  note("SEEN emulator did not come back")


def run(tag, arm, task, seed, gdir, log, pdir="p0", extra=()):
  out = ROOT / "raw" / tag / arm / task / pdir
  if out.exists():
    shutil.move(str(out), str(ROOT / "aborted" / f"{tag}_{arm}_{task}_{int(time.time())}"))
  out.mkdir(parents=True)
  emulator_ok()
  while not api_ok():
    note(f"SEEN api down ({API}); waiting")
    time.sleep(60)
  env = dict(os.environ, FULL_ARM="FULL45", SEQ_APPDIR=str(gdir), STREAM_IO="1")
  if arm.endswith("+WARM"):
    env.update(NO_EXEC="1", WARM_EXPLORE="1")
  if arm.endswith("+SEEN"):
    env.update(SAME_TPL="1", AGGR_COMMIT="1")
  if "+SEENR" in arm:
    env.update(SAME_TPL="1", AGGR_COMMIT="1", TPL_DEMOS=str(ROOT / "raw" / os.environ.get("SEEN_WARM", "seenW1") / "FULL45+WARM"))
  cmd = [sys.executable, str(ROOT / "scripts" / "full_run.py"), "--task", task, "--seed", str(seed),
         "--variant", f"{tag}_{arm}", "--mode", "agent", "--out", str(out), "--api_url", API, "--model", MODEL, *extra]
  t0 = time.time()
  try:
    with (ROOT / "logs" / f"{tag}.log").open("a") as f:
      rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env, timeout=2400).returncode
  except subprocess.TimeoutExpired:
    rc = "timeout"
  res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
  kinds = {}
  if (out / "steps.jsonl").exists():
    for l in open(out / "steps.jsonl"):
      k = json.loads(l).get("kind")
      kinds[k] = kinds.get(k, 0) + 1
  row = {"task": task, "arm": arm, "seed": seed, "rc": rc, "success": float(res.get("is_successful") or 0) == 1.0,
         "calls": res.get("model_calls"), "steps": res.get("steps"), "probes": res.get("n_probes"),
         "graph_commits": res.get("m_graph_commits", 0), "kinds": kinds, "wall_s": round(time.time() - t0, 1),
         "error": (str(res.get("error"))[:200] if res.get("error") else (None if res else "no result"))}
  with log.open("a") as f:
    f.write(json.dumps(row) + "\n")
  note(f"SEEN {tag} {arm} {task} s{seed} succ={res.get('is_successful')} calls={row['calls']} probes={row['probes']} "
       f"kinds={kinds} err={row['error']}")
  return row


def build_warm_library(tag):
  """Functions of the warm-up episodes: all of a successful episode, and the complete records (every typed text is a
  goal value, the record was committed) of a failed one (most multi-record add tasks run out of budget); the online
  builder only extracts an episode when the next one starts and only self-reported ones, so it is done here."""
  sys.path.insert(0, str(ROOT / "scripts"))
  sys.path.insert(0, str(ROOT.parent / "runtime_graph_exploration" / "scripts"))
  import full9  # noqa: F401,E402  (sets the import paths fgraph needs)
  import fgraph as FG  # noqa: E402
  import memory as MEM  # noqa: E402
  cache = ROOT / "graph_snapshots" / tag / "FULL45" / "function_cache_online.jsonl"
  cache.parent.mkdir(parents=True, exist_ok=True)
  n, keys = 0, []
  with cache.open("w") as fo:
    for ep in sorted((ROOT / "raw" / tag / "FULL45+WARM").glob("*/p*")):
      try:
        ok = json.loads((ep / "result.json").read_text()).get("is_successful") == 1.0
        goal = json.loads((ep / "meta.json").read_text())["goal"]
        for f in FG.extract(ep, goal):
          if not ok and not f.get("complete"):
            continue                 # a failed episode contributes only records it reproduced completely
          f["episode_ok"] = ok
          f["src"], f["template"] = f"warm/{ep.parent.name}", MEM.template_key(ep.parent.name)
          fo.write(json.dumps(f) + "\n")
          n += 1
        keys.append(ep.parent.name)
      except Exception as e:  # pylint: disable=broad-except
        note(f"warm extract failed {ep}: {e!r}")
  cache.with_suffix(".done").write_text("")      # eval episodes of the same task names are still extracted online
  note(f"SEEN warm library {tag}: {n} functions from {len(keys)} successful episodes")


def done_rows(log):
  rows = {}
  if log.exists():
    for l in open(log):
      r = json.loads(l)
      if r.get("error"):
        rows.pop((r["task"], r["arm"], r.get("seed", 30)), None)
      else:
        rows[(r["task"], r["arm"], r.get("seed", 30))] = r
  return rows


def main():
  dev, seen, unseen = split()
  if sys.argv[1] == "split":
    print(json.dumps({"dev": len(dev), "seen": seen, "unseen": unseen}, indent=1))
    return
  tag = sys.argv[2]
  log = ROOT / "reports" / f"{tag}_runs.jsonl"
  json.dump({"apps": APPS, "seen": seen, "unseen": unseen, "api": API}, open(ROOT / "reports" / f"{tag}_split.json", "w"), indent=1)
  if sys.argv[1] == "warm":
    gdir = ROOT / "graph_snapshots" / tag
    for seed, pdir in ((31, "p0"), (32, "p1")):
      for task in seen:
        if (task, "FULL45+WARM", seed) not in done_rows(log):
          run(tag, "FULL45+WARM", task, seed, gdir, log, pdir)
    build_warm_library(tag)
    note(f"SEEN WARM DONE {tag}")
    return
  if sys.argv[1] == "warm2":
    # templates still without a successful demo: two more seeds with a doubled call budget (latency does not
    # count in the warm-up; multi-record add tasks ran out of 20 calls)
    gdir = ROOT / "graph_snapshots" / tag
    okt = {}
    for r in done_rows(log).values():
      okt[tkey(r["task"])] = okt.get(tkey(r["task"]), False) or r["success"]
    miss = [t for t in seen if not okt.get(tkey(t))]
    note(f"SEEN warm2 {tag}: {len(miss)} templates without a successful demo")
    for seed, pdir in ((33, "p2"), (34, "p3")):
      for task in miss:
        if (task, "FULL45+WARM", seed) not in done_rows(log):
          run(tag, "FULL45+WARM", task, seed, gdir, log, pdir, extra=("--max_steps", "40"))
    build_warm_library(tag)
    note(f"SEEN WARM2 DONE {tag}")
    return
  warm = sys.argv[3]                                     # eval TAG WARMTAG
  arms = os.environ.get("SEEN_ARMS", "FULL45,FULL45+SEEN").split(",")
  for arm in arms:
    g = ROOT / "graph_snapshots" / tag / arm.replace("+", "_")
    if "+SEEN" in arm and not g.exists():
      shutil.copytree(ROOT / "graph_snapshots" / warm, g)
    g.mkdir(parents=True, exist_ok=True)
  for task in (seen if os.environ.get("SEEN_ONLY") == "1" else dev):
    for arm in arms:
      if (task, arm, 30) not in done_rows(log):
        run(tag, arm, task, 30, ROOT / "graph_snapshots" / tag / arm.replace("+", "_"), log)
  note(f"SEEN EVAL DONE {tag}")


if __name__ == "__main__":
  main()
