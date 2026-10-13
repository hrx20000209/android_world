"""Full AndroidWorld: 116 tasks x 2 passes, BASE/FULL interleaved per task."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
RAW, GDIR = ROOT / "raw" / "full", ROOT / "graph_snapshots" / "full"
ARMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BASE", "FULL"]
SEEDS = [30, 31]

def note(m):
  line = f"- {time.strftime('%m-%d %H:%M:%S')} {m}"
  print(line, flush=True)
  with (ROOT / "reports" / "execution_log.md").open("a") as f:
    f.write(line + "\n")

def tunnel_ok():
  r = subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "8", "http://127.0.0.1:18095/v1/models"], capture_output=True, text=True)
  return "GELAB" in r.stdout

from android_world import registry
tasks = sorted(registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY))
(ROOT / "configs" / "full_tasks.json").write_text(json.dumps({"tasks": tasks, "seeds": SEEDS}, indent=1))
n = 0
for p, seed in enumerate(SEEDS):
  for task in tasks:
    order = ARMS if n % 2 == 0 else ARMS[::-1]
    n += 1
    for arm in order:
      out = RAW / arm / task / f"p{p}"
      if (out / "result.json").exists():
        continue
      if out.exists():
        out.rename(out.with_name(out.name + f"_partial_{int(time.time())}"))
      while not tunnel_ok():
        note("tunnel down"); time.sleep(30)
      env = dict(os.environ, FULL_ARM=arm, SEQ_APPDIR=str(GDIR))
      cmd = [sys.executable, str(ROOT / "scripts" / "full_run.py"), "--task", task, "--seed", str(seed),
             "--variant", f"p{p}_{arm}", "--mode", "agent", "--out", str(out)]
      t0 = time.time()
      try:
        with (ROOT / "logs" / "full.log").open("a") as f:
          rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env, timeout=1500).returncode
      except subprocess.TimeoutExpired:
        rc = "timeout"
      r = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
      note(f"FULLRUN p{p} {arm} {task} rc={rc} succ={r.get('is_successful')} steps={r.get('steps')} calls={r.get('model_calls')} "
           f"jump={r.get('m_jump_steps', 0)} probes={r.get('n_probes')} err={str(r.get('error'))[:60]} {time.time()-t0:.0f}s")
  note(f"FULLRUN PASS {p} DONE")
note("FULLRUN DRIVER DONE " + ",".join(ARMS))
