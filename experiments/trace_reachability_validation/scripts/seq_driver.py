"""Run same-app consecutive task sequences for each arm (interleaved by episode)."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ARMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BASE", "G", "GX"]
RAW, APPDIR = ROOT / "raw" / "seq", ROOT / "graph_snapshots" / "seq"

def note(m):
  line = f"- {time.strftime('%m-%d %H:%M:%S')} {m}"
  print(line, flush=True)
  with (ROOT / "reports" / "execution_log.md").open("a") as f:
    f.write(line + "\n")

def tunnel_ok():
  r = subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "8", "http://127.0.0.1:18095/v1/models"], capture_output=True, text=True)
  return "GELAB" in r.stdout

man = json.loads((ROOT / "configs" / "seq_manifest.json").read_text())
n = 0
for app in man["apps"]:
  for p in range(3):
    for t in app["tasks"]:
      seed = t["seeds"][p]
      for j in range(len(ARMS)):
        arm = ARMS[(j + n) % len(ARMS)]
        out = RAW / arm / t["task"] / f"p{p}_s{seed}"
        if (out / "result.json").exists():
          continue
        if out.exists():
          out.rename(out.with_name(out.name + f"_partial_{int(time.time())}"))
        while not tunnel_ok():
          note("tunnel down"); time.sleep(30)
        env = dict(os.environ, SEQ_ARM=arm, SEQ_APPDIR=str(APPDIR))
        cmd = [sys.executable, str(ROOT / "scripts" / "seq_run.py"), "--task", t["task"], "--seed", str(seed),
               "--variant", f"p{p}_s{seed}_{arm}", "--mode", "agent", "--out", str(out)]
        t0 = time.time()
        with (ROOT / "logs" / "seq.log").open("a") as f:
          rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env, timeout=1800).returncode
        r = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
        note(f"SEQ {arm} {t['task']} p{p} s{seed} rc={rc} succ={r.get('is_successful')} steps={r.get('steps')} calls={r.get('model_calls')} "
             f"jumps={r.get('jumps')}/{r.get('jump_steps')} probes={r.get('n_probes')} err={r.get('error')} {time.time()-t0:.0f}s")
      n += 1
note("SEQ DRIVER DONE")
