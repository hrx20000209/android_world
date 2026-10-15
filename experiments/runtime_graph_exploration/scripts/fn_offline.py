"""Offline paired test: EchoTrail-style function-path injection (K <= 2).
M0 none | FN functions retrieved for this goal | PL functions retrieved for a
different template's goal of the same app (same format, wrong content)."""
import json, sys, threading, queue, random
from collections import defaultdict
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory as M, functions as F, oracle_pilot as P
C, U, R = M.C, M.U, M.R
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "metrics" / "fn_offline.jsonl"
RAW = M.TRV / "raw/full"
ref = defaultdict(set)
for ep in RAW.glob("*/*/p*"):
  try:
    r = json.loads((ep / "result.json").read_text()); m = json.loads((ep / "meta.json").read_text())
  except Exception:
    continue
  if r.get("is_successful") != 1.0:
    continue
  for l in open(ep / "steps.jsonl"):
    s = json.loads(l)
    if s.get("kind") == "model" and s.get("pre"):
      ui = R.load_ui(ep, s["pre"]["cid"]); ref[(m["task"], m["seed"], M.state_key(ui))].add(C.action_token(ui, s["action"])[0])
goals_by_pkg = defaultdict(set)
for ep in (RAW / "BASE").glob("*/p0"):
  m = json.loads((ep / "meta.json").read_text())
  for l in open(ep / "steps.jsonl"):
    s = json.loads(l); ui = R.load_ui(ep, s["pre"]["cid"])
    if "launcher" not in ui.package:
      goals_by_pkg[ui.package].add((m["task"], m["goal"])); break
jobs = []
rng = random.Random(0)
for ep in sorted((RAW / "BASE").glob("*/p*")):
  m = json.loads((ep / "meta.json").read_text())
  snap = M.Snapshot(m["task"]); lib = F.Library(snap)
  steps = [json.loads(l) for l in open(ep / "steps.jsonl")]; steps = [s for s in steps if s.get("kind") == "model"]
  for i, s in enumerate(steps):
    ui = R.load_ui(ep, s["pre"]["cid"])
    if "launcher" in ui.package:
      continue
    fn = lib.retrieve(m["goal"], ui.package, M.state_key(ui), M.page_labels(ui))
    if not fn:
      continue
    others = sorted(g for t, g in goals_by_pkg.get(ui.package, ()) if M.template_key(t) != M.template_key(m["task"]))
    pl = []
    for g in rng.sample(others, len(others)):
      pl = lib.retrieve(g, ui.package, M.state_key(ui), M.page_labels(ui))
      if pl and {f["target"] for f in pl} != {f["target"] for f in fn}:
        break
      pl = []
    jobs.append({"ep": str(ep), "task": m["task"], "goal": m["goal"], "step": s["step"], "cid": s["pre"]["cid"],
                 "hist": [x.get("summary") or "" for x in steps[:i]], "fn": F.render(fn), "pl": F.render(pl),
                 "ref": sorted(ref.get((m["task"], m["seed"], M.state_key(ui)), set()))})
print("states with functions", len(jobs), "with reference", sum(bool(j["ref"]) for j in jobs), flush=True)
done = {(j["ep"], j["step"]) for j in map(json.loads, open(OUT))} if OUT.exists() else set()
q = queue.Queue(); [q.put(j) for j in jobs if (j["ep"], j["step"]) not in done]
lock = threading.Lock()
def worker(api):
  while True:
    try:
      j = q.get_nowait()
    except queue.Empty:
      return
    ep = Path(j["ep"]); ui = R.load_ui(ep, j["cid"]); img = Image.open(ep / "img" / f"{j['cid']}.jpg").convert("RGB")
    out = {k: j[k] for k in ("ep", "task", "step", "ref", "fn", "pl")}
    for c, extra in (("M0", ""), ("FN", j["fn"]), ("PL", j["pl"])):
      if c == "PL" and not extra:
        continue
      text, lat, usage = P.query(api, j["goal"], j["hist"], img, extra)
      parsed, action, _ = P.parse(text, (1080, 2400))
      out[c] = {"tok": C.action_token(ui, action.__dict__)[0], "status": action.action_type}
    with lock:
      with OUT.open("a") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
ths = [threading.Thread(target=worker, args=(f"http://127.0.0.1:{p}/v1/chat/completions",)) for p in (18095, 18096, 18097)]
[t.start() for t in ths]; [t.join() for t in ths]
print("done")
