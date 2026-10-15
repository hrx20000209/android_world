"""Offline paired test of graph-derived screenshot highlighting.

States: BASE decisions where the graph (LOO snapshot, history only) yields a
highlight. Reference = clicks taken at the same state key by SUCCESSFUL
episodes of the same task instance (task + seed, any arm) -> used only for
scoring. Conditions on the identical stored screenshot / goal / history:
  M0 original | HL graph highlight | PL placebo box on another safe slot
"""
import json, sys, threading, queue
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory as M, retrieval as RT, highlight as H, oracle_pilot as P
C, U, R = M.C, M.U, M.R
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "metrics" / "hl_offline.jsonl"
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
    if s.get("kind") != "model" or not s.get("pre"):
      continue
    ui = R.load_ui(ep, s["pre"]["cid"]); tok, fam, _ = C.action_token(ui, s["action"])
    ref[(m["task"], m["seed"], M.state_key(ui))].add(tok)
jobs, snaps = [], {}
for ep in sorted((RAW / "BASE").glob("*/p*")):
  m = json.loads((ep / "meta.json").read_text()); tk = M.template_key(m["task"])
  snaps.setdefault(tk, M.Snapshot(m["task"])); ev = RT.Evidence(snaps[tk])
  steps = [json.loads(l) for l in open(ep / "steps.jsonl")]; steps = [s for s in steps if s.get("kind") == "model"]
  for i, s in enumerate(steps):
    ui = R.load_ui(ep, s["pre"]["cid"])
    rows, info = H.select(ui, m["goal"], ev)
    if not rows:
      continue
    hl = {r["fam"] for r in rows}
    others = [sl for sl in U.enumerate_slots(ui) if sl.safe and C.family_of_sel(sl.sel) not in hl and not sl.sel.get("in_list")]
    pl = [{"slot": others[len(others) // 2], "fam": C.family_of_sel(others[len(others) // 2].sel)}] if others else []
    jobs.append({"ep": str(ep), "task": m["task"], "seed": m["seed"], "goal": m["goal"], "step": s["step"], "cid": s["pre"]["cid"],
                 "hist": [x.get("summary") or "" for x in steps[:i]], "hl": rows, "pl": pl, "info": info,
                 "ref": sorted(ref.get((m["task"], m["seed"], M.state_key(ui)), set())), "orig": s["tok"]})
print("states with highlight", len(jobs), "with reference", sum(bool(j["ref"]) for j in jobs), flush=True)
done = set()
if OUT.exists():
  done = {(j["ep"], j["step"]) for j in map(json.loads, open(OUT))}
q = queue.Queue()
for j in jobs:
  if (j["ep"], j["step"]) not in done:
    q.put(j)
lock = threading.Lock()
def worker(api):
  while True:
    try:
      j = q.get_nowait()
    except queue.Empty:
      return
    ep = Path(j["ep"]); ui = R.load_ui(ep, j["cid"])
    img = Image.open(ep / "img" / f"{j['cid']}.jpg").convert("RGB")
    out = {k: j[k] for k in ("ep", "task", "step", "ref", "orig", "info")}
    out["hl_fams"] = [r["fam"] for r in j["hl"]]; out["pl_fams"] = [r["fam"] for r in j["pl"]]
    for c, rows in (("M0", []), ("HL", j["hl"]), ("PL", j["pl"])):
      if c == "PL" and not rows:
        continue
      im = H.draw(img, ui, rows) if rows else img
      text, lat, usage = P.query(api, j["goal"], j["hist"], im, "")
      parsed, action, _ = P.parse(text, (1080, 2400))
      tok = C.action_token(ui, action.__dict__)[0]
      out[c] = {"tok": tok, "lat": round(lat, 2)}
    with lock:
      with OUT.open("a") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
ths = [threading.Thread(target=worker, args=(f"http://127.0.0.1:{p}/v1/chat/completions",)) for p in (18095, 18096, 18097)]
[t.start() for t in ths]; [t.join() for t in ths]
print("done")
