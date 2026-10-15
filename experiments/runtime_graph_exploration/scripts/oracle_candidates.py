"""Stage 2 state selection (ORACLE upper bound: uses future trajectories only to pick states).

Candidate = BASE decision state s in a FAILED episode such that a SUCCESSFUL
episode of the same task and seed (same task instance; other arms) acted at a
state with the same activity+layout and took a different safe navigation slot
that is present on s. Also flags loops (state revisited >= 3 times)."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory as M
C, U, R = M.C, M.U, M.R
RAW = M.TRV / "raw/full"
succ_eps = defaultdict(list)
for ep in RAW.glob("*/*/p*"):
  if ep.parent.parent.name == "BASE" or not (ep / "result.json").exists():
    continue
  r = json.loads((ep / "result.json").read_text())
  if r.get("is_successful") == 1.0:
    m = json.loads((ep / "meta.json").read_text())
    succ_eps[(m["task"], m["seed"])].append(ep)
cands = []
for ep in sorted((RAW / "BASE").glob("*/p*")):
  m = json.loads((ep / "meta.json").read_text()); r = json.loads((ep / "result.json").read_text())
  if r.get("is_successful") == 1.0 or not succ_eps.get((m["task"], m["seed"])):
    continue
  # oracle actions by state key from successful same-instance episodes (model clicks only)
  oracle = defaultdict(Counter)
  for se in succ_eps[(m["task"], m["seed"])]:
    for l in open(se / "steps.jsonl"):
      s = json.loads(l)
      if s.get("kind") != "model" or (s.get("action") or {}).get("action_type") != "click":
        continue
      ui = R.load_ui(se, s["pre"]["cid"])
      tok, fam, _ = C.action_token(ui, s["action"])
      if fam:
        oracle[ui.activity][fam] += 1
  steps = [json.loads(l) for l in open(ep / "steps.jsonl")]
  steps = [s for s in steps if s.get("kind") == "model"]
  seen = Counter()
  for i, s in enumerate(steps):
    ui = R.load_ui(ep, s["pre"]["cid"]); key = M.state_key(ui)
    seen[key] += 1
    tok, fam, _ = C.action_token(ui, s["action"])
    loop = seen[key] >= 3
    if ui.activity not in oracle and not loop:
      continue
    slots = {C.family_of_sel(sl.sel): sl for sl in U.enumerate_slots(ui)}
    if loop and fam:
      cands.append({"task": m["task"], "seed": m["seed"], "ep": str(ep), "step_idx": i, "step": s["step"],
                    "cid": s["pre"]["cid"], "goal": m["goal"], "base_tok": tok, "oracle_fam": "LOOP", "oracle_n": 0,
                    "revisits": seen[key], "n_safe": 0, "kind": "loop"})
    for ofam, n in oracle.get(ui.activity, {}).items():
      sl = slots.get(ofam)
      if sl is None or ofam == fam or not sl.safe or sl.sel.get("in_list"):
        continue
      cands.append({"task": m["task"], "seed": m["seed"], "ep": str(ep), "step_idx": i, "step": s["step"],
                    "cid": s["pre"]["cid"], "goal": m["goal"], "base_tok": tok, "oracle_fam": ofam, "oracle_n": n,
                    "revisits": seen[key], "n_safe": sum(1 for x in slots.values() if x.safe), "kind": "branch"})
out = Path(__file__).resolve().parent.parent / "metrics" / "oracle_candidates.json"
out.write_text(json.dumps(cands, indent=1))
print(len(cands), "candidates from", len({c["ep"] for c in cands}), "failed episodes,", len({c["task"] for c in cands}), "tasks")
first = {}
for c in cands:
  first.setdefault((c["ep"], c["oracle_fam"]), c)
for c in first.values():
  print(f"{c['task'][:34]:34s} {Path(c['ep']).name} st{c['step']:2d} rv{c['revisits']} base={c['base_tok'][:32]:32s} oracle={c['oracle_fam'][:40]}")
