"""ORACLE lookahead bound: bypass at step t+1 enabled by a depth-2 probe in window t.
Step t+1's click b is a safe non-entity nav slot, (s_{t+1}, b) unknown in G0;
window t must fit a depth-2 probe sequence (2 actions + rollback, ~2x probe time).
Also counts the same with b known in G0 (historical reuse, NOT exploration)."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import stage1_audit as S
M, RT, C, U, R = S.M, S.RT, S.C, S.U, S.R
c = Counter(); snaps = {}
for ep in sorted((M.TRV / "raw/full/BASE").glob("*/p*")):
  meta = json.loads((ep / "meta.json").read_text()); res = json.loads((ep / "result.json").read_text())
  succ = res.get("is_successful") == 1.0; gn = M.norm(meta["goal"]); tk = M.template_key(meta["task"])
  snaps.setdefault(tk, M.Snapshot(meta["task"])); snap = snaps[tk]
  steps = [json.loads(l) for l in open(ep / "steps.jsonl")]; steps = [s for s in steps if s.get("kind") == "model"]
  prev = None
  for s in steps:
    ui = R.load_ui(ep, s["pre"]["cid"]); key = M.state_key(ui)
    tok, fam, _ = C.action_token(ui, s["action"])
    nav = {C.family_of_sel(sl.sel) for sl in S.nav_slots(ui, gn)}
    if prev is not None and fam in nav and prev["nav"]:
      known = bool(snap.by_edge.get((key, fam)))
      fits2 = (prev["lat"] or 0) >= 2 * S.PROBE_S
      c[("known_in_G0" if known else "unknown_in_G0", "fits2" if fits2 else "nofit", "succ" if succ else "fail")] += 1
      if succ and not known: print("  lookahead-unknown succ", meta["task"], ep.name, s["step"], fam[:45], "prev_lat", prev["lat"])
    prev = {"nav": fam in nav, "lat": s.get("model_latency_s")}
for k, v in sorted(c.items()): print(k, v)
