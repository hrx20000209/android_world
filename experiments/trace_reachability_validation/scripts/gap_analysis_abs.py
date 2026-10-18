"""gap_analysis with the variance-aware page identity (state_abstraction). The abstraction's dynamic-item
statistics are learned from captures of OTHER tasks observed before each step (causal)."""
import glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9, explore_value as EV
import state_abstraction as SA
R, U, C = full9.R, full9.U, full9.C


def fam_of(o):
  if o.get("origin") == "explore" and o.get("selector"):
    return C.family_of_sel(o["selector"])
  return o["slot_id"]


def main(arm, use_sources=("agent", "explore")):
  raw = Path("raw/full") / arm
  obs = sorted((json.loads(l) for l in open(f"graph_snapshots/full/{arm}/graph.jsonl")), key=lambda o: o.get("t", 0))
  obs = [o for o in obs if o.get("dst") and o.get("src_cid") and o.get("dst_cid")]
  steps = []
  for f in glob.glob(str(raw / "*" / "p0" / "steps.jsonl")):
    d = Path(f).parent; t = d.parts[-2]
    rows = [json.loads(l) for l in open(f)]
    for i, r in enumerate(rows):
      if r.get("kind") != "model" or not r.get("pre") or not r.get("tok"):
        continue
      nxt = next((x for x in rows[i + 1:] if x.get("pre")), None)
      if nxt is None:
        continue
      steps.append((r["pre"]["t"], t, r["tok"], d, r["pre"]["cid"], nxt["pre"]["cid"]))
  steps.sort()
  ab = SA.Abstraction()
  seen = []                                   # (src ui, fam, dst ui, task, origin)
  res = Counter(); j = 0
  for ts, task, tok, d, cid_p, cid_q in steps:
    while j < len(obs) and obs[j].get("t", 0) < ts:
      o = obs[j]; j += 1
      org = "explore" if o.get("origin") == "explore" else "agent"
      if org not in use_sources:
        continue
      tk, var = o["instance"].split("/")
      su = EV.load_cap(raw, o["instance"], o["src_cid"]); du = EV.load_cap(raw, o["instance"], o["dst_cid"])
      if su is None or du is None:
        continue
      if tk != task:
        ab.observe(su); ab.observe(du)
      seen.append((su, fam_of(o), du, tk))
    try:
      up, uq = R.load_ui(d, cid_p), R.load_ui(d, cid_q)
    except Exception:
      continue
    # keys are recomputed with the current statistics (identity sharpens as observations accumulate)
    P, Q = ab.key(up), ab.key(uq)
    if P == Q:
      continue
    res["steps"] += 1
    pages = defaultdict(set); acts = set(); edges = defaultdict(Counter); etasks = defaultdict(set)
    for su, fam, du, tk in seen:
      s, dd = ab.key(su), ab.key(du)
      pages[s].add(tk); pages[dd].add(tk); acts.add(s[:2]); acts.add(dd[:2])
      if s != dd:
        edges[(s, fam)][dd] += 1; etasks[(s, fam)].add(tk)
    if not (pages.get(P, set()) - {task}):
      res["break_page" if P[:2] in acts else "new_activity"] += 1
      continue
    if not (etasks.get((P, tok), set()) - {task}):
      res["break_edge"] += 1
      continue
    if edges[(P, tok)].most_common(1)[0][0] != Q:
      res["break_landing"] += 1
      continue
    res["reusable"] += 1
  n = res["steps"]
  print(f"[{arm} {'+'.join(use_sources)}] steps {n}  " + "  ".join(f"{k} {res[k]} ({res[k] / max(1, n):.0%})" for k in
        ("reusable", "break_page", "break_edge", "break_landing", "new_activity")))


if __name__ == "__main__":
  for a in sys.argv[1:]:
    main(a, ("agent",))
    main(a, ("agent", "explore"))
