"""Stage 1: why does each BASE decision state (not) yield a valid exploration query?

First matching reason, in this order (each later reason assumes the earlier
ones do not hold):
  outside_app            launcher / system UI
  anchor_visible         a goal anchor is already on screen: the next decision is
                         semantic (which visible element / value), not navigation
  no_safe_frontier       no safe non-entity click slot (safety eligibility)
  state_unseen           page absent from LOO memory (graph coverage)
  all_known_no_anchor    every safe slot has evidence; none exposes a goal anchor
                         within 2 hops (semantic applicability of history)
  unseen_frontier_only   some safe slots unknown, none evidenced toward an anchor:
                         no query rule fires (missing query-generation rule)
  valid_query            unknown slots block a unique / ambiguous anchor branch
  direct                 graph alone authorizes one slot (no query needed)
Also per state: safe unseen frontiers, whether they fit one inference window.
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory as M, retrieval as RT
C, U, R = M.C, M.U, M.R
PROBE_S, rows = 1.69, []
snaps = {}
for ep in sorted((M.TRV / "raw/full/BASE").glob("*/p*")):
  meta = json.loads((ep / "meta.json").read_text()); res = json.loads((ep / "result.json").read_text())
  task, goal = meta["task"], meta["goal"]; succ = res.get("is_successful") == 1.0
  tk = M.template_key(task)
  snaps.setdefault(tk, M.Snapshot(task)); snap = snaps[tk]
  ev = RT.Evidence(snap); gn = M.norm(goal)
  steps = [json.loads(l) for l in open(ep / "steps.jsonl")]
  steps = [s for s in steps if s.get("kind") == "model"]
  for i, s in enumerate(steps):
    ui = R.load_ui(ep, s["pre"]["cid"])
    tok, fam, sel = C.action_token(ui, s["action"])
    cls_rows, info = RT.classify(ui, goal, ev)
    nav = [r for r in cls_rows if r["slot"].safe and not r["slot"].sel.get("in_list") and not RT.goal_named(r["label"], gn)]
    unk = [r for r in nav if r["status"] == "UNKNOWN"]
    toward = [r for r in nav if r["cls"] in ("ANCHOR1", "ANCHOR2", "DIRECT", "ENVIRONMENT_UNCERTAIN")]
    if ui.package in ("com.google.android.apps.nexuslauncher", "com.android.systemui") or "launcher" in ui.activity.lower():
      why = "outside_app"
    elif info["anchors_here"]:
      why = "anchor_visible"
    elif not nav:
      why = "no_safe_frontier"
    elif not snap.by_state.get(M.state_key(ui)):
      why = "state_unseen"
    elif info["direct"] is not None:
      why = "direct"
    elif toward and unk:
      why = "valid_query"
    elif not unk:
      why = "all_known_no_anchor"
    else:
      why = "unseen_frontier_only"
    mrow = next((r for r in cls_rows if r["fam"] == fam), None) if fam else None
    m_kind = ("nonclick" if not fam else ("unslotted" if mrow is None else
              ("nav_unknown" if mrow in unk else ("nav_known" if mrow in nav else
               ("entity_or_named" if mrow["cls"] == "PARAMETERIZED" else "unsafe")))))
    lat = s.get("model_latency_s") or 0
    rows.append({"task": task, "pass": ep.name, "step": s["step"], "succ": succ, "why": why, "n_nav": len(nav),
                 "n_unknown": len(unk), "fit": int(lat // PROBE_S), "model": m_kind, "tok": tok[:60],
                 "next_same": i + 1 < len(steps) and steps[i + 1]["pre"].get("summary", {}).get("struct") == s["pre"].get("summary", {}).get("struct"),
                 "ep": str(ep), "cid": s["pre"]["cid"], "goal": goal})
out = Path(__file__).resolve().parent.parent / "metrics"
(out / "opportunity_states.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
c = Counter(r["why"] for r in rows)
print("states", len(rows))
for k, v in c.most_common():
  sub = [r for r in rows if r["why"] == k]
  mk = Counter(r["model"] for r in sub)
  print(f"{k:22s} {v:5d} {v/len(rows):5.1%}  succ-ep {sum(r['succ'] for r in sub):4d}  model: {dict(mk.most_common(4))}")
uf = [r for r in rows if r["why"] == "unseen_frontier_only"]
if uf:
  print("unseen_frontier_only: unknown frontiers per state median", sorted(r["n_unknown"] for r in uf)[len(uf)//2],
        "| states where all unknowns fit 1 window", sum(r["n_unknown"] <= r["fit"] for r in uf),
        "| >=1 fits", sum(r["fit"] >= 1 for r in uf),
        "| model then took an unknown frontier", sum(r["model"] == "nav_unknown" for r in uf))
