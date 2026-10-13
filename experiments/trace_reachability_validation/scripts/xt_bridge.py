"""Same-app consecutive tasks: can cross-task knowledge + a continuously built
exploration graph bridge the gap where traces of the CURRENT task are absent?

For every agent click in a task T of an app with >=2 tasks, the only allowed
knowledge is: other tasks of the same app (their agent traces and their
exploration/agent transitions) + the evaluated episode's own past.

Signals (all counts / set membership, combined lexicographically):
  trace   : continuation counts of other-task traces at the current page
            (longest matched suffix; struct key, layout backoff)
  bridge  : the slot's destination page, known from the graph, is a page on
            which other-task agents acted (a known workflow page)
  known   : the slot's destination is in the graph at all
  goal    : the slot label occurs verbatim in the task goal (deterministic)
Graphs: G_agent (only edges the agents themselves executed) vs G_full
(+ explorer probes)  -> isolates what exploration adds to bridging.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import pa_trace as PA  # noqa: E402

U = C.U
OUT = C.ROOT / "analysis" / "xt_bridge"


def norm(t):
  return " ".join(str(t or "").lower().split())


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  eps = C.episodes()
  obs = C.graph_observations()
  apps = defaultdict(set)
  for e in eps:
    apps[e["app"]].add(e["task"])
  multi = {a for a, ts in apps.items() if len(ts) >= 2}
  seqS = {e["ep"]: PA.build_episode_seq(e) for e in eps if e["app"] in multi}
  seqL = {e["ep"]: PA.build_episode_seq(e, key="layout_sig") for e in eps if e["app"] in multi}
  for e in eps:
    if e["ep"] in seqS:
      for a, b in zip(seqS[e["ep"]], seqL[e["ep"]]):
        a["layout"] = b["state"]
  rows = []
  for e in [x for x in eps if x["app"] in multi]:
    T = e["task"]
    hist = [h for h in eps if h["app"] == e["app"] and h["task"] != T]
    idxS, idxL = PA.Index(), PA.Index()
    work_pages = set()
    for h in hist:
      sS = [dict(x, fam=x["tok"]) for x in seqS[h["ep"]]]   # all actions (stage-aware)
      idxS.add_seq(sS)
      idxL.add_seq([dict(x, state=x["layout"]) for x in sS])
      for x in seqS[h["ep"]]:
        work_pages.add(x["layout"])
    hist_obs = [o for o in obs if C.APP_OF.get(o["task"]) == e["app"] and o["task"] != T]
    seq = seqS[e["ep"]]
    for i, st in enumerate(seq):
      if not st["fam"]:
        continue
      own_obs = [o for o in obs if o["ep"] == e["ep"] and o["t"] < st["t"]]
      ui = C.load_ui(e["ep"], st["cid"])
      safe, first = [], {}
      for s in U.enumerate_slots(ui):
        if s.safe:
          f = C.family_of_sel(s.sel)
          if f not in first:
            first[f] = s
            safe.append(f)
      if st["fam"] not in safe:
        continue
      # trace counts (struct, then layout)
      own_pref = [dict(x, fam=x["tok"]) for x in seq[:i]]
      cnt = Counter()
      for idx, key in ((idxS, "state"), (idxL, "layout")):
        q = [dict(x, state=x[key]) for x in own_pref] + [{"state": st[key if key == "layout" else "struct"], "tok": "", "fam": None}]
        c = PA.overlay(idx, q, len(q) - 1)
        for k in range(PA.K_MAX, -1, -1):
          if c[k] and sum(c[k].values()):
            cnt = c[k]
            break
        if cnt:
          break
      # graph destinations
      def dsts(pool):
        m = {}
        for o in pool:
          if o.get("dst") and o.get("family") and (o["src"]["struct_sig"] == ui.struct_sig or o["src"]["layout_sig"] == ui.layout_sig):
            if o.get("outcome") == "no_change":
              continue
            m.setdefault(o["family"], o["dst"]["layout_sig"])
        return m
      g_agent = dsts([o for o in hist_obs + own_obs if o["origin"] == "agent"])
      g_full = dsts(hist_obs + own_obs)
      # upper bound: exploration had covered every page of this app (all runs,
      # any time). Destinations are structural and 98% stable across runs.
      # no leakage: only explorer probes (any task/time; environment structure,
      # chosen without knowledge of agent intent) + other-task agent edges
      g_oracle = dsts([o for o in obs if C.APP_OF.get(o["task"]) == e["app"]
                       and (o["origin"] == "explore" or o["task"] != T)])
      goal = norm(e["goal"])
      pos = {f: j for j, f in enumerate(safe)}

      def lab(f):
        n = ui.nodes[first[f].node_idx]
        return norm(U.label_of(ui, n))

      def key_bridge(g):
        return lambda f: (-cnt[f], not (g.get(f) in work_pages), f not in g, pos[f])
      ranks = {
          "P0": sorted(safe, key=lambda f: pos[f]),
          "TRACE": sorted(safe, key=lambda f: (-cnt[f], pos[f])),
          "BR_agent": sorted(safe, key=key_bridge(g_agent)),
          "BR_full": sorted(safe, key=key_bridge(g_full)),
          "BR_oracleGraph": sorted(safe, key=key_bridge(g_oracle)),
          "GOAL+BR_oracleGraph": sorted(safe, key=lambda f: (not (len(lab(f)) >= 3 and lab(f) in goal), -cnt[f],
                                                             not (g_oracle.get(f) in work_pages), f not in g_oracle, pos[f])),
          "GOAL": sorted(safe, key=lambda f: (not (len(lab(f)) >= 3 and lab(f) in goal), pos[f])),
          "GOAL+BR_full": sorted(safe, key=lambda f: (not (len(lab(f)) >= 3 and lab(f) in goal), -cnt[f],
                                                      not (g_full.get(f) in work_pages), f not in g_full, pos[f])),
      }
      tgt = st["fam"]
      r = {"app": e["app"], "task": T, "ep": e["ep"], "step": st["step"], "gap": not any(cnt[f] for f in safe),
           "n_cands": len(safe), "target_dst_known_agent": tgt in g_agent, "target_dst_known_full": tgt in g_full,
           "target_bridge_full": g_full.get(tgt) in work_pages,
           "n_bridge_cands_oracle": sum(g_oracle.get(f) in work_pages for f in safe),
           "target_bridge_oracle": g_oracle.get(tgt) in work_pages, "dst_known_oracle": sum(f in g_oracle for f in safe)}
      for p, lst in ranks.items():
        r["rank_" + p] = lst.index(tgt) + 1
      rows.append(r)
  (OUT / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
  pols = ["P0", "TRACE", "BR_agent", "BR_full", "BR_oracleGraph", "GOAL", "GOAL+BR_full", "GOAL+BR_oracleGraph"]
  S = {}
  for name, sub in (("all", rows), ("gap (no cross-task trace at page)", [r for r in rows if r["gap"]]),
                    ("covered by cross-task trace", [r for r in rows if not r["gap"]])):
    S[name] = {"n": len(sub)} | {p: {"top1": round(np.mean([r[f"rank_{p}"] == 1 for r in sub]), 3) if sub else None,
                                    "top3": round(np.mean([r[f"rank_{p}"] <= 3 for r in sub]), 3) if sub else None}
                                for p in pols}
  S["target_outcome_known"] = {"agent_graph": sum(r["target_dst_known_agent"] for r in rows),
                               "full_graph": sum(r["target_dst_known_full"] for r in rows), "n": len(rows),
                               "bridge_full": sum(r["target_bridge_full"] for r in rows)}
  gp = [r for r in rows if r["gap"]]
  S["gap_bridge_diagnostics"] = {"n": len(gp), "target_is_bridge_oracle": sum(r["target_bridge_oracle"] for r in gp),
                                 "mean_bridge_cands_oracle": float(np.mean([r["n_bridge_cands_oracle"] for r in gp])),
                                 "mean_cands": float(np.mean([r["n_cands"] for r in gp])),
                                 "mean_dst_known_oracle": float(np.mean([r["dst_known_oracle"] for r in gp]))}
  S["per_app_top1"] = {a: {p: round(np.mean([r[f"rank_{p}"] == 1 for r in rows if r["app"] == a]), 3) for p in pols}
                       | {"n": sum(r["app"] == a for r in rows)} for a in sorted({r["app"] for r in rows})}
  (OUT / "summary.json").write_text(json.dumps(S, indent=1))
  print(json.dumps(S, indent=1))


if __name__ == "__main__":
  main()
