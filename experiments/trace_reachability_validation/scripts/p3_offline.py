"""Phase III offline analysis: page repetition, stage-dependence, knowledge
sources, and trace-prefill gate calibration (prior raw evidence only).

Q1 page repetition: how many distinct structural pages does an episode visit,
   how concentrated are steps on a few pages, and how many pages are shared
   across tasks of the same app?
Q2 stage: is the next action more predictable early in an episode (step
   index / normalized position), at one and two hops?
Q3 knowledge source: for each agent click, was the outcome of the chosen
   action already known before the decision, and from whom (prior agent runs
   of the same task, other tasks, or the explorer)?
Q4 prefill gate: with count thresholds on successful-trace continuations,
   how precise is "dominant historical action == actual action", and how often
   does the historical successor page differ from the real one (the case an
   explorer probe would catch before committing)?
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

OUT = C.ROOT / "analysis" / "phase3_offline"


def stage_bucket(i, n):
  if i == 0:
    return "s0"
  if i == 1:
    return "s1"
  if i <= 3:
    return "s2-3"
  return "s4+"


def main():
  OUT.mkdir(parents=True, exist_ok=True)
  eps = C.episodes()
  obs = C.graph_observations()
  seqs = {e["ep"]: PA.build_episode_seq(e) for e in eps}
  R = {}
  # ---------------- Q1 page repetition
  per_ep = []
  for e in eps:
    s = seqs[e["ep"]]
    if not s:
      continue
    states = [x["struct"] for x in s]
    c = Counter(states)
    per_ep.append({"task": e["task"], "steps": len(s), "distinct": len(c),
                   "revisit_steps": len(s) - len(c), "top1_share": c.most_common(1)[0][1] / len(s)})
  app_pages = defaultdict(lambda: defaultdict(set))
  for e in eps:
    for x in seqs[e["ep"]]:
      app_pages[e["app"]][x["struct"]].add(e["task"])
  shared = {a: {"pages": len(p), "pages_in_>=2_tasks": sum(len(t) >= 2 for t in p.values())}
            for a, p in app_pages.items()}
  # step-weighted: fraction of steps on pages seen in another task of the same app
  tot = shr = 0
  for e in eps:
    for x in seqs[e["ep"]]:
      tot += 1
      shr += len(app_pages[e["app"]][x["struct"]] - {e["task"]}) > 0
  R["q1"] = {"episodes": len(per_ep), "mean_steps": float(np.mean([p["steps"] for p in per_ep])),
             "mean_distinct_pages": float(np.mean([p["distinct"] for p in per_ep])),
             "revisit_step_fraction": sum(p["revisit_steps"] for p in per_ep) / sum(p["steps"] for p in per_ep),
             "mean_top1_page_share": float(np.mean([p["top1_share"] for p in per_ep])),
             "steps_on_pages_shared_with_other_tasks": shr / tot, "per_app_shared": shared,
             "per_task_distinct_pages_union": {t: len({x["struct"] for e in eps if e["task"] == t for x in seqs[e["ep"]]})
                                               for t in sorted({e["task"] for e in eps})}}
  # ---------------- Q2 stage dependence (reuse Pilot A decisions)
  rows = [json.loads(l) for l in (C.ROOT / "analysis/pilot_a/decisions_struct_sig.jsonl").read_text().splitlines()]
  by_ep = defaultdict(list)
  for r in rows:
    by_ep[(r["setting"], r["ep"])].append(r)
  seq_len = {e["ep"]: len(seqs[e["ep"]]) for e in eps}
  st = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))   # setting -> bucket -> [eligible, top1 P1, total clicks]
  two = defaultdict(lambda: defaultdict(lambda: [0, 0]))
  for (setting, ep), lst in by_ep.items():
    lst.sort(key=lambda r: r["step"])
    pos = {x["step"]: i for i, x in enumerate(seqs[ep])}
    for j, r in enumerate(lst):
      b = stage_bucket(pos.get(r["step"], 0), seq_len[ep])
      st[setting][b][2] += 1
      if r["eligible"]:
        st[setting][b][0] += 1
        st[setting][b][1] += int(r["rank_P1"] == 1)
      if j + 1 < len(lst) and lst[j + 1]["step"] == r["step"] + 1 and r["eligible"] and lst[j + 1]["eligible"]:
        two[setting][b][1] += 1
        two[setting][b][0] += int(r["rank_P1"] == 1 and lst[j + 1]["rank_P1"] == 1)
  R["q2"] = {s: {b: {"clicks": v[2], "eligible": v[0], "top1": v[1], "top1_rate": v[1] / v[0] if v[0] else None,
                     "eligible_rate": v[0] / v[2] if v[2] else None,
                     "twohop_top1": two[s][b][0], "twohop_pairs": two[s][b][1]}
                 for b, v in sorted(st[s].items())} for s in st}
  # ---------------- Q3 knowledge sources for each agent click
  q3 = defaultdict(Counter)
  for e in eps:
    for i, x in enumerate(seqs[e["ep"]]):
      if not x["fam"]:
        continue
      prior = [o for o in obs if o["t"] < x["t"] and o["src"]["struct_sig"] == x["struct"] and o.get("family") == x["fam"]]
      for setting in ("warm", "loto", "cold"):
        if setting == "loto":
          pr = [o for o in prior if o["task"] != e["task"] or o["ep"] == e["ep"]]
        elif setting == "cold":
          pr = [o for o in prior if o["ep"] == e["ep"]]
        else:
          pr = prior
        src = {"agent" if o["origin"] == "agent" else "explore" for o in pr}
        k = ("both" if src == {"agent", "explore"} else next(iter(src)) + "_only") if src else "unknown"
        q3[setting][k] += 1
        q3[setting]["n"] += 1
  R["q3"] = {s: dict(c) for s, c in q3.items()}
  # ---------------- Q4 prefill gate from SUCCESSFUL traces (chronological)
  gate = defaultdict(lambda: Counter())
  succ_seqs = []
  for e in eps:   # chronological
    s = seqs[e["ep"]]
    idx = PA.Index()
    for ss in succ_seqs:
      idx.add_seq(ss)
    succ_next = defaultdict(Counter)  # (state, fam) -> successor struct
    for ss in succ_seqs:
      for i in range(len(ss) - 1):
        if ss[i]["fam"]:
          succ_next[(ss[i]["struct"], ss[i]["fam"])][ss[i + 1]["struct"]] += 1
    for i, x in enumerate(s):
      if x["fam"] is None and not x["tok"].startswith(("TYPE", "SWIPE", "LONG")):
        continue
      counts = PA.overlay(idx, s, i)
      # longest order with support
      best = None
      for k in range(PA.K_MAX, -1, -1):
        if counts[k] and sum(counts[k].values()) > 0:
          best = counts[k]
          break
      if not best:
        gate["none"]["n"] += 1
        continue
      top, cnt = best.most_common(1)[0]
      tot = sum(best.values())
      for (thr_n, thr_d) in ((1, 0.0), (2, 0.6), (3, 0.8), (5, 0.9)):
        if cnt >= thr_n and cnt / tot >= thr_d:
          g = gate[f"n>={thr_n},dom>={thr_d}"]
          g["fires"] += 1
          g["correct"] += int(top == x["fam"])
          nxt = s[i + 1]["struct"] if i + 1 < len(s) else None
          exp = succ_next.get((x["struct"], top))
          if exp and nxt and top == x["fam"]:
            g["succ_known"] += 1
            g["succ_matches_real"] += int(exp.most_common(1)[0][0] == nxt)
      gate["all"]["n"] += 1
    if e["success"] == 1.0:
      succ_seqs.append(s)
  R["q4"] = {k: dict(v) for k, v in gate.items()}
  (OUT / "summary.json").write_text(json.dumps(R, indent=1))
  print(json.dumps(R, indent=1))


if __name__ == "__main__":
  main()
