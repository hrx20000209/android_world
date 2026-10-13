"""Pilot A: can historical agent traces predict which element the agent clicks next?

Offline over the previous study's raw traces. For every agent click decision we
hide the actual action, build the continuation index from ALLOWED history only,
rank the current screen's action-slot families with each policy, and record the
rank of the family the agent actually clicked.

History settings
  warm   : every episode that STARTED before the evaluated one (any task) + the
           evaluated episode's own earlier steps
  loto   : all episodes of OTHER task templates + own earlier steps
  loao   : all episodes of OTHER apps + own earlier steps
Graph knowledge (for P0u/P4): transitions observed before the decision time
(warm) or, for loto/loao, before that time AND outside the excluded set.

Policies (all lexicographic over counts; no weighted feature scores)
  P0   round-robin (screen order)
  P0u  previous explorer: graph-unknown first, then screen order
  P1   current-state frequency
  P2   longest matched suffix, backing off to shorter orders
  P3   variable order: only contexts with total support >= 2 are used
  P4   P3 + graph frontier: supported&unknown > supported&multi-outcome >
       supported&known > unsupported&unknown > rest
  ORA  oracle (headroom only)
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

U = C.U
K_MAX = 3
OUT = C.ROOT / "analysis" / "pilot_a"


def build_episode_seq(e, key="struct_sig"):
  seq = []
  for s in C.jl(Path(e["ep"]) / "steps.jsonl"):
    if s.get("label", "agent") != "agent":
      continue
    ui = C.load_ui(e["ep"], s["pre"]["cid"])
    tok, fam, sel = C.action_token(ui, s["action"])
    seq.append({"state": ui.summary()[key], "struct": ui.struct_sig, "tok": tok, "fam": fam, "sel": sel,
                "cid": s["pre"]["cid"], "step": s["step"], "t": s["pre"]["t"], "action": s["action"]})
  return seq


class Index:
  """counts[k][context] -> Counter(next family); context = (s_{i-k}, a_{i-k}, ..., s_i)."""

  def __init__(self):
    self.counts = [defaultdict(Counter) for _ in range(K_MAX + 1)]

  def add_seq(self, seq, upto=None):
    n = len(seq) if upto is None else upto
    for i in range(n):
      if seq[i]["fam"] is None:
        continue
      for k in range(K_MAX + 1):
        if i - k < 0:
          break
        ctx = context(seq, i, k)
        self.counts[k][ctx][seq[i]["fam"]] += 1

  def lookup(self, seq, i):
    out = []
    for k in range(K_MAX + 1):
      if i - k < 0:
        out.append(None)
        continue
      out.append(self.counts[k].get(context(seq, i, k)))
    return out


def context(seq, i, k):
  ctx = []
  for j in range(i - k, i):
    ctx += [seq[j]["state"], seq[j]["tok"]]
  ctx.append(seq[i]["state"])
  return tuple(ctx)


def overlay(base: Index, seq, i) -> list:
  """Counts from base history plus the evaluated episode's own prefix."""
  own = Index()
  own.add_seq(seq, upto=i)
  res = []
  b, o = base.lookup(seq, i), own.lookup(seq, i)
  for k in range(K_MAX + 1):
    c = Counter()
    if b[k]:
      c.update(b[k])
    if o[k]:
      c.update(o[k])
    res.append(c if (b[k] or o[k]) else None)
  return res


def rank_policies(cands, counts, known, multi):
  """cands: candidate families in screen order. Returns {policy: ordered list}."""
  pos = {f: i for i, f in enumerate(cands)}
  r = {"P0": list(cands)}
  r["P0u"] = sorted(cands, key=lambda f: (f in known, pos[f]))
  c0 = counts[0] or Counter()
  r["P1"] = sorted(cands, key=lambda f: (-c0[f], pos[f]))
  # P2: longest order with any candidate support, then shorter orders
  orders = [k for k in range(K_MAX, -1, -1) if counts[k] and any(counts[k][f] for f in cands)]
  r["P2"] = sorted(cands, key=lambda f: tuple(-counts[k][f] for k in orders) + (pos[f],))
  sup = [k for k in range(K_MAX, -1, -1) if counts[k] and sum(counts[k].values()) >= 2]
  r["P3"] = sorted(cands, key=lambda f: tuple(-counts[k][f] for k in sup) + (pos[f],))
  supported = {f for f in cands if any(counts[k] and counts[k][f] for k in range(K_MAX + 1))}
  p3pos = {f: i for i, f in enumerate(r["P3"])}

  def bucket(f):
    if f in supported:
      return 0 if f not in known else (1 if f in multi else 2)
    return 3 if f not in known else 4
  r["P4"] = sorted(cands, key=lambda f: (bucket(f), p3pos[f]))
  return r, orders[0] if orders else -1, (sup[0] if sup else -1)


def graph_knowledge(obs, t, struct, allowed):
  known, outcomes = set(), defaultdict(set)
  for o in obs:
    if o["t"] >= t:
      break
    if o["src"]["struct_sig"] != struct or not allowed(o):
      continue
    if o.get("family") and o.get("dst"):
      known.add(o["family"])
      outcomes[o["family"]].add(o["dst"]["layout_sig"])
  multi = {f for f, s in outcomes.items() if len(s) > 1}
  return known, multi


def main(key="struct_sig"):
  eps = C.episodes()
  obs = C.graph_observations()
  seqs = {e["ep"]: build_episode_seq(e, key) for e in eps}
  OUT.mkdir(parents=True, exist_ok=True)
  rows = []
  for setting in ("warm", "loto", "loao", "cold"):
    for e in eps:
      if setting == "warm":
        hist = [h for h in eps if h["started"] < e["started"]]
        allowed = lambda o: True  # noqa: E731
      elif setting == "loto":
        hist = [h for h in eps if h["task"] != e["task"]]
        allowed = lambda o, t=e["task"]: o["task"] != t  # noqa: E731
      elif setting == "loao":
        hist = [h for h in eps if h["app"] != e["app"]]
        allowed = lambda o, a=e["app"]: C.APP_OF.get(o["task"]) != a  # noqa: E731
      else:  # cold: only the evaluated episode's own past
        hist = []
        allowed = lambda o, ep=e["ep"]: o["ep"] == ep  # noqa: E731
      base = Index()
      for h in hist:
        base.add_seq(seqs[h["ep"]])
      seq = seqs[e["ep"]]
      for i, st in enumerate(seq):
        if st["fam"] is None:
          continue
        ui = C.load_ui(e["ep"], st["cid"])
        slots = U.enumerate_slots(ui)
        all_f, safe_f = [], []
        for s in slots:
          f = C.slot_family(ui, s)
          if f not in all_f:
            all_f.append(f)
          if s.safe and f not in safe_f:
            safe_f.append(f)
        target = st["fam"]
        counts = overlay(base, seq, i)
        known, multi = graph_knowledge(obs, st["t"], st["struct"], allowed)
        ranks, longest, longest_sup = rank_policies(safe_f, counts, known, multi)
        row = {"setting": setting, "key": key, "task": e["task"], "app": e["app"], "kind": e["kind"],
               "variant": e["variant"], "ep": e["ep"], "step": st["step"], "cid": st["cid"],
               "target": target, "eligible": target in safe_f, "in_screen": target in all_f,
               "n_cands": len(safe_f), "n_all": len(all_f), "longest_order": longest,
               "longest_supported_order": longest_sup, "target_known": target in known,
               "n_known": sum(f in known for f in safe_f), "hist_len": i}
        for p, lst in ranks.items():
          row["rank_" + p] = (lst.index(target) + 1) if target in lst else None
        row["rank_ORA"] = 1 if target in safe_f else None
        # next click in the same episode (for two-hop coverage)
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        row["next_is_click"] = bool(nxt and nxt["fam"])
        rows.append(row)
  (OUT / f"decisions_{key}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
  print(f"{key}: decisions={len(rows)//4} per setting; eps={len(eps)}")


if __name__ == "__main__":
  main(sys.argv[1] if len(sys.argv) > 1 else "struct_sig")
