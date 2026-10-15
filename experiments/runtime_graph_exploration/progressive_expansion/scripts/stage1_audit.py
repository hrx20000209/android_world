"""Stage 1: offline opportunity audit for progressive graph expansion.

Frozen initial graph G0 = leave-one-template-out snapshot of the 13,046
observations (memory.Snapshot). Every BASE model decision is a reasoning
frontier. No future agent decision is used as authorization in A-F; the
upper bound U uses future actions explicitly and is labelled ORACLE.

A  graph-first: a task-authorized (goal-anchor) safe continuation exists (DIRECT)
B  >= 1 unknown safe navigation slot (frontier) on the state
C  B and the state lies on a known partial path (<= 2 known edges) to a state
   exposing a goal anchor (task-grounded target, deployable)
D  2-hop frontier: some known successor of a safe slot has unknown safe slots
E  rejection reasons at B-states: no task target at all for this app / window
   fits 0 probes / all unknown slots ambiguous (grounding)
F  blind screen-order expansion: share of its first pick already known in G0
U  ORACLE upper bound of exploration-created bypasses: a later model click a at
   state s with (s, a) unknown in G0, safe, non-entity, and an earlier visit to
   the same state key in the same episode whose window fits >= 1 probe
   (U_oracle: the probe is a; U_fair: a is reached by screen-order probing of
   unknown slots across the earlier visits' windows). Within-episode only
   (held-out episodes may not leak into each other).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PE = HERE.parent
sys.path.insert(0, str(PE.parent / "scripts"))
import memory as M  # noqa: E402
import retrieval as RT  # noqa: E402

C, U, R = M.C, M.U, M.R
PROBE_S = 1.69


def nav_slots(ui, gn):
  out = []
  for sl in U.enumerate_slots(ui):
    if not sl.safe or sl.sel.get("in_list"):
      continue
    lab = sl.sel.get("label") or sl.sel.get("desc") or ""
    if RT.goal_named(lab, gn):
      continue                       # goal-named control = semantic choice, not a frontier
    out.append(sl)
  return out


def anchor_states(snap, gn):
  """State keys whose observed page exposes a goal anchor (structural labels only)."""
  out = set()
  for o in snap.obs:
    if o["dst"] and any(RT.goal_named(t, gn) for t in M.structural_labels(o)):
      out.add(o["dst"]["key"])
  return out


def backward(snap, targets, hops=2):
  rev = defaultdict(set)
  for o in snap.obs:
    if o["dst"] and o["op"] in ("click", "long_press"):
      rev[o["dst"]["key"]].add(o["src"]["key"])
  seen, frontier = set(targets), set(targets)
  for _ in range(hops):
    frontier = {p for t in frontier for p in rev.get(t, ())} - seen
    seen |= frontier
  return seen


def main():
  rows, per_task = [], defaultdict(Counter)
  snaps = {}
  U_oracle = U_fair = U_oracle_succ = U_fair_succ = 0
  u_examples = []
  for ep in sorted((M.TRV / "raw/full/BASE").glob("*/p*")):
    meta = json.loads((ep / "meta.json").read_text()); res = json.loads((ep / "result.json").read_text())
    task, goal = meta["task"], meta["goal"]; succ = res.get("is_successful") == 1.0
    tk = M.template_key(task)
    if tk not in snaps:
      snaps[tk] = M.Snapshot(task)
    snap = snaps[tk]; gn = M.norm(goal); ev = RT.Evidence(snap)
    targets = anchor_states(snap, gn)
    near = backward(snap, targets) if targets else set()
    visits = defaultdict(list)       # state key -> list of (fits, probed-set) windows seen so far
    probed_fair = defaultdict(set)   # state key -> slots a screen-order explorer would have probed
    steps = [json.loads(l) for l in open(ep / "steps.jsonl")]
    steps = [s for s in steps if s.get("kind") == "model"]
    for s in steps:
      ui = R.load_ui(ep, s["pre"]["cid"]); key = M.state_key(ui)
      tok, fam, sel = C.action_token(ui, s["action"])
      rows_c, info = RT.classify(ui, goal, ev)
      nav = nav_slots(ui, gn)
      unknown = [sl for sl in nav if not snap.by_edge.get((key, C.family_of_sel(sl.sel)))]
      fits = int((s.get("model_latency_s") or 0) // PROBE_S)
      flags = Counter()
      flags["A"] = info["direct"] is not None
      flags["B"] = bool(unknown)
      flags["C"] = bool(unknown) and key in near
      two_hop = False
      for sl in nav:
        for o in snap.by_edge.get((key, C.family_of_sel(sl.sel)), []):
          if o["dst"] and o["dst"]["key"] != key:
            two_hop = True
      flags["D"] = two_hop
      if unknown:
        if not targets:
          flags["E_no_target"] = 1
        if fits == 0:
          flags["E_budget0"] = 1
        fams = Counter(C.family_of_sel(sl.sel) for sl in unknown)
        if all(fams[C.family_of_sel(sl.sel)] > 1 for sl in unknown):
          flags["E_grounding"] = 1
      if nav:
        flags["F_first_known"] = bool(snap.by_edge.get((key, C.family_of_sel(nav[0].sel))))
        flags["F_any"] = 1
      # ORACLE upper bound of exploration-created bypass at this step
      if fam and (key, fam) and not snap.by_edge.get((key, fam)) and any(C.family_of_sel(sl.sel) == fam for sl in nav):
        earlier = visits.get(key, [])
        if any(f >= 1 for f in earlier):
          U_oracle += 1; U_oracle_succ += succ
          u_examples.append((succ, task, ep.name, s["step"], fam[:50], len(earlier), fam in probed_fair[key]))
        if fam in probed_fair[key]:
          U_fair += 1; U_fair_succ += succ
      # this window's probes (screen order over unknown, budget = fits)
      if unknown and fits:
        for sl in unknown:
          f2 = C.family_of_sel(sl.sel)
          if f2 not in probed_fair[key]:
            probed_fair[key].add(f2)
            fits_left = fits - 1
            break
      visits[key].append(fits)
      rows.append({"task": task, "pass": ep.name, "step": s["step"], "succ": succ, "n_unknown": len(unknown),
                   "fits": fits, **{k: int(v) for k, v in flags.items()}})
      for k, v in flags.items():
        per_task[task][k] += int(v)
      per_task[task]["states"] += 1
  out = PE / "metrics"
  (out / "stage1_states.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
  tot = Counter()
  for r in rows:
    for k, v in r.items():
      if k[:1] in "ABCDEF" and len(k) <= 15:
        tot[k] += v
  n = len(rows)
  summary = {"states": n, **{k: tot[k] for k in sorted(tot)},
             "U_oracle": U_oracle, "U_oracle_in_successful_eps": U_oracle_succ,
             "U_fair": U_fair, "U_fair_in_successful_eps": U_fair_succ}
  (out / "stage1_summary.json").write_text(json.dumps(summary, indent=1))
  print(json.dumps(summary, indent=1))
  apps = defaultdict(Counter)
  for t, c in per_task.items():
    a = C.APP_OF.get(t) or t[:10]
    for k, v in c.items():
      apps[a][k] += v
  (out / "stage1_per_task.json").write_text(json.dumps({t: dict(c) for t, c in per_task.items()}, indent=1))
  print("per app (states / B / C / U-relevant):")
  for a, c in sorted(apps.items(), key=lambda x: -x[1]["states"])[:20]:
    print(f"  {a[:22]:22s} {c['states']:4d} B={c['B']:4d} C={c['C']:3d} A={c['A']:2d} D={c['D']:4d}")
  (out / "stage1_U_cases.json").write_text(json.dumps(u_examples, indent=1))
  for e in u_examples:
    if e[0]:
      print("U in successful episode", e[1:])


if __name__ == "__main__":
  main()
