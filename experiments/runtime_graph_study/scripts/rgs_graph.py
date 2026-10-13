"""Study B: rebuild the observation graph from immutable raw observations
under different state-matching rules, and measure reuse / convergence /
runtime-outcome variation across variants.

Matching rules (node key):
  exact   exact_sig  (any a11y change -> new node)
  struct  struct_sig (online rule; list content and numbers ignored)
  layout  layout_sig (drops anchor texts; deliberately coarse, to expose false merges)
Memory modes:
  shared   one graph accumulated across variants in run order
  per_run  graph reset for every variant (no cross-run reuse)

Writes analysis/graph_metrics.json, analysis/graph_growth.csv,
analysis/revalidations.csv, analysis/convergence_events.csv,
graphs/<set>/<task>.<rule>.graph.json (serialized snapshot).
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = {"exact": "exact_sig", "struct": "struct_sig", "layout": "layout_sig"}


def jl(p: Path):
  return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def write_csv(path: Path, rows):
  if not rows:
    path.write_text("")
    return
  keys = []
  for r in rows:
    for k in r:
      if k not in keys:
        keys.append(k)
  with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)


def fix_agent_slots(obs, raw_task_dir, task):
  """Recompute agent-action slot ids from raw a11y (z-order aware hit test)."""
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  import rgs_analyze as A
  import rgs_run as R
  for o in obs:
    if o.get("origin") != "agent" or not o.get("action"):
      continue
    var = o["instance"].split("/")[1]
    ep = raw_task_dir / (var if var.startswith("v0_ref") else f"{var}_agent")
    try:
      ui = A.load_ui(str(ep), o["src_cid"])
      sid, sel = R.action_slot(ui, o["action"], {})
      o["slot_id"], o["selector"] = sid, sel
    except Exception:  # pylint: disable=broad-except
      pass


def resummarize(obs, raw_task_dir):
  """Recompute state summaries from raw a11y with the current definitions."""
  import rgs_analyze as A
  for o in obs:
    var = o["instance"].split("/")[1]
    ep = raw_task_dir / (var if var.startswith("v0_ref") else f"{var}_agent")
    for side in ("src", "dst"):
      cid = o.get(side + "_cid")
      if not cid or not o.get(side):
        continue
      try:
        o[side] = A.load_ui(str(ep), cid).summary()
      except Exception:  # pylint: disable=broad-except
        pass


def instance_order(obs):
  order = []
  for o in obs:
    if o["instance"] not in order:
      order.append(o["instance"])
  return order


def build(obs, rule: str, shared: bool, task: str):
  key = RULES[rule]
  nodes: dict[str, dict] = {}
  edges: dict[tuple, dict] = {}     # (src, slot, dst) -> info
  slot_obs: dict[tuple, list] = defaultdict(list)  # (src, slot) -> observations
  growth, reval, conv = [], [], []
  per_inst = {}
  cur_inst = None
  t_index = 0
  for o in obs:
    inst = o["instance"]
    if inst != cur_inst:
      if not shared:
        nodes, edges, slot_obs = {}, {}, defaultdict(list)
      cur_inst = inst
      per_inst[inst] = defaultdict(int)
      prior_nodes = set(nodes)
      per_inst[inst]["_prior_nodes"] = prior_nodes
    st = per_inst[inst]
    s = o["src"][key]
    d = o["dst"][key] if o.get("dst") else None
    is_probe = o["origin"] == "explore"
    # node reuse for states the agent actually visited
    if o["origin"] == "agent":
      st["agent_states"] += 1
      if s in st["_prior_nodes"]:
        st["agent_states_known_from_prior_runs"] += 1
      if s in nodes:
        st["agent_states_known_any"] += 1
    for n in (s, d):
      if n is None:
        continue
      if n not in nodes:
        nodes[n] = {"first_instance": inst, "first_origin": o["origin"], "activity": o["src"]["activity"] if n == s else o["dst"]["activity"],
                    "instances": set(), "layout": o["src"]["layout_sig"] if n == s else o["dst"]["layout_sig"]}
        st["new_nodes"] += 1
      nodes[n]["instances"].add(inst)
    sk = (s, o["slot_id"])
    prev = slot_obs[sk]
    prev_other = [p for p in prev if p["instance"] != inst]
    if prev_other and d is not None:
      prev_dsts = {p["dst"] for p in prev_other}
      same = d in prev_dsts
      reval.append({"task": task, "rule": rule, "shared": shared, "instance": inst, "origin": o["origin"],
                    "slot_id": o["slot_id"], "same_outcome": same, "n_prior": len(prev_other),
                    "prior_dsts": len(prev_dsts), "outcome": o.get("outcome"),
                    "dst_cid": o.get("dst_cid"), "src_cid": o.get("src_cid"),
                    "prior_dst_cid": prev_other[-1].get("dst_cid"), "prior_ep_instance": prev_other[-1]["instance"]})
      st["revalidated"] += 1
      st["revalidated_same"] += int(same)
    # convergence: dst already a node, reached via a different (src, slot) than before
    if d is not None and d != s:
      e = (s, o["slot_id"], d)
      existed = d in nodes and (nodes[d]["first_instance"] != inst or any(
          k[2] == d for k in edges))
      incoming_before = [k for k in edges if k[2] == d]
      if e not in edges:
        st["new_edges"] += 1
        if incoming_before:
          known_out = sum(1 for k in slot_obs if k[0] == d)
          st["convergence"] += 1
          st["convergence_avoided_slots"] += known_out
          conv.append({"task": task, "rule": rule, "instance": inst, "origin": o["origin"], "src": s[:8],
                       "slot_id": o["slot_id"], "dst": d[:8], "dst_known_out_slots": known_out,
                       "incoming_before": len(incoming_before), "src_cid": o.get("src_cid"),
                       "dst_cid": o.get("dst_cid")})
        edges[e] = {"first_instance": inst, "origin": o["origin"], "n": 0}
      edges[e]["n"] += 1
    slot_obs[sk].append({"instance": inst, "dst": d, "dst_cid": o.get("dst_cid")})
    if is_probe:
      st["probes"] += 1
      st["probe_rollback_fail"] += int(not o.get("rollback_ok", True))
    t_index += 1
    growth.append({"task": task, "rule": rule, "shared": shared, "i": t_index, "instance": inst,
                   "origin": o["origin"], "nodes": len(nodes), "edges": len(edges),
                   "observed_slots": len(slot_obs)})
  for inst, st in per_inst.items():
    st.pop("_prior_nodes", None)
  multi = sum(1 for k, v in slot_obs.items() if len({p["dst"] for p in v if p["dst"]}) > 1)
  snap = {"nodes": {k: {**v, "instances": sorted(v["instances"])} for k, v in nodes.items()},
          "edges": [{"src": k[0], "slot": k[1], "dst": k[2], **v} for k, v in edges.items()],
          "slots_with_multiple_outcomes": multi, "n_slots": len(slot_obs)}
  return {"per_instance": {k: dict(v) for k, v in per_inst.items()}, "n_nodes": len(nodes),
          "n_edges": len(edges), "n_slots": len(slot_obs), "multi_outcome_slots": multi}, growth, reval, conv, snap


def ambiguity(obs):
  """Under the online struct rule: states whose layout matches a known node
  but whose struct key differs (kept separate, flagged ambiguous)."""
  seen_struct, layout_to_struct = set(), defaultdict(set)
  amb, total = 0, 0
  rows = []
  for o in obs:
    for side in ("src", "dst"):
      if not o.get(side):
        continue
      s = o[side]
      if s["struct_sig"] in seen_struct:
        continue
      total += 1
      if layout_to_struct[s["layout_sig"]]:
        amb += 1
        rows.append({"instance": o["instance"], "struct": s["struct_sig"][:8], "layout": s["layout_sig"][:8],
                     "cid": o.get(side + "_cid"), "activity": s["activity"]})
      seen_struct.add(s["struct_sig"])
      layout_to_struct[s["layout_sig"]].add(s["struct_sig"])
  return amb, total, rows


def main():
  gset = sys.argv[1] if len(sys.argv) > 1 else "graphs/main"
  out = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "analysis")
  out.mkdir(parents=True, exist_ok=True)
  metrics, growth_all, reval_all, conv_all, amb_rows = {}, [], [], [], []
  for gx in sorted((ROOT / gset).glob("*.explore.jsonl")):
    task = gx.name.split(".")[0]
    obs = jl(gx)
    obs.sort(key=lambda o: o["t"])
    fix_agent_slots(obs, gx.parents[2] / "raw" / gx.parent.name / task, task)
    resummarize(obs, gx.parents[2] / "raw" / gx.parent.name / task)
    m = {"n_obs": len(obs), "instances": instance_order(obs),
         "n_probe_obs": sum(o["origin"] == "explore" for o in obs),
         "n_agent_obs": sum(o["origin"] == "agent" for o in obs)}
    for rule in RULES:
      for shared in (True, False):
        r, g, rv, cv, snap = build(obs, rule, shared, task)
        m[f"{rule}_{'shared' if shared else 'per_run'}"] = r
        growth_all += g
        if shared:
          reval_all += rv
          conv_all += cv
          (ROOT / gset / f"{task}.{rule}.graph.json").write_text(json.dumps(snap, ensure_ascii=False))
    a, tot, rows = ambiguity(obs)
    m["struct_ambiguous_new_states"] = a
    m["struct_distinct_states"] = tot
    amb_rows += [{"task": task, **r} for r in rows]
    metrics[task] = m
  (out / "graph_metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
  write_csv(out / "graph_growth.csv", growth_all)
  write_csv(out / "revalidations.csv", reval_all)
  write_csv(out / "convergence_events.csv", conv_all)
  write_csv(out / "ambiguous_states.csv", amb_rows)
  print(f"tasks={len(metrics)} growth_rows={len(growth_all)} reval={len(reval_all)} conv={len(conv_all)} amb={len(amb_rows)}")


if __name__ == "__main__":
  main()
