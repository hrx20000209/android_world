"""Pilot B, step 1: decision states + reachability memory (M0-M4).

Labels: for a state visited in a FAILED episode, the action that a SUCCESSFUL
episode of the same task variant (same seed) took at a struct-equal state is a
proposed correct action; for states from successful episodes the own action is
proposed. Every label is then checked manually on the contact sheets.

Memory is built only from observations recorded strictly before the decision
time ("warm" history: earlier variants + earlier steps).

  M1 one-hop UTG        : observed slot -> destination screen (activity + title)
  M2 child affordances  : M1 + clickable labels observed on the child page
  M3 depth-2            : M2 + two-hop paths, VERIFIED (executed consecutively in
                          one run) vs COMPOSED (two separately observed edges)
  M4 full raw graph     : every observed edge of this app before t (capped)
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

U = C.U
OUT = C.ROOT / "decision_states"


def short_act(a):
  return (a or "?").rsplit(".", 1)[-1]


def title_of(ui: U.UIState) -> str:
  """Deterministic screen name: first non-list anchor text near the top."""
  best = None
  for n in ui.app_nodes():
    if n.in_list or n.editable or not n.visible or n.bounds[1] > 700:
      continue
    t = n.text or ""
    if t and not U._looks_dynamic(t) and len(t) <= 40:  # pylint: disable=protected-access
      if best is None or n.bounds[1] < best[0]:
        best = (n.bounds[1], t)
  return best[1] if best else ""


def affordances(ui: U.UIState, parent: U.UIState | None, cap=8):
  """Clickable labels on the child page (raw text/desc; rid if unlabeled),
  excluding controls that were already present on the parent."""
  par = set()
  if parent is not None:
    par = {(n.short_rid, U.label_of(parent, n)) for n in parent.app_nodes() if n.clickable}
  out, seen = [], set()
  for s in U.enumerate_slots(ui):
    n = ui.nodes[s.node_idx]
    lab = U.label_of(ui, n) or n.short_rid
    if not lab or (n.short_rid, U.label_of(ui, n)) in par or lab in seen:
      continue
    seen.add(lab)
    out.append(lab[:32] + ("" if not n.in_list else " (list item)"))
    if len(out) >= cap:
      break
  return out


def elem_name(sel):
  if not sel:
    return "?"
  return (sel.get("label") or sel.get("desc") or sel.get("rid") or sel.get("cls"))[:40]


def center(sel):
  x1, y1, x2, y2 = sel["bounds"]
  return (x1 + x2) // 2, (y1 + y2) // 2


def dst_ui(o):
  try:
    return C.load_ui(o["ep"], o["dst_cid"])
  except Exception:  # pylint: disable=broad-except
    return None


def build_memory(obs_before, struct, cur_ui, seqs_before):
  """Returns dict of M1..M4 evidence text + structured record."""
  here = [o for o in obs_before if o["src"]["struct_sig"] == struct and o.get("selector") and o.get("dst")
          and o.get("outcome") != "no_change" and o["dst"]["struct_sig"] != struct]
  latest = {}
  for o in here:
    latest[o["family"]] = o
  recs = []
  for fam, o in sorted(latest.items(), key=lambda kv: (kv[1]["selector"]["bounds"][1], kv[1]["selector"]["bounds"][0])):
    du = dst_ui(o)
    if du is None:
      continue
    recs.append({"family": fam, "name": elem_name(o["selector"]), "xy": center(o["selector"]),
                 "dst_struct": o["dst"]["struct_sig"], "dst_act": short_act(o["dst"]["activity"]),
                 "dst_title": title_of(du), "afford": affordances(du, cur_ui), "origin": o["origin"],
                 "instance": o["instance"], "obs_t": o["t"]})
  # second hops out of each child struct
  by_src = defaultdict(list)
  for o in obs_before:
    if o.get("selector") and o.get("dst") and o.get("outcome") != "no_change":
      by_src[o["src"]["struct_sig"]].append(o)
  # verified two-hop: consecutive agent steps in one earlier run starting at struct
  verified = []
  for seq in seqs_before:
    for i in range(len(seq) - 2):
      if seq[i]["struct"] == struct and seq[i]["fam"] and seq[i + 1]["fam"]:
        verified.append((seq[i]["fam"], seq[i + 1]["sel"], seq[i + 2]))
  for r in recs:
    r["paths"] = []
    seen = set()
    for f1, sel2, s3 in verified:
      if f1 == r["family"] and elem_name(sel2) not in seen:
        seen.add(elem_name(sel2))
        r["paths"].append({"kind": "verified", "second": elem_name(sel2),
                           "dest": short_act(s3["activity"]), "dest_title": s3.get("title", "")})
    for o in by_src.get(r["dst_struct"], []):
      nm = elem_name(o["selector"])
      if nm in seen or o["dst"]["struct_sig"] == r["dst_struct"]:
        continue
      seen.add(nm)
      du = dst_ui(o)
      r["paths"].append({"kind": "composed", "second": nm, "dest": short_act(o["dst"]["activity"]),
                         "dest_title": title_of(du) if du else ""})
    r["paths"] = r["paths"][:4]
  hdr = ("Previously observed interactions from this screen (recorded by an automatic, model-free UI explorer; "
         "they describe UI transitions only, not whether an action fits the task):")
  m1 = [f"- Tap \"{r['name']}\" at ({r['xy'][0]},{r['xy'][1]}) -> opens {r['dst_act']}"
        + (f" titled '{r['dst_title']}'" if r["dst_title"] else "") for r in recs]
  m2 = [l + (f"; that screen offers: {', '.join(r['afford'])}" if r["afford"] else "") for l, r in zip(m1, recs)]
  m3 = []
  for l, r in zip(m2, recs):
    m3.append(l)
    for p in r["paths"]:
      tag = "verified path" if p["kind"] == "verified" else "composed path (edges observed separately, not verified together)"
      m3.append(f"    {tag}: \"{r['name']}\" -> \"{p['second']}\" -> {p['dest']}"
                + (f" '{p['dest_title']}'" if p["dest_title"] else ""))
  # M4: whole-app raw graph edges
  app_edges, seen = [], set()
  for o in obs_before:
    if o["src"]["package"] != cur_ui.package or not o.get("selector") or not o.get("dst"):
      continue
    k = (o["src"]["struct_sig"], o["family"])
    if k in seen:
      continue
    seen.add(k)
    app_edges.append(f"- [{o['src']['struct_sig'][:6]}] tap \"{elem_name(o['selector'])}\" -> "
                     f"[{o['dst']['struct_sig'][:6]}] {short_act(o['dst']['activity'])}")
  m4 = [f"(current screen id: [{struct[:6]}])"] + app_edges[-60:]
  g4 = ("Observed UI transition graph of this app (automatic model-free explorer; screen ids in brackets; "
        "transitions only):")
  return {"M1": hdr + "\n" + "\n".join(m1) if m1 else "", "M2": hdr + "\n" + "\n".join(m2) if m2 else "",
          "M3": hdr + "\n" + "\n".join(m3) if m3 else "", "M4": g4 + "\n" + "\n".join(m4) if app_edges else "",
          "records": recs}


def main(n_target=44, seed=0):
  eps = C.episodes()
  obs = C.graph_observations()
  import pa_trace as PA
  seqs = {e["ep"]: PA.build_episode_seq(e) for e in eps}
  for e in eps:
    for st in seqs[e["ep"]]:
      st["activity"] = C.load_ui(e["ep"], st["cid"]).activity
      st["title"] = title_of(C.load_ui(e["ep"], st["cid"]))
  # successful episodes per (task, seed)
  succ = defaultdict(list)
  for e in eps:
    if e["success"] == 1.0:
      succ[(e["task"], e["seed"])].append(e)
  cands = []
  for e in eps:
    seq = seqs[e["ep"]]
    for i, st in enumerate(seq):
      if st["tok"] in ("STATUS", "ANSWER") or st["tok"].startswith("OPEN_APP"):
        continue
      before = [o for o in obs if o["t"] < st["t"]]
      here = [o for o in before if o["src"]["struct_sig"] == st["struct"] and o["origin"] == "explore"
              and o.get("outcome") != "no_change"]
      if not here:
        continue
      # proposed label
      label, src = None, None
      if e["success"] == 1.0:
        label, src = [st["tok"]], "own_success"
      else:
        for se in succ[(e["task"], e["seed"])]:
          for s2 in seqs[se["ep"]]:
            if s2["struct"] == st["struct"]:
              label = (label or []) + [s2["tok"]]
              src = "paired_success"
      if not label:
        continue
      err = st["tok"] not in label
      cands.append({"ep": e["ep"], "task": e["task"], "app": e["app"], "variant": e["variant"],
                    "kind": e["kind"], "goal": e["goal"], "step": st["step"], "cid": st["cid"], "t": st["t"],
                    "struct": st["struct"], "orig_tok": st["tok"], "orig_action": st["action"],
                    "label_tokens": sorted(set(label)), "label_src": src, "orig_error": err,
                    "episode_success": e["success"], "history": [s["summary"] for s in C.jl(Path(e["ep"]) / "steps.jsonl") if s["step"] < st["step"]]})
  rng = random.Random(seed)
  errs = [c for c in cands if c["orig_error"]]
  oks = [c for c in cands if not c["orig_error"]]
  rng.shuffle(errs)
  rng.shuffle(oks)
  # at most 3 per task per group for diversity
  def cap(lst, k):
    out, per = [], defaultdict(int)
    for c in lst:
      if per[c["task"]] < k:
        per[c["task"]] += 1
        out.append(c)
    return out
  picked = cap(errs, 8)
  picked += cap(oks, 3)[: max(0, n_target - len(picked))]
  for c in picked:
    before = [o for o in obs if o["t"] < c["t"]]
    seqs_before = [seqs[e["ep"]] for e in eps if e["started"] < c["t"] and e["ep"] != c["ep"]]
    ui = C.load_ui(c["ep"], c["cid"])
    c["memory"] = build_memory(before, c["struct"], ui, seqs_before)
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "pilot_b_states.json").write_text(json.dumps(picked, indent=1, ensure_ascii=False, default=str))
  print(f"candidates={len(cands)} errors={len(errs)} ok={len(oks)} picked={len(picked)} "
        f"(err {sum(c['orig_error'] for c in picked)}) tasks={len({c['task'] for c in picked})} apps={len({c['app'] for c in picked})}")


if __name__ == "__main__":
  main()
