"""Offline: entity selection steps and where the entity is.
At each model step: goal parameter values V (FuncExec records). Rows on the current screen whose text equals a value
(normalized) are entity candidates. Measures
  (1) on-screen: a unique row matches a not-yet-handled value -> how often the model's action targets it (and how);
  (2) off-screen: the value's row is not on screen but appeared on a page observed earlier in this episode
      (agent visit or exploration probe) -> distance in observations, and whether the model went there next."""
import glob, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9, explore_value as EV
R, U, C, FG = full9.R, full9.U, full9.C, full9.FG


class Ep:
  def __init__(s, t): s.args = type("a", (), {"task": t})()


def rows_with_values(ui, vals, handled=()):
  """Selectable entity rows: a non-editable text equal to an unhandled value, inside a clickable list row/card."""
  out = []
  for n in ui.app_nodes():
    if not n.visible or n.editable or not (n.text or n.desc):
      continue
    v = FG.norm(n.text or n.desc)
    if v not in vals or len(v) < 3 or v in handled:
      continue
    cur, ok = n, False
    while cur is not None:
      if cur.clickable or cur.long_clickable:
        ok = cur.in_list or cur.long_clickable or cur.clickable
        break
      cur = ui.nodes[cur.parent] if cur.parent is not None else None
    if ok and n.in_list:
      out.append((v, n))
  return out


def main(arm):
  c = Counter(); how = Counter()
  for d in sorted(glob.glob(f"raw/full/{arm}/*/p0")):
    d = Path(d); t = d.parts[-2]
    goal = json.load(open(d / "meta.json"))["goal"]
    fx = full9.FuncExec(Ep(t), goal)
    vals = {FG.norm(v) for vs in fx.recs.values() for v in vs if v and not v.isdigit()}
    if not vals:
      continue
    seen_vals = {}                              # value -> step where it was visible on an observed page
    handled = set()
    probe_caps = {}
    if (d / "probes.jsonl").exists():
      for l in open(d / "probes.jsonl"):
        o = json.loads(l)
        if o.get("dst_cid"):
          probe_caps.setdefault(o["step"], []).append(o["dst_cid"])
    for l in open(d / "steps.jsonl"):
      r = json.loads(l)
      if r.get("kind") != "model" or not r.get("pre"):
        continue
      try:
        ui = R.load_ui(d, r["pre"]["cid"])
      except Exception:
        continue
      c["steps"] += 1
      on = rows_with_values(ui, vals, handled)
      a = r.get("action") or {}
      tgt = U.element_at(ui, int(a["x"]), int(a["y"])) if a.get("x") is not None else None
      tgt_txt = set()
      if tgt is not None:
        tgt_txt = {FG.norm(x.text or x.desc) for x in ui.app_nodes() if (x.text or x.desc) and
                   tgt.bounds[0] <= x.center[0] <= tgt.bounds[2] and tgt.bounds[1] <= x.center[1] <= tgt.bounds[3]}
      if on:
        c["value_row_on_screen"] += 1
        hit = any(v in tgt_txt for v, _ in on)
        c["model_targets_value_row"] += hit
        if hit:
          how[a.get("action_type")] += 1
      else:
        for v in vals:
          if v in seen_vals:
            c["value_seen_earlier_offscreen"] += 1
            break
      if a.get("action_type") == "input_text":
        handled.add(FG.norm(a.get("text") or ""))
      handled |= {v for v in vals if v in tgt_txt}
      for v, _ in on:
        seen_vals.setdefault(v, r["step"])
      for cid in probe_caps.get(r["step"], []):
        try:
          pu = R.load_ui(d, cid)
        except Exception:
          continue
        for v, _ in rows_with_values(pu, vals):
          if v not in seen_vals:
            seen_vals[v] = r["step"]; c["value_first_seen_by_probe"] += 1
  print(f"[{arm}] {dict(c)}  model action on matching row by type: {dict(how)}")


if __name__ == "__main__":
  for a in sys.argv[1:]:
    main(a)


def eval_action_type(arm, margins=(0.0, 0.1, 0.2)):
  """At unhandled value rows on screen: choose click vs long-press on the row's list role with the node x term matrix
  (cross-task, causal); compare with the model's action (same row and same action type)."""
  import term_graph as TG
  raw = Path("raw/full") / arm
  lines = sorted((json.loads(l) for l in open(f"graph_snapshots/full/{arm}/graph.jsonl")), key=lambda o: o.get("t", 0))
  ops = []
  for d in sorted(glob.glob(f"raw/full/{arm}/*/p0")):
    d = Path(d); t = d.parts[-2]
    goal = json.load(open(d / "meta.json"))["goal"]
    fx = full9.FuncExec(Ep(t), goal)
    vals = {FG.norm(v) for vs in fx.recs.values() for v in vs if v and not v.isdigit()}
    if not vals:
      continue
    g = TG._unit(EV.goal_vector(goal, fx))
    handled = set()
    for l in open(d / "steps.jsonl"):
      r = json.loads(l)
      if r.get("kind") != "model" or not r.get("pre"):
        continue
      try:
        ui = R.load_ui(d, r["pre"]["cid"])
      except Exception:
        continue
      a = r.get("action") or {}
      tgt = U.element_at(ui, int(a["x"]), int(a["y"])) if a.get("x") is not None else None
      tgt_txt = set()
      if tgt is not None:
        tgt_txt = {FG.norm(x.text or x.desc) for x in ui.app_nodes() if (x.text or x.desc) and
                   tgt.bounds[0] <= x.center[0] <= tgt.bounds[2] and tgt.bounds[1] <= x.center[1] <= tgt.bounds[3]}
      on = rows_with_values(ui, vals, handled)
      if on:
        v, node = on[0]
        cur = node
        while cur is not None and not (cur.clickable or cur.long_clickable):
          cur = ui.nodes[cur.parent] if cur.parent is not None else None
        role = C.family_of_sel(U.selector(ui, cur or node))
        model_kind = a.get("action_type") if v in tgt_txt else "other"
        ops.append((r["pre"]["t"], t, ui, g, role, model_kind))
      if a.get("action_type") == "input_text":
        handled.add(FG.norm(a.get("text") or ""))
      handled |= {v for v in vals if v in tgt_txt}
  ops.sort(key=lambda x: x[0])
  res = {m: Counter() for m in margins}
  tms = {}
  i = 0
  for ts, task, ui, g, role, model_kind in ops:
    tm = tms.setdefault(task, TG.TermMatrix(Path("/nonexistent"), raw, task))
    # causal ingestion per task matrix (own-task goal excluded inside TermMatrix)
    for o in lines:
      if o.get("t", 0) >= ts:
        break
      if id(o) in getattr(tm, "_seen", set()):
        continue
      tm.ingest(o)
      tm._seen = getattr(tm, "_seen", set()) | {id(o)}
    P = (ui.package, ui.activity, ui.layout_sig)
    opts = {}
    for kind, tok in (("click", role), ("long_press", "LONG_PRESS:" + role)):
      ds = tm.out.get(P, {}).get(tok)
      if ds and (tm.edge_tasks[(P, tok)] - {task}):
        opts[kind] = max(tm.backed(n, g) for n in ds)
    for m in margins:
      c = res[m]
      c["opportunities"] += 1
      if not opts:
        c["no_evidence"] += 1
        continue
      sc = sorted(((s, k) for k, s in opts.items()), reverse=True)
      top_s, top_k = sc[0]
      second = sc[1][0] if len(sc) > 1 else 0.0
      if top_s - second < m:
        c["abstain"] += 1
        continue
      c["act"] += 1
      c["correct"] += top_k == model_kind
  for m in margins:
    c = res[m]
    print(f"[{arm}] margin>={m}: {dict(c)} precision {c['correct'] / max(1, c['act']):.2f}")
