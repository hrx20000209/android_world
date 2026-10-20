"""FULL17: one graph for nodes reached by inference (agent) and nodes reached by exploration.

A page is (package, activity, layout signature) whatever discovered it; an edge is (page, control family) ->
page with the set of origins that observed it. The joint graph is used for:
  * restoration  - a probe Back could not undo: walk a safe path from where we are back to the origin page;
  * routing      - a function's entry page is a few edges away: walk there, then run the function (1 step);
and it is measured by the origin x outcome matrix (scripts/joint_matrix.py).
Every hop relocates its control by family on the live screen and verifies the landing activity.
"""
from __future__ import annotations

from collections import deque

import full9

R, U, C = full9.R, full9.U, full9.C
MAX_HOPS = 3


def page_key(ui_or_summary):
  if isinstance(ui_or_summary, dict):
    return (ui_or_summary.get("package"), ui_or_summary.get("activity"), ui_or_summary.get("layout_sig"))
  return (ui_or_summary.package, ui_or_summary.activity, ui_or_summary.layout_sig)


class JointGraph:

  def __init__(self, graph):
    self.out = {}          # page -> {family: {"dst": {page: count}, "origins": set, "sel": selector, "ents": [...]}}
    self.into = {}         # page -> [(instance, capture id, page)] captures of that page
    self.origin = {}       # page -> set of origins that reached it
    for (struct, sid), obs in graph.slots.items():
      for o in obs:
        d = o.get("dst")
        if not d or not o.get("src"):
          continue
        sp, dp = page_key(o["src"]), page_key(d)
        org = "explore" if o.get("origin") == "explore" else "agent"
        # a depth-1 probe starts on a page the agent stood on; a depth-2 probe starts on a page exploration reached
        self.origin.setdefault(sp, set()).add("agent" if o.get("depth", 1) == 1 or org == "agent" else "explore")
        self.origin.setdefault(dp, set()).add(org)
        if sp == dp:
          continue
        sel = o.get("selector") or {}
        fam = sid if sid.startswith("E|") else (C.family_of_sel(sel) if sel else "E|" + sid)
        e = self.out.setdefault(sp, {}).setdefault(fam, {"dst": {}, "origins": set(), "sel": sel, "ents": []})
        e["dst"][dp] = e["dst"].get(dp, 0) + 1
        e["origins"].add(org)
        if o.get("dst_cid"):
          e["ents"].append((o["instance"], o["dst_cid"], dp))
          self.into.setdefault(dp, []).append((o["instance"], o["dst_cid"], dp))

  def edge(self, page, fam):
    return self.out.get(page, {}).get(fam)

  @staticmethod
  def predicted(e):
    """Most frequently observed destination of an edge."""
    return max(e["dst"].items(), key=lambda kv: kv[1])[0] if e and e["dst"] else None

  def starts(self, ui):
    """Graph pages that the live screen is: same layout, else same activity (layouts drift with content)."""
    k = page_key(ui)
    if k in self.out:
      return [k]
    return [p for p in self.out if p[0] == ui.package and p[1] == ui.activity]

  def paths(self, ui, goal, max_hops=MAX_HOPS, safe=None, starts=None):
    """Breadth-first paths (lists of (family, selector, dst page, origins)) from the live page to pages
    satisfying goal(page) -> bool. Only edges whose control passes safe(selector)."""
    out = []
    for s in (starts if starts is not None else self.starts(ui)):
      q, seen = deque([(s, [])]), {s}
      while q:
        p, path = q.popleft()
        if path and goal(p):
          out.append(path)
          continue
        if len(path) >= max_hops:
          continue
        for fam, e in self.out.get(p, {}).items():
          if safe is not None and not safe(e["sel"]):
            continue
          d = self.predicted(e)
          if d is None or d in seen:
            continue
          seen.add(d)
          q.append((d, path + [(fam, e["sel"], d, frozenset(e["origins"]))]))
    out.sort(key=len)
    return out


def walk(ep, path, step, tag):
  """Execute a path hop by hop: relocate each control by family on the live screen, tap, verify the landing
  activity. Returns (n_hops_done, final capture, ok)."""
  import full7  # pylint: disable=import-outside-toplevel
  cap = ep.dev.capture(f"s{step}_{tag}0", settle=True)
  for i, (fam, sel, dst, _) in enumerate(path):
    live = cap["ui"]
    sl = next((t for t in U.enumerate_slots(live) if C.family_of_sel(t.sel) == fam and t.safe), None)
    if sl is None:
      return i, cap, False
    ep.dev.tap(*live.nodes[sl.node_idx].center)
    cap = full7.capture_stable(ep.dev, f"s{step}_{tag}{i + 1}")
    if cap["ui"].activity != dst[1]:
      return i + 1, cap, False
  return len(path), cap, True


def safe_control(sel):
  return not full9.effectful(sel) and not U.UNSAFE_LABEL_RE.search(
      " ".join([sel.get("label") or "", sel.get("desc") or "", (sel.get("rid") or "").replace("_", " ")]))


def restore(ep, graph, target_ui, step):
  """Joint-graph restoration after a failed Back: a safe path (agent or exploration edges) from the current page
  to the origin page; success only if the origin state is reproduced exactly."""
  jg = JointGraph(graph)
  cur = ep.dev.capture(f"s{step}_jrs", settle=True)["ui"]
  if R.same_state(cur, target_ui):
    return True, []
  tk = page_key(target_ui)
  ps = jg.paths(cur, lambda p: p == tk, safe=safe_control)
  if not ps:
    return False, None
  path = ps[0]
  n, cap, ok = walk(ep, path, step, "jr")
  ok = ok and R.same_state(cap["ui"], target_ui)
  return ok, [sorted(h[3]) for h in path]


def try_route(fx, ep, graph, pre, agent, steps, k, own, m):
  """Function routing over the joint graph: when no function grounds on this page but one grounds on a page the
  graph holds (its capture grounds the function's first step with this record's values) within MAX_HOPS safe
  edges, walk there and run the function. The whole thing is one counted step."""
  import json_action_shim as JA  # pylint: disable=import-outside-toplevel
  import explore_value as EV  # pylint: disable=import-outside-toplevel
  rec = fx.next_rec
  fx.route_tried = getattr(fx, "route_tried", set())
  if (rec >= fx._n_recs() or rec in fx.route_tried or fx.applied_rec is not None or fx.model_rec is not None):
    return 0
  ui = pre["ui"]
  multi = {fd for fd, vs in fx.recs.items() if len(vs) > 1}
  funcs = [f for f in fx.lib.funcs
           if f["verb"] is not None and f["verb"] == fx.P["verb"] and len(f["steps"]) >= 2 and f["fields"]
           and not (multi and not multi <= set(f["fields"])) and all(fx._value(fd, rec) is not None for fd in f["fields"])
           # a bare number extracted from a name or date is not a parameter of the task
           and set(f["fields"]) - {"number"}
           # the function must be about this goal's operation/object (TF-IDF relevance of its own text)
           and EV.rel(getattr(ep, "_gvec", {}), EV.tokens(" ".join([f.get("template", ""), f.get("commit") or ""] +
                                                            [(st.get("fam") or "").split("|")[-1] for st in f["steps"]]))) > 0]
  if not funcs:
    return 0
  if any(f["entry_act"] == ui.activity and fx._ground_first(f, ui, rec) for f in funcs):
    return 0
  fx.route_tried.add(rec)
  jg = JointGraph(graph)
  raw = ep.out.parents[1]
  acts = {f["entry_act"] for f in funcs}
  memo = {}

  def grounds(p):
    if p[1] not in acts or p[0] != ui.package:
      return False
    if p not in memo:
      memo[p] = False
      for inst, cid, _ in jg.into.get(p, [])[-4:]:
        dui = EV.load_cap(raw, inst, cid)
        if dui is not None and any(f["entry_act"] == dui.activity and fx._ground_first(f, dui, rec) for f in funcs):
          memo[p] = True
          break
    return memo[p]

  if getattr(fx, "route_soft", False):
    # FULL39: same as form routing - every recorded layout of this activity is a start, creation entries may be
    # crossed, and the first control must be on the live screen
    live_fams = {C.family_of_sel(t.sel) for t in U.enumerate_slots(ui)}
    starts = [p for p in jg.out if p[0] == ui.package and p[1] == ui.activity]
    paths = [pa for pa in jg.paths(ui, grounds, safe=lambda sel: safe_control(sel) or entry_control(sel), starts=starts)
             if pa and pa[0][0] in live_fams]
  else:
    paths = jg.paths(ui, grounds, safe=safe_control)
  m["route_lookups"] += 1
  if not paths:
    return 0
  path = paths[0]
  import full7  # pylint: disable=import-outside-toplevel
  cap, hops = pre, 0
  for fam, sel, dst, origins in path:
    live = cap["ui"]
    sl = next((t for t in U.enumerate_slots(live) if C.family_of_sel(t.sel) == fam
               and (t.safe or (getattr(fx, "route_soft", False) and entry_control(t.sel)))), None)
    if sl is None:
      break
    node = live.nodes[sl.node_idx]
    action = JA.click(*node.center)
    agent._execute_action(action, {})  # pylint: disable=protected-access
    if getattr(fx, "exec_log", None) is not None:
      fx.exec_log.append({"action": {kk: vv for kk, vv in action.__dict__.items() if vv is not None}, "commit": False})
    lab = U.label_of(live, node) or sel.get("desc") or sel.get("rid")
    summ = f"[graph] Tapped \"{lab}\"."
    agent._actions.append({"summary": summ})  # pylint: disable=protected-access
    agent._summaries.append(summ)  # pylint: disable=protected-access
    row = {"step": k, "kind": "function_route", "tok": fam, "summary": summ, "origins": sorted(origins),
           "hop": hops, "pre": R.strip(cap)}
    steps.append(row)
    ep.log("steps", row)
    hops += 1
    m["route_hops"] += 1
    m["route_hops_explore_only"] += origins == frozenset({"explore"})
    cap = full7.capture_stable(ep.dev, f"s{k}_route{hops}")
    if cap["ui"].activity != dst[1]:
      break
  if not hops:
    return 0
  m["route_apps"] += 1
  n = fx.try_apply(cap, agent, steps, k, own)
  m["route_then_applied"] += bool(n)
  return hops + n


ENTRY_WORDS = {"new", "add", "create", "compose"}


def entry_control(sel):
  """A creation entry (New Event / New Recipe / "+"): opens an editor, changes no data until the editor's commit."""
  if sel.get("in_list") or sel.get("cls") in ("Switch", "CheckBox", "ToggleButton", "RadioButton"):
    return False
  import re  # pylint: disable=import-outside-toplevel
  lab = (sel.get("label") or sel.get("desc") or sel.get("text") or "").strip().lower()
  first = re.split(r"[^a-z]+", lab)[0] if lab else ""
  rid = set(re.split(r"[^a-z]+", (sel.get("rid") or "").lower()))
  return first in ENTRY_WORDS or (not lab and bool(rid & (ENTRY_WORDS | {"fab"})))


def try_form_route(fx, ep, graph, pre, agent, steps, k, own, m, min_fields=2):
  """FULL38: form routing. When this record's fields are not on the live page but a page of the graph (from any
  task's inference trajectory or from exploration) shows >= min_fields empty fields that this record fills, walk
  there over <= MAX_HOPS safe or creation-entry edges and fill them (pattern edge). One counted step, no inference."""
  import json_action_shim as JA  # pylint: disable=import-outside-toplevel
  import explore_value as EV  # pylint: disable=import-outside-toplevel
  import full7  # pylint: disable=import-outside-toplevel
  rec = fx.next_rec
  fx.form_route_tried = getattr(fx, "form_route_tried", set())
  if (rec >= fx._n_recs() or rec in fx.form_route_tried or fx.applied_rec not in (None, rec)
      or (fx.model_rec is not None and fx.model_rec != rec)):
    return 0
  if getattr(fx, "form_filled", None) and any(r == rec for r, _ in fx.form_filled):
    return 0                                          # this record's editor was already reached
  ui = pre["ui"]
  if fx._form_plan(ui, rec):
    return 0
  jg = JointGraph(graph)
  raw = ep.out.parents[1]
  here = page_key(ui)
  memo = {}

  def grounds(p):
    if p[0] != ui.package or p == here:
      return False
    if p not in memo:
      memo[p] = False
      for inst, cid, _ in jg.into.get(p, [])[-3:]:
        dui = EV.load_cap(raw, inst, cid)
        if dui is not None and len(fx._form_plan(dui, rec)) >= min_fields:
          memo[p] = True
          break
    return memo[p]

  # every recorded variant of this activity is a start (layouts drift with list content); a path is usable only if
  # its first control is on the live screen
  live_fams = {C.family_of_sel(t.sel) for t in U.enumerate_slots(ui)}
  starts = [p for p in jg.out if p[0] == ui.package and p[1] == ui.activity]
  paths = [pa for pa in jg.paths(ui, grounds, safe=lambda sel: safe_control(sel) or entry_control(sel), starts=starts)
           if pa and pa[0][0] in live_fams]
  m["form_route_lookups"] += 1
  if not paths:
    return 0
  fx.form_route_tried.add(rec)
  path = paths[0]
  cap, hops = pre, 0
  for fam, sel, dst, origins in path:
    live = cap["ui"]
    sl = next((t for t in U.enumerate_slots(live) if C.family_of_sel(t.sel) == fam and (t.safe or entry_control(t.sel))), None)
    if sl is None:
      break
    node = live.nodes[sl.node_idx]
    action = JA.click(*node.center)
    agent._execute_action(action, {})  # pylint: disable=protected-access
    if getattr(ep, "_exec_log", None) is not None:
      ep._exec_log.append({"action": {kk: vv for kk, vv in action.__dict__.items() if vv is not None}, "commit": False})  # pylint: disable=protected-access
    lab = U.label_of(live, node) or sel.get("desc") or sel.get("rid")
    summ = f"[graph] Tapped \"{lab}\"."
    agent._actions.append({"summary": summ})  # pylint: disable=protected-access
    agent._summaries.append(summ)  # pylint: disable=protected-access
    row = {"step": k, "kind": "form_route", "tok": fam, "summary": summ, "origins": sorted(origins), "hop": hops,
           "pre": R.strip(cap)}
    steps.append(row)
    ep.log("steps", row)
    hops += 1
    m["form_route_hops"] += 1
    m["form_route_hops_explore_only"] += origins == frozenset({"explore"})
    cap = full7.capture_stable(ep.dev, f"s{k}_froute{hops}")
    if cap["ui"].activity != dst[1]:
      break
  if not hops:
    return 0
  m["form_route_apps"] += 1
  n = fx.try_form(cap, agent, steps, k, own, m, via="route")
  m["form_route_then_filled"] += bool(n)
  return hops + n
