"""FULL24: the node x term matrix that mixes inference and exploration trajectories (scripts/term_matrix.py) used
online to choose graph navigation without inference.

M[n] = W_CONTENT * tfidf(page words of n)
     + sum over other tasks T that reached n:  W_INFER (model navigated there) or W_EXPLORE (a probe opened it) * g_T
rel(n | g) = cos(g, M[n]);  R(n) = max(rel(n), GAMMA * max_{n->n'} rel(n')).
On page P, every live, unique, safe navigation control with a cross-task known out-edge is scored by R(dst);
the top control is executed when R(top) - R(second) >= MARGIN (offline: margin >= 0.55 ~ the most confident 25%,
13-15% of navigation actions, precision 0.91-0.97 on FULL21 / FULL21R / FULL23W).
The matrix is updated incrementally from the run's graph file; the current task's own goal never enters it.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import full9
import explore_value as EV
import joint_graph as JG
import state_abstraction as SA

R, U, C = full9.R, full9.U, full9.C
W_CONTENT, W_INFER, W_EXPLORE, GAMMA, MARGIN = 1.0, 1.0, 0.5, 0.7, 0.55


def _pk(s):
  return (s.get("package"), s.get("activity"), s.get("layout_sig"))


def _unit(v):
  n = math.sqrt(sum(x * x for x in v.values())) or 1.0
  return {t: x / n for t, x in v.items()}


def cos(a, b):
  if not a or not b:
    return 0.0
  dot = sum(v * b.get(t, 0.0) for t, v in a.items())
  if dot == 0:
    return 0.0
  return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


class _Ep:
  def __init__(s, t):
    s.args = type("a", (), {"task": t})()


class TermMatrix:
  def __init__(self, graph_path: Path, raw: Path, task: str):
    self.path, self.raw, self.task = graph_path, raw, task
    self.offset = 0
    self.goals = {}
    self.content = {}
    self.ctx = defaultdict(Counter)                  # node -> summed weighted goal terms of OTHER tasks
    self.out = defaultdict(lambda: defaultdict(Counter))   # src -> fam -> Counter(dst)
    self.edge_tasks = defaultdict(set)
    self.skel = {}                                   # page -> skeleton multiset (for soft matching)
    self.by_am = defaultdict(set)                    # (activity, mode) -> pages
    self.page_tasks = defaultdict(set)               # page -> tasks whose trajectories (agent) visited it
    self.update()

  def _goal(self, task):
    if task not in self.goals:
      try:
        goal = json.load(open(self.raw / task / "p0" / "meta.json"))["goal"]
        self.goals[task] = _unit(EV.goal_vector(goal, full9.FuncExec(_Ep(task), goal)))
      except Exception:  # pylint: disable=broad-except
        self.goals[task] = {}
    return self.goals[task]

  def update(self):
    if not self.path.exists():
      return
    with self.path.open() as f:
      f.seek(self.offset)
      for line in f:
        if line.strip():
          self.ingest(json.loads(line))
      self.offset = f.tell()

  def ingest(self, o):
        if not o.get("dst"):
          return
        src, dst = _pk(o["src"]), _pk(o["dst"])
        if src == dst:
          return
        for page, ck, u_ in ((src, "src_cid", None), (dst, "dst_cid", None)):
          if page not in self.skel and o.get(ck):
            ui_ = EV.load_cap(self.raw, o["instance"], o[ck])
            if ui_ is not None:
              self.skel[page] = Counter(i for i in ui_.skeleton() if i.split("|")[1] not in SA.SYSTEM)
              self.by_am[(ui_.activity, SA.mode(ui_))].add(page)
        if o.get("origin") != "explore":
          self.page_tasks[src].add(o["instance"].split("/")[0])
        sid = o["slot_id"]
        # agent edges carry the model's action token (E|…, L|… list role, LONG_PRESS:…); probe edges a slot id
        fam = C.family_of_sel(o["selector"]) if (o.get("origin") == "explore" and o.get("selector")) else sid
        task = o["instance"].split("/")[0]
        self.out[src][fam][dst] += 1
        self.edge_tasks[(src, fam)].add(task)
        if dst not in self.content and o.get("dst_cid"):
          ui = EV.load_cap(self.raw, o["instance"], o["dst_cid"])
          if ui is not None:
            v = Counter()
            for t in EV.page_tokens(ui):
              v[t] += EV.w_ui(t)
            self.content[dst] = _unit(v)
        if task != self.task:
          w = W_EXPLORE if o.get("origin") == "explore" else W_INFER
          for t, x in self._goal(task).items():
            self.ctx[dst][t] += w * x

  def node_vec(self, n):
    v = Counter({t: W_CONTENT * x for t, x in self.content.get(n, {}).items()})
    v.update(self.ctx.get(n, {}))
    return v

  def backed(self, n, g):
    best = cos(g, self.node_vec(n))
    for ds in self.out.get(n, {}).values():
      for n2 in ds:
        if n2 != n:
          best = max(best, GAMMA * cos(g, self.node_vec(n2)))
    return best


def pick(ep, ui, graph):
  tm = ep._tmat  # pylint: disable=protected-access
  tm.update()
  g = ep._gvec_unit  # pylint: disable=protected-access
  P, _sim = soft_page(tm, ui)
  slots = U.enumerate_slots(ui)
  fams = Counter(C.family_of_sel(t.sel) for t in slots)
  cands = {}
  for fam, ds in tm.out.get(P, {}).items():
    if fams.get(fam, 0) == 0 or not (tm.edge_tasks[(P, fam)] - {tm.task}):
      continue
    cands[fam] = ds
  if not cands:
    return None
  scored = sorted(((max(tm.backed(n, g) for n in ds), fam) for fam, ds in cands.items()), reverse=True)
  top, fam = scored[0]
  second = scored[1][0] if len(scored) > 1 else 0.0
  if top - second < MARGIN or (P, fam) in ep._macro_used:  # pylint: disable=protected-access
    return None
  ds = cands[fam]
  acts = Counter()
  for d, c in ds.items():
    acts[d[1]] += c
  act, n_act = acts.most_common(1)[0]
  if n_act / sum(acts.values()) < 0.9:
    return None
  if fams[fam] != 1:
    return None                    # repeated family = entity instances: a parameter, not navigation
  sl = next(t for t in slots if C.family_of_sel(t.sel) == fam)
  if not sl.safe or sl.sel.get("in_list") or full9.effectful(sl.sel) or sl.unsafe_reason in ("unsafe_label", "dialog_action_button"):
    return None
  return {"slot": sl, "fam": fam, "key": (ui.package, ui.activity, fam), "pred_act": act,
          "intent": round(top, 3), "margin": round(top - second, 3)}


TAU = 0.7


def _jac(a, b):
  inter = sum((a & b).values()); uni = sum((a | b).values())
  return inter / uni if uni else 1.0


def soft_page(tm, ui):
  """The known page the live screen is: the exact layout key if known, else the most similar known page of the same
  (activity, mode) with skeleton Jaccard >= TAU (page variants produced by dynamic content share edges)."""
  P = JG.page_key(ui)
  if P in tm.out:
    return P, 1.0
  s = Counter(i for i in ui.skeleton() if i.split("|")[1] not in SA.SYSTEM)
  best = None
  for page in tm.by_am.get((ui.activity, SA.mode(ui)), ()):
    if page not in tm.out:
      continue
    j = _jac(s, tm.skel.get(page, Counter()))
    if j >= TAU and (best is None or j > best[1]):
      best = (page, j)
  return best if best else (P, 0.0)


def traffic(tm, ui):
  """How many tasks' trajectories passed through this page (soft-matched)."""
  P, _ = soft_page(tm, ui)
  return len(tm.page_tasks.get(P, ()))
