"""Stage B: task-conditioned retrieval and step-level applicability.

Retrieval  BM25 over historical goals of other templates (one document per
           template). App names, numbers and stopwords are not query terms,
           so a shared app name alone never retrieves memory. No positive
           score -> NO_APPLICABLE_MEMORY.

Relevance  is never inferred from historical frequency (survivor-biased,
           FULL3). A navigation step is task-authorized only by GOAL
           ANCHORS: UI labels that occur verbatim (whole phrase) in the
           current instruction. A slot is anchor-authorized when observed
           evidence (historical structure or this runtime) shows that its
           successor exposes a goal anchor that is not on the current page.
           Structural evidence is task-independent, so it may come from any
           template; the decision it supports comes from the current goal.

Classes    DIRECT, PARAMETERIZED, ENVIRONMENT_UNCERTAIN, INCOMPLETE,
           INAPPLICABLE, UNSAFE, UNKNOWN (no evidence at all).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import memory as M

C, U = M.C, M.U
APP_WORDS = set("markor broccoli expense pro simple calendar sms messenger tasks opentracks osmand osm joplin "
                "retro music vlc gallery draw camera clock contacts files chrome settings audio recorder "
                "sports tracker".split())


def terms(goal):
  ws = re.findall(r"[a-z]+", goal.lower())
  return [w for w in ws if w not in M.STOP and w not in APP_WORDS and len(w) > 2]


class BM25:
  def __init__(self, snap, k1=1.2, b=0.75):
    self.docs = {t: Counter(w for g in gs for w in terms(g)) for t, gs in snap.goals.items()}
    self.N = len(self.docs)
    self.avg = sum(sum(d.values()) for d in self.docs.values()) / max(1, self.N)
    self.df = Counter(w for d in self.docs.values() for w in d)
    self.k1, self.b = k1, b

  def search(self, goal, k=3):
    q = set(terms(goal))
    out = []
    for t, d in self.docs.items():
      L = sum(d.values())
      s = 0.0
      for w in q:
        if w in d:
          idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
          s += idf * d[w] * (self.k1 + 1) / (d[w] + self.k1 * (1 - self.b + self.b * L / self.avg))
      if s > 0:
        out.append((t, round(s, 3)))
    out.sort(key=lambda x: -x[1])
    return out[:k]


_GN = {}


def goal_named(label, goal_n):
  k = (label, goal_n)
  if k not in _GN:
    _GN[k] = _goal_named(label, goal_n)
  return _GN[k]


def _goal_named(label, goal_n):
  t = M.norm(label)
  if len(t) < 3 or all(w in M.STOP or w in APP_WORDS for w in re.findall(r"[a-z]+", t)):
    return False                     # app titles / stopword-only labels are not task anchors
  return re.search(r"(?<![\w])" + re.escape(t) + r"(?![\w])", goal_n) is not None


def live_anchors(ui, goal_n):
  return {t for t in M.page_labels(ui) if goal_named(t, goal_n)}


class Evidence:
  """Successor evidence for (state, slot): historical (snapshot) + runtime."""

  def __init__(self, snap):
    self.snap = snap
    self.runtime = defaultdict(list)   # (src key, fam) -> [{"dst": key, "act":..., "labels": [...]}]

  def add_runtime(self, src_key, fam, ui_dst):
    self.runtime[(src_key, fam)].append({"key": M.state_key(ui_dst), "act": ui_dst.activity,
                                         "labels": M.page_labels(ui_dst)})

  def successors(self, src_key, fam):
    """(status, [dst records]) with status FRESH / STALE / CONFLICTING / UNKNOWN."""
    rt = self.runtime.get((src_key, fam))
    if rt:
      return "FRESH", rt
    hist = [dict(o["dst"], labels=M.structural_labels(o)) for o in self.snap.by_edge.get((src_key, fam), []) if o["dst"]]
    if not hist:
      return "UNKNOWN", []
    acts = {d["act"] for d in hist}
    return ("CONFLICTING" if len(acts) > 1 else "STALE"), hist


def classify(ui, goal, ev, max_hops=2):
  """Per-slot applicability on the live page. Returns (rows, page_info)."""
  gn = M.norm(goal)
  here = live_anchors(ui, gn)
  skey = M.state_key(ui)
  rows = []
  for sl in U.enumerate_slots(ui):
    fam = C.family_of_sel(sl.sel)
    lab = sl.sel.get("label") or sl.sel.get("desc") or ""
    st, dsts = ev.successors(skey, fam)
    reach = set()
    for d in dsts:
      reach |= {t for t in d.get("labels", []) if goal_named(t, gn)} - here
    hop2 = set()
    if not reach and max_hops >= 2:
      for k2 in {d["key"] for d in dsts}:
        for t in ev.snap.labels_after(k2):
          if goal_named(t, gn):
            hop2.add(t)
      hop2 -= here
    if not sl.safe:
      cls = "UNSAFE"
    elif sl.sel.get("in_list") or goal_named(lab, gn):
      cls = "PARAMETERIZED"          # entity row / goal-named control: semantic choice
    elif st == "UNKNOWN":
      cls = "UNKNOWN"
    elif reach and st == "CONFLICTING":
      cls = "ENVIRONMENT_UNCERTAIN"
    elif reach:
      cls = "ANCHOR1"
    elif hop2:
      cls = "ANCHOR2"
    else:
      cls = "INAPPLICABLE"
    rows.append({"slot": sl, "fam": fam, "label": lab, "status": st, "cls": cls, "reach": sorted(reach),
                 "reach2": sorted(hop2)})
  a1 = [r for r in rows if r["cls"] == "ANCHOR1"]
  unknown = [r for r in rows if r["cls"] == "UNKNOWN"]
  direct = None
  # DIRECT: no goal anchor already on the page (the semantic step is the
  # model's), exactly one safe slot is evidenced to expose a new anchor, and
  # no safe slot remains with unknown successor (it could expose one too).
  if not here and len(a1) == 1 and not unknown:
    direct = a1[0]
    direct["cls"] = "DIRECT"
  return rows, {"anchors_here": sorted(here), "direct": direct, "n_unknown": len(unknown), "n_anchor1": len(a1)}
