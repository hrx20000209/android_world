"""Function library over the observation graph (EchoTrail-style memory, model-free).

A function = a verified path of observed clicks from an app entry state (the
landing page after OPEN_APP) to a state that exposes a target affordance
(a visible label). Built by BFS over observed transitions; every edge is a
real observation (history, or this runtime's exploration). Task-parameter
text of the source episode is never used as a target or a step label.

retrieve(goal): functions whose target is an affordance (a control label some
episode actually tapped, not data text) that occurs verbatim in the goal
(phrase match, app titles excluded), shortest first, at most K = 2
(EchoTrail found K = 2 optimal). render(): step list for the prompt.
"""

from __future__ import annotations

from collections import defaultdict, deque

import memory as M
import retrieval as RT

C = M.C
K = 2
MAX_DEPTH = 4


def _step_label(o):
  s = o.get("sel") or {}
  return (s.get("label") or s.get("desc") or s.get("rid") or "").strip()


class Library:
  def __init__(self, snap, runtime=None):
    self.edges = defaultdict(dict)            # src key -> {dst key: (label, fam, title)}
    self.labels = defaultdict(set)            # state key -> structural labels
    self.title = {}
    self.entry = defaultdict(set)             # package -> entry state keys
    self.afford = set()                       # labels of controls someone actually tapped (non-parameter)
    for o in snap.obs:
      if o["op"] in ("click", "long_press") and not o["param"] and _step_label(o):
        self.afford.add(M.norm(_step_label(o)))
    for o in snap.obs:
      if not o["dst"]:
        continue
      if o["op"] == "open_app":
        self.entry[o["dst"]["pkg"]].add(o["dst"]["key"])
      labs = M.structural_labels(o)
      self.labels[o["dst"]["key"]].update(labs)
      self.title.setdefault(o["dst"]["key"], labs[0] if labs else "")
      safe = bool(o.get("slot") and o["slot"].get("safe"))      # navigation only: no commits / fields / toggles
      if o["op"] == "click" and safe and not o["param"] and o["src"]["key"] != o["dst"]["key"] and _step_label(o):
        self.edges[o["src"]["key"]].setdefault(o["dst"]["key"], (_step_label(o), o["fam"]))
    for (src, fam), recs in (runtime or {}).items():
      for d in recs:
        self.labels[d["key"]].update(d["labels"])
        self.title.setdefault(d["key"], d["labels"][0] if d["labels"] else "")

  def add_runtime_edge(self, src, dst, label, fam, labels):
    if src != dst and label:
      self.edges[src][dst] = (label, fam)
    self.labels[dst].update(labels)
    self.title.setdefault(dst, labels[0] if labels else "")

  def paths_from(self, starts):
    """BFS tree: state -> (path of (label, src title)) from the nearest start."""
    best = {s: [] for s in starts}
    q = deque(starts)
    while q:
      u = q.popleft()
      if len(best[u]) >= MAX_DEPTH:
        continue
      for v, (lab, fam) in self.edges.get(u, {}).items():
        if v not in best:
          best[v] = best[u] + [(lab, self.title.get(u, ""))]
          q.append(v)
    return best

  def retrieve(self, goal, pkg, here=None, visible=()):
    gn = M.norm(goal)
    if any(RT.goal_named(t, gn) and t in self.afford for t in visible):
      return []                        # the target affordance is already on screen
    starts = set(self.entry.get(pkg, set()))
    if here:
      starts.add(here)
    tree = self.paths_from(starts)
    found = []
    for st, path in tree.items():
      for lab in self.labels.get(st, ()):
        if lab in self.afford and RT.goal_named(lab, gn):   # targets are affordances, not data text
          found.append((len(path), lab, st, path))
    found.sort(key=lambda x: (x[0], x[1]))
    out, seen = [], set()
    for n, lab, st, path in found:
      if lab in seen or not path:
        continue
      seen.add(lab)
      out.append({"target": lab, "state": st, "path": path})
      if len(out) >= K:
        break
    return out


def render(funcs, header="Navigation memory for this app (paths actually observed earlier; they may be outdated):"):
  if not funcs:
    return ""
  lines = [header]
  for i, f in enumerate(funcs, 1):
    steps = "; ".join(f"on \"{t or 'the screen'}\" tap \"{lab}\"" for lab, t in f["path"])
    lines.append(f"{i}. To reach a screen showing \"{f['target']}\": {steps}.")
  return "\n".join(lines)
