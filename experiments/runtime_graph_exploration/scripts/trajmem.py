"""EchoTrail-style trajectory memory (the trajectory-reuse baseline).

Memory = successful historical episodes (any template, any seed; recorded in
the DEFAULT runtime). Retrieval = BM25 between the goal and historical goals
of episodes that used the same app package; K = 2. Each step is rendered as
{screen title, action}; typed text is kept verbatim as in EchoTrail (it is a
demonstration, not executed by us).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

import memory as M

C, U, R = M.C, M.U, M.R
K = 2


def _terms(t):
  return [w for w in re.findall(r"[a-z]+", t.lower()) if w not in M.STOP and len(w) > 2]


class TrajMemory:
  def __init__(self, raw_root=M.TRV / "raw/full", exclude_task_seed=None):
    self.eps = []
    for ep in raw_root.glob("*/*/p*"):
      try:
        r = json.loads((ep / "result.json").read_text()); m = json.loads((ep / "meta.json").read_text())
      except Exception:
        continue
      if r.get("is_successful") != 1.0 or (m["task"], m["seed"]) == exclude_task_seed:
        continue
      rows = [json.loads(l) for l in open(ep / "steps.jsonl")]
      if any(x.get("kind") != "model" for x in rows):
        continue                                   # keep pure agent demonstrations only
      self.eps.append({"ep": ep, "goal": m["goal"], "task": m["task"], "rows": rows, "pkg": None})
    self.df = Counter(w for e in self.eps for w in set(_terms(e["goal"])))
    self.N = max(1, len(self.eps))

  def _pkg(self, e):
    if e["pkg"] is None:
      e["pkg"] = ""
      for s in e["rows"]:
        ui = R.load_ui(e["ep"], s["pre"]["cid"])
        if "launcher" not in ui.package:
          e["pkg"] = ui.package
          break
    return e["pkg"]

  def retrieve(self, goal, pkg):
    q = set(_terms(goal))
    scored = []
    for e in self.eps:
      d = Counter(_terms(e["goal"]))
      s = sum(math.log(1 + self.N / (1 + self.df[w])) * d[w] for w in q if w in d)
      if s > 0:
        scored.append((s, e))
    scored.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for s, e in scored:
      if self._pkg(e) != pkg or e["goal"] in seen:
        continue
      seen.add(e["goal"])
      out.append(e)
      if len(out) >= K:
        break
    return out

  def render(self, eps):
    if not eps:
      return ""
    lines = ["Trajectories of similar tasks that succeeded earlier (you can follow them as guidance):"]
    for i, e in enumerate(eps, 1):
      lines.append(f"Example {i} - task: {e['goal']}")
      for j, s in enumerate(e["rows"], 1):
        ui = R.load_ui(e["ep"], s["pre"]["cid"])
        title = next((n.text for n in sorted(ui.app_nodes(), key=lambda n: (n.bounds[1], n.bounds[0]))
                      if n.visible and n.text), "")[:40]
        a = s["action"]
        at = a.get("action_type")
        if at in ("click", "long_press") and a.get("x") is not None:
          n = U.element_at(ui, int(a["x"]), int(a["y"]))
          lab = (U.label_of(ui, n) if n is not None else "") or "element"
          act = f"{at} \"{lab[:40]}\""
        elif at == "input_text":
          act = f"type \"{(a.get('text') or '')[:40]}\""
        elif at == "open_app":
          act = f"open app \"{a.get('app_name')}\""
        else:
          act = at + (f" {a.get('direction')}" if a.get("direction") else "")
        lines.append(f"  step {j}: on \"{title}\" -> {act}")
    return "\n".join(lines)
