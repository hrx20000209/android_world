"""Graph-derived next-element recommendation injected into the prompt.

One recommendation per screen, read from the graph (no model call):
  1. goal-directed: a safe slot whose observed successor exposes a goal anchor
     (label quoted in the instruction) that is not on this screen
  2. otherwise trace: the element most often tapped next at this screen in
     earlier tasks of the same app (support >= 2 distinct tasks, present on
     screen, not a task-parameter row)
No recommendation when a goal anchor is already visible (the target itself is
on screen) or when neither source has evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import memory as M
import retrieval as RT

C, U = M.C, M.U
TEMPLATE = ("Graph memory hint (derived automatically from earlier interactions with this app; it can be wrong, "
            "check it against the task and the screen):\n"
            "- The element most likely useful next is \"{label}\" at point ({x}, {y}). {reason}")


def recommend(ui, goal, ev):
  rows, info = RT.classify(ui, goal, ev)
  if info["anchors_here"]:
    return None
  gn = M.norm(goal)
  a1 = [r for r in rows if r["cls"] in ("ANCHOR1", "DIRECT", "ENVIRONMENT_UNCERTAIN")]
  if a1:
    r = a1[0]
    return {"slot": r["slot"], "fam": r["fam"], "source": "goal",
            "reason": f"In earlier runs, tapping it opened a screen showing \"{r['reach'][0]}\", which the task mentions."}
  key = M.state_key(ui)
  tasks = defaultdict(set)
  for o in ev.snap.by_state.get(key, []):
    if o["fam"] and o["op"] == "click" and not o["param"]:
      tasks[o["fam"]].add(o["template"])
  on_screen = {C.family_of_sel(sl.sel): sl for sl in U.enumerate_slots(ui) if not sl.sel.get("in_list")}
  cands = [(len(t), f) for f, t in tasks.items() if f in on_screen and len(t) >= 2]
  if not cands:
    return None
  n, f = max(cands)
  return {"slot": on_screen[f], "fam": f, "source": "trace",
          "reason": f"In {n} earlier tasks of this app, it was the next element tapped on this screen."}


def text(ui, rec):
  node = ui.nodes[rec["slot"].node_idx]
  label = U.label_of(ui, node) or rec["slot"].sel.get("desc") or rec["slot"].sel.get("rid") or "element"
  x, y = node.center
  W, H = ui.screen
  return TEMPLATE.format(label=label[:60], x=round(x * 1000 / W), y=round(y * 1000 / H), reason=rec["reason"])


def placebo(ui, rec):
  """Same template, a different safe non-row element on the same screen."""
  others = [sl for sl in U.enumerate_slots(ui)
            if sl.safe and not sl.sel.get("in_list") and C.family_of_sel(sl.sel) != rec["fam"]]
  if not others:
    return None
  sl = others[len(others) // 2]
  return {"slot": sl, "fam": C.family_of_sel(sl.sel), "source": "placebo", "reason": rec["reason"]}
