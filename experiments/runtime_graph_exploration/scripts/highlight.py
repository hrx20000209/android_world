"""Graph-derived visual highlighting for the VLM screenshot.

Instead of injecting text, draw a box around the on-screen element(s) that the
graph says lead toward the task goal. Read from the graph only (no model call):

  goal anchors   labels that occur verbatim in the instruction (retrieval.goal_named)
  priority 1     ANCHOR1: a safe slot whose observed successor (history or this
                 runtime's exploration) exposes a goal anchor not on this page
  (none)         a goal anchor already visible -> no highlight: the model can see
                 the target itself; nothing on this page needs pointing out

At most MAX_BOXES elements; ties keep screen order. The prompt text is unchanged.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

import memory as M
import retrieval as RT

U = M.U
MAX_BOXES = 2


def select(ui, goal, ev):
  rows, info = RT.classify(ui, goal, ev)
  if info["anchors_here"]:
    return [], {"why": "anchor_visible", "anchors": info["anchors_here"]}
  a1 = [r for r in rows if r["cls"] in ("ANCHOR1", "DIRECT", "ENVIRONMENT_UNCERTAIN")]
  # ANCHOR2 (two observed hops) is not used: offline it matched generic form
  # words ("description") two pages away and boxed time-picker numbers.
  pick = a1
  if not pick:
    return [], {"why": "no_graph_evidence"}
  sel = pick[:MAX_BOXES]
  return sel, {"why": "anchor1" if a1 else "anchor2", "n_candidates": len(pick),
               "reach": [r["reach"] or r["reach2"] for r in sel],
               "source": [r["status"] for r in sel]}


def draw(img: Image.Image, ui, rows, color=(255, 0, 0), width=8):
  img = img.copy()
  W, H = img.size
  sx, sy = W / ui.screen[0], H / ui.screen[1]
  d = ImageDraw.Draw(img)
  for r in rows:
    l, t, rr, b = ui.nodes[r["slot"].node_idx].bounds
    d.rectangle([l * sx - 4, t * sy - 4, rr * sx + 4, b * sy + 4], outline=color, width=width)
  return img
