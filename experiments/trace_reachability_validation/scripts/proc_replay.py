"""R3: episode-local entity procedure replay (FULL5).

Motivation (measured): across-run graph evidence arrives too late for
AndroidWorld (each task is seen <= 2 times, FULL4 executed 9 jump steps), and
cross-task evidence is survivor-biased (FULL3). Multi-entity tasks repeat
the SAME procedure for each goal entity inside one episode; replaying the
model's own just-demonstrated procedure saved 5.5 steps per triggered task
with unchanged success in the earlier serial-runner study.

Design:
- Hop log: every model click / long-press with its page (activity + layout)
  before and after.
- Entity: the text of a clicked list row (row-family token) that occurs in the
  goal as a whole word sequence. Model-free; no extra model call.
- Template: from the model's entity hop up to (not including) the next commit
  hop the model executed itself; the hops in between must be non-entity element
  clicks (reversible navigation); the body may be empty (replay = row tap).
- Plan: on the template's start page (same activity, entity rows visible), with no open
  entity, the next undone goal entity IN GOAL ORDER must be visible as a row of
  the same family; otherwise do nothing (never skip ahead of task order).
- Execute: substitute the row, follow the template, verify each landing
  (activity; the next hop's control must be present); stop on the first mismatch. Replay never presses the
  commit: the model does it, which closes the entity (open-entity lock).
- History: every executed replay hop is written into the agent's history.
"""

from __future__ import annotations

import re

import common as C

U = C.U


def inner_texts(ui, node):
  out, todo = [], [node.idx]
  while todo:
    n = ui.nodes[todo.pop(0)]
    for t in (n.text, n.desc):
      if t and t.strip():
        out.append(t.strip())
    todo.extend(n.children)
  return out


def row_fam(node):
  """Entity-row identity: class + resource-id only (text is the entity)."""
  return f"{node.short_cls}|{node.short_rid}"


class ProcReplay:
  def __init__(self, goal: str, enum_only: bool = False):
    self.enum_only = enum_only   # FULL7: the next entity must be listed next to a processed one in the goal
    self.goal = goal
    self.gl = goal.lower()
    self.hops = []          # model + replay hops, in order
    self.done = set()       # entities whose procedure reached a model commit
    self.tried = set()      # entities already replayed once (never twice)
    self.open = None        # entity whose procedure is in progress
    self.seen = {}          # entity -> goal position (all candidates seen so far)

  # ------------------------------------------------------------ entities
  def goal_pos(self, text):
    t = text.strip().lower()
    if len(t) < 3 or t == self.gl:
      return None
    m = re.search(r"(?<![\w])" + re.escape(t) + r"(?![\w])", self.gl)
    return m.start() if m else None

  def entity_of(self, ui, node):
    best = None
    for t in inner_texts(ui, node):
      p = self.goal_pos(t)
      if p is not None and (best is None or len(t) > len(best[0])):
        best = (t, p)
    return best

  def scan(self, ui, fam):
    """Visible rows of `fam` that carry a goal entity: entity -> node."""
    vis = {}
    for n in ui.app_nodes():
      if not n.visible or not (n.clickable or n.long_clickable) or row_fam(n) != fam:
        continue
      e = self.entity_of(ui, n)
      if e:
        vis.setdefault(e[0], n)
        self.seen.setdefault(e[0], e[1])
    return vis

  # -------------------------------------------------------------- record
  def record(self, ui, action, tok, sel, is_commit, origin="model"):
    at = action.get("action_type")
    hop = {"tok": tok, "at": at, "origin": origin, "commit": bool(is_commit), "entity": None, "row_fam": None,
           "pre_act": ui.activity, "pre_layout": ui.layout_sig, "post_act": None, "post_layout": None}
    if at in ("click", "long_press") and action.get("x") is not None:
      n = U.element_at(ui, int(action["x"]), int(action["y"]))
      if n is not None and not is_commit:
        e = self.entity_of(ui, n)
        if e:
          hop["entity"] = e[0]
          hop["row_fam"] = row_fam(n)
          self.seen.setdefault(e[0], e[1])
          if origin == "model":
            self.open = e[0]
    if is_commit and self.open is not None:
      self.done.add(self.open)
      self.open = None
    self.hops.append(hop)

  def set_post(self, ui):
    if self.hops and self.hops[-1]["post_layout"] is None:
      self.hops[-1]["post_act"] = ui.activity
      self.hops[-1]["post_layout"] = ui.layout_sig

  def _enumerated(self, ent):
    """True if `ent` appears in a goal enumeration next to an already processed entity:
    only separators (commas, semicolons, 'and', quotes, whitespace) lie between them."""
    pos = self.goal_pos(ent)
    for e in self.done | self.tried:
      p0 = self.goal_pos(e)
      if p0 is None or pos is None:
        continue
      a, b = (p0 + len(e), pos) if p0 < pos else (pos + len(ent), p0)
      gap = self.gl[a:b]
      if re.fullmatch(r"[\s,;'\"]*(and\s+)?[\s'\"]*", gap):
        return True
    return False

  # ------------------------------------------------------------ template
  def template(self):
    """Latest model-demonstrated entity procedure that ended in a model commit."""
    for i in range(len(self.hops) - 1, -1, -1):
      h = self.hops[i]
      if h["entity"] is None or h["origin"] != "model":
        continue
      body = []
      for g in self.hops[i + 1:]:
        if g["commit"]:
          if g["origin"] != "model":
            break
          return [h] + body
        if g["at"] != "click" or not g["tok"].startswith("E|") or g["entity"] is not None or g["post_layout"] is None:
          break
        body.append(g)
      # this entity hop is not a complete demonstration; try an earlier one
    return None

  def plan(self, ui):
    if self.open is not None:
      return None
    tpl = self.template()
    if not tpl:
      return None
    h0 = tpl[0]
    vis = self.scan(ui, h0["row_fam"])
    # start page = same activity AND entity rows of the demonstrated family on
    # screen; layout signatures re-land only 66% (snackbars, counts), activity 96%
    if ui.activity != h0["pre_act"] or not vis:
      self.why = "page"
      return None
    pending = sorted((p, e) for e, p in self.seen.items() if e not in self.done and e not in self.tried)
    if not pending:
      self.why = "no_pending"
      return None
    nxt = pending[0][1]
    if self.enum_only and not self._enumerated(nxt):
      self.why = "not_enumerated"
      return None
    if nxt not in vis:
      self.why = "not_visible"
      return None             # next entity in goal order is not on screen: do not skip ahead
    return {"entity": nxt, "node": vis[nxt], "template": tpl}
