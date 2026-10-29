"""Same-template trajectory replay (seen-task split, TPL_REPLAY).

A seen demonstration is the model's own action sequence on the same template at another seed. During the episode
a pointer walks that sequence forward: at each screen, the first not-yet-used demo step after the pointer whose
screen has the same activity and whose control can be grounded on the live screen is proposed. A control is
grounded by (control family, label); a label (or typed text) that was a goal value of the demo is replaced by the
value of the same field / record of the current goal; a label that is a value of the demo goal with no counterpart
here, swipes and answers are never replayed (they depend on the content). Demos: successful episodes; when the
template has none, only steps on which two failed demos agree.
"""
import json
import os
from pathlib import Path

import full9  # noqa: F401  (import paths)
import fgraph as FG
import memory as MEM

C, U, R = MEM.C, MEM.U, MEM.R


def _field_id(ui, a):
  """Identity of the field typed into: resource-id, else hint / label text of the focused editable."""
  n = next((m for m in ui.app_nodes() if m.editable and m.visible and m.focused), None)
  if n is None and a.get("x") is not None:
    n = U.element_at(ui, int(a["x"]), int(a["y"]), clickable_only=False)
  if n is None:
    return None
  return n.short_rid or FG.norm(getattr(n, "hint", "") or "") or None


def _same_page(st, ui, vals):
  if st["act"] != ui.activity:
    return False
  if st["struct"] == ui.struct_sig:
    return True
  a = set(st["labels"])
  b = {t for t in MEM.page_labels(ui) if t not in vals}
  return bool(a | b) and len(a & b) / len(a | b) >= 0.6


def _demo(ep_dir):
  ep_dir = Path(ep_dir)
  goal = json.loads((ep_dir / "meta.json").read_text())["goal"]
  P = FG.parse_goal(goal)
  vidx = FG.value_index(P)
  out, hole = [], False
  for l in open(ep_dir / "steps.jsonl"):
    r = json.loads(l)
    if r.get("kind") != "model" or not r.get("pre"):
      continue
    a = r.get("action") or {}
    at = a.get("action_type")
    try:
      ui = R.load_ui(ep_dir, r["pre"]["cid"])
    except Exception:  # pylint: disable=broad-except
      continue
    vals = set(vidx)
    st = {"act": ui.activity, "at": at, "struct": ui.struct_sig,
          "labels": sorted(t for t in MEM.page_labels(ui) if t not in vals)}
    if at in ("click", "long_press") and a.get("x") is not None:
      n = U.element_at(ui, int(a["x"]), int(a["y"]))
      if n is None:
        continue
      lab = FG.norm(U.label_of(ui, n))
      sel = U.selector(ui, n)
      sl = next((x for x in U.enumerate_slots(ui) if x.node_idx == n.idx), None)
      st.update(fam=C.family_of_sel(sel), label=lab, slot=vidx.get(lab), in_list=bool(sel.get("in_list")),
                commit=sl is not None and sl.unsafe_reason in ("unsafe_label", "dialog_action_button"))
      if st["slot"] is None and full9.content_choice(ui, n, st["fam"]):
        hole = True
        continue               # a day cell, a time, a date chip: the demo's content, not this task's
      if st["in_list"] and st["slot"] is None:
        hole = True
        continue               # a list row chosen by its content (which duplicate, which note): the model's choice
    elif at == "input_text" and a.get("text"):
      st.update(text=FG.norm(a["text"]), slot=vidx.get(FG.norm(a["text"])), field=_field_id(ui, a))
      if st["slot"] is None:
        hole = True
        continue               # typed text that is not a goal value cannot be transferred
    elif at in ("navigate_back",):
      pass
    else:
      if at in ("swipe", "scroll"):
        hole = True
      continue                 # swipe / status / answer / open_app / wait: content- or goal-dependent
    st["hole"], hole = hole, False
    out.append(st)
  # record segments: the steps up to (and including) a commit belong to the record whose values they enter
  seg, rec = [], None
  for st in out:
    seg.append(st)
    if st.get("slot") is not None and rec is None:
      rec = st["slot"].get("rec", 0)
    if st.get("commit"):
      for x in seg:
        x["rec"] = rec
      seg, rec = [], None
  for x in seg:
    x["rec"] = rec
  return out


def load_demos(warm_raw, task):
  tk = MEM.template_key(task)
  ok, bad = [], []
  for res in sorted(Path(warm_raw).glob("*/p*/result.json")):
    ep = res.parent
    if MEM.template_key(ep.parent.name) != tk:
      continue
    try:
      d = _demo(ep)
    except Exception:  # pylint: disable=broad-except
      continue
    (ok if json.loads(res.read_text()).get("is_successful") == 1.0 else bad).append(d)
  if ok:
    return ok
  if len(bad) >= 2 and os.environ.get("TPL_FAILED_DEMOS") == "1":   # off: agreement of failed demos is 50% wrong
    # failed demos: keep the steps both demos contain (same activity, family, label) in order
    a, b = bad[0], bad[1]
    key = lambda s: (s["act"], s["at"], s.get("fam"), s.get("label") if s.get("slot") is None else ("slot", s["slot"]["field"]))  # noqa: E731
    kb = [key(s) for s in b]
    return [[s for s in a if key(s) in kb]]
  return []


class Replayer:
  def __init__(self, demos, goal):
    self.demos = demos
    self.P = FG.parse_goal(goal)
    self.ptr = [0] * len(demos)
    self.vals = set(FG.value_index(self.P))
    self.acted = [False] * len(demos)     # the model acted since the demo's pointer last moved

  def _value(self, slot):
    for s in self.P["slots"]:
      if s["field"] == slot["field"] and s.get("rec", 0) == slot.get("rec", 0):
        return s["value"]          # original case (it is typed); compared via FG.norm
    return None

  def propose(self, ui, lookahead=3, done=()):
    """(demo index, step index, kind, node or text) for the live screen, or None. Steps of records already done
    (by a function, by the model) are skipped as a whole: the record must not be entered twice."""
    for di, d in enumerate(self.demos):
      while self.ptr[di] < len(d) and d[self.ptr[di]].get("rec") is not None and d[self.ptr[di]]["rec"] in done:
        self.ptr[di] += 1
      for j in range(self.ptr[di], min(len(d), self.ptr[di] + lookahead)):
        st = d[j]
        if st.get("hole") and not self.acted[di] and j == self.ptr[di]:
          break                # the demo did something here the replay cannot (free text, a content choice)
        if j > self.ptr[di] and (st.get("commit") or st.get("hole")):
          break                # never skip ahead onto a commit or past a hole
        if not _same_page(st, ui, self.vals):
          continue
        if st["at"] == "navigate_back":
          return di, j, "back", None
        want = self._value(st["slot"]) if st.get("slot") else None
        if st.get("slot") and want is None:
          continue
        if st["at"] == "input_text":
          foc = next((m for m in ui.app_nodes() if m.editable and m.visible and m.focused), None)
          if foc is not None and st.get("field") and (foc.short_rid or FG.norm(getattr(foc, "hint", "") or "")) == st["field"]:
            return di, j, "type", want
          continue
        lab = FG.norm(want) if st.get("slot") else st.get("label")
        for sl in U.enumerate_slots(ui):
          n = ui.nodes[sl.node_idx]
          if C.family_of_sel(sl.sel) == st["fam"] and FG.norm(U.label_of(ui, n)) == lab:
            return di, j, st["at"], n
          if st.get("slot") and FG.norm(U.label_of(ui, n)) == lab:
            return di, j, st["at"], n     # a value row: the family may differ with the content
    return None

  def advance(self, di, j):
    self.ptr[di] = j + 1
    self.acted[di] = False

  def observe(self, ui, action, lookahead=4):
    """The model acted itself: if its action is a later step of a demo, move that demo's pointer past it."""
    self.acted = [True] * len(self.demos)
    at = action.get("action_type")
    n = U.element_at(ui, int(action["x"]), int(action["y"])) if action.get("x") is not None else None
    lab = FG.norm(U.label_of(ui, n)) if n is not None else None
    fam = C.family_of_sel(U.selector(ui, n)) if n is not None else None
    for di, d in enumerate(self.demos):
      for j in range(self.ptr[di], min(len(d), self.ptr[di] + lookahead)):
        st = d[j]
        if st["at"] != at or st["act"] != ui.activity:
          continue
        if at == "input_text":
          want = self._value(st["slot"]) if st.get("slot") else None
          if want is not None and FG.norm(action.get("text")) == FG.norm(want):
            self.ptr[di], self.acted[di] = j + 1, False
            break
        elif n is not None and fam == st.get("fam") and (lab == st.get("label") or
                                                         (st.get("slot") and lab == FG.norm(self._value(st["slot"]) or ""))):
          self.ptr[di], self.acted[di] = j + 1, False
          break
