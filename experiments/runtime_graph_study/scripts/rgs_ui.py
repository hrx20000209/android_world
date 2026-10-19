"""UI-state parsing, signatures, action slots and element correspondence.

Everything here is model-free and deterministic: it only reads the raw flat
accessibility dump produced by the fast a11y service
(`tools/fast_a11y_dumper`, `flat 0 10000` socket request).

Definitions (used consistently by the runner and by every analysis script):

* system nodes: nodes whose package is com.android.systemui (status bar clock,
  battery, navigation bar). They are kept in raw evidence but excluded from
  every signature, because the clock alone would make every state unique.
* list content: nodes that have a scrollable ancestor. Their text is runtime
  data (expense names, notes, events), so it is excluded from the structural
  signature but kept in the content signature.
* exact_sig    = activity + every non-system node (class, rid, text, desc,
                 bounds, checked, selected).
* struct_sig   = activity + IME visibility + multiset of (class, rid, flags) of
                 nodes outside list content + "anchor" texts of non-editable
                 nodes outside list content (toolbar titles, button labels) +
                 (class, rid) of list containers.
* layout_sig   = struct_sig without anchor texts (coarser; used only to flag a
                 possible correspondence as AMBIGUOUS, never to merge).
* content_sig  = list-content text multiset + editable field contents +
                 checked/selected state. Used for rollback verification so a
                 probe that silently changed data or scroll position is caught.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

SYSTEM_PACKAGES = {"com.android.systemui"}
IME_PACKAGES_PREFIX = ("com.google.android.inputmethod", "com.android.inputmethod")
LAUNCHER_PACKAGES = {"com.google.android.apps.nexuslauncher"}

# Labels whose probing could have a side effect that BACK cannot undo.
# Deliberately conservative; the list is not a semantic scorer, it is a
# deny-list applied before deterministic round-robin ordering.
UNSAFE_LABEL_RE = re.compile(
    r"\b(delete|remove|trash|discard|clear|reset|erase|uninstall|send|call|dial|"
    r"buy|pay|purchase|order|save|done|ok|okay|confirm|submit|apply|accept|agree|"
    r"allow|deny|sign|log ?out|logout|share|post|publish|archive|move|rename|"
    r"record|stop|start|pause|resume|play|shutter|take photo|take picture|capture|"
    r"video|turn (on|off)|enable|disable|install|update|sync|backup|restore|"
    r"paste|cut|undo|redo|empty|format|factory|wipe|duplicate|copy|clone|import|"
    r"export|print|logout|forget|disconnect|pair|connect|download|upload|"
    r"set|yes|no|cancel|close|exit|quit|finish|navigate up|back)\b",
    re.I,
)


# Scrims / outside-touch areas close an overlay; BACK cannot reopen it.
DISMISS_RID_RE = re.compile(r"touch_outside|scrim|outside|dismiss|design_bottom_sheet_background", re.I)
# File-system parent entries ("..") are navigate-up, like "Navigate up".
NAV_UP_LABEL_RE = re.compile(r"(^|\s)\.\.$")
# Android AlertDialog action buttons commit or dismiss the dialog.
DIALOG_BUTTON_RID_RE = re.compile(r"^button[123]$")


def _h(obj: Any) -> str:
  return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


@dataclass
class Node:
  idx: int
  parent: int | None
  depth: int
  window: int
  cls: str
  pkg: str
  rid: str
  text: str
  desc: str
  bounds: tuple[int, int, int, int]
  clickable: bool
  long_clickable: bool
  scrollable: bool
  editable: bool
  enabled: bool
  visible: bool
  checkable: bool
  checked: bool
  selected: bool
  focused: bool
  child_count: int
  children: list[int] = field(default_factory=list)
  in_list: bool = False  # has a scrollable ancestor
  system: bool = False
  ime: bool = False
  hint: str = ""  # EditText hint (accessibility hintText); empty fields often expose it as their text

  @property
  def short_cls(self) -> str:
    return self.cls.rsplit(".", 1)[-1]

  @property
  def short_rid(self) -> str:
    return self.rid.split("/", 1)[-1] if self.rid else ""

  @property
  def center(self) -> tuple[int, int]:
    l, t, r, b = self.bounds
    return ((l + r) // 2, (t + b) // 2)

  @property
  def area(self) -> int:
    l, t, r, b = self.bounds
    return max(0, r - l) * max(0, b - t)


@dataclass
class UIState:
  activity: str
  package: str
  nodes: list[Node]
  screen: tuple[int, int] = (1080, 2400)

  # ---- derived -------------------------------------------------------
  def app_nodes(self) -> list[Node]:
    return [n for n in self.nodes if not n.system and not n.ime]

  def ime_visible(self) -> bool:
    return any(n.ime for n in self.nodes)

  def anchor_texts(self) -> list[str]:
    out = []
    for n in self.app_nodes():
      if n.in_list or n.editable or not n.visible:
        continue
      for t in (n.text, n.desc):
        t = _norm_text(t)
        if t and not _looks_dynamic(t):
          out.append(t)
    return sorted(out)

  def skeleton(self) -> list[str]:
    out = []
    for n in self.app_nodes():
      if n.in_list:
        continue
      out.append("|".join([
          n.short_cls, n.short_rid, "c" if n.clickable else "", "e" if n.editable else "",
          "s" if n.scrollable else "", "k" if n.checkable else ""]))
    return sorted(out)

  def content(self) -> dict[str, Any]:
    lst, edits, checks = [], [], []
    for n in self.app_nodes():
      if n.in_list and (n.text or n.desc):
        lst.append(_norm_text(n.text or n.desc))
      if n.editable:
        edits.append(f"{n.short_rid}={_norm_text(n.text)}")
      if n.checkable:
        checks.append(f"{n.short_rid}:{_norm_text(n.text or n.desc)}={int(n.checked)}")
    return {"list": sorted(lst), "edits": sorted(edits), "checks": sorted(checks)}

  @property
  def exact_sig(self) -> str:
    return _h([self.activity, [
        [n.cls, n.rid, n.text, n.desc, list(n.bounds), n.checked, n.selected]
        for n in self.app_nodes()]])

  @property
  def struct_sig(self) -> str:
    return _h([self.activity, self.ime_visible(), self.skeleton(), self.anchor_texts()])

  @property
  def layout_sig(self) -> str:
    return _h([self.activity, self.ime_visible(), self.skeleton()])

  @property
  def content_sig(self) -> str:
    return _h(self.content())

  def summary(self) -> dict[str, Any]:
    app = self.app_nodes()
    return {
        "activity": self.activity, "package": self.package,
        "n_nodes": len(app), "n_clickable": sum(n.clickable for n in app),
        "ime": self.ime_visible(), "exact_sig": self.exact_sig,
        "struct_sig": self.struct_sig, "layout_sig": self.layout_sig,
        "content_sig": self.content_sig,
    }


_NUMERIC_RE = re.compile(r"^[\d\s.,:/$€£%+\-−]+$")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")


def _norm_text(t: str | None) -> str:
  return re.sub(r"\s+", " ", str(t or "")).strip()


def _looks_dynamic(t: str) -> bool:
  """Pure numbers, amounts and clock times outside lists (totals, dates).

  Only syntactic: it does not try to guess meaning."""
  return bool(_NUMERIC_RE.match(t)) or bool(_TIME_RE.search(t))


def parse_flat(payload: dict[str, Any], activity: str = "", screen=(1080, 2400)) -> UIState:
  raw = payload.get("nodes") or []
  nodes: list[Node] = []
  stack: list[tuple[int, int]] = []  # (depth, idx) per window
  last_window = None
  for i, r in enumerate(raw):
    w = int(r.get("windowIndex", 0))
    if w != last_window:
      stack = []
      last_window = w
    d = int(r.get("depth", 0))
    while stack and stack[-1][0] >= d:
      stack.pop()
    parent = stack[-1][1] if stack else None
    pkg = str(r.get("package") or "")
    b = r.get("bounds") or [0, 0, 0, 0]
    n = Node(
        idx=i, parent=parent, depth=d, window=w, cls=str(r.get("class") or ""), pkg=pkg,
        rid=str(r.get("resourceId") or ""), text=_norm_text(r.get("text")),
        desc=_norm_text(r.get("contentDescription")), bounds=tuple(int(x) for x in b),
        hint=_norm_text(r.get("hint")),
        clickable=bool(r.get("clickable")), long_clickable=bool(r.get("longClickable")),
        scrollable=bool(r.get("scrollable")), editable=bool(r.get("editable")),
        enabled=bool(r.get("enabled", True)), visible=bool(r.get("visible", True)),
        checkable=bool(r.get("checkable")), checked=bool(r.get("checked")),
        selected=bool(r.get("selected")), focused=bool(r.get("focused")),
        child_count=int(r.get("childCount", 0)),
        system=pkg in SYSTEM_PACKAGES, ime=pkg.startswith(IME_PACKAGES_PREFIX),
    )
    if parent is not None:
      nodes[parent].children.append(i)
      p = nodes[parent]
      n.in_list = p.in_list or p.scrollable
    nodes.append(n)
    stack.append((d, i))
  # systemui windows are chrome (status/nav bar) only when bar-sized; a
  # full-size systemui window (quick-settings shade, system dialog) is the
  # foreground UI and must be kept.
  roots: dict[int, Node] = {}
  for n in nodes:
    roots.setdefault(n.window, n)
  for w, r in roots.items():
    if r.pkg in SYSTEM_PACKAGES and (r.bounds[3] - r.bounds[1]) >= 400:
      for n in nodes:
        if n.window == w:
          n.system = False
  app_pkgs = [n.pkg for n in nodes if not n.system and not n.ime and n.pkg]
  package = max(set(app_pkgs), key=app_pkgs.count) if app_pkgs else ""
  return UIState(activity=activity, package=package, nodes=nodes, screen=screen)


# ---------------------------------------------------------------- elements
def ancestry(state: UIState, n: Node, k: int = 4) -> list[str]:
  out, cur = [], n.parent
  while cur is not None and len(out) < k:
    p = state.nodes[cur]
    out.append(f"{p.short_cls}|{p.short_rid}")
    cur = p.parent
  return out


def label_of(state: UIState, n: Node) -> str:
  if n.text or n.desc:
    return n.text or n.desc
  # first descendant text, preorder
  todo = list(n.children)
  while todo:
    c = state.nodes[todo.pop(0)]
    if c.text or c.desc:
      return c.text or c.desc
    todo = list(c.children) + todo
  return ""


def selector(state: UIState, n: Node) -> dict[str, Any]:
  return {
      "cls": n.short_cls, "rid": n.short_rid, "text": n.text, "desc": n.desc,
      "label": label_of(state, n), "ancestry": ancestry(state, n),
      "in_list": n.in_list, "bounds": list(n.bounds),
  }


def element_at(state: UIState, x: int, y: int, clickable_only: bool = True) -> Node | None:
  """Touch target at (x, y): the clickable visible app node containing the
  point that is latest in preorder (drawn on top, e.g. a FAB over a list
  row). Falls back to the smallest-area node when nothing clickable hits."""
  hits = []
  for n in state.app_nodes():
    if not n.visible or n.area <= 0:
      continue
    l, t, r, b = n.bounds
    if l <= x < r and t <= y < b:
      hits.append(n)
  if not hits:
    return None
  click = [n for n in hits if n.clickable and n.enabled]
  if click and clickable_only:
    return max(click, key=lambda n: (n.window, n.idx))
  return min(hits, key=lambda n: n.area)


@dataclass
class Slot:
  slot_id: str
  node_idx: int
  sel: dict[str, Any]
  safe: bool
  unsafe_reason: str = ""


def _in_popup_list(state: UIState, n: Node) -> bool:
  cur = n.parent
  while cur is not None:
    p = state.nodes[cur]
    if p.short_cls in ("ListView", "DropDownListView") and not p.short_rid:
      return True
    cur = p.parent
  return False


def enumerate_slots(state: UIState) -> list[Slot]:
  """Safe clickable action frontiers of a state, in deterministic order."""
  W, H = state.screen
  seen_bounds: dict[tuple, int] = {}
  slots: list[Slot] = []
  counts: dict[str, int] = {}
  cands = [n for n in state.app_nodes() if n.clickable and n.visible]
  cands.sort(key=lambda n: (n.bounds[1], n.bounds[0], -n.area))
  for n in cands:
    l, t, r, b = n.bounds
    if r - l < 8 or b - t < 8 or r <= 0 or b <= 0 or l >= W or t >= H:
      continue
    # Duplicate hit target: nested clickables with identical bounds tap the
    # same point; keep the outermost one only.
    if n.bounds in seen_bounds:
      continue
    seen_bounds[n.bounds] = n.idx
    sel = selector(state, n)
    base = "|".join([sel["cls"], sel["rid"], sel["desc"], sel["label"]])
    k = counts.get(base, 0)
    counts[base] = k + 1
    sid = base + (f"#{k}" if k else "")
    reason = ""
    if not n.enabled:
      reason = "disabled"
    elif n.checkable:
      reason = "checkable_toggle"
    elif n.editable:
      reason = "text_field"   # focus/IME only; BACK-rollback is fragile
    elif n.pkg in LAUNCHER_PACKAGES:
      reason = "launcher"
    elif UNSAFE_LABEL_RE.search(" ".join([sel["label"], sel["desc"], sel["rid"].replace("_", " ")])):
      reason = "unsafe_label"
    elif n.short_cls == "CheckedTextView" or (n.rid == "android:id/text1" and _in_popup_list(state, n)):
      reason = "value_selection"   # spinner dropdown / single-choice item commits a value
    elif NAV_UP_LABEL_RE.search(sel["label"] or sel["desc"]):
      reason = "navigate_up_item"
    elif DIALOG_BUTTON_RID_RE.match(n.short_rid):
      reason = "dialog_action_button"
    elif DISMISS_RID_RE.search(n.short_rid):
      reason = "overlay_dismiss"
    elif n.long_clickable and not n.clickable:
      reason = "long_click_only"
    slots.append(Slot(sid, n.idx, sel, safe=not reason, unsafe_reason=reason))
  return slots


# ---------------------------------------------------------------- matching
def match_element(ref_sel: dict[str, Any], live: UIState) -> tuple[str, Node | None, int]:
  """Find the live element corresponding to a recorded selector.

  Returns (category, node, n_candidates) where category is one of
  exact_selector / stable_hierarchy / structural_unique / ambiguous / none.
  No weighted similarity: each tier is an exact predicate.
  """
  cands = [n for n in live.app_nodes() if n.visible]
  def lab(n):
    return label_of(live, n)
  tiers = [
      ("exact_selector", lambda n: n.short_cls == ref_sel["cls"] and n.short_rid == ref_sel["rid"]
       and n.text == ref_sel["text"] and n.desc == ref_sel["desc"] and lab(n) == ref_sel["label"]
       and (ref_sel["rid"] or ref_sel["text"] or ref_sel["desc"] or ref_sel["label"])),
      ("stable_hierarchy", lambda n: n.short_cls == ref_sel["cls"] and n.short_rid == ref_sel["rid"]
       and ancestry(live, n) == ref_sel["ancestry"] and (lab(n) == ref_sel["label"] or not ref_sel["label"])
       and (ref_sel["rid"] or ref_sel["label"])),
      ("structural_unique", lambda n: n.short_cls == ref_sel["cls"] and n.short_rid == ref_sel["rid"]
       and ancestry(live, n) == ref_sel["ancestry"] and n.clickable == True),
  ]
  for name, pred in tiers:
    hits = [n for n in cands if pred(n)]
    if len(hits) == 1:
      return name, hits[0], 1
    if len(hits) > 1:
      if name == "structural_unique":
        return "ambiguous", None, len(hits)
      # same exact selector several times on screen -> ambiguous
      return "ambiguous", None, len(hits)
  return "none", None, 0


def diff_elements(a: UIState, b: UIState) -> dict[str, Any]:
  """A11y-level diff between two states using explicit correspondence."""
  def key(n, s):
    return (n.short_cls, n.short_rid, n.text, n.desc, tuple(ancestry(s, n)))
  an = [n for n in a.app_nodes() if n.visible]
  bn = [n for n in b.app_nodes() if n.visible]
  from collections import Counter
  ka = Counter(key(n, a) for n in an)
  kb = Counter(key(n, b) for n in bn)
  added = sum((kb - ka).values())
  removed = sum((ka - kb).values())
  # bounds drift among exact-key matches that are unique on both sides
  amap = {}
  for n in an:
    amap.setdefault(key(n, a), []).append(n)
  bmap = {}
  for n in bn:
    bmap.setdefault(key(n, b), []).append(n)
  moved, drift = 0, []
  for k, lst in amap.items():
    if len(lst) == 1 and len(bmap.get(k, [])) == 1:
      x, y = lst[0], bmap[k][0]
      if x.bounds != y.bounds:
        moved += 1
        drift.append(abs(x.center[0] - y.center[0]) + abs(x.center[1] - y.center[1]))
  # text changes: same (cls, rid, ancestry) unique on both sides, text differs
  def skey(n, s):
    return (n.short_cls, n.short_rid, tuple(ancestry(s, n)))
  sa = Counter(skey(n, a) for n in an)
  sb = Counter(skey(n, b) for n in bn)
  ta = {skey(n, a): (n.text, n.desc) for n in an if sa[skey(n, a)] == 1}
  tb = {skey(n, b): (n.text, n.desc) for n in bn if sb[skey(n, b)] == 1}
  text_changed = sum(1 for k in ta if k in tb and ta[k] != tb[k])
  rids_a = {n.short_rid for n in an if n.short_rid}
  rids_b = {n.short_rid for n in bn if n.short_rid}
  click_a = Counter((n.short_cls, n.short_rid, label_of(a, n)) for n in an if n.clickable)
  click_b = Counter((n.short_cls, n.short_rid, label_of(b, n)) for n in bn if n.clickable)
  click_struct_a = Counter((n.short_cls, n.short_rid) for n in an if n.clickable and not n.in_list)
  click_struct_b = Counter((n.short_cls, n.short_rid) for n in bn if n.clickable and not n.in_list)
  clickable_flip = 0
  for k in set(ta) & set(tb):
    pass
  return {
      "n_a": len(an), "n_b": len(bn), "added": added, "removed": removed,
      "moved": moved, "mean_drift_px": (sum(drift) / len(drift)) if drift else 0.0,
      "max_drift_px": max(drift) if drift else 0,
      "text_changed": text_changed,
      "rid_added": len(rids_b - rids_a), "rid_removed": len(rids_a - rids_b),
      "clickable_a": sum(click_a.values()), "clickable_b": sum(click_b.values()),
      "clickable_set_changed": click_a != click_b,
      "clickable_struct_changed": click_struct_a != click_struct_b,
      "struct_equal": a.struct_sig == b.struct_sig,
      "layout_equal": a.layout_sig == b.layout_sig,
      "exact_equal": a.exact_sig == b.exact_sig,
      "same_activity": a.activity == b.activity,
  }
