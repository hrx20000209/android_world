#!/usr/bin/env python3
"""Fig. 6 (Cross-task Mix-Trajectory Graph), redrawn to show how observations merge and what the mix yields.

Follows the FULL45 implementation:
  * observations are append-only transitions (runtime_graph_study/scripts/rgs_run.py: Graph); every record keeps
    origin (agent / explore), task instance, source / destination summaries, capture ids, tap position, outcome,
    rollback result; the node and edge index is rebuilt from them, nothing is overwritten.
  * state key (rgs_ui.py: UIState.struct_sig): activity, IME visibility, the multiset of (class, resource-id,
    clickable / editable / scrollable / checkable) of nodes outside list content, and anchor texts (titles, button
    labels; numbers, amounts and times excluded). List rows, field contents and checked states form content_sig,
    used to verify rollback, not to merge. Routing (joint_graph.py) uses the coarser layout key without titles.
  * action key (common.py: family_of_sel): class | resource-id | desc | label; a list row is class | resource-id
    (its text is runtime data); raw coordinates are never part of the identity.
Example pages, tasks and hashes are illustrative.

Run:  python3 draw_fig6.py && node ../mobileexplorer_overview_20261102/render.js fig6.svg
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fig7", HERE.parent / "mobileexplorer_fig7_20261102" / "draw_fig7.py")
fig7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig7)
base, v2 = fig7.base, fig7.v2
Fig = base.Fig
base.W, base.H = 1000, 404
back_icon = v2.back_icon

TXT, TXT2, GRAY = "#111111", "#3a3a3a", "#8a8f98"
BLUE, BLUE_FILL = "#2f7fe0", "#dbeafb"
NEW, NEW_FILL = "#e8710f", "#ffe8d2"
OK, RED = "#2e9a57", "#d9363e"
GR_FILL, GR_STROKE = "#e3f3d9", "#a9d08f"
KEY = "#1d5fbf"


# ----------------------------------------------------------------------------- small pieces
def chip(f, x, y, sid, name, w=56, h=26, stroke="#4b5563", fill="white"):
  f.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.2, rx=6)
  f.text(x + 6, y + 17, f"[b]{sid}[/b]", size=10, color=TXT)
  f.text(x + 22, y + 17, name, size=9, color=TXT2)


def harrow(f, x1, x2, y, label, dash=None, color="#222"):
  f.arrow([(x1, y), (x2, y)], color=color, sw=1.6, dash=dash, size=6)
  f.text((x1 + x2) / 2, y - 6, label, size=9, color=color if color != "#222" else TXT, anchor="middle")


def mini_list(f, x, y, w, h, rows, outline=True, mark_new=None):
  """Notes list wireframe; rows = list of colours (one per note). Controls outside the list are outlined in blue
  (they form the state key); the list region is shaded grey (its rows are content)."""
  f.rect(x, y, w, h, fill="white", stroke="#222", sw=1.3, rx=w * 0.13)
  ix, iy, iw, ih = x + 3, y + 3, w - 6, h - 6
  f.rect(ix, iy, iw, ih, fill="#fafbfc", rx=w * 0.1)
  by = iy + 11
  # top bar: menu, title, search, more
  f.path(f"M{ix + 4:g},{by - 2.5:g} H{ix + 10:g} M{ix + 4:g},{by:g} H{ix + 10:g} M{ix + 4:g},{by + 2.5:g} H{ix + 10:g}",
         stroke="#222", sw=1.0)
  f.text(ix + 13, by + 2.6, "[b]Notes[/b]", size=6.6, color=TXT)
  f.circle(ix + iw - 14, by - 0.5, 2.8, stroke="#222", sw=0.9)
  f.path(f"M{ix + iw - 12:g},{by + 1.5:g} L{ix + iw - 10.2:g},{by + 3.3:g}", stroke="#222", sw=0.9)
  for k in (-2.5, 0, 2.5):
    f.circle(ix + iw - 4.5, by + k, 0.8, fill="#222")
  # list rows
  top = by + 9
  rh = 13.5
  f.rect(ix + 2, top - 2, iw - 4, len(rows) * rh + 2, fill="#eceef1" if outline else "none", rx=2)
  for k, c in enumerate(rows):
    ry = top + k * rh
    f.rect(ix + 4, ry, iw - 8, rh - 2.5, fill="white", stroke="#dfe3e8", sw=0.5, rx=1.5)
    f.rect(ix + 6, ry + 2, 7, 7, fill=c, rx=1.2)
    f.rect(ix + 16, ry + 2.4, iw * 0.42, 1.6, fill="#9aa3ae", rx=0.7)
    f.rect(ix + 16, ry + 6, iw * 0.28, 1.3, fill="#c9ced6", rx=0.6)
    if mark_new == k:
      f.rect(ix + 4, ry, iw - 8, rh - 2.5, fill="none", stroke=NEW, sw=1.0, rx=1.5)
  # floating add button
  fx, fy = ix + iw - 9, iy + ih - 10
  f.circle(fx, fy, 5.6, fill="#2563eb")
  f.path(f"M{fx - 2.6:g},{fy:g} H{fx + 2.6:g} M{fx:g},{fy - 2.6:g} V{fy + 2.6:g}", stroke="white", sw=1.1)
  if outline:
    for (cx, cy, r) in ((ix + 7, by, 5.2), (ix + iw - 13.5, by, 5.0), (ix + iw - 4.5, by, 4.2), (fx, fy, 7.4)):
      f.rect(cx - r, cy - r, 2 * r, 2 * r, fill="none", stroke=KEY, sw=1.1, rx=2, dash="2 1.4")


def mini_editor(f, x, y, w=34, h=60):
  f.rect(x, y, w, h, fill="white", stroke="#222", sw=1.1, rx=5)
  ix, iy, iw = x + 2, y + 2, w - 4
  f.rect(ix, iy, iw, h - 4, fill="#fafbfc", rx=4)
  f.rect(ix + 3, iy + 7, iw - 6, 6, fill="white", stroke="#b8c0cb", sw=0.5, rx=1)
  f.rect(ix + 3, iy + 16, iw - 6, 10, fill="white", stroke="#b8c0cb", sw=0.5, rx=1)
  f.check(ix + iw - 5, iy + 3.4, s=4, color="#222", sw=0.9)
  f.rect(ix, iy + h - 24, iw, 20, fill="#dfe3e8", rx=1.5)
  for r in range(3):
    for c in range(5):
      f.rect(ix + 1.5 + c * (iw - 3) / 5, iy + h - 22.5 + r * 6, (iw - 3) / 5 - 1, 4.4, fill="white", rx=0.6)


def mini_drawer(f, x, y, w=34, h=60):
  f.rect(x, y, w, h, fill="white", stroke="#222", sw=1.1, rx=5)
  ix, iy, iw = x + 2, y + 2, w - 4
  f.rect(ix, iy, iw, h - 4, fill="#c9ccd2", rx=4)
  f.rect(ix, iy, iw * 0.68, h - 4, fill="white", rx=4)
  f.rect(ix, iy, iw * 0.68, 9, fill="#e8eefb", rx=3)
  for k in range(4):
    f.rect(ix + 3, iy + 14 + k * 8, iw * 0.4, 1.8, fill="#6b7280", rx=0.8)


def key_chip(f, cx, y, h_txt, color=KEY, fill="#e6effc"):
  w = 44
  f.rect(cx - w / 2, y, w, 15, fill=fill, stroke=color, sw=1.0, rx=7.5)
  f.text(cx, y + 11, f"[b]{h_txt}[/b]", size=9, color=color, anchor="middle")


def node_chip(f, cx, cy, name, sub=None, w=70, h=26, stroke="#4b5563", fill="white"):
  f.rect(cx - w / 2, cy - h / 2, w, h, fill=fill, stroke=stroke, sw=1.4, rx=8)
  if sub:
    f.text(cx, cy + 0.5, f"[b]{name}[/b]", size=11, color=TXT, anchor="middle")
    f.text(cx, cy + 10, sub, size=7.6, color=TXT2, anchor="middle", italic=True)
  else:
    f.text(cx, cy + 4, f"[b]{name}[/b]", size=11, color=TXT, anchor="middle")


def edge(f, p, q, color="#222", dash=None, sw=1.6, rp=(35, 13), rq=(35, 13)):
  """Arrow between two node chips (axis-aligned boxes of half-size rp / rq)."""
  (x1, y1), (x2, y2) = p, q
  dx, dy = x2 - x1, y2 - y1

  def clip(hw, hh, sx, sy):
    tx = hw / abs(dx) if dx else 1e9
    ty = hh / abs(dy) if dy else 1e9
    t = min(tx, ty)
    return sx * t, sy * t
  ax, ay = clip(rp[0] + 2, rp[1] + 2, dx, dy)
  bx, by = clip(rq[0] + 3, rq[1] + 3, dx, dy)
  f.arrow([(x1 + ax, y1 + ay), (x2 - bx, y2 - by)], color=color, sw=sw, dash=dash, size=6.5)


# ----------------------------------------------------------------------------- figure
def draw():
  f = Fig()

  # =============================================================== ① trajectories (inputs)
  f.text(14, 22, "[b]① Trajectories[/b]", size=15, color=TXT)
  f.text(14, 38, "two tasks and one probe, in the same app", size=10, color=TXT2, italic=True)
  f.text(14, 58, "[b]Task 1[/b] (model): create note “shopping”", size=10, color=TXT)
  chip(f, 14, 66, "S1", "List")
  harrow(f, 72, 132, 79, "A1 Menu")
  chip(f, 134, 66, "S2", "Drawer")
  harrow(f, 192, 252, 79, "A2 New note")
  chip(f, 254, 66, "S3", "Editor")
  f.text(296, 106, "then A3 Type “shopping”, Save", size=8.8, color=TXT2, anchor="end", italic=True)
  f.text(14, 128, "[b]Task 2[/b] (model): delete note “Old Notes”", size=10, color=TXT)
  chip(f, 14, 136, "S4", "List")
  harrow(f, 72, 132, 149, "A4 Old Notes")
  chip(f, 134, 136, "S5", "Note")
  f.text(14, 190, "[b]Exploration[/b] during Task 2’s reasoning", size=10, color=TXT)
  chip(f, 14, 198, "S4", "List")
  harrow(f, 72, 132, 211, "A5 Add Note", dash="4.5 3", color=NEW)
  chip(f, 134, 198, "S6", "Editor", stroke=NEW)
  f.path("M 158,224 Q 116,248 50,226", stroke=GRAY, sw=1.1, dash="2.5 2")
  f.head(60, 229.5, 49, 225.5, GRAY, 5)
  f.text(100, 248, "Back", size=9, color=GRAY, anchor="middle")
  f.check(122, 244.5, s=7, color=OK, sw=1.6)
  f.text(128, 248, "rollback verified", size=9, color=OK)
  # legend (as in the original figure)
  f.rect(204, 190, 104, 40, fill="white", stroke="#29b6f6", sw=1.4, rx=4)
  f.arrow([(211, 204), (237, 204)], color="#222", sw=1.4, size=5)
  f.text(242, 207.5, "Reasoning", size=9.5, color=TXT)
  f.arrow([(211, 220), (237, 220)], color="#222", sw=1.4, dash="3.5 2.5", size=5)
  f.text(242, 223.5, "Exploration", size=9.5, color=TXT)

  # =============================================================== ② merge actions
  f.text(14, 278, "[b]② Merge actions[/b]", size=15, color=TXT)
  f.text(14, 294, "key = the control, not the tap position", size=10, color=TXT2, italic=True)
  rows = [("A5  Click (950, 2050)", "[b]Add Note[/b] button · fab_add"),
          ("A3  Type “shopping”", "Type [b][c=#e8710f][title][/c][/b] · title field"),
          ("A4  Tap “Old Notes”", "Tap [b][c=#e8710f][note][/c][/b] · note row")]
  for k, (a, b) in enumerate(rows):
    y = 302 + k * 28
    f.rect(14, y, 120, 22, fill="#eceef1", stroke="#c3c8cf", sw=0.9, rx=11)
    f.text(74, y + 15, a, size=9.5, color=TXT2, anchor="middle")
    f.arrow([(138, y + 11), (154, y + 11)], color="#222", sw=1.4, size=5)
    f.rect(158, y, 148, 22, fill=BLUE_FILL, stroke=BLUE, sw=1.0, rx=11)
    f.text(232, y + 15, b, size=9.5, color=TXT, anchor="middle")
  f.text(14, 395, "same control in another task or a probe → same edge", size=9, color=TXT2, italic=True)

  # =============================================================== ③ merge states
  f.line([(320, 12), (320, 396)], stroke="#e1e4ea", sw=1.0)
  f.text(332, 22, "[b]③ Merge states[/b]", size=15, color=TXT)
  f.text(332, 38, "same structure → one state; content kept per visit", size=10, color=TXT2, italic=True)
  f.text(372, 58, "[b]S1[/b] · Task 1", size=9.5, color=TXT, anchor="middle")
  f.text(452, 58, "[b]S4[/b] · Task 2", size=9.5, color=TXT, anchor="middle")
  mini_list(f, 340, 64, 64, 118, ["#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6"])
  mini_list(f, 420, 64, 64, 118, ["#10b981", "#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6"], mark_new=0)
  f.text(452, 192, "new row: “shopping”", size=8, color=NEW, anchor="middle", italic=True)
  key_chip(f, 372, 198, "#a3f9")
  key_chip(f, 452, 198, "#a3f9")
  f.text(412, 209, "[b]=[/b]", size=12, color=KEY, anchor="middle")
  f.line([(372, 213), (372, 222), (452, 222), (452, 213)], stroke=KEY, sw=1.2, cap="butt")
  f.arrow([(412, 222), (412, 232)], color=KEY, sw=1.3, size=5)
  node_chip(f, 412, 246, "List", "2 visits", w=74, stroke=KEY)
  # what is and is not in the key
  f.text(500, 74, "[b][c=#1d5fbf]in the key[/c][/b]", size=10, color=TXT)
  for k, s in enumerate(("app + activity", "controls outside lists:", "  class · id · flags", "titles, button labels")):
    f.text(500, 89 + k * 13, s, size=9.3, color=TXT2)
  f.rect(498, 136, 9, 9, fill="none", stroke=KEY, sw=1.1, rx=2, dash="2 1.4")
  f.text(511, 144, "outlined", size=9, color=KEY, italic=True)
  f.text(500, 164, "[b][c=#6b7280]not in the key[/c][/b]", size=10, color=TXT)
  for k, s in enumerate(("list rows (number, text)", "field text, checkboxes", "numbers, dates, clock")):
    f.text(500, 179 + k * 13, s, size=9.3, color=TXT2)
  f.rect(498, 213, 9, 9, fill="#eceef1", rx=2)
  f.text(511, 221, "shaded: kept per visit", size=9, color=GRAY, italic=True)
  # two more cases: merged although reached differently; not merged although same activity
  f.line([(332, 270), (636, 270)], stroke="#e1e4ea", sw=1.0)
  f.text(332, 288, "[b]reached by different actions[/b]", size=10, color=TXT)
  mini_editor(f, 336, 296)
  mini_editor(f, 380, 296)
  f.text(353, 368, "S3 · T1", size=8.6, color=TXT2, anchor="middle")
  f.text(397, 368, "S6 · probe", size=8.6, color=NEW, anchor="middle")
  key_chip(f, 353, 374, "#5e1c")
  key_chip(f, 397, 374, "#5e1c")
  f.arrow([(422, 326), (440, 326)], color=KEY, sw=1.3, size=5)
  node_chip(f, 474, 326, "Editor", "2 visits", w=62, stroke=KEY)
  f.text(512, 284, "[b]not merged[/b]", size=10, color=TXT)
  f.text(512, 296, "same activity, other controls", size=8.4, color=TXT2, italic=True)
  mini_drawer(f, 516, 302)
  f.text(533, 372, "S2 · T1", size=8.6, color=TXT2, anchor="middle")
  key_chip(f, 533, 378, "#7c21", color=RED, fill="#fde2e3")
  f.arrow([(556, 332), (574, 332)], color=RED, sw=1.3, size=5)
  node_chip(f, 606, 332, "Drawer", "own state", w=58, stroke=RED)

  # =============================================================== ④ mix across tasks and sources
  f.rect(648, 8, 344, 390, fill=GR_FILL, stroke=GR_STROKE, sw=1.2, rx=12)
  f.text(660, 30, "[b]④ Cross-task Mix-Trajectory Graph[/b]", size=14, color=TXT)
  P = {"List": (700, 132), "Drawer": (814, 72), "Editor": (930, 132), "Note": (814, 196)}
  # new-task route: glow under the explored edge
  f.line([(738, 132), (894, 132)], stroke="#f9cfa6", sw=9)
  edge(f, P["List"], P["Drawer"])
  edge(f, P["Drawer"], P["Editor"])
  edge(f, P["List"], P["Note"])
  edge(f, P["List"], P["Editor"], color=NEW, dash="5 3", sw=1.9)
  node_chip(f, *P["List"], "List", "S1 = S4")
  node_chip(f, *P["Drawer"], "Drawer", "S2")
  node_chip(f, *P["Editor"], "Editor", "S3 = S6")
  node_chip(f, *P["Note"], "Note", "S5")
  f.text(742, 96, "Menu · T1", size=9, color=TXT, anchor="middle")
  f.text(868, 92, "New note · T1", size=9, color=TXT)
  f.text(814, 124, "[b]Add Note · probe (T2)[/b]", size=9, color=NEW, anchor="middle")
  f.text(756, 180, "Old Notes · T2", size=9, color=TXT, anchor="end")
  # procedure recorded on the editor from task 1
  f.rect(888, 152, 92, 34, fill="white", stroke="#9aa3ae", sw=1.0, rx=5)
  f.text(934, 166, "[b]procedure · T1[/b]", size=8.8, color=TXT, anchor="middle")
  f.text(934, 179, "Type [title] → Save", size=8.8, color=TXT2, anchor="middle")
  f.rect(886, 150, 96, 38, fill="none", stroke="#f9cfa6", sw=2.4, rx=6)
  # the route a new task gets from the mix
  f.rect(660, 226, 320, 54, fill="white", stroke=NEW, sw=1.2, rx=7)
  f.text(670, 243, "[b]New task:[/b] add note “groceries”", size=10.5, color=TXT)
  f.text(670, 258, "List ⇢ Editor via the probe’s edge (T2), then T1’s procedure", size=9.5, color=TXT2)
  f.text(670, 272, "[b][c=#e8710f]a route that no single trajectory contains[/c][/b]", size=9.5, color=TXT)
  # what a node and an edge keep
  f.rect(660, 290, 320, 98, fill="white", stroke="#cfd5df", sw=1.0, rx=7)
  f.line([(818, 298), (818, 380)], stroke="#cfd5df", sw=1.0, dash="3 2")
  f.text(670, 307, "[b]state List[/b]", size=10, color=TXT)
  for k, s in enumerate(("key  Notes · Main · #a3f9", "visit T1: 4 notes, screen", "visit T2: 5 notes, screen",
                         "contents stay per visit")):
    f.text(670, 323 + k * 14, s, size=9, color=TXT2 if k < 3 else GRAY, italic=k == 3)
  f.text(828, 307, "[b]edge Add Note[/b]", size=10, color=TXT)
  for k, s in enumerate(("List → Editor, seen 1×", "origin  probe · T2", "rollback  verified", "tap  (950, 2050)")):
    f.text(828, 323 + k * 14, s, size=9, color=TXT2)
  return f


def main():
  f = draw()
  (HERE / "fig6.svg").write_text(f.svg())
  print("wrote", HERE / "fig6.svg")


if __name__ == "__main__":
  main()
