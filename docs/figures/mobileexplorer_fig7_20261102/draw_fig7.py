#!/usr/bin/env python3
"""Fig. 7 (Novelty-Seeking UI Exploration), redrawn to show which information decides a probe.

Follows the FULL45 implementation (full_run.py / full9.py / explore_value.py):
  (1) Novelty   page_family_index: a control whose (app, activity, control) transition the graph already holds,
                from the agent or from exploration in any task, is not probed.
  (2) Risk      _risky_slot / effectful / P_rev: mode-switch and destructive labels, toggles, ads leaving the app,
                and controls whose recovery history (graph) predicts that Back will not undo them.
  (3) Relevance TF-IDF between the task's operation terms (parameters removed) and the control's evidence: its own
                text and the pages that controls with the same label led to elsewhere in the graph; IDF over the
                screens of all apps, so general controls weigh little.
  (4) Probe     the top candidate is probed during reasoning, rolled back, checked, and the edge with its rollback
                outcome is recorded in the graph.
Example controls and scores are illustrative.

Run:  python3 draw_fig7.py && node ../mobileexplorer_overview_20261102/render.js fig7.svg
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASEDIR = HERE.parent / "mobileexplorer_overview_20261102"


def _load(name, file):
  spec = importlib.util.spec_from_file_location(name, BASEDIR / file)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


base = _load("base", "draw_overview.py")
v2 = _load("v2", "draw_overview_v2.py")
Fig = base.Fig
base.W, base.H = 1000, 432
back_icon, chevron = v2.back_icon, v2.chevron

TXT, TXT2, GRAY = "#111111", "#3a3a3a", "#8a8f98"
BLUE, BLUE_FILL = "#2f7fe0", "#dbeafb"
RED, RED_FILL = "#d9363e", "#fde2e3"
NEW, NEW_FILL = "#e8710f", "#ffe8d2"
SEEN, SEEN_FILL = "#8a8f98", "#eceef1"
OK = "#2e9a57"
GR_FILL, GR_STROKE = "#dee3f1", "#aeb8d6"
BAR = "#6cbf4f"


def pill(f, x, y, w, label, fill, stroke, color=TXT, h=19, size=11, strike=False):
  f.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.2, rx=h / 2)
  f.text(x + w / 2, y + h / 2 + size * 0.36, f"[b]{label}[/b]", size=size, color=color, anchor="middle")
  if strike:
    f.line([(x + 8, y + h / 2), (x + w - 8, y + h / 2)], stroke=color, sw=1.2)


def notes_screen(f, x, y, w=108, h=248):
  """Wireframe of the notes list in the original figure; returns anchors of its controls."""
  f.rect(x, y, w, h, fill="white", stroke="#222", sw=1.6, rx=14)
  ix, iy, iw = x + 4, y + 4, w - 8
  f.rect(ix, iy, iw, h - 8, fill="#fafbfc", rx=11)
  f.text(ix + 8, iy + 12, "20:34", size=6.5, color=TXT2)
  f.circle(x + w / 2, iy + 9, 2.6, fill="#222")
  ab = iy + 28
  f.path(f"M{ix + 7:g},{ab - 3:g} H{ix + 15:g} M{ix + 7:g},{ab:g} H{ix + 15:g} M{ix + 7:g},{ab + 3:g} H{ix + 15:g}",
         stroke="#222", sw=1.1)
  f.text(ix + 21, ab + 3.5, "[b]Notes[/b]", size=8, color=TXT)
  f.circle(ix + iw - 22, ab, 4.4, stroke="#222", sw=1.1)
  f.path(f"M{ix + iw - 19:g},{ab + 3:g} L{ix + iw - 16.5:g},{ab + 5.5:g}", stroke="#222", sw=1.1)
  for k in (-3, 0, 3):
    f.circle(ix + iw - 8, ab + k, 0.9, fill="#222")
  f.rect(ix + 5, ab + 10, iw - 10, 11, fill="#eef0f4", rx=5.5)
  f.text(ix + 12, ab + 18, "Search notes", size=5.6, color=GRAY)
  rows = [("#3b82f6", "Meeting Notes"), ("#ef4444", "Shopping List"), ("#f59e0b", "Travel Plan"),
          ("ad", "Plan Your Next Vacation"), ("#8b5cf6", "Old Notes")]
  ry = ab + 27
  anchors = {"menu": (ix + 11, ab), "search": (ix + iw - 22, ab), "setting": (ix + iw - 8, ab)}
  for color, title in rows:
    rh = 25
    f.rect(ix + 4, ry, iw - 8, rh, fill="white", stroke="#e3e6eb", sw=0.6, rx=3)
    if color == "ad":
      f.rect(ix + 4, ry, iw - 8, rh, fill="none", stroke=RED, sw=1.2, rx=3)
      f.rect(ix + 8, ry + 4, 15, 17, fill="#7cc4e8", rx=2)
      f.rect(ix + 8, ry + 14, 15, 7, fill="#e6c27a", rx=1)
      f.rect(ix + 27, ry + 14, 7, 5, fill="#2e9a57", rx=1)
      anchors["ad"] = (ix + 4, ry + rh / 2)
    else:
      f.rect(ix + 8, ry + 5, 13, 15, fill=color, rx=2.5)
    f.text(ix + 27, ry + 10, f"[b]{title}[/b]", size=5.4, color=TXT)
    f.rect(ix + 27, ry + 14 if color != "ad" else ry + 20, iw - 46, 1.6, fill="#c9ced6", rx=0.8)
    ry += rh + 3
  # delete button on the last row, add button, bottom navigation
  dx, dy = ix + iw - 24, ry - 24
  f.rect(dx, dy, 17, 17, fill="#e5484d", rx=3)
  f.path(f"M{dx + 4.5:g},{dy + 5.5:g} H{dx + 12.5:g} M{dx + 6:g},{dy + 5.5:g} L{dx + 6.8:g},{dy + 13:g} "
         f"H{dx + 10.2:g} L{dx + 11:g},{dy + 5.5:g} M{dx + 7.3:g},{dy + 5.5:g} V{dy + 4:g} H{dx + 9.7:g} V{dy + 5.5:g}",
         stroke="white", sw=0.9)
  anchors["delete"] = (dx + 17, dy + 8.5)
  fx, fy = ix + iw - 16, y + h - 46
  f.circle(fx, fy, 10, fill="#2563eb")
  f.path(f"M{fx - 4.5:g},{fy:g} H{fx + 4.5:g} M{fx:g},{fy - 4.5:g} V{fy + 4.5:g}", stroke="white", sw=1.6)
  anchors["add"] = (fx + 10, fy)
  ny = y + h - 22
  f.line([(ix, ny - 6), (ix + iw, ny - 6)], stroke="#e3e6eb", sw=0.8)
  f.rect(ix + 20, ny - 2, 9, 8, fill="#2563eb", rx=1.5)
  f.text(ix + 24.5, ny + 13, "Notes", size=5.4, color="#2563eb", anchor="middle")
  f.rect(ix + iw - 34, ny - 2, 9, 8, fill="none", stroke=GRAY, sw=1.0, rx=1.5)
  f.text(ix + iw - 29.5, ny + 13, "Calendar", size=5.4, color=GRAY, anchor="middle")
  anchors["calendar"] = (ix + iw - 29.5, ny + 2)
  return anchors


def tag(f, x, y, label, target, color=BLUE, size=10):
  w = len(label) * 6.1 + 14
  f.line([(x + w / 2, y + 9), target], stroke=color, sw=1.3)
  f.circle(*target, 2.2, fill=color)
  f.rect(x, y, w, 18, fill=color, rx=9)
  f.text(x + w / 2, y + 12.6, f"[b]{label}[/b]", size=size, color="white", anchor="middle")


def editor_screen(f, x, y, w=70, h=120):
  f.rect(x, y, w, h, fill="white", stroke="#222", sw=1.4, rx=9)
  ix, iy, iw = x + 3, y + 3, w - 6
  f.rect(ix, iy, iw, h - 6, fill="#fafbfc", rx=7)
  f.text(ix + 6, iy + 16, "[b]New note[/b]", size=6.5, color=TXT)
  for k, lab in enumerate(("Title", "Body")):
    f.rect(ix + 5, iy + 24 + k * 18, iw - 10, 12, fill="white", stroke="#b8c0cb", sw=0.7, rx=2)
    f.text(ix + 9, iy + 32.5 + k * 18, lab, size=5.6, color=GRAY)
  f.rect(ix, iy + h - 46, iw, 40, fill="#dfe3e8", rx=2)
  for r in range(3):
    for c in range(6):
      f.rect(ix + 3 + c * (iw - 6) / 6, iy + h - 42 + r * 11, (iw - 6) / 6 - 2, 8, fill="white", rx=1)


def bar(f, x, y, w, frac, label):
  f.rect(x, y, w, 13, fill="white", stroke=BAR, sw=1.6, rx=1.5)
  f.rect(x + 1.5, y + 1.5, max(0, (w - 3) * frac), 10, fill=BAR, rx=1)
  f.text(x + w + 6, y + 10.5, f"[b]{label}[/b]", size=11, color=TXT)


def draw():
  f = Fig()
  # ---------------------------------------------------------------- column headers
  heads = [(122, "State [i]i[/i] and task", None), (318, "① Novelty", "transition already in graph?"),
           (478, "② Risk", "can Back undo it?"), (680, "③ Relevance", "TF-IDF with the task"),
           (900, "④ Probe & record", "during model reasoning")]
  for x, h, sub in heads:
    f.text(x, 22, f"[b]{h}[/b]", size=15, color=TXT, anchor="middle")
    if sub:
      f.text(x, 38, sub, size=10.5, color=TXT2, anchor="middle", italic=True)

  # ---------------------------------------------------------------- (A) state i + task
  f.rect(14, 48, 216, 34, fill="#fff7e0", stroke="#e3c46b", sw=1.1, rx=6)
  f.text(22, 62, '[b]Task:[/b] [b][c=#0f766e]Add[/c][/b] a [b][c=#0f766e]note[/c][/b] named [i][c=#8a8f98]“groceries”[/c][/i]',
         size=11, color=TXT)
  f.text(22, 76, "terms: add, note (parameters removed)", size=9, color=TXT2, italic=True)
  A = notes_screen(f, 70, 92)
  tag(f, 14, 104, "Menu", A["menu"])
  tag(f, 120, 90, "Search", A["search"])
  tag(f, 184, 108, "Setting", A["setting"])
  tag(f, 10, 214, "Ad", A["ad"], color=BLUE)
  tag(f, 186, 236, "Delete", A["delete"])
  tag(f, 180, 278, "Add Note", A["add"])
  tag(f, 156, 346, "Calendar", A["calendar"])
  f.add('<polygon points="232,200 244,200 244,194 254,206 244,218 244,212 232,212" fill="#111"/>')

  # ---------------------------------------------------------------- (1) novelty against the graph
  items = ["Menu", "Search", "Setting", "Ad", "Delete", "Add Note", "Calendar"]
  y0, dy = 58, 37
  for k, it in enumerate(items):
    y = y0 + k * dy
    seen = it == "Menu"
    pill(f, 262, y, 92, it, SEEN_FILL if seen else BLUE_FILL, SEEN if seen else BLUE,
         color=SEEN if seen else TXT, strike=seen)
    if seen:
      f.text(359, y + 13.5, "[b]seen[/b]", size=10, color=SEEN)
      f.text(308, y + 31, "Task A opened it here", size=8.5, color=SEEN, anchor="middle", italic=True)
    else:
      f.rect(359, y + 2, 31, 15, fill=NEW_FILL, stroke=NEW, sw=1.0, rx=7.5)
      f.text(374.5, y + 13, "[b]new[/b]", size=9.5, color=NEW, anchor="middle")
  f.add('<polygon points="396,200 404,200 404,195 412,206 404,217 404,212 396,212" fill="#111"/>')

  # ---------------------------------------------------------------- (2) risk: attributes + recovery history
  risk = [("Search", "mode switch"), ("Setting", "mode switch"), ("Ad", "leaves the app"), ("Delete", "destructive"),
          ("Add Note", "empty editor, Back undoes"), ("Calendar", "tab, Back undoes")]
  for k, (it, why) in enumerate(risk):
    y = y0 + (k + 1) * dy
    bad = k < 4
    pill(f, 424, y, 92, it, RED_FILL if bad else BLUE_FILL, RED if bad else BLUE, color=RED if bad else TXT, strike=bad)
    if bad:
      f.cross(526, y + 9.5, s=7)
    else:
      f.check(526, y + 9.5, s=9)
    f.text(470, y + 31, why, size=8.5, color=RED if bad else OK, anchor="middle", italic=True)
  f.add('<polygon points="548,206 556,206 556,201 564,212 556,223 556,218 548,218" fill="#111"/>')

  # ---------------------------------------------------------------- (3) relevance: task terms x evidence
  f.rect(572, 52, 216, 26, fill="#fff7e0", stroke="#e3c46b", sw=1.0, rx=6)
  f.text(680, 69, "[b]query:[/b] task terms  [b][c=#0f766e]add · note[/c][/b]", size=11, color=TXT, anchor="middle")
  cards = [(96, "Add Note", ["own text: [b][c=#0f766e]add · note[/c][/b]",
                             "graph: same label in App 3", "→ editor (title, body)"], 0.83, "0.83"),
           (206, "Calendar", ["own text: calendar", "graph: same label in App 5", "→ month view"], 0.05, "0.05")]
  for y, it, ev, frac, lab in cards:
    f.rect(572, y, 216, 96, fill="white", stroke="#cfd5df", sw=1.0, rx=6)
    pill(f, 582, y + 8, 88, it, BLUE_FILL, BLUE)
    for j, line in enumerate(ev):
      f.text(582, y + 42 + j * 13, line, size=9.5, color=TXT2)
    bar(f, 680, y + 11, 66, frac, lab)
  f.text(680, 322, "IDF over screens of all apps:", size=9.5, color=TXT2, anchor="middle", italic=True)
  f.text(680, 335, "menu / search / settings appear everywhere → low weight", size=9.5, color=TXT2, anchor="middle",
         italic=True)
  f.add('<polygon points="792,138 800,138 800,133 808,144 800,155 800,150 792,150" fill="#111"/>')

  # ---------------------------------------------------------------- (4) probe, roll back, check, record
  f.text(850, 66, "[b]probe[/b]", size=11.5, color=NEW, anchor="middle")
  for k in range(3):
    chevron(f, 820 + k * 20, 76, w=19, h=16, fill="#fde2cc", notch=6)
  editor_screen(f, 886, 54)
  f.text(921, 188, "[b]new state[/b]", size=10, color=TXT, anchor="middle")
  back_icon(f, 850, 132, r=11, color=TXT, sw=2.2)
  f.text(850, 162, "[b][i]Back[/i][/b]", size=12, color=RED, anchor="middle")
  f.check(830, 200, s=11, color=OK, sw=2.4)
  f.text(842, 204, "[b]restored[/b]: matches State [i]i[/i]", size=10, color=OK)
  f.rect(816, 222, 176, 58, fill="white", stroke="#cfd5df", sw=1.0, rx=6)
  f.text(904, 238, "[b]record in graph[/b]", size=10.5, color=TXT, anchor="middle")
  f.text(904, 254, "List —Add Note→ Editor", size=10, color=TXT2, anchor="middle")
  f.text(904, 269, "origin: explore · rollback ✓", size=9.5, color=TXT2, anchor="middle", italic=True)
  f.text(816, 300, "new probes stop when the", size=9, color=TXT2, italic=True)
  f.text(816, 312, "model's action is ready", size=9, color=TXT2, italic=True)

  # ---------------------------------------------------------------- graph: the information each step reads
  f.rect(250, 362, 742, 64, fill=GR_FILL, stroke=GR_STROKE, sw=1.2, rx=10)
  f.text(768, 386, "[b]Mix-Trajectory Graph[/b]", size=13.5, color=TXT)
  f.text(768, 402, "execution + exploration,", size=9.5, color=TXT2, italic=True)
  f.text(768, 415, "all earlier tasks", size=9.5, color=TXT2, italic=True)
  links = [(318, "transitions recorded", "on this page"), (470, "recovery outcomes of", "similar controls"),
           (680, "pages reached by", "same-label controls")]
  for x, l1, l2 in links:
    f.arrow([(x, 362), (x, 345)], color="#3c4a7a", sw=1.8, size=6)
  for x, l1, l2 in links:
    f.text(x, 388, f"[b]{l1}[/b]", size=10, color="#3c4a7a", anchor="middle")
    f.text(x, 402, f"[b]{l2}[/b]", size=10, color="#3c4a7a", anchor="middle")
  f.arrow([(962, 281), (962, 361)], color=NEW, sw=1.8, dash="4 2.5", size=6)
  f.text(954, 334, "[b]new edge +[/b]", size=9.5, color=NEW, anchor="end")
  f.text(954, 347, "[b]rollback result[/b]", size=9.5, color=NEW, anchor="end")
  return f


def main():
  f = draw()
  (HERE / "fig7.svg").write_text(f.svg())
  print("wrote", HERE / "fig7.svg")


if __name__ == "__main__":
  main()
