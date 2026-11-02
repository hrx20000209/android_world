#!/usr/bin/env python3
"""Fig. 8 (Mix-Trajectory Reuse), redrawn to show what is reused, where it comes from, and what it saves.

Follows Sec. 5.4 and the FULL45 reuse paths (bootstrap launch, form routing / procedures, graph_reuse):
  (1) Retrieve  query = state key (app, activity, layout) + task goal; the graph returns a route that mixes an edge
                found by exploration (List -Add Note-> Editor, Fig. 7) and a procedure recorded from a model
                execution in an earlier task (type [title] into the editor).
  (2) Ground    bind the task's parameter, find each target on the live screen, stop before a commit.
  (3) Execute   execute and verify each landing against the record; on a mismatch return to the model.
                Benefit: the model makes 1 decision (Save, the commit) instead of 3.
Example pages and tasks are illustrative.

Run:  python3 draw_fig8.py && node ../mobileexplorer_overview_20261102/render.js fig8.svg
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fig7", HERE.parent / "mobileexplorer_fig7_20261102" / "draw_fig7.py")
fig7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig7)
base, v2 = fig7.base, fig7.v2
Fig = base.Fig
base.W, base.H = 1000, 400

TXT, TXT2, GRAY = fig7.TXT, fig7.TXT2, fig7.GRAY
BLUE, BLUE_FILL, RED, NEW, OK = fig7.BLUE, fig7.BLUE_FILL, fig7.RED, fig7.NEW, fig7.OK
REASON_FILL, REASON_EDGE = "#f6edb5", "#d9b93c"
EXEC_FILL, EXEC_EDGE = "#d6ecfb", "#3b8fd6"
GR_FILL, GR_STROKE = fig7.GR_FILL, fig7.GR_STROKE


def page_node(f, cx, cy, label, sub=None, w=74, h=34, stroke="#5b6b8c"):
  f.rect(cx - w / 2, cy - h / 2, w, h, fill="white", stroke=stroke, sw=1.4, rx=7)
  f.text(cx, cy + (0 if sub else 4.5), f"[b]{label}[/b]", size=11.5, color=TXT, anchor="middle")
  if sub:
    f.text(cx, cy + 12, sub, size=8.5, color=TXT2, anchor="middle", italic=True)


def edge(f, x1, x2, y, label, source, dash=None, color="#222", commit=False):
  f.arrow([(x1, y), (x2, y)], color=color, sw=2.0, dash=dash, size=7)
  mid = (x1 + x2) / 2
  f.text(mid, y - 9, f"[b]{label}[/b]", size=10, color=RED if commit else TXT, anchor="middle")
  f.text(mid, y + 17, source, size=8.8, color=color if not commit else RED, anchor="middle", italic=True)


def block(f, x, y, w, label, kind):
  fill, edge_ = (REASON_FILL, REASON_EDGE) if kind == "reason" else (EXEC_FILL, EXEC_EDGE)
  f.rect(x, y, w, 22, fill=fill, stroke=edge_, sw=1.2, rx=4)
  f.text(x + w / 2 - (5 if kind == "exec" else 0), y + 15, label.strip(), size=9.5, color=TXT, anchor="middle")


def draw():
  f = Fig()
  heads = [(105, "State [i] and task", None), (370, "① Retrieve", "query: state key + task goal"),
           (648, "② Bind & ground", "on the live UI"), (885, "③ Execute & verify", "model calls saved")]
  for x, h, sub in heads:
    f.text(x, 22, f"[b]{h}[/b]".replace("[i]", "[i]i[/i]"), size=15, color=TXT, anchor="middle")
    if sub:
      f.text(x, 38, sub, size=10.5, color=TXT2, anchor="middle", italic=True)

  # ---------------------------------------------------------------- (A) current state + task (Task B)
  f.rect(10, 48, 192, 46, fill="#fff7e0", stroke="#e3c46b", sw=1.1, rx=6)
  f.text(18, 64, '[b]Task B:[/b] [b][c=#0f766e]Add[/c][/b] a [b][c=#0f766e]note[/c][/b] named', size=11, color=TXT)
  f.text(18, 79, '“groceries”', size=11, color=TXT)
  f.text(18, 90, "parameter title = groceries", size=8.5, color=TXT2, italic=True)
  A = fig7.notes_screen(f, 50, 104, w=108, h=232)
  fig7.tag(f, 112, 300, "Add Note", A["add"])
  f.text(102, 356, "state key:", size=9.5, color=TXT2, anchor="middle", italic=True)
  f.text(102, 370, "[b](Notes, NotesList, layout)[/b]", size=9.5, color=TXT2, anchor="middle")
  f.add('<polygon points="206,200 218,200 218,194 228,206 218,218 218,212 206,212" fill="#111"/>')

  # ---------------------------------------------------------------- (1) retrieved mixed trajectory
  f.rect(232, 52, 278, 250, fill=GR_FILL, stroke=GR_STROKE, sw=1.2, rx=10)
  f.text(371, 72, "[b]Mix-Trajectory Graph[/b]", size=12.5, color=TXT, anchor="middle")
  f.text(371, 88, "route toward the goal from this state", size=9.5, color=TXT2, anchor="middle", italic=True)
  # route laid out in two rows so each edge keeps a readable label
  page_node(f, 282, 130, "List", "State i")
  page_node(f, 456, 130, "Editor", "empty")
  edge(f, 320, 418, 130, "Add Note", "explored · Task A", dash="5 3", color=NEW)
  page_node(f, 456, 226, "Editor", "title filled")
  f.arrow([(456, 148), (456, 207)], color="#222", sw=2.0, size=7)
  f.text(448, 176, "[b]Type [title][/b]", size=10, color=TXT, anchor="end")
  f.text(448, 190, "model · Task A", size=8.8, color="#222", italic=True, anchor="end")
  page_node(f, 282, 226, "List", "+ new note")
  f.arrow([(418, 226), (321, 226)], color=RED, sw=2.0, size=7)
  f.text(370, 217, "[b]Save[/b]", size=10, color=RED, anchor="middle")
  f.text(370, 243, "commit", size=8.8, color=RED, anchor="middle", italic=True)
  # legend
  f.line([(246, 274), (270, 274)], stroke=NEW, sw=2.0, dash="5 3")
  f.text(276, 278, "from exploration", size=9.5, color=TXT2)
  f.line([(372, 274), (396, 274)], stroke="#222", sw=2.0)
  f.text(402, 278, "from model execution", size=9.5, color=TXT2)
  f.text(371, 294, "one route, two sources: a mixed trajectory", size=9.5, color=TXT, anchor="middle", italic=True)
  f.add('<polygon points="514,176 524,176 524,170 534,182 524,194 524,188 514,188" fill="#111"/>')

  # ---------------------------------------------------------------- (2) bind parameters and ground targets
  rows = [("[title] ← “groceries”", "from the task", OK, True),
          ("“Add Note” found", "on State i", OK, True),
          ("title field found", "after landing on Editor", OK, True),
          ("“Save” is a commit", "left to the model", RED, False)]
  for k, (a, b, c, good) in enumerate(rows):
    y = 64 + k * 58
    f.rect(540, y, 216, 46, fill="white", stroke="#cfd5df", sw=1.0, rx=6)
    if good:
      f.check(556, y + 23, s=11, color=OK, sw=2.2)
    else:
      f.cross(556, y + 23, s=9)
    f.text(570, y + 20, f"[b]{a}[/b]", size=11, color=TXT)
    f.text(570, y + 36, b, size=9.5, color=c if not good else TXT2, italic=True)
  f.add('<polygon points="760,176 770,176 770,170 780,182 770,194 770,188 760,188" fill="#111"/>')

  # ---------------------------------------------------------------- (3) execute + verify; what it saves
  f.text(788, 70, "[b]Model only[/b]", size=10.5, color=TXT)
  block(f, 788, 78, 62, "reason", "reason")
  block(f, 853, 78, 62, "reason", "reason")
  block(f, 918, 78, 62, "reason", "reason")
  for k, lab in enumerate(("Add Note", "type", "Save")):
    f.text(819 + 65 * k, 115, lab, size=9, color=TXT2, anchor="middle", italic=True)
  f.text(788, 144, "[b]With reuse[/b]", size=10.5, color=TXT)
  block(f, 788, 152, 62, "execute ", "exec")
  block(f, 853, 152, 62, "execute ", "exec")
  block(f, 918, 152, 62, "reason", "reason")
  for k, lab in enumerate(("Add Note", "type", "Save")):
    f.text(819 + 65 * k, 189, lab, size=9, color=TXT2, anchor="middle", italic=True)
  for k in range(2):
    f.check(841 + 65 * k, 163, s=7, color=OK, sw=1.6)
  f.rect(788, 204, 192, 30, fill="#e8f5e9", stroke="#9fd3a6", sw=1.0, rx=6)
  f.text(884, 224, "[b]3 → 1 model calls[/b]", size=12.5, color=OK, anchor="middle")
  f.text(788, 256, "[b]verify[/b] each landing against the record", size=9.5, color=TXT2)
  f.text(788, 271, "[b]mismatch[/b] → stop, the model reasons", size=9.5, color=RED)
  f.text(788, 292, "executed steps join the history;", size=9.5, color=TXT2, italic=True)
  f.text(788, 305, "the model continues from the new state", size=9.5, color=TXT2, italic=True)

  # ---------------------------------------------------------------- bottom line: the two enhancements
  f.rect(232, 318, 758, 70, fill="#f7f8fb", stroke="#d5dae3", sw=1.0, rx=8)
  f.text(246, 342, "[b][c=#e8710f]Exploration → reuse:[/c][/b] the editor was first reached by a probe (Fig. 7), so a route to it exists",
         size=11, color=TXT)
  f.text(246, 362, "[b][c=#2f7fe0]Reuse → reasoning:[/c][/b] steps the graph already knows run without a model call;",
         size=11, color=TXT)
  f.text(246, 378, "the model keeps the decisions that need judgment (commits, unknown pages)", size=11, color=TXT)
  return f


def main():
  f = draw()
  (HERE / "fig8.svg").write_text(f.svg())
  print("wrote", HERE / "fig8.svg")


if __name__ == "__main__":
  main()
