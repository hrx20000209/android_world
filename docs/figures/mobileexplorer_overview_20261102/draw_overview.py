#!/usr/bin/env python3
"""Revised MobileExplorer system overview (paper Fig. 5) as an editable SVG.

The figure follows one decision step i and is organised around the two questions the
advisor asked:

  top     Mix-Trajectory Reuse (S5.4)    how model reasoning is enhanced: a reuse hit that
                                         is grounded and verified skips the VLM call
  middle  Cross-task Mix-Trajectory Graph (S5.2), with labelled read / write interfaces
  bottom  on a reuse miss, the reasoning window: Novelty-Seeking Exploration (S5.3) names
          the information it scores with, the Adaptive Runtime Optimizer (S5.5) picks the
          probes and configuration, and a two-lane timeline shows probing on the live UI
          while the VLM reasons             how exploration is enhanced

Example controls, edges and bar lengths illustrate the mechanism; they are not measured
values. Shapes use presentation attributes only (no CSS, no markers) so the SVG survives
import into PowerPoint / Keynote / Illustrator / Inkscape.

Run:  python3 draw_overview.py            -> overview.svg next to this file
      node render.js                      -> overview.pdf / overview.png (Chromium)
"""
from math import hypot
from pathlib import Path
import re
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent
W, H = 1000, 470
FONT = "'Liberation Sans', Arial, Helvetica, sans-serif"

# palette: one hue per role, reused for every element of that role
INK, INK2, INK3 = '#1f2937', '#4b5563', '#6b7280'
IN_FILL, IN_STROKE = '#f4f5f7', '#a3aab5'                     # input
RU_FILL, RU_STROKE, RU_INK = '#eaf2fd', '#2f6fd0', '#1d4e9e'  # reuse (blue)
GR_FILL, GR_STROKE, GR_INK = '#f3effb', '#7b5cc8', '#5a3da3'  # graph (purple)
EX_FILL, EX_STROKE, EX_INK = '#e5f5ef', '#1e9a7f', '#127260'  # exploration (teal)
VL_FILL, VL_STROKE, VL_INK = '#fdf1dc', '#d58b22', '#94570a'  # VLM reasoning (amber)
OP_FILL, OP_STROKE, OP_INK = '#eef1f6', '#5a6a82', '#36445a'  # optimizer (slate)
RISK, OK = '#c8423a', '#2e9a57'

_TOKEN = re.compile(r'(_\{[^}]*\}|\^\{[^}]*\}|\[/?[ib]\]|\[c=#[0-9a-fA-F]{6}\]|\[/c\]|\[s=[0-9.]+\]|\[/s\])')


def rich(s, size):
  """Tiny markup -> tspans: _{sub} ^{sup} [i]..[/i] [b]..[/b] [c=#rrggbb]..[/c] [s=9]..[/s]."""
  out, italic, bold, color, fsize, shift = [], False, False, None, None, 0.0
  for p in _TOKEN.split(s):
    if not p:
      continue
    if p in ('[i]', '[/i]'):
      italic = p == '[i]'
      continue
    if p in ('[b]', '[/b]'):
      bold = p == '[b]'
      continue
    if p.startswith('[c='):
      color = p[3:-1]
      continue
    if p == '[/c]':
      color = None
      continue
    if p.startswith('[s='):
      fsize = float(p[3:-1])
      continue
    if p == '[/s]':
      fsize = None
      continue
    attrs = []
    if italic:
      attrs.append('font-style="italic"')
    if bold:
      attrs.append('font-weight="bold"')
    if color:
      attrs.append(f'fill="{color}"')
    base = fsize or size
    if p[:2] in ('_{', '^{'):
      d = round(base * 0.26, 2) if p[0] == '_' else -round(base * 0.36, 2)
      attrs.append(f'font-size="{base * 0.72:.2f}"')
      attrs.append(f'dy="{d - shift:g}"')
      shift = d
      inner, it = [], False                    # italic pieces inside an index, e.g. s_{[i]i[/i]+1}
      for q in re.split(r'(\[/?i\])', p[2:-1]):
        if q in ('[i]', '[/i]'):
          it = q == '[i]'
        elif q:
          inner.append(f'<tspan font-style="italic">{escape(q)}</tspan>' if it and not italic else escape(q))
      out.append(f'<tspan {" ".join(attrs)}>{"".join(inner)}</tspan>')
      continue
    if fsize:
      attrs.append(f'font-size="{fsize:g}"')
    if shift:
      attrs.append(f'dy="{-shift:g}"')
      shift = 0.0
    out.append(f'<tspan {" ".join(attrs)}>{escape(p)}</tspan>' if attrs else escape(p))
  return ''.join(out)


class Fig:
  def __init__(self):
    self.items = []

  def add(self, s):
    self.items.append(s)

  def group(self, name):
    self.add(f'<g id="{name}">')

  def end(self):
    self.add('</g>')

  # ---- primitives -----------------------------------------------------------------------
  def rect(self, x, y, w, h, fill='none', stroke='none', sw=1.0, rx=5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    self.add(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}"{d}/>')

  def circle(self, cx, cy, r, fill='none', stroke='none', sw=1.0):
    self.add(f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}"/>')

  def line(self, pts, stroke=INK, sw=1.2, dash=None, cap='round'):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    p = ' '.join(f'{x:g},{y:g}' for x, y in pts)
    self.add(f'<polyline points="{p}" fill="none" stroke="{stroke}" stroke-width="{sw:g}" '
             f'stroke-linecap="{cap}" stroke-linejoin="round"{d}/>')

  def path(self, d, fill='none', stroke=INK, sw=1.2, dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ''
    self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}" '
             f'stroke-linecap="round" stroke-linejoin="round"{da}/>')

  def head(self, x1, y1, x2, y2, color, size):
    """Filled triangle whose tip is (x2, y2), pointing along (x1, y1) -> (x2, y2)."""
    L = hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    bx, by = x2 - ux * size, y2 - uy * size
    hw = size * 0.48
    pts = [(x2, y2), (bx - uy * hw, by + ux * hw), (bx + uy * hw, by - ux * hw)]
    self.add('<polygon points="' + ' '.join(f'{x:.2f},{y:.2f}' for x, y in pts) + f'" fill="{color}"/>')
    return bx, by

  def arrow(self, pts, color=INK, sw=1.3, dash=None, size=6.0):
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    bx, by = self.head(x1, y1, x2, y2, color, size)
    self.line(list(pts[:-1]) + [(bx, by)], color, sw, dash, cap='butt')

  def text(self, x, y, s, size=10, weight='normal', color=INK, anchor='start', italic=False):
    st = ' font-style="italic"' if italic else ''
    wt = f' font-weight="{weight}"' if weight != 'normal' else ''
    self.add(f'<text x="{x:g}" y="{y:g}" font-family="{FONT}" font-size="{size:g}" fill="{color}" '
             f'text-anchor="{anchor}"{wt}{st}>{rich(s, size)}</text>')

  # ---- glyph-free icons (fonts differ between machines) ------------------------------------
  def check(self, x, y, s=7, color=OK, sw=1.6):
    self.path(f'M{x - s * .45:g},{y:g} L{x - s * .1:g},{y + s * .35:g} L{x + s * .5:g},{y - s * .4:g}',
              stroke=color, sw=sw)

  def cross(self, x, y, s=6, color=RISK, sw=1.6):
    a = s / 2
    self.path(f'M{x - a:g},{y - a:g} L{x + a:g},{y + a:g} M{x + a:g},{y - a:g} L{x - a:g},{y + a:g}',
              stroke=color, sw=sw)

  def tag(self, cx, cy, n, r=5.6, fill=EX_INK, color='white'):
    self.circle(cx, cy, r, fill=fill)
    self.text(cx, cy + r * 0.56, str(n), size=r * 1.45, weight='bold', color=color, anchor='middle')

  def ico_plus(self, cx, cy, r=4.2, fill='#2563eb'):
    self.circle(cx, cy, r, fill=fill)
    a = r * 0.55
    self.path(f'M{cx - a:g},{cy:g} H{cx + a:g} M{cx:g},{cy - a:g} V{cy + a:g}', stroke='white', sw=1.1)

  def ico_more(self, cx, cy, s=3.2, color=INK):
    for dy in (-s, 0, s):
      self.circle(cx, cy + dy, 0.85, fill=color)

  def ico_menu(self, cx, cy, s=3.0, color=INK):
    w = s * 1.4
    self.path(' '.join(f'M{cx - w:g},{cy + dy:g} H{cx + w:g}' for dy in (-s, 0, s)), stroke=color, sw=0.95)

  def ico_trash(self, cx, cy, s=4.0, color=INK):
    self.path(f'M{cx - s * .75:g},{cy - s * .45:g} H{cx + s * .75:g} '
              f'M{cx - s * .25:g},{cy - s * .45:g} V{cy - s * .75:g} H{cx + s * .25:g} V{cy - s * .45:g} '
              f'M{cx - s * .55:g},{cy - s * .45:g} L{cx - s * .42:g},{cy + s * .8:g} H{cx + s * .42:g} '
              f'L{cx + s * .55:g},{cy - s * .45:g}', stroke=color, sw=0.9)

  def ico_task(self, x, y, s=10, color=INK2):
    self.rect(x, y, s * .8, s, fill='white', stroke=color, sw=0.9, rx=1.2)
    self.rect(x + s * .22, y - s * .12, s * .36, s * .22, fill=color, rx=0.8)
    self.path(f'M{x + s * .18:g},{y + s * .42:g} H{x + s * .62:g} M{x + s * .18:g},{y + s * .66:g} H{x + s * .56:g}',
              stroke=color, sw=0.8)

  def pill(self, x, y, w, h, label, fill, size=9.5):
    self.rect(x, y, w, h, fill=fill, rx=h / 2)
    self.text(x + w / 2, y + h / 2 + size * 0.36, label, size=size, weight='bold', color='white', anchor='middle')

  def node(self, cx, cy, w, label, stroke=GR_STROKE, fill='white', h=17, size=9.5):
    self.rect(cx - w / 2, cy - h / 2, w, h, fill=fill, stroke=stroke, sw=1.1, rx=h / 2)
    self.text(cx, cy + size * 0.35, label, size=size, color=INK, anchor='middle')

  def svg(self):
    body = '\n'.join(self.items)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
            f'<title>MobileExplorer system overview</title>\n'
            f'<rect width="{W}" height="{H}" fill="white"/>\n{body}\n</svg>\n')


def draw():
  f = Fig()

  # =============================================================================== input
  f.group('input')
  f.rect(6, 6, 134, 244, fill=IN_FILL, stroke=IN_STROKE, sw=1.1, rx=8)
  f.text(73, 25, '[b]Input at step [i]i[/i][/b]', size=13, anchor='middle')
  # phone wireframe: one screen whose controls (1-4) are referenced by the exploration table
  px, py, pw, ph = 38, 34, 70, 138
  f.rect(px, py, pw, ph, fill='white', stroke=INK, sw=1.4, rx=10)
  f.rect(px + 4, py + 9, pw - 8, ph - 20, fill='#f9fafb', stroke='#e5e7eb', sw=0.6, rx=2)
  f.rect(px + 4, py + 9, pw - 8, 15, fill='#e8ecf2', rx=2)
  f.ico_menu(px + 11, py + 16.5, s=2.5)
  f.text(px + 20, py + 19.8, '[b]Notes[/b]', size=7.6, color=INK)
  f.ico_more(px + pw - 10, py + 16.5, s=2.7)
  for k in range(5):
    ry = py + 29 + k * 16
    f.rect(px + 7, ry, pw - 14, 13, fill='white', stroke='#d6dbe2', sw=0.6, rx=2)
    f.line([(px + 11, ry + 5), (px + 38, ry + 5)], stroke='#9aa3ae', sw=1.4)
    f.line([(px + 11, ry + 9.3), (px + 30, ry + 9.3)], stroke='#c9ced6', sw=1.1)
  f.ico_trash(px + pw - 14, py + 35.5, s=3.6)
  f.ico_plus(px + pw - 14, py + ph - 22, r=6.2)
  f.rect(px + pw / 2 - 10, py + ph - 6.5, 20, 1.7, fill='#9ca3af', rx=0.8)
  for n, (tx, ty, ax, ay) in {1: (124, py + ph - 22, px + pw - 7.5, py + ph - 22),
                              2: (124, py + 16.5, px + pw - 6, py + 16.5),
                              3: (22, py + 16.5, px + 6.5, py + 16.5),
                              4: (124, py + 35.5, px + pw - 10, py + 35.5)}.items():
    f.line([(tx, ty), (ax, ay)], stroke=EX_INK, sw=0.7, dash='1.5 1.5')
    f.tag(tx, ty, n, r=6.0)
  f.text(73, 188, '[b]Screenshot [i]s[/i]_{[i]i[/i]}[/b] + a11y tree', size=10, anchor='middle')
  f.text(73, 201, 'controls 1–4 used below', size=9, color=INK3, anchor='middle', italic=True)
  f.rect(12, 208, 122, 37, fill='white', stroke='#c9cfd8', sw=0.9, rx=5)
  f.ico_task(18, 213.5, s=9.5)
  f.text(31, 222, '[b]Task[/b] “Create a note', size=10)
  f.text(18, 237, 'named groceries”', size=10)
  f.end()

  # =============================================================================== reuse (top)
  f.group('reuse')
  f.arrow([(140, 52), (159, 52)], color=INK, sw=1.4)
  f.rect(160, 6, 834, 92, fill=RU_FILL, stroke=RU_STROKE, sw=1.2, rx=8)
  f.text(172, 25, '[b]Mix-Trajectory Reuse[/b] [s=10][c=#6b7280]§5.4 · checked before any model call[/c][/s]',
         size=13.5, color=RU_INK)
  f.pill(694, 12, 290, 19, 'Enhances reasoning: a verified reuse skips the VLM call', RU_STROKE, size=10)
  steps = [(172, 150, 'Query graph', 'state key + task goal'),
           (344, 160, 'Retrieve', 'recorded edge / procedure'),
           (526, 150, 'Ground on live UI', 'find target, bind parameters'),
           (698, 150, 'Execute & verify', '[i]s[/i]_{[i]i[/i]+1} matches the record?')]
  for k, (x, w, t, d) in enumerate(steps):
    f.rect(x, 38, w, 50, fill='white', stroke=RU_STROKE, sw=1.0, rx=5)
    f.text(x + w / 2, 58, f'[b]{t}[/b]', size=12, color=RU_INK, anchor='middle')
    f.text(x + w / 2, 76, d, size=10, color=INK2, anchor='middle')
    if k:
      px_ = steps[k - 1][0] + steps[k - 1][1]
      f.arrow([(px_, 63), (x - 1, 63)], color=RU_STROKE, sw=1.3, size=5.5)
  f.arrow([(848, 63), (986, 63)], color=RU_STROKE, sw=1.8, size=6.5)
  f.check(861, 52, s=7.5, color=RU_STROKE)
  f.text(870, 56, '[b]hit → [i]s[/i]_{[i]i[/i]+1}[/b]', size=10.5, color=RU_INK)
  f.text(917, 81, '[b]no VLM call[/b]', size=10.5, color=RU_INK, anchor='middle')
  # miss: down the left channel into the reasoning window
  f.line([(160, 82), (150, 82), (150, 266)], stroke=RISK, sw=1.4, dash='4 2.5', cap='butt')
  f.head(150, 260, 150, 273.5, RISK, 6.0)
  f.cross(58, 262, s=6)
  f.text(66, 266, '[b]miss / mismatch[/b]', size=10, color=RISK)
  f.end()

  # =============================================================================== graph (middle)
  f.group('graph')
  f.rect(160, 122, 834, 128, fill=GR_FILL, stroke=GR_STROKE, sw=1.2, rx=8)
  f.text(172, 141, '[b]Cross-task Mix-Trajectory Graph[/b] [s=10][c=#6b7280]§5.2 · '
         'shared knowledge, accumulated across tasks[/c][/s]', size=13.5, color=GR_INK)
  defs = [(162, 'Node', ['logical UI state, keyed by (app, activity,', 'layout signature); screen contents kept as observations']),
          (196, 'Edge', ['(state, action, next state) + source,', 'outcome and recovery history']),
          (230, 'Procedure', ['parameterized action sequence, e.g.', '[i]Click “+” → Type [title][/i]'])]
  for y, k, lines in defs:
    f.text(172, y, f'[b]{k}[/b]', size=10, color=GR_INK)
    for j, s in enumerate(lines):
      f.text(232, y + 13 * j, s, size=10, color=INK2)
  # schematic: pages of the example app; edges carry the control tags of the input screen
  f.rect(578, 148, 214, 96, fill='white', stroke='#c8bde6', sw=0.9, rx=6, dash='3 2')
  f.text(584, 160, 'Notes app', size=9, color=GR_INK, italic=True)
  f.node(530, 197, 60, 'Launcher')
  f.node(622, 197, 46, 'List')
  f.node(748, 168, 62, 'Folders')
  f.node(748, 226, 62, 'Editor', stroke=EX_STROKE)
  f.arrow([(560, 197), (598, 197)], color=INK, sw=1.2, size=5)
  f.arrow([(645, 192), (716, 171)], color=INK, sw=1.2, size=5)
  f.tag(664, 177.5, 3, r=5.6)
  f.text(672, 169.5, 'Task A', size=9, color=INK3)
  f.line([(645, 202), (713, 222)], stroke='#bfe8dc', sw=6.5)
  f.arrow([(645, 202), (716, 223)], color=EX_STROKE, sw=1.7, dash='4 2', size=5.5)
  f.tag(674, 219, 1, r=5.6)
  f.path('M 717,233 Q 672,250 633,206', stroke=INK3, sw=0.9, dash='2 1.8')
  f.head(640, 217, 632.5, 205.5, INK3, 4.2)
  f.text(658, 240, 'Back', size=9, color=INK3, anchor='end')
  f.check(664, 236.5, s=5.2, color=OK, sw=1.3)
  # legend
  lx = 804
  f.line([(lx, 161), (lx + 22, 161)], stroke=INK, sw=1.3)
  f.text(lx + 28, 164.5, 'executed (agent / replay)', size=9.5, color=INK2)
  f.line([(lx, 179), (lx + 22, 179)], stroke=EX_STROKE, sw=1.7, dash='4 2')
  f.text(lx + 28, 182.5, 'explored by a probe', size=9.5, color=INK2)
  f.tag(lx + 11, 197, 3, r=5.6)
  f.text(lx + 28, 200.5, 'seen in Task A → not re-probed', size=9.5, color=INK2)
  f.line([(lx, 215), (lx + 22, 215)], stroke='#bfe8dc', sw=6.5)
  f.tag(lx + 11, 215, 1, r=5.6)
  f.text(lx + 28, 218.5, 'new at step [i]i[/i] → reusable by', size=9.5, color=INK2)
  f.text(lx + 28, 231, 'later steps and tasks', size=9.5, color=INK2)
  f.end()

  # =============================================================================== read / write interfaces
  f.group('interfaces')
  f.arrow([(247, 122), (247, 89)], color=GR_STROKE, sw=1.5, size=5.5)                    # graph -> reuse
  f.text(253, 114, 'recorded edges & procedures', size=10, color=GR_INK, italic=True)
  f.arrow([(773, 89), (773, 121)], color=INK, sw=1.3, size=5.5)                          # reuse -> graph
  f.text(779, 114, 'replayed transitions', size=10, color=INK2, italic=True)
  f.arrow([(300, 250), (300, 303)], color=GR_STROKE, sw=1.5, size=5.5)                   # graph -> exploration
  f.text(306, 266, 'coverage · destinations · recovery history', size=10, color=GR_INK, italic=True)
  f.arrow([(739, 334), (739, 251)], color=EX_STROKE, sw=1.5, dash='4 2', size=5.5)       # exploration -> graph
  f.text(733, 266, 'explored transitions + recovery', size=10, color=EX_INK, italic=True, anchor='end')
  f.arrow([(897, 334), (897, 251)], color=INK, sw=1.3, size=5.5)                         # execution -> graph
  f.text(891, 266, 'executed transitions', size=10, color=INK2, italic=True, anchor='end')
  f.end()

  # =============================================================================== reasoning window (bottom)
  f.group('reasoning_window')
  f.rect(6, 274, 988, 192, fill='#fafbfc', stroke='#aeb6c2', sw=1.0, rx=8, dash='4 3')
  f.text(16, 293, '[b]On a miss: reasoning window of step [i]i[/i][/b]', size=12.5, color=INK)
  f.pill(320, 281, 341, 19, 'Enhances exploration: graph-guided probes while the VLM reasons', EX_STROKE, size=10)

  # ---- novelty-seeking exploration: which information decides a probe
  f.rect(16, 304, 332, 156, fill=EX_FILL, stroke=EX_STROKE, sw=1.1, rx=6)
  f.text(26, 321, '[b]Novelty-Seeking Exploration[/b] [s=10][c=#6b7280]§5.3 · which control to probe?[/c][/s]',
         size=12, color=EX_INK)
  info = [(GR_STROKE, 'Novelty', 'transition (state, control) not yet in the graph'),
          (EX_STROKE, 'Relevance', 'TF-IDF: task terms vs. control text & destinations'),
          (RISK, 'Recovery', 'past recovery outcomes, side-effect attributes')]
  for k, (c, t, d) in enumerate(info):
    y = 340 + k * 14
    f.rect(26, y - 7.5, 7.5, 7.5, fill=c, rx=1.5)
    f.text(37, y, f'[b]{t}[/b]', size=10, color=INK)
    f.text(94, y, d, size=10, color=INK2)
  cols = {'ctl': 26, 'nov': 140, 'rel': 172, 'rec': 246, 'dec': 283}
  ty = 389
  f.line([(24, ty - 10), (340, ty - 10)], stroke='#b9dcd1', sw=0.8)
  for key, label in (('ctl', 'control'), ('nov', 'novel'), ('rel', 'relevance'), ('rec', 'recov.'),
                     ('dec', 'decision')):
    f.text(cols[key], ty, f'[b]{label}[/b]', size=9.5, color=INK3)
  f.line([(24, ty + 4.5), (340, ty + 4.5)], stroke='#b9dcd1', sw=0.8)
  rows = [(1, 'plus', 'Add note', 'new', 0.92, True, ('probe first', EX_INK)),
          (2, 'more', 'More', 'new', 0.22, True, ('if time left', EX_INK)),
          (3, 'menu', 'Folders', 'seen', None, None, ('skip: known', INK3)),
          (4, 'trash', 'Delete', 'new', None, False, ('filter: risky', RISK))]
  for k, (n, icon, name, nov, rel, rec, (dec, dcol)) in enumerate(rows):
    y = ty + 18 + k * 15
    f.tag(cols['ctl'] + 5.5, y - 3.5, n, r=5.4)
    ix = cols['ctl'] + 19
    {'plus': lambda: f.ico_plus(ix, y - 3.5, r=4.0), 'more': lambda: f.ico_more(ix, y - 3.5),
     'menu': lambda: f.ico_menu(ix, y - 3.5), 'trash': lambda: f.ico_trash(ix, y - 3.3)}[icon]()
    f.text(cols['ctl'] + 28, y, name, size=10, color=INK)
    if nov == 'new':
      f.check(cols['nov'] + 8, y - 3.6, s=7)
    else:
      f.cross(cols['nov'] + 3, y - 3.5, s=5.2, color=INK3)
      f.text(cols['nov'] + 9.5, y, 'seen', size=9.5, color=INK3)
    if rel is None:
      f.text(cols['rel'] + 32, y, '–', size=10, color=INK3, anchor='middle')
    else:
      f.rect(cols['rel'], y - 8, 64, 7, fill='white', stroke='#b9dcd1', sw=0.6, rx=1.5)
      f.rect(cols['rel'], y - 8, 64 * rel, 7, fill=EX_STROKE, rx=1.5)
    if rec is None:
      f.text(cols['rec'] + 13, y, '–', size=10, color=INK3, anchor='middle')
    elif rec:
      f.check(cols['rec'] + 13, y - 3.6, s=7)
    else:
      f.cross(cols['rec'] + 13, y - 3.5, s=5.8)
    f.text(cols['dec'], y, f'[b]{dec}[/b]', size=9.5, color=dcol)
  f.arrow([(348, 392), (359, 392)], color=EX_STROKE, sw=1.4, size=5)
  f.text(354, 385, '[i]q[/i]_{[i]j[/i]}', size=10, color=EX_INK, anchor='middle')

  # ---- adaptive runtime optimizer: how much to explore in this window
  ox = 360
  f.rect(ox, 304, 186, 156, fill=OP_FILL, stroke=OP_STROKE, sw=1.1, rx=6)
  f.text(ox + 10, 321, '[b]Adaptive Runtime Optimizer[/b]', size=12, color=OP_INK)
  f.text(ox + 10, 335, '§5.5 · how much to explore?', size=10, color=INK3)
  f.text(ox + 10, 353, 'explore more', size=9.5, color=EX_INK, weight='bold')
  f.text(ox + 176, 353, 'infer faster', size=9.5, color=VL_INK, weight='bold', anchor='end')
  f.line([(ox + 14, 361), (ox + 172, 361)], stroke='#c3cad6', sw=3)
  f.line([(ox + 14, 361), (ox + 88, 361)], stroke=EX_STROKE, sw=3)
  f.line([(ox + 98, 361), (ox + 172, 361)], stroke=VL_STROKE, sw=3)
  f.circle(ox + 93, 361, 4.8, fill='white', stroke=OP_INK, sw=1.4)
  f.text(ox + 93, 383, 'max Σ [i]q[/i]_{[i]j[/i]} [i]x[/i]_{[i]j[/i]}', size=12, color=INK, anchor='middle')
  f.text(ox + 93, 400, 's.t. Σ [i]c[/i]_{[i]j[/i]}([i]r[/i]) [i]x[/i]_{[i]j[/i]} ≤ [i]B[/i]([i]r[/i])',
         size=12, color=INK, anchor='middle')
  for k, s in enumerate(['[i]c[/i]_{[i]j[/i]}([i]r[/i]): profiled probe + recovery cost',
                         '[i]B[/i]([i]r[/i]): reasoning window of config [i]r[/i]',
                         'latency([i]r[/i]) ≤ (1+δ) × baseline']):
    f.text(ox + 10, 420 + k * 13, s, size=9.5, color=INK2)

  # ---- timeline: one live UI shared by probes and the agent, one VLM
  lx0 = 566
  f.rect(lx0, 304, 420, 156, fill='white', stroke='#d1d6de', sw=0.9, rx=6)
  x0, xr = 630, 822                       # reasoning starts / action fields ready
  f.arrow([(546, 346), (lx0 + 7, 346)], color=OP_STROKE, sw=1.3, size=4.5)
  f.text(556, 341, '[i]x[/i]_{[i]j[/i]}', size=10, color=OP_INK, anchor='middle')
  f.arrow([(546, 381), (lx0 + 7, 381)], color=OP_STROKE, sw=1.3, size=4.5)
  f.text(556, 376, '[i]r[/i]', size=10, color=OP_INK, anchor='middle')
  f.text(lx0 + 10, 350, '[b]Live UI[/b]', size=10, color=EX_INK)
  f.text(lx0 + 10, 385, '[b]VLM[/b]', size=10, color=VL_INK)
  f.line([(x0, 327), (x0, 323), (xr, 323), (xr, 327)], stroke=VL_STROKE, sw=0.9, cap='butt')
  f.text(683, 319, 'window [i]B[/i]([i]r[/i])', size=9.5, color=VL_INK, anchor='middle')
  lane = [(x0, 46, 1), (x0 + 47, 38, None), (x0 + 86, 46, 2), (x0 + 133, 38, None)]
  for x, w, n in lane:
    if n:
      f.rect(x, 334, w, 24, fill=EX_STROKE, rx=3)
      f.text(x + 18, 349.5, '[b]probe[/b]', size=9.5, color='white', anchor='middle')
      f.tag(x + 38, 346, n, r=5.2, fill='white', color=EX_INK)
    else:
      f.rect(x, 334, w, 24, fill='#c4ebdf', rx=3)
      f.text(x + w / 2, 349.5, 'restore', size=9.5, color=EX_INK, anchor='middle')
  f.rect(xr + 2, 334, 46, 24, fill=OP_FILL, stroke=OP_STROKE, sw=0.8, rx=3)
  f.text(xr + 25, 349.5, 'check [i]a[/i]_{[i]i[/i]}', size=9.5, color=OP_INK, anchor='middle')
  f.rect(xr + 49, 334, 56, 24, fill=RU_FILL, stroke=RU_STROKE, sw=0.8, rx=3)
  f.text(xr + 77, 349.5, 'execute [i]a[/i]_{[i]i[/i]}', size=9.5, color=RU_INK, anchor='middle')
  f.arrow([(xr + 105, 346), (xr + 138, 346)], color=INK, sw=1.3, size=5)
  f.text(xr + 142, 350, '[i]s[/i]_{[i]i[/i]+1}', size=11, color=INK)
  f.rect(x0, 369, xr - x0, 24, fill=VL_FILL, stroke=VL_STROKE, sw=0.9, rx=3)
  f.text((x0 + xr) / 2, 384.5, 'reasoning on captured [i]s[/i]_{[i]i[/i]} + task', size=9.5, color=VL_INK,
         anchor='middle')
  f.rect(xr, 369, 78, 24, fill='white', stroke=VL_STROKE, sw=0.9, rx=3, dash='3 2')
  f.text(xr + 39, 384.5, 'summary tail', size=9.5, color=VL_INK, anchor='middle')
  f.line([(xr, 328), (xr, 401)], stroke=VL_STROKE, sw=1.0, dash='3 2', cap='butt')
  f.arrow([(x0, 401), (978, 401)], color=INK3, sw=0.9, size=4.5)
  f.text(xr, 413, '[i]a[/i]_{[i]i[/i]} ready (streamed)', size=9.5, color=VL_INK, anchor='middle')
  f.text(978, 413, 'time', size=9, color=INK3, anchor='end')
  notes = ['• new probes stop when [i]a[/i]_{[i]i[/i]} is ready or recovery time runs short',
           '• UI is restored, then [i]a[/i]_{[i]i[/i]} is checked on it (mismatch → re-observe)',
           '• [i]a[/i]_{[i]i[/i]} executes while the summary is still decoding']
  for k, s in enumerate(notes):
    f.text(lx0 + 10, 429 + k * 13, s, size=9.5, color=INK2)
  f.end()
  return f


def main():
  f = draw()
  (OUT / 'overview.svg').write_text(f.svg())
  print('wrote', OUT / 'overview.svg')


if __name__ == '__main__':
  main()
