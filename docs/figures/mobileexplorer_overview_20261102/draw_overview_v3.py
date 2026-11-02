#!/usr/bin/env python3
"""MobileExplorer system overview (Fig. 5), revised on top of the authors' original layout.

Kept from the original: panel arrangement (Input | Adaptive Runtime Optimizer | Model Reasoning +
Novelty-Seeking Parallel Exploration | Mix-Trajectory Reuse, Cross-task graph below), colours and
visual vocabulary (chips, chevrons, reasoning banner, red latency line, element tags, fan of scores).

Changed:
  * Exploration panel: elements of State i are tagged as seen (grey) or never seen (orange) in the
    graph; each candidate carries its seen count n; the fan scores are Novelty x Task Relevance with
    Novelty = 1 / (1 + n); an arrow from the graph supplies n. Highest score (Add) is probed, then Back.
  * Reuse panel: State i+1 + task -> retrieve a mixed (reasoning + exploration) trajectory ->
    verify -> execute it, skipping the model's reasoning for those steps.
  * Graph panel: no screenshots; trajectories as nodes, one merged graph with the unseen frontier and
    the reused path marked.
  * Model / UI labels in the optimizer were swapped; step index unified to i; Mix-Trajectory naming.

Screens are vector wireframes standing in for the original screenshots.
Run:  python3 draw_overview_v3.py && node render.js overview_v3.svg
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, file):
  spec = importlib.util.spec_from_file_location(name, HERE / file)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


base = _load('base', 'draw_overview.py')
v2 = _load('v2', 'draw_overview_v2.py')
Fig, FONT = base.Fig, base.FONT
base.W, base.H = 1000, 410
vlm_icon, phone_icon, chip, chevron, arrow_bar, back_icon = (v2.vlm_icon, v2.phone_icon, v2.chip, v2.chevron,
                                                          v2.arrow_bar, v2.back_icon)

TXT, TXT2, GRAY = '#111111', '#3a3a3a', '#7d8590'
IN_FILL, IN_STROKE = '#fadcdd', '#e9a8ad'          # original input pink
MOD_FILL, MOD_STROKE = '#dee3f1', '#c3cbe3'        # original module lavender
GR_FILL, GR_STROKE = '#e3f3d9', '#b8dca4'          # original graph green
CHEV = '#fde2cc'
RED, AMBER, GREEN = '#d62f2f', '#e0a100', '#7cb342'
NEW, SEEN = '#ec6a1e', '#8a93a0'                   # never seen in graph / already seen
BLUE = '#2f6fd0'


# ----------------------------------------------------------------------------- pieces
def big_arrow(f, x0, y, x1, h=9, head=11):
  """Thick black arrow as in the original figure."""
  t = h / 2
  pts = [(x0, y - t * 0.55), (x1 - head, y - t * 0.55), (x1 - head, y - t * 1.25), (x1, y),
         (x1 - head, y + t * 1.25), (x1 - head, y + t * 0.55), (x0, y + t * 0.55)]
  f.add('<polygon points="' + ' '.join(f'{a:g},{b:g}' for a, b in pts) + f'" fill="{TXT}"/>')


def state_screen(f, x, y, w, h):
  """Wireframe of State i: search bar, settings button, contact list, Add button.
  Returns anchor points of the three controls."""
  f.rect(x, y, w, h, fill='white', stroke='#222', sw=1.2, rx=w * 0.12)
  ix, iy, iw, ih = x + 2.6, y + 2.6, w - 5.2, h - 5.2
  f.rect(ix, iy, iw, ih, fill='#fbfbfc', rx=w * 0.09)
  f.circle(x + w / 2, y + 5, 1.3, fill='#222')
  sb = (ix + 3, iy + 7, iw * 0.66, h * 0.065)
  f.rect(*sb, fill='#e7eefc', stroke='#9fb6ef', sw=0.6, rx=sb[3] / 2)
  f.circle(sb[0] + 3.4, sb[1] + sb[3] / 2, 1.4, stroke='#4a63c8', sw=0.6)
  gx, gy = ix + iw - 5, sb[1] + sb[3] / 2
  f.circle(gx, gy, 2.6, fill='#fff', stroke='#555', sw=0.8)
  f.circle(gx, gy, 0.9, fill='#555')
  colors = ['#f6a04d', '#ef6c9a', '#b07ad8', '#f6a04d', '#4db6ac', '#ef6c9a', '#b07ad8', '#f6a04d']
  top = sb[1] + sb[3] + 4
  step = (iy + ih - 12 - top) / len(colors)
  for k, c in enumerate(colors):
    ry = top + k * step
    f.circle(ix + 5, ry + step * 0.4, min(2.4, step * 0.32), fill=c)
    f.rect(ix + 9.5, ry + step * 0.2, iw * 0.5, 1.4, fill='#a6adb8', rx=0.6)
    f.rect(ix + 9.5, ry + step * 0.2 + 2.6, iw * 0.32, 1.1, fill='#cdd2d9', rx=0.5)
  ab = (ix + iw - 17, iy + ih - 10, 15, 7)
  f.rect(*ab, fill='#e7eefc', stroke='#4a63c8', sw=0.7, rx=3.5)
  f.path(f'M{ab[0] + 3:g},{ab[1] + 3.5:g} H{ab[0] + 6.4:g} M{ab[0] + 4.7:g},{ab[1] + 1.8:g} V{ab[1] + 5.2:g}',
         stroke='#4a63c8', sw=0.7)
  f.rect(ab[0] + 8, ab[1] + 2.9, 5, 1.2, fill='#4a63c8', rx=0.5)
  return {'search': (sb[0] + sb[2] / 2, sb[1] + sb[3] / 2), 'setting': (gx, gy),
          'add': (ab[0] + ab[2] / 2, ab[1] + ab[3] / 2)}


def tag(f, x, y, label, color, target=None, size=8.5, star=False):
  """Callout tag (as on the original screenshots): rounded label with a pointer to `target`."""
  w = {'Search': 28.4, 'Setting': 28.8, 'Add': 16.5}.get(label, len(label) * 5) + 9
  h = 12
  if target:
    tx, ty = target
    f.line([(x + w / 2, y + h / 2), (tx, ty)], stroke=color, sw=1.2)
    f.circle(tx, ty, 1.8, fill=color)
  f.rect(x, y, w, h, fill=color, rx=4)
  f.text(x + w / 2, y + 8.6, f'[b]{label}[/b]', size=size, color='white', anchor='middle')
  if star:
    cx, cy, r1, r2 = x + w + 5, y + 2, 4.6, 2.0
    from math import cos, sin, pi
    pts = []
    for k in range(10):
      r = r1 if k % 2 == 0 else r2
      a = -pi / 2 + k * pi / 5
      pts.append((cx + r * cos(a), cy + r * sin(a)))
    f.add('<polygon points="' + ' '.join(f'{a:.2f},{b:.2f}' for a, b in pts) + '" fill="#f7b500"/>')


def button(f, cx, cy, label, fill='white', stroke='#cfd3da', w=58, h=21, color=TXT):
  f.rect(cx - w / 2 + 1.2, cy - h / 2 + 1.6, w, h, fill='#b9bfca', rx=4)       # shadow
  f.rect(cx - w / 2, cy - h / 2, w, h, fill=fill, stroke=stroke, sw=1.0, rx=4)
  f.text(cx, cy + 4, f'[b]{label}[/b]', size=11.5, color=color, anchor='middle')


def node(f, cx, cy, r=7.5, fill='white', stroke='#4b5563', dash=None):
  f.add(f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" fill="{fill}" stroke="{stroke}" stroke-width="1.1"'
        + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')
  f.rect(cx - r * 0.36, cy - r * 0.55, r * 0.72, r * 1.1, fill='white', stroke='#6b7280', sw=0.7, rx=1)


def edge(f, p, q, dash=None, color='#1f1f1f', r=8.3, sw=1.3):
  (x1, y1), (x2, y2) = p, q
  L = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
  ux, uy = (x2 - x1) / L, (y2 - y1) / L
  f.arrow([(x1 + ux * r, y1 + uy * r), (x2 - ux * r, y2 - uy * r)], color=color, sw=sw, dash=dash, size=5)


def cursor(f, x, y, s=1.0, color=TXT):
  """Mouse-pointer glyph with click marks (the action is a click)."""
  pts = [(0, 0), (0, 12.5), (3.2, 9.6), (5.6, 14.6), (7.8, 13.6), (5.4, 8.8), (9.4, 8.8)]
  f.add('<polygon points="' + ' '.join(f'{x + a * s:.2f},{y + b * s:.2f}' for a, b in pts) +
        f'" fill="white" stroke="{color}" stroke-width="1.1" stroke-linejoin="round"/>')
  for dx, dy, ex, ey in ((-1.5, -1.5, -4.2, -4.2), (2.2, -2.4, 2.8, -6), (-2.4, 2.2, -6, 2.8)):
    f.path(f'M{x + dx * s:g},{y + dy * s:g} L{x + ex * s:g},{y + ey * s:g}', stroke=color, sw=1.0)


def tap_hand(f, cx, cy, s=1.0, color=TXT):
  """Simple pointing-hand glyph (index finger + palm) with tap marks."""
  p = (f'M{cx - 2.2 * s:g},{cy + 9 * s:g} V{cy - 1 * s:g} A{1.6 * s:g},{1.6 * s:g} 0 0 1 {cx + 1 * s:g},{cy - 1 * s:g} '
       f'V{cy + 3 * s:g} H{cx + 6.2 * s:g} A{1.6 * s:g},{1.6 * s:g} 0 0 1 {cx + 7.6 * s:g},{cy + 4.8 * s:g} '
       f'V{cy + 9.5 * s:g} Q{cx + 7 * s:g},{cy + 13 * s:g} {cx + 3 * s:g},{cy + 13 * s:g} H{cx + 1 * s:g} '
       f'Q{cx - 2.2 * s:g},{cy + 12.5 * s:g} {cx - 2.2 * s:g},{cy + 9 * s:g} Z')
  f.path(p, fill='white', stroke=color, sw=1.0)
  for dx, dy, ex, ey in ((-0.6, -4.2, -0.6, -6.6), (-4.2, -2.8, -6, -4.4), (3, -2.8, 4.8, -4.4)):
    f.path(f'M{cx + dx * s:g},{cy + dy * s:g} L{cx + ex * s:g},{cy + ey * s:g}', stroke=color, sw=0.9)


def app_grid(f, x, y, s=12, color=TXT):
  g = s * 0.42
  for k in range(4):
    gx, gy = x + (k % 2) * (s - g), y + (k // 2) * (s - g)
    if k == 3:
      f.circle(gx + g / 2, gy + g / 2, g / 2, stroke=color, sw=1.0)
    else:
      f.rect(gx, gy, g, g, stroke=color, sw=1.0, rx=1.2)


def type_glyph(f, x, y, color=TXT):
  f.text(x, y, '[b]Aa[/b]', size=10, color=color)
  f.path(f'M{x + 3:g},{y + 2.5:g} H{x + 13:g}', stroke=color, sw=0.9)


# ----------------------------------------------------------------------------- figure
def draw():
  f = Fig()
  f.add('<defs>'
        '<linearGradient id="reasonGrad" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#e3f3d8"/><stop offset="1" stop-color="#f1e6a2"/></linearGradient>'
        '<linearGradient id="vlmGrad" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#8b7cf6"/><stop offset="1" stop-color="#5b4fd0"/></linearGradient>'
        '</defs>')

  # ======================================================================= input
  f.group('input')
  f.rect(4, 4, 130, 402, fill=IN_FILL, stroke=IN_STROKE, sw=1.2, rx=6)
  f.text(69, 24, '[b]Input at Step [i]i[/i][/b]', size=13, color=TXT, anchor='middle')
  state_screen(f, 30, 34, 78, 138)
  f.text(69, 188, '[b]Screenshot[/b]', size=11.5, color=TXT, anchor='middle')
  f.rect(12, 196, 114, 92, fill='none', stroke='#333', sw=0.9, rx=2, dash='3 2')
  for k, (kind, val) in enumerate([('button 1', 'Search'), ('button 2', 'Setting'), ('button 3', 'Add'),
                                   ('list 4', 'Contact A'), ('list 5', 'Contact B'), ('text 6', 'Recent')]):
    f.text(18, 210 + k * 13.2, f'{kind}: [c={RED}]{val}[/c]', size=9.5, color=TXT)
  f.text(69, 304, '[b]UI Elements[/b]', size=11.5, color=TXT, anchor='middle')
  f.rect(49, 318, 40, 46, fill='white', stroke='#222', sw=1.3, rx=4)
  f.rect(61, 313, 16, 8, fill='#222', rx=2)
  for k in range(3):
    yy = 333 + k * 10
    f.rect(55, yy - 3, 6, 6, fill='white', stroke='#222', sw=1.0, rx=1)
    f.rect(65, yy - 1, 18, 2.2, fill='#222', rx=1)
  f.text(69, 386, '[b]Task Description[/b]', size=11.5, color=TXT, anchor='middle')
  f.end()
  big_arrow(f, 134, 100, 157)

  # ======================================================================= adaptive runtime optimizer
  f.group('optimizer')
  f.rect(158, 4, 208, 208, fill=MOD_FILL, stroke=MOD_STROKE, sw=1.0, rx=12)
  f.text(262, 25, '[b]Adaptive Runtime Optimizer[/b]', size=13, color=TXT, anchor='middle')
  phone_icon(f, 171, 37, w=19, h=33)
  f.text(180.5, 84, '[b]UI[/b]', size=11.5, color=TXT, anchor='middle')
  chip(f, 204, 38, s=22)
  f.rect(234, 44, 60, 10, fill='white', stroke='#67a83c', sw=1.2, rx=1.5)
  f.rect(235, 45, 18, 8, fill='#cfe8bd', rx=1)
  for k in range(3):
    chevron(f, 204 + k * 40, 66, w=38, h=20, fill=CHEV, notch=8)
  f.text(214, 102, '[b]Exploration[/b]', size=11.5, color=TXT)
  vlm_icon(f, 168, 113, s=26)
  f.text(181, 152, '[b]Model[/b]', size=11.5, color=TXT, anchor='middle')
  chip(f, 204, 113, s=22)
  f.rect(234, 119, 60, 10, fill='white', stroke='#67a83c', sw=1.2, rx=1.5)
  f.rect(235, 120, 54, 8, fill='#5a9632', rx=1)
  arrow_bar(f, 204, 334, 141, 20, 'url(#reasonGrad)', tip=10)
  f.text(265, 155.5, '[b]Reasoning[/b]', size=11.5, color=TXT, anchor='middle')
  f.line([(339, 62), (339, 166)], stroke=RED, sw=1.8, dash='4.5 3', cap='butt')
  f.text(262, 192, f'[b]Max [c={RED}]Exploration[/c] s.t. [c={RED}]Reasoning Latency[/c][/b]', size=10.5,
         color=TXT, anchor='middle')
  f.end()
  big_arrow(f, 367, 30, 391)
  big_arrow(f, 367, 138, 391)

  # ======================================================================= model reasoning banner
  f.group('reasoning')
  arrow_bar(f, 392, 806, 4, 50, 'url(#reasonGrad)', tip=26)
  vlm_icon(f, 402, 14, s=30)
  f.text(588, 22, '[b]Model Reasoning for Step [i]i[/i][/b]', size=13.5, color=TXT, anchor='middle')
  f.rect(442, 29, 300, 19, fill='white', stroke='#222', sw=0.9, rx=1)
  f.text(448, 43, '[b]Action:[/b] Click (90, 1800) ......', size=11, color=TXT)
  f.end()

  # ======================================================================= novelty-seeking parallel exploration
  f.group('exploration')
  f.rect(392, 62, 404, 150, fill=MOD_FILL, stroke=MOD_STROKE, sw=1.0, rx=12)
  f.text(594, 80, '[b]Novelty-Seeking Parallel Exploration[/b]', size=13.5, color=TXT, anchor='middle')
  anchors = state_screen(f, 428, 95, 56, 101)
  tag(f, 398, 86, 'Search', SEEN, target=anchors['search'])
  tag(f, 470, 86, 'Setting', SEEN, target=anchors['setting'])
  tag(f, 448, 182, 'Add', NEW, target=anchors['add'], star=True)
  f.text(447, 207, '[b]State [i]i[/i][/b]', size=11.5, color=TXT, anchor='middle')
  # fan of candidate scores (as in the original), now decomposed below
  rows = [(110, 'Add', '0.95', RED, 0, True), (146, 'Search', '0.42', AMBER, 1, False),
          (182, 'Setting', '0.17', GREEN, 3, False)]
  f.line([(485, 146), (512, 146)], stroke=TXT, sw=1.6, cap='butt')
  f.line([(512, 110), (512, 182)], stroke=TXT, sw=1.6, cap='butt')
  for y, name, score, col, n, chosen in rows:
    f.arrow([(512, y), (583, y)], color=TXT, sw=1.6, size=6)
    f.text(548, y - 5, f'[b]{score}[/b]', size=13, color=col, anchor='middle')
    if chosen:
      button(f, 614, y, name, fill='#ffd8b8', stroke=NEW)
    else:
      button(f, 614, y, name)
    f.text(651, y + 4, f'[b][i]n[/i] = {n}[/b]', size=10.5, color=NEW if n == 0 else SEEN)
  # probe the most novel candidate, then roll back
  f.arrow([(690, 110), (744, 110), (744, 123)], color=TXT, sw=1.6, size=6)
  f.text(716, 104, '[b][i]probe[/i][/b]', size=10, color=NEW, anchor='middle')
  back_icon(f, 744, 134, r=8, color=TXT, sw=1.8)
  f.text(757, 157, '[b][i]Back[/i][/b]', size=12, color=RED, anchor='middle')
  f.text(651, 204, f'[b]Score = [c={RED}]Novelty[/c] × Task Relevance,  [c={RED}]Novelty[/c] = 1 / (1 + [i]n[/i])[/b]',
         size=10.5, color=TXT, anchor='middle')
  f.end()
  big_arrow(f, 797, 138, 815)

  # ======================================================================= mix-trajectory reuse
  f.group('reuse')
  f.rect(816, 4, 180, 208, fill=MOD_FILL, stroke=MOD_STROKE, sw=1.0, rx=12)
  f.text(906, 25, '[b]Mix-Trajectory Reuse[/b]', size=13.5, color=TXT, anchor='middle')
  f.rect(830, 34, 152, 30, fill='#fbd9b8', rx=5)
  f.rect(846, 40, 10, 17, fill='white', stroke='#222', sw=1.1, rx=2)
  f.rect(848.5, 43, 5, 1.2, fill='#9aa0a8', rx=0.5)
  f.text(860, 53.5, '[b]State [i]i[/i]+1[/b]', size=11, color=TXT)
  f.text(916, 53.5, '[b]+[/b]', size=11, color=TXT, anchor='middle')
  f.rect(926, 41, 12, 15, fill='white', stroke='#222', sw=1.1, rx=2)
  f.rect(929, 39.5, 6, 3, fill='#222', rx=1)
  f.rect(928.5, 46, 7, 1.2, fill='#222', rx=0.5)
  f.rect(928.5, 50, 6, 1.2, fill='#222', rx=0.5)
  f.text(942, 53.5, '[b]Task[/b]', size=11, color=TXT)
  f.arrow([(906, 64), (906, 80)], color=TXT, sw=1.6, size=6)
  f.text(913, 76, '[i]retrieve[/i]', size=9.5, color=TXT2)
  f.rect(828, 82, 156, 44, fill='white', stroke='#222', sw=1.1, rx=4)
  f.text(906, 95, '[b]Mixed Trajectory[/b]', size=10.5, color=TXT, anchor='middle')
  xs = [846, 880, 914, 948, 974]
  ys = 112
  f.rect(836, 102, 20, 20, fill='none', stroke='#7cc35a', sw=1.4, rx=3, dash='3 2')
  for k, x in enumerate(xs[:-1]):
    node(f, x, ys, r=6.4)
  edge(f, (846, ys), (880, ys), r=7)
  edge(f, (880, ys), (914, ys), dash='3.5 2.5', color=NEW, r=7, sw=1.5)
  edge(f, (914, ys), (948, ys), r=7)
  f.check(846, 136, s=9, color='#2e9a57', sw=2.2)
  f.text(858, 140, '[b]verified[/b] on live UI', size=9.5, color=TXT2)
  f.arrow([(906, 145), (906, 158)], color=TXT, sw=1.6, size=6)
  # executed directly: open app, tap, type
  app_grid(f, 834, 165, s=13)
  f.arrow([(851, 171.5), (866, 171.5)], color=TXT, sw=1.3, size=4.5)
  cursor(f, 873, 164, s=0.95)
  f.arrow([(888, 171.5), (903, 171.5)], color=TXT, sw=1.3, size=4.5)
  type_glyph(f, 907, 176)
  f.arrow([(926, 171.5), (941, 171.5)], color=TXT, sw=1.3, size=4.5)
  node(f, 953, 171.5, r=7)
  f.text(906, 194, '[b]Trajectory Reuse[/b]', size=11.5, color=TXT, anchor='middle')
  f.text(906, 207, f'[b][c={RED}]skip reasoning for 3 steps[/c][/b]', size=10, color=RED, anchor='middle')
  f.end()

  # ======================================================================= cross-task mix-trajectory graph
  f.group('graph')
  f.rect(158, 236, 838, 170, fill=GR_FILL, stroke=GR_STROKE, sw=1.0, rx=12)
  f.text(577, 256, '[b]Cross-task Mix-Trajectory Graph[/b]', size=13.5, color=TXT, anchor='middle')
  # legend (original style)
  f.rect(884, 242, 104, 38, fill='white', stroke='#29b6f6', sw=1.6, rx=3)
  f.arrow([(891, 255), (921, 255)], color=TXT, sw=1.3, size=5)
  f.text(926, 259, 'Reasoning', size=10.5, color=TXT)
  f.arrow([(891, 271), (921, 271)], color=TXT, sw=1.3, dash='3.5 2.5', size=5)
  f.text(926, 275, 'Exploration', size=10.5, color=TXT)
  # left: trajectories from model execution and from exploration (no screenshots)
  f.text(176, 290, '[b]Task 1[/b]', size=11, color=TXT)
  t1 = [(236, 286), (276, 286), (316, 286), (356, 286)]
  f.text(176, 322, '[b]Task 2[/b]', size=11, color=TXT)
  t2 = [(236, 318), (276, 318), (316, 318)]
  for path in (t1, t2):
    for p, q in zip(path, path[1:]):
      edge(f, p, q)
    for p in path:
      node(f, *p)
  f.text(282, 345, '[b]Task Execution Trajectory[/b]', size=11.5, color=TXT, anchor='middle')
  ex = [(196, 370), (236, 370), (276, 370), (316, 370), (276, 394)]
  for p, q in ((0, 1), (1, 2), (2, 3), (1, 4)):
    edge(f, ex[p], ex[q], dash='3.5 2.5')
  for p in ex:
    node(f, *p)
  f.text(380, 374, '[b]Exploration[/b]', size=11.5, color=TXT, anchor='middle')
  f.text(380, 388, '[b]Trajectory[/b]', size=11.5, color=TXT, anchor='middle')
  # merge
  big_arrow(f, 420, 302, 470)
  big_arrow(f, 420, 372, 470)
  f.line([(474, 290), (490, 290), (490, 384), (474, 384)], stroke=TXT, sw=2.2, cap='butt')
  big_arrow(f, 490, 337, 540)
  f.text(515, 360, '[b]merge[/b]', size=10.5, color=TXT2, anchor='middle')
  f.text(515, 373, '[b]states[/b]', size=10.5, color=TXT2, anchor='middle')
  # right: one merged graph; unseen frontier for exploration, mixed path for reuse
  f.rect(548, 266, 326, 134, fill='#dfe5f3', stroke='#333', sw=1.0, rx=8)
  app_icon_x, app_icon_y = 566, 322
  f.rect(app_icon_x, app_icon_y, 18, 18, fill='white', stroke='#333', sw=1.0, rx=3)
  f.rect(app_icon_x, app_icon_y, 18, 5, fill=RED, rx=2)
  for k in range(6):
    f.rect(app_icon_x + 2.5 + (k % 3) * 4.8, app_icon_y + 7.5 + (k // 3) * 4.6, 3, 3, fill='#333', rx=0.5)
  f.text(575, 354, '[b]App[/b]', size=10.5, color=TXT, anchor='middle')
  G = {'r': (616, 331), 'a': (664, 296), 'b': (664, 366), 'c': (716, 290), 'd': (716, 331), 'e': (716, 376),
       'f': (770, 290), 'g': (770, 336), 'h': (770, 380)}
  reuse_path = [('r', 'a'), ('a', 'c'), ('c', 'f')]
  for p, q in reuse_path:
    f.line([G[p], G[q]], stroke='#b7e3a0', sw=8)
  for p, q in [('r', 'a'), ('r', 'b'), ('a', 'c'), ('b', 'e'), ('e', 'h'), ('d', 'g')]:
    edge(f, G[p], G[q])
  for p, q in [('c', 'f'), ('a', 'd'), ('b', 'd')]:
    edge(f, G[p], G[q], dash='3.5 2.5')
  for k, p in G.items():
    node(f, *p)
  f.text(784, 294, '[b]Task 1[/b]', size=10.5, color=TXT)
  f.text(784, 340, '[b]Task 2[/b]', size=10.5, color=TXT)
  f.text(784, 384, '[b]Task 3[/b]', size=10.5, color=TXT)
  # unseen frontier of State i (the probe target)
  q = (748, 311)
  f.line([G['d'], q], stroke='#f9cfae', sw=7)
  edge(f, G['d'], q, dash='3.5 2.5', color=NEW, sw=1.6)
  f.add(f'<circle cx="{q[0]}" cy="{q[1]}" r="8" fill="white" stroke="{NEW}" stroke-width="1.4" stroke-dasharray="2.5 2"/>')
  f.text(q[0], q[1] + 4, '[b]?[/b]', size=11, color=NEW, anchor='middle')
  f.text(761, 315, '[b]unseen[/b]', size=9.5, color=NEW)
  f.text(640, 284, '[b]reused[/b]', size=9.5, color='#3f8f2a', anchor='middle')
  f.end()

  # ======================================================================= links between the graph and the modules
  f.group('links')
  f.arrow([(660, 236), (660, 213)], color=NEW, sw=1.6, size=5.5)
  f.text(666, 229, '[b][i]seen count n[/i][/b]', size=9.5, color=NEW)
  f.arrow([(766, 213), (766, 236)], color=TXT, sw=1.4, dash='3.5 2.5', size=5.5)
  f.text(772, 229, '[i]record[/i]', size=9.5, color=TXT2)
  f.arrow([(906, 236), (906, 213)], color=TXT, sw=1.4, size=5.5)
  f.text(912, 229, '[i]retrieve[/i]', size=9.5, color=TXT2)
  f.end()
  return f


def main():
  f = draw()
  (HERE / 'overview_v3.svg').write_text(f.svg())
  print('wrote', HERE / 'overview_v3.svg')


if __name__ == '__main__':
  main()
