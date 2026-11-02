#!/usr/bin/env python3
"""MobileExplorer system overview, concept-level version (MMEdge / SenSys style).

One core picture -- model reasoning and live-UI exploration running in parallel along time --
with each module drawn on the part of the timeline it acts on:

  Novelty-Seeking Exploration   probes during step i's reasoning window, and names the three
                                cues used to pick a probe (unseen in graph, task-relevant,
                                recoverable)
  Mix-Trajectory Reuse          at step i+1 a verified recorded action skips the reasoning
  Cross-task Mix-Trajectory Graph  records both lanes and serves both modules
  Adaptive Runtime Optimizer    max exploration s.t. reasoning latency

Short labels only; scoring rules, formulas and thresholds are left to Sec. 5 and its figures.
Screens are vector wireframes (illustrative). Uses the drawing helpers of draw_overview.py.

Run:  python3 draw_overview_v2.py && node render.js overview_v2.svg
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('base', HERE / 'draw_overview.py')
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
Fig, FONT = base.Fig, base.FONT
base.W, base.H = 1000, 376

TXT, TXT2, GRAY = '#1a1a1a', '#3d3d3d', '#8a8f98'
IN_FILL, IN_STROKE = '#fbe4e8', '#e6a0ad'          # input (pink)
MOD_FILL, MOD_STROKE = '#dbebf8', '#9cc5ea'        # modules (light blue)
GR_FILL, GR_STROKE = '#e5f2dc', '#a9d08f'          # graph (light green)
CHEV, CHEV_EDGE = '#f7d0a9', '#e8a565'             # exploration
BLUE, DIAMOND, RED, ORANGE = '#2f6fd0', '#4a86c8', '#e0352b', '#e47a22'
APP_BLUE = '#5b8def'


# ----------------------------------------------------------------------------- icons
def screen(f, x, y, kind, w=34, h=60):
  """Phone wireframe. kind: list | editor | menu | detail."""
  f.rect(x, y, w, h, fill='white', stroke='#2a2a2a', sw=1.0, rx=5)
  ix, iy, iw, ih = x + 2.2, y + 5.2, w - 4.4, h - 9.6
  f.rect(ix, iy, iw, ih, fill='#f6f7f9', rx=1.5)
  f.rect(x + w / 2 - 4, y + 2.1, 8, 1.3, fill='#9aa0a8', rx=0.6)
  f.rect(x + w / 2 - 5, y + h - 2.9, 10, 1.1, fill='#9aa0a8', rx=0.5)
  f.rect(ix, iy, iw, 6.5, fill=APP_BLUE, rx=1.2)
  f.rect(ix + 2, iy + 2.6, iw * 0.35, 1.4, fill='white', rx=0.6)
  if kind in ('list', 'menu'):
    for k in range(5):
      ry = iy + 9.5 + k * 7.6
      f.circle(ix + 3.6, ry + 2.4, 1.9, fill='#c9cfd8')
      f.rect(ix + 7, ry + 0.8, iw * 0.55, 1.4, fill='#9aa3ae', rx=0.6)
      f.rect(ix + 7, ry + 3.4, iw * 0.38, 1.1, fill='#c9ced6', rx=0.5)
    if kind == 'list':
      f.circle(ix + iw - 4.6, iy + ih - 4.8, 3.4, fill=ORANGE)
      f.path(f'M{ix + iw - 6.4:g},{iy + ih - 4.8:g} H{ix + iw - 2.8:g} M{ix + iw - 4.6:g},{iy + ih - 6.6:g} '
             f'V{iy + ih - 3:g}', stroke='white', sw=0.8)
    else:                                   # navigation drawer over the list
      f.rect(ix, iy, iw, ih, fill='#1f2937', rx=1.5)
      f.add(f'<rect x="{ix:g}" y="{iy:g}" width="{iw:g}" height="{ih:g}" rx="1.5" fill="white" opacity="0.55"/>')
      f.rect(ix, iy, iw * 0.68, ih, fill='white', rx=1.5)
      f.rect(ix, iy, iw * 0.68, 9, fill='#e8eefb', rx=1.5)
      for k in range(5):
        f.rect(ix + 2.5, iy + 12.5 + k * 6.6, iw * 0.42, 1.5, fill='#7f8894', rx=0.6)
  elif kind == 'editor':
    for k, fw in enumerate((0.85, 0.85)):
      f.rect(ix + 2, iy + 9.5 + k * 8.5, iw * fw, 5.6, fill='white', stroke='#b8c0cb', sw=0.5, rx=1)
    f.rect(ix + 3.2, iy + 11.7, iw * 0.3, 1.2, fill='#8a93a0', rx=0.5)
    f.rect(ix, iy + ih - 15, iw, 15, fill='#dfe3e8', rx=1)
    for r in range(3):
      for c in range(5):
        f.rect(ix + 1.6 + c * (iw - 3) / 5, iy + ih - 13.6 + r * 4.5, (iw - 3) / 5 - 1.1, 3.4, fill='white', rx=0.5)
  elif kind == 'detail':
    f.rect(ix + 2, iy + 9.5, iw - 4, 12, fill='#cfe0fb', rx=1)
    for k in range(4):
      f.rect(ix + 2, iy + 25 + k * 4.6, (iw - 4) * (0.9 if k % 2 == 0 else 0.6), 1.4, fill='#9aa3ae', rx=0.6)
    f.rect(ix + iw - 14, iy + ih - 7, 12, 4.6, fill=APP_BLUE, rx=1.2)


def vlm_icon(f, x, y, s=24):
  f.add(f'<rect x="{x:g}" y="{y:g}" width="{s:g}" height="{s:g}" rx="{s * 0.24:g}" fill="url(#vlmGrad)"/>')
  cx, cy, r = x + s / 2, y + s / 2, s * 0.33
  f.add(f'<path d="M{cx:g},{cy - r:g} Q{cx:g},{cy:g} {cx + r:g},{cy:g} Q{cx:g},{cy:g} {cx:g},{cy + r:g} '
        f'Q{cx:g},{cy:g} {cx - r:g},{cy:g} Q{cx:g},{cy:g} {cx:g},{cy - r:g} Z" fill="white"/>')
  r2 = s * 0.12
  cx2, cy2 = x + s * 0.76, y + s * 0.25
  f.add(f'<path d="M{cx2:g},{cy2 - r2:g} Q{cx2:g},{cy2:g} {cx2 + r2:g},{cy2:g} Q{cx2:g},{cy2:g} {cx2:g},{cy2 + r2:g} '
        f'Q{cx2:g},{cy2:g} {cx2 - r2:g},{cy2:g} Q{cx2:g},{cy2:g} {cx2:g},{cy2 - r2:g} Z" fill="white"/>')


def phone_icon(f, x, y, w=17, h=29):
  f.rect(x, y, w, h, fill='#33373d', rx=3)
  f.rect(x + 1.6, y + 3, w - 3.2, h - 6, fill='#f3f5f8', rx=1)
  cols = ['#f25c54', '#4a90e2', '#f7b32b', '#36b37e', '#9b6dd6', '#ff8a3d']
  for k, c in enumerate(cols):
    cx = x + 3.6 + (k % 2) * (w - 7.2 - 4) + 2
    cy = y + 5.5 + (k // 2) * 6.3
    f.rect(cx - 2.2, cy - 2.2, 4.4, 4.4, fill=c, rx=1)


def chip(f, x, y, s=18, color='#2a2a2a'):
  for k in range(3):
    t = x + s * (0.3 + 0.2 * k)
    u = y + s * (0.3 + 0.2 * k)
    f.path(f'M{t:g},{y:g} V{y + s * 0.18:g} M{t:g},{y + s * 0.82:g} V{y + s:g} '
           f'M{x:g},{u:g} H{x + s * 0.18:g} M{x + s * 0.82:g},{u:g} H{x + s:g}', stroke=color, sw=1.2)
  f.rect(x + s * 0.18, y + s * 0.18, s * 0.64, s * 0.64, fill='white', stroke=color, sw=1.4, rx=2)
  f.rect(x + s * 0.36, y + s * 0.36, s * 0.28, s * 0.28, fill=color, rx=1)


def chevron(f, x, y, w=15, h=18, fill=CHEV, notch=5):
  pts = [(x, y), (x + w - notch, y), (x + w, y + h / 2), (x + w - notch, y + h), (x, y + h), (x + notch, y + h / 2)]
  f.add('<polygon points="' + ' '.join(f'{a:g},{b:g}' for a, b in pts) + f'" fill="{fill}"/>')


def arrow_bar(f, x0, x1, y, h, fill, stroke='none', dash=None, tip=10):
  pts = [(x0, y), (x1 - tip, y), (x1, y + h / 2), (x1 - tip, y + h), (x0, y + h)]
  d = f' stroke-dasharray="{dash}"' if dash else ''
  f.add('<polygon points="' + ' '.join(f'{a:g},{b:g}' for a, b in pts) +
        f'" fill="{fill}" stroke="{stroke}" stroke-width="1"{d} stroke-linejoin="round"/>')


def back_icon(f, cx, cy, r=5.5, color=ORANGE, sw=1.5):
  """U-turn arrow (return to the previous screen)."""
  f.path(f'M{cx + r:g},{cy + r * 0.9:g} V{cy:g} A{r:g},{r:g} 0 0 0 {cx - r:g},{cy:g} V{cy + r * 0.35:g}',
         stroke=color, sw=sw)
  f.head(cx - r, cy, cx - r, cy + r * 1.15, color, r * 0.95)


def tap_icon(f, cx, cy, color=TXT):
  f.circle(cx, cy, 2.4, fill=color)
  f.path(f'M{cx - 4.6:g},{cy - 1.6:g} A4.9,4.9 0 0 1 {cx + 4.6:g},{cy - 1.6:g}', stroke=color, sw=0.9)
  f.path(f'M{cx - 7:g},{cy - 2.6:g} A7.5,7.5 0 0 1 {cx + 7:g},{cy - 2.6:g}', stroke=color, sw=0.8)


def target_icon(f, cx, cy, color):
  f.circle(cx, cy, 5.2, fill='white', stroke=color, sw=1.1)
  f.circle(cx, cy, 2.9, fill='white', stroke=color, sw=1.1)
  f.circle(cx, cy, 1.1, fill=color)


def mini_graph_icon(f, cx, cy, color):
  a, b, c = (cx - 5, cy + 3.5), (cx, cy - 4), (cx + 5.5, cy + 3)
  f.line([a, b], stroke=color, sw=1.1)
  f.line([b, c], stroke=color, sw=1.1, dash='1.6 1.3')
  for p, fill in ((a, color), (b, color), (c, 'white')):
    f.circle(p[0], p[1], 2.1, fill=fill, stroke=color, sw=1.0)


def diamond(f, cx, cy, hw, hh, label):
  pts = [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]
  f.add('<polygon points="' + ' '.join(f'{a:g},{b:g}' for a, b in pts) + f'" fill="{DIAMOND}"/>')
  f.text(cx, cy + 3.8, f'[b]{label}[/b]', size=10.5, color='white', anchor='middle')


def cue(f, x, y, icon, label, color):
  w = 20.5 + {'Unseen': 34.3, 'Task-relevant': 60.6, 'Recoverable': 56.5}.get(label, len(label) * 5.6) + 8
  f.rect(x, y, w, 17, fill='white', stroke=color, sw=1.0, rx=8.5)
  icon(f, x + 9.5, y + 8.5, color)
  f.text(x + 20.5, y + 12, f'[b]{label}[/b]', size=9.5, color=TXT)
  return x + w


# ----------------------------------------------------------------------------- figure
def draw():
  f = Fig()
  f.add('<defs>'
        '<linearGradient id="reasonGrad" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#e9f2b6"/><stop offset="1" stop-color="#f8e39b"/></linearGradient>'
        '<linearGradient id="vlmGrad" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#8b7cf6"/><stop offset="1" stop-color="#5b4fd0"/></linearGradient>'
        '</defs>')

  # ---------------------------------------------------------------- input
  f.group('input')
  f.rect(8, 8, 176, 172, fill=IN_FILL, stroke=IN_STROKE, sw=1.2, rx=10)
  f.text(96, 28, '[b]Input at Step [i]i[/i][/b]', size=13, color=TXT, anchor='middle')
  screen(f, 26, 40, 'list', w=46, h=82)
  f.text(49, 138, '[b]Screenshot[/b]', size=10.5, color=TXT, anchor='middle')
  # task description card
  f.rect(110, 44, 46, 56, fill='white', stroke='#2a2a2a', sw=1.2, rx=4)
  f.rect(124, 39, 18, 8, fill='#2a2a2a', rx=2)
  for k in range(3):
    yy = 58 + k * 12
    f.rect(116, yy - 3, 6, 6, fill='white', stroke='#2a2a2a', sw=1.0, rx=1)
    f.rect(126, yy - 1, 24, 2.2, fill='#2a2a2a', rx=1)
  f.check(119, 57.5, s=5, color='#2e9a57', sw=1.3)
  f.text(133, 125, '[b]Task[/b]', size=10.5, color=TXT, anchor='middle')
  f.text(133, 138, '[b]Description[/b]', size=10.5, color=TXT, anchor='middle')
  f.text(96, 163, '[i]“Create a note named groceries”[/i]', size=9.5, color=TXT2, anchor='middle')
  f.end()
  f.add(f'<polygon points="184,104 196,104 196,98 207,109 196,120 196,114 184,114" fill="{TXT}"/>')

  # ---------------------------------------------------------------- adaptive runtime optimizer
  f.group('optimizer')
  f.rect(8, 192, 176, 176, fill=MOD_FILL, stroke=MOD_STROKE, sw=1.2, rx=10)
  f.text(96, 211, '[b]Adaptive Runtime[/b]', size=13, color=TXT, anchor='middle')
  f.text(96, 227, '[b]Optimizer[/b]', size=13, color=TXT, anchor='middle')
  vlm_icon(f, 17, 240, s=22)
  chip(f, 45, 242, s=18)
  arrow_bar(f, 69, 166, 242, 18, 'url(#reasonGrad)', tip=8)
  f.text(113, 255, '[b]Reasoning[/b]', size=10, color=TXT, anchor='middle')
  phone_icon(f, 20, 272, w=16, h=27)
  chip(f, 45, 276, s=18)
  for k in range(3):
    chevron(f, 69 + k * 21, 277, w=19, h=17)
  f.text(100, 289.5, '[b]Exploration[/b]', size=10, color=TXT, anchor='middle')
  f.line([(171, 236), (171, 302)], stroke=RED, sw=1.6, dash='4 2.5', cap='butt')
  f.text(96, 326, f'[b]Max [c={RED}]Exploration[/c] s.t.[/b]', size=11, color=TXT, anchor='middle')
  f.text(96, 342, f'[b][c={RED}]Reasoning Latency[/c][/b]', size=11, color=TXT, anchor='middle')
  f.end()
  f.arrow([(184, 199), (237, 199)], color=TXT, sw=1.3, size=5.5)
  f.text(210, 194, '[i]budget[/i]', size=9.5, color=TXT2, anchor='middle')

  # ---------------------------------------------------------------- core: parallel reasoning and exploration
  f.group('timeline')
  f.text(594, 21, '[b]Parallel Reasoning and Exploration[/b]', size=15, color=TXT, anchor='middle')
  f.text(494, 37, '[b]Step [i]i[/i][/b]', size=11, color=GRAY, anchor='middle')
  f.text(878, 37, '[b]Step [i]i[/i]+1[/b]', size=11, color=GRAY, anchor='middle')
  f.line([(757, 30), (757, 84)], stroke='#c3c8cf', sw=1.0, dash='3 3', cap='butt')
  # lane labels
  vlm_icon(f, 200, 45, s=24)
  f.text(212, 82, '[b]VLM[/b]', size=10, color=TXT, anchor='middle')
  phone_icon(f, 203.5, 124, w=17, h=29)
  f.text(212, 166, '[b]Live UI[/b]', size=10, color=TXT, anchor='middle')
  # VLM lane, step i: reasoning, then the action
  arrow_bar(f, 238, 598, 45, 24, 'url(#reasonGrad)', tip=11)
  f.text(418, 61.5, '[b]Reasoning for Step [i]i[/i][/b]', size=11.5, color=TXT, anchor='middle')
  f.rect(608, 46, 140, 22, fill='white', stroke='#2a2a2a', sw=0.9, rx=3)
  f.text(678, 61, '[b]Action:[/b] Click (90, 1800)', size=10.5, color=TXT, anchor='middle')
  f.line([(602, 40), (602, 178)], stroke=RED, sw=1.6, dash='4 2.5', cap='butt')
  # VLM lane, step i+1: reasoning skipped by reuse
  arrow_bar(f, 776, 984, 45, 24, '#f2f3f5', stroke='#a3a8b0', dash='4 3', tip=11)
  f.text(876, 61.5, '[i]Reasoning skipped[/i]', size=11, color='#6b7280', anchor='middle')
  f.end()

  # ---------------------------------------------------------------- novelty-seeking exploration (step i)
  f.group('exploration')
  f.rect(238, 88, 266, 124, fill=MOD_FILL, stroke=MOD_STROKE, sw=1.2, rx=10)
  f.text(371, 104, '[b]Novelty-Seeking Exploration[/b]', size=12.5, color=TXT, anchor='middle')
  screen(f, 250, 110, 'list')
  f.text(267, 181, '[i]s[/i]_{[i]i[/i]}', size=10.5, color=TXT, anchor='middle')
  for k in range(2):
    chevron(f, 289 + k * 12, 131, w=13, h=16, notch=4.5)
  screen(f, 318, 110, 'editor')
  back_icon(f, 365, 139)
  for k in range(2):
    chevron(f, 375 + k * 12, 131, w=13, h=16, notch=4.5)
  screen(f, 404, 110, 'menu')
  back_icon(f, 451, 139)
  f.text(478, 143, '[b]· · ·[/b]', size=11, color=TXT2, anchor='middle')
  x = cue(f, 249, 187, mini_graph_icon, 'Unseen', '#7b5cc8')
  x = cue(f, x + 6, 187, target_icon, 'Task-relevant', BLUE)
  cue(f, x + 6, 187, lambda g, cx, cy, c: back_icon(g, cx, cy + 1, r=3.8, color=c, sw=1.2), 'Recoverable', ORANGE)
  f.end()

  # ---------------------------------------------------------------- action handoff, then reuse at step i+1
  f.group('reuse')
  f.line([(504, 140), (600, 140)], stroke='#9aa0a8', sw=1.2, cap='butt')
  f.arrow([(604, 140), (697, 140)], color=TXT, sw=1.5, size=6)
  f.text(640, 133, '[i]a[/i]_{[i]i[/i]}', size=11, color=TXT, anchor='middle')
  screen(f, 699, 110, 'editor')
  f.text(716, 181, '[i]s[/i]_{[i]i[/i]+1}', size=10.5, color=TXT, anchor='middle')
  f.rect(764, 88, 228, 124, fill=MOD_FILL, stroke=MOD_STROKE, sw=1.2, rx=10)
  f.text(878, 104, '[b]Mix-Trajectory Reuse[/b]', size=12.5, color=TXT, anchor='middle')
  f.arrow([(733, 140), (777, 140)], color=TXT, sw=1.5, size=6)
  diamond(f, 806, 140, 28, 20, 'Reuse?')
  f.arrow([(834, 140), (944, 140)], color=BLUE, sw=1.8, size=6.5)
  f.text(851, 134, '[b]Yes[/b]', size=10, color=BLUE, anchor='middle')
  f.text(889, 155, '[i]recorded action[/i]', size=9.5, color=BLUE, anchor='middle')
  screen(f, 946, 110, 'detail')
  f.text(963, 181, '[i]s[/i]_{[i]i[/i]+2}', size=10.5, color=TXT, anchor='middle')
  f.arrow([(806, 160), (806, 190)], color=TXT2, sw=1.3, size=5.5)
  f.text(813, 200, '[b]No[/b] → reason & explore', size=9.5, color=TXT2)
  f.end()

  # ---------------------------------------------------------------- cross-task mix-trajectory graph
  f.group('graph')
  f.rect(196, 232, 796, 136, fill=GR_FILL, stroke=GR_STROKE, sw=1.2, rx=10)
  f.text(594, 251, '[b]Cross-task Mix-Trajectory Graph[/b]', size=13, color=TXT, anchor='middle')
  dx = 52
  node = {k: (x + dx, y) for k, (x, y) in {
      'R': (272, 317), 'a1': (352, 284), 'a2': (432, 284), 'a3': (512, 284), 'a4': (592, 284),
      'b1': (352, 350), 'b2': (432, 350), 'b3': (512, 350), 'x1': (432, 317), 'x2': (512, 317)}.items()}
  solid = [('R', 'a1'), ('a1', 'a2'), ('a2', 'a3'), ('a3', 'a4'), ('R', 'b1'), ('b1', 'b2'), ('b2', 'b3')]
  dashed = [('a1', 'x1'), ('b2', 'x2')]
  glow_new, glow_reuse = ('b2', 'x2'), [('a1', 'a2'), ('a2', 'a3')]

  def edge(p, q, color, sw, dash=None):
    (x1, y1), (x2, y2) = node[p], node[q]
    r1 = 12.5 if p == 'R' else 8
    L = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    f.arrow([(x1 + ux * r1, y1 + uy * r1), (x2 - ux * 8.5, y2 - uy * 8.5)], color=color, sw=sw, dash=dash, size=5)

  for p, q in glow_reuse:
    f.line([node[p], node[q]], stroke='#b9d3f7', sw=8)
  f.line([node[glow_new[0]], node[glow_new[1]]], stroke='#f9d2ab', sw=8)
  for p, q in solid:
    edge(p, q, '#2a2a2a', 1.3)
  for p, q in dashed:
    edge(p, q, '#2a2a2a', 1.3, dash='3.5 2.5')
  for k, (cx, cy) in node.items():
    if k == 'R':
      continue
    fill = '#fde3c8' if k == 'x2' else ('#d9e7fb' if k in ('a1', 'a2', 'a3') else 'white')
    f.circle(cx, cy, 7.5, fill=fill, stroke='#4b5563', sw=1.1)
  rx_, ry_ = node['R']
  f.rect(rx_ - 11, ry_ - 11, 22, 22, fill='white', stroke='#4b5563', sw=1.1, rx=5)
  for k, c in enumerate(['#f25c54', '#4a90e2', '#f7b32b', '#36b37e']):
    f.rect(rx_ - 6.5 + (k % 2) * 7, ry_ - 6.5 + (k // 2) * 7, 6, 6, fill=c, rx=1.2)
  f.text(rx_, ry_ + 25, '[b]App[/b]', size=10, color=TXT, anchor='middle')
  f.text(606 + dx, 288, '[b]Task 1[/b]', size=10, color=TXT)
  f.text(526 + dx, 354, '[b]Task 2[/b]', size=10, color=TXT)
  # call-outs that tie the graph to the timeline above
  f.text(432 + dx, 272, '[i]reused at step [/i][i]i[/i]+1', size=9.5, color=BLUE, anchor='middle')
  f.text(525 + dx, 321, '[i]explored at step [/i][i]i[/i]', size=9.5, color=ORANGE)
  # legend
  lx = 760
  f.arrow([(lx, 290), (lx + 30, 290)], color='#2a2a2a', sw=1.3, size=5)
  f.text(lx + 38, 294, '[b]Reasoning[/b]', size=10, color=TXT)
  f.arrow([(lx, 312), (lx + 30, 312)], color='#2a2a2a', sw=1.3, dash='3.5 2.5', size=5)
  f.text(lx + 38, 316, '[b]Exploration[/b]', size=10, color=TXT)
  f.rect(lx - 6, 278, 128, 46, fill='none', stroke='#9fc98a', sw=0.9, rx=5)
  f.end()

  # ---------------------------------------------------------------- read / write between the timeline and the graph
  f.group('links')
  f.arrow([(306, 232), (306, 213)], color=TXT, sw=1.3, size=5)
  f.text(312, 226, '[i]coverage[/i]', size=9.5, color=TXT2)
  f.arrow([(452, 213), (452, 232)], color=TXT, sw=1.3, dash='3 2', size=5)
  f.text(458, 226, '[i]record[/i]', size=9.5, color=TXT2)
  f.arrow([(650, 146), (650, 232)], color=TXT, sw=1.3, size=5)
  f.text(656, 226, '[i]record[/i]', size=9.5, color=TXT2)
  f.arrow([(878, 232), (878, 213)], color=TXT, sw=1.3, size=5)
  f.text(884, 226, '[i]retrieve[/i]', size=9.5, color=TXT2)
  f.end()
  return f


def main():
  f = draw()
  (HERE / 'overview_v2.svg').write_text(f.svg())
  print('wrote', HERE / 'overview_v2.svg')


if __name__ == '__main__':
  main()
