"""Editable, vector-native design sketches. No performance claims are encoded.

Run with Python 3; render the resulting SVGs with rsvg-convert if PNG is needed.
"""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parent
INK = "#202936"
MUTED = "#566273"
BLUE = "#245c9c"
TEAL = "#177466"
RED = "#9b4444"


class SVG:
    def __init__(self, width, height, title, desc):
        self.w, self.h = width, height
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">',
            f'<title>{escape(title)}</title><desc>{escape(desc)}</desc>',
            '<defs>',
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/></marker>',
            '<marker id="redarrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{RED}"/></marker>',
            '</defs>',
            '<style>text {font-family: Arial, Helvetica, sans-serif; '
            'font-variant-numeric: tabular-nums;} '
            '.edge {fill:none;stroke-linecap:round;stroke-linejoin:round;}</style>',
            f'<rect width="{width}" height="{height}" fill="white"/>',
        ]

    def rect(self, x, y, w, h, fill="#fff", stroke="#8793a1", radius=12,
             dash=None, sw=2):
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{ds}/>')

    def text(self, x, y, lines, size=27, color=INK, weight="normal", anchor="middle", gap=None):
        if isinstance(lines, str):
            lines = [lines]
        gap = gap or size * 1.32
        self.items.append(
            f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">')
        for i, line in enumerate(lines):
            self.items.append(f'<tspan x="{x}" dy="{0 if i == 0 else gap}">{escape(line)}</tspan>')
        self.items.append('</text>')

    def path(self, points, arrow=True, color=INK, dash=None, sw=2.8):
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        marker = 'redarrow' if color == RED else 'arrow'
        mk = f' marker-end="url(#{marker})"' if arrow else ""
        d = 'M ' + ' L '.join(f'{x} {y}' for x, y in points)
        self.items.append(f'<path class="edge" d="{d}" stroke="{color}" '
                          f'stroke-width="{sw}"{ds}{mk}/>')

    def box(self, x, y, w, h, title, lines=(), fill="#fff", stroke="#8793a1",
            title_color=INK, title_size=29, body_size=25):
        self.rect(x, y, w, h, fill=fill, stroke=stroke)
        self.text(x + w / 2, y + 37, title, size=title_size, weight="bold", color=title_color)
        if lines:
            self.text(x + w / 2, y + 73, lines, size=body_size, color=MUTED)

    def save(self, name):
        target = OUT / name
        target.write_text('\n'.join(self.items + ['</svg>']), encoding='utf-8')
        return target


def overview():
    s = SVG(1800, 1140, "MobileExplorer: revised system overview",
            "Control flow places reuse before inference. On a reuse miss, model inference "
            "and UI probing run concurrently, with recovery and action checking before execution. "
            "A separate panel shows two queries and two observation sources for logical memory.")
    s.text(40, 40, '(a) Control flow', size=33, weight='bold', anchor='start')

    # The UI observation closes the loop after either replay or model execution.
    s.path([(1625, 115), (1625, 70), (165, 70), (165, 115)])
    s.text(880, 58, 'next decision boundary', size=24, color=MUTED)
    s.box(40, 115, 250, 110, 'Task + live UI', ['pixels · a11y · history'], title_size=28, body_size=21)
    s.box(355, 115, 340, 110, 'Q1  Reuse lookup', ['task & current UI'],
          fill='#eef4fc', stroke=BLUE, title_color=BLUE)
    s.items.append(f'<polygon points="815,100 920,170 815,240 710,170" '
                   f'fill="#fff8e9" stroke="#b98a37" stroke-width="2"/>')
    s.text(815, 164, ['Applicable', 'candidate?'], size=25, weight='bold', gap=30)
    s.box(1000, 115, 350, 110, 'Ground & replay', ['check each step; stop on mismatch'],
          fill='#eef4fc', stroke=BLUE, title_size=28, body_size=21)
    s.box(1490, 115, 270, 110, 'Observe UI', ['record outcome'], fill='#f4f5f7', title_size=28)
    s.path([(290, 170), (355, 170)])
    s.path([(695, 170), (710, 170)])
    s.path([(920, 170), (1000, 170)])
    s.text(958, 156, 'yes', size=24)
    s.path([(1350, 170), (1490, 170)])

    # On a miss, one branch reads a frozen observation while the other changes live UI.
    s.rect(350, 300, 1010, 450, fill='#fafbfc', stroke='#a2acb7', dash='9 7', radius=16)
    s.text(372, 330, 'On a reuse miss', size=24, color=MUTED, anchor='start')
    s.path([(815, 240), (815, 350)], arrow=False)
    s.text(840, 268, 'no', size=24)
    s.path([(815, 350), (570, 350), (570, 375)])
    s.path([(815, 350), (1095, 350), (1095, 375)])
    s.box(390, 375, 360, 115, 'Model inference',
          ['frozen screenshot + task', 'stream action fields'],
          fill='#fff8e9', stroke='#b98a37', body_size=24)
    s.box(880, 375, 430, 115, 'Q2  Select & probe',
          ['unobserved UI transitions', 'operate on the live UI'],
          fill='#eef8f5', stroke=TEAL, title_color=TEAL, body_size=24)
    s.path([(750, 455), (880, 455)], dash='6 5')
    s.text(815, 408, ['stop new', 'probes'], size=21, color=MUTED, gap=27)
    s.box(880, 520, 430, 90, 'Restore & check UI',
          ['record recovery outcome'], fill='#eef8f5', stroke=TEAL, body_size=23)
    s.path([(1095, 490), (1095, 520)])
    s.path([(570, 490), (570, 625), (720, 625), (720, 650)])
    s.text(555, 548, ['action', 'available'], size=23, color=MUTED, anchor='end')
    s.path([(1095, 610), (1095, 630), (1020, 630), (1020, 650)])
    s.box(590, 650, 560, 80, 'Check action on live UI',
          ['execute, or obtain a fresh observation'], fill='#f5f0fa', stroke='#80669b',
          title_size=28, body_size=22)
    s.path([(590, 690), (315, 690), (315, 260), (165, 260), (165, 225)],
           color=RED, dash='7 5')
    s.text(440, 677, 'invalid / uncertain', size=22, color=RED)
    s.box(1470, 640, 290, 110, 'Execute action', ['wait & observe UI'],
          fill='#f4f5f7', title_size=28, body_size=24)
    s.path([(1150, 690), (1470, 690)])
    s.text(1307, 676, 'accepted', size=23)
    s.path([(1625, 640), (1625, 225)])
    s.text(350, 788, 'Runtime support: streamed action fields · adaptive UI observation',
           size=25, color=MUTED, anchor='start')

    # Memory use is a separate view so data edges do not obscure the control path.
    s.path([(40, 815), (1760, 815)], arrow=False, color='#c7cdd4', sw=1.5)
    s.text(40, 854, '(b) One logical memory, two read interfaces', size=33,
           weight='bold', anchor='start')
    s.box(40, 880, 460, 90, 'Q1  Reuse lookup', ['navigation · procedures · bindings'],
          fill='#eef4fc', stroke=BLUE, title_color=BLUE, body_size=23)
    s.box(1300, 880, 460, 90, 'Q2  Probe selection', ['unseen controls · evidence · history'],
          fill='#eef8f5', stroke=TEAL, title_color=TEAL, body_size=22)
    s.rect(650, 880, 500, 215, fill='#f4f5f7', stroke='#8793a1')
    s.text(900, 919, 'Cross-task logical memory', size=29, weight='bold')
    s.text(900, 958, ['Shared page index', 'Instance observations + outcomes',
                      'Procedure index (from execution)'], size=25, gap=37)
    s.box(40, 1010, 460, 85, 'Executed transitions', ['source: agent / replay'],
          title_size=28, body_size=23)
    s.box(1300, 1010, 460, 85, 'Probed transitions', ['source: exploration + recovery'],
          title_size=28, body_size=23)
    s.path([(650, 929), (500, 929)])
    s.text(575, 915, 'read', size=23, color=MUTED)
    s.path([(1150, 929), (1300, 929)])
    s.text(1225, 915, 'read', size=23, color=MUTED)
    s.path([(500, 1055), (650, 1055)])
    s.text(575, 1041, 'write', size=23, color=MUTED)
    s.path([(1300, 1055), (1150, 1055)])
    s.text(1225, 1041, 'write', size=23, color=MUTED)
    s.save('overview.svg')


def memory_contract():
    s = SVG(1600, 910, 'MobileExplorer: separate retrieval from execution acceptance',
            'Two UI instances share a structural page key while keeping separate contents and provenance. '
            'Transitions retain outcomes and conflicting destinations. A current UI and task are still needed '
            'to accept a retrieved behavior. This depicts the proposed logical contract, not a semantic guarantee.')
    s.text(40, 52, 'Retrieve by structure; accept using current evidence',
           size=35, weight='bold', anchor='start')
    s.text(40, 102, 'Different contents can share a page index without becoming interchangeable.',
           size=26, color=MUTED, anchor='start')

    s.box(40, 175, 420, 150, 'Observed instance A',
          ['Contact editor · task A', 'Name: Alice     Phone: 101', 'source: agent'],
          fill='#eef4fc', stroke=BLUE, body_size=24)
    s.box(40, 390, 420, 150, 'Observed instance B',
          ['Contact editor · task B', 'Name: Bob       Phone: 202', 'source: agent'],
          fill='#eef4fc', stroke=BLUE, body_size=24)

    s.rect(600, 180, 380, 360, fill='#f4f5f7', stroke='#8793a1')
    s.text(790, 221, 'Shared retrieval key', size=29, weight='bold')
    s.text(790, 266, ['package · activity', 'structural layout'], size=26, gap=34)
    s.path([(625, 324), (955, 324)], arrow=False, color='#c7cdd4', sw=1.5)
    s.text(790, 365, 'Keep instance references', size=27, weight='bold')
    s.text(790, 406, ['contents + UI attributes', 'origin + timestamp', 'interaction outcomes'], size=25, gap=34)
    s.text(790, 518, 'Index match ≠ execution equivalence', size=20, color=RED)
    s.path([(460, 250), (535, 250), (535, 280), (600, 280)])
    s.path([(460, 465), (535, 465), (535, 440), (600, 440)])
    s.text(530, 350, 'index', size=24, color=MUTED)

    s.rect(1110, 200, 450, 340, fill='#f5f0fa', stroke='#80669b')
    s.text(1335, 241, 'Execution acceptance', size=30, weight='bold')
    s.text(1335, 288, ['Bind current task parameters', 'Re-ground each live control',
                        'Check context / observed outcome', 'Stop on mismatch or uncertainty'],
           size=24, gap=45)
    s.text(1335, 502, 'Commit decisions remain with the model', size=22, color=MUTED)
    s.path([(980, 360), (1110, 360)])
    s.text(1045, 343, 'recall', size=23, color=MUTED)

    s.rect(40, 650, 940, 180, fill='#eef8f5', stroke=TEAL)
    s.text(65, 688, 'Retain transition evidence and separate procedure records',
           size=27, weight='bold', anchor='start')
    s.text(65, 731, ['agent: List — menu → Options; observed execution',
                     'explore: List — menu → Dialog; restoration rejected',
                     'Keep conflicting destinations; procedures come from execution.'],
           size=25, anchor='start', gap=38)
    s.path([(790, 650), (790, 540)])
    s.text(808, 600, 'evidence', size=23, color=MUTED, anchor='start')
    s.box(1110, 650, 450, 180, 'Live UI + current task',
          ['Name: Carol    Phone: 303', 'current fields / entity / selection', 'bind values; do not copy old contents'],
          fill='#fff8e9', stroke='#b98a37', body_size=23)
    s.path([(1335, 650), (1335, 540)])
    s.text(1350, 600, 'check', size=23, color=MUTED, anchor='start')
    s.text(40, 879, 'Logical contract for the revision; the design note identifies checks that still require implementation changes.',
           size=23, color=MUTED, anchor='start')
    s.save('memory_contract.svg')


if __name__ == '__main__':
    overview()
    memory_contract()
    print(f'Wrote two SVGs to {OUT}')
