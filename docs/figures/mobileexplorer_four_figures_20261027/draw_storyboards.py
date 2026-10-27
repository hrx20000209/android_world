"""Four editable paper-figure storyboards; examples are illustrative, not results.

The small SVG helper is reused from the preceding revision in this repository.
Run this file with Python 3, then render *.svg with rsvg-convert.
"""
from pathlib import Path
import importlib.util

OUT = Path(__file__).resolve().parent
HELPER = OUT.parent / 'mobileexplorer_revision_20261026' / 'draw_figures.py'
spec = importlib.util.spec_from_file_location('drawing_helper', HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
h.OUT = OUT
SVG = h.SVG
INK, MUTED, BLUE, TEAL, RED = h.INK, h.MUTED, h.BLUE, h.TEAL, h.RED
PURPLE = '#73568c'


def title(s, name, question):
    s.text(40, 45, name, size=34, weight='bold', anchor='start')
    s.text(40, 90, question, size=26, color=MUTED, anchor='start')


def node(s, x, y, name, color=INK):
    s.items.append(f'<circle cx="{x}" cy="{y}" r="42" fill="white" '
                   f'stroke="{color}" stroke-width="3"/>')
    s.text(x, y + 10, name, size=30, color=color, weight='bold')


def overview():
    # Reuse the reviewed control layout, with module IDs and concrete data outputs.
    old = h.OUT
    h.OUT = OUT
    h.overview()
    path = OUT / 'overview.svg'
    t = path.read_text()
    t = t.replace('(a) Control flow', '(a) One decision: reuse first; otherwise infer and explore')
    t = t.replace('Q1  Reuse lookup', 'M3  Reuse check', 1)
    t = t.replace('Q2  Select &amp; probe', 'M2  Select &amp; probe', 1)
    t = t.replace('Check action on live UI', 'M3  Check action on live UI', 1)
    t = t.replace('(b) One logical memory, two read interfaces', '(b) M1  Cross-task memory: two consumers, two observation sources')
    t = t.replace('Cross-task logical memory', 'M1  Mix-Trajectory Graph')
    t = t.replace('Q1  Reuse lookup', 'To M3: reuse candidates')
    t = t.replace('Q2  Probe selection', 'To M2: probe evidence')
    t = t.replace('unseen controls · evidence · history', 'coverage · relevance · recovery')
    t = t.replace('<text x="880" y="58"', '<text x="1280" y="58"')
    t = t.replace('>next decision boundary<', '>next observed UI<')
    t = t.replace('stroke="#80669b"', f'stroke="{BLUE}"')
    t = t.replace('fill="#f5f0fa"', 'fill="#eef4fc"')
    (OUT / '01_overview.svg').write_text(t)
    path.unlink()
    h.OUT = old


def memory():
    s = SVG(1800, 1120, 'M1: align observations and preserve evidence',
            'Illustrative observations from agent execution and exploration share a structural page key. '
            'A small graph names the pages and labels the actions. Instance contents and transition provenance '
            'are preserved. Separate outputs serve probe selection and reuse.')
    title(s, 'M1  Mix-Trajectory Graph',
          'What is shared across tasks, and what remains specific to each observation?')

    s.box(40, 145, 430, 190, 'Agent observation',
          ['Task A: find Alice’s category', 'L  — Search →  R',
           'List → search screen'], fill='#eef4fc', stroke=BLUE, body_size=25)
    s.box(40, 400, 430, 190, 'Exploration observation',
          ['During the same model request', 'L  — Categories →  C',
           'List → category page → return'], fill='#eef8f5', stroke=TEAL, body_size=24)

    s.rect(565, 260, 395, 255, fill='#f5f0fa', stroke=PURPLE)
    s.text(762, 302, 'Align by page + control', size=28, weight='bold')
    s.text(762, 351, ['page: app · activity · layout', 'control: family + selector'], size=24, gap=39)
    s.text(762, 445, ['Keep source observations', 'and their individual outcomes'], size=24, gap=35, color=MUTED)
    s.path([(470, 240), (520, 240), (520, 330), (565, 330)])
    s.path([(470, 495), (520, 495), (520, 445), (565, 445)])
    s.path([(960, 385), (1060, 385)])

    s.rect(1060, 145, 700, 470, fill='#fafbfc', stroke='#a2acb7')
    s.text(1090, 184, 'Small graph with meaningful page names', size=27, weight='bold', anchor='start')
    node(s, 1160, 370, 'L')
    node(s, 1630, 270, 'R', BLUE)
    node(s, 1630, 485, 'C', TEAL)
    s.path([(1202, 358), (1587, 280)], color=BLUE)
    s.path([(1202, 382), (1587, 474)], color=TEAL, dash='9 6')
    s.text(1380, 287, 'Search', size=27, color=BLUE)
    s.text(1380, 468, 'Categories', size=27, color=TEAL)
    s.text(1160, 444, 'Record list', size=25)
    s.text(1630, 225, 'Search screen', size=25, color=BLUE)
    s.text(1630, 565, ['Category page', 'Travel · Work'], size=24, color=TEAL, gap=30)
    s.text(1100, 575, 'blue: agent   green dashed: explore', size=22, color=MUTED, anchor='start')

    s.box(40, 660, 860, 170, 'Same page key; different contents',
          ['L at task A: Alice       L at task B: Bob',
           'Keep contents, control attributes and observation references.',
           'A shared index does not make the instances execution-equivalent.'],
          fill='#f5f0fa', stroke=PURPLE, body_size=24)
    s.box(1010, 660, 750, 170, 'One explicit transition record',
          ['source=L   control=Categories   destination=C',
           'origin=explore   recovery=accepted   observation=task A',
           'Store conflicting outcomes; do not silently replace them.'],
          fill='#eef8f5', stroke=TEAL, body_size=24)
    s.path([(1410, 615), (1410, 660)])

    s.box(40, 915, 535, 135, 'Read by M2: exploration',
          ['known / unseen transitions', 'relevance evidence + recovery history'],
          fill='#eef8f5', stroke=TEAL, body_size=24)
    s.box(630, 915, 535, 135, 'Read by M3: reuse',
          ['navigation candidates + source evidence', 'procedure entries + parameter slots'],
          fill='#eef4fc', stroke=BLUE, body_size=23)
    s.box(1220, 915, 540, 135, 'Procedure index',
          ['extracted from model execution paths', 'probe observations do not create full procedures'],
          fill='#f5f0fa', stroke=PURPLE, body_size=22)
    s.text(40, 1092, 'Illustrative UI and logical schema; physical storage and acceptance checks are specified separately.',
           size=22, color=MUTED, anchor='start')
    s.save('02_memory.svg')


def exploration():
    s = SVG(1800, 1020, 'M2: show the candidate evidence, selection and recovery',
            'An illustrative task asks for a record category. Known search and a risky sync control are excluded. '
            'Categories and Settings are compared using visible text evidence. The selected probe observes a category page, '
            'then returns and checks the starting UI before its outcome is recorded.')
    title(s, 'M2  Task-relevant online exploration',
          'Why probe this control, what does it reveal, and how is control returned to the agent?')
    s.box(40, 125, 1720, 95, 'Task A: find Alice’s category',
          ['Relevance terms: find · category     |     Alice is a task parameter'],
          fill='#fff8e9', stroke='#b98a37', title_size=29, body_size=25)

    s.box(40, 285, 335, 265, '1  Live controls',
          ['Search', 'Categories', 'Settings', 'Sync [toggle]'],
          fill='#f4f5f7', body_size=27)
    s.box(475, 285, 405, 265, '2  Exclude / admit',
          ['Search: already observed', 'Sync: risk filter rejects',
           'Keep: Categories, Settings', 'Use recorded recovery history'],
          fill='#eef8f5', stroke=TEAL, title_size=28, body_size=24)
    s.box(980, 285, 375, 265, '3  Rank by evidence',
          ['Categories → “category”', 'Settings → “settings”',
           'TF–IDF relevance proxy', 'Select Categories'],
          fill='#eef8f5', stroke=TEAL, title_size=28, body_size=25)
    s.box(1455, 285, 305, 265, '4  Probe',
          ['L — Categories → C', 'Newly observed:', 'Travel · Work'],
          fill='#eef8f5', stroke=TEAL, body_size=25)
    for a,b in [(375,475),(880,980),(1355,1455)]:
        s.path([(a, 420), (b, 420)])
    s.text(675, 259, 'M1: coverage + recovery', size=23, color=PURPLE)
    s.text(1170, 259, 'task + available text evidence', size=22, color=PURPLE)

    s.box(1055, 675, 400, 175, '5  Restore starting UI',
          ['Back / recorded navigation', 'finish recovery before handoff', 'record method and outcome'],
          fill='#eef8f5', stroke=TEAL, title_size=28, body_size=24)
    s.box(600, 675, 355, 175, '6  Compare live UI',
          ['with the starting observation', 'accepted: resume handoff', 'mismatch: fresh observation'],
          fill='#f5f0fa', stroke=PURPLE, title_size=28, body_size=24)
    s.box(40, 675, 455, 175, '7  Write outcome to M1',
          ['L — Categories → C', 'contents + source + recovery', 'retain failures as failures'],
          fill='#f5f0fa', stroke=PURPLE, title_size=28, body_size=25)
    s.path([(1607, 550), (1607, 762), (1455, 762)])
    s.path([(1055, 762), (955, 762)])
    s.path([(600, 762), (495, 762)])
    s.text(40, 935, 'Stop new probes when action fields are ready or recovery time is insufficient.',
           size=27, color=MUTED, anchor='start')
    s.text(40, 980, 'An in-flight probe can still delay execution. Example labels illustrate the mechanism; they are not measured scores.',
           size=24, color=MUTED, anchor='start')
    s.save('03_exploration.svg')


def execution():
    s = SVG(1800, 1220, 'M3: reuse before inference and coordinate shared UI execution',
            'A later task can use an observed category-page transition after checking current task and UI conditions. '
            'If no candidate applies, a lane-based timeline separates model output, live UI probing and recovery, '
            'runtime action checking, execution and observation. Summary output can overlap execution. '
            'Late recovery adds a wait and does not disappear from the critical path.')
    title(s, 'M3  Checked reuse and pipelined execution',
          'Which model call is avoided, and when can the shared live UI be used?')
    s.text(40, 145, '(a) Reuse occurs before a model request', size=29, weight='bold', anchor='start')
    s.box(40, 180, 315, 160, 'Task B + live list',
          ['“Open category list”', 'L now contains Bob'], fill='#f4f5f7', body_size=25)
    s.box(425, 180, 325, 160, 'Retrieve from M1',
          ['L — Categories → C', 'observed by exploration'], fill='#f5f0fa', stroke=PURPLE, body_size=24)
    s.box(820, 180, 430, 160, 'Check applicability',
          ['task / parameters · current controls', 'observed support · commit boundary'],
          fill='#eef4fc', stroke=BLUE, body_size=23)
    s.box(1320, 180, 440, 160, 'Execute + observe C',
          ['this transition needs no VLM request', 'stop replay on a mismatch'],
          fill='#eef4fc', stroke=BLUE, body_size=24)
    for a,b in [(355,425),(750,820),(1250,1320)]:
        s.path([(a,260),(b,260)])
    s.text(1285, 242, 'yes', size=23)
    s.path([(1035,340),(1035,385),(390,385),(390,410)], color=RED, dash='7 5')
    s.text(700, 375, 'no applicable candidate → request a model action', size=23, color=RED)

    s.text(40, 450, '(b) On a miss: one model request, one shared UI', size=29, weight='bold', anchor='start')
    s.path([(330, 505), (1760, 505)])
    s.text(1740, 487, 'time', size=23, color=MUTED)
    s.text(40, 595, 'Model output', size=27, weight='bold', anchor='start')
    s.text(40, 745, 'Live UI', size=27, weight='bold', anchor='start')
    s.text(40, 895, 'Runtime', size=27, weight='bold', anchor='start')
    for y in (620,770,920):
        s.path([(330,y),(1760,y)], arrow=False, color='#c7cdd4', sw=1.2)

    s.rect(360, 550, 630, 70, fill='#fff8e9', stroke='#b98a37')
    s.text(675, 594, 'decode action fields', size=27)
    s.rect(990, 550, 250, 70, fill='#fff8e9', stroke='#b98a37', dash='6 4')
    s.text(1115, 594, 'summary tail', size=25)
    s.rect(1620, 550, 140, 70, fill='#fff8e9', stroke='#b98a37')
    s.text(1690, 594, 'next call', size=24)
    s.rect(360, 700, 240, 70, fill='#eef8f5', stroke=TEAL)
    s.text(480, 744, 'probe', size=27)
    s.rect(600, 700, 270, 70, fill='#eef8f5', stroke=TEAL)
    s.text(735, 744, 'restore', size=27)
    s.rect(1140, 700, 220, 70, fill='#eef4fc', stroke=BLUE)
    s.text(1250, 744, 'execute', size=27)
    s.rect(1360, 700, 230, 70, fill='#f4f5f7')
    s.text(1475, 744, 'observe UI', size=25)
    s.rect(990, 850, 150, 70, fill='#eef4fc', stroke=BLUE)
    s.text(1065, 894, 'check', size=27)
    s.path([(990, 510), (990, 950)], arrow=False, color='#b98a37', dash='6 5', sw=2)
    s.text(980, 665, 'action ready', size=23, color='#90661f', anchor='end')
    s.path([(870, 780), (870, 820), (990, 820), (990, 850)], sw=2)
    s.text(780, 814, 'restored', size=22, color=TEAL)
    s.path([(1140,885),(1250,885),(1250,770)], sw=2)
    s.path([(1590,780),(1605,780),(1605,635),(1690,635),(1690,620)], sw=2)
    s.text(1450, 846, ['next prompt waits for', 'UI observation + required output'], size=22, color=MUTED)

    s.path([(40, 970), (1760, 970)], arrow=False, color='#c7cdd4', sw=1.5)
    s.text(40, 1012, '(c) Late recovery adds waiting', size=28, weight='bold', anchor='start')
    s.text(40, 1056, 'Action release ≥ max(action-ready, recovery end), followed by the live-UI check.',
           size=26, anchor='start')
    s.rect(360, 1100, 760, 55, fill='#eef8f5', stroke=TEAL)
    s.text(665, 1136, 'probe + delayed recovery', size=24)
    s.rect(990, 1100, 130, 55, fill='#f7e8e8', stroke=RED)
    s.text(1055, 1136, 'wait', size=24, color=RED)
    s.path([(990,1080),(990,1168)], arrow=False, color='#b98a37', dash='6 5', sw=2)
    s.text(990, 1198, 'action ready', size=22, color=MUTED)
    s.text(1260, 1136, 'Counts toward E2E latency', size=25, color=RED, anchor='start')
    s.save('04_execution.svg')


if __name__ == '__main__':
    overview()
    memory()
    exploration()
    execution()
    print(f'Wrote four SVG storyboards to {OUT}')
