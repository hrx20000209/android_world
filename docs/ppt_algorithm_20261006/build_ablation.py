"""Export the same three overall metrics for module and naive ablations."""
from pathlib import Path
import csv
import json
import os

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/mobileexplorer-ppt-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})

def draw(root, arms, labels, count, title, subtitle, stem, extra_note):
    records = {}
    for line in (root / 'pairs.jsonl').read_text().splitlines():
        row = json.loads(line)
        records[row['task']] = row
    tasks = [t for t, row in records.items()
             if all(a in row and not row[a].get('crash') for a in arms)]
    assert len(tasks) == count, (len(tasks), count)
    saved = json.loads((root / 'naive_ablation_report.json').read_text())
    metrics = {}
    for arm in arms:
        rows = [records[t][arm] for t in tasks]
        successful = [r['steps'] for r in rows if r['ok'] > 0]
        s = {'tasks': count, 'score_sum': sum(r['ok'] for r in rows),
             'success_rate_percent': 100 * sum(r['ok'] for r in rows) / count,
             'success_steps': sum(successful) / len(successful),
             'mean_steps': sum(r['steps'] for r in rows) / count,
             'successful_tasks_score_positive': len(successful)}
        assert s['score_sum'] == saved[arm]['score']
        assert abs(s['success_steps'] - saved[arm]['success_steps']) < 1e-10
        assert abs(s['mean_steps'] - saved[arm]['mean_steps']) < 1e-10
        metrics[arm] = s
    (OUT / f'{stem}_metrics.json').write_text(json.dumps({
        'source': str(root), 'arms': metrics,
        'definitions': {'success_rate': 'Mean evaluator score, partial credit included.',
                        'success_steps': 'Each arm\'s tasks with score > 0.',
                        'mean_steps': 'All tasks, including failures.'}}, indent=2))
    with (OUT / f'{stem}_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['arm'] + list(next(iter(metrics.values()))))
        writer.writeheader()
        writer.writerows({'arm': a, **s} for a, s in metrics.items())

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.4))
    panels = [('success_rate_percent', 'Task Success Rate', 'Mean evaluator score (%)', (0, 65), '{:.2f}%'),
              ('success_steps', 'Steps on Successful Tasks', 'Average decision steps', (0, 12), '{:.2f}'),
              ('mean_steps', 'Steps on All Tasks', 'Average decision steps', (0, 15), '{:.2f}')]
    colors = ['#788692' if a == 'base' else '#235DA3' if a == 'full_rec' else '#C48C42' for a in arms]
    for ax, (key, heading, ylabel, ylim, pattern) in zip(axes, panels):
        vals = [metrics[a][key] for a in arms]
        bars = ax.bar(range(len(arms)), vals, width=.64, color=colors)
        ax.set_xticks(range(len(arms)), labels, fontsize=11)
        ax.set_ylim(*ylim)
        ax.set_title(heading, fontsize=19, pad=22, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E8EBEF', linewidth=.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + ylim[1]*.024,
                    pattern.format(val), ha='center', va='bottom', fontsize=11.5)
    fig.suptitle(title, x=.065, ha='left', y=.97, fontsize=26, fontweight='bold')
    fig.text(.065, .885, subtitle, fontsize=15, color='#536373')
    fig.text(.065, .11, 'Success rate includes partial credit. Successful-task steps use each arm\'s own tasks with score > 0.',
             fontsize=12, color='#536373')
    fig.text(.065, .065, extra_note, fontsize=12, color='#536373')
    fig.subplots_adjust(left=.065, right=.987, top=.77, bottom=.24, wspace=.25)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(stem, json.dumps(metrics, indent=2))

draw(REPO / 'results/sensys27_task_navigation/20261005_ablation_r1',
     ['base', 'full_rec', 'no_replay', 'no_skip', 'no_context', 'no_explore', 'explore_untargeted'],
     ['Base', 'Full', 'w/o\nReplay', 'w/o\nSkip', 'w/o\nContext', 'w/o\nExplore', 'Untargeted'],
     29, 'Module Ablation on AndroidWorld',
     '29 tasks, r6 code, seed 30, one run per arm', '07_module_ablation_overall',
     'Untargeted disables entity-row targeting and the certifiable-app probe restriction. Steps count agent decisions.')
draw(REPO / 'results/sensys27_task_navigation/20261006_naive_full',
     ['base', 'full_rec', 'naive_graph', 'naive_explore', 'naive_use'],
     ['Base', 'Full', 'Naive\nGraph', 'Random\nExplore', 'Prompt\nOnly'],
     116, 'Naive Alternatives on AndroidWorld',
     '116 tasks, seed 30, Full = full_rec', '08_naive_ablation_overall',
     'Base/Full reuse r7. Naive arms reuse 29 earlier episodes and add 87 tasks. Steps count agent decisions.')
