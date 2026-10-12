"""PPT overall figure, checked against task-level records; no new experiments."""
from pathlib import Path
import csv
import json
import os

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
RUN = REPO / 'results/sensys27_task_navigation/20261005_full_r7'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/mobileexplorer-ppt-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pairs = {}
for line in (RUN / 'pairs.jsonl').read_text().splitlines():
    row = json.loads(line)
    pairs[row['task']] = row
reference = {}
for line in (REPO / 'results/4B_2.txt').read_text().splitlines()[2:]:
    parts = line.split()
    if len(parts) >= 5 and parts[0][:1].isupper():
        try:
            reference[parts[0]] = (float(parts[3]), float(parts[4]))
        except ValueError:
            pass
tasks = [t for t, row in pairs.items() if t in reference
         and not row['base'].get('crash') and not row['full_rec'].get('crash')]
assert len(tasks) == 116
raw = {
    '4B_2': [reference[t] for t in tasks],
    'Base': [(pairs[t]['base']['ok'], pairs[t]['base']['steps']) for t in tasks],
    'Full': [(pairs[t]['full_rec']['ok'], pairs[t]['full_rec']['steps']) for t in tasks],
}
metrics = {}
expected = json.loads((RUN / 'final_compare_full_rec.json').read_text())['arms']
for name, values in raw.items():
    success_steps = [steps for score, steps in values if score > 0]
    result = {
        'tasks': len(values), 'score_sum': sum(score for score, _ in values),
        'success_rate_percent': 100 * sum(score for score, _ in values) / len(values),
        'successful_tasks_score_positive': len(success_steps),
        'success_steps': sum(success_steps) / len(success_steps),
        'mean_steps': sum(steps for _, steps in values) / len(values),
    }
    original = expected[{'Base': 'base', 'Full': 'full_rec'}.get(name, name)]
    assert result['score_sum'] == original['score']
    assert abs(result['success_steps'] - original['success_steps']) < 1e-10
    assert abs(result['mean_steps'] - original['mean_steps']) < 1e-10
    metrics[name] = result
(OUT / 'overall_metrics.json').write_text(json.dumps({
    'run': str(RUN), 'reference': str(REPO / 'results/4B_2.txt'),
    'definitions': {
        'success_rate': 'Mean evaluator score; partial credit included.',
        'success_steps': 'Unweighted mean steps over each arm\'s tasks with score > 0.',
        'mean_steps': 'Mean steps over all 116 tasks, including failed tasks.',
        'comparison': '4B_2 is a historical baseline; Base and Full are from r7.'
    }, 'arms': metrics}, indent=2))
with (OUT / 'overall_metrics.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=['arm'] + list(next(iter(metrics.values()))))
    writer.writeheader()
    writer.writerows({'arm': name, **result} for name, result in metrics.items())

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
names = ['4B_2', 'Base', 'Full']
colors = ['#C48C42', '#788692', '#235DA3']
fig, axes = plt.subplots(1, 3, figsize=(16, 6.7))
panels = [
    ('success_rate_percent', 'Task Success Rate', 'Mean evaluator score (%)', (0, 65), '{:.2f}%'),
    ('success_steps', 'Steps on Successful Tasks', 'Average decision steps', (0, 10), '{:.2f}'),
    ('mean_steps', 'Steps on All Tasks', 'Average decision steps', (0, 13), '{:.2f}'),
]
for ax, (key, title, ylabel, limits, pattern) in zip(axes, panels):
    vals = [metrics[name][key] for name in names]
    bars = ax.bar(range(3), vals, color=colors, width=.57)
    ax.set_xticks(range(3), names, fontsize=17)
    ax.set_ylim(*limits)
    ax.set_title(title, fontsize=18, pad=23, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E8EBEF', linewidth=.9)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, value + limits[1]*.025,
                pattern.format(value), ha='center', va='bottom', fontsize=17)
fig.suptitle('Overall Performance on AndroidWorld', x=.07, ha='left',
             fontsize=25, fontweight='bold', y=.97)
fig.text(.07, .88, '116 tasks, Full = full_rec (r7)', fontsize=15, color='#536373')
fig.text(.07, .103, 'Success rate includes partial credit. Successful tasks use score > 0, separately for each arm.',
         fontsize=12, color='#536373')
fig.text(.07, .06, '4B_2 is historical. Base and Full use the paired r7 run. Steps count agent decisions, not physical GUI actions.',
         fontsize=12, color='#536373')
fig.subplots_adjust(left=.07, right=.985, top=.76, bottom=.23, wspace=.34)
for suffix in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / f'06_overall_comparison.{suffix}', dpi=220, bbox_inches='tight')
plt.close(fig)
print(json.dumps(metrics, indent=2))
