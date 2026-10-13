#!/usr/bin/env python3
"""Generate a timestamped interim MobileExplorer progress report and figures."""
from __future__ import annotations
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get(
    'MOBILEEXPLORER_PROFILE_ROOT',
    str(ROOT / 'experiments/info_value_profile_20261012/private_voi8-live-20261012-fd89ba'),
)).expanduser()
PILOT40 = BASE / 'onecall_formal_cohort40_safety_v1_20261013'
EXT76 = BASE / 'onecall_formal_remaining76_safety_v1_20261013'
P1 = ROOT / 'experiments/mobileexplorer_design_validation/topk_evidence_pilot_20261013/summary.json'
P0_AUDIT = ROOT / 'experiments/mobileexplorer_design_validation/p0_exact_match_smoke_20261013/me-p0-smoke/audit.json'
SMOKE_ROOT = BASE / 'onecall_runtime_smoke_fastprovider_20261013'
OUT = Path(os.environ.get(
    'MOBILEEXPLORER_REPORT_OUT',
    str(ROOT / 'experiments/mobileexplorer_design_validation/reports'),
)).expanduser()
FIG = OUT / 'figures/current_progress_20261013'
FIG.mkdir(parents=True, exist_ok=True)

now = datetime.now(timezone.utc).replace(microsecond=0)
summary40 = json.loads((PILOT40 / 'summary_formal_v1.json').read_text())
manifest40 = json.loads((PILOT40 / 'manifest.json').read_text())
manifest76 = json.loads((EXT76 / 'manifest.json').read_text())
status_rows = [json.loads(line) for line in (EXT76 / 'status.jsonl').read_text().splitlines() if line.strip()]
corrections = [json.loads(line) for line in (EXT76 / 'outcome_corrections.jsonl').read_text().splitlines() if line.strip()]
correction_by_task = {r['task']: r for r in corrections}
finished_events = [r for r in status_rows if r.get('event') == 'task_finished']
finished = {r['task']: r for r in finished_events}
started = {r['task'] for r in status_rows if r.get('event') == 'task_started'}
cohort_tasks = {x['task'] for x in manifest76['run_order']}
active = sorted((started & cohort_tasks) - set(finished))
not_started = sorted(cohort_tasks - started - set(finished))
baseline_mix = Counter(x.get('baseline_outcome', 'unknown') for x in manifest76['run_order'])
complexity_mix = Counter(x.get('complexity_band', 'unknown') for x in manifest76['run_order'])
app_count = len({x.get('app') for x in manifest76['run_order']})
all_manifest_rows = manifest40['run_order'] + manifest76['run_order']
full_baseline_mix = Counter(x.get('baseline_outcome', 'unknown') for x in all_manifest_rows)
full_complexity_mix = Counter(x.get('complexity_band', 'unknown') for x in all_manifest_rows)
full_app_count = len({x.get('app') for x in all_manifest_rows})
full_unique_tasks = len({x.get('task') for x in all_manifest_rows})

# Only actual requests count toward one-call integrity; zero-request init failures are unknowns.
request_tasks = [r for r in finished.values() if r.get('decision_requests', 0) > 0]
requests76 = sum(r.get('decision_requests', 0) for r in finished.values())
all_requests_one = all(r.get('inference_calls_all_one') is True for r in request_tasks)
probes76 = sum(r.get('probe_attempts', 0) for r in finished.values())
restored76 = sum(r.get('probe_recoveries', 0) for r in finished.values())
failed76 = sum(r.get('probe_recovery_failures', 0) for r in finished.values())

# Normalize outcomes: evaluator/protocol errors explicitly audited as unknown are excluded from failure.
outcomes76 = Counter()
unknown_breakdown = Counter()
for task, row in finished.items():
    corr = correction_by_task.get(task, {})
    cls = corr.get('classification', row.get('outcome_classification', ''))
    if str(cls).startswith('unknown'):
        outcomes76['unknown'] += 1
        if cls in {'unknown_action_protocol_error', 'unknown_protocol_error'}: unknown_breakdown['action protocol error'] += 1
        elif cls == 'unknown_evaluator_protocol_error': unknown_breakdown['evaluator protocol error'] += 1
        elif cls == 'unknown_task_initialization_error': unknown_breakdown['task initialization error'] += 1
        else: unknown_breakdown['other unknown'] += 1
    elif row.get('evaluator_score') == 1 or row.get('evaluator_success') is True:
        outcomes76['success'] += 1
    elif row.get('evaluator_score') == 0 or row.get('evaluator_success') is False:
        outcomes76['failure'] += 1
    else:
        outcomes76['unknown'] += 1

p1 = json.loads(P1.read_text())
p0_audit = json.loads(P0_AUDIT.read_text())
smoke_manifest = json.loads((SMOKE_ROOT / 'manifest.json').read_text())
cohort = summary40['cohort']
exec40 = summary40['execution']
exp40 = summary40['exploration']
ev40 = summary40['evidence']

def collect_evidence_stats(root: Path):
    totals = Counter()
    for path in root.glob('*/evidence_events.jsonl'):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('event') == 'retrieve':
                totals['retrieval_events'] += 1
                totals['eligible_candidates'] += row.get('eligible', 0)
                totals['selected_records'] += len(row.get('selected') or [])
            elif row.get('event') in {'insert', 'replace'}:
                totals['insert_or_replace_events'] += 1
    for path in root.glob('*/two_system_rounds.jsonl'):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            audit = row.get('exploration_budget', {}).get('alignment_audit')
            if not isinstance(audit, dict):
                continue
            totals['alignment_audits'] += 1
            if int(audit.get('source_action_match_count', 0) or 0) > 0:
                totals['source_action_matched_rounds'] += 1
            if audit.get('successor_verdict') == 'matched':
                totals['exact_source_action_successor_matches'] += 1
    return dict(totals)

evidence40 = collect_evidence_stats(PILOT40)
evidence76 = collect_evidence_stats(EXT76)
evidence_all = Counter(evidence40)
evidence_all.update(evidence76)

def collect_ui_structure_stats(root: Path):
    """Audit runtime structure changes using the repository's layout signature."""
    totals = Counter()
    by_probe_type = {}
    example = None
    for path in sorted(root.glob('*/probe_trace.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            graph = row.get('graph') or {}
            src, dst = graph.get('src') or {}, graph.get('dst') or {}
            if not src or not dst:
                continue
            probe_type = row.get('probe_type', 'unknown')
            changed = src.get('layout_signature') != dst.get('layout_signature')
            activity_changed = src.get('activity') != dst.get('activity')
            struct_changed = src.get('structural_signature') != dst.get('structural_signature')
            totals['attempts'] += 1
            totals['layout_changed'] += int(changed)
            totals['activity_changed'] += int(activity_changed)
            totals['structural_changed'] += int(struct_changed)
            bucket = by_probe_type.setdefault(probe_type, Counter())
            bucket['attempts'] += 1
            bucket['layout_changed'] += int(changed)
            bucket['activity_changed'] += int(activity_changed)
            if path.parent.name == 'BrowserMaze' and row.get('step_idx') == 8 and changed and example is None:
                example = {
                    'task': path.parent.name,
                    'trial_id': row.get('trial_id'),
                    'probe_type': probe_type,
                    'element': row.get('element', {}),
                    'activity_source': src.get('activity'),
                    'activity_successor': dst.get('activity'),
                    'layout_source': src.get('layout_signature'),
                    'layout_successor': dst.get('layout_signature'),
                    'source_control_count': len(src.get('elements') or []),
                    'successor_control_count': len(dst.get('elements') or []),
                    'new_texts': (row.get('discovered') or {}).get('new_texts', []),
                    'recovery_ok': row.get('recovery_ok'),
                }
    return dict(totals), {k: dict(v) for k, v in by_probe_type.items()}, example

ui40, ui40_by_type, ui_example = collect_ui_structure_stats(PILOT40)
ui76, ui76_by_type, _ = collect_ui_structure_stats(EXT76)
ui_all = Counter(ui40)
ui_all.update(ui76)
if ui_example:
    screenshots = sorted((PILOT40 / 'agent' / 'BrowserMaze').glob('*/opportunity_9.png'))
    ui_example['source_screenshot'] = str(screenshots[0].relative_to(ROOT)) if screenshots else None
else:
    ui_example = {}

p0_passed = sum(r.get('status') == 'passed' for r in p0_audit)
p0_checks = {
    'scenarios_passed': p0_passed,
    'scenarios_total': len(p0_audit),
    'positive_retrieval': sum(bool(r.get('graph', {}).get('positive_retrieval')) for r in p0_audit),
    'wrong_source_rejected': sum(
        r.get('graph', {}).get('negative_wrong_source_count') == 0
        and r.get('graph', {}).get('negative_wrong_source_version_count') == 0
        and bool(r.get('graph', {}).get('negative_wrong_action_rejected'))
        for r in p0_audit
    ),
    'exact_replay_successor': sum(bool(r.get('authoritative_replay', {}).get('successor_exact_runtime_match')) for r in p0_audit),
    'cleanup_verified': sum(bool(r.get('final_cleanup', {}).get('verified')) for r in p0_audit),
    'model_calls': sum(int(r.get('model_calls', 0) or 0) for r in p0_audit),
}
smoke_rows = []
for task_row in smoke_manifest.get('tasks', []):
    task = task_row['task']
    task_root = SMOKE_ROOT / task
    requests = [json.loads(line) for line in (task_root / 'request_latency.jsonl').read_text().splitlines() if line.strip()]
    probes = [json.loads(line) for line in (task_root / 'probe_trace.jsonl').read_text().splitlines() if line.strip()]
    smoke_rows.append({
        'task': task,
        'requests': len(requests),
        'one_inference_requests': sum(int(r.get('inference_calls', 0) or 0) == 1 for r in requests),
        'probes': len(probes),
        'restored': sum(r.get('recovery_ok') is True for r in probes),
        'failed': sum(r.get('recovery_ok') is not True for r in probes),
        'success': bool(task_row.get('success')),
    })
smoke_summary = {
    'tasks': smoke_rows,
    'requests': sum(r['requests'] for r in smoke_rows),
    'one_inference_requests': sum(r['one_inference_requests'] for r in smoke_rows),
    'probes': sum(r['probes'] for r in smoke_rows),
    'restored': sum(r['restored'] for r in smoke_rows),
    'failed': sum(r['failed'] for r in smoke_rows),
}

snapshot = {
    'generated_at_utc': now.isoformat(),
    'cutoff_latest_event_utc': status_rows[-1].get('time') if status_rows else None,
    'model': 'GELAB-ZERO-4B (pilot; not MAI-UI 2B final result)',
    'pilot40': {
        'tasks_finished': 40, 'tasks_total': 40, 'apps': cohort['apps'],
        'baseline_success': cohort['baseline_success'], 'baseline_failure': cohort['baseline_failure'],
        'complexity': cohort['complexity'], 'decisions': exec40['decisions'],
        'all_decisions_one_inference': exec40['all_decisions_exactly_one_inference'],
        'outcomes': summary40['outcomes']['corrected_counts'],
        'probe_attempts': exp40['attempts'], 'probe_restored': exp40['recovered'], 'probe_failed': exp40['failed'],
        'structure_changed': ui40['layout_changed'], 'activity_changed': ui40['activity_changed'],
        'structural_signature_changed': ui40['structural_changed'],
        'evidence_retrieval_events': ev40['retrieval_events'], 'eligible_candidates_total': ev40['eligible_candidates_total'],
    'selected_records_total': ev40['selected_records_total'], 'exact_alignment': ev40['alignment_events'],
    },
    'extension76': {
        'tasks_finished': len(finished), 'tasks_total': 76, 'tasks_active': active,
        'tasks_not_started': not_started, 'decision_requests': requests76,
        'all_actual_requests_exactly_one_inference': all_requests_one,
        'cohort_baseline_mix': dict(baseline_mix), 'cohort_complexity_mix': dict(complexity_mix),
        'cohort_apps': app_count,
        'outcomes_among_finished': dict(outcomes76), 'unknown_breakdown': dict(unknown_breakdown), 'probe_attempts': probes76,
        'probe_restored': restored76, 'probe_failed': failed76,
        'ui_structure': ui76,
        'ui_structure_by_probe_type': ui76_by_type,
        'finished_event_count': len(finished_events), 'started_unique': len(started),
    },
    'full_profile': {
        'tasks': full_unique_tasks, 'apps': full_app_count,
        'baseline_mix': dict(full_baseline_mix), 'complexity_mix': dict(full_complexity_mix),
        'ui_structure': dict(ui_all),
        'ui_structure_by_cohort': {'pilot40': ui40, 'extension76': ui76},
        'ui_structure_by_probe_type': {
            k: dict(Counter(ui40_by_type.get(k, {})) + Counter(ui76_by_type.get(k, {})))
            for k in sorted(set(ui40_by_type) | set(ui76_by_type))
        },
        'ui_structure_definition': {
            'signature': 'layout_signature(source) != layout_signature(probe successor)',
            'layout_signature': 'Actionable control interaction skeleton: class + resource ID; singleton controls include 64px-quantized position; repeated control positions and all text are excluded; keyboard/system overlays are ignored.',
            'structural_signature_note': 'structural_signature includes text, selected state, and 16px-quantized bounds, so scroll/text/value changes can produce noisy deltas; it is reported separately, not used as UI structure change.',
        },
        'ui_structure_example': ui_example,
        'evidence': dict(evidence_all),
        'evidence_by_cohort': {'pilot40': evidence40, 'extension76': evidence76},
    },
    'offline_proxy24': p1,
    'p0_exact_match_smoke': p0_checks,
    'onecall_live_smoke': smoke_summary,
}
(OUT / 'data_interim_progress_20261013.json').write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'figure.dpi':160,'savefig.dpi':200})
BLUE='#3274A1'; GREEN='#55A868'; ORANGE='#E08B32'; RED='#C44E52'; GREY='#9A9A9A'; DARK='#263238'

def save(fig, name):
    fig.savefig(FIG / name, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# 1) progress and task outcomes
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.5), gridspec_kw={'width_ratios':[1,1.3]})
labels=['Frozen pilot (40)','Extension at cutoff (76)']
vals=[40,len(finished)]; totals=[40,76]
y=np.arange(2)
ax[0].barh(y, totals, color='#E9EDF0', height=.55, label='Total')
ax[0].barh(y, vals, color=[BLUE, GREEN], height=.55, label='Terminal tasks')
for i,(v,t) in enumerate(zip(vals,totals)):
    ax[0].text(v+1, i, f'{v}/{t}', va='center', fontweight='bold', color=DARK)
ax[0].set_yticks(y,labels); ax[0].invert_yaxis(); ax[0].set_xlim(0,82); ax[0].set_xlabel('Tasks'); ax[0].set_title('Execution progress')
ax[0].text(0.02,-.24,f"Extension: {len(active)} active, {len(not_started)} not started",transform=ax[0].transAxes,color='#555')
ax[0].grid(axis='x',alpha=.2); ax[0].set_axisbelow(True)
parts=['Success','Failure','Unknown']; colors=[GREEN,RED,GREY]
outcomes40=summary40['outcomes']['corrected_counts']
cohort_outcomes=[
    [outcomes40.get('outcome_success',0), outcomes40.get('outcome_failure',0), outcomes40.get('outcome_unknown',0)],
    [outcomes76.get('success',0), outcomes76.get('failure',0), outcomes76.get('unknown',0)],
    [outcomes40.get('outcome_success',0)+outcomes76.get('success',0),
     outcomes40.get('outcome_failure',0)+outcomes76.get('failure',0),
     outcomes40.get('outcome_unknown',0)+outcomes76.get('unknown',0)],
]
labels_out=['Frozen pilot\n(n=40)','Extension\n(n=76)','All tasks\n(n=116)']
bottom=np.zeros(3)
for part_idx,(lab,col) in enumerate(zip(parts,colors)):
    vals=[x[part_idx] for x in cohort_outcomes]
    ax[1].bar(np.arange(3),vals,bottom=bottom,color=col,width=.62,label=lab)
    for i,val in enumerate(vals):
        if val: ax[1].text(i,bottom[i]+val/2,str(val),ha='center',va='center',color='white',fontweight='bold')
    bottom+=np.array(vals)
ax[1].set_xticks(np.arange(3),labels_out)
ax[1].set_ylim(0,126); ax[1].set_ylabel('Tasks'); ax[1].set_title('Audited task outcomes')
ax[1].legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.13))
ax[1].grid(axis='y',alpha=.2); ax[1].set_axisbelow(True)
fig.suptitle(f'116-task acquisition profile — {now:%Y-%m-%d %H:%M UTC}',fontweight='bold',color=DARK)
fig.tight_layout()
save(fig,'progress_and_outcomes.png')

# 2) probe recovery
fig,ax=plt.subplots(figsize=(8.2,4.2))
names=['40-task pilot','76-task extension','All 116 tasks']; attempts=[ui40['attempts'],probes76,ui_all['attempts']]; success=[exp40['recovered'],restored76,exp40['recovered']+restored76]; fail=[exp40['failed'],failed76,exp40['failed']+failed76]
x=np.arange(3); width=.5
ax.bar(x,success,width,color=GREEN,label='Strictly restored')
ax.bar(x,fail,width,bottom=success,color=RED,label='Restore failed')
for i,(s,f,a) in enumerate(zip(success,fail,attempts)):
    ax.text(i,s+f+2,f'{s}/{a} = {s/a:.1%}',ha='center',fontweight='bold',color=DARK)
ax.set_xticks(x,names); ax.set_ylabel('Probe attempts'); ax.set_title('Exploration probe recovery'); ax.set_ylim(0,max(attempts)*1.22); ax.legend(frameon=False); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
fig.tight_layout(); save(fig,'probe_recovery.png')

# 2b) sample profile balance
fig, ax = plt.subplots(figsize=(8.6,4.1))
mix_labels=['Baseline success','Baseline failure','Baseline unknown','Low complexity','Middle complexity','High complexity']
mix_values=[full_baseline_mix.get('success',0),full_baseline_mix.get('failure',0),full_baseline_mix.get('unknown',0),full_complexity_mix.get('low',0),full_complexity_mix.get('middle',0),full_complexity_mix.get('high',0)]
mix_colors=[GREEN,RED,GREY,BLUE,'#6B9FBF','#A5C5D7']
bars=ax.barh(np.arange(len(mix_values)),mix_values,color=mix_colors,height=.62)
ax.set_yticks(np.arange(len(mix_values)),mix_labels); ax.invert_yaxis(); ax.set_xlim(0,70); ax.set_xlabel('Tasks (n=116)'); ax.set_title('Full AndroidWorld profile: baseline outcomes and complexity')
for b,v in zip(bars,mix_values): ax.text(v+1,b.get_y()+b.get_height()/2,str(v),va='center',fontweight='bold',color=DARK)
ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True); fig.tight_layout(); save(fig,'cohort_profile.png')

# 3) runtime UI interaction-skeleton changes
fig,ax=plt.subplots(figsize=(8.2,4.5))
cohorts=[('40-task pilot',ui40),('76-task extension',ui76),('All 116 tasks',dict(ui_all))]
x=np.arange(3); changed=[c['layout_changed'] for _,c in cohorts]; unchanged=[c['attempts']-c['layout_changed'] for _,c in cohorts]
ax.bar(x,unchanged,color=GREY,width=.58,label='Same interaction skeleton')
ax.bar(x,changed,bottom=unchanged,color=ORANGE,width=.58,label='Layout signature changed')
for i,(_,c) in enumerate(cohorts):
    n=c['attempts']; v=c['layout_changed']
    ax.text(i,n+4,f'{v}/{n} = {v/n:.1%}',ha='center',fontweight='bold',color=DARK)
ax.set_xticks(x,[c[0] for c in cohorts]); ax.set_ylabel('Probe observations'); ax.set_title('Runtime UI structure changed within the same Activity')
ax.set_ylim(0,max(c['attempts'] for _,c in cohorts)*1.23)
ax.legend(frameon=False,loc='upper left'); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
ax.text(.99,.97,f"Activity changed: {ui_all['activity_changed']}/{ui_all['attempts']}",transform=ax.transAxes,ha='right',va='top',fontsize=10,bbox=dict(boxstyle='round,pad=.4',fc='#F4F7F8',ec='#CDD5D8'))
fig.tight_layout(); save(fig,'runtime_state_deltas.png')

# 3b) real Android screenshots and the corresponding trace-recorded probe branch.
if ui_example.get('source_screenshot'):
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1], hspace=.34, wspace=.22)
    ax_src = fig.add_subplot(gs[0, :2]); ax_src.imshow(mpimg.imread(ROOT / ui_example['source_screenshot'])); ax_src.axis('off')
    ax_src.set_title('Real captured probe source (BrowserMaze, opportunity_9.png)', fontsize=10, weight='bold')
    ax_edge = fig.add_subplot(gs[0, 2:]); ax_edge.axis('off')
    source_hash=ui_example['layout_source']; successor_hash=ui_example['layout_successor']
    element=ui_example['element']
    ax_edge.text(.02,.96,'Probe transition recorded in probe_trace.jsonl',va='top',fontsize=11,weight='bold',color=DARK)
    ax_edge.text(.02,.80,f"{ui_example['probe_type']} on {element.get('class', 'ScrollView')}\n{ui_example['activity_source']}\n\nlayout_signature: {source_hash}  →  {successor_hash}\nActivity unchanged: {ui_example['activity_source'] == ui_example['activity_successor']}   |   strict restore: {ui_example['recovery_ok']}\n\nTrace reports newly visible items: {', '.join(ui_example['new_texts'][:8])}",va='top',fontsize=9.5,color=DARK,linespacing=1.5)
    ax_edge.text(.02,.08,'The run saved no post-probe PNG. The successor above is shown from its recorded hierarchy/signature; it is not a reconstructed screenshot.',va='bottom',fontsize=8.4,color='#7A3E00',wrap=True)
    wifi_root=PILOT40/'agent'/'TurnOnWifiAndOpenApp'
    wifi_candidates=sorted(wifi_root.glob('*/opportunity_*.png'))
    wifi_dir=wifi_candidates[0].parent if wifi_candidates else None
    wifi_imgs=[wifi_dir/f'opportunity_{i}.png' for i in (2,3,4)] if wifi_dir else []
    titles=['Settings home','Network & internet','Internet']
    for j,(p,title) in enumerate(zip(wifi_imgs,titles)):
        a=fig.add_subplot(gs[1,j]);
        if p.exists(): a.imshow(mpimg.imread(p))
        a.axis('off'); a.set_title(title,fontsize=9,weight='bold')
    ax_note=fig.add_subplot(gs[1,3]); ax_note.axis('off')
    ax_note.text(.02,.96,'Actual AndroidWorld screenshots',va='top',fontsize=10,weight='bold',color=DARK)
    ax_note.text(.02,.77,'TurnOnWifiAndOpenApp: successive reasoning checkpoints from the same task. These illustrate live screen variation; they are not counted as probe successor screenshots.',va='top',fontsize=9,color=DARK,wrap=True)
    fig.suptitle('Real UI observations: screenshot evidence plus exact trace-level definition',fontsize=13,weight='bold',color=DARK)
    save(fig,'ui_structure_real_examples.png')

# 3c) P0 controlled exact-source/action-slot/replay mechanism smoke.
p0_metric_labels=['Scenario passed','Evidence retrieved','Wrong-source/action blocked','Exact replay successor','Cleanup verified']
p0_metric_values=[p0_checks['scenarios_passed'],p0_checks['positive_retrieval'],p0_checks['wrong_source_rejected'],p0_checks['exact_replay_successor'],p0_checks['cleanup_verified']]
fig,ax=plt.subplots(figsize=(8.4,4.2))
bars=ax.barh(np.arange(len(p0_metric_labels)),p0_metric_values,color=[GREEN if v==p0_checks['scenarios_total'] else ORANGE for v in p0_metric_values],height=.62)
ax.set_yticks(np.arange(len(p0_metric_labels)),p0_metric_labels); ax.invert_yaxis(); ax.set_xlim(0,p0_checks['scenarios_total']+.7); ax.set_xlabel('Passing scenarios (n=3)'); ax.set_title(f"P0 exact-source/action-slot smoke ({p0_checks['model_calls']} VLM calls)")
for b,v in zip(bars,p0_metric_values): ax.text(v+.08,b.get_y()+b.get_height()/2,f'{v}/{p0_checks["scenarios_total"]}',va='center',fontweight='bold',color=DARK)
ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True); fig.tight_layout(); save(fig,'p0_exact_match_validation.png')

# 3d) one-call live smoke: request integrity, probe recovery, and task outcomes.
fig,ax=plt.subplots(1,2,figsize=(10.2,4.4),gridspec_kw={'width_ratios':[1,1.25]})
names=[r['task'] for r in smoke_rows]; q=np.arange(len(smoke_rows))
ax[0].bar(q,[r['requests'] for r in smoke_rows],color='#E9EDF0',width=.55,label='Requests')
ax[0].bar(q,[r['one_inference_requests'] for r in smoke_rows],color=BLUE,width=.32,label='Exactly one inference')
for i,r in enumerate(smoke_rows): ax[0].text(i,r['requests']+.25,f"{r['one_inference_requests']}/{r['requests']}",ha='center',fontweight='bold')
ax[0].set_xticks(q,names,rotation=8); ax[0].set_ylabel('Decision requests'); ax[0].set_ylim(0,max(r['requests'] for r in smoke_rows)*1.28); ax[0].set_title('One-inference integrity'); ax[0].legend(frameon=False,fontsize=8); ax[0].grid(axis='y',alpha=.2); ax[0].set_axisbelow(True)
ax[1].bar(q,[r['restored'] for r in smoke_rows],color=GREEN,width=.55,label='Strictly restored')
ax[1].bar(q,[r['failed'] for r in smoke_rows],bottom=[r['restored'] for r in smoke_rows],color=RED,width=.55,label='Restore failed')
for i,r in enumerate(smoke_rows):
    ax[1].text(i,r['probes']+.18,f"{r['restored']}/{r['probes']} restored",ha='center',fontweight='bold',fontsize=9)
ax[1].set_xticks(q,[f"{r['task']}\nTask {'PASS' if r['success'] else 'FAIL'}" for r in smoke_rows]); ax[1].set_ylabel('Probe attempts'); ax[1].set_ylim(0,max(r['probes'] for r in smoke_rows)*1.35); ax[1].set_title('Probe recovery and task outcomes'); ax[1].legend(frameon=False,fontsize=8); ax[1].grid(axis='y',alpha=.2); ax[1].set_axisbelow(True)
fig.suptitle('Two-task one-call parallel-exploration live smoke',fontweight='bold',color=DARK); fig.tight_layout(); save(fig,'onecall_live_smoke.png')

# 4) evidence funnel
fig,ax=plt.subplots(figsize=(7.8,4.4))
flabs=['Retrieval events','Eligible candidates\n(accumulated)','Records selected\ninto prompts','Exact source + action\n+ successor alignment']
fvals=[evidence_all['retrieval_events'],evidence_all['eligible_candidates'],evidence_all['selected_records'],evidence_all['exact_source_action_successor_matches']]
xx=np.arange(len(fvals)); bars=ax.bar(xx,fvals,color=[BLUE,'#4F90B6',ORANGE,RED],width=.62)
ax.set_xticks(xx,flabs); ax.set_ylabel('Count'); ax.set_title('Evidence acquisition-to-use funnel (116-task profile)')
for b,v in zip(bars,fvals): ax.text(b.get_x()+b.get_width()/2,v+max(fvals)*.025,str(v),ha='center',fontweight='bold')
ax.set_ylim(0,max(fvals)*1.2); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
ax.text(.5,-.25,f"{evidence_all['selected_records']} records were selected; {evidence_all['exact_source_action_successor_matches']} exact matches in {evidence_all['alignment_audits']} alignment audits.",transform=ax.transAxes,ha='center',fontsize=9,color='#555')
fig.tight_layout(); save(fig,'evidence_funnel.png')

# 5) offline proxy chart
conditions=[('No evidence','E0_no_evidence'),('Fresh outcome','E1_fresh_outcome'),('Fresh + child\naffordances','E2_fresh_child_affordances'),('Wrong source','D_wrong_source_control')]
counts=[p1['by_condition'][k]['reference_path_consistent'] for _,k in conditions]
ns=[p1['by_condition'][k]['n'] for _,k in conditions]
fig,ax=plt.subplots(figsize=(8,4.4))
xx=np.arange(len(counts)); bars=ax.bar(xx,np.array(counts)/np.array(ns),color=[GREY,BLUE,BLUE,RED],width=.62)
for b,c,n in zip(bars,counts,ns): ax.text(b.get_x()+b.get_width()/2,b.get_height()+.025,f'{c}/{n}',ha='center',fontweight='bold')
ax.set_xticks(xx,[x[0] for x in conditions]); ax.set_ylim(0,1.12); ax.set_ylabel('Reference-path agreement'); ax.set_title('Offline evidence pilot: no source-specific signal')
ax.text(.5,-.22,'Path-agreement proxy only; actions were not executed and task success was not measured.',transform=ax.transAxes,ha='center',fontsize=9,color='#555')
ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
fig.tight_layout(); save(fig,'offline_evidence_proxy.png')

# 6) conceptual runtime evidence loop
fig,ax=plt.subplots(figsize=(11,3.5)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
boxes=[
 (.025,.39,.19,.31,'Live UI state','sₜ: screenshot + hierarchy\nlayout / scroll / modal / values','#EAF2F8'),
 (.275,.39,.19,.31,'Parallel, one probe','safe candidate action a\nobserve successor s′ + ΔUI\nstrict restore gate','#FFF2DF'),
 (.525,.39,.19,.31,'Runtime transition graph','(sₜ, action slot, context)\n→ multiple observed s′ branches\nwith provenance / confidence','#E9F4EC'),
 (.775,.39,.19,.31,'One-call reasoner + verifier','one inference from snapshot\nconsume prior matched edge\ncompare executed post-state','#F2EBF5')]
for x0,y0,w,h,title,body,fc in boxes:
    p=FancyBboxPatch((x0,y0),w,h,boxstyle='round,pad=0.012,rounding_size=.02',fc=fc,ec='#8A969B',lw=1.2)
    ax.add_patch(p); ax.text(x0+w/2,y0+h*.72,title,ha='center',va='center',weight='bold',fontsize=10,color=DARK)
    ax.text(x0+w/2,y0+h*.36,body,ha='center',va='center',fontsize=8.5,color=DARK)
for start,end in [(.215,.275),(.465,.525),(.715,.775)]: ax.add_patch(FancyArrowPatch((start,.545),(end,.545),arrowstyle='-|>',mutation_scale=13,lw=1.4,color='#53666F'))
ax.text(.5,.14,'The current probe cannot alter the prompt already sent this round; its edge becomes evidence for later matched decisions.',ha='center',fontsize=9.3,color='#555')
ax.text(.5,.91,'Design loop: runtime uncertainty → live transition evidence → future reasoning and verification',ha='center',fontsize=12,weight='bold',color=DARK)
fig.tight_layout(); save(fig,'exploration_graph_reasoning_loop.png')

p1e0=p1['by_condition']['E0_no_evidence']['reference_path_consistent']
p1e1=p1['by_condition']['E1_fresh_outcome']['reference_path_consistent']
p1e2=p1['by_condition']['E2_fresh_child_affordances']['reference_path_consistent']
p1wrong=p1['by_condition']['D_wrong_source_control']['reference_path_consistent']
all_tasks=40+len(finished)
known_total=sum(outcomes76.values())
report=f'''# MobileExplorer 阶段性实验进度与设计判断

**数据截点：{now:%Y-%m-%d %H:%M UTC}（最后状态事件 {status_rows[-1].get('time') if status_rows else 'unknown'}）。** safety-v1 one-call acquisition profile 已覆盖 116/116 task；这是阶段性设计与效用报告，模型为 GELAB-ZERO-4B pilot，不是 MAI-UI 2B 主结果，也不是 reasoning efficacy 结论。

## 当前进度

| 队列 | 进度 | 实际决策请求 | probe / 严格恢复 / 失败 | task outcome（已审计） |
|---|---:|---:|---:|---:|
| Frozen 40-task pilot | 40/40 | {exec40['decisions']}，全部一次推理 | {exp40['attempts']} / {exp40['recovered']} / {exp40['failed']} | 20 success、19 failure、1 protocol unknown |
| 76-task 扩展 | {len(finished)}/76 终态；{len(active)} active，{len(not_started)} 未启动 | {requests76}（{requests76}/{requests76} 实际请求各一次推理） | {probes76} / {restored76} / {failed76} | {outcomes76.get('success',0)} success、{outcomes76.get('failure',0)} failure、{outcomes76.get('unknown',0)} unknown（仅已终态） |
| 合计 | {40+len(finished)}/116 有终态 | {exec40['decisions']+requests76} | {exp40['attempts']+probes76} / {exp40['recovered']+restored76} / {exp40['failed']+failed76} | 51 success、54 failure、11 unknown；不作增益结论 |

76-task 扩展使用冻结的 disjoint cohort、`CoverageRanker`、单次推理、并行 exploration/reasoning、最多每决策 probe 一次和严格恢复。两个批次合计覆盖完整 116-task registry。恢复失败时 stale action 被丢弃并安全等待；这一点保证安全，但也损失该轮 agent action。逐 probe trace 按代码内 `layout_signature` 重新审计后，pilot 为 {ui40['layout_changed']}/{ui40['attempts']}、扩展为 {ui76['layout_changed']}/{ui76['attempts']}，合计 {ui_all['layout_changed']}/{ui_all['attempts']} 次改变 interaction skeleton；**0 次改变 Android Activity**。前一版报告中的 23/101 缺少明确可复核的定义，现已更正为按统一签名重算的结果。

完整 116-task profile 覆盖 {full_app_count} 个 app；历史 baseline 为 {full_baseline_mix.get('success',0)} success、{full_baseline_mix.get('failure',0)} failure、{full_baseline_mix.get('unknown',0)} unknown，难度为 low/middle/high {full_complexity_mix.get('low',0)}/{full_complexity_mix.get('middle',0)}/{full_complexity_mix.get('high',0)}。扩展 cohort 的 10 个 unknown 中，{unknown_breakdown.get('action protocol error',0)} 个是 action protocol error、{unknown_breakdown.get('evaluator protocol error',0)} 个是 evaluator protocol error、{unknown_breakdown.get('task initialization error',0)} 个是 initialization error，均不计为 task failure。

![阶段进度和已结束任务的审计结果](figures/current_progress_20261013/progress_and_outcomes.png)

一句话结论：全量 116 个 task 已结束，正式 outcome 是 51 success、54 failure、11 unknown；它们描述该 4B pilot 的结果，不代表 exploration 的因果增益。

![116-task profile 的 baseline outcome 与难度构成](figures/current_progress_20261013/cohort_profile.png)

一句话结论：样本覆盖 53 个历史 baseline 成功、62 个失败和 1 个未知任务，并含 62/31/23 个 low/middle/high 难度任务。

![Probe 严格恢复率](figures/current_progress_20261013/probe_recovery.png)

一句话结论：245 次 probe 中 224 次严格恢复成功（91.4%），21 次失败仍是并行探索的主要运行风险。

![基于真实 AndroidWorld 截图的 UI 状态示例与 trace 定义](figures/current_progress_20261013/ui_structure_real_examples.png)

一句话结论：真实 trace 中 BrowserMaze 的一次 ScrollView probe 在同一 `FilesActivity` 内产生了不同的 interaction skeleton；运行保存了源截图和后态 hierarchy/signature，但没有保存后态截图。

P0 的受控诊断 smoke 另有完整真实截图：Markor 的 To-Do tab probe 从 source 到 successor 保持同一 Activity，之后 authoritative replay 到达相同 successor；该图不是 116-task profile 的 probe 统计样本。

![Markor 的 source、实际 probe successor 与 authoritative replay successor 截图](figures/p0_actual_gui_example.png)

一句话结论：这张真实前后截图展示了“同一 Activity、不同 runtime UI、可复现的 action 后态”这一机制样例。

## 补充机制实验

![P0 exact-source/action-slot/replay 机制 smoke](figures/current_progress_20261013/p0_exact_match_validation.png)

一句话结论：3 个受控场景均通过 source 检查、证据检索、错误 source/action 拒绝、精确 successor replay 与 cleanup，且没有调用 VLM。

![单次推理并行探索 live smoke](figures/current_progress_20261013/onecall_live_smoke.png)

一句话结论：19/19 次决策请求都只做一次推理，8 次 probe 中 7 次严格恢复，但两任务的成功/失败与历史 baseline 一致，不能说明性能增益。

## 目前能回答什么

### Exploration 应该采集哪类信息？

目前最有根据的候选是 **当前 runtime source 下，一个语义 action slot 的真实后果**，不是泛化的页面摘要。本文将 UI structure change 操作性定义为：

`layout_signature(source) != layout_signature(probe successor)`，且两端 Activity 相同。

`layout_signature` 是交互骨架指纹：只用 actionable controls 的 class/resource-id；单例控件保留 64px 粗粒度位置，重复控件忽略位置；排除文本以及键盘/系统覆盖层。这样滚动列表中的项目数量/文本不改变“这是哪类页面”，而单例导航控件或布局位置变化仍可区分不同交互结构。相较之下，`structural_signature` 包含文本、selected 和 16px 量化 bounds，会对滚动和内容更新敏感；本报告单独审计它为 {ui_all['structural_changed']}/{ui_all['attempts']} 次变化，不把它等同于 UI structure change。

- source 的 Activity 加结构指纹/截图、滚动与弹窗等瞬时状态，以及关键可见值；
- action slot 的语义身份（resource-id 优先，其次唯一文本/描述/控件类型），并记录本次 hierarchy 中的实际 bounds；
- 点击/滚动后 successor 的结构差分、状态值变化、目标控件变化和系统条件；
- probe 是否恢复、后态是否由真实正式 action 重现，以及 observation 的时间、条件、来源和置信度。

实测证据支持“runtime 信息确实不同于 Activity”：{ui_all['layout_changed']}/{ui_all['attempts']} 次 probe 的 interaction skeleton 变化均发生在 Activity 未变时。但它**尚未证明哪些信息能提升 reasoning**。完整 116-task profile 有 {evidence_all['retrieval_events']} 次 evidence 检索、{evidence_all['eligible_candidates']} 个累计 eligible 候选，{evidence_all['selected_records']} 条事实曾进入 prompt；可用的 {evidence_all['alignment_audits']} 轮 action-slot 审计中，只有 {evidence_all['exact_source_action_successor_matches']} 轮实现 source/action/successor 精确匹配（40-task pilot: {evidence40['exact_source_action_successor_matches']}，76-task extension: {evidence76['exact_source_action_successor_matches']}）。所以 acquisition profile 显示当前主要瓶颈是“有效 transition evidence 很少到达可验证的 reasoning decision”，而不是 evidence 已经带来效果。

![按统一定义统计的 probe 后 UI structure change](figures/current_progress_20261013/runtime_state_deltas.png)

一句话结论：138/245 次 probe（56.3%）改变了 UI interaction skeleton，而 Activity 全程不变，说明 Activity 标签不足以识别 runtime UI。

### 现有 reasoning 对照说明了什么？

24-state 离线 pilot 的参考路径一致率为：无 evidence {p1e0}/24、新鲜 transition {p1e1}/24、新鲜 transition 加子页面 affordance {p1e2}/24、错误 source 对照 {p1wrong}/24。没有执行 action，也没有任务成功标签；wrong-source 与 fresh 相同，**所以现在不能声称 exploration evidence 改善 reasoning**，更不能声称 task success 提升。它指出了下一个实验的关键控制：fresh-source 必须优于 no-evidence，且不能被 wrong-source 复制。

![离线 proxy 对照：这不是任务成功率](figures/current_progress_20261013/offline_evidence_proxy.png)

一句话结论：fresh evidence 的路径一致率只比 no-evidence 高 1/24，且 wrong-source 达到相同结果，因此离线 pilot 没有提供 source-specific reasoning 增益证据。

## Graph 应该如何设计

把图的基本对象定为带 provenance 的**runtime transition multigraph**：

`runtime source + action slot + execution context → observed successor branch + UI delta`

同一个 source/action 保留多个 successor，不覆盖旧事实。source 要区分精确 runtime observation 和较粗的 state family；family 级匹配只作为迁移假设，不能冒充 exact evidence。边上存 probe/replay/authoritative 来源、恢复状态、时间、环境条件和复现次数。正式 action 执行后，verifier 根据 source、slot 和实际 successor 把边标成 known outcome、new branch、mismatch 或 unmatched。这个 verifier 验证的是转移预测，不等同于任务目标正确。

位置问题通过“当前 hierarchy hit-test → 语义 ActionSlot → 当下 bounds”处理，而不是复用旧坐标。当前 pilot 已用于 action-slot 匹配/验证；跨 screenshot 自动重新定位旧 slot 还需要单独实现和测量，不能当作已验证功能。

![Runtime exploration、图和单次推理的设计闭环](figures/current_progress_20261013/exploration_graph_reasoning_loop.png)

一句话设计结论：图应保存带 provenance 的多分支 runtime transition，explorer 并行采集新边，后续单次推理读取匹配边并由正式后态验证。

## 如何用 graph 来 exploration

探索目标应当是**任务无关的转移不确定性降低**，而非猜当前 task 要去哪一页。先用硬安全门筛选可逆且可严格恢复的候选，再按单位时间/恢复成本的 expected information gain 选 action：

`Score(a | s,D) = EIG(a | s,D) × P(strict restore | s,a) / (T_probe + T_restore + λ·risk)`

其中 `EIG = H[P(o | s,a,D)] − E_o[H[P(o | s,a,D ∪ {{(s,a,o)}})]]`，`o` 是观测到的 successor 与 UI delta。可先把结果抽象成新页面/弹窗、滚动或布局变化、关键值变化、无变化和 novel/unknown 等结果类型，同时保留原始 runtime fingerprint；用 Dirichlet-categorical posterior 估计各类分支的概率，再计算期望熵下降。安全门独立于信息价值：不可逆操作不因“信息量高”而放行。当前 `CoverageRanker` 是待比较的启发式 coverage baseline，不是 EIG，也不是最终设计。任务无关 selector 负责买到转移信息；fresh/none/wrong-source 的实际 action/evaluator replay 再判断这类信息是否帮助 reasoning。

保持每个决策恰好一次 VLM inference：reasoner 用决策起点截图与此前已提交的匹配边；model-free explorer 同时做 probe、记录 successor/delta 并严格恢复。本轮新边不能反向改写已经发出的 prompt。若正式 action 与 probe slot 相同，正式后态顺便验证这条边；否则边留给之后匹配 source 的一次推理使用。恢复失败就丢弃 stale action，fresh capture 后再做下一次正常单次推理。

![采集到使用的 evidence funnel](figures/current_progress_20261013/evidence_funnel.png)

一句话结论：24 条 evidence 进入 prompt，但在 1,019 次可审计决策中仅 5 次精确匹配 source、action 和 successor，当前主要瓶颈是证据到决策的匹配和可用性。

## 结论边界与下一步

现有结果可以支撑：**动态 UI 需要 runtime-scoped transitions；exploration 应验证 action slot 的后果，graph 同时承担 evidence store 与 transition verifier。** 现有结果不能支撑：这些信息已经改善 reasoning 或任务完成率。

下一步按这个顺序推进：

1. acquisition profile 已完成 116/116。先统一 task-level provenance、unknown 分类、decision log 与 transition observation，汇总 evidence exposure 和 source/action/successor 精确匹配率；协议 unknown 不算任务失败。
2. 从 profile 中找调用前已有 exact source/action evidence 的 checkpoint；每种条件独立 replay，同一个决策只做一次推理并执行 action。至少比较 fresh evidence / no evidence / wrong-source，固定 Explorer budget，报告 evidence exposure、action-slot 正确性、后态分支、官方 evaluator、恢复成本。
3. 只有当 fresh 在实际执行/任务指标上优于 no evidence，且 wrong-source 无同等效果，才能回答“哪些信息帮助 reasoning”。若 eligible checkpoint 少，先改进可用率，不要用离线路径一致性代替。
4. 用 append-only 多分支图与 EIG selector 对比当前 CoverageRanker/random，测每 probe 的不确定性下降、branch coverage、recoverability 与匹配决策收益；不要只看 node/edge 数。
5. 之后再复验 MAI-UI 2B；本报告所有 live 数据均为 4B pilot。

可用于 meeting 的一句话：**Exploration is a runtime experiment: it validates what a safe action slot actually does, records the resulting UI transition as source-conditioned evidence, and lets later one-call reasoning both use and verify that evidence.**

数据快照：[data_interim_progress_20261013.json](data_interim_progress_20261013.json)；生成脚本：[generate_interim_progress_report.py](../generate_interim_progress_report.py)。
'''
(OUT / 'interim_progress_and_design_20261013_zh.md').write_text(report)

result_index=f'''# 实验结果图表与一句话结论

数据截点为 {now:%Y-%m-%d %H:%M UTC}。Live profile 使用 GELAB-ZERO-4B；这是 pilot 结果，不是 MAI-UI 2B 最终结果。下列离线 proxy 不等于执行正确率或 task success。

## P0：source/action-slot/replay 机制 smoke

![P0 机制结果](figures/current_progress_20261013/p0_exact_match_validation.png)

一句话结论：3/3 个受控导航场景通过 exact source、action-slot、evidence retrieval、错误 source/action 拒绝及 replay successor 校验，且 VLM 调用数为 0。

![Markor 的真实 source、probe successor 与 replay successor 截图](figures/p0_actual_gui_example.png)

一句话结论：Markor To-Do tab 的真实截图对证明了同 Activity 内 UI successor 可被实际 probe 并被正式 replay 重现。

## 两任务 one-call 并行探索 live smoke

![请求完整性、恢复与 task outcome](figures/current_progress_20261013/onecall_live_smoke.png)

一句话结论：19/19 个请求保持单次推理且 7/8 个 probe 恢复，但两任务 outcome 与历史 baseline 相同，样本只验证运行链路。

## 完整 116-task acquisition profile

![任务进度与 outcome](figures/current_progress_20261013/progress_and_outcomes.png)

一句话结论：40-task pilot 与 76-task extension 均跑完，最终 outcome 为 51 success、54 failure、11 unknown，不能从中推断 exploration 的增益。

![样本 baseline 与复杂度构成](figures/current_progress_20261013/cohort_profile.png)

一句话结论：样本覆盖 53 个历史成功、62 个历史失败、1 个未知以及 low/middle/high 难度 62/31/23。

![Probe 严格恢复](figures/current_progress_20261013/probe_recovery.png)

一句话结论：245 次 profile probes 中 224 次严格恢复成功（91.4%），剩余 21 次是当前安全与可用性短板。

![统一定义下的 runtime UI structure change](figures/current_progress_20261013/runtime_state_deltas.png)

一句话结论：按 interaction-skeleton 签名统计，138/245 次 probe 改变 UI 结构（56.3%），且 245 次都没有改变 Activity。

![真实 UI 截图与 BrowserMaze trace 状态](figures/current_progress_20261013/ui_structure_real_examples.png)

一句话结论：BrowserMaze trace 显示相同 `FilesActivity` 内滚动改变 layout signature，但其 post-probe screenshot 没有保存，图中后态来自 hierarchy/signature。

![证据从检索到决策的 funnel](figures/current_progress_20261013/evidence_funnel.png)

一句话结论：24 条事实被选进 prompt，但 1,019 次 action-slot audit 仅 5 次精确 source/action/successor 匹配，证据可用率仍低。

## 24-state 离线 evidence-utility pilot

![No-evidence、fresh、fresh+affordance 与 wrong-source proxy](figures/current_progress_20261013/offline_evidence_proxy.png)

一句话结论：fresh 仅比 no-evidence 多 1/24 条参考路径一致，wrong-source 与 fresh 持平，因此没有 source-specific utility 证据。

## 设计图（不是实验结果）

![Runtime exploration、图和单次推理的设计闭环](figures/current_progress_20261013/exploration_graph_reasoning_loop.png)

一句话设计结论：并行 explorer 验证 action-slot 后果并追加多分支 runtime 图，未来匹配到该 source 的单次推理读取它，正式后态再验证它。

细节、定义与设计含义见[完整阶段报告](interim_progress_and_design_20261013_zh.md)；原始计数见[data snapshot](data_interim_progress_20261013.json)。
'''
(OUT / 'experiment_results_figures_and_conclusions_20261013_zh.md').write_text(result_index)
print(json.dumps({k:snapshot[k] for k in ['generated_at_utc','cutoff_latest_event_utc','pilot40','extension76']},indent=2,ensure_ascii=False))
print('REPORT', OUT / 'interim_progress_and_design_20261013_zh.md')
print('FIGURES', sorted(p.name for p in FIG.glob('*.png')))
