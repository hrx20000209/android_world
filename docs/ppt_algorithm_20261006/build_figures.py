"""Read existing experiment logs and export PPT figures; never run an agent."""
from pathlib import Path
import collections
import csv
import json
import os

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/mobileexplorer-ppt-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.font_manager import FontProperties

FONT = FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf').get_name()
plt.rcParams.update({'font.family': FONT, 'font.size': 13, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.unicode_minus': False, 'figure.facecolor': 'white'})
BLUE, GREEN, ORANGE, GRAY, RED = '#235DA3', '#20815F', '#D38A24', '#6D7884', '#BF4B45'
ROOT = REPO / 'results/sensys27_task_navigation/20261006_naive_full'
MODULE = REPO / 'results/sensys27_task_navigation/20261005_ablation_r1'
ARMS = ['base', 'naive_graph', 'naive_explore', 'naive_use', 'full_rec']
LABELS = ['Base', '朴素建图', '随机探索', '仅提示用图', 'Full']
COLORS = [GRAY, ORANGE, ORANGE, ORANGE, BLUE]

def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def pairs(path):
    return {r['task']: r for r in jsonl(path / 'pairs.jsonl')}

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=190, bbox_inches='tight')
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)

data = pairs(ROOT)
saved = json.loads((ROOT / 'naive_ablation_report.json').read_text())
summary = {}
for arm in ARMS:
    rows = [row[arm] for row in data.values() if arm in row and not row[arm].get('crash')]
    kinds = collections.Counter()
    for p in (ROOT / arm).glob('*/api_calls.jsonl'):
        kinds.update(row.get('purpose', 'step') for row in jsonl(p))
    stats = {
        'tasks': len(rows), 'score': sum(r['ok'] for r in rows),
        'mean_steps': sum(r['steps'] for r in rows) / len(rows),
        'total_calls': sum(r['calls'] for r in rows),
        'gui_calls': kinds['step'], 'contract_calls': kinds['obligation'],
        'mean_serial_runtime_s': sum(r['runtime'] for r in rows) / len(rows),
        'nodes_per_episode': saved[arm]['nodes_per_episode'],
        'revisit_share_gated': saved[arm]['revisit_share'] if arm != 'base' and arm != 'naive_use' else None,
        'probes': saved[arm]['probes'], 'probe_edges_later_used': saved[arm]['probe_edges_later_used'],
        'replay_hops': saved[arm]['replay_hops'], 'skips': saved[arm]['skips'],
    }
    assert stats['tasks'] == saved[arm]['tasks'] == 116
    assert stats['score'] == saved[arm]['score']
    assert abs(stats['mean_steps'] - saved[arm]['mean_steps']) < 1e-10
    assert stats['total_calls'] == saved[arm]['inference_calls'] == sum(kinds.values())
    summary[arm] = stats
(OUT / 'verified_metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
with (OUT / 'verified_metrics.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=['arm'] + list(next(iter(summary.values()))))
    writer.writeheader()
    writer.writerows({'arm': arm, **row} for arm, row in summary.items())

# Figure 1: process actually used by the serial full_rec experiments.
fig, ax = plt.subplots(figsize=(15.6, 10.3))
ax.set_xlim(0, 15.6); ax.set_ylim(0, 10.3); ax.axis('off')
def box(x, y, w, h, text, color=BLUE, size=13):
    ax.add_patch(Rectangle((x,y), w,h, facecolor=color+'12', edgecolor=color, linewidth=1.4))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color='#203046',linespacing=1.55)
def arrow(x1,y1,x2,y2,text='', color=GRAY):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=14,color=color,lw=1.5))
    if text:
        ax.text((x1+x2)/2,(y1+y2)/2+.13,text,ha='center',va='bottom',fontsize=11,color=color)
ax.text(.4,10.0,'full_rec 当前执行流程',fontsize=24,weight='bold',color=BLUE)
ax.text(.4,9.65,'116 任务实验使用串行 runner；探索结束并恢复后才推理',fontsize=13,color=GRAY)
box(.5,8.25,6.7,1.0,'① 结算上一模型动作\n前后状态检查，记录流程 hop 与提交边界')
box(.5,6.85,6.7,1.0,'② 读取当前 UI，登记屏幕节点\n读取探索开始前的图快照，解析上一步推理需求')
box(.5,5.45,6.7,1.0,'③ 流程图规划与实体 / 字段回放\n有已完成演示且下一实体可定位，执行到提交前',GREEN)
box(.5,4.05,6.7,1.0,'④ 屏幕图跳步门\n重访、可复用边、风险 / 提交守卫 / 次数限制')
box(.5,2.65,6.7,1.0,'⑤ 蒸馏图上下文并完成探索\n过滤、排序、探测、恢复、窗口状态检查')
box(.5,1.25,6.7,1.0,'⑥ 模型读取恢复后的 UI，推理并执行\n模型决定最终提交，或处理回放中断')
box(.5,.08,6.7,.75,'⑦ 写入真实屏幕转移，准备 pending hop，提交下一代')
for top, bottom in [(8.25,7.85),(6.85,6.45),(5.45,5.05),(4.05,3.65),(2.65,2.25),(1.25,.83)]:
    arrow(3.85,top,3.85,bottom)
box(8.25,7.55,6.7,1.7,'屏幕图 ProgressiveBeliefGraph\nactivity + 交互骨架的节点\n控件动作边，落点、恢复、代价与历史统计',BLUE)
box(8.25,5.15,6.7,1.8,'流程图 ProcedureGraph\n模型演示的动作序列，实体 / 字段槽位\nstate_ok，已处理实体，当前进行中的实体',GREEN)
arrow(8.25,5.95,7.2,5.95,'规划')
box(8.25,3.6,6.7,.95,'跳步允许：执行一跳，记录结果，结束本 step',ORANGE)
arrow(7.2,4.55,8.25,4.05,'允许')
box(8.25,1.25,6.7,1.6,'契约调用时机\n首次发现“演示片段 + 持久状态变化”之后\n额外一次纯文本调用，抽取任务实体与字段',GRAY)
save(fig, '01_algorithm_flow')

# Figure 2: instance and slot semantics, including typed fields.
fig, ax = plt.subplots(figsize=(15.6, 7.4))
ax.set_xlim(0,15.6); ax.set_ylim(0,7.4); ax.axis('off')
ax.text(.4,7.05,'在线流程模板与提交边界',fontsize=24,weight='bold',color=BLUE)
ax.text(.4,6.65,'例：RecipeDeleteMultipleRecipesWithNoise；r7 真正执行的回放后缀为 4 跳',fontsize=13,color=GRAY)
names=['输入实体名','打开对应菜谱','More options','Delete 菜单','DELETE 确认']
for j,name in enumerate(names):
    x=.45+j*3.08
    box(x,4.5,2.68,1.12,name,RED if j==4 else GREEN,14)
    ax.text(x+1.34,4.18,'state_ok=False' if j==4 else 'state_ok=True',ha='center',fontsize=12,color=RED if j==4 else GREEN)
    if j<4: arrow(x+2.68,5.06,x+3.08,5.06)
ax.text(.5,3.55,'第一个实体：模型完成演示，检查器记录覆盖范围内的持久状态变化',fontsize=15,color=BLUE)
box(.45,1.6,11.93,1.1,'第二 / 第三个实体：替换实体槽位，按实时 selector 重定位并回放这 4 跳',GREEN,15)
box(12.77,1.6,2.68,1.1,'模型确认提交',RED,15)
arrow(12.38,2.15,12.77,2.15)
ax.text(.5,.8,'records 模式还可替换金额、类别、描述等字段，并回放表单内滑动；它已超出纯导航',fontsize=14,color=GRAY)
ax.text(.5,.3,'证书来自演示时的检查。回放逐跳检查 activity / selector / 实体，未对新实体每跳重新出具持久状态证书',fontsize=12,color=GRAY)
save(fig, '02_procedure_boundary')

# Figure 3: paired-task measures, with calls and serial time kept distinct.
fig, axes = plt.subplots(2,2,figsize=(15.4,9.5))
panels=[('score','官方评估分数和 / 116','{:.1f}',(0,75)),
        ('mean_steps','平均 agent 决策步数','{:.2f}',(0,11)),
        ('total_calls','总 API 调用数（包含契约）','{:.0f}',(0,1260)),
        ('mean_serial_runtime_s','日志 runtime 均值，串行 / 秒','{:.1f}',(0,105))]
for ax,(key,title,fmt,lim) in zip(axes.flat,panels):
    vals=[summary[a][key] for a in ARMS]
    bars=ax.bar(range(5),vals,color=COLORS,width=.6)
    ax.set_xticks(range(5),LABELS); ax.set_title(title,loc='left',pad=19,fontsize=17)
    ax.set_ylim(*lim); ax.grid(axis='y',alpha=.16); ax.set_axisbelow(True)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,val+(lim[1]*.02),fmt.format(val),ha='center',fontsize=13)
fig.suptitle('完整设计与三个朴素组：116 任务',fontsize=24,x=.07,ha='left')
fig.text(.07,.025,'seed 30，单次；Base / Full 复用 r7，朴素组复用 29 个既有 episode 并补跑其余 87 个。不同指标不能互换。',fontsize=12,color=GRAY)
fig.subplots_adjust(top=.88,bottom=.12,hspace=.45,wspace=.23)
save(fig,'03_naive_comparison_116')

# Figure 4: graph/reuse diagnostics, and request breakdown.
fig, axes=plt.subplots(1,3,figsize=(16,5.8))
small=['naive_graph','naive_explore','full_rec']; lab=['朴素建图','随机探索','Full']
for ax,key,title,percent in [(axes[0],'nodes_per_episode','平均屏幕节点数',False),
                            (axes[1],'revisit_share_gated','有 gate 记录的重访比例',True)]:
    vals=[summary[a][key]*(100 if percent else 1) for a in small]
    ax.bar(range(3),vals,color=[ORANGE,ORANGE,BLUE],width=.55)
    ax.set_xticks(range(3),lab); ax.set_title(title,fontsize=17,pad=16)
    ax.set_ylim(0,max(vals)*1.23); ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    for i,v in enumerate(vals): ax.text(i,v+max(vals)*.035,f'{v:.1f}'+('%' if percent else ''),ha='center')
ax=axes[2]
for i,a in enumerate(['base','full_rec']):
    s=summary[a]
    ax.bar(i,s['gui_calls'],color=BLUE,width=.5)
    ax.bar(i,s['contract_calls'],bottom=s['gui_calls'],color=ORANGE,width=.5)
    ax.text(i,s['total_calls']+30,str(s['total_calls']),ha='center')
    ax.text(i,s['gui_calls']/2,str(s['gui_calls']),ha='center',color='white',fontsize=16)
    if s['contract_calls']: ax.text(i+.35,s['gui_calls']+s['contract_calls']/2,'45 契约',fontsize=12)
ax.set_xticks([0,1],['Base','Full']); ax.set_ylim(0,1250); ax.set_title('GUI 调用与额外契约调用',fontsize=17,pad=16)
fig.suptitle('图复用与真实调用成本',fontsize=24,x=.05,ha='left')
fig.text(.05,.025,'重访比例只在 gate 事件中统计；Base / 仅提示组没有 gate，汇总中的 0 不代表没有重访。',fontsize=12,color=GRAY)
fig.subplots_adjust(top=.79,bottom=.16,wspace=.37)
save(fig,'04_graph_and_calls')

# Figure 5: module deletion subset, fixed identical tasks.
moddata=pairs(MODULE)
tasks=['ExpenseDeleteMultiple','ExpenseDeleteMultiple2','RecipeDeleteMultipleRecipes','RecipeDeleteMultipleRecipesWithNoise']
modarms=['base','full_rec','no_replay','no_skip','no_context','no_explore','explore_untargeted']
modlabels=['Base','Full','去掉回放','去掉跳步','去掉上下文','去掉探索','无实体定向']
modcsv=[]
fig,axes=plt.subplots(1,2,figsize=(15.4,6.7),gridspec_kw={'width_ratios':[1.8,1]})
left,right=axes
totals=[sum(moddata[t][a]['steps'] for t in tasks) for a in modarms]
left.barh(range(7),totals,color=[GRAY,BLUE,RED,GRAY,GRAY,GRAY,GRAY],height=.57)
left.set_yticks(range(7),modlabels);left.invert_yaxis();left.set_xlim(0,64)
left.set_title('同四个删除任务的决策步数合计',fontsize=17,loc='left',pad=20)
for i,v in enumerate(totals): left.text(v+1,i,f'{v:g}',va='center',fontsize=14)
ms=json.loads((MODULE/'naive_ablation_report.json').read_text())
scores=[ms[a]['score'] for a in modarms]
right.barh(range(7),scores,color=[GRAY,BLUE,RED,GRAY,GRAY,GRAY,GRAY],height=.57)
right.set_yticks(range(7),modlabels);right.invert_yaxis();right.set_xlim(0,19)
right.set_title('29 任务的评估分数和',fontsize=17,loc='left',pad=20)
for i,v in enumerate(scores):right.text(v+.3,i,f'{v:g}',va='center',fontsize=14)
for i,a in enumerate(modarms):
    modcsv.append({'arm':a,'delete_four_steps':totals[i],'score_29':scores[i]})
with (OUT/'module_ablation_29.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(modcsv[0]));writer.writeheader();writer.writerows(modcsv)
fig.suptitle('去掉哪个模块，重复流程的节省会消失',fontsize=24,x=.07,ha='left')
fig.text(.07,.035,'r6 代码，29 个任务，单次模块删除实验。四个删除任务在这些组里均成功；分数波动不构成显著性证明。',fontsize=12,color=GRAY)
fig.subplots_adjust(top=.79,bottom=.15,left=.15,wspace=.43)
save(fig,'05_module_ablation_29')
print(json.dumps({'output':str(OUT),'verified':summary},ensure_ascii=False,indent=2))
