#!/usr/bin/env python3
"""Generate figures from MobileExplorer validation artifacts."""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "topk_evidence_pilot_20261013"
PLOT_INPUTS = HERE / "plot_inputs"
PAIRED_SCORES = PLOT_INPUTS / "topk_evidence_pilot_20261013" / "paired_scores_sanitized.jsonl"
P0_AUDIT = PLOT_INPUTS / "p0_gate_audit_sanitized.json"
OUT = HERE / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((RUN / "summary.json").read_text())
raw = [json.loads(line) for line in PAIRED_SCORES.read_text().splitlines()]
audit = json.loads(P0_AUDIT.read_text())

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
conditions = ["E0_no_evidence", "E1_fresh_outcome", "E2_fresh_child_affordances", "D_wrong_source_control"]
names = ["No evidence", "Fresh outcome", "Fresh + child UI", "Wrong-source control"]
colors = ["#7B8794", "#007C91", "#4F9D69", "#C66B5D"]

# P0: exact-mechanism gate audit, 3 screens by 5 required checks.
gates = []
screen_names = []
for item in audit:
    graph, replay = item["graph"], item["authoritative_replay"]
    gates.append([
        int(item["probe"]["rollback"]["verified"]),
        int(graph["positive_retrieval"]),
        int(graph["negative_wrong_source_count"] == 0 and graph["negative_wrong_source_version_count"] == 0
            and graph["negative_wrong_action_rejected"]),
        int(replay["action_slot_match"]),
        int(replay["successor_exact_runtime_match"] and replay["successor_family_match"]),
    ])
    screen_names.append(item["scenario"]["name"].replace("_", " ").title())
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.imshow(gates, cmap=plt.matplotlib.colors.ListedColormap(["#D8DEE6", "#398B68"]), vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(5), ["Source restored", "Fresh edge retrieved", "Wrong source rejected", "Slot matched", "Replay successor matched"], rotation=14, ha="right")
ax.set_yticks(range(3), screen_names)
ax.set_title("P0 exact-source / action-slot / replay smoke (n = 3)", loc="left", weight="bold")
for y, row in enumerate(gates):
    for x, value in enumerate(row):
        ax.text(x, y, "PASS" if value else "FAIL", ha="center", va="center", color="white" if value else "#263445", weight="bold", fontsize=9)
ax.set_xticks(np.arange(-.5, 5, 1), minor=True); ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", bottom=False, left=False)
fig.tight_layout(); fig.savefig(OUT / "p0_exact_match_smoke.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# One raw device example makes the smoke audit visually inspectable.
example = HERE / "p0_exact_match_smoke_20261013" / "me-p0-smoke" / "markor_todo_tab"
image_paths = [example / "source.png", example / "probe_successor.png",
               example / "authoritative_successor.png"]
if all(path.exists() for path in image_paths):
    image_titles = ["Captured source", "After exact-slot probe", "After authoritative replay"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.7))
    for ax, path, title in zip(axes, image_paths, image_titles):
        ax.imshow(plt.imread(path)); ax.set_title(title, fontsize=11, weight="bold"); ax.axis("off")
    fig.suptitle("Actual Markor UI captures: To-Do tab probe and replay successor", x=.02, ha="left", weight="bold", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "p0_actual_gui_example.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# P1: parsing, visible-control grounding, and the historical path proxy.
metrics = ["parse_valid", "target_grounded", "reference_path_consistent"]
metric_names = ["Parse-valid", "UI-grounded", "Historical-path agreement"]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(3); width = .19
for i, (condition, name, color) in enumerate(zip(conditions, names, colors)):
    row = summary["by_condition"][condition]
    bars = ax.bar(x + (i - 1.5) * width, [row[m] / row["n"] * 100 for m in metrics], width, label=name, color=color)
    for bar, key in zip(bars, metrics):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f"{row[key]}/24", ha="center", va="bottom", fontsize=8, rotation=90)
ax.set_xticks(x, metric_names); ax.set_ylim(0, 119); ax.set_ylabel("States (%)")
ax.set_title("P1 offline pilot: fresh evidence did not raise the path-agreement proxy", loc="left", weight="bold")
ax.legend(frameon=False, ncol=2, loc="lower right"); ax.grid(axis="y", alpha=.2)
fig.tight_layout(); fig.savefig(OUT / "p1_outcome_metrics.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# P1: paired directional changes versus no-evidence, on the 22 parse-valid pairs.
baseline = {r["decision_id"]: r for r in raw if r["condition"] == conditions[0]}
paired_counts = []
for condition in conditions[1:]:
    counts = Counter()
    for r in raw:
        if r["condition"] != condition: continue
        a, b = baseline[r["decision_id"]]["score"], r["score"]
        if not a.get("parse_ok") or not b.get("parse_ok"):
            counts["excluded"] += 1; continue
        old, new = bool(a["reference_path_consistent"]), bool(b["reference_path_consistent"])
        if old == new: counts["same"] += 1
        elif new: counts["toward"] += 1
        else: counts["away"] += 1
    paired_counts.append(counts)
fig, ax = plt.subplots(figsize=(9.5, 4.5)); y = np.arange(3); left = np.zeros(3)
for label, key, color in [("Same", "same", "#8C98A4"), ("Toward reference", "toward", "#398B68"), ("Away from reference", "away", "#C66B5D")]:
    vals = np.array([c[key] for c in paired_counts]); ax.barh(y, vals, left=left, color=color, label=label, height=.62)
    for i, (start, value) in enumerate(zip(left, vals)):
        if value: ax.text(start+value/2, i, str(int(value)), ha="center", va="center", color="white", weight="bold")
    left += vals
ax.set_yticks(y, names[1:]); ax.invert_yaxis(); ax.set_xlim(0, 22)
ax.set_xlabel("Comparable pairs (n = 22; two baseline parse failures excluded)")
ax.set_title("Fresh and wrong-source evidence show no directional advantage", loc="left", weight="bold")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, -.33)); ax.grid(axis="x", alpha=.2)
fig.tight_layout(); fig.savefig(OUT / "p1_paired_direction.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# P1: model-token and wall-clock cost measured from the two live vLLM services.
latencies = [summary["by_condition"][c]["mean_latency_s"] for c in conditions]
tokens = [summary["by_condition"][c]["mean_input_tokens"] for c in conditions]
fig, ax1 = plt.subplots(figsize=(9.5, 4.6)); x = np.arange(4)
bars = ax1.bar(x-.18, latencies, .36, color=colors, label="Mean latency (s)")
ax1.set_ylabel("Mean response latency (s)"); ax1.set_xticks(x, names, rotation=12, ha="right")
ax2 = ax1.twinx(); ax2.bar(x+.18, tokens, .36, color=colors, alpha=.4, hatch="//", label="Mean input tokens")
ax2.set_ylabel("Mean prompt tokens")
for bar, val in zip(bars, latencies): ax1.text(bar.get_x()+bar.get_width()/2, val+.04, f"{val:.2f}s", ha="center", fontsize=8)
for i, val in enumerate(tokens): ax2.text(i+.18, val+25, f"{val:.0f}", ha="center", fontsize=8)
ax1.set_title("P1 inference cost per decision (n = 24 per condition)", loc="left", weight="bold")
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax1.legend(h1+h2,l1+l2,frameon=False,loc="upper left"); ax1.grid(axis="y",alpha=.18)
fig.tight_layout(); fig.savefig(OUT / "p1_inference_cost.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# P2/P3: archive eligibility versus intervention evidence.
fig, ax = plt.subplots(figsize=(9.5, 4.5)); vals=[161,5,3,0]
labels=["Readable repeated-action\nreports", "Exact-source recovered\nsame-step cases", "Independent task\nepisodes", "Skip intervention\nmeasured"]
bars=ax.bar(np.arange(4),vals,color=["#7B8794","#007C91","#4F9D69","#C66B5D"],width=.62)
ax.set_xticks(np.arange(4),labels); ax.set_ylabel("Count"); ax.set_ylim(0,180)
ax.set_title("P2/P3: archive supports audit examples, not efficacy",loc="left",weight="bold")
for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+4,str(val),ha="center",weight="bold")
ax.text(.98,.74,"Five cases span one app (Settings);\nno controlled stagnation arm.",transform=ax.transAxes,ha="right",va="top",fontsize=9)
ax.text(.98,.54,"Skip/replay was disabled in the 116-task profile.",transform=ax.transAxes,ha="right",va="top",fontsize=9)
ax.grid(axis="y",alpha=.18);fig.tight_layout();fig.savefig(OUT/"p2_p3_evidence_gate.png",dpi=180,bbox_inches="tight");plt.close(fig)

# Final direction scorecard uses observed sample availability, not a predicted effect.
fig, ax = plt.subplots(figsize=(11, 5.7)); ax.axis("off")
columns = ["Direction", "Observed evidence", "Decision"]
rows = [
    ["One-call runtime acquisition", "Live smoke: 19/19 single-call decisions; 8 probes, 7 restored, 1 safe WAIT", "Mechanism only; continue 40-task pilot"],
    ["Evidence-guided correction", "P1 n=24; fresh 20/24 vs none 19/24; wrong-source 20/24", "NO-GO for efficacy claim; refine live paired test"],
    ["Stagnation intervention", "5 exact-source cases / 3 episodes / 1 app; no intervention", "Underpowered; collect independent episodes"],
    ["Verified navigation skip", "0 skip arms measured; old 116 profile disabled skip", "Not evaluated; require verified paths first"],
    ["Trace-guided Top-K", "K=1 only; K=2/3 and selector policy unavailable", "Defer until downstream utility is shown"],
]
table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left", colLoc="left",
                 colWidths=[.22, .47, .31])
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.8)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#D7DEE8")
    if r == 0:
        cell.set_facecolor("#243B53"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
    elif c == 2:
        cell.set_facecolor("#E5F3EA" if r == 1 else ("#F8E5E1" if r == 2 else "#F4F6F8"))
ax.set_title("Evidence-based direction gate (not a ranking of unmeasured methods)", loc="left", weight="bold", pad=16)
fig.tight_layout(); fig.savefig(OUT / "final_direction_gate.png", dpi=180, bbox_inches="tight"); plt.close(fig)

print(json.dumps({"figures": [str(p) for p in sorted(OUT.glob("*.png"))], "paired_direction": [dict(c) for c in paired_counts]}, indent=2))
