"""Motivation (Fig. 4d), e2e116n FULL45: Jaccard similarity between the live screen and the UI memory, in the
same Structure / Content terms as Fig. 4(a).

For the first screen of each layout the model meets in a task, compare it with the memory built by earlier tasks
(same app), recorded either by model execution only or by model execution + exploration probes:
  Structure = max Jaccard of control skeletons (class | resource-id | flags, outside lists)
  Content   = max Jaccard of contents (list-row texts, field values, checkbox states)
95% CIs: bootstrap over tasks.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import content_freshness as F  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / "figures" / "motivation"
order = open(ROOT / "reports" / "e2e116n_task_order.txt").read().strip().split(",")
rank = {t: i for i, t in enumerate(order)}
obs = [json.loads(l) for l in open(ROOT / "graph_snapshots" / "e2e116n" / "FULL45" / "graph.jsonl")]
for o in obs:
  o["task"] = o["instance"].split("/")[0]
obs.sort(key=lambda o: (rank[o["task"]], o["t"]))
_SK = {}


def skel(task, cid):
  if (task, cid) not in _SK:
    try:
      _SK[(task, cid)] = frozenset(C.load_ui(str(F.RAW / task / "p0"), cid).skeleton())
    except FileNotFoundError:
      _SK[(task, cid)] = None
  return _SK[(task, cid)]


mem = collections.defaultdict(list)          # package -> (rank, origin, skeleton, content)
for o in obs:
  for side in ("src", "dst"):
    s, cid = o.get(side), o.get(side + "_cid")
    if s and cid and "launcher" not in s["package"] and not (side == "src" and o["origin"] == "explore"):
      sk, ct = skel(o["task"], cid), F.content(o["task"], cid)
      if sk is not None and ct is not None:
        org = "explore" if side == "dst" and o["origin"] == "explore" else "agent"
        mem[s["package"]].append((rank[o["task"]], org, sk, ct))

probed = collections.defaultdict(list)       # layout -> (rank, origin) of probe destinations
for o in obs:
  if o["origin"] == "explore" and o.get("dst") and "launcher" not in o["dst"]["package"]:
    probed[F.lkey(o["dst"])].append((rank[o["task"]], "explore"))
rows, seen = [], set()
for o in obs:
  if o["origin"] != "agent" or not o.get("src_cid") or "launcher" in o["src"]["package"]:
    continue
  L = F.lkey(o["src"])
  if (o["task"], L) in seen:
    continue
  seen.add((o["task"], L))
  sk, ct = skel(o["task"], o["src_cid"]), F.content(o["task"], o["src_cid"])
  if sk is None or ct is None:
    continue
  prev = [x for x in mem[o["src"]["package"]] if x[0] < rank[o["task"]]]
  r = {"task": o["task"], "probed_before": any(x[1] == "explore" and x[0] < rank[o["task"]]
                                               for x in probed.get(L, ()))}
  for k in ("model", "all"):
    pool = [x for x in prev if k == "all" or x[1] == "agent"]
    r[k + "_struct"] = max([F.jac(sk, x[2]) for x in pool], default=0.0)
    r[k + "_content"] = max([F.jac(ct, x[3]) for x in pool], default=0.0)
  rows.append(r)

rng = np.random.default_rng(0)
by_task = collections.defaultdict(list)
for r in rows:
  by_task[r["task"]].append(r)
tasks = list(by_task)


def stat(col):
  v = np.mean([r[col] for r in rows])
  boots = []
  for _ in range(2000):
    pick = rng.choice(len(tasks), len(tasks))
    vals = [r[col] for i in pick for r in by_task[tasks[i]]]
    boots.append(np.mean(vals))
  return v, np.percentile(boots, [2.5, 97.5])


res = {c: stat(c) for c in ("model_struct", "all_struct", "model_content", "all_content")}
summary = {c: {"mean": round(float(v), 3), "ci": [round(float(x), 3) for x in ci]} for c, (v, ci) in res.items()}
summary["screens"], summary["tasks"] = len(rows), len(tasks)
json.dump(summary, open(OUT / "memory_similarity.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
GRAY, PINK = "#b8bec6", "#d97fb4"
MODEL_ONLY = "--model-only" in sys.argv          # recommended for the paper: memory similarity without the comparison
if MODEL_ONLY:
  fig, ax = plt.subplots(figsize=(2.6, 2.2))
  for i, (lab, col, color) in enumerate((("Structure", "model_struct", GRAY), ("Content", "model_content", PINK))):
    v, ci = res[col]
    ax.bar(i, v, 0.5, color=color, yerr=[[v - ci[0]], [ci[1] - v]], capsize=3)
    ax.text(i, ci[1] + 0.02, f"{v:.2f}", ha="center", fontsize=8)
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Structure", "Content"])
  ax.set_ylim(0, 1.05)
  ax.set_ylabel("Jaccard similarity to\nscreens of earlier tasks")
  fig.tight_layout()
  for ext in ("pdf", "png"):
    fig.savefig(OUT / f"memory_similarity_model_only.{ext}", dpi=300, bbox_inches="tight")
  print("saved", OUT / "memory_similarity_model_only.png")
  sys.exit(0)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
w = 0.36
for i, (lab, a, b) in enumerate((("Structure", "model_struct", "all_struct"), ("Content", "model_content", "all_content"))):
  for j, (col, color, name) in enumerate(((a, GRAY, "model only"), (b, PINK, "model + exploration"))):
    v, ci = res[col]
    x = i + (j - 0.5) * w
    ax.bar(x, v, w, color=color, label=name if i == 0 else None, yerr=[[v - ci[0]], [ci[1] - v]], capsize=3)
    ax.text(x, ci[1] + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Structure", "Content"])
ax.set_ylim(0, 1.15)
ax.set_ylabel("Jaccard similarity\nto UI memory")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
fig.tight_layout()
for ext in ("pdf", "png"):
  fig.savefig(OUT / f"memory_similarity.{ext}", dpi=300, bbox_inches="tight")
print("saved", OUT / "memory_similarity.png")
