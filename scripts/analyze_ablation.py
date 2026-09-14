#!/usr/bin/env python3
"""消融汇总：读原始运行日志，产出论文用的 CSV 与中文表。

口径（与图中三列一致）：
  成功率        success / 可判定任务数
  平均步数      所有任务的 AndroidWorld 步数均值
  成功任务步数  仅成功任务的步数均值

可判定 = 日志里同时有 "Task Successful/Failed" 和 "over N steps"，
且不含 "Connection refused"（隧道断开期间跑的任务一律作废）。
"""
import csv, json, glob, os, re, sys, statistics

R = "results/mobileexplorer_runs"
ASK = re.compile(r"(answer with|answer the following|what (is|are|do|does)"
                 r"|how many|do i have|which |tell me)", re.I)


def read_arm(path):
  """每个任务一行的原始记录。"""
  rows = []
  for f in sorted(glob.glob(os.path.join(path, "*.log"))):
    task = os.path.basename(f)[:-4]
    txt = open(f, errors="ignore").read()
    if "Connection refused" in txt:
      continue
    if "Task Successful" in txt:
      ok = True
    elif "Task Failed" in txt:
      ok = False
    else:
      continue
    steps = re.findall(r"over (\d+) steps", txt)
    if not steps:
      continue
    goal = re.search(r'with goal "(.*?)"', txt, re.S)
    goal = goal.group(1).strip() if goal else ""
    wall = re.findall(r"total=([0-9.]+)s", txt)
    lat = re.findall(r"avg_step=([0-9.]+)s", txt)
    rec = dict(task=task, success=int(ok), steps=int(steps[-1]),
               e2e_s=float(wall[-1]) if wall else None,
               step_latency_s=float(lat[-1]) if lat else None,
               kind="question" if ASK.search(goal) else "action",
               num_reasoning_calls=0, num_prefill_hops=0,
               num_prefill_hits=0, explicit_skipped_reasoning=0,
               num_rollbacks=0, rollback_restored=0,
               graph_nodes_after=None, num_unique_states=None)
    ev = os.path.join(path, task, "serial_events.jsonl")
    if os.path.exists(ev):
      for line in open(ev, errors="ignore"):
        try:
          e = json.loads(line)
        except Exception:
          continue
        k = e.get("kind")
        if k == "inference":
          rec["num_reasoning_calls"] += 1
        elif k == "skip":
          d = e.get("kind_detail")
          if d == "store_prefill":
            rec["num_prefill_hops"] += 1
            rec["num_prefill_hits"] += bool(e.get("matched"))
          elif d == "app_bootstrap":
            rec["explicit_skipped_reasoning"] += 1
        elif k == "prefill_rollback":
          rec["num_rollbacks"] += 1
          rec["rollback_restored"] += bool(e.get("restored"))
        elif k == "gate" and e.get("mode") == "SKIP_INFERENCE":
          rec["explicit_skipped_reasoning"] += 1
    g = os.path.join(path, task, "progressive_belief_graph.json")
    if os.path.exists(g):
      try:
        d = json.load(open(g))
        rec["graph_nodes_after"] = len(d.get("nodes") or ())
        rec["num_unique_states"] = len(
            {n["layout_signature"] for n in d.get("nodes") or ()})
      except Exception:
        pass
    rows.append(rec)
  return rows


def summarise(label, rows, note=""):
  n = len(rows)
  if not n:
    return None
  s = sum(r["success"] for r in rows)
  succ = [r["steps"] for r in rows if r["success"]]
  hops = sum(r["num_prefill_hops"] for r in rows)
  hits = sum(r["num_prefill_hits"] for r in rows)
  return dict(
      variant=label, n_tasks=n, success=s,
      success_rate=round(100.0 * s / n, 2),
      avg_steps=round(sum(r["steps"] for r in rows) / n, 2),
      avg_success_steps=round(sum(succ) / len(succ), 2) if succ else None,
      median_steps=statistics.median([r["steps"] for r in rows]),
      avg_reasoning_calls=round(
          sum(r["num_reasoning_calls"] for r in rows) / n, 2),
      total_reasoning_calls=sum(r["num_reasoning_calls"] for r in rows),
      explicit_skipped_reasoning=sum(
          r["explicit_skipped_reasoning"] for r in rows),
      prefill_hops=hops, prefill_hit_rate=round(100.0 * hits / hops, 1) if hops else 0.0,
      prefill_tasks=sum(1 for r in rows if r["num_prefill_hops"]),
      rollbacks=sum(r["num_rollbacks"] for r in rows),
      avg_e2e_s=round(sum(r["e2e_s"] for r in rows if r["e2e_s"]) /
                      max(sum(1 for r in rows if r["e2e_s"]), 1), 1),
      note=note)


ARMS = [
    ("Baseline 不建图 (h1)", f"{R}/full_nograph/h1", "无图、无预填"),
    ("Baseline 不建图 (h2)", f"{R}/full_nograph/h2", ""),
    ("Baseline 不建图 (h3)", f"{R}/full_nograph/h3", "同机基线"),
    ("MobileExplorer 完整 (h1)", f"{R}/g_lean/h1", "三模块全开"),
    ("MobileExplorer 完整 (h2)", f"{R}/g_lean/h2", ""),
    ("MobileExplorer+同类门 (h1)", f"{R}/g_kind/h1", "修问答类泄漏"),
    ("w/o 目标条件检索", f"{R}/ablate/nogoal/h1", "退回唯一控件门"),
    ("w/o 通过率门", f"{R}/ablate/nopass/h1", "岔路口也预填"),
    ("w/o 落点回退", f"{R}/ablate/noback/h1", "落点不符不撤销"),
    ("上一代 full_lean (h1)", f"{R}/full_lean/h1", "暖store(泄漏)+窄门"),
]

if __name__ == "__main__":
  os.makedirs("ablation_results", exist_ok=True)
  summaries, raw = [], []
  for label, path, note in ARMS:
    rows = read_arm(path)
    if not rows:
      continue
    for r in rows:
      r["variant"] = label
      raw.append(r)
    s = summarise(label, rows, note)
    if s:
      summaries.append(s)
  with open("ablation_results/per_task_raw.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(raw[0].keys()))
    w.writeheader(); w.writerows(raw)
  with open("ablation_results/summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
    w.writeheader(); w.writerows(summaries)
  hdr = f"{'方法':30s} {'成功率':>16s} {'平均步数':>9s} {'成功任务步数':>12s} {'推理调用':>9s} {'预填':>12s}"
  print(hdr); print("-" * len(hdr))
  for s in summaries:
    pf = f"{s['prefill_hops']}跳/{s['prefill_hit_rate']:.0f}%" if s["prefill_hops"] else "—"
    print(f"{s['variant']:30s} {s['success_rate']:6.2f}% ({s['success']:3d}/{s['n_tasks']:3d}) "
          f"{s['avg_steps']:9.2f} {str(s['avg_success_steps']):>12s} "
          f"{s['avg_reasoning_calls']:9.2f} {pf:>12s}")
  print(f"\n原始逐任务: ablation_results/per_task_raw.csv ({len(raw)} 行)")
  print(f"汇总:       ablation_results/summary.csv")
