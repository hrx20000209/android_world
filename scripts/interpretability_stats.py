#!/usr/bin/env python3
"""可解释性统计：机制触发次数 / 图规模 / 步数，按任务复杂度分箱。

AndroidWorld 没有 difficulty 标签，但每个任务类有 complexity（1–12），
官方步数预算就是 10 × complexity（suite_utils.py:525）。按它分三箱：
  易   complexity <= 1.4   (67 个任务)
  中   1.4 < complexity <= 2.4  (26)
  难   complexity > 2.4    (23)
"""
import csv, glob, json, os, re, sys, collections, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R = "results/mobileexplorer_runs"


def task_meta():
  from android_world import registry
  reg = registry.TaskRegistry().get_registry(
      registry.TaskRegistry.ANDROID_WORLD_FAMILY)
  out = {}
  for name, cls in reg.items():
    c = float(getattr(cls, "complexity", 1) or 1)
    out[name] = dict(complexity=c,
                     bin="易" if c <= 1.4 else ("中" if c <= 2.4 else "难"))
  return out


def read(path, meta):
  rows = []
  for f in glob.glob(os.path.join(path, "*.log")):
    t = os.path.basename(f)[:-4]
    x = open(f, errors="ignore").read()
    if "Connection refused" in x or "fast_a11y_socket_unavailable" in x:
      continue
    s = re.findall(r"over (\d+) steps", x)
    if not s:
      continue
    r = dict(task=t, bin=meta.get(t, {}).get("bin", "?"),
             complexity=meta.get(t, {}).get("complexity"),
             success=int("Task Successful" in x), steps=int(s[-1]),
             reasoning_calls=0, skip_bootstrap=0, skip_gate=0, skip_prefill=0,
             injections=0, probes=0, explore_rounds=0,
             prefix_checks=0, repairs=0,
             graph_nodes=0, graph_edges=0, executed_edges=0,
             revisited_states=0, unique_states=0)
    ev = os.path.join(path, t, "serial_events.jsonl")
    if os.path.exists(ev):
      for line in open(ev, errors="ignore"):
        try:
          e = json.loads(line)
        except Exception:
          continue
        k = e.get("kind")
        if k == "inference":
          r["reasoning_calls"] += 1
        elif k == "skip":
          d = e.get("kind_detail")
          if d == "app_bootstrap":
            r["skip_bootstrap"] += 1
          elif d == "store_prefill":
            r["skip_prefill"] += 1
        elif k == "gate":
          if e.get("mode") == "SKIP_INFERENCE":
            r["skip_gate"] += 1
          elif e.get("mode") == "GRAPH_ENHANCED_INFERENCE":
            r["injections"] += 1
        elif k == "graph_context" and e.get("injected"):
          r["injections"] += 1
        elif k == "explore":
          # 探测事件的 kind 是 "explore"。此前统计写成 "probe"，全表恒为 0。
          r["probes"] += 1
          r["explore_rounds"] += 1
        elif k == "prefix_check":
          r["prefix_checks"] += 1
        elif k == "repair":
          r["repairs"] += 1
    g = os.path.join(path, t, "progressive_belief_graph.json")
    if os.path.exists(g):
      try:
        d = json.load(open(g))
        nodes = d.get("nodes") or []
        edges = d.get("edges") or []
        r["graph_nodes"] = len(nodes)
        r["graph_edges"] = len(edges)
        r["executed_edges"] = sum(
            1 for e in edges if e.get("execution_hit_count", 0) > 0)
        vc = [n.get("visit_count", 0) for n in nodes]
        r["unique_states"] = len({n["layout_signature"] for n in nodes})
        r["revisited_states"] = sum(1 for v in vc if v >= 2)
      except Exception:
        pass
    r["total_skips"] = r["skip_bootstrap"] + r["skip_gate"] + r["skip_prefill"]
    rows.append(r)
  return rows


def table(name, rows):
  print(f"\n### {name}  (n={len(rows)})")
  hdr = (f"{'难度':>4s} {'n':>4s} {'成功率':>8s} {'步/任务':>8s} {'成功均步':>9s} "
         f"{'推理调用':>8s} {'跳过合计':>8s} {'引导':>5s} {'门跳过':>6s} {'预填':>5s} "
         f"{'注入':>5s} {'探测':>5s} {'图节点':>6s} {'重访态':>6s}")
  print(hdr)
  for b in ("易", "中", "难", "全部"):
    sel = rows if b == "全部" else [r for r in rows if r["bin"] == b]
    if not sel:
      continue
    n = len(sel); s = sum(r["success"] for r in sel)
    ok = [r["steps"] for r in sel if r["success"]]
    def m(k): return sum(r[k] for r in sel) / n
    print(f"{b:>4s} {n:4d} {100*s/n:6.1f}% {m('steps'):8.2f} "
          f"{(sum(ok)/len(ok) if ok else 0):9.2f} {m('reasoning_calls'):8.2f} "
          f"{m('total_skips'):8.2f} {m('skip_bootstrap'):5.2f} {m('skip_gate'):6.2f} "
          f"{m('skip_prefill'):5.2f} {m('injections'):5.2f} {m('probes'):5.2f} "
          f"{m('graph_nodes'):6.2f} {m('revisited_states'):6.2f}")


ARMS = [("v40（历史）", f"{R}/v40"),
        ("v48（图中那行）", f"{R}/v48"),
        ("full（本session复现）", f"{R}/v40x/full_h1"),
        ("noinject 不注入不跳过", f"{R}/v40x/noinject_h1"),
        ("rand 随机探索", f"{R}/v40x/rand_h1"),
        ("hitrate 提高触发率", f"{R}/v40x/hitrate_h1")]

if __name__ == "__main__":
  meta = task_meta()
  os.makedirs("ablation_results", exist_ok=True)
  allrows = []
  for label, path in ARMS:
    if not os.path.isdir(path):
      continue
    rows = read(path, meta)
    if not rows:
      continue
    for r in rows:
      r["variant"] = label
    allrows += rows
    table(label, rows)
  if allrows:
    with open("ablation_results/interpretability.csv", "w", newline="") as fh:
      w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys()))
      w.writeheader(); w.writerows(allrows)
    print(f"\n写出 ablation_results/interpretability.csv （{len(allrows)} 行）")
