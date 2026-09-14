#!/usr/bin/env python3
"""论文用表：全部走配对口径（只算两臂共同完成的任务），避免任务集不同造成的假差异。"""
import csv, glob, json, os, re, sys

R = "results/mobileexplorer_runs"
ASK = re.compile(r"(answer with|answer the following|what (is|are|do|does)"
                 r"|how many|do i have|which |tell me)", re.I)


def load(path):
  out = {}
  for f in glob.glob(os.path.join(path, "*.log")):
    t = os.path.basename(f)[:-4]
    x = open(f, errors="ignore").read()
    if "Connection refused" in x:
      continue
    s = re.findall(r"over (\d+) steps", x)
    if not s:
      continue
    g = re.search(r'with goal "(.*?)"', x, re.S)
    w = re.findall(r"total=([0-9.]+)s", x)
    out[t] = dict(ok="Task Successful" in x, steps=int(s[-1]),
                  e2e=float(w[-1]) if w else None,
                  kind="question" if (g and ASK.search(g.group(1))) else "action")
  return out


def mech(path, task):
  ev = os.path.join(path, task, "serial_events.jsonl")
  d = dict(reason=0, hops=0, hits=0, boot=0, gateskip=0, rb=0, nodes=0)
  if os.path.exists(ev):
    for l in open(ev, errors="ignore"):
      try:
        e = json.loads(l)
      except Exception:
        continue
      k = e.get("kind")
      if k == "inference":
        d["reason"] += 1
      elif k == "skip":
        det = e.get("kind_detail")
        if det == "store_prefill":
          d["hops"] += 1
          d["hits"] += bool(e.get("matched"))
        elif det == "app_bootstrap":
          d["boot"] += 1
      elif k == "prefill_rollback":
        d["rb"] += 1
      elif k == "gate" and e.get("mode") == "SKIP_INFERENCE":
        d["gateskip"] += 1
  g = os.path.join(path, task, "progressive_belief_graph.json")
  if os.path.exists(g):
    try:
      d["nodes"] = len(json.load(open(g)).get("nodes") or ())
    except Exception:
      pass
  return d


def compare(label, arm, ref, ref_label, kind=None):
  A, B = load(arm), load(ref)
  common = sorted(set(A) & set(B))
  if kind:
    common = [t for t in common if A[t]["kind"] == kind]
  if not common:
    return None
  def agg(D, path):
    ok = [t for t in common if D[t]["ok"]]
    m = [mech(path, t) for t in common]
    return dict(
        succ=len(ok), n=len(common),
        sr=round(100.0 * len(ok) / len(common), 2),
        steps=round(sum(D[t]["steps"] for t in common) / len(common), 2),
        succ_steps=round(sum(D[t]["steps"] for t in ok) / len(ok), 2) if ok else None,
        reason=round(sum(x["reason"] for x in m) / len(common), 2),
        hops=sum(x["hops"] for x in m), hits=sum(x["hits"] for x in m),
        boot=sum(x["boot"] for x in m), gateskip=sum(x["gateskip"] for x in m),
        rb=sum(x["rb"] for x in m),
        e2e=round(sum(D[t]["e2e"] for t in common if D[t]["e2e"]) /
                  max(sum(1 for t in common if D[t]["e2e"]), 1), 1))
  a, b = agg(A, arm), agg(B, ref)
  return dict(variant=label, ref=ref_label, n=len(common),
              success=a["succ"], ref_success=b["succ"], d_success=a["succ"] - b["succ"],
              success_rate=a["sr"], ref_success_rate=b["sr"],
              avg_steps=a["steps"], ref_avg_steps=b["steps"],
              d_steps=round(a["steps"] - b["steps"], 2),
              avg_success_steps=a["succ_steps"], ref_avg_success_steps=b["succ_steps"],
              avg_reasoning_calls=a["reason"], ref_avg_reasoning_calls=b["reason"],
              prefill_hops=a["hops"],
              prefill_hit_rate=round(100.0 * a["hits"] / a["hops"], 1) if a["hops"] else 0.0,
              explicit_skip_bootstrap=a["boot"], explicit_skip_gate=a["gateskip"],
              rollbacks=a["rb"], avg_e2e_s=a["e2e"], ref_avg_e2e_s=b["e2e"])


FULL = f"{R}/g_lean/h1"
BASE = f"{R}/full_nograph/h1"
ABL = [("w/o 目标条件检索", f"{R}/ablate/nogoal/h1"),
       ("w/o 通过率门", f"{R}/ablate/nopass/h1"),
       ("w/o 落点回退", f"{R}/ablate/noback/h1")]

if __name__ == "__main__":
  os.makedirs("ablation_results", exist_ok=True)
  rows = []
  r = compare("MobileExplorer 完整", FULL, BASE, "Baseline 不建图")
  if r: rows.append(r)
  for lab, p in ABL:
    r = compare(lab, p, FULL, "MobileExplorer 完整")
    if r: rows.append(r)
  with open("ablation_results/table_ablation.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
  print("表 1 — 消融（配对口径，只算两臂共同完成的任务）\n")
  print(f"{'方法':22s} {'n':>4s} {'成功':>12s} {'成功率':>8s} {'平均步数':>12s} {'成功任务步数':>14s} {'推理调用':>8s} {'预填':>11s}")
  print("-"*108)
  for r in rows:
    pf = f"{r['prefill_hops']}跳/{r['prefill_hit_rate']:.0f}%" if r["prefill_hops"] else "—"
    print(f"{r['variant']:22s} {r['n']:4d} {r['success']:4d} vs {r['ref_success']:<4d} "
          f"{r['success_rate']:7.2f}% {r['avg_steps']:6.2f} vs {r['ref_avg_steps']:<5.2f} "
          f"{str(r['avg_success_steps']):>6s} vs {str(r['ref_avg_success_steps']):<6s} "
          f"{r['avg_reasoning_calls']:8.2f} {pf:>11s}")
    print(f"{'':22s} {'':4s} Δ成功 {r['d_success']:+d} (对 {r['ref']})   Δ步 {r['d_steps']:+.2f}")
  # 按任务形态
  print("\n表 2 — 按任务形态拆分（完整设计 vs Baseline）")
  for k, nm in (("action", "执行类"), ("question", "问答类")):
    r = compare("完整", FULL, BASE, "Baseline", kind=k)
    if r:
      print(f"  {nm}  n={r['n']:3d}   完整 {r['success']:3d} ({r['success_rate']:5.2f}%)   "
            f"Baseline {r['ref_success']:3d} ({r['ref_success_rate']:5.2f}%)   Δ {r['d_success']:+d}   "
            f"步 {r['avg_steps']:5.2f} vs {r['ref_avg_steps']:5.2f}")
  print("\n写出: ablation_results/table_ablation.csv")
