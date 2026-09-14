#!/usr/bin/env python3
"""扫所有归档运行，找出"只差一个 flag"的臂对，逐对算配对差。

只在两臂共同完成的任务上比（配对口径），并且要求两臂的关键实验条件一致
（步数预算、模型端点形态）。跨步数预算的对比会被标出但不混入结论。
"""
import json, glob, os, re, collections, itertools, sys

R = "results/mobileexplorer_runs"
# 只关心这些会改变机制的 flag
KEYS = ["probes_per_step", "graph_context", "enable_skip", "bootstrap",
        "store_prefill", "node_summary", "exploration_policy", "max_depth",
        "prefill_min_tasks", "max_prefill", "decision_constraints",
        "prefill_retrieval", "prefill_pass_rate", "prefill_rollback",
        "prefill_after_bootstrap", "prefill_risk", "prefill_sources",
        "prefill_any_kind", "prefill_by_resource_id", "depth2_needs_known_inverse"]


def arms():
  out = {}
  for d in sorted(glob.glob(f"{R}/*/*/")):
    logs = glob.glob(d + "*.log")
    if len(logs) < 60:
      continue
    ra = sorted(glob.glob(d + "*/run_args.json"))
    if not ra:
      continue
    try:
      a = json.load(open(ra[0]))
    except Exception:
      continue
    cfg = {k: a["args"].get(k) for k in KEYS}
    cfg["max_steps"] = a["args"].get("max_steps")
    res = {}
    for f in logs:
      t = os.path.basename(f)[:-4]
      x = open(f, errors="ignore").read()
      if "Connection refused" in x or "fast_a11y_socket_unavailable" in x:
        continue
      s = re.findall(r"over (\d+) steps", x)
      if not s:
        continue
      res[t] = ("Task Successful" in x, int(s[-1]))
    if len(res) >= 60:
      out[d.rstrip("/").replace(R + "/", "")] = (cfg, res, a.get("started_at", "")[:10])
  return out


def main():
  A = arms()
  print(f"可用臂 {len(A)} 个（各 ≥60 个有效任务）\n")
  found = []
  for x, y in itertools.combinations(sorted(A), 2):
    cx, rx, dx = A[x]; cy, ry, dy = A[y]
    if cx.get("max_steps") != cy.get("max_steps"):
      continue                       # 步数预算不同，不可比
    diff = [k for k in cx if cx[k] != cy[k] and k != "max_steps"]
    if len(diff) != 1:
      continue
    k = diff[0]
    common = sorted(set(rx) & set(ry))
    if len(common) < 60:
      continue
    sx = sum(rx[t][0] for t in common); sy = sum(ry[t][0] for t in common)
    ax = sum(rx[t][1] for t in common) / len(common)
    ay = sum(ry[t][1] for t in common) / len(common)
    ox = [rx[t][1] for t in common if rx[t][0]]
    oy = [ry[t][1] for t in common if ry[t][0]]
    found.append(dict(flag=k, a=x, va=cx[k], b=y, vb=cy[k], n=len(common),
                      sa=sx, sb=sy, d_succ=sx - sy,
                      steps_a=round(ax, 2), steps_b=round(ay, 2),
                      d_steps=round(ax - ay, 2),
                      ss_a=round(sum(ox)/max(len(ox),1), 2),
                      ss_b=round(sum(oy)/max(len(oy),1), 2),
                      budget=cx.get("max_steps"), date_a=dx, date_b=dy))
  found.sort(key=lambda r: (r["flag"], -abs(r["d_succ"])))
  print(f"{'flag':24s} {'A (值)':>26s} {'B (值)':>26s} {'n':>4s} {'成功 A/B':>10s} {'Δ成功':>6s} {'Δ步':>7s} {'预算':>5s}")
  print("-" * 126)
  for r in found:
    print(f"{r['flag']:24s} {r['a'][:16]+'('+str(r['va'])[:8]+')':>26s} "
          f"{r['b'][:16]+'('+str(r['vb'])[:8]+')':>26s} {r['n']:4d} "
          f"{r['sa']:4d}/{r['sb']:<4d} {r['d_succ']:+6d} {r['d_steps']:+7.2f} {str(r['budget']):>5s}")
  import csv
  os.makedirs("ablation_results", exist_ok=True)
  if found:
    with open("ablation_results/single_flag_pairs.csv", "w", newline="") as fh:
      w = csv.DictWriter(fh, fieldnames=list(found[0].keys()))
      w.writeheader(); w.writerows(found)
    print(f"\n共 {len(found)} 对 -> ablation_results/single_flag_pairs.csv")


if __name__ == "__main__":
  main()
