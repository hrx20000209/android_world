"""Zero-cost analyses for the paper's 'Understanding MobileExplorer's performance' section (plan N1-N5).

N1 graph-execution precision: after a graph-executed block, does the model undo it (Back as its next action)?
N2 budget adaptation: exploration window vs decision time on Orin (e2e116n) and on the LabServer (seenE2).
N3 prompt length: prompt tokens / ttft per call, BASE vs FULL45 (v45 injects no evidence into the prompt).
N4 safety filter: candidates dropped by predicted reversibility / risky-control rules, probes that changed data,
   rollback failures.
N5 graph growth vs savings: graph size before each task (task order) and BASE - FULL45 calls of that task.
Writes figures/understand/extra_summary.json.
"""
import collections
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows_of(tag, arms):
  rows = {}
  for l in open(ROOT / "reports" / f"{tag}_runs.jsonl"):
    r = json.loads(l)
    if r["arm"] not in arms:
      continue
    if r.get("error"):
      rows.pop((r["task"], r["arm"]), None)
    else:
      rows[(r["task"], r["arm"])] = r
  return rows


def jl(tag, arm, task, f):
  p = ROOT / "raw" / tag / arm / task / "p0" / f"{f}.jsonl"
  return [json.loads(l) for l in open(p)] if p.exists() else []


def mean(x):
  return round(st.mean(x), 3) if x else None


def pct(x, q):
  x = sorted(x)
  return round(x[min(len(x) - 1, int(q * len(x)))], 3) if x else None


out = {}
rows = rows_of("e2e116n", ["BASE", "FULL45"])
order = open(ROOT / "reports" / "e2e116n_task_order.txt").read().strip().split(",")
tasks = [t for t in order if (t, "BASE") in rows and (t, "FULL45") in rows]

# ---------------------------------------------------------------- N1
undo = collections.Counter()
for t in tasks:
  S = jl("e2e116n", "FULL45", t, "steps")
  for i, s in enumerate(S):
    if s["kind"] == "model":
      continue
    if i + 1 < len(S) and S[i + 1]["kind"] != "model":
      continue                                     # end of a graph block only
    kind = s["kind"]
    nxt = next((x for x in S[i + 1:] if x["kind"] == "model"), None)
    undo[(kind, "n")] += 1
    if nxt is not None and (nxt.get("action") or {}).get("action_type") == "navigate_back":
      undo[(kind, "undone")] += 1
base_back = sum((s.get("action") or {}).get("action_type") == "navigate_back"
                for t in tasks for s in jl("e2e116n", "FULL45", t, "steps") if s["kind"] == "model")
base_n = sum(1 for t in tasks for s in jl("e2e116n", "FULL45", t, "steps") if s["kind"] == "model")
out["N1_graph_exec_undo"] = {k: {"blocks": undo[(k, "n")], "undone_next": undo[(k, "undone")]}
                             for k in sorted({k for k, _ in undo})}
out["N1_model_back_rate"] = round(base_back / base_n, 3)


# ---------------------------------------------------------------- N2
def windows(tag, arm, ts):
  W = []
  for t in ts:
    S = {s["step"]: s for s in jl(tag, arm, t, "steps") if s["kind"] == "model"}
    for w in jl(tag, arm, t, "windows"):
      s = S.get(w["step"])
      if s is None or w.get("stop") in ("disabled", "transient_state", "dialog_state"):
        continue
      W.append({"win": w.get("window_s", 0.0), "dec": s.get("model_latency_s") or 0.0, "n": len(w.get("probes", [])),
                "stop": w.get("stop") or "plan_exhausted"})
  P = [w for w in W if w["n"]]
  return {"windows_eligible": len(W), "windows_with_probes": len(P), "probes": sum(w["n"] for w in W),
          "decision_s_mean": mean([w["dec"] for w in W]), "window_s_mean_probed": mean([w["win"] for w in P]),
          "util_median_probed": pct([w["win"] / w["dec"] for w in P if w["dec"]], 0.5),
          "overrun": sum(w["win"] > w["dec"] for w in P), "stops": dict(collections.Counter(w["stop"] for w in P))}


rs = rows_of("seenE2", ["FULL45"])
out["N2_budget"] = {"orin_e2e116n": windows("e2e116n", "FULL45", tasks),
                    "labserver_seenE2": windows("seenE2", "FULL45", sorted({t for t, _ in rs}))}

# ---------------------------------------------------------------- N3
for arm in ("BASE", "FULL45"):
  G = [g for t in tasks for g in jl("e2e116n", arm, t, "gen")]
  out.setdefault("N3_prompt", {})[arm] = {"calls": len(G), "prompt_tokens_mean": mean([g["prompt_tokens"] for g in G if g.get("prompt_tokens")]),
                                          "ttft_s_mean": mean([g["ttft_s"] for g in G if g.get("ttft_s")]),
                                          "completion_tokens_mean": mean([g["completion_tokens"] for g in G if g.get("completion_tokens")])}

# ---------------------------------------------------------------- N4
agg = collections.Counter()
outc = collections.Counter()
for t in tasks:
  for w in jl("e2e116n", "FULL45", t, "windows"):
    agg["p_rev_dropped"] += w.get("p_rev_dropped") or 0
    agg["risky_dropped"] += w.get("risky_dropped") or 0
    agg["transient_skipped"] += w.get("stop") == "transient_state"
    agg["dialog_skipped"] += w.get("stop") == "dialog_state"
    for p in w.get("probes", []):
      outc[(p.get("outcome"), bool(p.get("rollback_ok")))] += 1
G = ROOT / "graph_snapshots" / "e2e116n" / "FULL45"
eff = [json.loads(l) for l in open(G / "effectful.jsonl")]
unr = [json.loads(l) for l in open(G / "unrestorable.jsonl")]
n_probe = sum(outc.values())
out["N4_safety"] = {"candidates_dropped": dict(agg), "probes": n_probe,
                    "probe_outcomes": {f"{o}|rollback_ok={ok}": n for (o, ok), n in outc.most_common()},
                    "rollback_fail": sum(n for (o, ok), n in outc.items() if not ok),
                    "learned_effectful_controls": len(eff), "learned_unrestorable_controls": len(unr),
                    "icon_only_among_learned": sum(1 for e in eff + unr if not any(x for x in e["key"][-1].split("|")[3:]))}

# ---------------------------------------------------------------- N5
inst_first = {}
nodes, edges, growth = set(), set(), {}
for l in open(G / "graph.jsonl"):
  e = json.loads(l)
  t = e.get("instance", "").split("/")[0]
  growth.setdefault(t, (len(nodes), len(edges)))   # size BEFORE the first record of task t
  src = (e.get("src") or {}).get("activity"), (e.get("src") or {}).get("layout_sig")
  dst = (e.get("dst") or {}).get("activity"), (e.get("dst") or {}).get("layout_sig")
  nodes.update([src, dst])
  edges.add((src, e.get("slot_id"), dst))
both = [t for t in tasks if rows[(t, "BASE")]["success"] and rows[(t, "FULL45")]["success"]]
bins = collections.defaultdict(list)
for t in both:
  n_e = growth.get(t, (len(nodes), len(edges)))[1]
  b = "<200" if n_e < 200 else "200-600" if n_e < 600 else ">=600"
  bins[b].append(rows[(t, "BASE")]["calls"] - rows[(t, "FULL45")]["calls"])
out["N5_graph_growth"] = {"final_nodes": len(nodes), "final_edges": len(edges),
                          "saved_calls_by_edges_before_task": {b: {"n": len(v), "saved_mean": mean(v)} for b, v in sorted(bins.items())}}

json.dump(out, open(ROOT / "figures" / "understand" / "extra_summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
