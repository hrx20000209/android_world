"""Where does FULL45's e2e saving come from? Paired e2e116n analysis for the paper's 'Understanding MobileExplorer' section.
(1) per-task latency decomposition: model wait vs non-model time; (2) inference calls removed, by graph-execution type;
(3) exploration window vs decision time (does exploration ever delay the action?); (4) rollback outcomes;
(5) savings vs a task's position inside its app group (how fast the graph becomes useful)."""
import json, glob, re, statistics as st, collections, sys
tag = sys.argv[1] if len(sys.argv) > 1 else "e2e116n"
RAW = f"raw/{tag}"
rows = {}
for l in open(f"reports/{tag}_runs.jsonl"):
  r = json.loads(l)
  if r.get("error"): rows.pop((r["task"], r["arm"]), None)
  else: rows[(r["task"], r["arm"])] = r
order = open(f"reports/{tag}_task_order.txt").read().split(",")
tasks = [t for t in order if (t, "BASE") in rows and (t, "FULL45") in rows]
both = [t for t in tasks if rows[(t, "BASE")]["success"] and rows[(t, "FULL45")]["success"]]
def steps(arm, t): return [json.loads(l) for l in open(f"{RAW}/{arm}/{t}/p0/steps.jsonl")]
def gen(arm, t):
  try: return [json.loads(l) for l in open(f"{RAW}/{arm}/{t}/p0/gen.jsonl")]
  except FileNotFoundError: return []
def jl(arm, t, f):
  try: return [json.loads(l) for l in open(f"{RAW}/{arm}/{t}/p0/{f}.jsonl")]
  except FileNotFoundError: return []
out = {}
# (1) decomposition on both-succeeded tasks
for arm in ("BASE", "FULL45"):
  D = collections.defaultdict(list)
  for t in both:
    r, S, G = rows[(t, arm)], steps(arm, t), gen(arm, t)
    model = sum(s.get("model_latency_s") or 0 for s in S if s["kind"] == "model")
    full = sum(g["t_end_s"] for g in G)
    prefill = sum(g["ttft_s"] for g in G)
    n_model = sum(s["kind"] == "model" for s in S)
    D["e2e"].append(r["e2e_s"]); D["agent"].append(r["agent_s"]); D["setup"].append(r["e2e_s"] - r["agent_s"])
    D["model_wait"].append(model); D["model_full"].append(full); D["prefill"].append(prefill)
    D["non_model"].append(r["agent_s"] - model); D["calls"].append(n_model); D["graph_steps"].append(len(S) - n_model)
    D["actions"].append(len(S))
  out[arm] = {k: round(st.mean(v), 2) for k, v in D.items()}
# (2) graph-executed steps by type (all tasks) and calls
kinds = collections.Counter(); per_task = []
for t in tasks:
  S = steps("FULL45", t)
  for s in S:
    if s["kind"] != "model": kinds[s["kind"]] += 1
out["graph_kinds_all"] = dict(kinds)
# (3) exploration window vs decision time
W = []
for t in tasks:
  S = {s["step"]: s for s in steps("FULL45", t) if s["kind"] == "model"}
  for w in jl("FULL45", t, "windows"):
    s = S.get(w["step"])
    if s is None or not w.get("probes"): continue
    W.append((w["window_s"], s.get("model_latency_s") or 0, len(w["probes"]), w.get("stop")))
over = [x for x in W if x[0] > x[1]]
out["explore"] = {"windows_with_probes": len(W), "probes": sum(x[2] for x in W),
                  "window_s_mean": round(st.mean(x[0] for x in W), 2), "decision_s_mean": round(st.mean(x[1] for x in W), 2),
                  "window_over_decision": len(over), "overrun_s_mean": round(st.mean(x[0] - x[1] for x in over), 2) if over else 0,
                  "stops": dict(collections.Counter(x[3] for x in W))}
LS = [x for t in tasks for x in jl("FULL45", t, "late_summary")]
out["summary_join_wait_s_mean"] = round(st.mean(x["join_wait_s"] for x in LS), 3) if LS else None
# (4) rollback outcomes per probe
P = [p for t in tasks for p in jl("FULL45", t, "probes")]
rb = collections.Counter(("ok" if p.get("rollback_ok") else "fail") for p in P)
out["rollback"] = dict(rb, outcomes=dict(collections.Counter(p.get("outcome") for p in P)),
                       t_probe_mean=round(st.mean(p.get("t_total_s", 0) for p in P), 2) if P else 0)
req = sum(1 for t in tasks for s in steps("FULL45", t) if s.get("requery"))
out["requery"] = req
# (5) savings vs position in app group (both-succeeded tasks)
def app(t):
  m = re.match(r"(Simple[A-Z][a-z]+|System|Expense|Markor|Recipe|Files|Contacts|Notes|Tasks|Retro|Vlc|OsmAnd|Browser|Camera|Clock|AudioRecorder|OpenApp|SaveCopy|Sports|Turn)", t)
  return m.group(1) if m else t
pos, seen = {}, collections.Counter()
for t in order:
  pos[t] = seen[app(t)]; seen[app(t)] += 1
B = collections.defaultdict(list)
for t in both:
  b = "1st" if pos[t] == 0 else ("2-3" if pos[t] <= 2 else "4+")
  B[b].append((rows[(t, "BASE")]["calls"], rows[(t, "FULL45")]["calls"], rows[(t, "BASE")]["e2e_s"], rows[(t, "FULL45")]["e2e_s"]))
out["by_position"] = {k: {"n": len(v), "calls": (round(st.mean(x[0] for x in v), 2), round(st.mean(x[1] for x in v), 2)),
                          "e2e": (round(st.mean(x[2] for x in v), 1), round(st.mean(x[3] for x in v), 1))} for k, v in sorted(B.items())}
print(json.dumps(out, indent=1, ensure_ascii=False))
