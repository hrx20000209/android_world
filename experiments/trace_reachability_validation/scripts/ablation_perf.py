"""Ablation analysis (paper 'Understanding MobileExplorer'): BASE/FULL45 from e2e116n, three replacement ablations from e2e116nA (Orin)."""
import json, collections, statistics as st
A = ["BASE", "FULL45", "FULL45+RAND", "FULL45+INJ", "FULL45+NOSYS"]
TAG = {"BASE": "e2e116n", "FULL45": "e2e116n"}
rows = {}
for tag, arms in (("e2e116n", A[:2]), ("e2e116nA", A[2:])):
  for l in open(f"reports/{tag}_runs.jsonl"):
    r = json.loads(l)
    if r["arm"] not in arms: continue
    if r.get("error"): rows.pop((r["task"], r["arm"]), None)
    else: rows[(r["task"], r["arm"])] = r
def jl(arm, t, f):
  try: return [json.loads(l) for l in open(f"raw/{TAG.get(arm,'e2e116nA')}/{arm}/{t}/p0/{f}.jsonl")]
  except FileNotFoundError: return []
T = sorted(t for t in {t for t, _ in rows} if all((t, a) in rows for a in A))
ALL5 = [t for t in T if all(rows[(t, a)]["success"] for a in A)]
m = lambda L: round(st.mean(L), 2) if L else None
def decomp(arm, t):
  S = [s for s in jl(arm, t, "steps") if s.get("pre")]
  D = collections.Counter()
  for i, s in enumerate(S):
    nxt = S[i + 1]["pre"]["t"] if i + 1 < len(S) else None
    if s["kind"] == "model":
      D["decide"] += s["t_decision"] - s["pre"]["t"]
      if nxt: D["after"] += nxt - s["t_decision"]
    elif nxt: D["graph"] += nxt - s["pre"]["t"]
  D["other"] = rows[(t, arm)]["e2e_s"] - sum(D.values())
  D["calls"] = sum(s["kind"] == "model" for s in S); D["graph_steps"] = sum(s["kind"] != "model" for s in S)
  G = jl(arm, t, "gen"); D["decode_full"] = sum(g.get("t_end_s", 0) for g in G); D["ttft"] = sum(g.get("ttft_s", 0) for g in G)
  return D
out = {"paired": len(T), "all5": len(ALL5), "arms": {}}
for a in A:
  S = [t for t in T if rows[(t, a)]["success"]]
  cut = lambda L: {"n": len(L), "calls": m([rows[(t, a)]["calls"] for t in L]), "e2e": m([rows[(t, a)]["e2e_s"] for t in L])}
  dec = collections.Counter()
  for t in ALL5: dec.update(decomp(a, t))
  kinds = collections.Counter(s["kind"] for t in T for s in jl(a, t, "steps") if s["kind"] != "model")
  probes = sum(rows[(t, a)].get("probes", 0) for t in T)
  out["arms"][a] = {"success": len(S), "all": cut(T), "succ": cut(S), "all5": cut(ALL5),
                    "decomp_all5": {k: round(v / len(ALL5), 2) for k, v in dec.items()},
                    "graph_kinds": dict(kinds), "probes_total": probes}
# success flips vs FULL45
fl = {}
for a in A:
  if a == "FULL45": continue
  fl[a] = {"FULL45_only": [t for t in T if rows[(t, "FULL45")]["success"] and not rows[(t, a)]["success"]],
           "arm_only": [t for t in T if rows[(t, a)]["success"] and not rows[(t, "FULL45")]["success"]]}
out["flips_vs_FULL45"] = fl
# per-task call deltas on tasks both FULL45 and arm succeeded
out["pair_vs_FULL45"] = {}
for a in A:
  if a == "FULL45": continue
  P = [t for t in T if rows[(t, "FULL45")]["success"] and rows[(t, a)]["success"]]
  d = [(t, rows[(t, a)]["calls"] - rows[(t, "FULL45")]["calls"], round(rows[(t, a)]["e2e_s"] - rows[(t, "FULL45")]["e2e_s"], 1)) for t in P]
  out["pair_vs_FULL45"][a] = {"n": len(P), "d_calls": m([x[1] for x in d]), "d_e2e": m([x[2] for x in d]),
                               "top": sorted(d, key=lambda x: -x[2])[:6]}
json.dump(out, open("figures/understand/ablation_summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
