"""Full paired report of an e2e run: three cuts (all / successful / both successful), per app, discordant tasks,
mechanism counters, generation statistics and latency decomposition."""
import json, re, statistics as st, sys
from collections import OrderedDict, Counter
from pathlib import Path
tag, A = sys.argv[1], sys.argv[2].split(",")
rows = {}
for l in open(f"reports/{tag}_runs.jsonl"):
  r = json.loads(l)
  if r.get("error"):
    rows.pop((r["task"], r["arm"]), None)
  else:
    rows[(r["task"], r["arm"])] = r
order = open(f"reports/{tag}_task_order.txt").read().split(",")
tasks = [t for t in order if all((t, a) in rows for a in A)]
m = lambda L, k: st.mean(r[k] for r in L) if L else float("nan")  # noqa: E731
def app(t):
  mm = re.match(r"(Simple[A-Z][a-z]+|System|Expense|Markor|Recipe|Files|Contacts|Notes|Tasks|Retro|Vlc|OsmAnd|Browser|Camera|Clock|AudioRecorder|OpenApp|SaveCopy|Sports|Turn)", t)
  return mm.group(1) if mm else t
print("paired", len(tasks))
both = [t for t in tasks if all(rows[(t, a)]["success"] for a in A)]
for cut, T in (("全部", tasks), ("两组都成功", both)):
  for a in A:
    R = [rows[(t, a)] for t in T]
    print(cut, a, len(R), sum(r["success"] for r in R), "calls %.2f e2e %.1f agent %.1f" % (m(R, "calls"), m(R, "e2e_s"), m(R, "agent_s")))
for a in A:
  S = [rows[(t, a)] for t in tasks if rows[(t, a)]["success"]]
  print("成功任务", a, len(S), "calls %.2f e2e %.1f agent %.1f" % (m(S, "calls"), m(S, "e2e_s"), m(S, "agent_s")))
groups = OrderedDict()
for t in tasks:
  groups.setdefault(app(t), []).append(t)
print("--- per app")
for g, T in groups.items():
  print(g, len(T), " ".join("%s:%d/%.1f/%.0f" % (a, sum(rows[(t, a)]["success"] for t in T), m([rows[(t, a)] for t in T], "calls"), m([rows[(t, a)] for t in T], "e2e_s")) for a in A))
print("--- discordant")
for t in tasks:
  s = [rows[(t, a)]["success"] for a in A]
  if len(set(s)) > 1:
    print(t, s, [(rows[(t, a)]["calls"], rows[(t, a)]["e2e_s"]) for a in A])
print("--- mechanisms / generation")
for a in A:
  c = Counter(); G = []; JW = []
  for t in tasks:
    d = Path(f"raw/{tag}/{a}/{t}/p0")
    r = json.load(open(d / "result.json"))
    for k, v in r.items():
      if k.startswith("m_") and isinstance(v, (int, float)):
        c[k[2:]] += v
    G += [json.loads(l) for l in open(d / "gen.jsonl")] if (d / "gen.jsonl").exists() else []
    if (d / "late_summary.jsonl").exists():
      JW.append(sum(json.loads(l)["join_wait_s"] for l in open(d / "late_summary.jsonl")))
  g = lambda k: st.mean(x[k] for x in G if x.get(k) is not None)  # noqa: E731
  print(a, {k: c[k] for k in ("calls", "early_actions", "func_steps", "form_steps", "form_options", "macro_hops", "probes_total", "requery", "spec_hit", "spec_miss", "bootstrap", "stale_pixels", "jump_commit_steps") if c[k]},
        "ttft %.2f decode_tok_s %.1f tokens %.0f t_end %.2f tail %.2f think %.0f%% join_wait/ep %.1f" % (
          g("ttft_s"), g("decode_tok_s"), g("completion_tokens"), g("t_end_s"),
          st.mean(x["t_end_s"] - x["t_action_s"] for x in G if x.get("t_action_s")), 100 * st.mean(float(x["think"]) for x in G),
          st.mean(JW) if JW else 0))
