"""Same-task progress comparison of a running arm against reference arms (errored episodes excluded everywhere)."""
import glob, json, sys, statistics as st

def load(a):
  o = {}
  for f in glob.glob(f"raw/full/{a}/*/p0/result.json"):
    r = json.load(open(f))
    o[f.split("/")[3]] = (r.get("is_successful") == 1.0, r.get("model_calls"), bool(r.get("error")))
  return o

run, refs = sys.argv[1], sys.argv[2:]
arms = {a: load(a) for a in [run] + refs}
bad = {t for o in arms.values() for t, v in o.items() if v[2] or v[1] is None}
tasks = sorted(t for t in arms[run] if t not in bad and all(t in arms[a] for a in refs))
print(f"{run} 已完成 {len(arms[run])} 题，同题可比 {len(tasks)} 题（剔除出错 {len(bad & set(arms[run]))}）")
for a in [run] + refs:
  o = arms[a]
  s = [o[t][1] for t in tasks if o[t][0]]; allc = [o[t][1] for t in tasks]
  print(f"  {a:8s} 成功 {len(s):3d}/{len(tasks)}  平均推理 {st.mean(allc):.2f}  成功任务推理 {st.mean(s) if s else 0:.2f}")
