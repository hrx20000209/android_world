"""Complete comparison table of all full arms on one denominator (episodes errored in any arm excluded)."""
import glob, json, statistics as st, sys
from collections import Counter

ARMS = sys.argv[1:] or ["BASE", "FULL23", "FULL23R", "FULL23W", "FULL28", "FULL28R", "FULL29", "FULL29X", "FULL30", "FULL31"]
G = {"bootstrap": "启动", "confirm": "确认", "function": "函数", "function_route": "函数", "function_entry": "函数",
     "form_function": "表单", "graph_macro": "宏", "graph_reuse": "repeat"}


def load(a):
  out = {}
  for f in glob.glob(f"raw/full/{a}/*/p0/result.json"):
    d = f.rsplit("/", 1)[0]; t = f.split("/")[3]
    r = json.load(open(f))
    kinds = Counter()
    try:
      for l in open(d + "/steps.jsonl"):
        k = json.loads(l).get("kind")
        if k in G:
          kinds[G[k]] += 1
    except FileNotFoundError:
      pass
    pr = [json.loads(l) for l in open(d + "/probes.jsonl")] if glob.glob(d + "/probes.jsonl") else []
    out[t] = dict(ok=r.get("is_successful") == 1.0, calls=r.get("model_calls"), err=bool(r.get("error")) or r.get("model_calls") is None,
                  g=kinds, probes=len(pr), rbfail=sum(not p.get("rollback_ok") for p in pr), requery=r.get("m_requery") or 0)
  return out


A = {a: load(a) for a in ARMS}
full = [a for a in ARMS if len(A[a]) >= 114]
bad = {t for a in ARMS for t, v in A[a].items() if v["err"]}
T = sorted(t for t in A["BASE"] if t not in bad and all(t in A[a] for a in full))
print(f"同口径 {len(T)} 题（剔除任一臂出错的 {len(bad)} 题）\n")
hdr = "| 配置 | 成功 | 平均推理 | 成功任务推理 | 与BASE共同成功（题数：BASE→本臂） | 图复用合计 | 启动边 | 确认边 | 参数化边(函数) | 模式边(表单) | 动作边:跨任务宏 | 动作边:本局重复 | 探索次数 | 回滚失败 | requery |"
print(hdr); print("|" + "---|" * hdr.count("|")[:-1] if False else "|---" * (hdr.count("|") - 1) + "|")
for a in ARMS:
  ts = [t for t in T if t in A[a]]
  o = A[a]
  part = "" if len(ts) == len(T) else f"（进行中 {len(ts)} 题）"
  s = [o[t]["calls"] for t in ts if o[t]["ok"]]
  allc = [o[t]["calls"] for t in ts]
  both = [t for t in ts if o[t]["ok"] and A["BASE"][t]["ok"]]
  pb = f"{len(both)}：{st.mean(A['BASE'][t]['calls'] for t in both):.2f}→{st.mean(o[t]['calls'] for t in both):.2f}" if both and a != "BASE" else "–"
  g = Counter()
  for t in ts:
    g.update(o[t]["g"])
  tot = sum(g.values())
  pr = sum(o[t]["probes"] for t in ts); rf = sum(o[t]["rbfail"] for t in ts); rq = sum(o[t]["requery"] for t in ts)
  print(f"| {a}{part} | {len(s)} | {st.mean(allc):.2f} | {st.mean(s):.2f} | {pb} | {tot} | {g['启动']} | {g['确认']} | {g['函数']} | {g['表单']} | {g['宏']} | {g['repeat']} | {pr} | {rf} | {rq} |")
