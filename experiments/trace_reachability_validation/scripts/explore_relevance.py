"""How relevant is an explored node to the current task? Offline check on an arm's probes.

Scores of an explored destination page n for task goal G (no model involved):
  rel_goal(n)  = TF-IDF cosine(goal operation terms, page content)   [explore_value.rel, global x graph-local IDF]
  rel_param(n) = TF-IDF cosine(goal parameter values, page content)  [dynamic information]
Label (hindsight): useful(n) = later in the same episode the model itself lands on n's page (activity + layout),
or (looser) on n's activity.
Reports AUC of each score and the useful-rate per score bin."""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402
import explore_value as EV  # noqa: E402

R = full9.R


class Ep:
  def __init__(s, t):
    s.args = type("a", (), {"task": t})()


def auc(pairs):
  pos = [s for s, y in pairs if y]
  neg = [s for s, y in pairs if not y]
  if not pos or not neg:
    return float("nan")
  wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
  return wins / (len(pos) * len(neg))


def main(arm):
  raw = Path("raw/full") / arm
  rows = []
  for d in sorted(glob.glob(str(raw / "*" / "p0"))):
    d = Path(d)
    if not (d / "probes.jsonl").exists():
      continue
    t = d.parts[-2]
    goal = json.load(open(d / "meta.json"))["goal"]
    fx = full9.FuncExec(Ep(t), goal)
    g, p = EV.goal_vector(goal, fx), EV.param_vector(fx)
    visits = []                      # (step, page key, activity) of pages the model stood on
    for l in open(d / "steps.jsonl"):
      s = json.loads(l)
      if s.get("pre") and s.get("kind") == "model":
        sm = s["pre"]["summary"]
        visits.append((s["step"], (sm["activity"], sm["layout_sig"]), sm["activity"]))
    for l in open(d / "probes.jsonl"):
      o = json.loads(l)
      dst = o.get("dst")
      if not dst or not o.get("dst_cid") or dst.get("package") != o["src"].get("package"):
        continue
      if (dst["activity"], dst["layout_sig"]) == (o["src"]["activity"], o["src"]["layout_sig"]):
        continue                     # in-page change: not a new node
      try:
        ui = R.load_ui(d, o["dst_cid"])
      except Exception:  # pylint: disable=broad-except
        continue
      toks = EV.page_tokens(ui)
      try:
        src_toks = Counter(EV.page_tokens(R.load_ui(d, o["src_cid"])))
      except Exception:  # pylint: disable=broad-except
        src_toks = Counter()
      new_toks = list((Counter(toks) - src_toks).elements())   # what the node adds beyond the page we stood on
      later = [v for v in visits if v[0] > o["step"]]
      rows.append({
          "rel_goal": EV.rel(g, toks), "rel_param": EV.rel(p, toks) if p else 0.0,
          "rel_diff": EV.rel(g, new_toks), "new_act": dst["activity"] != o["src"]["activity"],
          "useful_page": any(v[1] == (dst["activity"], dst["layout_sig"]) for v in later),
          "useful_act": any(v[2] == dst["activity"] and dst["activity"] != o["src"]["activity"] for v in later),
          "why": o.get("priority")})
  n = len(rows)
  print(f"[{arm}] explored new nodes: {n}; later visited by the model: page {sum(r['useful_page'] for r in rows)}"
        f" ({sum(r['useful_page'] for r in rows) / max(1, n):.1%}), activity {sum(r['useful_act'] for r in rows)}")
  for score in ("rel_goal", "rel_param"):
    for lab in ("useful_page", "useful_act"):
      print(f"  AUC {score:9s} vs {lab:11s} = {auc([(r[score], r[lab]) for r in rows]):.3f}")
  print(f"  AUC rel_diff  vs useful_page = {auc([(r['rel_diff'], r['useful_page']) for r in rows]):.3f}")
  for na in (True, False):
    sub = [r for r in rows if r["new_act"] == na]
    print(f"  {'new-activity' if na else 'same-activity'} nodes {len(sub)}: visited {sum(r['useful_page'] for r in sub)}"
          f" ({sum(r['useful_page'] for r in sub) / max(1, len(sub)):.1%}); AUC rel_goal {auc([(r['rel_goal'], r['useful_page']) for r in sub]):.3f}"
          f" rel_diff {auc([(r['rel_diff'], r['useful_page']) for r in sub]):.3f}")
  combo = [(r["rel_goal"] + r["rel_param"], r["useful_page"]) for r in rows]
  print(f"  AUC rel_goal+rel_param vs useful_page = {auc(combo):.3f}")
  bins = defaultdict(lambda: [0, 0])
  for r in rows:
    s = r["rel_diff"]
    b = "0" if s == 0 else "(0,0.1]" if s <= 0.1 else "(0.1,0.25]" if s <= 0.25 else ">0.25"
    bins[b][0] += 1
    bins[b][1] += r["useful_page"]
  for b in ("0", "(0,0.1]", "(0.1,0.25]", ">0.25"):
    c, u = bins[b]
    if c:
      print(f"  rel_diff {b:11s}: nodes {c:4d}  later visited {u:3d} ({u / c:.1%})")


if __name__ == "__main__":
  for a in sys.argv[1:]:
    main(a)
