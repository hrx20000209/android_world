"""Node x term TF-IDF matrix that mixes inference and exploration trajectories, evaluated for cross-task reuse.

Each graph node n (package, activity, layout) has a term vector M[n] accumulated from
  content : TF-IDF of the words on n's page (every capture of n, from agent visits or exploration probes)
  context : goal-term vector of each task whose trajectory reached n
            - inference trajectory (the model navigated there)        weight W_INFER
            - exploration trajectory (a probe opened it in that task)  weight W_EXPLORE
Relevance of n to a new goal g: rel(n) = cos(g, M[n]); one-step backup over the graph:
  R(n) = max(rel(n), GAMMA * max_{n -> n'} rel(n')).
Next-action prediction on page P: score each known out-edge (P, control) -> n by R(n); top-1 vs the model's action.
Evidence for a step of task T at time t comes only from other tasks and only from before t (cross-task, causal).
Variants: matrix from inference trajectories only / exploration only / both (mixed)."""
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402
import explore_value as EV  # noqa: E402

R, U, C = full9.R, full9.U, full9.C
W_CONTENT, W_INFER, W_EXPLORE, GAMMA = 1.0, 1.0, 0.5, 0.7


def pk(s):
  return (s.get("package"), s.get("activity"), s.get("layout_sig"))


def fam_of(o):
  if o.get("origin") == "explore" and o.get("selector"):
    return C.family_of_sel(o["selector"])
  return o["slot_id"]


def cos(a, b):
  if not a or not b:
    return 0.0
  dot = sum(v * b.get(t, 0.0) for t, v in a.items())
  if dot == 0:
    return 0.0
  return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


class Ep:
  def __init__(s, t):
    s.args = type("a", (), {"task": t})()


def content_vec(ui):
  v = Counter()
  for t in EV.page_tokens(ui):
    v[t] += EV.w_ui(t)
  n = math.sqrt(sum(x * x for x in v.values())) or 1.0
  return {t: x / n for t, x in v.items()}


def unit(g):
  n = math.sqrt(sum(x * x for x in g.values())) or 1.0
  return {t: x / n for t, x in g.items()}


def events(arm):
  """Time-ordered evidence: ('edge', t, task, origin, src, fam, dst, (inst, cid)) and the goal of each task."""
  raw = Path("raw/full") / arm
  goals, ev = {}, []
  for d in glob.glob(str(raw / "*" / "p0")):
    d = Path(d)
    if not (d / "meta.json").exists():
      continue
    t = d.parts[-2]
    goal = json.load(open(d / "meta.json"))["goal"]
    goals[t] = unit(EV.goal_vector(goal, full9.FuncExec(Ep(t), goal)))
  for l in open(f"graph_snapshots/full/{arm}/graph.jsonl"):
    o = json.loads(l)
    if not o.get("dst"):
      continue
    src, dst = pk(o["src"]), pk(o["dst"])
    if src == dst:
      continue
    task = o["instance"].split("/")[0]
    org = "explore" if o.get("origin") == "explore" else "infer"
    ev.append((o.get("t", 0), task, org, src, fam_of(o), dst, (o["instance"], o.get("dst_cid"))))
  ev.sort(key=lambda x: x[0])
  return raw, goals, ev


def model_steps(arm, raw):
  out = []
  for f in glob.glob(str(raw / "*" / "p0" / "steps.jsonl")):
    t = f.split("/")[3]
    rows = [json.loads(l) for l in open(f)]
    for i, r in enumerate(rows):
      if r.get("kind") != "model" or not (r.get("tok") or "").startswith("E|") or not r.get("pre"):
        continue
      nxt = next((x for x in rows[i + 1:] if x.get("pre")), None)
      if nxt is None or pk(nxt["pre"]["summary"]) == pk(r["pre"]["summary"]):
        continue
      out.append((r["pre"]["t"], t, pk(r["pre"]["summary"]), r["tok"], pk(nxt["pre"]["summary"]), Path(f).parent, r["pre"]["cid"]))
  out.sort()
  return out


def run(arm, variant):
  raw, goals, ev = events(arm)
  steps = model_steps(arm, raw)
  use = {"infer": variant in ("infer", "mixed"), "explore": variant in ("explore", "mixed")}
  # incremental state per task-exclusion is expensive; keep per-task contributions and subtract at query time
  content = {}                                    # node -> content vector (latest capture)
  ctx = defaultdict(lambda: defaultdict(Counter))  # node -> task -> context vector (weighted goal terms)
  out_edges = defaultdict(lambda: defaultdict(set))  # src -> fam -> {dst}, with tasks that observed it
  edge_tasks = defaultdict(set)
  i = 0
  stats = Counter(); conf = []
  for (ts, task, P, tok, dstP, d, cid) in steps:
    while i < len(ev) and ev[i][0] < ts:
      t0, tk, org, src, fam, dst, ent = ev[i]
      i += 1
      if not use[org]:
        continue
      out_edges[src][fam].add(dst)
      edge_tasks[(src, fam)].add(tk)
      if ent[1] and dst not in content:
        ui = EV.load_cap(raw, ent[0], ent[1])
        if ui is not None:
          content[dst] = content_vec(ui)
      w = W_INFER if org == "infer" else W_EXPLORE
      for term, x in goals.get(tk, {}).items():
        ctx[dst][tk][term] += w * x
    stats["nav"] += 1
    g = goals.get(task, {})
    cands = {fam: dsts for fam, dsts in out_edges.get(P, {}).items() if edge_tasks[(P, fam)] - {task}}
    if not cands or not g:
      continue
    try:
      live = {C.family_of_sel(s.sel) for s in U.enumerate_slots(R.load_ui(d, cid))}
    except Exception:  # pylint: disable=broad-except
      live = None
    if live is not None:
      cands = {f: v for f, v in cands.items() if f in live}
    if not cands:
      continue
    stats["with_cands"] += 1

    def node_vec(n):
      v = Counter({t: W_CONTENT * x for t, x in content.get(n, {}).items()})
      for tk, c in ctx.get(n, {}).items():
        if tk != task:
          v.update(c)
      return v

    def rel(n):
      return cos(g, node_vec(n))

    def backed(n):
      best = rel(n)
      for f2, ds in out_edges.get(n, {}).items():
        for n2 in ds:
          if n2 != n:
            best = max(best, GAMMA * rel(n2))
      return best

    scored = sorted(((max(backed(n) for n in ds), f) for f, ds in cands.items()), reverse=True)
    top_s, top_f = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    hit = top_f == tok
    stats["top1"] += hit
    # can the graph execute the top control by itself? (safe, unique, not a list row, not a commit / dialog button)
    kind = "unknown"
    try:
      ui_ = R.load_ui(d, cid)
      sl_ = [x for x in U.enumerate_slots(ui_) if C.family_of_sel(x.sel) == top_f]
      if len(sl_) != 1 or sl_[0].sel.get("in_list"):
        kind = "entity"
      elif sl_[0].unsafe_reason in ("unsafe_label", "dialog_action_button") or full9.effectful(sl_[0].sel):
        kind = "commit"
      elif not sl_[0].safe:
        kind = "unsafe"
      else:
        kind = "navigation"
    except Exception:  # pylint: disable=broad-except
      pass
    conf.append((top_s - second, hit, kind))
  conf.sort(key=lambda x: -x[0])
  n = stats["nav"]
  for m in (0.3, 0.45, 0.55):
    sub = [c for c in conf if c[0] >= m]
    kinds = Counter(c[2] for c in sub)
    nav = [c for c in sub if c[2] == "navigation"]
    print(f"  [{arm} {variant}] margin>={m}: fires {len(sub)} by kind {dict(kinds)} | executable navigation {len(nav)} "
          f"precision {sum(c[1] for c in nav) / max(1, len(nav)):.2f}")
  line = (f"[{arm} {variant:7s}] nav {n}  with cross-task candidates {stats['with_cands']} ({stats['with_cands'] / max(1, n):.0%})"
          f"  top-1 {stats['top1']}/{stats['with_cands']} = {stats['top1'] / max(1, stats['with_cands']):.2f}")
  for frac in (0.1, 0.25):
    k = max(1, int(len(conf) * frac))
    line += f" | most-confident {int(frac * 100)}% ({k}, {k / max(1, n):.0%} of nav): {sum(c[1] for c in conf[:k]) / k:.2f}"
  for frac in (0.05, 0.1, 0.15, 0.25):
    k = max(1, int(len(conf) * frac))
    line += f"\n      margin at top-{int(frac * 100)}%: {conf[k - 1][0]:.3f}"

  print(line)


if __name__ == "__main__":
  for arm in sys.argv[1:]:
    for variant in ("infer", "explore", "mixed"):
      run(arm, variant)
