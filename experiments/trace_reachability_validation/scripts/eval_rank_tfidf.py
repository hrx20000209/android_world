"""Offline: how much of the exploration ranking does plain TF-IDF already give?

At every depth-1 FX probe moment of an arm, rebuild the candidate set the online rule would have seen (graph as of
that moment: page-level dedup, side-effect exclusion, learned bad containers ignored, P_rev >= threshold), then rank it
with several scorers and check the top-ranked control.

Scorers
  screen     : screen order (first candidate)
  tfidf_self : TF-IDF cosine between goal terms (+ parameter terms for list rows) and the control's own text, global UI
               IDF only -- no graph, no weights
  tfidf_graph: TF-IDF cosine against the best text evidence the graph holds, unweighted:
               max(rel(own), max rel(destination pages of same-label / same-rid controls)), global UI IDF
  score      : current unified graph value (tiers 0.8 / 0.5, value backup, global x local IDF, parameter term)
  current    : score - 0.3 * (1 - P_rev)                                    (online rule of FULL23-FULL29)

Usefulness labels for a probed/ranked control family
  later  : the model clicked that family later in the same episode
  useful : the model clicked that family in some successful episode of the same task (any arm)
"""
import json, glob, sys, os, math
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9, explore_value as EV  # noqa: E402
R, U, C, FG = full9.R, full9.U, full9.C, full9.FG
P_MIN = float(os.environ.get("P_MIN", 0.7))


class Ep:
  def __init__(s, t): s.args = type("a", (), {"task": t})()


useful = defaultdict(set)
for f in glob.glob("raw/full/*/*/p0/agent_transitions.jsonl"):
  res = Path(f).parent / "result.json"
  if not res.exists() or json.load(open(res)).get("is_successful") != 1.0:
    continue
  t = f.split("/")[3]
  for l in open(f):
    sel = json.loads(l).get("selector") or {}
    if sel:
      useful[t].add(C.family_of_sel(sel))


def rel_global(vec, toks):
  return EV.rel(vec, toks, EV.w_ui)


# ---- app-as-document IDF (documents = apps; leave-one-app-out) --------------------------------------------------
APP_TOKS = {}
_app_cache = Path("../runtime_graph_exploration/graph/app_tokens.json")
if _app_cache.exists():
  APP_TOKS = {k: set(v) for k, v in json.loads(_app_cache.read_text()).items()}
else:
  import gzip
  for arm in ("FULL11", "FULL12", "FULL13", "FULL14"):
    for f in glob.glob(f"raw/full/{arm}/*/p0/a11y/*_pre.json.gz"):
      try:
        with gzip.open(f, "rt") as fh:
          dd = json.load(fh)
        u = U.parse_flat(dd["payload"], activity=dd["activity"])
      except Exception:
        continue
      APP_TOKS.setdefault(u.package, set()).update(EV.page_tokens(u))
  _app_cache.write_text(json.dumps({k: sorted(v) for k, v in APP_TOKS.items()}))
APP_DF = Counter()
for _toks in APP_TOKS.values():
  APP_DF.update(_toks)


def w_app(pkg, smooth=1.0):
  """IDF over the other apps: N = number of apps except pkg, df = how many of them contain the term."""
  own = APP_TOKS.get(pkg, set())
  n = len(APP_TOKS) - (1 if pkg in APP_TOKS else 0)
  def w(t):
    df = APP_DF.get(t, 0) - (1 if t in own else 0)
    return max(math.log((n + 1) / (df + 1)) + smooth, 0.0)
  return w


def run(arm):
  RAW = Path("raw/full") / arm
  lines = [json.loads(l) for l in open(f"graph_snapshots/full/{arm}/graph.jsonl")]
  lines.sort(key=lambda o: o.get("t", 0))
  c = Counter()
  ranks = defaultdict(list)
  for d in sorted(glob.glob(str(RAW / "*" / "p0"))):
    t = d.split("/")[3]
    pf = Path(d) / "probes.jsonl"
    if not pf.exists():
      continue
    goal = json.load(open(Path(d) / "meta.json"))["goal"]
    fx = full9.FuncExec(Ep(t), goal)
    gvec = EV.goal_vector(goal, fx)
    pvec = EV.param_vector(fx)
    later = defaultdict(set)
    for l in open(Path(d) / "steps.jsonl"):
      s = json.loads(l)
      if s.get("kind") == "model" and s.get("tok"):
        later[s["step"]].add(s["tok"])
    for l in open(pf):
      p = json.loads(l)
      if p.get("depth", 1) != 1 or p.get("origin") != "explore":
        continue
      t0 = p["t"]
      g = R.Graph(None)
      for o in lines:
        if o.get("t", 0) >= t0:
          break
        g.add(o)
      try:
        ui = R.load_ui(Path(d), p["src_cid"])
      except Exception:
        continue
      known, _ = full9.page_family_index(g)
      evid = full9.container_evidence(g)
      cands = []
      for s in U.enumerate_slots(ui):
        if not s.safe or full9.effectful(s.sel):
          continue
        if (ui.package, ui.activity, C.family_of_sel(s.sel)) in known:
          continue
        if full9.p_reversible(ui, s.sel, evid) < P_MIN:
          continue
        cands.append(s)
      if len(cands) < 2:
        continue
      gv = EV.GraphValue(g, RAW.parent, gvec, ui.package)
      gv.pvec = pvec

      def own(s):
        return EV.own_tokens(ui, s.sel)

      def tf_self(s):
        v = rel_global(gvec, own(s))
        if s.sel.get("in_list") and pvec:
          v += rel_global(pvec, own(s))
        return v

      wa1, wa0 = w_app(ui.package, 1.0), w_app(ui.package, 0.0)

      def tf_app(s, w):
        v = EV.rel(gvec, own(s), w)
        if s.sel.get("in_list") and pvec:
          v += EV.rel(pvec, own(s), w)
        return v

      def tf_graph(s):
        best = tf_self(s)
        lab = FG.norm(s.sel.get("label") or s.sel.get("desc") or "")
        nb = (gv.by_label.get(lab, []) if lab else []) + (gv.by_rid.get(s.sel.get("rid") or "", []) if s.sel.get("rid") else [])
        for ent in nb[:8]:
          dui = EV.load_cap(gv.raw, ent[0], ent[1])
          if dui is not None:
            best = max(best, rel_global(gvec, EV.page_tokens(dui)))
        return best

      def score(s):
        return gv.score(ui, s)[0]

      def current(s):
        return score(s) - 0.3 * (1 - full9.p_reversible(ui, s.sel, evid))

      scorers = {"screen": lambda s: 0.0, "tfidf_self": tf_self, "tfidf_app": lambda s: tf_app(s, wa1),
                 "tfidf_app0": lambda s: tf_app(s, wa0),
                 "tfidf_app+rev": lambda s: tf_app(s, wa1) - 0.3 * (1 - full9.p_reversible(ui, s.sel, evid)),
                 "score": score, "current": current}
      fut = set().union(*[v for k, v in later.items() if k >= p["step"]]) if later else set()
      fams = [C.family_of_sel(s.sel) for s in cands]
      c["moments"] += 1
      c["cands"] += len(cands)
      c["random_later"] += sum(f in fut for f in fams) / len(fams)
      c["random_useful"] += sum(f in useful[t] for f in fams) / len(fams)
      c["any_later"] += any(f in fut for f in fams)
      c["any_useful"] += any(f in useful[t] for f in fams)
      for name, fn in scorers.items():
        vals = [fn(s) for s in cands]
        order = sorted(range(len(cands)), key=lambda i: (-vals[i], i))
        top = fams[order[0]]
        c[name, "top1_later"] += top in fut
        c[name, "top1_useful"] += top in useful[t]
        c[name, "top3_useful"] += any(fams[i] in useful[t] for i in order[:3])
        c[name, "tie_at_top"] += sum(1 for v in vals if v == vals[order[0]]) > 1
        for r, i in enumerate(order):
          if fams[i] in useful[t]:
            ranks[name].append(r + 1)
            break
  return c, ranks


if __name__ == "__main__":
  for arm in sys.argv[1:]:
    c, ranks = run(arm)
    n = c["moments"]
    print(f"[{arm}] moments={n} mean_candidates={c['cands'] / max(n, 1):.1f} "
          f"any_useful={c['any_useful']} any_later={c['any_later']}")
    print(f"  random        top1_useful={c['random_useful'] / n:.3f} top1_later={c['random_later'] / n:.3f}")
    print(f"  apps in app-IDF corpus: {len(APP_TOKS)}")
    for name in ("screen", "tfidf_self", "tfidf_app", "tfidf_app0", "tfidf_app+rev", "score", "current"):
      rk = ranks[name]
      print(f"  {name:12s}  top1_useful={c[name, 'top1_useful'] / n:.3f} top3_useful={c[name, 'top3_useful'] / n:.3f} "
            f"top1_later={c[name, 'top1_later'] / n:.3f} ties_at_top={c[name, 'tie_at_top'] / n:.2f} "
            f"mean_rank_first_useful={sum(rk) / max(len(rk), 1):.2f}")
