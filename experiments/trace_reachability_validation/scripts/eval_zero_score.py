"""Offline: when TF-IDF gives every exploration candidate a zero score, what ranks them best?

Same probe moments and candidate sets as eval_rank_tfidf.py (depth-1 probes; graph rebuilt as of the moment).
All scorers stay inside the TF-IDF / vector-space model; they differ in how the query (goal) or the document
(control) is represented, so that a control can match the goal without sharing a literal word with it.

  screen    : screen order                                                   (current tie-break)
  rev       : P_rev (more reversible first)
  tfidf     : goal vs control's own text, app-as-document IDF                (current relevance)
  char3     : the same with character 3-grams (subword match: expense/expenses, fab_add/add)
  prf       : pseudo-relevance feedback (Rocchio): the query is expanded with the TF-IDF vectors of the graph pages
              of this app that match the goal best (before this moment)
  anchor    : anchor-text document expansion: a control's document = its own text + the goal terms of the earlier
              tasks of this run whose model clicked this control family in this app (inference trajectories as
              anchor text; online and causal)
  dest      : the control's document = its own text + the text of its destination pages known in the graph
              (agent or exploration edges of the same label / rid)
  fused     : anchor + dest + own text as fields of one document (sum of field cosines)

Labels (offline only): useful = the model clicked the family in a successful episode of the same task (any arm);
later = the model clicked it later in this episode. Reported separately for the moments where tfidf is all zero.
"""
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402
import explore_value as EV  # noqa: E402

R, U, C, FG = full9.R, full9.U, full9.C, full9.FG
P_MIN = 0.7


class Ep:
  def __init__(s, t):
    s.args = type("a", (), {"task": t})()


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

# character 3-gram IDF over apps (documents = apps)
APP_TOKS, _ = EV.app_doc()
GRAM_DF = Counter()
for toks in APP_TOKS.values():
  GRAM_DF.update({g for t in toks for g in (lambda w: [w[i:i + 3] for i in range(max(1, len(w) - 2))])("#" + t + "#")})
N_APPS = len(APP_TOKS)


def grams(toks):
  out = Counter()
  for t in toks:
    w = "#" + t + "#"
    for i in range(max(1, len(w) - 2)):
      out[w[i:i + 3]] += 1
  return out


def gw(g):
  return math.log((N_APPS + 1) / (GRAM_DF.get(g, 0) + 1)) + 1


def cos(a, b):
  if not a or not b:
    return 0.0
  dot = sum(v * b.get(k, 0.0) for k, v in a.items())
  if dot == 0:
    return 0.0
  return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def tfidf_vec(toks, w):
  v = Counter()
  for t in toks:
    v[t] += w(t)
  return v


def run(arm):
  RAW = Path("raw/full") / arm
  lines = [json.loads(l) for l in open(f"graph_snapshots/full/{arm}/graph.jsonl")]
  lines.sort(key=lambda o: o.get("t", 0))
  goals = {}
  for d in glob.glob(str(RAW / "*" / "p0")):
    t = d.split("/")[3]
    try:
      g = json.load(open(Path(d) / "meta.json"))["goal"]
      goals[t] = EV.goal_vector(g, full9.FuncExec(Ep(t), g))
    except Exception:  # pylint: disable=broad-except
      pass
  c = Counter()
  ranks = defaultdict(lambda: defaultdict(list))
  for d in sorted(glob.glob(str(RAW / "*" / "p0"))):
    t = d.split("/")[3]
    pf = Path(d) / "probes.jsonl"
    if not pf.exists() or t not in goals:
      continue
    gvec = goals[t]
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
      prior = []
      for o in lines:
        if o.get("t", 0) >= t0:
          break
        g.add(o)
        prior.append(o)
      try:
        ui = R.load_ui(Path(d), p["src_cid"])
      except Exception:  # pylint: disable=broad-except
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
      w = EV.w_app(ui.package)
      gv = EV.GraphValue(g, RAW.parent, gvec, ui.package, app_doc_idf=True)
      gq = tfidf_vec([], w)
      gq.update(gvec)
      own = {s.slot_id: EV.own_tokens(ui, s.sel) for s in cands}

      # anchor text: goal terms of earlier tasks (other than t) whose model clicked this family in this app
      anchor = defaultdict(Counter)
      for o in prior:
        if o.get("origin") != "agent" or not o.get("src") or o["src"].get("package") != ui.package:
          continue
        tk = o["instance"].split("/")[0]
        if tk == t or tk not in goals:
          continue
        anchor[o.get("slot_id")].update(goals[tk])
      # pseudo-relevance feedback: top pages of this app in the graph by relevance to the goal
      pages = {}
      for o in prior:
        dd = o.get("dst") or {}
        if dd.get("package") == ui.package and o.get("dst_cid"):
          pages.setdefault(dd.get("layout_sig"), (o["instance"], o["dst_cid"]))
      scored_pages = []
      for ent in list(pages.values())[:60]:
        pu = EV.load_cap(RAW, ent[0], ent[1])
        if pu is None:
          continue
        pv = tfidf_vec(EV.page_tokens(pu), w)
        r = cos(gq, pv)
        if r > 0:
          scored_pages.append((r, pv))
      scored_pages.sort(key=lambda x: -x[0])
      q_prf = Counter(gq)
      top_pages = scored_pages[:3]
      if top_pages:
        for _, pv in top_pages:
          n = math.sqrt(sum(v * v for v in pv.values())) or 1.0
          nq = math.sqrt(sum(v * v for v in gq.values())) or 1.0
          for k2, v2 in pv.items():
            q_prf[k2] += 0.5 * (v2 / n) * nq / len(top_pages)

      def s_tfidf(s):
        return cos(gq, tfidf_vec(own[s.slot_id], w))

      def s_char3(s):
        a = grams(list(gq.keys()))
        a = Counter({k: v * gw(k) for k, v in a.items()})
        b = grams(own[s.slot_id])
        b = Counter({k: v * gw(k) for k, v in b.items()})
        return cos(a, b)

      def s_prf(s):
        return cos(q_prf, tfidf_vec(own[s.slot_id], w))

      def s_anchor(s):
        fam = C.family_of_sel(s.sel)
        doc = tfidf_vec(own[s.slot_id], w)
        doc.update(anchor.get(fam, {}))
        return cos(gq, doc)

      def s_dest(s):
        best = 0.0
        lab = FG.norm(s.sel.get("label") or s.sel.get("desc") or "")
        nb = (gv.by_label.get(lab, []) if lab else []) + (gv.by_rid.get(s.sel.get("rid") or "", []) if s.sel.get("rid") else [])
        for ent in nb[:6]:
          dui = EV.load_cap(gv.raw, ent[0], ent[1])
          if dui is not None:
            best = max(best, cos(gq, tfidf_vec(EV.page_tokens(dui), w)))
        return best

      def s_fused(s):
        return s_tfidf(s) + s_anchor(s) + s_dest(s) + 0.5 * s_prf(s)

      def s_unc(s):
        # relevance unknown (no text of its own and no known destination) ranks above relevance known to be 0;
        # within each group, more reversible first
        has_text = bool(own[s.slot_id]) and not s.sel.get("in_list")
        return s_tfidf(s) * 10 + (0 if has_text or s_dest(s) > 0 else 1) + 0.1 * full9.p_reversible(ui, s.sel, evid)

      def s_unc_nav(s):
        # same, but list rows are entities, not navigation: they get no uncertainty bonus
        has_text = bool(own[s.slot_id]) or bool(s.sel.get("in_list"))
        return s_tfidf(s) * 10 + (0 if has_text or s_dest(s) > 0 else 1) + 0.1 * full9.p_reversible(ui, s.sel, evid)

      def s_norow(s):
        # list rows are entities (parameters), not navigation: rank them last unless the parameter terms match
        return s_tfidf(s) * 10 - (1 if s.sel.get("in_list") else 0) + 0.1 * full9.p_reversible(ui, s.sel, evid)

      scorers = {"unc_nav": s_unc_nav, "unc": s_unc, "norow": s_norow, "screen": lambda s: 0.0, "rev": lambda s: full9.p_reversible(ui, s.sel, evid), "tfidf": s_tfidf,
                 "char3": s_char3, "prf": s_prf, "anchor": s_anchor, "dest": s_dest, "fused": s_fused}
      fut = set().union(*[v for k, v in later.items() if k >= p["step"]]) if later else set()
      fams = [C.family_of_sel(s.sel) for s in cands]
      zero = all(s_tfidf(s) == 0 for s in cands)
      for scope in (("all", True), ("zero", zero)):
        if not scope[1]:
          continue
        sc = scope[0]
        c[sc, "moments"] += 1
        c[sc, "random_useful"] += sum(f in useful[t] for f in fams) / len(fams)
        c[sc, "any_useful"] += any(f in useful[t] for f in fams)
        for name, fn in scorers.items():
          vals = [fn(s) for s in cands]
          order = sorted(range(len(cands)), key=lambda i: (-vals[i], i))
          c[sc, name, "top1"] += fams[order[0]] in useful[t]
          c[sc, name, "top1_row"] += fams[order[0]] in useful[t] and bool(cands[order[0]].sel.get("in_list"))
          c[sc, name, "top3"] += any(fams[i] in useful[t] for i in order[:3])
          c[sc, name, "later"] += fams[order[0]] in fut
          c[sc, name, "nonzero"] += vals[order[0]] > 0 and name not in ("screen", "rev")
          for r, i in enumerate(order):
            if fams[i] in useful[t]:
              ranks[sc][name].append(r + 1)
              break
  return c, ranks


if __name__ == "__main__":
  names = ("screen", "rev", "tfidf", "char3", "prf", "fused", "unc", "unc_nav", "norow")
  for arm in sys.argv[1:]:
    c, ranks = run(arm)
    for sc in ("all", "zero"):
      n = c[sc, "moments"]
      if not n:
        continue
      print(f"[{arm} {sc}] moments={n} any_useful={c[sc, 'any_useful']} random_top1={c[sc, 'random_useful'] / n:.3f}")
      for name in names:
        rk = ranks[sc][name]
        print(f"  {name:7s} top1={c[sc, name, 'top1'] / n:.3f} top3={c[sc, name, 'top3'] / n:.3f} "
              f"later={c[sc, name, 'later'] / n:.3f} nonzero_top={c[sc, name, 'nonzero'] / n:.2f} "
              f"rank_first_useful={sum(rk) / max(len(rk), 1):.2f} top1_from_list_rows={c[sc, name, 'top1_row'] / n:.3f}")
