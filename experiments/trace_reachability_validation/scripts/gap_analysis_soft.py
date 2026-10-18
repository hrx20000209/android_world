"""Cross-task reuse with SOFT page matching: the live page matches the most similar known page of the same
(activity, mode) if the Jaccard similarity of their skeletons (named + anonymous items, multiset) >= TAU.
Edges of the matched page are reusable if the control family is present; landing is judged by (activity, mode) of
the destination and by its similarity to the edge's recorded destination."""
import glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9, explore_value as EV
import state_abstraction as SA
R, U, C = full9.R, full9.U, full9.C


def fam_of(o):
  if o.get("origin") == "explore" and o.get("selector"):
    return C.family_of_sel(o["selector"])
  return o["slot_id"]


def jac(a, b):
  inter = sum((a & b).values()); uni = sum((a | b).values())
  return inter / uni if uni else 1.0


def sk(ui):
  return Counter(i for i in ui.skeleton() if i.split("|")[1] not in SA.SYSTEM)


def main(arm, sources, taus=(0.7,)):
  raw = Path("raw/full") / arm
  obs = sorted((json.loads(l) for l in open(f"graph_snapshots/full/{arm}/graph.jsonl")), key=lambda o: o.get("t", 0))
  obs = [o for o in obs if o.get("dst") and o.get("src_cid") and o.get("dst_cid")]
  steps = []
  for f in glob.glob(str(raw / "*" / "p0" / "steps.jsonl")):
    d = Path(f).parent; t = d.parts[-2]
    rows = [json.loads(l) for l in open(f)]
    for i, r in enumerate(rows):
      if r.get("kind") != "model" or not r.get("pre") or not r.get("tok"):
        continue
      nxt = next((x for x in rows[i + 1:] if x.get("pre")), None)
      if nxt is not None:
        steps.append((r["pre"]["t"], t, r["tok"], d, r["pre"]["cid"], nxt["pre"]["cid"]))
  steps.sort()
  pages = defaultdict(list)      # (activity, mode) -> [(skeleton, page_id)]
  page_fams = defaultdict(set)   # page_id -> safe, non-effectful control families seen on its captures
  pid_of = {}                    # (instance, cid) -> page id
  edges = defaultdict(lambda: defaultdict(list))   # page_id -> fam -> [(dst (activity, mode), dst skeleton, task)]
  page_tasks = defaultdict(set)
  res = {tau: Counter() for tau in taus}
  j = 0

  def page_id(ui, inst, cid, tau_store=0.95):
    k = (ui.activity, SA.mode(ui)); s = sk(ui)
    for s2, p in pages[k]:
      if jac(s, s2) >= tau_store:
        return p, s
    p = (k, len(pages[k])); pages[k].append((s, p))
    return p, s

  for ts, task, tok, d, cp, cq in steps:
    while j < len(obs) and obs[j].get("t", 0) < ts:
      o = obs[j]; j += 1
      org = "explore" if o.get("origin") == "explore" else "agent"
      if org not in sources:
        continue
      su = EV.load_cap(raw, o["instance"], o["src_cid"]); du = EV.load_cap(raw, o["instance"], o["dst_cid"])
      if su is None or du is None:
        continue
      tk = o["instance"].split("/")[0]
      ps, _ = page_id(su, o["instance"], o["src_cid"])
      if ps not in page_fams or len(page_fams[ps]) < 400:
        for sl in U.enumerate_slots(su):
          if sl.safe and not full9.effectful(sl.sel):
            page_fams[ps].add(C.family_of_sel(sl.sel))
      pdd, sdd = page_id(du, o["instance"], o["dst_cid"])
      page_tasks[ps].add(tk); page_tasks[pdd].add(tk)
      if ps != pdd:
        edges[ps][fam_of(o)].append(((du.activity, SA.mode(du)), sdd, tk))
    try:
      up, uq = R.load_ui(d, cp), R.load_ui(d, cq)
    except Exception:
      continue
    if (up.activity, SA.mode(up)) == (uq.activity, SA.mode(uq)) and jac(sk(up), sk(uq)) >= 0.95:
      continue                                  # in-page action
    kp, sp = (up.activity, SA.mode(up)), sk(up)
    cands = sorted(((jac(sp, s2), p) for s2, p in pages.get(kp, []) if page_tasks[p] - {task}), reverse=True)
    for tau in taus:
      c = res[tau]; c["steps"] += 1
      if not cands:
        c["break_page" if any(k[0] == up.activity for k in pages) else "new_activity"] += 1
        continue
      if cands[0][0] < tau:
        c["break_page"] += 1
        continue
      P = cands[0][1]
      es = [e for e in edges[P].get(tok, []) if e[2] != task]
      if not es:
        c["break_edge"] += 1
        if tok in page_fams.get(P, ()):
          c["bridgeable_by_exploration"] += 1
        continue
      maj = Counter(e[0] for e in es).most_common(1)[0][0]
      if maj != (uq.activity, SA.mode(uq)):
        c["break_landing"] += 1
        continue
      c["reusable"] += 1
  for tau in taus:
    c = res[tau]; n = c["steps"]
    print(f"[{arm} {'+'.join(sources):13s} tau={tau}] steps {n}  " + "  ".join(
        f"{k} {c[k]} ({c[k] / max(1, n):.0%})" for k in ("reusable", "break_page", "break_edge", "bridgeable_by_exploration", "break_landing", "new_activity")))


if __name__ == "__main__":
  for a in sys.argv[1:]:
    main(a, ("agent",))
    main(a, ("agent", "explore"))
