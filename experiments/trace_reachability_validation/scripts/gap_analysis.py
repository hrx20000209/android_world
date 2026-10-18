"""Where does cross-task graph reuse break, why (dynamics), and could exploration have bridged it?

For every model navigation step (page P --control f--> page Q) of task T at time t, using only observations of OTHER
tasks before t (agent and exploration separately and together):
  reusable      : P known (layout key), edge (P, f) known, landing activity of Q matches the edge's majority
  break_page    : P's activity known but not P's layout     (dynamic page structure)
  break_edge    : P known but (P, f) never observed         (coverage gap)
  break_landing : (P, f) known but it landed elsewhere       (dynamic transition)
  new_activity  : P's activity never observed
For break_page, the structural difference between P and the closest known page of the same activity is attributed
to element kinds (list rows, dialog/popup window, keyboard, toggles/checks, empty-state, toolbar/title text, other).
"""
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402

R, U, C = full9.R, full9.U, full9.C


def pk(s):
  return (s.get("package"), s.get("activity"), s.get("layout_sig"))


def fam_of(o):
  if o.get("origin") == "explore" and o.get("selector"):
    return C.family_of_sel(o["selector"])
  return o["slot_id"]


def skel(ui):
  return Counter(ui.skeleton())


def attribute(ui_p, ui_k):
  """Which kind of elements make two pages of one activity differ."""
  a, b = skel(ui_p), skel(ui_k)
  diff = (a - b) + (b - a)
  kinds = Counter()
  wins = len({n.window for n in ui_p.app_nodes() if n.visible}) != len({n.window for n in ui_k.app_nodes() if n.visible})
  if wins:
    kinds["dialog_or_popup"] += 1
  if ui_p.ime_visible() != ui_k.ime_visible():
    kinds["keyboard"] += 1
  for item, cnt in diff.items():
    s = str(item).lower()
    if any(w in s for w in ("recycler", "listview", "list", "row", "item", "card")):
      kinds["list_rows"] += cnt
    elif any(w in s for w in ("checkbox", "switch", "toggle", "radio")):
      kinds["toggles"] += cnt
    elif any(w in s for w in ("empty", "placeholder", "no_")):
      kinds["empty_state"] += cnt
    elif any(w in s for w in ("toolbar", "title", "action_bar", "tab")):
      kinds["toolbar_tabs"] += cnt
    elif any(w in s for w in ("snackbar", "toast", "fab", "progress")):
      kinds["transient_widgets"] += cnt
    else:
      kinds["other"] += cnt
  return kinds


def main(arm):
  raw = Path("raw/full") / arm
  obs = []
  for l in open(f"graph_snapshots/full/{arm}/graph.jsonl"):
    o = json.loads(l)
    if o.get("dst"):
      obs.append(o)
  obs.sort(key=lambda o: o.get("t", 0))
  steps = []
  for f in glob.glob(str(raw / "*" / "p0" / "steps.jsonl")):
    d = Path(f).parent
    t = d.parts[-2]
    rows = [json.loads(l) for l in open(f)]
    for i, r in enumerate(rows):
      if r.get("kind") != "model" or not r.get("pre") or not r.get("tok"):
        continue
      nxt = next((x for x in rows[i + 1:] if x.get("pre")), None)
      if nxt is None:
        continue
      P, Q = pk(r["pre"]["summary"]), pk(nxt["pre"]["summary"])
      if P == Q:
        continue
      steps.append((r["pre"]["t"], t, P, r["tok"], Q, d, r["pre"]["cid"]))
  steps.sort()
  out = {}
  for src in ("agent", "explore", "both"):
    pages, acts, edges = {}, set(), defaultdict(Counter)
    page_cap = {}
    res, why_page, bridge = Counter(), Counter(), Counter()
    j = 0
    for ts, task, P, tok, Q, d, cid in steps:
      while j < len(obs) and obs[j].get("t", 0) < ts:
        o = obs[j]; j += 1
        org = "explore" if o.get("origin") == "explore" else "agent"
        tk = o["instance"].split("/")[0]
        if tk == task_of_current.get("x") and False:
          pass
        if src != "both" and org != src:
          continue
        s, dd = pk(o["src"]), pk(o["dst"])
        for page, cidk in ((s, "src_cid"), (dd, "dst_cid")):
          pages.setdefault(page, set()).add(tk)
          acts.add(page[:2])
          if o.get(cidk) and page not in page_cap:
            page_cap[page] = (o["instance"], o[cidk])
        if s != dd:
          edges[(s, fam_of(o))][dd[1]] += 1
          edges[(s, fam_of(o), "tasks")][tk] += 1
      other = lambda page: page in pages and (pages[page] - {task})  # noqa: E731
      res["steps"] += 1
      if not other(P):
        if P[:2] in acts:
          res["break_page"] += 1
          # closest known page of the activity (fewest skeleton differences)
          try:
            ui_p = R.load_ui(d, cid)
            best = None
            for page, (inst, c2) in page_cap.items():
              if page[:2] != P[:2] or not other(page):
                continue
              tk2, var = inst.split("/")
              ui_k = R.load_ui(raw / tk2 / var.split("_")[0], c2)
              dist = sum(((skel(ui_p) - skel(ui_k)) + (skel(ui_k) - skel(ui_p))).values())
              if best is None or dist < best[0]:
                best = (dist, ui_k)
            if best is not None:
              for k, v in attribute(ui_p, best[1]).items():
                why_page[k] += 1 if v else 0
          except Exception:  # pylint: disable=broad-except
            pass
        else:
          res["new_activity"] += 1
        continue
      e = edges.get((P, tok))
      tasks_e = edges.get((P, tok, "tasks"), Counter())
      if not e or not (set(tasks_e) - {task}):
        res["break_edge"] += 1
        continue
      if e.most_common(1)[0][0] != Q[1]:
        res["break_landing"] += 1
        continue
      res["reusable"] += 1
    out[src] = (res, why_page)
  n = out["both"][0]["steps"]
  print(f"[{arm}] model navigation steps {n}")
  for src in ("agent", "explore", "both"):
    res, why = out[src]
    print(f"  evidence={src:7s} " + "  ".join(f"{k} {res[k]} ({res[k] / max(1, n):.0%})" for k in
                                              ("reusable", "break_page", "break_edge", "break_landing", "new_activity")))
  print("  dynamic sources of break_page (both):", dict(out["both"][1].most_common()))


task_of_current = {}

if __name__ == "__main__":
  for a in sys.argv[1:]:
    main(a)
