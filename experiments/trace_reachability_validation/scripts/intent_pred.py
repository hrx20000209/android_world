"""Offline calibration of the goal-conditioned navigation macro (design review P0-2).

Replays an arm in time order. Before each model navigation action (page change, click on a control family):
  * reachability: out-edges of the current page known from OTHER tasks before this moment (agent or exploration);
  * use profile: goal-term vectors of earlier SUCCESSFUL episodes (other tasks) that traversed the edge;
  * unlock profile: TF-IDF content of the edge's destination captures.
  intent(e) = LAM * cos(g, use_e) + (1 - LAM) * rel(g + p, unlock_e)
DIRECT_OK(e) = n_obs >= 2, landing-activity share >= 0.9, n_valid >= 1, cos(g, use_e) >= THETA, margin >= M.
Reports precision (macro edge == model's action) and coverage (fraction of all navigation actions) per threshold.
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

R, U, C = full9.R, full9.U, full9.C
LAM = 0.5


def pk(s):
  return (s.get("package"), s.get("activity"), s.get("layout_sig"))


def fam_of(o):
  return o["slot_id"] if o["slot_id"].startswith("E|") else "E|" + o["slot_id"]


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


def episode_rows(arm):
  """(t_start, task, goal vectors, steps rows, success) for every episode of the arm."""
  out = []
  for f in glob.glob(f"raw/full/{arm}/*/p0/steps.jsonl"):
    d = Path(f).parent
    if not (d / "result.json").exists():
      continue
    t = f.split("/")[3]
    goal = json.load(open(d / "meta.json"))["goal"]
    fx = full9.FuncExec(Ep(t), goal)
    rows = [json.loads(l) for l in open(f)]
    ts = [r["pre"]["t"] for r in rows if r.get("pre") and r["pre"].get("t")]
    if not ts:
      continue
    succ = json.load(open(d / "result.json")).get("is_successful") == 1.0
    out.append((min(ts), max(ts), t, EV.goal_vector(goal, fx), EV.param_vector(fx), rows, succ))
  return sorted(out)


def nav_moves(rows):
  """(row, src page, family, dst page) for model / graph actions that changed page."""
  out = []
  for i, r in enumerate(rows):
    if not r.get("pre") or not (r.get("tok") or "").startswith("E|"):
      continue
    nxt = next((x for x in rows[i + 1:] if x.get("pre")), None)
    if nxt is None:
      continue
    src, dst = pk(r["pre"]["summary"]), pk(nxt["pre"]["summary"])
    if src != dst:
      out.append((r, src, r["tok"], dst))
  return out


def main(arm, grid):
  RAW = Path("raw/full") / arm
  eps = episode_rows(arm)
  obs = defaultdict(list)            # (pkg, act, fam) -> [(t, dst page, instance, cid, task)]
  for l in open(f"graph_snapshots/full/{arm}/graph.jsonl"):
    o = json.loads(l)
    if not o.get("dst") or pk(o["dst"]) == pk(o["src"]):
      continue
    s = o["src"]
    obs[(s["package"], s["activity"], fam_of(o))].append(
        (o.get("t", 0), pk(o["dst"]), o["instance"], o.get("dst_cid"), o["instance"].split("/")[0]))
  for v in obs.values():
    v.sort()
  by_page_act = defaultdict(set)
  for k in obs:
    by_page_act[(k[0], k[1])].add(k[2])
  # use profiles accumulate from successful episodes, keyed by end time
  use_events = []                    # (t_end, key, task, gvec)
  for t0, t1, task, g, p, rows, succ in eps:
    if succ:
      for r, src, fam, dst in nav_moves(rows):
        use_events.append((t1, (src[0], src[1], fam), task, g))
  use_events.sort(key=lambda x: x[0])
  samples = []                       # one per model navigation action
  for t0, t1, task, g, p, rows, succ in eps:
    gp = dict(g)
    for k2, v2 in p.items():
      gp[k2] = gp.get(k2, 0) + v2
    for r, src, fam, dst in nav_moves(rows):
      if r.get("kind") != "model":
        continue
      tt = r["pre"]["t"]
      cands = []
      try:
        live = {C.family_of_sel(t2.sel) for t2 in U.enumerate_slots(R.load_ui(RAW / task / "p0", r["pre"]["cid"]))}
      except Exception:  # pylint: disable=broad-except
        live = None
      for f2 in by_page_act.get((src[0], src[1]), ()):
        if live is not None and f2 not in live:
          continue
        key = (src[0], src[1], f2)
        es = [e for e in obs[key] if e[0] < tt and e[4] != task]
        if not es:
          continue
        use = Counter(); n_valid = 0
        for te, k3, tk, gv in use_events:
          if te >= tt:
            break
          if k3 == key and tk != task:
            use.update(gv); n_valid += 1
        acts = Counter(e[1][1] for e in es)
        land = acts.most_common(1)[0][1] / len(es)
        unlock = 0.0
        for e in es[-3:]:
          ui = EV.load_cap(RAW, e[2], e[3]) if e[3] else None
          if ui is not None:
            unlock = max(unlock, EV.rel(gp, EV.page_tokens(ui)))
        cu = cos(g, use)
        cands.append({"fam": f2, "n": len(es), "land": land, "n_valid": n_valid, "cos_use": cu,
                      "intent": LAM * cu + (1 - LAM) * unlock, "pred_act": acts.most_common(1)[0][0]})
      samples.append((fam, dst, cands, succ))
  n_nav = len(samples)
  print(f"[{arm}] model navigation actions {n_nav}; with cross-task candidates {sum(1 for s_ in samples if s_[2])}")
  for theta, m in grid:
    hit = fire = 0
    hit_s = fire_s = 0
    for fam, dst, cands, succ_ep in samples:
      if not cands:
        continue
      cs = sorted(cands, key=lambda c: -c["intent"])
      top = cs[0]; second = cs[1]["intent"] if len(cs) > 1 else 0.0
      if (top["n"] >= 2 and top["land"] >= 0.9 and top["n_valid"] >= 1 and top["cos_use"] >= theta
          and top["intent"] - second >= m):
        fire += 1
        ok = top["fam"] == fam and top["pred_act"] == dst[1]
        hit += ok
        if succ_ep:
          fire_s += 1; hit_s += ok
    print(f"  theta={theta:.2f} margin={m:.2f}: fire {fire} ({fire / max(1, n_nav):.1%} of nav)  "
          f"precision {hit / max(1, fire):.2f} | in successful episodes {hit_s}/{fire_s} = {hit_s / max(1, fire_s):.2f}")


if __name__ == "__main__":
  grid = [(th, m) for th in (0.5, 0.7) for m in (0.2, 0.3, 0.4)]
  for a in sys.argv[1:]:
    main(a, grid)
