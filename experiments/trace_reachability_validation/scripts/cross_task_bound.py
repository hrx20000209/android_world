"""Offline bound: how many model inferences could cross-task trajectory reuse (same app) replace?

For each model step s of task T (in run order), candidate predictions come from EARLIER episodes of OTHER tasks in
the same app (agent trajectories only):
  page match : the earlier step was on the same page (package, activity, layout) - or the same activity (soft)
  prediction : that earlier step's action token; typed text is re-bound to T's value of the same field (parameter
               substitution through the goal parse), entity rows are compared by row family
  choice     : among candidate earlier episodes, the one whose goal is most similar (TF-IDF cosine of goal vectors)
Reported: coverage (steps with a prediction), precision (prediction == the model's action), and the same split by
action kind (navigation / entity row / typing / commit), plus precision when the predictor is confident (k
agreeing earlier episodes).
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


class Ep:
  def __init__(s, t):
    s.args = type("a", (), {"task": t})()


def cos(a, b):
  dot = sum(v * b.get(k, 0) for k, v in a.items())
  if not dot:
    return 0.0
  return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def kind(tok):
  if tok.startswith("TYPE"):
    return "typing"
  if tok.startswith("L|"):
    return "entity_row"
  if tok.startswith(("STATUS", "ANSWER")):
    return "answer"
  if tok.startswith("OPEN_APP"):
    return "open_app"
  low = tok.lower()
  if any(w in low for w in ("save", "ok", "done", "delete", "send", "confirm", "create", "apply")):
    return "commit"
  return "navigation"


def episodes(arm):
  out = []
  for f in glob.glob(f"raw/full/{arm}/*/p0/steps.jsonl"):
    d = Path(f).parent
    t = d.parts[-2]
    rows = [json.loads(l) for l in open(f)]
    model = [r for r in rows if r.get("kind") == "model" and r.get("pre")]
    if not model:
      continue
    goal = json.load(open(d / "meta.json"))["goal"]
    fx = full9.FuncExec(Ep(t), goal)
    vals = {}
    for fd, vs in fx.recs.items():
      for v in vs:
        if v:
          vals[full9.FG.norm(v)] = fd
    out.append({"t0": model[0]["pre"]["t"], "task": t, "g": EV.goal_vector(goal, fx), "recs": fx.recs, "vals": vals,
                "steps": [(r["pre"]["summary"].get("package"), r["pre"]["summary"].get("activity"),
                           r["pre"]["summary"].get("layout_sig"), r.get("tok") or "", (r.get("action") or {}).get("text"))
                          for r in model]})
  return sorted(out, key=lambda e: e["t0"])


def run(arm):
  eps = episodes(arm)
  c = Counter()
  conf = Counter()
  for i, e in enumerate(eps):
    prior = eps[:i]
    for (pkg, act, lay, tok, text) in e["steps"]:
      k = kind(tok)
      if k == "open_app" or "launcher" in (pkg or ""):
        continue
      c["steps"] += 1
      c["steps", k] += 1
      cands = []
      for p in prior:
        if p["task"] == e["task"]:
          continue
        for (pkg2, act2, lay2, tok2, text2) in p["steps"]:
          if pkg2 != pkg or act2 != act:
            continue
          exact = lay2 == lay
          pred = tok2
          if tok2.startswith("TYPE") and text2:
            fd = p["vals"].get(full9.FG.norm(text2))
            if fd is not None:
              vs = e["recs"].get(fd) or []
              pred = (tok2, vs[0] if vs else None)       # parameter substitution: same field, this task's value
            else:
              pred = (tok2, text2)
          cands.append((exact, cos(e["g"], p["g"]), pred, p["task"]))
          break                                           # first matching step of that episode on this page
      if not cands:
        continue
      c["covered"] += 1
      c["covered", k] += 1
      cands.sort(key=lambda x: (-x[0], -x[1]))
      best = cands[0][2]
      truth = tok if not tok.startswith("TYPE") else (tok, text)
      hit = best == truth
      c["hit"] += hit
      c["hit", k] += hit
      agree = sum(1 for x in cands if x[2] == best)
      for kk in (1, 2, 3):
        if agree >= kk:
          conf[kk, "fire"] += 1
          conf[kk, "hit"] += hit
          conf[kk, "fire", k] += 1
          conf[kk, "hit", k] += hit
  n = c["steps"]
  print(f"[{arm}] model steps in apps {n}; covered {c['covered']} ({c['covered'] / n:.0%}); "
        f"precision {c['hit'] / max(1, c['covered']):.2f}; correct predictions {c['hit']} ({c['hit'] / n:.0%} of steps)")
  for k in ("navigation", "entity_row", "typing", "commit", "answer"):
    print(f"   {k:11s} steps {c['steps', k]:4d} covered {c['covered', k]:4d} correct {c['hit', k]:4d} "
          f"precision {c['hit', k] / max(1, c['covered', k]):.2f}")
  for kk in (1, 2, 3):
    print(f"   >= {kk} agreeing earlier episodes: fire {conf[kk, 'fire']} ({conf[kk, 'fire'] / n:.0%}) "
          f"precision {conf[kk, 'hit'] / max(1, conf[kk, 'fire']):.2f}; navigation-only precision "
          f"{conf[kk, 'hit', 'navigation'] / max(1, conf[kk, 'fire', 'navigation']):.2f} "
          f"({conf[kk, 'fire', 'navigation']} fires)")


if __name__ == "__main__":
  for a in sys.argv[1:]:
    run(a)
