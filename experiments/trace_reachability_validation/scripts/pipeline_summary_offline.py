"""Offline test: start decision i+1 before summary i is decoded, with explain_i (decoded before the action) standing in
for summary_i as the latest history entry. For consecutive model steps (i, i+1) of a finished run (lab server):
  step i  : regenerate the response with the real history -> explain_i, summary_i
  A       : decide i+1 with history + summary_i (the current pipeline)
  C       : decide i+1 with history + explain_i (pipelined)
Agreement of C with A, and of both with the original action of step i+1."""
import glob, json, random, sys
from collections import Counter
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402
from android_world.agents import gelab_agent, infer  # noqa: E402
R, C = full9.R, full9.C
API = "http://127.0.0.1:18095/v1/chat/completions"


def hist_text(h):
  return "\n".join(f"{k + 1}. {s}" for k, s in enumerate(h[-8:])) or "None yet."


def main(run, arm, n=120, seed=0):
  random.seed(seed)
  llm = infer.LlamaCppWrapper(api_url=API, temperature=0.0, model_name="GELAB-ZERO-4B")
  pairs = []
  for f in sorted(glob.glob(f"raw/{run}/{arm}/*/p0/steps.jsonl")):
    d = Path(f).parent
    rows = [json.loads(l) for l in open(f)]
    idx = [i for i, r in enumerate(rows) if r.get("kind") == "model" and r.get("pre")]
    for a_, b_ in zip(idx, idx[1:]):
      if b_ == a_ + 1:
        pairs.append((d, rows, a_, b_))
  random.shuffle(pairs)
  c = Counter()
  for d, rows, a_, b_ in pairs[:n]:
    goal = json.load(open(d / "meta.json"))["goal"]
    hist = [x.get("summary") for x in rows[:a_] if x.get("summary")]
    img_i = Image.open(d / "img" / f"{rows[a_]['pre']['cid']}.jpg").convert("RGB")
    img_j = Image.open(d / "img" / f"{rows[b_]['pre']['cid']}.jpg").convert("RGB")
    ui_j = R.load_ui(d, rows[b_]["pre"]["cid"])
    try:
      ri, _, _ = llm.predict_mm("", [], messages=gelab_agent.build_gelab_messages(goal, hist_text(hist), img_i))
      pi = gelab_agent.parse_gelab_response(ri)
    except Exception:  # pylint: disable=broad-except
      c["err"] += 1
      continue
    summ = gelab_agent._normalize_space(pi.get("summary")) or ""  # pylint: disable=protected-access
    expl = gelab_agent._normalize_space(pi.get("explain")) or ""  # pylint: disable=protected-access
    if not expl:
      c["no_explain"] += 1                 # outputs without an explain field: placeholder = action text
      expl = str(pi.get("action") or "")
    out = {}
    for k, last in (("A", summ), ("C", expl)):
      r, _, _ = llm.predict_mm("", [], messages=gelab_agent.build_gelab_messages(goal, hist_text(hist + [last]), img_j))
      try:
        p = gelab_agent.parse_gelab_response(r)
        a, _, _ = gelab_agent.gelab_action_to_json_action(p, (1080, 2400))
        out[k] = C.action_token(ui_j, a.__dict__)[0]
      except Exception:  # pylint: disable=broad-except
        out[k] = "?"
    c["n"] += 1
    c["C_eq_A"] += out["C"] == out["A"]
    o = rows[b_].get("tok") or ""
    c["A_orig"] += out["A"] == o
    c["C_orig"] += out["C"] == o
    if c["n"] % 30 == 0:
      print(dict(c), flush=True)
  print(f"[{run}/{arm}] pairs {c['n']}: pipelined decision equals current pipeline's {c['C_eq_A']} "
        f"({c['C_eq_A']/max(1,c['n']):.0%}); vs original action: current {c['A_orig']}, pipelined {c['C_orig']}; "
        f"no explain field {c['no_explain']}, errors {c['err']}")


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 120)
