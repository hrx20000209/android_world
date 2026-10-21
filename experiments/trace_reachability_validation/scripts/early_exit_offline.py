"""Offline test of early-exit decoding (stop after the action fields): the next decisions then see a history made of
the explain fields (generated before the action) instead of the summaries. Episodes of a finished arm are replayed
step by step on the same screenshots:
  A (control)    : history = the episode's original summaries      -> agreement with the original action (noise floor)
  B (early exit) : history = explain fields this replay generated  -> agreement with the original action
Agreement by control family for clicks/long presses, by action type otherwise."""
import glob, json, random, sys, time
from collections import Counter
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402
from android_world.agents import gelab_agent, infer  # noqa: E402
R, U, C = full9.R, full9.U, full9.C
API = "http://127.0.0.1:18095/v1/chat/completions"


def tok_of(ui, resp):
  p = gelab_agent.parse_gelab_response(resp)
  a, _, _ = gelab_agent.gelab_action_to_json_action(p, (1080, 2400))
  return C.action_token(ui, a.__dict__)[0], p


def main(arm, n_eps=15, seed=0):
  random.seed(seed)
  llm = infer.LlamaCppWrapper(api_url=API, temperature=0.0, model_name="GELAB-ZERO-4B")
  eps = sorted(glob.glob(f"raw/full/{arm}/*/p0/steps.jsonl"))
  random.shuffle(eps)
  c = Counter(); out = []
  for f in eps[:n_eps]:
    d = Path(f).parent
    goal = json.load(open(d / "meta.json"))["goal"]
    rows = [json.loads(l) for l in open(f)]
    histA, histB = [], []
    for r in rows:
      if r.get("kind") != "model" or not r.get("pre"):
        if r.get("summary"):              # graph-executed steps: same text in both histories
          histA.append(r["summary"]); histB.append(r["summary"])
        continue
      img = d / "img" / f"{r['pre']['cid']}.jpg"
      if not img.exists():
        break
      ui = R.load_ui(d, r["pre"]["cid"]); im = Image.open(img).convert("RGB")
      res = {}
      for cond, h in (("A", histA), ("B", histB)):
        hist = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(h[-8:])) or "None yet."
        resp, _, _ = llm.predict_mm("", [], messages=gelab_agent.build_gelab_messages(goal, hist, im))
        try:
          t, p = tok_of(ui, resp)
        except Exception:  # pylint: disable=broad-except
          t, p = "?", {}
        res[cond] = (t, p)
        c[cond + "_same"] += t == (r.get("tok") or "")
      c["steps"] += 1
      c["A_eq_B"] += res["A"][0] == res["B"][0]
      histA.append(r.get("summary") or "")
      histB.append(gelab_agent._normalize_space(res["B"][1].get("explain")) or r.get("summary") or "")  # pylint: disable=protected-access
      out.append({"task": d.parts[-2], "step": r["step"], "orig": r.get("tok"), "A": res["A"][0], "B": res["B"][0]})
    print(d.parts[-2], dict(c), flush=True)
  n = c["steps"]
  print(f"[{arm}] steps {n}: A(original summaries) agree {c['A_same']} ({c['A_same']/n:.0%}); "
        f"B(early exit, explain history) agree {c['B_same']} ({c['B_same']/n:.0%}); A==B {c['A_eq_B']} ({c['A_eq_B']/n:.0%})")
  Path("reports/early_exit_offline.jsonl").write_text("\n".join(json.dumps(x) for x in out))


if __name__ == "__main__":
  main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
