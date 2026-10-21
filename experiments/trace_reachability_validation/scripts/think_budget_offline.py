"""Bounded THINK: let the model think for at most K tokens, then close the segment and let it continue with the
action (assistant turn = the truncated THINK + "</THINK>\n", continue_final_message). Evaluated on the steps of
think_offline_<arm>.jsonl where the model thought (normal decoding); agreement with the original action."""
import json, sys
from collections import Counter
from pathlib import Path
import requests
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402
from android_world.agents import gelab_agent  # noqa: E402
import think_offline as T  # noqa: E402
R, C = full9.R, full9.C


def bounded(msgs, k):
  body = {"model": T.MODEL, "messages": msgs, "temperature": 0.0, "top_p": 1.0, "max_tokens": k}
  js = requests.post(T.API, json=body, timeout=300).json()
  part = js["choices"][0]["message"]["content"]
  n1 = js.get("usage", {}).get("completion_tokens") or 0
  if js["choices"][0].get("finish_reason") != "length":
    return part, n1                       # finished within the budget
  if "</think>" in part.lower() or "<think>" not in part.lower():
    txt, n2, _ = T.call(msgs, prefill=part)   # THINK already closed (or none): just continue the output
  else:
    txt, n2, _ = T.call(msgs, prefill=part.rstrip() + " </THINK>\n")   # budget reached inside THINK: close it
  return txt, n1 + (n2 or 0)


def main(arm, ks=(40, 80)):
  X = [json.loads(l) for l in open(f"reports/think_offline_{arm}.jsonl")]
  X = [x for x in X if x["think"]]
  c = Counter()
  for x in X:
    d = Path(f"raw/full/{arm}/{x['task']}/p0")
    rows = [json.loads(l) for l in open(d / "steps.jsonl")]
    i = next(i for i, r in enumerate(rows) if r.get("step") == x["step"] and r.get("kind") == "model")
    r = rows[i]
    goal = json.load(open(d / "meta.json"))["goal"]
    hist = [y.get("summary") for y in rows[:i] if y.get("summary")]
    history = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(hist[-8:])) or "None yet."
    msgs = gelab_agent.build_gelab_messages(goal, history, Image.open(d / "img" / f"{r['pre']['cid']}.jpg").convert("RGB"))
    ui = R.load_ui(d, r["pre"]["cid"])
    c["n"] += 1
    c["A"] += x["A"] == x["orig"]; c["B"] += x["B"] == x["orig"]
    c["tokA"] += x["tokA"] or 0; c["tokB"] += x["tokB"] or 0
    for k in ks:
      try:
        txt, n = bounded(msgs, k)
        t = T.tok_of(ui, txt)
      except Exception:  # pylint: disable=broad-except
        t, n = "?", 0
      c[f"K{k}"] += t == x["orig"]; c[f"tokK{k}"] += n
      c[f"K{k}_vsA_harm"] += x["A"] == x["orig"] and t != x["orig"]
      c[f"K{k}_vsA_help"] += x["A"] != x["orig"] and t == x["orig"]
  n = c["n"]
  print(f"[{arm}] THINK steps {n}: normal {c['A']} ({c['tokA']/n:.0f} tok), no-think {c['B']} ({c['tokB']/n:.0f} tok); " +
        "; ".join(f"THINK<= {k}: {c[f'K{k}']} ({c[f'tokK{k}']/n:.0f} tok, worse {c[f'K{k}_vsA_harm']} better {c[f'K{k}_vsA_help']})" for k in ks))


if __name__ == "__main__":
  main(sys.argv[1])
