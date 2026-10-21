"""Offline test: is the model's <THINK> segment useful? Steps of SUCCESSFUL episodes of a finished arm (their actions
are a proxy for correct ones) are re-decided on the same screenshot and history:
  A: normal decoding (the model decides whether to think)          -> agreement with the original action
  B: THINK skipped (assistant turn pre-filled with "action:", the opening of a non-thinking output)     -> agreement with the original action
Reported overall and split by whether A produced a THINK segment; plus generated tokens and latency."""
import glob, json, random, sys, time
from collections import Counter
from pathlib import Path
import requests
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402
from android_world.agents import gelab_agent  # noqa: E402
R, U, C = full9.R, full9.U, full9.C
API, MODEL = "http://127.0.0.1:18095/v1/chat/completions", "GELAB-ZERO-4B"


def call(msgs, prefill=None):
  body = {"model": MODEL, "messages": msgs, "temperature": 0.0, "top_p": 1.0, "max_tokens": 1024}
  if prefill is not None:
    body["messages"] = msgs + [{"role": "assistant", "content": prefill}]
    body.update(add_generation_prompt=False, continue_final_message=True)
  t0 = time.time()
  js = requests.post(API, json=body, timeout=300).json()
  txt = (prefill or "") + js["choices"][0]["message"]["content"]
  return txt, js.get("usage", {}).get("completion_tokens"), time.time() - t0


def tok_of(ui, resp):
  p = gelab_agent.parse_gelab_response(resp)
  a, _, _ = gelab_agent.gelab_action_to_json_action(p, (1080, 2400))
  return C.action_token(ui, a.__dict__)[0]


def main(arm, n_steps=240, seed=0):
  random.seed(seed)
  samples = []
  for f in sorted(glob.glob(f"raw/full/{arm}/*/p0/steps.jsonl")):
    d = Path(f).parent
    if json.load(open(d / "result.json")).get("is_successful") != 1.0:
      continue
    rows = [json.loads(l) for l in open(f)]
    for i, r in enumerate(rows):
      if r.get("kind") == "model" and r.get("pre"):
        samples.append((d, rows, i))
  random.shuffle(samples)
  c = Counter(); log = []
  for d, rows, i in samples[:n_steps]:
    r = rows[i]
    img = d / "img" / f"{r['pre']['cid']}.jpg"
    if not img.exists():
      continue
    goal = json.load(open(d / "meta.json"))["goal"]
    hist = [x.get("summary") for x in rows[:i] if x.get("summary")]
    history = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(hist[-8:])) or "None yet."
    msgs = gelab_agent.build_gelab_messages(goal, history, Image.open(img).convert("RGB"))
    ui = R.load_ui(d, r["pre"]["cid"])
    try:
      ra, na, la = call(msgs)
      rb, nb, lb = call(msgs, prefill="action:")
    except Exception as e:  # pylint: disable=broad-except
      c["error"] += 1
      continue
    th = "<think>" in ra.lower()
    try:
      ta = tok_of(ui, ra)
    except Exception:  # pylint: disable=broad-except
      ta = "?"
    try:
      tb = tok_of(ui, rb)
    except Exception:  # pylint: disable=broad-except
      tb = "?"
    o = r.get("tok") or ""
    k = "think" if th else "nothink"
    c["n"] += 1; c["n_" + k] += 1
    c["A"] += ta == o; c["B"] += tb == o
    c["A_" + k] += ta == o; c["B_" + k] += tb == o
    c["tokA"] += na or 0; c["tokB"] += nb or 0
    c["latA"] += la; c["latB"] += lb
    log.append({"task": d.parts[-2], "step": r["step"], "orig": o, "A": ta, "B": tb, "think": th, "tokA": na, "tokB": nb})
    if c["n"] % 40 == 0:
      print(dict(c), flush=True)
  n = c["n"]
  print(f"[{arm}] steps {n}: A(normal) agree {c['A']} ({c['A']/n:.0%}), B(no THINK) agree {c['B']} ({c['B']/n:.0%}); "
        f"steps where A thought {c['n_think']}: A {c['A_think']} B {c['B_think']}; A did not think {c['n_nothink']}: "
        f"A {c['A_nothink']} B {c['B_nothink']}; tokens/step A {c['tokA']/n:.0f} B {c['tokB']/n:.0f}; "
        f"latency/step A {c['latA']/n:.2f}s B {c['latB']/n:.2f}s")
  Path(f"reports/think_offline_{arm}.jsonl").write_text("\n".join(json.dumps(x) for x in log))


if __name__ == "__main__":
  main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 240)
