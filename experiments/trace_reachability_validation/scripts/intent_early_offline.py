"""Offline: can the target of a click be identified from the model's own stream before the action fields?
For model click steps (element target) of a finished arm, re-decide with streaming on the lab server and record, at
each streamed chunk, the first moment the text so far (THINK / explain) names a label that matches exactly one
on-screen control. Compare that control with the final action's control, and the moment with the action-field moment."""
import glob, json, random, re, sys, time
from collections import Counter
from pathlib import Path
import requests
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402
from android_world.agents import gelab_agent  # noqa: E402
R, U, C = full9.R, full9.U, full9.C
API, MODEL = "http://127.0.0.1:18095/v1/chat/completions", "GELAB-ZERO-4B"
QUOTE = re.compile(r"[\"'“”‘’「」『』]([^\"'“”‘’「」『』]{1,40})[\"'“”‘’「」『』]")


def label_map(ui):
  m = {}
  for sl in U.enumerate_slots(ui):
    lab = (U.label_of(ui, ui.nodes[sl.node_idx]) or sl.sel.get("desc") or "").strip().lower()
    if lab:
      m.setdefault(lab, set()).add(C.family_of_sel(sl.sel))
  return m


def named_target(text, labels):
  for q in QUOTE.findall(text):
    q = q.strip().lower()
    if q in labels and len(labels[q]) == 1:
      return next(iter(labels[q]))
  return None


def main(arm, n=80, seed=1):
  random.seed(seed)
  samples = []
  for f in sorted(glob.glob(f"raw/full/{arm}/*/p0/steps.jsonl")):
    d = Path(f).parent
    rows = [json.loads(l) for l in open(f)]
    for i, r in enumerate(rows):
      if r.get("kind") == "model" and r.get("pre") and (r.get("action") or {}).get("action_type") == "click" \
          and r.get("tok", "").startswith(("E|", "L|")):
        samples.append((d, rows, i))
  random.shuffle(samples)
  c = Counter(); lead = []; lead_think = []
  for d, rows, i in samples[:n]:
    r = rows[i]
    img = d / "img" / f"{r['pre']['cid']}.jpg"
    if not img.exists():
      continue
    goal = json.load(open(d / "meta.json"))["goal"]
    hist = [x.get("summary") for x in rows[:i] if x.get("summary")]
    history = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(hist[-8:])) or "None yet."
    msgs = gelab_agent.build_gelab_messages(goal, history, Image.open(img).convert("RGB"))
    ui = R.load_ui(d, r["pre"]["cid"])
    labels = label_map(ui)
    t0 = time.time(); txt = ""; t_named = None; named = None; t_act = None
    with requests.post(API, json={"model": MODEL, "messages": msgs, "temperature": 0.0, "top_p": 1.0, "max_tokens": 1024,
                                  "stream": True}, stream=True, timeout=300) as resp:
      for line in resp.iter_lines():
        if not line or not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
          continue
        ch = (json.loads(line[6:])["choices"][0].get("delta") or {}).get("content") or ""
        txt += ch
        low = txt.lower()
        if named is None and "action" not in low.split("</think>")[-1]:
          named = named_target(txt, labels)
          if named:
            t_named = time.time() - t0
        if t_act is None and re.search(r"\tsummary\s*[:：]", txt):
          t_act = time.time() - t0
    t_end = time.time() - t0
    try:
      p = gelab_agent.parse_gelab_response(txt)
      a, _, _ = gelab_agent.gelab_action_to_json_action(p, (1080, 2400))
      final = C.action_token(ui, a.__dict__)[0]
    except Exception:  # pylint: disable=broad-except
      continue
    think = "<think>" in txt.lower()
    c["n"] += 1; c["think"] += think
    if named:
      c["named"] += 1; c["named_correct"] += named == final
      c["named_think"] += think; c["named_correct_think"] += think and named == final
      if named == final and t_act:
        lead.append((t_act - t_named) / max(1e-6, t_end))
        if think:
          lead_think.append(t_act - t_named)
  n = c["n"]
  import statistics as st
  print(f"[{arm}] click steps {n} (THINK {c['think']}): a target named before the action fields in {c['named']} "
        f"({c['named']/n:.0%}), correct {c['named_correct']} ({c['named_correct']/max(1,c['named']):.0%} of named); "
        f"THINK steps named {c['named_think']}, correct {c['named_correct_think']}; lead before the action fields "
        f"(correct, THINK steps) mean {st.mean(lead_think) if lead_think else 0:.2f}s on the lab server; "
        f"as a fraction of the response {st.mean(lead) if lead else 0:.0%}")


if __name__ == "__main__":
  main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 80)
