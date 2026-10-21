"""Offline feasibility of speculative next inference.

At step i the action fields are known ~3.5 s before the response ends; if the graph (as of step i, all origins) holds
the edge (page_i, control_i) with one recorded destination page and a capture of it, the next decision could start at
once on that capture, before the action is executed and before summary_i is decoded. For consecutive model steps
(i, i+1) of a finished arm:
  coverage    : the edge exists, deterministic, with a capture (from an earlier moment)
  landing     : the real page at i+1 equals the predicted destination page
  agreement   : decision i+1 vs the original one, made on
                A  real screenshot, real history (noise floor)
                B1 graph capture, history without summary_i (what speculation can use)
                B2 graph capture, real history (isolates the screenshot effect)
"""
import glob, json, random, sys
from collections import Counter
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402
import joint_graph as JG  # noqa: E402
from android_world.agents import gelab_agent, infer  # noqa: E402
R, U, C = full9.R, full9.U, full9.C
API = "http://127.0.0.1:18095/v1/chat/completions"


def hist_text(h):
  return "\n".join(f"{k + 1}. {s}" for k, s in enumerate(h[-8:])) or "None yet."


def tok_on(ui, resp):
  p = gelab_agent.parse_gelab_response(resp)
  a, _, _ = gelab_agent.gelab_action_to_json_action(p, (1080, 2400))
  return C.action_token(ui, a.__dict__)[0]


def main(arm, n_max=120, seed=0):
  random.seed(seed)
  raw = Path("raw/full") / arm
  lines = sorted((json.loads(l) for l in open(f"graph_snapshots/full/{arm}/graph.jsonl")), key=lambda o: o.get("t", 0))
  llm = infer.LlamaCppWrapper(api_url=API, temperature=0.0, model_name="GELAB-ZERO-4B")
  pairs = []
  c = Counter()
  for f in sorted(glob.glob(str(raw / "*" / "p0" / "steps.jsonl"))):
    d = Path(f).parent
    rows = [json.loads(l) for l in open(f)]
    idx = [i for i, r in enumerate(rows) if r.get("kind") == "model" and r.get("pre")]
    for a_, b_ in zip(idx, idx[1:]):
      if b_ != a_ + 1:
        continue                         # a graph step in between: not a plain consecutive pair
      c["pairs"] += 1
      r = rows[a_]
      if (r.get("action") or {}).get("action_type") != "click" or not r.get("tok") or r["tok"].startswith("CLICK"):
        continue
      c["click_pairs"] += 1
      pairs.append((d, rows, a_, b_))
  random.shuffle(pairs)
  for d, rows, a_, b_ in pairs:
    if c["asked"] >= n_max:
      break
    r, nx = rows[a_], rows[b_]
    t_i = r["pre"]["t"]
    gr = R.Graph(None)
    for o in lines:
      if o.get("t", 0) >= t_i:
        break
      gr.add(o)
    jg = JG.JointGraph(gr)
    ui = R.load_ui(d, r["pre"]["cid"])
    e = jg.edge(JG.page_key(ui), r["tok"])
    if not e or len(e["dst"]) != 1 or not e["ents"]:
      c["no_edge"] += 1
      continue
    c["covered"] += 1
    pd = JG.JointGraph.predicted(e)
    inst, cid, _ = e["ents"][-1]
    gdir = raw / inst.split("/")[0] / "p0"
    gimg = gdir / "img" / f"{cid}.jpg"
    if not gimg.exists():
      c["no_capture"] += 1
      continue
    ui_next = R.load_ui(d, nx["pre"]["cid"])
    landed = JG.page_key(ui_next) == pd
    c["landed"] += landed
    goal = json.load(open(d / "meta.json"))["goal"]
    hist = [x.get("summary") for x in rows[:b_] if x.get("summary")]
    real_img = Image.open(d / "img" / f"{nx['pre']['cid']}.jpg").convert("RGB")
    g_img = Image.open(gimg).convert("RGB").resize(real_img.size)
    out = {}
    for k, img, h in (("A", real_img, hist), ("B1", g_img, hist[:-1]), ("B2", g_img, hist)):
      resp, _, _ = llm.predict_mm("", [], messages=gelab_agent.build_gelab_messages(goal, hist_text(h), img))
      try:
        out[k] = tok_on(ui_next, resp)
      except Exception:  # pylint: disable=broad-except
        out[k] = "?"
    c["asked"] += 1
    o = nx.get("tok") or ""
    for k in out:
      c[k + "_agree"] += out[k] == o
      c[k + "_agree_landed"] += landed and out[k] == o
    if c["asked"] % 20 == 0:
      print(dict(c), flush=True)
  n = max(1, c["asked"])
  print(f"[{arm}] consecutive model pairs {c['pairs']}, element-click pairs {c['click_pairs']}; examined until {c['asked']} "
        f"asked: covered {c['covered']} (no edge {c['no_edge']}, no capture {c['no_capture']}); landed on predicted page "
        f"{c['landed']}/{c['asked']}; agreement A {c['A_agree']} B1 {c['B1_agree']} B2 {c['B2_agree']} of {n}; "
        f"among landed: A {c['A_agree_landed']} B1 {c['B1_agree_landed']} B2 {c['B2_agree_landed']} of {c['landed']}")


if __name__ == "__main__":
  main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 120)
