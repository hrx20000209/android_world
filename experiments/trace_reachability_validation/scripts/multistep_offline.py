"""Offline feasibility of the multi-action interface with graph trajectories in the prompt (no emulator).

For model steps of a finished arm: rebuild the graph as of that step, list the known paths from the current screen
(inference + exploration edges, <= 2 hops, most goal-relevant first), each hop with the control's label and
0-1000 coordinates on the screen where it is tapped and a short description of the page it leads to. Ask the
model (same endpoint, same screenshot and history) for the next action, allowing up to 2 further actions taken
from the listed paths. Compare:
  first action  : agreement with the model's original choice at that step (risk to success)
  next actions  : agreement with the model's actual following actions (savings), mapped through the graph capture
"""
import glob
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9  # noqa: E402
import explore_value as EV  # noqa: E402
import joint_graph as JG  # noqa: E402
from android_world.agents import gelab_agent, infer  # noqa: E402

R, U, C = full9.R, full9.U, full9.C
API = "http://127.0.0.1:18095/v1/chat/completions"
MULTI_RULE = """

已知路径：用户消息里列出了"当前屏幕出发的已知路径"（之前在这个 app 里实际走过或探索过的转移，每一跳给出控件、坐标和到达的页面）。
多步输出：如果你本步要点的正是某条已知路径的第一跳，并且你确定之后还要继续沿这条路径点下去，就在本步输出行的末尾追加字段 then:x,y（最多两跳，用 ; 分隔，坐标照抄已知路径），例如：
explain:进入新建页面\taction:CLICK\tpoint:930,880\tthen:500,120\tsummary:...
不确定就不要追加 then。"""


def norm_pt(ui, node):
  w, h = 1080, 2400
  x, y = node.center
  return [int(x * 1000 / w), int(y * 1000 / h)]


def page_desc(dui):
  labs = [(n.text or n.desc) for n in dui.app_nodes() if n.visible and (n.text or n.desc)]
  return "、".join(dict.fromkeys(l[:24] for l in labs))[:120]


def known_paths(jg, ui, raw, g, max_paths=4):
  w = EV.w_app(ui.package)
  starts = [p for p in jg.out if p[0] == ui.package and p[1] == ui.activity]
  live = {C.family_of_sel(t.sel): t for t in U.enumerate_slots(ui)}
  paths = jg.paths(ui, lambda p: True, max_hops=2, safe=lambda s: JG.safe_control(s) or JG.entry_control(s), starts=starts)
  scored = []
  seen = set()
  for pa in paths:
    if not pa or pa[0][0] not in live:
      continue
    key = tuple(h[0] for h in pa)
    if key in seen:
      continue
    seen.add(key)
    dst = pa[-1][2]
    ents = jg.into.get(dst, [])
    dui = EV.load_cap(raw, ents[-1][0], ents[-1][1]) if ents else None
    rel = EV.rel(g, EV.page_tokens(dui), w) if dui is not None else 0.0
    scored.append((rel, pa))
  scored.sort(key=lambda x: -x[0])
  out = []
  for rel, pa in scored[:max_paths]:
    hops, cur = [], ui
    ok = True
    for i, (fam, sel, dst, origins) in enumerate(pa):
      if i == 0:
        sl = live[fam]
        node = ui.nodes[sl.node_idx]
      else:
        src_ents = jg.into.get(pa[i - 1][2], [])
        cur = EV.load_cap(raw, src_ents[-1][0], src_ents[-1][1]) if src_ents else None
        sl = next((t for t in U.enumerate_slots(cur) if C.family_of_sel(t.sel) == fam), None) if cur is not None else None
        if sl is None:
          ok = False
          break
        node = cur.nodes[sl.node_idx]
      ents = jg.into.get(dst, [])
      dui = EV.load_cap(raw, ents[-1][0], ents[-1][1]) if ents else None
      lab = U.label_of(cur, node) or sel.get("desc") or sel.get("rid") or "?"
      hops.append({"fam": fam, "label": lab, "point": norm_pt(cur, node), "dst": page_desc(dui) if dui else "?",
                   "origin": "+".join(sorted(origins))})
    if ok:
      out.append(hops)
  return out


def render(paths):
  lines = []
  for i, hops in enumerate(paths, 1):
    s = " → ".join(f"点「{h['label']}」{h['point']} 到达[{h['dst']}]" for h in hops)
    lines.append(f"{i}. {s}")
  return "\n".join(lines)


def main(arm, n_samples=60, seed=0):
  random.seed(seed)
  raw = Path("raw/full") / arm
  lines = sorted((json.loads(l) for l in open(f"graph_snapshots/full/{arm}/graph.jsonl")), key=lambda o: o.get("t", 0))
  samples = []
  for f in sorted(glob.glob(str(raw / "*" / "p0" / "steps.jsonl"))):
    d = Path(f).parent
    rows = [json.loads(l) for l in open(f)]
    model = [r for r in rows if r.get("kind") == "model" and r.get("pre")]
    for i, r in enumerate(model[:-1]):
      if "launcher" in (r["pre"]["summary"].get("package") or ""):
        continue
      samples.append((d, rows, model, i))
  random.shuffle(samples)
  llm = infer.LlamaCppWrapper(api_url=API, temperature=0.0, model_name="GELAB-ZERO-4B")
  c = Counter()
  log = []
  for d, rows, model, i in samples:
    if c["asked"] >= n_samples:
      break
    r = model[i]
    goal = json.load(open(d / "meta.json"))["goal"]
    g = EV.goal_vector(goal, full9.FuncExec(type("E", (), {"args": type("a", (), {"task": d.parts[-2]})()})(), goal))
    ui = R.load_ui(d, r["pre"]["cid"])
    gr = R.Graph(None)
    for o in lines:
      if o.get("t", 0) >= r["pre"]["t"]:
        break
      gr.add(o)
    jg = JG.JointGraph(gr)
    paths = known_paths(jg, ui, raw, g)
    if not paths:
      continue
    img = d / "img" / f"{r['pre']['cid']}.jpg"
    if not img.exists():
      continue
    # history = the summaries before this step (model + graph rows), as the agent saw them
    idx = rows.index(r)
    hist = [x.get("summary") for x in rows[:idx] if x.get("summary")]
    history = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(hist[-8:])) or "None yet."
    msgs = gelab_agent.build_gelab_messages(goal, history, Image.open(img).convert("RGB"))
    import os  # noqa: E402
    if not os.environ.get("NOPATH"):
      msgs[0]["content"][0]["text"] = gelab_agent.GELAB_SYSTEM_PROMPT + MULTI_RULE
      msgs[1]["content"][0]["text"] += "\n\n当前屏幕出发的已知路径：\n" + render(paths)
    t0 = time.time()
    try:
      resp, _, _ = llm.predict_mm("", [], messages=msgs)
    except Exception as e:  # pylint: disable=broad-except
      c["error"] += 1
      continue
    c["asked"] += 1
    try:
      parsed = gelab_agent.parse_gelab_response(resp)
      act, _, _ = gelab_agent.gelab_action_to_json_action(parsed, (1080, 2400))
    except Exception:  # pylint: disable=broad-except
      c["parse_fail"] += 1
      continue
    # first action agreement (by control family on the live screen, or same action type for non-clicks)
    orig = r.get("tok") or ""
    if act.action_type in ("click", "long_press") and act.x is not None:
      n = U.element_at(ui, int(act.x), int(act.y))
      new_tok = C.action_token(ui, act.__dict__)[0] if n is not None else "?"
    else:
      new_tok = C.action_token(ui, act.__dict__)[0]
    c["first_same"] += new_tok == orig
    m = re.search(r"then\s*[:：]\s*([0-9 ,;]+)", resp)
    extra = []
    if m:
      for part in m.group(1).split(";"):
        nums = re.findall(r"\d+", part)
        if len(nums) >= 2:
          extra.append({"action_type": "CLICK", "point": [int(nums[0]), int(nums[1])]})
    if extra:
      c["gave_next"] += 1
      c["next_actions"] += len(extra)
      # map each next action's point to a hop of a listed path; check against the model's actual next actions
      actual = [x.get("tok") or "" for x in model[i + 1:i + 1 + len(extra)]]
      hopmap = {tuple(h["point"]): h["fam"] for hops in paths for h in hops}
      for j, a in enumerate(extra):
        pt = tuple(a.get("point") or [])
        fam = hopmap.get(pt)
        c["next_on_path"] += fam is not None
        c["next_correct"] += fam is not None and j < len(actual) and actual[j] == fam
    log.append({"task": d.parts[-2], "step": r["step"], "orig": orig, "new": new_tok, "extra": extra,
                "actual_next": [x.get("tok") for x in model[i + 1:i + 3]], "paths": paths,
                "resp_tail": resp[-300:], "lat": round(time.time() - t0, 2)})
  print(f"[{arm}] asked {c['asked']}; first action unchanged {c['first_same']} ({c['first_same'] / max(1, c['asked']):.0%}); "
        f"gave next_actions {c['gave_next']} ({c['gave_next'] / max(1, c['asked']):.0%}); next actions {c['next_actions']}, "
        f"on a listed path {c['next_on_path']}, equal to the model's actual following action {c['next_correct']}; "
        f"parse fail {c['parse_fail']}, errors {c['error']}")
  out = Path(f"/private/tmp/claude-501/-Users-huangrunxi-Projects-android-world/1dd310ad-f151-42bc-a619-4cc3aef047ac/scratchpad/multistep_{arm}.jsonl")
  with out.open("w") as fo:
    for x in log:
      fo.write(json.dumps(x, ensure_ascii=False) + "\n")
  print("log:", out)


if __name__ == "__main__":
  main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60)
