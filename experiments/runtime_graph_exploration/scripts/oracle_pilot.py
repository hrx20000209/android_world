"""Stage 2: oracle exploration-utility pilot (UPPER BOUND, separately labeled).

For a selected BASE decision state: re-instantiate the same task instance
(same seed), replay the recorded actions up to the state, verify that the
live page matches, run real safe depth-1 probes (verified rollback), then
query GELAB-ZERO-4B with the identical screenshot / goal / history with and
without the observed child affordances, on two equivalent endpoints.

Closed loop (--cond M0|ME): same reconstruction; the first decision uses the
condition's prompt, then the standard agent continues; AndroidWorld's checker
decides success. Future trajectories were used only to pick states and probe
candidates (configs/oracle_pilot_labels.json), never inside a prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import memory as M  # noqa: E402

C, U, R = M.C, M.U, M.R
from android_world.agents import gelab_agent  # noqa: E402
from android_world.env import json_action  # noqa: E402

APIS = {"8095": "http://127.0.0.1:18095/v1/chat/completions", "8096": "http://127.0.0.1:18096/v1/chat/completions"}
STATES = json.loads((ROOT / "metrics" / "oracle_states_selected.json").read_text())
LABELS = {s["i"]: s for s in json.loads((ROOT / "configs" / "oracle_pilot_labels.json").read_text())["states"]}


def query(api, goal, history, img, extra):
  hist = "None yet." if not history else "\n".join(
      f"{i}. {h}" for i, h in enumerate(history[-8:], start=max(1, len(history) - 7)))
  msgs = gelab_agent.build_gelab_messages(goal, hist, img)
  if extra:
    msgs[1]["content"][0]["text"] += "\n\n" + extra
  payload = {"model": "GELAB-ZERO-4B", "messages": msgs, "temperature": 0.0, "top_p": 1.0, "max_tokens": 1024,
             "stream": False, "cache_salt": uuid.uuid4().hex}
  t0 = time.time()
  r = requests.post(api, json=payload, timeout=300, proxies={"http": None, "https": None})
  d = r.json()
  return d["choices"][0]["message"]["content"], time.time() - t0, d.get("usage", {})


def parse(text, screen):
  try:
    parsed = gelab_agent.parse_gelab_response(text)
    action, _, extras = gelab_agent.gelab_action_to_json_action(parsed, screen)
    return parsed, action, extras
  except Exception as e:  # pylint: disable=broad-except
    return {"parse_error": repr(e)}, json_action.JSONAction(action_type=json_action.WAIT), {"wait_seconds": 1}


def inside_text(ui, ad):
  if ad.get("x") is None:
    return ""
  n = U.element_at(ui, int(ad["x"]), int(ad["y"]))
  if n is None:
    return ""
  l, t, r, b = n.bounds
  out = []
  for m in ui.app_nodes():
    cx, cy = m.center
    if l <= cx < r and t <= cy < b:
      out += [m.text or "", m.desc or ""]
  return M.norm(" ".join(out))


def correct(label, tok, within):
  return any(a["tok"] in tok and (not a.get("within") or a["within"] in within) for a in label["accept"])


def title_of(ui):
  top = sorted((n for n in ui.app_nodes() if n.visible and n.text), key=lambda n: (n.bounds[1], n.bounds[0]))
  return top[0].text[:40] if top else ""


class Pilot(R.Episode):
  def run_agent(self, start_step=0, prior_summaries=None, label="agent"):
    a = self.args
    st, lab = STATES[a.state], LABELS[a.state]
    ref = Path(st["ep"])
    rows = [json.loads(l) for l in open(ref / "steps.jsonl")]
    rows = [r for r in rows if r.get("kind") == "model"]
    k = st["step_idx"]
    agent = self.make_agent()
    for r in rows[:k]:
      act = json_action.JSONAction(**{kk: v for kk, v in r["action"].items() if v is not None})
      if act.action_type in (json_action.STATUS, json_action.ANSWER):
        continue
      agent._execute_action(act, {})  # pylint: disable=protected-access
      time.sleep(1.0)
    pre = self.dev.capture("pilot_pre", settle=True, pause=1.0, full_res=True)
    ref_ui = R.load_ui(ref, st["cid"])
    match = ("struct" if pre["ui"].struct_sig == ref_ui.struct_sig else
             "layout" if pre["ui"].layout_sig == ref_ui.layout_sig else
             "activity" if pre["ui"].activity == ref_ui.activity else "none")
    history = [r.get("summary") or "" for r in rows[:k]]
    out = {"state": a.state, "task": st["task"], "kind": lab["kind"], "replay_match": match, "probes": []}
    evidence = ""
    if a.cond in ("pair", "ME"):
      ex = R.Explorer(self.dev, R.Graph(None), R.ExploreCfg(enabled=True, budget_s=120, max_probes=6, max_depth=1),
                      f"pilot{a.state}", self.log)
      lines = []
      for want in [w for w in lab["probe"] if w not in a.skip]:
        slot = next((s for s in U.enumerate_slots(pre["ui"]) if s.safe and want in C.family_of_sel(s.sel)), None)
        if slot is None:
          out["probes"].append({"want": want, "status": "absent"})
          continue
        p = ex.probe(pre, slot, 0, 0, depth=1)
        dst = R.load_ui(self.out, p["dst_cid"])
        aff = []
        for s2 in U.enumerate_slots(dst):
          t = s2.sel.get("label") or s2.sel.get("desc")
          if t and t not in aff:
            aff.append(t)
        node = pre["ui"].nodes[slot.node_idx]
        name = U.label_of(pre["ui"], node) or slot.sel.get("desc") or slot.sel.get("rid")
        out["probes"].append({"want": want, "status": "ok" if p["rollback_ok"] else "rollback_failed",
                              "outcome": p["outcome"], "t": p["t_total_s"], "dst_title": title_of(dst),
                              "aff": aff[:15], "rollback": p["rollback"]["method"]})
        if p["outcome"] not in ("no_change", "minor_change"):
          lines.append(f"- Tapping \"{name}\" at ({node.center[0]},{node.center[1]}) opens a screen "
                       f"titled \"{title_of(dst)}\" that offers: {', '.join(aff[:15])}")
        else:
          lines.append(f"- Tapping \"{name}\" at ({node.center[0]},{node.center[1]}) does not change the screen")
        if not p["rollback_ok"]:
          break
      if lines:
        evidence = ("Exploration evidence (each element below was actually tapped on this screen and the screen was "
                    "restored afterwards):\n" + "\n".join(lines))
      post = self.dev.capture("pilot_post", settle=True, pause=1.0, full_res=True)
      out["restored"] = R.same_state(post["ui"], pre["ui"])
      if not out["restored"] or any(p.get("status") == "rollback_failed" for p in out["probes"]):
        out["status"] = "restore_failed"     # never query on an unrestored screen
        (self.out / "pilot.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
        return {"steps": 0, "pilot": out}
      pre = post
    out["evidence"] = evidence
    img = Image.fromarray(pre["pixels"])
    ui = pre["ui"]
    goal = self.task.goal
    if a.cond == "pair":
      out["resp"] = {}
      for ep_name, api in APIS.items():
        for c in ("M0", "ME"):
          if c == "ME" and not evidence:
            continue
          text, lat, usage = query(api, goal, history, img, evidence if c == "ME" else "")
          parsed, action, _ = parse(text, self.env.logical_screen_size)
          ad = action.__dict__
          tok = C.action_token(ui, ad)[0]
          within = inside_text(ui, ad)
          out["resp"][f"{c}@{ep_name}"] = {"tok": tok, "within": within[:120], "correct": correct(lab, tok, within),
                                          "latency": round(lat, 2), "prompt_tokens": usage.get("prompt_tokens"),
                                          "summary": parsed.get("summary") if isinstance(parsed, dict) else None,
                                          "response": text}
      (self.out / "pilot.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
      return {"steps": 0, "pilot": {kk: v for kk, v in out.items() if kk != "resp"}}
    # closed loop: first decision with the condition's prompt, then the standard agent
    for h in history:
      agent._actions.append({"summary": h})  # pylint: disable=protected-access
      agent._summaries.append(h)  # pylint: disable=protected-access
    text, lat, usage = query(APIS["8095"], goal, history, img, evidence if a.cond == "ME" else "")
    parsed, action, extras = parse(text, self.env.logical_screen_size)
    first = C.action_token(ui, action.__dict__)[0]
    d = {"parsed": parsed if isinstance(parsed, dict) else {}, "action": action, "extras": extras or {},
         "tool_call": {}, "model_latency_s": lat}
    self.commit(agent, d, goal)
    n, done = 1, action.action_type in (json_action.STATUS, json_action.ANSWER)
    budget = self.max_steps() - k
    while not done and n < budget:
      cap = self.dev.capture(f"cl{n}", settle=True, pause=1.0, full_res=True)
      d = self.decide(agent, goal, cap["pixels"])
      self.commit(agent, d, goal)
      self.log("steps", {"step": k + n, "tok": C.action_token(cap["ui"], d["action"].__dict__)[0]})
      n += 1
      done = d["action"].action_type in (json_action.STATUS, json_action.ANSWER)
    out.update({"first": first, "first_correct": correct(lab, first, inside_text(ui, action.__dict__)),
                "cont_steps": n, "budget": budget})
    (self.out / "pilot.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return {"steps": n, "pilot": out}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--state", type=int, required=True)
  ap.add_argument("--cond", default="pair", choices=["pair", "M0", "ME"])
  ap.add_argument("--out", required=True)
  ap.add_argument("--skip", default="", help="';;'-separated probe wants to exclude")
  a = ap.parse_args()
  st = STATES[a.state]
  args = argparse.Namespace(task=st["task"], seed=st["seed"], variant=f"pilot{a.state}_{a.cond}", mode="agent",
                            out=a.out, ref=None, graph=None, explore=False, budget=6.0, max_probes=6,
                            no_revalidate=False, recover=False, no_graph_restore=False, app_memory=None,
                            max_steps=0, api_url=APIS["8095"], model="GELAB-ZERO-4B", state=a.state, cond=a.cond, skip=[w for w in a.skip.split(";;") if w])
  res = Pilot(args).run()
  print("RESULT", json.dumps({kk: v for kk, v in res.items() if kk != "traceback"}, ensure_ascii=False, default=str)[:3000])
  if "traceback" in res:
    print(res["traceback"])


if __name__ == "__main__":
  main()
