"""Closed-loop agent with graph-derived screenshot highlighting.

Arms (env HL_ARM):
  BASE  plain GELAB agent
  HL    highlight from the frozen LOO graph (history only)
  HLX   HL + model-free exploration in each inference window: probe one
        unknown safe slot (screen order, not yet observed in this episode),
        verified rollback; the observed successor is added to the episode's
        runtime evidence, so later highlights can use it. Model actions'
        own transitions are added too.
The prompt text is never changed; only the screenshot carries boxes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memory as M  # noqa: E402
import retrieval as RT  # noqa: E402
import highlight as H  # noqa: E402
import recommend as RC  # noqa: E402
from android_world.agents import gelab_agent  # noqa: E402

C, U, R = M.C, M.U, M.R
from android_world.env import json_action  # noqa: E402

ARM = os.environ.get("HL_ARM", "BASE")
EXTRA = {"text": ""}
_orig_build = gelab_agent.build_gelab_messages


def _build(goal, hist, img):
  msgs = _orig_build(goal, hist, img)
  if EXTRA["text"]:
    msgs[1]["content"][0]["text"] += "\n\n" + EXTRA["text"]
  return msgs


gelab_agent.build_gelab_messages = _build


class HLEpisode(R.Episode):
  def run_agent(self, start_step=0, prior_summaries=None, label="agent"):
    goal, agent = self.task.goal, self.make_agent()
    snap = M.Snapshot(self.args.task)
    ev = RT.Evidence(snap)
    ex = R.Explorer(self.dev, R.Graph(None), R.ExploreCfg(enabled=True, budget_s=6, max_probes=1, max_depth=1),
                    self.args.variant, self.log)
    m = {"calls": 0, "hl_steps": 0, "hl_from_runtime": 0, "probes": 0, "rollback_fail": 0, "requery": 0,
         "clicked_highlight": 0}
    done, k, prev = False, 0, None
    while k < self.max_steps():
      pre = self.dev.capture(f"s{k}_pre", settle=True, pause=1.0, full_res=True)
      ui = pre["ui"]
      if prev is not None and prev[1]:
        ev.add_runtime(prev[0], prev[1], ui)
      rows, info = H.select(ui, goal, ev) if ARM in ("HL", "HLX") else ([], {})
      img = Image.fromarray(pre["pixels"])
      EXTRA["text"] = ""
      if ARM == "REC":
        rec = RC.recommend(ui, goal, ev)
        if rec is not None:
          rows, info = [rec], {"source": rec["source"]}
          EXTRA["text"] = RC.text(ui, rec)
          m["hl_steps"] += 1
      if rows and ARM != "REC":
        img = H.draw(img, ui, rows)
        m["hl_steps"] += 1
        m["hl_from_runtime"] += any(r["status"] == "FRESH" for r in rows)
      box = {}
      th = threading.Thread(target=lambda: box.update(d=self.decide(agent, goal, np.array(img))), daemon=True)
      th.start()
      probe = None
      if ARM == "HLX" and not ex.disabled_reason:
        key = M.state_key(ui)
        for sl in U.enumerate_slots(ui):
          f = C.family_of_sel(sl.sel)
          if sl.safe and not sl.sel.get("in_list") and not snap.by_edge.get((key, f)) and (key, f) not in ev.runtime:
            p = ex.probe(pre, sl, 0, k, depth=1)
            m["probes"] += 1
            probe = {"fam": f, "rollback_ok": p["rollback_ok"], "outcome": p["outcome"]}
            try:
              ev.add_runtime(key, f, R.load_ui(self.out, p["dst_cid"]))
            except Exception:  # pylint: disable=broad-except
              pass
            if not p["rollback_ok"]:
              ex.disabled_reason = f"rollback_failed@{k}"
              m["rollback_fail"] += 1
            break
      th.join()
      d = box["d"]
      m["calls"] += 1
      if probe is not None:
        v = self.dev.capture(f"s{k}_verify", settle=True)
        if not R.same_state(v["ui"], ui):
          m["requery"] += 1
          pre = self.dev.capture(f"s{k}_pre_rq", settle=True, pause=1.0, full_res=True)
          ui = pre["ui"]
          rows, info = H.select(ui, goal, ev) if ARM in ("HL", "HLX") else ([], {})
          img = Image.fromarray(pre["pixels"])
          if rows:
            img = H.draw(img, ui, rows)
          d = self.decide(agent, goal, np.array(img))
          m["calls"] += 1
      a = d["action"]
      tok, fam, sel = C.action_token(ui, a.__dict__)
      if rows and fam in {r["fam"] for r in rows}:
        m["clicked_highlight"] += 1
      summary, _ = self.commit(agent, d, goal)
      self.log("steps", {"step": k, "tok": tok, "summary": summary, "hl": [r["fam"] for r in rows], "hl_info": info,
                         "probe": probe, "pre_cid": pre["cid"], "model_latency_s": d["model_latency_s"]})
      prev = (M.state_key(ui), fam)
      k += 1
      if a.action_type in (json_action.STATUS, json_action.ANSWER):
        done = True
        break
    return {"steps": k, "done": done, **{f"m_{kk}": v for kk, v in m.items()}}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--seed", type=int, default=32)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  args = argparse.Namespace(task=a.task, seed=a.seed, variant=f"hl_{ARM}", mode="agent", out=a.out, ref=None, graph=None,
                            explore=False, budget=6.0, max_probes=1, no_revalidate=False, recover=False,
                            no_graph_restore=False, app_memory=None, max_steps=0,
                            api_url="http://127.0.0.1:18095/v1/chat/completions", model="GELAB-ZERO-4B")
  res = HLEpisode(args).run()
  print("RESULT", json.dumps({k: v for k, v in res.items() if k != "traceback"}, default=str)[:1500])
  if "traceback" in res:
    print(res["traceback"])


if __name__ == "__main__":
  main()
