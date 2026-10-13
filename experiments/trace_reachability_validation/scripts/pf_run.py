"""Phase III: trace prefill with exploration-style verification (closed loop).

Arms (env PF_ARM):
  BASE  : plain agent (every step is a model call)
  PF    : trace prefill, NO verification (execute the trace's dominant action
          without a model call)
  PFV   : trace prefill + verification: execute the predicted action, observe,
          commit only if the page equals the historical successor page;
          otherwise roll back (probe-rollback ladder) and ask the model
  PFVS  : PFV with the strict gate
  PFVL  : PFV with traces from OTHER task templates only (leave-task-out)

Gate (counts only): successful-trace continuation counts at the longest
context order with support; prefill if top count >= N and share >= D
(loose N=3,D=0.8; strict N=5,D=0.9). Only label-identified (non list-row),
probe-safe click slots present on the current screen are prefilled; commits,
typing and list rows always go to the model. A (state, slot) is prefilled at
most twice per episode.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import pa_trace as PA  # noqa: E402
import rgs_run as R  # noqa: E402
from android_world.env import json_action  # noqa: E402

U = C.U
ARM = os.environ.get("PF_ARM", "BASE")
STORE = C.ROOT / "trace_index" / "success_store.json"
GATES = {"loose": (3, 0.8), "strict": (5, 0.9)}


def build_store():
  """Freeze all successful agent episodes available now (prev study + phase 2)."""
  dirs = [Path(e["ep"]) for e in C.episodes()]
  dirs += [d for d in sorted((C.ROOT / "raw" / "phase2").glob("*/*")) if (d / "result.json").exists()
           and "_partial_" not in d.name]
  out = []
  for d in dirs:
    m = json.loads((d / "meta.json").read_text())
    res = json.loads((d / "result.json").read_text())
    if res.get("is_successful") != 1.0:
      continue
    seq = PA.build_episode_seq({"ep": str(d), "task": m["task"]})
    lay = PA.build_episode_seq({"ep": str(d), "task": m["task"]}, key="layout_sig")
    out.append({"ep": str(d), "task": m["task"], "seq": [{"state": a["state"], "struct": a["struct"], "layout": b["state"],
                                                          "tok": a["tok"], "fam": a["fam"]} for a, b in zip(seq, lay)]})
  STORE.parent.mkdir(parents=True, exist_ok=True)
  STORE.write_text(json.dumps(out))
  print("store episodes", len(out), "tasks", len({o["task"] for o in out}))


class Predictor:
  """Struct-keyed continuation counts, backing off to layout-keyed counts when
  the struct context is unseen (titles that carry runtime data split struct
  states across variants)."""

  def __init__(self, task: str, loto: bool, gate: str, all_actions: bool = False):
    data = json.loads(STORE.read_text())
    if loto:
      data = [d for d in data if d["task"] != task]
    self.n_eps = len(data)
    self.idx = {"struct": PA.Index(), "layout": PA.Index()}
    self.succ = {"struct": defaultdict(Counter), "layout": defaultdict(Counter)}
    for d in data:
      for key in ("struct", "layout"):
        s = [dict(x, state=x[key]) for x in d["seq"]]
        # stage-aware: count every continuation (STATUS, TYPE, SWIPE, ...),
        # so a context whose successful continuation is "finish" or "type"
        # does not silently back off to a page-only context.
        self.idx[key].add_seq([dict(x, fam=x["tok"]) for x in s] if all_actions else s)
        for i in range(len(s) - 1):
          if s[i]["fam"]:
            self.succ[key][(s[i][key], s[i]["fam"])][s[i + 1][key]] += 1
    self.N, self.D = GATES[gate]

  def predict(self, own: list, ui: U.UIState):
    for key, sig in (("struct", ui.struct_sig), ("layout", ui.layout_sig)):
      seq = [dict(x, state=x[key]) for x in own] + [{"state": sig, "tok": "", "fam": None}]
      counts = self.idx[key].lookup(seq, len(seq) - 1)
      for k in range(PA.K_MAX, -1, -1):
        c = counts[k]
        if c and sum(c.values()) > 0:
          fam, n = c.most_common(1)[0]
          tot = sum(c.values())
          info = {"key": key, "order": k, "top": fam, "count": n, "total": tot}
          if n >= self.N and n / tot >= self.D:
            return fam, info
          return None, info
    return None, {"order": -1}

  def expected_successor(self, ui, fam, key):
    sig = ui.struct_sig if key == "struct" else ui.layout_sig
    c = self.succ[key].get((sig, fam))
    return c.most_common(1)[0][0] if c else None


class PFEpisode(R.Episode):
  def run_agent(self, start_step=0, prior_summaries=None, label="agent"):
    goal = self.task.goal
    agent = self.make_agent()
    arm = ARM
    pred = None
    if not arm.startswith("BASE"):
      pred = Predictor(self.args.task, loto=(arm == "PFVL"), gate="strict" if arm == "PFVS" else "loose",
                       all_actions=arm.startswith("PFVA"))
    graph = R.Graph(Path(self.args.graph) if self.args.graph else None)
    restorer = R.Explorer(self.dev, graph, R.ExploreCfg(enabled=False), f"{self.args.task}/{self.args.variant}", self.log)
    own, steps, used = [], [], Counter()
    done, n_calls, n_prefill, n_reject = False, 0, 0, 0
    k = 0
    while k < self.max_steps():
      pre = self.dev.capture(f"s{k}_pre", settle=True, pause=1.0, full_res=True)
      ui = pre["ui"]
      row = {"step": k, "label": label, "pre": R.strip(pre), "arm": arm, "prefill": False}
      fam, info = (None, {}) if pred is None else pred.predict(own, ui)
      row["gate"] = info
      slot = None
      if fam is not None and fam.startswith("E|") and used[(ui.struct_sig, fam)] < 2:
        for s in U.enumerate_slots(ui):
          if s.safe and C.family_of_sel(s.sel) == fam:
            slot = s
            break
      if slot is not None:
        used[(ui.struct_sig, fam)] += 1
        node = ui.nodes[slot.node_idx]
        key = info.get("key", "struct")
        exp = pred.expected_successor(ui, fam, key)
        t0 = time.time()
        self.dev.tap(*node.center)
        label_txt = U.label_of(ui, node) or slot.sel.get("rid") or slot.sel.get("cls")
        if arm == "PF":
          ok = True
          row["verify"] = "none"
        else:
          post = self.dev.capture(f"s{k}_pf_post", settle=True)
          got = post["ui"].struct_sig if key == "struct" else post["ui"].layout_sig
          ok = exp is not None and got == exp
          row["verify"] = "match" if ok else ("no_expected" if exp is None else "mismatch")
          row["pf_post_cid"] = post["cid"]
          if not ok:
            res = _restore(restorer, ui, post["ui"], node, k)
            row["rollback"] = res
            n_reject += 1
        row["prefill_s"] = round(time.time() - t0, 3)
        if ok:
          summary = f"Tapped \"{label_txt}\"."
          agent._actions.append({"summary": summary})  # pylint: disable=protected-access
          agent._summaries.append(summary)  # pylint: disable=protected-access
          row.update(prefill=True, fam=fam, summary=summary, action={"action_type": "click", "x": node.center[0], "y": node.center[1]},
                     model_latency_s=0.0, expected=exp)
          own.append({"state": ui.struct_sig, "struct": ui.struct_sig, "layout": ui.layout_sig, "tok": fam, "fam": fam})
          n_prefill += 1
          steps.append(row)
          self.log("steps", row)
          k += 1
          continue
        # rejected: re-observe the (restored) screen and fall through to the model
        pre = self.dev.capture(f"s{k}_pre_after_reject", settle=True, full_res=True)
        ui = pre["ui"]
        row["pre"] = R.strip(pre)
      d = self.decide(agent, goal, pre["pixels"])
      n_calls += 1
      summary, exec_s = self.commit(agent, d, goal)
      tok, f2, _ = C.action_token(ui, d["action"].__dict__)
      own.append({"state": ui.struct_sig, "struct": ui.struct_sig, "layout": ui.layout_sig, "tok": tok, "fam": f2})
      row.update(response=d["response"], parsed=d["parsed"], action=d["action"].__dict__, extras=d["extras"],
                 summary=summary, model_latency_s=d["model_latency_s"], exec_s=round(exec_s, 3), fam=f2,
                 parse_error=d["parse_error"])
      steps.append(row)
      self.log("steps", row)
      k += 1
      if d["action"].action_type in (json_action.STATUS, json_action.ANSWER):
        done = True
        break
    final = self.dev.capture("final", settle=True, pause=1.0, full_res=True)
    return {"steps": len(steps), "agent_steps": len(steps), "done": done, "model_calls": n_calls,
            "prefills": n_prefill, "prefill_rejected": n_reject, "final_cid": final["cid"],
            "store_episodes": pred.n_eps if pred else 0}


def _restore(ex, ui_target, ui_post, node, step):
  """Probe-rollback ladder (same as the explorer's)."""
  if R.same_state(ui_post, ui_target):
    return {"ok": True, "method": "none_needed"}
  ok, chk, method = False, None, "back"
  for i in range(2):
    ex.dev.back()
    chk = ex.dev.capture(f"s{step}_pf_rb{i}")
    if R.same_state(chk["ui"], ui_target):
      ok = True
      break
    if chk["ui"].struct_sig != ui_post.struct_sig:
      break
  if not ok and chk is not None:
    ok2, chk2, path = ex.restore_via_graph(chk["ui"], ui_target, step)
    if path:
      method = "back+graph"
    ok = ok2 or ok
  return {"ok": ok, "method": method}


_orig_setup = R.Episode.setup


def _setup(self):
  meta = _orig_setup(self)
  meta["pf_arm"] = ARM
  (self.out / "meta.json").write_text(R.jdump(meta))
  return meta


PFEpisode.setup = _setup
R.Episode = PFEpisode

if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "build_store":
    build_store()
  else:
    R.main()
