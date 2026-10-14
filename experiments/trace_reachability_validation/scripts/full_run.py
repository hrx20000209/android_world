"""Full AndroidWorld run: BASE vs FULL design (one episode per invocation).

FULL = G (global continuous graph) + E1/E2 exploration + R1 verified jumps
       + R2 pre-observation reuse. Each module's measured motivation is in
       reports/full_design_zh.md.

G   : one global store/graph for the whole run (pages carry activity in their
      signatures, so apps do not collide); append-only transitions; stage-aware
      continuation counts (all action tokens) tagged by task template.
E1  : predicted-next validation probe - the trace top-1 slot on this page, if
      its outcome was not yet observed in THIS run instance (1 probe).
E2  : gap-page completion - page with no continuation counts at all: probe
      unknown safe out-edges (<= 3). Pages that already have trace support
      get no coverage probing.
R1  : verified jump - follow gated dominant continuations; navigation clicks
      with any-scope evidence, commit clicks only with SAME-task evidence;
      every hop verified, mismatch -> back + hand to model.
R2  : pre-observation reuse - if the model picks the slot probed in this
      window and the post-action page equals the probed page, skip the fixed
      1 s pause before the next observation.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("SEQ_ARM", os.environ.get("FULL_ARM", "BASE"))
import seq_run as S  # noqa: E402  (patches rgs_run.Episode; we override below)
import common as C  # noqa: E402
import rgs_run as R  # noqa: E402
import proc_replay as P  # noqa: E402
from android_world.env import adb_utils  # noqa: E402
from android_world.env import json_action  # noqa: E402

U = C.U
ARM = os.environ.get("FULL_ARM", "BASE")
APPDIR = Path(os.environ.get("SEQ_APPDIR", "/tmp/full"))


class FullEpisode(S.SeqEpisode):
  def run_agent(self, start_step=0, prior_summaries=None, label="agent"):
    goal, agent = self.task.goal, self.make_agent()
    gdir = APPDIR / ARM
    gdir.mkdir(parents=True, exist_ok=True)
    full = ARM.startswith("FULL")
    store = S.Store(gdir / "store.json") if full else None
    self._same = S.Store(gdir / "store.json", only_task=self.args.task) if (full and ARM not in ("FULL2", "FULL3", "FULL4", "FULL5")) else None
    self._last_model_commit = False
    pr = P.ProcReplay(goal) if ARM == "FULL5" else None
    graph = R.Graph(gdir / "graph.jsonl", gdir / "unrestorable.jsonl") if full else R.Graph(None)
    instance = f"{self.args.task}/{self.args.variant}"
    cfg = R.ExploreCfg(enabled=full, budget_s=6.0, max_probes=3, max_depth=1)
    ex = R.Explorer(self.dev, graph, cfg, instance, self.log)
    own, steps, used = [], [], Counter()
    m = Counter()
    done, prev, k = False, None, 0
    reuse_next = None
    while k < self.max_steps():
      if reuse_next is not None:
        # R2: model picked the probed slot; check the page without the fixed pause
        pre = self.dev.capture(f"s{k}_pre_reuse", settle=True, pause=None, full_res=True)
        if R.same_state(pre["ui"], reuse_next):
          m["reuse_hits"] += 1
        else:
          m["reuse_miss"] += 1
          pre = self.dev.capture(f"s{k}_pre", settle=True, pause=1.0, full_res=True)
        reuse_next = None
      else:
        pre = self.dev.capture(f"s{k}_pre", settle=True, pause=1.0, full_res=True)
      ui = pre["ui"]
      if prev is not None:
        self.ingest_agent(graph, instance, prev, pre)
        prev = None
      if pr is not None:      # R3: episode-local entity procedure replay
        pr.set_post(ui)
        plan = pr.plan(ui)
        if plan:
          m["replays"] += 1
          ok = self.exec_replay(plan, pr, pre, own, agent, steps, k, graph, instance)
          m["replay_steps"] += ok
          m["replay_full"] += ok == len(plan["template"])
          if ok:
            k += ok
            continue
      if store is not None:   # R1
        path = self.plan_jump(store, graph, own, ui, used)
        if path:
          m["jumps"] += 1
          ok = self.exec_jump(path, own, agent, steps, k, ex, graph, instance)
          m["jump_steps"] += ok
          m["jump_commit_steps"] += sum(1 for h in path[:ok] if h.get("commit_ok"))
          if ok < len(path):
            m["jump_rejected"] += 1
          self._last_model_commit = False
          if ok:
            k += ok
            continue
          pre = self.dev.capture(f"s{k}_pre_after_jump", settle=True, full_res=True)
          ui = pre["ui"]
      box = {}
      th = threading.Thread(target=lambda: box.update(d=self.decide(agent, goal, pre["pixels"])), daemon=True)
      th.start()
      xrec = self.window(ex, store, own, pre, k, instance) if full else None
      if xrec is not None:
        self.log("windows", xrec)
        m["probes_e1"] += sum(p["why"] == "E1" for p in xrec["probes"])
        m["probes_e2"] += sum(p["why"] == "E2" for p in xrec["probes"])
      th.join()
      d = box["d"]
      m["calls"] += 1
      requery = False
      if xrec and xrec["probes"]:
        v = self.dev.capture(f"s{k}_verify", settle=True)
        if not R.same_state(v["ui"], ui):
          requery = True
          m["requery"] += 1
          pre = self.dev.capture(f"s{k}_pre_requery", settle=True, full_res=True)
          ui = pre["ui"]
          d = self.decide(agent, goal, pre["pixels"])
          m["calls"] += 1
      a = d["action"]
      tok, fam, sel = C.action_token(ui, a.__dict__)
      if xrec and fam:
        hit = [p for p in xrec["probes"] if p["family"] == fam and p.get("dst_ui") is not None and p["rollback_ok"]]
        if hit:
          m["e1_hit" if hit[0]["why"] == "E1" else "e2_hit"] += 1
          reuse_next = hit[0]["dst_ui"]
      summary, exec_s = self.commit(agent, d, goal)
      own.append(S.own_entry(ui, tok, fam))
      self._last_model_commit = bool(fam) and is_commit_token(ui, fam)
      if pr is not None:
        pr.record(ui, a.__dict__, tok, sel, self._last_model_commit)
      row = {"step": k, "kind": "model", "pre": R.strip(pre), "action": a.__dict__, "tok": tok, "summary": summary,
             "model_latency_s": d["model_latency_s"], "requery": requery,
             "probes": [{k2: v2 for k2, v2 in p.items() if k2 != "dst_ui"} for p in xrec["probes"]] if xrec else []}
      steps.append(row)
      self.log("steps", row)
      prev = {"pre": pre, "slot_id": tok, "sel": sel, "action": a.__dict__, "step": k}
      k += 1
      if a.action_type in (json_action.STATUS, json_action.ANSWER):
        done = True
        break
    final = self.dev.capture("final", settle=True, pause=1.0, full_res=True)
    if prev is not None:
      self.ingest_agent(graph, instance, prev, final)
    self._own_seq = own
    res = {"steps": len(steps), "agent_steps": len(steps), "done": done, "model_calls": m["calls"],
           "final_cid": final["cid"], "n_probes": ex.n_probes, "explore_disabled": ex.disabled_reason,
           "store_eps": len(store.data) if store else 0}
    m["jump_absent"] = getattr(self, "_jump_absent", 0)
    m["jump_mismatch"] = getattr(self, "_jump_mismatch", 0)
    res.update({f"m_{kk}": vv for kk, vv in m.items()})
    return res

  def plan_jump(self, store, graph, own, ui, used):
    if ARM in ("FULL3", "FULL4", "FULL5"):
      return self.plan_jump_invariant(store, graph, own, ui, used)
    path = S.SeqEpisode.plan_jump(self, store, graph, own, ui, used)
    if ARM == "FULL2":
      for i, hop in enumerate(path):
        # irreversible commits only to CONFIRM what the model itself just chose
        hop["commit_ok"] = (i == 0 and self._last_model_commit)
    return path

  def plan_jump_invariant(self, store, graph, own, ui, used):
    """Jump only along continuations that are (a) gated within this task
    template, or (b) unanimous across >= 2 task templates (the cross-task
    reusable prefix). Majority across templates is NOT enough: it reflects
    which tasks happened to run, not this task's intent."""
    chain, sim_own = [], list(own)
    cur = {"struct": ui.struct_sig, "layout": ui.layout_sig}
    me = self.args.task
    for i in range(S.MAX_CHAIN):
      tok, key, info = store.predict(sim_own, cur)
      top = info.get("top")
      if top is None or not top.startswith("E|") or key is None:
        break
      doms = store.task_dominants(sim_own, cur, key)
      same = doms.get(me)
      same_ok = same is not None and same[0] == top and same[1] >= S.GATE_N and same[1] / same[2] >= S.GATE_D
      # cross-task: unanimous across >= 2 templates AND the agent has already
      # revealed its branch by an own (non open-app) action in the matched
      # context; at the task entry, templates diverge (0% cross-task top-1).
      agent_acted = any(not x["tok"].startswith(("OPEN_APP", "STATUS", "WAIT")) for x in own)
      cross_ok = (len(doms) >= 2 and all(d[0] == top for d in doms.values())
                  and info.get("order", 0) >= 1 and agent_acted)
      if ARM in ("FULL4", "FULL5"):
        cross_ok = False   # cross-task evidence is survivor-biased: only orders E1 probes
      # FULL5: confirming the model's OWN just-chosen commit needs no history
      # evidence beyond which dialog button follows (intent is the model's).
      confirm_ok = (ARM == "FULL5" and i == 0 and self._last_model_commit and is_dialog_button(ui, top))
      if not (same_ok or cross_ok or confirm_ok):
        break
      nxt = {k2: store.successor(k2, cur[k2], top) for k2 in ("struct", "layout")}
      if nxt["layout"] is None:
        break
      chain.append({"tok": top, "key": key, "expect": nxt, "gate": info, "scope": "same" if same_ok else ("cross" if cross_ok else "confirm"),
                    "commit_ok": (i == 0 and self._last_model_commit)})
      sim_own.append({"state": cur["struct"], "struct": cur["struct"], "layout": cur["layout"], "tok": top, "fam": top})
      cur = {k2: (nxt[k2] or "") for k2 in ("struct", "layout")}
    if not chain or used[(ui.struct_sig, chain[0]["tok"])] >= 2:
      return []
    used[(ui.struct_sig, chain[0]["tok"])] += 1
    return chain

  def exec_replay(self, plan, pr, pre, own, agent, steps, k, graph, instance):
    """Follow the model's own demonstrated procedure for the next goal entity;
    stop before its commit and on the first landing mismatch."""
    ok, cap = 0, pre
    for j, h in enumerate(plan["template"]):
      ui = cap["ui"]
      if j == 0:
        node = plan["node"]
        label = plan["entity"]
      else:
        slot = next((sl for sl in U.enumerate_slots(ui) if C.family_of_sel(sl.sel) == h["tok"] and sl.safe), None)
        if slot is None:
          self.log("steps", {"step": k + j, "kind": "replay_absent", "tok": h["tok"], "entity": plan["entity"]})
          break
        node = ui.nodes[slot.node_idx]
        label = U.label_of(ui, node) or slot.sel.get("rid") or slot.sel.get("cls")
      x, y = node.center
      if h["at"] == "long_press":
        adb_utils.long_press(int(x), int(y), self.dev.ctrl)
        summary = f"Long-pressed \"{label}\"."
      else:
        self.dev.tap(x, y)
        summary = f"Tapped \"{label}\"."
      post = self.dev.capture(f"s{k + j}_rpost", settle=True, pause=1.0 if j == 0 else None)
      match = post["ui"].activity == h["post_act"]
      action = {"action_type": h["at"], "x": int(x), "y": int(y)}
      tok, _, sel = C.action_token(ui, action)
      agent._actions.append({"summary": summary})  # pylint: disable=protected-access
      agent._summaries.append(summary)  # pylint: disable=protected-access
      own.append(S.own_entry(ui, tok, tok))
      pr.record(ui, action, tok, sel, False, origin="replay")
      pr.set_post(post["ui"])
      self.ingest_agent(graph, instance, {"pre": cap, "slot_id": tok, "sel": sel, "action": action, "step": k + j}, post)
      row = {"step": k + j, "kind": "replay", "entity": plan["entity"], "tok": tok, "match": match,
             "summary": summary, "pre": R.strip(cap), "post_cid": post["cid"]}
      steps.append(row)
      self.log("steps", row)
      ok += 1
      cap = post
      if not match:
        break
    pr.tried.add(plan["entity"])
    pr.open = plan["entity"]      # closed only by the model's own commit
    return ok

  def window(self, ex, store, own, pre, step, instance):
    """E1 then E2; nothing else."""
    t0 = time.time()
    rec = {"step": step, "probes": [], "stop": ""}
    if ex.disabled_reason:
      return dict(rec, stop="disabled", window_s=0.0)
    ui = pre["ui"]
    plan, _ = ex.plan(ui)
    safe = {s.slot_id: s for _, s in plan}
    by_fam = {}
    for _, s in plan:
      by_fam.setdefault(C.family_of_sel(s.sel), s)
    tok, key, info = store.predict(own, {"struct": ui.struct_sig, "layout": ui.layout_sig})
    top = info.get("top")
    rec["gate"] = info
    targets = []
    if top and top in by_fam:
      s = by_fam[top]
      seen_here = any(o["instance"] == instance for o in graph_obs(ex.graph, ui.struct_sig, s.slot_id))
      if not seen_here:
        targets.append(("E1", s))
    elif info.get("order", -1) < 0 and ARM not in ("FULLNE", "FULL2", "FULL3", "FULL4", "FULL5"):   # gap page (E2 off in FULLNE/FULL2/FULL3)
      for p, s in plan:
        if p == 0 and len(targets) < 3:      # unknown out-edges only
          targets.append(("E2", s))
    rec["mode"] = "E1" if targets and targets[0][0] == "E1" else ("E2" if targets else "none")
    for why, s in targets:
      if time.time() - t0 >= ex.cfg.budget_s:
        rec["stop"] = "budget"
        break
      p = ex.probe(pre, s, 0, step, depth=1)
      dst_ui = None
      try:
        dst_ui = R.load_ui(self.out, p["dst_cid"])
      except Exception:  # pylint: disable=broad-except
        pass
      rec["probes"].append({"why": why, "slot": s.slot_id, "family": C.family_of_sel(s.sel), "outcome": p["outcome"],
                            "rollback_ok": p["rollback_ok"], "t": p["t_total_s"], "dst_ui": dst_ui})
      if not p["rollback_ok"]:
        ex.disabled_reason = f"rollback_failed@{step}"
        rec["stop"] = "rollback_failed"
        break
    rec["window_s"] = round(time.time() - t0, 3)
    return rec


def is_commit_token(ui, fam):
  """The model's clicked slot is a commit (deny-listed label or dialog action)."""
  for sl in U.enumerate_slots(ui):
    if C.family_of_sel(sl.sel) == fam:
      return sl.unsafe_reason in ("unsafe_label", "dialog_action_button")
  return False


def is_dialog_button(ui, fam):
  for sl in U.enumerate_slots(ui):
    if C.family_of_sel(sl.sel) == fam:
      return sl.unsafe_reason == "dialog_action_button"
  return False


def graph_obs(graph, struct, slot_id):
  return graph.slots.get((struct, slot_id), [])


def _run(self):
  res = S._orig_run(self)  # pylint: disable=protected-access
  if ARM.startswith("FULL") and res.get("is_successful") == 1.0 and getattr(self, "_own_seq", None):
    S.Store(APPDIR / ARM / "store.json").append(self._own_seq, self.args.task)
  return res


def _setup(self):
  meta = S._orig_setup(self)  # pylint: disable=protected-access
  meta["full_arm"] = ARM
  (self.out / "meta.json").write_text(R.jdump(meta))
  return meta


FullEpisode.run = _run
FullEpisode.setup = _setup
R.Episode = FullEpisode

if __name__ == "__main__":
  R.main()
