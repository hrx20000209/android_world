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

import json
import math
import re
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
import full7  # noqa: E402
import full9  # noqa: E402
import explore_value as EV  # noqa: E402
import joint_graph as JG  # noqa: E402
import intent_graph as IG  # noqa: E402
import term_graph as TG  # noqa: E402
from android_world.env import adb_utils  # noqa: E402
from android_world.env import json_action  # noqa: E402

U = C.U
ARM_NAME = os.environ.get("FULL_ARM", "BASE")
# replicate arms: "FULL19R" (or "FULL19R2") behaves exactly like FULL19 but keeps its own graph directory
# "FULL23W": FULL23 behaviour, warm-started with execution-validation evidence from earlier runs (seeded files)
# "FULL29X": ablation - FULL29 behaviour with every exploration probe switched off (graph built from inference only)
ARM = re.sub(r"(R\d*|W|X)$", "", ARM_NAME) if re.match(r"^FULL\d+(R\d*|W|X)$", ARM_NAME) else ARM_NAME
NOEXP = bool(re.match(r"^FULL\d+X$", ARM_NAME))
APPDIR = Path(os.environ.get("SEQ_APPDIR", "/tmp/full"))


class FullEpisode(S.SeqEpisode):
  def run_agent(self, start_step=0, prior_summaries=None, label="agent"):
    goal, agent = self.task.goal, self.make_agent()
    gdir = APPDIR / ARM_NAME
    gdir.mkdir(parents=True, exist_ok=True)
    full = ARM.startswith("FULL")
    store = S.Store(gdir / "store.json") if full else None
    self._same = S.Store(gdir / "store.json", only_task=self.args.task) if (full and ARM not in ("FULL2", "FULL3", "FULL4", "FULL5", "FULL7", "FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30")) else None
    self._last_model_commit = False
    # FULL9: the function graph subsumes episode-local replay (its own segments become functions)
    pr = P.ProcReplay(goal, enum_only=(ARM in ("FULL7", "FULL8"))) if ARM in ("FULL5", "FULL7", "FULL8") else None
    fx = full7.StuckExplorer(self, goal, gdir, hard=(ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"))) if ARM in ("FULL7", "FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else None
    fn = full9.FuncExec(self, goal) if ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else None
    self._exec_log = []           # FULL11: actions executed since the last app launch (replayable path)
    self._agent = agent
    self._probe_states = {}       # FULL11: (layout key, slot family) first observed by exploration -> step
    if fn is not None:
      fn.exec_log = self._exec_log
    self._fn_ops = fn.P["ops"] if fn is not None else []
    self._fn = fn
    self._gvec = EV.goal_vector(goal, fn) if ARM in ("FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else {}
    self._pvec = EV.param_vector(fn) if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else {}
    self._rv_seen = set()
    self._page_acts, self._rec_pages, self._pend = {}, {}, None      # FULL18: episode-local page -> model actions
    self._reuse_brake, self._last_reuse_step = False, -9
    self._macro_brake, self._last_macro_step, self._macro_used = False, -9, set()   # FULL23
    self._intent = IG.EdgeIntent(gdir / "edge_use.jsonl", self.args.task) if ARM in ("FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else None
    if ARM == "FULL27":
      # speculation drafts: exploration contributes transitions only, intent comes from inference trajectories
      self._tmat = TG.TermMatrix(gdir / "graph.jsonl", self.out.parents[1], self.args.task, w_explore=0.0)
      self._gvec_unit = TG._unit(EV.goal_vector(goal, fn))  # pylint: disable=protected-access
    self._spec = None
    if ARM in ("FULL24", "FULL26"):
      # node x term matrix over inference + exploration trajectories of the run (cross-task, leave-own-task-out)
      self._tmat = TG.TermMatrix(gdir / "graph.jsonl", self.out.parents[1], self.args.task)
      self._gvec_unit = TG._unit(EV.goal_vector(goal, fn))  # pylint: disable=protected-access
    self._eff_path = gdir / "effectful.jsonl"
    self._learned_eff = set()
    if ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and self._eff_path.exists():
      self._learned_eff = {tuple(json.loads(l)["key"]) for l in self._eff_path.open() if l.strip()}
    self._grp_path = gdir / "unrestorable_groups.jsonl"
    self._bad_groups = set()
    if ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and self._grp_path.exists():
      self._bad_groups = {tuple(json.loads(l)["key"]) for l in self._grp_path.open() if l.strip()}
    self._d2_path = gdir / "d2_unsafe.jsonl"
    self._d2_bad = set()
    if ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and self._d2_path.exists():
      self._d2_bad = {tuple(json.loads(l)["key"]) for l in self._d2_path.open() if l.strip()}
    if fn is not None and ARM in ("FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
      fn.strict_done = True
    if fn is not None and ARM in ("FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
      fn.form_retrigger = True
    if fn is not None and ARM in ("FULL28", "FULL29", "FULL30"):
      fn.form_v2 = True          # hint-aware, synonyms, single-field records
    if fn is not None and ARM in ("FULL29", "FULL30"):
      fn.form_v3 = True          # each record field typed once; the head field only into a title / name field
    self._rb_fail = 0
    if ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):     # known environment drift: the SMS role must belong to the benchmark's SMS app
      import subprocess as _sp
      _sp.run([R.ADB, "-s", "emulator-5554", "shell", "cmd", "role", "add-role-holder", "android.app.role.SMS",
               "com.simplemobiletools.smsmessenger"], capture_output=True, timeout=30)
    graph = R.Graph(gdir / "graph.jsonl", gdir / "unrestorable.jsonl") if full else R.Graph(None)
    instance = f"{self.args.task}/{self.args.variant}"
    cfg = R.ExploreCfg(enabled=full, budget_s=(30.0 if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else 6.0), max_probes=3, max_depth=1)
    ex = R.Explorer(self.dev, graph, cfg, instance, self.log)
    own, steps, used = [], [], Counter()
    m = Counter()
    done, prev, k = False, None, 0
    booted = False
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
      if fx is not None and prev is not None:
        fx.observe_model(prev, ui)
      if prev is not None:
        self.ingest_agent(graph, instance, prev, pre)
        prev = None
      if ARM in ("FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and self._pend is not None:
        P0, fam0, rec0, first0 = self._pend
        self._page_acts.setdefault(P0, []).append({"fam": fam0, "land": JG.page_key(ui), "rec": rec0, "first": first0})
        self._pend = None
      if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and k == 0 and not booted:
        booted = True
        app = full7.resolve_app(goal, ui)
        if app:
          # graph reuse before inference: the instruction names an app whose launch transition the graph has
          # observed; launch it without a model call and without a counted step (the model's first decision
          # is made on the app's first screen within the same counted step)
          full7.launch(self.env, app)
          self._exec_log[:] = [{"action": {"action_type": "open_app", "app_name": app}, "commit": False}]
          summ = f"[system] Opened the app \"{app}\"."
          agent._actions.append({"summary": summ})  # pylint: disable=protected-access
          agent._summaries.append(summ)  # pylint: disable=protected-access
          row0 = {"step": k, "kind": "bootstrap", "tok": f"OPEN_APP:{app}", "summary": summ, "pre": R.strip(pre)}
          steps.append(row0)
          self.log("steps", row0)
          m["bootstrap"] += 1
          m["env_actions_extra"] += 1
          pre = full7.capture_stable(self.dev, f"s{k}_pre_boot")
          ui = pre["ui"]
      if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and self._last_model_commit:
        # the model itself just chose a commit; the dialog that follows is confirmed with its positive button
        n_c = full7.confirm_own_commit(self, ui, agent, steps, k, own, pure=(ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30")))
        self._last_model_commit = False
        if n_c:
          self._exec_log.append({"action": None, "commit": True})
          m["jump_steps"] += 1
          m["jump_commit_steps"] += 1
          k += 1
          continue
      if fx is not None and ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and fx.needs_back(ui):
        # hard rule: the same navigation control tapped 3 times without any effect -> one Back
        self.dev.back()
        summ = "[system] Pressed Back: the last control was tapped 3 times without any effect."
        agent._actions.append({"summary": summ})  # pylint: disable=protected-access
        agent._summaries.append(summ)  # pylint: disable=protected-access
        steps.append({"step": k, "kind": "unstick", "tok": "NAVIGATE_BACK", "summary": summ})
        self.log("steps", steps[-1])
        m["unstick_back"] += 1
        self._exec_log.append({"action": {"action_type": "navigate_back"}, "commit": False})
        k += 1
        continue
      if fn is not None:      # FULL9: function graph (parameterized sub-trajectories), before inference
        if ARM == "FULL14":
          self.log("fn_gate", {"step": k, "rec": fn.next_rec, "tried": sorted(fn.tried), "form_tried": sorted(getattr(fn, "form_tried", set())),
                               "applied": fn.applied_rec, "model_rec": fn.model_rec, "done": sorted(fn.done), "act": ui.activity})
        n_f = fn.try_apply(pre, agent, steps, k, own)
        if not n_f and ARM == "FULL11":
          n_f = fn.try_apply_via_edge(self, graph, pre, agent, steps, k, own, m)
        elif not n_f and ARM in ("FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
          n_f = fn.try_apply_via_edge2(self, graph, pre, agent, steps, k, own, m)
        if not n_f and ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
          # function routing over the joint graph (agent + exploration pages), <= 3 safe hops, one counted step
          n_f = JG.try_route(fn, self, graph, pre, agent, steps, k, own, m)
        if not n_f and ARM in ("FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
          # schema-level form function, re-triggered wherever this record's empty fields appear
          n_f = fn.try_form(pre, agent, steps, k, own, m, via="here")
        if not n_f and ARM == "FULL14":
          # schema-level form function: on this screen, else one observed edge away (agent or exploration)
          n_f = fn.try_form(pre, agent, steps, k, own, m, via="here") or fn.try_form_via_edge(self, graph, pre, agent, steps, k, own, m)
        if n_f:
          m["func_apps"] += 1
          m["func_steps"] += n_f
          m["env_actions_extra"] += n_f - 1
          self._last_model_commit = False
          k += 1
          continue
      if ARM in ("FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and not self._reuse_brake and not full9.transient(ui):
        n_r = self.try_reuse(ui, pre, agent, steps, k, own, m, fn)
        if n_r:
          self._last_model_commit = False
          k += 1
          continue
      if ARM in ("FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and not self._macro_brake and not full9.transient(ui):
        # goal-conditioned navigation macro over the unified graph (design review P0-2)
        n_m = IG.try_macro(self, ui, pre, agent, steps, k, own, m, graph,
                           picker=(TG.pick if ARM in ("FULL24", "FULL26") else None))
        if n_m:
          self._last_model_commit = False
          k += 1
          continue
      if fx is not None:      # first-run screens: handled by a separate module, outside the main algorithm
        n_ob = fx.onboarding(ui, agent, steps, k)
        if n_ob:
          m["onboarding_steps"] += n_ob
          k += 1 if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else n_ob
          m["env_actions_extra"] += (n_ob - 1) if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else 0
          continue
      if pr is not None:      # R3: episode-local entity procedure replay
        pr.set_post(ui)
        plan = pr.plan(ui)
        if plan:
          m["replays"] += 1
          ok = self.exec_replay(plan, pr, pre, own, agent, steps, k, graph, instance)
          m["replay_steps"] += ok
          m["replay_full"] += ok == len(plan["template"])
          if ok:
            k += 1 if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else ok      # FULL8: a multi-hop graph execution is one counted step
            m["env_actions_extra"] += (ok - 1) if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else 0
            continue
      if store is not None:   # R1
        path = self.plan_jump(store, graph, own, ui, used)
        if path:
          m["jumps"] += 1
          ok = self.exec_jump(path, own, agent, steps, k, ex, graph, instance)
          m["jump_steps"] += ok
          m["jump_commit_steps"] += sum(1 for h in path[:ok] if h.get("commit_ok"))
          if ok:
            self._exec_log.append({"action": None, "commit": True})   # untracked taps: no replay restoration
          if ok < len(path):
            m["jump_rejected"] += 1
          self._last_model_commit = False
          if ok:
            k += 1 if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else ok
            m["env_actions_extra"] += (ok - 1) if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else 0
            continue
          pre = self.dev.capture(f"s{k}_pre_after_jump", settle=True, full_res=True)
          ui = pre["ui"]
      stuck_text = fx.hint(ui) if fx is not None else ""
      full7.EXTRA["text"] = stuck_text
      if stuck_text:
        m["stuck_hints"] += 1
      box = {}
      def _decide():
        try:
          box["d"] = self.decide(agent, goal, pre["pixels"])
        except Exception as e:  # pylint: disable=broad-except
          box["err"] = e
      th = threading.Thread(target=_decide, daemon=True)
      th.start()
      self._infer_thread, self._spec = th, None
      if NOEXP:
        xrec = None
      elif fx is not None and fx.is_stuck(ui) and not ex.disabled_reason:
        xrec = fx.window(ex, pre, k)          # stuck: probe an untried control instead of E1
        m["probes_stuck"] += len(xrec["probes"])
      else:
        xrec = self.window(ex, store, own, pre, k, instance) if full else None
      if xrec is not None:
        self.log("windows", xrec)
        m["probes_e1"] += sum(p["why"] == "E1" for p in xrec["probes"])
        m["probes_e2"] += sum(p["why"] == "E2" for p in xrec["probes"])
        m["probes_fx"] += sum(p["why"] == "FX" for p in xrec["probes"])
        m["probes_d2"] += sum(p["why"] == "D2" for p in xrec["probes"])
        m["probes_rv"] += sum(p["why"] == "RV" for p in xrec["probes"])
        m["learned_effectful"] += bool(xrec.get("learned_effectful"))
        m["learned_group"] += bool(xrec.get("learned_group"))
        m["p_rev_dropped"] += xrec.get("p_rev_dropped", 0)
        if xrec.get("joint_restore"):
          m["joint_restore_ok" if xrec["joint_restore"]["ok"] else "joint_restore_fail"] += 1
        m["drift_stops"] += xrec.get("stop") == "drift"
        m["learned_edit_effect"] += bool(xrec.get("learned_edit_effect"))
        m["probes_total"] += len(xrec["probes"])
        m["windows_explored"] += bool(xrec["probes"])
        if xrec.get("value"):
          m["fx_" + xrec["value"]["level"]] += 1
          m["fx_rel_pos"] += xrec["value"]["rel"] > 0
        m["fx_entry_need"] += bool(xrec.get("entry_need"))
      th.join()
      if "d" not in box:                   # the background decision failed (server outage): decide again here
        self.log("decide_retry", {"k": k, "err": repr(box.get("err"))[:300]})
        box["d"] = self.decide(agent, goal, pre["pixels"])
      d = box["d"]
      t_decision = time.time()
      m["calls"] += 1
      requery = False
      spec_hit = False
      if ARM == "FULL27" and self._spec is not None:
        sp = self._spec
        tok_s, fam_s, _ = C.action_token(ui, d["action"].__dict__)
        if d["action"].action_type == "click" and fam_s == sp["fam"]:
          spec_hit = True                        # the model chose exactly the drafted action: already executed
          m["spec_hit"] += 1
        else:
          m["spec_miss"] += 1
          self.dev.back()
          chk = self.dev.capture(f"s{k}_spec_undo", settle=True)
          if not R.same_state(chk["ui"], ui):
            ok_u, _ = JG.restore(self, ex.graph, ui, k)
            if not ok_u and not full9.relaunch_restore(self, ui, k):
              m["spec_undo_fail"] += 1
      if spec_hit:
        xrec = dict(xrec or {}, probes=[])     # nothing to verify: the screen moved on by the model's own choice
      if xrec and xrec["probes"]:
        v = self.dev.capture(f"s{k}_verify", settle=True)
        if ARM in ("FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and not R.same_state(v["ui"], ui) and full9.decision_still_valid(ui, v["ui"], d["action"]):
          # the screen differs only in ways that do not affect THIS decision (same page, same target at the
          # same place): execute it without asking the model again
          m["verify_relaxed"] += 1
        elif ARM in ("FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and not R.same_state(v["ui"], ui) and full9.relocate_decision(ui, v["ui"], d["action"]):
          # same activity and the control the model chose is still on screen (moved): the decision stands, it is
          # executed at the control's current position
          m["verify_relocated"] += 1
        elif ARM == "FULL22" and not R.same_state(v["ui"], ui):
          # no re-inference: bring the screen back to the state the model decided on, then execute that decision;
          # if the state cannot be restored the decision is dropped (never executed on a screen it was not made for)
          ok_s, _ = JG.restore(self, ex.graph, ui, k)
          if not ok_s and full9.relaunch_restore(self, ui, k):
            ok_s = True
          v2 = self.dev.capture(f"s{k}_verify2", settle=True)
          if ok_s and (R.same_state(v2["ui"], ui) or full9.decision_still_valid(ui, v2["ui"], d["action"])
                       or full9.relocate_decision(ui, v2["ui"], d["action"])):
            m["restored_then_executed"] += 1
          else:
            m["decision_dropped"] += 1
            row = {"step": k, "kind": "dropped_decision", "action": d["action"].__dict__, "summary": d.get("summary", ""),
                   "pre": R.strip(pre)}
            steps.append(row)
            self.log("steps", row)
            k += 1
            continue
        elif not R.same_state(v["ui"], ui):
          requery = True
          m["requery"] += 1
          pre = self.dev.capture(f"s{k}_pre_requery", settle=True, full_res=True)
          ui = pre["ui"]
          if fx is not None:
            full7.EXTRA["text"] = fx.hint(ui)
          d = self.decide(agent, goal, pre["pixels"])
          m["calls"] += 1
      full7.EXTRA["text"] = ""
      a = d["action"]
      tok, fam, sel = C.action_token(ui, a.__dict__)
      if xrec and fam:
        hit = [p for p in xrec["probes"] if p["family"] == fam and p.get("dst_ui") is not None and p["rollback_ok"]]
        if hit:
          m["e1_hit" if hit[0]["why"] == "E1" else "e2_hit"] += 1
          reuse_next = hit[0]["dst_ui"]
      if ARM in ("FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
        self._note_model_action(ui, a, m)
      if ARM in ("FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and self._last_macro_step == k - 1 and a.action_type == json_action.NAVIGATE_BACK:
        self._macro_brake = True                 # the model undid the graph's navigation: misuse
        m["macro_brake_undo"] += 1
        self._intent.misuse(getattr(self, "_last_macro_key", None))
      if ARM in ("FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
        if self._last_reuse_step == k - 1 and a.action_type == json_action.NAVIGATE_BACK:
          self._reuse_brake = True               # the model undid the graph's step: stop reusing in this episode
          m["reuse_brake_undo"] += 1
        P_ = JG.page_key(ui)
        rec_ = fn.next_rec if fn is not None else 0
        first_ = P_ not in self._rec_pages.get(rec_, set())
        self._rec_pages.setdefault(rec_, set()).add(P_)
        n_ = U.element_at(ui, int(a.x), int(a.y)) if a.x is not None else None
        if (a.action_type in ("click", "long_press") and n_ is not None and fam and not n_.in_list
            and not is_commit_token(ui, fam)):
          self._pend = (P_, fam, rec_, first_)
      if spec_hit:
        summary = (R.gelab_agent._normalize_space(d["parsed"].get("summary"))  # pylint: disable=protected-access
                   or R.gelab_agent._normalize_space(d["parsed"].get("explain")) or "tap")  # pylint: disable=protected-access
        agent._actions.append({"summary": summary})  # pylint: disable=protected-access
        agent._summaries.append(summary)  # pylint: disable=protected-access
        exec_s = 0.0
        reuse_next = self._spec["dst"]["ui"]
      else:
        summary, exec_s = self.commit(agent, d, goal)
      own.append(S.own_entry(ui, tok, fam))
      self._last_model_commit = bool(fam) and is_commit_token(ui, fam)
      if a.action_type == json_action.OPEN_APP:
        self._exec_log[:] = []
      self._exec_log.append({"action": {kk: vv for kk, vv in a.__dict__.items() if vv is not None},
                             "commit": self._last_model_commit})
      if pr is not None:
        pr.record(ui, a.__dict__, tok, sel, self._last_model_commit)
      if fn is not None:
        fn.observe_model_action(ui, a.__dict__)
        if self._last_model_commit:
          fn.on_model_commit()
      row = {"step": k, "kind": "model", "pre": R.strip(pre), "action": a.__dict__, "tok": tok, "summary": summary,
             "model_latency_s": d["model_latency_s"], "requery": requery, "t_decision": t_decision,
             "spec_hit": spec_hit,
             "probes": [{k2: v2 for k2, v2 in p.items() if k2 != "dst_ui"} for p in xrec["probes"]] if xrec else []}
      if stuck_text:
        row["stuck_hint"] = stuck_text[:300]
      steps.append(row)
      self.log("steps", row)
      prev = {"pre": pre, "slot_id": tok, "sel": sel, "action": a.__dict__, "step": k, "fam": fam,
              "label": (U.label_of(ui, U.element_at(ui, int(a.x), int(a.y))) if a.x is not None and
                        U.element_at(ui, int(a.x), int(a.y)) is not None else "")}
      k += 1
      if a.action_type in (json_action.STATUS, json_action.ANSWER):
        done = True
        break
    final = self.dev.capture("final", settle=True, pause=1.0, full_res=True)
    if fx is not None:
      fx.finish()
    if prev is not None:
      self.ingest_agent(graph, instance, prev, final)
    self._own_seq = own
    res = {"steps": (k if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else len(steps)), "env_actions": len(steps), "agent_steps": len(steps),
           "done": done, "model_calls": m["calls"],
           "final_cid": final["cid"], "n_probes": ex.n_probes, "explore_disabled": ex.disabled_reason,
           "store_eps": len(store.data) if store else 0}
    m["jump_absent"] = getattr(self, "_jump_absent", 0)
    m["jump_mismatch"] = getattr(self, "_jump_mismatch", 0)
    res.update({f"m_{kk}": vv for kk, vv in m.items()})
    return res

  def plan_jump(self, store, graph, own, ui, used):
    if ARM in ("FULL3", "FULL4", "FULL5", "FULL7", "FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
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
      gate_n = 1 if ARM in ("FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") else S.GATE_N
      same_ok = same is not None and same[0] == top and same[1] >= gate_n and same[1] / same[2] >= S.GATE_D
      # cross-task: unanimous across >= 2 templates AND the agent has already
      # revealed its branch by an own (non open-app) action in the matched
      # context; at the task entry, templates diverge (0% cross-task top-1).
      agent_acted = any(not x["tok"].startswith(("OPEN_APP", "STATUS", "WAIT")) for x in own)
      cross_ok = (len(doms) >= 2 and all(d[0] == top for d in doms.values())
                  and info.get("order", 0) >= 1 and agent_acted)
      if ARM in ("FULL4", "FULL5", "FULL7", "FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
        cross_ok = False   # cross-task evidence is survivor-biased: only orders E1 probes
      # FULL5: confirming the model's OWN just-chosen commit needs no history
      # evidence beyond which dialog button follows (intent is the model's).
      confirm_ok = (ARM in ("FULL5", "FULL7", "FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and i == 0 and self._last_model_commit and is_dialog_button(ui, top))
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

  def _note_model_action(self, ui, a, m):
    """Exploration's contribution to the graph: did the model take an edge exploration observed first?"""
    if a.action_type not in ("click", "long_press") or a.x is None:
      return
    n = U.element_at(ui, int(a.x), int(a.y))
    if n is None:
      return
    fam = C.family_of_sel(U.selector(ui, n))
    if hasattr(self, "_rv_seen"):
      self._rv_seen.add(((ui.package, ui.activity, fam), ui.content_sig))   # the agent itself observed it now
    if (ui.layout_sig, fam) in self._probe_states:
      m["model_used_explored_edge"] += 1

  def window(self, ex, store, own, pre, step, instance):
    """E1 then E2; nothing else."""
    t0 = time.time()
    rec = {"step": step, "probes": [], "stop": ""}
    if ex.disabled_reason:
      return dict(rec, stop="disabled", window_s=0.0)
    ui = pre["ui"]
    if ARM in ("FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and full9.transient(ui):
      # temporary states (popup menu / dialog / keyboard / selection mode / open drawer) are not on the back
      # stack: any probe destroys them and Back cannot bring them back
      return dict(rec, stop="transient_state", window_s=0.0)
    if ARM == "FULL13" and not full9.replayable(self._exec_log):
      # restorability guarantee: probe only while a failed Back can be repaired by relaunch + replay of the path
      return dict(rec, stop="not_restorable", window_s=0.0)
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
      if not seen_here and not (((ARM in ("FULL12", "FULL13", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and full9.effectful(s.sel)) or (ARM == "FULL14" and full9.effectful_v2(s.sel)))):
        targets.append(("E1", s))
    if (not targets or ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30")) and ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
      # graph expansion: an unknown safe out-edge of this screen, ordered by similarity to the goal's operation
      # terms, then by cross-app generality (app-specific controls first), then screen order
      unknown = [s for p0, s in plan if p0 == 0 and (ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") or not s.sel.get("in_list"))
                 and Graph_app_key(ui, s) not in getattr(self, "_bad_fams", set())
                 and not (((ARM in ("FULL12", "FULL13", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and full9.effectful(s.sel)) or (ARM == "FULL14" and full9.effectful_v2(s.sel))))]
      if ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
        # graph-aware: a control whose transition from this page is already in the graph (seen by the agent or
        # by exploration in any task) carries no new information; nor does one learned to change app data
        known, reversible = full9.page_family_index(ex.graph)
        unknown = [s for s in unknown if (ui.package, ui.activity, C.family_of_sel(s.sel)) not in known
                   and (ui.package, C.family_of_sel(s.sel)) not in self._learned_eff
                   and full9.container_key(ui.package, ui.activity, s.sel) not in self._bad_groups]
        # predicted reversibility: probe only controls Back is expected to undo; when only risky controls are
        # left unexplored, do not explore at all (deduplication must not push probes onto them)
        evid = full9.container_evidence(ex.graph)
        prev = {s.slot_id: full9.p_reversible(ui, s.sel, evid) for s in unknown}
        rec["p_rev_dropped"] = sum(1 for v in prev.values() if v < P_REV_MIN)
        if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
          # risk-adjusted: a less certain control is acceptable only while relaunch + replay can repair it
          rp = full9.replayable(self._exec_log)
          unknown = [s for s in unknown if prev[s.slot_id] >= P_REV_MIN or
                     (ARM not in ("FULL20", "FULL30") and rp and prev[s.slot_id] >= P_REV_RISKY)]
        else:
          unknown = [s for s in unknown if prev[s.slot_id] >= P_REV_MIN]
        rec["known_index"] = len(known)
      if unknown and ARM in ("FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
        # unified graph-based exploration value: TF-IDF relevance of the best text evidence the graph holds for
        # each control (known edge > graph neighbours > the control's own text), global x graph-local IDF
        gv = EV.GraphValue(ex.graph, self.out.parents[1], self._gvec, ui.package)
        if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
          gv.pvec = self._pvec
        scored = sorted(((gv.score(ui, sl), i, sl) for i, sl in enumerate(unknown)), key=lambda x: (-x[0][0], x[1]))
        if ARM in ("FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
          # expected benefit minus expected re-inference cost; only controls with positive priority are probed
          bonus = 0.0
          if ARM == "FULL26":
            # bridging: pages many tasks pass through are explored more completely (amortized over later tasks)
            bonus = LAMBDA_T * math.log1p(TG.traffic(self._tmat, ui))
            rec["traffic_bonus"] = round(bonus, 3)
          scored = sorted(((((sc[0] + bonus - KAPPA * (1 - prev.get(sl.slot_id, 1.0))),) + tuple(sc[1:]), i, sl)
                           for sc, i, sl in scored), key=lambda x: (-x[0][0], x[1]))
          rec["fx_nonpositive"] = sum(1 for sc, _, _ in scored if sc[0] <= 0)
          scored = [x for x in scored if x[0][0] > 0] or []
        if not scored:
          scored = None
      if unknown and ARM in ("FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and scored is None:
        pass
      elif unknown and ARM in ("FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
        (sc0, val0, lvl0), _, best = scored[0]
        rec["value"] = {"score": round(sc0, 3), "rel": round(val0, 3), "level": lvl0, "local_pages": gv.local_n}
        if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
          # siblings of one container share their behaviour: one probe per container per window
          taken = {t[1].slot_id for t in targets}
          conts = {full9.container_key(ui.package, ui.activity, t[1].sel) for t in targets}
          fx_cap = 2 if ARM == "FULL20" else PROBES_PER_STEP   # FULL20: FX carries 31% Back failures; RV 3%
          for _, _, sl in scored:
            if len(targets) >= PROBES_PER_STEP or sum(t[0] == "FX" for t in targets) >= fx_cap:
              break
            ck = full9.container_key(ui.package, ui.activity, sl.sel)
            if sl.slot_id in taken or (ck in conts and ck[2] != C.family_of_sel(sl.sel)):
              continue
            conts.add(ck)
            targets.append(("FX", sl))
        else:
          targets.append(("FX", best))
      elif unknown:
        order = full9.frontier_order(ui, unknown, self._fn_ops, ELEMENT_APPS)
        if ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
          need = full9.entry_tokens(self._fn, ui)
          if need:
            # information need = the entry screen of a pending function that does not ground here
            order = sorted(order, key=lambda sl: -len(need & set(full9._toks(sl.sel.get("label") or sl.sel.get("desc") or sl.sel.get("rid") or ""))))
            rec["entry_need"] = sorted(need)[:8]
        if ARM == "FULL14" and self._fn is not None and self._fn.form_pending():
          # information need = the entry screen of the pending form function: editor-opening controls first
          order = sorted(order, key=lambda sl: not full9.opens_editor(sl.sel))
        targets.append(("FX", order[0]))
      elif ARM in ("FULL15", "FULL16"):
        d2 = self.depth2_target(ex, ui, known, reversible)
        if d2 is not None:
          targets.append(("D2", d2))
      if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
        # fixed exploration effort: fill up to PROBES_PER_STEP with deeper probes behind distinct known edges
        used = set()
        while len(targets) < PROBES_PER_STEP:
          d2 = self.depth2_target(ex, ui, known, reversible, exclude=used)
          if d2 is None:
            break
          used.add(C.family_of_sel(d2.sel))
          targets.append(("D2", d2))
        if len(targets) < PROBES_PER_STEP:
          # re-observation: an edge the graph knows from this page but not in this episode. Its destination holds
          # this task's content (list rows, field values) that function routing grounds on; only controls Back has
          # been seen to undo (or predicted to)
          bp = getattr(ex.graph, "_by_page", {})
          gvr = EV.GraphValue(ex.graph, self.out.parents[1], self._gvec, ui.package)
          gvr.pvec = self._pvec
          taken = {t[1].slot_id for t in targets}
          conts = {full9.container_key(ui.package, ui.activity, t[1].sel) for t in targets}
          rv = []
          for sl in U.enumerate_slots(ui):
            fam = C.family_of_sel(sl.sel)
            key = (ui.package, ui.activity, fam)
            if not sl.safe or sl.slot_id in taken or full9.effectful(sl.sel) or key not in known:
              continue
            obs = bp.get(key, [])
            # the information unit is (edge, page content): re-observe once per content state of this page
            if (key, ui.content_sig) in self._rv_seen:
              continue
            ck = full9.container_key(ui.package, ui.activity, sl.sel)
            if (ck in self._bad_groups or (ui.package, fam) in self._learned_eff or Graph_app_key(ui, sl) in getattr(self, "_bad_fams", set())
                or key in self._d2_bad):
              continue
            if key not in reversible and full9.p_reversible(ui, sl.sel, evid) < P_REV_MIN:
              continue
            ents = [(o["instance"], o["dst_cid"], o["dst"].get("layout_sig")) for o in obs if o.get("dst_cid")][-3:]
            sc_rv = gvr.score(ui, sl, ents)[0]
            if ARM in ("FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and sc_rv <= 0:
              continue                       # P0-3: re-observe only edges whose content is relevant to this goal
            rv.append((sc_rv, sl, ck))
          for _, sl, ck in sorted(rv, key=lambda x: -x[0]):
            if len(targets) >= PROBES_PER_STEP:
              break
            if ck in conts and ck[2] != C.family_of_sel(sl.sel):
              continue
            conts.add(ck)
            targets.append(("RV", sl))
        for t in targets:
          self._rv_seen.add(((ui.package, ui.activity, C.family_of_sel(t[1].sel)), ui.content_sig))
    elif not targets and info.get("order", -1) < 0 and ARM not in ("FULLNE", "FULL2", "FULL3", "FULL4", "FULL5", "FULL7", "FULL8", "FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):   # gap page (E2 off in FULLNE/FULL2/FULL3)
      for p, s in plan:
        if p == 0 and len(targets) < 3:      # unknown out-edges only
          targets.append(("E2", s))
    rec["mode"] = targets[0][0] if targets else "none"
    for i_t, (why, s) in enumerate(targets):
      if time.time() - t0 >= ex.cfg.budget_s:
        rec["stop"] = "budget"
        break
      if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and i_t > 0:
        # every probe must start from the origin state: verify before probing again
        chk0 = self.dev.capture(f"s{step}_x{i_t}_chk", settle=True)
        if not R.same_state(chk0["ui"], ui):
          rec["stop"] = "drift"
          ok_j, _ = JG.restore(self, ex.graph, ui, step)
          if not ok_j and not full9.relaunch_restore(self, ui, step):
            rec["drift_unrestored"] = True
          break
      if why == "D2":
        p = self.probe_depth2(ex, pre, s, step)
        if p is None:
          rec["stop"] = "d2_not_landed"
          break
      else:
        p = ex.probe(pre, s, 1 if why == "RV" else 0, step, depth=1)
      dst_ui = None
      try:
        dst_ui = R.load_ui(self.out, p["dst_cid"])
      except Exception:  # pylint: disable=broad-except
        pass
      fam_p = C.family_of_sel(s.sel)
      kp = (ui.layout_sig, fam_p)
      if kp not in self._probe_states and not any(o.get("origin") != "explore" for o in graph_obs(ex.graph, ui.struct_sig, s.slot_id)
                                                     + graph_obs(ex.graph, ui.struct_sig, fam_p)):
        self._probe_states[kp] = step
      if ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and why != "D2" and dst_ui is not None and full9.edits_changed(ui, dst_ui):
        # the control wrote into a text field of this page: it edits data even if Back/retap restored the
        # screen. Its whole container is learned effectful (graph-persisted) and the window stops here
        gk = full9.container_key(ui.package, ui.activity, s.sel)
        if gk not in self._bad_groups:
          self._bad_groups.add(gk)
          with self._grp_path.open("a") as f:
            f.write(json.dumps({"key": list(gk), "family": C.family_of_sel(s.sel), "instance": instance, "step": step,
                                "reason": "edits_text"}) + "\n")
        rec["learned_edit_effect"] = list(gk)
      rec["probes"].append({"why": why, "slot": s.slot_id, "family": C.family_of_sel(s.sel), "outcome": p["outcome"],
                            "rollback_ok": p["rollback_ok"], "t": p["t_total_s"], "dst_ui": dst_ui if why != "D2" else None})
      if not p["rollback_ok"] and ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and why != "D2":
        # Back could not undo it: the whole container (its siblings) is learned non-reversible, across tasks
        gk = full9.container_key(ui.package, ui.activity, s.sel)
        if gk not in self._bad_groups:
          self._bad_groups.add(gk)
          with self._grp_path.open("a") as f:
            f.write(json.dumps({"key": list(gk), "family": C.family_of_sel(s.sel), "instance": instance, "step": step}) + "\n")
          rec["learned_group"] = list(gk)
      if not p["rollback_ok"] and why == "D2":
        # the two-level detour through this edge could not be undone by Back: never go deeper through it again
        key = (ui.package, ui.activity, C.family_of_sel(s.sel))
        self._d2_bad.add(key)
        with self._d2_path.open("a") as f:
          f.write(json.dumps({"key": list(key), "instance": instance, "step": step}) + "\n")
      if not p["rollback_ok"] and ARM in ("FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
        # restoration ladder, second rung: a safe path over the joint graph back to the origin page
        ok_j, orgs = JG.restore(self, ex.graph, ui, step)
        rec["joint_restore"] = {"ok": ok_j, "hops_origins": orgs}
        if ok_j:
          p["rollback_ok"] = True
      if not p["rollback_ok"] and ARM in ("FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
        ok_r = full9.relaunch_restore(self, ui, step)
        rec["relaunch_restore"] = ok_r
        if ok_r:
          p["rollback_ok"] = True
          if ARM in ("FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and not getattr(self, "_relaunch_content_same", True):
            # the screen is back but its data is not: the probed control changes app data. Remember it in the
            # graph so no later task explores it again
            key = (ui.package, C.family_of_sel(s.sel) if why != "D2" else p.get("family", ""))
            self._learned_eff.add(key)
            with self._eff_path.open("a") as f:
              f.write(json.dumps({"key": list(key), "instance": instance, "step": step}) + "\n")
            rec["learned_effectful"] = list(key)
      if not p["rollback_ok"]:
        rec["stop"] = "rollback_failed"
        if ARM in ("FULL9", "FULL10", "FULL11", "FULL12", "FULL13", "FULL14", "FULL15", "FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
          # reversibility gating by control family, not by episode: never probe this family again; stop
          # exploring only after a second failure
          self._bad_fams = getattr(self, "_bad_fams", set()) | {Graph_app_key(ui, s)}
          self._rb_fail += 1
          if self._rb_fail >= 2:
            ex.disabled_reason = f"rollback_failed_x2@{step}"
        else:
          ex.disabled_reason = f"rollback_failed@{step}"
        break
    if ARM == "FULL27" and not rec.get("stop") and getattr(self, "_infer_thread", None) is not None \
        and self._infer_thread.is_alive() and not full9.transient(ui):
      dr = TG.draft(self._tmat, ui, self._gvec_unit)
      if dr is not None:
        # speculative execution: perform the drafted action while the model reasons; kept only if it agrees
        chk = self.dev.capture(f"s{step}_spec_pre", settle=True)
        if R.same_state(chk["ui"], ui):
          sl, fam, act, mg = dr
          self.dev.tap(*ui.nodes[sl.node_idx].center)
          dst = full7.capture_stable(self.dev, f"s{step}_spec")
          self._spec = {"fam": fam, "dst": dst, "pred_act": act, "margin": mg}
          rec["spec"] = {"fam": fam, "margin": mg, "landed": dst["ui"].activity == act}
    rec["window_s"] = round(time.time() - t0, 3)
    return rec


def _depth2_target(self, ex, ui, known, reversible, exclude=()):
  """No unknown control left on this page: go one level deeper along a known edge that exploration has seen
  reversed by Back, to a page that still has unknown (non-effectful) controls."""
  raw = self.out.parents[1]
  cands = []
  for sl in U.enumerate_slots(ui):
    if not sl.safe or sl.sel.get("in_list") or full9.effectful(sl.sel):
      continue
    fam = C.family_of_sel(sl.sel)
    if (ui.package, ui.activity, fam) not in reversible or (ui.package, ui.activity, fam) in self._d2_bad or fam in exclude:
      continue
    for o in reversed(getattr(ex.graph, "_by_page", {}).get((ui.package, ui.activity, fam), [])):   # newest first
      dd = o.get("dst")
      # the intermediate page must be on the back stack: an activity transition (popup menus and dialogs stay in
      # the same activity and are not)
      if not dd or dd.get("package") != ui.package or dd.get("activity") == ui.activity or not o.get("dst_cid"):
        continue
      task, var = o["instance"].split("/")
      try:
        dui = R.load_ui(raw / task / var.split("_")[0], o["dst_cid"])
      except Exception:  # pylint: disable=broad-except
        continue
      if full9.transient(dui):
        continue
      evid = full9.container_evidence(ex.graph)
      fresh = [t for t in U.enumerate_slots(dui) if t.safe and not t.sel.get("in_list") and not full9.effectful(t.sel)
               and full9.p_reversible(dui, t.sel, evid) >= P_REV_MIN
               and (dui.package, dui.activity, C.family_of_sel(t.sel)) not in known
               and (dui.package, C.family_of_sel(t.sel)) not in self._learned_eff
               and full9.container_key(dui.package, dui.activity, t.sel) not in self._bad_groups]
      if fresh and ARM in ("FULL16", "FULL17", "FULL18", "FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30"):
        cands.append((sl, dd, dui, fresh, (o["instance"], o["dst_cid"], dd.get("layout_sig"))))
        break
      if fresh:
        sl.d2 = {"via_dst": dd, "inner": full9.frontier_order(dui, fresh, self._fn_ops, ELEMENT_APPS)[0].sel}
        return sl
  if cands:
    # value backup decides where to go deeper: V(destination page) + novelty of the best control behind it
    gv = EV.GraphValue(ex.graph, raw, self._gvec, ui.package)
    best = None
    for sl, dd, dui, fresh, ent in cands:
      inner = max(fresh, key=lambda t: gv.score(dui, t)[0])
      v = gv.V(ent) + gv.score(dui, inner)[0] + EV.BETA_D2 * gv.novelty(inner.sel)
      if best is None or v > best[0]:
        best = (v, sl, dd, inner)
    _, sl, dd, inner = best
    sl.d2 = {"via_dst": dd, "inner": inner.sel}
    return sl
  return None


def _probe_depth2(self, ex, pre, sl, step):
  """Tap the known edge, probe the unknown control on the page behind it (the probe restores that page), then
  Back to the origin page. The origin check (and relaunch-replay repair) is done by the caller."""
  ui0 = pre["ui"]
  x, y = ui0.nodes[sl.node_idx].center
  self.dev.tap(x, y)
  mid = self.dev.capture(f"s{step}_d2_mid", settle=True, pause=0.8)
  inner_fam = C.family_of_sel(sl.d2["inner"])
  target = next((t for t in U.enumerate_slots(mid["ui"]) if C.family_of_sel(t.sel) == inner_fam and t.safe), None)
  if mid["ui"].activity != sl.d2["via_dst"]["activity"] or target is None or full9.transient(mid["ui"]):
    self.dev.back()
    chk = self.dev.capture(f"s{step}_d2_back", settle=True)
    return None if R.same_state(chk["ui"], ui0) else {"rollback_ok": False, "outcome": "d2_not_landed", "t_total_s": 0.0,
                                                       "dst_cid": mid["cid"], "family": inner_fam}
  p = ex.probe(mid, target, 0, step, depth=2)
  self.dev.back()
  chk = self.dev.capture(f"s{step}_d2_back", settle=True)
  ok = p["rollback_ok"] and R.same_state(chk["ui"], ui0)
  return dict(p, rollback_ok=ok, family=inner_fam)


FullEpisode.depth2_target = _depth2_target
FullEpisode.probe_depth2 = _probe_depth2


def _try_reuse(self, ui, pre, agent, steps, k, own, m, fn):
  """Graph reuse of the model's own decisions in this episode (no inference):
  repeat - the model chose the same navigation control on this page the last two times, landing alike;
  loop   - in the previous record the model's first action on this page was this control, and the task has moved
           on to the next record (multi-record procedures repeat their navigation).
  Only safe, non-list, non-commit controls found on the live screen; landing is verified (brake on mismatch)."""
  P = JG.page_key(ui)
  hist = self._page_acts.get(P, [])
  cand = why = land = None
  moved = lambda h: (h["land"][1], h["land"][2]) != (P[1], P[2])   # noqa: E731
  if (len(hist) >= 2 and hist[-1]["fam"] == hist[-2]["fam"] and hist[-1]["land"][1] == hist[-2]["land"][1]
      and (ARM not in ("FULL19", "FULL20", "FULL21", "FULL22", "FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") or (moved(hist[-1]) and moved(hist[-2])))):
    # FULL19: only a repeated transition that leaves this page (an in-page repeat - next month, a digit, a stuck
    # tap - needs the model's judgement of when to stop)
    cand, why, land = hist[-1]["fam"], "repeat", hist[-1]["land"]
  elif fn is not None:
    cur = fn.next_rec
    prior = [h for h in hist if h["rec"] < cur and h["first"]]
    if prior and P not in self._rec_pages.get(cur, set()):
      cand, why, land = prior[-1]["fam"], "loop", prior[-1]["land"]
  if cand is None:
    return 0
  sl = next((t for t in U.enumerate_slots(ui) if C.family_of_sel(t.sel) == cand and t.safe and not t.sel.get("in_list")
             and not full9.effectful(t.sel) and t.unsafe_reason not in ("unsafe_label", "dialog_action_button")), None)
  if sl is None:
    m["reuse_absent"] += 1
    return 0
  node = ui.nodes[sl.node_idx]
  action = json_action.JSONAction(action_type="click", x=node.center[0], y=node.center[1])
  agent._execute_action(action, {})  # pylint: disable=protected-access
  self._exec_log.append({"action": {kk: vv for kk, vv in action.__dict__.items() if vv is not None}, "commit": False})
  lab = U.label_of(ui, node) or sl.sel.get("desc") or sl.sel.get("rid")
  summ = f"[graph] Tapped \"{lab}\"."
  agent._actions.append({"summary": summ})  # pylint: disable=protected-access
  agent._summaries.append(summ)  # pylint: disable=protected-access
  own.append(S.own_entry(ui, cand, cand))
  row = {"step": k, "kind": "graph_reuse", "why": why, "tok": cand, "summary": summ, "pre": R.strip(pre)}
  steps.append(row)
  self.log("steps", row)
  cap = full7.capture_stable(self.dev, f"s{k}_reuse")
  rec_ = fn.next_rec if fn is not None else 0
  self._rec_pages.setdefault(rec_, set()).add(P)
  self._page_acts.setdefault(P, []).append({"fam": cand, "land": JG.page_key(cap["ui"]), "rec": rec_,
                                            "first": why == "loop"})
  m["reuse_skip"] += 1
  m["reuse_" + why] += 1
  if cap["ui"].activity != land[1]:
    self._reuse_brake = True
    m["reuse_brake_landing"] += 1
  self._last_reuse_step = k
  return 1


FullEpisode.try_reuse = _try_reuse


def is_commit_token(ui, fam):
  """The model's clicked slot is a commit (deny-listed label or dialog action)."""
  for sl in U.enumerate_slots(ui):
    if C.family_of_sel(sl.sel) == fam:
      return sl.unsafe_reason in ("unsafe_label", "dialog_action_button")
  return False


P_REV_MIN = 0.7
if ARM == "FULL30":
  # FULL29 calibration: probes with predicted P_rev < 0.8 failed rollback 33% of the time (>= 0.8: 4%) and every
  # unrepaired failure costs a re-inference; probe only controls Back is expected to undo
  P_REV_MIN = 0.8
P_REV_RISKY = 0.5          # FULL17: accepted only while relaunch + replay can repair the state
LAMBDA_T = 0.15            # FULL26: exploration bonus per log(1 + tasks through the page)
KAPPA = 0.3                # FULL23: expected re-inference cost of a probe Back may fail to undo, in value units
PROBES_PER_STEP = 5        # FULL17: fixed exploration effort per inference
ELEMENT_APPS = {}
try:
  ELEMENT_APPS = json.load(open(HERE.parent.parent / "runtime_graph_exploration" / "graph" / "element_idf_base.json"))["apps"]
except Exception:  # pylint: disable=broad-except
  pass


def Graph_app_key(ui, slot):
  return (ui.package, C.family_of_sel(slot.sel))


def is_dialog_button(ui, fam):
  for sl in U.enumerate_slots(ui):
    if C.family_of_sel(sl.sel) == fam:
      return sl.unsafe_reason == "dialog_action_button"
  return False


def graph_obs(graph, struct, slot_id):
  return graph.slots.get((struct, slot_id), [])


def _run(self):
  res = S._orig_run(self)  # pylint: disable=protected-access
  if ARM in ("FULL23", "FULL24", "FULL26", "FULL27", "FULL28", "FULL29", "FULL30") and getattr(self, "_intent", None) is not None:
    # P0-1: a successful episode validates the edges it traversed for this kind of goal
    self._intent.record_episode(self.out / "steps.jsonl", self._gvec, res.get("is_successful") == 1.0)
  if ARM.startswith("FULL") and res.get("is_successful") == 1.0 and getattr(self, "_own_seq", None):
    S.Store(APPDIR / ARM_NAME / "store.json").append(self._own_seq, self.args.task)
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
