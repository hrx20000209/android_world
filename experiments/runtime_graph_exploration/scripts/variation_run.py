"""Controlled runtime-variation demo: does exploration repair historical paths
that genuinely become invalid?

Historical function = a successful recorded trajectory of the SAME task
instance (same task + seed), replayed with selector relocation and parameter
substitution (rgs_run replay_struct semantics). Only the runtime is changed:

  V1 first-run reset : `pm clear` of the app before task initialization
  V2 calendar view   : Simple Calendar default view set to daily (prefs)
  V4 display density : `wm density 340` (control: identities stay valid)

Arms:  plain   replay; stop at the first divergence (no model, no exploration)
       repair  on divergence, model-free repair exploration (<= 2 hops) to a
               state where the historical step applies again, then continue

Repair (Q3): goal predicate = live page has the reference pre-state's activity
and the reference step's target grounds uniquely (or the layouts match).
Candidates = safe slots of the live page. Order: slots with an observed
transition (any task, structural evidence in O) into the target activity
first, then screen order (deterministic tie-break). A probe is kept if its
successor satisfies the predicate, otherwise rolled back by BACK and verified;
a failed rollback ends exploration. Depth 2 only through a kept depth-1 page
that the graph connects to the target activity. Every variation is undone in
`finally`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import memory as M  # noqa: E402

C, U, R = M.C, M.U, M.R
from android_world.env import json_action  # noqa: E402

ADB = R.ADB
PKG = {"Markor": "net.gsantner.markor", "Expense": "com.arduia.expense", "Recipe": "com.flauschcode.broccoli",
       "Tasks": "org.tasks", "SimpleCalendar": "com.simplemobiletools.calendar.pro", "Contacts": "com.google.android.contacts"}
OBS = [json.loads(l) for l in open(ROOT / "graph" / "observations.jsonl")]
ACT_EDGES = {}
for o in OBS:
  if o["fam"] and o["dst"]:
    ACT_EDGES.setdefault((o["src"]["act"], o["fam"]), set()).add(o["dst"]["act"])


def sh(*a):
  return subprocess.run([ADB, "-s", "emulator-5554", "shell", *a], capture_output=True, text=True, timeout=60).stdout


def pkg_of(task):
  return next(v for k, v in PKG.items() if task.startswith(k))


class VarEpisode(R.Episode):
  def setup(self):
    meta = super().setup()
    a = self.args
    if a.var == "V1":
      sh("pm", "clear", pkg_of(a.task))
    elif a.var == "V4":
      sh("wm", "density", "340")
    meta.update({"var": a.var, "arm": a.arm, "ref": a.ref})
    (self.out / "meta.json").write_text(R.jdump(meta))
    return meta

  def run(self):
    try:
      return super().run()
    finally:
      if self.args.var == "V2":
        p = "/data/data/com.simplemobiletools.calendar.pro/shared_prefs/Prefs.xml"
        sh("am", "force-stop", "com.simplemobiletools.calendar.pro")
        sh(f"su 0 sed -i 's/<int name=\"view\" value=\"[0-9]*\"/<int name=\"view\" value=\"1\"/' {p}")
      elif self.args.var == "V4":
        sh("wm", "density", "reset")

  # ------------------------------------------------------------------
  def applies(self, live, ref_ui, rs, subst):
    dec = self.replay_decide(rs, ref_ui, live, False, subst)
    if not dec["execute"] and dec.get("reason") == "target_ambiguous" and dec.get("coord_hit_same_target"):
      # identical repeated controls (e.g. calendar day cells with broken a11y
      # bounds): the recorded point still hits an element with the same
      # selector, so keep the recorded point.
      dec.update(execute=True, action=dict(rs["action"]), ambiguous_kept_point=True)
    return dec

  def run_replay(self, strict: bool):
    ref_dir = Path(self.args.ref)
    rows = [json.loads(l) for l in open(ref_dir / "steps.jsonl")]
    rows = [r for r in rows if r.get("kind") == "model"]
    subst = R.param_substitution(json.loads((ref_dir / "meta.json").read_text()),
                                 json.loads((self.out / "meta.json").read_text()))
    stats = {"ref_steps": len(rows), "executed": 0, "gaps": [], "repairs": [], "probes": 0, "rollback_failures": 0}
    agent = self.make_agent()
    if self.args.var == "V2":       # after task init (init resets app prefs); app restarts at OPEN_APP
      p = "/data/data/com.simplemobiletools.calendar.pro/shared_prefs/Prefs.xml"
      sh("am", "force-stop", "com.simplemobiletools.calendar.pro")
      sh(f"su 0 sed -i 's/<int name=\"view\" value=\"[0-9]*\"/<int name=\"view\" value=\"5\"/' {p}")
      stats["v2_pref"] = sh(f"su 0 grep -o 'name=.view. value=.[0-9]*.' {p}").strip()
    i = 0
    while i < len(rows):
      rs = rows[i]
      pre = self.dev.capture(f"r{i}_pre", settle=True, pause=1.0)
      ref_ui = R.load_ui(ref_dir, rs["pre"]["cid"])
      dec = self.applies(pre["ui"], ref_ui, rs, subst)
      if not dec["execute"]:
        gap = {"ref_step": i, "reason": dec.get("reason"), "cause": dec.get("cause"), "state_match": dec.get("state_match"),
               "live_act": pre["ui"].activity.rsplit(".", 1)[-1], "ref_act": ref_ui.activity.rsplit(".", 1)[-1]}
        stats["gaps"].append(gap)
        self.log("replay", dict(gap, kind="gap"))
        if self.args.arm != "repair":
          break
        rep = self.repair(pre, ref_ui, rs, subst, i, stats)
        stats["repairs"].append(rep)
        self.log("replay", dict(rep, kind="repair"))
        if not rep["ok"]:
          break
        continue                      # re-evaluate the same reference step
      act = json_action.JSONAction(**{k: v for k, v in dec["action"].items() if v is not None})
      if act.action_type == json_action.ANSWER and act.text:
        self.env.interaction_cache = act.text
      agent._execute_action(act, {})  # pylint: disable=protected-access
      stats["executed"] += 1
      self.log("replay", {"kind": "exec", "ref_step": i, "action": dec["action"], "relocated": dec.get("relocated")})
      i += 1
    stats["completed"] = i == len(rows)
    return stats

  # ------------------------------------------------------------------
  def satisfied(self, ui, ref_ui, rs, subst):
    if ui.layout_sig == ref_ui.layout_sig:
      return True
    if ui.activity != ref_ui.activity:
      return False
    return self.applies(ui, ref_ui, rs, subst)["execute"]

  def repair(self, pre, ref_ui, rs, subst, i, stats, depth=1, budget=None):
    budget = budget if budget is not None else [8]
    ui0 = pre["ui"]
    slots = [s for s in U.enumerate_slots(ui0) if s.safe]
    into = lambda s: ref_ui.activity in ACT_EDGES.get((ui0.activity, C.family_of_sel(s.sel)), set())
    slots.sort(key=lambda s: 0 if into(s) else 1)          # graph evidence first; stable = screen order
    tried = []
    for s in slots:
      if budget[0] <= 0:
        break
      budget[0] -= 1
      stats["probes"] += 1
      node = ui0.nodes[s.node_idx]
      self.dev.tap(*node.center)
      post = self.dev.capture(f"r{i}_rp{stats['probes']}", settle=True, pause=1.0)
      fam = C.family_of_sel(s.sel)
      tried.append({"fam": fam, "graph": into(s), "dst_act": post["ui"].activity.rsplit(".", 1)[-1]})
      if self.satisfied(post["ui"], ref_ui, rs, subst):
        return {"ok": True, "ref_step": i, "path": [fam], "depth": depth, "tried": tried, "by_graph": into(s)}
      if depth == 1 and post["ui"].activity != ui0.activity and \
         any(ref_ui.activity in d for (a, f), d in ACT_EDGES.items() if a == post["ui"].activity):
        sub = self.repair(post, ref_ui, rs, subst, i, stats, depth=2, budget=budget)
        if sub["ok"]:
          return dict(sub, path=[fam] + sub["path"], depth=2, tried=tried + sub["tried"])
      # roll back and verify
      self.dev.back()
      chk = self.dev.capture(f"r{i}_rb{stats['probes']}", settle=True, pause=1.0)
      if not (R.same_state(chk["ui"], ui0) or chk["ui"].layout_sig == ui0.layout_sig):
        stats["rollback_failures"] += 1
        return {"ok": False, "ref_step": i, "why": "rollback_failed", "tried": tried}
    return {"ok": False, "ref_step": i, "why": "no_path_within_budget", "tried": tried}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--seed", type=int, default=30)
  ap.add_argument("--ref", required=True)
  ap.add_argument("--var", choices=["V0", "V1", "V2", "V4"], required=True)
  ap.add_argument("--arm", choices=["plain", "repair"], required=True)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  args = argparse.Namespace(task=a.task, seed=a.seed, variant=f"{a.var}_{a.arm}", mode="replay_struct", out=a.out,
                            ref=a.ref, graph=None, explore=False, budget=6.0, max_probes=6, no_revalidate=False,
                            recover=False, no_graph_restore=False, app_memory=None, max_steps=0,
                            api_url="http://127.0.0.1:18095/v1/chat/completions", model="GELAB-ZERO-4B",
                            var=a.var, arm=a.arm)
  res = VarEpisode(args).run()
  print("RESULT", json.dumps({k: v for k, v in res.items() if k != "traceback"}, default=str)[:2500])
  if "traceback" in res:
    print(res["traceback"])


if __name__ == "__main__":
  main()
