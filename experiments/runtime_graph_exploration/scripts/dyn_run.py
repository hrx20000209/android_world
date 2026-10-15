"""Runtime-dynamics test: trajectory reuse vs exploration-grounded graph memory.

Arms (env DYN_ARM), all with the identical GELAB agent and prompt format:
  BASE  no memory
  TRAJ  EchoTrail-style: K=2 retrieved successful trajectories of similar
        tasks (recorded in the default runtime), injected every step
  GX    graph functions grounded in the CURRENT runtime:
          - functions = observed click paths from the current screen to a
            screen exposing a goal-named affordance (functions.Library), using
            historical edges plus edges observed in this episode
          - a function is shown only if its first step is present on the live
            screen (entry check) and its first transition was not observed to
            diverge in this runtime (transition check)
          - inference-window exploration (model-free, 1 probe / window, verified
            rollback): first, the first step of the best shown function if its
            transition is not yet observed in this runtime; otherwise an unknown
            safe frontier of this screen (opportunistic expansion). Every probe
            and every agent transition is added to the runtime graph.
Variations (env DYN_VAR): V0 default, V2 calendar default view = daily, V1 app
data cleared (first-run onboarding). Every variation is undone and verified.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import memory as M  # noqa: E402
import functions as F  # noqa: E402
import trajmem as T  # noqa: E402

C, U, R = M.C, M.U, M.R
from android_world.agents import gelab_agent  # noqa: E402
from android_world.env import json_action  # noqa: E402

ARM = os.environ.get("DYN_ARM", "BASE")
VAR = os.environ.get("DYN_VAR", "V0")
ADB = R.ADB
CAL = "com.simplemobiletools.calendar.pro"
PREFS = f"/data/data/{CAL}/shared_prefs/Prefs.xml"
EXTRA = {"text": ""}
_orig_build = gelab_agent.build_gelab_messages


def _build(goal, hist, img):
  msgs = _orig_build(goal, hist, img)
  if EXTRA["text"]:
    msgs[1]["content"][0]["text"] += "\n\n" + EXTRA["text"]
  return msgs


gelab_agent.build_gelab_messages = _build


def sh(cmd):
  return subprocess.run([ADB, "-s", "emulator-5554", "shell", cmd], capture_output=True, text=True, timeout=60).stdout


def set_cal_view(v):
  sh(f"am force-stop {CAL}")
  sh(f"su 0 sed -i 's/<int name=\"view\" value=\"[0-9]*\"/<int name=\"view\" value=\"{v}\"/' {PREFS}")
  return sh(f"su 0 grep -o 'name=.view. value=.[0-9]*.' {PREFS}").strip()


MARKOR = "net.gsantner.markor"
MKPREF = f"/data/data/{MARKOR}/shared_prefs/app.xml"


def set_markor_pref(kind, value):
  """V4: notes open in view mode first; V5: app starts on another tab. Returns the verified line."""
  sh(f"am force-stop {MARKOR}")
  if kind == "preview":
    sh(f"su 0 sed -i 's/<boolean name=\"pref_key__is_preview_first\" value=\"[a-z]*\"/<boolean name=\"pref_key__is_preview_first\" value=\"{value}\"/' {MKPREF}")
    return sh(f"su 0 grep -o 'pref_key__is_preview_first.*' {MKPREF}").strip()
  if not sh(f"su 0 grep -c pref_key__app_start_tab_v2 {MKPREF}").strip().startswith(("1", "2")):
    # the key disappears after `pm clear` (Markor rewrites app.xml with defaults): insert it explicitly
    sh(f"su 0 sed -i 's#</map>#    <string name=\"pref_key__app_start_tab_v2\">0</string>\\n</map>#' {MKPREF}")
  sh(f"su 0 sed -i 's#<string name=\"pref_key__app_start_tab_v2\">[0-9]*</string>#<string name=\"pref_key__app_start_tab_v2\">{value}</string>#' {MKPREF}")
  return sh(f"su 0 grep -o 'pref_key__app_start_tab_v2.*' {MKPREF}").strip()
AUTH = None


def restore_markor_intro():
  """Undo V1: walk through Markor's first-run intro (4x next + done) and verify."""
  import re as _re
  global AUTH
  if AUTH is None:
    src = (Path(gelab_agent.__file__).resolve().parents[1] / "env" / "android_world_controller.py").read_text()
    AUTH = _re.search(r"FAST_A11Y_PROVIDER_AUTHORITY = '([^']*)'", src).group(1)
  sh(f"monkey -p {MARKOR} -c android.intent.category.LAUNCHER 1")
  time.sleep(4)
  for _ in range(10):
    out = sh(f"content read --uri content://{AUTH}/flat?compact=1")
    try:
      nodes = json.loads(out)["nodes"]
    except Exception:  # pylint: disable=broad-except
      break
    hit = next((n for n in nodes if n.get("package") == MARKOR and
                (n.get("resourceId") or "").rsplit("/", 1)[-1] in ("next", "done", "skip")), None)
    if hit is None:
      break
    b = hit["bounds"]
    sh(f"input tap {(b[0] + b[2]) // 2} {(b[1] + b[3]) // 2}")
    time.sleep(1.5)
  sh(f"am force-stop {MARKOR}")
  sh(f"monkey -p {MARKOR} -c android.intent.category.LAUNCHER 1")
  time.sleep(3)
  top = [l for l in sh("dumpsys activity activities").splitlines() if "topResumedActivity" in l][:1]
  sh("input keyevent KEYCODE_HOME")
  return top[0].strip() if top else ""


class DynEpisode(R.Episode):
  def setup(self):
    meta = super().setup()
    if VAR == "V1":
      sh(f"pm clear {MARKOR}")
    return meta

  def run_agent(self, start_step=0, prior_summaries=None, label="agent"):
    goal, agent = self.task.goal, self.make_agent()
    m = {"calls": 0, "injected": 0, "fn_shown": 0, "fn_dropped_entry": 0, "fn_dropped_transition": 0,
         "probes_fn": 0, "probes_frontier": 0, "rollback_fail": 0, "requery": 0, "fn_from_runtime": 0}
    if VAR == "V2":
      m["v2_pref"] = set_cal_view(5)
    if VAR == "V4":
      m["var_pref"] = set_markor_pref("preview", "true")
    if VAR == "V5":
      m["var_pref"] = set_markor_pref("tab", "3")
    snap = M.Snapshot(self.args.task) if ARM in ("GX", "GX2", "GX3", "GX3b", "GX4", "GX5", "GX6") else None
    known_states = {o["src"]["key"] for o in snap.obs} | {o["dst"]["key"] for o in snap.obs if o["dst"]} if snap else set()
    effect = {}               # (state key, fam) -> {"label", "changed", "n", "title", "known"} observed in THIS run
    # GX5: exploration outcomes persist in the graph across episodes (effect + reversibility), so a control
    # probed once is never probed again; its observed effect is reused as evidence in later episodes.
    emem_path = Path(os.environ.get("DYN_EFFECT_MEM", "/dev/null"))
    emem = {}
    if ARM in ("GX5", "GX6") and emem_path.exists() and str(emem_path) != "/dev/null":
      for l in open(emem_path):
        r0 = json.loads(l)
        emem[(r0["key"], r0["fam"])] = r0
    lib = F.Library(snap) if snap else None
    diverged = set()          # (src key, fam) whose runtime successor differed from history
    seen_rt = set()           # (src key, fam) observed in this runtime
    tm = T.TrajMemory() if ARM == "TRAJ" else None
    traj_text = None
    ex = R.Explorer(self.dev, R.Graph(None), R.ExploreCfg(enabled=True, budget_s=6, max_probes=1, max_depth=1),
                    self.args.variant, self.log)
    done, k, prev = False, 0, None
    ignored = {}              # GX6: state key -> consecutive steps with an unheeded recommendation
    while k < self.max_steps():
      pre = self.dev.capture(f"s{k}_pre", settle=True, pause=1.0, full_res=True)
      ui = pre["ui"]
      key = M.state_key(ui)
      if prev is not None and lib is not None and prev[1]:
        self._observe(lib, prev[0], prev[1], prev[2], ui, diverged, seen_rt)
      if prev is not None and ARM in ("GX2", "GX3", "GX3b", "GX4", "GX5", "GX6") and prev[3]:
        self._effect(effect, prev[0], prev[3], prev[2], prev[4], ui, known_states)
      EXTRA["text"], shown = "", []
      if ARM == "TRAJ" and "launcher" not in ui.package:
        if traj_text is None:
          traj_text = tm.render(tm.retrieve(goal, ui.package))
        EXTRA["text"] = traj_text
      elif ARM in ("GX", "GX2", "GX3", "GX3b", "GX4", "GX5", "GX6") and "launcher" not in ui.package:
        live = {C.family_of_sel(s.sel) for s in U.enumerate_slots(ui)}
        visible = set(M.page_labels(ui))
        tree = lib.paths_from({key})
        cands = []
        gn = M.norm(goal)
        target_visible = any(F.RT.goal_named(t, gn) and t in lib.afford for t in visible)
        for f in ([] if target_visible else self._funcs(lib, goal, key, tree)):
          first_fam = lib.edges[key][f["first_dst"]][1]
          if first_fam not in live:
            m["fn_dropped_entry"] += 1
            continue
          if (key, first_fam) in diverged:
            m["fn_dropped_transition"] += 1
            continue
          f["first_fam"] = first_fam
          cands.append(f)
        shown = cands[:F.K]
        if shown:
          EXTRA["text"] = F.render(shown, header="Navigation memory checked against the CURRENT app state "
                                   "(each path's first step is on this screen; paths were observed, may still be incomplete):")
          m["fn_shown"] += 1
          m["fn_from_runtime"] += any(f.get("runtime") for f in shown)
      # GX3b: a loop is a REPEATED no-op (same control, same screen, >= 2 times); one no-op is not a loop
      # GX4: the graph decides how much evidence a loop needs. On a screen the history has never seen
      # (runtime-novel state), history cannot help, so one observed no-op triggers exploration evidence;
      # on a known screen, require a repeated no-op (GX3b) to avoid false triggers.
      need = 2 if ARM == "GX3b" or (ARM in ("GX4", "GX5", "GX6") and key in known_states) else 1
      if ARM in ("GX5", "GX6"):        # import remembered probe effects for this screen (not the model's own no-ops)
        for (k2, f2), r0 in emem.items():
          if k2 == key and r0["changed"] and (k2, f2) not in effect:
            effect[(k2, f2)] = {"label": r0["label"], "changed": True, "n": 0, "title": r0["title"],
                                "known": r0["known"], "remembered": True}
            m["effect_reused"] = m.get("effect_reused", 0) + 1
      rec_target = None
      stuck = ARM in ("GX2", "GX3", "GX3b", "GX4", "GX5", "GX6") and any(k2 == key and not e["changed"] and e["n"] >= need
                                                   for (k2, _), e in effect.items())
      if stuck:
        ev_lines = []
        for (k2, f2), e in effect.items():
          if k2 != key:
            continue
          if not e["changed"]:
            ev_lines.append(f"- Tapping \"{e['label']}\" did not change the screen ({e['n']} time{'s' if e['n'] > 1 else ''}).")
          else:
            ev_lines.append(f"- Tapping \"{e['label']}\" changed the screen" + (f" to \"{e['title']}\"" if e["title"] else "")
                            + (" (a screen known from earlier use of this app)." if e["known"] else "."))
        head = "Observed on THIS screen in the current run:\n" + "\n".join(ev_lines)
        if ARM in ("GX3", "GX3b", "GX4", "GX5", "GX6"):
          # directive form: the model's own repeated action is verified to have no effect here, while the
          # recommended control is verified to change the screen (runtime evidence, not history)
          live = {C.family_of_sel(sl.sel): sl for sl in U.enumerate_slots(ui)}
          eff = [(f2, e) for (k2, f2), e in effect.items() if k2 == key and e["changed"] and f2 in live]
          eff.sort(key=lambda x: (not x[1]["known"], x[0]))
          if eff:
            f2, e = eff[0]
            rec_target = (f2, e["label"])
            cx, cy = ui.nodes[live[f2].node_idx].center
            W, H = ui.screen
            head = (f"Recommended next action: tap \"{e['label']}\" at point ({round(cx * 1000 / W)}, {round(cy * 1000 / H)}). "
                    f"It was observed to change this screen, while the elements below were observed to have no effect.\n" + head)
        EXTRA["text"] = head + ("\n\n" + EXTRA["text"] if EXTRA["text"] else "")
        m["stuck_evidence"] = m.get("stuck_evidence", 0) + 1
      if EXTRA["text"]:
        m["injected"] += 1
      # GX6: escalation. On a runtime-novel screen (absent from the graph) where the model keeps repeating a
      # verified no-op and has already ignored the verified-effective recommendation twice, the executor taps
      # that control itself (recorded in the model-visible history as a system action).
      if ARM == "GX6" and rec_target is not None and key not in known_states:
        ignored[key] = ignored.get(key, 0) + 1
        if ignored[key] > 2:
          f2, lab2 = rec_target
          sl2 = next((sl for sl in U.enumerate_slots(ui) if C.family_of_sel(sl.sel) == f2 and sl.safe), None)
          if sl2 is not None:
            cx, cy = ui.nodes[sl2.node_idx].center
            self.dev.tap(cx, cy)
            summ = f"[system] Tapped \"{lab2}\" (observed to change this screen; the previous taps had no effect)."
            agent._actions.append({"summary": summ})  # pylint: disable=protected-access
            agent._summaries.append(summ)  # pylint: disable=protected-access
            self.log("steps", {"step": k, "tok": f2, "summary": summ, "kind": "escalation", "pre_cid": pre["cid"]})
            m["escalations"] = m.get("escalations", 0) + 1
            ignored[key] = 0
            prev = None
            k += 1
            continue
      box = {}
      th = threading.Thread(target=lambda: box.update(d=self.decide(agent, goal, pre["pixels"])), daemon=True)
      th.start()
      probe = None
      if ARM in ("GX", "GX2", "GX3", "GX3b", "GX4", "GX5", "GX6") and not ex.disabled_reason and "launcher" not in ui.package:
        target, why = None, None
        if stuck:      # find a control that does change this screen (stuck-loop / novel-state query)
          for sl in U.enumerate_slots(ui):
            fam = C.family_of_sel(sl.sel)
            if sl.safe and (key, fam) not in effect and not (ARM in ("GX5", "GX6") and (key, fam) in emem):
              target, why = fam, "stuck"
              break
        for f in ([] if target else shown):
          if (key, f["first_fam"]) not in seen_rt:
            target, why = f["first_fam"], "fn"
            break
        if target is None:
          for sl in U.enumerate_slots(ui):
            fam = C.family_of_sel(sl.sel)
            if sl.safe and not sl.sel.get("in_list") and (key, fam) not in seen_rt and not snap.by_edge.get((key, fam)) \
               and not (ARM in ("GX5", "GX6") and (key, fam) in emem):
              target, why = fam, "frontier"
              break
        slot = next((s for s in U.enumerate_slots(ui) if C.family_of_sel(s.sel) == target and s.safe), None) if target else None
        if slot is not None:
          p = ex.probe(pre, slot, 0, k, depth=1)
          m["probes_" + why] = m.get("probes_" + why, 0) + 1
          try:
            self._observe(lib, key, target, U.label_of(ui, ui.nodes[slot.node_idx]) or slot.sel.get("desc") or "",
                          R.load_ui(self.out, p["dst_cid"]), diverged, seen_rt)
          except Exception:  # pylint: disable=broad-except
            pass
          probe = {"fam": target, "why": why, "rollback_ok": p["rollback_ok"]}
          if ARM in ("GX2", "GX3", "GX3b", "GX4", "GX5", "GX6"):
            try:
              self._effect(effect, key, target, U.label_of(ui, ui.nodes[slot.node_idx]) or slot.sel.get("desc") or "",
                           ui.exact_sig, R.load_ui(self.out, p["dst_cid"]), known_states)
              if ARM in ("GX5", "GX6"):
                e0 = effect[(key, target)]
                rec0 = {"key": key, "fam": target, "label": e0["label"], "changed": e0["changed"], "title": e0["title"],
                        "known": e0["known"], "rollback_ok": p["rollback_ok"], "episode": self.args.variant}
                emem[(key, target)] = rec0
                with emem_path.open("a") as fe:
                  fe.write(json.dumps(rec0) + "\n")
            except Exception:  # pylint: disable=broad-except
              pass
          if not p["rollback_ok"]:
            ex.disabled_reason = f"rollback_failed@{k}"
            m["rollback_fail"] += 1
      th.join()
      d = box["d"]
      m["calls"] += 1
      if probe is not None:
        v = self.dev.capture(f"s{k}_verify", settle=True)
        if not R.same_state(v["ui"], ui):
          m["requery"] += 1
          pre = self.dev.capture(f"s{k}_pre_rq", settle=True, pause=1.0, full_res=True)
          ui = pre["ui"]
          d = self.decide(agent, goal, pre["pixels"])
          m["calls"] += 1
      a = d["action"]
      tok, fam, sel = C.action_token(ui, a.__dict__)
      lab = ""
      if a.x is not None and a.action_type == "click":
        n = U.element_at(ui, int(a.x), int(a.y))
        lab = U.label_of(ui, n) if n is not None else ""
      summary, _ = self.commit(agent, d, goal)
      self.log("steps", {"step": k, "tok": tok, "summary": summary, "injected": EXTRA["text"][:600], "probe": probe,
                         "pre_cid": pre["cid"]})
      safe_click = fam is not None and any(C.family_of_sel(sl.sel) == fam and sl.safe for sl in U.enumerate_slots(ui))
      prev = (M.state_key(ui), fam if safe_click else None, lab, fam, ui.exact_sig)   # only safe navigation enters the runtime graph
      k += 1
      if a.action_type in (json_action.STATUS, json_action.ANSWER):
        done = True
        break
    EXTRA["text"] = ""
    return {"steps": k, "done": done, **{f"m_{kk}": v for kk, v in m.items()}}

  def _funcs(self, lib, goal, key, tree):
    gn = M.norm(goal)
    found = []
    for st, path in tree.items():
      if not path:
        continue
      for lab in lib.labels.get(st, ()):
        if lab in lib.afford and F.RT.goal_named(lab, gn):
          found.append((len(path), lab, st, path))
    found.sort(key=lambda x: (x[0], x[1]))
    out, seen = [], set()
    for n, lab, st, path in found:
      if lab in seen:
        continue
      seen.add(lab)
      first_dst = next(v for v, (l2, f2) in lib.edges[key].items() if l2 == path[0][0] and (v == st or v in tree))
      out.append({"target": lab, "state": st, "path": path, "first_dst": first_dst,
                  "runtime": any(isinstance(x, tuple) and len(x) > 2 for x in path)})
    return out

  def _effect(self, effect, src, fam, label, src_exact, ui_dst, known_states):
    changed = not (M.state_key(ui_dst) == src and ui_dst.exact_sig == src_exact)
    e = effect.setdefault((src, fam), {"label": label or fam.split("|")[-1], "changed": changed, "n": 0, "title": "",
                                       "known": False})
    e["n"] += 1
    e["changed"] = e["changed"] or changed
    if changed:
      labs = M.page_labels(ui_dst)
      e["title"] = labs[0][:40] if labs else ""
      e["known"] = M.state_key(ui_dst) in known_states

  def _observe(self, lib, src, fam, label, ui_dst, diverged, seen_rt):
    dst = M.state_key(ui_dst)
    seen_rt.add((src, fam))
    hist = {d for d, (l2, f2) in lib.edges.get(src, {}).items() if f2 == fam}
    if hist and dst not in hist and all(d.split("#")[0] != ui_dst.activity for d in hist):
      diverged.add((src, fam))
    lib.add_runtime_edge(src, dst, label, fam, M.page_labels(ui_dst))

  def run(self):
    try:
      return super().run()
    finally:
      if VAR == "V2":
        set_cal_view(1)
      if VAR == "V4":
        (self.out / "restore.txt").write_text(set_markor_pref("preview", "false"))
      if VAR == "V5":
        (self.out / "restore.txt").write_text(set_markor_pref("tab", "0"))
      if VAR == "V1":
        top = restore_markor_intro()
        (self.out / "restore.txt").write_text(top)
        if "MainActivity" not in top:
          raise SystemExit(f"Markor intro NOT restored: {top}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True)
  ap.add_argument("--seed", type=int, default=32)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  args = argparse.Namespace(task=a.task, seed=a.seed, variant=f"dyn_{VAR}_{ARM}", mode="agent", out=a.out, ref=None,
                            graph=None, explore=False, budget=6.0, max_probes=1, no_revalidate=False, recover=False,
                            no_graph_restore=False, app_memory=None, max_steps=0,
                            api_url="http://127.0.0.1:18095/v1/chat/completions", model="GELAB-ZERO-4B")
  res = DynEpisode(args).run()
  print("RESULT", json.dumps({k: v for k, v in res.items() if k != "traceback"}, default=str)[:1500])
  if "traceback" in res:
    print(res["traceback"])


if __name__ == "__main__":
  main()
