"""Phase II online runner: trace-guided (TG) vs round-robin (RR) depth-2 exploration.

Reuses the previous study's runner (rgs_run.Episode / Device / Graph) and only
replaces the Explorer. Nothing is injected into the agent prompt (Pilot B was
negative); exploration evidence is logged for post-hoc use measurements:
  * did the window probe the action the agent then chose (first hop)?
  * did the probe's observed destination equal the real post-action state
    (exact / struct)?  -> speculative-observation reuse potential
  * depth-2: did (a, b) probed match the agent's next two actions?

Policies (lexicographic, no weighted scores; rgs_ui deny-list applies):
  RR : graph-unknown first, then screen order (the previous Explorer)
  TG : trace counts (longest matched suffix -> shorter -> state frequency),
       supported & unknown > supported & multi-outcome > supported & known >
       unsupported & unknown > rest  (Pilot A's P4)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import pa_trace as PA  # noqa: E402
import rgs_run as R  # noqa: E402

U = C.U
ARGS_POLICY = os.environ.get("TR_POLICY", "TG")
NEW_RAW = Path(os.environ.get("TR_RAW", str(C.ROOT / "raw" / "phase2")))


def all_episode_dirs():
  dirs = [Path(e["ep"]) for e in C.episodes()]
  for d in sorted(NEW_RAW.glob("*/*")):
    if (d / "result.json").exists() and (d / "steps.jsonl").exists():
      m = json.loads((d / "meta.json").read_text())
      if m.get("tr_policy") in (ARGS_POLICY, "none"):
        dirs.append(d)
  return dirs


def build_history_index():
  idx = PA.Index()
  n = 0
  if ARGS_POLICY.endswith("C"):   # cold start: no history at all, own episode only
    return idx, 0
  for d in all_episode_dirs():
    m = json.loads((d / "meta.json").read_text())
    e = {"ep": str(d), "task": m["task"]}
    try:
      idx.add_seq(PA.build_episode_seq(e))
      n += 1
    except Exception:  # pylint: disable=broad-except
      continue
  return idx, n


class TraceExplorer(R.Explorer):
  INSTANCE = None

  def __init__(self, *a, **k):
    super().__init__(*a, **k)
    TraceExplorer.INSTANCE = self
    self.policy = ARGS_POLICY
    self.index, self.n_hist = build_history_index()
    self.own = []           # [{"state","struct","tok","fam"}] executed agent steps
    self.last_pre = None
    self.window_log = []

  # ------------------------------------------------------------ ranking
  def _counts(self, seq_ext):
    return PA.overlay(self.index, seq_ext, len(seq_ext) - 1)

  def rank(self, ui: U.UIState, prefix: list) -> tuple[list[tuple[str, U.Slot]], dict]:
    slots = [s for s in U.enumerate_slots(ui) if s.safe]
    fams, first = [], {}
    for s in slots:
      f = C.family_of_sel(s.sel)
      if f not in first:
        first[f] = s
        fams.append(f)
    known, outcomes = set(), {}
    for (src, sid), lst in self.graph.slots.items():
      if src != ui.struct_sig:
        continue
      for o in lst:
        f = C.family_of_sel(o.get("selector") or {})
        known.add(f)
        outcomes.setdefault(f, set()).add((o.get("dst") or {}).get("layout_sig"))
    multi = {f for f, s in outcomes.items() if len(s) > 1}
    # skip slots this run already probed or known unrestorable (same rules as before)
    plan_ok = {s.slot_id for _, s in self.plan(ui)[0]}
    fams = [f for f in fams if first[f].slot_id in plan_ok]
    seq = prefix + [{"state": ui.struct_sig, "tok": "", "fam": None}]
    counts = self._counts(seq)
    ranks, longest, _ = PA.rank_policies(fams, counts, known, multi)
    order = ranks["P4"] if self.policy.startswith("TG") else ranks["P0u"]
    info = {"policy": self.policy, "longest_order": longest, "n_fams": len(fams),
            "rank_P4": ranks["P4"][:6], "rank_P0u": ranks["P0u"][:6], "rank_P1": ranks["P1"][:6],
            "supported": [f for f in fams if any(c and c[f] for c in counts)][:6]}
    return [(f, first[f]) for f in order], info

  # ------------------------------------------------------------ window
  def run_window(self, pre, step, deadline_s):
    t0 = time.time()
    rec = {"step": step, "probes": [], "stop": "", "depth2": []}
    self.last_pre = pre
    if self.disabled_reason:
      rec.update(stop="disabled:" + self.disabled_reason, window_s=0.0)
      return rec
    ui0 = pre["ui"]
    order, info = self.rank(ui0, self.own)
    rec["rank_info"] = info
    rec["plan_stats"] = self.plan(ui0)[1]
    for fam, slot in order:
      if time.time() - t0 >= deadline_s:
        rec["stop"] = "budget"
        break
      if len(rec["probes"]) >= self.cfg.max_probes:
        rec["stop"] = "max_probes"
        break
      p = self.probe_d2(pre, slot, fam, step, t0, deadline_s, rec)
      rec["probes"].append(p)
      if not p["rollback_ok"]:
        rec["stop"] = "rollback_failed"
        self.disabled_reason = "rollback_failed@step%d" % step
        break
    else:
      rec["stop"] = rec["stop"] or "plan_exhausted"
    rec["window_s"] = round(time.time() - t0, 3)
    return rec

  def probe_d2(self, pre, slot, fam, step, t0, deadline_s, rec):
    """First hop with an optional second hop before restoring the first.

    Implemented by temporarily replacing the rollback of the first probe: the
    base probe() taps, observes and restores; for depth 2 we call it in two
    phases via _hop/_restore helpers built on the base implementation."""
    ui0 = pre["ui"]
    node = ui0.nodes[slot.node_idx]
    x, y = node.center
    th0 = time.time()
    self.n_probes += 1
    self.dev.tap(x, y)
    post = self.dev.capture(f"s{step}_p{self.n_probes}_h1")
    ui1 = post["ui"]
    outcome = R.probe_outcome(ui0, ui1)
    d2 = None
    if (outcome == "new_screen" and self.cfg.max_depth >= 2 and time.time() - t0 < deadline_s
        and ui1.package == ui0.package):
      prefix = self.own + [{"state": ui0.struct_sig, "tok": fam, "fam": fam}]
      order2, info2 = self.rank(ui1, prefix)
      if order2:
        fam2, slot2 = order2[0]
        d2 = self.second_hop(post, slot2, fam2, step)
        d2["rank_info"] = info2
        d2["first_family"] = fam
        rec["depth2"].append(d2)
    rb = self.restore(ui0, ui1, node, step, ok_after_d2=(d2 is None or d2["rollback_ok"]))
    t_end = time.time()
    obs = self.make_obs(pre, post, slot, fam, step, 1, outcome, rb, th0, t_end)
    obs["child_affordances"] = [U.label_of(ui1, ui1.nodes[s.node_idx]) or ui1.nodes[s.node_idx].short_rid
                                for s in U.enumerate_slots(ui1)][:20]
    if d2 is not None:
      obs["second_hop"] = {k: d2[k] for k in ("family", "outcome", "dst_cid", "rollback_ok", "t_total_s")}
    self.graph.add(obs)
    self.log("probes", obs)
    return obs

  def second_hop(self, post1, slot2, fam2, step):
    ui1 = post1["ui"]
    node = ui1.nodes[slot2.node_idx]
    t0 = time.time()
    self.dev.tap(*node.center)
    post2 = self.dev.capture(f"s{step}_p{self.n_probes}_h2")
    out2 = R.probe_outcome(ui1, post2["ui"])
    rb = self.restore(ui1, post2["ui"], node, step, ok_after_d2=True)
    obs = self.make_obs(post1, post2, slot2, fam2, step, 2, out2, rb, t0, time.time())
    self.graph.add(obs)
    self.log("probes", obs)
    return {"family": fam2, "outcome": out2, "dst_cid": post2["cid"], "src_cid": post1["cid"],
            "rollback_ok": rb["ok"], "t_total_s": obs["t_total_s"], "dst": post2["ui"].summary()}

  def restore(self, ui_target, ui_post, node, step, ok_after_d2=True):
    """The previous study's rollback ladder, factored out."""
    rb = {"actions": [], "ok": True, "method": "none"}
    if not ok_after_d2:
      rb.update(ok=False, method="aborted_after_failed_second_hop")
      return rb
    if ui_post.exact_sig == ui_target.exact_sig and ui_post.content_sig == ui_target.content_sig:
      rb["method"] = "none_needed"
      return rb
    if ui_post.struct_sig == ui_target.struct_sig and ui_post.content_sig == ui_target.content_sig:
      rb["method"] = "none_needed"
      return rb
    ok, chk = False, None
    if ui_post.struct_sig == ui_target.struct_sig:
      self.dev.tap(*node.center)
      rb["actions"].append("retap")
      chk = self.dev.capture(f"s{step}_p{self.n_probes}_rb_retap")
      ok = R.same_state(chk["ui"], ui_target)
      rb["method"] = "retap"
    else:
      sib = R.selected_sibling(ui_target, node)
      if sib is not None:
        hit = U.element_at(ui_post, *sib.center)
        if hit is None or U.label_of(ui_post, hit) != U.label_of(ui_target, sib) or hit.short_rid != sib.short_rid:
          sib = None
      if sib is not None:
        self.dev.tap(*sib.center)
        rb["actions"].append("reselect")
        chk = self.dev.capture(f"s{step}_p{self.n_probes}_rb_reselect")
        ok = R.same_state(chk["ui"], ui_target)
        rb["method"] = "reselect"
      for i in range(0 if ok else 2):
        self.dev.back()
        rb["actions"].append("back")
        chk = self.dev.capture(f"s{step}_p{self.n_probes}_rb{i}")
        if R.same_state(chk["ui"], ui_target):
          ok = True
          break
        if not (chk["ui"].struct_sig == ui_post.struct_sig or
                (ui_post.ime_visible() and not chk["ui"].ime_visible() and chk["ui"].layout_sig != ui_target.layout_sig)):
          break
      if rb["method"] == "none":
        rb["method"] = "back"
      if not ok and self.cfg.graph_restore and chk is not None:
        ok2, chk2, path = self.restore_via_graph(chk["ui"], ui_target, step)
        rb["graph_restore"] = path
        if path:
          rb["method"] += "+graph"
        if chk2 is not None:
          chk, ok = chk2, ok2
    rb["ok"] = ok
    rb["check"] = chk["cid"] if chk else None
    return rb

  def make_obs(self, pre, post, slot, fam, step, depth, outcome, rb, t0, t_end):
    ui0, ui1 = pre["ui"], post["ui"]
    gm, gm_nodes = self.graph.match(ui1)
    return {"kind": "transition", "origin": "explore", "instance": self.instance, "step": step, "depth": depth,
            "priority": self.policy, "slot_id": slot.slot_id, "family": fam, "selector": slot.sel,
            "tap": list(ui0.nodes[slot.node_idx].center), "src": ui0.summary(), "src_cid": pre["cid"],
            "dst": ui1.summary(), "dst_cid": post["cid"], "outcome": outcome, "dst_graph_match": gm,
            "rollback": rb, "rollback_ok": rb["ok"], "t": t0, "t_total_s": round(t_end - t0, 3)}


def probe_outcome(ui0, ui1):
  if ui1.exact_sig == ui0.exact_sig and ui1.content_sig == ui0.content_sig and ui1.activity == ui0.activity:
    return "no_change"
  if ui1.struct_sig == ui0.struct_sig and ui1.content_sig == ui0.content_sig:
    return "minor_change"
  if ui1.struct_sig == ui0.struct_sig:
    return "in_place_content_change"
  if ui1.package != ui0.package:
    return "left_app"
  return "new_screen"


R.probe_outcome = probe_outcome
R.Explorer = TraceExplorer

# Track the agent's executed steps for trace context, via the step log.
_orig_log = R.Episode.log


def _log(self, name, row):
  if name == "steps":
    row["logged_at"] = time.time()
    ex = TraceExplorer.INSTANCE
    if ex is not None:
      try:
        ui = R.load_ui(self.out, row["pre"]["cid"])
        tok, fam, _ = C.action_token(ui, row["action"])
        ex.own.append({"state": ui.struct_sig, "struct": ui.struct_sig, "tok": tok, "fam": fam})
        row["agent_family"] = fam
        row["agent_token"] = tok
      except Exception as e:  # pylint: disable=broad-except
        row["token_error"] = repr(e)
  _orig_log(self, name, row)


R.Episode.log = _log

_orig_cfg = R.Episode.explore_cfg


def _cfg(self):
  c = _orig_cfg(self)
  c.max_depth = int(os.environ.get("TR_DEPTH", "2"))
  return c


R.Episode.explore_cfg = _cfg

_orig_setup = R.Episode.setup


def _setup(self):
  meta = _orig_setup(self)
  meta["tr_policy"] = ARGS_POLICY if self.args.explore else "none"
  (self.out / "meta.json").write_text(R.jdump(meta))
  return meta


R.Episode.setup = _setup

if __name__ == "__main__":
  R.main()
