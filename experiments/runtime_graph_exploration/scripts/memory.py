"""Stage A: evidence-preserving cross-task interaction memory G = (V, E, O).

O  immutable observation records extracted from executed AndroidWorld
   episodes (model / graph-executed actions with their real successors).
V  structural states keyed by (activity, layout signature).
E  (state, action slot) -> observation ids; conflicting successors are kept.

An action is split into operation o (slot identity: cls|rid|desc|label, list
rows collapsed to their row family) and parameters theta (typed text, or the
row text when it occurs verbatim in that episode's goal).

Snapshots implement leave-one-template-out memory: the target template and
its numbered variants (same name after stripping trailing digits) are
excluded. Snapshots are views over the immutable O; nothing is rewritten.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRV = ROOT.parent / "trace_reachability_validation"
sys.path.insert(0, str(TRV / "scripts"))
import common as C  # noqa: E402

U = C.U
R = C.R if hasattr(C, "R") else None
if R is None:
  sys.path.insert(0, str(ROOT.parent / "runtime_graph_study" / "scripts"))
  import rgs_run as R  # noqa: E402

OBS = ROOT / "graph" / "observations.jsonl"
STOP = set("a an the of to in on at for from and or with my me i is are be this that it as by into your all any "
           "app application please using use then also its which what how many much".split())


def template_key(task: str) -> str:
  return re.sub(r"\d+$", "", task)


def norm(t: str) -> str:
  return re.sub(r"\s+", " ", (t or "").strip().lower())


def page_labels(ui, limit=120):
  """Visible text/desc of app nodes (normalized, dedup, screen order)."""
  out, seen = [], set()
  for n in sorted(ui.app_nodes(), key=lambda n: (n.bounds[1], n.bounds[0])):
    if not n.visible:
      continue
    for t in (n.text, n.desc):
      t = norm(t)
      if t and len(t) <= 80 and t not in seen:
        seen.add(t)
        out.append(t)
  return out[:limit]


def state_key(ui):
  return f"{ui.activity}#{ui.layout_sig}"


def _op(row):
  if row.get("kind") == "model":
    return (row.get("action") or {}).get("action_type", "?")
  if row.get("kind") == "replay":
    return "long_press" if (row.get("tok") or "").startswith("LONG_PRESS") else "click"
  return "click"


def extract(raw_roots):
  """Rebuild O from executed episodes. Returns number of observations."""
  OBS.parent.mkdir(parents=True, exist_ok=True)
  n = 0
  with OBS.open("w") as fo:
    for root in raw_roots:
      for ep in sorted(Path(root).glob("*/*/p*")):
        if not (ep / "result.json").exists():
          continue
        meta = json.loads((ep / "meta.json").read_text())
        res = json.loads((ep / "result.json").read_text())
        goal = meta.get("goal", "")
        rows = [json.loads(l) for l in open(ep / "steps.jsonl")]
        rows = [r for r in rows if r.get("kind") in ("model", "jump", "replay") and r.get("pre")]
        for i, r in enumerate(rows):
          post = r.get("post_cid")
          if post is None:
            post = rows[i + 1]["pre"]["cid"] if i + 1 < len(rows) else res.get("final_cid")
          try:
            ui0 = R.load_ui(ep, r["pre"]["cid"])
            ui1 = R.load_ui(ep, post) if post else None
          except Exception:  # pylint: disable=broad-except
            continue
          op = _op(r)
          act = r.get("action") or {}
          tok = r.get("tok") or ""
          sel, fam, param, grounding = None, None, None, "n/a"
          if op in ("click", "long_press", "double_tap"):
            if r.get("kind") == "model" and act.get("x") is not None:
              node = U.element_at(ui0, int(act["x"]), int(act["y"]))
            else:
              node = None
              want = tok.split(":", 1)[-1]
              for sl in U.enumerate_slots(ui0):
                if C.family_of_sel(sl.sel) == want:
                  node = ui0.nodes[sl.node_idx]
                  break
            if node is not None:
              sel = U.selector(ui0, node)
              fam = C.family_of_sel(sel)
              same = [sl for sl in U.enumerate_slots(ui0) if C.family_of_sel(sl.sel) == fam]
              grounding = "UNIQUE" if len(same) == 1 else ("AMBIGUOUS" if same else "UNSLOTTED")
              lab = norm(sel.get("label") or sel.get("text") or sel.get("desc"))
              if lab and len(lab) >= 3 and lab in norm(goal):
                param = lab
            else:
              grounding = "MISSING"
          elif op == "input_text":
            param = act.get("text")
            fam = tok
          else:
            fam = tok
          slot_safe = None
          if sel is not None:
            for sl in U.enumerate_slots(ui0):
              if sl.node_idx == node.idx:
                slot_safe = {"safe": sl.safe, "reason": sl.unsafe_reason}
                break
          fo.write(json.dumps({
              "obs": n, "task": meta["task"], "template": template_key(meta["task"]),
              "instance": f"{ep.parent.parent.name}/{ep.name}/s{meta.get('seed')}", "goal": goal,
              "episode_success": res.get("is_successful") == 1.0, "origin": r.get("kind"), "step": r.get("step"),
              "op": op, "fam": fam, "sel": {k: (sel or {}).get(k) for k in ("cls", "rid", "desc", "label", "in_list")} if sel else None,
              "param": param, "grounding": grounding, "slot": slot_safe,
              "src": {"act": ui0.activity, "pkg": ui0.package, "layout": ui0.layout_sig, "struct": ui0.struct_sig,
                      "key": state_key(ui0)},
              "dst": ({"act": ui1.activity, "pkg": ui1.package, "layout": ui1.layout_sig, "struct": ui1.struct_sig,
                       "key": state_key(ui1), "labels": page_labels(ui1)} if ui1 is not None else None),
          }) + "\n")
          n += 1
  return n


def structural_labels(o):
  """Successor labels minus that episode's own task parameters: text that
  occurs in the historical goal (titles, names, search terms) is runtime data
  of another task, not interface structure."""
  g = norm(o["goal"])
  return [t for t in o["dst"]["labels"] if t not in g]


class Snapshot:
  """Leave-one-template-out view of O for one target task."""

  def __init__(self, target_task: str | None, obs_path: Path = OBS):
    tk = template_key(target_task) if target_task else None
    self.target = target_task
    self.obs = []
    for l in open(obs_path):
      o = json.loads(l)
      if tk is not None and o["template"] == tk:
        continue
      self.obs.append(o)
    self.by_state = defaultdict(list)          # src key -> obs
    self.by_edge = defaultdict(list)           # (src key, fam) -> obs
    self.goals = defaultdict(set)              # template -> goals
    for o in self.obs:
      self.by_state[o["src"]["key"]].append(o)
      if o["fam"]:
        self.by_edge[(o["src"]["key"], o["fam"])].append(o)
      self.goals[o["template"]].add(o["goal"])

  def labels_after(self, key):
    """Union of labels on pages one click away from state `key` (cached)."""
    if not hasattr(self, "_la"):
      self._la = {}
    if key not in self._la:
      ls = set()
      for o in self.by_state.get(key, []):
        if o["dst"] and o["op"] in ("click", "long_press"):
          ls.update(structural_labels(o))
      self._la[key] = ls
    return self._la[key]

  def snapshot_id(self):
    return f"loo:{template_key(self.target) if self.target else 'all'}:{len(self.obs)}"
