"""Analysis of stored raw evidence (reproducible: reads only raw/ and graphs/).

Outputs (analysis/):
  episodes.csv          one row per episode
  replay_steps.csv      per replay decision: state match, a11y diff, screenshot diff, cause
  replay_outcomes.csv   per replayed (state, action): observed vs reference outcome
  agent_alignment.csv   ref-vs-variant agent trajectory alignment (sequence DP)
  aligned_pairs.csv     per aligned corresponding step: screenshot/a11y/action/transition diffs
  probes.csv            per exploration probe
  windows.csv           per inference window
  graph_metrics.json    Study B graph reconstruction under matching ablations
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rgs_ui as U  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATUS_FRAC = 128 / 2400   # status bar (clock) excluded from pixel metrics
NAV_FRAC = 0.0


def jl(p: Path) -> list[dict]:
  if not p.exists():
    return []
  return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_csv(path: Path, rows: list[dict]):
  path.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    path.write_text("")
    return
  keys = []
  for r in rows:
    for k in r:
      if k not in keys:
        keys.append(k)
  with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
      w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


@lru_cache(maxsize=4096)
def load_ui(ep: str, cid: str) -> U.UIState:
  with gzip.open(Path(ep) / "a11y" / f"{cid}.json.gz", "rt", encoding="utf-8") as f:
    d = json.load(f)
  return U.parse_flat(d["payload"], activity=d["activity"])


@lru_cache(maxsize=2048)
def load_gray(ep: str, cid: str) -> np.ndarray:
  im = Image.open(Path(ep) / "img" / f"{cid}.jpg").convert("L").resize((540, 1200))
  a = np.asarray(im, dtype=np.float32)
  top = int(STATUS_FRAC * a.shape[0])
  return a[top:]


def screen_diff(ep_a: str, cid_a: str, ep_b: str, cid_b: str) -> dict:
  from skimage.metrics import structural_similarity
  a, b = load_gray(ep_a, cid_a), load_gray(ep_b, cid_b)
  d = np.abs(a - b) > 25
  ratio = float(d.mean())
  ys, xs = np.nonzero(d)
  bbox = [int(xs.min()) * 2, int(ys.min() + STATUS_FRAC * 1200) * 2, int(xs.max()) * 2,
          int(ys.max() + STATUS_FRAC * 1200) * 2] if len(xs) else None
  ssim = float(structural_similarity(a[::2, ::2], b[::2, ::2], data_range=255))
  # 8x8 average hash distance
  def ahash(x):
    s = np.asarray(Image.fromarray(x.astype(np.uint8)).resize((8, 16)), dtype=np.float32)
    return s > s.mean()
  ph = int((ahash(a) != ahash(b)).sum())
  return {"pix_diff_ratio": round(ratio, 5), "ssim": round(ssim, 4), "ahash_dist": ph,
          "pix_changed": ratio > 0.001, "pix_changed_05": ratio > 0.005, "diff_bbox": bbox}


# ---------------------------------------------------------------- episodes
def episodes(raw: Path) -> list[dict]:
  rows = []
  for ep in sorted(raw.glob("*/*")):
    if not (ep / "result.json").exists() or "_partial_" in ep.name or "_void_" in ep.name:
      continue
    meta = json.loads((ep / "meta.json").read_text())
    res = json.loads((ep / "result.json").read_text())
    rows.append({"task": meta["task"], "variant": meta["variant"], "mode": meta["mode"], "seed": meta["seed"],
                 "ep": str(ep), "goal": meta["goal"], "success": res.get("is_successful"),
                 "error": res.get("error"), "steps": res.get("steps"), "done": res.get("done"),
                 "model_calls": res.get("model_calls"), "n_probes": res.get("n_probes"),
                 "explore_disabled": res.get("explore_disabled"), "wall_s": res.get("wall_s"),
                 "replay_executed": res.get("replay_executed"), "ref_steps": res.get("ref_steps"),
                 "divergence": res.get("divergence"), "recovery": res.get("recovery"),
                 "success_before_recovery": res.get("success_before_recovery")})
  return rows


def ref_of(task_dir: Path) -> str | None:
  p = task_dir / "reference.json"
  return json.loads(p.read_text()).get("ref") if p.exists() else None


# ---------------------------------------------------------------- replay
def refine_cause(d, ref: U.UIState, live: U.UIState) -> str:
  """Post-hoc cause for a stopped replay step, from raw evidence."""
  cause = d.get("cause", "")
  if d.get("execute"):
    return ""
  if cause in ("ambiguous_target", "missing_target", "coordinate_drift") and d.get("action", {}).get("x") is not None:
    import rgs_run as Rr  # noqa
    rn = U.element_at(ref, int(d["action"]["x"]), int(d["action"]["y"]))
    if rn is None:
      return cause
    lab = U.label_of(ref, rn)
    same = [n for n in live.app_nodes() if n.visible and U.label_of(live, n) == lab and n.short_rid == rn.short_rid]
    if lab and not same:
      return "target_not_on_screen"
    if len(same) > 1:
      return "duplicate_targets"
    if len(same) == 1:
      return "target_moved" if same[0].bounds != rn.bounds else "same_target_other_context"
  return cause


def replay_rows(eps: list[dict]):
  steps, outs = [], []
  for e in eps:
    if not e["mode"].startswith("replay"):
      continue
    ep = Path(e["ep"])
    meta = json.loads((ep / "meta.json").read_text())
    ref = meta["ref"]
    for d in jl(ep / "replay.jsonl"):
      r = {"task": e["task"], "variant": e["variant"], "mode": e["mode"], "ref_step": d["ref_step"],
           "action_type": d.get("action_type"), "state_match": d.get("state_match"),
           "execute": d.get("execute"), "reason": d.get("reason", ""), "cause": d.get("cause", ""),
           "match_category": d.get("match_category", ""), "coord_hit_same_target": d.get("coord_hit_same_target"),
           "relocated": d.get("relocated"), "parameter_bound": d.get("parameter_bound"),
           "n_candidates": d.get("n_candidates")}
      # recompute from raw with the current signature definitions
      try:
        ua, ub = load_ui(ref, d["ref_pre_cid"]), load_ui(str(ep), d["pre"]["cid"])
        diff = U.diff_elements(ua, ub)
        r["content_changed"] = ua.content_sig != ub.content_sig
        r["state_match_online"] = r["state_match"]
        r["state_match"] = ("struct" if ua.struct_sig == ub.struct_sig else "layout" if ua.layout_sig == ub.layout_sig
                            else "activity_only" if ua.activity == ub.activity else "none")
        r["cause_refined"] = refine_cause(d, ua, ub)
      except Exception:  # pylint: disable=broad-except
        diff = d.get("a11y_diff") or {}
      for k in ("added", "removed", "moved", "mean_drift_px", "max_drift_px", "text_changed", "rid_added",
                "rid_removed", "clickable_set_changed", "clickable_struct_changed", "struct_equal",
                "layout_equal", "exact_equal", "same_activity"):
        r["a11y_" + k] = diff.get(k)
      try:
        r.update(screen_diff(ref, d["ref_pre_cid"], str(ep), d["pre"]["cid"]))
      except Exception as ex:  # pylint: disable=broad-except
        r["screen_err"] = repr(ex)
      r["ref_ep"], r["ref_cid"], r["live_ep"], r["live_cid"] = ref, d["ref_pre_cid"], str(ep), d["pre"]["cid"]
      steps.append(r)
    for o in jl(ep / "outcomes.jsonl"):
      outs.append({"task": e["task"], "variant": e["variant"], "mode": e["mode"], **o})
  return steps, outs


# ---------------------------------------------------------------- alignment
def recompute_slot(ui, action, extras):
  """Slot of an executed action, recomputed from raw a11y with the current
  (z-order aware) hit test, so all episodes use the same definition."""
  import rgs_run as R
  try:
    return R.action_slot(ui, action, extras)[0]
  except Exception:  # pylint: disable=broad-except
    return str(action.get("action_type"))


def step_keys(ep: Path) -> list[dict]:
  out = []
  for s in jl(ep / "steps.jsonl"):
    if s.get("label", "agent") != "agent":
      continue
    ui = load_ui(str(ep), s["pre"]["cid"])
    out.append({"cid": s["pre"]["cid"], "struct": ui.struct_sig, "layout": ui.layout_sig,
                "activity": ui.activity, "slot": recompute_slot(ui, s["action"], s.get("extras") or {}), "atype": s["action"]["action_type"],
                "step": s["step"]})
  return out


def align(a: list[dict], b: list[dict]):
  """Needleman-Wunsch on pre-states. match: struct equal (score 2), layout
  equal (score 1, kept as 'ambiguous'), else mismatch not allowed (gap)."""
  n, m = len(a), len(b)
  S = np.full((n + 1, m + 1), -1e9)
  S[0, :] = -np.arange(m + 1)
  S[:, 0] = -np.arange(n + 1)
  bt = np.zeros((n + 1, m + 1), dtype=int)
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      sc = 2 if a[i-1]["struct"] == b[j-1]["struct"] else 1 if a[i-1]["layout"] == b[j-1]["layout"] else -1e9
      opts = [S[i-1, j-1] + sc, S[i-1, j] - 1, S[i, j-1] - 1]
      k = int(np.argmax(opts))
      S[i, j], bt[i, j] = opts[k], k
  i, j, path = n, m, []
  while i > 0 or j > 0:
    if i > 0 and j > 0 and bt[i, j] == 0:
      path.append((i-1, j-1)); i, j = i-1, j-1
    elif i > 0 and (j == 0 or bt[i, j] == 1):
      path.append((i-1, None)); i -= 1
    else:
      path.append((None, j-1)); j -= 1
  return path[::-1]


def agent_alignment(eps: list[dict], raw: Path):
  summary, pairs = [], []
  by_task = defaultdict(list)
  for e in eps:
    if e["mode"] == "agent" and not e["error"]:
      by_task[e["task"]].append(e)
  for task, lst in by_task.items():
    ref = ref_of(raw / task)
    if not ref:
      continue
    ka = step_keys(Path(ref))
    for e in lst:
      if e["ep"] == ref or e["variant"].startswith("v0"):
        continue
      kb = step_keys(Path(e["ep"]))
      path = align(ka, kb)
      matched = [(i, j) for i, j in path if i is not None and j is not None]
      n_struct = sum(1 for i, j in matched if ka[i]["struct"] == kb[j]["struct"])
      first_div = next((idx for idx, (i, j) in enumerate(path) if i is None or j is None), None)
      # re-convergence: a match after the first gap
      reconv = first_div is not None and any(i is not None and j is not None for i, j in path[first_div:])
      summary.append({"task": task, "variant": e["variant"], "ref_len": len(ka), "var_len": len(kb),
                      "matched": len(matched), "matched_struct": n_struct,
                      "matched_layout_only": len(matched) - n_struct,
                      "ref_only": sum(1 for i, j in path if j is None), "var_only": sum(1 for i, j in path if i is None),
                      "first_divergence_pos": first_div, "reconverged": reconv,
                      "ref_success": None, "var_success": e["success"]})
      for i, j in matched:
        A, B = ka[i], kb[j]
        ua, ub = load_ui(ref, A["cid"]), load_ui(e["ep"], B["cid"])
        diff = U.diff_elements(ua, ub)
        row = {"task": task, "variant": e["variant"], "ref_step": A["step"], "var_step": B["step"],
               "match": "struct" if A["struct"] == B["struct"] else "layout_ambiguous",
               "same_action_slot": A["slot"] == B["slot"], "same_action_type": A["atype"] == B["atype"],
               "ref_slot": A["slot"], "var_slot": B["slot"]}
        row.update({"a11y_" + k: v for k, v in diff.items()})
        row.update(screen_diff(ref, A["cid"], e["ep"], B["cid"]))
        # transition: next aligned states equal?
        na = ka[i+1]["struct"] if i + 1 < len(ka) else None
        nb = kb[j+1]["struct"] if j + 1 < len(kb) else None
        row["same_slot_same_next_struct"] = (A["slot"] == B["slot"] and na is not None and na == nb) if A["slot"] == B["slot"] else None
        row["ref_cid"], row["var_cid"], row["ref_ep"], row["var_ep"] = A["cid"], B["cid"], ref, e["ep"]
        pairs.append(row)
  return summary, pairs


# ---------------------------------------------------------------- study B
def probe_rows(eps):
  probes, windows = [], []
  for e in eps:
    if e["mode"] != "agent":
      continue
    ep = Path(e["ep"])
    for p in jl(ep / "probes.jsonl"):
      probes.append({"task": e["task"], "variant": e["variant"], "step": p["step"], "slot_id": p["slot_id"],
                     "priority": p["priority"], "outcome": p["outcome"], "rollback_ok": p["rollback_ok"],
                     "rollback_method": p["rollback"]["method"], "dst_graph_match": p["dst_graph_match"],
                     "t_act_s": p["t_act_s"], "t_observe_s": p["t_observe_s"], "t_rollback_s": p["t_rollback_s"],
                     "t_total_s": p["t_total_s"], "src_struct": p["src"]["struct_sig"],
                     "dst_struct": p["dst"]["struct_sig"], "src_cid": p["src_cid"], "dst_cid": p["dst_cid"],
                     "ep": e["ep"], "t": p["t"]})
    steps = {s["step"]: s for s in jl(ep / "steps.jsonl")}
    for w in jl(ep / "windows.jsonl"):
      st = steps.get(w["step"], {})
      ps = w.get("plan_stats") or {}
      windows.append({"task": e["task"], "variant": e["variant"], "step": w["step"], "stop": w["stop"],
                      "n_probes": len(w["probes"]), "window_s": w.get("window_s", 0),
                      "model_latency_s": st.get("model_latency_s"), "graph_match": w.get("graph_match"),
                      **{"plan_" + k: v for k, v in ps.items()}})
  return probes, windows


def main():
  raw = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "raw/main")
  out = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "analysis")
  out.mkdir(parents=True, exist_ok=True)
  eps = episodes(raw)
  write_csv(out / "episodes.csv", eps)
  rs, ro = replay_rows(eps)
  write_csv(out / "replay_steps.csv", rs)
  write_csv(out / "replay_outcomes.csv", ro)
  al, pairs = agent_alignment(eps, raw)
  write_csv(out / "agent_alignment.csv", al)
  write_csv(out / "aligned_pairs.csv", pairs)
  pr, wi = probe_rows(eps)
  write_csv(out / "probes.csv", pr)
  write_csv(out / "windows.csv", wi)
  print(f"episodes={len(eps)} replay_steps={len(rs)} outcomes={len(ro)} alignments={len(al)} "
        f"pairs={len(pairs)} probes={len(pr)} windows={len(wi)}")


if __name__ == "__main__":
  main()
