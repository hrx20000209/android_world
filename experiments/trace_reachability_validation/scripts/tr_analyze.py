"""Phase II analysis (online TG vs RR, depth <= 2)."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

U = C.U


def main(raw="raw/phase2", out="analysis/phase2"):
  RAW, OUT = C.ROOT / raw, C.ROOT / out
  OUT.mkdir(parents=True, exist_ok=True)
  rows, eps = [], []
  for d in sorted(RAW.glob("*/*")):
    if not (d / "result.json").exists() or "_partial_" in d.name:
      continue
    meta = json.loads((d / "meta.json").read_text())
    res = json.loads((d / "result.json").read_text())
    arm = meta.get("tr_policy")
    steps = C.jl(d / "steps.jsonl")
    wins = {w["step"]: w for w in C.jl(d / "windows.jsonl")}
    probes = C.jl(d / "probes.jsonl")
    eps.append({"task": meta["task"], "seed": meta["seed"], "arm": arm, "success": res.get("is_successful"),
                "steps": res.get("steps"), "model_calls": res.get("model_calls"), "n_probes": len(probes),
                "n_probes_d1": sum(p["depth"] == 1 for p in probes), "n_probes_d2": sum(p["depth"] == 2 for p in probes),
                "rollback_fail": sum(not p["rollback_ok"] for p in probes),
                "disabled": res.get("explore_disabled"), "wall_s": res.get("wall_s"), "error": res.get("error"),
                "requery": sum(bool(s.get("requery")) for s in steps), "ep": str(d)})
    for i, s in enumerate(steps):
      w = wins.get(s["step"], {})
      fam = s.get("agent_family")
      ui = C.load_ui(str(d), s["pre"]["cid"])
      safe = {C.family_of_sel(sl.sel) for sl in U.enumerate_slots(ui) if sl.safe}
      d1 = [p for p in w.get("probes", []) if p.get("depth", 1) == 1]
      probed = [p["family"] for p in d1]
      ri = w.get("rank_info") or {}
      r = {"task": meta["task"], "seed": meta["seed"], "arm": arm, "step": s["step"], "fam": fam,
           "is_click": fam is not None, "eligible": fam in safe if fam else False,
           "window_active": bool(ri), "n_probed": len(probed), "hit": fam in probed if fam else False,
           "model_latency_s": s["model_latency_s"], "window_s": w.get("window_s", 0.0),
           "requery": s.get("requery")}
      for p in ("P4", "P0u", "P1"):
        lst = ri.get(f"rank_{p}") or []
        r[f"rank_{p}"] = (lst.index(fam) + 1) if fam in lst else None
      # reuse validity of the pre-observation
      if r["hit"]:
        pr = [p for p in d1 if p["family"] == fam][0]
        nxt_cid = steps[i + 1]["pre"]["cid"] if i + 1 < len(steps) else res.get("final_cid")
        try:
          real = C.load_ui(str(d), nxt_cid)
          obs = C.load_ui(str(d), pr["dst_cid"])
          r["reuse_exact"] = real.exact_sig == obs.exact_sig
          r["reuse_struct"] = real.struct_sig == obs.struct_sig
          r["reuse_content"] = real.content_sig == obs.content_sig
        except Exception:  # pylint: disable=broad-except
          pass
        r["probe_done_before_model"] = (pr["t"] + pr["t_total_s"]) <= (s["pre"]["t"] + s["model_latency_s"] + 1.0 + s["pre"].get("settle_s", 0))
        if i + 1 < len(steps) and s.get("logged_at"):
          nx = steps[i + 1]["pre"]
          ready = nx["t"] + 1.0 + nx.get("settle_s", 0) + nx.get("capture_s", 0)
          r["obs_latency_saved_s"] = round(ready - s["logged_at"], 3)
      # depth-2 pair
      if fam and i + 1 < len(steps) and steps[i + 1].get("agent_family"):
        f2 = steps[i + 1]["agent_family"]
        r["next_fam"] = f2
        r["d2_pair_hit"] = any(x.get("first_family") == fam and x.get("family") == f2 for x in w.get("depth2", []))
        r["d2_first_hit"] = any(x.get("first_family") == fam for x in w.get("depth2", []))
      rows.append(r)
  (OUT / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
  (OUT / "episodes.json").write_text(json.dumps(eps, indent=1))
  S = {}
  for arm in ("TG", "RR", "TG1", "RR1", "TG1C"):
    R = [r for r in rows if r["arm"] == arm]
    E = [e for e in eps if e["arm"] == arm]
    clicks = [r for r in R if r["is_click"]]
    elig = [r for r in clicks if r["eligible"]]
    act = [r for r in elig if r["window_active"]]
    hits = [r for r in clicks if r["hit"]]
    s = {"episodes": len(E), "success": sum(e["success"] == 1.0 for e in E), "errors": sum(bool(e["error"]) for e in E),
         "steps": len(R), "clicks": len(clicks), "eligible": len(elig), "eligible_with_active_window": len(act),
         "hit_all_clicks": len(hits), "hit_eligible_active": sum(r["hit"] for r in act),
         "reuse_exact": sum(bool(r.get("reuse_exact")) for r in hits),
         "reuse_struct": sum(bool(r.get("reuse_struct")) for r in hits),
         "probe_done_before_model": sum(bool(r.get("probe_done_before_model")) for r in hits),
         "obs_saved_median_s": float(np.median([r["obs_latency_saved_s"] for r in hits if "obs_latency_saved_s" in r])) if any("obs_latency_saved_s" in r for r in hits) else None,
         "d2_first_hit": sum(bool(r.get("d2_first_hit")) for r in R), "d2_pair_hit": sum(bool(r.get("d2_pair_hit")) for r in R),
         "d2_pairs": sum("next_fam" in r for r in R),
         "probes": sum(e["n_probes"] for e in E), "probes_d2": sum(e["n_probes_d2"] for e in E),
         "rollback_fail": sum(e["rollback_fail"] for e in E),
         "episodes_disabled": sum(bool(e["disabled"]) for e in E), "requery": sum(e["requery"] for e in E),
         "model_calls": sum(e["model_calls"] or 0 for e in E), "agent_steps": sum(e["steps"] or 0 for e in E),
         "windows_active": sum(r["window_active"] for r in R),
         "mean_window_s": float(np.mean([r["window_s"] for r in R if r["window_active"]])) if R else None,
         "median_model_latency_s": float(np.median([r["model_latency_s"] for r in R])) if R else None}
    for p in ("P4", "P0u", "P1"):
      for k in (1, 2, 3):
        s[f"rank_{p}_top{k}"] = sum(1 for r in act if r[f"rank_{p}"] and r[f"rank_{p}"] <= k)
    S[arm] = s
  # paired success
  pe = defaultdict(dict)
  for e in eps:
    pe[(e["task"], e["seed"])][e["arm"]] = e["success"] == 1.0
  pairs = [v for v in pe.values() if "TG" in v and "RR" in v]
  S["paired"] = {"pairs": len(pairs), "both": sum(v["TG"] and v["RR"] for v in pairs),
                 "TG_only": sum(v["TG"] and not v["RR"] for v in pairs), "RR_only": sum(v["RR"] and not v["TG"] for v in pairs)}
  probes_all = []
  for e in eps:
    probes_all += [dict(p, arm=e["arm"]) for p in C.jl(Path(e["ep"]) / "probes.jsonl")]
  S["probe_cost"] = {d: {"n": sum(p["depth"] == d for p in probes_all),
                         "median_s": float(np.median([p["t_total_s"] for p in probes_all if p["depth"] == d])) if any(p["depth"] == d for p in probes_all) else None,
                         "rollback_ok": sum(p["rollback_ok"] for p in probes_all if p["depth"] == d)}
                     for d in (1, 2)}
  S["rollback_methods"] = dict(Counter((p["rollback"]["method"], p["rollback_ok"]) for p in probes_all).most_common()) if probes_all else {}
  S["rollback_methods"] = {f"{k[0]}|{k[1]}": v for k, v in Counter((p["rollback"]["method"], p["rollback_ok"]) for p in probes_all).items()}
  (OUT / "summary.json").write_text(json.dumps(S, indent=1))
  print(json.dumps(S, indent=1))


if __name__ == "__main__":
  main(*sys.argv[1:])
