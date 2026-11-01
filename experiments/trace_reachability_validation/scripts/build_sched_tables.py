"""Build the L (timing) and P (outcome) tables from the CALIBRATION episodes only and freeze them (SHA-256).

L row = one reasoning window of an assigned condition: app (package on the decision screen), priority (fixed
"normal": the Orin vLLM has no priority control), budget, strategy, inference_s (request -> action available, HTTP
streaming wall, not GPU kernel time), exploration_s (sum of complete probe cycles: tap, observe, return, verify),
completed probes, all recovered, exposed tail (window end after the action was available).
P row = one calibration episode: task kind, assigned (r, K, s), environment success, model calls.
"""
import json
from pathlib import Path

import sched_lookup as SL

ROOT = Path(__file__).resolve().parents[1]
CONDS = {"K0": (0, "full45_value"), "K1v": (1, "full45_value"), "K1s": (1, "screen_order")}


def rows(tag):
  out = {}
  for l in open(ROOT / "reports" / f"{tag}_runs.jsonl"):
    r = json.loads(l)
    out[(r["phase"], r["cond"], r["task"], r["seed"])] = r
  return [r for r in out.values() if r["phase"] == "calib"]


def build_and_freeze(tag, table_path, model_key="GELab-Zero-4B-preview"):
  data = dict(version=SL.VERSION, timings=[], outcomes=[], decisions=[], tag=tag, model_key=model_key,
              notes="Calibration only; held-out episodes never update this table. Priority r fixed to 'normal' "
                    "(no priority control on the Orin vLLM). GUI (Mac emulator) and model (Jetson AGX Orin GPU) do NOT "
                    "share a GPU; the co-located load is a CPU llama.cpp process inside the emulator.")
  for r in rows(tag):
    k, s = CONDS[r["cond"]]
    if r.get("prep_fail") or r.get("error"):
      data.setdefault("excluded_preparation", []).append({"source": r["raw"], "why": r.get("prep_fail") or r.get("error")})
      continue
    data["outcomes"].append(dict(task=r["task"], task_kind=r["task"], priority="normal", budget=k, strategy=s,
                                 success=bool(r["success"]), calls=r["calls"], model_key=model_key, source=r["raw"],
                                 stage="calibration", seed=r["seed"]))
    ep = ROOT / r["raw"]
    steps = {}
    for l in open(ep / "steps.jsonl"):
      x = json.loads(l)
      if x["kind"] == "model" and x.get("t_req"):
        steps[x["step"]] = x
    for l in open(ep / "windows.jsonl"):
      w = json.loads(l)
      st = steps.get(w["step"])
      if st is None:
        continue
      if w.get("stop") in ("transient_state", "dialog_state", "disabled"):
        continue                                  # the page did not permit exploration: not a measurement of (K, s)
      inf = st["t_decision"] - st["t_req"]
      probes = w.get("probes", [])
      tail = max(0.0, (w.get("t_end") or st["t_decision"]) - st["t_decision"])
      data["timings"].append(dict(app=st["pre"]["summary"]["package"], priority="normal", budget=k, strategy=s,
                                  inference_s=round(inf, 3), exploration_s=round(sum(p.get("t") or 0 for p in probes), 3),
                                  completed=len(probes), recovered=all(p.get("rollback_ok") for p in probes),
                                  tail_s=round(tail, 3), model_key=model_key, source=f"{r['raw']}#step{w['step']}"))
  return SL.freeze(data, table_path)


if __name__ == "__main__":
  import sys
  print(build_and_freeze(sys.argv[1], ROOT / "reports" / f"{sys.argv[1]}_lookup_frozen.json"))
