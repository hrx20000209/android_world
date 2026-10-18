"""Sum mechanism counters (m_*) over an arm's p0 results."""
import json, sys
from collections import Counter
from pathlib import Path
arm = sys.argv[1]
c, n = Counter(), 0
for f in Path(__file__).resolve().parents[1].glob(f"raw/full/{arm}/*/p0/result.json"):
  r = json.loads(f.read_text()); n += 1
  c.update({k[2:]: v for k, v in r.items() if k.startswith("m_") and isinstance(v, (int, float))})
  for l in (f.parent / "restore.jsonl").open() if (f.parent / "restore.jsonl").exists() else []:
    c["relaunch_ok" if json.loads(l)["ok"] else "relaunch_fail"] += 1
keys = ["calls", "bootstrap", "func_apps", "func_steps", "func_edge_no_edge", "func_entry_edges", "form_apps", "form_steps", "form_edge_lookups", "form_entry_edges", "form_entry_via_explored_edge", "form_entry_then_filled", "func_entry_via_explored_edge", "probes_e1",
        "probes_fx", "probes_d2", "probes_rv", "learned_effectful", "learned_group", "p_rev_dropped", "fx_entry_need", "fx_L1", "fx_L2", "fx_L3", "fx_rel_pos", "probes_total", "windows_explored", "joint_restore_ok", "joint_restore_fail", "drift_stops", "learned_edit_effect", "route_lookups", "route_apps", "route_hops", "route_hops_explore_only", "route_then_applied", "reuse_skip", "reuse_repeat", "reuse_loop", "reuse_absent", "reuse_brake_landing", "reuse_brake_undo", "requery", "verify_relaxed", "verify_relocated", "macro_steps", "macro_hops", "macro_none", "macro_brake_landing", "macro_brake_undo", "restored_then_executed", "decision_dropped", "relaunch_ok", "relaunch_fail", "model_used_explored_edge", "unstick_back"]
print(f"[{arm} mech] n={n} " + " ".join(f"{k}={c[k]}" for k in keys))
