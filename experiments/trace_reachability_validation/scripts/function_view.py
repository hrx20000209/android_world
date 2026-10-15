"""Unified view of graph execution: every inference-free action sequence is a function application.
open_app(app)   entry: launcher screen; parameter: app named in the goal (graph has seen its package)
confirm()       entry: pure confirmation dialog right after the model's own commit
fill_record(r)  entry: screen where a library/own function's first step grounds; parameters: goal record r
form(r)         entry: screen whose text fields are named by r's fields (FULL14)
Each application is one counted step and replaces one model call per executed action."""
import json, glob, sys
from collections import Counter
for arm in sys.argv[1:]:
  c, tasks, succ, n = Counter(), Counter(), 0, 0
  for f in glob.glob(f"raw/full/{arm}/*/p0/result.json"):
    r = json.load(open(f)); n += 1
    ok = r.get("is_successful") == 1.0; succ += ok
    apps = {"open_app": (r.get("m_bootstrap", 0), r.get("m_bootstrap", 0)),
            "confirm": (r.get("m_jump_commit_steps", 0), r.get("m_jump_commit_steps", 0)),
            "fill_record": (r.get("m_func_apps", 0) - r.get("m_form_apps", 0), r.get("m_func_steps", 0) - r.get("m_form_steps", 0)),
            "form": (r.get("m_form_apps", 0), r.get("m_form_steps", 0))}
    for k, (a, s) in apps.items():
      c[k, "apps"] += a; c[k, "actions"] += s; tasks[k] += a > 0; c[k, "succ_tasks"] += (a > 0 and ok)
    c["calls"] += r.get("model_calls") or 0
  tot_apps = sum(c[k, "apps"] for k in ("open_app", "confirm", "fill_record", "form"))
  tot_act = sum(c[k, "actions"] for k in ("open_app", "confirm", "fill_record", "form"))
  print(f"[{arm}] episodes={n} success={succ} model_calls={c['calls']} | function applications={tot_apps} "
        f"(replaced inference ~{tot_act}, {tot_act / max(1, tot_act + c['calls']):.1%} of all decisions)")
  for k in ("open_app", "confirm", "fill_record", "form"):
    if c[k, "apps"]:
      print(f"    {k:12s} applications={c[k, 'apps']:4d} actions={c[k, 'actions']:4d} tasks={tasks[k]:3d} (successful {c[k, 'succ_tasks']})")
