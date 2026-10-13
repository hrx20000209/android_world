"""Headline numbers with explicit denominators -> analysis/summary.json,
reports/experiment_summary.csv, reports/task_manifest.csv."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rgs_analyze as A  # noqa: E402
import rgs_run as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "analysis"


def rd(name):
  p = AN / name
  return list(csv.DictReader(p.open())) if p.exists() and p.stat().st_size else []


def b(x):
  return str(x) == "True"


def f(x, d=0.0):
  try:
    return float(x)
  except (TypeError, ValueError):
    return d


def rate(k, n):
  return {"k": k, "n": n, "p": round(k / n, 4) if n else None}


def per_task_avg(rows, fn, key="task"):
  by = defaultdict(list)
  for r in rows:
    by[r[key]].append(fn(r))
  vals = [np.mean(v) for v in by.values() if v]
  return round(float(np.mean(vals)), 4) if vals else None


def main():
  eps = rd("episodes.csv")
  S: dict = {}
  manifest = {}
  for m in ("batch1_manifest.json", "batch2_manifest.json"):
    p = ROOT / "configs" / m
    if p.exists():
      for t in json.loads(p.read_text())["tasks"]:
        manifest[t["task"]] = t
  tasks = sorted({e["task"] for e in eps})
  # ---------------- success table
  succ = {}
  for t in tasks:
    row = {"task": t, "app": manifest.get(t, {}).get("app", "")}
    te = [e for e in eps if e["task"] == t and not e["error"]]
    ref = [e for e in te if e["variant"].startswith("v0_ref")]
    row["ref_attempts"] = len(ref)
    row["ref_success"] = sum(f(e["success"]) == 1 for e in ref)
    ex = [e for e in te if e["mode"] == "agent" and e["variant"] in ("v0", "v1", "v2", "v3", "v4")]
    ne = [e for e in te if e["mode"] == "agent" and e["variant"].endswith("_noexp")]
    row["explore_success"] = f"{sum(f(e['success']) == 1 for e in ex)}/{len(ex)}"
    row["noexp_success"] = f"{sum(f(e['success']) == 1 for e in ne)}/{len(ne)}"
    for m in ("replay_strict", "replay_struct"):
      me = [e for e in te if e["mode"] == m]
      comp = [e for e in me if e["divergence"] in ("", None, "null")]
      ok = [e for e in me if f(e["success_before_recovery"] if e.get("recovery") not in ("", None) else e["success"]) == 1]
      row[m + "_completed"] = f"{len(comp)}/{len(me)}"
      row[m + "_task_success"] = f"{len(ok)}/{len(me)}"
    rec = [e for e in te if e["mode"] == "replay_struct" and e.get("recovery") not in ("", None)]
    row["recovery_success"] = f"{sum(f(e['success']) == 1 for e in rec)}/{len(rec)}"
    row["recovery_model_calls"] = sum(json.loads(e["recovery"]).get("model_calls", 0) for e in rec)
    row["errors"] = sum(1 for e in eps if e["task"] == t and e["error"])
    succ[t] = row
  S["success_table"] = list(succ.values())

  # paired exploration harm (same variant, explore vs noexp)
  pairs = []
  for t in tasks:
    for v in ("v1", "v2", "v3", "v4"):
      x = [e for e in eps if e["task"] == t and e["variant"] == v and e["mode"] == "agent" and not e["error"]]
      n = [e for e in eps if e["task"] == t and e["variant"] == v + "_noexp" and not e["error"]]
      if x and n:
        pairs.append((f(x[0]["success"]) == 1, f(n[0]["success"]) == 1, x[0]["explore_disabled"] not in ("", None)))
  S["explore_vs_noexp"] = {
      "pairs": len(pairs), "both": sum(a and c for a, c, _ in pairs),
      "explore_only": sum(a and not c for a, c, _ in pairs), "noexp_only": sum(c and not a for a, c, _ in pairs),
      "neither": sum((not a) and (not c) for a, c, _ in pairs),
      "explore_success": sum(a for a, _, _ in pairs), "noexp_success": sum(c for _, c, _ in pairs),
      "episodes_with_rollback_failure": sum(d for _, _, d in pairs),
      "noexp_only_with_rollback_failure": sum(1 for a, c, d in pairs if c and not a and d)}

  # ---------------- Study A: corresponding steps from structure-aware replay
  rs = rd("replay_steps.csv")
  cs = [r for r in rs if r["mode"] == "replay_struct" and r["state_match"] in ("struct", "layout")
        and r.get("pix_diff_ratio") not in ("", None)]
  lv = {
      "screenshot_changed": lambda r: b(r["pix_changed"]),
      "screenshot_changed_0.5pct": lambda r: b(r["pix_changed_05"]),
      "a11y_exact_changed": lambda r: not b(r["a11y_exact_equal"]),
      "struct_sig_changed": lambda r: not b(r["a11y_struct_equal"]),
      "clickable_set_changed": lambda r: b(r["a11y_clickable_set_changed"]),
      "clickable_struct_changed (tautologically 0 given layout match)": lambda r: b(r["a11y_clickable_struct_changed"]),
      "content_sig_changed": lambda r: b(r.get("content_changed")),
      "elements_added_or_removed": lambda r: f(r["a11y_added"]) + f(r["a11y_removed"]) > 0,
      "element_moved": lambda r: f(r["a11y_moved"]) > 0,
      "rid_set_changed": lambda r: f(r["a11y_rid_added"]) + f(r["a11y_rid_removed"]) > 0,
  }
  S["corresponding_steps"] = {"definition": "structure-aware replay decisions whose live pre-state matched the reference pre-state at struct or layout level",
                              "n": len(cs), "n_tasks": len({r['task'] for r in cs})}
  for k, fn in lv.items():
    S["corresponding_steps"][k] = {"pooled": rate(sum(fn(r) for r in cs), len(cs)),
                                   "per_task_avg": per_task_avg(cs, lambda r, fn=fn: float(fn(r)))}
  drift = [f(r["a11y_mean_drift_px"]) for r in cs if f(r["a11y_moved"]) > 0]
  S["corresponding_steps"]["drift_px_when_moved"] = {"n": len(drift), "median": float(np.median(drift)) if drift else None,
                                                     "max": float(max(drift)) if drift else None}
  # RQ4: screenshot change vs actionable change
  pc = [r for r in cs if b(r["pix_changed"])]
  S["rq4"] = {"screenshot_changed": len(pc),
              "of_which_clickable_struct_changed": sum(b(r["a11y_clickable_struct_changed"]) for r in pc),
              "of_which_clickable_set_changed": sum(b(r["a11y_clickable_set_changed"]) for r in pc),
              "of_which_struct_sig_changed": sum(not b(r["a11y_struct_equal"]) for r in pc),
              "screenshot_unchanged_but_clickable_set_changed": sum(1 for r in cs if not b(r["pix_changed"]) and b(r["a11y_clickable_set_changed"])),
              "a11y_exact_equal_but_screenshot_changed": sum(1 for r in cs if b(r["a11y_exact_equal"]) and b(r["pix_changed"])),
              "a11y_exact_equal_but_screenshot_changed_tasks": sorted({r["task"] for r in cs if b(r["a11y_exact_equal"]) and b(r["pix_changed"])})}
  # replay outcomes (transition level)
  ro = rd("replay_outcomes.csv")
  for m in ("replay_strict", "replay_struct"):
    c = Counter(o["category"] for o in ro if o["mode"] == m)
    S[f"transition_outcomes_{m}"] = dict(c) | {"n": sum(c.values())}
  # replay decisions
  refined = {(r["task"], r["variant"], r["mode"]): (r.get("cause_refined") or r.get("cause")) for r in rs
             if r["execute"] != "True"}
  for m in ("replay_strict", "replay_struct"):
    me = [e for e in eps if e["mode"] == m and not e["error"]]
    causes_ref = Counter(refined.get((e["task"], e["variant"], m), "completed") for e in me)
    causes = Counter()
    pos = []
    for e in me:
      dv = json.loads(e["divergence"]) if e["divergence"] not in ("", None, "null") else None
      causes[(dv or {}).get("cause", "completed") if dv else "completed"] += 1
      if dv:
        pos.append(dv["ref_step"])
    steps_m = [r for r in rs if r["mode"] == m and r["action_type"] in ("click", "long_press", "input_text") and r.get("match_category")]
    S[m] = {"episodes": len(me), "causes": dict(causes), "causes_refined": dict(causes_ref),
            "task_success_without_recovery": sum(1 for e in me if f(e["success_before_recovery"] if e.get("recovery") not in ("", None) else e["success"]) == 1),
            "divergence_ref_step_median": float(np.median(pos)) if pos else None,
            "grounded_steps": len(steps_m),
            "relocated": sum(b(r["relocated"]) for r in steps_m),
            "coord_hit_same_target": sum(b(r["coord_hit_same_target"]) for r in steps_m),
            "parameter_bound_steps": sum(b(r["parameter_bound"]) for r in rs if r["mode"] == m),
            "match_categories": dict(Counter(r["match_category"] for r in steps_m))}
  # agent alignment (clean noexp runs vs clean reference)
  al = [r for r in rd("agent_alignment.csv") if r["variant"].endswith("_noexp")]
  ap = [r for r in rd("aligned_pairs.csv") if r["variant"].endswith("_noexp")]
  S["agent_alignment_noexp"] = {
      "trajectories": len(al),
      "ref_steps": sum(int(r["ref_len"]) for r in al), "var_steps": sum(int(r["var_len"]) for r in al),
      "matched": sum(int(r["matched"]) for r in al), "matched_struct": sum(int(r["matched_struct"]) for r in al),
      "matched_layout_only": sum(int(r["matched_layout_only"]) for r in al),
      "ref_only": sum(int(r["ref_only"]) for r in al), "var_only": sum(int(r["var_only"]) for r in al),
      "identical_length_and_all_struct": sum(1 for r in al if r["ref_len"] == r["var_len"] == r["matched_struct"]),
      "reconverged_after_divergence": sum(b(r["reconverged"]) for r in al if r["first_divergence_pos"] not in ("", None)),
      "diverged": sum(1 for r in al if r["first_divergence_pos"] not in ("", None)),
      "pairs": len(ap), "pairs_same_slot": sum(b(r["same_action_slot"]) for r in ap),
      "pairs_pix_changed": sum(b(r["pix_changed"]) for r in ap),
      "pairs_clickable_set_changed": sum(b(r["a11y_clickable_set_changed"]) for r in ap),
      "pairs_same_slot_same_next": sum(1 for r in ap if r["same_slot_same_next_struct"] == "True"),
      "pairs_same_slot_diff_next": sum(1 for r in ap if r["same_slot_same_next_struct"] == "False"),
  }

  # ---------------- Study B
  pr = rd("probes.csv")
  wi = rd("windows.csv")
  S["probes"] = {"n": len(pr), "outcomes": dict(Counter(p["outcome"] for p in pr)),
                 "rollback_ok": rate(sum(b(p["rollback_ok"]) for p in pr), len(pr)),
                 "rollback_methods": {f"{k[0]}|{k[1]}": v for k, v in Counter((p["rollback_method"], p["rollback_ok"]) for p in pr).items()},
                 "t_total_median": float(np.median([f(p["t_total_s"]) for p in pr])) if pr else None,
                 "t_total_p90": float(np.percentile([f(p["t_total_s"]) for p in pr], 90)) if pr else None,
                 "t_rollback_median_when_needed": float(np.median([f(p["t_rollback_s"]) for p in pr if f(p["t_rollback_s"]) > 0.05])) if pr else None,
                 "priority": dict(Counter(p["priority"] for p in pr)),
                 "dst_graph_match": dict(Counter(p["dst_graph_match"] for p in pr))}
  lat = [f(w["model_latency_s"]) for w in wi if w.get("model_latency_s") not in ("", None)]
  S["windows"] = {"n": len(wi), "stop": dict(Counter(w["stop"].split(":")[0] for w in wi)),
                  "model_latency_median": float(np.median(lat)) if lat else None,
                  "model_latency_p90": float(np.percentile(lat, 90)) if lat else None,
                  "windows_with_probes": sum(int(f(w["n_probes"])) > 0 for w in wi),
                  "plan_sums": {k: int(sum(f(w.get(k)) for w in wi)) for k in
                                ("plan_slots", "plan_safe", "plan_discover", "plan_revalidate",
                                 "plan_skip_known_same_instance", "plan_skip_known_unrestorable",
                                 "plan_skip_family_unrestorable", "plan_skip_app_unrestorable")}}
  gm = json.loads((AN / "graph_metrics.json").read_text())
  S["graph"] = {}
  for rule in ("exact", "struct", "layout"):
    agg = Counter()
    for t, m in gm.items():
      insts = m["instances"]
      for i, inst in enumerate(insts):
        pi = m[f"{rule}_shared"]["per_instance"].get(inst, {})
        for k, v in pi.items():
          agg[k] += v
          if i > 0:
            agg["later_" + k] += v
    S["graph"][rule] = dict(agg) | {
        "nodes": sum(m[f"{rule}_shared"]["n_nodes"] for m in gm.values()),
        "edges": sum(m[f"{rule}_shared"]["n_edges"] for m in gm.values()),
        "slots": sum(m[f"{rule}_shared"]["n_slots"] for m in gm.values()),
        "multi_outcome_slots": sum(m[f"{rule}_shared"]["multi_outcome_slots"] for m in gm.values()),
        "per_run_nodes_sum": sum(sum(m[f"{rule}_per_run"]["per_instance"].get(i, {}).get("new_nodes", 0) for i in m["instances"]) for m in gm.values())}
  S["graph"]["struct_ambiguous_new_states"] = sum(m["struct_ambiguous_new_states"] for m in gm.values())
  S["graph"]["struct_distinct_states"] = sum(m["struct_distinct_states"] for m in gm.values())
  rv = rd("revalidations.csv")
  for rule in ("exact", "struct", "layout"):
    x = [r for r in rv if r["rule"] == rule and r["origin"] == "explore"]
    y = [r for r in rv if r["rule"] == rule and r["origin"] == "agent"]
    S["graph"][rule]["reval_explore"] = rate(sum(b(r["same_outcome"]) for r in x), len(x))
    S["graph"][rule]["reval_agent"] = rate(sum(b(r["same_outcome"]) for r in y), len(y))

  # ---------------- evidence relevance for the agent's own decisions
  rel = Counter()
  for gx in sorted((ROOT / "graphs" / "main").glob("*.explore.jsonl")):
    task = gx.name.split(".")[0]
    obs = [json.loads(l) for l in gx.read_text().splitlines() if l.strip()]
    for ep in sorted((ROOT / "raw" / "main" / task).glob("v[0-9]_agent")):
      if not (ep / "result.json").exists():
        continue
      steps = A.jl(ep / "steps.jsonl")
      for i, s in enumerate(steps):
        at = s["action"]["action_type"]
        if at not in ("click",):
          continue
        rel["agent_click_steps"] += 1
        ui = A.load_ui(str(ep), s["pre"]["cid"])
        slot = A.recompute_slot(ui, s["action"], s.get("extras") or {})
        prior = [o for o in obs if o["t"] < s["pre"]["t"] and o["src"]["struct_sig"] == ui.struct_sig]
        if prior:
          rel["state_known"] += 1
        hit = [o for o in prior if o["slot_id"] == slot]
        hx = [o for o in hit if o["origin"] == "explore"]
        if not hit:
          continue
        rel["slot_observed_before"] += 1
        if hx:
          rel["slot_observed_by_explorer_before"] += 1
        nxt = steps[i + 1]["pre"]["cid"] if i + 1 < len(steps) else json.loads((ep / "result.json").read_text()).get("final_cid")
        try:
          post = A.load_ui(str(ep), nxt)
        except Exception:  # pylint: disable=broad-except
          continue
        last = hit[-1]
        if last["dst"] and last["dst"]["struct_sig"] == post.struct_sig:
          rel["observed_outcome_matched_struct"] += 1
        elif last["dst"] and last["dst"]["layout_sig"] == post.layout_sig:
          rel["observed_outcome_matched_layout_only"] += 1
        else:
          rel["observed_outcome_differed"] += 1
        same_inst = [o for o in hit if o["instance"] == f"{task}/{ep.name.split('_')[0]}"]
        if same_inst:
          rel["observed_in_same_run"] += 1
  S["evidence_relevance"] = dict(rel)

  AN.mkdir(exist_ok=True)
  (AN / "summary.json").write_text(json.dumps(S, indent=1, ensure_ascii=False))
  with (ROOT / "reports" / "experiment_summary.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(S["success_table"][0].keys()))
    w.writeheader()
    w.writerows(S["success_table"])
  with (ROOT / "reports" / "task_manifest.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["Task ID", "App", "Task Name", "Workflow Category", "Variant Mechanism", "Seeds", "Planned Runs"])
    for i, (t, m) in enumerate(manifest.items()):
      w.writerow([i + 1, m["app"], t, m["category"], m["variant_mechanism"], " ".join(map(str, m["seeds"])),
                  "1-3 clean ref + 5 agent+explore + 4 agent no-explore + 4 strict + 4 struct replay"])
  print(json.dumps({k: v for k, v in S.items() if k != "success_table"}, indent=1)[:6000])


if __name__ == "__main__":
  main()
