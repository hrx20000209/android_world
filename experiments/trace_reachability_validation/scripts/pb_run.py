"""Pilot B: does parent-level reachability memory change / improve the next action?

Memory source (counterfactual, documented): observations from episodes of OTHER
task variants (different task or different seed; any time) plus the evaluated
episode's own observations recorded before the decision. This stands in for
"what exploration + memory would have revealed at this state"; the evaluated
run's future and any success label are never included.

Conditions (same screenshot, history, model, decoding):
  M0 none | M1 one-hop UTG | M2 + child affordances | M2P placebo (affordance
  lists rotated across elements: same length, wrong attachment) | M3 depth-2
  (verified vs composed) | M4 whole-app raw graph
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import pa_trace as PA  # noqa: E402
import pb_states as PB  # noqa: E402
from android_world.agents import gelab_agent  # noqa: E402

API = "http://127.0.0.1:18095/v1/chat/completions"
OUT = C.ROOT / "analysis" / "pilot_b"
CONDS = ["M0", "M1", "M2", "M2P", "M3", "M4", "M0b", "M2pre"]


def load_states():
  spec = json.loads((C.ROOT / "configs" / "pilot_b_labels.json").read_text())
  pb = json.loads((C.ROOT / "decision_states" / "pilot_b_states.json").read_text())
  dv = json.loads((C.ROOT / "decision_states" / "divergence_candidates.json").read_text())
  out = []
  for s in spec["states"]:
    base = dict(pb[s["i"]] if s["src"] == "pb" else dv[s["i"]])
    base.update({"sid": f"{s['src']}{s['i']}", "set": s["set"], "accept": s["accept"],
                 "efficient": s.get("efficient", [])})
    out.append(base)
  return out


def placebo(recs):
  """Rotate affordance lists by one position across records."""
  if len(recs) < 2:
    return None
  aff = [r["afford"] for r in recs]
  aff = aff[1:] + aff[:1]
  lines = []
  for r, a in zip(recs, aff):
    l = f"- Tap \"{r['name']}\" at ({r['xy'][0]},{r['xy'][1]}) -> opens {r['dst_act']}" + \
        (f" titled '{r['dst_title']}'" if r["dst_title"] else "")
    if a:
      l += f"; that screen offers: {', '.join(a)}"
    lines.append(l)
  return lines


def build_all(states):
  eps = C.episodes()
  obs = C.graph_observations()
  seqs = {e["ep"]: PA.build_episode_seq(e) for e in eps}
  for e in eps:
    for st in seqs[e["ep"]]:
      ui = C.load_ui(e["ep"], st["cid"])
      st["activity"], st["title"] = ui.activity, PB.title_of(ui)
  key_of = {e["ep"]: (e["task"], e["seed"]) for e in eps}
  for s in states:
    own = s["ep"]
    k = key_of[own]
    allowed = [o for o in obs if (key_of.get(o["ep"]) != k) or (o["ep"] == own and o["t"] < s["t"])]
    seqs_other = [seqs[e["ep"]] for e in eps if key_of[e["ep"]] != k]
    ui = C.load_ui(own, s["cid"])
    mem = PB.build_memory(allowed, s["struct"], ui, seqs_other)
    hdr = mem["M1"].split("\n", 1)[0] if mem["M1"] else ""
    p = placebo(mem["records"])
    mem["M2P"] = hdr + "\n" + "\n".join(p) if p else ""
    s["memory"] = {c: mem.get(c, "") for c in ("M1", "M2", "M2P", "M3", "M4")}
    s["mem_records"] = mem["records"]
    s["mem_origin"] = sorted({r["origin"] for r in mem["records"]})
  return states


def query(goal, history, img, extra, before=False):
  hist = "None yet." if not history else "\n".join(
      f"{i}. {h}" for i, h in enumerate(history[-8:], start=max(1, len(history) - 7)))
  msgs = gelab_agent.build_gelab_messages(goal, hist, img)
  if extra and before:
    msgs[1]["content"][0]["text"] = extra + "\n\n" + msgs[1]["content"][0]["text"]
  elif extra:
    msgs[1]["content"][0]["text"] += "\n\n" + extra
  payload = {"model": "GELAB-ZERO-4B", "messages": msgs, "temperature": 0.0, "top_p": 1.0, "max_tokens": 1024,
             "stream": False, "cache_salt": uuid.uuid4().hex}
  t0 = time.time()
  r = requests.post(API, json=payload, timeout=300, proxies={"http": None, "https": None})
  d = r.json()
  return d["choices"][0]["message"]["content"], time.time() - t0, d.get("usage", {})


def run():
  OUT.mkdir(parents=True, exist_ok=True)
  states = build_all(load_states())
  (C.ROOT / "reachability_memory" / "pilot_b_memory.json").write_text(
      json.dumps([{k: s[k] for k in ("sid", "task", "set", "memory", "mem_records", "mem_origin")} for s in states],
                 indent=1, ensure_ascii=False, default=str))
  resp_p = OUT / "responses.jsonl"
  done = {(r["sid"], r["cond"]) for r in C.jl(resp_p)}
  for s in states:
    img = Image.open(Path(s["ep"]) / "img" / f"{s['cid']}.jpg").convert("RGB")
    ui = C.load_ui(s["ep"], s["cid"])
    for c in CONDS:
      if (s["sid"], c) in done:
        continue
      extra = "" if c in ("M0", "M0b") else s["memory"]["M2" if c == "M2pre" else c]
      if c not in ("M0", "M0b") and not extra:
        continue  # no memory of this kind for this state: skip (counted as unavailable)
      text, lat, usage = query(s["goal"], s["history"], img, extra, before=(c == "M2pre"))
      try:
        parsed = gelab_agent.parse_gelab_response(text)
        action, _, _ = gelab_agent.gelab_action_to_json_action(parsed, (1080, 2400))
        ad = action.__dict__
      except Exception as e:  # pylint: disable=broad-except
        parsed, ad = {"parse_error": repr(e)}, {"action_type": "parse_error"}
      tok = C.action_token(ui, ad)[0]
      row = {"sid": s["sid"], "task": s["task"], "set": s["set"], "cond": c, "tok": tok, "action": ad,
             "summary": parsed.get("summary") if isinstance(parsed, dict) else None,
             "latency_s": round(lat, 3), "prompt_tokens": usage.get("prompt_tokens"),
             "completion_tokens": usage.get("completion_tokens"), "extra_chars": len(extra), "response": text}
      with resp_p.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
      print(s["sid"], c, tok[:60], f"{lat:.2f}s", usage.get("prompt_tokens"), flush=True)
  (OUT / "states_scored_input.json").write_text(json.dumps(
      [{k: s[k] for k in ("sid", "task", "app", "set", "accept", "efficient", "orig_tok", "goal", "ep", "cid")}
       for s in states], indent=1, ensure_ascii=False))


def score():
  states = {s["sid"]: s for s in json.loads((OUT / "states_scored_input.json").read_text())}
  R = defaultdict(dict)
  for r in C.jl(OUT / "responses.jsonl"):
    R[r["sid"]][r["cond"]] = r

  def ok(s, tok):
    return any(a in tok for a in s["accept"])

  def eff(s, tok):
    return any(a in tok for a in s["efficient"])
  res = {}
  for c in CONDS:
    rows = [(states[sid], R[sid]) for sid in R if c in R[sid] and "M0" in R[sid]]
    d = {"n": len(rows)}
    for st in ("E", "N", "C"):
      sub = [(s, r) for s, r in rows if s["set"] == st]
      d[f"{st}_n"] = len(sub)
      d[f"{st}_valid"] = sum(ok(s, r[c]["tok"]) for s, r in sub)
      d[f"{st}_valid_M0"] = sum(ok(s, r["M0"]["tok"]) for s, r in sub)
      if st == "N":
        d["N_efficient"] = sum(eff(s, r[c]["tok"]) for s, r in sub)
        d["N_efficient_M0"] = sum(eff(s, r["M0"]["tok"]) for s, r in sub)
    d["changed_vs_M0"] = sum(r[c]["tok"] != r["M0"]["tok"] for s, r in rows)
    d["fixed"] = sum((not ok(s, r["M0"]["tok"])) and ok(s, r[c]["tok"]) for s, r in rows)
    d["broken"] = sum(ok(s, r["M0"]["tok"]) and not ok(s, r[c]["tok"]) for s, r in rows)
    toks = [r[c]["prompt_tokens"] for s, r in rows]
    d["prompt_tokens_mean"] = sum(toks) / len(toks) if toks else None
    d["extra_tokens_mean"] = (sum(r[c]["prompt_tokens"] - r["M0"]["prompt_tokens"] for s, r in rows) / len(rows)) if rows else None
    d["out_tokens_mean"] = sum(r[c]["completion_tokens"] for s, r in rows) / len(rows) if rows else None
    res[c] = d
  (OUT / "score.json").write_text(json.dumps(res, indent=1))
  for c, d in res.items():
    print(c, json.dumps(d))


if __name__ == "__main__":
  {"run": run, "score": score}[sys.argv[1]]()
