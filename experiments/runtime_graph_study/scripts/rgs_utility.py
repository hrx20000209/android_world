"""One-step reasoning-utility check: does observed transition evidence change
GELAB-ZERO-4B's next action, and is the change beneficial?

select : pick decision states (agent steps) that already had graph evidence
         recorded strictly BEFORE the step's observation time (no leakage).
run    : query the model under A (standard input), B (local evidence for the
         current state), C (whole task graph, capped) on the same screenshot.
sheet  : write per-state contact sheets for manual correctness annotation.
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import rgs_ui as U  # noqa: E402
import rgs_analyze as A  # noqa: E402
from android_world.agents import gelab_agent  # noqa: E402

API = "http://127.0.0.1:18095/v1/chat/completions"
OUT = ROOT / "analysis" / "utility"


def jl(p):
  return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []


def short_act(a: str) -> str:
  return a.rsplit(".", 1)[-1] if a else "?"


def describe_dst(o, src_ui: U.UIState | None) -> str:
  if o.get("outcome") == "no_change":
    return "no visible change"
  task = o["instance"].split("/")[0]
  var = o["instance"].split("/")[1]
  ep = ROOT / "raw" / "main" / task / (f"{var}_agent" if not var.startswith("v0_ref") else var)
  try:
    ui = A.load_ui(str(ep), o["dst_cid"])
  except Exception:  # pylint: disable=broad-except
    return f"opens {short_act(o['dst']['activity'])}"
  src_anchor = set(src_ui.anchor_texts()) if src_ui else set()
  new = [t for t in ui.anchor_texts() if t not in src_anchor][:5]
  parts = []
  if o["dst"]["activity"] != o["src"]["activity"]:
    parts.append(f"opens {short_act(o['dst']['activity'])}")
  else:
    parts.append("same screen changes")
  if ui.ime_visible() and not (src_ui and src_ui.ime_visible()):
    parts.append("keyboard appears")
  if new:
    parts.append("shows " + ", ".join(f"'{t[:30]}'" for t in new))
  return "; ".join(parts)


def evidence_lines(obs, struct, t_before, src_ui, cap=10):
  latest = {}
  for o in obs:
    if o["t"] >= t_before or o["src"]["struct_sig"] != struct:
      continue
    if o.get("origin") not in ("explore", "agent") or not o.get("selector"):
      continue
    latest[o["slot_id"]] = o
  lines = []
  for sid, o in sorted(latest.items(), key=lambda kv: (kv[1]["selector"]["bounds"][1], kv[1]["selector"]["bounds"][0])):
    sel = o["selector"]
    name = sel.get("label") or sel.get("desc") or sel.get("rid") or sel.get("cls")
    x1, y1, x2, y2 = sel["bounds"]
    lines.append(f"- Tapping \"{name[:40]}\" at ({(x1+x2)//2},{(y1+y2)//2}) -> {describe_dst(o, src_ui)}")
  return lines[:cap]


def select(n_target=30, seed=0):
  rows = []
  for gx in sorted((ROOT / "graphs" / "main").glob("*.explore.jsonl")):
    task = gx.name.split(".")[0]
    obs = jl(gx)
    for ep in sorted((ROOT / "raw" / "main" / task).glob("v*_agent")):
      res = json.loads((ep / "result.json").read_text()) if (ep / "result.json").exists() else None
      if not res or res.get("error"):
        continue
      meta = json.loads((ep / "meta.json").read_text())
      steps = jl(ep / "steps.jsonl")
      for s in steps:
        if s.get("requery") or s["action"]["action_type"] in ("status", "answer"):
          continue
        ui = A.load_ui(str(ep), s["pre"]["cid"])
        lines = evidence_lines(obs, ui.struct_sig, s["pre"]["t"], ui)
        if len(lines) < 2:
          continue
        hist = [x["summary"] for x in steps if x["step"] < s["step"]]
        rows.append({"task": task, "variant": meta["variant"], "ep": str(ep), "step": s["step"],
                     "goal": meta["goal"], "cid": s["pre"]["cid"], "history": hist,
                     "orig_slot": s["slot_id"], "orig_action": s["action"], "episode_success": res.get("is_successful"),
                     "evidence": lines, "struct": ui.struct_sig, "t": s["pre"]["t"]})
  rng = random.Random(seed)
  # stratify: failed episodes first, then up to 3 states per task
  by = defaultdict(list)
  for r in rows:
    by[r["task"]].append(r)
  picked = []
  for t, lst in by.items():
    rng.shuffle(lst)
    lst.sort(key=lambda r: r["episode_success"] == 1.0)
    picked += lst[:max(2, n_target // max(1, len(by)))]
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "states.json").write_text(json.dumps(picked, ensure_ascii=False, indent=1))
  print(f"candidates={len(rows)} picked={len(picked)} tasks={len(by)}")
  return picked


def task_graph_lines(task, t_before, cap=40):
  obs = jl(ROOT / "graphs" / "main" / f"{task}.explore.jsonl")
  lines, seen = [], set()
  for o in obs:
    if o["t"] >= t_before or not o.get("selector") or o.get("outcome") == "no_change":
      continue
    key = (o["src"]["struct_sig"], o["slot_id"])
    if key in seen:
      continue
    seen.add(key)
    sel = o["selector"]
    name = sel.get("label") or sel.get("desc") or sel.get("rid") or sel.get("cls")
    lines.append(f"- On {short_act(o['src']['activity'])} [{o['src']['struct_sig'][:6]}]: tapping \"{name[:30]}\" -> "
                 f"{short_act(o['dst']['activity'])} [{o['dst']['struct_sig'][:6]}]")
  return lines[-cap:]


def query(goal, history, img: Image.Image, extra: str):
  hist = "None yet." if not history else "\n".join(f"{i}. {h}" for i, h in enumerate(history[-8:], start=max(1, len(history) - 7)))
  msgs = gelab_agent.build_gelab_messages(goal, hist, img)
  if extra:
    msgs[1]["content"][0]["text"] += "\n\n" + extra
  payload = {"model": "GELAB-ZERO-4B", "messages": msgs, "temperature": 0.0, "top_p": 1.0, "max_tokens": 1024,
             "stream": False, "cache_salt": uuid.uuid4().hex}
  t0 = time.time()
  r = requests.post(API, json=payload, timeout=300, proxies={"http": None, "https": None})
  lat = time.time() - t0
  d = r.json()
  text = d["choices"][0]["message"]["content"]
  usage = d.get("usage", {})
  return text, lat, usage


HEADER = ("Observed outcomes of tapping elements on the current screen (recorded earlier by an automatic, "
          "model-free UI explorer that tapped each element and pressed Back; they describe UI transitions only, "
          "not whether an action is right for the task):")
GHEADER = ("Observed UI transition graph of this app (recorded earlier by an automatic, model-free explorer; "
           "screen ids in brackets; describes transitions only):")


def run():
  states = json.loads((OUT / "states.json").read_text())
  res = []
  done = {(r["ep"], r["step"], r["cond"]) for r in jl(OUT / "responses.jsonl")}
  for s in states:
    img = Image.open(Path(s["ep"]) / "img" / f"{s['cid']}.jpg").convert("RGB")
    ui = A.load_ui(s["ep"], s["cid"])
    conds = {"A": "", "B": HEADER + "\n" + "\n".join(s["evidence"]),
             "C": GHEADER + "\n" + "\n".join(task_graph_lines(s["task"], s["t"]))}
    for c, extra in conds.items():
      if (s["ep"], s["step"], c) in done:
        continue
      text, lat, usage = query(s["goal"], s["history"], img, extra)
      try:
        parsed = gelab_agent.parse_gelab_response(text)
        action, _, extras = gelab_agent.gelab_action_to_json_action(parsed, (1080, 2400))
        ad = action.__dict__
      except Exception as e:  # pylint: disable=broad-except
        parsed, ad, extras = {"parse_error": repr(e)}, {"action_type": "parse_error"}, {}
      import rgs_run as R
      try:
        slot, _ = R.action_slot(ui, ad, extras)
      except Exception:  # pylint: disable=broad-except
        slot = str(ad.get("action_type"))
      row = {"ep": s["ep"], "step": s["step"], "task": s["task"], "cond": c, "slot": slot, "action": ad,
             "summary": parsed.get("summary") if isinstance(parsed, dict) else None,
             "latency_s": round(lat, 3), "prompt_tokens": usage.get("prompt_tokens"),
             "completion_tokens": usage.get("completion_tokens"), "extra_chars": len(extra), "response": text}
      with (OUT / "responses.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
      print(s["task"], s["step"], c, slot[:50], f"{lat:.2f}s", usage.get("prompt_tokens"))


def sheet():
  """Contact sheets for manual annotation: screenshot + goal + history + evidence + A/B/C actions."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  states = json.loads((OUT / "states.json").read_text())
  resp = defaultdict(dict)
  for r in jl(OUT / "responses.jsonl"):
    resp[(r["ep"], r["step"])][r["cond"]] = r
  (OUT / "sheets").mkdir(exist_ok=True)
  for i, s in enumerate(states):
    rr = resp[(s["ep"], s["step"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), gridspec_kw={"width_ratios": [1, 1.6]})
    img = Image.open(Path(s["ep"]) / "img" / f"{s['cid']}.jpg").convert("RGB")
    axes[0].imshow(img)
    for c, col in (("A", "#2a78d6"), ("B", "#eb6834"), ("C", "#1baf7a")):
      a = rr.get(c, {}).get("action", {})
      if a.get("x") is not None:
        axes[0].plot(a["x"] * img.width / 1080, a["y"] * img.height / 2400, "o", ms=14, mfc="none", mec=col, mew=2.5)
        axes[0].text(a["x"] * img.width / 1080 + 20, a["y"] * img.height / 2400, c, color=col, fontsize=12, weight="bold")
    axes[0].axis("off")
    txt = f"[{i}] {s['task']} {s['variant']} step {s['step']} (episode success={s['episode_success']})\n"
    txt += "GOAL: " + s["goal"][:300] + "\n\nHISTORY:\n" + "\n".join(f" {j+1}. {h[:110]}" for j, h in enumerate(s["history"][-5:]))
    txt += "\n\nEVIDENCE (B):\n" + "\n".join(x[:120] for x in s["evidence"])
    txt += f"\n\nORIGINAL: {s['orig_slot'][:60]} {s['orig_action'].get('text') or ''}"
    for c in "ABC":
      r = rr.get(c, {})
      txt += f"\n{c}: {r.get('slot', '')[:60]} {(r.get('action') or {}).get('text') or ''} | {str(r.get('summary'))[:90]}"
    axes[1].text(0, 1, txt, va="top", fontsize=6.6, wrap=True, family="DejaVu Sans")
    axes[1].axis("off")
    fig.savefig(OUT / "sheets" / f"state_{i:02d}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
  {"select": select, "run": run, "sheet": sheet}[sys.argv[1]]()
