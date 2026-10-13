"""Shared loaders over the previous study's raw evidence (read-only)."""

from __future__ import annotations

import gzip
import json
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
PREV = ROOT.parent / "runtime_graph_study"
REPO = ROOT.parents[1]
sys.path.insert(0, str(PREV / "scripts"))
sys.path.insert(0, str(REPO))
import rgs_ui as U  # noqa: E402

APP_OF = {}
for m in ("batch1_manifest.json", "batch2_manifest.json"):
  for t in json.loads((PREV / "configs" / m).read_text())["tasks"]:
    APP_OF[t["task"]] = t["app"]


def jl(p: Path):
  p = Path(p)
  return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


@lru_cache(maxsize=8192)
def load_ui(ep: str, cid: str) -> U.UIState:
  with gzip.open(Path(ep) / "a11y" / f"{cid}.json.gz", "rt", encoding="utf-8") as f:
    d = json.load(f)
  return U.parse_flat(d["payload"], activity=d["activity"])


def family_of_sel(sel: dict) -> str:
  """Action-slot identity used by the trace index.

  List rows with a resource-id collapse to their row family (their text is
  runtime data); everything else keeps its full selector identity. Raw
  coordinates are never part of the identity."""
  if not sel:
    return "?"
  if sel.get("in_list") and sel.get("rid"):
    return f"L|{sel['cls']}|{sel['rid']}"
  return f"E|{sel['cls']}|{sel['rid']}|{sel.get('desc', '')}|{sel.get('label', '')}"


def action_token(ui: U.UIState, action: dict) -> tuple[str, str | None, dict | None]:
  """(token, family-or-None, selector) for an executed agent action."""
  at = action.get("action_type")
  if at in ("click", "long_press", "double_tap") and action.get("x") is not None:
    n = U.element_at(ui, int(action["x"]), int(action["y"]))
    if n is None:
      return f"{at.upper()}:none", None, None
    sel = U.selector(ui, n)
    fam = family_of_sel(sel)
    return ("" if at == "click" else at.upper() + ":") + fam, (fam if at == "click" else None), sel
  if at == "input_text":
    if action.get("x") is not None:
      n = U.element_at(ui, int(action["x"]), int(action["y"]))
      rid = n.short_rid if n is not None else ""
    else:
      rid = ""
    return f"TYPE:{rid}", None, None
  if at in ("scroll", "swipe"):
    return f"SWIPE:{action.get('direction') or ''}", None, None
  if at == "open_app":
    return f"OPEN_APP:{action.get('app_name')}", None, None
  return str(at).upper(), None, None


def slot_family(ui: U.UIState, slot: U.Slot) -> str:
  return family_of_sel(slot.sel)


def episodes(include_recovery: bool = False):
  """All agent-driven episodes: clean refs, explore arms, no-explore arms."""
  out = []
  for d in sorted((PREV / "raw" / "main").glob("*/*")):
    if not (d / "result.json").exists() or not (d / "steps.jsonl").exists():
      continue
    meta = json.loads((d / "meta.json").read_text())
    if meta["mode"] != "agent":
      continue
    v = meta["variant"]
    kind = "ref" if "ref" in v else ("noexp" if "noexp" in v else "explore")
    res = json.loads((d / "result.json").read_text())
    out.append({"ep": str(d), "task": meta["task"], "app": APP_OF.get(meta["task"], meta["task"]),
                "variant": v, "kind": kind, "started": meta["started"], "seed": meta["seed"],
                "goal": meta["goal"], "success": res.get("is_successful")})
  out.sort(key=lambda e: e["started"])
  return out


def graph_observations():
  """All explore/agent transitions from the previous study, with summaries
  recomputed from raw a11y (current definitions) and a family id."""
  obs = []
  for gx in sorted((PREV / "graphs" / "main").glob("*.explore.jsonl")):
    task = gx.name.split(".")[0]
    for o in jl(gx):
      var = o["instance"].split("/")[1]
      ep = PREV / "raw" / "main" / task / (var if var.startswith("v0_ref") else f"{var}_agent")
      try:
        ui0 = load_ui(str(ep), o["src_cid"])
        o["src"] = ui0.summary()
        if o.get("dst_cid"):
          o["dst"] = load_ui(str(ep), o["dst_cid"]).summary()
      except Exception:  # pylint: disable=broad-except
        continue
      if o["origin"] == "agent" and o.get("action"):
        tok, fam, sel = action_token(ui0, o["action"])
        o["family"], o["selector"] = fam, sel
      else:
        o["family"] = family_of_sel(o.get("selector") or {})
      o["task"], o["ep"] = task, str(ep)
      obs.append(o)
  obs.sort(key=lambda o: o["t"])
  return obs
