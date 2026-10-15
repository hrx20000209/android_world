"""Reversibility prior for exploration: P(Back undoes a probe) from pre-probe features of the control, estimated
from earlier runs' probes (cross-app UI mechanics, not task content). Writes graph/rev_prior.json."""
import json, glob, sys, re
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
import full9
R, U, C = full9.R, full9.U, full9.C


def features(ui, sel):
  b = tuple(sel.get("bounds") or ())
  n = next((x for x in ui.app_nodes() if tuple(x.bounds) == b), None)
  if n is None:
    return None
  par = ui.nodes[n.parent] if n.parent is not None else None
  sibs = [ui.nodes[i] for i in par.children] if par else []
  while par is not None and len(sibs) < 2 and par.parent is not None:
    par = ui.nodes[par.parent]; sibs = [ui.nodes[i] for i in par.children]
  same = [s for s in sibs if s.short_cls == n.short_cls]
  selected = any(s.selected or s.checked for s in sibs) or any(ui.nodes[c].selected for s in sibs for c in s.children)
  H = max(1, ui.screen[1])
  y = (n.bounds[1] + n.bounds[3]) / 2 / H
  zone = "top" if y < 0.12 else "bottom" if y > 0.88 else "mid"
  role = "set_sel" if len(same) >= 3 and selected else "set" if len(same) >= 3 else "single"
  return {"role": role, "zone": zone, "cls": n.short_cls}


def keys(f):
  return [("role", f["role"]), ("role_zone", f["role"] + "/" + f["zone"]), ("cls", f["cls"])]


if __name__ == "__main__":
  stat = defaultdict(Counter)
  for arm in sys.argv[1:] or ["FULL11", "FULL12", "FULL13", "FULL14"]:
    for f in glob.glob(f"raw/full/{arm}/*/p0/probes.jsonl"):
      d = Path(f).parent
      for l in open(f):
        o = json.loads(l)
        if o.get("depth", 1) != 1:
          continue
        try:
          ui = R.load_ui(d, o["src_cid"])
        except Exception:
          continue
        ft = features(ui, o["selector"])
        if ft is None:
          continue
        for k in keys(ft):
          stat["|".join(k)]["n"] += 1
          stat["|".join(k)]["ok"] += bool(o["rollback_ok"])
  out = {k: {"n": v["n"], "ok": v["ok"]} for k, v in stat.items()}
  Path(full9.HERE.parent.parent / "runtime_graph_exploration" / "graph" / "rev_prior.json").write_text(json.dumps(out, indent=1))
  for k, v in sorted(out.items(), key=lambda kv: -kv[1]["n"]):
    if v["n"] >= 10:
      print(f"{k:28s} n={v['n']:4d} P(rev)={v['ok']/v['n']:.2f}")
