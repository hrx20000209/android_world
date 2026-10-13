"""Sample state pairs for manual review of state matching.

(a) struct_merge : same struct_sig, different exact_sig, different variants
                   -> the struct rule MERGES them; reviewer judges "same screen?"
(b) layout_only  : same layout_sig, different struct_sig
                   -> struct rule keeps them APART (flagged ambiguous);
                      reviewer judges whether that split is false.
Sheets: analysis/review/sheet_XX.png (4 pairs each); pairs.json.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "review"


def ep_dir(task, inst_var):
  return ROOT / "raw" / "main" / task / (inst_var if inst_var.startswith("v0_ref") else f"{inst_var}_agent")


def states():
  out = []
  for gx in sorted((ROOT / "graphs" / "main").glob("*.explore.jsonl")):
    task = gx.name.split(".")[0]
    for l in gx.read_text().splitlines():
      if not l.strip():
        continue
      o = json.loads(l)
      var = o["instance"].split("/")[1]
      for side in ("src", "dst"):
        if o.get(side) and o.get(side + "_cid"):
          out.append({"task": task, "var": var, "cid": o[side + "_cid"], **{k: o[side][k] for k in ("exact_sig", "struct_sig", "layout_sig", "activity")}})
  return out


def main(n_each=16, seed=0):
  rng = random.Random(seed)
  st = states()
  by_struct, by_layout = {}, {}
  for s in st:
    by_struct.setdefault((s["task"], s["struct_sig"]), []).append(s)
    by_layout.setdefault((s["task"], s["layout_sig"]), []).append(s)
  a, bpairs = [], []
  for k, lst in by_struct.items():
    cand = [(x, y) for i, x in enumerate(lst) for y in lst[i + 1:] if x["var"] != y["var"] and x["exact_sig"] != y["exact_sig"]]
    if cand:
      a.append(rng.choice(cand))
  for k, lst in by_layout.items():
    cand = [(x, y) for i, x in enumerate(lst) for y in lst[i + 1:] if x["struct_sig"] != y["struct_sig"]]
    if cand:
      bpairs.append(rng.choice(cand))
  rng.shuffle(a)
  rng.shuffle(bpairs)
  pairs = [{"kind": "struct_merge", "a": x, "b": y} for x, y in a[:n_each]] + \
          [{"kind": "layout_only", "a": x, "b": y} for x, y in bpairs[:n_each]]
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "pairs.json").write_text(json.dumps(pairs, indent=1))
  print(f"struct_merge candidates={len(a)} layout_only candidates={len(bpairs)} sampled={len(pairs)}")
  for si in range(0, len(pairs), 4):
    chunk = pairs[si:si + 4]
    fig, axes = plt.subplots(1, 2 * len(chunk), figsize=(3.0 * 2 * len(chunk), 6.2))
    for j, p in enumerate(chunk):
      for k, side in enumerate(("a", "b")):
        s = p[side]
        ax = axes[2 * j + k]
        try:
          im = Image.open(ep_dir(s["task"], s["var"]) / "img" / f"{s['cid']}.jpg")
          ax.imshow(im)
        except FileNotFoundError:
          pass
        ax.set_title(f"#{si + j} {p['kind']} {side}\n{s['task'][:18]} {s['var']}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"sheet_{si // 4:02d}.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
  main()
