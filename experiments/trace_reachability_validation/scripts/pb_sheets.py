"""Contact sheets (4 states per image) for manual label verification."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
S = json.loads((C.ROOT / "decision_states" / "pilot_b_states.json").read_text())
out = C.ROOT / "decision_states" / "sheets"; out.mkdir(exist_ok=True)
for b in range(0, len(S), 4):
  fig, axes = plt.subplots(1, 8, figsize=(30, 8), gridspec_kw={"width_ratios": [1, 1.3] * 4})
  for j, s in enumerate(S[b:b + 4]):
    im = Image.open(Path(s["ep"]) / "img" / f"{s['cid']}.jpg")
    ax = axes[2 * j]; ax.imshow(im); ax.axis("off")
    a = s["orig_action"]
    if a.get("x") is not None:
      ax.plot(a["x"] * im.width / 1080, a["y"] * im.height / 2400, "o", ms=16, mfc="none", mec="red", mew=3)
    t = (f"#{b+j} {s['task']} {s['variant']} s{s['step']} succ={s['episode_success']} err={s['orig_error']}\n"
         f"GOAL: {s['goal'][:160]}\n\nHIST: " + " | ".join(h[:60] for h in s['history'][-3:]) +
         f"\n\nORIG(red): {s['orig_tok'][:70]}\nLABEL: " + "\n  ".join(x[:70] for x in s['label_tokens']))
    axes[2 * j + 1].text(0, 1, t, va="top", fontsize=8, wrap=True); axes[2 * j + 1].axis("off")
  fig.savefig(out / f"sheet_{b // 4:02d}.png", dpi=55, bbox_inches="tight"); plt.close(fig)
print(len(S))
