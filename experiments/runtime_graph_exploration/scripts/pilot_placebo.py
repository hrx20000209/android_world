"""Placebo control for the oracle pilot: identical stored screenshot, goal and
history; the evidence block comes from a different state's real probes
(same format and similar length, wrong attachment). Also re-queries M0 from the
stored image to check that the stored input reproduces the live M0."""
import json, sys, glob
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_pilot as P
ROOT = P.ROOT
scores = json.load(open(ROOT / "metrics/pilot_scores.json"))
ok = [r for r in scores if r["status"] == "ok" and r["evidence"]]
data = {}
for r in ok:
  d = ROOT / "raw/pilot" / (f"s{r['i']}_pair_retry" if (ROOT / f"raw/pilot/s{r['i']}_pair_retry/pilot.json").exists() and "resp" in json.load(open(ROOT / f"raw/pilot/s{r['i']}_pair_retry/pilot.json")) else f"s{r['i']}_pair")
  data[r["i"]] = (d, json.load(open(d / "pilot.json")))
ids = sorted(data)
out = []
for j, i in enumerate(ids):
  d, pj = data[i]
  st = P.STATES[i]
  img = Image.open(sorted(glob.glob(str(d / "img" / "*pilot_post*.jpg")))[-1]).convert("RGB")
  rows = [json.loads(l) for l in open(Path(st["ep"]) / "steps.jsonl")]
  hist = [x.get("summary") or "" for x in rows if x.get("kind") == "model"][:st["step_idx"]]
  donor = data[ids[(j + 7) % len(ids)]][1]["evidence"]
  res = {"i": i, "donor": ids[(j + 7) % len(ids)]}
  for c, ev in (("M0img", ""), ("PL", donor)):
    text, lat, usage = P.query(P.APIS["8096"], st["goal"], hist, img, ev)
    parsed, action, _ = P.parse(text, (1080, 2400))
    res[c] = {"response": text, "summary": parsed.get("summary") if isinstance(parsed, dict) else None,
              "action": {k: v for k, v in action.__dict__.items() if v is not None}}
  res["M0live"] = pj["resp"]["M0@8096"]["response"]; res["MElive"] = pj["resp"]["ME@8096"]["response"]
  out.append(res)
  print(i, "donor", res["donor"], "| M0img==M0live", res["M0img"]["response"] == res["M0live"],
        "| PL==M0live", res["PL"]["response"] == res["M0live"], "| PL==MElive", res["PL"]["response"] == res["MElive"])
  print("   PL action", res["PL"]["action"], "|", (res["PL"]["summary"] or "")[:100])
json.dump(out, open(ROOT / "metrics/pilot_placebo.json", "w"), indent=1, ensure_ascii=False)
