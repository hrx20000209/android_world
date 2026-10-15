"""Score the oracle pilot (pair inference)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
rows = []
for i in range(20):
  d = None
  for name in (f"s{i}_pair_retry", f"s{i}_pair"):
    p = ROOT / "raw/pilot" / name / "pilot.json"
    if p.exists():
      x = json.loads(p.read_text())
      if "resp" in x:
        d = x; d["dir"] = name; break
      d = d or dict(x, dir=name)
  if d is None:
    continue
  r = d.get("resp", {})
  def g(k, f): return r[k][f] if k in r else None
  rows.append({"i": i, "task": d["task"], "kind": d["kind"], "match": d["replay_match"], "status": d.get("status", "ok"),
               "probes_ok": sum(p.get("status") == "ok" for p in d["probes"]), "evidence": bool(d.get("evidence")),
               "retry": d["dir"].endswith("retry"),
               **{f"{c}@{e}": (g(f"{c}@{e}", "tok") or "")[:46] for c in ("M0", "ME") for e in ("8095", "8096")},
               **{f"ok_{c}@{e}": g(f"{c}@{e}", "correct") for c in ("M0", "ME") for e in ("8095", "8096")},
               "tok_extra": (g("ME@8095", "prompt_tokens") or 0) - (g("M0@8095", "prompt_tokens") or 0)})
json.dump(rows, open(ROOT / "metrics/pilot_scores.json", "w"), indent=1, ensure_ascii=False)
v = [r for r in rows if r["status"] == "ok" and r["evidence"]]
print("states", len(rows), "scorable(with evidence, restored)", len(v))
for e in ("8095", "8096"):
  m0 = sum(bool(r[f"ok_M0@{e}"]) for r in v); me = sum(bool(r[f"ok_ME@{e}"]) for r in v)
  fixed = sum((not r[f"ok_M0@{e}"]) and r[f"ok_ME@{e}"] for r in v); broken = sum(r[f"ok_M0@{e}"] and not r[f"ok_ME@{e}"] for r in v)
  changed = sum(r[f"M0@{e}"] != r[f"ME@{e}"] for r in v)
  print(f"endpoint {e}: M0 correct {m0}/{len(v)}  ME correct {me}/{len(v)}  changed {changed} fixed {fixed} broken {broken}")
print("endpoint agreement M0", sum(r["M0@8095"] == r["M0@8096"] for r in v), "ME", sum(r["ME@8095"] == r["ME@8096"] for r in v), "of", len(v))
for k in ("F", "C"):
  s = [r for r in v if r["kind"] == k]
  print(k, len(s), "M0 ok", sum(bool(r["ok_M0@8095"]) for r in s), "ME ok", sum(bool(r["ok_ME@8095"]) for r in s))
for r in rows:
  print(f"{r['i']:2d} {r['kind']} {r['task'][:26]:26s} {r['match']:8s} {r['status']:14s} p{r['probes_ok']} | M0 {r['M0@8095'][:34]:34s} {str(r['ok_M0@8095'])[0]} | ME {r['ME@8095'][:34]:34s} {str(r['ok_ME@8095'])[0]} | +{r['tok_extra']}tok")
