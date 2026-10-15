"""Compare an arm with results/4B_2.txt on the tasks the arm has completed so far.
4B_2 tasks with an errored trial (num_fail_trials > 0) are excluded from both sides."""
import json, glob, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
arm = sys.argv[1] if len(sys.argv) > 1 else "FULL8"
ref = {}
for line in open(ROOT / "results" / "4B_2.txt"):
  m = re.match(r"^(\w+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+\S+\s+([\d.]+)", line)
  if m:
    ref[m.group(1)] = {"succ": float(m.group(3)) == 1.0, "steps": float(m.group(4)), "fail_trials": float(m.group(5))}
def stats(rows, key="steps"):
  n = len(rows); s = [r for r in rows if r["succ"]]
  return f"{len(s)}/{n} ({len(s)/n:.1%}) | mean steps {sum(r[key] for r in rows)/n:.2f} | mean success steps " + \
         (f"{sum(r[key] for r in s)/len(s):.2f}" if s else "-")
out = []
for ps in ("p0", "p1"):
  mine = []
  for p in glob.glob(f"raw/full/{arm}/*/{ps}/result.json"):
    t = p.split("/")[3]; r = json.load(open(p))
    if t in ref and ref[t]["fail_trials"] == 0:
      mine.append((t, {"succ": r.get("is_successful") == 1.0, "steps": r.get("steps") or 0,
                       "calls": r.get("model_calls") or 0}))
  if not mine:
    continue
  A = [r for _, r in mine]; B = [ref[t] for t, _ in mine]
  both = [(a, ref[t]) for t, a in mine if a["succ"] and ref[t]["succ"]]
  out.append(f"[{arm} {ps}] n={len(mine)} | {arm}: {stats(A)} | 4B_2: {stats(B)} | both-success {len(both)}: "
             f"steps {sum(b['steps'] for _, b in both)/max(1,len(both)):.2f} -> {sum(a['steps'] for a, _ in both)/max(1,len(both)):.2f}, "
             f"{arm} calls {sum(a['calls'] for a, _ in both)/max(1,len(both)):.2f}")
  # inference-step accounting: every model call (incl. requeries) is one step; 4B_2 steps are model calls
  for a in A: a["inf"] = a["calls"]
  out.append(f"[{arm} {ps} INFERENCE] {arm}: {stats(A, 'inf')} | 4B_2: {stats(B)}")
  base = {}
  for t, _ in mine:
    bp = Path(f"raw/full/BASE/{t}/{ps}/result.json")
    if bp.exists():
      r = json.load(open(bp)); base[t] = {"succ": r.get("is_successful") == 1.0, "inf": r.get("model_calls") or r.get("steps") or 0}
  if base:
    A2 = [a for t, a in mine if t in base]; Bb = [base[t] for t, _ in mine if t in base]
    out.append(f"[{arm} {ps} vs BASE] n={len(A2)} | {arm}: {stats(A2, 'inf')} | BASE: {stats(Bb, 'inf')}")
print("\n".join(out))
