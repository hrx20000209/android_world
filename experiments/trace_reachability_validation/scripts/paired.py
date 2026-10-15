"""Paired inference-call comparison of an arm against BASE / FULL8 on tasks both solved, plus where graph actions
replaced inference (bootstrap, functions, confirm, jumps)."""
import json, glob, os, re, sys
from collections import Counter
arm = sys.argv[1]
ref = {}
for line in open("../../results/4B_2.txt"):
  m = re.match(r"^(\w+)\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+\S+\s+([\d.]+)", line)
  if m and float(m.group(5)) == 0:
    ref[m.group(1)] = {"succ": float(m.group(3)) == 1.0, "inf": float(m.group(4))}
def load(a):
  out = {}
  for f in glob.glob(f"raw/full/{a}/*/p0/result.json"):
    r = json.load(open(f)); t = f.split("/")[3]
    if t in ref:
      out[t] = {"succ": r.get("is_successful") == 1.0, "inf": r.get("model_calls") or 0, "r": r}
  return out
A = load(arm)
for name, B in (("BASE", load("BASE")), ("FULL8", load("FULL8")), ("4B_2", ref)):
  both = [t for t in A if t in B and A[t]["succ"] and B[t]["succ"]]
  if not both: continue
  fa = sum(A[t]["inf"] for t in both) / len(both); fb = sum(B[t]["inf"] for t in both) / len(both)
  faster = sum(A[t]["inf"] < B[t]["inf"] for t in both); slower = sum(A[t]["inf"] > B[t]["inf"] for t in both)
  only_a = sum(1 for t in A if t in B and A[t]["succ"] and not B[t]["succ"]); only_b = sum(1 for t in A if t in B and B[t]["succ"] and not A[t]["succ"])
  print(f"{arm} vs {name}: both-success {len(both)}: inference {fb:.2f} -> {fa:.2f} ({faster} faster / {slower} slower) | wins {only_a} losses {only_b}")
c = Counter()
for t, x in A.items():
  r = x["r"]
  for k in ("m_bootstrap", "m_func_apps", "m_func_steps", "m_confirm", "m_jump_steps", "m_requery", "m_unstick_back", "m_onboard"):
    c[k] += r.get(k, 0) or 0
print("graph actions replacing inference:", dict(c))
