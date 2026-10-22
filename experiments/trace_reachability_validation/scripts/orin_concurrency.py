"""Orin vLLM: per-request latency (time to first token, decode speed, total) at concurrency 1/2/3 with real GELab
prompts (distinct screenshots), to see whether batch slack can host speculative next decisions."""
import glob, json, random, statistics as st, sys, threading, time
from pathlib import Path
import requests
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from android_world.agents import gelab_agent  # noqa: E402
API, MODEL = "http://127.0.0.1:18097/v1/chat/completions", "GELab-Zero-4B-preview"


def prompts(n, seed=0):
  random.seed(seed)
  out = []
  fs = sorted(glob.glob("raw/e2e40/BASE/*/p0/steps.jsonl"))
  random.shuffle(fs)
  for f in fs:
    d = Path(f).parent
    rows = [json.loads(l) for l in open(f)]
    goal = json.load(open(d / "meta.json"))["goal"]
    for i, r in enumerate(rows):
      if r.get("kind") == "model" and r.get("pre") and (d / "img" / f"{r['pre']['cid']}.jpg").exists():
        hist = [x.get("summary") for x in rows[:i] if x.get("summary")]
        h = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(hist[-8:])) or "None yet."
        out.append(gelab_agent.build_gelab_messages(goal, h, Image.open(d / "img" / f"{r['pre']['cid']}.jpg").convert("RGB")))
        break
    if len(out) >= n:
      return out
  return out


def one(msgs, res):
  t0 = time.time(); ttft = None; ntok = 0
  body = {"model": MODEL, "messages": msgs, "temperature": 0.0, "max_tokens": 160, "min_tokens": 160, "stream": True,
          "stream_options": {"include_usage": True}}
  with requests.post(API, json=body, stream=True, timeout=600) as r:
    for line in r.iter_lines():
      if not line or not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
        continue
      js = json.loads(line[6:])
      if js.get("usage"):
        ntok = js["usage"].get("completion_tokens") or ntok
      if js.get("choices") and (js["choices"][0].get("delta") or {}).get("content") and ttft is None:
        ttft = time.time() - t0
  tot = time.time() - t0
  res.append({"ttft": ttft, "total": tot, "tok": ntok, "dec": (ntok - 1) / (tot - ttft) if ttft and tot > ttft else None})


def main():
  P = prompts(12)
  for conc in (1, 2, 3):
    R = []
    for rep in range(3):
      batch = P[(rep * conc) % len(P):(rep * conc) % len(P) + conc]
      th = [threading.Thread(target=one, args=(m, R)) for m in batch]
      [t.start() for t in th]; [t.join() for t in th]
    print(f"concurrency {conc}: per request ttft {st.mean(x['ttft'] for x in R):.2f}s, decode {st.mean(x['dec'] for x in R if x['dec']):.1f} tok/s, "
          f"total {st.mean(x['total'] for x in R):.1f}s for {st.mean(x['tok'] for x in R):.0f} tokens; "
          f"aggregate throughput {sum(x['tok'] for x in R) / sum(max(x['total'] for x in R[i:i+conc]) for i in range(0, len(R), conc)):.1f} tok/s", flush=True)


if __name__ == "__main__":
  main()
