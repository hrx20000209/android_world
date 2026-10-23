"""Replay real decisions of a finished run against the Orin vLLM (streamed, temperature 0) and record the output
text and timing; run once per server configuration, then compare: identical outputs, time to first token, decode
time, tokens/s. usage: specdec_bench.py LABEL N"""
import glob, json, random, sys, time
from pathlib import Path
import requests
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from android_world.agents import gelab_agent  # noqa: E402
API, MODEL = "http://127.0.0.1:18097/v1/chat/completions", "GELab-Zero-4B-preview"


def main(label, n=30, seed=0):
  random.seed(seed)
  samples = []
  for f in sorted(glob.glob("raw/e2e116/BASE/*/p0/steps.jsonl")):
    d = Path(f).parent
    rows = [json.loads(l) for l in open(f)]
    for i, r in enumerate(rows):
      if r.get("kind") == "model" and r.get("pre"):
        samples.append((d, rows, i))
  random.shuffle(samples)
  out = []
  for d, rows, i in samples[:n]:
    r = rows[i]
    img = d / "img" / f"{r['pre']['cid']}.jpg"
    goal = json.load(open(d / "meta.json"))["goal"]
    hist = [x.get("summary") for x in rows[:i] if x.get("summary")]
    history = "\n".join(f"{k + 1}. {s}" for k, s in enumerate(hist[-8:])) or "None yet."
    msgs = gelab_agent.build_gelab_messages(goal, history, Image.open(img).convert("RGB"))
    body = {"model": MODEL, "messages": msgs, "temperature": 0.0, "top_p": 1.0, "max_tokens": 1024, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.time(); txt = ""; ttft = None; ntok = None
    with requests.post(API, json=body, stream=True, timeout=600) as resp:
      for line in resp.iter_lines():
        if not line or not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
          continue
        js = json.loads(line[6:])
        if js.get("usage"):
          ntok = js["usage"].get("completion_tokens")
        if js.get("choices"):
          ch = (js["choices"][0].get("delta") or {}).get("content") or ""
          if ch and ttft is None:
            ttft = time.time() - t0
          txt += ch
    t = time.time() - t0
    out.append({"task": d.parts[-2], "step": r["step"], "text": txt, "ttft": ttft, "t": t, "tokens": ntok})
    print(json.dumps({k: out[-1][k] for k in ("task", "step", "ttft", "t", "tokens")}), flush=True)
  Path(f"reports/specdec_{label}.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out))


if __name__ == "__main__":
  main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30)
