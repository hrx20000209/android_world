"""Replay request bodies built by bench_step_latency.py (BENCH_DUMP) against a local vLLM; stream, time each one.
usage: bench_client.py BODY_DIR OUT.jsonl [model_override] [extra_json]"""
import glob
import json
import sys
import time

import requests

API = "http://127.0.0.1:8000/v1/chat/completions"


def main():
  files = sorted(glob.glob(sys.argv[1] + "/*.json"))
  extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
  with open(sys.argv[2], "w") as f:
    for i, fn in enumerate(files):
      body = json.load(open(fn))
      if len(sys.argv) > 3 and sys.argv[3]:
        body["model"] = sys.argv[3]
      body.update(extra)
      t0 = time.time()
      ttft, txt, usage = None, "", {}
      with requests.post(API, json=body, stream=True, timeout=(15, 900)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
          if not line or not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
          js = json.loads(line[6:])
          if js.get("usage"):
            usage = js["usage"]
          for ch in js.get("choices") or []:
            dl = ch.get("delta") or {}
            c = dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning") or ""
            if c and ttft is None:
              ttft = time.time() - t0
            txt += c
      row = {"i": i, "warmup": i == 0, "ttft_s": round(ttft or 0, 3), "total_s": round(time.time() - t0, 3),
             "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
             "think": "<think>" in txt or "<thinking>" in txt, "text": txt[:3000], "extra": extra}
      f.write(json.dumps(row) + "\n")
      f.flush()
      print(i, {k: row[k] for k in ("ttft_s", "total_s", "prompt_tokens", "completion_tokens", "think")}, flush=True)


if __name__ == "__main__":
  main()
