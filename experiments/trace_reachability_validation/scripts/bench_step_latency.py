"""One-step inference latency of other agent models on the Orin vLLM, with the real agents' prompts and inputs.

  mai   MAI-UI agent (android_world/agents/mai_ui.py): build_mai_messages(goal, history, screenshot) with the screenshot
        resized exactly as the agent does (fetch_resized_image, scale 1), temperature 0, max_tokens 512
  t3a   T3A text agent (android_world/agents/t3a_mobicom.py, the earlier Qwen3-4B text-agent setting): the action
        selection prompt over the a11y element list (_generate_ui_elements_description_list_full) with the
        LlamaCppTextWrapper system message, temperature 0, max_tokens 1024; one LLM call per step (local summary)

Inputs are real steps of e2e116B BASE episodes (screenshot, a11y tree, goal, history of the previous steps' summaries).
Each request is streamed: time to first token, time to the end of the response, completion tokens.

usage: bench_step_latency.py mai|t3a MODEL_NAME OUT.jsonl [n_steps] [extra_json_body]
"""
import glob
import gzip
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
import ast  # noqa: E402
import types  # noqa: E402


def _load_mai():
  """The MAI-UI agent's own prompt constant and message/image builders, taken verbatim from mai_ui.py (importing the
  module pulls in qwen_agent, which building the prompt does not need)."""
  src = open(ROOT.parents[1] / "android_world" / "agents" / "mai_ui.py").read()
  keep = {"MAI_MOBILE_SYS_PROMPT", "pil_to_base64", "image_to_data_url", "_normalize_downsample_scale",
          "fetch_resized_image", "build_mai_messages"}
  # every module-level constant (the system prompt is formatted from others) and the builder functions
  body = [n for n in ast.parse(src).body if (isinstance(n, ast.FunctionDef) and n.name in keep) or
          isinstance(n, (ast.Assign, ast.AnnAssign))]
  ns = {}
  exec("import base64, copy, io, json, math, os, re, time\nfrom io import BytesIO\nfrom typing import Any\nfrom PIL import Image\n"  # noqa: S102
       "from android_world.agents.coordinate_resize import update_image_size_\n", ns)
  exec(compile(ast.Module(body=body, type_ignores=[]), "mai_ui.py", "exec"), ns)  # noqa: S102
  missing = keep - set(ns)
  assert not missing, missing
  return types.SimpleNamespace(**{k: ns[k] for k in keep})


mai_ui = _load_mai()
from android_world.agents import t3a_mobicom  # noqa: E402
from android_world.env import representation_utils as RU  # noqa: E402

API = os.environ.get("BENCH_API", "http://127.0.0.1:18097/v1/chat/completions")
SRC = ROOT / "raw" / "e2e116B" / "BASE"


def samples(n):
  """Steps 1 and 3 of the first n/2 BASE episodes in task order (a mid-task step with history)."""
  order = [t for x in open(ROOT / "reports" / "e2e116n_task_order.txt").read().split() for t in x.split(",") if t]
  out = []
  for t in order:
    d = SRC / t / "p0"
    if not (d / "steps.jsonl").exists():
      continue
    goal = json.load(open(d / "meta.json"))["goal"]
    steps = [json.loads(l) for l in open(d / "steps.jsonl")]
    for k in (1, 3):
      if k >= len(steps):
        continue
      cid = (steps[k].get("pre") or {}).get("cid")
      img = d / "img" / f"{cid}.jpg"
      a11y = d / "a11y" / f"{cid}.json.gz"
      if not cid or not img.exists() or not a11y.exists():
        continue
      hist = [s.get("summary") or "" for s in steps[:k]]
      out.append({"task": t, "step": k, "goal": goal, "hist": hist, "img": img, "a11y": a11y})
    if len(out) >= n:
      break
  return out[:n]


def mai_messages(s, tmp):
  png = os.path.join(tmp, "screenshot_0.png")
  Image.open(s["img"]).convert("RGB").save(png)          # the agent saves the raw screenshot as PNG
  resized, _, _, _ = mai_ui.fetch_resized_image(png, 1)
  rp = os.path.join(tmp, "screenshot_resized_0.png")
  resized.save(rp)
  hist = "\n".join(f"{i}. {h}" for i, h in enumerate(s["hist"], start=1)) or "None yet."
  return mai_ui.build_mai_messages(s["goal"], hist, rp), resized.size


def t3a_messages(s):
  nodes = json.load(gzip.open(s["a11y"]))["payload"]["nodes"]
  els = []
  for n in nodes:
    b = n.get("bounds") or [0, 0, 0, 0]
    els.append(RU.UIElement(
        text=n.get("text"), content_description=n.get("contentDescription") or n.get("desc"),
        class_name=n.get("class"), bbox_pixels=RU.BoundingBox(x_min=b[0], y_min=b[1], x_max=b[2], y_max=b[3]),
        hint_text=n.get("hintText"), is_checked=n.get("checked"), is_checkable=n.get("checkable"),
        is_clickable=n.get("clickable"), is_editable=n.get("editable"), is_enabled=n.get("enabled"),
        is_focused=n.get("focused"), is_focusable=n.get("focusable"), is_long_clickable=n.get("longClickable"),
        is_scrollable=n.get("scrollable"), is_selected=n.get("selected"), is_visible=n.get("visible"),
        package_name=n.get("package"), resource_name=n.get("resourceId")))
  desc = t3a_mobicom._generate_ui_elements_description_list_full(els, (1080, 2400))  # pylint: disable=protected-access
  prompt = t3a_mobicom._action_selection_prompt(  # pylint: disable=protected-access
      s["goal"], [f"Step {i + 1}: {h}" for i, h in enumerate(s["hist"])], desc, None)
  return [{"role": "system", "content": "Do not output chain-of-thought or reasoning. Only give the final answer."},
          {"role": "user", "content": [{"type": "text", "text": prompt}]}], len(els)


# ------------------------------------------------------------------ AutoDroid-style compression (MobiCom'24)
def autodroid_html(nodes, screen=(1080, 2400)):
  """Simplified HTML of the screen in the style of AutoDroid: only elements a user can act on or read, one line each
  (<button>, <checkbox>, <input>, <scroller>, <p>), system bars and pure layout containers dropped; the text of
  non-interactive descendants is merged into the nearest interactive ancestor; plus a <button>go back</button>."""
  stack, items, owner = [], [], {}
  for n in nodes:
    d = n.get("depth", 0)
    while stack and stack[-1][0] >= d:
      stack.pop()
    parent = stack[-1][1] if stack else None
    stack.append((d, n["id"]))
    owner[n["id"]] = owner.get(parent) if parent is not None else None
    if (n.get("package") or "").startswith("com.android.systemui") or not n.get("visible", True):
      continue
    b = n.get("bounds") or [0, 0, 0, 0]
    if b[2] <= b[0] or b[3] <= b[1] or b[3] <= 0 or b[1] >= screen[1]:
      continue
    txt = " ".join(x for x in (n.get("text"), n.get("contentDescription")) if x).strip()[:60]
    hint = (n.get("hint") or "").strip()[:40]
    rid = (n.get("resourceId") or "").split("/")[-1]
    if n.get("editable"):
      tag = "input"
    elif n.get("checkable"):
      tag = "checkbox"
    elif n.get("clickable") or n.get("longClickable"):
      tag = "button"
    elif n.get("scrollable"):
      tag = "scroller"
    elif txt:
      tag = "p"
    else:
      continue
    if tag == "p" and owner[n["id"]] is not None:
      it = items[owner[n["id"]]]                  # text inside a clickable container belongs to it
      if txt and txt not in it["txt"]:
        it["txt"] = (it["txt"] + " " + txt).strip()[:80]
      continue
    it = {"tag": tag, "txt": txt, "hint": hint, "rid": rid, "checked": n.get("checked")}
    items.append(it)
    if tag in ("button", "checkbox"):
      owner[n["id"]] = len(items) - 1
  lines = []
  for i, it in enumerate(items):
    attrs = f" id={i}"
    if it["tag"] == "checkbox":
      attrs += f" checked={bool(it['checked'])}"
    if it["tag"] == "input" and it["hint"] and it["hint"] != it["txt"]:
      attrs += f" hint='{it['hint']}'"
    label = it["txt"] or (it["rid"].replace("_", " ") if it["rid"] else "")
    if not label and it["tag"] == "button":
      continue                                       # an unlabelled clickable gives the model nothing to choose by
    lines.append(f"<{it['tag']}{attrs}>{label}</{it['tag']}>")
  lines.append(f"<button id={len(items)}>go back</button>")
  return "\n".join(lines)


AD_ACTIONS = ("tap, input, scroll_up, scroll_down, open_app (input_text = app name), "
              "answer (input_text = the answer), done (task finished), infeasible")

AD_FULL = """You are a smartphone assistant to help users complete tasks by interacting with mobile apps. Given a task, the previous UI actions, and the content of current UI state, your job is to decide whether the task is already finished by the previous actions, and if not, decide which UI element in current UI state should be interacted.
Task: {goal}
Previous UI actions:
{history}
Current UI State:
{state}
Your answer should always use the following format: {{"Steps": "...<steps usually involved to complete the above task on a smartphone>", "Analyses": "...<Analyses of the relations between the task and the previous UI actions and current UI state>", "Finished": "Yes/No", "Next step": "None or a <high level description of the next step>", "id": "an integer or -1 (if the task has been completed by previous UI actions)", "action": "one of: {actions}", "input_text": "N/A or ...<input text>"}}
**Note that the id is the id number of the UI element to interact with. If you think the task has been completed by previous UI actions, the id should be -1. If 'Finished' is 'Yes', then the 'description' of 'Next step' is 'None', otherwise it is a high level description of the next step. If the 'action' is 'tap', the 'input_text' is N/A, otherwise it is the '<input text>'. Please do not output any content other than the JSON format. **"""

AD_MIN = """You are a smartphone assistant to help users complete tasks by interacting with mobile apps. Given a task, the previous UI actions, and the content of current UI state, decide the next UI action.
Task: {goal}
Previous UI actions:
{history}
Current UI State:
{state}
Answer with one line of JSON only: {{"id": <element id, or -1>, "action": "<one of: {actions}>", "input_text": "<text or N/A>"}}"""


def compressed_messages(s, variant):
  nodes = json.load(gzip.open(s["a11y"]))["payload"]["nodes"]
  html = autodroid_html(nodes)
  hist = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(s["hist"])) or "None"
  if variant == "t3a":
    prompt = t3a_mobicom._action_selection_prompt(  # pylint: disable=protected-access
        s["goal"], [f"Step {i + 1}: {h}" for i, h in enumerate(s["hist"])], html, None)
    sys_msg = "Do not output chain-of-thought or reasoning. Only give the final answer."
  else:
    prompt = (AD_FULL if variant == "ad_full" else AD_MIN).format(goal=s["goal"], history=hist, state=html, actions=AD_ACTIONS)
    sys_msg = None
  msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + [{"role": "user", "content": prompt}]
  return msgs, html.count("\n") + 1


def call(model, messages, max_tokens, extra):
  body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
          "stream_options": {"include_usage": True}}
  body.update(extra)
  if os.environ.get("BENCH_DUMP"):
    # build only: the request bodies are replayed on the server host (bench_client.py), out of reach of the ssh
    # tunnel that dropped multi-MB screenshot uploads
    d = Path(os.environ["BENCH_DUMP"])
    d.mkdir(parents=True, exist_ok=True)
    i = len(list(d.glob("*.json")))
    (d / f"{i:03d}.json").write_text(json.dumps(body))
    return {"ttft_s": None, "total_s": None, "prompt_tokens": None, "completion_tokens": None, "think": None, "text": ""}
  t0 = time.time()
  ttft, txt, usage = None, "", {}
  with requests.post(API, json=body, stream=True, timeout=(15, 600)) as r:
    r.raise_for_status()
    for line in r.iter_lines():
      if not line or not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
        continue
      js = json.loads(line[6:])
      if js.get("usage"):
        usage = js["usage"]
      for ch in js.get("choices") or []:
        c = (ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("reasoning_content") or ""
        if c and ttft is None:
          ttft = time.time() - t0
        txt += c
  return {"ttft_s": round(ttft or 0, 3), "total_s": round(time.time() - t0, 3), "prompt_tokens": usage.get("prompt_tokens"),
          "completion_tokens": usage.get("completion_tokens"), "think": "<think>" in txt or "<thinking>" in txt,
          "text": txt[:4000]}


def main():
  kind, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
  n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
  extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
  S = samples(n + 1)
  with tempfile.TemporaryDirectory() as tmp, open(out, "w") as f:
    for i, s in enumerate(S):
      if kind == "mai":
        msgs, info = mai_messages(s, tmp)
        r = call(model, msgs, 512, extra)
      elif kind == "t3a":
        msgs, info = t3a_messages(s)
        r = call(model, msgs, 1024, extra)
      else:                                   # t3a_c / ad_full / ad_min: AutoDroid-compressed UI
        msgs, info = compressed_messages(s, {"t3a_c": "t3a"}.get(kind, kind))
        r = call(model, msgs, 1024, extra)
      r.update(task=s["task"], step=s["step"], warmup=i == 0, input_info=info)
      f.write(json.dumps(r) + "\n")
      f.flush()
      print(i, s["task"], s["step"], {k: r[k] for k in ("ttft_s", "total_s", "prompt_tokens", "completion_tokens", "think")}, flush=True)


if __name__ == "__main__":
  main()
