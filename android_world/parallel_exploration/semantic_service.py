"""A persistent 22M-parameter encoder the explorer can query in ~50 ms.

Why a service rather than an import: the explorer is a fresh interpreter per
step (`spawn_explorer`, multiprocessing "spawn"), and loading MiniLM costs
7.4 s. Spawning already overlaps the ~3.4 s inference wait, so a per-step load
would push exploration past the window it has to stay inside to remain free.
Loading once behind a socket keeps the per-call cost at the 52 ms the model
actually needs for a 108-candidate screen - the measured mean candidate set.

Same shape as the fast a11y provider this codebase already runs: one process,
one socket, newline-delimited JSON.

    python -m android_world.parallel_exploration.semantic_service --port 8766
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import socketserver
import threading

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
_MODEL = None
_LOCK = threading.Lock()


def _model():
  global _MODEL
  with _LOCK:
    if _MODEL is None:
      from sentence_transformers import SentenceTransformer
      _MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
  return _MODEL


def score_many(queries: list[str], candidates: list[str]) -> list[list[float]]:
  """Cosine similarity of every candidate against every query, in one encode.

  One call per query cost 24 round trips and took a probe round to 6.9 s
  (measured 2026-09-10) - the encoder is dominated by how many strings it sees,
  and the candidates are the same for every query, so they must be encoded
  once. Returns one row per query.

  Returns zeros rather than raising when there is nothing to compare: a screen
  whose controls are all unlabelled is the normal case (55.7% of edges cannot
  be named at all), and the caller has to fall back to its own ordering there.
  """
  queries = [q for q in queries if (q or "").strip()]
  if not candidates or not queries or not any((c or "").strip() for c in candidates):
    return [[0.0] * len(candidates) for _ in range(max(1, len(queries)))]
  vectors = _model().encode(
      list(queries) + list(candidates), batch_size=128, convert_to_numpy=True,
      normalize_embeddings=True, show_progress_bar=False)
  q, c = vectors[:len(queries)], vectors[len(queries):]
  return [[float(v) for v in row] for row in (q @ c.T)]


def score(need: str, candidates: list[str]) -> list[float]:
  """Single-query convenience wrapper over score_many."""
  rows = score_many([need], candidates)
  return rows[0] if rows else [0.0] * len(candidates)


DESCRIBER = "Qwen/Qwen3-0.6B"
_DESCRIBE_SHOTS = (
    # Every answer quotes at least one label from its own input. The previous
    # set answered with bare categories ("The main settings page.") and the
    # 0.6B learned a 4-way classifier instead of a reader: 48% of its
    # descriptions on 2026-09-10 named a category no label supported - a
    # first-run consent dialog came back "A settings page with Chrome
    # options.", an audio recorder "A media player with a recorder."
    ("weather", ["CANCEL", "DISCARD"],
     "A dialog asking to CANCEL or DISCARD."),
    ("notes", ["Camera", "Screenshots", "Download", "Pictures"],
     "A folder list: Camera, Screenshots, Download, Pictures."),
    ("shop", ["Search or type here", "Start voice search", "Sign in"],
     "A home screen with a 'Search or type here' box and Sign in."),
    ("reader", ["q", "w", "e", "r", "t", "y"],
     "A screen with the on-screen keyboard open."),
    ("bank", ["Agree and proceed", "Not now"],
     "A consent screen offering 'Agree and proceed' or 'Not now'."),
)

_LM = _LM_TOK = None
_DESCRIBE_FAILED = False
_CHOOSE_FAILED = False
_SUMMARY_FAILED = False
_LM_LOCK = threading.Lock()

# The task is deliberately NOT in this prompt. 39% of screens are visited by
# more than one task and cross-task reuse is the entire point of storing the
# description; a sentence that names the current task ("moving
# holiday_photos.jpg from Podcasts to DCIM") is wrong for every other task
# that lands on the same screen. The screen's own controls are the only input.
_DESCRIBE_SYS = (
    "Name an Android screen in ONE short sentence, using the words that "
    "actually appear on it. Quote at least one control label verbatim. "
    "Begin with 'A ' or 'The '. At most 16 words. Do not invent controls "
    "that are not listed, and do not guess a category the labels do not "
    "support. Never mention a task or a file name.")


def _lm():
  global _LM, _LM_TOK
  with _LM_LOCK:
    if _LM is None:
      import torch
      from transformers import AutoTokenizer, AutoModelForCausalLM
      mps = torch.backends.mps.is_available()
      # dtype="auto" resolves to bfloat16 for this checkpoint and MPS raises
      # "BFloat16 is not supported on MPS" on .to(device).
      device, dtype = ("mps", torch.float16) if mps else ("cpu", torch.float32)
      _LM_TOK = AutoTokenizer.from_pretrained(DESCRIBER)
      _LM = AutoModelForCausalLM.from_pretrained(
          DESCRIBER, dtype=dtype).to(device).eval()
  return _LM_TOK, _LM


# Words that carry no evidence either way, so their presence in a description
# proves nothing about whether the model read the screen.
_STOP = frozenset("""a an the this that with and or of on in to for from is are
was were be been it its at by as no not you your screen page view app android
open shows showing show displayed currently option options control controls
button buttons item items list bar menu tab icon has have having sentence one
short short. their there here where which what""".split())


def _grounded(reply: str, labels: list[str]) -> bool:
  """True when the sentence uses at least one word the screen actually shows.

  The 0.6B answers a hard question with an easy one: asked to name a screen it
  has never seen, it reaches for whichever few-shot category is nearest and
  states it with full confidence. Prompt wording moves that rate but does not
  floor it, so the check is deterministic and sits after generation. A node
  with no description is the status quo; a node confidently mislabelled
  "A settings page" is worse than blank, because both the injected briefing
  and the exploration ranker read it as fact.

  The package name is deliberately NOT part of the haystack. It is on every
  sentence the model writes about that app, so grounding on it passes exactly
  the failures this exists to catch: "A settings page with Chrome options."
  for a consent dialog, and "A media player with audio recording options."
  for a format picker, whose only supported token was the package.
  """
  haystack = " ".join(l or "" for l in labels).lower()
  for raw in re.findall(r"[a-z0-9]+", reply.lower()):
    if len(raw) < 3 or raw in _STOP:
      continue
    if raw in haystack:
      return True
    # "recordings" against a "Recording" label, "folders" against "folder".
    if len(raw) > 4 and raw.rstrip("s") in haystack:
      return True
  return False


def describe(task: str, activity: str, labels: list[str]) -> str:
  del task  # deliberately unused; see _DESCRIBE_SYS
  """One sentence for a screen the model itself said nothing about.

  42.5% of inference steps emit a bare tool call with no <THINK>, so the free
  source leaves those screens blank - measured 2026-09-10 over 1032 steps.
  This fills only those; a screen the model described in its own words keeps
  that description, which is better written and costs nothing.

  Returns "" on any failure. A node with no description is the status quo; a
  node with a wrong one is worse, and an optional model must never be able to
  stall a step.
  """
  named = [l for l in labels if (l or "").strip()][:12]
  if not named:
    return ""
  try:
    import torch
    tok, model = _lm()
    app = activity.split("/")[0].split(".")[-1]
    # Few-shot. Zero-shot, this model called every screen "a settings page"
    # and echoed the field labels back ("The App: markor has a settings page
    # with CANCEL and OK" for a confirmation dialog).
    msgs = [{"role": "system", "content": _DESCRIBE_SYS}]
    for shot_app, shot_ctrl, shot_out in _DESCRIBE_SHOTS:
      msgs.append({"role": "user",
                   "content": f"{shot_app} / {', '.join(shot_ctrl)}"})
      msgs.append({"role": "assistant", "content": shot_out})
    msgs.append({"role": "user", "content": f"{app} / {', '.join(named)}"})
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
      out = model.generate(**ids, max_new_tokens=32, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    reply = " ".join(reply.split()).strip().strip('"')
    # One sentence, and never longer than the free descriptions it sits beside.
    for stop in (". ", "! ", "? "):
      if stop in reply:
        reply = reply.split(stop)[0] + stop.strip()
        break
    reply = reply[:140]
    return reply if _grounded(reply, named) else ""
  except Exception as exc:
    # Said once, not swallowed. A broken describer and a screen with no named
    # controls both return "", and without this line they look identical -
    # which is how a bfloat16-on-MPS failure first read as "no labels".
    global _DESCRIBE_FAILED
    if not _DESCRIBE_FAILED:
      _DESCRIBE_FAILED = True
      print(f"describe disabled: {type(exc).__name__}: {exc}", flush=True)
    return ""


# --- choosing one control, with the small model ---------------------------
#
# The prompt shape is the one specified for this experiment: the task, a line
# saying what is already settled and what the next unfinished step is, then
# the candidate labels in arbitrary order. The model replies with one label,
# copied verbatim.
#
# Answer position is varied deliberately across the shots. An earlier attempt
# at using this model to rank elements failed on positional bias (it scored
# 2/11 against uniform random's 10/11), and a shot set whose answers all sit
# at the end teaches exactly that bias - the natural first example, "press
# Search", has its answer last.
_CHOOSE_SHOTS = (
    ("Find flights from Hong Kong to Paris.",
     "Origin, destination, date, passenger count, and class are already "
     "correct. The next unfinished step is to submit the search without "
     "editing those fields.",
     ["Departure airport/city Hong Kong", "Arrival airport/city Paris",
      "Departure date 2026-09-11", "Passengers: 1 adult", "Class: Economy",
      "Search"],
     "Search"),
    ("Delete the note called shopping.md.",
     "The note list is open and shopping.md is visible. The next unfinished "
     "step is to reach the actions for that one note.",
     ["shopping.md", "New note", "Sort by name", "Settings", "Search"],
     "shopping.md"),
    ("Turn off Wi-Fi.",
     "The settings list is open. The next unfinished step is to open the "
     "section that holds the Wi-Fi switch.",
     ["Network & internet", "Connected devices", "Apps", "Notifications",
      "Battery", "Storage"],
     "Network & internet"),
    ("Add a 7 a.m. alarm.",
     "The clock app is open on the Clock tab. The next unfinished step is to "
     "move to the part of the app that holds alarms.",
     ["Clock", "Alarm", "Timer", "Stopwatch", "Bedtime"],
     "Alarm"),
)

_CHOOSE_SYS = (
    "You pick ONE control on an Android screen to try next. "
    "Reply with exactly one of the candidate labels, copied verbatim, and "
    "nothing else - no explanation, no quotes, no bullet. Prefer the control "
    "that advances the next unfinished step. Never pick a field whose value "
    "is already correct.")


def _match_candidate(reply: str, labels: list[str]) -> str:
  """The reply, resolved back to a candidate, or "" if it is not one of them.

  A generated label that is not on the screen cannot be probed, and treating
  a near-miss as a hit is how a chooser silently becomes a random ranker.
  Exact match first, then whitespace/case-insensitive, then a unique prefix -
  the model does clip long labels.
  """
  # Quotes and a trailing period can wrap each other either way round, so
  # strip the whole set at once rather than in a fixed order.
  reply = " ".join((reply or "").split()).strip(" \"'`.*-")
  if not reply:
    return ""
  for label in labels:
    if label == reply:
      return label
  low = reply.casefold()
  for label in labels:
    if " ".join(label.split()).casefold() == low:
      return label
  hits = [l for l in labels if " ".join(l.split()).casefold().startswith(low)]
  return hits[0] if len(hits) == 1 else ""


def choose(task: str, progress: str, labels: list[str]) -> tuple[str, str]:
  """(matched candidate, raw first line). The match is "" when it refused.

  The raw line is returned only so a refusal can be attributed: measured on
  the first `llm` launch, the chooser declined 4 of 5 two-candidate screens
  and an empty string carries no way to tell a bad label set from a model
  that answered with a sentence.

  Returns "" on any failure, on a reply that is not one of the candidates, and
  when there is nothing to choose between - an optional model must never be
  able to stall an exploration round.
  """
  named = [" ".join(str(l).split()) for l in labels if str(l).strip()][:20]
  if len(named) < 2:
    return "", ""
  try:
    import torch
    tok, model = _lm()
    msgs = [{"role": "system", "content": _CHOOSE_SYS}]
    for shot_task, shot_prog, shot_labels, shot_out in _CHOOSE_SHOTS:
      msgs.append({"role": "user",
                   "content": _choose_prompt(shot_task, shot_prog, shot_labels)})
      msgs.append({"role": "assistant", "content": shot_out})
    msgs.append({"role": "user", "content": _choose_prompt(task, progress, named)})
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
      out = model.generate(**ids, max_new_tokens=24, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    first = reply.splitlines()[0] if reply else ""
    return _match_candidate(first, named), first
  except Exception as exc:
    global _CHOOSE_FAILED
    if not _CHOOSE_FAILED:
      _CHOOSE_FAILED = True
      print(f"choose disabled: {type(exc).__name__}: {exc}", flush=True)
    return "", f"<{type(exc).__name__}>"


def _choose_prompt(task: str, progress: str, labels: list[str]) -> str:
  head = f"Task: {task.strip()}"
  if progress.strip():
    head += "\n" + progress.strip()
  body = "\n".join(f"- {l}" for l in labels)
  return f"{head}\n\nCandidate control labels, in arbitrary order:\n{body}"


# --- summarising what the graph knows, for the reasoning prompt -----------
#
# The distiller renders facts deterministically and the walked path is a bare
# list of sentences. Both are true and neither is short: a mid-episode graph
# carries 20-40 nodes, and the prompt space spent on it is space not spent on
# the screenshot. This compresses what the graph knows about THIS task into
# one line, which is the form the reasoning prompt can actually afford.
_SUMMARY_SHOTS = (
    ("Delete the note called shopping.md.",
     ["I see the Markor file list.",
      "I see the note shopping.md open in the editor.",
      "A dialog asking to Confirm or Cancel.",
      "I see the Markor file list."],
     "Opened shopping.md, reached a Confirm/Cancel dialog, and came back to "
     "the file list."),
    ("Record audio and save it as presentation.",
     ["I see the main screen of the Audio Recorder app.",
      "I see that the audio recorder is currently recording.",
      "I see the 'New name' dialog on the screen."],
     "Started a recording from the main screen and reached the 'New name' "
     "dialog."),
)

_SUMMARY_SYS = (
    "Summarise, in ONE sentence of at most 24 words, what has already been "
    "done in this app during this task, from the list of screens visited in "
    "order. Report only what the list says. Do not give advice, do not say "
    "what to do next, and do not mention screens that are not listed.")


def summarize(task: str, lines: list[str]) -> str:
  """One line for what the graph knows about this episode, or "".

  Returns "" whenever it cannot be grounded in the lines it was given - the
  same guard the screen describer uses, and for the same reason: a confident
  sentence about screens the episode never visited is worse than no sentence,
  because the reasoning prompt reads it as fact.
  """
  named = [" ".join(str(l).split()) for l in lines if str(l).strip()][:8]
  if len(named) < 2:
    return ""
  try:
    import torch
    tok, model = _lm()
    msgs = [{"role": "system", "content": _SUMMARY_SYS}]
    for shot_task, shot_lines, shot_out in _SUMMARY_SHOTS:
      msgs.append({"role": "user",
                   "content": _summary_prompt(shot_task, shot_lines)})
      msgs.append({"role": "assistant", "content": shot_out})
    msgs.append({"role": "user", "content": _summary_prompt(task, named)})
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
      out = model.generate(**ids, max_new_tokens=48, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    reply = " ".join(tok.decode(out[0][ids.input_ids.shape[1]:],
                                skip_special_tokens=True).split()).strip('"')
    for stop in (". ", "! ", "? "):
      if stop in reply:
        reply = reply.split(stop)[0] + stop.strip()
        break
    reply = " ".join(reply.split()[:26])[:200]
    # Grounded against the screens themselves, not the task: a summary that
    # only echoes the goal says nothing the prompt does not already carry.
    return reply if _grounded(reply, named) else ""
  except Exception as exc:
    global _SUMMARY_FAILED
    if not _SUMMARY_FAILED:
      _SUMMARY_FAILED = True
      print(f"summarize disabled: {type(exc).__name__}: {exc}", flush=True)
    return ""


def _summary_prompt(task: str, lines: list[str]) -> str:
  body = "\n".join(f"- {l}" for l in lines)
  return f"Task: {task.strip()}\n\nScreens visited, in order:\n{body}"


class _Handler(socketserver.StreamRequestHandler):

  def handle(self) -> None:
    for line in self.rfile:
      try:
        payload = json.loads(line)
        cands = list(payload.get("candidates") or ())
        if payload.get("summarize") is not None:
          m = payload["summarize"]
          out = {"line": summarize(str(m.get("task") or ""),
                                   list(m.get("lines") or ()))}
        elif payload.get("choose") is not None:
          c = payload["choose"]
          label, raw = choose(str(c.get("task") or ""),
                              str(c.get("progress") or ""),
                              list(c.get("labels") or ()))
          out = {"label": label, "raw": raw}
        elif payload.get("describe") is not None:
          d = payload["describe"]
          out = {"sentence": describe(str(d.get("task") or ""),
                                      str(d.get("activity") or ""),
                                      list(d.get("labels") or ()))}
        elif payload.get("queries") is not None:
          out = {"rows": score_many([str(q) for q in payload["queries"]], cands)}
        else:
          out = {"scores": score(str(payload.get("need") or ""), cands)}
      except Exception as exc:  # a bad request must not kill the service
        out = {"error": f"{type(exc).__name__}: {exc}"}
      self.wfile.write((json.dumps(out) + "\n").encode())
      self.wfile.flush()


class _Server(socketserver.ThreadingTCPServer):
  allow_reuse_address = True
  daemon_threads = True


def query_many(port: int, queries: list[str], candidates: list[str],
               timeout_s: float = 2.0) -> list[list[float]] | None:
  """Score several queries against one candidate set in a single round trip."""
  if not candidates or not queries:
    return []
  reply = _request(port, {"queries": queries, "candidates": candidates}, timeout_s)
  rows = (reply or {}).get("rows")
  if not isinstance(rows, list) or len(rows) != len([q for q in queries if (q or "").strip()]):
    return None
  return [[float(x) for x in row] for row in rows]


def query_choose(port: int, task: str, progress: str, labels: list[str],
                 timeout_s: float = 6.0) -> tuple[str, str]:
  """(chosen label, raw reply). ("", "") means "no opinion" or unreachable."""
  if len(labels) < 2:
    return "", ""
  reply = _request(port, {"choose": {"task": task, "progress": progress,
                                     "labels": labels}}, timeout_s)
  return (str((reply or {}).get("label") or ""),
          str((reply or {}).get("raw") or ""))


def query_summarize(port: int, task: str, lines: list[str],
                    timeout_s: float = 6.0) -> str:
  """One line for what the graph knows about this episode. "" means nothing."""
  if len(lines) < 2:
    return ""
  reply = _request(port, {"summarize": {"task": task, "lines": lines}}, timeout_s)
  return str((reply or {}).get("line") or "")


def _request(port: int, payload: dict, timeout_s: float):
  try:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout_s) as sock:
      sock.settimeout(timeout_s)
      sock.sendall((json.dumps(payload) + "\n").encode())
      buf = b""
      while not buf.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
          return None
        buf += chunk
    return json.loads(buf)
  except Exception:
    return None


def query_describe(port: int, task: str, activity: str, labels: list[str],
                   timeout_s: float = 8.0) -> str:
  """Ask the service for a screen sentence; "" means "leave the node blank"."""
  if not labels:
    return ""
  reply = _request(port, {"describe": {"task": task, "activity": activity,
                                       "labels": labels}}, timeout_s)
  return str((reply or {}).get("sentence") or "")


def query(port: int, need: str, candidates: list[str],
          timeout_s: float = 1.0) -> list[float] | None:
  """Ask the service for scores; None means "unavailable, rank some other way".

  Never raises: exploration must degrade to its existing ranker rather than
  abort a probe window because an optional service is down.
  """
  if not candidates:
    return []
  try:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout_s) as sock:
      sock.settimeout(timeout_s)
      sock.sendall((json.dumps({"need": need, "candidates": candidates})
                    + "\n").encode())
      buf = b""
      while not buf.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
          return None
        buf += chunk
    reply = json.loads(buf)
    scores = reply.get("scores")
    if not isinstance(scores, list) or len(scores) != len(candidates):
      return None
    return [float(s) for s in scores]
  except Exception:
    return None


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--port", type=int, default=8766)
  args = parser.parse_args()
  _model()  # pay the 7.4 s load once, before accepting traffic
  print(f"semantic_service ready on {args.port} ({MODEL_NAME})", flush=True)
  _Server(("127.0.0.1", args.port), _Handler).serve_forever()


if __name__ == "__main__":
  main()
