"""Adaptive runtime optimizer (SCHED_MODE=rto): joint exploration x reasoning control at run time.

Per decision the optimizer reads one live lookup table (per app, kept across tasks, seeded with an offline prior) and
sets three knobs:

  1. reasoning path   - when the model opens a <think> segment, a fast no-think request (assistant pre-filled with
                        "action:") is raced against it at higher request priority. The fast action is ACCEPTED only
                        if exploration has verified it: a click whose destination the graph already knows (explored or
                        observed edge), not a commit, and EV = T_saved - (1 - A_class) * C_wrong > 0 with A_class >= TAU.
                        On accept the think request is aborted (frees the GPU), except for a random AUDIT share that
                        keeps decoding in the background and updates A_class with the think-vs-fast agreement.
  2. exploration budget - K = floor((T_pred - elapsed - guard) / T_probe) with T_pred the live time-to-action of this
                        app; as the fast path shortens decisions, the window and K shrink automatically (full_run's
                        heur selector with this live profile).
  3. request priority - fast request priority -1, think request 0 (vLLM priority scheduling when the server enables it).

A_class prior: think steps of successful FULL40 episodes re-decided without thinking agreed with the thinking
decision on 91% of clicks (n=69; reports/think_offline_FULL40.jsonl); swipes 18%, long-press 38%, status 90%.
"""
import json
import os
import random
import threading
import time

import common as C  # noqa: E402
import joint_graph as JG  # noqa: E402

TAU = 0.85
AUDIT = 0.25
P_THINK_PRIOR = 0.56      # seenE2 (LabServer): 1047 of 1875 decisions opened <think>
LAUNCH_AT_T0 = 1.01     # off: a request raced from t0 doubles the image prefill (rtosmoke3: decisions 1.64 -> 2.0 s)
# calibration (smoke runs rtosmoke3/4/O, 10-31, before the A/B): fast vs thinking decision on the same step agreed on
# 10/10 graph-known clicks (8/8 of them commits) and on 4/15 graph-unknown clicks
PRIOR = {"click|known": [0.91, 20], "click|known|commit": [(0.91 * 20 + 8) / 28, 28], "click|unknown": [4 / 15, 15],
         "other": [0.5, 20]}
# time to action with / without <think>: seenE2 (LabServer) 2.23 / 0.87 s; e2e116B (Orin) 13.79 / 6.07 s (fast after a
# first-token abort, prefix cached: ~7.4 s). Set per server through the environment.
DEF_T_THINK = float(os.environ.get("RTO_T_THINK", "2.23"))
DEF_T_FAST = float(os.environ.get("RTO_T_FAST", "0.87"))
DEF_GAP = 1.1
# serial (preemptive) mode: prior value of trying the fast path on a think step, p_acc 0.5 x 6.4 s saved - 0.5 x 3.6 s
# lost (Orin estimate), weight 3 tries
PRIOR_TRY_EV, PRIOR_TRY_N = float(os.environ.get("RTO_PRIOR_TRY_EV", "1.4")), 3
_LOCK = threading.Lock()


class Table:
  def __init__(self, path):
    self.path = path
    try:
      self.d = json.loads(path.read_text())
    except Exception:  # pylint: disable=broad-except
      self.d = {}

  def app(self, a):
    return self.d.setdefault(a, {"t_think": [], "t_fast": [], "agree": {}, "n_think": 0, "n_steps": 0,
                                 "fast_accept": 0, "fast_reject": 0})

  @staticmethod
  def med(xs, default):
    # median over the recent samples plus three prior pseudo-samples: one cold or stalled request (e.g. 20.8 s for
    # the first fast decision after a server start) does not flip the table
    xs = sorted(list(xs[-20:]) + [default] * 3)
    return xs[len(xs) // 2]

  def agreement(self, a, cls):
    p, n0 = PRIOR.get(cls, PRIOR["other"])
    ag, n = self.app(a)["agree"].get(cls, [0, 0])
    return (p * n0 + ag) / (n0 + n), n

  def ev(self, a, cls):  # noqa: D401
    x = self.app(a)
    t_think = self.med(x["t_think"], DEF_T_THINK)
    t_fast = self.med(x["t_fast"], DEF_T_FAST)
    saved = t_think - t_fast
    c_wrong = 2 * (t_think + DEF_GAP)          # a wrong click costs its own step and one step to undo it
    if cls.endswith("commit"):
      c_wrong *= 2                             # a wrong commit may not be undoable by one Back
    A, _ = self.agreement(a, cls)
    return saved - (1 - A) * c_wrong, A, saved, c_wrong

  def try_ev(self, a):
    t = self.app(a).get("tries", [])[-20:]
    return (sum(t) + PRIOR_TRY_EV * PRIOR_TRY_N) / (len(t) + PRIOR_TRY_N)

  def add_try(self, a, v):
    x = self.app(a)
    x["tries"] = (x.get("tries", []) + [round(v, 3)])[-30:]

  def explore_ratio(self, a):
    """Online calibration of exploration value: fast clicks accepted on an exploration-made edge, over the useful
    probes the table predicted for the probes this app actually made (prior ratio 1, weight 2)."""
    x = self.app(a)
    return (x.get("explore_credit", 0) + 2.0) / (x.get("explore_expected", 0.0) + 2.0)

  def save(self):
    try:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      self.path.write_text(json.dumps(self.d))
    except Exception:  # pylint: disable=broad-except
      pass


def classify(ui, graph, action, is_commit):
  """(class, family, known destination, commit) of a fast decision on this screen."""
  if action.action_type != "click":
    return "other", None, None, False
  try:
    _, fam, _ = C.action_token(ui, action.__dict__)
  except Exception:  # pylint: disable=broad-except
    fam = None
  if not fam:
    return "other", None, None, False
  dst = None
  if graph is not None:
    e = JG.JointGraph(graph).edge(JG.page_key(ui), fam)
    dst = JG.JointGraph.predicted(e)
    origins = sorted(e["origins"]) if e else []
  else:
    origins = []
  commit = is_commit(ui, fam)
  if dst is None:
    return "click|unknown", fam, None, commit
  return ("click|known|commit" if commit else "click|known"), fam, (dst, origins), commit


def gate(table, app, ui, graph, d_fast, is_commit):
  cls, fam, known, commit = classify(ui, graph, d_fast["action"], is_commit)
  ev, A, saved, c_wrong = table.ev(app, cls)
  why = []
  if not cls.startswith("click|known"):
    why.append("not_verified_by_graph" if cls == "click|unknown" else "not_a_click")
  if A < TAU:
    why.append("agreement_below_tau")
  if ev <= 0:
    why.append("ev_nonpositive")
  return {"accept": not why, "cls": cls, "fam": fam, "origins": (known or (None, []))[1] if known else [],
          "A": round(A, 3), "ev_s": round(ev, 3), "saved_s": round(saved, 3), "c_wrong_s": round(c_wrong, 3),
          "reject": why}


def fast_request(api_url, model, messages, max_tokens=160, priority=-1):
  import requests  # noqa: E402
  body = {"messages": messages + [{"role": "assistant", "content": "action:"}], "temperature": 0.0, "top_p": 1.0,
          "max_tokens": max_tokens, "add_generation_prompt": False, "continue_final_message": True}
  if priority is not None:
    body["priority"] = priority
  if model:
    body["model"] = model
  t0 = time.time()
  js = requests.post(api_url, json=body, timeout=(10, 60)).json()
  return "action:" + js["choices"][0]["message"]["content"], time.time() - t0, js.get("usage", {}).get("completion_tokens")


def same_click(ui, fam_fast, action):
  """The thinking decision clicks the same control family as the accepted fast one."""
  if action.action_type != "click":
    return False
  try:
    return C.action_token(ui, action.__dict__)[1] == fam_fast
  except Exception:  # pylint: disable=broad-except
    return False


# ------------------------------------------------------------------ explore-more vs infer-faster trade-off
# value of one probe = P_useful(class) x r_app x S  -  P_fail(class) x P_requery x T_think
#   S        seconds a verified fast decision saves (live: T_think - T_fast); exploration creates the known edges that
#            verify later fast decisions, so a useful probe is worth one such saving
#   P_useful, P_fail: reports/joint_table_frozen.json (896 calibration probes); r_app: online calibration above
P_REQUERY = 0.56
DELTA_MAX_PROBES = 2          # at most two complete probe cycles of action delay for exploration
DELTA_MAX_S = 6.0


def load_classes(path):
  try:
    return json.load(open(path))["classes"]
  except Exception:  # pylint: disable=broad-except
    return {}


def probe_value(cls, classes, ratio, S, t_think):
  e = classes.get(cls) or {"p_useful": 0.05, "p_fail": 0.17}
  return e["p_useful"] * ratio * S - e["p_fail"] * P_REQUERY * t_think


def plan(vals, v_fast, page_known, t_think, t_fast, elapsed, probe_s, guard, only=None):
  """Choose the mode and the exploration allocation of one reasoning window.
  infer-faster : abort <think>, fast verified decision at ~T_fast; window = T_fast; value = v_fast + probes that fit
  explore-more : keep the thinking decision (~T_think) and optionally delay the action by delta; value = probes that
                 fit in T_think + delta - delta
  vals: values of the candidate probes in priority order (only positive ones are ever taken)."""
  pos = sorted((v for v in vals if v > 0), reverse=True)

  def fit(t_end):
    return max(0, int((t_end - elapsed - guard) // probe_s)) if probe_s > 0 else 0

  opts = []
  if page_known and v_fast is not None:
    k = min(len(pos), fit(t_fast))
    opts.append(("infer_faster", 0.0, k, v_fast + sum(pos[:k])))
  for j in range(DELTA_MAX_PROBES + 1):
    delta = min(DELTA_MAX_S, j * probe_s)
    if j and delta < j * probe_s:
      break
    k = min(len(pos), fit(t_think + delta))
    opts.append(("explore_more", round(delta, 2), k, sum(pos[:k]) - delta))
  pool = [o for o in opts if only is None or o[0] == only] or opts
  best = max(pool, key=lambda o: (o[3], -o[1]))
  return {"mode": best[0], "delta_s": best[1], "k": best[2], "value_s": round(best[3], 3),
          "options": [(o[0], o[1], o[2], round(o[3], 3)) for o in opts]}
