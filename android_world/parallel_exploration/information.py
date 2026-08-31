"""Compact model-independent inference-to-explorer contract."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterable
from typing import Any


@dataclasses.dataclass(frozen=True)
class InformationNeed:
  current_subgoal: str
  target_entity: str = ""
  required_information_slots: tuple[str, ...] = ()
  expected_affordances: tuple[str, ...] = ()
  candidate_action_types: tuple[str, ...] = ()
  risk_constraints: tuple[str, ...] = ()
  source: str = "fallback"

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def _strings(value: Any) -> tuple[str, ...]:
  if isinstance(value, str):
    return (value.strip(),) if value.strip() else ()
  if isinstance(value, Iterable):
    return tuple(str(item).strip() for item in value if str(item).strip())
  return ()


def parse_information_need(model_output: str | None, task_goal: str, ui_labels: Iterable[str] = ()) -> InformationNeed:
  text = model_output or ""
  objects = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, flags=re.DOTALL)
  for raw in reversed(objects):
    try:
      data = json.loads(raw)
    except (TypeError, ValueError):
      continue
    if not isinstance(data, dict) or "current_subgoal" not in data:
      continue
    return InformationNeed(
        current_subgoal=str(data.get("current_subgoal") or task_goal),
        target_entity=str(data.get("target_entity") or ""),
        required_information_slots=_strings(data.get("required_information_slots")),
        expected_affordances=_strings(data.get("expected_affordances")),
        candidate_action_types=_strings(data.get("candidate_action_types")),
        risk_constraints=_strings(data.get("risk_constraints")),
        source="model",
    )
  labels = tuple(dict.fromkeys(str(label).strip() for label in ui_labels if str(label).strip()))[:8]
  return InformationNeed(
      current_subgoal=task_goal.strip(),
      expected_affordances=labels,
      candidate_action_types=("CLICK", "OPEN_APP", "NAVIGATE_BACK"),
      risk_constraints=("no irreversible side effects",),
      source="task_ui_fallback",
  )


# Affordance vocabulary for A_expect. Deliberately about widget kinds rather
# than app-specific names, so it carries across apps.
_AFFORDANCE_WORDS = (
    "search box", "search bar", "search field", "search",
    "list", "menu", "dialog", "toolbar", "tab", "drawer",
    "text field", "input field", "text box", "field",
    "button", "checkbox", "toggle", "switch", "icon",
    "detail page", "settings", "option", "entry", "item",
)

# K_risk. Only actions that are irreversible or externally visible; the point
# is to keep exploration away from them, not to classify UI semantics.
_RISK_WORDS = (
    "delete", "remove", "submit", "send", "call", "confirm", "pay",
    "purchase", "share", "discard", "clear", "reset", "sign out", "logout",
)

_STOPWORDS = frozenset("""
the a an to of in on at for with and or is are was were be been being this that
these those it its i we you they he she from by as into then than so if but not
need needs to will would should can could may might have has had do does did
next step first now current currently see seen click clicking tap tapping
""".split())


def _quoted_and_capitalized(text: str) -> tuple[str, ...]:
  """Names the model actually referred to: quoted strings and proper nouns."""
  found: list[str] = []
  for match in re.findall(r"['\"]([^'\"]{2,60})['\"]", text):
    found.append(match.strip())
  for match in re.findall(r"\b([A-Z][A-Za-z0-9_]{2,})\b", text):
    if match.lower() not in _STOPWORDS:
      found.append(match)
  for match in re.findall(r"\b([A-Za-z0-9_]+\.(?:txt|md|jpg|png|pdf|mp3|wav))\b", text):
    found.append(match)
  seen: dict[str, None] = {}
  for item in found:
    seen.setdefault(item, None)
  return tuple(seen)[:12]


def parse_reasoning_prior(model_output: str | None, task_goal: str) -> InformationNeed:
  """P_{i-1} = (K_target, A_expect, K_risk, U_miss) from the model's own text.

  The explorer has to guess which control the model will pick next, and the
  task goal alone cannot help: it is identical at every step of the episode,
  so it ranks candidates the same way on the first screen as on the last.
  Measured consequence - the model's action fell inside the probed candidate
  set 0-3% of the time (2026-08-31), which is what kept prefix alignment at
  ~2% and left the lookahead mechanism with nothing to promote.

  The model, however, has already said what it is looking for. Its <THINK>
  block and explain/summary fields name the file it wants, the control it
  expects to find, and what it still does not know - reasoning that has
  already been paid for. Reading it costs no extra model call, which is the
  property that makes this usable inside the inference window.

  Falls back to the task goal when the output carries no reasoning text, so a
  model that emits bare tool calls still works, just without the prior.
  """
  text = str(model_output or "")
  think = " ".join(re.findall(r"<THINK>(.*?)</THINK>", text, flags=re.S | re.I))
  explain = " ".join(re.findall(r"explain:([^\t\n]*)", text))
  summary = " ".join(re.findall(r"summary:([^\t\n]*)", text))
  # explain/summary also appear as JSON fields in tool-call style output.
  for key in ("explain", "summary", "cot"):
    explain += " " + " ".join(re.findall(rf'"{key}"\s*:\s*"([^"]*)"', text))
  reasoning = " ".join(part for part in (think, explain, summary) if part.strip())
  if not reasoning.strip():
    return parse_information_need(None, task_goal)

  low = reasoning.lower()
  targets = _quoted_and_capitalized(reasoning)
  affordances = tuple(w for w in _AFFORDANCE_WORDS if w in low)
  risks = tuple(w for w in _RISK_WORDS if w in low)
  # U_miss: what the model says it has not resolved yet.
  missing = tuple(
      m.strip() for m in re.findall(
          r"(?:need to|looking for|have to|must|should)\s+([^.;\n]{4,60})", low)
  )[:6]

  return InformationNeed(
      # The forward-looking sentence, when there is one, is the subgoal; the
      # recap half describes what is already behind us.
      current_subgoal=(explain.strip() or think.strip() or task_goal)[:300],
      target_entity=targets[0] if targets else "",
      required_information_slots=tuple(targets) + tuple(missing),
      expected_affordances=affordances,
      candidate_action_types=("CLICK", "OPEN_APP", "NAVIGATE_BACK"),
      risk_constraints=risks or ("no irreversible side effects",),
      source="reasoning_prior",
  )
