"""Executable exploration memory for online AndroidWorld agents.

This module is deliberately independent from the Android runner.  The runner
owns the authoritative environment and the existing ``live_probe`` process
owns the shadow device.  This file is the shared, value-free memory boundary:
it can consume an observation from either side, but it never gets an executor
for the primary device.

The public API is intentionally small enough for tests and for alternative
agents to use:

* :class:`ExecutableExplorationMemory` stores a deduplicated state/action graph
  with Beta posteriors and persistent hard negatives;
* :class:`SafeProbePlanner` implements the five-probe page budget and the
  utility rule from the design;
* :func:`extract_k_step_samples` turns a trace into GUI-Shift style samples;
* :class:`ActionFusion` performs selector relocation and conservative posterior
  fusion after reasoning has returned;
* :class:`MemoryMetrics` and :class:`JsonlMemoryLogger` provide reproducible
  per-round instrumentation.

The schema is versioned.  Unknown fields are ignored on load so records made
by older experiments remain readable.  No model is called here: prompt
compression is deterministic and bounded.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


MEMORY_SCHEMA_VERSION = 4
DEFAULT_PROBE_BUDGET = 5

_DYNAMIC_TEXT = re.compile(
    r"^(?:\d{1,4}(?::\d{1,4}){1,2}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"[0-9]+(?:\.[0-9]+)?|[a-f0-9]{8,}|"
    r"(?:added\s+)?(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2},?\s+\d{4})$",
    re.IGNORECASE,
)
_PHONE_TEXT = re.compile(r"(?<!\w)\+?[0-9][0-9 ().-]{5,}[0-9](?!\w)")
_STABLE_UI_CONCEPTS = frozenset(re.findall(r"[a-z0-9]+", """search menu more settings profile account home next back matching
continue confirm cancel save edit add create delete remove share open close login sign register
logout play pause music video image photo camera gallery message chat send download upload
notification privacy security help about location map filter sort refresh favorite follow comment
like cart buy payment order history calendar contact call note notes todo task tasks alarm stopwatch
timer folder notebook file inbox done new details phone email mobile setup view list today week month
year activity tab navigation drawer toolbar button save cancel clear all done set start stop finish
preferences options address search result result settings app applications storage network wifi bluetooth
recipe cookbook page page back next contact contacts permission account location volume brightness
reminder reminders event events date time task list marked complete important starred info no yet calling
ok yes deny allow skip got apply dismiss retry later not keep rename select choose navigate up reset"""))
# The last line above: Android's own dialog and navigation labels. Before it,
# 21 of 35 common framework labels were classified as dynamic - including OK,
# Yes, No, Allow, Deny, Skip, Got it, Apply and Navigate up - so every dialog
# with an OK button was flagged dynamic_content and could never be a route
# target, and the button's own text was stripped from its selector
# (2026-09-24). Whether pressing one is safe is a separate question that
# _UNSAFE_WORDS answers at execution time; this list only says the label is
# stable UI rather than task data.
_UI_FILLER_TOKENS = {"the", "and", "for", "with", "all", "your", "my", "no", "yet", "now", "it"}
_UNSAFE_WORDS = re.compile(
    r"\b(?:delete|remove|erase|send|publish|post|pay|purchase|buy|call|dial|login|"
    r"sign\s*in|permission|allow|camera|microphone|record|share|export|"
    r"install|uninstall|reset|clear\s+data|set\s*up|enable|activate|pair|"
    r"connect|configure|subscribe|upgrade)\b",
    re.IGNORECASE,
)
_GOAL_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "do", "for", "from", "in",
    "into", "is", "it", "make", "new", "of", "on", "or", "please",
    "the", "their", "then", "this", "to", "with", "your", "create", "app",
    "add", "open", "go", "navigate", "find", "tap", "click", "enter", "item",
    "type", "set", "mark", "number", "called", "named", "name", "titled",
    "marked", "list", "screen", "detail", "details", "following", "first",
    "last", "middle", "phone", "email", "label", "work", "enter", "field",
    "fields", "hit", "not", "do", "save",
}
_OPPOSITE_INTENTS = (("create", "delete"), ("delete", "create"))
_ROUTE_INTENT_TOKENS = frozenset({
    "create", "navigate", "enter", "select", "save", "delete", "search", "mark",
})
_TASK_VALUE_PATTERNS = (
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"(?i:\b(?:first|last|middle)\s+name\s*:)\s*([^,.!?;]+)"),
    re.compile(r"(?i:\b(?:phone(?:\s+label)?|email(?:\s+label)?|address|number)\s*:)\s*([^,.!?;]+)"),
    re.compile(r"(?i:\b(?:for|named|called|name is|titled)\s+)"
               r"([A-Z][a-z]+(?:[ '\-]+[A-Z][a-z]+){0,2})"),
    re.compile(r"(?i:\b(?:number|phone|email|address|code|password)\s+(?:is|of)\s+)"
               r"([^,.!?;]+)"),
    re.compile(r"\+?\d[\d ()-]{5,}\d"),
    re.compile(r"['\"]([^'\"]{2,80})['\"]"),
)
_COMPLEX_ACTION_WORDS = re.compile(
    r"\b(?:save|submit|confirm|finish|complete|delete|remove|erase|send|publish|post|"
    r"pay|purchase|buy|call|dial|share|export|install|uninstall|reset|clear\s+data)\b",
    re.IGNORECASE,
)
_ACTION_TYPE_ALIASES = {
    "tap": "click", "longpress": "long_press", "slide": "swipe",
    "type": "input_text", "back": "navigate_back",
}


def _norm(value: Any) -> str:
  return re.sub(r"\s+", " ", str(value or "")).strip()


def _canonical_activity(value: Any) -> str:
  """Expand Android's ``package/.Class`` shorthand to one stable identity.

  The live evaluator and shadow accessibility provider can report the same
  activity as either ``net.example/.MainActivity`` or
  ``net.example/net.example.MainActivity``.  Treating those as distinct graph
  roots made a correctly observed probe path unreachable from reasoning.
  """
  activity = _norm(value)
  package, separator, component = activity.partition("/")
  if not separator or not package or not component:
    return activity
  if component.startswith("."):
    component = package + component
  return f"{package}/{component}"


def _tokens(value: Any) -> set[str]:
  text = re.sub(r"([a-z])([A-Z])", r"\1 \2", _norm(value))
  text = re.sub(r"[_./:|]+", " ", text).casefold()
  return {
      token for token in re.findall(r"[a-z0-9]{2,}", text)
      if token not in {
          "the", "and", "for", "with", "from", "this", "that",
          "android", "widget", "view", "resource", "class", "layout",
          "button", "frame", "textview", "com",
      }
  }


def goal_semantic_tokens(value: Any) -> set[str]:
  """Return task-object tokens, excluding generic instructions and literals."""
  text = _norm(value)
  for pattern in _TASK_VALUE_PATTERNS:
    def keep_semantic_terms(match: re.Match[str]) -> str:
      literal = match.group(1) if match.lastindex else match.group(0)
      literal = re.sub(r"\bto[\s-]*do\b", "todo", literal, flags=re.IGNORECASE)
      concepts = _tokens(literal) & _STABLE_UI_CONCEPTS
      return " " + " ".join(sorted(concepts)) + " "
    text = pattern.sub(keep_semantic_terms, text)
  text = re.sub(r"\bto[\s-]*do\b", " todo ", text, flags=re.IGNORECASE)
  result = set()
  for token in _tokens(text) - _GOAL_STOPWORDS:
    # Tiny deterministic plural normalization handles common UI nouns
    # (e.g. Notes -> note) without adding a model or a stemming dependency.
    if len(token) > 4 and token.endswith("ies"):
      token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
      token = token[:-1]
    if token == "todo":
      token = "task"
    result.add(token)
  return result


def _route_relevance_tokens(value: Any) -> set[str]:
  """Object terms plus a tiny deterministic task-intent vocabulary.

  Object-only overlap made every Contacts route look equally relevant (the
  sole query token was ``contact``), so a Save/detail route ranked alongside
  Create contact for an unsaved draft task. Intent words distinguish those
  routes without embeddings or an LLM.
  """
  text = _norm(value).casefold()
  tokens = goal_semantic_tokens(value)
  intent_patterns = {
      "create": r"\b(?:new|create|add|make|write|compose)\b",
      "navigate": r"\b(?:go|open|navigate|launch|switch|visit)\b",
      "enter": r"\b(?:enter|type|fill|input)\b",
      "select": r"\b(?:select|choose|change|set)\b",
      "save": r"\bsave\b",
      "delete": r"\b(?:delete|remove|erase)\b",
      # A large share of AndroidWorld questions are information-retrieval
      # requests phrased as "what/how many/which ..." rather than commands to
      # search. Treat those as search intent so a verified Search entry point
      # can be retrieved without requiring literal lexical overlap with the
      # word "search".
      "search": r"\b(?:search|find|look\s+up|what|which|where|who|when|"
                r"how\s+(?:many|much|often|long))\b",
      "mark": r"\b(?:mark|toggle|check|uncheck)\b",
  }
  tokens.update(token for token, pattern in intent_patterns.items()
                if re.search(pattern, text))
  return tokens


def _task_context_signature(value: Any) -> str:
  """A value-filtered task/action fingerprint for graph attribution.

  Store only the bounded UI-concept and intent vocabulary. Unknown object
  tokens, names, dates, addresses, and other instance-specific values are
  excluded before hashing; a digest is not treated as encryption.
  """
  tokens = _task_signature_tokens(_route_relevance_tokens(value))
  return _hash(tokens, 20) if tokens else ""


def _task_signature_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
  """Return only known UI concepts/intents, excluding task instance values."""
  normalized = set(tokens)
  return tuple(sorted(normalized & (_STABLE_UI_CONCEPTS | _ROUTE_INTENT_TOKENS)))


def _task_sensitive_literals(task: Any) -> set[str]:
  text = _norm(task)
  literals: set[str] = set()
  for pattern in _TASK_VALUE_PATTERNS:
    for match in pattern.finditer(text):
      value = match.group(1) if match.lastindex else match.group(0)
      value = _norm(value).strip(" '\".,;:")
      if value:
        literals.add(value)
  return literals


def is_task_specific_value(task: Any, value: Any) -> bool:
  """Whether an observed label repeats a literal supplied by the task."""
  text = _norm(value)
  digits = re.sub(r"\D", "", text)
  for literal in _task_sensitive_literals(task):
    if literal.casefold() in text.casefold():
      return True
    literal_digits = re.sub(r"\D", "", literal)
    if len(literal_digits) >= 6 and literal_digits in digits:
      return True
  return False


def _selector_is_actionable(selector: ElementSelector) -> bool:
  return bool(selector.resource_id or selector.text or selector.content_desc) and not selector.dynamic_text


def _click_is_navigation_selector(edge: EdgeEvidence) -> bool:
  """Reject inferred click edges whose target is structurally a scroll view."""
  if edge.action_type != "click":
    return True
  selector_identity = f"{edge.selector.class_name} {edge.selector.resource_id}".casefold()
  return not re.search(r"(?:scrollview|scroll_view|scroller|recyclerview)", selector_identity)


def _hash(value: Any, length: int = 20) -> str:
  payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), default=str)
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _get(value: Any, *names: str, default: Any = None) -> Any:
  if isinstance(value, Mapping):
    for name in names:
      if name in value:
        return value[name]
    return default
  for name in names:
    if hasattr(value, name):
      result = getattr(value, name)
      if result is not None:
        return result
  return default


def _bbox(element: Any) -> tuple[float, float, float, float] | None:
  box = _get(element, "bbox_pixels", "bbox", "bounds")
  if box is None:
    return None
  if isinstance(box, Mapping):
    values = (_get(box, "x_min", "left"), _get(box, "y_min", "top"),
              _get(box, "x_max", "right"), _get(box, "y_max", "bottom"))
  else:
    try:
      values = (box.x_min, box.y_min, box.x_max, box.y_max)
    except AttributeError:
      try:
        values = (box[0], box[1], box[2], box[3])
      except (IndexError, TypeError):
        return None
  try:
    result = tuple(float(x) for x in values)
  except (TypeError, ValueError):
    return None
  return result if result[2] >= result[0] and result[3] >= result[1] else None


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
  x1, y1 = max(left[0], right[0]), max(left[1], right[1])
  x2, y2 = min(left[2], right[2]), min(left[3], right[3])
  intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
  area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
  area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
  union = area_left + area_right - intersection
  return intersection / union if union else 0.0


def _dynamic(text: str) -> bool:
  text = _norm(text)
  if not text:
    return False
  if _DYNAMIC_TEXT.fullmatch(text) or _PHONE_TEXT.search(text):
    return True
  raw_tokens = _tokens(text)
  tokens = raw_tokens - _UI_FILLER_TOKENS
  if raw_tokens and not tokens:
    # Nothing but function words ("No", "Not now" minus "not"): not task data.
    return False
  stable = tokens & _STABLE_UI_CONCEPTS
  return not stable or len(stable) / max(1, len(tokens)) < 0.50


def is_dynamic_ui_text(value: Any) -> bool:
  """Public predicate for deciding whether a UI label is reusable memory."""
  return _dynamic(_norm(value))


@dataclasses.dataclass(frozen=True)
class ExecutableMemoryConfig:
  """Feature switches and conservative thresholds.

  ``enabled`` gates the whole layer.  The remaining switches are independent
  to make the requested ablations possible without changing the baseline
  agent or runner.
  """

  enabled: bool = False
  graph_enabled: bool = True
  exploration_enabled: bool = True
  exploration_guidance_enabled: bool = True
  graph_prompt_enabled: bool = True
  post_fusion_enabled: bool = True
  high_confidence_skip_enabled: bool = False
  k_step_memory_enabled: bool = True
  hard_negatives_enabled: bool = True
  action_groups_enabled: bool = True
  repeated_validation_enabled: bool = True
  compact_graph_enabled: bool = True
  compact_max_nodes: int = 128
  compact_max_edges: int = 256
  max_probes: int = DEFAULT_PROBE_BUDGET
  max_depth: int = 4
  top_k_paths: int = 3
  min_skill_support: int = 2
  min_skill_validations: int = 2
  override_confidence: float = 0.85
  state_landmark_threshold: float = 0.55
  state_element_threshold: float = 0.55
  max_prompt_edges: int = 4
  max_prompt_chars: int = 1800
  min_task_relevance: float = 0.30
  # Advisory text may surface a lower-confidence relevant route. Executing a
  # route remains governed by min_task_relevance plus the stricter support,
  # confidence, reversibility, ambiguity, and trap gates below.
  prompt_min_task_relevance: float = 0.30
  # Route-gate reversibility semantics. Only probes attempt a recovery, so an
  # edge the agent itself walked never gains a recovery record: under the
  # strict rule (reversible_rate >= 1.0) every execution-derived edge is
  # permanently ineligible, which is why the full 116-task run logged 2893
  # route candidates and 0 shortcut attempts. When this is True an edge with
  # no recovery attempts is treated as "reversibility unknown" rather than
  # "irreversible"; any observed recovery failure still disqualifies it, and
  # every executed hop is still verified live and rolled back on a miss.
  route_allow_unknown_reversibility: bool = False
  # Level at which a route's landing is judged. "node" is the specification's
  # rule: the landed page must be one of the edge's recorded target nodes, and
  # confidence is posterior x node-landing stability. Node identity is fragile
  # for exactly the pages routes pass through: the same Markor file dialog was
  # recorded as two nodes because one accessibility dump included the activity
  # behind the dialog and the other did not (2026-09-24), and over 480 repeated
  # (screen, control) observations the landing layout signature agreed only
  # 66% of the time against 96% for the landing activity (2026-09-13). With
  # "activity", stability and ambiguity are computed over target activities and
  # a hop counts as landed when the activity matches AND the page actually
  # changed - the second clause keeps a press that did nothing from passing.
  route_landing_level: str = "node"

  @classmethod
  def from_mapping(cls, value: Mapping[str, Any] | None) -> "ExecutableMemoryConfig":
    if not value:
      return cls()
    fields = {field.name for field in dataclasses.fields(cls)}
    cleaned = {key: value[key] for key in fields if key in value}
    for key in ("enabled", "graph_enabled", "exploration_enabled",
                "exploration_guidance_enabled",
                "graph_prompt_enabled", "post_fusion_enabled",
                "high_confidence_skip_enabled", "k_step_memory_enabled",
                "hard_negatives_enabled", "action_groups_enabled",
                "repeated_validation_enabled", "compact_graph_enabled",
                "route_allow_unknown_reversibility"):
      if key in cleaned:
        cleaned[key] = bool(cleaned[key])
    if "route_landing_level" in cleaned:
      level = str(cleaned["route_landing_level"]).strip().lower()
      cleaned["route_landing_level"] = level if level in {"node", "activity"} else "node"
    if "max_probes" in cleaned:
      cleaned["max_probes"] = max(0, min(DEFAULT_PROBE_BUDGET, int(cleaned["max_probes"])))
    return cls(**cleaned)

  @classmethod
  def from_json(cls, path: str | Path | None) -> "ExecutableMemoryConfig":
    if not path or not Path(path).is_file():
      return cls()
    with Path(path).open(encoding="utf-8") as stream:
      return cls.from_mapping(json.load(stream))


@dataclasses.dataclass(frozen=True)
class ElementSelector:
  """A relocatable selector; absolute coordinates are only a fallback hint."""

  resource_id: str = ""
  text: str = ""
  content_desc: str = ""
  class_name: str = ""
  relative_region: tuple[float, float, float, float] | None = None
  bbox: tuple[float, float, float, float] | None = None
  dynamic_text: bool = False

  @classmethod
  def from_element(cls, element: Any, screen_size: tuple[float, float] | None = None) -> "ElementSelector":
    if isinstance(element, Mapping) and isinstance(element.get("selector"), Mapping):
      nested = dict(element["selector"])
      box = _bbox(element)
      raw_box = nested.get("bbox") or box
      return cls(
          resource_id=_norm(nested.get("resource_id", "")),
          text=_norm(nested.get("text", "")),
          content_desc=_norm(nested.get("content_desc", "")),
          class_name=_norm(nested.get("class_name", "")),
          relative_region=(tuple(nested["relative_region"])
                           if nested.get("relative_region") else None),
          bbox=(tuple(float(x) for x in raw_box)
                if raw_box and len(raw_box) == 4 else None),
          dynamic_text=bool(nested.get("dynamic_text", False)),
      )
    box = _bbox(element)
    width, height = screen_size or (0.0, 0.0)
    region = None
    if box and width > 1 and height > 1:
      region = tuple(round(x, 4) for x in (
          box[0] / width, box[1] / height, box[2] / width, box[3] / height))
    text = _norm(_get(element, "text", default=""))
    content_desc = _norm(_get(
        element, "content_description", "content_desc", "contentDescription", default=""))
    dynamic_text = _dynamic(text) or _dynamic(content_desc)
    return cls(
        resource_id=_norm(_get(element, "resource_id", "resource_name", "resourceId", default="")),
        text=text if not _dynamic(text) else "",
        content_desc=content_desc if not _dynamic(content_desc) else "",
        class_name=_norm(_get(element, "class_name", "class", default="")),
        relative_region=region,
        bbox=tuple(round(x, 3) for x in box) if box else None,
        dynamic_text=dynamic_text,
    )

  def key(self) -> str:
    # Geometry is a relocation hint, never the identity: scrolling or a
    # reflow must not split one control into several edges. For icon-only
    # controls without stable labels, retain a coarse relative region so two
    # unrelated icons on the same page do not collapse into one key.
    if self.resource_id:
      stable = {"resource_id": self.resource_id, "class_name": self.class_name}
    elif self.content_desc:
      stable = {"content_desc": self.content_desc, "class_name": self.class_name}
    elif self.text:
      stable = {"text": self.text, "class_name": self.class_name}
    else:
      stable = {
          "class_name": self.class_name,
          "relative_region": tuple(round(value, 1) for value in (self.relative_region or ())),
      }
    return _hash(stable, 24)

  def describe(self) -> str:
    label = self.text or self.content_desc or self.resource_id or self.class_name or "control"
    return _norm(label)[:80]

  def relocate(
      self, elements: Sequence[Any], *, screen_size: tuple[float, float] | None = None,
  ) -> "RelocationResult":
    candidates: list[tuple[float, Any, tuple[float, float, float, float] | None]] = []
    for element in elements:
      if _get(element, "is_visible", "visible", default=True) is False:
        continue
      if _get(element, "is_enabled", "enabled", default=True) is False:
        continue
      current = ElementSelector.from_element(element, screen_size)
      score = 0.0
      if self.resource_id and current.resource_id == self.resource_id:
        score += 0.55
      if self.content_desc and current.content_desc.casefold() == self.content_desc.casefold():
        score += 0.22
      if self.text and current.text.casefold() == self.text.casefold():
        score += 0.18
      if self.class_name and current.class_name == self.class_name:
        score += 0.12
      if self.relative_region and current.relative_region:
        score += 0.08 * (1.0 - min(1.0, sum(
            abs(a - b) for a, b in zip(self.relative_region, current.relative_region))))
      if self.bbox and (box := _bbox(element)):
        score += 0.08 * _box_iou(self.bbox, box)
      if score > 0:
        candidates.append((score, element, _bbox(element)))
    candidates.sort(key=lambda item: (-item[0], str(_get(item[1], "text", default=""))))
    # Text + class is a valid fallback when an app omits resource IDs, but it
    # must still pass the ambiguity check below before it can move a click.
    if not candidates or candidates[0][0] < 0.25:
      return RelocationResult(None, 0.0, False, "selector_not_found")
    tied = [item for item in candidates if item[0] >= candidates[0][0] - 0.03]
    if len(tied) > 1:
      return RelocationResult(None, candidates[0][0], True, "ambiguous_selector")
    box = candidates[0][2]
    center = None if box is None else (round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2))
    return RelocationResult(center, candidates[0][0], False, "matched")


@dataclasses.dataclass(frozen=True)
class RelocationResult:
  center: tuple[int, int] | None
  score: float
  ambiguous: bool
  reason: str


@dataclasses.dataclass(frozen=True)
class PageObservation:
  package: str
  activity: str
  landmarks: tuple[str, ...]
  elements: tuple[Mapping[str, Any], ...]
  structural_signature: str
  semantic_signature: str
  visual_signature: str = ""
  dynamic_signature: str = ""

  @classmethod
  def from_ui(
      cls, elements: Sequence[Any], *, package: str = "", activity: str = "",
      visual_signature: str = "", screen_size: tuple[float, float] | None = None,
  ) -> "PageObservation":
    rows: list[dict[str, Any]] = []
    landmarks: set[str] = set()
    structural: list[dict[str, Any]] = []
    dynamic: list[str] = []
    for element in elements:
      box = _bbox(element)
      editable = bool(_get(element, "is_editable", "editable", default=False))
      selector = ElementSelector.from_element(element, screen_size)
      if _dynamic(selector.content_desc) or _dynamic(selector.text):
        selector = dataclasses.replace(
            selector,
            text="" if _dynamic(selector.text) else selector.text,
            content_desc=("" if _dynamic(selector.content_desc)
                          else selector.content_desc),
            # Preserve a stable resource-id as a relocation key even when
            # its visible label is volatile.
            dynamic_text=not bool(selector.resource_id),
        )
      if editable:
        # Text fields often contain names, phone numbers, emails or task
        # answers. Persist their stable field identity, never their live
        # value; the content itself is task-specific and not route knowledge.
        selector = dataclasses.replace(
            selector, text="", content_desc="", dynamic_text=True)
      role = "scroll" if _get(element, "is_scrollable", "scrollable", default=False) else (
          "input" if editable else "control")
      row = {
          "selector": dataclasses.asdict(selector),
          "role": role,
          "clickable": bool(_get(element, "is_clickable", "clickable", default=False)),
          "enabled": _get(element, "is_enabled", "enabled", default=True) is not False,
          "visible": _get(element, "is_visible", "visible", default=True) is not False,
          "bbox": box,
      }
      rows.append(row)
      structural.append({
          "resource_id": selector.resource_id,
          "class_name": selector.class_name,
          "role": role,
          "clickable": row["clickable"],
          "region": selector.relative_region,
      })
      for landmark in (selector.resource_id, selector.content_desc):
        if landmark:
          landmarks.add(f"{selector.class_name}|{landmark}"[:160])
      text = _norm(_get(element, "text", default=""))
      if editable:
        dynamic.append("editable_field_value")
      elif text and not _dynamic(text):
        landmarks.add(f"text|{text}"[:160])
        if not row["clickable"] and row["role"] != "scroll":
          stable_concepts = sorted(_tokens(text) & _STABLE_UI_CONCEPTS)
          if (stable_concepts and box and screen_size
              and box[1] <= screen_size[1] * 0.30):
            landmarks.add("state_text|" + " ".join(stable_concepts))
      elif text:
        if not row["clickable"] and row["role"] != "scroll":
          stable_concepts = sorted(_tokens(text) & _STABLE_UI_CONCEPTS)
          if (stable_concepts and box and screen_size
              and box[1] <= screen_size[1] * 0.30):
            # Preserve only stable UI concepts from volatile headings. For
            # example, "Matching October 15 2023" becomes
            # "state_text|matching"; task data itself is not stored.
            landmarks.add("state_text|" + " ".join(stable_concepts))
        dynamic.append(text)
    structural.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))
    landmark_tuple = tuple(sorted(landmarks))
    package_name = _norm(package)
    activity_name = _canonical_activity(activity)
    return cls(
        package=package_name, activity=activity_name,
        landmarks=landmark_tuple, elements=tuple(rows),
        structural_signature=_hash((package_name, activity_name, structural), 24),
        semantic_signature=_hash((package_name, activity_name, landmark_tuple), 24),
        visual_signature=_norm(visual_signature),
        dynamic_signature=_hash(sorted(dynamic), 16),
    )

  @classmethod
  def from_trace(cls, value: Mapping[str, Any], side: str) -> "PageObservation":
    page = value.get("graph", {}).get(side, {}) if isinstance(value.get("graph"), Mapping) else {}
    activity = _canonical_activity(page.get("activity", ""))
    structural = _norm(page.get("layout_signature") or page.get("structural_signature", ""))
    visual = _norm(page.get("visual_signature", ""))
    trace_elements = page.get("elements") or ()
    if trace_elements:
      return cls.from_ui(
          trace_elements,
          package=activity.split("/", 1)[0] if activity else _norm(value.get("app_package", "")),
          activity=activity,
          visual_signature=visual,
      )
    return cls(
        package=activity.split("/", 1)[0] if activity else _norm(value.get("app_package", "")),
        activity=activity,
        landmarks=tuple(sorted(str(x) for x in page.get("landmarks", ()) if x)),
        elements=tuple(), structural_signature=structural or _hash(page),
        semantic_signature=structural or _hash(page), visual_signature=visual,
    )

  @property
  def coarse_key(self) -> tuple[str, str, str]:
    return self.package, self.activity, self.structural_signature

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def _landmark_similarity(left: Iterable[str], right: Iterable[str]) -> float:
  a, b = set(left), set(right)
  if not a or not b:
    return 0.0
  # Symmetric overlap prevents a page whose landmarks are a strict subset of
  # another (e.g. Notes list vs same-Activity Tasks tab) from being merged
  # merely because every small-set landmark was found in the larger page.
  return 2.0 * len(a & b) / (len(a) + len(b))


def _matching_landmarks(page: PageObservation) -> tuple[str, ...]:
  """Landmarks from actionable app controls, excluding provider-specific chrome.

  AndroidWorld's evaluator tree and the shadow AccessibilityService do not
  expose identical wrapper/system-bar nodes. Those nodes are useful in a raw
  trace, but should not make the same app page look different to graph lookup.
  """
  result: set[str] = set()
  result.update(item for item in page.landmarks if item.startswith("state_text|"))
  for row in page.elements:
    selector = row.get("selector", {})
    resource_id = _norm(selector.get("resource_id"))
    if resource_id.startswith(("com.android.systemui:", "android:")):
      continue
    actionable = bool(
        row.get("clickable") or row.get("scrollable")
        or row.get("role") == "input"
    )
    if not actionable:
      continue
    class_name = _norm(selector.get("class_name"))
    identity = resource_id or _norm(selector.get("content_desc")) or _norm(selector.get("text"))
    if identity:
      # A resource-id outranks provider-specific class wrappers. Anonymous
      # controls still need a class+label pair to avoid over-merging.
      result.add(f"id|{resource_id}" if resource_id else f"{class_name}|{identity}")
  return tuple(sorted(result))


def _element_similarity(left: PageObservation, right: PageObservation) -> float:
  def stable_keys(page: PageObservation) -> set[tuple[str, str, str, bool]]:
    keys = set()
    for row in page.elements:
      if not (row.get("clickable") or row.get("scrollable") or row.get("role") == "input"):
        continue
      selector = row.get("selector", {})
      resource_id = _norm(selector.get("resource_id"))
      if resource_id.startswith(("com.android.systemui:", "android:")):
        continue
      # A resource-id is the stable identity when present. Visible labels can
      # legitimately change across locales, app versions, or runtime state;
      # treating those labels as identity prevented retrieval of otherwise
      # identical screens and routes.
      text = (_norm(selector.get("text"))
              if not resource_id and not selector.get("dynamic_text") else "")
      desc = (_norm(selector.get("content_desc"))
              if not resource_id and not selector.get("dynamic_text") else "")
      # Bounds/relative position are deliberately excluded: they are useful
      # for live selector relocation, not identity, and can shift between the
      # explorer's AccessibilityService and the app's reasoning observation.
      keys.add((resource_id, "" if resource_id else _norm(selector.get("class_name")), text or desc,
                bool(row.get("clickable") or row.get("scrollable"))))
    return keys
  left_keys = stable_keys(left)
  right_keys = stable_keys(right)
  if not left_keys or not right_keys:
    return 1.0 if left.structural_signature == right.structural_signature else 0.0
  return 2.0 * len(left_keys & right_keys) / (len(left_keys) + len(right_keys))


def _transition_function(
    source: PageObservation, destination: PageObservation | None,
    action_type: str, selector: ElementSelector,
) -> str:
  """Generate a short functional description from the observed delta."""
  if destination is None:
    return f"{action_type} {selector.describe()} with no observed destination"
  added = sorted(set(destination.landmarks) - set(source.landmarks))
  removed = sorted(set(source.landmarks) - set(destination.landmarks))
  if added:
    return f"{action_type} {selector.describe()} reveals {', '.join(added[:3])}"
  if removed:
    return f"{action_type} {selector.describe()} changes the page; removed {', '.join(removed[:3])}"
  if source.activity != destination.activity:
    return f"{action_type} {selector.describe()} opens {destination.activity}"
  if source.structural_signature != destination.structural_signature:
    return f"{action_type} {selector.describe()} changes the page layout"
  return f"{action_type} {selector.describe()} has no stable page delta"


@dataclasses.dataclass
class StateNode:
  node_id: str
  package: str
  activity: str
  structural_signature: str
  semantic_signature: str
  landmarks: tuple[str, ...] = ()
  elements: tuple[Mapping[str, Any], ...] = ()
  observation_count: int = 0
  visit_count: int = 0
  dynamic_variants: int = 0
  parent_state_ids: tuple[str, ...] = ()
  incoming_action_tokens: tuple[str, ...] = ()
  depth: int = 0
  semantic_aliases: tuple[str, ...] = ()
  dynamic_content: bool = False
  dynamic_reasons: tuple[str, ...] = ()
  reversible: bool = False
  return_cost_s: float = 0.0
  last_seen: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass
class EdgeEvidence:
  edge_id: str
  source_node: str
  selector: ElementSelector
  action_type: str
  function: str
  target_states: Counter[str] = dataclasses.field(default_factory=Counter)
  support_count: int = 0
  alpha: float = 1.0
  beta: float = 1.0
  meaningful_count: int = 0
  no_op_count: int = 0
  external_count: int = 0
  trap_count: int = 0
  recovery_success_count: int = 0
  recovery_failure_count: int = 0
  task_success_count: int = 0
  total_cost_s: float = 0.0
  provenance: Counter[str] = dataclasses.field(default_factory=Counter)
  observed_at_s: float = dataclasses.field(default_factory=time.time)
  dynamic: bool = False
  normalized_action_token: str = ""
  route_hit_count: int = 0
  route_miss_count: int = 0
  # Hashed, value-free task intent/object fingerprints.  This is attribution,
  # not an execution recipe: route matching still has to pass live task gates.
  task_signatures: Counter[str] = dataclasses.field(default_factory=Counter)

  @property
  def posterior_mean(self) -> float:
    return self.alpha / (self.alpha + self.beta)

  @property
  def stability(self) -> float:
    if not self.target_states:
      return 0.0
    return max(self.target_states.values()) / max(1, sum(self.target_states.values()))

  @property
  def reversible_rate(self) -> float:
    total = self.recovery_success_count + self.recovery_failure_count
    return self.recovery_success_count / total if total else 0.0

  @property
  def confidence(self) -> float:
    # Confidence estimates conditional landing reliability; evidence sufficiency
    # is enforced separately by consistent_validations/task_success_count in
    # every executable-skip gate. Multiplying by a second support shrink made
    # the documented 0.82 threshold unreachable for ordinary short routes
    # (four consistent positives read as only ~0.68 despite a 5/6 Beta mean).
    # Destination entropy and traps still lower this value directly.
    trap_factor = 0.0 if self.trap_count else 1.0
    return max(0.0, min(1.0, self.posterior_mean * self.stability * trap_factor))

  @property
  def consistent_validations(self) -> int:
    return max(self.target_states.values(), default=0)

  @property
  def ambiguous(self) -> bool:
    return len(self.target_states) > 1

  @property
  def mean_cost_s(self) -> float:
    return self.total_cost_s / self.support_count if self.support_count else 0.0

  def q_value(self, *, task_relevance: float = 0.0, navigation_prior: float = 0.0) -> float:
    return (
        (0.45 * self.confidence + 0.25 * self.posterior_mean +
         0.20 * max(0.0, min(1.0, task_relevance)) +
         0.10 * max(0.0, min(1.0, navigation_prior))) *
        max(0.0, min(1.0, self.reversible_rate if self.recovery_failure_count else 1.0)) /
        (1.0 + self.mean_cost_s)
    )


@dataclasses.dataclass(frozen=True)
class ProbeCandidate:
  selector: ElementSelector
  action: Mapping[str, Any]
  function: str = ""
  task_relevance: float = 0.0
  navigation_prior: float = 0.0
  uncertainty_bonus: float = 0.0
  novel_delta_bonus: float = 0.0
  side_effect_risk: float = 0.0
  return_cost: float = 0.0
  support_count: int = 0
  destination_entropy: float = 1.0
  reversible_known: bool = False
  trap: bool = False
  safe: bool = True
  reason: str = ""

  @property
  def action_type(self) -> str:
    raw = _norm(self.action.get("action_type") or self.action.get("action") or "").lower()
    return _ACTION_TYPE_ALIASES.get(raw, raw)

  def utility(self) -> float:
    uncertainty = self.uncertainty_bonus
    if self.support_count < 2 or self.destination_entropy > 0.45 or not self.reversible_known:
      uncertainty += 0.25
    return (self.task_relevance + self.navigation_prior + uncertainty +
            self.novel_delta_bonus - self.side_effect_risk - self.return_cost)


@dataclasses.dataclass(frozen=True)
class ProbeDecision:
  candidate: ProbeCandidate | None
  reason: str
  utility: float = 0.0
  rejected: Mapping[str, str] = dataclasses.field(default_factory=dict)


class SafeProbePolicy:
  """Hard safety gate; risk is not merely a ranking penalty."""

  def evaluate(self, candidate: ProbeCandidate) -> tuple[bool, str]:
    if not candidate.safe:
      return False, "candidate_marked_unsafe"
    if candidate.trap:
      return False, "known_trap"
    action_text = json.dumps(dict(candidate.action), ensure_ascii=False)
    action_type = candidate.action_type
    if action_type in {"input_text", "answer", "status", "navigate_home", "open_app"}:
      return False, "persistent_or_external_action"
    if _UNSAFE_WORDS.search(action_text) or _UNSAFE_WORDS.search(candidate.function):
      return False, "unsafe_side_effect_keyword"
    if candidate.side_effect_risk >= 0.35:
      return False, "side_effect_risk"
    if candidate.return_cost >= 1.0 and not candidate.reversible_known:
      return False, "unknown_recovery_cost"
    return True, "safe"


class SafeProbePlanner:
  """Five-probe, no-repeat, information-gain selector."""

  def __init__(self, config: ExecutableMemoryConfig | None = None,
               policy: SafeProbePolicy | None = None) -> None:
    self.config = config or ExecutableMemoryConfig()
    self.policy = policy or SafeProbePolicy()

  def choose(
      self, candidates: Iterable[ProbeCandidate], *, already_selected: Iterable[str] = (),
      used_budget: int = 0,
  ) -> ProbeDecision:
    seen = set(already_selected)
    rejected: dict[str, str] = {}
    if used_budget >= self.config.max_probes:
      return ProbeDecision(None, "budget_exhausted", rejected=rejected)
    admissible: list[ProbeCandidate] = []
    for candidate in candidates:
      key = candidate.selector.key()
      if key in seen:
        rejected[key] = "duplicate_selector_this_round"
        continue
      allowed, reason = self.policy.evaluate(candidate)
      if not allowed:
        rejected[key] = reason
        continue
      admissible.append(candidate)
    if not admissible:
      return ProbeDecision(None, "no_safe_candidate", rejected=rejected)
    admissible.sort(key=lambda item: (-item.utility(), item.selector.key()))
    selected = admissible[0]
    return ProbeDecision(selected, "max_expected_information_gain", selected.utility(), rejected)

  @staticmethod
  def dfs_decision(*, depth: int, max_depth: int, child_relevant: bool,
                   recovery_ok: bool, completed: bool = False) -> str:
    if completed:
      return "COMPLETE"
    if not recovery_ok or depth >= max_depth or not child_relevant:
      return "BACKTRACK"
    return "CONTINUE"


@dataclasses.dataclass
class TransitionRecord:
  source: PageObservation
  action: Mapping[str, Any]
  destination: PageObservation | None
  selector: ElementSelector
  function: str = ""
  meaningful: bool = True
  recovered: bool = True
  recovery_attempted: bool = True
  no_op: bool = False
  external: bool = False
  trap: bool = False
  cost_s: float = 0.0
  provenance: str = "exploration"
  task_success: bool = False
  task_failed: bool = False
  return_action: Mapping[str, Any] | None = None
  latency_s: float = 0.0
  task_signature: str = ""


@dataclasses.dataclass
class KStepSample:
  source_state: str
  future_state: str
  k: int
  first_action: ElementSelector
  action_sequence: tuple[Mapping[str, Any], ...]
  target_states: tuple[str, ...]
  return_action: Mapping[str, Any] | None
  elapsed_s: float
  meaningful: bool
  hard_negative: bool
  evidence: dict[str, Any]
  provenance: str = "exploration"

  def to_dict(self) -> dict[str, Any]:
    result = dataclasses.asdict(self)
    result["first_action"] = dataclasses.asdict(self.first_action)
    return result


def extract_k_step_samples(
    trace: Sequence[TransitionRecord], *, k_max: int = 4,
    task_success: bool = False,
) -> list[KStepSample]:
  """Extract k=1..4 transition samples, retaining unstable negatives."""
  samples: list[KStepSample] = []
  for start in range(len(trace)):
    for k in range(1, max(1, k_max) + 1):
      end = start + k - 1
      if end >= len(trace):
        break
      window = trace[start:end + 1]
      final = window[-1]
      destination = final.destination
      source_id = window[0].source.semantic_signature
      future_id = destination.semantic_signature if destination else ""
      hard_negative = bool(
          any(item.trap or not item.recovered or item.external or item.no_op
              for item in window) or not future_id or not final.meaningful)
      support = sum(1 for item in window if item.meaningful and item.recovered)
      distinct = len({item.destination.semantic_signature for item in window if item.destination})
      samples.append(KStepSample(
          source_state=source_id, future_state=future_id, k=k,
          first_action=window[0].selector,
          action_sequence=tuple(dict(item.action) for item in window),
          target_states=tuple(item.destination.semantic_signature for item in window if item.destination),
          return_action=final.return_action,
          elapsed_s=sum(max(0.0, item.latency_s or item.cost_s) for item in window),
          meaningful=not hard_negative,
          hard_negative=hard_negative,
          evidence={
              "support_count": support, "distinct_landing_states": distinct,
              "reversible": all(item.recovered for item in window),
              "no_op": any(item.no_op for item in window),
              "trap": any(item.trap for item in window),
              "task_success": bool(task_success and not hard_negative),
              "task_failed": bool(any(item.task_failed for item in window)),
          },
          provenance=window[0].provenance,
      ))
  return samples


@dataclasses.dataclass
class SkillGroup:
  skill_id: str
  action_sequence: tuple[str, ...]
  support_count: int
  validation_count: int
  success_count: int
  executable: bool
  source: str = "exploration"


def _action_signature(action: Mapping[str, Any]) -> str:
  clean = {key: value for key, value in action.items()
           if key not in {"x", "y", "coordinate", "bbox"}}
  return _hash(clean, 18)


def mine_action_groups(
    samples: Sequence[KStepSample], *, min_support: int = 2,
    min_validations: int = 2,
) -> list[SkillGroup]:
  """BPE-like adjacent merge of repeatedly validated action subsequences."""
  counts: Counter[tuple[str, ...]] = Counter()
  valid: Counter[tuple[str, ...]] = Counter()
  successes: Counter[tuple[str, ...]] = Counter()
  for sample in samples:
    if sample.k < 2:
      continue
    sequence = tuple(_action_signature(action) for action in sample.action_sequence)
    for width in range(2, len(sequence) + 1):
      for start in range(len(sequence) - width + 1):
        part = sequence[start:start + width]
        counts[part] += 1
        if not sample.hard_negative:
          valid[part] += 1
        if sample.evidence.get("task_success"):
          successes[part] += 1
  groups = []
  for sequence, support in counts.items():
    validation_count = valid[sequence]
    if support < min_support:
      continue
    groups.append(SkillGroup(
        skill_id=_hash(sequence, 20), action_sequence=sequence,
        support_count=support, validation_count=validation_count,
        success_count=successes[sequence],
        executable=validation_count >= min_validations and successes[sequence] > 0,
    ))
  groups.sort(key=lambda group: (-len(group.action_sequence), -group.validation_count, group.skill_id))
  return groups


@dataclasses.dataclass(frozen=True)
class FusionDecision:
  action: Mapping[str, Any]
  source: str
  confidence: float
  coordinate_corrected: bool = False
  graph_overrode: bool = False
  graph_rejected: bool = False
  disagreement: bool = False
  reason: str = ""


class ActionFusion:
  """Fuse a reasoning action with memory only after reasoning completes."""

  def __init__(self, config: ExecutableMemoryConfig | None = None) -> None:
    self.config = config or ExecutableMemoryConfig()

  def fuse(
      self, action: Mapping[str, Any], page: PageObservation,
      edges: Sequence[EdgeEvidence], *, task_relevance: Callable[[EdgeEvidence], float] | None = None,
  ) -> FusionDecision:
    original = dict(action)
    if not self.config.post_fusion_enabled:
      return FusionDecision(original, "reasoning", 0.0, reason="post_fusion_disabled")
    edges = tuple(edge for edge in edges if _selector_is_actionable(edge.selector))
    current_elements = page.elements
    selector = None
    if original.get("x") is not None and original.get("y") is not None:
      point = (float(original["x"]), float(original["y"]))
      screen_right = max((float((row.get("bbox") or (0, 0, 0, 0))[2])
                          for row in current_elements), default=0.0)
      screen_bottom = max((float((row.get("bbox") or (0, 0, 0, 0))[3])
                           for row in current_elements), default=0.0)
      screen_area = max(1.0, screen_right * screen_bottom)
      containing = []
      for row in current_elements:
        box = row.get("bbox")
        if not box or not row.get("clickable") or not row.get("enabled", True):
          continue
        area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        if (area <= 0.25 * screen_area
            and box[0] <= point[0] <= box[2]
            and box[1] <= point[1] <= box[3]):
          containing.append((area, row))
      if containing:
        # Nested accessibility trees put a full-page container before the
        # actual button. The smallest clickable hit target is the only one
        # that justifies snapping a model coordinate to its center.
        _, row = min(containing, key=lambda item: item[0])
        selector = ElementSelector(**dict(row.get("selector") or {}))
    candidates = [edge for edge in edges if edge.selector.key() == (selector.key() if selector else edge.selector.key())]
    if selector is not None:
      candidates = [edge for edge in edges if edge.selector.key() == selector.key()]
    if candidates and selector is not None:
      # The model's point already lies inside this element - that is how
      # `selector` was found - so the tap already reaches it. Moving it to the
      # element's center cannot fix a miss and changes what position-sensitive
      # views (month grids, lists, maps) receive: on SimpleCalendarDeleteEvents
      # a tap on one day was moved 127 px to the center of
      # month_view_background nine times running (em_full, 2026-09-24).
      return FusionDecision(original, "graph_consistent",
                            max(edge.confidence for edge in candidates),
                            reason="reasoning_inside_graph_target")
    # With no matching graph edge, snapping is ordinary coordinate rewriting
    # rather than memory fusion. Preserve the baseline action in that case.
    if not edges:
      return FusionDecision(original, "reasoning", 0.0, reason="no_graph_candidate")
    rank = task_relevance or (lambda edge: 0.0)
    best = max(edges, key=lambda edge: (edge.q_value(task_relevance=rank(edge)), edge.edge_id))
    relation = rank(best)
    supports_override = best.consistent_validations >= 2 or best.task_success_count >= 1
    safe_override = (supports_override and best.confidence >= self.config.override_confidence
                     and not best.ambiguous and not best.dynamic and not best.trap_count)
    same = selector is not None and selector.key() == best.selector.key()
    if same:
      return FusionDecision(original, "graph_consistent", best.confidence,
                            disagreement=False, reason="reasoning_matches_graph")
    if safe_override and relation > 0.5:
      fused = dict(best_action_from_edge(best))
      return FusionDecision(fused, "graph_override", best.confidence,
                            graph_overrode=True, reason="validated_unambiguous_graph_path")
    return FusionDecision(original, "reasoning", best.confidence,
                          graph_rejected=True, disagreement=True,
                          reason="graph_conflict_not_past_override_gate")


def best_action_from_edge(edge: EdgeEvidence) -> dict[str, Any]:
  action = {"action_type": edge.action_type}
  if edge.selector.bbox:
    action["x"] = round((edge.selector.bbox[0] + edge.selector.bbox[2]) / 2)
    action["y"] = round((edge.selector.bbox[1] + edge.selector.bbox[3]) / 2)
  return action


@dataclasses.dataclass
class MemoryMetrics:
  probe_rounds: int = 0
  probe_rounds_complete: int = 0
  probes_target: int = 0
  probes_completed: int = 0
  stop_reasons: Counter[str] = dataclasses.field(default_factory=Counter)
  node_count: int = 0
  edge_count: int = 0
  skill_count: int = 0
  repeated_validations: int = 0
  state_merges: int = 0
  prompt_adoptions: int = 0
  prompt_context_queries: int = 0
  prompt_context_count: int = 0
  prompt_context_edges: int = 0
  retrieval_path_candidates: int = 0
  exploration_guidance_queries: int = 0
  exploration_guidance_state_matches: int = 0
  exploration_guidance_controls_seen: int = 0
  exploration_guidance_controls_mature: int = 0
  exploration_guidance_controls_relevant: int = 0
  exploration_frontier_controls: int = 0
  coordinate_corrections: int = 0
  graph_overrides: int = 0
  graph_rejections: int = 0
  graph_coverage: int = 0
  skip_attempts: int = 0
  skip_hits: int = 0
  route_hit_count: int = 0
  route_miss_count: int = 0
  recovery_failures: int = 0
  extra_time_s: float = 0.0
  tasks: int = 0
  successes: int = 0
  success_steps: list[int] = dataclasses.field(default_factory=list)

  @property
  def five_probe_completion_rate(self) -> float:
    return self.probe_rounds_complete / self.probe_rounds if self.probe_rounds else 0.0

  @property
  def success_rate(self) -> float:
    return self.successes / self.tasks if self.tasks else 0.0

  @property
  def jump_hit_rate(self) -> float:
    return self.skip_hits / self.skip_attempts if self.skip_attempts else 0.0

  @property
  def recovery_failure_rate(self) -> float:
    attempts = self.probes_completed
    return self.recovery_failures / attempts if attempts else 0.0

  def to_dict(self) -> dict[str, Any]:
    result = dataclasses.asdict(self)
    result["stop_reasons"] = dict(self.stop_reasons)
    result["five_probe_completion_rate"] = self.five_probe_completion_rate
    result["success_rate"] = self.success_rate
    result["jump_hit_rate"] = self.jump_hit_rate
    result["recovery_failure_rate"] = self.recovery_failure_rate
    result["mean_success_steps"] = (
        sum(self.success_steps) / len(self.success_steps) if self.success_steps else None)
    return result


class JsonlMemoryLogger:
  """Append-only trace/logger with schema version on every row."""

  def __init__(self, path: str | Path | None = None) -> None:
    self.path = Path(path) if path else None

  def emit(self, event: str, **payload: Any) -> dict[str, Any]:
    row = {"schema_version": MEMORY_SCHEMA_VERSION, "event": event,
           "timestamp_s": time.time(), **payload}
    if self.path:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      with self.path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return row


class ExecutableExplorationMemory:
  """Persistent graph plus episode-local planning/fusion state."""

  def __init__(self, config: ExecutableMemoryConfig | None = None,
               *, path: str | Path | None = None,
               logger: JsonlMemoryLogger | None = None) -> None:
    self.config = config or ExecutableMemoryConfig()
    self.path = Path(path) if path else None
    self.logger = logger or JsonlMemoryLogger()
    self.states: dict[str, StateNode] = {}
    self.edges: dict[str, EdgeEvidence] = {}
    self._coarse: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    self._by_activity: dict[tuple[str, str], list[str]] = defaultdict(list)
    self._outgoing: dict[str, list[str]] = defaultdict(list)
    self.samples: list[KStepSample] = []
    self.skills: dict[str, SkillGroup] = {}
    self.metrics = MemoryMetrics()
    self._round_selected: set[str] = set()
    self._round_probe_count = 0
    self._episode_edges: set[str] = set()
    self._episode_authoritative_edges: set[str] = set()
    self._last_prompt_edge_ids: tuple[str, ...] = ()
    # Values supplied by the current task/action exist only in memory and are
    # used solely to redact dynamic text before it reaches persistent graph
    # state or prompt context.
    self._redacted_literals: set[str] = set()
    # live_probe emits nested rows before their root row. Buffer them until
    # the root arrives so k-step memory receives an executable sequence.
    self._pending_probe_paths: dict[
        tuple[str, int], list[tuple[int, TransitionRecord]]
    ] = defaultdict(list)
    if self.path and self.path.is_file():
      self._load(self.path)

  def begin_round(self) -> None:
    self._round_selected.clear()
    self._round_probe_count = 0
    self.metrics.probe_rounds += 1
    self.metrics.probes_target += self.config.max_probes

  def _is_task_value(self, value: Any) -> bool:
    text = _norm(value)
    if not text:
      return False
    folded = text.casefold()
    digits = re.sub(r"\D", "", text)
    for literal in self._redacted_literals:
      if literal.casefold() in folded:
        return True
      literal_digits = re.sub(r"\D", "", literal)
      if len(literal_digits) >= 6 and literal_digits in digits:
        return True
    return False

  def _sanitize_page(self, page: PageObservation) -> PageObservation:
    if not self._redacted_literals:
      return page
    landmarks = tuple(sorted(
        landmark for landmark in page.landmarks
        if not self._is_task_value(landmark)))
    elements: list[Mapping[str, Any]] = []
    removed = len(landmarks) != len(page.landmarks)
    for row in page.elements:
      clean = copy.deepcopy(dict(row))
      selector = dict(clean.get("selector") or {})
      if self._is_task_value(selector.get("text")) or self._is_task_value(
          selector.get("content_desc")):
        selector["text"] = ""
        selector["content_desc"] = ""
        selector["dynamic_text"] = True
        removed = True
      clean["selector"] = selector
      elements.append(clean)
    if not removed:
      return page
    return dataclasses.replace(
        page, landmarks=landmarks, elements=tuple(elements),
        semantic_signature=_hash((page.package, page.activity, landmarks), 24),
        dynamic_signature=_hash((page.dynamic_signature, "task_value_redacted"), 16),
    )

  def observe_page(self, page: PageObservation, *, visited: bool = True) -> StateNode:
    page = self._sanitize_page(page)
    selected = self._match_existing_state(page)
    if selected is None:
      stable_header = tuple(sorted(
          landmark for landmark in page.landmarks
          if landmark.startswith("state_text|")))
      node_id = _hash((page.package, page.activity, page.structural_signature,
                       stable_header), 20)
      selected = StateNode(node_id=node_id, package=page.package,
                           activity=page.activity,
                           structural_signature=page.structural_signature,
                           semantic_signature=page.semantic_signature,
                           landmarks=page.landmarks, elements=page.elements)
      self.states[node_id] = selected
      self._coarse[page.coarse_key].append(node_id)
      self._by_activity[(page.package, page.activity)].append(node_id)
    else:
      self.metrics.state_merges += 1
      selected.landmarks = tuple(sorted(set(selected.landmarks) | set(page.landmarks)))[:256]
      selected.dynamic_variants += int(page.dynamic_signature != _hash([], 16))
    selected.observation_count += 1
    selected.visit_count += int(visited)
    selected.last_seen = time.time()
    return selected

  def _match_existing_state(self, page: PageObservation) -> StateNode | None:
    """Read-only state lookup shared by observation and guidance queries."""
    candidates = [self.states[node_id] for node_id in self._coarse.get(page.coarse_key, ())]
    page_action_landmarks = _matching_landmarks(page)
    # Coarse recall first uses the interaction skeleton. If a list gains or
    # loses a row, fall back to a cheap semantic/landmark recall before making
    # a new node; this is the deterministic embedding approximation used by
    # the dependency-free runtime.
    if not candidates:
      same_activity = [self.states[node_id] for node_id in
                       self._by_activity.get((page.package, page.activity), ())
                       if node_id in self.states]
      candidates = []
      for node in same_activity:
        node_page = PageObservation(
            page.package, page.activity, node.landmarks, node.elements,
            node.structural_signature, node.semantic_signature)
        node_action_landmarks = _matching_landmarks(node_page)
        recall_score = (
            _landmark_similarity(node_action_landmarks, page_action_landmarks)
            if node_action_landmarks and page_action_landmarks
            else _landmark_similarity(node.landmarks, page.landmarks)
        )
        if recall_score >= 0.25:
          candidates.append(node)
    for node in candidates:
      node_page = PageObservation(
          page.package, page.activity, node.landmarks, node.elements,
          node.structural_signature, node.semantic_signature)
      element_similarity = _element_similarity(page, node_page)
      node_action_landmarks = _matching_landmarks(node_page)
      landmark_similarity = (
          _landmark_similarity(node_action_landmarks, page_action_landmarks)
          if node_action_landmarks and page_action_landmarks
          else _landmark_similarity(node.landmarks, page.landmarks)
      )
      page_state_text = set().union(*(
          _tokens(item.partition("|")[2]) for item in page.landmarks
          if item.startswith("state_text|")
      ))
      node_state_text = set().union(*(
          _tokens(item.partition("|")[2]) for item in node.landmarks
          if item.startswith("state_text|")
      ))
      state_text_compatible = not (
          page_state_text and node_state_text
          and 2.0 * len(page_state_text & node_state_text)
              / (len(page_state_text) + len(node_state_text)) < 0.50
      )
      exact_interaction_skeleton = (
          node.structural_signature == page.structural_signature
          and element_similarity >= max(0.80, self.config.state_element_threshold)
          and state_text_compatible
      )
      if exact_interaction_skeleton or (state_text_compatible and
          landmark_similarity >= self.config.state_landmark_threshold
          and element_similarity >= self.config.state_element_threshold
      ):
        return node
    return None

  def exploration_guidance(
      self, page: PageObservation, goal: str,
  ) -> dict[str, Any]:
    """Return compact, read-only EAM evidence for this page's probe chooser.

    The guidance is advisory: it neither admits nor rejects a candidate and
    cannot bypass the Ex5 safety gate. Candidate selector keys are resolved
    against the live UI again in the explorer process.
    """
    self.metrics.exploration_guidance_queries += 1
    if not (self.config.enabled and self.config.graph_enabled
            and self.config.exploration_enabled
            and self.config.exploration_guidance_enabled):
      return {"enabled": False, "state_matched": False, "controls": {}}
    page = self._sanitize_page(page)
    node = self._match_existing_state(page)
    if node is None:
      same_activity = [self.states[node_id] for node_id in
                       self._by_activity.get((page.package, page.activity), ())
                       if node_id in self.states]
      page_action_landmarks = _matching_landmarks(page)
      diagnostics = []
      for candidate in same_activity:
        candidate_page = PageObservation(
            page.package, page.activity, candidate.landmarks, candidate.elements,
            candidate.structural_signature, candidate.semantic_signature)
        candidate_action_landmarks = _matching_landmarks(candidate_page)
        landmark_score = (
            _landmark_similarity(candidate_action_landmarks, page_action_landmarks)
            if candidate_action_landmarks and page_action_landmarks
            else _landmark_similarity(candidate.landmarks, page.landmarks)
        )
        diagnostics.append({
            "node_id": candidate.node_id,
            "landmark_similarity": round(landmark_score, 3),
            "element_similarity": round(_element_similarity(page, candidate_page), 3),
            "same_structure": candidate.structural_signature == page.structural_signature,
        })
      diagnostics.sort(key=lambda row: (
          -(row["landmark_similarity"] + row["element_similarity"]), row["node_id"]))
      result = {"enabled": True, "state_matched": False, "controls": {},
                "lookup_diagnostics": diagnostics[:3]}
      self.logger.emit("exploration_guidance", state_matched=False,
                       activity=page.activity, control_count=0,
                       lookup_diagnostics=diagnostics[:3])
      return result
    self.metrics.exploration_guidance_state_matches += 1
    query = _route_relevance_tokens(goal)
    controls: dict[str, dict[str, Any]] = {}
    for edge_id in self._outgoing.get(node.node_id, ()):
      edge = self.edges.get(edge_id)
      if edge is None:
        continue
      relevance = self._edge_task_relevance(edge, query)
      stable = edge.stability >= 0.8 and len(edge.target_states) <= 1
      mature = (edge.support_count >= 2 and edge.confidence >= self.config.override_confidence
                and stable and edge.recovery_success_count > 0
                and edge.reversible_rate >= 1.0
                and edge.recovery_failure_count == 0 and edge.trap_count == 0
                and not edge.dynamic)
      row = controls.setdefault(edge.selector.key(), {
          "support_count": 0, "destination_count": 0,
          "no_op_click_support_count": 0, "no_op_click_count": 0,
          "reversible": False, "reversibility": 0.0,
          "confidence": 0.0, "task_relevance": 0.0,
          "mature": False, "trap": False, "dynamic": False,
          "no_op_count": 0, "meaningful_count": 0,
          "edge_ids": [], "observed_transition": True, "frontier": False,
      })
      row["support_count"] += edge.support_count
      row["destination_count"] += len(edge.target_states)
      row["reversible"] = bool(row["reversible"] or edge.recovery_success_count > 0)
      row["reversibility"] = max(row["reversibility"], edge.reversible_rate)
      row["confidence"] = max(row["confidence"], edge.confidence)
      row["task_relevance"] = max(row["task_relevance"], relevance)
      row["mature"] = bool(row["mature"] or mature)
      row["trap"] = bool(row["trap"] or edge.trap_count > 0)
      row["dynamic"] = bool(row["dynamic"] or edge.dynamic)
      row["no_op_count"] += edge.no_op_count
      row["meaningful_count"] += edge.meaningful_count
      row["edge_ids"].append(edge.edge_id)
      if edge.action_type == "click":
        row["no_op_click_support_count"] += edge.support_count
        row["no_op_click_count"] += edge.no_op_count
    for row in self._unseen_frontier_controls(node, page, query, controls):
      controls[row["selector_key"]] = row
    for row in controls.values():
      click_support = int(row["no_op_click_support_count"])
      no_op_clicks = int(row["no_op_click_count"])
      # Two stable click no-ops are useful hard-negative evidence. Scrolls are
      # excluded because their direction is not part of this edge identity.
      row["known_noop"] = click_support >= 2 and no_op_clicks >= click_support
    observed = [row for row in controls.values() if row.get("observed_transition")]
    frontier_count = sum(bool(row.get("frontier")) for row in controls.values())
    mature_count = sum(bool(row["mature"]) for row in observed)
    relevant_count = sum(float(row["task_relevance"]) > 0 for row in controls.values())
    self.metrics.exploration_guidance_controls_seen += len(observed)
    self.metrics.exploration_guidance_controls_mature += mature_count
    self.metrics.exploration_guidance_controls_relevant += relevant_count
    self.metrics.exploration_frontier_controls += frontier_count
    result = {
        "enabled": True, "state_matched": True, "state_id": node.node_id,
        "state_visits": node.visit_count, "controls": controls,
        "frontier_count": frontier_count,
    }
    self.logger.emit(
        "exploration_guidance", state_matched=True, state_id=node.node_id,
        state_visits=node.visit_count, control_count=len(controls),
        mature_control_count=mature_count, relevant_control_count=relevant_count,
        frontier_count=frontier_count,
        task_intent_count=len(query & _ROUTE_INTENT_TOKENS),
        task_concept_count=len(query & _STABLE_UI_CONCEPTS),
    )
    return result

  def _unseen_frontier_controls(
      self, node: StateNode, page: PageObservation, query: set[str],
      observed_controls: Mapping[str, Mapping[str, Any]],
  ) -> list[dict[str, Any]]:
    """Task-relevant, currently visible controls with no edge from this node."""
    result = []
    for element in page.elements:
      if (not element.get("clickable") or not element.get("enabled", True)
          or not element.get("visible", True)):
        continue
      selector = ElementSelector.from_element(element)
      if not _selector_is_actionable(selector) or selector.key() in observed_controls:
        continue
      label = selector.describe()
      resource_label = selector.resource_id.replace("_", " ")
      if (_UNSAFE_WORDS.search(label) or _COMPLEX_ACTION_WORDS.search(label)
          or _COMPLEX_ACTION_WORDS.search(resource_label)):
        continue
      probe_edge = EdgeEvidence(
          edge_id="", source_node=node.node_id, selector=selector,
          action_type="click", function=f"click {label}")
      relevance = self._edge_task_relevance(probe_edge, query)
      if relevance <= 0.0:
        continue
      result.append({
          "selector_key": selector.key(), "selector": dataclasses.asdict(selector),
          "support_count": 0, "destination_count": 0,
          "no_op_click_support_count": 0, "no_op_click_count": 0,
          "reversible": False, "reversibility": 0.0,
          "confidence": 0.0, "task_relevance": relevance,
          "mature": False, "trap": False, "dynamic": False,
          "no_op_count": 0, "meaningful_count": 0,
          "edge_ids": [], "observed_transition": False, "frontier": True,
          "node_discovery_opportunity": True,
      })
    return result

  def _edge_key(self, source_node: StateNode | str, selector: ElementSelector,
                action_type: str) -> str:
    source_id = source_node.node_id if isinstance(source_node, StateNode) else source_node
    return _hash((source_id, selector.key(), _ACTION_TYPE_ALIASES.get(action_type.lower(), action_type.lower())), 24)

  def _edge_task_relevance(self, edge: EdgeEvidence, query: set[str], *,
                           require_object: bool = False) -> float:
    if not query:
      return 0.0
    signature_tokens = _task_signature_tokens(query)
    task_signature = _hash(signature_tokens, 20) if signature_tokens else ""
    task_support = edge.task_signatures.get(task_signature, 0)
    # The app's own name is on every one of its screens ("markor" in the
    # package, the landmarks and the goal), so it made every Markor route look
    # related to every Markor task: "Delete all my notes" scored 0.39 against
    # "Create a new file or folder" almost entirely from that word (em_full,
    # 2026-09-24). Only the package's words are dropped - they carry no
    # information about which screen inside the app is meant.
    source = self.states.get(edge.source_node)
    app_tokens = _route_relevance_tokens(source.package) if source is not None else set()
    query = set(query) - app_tokens
    if not query:
      return 0.0
    # Keep action semantics separate from destination deltas. Otherwise a
    # generic "Save -> Contact detail" edge falsely scores as relevant to
    # "enter a contact draft" merely because both destination pages mention
    # contact. A destination match is useful, but intentionally weaker.
    action_parts = re.split(
        r"\b(?:reveals|opens|changes|has no stable page delta)\b",
        edge.function, maxsplit=1, flags=re.IGNORECASE)
    action_text = action_parts[0]
    action_tokens = _route_relevance_tokens(action_text + " " + edge.selector.describe())
    destination_text = action_parts[1] if len(action_parts) > 1 else ""
    destination_tokens = _route_relevance_tokens(destination_text)
    for target_id in edge.target_states:
      target = self.states.get(target_id)
      if target is not None:
        destination_tokens.update(_route_relevance_tokens(" ".join(target.landmarks)))
    destination_tokens -= app_tokens
    query_intents = query & _ROUTE_INTENT_TOKENS
    query_objects = query - _ROUTE_INTENT_TOKENS
    edge_intents = action_tokens & _ROUTE_INTENT_TOKENS
    if any(a in query_intents and b in edge_intents and a not in edge_intents
           and b not in query_intents for a, b in _OPPOSITE_INTENTS):
      # A delete task and a create control are opposite operations, not a
      # weak match: the lexical score gave them 0.39 and the route was
      # injected into all 8 prompts of a "delete all notes" episode. Only
      # true opposites count - "open Contacts" through a Search button is a
      # different verb, not a contradiction.
      return 0.0
    action_intent_score = (
        len(query_intents & action_tokens) / len(query_intents)
        if query_intents else 0.0
    )
    action_object_score = (
        len(query_objects & action_tokens) / len(query_objects)
        if query_objects else 0.0
    )
    # Intent match is normalized within the small intent vocabulary rather
    # than all task words; otherwise one useful "Search" edge is diluted by
    # every noun and formatting phrase in a long question.
    destination_score = (
        0.65 * len(query_objects & destination_tokens) / len(query)
    )
    if require_object and not (query_objects & action_tokens) and not destination_score:
      # Intent alone says what kind of operation, not on what. "Add the
      # recipes from recipes.txt in Markor to Broccoli" shares only "add" ->
      # create with Markor's new-file button, and a shortcut took that button
      # for it twice. An action taken without the model must name something
      # the task names.
      return 0.0
    action_score = max(0.65 * action_intent_score, action_object_score)
    lexical_score = max(action_score, destination_score)
    if lexical_score <= 0.0 or not task_support:
      return lexical_score
    # Reuse task-family evidence only as a small tie-break after current-task
    # grounding; it cannot make an unrelated edge relevant or bypass vetoes.
    return min(1.0, lexical_score + min(0.12, 0.03 * math.log1p(task_support)))

  def record_transition(self, record: TransitionRecord) -> EdgeEvidence:
    source = self.observe_page(record.source, visited=True)
    destination = self.observe_page(record.destination, visited=False) if record.destination else None
    action_type = _ACTION_TYPE_ALIASES.get(
        _norm(record.action.get("action_type") or record.action.get("action") or "").lower(),
        _norm(record.action.get("action_type") or record.action.get("action") or "").lower())
    edge_id = self._edge_key(source, record.selector, action_type)
    edge = self.edges.get(edge_id)
    if edge is None:
      action_token = " ".join(sorted(_tokens(
          f"{action_type} {record.selector.describe()}")))
      edge = EdgeEvidence(edge_id=edge_id, source_node=source.node_id,
                          selector=record.selector, action_type=action_type,
                          function=record.function or _transition_function(
                              record.source, record.destination, action_type,
                              record.selector),
                          dynamic=record.selector.dynamic_text,
                          normalized_action_token=action_token)
      self.edges[edge_id] = edge
      self._outgoing[source.node_id].append(edge_id)
    edge.support_count += 1
    edge.provenance[record.provenance] += 1
    if record.task_signature:
      edge.task_signatures[record.task_signature] += 1
    edge.meaningful_count += int(record.meaningful)
    edge.no_op_count += int(record.no_op)
    edge.external_count += int(record.external)
    recovery_failed = record.recovery_attempted and not record.recovered and not record.external
    edge.trap_count += int(record.trap or recovery_failed)
    if record.recovery_attempted:
      edge.recovery_success_count += int(record.recovered)
      edge.recovery_failure_count += int(not record.recovered)
    edge.task_success_count += int(record.task_success)
    edge.total_cost_s += max(0.0, record.cost_s or record.latency_s)
    if destination:
      edge.target_states[destination.node_id] += 1
      parent_ids = set(destination.parent_state_ids)
      parent_ids.add(source.node_id)
      destination.parent_state_ids = tuple(sorted(parent_ids))
      action_token = edge.normalized_action_token or " ".join(sorted(_tokens(
          f"{action_type} {record.selector.describe()}")))
      destination.incoming_action_tokens = tuple(sorted(
          set(destination.incoming_action_tokens) | {action_token}))
      destination.depth = min(
          destination.depth or (source.depth + 1), source.depth + 1)
      destination.semantic_aliases = tuple(sorted(
          set(destination.semantic_aliases) | set(destination.landmarks)))[:64]
      # Dynamic is element-level (spec: a clock, a notification or one
      # dynamic title must not make a whole page unreusable). The page as a
      # target is dynamic only when its core structure is: nothing stable is
      # left to recognise it by. The old rule - any non-input element with
      # dynamic text - held on every real page measured (Markor's file list
      # and its new-file dialog both carry task-shaped text), so no route
      # target ever passed dynamic_target_state (2026-09-24).
      dynamic_rows = sum(
          1 for row in record.destination.elements
          if bool((row.get("selector") or {}).get("dynamic_text"))
          and row.get("role") != "input")
      destination.dynamic_content = not destination.landmarks
      destination.dynamic_reasons = tuple(
          reason for reason, present in (
              ("no_stable_landmark", not destination.landmarks),
              (f"dynamic_elements:{dynamic_rows}", dynamic_rows > 0),
          ) if present)
      destination.reversible = destination.reversible or bool(
          record.recovery_attempted and record.recovered)
      destination.return_cost_s = (
          min(destination.return_cost_s, record.cost_s)
          if destination.return_cost_s and record.cost_s > 0
          else max(0.0, record.cost_s)
      )
    if (record.meaningful and (not record.recovery_attempted or record.recovered)
        and not record.no_op and not record.external and not record.trap):
      edge.alpha += 1.0
    else:
      edge.beta += 1.0
    edge.observed_at_s = time.time()
    self._episode_edges.add(edge.edge_id)
    if record.recovery_attempted:
      self.metrics.recovery_failures += int(not record.recovered)
    self.metrics.graph_coverage = len({item.source_node for item in self.edges.values()})
    self.logger.emit("transition", edge_id=edge.edge_id, source_node=edge.source_node,
                     target_node=destination.node_id if destination else None,
                     action=dict(record.action), selector=dataclasses.asdict(record.selector),
                     provenance=record.provenance, meaningful=record.meaningful,
                     recovered=record.recovered, trap=record.trap,
                     posterior=edge.posterior_mean, confidence=edge.confidence)
    return edge

  def begin_task(self) -> None:
    """Clear only episode-local action attribution; preserve learned graph."""
    self._episode_edges.clear()
    self._episode_authoritative_edges.clear()

  def record_task_outcome(self, *, success: bool, steps: int) -> None:
    self.metrics.tasks += 1
    self.metrics.successes += int(success)
    if success:
      self.metrics.success_steps.append(max(0, int(steps)))
      for edge_id in self._episode_authoritative_edges:
        edge = self.edges.get(edge_id)
        if (edge is not None and edge.action_type in {"click", "navigate_back"}
            and not edge.dynamic and edge.target_states and not edge.trap_count):
          edge.task_success_count += 1
    attributed_edges = sorted(self._episode_authoritative_edges)
    self.logger.emit("task_outcome", success=bool(success), steps=int(steps),
                     authoritative_route_edges=attributed_edges,
                     success_edges_attributed=(len(attributed_edges) if success else 0))
    self.begin_task()

  def executable_skills(self) -> list[SkillGroup]:
    """Validated action groups that are allowed to become direct executors."""
    return sorted((skill for skill in self.skills.values() if skill.executable),
                  key=lambda skill: (-skill.validation_count, -skill.support_count, skill.skill_id))

  def ingest_probe_row(self, row: Mapping[str, Any]) -> EdgeEvidence | None:
    """Adapt an existing ``live_probe`` JSON row into the executable graph."""
    if not isinstance(row.get("graph"), Mapping):
      return None
    self._redacted_literals.update(_task_sensitive_literals(row.get("task", "")))
    source = PageObservation.from_trace(row, "src")
    destination = PageObservation.from_trace(row, "dst")
    source = self._sanitize_page(source)
    destination = self._sanitize_page(destination)
    action = dict(row.get("graph", {}).get("action", {}))
    element = row.get("element", {})
    element_text = _norm(element.get("text", ""))
    element_desc = _norm(element.get("content_desc") or element.get("content-description", ""))
    dynamic_element = (
        self._is_task_value(element_text) or self._is_task_value(element_desc)
        or _dynamic(element_text) or _dynamic(element_desc)
    )
    if dynamic_element:
      element_text = ""
      element_desc = ""
      action = dict(action)
      if action.get("element_identity"):
        parts = str(action["element_identity"]).split("|")
        if len(parts) >= 4:
          parts[1] = ""
          parts[2] = ""
          action["element_identity"] = "|".join(parts)
        else:
          action["element_identity"] = "[dynamic control label redacted]"
    selector = ElementSelector(
        resource_id=_norm(element.get("resource_id") or element.get("resource-id", "")),
        text=element_text if not _dynamic(element_text) else "",
        content_desc=element_desc,
        class_name=_norm(element.get("class") or element.get("class_name", "")),
        bbox=tuple(float(x) for x in (element.get("bounds") or ()))
        if len(element.get("bounds") or ()) == 4 else None,
        dynamic_text=dynamic_element,
    )
    discovered = row.get("discovered") or {}
    stable_delta = [
        _norm(value) for value in (discovered.get("new_texts") or ())
        if _norm(value) and not _dynamic(str(value)) and not self._is_task_value(value)
    ][:5]
    action_type = _ACTION_TYPE_ALIASES.get(
        _norm(action.get("action_type") or "click").lower(),
        _norm(action.get("action_type") or "click").lower())
    destination_name = destination.activity.rsplit("/", 1)[-1]
    if stable_delta:
      function = f"{action_type} {selector.describe()} reveals {'; '.join(stable_delta)}"
    elif source.activity != destination.activity:
      function = f"{action_type} {selector.describe()} opens {destination_name}"
    elif (source.structural_signature != destination.structural_signature
          or source.activity != destination.activity):
      function = f"{action_type} {selector.describe()} changes the page layout"
    else:
      function = f"{action_type} {selector.describe()} has no stable page delta"
    meaningful = bool(
        stable_delta or source.structural_signature != destination.structural_signature or
        source.activity != destination.activity)
    recovered = bool(row.get("recovery_ok", False))
    external = bool(
        destination.package and source.package and destination.package != source.package)
    trap = (not recovered) or external or row.get("notes") in {
        "RESTORE_FAILED", "NESTED_RESTORE_FAILED"}
    record = TransitionRecord(
        source=source, action=action, destination=destination,
        selector=selector, function=function,
        meaningful=meaningful, recovered=recovered, no_op=not meaningful,
        external=external, trap=trap,
        cost_s=float((row.get("timings_ms") or {}).get("total", 0.0)) / 1000.0,
        provenance="exploration", latency_s=float((row.get("timings_ms") or {}).get("total", 0.0)) / 1000.0,
        task_signature=_task_context_signature(row.get("task", "")),
    )
    edge = self.record_transition(record)
    if self.config.k_step_memory_enabled:
      path_key = (str(row.get("trial_id") or ""), int(row.get("probe_idx") or 0))
      depth = max(1, int(row.get("depth") or 1))
      if depth > 1:
        self._pending_probe_paths[path_key].append((depth, record))
      else:
        nested = sorted(
            self._pending_probe_paths.pop(path_key, ()), key=lambda item: item[0]
        )
        self.add_k_step_trace([record] + [item for _, item in nested])
    return edge

  def record_authoritative(
      self, source: PageObservation, action: Mapping[str, Any], destination: PageObservation | None,
      *, task_success: bool = False, latency_s: float = 0.0,
      task_context: str = "",
  ) -> EdgeEvidence:
    task_signature = _task_context_signature(task_context)
    action_type = _ACTION_TYPE_ALIASES.get(
        _norm(action.get("action_type") or "click").lower(),
        _norm(action.get("action_type") or "click").lower())
    if action_type == "input_text":
      entered_value = _norm(action.get("text") or action.get("value") or "")
      if entered_value:
        self._redacted_literals.add(entered_value)
      # User-entered values are neither replayable routes nor safe persistent
      # memory. Keep a redacted, explicitly dynamic hard-negative record.
      selector = ElementSelector(class_name="android.widget.EditText", dynamic_text=True)
      return self.record_transition(TransitionRecord(
          source=source,
          action={"action_type": "input_text", "payload_redacted": True},
          destination=destination, selector=selector,
          function="input_text into a task-specific field (payload redacted)",
          meaningful=destination is not None, recovered=False, recovery_attempted=False,
          cost_s=latency_s, latency_s=latency_s, provenance="inference",
          task_success=task_success, task_signature=task_signature,
      ))
    selector = selector_for_action(action, source)
    edge = self.record_transition(TransitionRecord(
        source=source, action=action, destination=destination, selector=selector,
        function=_transition_function(source, destination, action_type, selector),
        meaningful=destination is not None,
        recovered=False, recovery_attempted=False,
        cost_s=latency_s, latency_s=latency_s,
        provenance="inference", task_success=task_success,
        task_signature=task_signature))
    if (action_type in {"click", "navigate_back"} and destination is not None
        and _selector_is_actionable(selector) and not selector.dynamic_text):
      self._episode_authoritative_edges.add(edge.edge_id)
    return edge

  def record_skip_validation(self, edge: EdgeEvidence, *, matched: bool,
                             destination: PageObservation | None = None,
                             latency_s: float = 0.0) -> None:
    """Validate a high-confidence skip and immediately demote a miss."""
    self.metrics.skip_attempts += 1
    self.metrics.skip_hits += int(matched)
    if matched:
      edge.alpha += 1.0
    else:
      edge.beta += 2.0
      edge.trap_count += 1
      self.metrics.graph_rejections += 1
    if destination is not None:
      node = self.observe_page(destination, visited=False)
      edge.target_states[node.node_id] += 1
    self.logger.emit("skip_validation", edge_id=edge.edge_id, matched=matched,
                     destination=destination.semantic_signature if destination else None,
                     latency_s=latency_s)

  def record_route_result(
      self, edges: Sequence[EdgeEvidence], *, hit: bool,
      failed_edge: EdgeEvidence | None = None, reason: str = "",
  ) -> None:
    """Update route-level evidence on this graph's edges, never a sidecar."""
    if hit:
      self.metrics.route_hit_count += 1
      self.metrics.skip_attempts += 1
      self.metrics.skip_hits += 1
      for edge in edges:
        edge.route_hit_count += 1
        edge.alpha += 1.0
    else:
      self.metrics.route_miss_count += 1
      self.metrics.skip_attempts += 1
      self.metrics.graph_rejections += 1
      target = failed_edge or (edges[-1] if edges else None)
      if target is not None:
        target.route_miss_count += 1
        target.beta += 1.0
    self.logger.emit(
        "route_validation", hit=hit, reason=reason,
        edge_ids=[edge.edge_id for edge in edges],
        failed_edge_id=failed_edge.edge_id if failed_edge else None,
    )

  def plan_probes(self, candidates: Iterable[ProbeCandidate]) -> ProbeDecision:
    planner = SafeProbePlanner(self.config)
    decision = planner.choose(candidates, already_selected=self._round_selected,
                              used_budget=self._round_probe_count)
    if decision.candidate is not None:
      self._round_selected.add(decision.candidate.selector.key())
      self._round_probe_count += 1
      self.metrics.probes_completed += 1
    else:
      self.metrics.stop_reasons[decision.reason] += 1
      if decision.reason != "budget_exhausted":
        self.logger.emit("exploration_stop", reason=decision.reason,
                         rejected=dict(decision.rejected))
    if self.config.max_probes and self._round_probe_count == self.config.max_probes:
      self.metrics.probe_rounds_complete += 1
    return decision

  def add_k_step_trace(self, trace: Sequence[TransitionRecord], *, task_success: bool = False) -> list[KStepSample]:
    if not self.config.k_step_memory_enabled:
      return []
    extracted = extract_k_step_samples(trace, k_max=self.config.max_depth, task_success=task_success)
    if not self.config.hard_negatives_enabled:
      extracted = [sample for sample in extracted if not sample.hard_negative]
    self.samples.extend(extracted)
    if self.config.action_groups_enabled:
      self.skills = {skill.skill_id: skill for skill in mine_action_groups(
          self.samples, min_support=self.config.min_skill_support,
          min_validations=self.config.min_skill_validations if self.config.repeated_validation_enabled else 1)}
    self.metrics.skill_count = len(self.skills)
    self.metrics.repeated_validations = sum(skill.validation_count for skill in self.skills.values())
    return extracted

  def retrieve_paths(
      self, page: PageObservation, goal: str, *, top_k: int | None = None,
      min_task_relevance: float | None = None,
  ) -> list[list[EdgeEvidence]]:
    if not self.config.graph_enabled:
      return []
    node = self.observe_page(page, visited=False)
    limit = top_k or self.config.top_k_paths
    relevance_threshold = (
        self.config.min_task_relevance
        if min_task_relevance is None else max(0.0, min(1.0, min_task_relevance))
    )
    query = _route_relevance_tokens(goal)
    if not query:
      return []
    paths: list[tuple[float, list[EdgeEvidence], str, float]] = []
    frontier: list[tuple[float, list[EdgeEvidence], str, frozenset[str]]] = [
        (0.0, [], node.node_id, frozenset({node.node_id}))]
    for _ in range(max(1, self.config.max_depth)):
      next_frontier = []
      for score, path, source_id, visited in frontier:
        outgoing = [self.edges[edge_id] for edge_id in self._outgoing.get(source_id, ())
                    if edge_id in self.edges
                    and self.edges[edge_id].trap_count == 0
                    and not self.edges[edge_id].dynamic
                    and _selector_is_actionable(self.edges[edge_id].selector)
                    and _click_is_navigation_selector(self.edges[edge_id])
                    and not _UNSAFE_WORDS.search(
                        self.edges[edge_id].action_type + " "
                        + self.edges[edge_id].selector.describe())]
        for edge in outgoing:
          targets = [target for target in edge.target_states if target not in visited]
          if not targets:
            continue
          relevance = self._edge_task_relevance(edge, query)
          edge_score = score + edge.q_value(task_relevance=relevance,
                                            navigation_prior=0.5 if edge.action_type in {"click", "swipe"} else 0.1)
          next_path = path + [edge]
          path_relevance = max(
              self._edge_task_relevance(item, query) for item in next_path)
          path_stability = sum(item.reversible_rate for item in next_path) / len(next_path)
          ranked_score = (
              0.55 * (edge_score / len(next_path))
              + 0.35 * path_relevance
              + 0.10 * path_stability
          ) / (1.0 + 0.15 * (len(next_path) - 1))
          paths.append((ranked_score, next_path, edge.edge_id, path_relevance))
          for target in targets:
            next_frontier.append((edge_score, next_path, target, visited | {target}))
      frontier = next_frontier
      if not frontier:
        break
    paths.sort(key=lambda item: (-item[0], -item[3], item[2]))
    self.metrics.retrieval_path_candidates += len(paths)
    selected: list[list[EdgeEvidence]] = []
    used: set[str] = set()
    for _, path, _, relevance in paths:
      # Filter before top-K: unrelated high-Q routes must not crowd out a
      # lower-support path that actually matches the task's object/action.
      if relevance < relevance_threshold:
        continue
      key = "|".join(edge.edge_id for edge in path)
      if key in used:
        continue
      used.add(key)
      selected.append(path)
      if len(selected) >= limit:
        break
    return selected

  def prompt_context(self, page: PageObservation, goal: str) -> str:
    self._last_prompt_edge_ids = ()
    if not (self.config.enabled and self.config.graph_enabled and self.config.graph_prompt_enabled):
      return ""
    self.metrics.prompt_context_queries += 1
    safe_page = self._sanitize_page(page)
    node = self.observe_page(safe_page, visited=False)
    boundary = self.action_boundary_reason(safe_page)
    guidance = self.exploration_guidance(safe_page, goal)
    frontiers = sorted(
        (row for row in guidance.get("controls", {}).values()
         if row.get("frontier")),
        key=lambda row: (-float(row.get("task_relevance", 0.0)),
                         str(row.get("selector_key", ""))))
    paths = [path for path in self.retrieve_paths(
        page, goal,
        min_task_relevance=self.config.prompt_min_task_relevance)
             if path and len(path) <= 3]
    query = _route_relevance_tokens(goal)
    ranked_paths = []
    for path in paths:
      path = self._navigation_prefix(path)
      if not path:
        continue
      if any(edge.trap_count or edge.dynamic for edge in path):
        continue
      if any(not _selector_is_actionable(edge.selector) for edge in path):
        continue
      if any(edge.meaningful_count <= 0 or edge.no_op_count >= edge.support_count
             for edge in path):
        continue
      relevance = max(self._edge_task_relevance(edge, query) for edge in path)
      if relevance < self.config.prompt_min_task_relevance:
        continue
      q = sum(edge.q_value(task_relevance=self._edge_task_relevance(edge, query))
              for edge in path) / len(path)
      ranked_paths.append((q + relevance, path, relevance))
    ranked_paths.sort(key=lambda item: (-item[0], tuple(e.edge_id for e in item[1])))
    traps = []
    for edge_id in self._outgoing.get(node.node_id, ()):
      edge = self.edges.get(edge_id)
      if edge is None or not (edge.trap_count or (
          edge.support_count >= 2 and edge.no_op_count >= edge.support_count)):
        continue
      if self._edge_task_relevance(edge, query) > 0.0:
        traps.append(edge.selector.describe())
    if not ranked_paths and not frontiers and not traps and not boundary:
      return ""
    lines = ["[TASK-CONDITIONED GUI GRAPH]"]
    if node is not None:
      stable_landmarks = sorted({item.partition("|")[2] or item for item in node.landmarks
                                 if not item.startswith("state_text|")})
      lines.append(
          f"Current UI: {node.activity}; stable landmarks: "
          + (", ".join(stable_landmarks[:6]) if stable_landmarks else "none"))
    if boundary:
      lines.append(
          f"Action boundary: {boundary}; do not execute a graph shortcut here. "
          "Inspect the live screen and let the agent reason about the task action.")
    emitted_edges = 0
    emitted_edge_ids = []
    for rank, (_, path, relevance) in enumerate(ranked_paths[:self.config.top_k_paths], 1):
      steps = []
      for edge in path:
        if emitted_edges >= self.config.max_prompt_edges:
          break
        steps.append(
            f"{edge.selector.describe()} [{edge.action_type}] -> {edge.function} "
            f"(n={edge.support_count}, q={edge.q_value(task_relevance=relevance):.2f}, "
            f"confidence={edge.confidence:.2f}, reversible={edge.reversible_rate:.2f})")
        emitted_edges += 1
        emitted_edge_ids.append(edge.edge_id)
      if steps:
        lines.append(f"- Route {rank} (task relevance={relevance:.2f}): " + " ; then ".join(steps))
      if emitted_edges >= self.config.max_prompt_edges:
        break
    if frontiers:
      labels = [ElementSelector.from_element({"selector": row["selector"]}).describe()
                for row in frontiers[:3]]
      lines.append(
          "Unmeasured task-relevant frontier (candidate only; not a verified route): "
          + ", ".join(labels))
    if traps:
      lines.append("Task-relevant observed trap/no-op controls: "
                   + ", ".join(sorted(set(traps))[:3]))
    if len(lines) == 1:
      return ""
    lines.append(
        "Observed navigation is prior evidence, not a command: relocate each control on the "
        "live UI and verify every landing. Stop graph reuse at editable forms or commit actions.")
    self.metrics.prompt_context_count += 1
    self.metrics.prompt_context_edges += emitted_edges
    self._last_prompt_edge_ids = tuple(emitted_edge_ids)
    return "\n".join(lines)[:self.config.max_prompt_chars]

  def action_boundary_reason(self, page: PageObservation) -> str:
    """Whether the current UI is at a task action rather than navigation."""
    for row in page.elements:
      if not row.get("visible", True) or not row.get("enabled", True):
        continue
      if row.get("role") == "input":
        return "editable_form"
      if not row.get("clickable"):
        continue
      selector = ElementSelector.from_element(row)
      label = selector.describe() + " " + selector.resource_id.replace("_", " ")
      if _COMPLEX_ACTION_WORDS.search(label):
        return "commit_or_side_effect_control"
    return ""

  def _node_action_boundary_reason(self, node: StateNode | None) -> str:
    if node is None:
      return ""
    page = PageObservation(
        package=node.package, activity=node.activity, landmarks=node.landmarks,
        elements=node.elements, structural_signature=node.structural_signature,
        semantic_signature=node.semantic_signature)
    return self.action_boundary_reason(page)

  def _navigation_prefix(self, path: Sequence[EdgeEvidence]) -> list[EdgeEvidence]:
    """Keep navigation hops only; stop before commits and after landing on a form."""
    prefix = []
    for edge in path:
      action_phrase = re.split(
          r"\b(?:reveals|opens|changes|has no stable page delta)\b",
          edge.function, maxsplit=1, flags=re.IGNORECASE)[0]
      label = (edge.action_type + " " + edge.selector.describe() + " "
               + edge.selector.resource_id.replace("_", " ") + " " + action_phrase)
      if _COMPLEX_ACTION_WORDS.search(label):
        break
      prefix.append(edge)
      target_reasons = [self._node_action_boundary_reason(self.states.get(target_id))
                        for target_id in edge.target_states]
      if any(target_reasons):
        break
    return prefix

  def record_prompt_action(self, action: Mapping[str, Any], page: PageObservation) -> bool:
    """Count a model action that actually targets a selector we injected."""
    if not self._last_prompt_edge_ids or action.get("x") is None or action.get("y") is None:
      return False
    point = (float(action["x"]), float(action["y"]))
    action_type = _ACTION_TYPE_ALIASES.get(
        _norm(action.get("action_type", "")).lower(),
        _norm(action.get("action_type", "")).lower())
    for edge_id in self._last_prompt_edge_ids:
      edge = self.edges.get(edge_id)
      if edge is None or action_type != edge.action_type:
        continue
      for row in page.elements:
        if not row.get("clickable") or not row.get("enabled", True):
          continue
        box = row.get("bbox")
        if not box or not (box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]):
          continue
        relocation = edge.selector.relocate([row])
        if relocation.center is not None and not relocation.ambiguous:
          self.metrics.prompt_adoptions += 1
          self.logger.emit("prompt_adoption", edge_id=edge_id, action_type=action_type)
          return True
    return False

  def fuse_action(self, action: Mapping[str, Any], page: PageObservation, goal: str) -> FusionDecision:
    edges = [edge for path in self.retrieve_paths(page, goal, top_k=5) for edge in path]
    decision = ActionFusion(self.config).fuse(
        action, page, list({edge.edge_id: edge for edge in edges}.values()),
        # An override replaces the model's action, so it is held to the same
        # relevance as a shortcut: app name excluded, opposite intents
        # rejected, and an object the task names required.
        task_relevance=lambda edge: self._edge_task_relevance(
            edge, _route_relevance_tokens(goal), require_object=True))
    self.metrics.coordinate_corrections += int(decision.coordinate_corrected)
    self.metrics.graph_overrides += int(decision.graph_overrode)
    self.metrics.graph_rejections += int(decision.graph_rejected)
    self.logger.emit("fusion", source=decision.source, confidence=decision.confidence,
                     coordinate_corrected=decision.coordinate_corrected,
                     graph_overrode=decision.graph_overrode,
                     graph_rejected=decision.graph_rejected,
                     disagreement=decision.disagreement, reason=decision.reason)
    return decision

  def high_confidence_edge(self, page: PageObservation, goal: str) -> EdgeEvidence | None:
    if not (self.config.enabled and self.config.high_confidence_skip_enabled):
      return None
    query = _route_relevance_tokens(goal)
    paths = [path for path in self.retrieve_paths(page, goal, top_k=5) if len(path) <= 3]
    candidates = [edge for path in paths for edge in path]
    candidates = [edge for edge in candidates if edge.source_node == self.observe_page(page, visited=False).node_id
                  and _selector_is_actionable(edge.selector)
                  and edge.action_type in {"click", "navigate_back"}
                  and not _UNSAFE_WORDS.search(
                      edge.action_type + " " + edge.selector.describe())
                  and self._edge_task_relevance(edge, query)
                      >= self.config.min_task_relevance
                  and edge.confidence >= self.config.override_confidence
                  and (edge.consistent_validations >= 3 or edge.task_success_count >= 1)
                  and edge.reversible_rate >= 1.0
                  and not edge.ambiguous and not edge.dynamic and not edge.trap_count]
    return max(candidates, key=lambda edge: edge.q_value(task_relevance=1.0)) if candidates else None

  def _route_reversible(self, edge: EdgeEvidence) -> bool:
    if edge.recovery_failure_count:
      return False
    if edge.reversible_rate >= 1.0:
      return True
    return (self.config.route_allow_unknown_reversibility
            and edge.recovery_success_count + edge.recovery_failure_count == 0)

  def target_activities(self, edge: EdgeEvidence) -> Counter[str]:
    activities: Counter[str] = Counter()
    for target_id, count in edge.target_states.items():
      node = self.states.get(target_id)
      activities[node.activity if node is not None else target_id] += count
    return activities

  def route_edge_confidence(self, edge: EdgeEvidence) -> float:
    if self.config.route_landing_level != "activity":
      return edge.confidence
    activities = self.target_activities(edge)
    if not activities or edge.trap_count:
      return 0.0
    stability = max(activities.values()) / max(1, sum(activities.values()))
    return max(0.0, min(1.0, edge.posterior_mean * stability))

  def route_edge_ambiguous(self, edge: EdgeEvidence) -> bool:
    if self.config.route_landing_level != "activity":
      return edge.ambiguous
    return len(self.target_activities(edge)) > 1

  def route_consistent_validations(self, edge: EdgeEvidence) -> int:
    if self.config.route_landing_level != "activity":
      return edge.consistent_validations
    return max(self.target_activities(edge).values(), default=0)

  def route_landed(self, edge: EdgeEvidence, source_node_id: str,
                   landed: StateNode) -> dict[str, Any]:
    """Judge one executed hop at both levels; ``ok`` uses the configured one."""
    node_ok = landed.node_id in edge.target_states
    moved = landed.node_id != source_node_id
    activity_ok = moved and landed.activity in self.target_activities(edge)
    ok = activity_ok if self.config.route_landing_level == "activity" else node_ok
    return {"ok": ok, "node_ok": node_ok, "activity_ok": activity_ok, "moved": moved}

  def route_block_reason(self, path: Sequence[EdgeEvidence]) -> str:
    """First gate a candidate route fails, or "" when it may be executed.

    The conditions and their order are exactly those high_confidence_path has
    always applied; naming the failing one is what turns "0 shortcuts" into a
    gate-blocking distribution that says which requirement is binding.
    """
    if not path:
      return "empty_route"
    if len(path) > 3:
      return "route_too_long"
    for edge in path:
      action_phrase = re.split(
          r"\b(?:reveals|opens|changes|has no stable page delta)\b",
          edge.function, maxsplit=1, flags=re.IGNORECASE)[0]
      boundary_label = (edge.action_type + " " + edge.selector.describe() + " "
                        + edge.selector.resource_id.replace("_", " ") + " "
                        + action_phrase)
      if _COMPLEX_ACTION_WORDS.search(boundary_label):
        return "complex_action_boundary"
      if self.route_edge_confidence(edge) < max(0.82, self.config.override_confidence):
        return "low_confidence"
      if not (self.route_consistent_validations(edge) >= 3
              or edge.task_success_count >= 1 or edge.route_hit_count >= 1):
        return "insufficient_validation"
      if not self._route_reversible(edge):
        return "reversibility_unproven"
      if self.route_edge_ambiguous(edge):
        return "ambiguous_landing"
      if edge.dynamic:
        return "dynamic_edge"
      if edge.trap_count:
        return "trap"
      if edge.route_miss_count:
        return "prior_route_miss"
      if any(self.states.get(target_id) is None
             or self.states[target_id].dynamic_content
             for target_id in edge.target_states):
        return "dynamic_target_state"
      if edge.action_type not in {"click", "navigate_back"}:
        return "unsafe_action_type"
      if not _selector_is_actionable(edge.selector):
        return "selector_not_actionable"
      if _UNSAFE_WORDS.search(boundary_label):
        return "side_effect_risk"
    return ""

  def route_task_block(self, path: Sequence[EdgeEvidence], goal: str) -> str:
    """"task_object_mismatch" unless some hop names an object the task names.

    route_block_reason answers whether a route can be executed safely; this
    answers whether it is this task's route. Retrieval ranks on intent as well
    as objects, which is right for advice the model may ignore and wrong for
    an action taken on its behalf.
    """
    query = _route_relevance_tokens(goal)
    grounded = max((self._edge_task_relevance(edge, query, require_object=True)
                    for edge in path), default=0.0)
    return "" if grounded >= self.config.min_task_relevance else "task_object_mismatch"

  def _route_block(self, path: Sequence[EdgeEvidence], goal: str) -> str:
    return self.route_block_reason(path) or self.route_task_block(path, goal)

  def route_gate_report(self, page: PageObservation, goal: str) -> dict[str, Any]:
    """Blocking distribution over this page's retrieved candidate routes."""
    paths = self.retrieve_paths(page, goal, top_k=max(5, self.config.top_k_paths))
    blocks = Counter(self._route_block(path, goal) or "passed" for path in paths)
    return {"candidates": len(paths), "blocks": dict(blocks)}

  def high_confidence_path(
      self, page: PageObservation, goal: str,
  ) -> list[EdgeEvidence] | None:
    """Return a short, repeatedly verified navigation path, if one is safe."""
    if not (self.config.enabled and self.config.high_confidence_skip_enabled):
      return None
    if self.action_boundary_reason(self._sanitize_page(page)):
      return None
    for path in self.retrieve_paths(page, goal, top_k=max(5, self.config.top_k_paths)):
      prefix = self._navigation_prefix(path)
      if prefix and not self._route_block(prefix, goal):
        return prefix
    return None

  def record_observed_reversal(self, edge_id: str) -> bool:
    """The agent itself undid this edge: it pressed Back from the edge's
    destination and landed on the edge's source. That is direct, observed
    evidence of reversibility, and the only kind an execution-derived edge can
    ever acquire (probes are the only other source of recovery records).
    """
    edge = self.edges.get(edge_id)
    if edge is None:
      return False
    edge.recovery_success_count += 1
    self.logger.emit("observed_reversal", edge_id=edge_id,
                     recovery_success_count=edge.recovery_success_count)
    return True

  def save(self, path: str | Path | None = None) -> None:
    target = Path(path) if path else self.path
    if target is None:
      return
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = self.to_dict()
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
      with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
      os.replace(temp_name, target)
    finally:
      if os.path.exists(temp_name):
        os.unlink(temp_name)
    if self.config.compact_graph_enabled:
      try:
        from android_world.parallel_exploration.lightweight_selector import write_compact_graph
        write_compact_graph(
            target.with_name(f"{target.stem}.compact.json"), self)
      except (ImportError, OSError, TypeError, ValueError) as error:
        self.logger.emit("compact_graph_write_failed", error=str(error))

  def refresh(self) -> None:
    if self.path and self.path.is_file():
      fresh = ExecutableExplorationMemory(self.config, path=self.path, logger=self.logger)
      self.states, self.edges, self._coarse = fresh.states, fresh.edges, fresh._coarse
      self._by_activity, self._outgoing = fresh._by_activity, fresh._outgoing
      self.samples, self.skills = fresh.samples, fresh.skills
      # The authoritative agent and shadow ingester share this file but keep
      # separate in-process metric objects. Merge monotonic counters instead
      # of letting whichever process writes last erase the other's probe or
      # prompt measurements.
      for field in dataclasses.fields(MemoryMetrics):
        name = field.name
        local = getattr(self.metrics, name)
        remote = getattr(fresh.metrics, name)
        if name == "stop_reasons":
          merged = Counter()
          for key in set(local) | set(remote):
            merged[key] = max(int(local.get(key, 0)), int(remote.get(key, 0)))
          setattr(self.metrics, name, merged)
        elif name == "success_steps":
          if len(remote) > len(local):
            setattr(self.metrics, name, list(remote))
        elif name == "extra_time_s":
          setattr(self.metrics, name, max(float(local), float(remote)))
        else:
          setattr(self.metrics, name, max(int(local), int(remote)))

  def to_dict(self) -> dict[str, Any]:
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "states": [dict(dataclasses.asdict(node)) for node in self.states.values()],
        "edges": [
            {**dataclasses.asdict(edge), "selector": dataclasses.asdict(edge.selector),
             "target_states": dict(edge.target_states), "provenance": dict(edge.provenance),
             "task_signatures": dict(edge.task_signatures)}
            for edge in self.edges.values()
        ],
        "samples": [sample.to_dict() for sample in self.samples],
        "skills": [dataclasses.asdict(skill) for skill in self.skills.values()],
        "metrics": self.metrics.to_dict(),
    }

  def _load(self, path: Path) -> None:
    try:
      payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
      return
    if not isinstance(payload, Mapping):
      return
    version = int(payload.get("schema_version", 0) or 0)
    if version > MEMORY_SCHEMA_VERSION:
      return
    raw_metrics = payload.get("metrics")
    if isinstance(raw_metrics, Mapping):
      for field in dataclasses.fields(MemoryMetrics):
        if field.name not in raw_metrics:
          continue
        value = raw_metrics[field.name]
        if field.name == "stop_reasons":
          setattr(self.metrics, field.name, Counter(dict(value or {})))
        elif field.name == "success_steps":
          setattr(self.metrics, field.name, [int(x) for x in (value or ())])
        elif field.name in {"extra_time_s"}:
          setattr(self.metrics, field.name, float(value or 0.0))
        else:
          try:
            setattr(self.metrics, field.name, int(value or 0))
          except (TypeError, ValueError):
            pass
    for raw in payload.get("states", ()):
      if not isinstance(raw, Mapping):
        continue
      try:
        node = StateNode(**{key: raw[key] for key in (
            "node_id", "package", "activity", "structural_signature", "semantic_signature")},
            landmarks=tuple(raw.get("landmarks", ())), elements=tuple(raw.get("elements", ())),
            observation_count=int(raw.get("observation_count", 0)),
            visit_count=int(raw.get("visit_count", 0)), dynamic_variants=int(raw.get("dynamic_variants", 0)),
            parent_state_ids=tuple(str(x) for x in raw.get("parent_state_ids", ())),
            incoming_action_tokens=tuple(str(x) for x in raw.get("incoming_action_tokens", ())),
            depth=int(raw.get("depth", 0)),
            semantic_aliases=tuple(str(x) for x in raw.get("semantic_aliases", ())),
            dynamic_content=bool(raw.get("dynamic_content", False)),
            dynamic_reasons=tuple(str(x) for x in raw.get("dynamic_reasons", ())),
            reversible=bool(raw.get("reversible", False)),
            return_cost_s=float(raw.get("return_cost_s", 0.0)),
            last_seen=float(raw.get("last_seen", time.time())))
      except (KeyError, TypeError, ValueError):
        continue
      # Migrate legacy shorthand in memory without changing the serialized
      # node id: edge references from earlier schema-v1 records remain valid.
      node.activity = _canonical_activity(node.activity)
      self.states[node.node_id] = node
      self._coarse[(node.package, node.activity, node.structural_signature)].append(node.node_id)
      self._by_activity[(node.package, node.activity)].append(node.node_id)
    for raw in payload.get("edges", ()):
      if not isinstance(raw, Mapping):
        continue
      try:
        selector = ElementSelector(**dict(raw.get("selector") or {}))
        provenance = Counter(dict(raw.get("provenance") or {}))
        recovery_success_count = int(raw.get("recovery_success_count", 0))
        if version < 2:
          # Schema v1 accidentally counted authoritative reasoning transitions
          # as successful rollbacks. Remove that synthetic reversibility while
          # retaining actual shadow-probe recovery evidence.
          recovery_success_count = max(
              0, recovery_success_count - int(provenance.get("inference", 0)))
        edge = EdgeEvidence(
            edge_id=str(raw["edge_id"]), source_node=str(raw["source_node"]), selector=selector,
            action_type=str(raw.get("action_type", "")), function=str(raw.get("function", "")),
            target_states=Counter({str(k): int(v) for k, v in dict(raw.get("target_states") or {}).items()}),
            support_count=int(raw.get("support_count", 0)), alpha=float(raw.get("alpha", 1.0)),
            beta=float(raw.get("beta", 1.0)), meaningful_count=int(raw.get("meaningful_count", 0)),
            no_op_count=int(raw.get("no_op_count", 0)), external_count=int(raw.get("external_count", 0)),
            trap_count=int(raw.get("trap_count", 0)), recovery_success_count=recovery_success_count,
            recovery_failure_count=int(raw.get("recovery_failure_count", 0)), task_success_count=int(raw.get("task_success_count", 0)),
            total_cost_s=float(raw.get("total_cost_s", 0.0)), provenance=provenance,
            observed_at_s=float(raw.get("observed_at_s", time.time())), dynamic=bool(raw.get("dynamic", False)),
            normalized_action_token=str(raw.get("normalized_action_token", "")),
            route_hit_count=int(raw.get("route_hit_count", 0)),
            route_miss_count=int(raw.get("route_miss_count", 0)),
            task_signatures=Counter({str(k): int(v) for k, v in
                                     dict(raw.get("task_signatures") or {}).items()}))
      except (KeyError, TypeError, ValueError):
        continue
      self.edges[edge.edge_id] = edge
      self._outgoing[edge.source_node].append(edge.edge_id)
    for raw in payload.get("samples", ()):
      if not isinstance(raw, Mapping):
        continue
      try:
        self.samples.append(KStepSample(
            source_state=str(raw["source_state"]), future_state=str(raw["future_state"]),
            k=int(raw["k"]), first_action=ElementSelector(**dict(raw.get("first_action") or {})),
            action_sequence=tuple(dict(item) for item in raw.get("action_sequence", ())),
            target_states=tuple(str(item) for item in raw.get("target_states", ())),
            return_action=(dict(raw["return_action"]) if raw.get("return_action") else None),
            elapsed_s=float(raw.get("elapsed_s", 0.0)), meaningful=bool(raw.get("meaningful", False)),
            hard_negative=bool(raw.get("hard_negative", False)), evidence=dict(raw.get("evidence") or {}),
            provenance=str(raw.get("provenance", "exploration")),
        ))
      except (KeyError, TypeError, ValueError):
        continue
    for raw in payload.get("skills", ()):
      try:
        skill = SkillGroup(skill_id=str(raw["skill_id"]), action_sequence=tuple(raw.get("action_sequence", ())),
                           support_count=int(raw.get("support_count", 0)), validation_count=int(raw.get("validation_count", 0)),
                           success_count=int(raw.get("success_count", 0)), executable=bool(raw.get("executable", False)),
                           source=str(raw.get("source", "exploration")))
      except (KeyError, TypeError, ValueError):
        continue
      self.skills[skill.skill_id] = skill


def selector_for_action(action: Mapping[str, Any], page: PageObservation) -> ElementSelector:
  """Resolve coordinates to the smallest live clickable semantic control."""
  if action.get("x") is not None and action.get("y") is not None:
    x, y = float(action["x"]), float(action["y"])
    right = max((float((row.get("bbox") or (0, 0, 0, 0))[2])
                 for row in page.elements), default=0.0)
    bottom = max((float((row.get("bbox") or (0, 0, 0, 0))[3])
                  for row in page.elements), default=0.0)
    screen_area = max(1.0, right * bottom)
    matches = []
    for row in page.elements:
      box = row.get("bbox")
      if not box or not (box[0] <= x <= box[2] and box[1] <= y <= box[3]):
        continue
      area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
      if area <= 0 or area >= 0.80 * screen_area:
        continue
      try:
        selector = ElementSelector(**dict(row.get("selector") or {}))
      except TypeError:
        continue
      if not _selector_is_actionable(selector):
        continue
      # Clickable parents are the actual hit targets; within those, prefer
      # the tightest semantic region. Full-screen containers and unlabelled
      # wrappers never become executable memory selectors.
      matches.append((not bool(row.get("clickable")), area, selector.key(), selector))
    if matches:
      return min(matches, key=lambda item: item[:3])[3]
  return ElementSelector(
      text=_norm(action.get("text", "")),
      class_name=_norm(action.get("class_name", "")),
  )


__all__ = [
    "ActionFusion", "EdgeEvidence", "ElementSelector", "ExecutableExplorationMemory",
    "ExecutableMemoryConfig", "FusionDecision", "JsonlMemoryLogger", "KStepSample",
    "MemoryMetrics", "PageObservation", "ProbeCandidate", "ProbeDecision",
    "SafeProbePlanner", "SafeProbePolicy", "SkillGroup", "StateNode", "TransitionRecord",
    "extract_k_step_samples", "mine_action_groups", "selector_for_action",
]
