"""Entity-parameterised procedure replay inside one episode.

The online, task-specific graph. Built from empty in each episode out of the
model's own executed clicks, each tagged with an independent app-state
verdict (app_state_checker). An edge whose pressed control contains one of
the task's obligation entities is stored with the entity abstracted to a
slot, so the route the model worked out for entity 1 becomes a template for
entities 2..n named by the same task.

Contract and boundaries:
  * Obligation entities come from one model call at episode start, kept only
    when they occur verbatim in the goal. No evaluator data, no lexical
    similarity: an entity must match a label on screen exactly.
  * The template is the prefix of a COMPLETED demonstration: an entity hop,
    then fixed hops, ending right before the first hop the checker saw
    change persistent state (the commit). Replay never performs the commit,
    typing, or COMPLETE; the model makes that decision on the landed screen.
  * Every replayed hop must have been certified unchanged-state when the
    model performed it (strict). ``allow_unverified`` is an ablation only.
  * Each hop is relocated on the live screen by an entity-bound selector
    (resource id + class + own label or inner labels) and must be unique;
    landing is checked by activity, and the entity hop's landing must show
    the new entity if the demonstrated landing showed the old one.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _norm(value: Any) -> str:
  return " ".join(str(value or "").casefold().split())


OBLIGATION_PROMPT = (
    "Read the phone task below. List the distinct items (records, files, contacts, "
    "events, recipes, expenses) that the task asks you to act on one by one, copying "
    "each item name EXACTLY as written in the task. Do not include text to type, new "
    'names, or app names. Answer with JSON only: {"items": ["..."]}.\nTask: ')


OBLIGATION_RECORDS_PROMPT = (
    "Read the phone task below. List the distinct items (records, files, contacts, "
    "events, recipes, expenses) that the task asks you to act on one by one. For each "
    "item give its name and every attribute value the task states for it, copying all "
    "text EXACTLY as written in the task. Answer with JSON only: "
    '{"items": [{"name": "...", "fields": {"attribute": "value"}}]}.\nTask: ')


def parse_obligation_records(output: str, goal: str) -> tuple[tuple[str, ...], dict[str, dict[str, str]]]:
  """(entity names, {name: {field: value}}); every value verbatim in the goal."""
  match = re.search(r"\{.*\}", output or "", re.S)
  if not match:
    return (), {}
  try:
    items = json.loads(match.group(0)).get("items")
  except (ValueError, AttributeError):
    return (), {}
  if not isinstance(items, list):
    return (), {}
  goal_n = _norm(goal)
  names, records = [], {}
  for item in items:
    if not isinstance(item, dict):
      continue
    name = str(item.get("name") or "").strip()
    if ":" in name and _norm(name.split(":", 1)[1]) in goal_n:
      name = name.split(":", 1)[1].strip()  # "Expense: Legal Fees" -> "Legal Fees"
    if len(_norm(name)) < 3 or _norm(name) not in goal_n:
      continue
    fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
    kept = {}
    for key, value in fields.items():
      if not isinstance(value, (str, int, float)) or _norm(value) not in goal_n:
        continue
      # The model types "67.41" for "$67.41": keep the form one types.
      value = re.sub(r"^[^\w]+|[^\w]+$", "", str(value).strip())
      if value and _norm(value) != _norm(name):
        kept[str(key)] = value
    names.append(name)
    records[name] = kept
  names = list(dict.fromkeys(names))[:12]
  return tuple(names), {n: records[n] for n in names}


def parse_obligations(output: str, goal: str) -> tuple[str, ...]:
  """Entities the model named, kept only if they occur verbatim in the goal."""
  match = re.search(r"\{.*\}", output or "", re.S)
  if not match:
    return ()
  try:
    items = json.loads(match.group(0)).get("items")
  except (ValueError, AttributeError):
    return ()
  if not isinstance(items, list):
    return ()
  goal_n = _norm(goal)
  kept = []
  for item in items:
    if isinstance(item, str) and len(_norm(item)) >= 3 and _norm(item) in goal_n:
      kept.append(item.strip())
  return tuple(dict.fromkeys(kept))[:12]


def _get(element: Any, *keys: str, default: Any = None) -> Any:
  for key in keys:
    value = element.get(key) if isinstance(element, Mapping) else getattr(element, key, None)
    if value is not None:
      return value
  return default


def _inside(inner: Sequence[int], outer: Sequence[int]) -> bool:
  return (len(inner) == 4 and len(outer) == 4 and outer[0] <= inner[0] and outer[1] <= inner[1]
          and inner[2] <= outer[2] and inner[3] <= outer[3])


def inner_labels(element: Any, elements: Sequence[Any]) -> tuple[str, ...]:
  """Own text/description plus every label drawn inside the element's bounds."""
  bounds = tuple(_get(element, "bounds", default=()))
  labels = []
  for other in elements:
    if other is element or _inside(tuple(_get(other, "bounds", default=())), bounds):
      for key in ("text", "content_desc"):
        value = str(_get(other, key, default="") or "").strip()
        if value and len(value) <= 120:
          labels.append(value)
  return tuple(dict.fromkeys(labels))


@dataclasses.dataclass(frozen=True)
class Hop:
  step: int
  source_activity: str
  resource_id: str
  class_name: str
  own_label: str
  inner: tuple[str, ...]
  entity: str  # obligation entity bound by this press, "" for a fixed hop
  target_activity: str
  target_labels: tuple[str, ...]
  state_ok: bool | None  # checker: True unchanged, False changed, None unknown
  barrier: bool = False  # typing, swipe, answer: ends any procedure
  action_type: str = "click"  # click or long_press, replayed as demonstrated
  # Record slot (--procedure_records): this hop typed or pressed the value of
  # the demonstrated entity's field `field`; replay uses the next entity's.
  field: str = ""
  direction: str = ""  # swipe hops: replayed as the same-direction swipe
  # How the model wrapped the field value when it typed it ("…day." for the
  # value "…day"), reproduced around the next entity's value.
  affix: tuple[str, str] = ("", "")
  # Where the demonstration pressed (centre of the control). Only the naive
  # coordinate-UTG ablation replays by it.
  xy: tuple[int, ...] = ()

  @property
  def key_label(self) -> str:
    """The label that names a fixed control, own label first."""
    if self.own_label:
      return self.own_label
    return self.inner[0] if self.inner else ""


def make_hop(step: int, source: Any, element: Any, landed: Any,
             entities: Sequence[str], state_ok: bool | None,
             action_type: str = "click") -> Hop:
  elements = tuple(_get(source, "elements", default=()))
  labels = inner_labels(element, elements)
  bound = [e for e in entities if _norm(e) in {_norm(x) for x in labels}]
  own = str(_get(element, "text", default="") or _get(element, "content_desc", default="") or "")
  landed_labels = tuple(dict.fromkeys(
      str(_get(x, key, default="") or "") for x in _get(landed, "elements", default=())
      for key in ("text", "content_desc") if _get(x, key, default="")))
  return Hop(
      step=step,
      source_activity=_get(_get(source, "activity"), "component", default=""),
      resource_id=str(_get(element, "resource_id", default="") or ""),
      class_name=str(_get(element, "class_name", default="") or ""),
      own_label=own.strip(), inner=labels,
      entity=bound[0] if len(bound) == 1 else "",
      target_activity=_get(_get(landed, "activity"), "component", default=""),
      target_labels=landed_labels, state_ok=state_ok, action_type=action_type,
      xy=_center_xy(element))


def _center_xy(element: Any) -> tuple[int, ...]:
  b = tuple(_get(element, "bounds", default=()) or ())
  return (round((b[0] + b[2]) / 2), round((b[1] + b[3]) / 2)) if len(b) == 4 else ()


def _naive() -> bool:
  return os.environ.get("ANDROID_WORLD_NAIVE_GRAPH") == "1"


def barrier(step: int) -> Hop:
  return Hop(step, "", "", "", "", (), "", "", (), None, barrier=True)


@dataclasses.dataclass(frozen=True)
class Plan:
  entity: str
  template_entity: str
  hops: tuple[Hop, ...]
  values: Mapping[str, str] = dataclasses.field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {"entity": self.entity, "template_entity": self.template_entity,
            "hops": [dataclasses.asdict(h) for h in self.hops]}


class ProcedureGraph:
  """Episode-local executed hops plus the obligation entities they bind.

  A demonstration is the run of hops after the previous commit (or start, or
  barrier) up to the next commit. Its entity hops - a press on a control
  that contains the entity, or typing exactly the entity - are slots; every
  other hop is fixed. Replay substitutes the next undone entity.
  """

  def __init__(self, entities: Sequence[str], *, allow_unverified: bool = False,
               max_hops: int = 6):
    self.entities = tuple(entities)
    self.allow_unverified = allow_unverified
    self.max_hops = max_hops
    self.hops: list[Hop] = []
    self.done: set[str] = set()
    # Replays that stopped early: never retried automatically, and NOT done.
    self.failed: set[str] = set()
    # An entity a replay has started on and nobody has committed yet. No new
    # replay may begin until the checker sees a commit: a replay that stopped
    # early on 'Legal Fees' was followed by one for 'Stationery' that typed
    # over the unsaved Legal Fees form (2026-10-04, ExpenseAddMultiple).
    self.open_entity = ""
    self.records: dict[str, dict[str, str]] = {}
    self.last_rejection = ""

  def set_records(self, records: Mapping[str, Mapping[str, str]]) -> None:
    """Per-entity field values from the contract, e.g. amount, category."""
    self.records = {_norm(k): {f: str(v) for f, v in dict(fields).items() if str(v).strip()}
                    for k, fields in records.items()}

  def _field_values(self) -> set[str]:
    return {_canon(v) for fields in self.records.values() for v in fields.values()}

  def _is_boundary(self, hop: Hop) -> bool:
    """Barriers, commits, and typing anything that is neither an entity nor
    (with records) one of the contract's field values."""
    if hop.barrier or hop.state_ok is False:
      return True
    if hop.action_type == "input_text" and not hop.entity:
      return not (self.records and _canon(hop.inner[0] if hop.inner else "") in self._field_values())
    return False

  def _with_fields(self, run: Sequence[Hop], entity: str) -> list[Hop]:
    """Mark hops whose typed text or pressed label is a field of `entity`."""
    fields = self.records.get(_norm(entity), {})
    if not fields:
      return list(run)
    by_value = {}
    for key, value in fields.items():
      by_value.setdefault(_canon(value), key)
    out = []
    for hop in run:
      key, affix = "", ("", "")
      if not hop.entity and not hop.barrier:
        if hop.action_type == "input_text":
          typed = hop.inner[0] if hop.inner else ""
          key = by_value.get(_canon(typed), "")
          if key:
            core = re.sub(r"^[^\w]+|[^\w]+$", "", typed.strip())
            start = typed.find(core) if core else -1
            if start >= 0:
              affix = (typed[:start], typed[start + len(core):])
        else:
          hits = [by_value[_canon(x)] for x in hop.inner if _canon(x) in by_value]
          key = hits[0] if len(set(hits)) == 1 else ""
      out.append(dataclasses.replace(hop, field=key, affix=affix) if key else hop)
    return out

  def set_entities(self, entities: Sequence[str]) -> None:
    """Bind a contract issued after hops were recorded (lazy obligation call).

    Labels and typed text were stored with every hop, so binding is
    recomputed exactly as if the entities had been known at the time.
    """
    self.entities = tuple(entities)
    self.hops = [h if h.barrier else dataclasses.replace(h, entity=_bind(h.inner, self.entities))
                 for h in self.hops]
    self.done = {_norm(h.entity) for h in self.hops if h.entity}

  def has_completed_demonstration_candidate(self) -> bool:
    """A non-commit hop followed by a commit: worth one contract call."""
    seen = False
    for hop in self.hops:
      if hop.barrier:
        seen = False
      elif hop.state_ok is False:
        if seen:
          return True
      else:
        seen = True
    return False

  def record(self, hop: Hop) -> None:
    self.hops.append(hop)
    if hop.entity:
      self.done.add(_norm(hop.entity))
    if hop.state_ok is False:
      self.open_entity = ""

  def opened(self, entity: str) -> None:
    """A replay acted on `entity`; it stays open until a commit is recorded."""
    self.open_entity = _norm(entity)

  def mark_done(self, entity: str) -> None:
    self.done.add(_norm(entity))

  def mark_failed(self, entity: str) -> None:
    self.failed.add(_norm(entity))

  def template(self) -> tuple[Hop, ...] | None:
    """Navigation part of the latest completed single-entity demonstration."""
    commit = None
    for index in range(len(self.hops) - 1, -1, -1):
      hop = self.hops[index]
      if hop.barrier:
        continue
      if hop.state_ok is False:
        commit = index
        # Walk back to the previous boundary.
        start = index
        while start > 0 and not self._is_boundary(self.hops[start - 1]):
          start -= 1
        run = self.hops[start:commit]
        entities = {_norm(h.entity) for h in run if h.entity}
        if len(entities) != 1:
          continue  # e.g. the commit of a replayed entity: look further back
        demonstrated = next(h.entity for h in run if h.entity)
        run = self._with_fields(run, demonstrated)
        # A typed value that is not a slot cannot be replayed for another
        # entity: the template is unusable rather than wrong.
        if any(h.action_type == "input_text" and not h.entity and not h.field for h in run):
          self.last_rejection = "typed_value_not_a_field"
          return None
        # Hops that cannot be replayed BEFORE the entity's sub-procedure are
        # lead-in (e.g. an exploratory tap) and are trimmed; after the first
        # entity hop they invalidate the template.
        return self._validated(run)
    self.last_rejection = ("demonstration_not_completed" if any(h.entity for h in self.hops)
                           else "no_entity_demonstration")
    return None

  def _validated(self, run: Sequence[Hop]) -> tuple[Hop, ...] | None:
    """Validate a trimmed run (recursion ends: each trim drops >= 1 hop)."""
    if not any(h.entity for h in run):
      self.last_rejection = "no_entity_after_trim"
      return None
    first_entity = next(i for i, h in enumerate(run) if h.entity)
    for position, hop in enumerate(run):
      bad = ((hop.state_ok is None and not self.allow_unverified)
             or (hop.action_type == "input_text" and not hop.resource_id)
             or (not hop.entity and not hop.field and hop.action_type != "swipe"
                 and not (hop.key_label or hop.resource_id)))
      if bad:
        if position < first_entity:
          return self._validated(run[position + 1:])
        self.last_rejection = ("uncertified_hop" if hop.state_ok is None and not self.allow_unverified
                               else "unnamed_field" if hop.action_type == "input_text"
                               else "unnamed_fixed_hop")
        return None
    run = tuple(run[-self.max_hops:])
    if not any(h.entity for h in run):
      self.last_rejection = "too_long"
      return None
    return run

  def _next_entities(self) -> list[str]:
    return [e for e in self.entities if _norm(e) not in self.done and _norm(e) not in self.failed]

  def plan(self, screen: Any) -> tuple[Plan, Any] | None:
    """Pick the shortest replayable suffix of the template for this screen.

    1. The entity is already on screen as the template's entity press: start
       there (no search). It must be unique.
    2. Otherwise start at the earliest hop whose source is this screen's
       activity, provided the entity is SUPPLIED before it is pressed (typed
       into a field), so its location is the template's own sub-procedure.
    """
    self.last_rejection = ""
    template = self.template()
    if not template:
      return None
    if self.open_entity:
      self.last_rejection = "entity_in_progress"
      return None
    undone = self._next_entities()
    if not undone:
      self.last_rejection = "no_undone_entity"
      return None
    # Only the next entity in the task's own order. Skipping ahead to one that
    # happens to be visible breaks the model's progress narrative: it then
    # confirmed 'Pet Supplies' believing it was 'Unexpected Expenses', the
    # last item, and declared the task complete with one item left.
    undone = undone[:1]
    activity = _get(_get(screen, "activity"), "component", default="")
    template_entity = next(h.entity for h in template if h.entity)
    values = self.records.get(_norm(undone[0]), {})
    needed = {h.field for h in template if h.field}
    if needed - set(values):
      self.last_rejection = "record_missing_field"
      return None
    elements = tuple(_get(screen, "elements", default=()))
    for index, hop in enumerate(template):
      if hop.entity and hop.action_type != "input_text" and hop.source_activity == activity:
        for entity in undone:
          hits = _entity_controls(hop, entity, elements)
          if len(hits) == 1:
            return Plan(entity, template_entity, template[index:], values), hits[0]
          if len(hits) > 1:
            self.last_rejection = "entity_ambiguous"
            return None
    # Latest feasible start first: the shortest replay whose first control is
    # on this screen and whose entity is typed (supplied) before any press.
    reasons = []
    for index in range(len(template) - 1, -1, -1):
      hop = template[index]
      if hop.source_activity != activity:
        continue
      suffix = template[index:]
      first_entity = next((h for h in suffix if h.entity), None)
      if first_entity is None:
        reasons.append("no_entity_in_suffix")
        continue
      if first_entity.action_type != "input_text":
        reasons.append("entity_not_visible")
        continue
      element = (relocate_field(hop, screen) if hop.action_type == "input_text"
                 else relocate_fixed(hop, screen))
      if element is None:
        reasons.append("selector_missing_or_ambiguous")
        continue
      return Plan(undone[0], template_entity, suffix, values), element
    if reasons:
      self.last_rejection = reasons[-1]
      return None
    self.last_rejection = "not_on_template_source"
    return None


def _boundary(hop: Hop) -> bool:
  """Barriers, commits, and typing anything other than an obligation entity."""
  return (hop.barrier or hop.state_ok is False
          or (hop.action_type == "input_text" and not hop.entity))


def _canon(value: Any) -> str:
  """Normalised text without outer punctuation, for value matching."""
  return re.sub(r"^[^\w]+|[^\w]+$", "", _norm(value))


def _bind(inner: Sequence[str], entities: Sequence[str]) -> str:
  bound = [e for e in entities if _norm(e) in {_norm(x) for x in inner}]
  return bound[0] if len(bound) == 1 else ""


def _entity_controls(hop: Hop, entity: str, elements: Sequence[Any]) -> list[Any]:
  return [e for e in elements if _get(e, "clickable", default=False)
          and str(_get(e, "resource_id", default="") or "") == hop.resource_id
          and str(_get(e, "class_name", default="") or "") == hop.class_name
          and _norm(entity) in {_norm(x) for x in inner_labels(e, elements)}]


def make_swipe_hop(step: int, source: Any, direction: str, landed: Any,
                   state_ok: bool | None) -> Hop:
  """A scroll inside a procedure (e.g. down a form to its category list)."""
  return Hop(step=step,
             source_activity=_get(_get(source, "activity"), "component", default=""),
             resource_id="", class_name="", own_label="", inner=(), entity="",
             target_activity=_get(_get(landed, "activity"), "component", default=""),
             target_labels=(), state_ok=state_ok, action_type="swipe", direction=direction)


def make_typed_hop(step: int, source: Any, field: Any, text: str, landed: Any,
                   entities: Sequence[str], state_ok: bool | None) -> Hop:
  """Typing is a slot only when the text IS an obligation entity."""
  hop = make_hop(step, source, field, landed, (), state_ok, action_type="input_text")
  return dataclasses.replace(hop, inner=(text,), own_label="",
                             entity=_bind((text,), entities))


def relocate_fixed(hop: Hop, screen: Any) -> Any | None:
  """The unique clickable with the demonstrated id, class and naming label.

  An unlabelled icon is named by its resource id alone; uniqueness on the
  live screen is still required, so a per-row icon repeated down a list is
  refused rather than guessed.
  """
  elements = tuple(_get(screen, "elements", default=()))
  key = _norm(hop.key_label)
  hits = []
  for element in elements:
    if not _get(element, "clickable", default=False):
      continue
    if (str(_get(element, "resource_id", default="") or "") != hop.resource_id
        or str(_get(element, "class_name", default="") or "") != hop.class_name):
      continue
    labels = {_norm(x) for x in inner_labels(element, elements)}
    if (key and key in labels) or (not key and hop.resource_id):
      hits.append(element)
  return hits[0] if len(hits) == 1 else None


def _on_screen(element: Any, size: Sequence[int] | None) -> bool:
  """Fully inside the display: a control still sliding in during a
  transition has bounds past the edge, and a tap there does nothing."""
  if not size:
    return True
  b = tuple(_get(element, "bounds", default=()))
  return len(b) == 4 and b[0] >= 0 and b[1] >= 0 and b[2] <= size[0] and b[3] <= size[1]


def relocate_field(hop: Hop, screen: Any) -> Any | None:
  """A text field is named by id and class; its content is what changes."""
  hits = [e for e in tuple(_get(screen, "elements", default=()))
          if str(_get(e, "resource_id", default="") or "") == hop.resource_id
          and str(_get(e, "class_name", default="") or "") == hop.class_name]
  return hits[0] if len(hits) == 1 and hop.resource_id else None


def center_click(element: Any, action_type: str = "click") -> dict[str, Any]:
  left, top, right, bottom = _get(element, "bounds")
  return {"action_type": action_type, "x": round((left + right) / 2),
          "y": round((top + bottom) / 2)}


def _option_family(hop: Hop, elements: Sequence[Any]) -> list[Any]:
  return [e for e in elements if _get(e, "clickable", default=False)
          and str(_get(e, "resource_id", default="") or "") == hop.resource_id
          and str(_get(e, "class_name", default="") or "") == hop.class_name]


def _scroll_to_option(hop: Hop, value: str, screen: Any, capture: Callable[[], Any],
                      click: Callable[[Mapping[str, Any]], None],
                      actions: list[Mapping[str, Any]],
                      screen_size: Sequence[int] | None,
                      max_swipes: int = 3) -> tuple[Any | None, Any]:
  """Find an option chosen by value that is not on screen yet.

  Options picked by value (a category chip, a list choice) live in a list the
  demonstration may never have scrolled: 'Social' was visible, 'Others' was
  two swipes to the right, and the replay stopped there in all three runs
  (r4-r6 ExpenseAddMultiple). Drag the option row toward the next options,
  then back, at most max_swipes each way, and stop as soon as the option
  family (same id and class as the demonstrated option) leaves the screen or
  the activity changes: then the drag moved something other than the list.
  """
  if not value or not hop.resource_id:
    return None, screen
  family = _option_family(hop, tuple(_get(screen, "elements", default=())))
  if not family:
    return None, screen
  # Drag along the option row itself, well inside the screen. AndroidWorld's
  # direction swipe starts at the screen edge (x=0 for "left"), which gesture
  # navigation reads as Back: in r7 the first version closed the unsaved
  # Legal Fees form instead of scrolling the category chips.
  rows = [tuple(_get(e, "bounds")) for e in family]
  y = round(sum((b[1] + b[3]) / 2 for b in rows) / len(rows))
  width = (screen_size[0] if screen_size else max(b[2] for b in rows))
  near, far = round(width * 0.8), round(width * 0.2)
  for reveal in ("next", "previous"):
    start, end = (near, far) if reveal == "next" else (far, near)
    for _ in range(max_swipes if reveal == "next" else 2 * max_swipes):
      action = {"action_type": "swipe_xy", "start": [start, y], "end": [end, y]}
      try:
        click(action)
        screen = capture()
      except Exception:  # pylint: disable=broad-exception-caught
        return None, screen
      actions.append(action)
      if _get(_get(screen, "activity"), "component", default="") != hop.source_activity:
        return None, screen
      elements = tuple(_get(screen, "elements", default=()))
      if not _option_family(hop, elements):
        return None, screen
      hits = [e for e in _entity_controls(hop, value, elements) if _on_screen(e, screen_size)]
      if len(hits) == 1:
        return hits[0], screen
      if len(hits) > 1:
        return None, screen
  return None, screen


@dataclasses.dataclass(frozen=True)
class Replay:
  entity: str
  attempted: int
  completed: int
  reason: str
  actions: tuple[Mapping[str, Any], ...]


def execute(plan: Plan, first_element: Any, *, capture: Callable[[], Any],
            click: Callable[[Mapping[str, Any]], None],
            on_hop: Callable[[Hop, Mapping[str, Any]], None] | None = None,
            retries: int = 3, sleep: Callable[[float], None] = time.sleep,
            screen_size: Sequence[int] | None = None) -> Replay:
  """Replay the navigation part for a new entity, checking every landing.

  Stops at the first mismatch and returns control from the actual screen;
  a real press is never pretended away. ``click`` executes any action dict.
  """
  actions: list[Mapping[str, Any]] = []
  done = 0
  element = first_element
  for index, hop in enumerate(plan.hops):
    if hop.action_type == "swipe":
      if index > 0:
        try:
          screen = capture()
        except Exception:  # pylint: disable=broad-exception-caught
          return Replay(plan.entity, len(actions), done, "capture_failed", tuple(actions))
        if _get(_get(screen, "activity"), "component", default="") != hop.source_activity:
          return Replay(plan.entity, len(actions), done, "source_mismatch", tuple(actions))
      action = {"action_type": "swipe", "direction": hop.direction}
      actions.append(action)
      if on_hop is not None:
        on_hop(hop, action)
      try:
        click(action)
        landed = capture()
      except Exception:  # pylint: disable=broad-exception-caught
        return Replay(plan.entity, len(actions), done, "execution_failed", tuple(actions))
      if _get(_get(landed, "activity"), "component", default="") != hop.target_activity:
        return Replay(plan.entity, len(actions), done, "landing_activity_mismatch", tuple(actions))
      done += 1
      continue
    if _naive() and hop.xy and hop.action_type != "swipe":
      # Naive coordinate UTG: press where the demonstration pressed.
      x, y = hop.xy
      element = {"bounds": (x - 1, y - 1, x + 1, y + 1)}
    elif index > 0:
      element = None
      for attempt in range(retries):
        try:
          screen = capture()
        except Exception:  # pylint: disable=broad-exception-caught
          return Replay(plan.entity, len(actions), done, "capture_failed", tuple(actions))
        if _get(_get(screen, "activity"), "component", default="") != hop.source_activity:
          element = None
        elif hop.entity and hop.action_type != "input_text":
          hits = _entity_controls(hop, plan.entity, tuple(_get(screen, "elements", default=())))
          element = hits[0] if len(hits) == 1 else None
        elif hop.field and hop.action_type != "input_text":
          # An option chosen by value (e.g. a category): press the one whose
          # label is THIS entity's value for the same field.
          hits = _entity_controls(hop, plan.values.get(hop.field, ""),
                                  tuple(_get(screen, "elements", default=())))
          element = hits[0] if len(hits) == 1 else None
        elif hop.action_type == "input_text":
          element = relocate_field(hop, screen)
        else:
          element = relocate_fixed(hop, screen)
        if element is not None and not _on_screen(element, screen_size):
          element = None
        if element is not None:
          break
        if attempt + 1 < retries:
          sleep(0.4)  # a transition still animating; look again
      if _get(_get(screen, "activity"), "component", default="") != hop.source_activity:
        return Replay(plan.entity, len(actions), done, "source_mismatch", tuple(actions))
      if element is None and hop.field and hop.action_type != "input_text":
        element, screen = _scroll_to_option(hop, plan.values.get(hop.field, ""), screen,
                                            capture, click, actions, screen_size)
      if element is None:
        reason = ("entity_not_found" if hop.entity and hop.action_type != "input_text"
                  else "field_missing_or_ambiguous" if hop.action_type == "input_text"
                  else "selector_missing_or_ambiguous")
        return Replay(plan.entity, len(actions), done, reason, tuple(actions))
    action = center_click(element, hop.action_type)
    if hop.action_type == "input_text":
      # The demonstration typed into the field as it then was; a field left
      # holding the previous entity must be cleared first, or the text is
      # appended (observed: "Chicken Alfredo PastaTomato Basil Bruschetta").
      text = (plan.entity if hop.entity
              else (hop.affix[0] + plan.values.get(hop.field, "") + hop.affix[1]
                    if plan.values.get(hop.field) else "") if hop.field else hop.inner[0])
      if not text:
        return Replay(plan.entity, len(actions), done, "missing_field_value", tuple(actions))
      action = {**action, "text": text, "clear_text": True}
    actions.append(action)
    if on_hop is not None:
      on_hop(hop, action)
    try:
      click(action)
      landed = capture()
    except Exception:  # pylint: disable=broad-exception-caught
      return Replay(plan.entity, len(actions), done, "execution_failed", tuple(actions))
    if _get(_get(landed, "activity"), "component", default="") != hop.target_activity:
      return Replay(plan.entity, len(actions), done, "landing_activity_mismatch", tuple(actions))
    if (hop.entity and hop.action_type != "input_text" and plan.template_entity
        and _norm(plan.template_entity) in {_norm(x) for x in hop.target_labels}):
      landed_labels = {_norm(_get(x, key, default="")) for x in _get(landed, "elements", default=())
                       for key in ("text", "content_desc")}
      if _norm(plan.entity) not in landed_labels:
        return Replay(plan.entity, len(actions), done, "landing_entity_mismatch", tuple(actions))
    done += 1
  return Replay(plan.entity, len(actions), done, "stopped_before_commit", tuple(actions))
