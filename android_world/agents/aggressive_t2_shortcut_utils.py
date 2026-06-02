"""Pure helpers for aggressive t+2 shortcut planning diagnostics.

These helpers deliberately avoid AndroidWorld/emulator dependencies so the
shortcut planner can be unit-tested before running benchmark episodes.
"""

from __future__ import annotations

import re
from typing import Any


SAFE_SEARCH_TOKENS = re.compile(r"\b(search|query|clear query|submit query|search plate|search_src_text|search_plate)\b", re.I)
FORM_FIELD_TOKENS = re.compile(r"\b(body|message|note body|description|amount|phone|email|contact|expense|title|name)\b", re.I)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_task_search_entity(task: str) -> str:
    text = clean_text(task)
    quoted = re.findall(r'"([^"]{2,80})"|\'([^\']{2,80})\'', text)
    for left, right in quoted:
        phrase = clean_text(left or right)
        if phrase:
            return phrase
    filename = re.search(r"\b([A-Za-z0-9_.-]+\.(?:html|md|txt|mp3|m4a|jpg|png|pdf))\b", text)
    if filename:
        return clean_text(filename.group(1))
    recipe = re.search(r"\bfor\s+([A-Z][A-Za-z0-9' -]{2,60}?)(?:\?|\.|,|$)", text)
    if recipe:
        return clean_text(recipe.group(1))
    date = re.search(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:,\s*|\s+)20\d{2})\b",
        text,
        flags=re.I,
    )
    if date:
        return clean_text(date.group(1).replace(",", ""))
    meeting = re.search(r"\bmeeting\s+with\s+([A-Z][a-z]{2,})\b", text)
    if meeting:
        return clean_text(meeting.group(1))
    for token in ("kayaking", "skiing", "running", "walking", "cycling", "hiking", "rowing"):
        if re.search(rf"\b{token}\b", text, flags=re.I):
            return token
    return ""


def state_has_search_ui(state: dict[str, Any] | None) -> bool:
    if not isinstance(state, dict):
        return False
    labels = " ".join(clean_text(x) for x in state.get("anchor_labels", []) + state.get("visible_targets", []))
    elements = state.get("ui_elements") or []
    for element in elements:
        if isinstance(element, dict):
            labels += " " + " ".join(clean_text(element.get(k)) for k in ("text", "content_description", "hint_text", "resource_id"))
    return bool(SAFE_SEARCH_TOKENS.search(labels))


def is_safe_search_input(task: str, state: dict[str, Any] | None, value: str, field: dict[str, Any] | None = None) -> bool:
    value = clean_text(value)
    if not value:
        return False
    entity = extract_task_search_entity(task)
    if entity and value.lower() != entity.lower():
        value_tokens = set(re.findall(r"[a-z0-9]{3,}", value.lower()))
        entity_tokens = set(re.findall(r"[a-z0-9]{3,}", entity.lower()))
        overlap = len(value_tokens & entity_tokens) / float(max(1, len(value_tokens | entity_tokens)))
        if overlap < 0.8:
            return False
    field_text = ""
    if isinstance(field, dict):
        field_text = " ".join(clean_text(field.get(k)) for k in ("text", "content_description", "hint_text", "resource_id", "class_name"))
        if FORM_FIELD_TOKENS.search(field_text) and not SAFE_SEARCH_TOKENS.search(field_text):
            return False
    return bool(state_has_search_ui(state) or SAFE_SEARCH_TOKENS.search(field_text))


def classify_evidence(label: str, task: str) -> str:
    label_low = clean_text(label).lower()
    task_low = clean_text(task).lower()
    if any(token in label_low for token in ("search", "filter", "stats", "statistics", "detail")):
        return "PARTIAL_PROGRESS"
    if "markers" in label_low and "markers" not in task_low:
        return "AVOID_HINT"
    if "new event" in label_low and any(token in task_low for token in ("event", "meeting", "next")):
        return "AVOID_HINT"
    if "privacy" in label_low or "help" in label_low:
        return "AVOID_HINT"
    return "NONE"


def build_shortcut_plan_for_test(
    *,
    task: str,
    planned_action: dict[str, Any],
    explored_t1_state: dict[str, Any],
    step_t: int = 1,
) -> dict[str, Any]:
    action_type = clean_text(planned_action.get("action_type")).lower()
    label = clean_text(planned_action.get("label") or planned_action.get("text"))
    merged = f"{action_type} {label}".lower()
    if action_type in {"long_press", "longpress"} or any(token in merged for token in ("delete", "confirm", "save")):
        return {"task_id": task, "step_t": step_t, "confidence": 0.0, "no_plan_reason": "risky_action"}
    task_low = task.lower()
    if any(token in task_low for token in ("add contact", "create contact", "add the following expenses")):
        return {"task_id": task, "step_t": step_t, "confidence": 0.0, "no_plan_reason": "unsafe_form_input"}
    entity = extract_task_search_entity(task)
    if state_has_search_ui(explored_t1_state) and entity:
        safe = is_safe_search_input(task, explored_t1_state, entity, {"resource_id": "search_src_text", "hint_text": "Search"})
        return {
            "task_id": task,
            "episode_id": task[:80],
            "step_t": step_t,
            "planned_root_action": {
                "action_type": planned_action.get("action_type", "click"),
                "label": label,
                "bbox": planned_action.get("bbox"),
                "resource_id": planned_action.get("resource_id"),
                "class_name": planned_action.get("class_name"),
                "value": planned_action.get("value"),
                "operator": "SearchPeek",
                "risk_level": "low",
            },
            "explored_t1_state": explored_t1_state,
            "shortcut_t2_action": {
                "action_type": "type",
                "label": f'type "{entity}" into search',
                "bbox": None,
                "resource_id": "search_src_text",
                "class_name": "EditText",
                "value": entity,
                "operator": "SearchPeek",
                "risk_level": "low",
                "is_safe_search_input": bool(safe),
                "expected_missing_slot_gain": 1.0,
            },
            "explored_t2_state": {
                "package": explored_t1_state.get("package"),
                "activity": explored_t1_state.get("activity"),
                "screen_role": "search_results_or_query",
                "p_hash": None,
                "anchor_labels": [entity],
                "filled_slots": ["target_entity"],
                "missing_slots": [],
            },
            "confidence": 0.90 if safe else 0.0,
            "no_plan_reason": "" if safe else "unsafe_search_input",
        }
    return {"task_id": task, "step_t": step_t, "confidence": 0.0, "no_plan_reason": "no_safe_t2_search_input"}
