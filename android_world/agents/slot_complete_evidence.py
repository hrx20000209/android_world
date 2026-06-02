"""Lightweight slot-complete evidence extraction for MobileExplorer.

This module intentionally avoids model calls. It uses task text plus visible
UI labels to decide whether an exploration observation contains a complete
answer-worthy fact, or merely partial context that should stay out of prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Any


def clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\u2013", "-").replace("\u2014", "-").split())


def _low(value: Any) -> str:
    return clean(value).lower()


def _quoted(text: str) -> list[str]:
    return [clean(x) for x in re.findall(r"['\"]([^'\"]+)['\"]", text) if clean(x)]


def _tokens(text: str) -> set[str]:
    stop = {
        "android", "world", "answer", "format", "express", "with", "only", "what",
        "which", "when", "where", "have", "many", "count", "total", "simple",
        "calendar", "opentracks", "joplin", "notes", "tasks", "task", "activity",
        "activities", "event", "events", "recipe", "ingredient", "amount",
        "quantity", "duration", "distance", "folder", "app", "open",
    }
    return {t for t in re.findall(r"[a-z0-9]{3,}", _low(text)) if t not in stop}


def _amount_unit_re() -> re.Pattern[str]:
    return re.compile(
        r"\b(?P<amount>(?:\d+\s*/\s*\d+)|(?:\d+(?:\.\d+)?)|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s*"
        r"(?P<unit>tsp|teaspoons?|tbsp|tablespoons?|cups?|oz|ounces?|g|grams?|kg|ml|l|pinch|cloves?)\b",
        re.IGNORECASE,
    )


DATE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+20\d{2})?\b|"
    r"\b\d{1,2}/\d{1,2}/20\d{2}\b|"
    r"\b20\d{2}-\d{1,2}-\d{1,2}\b",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b", re.IGNORECASE)
TIME_RANGE_RE = re.compile(
    r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\s*-\s*\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b",
    re.IGNORECASE,
)
DURATION_RE = re.compile(
    r"\b\d+\s*h\s*\d+\s*m\b|\b\d+(?:\.\d+)?\s*(?:min|mins|minute|minutes|h|hr|hrs|hour|hours)\b|"
    r"\b\d{1,2}[:.]\d{2}(?::\d{2})?\b",
    re.IGNORECASE,
)
DISTANCE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:m|meter|meters|km|kilometer|kilometers|mi|mile|miles)\b",
    re.IGNORECASE,
)
CATEGORY_WORDS = ("skiing", "kayaking", "running", "walking", "rowing", "cycling", "hiking")
WEEKDAY_WORDS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
CATEGORY_ALIASES = {
    "skiing": ("skiing", "ski"),
    "kayaking": ("kayaking", "kayak"),
    "running": ("running", "run"),
    "walking": ("walking", "walk"),
    "rowing": ("rowing", "row"),
    "cycling": ("cycling", "cycle", "bike", "biking"),
    "hiking": ("hiking", "hike"),
}


def _date_keys(text: str) -> set[str]:
    low = clean(text).lower()
    keys: set[str] = set()
    month_to_num = {
        "jan": "01", "january": "01",
        "feb": "02", "february": "02",
        "mar": "03", "march": "03",
        "apr": "04", "april": "04",
        "may": "05",
        "jun": "06", "june": "06",
        "jul": "07", "july": "07",
        "aug": "08", "august": "08",
        "sep": "09", "sept": "09", "september": "09",
        "oct": "10", "october": "10",
        "nov": "11", "november": "11",
        "dec": "12", "december": "12",
    }
    for match in re.finditer(
        r"\b(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
        r"(?P<day>\d{1,2})(?:,?\s+(?P<year>20\d{2}))?\b",
        low,
    ):
        mon = month_to_num.get(match.group("mon").rstrip("."))
        day = f"{int(match.group('day')):02d}"
        year = match.group("year") or ""
        keys.add(f"{year}-{mon}-{day}" if year else f"{mon}-{day}")
    for match in re.finditer(r"\b(?P<mon>\d{1,2})/(?P<day>\d{1,2})/(?P<year>20\d{2})\b", low):
        keys.add(f"{match.group('year')}-{int(match.group('mon')):02d}-{int(match.group('day')):02d}")
    for match in re.finditer(r"\b(?P<year>20\d{2})-(?P<mon>\d{1,2})-(?P<day>\d{1,2})\b", low):
        keys.add(f"{match.group('year')}-{int(match.group('mon')):02d}-{int(match.group('day')):02d}")
    return keys


def _date_scope_matches(text: str, goal: str) -> bool:
    goal_keys = _date_keys(goal)
    text_keys = _date_keys(text)
    if goal_keys:
        # Match either exact year-month-day or month-day when one side omits year.
        goal_suffixes = {key[-5:] for key in goal_keys}
        text_suffixes = {key[-5:] for key in text_keys}
        return bool(goal_keys.intersection(text_keys) or goal_suffixes.intersection(text_suffixes))
    goal_low = goal.lower()
    required_weekdays = {day for day in WEEKDAY_WORDS if day in goal_low}
    if required_weekdays:
        text_low = text.lower()
        return any(day in text_low for day in required_weekdays)
    return True


def _category_in_text(category: str, text: str) -> bool:
    low = text.lower()
    return any(alias in low for alias in CATEGORY_ALIASES.get(category, (category,)))


def _best_matching_date(text: str, goal: str) -> str:
    if not _date_scope_matches(text, goal):
        return ""
    matches = DATE_RE.findall(text)
    if isinstance(matches, list) and matches:
        first = matches[0]
        return clean(first if isinstance(first, str) else " ".join(first))
    for day in WEEKDAY_WORDS:
        if day in goal.lower() and day in text.lower():
            return day
    return ""


@dataclass
class TaskSlots:
    task_mode: str
    task_subtype: str
    required_slots: list[str]
    target_entities: list[str]
    metric_type: str = ""
    item_type: str = ""
    app_family: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_task_slots(task_instruction: str) -> TaskSlots:
    goal = clean(task_instruction)
    low = goal.lower()
    quoted = _quoted(goal)
    entities = quoted + sorted(_tokens(goal))
    app_family = ""
    if "opentracks" in low or "sports tracker" in low:
        app_family = "opentracks"
    elif "simple calendar" in low or re.search(r"\b(events?|meetings?)\b", low):
        app_family = "calendar"
    elif "joplin" in low or re.search(r"\b(recipe|ingredient|todo|to-do|note)\b", low):
        app_family = "joplin"
    elif re.search(r"\btasks?\s+app\b|\btodo\b|\bto-do\b|high priority|due\b", low):
        app_family = "tasks"

    question_like = bool(
        re.search(r"\b(how many|how long|what|which|who|where|when|answer with|count|total|longest|next meeting)\b", low)
        or low.rstrip().endswith("?")
    )
    media_capture = bool(
        re.search(
            r"\b(take (?:one |a |the )?(?:photo|picture|video)|record (?:an? )?(?:audio|video|clip)|"
            r"capture (?:an? )?(?:photo|picture|video)|start recording)\b",
            low,
        )
    )
    explicit_form = bool(
        re.search(r"\b(create|add|edit|rename|input|type|new folder|new contact|draft)\b", low)
    )

    if re.search(r"\b(delete|remove|clear|turn on|turn off|toggle|grant permission)\b", low):
        return TaskSlots("DELETE_COMMIT", "DELETE_COMMIT", ["target_item", "risky_action_present"], entities, app_family=app_family)
    if media_capture:
        return TaskSlots("SIMPLE_VERIFY_OPEN", "MEDIA_CAPTURE", ["target_app_or_state"], entities, app_family=app_family)
    if explicit_form and not question_like:
        return TaskSlots("FORM_CREATE_EDIT", "FORM_CREATE_EDIT", ["field_labels", "submit_button"], entities, app_family=app_family)
    if re.search(r"\b(open app|verify|run\s+(?:the\s+)?stopwatch)\b", low):
        return TaskSlots("SIMPLE_VERIFY_OPEN", "SIMPLE_VERIFY_OPEN", ["target_app_or_state"], entities, app_family=app_family)

    if re.search(r"\b(quantity|amount|unit|ingredient)\b", low):
        return TaskSlots(
            "INFO_QUERY_COUNT",
            "RECIPE_INGREDIENT",
            ["recipe_title", "ingredient_name", "amount_value", "unit"],
            entities,
            metric_type="amount_unit",
            item_type="ingredient",
            app_family=app_family or "joplin",
        )
    if app_family == "opentracks" or re.search(r"\b(duration|distance|activity type|activities?)\b", low):
        metric = "count"
        required = ["metric_type", "metric_value"]
        if "distance" in low:
            metric = "distance"
            required = ["metric_type", "metric_value", "unit"]
        elif re.search(r"\bduration|how long|minutes?\b", low):
            metric = "duration"
            required = ["metric_type", "metric_value", "unit"]
        elif re.search(r"\bwhat activities|activity type\b", low):
            metric = "activity_type"
            required = ["date", "category"]
        elif re.search(r"\bhow many|count\b", low):
            metric = "count"
            required = ["count_value"]
        if any(cat in low for cat in CATEGORY_WORDS):
            required.append("category")
        if DATE_RE.search(goal) or re.search(r"\bweek|today|yesterday|interval|between|from|to\b", low):
            required.append("date")
        return TaskSlots("INFO_QUERY_COUNT", "ACTIVITY_STATS", _dedupe(required), entities, metric_type=metric, item_type="activity", app_family=app_family or "opentracks")
    if app_family == "calendar" or re.search(r"\b(events?|meetings?|next meeting)\b", low):
        required = ["event_title"]
        if DATE_RE.search(goal) or re.search(r"\bfriday|monday|tuesday|wednesday|thursday|saturday|sunday|today|tomorrow\b", low):
            required.append("date")
        if TIME_RE.search(goal) or "time range" in low or "between" in low:
            required.append("event_time")
        if re.search(r"\bwith [A-Z][a-z]+\b|person|meeting with\b", goal):
            required.append("person")
        return TaskSlots("INFO_QUERY_COUNT", "EVENT_QUERY", _dedupe(required), entities, metric_type="event", item_type="event", app_family=app_family or "calendar")
    if re.search(r"\bis\b.*\b(todo|to-do|complete|completed|checked)\b", low):
        return TaskSlots("INFO_QUERY_COUNT", "BOOLEAN_STATUS", ["target_entity", "status_type", "status_value"], entities, item_type="todo", app_family=app_family)
    if re.search(r"\bhow many|count|which|what tasks|what events|todo items?\b", low):
        required = ["count_value"]
        if "which" in low or "titles" in low:
            required = ["item_titles"]
        if DATE_RE.search(goal) or "next week" in low or "due" in low:
            required.append("scope_entity")
        return TaskSlots("INFO_QUERY_COUNT", "COUNT_LIST", _dedupe(required), entities, metric_type="count", item_type="list_item", app_family=app_family)
    if re.search(r"\b(open|find|search)\b", low):
        return TaskSlots("NAVIGATION_SEARCH", "NAVIGATION_SEARCH", ["target_entity", "current_location"], entities, app_family=app_family)
    return TaskSlots("INFO_QUERY_COUNT", "COUNT_LIST", ["item_titles"], entities, app_family=app_family)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def infer_screen_role(labels: list[str], activity: str = "") -> str:
    text = " ".join(clean(x) for x in labels).lower() + " " + activity.lower()
    if "markers" in text:
        return "MarkerList"
    if re.search(r"\bfilter|create new filter|date shortcut|today|tomorrow|next sun\b", text):
        return "FilterOrDatePicker"
    if re.search(r"\bnew event|event name|save event|event creation\b", text):
        return "EventCreationForm"
    if re.search(r"\bsearch\b", text):
        return "SearchScreen"
    if re.search(r"\bstats|statistics|distance|duration|speed|pace\b", text) and "opentracks" in text:
        return "TrackStats"
    if re.search(r"\btrack detail|track list|activity\b", text) and "opentracks" in text:
        return "TrackDetail"
    if re.search(r"\bcheckbox|checked|unchecked|high priority|due\b", text):
        return "TaskList"
    if re.search(r"\bcalendar|event\b", text):
        return "CalendarList"
    if re.search(r"\brecipe|ingredient|note\b", text):
        return "NoteDetail"
    return "Unknown"


def extract_slot_evidence(
    labels: list[str],
    task_instruction: str,
    *,
    activity: str = "",
    screen_role: str = "",
) -> dict[str, Any]:
    slots = parse_task_slots(task_instruction)
    cleaned = [clean(x) for x in labels if clean(x)]
    role = screen_role or infer_screen_role(cleaned, activity)
    extracted: dict[str, Any] = {}
    facts: list[str] = []
    hard_negative = ""
    wrong_role = False

    if slots.task_subtype == "RECIPE_INGREDIENT":
        _extract_recipe(cleaned, task_instruction, slots, extracted, facts)
    elif slots.task_subtype == "ACTIVITY_STATS":
        hard_negative = _extract_opentracks(cleaned, task_instruction, slots, extracted, facts, role)
    elif slots.task_subtype == "EVENT_QUERY":
        hard_negative, wrong_role = _extract_calendar(cleaned, task_instruction, slots, extracted, facts, role)
    elif slots.task_subtype in {"COUNT_LIST", "BOOLEAN_STATUS"} and slots.app_family == "tasks":
        hard_negative, wrong_role = _extract_tasks(cleaned, task_instruction, slots, extracted, facts, role)
    elif slots.task_subtype in {"COUNT_LIST", "BOOLEAN_STATUS"}:
        _extract_generic_list(cleaned, task_instruction, slots, extracted, facts)
    elif slots.task_mode == "FORM_CREATE_EDIT":
        _extract_form_schema(cleaned, extracted, facts)
    elif slots.task_mode == "DELETE_COMMIT":
        _extract_risk(cleaned, task_instruction, extracted, facts)

    present = {slot: _slot_present(extracted.get(slot)) for slot in slots.required_slots}
    present_count = sum(1 for ok in present.values() if ok)
    total = max(1, len(slots.required_slots))
    complete = present_count == len(slots.required_slots)
    missing = [slot for slot, ok in present.items() if not ok]
    hint_type = "NONE"
    if hard_negative:
        hint_type = "AVOID_HINT"
        facts = [hard_negative]
    elif slots.task_mode == "INFO_QUERY_COUNT" and complete and facts:
        hint_type = "ANSWER_HINT"
    elif slots.task_mode == "FORM_CREATE_EDIT" and present_count >= 2:
        hint_type = "SCHEMA_HINT"
    elif slots.task_mode == "DELETE_COMMIT" and extracted.get("risky_action_present"):
        hint_type = "RISK_HINT"

    return {
        "task_slots": slots.to_dict(),
        "screen_role": role,
        "extracted_slots": extracted,
        "missing_slots": missing,
        "slot_presence": present,
        "slot_coverage": round(present_count / total, 4),
        "slot_complete": bool(complete),
        "answer_reconstructable": bool(complete and hint_type == "ANSWER_HINT"),
        "hint_type": hint_type,
        "facts": facts[:6],
        "hard_negative": hard_negative,
        "wrong_screen_role": bool(wrong_role),
        "reason_if_not_reconstructable": "" if complete else f"missing_slots={','.join(missing)}",
    }


def _slot_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(clean(value))
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def answer_facts_if_slot_complete(labels: list[str], task_instruction: str, *, activity: str = "", limit: int = 6) -> list[str]:
    evidence = extract_slot_evidence(labels, task_instruction, activity=activity)
    if evidence.get("hint_type") in {"ANSWER_HINT", "AVOID_HINT"}:
        return list(evidence.get("facts") or [])[:limit]
    return []


def _extract_recipe(labels: list[str], goal: str, slots: TaskSlots, extracted: dict[str, Any], facts: list[str]) -> None:
    low_goal = goal.lower()
    quoted = _quoted(goal)
    recipe = quoted[0] if quoted else ""
    if not recipe:
        recipe_match = re.search(r"\brecipe\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,4})", goal)
        if recipe_match:
            recipe = clean(recipe_match.group(1))
    ingredient_match = re.search(r"\b(?:quantity|amount)\s+of\s+([a-zA-Z][a-zA-Z0-9 _-]{1,40})\b", low_goal)
    ingredient = clean(ingredient_match.group(1) if ingredient_match else "")
    ingredient = re.sub(r"\b(do|i|need|for|the|recipe|in|app|express|answer)\b.*$", "", ingredient).strip()
    if not ingredient:
        for token in _tokens(goal):
            if token not in {"joplin", "recipe"}:
                ingredient = token
                break
    joined = " ".join(labels).lower()
    if recipe and recipe.lower() in joined:
        extracted["recipe_title"] = recipe
    if ingredient:
        extracted["ingredient_name"] = ingredient
    amount_re = _amount_unit_re()
    for idx, label in enumerate(labels):
        window = " ".join(labels[max(0, idx - 3) : min(len(labels), idx + 4)])
        if ingredient and ingredient.lower() not in window.lower():
            continue
        match = amount_re.search(window)
        if match:
            extracted["amount_value"] = clean(match.group("amount"))
            extracted["unit"] = clean(match.group("unit"))
            if not extracted.get("recipe_title") and recipe:
                extracted["recipe_title"] = recipe
            facts.append(
                f'In "{recipe or "target recipe"}", ingredient "{ingredient}" appears with amount '
                f'"{extracted["amount_value"]} {extracted["unit"]}".'
            )
            return


def _extract_opentracks(
    labels: list[str],
    goal: str,
    slots: TaskSlots,
    extracted: dict[str, Any],
    facts: list[str],
    role: str,
) -> str:
    goal_low = goal.lower()
    joined = " ".join(labels)
    joined_low = joined.lower()
    if "markers" in joined_low and "marker" not in goal_low:
        return 'Avoid "Markers": it is not useful for duration/distance/activity statistics.'
    extracted["metric_type"] = slots.metric_type
    goal_categories = [cat for cat in CATEGORY_WORDS if cat in goal_low]
    if goal_categories:
        extracted["category"] = goal_categories[0]
    elif any(_category_in_text(cat, joined_low) for cat in CATEGORY_WORDS):
        extracted["category"] = next(cat for cat in CATEGORY_WORDS if _category_in_text(cat, joined_low))
    best_date = _best_matching_date(joined, goal)
    if best_date:
        extracted["date"] = best_date
    for idx, label in enumerate(labels):
        window = " ".join(labels[max(0, idx - 8) : min(len(labels), idx + 9)])
        wlow = window.lower()
        if goal_categories and not any(_category_in_text(cat, wlow) for cat in goal_categories):
            continue
        if "date" in slots.required_slots and not _date_scope_matches(window, goal):
            continue
        if slots.metric_type == "duration":
            match = DURATION_RE.search(window)
            if match:
                if "date" in slots.required_slots and not extracted.get("date"):
                    extracted["date"] = _best_matching_date(window, goal)
                extracted["metric_value"] = match.group(0)
                extracted["unit"] = "duration"
                facts.append(f"OpenTracks duration fact: {window}; duration={match.group(0)}.")
                return ""
        elif slots.metric_type == "distance":
            match = DISTANCE_RE.search(window)
            if match:
                if "date" in slots.required_slots and not extracted.get("date"):
                    extracted["date"] = _best_matching_date(window, goal)
                parts = match.group(0).split()
                extracted["metric_value"] = parts[0] if parts else match.group(0)
                extracted["unit"] = parts[1] if len(parts) > 1 else "distance"
                facts.append(f"OpenTracks distance fact: {window}; distance={match.group(0)}.")
                return ""
        elif slots.metric_type == "activity_type":
            cats = [cat for cat in CATEGORY_WORDS if _category_in_text(cat, wlow)]
            if cats:
                extracted["category"] = cats[0]
                if _best_matching_date(window, goal) or extracted.get("date"):
                    extracted["date"] = extracted.get("date") or _best_matching_date(window, goal)
                    facts.append(f"OpenTracks activity type fact: {window}; activity_type={cats[0]}.")
                    return ""
        elif slots.metric_type == "count":
            valid_rows = _valid_activity_rows(labels, goal)
            if valid_rows:
                extracted["count_value"] = len(valid_rows)
                facts.append(f"OpenTracks visible activity count: {len(valid_rows)}; rows: {', '.join(valid_rows[:6])}.")
                return ""
    if role in {"FilterOrDatePicker", "SearchScreen"}:
        # Partial scope evidence, but not an answer.
        facts.append(f"Partial OpenTracks scope screen: {role}.")
    return ""


def _valid_activity_rows(labels: list[str], goal: str) -> list[str]:
    goal_low = goal.lower()
    rows: list[str] = []
    for label in labels:
        low = label.lower()
        if not any(cat in low for cat in CATEGORY_WORDS):
            continue
        if "marker" in low or "search" in low or "filter" in low:
            continue
        if DATE_RE.search(label) or DURATION_RE.search(label) or DISTANCE_RE.search(label) or any(cat in goal_low for cat in CATEGORY_WORDS):
            rows.append(label)
    return _dedupe(rows)


def _extract_calendar(labels: list[str], goal: str, slots: TaskSlots, extracted: dict[str, Any], facts: list[str], role: str) -> tuple[str, bool]:
    goal_low = goal.lower()
    joined = " ".join(labels)
    if role == "EventCreationForm" or re.search(r"\bnew event|event name|save event\b", joined.lower()):
        return ('Avoid "New Event": it opens event creation, not existing event results.', True)
    requested_people = [
        x for x in re.findall(r"\b[A-Z][a-z]{2,}\b", goal)
        if x.lower() not in {
            "simple", "calendar", "answer", "pro", "january", "february", "march",
            "april", "june", "july", "august", "september", "october", "november",
            "december",
        }
        and x.lower() not in set(WEEKDAY_WORDS)
    ]
    requested_times = TIME_RE.findall(goal)
    best_date = _best_matching_date(joined, goal)
    if best_date:
        extracted["date"] = best_date
    for idx, label in enumerate(labels):
        window = " ".join(labels[max(0, idx - 3) : min(len(labels), idx + 4)])
        if "new event" in window.lower():
            continue
        time_range = TIME_RANGE_RE.search(window)
        times = TIME_RE.findall(window)
        people_ok = not requested_people or any(p.lower() in window.lower() for p in requested_people)
        date_ok = "date" not in slots.required_slots or _date_scope_matches(f"{window} {joined}", goal)
        time_ok = (
            not requested_times
            or any(t.lower().replace(".", ":") in window.lower().replace(".", ":") for t in requested_times)
            or ("between" in goal_low and bool(time_range))
        )
        title = _event_title_from_window(window, labels, idx)
        if title and people_ok and date_ok and (time_ok or not requested_times):
            extracted["event_title"] = title
            if times or time_range:
                extracted["event_time"] = time_range.group(0) if time_range else times[0]
            if requested_people:
                extracted["person"] = requested_people[0]
            if "date" in slots.required_slots and not extracted.get("date") and _best_matching_date(window, goal):
                extracted["date"] = _best_matching_date(window, goal)
            if "date" in slots.required_slots and not extracted.get("date"):
                for day in WEEKDAY_WORDS:
                    if day in goal_low and day in window.lower():
                        extracted["date"] = day
                        break
            facts.append(f"Calendar event fact: title={title}; time={extracted.get('event_time', '')}; context={window}.")
            return "", False
    return "", False


def _event_title_from_window(window: str, labels: list[str], idx: int) -> str:
    ignored = re.compile(r"\b(calendar|search|settings|more options|change view|today|month|week|day|october|november)\b", re.I)
    for candidate in labels[max(0, idx - 3) : min(len(labels), idx + 2)]:
        if len(candidate) >= 3 and not ignored.search(candidate) and not TIME_RE.fullmatch(candidate):
            return candidate
    if TIME_RE.search(window):
        parts = [p.strip(" ,;") for p in re.split(TIME_RE, window) if clean(p)]
        for part in parts:
            if len(part) >= 3 and not ignored.search(part):
                return clean(part)[:80]
    return ""


def _extract_tasks(labels: list[str], goal: str, slots: TaskSlots, extracted: dict[str, Any], facts: list[str], role: str) -> tuple[str, bool]:
    joined_low = " ".join(labels).lower()
    if re.search(r"\b(create new filter|display name|name cannot be empty|color row|icon row|date picker|today|tomorrow|next sun)\b", joined_low):
        return ("Avoid filter/date-picker/form pages when the task asks for actual task result rows.", True)
    if not re.search(r"\b(checkbox|checked|unchecked|due|priority|task|todo|to-do)\b", joined_low):
        return "", False
    rows = []
    shortcut_re = re.compile(
        r"\b(today|tomorrow|yesterday|next\s+\w+|this week|next week|overdue|no date|"
        r"date picker|filter|sort|search|settings|create|display name|color|icon|save|cancel)\b",
        re.IGNORECASE,
    )
    for label in labels:
        low = label.lower()
        if len(label) < 3:
            continue
        if shortcut_re.search(label):
            continue
        if re.fullmatch(r"\d{1,2}[:.]\d{2}\s*(?:am|pm)?", low):
            continue
        looks_like_row = bool(
            re.search(r"\b(checkbox|checked|unchecked|due|priority|task|todo|to-do)\b", low)
            or (len(label) >= 5 and not DATE_RE.search(label) and not TIME_RE.search(label))
        )
        if looks_like_row:
            rows.append(label)
    rows = _dedupe(rows)
    if rows:
        if "item_titles" in slots.required_slots:
            extracted["item_titles"] = rows[:8]
        extracted["count_value"] = len(rows)
        if "scope_entity" in slots.required_slots:
            scope_bits = []
            if "high priority" in goal.lower() and "priority" in joined_low:
                scope_bits.append("high priority")
            if "next week" in goal.lower() and ("due" in joined_low or DATE_RE.search(" ".join(labels))):
                scope_bits.append("next week")
            if DATE_RE.search(goal) and _date_scope_matches(" ".join(labels), goal):
                scope_bits.append(_best_matching_date(" ".join(labels), goal) or "date")
            if scope_bits:
                extracted["scope_entity"] = " ".join(scope_bits)
        facts.append(f"Task result rows: {', '.join(rows[:8])}; count={len(rows)}.")
    return "", False


def _extract_generic_list(labels: list[str], goal: str, slots: TaskSlots, extracted: dict[str, Any], facts: list[str]) -> None:
    ignored = re.compile(r"\b(search|settings|more options|navigate up|toolbar|content|main|calendar|filter|create|help)\b", re.I)
    rows = [x for x in labels if len(x) >= 3 and not ignored.search(x)]
    rows = _dedupe(rows[:12])
    if rows:
        if "item_titles" in slots.required_slots:
            extracted["item_titles"] = rows[:8]
        extracted["count_value"] = len(rows)
        if "scope_entity" in slots.required_slots:
            extracted["scope_entity"] = "visible list"
        facts.append(f"Visible result rows: {', '.join(rows[:8])}; count={len(rows)}.")


def _extract_form_schema(labels: list[str], extracted: dict[str, Any], facts: list[str]) -> None:
    fields = [x for x in labels if re.search(r"\b(name|title|date|time|phone|email|amount|description|note|folder|file|field)\b", x.lower())]
    submit = [x for x in labels if re.search(r"\b(add|create|save|done|ok|submit|next)\b", x.lower())]
    if fields:
        extracted["field_labels"] = fields[:6]
    if submit:
        extracted["submit_button"] = submit[0]
    if fields:
        facts.append(f"Form fields: {', '.join(fields[:6])}; submit={submit[0] if submit else ''}.")


def _extract_risk(labels: list[str], goal: str, extracted: dict[str, Any], facts: list[str]) -> None:
    target = _quoted(goal)
    if target:
        extracted["target_item"] = target[0]
    risky = [x for x in labels if re.search(r"\b(delete|remove|confirm|save|send|allow|turn on|turn off)\b", x.lower())]
    if risky:
        extracted["risky_action_present"] = risky[0]
        facts.append(f"Risk action visible: {risky[0]}.")
