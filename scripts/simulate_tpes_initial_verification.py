#!/usr/bin/env python3
"""Initial TPES verification simulation on synthetic AndroidWorld-like tasks.

This script does not call a VLM. It uses deterministic random numbers,
token-overlap scoring, accessibility-like labels, and simple rollback rules.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import dataclasses
import datetime as dt
import json
from pathlib import Path
import random
import re
import statistics
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "tpes_initial_verification"
MAX_EPISODE_STEPS = 10

STOPWORDS = {
    "a",
    "an",
    "and",
    "app",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "new",
    "of",
    "on",
    "or",
    "the",
    "to",
    "turn",
    "using",
    "with",
}


@dataclasses.dataclass(frozen=True)
class Action:
    kind: str
    target: str
    value: str = ""

    @property
    def label(self) -> str:
        return f"{self.kind} {self.target} {self.value}".strip()


@dataclasses.dataclass(frozen=True)
class UIElement:
    element_id: str
    label: str
    role: str
    bbox: list[int]
    risk: float
    outcomes: list[str]
    children: list[str] = dataclasses.field(default_factory=list)
    tags: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class SyntheticTask:
    task_id: str
    task_type: str
    baseline_success: bool
    baseline_steps: int
    goal: str
    initial_a11y_labels: list[str]
    elements: list[UIElement]
    baseline_actions: list[Action]
    oracle_actions: list[Action]
    failure_mode: str


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_+:.]+", text.lower())
    return {w for w in words if len(w) > 1 and w not in STOPWORDS}


def _similarity(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _action_match(action: Action, labels: list[str], threshold: float = 0.24) -> bool:
    action_text = action.label
    return any(_similarity(action_text, label) >= threshold for label in labels)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return statistics.mean(values)


def _element(
    element_id: str,
    label: str,
    role: str,
    bbox: list[int],
    risk: float,
    outcomes: list[str],
    children: list[str] | None = None,
    tags: list[str] | None = None,
) -> UIElement:
    return UIElement(
        element_id=element_id,
        label=label,
        role=role,
        bbox=bbox,
        risk=risk,
        outcomes=outcomes,
        children=children or [],
        tags=tags or [],
    )


def _action(kind: str, target: str, value: str = "") -> Action:
    return Action(kind=kind, target=target, value=value)


def _make_taskbed() -> list[SyntheticTask]:
    """Build 20 synthetic tasks: 10 baseline successes and 10 failures."""

    def t(
        task_id: str,
        task_type: str,
        baseline_success: bool,
        baseline_steps: int,
        goal: str,
        labels: list[str],
        elements: list[UIElement],
        baseline_actions: list[Action],
        oracle_actions: list[Action],
        failure_mode: str = "none",
    ) -> SyntheticTask:
        return SyntheticTask(
            task_id=task_id,
            task_type=task_type,
            baseline_success=baseline_success,
            baseline_steps=min(baseline_steps, MAX_EPISODE_STEPS),
            goal=goal,
            initial_a11y_labels=labels,
            elements=elements,
            baseline_actions=baseline_actions,
            oracle_actions=oracle_actions,
            failure_mode=failure_mode,
        )

    return [
        t(
            "S01_AudioRecorderRecordAudio",
            "data_entry",
            True,
            4,
            "Record an audio clip using Audio Recorder and save it.",
            ["Audio Recorder", "Record", "Settings", "Recordings"],
            [
                _element("record", "Record button", "primary", [450, 1900, 620, 2070], 0.06, ["Recording screen", "Stop button", "Save recording"], ["Stop button", "Save recording"]),
                _element("settings", "Recording settings", "navigation", [850, 120, 980, 240], 0.18, ["Format", "Quality"]),
                _element("list", "Recordings list", "content", [80, 360, 980, 880], 0.08, ["Saved recordings"]),
            ],
            [_action("open", "Audio Recorder"), _action("click", "Record button"), _action("click", "Stop button"), _action("click", "Save recording")],
            [_action("open", "Audio Recorder"), _action("click", "Record button"), _action("click", "Stop button"), _action("click", "Save recording")],
        ),
        t(
            "S02_ContactsAddContact",
            "data_entry",
            True,
            5,
            "Create a contact Hugo Pereira with phone +13920741751.",
            ["Contacts", "Create contact", "Search contacts", "Favorites"],
            [
                _element("add", "Create contact", "primary", [870, 1860, 1030, 2040], 0.05, ["New contact form", "Name field", "Phone field", "Save"], ["Name field", "Phone field", "Save"]),
                _element("search", "Search contacts", "navigation", [720, 120, 830, 230], 0.08, ["Search field"]),
                _element("favorite", "Favorites", "content", [80, 360, 900, 500], 0.12, ["Favorite contacts"]),
            ],
            [_action("open", "Contacts"), _action("click", "Create contact"), _action("type", "Name field", "Hugo Pereira"), _action("type", "Phone field", "+13920741751"), _action("click", "Save")],
            [_action("open", "Contacts"), _action("click", "Create contact"), _action("type", "Name field", "Hugo Pereira"), _action("type", "Phone field", "+13920741751"), _action("click", "Save")],
        ),
        t(
            "S03_MarkorCreateFolder",
            "data_edit",
            True,
            4,
            "Create a new folder in Markor named folder_alpha.",
            ["Markor", "Files", "New file or folder", "Search"],
            [
                _element("new", "New file or folder", "primary", [890, 1940, 1040, 2100], 0.06, ["New dialog", "File", "Folder", "Name input"], ["Name input", "Folder button"]),
                _element("search", "Search notes", "navigation", [820, 130, 920, 230], 0.08, ["Search input"]),
                _element("sort", "Sort files", "navigation", [700, 130, 800, 230], 0.10, ["Sort by name"]),
            ],
            [_action("open", "Markor"), _action("click", "New file or folder"), _action("type", "Name input", "folder_alpha"), _action("click", "Folder button")],
            [_action("open", "Markor"), _action("click", "New file or folder"), _action("type", "Name input", "folder_alpha"), _action("click", "Folder button")],
        ),
        t(
            "S04_NotesTodoItemCount",
            "information_retrieval",
            True,
            3,
            "How many to-dos are in the Ideas folder in Joplin? Answer as a number.",
            ["Joplin", "Ideas", "Recipe Ideas", "Financial Plan", "Work Tasks"],
            [
                _element("ideas", "Ideas folder", "content", [60, 280, 420, 400], 0.04, ["Recipe Ideas checkbox", "Financial Plan checkbox", "Work Tasks checkbox", "3 to-dos"], ["3 to-dos"]),
                _element("search", "Search notes", "navigation", [840, 120, 950, 230], 0.05, ["Search query"]),
                _element("new", "New note", "primary", [890, 1930, 1040, 2090], 0.12, ["Editor"]),
            ],
            [_action("open", "Joplin"), _action("click", "Ideas folder"), _action("answer", "3")],
            [_action("open", "Joplin"), _action("click", "Ideas folder"), _action("answer", "3")],
        ),
        t(
            "S05_SystemWifiTurnOnVerify",
            "settings_verification",
            True,
            3,
            "Turn Wi-Fi on.",
            ["Quick settings", "Wi-Fi", "Bluetooth", "Settings"],
            [
                _element("wifi", "Wi-Fi toggle", "primary", [90, 360, 380, 650], 0.18, ["Wi-Fi on", "Connected"], ["Connected"]),
                _element("bt", "Bluetooth toggle", "risky", [430, 360, 720, 650], 0.22, ["Bluetooth on"]),
                _element("settings", "Settings app", "navigation", [760, 120, 980, 260], 0.15, ["Network settings"]),
            ],
            [_action("slide", "Quick settings"), _action("click", "Wi-Fi toggle"), _action("status", "done")],
            [_action("slide", "Quick settings"), _action("click", "Wi-Fi toggle"), _action("status", "done")],
        ),
        t(
            "S06_SimpleSmsSend",
            "data_entry",
            True,
            5,
            "Send an SMS to Alex saying running late.",
            ["Messages", "Start chat", "Search", "Conversations"],
            [
                _element("compose", "Start chat", "primary", [840, 1840, 1030, 2040], 0.06, ["Recipient field", "Message field", "Send button"], ["Recipient field", "Message field", "Send button"]),
                _element("search", "Search messages", "navigation", [760, 120, 900, 230], 0.08, ["Search query"]),
                _element("thread", "Alex conversation", "content", [60, 360, 980, 520], 0.10, ["Message field", "Send button"]),
            ],
            [_action("open", "Messages"), _action("click", "Start chat"), _action("type", "Recipient field", "Alex"), _action("type", "Message field", "running late"), _action("click", "Send button")],
            [_action("open", "Messages"), _action("click", "Start chat"), _action("type", "Recipient field", "Alex"), _action("type", "Message field", "running late"), _action("click", "Send button")],
        ),
        t(
            "S07_SimpleCalendarEventsOnDate",
            "screen_reading",
            True,
            3,
            "What events are on June 12 in Simple Calendar?",
            ["Simple Calendar", "June 12", "Dentist 09:00", "Dinner 19:00"],
            [
                _element("date", "June 12 day cell", "content", [90, 520, 260, 690], 0.04, ["Dentist 09:00", "Dinner 19:00"], ["Dentist 09:00", "Dinner 19:00"]),
                _element("search", "Search events", "navigation", [820, 120, 940, 230], 0.06, ["Search query"]),
                _element("add", "Add event", "primary", [890, 1920, 1040, 2080], 0.12, ["New event form"]),
            ],
            [_action("open", "Simple Calendar"), _action("click", "June 12 day cell"), _action("answer", "Dentist and Dinner")],
            [_action("open", "Simple Calendar"), _action("click", "June 12 day cell"), _action("answer", "Dentist and Dinner")],
        ),
        t(
            "S08_ExpenseDeleteSingle",
            "data_edit",
            True,
            4,
            "Delete the expense named Taxi Ride in Pro Expense.",
            ["Pro Expense", "Taxi Ride", "Coffee", "Add"],
            [
                _element("taxi", "Taxi Ride expense row", "content", [60, 520, 980, 680], 0.18, ["Expense detail", "Delete button"], ["Delete button", "Confirm delete"], ["destructive_goal"]),
                _element("add", "Add expense", "primary", [890, 1880, 1040, 2040], 0.08, ["Add expense form"]),
                _element("search", "Search expenses", "navigation", [790, 120, 920, 230], 0.06, ["Search query"]),
            ],
            [_action("open", "Pro Expense"), _action("click", "Taxi Ride expense row"), _action("click", "Delete button"), _action("click", "Confirm delete")],
            [_action("open", "Pro Expense"), _action("click", "Taxi Ride expense row"), _action("click", "Delete button"), _action("click", "Confirm delete")],
        ),
        t(
            "S09_OsmAndFavorite",
            "search",
            True,
            5,
            "Add a favorite for coordinates 47.1303814, 9.5930117 in OsmAnd.",
            ["OsmAnd", "Search", "Map", "Favorites"],
            [
                _element("search", "Search coordinates", "primary", [60, 120, 900, 260], 0.10, ["Search field", "Result 47.1303814 9.5930117", "Add favorite"], ["Result 47.1303814 9.5930117", "Add favorite"]),
                _element("map", "Map area", "risky", [0, 360, 1080, 1800], 0.30, ["Map panned"], ["Marker menu"], ["dynamic_map"]),
                _element("fav", "Favorites", "navigation", [760, 1800, 980, 2020], 0.10, ["Favorite list"]),
            ],
            [_action("open", "OsmAnd"), _action("click", "Search coordinates"), _action("type", "Search field", "47.1303814, 9.5930117"), _action("click", "Result 47.1303814 9.5930117"), _action("click", "Add favorite")],
            [_action("open", "OsmAnd"), _action("click", "Search coordinates"), _action("type", "Search field", "47.1303814, 9.5930117"), _action("click", "Result 47.1303814 9.5930117"), _action("click", "Add favorite")],
        ),
        t(
            "S10_RecipeDeleteConstraint",
            "data_edit",
            True,
            4,
            "Delete duplicate recipe Tomato Soup but keep the newest one.",
            ["Broccoli", "Tomato Soup", "Tomato Soup duplicate", "Sort"],
            [
                _element("old", "Older Tomato Soup duplicate", "content", [60, 580, 980, 760], 0.28, ["Recipe detail", "Delete recipe", "Confirm"], ["Delete recipe", "Confirm"], ["destructive_goal"]),
                _element("new", "Newest Tomato Soup", "content", [60, 360, 980, 540], 0.20, ["Recipe detail", "Keep newest"]),
                _element("sort", "Sort by date", "navigation", [780, 120, 930, 230], 0.10, ["Newest first"]),
            ],
            [_action("open", "Broccoli"), _action("click", "Older Tomato Soup duplicate"), _action("click", "Delete recipe"), _action("click", "Confirm")],
            [_action("open", "Broccoli"), _action("click", "Older Tomato Soup duplicate"), _action("click", "Delete recipe"), _action("click", "Confirm")],
        ),
        t(
            "F01_BrowserDraw",
            "game_playing",
            False,
            10,
            "Open task.html from Downloads with Chrome, draw the three colors, then submit.",
            ["Files", "Downloads", "task.html", "Open with"],
            [
                _element("file", "task.html file", "content", [80, 740, 980, 900], 0.10, ["Open with dialog", "Chrome option", "Just once"], ["Chrome option", "Canvas red green blue submit"]),
                _element("downloads", "Downloads folder", "navigation", [60, 260, 360, 380], 0.06, ["task.html file"]),
                _element("recent", "Recent files", "navigation", [60, 420, 360, 540], 0.12, ["old task.html"]),
            ],
            [_action("open", "Files"), _action("click", "task.html file"), _action("click", "task.html file"), _action("click", "task.html file")],
            [_action("open", "Files"), _action("click", "task.html file"), _action("click", "Chrome option"), _action("draw", "Canvas red green blue"), _action("click", "Submit")],
            "repeat_same_file",
        ),
        t(
            "F02_ExpenseAddSingle",
            "data_entry",
            False,
            10,
            "Add expense Therapy Sessions amount 307.01 category Health Care.",
            ["Pro Expense", "Add", "Amount", "Category", "Note"],
            [
                _element("add", "Add expense", "primary", [890, 1880, 1040, 2040], 0.06, ["Expense form", "Amount field", "Category Health Care", "Save"], ["Amount field", "Category Health Care", "Save"]),
                _element("search", "Search expenses", "navigation", [780, 120, 920, 230], 0.07, ["Search query"]),
                _element("old", "Therapy old expense", "content", [60, 560, 980, 720], 0.18, ["Expense detail"]),
            ],
            [_action("open", "Pro Expense"), _action("click", "Add expense"), _action("type", "Note field", "307.01"), _action("click", "Save")],
            [_action("open", "Pro Expense"), _action("click", "Add expense"), _action("type", "Amount field", "307.01"), _action("click", "Category Health Care"), _action("click", "Save")],
            "wrong_input_field",
        ),
        t(
            "F03_FilesMoveFile",
            "multi_app",
            False,
            10,
            "Move report.pdf from Documents to Download.",
            ["Files", "Documents", "report.pdf", "Download"],
            [
                _element("report", "report.pdf", "content", [80, 600, 980, 760], 0.14, ["File selected", "Move to", "Download folder", "Paste"], ["Move to", "Download folder", "Paste"]),
                _element("documents", "Documents folder", "navigation", [60, 260, 420, 380], 0.06, ["report.pdf"]),
                _element("trash", "Delete", "risky", [780, 120, 920, 230], 0.65, ["Delete confirmation"], [], ["destructive"]),
            ],
            [_action("open", "Files"), _action("click", "Documents folder"), _action("click", "report.pdf"), _action("click", "Back")],
            [_action("open", "Files"), _action("click", "Documents folder"), _action("click", "report.pdf"), _action("click", "Move to"), _action("click", "Download folder"), _action("click", "Paste")],
            "lost_navigation",
        ),
        t(
            "F04_MarkorAddNoteHeader",
            "data_edit",
            False,
            10,
            "Add markdown header '# Trip Plan' to note travel.md in Markor.",
            ["Markor", "travel.md", "Editor", "Search"],
            [
                _element("travel", "travel.md note", "content", [80, 500, 980, 660], 0.08, ["Editor", "Top of note", "Header line"], ["Top of note", "Header line"]),
                _element("search", "Search notes", "navigation", [820, 120, 940, 230], 0.08, ["Search query"]),
                _element("new", "New note", "primary", [890, 1900, 1040, 2060], 0.10, ["New file dialog"]),
            ],
            [_action("open", "Markor"), _action("click", "travel.md note"), _action("type", "Body", "# Trip Plan")],
            [_action("open", "Markor"), _action("click", "travel.md note"), _action("click", "Top of note"), _action("type", "Header line", "# Trip Plan"), _action("click", "Save")],
            "wrong_cursor_position",
        ),
        t(
            "F05_RecipeAddSingleRecipe",
            "data_entry",
            False,
            10,
            "Add Chicken Caesar Salad Wrap with servings 3-4 and prep time 2 hrs.",
            ["Broccoli", "Add recipe", "Recipes", "Search"],
            [
                _element("add", "Add recipe", "primary", [890, 1880, 1040, 2040], 0.07, ["Recipe form", "Title field", "Servings field", "Prep time field", "Directions field", "Save"], ["Title field", "Servings field", "Prep time field", "Save"]),
                _element("search", "Search recipes", "navigation", [780, 120, 920, 230], 0.07, ["Search query"]),
                _element("old", "Chicken recipes", "content", [60, 400, 980, 560], 0.12, ["Recipe list"]),
            ],
            [_action("open", "Broccoli"), _action("click", "Add recipe"), _action("type", "Title field", "Chicken Caesar Salad Wrap"), _action("click", "Save")],
            [_action("open", "Broccoli"), _action("click", "Add recipe"), _action("type", "Title field", "Chicken Caesar Salad Wrap"), _action("type", "Servings field", "3-4"), _action("type", "Prep time field", "2 hrs"), _action("click", "Save")],
            "missing_required_fields",
        ),
        t(
            "F06_OsmAndMarker",
            "search",
            False,
            10,
            "Add a map marker at 47.1303814, 9.5930117 in OsmAnd.",
            ["OsmAnd", "Search", "Map area", "Markers"],
            [
                _element("search", "Search coordinates", "primary", [60, 120, 900, 260], 0.10, ["Search field", "Coordinate result", "Show on map"], ["Coordinate result", "Show on map", "Add marker"]),
                _element("map", "Map area", "risky", [0, 360, 1080, 1800], 0.38, ["Map panned", "Unknown marker"], ["Long press marker"], ["dynamic_map"]),
                _element("markers", "Markers menu", "navigation", [780, 1800, 980, 2040], 0.14, ["Markers list"]),
            ],
            [_action("open", "OsmAnd"), _action("click", "Search coordinates"), _action("type", "Search field", "47.1303814 9.5930117"), _action("click", "Map area")],
            [_action("open", "OsmAnd"), _action("click", "Search coordinates"), _action("type", "Search field", "47.1303814 9.5930117"), _action("click", "Coordinate result"), _action("click", "Add marker")],
            "dynamic_map_mismatch",
        ),
        t(
            "F07_TasksHighPriority",
            "information_retrieval",
            False,
            10,
            "List all high priority tasks due today.",
            ["Tasks", "Today", "All priorities", "Filter"],
            [
                _element("filter", "Filter priority", "navigation", [780, 120, 930, 230], 0.07, ["Priority menu", "High priority", "Today", "High priority due today"], ["High priority", "High priority due today"]),
                _element("today", "Today list", "content", [60, 360, 980, 760], 0.06, ["All tasks", "low priority task", "high priority task"]),
                _element("add", "Add task", "primary", [890, 1900, 1040, 2060], 0.10, ["New task form"]),
            ],
            [_action("open", "Tasks"), _action("click", "Today list"), _action("answer", "all visible tasks")],
            [_action("open", "Tasks"), _action("click", "Filter priority"), _action("click", "High priority"), _action("answer", "High priority due today")],
            "missing_filter",
        ),
        t(
            "F08_CalendarAddEvent",
            "data_entry",
            False,
            10,
            "Add event Dentist tomorrow at 9 AM in Simple Calendar.",
            ["Simple Calendar", "Add event", "Today", "Tomorrow"],
            [
                _element("add", "Add event", "primary", [890, 1900, 1040, 2060], 0.07, ["Event form", "Title field", "Date tomorrow", "Time 09:00", "Save"], ["Title field", "Date tomorrow", "Time 09:00", "Save"]),
                _element("today", "Today cell", "content", [90, 520, 260, 690], 0.10, ["Today events"]),
                _element("search", "Search events", "navigation", [820, 120, 940, 230], 0.06, ["Search query"]),
            ],
            [_action("open", "Simple Calendar"), _action("click", "Add event"), _action("type", "Title field", "Dentist"), _action("click", "Today cell"), _action("click", "Save")],
            [_action("open", "Simple Calendar"), _action("click", "Add event"), _action("type", "Title field", "Dentist"), _action("click", "Date tomorrow"), _action("click", "Time 09:00"), _action("click", "Save")],
            "wrong_date",
        ),
        t(
            "F09_SystemCopyClipboard",
            "system",
            False,
            10,
            "Copy the displayed code ZX-491 to the clipboard.",
            ["Browser", "code ZX-491", "Share", "Select text"],
            [
                _element("code", "code ZX-491 text", "content", [80, 620, 980, 760], 0.10, ["Text selection toolbar", "Copy"], ["Copy"]),
                _element("share", "Share", "risky", [820, 120, 940, 230], 0.26, ["Share sheet"], ["Copy link"], ["share_sheet"]),
                _element("search", "Find in page", "navigation", [680, 120, 800, 230], 0.08, ["Find field"]),
            ],
            [_action("open", "Browser"), _action("click", "Share"), _action("click", "Copy link")],
            [_action("open", "Browser"), _action("long_click", "code ZX-491 text"), _action("click", "Copy")],
            "share_instead_of_select",
        ),
        t(
            "F10_AudioRecorderRecordAudioWithFileName",
            "data_entry",
            False,
            10,
            "Record audio and save it as presentation_fGwr.m4a.",
            ["Audio Recorder", "Record", "Record-7", "Name field", "Save"],
            [
                _element("record", "Record button", "primary", [450, 1900, 620, 2070], 0.06, ["Recording screen", "Stop button", "Name field", "Save"], ["Stop button", "Name field", "Clear name", "Save"]),
                _element("name", "Name field Record-7", "input", [120, 740, 960, 860], 0.12, ["Default name Record-7", "Clear name"], ["Clear name", "Save"]),
                _element("backspace", "Backspace key", "risky", [880, 1900, 1050, 2100], 0.20, ["Partial delete"]),
            ],
            [_action("open", "Audio Recorder"), _action("click", "Record button"), _action("click", "Stop button"), _action("type", "Name field", "presentation_fGwr.m4a"), _action("click", "Backspace key")],
            [_action("open", "Audio Recorder"), _action("click", "Record button"), _action("click", "Stop button"), _action("click", "Clear name"), _action("type", "Name field", "presentation_fGwr.m4a"), _action("click", "Save")],
            "text_not_cleared",
        ),
    ]


def _score_element(task: SyntheticTask, element: UIElement, planned: Action, prior_hint: dict[str, Any] | None) -> dict[str, float]:
    role_prior = {
        "primary": 0.22,
        "input": 0.20,
        "content": 0.16,
        "navigation": 0.12,
        "risky": -0.02,
    }.get(element.role, 0.0)
    goal_sim = _similarity(task.goal, " ".join([element.label, *element.outcomes, *element.children]))
    planned_sim = _similarity(planned.label, element.label)
    hint_text = " ".join(prior_hint.get("supported_next_labels", [])) if prior_hint else ""
    hint_sim = _similarity(hint_text, " ".join([element.label, *element.outcomes, *element.children]))
    destructive_goal_bonus = 0.10 if "destructive_goal" in element.tags and any(w in task.goal.lower() for w in ["delete", "remove"]) else 0.0
    risk_penalty = 0.28 * element.risk
    score = 0.42 * goal_sim + 0.25 * planned_sim + 0.18 * hint_sim + role_prior + destructive_goal_bonus - risk_penalty
    return {
        "score": round(score, 4),
        "goal_similarity": round(goal_sim, 4),
        "planned_action_similarity": round(planned_sim, 4),
        "prior_hint_similarity": round(hint_sim, 4),
        "role_prior": round(role_prior, 4),
        "risk_penalty": round(risk_penalty, 4),
    }


def _select_roots(
    task: SyntheticTask,
    planned: Action,
    prior_hint: dict[str, Any] | None,
    max_roots: int,
) -> list[tuple[UIElement, dict[str, float]]]:
    scored = [(element, _score_element(task, element, planned, prior_hint)) for element in task.elements]
    by_role: dict[str, list[tuple[UIElement, dict[str, float]]]] = defaultdict(list)
    for item in scored:
        by_role[item[0].role].append(item)
    for role_items in by_role.values():
        role_items.sort(key=lambda x: x[1]["score"], reverse=True)

    selected: list[tuple[UIElement, dict[str, float]]] = []
    for role in ["primary", "input", "content", "navigation", "risky"]:
        if by_role.get(role):
            selected.append(by_role[role][0])
        if len(selected) >= max_roots:
            break
    for item in sorted(scored, key=lambda x: x[1]["score"], reverse=True):
        if item not in selected:
            selected.append(item)
        if len(selected) >= max_roots:
            break
    return selected


def _rollback(element: UIElement, depth: int, rng: random.Random) -> dict[str, Any]:
    failure_prob = 0.015 + 0.23 * element.risk + 0.04 * max(0, depth - 1)
    if "destructive" in element.tags:
        failure_prob += 0.35
    if "dynamic_map" in element.tags:
        failure_prob += 0.10
    if "share_sheet" in element.tags:
        failure_prob += 0.08
    success = rng.random() > failure_prob
    if success:
        mode = "state_consistent"
    elif "destructive" in element.tags:
        mode = "destructive_guard_failed"
    elif "dynamic_map" in element.tags:
        mode = "dynamic_state_mismatch"
    else:
        mode = "state_mismatch"
    return {
        "success": success,
        "mode": mode,
        "failure_probability": round(failure_prob, 4),
    }


def _explore_branch(
    task: SyntheticTask,
    root: UIElement,
    score: dict[str, float],
    step: int,
    branch_rank: int,
    upcoming_actions: list[Action],
    max_depth: int,
    rng: random.Random,
) -> dict[str, Any]:
    depth = 1
    path = [root.label]
    observations = [*root.outcomes]
    next_boundary_keywords = {"field", "dialog", "confirm", "canvas", "result", "menu", "editor", "toolbar"}
    for child in root.children:
        if depth >= max_depth:
            break
        depth += 1
        path.append(child)
        observations.append(child)
        if any(keyword in child.lower() for keyword in next_boundary_keywords):
            break
        if rng.random() < 0.20:
            break

    rollback = _rollback(root, depth, rng)
    labels = [*path, *observations]
    supported_future = [action.label for action in upcoming_actions if _action_match(action, labels)]
    useful = rollback["success"] and bool(supported_future)
    return {
        "capsule_id": f"{task.task_id}_s{step}_b{branch_rank}",
        "step": step,
        "branch_rank": branch_rank,
        "root_action_label": root.label,
        "root_role": root.role,
        "bbox": root.bbox,
        "path": path,
        "observations": observations,
        "depth": depth,
        "score": score,
        "rollback": rollback,
        "supported_next_labels": supported_future,
        "useful": useful,
    }


def _choose_committed_action(
    task: SyntheticTask,
    oracle_idx: int,
    step: int,
    prior_hint: dict[str, Any] | None,
    rng: random.Random,
) -> tuple[Action, str]:
    oracle_action = task.oracle_actions[min(oracle_idx, len(task.oracle_actions) - 1)]
    baseline_action = task.baseline_actions[min(step - 1, len(task.baseline_actions) - 1)]
    if task.baseline_success:
        return oracle_action, "baseline_already_correct"
    if prior_hint and prior_hint.get("rollback", {}).get("success"):
        hint_labels = prior_hint.get("supported_next_labels", []) + prior_hint.get("observations", [])
        if _action_match(oracle_action, hint_labels, threshold=0.18):
            ambiguity_penalty = 0.18 if task.failure_mode in {"wrong_input_field", "text_not_cleared", "dynamic_map_mismatch"} else 0.0
            if rng.random() < 0.88 - ambiguity_penalty:
                return oracle_action, "tpes_evidence_corrected"
    return baseline_action, "baseline_policy"


def _failure_reason(task: SyntheticTask, success: bool, hit_rate: float, rollback_failures: int, max_depth: int) -> str:
    if success:
        return "success"
    if rollback_failures:
        return "rollback_failure_discarded_or_corrupted_evidence"
    if max_depth < 2:
        return "evidence_too_shallow"
    if hit_rate < 0.4:
        return "low_action_conditioned_hit_rate"
    if task.failure_mode in {"wrong_input_field", "text_not_cleared"}:
        return "input_semantics_not_precise"
    if task.failure_mode == "dynamic_map_mismatch":
        return "dynamic_ui_state_hard_to_match"
    return task.failure_mode or "task_not_completed"


def simulate_task(
    task: SyntheticTask,
    rng: random.Random,
    max_roots: int,
    max_depth: int,
    max_steps: int,
) -> dict[str, Any]:
    oracle_idx = 0
    step = 0
    prior_hint: dict[str, Any] | None = None
    capsules: list[dict[str, Any]] = []
    step_traces: list[dict[str, Any]] = []
    action_hits = 0
    useful_commit_evidence = 0
    rollback_failures = 0
    max_depth_reached = 0
    repeated_wrong_actions = 0

    while step < max_steps and oracle_idx < len(task.oracle_actions):
        step += 1
        planned = task.baseline_actions[min(step - 1, len(task.baseline_actions) - 1)]
        roots = _select_roots(task, planned, prior_hint, max_roots)
        upcoming = task.oracle_actions[oracle_idx : min(len(task.oracle_actions), oracle_idx + 3)]
        branch_capsules = [
            _explore_branch(task, root, score, step, idx, upcoming, max_depth, rng)
            for idx, (root, score) in enumerate(roots, start=1)
        ]
        capsules.extend(branch_capsules)
        rollback_failures += sum(1 for capsule in branch_capsules if not capsule["rollback"]["success"])
        max_depth_reached = max(max_depth_reached, *(capsule["depth"] for capsule in branch_capsules), 0)

        committed, source = _choose_committed_action(task, oracle_idx, step, prior_hint, rng)
        action_hit = any(
            capsule["rollback"]["success"] and _action_match(committed, capsule["path"] + capsule["observations"], threshold=0.18)
            for capsule in branch_capsules
        )
        action_hits += int(action_hit)

        current_oracle = task.oracle_actions[oracle_idx]
        progressed = _action_match(current_oracle, [committed.label], threshold=0.30)
        if progressed:
            oracle_idx += 1
            repeated_wrong_actions = 0
        else:
            repeated_wrong_actions += 1

        commit_branch = next(
            (
                capsule
                for capsule in branch_capsules
                if capsule["rollback"]["success"]
                and _action_match(committed, [capsule["root_action_label"], *capsule["path"], *capsule["observations"]], threshold=0.18)
            ),
            None,
        )
        prior_hint = commit_branch if commit_branch else None
        if prior_hint and prior_hint.get("useful"):
            useful_commit_evidence += 1

        step_traces.append(
            {
                "step": step,
                "planned_baseline_action": dataclasses.asdict(planned),
                "predicted_root_actions": [
                    {
                        "rank": idx,
                        "label": root.label,
                        "role": root.role,
                        "score": score,
                    }
                    for idx, (root, score) in enumerate(roots, start=1)
                ],
                "branch_capsule_ids": [capsule["capsule_id"] for capsule in branch_capsules],
                "committed_action": dataclasses.asdict(committed),
                "commit_source": source,
                "action_hit_by_speculation": action_hit,
                "oracle_action_before_commit": dataclasses.asdict(current_oracle),
                "progressed": progressed,
                "injected_next_step_capsule_id": prior_hint["capsule_id"] if prior_hint else None,
            }
        )

        if repeated_wrong_actions >= 3:
            break

    success = oracle_idx >= len(task.oracle_actions)
    tpes_steps = step
    hit_rate = action_hits / max(1, tpes_steps)
    useful_evidence_rate = sum(1 for capsule in capsules if capsule["useful"]) / max(1, len(capsules))
    commit_evidence_rate = useful_commit_evidence / max(1, tpes_steps)
    failure_reason = _failure_reason(task, success, hit_rate, rollback_failures, max_depth_reached)

    if success and not task.baseline_success:
        analysis = "TPES 通过 commit branch evidence 修正了 baseline 的错误路径。"
    elif success:
        analysis = "baseline 已经能完成；TPES evidence 主要用于确认下一步，没有明显改变路径。"
    elif rollback_failures:
        analysis = "rollback 失败导致部分 speculative evidence 被丢弃，下一步 reasoning 缺少可用证据。"
    elif hit_rate < 0.4:
        analysis = "探索分支与实际 commit action 对齐率低，candidate selection 没覆盖关键动作。"
    else:
        analysis = "探索有一定命中，但深度或输入语义不足，未能完成任务。"

    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "goal": task.goal,
        "baseline_success": task.baseline_success,
        "baseline_steps": task.baseline_steps,
        "baseline_actions": [dataclasses.asdict(action) for action in task.baseline_actions],
        "tpes_success": success,
        "tpes_steps": tpes_steps,
        "tpes_minus_baseline_steps": tpes_steps - task.baseline_steps,
        "hit_rate": round(hit_rate, 4),
        "max_depth": max_depth_reached,
        "rollback_failures": rollback_failures,
        "useful_evidence_rate": round(useful_evidence_rate, 4),
        "commit_branch_useful_evidence_rate": round(commit_evidence_rate, 4),
        "failure_mode": task.failure_mode,
        "failure_reason": failure_reason,
        "analysis": analysis,
        "step_traces": step_traces,
        "evidence_capsules": capsules,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    for task_type in sorted({result["task_type"] for result in results}):
        rows = [result for result in results if result["task_type"] == task_type]
        by_type[task_type] = {
            "num_tasks": len(rows),
            "baseline_success_rate": round(_mean([float(row["baseline_success"]) for row in rows]), 4),
            "tpes_success_rate": round(_mean([float(row["tpes_success"]) for row in rows]), 4),
            "avg_baseline_steps": round(_mean([float(row["baseline_steps"]) for row in rows]), 4),
            "avg_tpes_steps": round(_mean([float(row["tpes_steps"]) for row in rows]), 4),
            "avg_step_delta": round(_mean([float(row["tpes_minus_baseline_steps"]) for row in rows]), 4),
            "avg_hit_rate": round(_mean([float(row["hit_rate"]) for row in rows]), 4),
        }

    failure_counter = Counter(row["failure_reason"] for row in results if not row["tpes_success"])
    avg_baseline_steps = _mean([float(result["baseline_steps"]) for result in results])
    avg_tpes_steps = _mean([float(result["tpes_steps"]) for result in results])
    step_delta = avg_tpes_steps - avg_baseline_steps
    success_gain = _mean([float(result["tpes_success"]) for result in results]) - _mean(
        [float(result["baseline_success"]) for result in results]
    )
    benefited_types = [
        task_type
        for task_type, metrics in by_type.items()
        if metrics["tpes_success_rate"] > metrics["baseline_success_rate"]
    ]
    return {
        "num_tasks": len(results),
        "baseline_success_rate": round(_mean([float(result["baseline_success"]) for result in results]), 4),
        "tpes_success_rate": round(_mean([float(result["tpes_success"]) for result in results]), 4),
        "success_rate_delta": round(success_gain, 4),
        "average_baseline_episode_length": round(avg_baseline_steps, 4),
        "average_tpes_episode_length": round(avg_tpes_steps, 4),
        "average_tpes_minus_baseline_steps": round(step_delta, 4),
        "average_hit_rate": round(_mean([float(result["hit_rate"]) for result in results]), 4),
        "average_max_depth": round(_mean([float(result["max_depth"]) for result in results]), 4),
        "average_useful_evidence_rate": round(_mean([float(result["useful_evidence_rate"]) for result in results]), 4),
        "total_rollback_failures": sum(int(result["rollback_failures"]) for result in results),
        "common_failure_reasons": dict(failure_counter.most_common()),
        "reduced_steps_compared_to_baseline": avg_tpes_steps < avg_baseline_steps,
        "benefited_task_types": benefited_types,
        "by_task_type": by_type,
        "interpretation": {
            "success": "TPES success improves when commit-branch evidence covers the next oracle action and rollback succeeds.",
            "steps": "Episode length is capped at 10; lower TPES steps mainly appear when failed baseline tasks are corrected before reaching max-step budget.",
            "depth": "Depth >= 2 is important for form, file-open, and multi-step tasks because the next useful evidence often appears after the root transition.",
            "rollback": "Rollback-failed capsules are recorded but not injected into the next prompt.",
        },
    }


def _write_csv(results: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "task_id",
        "task_type",
        "baseline_success",
        "tpes_success",
        "baseline_steps",
        "tpes_steps",
        "tpes_minus_baseline_steps",
        "hit_rate",
        "max_depth",
        "rollback_failures",
        "useful_evidence_rate",
        "commit_branch_useful_evidence_rate",
        "failure_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})


def _write_charts(results: list[dict[str, Any]], summary: dict[str, Any], run_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts: dict[str, str] = {}

    def save(name: str) -> None:
        path = run_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        charts[name] = str(path.resolve())

    plt.figure(figsize=(6.4, 4.2))
    labels = ["Baseline", "TPES"]
    values = [summary["baseline_success_rate"] * 100, summary["tpes_success_rate"] * 100]
    bars = plt.bar(labels, values, color=["#7f8c8d", "#2f6f9f"])
    plt.ylim(0, 105)
    plt.ylabel("success rate (%)")
    plt.title("Success rate comparison")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center")
    save("success_rate_baseline_vs_tpes.png")

    plt.figure(figsize=(6.4, 4.2))
    values = [summary["average_baseline_episode_length"], summary["average_tpes_episode_length"]]
    bars = plt.bar(labels, values, color=["#7f8c8d", "#2f6f9f"])
    plt.ylabel("average episode length")
    plt.title("Average steps comparison")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value:.2f}", ha="center")
    save("average_steps_baseline_vs_tpes.png")

    task_labels = [result["task_id"].replace("_", "\n", 1) for result in results]
    x = list(range(len(results)))
    width = 0.38
    plt.figure(figsize=(16, 5.8))
    plt.bar([i - width / 2 for i in x], [result["baseline_steps"] for result in results], width, label="Baseline", color="#7f8c8d")
    plt.bar([i + width / 2 for i in x], [result["tpes_steps"] for result in results], width, label="TPES", color="#2f6f9f")
    plt.axhline(MAX_EPISODE_STEPS, color="#b55", linewidth=1, linestyle="--", label="max steps")
    plt.xticks(x, task_labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("episode length")
    plt.title("Per-task steps")
    plt.legend()
    save("per_task_steps_baseline_vs_tpes.png")

    type_names = list(summary["by_task_type"].keys())
    x = list(range(len(type_names)))
    width = 0.38
    plt.figure(figsize=(12, 4.8))
    plt.bar(
        [i - width / 2 for i in x],
        [summary["by_task_type"][name]["baseline_success_rate"] * 100 for name in type_names],
        width,
        label="Baseline",
        color="#7f8c8d",
    )
    plt.bar(
        [i + width / 2 for i in x],
        [summary["by_task_type"][name]["tpes_success_rate"] * 100 for name in type_names],
        width,
        label="TPES",
        color="#2f6f9f",
    )
    plt.xticks(x, type_names, rotation=35, ha="right", fontsize=8)
    plt.ylabel("success rate (%)")
    plt.title("Success rate by task type")
    plt.legend()
    save("success_rate_by_task_type.png")

    plt.figure(figsize=(7.2, 4.4))
    plt.scatter(
        [result["max_depth"] for result in results],
        [result["hit_rate"] for result in results],
        s=[50 + 20 * result["rollback_failures"] for result in results],
        c=["#2f6f9f" if result["tpes_success"] else "#c44e52" for result in results],
        alpha=0.78,
    )
    plt.xlabel("max exploration depth")
    plt.ylabel("hit rate")
    plt.ylim(-0.02, 1.05)
    plt.title("Depth / hit-rate / rollback overview")
    save("depth_hit_rate_scatter.png")
    return charts


def _write_report(payload: dict[str, Any], report_path: Path) -> None:
    summary = payload["summary"]
    results = payload["tasks"]
    charts = payload["charts"]
    lines = [
        "# TPES Initial Verification 仿真实验报告",
        "",
        "本实验使用 20 个 synthetic AndroidWorld-like GUI task 验证 TPES（Transactional Progressive Evidence Search）的在线探索设计。实验不调用真实 VLM，只使用 a11y label、UI 元素 bbox、任务关键词、轻量 string matching 和 deterministic random rollback。",
        "",
        "## 实验口径",
        "",
        "- 每个任务最多 10 个主流程 step。",
        "- 每个 step 先预测 top root actions，并对 root branch speculative exploration 到 next-decision boundary。",
        "- 每个 branch 都执行 rollback simulation；rollback 失败的 evidence capsule 只记录，不注入 prompt。",
        "- VLM action commit 之后，只把 commit branch 对应的 evidence 注入下一步 reasoning。",
        "- baseline steps 是 synthetic baseline policy 的 episode length；TPES steps 是同一任务下模拟 TPES 后的主流程 episode length。",
        "",
        "## 总体结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 任务数 | {summary['num_tasks']} |",
        f"| baseline success rate | {summary['baseline_success_rate']:.2f} |",
        f"| TPES success rate | {summary['tpes_success_rate']:.2f} |",
        f"| success rate delta | {summary['success_rate_delta']:+.2f} |",
        f"| baseline 平均 episode length | {summary['average_baseline_episode_length']:.2f} |",
        f"| TPES 平均 episode length | {summary['average_tpes_episode_length']:.2f} |",
        f"| TPES-baseline 平均步数差 | {summary['average_tpes_minus_baseline_steps']:+.2f} |",
        f"| 平均 hit rate | {summary['average_hit_rate']:.2f} |",
        f"| 平均 max depth | {summary['average_max_depth']:.2f} |",
        f"| 平均 useful evidence rate | {summary['average_useful_evidence_rate']:.2f} |",
        f"| rollback failures | {summary['total_rollback_failures']} |",
        f"| speculative exploration 是否减少步数 | {summary['reduced_steps_compared_to_baseline']} |",
        "",
        "## 图表",
        "",
        f"![success_rate]({charts['success_rate_baseline_vs_tpes.png']})",
        "",
        f"![average_steps]({charts['average_steps_baseline_vs_tpes.png']})",
        "",
        f"![per_task_steps]({charts['per_task_steps_baseline_vs_tpes.png']})",
        "",
        f"![success_rate_by_task_type]({charts['success_rate_by_task_type.png']})",
        "",
        f"![depth_hit_rate]({charts['depth_hit_rate_scatter.png']})",
        "",
        "## 逐任务结果",
        "",
        "| task | type | baseline success | TPES success | baseline steps | TPES steps | delta | hit rate | max depth | rollback fail | useful evidence | analysis |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            "| {task_id} | {task_type} | {bs} | {ts} | {bsteps} | {tsteps} | {delta:+d} | {hit:.2f} | {depth} | {rb} | {useful:.2f} | {analysis} |".format(
                task_id=result["task_id"],
                task_type=result["task_type"],
                bs=int(result["baseline_success"]),
                ts=int(result["tpes_success"]),
                bsteps=int(result["baseline_steps"]),
                tsteps=int(result["tpes_steps"]),
                delta=int(result["tpes_minus_baseline_steps"]),
                hit=float(result["hit_rate"]),
                depth=int(result["max_depth"]),
                rb=int(result["rollback_failures"]),
                useful=float(result["useful_evidence_rate"]),
                analysis=result["analysis"],
            )
        )

    lines.extend(
        [
            "",
            "## 失败原因",
            "",
            "| reason | count |",
            "| --- | ---: |",
        ]
    )
    for reason, count in summary["common_failure_reasons"].items():
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            "## 按任务类型观察",
            "",
            "| task type | n | baseline SR | TPES SR | baseline steps | TPES steps | step delta | hit rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task_type, metrics in summary["by_task_type"].items():
        lines.append(
            f"| {task_type} | {metrics['num_tasks']} | {metrics['baseline_success_rate']:.2f} | {metrics['tpes_success_rate']:.2f} | {metrics['avg_baseline_steps']:.2f} | {metrics['avg_tpes_steps']:.2f} | {metrics['avg_step_delta']:+.2f} | {metrics['avg_hit_rate']:.2f} |"
        )

    benefited = ", ".join(summary["benefited_task_types"]) or "无"
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 本次 synthetic verification 中，TPES success rate 相比 baseline 的变化是 {summary['success_rate_delta']:+.2f}。",
            f"- 平均步数变化是 {summary['average_tpes_minus_baseline_steps']:+.2f}；`reduced_steps_compared_to_baseline={summary['reduced_steps_compared_to_baseline']}`。",
            f"- 受益最多的任务类型：{benefited}。",
            "- TPES 更适合 root action 正确但后续 decision boundary 容易出错的任务，例如文件打开后的 app chooser、表单字段选择、filter/search 后的信息读取。",
            "- TPES 不适合 rollback 不稳定或动态 UI 强的任务，例如地图 pan、share sheet、destructive action；这些场景应该降低探索优先级，或引入更严格的 state-consistent rollback verifier。",
            "- 如果 max depth 小于 2，很多 evidence 只能说明 root transition，而不能指导 t+1 的下一步 reasoning；因此 depth-2/3 对多步任务是必要的。",
            "",
            "## 输出文件",
            "",
            f"- JSON: `{payload['output_json']}`",
            f"- CSV: `{payload['output_csv']}`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max_roots", type=int, default=4)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=MAX_EPISODE_STEPS)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    run_dir = args.output_root / f"run_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = _make_taskbed()
    results = [
        simulate_task(
            task,
            rng=rng,
            max_roots=args.max_roots,
            max_depth=args.max_depth,
            max_steps=args.max_steps,
        )
        for task in tasks
    ]
    summary = _summarize(results)

    output_json = run_dir / "tpes_initial_verification_results.json"
    output_csv = run_dir / "tpes_initial_verification_task_metrics.csv"
    report_path = run_dir / "tpes_initial_verification_report_cn.md"
    _write_csv(results, output_csv)
    charts = _write_charts(results, summary, run_dir)
    payload = {
        "metadata": {
            "strategy": "TPES initial verification simulation",
            "seed": args.seed,
            "max_roots": args.max_roots,
            "max_depth": args.max_depth,
            "max_steps": args.max_steps,
            "num_synthetic_tasks": len(tasks),
            "uses_real_vlm": False,
        },
        "summary": summary,
        "tasks": results,
        "charts": charts,
        "output_json": str(output_json.resolve()),
        "output_csv": str(output_csv.resolve()),
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(payload, report_path)
    print(
        json.dumps(
            {
                "json": str(output_json.resolve()),
                "csv": str(output_csv.resolve()),
                "report": str(report_path.resolve()),
                "charts": charts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
