#!/usr/bin/env python3
"""Simulate TPES exploration on a synthetic mobile GUI testbed."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
import random
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "tpes_simulation"
STOPWORDS = {
    "a",
    "an",
    "and",
    "app",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "me",
    "new",
    "of",
    "on",
    "or",
    "save",
    "the",
    "to",
    "with",
}


@dataclasses.dataclass
class UIElement:
    element_id: str
    text: str
    content_desc: str
    class_name: str
    bbox: list[int]
    role: str
    risk: float = 0.05
    outcome_labels: list[str] = dataclasses.field(default_factory=list)
    depth_children: list[str] = dataclasses.field(default_factory=list)
    constraint_tags: list[str] = dataclasses.field(default_factory=list)

    @property
    def label(self) -> str:
        return self.text or self.content_desc or self.element_id


@dataclasses.dataclass
class Action:
    action_type: str
    target: str
    text: str = ""

    @property
    def label(self) -> str:
        if self.text:
            return f"{self.action_type} {self.target} {self.text}"
        return f"{self.action_type} {self.target}"


@dataclasses.dataclass
class SyntheticTask:
    task_id: str
    baseline_success: bool
    goal: str
    initial_labels: list[str]
    candidates: list[UIElement]
    baseline_actions: list[Action]
    oracle_actions: list[Action]
    failure_mode: str
    notes: str


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return {w for w in words if len(w) > 1 and w not in STOPWORDS}


def _similarity(a: str, b: str) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _any_action_match(action: Action, labels: list[str], threshold: float = 0.26) -> bool:
    action_text = action.label
    return any(_similarity(action_text, label) >= threshold for label in labels)


def _risk_band(risk: float) -> str:
    if risk >= 0.55:
        return "high"
    if risk >= 0.25:
        return "medium"
    return "low"


def _mean(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return sum(float(v) for v in values) / len(values)


def _make_taskbed() -> list[SyntheticTask]:
    """Builds 10 synthetic Android-like tasks: 5 baseline success, 5 failure."""

    def e(
        element_id: str,
        text: str,
        desc: str,
        klass: str,
        bbox: list[int],
        role: str,
        risk: float,
        outcome: list[str],
        children: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> UIElement:
        return UIElement(
            element_id=element_id,
            text=text,
            content_desc=desc,
            class_name=klass,
            bbox=bbox,
            role=role,
            risk=risk,
            outcome_labels=outcome,
            depth_children=children or [],
            constraint_tags=constraints or [],
        )

    return [
        SyntheticTask(
            task_id="S1_MarkorCreateNote",
            baseline_success=True,
            goal="Create a new Markor note named wise_yacht.md and write Ignorance is bliss.",
            initial_labels=["Markor", "Files", "search", "sort", "create new file"],
            candidates=[
                e("fab_new", "", "Create a new file or folder", "ImageButton", [920, 2000, 1040, 2140], "primary", 0.05, ["New file dialog", "Name", "my_note", ".md"], ["name_input"]),
                e("sort", "", "Sort by", "ImageButton", [790, 150, 850, 240], "navigation", 0.10, ["Sort by name", "Sort by date"]),
                e("search", "", "Search", "ImageButton", [890, 150, 960, 240], "navigation", 0.08, ["Search notes", "query field"]),
                e("quicknote", "QuickNote", "", "TextView", [600, 2200, 780, 2350], "content", 0.20, ["QuickNote editor", "body"]),
            ],
            baseline_actions=[Action("click", "Create a new file or folder"), Action("type", "Name", "wise_yacht.md"), Action("type", "body", "Ignorance is bliss")],
            oracle_actions=[Action("click", "Create a new file or folder"), Action("type", "Name", "wise_yacht.md"), Action("type", "body", "Ignorance is bliss")],
            failure_mode="none",
            notes="Baseline can follow the primary creation flow.",
        ),
        SyntheticTask(
            task_id="S2_ExpenseAddSingle",
            baseline_success=True,
            goal="Add one expense Therapy Sessions amount 307.01 in Pro Expense.",
            initial_labels=["Pro Expense", "Expenses", "add", "search", "category"],
            candidates=[
                e("add", "+", "Add expense", "FloatingActionButton", [910, 2020, 1040, 2160], "primary", 0.05, ["Add expense", "Amount", "Category", "Note"], ["amount_field"]),
                e("search", "", "Search expenses", "ImageButton", [820, 120, 900, 220], "navigation", 0.05, ["Search", "Expense query"]),
                e("settings", "", "Settings", "ImageButton", [940, 120, 1020, 220], "risky", 0.35, ["Settings", "Currency", "Backup"], [], ["avoid_settings"]),
                e("therapy_row", "Therapy", "", "TextView", [60, 500, 900, 620], "content", 0.12, ["Expense detail", "Therapy"]),
            ],
            baseline_actions=[Action("click", "Add expense"), Action("type", "Amount", "307.01"), Action("click", "Category")],
            oracle_actions=[Action("click", "Add expense"), Action("type", "Amount", "307.01"), Action("click", "Category")],
            failure_mode="none",
            notes="Goal terms strongly align with the add form.",
        ),
        SyntheticTask(
            task_id="S3_TasksHighPriority",
            baseline_success=True,
            goal="What are my high priority tasks? Answer with titles only.",
            initial_labels=["My Tasks", "Call client for follow-up", "Buy groceries", "High priority", "filters"],
            candidates=[
                e("task_red", "Call client for follow-up", "High priority", "CheckBox", [60, 360, 980, 500], "content", 0.02, ["Task detail", "priority high", "Call client for follow-up"]),
                e("filter", "", "Filter", "ImageButton", [800, 120, 880, 220], "navigation", 0.05, ["Filter tasks", "Priority", "Due date"], ["priority_filter"]),
                e("search", "", "Search", "ImageButton", [900, 120, 980, 220], "navigation", 0.05, ["Search tasks", "query"]),
                e("add", "+", "Add task", "FloatingActionButton", [910, 2030, 1040, 2160], "primary", 0.08, ["New task", "Title", "Priority"]),
            ],
            baseline_actions=[Action("answer", "Call client for follow-up")],
            oracle_actions=[Action("answer", "Call client for follow-up")],
            failure_mode="none",
            notes="The answer is visible; evidence can be shallow.",
        ),
        SyntheticTask(
            task_id="S4_CalendarTimeRange",
            baseline_success=True,
            goal="Find events between 4:15pm and 8pm Friday in Simple Calendar.",
            initial_labels=["Friday", "12:51 Annual Report", "16:15 Cooking Class", "calendar"],
            candidates=[
                e("event_cooking", "Cooking Class", "16:15 - 17:15", "TextView", [100, 900, 980, 1040], "content", 0.03, ["Event detail", "Cooking Class", "16:15", "17:15"]),
                e("date_picker", "", "Go to date", "ImageButton", [760, 120, 850, 220], "navigation", 0.08, ["Date picker", "Friday"]),
                e("search", "", "Search events", "ImageButton", [880, 120, 960, 220], "navigation", 0.06, ["Search calendar", "event query"]),
                e("add", "+", "Add event", "FloatingActionButton", [900, 2000, 1040, 2160], "primary", 0.12, ["New event", "Title", "Time"]),
            ],
            baseline_actions=[Action("answer", "Cooking Class")],
            oracle_actions=[Action("answer", "Cooking Class")],
            failure_mode="none",
            notes="Screen reading evidence is reliable and low-risk.",
        ),
        SyntheticTask(
            task_id="S5_FileCopyReceipt",
            baseline_success=True,
            goal="Copy receipt_2023_01_22.jpg from DCIM and save a copy in Download.",
            initial_labels=["Gallery", "Albums", "DCIM", "receipt_2023_01_22.jpg", "more"],
            candidates=[
                e("receipt", "receipt_2023_01_22.jpg", "", "ImageView", [80, 430, 430, 760], "content", 0.08, ["Image preview", "More options", "Copy to"], ["copy_action"]),
                e("dcim", "DCIM", "", "TextView", [80, 240, 300, 340], "navigation", 0.05, ["DCIM album", "receipt_2023_01_22.jpg"]),
                e("more", "", "More options", "ImageButton", [920, 120, 1010, 220], "risky", 0.22, ["Options", "copy", "move", "delete"], ["copy_action"]),
                e("trash", "", "Delete", "ImageButton", [760, 120, 850, 220], "risky", 0.70, ["Delete confirmation"], [], ["destructive"]),
            ],
            baseline_actions=[Action("click", "receipt_2023_01_22.jpg"), Action("click", "Copy to"), Action("click", "Download")],
            oracle_actions=[Action("click", "receipt_2023_01_22.jpg"), Action("click", "Copy to"), Action("click", "Download")],
            failure_mode="none",
            notes="Useful evidence requires avoiding destructive action.",
        ),
        SyntheticTask(
            task_id="F1_AudioRenameLoop",
            baseline_success=False,
            goal="Record audio and save it as presentation_fGwr.m4a.",
            initial_labels=["Audio Recorder", "record button", "Record-7", "New name", "keyboard"],
            candidates=[
                e("record", "", "Record", "ImageButton", [480, 2050, 600, 2180], "primary", 0.04, ["Recording", "Stop button", "timer"], ["stop"]),
                e("name", "Record-7", "New name", "EditText", [120, 650, 950, 760], "input", 0.12, ["Name field", "Record-7presentation_fGwr.m4a", "Save"], ["clear_name"]),
                e("backspace", "", "Backspace", "Key", [900, 1920, 1060, 2130], "risky", 0.18, ["Name field", "partial deletion"], []),
                e("save", "OK", "Save", "Button", [760, 900, 980, 1010], "primary", 0.08, ["Saved recording", "Recordings list"]),
            ],
            baseline_actions=[Action("click", "Record"), Action("click", "Stop"), Action("type", "New name", "presentation_fGwr.m4a"), Action("click", "Backspace")],
            oracle_actions=[Action("click", "Record"), Action("click", "Stop"), Action("clear_type", "New name", "presentation_fGwr.m4a"), Action("click", "Save")],
            failure_mode="text_not_cleared",
            notes="Baseline appends text to default filename and loops on backspace.",
        ),
        SyntheticTask(
            task_id="F2_BrowserDrawFileOpen",
            baseline_success=False,
            goal="Open task.html from Downloads with Chrome, draw three colors, submit.",
            initial_labels=["Files", "Downloads", "task.html", "Open with", "Chrome"],
            candidates=[
                e("task_html", "task.html", "", "TextView", [80, 860, 980, 1000], "content", 0.10, ["Open with", "Chrome", "Just once"], ["chrome_choice"]),
                e("downloads", "Downloads", "", "TextView", [60, 260, 360, 360], "navigation", 0.05, ["Downloads", "task.html"]),
                e("chrome", "Chrome", "", "TextView", [120, 980, 900, 1120], "content", 0.15, ["Browser canvas", "red", "green", "blue", "submit"], ["draw_canvas"]),
                e("recent", "Recent", "", "TextView", [40, 400, 400, 520], "navigation", 0.12, ["Recent files", "task.html"]),
            ],
            baseline_actions=[Action("click", "task.html"), Action("click", "task.html"), Action("click", "task.html")],
            oracle_actions=[Action("click", "task.html"), Action("click", "Chrome"), Action("click", "red green blue canvas"), Action("click", "Submit")],
            failure_mode="repeated_file_click",
            notes="The key evidence is after the file-open transition, not at root only.",
        ),
        SyntheticTask(
            task_id="F3_ContactsPhoneField",
            baseline_success=False,
            goal="Create contact Hugo Pereira with phone +13920741751.",
            initial_labels=["Contacts", "Create contact", "First name", "Last name", "Company", "Phone"],
            candidates=[
                e("first", "First name", "", "EditText", [110, 340, 940, 450], "input", 0.05, ["First name", "Hugo"]),
                e("last", "Last name", "", "EditText", [110, 470, 940, 580], "input", 0.05, ["Last name", "Pereira"]),
                e("company", "Company", "", "EditText", [110, 600, 940, 710], "input", 0.15, ["Company field", "wrong focus"], []),
                e("phone", "Phone", "", "EditText", [110, 820, 940, 930], "input", 0.07, ["Phone", "+13920741751", "Save"], ["save_contact"]),
                e("save", "Save", "", "Button", [850, 80, 1030, 200], "primary", 0.05, ["Contact detail", "Hugo Pereira"]),
            ],
            baseline_actions=[Action("type", "First name", "Hugo"), Action("type", "Last name", "Pereira"), Action("type", "Company", "+13920741751")],
            oracle_actions=[Action("type", "First name", "Hugo"), Action("type", "Last name", "Pereira"), Action("type", "Phone", "+13920741751"), Action("click", "Save")],
            failure_mode="wrong_input_field",
            notes="Candidate selection must distinguish semantically similar input boxes.",
        ),
        SyntheticTask(
            task_id="F4_RecipeMultiApp",
            baseline_success=False,
            goal="Read recipes.txt in Markor and add all recipes into Broccoli.",
            initial_labels=["Markor", "recipes.txt", "Broccoli", "Share", "Open"],
            candidates=[
                e("recipes", "recipes.txt", "", "TextView", [90, 500, 960, 650], "content", 0.12, ["Recipe list", "Pasta", "Soup", "Share"], ["copy_recipes"]),
                e("share", "", "Share", "ImageButton", [850, 120, 940, 220], "risky", 0.30, ["Share sheet", "Broccoli", "Copy"], ["broccoli"]),
                e("broccoli", "Broccoli", "", "TextView", [100, 900, 930, 1040], "content", 0.35, ["Broccoli add recipe", "Title", "Ingredients"], ["add_recipe"]),
                e("delete", "", "Delete", "ImageButton", [940, 120, 1020, 220], "risky", 0.70, ["Delete confirmation"], [], ["destructive"]),
            ],
            baseline_actions=[Action("click", "recipes.txt"), Action("click", "Share"), Action("click", "Broccoli"), Action("type", "Title", "one recipe only")],
            oracle_actions=[Action("click", "recipes.txt"), Action("click", "Share"), Action("click", "Broccoli"), Action("repeat_type", "all recipes")],
            failure_mode="multi_app_memory_loss",
            notes="Multi-app transfer needs deeper evidence and stable rollback.",
        ),
        SyntheticTask(
            task_id="F5_OsmAndMarker",
            baseline_success=False,
            goal="Add a location marker for Planken, Liechtenstein in OsmAnd.",
            initial_labels=["OsmAnd", "Search", "Map", "Favorites", "Marker"],
            candidates=[
                e("search", "", "Search", "ImageButton", [50, 120, 150, 230], "navigation", 0.05, ["Search field", "Planken Liechtenstein"], ["result"]),
                e("map", "Map", "", "MapView", [0, 280, 1080, 2000], "risky", 0.40, ["Map pan", "unknown location"], []),
                e("marker", "", "Add marker", "ImageButton", [880, 1900, 1030, 2060], "primary", 0.30, ["Marker details", "Name", "Save"], ["save_marker"]),
                e("favorite", "Favorites", "", "TextView", [50, 2100, 250, 2300], "navigation", 0.18, ["Favorites list", "Add favorite"]),
            ],
            baseline_actions=[Action("click", "Map"), Action("click", "Add marker"), Action("click", "Save")],
            oracle_actions=[Action("click", "Search"), Action("type", "Search field", "Planken Liechtenstein"), Action("click", "Search result"), Action("click", "Add marker")],
            failure_mode="map_state_ambiguous",
            notes="Map state is dynamic; rollback and matching are fragile.",
        ),
    ]


def _score_candidate(task: SyntheticTask, candidate: UIElement, planned_action: Action) -> dict[str, float]:
    text = " ".join([candidate.label, candidate.content_desc, candidate.class_name, candidate.role])
    goal_sim = _similarity(task.goal, text)
    plan_sim = _similarity(planned_action.label, text)
    role_prior = {
        "primary": 0.22,
        "input": 0.18,
        "content": 0.14,
        "navigation": 0.10,
        "risky": -0.06,
    }.get(candidate.role, 0.0)
    risk_penalty = 0.30 * candidate.risk
    score = max(0.0, 0.52 * goal_sim + 0.38 * plan_sim + role_prior - risk_penalty)
    return {
        "score": round(score, 4),
        "goal_similarity": round(goal_sim, 4),
        "planned_action_similarity": round(plan_sim, 4),
        "role_prior": round(role_prior, 4),
        "risk_penalty": round(risk_penalty, 4),
    }


def _choose_root_candidates(
    task: SyntheticTask,
    planned_action: Action,
    max_roots: int,
) -> list[tuple[UIElement, dict[str, float]]]:
    scored = [(candidate, _score_candidate(task, candidate, planned_action)) for candidate in task.candidates]
    by_role: dict[str, list[tuple[UIElement, dict[str, float]]]] = defaultdict(list)
    for item in scored:
        by_role[item[0].role].append(item)
    for role in by_role:
        by_role[role].sort(key=lambda x: x[1]["score"], reverse=True)

    order = ["primary", "input", "content", "navigation", "risky"]
    selected: list[tuple[UIElement, dict[str, float]]] = []
    for role in order:
        if by_role.get(role):
            selected.append(by_role[role][0])
        if len(selected) >= max_roots:
            break
    remaining = sorted(scored, key=lambda x: x[1]["score"], reverse=True)
    for item in remaining:
        if item not in selected:
            selected.append(item)
        if len(selected) >= max_roots:
            break
    return selected[:max_roots]


def _simulate_rollback(candidate: UIElement, rng: random.Random, strict: bool = True) -> dict[str, Any]:
    failure_probability = 0.02 + candidate.risk * (0.42 if strict else 0.25)
    if "destructive" in candidate.constraint_tags:
        failure_probability += 0.45
    success = rng.random() > failure_probability
    level = "level1" if success and candidate.risk < 0.30 else "level2"
    mode = "backtrack" if success else ("constraint_violation" if "destructive" in candidate.constraint_tags else "state_mismatch")
    return {
        "success": success,
        "level": level,
        "mode": mode,
        "risk_band": _risk_band(candidate.risk),
        "failure_probability": round(failure_probability, 3),
    }


def _make_deep_candidate(parent: UIElement, child_label: str, depth: int) -> UIElement:
    role = "input" if any(x in child_label.lower() for x in ["input", "field", "name", "amount", "phone"]) else "content"
    if any(x in child_label.lower() for x in ["delete", "share", "map"]):
        role = "risky"
    risk = min(0.75, parent.risk + 0.06 * depth + (0.10 if role == "risky" else 0.0))
    return UIElement(
        element_id=f"{parent.element_id}_d{depth}_{re.sub(r'[^a-z0-9]+', '_', child_label.lower()).strip('_')}",
        text=child_label,
        content_desc=child_label,
        class_name="SyntheticNode",
        bbox=parent.bbox,
        role=role,
        risk=risk,
        outcome_labels=[child_label, *parent.outcome_labels[:3]],
        depth_children=[],
        constraint_tags=list(parent.constraint_tags),
    )


def _capsule_usefulness(task: SyntheticTask, labels: list[str], candidate: UIElement) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if any(_any_action_match(action, [candidate.label, *labels]) for action in task.oracle_actions):
        reasons.append("matches_oracle_action")
    goal_terms = _tokens(task.goal)
    label_terms = _tokens(" ".join(labels + [candidate.label]))
    overlap = goal_terms & label_terms
    if len(overlap) >= 2:
        reasons.append("goal_term_overlap:" + ",".join(sorted(list(overlap))[:5]))
    if candidate.role in {"input", "primary", "content"} and candidate.risk < 0.35:
        reasons.append("low_risk_actionable")
    return bool(reasons and "destructive" not in candidate.constraint_tags), reasons


def simulate_task(
    task: SyntheticTask,
    rng: random.Random,
    budget_sec: float,
    max_roots: int,
    max_depth: int,
    deepen_threshold: float,
) -> dict[str, Any]:
    planned_action = task.baseline_actions[0] if task.baseline_actions else Action("noop", "none")
    selected = _choose_root_candidates(task, planned_action, max_roots=max_roots)
    elapsed = 0.0
    transaction_id = 0
    transactions: list[dict[str, Any]] = []
    capsules: list[dict[str, Any]] = []
    rollback_events: list[dict[str, Any]] = []
    constraint_violations: list[dict[str, Any]] = []
    explored_labels: list[str] = []
    max_depth_reached = 0
    budget_exhausted = False

    for root_idx, (candidate, score_parts) in enumerate(selected, start=1):
        if elapsed >= budget_sec:
            budget_exhausted = True
            break
        transaction_id += 1
        depth = 1
        cost = round(rng.uniform(0.22, 0.45), 3)
        elapsed += cost
        rollback = _simulate_rollback(candidate, rng)
        rollback_events.append({"transaction_id": transaction_id, "depth": depth, "candidate": candidate.label, **rollback})
        if "destructive" in candidate.constraint_tags:
            constraint_violations.append({"transaction_id": transaction_id, "candidate": candidate.label, "reason": "destructive_candidate_executed"})

        labels = [candidate.label, *candidate.outcome_labels]
        explored_labels.extend(labels)
        useful, reasons = _capsule_usefulness(task, labels, candidate)
        capsule = {
            "capsule_id": f"{task.task_id}_tx{transaction_id}_d{depth}",
            "transaction_id": transaction_id,
            "path": [candidate.label],
            "depth": depth,
            "root_rank": root_idx,
            "candidate": dataclasses.asdict(candidate),
            "score": score_parts,
            "after_state_labels": candidate.outcome_labels,
            "rollback_verified": rollback["success"],
            "useful": useful and rollback["success"],
            "usefulness_reasons": reasons,
            "risk_band": rollback["risk_band"],
        }
        capsules.append(capsule)
        transactions.append(
            {
                "transaction_id": transaction_id,
                "path": [candidate.label],
                "depth": depth,
                "cost_sec": cost,
                "elapsed_sec": round(elapsed, 3),
                "rollback": rollback,
                "capsule_id": capsule["capsule_id"],
            }
        )
        max_depth_reached = max(max_depth_reached, depth)

        should_deepen = (
            rollback["success"]
            and score_parts["score"] >= deepen_threshold
            and candidate.risk < 0.45
            and candidate.depth_children
        )
        current_parent = candidate
        current_path = [candidate.label]
        while should_deepen and depth < max_depth and elapsed < budget_sec:
            depth += 1
            child_label = current_parent.depth_children[min(depth - 2, len(current_parent.depth_children) - 1)]
            child = _make_deep_candidate(current_parent, child_label, depth)
            transaction_id += 1
            cost = round(rng.uniform(0.32, 0.62), 3)
            elapsed += cost
            rollback = _simulate_rollback(child, rng)
            rollback_events.append({"transaction_id": transaction_id, "depth": depth, "candidate": child.label, **rollback})
            current_path = [*current_path, child.label]
            labels = [child.label, *child.outcome_labels, *current_parent.outcome_labels]
            explored_labels.extend(labels)
            useful, reasons = _capsule_usefulness(task, labels, child)
            capsule = {
                "capsule_id": f"{task.task_id}_tx{transaction_id}_d{depth}",
                "transaction_id": transaction_id,
                "path": current_path,
                "depth": depth,
                "root_rank": root_idx,
                "candidate": dataclasses.asdict(child),
                "score": {
                    "score": max(score_parts["score"] - 0.05 * (depth - 1), 0.0),
                    "goal_similarity": _similarity(task.goal, child.label),
                    "planned_action_similarity": _similarity(planned_action.label, child.label),
                    "role_prior": 0.10,
                    "risk_penalty": round(0.30 * child.risk, 4),
                },
                "after_state_labels": labels,
                "rollback_verified": rollback["success"],
                "useful": useful and rollback["success"],
                "usefulness_reasons": reasons,
                "risk_band": rollback["risk_band"],
            }
            capsules.append(capsule)
            transactions.append(
                {
                    "transaction_id": transaction_id,
                    "path": current_path,
                    "depth": depth,
                    "cost_sec": cost,
                    "elapsed_sec": round(elapsed, 3),
                    "rollback": rollback,
                    "capsule_id": capsule["capsule_id"],
                }
            )
            max_depth_reached = max(max_depth_reached, depth)
            current_parent = child
            should_deepen = rollback["success"] and capsule["useful"] and bool(current_parent.depth_children)

    oracle_hits = [
        _any_action_match(action, explored_labels)
        for action in task.oracle_actions
    ]
    hit_rate = sum(oracle_hits) / max(1, len(oracle_hits))
    useful_capsules = [capsule for capsule in capsules if capsule["useful"]]

    if task.baseline_success:
        simulated_vlm_action = task.oracle_actions[0]
        vlm_source = "baseline_success_action"
    else:
        if useful_capsules and rng.random() < 0.72:
            simulated_vlm_action = task.oracle_actions[0]
            vlm_source = "evidence_corrected_action"
        else:
            simulated_vlm_action = task.baseline_actions[0]
            vlm_source = "baseline_failure_action"
    alignment_match = _any_action_match(simulated_vlm_action, explored_labels)
    aligned_capsules = [
        capsule["capsule_id"]
        for capsule in capsules
        if _any_action_match(simulated_vlm_action, capsule["path"] + capsule["after_state_labels"])
    ]

    failure_reasons: list[str] = []
    if not useful_capsules:
        failure_reasons.append("no_useful_capsule")
    if any(not event["success"] for event in rollback_events):
        failure_reasons.append("rollback_failure")
    if constraint_violations:
        failure_reasons.append("constraint_violation")
    if budget_exhausted:
        failure_reasons.append("budget_exhausted")
    if max_depth_reached < 2:
        failure_reasons.append("evidence_too_shallow")
    if not alignment_match:
        failure_reasons.append("action_conditioned_mismatch")
    if task.failure_mode in {"wrong_input_field", "text_not_cleared"} and hit_rate < 0.75:
        failure_reasons.append("input_semantics_not_precise")

    if useful_capsules and alignment_match and hit_rate >= 0.5 and not constraint_violations:
        analysis = "TPES 产生了通过 rollback 验证、且与下一步 action 对齐的 evidence。"
    elif useful_capsules and not alignment_match:
        analysis = "存在 evidence，但它没有和模拟 VLM action 后到达的页面对齐。"
    elif max_depth_reached < 2:
        analysis = "搜索停留在浅层，缺少可用于 t+1 reasoning 的 t+2 evidence。"
    elif any(not event["success"] for event in rollback_events):
        analysis = "rollback 不稳定，限制了 evidence 的复用。"
    else:
        analysis = "候选元素打分没有把任务关键 transition 排到前面。"

    rollback_safe = not any(not event["success"] for event in rollback_events)
    correction_threshold = 0.75 if task.failure_mode in {"wrong_input_field", "text_not_cleared"} else 0.60
    if task.baseline_success:
        simulated_tpes_success = rollback_safe and not constraint_violations
    else:
        simulated_tpes_success = bool(
            useful_capsules
            and alignment_match
            and hit_rate >= correction_threshold
            and rollback_safe
            and not constraint_violations
        )
    tpes_main_steps = len(task.oracle_actions) if useful_capsules and alignment_match else len(task.baseline_actions)
    step_metrics = {
        "baseline_success": task.baseline_success,
        "baseline_main_steps": len(task.baseline_actions),
        "tpes_success": simulated_tpes_success,
        "tpes_main_steps": tpes_main_steps,
        "tpes_exploration_transactions": len(transactions),
        "tpes_total_steps_with_exploration": tpes_main_steps + len(transactions),
        "tpes_success_rule": (
            "baseline task stays successful only if rollback is safe"
            if task.baseline_success
            else f"failure task is corrected if hit_rate >= {correction_threshold:.2f}, aligned, and rollback is safe"
        ),
    }

    return {
        "task_id": task.task_id,
        "baseline_success": task.baseline_success,
        "goal": task.goal,
        "failure_mode": task.failure_mode,
        "notes": task.notes,
        "budget_sec": budget_sec,
        "elapsed_sec": round(elapsed, 3),
        "planned_action": dataclasses.asdict(planned_action),
        "simulated_vlm_action_after_exploration": dataclasses.asdict(simulated_vlm_action),
        "simulated_vlm_action_source": vlm_source,
        "action_conditioned_alignment": {
            "matched": alignment_match,
            "matched_capsules": aligned_capsules,
        },
        "hit_rate_vs_oracle_success_path": round(hit_rate, 4),
        "oracle_action_hits": [
            {"action": action.label, "hit": hit}
            for action, hit in zip(task.oracle_actions, oracle_hits)
        ],
        "found_useful_evidence": bool(useful_capsules and alignment_match),
        "max_depth_reached": max_depth_reached,
        "transaction_count": len(transactions),
        "rollback_count": len(rollback_events),
        "rollback_failures": [event for event in rollback_events if not event["success"]],
        "constraint_violations": constraint_violations,
        "evidence_capsules": capsules,
        "transactions": transactions,
        "failure_reasons": failure_reasons,
        "analysis": analysis,
        "step_metrics": step_metrics,
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    failure_counter: Counter[str] = Counter()
    for result in results:
        failure_counter.update(result["failure_reasons"])
    by_baseline: dict[str, dict[str, float]] = {}
    for key, rows in {
        "baseline_success_tasks": [r for r in results if r["baseline_success"]],
        "baseline_failure_tasks": [r for r in results if not r["baseline_success"]],
    }.items():
        by_baseline[key] = {
            "num_tasks": len(rows),
            "avg_hit_rate": round(sum(r["hit_rate_vs_oracle_success_path"] for r in rows) / max(1, len(rows)), 4),
            "avg_max_depth": round(sum(r["max_depth_reached"] for r in rows) / max(1, len(rows)), 4),
            "useful_evidence_rate": round(sum(bool(r["found_useful_evidence"]) for r in rows) / max(1, len(rows)), 4),
            "avg_baseline_main_steps": round(_mean([r["step_metrics"]["baseline_main_steps"] for r in rows]), 4),
            "avg_tpes_main_steps": round(_mean([r["step_metrics"]["tpes_main_steps"] for r in rows]), 4),
            "avg_tpes_total_steps_with_exploration": round(
                _mean([r["step_metrics"]["tpes_total_steps_with_exploration"] for r in rows]), 4
            ),
        }
    baseline_success_rows = [r for r in results if r["step_metrics"]["baseline_success"]]
    tpes_success_rows = [r for r in results if r["step_metrics"]["tpes_success"]]
    recommendations = [
        "只在 root transition 已通过 rollback verification 后选择性加深到 depth-2/3；答案直接可见的任务浅层 evidence 通常足够，但表单编辑、多应用跳转和文件打开任务需要更深 evidence。",
        "增强输入框消歧：结合 label 距离、focused field text、期望值类型，并在 type 之前支持 clear_text，避免把目标文本追加到默认值后面。",
        "对 delete、map pan、share sheet、settings 等高风险动作加惩罚；除非 task goal 或 VLM planned action 明确需要，否则不应优先探索。",
        "保持严格 rollback verifier；rollback 失败的 evidence capsule 只能记录到 trace，不能注入下一步 prompt。",
        "统计 action-conditioned alignment，而不是只比较原始 state similarity；只有与 VLM 实际 action 后到达页面匹配的 evidence，才算真正能增强 reasoning。",
    ]
    return {
        "num_tasks": len(results),
        "average_hit_rate": round(sum(r["hit_rate_vs_oracle_success_path"] for r in results) / max(1, len(results)), 4),
        "average_max_depth": round(sum(r["max_depth_reached"] for r in results) / max(1, len(results)), 4),
        "useful_evidence_rate": round(sum(bool(r["found_useful_evidence"]) for r in results) / max(1, len(results)), 4),
        "baseline_success_rate": round(_mean([int(r["step_metrics"]["baseline_success"]) for r in results]), 4),
        "tpes_success_rate": round(_mean([int(r["step_metrics"]["tpes_success"]) for r in results]), 4),
        "average_baseline_main_steps_all_tasks": round(_mean([r["step_metrics"]["baseline_main_steps"] for r in results]), 4),
        "average_tpes_main_steps_all_tasks": round(_mean([r["step_metrics"]["tpes_main_steps"] for r in results]), 4),
        "average_tpes_exploration_transactions_all_tasks": round(
            _mean([r["step_metrics"]["tpes_exploration_transactions"] for r in results]), 4
        ),
        "average_tpes_total_steps_with_exploration_all_tasks": round(
            _mean([r["step_metrics"]["tpes_total_steps_with_exploration"] for r in results]), 4
        ),
        "average_baseline_main_steps_success_tasks": round(
            _mean([r["step_metrics"]["baseline_main_steps"] for r in baseline_success_rows]), 4
        ),
        "average_tpes_main_steps_success_tasks": round(
            _mean([r["step_metrics"]["tpes_main_steps"] for r in tpes_success_rows]), 4
        ),
        "average_tpes_total_steps_with_exploration_success_tasks": round(
            _mean([r["step_metrics"]["tpes_total_steps_with_exploration"] for r in tpes_success_rows]), 4
        ),
        "rollback_failure_rate": round(
            sum(len(r["rollback_failures"]) for r in results) / max(1, sum(r["rollback_count"] for r in results)),
            4,
        ),
        "constraint_violation_count": sum(len(r["constraint_violations"]) for r in results),
        "failure_reason_counts": dict(failure_counter.most_common()),
        "by_baseline_group": by_baseline,
        "recommendations": recommendations,
        "design_aspects": {
            "root_stratification": "先从 primary/input/content/navigation/risky 分层各选高分候选，再用总分补足名额，避免探索只集中在同一类元素。",
            "selective_deepening": "只有 root 分数过阈值、rollback 成功、risk 低于 0.45 且仍有预算时才继续加深。",
            "rollback_policy": "每个 transaction 都必须 rollback；rollback 失败的 evidence 仍写入 trace，但不计为可用 evidence。",
            "alignment_metric": "Evidence 必须匹配模拟 VLM action 后的下一页面，而不是只匹配 exploration 起点页面。",
            "budget_model": "每个任务分配 1-2 秒模拟 VLM-side exploration 时间窗口。",
        },
    }


def _write_charts(payload: dict[str, Any], run_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = payload["tasks"]
    summary = payload["summary"]
    charts: dict[str, str] = {}

    def save_current(name: str) -> None:
        path = run_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        charts[name] = str(path.resolve())

    plt.figure(figsize=(6.4, 4.2))
    labels = ["Baseline", "TPES"]
    values = [summary["baseline_success_rate"] * 100, summary["tpes_success_rate"] * 100]
    bars = plt.bar(labels, values, color=["#7f8c8d", "#2f6f9f"])
    plt.ylabel("success rate (%)")
    plt.title("Baseline vs TPES success rate")
    plt.ylim(0, 105)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", va="bottom")
    save_current("success_rate_baseline_vs_tpes.png")

    plt.figure(figsize=(7.6, 4.6))
    labels = ["All tasks", "Successful tasks"]
    baseline_values = [
        summary["average_baseline_main_steps_all_tasks"],
        summary["average_baseline_main_steps_success_tasks"],
    ]
    tpes_values = [
        summary["average_tpes_main_steps_all_tasks"],
        summary["average_tpes_main_steps_success_tasks"],
    ]
    tpes_total_values = [
        summary["average_tpes_total_steps_with_exploration_all_tasks"],
        summary["average_tpes_total_steps_with_exploration_success_tasks"],
    ]
    x = list(range(len(labels)))
    width = 0.25
    plt.bar([i - width for i in x], baseline_values, width=width, label="Baseline main", color="#7f8c8d")
    plt.bar(x, tpes_values, width=width, label="TPES main", color="#2f6f9f")
    plt.bar([i + width for i in x], tpes_total_values, width=width, label="TPES total + exploration", color="#d08730")
    plt.xticks(x, labels)
    plt.ylabel("average steps / transactions")
    plt.title("Average steps comparison")
    plt.legend()
    save_current("average_steps_baseline_vs_tpes.png")

    task_labels = [r["task_id"].replace("_", "\n", 1) for r in results]
    base_steps = [r["step_metrics"]["baseline_main_steps"] for r in results]
    tpes_main = [r["step_metrics"]["tpes_main_steps"] for r in results]
    tpes_total = [r["step_metrics"]["tpes_total_steps_with_exploration"] for r in results]
    x = list(range(len(results)))
    width = 0.25
    plt.figure(figsize=(13.5, 5.4))
    plt.bar([i - width for i in x], base_steps, width=width, label="Baseline main", color="#7f8c8d")
    plt.bar(x, tpes_main, width=width, label="TPES main", color="#2f6f9f")
    plt.bar([i + width for i in x], tpes_total, width=width, label="TPES total + exploration", color="#d08730")
    plt.xticks(x, task_labels, rotation=35, ha="right", fontsize=8)
    plt.ylabel("steps / transactions")
    plt.title("Per-task step comparison")
    plt.legend()
    save_current("per_task_steps_baseline_vs_tpes.png")

    return charts


def _write_report(payload: dict[str, Any], report_path: Path) -> None:
    summary = payload["summary"]
    results = payload["tasks"]
    charts = payload.get("charts", {})
    lines: list[str] = [
        "# TPES 仿真实验报告",
        "",
        "本报告使用 10 个 synthetic mobile GUI task 测试 TPES（Transactional Progressive Evidence Search）。实验不调用真实 VLM，只用轻量关键词匹配、规则打分和 deterministic random 来模拟探索、rollback 和证据注入。",
        "",
        "## 总体结果",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 任务数 | {summary['num_tasks']} |",
        f"| 平均 hit rate | {summary['average_hit_rate']:.2f} |",
        f"| 平均最大深度 | {summary['average_max_depth']:.2f} |",
        f"| baseline success rate | {summary['baseline_success_rate']:.2f} |",
        f"| TPES success rate | {summary['tpes_success_rate']:.2f} |",
        f"| baseline 平均主流程步数（全部任务） | {summary['average_baseline_main_steps_all_tasks']:.2f} |",
        f"| TPES 平均主流程步数（全部任务） | {summary['average_tpes_main_steps_all_tasks']:.2f} |",
        f"| TPES 平均 exploration transactions（全部任务） | {summary['average_tpes_exploration_transactions_all_tasks']:.2f} |",
        f"| TPES 平均总步数+探索开销（全部任务） | {summary['average_tpes_total_steps_with_exploration_all_tasks']:.2f} |",
        f"| baseline 成功任务平均主流程步数 | {summary['average_baseline_main_steps_success_tasks']:.2f} |",
        f"| TPES 成功任务平均主流程步数 | {summary['average_tpes_main_steps_success_tasks']:.2f} |",
        f"| TPES 成功任务平均总步数+探索开销 | {summary['average_tpes_total_steps_with_exploration_success_tasks']:.2f} |",
        f"| useful evidence rate | {summary['useful_evidence_rate']:.2f} |",
        f"| rollback failure rate | {summary['rollback_failure_rate']:.2f} |",
        f"| constraint violations | {summary['constraint_violation_count']} |",
        "",
        "这里的 `main steps` 表示任务主流程动作数，可近似对应 AndroidWorld episode length；`TPES total + exploration` 是主流程动作数加 exploration transaction 数，用来观察额外探索开销。",
        "",
        "## 图表",
        "",
    ]
    if charts:
        chart_titles = {
            "success_rate_baseline_vs_tpes.png": "成功率对比",
            "average_steps_baseline_vs_tpes.png": "平均步数对比",
            "per_task_steps_baseline_vs_tpes.png": "逐任务步数对比",
        }
        for name, path in charts.items():
            title = chart_titles.get(name, name)
            lines.extend([f"### {title}", "", f"![{title}]({path})", ""])

    lines.extend(
        [
            "## 分组统计",
            "",
        "按 baseline 成败分组：",
        "",
            "| 分组 | 任务数 | 平均 hit rate | 平均最大深度 | useful evidence rate | baseline 平均主流程步数 | TPES 平均主流程步数 | TPES 平均总步数+探索 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, values in summary["by_baseline_group"].items():
        lines.append(
            f"| {group} | {values['num_tasks']} | {values['avg_hit_rate']:.2f} | {values['avg_max_depth']:.2f} | {values['useful_evidence_rate']:.2f} | {values['avg_baseline_main_steps']:.2f} | {values['avg_tpes_main_steps']:.2f} | {values['avg_tpes_total_steps_with_exploration']:.2f} |"
        )

    lines.extend(
        [
            "",
            "常见失败原因：",
            "",
            "| 原因 | 次数 |",
            "| --- | ---: |",
        ]
    )
    for reason, count in summary["failure_reason_counts"].items():
        lines.append(f"| {reason} | {count} |")

    lines.extend(
        [
            "",
            "## Per-task 结果",
            "",
            "| task | baseline | TPES success | baseline steps | TPES main steps | TPES total+explore | useful evidence | hit rate | max depth | rollback fail | constraint | analysis |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in results:
        baseline = "success" if result["baseline_success"] else "failure"
        steps = result["step_metrics"]
        lines.append(
            "| {task} | {baseline} | {tpes_success} | {base_steps} | {tpes_main} | {tpes_total} | {useful} | {hit:.2f} | {depth} | {rb} | {cv} | {analysis} |".format(
                task=result["task_id"],
                baseline=baseline,
                tpes_success=int(bool(steps["tpes_success"])),
                base_steps=steps["baseline_main_steps"],
                tpes_main=steps["tpes_main_steps"],
                tpes_total=steps["tpes_total_steps_with_exploration"],
                useful=int(bool(result["found_useful_evidence"])),
                hit=float(result["hit_rate_vs_oracle_success_path"]),
                depth=int(result["max_depth_reached"]),
                rb=len(result["rollback_failures"]),
                cv=len(result["constraint_violations"]),
                analysis=result["analysis"],
            )
        )

    lines.extend(
        [
            "",
            "## 设计因素分析",
            "",
            "- root-level stratified search 能避免只盯着最高相似度元素。对 `TasksHighPriority`、`CalendarTimeRange` 这类答案可见任务，浅层 content evidence 就足够。",
            "- selective deepening 对创建笔记、打开文件、跨应用复制这类任务更关键，因为需要先匹配 t+1 页面，再利用 t+2/t+3 证据指导下一步。",
            "- rollback 是 TPES 的硬约束。仿真中高风险 action（delete、map pan、share sheet）更容易产生 rollback failure；这些 capsule 即使信息有用，也不能直接注入 prompt。",
            "- 仅做 state similarity 不够。报告中使用 action-conditioned alignment，要求 evidence 与模拟 VLM 实际执行后的页面/动作一致。",
            "- input 类任务的失败主要来自 field disambiguation 和 clear_text 缺失，例如 Audio rename 和 Contacts phone field。",
            "",
            "## 改进建议",
            "",
        ]
    )
    for item in summary["recommendations"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## JSON 输出",
            "",
            f"- Structured JSON: `{payload['output_json']}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget_min", type=float, default=1.0)
    parser.add_argument("--budget_max", type=float, default=2.0)
    parser.add_argument("--max_roots", type=int, default=4)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--deepen_threshold", type=float, default=0.22)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    run_dir = args.output_root / f"run_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tasks = _make_taskbed()
    results: list[dict[str, Any]] = []
    for task in tasks:
        budget = round(rng.uniform(float(args.budget_min), float(args.budget_max)), 3)
        results.append(
            simulate_task(
                task,
                rng=rng,
                budget_sec=budget,
                max_roots=int(args.max_roots),
                max_depth=int(args.max_depth),
                deepen_threshold=float(args.deepen_threshold),
            )
        )

    output_json = run_dir / "tpes_simulation_results.json"
    report_path = run_dir / "tpes_simulation_report_cn.md"
    payload = {
        "metadata": {
            "strategy": "TPES",
            "seed": args.seed,
            "budget_sec_range": [args.budget_min, args.budget_max],
            "max_roots": args.max_roots,
            "max_depth": args.max_depth,
            "deepen_threshold": args.deepen_threshold,
        },
        "summary": _summarize(results),
        "tasks": results,
    }
    payload["output_json"] = str(output_json.resolve())
    payload["charts"] = _write_charts(payload, run_dir)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(payload, report_path)
    print(
        json.dumps(
            {
                "json": str(output_json),
                "report": str(report_path),
                "charts": payload["charts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
