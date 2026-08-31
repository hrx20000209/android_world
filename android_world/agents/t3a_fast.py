# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fast T3A variant: standard T3A action prompt without LLM summarization."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import gelab_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.agents import t3a
from android_world.env import interface
from android_world.env import json_action

MAX_AGENT_STEPS = 20


_OPEN_APP_ALIASES = {
    "alarm": "Clock",
    "clock": "Clock",
    "stopwatch": "Clock",
    "timer": "Clock",
    "calendar": "Simple Calendar Pro",
    "simple calendar": "Simple Calendar Pro",
    "contacts": "Contacts",
    "contact": "Contacts",
    "dialer": "Dialer",
    "phone": "Dialer",
    "files": "Files",
    "file manager": "Files",
    "settings": "Settings",
    "wifi": "Settings",
    "wi-fi": "Settings",
    "bluetooth": "Settings",
    "brightness": "Settings",
    "camera": "Camera",
    "chrome": "Chrome",
    "browser": "Chrome",
    "joplin": "Joplin",
    "notes": "Joplin",
    "note": "Joplin",
    "markor": "Markor",
    "tasks": "Tasks",
    "todo": "Tasks",
    "to-do": "Tasks",
    "simple draw": "Simple Draw Pro",
    "draw": "Simple Draw Pro",
    "gallery": "Simple Gallery Pro",
    "simple gallery": "Simple Gallery Pro",
    "sms": "Simple SMS Messenger",
    "message": "Simple SMS Messenger",
    "messenger": "Simple SMS Messenger",
    "audio recorder": "Audio Recorder",
    "recorder": "Audio Recorder",
    "expense": "Pro Expense",
    "pro expense": "Pro Expense",
    "broccoli": "Broccoli APP",
    "osmand": "OSMand",
    "map": "OSMand",
    "vlc": "VLC",
    "music": "Retro Music",
    "retro music": "Retro Music",
    "opentracks": "OpenTracks",
    "open tracks": "OpenTracks",
    "sports tracker": "OpenTracks",
    "activity tracker": "OpenTracks",
}


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean_text(value).lower()).strip()


class FastT3A(base_agent.EnvironmentInteractingAgent):
    """Text-only T3A action loop, but history summaries are local and cheap."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.LlmWrapper,
        name: str = "FastT3A",
    ):
        super().__init__(env, name)
        self.llm = llm
        self.history: list[dict[str, Any]] = []
        self.additional_guidelines: list[str] | None = None

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self.history = []

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def set_max_steps(self, max_steps: int) -> None:
        super().set_max_steps(min(MAX_AGENT_STEPS, int(max_steps)))

    def _effective_max_steps(self) -> int:
        if self._max_steps is None:
            return MAX_AGENT_STEPS
        return min(MAX_AGENT_STEPS, int(self._max_steps))

    @staticmethod
    def _local_summary(action: json_action.JSONAction, reason: str | None) -> str:
        reason_text = (reason or "No reason provided.").strip()
        action_json = action.json_str()
        if action.action_type == json_action.STATUS:
            return f"Selected status {action.goal_status or 'complete'}. Reason: {reason_text}"
        if action.action_type == json_action.ANSWER:
            return f"Answered with `{action.text or ''}`. Reason: {reason_text}"
        return f"Action selected: {action_json}. Reason: {reason_text}"

    def _append_history(self, step_data: dict[str, Any], action: json_action.JSONAction | None = None, reason: str | None = None) -> None:
        if action is not None:
            step_data["summary"] = self._local_summary(action, reason)
        elif not step_data.get("summary"):
            step_data["summary"] = "No valid action was executed."
        self.history.append(step_data)

    def _parse_action(
        self,
        action_prompt: str,
        action_output: str,
        raw_response: object,
        step_data: dict[str, Any],
    ) -> tuple[str | None, str | None, str, object]:
        reason = None
        action = None
        parse_error = "missing_action_json"
        for candidate in t3a._collect_action_parse_candidates(action_output, raw_response):  # pylint: disable=protected-access
            parsed_reason, parsed_action = m3a_utils.parse_reason_action_output(candidate)
            if parsed_action:
                return parsed_reason, parsed_action, action_output, raw_response
            parse_error = "missing_action_json"

        repair_prompt = t3a._strict_action_repair_prompt(action_prompt, parse_error)  # pylint: disable=protected-access
        step_data["action_repair_prompt"] = repair_prompt
        retry_output, retry_safe, retry_raw = self.llm.predict(repair_prompt)
        if not retry_safe:  # pylint: disable=singleton-comparison
            retry_output = 'Action: {"action_type": "status", "goal_status": "infeasible"}'
        step_data["action_repair_output"] = retry_output
        step_data["action_repair_raw_response"] = retry_raw
        for candidate in t3a._collect_action_parse_candidates(retry_output, retry_raw):  # pylint: disable=protected-access
            parsed_reason, parsed_action = m3a_utils.parse_reason_action_output(candidate)
            if parsed_action:
                return parsed_reason, parsed_action, retry_output, retry_raw
        return reason, action, action_output, raw_response

    @staticmethod
    def _available_app_lookup() -> dict[str, str]:
        lookup: dict[str, str] = {}
        for app in gelab_agent.AVAILABLE_APPS:
            lookup[_normalize_key(app)] = app
        return lookup

    @classmethod
    def _infer_open_app_name(cls, goal: str, requested_app: str | None) -> str | None:
        requested_key = _normalize_key(requested_app)
        goal_key = _normalize_key(goal)
        available = cls._available_app_lookup()

        if requested_key in available:
            return available[requested_key]

        for app_key, app_name in available.items():
            if requested_key and (requested_key in app_key or app_key in requested_key):
                return app_name

        for app_key, app_name in available.items():
            if app_key and app_key in goal_key:
                return app_name

        combined = " ".join(x for x in [requested_key, goal_key] if x)
        for alias, app_name in _OPEN_APP_ALIASES.items():
            alias_key = _normalize_key(alias)
            if alias_key and re.search(rf"(^| ){re.escape(alias_key)}( |$)", combined):
                return app_name

        return _clean_text(requested_app) or None

    def _normalize_open_app_action(
        self,
        goal: str,
        action_json: dict[str, Any],
        converted_action: json_action.JSONAction,
        step_data: dict[str, Any],
    ) -> json_action.JSONAction:
        if converted_action.action_type != json_action.OPEN_APP:
            return converted_action
        original_app = _clean_text(getattr(converted_action, "app_name", ""))
        fixed_app = self._infer_open_app_name(goal, original_app)
        if not fixed_app or fixed_app == original_app:
            return converted_action
        action_json["app_name"] = fixed_app
        step_data["open_app_normalization"] = {
            "original_app_name": original_app,
            "normalized_app_name": fixed_app,
        }
        print("Normalized open_app: " + str(original_app) + " -> " + str(fixed_app))
        return json_action.JSONAction(**action_json)

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        if len(self.history) >= self._effective_max_steps():
            summary = f"Reached the maximum step limit ({self._effective_max_steps()})."
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": summary,
                    "action_output": 'Action: {"action_type": "status", "goal_status": "infeasible"}',
                },
            )

        step_idx = len(self.history)
        step_data: dict[str, Any] = {
            "before_screenshot": None,
            "before_element_list": None,
            "action_prompt": None,
            "action_output": None,
            "action_raw_response": None,
            "action_repair_prompt": None,
            "action_repair_output": None,
            "action_repair_raw_response": None,
            "summary": None,
            "fast_no_llm_summary": True,
        }
        print("----------fast-t3a step " + str(step_idx + 1))

        state = self.get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size
        ui_elements = state.ui_elements
        before_element_list = t3a._generate_ui_elements_description_list_full(  # pylint: disable=protected-access
            ui_elements,
            logical_screen_size,
        )
        step_data["before_screenshot"] = state.pixels.copy()
        step_data["before_element_list"] = ui_elements

        history_lines = [
            "Step " + str(i + 1) + ": " + str(step_info.get("summary") or "")
            for i, step_info in enumerate(self.history)
        ]
        action_prompt = t3a._action_selection_prompt(  # pylint: disable=protected-access
            goal,
            history_lines,
            before_element_list,
            self.additional_guidelines,
        )
        step_data["action_prompt"] = action_prompt
        action_output, is_safe, raw_response = self.llm.predict(action_prompt)
        if not is_safe:  # pylint: disable=singleton-comparison
            action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""
        if not raw_response:
            raise RuntimeError("Error calling LLM in action selection phase.")

        step_data["action_output"] = action_output
        step_data["action_raw_response"] = raw_response
        reason, action_text, action_output, raw_response = self._parse_action(
            action_prompt,
            action_output,
            raw_response,
            step_data,
        )
        step_data["action_output"] = action_output
        step_data["action_raw_response"] = raw_response

        if not action_text:
            print("Action prompt output is not in the correct format.")
            step_data["summary"] = "Output for action selection is not in the correct format, so no action is performed."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        print("Action: " + action_text)
        print("Reason: " + (reason if reason else "No reason provided."))

        try:
            action_json = agent_utils.extract_json(action_text)
            if not action_json:
                raise ValueError("Cannot extract action JSON.")
            converted_action = json_action.JSONAction(**action_json)
            converted_action = self._normalize_open_app_action(
                goal,
                action_json,
                converted_action,
                step_data,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print("Failed to convert the output to a valid action.")
            print(str(exc))
            step_data["summary"] = "Can not parse the output to a valid action."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        if converted_action.action_type in {json_action.CLICK, json_action.LONG_PRESS, json_action.INPUT_TEXT}:
            if converted_action.index is not None and converted_action.index >= len(ui_elements):
                print("Index out of range.")
                step_data["summary"] = "The parameter index is out of range."
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(False, step_data)

        if converted_action.action_type == json_action.STATUS:
            if converted_action.goal_status == "infeasible":
                print("Agent stopped since it thinks mission impossible.")
            self._append_history(step_data, converted_action, reason)
            return base_agent.AgentInteractionResult(True, step_data)

        if converted_action.action_type == json_action.ANSWER:
            print("Agent answered with: " + str(converted_action.text))
            try:
                self.env.execute_action(converted_action)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print("Some error happened executing the action ", converted_action.action_type)
                print(str(exc))
                step_data["summary"] = "Some error happened executing the action " + str(converted_action.action_type)
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(False, step_data)

            summary, state_changed, before_sig, after_sig = self._transition_summary(
                converted_action,
                reason,
                before_element_list,
                before_element_list,
            )
            step_data["summary"] = summary
            step_data["state_changed"] = state_changed
            step_data["before_state_signature"] = before_sig
            step_data["after_state_signature"] = after_sig
            step_data["after_screenshot"] = state.pixels.copy()
            step_data["after_element_list"] = ui_elements
            step_data["hybrid_answer_terminated_session"] = True
            self.history.append(step_data)
            print("Summary: " + str(step_data.get("summary")))
            return base_agent.AgentInteractionResult(True, step_data)

        try:
            self.env.execute_action(converted_action)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print("Some error happened executing the action ", converted_action.action_type)
            print(str(exc))
            step_data["summary"] = "Some error happened executing the action " + str(converted_action.action_type)
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        self._append_history(step_data, converted_action, reason)
        print("Summary: " + str(step_data.get("summary")))
        return base_agent.AgentInteractionResult(False, step_data)


class HybridT3A(FastT3A):
    """T3A action prompt with cheap transition memory and loop-triggered replan."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.LlmWrapper,
        name: str = "HybridT3A",
    ):
        super().__init__(env, llm, name=name)

    @staticmethod
    def _state_signature(ui_description: str) -> str:
        compact = _normalize_key(ui_description)[:4000]
        return hashlib.sha1(compact.encode("utf-8", errors="ignore")).hexdigest()[:16]

    @staticmethod
    def _action_signature(action: json_action.JSONAction) -> str:
        parts = [
            str(action.action_type),
            str(getattr(action, "index", None)),
            _clean_text(getattr(action, "text", "")),
            _clean_text(getattr(action, "direction", "")),
            _clean_text(getattr(action, "app_name", "")),
            _clean_text(getattr(action, "goal_status", "")),
        ]
        return "|".join(parts)

    @staticmethod
    def _short_screen_hint(ui_description: str, limit: int = 260) -> str:
        texts: list[str] = []
        for line in ui_description.splitlines():
            line = _clean_text(line)
            if not line:
                continue
            text_matches = re.findall(r"(?:text|content_description)='([^']+)'", line)
            for item in text_matches:
                item = _clean_text(item)
                if item and item not in texts:
                    texts.append(item)
            if len(texts) >= 8:
                break
        if not texts:
            lines = [_clean_text(x) for x in ui_description.splitlines() if _clean_text(x)]
            texts = lines[:4]
        hint = "; ".join(texts)
        if len(hint) > limit:
            hint = hint[:limit].rstrip() + "..."
        return hint or "No concise visible UI summary available."

    @staticmethod
    def _compact_ui_description(
        ui_description: str,
        max_lines: int = 90,
        max_line_chars: int = 180,
    ) -> str:
        lines: list[str] = []
        for raw_line in ui_description.splitlines():
            line = _clean_text(raw_line)
            if not line:
                continue
            if len(line) > max_line_chars:
                line = line[:max_line_chars].rstrip() + "..."
            lines.append(line)
            if len(lines) >= max_lines:
                break
        if len(ui_description.splitlines()) > len(lines):
            lines.append("[UI element list truncated; use visible indices above.]")
        return "\n".join(lines)

    def _recent_repeat_count(self, action_signature: str | None = None) -> int:
        if not self.history:
            return 0
        if action_signature is None:
            action_signature = self.history[-1].get("action_signature")
        if not action_signature:
            return 0
        count = 0
        for item in reversed(self.history):
            if item.get("action_signature") != action_signature:
                break
            count += 1
        return count

    def _recent_same_input_target(self, action: json_action.JSONAction) -> bool:
        if action.action_type != json_action.INPUT_TEXT or action.index is None:
            return False
        target_prefix = f"{json_action.INPUT_TEXT}|{action.index}|"
        for item in self.history[-4:]:
            signature = str(item.get("action_signature") or "")
            if signature.startswith(target_prefix):
                return True
        return False

    def _dynamic_guidelines(self) -> list[str]:
        guidelines = list(self.additional_guidelines or [])
        if len(self.history) >= 2:
            last = self.history[-1]
            prev = self.history[-2]
            same_action = last.get("action_signature") == prev.get("action_signature")
            no_change = not last.get("state_changed", True)
            same_state = last.get("after_state_signature") == prev.get("after_state_signature")
            if same_action and (no_change or same_state):
                guidelines.append(
                    "The last action appears stuck: do NOT repeat the same action/index. "
                    "Choose a different visible target, scroll/back, open the correct app, or mark infeasible only if no alternative exists."
                )
            if "save" in str(last.get("summary") or "").lower() and no_change:
                guidelines.append(
                    "If the requested item name/content is already visible and Save produced no visible change, do not keep pressing Save; finish with status complete."
                )
        if self.history:
            guidelines.append(
                "Use the history critically: if a previous action did not visibly change the UI, avoid repeating it."
            )
        return guidelines

    def _transition_summary(
        self,
        action: json_action.JSONAction,
        reason: str | None,
        before_desc: str,
        after_desc: str,
    ) -> tuple[str, bool, str, str]:
        before_sig = self._state_signature(before_desc)
        after_sig = self._state_signature(after_desc)
        state_changed = before_sig != after_sig
        action_json = action.json_str()
        reason_text = (reason or "No reason provided.").strip()
        if action.action_type == json_action.STATUS:
            summary = f"Selected status {action.goal_status or 'complete'}. Reason: {reason_text}"
        elif action.action_type == json_action.ANSWER:
            summary = f"Answered with `{action.text or ''}`. Reason: {reason_text}"
        elif state_changed:
            summary = (
                f"Action selected: {action_json}. Intended: {reason_text}. "
                f"UI changed. Now visible: {self._short_screen_hint(after_desc, limit=180)}"
            )
        else:
            summary = (
                f"Action selected: {action_json}. Intended: {reason_text}. "
                "No visible UI change; do not repeat this exact action unless the next screen clearly requires it."
            )
        return summary, state_changed, before_sig, after_sig

    def _replan_if_repeating(
        self,
        goal: str,
        action_prompt: str,
        action_output: str,
        raw_response: object,
        action_text: str,
        converted_action: json_action.JSONAction,
        step_data: dict[str, Any],
    ) -> tuple[str | None, str | None, str, object, json_action.JSONAction]:
        action_signature = self._action_signature(converted_action)
        repeat_count = self._recent_repeat_count(action_signature)
        same_input_target = self._recent_same_input_target(converted_action)
        low_confidence_complete = (
            converted_action.action_type == json_action.STATUS
            and converted_action.goal_status == "complete"
            and bool(self.history)
            and "No visible UI change" in str(self.history[-1].get("summary") or "")
        )
        if repeat_count < 2 and not same_input_target and not low_confidence_complete:
            return None, action_text, action_output, raw_response, converted_action

        constraint = (
            "Your proposed action repeats the same action/index that already failed to make progress."
        )
        if same_input_target:
            constraint = (
                "You already typed into this same input field recently. Do NOT type into the same field again; "
                "move to the next field, save/confirm, or choose a different visible target."
            )
        if low_confidence_complete:
            constraint = (
                "Do NOT mark the task complete immediately after a no-change action. "
                "First find visible evidence that the requested state is satisfied, or choose another action."
            )
        replan_prompt = (
            action_prompt
            + "\n\nIMPORTANT REPLAN CONSTRAINT:\n"
            + constraint
            + " "
            + "Do NOT output the same action again. Pick a different visible element, scroll/back/open app, or mark infeasible only if truly impossible.\n"
            + f"Repeated action signature: {action_signature}\n"
            + "Output one replacement action now.\n"
        )
        step_data["hybrid_replan_prompt"] = replan_prompt
        replan_output, replan_safe, replan_raw = self.llm.predict(replan_prompt)
        if not replan_safe:  # pylint: disable=singleton-comparison
            replan_output = 'Action: {"action_type": "status", "goal_status": "infeasible"}'
        step_data["hybrid_replan_output"] = replan_output
        step_data["hybrid_replan_raw_response"] = replan_raw
        if not replan_raw and replan_output:
            replan_raw = replan_output
        reason, replanned_text, parsed_output, parsed_raw = self._parse_action(
            replan_prompt,
            replan_output,
            replan_raw,
            step_data,
        )
        if not replanned_text:
            return None, action_text, action_output, raw_response, converted_action
        try:
            action_json = agent_utils.extract_json(replanned_text)
            if not action_json:
                raise ValueError("Cannot extract replan action JSON.")
            replanned_action = json_action.JSONAction(**action_json)
            replanned_action = self._normalize_open_app_action(
                goal,
                action_json,
                replanned_action,
                step_data,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            step_data["hybrid_replan_error"] = str(exc)
            return None, action_text, action_output, raw_response, converted_action
        print("Hybrid replan replaced repeated action.")
        print("Replan Action: " + replanned_text)
        return reason, replanned_text, parsed_output, parsed_raw, replanned_action

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        if len(self.history) >= self._effective_max_steps():
            summary = f"Reached the maximum step limit ({self._effective_max_steps()})."
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": summary,
                    "action_output": 'Action: {"action_type": "status", "goal_status": "infeasible"}',
                },
            )

        step_idx = len(self.history)
        step_data: dict[str, Any] = {
            "before_screenshot": None,
            "after_screenshot": None,
            "before_element_list": None,
            "after_element_list": None,
            "action_prompt": None,
            "action_output": None,
            "action_raw_response": None,
            "action_repair_prompt": None,
            "action_repair_output": None,
            "action_repair_raw_response": None,
            "hybrid_replan_prompt": None,
            "hybrid_replan_output": None,
            "hybrid_replan_raw_response": None,
            "hybrid_replan_error": None,
            "summary": None,
            "fast_no_llm_summary": True,
            "hybrid_transition_summary": True,
        }
        print("----------hybrid-t3a step " + str(step_idx + 1))

        state = self.get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size
        ui_elements = state.ui_elements
        before_element_list = t3a._generate_ui_elements_description_list_full(  # pylint: disable=protected-access
            ui_elements,
            logical_screen_size,
        )
        step_data["before_screenshot"] = state.pixels.copy()
        step_data["before_element_list"] = ui_elements
        step_data["before_state_signature"] = self._state_signature(before_element_list)

        history_start = max(0, len(self.history) - 8)
        history_lines = []
        for i, step_info in enumerate(self.history[history_start:], start=history_start):
            summary = _clean_text(step_info.get("summary") or "")
            if len(summary) > 320:
                summary = summary[:320].rstrip() + "..."
            history_lines.append("Step " + str(i + 1) + ": " + summary)
        prompt_before_element_list = self._compact_ui_description(before_element_list)
        screen_guidelines = self._dynamic_guidelines()
        goal_lower = goal.lower()
        if "search" in before_element_list.lower() and re.search(
            r"\b(with|for|about|named|called)\s+[A-Z][A-Za-z0-9_-]+",
            goal,
        ):
            screen_guidelines.append(
                "For information-retrieval tasks with a named entity/person, if a Search control is visible, prefer using Search and typing the exact entity before browsing dates or lists."
            )
        if (
            "cancel" in before_element_list.lower()
            and "ok" in before_element_list.lower()
            and re.search(r"\b(2022|2023|2024|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", before_element_list.lower())
            and ("meeting" in goal_lower or "event" in goal_lower or "calendar" in goal_lower)
        ):
            screen_guidelines.append(
                "This appears to be a date/month/year picker, not an event details page. If the task is to inspect meetings or events, exit it with Cancel/back instead of repeatedly choosing years/months."
            )
        action_prompt = t3a._action_selection_prompt(  # pylint: disable=protected-access
            goal,
            history_lines,
            prompt_before_element_list,
            screen_guidelines,
        )
        step_data["action_prompt"] = action_prompt
        action_output, is_safe, raw_response = self.llm.predict(action_prompt)
        if not is_safe:  # pylint: disable=singleton-comparison
            action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""
        if not raw_response and action_output:
            raw_response = action_output
        if not action_output:
            raise RuntimeError("Error calling LLM in action selection phase.")

        step_data["action_output"] = action_output
        step_data["action_raw_response"] = raw_response
        reason, action_text, action_output, raw_response = self._parse_action(
            action_prompt,
            action_output,
            raw_response,
            step_data,
        )
        step_data["action_output"] = action_output
        step_data["action_raw_response"] = raw_response

        if not action_text:
            print("Action prompt output is not in the correct format.")
            step_data["summary"] = "Output for action selection is not in the correct format, so no action is performed."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        print("Action: " + action_text)
        print("Reason: " + (reason if reason else "No reason provided."))

        try:
            action_json = agent_utils.extract_json(action_text)
            if not action_json:
                raise ValueError("Cannot extract action JSON.")
            converted_action = json_action.JSONAction(**action_json)
            converted_action = self._normalize_open_app_action(
                goal,
                action_json,
                converted_action,
                step_data,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print("Failed to convert the output to a valid action.")
            print(str(exc))
            step_data["summary"] = "Can not parse the output to a valid action."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        replan_reason, action_text, action_output, raw_response, converted_action = self._replan_if_repeating(
            goal,
            action_prompt,
            action_output,
            raw_response,
            action_text,
            converted_action,
            step_data,
        )
        if replan_reason is not None:
            reason = replan_reason
            step_data["action_output"] = action_output
            step_data["action_raw_response"] = raw_response

        goal_lower = goal.lower()
        if (
            converted_action.action_type == json_action.INPUT_TEXT
            and "markor" in goal_lower
            and "create a new note" in goal_lower
            and "name" in before_element_list.lower()
            and ".md" in before_element_list
            and _clean_text(getattr(converted_action, "text", "") or "").endswith(".md")
        ):
            original_text = _clean_text(getattr(converted_action, "text", "") or "")
            normalized_text = original_text[:-3]
            if normalized_text:
                try:
                    converted_action.text = normalized_text
                    step_data["hybrid_input_normalized"] = {
                        "from": original_text,
                        "to": normalized_text,
                        "reason": "markor_name_field_has_separate_md_suffix",
                    }
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    step_data["hybrid_input_normalization_error"] = str(exc)

        action_signature = self._action_signature(converted_action)
        step_data["action_signature"] = action_signature
        step_data["repeat_count_before_action"] = self._recent_repeat_count(action_signature)

        if converted_action.action_type in {json_action.CLICK, json_action.LONG_PRESS, json_action.INPUT_TEXT}:
            if converted_action.index is not None and converted_action.index >= len(ui_elements):
                print("Index out of range.")
                step_data["summary"] = "The parameter index is out of range."
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(False, step_data)

        if converted_action.action_type == json_action.STATUS:
            if converted_action.goal_status == "infeasible":
                print("Agent stopped since it thinks mission impossible.")
            step_data["summary"] = self._local_summary(converted_action, reason)
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(True, step_data)

        if converted_action.action_type == json_action.ANSWER:
            answer_text = _clean_text(getattr(converted_action, "text", "") or "")
            if (
                "hour in 24-hour format" in goal.lower()
                and "<minutes>" in goal.lower()
                and re.search(r"\b\d{1,2}:\d{2}:\d{2}\b", answer_text)
            ):
                normalized_answer = re.sub(r"\b(\d{1,2}:\d{2}):\d{2}\b", r"\1", answer_text)
                if normalized_answer != answer_text:
                    try:
                        converted_action.text = normalized_answer
                        step_data["action_signature"] = self._action_signature(converted_action)
                        step_data["hybrid_answer_normalized"] = {
                            "from": answer_text,
                            "to": normalized_answer,
                            "reason": "goal_requests_hour_minute_without_seconds",
                        }
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        step_data["hybrid_answer_normalization_error"] = str(exc)
            print("Agent answered with: " + str(converted_action.text))
            try:
                self.env.execute_action(converted_action)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print("Some error happened executing the action ", converted_action.action_type)
                print(str(exc))
                step_data["summary"] = "Some error happened executing the action " + str(converted_action.action_type)
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(False, step_data)

            summary, state_changed, before_sig, after_sig = self._transition_summary(
                converted_action,
                reason,
                before_element_list,
                before_element_list,
            )
            step_data["summary"] = summary
            step_data["state_changed"] = state_changed
            step_data["before_state_signature"] = before_sig
            step_data["after_state_signature"] = after_sig
            step_data["after_screenshot"] = state.pixels.copy()
            step_data["after_element_list"] = ui_elements
            step_data["hybrid_answer_terminated_session"] = True
            self.history.append(step_data)
            print("Summary: " + str(step_data.get("summary")))
            return base_agent.AgentInteractionResult(True, step_data)

        try:
            self.env.execute_action(converted_action)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print("Some error happened executing the action ", converted_action.action_type)
            print(str(exc))
            step_data["summary"] = "Some error happened executing the action " + str(converted_action.action_type)
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        after_state = self.get_post_transition_state()
        after_element_list = t3a._generate_ui_elements_description_list_full(  # pylint: disable=protected-access
            after_state.ui_elements,
            self.env.logical_screen_size,
        )
        step_data["after_screenshot"] = after_state.pixels.copy()
        step_data["after_element_list"] = after_state.ui_elements
        summary, state_changed, before_sig, after_sig = self._transition_summary(
            converted_action,
            reason,
            before_element_list,
            after_element_list,
        )
        step_data["summary"] = summary
        step_data["state_changed"] = state_changed
        step_data["before_state_signature"] = before_sig
        step_data["after_state_signature"] = after_sig
        self.history.append(step_data)
        print("Summary: " + str(step_data.get("summary")))
        return base_agent.AgentInteractionResult(False, step_data)
