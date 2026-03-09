# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Baseline-like explorer agent built on gelab_agent_resize.

Design goal:
- Keep decision/action path as close as possible to GELABResizeAgent.
- Inject at most one concise hint from previous-step exploration.
- Exploration is real (execute action + rollback), but very occasional and safe.
"""

from __future__ import annotations

import os
import re
import time
from collections import OrderedDict
from typing import Any

from PIL import Image

from android_world.agents import base_agent
from android_world.agents import gelab_agent
from android_world.agents import gelab_agent_resize
from android_world.agents import seeact_utils
from android_world.agents.explorer_agent_utils import _hash_diff
from android_world.agents.explorer_agent_utils import _phash_pixels
from android_world.env import json_action

MAX_EXPLORER_STEPS = 15


def _now_hms() -> str:
    return time.strftime("%H:%M:%S")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _to_user_text(goal: str, history: str, hint: str) -> str:
    if hint:
        return (
            f"Task:\n{goal}\n\n"
            f"History actions:\n{history or 'None yet.'}\n\n"
            f"Exploration hint:\n{hint}\n\n"
            "Current screenshot is attached below.\n"
            "Choose the next single action."
        )
    return (
        f"Task:\n{goal}\n\n"
        f"History actions:\n{history or 'None yet.'}\n\n"
        "Current screenshot is attached below.\n"
        "Choose the next single action."
    )


class ExplorerElementAgent(gelab_agent_resize.GELABResizeAgent):
    """GELAB-resize core with occasional early exploration hints."""

    def set_max_steps(self, max_steps: int) -> None:
        super().set_max_steps(min(MAX_EXPLORER_STEPS, int(max_steps)))

    def _effective_max_steps(self) -> int:
        if self._max_steps is None:
            return MAX_EXPLORER_STEPS
        return min(MAX_EXPLORER_STEPS, int(self._max_steps))

    def __init__(
        self,
        env,
        vllm: Any,
        name: str = "ExplorerElementAgent",
        output_path: str = "",
        history_limit: int = 8,
        image_downsample_scale: float = 2.0,
        enable_light_exploration: bool = True,
        light_explore_max_runs: int = 1,
        light_explore_max_step: int = 2,
        light_explore_launcher_only: bool = True,
        light_explore_require_keyword: bool = True,
        light_explore_require_stall: bool = True,
        light_explore_back_limit: int = 2,
        light_explore_hash_threshold: int = 10,
        light_explore_replay_max_actions: int = 1,
        **kwargs: Any,
    ):
        _ = kwargs
        super().__init__(
            env=env,
            vllm=vllm,
            name=name,
            output_path=output_path,
            history_limit=history_limit,
            image_downsample_scale=image_downsample_scale,
        )
        self.enable_light_exploration = bool(enable_light_exploration)
        self.light_explore_max_runs = max(0, int(light_explore_max_runs))
        self.light_explore_max_step = max(0, int(light_explore_max_step))
        self.light_explore_launcher_only = bool(light_explore_launcher_only)
        self.light_explore_require_keyword = bool(light_explore_require_keyword)
        self.light_explore_require_stall = bool(light_explore_require_stall)
        self.light_explore_back_limit = max(1, int(light_explore_back_limit))
        self.light_explore_hash_threshold = max(1, int(light_explore_hash_threshold))
        self.light_explore_replay_max_actions = max(0, int(light_explore_replay_max_actions))

        self._pending_explore_hint: str = ""
        self._light_explore_runs: int = 0
        self._last_probe_candidates: list[dict[str, Any]] = []

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home=go_home)
        self._pending_explore_hint = ""
        self._light_explore_runs = 0
        self._last_probe_candidates = []

    @staticmethod
    def _normalize_activity_name(activity: str | None) -> str:
        return _clean_text(activity).lower()

    def _foreground_activity_name(self) -> str:
        try:
            return str(self.env.foreground_activity_name or "").strip()
        except Exception:  # pylint: disable=broad-exception-caught
            return ""

    @staticmethod
    def _is_launcher_activity(activity: str | None) -> bool:
        value = _clean_text(activity).lower()
        if not value:
            return False
        return bool("launcher" in value or "nexuslauncher" in value or "quickstep" in value)

    def _goal_keywords(self, goal: str) -> list[str]:
        text = _clean_text(goal).lower()
        out: list[str] = []
        seen: set[str] = set()
        for app in gelab_agent.AVAILABLE_APPS:
            app_low = _clean_text(app).lower()
            if app_low and app_low in text and app_low not in seen:
                seen.add(app_low)
                out.append(app_low)
            for token in re.findall(r"[a-z0-9]{4,}", app_low):
                if token in text and token not in seen:
                    seen.add(token)
                    out.append(token)
        for token in re.findall(r"[a-z0-9]{5,}", text):
            if token not in seen:
                seen.add(token)
                out.append(token)
            if len(out) >= 16:
                break
        return out

    def _goal_app_keywords(self, goal: str) -> list[str]:
        text = _clean_text(goal).lower()
        out: list[str] = []
        seen: set[str] = set()
        for app in gelab_agent.AVAILABLE_APPS:
            app_low = _clean_text(app).lower()
            if app_low and app_low in text and app_low not in seen:
                seen.add(app_low)
                out.append(app_low)
            for token in re.findall(r"[a-z0-9]{4,}", app_low):
                if token in text and token not in seen:
                    seen.add(token)
                    out.append(token)
        return out

    def _element_text(self, element: Any) -> str:
        text = _clean_text(getattr(element, "text", ""))
        desc = _clean_text(getattr(element, "content_description", ""))
        element_id = _clean_text(getattr(element, "resource_id", ""))
        merged = " ".join(x for x in [text, desc, element_id] if x)
        return _clean_text(merged).lower()

    @staticmethod
    def _safe_center_from_element(element: Any) -> tuple[int, int] | None:
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            return None
        try:
            return int((bbox.x_min + bbox.x_max) / 2.0), int((bbox.y_min + bbox.y_max) / 2.0)
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    @staticmethod
    def _is_interactive(element: Any) -> bool:
        return bool(
            getattr(element, "is_clickable", False)
            or getattr(element, "is_long_clickable", False)
            or getattr(element, "is_editable", False)
        )

    @staticmethod
    def _state_hash(state: Any) -> int:
        try:
            return int(_phash_pixels(state.pixels))
        except Exception:  # pylint: disable=broad-exception-caught
            return -1

    def _same_root_page(
        self,
        curr_state: Any,
        root_activity: str,
        root_hash: int,
    ) -> tuple[bool, str]:
        curr_activity = self._normalize_activity_name(self._foreground_activity_name())
        root_activity_norm = self._normalize_activity_name(root_activity)
        curr_hash = self._state_hash(curr_state)
        if root_hash < 0 or curr_hash < 0:
            same_activity = bool(curr_activity and root_activity_norm and curr_activity == root_activity_norm)
            return same_activity, "activity_match_only"
        try:
            diff = int(_hash_diff(root_hash, curr_hash))
        except Exception:  # pylint: disable=broad-exception-caught
            diff = 10**9
        same_activity = bool(curr_activity and root_activity_norm and curr_activity == root_activity_norm)
        if same_activity and diff <= int(self.light_explore_hash_threshold):
            return True, f"activity+phash<={self.light_explore_hash_threshold}:{diff}"
        if same_activity and diff <= int(self.light_explore_hash_threshold + 4):
            return True, f"activity+phash<={self.light_explore_hash_threshold + 4}:{diff}"
        return False, f"activity_match={same_activity}|phash_diff={diff}"

    def _collect_probe_candidates(self, state: Any, goal: str) -> list[dict[str, Any]]:
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        if not ui_elements:
            return []
        keywords = self._goal_app_keywords(goal)
        if self.light_explore_require_keyword and not keywords:
            return []

        candidates: list[dict[str, Any]] = []
        for idx, element in enumerate(ui_elements):
            if not self._is_interactive(element):
                continue
            center = self._safe_center_from_element(element)
            if center is None:
                continue
            merged = self._element_text(element)
            if not merged:
                continue
            if "inputmethod" in merged or "systemui" in merged:
                continue

            score = 0.0
            if keywords:
                for kw in keywords:
                    if kw and kw in merged:
                        score += 4.0 + min(len(kw), 12) * 0.1
            else:
                score = 1.0
            if self.light_explore_require_keyword and score <= 0.0:
                continue

            label = _clean_text(getattr(element, "text", "")) or _clean_text(
                getattr(element, "content_description", "")
            )
            if not label:
                label = keywords[0] if keywords else "candidate"

            candidates.append(
                {
                    "index": idx,
                    "element": element,
                    "center": center,
                    "label": label,
                    "score": float(score),
                    "merged": merged,
                }
            )

        candidates.sort(key=lambda item: (float(item.get("score", 0.0)), -int(item.get("index", 0))), reverse=True)
        return candidates[:8]

    def _choose_probe_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        return candidates[0]

    @staticmethod
    def _is_replay_safe_action(action: json_action.JSONAction | None) -> bool:
        if action is None:
            return False
        return action.action_type in {
            json_action.CLICK,
            json_action.LONG_PRESS,
            json_action.SWIPE,
            json_action.NAVIGATE_BACK,
            json_action.NAVIGATE_HOME,
            json_action.OPEN_APP,
        }

    def _json_action_from_record(self, record: dict[str, Any] | None) -> json_action.JSONAction | None:
        if not isinstance(record, dict):
            return None
        fields = {
            "action_type": record.get("action_type"),
            "index": record.get("index"),
            "x": record.get("x"),
            "y": record.get("y"),
            "text": record.get("text"),
            "direction": record.get("direction"),
            "goal_status": record.get("goal_status"),
            "app_name": record.get("app_name"),
            "keycode": record.get("keycode"),
            "clear_text": record.get("clear_text"),
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        if not fields.get("action_type"):
            return None
        try:
            action = json_action.JSONAction(**fields)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        if not self._is_replay_safe_action(action):
            return None
        return action

    def _select_replay_actions_for_probe(
        self,
        current_action: json_action.JSONAction | None,
    ) -> list[json_action.JSONAction]:
        actions: list[json_action.JSONAction] = []
        if self._is_replay_safe_action(current_action):
            actions.append(current_action)
        if self.light_explore_replay_max_actions <= 0:
            return actions
        for record in reversed(list(self._actions)):
            action = self._json_action_from_record(record.get("action_dict"))
            if action is None:
                continue
            actions.append(action)
            if len(actions) >= max(1, int(self.light_explore_replay_max_actions)):
                break
        actions.reverse()
        return actions

    def _rollback_to_probe_root(
        self,
        root_activity: str,
        root_hash: int,
        replay_actions: list[json_action.JSONAction],
        step_idx: int,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "success": False,
            "mode": "backtrack_failed",
            "back_presses": 0,
            "replayed_actions": 0,
            "matched_by": None,
        }

        back_limit = max(1, int(self.light_explore_back_limit))
        for i in range(back_limit + 1):
            curr_state = self.env.get_state(wait_to_stabilize=True)
            same, matched_by = self._same_root_page(curr_state, root_activity, root_hash)
            if same:
                result["success"] = True
                result["mode"] = "backtrack"
                result["matched_by"] = matched_by
                print(
                    f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
                    f"rollback_done back_presses={result['back_presses']} matched_by={matched_by}"
                )
                return result
            if i >= back_limit:
                break
            print(
                f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
                f"rollback_back#{i + 1}/{back_limit} matched={matched_by}"
            )
            self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK))
            result["back_presses"] += 1

        if replay_actions:
            result["mode"] = "replay"
            print(f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} rollback_fallback=replay")
            for action in replay_actions:
                try:
                    self.env.execute_action(action)
                    result["replayed_actions"] += 1
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    print(
                        f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
                        f"rollback_replay_action_failed action={action.action_type} error={exc}"
                    )
            final_state = self.env.get_state(wait_to_stabilize=True)
            same, matched_by = self._same_root_page(final_state, root_activity, root_hash)
            result["success"] = bool(same)
            result["matched_by"] = matched_by
            result["mode"] = "replay" if same else "replay_failed"
            print(
                f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
                f"rollback_replay_done success={result['success']} replayed={result['replayed_actions']} "
                f"matched_by={matched_by}"
            )
            return result

        print(
            f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
            f"rollback_failed back_presses={result['back_presses']}"
        )
        return result

    def _probe_page_changed(
        self,
        root_activity: str,
        root_hash: int,
        after_state: Any,
    ) -> tuple[bool, str, int | None]:
        after_activity = self._foreground_activity_name()
        root_norm = self._normalize_activity_name(root_activity)
        after_norm = self._normalize_activity_name(after_activity)
        changed = bool(root_norm and after_norm and root_norm != after_norm)
        diff: int | None = None
        after_hash = self._state_hash(after_state)
        if root_hash >= 0 and after_hash >= 0:
            try:
                diff = int(_hash_diff(root_hash, after_hash))
            except Exception:  # pylint: disable=broad-exception-caught
                diff = None
        if diff is not None and diff >= 6:
            changed = True
        return changed, after_activity, diff

    def _build_hint_from_observation(
        self,
        candidate: dict[str, Any],
        changed: bool,
        after_activity: str,
    ) -> str:
        if not changed:
            return ""
        label = _clean_text(candidate.get("label")) or "that element"
        after_short = _clean_text(after_activity).split("/")[-1]
        if after_short:
            return f'Quick exploration: tapping "{label}" opened {after_short}; consider this action first.'
        return f'Quick exploration: tapping "{label}" opened a different page; consider this action first.'

    def _is_page_stalled(self, current_activity: str, current_hash: int) -> bool:
        if not self._actions:
            return False
        prev = self._actions[-1]
        prev_activity = self._normalize_activity_name(str(prev.get("start_page_activity", "")))
        curr_activity = self._normalize_activity_name(current_activity)
        try:
            prev_hash = int(prev.get("start_page_hash"))
        except Exception:  # pylint: disable=broad-exception-caught
            prev_hash = -1
        if prev_hash < 0 or current_hash < 0:
            return bool(prev_activity and curr_activity and prev_activity == curr_activity)
        try:
            diff = int(_hash_diff(prev_hash, current_hash))
        except Exception:  # pylint: disable=broad-exception-caught
            diff = 10**9
        return bool(prev_activity and curr_activity and prev_activity == curr_activity and diff <= 2)

    def _should_run_light_exploration(
        self,
        step_idx: int,
        root_activity: str,
        page_stalled: bool,
    ) -> tuple[bool, str]:
        if not self.enable_light_exploration:
            return False, "disabled"
        if self.light_explore_max_runs <= 0:
            return False, "run_budget_zero"
        if self._light_explore_runs >= self.light_explore_max_runs:
            return False, "run_budget_exhausted"
        if step_idx >= self.light_explore_max_step:
            return False, "beyond_step_window"
        if self.light_explore_require_stall and not page_stalled:
            return False, "page_not_stalled"
        if self.light_explore_launcher_only and not self._is_launcher_activity(root_activity):
            return False, "not_launcher_activity"
        return True, "ok"

    def _run_light_exploration(
        self,
        goal: str,
        step_idx: int,
        current_action: json_action.JSONAction | None = None,
        page_stalled: bool = False,
    ) -> str:
        self._last_probe_candidates = []
        try:
            root_state = self.env.get_state(wait_to_stabilize=True)
            root_hash = self._state_hash(root_state)
            root_activity = self._foreground_activity_name()
            should_run, reason = self._should_run_light_exploration(
                step_idx=step_idx,
                root_activity=root_activity,
                page_stalled=page_stalled,
            )
            print(
                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                f"light_explore_should_run={should_run} reason={reason} page_stalled={page_stalled}"
            )
            if not should_run:
                return ""

            candidates = self._collect_probe_candidates(root_state, goal)
            self._last_probe_candidates = list(candidates)
            if not candidates:
                print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_no_candidates")
                self._light_explore_runs += 1
                return ""

            candidate = self._choose_probe_candidate(candidates)
            if not candidate:
                print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_candidate_none")
                self._light_explore_runs += 1
                return ""

            center = candidate.get("center")
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_center_invalid")
                self._light_explore_runs += 1
                return ""

            probe_action = json_action.JSONAction(
                action_type=json_action.CLICK,
                x=int(center[0]),
                y=int(center[1]),
            )
            label = _clean_text(candidate.get("label")) or "candidate"
            print(
                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                f"light_explore_probe_action=click label={label} center={[int(center[0]), int(center[1])]}"
            )
            self.env.execute_action(probe_action)
            after_state = self.env.get_state(wait_to_stabilize=True)
            changed, after_activity, hash_diff = self._probe_page_changed(root_activity, root_hash, after_state)
            print(
                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                f"light_explore_probe_effect changed={changed} hash_diff={hash_diff} after_activity={after_activity}"
            )

            replay_actions = self._select_replay_actions_for_probe(current_action=current_action)
            rollback_info = self._rollback_to_probe_root(
                root_activity=root_activity,
                root_hash=root_hash,
                replay_actions=replay_actions,
                step_idx=step_idx,
            )
            self._light_explore_runs += 1
            if not bool(rollback_info.get("success")):
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    f"light_explore_restore_failed={rollback_info}"
                )
                self.enable_light_exploration = False
                return ""

            hint = self._build_hint_from_observation(
                candidate=candidate,
                changed=changed,
                after_activity=after_activity,
            )
            if hint:
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    f"light_explore_hint_generated: {hint}"
                )
            else:
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    "light_explore_no_useful_hint"
                )
            return hint
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(
                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                f"light_explore_failed: {exc}"
            )
            self._light_explore_runs += 1
            return ""

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        start_time = time.time()
        step_idx = len(self._actions)
        if step_idx >= self._effective_max_steps():
            summary = f"Reached the maximum step limit ({self._effective_max_steps()})."
            action = json_action.JSONAction(action_type=json_action.STATUS, goal_status="infeasible")
            tool_call = {"name": "mobile_use", "arguments": {"action": "terminate", "status": "fail"}}
            print("=" * 96)
            print(f"Step {step_idx}: Result")
            print(summary)
            print("=" * 96)
            return base_agent.AgentInteractionResult(
                done=True,
                data={
                    "response": "",
                    "parsed_action": {
                        "action": "ABORT",
                        "summary": summary,
                        "value": summary,
                    },
                    "tool_call": tool_call,
                    "action": repr(action),
                    "action_dict": action.__dict__,
                    "summary": summary,
                    "hints": [],
                    "latency_sec": float(max(0.0, time.time() - start_time)),
                },
            )

        print("=" * 96)
        print(f"Step {step_idx}: Goal")
        print(goal)

        state = self.get_post_transition_state()
        screenshot = Image.fromarray(state.pixels)
        model_screenshot, original_size, resized_size = self._build_model_screenshot(screenshot)
        screen_size = self.env.logical_screen_size
        history = self._history_text()
        start_page_activity = self._foreground_activity_name()
        start_page_hash = self._state_hash(state)
        page_stalled = self._is_page_stalled(
            current_activity=start_page_activity,
            current_hash=start_page_hash,
        )

        hint_for_prompt = _clean_text(self._pending_explore_hint)
        self._pending_explore_hint = ""
        prompt_mode = "baseline_plus_hint" if hint_for_prompt else "baseline"
        print(
            f"[EXPLORE {_now_hms()}] step: {step_idx + 1} prompt_mode: {prompt_mode}; "
            f"hint: {hint_for_prompt or '<none>'}"
        )

        gelab_agent._print_step_section(  # pylint: disable=protected-access
            step_idx,
            "Resolution",
            (
                f"original={original_size[0]}x{original_size[1]}, "
                f"model={resized_size[0]}x{resized_size[1]}, "
                f"image_downsample_scale={self.image_downsample_scale:.3f}"
            ),
        )

        user_text = _to_user_text(goal, history, hint_for_prompt)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": gelab_agent.GELAB_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": gelab_agent._image_to_data_url(model_screenshot)},  # pylint: disable=protected-access
                    },
                ],
            },
        ]
        message_text = gelab_agent._messages_text_for_logging(messages)  # pylint: disable=protected-access
        if message_text:
            gelab_agent._print_step_section(step_idx, "Model input", message_text)  # pylint: disable=protected-access

        response, _, _ = self.vllm.predict_mm("", [], messages=messages)
        gelab_agent._print_step_section(step_idx, "Model output", str(response))  # pylint: disable=protected-access

        parse_error = None
        try:
            parsed_action = gelab_agent.parse_gelab_response(response)
            action, tool_call, extras = gelab_agent.gelab_action_to_json_action(parsed_action, screen_size)
        except seeact_utils.ParseActionError as error:
            parse_error = str(error)
            parsed_action = OrderedDict(
                cot="",
                action="WAIT",
                value="1",
                summary="Parser fallback wait",
                parse_error=parse_error,
            )
            action = json_action.JSONAction(action_type=json_action.WAIT)
            tool_call = {"name": "mobile_use", "arguments": {"action": "wait", "value": 1}}
            extras = {"wait_seconds": 1, "parse_error": parse_error, "fallback": "parse_error_wait"}

        if parse_error:
            gelab_agent._print_step_section(step_idx, "Parse fallback", parse_error)  # pylint: disable=protected-access
        gelab_agent._print_step_section(step_idx, "Parsed action", gelab_agent._json_dumps_safe(dict(parsed_action)))  # pylint: disable=protected-access
        gelab_agent._print_step_section(step_idx, "Tool call", gelab_agent._json_dumps_safe(tool_call))  # pylint: disable=protected-access

        if extras.get("return_text"):
            self.env.interaction_cache = str(extras["return_text"])

        self._execute_action(action, extras)
        gelab_agent._print_step_section(step_idx, "Action", gelab_agent._json_dumps_safe(action.__dict__))  # pylint: disable=protected-access
        if extras:
            gelab_agent._print_step_section(step_idx, "Action extras", gelab_agent._json_dumps_safe(extras))  # pylint: disable=protected-access

        summary = gelab_agent._normalize_space(parsed_action.get("summary")) or gelab_agent._normalize_space(parsed_action.get("explain"))  # pylint: disable=protected-access
        if not summary:
            summary = str(tool_call.get("arguments") or tool_call)

        # Exploration runs after current-step action; hint is consumed at next step.
        next_hint = self._run_light_exploration(
            goal=goal,
            step_idx=step_idx,
            current_action=action,
            page_stalled=page_stalled,
        )
        self._pending_explore_hint = _clean_text(next_hint)

        latency_sec = float(max(0.0, time.time() - start_time))
        step_record = {
            "goal": goal,
            "response": response,
            "parsed_action": dict(parsed_action),
            "tool_call": tool_call,
            "action_dict": action.__dict__,
            "summary": summary,
            "latency_sec": latency_sec,
            "model_input_resolution": {"width": resized_size[0], "height": resized_size[1]},
            "original_resolution": {"width": original_size[0], "height": original_size[1]},
            "image_downsample_scale": self.image_downsample_scale,
            "prompt_mode": prompt_mode,
            "prompt_hint": hint_for_prompt,
            "next_step_hint": self._pending_explore_hint,
            "light_explore_runs": self._light_explore_runs,
            "start_page_activity": start_page_activity,
            "start_page_hash": start_page_hash,
            "page_stalled": page_stalled,
        }
        self._actions.append(step_record)
        self._summaries.append(summary)
        self._responses.append(str(response))

        task_dir = self._task_output_dir(goal)
        if task_dir:
            os.makedirs(task_dir, exist_ok=True)
            screenshot.save(os.path.join(task_dir, f"screenshot_{len(self._actions) - 1}.png"))
            model_screenshot.save(os.path.join(task_dir, f"screenshot_model_input_{len(self._actions) - 1}.png"))
            self._write_action_log(goal)

        done = action.action_type in {json_action.STATUS, json_action.ANSWER}
        print(f"Step {step_idx}: Latency")
        print(f"{latency_sec:.3f}s")
        print(f"Step {step_idx}: Result")
        print(f"done={done}, action_type={action.action_type}, summary={summary}")
        print("=" * 96)

        return base_agent.AgentInteractionResult(
            done=done,
            data={
                "response": response,
                "parsed_action": dict(parsed_action),
                "tool_call": tool_call,
                "action": repr(action),
                "action_dict": action.__dict__,
                "summary": summary,
                "hints": [],
                "latency_sec": latency_sec,
                "model_input_resolution": {"width": resized_size[0], "height": resized_size[1]},
                "original_resolution": {"width": original_size[0], "height": original_size[1]},
                "image_downsample_scale": self.image_downsample_scale,
                "prompt_mode": prompt_mode,
                "prompt_hint": hint_for_prompt,
                "next_step_hint": self._pending_explore_hint,
                "light_explore_runs": self._light_explore_runs,
            },
        )


class ElementTextAgent(ExplorerElementAgent):
    pass
