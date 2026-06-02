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

"""Baseline-like explorer agent with effectiveness instrumentation.

Design goal:
- Keep decision/action path as close as possible to GELABResizeAgent.
- Inject at most one concise hint from previous-step exploration.
- Exploration is real (execute action + rollback), but very occasional and safe.
- Record structured effectiveness statistics for post-hoc analysis.
"""

from __future__ import annotations

import os
import re
import time
import json
from collections import OrderedDict
from typing import Any

from PIL import Image

from android_world.agents import base_agent
from android_world.agents import gelab_agent
from android_world.agents import gelab_agent_resize
from android_world.agents import seeact_utils
from android_world.agents.explorer_agent_utils import _hash_diff
from android_world.agents.explorer_agent_utils import _phash_pixels
from android_world.agents.explorer_agent_utils import _to_json_action
from android_world.agents.explorer_agent_utils import parse_tool_call
from android_world.env import json_action

MAX_EXPLORER_STEPS = 15
FIXED_SCREENSHOT_SCALE = 2.0
PAGE_CHANGED_HASH_DIFF = 6
NO_EFFECT_HASH_DIFF = 2


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
        light_explore_max_runs: int = 3,
        light_explore_max_step: int = 6,
        light_explore_launcher_only: bool = False,
        light_explore_require_keyword: bool = False,
        light_explore_require_stall: bool = False,
        light_explore_branch_budget: int = 3,
        light_explore_branch_depth: int = 2,
        light_explore_back_limit: int = 3,
        light_explore_hash_threshold: int = 12,
        light_explore_high_complexity_clickable_threshold: int = 14,
        light_explore_relaxed_period: int = 2,
        light_explore_disable_after_rollback_failures: int = 2,
        **kwargs: Any,
    ):
        _ = kwargs, image_downsample_scale
        effective_output_path = str(output_path or "./output/effectiveness_traces").strip()
        super().__init__(
            env=env,
            vllm=vllm,
            name=name,
            output_path=effective_output_path,
            history_limit=history_limit,
            image_downsample_scale=FIXED_SCREENSHOT_SCALE,
        )
        # Fix screenshot scale at 2 for effectiveness analysis comparability.
        self.fixed_image_downsample_scale = float(FIXED_SCREENSHOT_SCALE)
        self.image_downsample_scale = float(FIXED_SCREENSHOT_SCALE)
        self._enable_light_exploration_default = bool(enable_light_exploration)
        self.enable_light_exploration = bool(enable_light_exploration)
        self.light_explore_max_runs = max(0, int(light_explore_max_runs))
        self.light_explore_max_step = max(0, int(light_explore_max_step))
        self.light_explore_launcher_only = bool(light_explore_launcher_only)
        self.light_explore_require_keyword = bool(light_explore_require_keyword)
        self.light_explore_require_stall = bool(light_explore_require_stall)
        self.light_explore_branch_budget = max(1, int(light_explore_branch_budget))
        self.light_explore_branch_depth = max(1, int(light_explore_branch_depth))
        self.light_explore_back_limit = max(1, int(light_explore_back_limit))
        self.light_explore_hash_threshold = max(1, int(light_explore_hash_threshold))
        self.light_explore_high_complexity_clickable_threshold = max(
            1, int(light_explore_high_complexity_clickable_threshold)
        )
        self.light_explore_relaxed_period = max(1, int(light_explore_relaxed_period))
        self.light_explore_disable_after_rollback_failures = max(
            1, int(light_explore_disable_after_rollback_failures)
        )

        self._pending_explore_hint: str = ""
        self._pending_explore_hint_payload: dict[str, Any] = {}
        self._light_explore_runs: int = 0
        self._last_probe_candidates: list[dict[str, Any]] = []
        # Effectiveness-analysis fields are persisted into step traces.
        self._last_explore_stats: dict[str, Any] = self._default_explore_stats()
        self._last_rollback_stats: dict[str, Any] = self._default_rollback_stats()
        self._last_generated_hint_stats: dict[str, Any] = self._default_hint_stats()
        self._repeated_no_effect_count: int = 0
        self._episode_goal: str = ""
        self._episode_task_id: str = ""
        self._episode_task_name: str = ""
        self._episode_started_at: float = 0.0
        self._episode_trace_finalized: bool = False
        self._rollback_failure_streak: int = 0

    def reset(self, go_home: bool = False) -> None:
        if self._episode_goal and self._actions and not self._episode_trace_finalized:
            self._write_effectiveness_trace(goal=self._episode_goal, episode_success=False, finalize=True)
        super().reset(go_home=go_home)
        self.enable_light_exploration = bool(self._enable_light_exploration_default)
        self._pending_explore_hint = ""
        self._pending_explore_hint_payload = {}
        self._light_explore_runs = 0
        self._last_probe_candidates = []
        self._last_explore_stats = self._default_explore_stats()
        self._last_rollback_stats = self._default_rollback_stats()
        self._last_generated_hint_stats = self._default_hint_stats()
        self._repeated_no_effect_count = 0
        self._episode_goal = ""
        self._episode_task_id = ""
        self._episode_task_name = ""
        self._episode_started_at = 0.0
        self._episode_trace_finalized = False
        self._rollback_failure_streak = 0

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

    def _goal_tokens(self, goal: str) -> list[str]:
        text = _clean_text(goal).lower()
        out: list[str] = []
        seen: set[str] = set()
        for token in re.findall(r"[a-z0-9]{3,}", text):
            if token not in seen:
                seen.add(token)
                out.append(token)
            if len(out) >= 32:
                break
        return out

    @staticmethod
    def _normalize_resource_id(resource_id: str) -> str:
        rid = _clean_text(resource_id)
        if not rid:
            return ""
        # Drop package prefix to avoid app-name leakage like "net.gsantner.markor:*".
        rid = rid.split("/")[-1]
        rid = rid.split(":")[-1]
        rid = rid.replace("_", " ")
        rid = re.sub(r"[^a-zA-Z0-9 ]+", " ", rid)
        return _clean_text(rid)

    @staticmethod
    def _text_similarity_score(merged: str, goal_tokens: list[str], app_keywords: list[str]) -> float:
        merged_low = _clean_text(merged).lower()
        if not merged_low:
            return 0.0
        score = 0.0
        goal_token_set = set(goal_tokens)
        for kw in app_keywords:
            if kw and kw in merged_low:
                score += 1.5 + min(len(kw), 12) * 0.06
        merged_tokens = set(re.findall(r"[a-z0-9]{3,}", merged_low))
        if merged_tokens and goal_token_set:
            overlap = merged_tokens.intersection(goal_token_set)
            if overlap:
                score += float(len(overlap)) * 1.2
                score += float(len(overlap)) / max(1.0, float(len(merged_tokens)))
        return float(score)

    def _element_text(self, element: Any) -> str:
        text = _clean_text(getattr(element, "text", ""))
        desc = _clean_text(getattr(element, "content_description", ""))
        element_id = self._normalize_resource_id(str(getattr(element, "resource_id", "")))
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

    @staticmethod
    def _safe_hash_diff(lhs_hash: int, rhs_hash: int) -> int | None:
        if lhs_hash < 0 or rhs_hash < 0:
            return None
        try:
            return int(_hash_diff(lhs_hash, rhs_hash))
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    @staticmethod
    def _bbox_dict_from_element(element: Any) -> dict[str, int] | None:
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            return None
        try:
            return {
                "x_min": int(bbox.x_min),
                "y_min": int(bbox.y_min),
                "x_max": int(bbox.x_max),
                "y_max": int(bbox.y_max),
            }
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    @staticmethod
    def _point_in_bbox(point_xy: list[int] | tuple[int, int] | None, bbox: dict[str, Any] | None) -> bool | None:
        if point_xy is None or bbox is None:
            return None
        if not isinstance(point_xy, (list, tuple)) or len(point_xy) < 2:
            return None
        try:
            x = int(point_xy[0])
            y = int(point_xy[1])
            x_min = int(bbox.get("x_min"))
            y_min = int(bbox.get("y_min"))
            x_max = int(bbox.get("x_max"))
            y_max = int(bbox.get("y_max"))
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        return bool(x_min <= x <= x_max and y_min <= y <= y_max)

    @staticmethod
    def _count_page_complexity(state: Any) -> dict[str, int]:
        total = 0
        clickable = 0
        editable = 0
        for element in list(getattr(state, "ui_elements", None) or []):
            total += 1
            if bool(getattr(element, "is_clickable", False) or getattr(element, "is_long_clickable", False)):
                clickable += 1
            if bool(getattr(element, "is_editable", False)):
                editable += 1
        return {
            "page_total_element_count": int(total),
            "page_clickable_element_count": int(clickable),
            "page_editable_element_count": int(editable),
        }

    @staticmethod
    def _default_explore_stats() -> dict[str, Any]:
        return {
            "exploration_triggered": False,
            "exploration_trigger_reason": "not_called",
            "exploration_budget_remaining": 0,
            "exploration_candidate_count": 0,
            "exploration_selected_target_count": 0,
            "exploration_branch_count": 0,
            "exploration_best_branch_depth": 0,
        }

    @staticmethod
    def _default_rollback_stats() -> dict[str, Any]:
        return {
            "rollback_triggered": False,
            "rollback_level": "none",
            "rollback_success": False,
            "rollback_latency_ms": 0.0,
            "rollback_back_presses": 0,
            "rollback_replayed_actions": 0,
            "rollback_matched_by": None,
        }

    @staticmethod
    def _default_hint_stats() -> dict[str, Any]:
        return {
            "hint_generated": False,
            "hint_text": "",
            "hint_source_depth": None,
            "hint_bbox": None,
            "hint_target_element_text": "",
            "hint_target_element_id": "",
            "hint_injected_into_prompt": False,
        }

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): ExplorerElementAgent._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ExplorerElementAgent._json_safe(v) for v in value]
        return value

    def _ensure_episode_identity(self, goal: str) -> None:
        if self._episode_goal == goal and self._episode_task_id:
            return
        if self._episode_goal and self._actions and not self._episode_trace_finalized:
            self._write_effectiveness_trace(goal=self._episode_goal, episode_success=False, finalize=True)
        task_name = str(gelab_agent._safe_task_name(goal))  # pylint: disable=protected-access
        now_ms = int(time.time() * 1000)
        self._episode_goal = str(goal)
        self._episode_task_name = task_name
        self._episode_task_id = f"{task_name}_{now_ms}"
        self._episode_started_at = time.time()
        self._episode_trace_finalized = False

    @staticmethod
    def _infer_episode_success(action: json_action.JSONAction) -> bool:
        if action.action_type == json_action.ANSWER:
            return True
        if action.action_type != json_action.STATUS:
            return False
        status = _clean_text(getattr(action, "goal_status", "")).lower()
        return status in {
            "success",
            "succeeded",
            "done",
            "complete",
            "completed",
            "task_complete",
            "task-complete",
        }

    def _build_task_aggregation(self, goal: str, episode_success: bool | None) -> dict[str, Any]:
        total_rollbacks = 0
        total_level1_rollbacks = 0
        total_level2_rollbacks = 0
        hint_follow_match = 0
        hint_follow_total = 0
        for step in self._actions:
            if bool(step.get("rollback_triggered")):
                total_rollbacks += 1
                level = str(step.get("rollback_level") or "none")
                if level == "level1":
                    total_level1_rollbacks += 1
                elif level == "level2":
                    total_level2_rollbacks += 1
            match = step.get("final_action_bbox_match_hint")
            if match is True:
                hint_follow_match += 1
                hint_follow_total += 1
            elif match is False:
                hint_follow_total += 1
        hint_follow_rate = (
            float(hint_follow_match) / float(hint_follow_total)
            if hint_follow_total > 0
            else None
        )
        return {
            "task_id": self._episode_task_id,
            "task_name": self._episode_task_name or str(gelab_agent._safe_task_name(goal)),  # pylint: disable=protected-access
            "task_goal": str(goal),
            "episode_success": episode_success,
            "episode_length": len(self._actions),
            "total_rollbacks": int(total_rollbacks),
            "total_level1_rollbacks": int(total_level1_rollbacks),
            "total_level2_rollbacks": int(total_level2_rollbacks),
            "hint_follow_rate": hint_follow_rate,
            "hint_follow_match_count": int(hint_follow_match),
            "hint_follow_total_count": int(hint_follow_total),
            "episode_started_at": float(self._episode_started_at) if self._episode_started_at else None,
            "episode_duration_sec": (
                float(max(0.0, time.time() - float(self._episode_started_at)))
                if self._episode_started_at
                else None
            ),
        }

    def _write_effectiveness_trace(
        self,
        goal: str,
        episode_success: bool | None,
        finalize: bool = False,
    ) -> None:
        task_dir = self._task_output_dir(goal)
        if not task_dir:
            return
        payload = {
            "task": self._build_task_aggregation(goal=goal, episode_success=episode_success),
            # Keep per-step records in one JSON for effectiveness analysis post-processing.
            "steps": self._json_safe(self._actions),
        }
        out_path = os.path.join(task_dir, "effectiveness_trace.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[TRACE {_now_hms()}] write_effectiveness_trace_failed: {exc}")
        if finalize:
            self._episode_trace_finalized = True

    def _current_exploration_budget_remaining(self) -> int:
        return max(0, int(self.light_explore_max_runs) - int(self._light_explore_runs))

    @staticmethod
    def _action_coordinate_if_click(action: json_action.JSONAction) -> list[int] | None:
        if action.action_type != json_action.CLICK:
            return None
        try:
            return [int(action.x), int(action.y)]
        except Exception:  # pylint: disable=broad-exception-caught
            return None

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
        if same_activity and diff <= int(self.light_explore_hash_threshold + 6):
            return True, f"activity+phash<={self.light_explore_hash_threshold + 6}:{diff}"
        return False, f"activity_match={same_activity}|phash_diff={diff}"

    def _collect_probe_candidates(self, state: Any, goal: str) -> list[dict[str, Any]]:
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        if not ui_elements:
            return []
        app_keywords = self._goal_app_keywords(goal)
        goal_tokens = self._goal_tokens(goal)

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

            score = self._text_similarity_score(
                merged=merged,
                goal_tokens=goal_tokens,
                app_keywords=app_keywords,
            )
            if not app_keywords and not goal_tokens:
                score = 1.0
            if self.light_explore_require_keyword and app_keywords and score <= 0.0:
                continue
            # Keep a tiny fallback score to guarantee one safe explore attempt.
            if score <= 0.0:
                score = 0.01

            label = _clean_text(getattr(element, "text", "")) or _clean_text(
                getattr(element, "content_description", "")
            )
            if not label:
                label = app_keywords[0] if app_keywords else "candidate"

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
        keep_n = max(3, int(self.light_explore_branch_budget) * 2)
        return candidates[:keep_n]

    def _choose_probe_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        return candidates[0]

    def _choose_secondary_probe_candidate(
        self,
        candidates: list[dict[str, Any]],
        first_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not candidates:
            return None
        first_center = None
        if isinstance(first_candidate, dict):
            first_center = first_candidate.get("center")
        for cand in candidates:
            if first_center is not None and cand.get("center") == first_center:
                continue
            return cand
        return candidates[0]

    def _hint_payload_from_candidate(
        self,
        candidate: dict[str, Any] | None,
        hint_text: str,
        hint_source_depth: int | None,
    ) -> dict[str, Any]:
        element = (candidate or {}).get("element") if isinstance(candidate, dict) else None
        bbox = self._bbox_dict_from_element(element) if element is not None else None
        target_text = _clean_text((candidate or {}).get("label")) if isinstance(candidate, dict) else ""
        target_id = ""
        if element is not None:
            target_id = _clean_text(getattr(element, "resource_id", ""))
        return {
            "hint_generated": bool(_clean_text(hint_text)),
            "hint_text": _clean_text(hint_text),
            "hint_source_depth": hint_source_depth,
            "hint_bbox": bbox,
            "hint_target_element_text": target_text,
            "hint_target_element_id": target_id,
            "hint_injected_into_prompt": False,
        }

    @staticmethod
    def _is_replay_safe_action(action: json_action.JSONAction | None) -> bool:
        if action is None:
            return False
        if action.action_type == json_action.OPEN_APP:
            return bool(_clean_text(getattr(action, "app_name", "")))
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

    def _infer_open_app_name_from_goal(self, goal: str) -> str:
        goal_low = _clean_text(goal).lower()
        if not goal_low:
            return ""
        for app in gelab_agent.AVAILABLE_APPS:
            app_name = _clean_text(app)
            if app_name and app_name.lower() in goal_low:
                return app_name
        return ""

    def _recover_action_from_tool_call(
        self,
        response: str,
        state: Any,
        screen_size: tuple[int, int],
        goal: str,
    ) -> tuple[json_action.JSONAction, dict[str, Any], dict[str, Any], OrderedDict[str, Any]]:
        recovered_tool_call = parse_tool_call(str(response))
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        recovered_action = _to_json_action(
            recovered_tool_call,
            ui_elements,
            fallback_index=None,
            logical_screen_size=screen_size,
            coordinate_mode="1000",
        )
        if recovered_action.action_type == json_action.OPEN_APP and not _clean_text(recovered_action.app_name):
            inferred = self._infer_open_app_name_from_goal(goal)
            if inferred:
                recovered_action = json_action.JSONAction(
                    action_type=json_action.OPEN_APP,
                    app_name=inferred,
                )
                recovered_tool_call = {
                    "name": "mobile_use",
                    "arguments": {"action": "open_app", "text": inferred},
                }
            else:
                raise seeact_utils.ParseActionError("open_app_missing_name")
        if recovered_action.action_type == json_action.UNKNOWN:
            raise seeact_utils.ParseActionError("tool_call_recovery_unknown_action")

        recovered_parsed_action = OrderedDict(
            cot="",
            action="RECOVERED_TOOL_CALL",
            summary="Recovered action from tool_call parser.",
        )
        recovered_extras: dict[str, Any] = {"recovered_from_tool_call": True}
        return recovered_action, recovered_tool_call, recovered_extras, recovered_parsed_action

    def _select_replay_actions_for_probe(
        self,
        current_action: json_action.JSONAction | None,
    ) -> list[json_action.JSONAction]:
        actions: list[json_action.JSONAction] = []
        if self._is_replay_safe_action(current_action):
            actions.append(current_action)
        for record in reversed(list(self._actions)):
            action = self._json_action_from_record(record.get("action_dict"))
            if action is None:
                continue
            actions.append(action)
        actions.reverse()
        return actions

    def _rollback_to_probe_root(
        self,
        root_activity: str,
        root_hash: int,
        replay_actions: list[json_action.JSONAction],
        step_idx: int,
    ) -> dict[str, Any]:
        def _verify_root() -> tuple[bool, str]:
            curr_state = self.env.get_state(wait_to_stabilize=True)
            return self._same_root_page(curr_state, root_activity, root_hash)

        def _execute_replay(tag: str, actions: list[json_action.JSONAction], counter_key: str) -> None:
            for action in actions:
                try:
                    self.env.execute_action(action)
                    result[counter_key] += 1
                    # Stabilize after replay so the next replay action is grounded on a stable UI.
                    self.env.get_state(wait_to_stabilize=True)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    print(
                        f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
                        f"{tag}_action_failed action={action.action_type} error={exc}"
                    )

        rollback_start = time.time()
        result: dict[str, Any] = {
            "success": False,
            "mode": "backtrack_failed",
            "back_presses": 0,
            "replayed_actions": 0,
            "matched_by": None,
            "level": "level1",
            "latency_ms": 0.0,
        }

        back_limit = max(1, int(self.light_explore_back_limit))
        for i in range(back_limit + 1):
            curr_state = self.env.get_state(wait_to_stabilize=True)
            same, matched_by = self._same_root_page(curr_state, root_activity, root_hash)
            if same:
                result["success"] = True
                result["mode"] = "backtrack"
                result["matched_by"] = matched_by
                result["latency_ms"] = float(max(0.0, time.time() - rollback_start) * 1000.0)
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
            result["mode"] = "replay_in_place"
            result["level"] = "level2"
            print(f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} rollback_fallback=replay_in_place")
            _execute_replay("rollback_replay_in_place", replay_actions, "replayed_actions")
            same, matched_by = _verify_root()
            if same:
                result["success"] = True
                result["matched_by"] = matched_by
                result["mode"] = "replay_in_place"
                result["latency_ms"] = float(max(0.0, time.time() - rollback_start) * 1000.0)
                print(
                    f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
                    f"rollback_replay_in_place_done success=True replayed={result['replayed_actions']} "
                    f"matched_by={matched_by}"
                )
                return result

            print(f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} rollback_fallback=home_replay")
            try:
                self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_HOME))
                self.env.get_state(wait_to_stabilize=True)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(
                    f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
                    f"rollback_home_action_failed error={exc}"
                )
            _execute_replay("rollback_home_replay", replay_actions, "replayed_actions")
            same, matched_by = _verify_root()
            result["success"] = bool(same)
            result["matched_by"] = matched_by
            result["mode"] = "home_replay" if same else "home_replay_failed"
            result["latency_ms"] = float(max(0.0, time.time() - rollback_start) * 1000.0)
            print(
                f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
                f"rollback_home_replay_done success={result['success']} replayed={result['replayed_actions']} "
                f"matched_by={matched_by}"
            )
            return result

        result["latency_ms"] = float(max(0.0, time.time() - rollback_start) * 1000.0)
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
        if diff is not None and diff >= PAGE_CHANGED_HASH_DIFF:
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

    @staticmethod
    def _has_keyboard_ui(state: Any) -> bool:
        for element in list(getattr(state, "ui_elements", None) or []):
            text = _clean_text(getattr(element, "resource_id", "")) + " " + _clean_text(
                getattr(element, "content_description", "")
            )
            low = text.lower()
            if "inputmethod" in low or "keyboard" in low or "key_pos_" in low:
                return True
        return False

    def _should_run_light_exploration(
        self,
        step_idx: int,
        root_activity: str,
        page_stalled: bool,
        page_clickable_element_count: int,
    ) -> tuple[bool, str]:
        if not self.enable_light_exploration:
            return False, "disabled"
        if self.light_explore_max_runs <= 0:
            return False, "run_budget_zero"
        if self._light_explore_runs >= self.light_explore_max_runs:
            return False, "run_budget_exhausted"
        if step_idx >= self.light_explore_max_step:
            return False, "beyond_step_window"
        high_complexity = page_clickable_element_count >= int(
            self.light_explore_high_complexity_clickable_threshold
        )
        periodic_relaxed = bool(((int(step_idx) + 1) % int(self.light_explore_relaxed_period)) == 0)
        if step_idx > 1 and (not page_stalled) and (not self._is_launcher_activity(root_activity)) and (not high_complexity):
            if periodic_relaxed:
                return True, "periodic_relaxed"
            return False, "non_stall_non_launcher"
        if self.light_explore_require_stall and not page_stalled:
            return False, "page_not_stalled"
        if self.light_explore_launcher_only and not self._is_launcher_activity(root_activity):
            return False, "not_launcher_activity"
        if step_idx == 0:
            if self._is_launcher_activity(root_activity):
                return False, "before_first_action_launcher"
            if page_stalled or high_complexity:
                return True, "early_step_boost_non_launcher"
            return False, "before_first_action"
        if high_complexity and not page_stalled:
            return True, "high_complexity_page"
        return True, "ok"

    def _run_light_exploration(
        self,
        goal: str,
        step_idx: int,
        current_action: json_action.JSONAction | None = None,
        page_stalled: bool = False,
        page_clickable_element_count: int = 0,
    ) -> dict[str, Any]:
        self._last_probe_candidates = []
        explore_stats = self._default_explore_stats()
        rollback_stats = self._default_rollback_stats()
        hint_stats = self._default_hint_stats()
        explore_stats["exploration_budget_remaining"] = self._current_exploration_budget_remaining()
        self._last_explore_stats = dict(explore_stats)
        self._last_rollback_stats = dict(rollback_stats)
        self._last_generated_hint_stats = dict(hint_stats)

        try:
            root_state = self.env.get_state(wait_to_stabilize=True)
            root_hash = self._state_hash(root_state)
            root_activity = self._foreground_activity_name()
            if self._has_keyboard_ui(root_state):
                explore_stats["exploration_trigger_reason"] = "keyboard_visible"
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    "light_explore_skipped=keyboard_visible"
                )
                self._last_explore_stats = dict(explore_stats)
                return {"hint_text": "", "hint_payload": dict(hint_stats)}
            should_run, reason = self._should_run_light_exploration(
                step_idx=step_idx,
                root_activity=root_activity,
                page_stalled=page_stalled,
                page_clickable_element_count=page_clickable_element_count,
            )
            explore_stats["exploration_trigger_reason"] = str(reason)
            print(
                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                f"light_explore_should_run={should_run} reason={reason} page_stalled={page_stalled}"
            )
            if not should_run:
                self._last_explore_stats = dict(explore_stats)
                return {"hint_text": "", "hint_payload": dict(hint_stats)}

            explore_stats["exploration_triggered"] = True
            candidates = self._collect_probe_candidates(root_state, goal)
            self._last_probe_candidates = list(candidates)
            explore_stats["exploration_candidate_count"] = int(len(candidates))
            if not candidates:
                explore_stats["exploration_trigger_reason"] = "no_candidates"
                print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_no_candidates")
                self._last_explore_stats = dict(explore_stats)
                return {"hint_text": "", "hint_payload": dict(hint_stats)}
            replay_actions = self._select_replay_actions_for_probe(current_action=current_action)
            pool = list(candidates)
            branch_candidates: list[dict[str, Any]] = []
            for _ in range(max(1, int(self.light_explore_branch_budget))):
                chosen = self._choose_probe_candidate(pool)
                if chosen is None:
                    break
                branch_candidates.append(chosen)
                chosen_center = chosen.get("center")
                pool = [c for c in pool if c.get("center") != chosen_center]
            explore_stats["exploration_selected_target_count"] = int(len(branch_candidates))
            explore_stats["exploration_branch_count"] = int(len(branch_candidates))
            if not branch_candidates:
                explore_stats["exploration_trigger_reason"] = "no_branch_candidates"
                print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_no_branch_candidates")
                self._last_explore_stats = dict(explore_stats)
                return {"hint_text": "", "hint_payload": dict(hint_stats)}
            shortlist = ", ".join(
                f"{_clean_text(c.get('label')) or 'candidate'}:{float(c.get('score') or 0.0):.2f}"
                for c in branch_candidates
            )
            print(
                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                f"light_explore_branch_begin candidates={len(branch_candidates)} depth={self.light_explore_branch_depth} "
                f"shortlist=[{shortlist}]"
            )

            best_observation: dict[str, Any] | None = None
            attempted_any = False
            rollback_all_success = True
            rollback_matched_by = None
            rollback_highest_level = "none"
            for b_idx, first_candidate in enumerate(branch_candidates):
                labels: list[str] = []
                total_score = float(first_candidate.get("score") or 0.0)
                changed_any = False
                final_activity = root_activity
                depth_reached = 0
                hint_target_candidate: dict[str, Any] | None = first_candidate
                hint_source_depth = 1
                rollback_info = None

                first_center = first_candidate.get("center")
                if not isinstance(first_center, (list, tuple)) or len(first_center) < 2:
                    continue
                first_label = _clean_text(first_candidate.get("label")) or f"candidate_{b_idx}"
                labels.append(first_label)
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    f"light_explore_branch={b_idx + 1} depth=1 click={first_label} center={[int(first_center[0]), int(first_center[1])]}"
                )
                attempted_any = True
                self.env.execute_action(
                    json_action.JSONAction(
                        action_type=json_action.CLICK,
                        x=int(first_center[0]),
                        y=int(first_center[1]),
                    )
                )
                state_after_first = self.env.get_state(wait_to_stabilize=True)
                changed1, activity1, hash_diff1 = self._probe_page_changed(root_activity, root_hash, state_after_first)
                depth_reached = 1
                changed_any = bool(changed_any or changed1)
                final_activity = activity1
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    f"light_explore_branch={b_idx + 1} depth=1 changed={changed1} hash_diff={hash_diff1} after={activity1}"
                )

                allow_depth2 = int(self.light_explore_branch_depth) >= 2 and b_idx == 0
                if allow_depth2:
                    second_candidates = self._collect_probe_candidates(state_after_first, goal)
                    second_candidate = self._choose_secondary_probe_candidate(second_candidates, first_candidate=first_candidate)
                    if second_candidate is not None:
                        second_center = second_candidate.get("center")
                        if isinstance(second_center, (list, tuple)) and len(second_center) >= 2:
                            second_label = _clean_text(second_candidate.get("label")) or f"candidate_{b_idx}_2"
                            labels.append(second_label)
                            total_score += float(second_candidate.get("score") or 0.0)
                            print(
                                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                                f"light_explore_branch={b_idx + 1} depth=2 click={second_label} center={[int(second_center[0]), int(second_center[1])]}"
                            )
                            self.env.execute_action(
                                json_action.JSONAction(
                                    action_type=json_action.CLICK,
                                    x=int(second_center[0]),
                                    y=int(second_center[1]),
                                )
                            )
                            state_after_second = self.env.get_state(wait_to_stabilize=True)
                            changed2, activity2, hash_diff2 = self._probe_page_changed(root_activity, root_hash, state_after_second)
                            depth_reached = 2
                            changed_any = bool(changed_any or changed2)
                            final_activity = activity2
                            if changed2:
                                hint_target_candidate = second_candidate
                                hint_source_depth = 2
                            print(
                                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                                f"light_explore_branch={b_idx + 1} depth=2 changed={changed2} hash_diff={hash_diff2} after={activity2}"
                            )

                rollback_info = self._rollback_to_probe_root(
                    root_activity=root_activity,
                    root_hash=root_hash,
                    replay_actions=replay_actions,
                    step_idx=step_idx,
                )
                rollback_stats["rollback_triggered"] = True
                rollback_stats["rollback_latency_ms"] = float(
                    rollback_stats["rollback_latency_ms"] + float((rollback_info or {}).get("latency_ms") or 0.0)
                )
                rollback_stats["rollback_back_presses"] = int(
                    rollback_stats["rollback_back_presses"] + int((rollback_info or {}).get("back_presses") or 0)
                )
                rollback_stats["rollback_replayed_actions"] = int(
                    rollback_stats["rollback_replayed_actions"]
                    + int((rollback_info or {}).get("replayed_actions") or 0)
                )
                rollback_stats["rollback_matched_by"] = (rollback_info or {}).get("matched_by")
                rollback_matched_by = rollback_stats["rollback_matched_by"] or rollback_matched_by
                level = str((rollback_info or {}).get("level") or "level1")
                if level == "level2":
                    rollback_highest_level = "level2"
                elif rollback_highest_level == "none":
                    rollback_highest_level = "level1"
                if not bool((rollback_info or {}).get("success")):
                    print(
                        f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                        f"light_explore_restore_failed={rollback_info}"
                    )
                    rollback_all_success = False
                    self._light_explore_runs += 1
                    self._rollback_failure_streak += 1
                    if self._rollback_failure_streak >= int(self.light_explore_disable_after_rollback_failures):
                        self.enable_light_exploration = False
                    explore_stats["exploration_trigger_reason"] = "rollback_failed"
                    explore_stats["exploration_budget_remaining"] = self._current_exploration_budget_remaining()
                    rollback_stats["rollback_level"] = rollback_highest_level
                    rollback_stats["rollback_success"] = False
                    rollback_stats["rollback_matched_by"] = rollback_matched_by
                    self._last_explore_stats = dict(explore_stats)
                    self._last_rollback_stats = dict(rollback_stats)
                    return {"hint_text": "", "hint_payload": dict(hint_stats)}
                self._rollback_failure_streak = 0

                observation = {
                    "labels": list(labels),
                    "changed": bool(changed_any),
                    "after_activity": final_activity,
                    "score": float(total_score),
                    "depth_reached": int(depth_reached),
                    "hint_source_depth": int(hint_source_depth),
                    "hint_target_candidate": hint_target_candidate,
                }
                if best_observation is None:
                    best_observation = observation
                else:
                    curr_key = (
                        1 if bool(observation.get("changed")) else 0,
                        int(observation.get("depth_reached") or 0),
                        float(observation.get("score") or 0.0),
                    )
                    best_key = (
                        1 if bool(best_observation.get("changed")) else 0,
                        int(best_observation.get("depth_reached") or 0),
                        float(best_observation.get("score") or 0.0),
                    )
                    if curr_key > best_key:
                        best_observation = observation

            if attempted_any:
                self._light_explore_runs += 1
            explore_stats["exploration_budget_remaining"] = self._current_exploration_budget_remaining()
            rollback_stats["rollback_level"] = rollback_highest_level
            rollback_stats["rollback_success"] = bool(rollback_all_success and rollback_stats["rollback_triggered"])
            rollback_stats["rollback_matched_by"] = rollback_matched_by
            if not best_observation:
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    "light_explore_no_branch_observation"
                )
                self._last_explore_stats = dict(explore_stats)
                self._last_rollback_stats = dict(rollback_stats)
                return {"hint_text": "", "hint_payload": dict(hint_stats)}

            path_label = " -> ".join([_clean_text(x) for x in list(best_observation.get("labels") or []) if _clean_text(x)])
            synthetic_candidate = {"label": path_label or "candidate-path"}
            hint_candidate = best_observation.get("hint_target_candidate")
            if not isinstance(hint_candidate, dict):
                hint_candidate = synthetic_candidate
            hint_source_depth = best_observation.get("hint_source_depth")
            hint = self._build_hint_from_observation(
                candidate=synthetic_candidate,
                changed=bool(best_observation.get("changed")),
                after_activity=str(best_observation.get("after_activity") or ""),
            )
            explore_stats["exploration_best_branch_depth"] = int(best_observation.get("depth_reached") or 0)
            if hint:
                hint = _clean_text(f"{hint} (rollback verified)")
                hint_stats = self._hint_payload_from_candidate(
                    candidate=hint_candidate,
                    hint_text=hint,
                    hint_source_depth=int(hint_source_depth) if hint_source_depth is not None else None,
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
            self._last_explore_stats = dict(explore_stats)
            self._last_rollback_stats = dict(rollback_stats)
            self._last_generated_hint_stats = dict(hint_stats)
            return {"hint_text": _clean_text(hint), "hint_payload": dict(hint_stats)}
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(
                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                f"light_explore_failed: {exc}"
            )
            self._light_explore_runs += 1
            explore_stats["exploration_trigger_reason"] = f"exception:{_clean_text(exc)}"
            explore_stats["exploration_budget_remaining"] = self._current_exploration_budget_remaining()
            self._last_explore_stats = dict(explore_stats)
            self._last_rollback_stats = dict(rollback_stats)
            self._last_generated_hint_stats = dict(hint_stats)
            return {"hint_text": "", "hint_payload": dict(hint_stats)}

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        start_time = time.time()
        self._ensure_episode_identity(goal)
        # Keep screenshot scaling fixed at 2 through the full episode.
        self.image_downsample_scale = float(self.fixed_image_downsample_scale)
        step_idx = len(self._actions)
        if step_idx >= self._effective_max_steps():
            summary = f"Reached the maximum step limit ({self._effective_max_steps()})."
            action = json_action.JSONAction(action_type=json_action.STATUS, goal_status="infeasible")
            tool_call = {"name": "mobile_use", "arguments": {"action": "terminate", "status": "fail"}}
            print("=" * 96)
            print(f"Step {step_idx}: Result")
            print(summary)
            print("=" * 96)
            self._write_effectiveness_trace(goal=goal, episode_success=False, finalize=True)
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
                    "task_id": self._episode_task_id,
                    "task_name": self._episode_task_name,
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
        page_complexity = self._count_page_complexity(state)
        page_stalled = self._is_page_stalled(
            current_activity=start_page_activity,
            current_hash=start_page_hash,
        )

        hint_for_prompt_payload = dict(self._pending_explore_hint_payload or {})
        hint_for_prompt = _clean_text(hint_for_prompt_payload.get("hint_text") or self._pending_explore_hint)
        hint_bbox_for_prompt = hint_for_prompt_payload.get("hint_bbox")
        self._pending_explore_hint = ""
        self._pending_explore_hint_payload = {}
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
            try:
                action, tool_call, extras, parsed_action = self._recover_action_from_tool_call(
                    response=str(response),
                    state=state,
                    screen_size=screen_size,
                    goal=goal,
                )
                parsed_action["parse_error"] = parse_error
                parsed_action["fallback"] = "tool_call_recovery"
            except Exception as recovery_error:  # pylint: disable=broad-exception-caught
                parsed_action = OrderedDict(
                    cot="",
                    action="WAIT",
                    value="1",
                    summary="Parser fallback wait",
                    parse_error=parse_error,
                    recovery_error=_clean_text(recovery_error),
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
        after_action_state = self.env.get_state(wait_to_stabilize=True)
        after_action_activity = self._foreground_activity_name()
        after_action_hash = self._state_hash(after_action_state)
        gelab_agent._print_step_section(step_idx, "Action", gelab_agent._json_dumps_safe(action.__dict__))  # pylint: disable=protected-access
        if extras:
            gelab_agent._print_step_section(step_idx, "Action extras", gelab_agent._json_dumps_safe(extras))  # pylint: disable=protected-access

        summary = gelab_agent._normalize_space(parsed_action.get("summary")) or gelab_agent._normalize_space(parsed_action.get("explain"))  # pylint: disable=protected-access
        if not summary:
            summary = str(tool_call.get("arguments") or tool_call)

        screen_diff = self._safe_hash_diff(start_page_hash, after_action_hash)
        activity_changed = (
            self._normalize_activity_name(start_page_activity)
            != self._normalize_activity_name(after_action_activity)
        )
        page_changed = bool(
            activity_changed or (screen_diff is not None and screen_diff >= PAGE_CHANGED_HASH_DIFF)
        )
        no_effect = bool(
            action.action_type not in {json_action.STATUS, json_action.ANSWER}
            and (not page_changed)
            and (screen_diff is not None and screen_diff <= NO_EFFECT_HASH_DIFF)
        )
        if no_effect:
            self._repeated_no_effect_count += 1
        else:
            self._repeated_no_effect_count = 0
        valid_transition = bool(
            action.action_type in {json_action.STATUS, json_action.ANSWER}
            or page_changed
            or (screen_diff is not None and screen_diff > NO_EFFECT_HASH_DIFF)
        )

        final_action_type = str(action.action_type)
        final_action_coordinate = self._action_coordinate_if_click(action)
        final_action_bbox_match_hint = self._point_in_bbox(final_action_coordinate, hint_bbox_for_prompt)

        # Exploration runs after current-step action; hint is consumed at next step.
        explore_out = self._run_light_exploration(
            goal=goal,
            step_idx=step_idx,
            current_action=action,
            page_stalled=page_stalled,
            page_clickable_element_count=page_complexity["page_clickable_element_count"],
        )
        next_hint = _clean_text(explore_out.get("hint_text") if isinstance(explore_out, dict) else "")
        next_hint_payload = explore_out.get("hint_payload") if isinstance(explore_out, dict) else None
        if not isinstance(next_hint_payload, dict):
            next_hint_payload = self._default_hint_stats()
        next_hint_payload = dict(next_hint_payload)
        next_hint_payload["hint_text"] = next_hint
        self._pending_explore_hint = next_hint
        self._pending_explore_hint_payload = dict(next_hint_payload)
        generated_hint_stats = dict(self._last_generated_hint_stats or self._default_hint_stats())
        generated_hint_stats["hint_injected_into_prompt"] = bool(hint_for_prompt)

        latency_sec = float(max(0.0, time.time() - start_time))
        done = action.action_type in {json_action.STATUS, json_action.ANSWER}
        is_last_budget_step = (step_idx + 1) >= self._effective_max_steps()
        episode_success = self._infer_episode_success(action) if done else (False if is_last_budget_step else None)
        finalize_trace = bool(done or is_last_budget_step)
        explore_stats = dict(self._last_explore_stats or self._default_explore_stats())
        rollback_stats = dict(self._last_rollback_stats or self._default_rollback_stats())

        # These fields are consumed by paper-level effectiveness analysis.
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
            "page_total_element_count": page_complexity["page_total_element_count"],
            "page_clickable_element_count": page_complexity["page_clickable_element_count"],
            "page_editable_element_count": page_complexity["page_editable_element_count"],
            "exploration_triggered": bool(explore_stats.get("exploration_triggered")),
            "exploration_trigger_reason": explore_stats.get("exploration_trigger_reason"),
            "exploration_budget_remaining": int(explore_stats.get("exploration_budget_remaining") or 0),
            "exploration_candidate_count": int(explore_stats.get("exploration_candidate_count") or 0),
            "exploration_selected_target_count": int(explore_stats.get("exploration_selected_target_count") or 0),
            "exploration_branch_count": int(explore_stats.get("exploration_branch_count") or 0),
            "exploration_best_branch_depth": int(explore_stats.get("exploration_best_branch_depth") or 0),
            "rollback_triggered": bool(rollback_stats.get("rollback_triggered")),
            "rollback_level": rollback_stats.get("rollback_level"),
            "rollback_success": bool(rollback_stats.get("rollback_success")),
            "rollback_latency_ms": float(rollback_stats.get("rollback_latency_ms") or 0.0),
            "rollback_back_presses": int(rollback_stats.get("rollback_back_presses") or 0),
            "rollback_replayed_actions": int(rollback_stats.get("rollback_replayed_actions") or 0),
            "rollback_matched_by": rollback_stats.get("rollback_matched_by"),
            "hint_generated": bool(generated_hint_stats.get("hint_generated")),
            "hint_text": generated_hint_stats.get("hint_text"),
            "hint_source_depth": generated_hint_stats.get("hint_source_depth"),
            "hint_bbox": generated_hint_stats.get("hint_bbox"),
            "hint_target_element_text": generated_hint_stats.get("hint_target_element_text"),
            "hint_target_element_id": generated_hint_stats.get("hint_target_element_id"),
            "hint_injected_into_prompt": bool(generated_hint_stats.get("hint_injected_into_prompt")),
            "final_action_type": final_action_type,
            "final_action_coordinate": final_action_coordinate,
            "final_action_bbox_match_hint": final_action_bbox_match_hint,
            "screen_diff": screen_diff,
            "no_effect": no_effect,
            "repeated_no_effect_count": int(self._repeated_no_effect_count),
            "page_changed": page_changed,
            "activity_changed": activity_changed,
            "valid_transition": valid_transition,
            "task_id": self._episode_task_id,
            "task_name": self._episode_task_name,
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
            self._write_effectiveness_trace(
                goal=goal,
                episode_success=episode_success,
                finalize=finalize_trace,
            )

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
                "task_id": self._episode_task_id,
                "task_name": self._episode_task_name,
                "exploration_triggered": bool(explore_stats.get("exploration_triggered")),
                "exploration_trigger_reason": explore_stats.get("exploration_trigger_reason"),
                "exploration_budget_remaining": int(explore_stats.get("exploration_budget_remaining") or 0),
                "rollback_triggered": bool(rollback_stats.get("rollback_triggered")),
                "rollback_level": rollback_stats.get("rollback_level"),
                "rollback_success": bool(rollback_stats.get("rollback_success")),
                "hint_generated": bool(generated_hint_stats.get("hint_generated")),
                "hint_bbox": generated_hint_stats.get("hint_bbox"),
                "final_action_type": final_action_type,
                "final_action_coordinate": final_action_coordinate,
                "final_action_bbox_match_hint": final_action_bbox_match_hint,
                "screen_diff": screen_diff,
                "no_effect": no_effect,
                "repeated_no_effect_count": int(self._repeated_no_effect_count),
                "page_changed": page_changed,
                "activity_changed": activity_changed,
                "valid_transition": valid_transition,
            },
        )


class ElementTextAgent(ExplorerElementAgent):
    pass
