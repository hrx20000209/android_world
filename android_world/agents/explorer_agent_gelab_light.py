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

For effectiveness instrumentation used in paper analysis, see
`explorer_agent_gelab_effectiveness.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import concurrent.futures
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
from android_world.agents import slot_complete_evidence
from android_world.env import adb_utils
from android_world.env import json_action

MAX_EXPLORER_STEPS = int(os.environ.get("ANDROID_WORLD_EXPLORER_MAX_STEPS", "16") or "16")
PAGE_CHANGED_HASH_DIFF = 6
PROMPT_RESULT_LIMIT = 3
TRACE_A11Y_LIMIT = 80
TRACE_OBSERVED_ELEMENT_LIMIT = 12
GOAL_TOKEN_STOPWORDS = {
    "app",
    "application",
    "using",
    "use",
    "open",
    "then",
    "when",
    "with",
    "their",
    "task",
    "current",
    "screen",
}
SECONDARY_TOKEN_STOPWORDS = GOAL_TOKEN_STOPWORDS.union(
    {
        "pro",
        "file",
        "files",
        "folder",
        "folders",
        "expense",
        "expenses",
        "recipe",
        "recipes",
        "note",
        "notes",
        "task",
        "tasks",
    }
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _now_hms() -> str:
    return time.strftime("%H:%M:%S")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _to_user_text(goal: str, history: str, hint: str) -> str:
    goal_l = _clean_text(goal).lower()
    media_capture_note = ""
    if re.search(
        r"\b(record (?:an? )?(?:audio|video|clip)|take (?:a |the )?(?:photo|picture|video)|capture (?:a |the )?(?:photo|picture|video))\b",
        goal_l,
    ):
        media_capture_note = (
            "- For recording/photo/video capture tasks, once the current screen shows the clip/photo/video was captured, saved, or appears in the media/recording list, choose COMPLETE instead of repeatedly starting/stopping/capturing again.\n"
        )
    destructive_note = ""
    if re.search(r"\b(delete|remove|trash|discard|clear all|erase)\b", goal_l):
        destructive_note = (
            "- For delete/remove tasks involving files, expenses, recipes, notes, events, tasks, contacts, playlists, or list rows, do not COMPLETE immediately after a dialog confirmation; first verify on the current screen that each exact target item is absent from the relevant list/search/folder.\n"
        )
    decision_constraints = (
        "Decision constraints:\n"
        "- Do not choose COMPLETE, ANSWER, or task_complete unless the current screen visibly proves the requested final state.\n"
        "- For pure operation tasks, if the current screen already visibly proves the requested action is done, choose COMPLETE rather than repeating the same click/type action.\n"
        f"{media_capture_note}"
        "- For file delete or file move tasks, do not complete immediately after a destructive dialog or one list observation; first verify the folder/path and the exact source absence or destination presence on the current screen.\n"
        f"{destructive_note}"
        "- If exploration context conflicts with the current screen, ignore exploration context and act only on the current screen.\n\n"
    )
    if hint:
        return (
            f"Task:\n{goal}\n\n"
            f"History actions:\n{history or 'None yet.'}\n\n"
            f"Exploration context:\n{hint}\n\n"
            f"{decision_constraints}"
            "Current screenshot is attached below.\n"
            "Choose the next single action."
        )
    return (
        f"Task:\n{goal}\n\n"
        f"History actions:\n{history or 'None yet.'}\n\n"
        f"{decision_constraints}"
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
        output_path: str = "./output/explorer_agent_gelab_light",
        history_limit: int = 8,
        image_downsample_scale: float = 2.0,
        enable_light_exploration: bool = True,
        light_explore_max_runs: int = 3,
        light_explore_max_step: int = 8,
        light_explore_launcher_only: bool = False,
        light_explore_require_keyword: bool = False,
        light_explore_require_stall: bool = False,
        light_explore_branch_budget: int = 1,
        light_explore_branch_depth: int = 2,
        light_explore_back_limit: int = 3,
        light_explore_hash_threshold: int = 10,
        light_explore_visit_penalty: float = 0.18,
        light_explore_min_launcher_relevance: float = 0.20,
        light_explore_prompt_result_limit: int = PROMPT_RESULT_LIMIT,
        **kwargs: Any,
    ):
        _ = kwargs
        output_path = os.environ.get("ANDROID_WORLD_EXPLORATION_TRACE_ROOT", output_path)
        super().__init__(
            env=env,
            vllm=vllm,
            name=name,
            output_path=output_path,
            history_limit=history_limit,
            image_downsample_scale=image_downsample_scale,
        )
        enable_light_exploration = _env_bool("ANDROID_WORLD_LIGHT_EXPLORE_ENABLE", enable_light_exploration)
        light_explore_max_runs = _env_int("ANDROID_WORLD_LIGHT_EXPLORE_MAX_RUNS", light_explore_max_runs)
        light_explore_max_step = _env_int("ANDROID_WORLD_LIGHT_EXPLORE_MAX_STEP", light_explore_max_step)
        light_explore_branch_budget = _env_int("ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_BUDGET", light_explore_branch_budget)
        light_explore_branch_depth = _env_int("ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_DEPTH", light_explore_branch_depth)
        light_explore_back_limit = _env_int("ANDROID_WORLD_LIGHT_EXPLORE_BACK_LIMIT", light_explore_back_limit)
        self._initial_enable_light_exploration = bool(enable_light_exploration)
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
        self.light_explore_visit_penalty = max(0.0, float(light_explore_visit_penalty))
        self.light_explore_min_launcher_relevance = max(0.0, float(light_explore_min_launcher_relevance))
        self.light_explore_prompt_result_limit = max(1, int(light_explore_prompt_result_limit))
        self.light_explore_planned_only = _env_bool("ANDROID_WORLD_LIGHT_EXPLORE_PLANNED_ONLY", True)
        self.light_explore_fallback_safe_candidates = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_FALLBACK_SAFE_CANDIDATES",
            True,
        )
        self.light_explore_safe_click_only = _env_bool("ANDROID_WORLD_LIGHT_EXPLORE_SAFE_CLICK_ONLY", True)
        self.light_explore_skip_launcher = _env_bool("ANDROID_WORLD_LIGHT_EXPLORE_SKIP_LAUNCHER", True)
        self.light_explore_filter_launcher_relevance = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_FILTER_LAUNCHER_RELEVANCE",
            True,
        )
        self.light_explore_diagnostic_full = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_DIAGNOSTIC_FULL",
            False,
        )
        self.light_explore_force_every_step = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_FORCE_EVERY_STEP",
            False,
        )
        self.light_explore_fast_mode = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_FAST_MODE",
            False,
        )
        self.light_explore_fast_state = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_FAST_STATE",
            self.light_explore_fast_mode,
        )
        self.light_explore_save_screenshots = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_SAVE_SCREENSHOTS",
            True,
        )
        self.light_explore_transaction_safe = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_TRANSACTION_SAFE",
            True,
        )
        # Legacy experiments accumulated English label allow/deny lists and
        # app-specific keyword fixes.  They are disabled by default because
        # they do not generalize to icons, localization, or unseen apps.  Keep
        # the switch only so historical runs remain reproducible.
        self.light_explore_lexical_rules = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_LEXICAL_RULES",
            False,
        )
        self.light_explore_action_settle_s = max(
            0.0,
            _env_float(
                "ANDROID_WORLD_LIGHT_EXPLORE_ACTION_SETTLE_S",
                0.25 if self.light_explore_fast_mode else 0.0,
            ),
        )
        self.light_explore_strategy = _clean_text(
            os.environ.get("ANDROID_WORLD_LIGHT_EXPLORE_STRATEGY", "dfs")
        ).lower()
        if self.light_explore_strategy not in {"dfs", "bfs"}:
            self.light_explore_strategy = "dfs"
        self.light_explore_skip_risky_planned = _env_bool("ANDROID_WORLD_LIGHT_EXPLORE_SKIP_RISKY_PLANNED", True)
        self.light_explore_require_a11y_anchor = _env_bool("ANDROID_WORLD_LIGHT_EXPLORE_REQUIRE_A11Y_ANCHOR", True)
        self.light_explore_skip_destructive_goals = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_SKIP_DESTRUCTIVE_GOALS",
            True,
        )
        self.light_explore_disable_prompt_injection = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_DISABLE_PROMPT_INJECTION",
            False,
        )
        self.light_explore_hint_policy = _clean_text(
            os.environ.get("ANDROID_WORLD_LIGHT_EXPLORE_HINT_POLICY", "current")
        ).lower()
        if self.light_explore_hint_policy not in {"current", "strict"}:
            self.light_explore_hint_policy = "current"
        self.light_explore_search_policy = _clean_text(
            os.environ.get("ANDROID_WORLD_LIGHT_EXPLORE_SEARCH_POLICY", "current")
        ).lower()
        if self.light_explore_search_policy not in {"current", "operator", "task_gate"}:
            self.light_explore_search_policy = "current"
        self.light_explore_search_strategy = _clean_text(
            os.environ.get("ANDROID_WORLD_LIGHT_EXPLORE_SEARCH_STRATEGY", "greedy")
        ).lower()
        strategy_aliases = {
            "current": "greedy",
            "similarity": "greedy",
            "greedy_similarity": "greedy",
            "stratified": "stratified_bfs",
            "bfs": "stratified_bfs",
            "dfs": "iddfs",
            "iterative_deepening": "iddfs",
            "bestfirst": "best_first",
            "evidence_best_first": "best_first",
            "beam_search": "beam",
            "uct": "mcts",
            "voc": "value_of_computation",
            "adaptive_voc": "value_of_computation",
            "value_computation": "value_of_computation",
            "value_of_computation_guided": "value_of_computation",
        }
        self.light_explore_search_strategy = strategy_aliases.get(
            self.light_explore_search_strategy,
            self.light_explore_search_strategy,
        )
        if self.light_explore_search_strategy not in {
            "greedy",
            "stratified_bfs",
            "iddfs",
            "best_first",
            "beam",
            "mcts",
            "value_of_computation",
        }:
            self.light_explore_search_strategy = "greedy"
        self._active_voc_depth_budget: int | None = None
        self.light_explore_rollback_policy = _clean_text(
            os.environ.get("ANDROID_WORLD_LIGHT_EXPLORE_ROLLBACK_POLICY", "current")
        ).lower()
        if self.light_explore_rollback_policy not in {"current", "improved"}:
            self.light_explore_rollback_policy = "current"
        self.light_explore_fixed_framework = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_FIXED_FRAMEWORK",
            False,
        )
        self.light_explore_variant = _clean_text(
            os.environ.get("ANDROID_WORLD_LIGHT_EXPLORE_VARIANT", "")
        )
        self.light_explore_pattern_aware_operator_best_first = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_PATTERN_AWARE_OPERATOR_BEST_FIRST",
            "PATTERN_AWARE_OPERATOR_BEST_FIRST" in self.light_explore_variant.upper(),
        )
        self.light_explore_safe_mcts = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_SAFE_MCTS",
            False,
        )
        self.light_explore_answer_extractors = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_ANSWER_EXTRACTORS",
            False,
        )
        self.light_explore_slot_complete = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_SLOT_COMPLETE",
            False,
        )
        self.light_explore_no_action_hint = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_NO_ACTION_HINT",
            False,
        )
        self.light_explore_lightweight_a11y_trace = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_LIGHTWEIGHT_A11Y_TRACE",
            False,
        )
        self.light_explore_trace_a11y_limit = max(
            1,
            _env_int(
                "ANDROID_WORLD_LIGHT_EXPLORE_TRACE_A11Y_LIMIT",
                TRACE_A11Y_LIMIT,
            ),
        )
        self.light_explore_slot_policy_switcher = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_SLOT_POLICY_SWITCHER",
            False,
        )
        self.light_explore_parallel_lookahead = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_PARALLEL_LOOKAHEAD",
            False,
        )
        self.light_explore_decouple_planned = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_DECOUPLE_PLANNED",
            True,
        )
        self.light_explore_parallel_vlm = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_PARALLEL_VLM",
            self.light_explore_decouple_planned,
        )
        self.exploration_timing = _clean_text(
            os.environ.get("ANDROID_WORLD_EXPLORATION_TIMING", "parallel_shadow")
        ).lower()
        timing_aliases = {
            "pre_reasoning": "pre_reasoning_decoupled",
            "pre": "pre_reasoning_decoupled",
            "post": "post_reasoning_decoupled",
            "invalid_post_planned": "invalid_post_planned_action",
        }
        self.exploration_timing = timing_aliases.get(self.exploration_timing, self.exploration_timing)
        if self.exploration_timing not in {
            "pre_reasoning_decoupled",
            "parallel_shadow",
            "post_reasoning_decoupled",
            "invalid_post_planned_action",
        }:
            self.exploration_timing = "parallel_shadow"
        self.decoupled_exploration = _env_bool("ANDROID_WORLD_DECOUPLED_EXPLORATION", True)
        self.light_explore_use_current_action = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_USE_CURRENT_ACTION",
            False,
        )
        self.light_explore_use_planning_text = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_USE_PLANNING_TEXT",
            False,
        )
        self.light_explore_shadow_only = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_SHADOW_ONLY",
            False,
        )
        self.light_explore_relaxed_diagnostic_injection = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_RELAXED_DIAGNOSTIC_INJECTION",
            False,
        )
        self.light_explore_upper_bound_evidence = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_UPPER_BOUND_EVIDENCE",
            False,
        )
        self.light_explore_summary_all_debug = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_SUMMARY_ALL_DEBUG",
            False,
        )
        self.light_explore_trace_screenshot_mode = _clean_text(
            os.environ.get(
                "ANDROID_WORLD_TRACE_SCREENSHOT_MODE",
                "failure+level2+depth2+injected+sampled",
            )
        )
        self.reasoning_prior_enabled = _env_bool("ANDROID_WORLD_REASONING_PRIOR_ENABLE", True)
        self.reasoning_prior_delayed = _env_bool("ANDROID_WORLD_REASONING_PRIOR_DELAYED", True)
        self.reasoning_prior_promotion = _env_bool("ANDROID_WORLD_REASONING_PRIOR_PROMOTION", True)
        self.reasoning_prior_template_fill = _env_bool("ANDROID_WORLD_REASONING_PRIOR_TEMPLATE_FILL", True)
        self.reasoning_prior_ewma_ms = _env_float("ANDROID_WORLD_REASONING_PRIOR_EWMA_MS", 10000.0)
        self.light_explore_quality_filters = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_QUALITY_FILTERS",
            True,
        )
        if self.decoupled_exploration and self.exploration_timing != "invalid_post_planned_action":
            self.light_explore_decouple_planned = True
            self.light_explore_planned_only = False
            if not self.light_explore_use_current_action:
                self.light_explore_use_current_action = False
            if not self.light_explore_use_planning_text:
                self.light_explore_use_planning_text = False
        if self.exploration_timing == "parallel_shadow":
            self.light_explore_parallel_vlm = True
        elif self.exploration_timing in {"pre_reasoning_decoupled", "post_reasoning_decoupled"}:
            self.light_explore_parallel_vlm = False
        self.light_explore_min_attempts_per_step = max(
            0,
            _env_int("ANDROID_WORLD_LIGHT_EXPLORE_MIN_ATTEMPTS_PER_STEP", 0),
        )
        self.light_explore_diagnostic_artifacts = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_DIAGNOSTIC_ARTIFACTS",
            True,
        )
        self.light_explore_enable_t2_lookahead = _env_bool(
            "ANDROID_WORLD_LIGHT_EXPLORE_ENABLE_T2_LOOKAHEAD",
            True,
        )
        if (
            self.light_explore_decouple_planned
            and os.environ.get("ANDROID_WORLD_LIGHT_EXPLORE_PLANNED_ONLY") is None
        ):
            self.light_explore_planned_only = False
        if (
            self.light_explore_min_attempts_per_step > 0
            and os.environ.get("ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_BUDGET") is None
        ):
            self.light_explore_branch_budget = max(
                self.light_explore_branch_budget,
                self.light_explore_min_attempts_per_step,
            )
        if (
            _clean_text(os.environ.get("ANDROID_WORLD_A11Y_METHOD", "")).lower() == "fast_provider"
            and os.environ.get("ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_BUDGET") is None
        ):
            self.light_explore_branch_budget = max(self.light_explore_branch_budget, 2)
        self.latency_profile_enabled = _env_bool(
            "ANDROID_WORLD_LATENCY_PROFILE",
            False,
        )
        self.t2_shortcut_mode = _clean_text(os.environ.get("ANDROID_WORLD_T2_MODE", "off")).lower()
        self.t2_shortcut_mode = {
            "active": "active_safe",
            "active_strict": "active_safe",
            "active_safe_strict": "active_safe",
            "direct_answer": "direct_answer_shadow",
        }.get(self.t2_shortcut_mode, self.t2_shortcut_mode)
        if self.t2_shortcut_mode not in {"off", "shadow", "active_safe", "direct_answer_shadow"}:
            self.t2_shortcut_mode = "off"
        self.t2_state_mode = _clean_text(os.environ.get("ANDROID_WORLD_T2_STATE_MODE", "FULL_A11Y_CONTROL")) or "FULL_A11Y_CONTROL"
        self.t2_allow_safe_search_input = _env_bool(
            "ANDROID_WORLD_T2_ALLOW_SAFE_SEARCH_INPUT",
            self.t2_shortcut_mode == "active_safe"
            and "SAFE_SEARCH" in _clean_text(self.light_explore_variant).upper(),
        )
        self.t2_confidence_threshold = max(
            0.0,
            min(1.0, _env_float("ANDROID_WORLD_T2_CONFIDENCE_THRESHOLD", 0.85)),
        )
        self.t2_min_top1_score = _env_float("ANDROID_WORLD_T2_MIN_TOP1_SCORE", 7.0)
        self.t2_min_score_margin = _env_float("ANDROID_WORLD_T2_MIN_SCORE_MARGIN", 2.0)
        self.light_explore_min_secondary_score = max(
            0.0,
            _env_float("ANDROID_WORLD_LIGHT_EXPLORE_MIN_SECONDARY_SCORE", 0.10),
        )
        self.light_explore_lb_mcts = bool(
            _env_bool("ANDROID_WORLD_LIGHT_EXPLORE_LB_MCTS", False)
            or _clean_text(getattr(self, "light_explore_variant", "")).upper().startswith("LB_MCTS")
        )
        self.lb_mcts_c_puct = _env_float("ANDROID_WORLD_LB_MCTS_C_PUCT", 1.4)
        self.lb_mcts_lambda_risk = _env_float("ANDROID_WORLD_LB_MCTS_LAMBDA_RISK", 1.0)
        self.lb_mcts_lambda_latency = _env_float("ANDROID_WORLD_LB_MCTS_LAMBDA_LATENCY", 0.5)
        self.lb_mcts_min_budget_ms = _env_float("ANDROID_WORLD_LB_MCTS_MIN_BUDGET_MS", 3000.0)
        self.lb_mcts_max_budget_ms = _env_float("ANDROID_WORLD_LB_MCTS_MAX_BUDGET_MS", 12000.0)
        self.lb_mcts_slack_ratio = _env_float("ANDROID_WORLD_LB_MCTS_SLACK_RATIO", 0.8)
        self.lb_mcts_max_rollouts_per_step = max(
            1,
            _env_int("ANDROID_WORLD_LB_MCTS_MAX_ROLLOUTS_PER_STEP", 12),
        )
        self.lb_mcts_min_depth2_attempts = max(
            0,
            _env_int("ANDROID_WORLD_LB_MCTS_MIN_DEPTH2_ATTEMPTS", 1),
        )
        self.lb_mcts_default_reasoning_latency_ms = _env_float(
            "ANDROID_WORLD_LB_MCTS_DEFAULT_REASONING_LATENCY_MS",
            10000.0,
        )
        self._lb_mcts_reasoning_ewma_ms = float(self.lb_mcts_default_reasoning_latency_ms)
        self._lb_mcts_selection_step_records: dict[int, list[dict[str, Any]]] = {}
        self._lb_mcts_latency_budget_records: dict[int, dict[str, Any]] = {}
        if self.light_explore_diagnostic_full:
            self.light_explore_planned_only = False
            self.light_explore_fallback_safe_candidates = True
            self.light_explore_safe_click_only = False
            self.light_explore_skip_launcher = False
            self.light_explore_filter_launcher_relevance = False
            self.light_explore_skip_risky_planned = False
            self.light_explore_require_a11y_anchor = False
            self.light_explore_skip_destructive_goals = False

        self._pending_explore_hint: str = ""
        self._light_explore_runs: int = 0
        self._last_probe_candidates: list[dict[str, Any]] = []
        self._explored_element_visits: dict[str, int] = {}
        self._no_effect_probe_keys: set[str] = set()
        self._exploration_traces: list[dict[str, Any]] = []
        self._rollback_traces: list[dict[str, Any]] = []
        self._exploration_match_traces: list[dict[str, Any]] = []
        self._all_exploration_traces: list[dict[str, Any]] = []
        self._all_rollback_traces: list[dict[str, Any]] = []
        self._all_exploration_match_traces: list[dict[str, Any]] = []
        self._last_exploration_trace: dict[str, Any] = {}
        self._pending_speculative_traces: list[dict[str, Any]] = []
        self._pending_shortcut_plans: list[dict[str, Any]] = []
        self._shortcut_plans: list[dict[str, Any]] = []
        self._shortcut_events: list[dict[str, Any]] = []
        self._shortcut_shadow_evals: list[dict[str, Any]] = []
        self._direct_answer_candidates: list[dict[str, Any]] = []
        self._all_shortcut_plans: list[dict[str, Any]] = []
        self._all_shortcut_events: list[dict[str, Any]] = []
        self._all_shortcut_shadow_evals: list[dict[str, Any]] = []
        self._all_direct_answer_candidates: list[dict[str, Any]] = []
        self._reasoning_prior_for_next_step: dict[str, Any] | None = None
        self._last_reasoning_prior: dict[str, Any] | None = None
        self._reasoning_prior_ewma_ms: float = 10000.0
        self._pending_promotion: dict[str, Any] | None = None
        self._pending_promotion_guidance_context: str = ""
        self._pending_promotion_events: list[dict[str, Any]] = []
        self._all_pending_promotion_events: list[dict[str, Any]] = []
        self._session_exploration_memory: dict[str, Any] = {}
        self._state_acquisition_metrics: dict[str, float] = {
            "full_a11y_calls": 0.0,
            "screenshot_calls": 0.0,
            "activity_calls": 0.0,
            "cached_state_hits": 0.0,
            "rollback_full_a11y_fallbacks": 0.0,
            "branch_full_a11y_calls": 0.0,
            "root_state_reuse_count": 0.0,
            "passive_no_get_state_count": 0.0,
            "rollback_light_verify_count": 0.0,
            "rollback_full_a11y_fallback_count": 0.0,
            "rooted_depth1_only_count": 0.0,
            "rooted_depth2_count": 0.0,
            "t2_candidate_count": 0.0,
            "searchinput_t2_candidate_count": 0.0,
            "depth2_blocked_by_hash_unchanged_count": 0.0,
            "depth2_blocked_by_semantic_unchanged_count": 0.0,
            "depth2_blocked_by_no_typed_candidate_count": 0.0,
            "depth2_blocked_by_safety_count": 0.0,
            "depth2_blocked_by_should_expand_count": 0.0,
        }
        self._latency_profile_events: list[dict[str, Any]] = []
        self._state_acquisition_context: str = ""
        self._last_strict_not_injected_reasons: list[dict[str, Any]] = []
        self._search_policy_stats: dict[str, dict[str, float]] = {}
        self._last_hint_harmful: bool = False

    @staticmethod
    def _normalize_for_matching(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float, bool)):
            return str(value)
        return _clean_text(value).lower()

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home=go_home)
        self.enable_light_exploration = bool(self._initial_enable_light_exploration)
        self._pending_explore_hint = ""
        self._light_explore_runs = 0
        self._last_probe_candidates = []
        self._explored_element_visits = {}
        self._no_effect_probe_keys = set()
        self._exploration_traces = []
        self._rollback_traces = []
        self._exploration_match_traces = []
        self._last_exploration_trace = {}
        self._pending_speculative_traces = []
        self._pending_shortcut_plans = []
        self._shortcut_plans = []
        self._shortcut_events = []
        self._shortcut_shadow_evals = []
        self._direct_answer_candidates = []
        self._reasoning_prior_for_next_step = None
        self._last_reasoning_prior = None
        self._reasoning_prior_ewma_ms = 10000.0
        self._pending_promotion = None
        self._pending_promotion_guidance_context = ""
        self._session_exploration_memory = {}
        self._last_strict_not_injected_reasons = []
        self._search_policy_stats = {}

    def _get_probe_state(self, wait_to_stabilize: bool = True) -> Any:
        """Fetch state for speculative probing without repeated stabilization dumps."""
        self._state_acquisition_metrics["full_a11y_calls"] = (
            float(self._state_acquisition_metrics.get("full_a11y_calls") or 0.0) + 1.0
        )
        if self._state_acquisition_context == "branch":
            self._state_acquisition_metrics["branch_full_a11y_calls"] = (
                float(self._state_acquisition_metrics.get("branch_full_a11y_calls") or 0.0) + 1.0
            )
        start = time.time()
        state = self.env.get_state(
            wait_to_stabilize=bool(wait_to_stabilize and not self.light_explore_fast_state)
        )
        total_ms = float(max(0.0, time.time() - start) * 1000.0)
        state_aux = self._state_aux_trace(state)
        latency_row = {
            "event": "probe_get_state",
            "context": self._state_acquisition_context,
            "wait_to_stabilize": bool(wait_to_stabilize and not self.light_explore_fast_state),
            "total_ms": total_ms,
        }
        latency_row.update(state_aux)
        self._append_latency_profile_event("", latency_row)
        return state

    @staticmethod
    def _state_auxiliary(state: Any, key: str, default: Any = None) -> Any:
        aux = getattr(state, "auxiliaries", None)
        if not isinstance(aux, dict):
            return default
        return aux.get(key, default)

    def _state_a11y_latency_ms(self, state: Any) -> float:
        try:
            return float(self._state_auxiliary(state, "a11y_latency_sec", 0.0) or 0.0) * 1000.0
        except (TypeError, ValueError):
            return 0.0

    def _state_aux_trace(self, state: Any) -> dict[str, Any]:
        fallback_used = self._state_auxiliary(state, "a11y_fallback_used", False)
        if isinstance(fallback_used, str):
            fallback_used = fallback_used.strip().lower() in {"1", "true", "yes"}
        trace = {
            "a11y_latency_ms": self._state_a11y_latency_ms(state),
            "a11y_method": _clean_text(self._state_auxiliary(state, "a11y_method", "")),
            "a11y_actual_method": _clean_text(self._state_auxiliary(state, "a11y_actual_method", "")),
            "a11y_fallback_used": bool(fallback_used),
            "ui_element_count": int(self._state_auxiliary(state, "ui_element_count", 0) or 0),
        }
        for key in (
            "fast_a11y_capture_ms",
            "fast_a11y_serialize_ms",
            "fast_a11y_service_ms",
            "fast_a11y_node_count",
            "fast_a11y_emitted_count",
            "fast_a11y_payload_bytes",
        ):
            value = self._state_auxiliary(state, key, None)
            if value is not None:
                trace[key] = value
        error = self._state_auxiliary(state, "fast_provider_error", "")
        if error:
            trace["fast_provider_error"] = _clean_text(error)[:500]
        return trace

    def _append_latency_profile_event(self, goal: str, row: dict[str, Any]) -> None:
        if not getattr(self, "latency_profile_enabled", False):
            return
        event = dict(row)
        event.setdefault("timestamp", time.time())
        event.setdefault("task_id", _clean_text(goal)[:200] if goal else "")
        event.setdefault("variant", self.light_explore_variant)
        self._latency_profile_events.append(event)
        task_dir = self._task_output_dir(goal) if goal else ""
        targets = [self.output_path]
        if task_dir:
            targets.insert(0, task_dir)
        for base in targets:
            if not base:
                continue
            try:
                os.makedirs(base, exist_ok=True)
                with open(os.path.join(base, "latency_profile.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def _diagnostic_targets(self, goal: str) -> list[str]:
        targets: list[str] = []
        task_dir = self._task_output_dir(goal) if goal else ""
        if task_dir:
            targets.append(task_dir)
        if self.output_path:
            targets.append(self.output_path)
        deduped: list[str] = []
        seen: set[str] = set()
        for target in targets:
            if target and target not in seen:
                seen.add(target)
                deduped.append(target)
        return deduped

    def _append_diagnostic_jsonl(self, goal: str, filename: str, row: dict[str, Any]) -> None:
        if not getattr(self, "light_explore_diagnostic_artifacts", True):
            return
        event = dict(row)
        event.setdefault("timestamp", time.time())
        event.setdefault("task_id", _clean_text(goal)[:200] if goal else "")
        event.setdefault("variant", self.light_explore_variant)
        event.setdefault("exploration_timing", getattr(self, "exploration_timing", "parallel_shadow"))
        event.setdefault("exploration_timing_mode", getattr(self, "exploration_timing", "parallel_shadow"))
        event.setdefault("uses_current_planned_action", bool(getattr(self, "light_explore_use_current_action", False)))
        event.setdefault("uses_current_vlm_text", bool(getattr(self, "light_explore_use_planning_text", False)))
        event.setdefault("evidence_injected_same_step", False)
        for base in self._diagnostic_targets(goal):
            try:
                os.makedirs(base, exist_ok=True)
                with open(os.path.join(base, filename), "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def _append_step_decoupling_status(
        self,
        goal: str,
        step_idx: int,
        evidence_injected_same_step: bool = False,
    ) -> None:
        uses_action = bool(getattr(self, "light_explore_use_current_action", False))
        uses_vlm = bool(getattr(self, "light_explore_use_planning_text", False))
        invalid_timing = getattr(self, "exploration_timing", "") == "invalid_post_planned_action"
        row = {
            "task_id": _clean_text(goal)[:200],
            "app": ",".join(self._goal_app_keywords(goal)[:3]),
            "task_mode": self._task_mode(goal),
            "step": int(step_idx + 1),
            "variant": self.light_explore_variant,
            "exploration_timing_mode": getattr(self, "exploration_timing", "parallel_shadow"),
            "uses_current_planned_action": uses_action,
            "uses_current_vlm_text": uses_vlm,
            "evidence_injected_same_step": bool(evidence_injected_same_step),
            "decoupled_valid": not bool(uses_action or uses_vlm or evidence_injected_same_step or invalid_timing),
        }
        self._append_diagnostic_jsonl(goal, "step_decoupling_status.jsonl", row)

    def _write_runtime_config_once(self, goal: str) -> None:
        if getattr(self, "_diagnostic_runtime_config_written", False):
            return
        self._diagnostic_runtime_config_written = True
        config = {
            "agent": self.__class__.__name__,
            "variant": self.light_explore_variant,
            "exploration_timing": getattr(self, "exploration_timing", "parallel_shadow"),
            "exploration_timing_mode": getattr(self, "exploration_timing", "parallel_shadow"),
            "decoupled_exploration": bool(getattr(self, "decoupled_exploration", True)),
            "decoupled_from_planned_action": bool(self.light_explore_decouple_planned),
            "uses_current_planned_action": bool(getattr(self, "light_explore_use_current_action", False)),
            "uses_current_vlm_text": bool(getattr(self, "light_explore_use_planning_text", False)),
            "evidence_injected_same_step": False,
            "invalid_run": bool(
                getattr(self, "exploration_timing", "") == "invalid_post_planned_action"
                or (
                    bool(getattr(self, "decoupled_exploration", True))
                    and (
                        bool(getattr(self, "light_explore_use_current_action", False))
                        or bool(getattr(self, "light_explore_use_planning_text", False))
                    )
                )
            ),
            "parallel_with_vlm": bool(self.light_explore_parallel_vlm),
            "search_strategy_name": (
                "Latency-Bounded Task-Aware MCTS Exploration"
                if bool(getattr(self, "light_explore_lb_mcts", False))
                else "Operator-Stratified Best-First Exploration"
            ),
            "root_level_strategy": (
                "latency-bounded task-aware UCT over rollback-safe UI actions"
                if bool(getattr(self, "light_explore_lb_mcts", False))
                else "stratified breadth-first across operator groups"
            ),
            "branch_continuation": "depth-prioritized gated continuation to depth 2",
            "ranking": (
                "UCT = Q + c_puct*P*sqrt(log(N_parent+1)/(N_child+1)) - lambda_risk*RollbackRisk - lambda_latency*LatencyCostNorm"
                if bool(getattr(self, "light_explore_lb_mcts", False))
                else "best-first score"
            ),
            "candidate_score_formula": (
                "PatternPrior + MissingSlotGain + TaskEntityMatch - RiskPenalty - RollbackRisk"
                if getattr(self, "light_explore_pattern_aware_operator_best_first", False)
                else (
                    "MissingSlotGain + TaskEntityMatch - RiskPenalty"
                    if self.light_explore_upper_bound_evidence
                    else (
                        "2.0*MissingSlotGain + 1.5*TaskEntityMatch + 1.0*OperatorPriority + "
                        "1.0*TaskProgress + 0.5*Novelty - 2.0*RiskPenalty - 1.0*RollbackCost - "
                        "1.0*WrongScreenRolePenalty - 0.5*RevisitPenalty"
                    )
                )
            ),
            "pattern_aware_operator_best_first": bool(
                getattr(self, "light_explore_pattern_aware_operator_best_first", False)
            ),
            "lb_mcts": bool(getattr(self, "light_explore_lb_mcts", False)),
            "lb_mcts_paper_name": "Latency-Bounded Task-Aware MCTS Exploration with Pattern-Aware Speculative UI Probing",
            "lb_mcts_chinese_name": "时延约束的任务感知 MCTS 探索",
            "lb_mcts_c_puct": float(getattr(self, "lb_mcts_c_puct", 1.4)),
            "lb_mcts_lambda_risk": float(getattr(self, "lb_mcts_lambda_risk", 1.0)),
            "lb_mcts_lambda_latency": float(getattr(self, "lb_mcts_lambda_latency", 0.5)),
            "lb_mcts_min_budget_ms": float(getattr(self, "lb_mcts_min_budget_ms", 3000.0)),
            "lb_mcts_max_budget_ms": float(getattr(self, "lb_mcts_max_budget_ms", 12000.0)),
            "lb_mcts_slack_ratio": float(getattr(self, "lb_mcts_slack_ratio", 0.8)),
            "max_step_limit": int(os.environ.get("ANDROID_WORLD_MAX_STEPS") or os.environ.get("ANDROID_WORLD_MAX_N_STEPS") or 0),
            "branch_budget": int(self.light_explore_branch_budget),
            "max_depth": int(self.light_explore_branch_depth),
            "min_attempts_per_step": int(getattr(self, "light_explore_min_attempts_per_step", 0)),
            "shadow_only": bool(getattr(self, "light_explore_shadow_only", False)),
            "relaxed_diagnostic_injection": bool(getattr(self, "light_explore_relaxed_diagnostic_injection", False)),
            "upper_bound_evidence": bool(getattr(self, "light_explore_upper_bound_evidence", False)),
            "summary_all_debug": bool(getattr(self, "light_explore_summary_all_debug", False)),
            "quality_filters": bool(getattr(self, "light_explore_quality_filters", True)),
            "safe_click_only": bool(self.light_explore_safe_click_only),
            "transaction_safe": bool(self.light_explore_transaction_safe),
            "lexical_rules": bool(self.light_explore_lexical_rules),
            "a11y_method": _clean_text(os.environ.get("ANDROID_WORLD_A11Y_METHOD", "")),
        }
        for base in self._diagnostic_targets(goal):
            try:
                os.makedirs(base, exist_ok=True)
                with open(os.path.join(base, "runtime_config.json"), "w", encoding="utf-8") as f:
                    f.write(json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n")
                with open(os.path.join(base, "runtime_config.yaml"), "w", encoding="utf-8") as f:
                    for key, value in config.items():
                        f.write(f"{key}: {json.dumps(value, ensure_ascii=False, default=str)}\n")
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def _tokenize_for_prior(self, text: str) -> list[str]:
        raw = _clean_text(text).lower()
        if not raw:
            return []
        tokens = re.findall(r"[a-z0-9._'-]+|\[[^\]]+\]|\d{1,4}-\d{1,2}-\d{1,2}|\b\d{1,2} [a-z]{3}\b", raw)
        out = []
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "for",
            "with",
            "on",
            "in",
            "from",
            "file",
            "files",
            "note",
            "notes",
            "task",
            "tasks",
            "app",
            "application",
            "status",
            "bar",
            "container",
        }
        for token in tokens:
            token = _clean_text(token).lower()
            if not token or token in stopwords:
                continue
            if re.fullmatch(r"[._-]+", token):
                continue
            out.append(token)
        return out

    def _extract_reasoning_prior_block(self, raw_text: str) -> dict[str, str]:
        block = {"keywords": "", "expected_ui": "", "template_answer": "", "template_action": "", "template_schema": "", "template_avoid": "", "template_risk": ""}
        normalized = _clean_text(raw_text)
        if not normalized:
            return block
        match = re.search(r"<exploration_prior>(.*?)</exploration_prior>", normalized, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return block
        body = match.group(1)
        for key in block:
            pattern = rf"{re.escape(key)}\s*:\s*(.*?)(?:\n\w|$)"
            found = re.search(pattern, body, flags=re.IGNORECASE | re.DOTALL)
            if found:
                value = found.group(1)
                block[key] = _clean_text(value).replace("\n", " ")
        return block

    def _derive_reasoning_prior_from_regex(
        self,
        goal: str,
        raw_vlm_output: str,
        parsed_action: dict[str, Any],
    ) -> dict[str, Any]:
        raw = _clean_text(raw_vlm_output)
        goal_summary = _clean_text(
            _clean_text(parsed_action.get("summary") or parsed_action.get("action") or goal)
        )
        next_intent = goal_summary
        quoted_patterns = re.findall(r"\"([^\"]{2,120})\"|\'([^\']{2,120})\'", raw)
        candidates = [item for pair in quoted_patterns for item in pair if item]
        target_keywords: list[str] = []
        for item in candidates:
            if item and item not in target_keywords:
                target_keywords.append(item)
        for item in self._extract_goal_entry_text(goal):
            if _clean_text(item) and item not in target_keywords:
                target_keywords.append(item)
        expected_ui_elements: list[str] = []
        expected_tokens = re.findall(
            r"\b(search|find|lookup|detail|stats|form|filter|list|result|file|note|recipe|event|folder|calendar)\b",
            raw.lower(),
        )
        for token in expected_tokens:
            token = token[0] if isinstance(token, tuple) else token
            if token and token not in expected_ui_elements:
                expected_ui_elements.append(token)
        derived_parameters: list[str] = []
        for text in (
            self._extract_goal_search_text(goal),
            self._extract_goal_entry_text(goal),
            parsed_action.get("value") or parsed_action.get("return"),
            parsed_action.get("action"),
        ):
            item = _clean_text(text)
            if item and item not in derived_parameters:
                derived_parameters.append(item)
        avoid_keywords: list[str] = [
            key
            for key in ("new event", "marker", "wrong page", "irrelevant", "do not")
            if key in raw.lower()
        ]
        risk_keywords: list[str] = [
            key
            for key in ("delete", "remove", "save", "confirm", "toggle", "permission", "share", "send")
            if key in raw.lower()
        ]
        risk_keywords = list(dict.fromkeys(risk_keywords))
        if not derived_parameters and target_keywords:
            derived_parameters = target_keywords[:1]
        return {
            "target_keywords": target_keywords,
            "expected_ui_elements": expected_ui_elements,
            "derived_parameters": derived_parameters,
            "avoid_keywords": avoid_keywords,
            "risk_keywords": risk_keywords,
            "goal_summary": goal_summary,
            "next_intent": next_intent,
            "expected_ui_role": "",
            "preferred_operators": self._infer_preferred_operators(raw, parsed_action),
            "evidence_template": {
                "template_answer": "",
                "template_action": "",
                "template_schema": "",
                "template_avoid": "",
                "template_risk": "",
            },
            "parse_method": "regex_fallback",
            "parse_success": True,
        }

    def _infer_preferred_operators(self, raw_text: str, parsed_action: dict[str, Any]) -> list[str]:
        text = (_clean_text(raw_text) + " " + _clean_text(parsed_action.get("explain") or parsed_action.get("cot") or "")).lower()
        operators: list[str] = []
        if any(token in text for token in ("search", "find", "lookup", "query")):
            operators.append("SearchPeek")
        if any(token in text for token in ("open", "click", "select", "result", "row")):
            operators.append("DetailPeek")
        if any(token in text for token in ("detail", "open detail", "open file", "open note")):
            operators.append("DetailPeek")
        if any(token in text for token in ("stats", "statistics", "duration", "distance")):
            operators.append("StatsPeek")
        if any(token in text for token in ("form", "field", "title", "name", "phone", "amount", "date")):
            operators.append("FormSchema")
        if any(token in text for token in ("avoid", "wrong", "not useful")):
            operators.append("AvoidPath")
        if any(token in text for token in ("delete", "confirm", "toggle", "send", "save", "permission")):
            operators.append("RiskBoundary")
        if not operators:
            operators = ["NavigationPeek"]
        # dedupe but keep order
        return [item for idx, item in enumerate(operators) if item not in operators[:idx]]

    def _build_reasoning_prior(self, goal: str, raw_vlm_output: str, parsed_action: dict[str, Any], previous_screen_summary: str) -> tuple[dict[str, Any], dict[str, Any]]:
        step_idx = len(self._actions) + 1
        prior_block = self._extract_reasoning_prior_block(raw_vlm_output)
        parse_success = False
        parse_method = "empty"
        if any(value for value in prior_block.values()):
            parse_success = True
            parse_method = "json_block"
            prior = {
                "task_id": _clean_text(goal)[:200],
                "source_step": int(step_idx),
                "raw_text": _clean_text(raw_vlm_output),
                "parse_method": parse_method,
                "parse_success": True,
                "goal_summary": _clean_text(
                    prior_block.get("goal_summary")
                    or parsed_action.get("summary")
                    or parsed_action.get("action")
                    or _clean_text(goal)
                ),
                "next_intent": _clean_text(prior_block.get("next_intent") or parsed_action.get("action") or goal),
                "target_keywords": [
                    item for item in self._tokenize_for_prior(prior_block.get("keywords") or "") if item
                ],
                "expected_ui_elements": [
                    item for item in self._tokenize_for_prior(prior_block.get("expected_ui") or "") if item
                ],
                "expected_ui_role": _clean_text(prior_block.get("expected_ui") or "unknown").lower() if prior_block.get("expected_ui") else "unknown",
                "preferred_operators": [
                    _clean_text(item) for item in self._infer_preferred_operators(
                        _clean_text(prior_block.get("keywords") + " " + prior_block.get("template_answer")),
                        parsed_action,
                    )
                ],
                "derived_parameters": [
                    item for item in [
                        self._extract_goal_entry_text(goal),
                        self._extract_goal_search_text(goal),
                        parsed_action.get("value"),
                    ] if _clean_text(item)
                ],
                "avoid_keywords": [
                    item for item in [
                        "wrong", "avoid", "irrelevant", "not relevant", "do not"
                    ] if item in (_clean_text(raw_vlm_output) + " " + _clean_text(parsed_action.get("explain", "")).lower())
                ],
                "risk_keywords": [
                    item
                    for item in ["delete", "confirm", "toggle", "permission", "save", "send"]
                    if item in _clean_text(raw_vlm_output).lower()
                ],
                "evidence_template": {
                    "template_answer": _clean_text(prior_block.get("template_answer")),
                    "template_action": _clean_text(prior_block.get("template_action")),
                    "template_schema": _clean_text(prior_block.get("template_schema")),
                    "template_avoid": _clean_text(prior_block.get("template_avoid")),
                    "template_risk": _clean_text(prior_block.get("template_risk")),
                },
                "confidence": 0.3,
                "parse_error": "",
            }
        else:
            prior = self._derive_reasoning_prior_from_regex(
                goal,
                raw_vlm_output,
                parsed_action,
            )
            prior["task_id"] = _clean_text(goal)[:200]
            prior["source_step"] = int(step_idx)
            prior["raw_text"] = _clean_text(raw_vlm_output)
            prior["parse_method"] = "regex_fallback"
            parse_method = "regex_fallback"
            prior["parse_error"] = ""
            prior_success = bool(prior.get("parse_success", False))
            prior["parse_success"] = bool(prior_success)
            prior["confidence"] = 0.3
            if prior["target_keywords"]:
                prior["confidence"] += 0.2
            if prior["derived_parameters"]:
                prior["confidence"] += 0.2
            if prior["expected_ui_elements"]:
                prior["confidence"] += 0.1
            prior["confidence"] = max(0.0, min(1.0, float(prior["confidence"])))
            prior["goal_summary"] = prior.get("goal_summary", "") or _clean_text(goal)
            prior["next_intent"] = prior.get("next_intent", _clean_text(goal))

        prior["goal_id"] = _clean_text(goal)[:200]
        prior["raw_text"] = _clean_text(prior.get("raw_text", ""))
        prior["expected_ui_role"] = _clean_text(prior.get("expected_ui_role") or "unknown")
        prior["evidence_template"] = prior.get("evidence_template") or {}
        prior["parse_success"] = bool(prior.get("parse_success"))
        prior["parse_method"] = parse_method

        if not parse_success:
            prior["parse_success"] = bool(_clean_text(raw_vlm_output) or _clean_text(previous_screen_summary))

        prior["parse_error"] = "" if prior.get("parse_success") else "prior_parse_empty"

        if previous_screen_summary:
            prior["goal_summary"] = prior.get("goal_summary") or previous_screen_summary
        # confidence rules
        confidence = 0.3
        txt = f"{prior.get('goal_summary','')} {prior.get('next_intent','')} {_clean_text(raw_vlm_output)}"
        if re.search(r"\b(open|click|select|find|search|navigate|detail|stats|form|schema|avoid)\b", txt.lower()):
            confidence += 0.2
        if prior.get("target_keywords"):
            confidence += 0.2
        if prior.get("derived_parameters"):
            confidence += 0.2
        if prior.get("expected_ui_elements"):
            confidence += 0.1
        prior["confidence"] = max(0.0, min(1.0, float(confidence)))

        template_record = {
            "task_id": _clean_text(goal)[:200],
            "step": int(step_idx),
            "source_prior_step": int(step_idx),
            "template_source": parse_method,
            "template_parse_success": bool(prior.get("parse_success")),
            "template_answer": _clean_text(prior.get("evidence_template", {}).get("template_answer") or ""),
            "template_action": _clean_text(prior.get("evidence_template", {}).get("template_action") or ""),
            "template_schema": _clean_text(prior.get("evidence_template", {}).get("template_schema") or ""),
            "template_avoid": _clean_text(prior.get("evidence_template", {}).get("template_avoid") or ""),
            "template_risk": _clean_text(prior.get("evidence_template", {}).get("template_risk") or ""),
        }
        return prior, template_record

    def _append_reasoning_prior_record(self, goal: str, prior: dict[str, Any]) -> None:
        if not getattr(self, "reasoning_prior_enabled", False):
            return
        row = {
            "task_id": _clean_text(goal)[:200],
            "source_step": int(prior.get("source_step") or 0),
            "raw_text": _clean_text(prior.get("raw_text") or ""),
            "parse_method": _clean_text(prior.get("parse_method") or "empty"),
            "parse_success": bool(prior.get("parse_success")),
            "goal_summary": _clean_text(prior.get("goal_summary") or ""),
            "next_intent": _clean_text(prior.get("next_intent") or ""),
            "target_keywords": list(prior.get("target_keywords") or []),
            "expected_ui_elements": list(prior.get("expected_ui_elements") or []),
            "expected_screen_role": _clean_text(prior.get("expected_ui_role") or prior.get("expected_screen_role") or ""),
            "preferred_operators": list(prior.get("preferred_operators") or []),
            "derived_parameters": list(prior.get("derived_parameters") or []),
            "avoid_keywords": list(prior.get("avoid_keywords") or []),
            "risk_keywords": list(prior.get("risk_keywords") or []),
            "evidence_template": prior.get("evidence_template") or {},
            "confidence": float(prior.get("confidence") or 0.0),
            "parse_error": _clean_text(prior.get("parse_error") or ""),
        }
        self._append_diagnostic_jsonl(goal, "reasoning_prior_records.jsonl", row)

    def _append_prompt_template_record(self, goal: str, template_record: dict[str, Any]) -> None:
        if not getattr(self, "reasoning_prior_template_fill", False):
            return
        row = {
            "task_id": _clean_text(goal)[:200],
            "step": int(template_record.get("step") or 0),
            "source_prior_step": int(template_record.get("source_prior_step") or 0),
            "template_source": _clean_text(template_record.get("template_source") or "fallback"),
            "template_parse_success": bool(template_record.get("template_parse_success")),
            "template_answer": _clean_text(template_record.get("template_answer") or ""),
            "template_action": _clean_text(template_record.get("template_action") or ""),
            "template_schema": _clean_text(template_record.get("template_schema") or ""),
            "template_avoid": _clean_text(template_record.get("template_avoid") or ""),
            "template_risk": _clean_text(template_record.get("template_risk") or ""),
        }
        self._append_diagnostic_jsonl(goal, "prompt_templates.jsonl", row)

    def _delayed_reasoning_prior_for_exploration(self, goal: str, step_idx: int) -> dict[str, Any]:
        if not getattr(self, "reasoning_prior_enabled", False):
            return {}
        if getattr(self, "reasoning_prior_delayed", True) and step_idx <= 0:
            return {}
        prior = self._reasoning_prior_for_next_step if isinstance(self._reasoning_prior_for_next_step, dict) else {}
        if not prior or not prior.get("parse_success"):
            return {}
        if _clean_text(prior.get("task_id") or prior.get("goal_id")) and _clean_text(prior.get("task_id") or prior.get("goal_id")) != _clean_text(goal)[:200]:
            return {}
        source_step = int(prior.get("source_step") or 0)
        current_one_based = int(step_idx + 1)
        if getattr(self, "reasoning_prior_delayed", True) and source_step >= current_one_based:
            return {}
        return dict(prior)

    def _prior_has_exact_search_keyword(self, prior: dict[str, Any], goal: str) -> bool:
        if not isinstance(prior, dict) or not prior.get("parse_success"):
            return False
        expected = _clean_text(self._extract_goal_search_text(goal)).lower()
        if not expected:
            entries = self._extract_goal_entry_text(goal)
            if isinstance(entries, (list, tuple)):
                expected = _clean_text(next((x for x in entries if _clean_text(x)), "")).lower()
            else:
                expected = _clean_text(entries).lower()
        if not expected:
            return False
        prior_values = [
            _clean_text(x).lower()
            for x in (
                list(prior.get("target_keywords") or [])
                + list(prior.get("derived_parameters") or [])
                + list(prior.get("expected_ui_elements") or [])
            )
            if _clean_text(x)
        ]
        preferred_ops = {_clean_text(x) for x in list(prior.get("preferred_operators") or []) if _clean_text(x)}
        has_search_operator = bool(preferred_ops & {"SearchPeek", "FilterPeek"})
        exact = any(expected == value or expected in value or value in expected for value in prior_values)
        return bool(has_search_operator and exact)

    def _protected_task_passive_only(self, goal: str, prior: dict[str, Any]) -> bool:
        task_mode = self._task_mode(goal)
        if task_mode not in {"FORM_CREATE_EDIT", "DELETE_COMMIT", "SIMPLE_VERIFY", "MEDIA_CAPTURE"}:
            return False
        if not _env_bool("ANDROID_WORLD_ADAPTIVE_ALLOW_RISKY_SEARCH_PROBES", False):
            return True
        return not self._prior_has_exact_search_keyword(prior, goal)

    def _goal_search_reference_text(self, goal: str) -> str:
        search_text = _clean_text(self._extract_goal_search_text(goal))
        if search_text:
            return search_text.lower()
        entry_text = self._extract_goal_entry_text(goal)
        return _clean_text(entry_text).lower()

    def _candidate_search_term_match(self, candidate: dict[str, Any], goal: str) -> bool:
        expected = _clean_text(self._goal_search_reference_text(goal)).lower()
        if not expected:
            return False
        candidate_text = _clean_text(candidate.get("text") or candidate.get("label") or candidate.get("merged") or "").lower()
        if not candidate_text:
            return False
        if expected == candidate_text:
            return True
        if expected in candidate_text or candidate_text in expected:
            return True
        expected_tokens = set(expected.split())
        candidate_tokens = set(candidate_text.split())
        if expected_tokens and candidate_tokens:
            overlap = len(expected_tokens.intersection(candidate_tokens))
            return overlap / float(len(expected_tokens) + 1e-9) >= 0.66
        return False

    def _is_search_only_candidate(self, candidate: dict[str, Any], goal: str, prior: dict[str, Any]) -> bool:
        if not isinstance(candidate, dict):
            return False
        operator = _clean_text(candidate.get("operator") or self._candidate_operator(candidate, goal)[0])
        if operator not in {"SearchPeek", "FilterPeek"}:
            return False
        if not self._prior_has_exact_search_keyword(prior, goal):
            return False
        action_kind = _clean_text(candidate.get("action_kind") or json_action.CLICK).lower()
        if action_kind in {"type", json_action.INPUT_TEXT}:
            return self._candidate_search_term_match(candidate, goal)
        return self._candidate_search_term_match(candidate, goal)

    def _apply_protected_task_candidate_policy(
        self,
        candidates: list[dict[str, Any]],
        goal: str,
        prior: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        task_mode = self._task_mode(goal)
        if task_mode not in {"FORM_CREATE_EDIT", "DELETE_COMMIT", "SIMPLE_VERIFY", "MEDIA_CAPTURE"}:
            return list(candidates), {
                "applied": False,
                "mode": "unprotected",
                "before_count": int(len(candidates)),
                "after_count": int(len(candidates)),
                "filtered_count": 0,
            }
        has_exact_search = bool(self._prior_has_exact_search_keyword(prior, goal))
        allowed_operators = {"SearchPeek", "FilterPeek", "FormSchema", "RiskBoundary"}
        if self._protected_task_passive_only(goal, prior):
            filtered_candidates = [
                candidate
                for candidate in candidates
                if (
                    _clean_text(candidate.get("operator")) in allowed_operators
                    and (
                        _clean_text(candidate.get("operator")) not in {"SearchPeek", "FilterPeek"}
                        or self._is_search_only_candidate(candidate, goal, prior)
                    )
                )
            ]
            return filtered_candidates, {
                "applied": True,
                "mode": "passive_only",
                "before_count": int(len(candidates)),
                "after_count": int(len(filtered_candidates)),
                "filtered_count": int(len(candidates) - len(filtered_candidates)),
            }
        filtered_candidates = []
        for candidate in candidates:
            candidate_operator = _clean_text(candidate.get("operator"))
            if candidate_operator not in allowed_operators:
                continue
            if candidate_operator in {"SearchPeek", "FilterPeek"} and not has_exact_search:
                continue
            if candidate_operator in {"SearchPeek", "FilterPeek"} and not self._is_search_only_candidate(candidate, goal, prior):
                continue
            filtered_candidates.append(candidate)
        return filtered_candidates, {
            "applied": True,
            "mode": "schema_risk_only" if not has_exact_search else "search_schema_risk_only",
            "before_count": int(len(candidates)),
            "after_count": int(len(filtered_candidates)),
            "filtered_count": int(len(candidates) - len(filtered_candidates)),
        }

    @staticmethod
    def _candidate_reasoning_signature_text(
        candidate: dict[str, Any],
        goal: str,
        current_screen_role: str,
        app_name: str,
    ) -> str:
        try:
            candidate_text = " ".join(
                str(x)
                for x in (
                    candidate.get("text"),
                    candidate.get("content_description"),
                    candidate.get("merged"),
                    candidate.get("label"),
                    candidate.get("resource_id") if not isinstance(candidate.get("resource_id"), dict) else None,
                    candidate.get("class_name"),
                    current_screen_role,
                    app_name,
                )
                if _clean_text(x)
            )
            a11y = candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {}
            sibling = a11y.get("sibling_labels") if isinstance(a11y.get("sibling_labels"), (list, tuple)) else []
            candidate_text = f"{candidate_text} {' '.join(_clean_text(x) for x in sibling if _clean_text(x))}"
            return _clean_text(candidate_text).lower()
        except Exception:  # pylint: disable=broad-exception-caught
            return _clean_text(candidate.get("label") or candidate.get("merged") or "").lower()

    def _evaluate_candidate_priors(
        self,
        candidates: list[dict[str, Any]],
        goal: str,
        reasoning_prior: dict[str, Any],
        current_screen_role: str,
        app_name: str,
        step_idx: int,
        trace_step: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], float, str, dict[str, Any]]:
        has_prior = bool(reasoning_prior and reasoning_prior.get("parse_success"))
        if not has_prior:
            return candidates, 0.0, "none", {}
        target = [
            _clean_text(x).lower()
            for x in list(reasoning_prior.get("target_keywords") or []) + list(reasoning_prior.get("derived_parameters") or [])
            if _clean_text(x)
        ]
        ui_expected = [
            _clean_text(x).lower()
            for x in list(reasoning_prior.get("expected_ui_elements") or [])
            if _clean_text(x)
        ]
        prior_ops = {
            _clean_text(op)
            for op in list(reasoning_prior.get("preferred_operators") or [])
            if _clean_text(op)
        }
        query = " ".join(
            _clean_text(v)
            for v in (
                _clean_text(reasoning_prior.get("goal_summary") or ""),
                _clean_text(reasoning_prior.get("next_intent") or ""),
                " ".join(target),
                " ".join(ui_expected),
            )
            if _clean_text(v)
        )
        query_tokens = set(self._tokenize_for_prior(query))
        if not query_tokens:
            return candidates, 0.0, "none", {}
        scored: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            candidate_text = self._candidate_reasoning_signature_text(
                candidate,
                goal=goal,
                current_screen_role=current_screen_role,
                app_name=app_name,
            )
            candidate_tokens = set(self._tokenize_for_prior(candidate_text))
            token_overlap_score = 0.0
            if candidate_tokens and query_tokens:
                token_overlap_score = len(candidate_tokens.intersection(query_tokens)) / float(len(query_tokens))
            exact_match_count = 0
            matched_keywords: list[str] = []
            matched_derived: list[str] = []
            for item in target:
                if item and item in candidate_text:
                    exact_match_count += 1
                    matched_keywords.append(item)
            for item in list(reasoning_prior.get("derived_parameters") or []):
                norm = _clean_text(item).lower()
                if norm and norm in candidate_text:
                    matched_derived.append(norm)
            operator = _clean_text(candidate.get("operator") or self._candidate_operator(candidate, goal)[0])
            safe = not self._is_transaction_unsafe_candidate(candidate) and not self._is_risky_probe_text(_clean_text(candidate.get("label") or ""))
            rollback_risk_level = float(
                min(
                    1.0,
                    self._candidate_rollback_risk(
                        candidate,
                        goal,
                        operator=operator,
                        label=_clean_text(candidate.get("label") or candidate.get("merged") or ""),
                    ),
                )
            )
            operator_match = operator in prior_ops
            s = float(exact_match_count) + token_overlap_score + (1.0 if operator_match else 0.0)
            # lower rollback risk preferred
            s = float(s) - 0.5 * float(rollback_risk_level)
            candidate_sc = {
                "reasoning_prior_score": float(s),
                "exact_match_count": int(exact_match_count),
                "matched_keywords": matched_keywords,
                "matched_derived_parameters": matched_derived,
                "candidate_text": candidate_text,
                "candidate_id": _clean_text(candidate.get("index") or candidate.get("key") or candidate.get("label") or ""),
                "operator": operator,
                "token_overlap_score": float(token_overlap_score),
                "operator_prior_match": bool(operator_match),
                "safe_to_execute": bool(safe),
                "rollback_risk_level": float(rollback_risk_level),
                "score": float(s),
            }
            candidate.update(candidate_sc)
            self._append_diagnostic_jsonl(
                self._current_goal_for_depth,
                "candidate_reasoning_prior_scores.jsonl",
                {
                    "task_id": _clean_text(goal)[:200],
                    "step": int(step_idx),
                    "prior_source_step": int(reasoning_prior.get("source_step") or 0),
                    "candidate_id": _clean_text(candidate_sc["candidate_id"]),
                    "label": _clean_text(candidate.get("label") or candidate.get("merged")),
                    "candidate_text": candidate_text,
                    "operator": operator,
                    "exact_match_count": int(exact_match_count),
                    "matched_keywords": matched_keywords,
                    "matched_derived_parameters": matched_derived,
                    "token_overlap_score": float(token_overlap_score),
                    "operator_prior_match": bool(operator_match),
                    "safe_to_execute": bool(safe),
                    "rollback_risk_level": float(rollback_risk_level),
                    "selected": False,
                    "selected_rank": 0,
                    "rejected_reason": "",
                },
            )
            scored.append((s, candidate))
        if not scored:
            return candidates, 0.0, "none", {}
        # sort and rank
        scored.sort(key=lambda item: (
            float(item[0]),
            float(item[1].get("exact_match_count") or 0),
            float(item[1].get("token_overlap_score") or 0),
            1 if item[1].get("operator_prior_match") else 0,
            -float(item[1].get("rollback_risk_level") or 0),
            -int(self._explored_element_visits.get(_clean_text(item[1].get("key") or item[1].get("index") or ""), 0)),
        ), reverse=True)
        ordered = [row[1] for row in scored]
        for idx, cand in enumerate(ordered, start=1):
            cand["reason_selected"] = cand.get("reason_selected") or "reasoning_prior_guided"
            row = {
                "task_id": _clean_text(goal)[:200],
                "step": int(step_idx),
                "prior_source_step": int(reasoning_prior.get("source_step") or 0),
                "candidate_id": _clean_text(cand.get("index") or cand.get("key") or cand.get("label") or ""),
                "label": _clean_text(cand.get("label") or cand.get("merged")),
                "candidate_text": _clean_text(cand.get("candidate_text") or ""),
                "operator": _clean_text(cand.get("operator") or ""),
                "exact_match_count": int(cand.get("exact_match_count") or 0),
                "matched_keywords": list(cand.get("matched_keywords") or []),
                "matched_derived_parameters": list(cand.get("matched_derived_parameters") or []),
                "token_overlap_score": float(cand.get("token_overlap_score") or 0.0),
                "operator_prior_match": bool(cand.get("operator_prior_match")),
                "safe_to_execute": bool(cand.get("safe_to_execute")),
                "rollback_risk_level": float(cand.get("rollback_risk_level") or 0.0),
                "selected": False,
                "selected_rank": int(idx),
                "rejected_reason": "",
            }
            self._append_diagnostic_jsonl(self._current_goal_for_depth, "candidate_reasoning_prior_scores.jsonl", row)

        mode = self._choose_reasoning_adaptive_mode(
            reasoning_prior=reasoning_prior,
            candidates=ordered,
            step_idx=step_idx,
        )
        return ordered, mode.get("prior_entropy", 0.0), mode.get("mode", "broad_shallow"), mode

    def _choose_reasoning_adaptive_mode(
        self,
        reasoning_prior: dict[str, Any],
        candidates: list[dict[str, Any]],
        step_idx: int,
    ) -> dict[str, Any]:
        top_scores = [
            float(c.get("score") or 0.0)
            for c in candidates
            if isinstance(c, dict)
        ]
        if not top_scores:
            return {
                "mode": "broad_shallow",
                "prior_entropy": 0.0,
                "root_budget": int(getattr(self, "light_explore_branch_budget", 12)),
                "depth2_required_for_top1": False,
                "depth2_policy": "semantic_changed_only",
                "latency_budget_ms": float(max(2000.0, 0.8 * self._reasoning_prior_ewma_ms)),
            }
        temp = max(0.1, min(2.0, float(self._reasoning_prior_ewma_ms or 1.0) / 12000.0 * 1.0))
        exp_scores = [math.exp(max(-20.0, float(v) / temp)) for v in top_scores]
        norm = sum(exp_scores) or 1.0
        probs = [float(v) / norm for v in exp_scores]
        entropy = float(-sum(p * math.log(max(1e-9, p)) for p in probs))
        confidence = float(reasoning_prior.get("confidence") or 0.0)
        mode = "broad_shallow"
        root_budget = int(getattr(self, "light_explore_branch_budget", 12))
        depth2_req = False
        depth2_policy = "semantic_changed_only"
        if confidence >= 0.7 and entropy <= 0.6:
            mode = "focused_depth"
            root_budget = 2
            depth2_req = True
            depth2_policy = "focused_depth"
        elif confidence >= 0.4:
            mode = "balanced"
            root_budget = 4
            depth2_policy = "semantic_changed_only"
        else:
            mode = "broad_shallow"
            root_budget = max(6, int(getattr(self, "light_explore_branch_budget", 12)))
            depth2_policy = "disabled_except_search"
        budget_ms = max(2000.0, min(12000.0, 0.8 * float(self._reasoning_prior_ewma_ms or 10000.0)))
        return {
            "mode": mode,
            "prior_entropy": float(entropy),
            "root_budget": int(root_budget),
            "depth2_required_for_top1": bool(depth2_req),
            "depth2_policy": depth2_policy,
            "latency_budget_ms": float(budget_ms),
            "candidates": list(candidates[:root_budget]),
        }

    def _append_adaptive_search_record(self, goal: str, record: dict[str, Any]) -> None:
        if not getattr(self, "reasoning_prior_enabled", False):
            return
        self._append_diagnostic_jsonl(goal, "adaptive_search_records.jsonl", record)

    @staticmethod
    def _safe_action_signature_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
        action = ExplorerElementAgent._probe_action_from_candidate(candidate)
        a11y = candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {}
        bbox = a11y.get("bbox") if isinstance(a11y.get("bbox"), dict) else {}
        signature = {
            "action_type": _clean_text(candidate.get("action_kind") or (action.action_type if action else "")),
            "normalized_label": _clean_text(candidate.get("label") or candidate.get("merged") or ""),
            "resource_id": _clean_text(a11y.get("resource_id") or candidate.get("resource_id") or ""),
            "typed_text": _clean_text(action.text if action else (candidate.get("text") or "")),
            "center": [int(action.x), int(action.y)] if action and action.x is not None and action.y is not None else [],
            "bbox": {
                "x_min": float(bbox.get("x_min", 0.0)),
                "y_min": float(bbox.get("y_min", 0.0)),
                "x_max": float(bbox.get("x_max", 0.0)),
                "y_max": float(bbox.get("y_max", 0.0)),
            } if isinstance(bbox, dict) else {},
        }
        return signature

    @staticmethod
    def _safe_signature_match_reason(candidate_sig: dict[str, Any], action: json_action.JSONAction) -> float:
        if not isinstance(action, json_action.JSONAction):
            return 0.0
        if _clean_text(candidate_sig.get("action_type") or "") == json_action.CLICK:
            if action.action_type == json_action.CLICK and int(action.x or 0) and int(action.y or 0):
                center = candidate_sig.get("center")
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    try:
                        dist = math.dist([float(center[0]), float(center[1])], [float(action.x), float(action.y)])
                    except Exception:  # pylint: disable=broad-exception-caught
                        dist = 1e9
                    if dist <= 120.0:
                        return 1.0
                    if dist <= 220.0:
                        return 0.8
                    return 0.4
                return 0.4
        if _clean_text(candidate_sig.get("action_type") or "") == json_action.OPEN_APP:
            cand_app = _clean_text(candidate_sig.get("typed_text") or candidate_sig.get("normalized_label"))
            real_app = _clean_text(getattr(action, "app_name", ""))
            if cand_app and real_app and (cand_app in real_app or real_app in cand_app):
                return 1.0
        if _clean_text(candidate_sig.get("action_type") or "") in {json_action.INPUT_TEXT, "type"}:
            cand_text = _clean_text(candidate_sig.get("typed_text") or "")
            real_text = _clean_text(getattr(action, "text", ""))
            if cand_text and real_text and cand_text == real_text:
                return 1.0
            if cand_text and real_text and cand_text in real_text:
                return 0.8
        return 0.0

    def _promote_signature_match_reason(self, target_signature: dict[str, Any], state: Any, current_activity: str) -> tuple[float, str]:
        if not isinstance(target_signature, dict):
            return 0.0, "invalid_signature"
        current_activity = _clean_text(current_activity or "")
        target_activity = _clean_text(target_signature.get("activity") or "")
        target_role = _clean_text(target_signature.get("role") or "")
        target_labels = [
            _clean_text(item).lower()
            for item in list(target_signature.get("anchor_labels") or [])
            if _clean_text(item)
        ]
        try:
            current_labels = [
                _clean_text(item).lower()
                for item in list(self._state_semantic_summary(state, limit=80))
                if _clean_text(item)
            ]
        except Exception:  # pylint: disable=broad-exception-caught
            current_labels = []
        target_set = {_normalize for _normalize in target_labels}
        curr_set = {_normalize for _normalize in current_labels}
        overlap = len(target_set.intersection(curr_set)) / float(max(1, len(target_set) or 1))
        same_activity = bool(
            _clean_text(target_activity) and _clean_text(current_activity)
            and _clean_text(target_activity).lower() == _clean_text(current_activity).lower()
        )
        current_screen_role = _clean_text(self._screen_role_from_state(state))
        if same_activity and overlap >= 0.60:
            return float(overlap), f"activity_match:overlap={overlap:.3f}"
        if target_role and current_screen_role and target_role == current_screen_role and overlap >= 0.45:
            return float(overlap), f"role_match:overlap={overlap:.3f}"
        return 0.0, f"no_match:activity={bool(same_activity)} overlap={overlap:.3f}"

    def _build_promotion_record(
        self,
        goal: str,
        step_idx: int,
        prior_source_step: int,
        branch_observation: dict[str, Any],
        root_match_score: float,
        state_match_score: float,
        state_match_reason: str,
        depth2_entropy: float,
        state_match: bool,
    ) -> dict[str, Any]:
        return {
            "task_id": _clean_text(goal)[:200],
            "step": int(step_idx + 1),
            "prior_source_step": int(prior_source_step),
            "branch_id": branch_observation.get("branch_id"),
            "root_action_signature": dict(branch_observation.get("root_action_signature") or {}),
            "depth1_state_signature": dict(branch_observation.get("depth1_state_signature") or {}),
            "depth2_action_signature": dict(branch_observation.get("depth2_action_signature") or {}),
            "depth2_state_signature": dict(branch_observation.get("depth2_state_signature") or {}),
            "depth1_candidate": dict(branch_observation.get("depth1_candidate") or {}),
            "depth2_candidate": dict(branch_observation.get("depth2_candidate") or {}),
            "root_action_match_score": float(root_match_score),
            "state_match_score": float(state_match_score),
            "state_match_reason": state_match_reason,
            "state_matched": bool(state_match),
            "depth2_action_locatable": bool(branch_observation.get("depth2_action_locatable") or False),
            "promotable": bool(branch_observation.get("promotable_for_promotion") or False),
            "stored": bool(state_match and bool(branch_observation.get("promotable_for_promotion") or False)),
            "stored_reason": "promotion_candidate" if bool(state_match and branch_observation.get("promotable_for_promotion")) else state_match_reason,
            "entropy": float(depth2_entropy),
        }

    def _append_pending_promotion_record(self, goal: str, record: dict[str, Any]) -> None:
        if not getattr(self, "reasoning_prior_promotion", False):
            return
        self._pending_promotion_events.append(dict(record))
        self._all_pending_promotion_events.append(dict(record))
        self._append_diagnostic_jsonl(goal, "pending_promotion_records.jsonl", record)

    def _append_promotion_event(self, goal: str, record: dict[str, Any]) -> None:
        self._append_diagnostic_jsonl(goal, "promotion_events.jsonl", record)

    def _state_signature_overlap(
        self,
        state: Any,
        signature: dict[str, Any],
    ) -> tuple[float, str, list[str], list[str], list[str]]:
        current_activity = _clean_text(signature.get("activity") or "")
        signature_anchors = [
            _clean_text(x).lower()
            for x in list(signature.get("anchor_labels") or [])
            if _clean_text(x)
        ]
        current_anchors = [
            _clean_text(x).lower()
            for x in list(signature.get("_current_anchors", []) or [])
            if _clean_text(x)
        ]
        if not signature_anchors:
            signature_anchors = [
                _clean_text(x).lower()
                for x in list(signature.get("anchor_labels", []))
                if _clean_text(x)
            ]
        if not current_anchors:
            try:
                current_anchors = [
                    _clean_text(x).lower() for x in self._state_semantic_summary(state, limit=80)
                ]
            except Exception:  # pylint: disable=broad-exception-caught
                current_anchors = []
        overlap_set = set(signature_anchors).intersection(current_anchors)
        overlap = float(len(overlap_set)) / float(max(1, len(set(signature_anchors))))
        reason_parts = [
            f"activity:{current_activity}",
            f"anchor_overlap:{overlap:.3f}",
        ]
        return overlap, current_activity, signature_anchors, current_anchors, reason_parts

    def _append_filled_exploration_block(self, goal: str, block: str, step_idx: int, prior: dict[str, Any], matches: list[dict[str, Any]]) -> None:
        if not block:
            return
        answer_count = 0
        action_count = 0
        avoid_count = 0
        schema_count = 0
        risk_count = 0
        for item in matches or []:
            hint_type = _clean_text(item.get("evidence_type") or item.get("hint_type") or "")
            if hint_type == "ANSWER_HINT":
                answer_count += 1
            elif hint_type == "ACTION_HINT":
                action_count += 1
            elif hint_type == "AVOID_HINT":
                avoid_count += 1
            elif hint_type == "SCHEMA_HINT":
                schema_count += 1
            elif hint_type == "RISK_HINT":
                risk_count += 1
        self._append_diagnostic_jsonl(
            goal,
            "filled_exploration_blocks.jsonl",
            {
                "task_id": _clean_text(goal)[:200],
                "step": int(step_idx),
                "prior_source_step": int(prior.get("source_step") or 0),
                "evidence_count": int(len(matches) or 0),
                "answer_count": int(answer_count),
                "action_count": int(action_count),
                "avoid_count": int(avoid_count),
                "schema_count": int(schema_count),
                "risk_count": int(risk_count),
                "rendered_block": block,
            },
        )

    def _append_depth_decision_record(self, goal: str, record: dict[str, Any]) -> None:
        self._append_diagnostic_jsonl(goal, "depth_decision_records.jsonl", record)

    def _append_evidence_decision_record(self, goal: str, record: dict[str, Any]) -> None:
        self._append_diagnostic_jsonl(goal, "evidence_decisions.jsonl", record)

    def _append_hint_record_with_template(self, goal: str, record: dict[str, Any]) -> None:
        self._append_diagnostic_jsonl(goal, "prompt_hints.jsonl", record)

    @staticmethod
    def _avoid_target_from_reason(reason: str) -> str:
        text = _clean_text(reason)
        if not text:
            return ""
        quoted = re.search(r"[\"'`](.{1,80}?)[\"'`]", text)
        if quoted:
            return _clean_text(quoted.group(1))
        avoid = re.search(r"\bavoid\s+(.{1,80}?)(?:[:.;]| because |$)", text, flags=re.IGNORECASE)
        if avoid:
            return _clean_text(avoid.group(1)).strip("\"'` ")
        return ""

    def _build_reasoning_template_render(self, hint_type: str, item: dict[str, Any], prior: dict[str, Any]) -> str:
        fallback_map = {
            "ANSWER_HINT": "Exploration found: {fact}",
            "ACTION_HINT": "A safe next action may be: {action}",
            "SCHEMA_HINT": "The explored page exposes {schema_fields}",
            "AVOID_HINT": "Avoid {target}: {reason}",
            "RISK_HINT": "Do not speculatively execute {target}: {reason}",
        }
        template_key_map = {
            "ANSWER_HINT": "template_answer",
            "ACTION_HINT": "template_action",
            "SCHEMA_HINT": "template_schema",
            "AVOID_HINT": "template_avoid",
            "RISK_HINT": "template_risk",
        }
        template_key = template_key_map.get(hint_type, "")
        template = _clean_text((prior.get("evidence_template") or {}).get(template_key))
        template = template or fallback_map.get(hint_type, "")
        fact = _clean_text(item.get("fact") or _clean_text(item.get("observed") or ""))
        action = _clean_text(item.get("suggested_action") or _clean_text(item.get("candidate_label") or item.get("matched_prefix") or ""))
        schema_fields = "; ".join(_clean_text(x) for x in (item.get("observed_elements") or []) if _clean_text(x))
        reason = _clean_text(item.get("avoid_reason") or "unknown")
        target = _clean_text(item.get("target") or item.get("next_label") or item.get("label") or "")
        screen = _clean_text(item.get("screen") or "")
        confidence = f"{float(item.get('confidence') or 0.0):.2f}"
        rendered = template.replace("{fact}", fact).replace("{action}", action).replace("{schema_fields}", schema_fields).replace(
            "{reason}", reason
        ).replace("{target}", target).replace("{screen}", screen).replace("{confidence}", confidence)
        return rendered if rendered else template

    def _candidate_reasoning_prior_record(self, candidate: dict[str, Any], selected_rank: int, rejected_reason: str = "") -> dict[str, Any]:
        return {
            "candidate_id": _clean_text(candidate.get("index") or candidate.get("key") or candidate.get("label") or ""),
            "label": _clean_text(candidate.get("label") or candidate.get("merged") or ""),
            "operator": _clean_text(candidate.get("operator") or ""),
            "selected_rank": int(selected_rank),
            "rejected_reason": _clean_text(rejected_reason),
        }

    def _append_candidate_prior_score_row(self, goal: str, row: dict[str, Any]) -> None:
        if not getattr(self, "reasoning_prior_enabled", False):
            return
        payload = dict(row)
        payload.setdefault("task_id", _clean_text(goal)[:200])
        self._append_diagnostic_jsonl(goal, "candidate_reasoning_prior_scores.jsonl", payload)

    def _build_state_signature_from_probe(self, state: Any, step: int = 0) -> dict[str, Any]:
        activity = self._foreground_activity_name()
        anchors = self._informative_observed_elements(self._state_semantic_summary(state, limit=80))
        return {
            "activity": activity,
            "anchor_labels": anchors,
            "role": self._screen_role_from_state(state),
            "phash": self._state_hash(state),
            "_current_anchors": anchors,
        }

    def _candidate_signature_from_candidate(self, candidate: dict[str, Any], state: Any) -> dict[str, Any]:
        root_action = self._safe_action_signature_from_candidate(candidate)
        root_action["normalized_label"] = _clean_text(candidate.get("label") or candidate.get("merged") or "")
        root_action["resource_id"] = _clean_text((candidate.get("a11y") or {}).get("resource_id") or candidate.get("resource_id") or "")
        state_sig = self._build_state_signature_from_probe(state)
        return {
            "action_signature": root_action,
            "state_signature": state_sig,
            "safe": bool(not self._is_transaction_unsafe_candidate(candidate)),
            "candidate": dict(candidate),
        }

    def _candidate_root_match_score(self, root_sig: dict[str, Any], action: json_action.JSONAction) -> float:
        if not isinstance(root_sig, dict):
            return 0.0
        candidate_sig = root_sig.get("action_signature") if isinstance(root_sig.get("action_signature"), dict) else {}
        if not candidate_sig:
            return 0.0
        target_type = _clean_text(candidate_sig.get("action_type") or json_action.CLICK)
        action_type = _clean_text(action.action_type)
        if target_type and action_type and target_type != action_type:
            return 0.0
        if action.action_type == json_action.CLICK:
            cand_label = _clean_text(candidate_sig.get("normalized_label"))
            act_x = int(getattr(action, "x", -1) or -1)
            act_y = int(getattr(action, "y", -1) or -1)
            if cand_label and any(token in cand_label for token in (_clean_text(getattr(action, "summary", "")), _clean_text(getattr(action, "label", "")))):
                return 0.9
            center = candidate_sig.get("center")
            if isinstance(center, (list, tuple)) and len(center) >= 2 and act_x >= 0 and act_y >= 0:
                try:
                    dist = math.dist([float(center[0]), float(center[1])], [float(act_x), float(act_y)])
                except Exception:  # pylint: disable=broad-exception-caught
                    dist = 1e9
                if dist <= 120:
                    return 1.0
                if dist <= 220:
                    return 0.7
        elif action.action_type == json_action.INPUT_TEXT:
            typed = _clean_text(candidate_sig.get("typed_text") or "")
            act_text = _clean_text(getattr(action, "text", ""))
            if typed and act_text and typed == act_text:
                return 1.0
            if typed and act_text and typed in act_text:
                return 0.8
        elif action.action_type == json_action.OPEN_APP:
            app_name = _clean_text(getattr(action, "app_name", ""))
            cand_name = _clean_text(candidate_sig.get("typed_text") or candidate_sig.get("normalized_label"))
            if app_name and cand_name and (app_name in cand_name or cand_name in app_name):
                return 1.0
        # typed text exact by fallback
        if action.action_type != candidate_sig.get("action_type"):
            return 0.0
        return 0.4

    def _locate_candidate_action_in_state(self, state: Any, signature: dict[str, Any]) -> bool:
        if not isinstance(signature, dict):
            return False
        if state is None:
            return False
        action_type = _clean_text(signature.get("action_type") or "").lower()
        target_label = _clean_text(signature.get("normalized_label") or signature.get("typed_text") or "")
        target_resource = _clean_text(signature.get("resource_id") or "")
        if action_type == json_action.OPEN_APP:
            return bool(target_label)
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        for element in ui_elements:
            txt = _clean_text(getattr(element, "text", ""))
            content = _clean_text(getattr(element, "content_description", ""))
            rid = _clean_text(getattr(element, "resource_id", ""))
            if target_label and (target_label in txt or target_label in content):
                return True
            if target_resource and target_resource in rid:
                return True
        return False

    def _candidate_score_components(self, candidate: dict[str, Any] | None, goal: str = "") -> dict[str, float | str]:
        if not isinstance(candidate, dict):
            return {
                "MissingSlotGain": 0.0,
                "TaskEntityMatch": 0.0,
                "OperatorPriority": 0.0,
                "TaskProgress": 0.0,
                "Novelty": 1.0,
                "RiskPenalty": 1.0,
                "RollbackRisk": 1.0,
                "PatternPrior": 0.0,
                "RollbackCost": 1.0,
                "WrongScreenRolePenalty": 0.0,
                "RevisitPenalty": 0.0,
                "final_score": 0.0,
                "final_score_or_utility": 0.0,
                "operator": "Unknown",
                "operator_reason": "invalid_candidate",
                "quality_filter_reason": "invalid_candidate",
                "filtered_by_quality": True,
            }
        operator, operator_reason = self._candidate_operator(candidate, goal)
        task_mode = self._task_mode(goal)
        label = _clean_text(candidate.get("label") or candidate.get("merged")).lower()
        task_entities = self._task_entities(goal)
        quality_rejection = self._candidate_quality_filter_reason(candidate, goal)
        missing_slot_gain = min(1.0, float(candidate.get("estimated_evidence_gain") or candidate.get("score") or 0.0) / 5.0)
        task_entity_match = 1.0 if any(entity and entity in label for entity in task_entities) else 0.0
        operator_priority_map = {
            "INFO_QUERY": {
                "ListInspect": 1.0,
                "SearchPeek": 0.9,
                "FilterPeek": 0.8,
                "DetailPeek": 0.7,
                "StatsPeek": 0.7,
                "NavigationPeek": 0.45,
            },
            "RECIPE_INGREDIENT": {
                "SearchPeek": 1.0,
                "DetailPeek": 0.85,
                "ListInspect": 0.7,
                "NavigationPeek": 0.45,
            },
            "ACTIVITY_STATS": {
                "StatsPeek": 1.0,
                "DetailPeek": 0.95,
                "SearchPeek": 0.8,
                "FilterPeek": 0.75,
                "ListInspect": 0.65,
                "NavigationPeek": 0.45,
            },
            "EVENT_QUERY": {
                "SearchPeek": 1.0,
                "ListInspect": 0.85,
                "DetailPeek": 0.75,
                "NavigationPeek": 0.55,
            },
            "FORM_CREATE_EDIT": {"FormSchema": 1.0, "RiskBoundary": 0.2},
            "DELETE_COMMIT": {"RiskBoundary": 1.0, "ListInspect": 0.35},
            "SIMPLE_VERIFY": {"Other": 0.2, "NavigationPeek": 0.15},
        }
        operator_priority = float(operator_priority_map.get(task_mode, {}).get(operator, 0.5))
        if operator == "RiskBoundary":
            operator_priority = min(operator_priority, 0.25)
        task_progress = 1.0 if re.search(r"\b(search|filter|detail|stats|result|file|note|recipe|contact|event)\b", label) else 0.0
        novelty = 1.0 / float(int(candidate.get("visits") or 0) + 1)
        risk_penalty = 1.0 if operator == "RiskBoundary" or self._is_transaction_unsafe_candidate(candidate) else 0.0
        rollback_cost = 0.35 if operator in {"NavigationPeek", "DetailPeek", "StatsPeek"} else 0.15
        if operator in {"NavigationPeek", "DetailPeek", "StatsPeek"} and any(
            token in label for token in ("resolver", "browser", "chrome", "map", "route", "media")
        ):
            rollback_cost = max(rollback_cost, 0.7)
        wrong_screen_role_penalty = 1.0 if (
            task_mode in {"FORM_CREATE_EDIT", "DELETE_COMMIT"} and operator not in {"FormSchema", "RiskBoundary", "ListInspect"}
        ) else 0.0
        if task_mode == "EVENT_QUERY" and any(token in label for token in ("new event", "add event", "create event", "+")):
            wrong_screen_role_penalty = 1.0
            risk_penalty = max(risk_penalty, 0.8)
        if task_mode == "ACTIVITY_STATS" and "marker" in label:
            wrong_screen_role_penalty = 1.0
        if task_mode == "NAVIGATION_SEARCH" and any(token in label for token in ("audio", "videos", "images", "large files", "this week")):
            wrong_screen_role_penalty = max(wrong_screen_role_penalty, 0.7)
        if quality_rejection:
            wrong_screen_role_penalty = max(wrong_screen_role_penalty, 1.0)
        revisit_penalty = min(1.0, float(candidate.get("visits") or 0.0))
        pattern_prior = self._candidate_pattern_prior(candidate, goal, operator=operator, label=label)
        rollback_risk = self._candidate_rollback_risk(candidate, goal, operator=operator, label=label)
        if getattr(self, "light_explore_pattern_aware_operator_best_first", False):
            final_score = pattern_prior + missing_slot_gain + task_entity_match - risk_penalty - rollback_risk
        elif getattr(self, "light_explore_upper_bound_evidence", False):
            final_score = missing_slot_gain + task_entity_match - risk_penalty
        else:
            final_score = (
                2.0 * missing_slot_gain
                + 1.5 * task_entity_match
                + 1.0 * operator_priority
                + 1.0 * task_progress
                + 0.5 * novelty
                - 2.0 * risk_penalty
                - 1.0 * rollback_cost
                - 1.0 * wrong_screen_role_penalty
                - 0.5 * revisit_penalty
            )
        return {
            "MissingSlotGain": float(missing_slot_gain),
            "TaskEntityMatch": float(task_entity_match),
            "OperatorPriority": float(operator_priority),
            "TaskProgress": float(task_progress),
            "Novelty": float(novelty),
            "RiskPenalty": float(risk_penalty),
            "RollbackRisk": float(rollback_risk),
            "PatternPrior": float(pattern_prior),
            "RollbackCost": float(rollback_cost),
            "WrongScreenRolePenalty": float(wrong_screen_role_penalty),
            "RevisitPenalty": float(revisit_penalty),
            "final_score": float(final_score),
            "final_score_or_utility": float(final_score),
            "operator": operator,
            "operator_reason": operator_reason,
            "quality_filter_reason": quality_rejection,
            "filtered_by_quality": bool(quality_rejection),
        }

    def _candidate_pattern_prior(
        self,
        candidate: dict[str, Any],
        goal: str,
        *,
        operator: str = "",
        label: str = "",
    ) -> float:
        """Task-family/operator-level pattern prior inspired by PASTE."""
        operator = _clean_text(operator or candidate.get("operator") or "").strip()
        label = _clean_text(label or candidate.get("label") or candidate.get("merged") or "").lower()
        merged = _clean_text(candidate.get("merged") or label).lower()
        text = f"{label} {merged}".strip()
        task_mode = self._task_mode(goal)
        apps = set(self._goal_app_keywords(goal))
        entities = [e.lower() for e in self._task_entities(goal) if e]
        exact_entity = any(entity and entity in text for entity in entities)
        has_search = any(token in text for token in ("search", "find", "filter"))
        if operator in {"SearchPeek", "FilterPeek"} and has_search and (
            apps & {"joplin", "files", "calendar", "opentracks", "sportstracker", "sports"}
            or task_mode in {"RECIPE_INGREDIENT", "NAVIGATION_SEARCH", "EVENT_QUERY", "ACTIVITY_STATS"}
        ):
            return 1.0 if exact_entity or has_search else 0.5
        if operator in {"SearchPeek", "FormSchema"} and task_mode in {"RECIPE_INGREDIENT", "NAVIGATION_SEARCH", "EVENT_QUERY"}:
            return 0.75
        if operator == "ListInspect" and exact_entity:
            return 1.0
        if operator in {"DetailPeek", "StatsPeek"} and task_mode in {"ACTIVITY_STATS", "RECIPE_INGREDIENT", "INFO_QUERY"}:
            return 1.0 if exact_entity or operator == "StatsPeek" else 0.5
        if operator == "ListInspect" and task_mode in {"INFO_QUERY", "EVENT_QUERY", "ACTIVITY_STATS"}:
            return 0.5
        if task_mode == "EVENT_QUERY" and any(token in text for token in ("new event", "add event", "create event")):
            return 1.0
        if task_mode == "ACTIVITY_STATS" and any(token in text for token in ("markers", "marker")):
            return 1.0
        if operator == "RiskBoundary":
            return 0.0
        return 0.0

    def _candidate_rollback_risk(
        self,
        candidate: dict[str, Any],
        goal: str,
        *,
        operator: str = "",
        label: str = "",
    ) -> float:
        operator = _clean_text(operator or candidate.get("operator") or "").strip()
        label = _clean_text(label or candidate.get("label") or candidate.get("merged") or "").lower()
        merged = _clean_text(candidate.get("merged") or label).lower()
        text = f"{label} {merged}".strip()
        apps = set(self._goal_app_keywords(goal))
        task_mode = self._task_mode(goal)
        if operator == "RiskBoundary" or self._is_transaction_unsafe_candidate(candidate):
            return 1.0
        if any(token in text for token in ("delete", "save", "confirm", "toggle", "remove", "discard")):
            return 1.0
        fragile_app = bool(
            apps & {"joplin", "calendar", "opentracks", "sportstracker", "sports", "files", "browser"}
            or task_mode in {"RECIPE_INGREDIENT", "EVENT_QUERY", "ACTIVITY_STATS", "NAVIGATION_SEARCH"}
        )
        if fragile_app and operator in {"DetailPeek", "StatsPeek", "NavigationPeek"}:
            return 1.0 if any(token in text for token in ("resolver", "chrome", "browser", "open with", "map")) else 0.5
        if fragile_app and operator in {"SearchPeek", "FilterPeek", "ListInspect"}:
            return 0.25
        return 0.0

    def _candidate_quality_filter_reason(self, candidate: dict[str, Any], goal: str) -> str:
        if not getattr(self, "light_explore_quality_filters", True):
            return ""
        label = _clean_text(candidate.get("label") or candidate.get("merged") or "").lower()
        merged = _clean_text(candidate.get("merged") or label).lower()
        text = f"{label} {merged}".strip()
        center = candidate.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            try:
                if float(center[0]) < 0 or float(center[1]) < 0:
                    return "invalid_or_negative_center"
            except (TypeError, ValueError):
                return "invalid_or_negative_center"
        a11y = candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {}
        resource_id = _clean_text(a11y.get("resource_id") or "").lower()
        class_name = _clean_text(a11y.get("class_name") or "").lower()
        bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), dict) else a11y.get("bbox")
        if isinstance(bbox, dict):
            try:
                if float(bbox.get("x_max", 0)) <= float(bbox.get("x_min", 0)) or float(bbox.get("y_max", 0)) <= float(bbox.get("y_min", 0)):
                    return "invalid_bbox"
            except (TypeError, ValueError):
                return "invalid_bbox"
        if not getattr(self, "light_explore_lexical_rules", False):
            return ""
        generic = {
            "content",
            "container",
            "status bar",
            "action bar root",
            "drawer layout",
            "coordinator",
            "frame layout",
            "linear layout",
            "relative layout",
            "recycler view",
            "navigation bar",
            "root",
        }
        if text in generic or any(text == item for item in generic):
            return "generic_container"
        if any(token in text for token in ("status bar", "navigation bar", "action bar root")):
            return "system_ui"
        app_names = {
            "joplin",
            "chrome",
            "markor",
            "osmand",
            "simple",
            "simple calendar",
            "simple calendar pro",
            "opentracks",
            "open tracks",
            "clock",
            "files",
            "file manager",
        }
        compact_text = re.sub(r"[^a-z0-9]+", "", text)
        app_name_compacts = {re.sub(r"[^a-z0-9]+", "", item) for item in app_names}
        compact_is_repeated_app_name = any(
            app_name and compact_text and compact_text.replace(app_name, "") == ""
            for app_name in app_name_compacts
        )
        if (
            text in app_names
            or compact_text in app_name_compacts
            or compact_is_repeated_app_name
            or text.startswith("simple calendar")
            or text.startswith("opentracks")
        ) and "launcher" not in _clean_text(goal).lower():
            return "app_name_only"
        task_mode = self._task_mode(goal)
        try:
            task_subtype = slot_complete_evidence.parse_task_slots(goal).task_subtype
        except Exception:
            task_subtype = ""
        is_event_query = task_mode == "EVENT_QUERY" or task_subtype == "EVENT_QUERY"
        if is_event_query and text in {"simple", "simple calendar", "simple calendar pro"}:
            return "app_name_only"
        if is_event_query and (
            "calendar_fab" in resource_id
            or "new event" in text
            or "add event" in text
            or "create event" in text
        ):
            return "calendar_new_event"
        if is_event_query and (
            "top_left_arrow" in resource_id
            or "top_right_arrow" in resource_id
        ):
            return "calendar_date_nav_arrow"
        if is_event_query and (
            "month_view_background" in resource_id
            or "month view background" in text
            or (
                "month_view" in resource_id
                and "textview" not in class_name
                and not re.search(r"\b(search|event|meeting|emily|[0-9]{1,2}:[0-9]{2})\b", text)
            )
        ):
            return "calendar_empty_grid"
        if is_event_query and isinstance(bbox, dict):
            try:
                box_width = float(bbox.get("x_max", 0)) - float(bbox.get("x_min", 0))
                box_height = float(bbox.get("y_max", 0)) - float(bbox.get("y_min", 0))
                if not resource_id and box_width >= 900 and box_height >= 1400:
                    return "generic_container"
            except (TypeError, ValueError):
                pass
        return ""

    def _append_prompt_hint_decision(
        self,
        goal: str,
        step_idx: int,
        hint_for_prompt: str,
        matched_prompt_results: list[dict[str, Any]],
        match_trace: dict[str, Any],
    ) -> None:
        if not match_trace:
            match_trace = {}
        selected_ids = {id(item) for item in list(matched_prompt_results or [])}
        # Per-hint records for traceability in phase-A style analysis.
        for idx, item in enumerate(list(matched_prompt_results) or []):
            candidate = item.get("next_candidate") if isinstance(item, dict) else {}
            candidate = candidate if isinstance(candidate, dict) else {}
            hint_id = f"hint_{int(step_idx + 1)}_{idx + 1}"
            item["hint_id"] = hint_id
            hint_type = _clean_text(item.get("hint_type") or item.get("evidence_type") or "EVIDENCE")
            injected = bool(hint_for_prompt and id(item) in selected_ids)
            threshold = float(item.get("threshold") or self._upper_bound_threshold(hint_type, bool(getattr(self, "light_explore_summary_all_debug", False))))
            rendered_text = _clean_text(item.get("rendered_prompt_text") or item.get("prompt_line") or item.get("path") or item.get("next_label") or item.get("matched_prefix") or "")[:1200]
            record = {
                "record_type": "PromptHintRecord",
                "step": int(step_idx + 1),
                "task_id": _clean_text(goal)[:200],
                "app": ",".join(self._goal_app_keywords(goal)[:3]),
                "task_mode": self._task_mode(goal),
                "source_exploration_step": int(item.get("source_step") or 0),
                "variant": self.light_explore_variant,
                "branch_id": item.get("branch_id"),
                "hint_id": hint_id,
                "hint_type": hint_type,
                "candidate_label": _clean_text(item.get("next_label") or candidate.get("label")),
                "operator": _clean_text(item.get("operator") or candidate.get("operator")),
                "injected": bool(hint_for_prompt and id(item) in selected_ids),
                "confidence": float(item.get("confidence") or 0.0),
                "threshold": threshold,
                "source_branch_id": item.get("branch_id"),
                "rendered_text": rendered_text,
                "rejected_reason": "" if injected else "not_selected_for_prompt",
                "state_match_score": float(item.get("state_match_score") or 0.0),
                "rollback_verified": bool(item.get("rollback_verified")),
                "slot_or_target_score": float(item.get("slot_or_target_score") or item.get("evidence_gain") or 0.0),
                "slot_complete": bool(item.get("answer_complete") or item.get("slot_complete")),
                "missing_slots": list(item.get("missing_slots") or []),
                "action_safe": bool(not self._is_transaction_unsafe_candidate(candidate)),
                "matched_by": _clean_text(item.get("matched_by") or ""),
                "match_reason": _clean_text(match_trace.get("status") or ""),
            }
            self._append_diagnostic_jsonl(goal, "prompt_hint_records.jsonl", record)
            self._append_diagnostic_jsonl(goal, "prompt_hints.jsonl", record)
            self._append_diagnostic_jsonl(
                goal,
                "evidence_decisions.jsonl",
                {
                    "record_type": "EvidenceDecision",
                    "step": int(step_idx + 1),
                    "source_exploration_step": int(item.get("source_step") or 0),
                    "branch_id": item.get("branch_id"),
                    "depth": item.get("evidence_depth") or item.get("depth_reached"),
                    "candidate_label": record["candidate_label"],
                    "operator": record["operator"],
                    "evidence_type_candidate": _clean_text(item.get("evidence_type") or hint_type),
                    "final_evidence_type": hint_type,
                    "MissingSlotGain": (item.get("score_components") or {}).get("MissingSlotGain") if isinstance(item.get("score_components"), dict) else None,
                    "TaskEntityMatch": (item.get("score_components") or {}).get("TaskEntityMatch") if isinstance(item.get("score_components"), dict) else None,
                    "RiskPenalty": (item.get("score_components") or {}).get("RiskPenalty") if isinstance(item.get("score_components"), dict) else None,
                    "final_score": item.get("score"),
                    "state_match_score": record["state_match_score"],
                    "slot_coverage": item.get("slot_coverage"),
                    "slot_complete": record["slot_complete"],
                    "missing_slots": record["missing_slots"],
                    "rollback_verified": record["rollback_verified"],
                    "target_visible": bool(item.get("target_visible")),
                    "action_safe": record["action_safe"],
                    "confidence": record["confidence"],
                    "threshold": record["threshold"],
                    "injected": injected,
                    "rejected_reason": "" if injected else "not_selected_for_prompt",
                    "rendered_prompt_text": rendered_text,
                },
            )
        rejected_samples = list(getattr(self, "_last_strict_not_injected_reasons", []) or [])
        for ridx, rejected in enumerate(rejected_samples, start=1):
            hint_type = _clean_text(rejected.get("evidence_type") or rejected.get("hint_type") or "NONE")
            rendered_text = _clean_text(rejected.get("rendered_text") or rejected.get("prompt_line") or "")
            rejected_reason = _clean_text(rejected.get("reason") or rejected.get("rejected_reason") or "strict_conditions_not_met")
            record = {
                "record_type": "PromptHintRecord",
                "step": int(step_idx + 1),
                "task_id": _clean_text(goal)[:200],
                "app": ",".join(self._goal_app_keywords(goal)[:3]),
                "task_mode": self._task_mode(goal),
                "variant": self.light_explore_variant,
                "hint_id": f"hint_{int(step_idx + 1)}_rejected_{ridx}",
                "hint_type": hint_type,
                "candidate_label": _clean_text(rejected.get("next_label") or rejected.get("candidate_label")),
                "operator": _clean_text(rejected.get("operator") or ""),
                "injected": False,
                "confidence": float(rejected.get("confidence") or 0.0),
                "threshold": float(rejected.get("threshold") or 0.0),
                "rendered_text": rendered_text,
                "rejected_reason": rejected_reason,
                "state_match_score": float(rejected.get("state_match_score") or 0.0),
                "rollback_verified": bool(rejected.get("rollback_verified")),
                "action_safe": bool(rejected.get("action_safe")),
            }
            self._append_diagnostic_jsonl(goal, "prompt_hints.jsonl", record)
            self._append_diagnostic_jsonl(
                goal,
                "evidence_decisions.jsonl",
                {
                    "record_type": "EvidenceDecision",
                    "step": int(step_idx + 1),
                    "source_exploration_step": rejected.get("source_step"),
                    "branch_id": rejected.get("branch_id"),
                    "candidate_label": record["candidate_label"],
                    "operator": record["operator"],
                    "evidence_type_candidate": hint_type,
                    "final_evidence_type": hint_type,
                    "state_match_score": record["state_match_score"],
                    "rollback_verified": record["rollback_verified"],
                    "action_safe": record["action_safe"],
                    "confidence": record["confidence"],
                    "threshold": record["threshold"],
                    "injected": False,
                    "rejected_reason": rejected_reason,
                    "rendered_prompt_text": rendered_text,
                },
            )
        if not matched_prompt_results:
            rejected_reason = (
                (match_trace or {}).get("no_hint_reason")
                or ("no_matched_evidence" if not int((match_trace or {}).get("matched_count") or 0) else "matched_evidence_not_promptable")
            )
            self._append_diagnostic_jsonl(
                goal,
                "prompt_hints.jsonl",
                {
                    "record_type": "PromptHintRecord",
                    "step": int(step_idx + 1),
                    "task_id": _clean_text(goal)[:200],
                    "app": ",".join(self._goal_app_keywords(goal)[:3]),
                    "task_mode": self._task_mode(goal),
                    "variant": self.light_explore_variant,
                    "hint_id": f"hint_{int(step_idx + 1)}_none",
                    "hint_type": "NONE",
                    "injected": False,
                    "confidence": 0.0,
                    "threshold": 0.0,
                    "rendered_text": "",
                    "rejected_reason": rejected_reason,
                    "state_match_score": 0.0,
                    "rollback_verified": False,
                    "slot_or_target_score": 0.0,
                    "action_safe": False,
                },
            )
        matched_count = len(matched_prompt_results or [])
        decision = {
            "record_type": "PromptHintDecision",
            "step": int(step_idx + 1),
            "pending_exploration_count": int((match_trace or {}).get("pending_trace_count") or 0),
            "matched_state_count": int((match_trace or {}).get("matched_count") or matched_count),
            "injected_hint_count": len(matched_prompt_results or []) if hint_for_prompt else 0,
            "rejected_hint_count": max(0, matched_count - (len(matched_prompt_results or []) if hint_for_prompt else 0)),
            "injected": bool(hint_for_prompt),
            "rendered_prompt_text": hint_for_prompt or "",
            "rejected_reason": "" if hint_for_prompt else (
                (match_trace or {}).get("no_hint_reason")
                or ("no_matched_evidence" if not matched_count else "matched_evidence_not_promptable")
            ),
            "matched_results": matched_prompt_results or [],
            "state_match_trace": match_trace or {},
        }
        self._append_diagnostic_jsonl(goal, "prompt_hint_decisions.jsonl", decision)

    def _write_prompt_trace_files(
        self,
        goal: str,
        step_idx: int,
        messages: list[dict[str, Any]],
        exploration_context: str,
        message_text: str,
        hint_for_prompt: str,
    ) -> tuple[str, str]:
        task_dir = self._task_output_dir(goal)
        if not task_dir:
            return "", ""
        variant = _clean_text(self.light_explore_variant or "default") or "default"
        safe_variant = re.sub(r"[^a-zA-Z0-9_.-]+", "_", variant).strip("._-") or "default"
        prompt_dir = os.path.join(task_dir, "prompts", safe_variant)
        full_path = os.path.join(prompt_dir, f"step_{step_idx + 1:03d}_full_prompt.txt")
        context_path = os.path.join(prompt_dir, f"step_{step_idx + 1:03d}_exploration_context.txt")
        if os.name == "nt" and max(len(os.path.abspath(full_path)), len(os.path.abspath(context_path))) >= 240:
            compact_variant = re.sub(r"[^a-zA-Z0-9]+", "", safe_variant)[:8] or "v"
            prompt_dir = os.path.join(task_dir, "p", compact_variant)
            full_path = os.path.join(prompt_dir, f"s{step_idx + 1:03d}_full.txt")
            context_path = os.path.join(prompt_dir, f"s{step_idx + 1:03d}_ctx.txt")
        os.makedirs(prompt_dir, exist_ok=True)
        prompt_snapshot = {
            "step": int(step_idx + 1),
            "goal": goal,
            "variant": self.light_explore_variant,
            "hint_for_prompt": _clean_text(hint_for_prompt),
            "message_text": _clean_text(message_text),
            "messages": messages,
        }
        context_payload = [
            "Exploration context block:",
            _clean_text(exploration_context),
            "",
            "Prompt snapshot:",
            json.dumps(prompt_snapshot, ensure_ascii=False, indent=2, default=str),
        ]
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(prompt_snapshot, ensure_ascii=False, indent=2, default=str))
        with open(context_path, "w", encoding="utf-8") as f:
            f.write("\n".join(context_payload))
        return full_path, context_path

    def _build_relaxed_diagnostic_prompt_hint(
        self,
        matched_prompt_results: list[dict[str, Any]],
        match_trace: dict[str, Any],
    ) -> str:
        if not getattr(self, "light_explore_relaxed_diagnostic_injection", False):
            return ""
        rows: list[str] = []
        source = list(matched_prompt_results or [])
        if not source:
            source = [
                item for item in list((match_trace or {}).get("matches") or [])
                if isinstance(item, dict) and bool(item.get("matched"))
            ]
        for item in source[:3]:
            if not isinstance(item, dict):
                continue
            rollback_verified = bool(item.get("rollback_verified", True))
            state_match = 1.0 if bool(item.get("matched")) else float(item.get("state_match_score") or 0.0)
            if not rollback_verified or state_match < 0.55:
                continue
            path = _clean_text(item.get("path") or item.get("matched_prefix") or item.get("next_label"))[:120]
            evidence_type = _clean_text(item.get("evidence_type") or item.get("boundary_type") or "EVIDENCE")[:40]
            elements = item.get("observed_elements") if isinstance(item.get("observed_elements"), list) else []
            observed = ", ".join(_clean_text(x)[:40] for x in elements[:5] if _clean_text(x))
            if not observed:
                observed = _clean_text(item.get("next_label") or item.get("matched_prefix"))[:120]
            if path or observed:
                rows.append(f"- [{evidence_type}] {path}: {observed}".strip())
        if not rows:
            return ""
        return "Exploration evidence from previous step (diagnostic relaxed, verify visually before acting):\n" + "\n".join(rows)

    def _evidence_type_threshold(
        self,
        goal: str,
        obs: dict[str, Any],
        candidate: dict[str, Any],
    ) -> tuple[str, float]:
        task_mode = self._task_mode(goal)
        label = _clean_text(candidate.get("label") or candidate.get("merged")).lower()
        operator = _clean_text(candidate.get("operator") or obs.get("operator"))
        source_type = _clean_text(obs.get("evidence_type") or obs.get("hint_type"))
        if source_type == "ANSWER_HINT":
            return "ANSWER_HINT", 0.65
        if source_type == "AVOID_HINT":
            return "AVOID_HINT", 0.55
        if source_type == "SCHEMA_HINT":
            return "SCHEMA_HINT", 0.35
        if source_type == "RISK_HINT":
            return "RISK_HINT", 0.60
        if operator == "RiskBoundary" or self._is_transaction_unsafe_candidate(candidate):
            return "RISK_HINT", 0.65
        if task_mode in {"RECIPE_INGREDIENT", "ACTIVITY_STATS", "EVENT_QUERY", "INFO_QUERY"}:
            if bool(obs.get("answer_complete")) or bool(obs.get("slot_complete")):
                return "ANSWER_HINT", 0.75
            if task_mode == "ACTIVITY_STATS" and "marker" in label:
                return "AVOID_HINT", 0.70
            if task_mode == "EVENT_QUERY" and any(token in label for token in ("new event", "add event", "create event")):
                return "AVOID_HINT", 0.70
        if task_mode in {"FORM_CREATE_EDIT"}:
            return "SCHEMA_HINT", 0.65
        if int(obs.get("depth_reached") or 0) >= 1 or bool(obs.get("target_visible")) or str(obs.get("boundary_type") or "").endswith("BOUNDARY"):
            return "ACTION_HINT", 0.35
        return "NONE", 1.0

    @staticmethod
    def _score_margin_normalized(score_components: dict[str, Any]) -> float:
        try:
            score = float(score_components.get("final_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return max(0.0, min(1.0, score / 5.0))

    @staticmethod
    def _float01(value: Any, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return float(default)

    def _evidence_confidence_components(
        self,
        obs: dict[str, Any],
        rollback_info: dict[str, Any],
        score_components: dict[str, Any],
    ) -> dict[str, float]:
        state_match = self._float01(obs.get("state_match_score"), 1.0 if bool(obs.get("state_matched")) else 0.0)
        slot_coverage = self._float01(obs.get("slot_coverage"), 1.0 if bool(obs.get("slot_complete") or obs.get("answer_complete")) else 0.0)
        rollback_verified = 1.0 if bool(rollback_info.get("success")) else 0.0
        target_or_answer = 1.0 if bool(obs.get("target_visible") or obs.get("answer_complete") or obs.get("slot_complete")) else 0.0
        score_margin = self._score_margin_normalized(score_components)
        operator = _clean_text(obs.get("operator") or score_components.get("operator"))
        evidence_type = _clean_text(obs.get("evidence_type") or obs.get("boundary_type"))
        safe_action_boundary_operators = {
            "SearchPeek",
            "FilterPeek",
            "DetailPeek",
            "StatsPeek",
            "ListInspect",
            "NavigationPeek",
            "FormSchema",
        }
        safe_action_boundary = (
            rollback_verified >= 1.0
            and int(obs.get("depth_reached") or 0) >= 1
            and operator in safe_action_boundary_operators
            and evidence_type
            in {
                "ACTION_HINT",
                "AVOID_HINT",
                "SHORTCUT_BOUNDARY",
                "TARGET_BOUNDARY",
                "CHOICE_BOUNDARY",
                "SCHEMA_BOUNDARY",
                "ANSWER_BOUNDARY",
                "NEGATIVE_BOUNDARY",
            }
        )
        if safe_action_boundary:
            state_match = max(state_match, 0.55)
            slot_coverage = max(slot_coverage, 0.65)
            target_or_answer = max(target_or_answer, 0.65)
            score_margin = max(score_margin, 0.50)
        if evidence_type in {"SCHEMA_HINT", "AVOID_HINT", "RISK_HINT"} and rollback_verified >= 1.0:
            state_match = max(state_match, 0.70)
            slot_coverage = max(slot_coverage, 0.60)
            target_or_answer = max(target_or_answer, 0.60)
        confidence = (
            0.30 * state_match
            + 0.20 * slot_coverage
            + 0.20 * rollback_verified
            + 0.15 * target_or_answer
            + 0.15 * score_margin
        )
        return {
            "StateMatch": float(state_match),
            "SlotCompleteness": float(slot_coverage),
            "RollbackVerified": float(rollback_verified),
            "TargetVisibleOrAnswerComplete": float(target_or_answer),
            "ScoreMarginNormalized": float(score_margin),
            "confidence": float(confidence),
        }

    def _evidence_rejection_reason(
        self,
        evidence_type: str,
        confidence: float,
        threshold: float,
        obs: dict[str, Any],
        rollback_info: dict[str, Any],
        candidate: dict[str, Any],
    ) -> str:
        if evidence_type == "NONE":
            return "evidence_only_but_no_injection_rule"
        if not bool(rollback_info.get("success")):
            return "rollback_failed"
        if confidence < threshold:
            return "low_confidence"
        if evidence_type == "ANSWER_HINT" and not bool(obs.get("answer_complete") or obs.get("slot_complete")):
            return "answer_fact_incomplete"
        operator = _clean_text(candidate.get("operator") or obs.get("operator"))
        safe_action_boundary_operators = {
            "SearchPeek",
            "FilterPeek",
            "DetailPeek",
            "StatsPeek",
            "ListInspect",
            "NavigationPeek",
            "FormSchema",
        }
        safe_action_boundary = (
            bool(rollback_info.get("success"))
            and int(obs.get("depth_reached") or 0) >= 1
            and operator in safe_action_boundary_operators
        )
        if evidence_type == "ACTION_HINT" and self._is_transaction_unsafe_candidate(candidate):
            return "action_not_safe"
        if evidence_type == "ACTION_HINT" and not bool(
            obs.get("target_visible") or int(obs.get("depth_reached") or 0) >= 1 or safe_action_boundary
        ):
            return "no_visible_target"
        if evidence_type == "SCHEMA_HINT":
            return ""
        if evidence_type in {"AVOID_HINT", "RISK_HINT"}:
            return ""
        if evidence_type == "ACTION_HINT":
            return ""
        if self._is_transaction_unsafe_candidate(candidate) and evidence_type != "RISK_HINT":
            return "action_not_safe"
        return "state_mismatch"

    @staticmethod
    def _infer_level2_trigger_reason(rollback: dict[str, Any]) -> str:
        if rollback.get("level2_trigger_reason"):
            return _clean_text(rollback.get("level2_trigger_reason"))
        if rollback.get("mode") in {"home_replay", "replay"} or int(rollback.get("replayed_actions") or 0) > 0:
            return "replay_required_after_home"
        if rollback.get("activity_match") is False:
            return "activity_mismatch_after_back"
        try:
            if float(rollback.get("phash_diff") or 0.0) > 14:
                return "phash_mismatch_after_back"
        except (TypeError, ValueError):
            pass
        try:
            anchor_jaccard = rollback.get("anchor_jaccard")
            if anchor_jaccard is not None and float(anchor_jaccard) < 0.5:
                return "anchor_jaccard_low_after_back"
        except (TypeError, ValueError):
            pass
        if int(rollback.get("back_presses") or 0) >= int(rollback.get("back_limit") or rollback.get("back_presses") or 0):
            return "back_limit_exhausted"
        return "unknown"

    def _emit_diagnostic_artifacts(self, goal: str, trace: dict[str, Any]) -> None:
        if not isinstance(trace, dict):
            return
        self._write_runtime_config_once(goal)
        selected = trace.get("selected_targets") or []
        all_candidates = trace.get("candidates") or trace.get("candidate_table") or selected
        risk_candidates = [
            c for c in all_candidates
            if isinstance(c, dict) and (
                c.get("operator") == "RiskBoundary"
                or c.get("risk_boundary")
                or self._is_transaction_unsafe_candidate(c)
            )
        ]
        safe_candidates = [
            c for c in all_candidates
            if isinstance(c, dict) and c not in risk_candidates
        ]
        attempt_count = int(trace.get("attempt_count") or (len(selected) + len(risk_candidates)))
        quality = trace.get("candidate_quality") if isinstance(trace.get("candidate_quality"), dict) else {}
        exploration_latency = {
            "step": trace.get("step"),
            "status": trace.get("status"),
            "strategy_name": "Selective Consumable Exploration",
            "root_level_strategy": "consume-or-promote gate",
            "branch_continuation": "depth2 only for consumable search/filter/result/detail/schema states",
            "ranking": "consumable evidence first",
            "exploration_total_ms": float(trace.get("latency_ms") or 0.0),
            "candidate_collection_ms": float(trace.get("candidate_collection_ms") or 0.0),
            "candidate_scoring_ms": float(trace.get("candidate_scoring_ms") or 0.0),
            "operator_classification_ms": float(trace.get("operator_classification_ms") or 0.0),
            "root_state_fetch_ms": float(trace.get("exploration_state_fetch_ms") or 0.0),
            "root_a11y_ms": float(trace.get("exploration_a11y_dump_ms") or 0.0),
            "root_screenshot_ms": float(trace.get("root_screenshot_ms") or 0.0),
            "a11y_method": _clean_text(os.environ.get("ANDROID_WORLD_A11Y_METHOD", "")),
            "candidate_count": int(trace.get("candidate_count") or len(all_candidates)),
            "candidate_count_raw": int(quality.get("candidate_count_raw") or trace.get("candidate_count_raw") or len(all_candidates)),
            "candidate_count_after_filter": int(quality.get("candidate_count_after_filter") or trace.get("candidate_count") or len(all_candidates)),
            "filtered_system_ui_count": int(quality.get("filtered_system_ui_count") or 0),
            "filtered_invalid_bbox_count": int(quality.get("filtered_invalid_bbox_count") or 0),
            "filtered_generic_container_count": int(quality.get("filtered_generic_container_count") or 0),
            "filtered_app_name_only_count": int(quality.get("filtered_app_name_only_count") or 0),
            "filtered_duplicate_count": int(quality.get("filtered_duplicate_count") or 0),
            "filtered_wrong_task_family_count": int(quality.get("filtered_wrong_task_family_count") or 0),
            "filtered_wrong_operator_count": int(quality.get("filtered_wrong_operator_count") or 0),
            "filtered_risky_count": int(quality.get("filtered_risky_count") or 0),
            "top_filtered_examples": list(quality.get("top_filtered_examples") or [])[:12],
            "selected_branch_count": len(selected),
            "attempt_count": int(attempt_count),
            "passive_attempt_count": int(trace.get("passive_attempt_count") or 0),
            "passive_padding_reason": trace.get("passive_padding_reason") or "",
            "min_attempts_per_step": int(getattr(self, "light_explore_min_attempts_per_step", 0)),
            "safe_executable_count": len(safe_candidates),
            "risk_candidate_count": len(risk_candidates),
            "no_candidate_reason": (
                "safe_candidate_shortage"
                if int(getattr(self, "light_explore_min_attempts_per_step", 0)) and attempt_count < int(getattr(self, "light_explore_min_attempts_per_step", 0))
                else ""
            ),
            "branch_latencies": [],
            "branches": [],
        }
        for obs in trace.get("observations") or []:
            if not isinstance(obs, dict):
                continue
            rb = obs.get("rollback") if isinstance(obs.get("rollback"), dict) else {}
            steps = [step for step in (obs.get("steps") or []) if isinstance(step, dict)]
            d1 = next((step for step in steps if int(step.get("depth") or 0) == 1), {})
            d2 = next((step for step in steps if int(step.get("depth") or 0) == 2), {})
            row = {
                "branch_id": obs.get("branch_id"),
                "candidate_label": _clean_text((obs.get("labels") or [""])[0] if isinstance(obs.get("labels"), list) else ""),
                "operator": obs.get("operator") or "",
                "final_score": float(obs.get("score") or 0.0),
                "depth_reached": obs.get("depth_reached"),
                "state_fetch_ms_by_depth": [float(step.get("state_fetch_ms") or 0.0) for step in steps],
                "a11y_ms_by_depth": [float(step.get("a11y_dump_ms") or 0.0) for step in steps],
                "screenshot_ms_by_depth": [float(step.get("screenshot_ms") or 0.0) for step in steps],
                "semantic_summary_ms_by_depth": [float(step.get("summary_ms") or 0.0) for step in steps],
                "depth1_action_ms": float(d1.get("action_ms") or 0.0),
                "depth1_state_fetch_ms": float(d1.get("state_fetch_ms") or 0.0),
                "depth1_a11y_ms": float(d1.get("a11y_dump_ms") or 0.0),
                "depth1_screenshot_ms": float(d1.get("screenshot_ms") or 0.0),
                "depth1_phash_ms": float(d1.get("phash_ms") or 0.0),
                "depth1_semantic_summary_ms": float(d1.get("summary_ms") or 0.0),
                "depth2_action_ms": float(d2.get("action_ms") or 0.0),
                "depth2_state_fetch_ms": float(d2.get("state_fetch_ms") or 0.0),
                "depth2_a11y_ms": float(d2.get("a11y_dump_ms") or 0.0),
                "depth2_screenshot_ms": float(d2.get("screenshot_ms") or 0.0),
                "depth2_phash_ms": float(d2.get("phash_ms") or 0.0),
                "depth2_semantic_summary_ms": float(d2.get("summary_ms") or 0.0),
                "evidence_assessment_ms": float(obs.get("evidence_assessment_ms") or 0.0),
                "rollback_total_ms": float(rb.get("latency_ms") or rb.get("rollback_total_ms") or 0.0),
                "rollback_verify_ms": float(rb.get("verify_ms") or rb.get("rollback_verify_ms") or 0.0),
                "rollback_action_ms": float(rb.get("rollback_action_ms") or 0.0),
                "rollback_a11y_ms": float(rb.get("verify_a11y_ms") or rb.get("rollback_a11y_ms") or 0.0),
            }
            exploration_latency["branch_latencies"].append(row)
            exploration_latency["branches"].append(row)
        self._append_diagnostic_jsonl(goal, "exploration_latency.jsonl", exploration_latency)
        self._append_diagnostic_jsonl(
            goal,
            "candidate_filter_stats.jsonl",
            {
                "step": trace.get("step"),
                "raw_candidate_count": int(quality.get("candidate_count_raw") or trace.get("candidate_count_raw") or len(all_candidates)),
                "after_filter_count": int(quality.get("candidate_count_after_filter") or trace.get("candidate_count") or len(all_candidates)),
                "filtered_candidate_count": max(
                    0,
                    int(quality.get("candidate_count_raw") or trace.get("candidate_count_raw") or len(all_candidates))
                    - int(quality.get("candidate_count_after_filter") or trace.get("candidate_count") or len(all_candidates)),
                ),
                "filtered_system_ui_count": int(quality.get("filtered_system_ui_count") or 0),
                "filtered_generic_container_count": int(quality.get("filtered_generic_container_count") or 0),
                "filtered_invalid_bbox_count": int(quality.get("filtered_invalid_bbox_count") or 0),
                "filtered_app_name_only_count": int(quality.get("filtered_app_name_only_count") or 0),
                "filtered_duplicate_count": int(quality.get("filtered_duplicate_count") or 0),
                "filtered_wrong_task_family_count": int(quality.get("filtered_wrong_task_family_count") or 0),
                "filtered_risky_count": int(quality.get("filtered_risky_count") or 0),
                "top_filtered_examples": list(quality.get("top_filtered_examples") or [])[:12],
            },
        )
        observations = [obs for obs in list(trace.get("observations") or []) if isinstance(obs, dict)]
        stop_distribution: dict[str, int] = {}
        for obs in observations:
            reason = _clean_text(obs.get("stop_reason") or obs.get("boundary_type") or "UNKNOWN")
            stop_distribution[reason] = stop_distribution.get(reason, 0) + 1
        executed_attempt_count = sum(
            1 for obs in observations
            if _clean_text((obs.get("rollback") or {}).get("level")) != "risk_boundary"
            and not bool(obs.get("no_action_inspect"))
            and not bool(obs.get("passive_coverage_padding"))
        )
        passive_attempt_count = sum(
            1 for obs in observations
            if bool(obs.get("no_action_inspect"))
            or bool(obs.get("passive_coverage_padding"))
        )
        risk_attempt_count = sum(
            1 for obs in observations
            if _clean_text(obs.get("operator")) == "RiskBoundary"
            or _clean_text((obs.get("rollback") or {}).get("level")) == "risk_boundary"
        )
        min_attempts = int(getattr(self, "light_explore_min_attempts_per_step", 0))
        attempted_count = int(trace.get("attempt_count") or len(observations))
        self._append_diagnostic_jsonl(
            goal,
            "exploration_step_summary.jsonl",
            {
                "step": trace.get("step"),
                "exploration_triggered": str(trace.get("status")) not in {"skipped", "not_started"},
                "trigger_reason": trace.get("trigger_reason") or trace.get("gate_reason") or "",
                "raw_candidate_count": int(quality.get("candidate_count_raw") or 0),
                "filtered_candidate_count": int(trace.get("candidate_count") or len(all_candidates)),
                "selected_candidate_count": len(selected),
                "attempted_count": attempted_count,
                "executed_attempt_count": int(executed_attempt_count),
                "passive_attempt_count": int(passive_attempt_count),
                "risk_classified_attempt_count": int(risk_attempt_count),
                "less_than_30_reason": (
                    trace.get("no_candidate_reason")
                    or trace.get("reason_less_than_30")
                    or ("safe_candidate_shortage_or_budget" if min_attempts and attempted_count < min_attempts else "")
                ),
                "less_than_12_reason": (
                    trace.get("no_candidate_reason")
                    or trace.get("reason_less_than_12")
                    or ("safe_candidate_shortage_or_budget" if min_attempts and attempted_count < min_attempts else "")
                ),
                "depth1_count": sum(1 for obs in observations if int(obs.get("depth_reached") or 0) >= 1),
                "depth2_count": sum(1 for obs in observations if int(obs.get("depth_reached") or 0) >= 2),
                "max_depth_reached": max([int(obs.get("depth_reached") or 0) for obs in observations] or [0]),
                "stop_reason_distribution": stop_distribution,
                "useful_evidence_count": sum(
                    1 for obs in observations
                    if _clean_text(obs.get("evidence_type")) in {"ANSWER_HINT", "ACTION_HINT", "AVOID_HINT", "SCHEMA_HINT", "RISK_HINT"}
                ),
                "injected_evidence_count": 0,
            },
        )
        for obs in observations:
            steps = [step for step in list(obs.get("steps") or []) if isinstance(step, dict)]
            d1 = next((step for step in steps if int(step.get("depth") or 0) == 1), {})
            d2 = next((step for step in steps if int(step.get("depth") or 0) == 2), {})
            rb = obs.get("rollback") if isinstance(obs.get("rollback"), dict) else {}
            if bool(getattr(self, "light_explore_lb_mcts", False)) and self.light_explore_search_strategy == "mcts":
                cand_for_reward = {}
                if d2 and isinstance(d2.get("candidate"), dict):
                    cand_for_reward = d2.get("candidate") or {}
                elif d1 and isinstance(d1.get("candidate"), dict):
                    cand_for_reward = d1.get("candidate") or {}
                components = cand_for_reward.get("score_components") if isinstance(cand_for_reward.get("score_components"), dict) else {}
                latency_cost_norm = float(cand_for_reward.get("mcts_latency_cost_norm") or 0.0)
                reward_components = {
                    "AnswerBoundary": 1.0 if _clean_text(obs.get("evidence_type")) == "ANSWER_HINT" else 0.0,
                    "ActionBoundary": 1.0 if _clean_text(obs.get("evidence_type")) == "ACTION_HINT" else 0.0,
                    "TargetEntityFound": float(components.get("TaskEntityMatch") or 0.0),
                    "SlotCompletionGain": float(obs.get("slot_coverage") or components.get("MissingSlotGain") or 0.0),
                    "UsefulSchema": 1.0 if _clean_text(obs.get("evidence_type")) == "SCHEMA_HINT" else 0.0,
                    "HardNegative": 1.0 if _clean_text(obs.get("evidence_type")) == "AVOID_HINT" else 0.0,
                    "RiskViolation": 1.0 if self._is_transaction_unsafe_candidate(cand_for_reward) else 0.0,
                    "RollbackFailure": 0.0 if bool(rb.get("success", True)) else 1.0,
                    "LowQualityObservation": 1.0 if not obs.get("observed_elements") else 0.0,
                    "LatencyCostNorm": latency_cost_norm,
                }
                reward_value = (
                    3.0 * reward_components["AnswerBoundary"]
                    + 2.0 * reward_components["ActionBoundary"]
                    + 2.0 * reward_components["TargetEntityFound"]
                    + 1.5 * reward_components["SlotCompletionGain"]
                    + 1.0 * reward_components["UsefulSchema"]
                    + 1.0 * reward_components["HardNegative"]
                    - 2.0 * reward_components["RiskViolation"]
                    - 2.0 * reward_components["RollbackFailure"]
                    - 1.0 * reward_components["LowQualityObservation"]
                    - 0.5 * reward_components["LatencyCostNorm"]
                )
                self._append_diagnostic_jsonl(
                    goal,
                    "mcts_reward_records.jsonl",
                    {
                        "record_type": "MCTSRewardRecord",
                        "task_id": _clean_text(goal)[:200],
                        "step": trace.get("step"),
                        "branch_id": obs.get("branch_id"),
                        **reward_components,
                        "reward": float(reward_value),
                    },
                )
            self._append_diagnostic_jsonl(
                goal,
                "exploration_branch_trace.jsonl",
                {
                    "step": trace.get("step"),
                    "branch_id": obs.get("branch_id"),
                    "root_candidate_label": _clean_text((obs.get("labels") or [""])[0] if isinstance(obs.get("labels"), list) else ""),
                    "root_operator": obs.get("operator"),
                    "root_score": obs.get("score"),
                    "selected_rank": obs.get("branch_id"),
                    "selected_reason": (
                        (d1.get("candidate") or {}).get("reason_selected")
                        if isinstance(d1.get("candidate"), dict)
                        else None
                    )
                    or obs.get("selected_reason")
                    or "operator_stratified_budget",
                    "depth_reached": obs.get("depth_reached"),
                    "depth1_action": (d1.get("candidate") or {}).get("label") if isinstance(d1.get("candidate"), dict) else "",
                    "depth1_state_summary": d1.get("observed_elements") or [],
                    "depth2_action": (d2.get("candidate") or {}).get("label") if isinstance(d2.get("candidate"), dict) else "",
                    "depth2_state_summary": d2.get("observed_elements") or [],
                    "semantic_changed": any(bool(step.get("semantic_changed")) for step in steps),
                    "semantic_change_reason": ";".join(_clean_text(step.get("semantic_change_reason")) for step in steps if step.get("semantic_change_reason")),
                    "stop_condition": obs.get("boundary_type"),
                    "stop_reason": obs.get("stop_reason"),
                    "evidence_type_candidate": obs.get("evidence_type"),
                    "evidence_gain": obs.get("evidence_gain"),
                    "rollback_success": bool(rb.get("success")),
                    "rollback_level": rb.get("level"),
                    "rollback_mode": rb.get("mode"),
                    "depth1_screenshot_path": d1.get("screenshot"),
                    "depth2_screenshot_path": d2.get("screenshot"),
                    "screenshot_paths": [step.get("screenshot") for step in steps if step.get("screenshot")],
                },
            )
            for page in steps:
                self._append_diagnostic_jsonl(
                    goal,
                    "exploration_page_trace.jsonl",
                    {
                        "step": trace.get("step"),
                        "branch_id": obs.get("branch_id"),
                        "depth": page.get("depth"),
                        "candidate": page.get("candidate"),
                        "activity": page.get("after_activity"),
                        "state_summary": page.get("observed_elements") or [],
                        "semantic_changed": bool(page.get("semantic_changed")),
                        "semantic_change_reason": page.get("semantic_change_reason"),
                        "screenshot": page.get("screenshot"),
                        "state_fetch_ms": page.get("state_fetch_ms"),
                        "a11y_latency_ms": page.get("a11y_dump_ms"),
                    },
                )

        for cand in all_candidates:
            if not isinstance(cand, dict):
                continue
            if "score_components" not in cand:
                cand["score_components"] = self._candidate_score_components(cand, goal)
            cand.setdefault("final_score", (cand.get("score_components") or {}).get("final_score"))
            cand.setdefault("reason_selected", "selected" if cand in selected else "")
            cand.setdefault("reason_rejected", "" if cand in selected else "not_selected_budget_or_lower_rank")
            components = cand.get("score_components") if isinstance(cand.get("score_components"), dict) else {}
            center = cand.get("center")
            self._append_diagnostic_jsonl(
                goal,
                "candidate_scores.jsonl",
                {
                    "step": trace.get("step"),
                    "candidate_id": cand.get("key") or cand.get("index"),
                    "label": _clean_text(cand.get("label")),
                    "text": _clean_text((cand.get("a11y") or {}).get("text") if isinstance(cand.get("a11y"), dict) else ""),
                    "content_desc": _clean_text((cand.get("a11y") or {}).get("content_description") if isinstance(cand.get("a11y"), dict) else ""),
                    "resource_id": _clean_text((cand.get("a11y") or {}).get("resource_id") if isinstance(cand.get("a11y"), dict) else ""),
                    "class_name": _clean_text((cand.get("a11y") or {}).get("class_name") if isinstance(cand.get("a11y"), dict) else ""),
                    "bbox": (cand.get("a11y") or {}).get("bbox") if isinstance(cand.get("a11y"), dict) else cand.get("bbox"),
                    "center": [int(center[0]), int(center[1])] if isinstance(center, (list, tuple)) and len(center) >= 2 else center,
                    "operator": cand.get("operator") or components.get("operator"),
                    "operator_reason": cand.get("operator_reason") or components.get("operator_reason"),
                    "PatternPrior": components.get("PatternPrior"),
                    "MissingSlotGain": components.get("MissingSlotGain"),
                    "TaskEntityMatch": components.get("TaskEntityMatch"),
                    "OperatorPriority": components.get("OperatorPriority"),
                    "TaskProgress": components.get("TaskProgress"),
                    "Novelty": components.get("Novelty"),
                    "RiskPenalty": components.get("RiskPenalty"),
                    "RollbackRisk": components.get("RollbackRisk"),
                    "RollbackCost": components.get("RollbackCost"),
                    "WrongScreenRolePenalty": components.get("WrongScreenRolePenalty"),
                    "RevisitPenalty": components.get("RevisitPenalty"),
                    "final_score": components.get("final_score") or cand.get("final_score"),
                    "final_score_or_utility": components.get("final_score_or_utility") or components.get("final_score") or cand.get("final_score"),
                    "selected": bool(cand in selected),
                    "selected_rank": next((i + 1 for i, item in enumerate(selected) if item == cand), None),
                    "selected_reason": cand.get("reason_selected"),
                    "rejected_reason": cand.get("reason_rejected"),
                    "quality_filter_reason": cand.get("quality_filter_reason") or components.get("quality_filter_reason"),
                },
            )
        for obs in trace.get("observations") or []:
            if not isinstance(obs, dict):
                continue
            cand = {}
            steps = obs.get("steps") or []
            if steps and isinstance(steps[0], dict) and isinstance(steps[0].get("candidate"), dict):
                cand = steps[0]["candidate"]
            components = cand.get("score_components") if isinstance(cand, dict) else {}
            rollback_info = obs.get("rollback") if isinstance(obs.get("rollback"), dict) else {}
            if not isinstance(components, dict):
                components = {}
            evidence_type, threshold = self._evidence_type_threshold(goal, obs, cand if isinstance(cand, dict) else {})
            candidate_type = _clean_text(obs.get("evidence_type") or obs.get("hint_type") or evidence_type or "NONE") or "NONE"
            confidence_components = self._evidence_confidence_components(obs, rollback_info, components)
            confidence = float(confidence_components.get("confidence") or 0.0)
            rejected_reason = self._evidence_rejection_reason(
                evidence_type,
                confidence,
                threshold,
                obs,
                rollback_info,
                cand if isinstance(cand, dict) else {},
            )
            decision = {
                "step": trace.get("step"),
                "source_exploration_step": trace.get("step"),
                "branch_id": obs.get("branch_id"),
                "depth": obs.get("depth_reached"),
                "candidate_label": _clean_text(cand.get("label") or cand.get("merged")) if isinstance(cand, dict) else "",
                "operator": cand.get("operator") if isinstance(cand, dict) else obs.get("operator"),
                "evidence_type_candidate": candidate_type,
                "final_evidence_type": evidence_type,
                "score_components": components,
                "PatternPrior": components.get("PatternPrior"),
                "MissingSlotGain": components.get("MissingSlotGain"),
                "TaskEntityMatch": components.get("TaskEntityMatch"),
                "RiskPenalty": components.get("RiskPenalty"),
                "RollbackRisk": components.get("RollbackRisk"),
                "final_score_or_utility": components.get("final_score_or_utility")
                or components.get("final_score")
                or components.get("score"),
                "evidence_gain": obs.get("evidence_gain") or obs.get("score"),
                "confidence": float(confidence),
                "confidence_components": confidence_components,
                "threshold": float(threshold),
                "state_match_score": obs.get("state_match_score"),
                "rollback_verified": bool(rollback_info.get("success")),
                "slot_coverage": obs.get("slot_coverage"),
                "slot_complete": bool(obs.get("slot_complete") or obs.get("answer_complete")),
                "missing_slots": obs.get("missing_slots") or [],
                "target_visible": bool(obs.get("target_visible")),
                "action_safe": not bool(cand.get("risk_boundary")) if isinstance(cand, dict) else True,
                "injected": False,
                "rejected_reason": rejected_reason,
                "rendered_prompt_text": "",
                "observation": obs,
            }
            self._append_diagnostic_jsonl(goal, "evidence_decisions.jsonl", decision)
        for rb in trace.get("rollbacks") or []:
            if not isinstance(rb, dict):
                continue
            event = dict(rb)
            event.setdefault("step", trace.get("step"))
            event.setdefault("task_id", _clean_text(goal)[:200])
            level2_trigger_reason = self._infer_level2_trigger_reason(event)
            event.setdefault("level2_triggered", bool(event.get("level") == "level2" or event.get("mode") in {"home_replay", "replay"} or event.get("replayed_actions")))
            event.setdefault("level2_trigger_reason", level2_trigger_reason if event.get("level2_triggered") else "")
            event.setdefault("rollback_level", event.get("level"))
            event.setdefault("rollback_mode", event.get("mode"))
            event.setdefault("root_activity", trace.get("root_activity") or event.get("root_activity") or "")
            event.setdefault("final_activity", event.get("final_activity") or event.get("current_activity") or "")
            event.setdefault("activity_match", event.get("activity_match"))
            event.setdefault("root_phash", trace.get("root_hash") or event.get("root_phash"))
            event.setdefault("final_phash", event.get("final_phash"))
            event.setdefault("phash_diff", event.get("phash_diff"))
            event.setdefault("anchor_jaccard", event.get("anchor_jaccard"))
            event.setdefault("rollback_total_ms", float(rb.get("latency_ms") or 0.0))
            event.setdefault("rollback_action_ms", float(rb.get("rollback_action_ms") or 0.0))
            event.setdefault("rollback_verify_ms", float(rb.get("verify_ms") or 0.0))
            event.setdefault("rollback_verify_get_state_ms", float(rb.get("verify_get_state_ms") or 0.0))
            event.setdefault("rollback_verify_a11y_ms", float(rb.get("verify_a11y_ms") or 0.0))
            event.setdefault("rollback_phash_ms", float(rb.get("phash_ms") or 0.0))
            event.setdefault("replay_action_ms", float(rb.get("replay_action_ms") or 0.0))
            event.setdefault("final_verify_ms", float(rb.get("final_verify_ms") or 0.0))
            event.setdefault(
                "latency_breakdown",
                {
                    "rollback_total_ms": event.get("rollback_total_ms"),
                    "rollback_action_ms": event.get("rollback_action_ms"),
                    "rollback_verify_ms": event.get("rollback_verify_ms"),
                    "rollback_verify_get_state_ms": event.get("rollback_verify_get_state_ms"),
                    "rollback_verify_a11y_ms": event.get("rollback_verify_a11y_ms"),
                    "rollback_phash_ms": event.get("rollback_phash_ms"),
                    "replay_action_ms": event.get("replay_action_ms"),
                    "final_verify_ms": event.get("final_verify_ms"),
                },
            )
            event.setdefault("planned_action_suppressed", bool(trace.get("planned_action_suppressed")))
            event.setdefault("suppression_reason", trace.get("suppression_reason") or "")
            event.setdefault("failure_reasons", rb.get("failure_reasons") or rb.get("failure_reason") or [])
            event.setdefault("branch_evidence_discarded", not bool(event.get("success", True)))
            event.setdefault("prompt_history_contaminated", False)
            event.setdefault(
                "main_action_suppressed",
                bool(trace.get("main_action_blocked_by_rollback_gate")),
            )
            event.setdefault("rollback_timeline_image", "")
            event.setdefault("rollback_case_markdown", "")
            self._append_diagnostic_jsonl(goal, "rollback_events.jsonl", event)
            self._append_diagnostic_jsonl(
                goal,
                "rollback_gate_decisions.jsonl",
                {
                    "timestamp": time.time(),
                    "task": goal,
                    "variant": self.light_explore_variant,
                    "step": trace.get("step"),
                    "branch_id": event.get("branch_id")
                    or event.get("candidate_id")
                    or trace.get("branch_id")
                    or trace.get("trace_id"),
                    "rollback_level": event.get("rollback_level"),
                    "rollback_mode": event.get("rollback_mode"),
                    "trigger_reason": event.get("trigger_reason")
                    or event.get("reason")
                    or event.get("failure_reason"),
                    "success": bool(event.get("success", True)),
                    "state_aligned": bool(event.get("state_aligned", event.get("success", True))),
                    "branch_evidence_discarded": bool(event.get("branch_evidence_discarded")),
                    "prompt_history_contaminated": bool(event.get("prompt_history_contaminated")),
                    "main_action_suppressed": bool(event.get("main_action_suppressed")),
                    "screenshot_path": event.get("screenshot_path")
                    or event.get("screenshot")
                    or "",
                    "before_anchor": event.get("before_anchor", ""),
                    "after_anchor": event.get("after_anchor", ""),
                    "before_activity": event.get("before_activity", ""),
                    "after_activity": event.get("after_activity", ""),
                    "before_phash": event.get("before_phash", ""),
                    "after_phash": event.get("after_phash", ""),
                    "rollback_timeline_image": event.get("rollback_timeline_image", ""),
                    "rollback_case_markdown": event.get("rollback_case_markdown", ""),
                },
            )
            if bool(event.get("level2_triggered")) or not bool(event.get("success", True)):
                task_dir = self._task_output_dir(goal)
                if task_dir:
                    try:
                        branch_id = _clean_text(event.get("branch_id") or event.get("candidate_id") or "unknown")
                        base_name = (
                            "rollback_failure_cases.md"
                            if not bool(event.get("success", True))
                            else "rollback_level2_cases.md"
                        )
                        with open(os.path.join(task_dir, base_name), "a", encoding="utf-8") as f:
                            f.write(f"## step {trace.get('step')} branch {branch_id}\n\n")
                            f.write(f"- task: {_clean_text(goal)}\n")
                            f.write(f"- level: {event.get('rollback_level')}\n")
                            f.write(f"- mode: {event.get('rollback_mode')}\n")
                            f.write(f"- success: {event.get('success')}\n")
                            f.write(f"- level2 trigger: {event.get('level2_trigger_reason')}\n")
                            f.write(f"- suppression: {event.get('planned_action_suppressed')} {event.get('suppression_reason')}\n")
                            f.write(f"- failure reasons: {event.get('failure_reasons')}\n\n")
                            f.write("```json\n")
                            f.write(json.dumps(event, ensure_ascii=False, indent=2, default=str)[:4000])
                            f.write("\n```\n\n")
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass

    def _execute_probe_action(self, action: json_action.JSONAction) -> str:
        """Execute speculative actions with a fast direct-ADB path when possible."""
        action_start = time.time()
        if not self.light_explore_fast_mode:
            self.env.execute_action(action)
            latency_ms = float(max(0.0, time.time() - action_start) * 1000.0)
            self._append_latency_profile_event(
                getattr(self, "_current_goal_for_depth", ""),
                {
                    "event": "probe_action",
                    "action_type": str(action.action_type),
                    "mode": "env_execute",
                    "latency_ms": latency_ms,
                    "x": action.x,
                    "y": action.y,
                    "text_len": len(str(action.text or "")),
                    "includes_settle": False,
                },
            )
            return "env_execute"

        action_type = action.action_type
        mode = "fast_fallback"
        try:
            if action_type == json_action.CLICK and action.index is None and action.x is not None and action.y is not None:
                adb_utils.tap_screen(int(action.x), int(action.y), self.env.controller)
                mode = "fast_tap"
            elif action_type == json_action.NAVIGATE_BACK:
                adb_utils.press_back_button(self.env.controller)
                mode = "fast_back"
            elif action_type == json_action.NAVIGATE_HOME:
                adb_utils.press_home_button(self.env.controller)
                mode = "fast_home"
            elif action_type == json_action.WAIT:
                time.sleep(1.0)
                mode = "fast_wait"
            elif action_type == json_action.OPEN_APP and _clean_text(getattr(action, "app_name", "")):
                adb_utils.launch_app(_clean_text(getattr(action, "app_name", "")), self.env.controller)
                mode = "fast_open_app"
            elif action_type in {json_action.SWIPE, json_action.SCROLL} and action.direction and action.index is None:
                screen_width, screen_height = self.env.logical_screen_size
                mid_x, mid_y = 0.5 * screen_width, 0.5 * screen_height
                direction = action.direction
                if action_type == json_action.SWIPE:
                    if direction == "down":
                        start_x, start_y, end_x, end_y = mid_x, 0, mid_x, screen_height // 2
                    elif direction == "up":
                        start_x, start_y, end_x, end_y = mid_x, screen_height // 2, mid_x, 0
                    elif direction == "left":
                        start_x, start_y, end_x, end_y = 0, mid_y, screen_width // 2, mid_y
                    elif direction == "right":
                        start_x, start_y, end_x, end_y = screen_width // 2, mid_y, 0, mid_y
                    else:
                        raise ValueError(f"Unsupported swipe direction: {direction}")
                else:
                    if direction == "down":
                        start_x, start_y, end_x, end_y = mid_x, mid_y, mid_x, 0
                    elif direction == "up":
                        start_x, start_y, end_x, end_y = mid_x, mid_y, mid_x, screen_height
                    elif direction == "right":
                        start_x, start_y, end_x, end_y = mid_x, mid_y, 0, mid_y
                    elif direction == "left":
                        start_x, start_y, end_x, end_y = mid_x, mid_y, screen_width, mid_y
                    else:
                        raise ValueError(f"Unsupported scroll direction: {direction}")
                adb_utils.issue_generic_request(
                    adb_utils.generate_swipe_command(
                        int(start_x),
                        int(start_y),
                        int(end_x),
                        int(end_y),
                        500,
                    ),
                    self.env.controller,
                )
                mode = f"fast_{action_type}"
            elif (
                action_type == json_action.INPUT_TEXT
                and action.index is None
                and action.text
            ):
                if action.x is not None and action.y is not None:
                    adb_utils.tap_screen(int(action.x), int(action.y), self.env.controller)
                    time.sleep(max(0.25, min(1.0, self.light_explore_action_settle_s)))
                if action.clear_text:
                    adb_utils.issue_generic_request(
                        [
                            "shell",
                            "input",
                            "keycombination",
                            "113",
                            "29",
                            "&&",
                            "input",
                            "keyevent",
                            "67",
                        ],
                        self.env.controller,
                    )
                    time.sleep(0.25)
                adb_utils.type_text(str(action.text), self.env.controller, timeout_sec=10)
                adb_utils.press_enter_button(self.env.controller)
                mode = "fast_input_text"
            else:
                self.env.execute_action(action)
        except Exception:  # pylint: disable=broad-exception-caught
            self.env.execute_action(action)
            mode = "env_execute_after_fast_error"

        if self.light_explore_action_settle_s > 0:
            time.sleep(float(self.light_explore_action_settle_s))
        latency_ms = float(max(0.0, time.time() - action_start) * 1000.0)
        self._append_latency_profile_event(
            getattr(self, "_current_goal_for_depth", ""),
            {
                "event": "probe_action",
                "action_type": str(action.action_type),
                "mode": mode,
                "latency_ms": latency_ms,
                "x": action.x,
                "y": action.y,
                "text_len": len(str(action.text or "")),
                "clear_text": bool(action.clear_text),
                "includes_settle": bool(self.light_explore_action_settle_s > 0),
                "settle_s": float(self.light_explore_action_settle_s),
            },
        )
        return mode

    def _execute_root_open_anchor(
        self,
        action: json_action.JSONAction,
        root_activity: str,
    ) -> str:
        root_pkg = self._package_from_activity(root_activity)
        action_pkg = self._package_for_app_name(getattr(action, "app_name", ""))
        if (
            action.action_type != json_action.OPEN_APP
            or not root_pkg
            or action_pkg != root_pkg
            or self._is_launcher_activity(root_activity)
            or "/" not in _clean_text(root_activity)
        ):
            return self._execute_probe_action(action)

        action_start = time.time()
        try:
            adb_utils.start_activity(root_activity, extra_args=[], env=self.env.controller, timeout_sec=5)
            if self.light_explore_action_settle_s > 0:
                time.sleep(float(self.light_explore_action_settle_s))
            curr_pkg = self._package_from_activity(self._foreground_activity_name())
            if curr_pkg == root_pkg:
                latency_ms = float(max(0.0, time.time() - action_start) * 1000.0)
                self._append_latency_profile_event(
                    getattr(self, "_current_goal_for_depth", ""),
                    {
                        "event": "probe_action",
                        "action_type": str(action.action_type),
                        "mode": "fast_start_root_activity",
                        "latency_ms": latency_ms,
                        "app_name": _clean_text(getattr(action, "app_name", "")),
                        "root_activity": root_activity,
                        "includes_settle": bool(self.light_explore_action_settle_s > 0),
                        "settle_s": float(self.light_explore_action_settle_s),
                    },
                )
                return "fast_start_root_activity"
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return self._execute_probe_action(action)

    @staticmethod
    def _normalize_activity_name(activity: str | None) -> str:
        return _clean_text(activity).lower()

    def _foreground_activity_name(self) -> str:
        self._state_acquisition_metrics["activity_calls"] = (
            float(self._state_acquisition_metrics.get("activity_calls") or 0.0) + 1.0
        )
        start = time.time()
        try:
            value = str(self.env.foreground_activity_name or "").strip()
        except Exception:  # pylint: disable=broad-exception-caught
            value = ""
        self._append_latency_profile_event(
            "",
            {
                "event": "activity_fetch",
                "latency_ms": float(max(0.0, time.time() - start) * 1000.0),
                "activity": value,
            },
        )
        return value

    @staticmethod
    def _is_launcher_activity(activity: str | None) -> bool:
        value = _clean_text(activity).lower()
        if not value:
            return False
        package = value.split("/", 1)[0]
        return bool(
            package in {
                "com.google.android.apps.nexuslauncher",
                "com.android.launcher",
                "com.android.launcher2",
                "com.android.launcher3",
            }
            or package.endswith(".quickstep")
            or package.endswith(".nexuslauncher")
            or value.startswith("com.google.android.apps.nexuslauncher/")
        )

    @staticmethod
    def _package_from_activity(activity: str | None) -> str:
        value = _clean_text(activity).lower()
        if not value or "/" not in value:
            return value
        return value.split("/", 1)[0]

    def _package_for_app_name(self, app_name: str | None) -> str:
        app_text = _clean_text(app_name)
        if not app_text:
            return ""
        try:
            normalized = _clean_text(adb_utils.normalize_app_name(app_text))
        except Exception:  # pylint: disable=broad-exception-caught
            normalized = app_text
        for candidate in (normalized, app_text):
            if not candidate:
                continue
            try:
                activity = adb_utils.get_adb_activity(candidate)
            except Exception:  # pylint: disable=broad-exception-caught
                activity = None
            package = self._package_from_activity(activity)
            if package:
                return package
            if "/" in candidate:
                return self._package_from_activity(candidate)
            if "." in candidate and " " not in candidate:
                return candidate.lower()
        return ""

    def _app_name_for_root_activity(self, root_activity: str | None) -> str:
        root_pkg = self._package_from_activity(root_activity)
        if not root_pkg or self._is_launcher_activity(root_activity):
            return ""
        package_aliases = {
            "com.android.chrome": "Chrome",
            "com.android.settings": "Settings",
            "com.google.android.documentsui": "Files",
            "com.google.android.deskclock": "Clock",
            "com.google.android.contacts": "Contacts",
            "com.android.camera2": "Camera",
            "com.dimowner.audiorecorder": "Audio Recorder",
            "net.gsantner.markor": "Markor",
            "org.tasks": "Tasks",
            "com.simplemobiletools.calendar.pro": "Simple Calendar Pro",
            "com.simplemobiletools.draw.pro": "Simple Draw Pro",
            "com.simplemobiletools.gallery.pro": "Simple Gallery Pro",
            "com.simplemobiletools.smsmessenger": "Simple SMS Messenger",
            "com.arduia.expense": "Pro Expense",
            "com.flauschcode.broccoli": "Broccoli APP",
            "net.osmand": "OSMand",
            "de.dennisguse.opentracks": "OpenTracks",
            "net.cozic.joplin": "Joplin",
            "org.videolan.vlc": "VLC",
            "code.name.monkey.retromusic": "Retro Music",
        }
        if root_pkg in package_aliases:
            return package_aliases[root_pkg]
        for app_name in sorted(gelab_agent.AVAILABLE_APPS, key=lambda value: len(str(value)), reverse=True):
            if self._package_for_app_name(str(app_name)) == root_pkg:
                return str(app_name)
        if "." in root_pkg:
            return root_pkg
        return ""

    def _prepare_home_replay_actions(
        self,
        replay_actions: list[json_action.JSONAction],
        root_activity: str,
    ) -> tuple[list[json_action.JSONAction], dict[str, Any]]:
        root_pkg = self._package_from_activity(root_activity)
        root_app_name = self._app_name_for_root_activity(root_activity)
        original_actions = list(replay_actions or [])
        info: dict[str, Any] = {
            "root_package": root_pkg,
            "root_app_name": root_app_name,
            "original_replay_action_types": [str(action.action_type) for action in original_actions],
            "inserted_open_app_anchor": False,
            "dropped_actions_before_anchor": 0,
            "dropped_foreign_open_app_actions": 0,
        }
        if not root_pkg or self._is_launcher_activity(root_activity) or not root_app_name:
            info["replay_action_types"] = [str(action.action_type) for action in original_actions]
            return original_actions, info

        root_open_idx: int | None = None
        for idx, action in enumerate(original_actions):
            if action.action_type != json_action.OPEN_APP:
                continue
            if self._package_for_app_name(getattr(action, "app_name", "")) == root_pkg:
                root_open_idx = idx
                break

        if root_open_idx is None:
            anchored = [
                json_action.JSONAction(action_type=json_action.OPEN_APP, app_name=root_app_name),
                *original_actions,
            ]
            info["inserted_open_app_anchor"] = True
        else:
            anchored = list(original_actions[root_open_idx:])
            info["dropped_actions_before_anchor"] = int(root_open_idx)
            if anchored:
                anchored[0] = json_action.JSONAction(
                    action_type=json_action.OPEN_APP,
                    app_name=root_app_name,
                )

        filtered: list[json_action.JSONAction] = []
        for idx, action in enumerate(anchored):
            if action.action_type == json_action.NAVIGATE_HOME:
                info["dropped_actions_before_anchor"] += 1 if idx == 0 else 0
                continue
            if action.action_type == json_action.OPEN_APP:
                action_pkg = self._package_for_app_name(getattr(action, "app_name", ""))
                if action_pkg and action_pkg != root_pkg:
                    info["dropped_foreign_open_app_actions"] += 1
                    continue
            filtered.append(action)

        info["replay_action_types"] = [str(action.action_type) for action in filtered]
        return filtered, info

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
        exact_app_matched = False
        for app in gelab_agent.AVAILABLE_APPS:
            app_low = _clean_text(app).lower()
            if app_low and app_low in text and app_low not in seen:
                seen.add(app_low)
                out.append(app_low)
                exact_app_matched = True
            for token in re.findall(r"[a-z0-9]{4,}", app_low):
                if token in text and token not in seen:
                    seen.add(token)
                    out.append(token)
        if exact_app_matched:
            return out
        synonym_map = {
            "clock": ("stopwatch", "timer", "alarm"),
            "camera": ("photo", "picture", "video", "record a video", "take"),
            "messages": ("sms", "message", "text message", "reply", "resend"),
            "chrome": ("browser", "web", "website", "search", "url"),
            "vlc": ("playlist", "music"),
            "markor": ("note", "notes", "markdown", "todo"),
            "calendar": ("event", "meeting", "schedule", "date"),
            "contacts": ("contact", "phone number"),
            "settings": ("wifi", "bluetooth", "brightness", "system"),
            "files": ("file", "folder", "copy", "move"),
            "recipes": ("recipe", "ingredient"),
            "expense": ("receipt", "expense", "price"),
            "tasks": ("task", "priority", "due"),
            "audio recorder": ("audio", "recording", "record audio"),
        }
        for app_hint, triggers in synonym_map.items():
            if app_hint in seen:
                continue
            if any(trigger in text for trigger in triggers):
                seen.add(app_hint)
                out.append(app_hint)
        return out

    def _goal_tokens(self, goal: str) -> list[str]:
        text = _clean_text(goal).lower()
        out: list[str] = []
        seen: set[str] = set()
        for token in re.findall(r"[a-z0-9]{3,}", text):
            if token in GOAL_TOKEN_STOPWORDS:
                continue
            if token not in seen:
                seen.add(token)
                out.append(token)
            if len(out) >= 32:
                break
        return out

    @staticmethod
    def _normalize_resource_id(resource_id: str) -> str:
        rid = _clean_text(resource_id)
        if not rid or rid.lower() in {"none", "null"}:
            return ""
        # Drop package prefix to avoid app-name leakage like "net.gsantner.markor:*".
        rid = rid.split("/")[-1]
        rid = rid.split(":")[-1]
        rid = rid.replace("_", " ")
        rid = re.sub(r"[^a-zA-Z0-9 ]+", " ", rid)
        return _clean_text(rid)

    @staticmethod
    def _tokens_for_embedding(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]{2,}", _clean_text(text).lower())
        if not tokens:
            return []
        out = list(tokens)
        for left, right in zip(tokens, tokens[1:]):
            out.append(f"{left}_{right}")
        return out

    @staticmethod
    def _hash_embedding(text: str, dim: int = 128) -> list[float]:
        vec = [0.0] * max(8, int(dim))
        for token in ExplorerElementAgent._tokens_for_embedding(text):
            digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
            value = int.from_bytes(digest, byteorder="big", signed=False)
            idx = value % len(vec)
            sign = 1.0 if ((value >> 8) & 1) else -1.0
            # Slightly downweight very short tokens to reduce noise from UI chrome.
            weight = 0.65 if len(token) <= 3 else 1.0
            vec[idx] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 1e-8:
            return vec
        return [v / norm for v in vec]

    @staticmethod
    def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        size = min(len(vec_a), len(vec_b))
        if size <= 0:
            return 0.0
        score = sum(vec_a[i] * vec_b[i] for i in range(size))
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _lexical_overlap_score(merged: str, goal_tokens: list[str], app_keywords: list[str]) -> float:
        merged_low = _clean_text(merged).lower()
        if not merged_low:
            return 0.0
        score = 0.0
        for kw in app_keywords:
            if kw and kw in merged_low:
                score += 0.28 + min(len(kw), 12) * 0.015
        merged_tokens = set(re.findall(r"[a-z0-9]{3,}", merged_low))
        goal_token_set = set(goal_tokens)
        if merged_tokens and goal_token_set:
            overlap = merged_tokens.intersection(goal_token_set)
            if overlap:
                score += min(0.5, float(len(overlap)) * 0.12)
                score += min(0.3, float(len(overlap)) / max(1.0, float(len(merged_tokens))))
        return float(max(0.0, min(1.0, score)))

    def _task_relevance_score(self, merged: str, goal: str, goal_tokens: list[str], app_keywords: list[str]) -> float:
        embed_score = self._cosine(self._hash_embedding(goal), self._hash_embedding(merged))
        lexical_score = self._lexical_overlap_score(
            merged=merged,
            goal_tokens=goal_tokens,
            app_keywords=app_keywords,
        )
        # Local hashing is only a weak semantic approximation; keep lexical/app
        # overlap dominant to avoid false positives on short labels like "On".
        return float(max(lexical_score, min(0.05, embed_score * 0.2)))

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
        hint = _clean_text(getattr(element, "hint_text", ""))
        element_id = self._normalize_resource_id(getattr(element, "resource_id", ""))
        merged = " ".join(x for x in [text, desc, hint, element_id] if x)
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
    def _bbox_dict(element: Any) -> dict[str, int] | None:
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

    def _element_trace(self, element: Any, index: int | None = None) -> dict[str, Any]:
        center = self._safe_center_from_element(element)
        return {
            "index": index,
            "text": _clean_text(getattr(element, "text", "")),
            "content_description": _clean_text(getattr(element, "content_description", "")),
            "hint_text": _clean_text(getattr(element, "hint_text", "")),
            "resource_id": _clean_text(getattr(element, "resource_id", "")),
            "class_name": _clean_text(getattr(element, "class_name", "")),
            "bbox": self._bbox_dict(element),
            "center": [int(center[0]), int(center[1])] if center is not None else None,
            "is_clickable": bool(getattr(element, "is_clickable", False)),
            "is_long_clickable": bool(getattr(element, "is_long_clickable", False)),
            "is_editable": bool(getattr(element, "is_editable", False)),
            "is_scrollable": bool(getattr(element, "is_scrollable", False)),
        }

    def _state_a11y_trace(self, state: Any, limit: int = TRACE_A11Y_LIMIT) -> list[dict[str, Any]]:
        elements = list(getattr(state, "ui_elements", None) or [])
        trace_limit = min(max(0, int(limit)), int(self.light_explore_trace_a11y_limit))
        if not self.light_explore_lightweight_a11y_trace:
            return [self._element_trace(element, idx) for idx, element in enumerate(elements[:trace_limit])]
        out: list[dict[str, Any]] = []
        for idx, element in enumerate(elements[:trace_limit]):
            text = _clean_text(getattr(element, "text", ""))
            desc = _clean_text(getattr(element, "content_description", ""))
            rid = _clean_text(getattr(element, "resource_id", ""))
            cls = _clean_text(getattr(element, "class_name", "")).split(".")[-1]
            center = self._safe_center_from_element(element)
            out.append(
                {
                    "index": idx,
                    "label": text or desc or self._normalize_resource_id(rid) or cls,
                    "resource_id": rid,
                    "class": cls,
                    "bbox": self._bbox_dict(element),
                    "center": [int(center[0]), int(center[1])] if center is not None else None,
                    "flags": "".join(
                        flag
                        for flag, enabled in (
                            ("C", bool(getattr(element, "is_clickable", False))),
                            ("L", bool(getattr(element, "is_long_clickable", False))),
                            ("E", bool(getattr(element, "is_editable", False))),
                            ("S", bool(getattr(element, "is_scrollable", False))),
                        )
                        if enabled
                    ),
                }
            )
        return out

    def _state_semantic_summary(self, state: Any, limit: int = TRACE_OBSERVED_ELEMENT_LIMIT) -> list[str]:
        out: list[str] = []
        for element in list(getattr(state, "ui_elements", None) or []):
            text = _clean_text(getattr(element, "text", ""))
            desc = _clean_text(getattr(element, "content_description", ""))
            rid = self._normalize_resource_id(getattr(element, "resource_id", ""))
            label = text or desc or rid
            if not label:
                continue
            if label in out:
                continue
            out.append(label)
            if len(out) >= max(1, int(limit)):
                break
        return out

    def _state_structural_summary(self, state: Any, limit: int = 24) -> list[str]:
        out: list[str] = []
        for element in list(getattr(state, "ui_elements", None) or []):
            cls = _clean_text(getattr(element, "class_name", "")).split(".")[-1]
            rid = self._normalize_resource_id(getattr(element, "resource_id", ""))
            text = _clean_text(getattr(element, "text", ""))
            desc = _clean_text(getattr(element, "content_description", ""))
            label = text or desc or rid or cls
            if not label:
                continue
            flags = []
            if bool(getattr(element, "is_clickable", False)):
                flags.append("click")
            if bool(getattr(element, "is_editable", False)):
                flags.append("edit")
            if bool(getattr(element, "is_scrollable", False)):
                flags.append("scroll")
            item = f"{cls}:{rid or label}" if cls else (rid or label)
            if flags:
                item += f"[{','.join(flags)}]"
            if item not in out:
                out.append(item)
            if len(out) >= max(1, int(limit)):
                break
        return out

    def _candidate_key(self, element: Any, center: tuple[int, int], merged: str) -> str:
        bucket = f"{int(center[0] // 24)}:{int(center[1] // 24)}"
        label = _clean_text(merged).lower()[:96]
        cls = _clean_text(getattr(element, "class_name", "")).lower()
        return f"{bucket}|{cls}|{label}"

    @staticmethod
    def _is_interactive(element: Any) -> bool:
        return bool(
            getattr(element, "is_clickable", False)
            or getattr(element, "is_long_clickable", False)
            or getattr(element, "is_editable", False)
        )

    def _is_risky_probe_text(self, merged: str) -> bool:
        if not getattr(self, "light_explore_lexical_rules", False):
            return False
        low = _clean_text(merged).lower()
        if not low:
            return False
        risky_patterns = (
            r"\b(delete|remove|trash|discard|clear all|erase)\b",
            r"\b(save|done|ok|confirm|submit|finish|send|share|call)\b",
            r"\b(record|stop|pause|start|shutter|capture)\b",
            r"\b(on|off|silent|none|enabled|disabled)\b",
            r"\b(pay|buy|checkout|purchase|order)\b",
            r"\b(sign out|logout|log out|reset)\b",
        )
        return any(re.search(pattern, low) for pattern in risky_patterns)

    def _is_answer_or_lookup_goal(self, goal: str) -> bool:
        if not getattr(self, "light_explore_lexical_rules", False):
            return False
        low = _clean_text(goal).lower()
        if not low:
            return False
        if re.search(r"\b(answer with|what|which|who|where|when|how many|count|duration|total|longest|next upcoming)\b", low):
            return True
        return low.rstrip().endswith("?")

    def _task_mode(self, goal: str) -> str:
        if not getattr(self, "light_explore_lexical_rules", False):
            return "GENERAL"
        low = _clean_text(goal).lower()
        if re.search(r"\b(delete|remove|clear|trash|erase|discard)\b", low):
            return "DELETE_COMMIT"
        if re.search(r"\b(take (?:one |a |the )?(?:photo|picture|video)|record (?:an? )?(?:audio|video|clip)|capture (?:an? )?(?:photo|picture|video)|start recording)\b", low):
            return "MEDIA_CAPTURE"
        if re.search(r"\b(turn on|turn off|verify|brightness|wifi|wi-fi|bluetooth|stopwatch|timer|toggle)\b", low):
            return "SIMPLE_VERIFY"
        if re.search(r"\b(create|add|edit|enter|fill|write|transcribe|playlist|recipe|contact|calendar event|rename|type|input|draft|new folder|new contact)\b", low):
            return "FORM_CREATE_EDIT"
        if re.search(r"\b(what|when|how many|how long|duration|distance|count|total|longest|first|next|is|do i have|answer with|which|who|where)\b", low) or low.rstrip().endswith("?"):
            return "INFO_QUERY"
        if re.search(r"\b(open|find|search|locate|file|folder|note|titled|named|move the file|in downloads|in markor|in joplin)\b", low):
            return "NAVIGATION_SEARCH"
        return "UNKNOWN"

    @staticmethod
    def _mode_is_deterministic_or_risky(task_mode: str) -> bool:
        return _clean_text(task_mode) in {"FORM_CREATE_EDIT", "DELETE_COMMIT", "SIMPLE_VERIFY", "MEDIA_CAPTURE"}

    def _selective_visible_flags(self, goal: str, labels: list[str], page_stalled: bool = False) -> dict[str, Any]:
        labels_low = " ".join(_clean_text(x).lower() for x in labels if _clean_text(x))
        entities = [e for e in self._task_entities(goal) if len(e) >= 3][:10]
        target_hits = [e for e in entities if e.lower() in labels_low]
        facts = self._extract_answer_facts_from_labels(labels, goal, limit=5) if self.light_explore_answer_extractors else []
        useful_facts = [x for x in facts if not _clean_text(x).lower().startswith("avoid ")]
        schema_visible = bool(re.search(r"\b(search|find|filter|query|title|name|date|time|description|phone|email|amount|folder|file|field)\b", labels_low))
        return {
            "target_visible": bool(target_hits),
            "answer_visible": bool(useful_facts),
            "schema_visible": bool(schema_visible),
            "repeated_or_stuck": bool(page_stalled),
            "target_hits": target_hits,
            "answer_facts": useful_facts,
        }

    def _task_derived_schema_labels(self, goal: str, task_mode: str, labels: list[str] | None = None) -> list[str]:
        """Derive no-action schema guidance from the task when UI labels are sparse."""
        low = _clean_text(goal).lower()
        observed_low = " ".join(_clean_text(x).lower() for x in list(labels or []))
        rows: list[str] = []

        def add(text: str) -> None:
            text = _clean_text(text)
            if text and text not in rows:
                rows.append(text)

        for entity in self._task_entities(goal)[:8]:
            if len(entity) >= 3:
                add(f"target name/entity field: {entity}")

        if task_mode == "FORM_CREATE_EDIT":
            add("form schema fields: name title date time description note amount category phone email folder file")
            field_rules = (
                ("title", r"\b(title|titled|subject|event)\b"),
                ("name", r"\b(name|contact|person|recipe|playlist)\b"),
                ("date", r"\b(date|20\\d\\d-|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b"),
                ("time", r"\b(time|\\d{1,2}h|\\d{1,2}:\\d{2}|minutes?|mins?)\b"),
                ("description", r"\b(description|describe)\b"),
                ("note", r"\b(note|notes?)\b"),
                ("amount", r"\b(amount|dollars?|\\$)\b"),
                ("category", r"\b(category|type)\b"),
                ("phone", r"\b(phone|number|\\+\\d)\b"),
                ("email", r"\b(email|mail)\b"),
                ("folder", r"\b(folder|directory)\b"),
                ("file", r"\b(file|filename|rename)\b"),
            )
            for field, pattern in field_rules:
                if re.search(pattern, low) or re.search(pattern, observed_low):
                    add(f"required form field: {field}")
            add("safe completion schema: fill exact task fields first; save/complete only after visible confirmation")
        elif task_mode == "DELETE_COMMIT":
            add("delete schema fields: target name row search filter list absence verification")
            add("safe delete schema: delete exact targets only; after confirm, verify each target name is absent before COMPLETE")
        elif task_mode == "SIMPLE_VERIFY":
            add("verification schema fields: current setting state target state visible confirmation")
            add("safe verify schema: toggle only if current state differs; COMPLETE only after the requested state is visible")
        elif task_mode == "MEDIA_CAPTURE":
            add("media schema fields: record capture stop save filename saved item confirmation")
            add("safe media schema: capture/record once; stop/save only when prompted; COMPLETE after saved/list confirmation")
        elif task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH"}:
            add("search schema fields: search query filter result list detail answer")

        for label in list(labels or [])[:24]:
            cleaned = _clean_text(label)
            if re.search(
                r"\b(name|title|date|time|phone|email|amount|description|note|folder|file|field|search|filter|query)\b",
                cleaned.lower(),
            ):
                add(f"visible schema/control: {cleaned}")
        return rows[:32]

    def _passive_schema_observation(
        self,
        *,
        goal: str,
        strategy_id: str,
        step_id: int,
        branch_id: int,
        root_state: Any,
        root_activity: str,
        root_hash: int,
        root_screenshot: str,
        observed_elements: list[str],
        task_mode: str,
        reason: str,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        schema_labels = self._task_derived_schema_labels(goal, task_mode, observed_elements)
        if not schema_labels:
            return None
        schema_observed = list(dict.fromkeys(schema_labels + list(observed_elements or [])))[:96]
        pseudo_candidate = {
            "label": "Task-derived visible schema",
            "merged": "Task-derived visible schema",
            "operator": "FormSchema",
            "action_kind": "inspect",
            "risk_boundary": False,
            "layout_region": "root_page",
            "semantic_role": "schema_observation",
        }
        schema_assessment = {
            "evidence_gain": 1.0,
            "confidence": 0.72,
            "evidence_type": "SCHEMA_HINT",
            "boundary_type": "SCHEMA_BOUNDARY",
            "stop_reason": reason or "passive_task_schema",
            "schema_labels": schema_observed[:32],
            "new_labels": schema_labels[:16],
            "depth": 1,
        }
        schema_rollback = {"success": True, "mode": "passive_no_action", "level": "level0"}
        schema_capsule = self._build_evidence_capsule(
            goal=goal,
            strategy_id=strategy_id,
            step_id=step_id,
            branch_id=branch_id,
            depth=1,
            parent_state=root_state,
            parent_activity=root_activity,
            child_state=root_state,
            child_activity=root_activity,
            candidate=pseudo_candidate,
            assessment=schema_assessment,
            rollback_info=schema_rollback,
        )
        schema_step = {
            "depth": 1,
            "candidate": self._candidate_trace(pseudo_candidate),
            "changed": False,
            "semantic_changed": False,
            "after_activity": root_activity,
            "after_hash": root_hash,
            "hash_diff_from_root": 0,
            "screenshot": root_screenshot or "",
            "observed_elements": schema_observed,
            "a11y": [],
            "stop_reason": reason or "passive_task_schema",
        }
        schema_observation = {
            "branch_id": branch_id,
            "labels": ["Task-derived visible schema"],
            "changed": False,
            "after_activity": root_activity,
            "score": 1.0,
            "depth_reached": 1,
            "observed_elements": schema_observed,
            "steps": [schema_step],
            "rollback": schema_rollback,
            "operator": "FormSchema",
            "operator_reason": reason or "passive_task_schema",
            "layout_region": "root_page",
            "semantic_role": "schema_observation",
            "boundary_type": "SCHEMA_BOUNDARY",
            "stop_reason": reason or "passive_task_schema",
            "evidence_type": "SCHEMA_HINT",
            "evidence_gain": 1.0,
            "confidence": 0.72,
            "slot_evidence": {},
            "evidence_capsule": schema_capsule,
            "no_action_inspect": True,
            "passive_task_schema": True,
        }
        return schema_observation, schema_capsule

    def _selective_task_phase(
        self,
        goal: str,
        step_idx: int,
        labels: list[str],
        page_stalled: bool,
        prior: dict[str, Any] | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        del prior
        task_mode = self._task_mode(goal)
        flags = self._selective_visible_flags(goal, labels, page_stalled)
        if step_idx <= 0:
            return "BOOTSTRAP", "step1_or_no_prior", flags
        if flags["repeated_or_stuck"]:
            return "STUCK_RECOVERY", "repeated_or_stuck" , flags
        if task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH"}:
            if not flags["target_visible"]:
                return "TARGET_ACQUISITION", "target_entity_not_visible", flags
            if not flags["answer_visible"]:
                return "EVIDENCE_GATHERING", "target_visible_answer_incomplete", flags
            return "VERIFICATION", "answer_visible", flags
        if task_mode in {"FORM_CREATE_EDIT", "DELETE_COMMIT", "SIMPLE_VERIFY", "MEDIA_CAPTURE"}:
            if flags["answer_visible"]:
                return "VERIFICATION", "final_state_or_answer_visible", flags
            return "ACTION_EXECUTION", f"deterministic_mode:{task_mode}", flags
        if flags["schema_visible"] and not flags["answer_visible"]:
            return "TARGET_ACQUISITION", "unknown_with_schema_visible", flags
        return "TARGET_ACQUISITION", "default_unknown_target_acquisition", flags

    def _selective_exploration_gate(
        self,
        goal: str,
        task_mode: str,
        task_phase: str,
        page_stalled: bool,
    ) -> dict[str, Any]:
        active_allowed = False
        passive_allowed = True
        reason = "passive_schema_risk_only"
        if task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH"} and task_phase in {"TARGET_ACQUISITION", "EVIDENCE_GATHERING"}:
            active_allowed = True
            reason = "query_navigation_consumable_active"
        elif task_phase == "STUCK_RECOVERY" and not self._mode_is_deterministic_or_risky(task_mode):
            active_allowed = True
            reason = "stuck_recovery_safe_active"
        elif task_phase == "STUCK_RECOVERY" and page_stalled:
            active_allowed = True
            reason = "deterministic_stuck_two_state_escape"
        elif task_mode in {"FORM_CREATE_EDIT", "DELETE_COMMIT", "SIMPLE_VERIFY", "MEDIA_CAPTURE"}:
            active_allowed = False
            reason = f"deterministic_or_risky_passive_only:{task_mode}"
        return {
            "active_allowed": bool(active_allowed),
            "passive_allowed": bool(passive_allowed),
            "gate_reason": reason,
        }

    def _append_selective_phase_gate_records(
        self,
        goal: str,
        step_idx: int,
        task_mode: str,
        task_phase: str,
        phase_reason: str,
        flags: dict[str, Any],
        gate: dict[str, Any],
    ) -> None:
        phase_row = {
            "step": int(step_idx + 1),
            "task_mode": task_mode,
            "task_phase": task_phase,
            "target_visible": bool(flags.get("target_visible")),
            "answer_visible": bool(flags.get("answer_visible")),
            "schema_visible": bool(flags.get("schema_visible")),
            "repeated_or_stuck": bool(flags.get("repeated_or_stuck")),
            "phase_reason": phase_reason,
            "target_hits": list(flags.get("target_hits") or [])[:8],
            "answer_facts": list(flags.get("answer_facts") or [])[:4],
        }
        self._append_diagnostic_jsonl(goal, "task_phase_records.jsonl", phase_row)
        self._append_diagnostic_jsonl(goal, "exploration_gate_records.jsonl", {
            "step": int(step_idx + 1),
            "task_mode": task_mode,
            "task_phase": task_phase,
            "active_allowed": bool(gate.get("active_allowed")),
            "passive_allowed": bool(gate.get("passive_allowed")),
            "gate_reason": _clean_text(gate.get("gate_reason")),
        })
        if task_mode in {"FORM_CREATE_EDIT", "DELETE_COMMIT", "SIMPLE_VERIFY", "MEDIA_CAPTURE"}:
            self._append_diagnostic_jsonl(goal, "deterministic_task_gate_records.jsonl", {
                "step": int(step_idx + 1),
                "task_mode": task_mode,
                "active_disabled": not bool(gate.get("active_allowed")),
                "reason": _clean_text(gate.get("gate_reason")),
            })

    @staticmethod
    def _candidate_operator(candidate: dict[str, Any] | None, goal: str = "") -> tuple[str, str]:
        if not isinstance(candidate, dict):
            return "Unknown", "no_candidate"
        label = _clean_text(candidate.get("label"))
        merged = _clean_text(candidate.get("merged") or label)
        low = f"{label} {merged}".lower()
        action_kind = _clean_text(candidate.get("action_kind") or json_action.CLICK).lower()
        task_mode = ExplorerElementAgent._task_mode(goal)
        if action_kind in {"type", json_action.INPUT_TEXT}:
            return "FormSchema", "input_schema"
        if re.search(
            r"\b(delete|remove|trash|discard|save|send|share|confirm|ok|done|finish|allow|yes|no)\b",
            low,
        ):
            return "RiskBoundary", "commit_or_external_risk"
        if re.search(r"\b(filter|date picker|date|category)\b", low):
            return "FilterPeek", "filter_or_date_options"
        if re.search(r"\b(search|find|sort)\b", low):
            return "SearchPeek", "search_or_filter"
        if re.search(r"\b(stats|statistics|duration|distance|summary|details?|info|properties)\b", low):
            return "StatsPeek" if re.search(r"\b(stats|statistics|duration|distance|summary)\b", low) else "DetailPeek", "detail_or_stats"
        if re.search(r"\b(row|item|entry|result|record|receipt|transaction|expense|event|ingredient)\b", low):
            return "DetailPeek", "result_or_detail_row"
        if task_mode == "INFO_QUERY" and not re.search(r"\b(add|new|create|edit)\b", low):
            return "ListInspect", "query_visible_list"
        if task_mode == "FORM_CREATE_EDIT":
            return "FormSchema", "form_or_create_schema"
        if re.search(r"\b(menu|settings|folder|tab|drawer|category|calendar|date|month|week|day|file|note|recipe|task)\b", low):
            return "NavigationPeek", "navigation_or_section"
        return "Other", "default_other"

    @staticmethod
    def _candidate_layout_region(candidate: dict[str, Any]) -> str:
        a11y = candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {}
        bbox = a11y.get("bbox") if isinstance(a11y.get("bbox"), dict) else {}
        try:
            x_min = int(bbox.get("x_min", 0))
            x_max = int(bbox.get("x_max", 0))
            y_min = int(bbox.get("y_min", 0))
            y_max = int(bbox.get("y_max", 0))
        except Exception:  # pylint: disable=broad-exception-caught
            x_min = x_max = y_min = y_max = 0
        label = _clean_text(candidate.get("label") or candidate.get("merged")).lower()
        class_name = _clean_text(a11y.get("class_name")).lower()
        if "dialog" in class_name or re.search(r"\b(cancel|ok|confirm|allow|deny)\b", label):
            return "dialog"
        if y_max and y_max <= 320:
            return "top_bar"
        if y_min >= 1850:
            return "bottom_nav"
        area = max(0, x_max - x_min) * max(0, y_max - y_min)
        if area and area < 42000 and x_min >= 780 and y_min >= 1450:
            return "floating_button"
        if "list" in class_name or "recyclerview" in class_name or y_min >= 320:
            return "content_list"
        return "content"

    def _candidate_semantic_role(self, candidate: dict[str, Any], goal: str) -> str:
        operator, _ = self._candidate_operator(candidate, goal)
        label = _clean_text(candidate.get("label") or candidate.get("merged")).lower()
        if operator == "RiskBoundary":
            return "risk"
        if operator in {"SearchPeek", "FilterPeek"} or re.search(r"\b(search|find|filter|sort)\b", label):
            return "search"
        if operator == "FormSchema":
            return "form"
        if operator == "ListInspect":
            return "list_item"
        if self._lexical_overlap_score(
            merged=label,
            goal_tokens=self._goal_tokens(goal),
            app_keywords=self._goal_app_keywords(goal),
        ) > 0.05:
            return "task_relevant"
        return "navigation"

    def _task_entities(self, goal: str) -> list[str]:
        text = _clean_text(goal)
        entities: list[str] = []
        for match in re.finditer(r"['\"]([^'\"]{2,80})['\"]", text):
            entities.append(_clean_text(match.group(1)).lower())
        entities.extend(re.findall(r"\b\d{1,4}(?::\d{2})?\b", text.lower()))
        entities.extend(
            token
            for token in self._goal_tokens(text)
            if len(token) >= 3 and token not in GOAL_TOKEN_STOPWORDS and token not in SECONDARY_TOKEN_STOPWORDS
        )
        seen: set[str] = set()
        out: list[str] = []
        for entity in entities:
            entity = _clean_text(entity).lower()
            if entity and entity not in seen:
                seen.add(entity)
                out.append(entity)
        return out[:16]

    def _estimate_candidate_evidence_gain(self, candidate: dict[str, Any], goal: str) -> float:
        operator, _ = self._candidate_operator(candidate, goal)
        label = _clean_text(candidate.get("label") or candidate.get("merged")).lower()
        task_mode = self._task_mode(goal)
        gain = float(candidate.get("score") or 0.0)
        entities = self._task_entities(goal)
        if any(entity and entity in label for entity in entities):
            gain += 4.0
        if self.light_explore_slot_complete:
            slots = slot_complete_evidence.parse_task_slots(goal)
            if slots.task_subtype == "ACTIVITY_STATS":
                if operator in {"SearchPeek", "FilterPeek", "ListInspect", "NavigationPeek"}:
                    gain += 2.5
                if re.search(r"\b(stats|statistics|detail|track|activity|filter|search)\b", label):
                    gain += 3.0
                if "marker" in label and slots.metric_type in {"duration", "distance", "count"}:
                    gain -= 5.0
            elif slots.task_subtype == "EVENT_QUERY":
                if operator == "ListInspect":
                    gain += 3.0
                if re.search(r"\b(new event|add event|create)\b", label):
                    gain -= 5.0
            elif slots.task_subtype in {"COUNT_LIST", "BOOLEAN_STATUS"} and slots.app_family == "tasks":
                if operator == "ListInspect":
                    gain += 3.0
                if re.search(r"\b(create new filter|display name|date picker)\b", label):
                    gain -= 5.0
            elif slots.task_subtype == "RECIPE_INGREDIENT":
                if any(entity and entity in label for entity in slots.target_entities):
                    gain += 3.0
                elif operator == "ListInspect":
                    gain += 1.0
        if operator in {"SearchPeek", "FilterPeek"} and task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH"}:
            gain += 3.0
        if operator == "ListInspect" and task_mode == "INFO_QUERY":
            gain += 3.0
        if operator == "FormSchema" and task_mode == "FORM_CREATE_EDIT":
            gain += 3.0
        if operator == "NavigationPeek" and task_mode == "NAVIGATION_SEARCH":
            gain += 2.0
        if operator == "RiskBoundary":
            gain -= 3.0
        if self._is_transaction_unsafe_candidate(candidate):
            gain -= 2.0
        gain -= min(2.0, 0.25 * float(candidate.get("visits") or 0.0))
        return gain

    def _search_strategy_candidate_score(self, candidate: dict[str, Any], goal: str) -> float:
        strategy = self.light_explore_search_strategy
        components = self._candidate_score_components(candidate, goal)
        prior_boost = float(candidate.get("reasoning_prior_score") or 0.0)
        candidate["score_components"] = components
        candidate["final_score"] = float(components.get("final_score") or 0.0) + prior_boost
        if strategy == "best_first":
            return float(candidate["final_score"])
        base = float(candidate.get("score") or 0.0)
        estimated_gain = self._estimate_candidate_evidence_gain(candidate, goal)
        operator, _ = self._candidate_operator(candidate, goal)
        risk = 1.0 if operator == "RiskBoundary" or self._is_transaction_unsafe_candidate(candidate) else 0.0
        novelty = 1.0 / float(int(candidate.get("visits") or 0) + 1)
        if strategy == "best_first":
            return 1.5 * estimated_gain + base + 0.5 * novelty + prior_boost - 2.0 * risk
        if strategy == "beam":
            return estimated_gain + 0.75 * base + 0.25 * novelty + prior_boost - 1.5 * risk
        if strategy == "iddfs":
            return estimated_gain + base + prior_boost - 1.0 * risk
        if strategy == "mcts":
            key = _clean_text(candidate.get("key") or candidate.get("merged") or candidate.get("label"))
            stats = self._search_policy_stats.get(key, {})
            visits = float(stats.get("n") or 0.0)
            parent_visits = max(1.0, sum(float(s.get("n") or 0.0) for s in self._search_policy_stats.values()) + 1.0)
            q_value = float(stats.get("q") or estimated_gain)
            risk_value = float(stats.get("risk") or risk)
            risk_penalty = 1.5 if self.light_explore_safe_mcts else 1.0
            return q_value + prior_boost + 1.4 * math.sqrt(math.log(parent_visits + 1.0) / (visits + 1.0)) - risk_penalty * risk_value
        return base + 0.5 * estimated_gain + prior_boost - risk

    def _lb_mcts_operator_name(self, candidate: dict[str, Any], goal: str) -> str:
        operator, _ = self._candidate_operator(candidate, goal)
        label = _clean_text(candidate.get("label") or candidate.get("merged")).lower()
        action_kind = _clean_text(candidate.get("action_kind")).lower()
        if operator == "RiskBoundary":
            return "RiskWarning"
        if self._is_transaction_unsafe_candidate(candidate):
            return "AvoidPath"
        if action_kind in {"type", json_action.INPUT_TEXT}:
            return "SafeSearchInput"
        if operator == "SearchPeek":
            return "SearchPeek"
        if operator == "FilterPeek":
            return "FilterInspect"
        if operator == "StatsPeek":
            return "StatsInspect"
        if operator == "DetailPeek":
            return "DetailInspect"
        if operator == "FormSchema":
            return "SchemaInspect"
        if operator == "NavigationPeek":
            return "NavigationPeek"
        if operator == "ListInspect":
            if any(entity and entity in label for entity in self._task_entities(goal)):
                return "ClickExactResult"
            return "DetailInspect"
        return "PassiveInspect" if candidate.get("passive_only") else "NavigationPeek"

    def _lb_mcts_speculation_level(self, candidate: dict[str, Any], goal: str) -> str:
        operator_name = self._lb_mcts_operator_name(candidate, goal)
        if operator_name in {"RiskWarning", "AvoidPath"} or self._is_transaction_unsafe_candidate(candidate):
            return "D"
        action_kind = _clean_text(candidate.get("action_kind")).lower()
        if candidate.get("passive_only") or operator_name == "PassiveInspect":
            return "A"
        if action_kind in {"type", json_action.INPUT_TEXT}:
            return "C"
        return "B"

    def _lb_mcts_estimated_latency_ms(self, candidate: dict[str, Any], goal: str) -> float:
        operator_name = self._lb_mcts_operator_name(candidate, goal)
        base = {
            "PassiveInspect": 250.0,
            "RiskWarning": 250.0,
            "AvoidPath": 250.0,
            "SearchPeek": 1300.0,
            "SafeSearchInput": 2200.0,
            "ClickExactResult": 1600.0,
            "DetailInspect": 1600.0,
            "StatsInspect": 1800.0,
            "FilterInspect": 1600.0,
            "SchemaInspect": 1200.0,
            "NavigationPeek": 1400.0,
        }.get(operator_name, 1400.0)
        components = candidate.get("score_components") if isinstance(candidate.get("score_components"), dict) else {}
        rollback_risk = float(components.get("RollbackRisk") or 0.0)
        return float(base + 900.0 * max(0.0, min(1.0, rollback_risk)))

    def _lb_mcts_pattern_tuple(
        self,
        candidate: dict[str, Any],
        goal: str,
        root_state: Any,
        root_activity: str,
        root_labels: list[str],
    ) -> dict[str, Any]:
        del root_state
        components = candidate.get("score_components") if isinstance(candidate.get("score_components"), dict) else {}
        operator_name = self._lb_mcts_operator_name(candidate, goal)
        task_mode = self._task_mode(goal)
        screen_role = self._infer_screen_role_from_labels(root_labels)
        entities = self._task_entities(goal)
        derived_parameter = ""
        action_kind = _clean_text(candidate.get("action_kind")).lower()
        if action_kind in {"type", json_action.INPUT_TEXT}:
            derived_parameter = _clean_text(candidate.get("text") or self._extract_goal_search_text(goal))
        elif entities:
            derived_parameter = entities[0]
        prediction = {
            "SearchPeek": "SearchInput",
            "SafeSearchInput": "SearchInput",
            "ClickExactResult": "ClickExactResult",
            "DetailInspect": "DetailInspect",
            "StatsInspect": "StatsInspect",
            "FilterInspect": "FilterInspect",
            "SchemaInspect": "SchemaInspect",
            "AvoidPath": "AvoidPath",
            "RiskWarning": "RiskWarning",
        }.get(operator_name, "DetailInspect")
        function_name = {
            "SearchInput": "type_exact_task_entity",
            "ClickExactResult": "click_row_containing_target_entity",
            "DetailInspect": "inspect_detail_page_for_answer",
            "StatsInspect": "inspect_stat_page_for_answer",
            "FilterInspect": "inspect_filter_or_stats_schema",
            "SchemaInspect": "inspect_schema_no_content_input",
            "AvoidPath": "avoid_known_wrong_or_risky_path",
            "RiskWarning": "avoid_speculative_commit_or_mutation",
        }.get(prediction, "inspect_task_relevant_ui")
        context_signature = hashlib.sha1(
            "|".join(
                [
                    ",".join(self._goal_app_keywords(goal)[:3]),
                    task_mode,
                    screen_role,
                    _clean_text(root_activity),
                    "|".join(_clean_text(x).lower() for x in root_labels[:12]),
                    _clean_text(candidate.get("operator") or ""),
                    _clean_text(candidate.get("label") or candidate.get("merged")),
                ]
            ).encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        probability = max(
            0.05,
            min(
                0.95,
                float(components.get("PatternPrior") or 0.0)
                + 0.15 * float(components.get("TaskEntityMatch") or 0.0)
                - 0.15 * float(components.get("RiskPenalty") or 0.0),
            ),
        )
        return {
            "app": ",".join(self._goal_app_keywords(goal)[:3]),
            "task_mode": task_mode,
            "screen_role": screen_role,
            "context_signature": context_signature,
            "prediction": prediction,
            "function_name": function_name,
            "derived_parameter": derived_parameter,
            "probability": float(probability),
        }

    def _lb_mcts_latency_budget(self) -> tuple[float, dict[str, Any]]:
        expected = float(getattr(self, "_lb_mcts_reasoning_ewma_ms", 0.0) or self.lb_mcts_default_reasoning_latency_ms)
        min_budget = max(0.0, float(getattr(self, "lb_mcts_min_budget_ms", 3000.0)))
        max_budget = max(min_budget, float(getattr(self, "lb_mcts_max_budget_ms", 12000.0)))
        budget = min(max_budget, max(min_budget, float(getattr(self, "lb_mcts_slack_ratio", 0.8)) * expected))
        return float(budget), {
            "expected_reasoning_latency_ms": float(expected),
            "latency_budget_ms": float(budget),
            "max_rollouts_per_step": int(getattr(self, "lb_mcts_max_rollouts_per_step", 12)),
        }

    def _lb_mcts_reward_from_components(
        self,
        candidate: dict[str, Any],
        goal: str,
        latency_cost_norm: float,
    ) -> tuple[float, dict[str, float]]:
        components = candidate.get("score_components") if isinstance(candidate.get("score_components"), dict) else {}
        operator_name = self._lb_mcts_operator_name(candidate, goal)
        label = _clean_text(candidate.get("label") or candidate.get("merged")).lower()
        answer_boundary = 1.0 if operator_name in {"StatsInspect"} else 0.0
        action_boundary = 1.0 if operator_name in {"SearchPeek", "SafeSearchInput", "ClickExactResult", "DetailInspect", "StatsInspect"} else 0.0
        target_entity_found = float(components.get("TaskEntityMatch") or 0.0)
        slot_gain = max(0.0, min(1.0, float(components.get("MissingSlotGain") or 0.0)))
        useful_schema = 1.0 if operator_name in {"SearchPeek", "SafeSearchInput", "FilterInspect", "SchemaInspect"} else 0.0
        hard_negative = 1.0 if operator_name in {"AvoidPath", "RiskWarning"} or "marker" in label else 0.0
        risk_violation = 1.0 if self._is_transaction_unsafe_candidate(candidate) else 0.0
        low_quality = 1.0 if _clean_text(candidate.get("quality_filter_reason") or components.get("quality_filter_reason")) else 0.0
        reward = (
            3.0 * answer_boundary
            + 2.0 * action_boundary
            + 2.0 * target_entity_found
            + 1.5 * slot_gain
            + 1.0 * useful_schema
            + 1.0 * hard_negative
            - 2.0 * risk_violation
            - 1.0 * low_quality
            - 0.5 * latency_cost_norm
        )
        return float(reward), {
            "AnswerBoundary": answer_boundary,
            "ActionBoundary": action_boundary,
            "TargetEntityFound": target_entity_found,
            "SlotCompletionGain": slot_gain,
            "UsefulSchema": useful_schema,
            "HardNegative": hard_negative,
            "RiskViolation": risk_violation,
            "RollbackFailure": 0.0,
            "LowQualityObservation": low_quality,
            "LatencyCostNorm": float(latency_cost_norm),
        }

    def _record_lb_mcts_candidate_filter_stats(
        self,
        *,
        goal: str,
        step: int,
        raw_count: int,
        candidates: list[dict[str, Any]],
    ) -> None:
        quality_reasons = [_clean_text(c.get("quality_filter_reason") or "") for c in candidates]
        risky = [c for c in candidates if self._is_transaction_unsafe_candidate(c) or bool(c.get("risk_boundary"))]
        examples = [
            {
                "label": _clean_text(c.get("label") or c.get("merged")),
                "reason": _clean_text(c.get("quality_filter_reason") or ("risky" if c in risky else "")),
            }
            for c in candidates
            if _clean_text(c.get("quality_filter_reason") or "") or c in risky
        ][:8]
        self._append_diagnostic_jsonl(
            goal,
            "candidate_filter_stats.jsonl",
            {
                "record_type": "CandidateFilterStats",
                "task_id": _clean_text(goal)[:200],
                "step": int(step),
                "raw_candidate_count": int(raw_count),
                "after_filter_count": int(len(candidates)),
                "filtered_system_ui_count": sum(1 for r in quality_reasons if "system" in r or "status" in r),
                "filtered_generic_container_count": sum(1 for r in quality_reasons if "container" in r or "generic" in r),
                "filtered_invalid_bbox_count": sum(1 for r in quality_reasons if "bbox" in r or "geometry" in r),
                "filtered_app_name_only_count": sum(1 for r in quality_reasons if "app_name" in r),
                "filtered_duplicate_count": sum(1 for r in quality_reasons if "duplicate" in r),
                "filtered_wrong_task_family_count": sum(1 for r in quality_reasons if "wrong" in r),
                "filtered_risky_count": int(len(risky)),
                "top_filtered_examples": examples,
            },
        )

    def _select_lb_mcts_branch_candidates(
        self,
        *,
        goal: str,
        step_idx: int,
        candidates: list[dict[str, Any]],
        root_state: Any,
        root_activity: str,
        root_labels: list[str],
        trace: dict[str, Any],
    ) -> list[dict[str, Any]]:
        latency_budget_ms, budget_record = self._lb_mcts_latency_budget()
        max_rollouts = min(
            max(1, int(self.light_explore_branch_budget)),
            max(1, int(getattr(self, "lb_mcts_max_rollouts_per_step", 12))),
        )
        parent_visits = max(
            1.0,
            sum(float(s.get("n") or 0.0) for s in self._search_policy_stats.values()) + 1.0,
        )
        records: list[dict[str, Any]] = []
        selected: list[dict[str, Any]] = []
        seen_centers: set[tuple[int, int]] = set()
        for idx, candidate in enumerate(candidates):
            components = candidate.get("score_components") if isinstance(candidate.get("score_components"), dict) else {}
            candidate["quality_filter_reason"] = _clean_text(
                candidate.get("quality_filter_reason")
                or components.get("quality_filter_reason")
                or self._candidate_quality_filter_reason(candidate, goal)
            )
            operator_name = self._lb_mcts_operator_name(candidate, goal)
            pattern_tuple = self._lb_mcts_pattern_tuple(candidate, goal, root_state, root_activity, root_labels)
            pattern_prior = float(pattern_tuple.get("probability") or components.get("PatternPrior") or 0.0)
            rollback_risk = max(0.0, min(1.0, float(components.get("RollbackRisk") or 0.0)))
            estimated_latency_ms = self._lb_mcts_estimated_latency_ms(candidate, goal)
            latency_cost_norm = max(0.0, min(1.0, estimated_latency_ms / float(latency_budget_ms or 1.0)))
            reward, reward_components = self._lb_mcts_reward_from_components(candidate, goal, latency_cost_norm)
            key = _clean_text(candidate.get("key") or candidate.get("merged") or candidate.get("label") or f"candidate_{idx}")
            stats = self._search_policy_stats.get(key, {})
            child_visits = float(stats.get("n") or 0.0)
            q_value = float(stats.get("q") or reward)
            uct = (
                q_value
                + float(self.lb_mcts_c_puct) * pattern_prior * math.sqrt(math.log(parent_visits + 1.0) / (child_visits + 1.0))
                - float(self.lb_mcts_lambda_risk) * rollback_risk
                - float(self.lb_mcts_lambda_latency) * latency_cost_norm
            )
            safe_to_execute = self._lb_mcts_speculation_level(candidate, goal) != "D" or operator_name in {"AvoidPath", "RiskWarning"}
            record = {
                "record_type": "MCTSSelectionRecord",
                "task_id": _clean_text(goal)[:200],
                "step": int(step_idx + 1),
                "node_id": f"{step_idx + 1}:root",
                "action_id": key,
                "operator": operator_name,
                "label": _clean_text(candidate.get("label") or candidate.get("merged")),
                "Q": float(q_value),
                "P": float(pattern_prior),
                "N_parent": float(parent_visits),
                "N_child": float(child_visits),
                "RollbackRisk": float(rollback_risk),
                "LatencyCostNorm": float(latency_cost_norm),
                "UCT": float(uct),
                "selected": False,
                "selected_rank": None,
                "reward_prior": float(reward),
                "reward_components": reward_components,
                "estimated_latency_ms": float(estimated_latency_ms),
                "safe_to_execute": bool(safe_to_execute),
                "speculation_level": self._lb_mcts_speculation_level(candidate, goal),
                "pattern_tuple": pattern_tuple,
            }
            candidate["mcts_uct"] = float(uct)
            candidate["mcts_q"] = float(q_value)
            candidate["mcts_pattern_prior"] = float(pattern_prior)
            candidate["mcts_latency_cost_norm"] = float(latency_cost_norm)
            candidate["mcts_estimated_latency_ms"] = float(estimated_latency_ms)
            candidate["mcts_reward_prior"] = float(reward)
            candidate["mcts_reward_components"] = reward_components
            candidate["search_strategy_score"] = float(uct)
            records.append(record)
            self._append_diagnostic_jsonl(
                goal,
                "pattern_tuple_records.jsonl",
                {
                    "record_type": "PatternTupleRecord",
                    "task_id": _clean_text(goal)[:200],
                    "step": int(step_idx + 1),
                    **pattern_tuple,
                    "context": {
                        "app_family": pattern_tuple.get("app"),
                        "task_mode": pattern_tuple.get("task_mode"),
                        "screen_role": pattern_tuple.get("screen_role"),
                        "current_visible_anchors": root_labels[:12],
                        "rollback_status": "eligible",
                    },
                    "matched": bool(pattern_prior >= 0.50),
                    "used_by_mcts": True,
                    "evidence_result": "",
                },
            )
        ranked = sorted(records, key=lambda r: float(r.get("UCT") or 0.0), reverse=True)
        by_action_id = {
            _clean_text(c.get("key") or c.get("merged") or c.get("label") or f"candidate_{idx}"): c
            for idx, c in enumerate(candidates)
        }
        for record in ranked:
            if len(selected) >= max_rollouts:
                break
            candidate = by_action_id.get(_clean_text(record.get("action_id")))
            if not candidate:
                continue
            center = candidate.get("center")
            center_key = tuple(center) if isinstance(center, (list, tuple)) and len(center) >= 2 else (id(candidate), 0)
            if center_key in seen_centers:
                continue
            seen_centers.add(center_key)
            candidate["reason_selected"] = "lb_mcts_uct_latency_bounded"
            selected.append(candidate)
            record["selected"] = True
            record["selected_rank"] = len(selected)
        selected_ids = {
            _clean_text(c.get("key") or c.get("merged") or c.get("label"))
            for c in selected
        }
        for candidate in candidates:
            key = _clean_text(candidate.get("key") or candidate.get("merged") or candidate.get("label"))
            if key not in selected_ids:
                candidate["reason_rejected"] = "not_selected_by_lb_mcts_budget_or_uct"
        for record in records:
            self._append_diagnostic_jsonl(goal, "mcts_selection_records.jsonl", record)
        trace["latency_budget_ms"] = float(latency_budget_ms)
        trace["expected_reasoning_latency_ms"] = float(budget_record["expected_reasoning_latency_ms"])
        trace["mcts_actual_rollouts"] = int(len(selected))
        trace["mcts_selection_record_count"] = int(len(records))
        trace["mcts_stopped_by_latency_budget"] = False
        budget_record.update(
            {
                "record_type": "LatencyBudgetRecord",
                "task_id": _clean_text(goal)[:200],
                "step": int(step_idx + 1),
                "elapsed_exploration_ms": 0.0,
                "actual_rollouts": int(len(selected)),
                "depth2_attempted": False,
                "stopped_by_latency_budget": False,
                "stopped_by_rollback_failure": False,
                "stopped_by_no_safe_action": not bool(selected),
            }
        )
        self._lb_mcts_latency_budget_records[int(step_idx + 1)] = dict(budget_record)
        self._append_diagnostic_jsonl(goal, "latency_budget_records.jsonl", budget_record)
        return selected

    def _prepare_search_strategy_candidates(
        self,
        candidates: list[dict[str, Any]],
        goal: str,
    ) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for candidate in candidates:
            operator, operator_reason = self._candidate_operator(candidate, goal)
            candidate["operator"] = operator
            candidate["operator_reason"] = operator_reason
            candidate["risk_boundary"] = operator == "RiskBoundary"
            candidate["layout_region"] = self._candidate_layout_region(candidate)
            candidate["semantic_role"] = self._candidate_semantic_role(candidate, goal)
            candidate["estimated_evidence_gain"] = float(self._estimate_candidate_evidence_gain(candidate, goal))
            candidate["search_strategy_score"] = float(self._search_strategy_candidate_score(candidate, goal))
            enriched.append(candidate)
        if self.light_explore_search_strategy == "stratified_bfs":
            groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
            for candidate in enriched:
                key = (
                    _clean_text(candidate.get("operator")),
                    _clean_text(candidate.get("layout_region")),
                    _clean_text(candidate.get("semantic_role")),
                )
                groups.setdefault(key, []).append(candidate)
            for group in groups.values():
                group.sort(key=lambda c: float(c.get("search_strategy_score") or 0.0), reverse=True)
            ordered: list[dict[str, Any]] = []
            group_keys = sorted(
                groups,
                key=lambda key: float(groups[key][0].get("search_strategy_score") or 0.0),
                reverse=True,
            )
            while group_keys:
                next_keys: list[tuple[str, str, str]] = []
                for key in group_keys:
                    group = groups.get(key) or []
                    if not group:
                        continue
                    item = group.pop(0)
                    item["strategy_group"] = "|".join(key)
                    ordered.append(item)
                    if group:
                        next_keys.append(key)
                group_keys = next_keys
            return ordered
        enriched.sort(
            key=lambda c: (
                1 if bool(c.get("is_planned_action")) else 0,
                float(c.get("search_strategy_score") or 0.0),
                float(c.get("score") or 0.0),
            ),
            reverse=True,
        )
        return enriched

    def _search_strategy_depth_limit(self, task_mode: str) -> int:
        configured = max(1, int(self.light_explore_branch_depth))
        strategy = self.light_explore_search_strategy
        if strategy == "value_of_computation":
            active_budget = getattr(self, "_active_voc_depth_budget", None)
            if active_budget is not None:
                try:
                    return max(1, min(configured, int(active_budget)))
                except (TypeError, ValueError):
                    return configured
        if self.light_explore_fixed_framework:
            if self.light_explore_slot_complete:
                slots = slot_complete_evidence.parse_task_slots(getattr(self, "_current_goal_for_depth", ""))
                if (
                    strategy == "best_first"
                    and slots.task_subtype == "ACTIVITY_STATS"
                    and slots.metric_type in {"duration", "distance", "count", "activity_type"}
                ):
                    return min(configured, 3)
                if task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH", "FORM_CREATE_EDIT"}:
                    return min(configured, 2)
                return 1
            if self.light_explore_safe_mcts and strategy == "mcts" and task_mode in {
                "INFO_QUERY",
                "NAVIGATION_SEARCH",
            }:
                return min(configured, 3)
            if strategy in {"iddfs", "beam", "mcts"} and task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH"}:
                return min(configured, 3)
            if strategy in {"stratified_bfs", "best_first", "greedy"} and task_mode in {
                "INFO_QUERY",
                "NAVIGATION_SEARCH",
                "FORM_CREATE_EDIT",
            }:
                return min(configured, 2)
            return 1
        if strategy == "stratified_bfs":
            return min(configured, 2)
        if strategy in {"iddfs", "beam", "mcts", "best_first"} and task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH"}:
            return min(configured, 3)
        if strategy in {"iddfs", "beam", "mcts", "best_first"}:
            return min(configured, 2)
        return configured

    def _assess_branch_evidence(
        self,
        *,
        goal: str,
        root_labels: list[str],
        observed_elements: list[str],
        candidate: dict[str, Any],
        operator: str,
        changed: bool,
        depth: int,
        rollback_success: bool = True,
        after_activity: str = "",
    ) -> dict[str, Any]:
        task_mode = self._task_mode(goal)
        root_set = {_clean_text(x).lower() for x in root_labels if _clean_text(x)}
        observed = [_clean_text(x) for x in observed_elements if _clean_text(x)]
        observed_set = {_clean_text(x).lower() for x in observed}
        slot_evidence: dict[str, Any] = {}
        if self.light_explore_slot_complete:
            slot_evidence = slot_complete_evidence.extract_slot_evidence(
                observed,
                goal,
                activity=after_activity,
            )
        new_labels = sorted(x for x in observed_set - root_set if x)
        disappeared_labels = sorted(x for x in root_set - observed_set if x)[:12]
        all_observed = " ".join(observed_set)
        entities = self._task_entities(goal)
        new_task_entities = [entity for entity in entities if entity and entity in all_observed]
        label = _clean_text(candidate.get("label") or candidate.get("merged"))
        semantic_close = self._lexical_overlap_score(
            merged=label,
            goal_tokens=self._goal_tokens(goal),
            app_keywords=self._goal_app_keywords(goal),
        ) > 0.05
        query_facts = self._extract_answer_facts_from_labels(observed, goal, limit=8)
        hard_negative_fact = any(_clean_text(x).lower().startswith("avoid ") for x in query_facts)
        schema_labels = [
            x
            for x in observed
            if re.search(r"\b(name|title|date|time|phone|email|amount|description|note|folder|file|field|search|filter)\b", x.lower())
        ]
        useful_actions = [
            x
            for x in observed
            if re.search(r"\b(open|search|filter|next|add|create|edit|save|done|select|folder|note|recipe|task)\b", x.lower())
        ]
        off_task = any(
            token in all_observed
            for token in ("youtube", "google account", "update your app", "launcher", "resolver", "assistant")
        )
        gain = 0.0
        if new_task_entities:
            gain += 4.0
        if task_mode == "INFO_QUERY" and query_facts:
            gain += 3.0
        if self.light_explore_slot_complete and slot_evidence:
            coverage = float(slot_evidence.get("slot_coverage") or 0.0)
            gain += 4.0 * coverage
            if slot_evidence.get("slot_complete"):
                gain += 4.0
            if slot_evidence.get("wrong_screen_role"):
                gain -= 3.0
        if useful_actions and operator in {"NavigationPeek", "SearchPeek", "FilterPeek"}:
            gain += 3.0
        if task_mode == "FORM_CREATE_EDIT" and schema_labels:
            gain += 3.0
        if off_task and semantic_close:
            gain += 2.0
        if new_labels:
            gain += min(3.0, float(len(new_labels)) * 0.4)
        if changed and operator in {"NavigationPeek", "SearchPeek", "FilterPeek"}:
            gain += 2.0
        if off_task:
            gain -= 2.0
        if not changed and not new_labels:
            gain -= 2.0
        if operator == "RiskBoundary" or self._is_transaction_unsafe_candidate(candidate):
            gain -= 3.0
        if not rollback_success:
            gain -= 4.0

        boundary_type = "NONE"
        stop_reason = "continue"
        evidence_type = "NONE"
        if self.light_explore_slot_complete and slot_evidence.get("hint_type") == "AVOID_HINT":
            label_text = _clean_text(candidate.get("label") or candidate.get("merged")).lower()
            partial_progress = bool(
                operator in {"SearchPeek", "FilterPeek", "NavigationPeek"}
                and re.search(r"\b(search|filter|stats|statistics|detail)\b", label_text)
                and not hard_negative_fact
                and not off_task
            )
            if partial_progress:
                boundary_type = "CHOICE_BOUNDARY"
                stop_reason = "partial_progress_not_hard_negative"
                evidence_type = "ACTION_HINT" if not self.light_explore_no_action_hint else "NONE"
            else:
                boundary_type = "NEGATIVE_BOUNDARY"
                stop_reason = "slot_complete_hard_negative"
                evidence_type = "AVOID_HINT"
        elif operator == "RiskBoundary":
            boundary_type = "RISK_BOUNDARY"
            stop_reason = "risk_boundary_not_executed"
            evidence_type = "RISK_HINT"
        elif hard_negative_fact:
            boundary_type = "NEGATIVE_BOUNDARY"
            stop_reason = "hard_negative_domain_rule"
            evidence_type = "AVOID_HINT"
        elif off_task and semantic_close:
            boundary_type = "NEGATIVE_BOUNDARY"
            stop_reason = "hard_negative_irrelevant_screen"
            evidence_type = "AVOID_HINT"
        elif self.light_explore_slot_complete and slot_evidence.get("hint_type") == "ANSWER_HINT":
            boundary_type = "TARGET_BOUNDARY"
            stop_reason = "slot_complete_answer_fact_observed"
            evidence_type = "ANSWER_HINT"
        elif self.light_explore_slot_complete and slot_evidence:
            # In slot-complete mode, partial matches such as seeing the target
            # note title are useful for search scoring but are not answer
            # evidence. Keep them in traces only so metrics and prompts do not
            # treat incomplete facts as successful ANSWER_HINTs.
            if float(slot_evidence.get("slot_coverage") or 0.0) > 0.0:
                boundary_type = "TARGET_BOUNDARY"
                stop_reason = "slot_partial_evidence_observed"
            else:
                boundary_type = "NONE"
                stop_reason = "slot_no_complete_evidence"
            evidence_type = "NONE"
        elif new_task_entities or (task_mode == "INFO_QUERY" and query_facts):
            boundary_type = "TARGET_BOUNDARY"
            stop_reason = "target_entity_or_answer_fact_observed"
            evidence_type = "ANSWER_HINT" if task_mode == "INFO_QUERY" else "ACTION_HINT"
        elif task_mode == "FORM_CREATE_EDIT" and schema_labels:
            boundary_type = "SCHEMA_BOUNDARY"
            stop_reason = "form_schema_observed"
            evidence_type = "SCHEMA_HINT"
        elif operator in {"FormSchema", "SearchPeek", "FilterPeek"} and schema_labels:
            boundary_type = "SCHEMA_BOUNDARY"
            stop_reason = "search_or_form_schema_observed"
            evidence_type = "SCHEMA_HINT"
        elif useful_actions and changed and not self.light_explore_no_action_hint:
            boundary_type = "SHORTCUT_BOUNDARY"
            stop_reason = "useful_next_action_observed"
            evidence_type = "ACTION_HINT"
        elif changed and new_labels:
            boundary_type = "CHOICE_BOUNDARY" if operator in {"NavigationPeek", "SearchPeek", "FilterPeek"} else "TARGET_BOUNDARY"
            stop_reason = "new_relevant_labels_observed"
            evidence_type = "ACTION_HINT" if operator in {"NavigationPeek", "SearchPeek", "FilterPeek"} and not self.light_explore_no_action_hint else "NONE"
        elif not changed:
            boundary_type = "DEAD_END"
            stop_reason = "no_meaningful_ui_change"
        confidence = max(0.0, min(1.0, 0.45 + (gain / 10.0)))
        return {
            "evidence_gain": float(gain),
            "confidence": float(confidence),
            "evidence_type": evidence_type,
            "boundary_type": boundary_type,
            "stop_reason": stop_reason,
            "new_labels": new_labels[:20],
            "disappeared_labels": disappeared_labels,
            "new_task_entities": new_task_entities[:10],
            "new_goal_actions": useful_actions[:10],
            "new_widget_types": [],
            "changed_checked_states": [],
            "query_facts": query_facts[:8],
            "slot_evidence": slot_evidence,
            "schema_labels": schema_labels[:8],
            "off_task": bool(off_task),
            "semantic_close": bool(semantic_close),
            "depth": int(depth),
        }

    def _should_expand_for_search_strategy(
        self,
        assessment: dict[str, Any],
        *,
        branch_operator: str,
        depth: int,
        task_mode: str,
        branch_index: int,
    ) -> bool:
        max_depth = self._search_strategy_depth_limit(task_mode)
        if depth >= max_depth:
            return False
        boundary_type = _clean_text(assessment.get("boundary_type"))
        if boundary_type in {"RISK_BOUNDARY", "NEGATIVE_BOUNDARY", "DEAD_END"}:
            return False
        gain = float(assessment.get("evidence_gain") or 0.0)
        strategy = self.light_explore_search_strategy
        if self.light_explore_slot_complete:
            slot_evidence = assessment.get("slot_evidence") if isinstance(assessment.get("slot_evidence"), dict) else {}
            slots = slot_evidence.get("task_slots") if isinstance(slot_evidence.get("task_slots"), dict) else {}
            missing = set(slot_evidence.get("missing_slots") or [])
            coverage = float(slot_evidence.get("slot_coverage") or 0.0)
            if slot_evidence.get("slot_complete"):
                return False
            if strategy == "best_first":
                if (
                    depth == 2
                    and slots.get("task_subtype") == "ACTIVITY_STATS"
                    and "metric_value" in missing
                    and branch_operator in {"NavigationPeek", "SearchPeek", "FilterPeek", "ListInspect"}
                    and float(assessment.get("evidence_gain_delta") or 0.0) > 0.0
                ):
                    return True
                return bool(
                    depth == 1
                    and branch_operator in {"NavigationPeek", "SearchPeek", "FilterPeek", "ListInspect"}
                    and (coverage > 0.0 or boundary_type in {"CHOICE_BOUNDARY", "SHORTCUT_BOUNDARY", "SCHEMA_BOUNDARY"})
                )
            return bool(depth == 1 and coverage > 0.0 and branch_operator in {"NavigationPeek", "SearchPeek", "FilterPeek", "ListInspect"})
        if strategy == "stratified_bfs":
            return depth == 1 and boundary_type in {"SHORTCUT_BOUNDARY", "SCHEMA_BOUNDARY", "CHOICE_BOUNDARY"}
        if strategy == "iddfs":
            if depth == 1:
                return gain >= 1.0 and branch_operator in {"NavigationPeek", "SearchPeek", "FilterPeek", "ListInspect"}
            return task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH"} and gain >= 2.0
        if strategy == "best_first":
            return gain >= 1.5 and branch_operator in {"NavigationPeek", "SearchPeek", "FilterPeek", "ListInspect"}
        if strategy == "beam":
            return gain >= 1.0 and (branch_index <= 2 or task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH"})
        if strategy == "mcts":
            if self.light_explore_safe_mcts:
                if branch_operator not in {"NavigationPeek", "SearchPeek", "FilterPeek", "ListInspect"}:
                    return False
                if depth == 1:
                    return boundary_type in {
                        "TARGET_BOUNDARY",
                        "ANSWER_BOUNDARY",
                        "SCHEMA_BOUNDARY",
                        "CHOICE_BOUNDARY",
                        "SHORTCUT_BOUNDARY",
                    } and gain >= 1.0
                if depth == 2:
                    recent_failures = sum(
                        1
                        for item in list(self._rollback_traces)[-2:]
                        if isinstance(item, dict) and not bool(item.get("success"))
                    )
                    gain_delta = float(assessment.get("evidence_gain_delta") or 0.0)
                    return bool(
                        task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH"}
                        and recent_failures == 0
                        and gain_delta > 0.0
                        and boundary_type not in {"RISK_BOUNDARY", "NEGATIVE_BOUNDARY", "DEAD_END"}
                    )
                return False
            return gain >= 0.5 and branch_operator in {"NavigationPeek", "SearchPeek", "FilterPeek", "ListInspect"}
        return bool(
            depth < max_depth
            and branch_operator in {"NavigationPeek", "SearchPeek", "FilterPeek"}
            and task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH", "FORM_CREATE_EDIT"}
        )

    def _state_capsule_summary(self, state: Any, activity: str = "") -> dict[str, Any]:
        labels = self._state_semantic_summary(state, limit=24)
        salient = self._informative_observed_elements(labels)[:12]
        signature_text = "|".join(_clean_text(x).lower() for x in salient)
        return {
            "package": _clean_text(getattr(getattr(state, "foreground_activity", None), "package_name", "")),
            "activity": _clean_text(activity or self._foreground_activity_name()),
            "screen_role": self._infer_screen_role_from_labels(salient),
            "salient_labels": salient,
            "layout_summary": f"{len(list(getattr(state, 'ui_elements', None) or []))} ui elements",
            "p_hash": self._state_hash(state),
            "tree_signature": hashlib.sha1(signature_text.encode("utf-8", errors="ignore")).hexdigest()[:16],
        }

    @staticmethod
    def _infer_screen_role_from_labels(labels: list[str]) -> str:
        merged = " ".join(_clean_text(x).lower() for x in labels)
        if re.search(r"\b(search|filter|sort)\b", merged):
            return "search_or_filter"
        if re.search(r"\b(save|done|name|title|description|email|phone)\b", merged):
            return "form_or_editor"
        if re.search(r"\b(delete|remove|confirm|allow|cancel)\b", merged):
            return "risk_or_confirmation"
        if len(labels) >= 8:
            return "list_or_collection"
        return "general"

    def _build_evidence_capsule(
        self,
        *,
        goal: str,
        strategy_id: str,
        step_id: int,
        branch_id: int,
        depth: int,
        parent_state: Any,
        parent_activity: str,
        child_state: Any,
        child_activity: str,
        candidate: dict[str, Any],
        assessment: dict[str, Any],
        rollback_info: dict[str, Any],
    ) -> dict[str, Any]:
        action_trace = self._candidate_trace(candidate)
        return {
            "strategy_id": strategy_id,
            "task_id": _clean_text(goal)[:160],
            "step_id": int(step_id),
            "branch_id": int(branch_id),
            "parent_state": self._state_capsule_summary(parent_state, parent_activity),
            "action": {
                "operator": action_trace.get("operator"),
                "action_type": action_trace.get("action_kind") or "click",
                "label": action_trace.get("label"),
                "resource_id": (action_trace.get("a11y") or {}).get("resource_id"),
                "class_name": (action_trace.get("a11y") or {}).get("class_name"),
                "bbox": (action_trace.get("a11y") or {}).get("bbox"),
                "risk_level": "high" if action_trace.get("risk_boundary") else "low",
            },
            "child_state": self._state_capsule_summary(child_state, child_activity),
            "delta": {
                "new_labels": list(assessment.get("new_labels") or []),
                "disappeared_labels": list(assessment.get("disappeared_labels") or []),
                "new_task_entities": list(assessment.get("new_task_entities") or []),
                "new_goal_actions": list(assessment.get("new_goal_actions") or []),
                "new_widget_types": list(assessment.get("new_widget_types") or []),
                "changed_checked_states": list(assessment.get("changed_checked_states") or []),
            },
            "evidence": {
                "type": assessment.get("evidence_type") or "NONE",
                "evidence_gain": float(assessment.get("evidence_gain") or 0.0),
                "confidence": float(assessment.get("confidence") or 0.0),
                "depth": int(depth),
                "rollback_verified": bool((rollback_info or {}).get("success")),
                "boundary_type": assessment.get("boundary_type") or "NONE",
                "stop_reason": assessment.get("stop_reason") or "",
                "slot_evidence": assessment.get("slot_evidence") or {},
            },
            "validity": {
                "root_action_match_required": True,
                "child_state_match_required": int(depth) >= 2,
                "current_screen_anchor_labels": self._informative_observed_elements(
                    self._state_semantic_summary(parent_state, limit=16)
                )[:8],
            },
        }

    def _update_search_strategy_stats(self, observation: dict[str, Any]) -> None:
        steps = [s for s in list(observation.get("steps") or []) if isinstance(s, dict)]
        if not steps:
            return
        reward = float(observation.get("evidence_gain") or observation.get("score") or 0.0)
        if not bool((observation.get("rollback") or {}).get("success")):
            reward -= 4.0
        for step in steps:
            cand = step.get("candidate") if isinstance(step.get("candidate"), dict) else {}
            key = _clean_text(cand.get("key") or cand.get("merged") or cand.get("label"))
            if not key:
                continue
            stats = self._search_policy_stats.setdefault(key, {"n": 0.0, "q": 0.0, "risk": 0.0})
            n = float(stats.get("n") or 0.0)
            q = float(stats.get("q") or 0.0)
            risk = 1.0 if observation.get("boundary_type") == "RISK_BOUNDARY" else 0.0
            stats["n"] = n + 1.0
            stats["q"] = ((q * n) + reward) / (n + 1.0)
            stats["risk"] = ((float(stats.get("risk") or 0.0) * n) + risk) / (n + 1.0)

    @staticmethod
    def _is_query_fact_label(label: str, goal: str) -> bool:
        low = _clean_text(label).lower()
        if not low:
            return False
        goal_low = _clean_text(goal).lower()
        if re.search(r"\d", low):
            return True
        if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|mon|tue|wed|thu|fri|sat|sun)\b", low):
            return True
        if any(token in low for token in ("mile", "meter", "minute", "hour", "cup", "tbsp", "tsp", "km", "mi")):
            return True
        goal_tokens = set(re.findall(r"[a-z0-9]{3,}", goal_low))
        label_tokens = set(re.findall(r"[a-z0-9]{3,}", low))
        return bool(goal_tokens.intersection(label_tokens)) and len(low) >= 4

    def _extract_answer_facts_from_labels(
        self,
        labels: list[str],
        goal: str,
        *,
        limit: int = 8,
    ) -> list[str]:
        """Extract compact answer facts from visible labels without executing UI actions."""
        cleaned = [_clean_text(x) for x in labels if _clean_text(x)]
        if not cleaned:
            return []
        if self.light_explore_slot_complete:
            slot_evidence = slot_complete_evidence.extract_slot_evidence(cleaned, goal)
            if slot_evidence.get("hint_type") in {"ANSWER_HINT", "AVOID_HINT"}:
                return list(slot_evidence.get("facts") or [])[:limit]
            return []
        if self.light_explore_answer_extractors:
            strict = self._extract_answer_facts_with_domain_rules(cleaned, goal, limit=limit)
            if strict:
                return strict[:limit]
        facts: list[str] = []
        seen: set[str] = set()
        goal_low = _clean_text(goal).lower()
        wants_titles = bool(re.search(r"\b(title|titles|event|events|meeting|meetings)\b", goal_low))
        measurement_re = re.compile(
            r"\b(\d+(?:[.:/]\d+)?(?:\s*[-–]\s*\d+(?:[.:/]\d+)?)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
            r"(cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons|g|gram|grams|kg|ml|oz|ounce|ounces|"
            r"hour|hours|hr|hrs|min|mins|minute|minutes|km|mi|mile|miles|meter|meters)\b",
            re.IGNORECASE,
        )
        time_range_re = re.compile(
            r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\s*[-–]\s*\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b",
            re.IGNORECASE,
        )

        def add_fact(value: str) -> None:
            fact = _clean_text(value)
            if not fact:
                return
            key = fact.lower()
            if key in seen:
                return
            seen.add(key)
            facts.append(fact)

        entities = [x for x in self._task_entities(goal) if x]
        for idx, label in enumerate(cleaned):
            low = label.lower()
            prev_label = cleaned[idx - 1] if idx > 0 else ""
            next_label = cleaned[idx + 1] if idx + 1 < len(cleaned) else ""
            if time_range_re.search(label) and wants_titles and prev_label:
                add_fact(f"{prev_label}: {label}")
                continue
            if measurement_re.search(label):
                if prev_label and not measurement_re.search(prev_label):
                    add_fact(f"{prev_label}: {label}")
                else:
                    add_fact(label)
                continue
            if any(entity in low for entity in entities):
                if next_label and (measurement_re.search(next_label) or time_range_re.search(next_label)):
                    add_fact(f"{label}: {next_label}")
                else:
                    add_fact(label)
                continue
            if self._is_query_fact_label(label, goal):
                add_fact(label)
            if len(facts) >= limit:
                break
        return facts[:limit]

    def _extract_answer_facts_with_domain_rules(
        self,
        labels: list[str],
        goal: str,
        *,
        limit: int = 8,
    ) -> list[str]:
        goal_low = _clean_text(goal).lower()
        labels = [_clean_text(x) for x in labels if _clean_text(x)]
        facts: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            value = _clean_text(value)
            if not value:
                return
            key = value.lower()
            if key not in seen:
                seen.add(key)
                facts.append(value)

        if re.search(r"\b(recipe|ingredient|quantity|amount|unit)\b", goal_low):
            for fact in self._extract_recipe_ingredient_facts(labels, goal):
                add(fact)
            if facts:
                return facts[:limit]
            # For recipe amount tasks, a visible recipe title alone is not an answer.
            if re.search(r"\b(quantity|amount|unit)\b", goal_low):
                return []

        if "simple calendar" in goal_low or re.search(r"\b(event|events|meeting|meetings)\b", goal_low):
            for fact in self._extract_calendar_event_facts(labels, goal):
                add(fact)
            if facts:
                return facts[:limit]

        if "opentracks" in goal_low or re.search(r"\b(activity|activities|duration|distance|running|skiing|kayaking)\b", goal_low):
            for fact in self._extract_opentracks_facts(labels, goal):
                add(fact)
            if facts:
                return facts[:limit]

        for fact in self._extract_generic_count_facts(labels, goal):
            add(fact)
        return facts[:limit]

    @staticmethod
    def _quoted_phrases(text: str) -> list[str]:
        return [_clean_text(x) for x in re.findall(r"['\"]([^'\"]+)['\"]", text) if _clean_text(x)]

    @staticmethod
    def _amount_unit_re() -> re.Pattern[str]:
        return re.compile(
            r"(?P<amount>(?:\d+\s*/\s*\d+)|(?:\d+(?:\.\d+)?)|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s*"
            r"(?P<unit>tsp|teaspoons?|tbsp|tablespoons?|cups?|oz|ounces?|g|grams?|kg|ml|l|pinch|cloves?)\b",
            re.IGNORECASE,
        )

    def _extract_recipe_ingredient_facts(self, labels: list[str], goal: str) -> list[str]:
        goal_low = _clean_text(goal).lower()
        quoted = self._quoted_phrases(goal)
        recipe = quoted[0] if quoted else ""
        ingredient_match = re.search(r"\b(?:quantity|amount)\s+of\s+([a-zA-Z][a-zA-Z0-9 _-]{1,40})\b", goal_low)
        ingredient = _clean_text(ingredient_match.group(1) if ingredient_match else "")
        ingredient = re.sub(r"\b(do|i|need|for|the|recipe|in|app)\b.*$", "", ingredient).strip()
        if not ingredient:
            for token in self._goal_tokens(goal):
                if token not in {"quantity", "amount", "recipe", "ingredient", "express", "answer", "format", "unit", "joplin"}:
                    ingredient = token
                    break
        amount_re = self._amount_unit_re()
        facts: list[str] = []
        for idx, label in enumerate(labels):
            window = " ".join(labels[max(0, idx - 2) : min(len(labels), idx + 3)])
            low_window = window.lower()
            if ingredient and ingredient.lower() not in low_window:
                continue
            amount = amount_re.search(window)
            if amount:
                amount_text = f"{amount.group('amount')} {amount.group('unit')}"
                prefix = f'In "{recipe}", ' if recipe else ""
                facts.append(f'{prefix}ingredient "{ingredient}" appears with amount "{amount_text}".')
        return facts[:4]

    @staticmethod
    def _time_to_minutes(text: str) -> int | None:
        match = re.search(r"\b(\d{1,2})[:.](\d{2})\s*(am|pm)?\b", text.lower())
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2))
        suffix = match.group(3)
        if suffix == "pm" and hour < 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0
        return hour * 60 + minute

    def _extract_calendar_event_facts(self, labels: list[str], goal: str) -> list[str]:
        goal_low = _clean_text(goal).lower()
        requested_times = re.findall(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b", goal_low)
        start_req = self._time_to_minutes(requested_times[0]) if requested_times else None
        end_req = self._time_to_minutes(requested_times[1]) if len(requested_times) > 1 else None
        person_names = [
            token
            for token in re.findall(r"\b[A-Z][a-z]{2,}\b", goal)
            if token.lower() not in {"Simple", "Calendar", "Pro", "Answer", "Express", "October"}
        ]
        quoted = self._quoted_phrases(goal)
        title_entities = [x for x in quoted + person_names if _clean_text(x)]
        time_range_re = re.compile(
            r"\b(\d{1,2}[:.]\d{2}\s*(?:am|pm)?)\s*[-–]\s*(\d{1,2}[:.]\d{2}\s*(?:am|pm)?)\b",
            re.IGNORECASE,
        )
        single_time_re = re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)?\b", re.IGNORECASE)
        facts: list[str] = []
        for idx, label in enumerate(labels):
            match = time_range_re.search(label)
            window_labels = labels[max(0, idx - 3) : min(len(labels), idx + 4)]
            window = " ".join(window_labels)
            title = labels[idx - 1] if idx > 0 else ""
            if not title or re.search(r"\b(october|november|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", title.lower()):
                title = labels[idx - 2] if idx > 1 else title
            if match:
                start = self._time_to_minutes(match.group(1))
                end = self._time_to_minutes(match.group(2))
                in_range = True
                if start_req is not None and end_req is not None and start is not None and end is not None:
                    in_range = bool(start < end_req and end > start_req)
                entity_ok = not title_entities or any(entity.lower() in window.lower() for entity in title_entities)
                if in_range and entity_ok and title:
                    facts.append(f"Calendar event: {title}, time {match.group(0)}.")
                continue
            if title_entities and any(entity.lower() in window.lower() for entity in title_entities):
                times = single_time_re.findall(window)
                if times:
                    facts.append(f"Calendar matching entity: {', '.join(title_entities[:3])}; nearby time(s): {', '.join(times[:3])}; context: {window}.")
        if not facts and re.search(r"\bhow many|count\b", goal_low):
            item_titles = [
                x
                for x in labels
                if len(x) >= 3 and not re.search(r"\b(search|settings|more options|october|calendar)\b", x.lower())
            ]
            if item_titles:
                facts.append(f"Visible event count: {len(item_titles)}; items: {', '.join(item_titles[:6])}.")
        return facts[:6]

    def _extract_opentracks_facts(self, labels: list[str], goal: str) -> list[str]:
        goal_low = _clean_text(goal).lower()
        if "markers" in " ".join(labels).lower() and "marker" not in goal_low:
            return ['Avoid "Markers": it is not useful for duration/distance/activity statistics.']
        category_tokens = [
            token
            for token in ("skiing", "kayaking", "running", "walking", "cycling", "rowing", "hiking")
            if token in goal_low
        ]
        if "activity type" in goal_low or re.search(r"\bwhat activities\b", goal_low):
            category_tokens = category_tokens or ["skiing", "kayaking", "running", "walking", "cycling", "rowing", "hiking"]
        date_terms = set(re.findall(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|\b\d{1,2}\b|\b20\d{2}\b", goal_low))
        weekday_terms = set(
            token
            for token in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "yesterday")
            if token in goal_low
        )
        metric_patterns = [
            r"\b\d+(?:\.\d+)?\s*(?:min|mins|minute|minutes|h|hr|hrs|hour|hours)\b",
            r"\b\d+(?:\.\d+)?\s*(?:km|mi|mile|miles|m|meters?)\b",
            r"\b\d{1,2}[:.]\d{2}(?::\d{2})?\b",
        ]
        duration_re = re.compile(metric_patterns[0] + r"|\b\d{1,2}[:.]\d{2}(?::\d{2})?\b", re.IGNORECASE)
        distance_re = re.compile(metric_patterns[1], re.IGNORECASE)
        facts: list[str] = []
        for idx, label in enumerate(labels):
            low = label.lower()
            has_category = any(cat in low for cat in category_tokens) if category_tokens else False
            has_metric = any(re.search(pattern, low, re.IGNORECASE) for pattern in metric_patterns)
            window = " ".join(labels[max(0, idx - 4) : min(len(labels), idx + 5)])
            window_low = window.lower()
            window_has_category = any(cat in window_low for cat in category_tokens) if category_tokens else False
            window_has_date = any(term in window_low for term in date_terms) or any(term in window_low for term in weekday_terms)
            duration = duration_re.search(window)
            distance = distance_re.search(window)
            if re.search(r"\bduration|how long|minutes?\b", goal_low) and window_has_category and duration:
                facts.append(f"OpenTracks duration fact: category/date context `{window}`; duration candidate `{duration.group(0)}`.")
                continue
            if "distance" in goal_low and window_has_category and distance:
                facts.append(f"OpenTracks distance fact: category/date context `{window}`; distance candidate `{distance.group(0)}`.")
                continue
            if re.search(r"\bwhat activities|activity type\b", goal_low) and window_has_date and window_has_category:
                matched = [cat for cat in category_tokens if cat in window_low]
                facts.append(f"OpenTracks activity type fact: requested date context `{window}`; activity type candidate `{', '.join(matched[:3])}`.")
                continue
            if re.search(r"\bhow many|count\b", goal_low) and window_has_date and window_has_category:
                matched = [cat for cat in category_tokens if cat in window_low]
                facts.append(f"OpenTracks count-related fact: requested interval context `{window}`; visible activity type(s): {', '.join(matched[:3])}.")
                continue
            if (has_category or any(cat in window_low for cat in category_tokens)) and any(
                re.search(pattern, window_low, re.IGNORECASE) for pattern in metric_patterns
            ):
                facts.append(f"OpenTracks activity fact: {window}.")
            elif has_metric and re.search(r"\b(duration|distance|total|time)\b", window_low):
                facts.append(f"OpenTracks metric fact: {window}.")
        return facts[:6]

    def _extract_generic_count_facts(self, labels: list[str], goal: str) -> list[str]:
        goal_low = _clean_text(goal).lower()
        if not re.search(r"\b(how many|count|which|what tasks|what events|activities)\b", goal_low):
            return []
        ignored = re.compile(r"\b(search|settings|more options|navigate up|toolbar|content|main|calendar)\b", re.IGNORECASE)
        items = [x for x in labels if len(x) >= 3 and not ignored.search(x)]
        if not items:
            return []
        return [f"Visible matching items: {', '.join(items[:8])}.", f"Visible count: {len(items)}."]

    @staticmethod
    def _normalize_terminal_answer_text(goal: str, answer_text: str) -> tuple[str, str]:
        """Extract the exact answer when the task specifies a strict format."""
        goal_low = _clean_text(goal).lower()
        text = _clean_text(answer_text)
        if not text:
            return text, ""
        if (
            "<amount> <unit>" in goal_low
            or "amount and unit" in goal_low
            or ("quantity" in goal_low and "unit" in goal_low)
        ):
            amount_unit = re.compile(
                r"\b((?:\d+\s*/\s*\d+)|(?:\d+(?:\.\d+)?)|(?:one|two|three|four|five|six|seven|eight|nine|ten))\s+"
                r"(tsp|teaspoons?|tbsp|tablespoons?|cups?|oz|ounces?|g|grams?|kg|ml|l|pinch|cloves?)\b",
                re.IGNORECASE,
            )
            matches = amount_unit.findall(text)
            if matches:
                amount, unit = matches[-1]
                return f"{_clean_text(amount)} {_clean_text(unit)}", "strict_amount_unit"
        if (
            "<month name> <day> <year> <hour in 24-hour format>:<minutes>" in goal_low
            or ("month name" in goal_low and "24-hour" in goal_low and "minutes" in goal_low)
        ):
            month_names = (
                "january|february|march|april|may|june|july|august|"
                "september|october|november|december"
            )
            date_time = re.compile(
                rf"\b({month_names})\s+(\d{{1,2}}),?\s+(\d{{4}})(?:,?\s+at)?\s+"
                r"(\d{1,2}):(\d{2})(?::\d{2})?\b",
                re.IGNORECASE,
            )
            matches = date_time.findall(text)
            if matches:
                month, day, year, hour, minute = matches[0]
                return f"{month.capitalize()} {int(day)} {year} {int(hour)}:{minute}", "strict_month_day_year_time"
        if "single integer" in goal_low:
            candidates = re.findall(r"\b\d+\b", text)
            if candidates:
                return candidates[-1], "strict_single_integer"
        return text, ""

    def _normalize_terminal_action_answer(
        self,
        *,
        goal: str,
        action: json_action.JSONAction,
        parsed_action: OrderedDict[str, Any],
        tool_call: dict[str, Any],
        extras: dict[str, Any],
    ) -> tuple[json_action.JSONAction, dict[str, Any], dict[str, Any], OrderedDict[str, Any]]:
        if action.action_type not in {json_action.STATUS, json_action.ANSWER}:
            return action, tool_call, extras, parsed_action
        raw_answer = _clean_text(
            extras.get("return_text")
            or getattr(action, "text", "")
            or parsed_action.get("return")
            or parsed_action.get("value")
        )
        normalized, reason = self._normalize_terminal_answer_text(goal, raw_answer)
        if not reason or not normalized or normalized == raw_answer:
            return action, tool_call, extras, parsed_action
        extras = dict(extras)
        tool_call = dict(tool_call or {})
        parsed_action = OrderedDict(parsed_action)
        extras["return_text"] = normalized
        extras["answer_normalization"] = reason
        parsed_action["answer_normalization"] = reason
        parsed_action["normalized_return"] = normalized
        if "return" in parsed_action:
            parsed_action["return"] = normalized
        elif "value" in parsed_action:
            parsed_action["value"] = normalized
        args = dict(tool_call.get("arguments") or {})
        if args.get("action") == "terminate":
            args["return_text"] = normalized
        elif args.get("action") in {"answer", "respond"}:
            args["text"] = normalized
            args["value"] = normalized
        tool_call["arguments"] = args
        if action.action_type == json_action.ANSWER:
            action = json_action.JSONAction(action_type=json_action.ANSWER, text=normalized)
        return action, tool_call, extras, parsed_action

    def _root_list_inspect_observation(
        self,
        *,
        goal: str,
        strategy_id: str,
        step_id: int,
        root_state: Any,
        root_activity: str,
        root_hash: int,
        root_screenshot: str,
        root_labels: list[str],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if self._task_mode(goal) != "INFO_QUERY":
            return None, None
        facts = self._extract_answer_facts_from_labels(root_labels, goal, limit=8)
        if not facts:
            return None, None
        is_hard_negative = any(_clean_text(x).lower().startswith("avoid ") for x in facts)
        slot_evidence = (
            slot_complete_evidence.extract_slot_evidence(root_labels, goal, activity=root_activity)
            if self.light_explore_slot_complete
            else {}
        )
        pseudo_candidate = {
            "index": -1,
            "label": "visible answer facts",
            "center": None,
            "score": 8.0,
            "relevance": 1.0,
            "planning_relevance": 1.0,
            "token_relevance": 1.0,
            "affordance_bonus": 0.0,
            "visits": 0,
            "key": f"root-list-inspect:{step_id}",
            "merged": "visible answer facts",
            "a11y": {},
            "is_planned_action": False,
            "planned_action_alignment": 0.0,
            "action_kind": "inspect",
            "text": "",
            "clear_text": False,
            "operator": "ListInspect",
            "operator_reason": "root_visible_answer_facts",
            "risk_boundary": False,
            "layout_region": "content_list",
            "semantic_role": "list_item",
            "estimated_evidence_gain": 3.0 + min(4.0, float(len(facts))),
            "search_strategy_score": 3.0 + min(4.0, float(len(facts))),
            "strategy_group": "ListInspect|content_list|answer_facts",
        }
        assessment = {
            "evidence_gain": 3.0 + min(4.0, float(len(facts))),
            "confidence": min(0.95, 0.62 + 0.04 * len(facts)),
            "evidence_type": "AVOID_HINT" if is_hard_negative else "ANSWER_HINT",
            "boundary_type": "NEGATIVE_BOUNDARY" if is_hard_negative else "TARGET_BOUNDARY",
            "stop_reason": "root_visible_hard_negative" if is_hard_negative else "root_visible_answer_facts",
            "new_labels": [],
            "disappeared_labels": [],
            "new_task_entities": [x for x in self._task_entities(goal) if x in " ".join(facts).lower()][:10],
            "new_goal_actions": [],
            "new_widget_types": [],
            "changed_checked_states": [],
            "query_facts": list(facts),
            "slot_evidence": slot_evidence,
            "schema_labels": [],
            "off_task": False,
            "semantic_close": True,
            "depth": 1,
        }
        rollback = {"success": True, "mode": "no_action_list_inspect", "level": "level0"}
        capsule = self._build_evidence_capsule(
            goal=goal,
            strategy_id=strategy_id,
            step_id=step_id,
            branch_id=0,
            depth=1,
            parent_state=root_state,
            parent_activity=root_activity,
            child_state=root_state,
            child_activity=root_activity,
            candidate=pseudo_candidate,
            assessment=assessment,
            rollback_info=rollback,
        )
        step = {
            "depth": 1,
            "candidate": self._candidate_trace(pseudo_candidate),
            "changed": True,
            "after_activity": root_activity,
            "after_hash": root_hash,
            "hash_diff_from_root": 0,
            "screenshot": root_screenshot,
            "observed_elements": list(facts),
            "a11y": [],
        }
        observation = {
            "branch_id": 0,
            "labels": ["visible answer facts"],
            "changed": True,
            "after_activity": root_activity,
            "score": float(pseudo_candidate["score"]),
            "depth_reached": 1,
            "observed_elements": list(facts),
            "steps": [step],
            "rollback": rollback,
            "operator": "ListInspect",
            "operator_reason": "root_visible_answer_facts",
            "layout_region": "content_list",
            "semantic_role": "list_item",
            "boundary_type": "NEGATIVE_BOUNDARY" if is_hard_negative else "TARGET_BOUNDARY",
            "stop_reason": "root_visible_hard_negative" if is_hard_negative else "root_visible_answer_facts",
            "evidence_type": "AVOID_HINT" if is_hard_negative else "ANSWER_HINT",
            "evidence_gain": float(assessment["evidence_gain"]),
            "confidence": float(assessment["confidence"]),
            "slot_evidence": slot_evidence,
            "delta": {
                "new_labels": [],
                "new_task_entities": list(assessment.get("new_task_entities") or []),
                "new_goal_actions": [],
            },
            "evidence_capsule": capsule,
            "no_action_inspect": True,
        }
        return observation, capsule

    def _passive_coverage_observations(
        self,
        *,
        goal: str,
        strategy_id: str,
        step_id: int,
        root_state: Any,
        root_activity: str,
        root_hash: int,
        root_screenshot: str,
        root_labels: list[str],
        start_branch_id: int,
        count: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if count <= 0:
            return [], []
        labels = [_clean_text(x) for x in root_labels if _clean_text(x)]
        if not labels:
            labels = ["passive page sample"]
        observations: list[dict[str, Any]] = []
        capsules: list[dict[str, Any]] = []
        window = max(1, min(6, len(labels)))
        for offset in range(count):
            branch_id = int(start_branch_id + offset)
            start = offset % len(labels)
            observed = [labels[(start + j) % len(labels)] for j in range(window)]
            pseudo_candidate = {
                "index": -1000 - offset,
                "label": f"passive coverage sample {offset + 1}",
                "center": None,
                "score": 0.0,
                "relevance": 0.0,
                "planning_relevance": 0.0,
                "token_relevance": 0.0,
                "affordance_bonus": 0.0,
                "visits": 0,
                "key": f"passive-coverage:{step_id}:{offset}",
                "merged": "passive coverage sample",
                "a11y": {},
                "is_planned_action": False,
                "planned_action_alignment": 0.0,
                "action_kind": "inspect",
                "text": "",
                "clear_text": False,
                "operator": "PassiveCoverage",
                "operator_reason": "min_attempts_padding_no_action",
                "risk_boundary": False,
                "layout_region": "root_page",
                "semantic_role": "passive_sample",
                "estimated_evidence_gain": 0.0,
                "search_strategy_score": 0.0,
                "strategy_group": "PassiveCoverage|root_page",
            }
            assessment = {
                "evidence_gain": 0.0,
                "confidence": 0.0,
                "evidence_type": "NONE",
                "boundary_type": "DEAD_END",
                "stop_reason": "passive_coverage_padding",
                "new_labels": [],
                "disappeared_labels": [],
                "new_task_entities": [],
                "new_goal_actions": [],
                "new_widget_types": [],
                "changed_checked_states": [],
                "query_facts": [],
                "slot_evidence": {},
                "schema_labels": [],
                "off_task": False,
                "semantic_close": False,
                "depth": 0,
            }
            rollback = {"success": True, "mode": "passive_no_action", "level": "level0"}
            capsule = self._build_evidence_capsule(
                goal=goal,
                strategy_id=strategy_id,
                step_id=step_id,
                branch_id=branch_id,
                depth=0,
                parent_state=root_state,
                parent_activity=root_activity,
                child_state=root_state,
                child_activity=root_activity,
                candidate=pseudo_candidate,
                assessment=assessment,
                rollback_info=rollback,
            )
            step = {
                "depth": 0,
                "candidate": self._candidate_trace(pseudo_candidate),
                "changed": False,
                "semantic_changed": False,
                "after_activity": root_activity,
                "after_hash": root_hash,
                "hash_diff_from_root": 0,
                "screenshot": root_screenshot,
                "observed_elements": list(observed),
                "a11y": [],
                "stop_reason": "passive_coverage_padding",
            }
            observation = {
                "branch_id": branch_id,
                "labels": [pseudo_candidate["label"]],
                "changed": False,
                "after_activity": root_activity,
                "score": 0.0,
                "depth_reached": 0,
                "observed_elements": list(observed),
                "steps": [step],
                "rollback": rollback,
                "operator": "PassiveCoverage",
                "operator_reason": "min_attempts_padding_no_action",
                "layout_region": "root_page",
                "semantic_role": "passive_sample",
                "boundary_type": "DEAD_END",
                "stop_reason": "passive_coverage_padding",
                "evidence_type": "NONE",
                "evidence_gain": 0.0,
                "confidence": 0.0,
                "slot_evidence": {},
                "delta": {"new_labels": [], "new_task_entities": [], "new_goal_actions": []},
                "evidence_capsule": capsule,
                "no_action_inspect": True,
                "passive_coverage_padding": True,
            }
            observations.append(observation)
            capsules.append(capsule)
        return observations, capsules

    def _candidate_visible_in_state(self, candidate: dict[str, Any], state: Any) -> bool:
        label = _clean_text(candidate.get("label") or candidate.get("merged")).lower()
        action_kind = _clean_text(candidate.get("action_kind")).lower()
        if action_kind in {"type", json_action.INPUT_TEXT} and self._state_has_search_ui(state):
            return True
        center = candidate.get("center")
        labels = [_clean_text(x).lower() for x in self._state_semantic_summary(state, limit=40)]
        if label and any(label in item or item in label for item in labels if item):
            return True
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            try:
                x, y = int(center[0]), int(center[1])
            except Exception:  # pylint: disable=broad-exception-caught
                x, y = -1, -1
            for element in list(getattr(state, "ui_elements", None) or []):
                bbox = getattr(element, "bbox_pixels", None)
                if bbox is None:
                    continue
                try:
                    if int(bbox.x_min) <= x <= int(bbox.x_max) and int(bbox.y_min) <= y <= int(bbox.y_max):
                        return True
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
        return False

    def _planned_action_precondition(
        self,
        current_state: Any,
        root_state: Any,
        root_activity: str,
        root_hash: int,
        action: json_action.JSONAction | None,
    ) -> tuple[bool, str]:
        if action is None:
            return False, "no_planned_action"
        curr_activity = self._normalize_activity_name(self._foreground_activity_name())
        root_activity_norm = self._normalize_activity_name(root_activity)
        same_activity = bool(curr_activity and root_activity_norm and curr_activity == root_activity_norm)
        root_labels = {
            _clean_text(x).lower()
            for x in self._informative_observed_elements(self._state_semantic_summary(root_state, limit=24))
            if _clean_text(x)
        }
        curr_labels = {
            _clean_text(x).lower()
            for x in self._informative_observed_elements(self._state_semantic_summary(current_state, limit=24))
            if _clean_text(x)
        }
        overlap = root_labels.intersection(curr_labels)
        jaccard = float(len(overlap)) / float(len(root_labels.union(curr_labels)) or 1)
        curr_hash = self._state_hash(current_state)
        try:
            diff = int(_hash_diff(int(root_hash), int(curr_hash)))
        except Exception:  # pylint: disable=broad-exception-caught
            diff = 10**9
        action_type = _clean_text(getattr(action, "action_type", "")).lower()
        target_visible = False
        if action_type == json_action.CLICK:
            x = getattr(action, "x", None)
            y = getattr(action, "y", None)
            if x is not None and y is not None:
                target_visible = self._candidate_visible_in_state(
                    {"label": "planned action", "center": [x, y]},
                    current_state,
                )
        elif action_type == json_action.OPEN_APP:
            target_visible = True
        elif action_type in {json_action.STATUS, json_action.ANSWER}:
            target_visible = True
        elif action_type == json_action.INPUT_TEXT:
            target_visible = self._has_keyboard_ui(current_state) or any(
                bool(getattr(element, "is_editable", False))
                for element in list(getattr(current_state, "ui_elements", None) or [])
            )
        else:
            target_visible = same_activity
        ok = bool(target_visible and same_activity and (jaccard >= 0.35 or diff <= int(self.light_explore_hash_threshold + 8)))
        return ok, (
            f"activity={same_activity};anchors_jaccard={jaccard:.2f};"
            f"phash_diff={diff};target_visible={target_visible}"
        )

    def _is_noise_probe_text(self, merged: str) -> bool:
        if not getattr(self, "light_explore_lexical_rules", False):
            return False
        low = _clean_text(merged).lower()
        if not low:
            return True
        noise_tokens = (
            "inputmethod",
            "keyboard",
            "key_pos_",
            "systemui",
            "navigationbar",
            "statusbar",
        )
        if any(token in low for token in noise_tokens):
            return True
        if re.fullmatch(r"\.[a-z0-9]{1,6}", low):
            return True
        # These controls are often useful to the main policy, but poor probe
        # targets: they mostly test navigation chrome or launcher affordances
        # and make rollback expensive without adding task-specific knowledge.
        noise_phrases = (
            "navigate up",
            "open navigation drawer",
            "show navigation drawer",
            "search apps, web and more",
            "google lens",
            "touch outside",
            "drop close",
            "close app",
            "wait",
            "app info",
        )
        return any(phrase in low for phrase in noise_phrases)

    def _is_commit_probe_label(self, label: str) -> bool:
        if not getattr(self, "light_explore_lexical_rules", False):
            return False
        low = re.sub(r"^planned:\s*", "", _clean_text(label).lower()).strip()
        if not low:
            return False
        if re.fullmatch(r"(allow|apply|confirm|done|finish|ok|save|submit|yes)(\s+button[0-9]*)?", low):
            return True
        if re.search(r"\b(allow|apply|confirm|done|finish|ok|save|submit|yes)\b", low):
            return True
        return low in {
            "allow",
            "apply",
            "confirm",
            "done",
            "finish",
            "ok",
            "save",
            "submit",
            "yes",
        }

    def _is_transaction_unsafe_candidate(self, candidate: dict[str, Any] | None) -> bool:
        if not self.light_explore_transaction_safe or not isinstance(candidate, dict):
            return False
        label = _clean_text(candidate.get("label"))
        merged = _clean_text(candidate.get("merged"))
        action_kind = _clean_text(candidate.get("action_kind") or json_action.CLICK)
        combined = f"{label} {merged}".strip()
        if action_kind in {json_action.NAVIGATE_BACK, json_action.NAVIGATE_HOME, json_action.WAIT, json_action.OPEN_APP}:
            return False
        if action_kind in {
            json_action.INPUT_TEXT,
            json_action.SWIPE,
            json_action.SCROLL,
            json_action.LONG_PRESS,
            json_action.KEYBOARD_ENTER,
        }:
            return True
        if self._is_commit_probe_label(label) or self._is_commit_probe_label(merged):
            return True
        return self._is_risky_probe_text(combined)

    def _is_goal_irrelevant_probe_text(self, merged: str, goal: str) -> bool:
        if not getattr(self, "light_explore_lexical_rules", False):
            return False
        low = _clean_text(merged).lower()
        goal_low = _clean_text(goal).lower()
        if not low:
            return True
        media_terms = ("photo", "picture", "image", "avatar", "camera")
        if any(term in low for term in media_terms) and not any(term in goal_low for term in media_terms):
            return True
        account_terms = ("signed in", "@gmail.com", "google account")
        if any(term in low for term in account_terms) and not any(term in goal_low for term in ("account", "gmail", "email")):
            return True
        return False

    def _state_hash(self, state: Any) -> int:
        start = time.time()
        try:
            value = int(_phash_pixels(state.pixels))
        except Exception:  # pylint: disable=broad-exception-caught
            value = -1
        self._append_latency_profile_event(
            getattr(self, "_current_goal_for_depth", ""),
            {
                "event": "phash_compute",
                "latency_ms": float(max(0.0, time.time() - start) * 1000.0),
                "hash": value,
            },
        )
        return value

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

    @staticmethod
    def _merge_unique_tokens(*token_lists: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for tokens in token_lists:
            for token in tokens:
                cleaned = _clean_text(token).lower()
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    out.append(cleaned)
        return out

    def _collect_probe_candidates(self, state: Any, goal: str, planning_text: str = "") -> list[dict[str, Any]]:
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        if not ui_elements:
            self._last_candidate_quality_stats = {
                "candidate_count_raw": 0,
                "candidate_count_after_filter": 0,
                "filtered_system_ui_count": 0,
                "filtered_invalid_bbox_count": 0,
                "filtered_generic_container_count": 0,
                "filtered_app_name_only_count": 0,
                "filtered_duplicate_count": 0,
                "filtered_wrong_task_family_count": 0,
                "filtered_wrong_operator_count": 0,
                "filtered_risky_count": 0,
                "top_filtered_examples": [],
            }
            return []
        app_keywords = self._goal_app_keywords(goal)
        goal_tokens = self._goal_tokens(goal)
        if self.light_explore_decouple_planned and not getattr(self, "light_explore_use_planning_text", False):
            planning_text = ""
        planning_tokens = self._goal_tokens(planning_text)
        combined_tokens = self._merge_unique_tokens(goal_tokens, planning_tokens)
        secondary_tokens = [token for token in combined_tokens if token not in SECONDARY_TOKEN_STOPWORDS]
        min_attempts = int(getattr(self, "light_explore_min_attempts_per_step", 0) or 0)
        quality_stats = {
            "candidate_count_raw": 0,
            "candidate_count_after_filter": 0,
            "filtered_system_ui_count": 0,
            "filtered_invalid_bbox_count": 0,
            "filtered_generic_container_count": 0,
            "filtered_app_name_only_count": 0,
            "filtered_duplicate_count": 0,
            "filtered_wrong_task_family_count": 0,
            "filtered_wrong_operator_count": 0,
            "filtered_risky_count": 0,
            "top_filtered_examples": [],
        }

        candidates: list[dict[str, Any]] = []
        duplicate_keys: set[str] = set()
        for idx, element in enumerate(ui_elements):
            if not self._is_interactive(element):
                continue
            quality_stats["candidate_count_raw"] += 1
            center = self._safe_center_from_element(element)
            if center is None:
                quality_stats["filtered_invalid_bbox_count"] += 1
                continue
            merged = self._element_text(element)
            if not merged:
                merged = self._normalize_resource_id(getattr(element, "resource_id", "")) or _clean_text(
                    getattr(element, "class_name", "")
                )
            quality_probe = {
                "index": idx,
                "center": center,
                "label": _clean_text(getattr(element, "text", "")) or _clean_text(getattr(element, "content_description", "")) or merged,
                "merged": merged,
                "a11y": self._element_trace(element, idx),
            }
            fallback_quality_reason = ""
            quality_reason = self._candidate_quality_filter_reason(quality_probe, goal)
            if quality_reason and not self.light_explore_diagnostic_full:
                hard_filter_reasons = {
                    "system_ui",
                    "invalid_bbox",
                    "invalid_or_negative_center",
                    "generic_container",
                    "app_name_only",
                    "calendar_new_event",
                }
                if min_attempts and len(candidates) < min_attempts and quality_reason not in hard_filter_reasons:
                    fallback_quality_reason = f"fallback_safe_shortage:{quality_reason}"
                else:
                    if quality_reason == "system_ui":
                        quality_stats["filtered_system_ui_count"] += 1
                    elif quality_reason in {"invalid_bbox", "invalid_or_negative_center"}:
                        quality_stats["filtered_invalid_bbox_count"] += 1
                    elif quality_reason == "generic_container":
                        quality_stats["filtered_generic_container_count"] += 1
                    elif quality_reason == "app_name_only":
                        quality_stats["filtered_app_name_only_count"] += 1
                    elif quality_reason in {
                        "calendar_new_event",
                        "calendar_date_nav_arrow",
                        "calendar_empty_grid",
                    }:
                        quality_stats["filtered_wrong_task_family_count"] += 1
                    else:
                        quality_stats["filtered_wrong_operator_count"] += 1
                    if len(quality_stats["top_filtered_examples"]) < 12:
                        quality_stats["top_filtered_examples"].append(
                            {
                                "label": quality_probe["label"],
                                "merged": merged,
                                "reason": quality_reason,
                                "center": center,
                            }
                        )
                    continue
            if self._is_noise_probe_text(merged) and not self.light_explore_diagnostic_full:
                if min_attempts and len(candidates) < min_attempts:
                    fallback_quality_reason = fallback_quality_reason or "fallback_safe_shortage:noise_or_low_text"
                else:
                    continue
            if self._is_goal_irrelevant_probe_text(merged, goal) and not self.light_explore_diagnostic_full:
                if min_attempts and len(candidates) < min_attempts:
                    fallback_quality_reason = fallback_quality_reason or "fallback_safe_shortage:goal_irrelevant"
                else:
                    continue
            risky_or_commit = self._is_risky_probe_text(merged) or self._is_commit_probe_label(merged)
            if self.light_explore_transaction_safe and risky_or_commit and not self.light_explore_fixed_framework:
                quality_stats["filtered_risky_count"] += 1
                continue
            if self._is_risky_probe_text(merged) and not self.light_explore_diagnostic_full and not self.light_explore_fixed_framework:
                quality_stats["filtered_risky_count"] += 1
                continue

            relevance = self._task_relevance_score(
                merged=merged,
                goal=goal,
                goal_tokens=combined_tokens,
                app_keywords=app_keywords,
            )
            planning_relevance = self._lexical_overlap_score(
                merged=merged,
                goal_tokens=planning_tokens,
                app_keywords=[],
            )
            token_relevance = self._lexical_overlap_score(
                merged=merged,
                goal_tokens=secondary_tokens,
                app_keywords=[],
            )
            if planning_text:
                planning_embed_score = self._cosine(
                    self._hash_embedding(planning_text),
                    self._hash_embedding(merged),
                )
                planning_relevance = max(planning_relevance, min(0.05, planning_embed_score * 0.2))
                relevance = max(relevance, min(1.0, planning_relevance * 0.9))
            if not app_keywords and not goal_tokens:
                relevance = 0.5
            if self.light_explore_require_keyword and app_keywords and relevance <= 0.0:
                continue

            key = self._candidate_key(element, center, merged)
            duplicate_key = (
                re.sub(r"[^a-z0-9]+", " ", _clean_text(merged).lower()).strip(),
                int(center[0] // 96),
                int(center[1] // 96),
            )
            min_attempts = int(getattr(self, "light_explore_min_attempts_per_step", 0) or 0)
            if (
                duplicate_key in duplicate_keys
                and not self.light_explore_diagnostic_full
                and (not min_attempts or len(candidates) >= min_attempts)
            ):
                quality_stats["filtered_duplicate_count"] += 1
                if len(quality_stats["top_filtered_examples"]) < 12:
                    quality_stats["top_filtered_examples"].append(
                        {
                            "label": quality_probe["label"],
                            "merged": merged,
                            "reason": "duplicate_candidate",
                            "center": center,
                        }
                    )
                continue
            duplicate_keys.add(duplicate_key)
            visits = int(self._explored_element_visits.get(key, 0))
            interactivity_bonus = 0.05 if bool(getattr(element, "is_clickable", False)) else 0.0
            text_bonus = 0.04 if _clean_text(getattr(element, "text", "")) else 0.0
            affordance_bonus = self._goal_affordance_bonus(merged, goal, planning_text)
            score = float(
                relevance
                + interactivity_bonus
                + text_bonus
                + affordance_bonus
                - (self.light_explore_visit_penalty * visits)
            )
            # Keep a small but bounded fallback so exploration can still inspect
            # sparse screens without letting unrelated visited nodes dominate.
            if score <= 0.0:
                if min_attempts and len(candidates) < min_attempts:
                    score = 0.01
                    fallback_quality_reason = fallback_quality_reason or "fallback_safe_shortage:non_positive_score"
                elif visits <= 0:
                    score = 0.02
                else:
                    continue

            label = _clean_text(getattr(element, "text", "")) or _clean_text(
                getattr(element, "content_description", "")
            )
            if not label:
                label = app_keywords[0] if app_keywords else "candidate"

            candidate_record = {
                    "index": idx,
                    "element": element,
                    "center": center,
                    "label": label,
                    "score": float(score),
                    "relevance": float(relevance),
                    "planning_relevance": float(planning_relevance),
                    "token_relevance": float(token_relevance),
                    "affordance_bonus": float(affordance_bonus),
                    "visits": int(visits),
                    "key": key,
                    "merged": merged,
                    "a11y": self._element_trace(element, idx),
                }
            operator, operator_reason = self._candidate_operator(candidate_record, goal)
            candidate_record["operator"] = operator
            candidate_record["operator_reason"] = operator_reason
            candidate_record["risk_boundary"] = operator == "RiskBoundary"
            if fallback_quality_reason:
                candidate_record["fallback_safe_candidate"] = True
                candidate_record["quality_filter_reason"] = fallback_quality_reason
            candidate_record["score_components"] = self._candidate_score_components(candidate_record, goal)
            candidate_record["final_score"] = float(
                (candidate_record.get("score_components") or {}).get("final_score") or score
            )
            candidates.append(candidate_record)

        candidates.sort(
            key=lambda item: (
                float(item.get("final_score") or item.get("score", 0.0)),
                float(item.get("score", 0.0)),
                -int(item.get("index", 0)),
            ),
            reverse=True,
        )
        keep_n = max(
            3,
            int(self.light_explore_branch_budget) * 2,
            int(getattr(self, "light_explore_min_attempts_per_step", 0)) * 2,
        )
        kept = candidates[:keep_n]
        quality_stats["candidate_count_after_filter"] = len(kept)
        self._last_candidate_quality_stats = quality_stats
        return kept

    def _choose_probe_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        return candidates[0]

    @staticmethod
    def _goal_affordance_bonus(merged: str, goal: str, planning_text: str = "") -> float:
        text = _clean_text(merged).lower()
        context = f"{_clean_text(goal)} {_clean_text(planning_text)}".lower()
        if not text or not context:
            return 0.0
        needs_lookup = any(
            token in context
            for token in (
                "find",
                "search",
                "locate",
                "titled",
                "named",
                "called",
                "which",
                "what",
                "count",
                "whether",
                "if",
            )
        )
        if needs_lookup and any(token in text for token in ("search", "find", "filter", "lookup")):
            return 0.65
        if any(token in context for token in ("create", "add", "new")) and any(
            token in text for token in ("add", "new", "create", "+")
        ):
            return 0.45
        if any(token in context for token in ("delete", "remove")) and any(
            token in text for token in ("delete", "remove", "trash")
        ):
            return 0.35
        return 0.0

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
            if self._is_transaction_unsafe_candidate(cand):
                continue
            if self.light_explore_diagnostic_full:
                return cand
            if self._is_commit_probe_label(str(cand.get("label") or cand.get("merged") or "")):
                continue
            relevance = float(cand.get("relevance") or 0.0)
            planning_relevance = float(cand.get("planning_relevance") or 0.0)
            token_relevance = float(cand.get("token_relevance") or 0.0)
            score = float(cand.get("score") or 0.0)
            a11y = cand.get("a11y") if isinstance(cand.get("a11y"), dict) else {}
            bbox = a11y.get("bbox") if isinstance(a11y.get("bbox"), dict) else {}
            try:
                area = max(0, int(bbox.get("x_max", 0)) - int(bbox.get("x_min", 0))) * max(
                    0,
                    int(bbox.get("y_max", 0)) - int(bbox.get("y_min", 0)),
                )
            except Exception:  # pylint: disable=broad-exception-caught
                area = 0
            if area > int(1080 * 2400 * 0.35):
                continue
            if max(token_relevance, planning_relevance) <= 0.0:
                continue
            if max(relevance, planning_relevance, token_relevance) <= 0.0 and score < float(
                self.light_explore_min_secondary_score
            ):
                continue
            return cand
        return None

    @staticmethod
    def _extract_goal_entry_text(goal: str) -> str:
        text = _clean_text(goal)
        if not text:
            return ""
        quoted = re.search(r"['\"]([^'\"]{2,80})['\"]", text)
        if quoted:
            return _clean_text(quoted.group(1))
        patterns = (
            r"\bnamed\s+([A-Za-z0-9_.-]{2,80})",
            r"\bcalled\s+([A-Za-z0-9_.-]{2,80})",
            r"\bwith file name\s+([A-Za-z0-9_.-]{2,80})",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return _clean_text(match.group(1).rstrip(".,;:"))
        return ""

    def _extract_goal_search_text(self, goal: str) -> str:
        text = _clean_text(goal)
        low = text.lower()
        quoted = self._quoted_phrases(text)
        if quoted:
            return quoted[0]
        filename = re.search(r"\b([A-Za-z0-9_.-]+\.(?:html|md|txt|mp3|m4a|jpg|png|pdf))\b", text)
        if filename:
            return _clean_text(filename.group(1))
        date = re.search(
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{1,2}(?:,\s*|\s+)20\d{2})\b",
            text,
        )
        if date:
            return _clean_text(date.group(1).replace(",", ""))
        recipe = re.search(r"\brecipe\s+([A-Z][A-Za-z0-9' -]{2,60}?)(?:\s+in\b|\s+from\b|\s+answer\b|$)", text)
        if recipe:
            return _clean_text(recipe.group(1))
        for_recipe = re.search(r"\bfor\s+([A-Z][A-Za-z0-9' -]{2,60}?)(?:\?|\.|,|$)", text)
        if for_recipe:
            return _clean_text(for_recipe.group(1))
        meeting = re.search(r"\bmeeting\s+with\s+([A-Z][a-z]{2,})\b", text)
        if meeting:
            return _clean_text(meeting.group(1))
        titled = re.search(r"\b(?:titled|named|called)\s+([A-Za-z0-9_. -]{2,60}?)(?:\s+in\b|\s+from\b|[.;]|$)", text)
        if titled:
            return _clean_text(titled.group(1))
        for token in ("kayaking", "skiing", "running", "walking", "cycling", "hiking", "rowing"):
            if token in low:
                return token
        return self._extract_goal_entry_text(goal)

    def _state_has_search_ui(self, state: Any) -> bool:
        labels = " ".join(self._state_semantic_summary(state, limit=60)).lower()
        if re.search(r"\b(search|clear query|submit query|search plate|query)\b", labels):
            return True
        for element in list(getattr(state, "ui_elements", None) or []):
            merged = " ".join(
                _clean_text(x).lower()
                for x in (
                    getattr(element, "text", ""),
                    getattr(element, "content_description", ""),
                    getattr(element, "hint_text", ""),
                    getattr(element, "resource_id", ""),
                )
            )
            if re.search(r"\b(search|query|search_src_text|search_plate)\b", merged):
                return True
        return False

    def _state_has_dialog_or_sheet(self, state: Any) -> bool:
        labels = " ".join(self._state_semantic_summary(state, limit=80)).lower()
        if re.search(r"\b(dialog|bottom sheet|date picker|filter|choose|open with|complete action using)\b", labels):
            return True
        for element in list(getattr(state, "ui_elements", None) or []):
            merged = " ".join(
                _clean_text(x).lower()
                for x in (
                    getattr(element, "text", ""),
                    getattr(element, "content_description", ""),
                    getattr(element, "hint_text", ""),
                    getattr(element, "resource_id", ""),
                    getattr(element, "class_name", ""),
                )
            )
            if re.search(r"\b(dialog|bottomsheet|bottom_sheet|datepicker|date_picker|filter)\b", merged):
                return True
        return False

    def _state_editable_count(self, state: Any) -> int:
        count = 0
        for element in list(getattr(state, "ui_elements", None) or []):
            class_name = _clean_text(getattr(element, "class_name", "")).lower()
            resource_id = _clean_text(getattr(element, "resource_id", "")).lower()
            if bool(getattr(element, "is_editable", False)) or "edittext" in class_name or "edit_text" in resource_id:
                count += 1
        return count

    def _screen_role_from_state(self, state: Any) -> str:
        labels = " ".join(self._state_semantic_summary(state, limit=80)).lower()
        if self._state_has_search_ui(state):
            return "search"
        if self._state_has_dialog_or_sheet(state):
            return "dialog_or_sheet"
        if re.search(r"\b(stats|statistics|duration|distance|intervals|track detail)\b", labels):
            return "stats_or_detail"
        if re.search(r"\b(form|title|description|amount|name|email|phone|save|done)\b", labels):
            return "form"
        if re.search(r"\b(list|recycler|task|event|note|file|recipe)\b", labels):
            return "list"
        return "unknown"

    def _semantic_state_change_details(
        self,
        root_state: Any,
        child_state: Any,
        candidate: dict[str, Any] | None,
        goal: str,
        root_activity: str = "",
        child_activity: str = "",
        hash_diff: int | None = None,
    ) -> dict[str, Any]:
        candidate = candidate if isinstance(candidate, dict) else {}
        root_activity_norm = self._normalize_activity_name(root_activity)
        child_activity_norm = self._normalize_activity_name(child_activity)
        activity_changed = bool(root_activity_norm and child_activity_norm and root_activity_norm != child_activity_norm)
        if hash_diff is None:
            try:
                root_hash = self._state_hash(root_state)
                child_hash = self._state_hash(child_state)
                hash_diff = int(_hash_diff(root_hash, child_hash)) if root_hash >= 0 and child_hash >= 0 else None
            except Exception:  # pylint: disable=broad-exception-caught
                hash_diff = None
        hash_changed = bool(hash_diff is not None and hash_diff >= int(self.light_explore_hash_threshold))
        root_labels = {
            _clean_text(x).lower()
            for x in self._informative_observed_elements(self._state_semantic_summary(root_state, limit=80))
            if _clean_text(x)
        }
        child_labels = {
            _clean_text(x).lower()
            for x in self._informative_observed_elements(self._state_semantic_summary(child_state, limit=80))
            if _clean_text(x)
        }
        label_delta = sorted(child_labels - root_labels)
        root_search = self._state_has_search_ui(root_state)
        child_search = self._state_has_search_ui(child_state)
        search_ui_transition = bool(child_search and not root_search)
        editable_transition = bool(self._state_editable_count(child_state) > self._state_editable_count(root_state))
        dialog_transition = bool(self._state_has_dialog_or_sheet(child_state) and not self._state_has_dialog_or_sheet(root_state))
        root_role = self._screen_role_from_state(root_state)
        child_role = self._screen_role_from_state(child_state)
        screen_role_changed = bool(root_role != child_role and child_role != "unknown")
        entity_transition = False
        root_text = " ".join(root_labels)
        child_text = " ".join(child_labels)
        for entity in self._task_entities(goal):
            if entity and entity not in root_text and entity in child_text:
                entity_transition = True
                break
        operator, _ = self._candidate_operator(candidate, goal)
        filter_transition = bool(
            operator == "FilterPeek"
            and re.search(r"\b(filter|date|category|from|to|calendar)\b", child_text)
        )
        reasons: list[str] = []
        if activity_changed:
            reasons.append("activity_changed")
        if hash_changed:
            reasons.append(f"phash_changed:{hash_diff}")
        if search_ui_transition:
            reasons.append("search_ui_transition")
        if editable_transition:
            reasons.append("editable_field_transition")
        if dialog_transition:
            reasons.append("dialog_or_sheet_transition")
        if len(label_delta) >= 2:
            reasons.append(f"informative_label_delta:{len(label_delta)}")
        if entity_transition:
            reasons.append("task_entity_transition")
        if operator == "SearchPeek" and child_search:
            reasons.append("searchpeek_search_ui")
        if filter_transition:
            reasons.append("filterpeek_options")
        if operator == "NavigationPeek" and screen_role_changed:
            reasons.append(f"screen_role_changed:{root_role}->{child_role}")
        return {
            "semantic_changed": bool(reasons),
            "semantic_change_reason": ",".join(reasons) if reasons else "no_semantic_change",
            "hash_changed": hash_changed,
            "activity_changed": activity_changed,
            "label_delta_count": int(len(label_delta)),
            "new_labels": label_delta[:12],
            "search_ui_transition": search_ui_transition,
            "dialog_or_sheet_transition": dialog_transition,
            "editable_transition": editable_transition,
            "screen_role_before": root_role,
            "screen_role_after": child_role,
        }

    def semantic_state_changed(
        self,
        root_state: Any,
        child_state: Any,
        candidate: dict[str, Any] | None,
        goal: str,
    ) -> tuple[bool, str]:
        details = self._semantic_state_change_details(root_state, child_state, candidate, goal)
        return bool(details.get("semantic_changed")), _clean_text(details.get("semantic_change_reason"))

    def _is_safe_search_input_candidate(
        self,
        candidate: dict[str, Any] | None,
        goal: str,
        state: Any | None = None,
    ) -> bool:
        if not isinstance(candidate, dict):
            return False
        action_kind = _clean_text(candidate.get("action_kind") or "").lower()
        if action_kind not in {"type", json_action.INPUT_TEXT}:
            return False
        text_value = _clean_text(candidate.get("text"))
        if not text_value:
            return False
        allowed_queries = {self._extract_goal_search_text(goal).lower(), self._extract_goal_entry_text(goal).lower()}
        allowed_queries = {x for x in allowed_queries if x}
        if allowed_queries and text_value.lower() not in allowed_queries:
            token_overlap = 0.0
            text_tokens = set(re.findall(r"[a-z0-9]{3,}", text_value.lower()))
            allowed_tokens = set().union(*(set(re.findall(r"[a-z0-9]{3,}", x)) for x in allowed_queries))
            if text_tokens and allowed_tokens:
                token_overlap = len(text_tokens & allowed_tokens) / float(len(text_tokens | allowed_tokens))
            if token_overlap < 0.8:
                return False
        a11y = candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {}
        field_text = " ".join(
            _clean_text(a11y.get(k)).lower()
            for k in ("text", "content_description", "hint_text", "resource_id", "class_name")
        )
        if re.search(r"\b(body|message|note body|description|amount|phone|email|contact|expense|title|name)\b", field_text):
            return False
        field_has_search = bool(re.search(r"\b(search|query|search_src_text|search plate|search_plate)\b", field_text))
        screen_has_search = bool(state is not None and self._state_has_search_ui(state))
        return bool(field_has_search or screen_has_search)

    def _typed_probe_candidate(self, state: Any, goal: str, planning_text: str = "") -> dict[str, Any] | None:
        entry_text = self._extract_goal_entry_text(goal)
        if self.t2_allow_safe_search_input and self._state_has_search_ui(state):
            entry_text = self._extract_goal_search_text(goal) or entry_text
        if not entry_text:
            return None
        state_labels = " ".join(self._state_semantic_summary(state, limit=30)).lower()
        if entry_text.lower() in state_labels:
            return None
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        editable_elements = []
        for idx, element in enumerate(ui_elements):
            class_name = _clean_text(getattr(element, "class_name", "")).lower()
            resource_id = _clean_text(getattr(element, "resource_id", "")).lower()
            if bool(getattr(element, "is_editable", False)) or "edittext" in class_name or "edit_text" in resource_id:
                editable_elements.append((idx, element))
        if not editable_elements:
            dialogish = any(
                _clean_text(getattr(element, "text", "")).lower() in {"folder", "file", "ok", "create", "add"}
                for element in ui_elements
            )
            if not dialogish:
                return None
            return {
                "index": -1,
                "element": None,
                "center": (0, 0),
                "label": f'type "{entry_text}" into focused text field',
                "score": 8.0,
                "relevance": 1.0,
                "planning_relevance": 0.5,
                "token_relevance": 1.0,
                "visits": 0,
                "key": f"type|focused|{entry_text[:64]}",
                "merged": entry_text,
                "a11y": {},
                "action_kind": "type",
                "text": entry_text,
                "clear_text": True,
            }
        idx, element = editable_elements[0]
        center = self._safe_center_from_element(element) or (0, 0)
        label = _clean_text(getattr(element, "text", "")) or _clean_text(
            getattr(element, "content_description", "")
        )
        if not label:
            label = self._normalize_resource_id(getattr(element, "resource_id", "")) or "editable field"
        planning_relevance = self._lexical_overlap_score(
            label,
            goal_tokens=self._goal_tokens(planning_text),
            app_keywords=[],
        )
        return {
            "index": idx,
            "element": element,
            "center": center,
            "label": f'type "{entry_text}" into {label}',
            "score": 9.0,
            "relevance": 1.0,
            "planning_relevance": float(max(0.5, planning_relevance)),
            "token_relevance": 1.0,
            "visits": 0,
            "key": f"type|{entry_text[:64]}|{self._candidate_key(element, center, label)}",
            "merged": f"{label} {entry_text}",
            "a11y": self._element_trace(element, idx),
            "action_kind": "type",
            "text": entry_text,
            "clear_text": True,
        }

    @staticmethod
    def _planned_action_alignment(candidate: dict[str, Any], action: json_action.JSONAction | None) -> float:
        if action is None:
            return 0.0
        if action.action_type == json_action.CLICK:
            center = candidate.get("center")
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                return 0.0
            try:
                dist = math.sqrt((float(center[0]) - float(action.x)) ** 2 + (float(center[1]) - float(action.y)) ** 2)
            except Exception:  # pylint: disable=broad-exception-caught
                return 0.0
            return float(max(0.0, 1.0 - dist / 260.0))
        if action.action_type == json_action.OPEN_APP:
            app = _clean_text(getattr(action, "app_name", "")).lower()
            merged = _clean_text(candidate.get("merged")).lower()
            return 1.0 if app and app in merged else 0.0
        return 0.0

    def _rank_candidates_for_planned_action(
        self,
        candidates: list[dict[str, Any]],
        action: json_action.JSONAction | None,
    ) -> list[dict[str, Any]]:
        if not candidates or action is None:
            return candidates
        ranked: list[dict[str, Any]] = []
        for item in candidates:
            cand = dict(item)
            alignment = self._planned_action_alignment(cand, action)
            cand["planned_action_alignment"] = float(alignment)
            if alignment > 0.0:
                cand["score"] = float(cand.get("score") or 0.0) + 0.75 * alignment
            ranked.append(cand)
        ranked.sort(
            key=lambda item: (
                float(item.get("planned_action_alignment") or 0.0),
                float(item.get("score") or 0.0),
            ),
            reverse=True,
        )
        return ranked

    @staticmethod
    def _is_replay_safe_action(action: json_action.JSONAction | None) -> bool:
        if action is None:
            return False
        if action.action_type == json_action.OPEN_APP:
            return bool(_clean_text(getattr(action, "app_name", "")))
        if action.action_type == json_action.INPUT_TEXT:
            return bool(_clean_text(getattr(action, "text", "")))
        return action.action_type in {
            json_action.CLICK,
            json_action.LONG_PRESS,
            json_action.SWIPE,
            json_action.NAVIGATE_BACK,
            json_action.NAVIGATE_HOME,
            json_action.OPEN_APP,
            json_action.INPUT_TEXT,
            json_action.KEYBOARD_ENTER,
            json_action.WAIT,
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

    def _analyze_rollback_failure(
        self,
        root_activity: str,
        root_hash: int,
        final_state: Any,
        replay_actions: list[json_action.JSONAction],
        matched_by: str,
        replay_anchor: dict[str, Any] | None = None,
        post_replay_back_stop_reason: str = "",
    ) -> dict[str, Any]:
        final_activity = self._foreground_activity_name()
        final_hash = self._state_hash(final_state)
        hash_diff = None
        if root_hash >= 0 and final_hash >= 0:
            try:
                hash_diff = int(_hash_diff(root_hash, final_hash))
            except Exception:  # pylint: disable=broad-exception-caught
                hash_diff = None
        reasons: list[str] = []
        if self._normalize_activity_name(root_activity) != self._normalize_activity_name(final_activity):
            reasons.append("activity_mismatch")
        if hash_diff is None:
            reasons.append("hash_unavailable")
        elif hash_diff > int(self.light_explore_hash_threshold + 4):
            reasons.append("visual_hash_mismatch")
        if not replay_actions:
            reasons.append("no_replay_trace")
        anchor_info = dict(replay_anchor or {})
        original_types = list(anchor_info.get("original_replay_action_types") or [])
        if (
            self._package_from_activity(root_activity)
            and not self._is_launcher_activity(root_activity)
            and json_action.OPEN_APP not in original_types
        ):
            reasons.append("missing_open_app_anchor")
        if int(anchor_info.get("dropped_actions_before_anchor") or 0) > 0:
            reasons.append("unsafe_replay_actions_before_open_app")
        if int(anchor_info.get("dropped_foreign_open_app_actions") or 0) > 0:
            reasons.append("foreign_open_app_removed_from_replay")
        if post_replay_back_stop_reason:
            reasons.append(str(post_replay_back_stop_reason))
        if self._has_keyboard_ui(final_state):
            reasons.append("keyboard_or_ime_visible")
        if not reasons:
            reasons.append("unknown_dynamic_ui_drift")
        return {
            "root_activity": root_activity,
            "final_activity": final_activity,
            "root_hash": root_hash,
            "final_hash": final_hash,
            "hash_diff": hash_diff,
            "matched_by": matched_by,
            "replay_action_types": [str(action.action_type) for action in replay_actions],
            "replay_anchor": anchor_info,
            "post_replay_back_stop_reason": post_replay_back_stop_reason,
            "likely_reasons": reasons,
            "final_semantic_summary": self._state_semantic_summary(final_state),
        }

    def _rollback_to_probe_root(
        self,
        root_activity: str,
        root_hash: int,
        replay_actions: list[json_action.JSONAction],
        step_idx: int,
        current_state: Any | None = None,
    ) -> dict[str, Any]:
        rollback_start = time.time()
        pending_verify_state = current_state

        def _verify_root() -> tuple[bool, str, Any]:
            nonlocal pending_verify_state
            if pending_verify_state is not None:
                curr_state = pending_verify_state
                pending_verify_state = None
            else:
                curr_state = self._get_probe_state(wait_to_stabilize=True)
            same, matched_by = self._same_root_page(curr_state, root_activity, root_hash)
            return same, matched_by, curr_state

        result: dict[str, Any] = {
            "success": False,
            "mode": "backtrack_failed",
            "level": "level1",
            "back_presses": 0,
            "replayed_actions": 0,
            "matched_by": None,
            "latency_ms": 0.0,
            "home_attempted": False,
            "original_replay_action_types": [str(action.action_type) for action in replay_actions],
            "replay_action_types": [str(action.action_type) for action in replay_actions],
            "replay_failures": [],
            "replay_anchor": {},
            "post_replay_back_stop_reason": "",
            "post_replay_back_presses": 0,
            "home_replay_action_limit_removed": True,
            "home_replay_timeout_sec": float(
                max(2.0, _env_float("ANDROID_WORLD_LIGHT_EXPLORE_HOME_REPLAY_TIMEOUT_SEC", 30.0))
            ),
            "profile_events": [],
        }

        back_limit = max(1, int(self.light_explore_back_limit))
        for i in range(back_limit + 1):
            verify_start = time.time()
            same, matched_by, _ = _verify_root()
            verify_event = {
                "event": "rollback_verify",
                "step": int(step_idx + 1),
                "phase": "level1_backtrack",
                "attempt": int(i),
                "same": bool(same),
                "matched_by": matched_by,
                "latency_ms": float(max(0.0, time.time() - verify_start) * 1000.0),
                "back_presses_so_far": int(result["back_presses"]),
            }
            result["profile_events"].append(verify_event)
            self._append_latency_profile_event(getattr(self, "_current_goal_for_depth", ""), verify_event)
            if same:
                result["success"] = True
                result["mode"] = "backtrack"
                result["level"] = "level1"
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
            back_start = time.time()
            self._execute_probe_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK))
            back_event = {
                "event": "rollback_action",
                "step": int(step_idx + 1),
                "phase": "level1_backtrack",
                "action_type": json_action.NAVIGATE_BACK,
                "latency_ms": float(max(0.0, time.time() - back_start) * 1000.0),
            }
            result["profile_events"].append(back_event)
            self._append_latency_profile_event(getattr(self, "_current_goal_for_depth", ""), back_event)
            result["back_presses"] += 1

        result["level"] = "level2"
        result["mode"] = "home_replay"
        result["level2_triggered"] = True
        result["level2_trigger_reason"] = self._infer_level2_trigger_reason(result)
        print(
            f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
            f"rollback_fallback=home_replay trigger={result['level2_trigger_reason']}"
        )
        try:
            result["home_attempted"] = True
            home_start = time.time()
            self._execute_probe_action(json_action.JSONAction(action_type=json_action.NAVIGATE_HOME))
            home_event = {
                "event": "rollback_action",
                "step": int(step_idx + 1),
                "phase": "level2_home_replay",
                "action_type": json_action.NAVIGATE_HOME,
                "latency_ms": float(max(0.0, time.time() - home_start) * 1000.0),
            }
            result["profile_events"].append(home_event)
            self._append_latency_profile_event(getattr(self, "_current_goal_for_depth", ""), home_event)
            self._get_probe_state(wait_to_stabilize=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            result["replay_failures"].append({"action_type": json_action.NAVIGATE_HOME, "error": _clean_text(exc)})
            print(
                f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
                f"rollback_home_action_failed error={exc}"
            )

        replay_actions, replay_anchor = self._prepare_home_replay_actions(replay_actions, root_activity)
        result["replay_anchor"] = dict(replay_anchor)
        result["replay_action_types"] = [str(action.action_type) for action in replay_actions]

        if replay_actions:
            for replay_idx, action in enumerate(replay_actions):
                try:
                    replay_start = time.time()
                    if action.action_type == json_action.WAIT:
                        time.sleep(1.0)
                    elif replay_idx == 0 and action.action_type == json_action.OPEN_APP:
                        self._execute_root_open_anchor(action, root_activity)
                    else:
                        self._execute_probe_action(action)
                    replay_event = {
                        "event": "rollback_action",
                        "step": int(step_idx + 1),
                        "phase": "level2_replay",
                        "action_type": str(action.action_type),
                        "latency_ms": float(max(0.0, time.time() - replay_start) * 1000.0),
                    }
                    result["profile_events"].append(replay_event)
                    self._append_latency_profile_event(getattr(self, "_current_goal_for_depth", ""), replay_event)
                    result["replayed_actions"] += 1
                    self._get_probe_state(wait_to_stabilize=True)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    result["replay_failures"].append(
                        {"action_type": str(action.action_type), "error": _clean_text(exc)}
                    )
                    print(
                        f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
                        f"rollback_replay_action_failed action={action.action_type} error={exc}"
                    )

        verify_start = time.time()
        same, matched_by, final_state = _verify_root()
        verify_event = {
            "event": "rollback_verify",
            "step": int(step_idx + 1),
            "phase": "level2_final",
            "same": bool(same),
            "matched_by": matched_by,
            "latency_ms": float(max(0.0, time.time() - verify_start) * 1000.0),
        }
        result["profile_events"].append(verify_event)
        self._append_latency_profile_event(getattr(self, "_current_goal_for_depth", ""), verify_event)
        if not same:
            root_pkg = self._normalize_activity_name(root_activity).split("/", 1)[0]
            final_activity = self._normalize_activity_name(self._foreground_activity_name())
            final_pkg = final_activity.split("/", 1)[0] if final_activity else ""
            if root_pkg and final_pkg and root_pkg == final_pkg:
                post_replay_deadline = time.time() + float(result["home_replay_timeout_sec"])
                post_replay_seen: dict[tuple[str, int], int] = {}
                i = 0
                while True:
                    if time.time() >= post_replay_deadline:
                        result["post_replay_back_stop_reason"] = "post_replay_back_timeout"
                        break
                    curr_activity = self._normalize_activity_name(self._foreground_activity_name())
                    curr_pkg = curr_activity.split("/", 1)[0] if curr_activity else ""
                    if curr_pkg != root_pkg or self._is_launcher_activity(curr_activity):
                        result["post_replay_back_stop_reason"] = "left_root_package_before_back"
                        break
                    try:
                        final_hash_before_back = int(self._state_hash(final_state))
                    except Exception:  # pylint: disable=broad-exception-caught
                        final_hash_before_back = -1
                    state_key = (curr_activity, final_hash_before_back)
                    post_replay_seen[state_key] = post_replay_seen.get(state_key, 0) + 1
                    if post_replay_seen[state_key] >= 2:
                        result["post_replay_back_stop_reason"] = "post_replay_back_stalled_no_state_change"
                        break
                    print(
                        f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
                        f"rollback_post_replay_back#{i + 1} matched={matched_by}"
                    )
                    post_back_start = time.time()
                    self._execute_probe_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK))
                    back_event = {
                        "event": "rollback_action",
                        "step": int(step_idx + 1),
                        "phase": "level2_post_replay_back",
                        "action_type": json_action.NAVIGATE_BACK,
                        "latency_ms": float(max(0.0, time.time() - post_back_start) * 1000.0),
                    }
                    result["profile_events"].append(back_event)
                    self._append_latency_profile_event(getattr(self, "_current_goal_for_depth", ""), back_event)
                    result["back_presses"] += 1
                    result["post_replay_back_presses"] += 1
                    verify_start = time.time()
                    same, matched_by, final_state = _verify_root()
                    verify_event = {
                        "event": "rollback_verify",
                        "step": int(step_idx + 1),
                        "phase": "level2_post_replay_back_verify",
                        "attempt": int(i + 1),
                        "same": bool(same),
                        "matched_by": matched_by,
                        "latency_ms": float(max(0.0, time.time() - verify_start) * 1000.0),
                    }
                    result["profile_events"].append(verify_event)
                    self._append_latency_profile_event(getattr(self, "_current_goal_for_depth", ""), verify_event)
                    if same:
                        break
                    curr_activity = self._normalize_activity_name(self._foreground_activity_name())
                    curr_pkg = curr_activity.split("/", 1)[0] if curr_activity else ""
                    if curr_pkg != root_pkg or self._is_launcher_activity(curr_activity):
                        result["post_replay_back_stop_reason"] = "left_root_package_after_back"
                        root_app_name = _clean_text(replay_anchor.get("root_app_name") or "")
                        if root_app_name:
                            reopen_start = time.time()
                            try:
                                self._execute_root_open_anchor(
                                    json_action.JSONAction(
                                        action_type=json_action.OPEN_APP,
                                        app_name=root_app_name,
                                    ),
                                    root_activity,
                                )
                                reopen_event = {
                                    "event": "rollback_action",
                                    "step": int(step_idx + 1),
                                    "phase": "level2_recover_root_app",
                                    "action_type": json_action.OPEN_APP,
                                    "latency_ms": float(max(0.0, time.time() - reopen_start) * 1000.0),
                                    "app_name": root_app_name,
                                }
                                result["profile_events"].append(reopen_event)
                                self._append_latency_profile_event(
                                    getattr(self, "_current_goal_for_depth", ""),
                                    reopen_event,
                                )
                                verify_start = time.time()
                                same, matched_by, final_state = _verify_root()
                                verify_event = {
                                    "event": "rollback_verify",
                                    "step": int(step_idx + 1),
                                    "phase": "level2_recover_root_app_verify",
                                    "same": bool(same),
                                    "matched_by": matched_by,
                                    "latency_ms": float(max(0.0, time.time() - verify_start) * 1000.0),
                                }
                                result["profile_events"].append(verify_event)
                                self._append_latency_profile_event(
                                    getattr(self, "_current_goal_for_depth", ""),
                                    verify_event,
                                )
                            except Exception as exc:  # pylint: disable=broad-exception-caught
                                result["replay_failures"].append(
                                    {"action_type": "recover_root_app", "error": _clean_text(exc)}
                                )
                        break
                    i += 1
        result["success"] = bool(same)
        result["matched_by"] = matched_by
        result["mode"] = "home_replay" if same else "home_replay_failed"
        if same and result.get("home_attempted") and int(result.get("post_replay_back_presses") or 0) > 0:
            result["mode"] = "home_replay_backtrack"
        result["latency_ms"] = float(max(0.0, time.time() - rollback_start) * 1000.0)
        if not same:
            result["failure_analysis"] = self._analyze_rollback_failure(
                root_activity=root_activity,
                root_hash=root_hash,
                final_state=final_state,
                replay_actions=replay_actions,
                matched_by=matched_by,
                replay_anchor=result.get("replay_anchor") if isinstance(result.get("replay_anchor"), dict) else {},
                post_replay_back_stop_reason=_clean_text(result.get("post_replay_back_stop_reason") or ""),
            )
            analysis = result.get("failure_analysis") if isinstance(result.get("failure_analysis"), dict) else {}
            likely_reasons = analysis.get("likely_reasons") if isinstance(analysis, dict) else []
            result["failure_reason"] = (
                ",".join(str(x) for x in likely_reasons)
                if isinstance(likely_reasons, list) and likely_reasons
                else _clean_text(matched_by) or "root_not_restored"
            )
            result["failure_screenshot"] = self._save_trace_screenshot(
                getattr(self, "_current_goal_for_depth", ""),
                final_state,
                f"rollback_failed_step_{step_idx + 1:02d}.png",
            )
        print(
            f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
            f"rollback_home_replay_done success={result['success']} replayed={result['replayed_actions']} "
            f"matched_by={matched_by}"
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
        goal: str,
        step_idx: int,
        root_activity: str,
        page_stalled: bool,
        root_state: Any | None = None,
        current_action: json_action.JSONAction | None = None,
    ) -> tuple[bool, str]:
        if not self.enable_light_exploration:
            return False, "disabled"
        if self.light_explore_skip_destructive_goals and self._is_destructive_or_toggle_goal(goal):
            return False, "destructive_goal_skipped"
        if self.light_explore_skip_launcher and self._is_launcher_activity(root_activity):
            return False, "launcher_skipped"
        if step_idx <= 0:
            return False, "first_step_no_exploration"
        if self.light_explore_diagnostic_full:
            return True, "diagnostic_full_every_step"
        if self.light_explore_fixed_framework:
            task_mode = self._task_mode(goal)
            complexity = self._screen_complexity(root_state) if root_state is not None else {}
            target_visible = self._target_visible_in_state(root_state, goal) if root_state is not None else False
            planned_risk = self._main_action_risk_reason(current_action, goal=goal, state=root_state)
            if planned_risk and (
                task_mode == "DELETE_COMMIT" or not self.light_explore_fallback_safe_candidates
            ):
                return False, f"fixed_gate_disabled:{task_mode}:{planned_risk}"
            if self.light_explore_slot_policy_switcher:
                slots = slot_complete_evidence.parse_task_slots(goal)
                labels = self._state_semantic_summary(root_state, limit=TRACE_A11Y_LIMIT) if root_state is not None else []
                slot_evidence = slot_complete_evidence.extract_slot_evidence(
                    labels,
                    goal,
                    activity=root_activity,
                )
                if slots.task_subtype == "MEDIA_CAPTURE":
                    return False, f"slot_gate_disabled:{slots.task_subtype}:media_capture_no_speculation"
                if task_mode in {"FORM_CREATE_EDIT", "DELETE_COMMIT", "SIMPLE_VERIFY", "MEDIA_CAPTURE"}:
                    return True, f"slot_gate_enabled:{task_mode}:passive_schema_risk_only"
                if task_mode == "INFO_QUERY":
                    if slot_evidence.get("hint_type") in {"ANSWER_HINT", "AVOID_HINT"} or float(slot_evidence.get("slot_coverage") or 0.0) > 0.0:
                        return True, f"slot_gate_enabled:{slots.task_subtype}:passive_or_partial_slots"
                    if page_stalled or not target_visible or int(complexity.get("clickable", 0)) >= 8:
                        return True, f"slot_gate_enabled:{slots.task_subtype}:search_missing_slots"
                    return False, f"slot_gate_disabled:{slots.task_subtype}:no_missing_slot_signal"
                if task_mode == "NAVIGATION_SEARCH":
                    if page_stalled or not target_visible or int(complexity.get("clickable", 0)) >= 8:
                        return True, f"slot_gate_enabled:{task_mode}:navigation_search"
                    return False, f"slot_gate_disabled:{task_mode}:target_visible_low_branching"
            if task_mode in {"FORM_CREATE_EDIT", "DELETE_COMMIT", "SIMPLE_VERIFY", "MEDIA_CAPTURE"}:
                return True, f"fixed_gate_enabled:{task_mode}:passive_schema_risk_only"
            if task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH"}:
                if page_stalled:
                    return True, f"fixed_gate_enabled:{task_mode}:stalled_page"
                if not target_visible:
                    return True, f"fixed_gate_enabled:{task_mode}:target_not_visible"
                if int(complexity.get("clickable", 0)) >= 8:
                    return True, f"fixed_gate_enabled:{task_mode}:high_branching"
                return False, f"fixed_gate_disabled:{task_mode}:target_visible_low_branching"
            if page_stalled:
                return True, f"fixed_gate_enabled:{task_mode}:stalled_page"
            return False, f"fixed_gate_disabled:{task_mode}"
        if self.light_explore_search_policy == "task_gate":
            task_mode = self._task_mode(goal)
            if task_mode == "SIMPLE_VERIFY" and not page_stalled and step_idx <= 2:
                return True, f"task_gate_enabled:{task_mode}:passive_schema_risk_only"
            if task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH"}:
                return True, f"task_gate_enabled:{task_mode}"
            if task_mode in {"FORM_CREATE_EDIT", "DELETE_COMMIT"}:
                return True, f"task_gate_enabled:{task_mode}:passive_schema_risk_only"
            if page_stalled:
                return True, f"task_gate_enabled:{task_mode}:stalled_page"
            return False, f"task_gate_disabled:{task_mode}"
        if self.light_explore_force_every_step:
            return True, "force_every_step"
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
        if page_stalled:
            return True, "stalled_page"
        if self._is_launcher_activity(root_activity):
            return True, "launcher_page"
        return True, "post_planning_for_next_step"

    @staticmethod
    def _is_destructive_or_toggle_goal(goal: str) -> bool:
        low = _clean_text(goal).lower()
        patterns = (
            r"\b(delete|remove|trash|discard|clear all|erase)\b",
            r"\b(turn on|turn off|toggle|enable|disable)\b",
            r"\b(send|reply|resend|call|share)\b",
            r"\b(move the file|move .* from|copy .* file)\b",
        )
        return any(re.search(pattern, low) for pattern in patterns)

    def _planned_click_candidate_from_action(
        self,
        state: Any,
        current_action: json_action.JSONAction | None,
    ) -> dict[str, Any] | None:
        if current_action is None:
            return None
        if _clean_text(getattr(current_action, "action_type", "")).lower() != json_action.CLICK:
            return None
        x = getattr(current_action, "x", None)
        y = getattr(current_action, "y", None)
        if x is None or y is None:
            return None
        try:
            center = (int(x), int(y))
        except (TypeError, ValueError):
            return None

        nearest: tuple[float, int, Any, tuple[int, int], str] | None = None
        for idx, element in enumerate(list(getattr(state, "ui_elements", None) or [])):
            if not self._is_interactive(element):
                continue
            element_center = self._safe_center_from_element(element)
            if element_center is None:
                continue
            dist = math.dist(center, element_center)
            if dist > 220:
                continue
            merged = self._element_text(element)
            if nearest is None or dist < nearest[0]:
                nearest = (dist, idx, element, element_center, merged)

        label = "planned click"
        a11y: dict[str, Any] = {}
        merged = label
        index = -1
        key = f"planned-click:{center[0] // 48}:{center[1] // 48}:planned"
        if nearest is not None:
            _, index, element, element_center, merged = nearest
            label = _clean_text(merged) or label
            a11y = self._element_trace(element, index)
            key = "planned-element:" + self._candidate_key(element, element_center, merged)
        return {
            "index": index,
            "element": nearest[2] if nearest is not None else None,
            "center": center,
            "label": f"planned: {label}",
            "merged": f"planned action {label}",
            "key": key,
            "score": 10.0,
            "relevance": 1.0,
            "visits": 0,
            "a11y": a11y,
            "is_planned_action": True,
            "action_kind": json_action.CLICK,
            "probe_action": current_action,
        }

    def _planned_probe_candidate_from_action(
        self,
        state: Any,
        current_action: json_action.JSONAction | None,
    ) -> dict[str, Any] | None:
        click_candidate = self._planned_click_candidate_from_action(state, current_action)
        if click_candidate is not None:
            return click_candidate
        if current_action is None or not self._is_replay_safe_action(current_action):
            return None
        action_type = _clean_text(getattr(current_action, "action_type", ""))
        if action_type in {json_action.STATUS, json_action.ANSWER, json_action.UNKNOWN}:
            return None
        app_name = _clean_text(getattr(current_action, "app_name", ""))
        text = _clean_text(getattr(current_action, "text", ""))
        direction = _clean_text(getattr(current_action, "direction", ""))
        if action_type == json_action.OPEN_APP and app_name:
            label = f"planned: open {app_name}"
            merged = f"planned action open app {app_name}"
            key = f"planned-open-app:{app_name.lower()}"
        elif action_type in {json_action.SWIPE, json_action.SCROLL} and direction:
            label = f"planned: {action_type} {direction}"
            merged = f"planned action {action_type} {direction}"
            key = f"planned-{action_type}:{direction}"
        elif action_type == json_action.INPUT_TEXT and text:
            label = f'planned: type "{text[:48]}"'
            merged = f"planned action type {text}"
            key = f"planned-input:{text[:64].lower()}"
        elif action_type in {json_action.NAVIGATE_BACK, json_action.NAVIGATE_HOME, json_action.WAIT}:
            label = f"planned: {action_type}"
            merged = f"planned action {action_type}"
            key = f"planned-{action_type}"
        else:
            return None
        return {
            "index": -1,
            "element": None,
            "center": None,
            "label": label,
            "merged": merged,
            "key": key,
            "score": 10.0,
            "relevance": 1.0,
            "planning_relevance": 1.0,
            "token_relevance": 1.0,
            "visits": 0,
            "a11y": {},
            "is_planned_action": True,
            "planned_action_alignment": 1.0,
            "action_kind": action_type,
            "text": text,
            "probe_action": current_action,
        }

    @staticmethod
    def _probe_action_from_candidate(candidate: dict[str, Any]) -> json_action.JSONAction | None:
        probe_action = candidate.get("probe_action")
        if isinstance(probe_action, json_action.JSONAction):
            return probe_action
        center = candidate.get("center")
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            return None
        action_kind = _clean_text(candidate.get("action_kind") or json_action.CLICK)
        if action_kind == "type":
            return json_action.JSONAction(
                action_type=json_action.INPUT_TEXT,
                x=int(center[0]) if int(center[0]) > 0 else None,
                y=int(center[1]) if int(center[1]) > 0 else None,
                text=_clean_text(candidate.get("text")),
                clear_text=bool(candidate.get("clear_text")),
            )
        return json_action.JSONAction(
            action_type=json_action.CLICK,
            x=int(center[0]),
            y=int(center[1]),
        )

    @staticmethod
    def _is_click_action(action: json_action.JSONAction | None) -> bool:
        if action is None or action.action_type != json_action.CLICK:
            return False
        return getattr(action, "x", None) is not None and getattr(action, "y", None) is not None

    def _planned_action_skip_reason(self, action: json_action.JSONAction | None) -> str:
        if not self.light_explore_safe_click_only:
            return ""
        if action is not None and action.action_type == json_action.OPEN_APP and _clean_text(getattr(action, "app_name", "")):
            return ""
        if not self._is_click_action(action):
            action_type = _clean_text(getattr(action, "action_type", "")) or "none"
            return f"unsafe_anchor_action:{action_type}"
        return ""

    def _main_action_risk_reason(
        self,
        action: json_action.JSONAction | None,
        goal: str = "",
        state: Any | None = None,
    ) -> str:
        if action is None:
            return ""
        action_type = _clean_text(getattr(action, "action_type", "")).lower()
        if action_type in {json_action.STATUS, json_action.ANSWER, json_action.WAIT, json_action.OPEN_APP}:
            return ""
        if action_type in {json_action.LONG_PRESS, json_action.INPUT_TEXT, json_action.KEYBOARD_ENTER}:
            return f"planned_{action_type}_not_speculated"
        if action_type in {json_action.SWIPE, json_action.SCROLL}:
            return ""
        if action_type == json_action.CLICK:
            pseudo = self._planned_click_candidate_from_action(state, action) if state is not None else None
            label = _clean_text((pseudo or {}).get("label") or "")
            if self._is_commit_probe_label(label) or self._is_risky_probe_text(label):
                return "planned_click_commit_or_risky"
            if self._task_mode(goal) == "DELETE_COMMIT" and re.search(
                r"\b(delete|remove|trash|confirm|ok|done|yes)\b",
                label.lower(),
            ):
                return "planned_delete_commit_stage"
        return ""

    def _target_visible_in_state(self, state: Any, goal: str) -> bool:
        labels = " ".join(self._state_semantic_summary(state, limit=40)).lower()
        if not labels:
            return False
        for entity in self._task_entities(goal):
            if len(entity) >= 3 and entity.lower() in labels:
                return True
        entry = self._extract_goal_entry_text(goal).lower()
        return bool(entry and entry in labels)

    def _screen_complexity(self, state: Any) -> dict[str, int]:
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        clickable = 0
        editable = 0
        for element in ui_elements:
            if self._is_interactive(element):
                clickable += 1
            class_name = _clean_text(getattr(element, "class_name", "")).lower()
            if bool(getattr(element, "is_editable", False)) or "edittext" in class_name:
                editable += 1
        return {"ui_elements": len(ui_elements), "clickable": clickable, "editable": editable}

    def _safe_fallback_probe_candidates(
        self,
        candidates: list[dict[str, Any]],
        planned_candidate: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Pick safe evidence candidates when the planned action cannot be probed transactionally."""
        planned_center = planned_candidate.get("center") if isinstance(planned_candidate, dict) else None
        out: list[dict[str, Any]] = []
        for candidate in candidates:
            if bool(candidate.get("is_planned_action")):
                continue
            action_kind = _clean_text(candidate.get("action_kind") or json_action.CLICK)
            if action_kind != json_action.CLICK:
                continue
            center = candidate.get("center")
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            if (
                isinstance(planned_center, (list, tuple))
                and len(planned_center) >= 2
                and math.dist(planned_center, center) <= 80
            ):
                continue
            if self._is_transaction_unsafe_candidate(candidate):
                continue
            merged = str(candidate.get("merged") or candidate.get("label") or "")
            if self._is_risky_probe_text(merged) or self._is_commit_probe_label(merged):
                continue
            if (
                self.light_explore_require_a11y_anchor
                and int(candidate.get("index") or -1) < 0
            ):
                continue
            key = _clean_text(candidate.get("key"))
            if key and key in self._no_effect_probe_keys and not self.light_explore_diagnostic_full:
                continue
            out.append(candidate)
        out.sort(
            key=lambda c: (
                float(c.get("score") or 0.0),
                float(c.get("planning_relevance") or 0.0),
                float(c.get("relevance") or 0.0),
            ),
            reverse=True,
        )
        return out

    def _candidate_trace(self, candidate: dict[str, Any]) -> dict[str, Any]:
        center = candidate.get("center")
        operator, operator_reason = self._candidate_operator(candidate, "")
        return {
            "index": candidate.get("index"),
            "label": _clean_text(candidate.get("label")),
            "center": [int(center[0]), int(center[1])] if isinstance(center, (list, tuple)) and len(center) >= 2 else None,
            "score": float(candidate.get("score") or 0.0),
            "relevance": float(candidate.get("relevance") or 0.0),
            "planning_relevance": float(candidate.get("planning_relevance") or 0.0),
            "token_relevance": float(candidate.get("token_relevance") or 0.0),
            "affordance_bonus": float(candidate.get("affordance_bonus") or 0.0),
            "visits": int(candidate.get("visits") or 0),
            "key": _clean_text(candidate.get("key")),
            "merged": _clean_text(candidate.get("merged")),
            "a11y": candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {},
            "is_planned_action": bool(candidate.get("is_planned_action")),
            "planned_action_alignment": float(candidate.get("planned_action_alignment") or 0.0),
            "action_kind": _clean_text(candidate.get("action_kind") or "click"),
            "text": _clean_text(candidate.get("text")),
            "clear_text": bool(candidate.get("clear_text")),
            "operator": _clean_text(candidate.get("operator") or operator),
            "operator_reason": _clean_text(candidate.get("operator_reason") or operator_reason),
            "risk_boundary": bool(candidate.get("risk_boundary")),
            "layout_region": _clean_text(candidate.get("layout_region")),
            "semantic_role": _clean_text(candidate.get("semantic_role")),
            "estimated_evidence_gain": float(candidate.get("estimated_evidence_gain") or 0.0),
            "search_strategy_score": float(candidate.get("search_strategy_score") or 0.0),
            "score_components": candidate.get("score_components") if isinstance(candidate.get("score_components"), dict) else {},
            "final_score": float(candidate.get("final_score") or candidate.get("search_strategy_score") or 0.0),
            "reason_selected": _clean_text(candidate.get("reason_selected")),
            "reason_rejected": _clean_text(candidate.get("reason_rejected")),
            "strategy_group": _clean_text(candidate.get("strategy_group")),
            "session_branch_key": _clean_text(candidate.get("session_branch_key")),
            "session_branch_utility": float(candidate.get("session_branch_utility") or 0.0),
            "session_depth2_utility": float(candidate.get("session_depth2_utility") or 0.0),
            "session_value_features": candidate.get("session_value_features")
            if isinstance(candidate.get("session_value_features"), dict)
            else {},
            "session_adaptive_weights": candidate.get("session_adaptive_weights")
            if isinstance(candidate.get("session_adaptive_weights"), dict)
            else {},
        }

    @staticmethod
    def _extract_depth1_depth2_pair(
        observations: list[dict[str, Any]],
        *,
        allow_unplanned_root: bool = False,
        planned_action: json_action.JSONAction | None = None,
        planned_match_threshold: float = 0.25,
    ) -> list[tuple[float, dict[str, Any], dict[str, Any], dict[str, Any], float]]:
        candidates: list[tuple[float, dict[str, Any], dict[str, Any], dict[str, Any], float]] = []
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            steps = [step for step in list(observation.get("steps") or []) if isinstance(step, dict)]
            depth1 = next((step for step in steps if int(step.get("depth") or 0) == 1), None)
            depth2 = next((step for step in steps if int(step.get("depth") or 0) == 2), None)
            if depth1 is None or depth2 is None:
                continue
            root_candidate = depth1.get("candidate") if isinstance(depth1.get("candidate"), dict) else {}
            if not isinstance(root_candidate, dict):
                continue

            if bool(root_candidate.get("is_planned_action")):
                root_match = 1.0
            elif not allow_unplanned_root:
                continue
            elif planned_action is None:
                root_match = 1.0
            else:
                try:
                    root_match = ExplorerElementAgent._planned_action_alignment(root_candidate, planned_action)
                except Exception:  # pylint: disable=broad-exception-caught
                    root_match = 0.0
                if root_match < planned_match_threshold:
                    continue

            score = float(depth1.get("score") or 0.0) + float(depth2.get("score") or 0.0) + float(root_match) * 4.0
            candidates.append((score, observation, depth1, depth2, float(root_match)))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def _save_trace_screenshot(self, goal: str, state: Any, filename: str) -> str:
        if not self.light_explore_save_screenshots:
            return ""
        start = time.time()
        self._state_acquisition_metrics["screenshot_calls"] = (
            float(self._state_acquisition_metrics.get("screenshot_calls") or 0.0) + 1.0
        )
        task_dir = self._task_output_dir(goal)
        if not task_dir:
            return ""
        try:
            out_path = os.path.join(task_dir, filename)
            Image.fromarray(state.pixels).save(out_path)
            self._append_latency_profile_event(
                goal,
                {
                    "event": "screenshot_save",
                    "filename": filename,
                    "latency_ms": float(max(0.0, time.time() - start) * 1000.0),
                    "path": out_path,
                },
            )
            return out_path
        except Exception:  # pylint: disable=broad-exception-caught
            self._append_latency_profile_event(
                goal,
                {
                    "event": "screenshot_save",
                    "filename": filename,
                    "latency_ms": float(max(0.0, time.time() - start) * 1000.0),
                    "failed": True,
                },
            )
            return ""

    def _append_exploration_trace(self, goal: str, trace: dict[str, Any]) -> None:
        self._last_exploration_trace = dict(trace)
        self._exploration_traces.append(dict(trace))
        self._all_exploration_traces.append(dict(trace))
        for rollback in list(trace.get("rollbacks") or []):
            if isinstance(rollback, dict):
                self._rollback_traces.append(dict(rollback))
                self._all_rollback_traces.append(dict(rollback))
        task_dir = self._task_output_dir(goal)
        if not task_dir:
            return
        try:
            with open(os.path.join(task_dir, "exploration_trace.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
            with open(os.path.join(task_dir, "rollback_trace.jsonl"), "a", encoding="utf-8") as f:
                for rollback in list(trace.get("rollbacks") or []):
                    f.write(json.dumps(rollback, ensure_ascii=False, default=str) + "\n")
            with open(os.path.join(task_dir, "t2_branch_debug.jsonl"), "a", encoding="utf-8") as f:
                for row in list(trace.get("t2_branch_debug") or []):
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            with open(os.path.join(task_dir, "evidence_capsules.jsonl"), "a", encoding="utf-8") as f:
                for capsule in list(trace.get("evidence_capsules") or []):
                    f.write(json.dumps(capsule, ensure_ascii=False, default=str) + "\n")
            self._write_state_acquisition_metrics(goal)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[EXPLORE {_now_hms()}] trace_write_failed: {exc}")

    def _append_match_trace(self, goal: str, trace: dict[str, Any]) -> None:
        self._exploration_match_traces.append(dict(trace))
        self._all_exploration_match_traces.append(dict(trace))
        task_dir = self._task_output_dir(goal)
        if not task_dir:
            return
        try:
            with open(os.path.join(task_dir, "exploration_match_trace.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[EXPLORE {_now_hms()}] match_trace_write_failed: {exc}")

    @staticmethod
    def _informative_observed_elements(labels: list[str]) -> list[str]:
        generic = {
            "action bar root",
            "android",
            "app",
            "bottom sheet",
            "button",
            "container",
            "content",
            "content panel",
            "coordinator",
            "decor content parent",
            "design bottom sheet",
            "drawer",
            "frame layout",
            "linear layout",
            "list",
            "main content",
            "navigation bar",
            "recycler view",
            "relative layout",
            "root",
            "scroll view",
            "status bar",
            "touch outside",
            "view",
            "view bg",
        }
        out: list[str] = []
        seen: set[str] = set()
        for label in labels:
            cleaned = _clean_text(label)
            low = cleaned.lower()
            if not cleaned or low in generic:
                continue
            if len(low) <= 1:
                continue
            if re.fullmatch(r"(content|container|view|button|textview|imageview)[0-9_ -]*", low):
                continue
            if low not in seen:
                seen.add(low)
                out.append(cleaned)
        return out

    def _evidence_observed_elements(self, labels: list[str], goal: str = "") -> list[str]:
        """Return labels useful as state evidence even when no next action is safe to recommend."""
        informative = self._informative_observed_elements(labels)
        if not informative:
            return []
        goal_tokens = set(self._goal_tokens(goal))
        app_keywords = set(self._goal_app_keywords(goal))
        weak_chrome = {
            "action bar",
            "button",
            "cancel",
            "change view",
            "close",
            "go to today",
            "home",
            "main menu",
            "menu",
            "more options",
            "navigate up",
            "ok",
            "search",
            "settings",
            "toolbar",
            "top app bar",
        }
        out: list[str] = []
        seen: set[str] = set()
        for label in informative:
            low = label.lower()
            if low in seen:
                continue
            seen.add(low)
            if low in weak_chrome:
                continue
            label_tokens = {
                token
                for token in re.findall(r"[a-z0-9]{3,}", low)
                if token not in GOAL_TOKEN_STOPWORDS
            }
            has_goal_overlap = bool((label_tokens - app_keywords).intersection(goal_tokens))
            has_number_or_date = bool(re.search(r"\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", low))
            has_specific_shape = len(low) >= 8 and not re.fullmatch(r"[a-z ]{1,10}", low)
            if has_goal_overlap or has_number_or_date or has_specific_shape:
                out.append(label)
        if out:
            return out[:8]
        # Fall back to a small number of non-chrome labels so matched branches
        # can still provide page-level evidence, but avoid flooding the prompt.
        fallback = [x for x in informative if x.lower() not in weak_chrome]
        return fallback[:5]

    def _is_actionable_match(self, item: dict[str, Any], goal: str = "") -> bool:
        next_candidate = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
        if not next_candidate:
            return False
        if self._is_transaction_unsafe_candidate(next_candidate):
            return False
        next_label = _clean_text(next_candidate.get("label")) or _clean_text(item.get("next_label"))
        if not next_label:
            return False
        merged = _clean_text(next_candidate.get("merged")) or next_label
        if self._is_noise_probe_text(merged):
            return False
        app_keywords = set(self._goal_app_keywords(goal))
        if next_label.lower() in app_keywords or merged.lower() in app_keywords:
            return False
        goal_tokens = self._goal_tokens(goal)
        label_goal_score = self._lexical_overlap_score(
            merged=merged,
            goal_tokens=goal_tokens,
            app_keywords=[],
        )
        command_like = any(
            token in merged.lower()
            for token in (
                "add",
                "create",
                "delete",
                "expense log",
                "filter",
                "find",
                "log",
                "menu",
                "new",
                "remove",
                "save",
                "search",
            )
        )
        if label_goal_score <= 0.05 and not command_like:
            return False
        action_kind = _clean_text(next_candidate.get("action_kind") or json_action.CLICK).lower()
        a11y = next_candidate.get("a11y") if isinstance(next_candidate.get("a11y"), dict) else {}
        class_name = _clean_text(a11y.get("class_name")).lower()
        resource_id = _clean_text(a11y.get("resource_id")).lower()
        if action_kind == json_action.CLICK and (
            "message_body" in resource_id
            or "thread_message_body" in resource_id
        ):
            return False
        field_labels = {
            "first name",
            "last name",
            "company",
            "phone",
            "email",
            "name",
            "number",
            "title",
            "description",
            "notes",
        }
        if action_kind == json_action.CLICK and (
            "edittext" in class_name or next_label.lower() in field_labels
        ):
            return False
        observed = self._informative_observed_elements(list(item.get("observed_elements") or []))
        goal_low = _clean_text(goal).lower()
        label_low = next_label.lower()
        merged_low = merged.lower()
        if self._is_answer_or_lookup_goal(goal):
            answer_task_bad_next_action = any(
                token in merged_low
                for token in (
                    "add",
                    "create",
                    "delete",
                    "done",
                    "edit",
                    "finish",
                    "new",
                    "ok",
                    "remove",
                    "save",
                    "submit",
                )
            )
            if answer_task_bad_next_action:
                return False
        if any(token in goal_low for token in ("create", "add", "new")):
            create_affordance = any(
                token in merged_low
                for token in (
                    "add",
                    "create",
                    "new",
                    "plus",
                    "first name",
                    "last name",
                    "name",
                    "phone",
                    "number",
                )
            )
            contact_bucket = label_low in {"phone contacts", "contacts", "all contacts"}
            if contact_bucket or not create_affordance:
                return False
        if observed:
            return True
        # Keep a match only when the suggested next action itself carries
        # task-level information. This blocks hints such as "click content" or
        # "click pro expense" with only generic container observations.
        return not self._is_noise_probe_text(next_label) and len(next_label) >= 3

    def _is_evidence_only_match(self, item: dict[str, Any], goal: str = "") -> bool:
        if not item.get("matched"):
            return False
        if self._is_actionable_match(item, goal=goal):
            return False
        if int(item.get("depth_reached") or 0) < 2:
            return False
        observed = self._evidence_observed_elements(list(item.get("observed_elements") or []), goal=goal)
        if not observed:
            return False
        next_candidate = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
        merged = _clean_text(next_candidate.get("merged") or next_candidate.get("label") or item.get("next_label"))
        # Keep query tasks from being nudged toward creation/deletion controls.
        if self._is_answer_or_lookup_goal(goal) and re.search(
            r"\b(add|create|delete|done|edit|finish|new|remove|save|submit)\b",
            merged.lower(),
        ):
            return False
        return True

    def _build_prompt_context_from_observations(
        self,
        observations: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        eligible = [
            obs
            for obs in observations
            if bool((obs.get("rollback") or {}).get("success"))
            and bool(obs.get("changed"))
            and (
                self._informative_observed_elements(list(obs.get("observed_elements") or []))
                or int(obs.get("depth_reached") or 0) >= 2
            )
        ]
        eligible.sort(
            key=lambda obs: (
                1 if bool(obs.get("changed")) else 0,
                float(obs.get("score") or 0.0),
                int(obs.get("depth_reached") or 0),
            ),
            reverse=True,
        )
        selected = eligible[: int(self.light_explore_prompt_result_limit)]
        lines: list[str] = []
        selected_traces: list[dict[str, Any]] = []
        for rank, obs in enumerate(selected, start=1):
            path = " -> ".join([_clean_text(x) for x in list(obs.get("labels") or []) if _clean_text(x)])
            if not path:
                path = "candidate"
            activity = _clean_text(obs.get("after_activity")).split("/")[-1] or "a different screen"
            observed = self._informative_observed_elements(list(obs.get("observed_elements") or []))[:5]
            observed_text = "; ".join(observed) if observed else "no compact labels captured"
            rollback = obs.get("rollback") if isinstance(obs.get("rollback"), dict) else {}
            line = (
                f"{rank}. I simulated `{path}` and returned by {rollback.get('level')}/{rollback.get('mode')}. "
                f"The preview ended on `{activity}` and showed: {observed_text}."
            )
            lines.append(line)
            selected_trace = {
                "rank": rank,
                "branch_id": obs.get("branch_id"),
                "path": path,
                "after_activity": obs.get("after_activity"),
                "observed_elements": observed,
                "score": float(obs.get("score") or 0.0),
                "depth_reached": int(obs.get("depth_reached") or 0),
                "prompt_line": line,
            }
            selected_traces.append(selected_trace)
        if not lines:
            return "", []
        context = (
            "Rollback-verified preview for the next step:\n"
            + "\n".join(lines)
        )
        return context, selected_traces

    def _state_matches_explored_step(
        self,
        state: Any,
        current_activity: str,
        explored_step: dict[str, Any],
    ) -> tuple[bool, str]:
        curr_activity = self._normalize_activity_name(current_activity or self._foreground_activity_name())
        target_activity = self._normalize_activity_name(str(explored_step.get("after_activity") or ""))
        same_activity = bool(curr_activity and target_activity and curr_activity == target_activity)
        try:
            target_hash = int(explored_step.get("after_hash"))
        except Exception:  # pylint: disable=broad-exception-caught
            target_hash = -1
        curr_hash = self._state_hash(state)
        if target_hash < 0 or curr_hash < 0:
            return same_activity, "activity_match_only" if same_activity else "activity_mismatch"
        try:
            diff = int(_hash_diff(target_hash, curr_hash))
        except Exception:  # pylint: disable=broad-exception-caught
            diff = 10**9
        target_labels = {
            _clean_text(x).lower()
            for x in self._informative_observed_elements(list(explored_step.get("observed_elements") or []))
            if _clean_text(x)
        }
        current_labels = {
            _clean_text(x).lower()
            for x in self._informative_observed_elements(self._state_semantic_summary(state))
            if _clean_text(x)
        }
        if same_activity and diff <= 4:
            return True, f"activity+strict_phash<=4:{diff}"
        if same_activity and len(target_labels) >= 3 and len(current_labels) >= 3:
            overlap = target_labels.intersection(current_labels)
            jaccard = float(len(overlap)) / float(len(target_labels.union(current_labels)) or 1)
            if diff <= int(self.light_explore_hash_threshold + 4) and jaccard >= 0.35:
                return True, f"activity+phash+a11y_jaccard:{diff}/{jaccard:.2f}"
            if jaccard >= 0.55:
                return True, f"activity+a11y_jaccard:{jaccard:.2f}"
        return False, f"activity_match={same_activity}|phash_diff={diff}"

    def _anchor_overlap_alignment(
        self,
        state: Any,
        current_activity: str,
        explored_step: dict[str, Any],
    ) -> dict[str, Any]:
        curr_activity = self._normalize_activity_name(current_activity or self._foreground_activity_name())
        explored_activity = self._normalize_activity_name(str(explored_step.get("after_activity") or ""))
        explored_anchors = [
            _clean_text(x).lower()
            for x in self._informative_observed_elements(list(explored_step.get("observed_elements") or []))
            if _clean_text(x)
        ]
        current_anchors = [
            _clean_text(x).lower()
            for x in self._informative_observed_elements(self._state_semantic_summary(state, limit=80))
            if _clean_text(x)
        ]
        explored_set = set(explored_anchors)
        current_set = set(current_anchors)
        overlap = len(explored_set.intersection(current_set)) / float(max(1, len(explored_set)))
        try:
            target_hash = int(explored_step.get("after_hash"))
            curr_hash = self._state_hash(state)
            phash_diff = int(_hash_diff(target_hash, curr_hash)) if target_hash >= 0 and curr_hash >= 0 else None
        except Exception:  # pylint: disable=broad-exception-caught
            phash_diff = None
        activity_match = bool(curr_activity and explored_activity and curr_activity == explored_activity)
        aligned = bool(activity_match and overlap >= 0.50)
        return {
            "activity_match": activity_match,
            "explored_activity": explored_activity,
            "current_activity": curr_activity,
            "explored_anchors": explored_anchors[:40],
            "current_anchors": current_anchors[:40],
            "anchor_overlap": float(overlap),
            "state_match_score": float(overlap if activity_match else 0.0),
            "aligned": aligned,
            "phash_diff_diagnostic": phash_diff,
        }

    def _search_anchor_count(self, state: Any) -> int:
        labels = [_clean_text(x).lower() for x in self._state_semantic_summary(state, limit=80)]
        merged = " ".join(labels)
        anchors = {
            "search": bool(re.search(r"\bsearch\b", merged)),
            "clear query": "clear query" in merged,
            "submit query": "submit query" in merged,
            "back": bool(re.search(r"\b(back|navigate up)\b", merged)),
            "search edit frame": bool(re.search(r"\b(search_src_text|search edit|edit text|query)\b", merged)),
            "search plate": bool(re.search(r"\b(search plate|search_plate)\b", merged)),
            "toolbar": bool(re.search(r"\b(toolbar|action bar|top app bar)\b", merged)),
        }
        for element in list(getattr(state, "ui_elements", None) or []):
            text = " ".join(
                _clean_text(getattr(element, key, "")).lower()
                for key in ("text", "content_description", "hint_text", "resource_id", "class_name")
            )
            if re.search(r"\b(search|query|search_src_text|search_plate)\b", text):
                anchors["search edit frame"] = True
                anchors["search"] = True
            if "clear query" in text:
                anchors["clear query"] = True
            if "submit query" in text:
                anchors["submit query"] = True
        return sum(1 for value in anchors.values() if value)

    def _shortcut_plan_pattern(self, plan: dict[str, Any]) -> str:
        action = plan.get("shortcut_t2_action") if isinstance(plan.get("shortcut_t2_action"), dict) else {}
        if bool(action.get("is_safe_search_input")):
            return "SearchInputShortcut"
        if bool(action.get("is_exact_result_click")):
            return "ExactResultClickShortcut"
        return _clean_text(plan.get("shortcut_pattern") or action.get("shortcut_pattern") or "generic")

    def _shortcut_t1_state_matches(
        self,
        *,
        goal: str,
        plan: dict[str, Any],
        state: Any,
        current_activity: str,
    ) -> tuple[bool, str, float]:
        explored = plan.get("explored_t1_state") if isinstance(plan.get("explored_t1_state"), dict) else {}
        t1_step = {
            "after_activity": explored.get("activity"),
            "after_hash": explored.get("p_hash"),
            "observed_elements": explored.get("anchor_labels") or [],
        }
        pattern = self._shortcut_plan_pattern(plan)
        curr_activity = self._normalize_activity_name(current_activity or self._foreground_activity_name())
        target_activity = self._normalize_activity_name(_clean_text(explored.get("activity")))
        same_activity = bool(curr_activity and target_activity and curr_activity == target_activity)
        if pattern == "SearchInputShortcut":
            anchor_count = self._search_anchor_count(state)
            target_visible = self._state_has_search_ui(state)
            if same_activity and target_visible and anchor_count >= 2:
                return True, "search_anchor", 1.0
        if pattern == "ExactResultClickShortcut":
            t2 = plan.get("shortcut_t2_action") if isinstance(plan.get("shortcut_t2_action"), dict) else {}
            label = _clean_text(t2.get("label") or t2.get("value")).lower()
            labels = [_clean_text(x).lower() for x in self._state_semantic_summary(state, limit=80)]
            if same_activity and label and any(label in item or item in label for item in labels):
                return True, "exact_label", 1.0
        matched, reason = self._state_matches_explored_step(state, current_activity, t1_step)
        score = 1.0 if matched else 0.0
        if "a11y_jaccard" in reason:
            try:
                score = float(reason.rsplit(":", 1)[-1].split("/")[-1])
            except Exception:  # pylint: disable=broad-exception-caught
                score = 1.0 if matched else 0.0
        return matched, ("phash" if matched and "phash" in reason else reason), score

    def _build_prompt_context_from_state_aligned_matches(
        self,
        matches: list[dict[str, Any]],
        goal: str = "",
        current_state: Any | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        if self.light_explore_hint_policy == "strict":
            return self._build_strict_prompt_context_from_state_aligned_matches(
                matches=matches,
                goal=goal,
                current_state=current_state,
            )
        matches.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                int(item.get("depth_reached") or 0),
            ),
            reverse=True,
        )
        actionable = [
            item for item in matches if self._is_actionable_match(item, goal=goal)
        ]
        evidence_only = [
            item for item in matches if self._is_evidence_only_match(item, goal=goal)
        ]
        selected = (actionable + evidence_only)[: int(self.light_explore_prompt_result_limit)]
        lines: list[str] = []
        selected_traces: list[dict[str, Any]] = []
        for rank, item in enumerate(selected, start=1):
            evidence_mode = not self._is_actionable_match(item, goal=goal)
            next_candidate = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
            next_label = _clean_text(next_candidate.get("label")) or _clean_text(item.get("next_label")) or "that element"
            action_kind = _clean_text(next_candidate.get("action_kind") or "click").lower()
            center = next_candidate.get("center")
            center_text = ""
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                center_text = f" near pixel [{int(center[0])}, {int(center[1])}]"
            if action_kind == "type":
                target_text = _clean_text(next_candidate.get("text"))
                a11y = next_candidate.get("a11y") if isinstance(next_candidate.get("a11y"), dict) else {}
                field_label = (
                    _clean_text(a11y.get("text"))
                    or _clean_text(a11y.get("hint_text"))
                    or _clean_text(a11y.get("content_description"))
                )
                if field_label:
                    action_text = f"typing `{target_text}` into the field currently showing `{field_label}`"
                else:
                    action_text = f"typing `{target_text}` into the focused input field"
            else:
                action_text = f"clicking `{next_label}`{center_text}"
            observed = (
                self._evidence_observed_elements(list(item.get("observed_elements") or []), goal=goal)[:6]
                if evidence_mode
                else self._informative_observed_elements(list(item.get("observed_elements") or []))[:6]
            )
            observed_text = "; ".join(observed) if observed else "no compact labels captured"
            if evidence_mode:
                line = (
                    f"{rank}. Evidence only, not a recommended action. The current screen matches an intermediate "
                    f"screen from a rollback-verified preview ({item.get('matched_by')}). When that preview continued "
                    f"after `{item.get('matched_prefix')}`, the next screen showed: {observed_text}. "
                    "Use these labels as context about what this branch reveals; do not click the low-confidence "
                    "preview target unless it matches your own plan."
                )
            else:
                line = (
                    f"{rank}. The current screen matches the intermediate screen from a previous rollback-verified "
                    f"preview ({item.get('matched_by')}). In that preview, after `{item.get('matched_prefix')}`, "
                    f"the next action was {action_text}, and the following screen showed: {observed_text}. "
                    "Use it only when the current UI labels are consistent with this preview."
                )
            lines.append(line)
            trace = dict(item)
            trace["rank"] = rank
            trace["prompt_line"] = line
            trace["hint_kind"] = "evidence_only" if evidence_mode else "actionable"
            selected_traces.append(trace)
        if not lines:
            return "", []
        context = (
            "Rollback-verified exploration preview. The current screen appears to match a previously "
            "simulated intermediate state. Some items may be evidence-only and should not be treated as commands:\n"
            + "\n".join(lines)
        )
        return context, selected_traces

    @staticmethod
    def _upper_bound_threshold(evidence_type: str, summary_all_debug: bool = False) -> float:
        del summary_all_debug
        thresholds = {
            "ANSWER_HINT": 0.70,
            "ACTION_HINT": 0.70,
            "AVOID_HINT": 0.65,
            "SCHEMA_HINT": 0.65,
            "RISK_HINT": 0.60,
        }
        return float(thresholds.get(_clean_text(evidence_type), 1.0))

    def _upper_bound_exact_entity_match_score(
        self,
        item: dict[str, Any],
        observed: list[str],
        goal: str,
    ) -> float:
        haystack = " ".join(
            [
                " ".join(observed),
                _clean_text(item.get("next_label")),
                _clean_text(item.get("matched_prefix")),
                _clean_text(item.get("prompt_line")),
            ]
        ).lower()
        entities = [e for e in self._task_entities(goal) if len(e) >= 3]
        for entity in entities:
            if re.search(rf"(?<![a-z0-9]){re.escape(entity.lower())}(?![a-z0-9])", haystack):
                return 1.0
        if bool(item.get("answer_complete") or item.get("slot_complete") or item.get("target_visible")):
            return 1.0
        return 0.0

    @staticmethod
    def _upper_bound_observed_noise_only(observed: list[str]) -> bool:
        cleaned = [_clean_text(x).lower() for x in observed if _clean_text(x)]
        if not cleaned:
            return True
        noise_patterns = (
            "status bar",
            "navigation bar",
            "system icons",
            "notification",
            "launch animation",
            "cutout space",
            "gesture handle",
            "clock",
            "battery",
        )
        informative = [
            item for item in cleaned
            if not any(pattern in item for pattern in noise_patterns)
        ]
        return not informative

    def _upper_bound_slot_or_target_score(
        self,
        item: dict[str, Any],
        evidence_type: str,
        observed: list[str],
        goal: str,
    ) -> float:
        if bool(item.get("answer_complete") or item.get("slot_complete")):
            return 1.0
        slot_evidence = item.get("slot_evidence") if isinstance(item.get("slot_evidence"), dict) else {}
        if slot_evidence.get("slot_complete"):
            return 1.0
        exact = self._upper_bound_exact_entity_match_score(item, observed, goal)
        if exact >= 1.0:
            return 1.0
        candidate = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
        operator = _clean_text(item.get("operator") or candidate.get("operator"))
        safe_action_boundary_operators = {
            "SearchPeek",
            "FilterPeek",
            "DetailPeek",
            "StatsPeek",
            "ListInspect",
            "NavigationPeek",
            "FormSchema",
        }
        try:
            depth_reached = int(item.get("depth_reached") or item.get("evidence_depth") or 0)
        except (TypeError, ValueError):
            depth_reached = 0
        if evidence_type == "ACTION_HINT":
            if self._is_transaction_unsafe_candidate(candidate):
                return 0.0
            if bool(item.get("rollback_verified")) and depth_reached >= 1 and operator in safe_action_boundary_operators:
                return 0.80
            if float(item.get("evidence_gain") or 0.0) > 0.0 and operator in safe_action_boundary_operators:
                return 0.60
        if evidence_type == "SCHEMA_HINT":
            if bool(item.get("rollback_verified")) and operator in {"SearchPeek", "FilterPeek", "FormSchema"}:
                return 0.65
        if evidence_type == "RISK_HINT":
            return 0.60
        if float(item.get("evidence_gain") or 0.0) > 0.0:
            return 0.5 if exact >= 1.0 else 0.0
        return 0.0

    def _upper_bound_evidence_type(
        self,
        item: dict[str, Any],
        candidate: dict[str, Any],
        observed: list[str],
        goal: str,
    ) -> str:
        source_type = _clean_text(item.get("evidence_type") or item.get("boundary_type"))
        operator = _clean_text(item.get("operator") or candidate.get("operator"))
        label = _clean_text(candidate.get("label") or item.get("next_label")).lower()
        task_mode = self._task_mode(goal)
        if operator == "RiskBoundary" or self._is_transaction_unsafe_candidate(candidate):
            return "RISK_HINT"
        if source_type in {"ANSWER_HINT", "ACTION_HINT", "AVOID_HINT", "SCHEMA_HINT", "RISK_HINT"}:
            return source_type
        if task_mode == "INFO_QUERY":
            facts = self._extract_answer_facts_from_labels(observed, goal, limit=3)
            if facts:
                if any(_clean_text(x).lower().startswith("avoid ") for x in facts):
                    return "AVOID_HINT"
                return "ANSWER_HINT"
        if operator in {"SearchPeek", "FilterPeek", "FormSchema"}:
            return "SCHEMA_HINT"
        if operator in {"NavigationPeek", "DetailPeek", "StatsPeek", "ListInspect"}:
            if task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH"}:
                return "ACTION_HINT"
        if task_mode == "EVENT_QUERY" and re.search(r"\b(new event|add event|create)\b", label):
            return "AVOID_HINT"
        return "NONE"

    def _upper_bound_render_evidence(
        self,
        item: dict[str, Any],
        evidence_type: str,
        candidate: dict[str, Any],
        observed: list[str],
        goal: str,
    ) -> str:
        label = _clean_text(candidate.get("label") or item.get("next_label") or item.get("matched_prefix") or "target")
        operator = _clean_text(item.get("operator") or candidate.get("operator") or "Other")
        facts = self._extract_answer_facts_from_labels(observed, goal, limit=3)
        facts = [x for x in facts if not _clean_text(x).lower().startswith("avoid ")]
        center = candidate.get("center")
        center_text = ""
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            center_text = f" near [{int(center[0])}, {int(center[1])}]"
        if evidence_type == "ANSWER_HINT":
            body = "; ".join(facts[:3] or observed[:3])
            return f"[ANSWER_HINT] {body}"
        if evidence_type == "ACTION_HINT":
            if _clean_text(candidate.get("action_kind")).lower() in {"type", json_action.INPUT_TEXT}:
                text = _clean_text(candidate.get("text") or self._extract_goal_search_text(goal))
                return f"[ACTION_HINT] Type `{text}` into the visible search/input field."
            return f"[ACTION_HINT] If visible and task-relevant, click `{label}`{center_text}; operator={operator}."
        if evidence_type == "AVOID_HINT":
            body = "; ".join(observed[:3]) or "it did not expose task-relevant evidence"
            return f"[AVOID_HINT] Avoid `{label}` because the preview showed {body}."
        if evidence_type == "SCHEMA_HINT":
            body = "; ".join(observed[:5])
            return f"[SCHEMA_HINT] `{label}` exposes schema/search/filter controls: {body}."
        if evidence_type == "RISK_HINT":
            return f"[RISK_HINT] `{label}` is a risky or commit action; do not execute speculatively."
        return ""

    def _build_upper_bound_prompt_context_from_state_aligned_matches(
        self,
        matches: list[dict[str, Any]],
        goal: str = "",
        current_state: Any | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        buckets: dict[str, list[dict[str, Any]]] = {
            "ANSWER_HINT": [],
            "ACTION_HINT": [],
            "AVOID_HINT": [],
            "SCHEMA_HINT": [],
            "RISK_HINT": [],
        }
        rejected: list[dict[str, Any]] = []
        task_normalized = _clean_text(goal)
        current_activity = _clean_text(self._foreground_activity_name())
        current_package = current_activity.split("/", 1)[0] if "/" in current_activity else current_activity

        def _same_task_or_app(trace_goal: str, trace_activity: str) -> bool:
            trace_goal_clean = _clean_text(trace_goal)
            if trace_goal_clean and trace_goal_clean == task_normalized:
                return True
            trace_activity_norm = _clean_text(trace_activity)
            trace_package = trace_activity_norm.split("/", 1)[0] if "/" in trace_activity_norm else trace_activity_norm
            return bool(current_package and trace_package and current_package == trace_package)

        def _status_or_system_label(label: str) -> bool:
            low = _clean_text(label).lower()
            if not low:
                return True
            noise_markers = (
                "android system notification",
                "systemui",
                "system ui",
                "status bar",
                "notification icon",
                "switch input method",
                "more features",
                "battery ",
                "airplane mode",
                "no phone",
            )
            if any(marker in low for marker in noise_markers):
                return True
            if re.fullmatch(r"\d{1,2}:\d{2}", low):
                return True
            return False

        for item in sorted(matches, key=lambda x: float(x.get("confidence") or x.get("score") or 0.0), reverse=True):
            candidate = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
            observed_raw = self._evidence_observed_elements(list(item.get("observed_elements") or []), goal=goal)
            observed = [x for x in observed_raw if not _status_or_system_label(x)]
            evidence_type = self._upper_bound_evidence_type(item, candidate, observed, goal)
            source_type = _clean_text(item.get("evidence_type") or evidence_type)
            state_match_score = self._float01(item.get("state_match_score"), 1.0 if bool(item.get("matched")) else 0.0)
            slot_or_target_score = self._upper_bound_slot_or_target_score(item, evidence_type, observed, goal)
            rollback_verified_score = 1.0 if bool(item.get("rollback_verified")) else 0.0
            confidence = 0.4 * state_match_score + 0.3 * slot_or_target_score + 0.3 * rollback_verified_score
            threshold = self._upper_bound_threshold(
                evidence_type,
                summary_all_debug=bool(self.light_explore_summary_all_debug),
            )
            rendered = self._upper_bound_render_evidence(item, evidence_type, candidate, observed, goal)
            required_slot_or_target_score = 1.0
            if evidence_type == "ACTION_HINT":
                required_slot_or_target_score = 0.60
            elif evidence_type in {"SCHEMA_HINT", "RISK_HINT"}:
                required_slot_or_target_score = 0.60
            facts = [
                x
                for x in self._extract_answer_facts_from_labels(observed, goal, limit=5)
                if not _clean_text(x).lower().startswith("avoid ")
            ]
            schema_fields = [
                x
                for x in observed
                if re.search(
                    r"\b(name|title|date|time|phone|email|amount|description|note|folder|file|field|search|filter|query)\b",
                    x.lower(),
                )
            ]
            operator = _clean_text(item.get("operator") or candidate.get("operator"))
            action_safe = bool(candidate) and not self._is_transaction_unsafe_candidate(candidate)
            if action_safe and current_state is not None:
                action_safe = (
                    self._candidate_visible_in_state(candidate, current_state)
                    or operator in {"SearchPeek", "FilterPeek"}
                )
            stop_reason_low = _clean_text(item.get("stop_reason") or item.get("boundary_type") or "").lower()
            source_goal = _clean_text(item.get("source_goal") or item.get("goal") or item.get("source") or "")
            source_activity = _clean_text(item.get("after_activity") or item.get("source_activity") or "")
            same_task_or_app = bool(_same_task_or_app(source_goal, source_activity))
            hard_negative = bool(
                item.get("hard_negative")
                or (
                    isinstance(item.get("slot_evidence"), dict)
                    and bool((item.get("slot_evidence") or {}).get("hard_negative"))
                )
                or _clean_text(item.get("boundary_type")).upper() == "NEGATIVE_BOUNDARY"
                or any(
                    _clean_text(x).lower().startswith("avoid ")
                    for x in observed_raw
                )
            )
            globally_valid_avoid = bool(
                same_task_or_app
                and (
                    hard_negative
                    or any(
                        token in stop_reason_low
                        for token in (
                            "hard_negative",
                            "wrong_target",
                            "wrong_app",
                            "not_found",
                            "dead_end",
                            "dead-end",
                            "negative_boundary",
                            "slot_complete_hard_negative",
                            "root_visible_hard_negative",
                        )
                    )
                    or source_type == "AVOID_HINT"
                )
            )
            globally_valid_risk = bool(
                same_task_or_app
                and (
                    operator == "RiskBoundary"
                    or any(token in stop_reason_low for token in ("risk", "unsafe", "destructive", "delete", "commit"))
                    or source_type == "RISK_HINT"
                )
            )
            schema_visible = bool(
                schema_fields
                or operator in {"SearchPeek", "FilterPeek", "FormSchema"}
            )
            answer_complete = bool(
                item.get("answer_complete")
                or item.get("slot_complete")
                or (isinstance(item.get("slot_evidence"), dict) and item.get("slot_evidence", {}).get("slot_complete"))
            )
            reason = ""
            if evidence_type == "NONE":
                reason = "no_supported_evidence_type"
            elif not rendered:
                reason = "empty_rendered_text"
            elif not bool(item.get("rollback_verified")):
                reason = "rollback_not_verified"
            elif evidence_type == "ACTION_HINT" and not action_safe:
                reason = "unsafe_action"
            elif slot_or_target_score < required_slot_or_target_score:
                reason = (
                    "safe_action_boundary_not_supported"
                    if evidence_type == "ACTION_HINT"
                    else "no_exact_target_or_entity_match"
                )
            elif evidence_type == "ANSWER_HINT":
                if not answer_complete:
                    reason = "answer_fact_incomplete"
                elif not facts:
                    reason = "answer_hint_no_fact_found"
            elif evidence_type == "SCHEMA_HINT":
                if not schema_visible:
                    reason = "schema_not_visible"
            elif evidence_type == "RISK_HINT":
                if not globally_valid_risk:
                    reason = "risk_not_globally_valid"
            elif evidence_type == "AVOID_HINT":
                if not globally_valid_avoid:
                    reason = "avoid_not_globally_valid"
            elif self._upper_bound_observed_noise_only(observed):
                reason = "system_or_status_bar_only"
            elif confidence < threshold:
                reason = "low_confidence"
            elif evidence_type in {"ACTION_HINT", "SCHEMA_HINT", "ANSWER_HINT"} and state_match_score < 0.50:
                reason = "state_not_aligned"
            if reason:
                rejected.append(
                    {
                        "source_step": item.get("source_step"),
                        "branch_id": item.get("branch_id"),
                        "next_label": _clean_text(candidate.get("label") or item.get("next_label")),
                        "operator": _clean_text(item.get("operator") or candidate.get("operator")),
                        "reason": reason,
                        "evidence_type": evidence_type,
                        "confidence": confidence,
                        "threshold": threshold,
                        "state_match_score": float(state_match_score),
                        "rollback_verified": bool(item.get("rollback_verified")),
                        "action_safe": bool(action_safe),
                        "same_task_or_app": bool(same_task_or_app),
                        "rendered_text": rendered,
                    }
                )
                continue
            trace = dict(item)
            trace.update(
                {
                    "hint_type": evidence_type,
                    "hint_kind": "actionable" if evidence_type == "ACTION_HINT" else "evidence_only",
                    "prompt_line": rendered,
                    "rendered_prompt_text": rendered,
                    "source_type": source_type,
                    "confidence": float(confidence),
                    "threshold": float(threshold),
                    "state_match_score": float(state_match_score),
                    "slot_or_target_score": float(slot_or_target_score),
                    "rollback_verified": bool(item.get("rollback_verified")),
                    "action_safe": bool(action_safe),
                    "same_task_or_app": bool(same_task_or_app),
                }
            )
            buckets[evidence_type].append(trace)

        limits = {
            "ANSWER_HINT": 3,
            "ACTION_HINT": 3,
            "AVOID_HINT": 3,
            "SCHEMA_HINT": 2,
            "RISK_HINT": 2,
        }
        selected: list[dict[str, Any]] = []
        lines = [
            "[Exploration Evidence from Previous Step]",
            "",
        ]
        aligned_scores = [float(item.get("state_match_score") or 0.0) for item in matches if item.get("matched")]
        lines.extend(
            [
                "State match:",
                f"- current screen matches explored state by activity + anchor overlap = {max(aligned_scores) if aligned_scores else 0.0:.2f}",
                "",
            ]
        )
        section_names = [
            ("ANSWER_HINT", "Useful facts"),
            ("ACTION_HINT", "Suggested safe actions"),
            ("AVOID_HINT", "Avoid"),
            ("SCHEMA_HINT", "Schema/Risk"),
            ("RISK_HINT", "Schema/Risk"),
        ]
        emitted_schema_risk_header = False
        rank_by_type: dict[str, int] = {}
        for evidence_type, section in section_names:
            entries = buckets.get(evidence_type, [])[: limits[evidence_type]]
            if not entries:
                continue
            if section == "Schema/Risk":
                if not emitted_schema_risk_header:
                    lines.extend([section + ":", ""])
                    emitted_schema_risk_header = True
            else:
                lines.extend([section + ":", ""])
            for entry in entries:
                rank_by_type[evidence_type] = rank_by_type.get(evidence_type, 0) + 1
                entry["rank"] = len(selected) + 1
                entry["rank_within_type"] = rank_by_type[evidence_type]
                selected.append(entry)
                lines.append(f"{rank_by_type[evidence_type]}. {entry['prompt_line']}")
            lines.append("")
        if not selected:
            self._last_strict_not_injected_reasons = rejected
            return "", []
        for entry in selected:
            entry["not_injected_reasons_sample"] = rejected[:5]
        self._last_strict_not_injected_reasons = rejected
        return "\n".join(lines).strip(), selected

    def _build_rule_prompt_context_from_state_aligned_matches(
        self,
        matches: list[dict[str, Any]],
        goal: str = "",
        current_state: Any | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        buckets: dict[str, list[dict[str, Any]]] = {
            "ANSWER_HINT": [],
            "ACTION_HINT": [],
            "AVOID_HINT": [],
            "SCHEMA_HINT": [],
            "RISK_HINT": [],
        }
        rejected: list[dict[str, Any]] = []
        prior = self._reasoning_prior_for_next_step if isinstance(self._reasoning_prior_for_next_step, dict) else {}
        task_mode = self._task_mode(goal)
        goal_normalized = _clean_text(goal)
        current_activity = _clean_text(self._foreground_activity_name())
        current_package = current_activity.split("/", 1)[0] if "/" in current_activity else current_activity

        def _status_or_system_label(label: str) -> bool:
            low = _clean_text(label).lower()
            if not low:
                return True
            noise_markers = (
                "android system notification",
                "systemui",
                "system ui",
                "status bar",
                "notification icon",
                "switch input method",
                "more features",
                "battery ",
                "airplane mode",
                "no phone",
            )
            if any(marker in low for marker in noise_markers):
                return True
            if re.fullmatch(r"\d{1,2}:\d{2}", low):
                return True
            return False

        def _same_task_or_app(trace_goal: str, trace_activity: str) -> bool:
            trace_goal_clean = _clean_text(trace_goal)
            if trace_goal_clean and trace_goal_clean == goal_normalized:
                return True
            trace_activity_norm = _clean_text(trace_activity)
            trace_package = trace_activity_norm.split("/", 1)[0] if "/" in trace_activity_norm else trace_activity_norm
            return bool(current_package and trace_package and current_package == trace_package)

        for item in sorted(matches, key=lambda x: float(x.get("confidence") or x.get("score") or 0.0), reverse=True):
            candidate = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
            observed_raw = self._evidence_observed_elements(list(item.get("observed_elements") or []), goal=goal)
            observed = [x for x in observed_raw if not _status_or_system_label(x)]
            source_type = _clean_text(item.get("evidence_type") or "")
            operator = _clean_text(item.get("operator") or candidate.get("operator"))
            state_match_score = self._float01(item.get("state_match_score"), 1.0 if item.get("matched") else 0.0)
            source_confidence = self._float01(item.get("confidence") or item.get("score"), 0.0)
            rollback_verified = bool(item.get("rollback_verified"))
            action_safe = bool(candidate) and not self._is_transaction_unsafe_candidate(candidate)
            if current_state is not None and candidate:
                action_safe = bool(
                    action_safe
                    and (
                        self._candidate_visible_in_state(candidate, current_state)
                        or operator in {"SearchPeek", "FilterPeek"}
                    )
                )
            slot_evidence = item.get("slot_evidence") if isinstance(item.get("slot_evidence"), dict) else {}
            if not slot_evidence and self.light_explore_slot_complete:
                slot_evidence = slot_complete_evidence.extract_slot_evidence(
                    list(item.get("observed_elements") or []),
                    goal,
                    activity=_clean_text(item.get("after_activity") or ""),
                )
            facts = [
                x
                for x in self._extract_answer_facts_from_labels(observed, goal, limit=5)
                if not _clean_text(x).lower().startswith("avoid ")
            ]
            schema_fields = [
                x
                for x in observed
                if re.search(
                    r"\b(name|title|date|time|phone|email|amount|description|note|folder|file|field|search|filter|query)\b",
                    x.lower(),
                )
            ]
            stop_reason = _clean_text(item.get("stop_reason") or item.get("boundary_type") or item.get("reason") or "")
            stop_reason_low = stop_reason.lower()
            slot_hard_negative = _clean_text(slot_evidence.get("hard_negative") if isinstance(slot_evidence, dict) else "")
            observed_avoid_text = next(
                (
                    _clean_text(x)
                    for x in observed
                    if _clean_text(x).lower().startswith("avoid ")
                ),
                "",
            )
            hard_negative = bool(
                item.get("hard_negative")
                or item.get("is_hard_negative")
                or item.get("negative_evidence")
                or slot_hard_negative
                or observed_avoid_text
            )
            same_task_or_app = bool(
                _same_task_or_app(
                    _clean_text(item.get("source_goal") or item.get("goal") or item.get("source") or ""),
                    _clean_text(item.get("after_activity") or item.get("source_activity") or ""),
                )
            )
            globally_valid_avoid = bool(
                same_task_or_app
                and (
                    hard_negative
                    or any(
                        token in stop_reason_low
                        for token in (
                            "hard_negative",
                            "wrong_target",
                            "wrong_app",
                            "not_found",
                            "dead_end",
                            "dead-end",
                            "negative_boundary",
                            "slot_complete_hard_negative",
                            "root_visible_hard_negative",
                        )
                    )
                    or _clean_text(item.get("boundary_type")).upper() == "NEGATIVE_BOUNDARY"
                )
            )
            globally_valid_risk = bool(
                same_task_or_app
                and (
                    operator == "RiskBoundary"
                    or self._is_transaction_unsafe_candidate(candidate)
                    or any(token in stop_reason_low for token in ("risk", "unsafe", "destructive", "delete", "commit"))
                )
            )
            schema_visible = bool(
                schema_fields
                or operator in {"SearchPeek", "FilterPeek", "FormSchema"}
            )
            system_or_status_only = bool(observed_raw and not observed)
            answer_complete = bool(
                item.get("answer_complete")
                or item.get("slot_complete")
                or (isinstance(slot_evidence, dict) and slot_evidence.get("slot_complete"))
            )

            hint_type = "NONE"
            reason = ""
            if not rollback_verified:
                reason = "rollback_failed"
            elif source_type == "ANSWER_HINT" and state_match_score < 0.50:
                reason = "state_mismatch"
            elif source_type == "ACTION_HINT" and state_match_score < 0.35:
                reason = "state_mismatch"
            elif source_type == "ANSWER_HINT" and answer_complete:
                hint_type = "ANSWER_HINT"
            elif source_type == "ACTION_HINT" and state_match_score >= 0.50 and action_safe:
                hint_type = "ACTION_HINT"
            elif source_type in {"AVOID_HINT", "RISK_HINT"}:
                if source_type == "RISK_HINT":
                    if not globally_valid_risk:
                        reason = "risk_not_globally_valid"
                    elif source_confidence < 0.60:
                        reason = "low_confidence_risk"
                    else:
                        hint_type = "RISK_HINT"
                else:
                    if system_or_status_only:
                        reason = "system_or_status_bar_only"
                    elif not globally_valid_avoid:
                        reason = "avoid_not_globally_valid"
                    elif source_confidence < (0.50 if hard_negative else 0.65):
                        reason = "low_confidence_avoid"
                    else:
                        hint_type = "AVOID_HINT"
            elif operator == "RiskBoundary":
                if globally_valid_risk:
                    hint_type = "RISK_HINT"
                else:
                    reason = "risk_not_globally_valid"
            elif source_type == "SCHEMA_HINT":
                if rollback_verified and schema_visible and same_task_or_app:
                    hint_type = "SCHEMA_HINT"
                else:
                    reason = "schema_not_visible_or_wrong_app"
            elif source_type == "ANSWER_HINT":
                reason = "answer_fact_incomplete"
            elif source_type == "ACTION_HINT":
                reason = "unsafe_or_unlocatable_action"
            else:
                reason = "unsupported_evidence_type"
            if hint_type == "NONE":
                rejected.append(
                    {
                        "source_step": item.get("source_step"),
                        "branch_id": item.get("branch_id"),
                        "next_label": _clean_text(candidate.get("label") or item.get("next_label")),
                        "operator": operator,
                        "reason": reason or "strict_conditions_not_met",
                        "evidence_type": source_type or "NONE",
                        "state_match_score": state_match_score,
                        "rollback_verified": rollback_verified,
                        "action_safe": action_safe,
                        "same_task_or_app": same_task_or_app,
                        "rendered_text": "",
                    }
                )
                continue
            enriched = dict(item)
            enriched.update(
                {
                    "hint_type": hint_type,
                    "hint_kind": "actionable" if hint_type == "ACTION_HINT" else "evidence_only",
                    "operator": operator,
                    "fact": "; ".join(facts[:5] or observed[:5]),
                    "suggested_action": _clean_text(
                        item.get("suggested_action")
                        or self._upper_bound_render_evidence(item, hint_type, candidate, observed, goal)
                        or _clean_text(candidate.get("label") or item.get("next_label") or "")
                    ),
                    "candidate_label": _clean_text(candidate.get("label") or item.get("next_label")),
                    "observed_elements": observed,
                    "avoid_reason": slot_hard_negative
                    or observed_avoid_text
                    or "; ".join(observed[:3])
                    or _clean_text(item.get("stop_reason") or item.get("boundary_type") or "risk"),
                    "target": self._avoid_target_from_reason(slot_hard_negative or observed_avoid_text)
                    or _clean_text(candidate.get("label") or item.get("next_label")),
                    "confidence": float(max(float(item.get("confidence") or 0.0), state_match_score)),
                    "threshold": 0.50,
                    "state_match_score": state_match_score,
                    "rollback_verified": rollback_verified,
                    "action_safe": action_safe,
                    "slot_complete": bool(item.get("slot_complete") or (isinstance(slot_evidence, dict) and slot_evidence.get("slot_complete"))),
                    "answer_complete": bool(answer_complete),
                    "same_task_or_app": same_task_or_app,
                }
            )
            rendered = self._build_reasoning_template_render(hint_type, enriched, prior)
            if not rendered:
                rejected.append(
                    {
                        "source_step": item.get("source_step"),
                        "branch_id": item.get("branch_id"),
                        "next_label": enriched["candidate_label"],
                        "operator": operator,
                        "reason": "empty_rendered_text",
                        "evidence_type": hint_type,
                        "state_match_score": state_match_score,
                        "rollback_verified": rollback_verified,
                        "action_safe": action_safe,
                        "same_task_or_app": same_task_or_app,
                        "rendered_text": "",
                    }
                )
                continue
            enriched["prompt_line"] = rendered
            enriched["rendered_prompt_text"] = rendered
            buckets[hint_type].append(enriched)

        limits = {
            "ANSWER_HINT": 3,
            "ACTION_HINT": 2,
            "AVOID_HINT": 2,
            "SCHEMA_HINT": 2,
            "RISK_HINT": 2,
        }
        selected: list[dict[str, Any]] = []
        lines = ["[Exploration Evidence from Previous Step]"]
        for hint_type, title in (
            ("ANSWER_HINT", "Useful facts"),
            ("ACTION_HINT", "Suggested safe actions"),
            ("SCHEMA_HINT", "Schema"),
            ("AVOID_HINT", "Avoid"),
            ("RISK_HINT", "Risk"),
        ):
            entries = buckets.get(hint_type, [])[: limits[hint_type]]
            if not entries:
                continue
            lines.append("")
            lines.append(f"{title}:")
            for entry in entries:
                entry["rank"] = len(selected) + 1
                selected.append(entry)
                lines.append(f"{len(selected)}. {entry['prompt_line']}")
        self._last_strict_not_injected_reasons = rejected
        if not selected:
            return "", []
        for entry in selected:
            entry["not_injected_reasons_sample"] = rejected[:5]
        return "\n".join(lines).strip(), selected

    def _build_strict_prompt_context_from_state_aligned_matches(
        self,
        matches: list[dict[str, Any]],
        goal: str = "",
        current_state: Any | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        if getattr(self, "light_explore_upper_bound_evidence", False):
            return self._build_upper_bound_prompt_context_from_state_aligned_matches(
                matches=matches,
                goal=goal,
                current_state=current_state,
            )
        return self._build_rule_prompt_context_from_state_aligned_matches(
            matches=matches,
            goal=goal,
            current_state=current_state,
        )
        task_mode = self._task_mode(goal)
        matches.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                int(item.get("depth_reached") or 0),
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        lines: list[str] = []
        not_injected: list[dict[str, Any]] = []
        for item in matches:
            next_candidate = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
            observed = self._evidence_observed_elements(list(item.get("observed_elements") or []), goal=goal)
            operator, operator_reason = self._candidate_operator(next_candidate, goal)
            if operator == "Unknown" and item.get("operator"):
                operator = _clean_text(item.get("operator"))
                operator_reason = _clean_text(item.get("operator_reason") or "matched_observation_operator")
            label = _clean_text(next_candidate.get("label") or item.get("next_label") or "target")
            center = next_candidate.get("center")
            center_text = ""
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                center_text = f"[{int(center[0])}, {int(center[1])}]"
            visible = bool(current_state is not None and self._candidate_visible_in_state(next_candidate, current_state))
            source_evidence_type = _clean_text(item.get("evidence_type"))
            reason = ""
            hint_type = ""
            line = ""
            slot_evidence: dict[str, Any] = {}
            if self.light_explore_slot_complete:
                slot_evidence = slot_complete_evidence.extract_slot_evidence(
                    observed,
                    goal,
                    activity=_clean_text(item.get("after_activity") or ""),
                )
            allow_action_hint = not self.light_explore_no_action_hint
            if self.light_explore_slot_complete:
                allow_action_hint = bool(
                    allow_action_hint
                    and operator in {"SearchPeek", "NavigationPeek"}
                    and source_evidence_type == "ACTION_HINT"
                    and float(item.get("evidence_gain") or 0.0) >= 3.0
                )
            if (
                self._is_actionable_match(item, goal=goal)
                and visible
                and not self._is_transaction_unsafe_candidate(next_candidate)
                and allow_action_hint
            ):
                effect = "; ".join(self._informative_observed_elements(list(item.get("observed_elements") or []))[:4])
                hint_type = "ACTION_HINT"
                line = (
                    "[Exploration Evidence]\n"
                    f"Recommended next action: CLICK \"{label}\" near {center_text or 'the visible target'}.\n"
                    f"Observed effect: {effect or 'the next screen changed after this action'}.\n"
                    "Use only if the target is visible in the current screenshot."
                )
            elif task_mode == "INFO_QUERY":
                if self.light_explore_slot_complete:
                    facts = list(slot_evidence.get("facts") or [])[:5]
                    if facts and slot_evidence.get("hint_type") == "AVOID_HINT":
                        reason = "slot_complete_hard_negative"
                    elif facts and slot_evidence.get("hint_type") == "ANSWER_HINT":
                        hint_type = "ANSWER_HINT"
                        fact_lines = "\n".join(f"- {fact}" for fact in facts[:5])
                        line = (
                            "[Exploration Evidence]\n"
                            "Observed answer-related facts:\n"
                            f"{fact_lines}\n"
                            "Use these facts only if consistent with the current screenshot."
                        )
                    else:
                        reason = "slot_complete_missing:" + ",".join(slot_evidence.get("missing_slots") or [])
                        # Slot-complete mode is intentionally strict: partial facts are
                        # logged for diagnosis but never promoted to prompt hints.
                        facts = []
                else:
                    facts = (
                        self._extract_answer_facts_from_labels(observed, goal, limit=5)
                        if self.light_explore_answer_extractors
                        else [x for x in observed if self._is_query_fact_label(x, goal)]
                    )
                if line:
                    pass
                elif facts and any(_clean_text(x).lower().startswith("avoid ") for x in facts):
                    reason = "answer_hint_domain_hard_negative"
                elif facts:
                    hint_type = "ANSWER_HINT"
                    fact_lines = "\n".join(f"- {fact}" for fact in facts[:5])
                    line = (
                        "[Exploration Evidence]\n"
                        "Observed answer-related facts:\n"
                        f"{fact_lines}\n"
                        "Use these facts only if consistent with the current screenshot."
                    )
                else:
                    reason = "answer_hint_no_target_fact"
            elif task_mode == "FORM_CREATE_EDIT":
                fields = [
                    x
                    for x in observed
                    if re.search(r"\b(name|title|date|time|phone|email|amount|description|note|folder|file|field)\b", x.lower())
                ]
                submit = next(
                    (
                        x
                        for x in observed
                        if re.search(r"\b(add|create|save|done|ok|submit|next)\b", x.lower())
                    ),
                    "",
                )
                if fields:
                    hint_type = "SCHEMA_HINT"
                    line = (
                        "[Exploration Evidence]\n"
                        f"This page has fields: {', '.join(fields[:6])}.\n"
                        f"Use field \"{fields[0]}\" for the expected value type.\n"
                        f"Submit using \"{submit}\" if visible."
                    )
                else:
                    reason = "schema_hint_no_fields"
            if not line and (operator == "RiskBoundary" or source_evidence_type == "RISK_HINT"):
                rollback_verified = bool(item.get("rollback_verified", True))
                near_commit = task_mode in {"DELETE_COMMIT", "SIMPLE_VERIFY", "FORM_CREATE_EDIT"}
                if rollback_verified and near_commit:
                    hint_type = "RISK_HINT"
                    line = (
                        "[Exploration Evidence]\n"
                        f"\"{label}\" is a risky/commit action. Execute only if it is the final intended step."
                    )
                else:
                    reason = "risk_hint_not_near_commit_or_unverified"
            if not line and (operator in {"SearchPeek", "NavigationPeek", "RiskBoundary"} or source_evidence_type == "AVOID_HINT"):
                if self.light_explore_slot_complete and slot_evidence.get("hint_type") != "AVOID_HINT":
                    reason = "slot_complete_avoid_requires_hard_negative"
                    not_injected.append(
                        {
                            "source_step": item.get("source_step"),
                            "branch_id": item.get("branch_id"),
                            "path": item.get("path"),
                            "next_label": label,
                            "operator": operator,
                            "reason": reason,
                            "visible": visible,
                            "task_mode": task_mode,
                        }
                    )
                    continue
                irrelevant = any(
                    token in " ".join(observed).lower()
                    for token in ("assistant", "youtube", "update your app", "launcher", "google account", "resolver")
                )
                semantic_close = self._lexical_overlap_score(
                    merged=label,
                    goal_tokens=self._goal_tokens(goal),
                    app_keywords=self._goal_app_keywords(goal),
                ) > 0.05 or operator == "RiskBoundary"
                if (irrelevant and semantic_close) or source_evidence_type == "AVOID_HINT":
                    hint_type = "AVOID_HINT"
                    line = (
                        "[Exploration Evidence]\n"
                        "Avoid:\n"
                        f"- \"{label}\" led to {', '.join(observed[:4]) or 'an irrelevant screen'}, "
                        f"not useful for {_clean_text(goal)[:80]}."
                    )
            if line:
                trace = dict(item)
                trace["rank"] = len(selected) + 1
                trace["hint_type"] = hint_type
                trace["hint_kind"] = hint_type
                trace["operator"] = operator
                trace["operator_reason"] = operator_reason
                trace["prompt_line"] = line
                if slot_evidence:
                    trace["slot_evidence"] = slot_evidence
                    trace["slot_complete"] = bool(slot_evidence.get("slot_complete"))
                    trace["slot_coverage"] = slot_evidence.get("slot_coverage")
                    trace["missing_slots"] = slot_evidence.get("missing_slots")
                selected.append(trace)
                lines.append(line)
                if len(selected) >= int(self.light_explore_prompt_result_limit):
                    break
            else:
                not_injected.append(
                    {
                        "source_step": item.get("source_step"),
                        "branch_id": item.get("branch_id"),
                        "path": item.get("path"),
                        "next_label": label,
                        "operator": operator,
                        "reason": reason or "strict_conditions_not_met",
                        "visible": visible,
                        "task_mode": task_mode,
                    }
                )
        if not selected:
            self._last_strict_not_injected_reasons = not_injected
            return "", []
        context = "\n\n".join(lines)
        for trace in selected:
            trace["not_injected_reasons_sample"] = not_injected[:3]
        self._last_strict_not_injected_reasons = not_injected
        return context, selected

    def _shortcut_plan_id(self, goal: str, step_idx: int, branch_id: Any = "") -> str:
        raw = f"{_clean_text(goal)[:120]}|{step_idx + 1}|{branch_id}|{time.time():.6f}"
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _action_summary_from_json_action(
        self,
        action: json_action.JSONAction | None,
        candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate = candidate if isinstance(candidate, dict) else {}
        a11y = candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {}
        center = candidate.get("center")
        bbox = a11y.get("bbox") if isinstance(a11y.get("bbox"), dict) else None
        label = _clean_text(candidate.get("label") or candidate.get("merged"))
        action_type = _clean_text(getattr(action, "action_type", "") if action is not None else candidate.get("action_kind") or "click")
        if action_type == "type":
            action_type = json_action.INPUT_TEXT
        return {
            "action_type": action_type,
            "label": label,
            "bbox": bbox,
            "center": [int(center[0]), int(center[1])] if isinstance(center, (list, tuple)) and len(center) >= 2 else (
                [int(getattr(action, "x")), int(getattr(action, "y"))]
                if action is not None and getattr(action, "x", None) is not None and getattr(action, "y", None) is not None
                else None
            ),
            "resource_id": _clean_text(a11y.get("resource_id")),
            "class_name": _clean_text(a11y.get("class_name") or a11y.get("class")),
            "value": _clean_text(getattr(action, "text", "") if action is not None else candidate.get("text")),
            "operator": _clean_text(candidate.get("operator")),
            "risk_level": "high" if self._is_transaction_unsafe_candidate(candidate) else "low",
        }

    def _shortcut_state_from_step(self, step: dict[str, Any]) -> dict[str, Any]:
        activity = _clean_text(step.get("after_activity"))
        labels = self._informative_observed_elements(list(step.get("observed_elements") or []))[:16]
        visible_targets: list[dict[str, Any]] = []
        for item in list(step.get("a11y") or [])[:20]:
            if not isinstance(item, dict):
                continue
            label = _clean_text(item.get("label") or item.get("text") or item.get("content_description") or item.get("resource_id"))
            if label:
                visible_targets.append(
                    {
                        "label": label,
                        "center": item.get("center"),
                        "resource_id": item.get("resource_id"),
                        "class_name": item.get("class_name") or item.get("class"),
                    }
                )
        signature_text = "|".join(x.lower() for x in labels)
        return {
            "package": activity.split("/", 1)[0] if "/" in activity else "",
            "activity": activity,
            "screen_role": self._infer_screen_role_from_labels(labels),
            "p_hash": step.get("after_hash"),
            "anchor_labels": labels,
            "tree_signature": hashlib.sha1(signature_text.encode("utf-8", errors="ignore")).hexdigest()[:16],
            "visible_targets": visible_targets,
        }

    def _shortcut_state_from_current(self, state: Any, activity: str = "") -> dict[str, Any]:
        labels = self._informative_observed_elements(self._state_semantic_summary(state, limit=40))[:16]
        signature_text = "|".join(x.lower() for x in labels)
        activity = _clean_text(activity or self._foreground_activity_name())
        return {
            "package": activity.split("/", 1)[0] if "/" in activity else "",
            "activity": activity,
            "screen_role": self._infer_screen_role_from_labels(labels),
            "p_hash": self._state_hash(state),
            "anchor_labels": labels,
            "tree_signature": hashlib.sha1(signature_text.encode("utf-8", errors="ignore")).hexdigest()[:16],
            "visible_targets": [],
        }

    def _slot_coverage_for_labels(self, labels: list[str], goal: str, activity: str = "") -> dict[str, Any]:
        if not self.light_explore_slot_complete:
            return {"slot_coverage": 0.0, "missing_slots": [], "filled_slots": [], "slot_complete": False}
        evidence = slot_complete_evidence.extract_slot_evidence(labels, goal, activity=activity)
        return evidence if isinstance(evidence, dict) else {}

    def _t2_score_components(
        self,
        *,
        goal: str,
        candidate: dict[str, Any],
        t1_step: dict[str, Any],
        t2_step: dict[str, Any],
        before_evidence: dict[str, Any],
        after_evidence: dict[str, Any],
    ) -> dict[str, float]:
        before_missing = set(before_evidence.get("missing_slots") or [])
        after_missing = set(after_evidence.get("missing_slots") or [])
        newly_filled = before_missing - after_missing
        coverage_before = float(before_evidence.get("slot_coverage") or 0.0)
        coverage_after = float(after_evidence.get("slot_coverage") or 0.0)
        missing_slot_gain = float(len(newly_filled))
        if coverage_after > coverage_before and missing_slot_gain <= 0.0:
            missing_slot_gain = 1.0
        merged_text = " ".join(
            [
                _clean_text(candidate.get("label")),
                _clean_text(candidate.get("merged")),
                " ".join(_clean_text(x) for x in list(t2_step.get("observed_elements") or [])[:24]),
            ]
        ).lower()
        target_match = 1.0 if any(entity and entity in merged_text for entity in self._task_entities(goal)) else 0.0
        if self._extract_goal_search_text(goal).lower() and self._extract_goal_search_text(goal).lower() in merged_text:
            target_match = 1.0
        screen_role_progress = 1.0 if bool(t2_step.get("changed")) or coverage_after > coverage_before else 0.0
        operator, _ = self._candidate_operator(candidate, goal)
        task_mode = self._task_mode(goal)
        operator_priority = 0.0
        if task_mode == "INFO_QUERY" and operator in {"ListInspect", "NavigationPeek", "SearchPeek", "FilterPeek"}:
            operator_priority = 1.0
        elif task_mode == "NAVIGATION_SEARCH" and operator in {"NavigationPeek", "SearchPeek", "FilterPeek"}:
            operator_priority = 1.0
        elif task_mode == "FORM_CREATE_EDIT" and operator == "FormSchema":
            operator_priority = 0.5
        novelty = 1.0 if bool(t2_step.get("changed")) else 0.0
        label = _clean_text(candidate.get("label") or candidate.get("merged")).lower()
        ambiguity = 1.0 if (not label or label in {"content", "button", "item", "text", "view", "candidate"}) else 0.0
        risk = 1.0 if self._is_transaction_unsafe_candidate(candidate) else 0.0
        score = (
            3.0 * missing_slot_gain
            + 2.0 * target_match
            + 1.5 * screen_role_progress
            + 1.0 * operator_priority
            + 0.5 * novelty
            - 3.0 * ambiguity
            - 5.0 * risk
        )
        return {
            "score": float(score),
            "missing_slot_gain": float(missing_slot_gain),
            "target_entity_match": float(target_match),
            "screen_role_progress": float(screen_role_progress),
            "operator_priority": float(operator_priority),
            "novelty": float(novelty),
            "ambiguity_penalty": float(ambiguity),
            "risk_penalty": float(risk),
        }

    def _shortcut_task_mode_allowed(self, goal: str, candidate: dict[str, Any]) -> tuple[bool, str]:
        task_mode = self._task_mode(goal)
        if task_mode in {"DELETE_COMMIT", "FORM_CREATE_EDIT"}:
            return False, f"task_mode_disabled:{task_mode}"
        if task_mode == "SIMPLE_VERIFY":
            return False, "task_mode_disabled:SIMPLE_VERIFY_OPEN"
        if task_mode in {"INFO_QUERY", "NAVIGATION_SEARCH", "SIMPLE_VERIFY"}:
            return True, "task_mode_allowed"
        operator, _ = self._candidate_operator(candidate, goal)
        if operator in {"NavigationPeek", "SearchPeek", "FilterPeek", "ListInspect"}:
            return True, "operator_allowed"
        return False, f"task_mode_disabled:{task_mode}"

    def _shortcut_candidate_safe(
        self,
        candidate: dict[str, Any] | None,
        goal: str,
        state: Any | None = None,
    ) -> tuple[bool, str]:
        if not isinstance(candidate, dict):
            return False, "missing_candidate"
        action_kind = _clean_text(candidate.get("action_kind") or json_action.CLICK).lower()
        merged = f"{_clean_text(candidate.get('label'))} {_clean_text(candidate.get('merged'))}".lower()
        forbidden = (
            "delete",
            "confirm",
            "save",
            "send",
            "permission allow",
            "allow",
            "toggle",
            "long press",
            "long_press",
            "share",
            "resolver",
            "purchase",
            "done",
            "submit",
            "ok",
        )
        if any(token in merged for token in forbidden):
            return False, "forbidden_label_or_commit"
        if action_kind in {json_action.LONG_PRESS, "long_press"}:
            return False, "forbidden_long_press"
        if action_kind in {"type", json_action.INPUT_TEXT}:
            if self.t2_allow_safe_search_input and state is None:
                expected = _clean_text(self._extract_goal_search_text(goal)).lower()
                actual = _clean_text(candidate.get("text") or candidate.get("value")).lower()
                if expected and actual and expected == actual:
                    return True, "safe_search_input_deferred_state_check"
            if self.t2_allow_safe_search_input and self._is_safe_search_input_candidate(candidate, goal, state):
                return True, "safe_search_input"
            return False, "input_text_not_safe_search"
        if self._is_transaction_unsafe_candidate(candidate):
            return False, "transaction_unsafe"
        return True, "safe_click_or_navigation"

    def _json_action_from_shortcut_candidate(self, candidate: dict[str, Any]) -> json_action.JSONAction | None:
        action_kind = _clean_text(candidate.get("action_kind") or json_action.CLICK).lower()
        center = candidate.get("center")
        if action_kind in {"type", json_action.INPUT_TEXT}:
            return json_action.JSONAction(
                action_type=json_action.INPUT_TEXT,
                x=int(center[0]) if isinstance(center, (list, tuple)) and len(center) >= 2 and int(center[0]) > 0 else None,
                y=int(center[1]) if isinstance(center, (list, tuple)) and len(center) >= 2 and int(center[1]) > 0 else None,
                text=_clean_text(candidate.get("text")),
                clear_text=bool(candidate.get("clear_text")),
            )
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            return None
        return json_action.JSONAction(action_type=json_action.CLICK, x=int(center[0]), y=int(center[1]))

    def _actions_match_for_shortcut(
        self,
        planned: dict[str, Any] | None,
        actual: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        if not isinstance(planned, dict) or not isinstance(actual, dict):
            return False, "missing_action"
        p_type = _clean_text(planned.get("action_type")).lower()
        a_type = _clean_text(actual.get("action_type")).lower()
        if p_type != a_type:
            return False, f"type_mismatch:{p_type}!={a_type}"
        if p_type == json_action.INPUT_TEXT:
            p_text = _clean_text(planned.get("value") or planned.get("text")).lower()
            a_text = _clean_text(actual.get("text") or actual.get("value")).lower()
            if p_text and a_text and p_text == a_text:
                return True, "input_text_exact_match"
            p_tokens = set(re.findall(r"[a-z0-9]+", p_text))
            a_tokens = set(re.findall(r"[a-z0-9]+", a_text))
            overlap = (len(p_tokens & a_tokens) / max(1, len(p_tokens | a_tokens))) if (p_tokens or a_tokens) else 0.0
            return (overlap >= 0.8, f"input_text_token_overlap:{overlap:.2f}")
        if p_type == json_action.OPEN_APP:
            p_app = _clean_text(planned.get("value") or planned.get("app_name") or planned.get("label")).lower()
            a_app = _clean_text(actual.get("app_name") or actual.get("value") or actual.get("label")).lower()
            return (bool(p_app and (p_app in a_app or a_app in p_app)), "open_app_match")
        p_center = planned.get("center")
        if not p_center and isinstance(planned.get("bbox"), dict):
            bbox = planned["bbox"]
            try:
                p_center = [(float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0, (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0]
            except Exception:  # pylint: disable=broad-exception-caught
                p_center = None
        a_center = actual.get("center")
        if not a_center and actual.get("x") is not None and actual.get("y") is not None:
            a_center = [actual.get("x"), actual.get("y")]
        if isinstance(p_center, (list, tuple)) and isinstance(a_center, (list, tuple)) and len(p_center) >= 2 and len(a_center) >= 2:
            try:
                dist = math.dist([float(p_center[0]), float(p_center[1])], [float(a_center[0]), float(a_center[1])])
                if dist <= 120.0:
                    return True, f"center_distance:{dist:.1f}"
                bbox = planned.get("bbox") if isinstance(planned.get("bbox"), dict) else {}
                if bbox:
                    x, y = float(a_center[0]), float(a_center[1])
                    if float(bbox.get("x_min", -1)) <= x <= float(bbox.get("x_max", -1)) and float(bbox.get("y_min", -1)) <= y <= float(bbox.get("y_max", -1)):
                        return True, "row_container_match"
                if dist <= 220.0 and _clean_text(planned.get("label")):
                    return True, f"semantic_near_row:{dist:.1f}"
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        for key in ("label", "resource_id"):
            p_val = _clean_text(planned.get(key)).lower()
            a_val = _clean_text(actual.get(key)).lower()
            if p_val and a_val and (p_val in a_val or a_val in p_val):
                return True, f"{key}_match"
        p_label = _clean_text(planned.get("label") or planned.get("value")).lower()
        a_blob = " ".join(_clean_text(actual.get(k)).lower() for k in ("label", "value", "text", "summary", "resource_id"))
        if p_label and a_blob:
            p_tokens = set(re.findall(r"[a-z0-9]{3,}", p_label))
            a_tokens = set(re.findall(r"[a-z0-9]{3,}", a_blob))
            if p_tokens and a_tokens and len(p_tokens & a_tokens) / float(len(p_tokens | a_tokens)) >= 0.8:
                return True, "semantic_entity_match"
        return False, "mismatch"

    def _write_shortcut_jsonl(self, goal: str, filename: str, row: dict[str, Any]) -> None:
        task_dir = self._task_output_dir(goal)
        if not task_dir:
            return
        try:
            with open(os.path.join(task_dir, filename), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[SHORTCUT {_now_hms()}] write_failed {filename}: {exc}")
        self._write_state_acquisition_metrics(goal)

    def _write_state_acquisition_metrics(self, goal: str = "") -> None:
        task_dir = self._task_output_dir(goal) if goal else ""
        trace_root = os.path.dirname(task_dir) if task_dir else self.output_path
        rows = dict(self._state_acquisition_metrics)
        for key in (
            "full_a11y_calls",
            "screenshot_calls",
            "activity_calls",
            "root_state_reuse_count",
            "rollback_light_verify_count",
            "rollback_full_a11y_fallback_count",
            "passive_no_get_state_count",
            "branch_full_a11y_calls",
            "rooted_depth1_only_count",
            "rooted_depth2_count",
            "t2_candidate_count",
            "searchinput_t2_candidate_count",
            "depth2_blocked_by_hash_unchanged_count",
            "depth2_blocked_by_semantic_unchanged_count",
            "depth2_blocked_by_no_typed_candidate_count",
            "depth2_blocked_by_safety_count",
            "depth2_blocked_by_should_expand_count",
        ):
            rows.setdefault(key, 0.0)
        for base in [trace_root, self.output_path]:
            if not base:
                continue
            try:
                os.makedirs(base, exist_ok=True)
                with open(os.path.join(base, "state_acquisition_metrics.csv"), "w", encoding="utf-8") as f:
                    f.write("metric,value\n")
                    for key, value in sorted(rows.items()):
                        f.write(f"{key},{value}\n")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"[SHORTCUT {_now_hms()}] state_metrics_write_failed: {exc}")

    def _append_shortcut_plan(self, goal: str, plan: dict[str, Any]) -> None:
        self._shortcut_plans.append(dict(plan))
        self._all_shortcut_plans.append(dict(plan))
        self._write_shortcut_jsonl(goal, "shortcut_plans.jsonl", plan)

    def _append_shortcut_event(self, goal: str, event: dict[str, Any]) -> None:
        self._shortcut_events.append(dict(event))
        self._all_shortcut_events.append(dict(event))
        self._write_shortcut_jsonl(goal, "shortcut_events.jsonl", event)

    def _append_shortcut_shadow_eval(self, goal: str, event: dict[str, Any]) -> None:
        self._shortcut_shadow_evals.append(dict(event))
        self._all_shortcut_shadow_evals.append(dict(event))
        self._write_shortcut_jsonl(goal, "shortcut_shadow_eval.jsonl", event)

    def _append_generic_shortcut_shadow(self, goal: str, event: dict[str, Any]) -> None:
        self._write_shortcut_jsonl(goal, "generic_shortcut_shadow.jsonl", event)

    def _append_direct_answer_candidate(self, goal: str, event: dict[str, Any]) -> None:
        self._direct_answer_candidates.append(dict(event))
        self._all_direct_answer_candidates.append(dict(event))
        self._write_shortcut_jsonl(goal, "direct_answer_candidates.jsonl", event)

    def _build_shortcut_plan_from_trace(
        self,
        *,
        goal: str,
        step_idx: int,
        planned_action: json_action.JSONAction,
        explore_trace: dict[str, Any],
        root_state: Any,
        root_activity: str,
    ) -> dict[str, Any] | None:
        if self.t2_shortcut_mode == "off":
            return None
        base: dict[str, Any] = {
            "task_id": _clean_text(goal)[:200],
            "episode_id": _clean_text(goal)[:80],
            "step_t": int(step_idx + 1),
            "plan_id": self._shortcut_plan_id(goal, step_idx),
            "variant": self.light_explore_variant,
            "mode": self.t2_shortcut_mode,
            "planned_root_action": self._action_summary_from_json_action(planned_action),
            "validity": {
                "root_action_match_required": True,
                "t1_state_match_required": True,
                "t2_target_visible_required": True,
                "risky_action_forbidden": True,
                "planned_root_action_alignment_threshold": 0.25,
                "planned_root_action_alignment": 0.0,
                "planned_root_action_is_planned": False,
            },
        }
        if not isinstance(explore_trace, dict) or str(explore_trace.get("status")) != "completed":
            base["no_plan_reason"] = f"exploration_not_completed:{(explore_trace or {}).get('status') if isinstance(explore_trace, dict) else 'none'}"
            self._append_shortcut_plan(goal, base)
            return None
        observations = [obs for obs in list(explore_trace.get("observations") or []) if isinstance(obs, dict)]
        scored: list[
            tuple[float, float, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float], dict[str, Any], dict[str, Any]]
        ] = []
        root_labels = self._state_semantic_summary(root_state, limit=TRACE_A11Y_LIMIT)
        before_evidence = self._slot_coverage_for_labels(root_labels, goal, root_activity)
        candidate_pairs = self._extract_depth1_depth2_pair(
            observations=observations,
            allow_unplanned_root=bool(self.light_explore_parallel_lookahead),
            planned_action=planned_action,
            planned_match_threshold=0.25,
        )
        for _, obs, depth1, depth2, root_match in candidate_pairs:
            if not bool((obs.get("rollback") or {}).get("success")):
                continue
            t2_candidate = depth2.get("candidate") if isinstance(depth2.get("candidate"), dict) else {}
            if not t2_candidate:
                continue
            t2_a11y = t2_candidate.get("a11y") if isinstance(t2_candidate.get("a11y"), dict) else {}
            t2_class = _clean_text(t2_a11y.get("class_name")).lower()
            t2_kind = _clean_text(t2_candidate.get("action_kind") or json_action.CLICK).lower()
            if t2_kind in {json_action.CLICK, "click"} and "edittext" in t2_class:
                query = _clean_text(self._extract_goal_search_text(goal)).lower()
                label_low = _clean_text(t2_candidate.get("label") or t2_candidate.get("merged")).lower()
                if query and query in label_low:
                    continue
            after_evidence = self._slot_coverage_for_labels(
                list(depth2.get("observed_elements") or []),
                goal,
                _clean_text(depth2.get("after_activity")),
            )
            components = self._t2_score_components(
                goal=goal,
                candidate=t2_candidate,
                t1_step=depth1,
                t2_step=depth2,
                before_evidence=before_evidence,
                after_evidence=after_evidence,
            )
            scored.append((
                float(components["score"]),
                float(root_match),
                obs,
                depth1,
                depth2,
                components,
                before_evidence,
                after_evidence,
            ))
        if not scored:
            base["no_plan_reason"] = "no_rooted_depth2_observation"
            self._append_shortcut_plan(goal, base)
            return None
        scored.sort(key=lambda item: (float(item[0] or 0.0), float(item[1] or 0.0)), reverse=True)
        top1, top1_root_match, obs, depth1, depth2, components, before_evidence, after_evidence = scored[0]
        top2 = float(scored[1][0]) if len(scored) >= 2 else max(0.0, top1 - 2.5)
        t2_candidate = depth2.get("candidate") if isinstance(depth2.get("candidate"), dict) else {}
        t2_safe, t2_safe_reason = self._shortcut_candidate_safe(t2_candidate, goal, None)
        task_allowed, task_allowed_reason = self._shortcut_task_mode_allowed(goal, t2_candidate)
        t2_action_kind = _clean_text(t2_candidate.get("action_kind") or json_action.CLICK).lower()
        t2_label = _clean_text(t2_candidate.get("label") or t2_candidate.get("merged"))
        t2_is_search_input = bool(
            t2_action_kind in {"type", json_action.INPUT_TEXT}
            and (
                self._is_safe_search_input_candidate(t2_candidate, goal, None)
                or str(t2_safe_reason).startswith("safe_search_input")
            )
        )
        target_entity = _clean_text(self._extract_goal_search_text(goal)).lower()
        t2_a11y = t2_candidate.get("a11y") if isinstance(t2_candidate.get("a11y"), dict) else {}
        t2_class = _clean_text(t2_a11y.get("class_name")).lower()
        t2_is_exact_click = bool(
            t2_action_kind in {json_action.CLICK, "click"}
            and target_entity
            and target_entity in t2_label.lower()
            and "edittext" not in t2_class
        )
        shortcut_pattern = (
            "SearchInputShortcut"
            if t2_is_search_input
            else ("ExactResultClickShortcut" if t2_is_exact_click else "generic")
        )
        before_missing = set(before_evidence.get("missing_slots") or [])
        after_missing = set(after_evidence.get("missing_slots") or [])
        newly_filled = sorted(before_missing - after_missing)
        rollback_verified = bool((obs.get("rollback") or {}).get("success"))
        margin = float(top1 - top2)
        confidence = max(0.0, min(1.0, 0.55 + 0.08 * margin + 0.12 * float(components.get("missing_slot_gain") or 0.0)))
        plan = dict(base)
        plan.update(
            {
                "plan_id": self._shortcut_plan_id(goal, step_idx, obs.get("branch_id")),
                "planned_root_action": self._action_summary_from_json_action(
                    planned_action,
                    depth1.get("candidate") if isinstance(depth1.get("candidate"), dict) else {},
                ),
                "explored_t1_state": self._shortcut_state_from_step(depth1),
                "shortcut_pattern": shortcut_pattern,
                "shortcut_t2_action": {
                    **self._action_summary_from_json_action(None, t2_candidate),
                    "expected_missing_slot_gain": components.get("missing_slot_gain"),
                    "is_safe_search_input": t2_is_search_input,
                    "is_exact_result_click": t2_is_exact_click,
                    "shortcut_pattern": shortcut_pattern,
                    "candidate": t2_candidate,
                },
                "explored_t2_state": {
                    **self._shortcut_state_from_step(depth2),
                    "filled_slots": sorted(set(before_evidence.get("missing_slots") or []) - after_missing),
                    "missing_slots": sorted(after_missing),
                },
                "confidence": confidence,
                "no_plan_reason": "",
                "evidence": {
                    "evidence_type": obs.get("evidence_type") or (after_evidence.get("hint_type") or "NONE"),
                    "planned_root_action_alignment": float(root_match),
                    "slot_coverage_before": before_evidence.get("slot_coverage", 0.0),
                    "slot_coverage_after": after_evidence.get("slot_coverage", 0.0),
                    "newly_filled_slots": newly_filled,
                    "confidence": confidence,
                    "top1_score": top1,
                    "top2_score": top2,
                    "score_margin": margin,
                    "rollback_verified": rollback_verified,
                    "score_components": components,
                },
                "eligibility": {
                    "t2_safe": bool(t2_safe),
                    "t2_safe_reason": t2_safe_reason,
                    "task_allowed": bool(task_allowed),
                    "task_allowed_reason": task_allowed_reason,
                    "top1_ok": bool(top1 >= self.t2_min_top1_score),
                    "margin_ok": bool(margin >= self.t2_min_score_margin),
                    "gain_or_target_ok": bool(
                        components.get("missing_slot_gain", 0.0) > 0.0
                        or components.get("target_entity_match", 0.0) > 0.0
                        or components.get("screen_role_progress", 0.0) > 0.0
                    ),
                    "rollback_verified": rollback_verified,
                },
            }
        )
        plan["validity"].update(
            {
                "planned_root_action_alignment": float(top1_root_match),
                "planned_root_action_is_planned": bool(
                    ((depth1.get("candidate") if isinstance(depth1.get("candidate"), dict) else {}).get("is_planned_action"))
                ),
            }
        )
        if after_evidence.get("slot_complete") and after_evidence.get("hint_type") == "ANSWER_HINT":
            direct = {
                "plan_id": plan["plan_id"],
                "task_id": plan["task_id"],
                "step_t": int(step_idx + 1),
                "confidence": confidence,
                "direct_answer_format": after_evidence.get("facts") or [],
                "would_have_completed": False,
                "would_match_ground_truth_if_available": None,
            }
            self._append_direct_answer_candidate(goal, direct)
        if not all(
            bool(plan["eligibility"].get(key))
            for key in ("t2_safe", "task_allowed", "top1_ok", "margin_ok", "gain_or_target_ok", "rollback_verified")
        ):
            plan["no_fire_reason"] = ",".join(
                key
                for key, value in plan["eligibility"].items()
                if key.endswith("_reason") is False and not bool(value)
            )
        self._append_shortcut_plan(goal, plan)
        self._pending_shortcut_plans = [plan]
        return plan

    def _evaluate_pending_shortcut_pre_reasoning(
        self,
        *,
        goal: str,
        step_idx: int,
        state: Any,
        current_activity: str,
        current_hash: str | None = None,
    ) -> dict[str, Any]:
        alignment_start = time.time()
        pending = list(self._pending_shortcut_plans)
        self._pending_shortcut_plans = []
        event: dict[str, Any] = {
            "task_id": _clean_text(goal)[:200],
            "step": int(step_idx + 1),
            "mode": self.t2_shortcut_mode,
            "pending_plan_count": len(pending),
            "would_fire": False,
            "fired": False,
            "skipped_vlm_reasoning": False,
            "no_fire_reason": "no_pending_plan" if not pending else "",
        }
        if self.t2_shortcut_mode == "off" or not pending:
            if self.t2_shortcut_mode != "off":
                event["alignment_latency_ms"] = float(max(0.0, time.time() - alignment_start) * 1000.0)
                self._append_shortcut_event(goal, event)
                self._append_latency_profile_event(
                    goal,
                    {
                        "event": "shortcut_alignment_eval",
                        "step": int(step_idx + 1),
                        "pending_plan_count": len(pending),
                        "would_fire": False,
                        "latency_ms": event["alignment_latency_ms"],
                        "no_fire_reason": event.get("no_fire_reason"),
                    },
                )
            return event
        goal_key = _clean_text(goal)[:200]
        if self.t2_shortcut_mode == "active_safe":
            fired_goal_keys = getattr(self, "_active_t2_fired_goal_keys", set())
            if goal_key in fired_goal_keys:
                event["no_fire_reason"] = "active_shortcut_already_fired_for_task"
                event["active_searchinput_only"] = "SEARCHINPUT_ONLY" in _clean_text(self.light_explore_variant).upper()
                self._append_shortcut_event(goal, event)
                return event
        evaluations: list[dict[str, Any]] = []
        parallel_mode = bool(self.light_explore_parallel_lookahead)
        planned_root_alignment_threshold = 0.25
        previous_action = self._actions[-1].get("action_dict") if self._actions else {}
        for plan in pending:
            plan_eval_start = time.time()
            t2_candidate = ((plan.get("shortcut_t2_action") or {}).get("candidate") or {})
            shortcut_pattern = self._shortcut_plan_pattern(plan)
            active_searchinput_only = bool(
                self.t2_shortcut_mode == "active_safe"
                and "SEARCHINPUT_ONLY" in _clean_text(self.light_explore_variant).upper()
            )
            planned_match, planned_reason = self._actions_match_for_shortcut(
                plan.get("planned_root_action"),
                previous_action,
            )
            plan_root_alignment = float(
                ((plan.get("validity") or {}).get("planned_root_action_alignment")
                 if isinstance(plan.get("validity"), dict)
                 else 0.0)
                or 0.0
            )
            state_match, state_reason, state_score = self._shortcut_t1_state_matches(
                goal=goal,
                plan=plan,
                state=state,
                current_activity=current_activity,
            )
            target_visible = self._candidate_visible_in_state(t2_candidate, state)
            safe, safe_reason = self._shortcut_candidate_safe(t2_candidate, goal, state)
            task_allowed, task_allowed_reason = self._shortcut_task_mode_allowed(goal, t2_candidate)
            if shortcut_pattern != "SearchInputShortcut":
                self._append_generic_shortcut_shadow(
                    goal,
                    {
                        "task_id": _clean_text(goal)[:200],
                        "variant": self.light_explore_variant,
                        "step": int(step_idx + 1),
                        "plan_id": plan.get("plan_id"),
                        "shortcut_pattern": shortcut_pattern,
                        "generic_shortcut_shadow_candidate": True,
                        "active_forbidden": True,
                        "planned_root_action": plan.get("planned_root_action"),
                        "shortcut_t2_action": plan.get("shortcut_t2_action"),
                        "planned_action_match": bool(planned_match),
                        "planned_action_match_reason": planned_reason,
                        "t1_state_match": bool(state_match),
                        "t1_state_match_method": state_reason,
                        "t1_state_match_score": state_score,
                        "target_visible": bool(target_visible),
                        "safe_before_active_block": bool(safe),
                        "safe_reason_before_active_block": safe_reason,
                        "task_allowed_before_active_block": bool(task_allowed),
                        "task_allowed_reason_before_active_block": task_allowed_reason,
                    },
                )
                if active_searchinput_only:
                    safe = False
                    safe_reason = "active_searchinput_only_blocks_generic"
                    task_allowed = False
                    task_allowed_reason = "active_searchinput_only_blocks_generic"
            evidence = plan.get("evidence") if isinstance(plan.get("evidence"), dict) else {}
            margin = float(evidence.get("score_margin") or 0.0)
            margin_norm = max(0.0, min(1.0, margin / 5.0))
            rollback_verified = bool(evidence.get("rollback_verified"))
            planned_match_score = float(plan_root_alignment) if parallel_mode else float(planned_match)
            confidence = (
                0.30 * planned_match_score
                + 0.30 * float(state_match)
                + 0.20 * float(target_visible)
                + 0.10 * margin_norm
                + 0.10 * float(rollback_verified)
            )
            top1 = float(evidence.get("top1_score") or 0.0)
            no_fire_reasons: list[str] = []
            if not planned_match:
                if parallel_mode:
                    if plan_root_alignment < planned_root_alignment_threshold:
                        no_fire_reasons.append(
                            f"planned_root_alignment_too_low:{plan_root_alignment:.2f}:{planned_root_alignment_threshold:.2f}"
                        )
                else:
                    no_fire_reasons.append(f"planned_action_mismatch:{planned_reason}")
            if not state_match:
                no_fire_reasons.append(f"t1_state_mismatch:{state_reason}")
            if not target_visible:
                no_fire_reasons.append("t2_target_not_visible")
            if not safe:
                no_fire_reasons.append(f"t2_not_safe:{safe_reason}")
            if not task_allowed:
                no_fire_reasons.append(task_allowed_reason)
            if confidence < self.t2_confidence_threshold:
                no_fire_reasons.append(f"confidence_below_threshold:{confidence:.2f}")
            if top1 < self.t2_min_top1_score:
                no_fire_reasons.append(f"top1_below_threshold:{top1:.2f}")
            if margin < self.t2_min_score_margin:
                no_fire_reasons.append(f"margin_below_threshold:{margin:.2f}")
            evaluations.append(
                {
                    "plan": plan,
                    "plan_id": plan.get("plan_id"),
                    "planned_action_match": planned_match,
                    "planned_action_match_reason": planned_reason,
                    "state_match": state_match,
                    "state_match_reason": state_reason,
                    "t1_state_match_method": state_reason.split(":", 1)[0],
                    "t1_state_match_score": state_score,
                    "target_visible": target_visible,
                    "safe": safe,
                    "safe_reason": safe_reason,
                    "task_allowed": task_allowed,
                    "task_allowed_reason": task_allowed_reason,
                    "planned_match": planned_match,
                    "planned_match_reason": planned_reason,
                    "planned_match_score": planned_match_score,
                    "confidence": confidence,
                    "top1_score": top1,
                    "score_margin": margin,
                    "planned_root_alignment": plan_root_alignment,
                    "parallel_mode": parallel_mode,
                    "parallel_alignment_threshold": planned_root_alignment_threshold,
                    "no_fire_reasons": no_fire_reasons,
                    "alignment_latency_ms": float(max(0.0, time.time() - plan_eval_start) * 1000.0),
                }
            )
        eligible = [x for x in evaluations if not x["no_fire_reasons"]]
        if eligible:
            eligible.sort(key=lambda x: float(x.get("confidence") or 0.0), reverse=True)
            if len(eligible) >= 2 and float(eligible[0]["confidence"]) - float(eligible[1]["confidence"]) < 0.10:
                eligible = []
                evaluations[0]["no_fire_reasons"].append("multiple_eligible_low_margin")
        selected = eligible[0] if eligible else max(evaluations, key=lambda x: float(x.get("confidence") or 0.0))
        event.update(
            {
                "plan_id": selected.get("plan_id"),
                "planned_root_action": (selected.get("plan") or {}).get("planned_root_action"),
                "shortcut_t2_action": (selected.get("plan") or {}).get("shortcut_t2_action"),
                "confidence": selected.get("confidence"),
                "state_match": selected.get("state_match"),
                "state_match_reason": selected.get("state_match_reason"),
                "t1_state_match_method": selected.get("t1_state_match_method"),
                "t1_state_match_score": selected.get("t1_state_match_score"),
                "target_visible": selected.get("target_visible"),
                "score_margin": selected.get("score_margin"),
                "would_fire": bool(eligible),
                "no_fire_reason": ";".join(selected.get("no_fire_reasons") or []),
                "evaluations": [
                    {k: v for k, v in item.items() if k != "plan"}
                    for item in evaluations
                ][:4],
            }
        )
        event["alignment_latency_ms"] = float(max(0.0, time.time() - alignment_start) * 1000.0)
        self._append_latency_profile_event(
            goal,
            {
                "event": "shortcut_alignment_eval",
                "step": int(step_idx + 1),
                "pending_plan_count": len(pending),
                "eligible_count": len(eligible),
                "would_fire": bool(eligible),
                "selected_plan_id": selected.get("plan_id"),
                "latency_ms": event["alignment_latency_ms"],
                "no_fire_reason": event.get("no_fire_reason"),
                "state_match": event.get("state_match"),
                "target_visible": event.get("target_visible"),
                "confidence": event.get("confidence"),
            },
        )
        self._append_shortcut_event(goal, event)
        return event

    def _complete_shadow_shortcut_eval(
        self,
        *,
        goal: str,
        shortcut_event: dict[str, Any],
        vlm_action: json_action.JSONAction,
    ) -> dict[str, Any] | None:
        if self.t2_shortcut_mode != "shadow" or not shortcut_event or not shortcut_event.get("plan_id"):
            return None
        shortcut_action = shortcut_event.get("shortcut_t2_action") if isinstance(shortcut_event.get("shortcut_t2_action"), dict) else {}
        shortcut_candidate = shortcut_action.get("candidate") if isinstance(shortcut_action.get("candidate"), dict) else {}
        shortcut_summary = self._action_summary_from_json_action(None, shortcut_candidate)
        vlm_summary = self._action_summary_from_json_action(vlm_action)
        match, reason = self._actions_match_for_shortcut(shortcut_summary, {**vlm_action.__dict__, **vlm_summary})
        evaluation = {
            "task_id": _clean_text(goal)[:200],
            "plan_id": shortcut_event.get("plan_id"),
            "step": shortcut_event.get("step"),
            "would_fire": bool(shortcut_event.get("would_fire")),
            "action_match": bool(match) if shortcut_event.get("would_fire") else False,
            "action_match_reason": reason,
            "shortcut_t2_action": shortcut_summary,
            "vlm_action": vlm_summary,
            "potential_harm": bool(shortcut_event.get("would_fire") and not match and shortcut_summary.get("risk_level") != "low"),
        }
        self._append_shortcut_shadow_eval(goal, evaluation)
        return evaluation

    def _json_action_from_promotion(
        self,
        candidate: dict[str, Any],
        signature: dict[str, Any],
    ) -> json_action.JSONAction | None:
        action = self._json_action_from_shortcut_candidate(candidate) if isinstance(candidate, dict) else None
        if action is not None:
            return action
        if not isinstance(signature, dict):
            return None
        action_type = _clean_text(signature.get("action_type") or json_action.CLICK).lower()
        center = signature.get("center")
        if action_type in {"type", json_action.INPUT_TEXT}:
            return json_action.JSONAction(
                action_type=json_action.INPUT_TEXT,
                x=int(center[0]) if isinstance(center, (list, tuple)) and len(center) >= 2 and int(center[0]) > 0 else None,
                y=int(center[1]) if isinstance(center, (list, tuple)) and len(center) >= 2 and int(center[1]) > 0 else None,
                text=_clean_text(signature.get("typed_text") or ""),
            )
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            return json_action.JSONAction(action_type=json_action.CLICK, x=int(center[0]), y=int(center[1]))
        return None

    def _maybe_store_pending_promotion_from_trace(
        self,
        goal: str,
        step_idx: int,
        action: json_action.JSONAction,
        explore_trace: dict[str, Any],
    ) -> None:
        if not isinstance(explore_trace, dict):
            self._append_promotion_event(
                goal,
                {
                    "event_type": "promotion_not_stored",
                    "step": int(step_idx + 1),
                    "reason": "missing_explore_trace",
                    "unsafe_promotion": False,
                },
            )
            return
        if not getattr(self, "reasoning_prior_promotion", False):
            self._append_promotion_event(
                goal,
                {
                    "event_type": "promotion_not_stored",
                    "step": int(step_idx + 1),
                    "reason": "reasoning_prior_promotion_disabled",
                    "unsafe_promotion": False,
                },
            )
            return
        if explore_trace.get("status") == "rollback_failed":
            self._append_promotion_event(
                goal,
                {
                    "event_type": "promotion_not_stored",
                    "step": int(step_idx + 1),
                    "reason": "exploration_rollback_failed",
                    "unsafe_promotion": False,
                },
            )
            return
        candidates: list[dict[str, Any]] = []
        guidance_candidates: list[dict[str, Any]] = []
        for obs in list(explore_trace.get("observations") or []):
            if not isinstance(obs, dict):
                continue
            if not bool((obs.get("rollback") or {}).get("success")):
                continue
            root_sig = obs.get("root_action_signature") if isinstance(obs.get("root_action_signature"), dict) else {}
            depth2_candidate = obs.get("depth2_candidate") if isinstance(obs.get("depth2_candidate"), dict) else {}
            root_match = self._safe_signature_match_reason(root_sig, action)
            if not bool(obs.get("promotable_for_promotion")):
                evidence_type_for_guidance = _clean_text(obs.get("evidence_type"))
                if root_match >= 0.8 and evidence_type_for_guidance in {"ACTION_HINT", "SCHEMA_HINT", "ANSWER_HINT"}:
                    guidance_candidates.append({
                        "event_type": "promotion_stored",
                        "promotion_level": "guidance",
                        "stored_at_step": int(step_idx + 1),
                        "fire_at_step": int(step_idx + 2),
                        "step": int(step_idx + 1),
                        "branch_id": obs.get("branch_id"),
                        "root_action_match_score": float(root_match),
                        "depth1_state_signature": obs.get("depth1_state_signature") if isinstance(obs.get("depth1_state_signature"), dict) else {},
                        "guidance_context": _clean_text(obs.get("evidence_type")) + ": " + _clean_text((obs.get("labels") or [""])[-1] if isinstance(obs.get("labels"), list) else obs.get("stop_reason")),
                        "reason": "guidance_promotion_available",
                        "unsafe_promotion": False,
                    })
                continue
            safe, safe_reason = self._shortcut_candidate_safe(depth2_candidate, goal, None)
            if root_match < 0.8 or not safe:
                self._append_promotion_event(
                    goal,
                    {
                        "event_type": "promotion_not_stored",
                        "step": int(step_idx + 1),
                        "branch_id": obs.get("branch_id"),
                        "root_action_match_score": float(root_match),
                        "depth2_safe": bool(safe),
                        "safe_reason": safe_reason,
                        "reason": "root_mismatch_or_depth2_unsafe",
                        "unsafe_promotion": bool(not safe),
                    },
                )
                continue
            record = self._build_promotion_record(
                goal=goal,
                step_idx=step_idx,
                prior_source_step=int(explore_trace.get("reasoning_prior_source_step") or 0),
                branch_observation=obs,
                root_match_score=float(root_match),
                state_match_score=0.0,
                state_match_reason="deferred_to_next_step",
                depth2_entropy=float(explore_trace.get("reasoning_prior_entropy") or 0.0),
                state_match=True,
            )
            record.update(
                {
                    "event_type": "promotion_stored",
                    "stored_at_step": int(step_idx + 1),
                    "fire_at_step": int(step_idx + 2),
                    "depth2_safe": True,
                    "safe_reason": safe_reason,
                    "unsafe_promotion": False,
                }
            )
            candidates.append(record)
        if not candidates and guidance_candidates:
            guidance_candidates.sort(key=lambda row: float(row.get("root_action_match_score") or 0.0), reverse=True)
            self._pending_promotion = dict(guidance_candidates[0])
            self._append_pending_promotion_record(goal, guidance_candidates[0])
            self._append_promotion_event(goal, guidance_candidates[0])
            return
        if not candidates:
            self._append_promotion_event(
                goal,
                {
                    "event_type": "promotion_not_stored",
                    "step": int(step_idx + 1),
                    "reason": "no_depth2",
                    "unsafe_promotion": False,
                },
            )
            return
        candidates.sort(
            key=lambda row: (
                float(row.get("root_action_match_score") or 0.0),
                float(row.get("entropy") or 0.0) * -1.0,
            ),
            reverse=True,
        )
        self._pending_promotion = dict(candidates[0])
        self._append_pending_promotion_record(goal, candidates[0])
        self._append_promotion_event(goal, candidates[0])

    def _try_execute_pending_promotion(
        self,
        *,
        goal: str,
        step_idx: int,
        start_time: float,
        state: Any,
        current_activity: str,
        current_hash: int,
    ) -> base_agent.AgentInteractionResult | None:
        del current_hash
        pending = self._pending_promotion if isinstance(self._pending_promotion, dict) else {}
        self._pending_promotion = None
        if not pending:
            return None
        state_sig = pending.get("depth1_state_signature") if isinstance(pending.get("depth1_state_signature"), dict) else {}
        if _clean_text(pending.get("promotion_level")) == "guidance":
            state_score, state_reason = self._promote_signature_match_reason(state_sig, state, current_activity)
            event = dict(pending)
            event.update({
                "event_type": "promotion_guidance_injected" if state_score >= 0.60 else "promotion_eval",
                "step": int(step_idx + 1),
                "state_match_score": float(state_score),
                "state_match_reason": state_reason,
                "would_fire": False,
                "fired": False,
                "no_fire_reason": "" if state_score >= 0.60 else "state_mismatch",
                "unsafe_promotion": False,
            })
            if state_score >= 0.60:
                self._pending_promotion_guidance_context = _clean_text(pending.get("guidance_context"))
            self._append_promotion_event(goal, event)
            return None
        action_sig = pending.get("depth2_action_signature") if isinstance(pending.get("depth2_action_signature"), dict) else {}
        candidate = pending.get("depth2_candidate") if isinstance(pending.get("depth2_candidate"), dict) else {}
        state_score, state_reason = self._promote_signature_match_reason(state_sig, state, current_activity)
        locatable = self._locate_candidate_action_in_state(state, action_sig) or self._candidate_visible_in_state(candidate, state)
        safe, safe_reason = self._shortcut_candidate_safe(candidate, goal, state)
        action = self._json_action_from_promotion(candidate, action_sig)
        no_fire_reasons: list[str] = []
        if state_score < 0.80:
            no_fire_reasons.append(f"depth1_state_mismatch:{state_reason}")
        if not locatable:
            no_fire_reasons.append("depth2_action_not_locatable")
        if not safe:
            no_fire_reasons.append(f"depth2_action_unsafe:{safe_reason}")
        if action is None:
            no_fire_reasons.append("promotion_action_build_failed")
        event = dict(pending)
        event.update(
            {
                "event_type": "promotion_eval",
                "step": int(step_idx + 1),
                "state_match_score": float(state_score),
                "state_match_reason": state_reason,
                "depth2_action_locatable_now": bool(locatable),
                "depth2_safe_now": bool(safe),
                "safe_reason_now": safe_reason,
                "would_fire": not no_fire_reasons,
                "fired": False,
                "no_fire_reason": ";".join(no_fire_reasons),
                "unsafe_promotion": bool(not safe),
            }
        )
        if no_fire_reasons or action is None:
            self._append_promotion_event(goal, event)
            return None
        extras = {
            "promotion_id": _clean_text(pending.get("plan_id") or f"promotion_{step_idx + 1}"),
            "skipped_vlm_reasoning": True,
            "reasoning_prior_promotion": True,
        }
        active_action_start = time.time()
        self._execute_action(action, extras)
        self._append_latency_profile_event(
            goal,
            {
                "event": "promotion_action_execute",
                "step": int(step_idx + 1),
                "prompt_mode": "promotion_active",
                "action_type": str(action.action_type),
                "latency_ms": float(max(0.0, time.time() - active_action_start) * 1000.0),
                "skipped_vlm_reasoning": True,
            },
        )
        event["fired"] = True
        event["active_action_dict"] = dict(action.__dict__)
        event["unsafe_promotion"] = False
        self._append_promotion_event(goal, event)
        summary = f"Executed rollback-verified promotion from step {pending.get('stored_at_step')}"
        parsed_action = {
            "action": "PROMOTION",
            "summary": summary,
            "skipped_vlm_reasoning": True,
        }
        tool_call = {"name": "mobile_use", "arguments": {"action": action.action_type, "promotion": True}}
        screenshot = Image.fromarray(state.pixels)
        latency_sec = float(max(0.0, time.time() - start_time))
        step_record = {
            "goal": goal,
            "response": "",
            "parsed_action": parsed_action,
            "tool_call": tool_call,
            "action_dict": action.__dict__,
            "summary": summary,
            "latency_sec": latency_sec,
            "prompt_mode": "promotion_active",
            "prompt_hint": "",
            "matched_exploration_status": "promotion_fired",
            "matched_exploration_count": 0,
            "matched_exploration_results": [],
            "light_explore_runs": self._light_explore_runs,
            "start_page_activity": current_activity,
            "start_page_hash": self._state_hash(state),
            "page_stalled": False,
            "task_mode": self._task_mode(goal),
            "ablation_variant": self.light_explore_variant,
            "exploration_status": "promotion_fired",
            "exploration_trigger_reason": "pending_promotion",
            "exploration_candidate_count": 0,
            "exploration_selected_target_count": 0,
            "exploration_observation_count": 0,
            "rollback_success": None,
            "rollback_levels": [],
            "planned_action_suppressed": False,
            "shortcut_fired": False,
            "promotion_fired": True,
            "skipped_vlm_call": True,
            "skipped_vlm_calls": 1,
            "promotion_event": event,
            "state_acquisition_metrics": dict(self._state_acquisition_metrics),
        }
        self._actions.append(step_record)
        self._summaries.append(summary)
        self._responses.append("")
        task_dir = self._task_output_dir(goal)
        if task_dir:
            os.makedirs(task_dir, exist_ok=True)
            screenshot.save(os.path.join(task_dir, f"screenshot_{len(self._actions) - 1}.png"))
            self._write_action_log(goal)
        print(f"[PROMOTION {_now_hms()}] step: {step_idx + 1} fired action={action.action_type}")
        return base_agent.AgentInteractionResult(
            done=False,
            data={
                "response": "",
                "parsed_action": parsed_action,
                "tool_call": tool_call,
                "action": repr(action),
                "action_dict": action.__dict__,
                "summary": summary,
                "hints": [],
                "latency_sec": latency_sec,
                "prompt_mode": "promotion_active",
                "promotion_fired": True,
                "skipped_vlm_calls": 1,
            },
        )

    def _execute_active_shortcut_step(
        self,
        *,
        goal: str,
        step_idx: int,
        start_time: float,
        state: Any,
        shortcut_event: dict[str, Any],
    ) -> base_agent.AgentInteractionResult | None:
        if self.t2_shortcut_mode != "active_safe" or not isinstance(shortcut_event, dict) or not shortcut_event.get("would_fire"):
            return None
        shortcut_action_info = shortcut_event.get("shortcut_t2_action") if isinstance(shortcut_event.get("shortcut_t2_action"), dict) else {}
        candidate = shortcut_action_info.get("candidate") if isinstance(shortcut_action_info.get("candidate"), dict) else {}
        action = self._json_action_from_shortcut_candidate(candidate)
        if action is None:
            shortcut_event["fired"] = False
            shortcut_event["no_fire_reason"] = "shortcut_action_build_failed"
            self._append_shortcut_event(goal, dict(shortcut_event))
            return None
        extras: dict[str, Any] = {
            "shortcut_plan_id": shortcut_event.get("plan_id"),
            "skipped_vlm_reasoning": True,
        }
        active_action_start = time.time()
        self._execute_action(action, extras)
        self._append_latency_profile_event(
            goal,
            {
                "event": "main_action_execute",
                "step": int(step_idx + 1),
                "prompt_mode": "shortcut_t2_active",
                "action_type": str(action.action_type),
                "latency_ms": float(max(0.0, time.time() - active_action_start) * 1000.0),
                "shortcut_plan_id": shortcut_event.get("plan_id"),
                "skipped_vlm_reasoning": True,
            },
        )
        fired_goal_keys = getattr(self, "_active_t2_fired_goal_keys", set())
        fired_goal_keys.add(_clean_text(goal)[:200])
        self._active_t2_fired_goal_keys = fired_goal_keys
        shortcut_event = dict(shortcut_event)
        shortcut_event["fired"] = True
        shortcut_event["skipped_vlm_reasoning"] = True
        shortcut_event["active_action_dict"] = dict(action.__dict__)
        self._append_shortcut_event(goal, shortcut_event)
        summary = f"Executed active t+2 shortcut plan {shortcut_event.get('plan_id')}"
        parsed_action = {
            "action": "SHORTCUT_T2_ACTIVE",
            "summary": summary,
            "shortcut_plan_id": shortcut_event.get("plan_id"),
            "skipped_vlm_reasoning": True,
        }
        tool_call = {"name": "mobile_use", "arguments": {"action": action.action_type, "shortcut_plan_id": shortcut_event.get("plan_id")}}
        screenshot = Image.fromarray(state.pixels)
        latency_sec = float(max(0.0, time.time() - start_time))
        step_record = {
            "goal": goal,
            "response": "",
            "parsed_action": parsed_action,
            "tool_call": tool_call,
            "action_dict": action.__dict__,
            "summary": summary,
            "latency_sec": latency_sec,
            "prompt_mode": "shortcut_t2_active",
            "prompt_hint": "",
            "matched_exploration_status": "shortcut_active_fired",
            "matched_exploration_count": 0,
            "matched_exploration_results": [],
            "light_explore_runs": self._light_explore_runs,
            "start_page_activity": self._foreground_activity_name(),
            "start_page_hash": self._state_hash(state),
            "page_stalled": False,
            "task_mode": self._task_mode(goal),
            "task_slots": slot_complete_evidence.parse_task_slots(goal).to_dict()
            if self.light_explore_slot_complete
            else {},
            "ablation_variant": self.light_explore_variant,
            "exploration_status": "shortcut_fired",
            "exploration_trigger_reason": "active_t2_shortcut",
            "exploration_candidate_count": 0,
            "exploration_selected_target_count": 0,
            "exploration_observation_count": 0,
            "post_planning_exploration_speculative_results": [],
            "rollback_success": None,
            "rollback_levels": [],
            "planned_action_suppressed": False,
            "shortcut_plan_id": shortcut_event.get("plan_id"),
            "shortcut_mode": self.t2_shortcut_mode,
            "shortcut_attempted": True,
            "shortcut_plan_created": False,
            "shortcut_would_fire": bool(shortcut_event.get("would_fire")),
            "shortcut_fired": True,
            "skipped_vlm_call": True,
            "skipped_vlm_calls": 1,
            "no_plan_reason": "",
            "no_fire_reason": shortcut_event.get("no_fire_reason") or "",
            "planned_root_action_summary": shortcut_event.get("planned_root_action") or {},
            "shortcut_t2_action_summary": shortcut_event.get("shortcut_t2_action") or {},
            "t1_state_match_method": shortcut_event.get("t1_state_match_method"),
            "t1_state_match_score": shortcut_event.get("t1_state_match_score"),
            "t2_target_visible": bool(shortcut_event.get("target_visible")),
            "shortcut_confidence": shortcut_event.get("confidence"),
            "shortcut_event": shortcut_event,
        }
        if "shortcut_plan_id" not in step_record:
            plan_summary = shortcut_plan if isinstance(shortcut_plan, dict) else {}
            event_summary = shortcut_event if isinstance(shortcut_event, dict) else {}
            planned_root_action_summary = (
                event_summary.get("planned_root_action")
                or plan_summary.get("planned_root_action")
                or self._action_summary_from_json_action(action)
            )
            shortcut_t2_action_summary = (
                event_summary.get("shortcut_t2_action")
                or plan_summary.get("shortcut_t2_action")
                or {}
            )
            step_record.update(
                {
                    "shortcut_mode": self.t2_shortcut_mode,
                    "shortcut_attempted": self.t2_shortcut_mode != "off",
                    "shortcut_plan_created": bool(plan_summary and not _clean_text(plan_summary.get("no_plan_reason"))),
                    "shortcut_plan_id": event_summary.get("plan_id") or plan_summary.get("plan_id"),
                    "shortcut_would_fire": bool(event_summary.get("would_fire")),
                    "shortcut_event_would_fire": bool(event_summary.get("would_fire")),
                    "shortcut_event_confidence": event_summary.get("confidence"),
                    "shortcut_event_no_fire_reason": event_summary.get("no_fire_reason"),
                    "shortcut_fired": False,
                    "skipped_vlm_call": False,
                    "skipped_vlm_calls": 0,
                    "no_plan_reason": plan_summary.get("no_plan_reason") or "",
                    "no_fire_reason": event_summary.get("no_fire_reason") or plan_summary.get("no_fire_reason") or "",
                    "planned_root_action_summary": planned_root_action_summary,
                    "shortcut_t2_action_summary": shortcut_t2_action_summary,
                    "t1_state_match_method": event_summary.get("t1_state_match_method"),
                    "t1_state_match_score": event_summary.get("t1_state_match_score"),
                    "t2_target_visible": bool(event_summary.get("target_visible")),
                    "shortcut_confidence": event_summary.get("confidence") or plan_summary.get("confidence"),
                    "shortcut_event": event_summary,
                }
            )
        self._actions.append(step_record)
        self._summaries.append(summary)
        self._responses.append("")
        task_dir = self._task_output_dir(goal)
        if task_dir:
            os.makedirs(task_dir, exist_ok=True)
            screenshot.save(os.path.join(task_dir, f"screenshot_{len(self._actions) - 1}.png"))
            self._write_action_log(goal)
        print(f"[SHORTCUT {_now_hms()}] step: {step_idx + 1} active_t2_fired plan={shortcut_event.get('plan_id')}")
        return base_agent.AgentInteractionResult(
            done=False,
            data={
                "response": "",
                "parsed_action": parsed_action,
                "tool_call": tool_call,
                "action": repr(action),
                "action_dict": action.__dict__,
                "summary": summary,
                "hints": [],
                "latency_sec": latency_sec,
                "prompt_mode": "shortcut_t2_active",
                "shortcut_fired": True,
                "skipped_vlm_calls": 1,
            },
        )

    def _match_pending_speculative_exploration(
        self,
        goal: str,
        step_idx: int,
        state: Any,
        current_activity: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        pending = list(self._pending_speculative_traces)
        self._pending_speculative_traces = []
        match_trace: dict[str, Any] = {
            "step": int(step_idx + 1),
            "goal": goal,
            "current_activity": current_activity,
            "pending_trace_count": len(pending),
            "candidate_match_count": 0,
            "matched_count": 0,
            "matches": [],
            "status": "no_pending",
            "prompt_context": "",
            "selected_prompt_results": [],
        }
        if not pending:
            self._append_match_trace(goal, match_trace)
            return "", [], match_trace

        goal_normalized = _clean_text(goal)
        current_activity_norm = _clean_text(current_activity)
        current_package = current_activity_norm.split("/", 1)[0] if "/" in current_activity_norm else current_activity_norm

        def _same_task_or_app(trace_goal: str, trace_activity: str) -> bool:
            trace_goal_clean = _clean_text(trace_goal)
            if trace_goal_clean and trace_goal_clean == goal_normalized:
                return True
            trace_activity_norm = _clean_text(trace_activity)
            trace_package = trace_activity_norm.split("/", 1)[0] if "/" in trace_activity_norm else trace_activity_norm
            return bool(current_package and trace_package and current_package == trace_package)

        matches: list[dict[str, Any]] = []
        all_attempts: list[dict[str, Any]] = []
        for trace in pending:
            for obs in list(trace.get("observations") or []):
                if not bool((obs.get("rollback") or {}).get("success")):
                    continue
                steps = [s for s in list(obs.get("steps") or []) if isinstance(s, dict)]
                depth1 = next((s for s in steps if int(s.get("depth") or 0) == 1), None)
                evidence_steps = [
                    s for s in steps if int(s.get("depth") or 0) >= 2
                ]
                evidence_step = max(
                    evidence_steps,
                    key=lambda s: int(s.get("depth") or 0),
                    default=None,
                )
                if not depth1:
                    continue
                source_type_for_depth1 = _clean_text(obs.get("evidence_type") or obs.get("boundary_type"))
                if (
                    evidence_step is None
                    and (
                        self.light_explore_fixed_framework
                        or bool(obs.get("no_action_inspect"))
                        or source_type_for_depth1
                        in {"ANSWER_HINT", "ACTION_HINT", "SCHEMA_HINT", "AVOID_HINT", "RISK_HINT"}
                    )
                ):
                    evidence_step = depth1
                if not evidence_step:
                    continue
                if not bool(depth1.get("changed")):
                    labels = [_clean_text(x) for x in list(obs.get("labels") or []) if _clean_text(x)]
                    alignment = self._anchor_overlap_alignment(state, current_activity, depth1)
                    observed_elements = list(evidence_step.get("observed_elements") or obs.get("observed_elements") or [])
                    observed_low = " ".join(_clean_text(x).lower() for x in observed_elements)
                    stop_reason = _clean_text(obs.get("stop_reason") or obs.get("boundary_type") or "")
                    stop_reason_low = stop_reason.lower()
                    source_type = _clean_text(obs.get("evidence_type") or obs.get("boundary_type"))
                    operator = _clean_text(obs.get("operator"))
                    trace_goal = _clean_text(trace.get("goal"))
                    trace_activity = _clean_text(obs.get("after_activity") or depth1.get("after_activity"))
                    hard_negative = bool(
                        obs.get("hard_negative")
                        or (
                            isinstance(obs.get("slot_evidence"), dict)
                            and bool((obs.get("slot_evidence") or {}).get("hard_negative"))
                        )
                        or _clean_text(obs.get("boundary_type")).upper() == "NEGATIVE_BOUNDARY"
                        or "hard_negative" in stop_reason_low
                        or observed_low.startswith("avoid ")
                        or " avoid " in f" {observed_low} "
                    )
                    risk_global = bool(
                        source_type == "RISK_HINT"
                        or operator == "RiskBoundary"
                        or any(token in stop_reason_low for token in ("risk", "unsafe", "destructive", "delete", "commit"))
                    )
                    schema_global = bool(
                        source_type == "SCHEMA_HINT"
                        and re.search(
                            r"\b(name|title|date|time|phone|email|amount|description|note|folder|file|field|search|filter|query)\b",
                            observed_low,
                        )
                    )
                    same_task_app = _same_task_or_app(trace_goal, trace_activity)
                    global_promptable = bool(
                        same_task_app
                        and (
                            (source_type == "AVOID_HINT" and hard_negative)
                            or risk_global
                            or schema_global
                        )
                    )
                    next_candidate = evidence_step.get("candidate") if isinstance(evidence_step.get("candidate"), dict) else {}
                    matched_by = "depth1_no_state_change"
                    state_match_score = float(alignment.get("state_match_score") or 0.0)
                    if global_promptable:
                        if source_type == "AVOID_HINT" or hard_negative:
                            matched_by = "global_avoid_same_task_app"
                        elif risk_global:
                            matched_by = "global_risk_same_task_app"
                        else:
                            matched_by = "global_schema_same_task_app"
                        state_match_score = max(0.55, state_match_score)
                    record = {
                        "source_step": trace.get("step"),
                        "source_goal": trace_goal,
                        "branch_id": obs.get("branch_id"),
                        "path": " -> ".join(labels),
                        "matched_prefix": labels[0] if labels else "depth1",
                        "next_label": _clean_text(next_candidate.get("label") or (labels[0] if labels else "")),
                        "matched": bool(global_promptable),
                        "matched_by": matched_by,
                        "state_match_score": float(state_match_score),
                        "anchor_overlap": float(alignment.get("anchor_overlap") or 0.0),
                        "depth_reached": int(obs.get("depth_reached") or 0),
                        "evidence_depth": int(evidence_step.get("depth") or 0),
                        "evidence_changed": bool(evidence_step.get("changed")),
                        "score": float(obs.get("score") or 0.0),
                        "next_candidate": next_candidate,
                        "observed_elements": observed_elements,
                        "depth1_screenshot": depth1.get("screenshot"),
                        "depth2_screenshot": evidence_step.get("screenshot"),
                        "evidence_screenshot": evidence_step.get("screenshot"),
                        "after_activity": obs.get("after_activity") or evidence_step.get("after_activity"),
                        "rollback_level": (obs.get("rollback") or {}).get("level"),
                        "rollback_mode": (obs.get("rollback") or {}).get("mode"),
                        "rollback_verified": bool((obs.get("rollback") or {}).get("success")),
                        "operator": obs.get("operator"),
                        "operator_reason": obs.get("operator_reason"),
                        "boundary_type": obs.get("boundary_type"),
                        "stop_reason": obs.get("stop_reason"),
                        "evidence_type": obs.get("evidence_type"),
                        "evidence_gain": obs.get("evidence_gain"),
                        "confidence": obs.get("confidence"),
                        "slot_evidence": obs.get("slot_evidence") if isinstance(obs.get("slot_evidence"), dict) else {},
                        "hard_negative": hard_negative,
                        "global_promptable_no_state_change": bool(global_promptable),
                        "same_task_or_app": bool(same_task_app),
                    }
                    self._append_diagnostic_jsonl(
                        goal,
                        "state_alignment.jsonl",
                        {
                            "step": int(step_idx + 1),
                            "source_exploration_step": trace.get("step"),
                            "branch_id": obs.get("branch_id"),
                            **alignment,
                            "aligned": bool(global_promptable),
                            "matched_by": matched_by,
                            "global_promptable_no_state_change": bool(global_promptable),
                        },
                    )
                    all_attempts.append(record)
                    if global_promptable:
                        matches.append(record)
                    continue
                same, matched_by = self._state_matches_explored_step(state, current_activity, depth1)
                alignment = self._anchor_overlap_alignment(state, current_activity, depth1)
                if getattr(self, "light_explore_upper_bound_evidence", False):
                    same = bool(alignment.get("aligned"))
                    matched_by = (
                        f"activity_anchor_overlap:{float(alignment.get('anchor_overlap') or 0.0):.2f}"
                        if same
                        else (
                            "activity_mismatch"
                            if not alignment.get("activity_match")
                            else f"anchor_overlap_low:{float(alignment.get('anchor_overlap') or 0.0):.2f}"
                        )
                    )
                source_type = _clean_text(obs.get("evidence_type") or obs.get("boundary_type") or "")
                same_task_app = bool(_same_task_or_app(trace.get("goal"), obs.get("after_activity") or depth1.get("after_activity")))
                stop_reason_low = _clean_text(obs.get("stop_reason") or obs.get("boundary_type") or "").lower()
                operator = _clean_text(obs.get("operator"))
                hard_negative = bool(
                    obs.get("hard_negative")
                    or (
                        isinstance(obs.get("slot_evidence"), dict)
                        and bool((obs.get("slot_evidence") or {}).get("hard_negative"))
                    )
                    or _clean_text(obs.get("boundary_type")).upper() == "NEGATIVE_BOUNDARY"
                )
                observed_for_match = " ".join(_clean_text(x).lower() for x in list(evidence_step.get("observed_elements") or obs.get("observed_elements") or []))
                schema_promptable = bool(
                    source_type == "SCHEMA_HINT"
                    and same_task_app
                    and re.search(
                        r"\b(name|title|date|time|phone|email|amount|description|note|folder|file|field|search|filter|query)\b",
                        observed_for_match,
                    )
                )
                action_promptable = bool(
                    source_type == "ACTION_HINT"
                    and same_task_app
                    and same
                    and not self._is_transaction_unsafe_candidate(obs.get("depth1_candidate") if isinstance(obs.get("depth1_candidate"), dict) else {})
                )
                global_promptable_no_state_match = bool(
                    same_task_app
                    and (
                        action_promptable
                        or schema_promptable
                        or (
                            source_type == "AVOID_HINT"
                            and (
                                hard_negative
                                or any(
                                    token in stop_reason_low
                                    for token in ("hard_negative", "wrong_target", "wrong_app", "not_found", "dead_end", "negative_boundary")
                                )
                            )
                        )
                        or (
                            source_type == "RISK_HINT"
                            and (
                                operator == "RiskBoundary"
                                or any(token in stop_reason_low for token in ("risk", "unsafe", "destructive", "delete", "commit"))
                            )
                        )
                    )
                )
                if not same and global_promptable_no_state_match:
                    same = True
                    alignment = dict(alignment)
                    alignment["state_match_score"] = max(float(alignment.get("state_match_score") or 0.0), 0.55)
                    alignment["state_match_anchor_reason"] = "global_prompt_no_state_match"
                    matched_by = (
                        "global_avoid_same_task_app"
                        if source_type == "AVOID_HINT"
                        else "global_risk_same_task_app"
                    )
                labels = [_clean_text(x) for x in list(obs.get("labels") or []) if _clean_text(x)]
                evidence_depth = int(evidence_step.get("depth") or 0)
                next_candidate = evidence_step.get("candidate") if isinstance(evidence_step.get("candidate"), dict) else {}
                if self.light_explore_fixed_framework and evidence_depth <= 1 and obs.get("evidence_type") != "RISK_HINT":
                    next_candidate = {}
                if global_promptable_no_state_match and source_type in {"ACTION_HINT", "SCHEMA_HINT"}:
                    same = True
                    matched_by = "state_aligned_action" if source_type == "ACTION_HINT" else "same_app_schema"
                    alignment = dict(alignment)
                    alignment["state_match_score"] = max(0.55, float(alignment.get("state_match_score") or 0.0))
                record = {
                    "source_step": trace.get("step"),
                    "source_goal": _clean_text(trace.get("goal")),
                    "branch_id": obs.get("branch_id"),
                    "path": " -> ".join(labels),
                    "matched_prefix": labels[0] if labels else "depth1",
                    "next_label": (
                        labels[evidence_depth - 1]
                        if len(labels) >= evidence_depth and evidence_depth > 0
                        else _clean_text(next_candidate.get("label"))
                    ),
                    "matched": bool(same),
                    "matched_by": matched_by,
                    "state_match_score": float(alignment.get("state_match_score") or (1.0 if same else 0.0)),
                    "anchor_overlap": float(alignment.get("anchor_overlap") or 0.0),
                    "depth_reached": int(obs.get("depth_reached") or 0),
                    "evidence_depth": evidence_depth,
                    "evidence_changed": bool(evidence_step.get("changed")),
                    "score": float(obs.get("score") or 0.0),
                    "next_candidate": next_candidate,
                    "observed_elements": list(evidence_step.get("observed_elements") or obs.get("observed_elements") or []),
                    "depth1_screenshot": depth1.get("screenshot"),
                    "depth2_screenshot": evidence_step.get("screenshot"),
                    "evidence_screenshot": evidence_step.get("screenshot"),
                    "after_activity": obs.get("after_activity") or evidence_step.get("after_activity"),
                    "rollback_level": (obs.get("rollback") or {}).get("level"),
                    "rollback_mode": (obs.get("rollback") or {}).get("mode"),
                    "rollback_verified": bool((obs.get("rollback") or {}).get("success")),
                    "operator": obs.get("operator"),
                    "operator_reason": obs.get("operator_reason"),
                    "boundary_type": obs.get("boundary_type"),
                    "stop_reason": obs.get("stop_reason"),
                    "evidence_type": obs.get("evidence_type"),
                    "evidence_gain": obs.get("evidence_gain"),
                    "confidence": obs.get("confidence"),
                    "slot_evidence": obs.get("slot_evidence") if isinstance(obs.get("slot_evidence"), dict) else {},
                    "hard_negative": bool(
                        obs.get("hard_negative")
                        or (
                            isinstance(obs.get("slot_evidence"), dict)
                            and bool((obs.get("slot_evidence") or {}).get("hard_negative"))
                        )
                        or _clean_text(obs.get("boundary_type")).upper() == "NEGATIVE_BOUNDARY"
                    ),
                    "same_task_or_app": bool(same_task_app),
                }
                self._append_diagnostic_jsonl(
                    goal,
                    "state_alignment.jsonl",
                    {
                        "step": int(step_idx + 1),
                        "source_exploration_step": trace.get("step"),
                        "branch_id": obs.get("branch_id"),
                        **alignment,
                        "aligned": bool(same),
                        "matched_by": matched_by,
                    },
                )
                all_attempts.append(record)
                if same:
                    matches.append(record)

        match_trace["candidate_match_count"] = len(all_attempts)
        match_trace["matched_count"] = len(matches)
        match_trace["matches"] = all_attempts
        if matches:
            context, selected = self._build_prompt_context_from_state_aligned_matches(
                matches,
                goal=goal,
                current_state=state,
            )
            if self.light_explore_disable_prompt_injection:
                match_trace["would_prompt_context"] = context
                match_trace["would_selected_prompt_results"] = selected
                match_trace["status"] = "matched_prompt_injection_disabled"
                context, selected = "", []
            else:
                match_trace["status"] = "matched" if context else "matched_no_actionable_hint"
            match_trace["prompt_context"] = context
            match_trace["selected_prompt_results"] = selected
            if context and selected:
                prior_for_block = self._reasoning_prior_for_next_step if isinstance(self._reasoning_prior_for_next_step, dict) else {}
                self._append_filled_exploration_block(goal, context, step_idx + 1, prior_for_block, selected)
            if self.light_explore_hint_policy == "strict" and not selected:
                match_trace["not_injected_reasons"] = list(getattr(self, "_last_strict_not_injected_reasons", []))
        else:
            context, selected = "", []
            match_trace["status"] = "no_match"
        self._append_match_trace(goal, match_trace)
        return context, selected, match_trace

    @staticmethod
    def _voc_norm(value: Any, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return float(default)

    def _voc_task_phase(
        self,
        goal: str,
        step_idx: int,
        root_labels: list[str],
        page_stalled: bool,
        prior: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        phase, reason, _ = self._selective_task_phase(
            goal=goal,
            step_idx=step_idx,
            labels=root_labels,
            page_stalled=page_stalled,
            prior=prior,
        )
        return phase, reason

    def _voc_information_deficit(
        self,
        goal: str,
        root_labels: list[str],
        phase: str,
    ) -> tuple[float, dict[str, Any]]:
        labels_low = " ".join(_clean_text(x).lower() for x in root_labels if _clean_text(x))
        entities = [e for e in self._task_entities(goal) if len(e) >= 3][:8]
        missing_entities = [e for e in entities if e.lower() not in labels_low]
        entity_deficit = len(missing_entities) / float(max(1, len(entities)))
        facts = self._extract_answer_facts_from_labels(root_labels, goal, limit=4) if self.light_explore_answer_extractors else []
        useful_facts = [x for x in facts if not _clean_text(x).lower().startswith("avoid ")]
        answer_deficit = 0.0 if useful_facts else 1.0
        phase_bias = {
            "TARGET_ACQUISITION": 0.90,
            "EVIDENCE_GATHERING": 0.75,
            "STUCK_RECOVERY": 0.85,
            "ACTION_EXECUTION": 0.35,
            "VERIFICATION": 0.25,
            "BOOTSTRAP": 0.50,
        }.get(phase, 0.60)
        deficit = max(0.0, min(1.0, 0.40 * entity_deficit + 0.35 * answer_deficit + 0.25 * phase_bias))
        return deficit, {
            "entity_deficit": float(entity_deficit),
            "answer_deficit": float(answer_deficit),
            "phase_bias": float(phase_bias),
            "missing_entities": missing_entities,
            "visible_answer_facts": useful_facts[:4],
        }

    def _voc_candidate_ambiguity(
        self,
        candidate: dict[str, Any],
        candidates: list[dict[str, Any]],
        goal: str,
    ) -> tuple[float, list[str]]:
        label = _clean_text(candidate.get("label") or candidate.get("merged"))
        label_low = label.lower()
        reasons: list[str] = []
        score = 0.0
        if not label_low or label_low in {"android.view.view", "android.widget.imageview", "view", "imageview", "simple"}:
            score += 0.35
            reasons.append("generic_or_empty_label")
        same_label_count = sum(
            1
            for other in candidates
            if _clean_text(other.get("label") or other.get("merged")).lower() == label_low and label_low
        )
        if same_label_count >= 2:
            score += min(0.30, 0.08 * same_label_count)
            reasons.append(f"duplicate_label_count:{same_label_count}")
        lexical = self._lexical_overlap_score(
            merged=label,
            goal_tokens=self._goal_tokens(goal),
            app_keywords=self._goal_app_keywords(goal),
        )
        if lexical < 0.05:
            score += 0.20
            reasons.append("low_task_lexical_grounding")
        a11y = candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {}
        if not any(_clean_text(a11y.get(k)) for k in ("text", "content_description", "hint_text", "resource_id")):
            score += 0.15
            reasons.append("weak_a11y_grounding")
        return max(0.0, min(1.0, score)), reasons

    def _voc_candidate_grounding(
        self,
        candidate: dict[str, Any],
        goal: str,
    ) -> dict[str, Any]:
        label = _clean_text(candidate.get("label") or candidate.get("merged"))
        a11y = candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {}
        lexical = self._lexical_overlap_score(
            merged=label,
            goal_tokens=self._goal_tokens(goal),
            app_keywords=self._goal_app_keywords(goal),
        )
        entity_hits = [
            e
            for e in self._task_entities(goal)[:8]
            if e and e.lower() in label.lower()
        ]
        return {
            "label": label,
            "operator": _clean_text(candidate.get("operator")),
            "action_kind": _clean_text(candidate.get("action_kind") or "click"),
            "layout_region": _clean_text(candidate.get("layout_region")),
            "semantic_role": _clean_text(candidate.get("semantic_role")),
            "lexical_overlap": float(lexical),
            "entity_hits": entity_hits,
            "resource_id": _clean_text(a11y.get("resource_id")),
            "class_name": _clean_text(a11y.get("class_name")),
            "center": candidate.get("center"),
        }

    def _voc_estimated_rollback_cost(self, candidate: dict[str, Any], operator: str) -> tuple[float, list[str]]:
        reasons: list[str] = []
        cost = 0.15
        action_kind = _clean_text(candidate.get("action_kind") or "click").lower()
        layout = _clean_text(candidate.get("layout_region")).lower()
        role = _clean_text(candidate.get("semantic_role")).lower()
        label = _clean_text(candidate.get("label") or candidate.get("merged")).lower()
        if operator in {"SearchPeek", "FilterPeek", "ListInspect"}:
            cost -= 0.05
            reasons.append("low_cost_information_operator")
        if action_kind in {"type", json_action.INPUT_TEXT}:
            cost += 0.20
            reasons.append("text_input_requires_precise_restore")
        if layout in {"top_bar", "nav_bar", "bottom_nav"}:
            cost += 0.05
            reasons.append(f"navigation_region:{layout}")
        if role in {"button", "menu_item"} and re.search(r"\b(add|new|create|save|delete|remove|ok|done)\b", label):
            cost += 0.30
            reasons.append("commit_like_label")
        if operator == "RiskBoundary":
            cost = 1.0
            reasons.append("risk_boundary_not_executable")
        return max(0.0, min(1.0, cost)), reasons

    def _selective_depth2_consumable_decision(
        self,
        *,
        goal: str,
        task_mode: str,
        task_phase: str,
        state_after_depth1: Any,
        observed_elements: list[str],
        branch_operator: str,
        candidate: dict[str, Any],
        rollback_cost: float = 0.0,
    ) -> dict[str, Any]:
        label_blob = " ".join(_clean_text(x).lower() for x in observed_elements if _clean_text(x))
        operator = _clean_text(branch_operator or candidate.get("operator"))
        if self._is_transaction_unsafe_candidate(candidate) or operator == "RiskBoundary":
            return {"depth2_allowed": False, "depth1_result_type": "unsafe", "depth2_reason": "", "depth2_block_reason": "root_action_unsafe"}
        if task_mode in {"FORM_CREATE_EDIT", "DELETE_COMMIT", "SIMPLE_VERIFY", "MEDIA_CAPTURE"} and task_phase != "STUCK_RECOVERY":
            schema = bool(re.search(r"\b(search|filter|field|title|name|date|time|description|phone|email)\b", label_blob) or operator == "FormSchema")
            return {"depth2_allowed": False, "depth1_result_type": "form_schema" if schema else "passive_only", "depth2_reason": "schema_record_only", "depth2_block_reason": "deterministic_passive_only"}
        if rollback_cost and rollback_cost > 0.75:
            return {"depth2_allowed": False, "depth1_result_type": "high_rollback_risk", "depth2_reason": "", "depth2_block_reason": "rollback_risk_too_high"}
        if self._state_has_search_ui(state_after_depth1):
            return {"depth2_allowed": True, "depth1_result_type": "search_field", "depth2_action_type": "type_search_query", "depth2_reason": "search_field_appeared", "depth2_block_reason": ""}
        if operator == "FilterPeek" or re.search(r"\b(filter|category|date|october|snow boarding|kayaking|climbing|activity type)\b", label_blob):
            return {"depth2_allowed": True, "depth1_result_type": "filter_page", "depth2_action_type": "select_exact_filter", "depth2_reason": "filter_page_appeared", "depth2_block_reason": ""}
        target_terms = [e.lower() for e in self._task_entities(goal) if len(e) >= 3][:10]
        if target_terms and any(term in label_blob for term in target_terms):
            return {"depth2_allowed": True, "depth1_result_type": "result_list", "depth2_action_type": "click_exact_target", "depth2_reason": "target_result_visible", "depth2_block_reason": ""}
        if re.search(r"\b(duration|distance|total|longest|count|statistics|stats|details|summary)\b", label_blob):
            return {"depth2_allowed": False, "depth1_result_type": "detail_or_stats_page", "depth2_action_type": "extract_answer", "depth2_reason": "detail_stats_extract_answer", "depth2_block_reason": "extract_answer_no_click"}
        if re.search(r"\b(title|name|date|time|description|phone|email|field|form)\b", label_blob):
            return {"depth2_allowed": False, "depth1_result_type": "form_schema", "depth2_action_type": "record_schema", "depth2_reason": "form_schema_record_only", "depth2_block_reason": "schema_no_content_input"}
        return {"depth2_allowed": False, "depth1_result_type": "no_consumable_state", "depth2_reason": "", "depth2_block_reason": "no_target_query_schema_detail"}

    def _session_memory_key(self, goal: str, app_name: str) -> str:
        app = _clean_text(app_name or ",".join(self._goal_app_keywords(goal)[:2]) or "unknown_app").lower()
        return "|".join([app, self._task_mode(goal), _clean_text(goal)[:120].lower()])

    def _branch_memory_key(self, candidate: dict[str, Any], goal: str) -> str:
        operator = _clean_text(candidate.get("operator") or self._candidate_operator(candidate, goal)[0] or "Other")
        label = _clean_text(candidate.get("label") or candidate.get("merged") or "")
        a11y = candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {}
        rid = _clean_text(a11y.get("resource_id") or candidate.get("resource_id") or "")
        role = _clean_text(candidate.get("semantic_role") or "")
        return "|".join([operator.lower(), role.lower(), rid.lower(), label.lower()[:80]])

    def _session_memory_for(self, goal: str, app_name: str) -> dict[str, Any]:
        key = self._session_memory_key(goal, app_name)
        memory = self._session_exploration_memory.setdefault(
            key,
            {
                "session_key": key,
                "branch_stats": {},
                "component_history": [],
                "reward_history": [],
                "latency_ewma_ms": float(getattr(self, "_reasoning_prior_ewma_ms", 10000.0) or 10000.0),
                "rollback_success": 0.0,
                "rollback_total": 0.0,
            },
        )
        return memory

    @staticmethod
    def _clip01(value: Any, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return float(default)

    def _session_component_weights(self, memory: dict[str, Any], feature_names: list[str]) -> dict[str, float]:
        history = [
            row for row in list(memory.get("component_history") or [])
            if isinstance(row, dict) and isinstance(row.get("features"), dict)
        ]
        if len(history) < 3:
            uniform = 1.0 / float(max(1, len(feature_names)))
            return {name: uniform for name in feature_names}
        rewards = [float(row.get("reward") or 0.0) for row in history[-24:]]
        mean_reward = sum(rewards) / float(max(1, len(rewards)))
        weights: dict[str, float] = {}
        for name in feature_names:
            values = [self._clip01((row.get("features") or {}).get(name)) for row in history[-24:]]
            mean_value = sum(values) / float(max(1, len(values)))
            covariance = sum(
                (value - mean_value) * (reward - mean_reward)
                for value, reward in zip(values, rewards)
            ) / float(max(1, len(values)))
            weights[name] = max(0.0, covariance)
        total = sum(weights.values())
        if total <= 1e-9:
            uniform = 1.0 / float(max(1, len(feature_names)))
            return {name: uniform for name in feature_names}
        return {name: float(value) / total for name, value in weights.items()}

    def _session_adaptive_candidate_value(
        self,
        *,
        goal: str,
        step_idx: int,
        candidate: dict[str, Any],
        candidates: list[dict[str, Any]],
        memory: dict[str, Any],
        phase: str,
        info_deficit: float,
        grounding: dict[str, Any],
        ambiguity: float,
        rollback_cost: float,
        score_components: dict[str, Any],
        prior: dict[str, Any] | None,
    ) -> dict[str, Any]:
        branch_key = self._branch_memory_key(candidate, goal)
        branch_stats = (memory.get("branch_stats") or {}).get(branch_key, {})
        total_rollbacks = float(memory.get("rollback_total") or 0.0)
        rollback_success_rate = (
            float(memory.get("rollback_success") or 0.0) / total_rollbacks
            if total_rollbacks > 0
            else 1.0
        )
        prior_confidence = self._clip01((prior or {}).get("confidence"), 0.0)
        feature_names = [
            "prior_alignment",
            "slot_gain",
            "entity_match",
            "state_grounding",
            "evidence_gain",
            "ambiguity_resolution",
            "rollback_safety",
            "latency_fit",
            "novelty",
            "session_success",
        ]
        latency_budget_ms = max(1000.0, float(memory.get("latency_ewma_ms") or self._reasoning_prior_ewma_ms or 10000.0))
        estimated_latency_ms = latency_budget_ms * (0.40 + rollback_cost)
        session_n = float(branch_stats.get("n") or 0.0)
        session_q = float(branch_stats.get("q") or 0.0)
        features = {
            "prior_alignment": self._clip01(candidate.get("reasoning_prior_score"), 0.0) * max(0.25, prior_confidence),
            "slot_gain": self._clip01(score_components.get("MissingSlotGain"), 0.0),
            "entity_match": self._clip01(score_components.get("TaskEntityMatch"), 0.0),
            "state_grounding": max(
                self._clip01(grounding.get("lexical_overlap"), 0.0),
                self._clip01(candidate.get("relevance"), 0.0),
            ),
            "evidence_gain": self._clip01(float(candidate.get("estimated_evidence_gain") or 0.0) / 10.0, 0.0),
            "ambiguity_resolution": self._clip01(float(info_deficit) * float(ambiguity), 0.0),
            "rollback_safety": self._clip01((1.0 - rollback_cost) * rollback_success_rate, 0.0),
            "latency_fit": self._clip01(1.0 - (estimated_latency_ms / max(1.0, latency_budget_ms * 2.0)), 0.0),
            "novelty": self._clip01(1.0 / float(session_n + 1.0), 0.0),
            "session_success": self._clip01((session_q + 1.0) / 2.0, 0.5),
        }
        weights = self._session_component_weights(memory, feature_names)
        utility = sum(float(weights.get(name) or 0.0) * float(features.get(name) or 0.0) for name in feature_names)
        if phase in {"action_execution", "verification"} and float(info_deficit) < 0.35:
            utility = min(utility, 0.0)
        if rollback_cost >= 0.75:
            utility = min(utility, 0.0)
        depth2_features = dict(features)
        depth2_features["novelty"] = self._clip01(depth2_features["novelty"] * 0.75, 0.0)
        depth2_features["latency_fit"] = self._clip01(depth2_features["latency_fit"] - 0.15, 0.0)
        depth2_utility = sum(float(weights.get(name) or 0.0) * float(depth2_features.get(name) or 0.0) for name in feature_names)
        depth2_utility = float(depth2_utility - max(0.0, rollback_cost - (1.0 - rollback_success_rate)))
        entropy = self._candidate_score_entropy(candidates)
        row = {
            "branch_key": branch_key,
            "feature_names": feature_names,
            "features": features,
            "adaptive_weights": weights,
            "branch_utility": float(utility),
            "depth2_utility": float(depth2_utility),
            "prior_confidence": float(prior_confidence),
            "candidate_entropy": float(entropy),
            "session_branch_n": float(session_n),
            "session_branch_q": float(session_q),
            "session_rollback_success_rate": float(rollback_success_rate),
            "estimated_latency_ms": float(estimated_latency_ms),
            "latency_budget_ms": float(latency_budget_ms),
            "selection_policy": "session_local_adaptive_utility",
            "step": int(step_idx + 1),
        }
        candidate["session_branch_key"] = branch_key
        candidate["session_value_features"] = features
        candidate["session_adaptive_weights"] = weights
        candidate["session_branch_utility"] = float(utility)
        candidate["session_depth2_utility"] = float(depth2_utility)
        return row

    def _candidate_score_entropy(self, candidates: list[dict[str, Any]]) -> float:
        scores = [float(c.get("reasoning_prior_score") or c.get("search_strategy_score") or c.get("score") or 0.0) for c in candidates[:12]]
        if not scores:
            return 0.0
        shifted = [score - max(scores) for score in scores]
        exps = [math.exp(max(-20.0, value)) for value in shifted]
        denom = sum(exps) or 1.0
        probs = [value / denom for value in exps]
        return float(-sum(p * math.log(max(1e-9, p)) for p in probs))

    def _session_adaptive_budget(
        self,
        *,
        memory: dict[str, Any],
        configured_budget: int,
        min_attempts: int,
        positive_count: int,
        info_deficit: float,
        candidate_entropy: float,
        prior_confidence: float,
        task_mode: str,
        protected_passive: bool,
    ) -> int:
        if protected_passive or task_mode in {"FORM_CREATE_EDIT", "DELETE_COMMIT", "SIMPLE_VERIFY", "MEDIA_CAPTURE"}:
            return 0
        total_rollbacks = float(memory.get("rollback_total") or 0.0)
        rollback_success_rate = (
            float(memory.get("rollback_success") or 0.0) / total_rollbacks
            if total_rollbacks > 0
            else 1.0
        )
        value_pressure = max(0.0, min(1.0, 0.5 * float(info_deficit) + 0.5 * min(1.0, float(candidate_entropy))))
        confidence_focus = max(0.0, min(1.0, float(prior_confidence)))
        if confidence_focus >= 0.70 and candidate_entropy <= 0.70:
            desired = 1 + int(value_pressure > 0.45)
        else:
            desired = 1 + int(value_pressure > 0.25) + int(value_pressure > 0.55) + int(candidate_entropy > 1.20)
        if rollback_success_rate < 0.80:
            desired = max(1, desired - 1)
        desired = max(desired, min_attempts if value_pressure > 0.55 else min(1, min_attempts))
        return int(max(0, min(max(1, configured_budget), desired, max(1, positive_count))))

    def _update_session_exploration_memory(self, goal: str, root_activity: str, observation: dict[str, Any]) -> None:
        memory = self._session_memory_for(goal, root_activity)
        steps = [s for s in list(observation.get("steps") or []) if isinstance(s, dict)]
        rb = observation.get("rollback") if isinstance(observation.get("rollback"), dict) else {}
        rollback_success = bool(rb.get("success", True))
        memory["rollback_total"] = float(memory.get("rollback_total") or 0.0) + 1.0
        memory["rollback_success"] = float(memory.get("rollback_success") or 0.0) + (1.0 if rollback_success else 0.0)
        latency_ms = float(observation.get("branch_state_fetch_ms") or 0.0) + float(observation.get("branch_a11y_latency_ms") or 0.0)
        previous_latency = float(memory.get("latency_ewma_ms") or self._reasoning_prior_ewma_ms or 10000.0)
        if latency_ms > 0.0:
            memory["latency_ewma_ms"] = 0.7 * previous_latency + 0.3 * latency_ms
        evidence_type = _clean_text(observation.get("evidence_type") or "")
        confidence = self._clip01(observation.get("confidence"), 0.0)
        evidence_gain = self._clip01(float(observation.get("evidence_gain") or 0.0) / 5.0, 0.0)
        reward = 0.0
        if rollback_success:
            reward += confidence
            reward += evidence_gain
            if evidence_type in {"ANSWER_HINT", "ACTION_HINT"}:
                reward += 1.0
            elif evidence_type in {"SCHEMA_HINT", "AVOID_HINT", "RISK_HINT"}:
                reward += 0.5
        else:
            reward -= 1.0
        if not observation.get("observed_elements"):
            reward -= 0.25
        reward = max(-1.0, min(2.0, reward))
        branch_stats = memory.setdefault("branch_stats", {})
        for step in steps:
            cand = step.get("candidate") if isinstance(step.get("candidate"), dict) else {}
            branch_key = _clean_text(cand.get("session_branch_key") or self._branch_memory_key(cand, goal))
            if not branch_key:
                continue
            stats = branch_stats.setdefault(branch_key, {"n": 0.0, "q": 0.0, "rollback_fail": 0.0})
            n = float(stats.get("n") or 0.0)
            q = float(stats.get("q") or 0.0)
            stats["n"] = n + 1.0
            stats["q"] = ((q * n) + reward) / (n + 1.0)
            stats["rollback_fail"] = float(stats.get("rollback_fail") or 0.0) + (0.0 if rollback_success else 1.0)
            features = cand.get("session_value_features") if isinstance(cand.get("session_value_features"), dict) else {}
            if features:
                memory.setdefault("component_history", []).append(
                    {
                        "branch_key": branch_key,
                        "features": dict(features),
                        "reward": float(reward),
                        "evidence_type": evidence_type,
                        "rollback_success": bool(rollback_success),
                    }
                )
        memory.setdefault("reward_history", []).append(float(reward))
        memory["last_update"] = {
            "reward": float(reward),
            "evidence_type": evidence_type,
            "rollback_success": bool(rollback_success),
            "branch_count": int(len(steps)),
        }
        self._append_diagnostic_jsonl(
            goal,
            "session_adaptive_memory_updates.jsonl",
            {
                "task_id": _clean_text(goal)[:200],
                "session_key": memory.get("session_key"),
                "root_activity": _clean_text(root_activity),
                "branch_id": observation.get("branch_id"),
                "reward": float(reward),
                "evidence_type": evidence_type,
                "confidence": float(confidence),
                "rollback_success": bool(rollback_success),
                "latency_ms": float(latency_ms),
                "memory": {
                    "rollback_total": memory.get("rollback_total"),
                    "rollback_success": memory.get("rollback_success"),
                    "latency_ewma_ms": memory.get("latency_ewma_ms"),
                    "branch_stats_count": len(memory.get("branch_stats") or {}),
                    "component_history_count": len(memory.get("component_history") or []),
                },
            },
        )

    def _select_voc_branch_candidates(
        self,
        *,
        goal: str,
        step_idx: int,
        candidates: list[dict[str, Any]],
        root_state: Any,
        root_activity: str,
        page_stalled: bool,
        prior: dict[str, Any] | None,
        budget: int,
        trace: dict[str, Any],
    ) -> list[dict[str, Any]]:
        root_labels = self._state_semantic_summary(root_state, limit=80)
        phase, phase_reason = self._voc_task_phase(goal, step_idx, root_labels, page_stalled, prior)
        info_deficit, deficit_components = self._voc_information_deficit(goal, root_labels, phase)
        task_mode = self._task_mode(goal)
        memory = self._session_memory_for(goal, root_activity)
        prior_confidence = self._clip01((prior or {}).get("confidence"), 0.0)
        candidate_entropy = self._candidate_score_entropy(candidates)
        phase2, phase_reason2, flags = self._selective_task_phase(goal, step_idx, root_labels, page_stalled, prior)
        phase, phase_reason = phase2, phase_reason2
        gate = self._selective_exploration_gate(goal, task_mode, phase, page_stalled)
        self._append_selective_phase_gate_records(goal, step_idx, task_mode, phase, phase_reason, flags, gate)
        protected_passive = self._protected_task_passive_only(goal, prior or {}) or not bool(gate.get("active_allowed"))
        trace["selective_gate"] = dict(gate)
        trace["selective_task_mode"] = task_mode
        trace["selective_task_phase"] = phase
        value_rows: list[dict[str, Any]] = []
        frontier: list[dict[str, Any]] = []
        safe_rows: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for idx, candidate in enumerate(candidates):
            operator = _clean_text(candidate.get("operator") or self._candidate_operator(candidate, goal)[0] or "Other")
            grounding = self._voc_candidate_grounding(candidate, goal)
            consumable_operator = operator in {"SearchPeek", "FilterPeek", "DetailPeek", "StatsPeek", "ListInspect", "NavigationPeek", "FormSchema"}
            safety_rejected = bool(
                self._is_transaction_unsafe_candidate(candidate)
                or bool(candidate.get("risk_boundary"))
                or operator == "RiskBoundary"
                or not bool(gate.get("active_allowed"))
                or not consumable_operator
            )
            ambiguity, ambiguity_reasons = self._voc_candidate_ambiguity(candidate, candidates, goal)
            rollback_cost, rollback_cost_reasons = self._voc_estimated_rollback_cost(candidate, operator)
            score_components = candidate.get("score_components") if isinstance(candidate.get("score_components"), dict) else {}
            missing_slot_gain = self._voc_norm(score_components.get("MissingSlotGain"), 0.0)
            task_entity_match = self._voc_norm(score_components.get("TaskEntityMatch"), 0.0)
            prior_score = self._voc_norm(candidate.get("reasoning_prior_score") or score_components.get("PatternPrior"), 0.0)
            evidence_gain = min(1.0, max(0.0, float(candidate.get("estimated_evidence_gain") or 0.0) / 10.0))
            grounding_score = max(
                self._voc_norm(grounding.get("lexical_overlap"), 0.0),
                task_entity_match,
                self._voc_norm(candidate.get("relevance"), 0.0),
            )
            adaptive_value = self._session_adaptive_candidate_value(
                goal=goal,
                step_idx=step_idx,
                candidate=candidate,
                candidates=candidates,
                memory=memory,
                phase=phase,
                info_deficit=info_deficit,
                grounding=grounding,
                ambiguity=ambiguity,
                rollback_cost=rollback_cost,
                score_components=score_components,
                prior=prior,
            )
            information_value = float(adaptive_value.get("branch_utility") or 0.0)
            latency_cost = 1.0 - self._clip01((adaptive_value.get("features") or {}).get("latency_fit"), 0.0)
            expected_value = float(adaptive_value.get("branch_utility") or 0.0)
            depth2_ev = float(adaptive_value.get("depth2_utility") or 0.0)
            depth2_allowed = bool(
                depth2_ev > 0.0
                and info_deficit >= 0.35
                and rollback_cost <= 0.60
                and task_mode not in {"FORM_CREATE_EDIT", "DELETE_COMMIT", "SIMPLE_VERIFY", "MEDIA_CAPTURE"}
                and phase in {"TARGET_ACQUISITION", "EVIDENCE_GATHERING", "STUCK_RECOVERY"}
            )
            row = {
                "record_type": "VOCFrontierValue",
                "step": int(step_idx + 1),
                "candidate_rank_input": int(idx + 1),
                "task_id": _clean_text(goal)[:200],
                "task_mode": task_mode,
                "task_phase": phase,
                "phase_reason": phase_reason,
                "information_deficit": float(info_deficit),
                "deficit_components": deficit_components,
                "candidate_label": grounding.get("label"),
                "operator": operator,
                "grounding": grounding,
                "candidate_ambiguity": float(ambiguity),
                "ambiguity_reasons": ambiguity_reasons,
                "rollback_cost": float(rollback_cost),
                "rollback_cost_reasons": rollback_cost_reasons,
                "missing_slot_gain": float(missing_slot_gain),
                "task_entity_match": float(task_entity_match),
                "prior_score": float(prior_score),
                "evidence_gain": float(evidence_gain),
                "grounding_score": float(grounding_score),
                "phase_weight": 1.0,
                "information_value": float(information_value),
                "latency_cost": float(latency_cost),
                "expected_value": float(expected_value),
                "branch_utility": float(expected_value),
                "depth2_expected_value": float(depth2_ev),
                "depth2_utility": float(depth2_ev),
                "depth2_allowed": bool(depth2_allowed),
                "safety_rejected": bool(safety_rejected),
                "rejected_reason": ("exploration_gate_or_safety_reject" if safety_rejected else ""),
                "session_key": memory.get("session_key"),
                "session_branch_key": adaptive_value.get("branch_key"),
                "session_value_features": adaptive_value.get("features"),
                "session_adaptive_weights": adaptive_value.get("adaptive_weights"),
                "prior_confidence": float(prior_confidence),
                "candidate_entropy": float(candidate_entropy),
                "selection_policy": adaptive_value.get("selection_policy"),
            }
            value_rows.append(row)
            if safety_rejected:
                candidate["reason_rejected"] = "safety_policy_reject"
                continue
            candidate["voc_expected_value"] = float(expected_value)
            candidate["voc_information_value"] = float(information_value)
            candidate["voc_latency_cost"] = float(latency_cost)
            candidate["voc_rollback_cost"] = float(rollback_cost)
            candidate["voc_candidate_ambiguity"] = float(ambiguity)
            candidate["voc_task_phase"] = phase
            candidate["voc_depth2_decision"] = {
                "allow": bool(depth2_allowed),
                "expected_value": float(depth2_ev),
                "reason": "positive_depth2_value" if depth2_allowed else "depth2_value_or_safety_below_threshold",
            }
            safe_rows.append((float(expected_value), candidate, row))
        safe_rows.sort(key=lambda item: item[0], reverse=True)
        positive_count = sum(1 for value, _, _ in safe_rows if value > 0.0)
        configured_budget = max(1, int(budget))
        min_attempts = max(0, int(getattr(self, "light_explore_min_attempts_per_step", 0) or 0))
        adaptive_budget = self._session_adaptive_budget(
            memory=memory,
            configured_budget=configured_budget,
            min_attempts=min_attempts,
            positive_count=positive_count,
            info_deficit=info_deficit,
            candidate_entropy=candidate_entropy,
            prior_confidence=prior_confidence,
            task_mode=task_mode,
            protected_passive=protected_passive,
        )
        selected = [candidate for _, candidate, _ in safe_rows[:adaptive_budget]]
        selected_keys = {_clean_text(c.get("key") or c.get("label") or c.get("merged")) for c in selected}
        for rank, candidate in enumerate(selected, start=1):
            candidate["reason_selected"] = "selective_consumable_expected_value"
            candidate["voc_selected_rank"] = int(rank)
        for candidate in candidates:
            key = _clean_text(candidate.get("key") or candidate.get("label") or candidate.get("merged"))
            if key not in selected_keys and not _clean_text(candidate.get("reason_rejected")):
                candidate["reason_rejected"] = "not_selected_after_voc_budget"
        depth2_allowed_count = sum(1 for c in selected if bool((c.get("voc_depth2_decision") or {}).get("allow")))
        active_depth_budget = 2 if depth2_allowed_count > 0 else 1
        self._active_voc_depth_budget = int(active_depth_budget)
        budget_record = {
            "record_type": "VOCAdaptiveBudgetDecision",
            "step": int(step_idx + 1),
            "task_id": _clean_text(goal)[:200],
            "task_phase": phase,
            "phase_reason": phase_reason,
            "information_deficit": float(info_deficit),
            "configured_root_budget": int(configured_budget),
            "adaptive_root_budget": int(adaptive_budget),
            "positive_value_candidate_count": int(positive_count),
            "selected_candidate_count": int(len(selected)),
            "active_depth_budget": int(active_depth_budget),
            "depth2_allowed_count": int(depth2_allowed_count),
            "candidate_entropy": float(candidate_entropy),
            "prior_confidence": float(prior_confidence),
            "session_key": memory.get("session_key"),
            "session_rollback_total": float(memory.get("rollback_total") or 0.0),
            "session_rollback_success": float(memory.get("rollback_success") or 0.0),
            "session_memory_branch_count": int(len(memory.get("branch_stats") or {})),
            "selection_policy": "session_local_adaptive_utility",
            "root_activity": _clean_text(root_activity),
            "safety_rejected_count": int(sum(1 for row in value_rows if row["safety_rejected"])),
        }
        trace["search_strategy_name"] = "Selective Consumable Exploration"
        trace["search_strategy_cn"] = "可消费证据优先的选择性探索"
        trace["root_level_strategy"] = "only explore branches consumable by next prompt or promotion"
        trace["branch_continuation"] = "depth2 only for search/filter/result/detail/schema consumable states"
        trace["ranking"] = "consumable evidence/promotion gate before utility"
        trace["voc_task_phase"] = phase
        trace["voc_phase_reason"] = phase_reason
        trace["voc_information_deficit"] = float(info_deficit)
        trace["voc_deficit_components"] = deficit_components
        trace["voc_adaptive_root_budget"] = int(adaptive_budget)
        trace["voc_active_depth_budget"] = int(active_depth_budget)
        trace["voc_depth2_allowed_count"] = int(depth2_allowed_count)
        trace["session_adaptive_candidate_entropy"] = float(candidate_entropy)
        trace["session_adaptive_prior_confidence"] = float(prior_confidence)
        trace["session_adaptive_memory_key"] = memory.get("session_key")
        trace["voc_frontier_values"] = value_rows[:80]
        trace["frontier_values"] = value_rows[:80]
        trace["adaptive_budget_decision"] = budget_record
        for row in value_rows:
            self._append_diagnostic_jsonl(goal, "voc_candidate_values.jsonl", row)
            self._append_diagnostic_jsonl(goal, "frontier_values.jsonl", row)
        self._append_diagnostic_jsonl(goal, "adaptive_budget_decisions.jsonl", budget_record)
        return selected

    def _run_light_exploration(
        self,
        goal: str,
        step_idx: int,
        current_action: json_action.JSONAction | None = None,
        page_stalled: bool = False,
        root_state: Any | None = None,
        root_activity: str = "",
        root_hash: int | None = None,
        planning_text: str = "",
        reasoning_prior_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._last_probe_candidates = []
        self._current_goal_for_depth = goal
        self._active_voc_depth_budget = None
        if self.light_explore_decouple_planned and self.exploration_timing != "invalid_post_planned_action":
            if not getattr(self, "light_explore_use_current_action", False):
                current_action = None
            if not getattr(self, "light_explore_use_planning_text", False):
                planning_text = ""
        trace: dict[str, Any] = {
            "step": int(step_idx + 1),
            "goal": goal,
            "mode": "parallel_independent_for_next_step"
            if self.light_explore_decouple_planned
            else "post_planning_for_next_step",
            "planned_action": current_action.__dict__ if current_action is not None else None,
            "decoupled_from_planned_action": bool(self.light_explore_decouple_planned),
            "uses_current_planned_action": bool(current_action is not None),
            "uses_current_vlm_text": bool(planning_text),
            "evidence_injected_same_step": False,
            "invalid_run": bool(
                getattr(self, "exploration_timing", "") == "invalid_post_planned_action"
                or (
                    bool(getattr(self, "decoupled_exploration", True))
                    and (current_action is not None or bool(planning_text))
                )
            ),
            "exploration_timing": getattr(self, "exploration_timing", "parallel_shadow"),
            "exploration_timing_mode": getattr(self, "exploration_timing", "parallel_shadow"),
            "parallel_with_vlm": bool(self.light_explore_parallel_vlm),
            "shadow_only": bool(getattr(self, "light_explore_shadow_only", False)),
            "search_strategy_name": "Selective Consumable Exploration",
            "strategy": self.light_explore_strategy,
            "search_strategy": self.light_explore_search_strategy,
            "depth_budget": int(self.light_explore_branch_depth),
            "effective_depth_budget": int(
                self._search_strategy_depth_limit(self._task_mode(goal))
            ),
            "status": "not_started",
            "trigger_reason": "",
            "root_activity": "",
            "root_hash": None,
            "root_screenshot": "",
            "root_a11y": [],
            "root_structural_summary": [],
            "root_state_fetch_ms": 0.0,
            "root_a11y_dump_ms": 0.0,
            "root_a11y_trace_ms": 0.0,
            "root_summary_ms": 0.0,
            "root_structural_summary_ms": 0.0,
            "lightweight_a11y_trace": bool(self.light_explore_lightweight_a11y_trace),
            "candidate_count": 0,
            "candidates": [],
            "selected_targets": [],
            "observations": [],
            "evidence_capsules": [],
            "rollbacks": [],
            "selected_prompt_results": [],
            "prompt_context": "",
            "speculative_results": [],
            "speculative_context": "",
            "available_for_next_prompt": False,
            "fast_mode": bool(self.light_explore_fast_mode),
            "fast_state": bool(self.light_explore_fast_state),
            "action_settle_s": float(self.light_explore_action_settle_s),
            "save_screenshots": bool(self.light_explore_save_screenshots),
            "transaction_safe": bool(self.light_explore_transaction_safe),
            "variant": self.light_explore_variant,
            "hint_policy": self.light_explore_hint_policy,
            "search_policy": self.light_explore_search_policy,
            "rollback_policy": self.light_explore_rollback_policy,
            "task_mode": self._task_mode(goal),
            "task_slots": slot_complete_evidence.parse_task_slots(goal).to_dict()
            if self.light_explore_slot_complete
            else {},
            "slot_complete": bool(self.light_explore_slot_complete),
            "started_at": time.time(),
            "latency_ms": 0.0,
            "t2_branch_debug": [],
            "rooted_depth1_only_count": 0,
            "rooted_depth2_count": 0,
            "rooted_depth2_rate": 0.0,
            "t2_candidate_count": 0,
            "searchinput_t2_candidate_count": 0,
            "depth2_blocked_by_hash_unchanged_count": 0,
            "depth2_blocked_by_semantic_unchanged_count": 0,
            "depth2_blocked_by_no_typed_candidate_count": 0,
            "depth2_blocked_by_safety_count": 0,
            "depth2_blocked_by_should_expand_count": 0,
        }
        previous_state_acquisition_context = self._state_acquisition_context
        try:
            reasoning_prior = (
                dict(reasoning_prior_snapshot)
                if isinstance(reasoning_prior_snapshot, dict)
                else self._delayed_reasoning_prior_for_exploration(goal, step_idx)
            )
            protected_passive_only = self._protected_task_passive_only(goal, reasoning_prior)
            trace["reasoning_prior_enabled"] = bool(getattr(self, "reasoning_prior_enabled", False))
            trace["reasoning_prior_delayed"] = bool(getattr(self, "reasoning_prior_delayed", True))
            trace["reasoning_prior_available"] = bool(reasoning_prior)
            trace["reasoning_prior_source_step"] = int(reasoning_prior.get("source_step") or 0) if reasoning_prior else 0
            trace["protected_passive_only"] = bool(protected_passive_only)
            if getattr(self, "reasoning_prior_delayed", True) and step_idx <= 0:
                trace["status"] = "skipped"
                trace["trigger_reason"] = "reasoning_prior_delayed_step1_no_exploration"
                return trace
            if getattr(self, "reasoning_prior_enabled", False) and getattr(self, "reasoning_prior_delayed", True) and not reasoning_prior:
                trace["status"] = "skipped"
                trace["trigger_reason"] = "reasoning_prior_delayed_no_previous_prior"
                return trace
            action_skip_reason = (
                ""
                if current_action is None or self.light_explore_decouple_planned
                else self._planned_action_skip_reason(current_action)
            )
            planned_skip_reason = ""
            if action_skip_reason and not self.light_explore_fallback_safe_candidates:
                trace["status"] = "skipped"
                trace["trigger_reason"] = action_skip_reason
                return trace
            if action_skip_reason:
                planned_skip_reason = action_skip_reason

            if root_state is None:
                root_fetch_start = time.time()
                root_state = self._get_probe_state(wait_to_stabilize=True)
                trace["root_state_fetch_ms"] = float(max(0.0, time.time() - root_fetch_start) * 1000.0)
            if root_hash is None:
                root_hash = self._state_hash(root_state)
            if not root_activity:
                root_activity = self._foreground_activity_name()
            trace["root_activity"] = root_activity
            trace["root_hash"] = root_hash
            trace["root_a11y_dump_ms"] = self._state_a11y_latency_ms(root_state)
            trace["root_state_aux"] = self._state_aux_trace(root_state)
            if self._has_keyboard_ui(root_state) and not self.light_explore_diagnostic_full:
                trace["status"] = "skipped"
                trace["trigger_reason"] = "keyboard_visible"
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    "light_explore_skipped=keyboard_visible"
                )
                return trace
            should_run, reason = self._should_run_light_exploration(
                goal=goal,
                step_idx=step_idx,
                root_activity=root_activity,
                page_stalled=page_stalled,
                root_state=root_state,
                current_action=current_action,
            )
            trace["trigger_reason"] = str(reason)
            trace["gate_enabled"] = bool(should_run)
            trace["gate_reason"] = str(reason)
            trace["target_visible"] = self._target_visible_in_state(root_state, goal)
            trace["screen_complexity"] = self._screen_complexity(root_state)
            trace["current_stage"] = (
                "early" if step_idx <= 1 else "middle" if step_idx < self._effective_max_steps() - 2 else "late"
            )
            print(
                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                f"light_explore_should_run={should_run} reason={reason} page_stalled={page_stalled}"
            )
            if not should_run:
                skip_observed_start = time.time()
                skip_observed_elements = self._state_semantic_summary(root_state, limit=TRACE_OBSERVED_ELEMENT_LIMIT)
                trace["root_summary_ms"] = float(max(0.0, time.time() - skip_observed_start) * 1000.0)
                skip_task_mode = self._task_mode(goal)
                skip_phase, skip_phase_reason, skip_flags = self._selective_task_phase(
                    goal,
                    step_idx,
                    skip_observed_elements,
                    page_stalled,
                    reasoning_prior,
                )
                skip_gate = self._selective_exploration_gate(goal, skip_task_mode, skip_phase, page_stalled)
                trace["selective_task_mode"] = skip_task_mode
                trace["selective_task_phase"] = skip_phase
                trace["selective_phase_reason"] = skip_phase_reason
                trace["selective_gate"] = dict(skip_gate)
                self._append_selective_phase_gate_records(
                    goal,
                    step_idx,
                    skip_task_mode,
                    skip_phase,
                    skip_phase_reason,
                    skip_flags,
                    skip_gate,
                )
                passive_skip_allowed = bool(
                    step_idx > 0
                    and bool(skip_gate.get("passive_allowed"))
                    and skip_task_mode in {"FORM_CREATE_EDIT", "DELETE_COMMIT", "SIMPLE_VERIFY", "MEDIA_CAPTURE"}
                    and "launcher" not in _clean_text(reason).lower()
                    and "keyboard" not in _clean_text(reason).lower()
                )
                if passive_skip_allowed:
                    passive_pair = self._passive_schema_observation(
                        goal=goal,
                        strategy_id=self.light_explore_search_strategy,
                        step_id=step_idx + 1,
                        branch_id=1,
                        root_state=root_state,
                        root_activity=root_activity,
                        root_hash=int(root_hash or -1),
                        root_screenshot=str(trace.get("root_screenshot") or ""),
                        observed_elements=list(skip_observed_elements),
                        task_mode=skip_task_mode,
                        reason=f"passive_schema_on_skipped_exploration:{reason}",
                    )
                    produced_schema = bool(passive_pair)
                    self._append_diagnostic_jsonl(goal, "passive_observation_records.jsonl", {
                        "step": int(step_idx + 1),
                        "observed_labels": list(skip_observed_elements)[:80],
                        "produced_evidence": produced_schema,
                        "injected": False,
                        "reason": (
                            f"passive_schema_on_skipped_exploration:{reason}"
                            if produced_schema
                            else f"skipped_without_schema:{reason}"
                        ),
                    })
                    if passive_pair:
                        schema_observation, schema_capsule = passive_pair
                        trace["observations"] = [schema_observation]
                        trace["evidence_capsules"] = [schema_capsule]
                        trace["speculative_results"] = []
                        trace["speculative_context"] = ""
                        trace["selected_prompt_results"] = []
                        trace["prompt_context"] = ""
                        trace["available_for_next_prompt"] = True
                        trace["rollback_success"] = True
                        trace["passive_only_reason"] = f"should_run_false_but_schema_consumable:{reason}"
                        trace["status"] = "passive_schema_only"
                        self._pending_speculative_traces = [trace]
                        print(
                            f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                            "light_explore_passive_schema_cached_on_skip"
                        )
                        return trace
                trace["status"] = "skipped"
                return trace

            trace["root_screenshot"] = self._save_trace_screenshot(
                goal, root_state, f"explore_step_{step_idx + 1:02d}_root.png"
            )
            a11y_trace_start = time.time()
            trace["root_a11y"] = self._state_a11y_trace(root_state)
            trace["root_a11y_trace_ms"] = float(max(0.0, time.time() - a11y_trace_start) * 1000.0)
            structural_start = time.time()
            trace["root_structural_summary"] = self._state_structural_summary(root_state)
            trace["root_structural_summary_ms"] = float(max(0.0, time.time() - structural_start) * 1000.0)

            candidate_collection_start = time.time()
            candidates = self._collect_probe_candidates(root_state, goal, planning_text=planning_text)
            trace["candidate_collection_ms"] = float(max(0.0, time.time() - candidate_collection_start) * 1000.0)
            trace["candidate_quality"] = dict(getattr(self, "_last_candidate_quality_stats", {}) or {})
            current_screen_role = self._screen_role_from_state(root_state)
            candidates, prior_entropy, prior_mode, prior_mode_record = self._evaluate_candidate_priors(
                candidates,
                goal=goal,
                reasoning_prior=reasoning_prior,
                current_screen_role=current_screen_role,
                app_name=root_activity,
                step_idx=step_idx + 1,
                trace_step=trace,
            )
            trace["reasoning_prior_entropy"] = float(prior_entropy)
            trace["reasoning_prior_mode"] = prior_mode
            trace["reasoning_prior_root_budget"] = int(prior_mode_record.get("root_budget") or self.light_explore_branch_budget)
            trace["reasoning_prior_depth2_policy"] = _clean_text(prior_mode_record.get("depth2_policy") or "semantic_changed_only")
            self._append_adaptive_search_record(
                goal,
                {
                    "step": int(step_idx + 1),
                    "prior_source_step": int(reasoning_prior.get("source_step") or 0),
                    "mode": prior_mode,
                    "prior_entropy": float(prior_entropy),
                    "prior_confidence": float(reasoning_prior.get("confidence") or 0.0),
                    "root_budget": int(trace["reasoning_prior_root_budget"]),
                    "depth2_policy": trace["reasoning_prior_depth2_policy"],
                    "candidate_count": int(len(candidates)),
                    "protected_passive_only": bool(protected_passive_only),
                },
            )
            candidates, protected_candidate_policy = self._apply_protected_task_candidate_policy(
                candidates,
                goal=goal,
                prior=reasoning_prior,
            )
            trace["protected_candidate_policy"] = dict(protected_candidate_policy)
            candidates = self._rank_candidates_for_planned_action(candidates, current_action)
            planned_candidate = (
                None
                if self.light_explore_decouple_planned
                else self._planned_probe_candidate_from_action(root_state, current_action)
            )
            if planned_candidate is not None and not planned_skip_reason:
                candidate_skip_reason = ""
                if self._is_transaction_unsafe_candidate(planned_candidate):
                    candidate_skip_reason = "planned_candidate_transaction_unsafe"
                planned_action_kind = _clean_text(planned_candidate.get("action_kind") or "")
                if (
                    not candidate_skip_reason
                    and
                    self.light_explore_require_a11y_anchor
                    and planned_action_kind == json_action.CLICK
                    and int(planned_candidate.get("index") or -1) < 0
                ):
                    candidate_skip_reason = "no_a11y_planned_anchor"
                if (
                    not candidate_skip_reason
                    and
                    self.light_explore_skip_risky_planned
                    and self._is_risky_probe_text(str(planned_candidate.get("merged") or planned_candidate.get("label") or ""))
                ):
                    candidate_skip_reason = "risky_planned_anchor"
                if (
                    not candidate_skip_reason
                    and self._is_commit_probe_label(str(planned_candidate.get("label") or ""))
                    and not self.light_explore_diagnostic_full
                ):
                    candidate_skip_reason = "commit_planned_anchor"
                planned_key = _clean_text(planned_candidate.get("key"))
                if (
                    not candidate_skip_reason
                    and planned_key
                    and planned_key in self._no_effect_probe_keys
                    and not self.light_explore_diagnostic_full
                ):
                    candidate_skip_reason = "planned_anchor_previous_no_effect"
                if candidate_skip_reason:
                    planned_skip_reason = candidate_skip_reason
                    trace["skipped_planned_target"] = self._candidate_trace(planned_candidate)
                else:
                    planned_center = planned_candidate.get("center")
                    duplicate_planned = False
                    if isinstance(planned_center, (list, tuple)) and len(planned_center) >= 2:
                        for candidate in candidates:
                            candidate_center = candidate.get("center")
                            if isinstance(candidate_center, (list, tuple)) and len(candidate_center) >= 2:
                                if math.dist(planned_center, candidate_center) <= 80:
                                    candidate["is_planned_action"] = True
                                    candidate["score"] = max(float(candidate.get("score") or 0.0), 10.0)
                                    candidate["label"] = f"planned: {_clean_text(candidate.get('label')) or 'click'}"
                                    candidate["key"] = planned_candidate.get("key") or candidate.get("key")
                                    candidate["merged"] = planned_candidate.get("merged") or candidate.get("merged")
                                    duplicate_planned = True
                                    break
                    if not duplicate_planned:
                        candidates.insert(0, planned_candidate)
                    candidates.sort(
                        key=lambda c: (
                            1 if bool(c.get("is_planned_action")) else 0,
                            float(c.get("score") or 0.0),
                        ),
                        reverse=True,
                    )
            elif planned_candidate is not None:
                trace["skipped_planned_target"] = self._candidate_trace(planned_candidate)
            if self.light_explore_planned_only:
                planned_candidates = [c for c in candidates if bool(c.get("is_planned_action"))]
                if not planned_candidates:
                    fallback_candidates = (
                        self._safe_fallback_probe_candidates(candidates, planned_candidate=planned_candidate)
                        if self.light_explore_fallback_safe_candidates
                        else []
                    )
                    if not fallback_candidates:
                        trace["status"] = (
                            "skipped_transaction_unsafe"
                            if (planned_skip_reason or "").endswith("transaction_unsafe")
                            else "skipped"
                        )
                        trace["trigger_reason"] = planned_skip_reason or "no_planned_anchor"
                        if planned_candidate is not None:
                            trace["selected_targets"] = [self._candidate_trace(planned_candidate)]
                        return trace
                    fallback_reason = planned_skip_reason or "no_planned_anchor"
                    trace["planned_only_fallback"] = True
                    trace["planned_candidate_skip_reason"] = fallback_reason
                    trace["trigger_reason"] = f"fallback_safe_candidates_after_{fallback_reason}"
                    candidates = fallback_candidates
                else:
                    candidates = planned_candidates
            for candidate in candidates:
                operator, operator_reason = self._candidate_operator(candidate, goal)
                candidate["operator"] = operator
                candidate["operator_reason"] = operator_reason
                candidate["risk_boundary"] = operator == "RiskBoundary"
            candidate_scoring_start = time.time()
            candidates = self._prepare_search_strategy_candidates(candidates, goal)
            if protected_candidate_policy.get("applied"):
                trace["protected_task_mode"] = self._task_mode(goal)
                trace["protected_passive_only"] = bool(protected_passive_only)
                trace["protected_mode_before_count"] = int(protected_candidate_policy.get("before_count", 0))
                trace["protected_mode_after_count"] = int(protected_candidate_policy.get("after_count", 0))
                trace["protected_mode_filtered_count"] = int(protected_candidate_policy.get("filtered_count", 0))
                trace["protected_mode_reason"] = _clean_text(protected_candidate_policy.get("mode") or "")
            trace["candidate_scoring_ms"] = float(max(0.0, time.time() - candidate_scoring_start) * 1000.0)
            self._last_probe_candidates = list(candidates)
            trace["candidate_count"] = int(len(candidates))
            if getattr(self, "light_explore_shadow_only", False):
                trace["status"] = "completed_shadow_only"
                trace["trigger_reason"] = f"{trace.get('trigger_reason') or 'shadow_only'}"
                trace["candidates"] = [self._candidate_trace(c) for c in candidates]
                trace["selected_targets"] = []
                trace["observations"] = []
                trace["available_for_next_prompt"] = False
                trace["latency_ms"] = float(max(0.0, time.time() - trace["started_at"]) * 1000.0)
                return trace
            trace["candidates"] = [self._candidate_trace(c) for c in candidates]
            trace["search_strategy_name"] = "Operator-Stratified Best-First Exploration"
            trace["root_level_strategy"] = "stratified breadth-first across operator groups"
            trace["branch_continuation"] = "gated depth-first continuation to depth 2"
            trace["ranking"] = "best-first score"
            trace["min_attempts_per_step"] = int(getattr(self, "light_explore_min_attempts_per_step", 0))
            if not candidates:
                trace["status"] = "no_candidates"
                print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_no_candidates")
                return trace
            if (
                self._is_launcher_activity(root_activity)
                and self.light_explore_filter_launcher_relevance
                and not self.light_explore_diagnostic_full
            ):
                relevant_candidates = [
                    c
                    for c in candidates
                    if bool(c.get("is_planned_action"))
                    or float(c.get("relevance") or 0.0) >= float(self.light_explore_min_launcher_relevance)
                ]
                if not relevant_candidates:
                    trace["status"] = "no_task_relevant_launcher_candidates"
                    print(
                        f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                        "light_explore_launcher_skipped=no_task_relevant_candidates"
                    )
                    return trace
                candidates = relevant_candidates
                trace["candidate_count_after_launcher_relevance_filter"] = int(len(candidates))
            replay_actions = self._select_replay_actions_for_probe(current_action=None)
            budget = max(1, int(trace.get("reasoning_prior_root_budget") or self.light_explore_branch_budget))
            if self.light_explore_search_strategy == "value_of_computation":
                branch_candidates = self._select_voc_branch_candidates(
                    goal=goal,
                    step_idx=step_idx,
                    candidates=candidates,
                    root_state=root_state,
                    root_activity=root_activity,
                    page_stalled=page_stalled,
                    prior=reasoning_prior,
                    budget=budget,
                    trace=trace,
                )
            elif bool(getattr(self, "light_explore_lb_mcts", False)) and self.light_explore_search_strategy == "mcts":
                self._record_lb_mcts_candidate_filter_stats(
                    goal=goal,
                    step=step_idx + 1,
                    raw_count=int(len(candidates)),
                    candidates=candidates,
                )
                branch_candidates = self._select_lb_mcts_branch_candidates(
                    goal=goal,
                    step_idx=step_idx,
                    candidates=candidates,
                    root_state=root_state,
                    root_activity=root_activity,
                    root_labels=self._state_semantic_summary(root_state, limit=24),
                    trace=trace,
                )
                trace["search_strategy_name"] = "Latency-Bounded Task-Aware MCTS Exploration"
                trace["root_level_strategy"] = "latency-bounded UCT over rollback-safe UI actions"
                trace["branch_continuation"] = "depth-prioritized gated DFS to depth 2"
                trace["ranking"] = "latency-bounded UCT"
            else:
                branch_candidates = []
                grouped: dict[str, list[dict[str, Any]]] = {}
                for candidate in candidates:
                    grouped.setdefault(_clean_text(candidate.get("operator") or "Other"), []).append(candidate)
                for group in grouped.values():
                    group.sort(
                        key=lambda c: (
                            float(c.get("search_strategy_score") or c.get("final_score") or c.get("score") or 0.0),
                            float(c.get("score") or 0.0),
                        ),
                        reverse=True,
                    )
                operator_order = sorted(
                    grouped,
                    key=lambda op: float(
                        grouped[op][0].get("search_strategy_score")
                        or grouped[op][0].get("final_score")
                        or grouped[op][0].get("score")
                        or 0.0
                    ),
                    reverse=True,
                )
                seen_centers: set[tuple[int, int]] = set()
                while len(branch_candidates) < budget and operator_order:
                    made_progress = False
                    for operator in list(operator_order):
                        group = grouped.get(operator) or []
                        while group:
                            chosen = group.pop(0)
                            center = chosen.get("center")
                            center_key = tuple(center) if isinstance(center, (list, tuple)) and len(center) >= 2 else (id(chosen), 0)
                            if center_key in seen_centers:
                                continue
                            seen_centers.add(center_key)
                            chosen["reason_selected"] = "stratified_bfs_operator_best"
                            branch_candidates.append(chosen)
                            made_progress = True
                            break
                        if len(branch_candidates) >= budget:
                            break
                        if not group and operator in operator_order:
                            operator_order.remove(operator)
                    if not made_progress:
                        break
                selected_keys = {_clean_text(c.get("key") or c.get("label") or c.get("merged")) for c in branch_candidates}
                for candidate in candidates:
                    key = _clean_text(candidate.get("key") or candidate.get("label") or candidate.get("merged"))
                    if key not in selected_keys:
                        candidate["reason_rejected"] = "not_selected_after_operator_stratification_or_budget"
            trace["selected_targets"] = [self._candidate_trace(c) for c in branch_candidates]
            trace["attempt_count"] = int(
                len(branch_candidates)
                + len([c for c in candidates if bool(c.get("risk_boundary")) or self._is_transaction_unsafe_candidate(c)])
            )
            trace["safe_executable_count"] = int(
                len([c for c in candidates if not bool(c.get("risk_boundary")) and not self._is_transaction_unsafe_candidate(c)])
            )
            trace["risk_candidate_count"] = int(
                len([c for c in candidates if bool(c.get("risk_boundary")) or self._is_transaction_unsafe_candidate(c)])
            )
            if (
                int(getattr(self, "light_explore_min_attempts_per_step", 0)) > 0
                and int(trace["attempt_count"]) < int(getattr(self, "light_explore_min_attempts_per_step", 0))
            ):
                trace["no_candidate_reason"] = "fewer_than_min_attempts_due_to_safe_candidate_shortage_or_budget"

            attempted_any = False
            all_rollback_success = True
            observations: list[dict[str, Any]] = []
            evidence_capsules: list[dict[str, Any]] = []
            answer_label_limit = TRACE_A11Y_LIMIT if self.light_explore_answer_extractors else TRACE_OBSERVED_ELEMENT_LIMIT
            root_summary_start = time.time()
            root_observed_elements = self._state_semantic_summary(root_state, limit=answer_label_limit)
            trace["root_summary_ms"] = float(max(0.0, time.time() - root_summary_start) * 1000.0)
            if self.light_explore_search_strategy == "value_of_computation" and "voc_task_phase" not in trace:
                phase, phase_reason = self._voc_task_phase(goal, step_idx, root_observed_elements, page_stalled, reasoning_prior)
                info_deficit, deficit_components = self._voc_information_deficit(goal, root_observed_elements, phase)
                trace["voc_task_phase"] = phase
                trace["voc_phase_reason"] = phase_reason
                trace["voc_information_deficit"] = float(info_deficit)
                trace["voc_deficit_components"] = deficit_components
            task_mode = self._task_mode(goal)
            selective_phase, selective_phase_reason, selective_flags = self._selective_task_phase(goal, step_idx, root_observed_elements, page_stalled, reasoning_prior)
            selective_gate = self._selective_exploration_gate(goal, task_mode, selective_phase, page_stalled)
            trace["selective_task_mode"] = task_mode
            trace["selective_task_phase"] = selective_phase
            trace["selective_phase_reason"] = selective_phase_reason
            trace["selective_gate"] = dict(selective_gate)
            self._append_selective_phase_gate_records(goal, step_idx, task_mode, selective_phase, selective_phase_reason, selective_flags, selective_gate)
            if protected_passive_only:
                branch_candidates = []
                trace["selected_targets"] = []
                trace["passive_attempt_count"] = 0
                trace["executed_attempt_count"] = 0
                trace["passive_only_reason"] = f"protected_or_selective_gate:{task_mode}:{(trace.get('selective_gate') or {}).get('gate_reason', '')}"
                passive_schema_flags = self._selective_visible_flags(goal, root_observed_elements, page_stalled)
                passive_schema_labels = self._task_derived_schema_labels(goal, task_mode, root_observed_elements)
                produced_schema = bool(passive_schema_flags.get("schema_visible") or passive_schema_labels)
                self._append_diagnostic_jsonl(goal, "passive_observation_records.jsonl", {
                    "step": int(step_idx + 1),
                    "observed_labels": list(root_observed_elements)[:80],
                    "produced_evidence": produced_schema,
                    "injected": False,
                    "reason": trace["passive_only_reason"] if not produced_schema else "passive_schema_hint_visible",
                })
                if produced_schema:
                    pseudo_candidate = {
                        "label": "Visible screen schema",
                        "merged": "Visible screen schema",
                        "operator": "FormSchema",
                        "action_kind": "inspect",
                        "risk_boundary": False,
                        "layout_region": "root_page",
                        "semantic_role": "schema_observation",
                    }
                    schema_assessment = {
                        "evidence_gain": 1.0,
                        "confidence": 0.70,
                        "evidence_type": "SCHEMA_HINT",
                        "boundary_type": "SCHEMA_BOUNDARY",
                        "stop_reason": "passive_schema_visible",
                        "schema_labels": list(dict.fromkeys(passive_schema_labels + list(root_observed_elements)))[:32],
                        "new_labels": list(passive_schema_labels)[:16],
                        "depth": 1,
                    }
                    schema_rollback = {"success": True, "mode": "passive_no_action", "level": "level0"}
                    schema_capsule = self._build_evidence_capsule(
                        goal=goal,
                        strategy_id=self.light_explore_search_strategy,
                        step_id=step_idx + 1,
                        branch_id=len(observations) + 1,
                        depth=1,
                        parent_state=root_state,
                        parent_activity=root_activity,
                        child_state=root_state,
                        child_activity=root_activity,
                        candidate=pseudo_candidate,
                        assessment=schema_assessment,
                        rollback_info=schema_rollback,
                    )
                    schema_step = {
                        "depth": 1,
                        "candidate": self._candidate_trace(pseudo_candidate),
                        "changed": False,
                        "semantic_changed": False,
                        "after_activity": root_activity,
                        "after_hash": root_hash,
                        "hash_diff_from_root": 0,
                        "screenshot": trace.get("root_screenshot") or "",
                        "observed_elements": list(dict.fromkeys(passive_schema_labels + list(root_observed_elements)))[:96],
                        "a11y": [],
                        "stop_reason": "passive_schema_visible",
                    }
                    schema_observation = {
                        "branch_id": len(observations) + 1,
                        "labels": ["Visible screen schema"],
                        "changed": False,
                        "after_activity": root_activity,
                        "score": 1.0,
                        "depth_reached": 1,
                        "observed_elements": list(dict.fromkeys(passive_schema_labels + list(root_observed_elements)))[:96],
                        "steps": [schema_step],
                        "rollback": schema_rollback,
                        "operator": "FormSchema",
                        "operator_reason": "passive_schema_visible",
                        "layout_region": "root_page",
                        "semantic_role": "schema_observation",
                        "boundary_type": "SCHEMA_BOUNDARY",
                        "stop_reason": "passive_schema_visible",
                        "evidence_type": "SCHEMA_HINT",
                        "evidence_gain": 1.0,
                        "confidence": 0.70,
                        "slot_evidence": {},
                        "evidence_capsule": schema_capsule,
                        "no_action_inspect": True,
                    }
                    observations.append(schema_observation)
                    evidence_capsules.append(schema_capsule)
                attempted_any = False
            if self.light_explore_fixed_framework and task_mode == "INFO_QUERY":
                root_observation, root_capsule = self._root_list_inspect_observation(
                    goal=goal,
                    strategy_id=self.light_explore_search_strategy,
                    step_id=step_idx + 1,
                    root_state=root_state,
                    root_activity=root_activity,
                    root_hash=int(root_hash or -1),
                    root_screenshot=str(trace.get("root_screenshot") or ""),
                    root_labels=root_observed_elements,
                )
                if root_observation and root_capsule:
                    observations.append(root_observation)
                    evidence_capsules.append(root_capsule)
                    attempted_any = True
                    self._update_search_strategy_stats(root_observation)
                    print(
                        f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                        "light_explore_root_list_inspect facts="
                        f"{len(root_observation.get('observed_elements') or [])}"
                    )
                    slot_evidence = root_observation.get("slot_evidence") if isinstance(root_observation.get("slot_evidence"), dict) else {}
                    if (
                        _clean_text(root_observation.get("evidence_type")) == "ANSWER_HINT"
                        and bool(slot_evidence.get("slot_complete"))
                    ):
                        branch_candidates = []
                        trace["selected_targets"] = [self._candidate_trace(root_capsule.get("action") or {})]
            if not branch_candidates and not observations:
                trace["status"] = "no_branch_candidates"
                print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} light_explore_no_branch_candidates")
                return trace
            if branch_candidates:
                shortlist = ", ".join(
                    f"{_clean_text(c.get('label')) or 'candidate'}:{float(c.get('score') or 0.0):.2f}"
                    for c in branch_candidates
                )
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    f"light_explore_branch_begin strategy={self.light_explore_search_strategy} "
                    f"candidates={len(branch_candidates)} depth={self._search_strategy_depth_limit(self._task_mode(goal))} "
                    f"shortlist=[{shortlist}]"
                )
            self._state_acquisition_context = "branch"
            for b_idx, first_candidate in enumerate(branch_candidates):
                if bool(getattr(self, "light_explore_lb_mcts", False)) and self.light_explore_search_strategy == "mcts":
                    elapsed_ms = float(max(0.0, time.time() - float(trace.get("started_at") or time.time())) * 1000.0)
                    if elapsed_ms > float(trace.get("latency_budget_ms") or 0.0):
                        trace["mcts_stopped_by_latency_budget"] = True
                        trace["stopped_by_latency_budget"] = True
                        trace["stop_reason"] = "latency_budget_exhausted"
                        break
                labels: list[str] = []
                branch_steps: list[dict[str, Any]] = []
                total_score = float(first_candidate.get("score") or 0.0)
                changed_any = False
                final_activity = root_activity
                observed_elements: list[str] = []
                depth_reached = 0
                branch_operator, branch_operator_reason = self._candidate_operator(first_candidate, goal)
                boundary_type = "none"
                stop_reason = "not_started"
                evidence_type = "none"
                root_action_signature = self._safe_action_signature_from_candidate(first_candidate)
                depth1_state_signature: dict[str, Any] = {}
                depth2_action_signature: dict[str, Any] = {}
                depth2_state_signature: dict[str, Any] = {}
                depth2_candidate_for_promotion: dict[str, Any] = {}
                depth2_action_locatable = False

                if self.light_explore_search_policy in {"operator", "task_gate"} and branch_operator == "RiskBoundary":
                    risk_rollback = {"success": True, "mode": "not_executed", "level": "risk_boundary"}
                    risk_assessment = self._assess_branch_evidence(
                        goal=goal,
                        root_labels=root_observed_elements,
                        observed_elements=root_observed_elements,
                        candidate=first_candidate,
                        operator=branch_operator,
                        changed=False,
                        depth=0,
                        rollback_success=True,
                        after_activity=root_activity,
                    )
                    risk_capsule = self._build_evidence_capsule(
                        goal=goal,
                        strategy_id=self.light_explore_search_strategy,
                        step_id=step_idx + 1,
                        branch_id=b_idx + 1,
                        depth=0,
                        parent_state=root_state,
                        parent_activity=root_activity,
                        child_state=root_state,
                        child_activity=root_activity,
                        candidate=first_candidate,
                        assessment=risk_assessment,
                        rollback_info=risk_rollback,
                    )
                    evidence_capsules.append(risk_capsule)
                    trace.setdefault("risk_boundaries", []).append(self._candidate_trace(first_candidate))
                    observations.append(
                        {
                            "branch_id": int(b_idx + 1),
                            "labels": [_clean_text(first_candidate.get("label")) or "risk boundary"],
                            "changed": False,
                            "after_activity": root_activity,
                            "score": float(total_score),
                            "depth_reached": 0,
                            "observed_elements": self._state_semantic_summary(root_state),
                            "steps": [],
                            "rollback": risk_rollback,
                            "operator": branch_operator,
                            "operator_reason": branch_operator_reason,
                            "layout_region": _clean_text(first_candidate.get("layout_region")),
                            "semantic_role": _clean_text(first_candidate.get("semantic_role")),
                            "boundary_type": risk_assessment.get("boundary_type"),
                            "stop_reason": "risk_boundary_not_executed",
                            "evidence_type": risk_assessment.get("evidence_type"),
                            "evidence_gain": float(risk_assessment.get("evidence_gain") or 0.0),
                            "confidence": float(risk_assessment.get("confidence") or 0.0),
                            "evidence_capsule": risk_capsule,
                        }
                    )
                    continue

                first_action = self._probe_action_from_candidate(first_candidate)
                if first_action is None:
                    continue
                first_center = first_candidate.get("center")
                first_center_text = (
                    f" center={[int(first_center[0]), int(first_center[1])]}"
                    if isinstance(first_center, (list, tuple)) and len(first_center) >= 2
                    else ""
                )
                first_label = _clean_text(first_candidate.get("label")) or f"candidate_{b_idx}"
                if self._is_transaction_unsafe_candidate(first_candidate):
                    trace["status"] = "skipped_transaction_unsafe"
                    trace["trigger_reason"] = "selected_candidate_transaction_unsafe"
                    trace["selected_targets"] = [self._candidate_trace(first_candidate)]
                    return trace
                labels.append(first_label)
                key = _clean_text(first_candidate.get("key"))
                if key:
                    self._explored_element_visits[key] = int(self._explored_element_visits.get(key, 0)) + 1
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    f"light_explore_branch={b_idx + 1} depth=1 "
                    f"{first_action.action_type}={first_label}{first_center_text}"
                )
                attempted_any = True
                self._execute_probe_action(first_action)
                state_fetch_start = time.time()
                state_after_first = self._get_probe_state(wait_to_stabilize=True)
                state_fetch_ms = float(max(0.0, time.time() - state_fetch_start) * 1000.0)
                last_branch_state = state_after_first
                depth1_state_signature = self._build_state_signature_from_probe(state_after_first)
                changed1, activity1, hash_diff1 = self._probe_page_changed(root_activity, root_hash, state_after_first)
                semantic1 = self._semantic_state_change_details(
                    root_state=root_state,
                    child_state=state_after_first,
                    candidate=first_candidate,
                    goal=goal,
                    root_activity=root_activity,
                    child_activity=activity1,
                    hash_diff=hash_diff1,
                )
                semantic_changed1 = bool(semantic1.get("semantic_changed"))
                if not bool(changed1 or semantic_changed1):
                    no_effect_key = _clean_text(first_candidate.get("key"))
                    if no_effect_key:
                        self._no_effect_probe_keys.add(no_effect_key)
                depth_reached = 1
                changed_any = bool(changed_any or changed1 or semantic_changed1)
                final_activity = activity1
                summary_start = time.time()
                observed_elements = self._state_semantic_summary(state_after_first, limit=answer_label_limit)
                summary_ms = float(max(0.0, time.time() - summary_start) * 1000.0)
                a11y_trace_start = time.time()
                a11y_trace = self._state_a11y_trace(state_after_first, limit=TRACE_OBSERVED_ELEMENT_LIMIT)
                a11y_trace_ms = float(max(0.0, time.time() - a11y_trace_start) * 1000.0)
                branch_steps.append(
                    {
                        "depth": 1,
                        "candidate": self._candidate_trace(first_candidate),
                        "changed": bool(changed1),
                        "hash_changed": bool(semantic1.get("hash_changed")),
                        "activity_changed": bool(semantic1.get("activity_changed")),
                        "semantic_changed": bool(semantic_changed1),
                        "semantic_change_reason": semantic1.get("semantic_change_reason"),
                        "label_delta_count": int(semantic1.get("label_delta_count") or 0),
                        "search_ui_transition": bool(semantic1.get("search_ui_transition")),
                        "dialog_or_sheet_transition": bool(semantic1.get("dialog_or_sheet_transition")),
                        "screen_role_before": semantic1.get("screen_role_before"),
                        "screen_role_after": semantic1.get("screen_role_after"),
                        "after_activity": activity1,
                        "after_hash": self._state_hash(state_after_first),
                        "hash_diff_from_root": hash_diff1,
                        "state_fetch_ms": state_fetch_ms,
                        "a11y_dump_ms": self._state_a11y_latency_ms(state_after_first),
                        "summary_ms": summary_ms,
                        "a11y_trace_ms": a11y_trace_ms,
                        "state_aux": self._state_aux_trace(state_after_first),
                        "screenshot": self._save_trace_screenshot(
                            goal, state_after_first, f"explore_step_{step_idx + 1:02d}_b{b_idx + 1}_d1.png"
                        ),
                        "observed_elements": list(observed_elements),
                        "a11y": a11y_trace,
                    }
                )
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    f"light_explore_branch={b_idx + 1} depth=1 changed={changed1} hash_diff={hash_diff1} after={activity1}"
                )
                if self.light_explore_search_policy in {"operator", "task_gate"}:
                    if branch_operator == "ListInspect":
                        boundary_type = "visible_list_evidence"
                        evidence_type = "ANSWER_HINT"
                        stop_reason = "list_inspect_boundary"
                    elif branch_operator == "FormSchema":
                        boundary_type = "schema_boundary"
                        evidence_type = "SCHEMA_HINT"
                        stop_reason = "form_schema_boundary"
                    elif branch_operator == "SearchPeek":
                        boundary_type = "search_schema_or_filter"
                        evidence_type = "SCHEMA_HINT"
                        stop_reason = "search_peek_boundary"
                    elif branch_operator == "NavigationPeek":
                        boundary_type = "navigation_boundary"
                        evidence_type = "ACTION_HINT"
                        stop_reason = "navigation_peek_boundary"

                depth1_assessment = self._assess_branch_evidence(
                    goal=goal,
                    root_labels=root_observed_elements,
                    observed_elements=list(observed_elements),
                    candidate=first_candidate,
                    operator=branch_operator,
                    changed=bool(changed1 or semantic_changed1),
                    depth=1,
                    rollback_success=True,
                    after_activity=activity1,
                )
                boundary_type = _clean_text(depth1_assessment.get("boundary_type") or boundary_type)
                stop_reason = _clean_text(depth1_assessment.get("stop_reason") or stop_reason)
                evidence_type = _clean_text(depth1_assessment.get("evidence_type") or evidence_type)
                last_evidence_candidate = first_candidate
                rooted_in_planned = bool(first_candidate.get("is_planned_action")) or float(
                    first_candidate.get("planned_action_alignment") or 0.0
                ) >= 0.25
                depth2_root_eligible = bool(rooted_in_planned or self.light_explore_decouple_planned)
                root_label_for_t2 = _clean_text(first_candidate.get("label") or first_candidate.get("merged")).lower()
                root_is_search = bool(branch_operator == "SearchPeek" or re.search(r"\b(search|find)\b", root_label_for_t2))
                force_searchinput_t2 = bool(
                    self.t2_allow_safe_search_input
                    and depth2_root_eligible
                    and root_is_search
                    and self._state_has_search_ui(state_after_first)
                    and _clean_text(self._extract_goal_search_text(goal))
                )
                should_expand_depth2 = True
                if self.light_explore_search_policy in {"operator", "task_gate"}:
                    should_expand_depth2 = bool(
                        self._should_expand_for_search_strategy(
                            depth1_assessment,
                            branch_operator=branch_operator,
                            depth=1,
                            task_mode=task_mode,
                            branch_index=b_idx + 1,
                        )
                    )
                if force_searchinput_t2:
                    should_expand_depth2 = True
                depth2_policy = _clean_text(trace.get("reasoning_prior_depth2_policy") or "semantic_changed_only")
                if depth2_policy == "disabled_except_search" and not force_searchinput_t2:
                    should_expand_depth2 = False
                voc_depth2_decision = (
                    first_candidate.get("voc_depth2_decision")
                    if isinstance(first_candidate.get("voc_depth2_decision"), dict)
                    else {}
                )
                if (
                    self.light_explore_search_strategy == "value_of_computation"
                    and voc_depth2_decision
                    and not bool(voc_depth2_decision.get("allow"))
                    and not force_searchinput_t2
                ):
                    should_expand_depth2 = False
                session_depth2_utility = float(first_candidate.get("session_depth2_utility") or 0.0)
                if (
                    self.light_explore_search_strategy == "value_of_computation"
                    and session_depth2_utility <= 0.0
                    and not force_searchinput_t2
                ):
                    should_expand_depth2 = False
                rollback_cost_for_depth2 = float(first_candidate.get("voc_rollback_cost") or 0.0)
                consumable_depth2 = self._selective_depth2_consumable_decision(
                    goal=goal,
                    task_mode=task_mode,
                    task_phase=_clean_text(trace.get("selective_task_phase") or trace.get("voc_task_phase")),
                    state_after_depth1=state_after_first,
                    observed_elements=list(observed_elements),
                    branch_operator=branch_operator,
                    candidate=first_candidate,
                    rollback_cost=rollback_cost_for_depth2,
                )
                force_consumable_t2 = bool(consumable_depth2.get("depth2_allowed"))
                allow_depth2 = bool(
                    self.light_explore_enable_t2_lookahead
                    and int(self.light_explore_branch_depth) >= 2
                    and depth2_root_eligible
                    and not self._is_transaction_unsafe_candidate(first_candidate)
                    and (semantic_changed1 or force_searchinput_t2 or force_consumable_t2)
                    and (should_expand_depth2 or force_consumable_t2)
                    and (b_idx == 0 or len(branch_candidates) == 1 or force_searchinput_t2 or force_consumable_t2)
                    and not self._is_launcher_activity(activity1)
                )
                t2_debug = {
                    "task_id": _clean_text(goal)[:200],
                    "step": int(step_idx + 1),
                    "branch_id": int(b_idx + 1),
                    "rooted_in_planned_action": bool(rooted_in_planned),
                    "decoupled_from_planned_action": bool(self.light_explore_decouple_planned),
                    "depth2_root_eligible": bool(depth2_root_eligible),
                    "root_operator": branch_operator,
                    "root_label": _clean_text(first_candidate.get("label") or first_candidate.get("merged")),
                    "depth1_changed": bool(changed1),
                    "semantic_changed": bool(semantic_changed1),
                    "semantic_change_reason": semantic1.get("semantic_change_reason"),
                    "search_ui_transition": bool(semantic1.get("search_ui_transition")),
                    "force_searchinput_t2": bool(force_searchinput_t2),
                    "should_expand_depth2": bool(should_expand_depth2),
                    "reasoning_prior_depth2_policy": depth2_policy,
                    "voc_depth2_decision": voc_depth2_decision,
                    "session_branch_utility": float(first_candidate.get("session_branch_utility") or 0.0),
                    "session_depth2_utility": float(session_depth2_utility),
                    "allow_depth2": bool(allow_depth2),
                    "consumable_depth2": dict(consumable_depth2),
                    "depth2_attempted": False,
                    "depth2_candidate_found": False,
                    "depth2_candidate_kind": "",
                    "depth2_block_reason": "",
                }
                if not allow_depth2:
                    reason_parts = []
                    if not self.light_explore_enable_t2_lookahead:
                        reason_parts.append("t2_disabled")
                    if int(self.light_explore_branch_depth) < 2:
                        reason_parts.append("depth_budget_lt2")
                    if not depth2_root_eligible:
                        reason_parts.append("not_rooted_in_planned_action")
                    if not bool(changed1):
                        reason_parts.append("hash_or_activity_unchanged")
                        trace["depth2_blocked_by_hash_unchanged_count"] += 1
                        self._state_acquisition_metrics["depth2_blocked_by_hash_unchanged_count"] = (
                            float(self._state_acquisition_metrics.get("depth2_blocked_by_hash_unchanged_count") or 0.0) + 1.0
                        )
                    if not semantic_changed1 and not force_searchinput_t2:
                        reason_parts.append("semantic_unchanged")
                        trace["depth2_blocked_by_semantic_unchanged_count"] += 1
                        self._state_acquisition_metrics["depth2_blocked_by_semantic_unchanged_count"] = (
                            float(self._state_acquisition_metrics.get("depth2_blocked_by_semantic_unchanged_count") or 0.0) + 1.0
                        )
                    if not should_expand_depth2:
                        reason_parts.append("should_expand_false")
                        if voc_depth2_decision:
                            reason_parts.append(
                                "voc_depth2:"
                                + _clean_text(voc_depth2_decision.get("reason") or "blocked")
                            )
                        if (
                            self.light_explore_search_strategy == "value_of_computation"
                            and session_depth2_utility <= 0.0
                        ):
                            reason_parts.append("session_depth2_utility_non_positive")
                        if depth2_policy:
                            reason_parts.append(f"prior_depth2_policy:{depth2_policy}")
                        trace["depth2_blocked_by_should_expand_count"] += 1
                        self._state_acquisition_metrics["depth2_blocked_by_should_expand_count"] = (
                            float(self._state_acquisition_metrics.get("depth2_blocked_by_should_expand_count") or 0.0) + 1.0
                        )
                    if self._is_transaction_unsafe_candidate(first_candidate):
                        reason_parts.append("root_safety_block")
                        trace["depth2_blocked_by_safety_count"] += 1
                        self._state_acquisition_metrics["depth2_blocked_by_safety_count"] = (
                            float(self._state_acquisition_metrics.get("depth2_blocked_by_safety_count") or 0.0) + 1.0
                        )
                    if self._is_launcher_activity(activity1):
                        reason_parts.append("launcher_after_depth1")
                    if not bool(consumable_depth2.get("depth2_allowed")):
                        reason_parts.append(_clean_text(consumable_depth2.get("depth2_block_reason") or "not_consumable"))
                    t2_debug["depth2_block_reason"] = ",".join(reason_parts) or "blocked"
                self._append_diagnostic_jsonl(goal, "depth2_consumable_records.jsonl", {
                    "step": int(step_idx + 1),
                    "branch_id": int(b_idx + 1),
                    "depth1_result_type": _clean_text(consumable_depth2.get("depth1_result_type")),
                    "depth2_candidate": _clean_text(first_candidate.get("label") or first_candidate.get("merged")),
                    "depth2_action_type": _clean_text(consumable_depth2.get("depth2_action_type")),
                    "depth2_allowed": bool(allow_depth2),
                    "depth2_reason": _clean_text(consumable_depth2.get("depth2_reason")),
                    "depth2_block_reason": _clean_text(t2_debug.get("depth2_block_reason") or consumable_depth2.get("depth2_block_reason")),
                })
                if allow_depth2:
                    state_for_deeper = state_after_first
                    previous_candidate = first_candidate
                    second_candidate = self._typed_probe_candidate(
                        state_after_first,
                        goal,
                        planning_text=planning_text,
                    )
                    selective_safe_search_t2 = bool(consumable_depth2.get("depth1_result_type") == "search_field")
                    if force_searchinput_t2 and second_candidate is None:
                        trace["depth2_blocked_by_no_typed_candidate_count"] += 1
                        self._state_acquisition_metrics["depth2_blocked_by_no_typed_candidate_count"] = (
                            float(self._state_acquisition_metrics.get("depth2_blocked_by_no_typed_candidate_count") or 0.0) + 1.0
                        )
                    if (
                        second_candidate is not None
                        and self._is_transaction_unsafe_candidate(second_candidate)
                        and not (
                            (self.t2_allow_safe_search_input or selective_safe_search_t2)
                            and self._is_safe_search_input_candidate(second_candidate, goal, state_after_first)
                        )
                    ):
                        second_candidate = None
                    if second_candidate is None:
                        second_candidates = self._collect_probe_candidates(
                            state_after_first,
                            goal,
                            planning_text=planning_text,
                        )
                        second_candidates, second_protection_trace = self._apply_protected_task_candidate_policy(
                            second_candidates,
                            goal=goal,
                            prior=reasoning_prior,
                        )
                        if second_protection_trace.get("applied"):
                            t2_debug["depth2_protection_applied"] = True
                            t2_debug["depth2_protection_mode"] = _clean_text(second_protection_trace.get("mode") or "")
                            t2_debug["depth2_protection_before_count"] = int(second_protection_trace.get("before_count", 0))
                            t2_debug["depth2_protection_after_count"] = int(second_protection_trace.get("after_count", 0))
                        second_candidates, _, _, _ = self._evaluate_candidate_priors(
                            second_candidates,
                            goal=goal,
                            reasoning_prior=reasoning_prior,
                            current_screen_role=self._screen_role_from_state(state_after_first),
                            app_name=activity1,
                            step_idx=step_idx + 1,
                        )
                        second_candidates = self._prepare_search_strategy_candidates(second_candidates, goal)
                        second_candidate = self._choose_secondary_probe_candidate(
                            second_candidates,
                            first_candidate=first_candidate,
                        )
                    if second_candidate is None:
                        t2_debug["depth2_block_reason"] = "no_t2_candidate"
                        trace["depth2_blocked_by_no_typed_candidate_count"] += 1
                        self._state_acquisition_metrics["depth2_blocked_by_no_typed_candidate_count"] = (
                            float(self._state_acquisition_metrics.get("depth2_blocked_by_no_typed_candidate_count") or 0.0) + 1.0
                        )
                    if second_candidate is not None:
                        second_center = second_candidate.get("center")
                        if isinstance(second_center, (list, tuple)) and len(second_center) >= 2:
                            t2_debug["depth2_candidate_found"] = True
                            t2_debug["depth2_candidate_kind"] = _clean_text(second_candidate.get("action_kind") or "click")
                            t2_debug["depth2_candidate_label"] = _clean_text(second_candidate.get("label") or second_candidate.get("merged"))
                            trace["t2_candidate_count"] += 1
                            self._state_acquisition_metrics["t2_candidate_count"] = (
                                float(self._state_acquisition_metrics.get("t2_candidate_count") or 0.0) + 1.0
                            )
                            if _clean_text(second_candidate.get("action_kind")).lower() in {"type", json_action.INPUT_TEXT}:
                                trace["searchinput_t2_candidate_count"] += 1
                                self._state_acquisition_metrics["searchinput_t2_candidate_count"] = (
                                    float(self._state_acquisition_metrics.get("searchinput_t2_candidate_count") or 0.0) + 1.0
                                )
                            second_label = _clean_text(second_candidate.get("label")) or f"candidate_{b_idx}_2"
                            labels.append(second_label)
                            total_score += float(second_candidate.get("score") or 0.0)
                            second_key = _clean_text(second_candidate.get("key"))
                            if second_key:
                                self._explored_element_visits[second_key] = int(
                                    self._explored_element_visits.get(second_key, 0)
                                ) + 1
                            print(
                                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                                f"light_explore_branch={b_idx + 1} depth=2 {second_candidate.get('action_kind') or 'click'}="
                                f"{second_label} center={[int(second_center[0]), int(second_center[1])]}"
                            )
                            second_action = self._probe_action_from_candidate(second_candidate)
                            if second_action is None:
                                break
                            depth2_action_signature = self._safe_action_signature_from_candidate(second_candidate)
                            depth2_candidate_for_promotion = dict(second_candidate)
                            depth2_action_locatable = self._locate_candidate_action_in_state(
                                state_after_first,
                                depth2_action_signature,
                            ) or self._candidate_visible_in_state(second_candidate, state_after_first)
                            self._execute_probe_action(second_action)
                            state_fetch_start = time.time()
                            state_after_second = self._get_probe_state(wait_to_stabilize=True)
                            state_fetch_ms = float(max(0.0, time.time() - state_fetch_start) * 1000.0)
                            last_branch_state = state_after_second
                            depth2_state_signature = self._build_state_signature_from_probe(state_after_second)
                            state_for_deeper = state_after_second
                            previous_candidate = second_candidate
                            last_evidence_candidate = second_candidate
                            changed2, activity2, hash_diff2 = self._probe_page_changed(
                                root_activity,
                                root_hash,
                                state_after_second,
                            )
                            semantic2 = self._semantic_state_change_details(
                                root_state=state_after_first,
                                child_state=state_after_second,
                                candidate=second_candidate,
                                goal=goal,
                                root_activity=activity1,
                                child_activity=activity2,
                                hash_diff=hash_diff2,
                            )
                            semantic_changed2 = bool(semantic2.get("semantic_changed"))
                            t2_debug["depth2_attempted"] = True
                            t2_debug["depth2_changed"] = bool(changed2)
                            t2_debug["depth2_semantic_changed"] = bool(semantic_changed2)
                            t2_debug["depth2_semantic_change_reason"] = semantic2.get("semantic_change_reason")
                            depth_reached = 2
                            changed_any = bool(changed_any or changed2 or semantic_changed2)
                            final_activity = activity2
                            summary_start = time.time()
                            observed_elements = self._state_semantic_summary(state_after_second, limit=answer_label_limit)
                            summary_ms = float(max(0.0, time.time() - summary_start) * 1000.0)
                            a11y_trace_start = time.time()
                            a11y_trace = self._state_a11y_trace(
                                state_after_second,
                                limit=TRACE_OBSERVED_ELEMENT_LIMIT,
                            )
                            a11y_trace_ms = float(max(0.0, time.time() - a11y_trace_start) * 1000.0)
                            branch_steps.append(
                                {
                                    "depth": 2,
                                    "candidate": self._candidate_trace(second_candidate),
                                    "changed": bool(changed2),
                                    "hash_changed": bool(semantic2.get("hash_changed")),
                                    "activity_changed": bool(semantic2.get("activity_changed")),
                                    "semantic_changed": bool(semantic_changed2),
                                    "semantic_change_reason": semantic2.get("semantic_change_reason"),
                                    "label_delta_count": int(semantic2.get("label_delta_count") or 0),
                                    "search_ui_transition": bool(semantic2.get("search_ui_transition")),
                                    "dialog_or_sheet_transition": bool(semantic2.get("dialog_or_sheet_transition")),
                                    "screen_role_before": semantic2.get("screen_role_before"),
                                    "screen_role_after": semantic2.get("screen_role_after"),
                                    "after_activity": activity2,
                                    "after_hash": self._state_hash(state_after_second),
                                    "hash_diff_from_root": hash_diff2,
                                    "state_fetch_ms": state_fetch_ms,
                                    "a11y_dump_ms": self._state_a11y_latency_ms(state_after_second),
                                    "summary_ms": summary_ms,
                                    "a11y_trace_ms": a11y_trace_ms,
                                    "state_aux": self._state_aux_trace(state_after_second),
                                    "screenshot": self._save_trace_screenshot(
                                        goal,
                                        state_after_second,
                                        f"explore_step_{step_idx + 1:02d}_b{b_idx + 1}_d2.png",
                                    ),
                                    "observed_elements": list(observed_elements),
                                    "a11y": a11y_trace,
                                }
                            )
                            print(
                                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                                f"light_explore_branch={b_idx + 1} depth=2 changed={changed2} hash_diff={hash_diff2} after={activity2}"
                            )
                            current_assessment = self._assess_branch_evidence(
                                goal=goal,
                                root_labels=root_observed_elements,
                                observed_elements=list(observed_elements),
                                candidate=second_candidate,
                                operator=branch_operator,
                                changed=bool(changed2 or semantic_changed2),
                                depth=2,
                                rollback_success=True,
                                after_activity=activity2,
                            )
                            current_assessment["evidence_gain_delta"] = float(
                                current_assessment.get("evidence_gain") or 0.0
                            ) - float(depth1_assessment.get("evidence_gain") or 0.0)
                            boundary_type = _clean_text(current_assessment.get("boundary_type") or boundary_type)
                            stop_reason = _clean_text(current_assessment.get("stop_reason") or stop_reason)
                            evidence_type = _clean_text(current_assessment.get("evidence_type") or evidence_type)
                            max_extra_depth = self._search_strategy_depth_limit(task_mode)
                            if self.light_explore_search_policy in {"operator", "task_gate"} and not self._should_expand_for_search_strategy(
                                current_assessment,
                                branch_operator=branch_operator,
                                depth=2,
                                task_mode=task_mode,
                                branch_index=b_idx + 1,
                            ):
                                max_extra_depth = min(max_extra_depth, 2)
                            for extra_depth in range(3, max_extra_depth + 1):
                                if not bool(changed2):
                                    break
                                deeper_candidates = self._collect_probe_candidates(
                                    state_for_deeper,
                                    goal,
                                    planning_text=planning_text,
                                )
                                deeper_candidates, deeper_protection_trace = self._apply_protected_task_candidate_policy(
                                    deeper_candidates,
                                    goal=goal,
                                    prior=reasoning_prior,
                                )
                                if deeper_protection_trace.get("applied"):
                                    t2_debug["deeper_protection_applied"] = True
                                    t2_debug["deeper_protection_mode"] = _clean_text(deeper_protection_trace.get("mode") or "")
                                    t2_debug["deeper_protection_before_count"] = int(deeper_protection_trace.get("before_count", 0))
                                    t2_debug["deeper_protection_after_count"] = int(deeper_protection_trace.get("after_count", 0))
                                deeper_candidates, _, _, _ = self._evaluate_candidate_priors(
                                    deeper_candidates,
                                    goal=goal,
                                    reasoning_prior=reasoning_prior,
                                    current_screen_role=self._screen_role_from_state(state_for_deeper),
                                    app_name=final_activity,
                                    step_idx=step_idx + 1,
                                )
                                deeper_candidates = self._prepare_search_strategy_candidates(deeper_candidates, goal)
                                deeper_candidate = self._choose_secondary_probe_candidate(
                                    deeper_candidates,
                                    first_candidate=previous_candidate,
                                )
                                if deeper_candidate is None:
                                    break
                                if self._is_transaction_unsafe_candidate(deeper_candidate):
                                    break
                                deeper_center = deeper_candidate.get("center")
                                if not isinstance(deeper_center, (list, tuple)) or len(deeper_center) < 2:
                                    break
                                deeper_label = _clean_text(deeper_candidate.get("label")) or f"candidate_{b_idx}_{extra_depth}"
                                labels.append(deeper_label)
                                total_score += float(deeper_candidate.get("score") or 0.0)
                                deeper_key = _clean_text(deeper_candidate.get("key"))
                                if deeper_key:
                                    self._explored_element_visits[deeper_key] = int(
                                        self._explored_element_visits.get(deeper_key, 0)
                                    ) + 1
                                print(
                                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                                    f"light_explore_branch={b_idx + 1} depth={extra_depth} "
                                    f"{deeper_candidate.get('action_kind') or 'click'}={deeper_label} "
                                    f"center={[int(deeper_center[0]), int(deeper_center[1])]}"
                                )
                                deeper_action = self._probe_action_from_candidate(deeper_candidate)
                                if deeper_action is None:
                                    break
                                self._execute_probe_action(deeper_action)
                                state_fetch_start = time.time()
                                state_after_deeper = self._get_probe_state(wait_to_stabilize=True)
                                state_fetch_ms = float(max(0.0, time.time() - state_fetch_start) * 1000.0)
                                last_branch_state = state_after_deeper
                                changed_deeper, activity_deeper, hash_diff_deeper = self._probe_page_changed(
                                    root_activity,
                                    root_hash,
                                    state_after_deeper,
                                )
                                depth_reached = extra_depth
                                changed_any = bool(changed_any or changed_deeper)
                                final_activity = activity_deeper
                                summary_start = time.time()
                                observed_elements = self._state_semantic_summary(state_after_deeper, limit=answer_label_limit)
                                summary_ms = float(max(0.0, time.time() - summary_start) * 1000.0)
                                a11y_trace_start = time.time()
                                a11y_trace = self._state_a11y_trace(
                                    state_after_deeper,
                                    limit=TRACE_OBSERVED_ELEMENT_LIMIT,
                                )
                                a11y_trace_ms = float(max(0.0, time.time() - a11y_trace_start) * 1000.0)
                                branch_steps.append(
                                    {
                                        "depth": int(extra_depth),
                                        "candidate": self._candidate_trace(deeper_candidate),
                                        "changed": bool(changed_deeper),
                                        "after_activity": activity_deeper,
                                        "after_hash": self._state_hash(state_after_deeper),
                                        "hash_diff_from_root": hash_diff_deeper,
                                        "state_fetch_ms": state_fetch_ms,
                                        "a11y_dump_ms": self._state_a11y_latency_ms(state_after_deeper),
                                        "summary_ms": summary_ms,
                                        "a11y_trace_ms": a11y_trace_ms,
                                        "state_aux": self._state_aux_trace(state_after_deeper),
                                        "screenshot": self._save_trace_screenshot(
                                            goal,
                                            state_after_deeper,
                                            f"explore_step_{step_idx + 1:02d}_b{b_idx + 1}_d{extra_depth}.png",
                                        ),
                                        "observed_elements": list(observed_elements),
                                        "a11y": a11y_trace,
                                    }
                                )
                                print(
                                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                                    f"light_explore_branch={b_idx + 1} depth={extra_depth} "
                                    f"changed={changed_deeper} hash_diff={hash_diff_deeper} after={activity_deeper}"
                                )
                                current_assessment = self._assess_branch_evidence(
                                    goal=goal,
                                    root_labels=root_observed_elements,
                                    observed_elements=list(observed_elements),
                                    candidate=deeper_candidate,
                                    operator=branch_operator,
                                    changed=bool(changed_deeper),
                                    depth=int(extra_depth),
                                    rollback_success=True,
                                    after_activity=activity_deeper,
                                )
                                boundary_type = _clean_text(current_assessment.get("boundary_type") or boundary_type)
                                stop_reason = _clean_text(current_assessment.get("stop_reason") or stop_reason)
                                evidence_type = _clean_text(current_assessment.get("evidence_type") or evidence_type)
                                state_for_deeper = state_after_deeper
                                previous_candidate = deeper_candidate
                                last_evidence_candidate = deeper_candidate
                                changed2 = changed_deeper
                                if self.light_explore_search_policy in {"operator", "task_gate"} and not self._should_expand_for_search_strategy(
                                    current_assessment,
                                    branch_operator=branch_operator,
                                    depth=int(extra_depth),
                                    task_mode=task_mode,
                                    branch_index=b_idx + 1,
                                ):
                                    break

                if int(depth_reached) >= 2:
                    trace["rooted_depth2_count"] += 1
                    self._state_acquisition_metrics["rooted_depth2_count"] = (
                        float(self._state_acquisition_metrics.get("rooted_depth2_count") or 0.0) + 1.0
                    )
                elif rooted_in_planned:
                    trace["rooted_depth1_only_count"] += 1
                    self._state_acquisition_metrics["rooted_depth1_only_count"] = (
                        float(self._state_acquisition_metrics.get("rooted_depth1_only_count") or 0.0) + 1.0
                    )
                trace["t2_branch_debug"].append(dict(t2_debug))

                rollback_info = self._rollback_to_probe_root(
                    root_activity=root_activity,
                    root_hash=root_hash,
                    replay_actions=replay_actions,
                    step_idx=step_idx,
                    current_state=last_branch_state,
                )
                trace["rollbacks"].append(dict(rollback_info))
                final_assessment = self._assess_branch_evidence(
                    goal=goal,
                    root_labels=root_observed_elements,
                    observed_elements=list(observed_elements),
                    candidate=last_evidence_candidate,
                    operator=branch_operator,
                    changed=bool(changed_any),
                    depth=int(depth_reached),
                    rollback_success=bool((rollback_info or {}).get("success")),
                    after_activity=final_activity,
                )
                if final_assessment.get("boundary_type") and final_assessment.get("boundary_type") != "NONE":
                    boundary_type = _clean_text(final_assessment.get("boundary_type"))
                if final_assessment.get("stop_reason"):
                    stop_reason = _clean_text(final_assessment.get("stop_reason"))
                if final_assessment.get("evidence_type") and final_assessment.get("evidence_type") != "NONE":
                    evidence_type = _clean_text(final_assessment.get("evidence_type"))
                rollback_verified = bool((rollback_info or {}).get("success"))
                if not rollback_verified:
                    boundary_type = "ROLLBACK_FAILED"
                    stop_reason = "rollback_failed_discard_evidence"
                    evidence_type = "NONE"
                    final_assessment = dict(final_assessment)
                    final_assessment.update(
                        {
                            "boundary_type": boundary_type,
                            "stop_reason": stop_reason,
                            "evidence_type": "NONE",
                            "confidence": 0.0,
                            "evidence_gain": 0.0,
                        }
                    )
                evidence_capsule = self._build_evidence_capsule(
                    goal=goal,
                    strategy_id=self.light_explore_search_strategy,
                    step_id=step_idx + 1,
                    branch_id=b_idx + 1,
                    depth=int(depth_reached),
                    parent_state=root_state,
                    parent_activity=root_activity,
                    child_state=last_branch_state,
                    child_activity=final_activity,
                    candidate=last_evidence_candidate,
                    assessment=final_assessment,
                    rollback_info=dict(rollback_info),
                )
                if not rollback_verified:
                    evidence_capsule["discarded_due_to_rollback_failed"] = True
                    evidence_capsule["discard_reason"] = "rollback_failed"
                evidence_capsules.append(evidence_capsule)
                observation = {
                    "branch_id": int(b_idx + 1),
                    "labels": list(labels),
                    "changed": bool(changed_any),
                    "after_activity": final_activity,
                    "score": float(total_score),
                    "depth_reached": int(depth_reached),
                    "state_fetch_ms": float(sum(float(step.get("state_fetch_ms") or 0.0) for step in branch_steps)),
                    "a11y_dump_ms": float(sum(float(step.get("a11y_dump_ms") or 0.0) for step in branch_steps)),
                    "branch_state_fetch_ms": float(
                        sum(float(step.get("state_fetch_ms") or 0.0) for step in branch_steps)
                    ),
                    "branch_a11y_latency_ms": float(
                        sum(float(step.get("a11y_dump_ms") or 0.0) for step in branch_steps)
                    ),
                    "summary_ms": float(sum(float(step.get("summary_ms") or 0.0) for step in branch_steps)),
                    "a11y_trace_ms": float(sum(float(step.get("a11y_trace_ms") or 0.0) for step in branch_steps)),
                    "observed_elements": list(observed_elements),
                    "steps": list(branch_steps),
                    "rollback": dict(rollback_info),
                    "operator": branch_operator,
                    "operator_reason": branch_operator_reason,
                    "layout_region": _clean_text(first_candidate.get("layout_region")),
                    "semantic_role": _clean_text(first_candidate.get("semantic_role")),
                    "boundary_type": boundary_type,
                    "stop_reason": stop_reason,
                    "evidence_type": evidence_type,
                    "discarded_due_to_rollback_failed": not rollback_verified,
                    "root_action_signature": dict(root_action_signature),
                    "depth1_state_signature": dict(depth1_state_signature),
                    "depth2_action_signature": dict(depth2_action_signature),
                    "depth2_state_signature": dict(depth2_state_signature),
                    "depth1_candidate": self._candidate_trace(first_candidate),
                    "depth2_candidate": dict(depth2_candidate_for_promotion),
                    "depth2_action_locatable": bool(depth2_action_locatable),
                    "promotable_for_promotion": bool(
                        rollback_verified
                        and depth_reached >= 2
                        and bool(depth2_action_signature)
                        and bool(depth2_candidate_for_promotion)
                        and depth2_action_locatable
                    ),
                    "evidence_gain": float(final_assessment.get("evidence_gain") or 0.0),
                    "confidence": float(final_assessment.get("confidence") or 0.0),
                    "slot_evidence": final_assessment.get("slot_evidence") or {},
                    "delta": {
                        "new_labels": list(final_assessment.get("new_labels") or []),
                        "new_task_entities": list(final_assessment.get("new_task_entities") or []),
                        "new_goal_actions": list(final_assessment.get("new_goal_actions") or []),
                    },
                    "evidence_capsule": evidence_capsule,
                }
                observations.append(observation)
                self._update_search_strategy_stats(observation)
                if self.light_explore_search_strategy == "value_of_computation":
                    self._update_session_exploration_memory(goal, root_activity, observation)
                if not rollback_verified:
                    all_rollback_success = False
                    print(
                        f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                        f"light_explore_restore_failed={rollback_info}"
                    )
                    break

            min_attempts = int(getattr(self, "light_explore_min_attempts_per_step", 0) or 0)
            if min_attempts > 0:
                current_attempt_count = len(observations) + int(trace.get("risk_candidate_count") or 0)
                padding_needed = max(0, min_attempts - current_attempt_count)
                if padding_needed:
                    trace["passive_attempt_count"] = 0
                    trace["passive_padding_reason"] = "selective_consumable_skips_empty_passive_padding"
                    self._append_diagnostic_jsonl(goal, "passive_observation_records.jsonl", {
                        "step": int(step_idx + 1),
                        "observed_labels": list(root_observed_elements)[:80],
                        "produced_evidence": False,
                        "injected": False,
                        "reason": "empty_passive_padding_skipped",
                    })
                    print(
                        f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                        f"passive_coverage_padding_skipped={padding_needed}"
                    )

            if attempted_any:
                self._light_explore_runs += 1
            trace["observations"] = observations
            trace["evidence_capsules"] = evidence_capsules
            if min_attempts > 0:
                trace["attempt_count"] = int(max(int(trace.get("attempt_count") or 0), len(observations) + int(trace.get("risk_candidate_count") or 0)))
            depth_counts: dict[str, int] = {}
            branch_depth_counts: list[dict[str, Any]] = []
            for obs in observations:
                per_branch: dict[str, int] = {}
                for step in list(obs.get("steps") or []):
                    depth_key = str(int(step.get("depth") or 0))
                    depth_counts[depth_key] = int(depth_counts.get(depth_key, 0)) + 1
                    per_branch[depth_key] = int(per_branch.get(depth_key, 0)) + 1
                branch_depth_counts.append(
                    {
                        "branch_id": obs.get("branch_id"),
                        "depth_reached": obs.get("depth_reached"),
                        "depth_counts": per_branch,
                    }
                )
            trace["depth_counts"] = depth_counts
            trace["branch_depth_counts"] = branch_depth_counts
            rooted_total = int(trace.get("rooted_depth1_only_count") or 0) + int(trace.get("rooted_depth2_count") or 0)
            trace["rooted_depth2_rate"] = float(int(trace.get("rooted_depth2_count") or 0)) / float(rooted_total or 1)
            trace["exploration_state_fetch_ms"] = float(
                trace.get("root_state_fetch_ms") or 0.0
            ) + float(sum(float(obs.get("state_fetch_ms") or 0.0) for obs in observations))
            trace["exploration_a11y_dump_ms"] = float(
                trace.get("root_a11y_dump_ms") or 0.0
            ) + float(sum(float(obs.get("a11y_dump_ms") or 0.0) for obs in observations))
            trace["exploration_summary_ms"] = float(
                trace.get("root_summary_ms") or 0.0
            ) + float(sum(float(obs.get("summary_ms") or 0.0) for obs in observations))
            trace["exploration_a11y_trace_ms"] = float(
                trace.get("root_a11y_trace_ms") or 0.0
            ) + float(sum(float(obs.get("a11y_trace_ms") or 0.0) for obs in observations))
            has_depth2_observation = any(
                bool((obs.get("rollback") or {}).get("success"))
                and int(obs.get("depth_reached") or 0) >= 2
                and any(
                    int(step.get("depth") or 0) == 1 and bool(step.get("changed"))
                    or int(step.get("depth") or 0) == 1 and bool(step.get("semantic_changed"))
                    for step in list(obs.get("steps") or [])
                    if isinstance(step, dict)
                )
                for obs in observations
            )
            has_fixed_prompt_candidate = bool(
                self.light_explore_fixed_framework
                and any(
                    bool((obs.get("rollback") or {}).get("success"))
                    and bool(obs.get("changed"))
                    and (
                        _clean_text(obs.get("evidence_type")) in {
                            "ACTION_HINT",
                            "ANSWER_HINT",
                            "AVOID_HINT",
                            "SCHEMA_HINT",
                            "RISK_HINT",
                        }
                    )
                    and float(obs.get("confidence") or 0.0) >= 0.45
                    for obs in observations
                    if isinstance(obs, dict)
                )
            )
            speculative_context, speculative_results = self._build_prompt_context_from_observations(observations)
            if not all_rollback_success:
                speculative_context = ""
                speculative_results = []
            if not has_depth2_observation and not has_fixed_prompt_candidate:
                speculative_context = ""
                speculative_results = []
            trace["speculative_results"] = speculative_results
            trace["speculative_context"] = speculative_context
            trace["selected_prompt_results"] = []
            trace["prompt_context"] = ""
            trace["status"] = "completed" if all_rollback_success else "rollback_failed"
            trace["rollback_success"] = bool(all_rollback_success)
            has_consumable_observation = any(
                _clean_text(obs.get("evidence_type")) in {"ANSWER_HINT", "ACTION_HINT", "SCHEMA_HINT", "AVOID_HINT", "RISK_HINT"}
                for obs in observations if isinstance(obs, dict)
            )
            trace["available_for_next_prompt"] = bool(
                all_rollback_success and (has_depth2_observation or has_fixed_prompt_candidate or has_consumable_observation)
            )
            if not all_rollback_success:
                trace["exploration_step_stopped_after_rollback_failed"] = True
            if trace["available_for_next_prompt"]:
                self._pending_speculative_traces = [trace]
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    f"light_explore_speculative_results_cached items={len(speculative_results)}"
                )
            else:
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    "light_explore_no_prompt_context"
                )
            return trace
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(
                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                f"light_explore_failed: {exc}"
            )
            self._light_explore_runs += 1
            trace["status"] = "exception"
            trace["error"] = _clean_text(exc)
            return trace
        finally:
            self._state_acquisition_context = previous_state_acquisition_context
            trace["latency_ms"] = float(max(0.0, time.time() - float(trace.get("started_at") or time.time())) * 1000.0)
            if bool(getattr(self, "light_explore_lb_mcts", False)) and self.light_explore_search_strategy == "mcts":
                budget_record = dict(self._lb_mcts_latency_budget_records.get(int(trace.get("step") or 0), {}))
                if budget_record:
                    observations = [obs for obs in list(trace.get("observations") or []) if isinstance(obs, dict)]
                    depth2_attempted = any(
                        any(int(step.get("depth") or 0) >= 2 for step in list(obs.get("steps") or []) if isinstance(step, dict))
                        for obs in observations
                    )
                    budget_record.update(
                        {
                            "phase": "final",
                            "elapsed_exploration_ms": float(trace.get("latency_ms") or 0.0),
                            "actual_rollouts": int(len(observations)),
                            "depth2_attempted": bool(depth2_attempted),
                            "stopped_by_latency_budget": bool(trace.get("stopped_by_latency_budget") or trace.get("mcts_stopped_by_latency_budget")),
                            "stopped_by_rollback_failure": bool(trace.get("exploration_step_stopped_after_rollback_failed") or trace.get("status") == "rollback_failed"),
                            "stopped_by_no_safe_action": bool(trace.get("status") in {"no_candidates", "no_branch_candidates"}),
                        }
                    )
                    self._append_diagnostic_jsonl(goal, "latency_budget_records.jsonl", budget_record)
            self._emit_diagnostic_artifacts(goal, trace)
            self._append_exploration_trace(goal, trace)

    def _after_reasoning_action(
        self,
        goal: str,
        step_idx: int,
        action: json_action.JSONAction,
        matched_prompt_results: list[dict[str, Any]],
    ) -> None:
        del goal, step_idx, action, matched_prompt_results

    def _append_hint_hit_follow_records(
        self,
        goal: str,
        step_idx: int,
        action: json_action.JSONAction,
        parsed_action: dict[str, Any],
        matched_prompt_results: list[dict[str, Any]],
    ) -> None:
        actual = self._action_summary_from_json_action(action)
        vlm_answer = _clean_text(parsed_action.get("return") or parsed_action.get("answer") or parsed_action.get("value"))
        for idx, item in enumerate(list(matched_prompt_results or []), start=1):
            if not isinstance(item, dict):
                continue
            candidate = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
            hint_type = _clean_text(item.get("hint_type") or item.get("evidence_type") or "EVIDENCE")
            suggested_action = self._action_summary_from_json_action(None, candidate)
            suggested_answer = ""
            if hint_type == "ANSWER_HINT":
                suggested_answer = _clean_text(item.get("rendered_prompt_text") or item.get("prompt_line"))
            hit = False
            followed = False
            follow_reason = "not_action_hint"
            violated = False
            violation_reason = ""
            if hint_type == "ACTION_HINT":
                hit, follow_reason = self._actions_match_for_shortcut(suggested_action, actual)
                followed = bool(hit)
            elif hint_type == "ANSWER_HINT":
                if vlm_answer and suggested_answer:
                    hint_tokens = set(re.findall(r"[a-z0-9.:%-]+", suggested_answer.lower()))
                    answer_tokens = set(re.findall(r"[a-z0-9.:%-]+", vlm_answer.lower()))
                    overlap = len(hint_tokens.intersection(answer_tokens)) / float(max(1, len(answer_tokens)))
                    hit = overlap >= 0.5
                    followed = hit
                    follow_reason = f"answer_token_overlap:{overlap:.2f}"
                else:
                    follow_reason = "answer_not_evaluable"
            elif hint_type == "AVOID_HINT":
                avoid_match, reason = self._actions_match_for_shortcut(suggested_action, actual)
                violated = bool(avoid_match)
                followed = not violated
                follow_reason = "avoided_warned_action" if followed else "violated_avoid_hint"
                violation_reason = reason if violated else ""
            self._append_diagnostic_jsonl(
                goal,
                "hint_hit_follow.jsonl",
                {
                    "step": int(step_idx + 1),
                    "hint_id": _clean_text(item.get("hint_id") or f"hint_{step_idx + 1}_{idx}"),
                    "hint_type": hint_type,
                    "injected": True,
                    "suggested_action": suggested_action,
                    "suggested_answer": suggested_answer,
                    "avoided_action": suggested_action if hint_type == "AVOID_HINT" else {},
                    "vlm_next_action": actual,
                    "vlm_next_answer": vlm_answer,
                    "hit": bool(hit),
                    "followed": bool(followed),
                    "follow_reason": follow_reason,
                    "violated": bool(violated),
                    "violation_reason": violation_reason,
                    "source_branch_id": item.get("branch_id"),
                    "source_exploration_step": item.get("source_step"),
                },
            )

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
        self._state_acquisition_metrics["full_a11y_calls"] = float(
            self._state_acquisition_metrics.get("full_a11y_calls") or 0.0
        ) + 1.0
        history = self._history_text()
        start_page_activity = self._foreground_activity_name()
        start_page_hash = self._state_hash(state)
        page_stalled = self._is_page_stalled(
            current_activity=start_page_activity,
            current_hash=start_page_hash,
        )

        promotion_result = self._try_execute_pending_promotion(
            goal=goal,
            step_idx=step_idx,
            start_time=start_time,
            state=state,
            current_activity=start_page_activity,
            current_hash=start_page_hash,
        )
        if promotion_result is not None:
            return promotion_result

        shortcut_event = self._evaluate_pending_shortcut_pre_reasoning(
            goal=goal,
            step_idx=step_idx,
            state=state,
            current_activity=start_page_activity,
            current_hash=start_page_hash,
        )
        active_shortcut_result = self._execute_active_shortcut_step(
            goal=goal,
            step_idx=step_idx,
            start_time=start_time,
            state=state,
            shortcut_event=shortcut_event,
        )
        if active_shortcut_result is not None:
            return active_shortcut_result

        hint_for_prompt, matched_prompt_results, match_trace = self._match_pending_speculative_exploration(
            goal=goal,
            step_idx=step_idx,
            state=state,
            current_activity=start_page_activity,
        )
        if not hint_for_prompt and getattr(self, "light_explore_relaxed_diagnostic_injection", False):
            relaxed_hint = self._build_relaxed_diagnostic_prompt_hint(
                matched_prompt_results=matched_prompt_results,
                match_trace=match_trace,
            )
            if relaxed_hint:
                hint_for_prompt = relaxed_hint
                matched_prompt_results = list(matched_prompt_results or [])
                match_trace["relaxed_diagnostic_injection"] = True
                match_trace["prompt_context"] = relaxed_hint
                match_trace["selected_prompt_results"] = matched_prompt_results
        self._append_prompt_hint_decision(
            goal=goal,
            step_idx=step_idx,
            hint_for_prompt=hint_for_prompt,
            matched_prompt_results=matched_prompt_results,
            match_trace=match_trace,
        )
        self._append_step_decoupling_status(
            goal=goal,
            step_idx=step_idx,
            evidence_injected_same_step=False,
        )
        screenshot = Image.fromarray(state.pixels)
        model_screenshot, original_size, resized_size = self._build_model_screenshot(screenshot)
        screen_size = self.env.logical_screen_size
        explore_trace: dict[str, Any] = {"status": "not_run", "trigger_reason": "before_planning"}

        prompt_mode = "baseline_plus_exploration" if hint_for_prompt else "baseline"
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
        full_prompt_text_path, exploration_context_text_path = self._write_prompt_trace_files(
            goal=goal,
            step_idx=step_idx,
            messages=messages,
            exploration_context=hint_for_prompt,
            message_text=message_text,
            hint_for_prompt=hint_for_prompt,
        )

        exploration_executor: concurrent.futures.ThreadPoolExecutor | None = None
        exploration_future: concurrent.futures.Future | None = None
        exploration_async: dict[str, Any] = {
            "started_before_vlm_response": False,
            "decoupled_from_planned_action": bool(self.light_explore_decouple_planned),
            "parallel_with_vlm": bool(self.light_explore_parallel_vlm),
            "future_start_ms": 0.0,
            "wait_after_vlm_ms": 0.0,
            "future_error": "",
        }
        delayed_prior_snapshot = self._delayed_reasoning_prior_for_exploration(goal, step_idx)
        if self.light_explore_parallel_vlm and self.light_explore_decouple_planned:
            exploration_async["started_before_vlm_response"] = True
            exploration_start = time.time()
            self._state_acquisition_metrics["root_state_reuse_count"] = (
                float(self._state_acquisition_metrics.get("root_state_reuse_count") or 0.0) + 1.0
            )
            exploration_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            exploration_future = exploration_executor.submit(
                self._run_light_exploration,
                goal,
                step_idx,
                None,
                page_stalled,
                state,
                start_page_activity,
                start_page_hash,
                "",
                delayed_prior_snapshot,
            )
            exploration_async["future_start_ms"] = float(max(0.0, time.time() - exploration_start) * 1000.0)

        vlm_start = time.time()
        response, _, _ = self.vllm.predict_mm("", [], messages=messages)
        vlm_latency_ms = float(max(0.0, time.time() - vlm_start) * 1000.0)
        if bool(getattr(self, "light_explore_lb_mcts", False)):
            previous_ewma = float(getattr(self, "_lb_mcts_reasoning_ewma_ms", 0.0) or vlm_latency_ms)
            self._lb_mcts_reasoning_ewma_ms = 0.7 * previous_ewma + 0.3 * float(vlm_latency_ms)
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
        action, tool_call, extras, parsed_action = self._normalize_terminal_action_answer(
            goal=goal,
            action=action,
            parsed_action=parsed_action,
            tool_call=tool_call,
            extras=extras,
        )
        action, tool_call, parsed_action, extras = gelab_agent.sanitize_launcher_action(
            goal=goal,
            state=state,
            action=action,
            tool_call=tool_call,
            parsed_action=parsed_action,
            extras=extras,
        )
        if getattr(self, "reasoning_prior_enabled", False):
            try:
                previous_screen_summary = "; ".join(self._state_semantic_summary(state, limit=40))
                reasoning_prior, template_record = self._build_reasoning_prior(
                    goal=goal,
                    raw_vlm_output=str(response),
                    parsed_action=dict(parsed_action),
                    previous_screen_summary=previous_screen_summary,
                )
                self._last_reasoning_prior = dict(reasoning_prior)
                self._reasoning_prior_for_next_step = dict(reasoning_prior)
                self._append_reasoning_prior_record(goal, reasoning_prior)
                self._append_prompt_template_record(goal, template_record)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._append_diagnostic_jsonl(
                    goal,
                    "reasoning_prior_records.jsonl",
                    {
                        "task_id": _clean_text(goal)[:200],
                        "source_step": int(step_idx + 1),
                        "parse_success": False,
                        "parse_error": f"reasoning_prior_build_failed:{_clean_text(exc)}",
                    },
                )
        task_dir_for_prompt = self._task_output_dir(goal)
        current_screenshot_path = (
            os.path.join(task_dir_for_prompt, f"screenshot_{len(self._actions)}.png")
            if task_dir_for_prompt
            else ""
        )
        self._append_diagnostic_jsonl(
            goal,
            "prompt_traces.jsonl",
            {
                "task_id": _clean_text(goal)[:200],
                "app": ",".join(self._goal_app_keywords(goal)[:3]),
                "task_mode": self._task_mode(goal),
                "step": int(step_idx + 1),
                "variant": self.light_explore_variant,
                "prompt_mode": "baseline_plus_exploration" if hint_for_prompt else "baseline",
                "full_prompt_text_path": full_prompt_text_path,
                "exploration_context_text_path": exploration_context_text_path,
                "exploration_context_text": hint_for_prompt,
                "injected_hint_count": len(matched_prompt_results or []) if hint_for_prompt else 0,
                "injected_hint_types": [
                    _clean_text(item.get("hint_type") or item.get("evidence_type") or "EVIDENCE")
                    for item in list(matched_prompt_results or [])
                    if isinstance(item, dict)
                ],
                "injected_hint_ids": [
                    _clean_text(item.get("hint_id") or f"hint_{step_idx + 1}_{idx + 1}")
                    for idx, item in enumerate(list(matched_prompt_results or []))
                    if isinstance(item, dict)
                ],
                "pending_exploration_count": int((match_trace or {}).get("pending_trace_count") or 0),
                "matched_evidence_count": int((match_trace or {}).get("matched_count") or 0),
                "rejected_evidence_count": len(getattr(self, "_last_strict_not_injected_reasons", []) or []),
                "top_rejected_reasons": list(getattr(self, "_last_strict_not_injected_reasons", []) or [])[:5],
                "screenshot_path": current_screenshot_path,
                "vlm_raw_output": str(response),
                "parsed_action": dict(parsed_action),
                "executed_action": action.__dict__,
                "tool_call": tool_call,
            },
        )
        self._append_hint_hit_follow_records(
            goal=goal,
            step_idx=step_idx,
            action=action,
            parsed_action=dict(parsed_action),
            matched_prompt_results=matched_prompt_results,
        )
        gelab_agent._print_step_section(step_idx, "Parsed action", gelab_agent._json_dumps_safe(dict(parsed_action)))  # pylint: disable=protected-access
        gelab_agent._print_step_section(step_idx, "Tool call", gelab_agent._json_dumps_safe(tool_call))  # pylint: disable=protected-access
        if isinstance(shortcut_event, dict) and self.t2_shortcut_mode == "shadow":
            try:
                self._complete_shadow_shortcut_eval(
                    goal=goal,
                    shortcut_event=shortcut_event,
                    vlm_action=action,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"[SHORTCUT {_now_hms()}] shadow_eval_failed: {exc}")

        try:
            self._after_reasoning_action(
                goal=goal,
                step_idx=step_idx,
                action=action,
                matched_prompt_results=matched_prompt_results,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} policy_update_failed: {exc}")

        if extras.get("return_text"):
            self.env.interaction_cache = str(extras["return_text"])

        if exploration_future is not None:
            wait_start = time.time()
            try:
                explore_trace = exploration_future.result()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                exploration_async["future_error"] = _clean_text(exc)
                explore_trace = {
                    "step": int(step_idx + 1),
                    "goal": goal,
                    "status": "exception",
                    "trigger_reason": "parallel_future_exception",
                    "error": _clean_text(exc),
                    "rollbacks": [],
                    "decoupled_from_planned_action": bool(self.light_explore_decouple_planned),
                    "parallel_with_vlm": True,
                    "started_at": time.time(),
                    "latency_ms": 0.0,
                }
                self._append_exploration_trace(goal, explore_trace)
            finally:
                exploration_async["wait_after_vlm_ms"] = float(max(0.0, time.time() - wait_start) * 1000.0)
                if exploration_executor is not None:
                    exploration_executor.shutdown(wait=False)
            if isinstance(explore_trace, dict):
                explore_trace["vlm_parallel"] = dict(exploration_async)
                explore_trace["vlm_latency_ms"] = float(vlm_latency_ms)
        elif action.action_type in {json_action.STATUS, json_action.ANSWER} and not self.light_explore_force_every_step:
            explore_trace = {"status": "skipped", "trigger_reason": "terminal_action", "rollbacks": []}
        else:
            planning_text = " ".join(
                _clean_text(parsed_action.get(key))
                for key in ("cot", "explain", "summary", "action", "value", "return")
                if _clean_text(parsed_action.get(key))
            )
            self._state_acquisition_metrics["root_state_reuse_count"] = (
                float(self._state_acquisition_metrics.get("root_state_reuse_count") or 0.0) + 1.0
            )
            explore_trace = self._run_light_exploration(
                goal=goal,
                step_idx=step_idx,
                current_action=None,
                page_stalled=page_stalled,
                root_state=state,
                root_activity=start_page_activity,
                root_hash=start_page_hash,
                planning_text="",
                reasoning_prior_snapshot=delayed_prior_snapshot,
            )
            if isinstance(explore_trace, dict):
                explore_trace["vlm_latency_ms"] = float(vlm_latency_ms)
        shortcut_plan: dict[str, Any] | None = None
        if isinstance(explore_trace, dict):
            try:
                shortcut_plan = self._build_shortcut_plan_from_trace(
                    goal=goal,
                    step_idx=step_idx,
                    planned_action=action,
                    explore_trace=explore_trace,
                    root_state=state,
                    root_activity=start_page_activity,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"[SHORTCUT {_now_hms()}] plan_build_failed: {exc}")
                if self.t2_shortcut_mode != "off":
                    failed_plan = {
                        "task_id": _clean_text(goal)[:200],
                        "episode_id": _clean_text(goal)[:80],
                        "step_t": int(step_idx + 1),
                        "plan_id": self._shortcut_plan_id(goal, step_idx, "exception"),
                        "mode": self.t2_shortcut_mode,
                        "planned_root_action": self._action_summary_from_json_action(action),
                        "confidence": 0.0,
                        "no_plan_reason": f"plan_build_exception:{_clean_text(exc)}",
                    }
                    self._append_shortcut_plan(goal, failed_plan)
            try:
                self._maybe_store_pending_promotion_from_trace(
                    goal=goal,
                    step_idx=step_idx,
                    action=action,
                    explore_trace=explore_trace,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._append_promotion_event(
                    goal,
                    {
                        "event_type": "promotion_store_failed",
                        "step": int(step_idx + 1),
                        "reason": _clean_text(exc),
                        "unsafe_promotion": False,
                    },
                )
            if isinstance(explore_trace, dict) and explore_trace.get("status") == "rollback_failed":
                planned_action_dict = dict(action.__dict__)
                precondition_ok = False
                precondition_reason = "rollback_failed_conservative_gate"
                if self.light_explore_rollback_policy == "improved":
                    try:
                        current_state_after_rollback = self._get_probe_state(wait_to_stabilize=True)
                        precondition_ok, precondition_reason = self._planned_action_precondition(
                            current_state=current_state_after_rollback,
                            root_state=state,
                            root_activity=start_page_activity,
                            root_hash=start_page_hash,
                            action=action,
                        )
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        precondition_ok = False
                        precondition_reason = f"precondition_check_failed:{_clean_text(exc)}"
                    if (
                        self.light_explore_fixed_framework
                        and not precondition_ok
                        and action.action_type in {
                            json_action.OPEN_APP,
                            json_action.SWIPE,
                            json_action.SCROLL,
                            json_action.NAVIGATE_BACK,
                            json_action.NAVIGATE_HOME,
                        }
                    ):
                        precondition_ok = True
                        precondition_reason = f"fixed_safe_non_target_action:{action.action_type}"
                explore_trace["planned_action_precondition_satisfied"] = bool(precondition_ok)
                explore_trace["planned_action_precondition_reason"] = precondition_reason
                parsed_action["exploration_rollback_gate_blocked_action"] = planned_action_dict
                parsed_action["action"] = "WAIT"
                parsed_action["summary"] = "Internal rollback-gate no-op; not added to prompt history."
                action = json_action.JSONAction(action_type=json_action.WAIT)
                tool_call = {"name": "mobile_use", "arguments": {"action": "wait", "value": 1}}
                extras = {
                    "wait_seconds": 1,
                    "internal_exploration_rollback_gate_noop": True,
                    "suppress_history_append": True,
                    "planned_action": planned_action_dict,
                    "planned_action_precondition_satisfied": bool(precondition_ok),
                    "rollback_gate_reason": precondition_reason,
                }
                explore_trace["planned_action_suppressed"] = True
                explore_trace["main_action_blocked_by_rollback_gate"] = True
                explore_trace["suppression_reason"] = "rollback_failed_conservative_gate"
                explore_trace["rollback_gate_reason"] = precondition_reason

        main_action_start = time.time()
        self._execute_action(action, extras)
        self._append_latency_profile_event(
            goal,
            {
                "event": "main_action_execute",
                "step": int(step_idx + 1),
                "prompt_mode": "baseline",
                "action_type": str(action.action_type),
                "latency_ms": float(max(0.0, time.time() - main_action_start) * 1000.0),
                "x": action.x,
                "y": action.y,
                "text_len": len(str(action.text or "")),
                "goal_status": action.goal_status,
            },
        )
        gelab_agent._print_step_section(step_idx, "Action", gelab_agent._json_dumps_safe(action.__dict__))  # pylint: disable=protected-access
        if extras:
            gelab_agent._print_step_section(step_idx, "Action extras", gelab_agent._json_dumps_safe(extras))  # pylint: disable=protected-access

        summary = gelab_agent._normalize_space(parsed_action.get("summary")) or gelab_agent._normalize_space(parsed_action.get("explain"))  # pylint: disable=protected-access
        if not summary:
            summary = str(tool_call.get("arguments") or tool_call)

        latency_sec = float(max(0.0, time.time() - start_time))
        step_record = {
            "goal": goal,
            "response": response,
            "parsed_action": dict(parsed_action),
            "tool_call": tool_call,
            "action_dict": action.__dict__,
            "summary": summary,
            "latency_sec": latency_sec,
            "vlm_latency_ms": float(vlm_latency_ms),
            "model_input_resolution": {"width": resized_size[0], "height": resized_size[1]},
            "original_resolution": {"width": original_size[0], "height": original_size[1]},
            "image_downsample_scale": self.image_downsample_scale,
            "prompt_mode": prompt_mode,
            "prompt_hint": hint_for_prompt,
            "matched_exploration_status": match_trace.get("status") if isinstance(match_trace, dict) else None,
            "matched_exploration_count": match_trace.get("matched_count") if isinstance(match_trace, dict) else 0,
            "matched_exploration_results": matched_prompt_results,
            "light_explore_runs": self._light_explore_runs,
            "start_page_activity": start_page_activity,
            "start_page_hash": start_page_hash,
            "page_stalled": page_stalled,
            "task_mode": self._task_mode(goal),
            "task_slots": slot_complete_evidence.parse_task_slots(goal).to_dict()
            if self.light_explore_slot_complete
            else {},
            "ablation_variant": self.light_explore_variant,
            "hint_policy": self.light_explore_hint_policy,
            "search_policy": self.light_explore_search_policy,
            "search_strategy": self.light_explore_search_strategy,
            "rollback_policy": self.light_explore_rollback_policy,
            "exploration_status": explore_trace.get("status") if isinstance(explore_trace, dict) else None,
            "exploration_mode": explore_trace.get("mode") if isinstance(explore_trace, dict) else None,
            "exploration_decoupled_from_planned_action": (
                bool(explore_trace.get("decoupled_from_planned_action")) if isinstance(explore_trace, dict) else False
            ),
            "exploration_parallel_with_vlm": (
                bool(explore_trace.get("parallel_with_vlm")) if isinstance(explore_trace, dict) else False
            ),
            "exploration_async": dict(exploration_async),
            "exploration_trigger_reason": explore_trace.get("trigger_reason") if isinstance(explore_trace, dict) else None,
            "exploration_candidate_count": explore_trace.get("candidate_count") if isinstance(explore_trace, dict) else 0,
            "exploration_selected_target_count": (
                len(explore_trace.get("selected_targets") or []) if isinstance(explore_trace, dict) else 0
            ),
            "exploration_observation_count": (
                len(explore_trace.get("observations") or []) if isinstance(explore_trace, dict) else 0
            ),
            "exploration_prompt_context": hint_for_prompt,
            "exploration_selected_prompt_results": matched_prompt_results,
            "post_planning_exploration_speculative_results": (
                explore_trace.get("speculative_results") if isinstance(explore_trace, dict) else []
            ),
            "rollback_success": explore_trace.get("rollback_success") if isinstance(explore_trace, dict) else None,
            "rollback_levels": [
                rb.get("level")
                for rb in (explore_trace.get("rollbacks") or [])
                if isinstance(rb, dict)
            ] if isinstance(explore_trace, dict) else [],
            "planned_action_suppressed": bool(explore_trace.get("planned_action_suppressed")) if isinstance(explore_trace, dict) else False,
            "planned_action_precondition_satisfied": explore_trace.get("planned_action_precondition_satisfied") if isinstance(explore_trace, dict) else None,
            "suppression_reason": explore_trace.get("suppression_reason") if isinstance(explore_trace, dict) else "",
            "operator_distribution": [
                obs.get("operator")
                for obs in (explore_trace.get("observations") or [])
                if isinstance(obs, dict)
            ] if isinstance(explore_trace, dict) else [],
            "boundary_stop_distribution": [
                obs.get("boundary_type")
                for obs in (explore_trace.get("observations") or [])
                if isinstance(obs, dict)
            ] if isinstance(explore_trace, dict) else [],
            "evidence_type_distribution": [
                obs.get("evidence_type")
                for obs in (explore_trace.get("observations") or [])
                if isinstance(obs, dict)
            ] if isinstance(explore_trace, dict) else [],
            "exploration_evidence_capsule_count": (
                len(explore_trace.get("evidence_capsules") or []) if isinstance(explore_trace, dict) else 0
            ),
            "exploration_latency_ms": float(explore_trace.get("latency_ms") or 0.0) if isinstance(explore_trace, dict) else 0.0,
            "exploration_state_fetch_ms": float(explore_trace.get("exploration_state_fetch_ms") or 0.0) if isinstance(explore_trace, dict) else 0.0,
            "exploration_a11y_dump_ms": float(explore_trace.get("exploration_a11y_dump_ms") or 0.0) if isinstance(explore_trace, dict) else 0.0,
            "exploration_summary_ms": float(explore_trace.get("exploration_summary_ms") or 0.0) if isinstance(explore_trace, dict) else 0.0,
            "exploration_a11y_trace_ms": float(explore_trace.get("exploration_a11y_trace_ms") or 0.0) if isinstance(explore_trace, dict) else 0.0,
            "state_acquisition_metrics": dict(self._state_acquisition_metrics),
        }
        if "shortcut_plan_id" not in step_record:
            step_record.update(
                {
                    "shortcut_plan_id": shortcut_event.get("plan_id") if isinstance(shortcut_event, dict) else None,
                    "shortcut_event_would_fire": bool(shortcut_event.get("would_fire")) if isinstance(shortcut_event, dict) else False,
                    "shortcut_event_confidence": shortcut_event.get("confidence") if isinstance(shortcut_event, dict) else None,
                    "shortcut_event_no_fire_reason": shortcut_event.get("no_fire_reason") if isinstance(shortcut_event, dict) else None,
                    "shortcut_fired": False,
                    "skipped_vlm_calls": 0,
                    "shortcut_event": shortcut_event if isinstance(shortcut_event, dict) else {},
                }
            )
        self._actions.append(step_record)
        if not bool(extras.get("suppress_history_append")):
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
                "vlm_latency_ms": float(vlm_latency_ms),
                "model_input_resolution": {"width": resized_size[0], "height": resized_size[1]},
                "original_resolution": {"width": original_size[0], "height": original_size[1]},
                "image_downsample_scale": self.image_downsample_scale,
                "prompt_mode": prompt_mode,
                "prompt_hint": hint_for_prompt,
                "matched_exploration_status": match_trace.get("status") if isinstance(match_trace, dict) else None,
                "matched_exploration_results": matched_prompt_results,
                "light_explore_runs": self._light_explore_runs,
                "exploration_status": explore_trace.get("status") if isinstance(explore_trace, dict) else None,
                "exploration_mode": explore_trace.get("mode") if isinstance(explore_trace, dict) else None,
                "exploration_decoupled_from_planned_action": (
                    bool(explore_trace.get("decoupled_from_planned_action")) if isinstance(explore_trace, dict) else False
                ),
                "exploration_parallel_with_vlm": (
                    bool(explore_trace.get("parallel_with_vlm")) if isinstance(explore_trace, dict) else False
                ),
                "exploration_async": dict(exploration_async),
                "exploration_candidate_count": explore_trace.get("candidate_count") if isinstance(explore_trace, dict) else 0,
                "exploration_selected_prompt_results": matched_prompt_results,
                "post_planning_exploration_speculative_results": (
                    explore_trace.get("speculative_results") if isinstance(explore_trace, dict) else []
                ),
                "shortcut_plan_id": shortcut_event.get("plan_id") if isinstance(shortcut_event, dict) else None,
                "shortcut_event_would_fire": bool(shortcut_event.get("would_fire")) if isinstance(shortcut_event, dict) else False,
                "shortcut_event_confidence": shortcut_event.get("confidence") if isinstance(shortcut_event, dict) else None,
                "shortcut_event_no_fire_reason": shortcut_event.get("no_fire_reason") if isinstance(shortcut_event, dict) else None,
                "shortcut_fired": False,
                "skipped_vlm_calls": 0,
            },
        )

    def save_summary(self) -> None:
        if not self.output_path:
            return
        os.makedirs(self.output_path, exist_ok=True)
        rollback_traces = self._all_rollback_traces or self._rollback_traces
        exploration_traces = self._all_exploration_traces or self._exploration_traces
        rollback_total = len(rollback_traces)
        rollback_success = sum(1 for item in rollback_traces if bool(item.get("success")))
        level1_total = sum(1 for item in rollback_traces if item.get("level") == "level1")
        level2_total = sum(1 for item in rollback_traces if item.get("level") == "level2")
        level1_success = sum(
            1 for item in rollback_traces if item.get("level") == "level1" and bool(item.get("success"))
        )
        level2_success = sum(
            1 for item in rollback_traces if item.get("level") == "level2" and bool(item.get("success"))
        )
        failures = [
            item
            for item in rollback_traces
            if not bool(item.get("success"))
        ]
        summary = {
            "exploration_trace_count": len(exploration_traces),
            "rollback_total": rollback_total,
            "rollback_success": rollback_success,
            "rollback_success_rate": (float(rollback_success) / rollback_total if rollback_total else None),
            "rollback_level1_total": level1_total,
            "rollback_level1_success": level1_success,
            "rollback_level1_success_rate": (float(level1_success) / level1_total if level1_total else None),
            "rollback_level2_total": level2_total,
            "rollback_level2_success": level2_success,
            "rollback_level2_success_rate": (float(level2_success) / level2_total if level2_total else None),
            "rollback_failures": failures[:20],
        }
        shortcut_plans = self._all_shortcut_plans or self._shortcut_plans
        shortcut_events = self._all_shortcut_events or self._shortcut_events
        shortcut_shadow_evals = self._all_shortcut_shadow_evals or self._shortcut_shadow_evals
        direct_answer_candidates = self._all_direct_answer_candidates or self._direct_answer_candidates
        fired_events = [item for item in shortcut_events if bool(item.get("fired"))]
        would_fire_events = [item for item in shortcut_events if bool(item.get("would_fire"))]
        shadow_matches = [item for item in shortcut_shadow_evals if bool(item.get("action_match"))]
        plan_with_action = [
            item for item in shortcut_plans
            if not _clean_text(item.get("no_plan_reason"))
        ]
        no_plan_reasons: dict[str, int] = {}
        no_fire_reasons: dict[str, int] = {}
        for item in shortcut_plans:
            reason = _clean_text(item.get("no_plan_reason"))
            if reason:
                no_plan_reasons[reason] = no_plan_reasons.get(reason, 0) + 1
        for item in shortcut_events:
            reason = _clean_text(item.get("no_fire_reason"))
            if reason:
                no_fire_reasons[reason] = no_fire_reasons.get(reason, 0) + 1
        summary["shortcut"] = {
            "mode": self.t2_shortcut_mode,
            "state_mode": self.t2_state_mode,
            "allow_safe_search_input": bool(self.t2_allow_safe_search_input),
            "shortcut_plan_count": len(shortcut_plans),
            "shortcut_candidate_count": len(plan_with_action),
            "shortcut_would_fire_count": len(would_fire_events),
            "active_shortcut_fired_count": len(fired_events),
            "skipped_vlm_calls": sum(1 for item in fired_events if bool(item.get("skipped_vlm_reasoning"))),
            "shortcut_shadow_eval_count": len(shortcut_shadow_evals),
            "shortcut_shadow_action_match_count": len(shadow_matches),
            "shortcut_shadow_precision": (
                float(len(shadow_matches)) / len(shortcut_shadow_evals)
                if shortcut_shadow_evals
                else None
            ),
            "direct_answer_candidate_count": len(direct_answer_candidates),
            "no_plan_reason_distribution": no_plan_reasons,
            "no_fire_reason_distribution": no_fire_reasons,
        }
        summary["state_acquisition"] = dict(self._state_acquisition_metrics)
        for filename, rows in (
            ("shortcut_plans.jsonl", shortcut_plans),
            ("shortcut_events.jsonl", shortcut_events),
            ("shortcut_shadow_eval.jsonl", shortcut_shadow_evals),
            ("direct_answer_candidates.jsonl", direct_answer_candidates),
            ("latency_profile.jsonl", self._latency_profile_events),
        ):
            with open(os.path.join(self.output_path, filename), "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        with open(os.path.join(self.output_path, "state_acquisition_metrics.csv"), "w", encoding="utf-8") as f:
            f.write("metric,value\n")
            for key, value in sorted(self._state_acquisition_metrics.items()):
                f.write(f"{key},{value}\n")
        with open(os.path.join(self.output_path, "exploration_rollup.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        md_lines = [
            "# Exploration Rollup",
            "",
            f"- exploration traces: {summary['exploration_trace_count']}",
            f"- rollback total: {rollback_total}",
            f"- rollback success: {rollback_success}",
            f"- level1: {level1_success}/{level1_total}",
            f"- level2: {level2_success}/{level2_total}",
            f"- shortcut plans: {summary['shortcut']['shortcut_plan_count']}",
            f"- shortcut would_fire: {summary['shortcut']['shortcut_would_fire_count']}",
            f"- active shortcut fired: {summary['shortcut']['active_shortcut_fired_count']}",
            f"- skipped VLM calls: {summary['shortcut']['skipped_vlm_calls']}",
            f"- shadow precision: {summary['shortcut']['shortcut_shadow_precision']}",
            "",
            "## Recent Failures",
        ]
        if failures:
            for item in failures[:10]:
                analysis = item.get("failure_analysis") if isinstance(item.get("failure_analysis"), dict) else {}
                reasons = ", ".join(analysis.get("likely_reasons") or [])
                md_lines.append(
                    f"- mode={item.get('mode')} matched_by={item.get('matched_by')} reasons={reasons or 'unknown'}"
                )
        else:
            md_lines.append("- none")
        with open(os.path.join(self.output_path, "exploration_rollup.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")


class ElementTextAgent(ExplorerElementAgent):
    pass
