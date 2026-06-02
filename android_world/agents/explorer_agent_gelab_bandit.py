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

"""Adaptive speculative explorer for GELAB.

This variant keeps the main GELAB planning loop unchanged, but replaces the
one-shot depth-2 probe with a latency-bounded bandit-style policy:

1. Execute the planned t -> t+1 action speculatively.
2. On the matched t+1 page, probe several safe t+2 candidates.
3. Score candidates with task/planning relevance, information gain, risk, and
   a tiny online reward from whether previous hints were followed.
4. Roll back synchronously before executing the real main action.

The implementation is intentionally conservative: destructive goals are still
skipped by default, and typed probes are kept last because they can mutate form
state before rollback.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any

from android_world.agents.explorer_agent_gelab_light import _clean_text
from android_world.agents.explorer_agent_gelab_light import _env_bool
from android_world.agents.explorer_agent_gelab_light import _env_float
from android_world.agents.explorer_agent_gelab_light import _env_int
from android_world.agents.explorer_agent_gelab_light import _now_hms
from android_world.agents.explorer_agent_gelab_light import ExplorerElementAgent as _LightExplorer
from android_world.env import json_action


def _rate(num: float, den: float, default: float = 0.0) -> float:
    if den <= 0:
        return float(default)
    return float(num) / float(den)


class ExplorerElementAgent(_LightExplorer):
    """GELAB explorer with multi-probe t+2 search and online candidate priors."""

    def __init__(
        self,
        env,
        vllm: Any,
        name: str = "ExplorerBanditAgent",
        output_path: str = "./output/explorer_agent_gelab_bandit",
        **kwargs: Any,
    ):
        super().__init__(
            env=env,
            vllm=vllm,
            name=name,
            output_path=output_path,
            light_explore_max_runs=_env_int("ANDROID_WORLD_LIGHT_EXPLORE_MAX_RUNS", 4),
            light_explore_max_step=_env_int("ANDROID_WORLD_LIGHT_EXPLORE_MAX_STEP", 9),
            light_explore_branch_budget=_env_int("ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_BUDGET", 1),
            light_explore_branch_depth=_env_int("ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_DEPTH", 2),
            **kwargs,
        )
        self.bandit_t2_budget = max(1, _env_int("ANDROID_WORLD_BANDIT_T2_BUDGET", 3))
        self.bandit_stalled_t2_budget = max(
            self.bandit_t2_budget,
            _env_int("ANDROID_WORLD_BANDIT_STALLED_T2_BUDGET", 4),
        )
        self.bandit_secondary_pool = max(
            self.bandit_stalled_t2_budget + 2,
            _env_int("ANDROID_WORLD_BANDIT_SECONDARY_POOL", 8),
        )
        self.bandit_ucb_weight = max(0.0, _env_float("ANDROID_WORLD_BANDIT_UCB_WEIGHT", 0.18))
        self.bandit_follow_weight = max(0.0, _env_float("ANDROID_WORLD_BANDIT_FOLLOW_WEIGHT", 0.35))
        self.bandit_info_gain_weight = max(0.0, _env_float("ANDROID_WORLD_BANDIT_INFO_GAIN_WEIGHT", 0.10))
        self.bandit_enable_negative_context = _env_bool("ANDROID_WORLD_BANDIT_NEGATIVE_CONTEXT", True)
        self.bandit_allow_alternate_when_stalled = _env_bool("ANDROID_WORLD_BANDIT_ALTERNATE_ON_STALL", True)
        self._bandit_stats: dict[str, dict[str, float]] = {}
        self._pending_negative_evidence: list[dict[str, Any]] = []

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home=go_home)
        self._bandit_stats = {}
        self._pending_negative_evidence = []

    def _candidate_policy_key(self, candidate: dict[str, Any]) -> str:
        action_kind = _clean_text(candidate.get("action_kind") or "click").lower()
        label = _clean_text(candidate.get("label") or candidate.get("merged")).lower()
        label = re.sub(r"[^a-z0-9 ]+", " ", label)
        tokens = [t for t in label.split() if len(t) >= 3][:4]
        a11y = candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {}
        cls = _clean_text(a11y.get("class_name")).split(".")[-1].lower()
        return "|".join([action_kind, cls, " ".join(tokens) or "unlabeled"])

    def _candidate_hit_by_action(self, action: json_action.JSONAction, candidate: dict[str, Any]) -> bool:
        action_kind = _clean_text(candidate.get("action_kind") or "click").lower()
        if action_kind == "type":
            expected = _clean_text(candidate.get("text"))
            actual = _clean_text(getattr(action, "text", ""))
            return bool(expected and actual and expected == actual)
        if action.action_type != json_action.CLICK:
            return False
        center = candidate.get("center")
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            return False
        try:
            dist = math.dist((float(action.x), float(action.y)), (float(center[0]), float(center[1])))
        except Exception:  # pylint: disable=broad-exception-caught
            return False
        if dist <= 120.0:
            return True
        a11y = candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {}
        bbox = a11y.get("bbox") if isinstance(a11y.get("bbox"), dict) else {}
        try:
            return (
                float(bbox["x_min"]) <= float(action.x) <= float(bbox["x_max"])
                and float(bbox["y_min"]) <= float(action.y) <= float(bbox["y_max"])
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return False

    def _after_reasoning_action(
        self,
        goal: str,
        step_idx: int,
        action: json_action.JSONAction,
        matched_prompt_results: list[dict[str, Any]],
    ) -> None:
        del goal, step_idx
        if not matched_prompt_results:
            return
        for item in matched_prompt_results:
            if not isinstance(item, dict):
                continue
            candidate = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
            if not candidate:
                continue
            key = self._candidate_policy_key(candidate)
            stats = self._bandit_stats.setdefault(key, {"shown": 0.0, "followed": 0.0, "missed": 0.0})
            stats["shown"] += 1.0
            if self._candidate_hit_by_action(action, candidate):
                stats["followed"] += 1.0
            else:
                stats["missed"] += 1.0

    def _candidate_bandit_bonus(self, candidate: dict[str, Any]) -> float:
        key = self._candidate_policy_key(candidate)
        stats = self._bandit_stats.get(key, {})
        shown = float(stats.get("shown") or 0.0)
        followed = float(stats.get("followed") or 0.0)
        total_shown = sum(float(v.get("shown") or 0.0) for v in self._bandit_stats.values())
        exploit = _rate(followed + 0.35, shown + 1.0, default=0.35)
        explore = math.sqrt(math.log(total_shown + 2.0) / (shown + 1.0))
        return self.bandit_follow_weight * exploit + self.bandit_ucb_weight * explore

    def _goal_mode(self, goal: str) -> str:
        low = _clean_text(goal).lower()
        if any(token in low for token in ("contact", "recipe", "event", "note", "folder", "file name", "named")):
            return "form"
        if any(token in low for token in ("task.html", "open with", "downloads", "file manager", "vlc", "playlist")):
            return "navigation"
        if low.startswith(("what ", "is ", "do i ", "how many")) or "answer with" in low:
            return "qa"
        return "general"

    @staticmethod
    def _generic_candidate_penalty(label: str) -> float:
        low = re.sub(r"^planned:\s*", "", _clean_text(label).lower()).strip()
        if low in {"home", "files", "content", "item", "button", "description", "categories"}:
            return 0.45
        if re.fullmatch(r"(home|back|up|more options|menu|search)", low):
            return 0.35
        return 0.0

    def _secondary_policy_score(
        self,
        candidate: dict[str, Any],
        goal: str,
        planning_text: str,
        depth1_labels: set[str],
    ) -> tuple[float, dict[str, float]]:
        mode = self._goal_mode(goal)
        label = _clean_text(candidate.get("label") or candidate.get("merged"))
        action_kind = _clean_text(candidate.get("action_kind") or "click").lower()
        relevance = float(candidate.get("relevance") or 0.0)
        planning_relevance = float(candidate.get("planning_relevance") or 0.0)
        token_relevance = float(candidate.get("token_relevance") or 0.0)
        base_score = float(candidate.get("score") or 0.0)
        merged = _clean_text(candidate.get("merged") or label).lower()
        goal_tokens = set(self._goal_tokens(goal))
        label_tokens = set(re.findall(r"[a-z0-9]{3,}", merged))
        exact_goal_overlap = len(goal_tokens.intersection(label_tokens))
        mode_bonus = 0.0
        if mode == "form":
            if action_kind == "type":
                mode_bonus += 1.2
            if any(x in merged for x in ("name", "title", "phone", "number", "date", "time", "description")):
                mode_bonus += 0.35
        elif mode == "navigation":
            if exact_goal_overlap:
                mode_bonus += 0.45
            if any(x in merged for x in ("chrome", "open", "downloads", "dcim", "vlcvideos", "task.html")):
                mode_bonus += 0.35
        elif mode == "qa":
            if any(x in merged for x in ("search", "calendar", "tasks", "note")):
                mode_bonus += 0.25
        novelty = 0.0
        if label and label.lower() not in depth1_labels:
            novelty = 0.15
        penalty = self._generic_candidate_penalty(label)
        if self._is_commit_probe_label(label):
            penalty += 0.8
        if self._is_risky_probe_text(merged):
            penalty += 1.0
        score = (
            0.45 * base_score
            + 1.15 * max(relevance, token_relevance)
            + 0.75 * planning_relevance
            + 0.20 * exact_goal_overlap
            + mode_bonus
            + novelty
            + self._candidate_bandit_bonus(candidate)
            - penalty
        )
        components = {
            "base": base_score,
            "relevance": relevance,
            "planning_relevance": planning_relevance,
            "token_relevance": token_relevance,
            "goal_overlap": float(exact_goal_overlap),
            "mode_bonus": mode_bonus,
            "novelty": novelty,
            "bandit_bonus": self._candidate_bandit_bonus(candidate),
            "penalty": penalty,
            "score": score,
        }
        return float(score), components

    def _secondary_probe_candidates(
        self,
        state: Any,
        goal: str,
        planning_text: str,
        first_candidate: dict[str, Any],
        budget: int,
    ) -> list[dict[str, Any]]:
        depth1_labels = {_clean_text(x).lower() for x in self._state_semantic_summary(state, limit=40)}
        old_branch_budget = self.light_explore_branch_budget
        try:
            self.light_explore_branch_budget = max(
                int(self.light_explore_branch_budget),
                int(math.ceil(float(self.bandit_secondary_pool) / 2.0)),
            )
            raw_candidates = self._collect_probe_candidates(state, goal, planning_text=planning_text)
        finally:
            self.light_explore_branch_budget = old_branch_budget
        typed = self._typed_probe_candidate(state, goal, planning_text=planning_text)
        if typed is not None:
            raw_candidates.append(typed)
        raw_candidates.extend(self._field_typed_candidates(state, goal, planning_text=planning_text))
        first_center = first_candidate.get("center") if isinstance(first_candidate, dict) else None
        scored: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in raw_candidates[: self.bandit_secondary_pool]:
            center = candidate.get("center")
            if first_center is not None and center == first_center:
                continue
            key = _clean_text(candidate.get("key")) or self._candidate_policy_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            action_kind = _clean_text(candidate.get("action_kind") or "click").lower()
            if action_kind != "type" and self._is_commit_probe_label(str(candidate.get("label") or "")):
                continue
            score, components = self._secondary_policy_score(
                candidate,
                goal=goal,
                planning_text=planning_text,
                depth1_labels=depth1_labels,
            )
            if score <= 0.05:
                continue
            item = dict(candidate)
            item["score"] = float(score)
            item["policy_components"] = components
            scored.append(item)
        scored.sort(
            key=lambda item: (
                1 if _clean_text(item.get("action_kind") or "click").lower() != "type" else 0,
                float(item.get("score") or 0.0),
            ),
            reverse=True,
        )
        click_like = [c for c in scored if _clean_text(c.get("action_kind") or "click").lower() != "type"]
        typed_like = [c for c in scored if _clean_text(c.get("action_kind") or "click").lower() == "type"]
        selected = click_like[: max(0, budget - min(1, len(typed_like)))]
        if typed_like and len(selected) < budget:
            selected.append(typed_like[0])
        if not selected:
            selected = scored[:budget]
        return selected[:budget]

    def _extract_goal_field_values(self, goal: str) -> dict[str, str]:
        text = _clean_text(goal)
        values: dict[str, str] = {}
        contact = re.search(
            r"contact\s+for\s+(.+?)\.\s*Their\s+number\s+is\s+([+0-9 ()-]+)",
            text,
            flags=re.IGNORECASE,
        )
        if contact:
            name = _clean_text(contact.group(1))
            phone = _clean_text(contact.group(2)).rstrip(".")
            parts = name.split()
            if parts:
                values["first"] = parts[0]
                values["first name"] = parts[0]
                values["name"] = name
            if len(parts) > 1:
                values["last"] = " ".join(parts[1:])
                values["last name"] = " ".join(parts[1:])
            if phone:
                values["phone"] = phone
                values["number"] = phone
        titled = re.search(r"\btitle(?:d)?\s+['\"]([^'\"]{2,120})['\"]", text, flags=re.IGNORECASE)
        if titled:
            values["title"] = _clean_text(titled.group(1))
        description = re.search(r"\bdescription\s+['\"]([^'\"]{2,200})['\"]", text, flags=re.IGNORECASE)
        if description:
            values["description"] = _clean_text(description.group(1))
        named = self._extract_goal_entry_text(goal)
        if named:
            values.setdefault("name", named)
            values.setdefault("title", named)
        return values

    def _field_typed_candidates(self, state: Any, goal: str, planning_text: str = "") -> list[dict[str, Any]]:
        values = self._extract_goal_field_values(goal)
        if not values:
            return []
        candidates: list[dict[str, Any]] = []
        used_fields: set[str] = set()
        for idx, element in enumerate(list(getattr(state, "ui_elements", None) or [])):
            class_name = _clean_text(getattr(element, "class_name", "")).lower()
            resource_id = _clean_text(getattr(element, "resource_id", "")).lower()
            if not (bool(getattr(element, "is_editable", False)) or "edittext" in class_name or "edit_text" in resource_id):
                continue
            center = self._safe_center_from_element(element) or (0, 0)
            label = (
                _clean_text(getattr(element, "text", ""))
                or _clean_text(getattr(element, "content_description", ""))
                or self._normalize_resource_id(getattr(element, "resource_id", ""))
                or "editable field"
            )
            label_low = label.lower()
            selected_value = ""
            selected_field = ""
            field_items = sorted(
                values.items(),
                key=lambda item: (
                    0 if item[0] in {"first name", "last name", "phone", "number", "description", "title"} else 1,
                    -len(item[0]),
                ),
            )
            for field, value in field_items:
                if field in label_low and field not in used_fields:
                    selected_field = field
                    selected_value = value
                    break
            if not selected_value:
                continue
            used_fields.add(selected_field)
            planning_relevance = self._lexical_overlap_score(
                label,
                goal_tokens=self._goal_tokens(planning_text),
                app_keywords=[],
            )
            candidates.append(
                {
                    "index": idx,
                    "element": element,
                    "center": center,
                    "label": f'type "{selected_value}" into {label}',
                    "score": 10.0,
                    "relevance": 1.0,
                    "planning_relevance": float(max(0.5, planning_relevance)),
                    "token_relevance": 1.0,
                    "visits": 0,
                    "key": f"type-field|{selected_field}|{selected_value[:64]}",
                    "merged": f"{label} {selected_value}",
                    "a11y": self._element_trace(element, idx),
                    "action_kind": "type",
                    "text": selected_value,
                    "clear_text": True,
                    "field_name": selected_field,
                }
            )
        return candidates

    def _execute_probe_candidate(self, candidate: dict[str, Any]) -> None:
        center = candidate.get("center")
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            center = (0, 0)
        if _clean_text(candidate.get("action_kind") or "click").lower() == "type":
            self.env.execute_action(
                json_action.JSONAction(
                    action_type=json_action.INPUT_TEXT,
                    x=int(center[0]) if int(center[0]) > 0 else None,
                    y=int(center[1]) if int(center[1]) > 0 else None,
                    text=_clean_text(candidate.get("text")),
                    clear_text=bool(candidate.get("clear_text")),
                )
            )
        else:
            self.env.execute_action(
                json_action.JSONAction(
                    action_type=json_action.CLICK,
                    x=int(center[0]),
                    y=int(center[1]),
                )
            )

    def _snapshot_matches(self, state: Any, snapshot_step: dict[str, Any]) -> tuple[bool, str]:
        return self._state_matches_explored_step(
            state,
            self._foreground_activity_name(),
            snapshot_step,
        )

    def _rollback_to_depth1(
        self,
        depth1_step: dict[str, Any],
        step_idx: int,
        max_back: int = 2,
    ) -> dict[str, Any]:
        start = time.time()
        result = {
            "success": False,
            "mode": "back_to_depth1_failed",
            "level": "inner",
            "back_presses": 0,
            "matched_by": "",
            "latency_ms": 0.0,
        }
        for i in range(max(0, int(max_back)) + 1):
            curr_state = self.env.get_state(wait_to_stabilize=True)
            same, matched_by = self._snapshot_matches(curr_state, depth1_step)
            if same:
                result.update(
                    {
                        "success": True,
                        "mode": "back_to_depth1",
                        "matched_by": matched_by,
                        "latency_ms": float(max(0.0, time.time() - start) * 1000.0),
                    }
                )
                return result
            if i >= int(max_back):
                break
            self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK))
            result["back_presses"] += 1
            print(
                f"[ROLLBACK {_now_hms()}] step: {step_idx + 1} "
                f"inner_depth2_back#{i + 1}/{max_back} matched_by={matched_by}"
            )
        result["latency_ms"] = float(max(0.0, time.time() - start) * 1000.0)
        return result

    def _info_gain_reward(
        self,
        depth1_labels: list[str],
        depth2_labels: list[str],
        goal: str,
        changed: bool,
    ) -> tuple[float, dict[str, Any]]:
        before = {_clean_text(x).lower() for x in depth1_labels if _clean_text(x)}
        after = {_clean_text(x).lower() for x in depth2_labels if _clean_text(x)}
        new_labels = sorted(after.difference(before))[:8]
        goal_tokens = set(self._goal_tokens(goal))
        after_text = " ".join(after)
        goal_hits = [token for token in goal_tokens if token and token in after_text]
        reward = (0.8 if changed else -0.4)
        reward += self.bandit_info_gain_weight * float(len(new_labels))
        reward += 0.25 * float(len(goal_hits))
        return float(reward), {
            "new_label_count": len(new_labels),
            "new_labels": new_labels,
            "goal_hits": goal_hits[:8],
            "changed_bonus": 0.8 if changed else -0.4,
            "reward": reward,
        }

    def _remember_negative_no_effect(
        self,
        goal: str,
        root_activity: str,
        root_hash: int,
        root_state: Any,
        candidate: dict[str, Any],
        step_idx: int,
    ) -> None:
        if not self.bandit_enable_negative_context:
            return
        self._pending_negative_evidence = [
            {
                "goal": goal,
                "source_step": int(step_idx + 1),
                "root_activity": root_activity,
                "root_hash": root_hash,
                "root_labels": self._state_semantic_summary(root_state, limit=30),
                "label": _clean_text(candidate.get("label") or "planned click"),
                "center": candidate.get("center"),
            }
        ]

    def _match_pending_speculative_exploration(
        self,
        goal: str,
        step_idx: int,
        state: Any,
        current_activity: str,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        context, selected, trace = super()._match_pending_speculative_exploration(
            goal=goal,
            step_idx=step_idx,
            state=state,
            current_activity=current_activity,
        )
        negative_lines: list[str] = []
        remaining: list[dict[str, Any]] = []
        for item in self._pending_negative_evidence:
            fake_step = {
                "after_activity": item.get("root_activity"),
                "after_hash": item.get("root_hash"),
                "observed_elements": item.get("root_labels") or [],
            }
            matched, matched_by = self._state_matches_explored_step(state, current_activity, fake_step)
            if matched:
                center = item.get("center")
                center_text = ""
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    center_text = f" near [{int(center[0])}, {int(center[1])}]"
                negative_lines.append(
                    f"- Previous speculative click `{item.get('label')}`{center_text} did not change the screen "
                    f"({matched_by}). Avoid repeating that exact target; try another visible target, long press, "
                    "or an open-with/menu path if it fits the task."
                )
            else:
                remaining.append(item)
        self._pending_negative_evidence = remaining
        if negative_lines:
            negative_context = "Negative exploration evidence from the previous step:\n" + "\n".join(negative_lines)
            context = f"{context}\n\n{negative_context}" if context else negative_context
            trace = dict(trace)
            trace["negative_context"] = negative_context
            trace["status"] = "matched_with_negative" if selected else "negative_only"
        return context, selected, trace

    def _build_prompt_context_from_state_aligned_matches(
        self,
        matches: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        matches.sort(
            key=lambda item: (
                float(((item.get("reward") or {}).get("reward") if isinstance(item.get("reward"), dict) else 0.0)),
                float(item.get("score") or 0.0),
                int(item.get("depth_reached") or 0),
            ),
            reverse=True,
        )
        selected = matches[: int(self.light_explore_prompt_result_limit)]
        lines: list[str] = []
        selected_traces: list[dict[str, Any]] = []
        for rank, item in enumerate(selected, start=1):
            next_candidate = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
            next_label = _clean_text(next_candidate.get("label")) or _clean_text(item.get("next_label")) or "that element"
            action_kind = _clean_text(next_candidate.get("action_kind") or "click").lower()
            center = next_candidate.get("center")
            center_text = ""
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                center_text = f" near pixel [{int(center[0])}, {int(center[1])}]"
            if action_kind == "type":
                action_text = f"TYPE `{_clean_text(next_candidate.get('text'))}` into `{next_label}`"
            else:
                action_text = f"CLICK `{next_label}`{center_text}"
            reward = item.get("reward") if isinstance(item.get("reward"), dict) else {}
            new_labels = "; ".join(_clean_text(x) for x in list(reward.get("new_labels") or [])[:4] if _clean_text(x))
            observed = [_clean_text(x) for x in list(item.get("observed_elements") or []) if _clean_text(x)][:5]
            observed_text = "; ".join(observed) if observed else "no compact labels captured"
            evidence = f"new labels: {new_labels}" if new_labels else f"observed: {observed_text}"
            line = (
                f"{rank}. Matched t+1 path `{item.get('matched_prefix')}` ({item.get('matched_by')}). "
                f"Best t+2 candidate: {action_text}. Evidence score={float(item.get('score') or 0.0):.2f}; "
                f"{evidence}. Use it only if it supports the task; rollback={item.get('rollback_level')}/"
                f"{item.get('rollback_mode')} verified."
            )
            trace = dict(item)
            trace["rank"] = rank
            trace["prompt_line"] = line
            selected_traces.append(trace)
            lines.append(line)
        if not lines:
            return "", []
        return (
            "State-aligned multi-probe exploration from the previous step matched the current screen. "
            "Several t+2 candidates were explored and ranked; prefer the most task-relevant evidence:\n"
            + "\n".join(lines),
            selected_traces,
        )

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
    ) -> dict[str, Any]:
        self._last_probe_candidates = []
        trace: dict[str, Any] = {
            "step": int(step_idx + 1),
            "goal": goal,
            "mode": "post_planning_bandit_multi_t2",
            "planned_action": current_action.__dict__ if current_action is not None else None,
            "status": "not_started",
            "trigger_reason": "",
            "root_activity": "",
            "root_hash": None,
            "root_screenshot": "",
            "root_a11y": [],
            "candidate_count": 0,
            "candidates": [],
            "selected_targets": [],
            "observations": [],
            "rollbacks": [],
            "selected_prompt_results": [],
            "prompt_context": "",
            "speculative_results": [],
            "speculative_context": "",
            "available_for_next_prompt": False,
            "started_at": time.time(),
            "latency_ms": 0.0,
            "bandit_config": {
                "t2_budget": self.bandit_t2_budget,
                "stalled_t2_budget": self.bandit_stalled_t2_budget,
                "secondary_pool": self.bandit_secondary_pool,
            },
        }
        try:
            action_skip_reason = self._planned_action_skip_reason(current_action)
            if action_skip_reason:
                trace["status"] = "skipped"
                trace["trigger_reason"] = action_skip_reason
                return trace

            if root_state is None:
                root_state = self.env.get_state(wait_to_stabilize=True)
            if root_hash is None:
                root_hash = self._state_hash(root_state)
            if not root_activity:
                root_activity = self._foreground_activity_name()
            trace["root_activity"] = root_activity
            trace["root_hash"] = root_hash
            if self._has_keyboard_ui(root_state):
                trace["status"] = "skipped"
                trace["trigger_reason"] = "keyboard_visible"
                return trace
            should_run, reason = self._should_run_light_exploration(goal, step_idx, root_activity, page_stalled)
            trace["trigger_reason"] = reason
            print(
                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                f"bandit_should_run={should_run} reason={reason} page_stalled={page_stalled}"
            )
            if not should_run:
                trace["status"] = "skipped"
                return trace

            trace["root_screenshot"] = self._save_trace_screenshot(
                goal,
                root_state,
                f"explore_step_{step_idx + 1:02d}_root.png",
            )
            trace["root_a11y"] = self._state_a11y_trace(root_state)

            candidates = self._collect_probe_candidates(root_state, goal, planning_text=planning_text)
            candidates = self._rank_candidates_for_planned_action(candidates, current_action)
            planned_candidate = self._planned_click_candidate_from_action(root_state, current_action)
            if planned_candidate is not None:
                planned_key = _clean_text(planned_candidate.get("key"))
                if planned_key and planned_key in self._no_effect_probe_keys and not page_stalled:
                    trace["status"] = "skipped"
                    trace["trigger_reason"] = "planned_anchor_previous_no_effect"
                    return trace
                if (
                    self.light_explore_require_a11y_anchor
                    and int(planned_candidate.get("index") or -1) < 0
                    and not page_stalled
                ):
                    trace["status"] = "skipped"
                    trace["trigger_reason"] = "no_a11y_planned_anchor"
                    return trace
                candidates.insert(0, planned_candidate)
            if self.light_explore_planned_only:
                planned_candidates = [c for c in candidates if bool(c.get("is_planned_action"))]
                if not planned_candidates:
                    trace["status"] = "skipped"
                    trace["trigger_reason"] = "no_planned_anchor"
                    return trace
                candidates = planned_candidates
            elif page_stalled and self.bandit_allow_alternate_when_stalled:
                candidates = candidates[: max(1, int(self.light_explore_branch_budget))]
            self._last_probe_candidates = list(candidates)
            trace["candidate_count"] = int(len(candidates))
            trace["candidates"] = [self._candidate_trace(c) for c in candidates]
            if not candidates:
                trace["status"] = "no_candidates"
                return trace

            first_candidate = self._choose_probe_candidate(candidates)
            if first_candidate is None:
                trace["status"] = "no_branch_candidates"
                return trace
            trace["selected_targets"] = [self._candidate_trace(first_candidate)]
            first_center = first_candidate.get("center")
            if not isinstance(first_center, (list, tuple)) or len(first_center) < 2:
                trace["status"] = "no_branch_candidates"
                return trace

            first_label = _clean_text(first_candidate.get("label")) or "planned candidate"
            print(
                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} bandit_depth1 click={first_label} "
                f"center={[int(first_center[0]), int(first_center[1])]}"
            )
            self.env.execute_action(
                json_action.JSONAction(
                    action_type=json_action.CLICK,
                    x=int(first_center[0]),
                    y=int(first_center[1]),
                )
            )
            state_after_first = self.env.get_state(wait_to_stabilize=True)
            changed1, activity1, hash_diff1 = self._probe_page_changed(root_activity, root_hash, state_after_first)
            if not changed1:
                no_effect_key = _clean_text(first_candidate.get("key"))
                if no_effect_key:
                    self._no_effect_probe_keys.add(no_effect_key)
                self._remember_negative_no_effect(goal, root_activity, root_hash, root_state, first_candidate, step_idx)
            depth1_step = {
                "depth": 1,
                "candidate": self._candidate_trace(first_candidate),
                "changed": bool(changed1),
                "after_activity": activity1,
                "after_hash": self._state_hash(state_after_first),
                "hash_diff_from_root": hash_diff1,
                "screenshot": self._save_trace_screenshot(
                    goal,
                    state_after_first,
                    f"explore_step_{step_idx + 1:02d}_b1_d1.png",
                ),
                "observed_elements": self._state_semantic_summary(state_after_first),
                "a11y": self._state_a11y_trace(state_after_first, limit=12),
            }
            print(
                f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                f"bandit_depth1 changed={changed1} hash_diff={hash_diff1} after={activity1}"
            )

            observations: list[dict[str, Any]] = []
            all_rollback_success = True
            if changed1 and int(self.light_explore_branch_depth) >= 2:
                budget = self.bandit_stalled_t2_budget if page_stalled else self.bandit_t2_budget
                secondary_candidates = self._secondary_probe_candidates(
                    state_after_first,
                    goal=goal,
                    planning_text=planning_text,
                    first_candidate=first_candidate,
                    budget=budget,
                )
                trace["selected_t2_targets"] = [self._candidate_trace(c) for c in secondary_candidates]
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    f"bandit_t2_candidates={len(secondary_candidates)} budget={budget} "
                    f"shortlist={[(_clean_text(c.get('label')), round(float(c.get('score') or 0), 2)) for c in secondary_candidates]}"
                )
                for t2_idx, second_candidate in enumerate(secondary_candidates, start=1):
                    second_center = second_candidate.get("center")
                    if not isinstance(second_center, (list, tuple)) or len(second_center) < 2:
                        continue
                    second_label = _clean_text(second_candidate.get("label")) or f"t2_candidate_{t2_idx}"
                    action_kind = _clean_text(second_candidate.get("action_kind") or "click").lower()
                    print(
                        f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                        f"bandit_depth2#{t2_idx} {action_kind}={second_label} "
                        f"center={[int(second_center[0]), int(second_center[1])]}"
                    )
                    self._execute_probe_candidate(second_candidate)
                    state_after_second = self.env.get_state(wait_to_stabilize=True)
                    changed2, activity2, hash_diff2 = self._probe_page_changed(activity1, self._state_hash(state_after_first), state_after_second)
                    labels2 = self._state_semantic_summary(state_after_second)
                    reward_value, reward = self._info_gain_reward(
                        depth1_step.get("observed_elements") or [],
                        labels2,
                        goal=goal,
                        changed=bool(changed2),
                    )
                    depth2_step = {
                        "depth": 2,
                        "candidate": self._candidate_trace(second_candidate),
                        "changed": bool(changed2),
                        "after_activity": activity2,
                        "after_hash": self._state_hash(state_after_second),
                        "hash_diff_from_root": hash_diff2,
                        "screenshot": self._save_trace_screenshot(
                            goal,
                            state_after_second,
                            f"explore_step_{step_idx + 1:02d}_b1_d2_{t2_idx}.png",
                        ),
                        "observed_elements": list(labels2),
                        "a11y": self._state_a11y_trace(state_after_second, limit=12),
                    }
                    total_score = float(first_candidate.get("score") or 0.0) + float(second_candidate.get("score") or 0.0) + reward_value
                    observations.append(
                        {
                            "branch_id": int(t2_idx),
                            "labels": [first_label, second_label],
                            "changed": bool(changed1 or changed2),
                            "after_activity": activity2,
                            "score": float(total_score),
                            "depth_reached": 2,
                            "observed_elements": list(labels2),
                            "steps": [dict(depth1_step), dict(depth2_step)],
                            "rollback": {},
                            "inner_rollback": {},
                            "reward": reward,
                            "policy_components": second_candidate.get("policy_components") or {},
                        }
                    )
                    print(
                        f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                        f"bandit_depth2#{t2_idx} changed={changed2} reward={reward_value:.2f} after={activity2}"
                    )
                    inner = self._rollback_to_depth1(depth1_step, step_idx=step_idx, max_back=2)
                    observations[-1]["inner_rollback"] = dict(inner)
                    if not bool(inner.get("success")):
                        all_rollback_success = False
                        print(
                            f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                            f"bandit_inner_rollback_failed={inner}"
                        )
                        break
                    state_after_first = self.env.get_state(wait_to_stabilize=True)

            replay_actions = self._select_replay_actions_for_probe(current_action=None)
            rollback_info = self._rollback_to_probe_root(
                root_activity=root_activity,
                root_hash=root_hash,
                replay_actions=replay_actions,
                step_idx=step_idx,
            )
            trace["rollbacks"].append(dict(rollback_info))
            all_rollback_success = bool(all_rollback_success and rollback_info.get("success"))
            if not observations:
                observations.append(
                    {
                        "branch_id": 1,
                        "labels": [first_label],
                        "changed": bool(changed1),
                        "after_activity": activity1,
                        "score": float(first_candidate.get("score") or 0.0),
                        "depth_reached": 1,
                        "observed_elements": list(depth1_step.get("observed_elements") or []),
                        "steps": [dict(depth1_step)],
                        "rollback": dict(rollback_info),
                        "reward": {"reward": 0.0, "new_labels": [], "goal_hits": []},
                    }
                )
            else:
                for obs in observations:
                    obs["rollback"] = dict(rollback_info)

            self._light_explore_runs += 1
            trace["observations"] = observations
            has_depth2_observation = any(
                any(int(step.get("depth") or 0) == 2 and bool(step.get("changed")) for step in obs.get("steps") or [])
                for obs in observations
            )
            speculative_context, speculative_results = self._build_prompt_context_from_observations(observations)
            if not all_rollback_success or not has_depth2_observation:
                speculative_context = ""
                speculative_results = []
            trace["speculative_results"] = speculative_results
            trace["speculative_context"] = speculative_context
            trace["selected_prompt_results"] = []
            trace["prompt_context"] = ""
            trace["status"] = "completed" if all_rollback_success else "rollback_failed"
            trace["rollback_success"] = bool(all_rollback_success)
            trace["available_for_next_prompt"] = bool(all_rollback_success and has_depth2_observation)
            if not all_rollback_success:
                self.enable_light_exploration = False
            if trace["available_for_next_prompt"]:
                self._pending_speculative_traces = [trace]
                print(
                    f"[EXPLORE {_now_hms()}] step: {step_idx + 1} "
                    f"bandit_speculative_results_cached items={len(speculative_results)}"
                )
            else:
                print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} bandit_no_prompt_context")
            return trace
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"[EXPLORE {_now_hms()}] step: {step_idx + 1} bandit_explore_failed: {exc}")
            self._light_explore_runs += 1
            trace["status"] = "exception"
            trace["error"] = _clean_text(exc)
            return trace
        finally:
            trace["latency_ms"] = float(max(0.0, time.time() - float(trace.get("started_at") or time.time())) * 1000.0)
            self._append_exploration_trace(goal, trace)
