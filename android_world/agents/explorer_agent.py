"""Explorer-style element agent for AndroidWorld.

This file keeps the original tool-call parser/action mapping style, and adds
parallel threaded depth-first exploration that runs during model reasoning,
then rolls back before executing the reasoning action.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import threading
import time
from collections import deque
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from android_world.agents import base_agent
from android_world.agents import seeact_utils
from android_world.agents.explorer_agent_constants import (
    SYSTEM_PROMPT,
    _CHOICE_HINT_TOKENS,
    _DIALOG_HINT_TOKENS,
    _DISMISS_ACTION_KEYWORDS,
    _INFO_ONLY_TEXT_HINTS,
    _KEYBOARD_HINT_TOKENS,
    _LOW_VALUE_CLASS_HINTS,
    _LOW_VALUE_TEXT_HINTS,
    _NAV_HELPFUL_KEYWORDS,
    _RISKY_KEYWORDS,
    _SUBMIT_ACTION_KEYWORDS,
    _TASK_INPUT_KEYWORDS,
    _TASK_SELECT_KEYWORDS,
)
from android_world.agents.explorer_agent_utils import (
    _element_hint_compact_label,
    _extract_task_queries,
    _hash_diff,
    _looks_like_back_intent,
    _mae_small,
    _normalize_space,
    _phash_pixels,
    _pixel_delta,
    _safe_int,
    _target_index,
    _to_data_url,
    _to_json_action,
    _tokenize,
    parse_tool_call,
)
from android_world.env import adb_utils
from android_world.env import actuation
from android_world.env import interface
from android_world.env import json_action

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pylint: disable=broad-exception-caught
    SentenceTransformer = None




@dataclasses.dataclass(frozen=True)
class ExplorerHint:
    index: int
    score: float
    label: str


@dataclasses.dataclass
class CandidateScore:
    index: int
    key: str
    text: str
    score: float
    task_similarity: float
    runtime_similarity: float
    similarity: float
    visits: float
    is_clickable: bool




class ExplorerElementAgent(base_agent.EnvironmentInteractingAgent):
    """AndroidWorld agent with parallel threaded exploration."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        vllm: Any,
        name: str = "ExplorerElementAgent",
        max_history: int = 10,
        max_hints: int = 6,
        no_effect_delta_threshold: float = 1.2,
        force_explore_after_repeats: int = 2,
        explore_max_steps: int = 10,
        explore_max_depth: int = 2,
        explore_leaf_width: int = 3,
        explore_max_branches: int | None = None,
        explore_action_pause_sec: float = 0.25,
        reasoning_sleep_sec: float = 0.0,
        embed_model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2",
        verbose_step_logs: bool = True,
        reasoning_preview_chars: int = 180,
        log_full_ui_tree_every_n_steps: int = 0,
        model_coordinate_mode: str = "auto",
        explore_mask_output_dir: str = "explore_k1_masks",
        save_explore_masks: bool = True,
        trace_output_dir: str = "explorer_traces",
        prompt_ui_element_limit: int | None = None,
        explore_warmup_timeout_sec: float = 1.5,
        explore_min_actions_before_reasoning: int = 1,
    ):
        super().__init__(env, name)
        # Backward-compatible arg retained to avoid breaking older configs.
        _ = prompt_ui_element_limit
        self.vllm = vllm
        self.max_history = max_history
        self.max_hints = max_hints
        self.no_effect_delta_threshold = no_effect_delta_threshold
        self.force_explore_after_repeats = force_explore_after_repeats

        self.explore_max_steps = max(1, int(explore_max_steps))
        self.explore_max_depth = max(1, int(explore_max_depth))
        self.explore_leaf_width = max(1, int(explore_leaf_width))
        self.explore_max_branches = explore_max_branches
        self.explore_action_pause_sec = max(0.05, float(explore_action_pause_sec))
        # Disable simulated reasoning sleep; exploration should be event-driven.
        self.reasoning_sleep_sec = 0.0
        self.embed_model_name = embed_model_name
        self.verbose_step_logs = bool(verbose_step_logs)
        self.reasoning_preview_chars = max(60, int(reasoning_preview_chars))
        self.log_full_ui_tree_every_n_steps = max(0, int(log_full_ui_tree_every_n_steps))
        self.model_coordinate_mode = str(model_coordinate_mode or "auto").strip().lower()
        self.save_explore_masks = bool(save_explore_masks)
        self.explore_mask_output_dir = str(explore_mask_output_dir or "explore_k1_masks").strip()
        if self.save_explore_masks and self.explore_mask_output_dir:
            try:
                os.makedirs(self.explore_mask_output_dir, exist_ok=True)
            except Exception:  # pylint: disable=broad-exception-caught
                self.save_explore_masks = False
        self.trace_output_dir = str(trace_output_dir or "explorer_traces").strip()
        if self.trace_output_dir:
            try:
                os.makedirs(self.trace_output_dir, exist_ok=True)
            except Exception:  # pylint: disable=broad-exception-caught
                self.trace_output_dir = ""

        self.actions: list[dict[str, Any]] = []
        self.history: list[str] = []
        self._recent_indices: deque[int] = deque(maxlen=50)
        self._last_pixels: np.ndarray | None = None
        self._last_action_text: str = ""
        self._no_effect_repeat = 0
        self._execution_feedback = ""

        self._ui_lock = threading.Lock()
        self._rollback_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._explore_stop_event = threading.Event()
        self._explore_progress_event = threading.Event()
        self._explore_thread: threading.Thread | None = None
        self._explore_thread_stop_clean: bool = True
        self._explore_action_count = 0
        self._explore_action_count_lock = threading.Lock()

        self._explore_iteration_candidates: list[dict[str, Any]] = []
        self._pending_explore_payload: dict[str, Any] | None = None
        self._last_clue_debug: dict[str, Any] = {}

        self._explore_root_hash: int | None = None
        self._explore_root_pixels: np.ndarray | None = None
        self._explore_root_activity: str | None = None
        self._replay_action_history: list[dict[str, Any]] = []
        self._reasoning_action_history: list[dict[str, Any]] = []
        self._branch_action_history: list[dict[str, Any]] = []

        self._clicked_bounds: set[str] = set()
        self._bound_visit_count: dict[str, float] = {}
        self._bound_effect_ema: dict[str, float] = {}
        self._bound_effect_count: dict[str, int] = {}
        self._bound_seen_count: dict[str, int] = {}
        self._bound_skip_count: dict[str, int] = {}
        self._recent_clicked_bounds: deque[str] = deque(maxlen=128)
        self._recent_clicked_regions: deque[str] = deque(maxlen=128)
        self._recent_clicked_window = 12
        self._last_filter_stats: dict[str, Any] = {}

        self._emb_cache: dict[str, np.ndarray] = {}
        self._embed_model = None
        self._embed_ready = False

        self._task_goal: str | None = None
        self._task_start_ts: float | None = None
        self._task_trace_dir: str | None = None
        self._task_step_latencies: list[float] = []
        self.explore_warmup_timeout_sec = max(0.0, float(explore_warmup_timeout_sec))
        self.explore_min_actions_before_reasoning = max(0, int(explore_min_actions_before_reasoning))

    def _emit_log(self, message: str, tag: str = "EXPLORE") -> None:
        if not self.verbose_step_logs:
            return
        stamp = time.strftime("%H:%M:%S")
        with self._log_lock:
            print(f"[{tag} {stamp}] {message}")

    def _emit_log_block(self, title: str, content: str, tag: str = "REASON") -> None:
        if not self.verbose_step_logs:
            return
        stamp = time.strftime("%H:%M:%S")
        with self._log_lock:
            print(f"[{tag} {stamp}] {title} BEGIN")
            print(content if content else "<EMPTY>")
            print(f"[{tag} {stamp}] {title} END")

    def _step_separator(
        self,
        step_no: int,
        phase: str,
        goal: str | None = None,
        summary: str | None = None,
    ) -> None:
        if not self.verbose_step_logs:
            return
        sep = "=" * 96
        with self._log_lock:
            print(sep)
            if phase == "start":
                print(f"[STEP {step_no:03d} START] goal={goal}")
            else:
                print(f"[STEP {step_no:03d} END] {summary or ''}".rstrip())
            print(sep)

    @staticmethod
    def _safe_task_name(goal: str, max_len: int = 72) -> str:
        value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(goal or "").strip())
        value = re.sub(r"_+", "_", value).strip("._")
        if not value:
            value = "task"
        return value[:max_len]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): ExplorerElementAgent._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ExplorerElementAgent._json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _write_json(self, path: str, payload: dict[str, Any]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._json_safe(payload), f, ensure_ascii=False, indent=2)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._emit_log(f"write_json_failed path={path} error={exc}", tag="TRACE")

    @staticmethod
    def _save_png(path: str, pixels: np.ndarray) -> None:
        Image.fromarray(np.array(pixels, copy=True)).save(path)

    def _ensure_task_context(self, goal: str) -> None:
        if self._task_goal == goal and self._task_start_ts is not None:
            return
        self._finalize_task_context(status="switch")
        self._task_goal = str(goal)
        self._task_start_ts = time.time()
        self._task_step_latencies = []
        self._task_trace_dir = None
        if self.trace_output_dir:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            task_name = self._safe_task_name(goal)
            task_dir = os.path.join(self.trace_output_dir, f"{stamp}_{task_name}")
            try:
                os.makedirs(task_dir, exist_ok=True)
                self._task_trace_dir = task_dir
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._emit_log(f"task_trace_dir_create_failed error={exc}", tag="TRACE")
                self._task_trace_dir = None

    def _finalize_task_context(self, status: str = "completed") -> None:
        if self._task_start_ts is None or self._task_goal is None:
            return
        total_sec = float(max(0.0, time.time() - float(self._task_start_ts)))
        steps = len(self._task_step_latencies)
        avg_sec = float(sum(self._task_step_latencies) / steps) if steps > 0 else 0.0
        self._emit_log(
            (
                f"task_summary goal={self._task_goal} status={status} steps={steps} "
                f"total_latency_sec={total_sec:.3f} avg_step_latency_sec={avg_sec:.3f}"
            ),
            tag="INFO",
        )
        if self._task_trace_dir:
            summary_path = os.path.join(self._task_trace_dir, "task_summary.json")
            self._write_json(
                summary_path,
                {
                    "goal": self._task_goal,
                    "status": status,
                    "steps": steps,
                    "total_latency_sec": total_sec,
                    "avg_step_latency_sec": avg_sec,
                    "step_latencies_sec": [float(x) for x in self._task_step_latencies],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        self._task_goal = None
        self._task_start_ts = None
        self._task_trace_dir = None
        self._task_step_latencies = []

    def _save_explore_trace(
        self,
        source_step: int,
        branch_id: int,
        depth: int,
        before_pixels: np.ndarray,
        after_pixels: np.ndarray,
        record: dict[str, Any],
    ) -> None:
        if not self._task_trace_dir:
            return
        base = f"step_{int(source_step):03d}_explore_branch_{int(branch_id):02d}_depth_{int(depth):02d}"
        before_path = os.path.join(self._task_trace_dir, f"{base}_before.png")
        after_path = os.path.join(self._task_trace_dir, f"{base}_after.png")
        json_path = os.path.join(self._task_trace_dir, f"{base}.json")
        try:
            self._save_png(before_path, before_pixels)
            self._save_png(after_path, after_pixels)
            payload = dict(record or {})
            payload["before_image"] = before_path
            payload["after_image"] = after_path
            payload["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._write_json(json_path, payload)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._emit_log(f"save_explore_trace_failed step={source_step} error={exc}", tag="TRACE")

    def _save_step_trace(
        self,
        step_no: int,
        start_pixels: np.ndarray,
        end_pixels: np.ndarray,
        payload: dict[str, Any],
    ) -> None:
        if not self._task_trace_dir:
            return
        base = f"step_{int(step_no):03d}"
        start_img = os.path.join(self._task_trace_dir, f"{base}_start.png")
        end_img = os.path.join(self._task_trace_dir, f"{base}_end.png")
        json_file = os.path.join(self._task_trace_dir, f"{base}.json")
        try:
            self._save_png(start_img, start_pixels)
            self._save_png(end_img, end_pixels)
            out = dict(payload or {})
            out["start_image"] = start_img
            out["end_image"] = end_img
            out["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._write_json(json_file, out)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._emit_log(f"save_step_trace_failed step={step_no} error={exc}", tag="TRACE")

    @staticmethod
    def _normalize_activity_name(activity: str | None) -> str:
        return _normalize_space(activity or "").lower()

    @staticmethod
    def _activity_package_name(activity: str | None) -> str:
        value = _normalize_space(activity or "")
        if not value:
            return ""
        if "/" in value:
            return value.split("/", 1)[0].strip().lower()
        return value.strip().lower()

    def _foreground_activity_name(self) -> str:
        try:
            return str(self.env.foreground_activity_name or "").strip()
        except Exception:  # pylint: disable=broad-exception-caught
            return ""

    @staticmethod
    def _clue_text_snippet(text: str, max_chars: int = 160) -> str:
        value = _normalize_space(text)
        if not value:
            return ""
        if len(value) <= max_chars:
            return value
        return value[: max(24, int(max_chars))].rstrip(" ,.;:") + "..."

    @staticmethod
    def _element_short_label(element: Any) -> str:
        return _element_hint_compact_label(element)

    def _state_page_hint(self, state: Any, max_cues: int = 8) -> str:
        try:
            page_hash = _phash_pixels(state.pixels)
        except Exception:  # pylint: disable=broad-exception-caught
            page_hash = -1

        try:
            foreground_activity = self.env.foreground_activity_name
        except Exception:  # pylint: disable=broad-exception-caught
            foreground_activity = ""
        try:
            logical_size = self.env.logical_screen_size
        except Exception:  # pylint: disable=broad-exception-caught
            logical_size = None
        try:
            orientation = self.env.orientation
        except Exception:  # pylint: disable=broad-exception-caught
            orientation = None

        cues = []
        seen = set()
        all_elements = list(getattr(state, "ui_elements", None) or [])
        for idx, element in enumerate(all_elements):
            if not self._is_valid_element(element):
                continue
            label = f"index={idx}, {self._element_short_label(element)}"
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            cues.append(label)
            if len(cues) >= max(1, int(max_cues)):
                break

        cue_text = "; ".join(cues) if cues else "no_semantic_nodes"
        return (
            f"activity={foreground_activity}, logical_size={logical_size}, orientation={orientation}, "
            f"ui_elements_total={len(all_elements)}, hash={page_hash}, cues={cue_text}"
        )

    # -----------------------------
    # Reset / thread lifecycle
    # -----------------------------
    def reset(self, go_home: bool = False) -> None:
        self._finalize_task_context(status="reset")
        self._stop_explorer_thread()
        super().reset(go_home)
        self.actions.clear()
        self.history.clear()
        self._recent_indices.clear()
        self._last_pixels = None
        self._last_action_text = ""
        self._no_effect_repeat = 0
        self._execution_feedback = ""

        self._explore_iteration_candidates = []
        self._pending_explore_payload = None
        self._last_clue_debug = {}
        self._explore_root_hash = None
        self._explore_root_pixels = None
        self._explore_root_activity = None
        self._explore_thread_stop_clean = True
        self._replay_action_history = []
        self._reasoning_action_history = []
        self._branch_action_history = []

        self._clicked_bounds.clear()
        self._bound_visit_count.clear()
        self._bound_effect_ema.clear()
        self._bound_effect_count.clear()
        self._bound_seen_count.clear()
        self._bound_skip_count.clear()
        self._recent_clicked_bounds.clear()
        self._recent_clicked_regions.clear()
        self._last_filter_stats = {}
        self._emb_cache.clear()
        self._embed_model = None
        self._embed_ready = False
        self._task_goal = None
        self._task_start_ts = None
        self._task_trace_dir = None
        self._task_step_latencies = []

    def _start_explorer_thread(
        self,
        goal: str,
        history_tail: list[str],
        clues_text: str,
        source_step: int,
    ) -> None:
        self._stop_explorer_thread()
        if not self._explore_thread_stop_clean:
            self._emit_log(
                f"step={source_step} explorer_thread_start_skipped reason=previous_thread_not_stopped",
                tag="EXPLORE",
            )
            return
        self._explore_stop_event.clear()
        self._explore_progress_event.clear()
        with self._explore_action_count_lock:
            self._explore_action_count = 0
        self._explore_iteration_candidates = []
        self._clicked_bounds = set()
        self._branch_action_history = []
        self._replay_action_history = list(self._reasoning_action_history)

        with self._ui_lock:
            root_state = self.env.get_state(wait_to_stabilize=False)
        self._explore_root_pixels = np.array(root_state.pixels, copy=True)
        self._explore_root_hash = _phash_pixels(root_state.pixels)
        self._explore_root_activity = self._foreground_activity_name() or None
        self._emit_log(
            f"step={source_step} explorer_thread_started root_page=({self._state_page_hint(root_state)})",
            tag="EXPLORE",
        )

        self._explore_thread = threading.Thread(
            target=self._explore_worker,
            args=(goal, history_tail, clues_text, source_step),
            daemon=True,
        )
        self._explore_thread.start()

    def _wait_for_explore_warmup(self, step_no: int) -> None:
        """Wait until exploration makes real progress or timeout."""
        timeout = float(self.explore_warmup_timeout_sec)
        min_actions = int(self.explore_min_actions_before_reasoning)
        if timeout <= 0.0 or min_actions <= 0:
            return
        start = time.perf_counter()
        waited = 0.0
        reason = "timeout"
        while waited < timeout and not self._explore_stop_event.is_set():
            with self._explore_action_count_lock:
                count = int(self._explore_action_count)
            if count >= min_actions:
                reason = "progress"
                break
            if self._explore_progress_event.wait(timeout=0.08):
                with self._explore_action_count_lock:
                    count = int(self._explore_action_count)
                if count >= min_actions:
                    reason = "progress"
                    break
            waited = float(time.perf_counter() - start)
        with self._explore_action_count_lock:
            final_count = int(self._explore_action_count)
        self._emit_log(
            (
                f"step={step_no} explore_warmup_done reason={reason} "
                f"actions={final_count} waited_sec={min(waited, timeout):.3f}"
            ),
            tag="EXPLORE",
        )

    def _get_explore_action_count(self) -> int:
        with self._explore_action_count_lock:
            return int(self._explore_action_count)

    def _stop_explorer_thread(self) -> list[dict[str, Any]]:
        self._explore_stop_event.set()
        self._explore_progress_event.set()
        clean = True
        if self._explore_thread and self._explore_thread.is_alive():
            self._explore_thread.join(timeout=6.0)
            if self._explore_thread.is_alive():
                self._emit_log("explorer_thread_join_timeout waiting_extra=2s", tag="EXPLORE")
                self._explore_thread.join(timeout=2.0)
            if self._explore_thread.is_alive():
                self._emit_log("explorer_thread_still_alive_after_stop", tag="EXPLORE")
                clean = False
        self._explore_thread_stop_clean = clean
        if clean:
            self._explore_thread = None
        return list(self._explore_iteration_candidates)

    # -----------------------------
    # Embedding / scoring
    # -----------------------------
    def _ensure_embed_model(self) -> bool:
        if self._embed_ready:
            return self._embed_model is not None
        self._embed_ready = True
        if SentenceTransformer is None:
            return False
        try:
            self._embed_model = SentenceTransformer(self.embed_model_name)
        except Exception:  # pylint: disable=broad-exception-caught
            self._embed_model = None
        return self._embed_model is not None

    def _embed_text(self, text: str) -> np.ndarray | None:
        key = _normalize_space(text)
        if not key:
            return None
        if key in self._emb_cache:
            return self._emb_cache[key]
        if not self._ensure_embed_model():
            return None
        try:
            emb = np.asarray(self._embed_model.encode(key), dtype=np.float32)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        self._emb_cache[key] = emb
        return emb

    @staticmethod
    def _cosine_max(goal_vecs: list[np.ndarray], cand_vec: np.ndarray | None) -> float:
        if cand_vec is None or not goal_vecs:
            return 0.0
        cand_norm = float(np.linalg.norm(cand_vec)) + 1e-8
        best = 0.0
        for vec in goal_vecs:
            vec_norm = float(np.linalg.norm(vec)) + 1e-8
            sim = float(np.dot(vec, cand_vec) / (vec_norm * cand_norm))
            if sim > best:
                best = sim
        return best

    @staticmethod
    def _lexical_similarity(queries: list[str], candidate_text: str) -> float:
        if not queries:
            return 0.0
        c_low = candidate_text.lower()
        c_tokens = set(_tokenize(c_low))
        best = 0.0
        for query in queries:
            q_low = query.lower()
            q_tokens = set(_tokenize(q_low))
            if not q_tokens:
                continue
            overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            jaccard = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))
            contains = 0.25 if q_low and (q_low in c_low or c_low in q_low) else 0.0
            score = min(1.0, 0.6 * overlap + 0.4 * jaccard + contains)
            best = max(best, score)
        return float(best)

    def _semantic_similarity(self, queries: list[str], candidate_text: str) -> float:
        if not queries:
            return 0.0
        if self._ensure_embed_model():
            goal_vecs = []
            for query in queries:
                vec = self._embed_text(query)
                if vec is not None:
                    goal_vecs.append(vec)
            if goal_vecs:
                cand_vec = self._embed_text(candidate_text)
                return self._cosine_max(goal_vecs, cand_vec)
        return self._lexical_similarity(queries, candidate_text)

    def _runtime_queries(self, history_tail: list[str], clues_text: str) -> list[str]:
        sources: list[str] = []
        if history_tail:
            for item in history_tail[-4:]:
                text = _normalize_space(item)
                if not text:
                    continue
                compact_fields: list[str] = []
                match = re.search(r"action=([^|,\s]+)", text, flags=re.IGNORECASE)
                if match:
                    compact_fields.append(_normalize_space(match.group(1)))
                for key in ("text", "button", "direction", "status"):
                    for field_match in re.finditer(
                        rf"{key}=([^|,]+)", text, flags=re.IGNORECASE
                    ):
                        value = _normalize_space(field_match.group(1))
                        if value:
                            compact_fields.append(value)
                if compact_fields:
                    sources.extend(compact_fields[:3])
                else:
                    sources.append(text[:140])
        if clues_text:
            clues = str(clues_text)
            keyword_matches = re.findall(r"candidate_keywords=([^\n]+)", clues, flags=re.IGNORECASE)
            for keyword_line in keyword_matches[:2]:
                keyword_line = _normalize_space(keyword_line)
                if keyword_line:
                    sources.append(keyword_line)
            if not keyword_matches:
                trunk_match = re.search(r"trunk_text=([^,\n]+)", clues, flags=re.IGNORECASE)
                if trunk_match:
                    trunk_text = _normalize_space(trunk_match.group(1))
                    if trunk_text:
                        sources.append(trunk_text)
        queries = []
        seen = set()
        for src in sources:
            src = _normalize_space(src)
            if not src:
                continue
            if len(src) > 220:
                src = src[:220]
            for query in _extract_task_queries(src):
                query = _normalize_space(query).strip("[]|")
                key = query.lower()
                if not query or key in seen:
                    continue
                if len(query) > 96:
                    query = query[:96].rstrip(" ,.;")
                    key = query.lower()
                seen.add(key)
                queries.append(query)
                if len(queries) >= 18:
                    return queries
        return queries

    @staticmethod
    def _task_wants_settings(goal_queries: list[str], runtime_queries: list[str]) -> bool:
        merged = " ".join([*(goal_queries or []), *(runtime_queries or [])]).lower()
        if not merged:
            return False
        tokens = {
            "setting",
            "settings",
            "setup",
            "configuration",
            "config",
            "sample rate",
            "recording format",
            "quality",
            "bitrate",
            "rename",
            "name format",
            "theme",
        }
        return any(token in merged for token in tokens)

    @staticmethod
    def _is_settings_like_text(merged_text: str) -> bool:
        text = _normalize_space(merged_text).lower()
        if not text:
            return False
        tokens = {
            "setup",
            "settings",
            "recording format",
            "sample rate",
            "name format",
            "theme",
            "reset",
            "apply",
            "information",
            "setting_btn_info",
            "txt_information",
        }
        return any(token in text for token in tokens)

    # -----------------------------
    # Candidate collection / scoring
    # -----------------------------
    @staticmethod
    def _is_interactive(element: Any) -> bool:
        return bool(
            getattr(element, "is_clickable", False)
            or getattr(element, "is_editable", False)
            or getattr(element, "is_long_clickable", False)
            or getattr(element, "is_scrollable", False)
        )

    @staticmethod
    def _is_valid_element(element: Any) -> bool:
        visible = bool(getattr(element, "is_visible", True))
        enabled = getattr(element, "is_enabled", True)
        return visible and enabled is not False

    def _element_key(self, index: int, element: Any) -> str:
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is not None:
            return f"{int(bbox.x_min)}:{int(bbox.y_min)}:{int(bbox.x_max)}:{int(bbox.y_max)}"
        rid = _normalize_space(getattr(element, "resource_id", "") or getattr(element, "resource_name", ""))
        cls = _normalize_space(getattr(element, "class_name", ""))
        txt = _normalize_space(getattr(element, "text", ""))[:40]
        return f"idx={index}|rid={rid}|cls={cls}|txt={txt}"

    def _element_text(self, index: int, element: Any) -> tuple[str, bool]:
        text = _normalize_space(getattr(element, "text", ""))
        desc = _normalize_space(getattr(element, "content_description", ""))
        hint = _normalize_space(getattr(element, "hint_text", ""))
        rid = _normalize_space(getattr(element, "resource_id", "") or getattr(element, "resource_name", ""))
        cls = _normalize_space(getattr(element, "class_name", ""))
        has_semantic_text = bool(text or desc or hint)
        joined = " | ".join([x for x in [text, desc, hint, rid, cls] if x])
        if not joined:
            joined = f"[icon_only] {self._element_key(index, element)}"
        return joined, has_semantic_text

    @staticmethod
    def _element_merged_text(element: Any) -> str:
        text = _normalize_space(getattr(element, "text", "")).lower()
        desc = _normalize_space(getattr(element, "content_description", "")).lower()
        hint = _normalize_space(getattr(element, "hint_text", "")).lower()
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        return " ".join([x for x in [text, desc, hint, rid, cls] if x]).strip()

    @staticmethod
    def _query_keywords(goal_queries: list[str], runtime_queries: list[str], limit: int = 24) -> list[str]:
        merged = " ".join([*(goal_queries or []), *(runtime_queries or [])])
        out: list[str] = []
        seen: set[str] = set()
        for token in _tokenize(merged):
            if len(token) < 3 or token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= max(1, int(limit)):
                break
        return out

    @staticmethod
    def _query_overlap_score(query_keywords: list[str], merged_text: str) -> float:
        if not query_keywords or not merged_text:
            return 0.0
        hit = 0
        for keyword in query_keywords:
            if keyword and keyword in merged_text:
                hit += 1
        if hit <= 0:
            return 0.0
        denom = max(2.0, min(6.0, float(len(query_keywords))))
        return min(1.0, float(hit) / denom)

    @staticmethod
    def _is_submit_or_dismiss_control(element: Any) -> bool:
        merged = ExplorerElementAgent._element_merged_text(element)
        if not merged:
            return False
        return any(token in merged for token in (_SUBMIT_ACTION_KEYWORDS | _DISMISS_ACTION_KEYWORDS))

    @staticmethod
    def _is_keyboard_key_like_element(element: Any) -> bool:
        text = _normalize_space(getattr(element, "text", ""))
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        merged = f"{rid} {cls}".strip()
        if any(token in merged for token in _KEYBOARD_HINT_TOKENS):
            return True
        if len(text) == 1 and re.fullmatch(r"[A-Za-z0-9]", text):
            return True
        return False

    @staticmethod
    def _is_choice_option_like(element: Any) -> bool:
        if bool(getattr(element, "is_checkable", False)):
            return True
        merged = ExplorerElementAgent._element_merged_text(element)
        if not merged:
            return False
        return any(token in merged for token in _CHOICE_HINT_TOKENS)

    @staticmethod
    def _is_date_time_picker_like(element: Any) -> bool:
        merged = ExplorerElementAgent._element_merged_text(element)
        if not merged:
            return False
        picker_tokens = {
            "datepicker",
            "timepicker",
            "calendarview",
            "calendar",
            "month",
            "year",
            "numberpicker",
            "date_picker",
            "time_picker",
            "clockface",
        }
        return any(token in merged for token in picker_tokens)

    @staticmethod
    def _is_meaningless_element(element: Any, intent_flags: dict[str, bool] | None = None) -> bool:
        intent_flags = intent_flags or {"input": False, "select": False, "nav": False}
        merged = ExplorerElementAgent._element_merged_text(element)
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        text = _normalize_space(getattr(element, "text", "")).lower()

        if ExplorerElementAgent._is_keyboard_key_like_element(element):
            return not bool(intent_flags.get("input"))
        if ExplorerElementAgent._is_date_time_picker_like(element):
            return True
        # Keep only obvious no-op widgets filtered out.
        if "progressbar" in cls:
            return True
        if len(text) == 1 and not bool(intent_flags.get("input")) and not bool(getattr(element, "is_clickable", False)):
            # Usually non-action single-char decorations.
            return True
        if not bool(getattr(element, "is_clickable", False)):
                if any(token in merged for token in {"systemui", "wifi signal", "battery", "phone signal", "do not disturb"}):
                    return True
        return False

    @staticmethod
    def _keyword_normalized_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()

    @staticmethod
    def _is_back_navigation_element(element: Any) -> bool:
        text = _normalize_space(getattr(element, "text", "")).lower()
        desc = _normalize_space(getattr(element, "content_description", "")).lower()
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        merged = " ".join([text, desc, rid, cls])
        normalized = ExplorerElementAgent._keyword_normalized_text(merged)
        if any(phrase in normalized for phrase in ("navigate up", "go back", "back")):
            return True
        rid_markers = ("btn_back", "navigate_up", "up_button", "nav_back")
        return any(marker in rid for marker in rid_markers)

    def _screen_mode_flags(self, ui_elements: list[Any]) -> dict[str, Any]:
        interactive_total = 0
        keyboard_hits = 0
        dialog_hits = 0
        choice_hits = 0
        for element in ui_elements or []:
            if not self._is_valid_element(element):
                continue
            if not self._is_interactive(element):
                continue
            interactive_total += 1
            merged = self._element_merged_text(element)
            if merged:
                if any(token in merged for token in _KEYBOARD_HINT_TOKENS):
                    keyboard_hits += 1
                if any(token in merged for token in _DIALOG_HINT_TOKENS):
                    dialog_hits += 1
            if self._is_choice_option_like(element):
                choice_hits += 1
        denom = max(1, interactive_total)
        keyboard = bool(keyboard_hits >= 3 or (keyboard_hits >= 2 and (keyboard_hits / denom) >= 0.20))
        dialog = bool(dialog_hits >= 1 and (dialog_hits / denom) >= 0.08)
        choice = bool(choice_hits >= 4 and (choice_hits / denom) >= 0.30)
        return {
            "keyboard": keyboard,
            "dialog": dialog,
            "choice": choice,
            "interactive_total": interactive_total,
            "keyboard_hits": keyboard_hits,
            "dialog_hits": dialog_hits,
            "choice_hits": choice_hits,
        }

    @staticmethod
    def _is_risky_element(element: Any) -> bool:
        merged = " ".join(
            [
                _normalize_space(getattr(element, "text", "")).lower(),
                _normalize_space(getattr(element, "content_description", "")).lower(),
                _normalize_space(getattr(element, "resource_id", "") or getattr(element, "resource_name", "")).lower(),
                _normalize_space(getattr(element, "class_name", "")).lower(),
            ]
        )
        merged = ExplorerElementAgent._keyword_normalized_text(merged)
        # Keep this strict: avoid only clearly destructive/escape controls.
        if not merged:
            return False
        risky_patterns = [
            r"\bnavigate up\b",
            r"\bgo back\b",
            r"\bback\b",
            r"\bexit\b",
            r"\bquit\b",
            r"\bclose\b",
            r"\bcancel\b",
            r"\bdismiss\b",
            r"\bdelete\b",
            r"\bremove\b",
            r"\bclear all\b",
            r"\bclear\b",
            r"\bdiscard\b",
            r"\breset\b",
            r"\bstop\b",
        ]
        if any(re.search(pattern, merged) for pattern in risky_patterns):
            return True
        text_only = _normalize_space(getattr(element, "text", "")).lower()
        desc_only = _normalize_space(getattr(element, "content_description", "")).lower()
        if re.fullmatch(r"x|×", text_only or "") or re.fullmatch(r"x|×", desc_only or ""):
            return True
        return False

    @staticmethod
    def _is_critical_risky_element(element: Any) -> bool:
        """Narrow risk gate used by fallback-relaxed candidate recovery."""
        merged = ExplorerElementAgent._element_merged_text(element)
        merged = ExplorerElementAgent._keyword_normalized_text(merged)
        if not merged:
            return False
        critical_patterns = [
            r"\bnavigate up\b",
            r"\bback\b",
            r"\bexit\b",
            r"\bquit\b",
            r"\bdelete\b",
            r"\bremove\b",
            r"\bclear all\b",
            r"\bdiscard\b",
        ]
        if any(re.search(pattern, merged) for pattern in critical_patterns):
            return True
        text_only = _normalize_space(getattr(element, "text", "")).lower()
        return bool(re.fullmatch(r"x|×", text_only or ""))

    def _element_region_label(self, element: Any) -> str:
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            return "unknown"
        cx = float(bbox.x_min + bbox.x_max) / 2.0
        cy = float(bbox.y_min + bbox.y_max) / 2.0
        try:
            width, height = self.env.logical_screen_size
            width = max(1.0, float(width))
            height = max(1.0, float(height))
        except Exception:  # pylint: disable=broad-exception-caught
            width = max(1.0, float(bbox.x_max))
            height = max(1.0, float(bbox.y_max))
        horiz = "left" if cx < width / 3.0 else ("right" if cx > 2.0 * width / 3.0 else "center")
        vert = "top" if cy < height / 3.0 else ("bottom" if cy > 2.0 * height / 3.0 else "middle")
        return f"{vert}-{horiz}"

    @staticmethod
    def _is_low_value_explore_element(element: Any) -> bool:
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        text = _normalize_space(getattr(element, "text", "")).lower()
        desc = _normalize_space(getattr(element, "content_description", "")).lower()
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        merged = " ".join([text, desc, rid, cls])

        class_low_value = any(token in cls for token in _LOW_VALUE_CLASS_HINTS)
        checkable_low_value = bool(getattr(element, "is_checkable", False))
        statusy_text = any(token in merged for token in _LOW_VALUE_TEXT_HINTS)
        has_semantic = bool(text or desc)
        # Typical pitfall: repeatedly toggling checkable/status controls with weak semantics.
        return bool(class_low_value or checkable_low_value or (statusy_text and not has_semantic))

    @staticmethod
    def _relevance_attenuation(task_score: float, reason_score: float) -> float:
        rel = max(float(task_score), float(reason_score))
        if rel >= 0.70:
            return 0.25
        if rel >= 0.55:
            return 0.50
        if rel >= 0.40:
            return 0.75
        return 1.0

    @staticmethod
    def _intent_flags(goal_queries: list[str], runtime_queries: list[str]) -> dict[str, bool]:
        merged = " ".join([*(goal_queries or []), *(runtime_queries or [])]).lower()
        wants_input = any(keyword in merged for keyword in _TASK_INPUT_KEYWORDS)
        wants_select = any(keyword in merged for keyword in _TASK_SELECT_KEYWORDS)
        wants_nav = any(keyword in merged for keyword in _NAV_HELPFUL_KEYWORDS)
        return {
            "input": bool(wants_input),
            "select": bool(wants_select),
            "nav": bool(wants_nav),
        }

    @staticmethod
    def _navigation_helpfulness(element: Any) -> float:
        cls = _normalize_space(getattr(element, "class_name", "")).lower()
        text = _normalize_space(getattr(element, "text", "")).lower()
        desc = _normalize_space(getattr(element, "content_description", "")).lower()
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        merged = " ".join([text, desc, rid, cls])
        bonus = 0.0
        if any(keyword in merged for keyword in _NAV_HELPFUL_KEYWORDS):
            bonus += 0.22
        if "button" in cls or "tab" in cls or "toolbar" in cls:
            bonus += 0.06
        if bool(getattr(element, "is_scrollable", False)):
            bonus += 0.06
        if bool(getattr(element, "is_clickable", False)) and (text or desc or rid):
            bonus += 0.04
        return min(0.34, float(bonus))

    @staticmethod
    def _is_information_only_element(element: Any) -> bool:
        text = _normalize_space(getattr(element, "text", "")).lower()
        desc = _normalize_space(getattr(element, "content_description", "")).lower()
        rid = _normalize_space(
            getattr(element, "resource_id", "") or getattr(element, "resource_name", "")
        ).lower()
        merged = " ".join([text, desc, rid])
        if text:
            allowed_chars = set("0123456789:./- \t")
            numeric_like = all(ch in allowed_chars for ch in text)
        else:
            numeric_like = False
        has_info_token = any(token in merged for token in _INFO_ONLY_TEXT_HINTS)
        return bool(numeric_like or has_info_token)

    @staticmethod
    def _filter_level_by_density(interactive_total: int) -> str:
        if interactive_total >= 28:
            return "strict"
        if interactive_total >= 12:
            return "balanced"
        return "loose"

    def _collect_candidates(
        self,
        ui_elements: list[Any],
        safe: bool = True,
        intent_flags: dict[str, bool] | None = None,
        query_keywords: list[str] | None = None,
        avoid_keys: set[str] | None = None,
        hard_avoid: bool = False,
    ) -> tuple[list[tuple[int, Any]], dict[str, Any]]:
        intent_flags = intent_flags or {"input": False, "select": False, "nav": False}
        query_keywords = query_keywords or []
        avoid_keys = avoid_keys or set()
        allow_input = bool(intent_flags.get("input"))
        allow_select = bool(intent_flags.get("select"))
        screen_flags = self._screen_mode_flags(ui_elements)
        interactive_total = sum(
            1
            for element in ui_elements
            if self._is_valid_element(element) and self._is_interactive(element)
        )
        filter_level = self._filter_level_by_density(interactive_total)
        stats: dict[str, Any] = {
            "total": len(ui_elements),
            "interactive_total": interactive_total,
            "filter_level": filter_level,
            "screen_flags": screen_flags,
            "valid_interactive": 0,
            "removed_visited": 0,
            "removed_low_effect_repeat": 0,
            "removed_risky": 0,
            "removed_meaningless": 0,
            "removed_intent_mismatch": 0,
            "after_level1": 0,
            "after_level2": 0,
            "after_level3": 0,
            "fallback_relaxed": 0,
        }
        level1: list[tuple[int, Any]] = []
        recent_keys = set(list(self._recent_clicked_bounds)[-self._recent_clicked_window :])

        for idx, element in enumerate(ui_elements):
            if not self._is_valid_element(element):
                continue
            if not self._is_interactive(element):
                continue
            stats["valid_interactive"] += 1
            key = self._element_key(idx, element)
            if key in recent_keys or (hard_avoid and key in avoid_keys):
                stats["removed_visited"] += 1
                continue
            effect_ema = self._bound_effect_ema.get(key)
            effect_cnt = int(self._bound_effect_count.get(key, 0))
            visits = float(self._bound_visit_count.get(key, 0.0))
            min_repeat = 3.0 if filter_level == "loose" else 2.0
            if not bool(intent_flags.get("nav")) and self._is_back_navigation_element(element):
                stats["removed_intent_mismatch"] += 1
                continue
            # Keep exploration alive: only suppress repeated low-yield elements.
            if (
                visits >= min_repeat
                and effect_cnt >= 2
                and effect_ema is not None
                and float(effect_ema) <= (self.no_effect_delta_threshold * 1.35)
            ):
                stats["removed_low_effect_repeat"] += 1
                continue
            # Never allow critically destructive controls in exploration.
            if self._is_critical_risky_element(element):
                stats["removed_risky"] += 1
                continue
            if safe and self._is_risky_element(element):
                stats["removed_risky"] += 1
                continue
            if filter_level != "loose" and self._is_meaningless_element(
                element, intent_flags=intent_flags
            ):
                stats["removed_meaningless"] += 1
                continue
            level1.append((idx, element))

        stats["after_level1"] = len(level1)
        if not level1:
            # Relaxation path: keep safe + meaningful interactives even if previously explored.
            relaxed: list[tuple[int, Any]] = []
            for idx, element in enumerate(ui_elements):
                if not self._is_valid_element(element):
                    continue
                if not self._is_interactive(element):
                    continue
                key = self._element_key(idx, element)
                if key in recent_keys:
                    continue
                if safe and self._is_critical_risky_element(element):
                    continue
                if self._is_meaningless_element(element, intent_flags=intent_flags):
                    continue
                relaxed.append((idx, element))
            stats["fallback_relaxed"] = len(relaxed)
            if relaxed:
                stats["after_level1"] = len(relaxed)
                return relaxed, stats
            return [], stats

        # Level-2: very light screen-mode constraints (avoid over-pruning).
        level2: list[tuple[int, Any]] = []
        for idx, element in level1:
            if bool(screen_flags.get("keyboard")) and not allow_input:
                if self._is_keyboard_key_like_element(element):
                    continue
            level2.append((idx, element))
        stats["after_level2"] = len(level2)
        if not level2:
            level2 = list(level1)
            stats["after_level2"] = len(level2)

        # Level-3: relevance hinting with high recall (retain most candidates).
        if not query_keywords or filter_level == "loose":
            stats["after_level3"] = len(level2)
            return level2, stats
        focus: list[tuple[int, Any]] = []
        tail: list[tuple[int, Any]] = []
        for idx, element in level2:
            merged = self._element_merged_text(element)
            overlap = self._query_overlap_score(query_keywords, merged)
            nav_helpful = self._navigation_helpfulness(element) >= 0.12
            if overlap >= 0.12 or nav_helpful or self._is_submit_or_dismiss_control(element):
                focus.append((idx, element))
            else:
                tail.append((idx, element))
        if len(focus) >= 2:
            # Keep high recall: only trim a tiny fraction of tail.
            tail_ratio = 0.85 if filter_level == "strict" else 0.95
            keep_tail = max(5, int(math.ceil(len(level2) * tail_ratio)))
            level3 = focus + tail[:keep_tail]
        else:
            level3 = level2
        stats["after_level3"] = len(level3)
        return level3, stats

    def _score_candidate(
        self,
        index: int,
        element: Any,
        goal_queries: list[str],
        runtime_queries: list[str],
        intent_flags: dict[str, bool] | None = None,
        query_keywords: list[str] | None = None,
    ) -> CandidateScore:
        intent_flags = intent_flags or {}
        query_keywords = query_keywords or []
        key = self._element_key(index, element)
        cand_text, _ = self._element_text(index, element)
        merged_text = self._element_merged_text(element)
        task_similarity = float(self._semantic_similarity(goal_queries, cand_text))
        runtime_similarity = float(self._semantic_similarity(runtime_queries, cand_text)) if runtime_queries else 0.0
        similarity = float(max(task_similarity, runtime_similarity))
        visits = float(self._bound_visit_count.get(key, 0.0))
        is_clickable = bool(getattr(element, "is_clickable", False))
        # Similarity-dominant score. Repetition/risk control is handled by filtering layers.
        total_score = float(similarity)

        overlap = float(self._query_overlap_score(query_keywords, merged_text))
        if overlap > 0.0:
            total_score += min(0.14, overlap * 0.22)

        nav_help = float(self._navigation_helpfulness(element))
        if bool(intent_flags.get("nav")) and nav_help > 0.0:
            total_score += min(0.10, nav_help * 0.45)
        elif self._is_back_navigation_element(element):
            total_score -= 0.24

        # Encourage trying unseen controls so exploration doesn't collapse to one slot.
        if visits <= 0.01:
            total_score += 0.04
        elif visits >= 2.0:
            total_score -= min(0.06, (visits - 1.0) * 0.02)

        # Penalize setup/settings style controls when task doesn't ask for configuration.
        if self._is_settings_like_text(merged_text):
            wants_settings = self._task_wants_settings(goal_queries, runtime_queries)
            if not wants_settings:
                total_score -= 0.16

        total_score = max(-0.5, min(1.25, float(total_score)))

        return CandidateScore(
            index=index,
            key=key,
            text=cand_text,
            score=float(total_score),
            task_similarity=float(task_similarity),
            runtime_similarity=float(runtime_similarity),
            similarity=float(similarity),
            visits=float(visits),
            is_clickable=bool(is_clickable),
        )

    def _pick_topk(
        self,
        ui_elements: list[Any],
        goal_queries: list[str],
        runtime_queries: list[str],
        k: int,
        avoid_keys: set[str] | None = None,
        hard_avoid: bool = False,
        intent_flags: dict[str, bool] | None = None,
    ) -> tuple[list[CandidateScore], int]:
        query_keywords = self._query_keywords(goal_queries, runtime_queries)
        candidates, filter_stats = self._collect_candidates(
            ui_elements,
            safe=True,
            intent_flags=intent_flags,
            query_keywords=query_keywords,
            avoid_keys=avoid_keys,
            hard_avoid=hard_avoid,
        )
        # Safety-relaxed fallback: when strict safe filtering leaves too few options,
        # add additional candidates without risky filtering to keep exploration active.
        min_needed = max(2, min(int(k), 4))
        if len(candidates) < min_needed:
            relaxed_candidates, relaxed_stats = self._collect_candidates(
                ui_elements,
                safe=False,
                intent_flags=intent_flags,
                query_keywords=query_keywords,
                avoid_keys=avoid_keys,
                hard_avoid=hard_avoid,
            )
            if relaxed_candidates:
                seen_keys: set[str] = set()
                merged: list[tuple[int, Any]] = []
                for idx, element in candidates:
                    key = self._element_key(idx, element)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    merged.append((idx, element))
                added = 0
                for idx, element in relaxed_candidates:
                    key = self._element_key(idx, element)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    merged.append((idx, element))
                    added += 1
                if added > 0:
                    candidates = merged
                    filter_stats["safety_relaxed_added"] = int(added)
                    filter_stats["safety_relaxed_total"] = int(len(relaxed_candidates))
                    filter_stats["safety_relaxed_stats"] = relaxed_stats
        filter_stats.setdefault("safety_relaxed_added", 0)
        self._last_filter_stats = filter_stats
        n_candidates = len(candidates)
        if n_candidates == 0:
            return [], 0
        scored = []
        for idx, element in candidates:
            key = self._element_key(idx, element)
            self._bound_seen_count[key] = int(self._bound_seen_count.get(key, 0)) + 1
            try:
                cand_score = self._score_candidate(
                    idx,
                    element,
                    goal_queries,
                    runtime_queries,
                    intent_flags=intent_flags,
                    query_keywords=query_keywords,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._emit_log(
                    f"_pick_topk score_failed index={idx} key={key} error={exc}",
                    tag="EXPLORE",
                )
                continue
            scored.append(cand_score)
        if not scored:
            return [], n_candidates
        scored.sort(key=lambda item: item.score, reverse=True)
        want = max(1, int(k))
        return scored[:want], n_candidates

    def _decay_visit_counts(self, decay: float = 0.92) -> None:
        if not self._bound_visit_count:
            return
        new_count = {}
        for bound_key, val in self._bound_visit_count.items():
            decayed = float(val) * float(decay)
            if decayed >= 0.15:
                new_count[bound_key] = decayed
        self._bound_visit_count = new_count
        # Keep effect memory aligned with active explored bounds.
        active_keys = set(new_count.keys())
        for key in list(self._bound_effect_ema.keys()):
            if key not in active_keys:
                self._bound_effect_ema.pop(key, None)
                self._bound_effect_count.pop(key, None)
                self._bound_seen_count.pop(key, None)
                self._bound_skip_count.pop(key, None)

    # -----------------------------
    # Exploration worker / rollback
    # -----------------------------
    def _same_root_page(
        self,
        curr_pixels: np.ndarray,
        curr_activity: str | None = None,
        phash_thr: int = 18,
        mae_thr: float = 14.0,
    ) -> tuple[bool, str]:
        if self._explore_root_pixels is None or self._explore_root_hash is None:
            return False, "missing_root"
        root_activity = self._normalize_activity_name(self._explore_root_activity)
        root_pkg = self._activity_package_name(self._explore_root_activity)
        if curr_activity is None:
            curr_activity = self._foreground_activity_name()
        curr_activity_norm = self._normalize_activity_name(curr_activity)
        curr_pkg = self._activity_package_name(curr_activity)
        different_package = bool(
            root_pkg and curr_pkg and curr_pkg != root_pkg
        )
        try:
            curr_hash = _phash_pixels(curr_pixels)
            phash_diff = _hash_diff(self._explore_root_hash, curr_hash)
            if phash_diff <= int(phash_thr):
                return True, "phash"
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        try:
            mae = _mae_small(self._explore_root_pixels, curr_pixels)
            if different_package and mae <= mae_thr:
                return False, f"pkg_mismatch_mae:{mae:.2f}"
            if root_activity and curr_activity_norm and curr_activity_norm != root_activity:
                # Same package but different activity: allow MAE, but stricter threshold.
                mae_thr = min(float(mae_thr), 11.0)
            if mae <= mae_thr:
                return True, f"mae:{mae:.2f}"
            return False, f"mae:{mae:.2f}"
        except Exception:  # pylint: disable=broad-exception-caught
            return False, "mae:err"

    def _rollback_to_root(
        self,
        max_depth: int = 2,
        enable_replay: bool = True,
        trigger: str = "",
    ) -> dict[str, Any]:
        result = {
            "trigger": trigger or "unspecified",
            "success": False,
            "mode": "skip_no_root",
            "back_presses": 0,
            "replayed_actions": 0,
            "matched_by": None,
            "replay_open_resets": [],
        }
        if self._explore_root_hash is None:
            self._emit_log(
                f"trigger={result['trigger']} rollback_skipped reason=no_root_hash",
                tag="ROLLBACK",
            )
            return result
        if self._explore_stop_event.is_set() and str(result["trigger"]).startswith("explore_"):
            result["mode"] = "interrupted"
            self._emit_log(
                f"trigger={result['trigger']} rollback_interrupted by stop_event",
                tag="ROLLBACK",
            )
            return result

        rollback_success = False
        with self._rollback_lock:
            for i in range(max_depth):
                if self._explore_stop_event.is_set() and str(result["trigger"]).startswith("explore_"):
                    result["mode"] = "interrupted"
                    return result
                with self._ui_lock:
                    state = self.env.get_state(wait_to_stabilize=False)
                same, same_reason = self._same_root_page(
                    state.pixels,
                    curr_activity=self._foreground_activity_name(),
                )
                if same:
                    rollback_success = True
                    result["success"] = True
                    result["mode"] = "backtrack"
                    result["matched_by"] = same_reason
                    self._emit_log(
                        f"trigger={result['trigger']} rollback_done back_presses={result['back_presses']} "
                        f"matched_by={same_reason}",
                        tag="ROLLBACK",
                    )
                    break

                self._emit_log(
                    f"trigger={result['trigger']} rollback_back#{i + 1}/{max_depth} "
                    f"from_page=({self._state_page_hint(state)})",
                    tag="ROLLBACK",
                )
                with self._ui_lock:
                    self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK))
                result["back_presses"] += 1
                time.sleep(0.35)

            if not rollback_success and enable_replay:
                if self._explore_stop_event.is_set() and str(result["trigger"]).startswith("explore_"):
                    result["mode"] = "interrupted"
                    return result
                result["mode"] = "home_replay"
                self._emit_log(
                    f"trigger={result['trigger']} rollback_fallback=navigate_home+replay",
                    tag="ROLLBACK",
                )
                with self._ui_lock:
                    self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_HOME))
                time.sleep(0.8)
                replay_actions = self._replay_action_history or self._reasoning_action_history
                for action_dict in replay_actions:
                    try:
                        action = json_action.JSONAction(**action_dict)
                    except Exception:  # pylint: disable=broad-exception-caught
                        continue
                    if action.action_type == json_action.OPEN_APP:
                        reset_info = self._replay_open_app_with_force_stop(action.app_name)
                        result["replay_open_resets"].append(reset_info)
                    else:
                        with self._ui_lock:
                            self.env.execute_action(action)
                    result["replayed_actions"] += 1
                    if action.action_type == json_action.OPEN_APP:
                        time.sleep(1.2)
                    else:
                        time.sleep(0.35)

                with self._ui_lock:
                    final_state = self.env.get_state(wait_to_stabilize=False)
                same, same_reason = self._same_root_page(
                    final_state.pixels,
                    curr_activity=self._foreground_activity_name(),
                )
                result["success"] = bool(same)
                result["matched_by"] = same_reason
                self._emit_log(
                    f"trigger={result['trigger']} rollback_replay_done success={result['success']} "
                    f"replayed={result['replayed_actions']} matched_by={same_reason} "
                    f"final_page=({self._state_page_hint(final_state)})",
                    tag="ROLLBACK",
                )
            elif not rollback_success:
                result["mode"] = "backtrack_failed"
                self._emit_log(
                    f"trigger={result['trigger']} rollback_failed back_presses={result['back_presses']}",
                    tag="ROLLBACK",
                )

        return result

    def _replay_open_app_with_force_stop(self, app_name: str | None) -> dict[str, Any]:
        """Replay helper: force-stop target app before relaunching it."""
        app_text = str(app_name or "").strip()
        info: dict[str, Any] = {
            "app_name": app_text,
            "force_stop_attempted": False,
            "force_stop_ok": False,
            "force_stop_method": None,
            "force_stop_error": None,
            "launch_ok": False,
            "launch_error": None,
        }
        adb_target = getattr(self.env, "controller", self.env)
        if not app_text:
            info["force_stop_error"] = "empty_app_name"
            self._emit_log(
                f"replay_open_app skipped force-stop: empty app name",
                tag="ROLLBACK",
            )
            return info

        # Prefer official helper first.
        try:
            info["force_stop_attempted"] = True
            closed = adb_utils.close_app(app_text, adb_target, timeout_sec=5)
            if closed is not None:
                info["force_stop_ok"] = True
                info["force_stop_method"] = "close_app"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            info["force_stop_error"] = str(exc)

        # Fallback: derive package and issue explicit force-stop.
        if not info["force_stop_ok"]:
            package_name = ""
            try:
                mapped = str(adb_utils.normalize_app_name(app_text) or "").strip()
                if "/" in mapped:
                    mapped = mapped.split("/", 1)[0]
                package_name = mapped
                if "." not in package_name:
                    activity = adb_utils.get_adb_activity(app_text)
                    if activity:
                        package_name = adb_utils.extract_package_name(activity)
            except Exception:  # pylint: disable=broad-exception-caught
                package_name = ""

            if package_name and "." in package_name:
                try:
                    info["force_stop_attempted"] = True
                    adb_utils.issue_generic_request(
                        ["shell", "am", "force-stop", package_name],
                        adb_target,
                        timeout_sec=5,
                    )
                    info["force_stop_ok"] = True
                    info["force_stop_method"] = f"force_stop:{package_name}"
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    if not info["force_stop_error"]:
                        info["force_stop_error"] = str(exc)

        try:
            adb_utils.launch_app(app_text, adb_target)
            info["launch_ok"] = True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            info["launch_error"] = str(exc)
            with self._ui_lock:
                self.env.execute_action(
                    json_action.JSONAction(action_type=json_action.OPEN_APP, app_name=app_text)
                )
            info["launch_ok"] = True

        self._emit_log(
            (
                f"replay_open_app app={app_text} force_stop_ok={info['force_stop_ok']} "
                f"method={info['force_stop_method']} launch_ok={info['launch_ok']}"
            ),
            tag="ROLLBACK",
        )
        return info

    def _ensure_root_before_reasoning(
        self,
        step_no: int,
        max_attempts: int = 2,
    ) -> dict[str, Any]:
        def _is_root_stable(checks: int = 2, interval_sec: float = 0.12) -> tuple[bool, list[str]]:
            reasons: list[str] = []
            for idx in range(max(1, int(checks))):
                with self._ui_lock:
                    state = self.env.get_state(wait_to_stabilize=False)
                same_root, same_reason = self._same_root_page(
                    state.pixels,
                    curr_activity=self._foreground_activity_name(),
                )
                reasons.append(str(same_reason))
                if not same_root:
                    return False, reasons
                if idx + 1 < checks:
                    time.sleep(max(0.0, float(interval_sec)))
            return True, reasons

        verify_result: dict[str, Any] = {
            "success": False,
            "verified": False,
            "attempts": 0,
            "details": [],
        }
        for attempt in range(1, max(1, int(max_attempts)) + 1):
            info = self._rollback_to_root(
                max_depth=self.explore_max_depth + 2 + attempt,
                enable_replay=True,
                trigger=f"reasoning_step_{step_no}_post_llm_attempt_{attempt}",
            )
            with self._ui_lock:
                state = self.env.get_state(wait_to_stabilize=False)
            same_root, same_reason = self._same_root_page(
                state.pixels,
                curr_activity=self._foreground_activity_name(),
            )
            detail = {
                "attempt": attempt,
                "rollback": info,
                "same_root": bool(same_root),
                "same_reason": same_reason,
            }
            if same_root:
                stable, stable_reasons = _is_root_stable(checks=2, interval_sec=0.12)
                detail["stable"] = bool(stable)
                detail["stable_reasons"] = stable_reasons
                same_root = bool(stable)
            verify_result["details"].append(detail)
            verify_result["attempts"] = attempt
            if same_root:
                verify_result["success"] = True
                verify_result["verified"] = True
                verify_result["matched_by"] = detail.get("stable_reasons") or same_reason
                break
        return verify_result

    @staticmethod
    def _center_and_bounds(element: Any) -> tuple[list[int] | None, list[int] | None]:
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            return None, None
        center = [int((bbox.x_min + bbox.x_max) / 2.0), int((bbox.y_min + bbox.y_max) / 2.0)]
        bounds = [int(bbox.x_min), int(bbox.y_min), int(bbox.x_max), int(bbox.y_max)]
        return center, bounds

    def _candidate_action(self, ui_elements: list[Any], cand: CandidateScore) -> json_action.JSONAction:
        element = ui_elements[cand.index]
        center, _ = self._center_and_bounds(element)
        if bool(getattr(element, "is_clickable", False) or getattr(element, "is_long_clickable", False)):
            if center is not None:
                return json_action.JSONAction(action_type=json_action.CLICK, x=center[0], y=center[1])
            return json_action.JSONAction(action_type=json_action.CLICK, index=cand.index)
        if bool(getattr(element, "is_scrollable", False)):
            return json_action.JSONAction(action_type=json_action.SCROLL, direction="down", index=cand.index)
        if center is not None:
            return json_action.JSONAction(action_type=json_action.CLICK, x=center[0], y=center[1])
        return json_action.JSONAction(action_type=json_action.CLICK, index=cand.index)

    def _execute_action_with_coordinate_priority(self, action: json_action.JSONAction) -> str:
        """Execute coordinate actions directly through adb actuation when possible."""
        if action.action_type in {
            json_action.CLICK,
            json_action.LONG_PRESS,
            json_action.INPUT_TEXT,
        } and action.x is not None and action.y is not None:
            actuation.execute_adb_action(
                action,
                [],
                self.env.logical_screen_size,
                self.env.controller,
            )
            return "actuation_coordinate"
        self.env.execute_action(action)
        return "env_execute_action"

    def _click_and_record(
        self,
        cand: CandidateScore,
        ui_elements: list[Any],
        before_pixels: np.ndarray,
        n_candidates: int,
        depth: int,
        branch_id: int,
        source_step: int,
    ) -> dict[str, Any] | None:
        if self._explore_stop_event.is_set():
            return None
        action = self._candidate_action(ui_elements, cand)
        before_hash = None
        try:
            before_hash = _phash_pixels(before_pixels)
        except Exception:  # pylint: disable=broad-exception-caught
            before_hash = None
        if self._explore_stop_event.is_set():
            return None
        try:
            with self._ui_lock:
                self._execute_action_with_coordinate_priority(action)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        with self._explore_action_count_lock:
            self._explore_action_count += 1
        self._explore_progress_event.set()

        time.sleep(self.explore_action_pause_sec)
        with self._ui_lock:
            after_state = self.env.get_state(wait_to_stabilize=False)
        after_hash = None
        try:
            after_hash = _phash_pixels(after_state.pixels)
        except Exception:  # pylint: disable=broad-exception-caught
            after_hash = None
        effect_delta = _pixel_delta(before_pixels, after_state.pixels)
        page_hash_diff = (
            int(_hash_diff(before_hash, after_hash))
            if before_hash is not None and after_hash is not None
            else None
        )

        self._clicked_bounds.add(cand.key)
        self._bound_visit_count[cand.key] = float(self._bound_visit_count.get(cand.key, 0.0)) + 1.0
        self._recent_clicked_bounds.append(cand.key)
        if effect_delta is not None:
            prev_ema = self._bound_effect_ema.get(cand.key)
            if prev_ema is None:
                new_ema = float(effect_delta)
            else:
                new_ema = 0.65 * float(prev_ema) + 0.35 * float(effect_delta)
            self._bound_effect_ema[cand.key] = float(new_ema)
            self._bound_effect_count[cand.key] = int(self._bound_effect_count.get(cand.key, 0)) + 1

        element = ui_elements[cand.index]
        self._recent_clicked_regions.append(self._element_region_label(element))
        center, bounds = self._center_and_bounds(element)
        low_value_hit = self._is_low_value_explore_element(element)
        semantic_rel = float(cand.similarity)
        useful_by_change = bool(
            (effect_delta is not None and effect_delta > (self.no_effect_delta_threshold * 1.8))
            or (page_hash_diff is not None and page_hash_diff >= 6)
        )
        useful_by_semantic = bool(semantic_rel >= 0.35)
        is_useful = bool(useful_by_change or useful_by_semantic)
        if low_value_hit and semantic_rel < 0.35 and not useful_by_change:
            is_useful = False
        score_detail = {
            "score": round(cand.score, 4),
            "task_similarity": round(cand.task_similarity, 4),
            "runtime_similarity": round(cand.runtime_similarity, 4),
            "similarity": round(cand.similarity, 4),
            "visits": round(cand.visits, 3),
            "is_clickable": cand.is_clickable,
            "node_txt": cand.text,
            "effect_delta": None if effect_delta is None else round(float(effect_delta), 3),
            "page_hash_diff": page_hash_diff,
            "effect_ema": None if self._bound_effect_ema.get(cand.key) is None else round(float(self._bound_effect_ema[cand.key]), 3),
        }
        record = {
            "best_sim": round(cand.score, 3),
            "score_detail": score_detail,
            "executed_action": action.as_dict(skip_none=True),
            "coordinate": center,
            "bounds": bounds,
            "node_text": getattr(element, "text", None),
            "node_desc": getattr(element, "content_description", None),
            "node_resource_id": getattr(element, "resource_name", None) or getattr(element, "resource_id", None),
            "node_class": getattr(element, "class_name", None),
            "node_match_text": cand.text,
            "n_candidates": n_candidates,
            "depth": depth,
            "branch": branch_id,
            "hash": _phash_pixels(after_state.pixels),
            "page_hint": self._state_page_hint(after_state),
            "action_type": action.action_type,
            "effect_delta": effect_delta,
            "page_hash_diff": page_hash_diff,
            "is_useful": is_useful,
            "useful_by_change": useful_by_change,
            "useful_by_semantic": useful_by_semantic,
            "low_value_hit": low_value_hit,
            "filter_stats": dict(self._last_filter_stats or {}),
        }
        self._save_explore_trace(
            source_step=source_step,
            branch_id=branch_id,
            depth=depth,
            before_pixels=before_pixels,
            after_pixels=after_state.pixels,
            record=record,
        )
        return record

    def _select_depth_candidate(
        self,
        candidates: list[CandidateScore],
        semantic_low: float = 0.20,
        intent_flags: dict[str, bool] | None = None,
    ) -> tuple[CandidateScore | None, int]:
        intent_flags = intent_flags or {}
        if not candidates:
            return None, 0
        skipped = 0
        low_scored: list[CandidateScore] = []
        for cand in candidates:
            if cand.similarity < semantic_low and cand.visits > 0.0:
                self._bound_skip_count[cand.key] = int(self._bound_skip_count.get(cand.key, 0)) + 1
                skipped += 1
                low_scored.append(cand)
                continue
            return cand, skipped
        # If everything is weak, prefer the least-visited option to preserve diversity
        # and avoid repeatedly looping on the same low-value control.
        fallback_pool = low_scored or candidates
        fallback = min(fallback_pool, key=lambda c: (c.visits, -c.score))
        if bool(intent_flags.get("nav")) and len(fallback_pool) > 1:
            nav_pref = sorted(fallback_pool, key=lambda c: (-c.score, c.visits))
            fallback = nav_pref[0]
        self._bound_skip_count[fallback.key] = int(self._bound_skip_count.get(fallback.key, 0))
        return fallback, skipped

    def _collect_useless_keys_for_mask(self, ui_elements: list[Any]) -> set[str]:
        keys: set[str] = set()
        thr = float(self.no_effect_delta_threshold)
        for idx, element in enumerate(ui_elements):
            key = self._element_key(idx, element)
            seen = int(self._bound_seen_count.get(key, 0))
            skipped = int(self._bound_skip_count.get(key, 0))
            clicked = float(self._bound_visit_count.get(key, 0.0))
            effect_cnt = int(self._bound_effect_count.get(key, 0))
            effect_ema = self._bound_effect_ema.get(key)
            low_value = self._is_low_value_explore_element(element)

            repeated_no_effect = bool(
                clicked >= 2.0
                and effect_cnt >= 2
                and effect_ema is not None
                and float(effect_ema) <= thr * 1.5
            )
            explored_not_clicked = bool((seen >= 6 and clicked < 1.0) or (seen >= 10 and clicked <= 1.0 and skipped >= 4))
            repeated_low_value = bool(low_value and skipped >= 2 and clicked <= 1.0)
            if repeated_no_effect or explored_not_clicked or repeated_low_value:
                keys.add(key)
        return keys

    def _save_k1_masked_snapshot(
        self,
        source_step: int,
        branch_id: int,
        state_pixels: np.ndarray,
        ui_elements: list[Any],
        masked_keys: set[str],
        selected_key: str | None = None,
    ) -> None:
        if not self.save_explore_masks or not self.explore_mask_output_dir:
            return
        if not masked_keys:
            return
        try:
            image = Image.fromarray(np.array(state_pixels, copy=True))
            draw = ImageDraw.Draw(image, "RGBA")
            masked_count = 0
            for idx, element in enumerate(ui_elements):
                bbox = getattr(element, "bbox_pixels", None)
                if bbox is None:
                    continue
                key = self._element_key(idx, element)
                rect = [int(bbox.x_min), int(bbox.y_min), int(bbox.x_max), int(bbox.y_max)]
                if key in masked_keys:
                    draw.rectangle(rect, fill=(0, 0, 0, 150), outline=(255, 70, 70, 230), width=3)
                    masked_count += 1
                elif selected_key is not None and key == selected_key:
                    draw.rectangle(rect, outline=(60, 220, 60, 255), width=4)
            if masked_count <= 0:
                return
            file_name = f"step_{int(source_step):03d}_branch_{int(branch_id):02d}_k1_masked.png"
            out_path = os.path.join(self.explore_mask_output_dir, file_name)
            image.save(out_path)
            self._emit_log(
                f"step={source_step} branch={branch_id} k1_mask_saved path={out_path} masked={masked_count}",
                tag="EXPLORE",
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._emit_log(
                f"step={source_step} branch={branch_id} k1_mask_save_failed error={exc}",
                tag="EXPLORE",
            )

    def _explore_worker(
        self,
        goal: str,
        history_tail: list[str],
        clues_text: str,
        source_step: int,
    ) -> None:
        self._decay_visit_counts(decay=0.92)
        goal_queries = _extract_task_queries(goal)
        runtime_queries = self._runtime_queries(history_tail=history_tail, clues_text=clues_text)
        intent_flags = self._intent_flags(goal_queries, runtime_queries)
        wants_settings = self._task_wants_settings(goal_queries, runtime_queries)
        self._emit_log(
            f"step={source_step} explore_begin goal_queries={goal_queries} runtime_queries={runtime_queries} "
            f"intent_flags={intent_flags} wants_settings={wants_settings}",
            tag="EXPLORE",
        )

        max_depth = max(1, self.explore_max_depth)
        depth_topk = max(2, self.explore_leaf_width + 3)
        max_steps = max(1, self.explore_max_steps)
        max_branches = self.explore_max_branches
        if max_branches is None:
            max_branches = max(1, max_steps // max_depth)

        steps_used = 0
        branches_done = 0
        self._explore_iteration_candidates = []
        explored_root_keys: set[str] = set()

        while (
            not self._explore_stop_event.is_set()
            and branches_done < max_branches
            and steps_used < max_steps
        ):
            branch_id = branches_done + 1
            explored_branch_keys: set[str] = set()
            self._emit_log(
                f"step={source_step} branch={branch_id} begin steps_used={steps_used}/{max_steps}",
                tag="EXPLORE",
            )

            with self._ui_lock:
                root_state = self.env.get_state(wait_to_stabilize=False)
            same_root, same_reason = self._same_root_page(
                root_state.pixels,
                curr_activity=self._foreground_activity_name(),
            )
            if not same_root:
                rollback_info = self._rollback_to_root(
                    max_depth=max_depth + 1,
                    enable_replay=True,
                    trigger=f"explore_step_{source_step}_branch_{branch_id}_pre_root",
                )
                self._emit_log(
                    f"step={source_step} branch={branch_id} pre_root_rollback={rollback_info}",
                    tag="EXPLORE",
                )
                with self._ui_lock:
                    root_state = self.env.get_state(wait_to_stabilize=False)
                same_root, same_reason = self._same_root_page(
                    root_state.pixels,
                    curr_activity=self._foreground_activity_name(),
                )
            self._emit_log(
                f"step={source_step} branch={branch_id} root_check same_root={same_root} by={same_reason}",
                tag="EXPLORE",
            )
            if not same_root:
                self._emit_log(
                    f"step={source_step} branch={branch_id} root_check_failed_after_recovery -> skip_branch",
                    tag="EXPLORE",
                )
                branches_done += 1
                time.sleep(0.05)
                continue
            root_screen_flags = self._screen_mode_flags(root_state.ui_elements)
            self._emit_log(
                f"step={source_step} branch={branch_id} root_screen_flags={root_screen_flags}",
                tag="EXPLORE",
            )
            if bool(root_screen_flags.get("keyboard")) and not bool(intent_flags.get("input")):
                self._emit_log(
                    f"step={source_step} branch={branch_id} keyboard_detected_without_input_intent -> stop_explore",
                    tag="EXPLORE",
                )
                break

            branch_max_depth = max_depth
            branch_depth_topk = depth_topk
            if bool(root_screen_flags.get("keyboard")):
                branch_max_depth = 1
                branch_depth_topk = min(depth_topk, 2)
            elif bool(root_screen_flags.get("choice")) and not bool(intent_flags.get("select")):
                branch_max_depth = 1
                branch_depth_topk = min(depth_topk, 2)
            elif bool(root_screen_flags.get("dialog")) and not bool(intent_flags.get("select")):
                branch_max_depth = min(branch_max_depth, 2)
                branch_depth_topk = min(depth_topk, 3)

            root_candidates, n_candidates = self._pick_topk(
                ui_elements=root_state.ui_elements,
                goal_queries=goal_queries,
                runtime_queries=runtime_queries,
                k=max(branch_depth_topk, max_branches + 2),
                avoid_keys=set(explored_root_keys),
                hard_avoid=True,
                intent_flags=intent_flags,
            )
            root_filter_stats = dict(self._last_filter_stats or {})
            if root_filter_stats:
                self._emit_log(
                    f"step={source_step} branch={branch_id} root_filter_level={root_filter_stats.get('filter_level')} "
                    f"interactive_total={root_filter_stats.get('interactive_total')} "
                    f"removed_intent_mismatch={root_filter_stats.get('removed_intent_mismatch')}",
                    tag="EXPLORE",
                )
            if not root_candidates:
                if explored_root_keys:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} no_root_candidates_with_hard_avoid -> retry_relaxed",
                        tag="EXPLORE",
                    )
                    root_candidates, n_candidates = self._pick_topk(
                        ui_elements=root_state.ui_elements,
                        goal_queries=goal_queries,
                        runtime_queries=runtime_queries,
                        k=max(branch_depth_topk, max_branches + 2),
                        avoid_keys=set(),
                        hard_avoid=False,
                        intent_flags=intent_flags,
                    )
                if not root_candidates:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} no_root_candidates remaining",
                        tag="EXPLORE",
                    )
                    explored_root_keys.clear()
                    branches_done += 1
                    time.sleep(0.05)
                    continue
            root_cand, skipped_root = self._select_depth_candidate(
                root_candidates,
                semantic_low=0.45,
                intent_flags=intent_flags,
            )
            if root_cand is None:
                self._emit_log(
                    f"step={source_step} branch={branch_id} root_selection_failed",
                    tag="EXPLORE",
                )
                break
            if skipped_root > 0:
                self._emit_log(
                    f"step={source_step} branch={branch_id} root_candidate_filter_skipped="
                    f"{skipped_root}/{len(root_candidates)}",
                    tag="EXPLORE",
                )
            explored_root_keys.add(root_cand.key)
            explored_branch_keys.add(root_cand.key)

            useless_keys = self._collect_useless_keys_for_mask(root_state.ui_elements)
            self._save_k1_masked_snapshot(
                source_step=source_step,
                branch_id=branch_id,
                state_pixels=root_state.pixels,
                ui_elements=root_state.ui_elements,
                masked_keys=useless_keys,
                selected_key=root_cand.key,
            )

            branch_candidate = {"branch_id": branch_id, "trunk": None, "leaf_observations": []}
            obs_root = self._click_and_record(
                cand=root_cand,
                ui_elements=root_state.ui_elements,
                before_pixels=root_state.pixels,
                n_candidates=n_candidates,
                depth=1,
                branch_id=branch_id,
                source_step=source_step,
            )
            if obs_root is None:
                self._emit_log(
                    f"step={source_step} branch={branch_id} root_click_failed",
                    tag="EXPLORE",
                )
                rollback_info = self._rollback_to_root(
                    max_depth=max_depth + 1,
                    enable_replay=True,
                    trigger=f"explore_step_{source_step}_branch_{branch_id}_after_root_fail",
                )
                self._emit_log(
                    f"step={source_step} branch={branch_id} rollback_result={rollback_info}",
                    tag="EXPLORE",
                )
                branches_done += 1
                time.sleep(0.05)
                continue

            branch_candidate["trunk"] = obs_root
            steps_used += 1
            self._emit_log(
                f"step={source_step} branch={branch_id} depth=1 root_action={obs_root.get('action_type')} "
                f"node={obs_root.get('node_match_text')} score={obs_root.get('best_sim')} "
                f"to_page=({obs_root.get('page_hint')})",
                tag="EXPLORE",
            )

            current_depth = 1
            while (
                current_depth < branch_max_depth
                and steps_used < max_steps
                and not self._explore_stop_event.is_set()
            ):
                with self._ui_lock:
                    state = self.env.get_state(wait_to_stabilize=False)
                screen_flags = self._screen_mode_flags(state.ui_elements)
                if bool(screen_flags.get("keyboard")) and not bool(intent_flags.get("input")):
                    self._emit_log(
                        f"step={source_step} branch={branch_id} depth={current_depth + 1} "
                        f"keyboard_detected_without_input_intent -> stop_branch",
                        tag="EXPLORE",
                    )
                    break
                depth_k = branch_depth_topk
                if bool(screen_flags.get("keyboard")):
                    depth_k = min(depth_k, 2)
                if bool(screen_flags.get("choice")) and not bool(intent_flags.get("select")):
                    depth_k = min(depth_k, 2)
                if bool(screen_flags.get("dialog")) and not bool(intent_flags.get("select")):
                    depth_k = min(depth_k, 3)
                depth_candidates, n_candidates = self._pick_topk(
                    ui_elements=state.ui_elements,
                    goal_queries=goal_queries,
                    runtime_queries=runtime_queries,
                    k=depth_k,
                    avoid_keys=set(explored_branch_keys),
                    hard_avoid=True,
                    intent_flags=intent_flags,
                )
                if not depth_candidates:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} depth={current_depth + 1} no_candidates",
                        tag="EXPLORE",
                    )
                    break
                if not bool(intent_flags.get("nav")):
                    non_back = [
                        cand
                        for cand in depth_candidates
                        if not self._is_back_navigation_element(state.ui_elements[cand.index])
                    ]
                    if non_back:
                        depth_candidates = non_back
                selected_cand, skipped_depth = self._select_depth_candidate(
                    depth_candidates,
                    semantic_low=0.50,
                    intent_flags=intent_flags,
                )
                if selected_cand is None:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} depth={current_depth + 1} selection_failed",
                        tag="EXPLORE",
                    )
                    break
                if skipped_depth > 0:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} depth={current_depth + 1} "
                        f"candidate_filter_skipped={skipped_depth}/{len(depth_candidates)}",
                        tag="EXPLORE",
                    )
                explored_branch_keys.add(selected_cand.key)
                obs = self._click_and_record(
                    cand=selected_cand,
                    ui_elements=state.ui_elements,
                    before_pixels=state.pixels,
                    n_candidates=n_candidates,
                    depth=current_depth + 1,
                    branch_id=branch_id,
                    source_step=source_step,
                )
                if obs is None:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} depth={current_depth + 1} click_failed",
                        tag="EXPLORE",
                    )
                    break
                if bool(obs.get("is_useful")):
                    score_detail = obs.get("score_detail") or {}
                    leaf_sim = float(score_detail.get("similarity", 0.0))
                    leaf_score = float(score_detail.get("score", 0.0))
                    useful_by_change = bool(obs.get("useful_by_change"))
                    match_text = _normalize_space(obs.get("node_match_text") or "")
                    looks_like_settings = self._is_settings_like_text(match_text)
                    min_sim = 0.10
                    min_score = 0.08
                    if looks_like_settings and not wants_settings:
                        min_sim = 0.18
                        min_score = 0.14
                    keep_leaf = bool(
                        (leaf_sim >= min_sim or leaf_score >= min_score)
                        or (useful_by_change and leaf_sim >= 0.05 and not looks_like_settings)
                    )
                    if keep_leaf:
                        branch_candidate["leaf_observations"].append(obs)
                    else:
                        self._emit_log(
                            f"step={source_step} branch={branch_id} depth={current_depth + 1} "
                            f"discarded_low_relevance_leaf sim={leaf_sim:.3f} score={leaf_score:.3f} "
                            f"useful_by_change={useful_by_change} settings_like={looks_like_settings}",
                            tag="EXPLORE",
                        )
                else:
                    self._emit_log(
                        f"step={source_step} branch={branch_id} depth={current_depth + 1} discarded_non_useful "
                        f"node={obs.get('node_match_text')} effect_delta={obs.get('effect_delta')} "
                        f"page_hash_diff={obs.get('page_hash_diff')} low_value={obs.get('low_value_hit')}",
                        tag="EXPLORE",
                    )
                self._emit_log(
                    f"step={source_step} branch={branch_id} depth={current_depth + 1} path_action={obs.get('action_type')} "
                    f"node={obs.get('node_match_text')} score={obs.get('best_sim')} "
                    f"to_page=({obs.get('page_hint')})",
                    tag="EXPLORE",
                )
                steps_used += 1
                current_depth += 1

            if self._explore_stop_event.is_set():
                self._emit_log(
                    f"step={source_step} branch={branch_id} interrupted_before_rollback",
                    tag="EXPLORE",
                )
                break

            rollback_info = self._rollback_to_root(
                max_depth=max_depth + 1,
                enable_replay=True,
                trigger=f"explore_step_{source_step}_branch_{branch_id}",
            )
            self._emit_log(
                f"step={source_step} branch={branch_id} rollback_result={rollback_info}",
                tag="EXPLORE",
            )
            if not bool(rollback_info.get("success")):
                self._emit_log(
                    f"step={source_step} branch={branch_id} dropped reason=rollback_not_restored",
                    tag="EXPLORE",
                )
                branches_done += 1
                time.sleep(0.05)
                continue
            trunk_obs = branch_candidate.get("trunk") or {}
            trunk_useful = bool(trunk_obs.get("is_useful"))
            has_useful_leaf = bool(branch_candidate.get("leaf_observations"))
            if trunk_obs and (trunk_useful or has_useful_leaf):
                self._explore_iteration_candidates.append(branch_candidate)
                self._emit_log(
                    f"step={source_step} branch={branch_id} recorded "
                    f"trunk_useful={trunk_useful} leaves={len(branch_candidate.get('leaf_observations', []))}",
                    tag="EXPLORE",
                )
            else:
                self._emit_log(
                    f"step={source_step} branch={branch_id} dropped "
                    f"reason=low_value_or_no_signal trunk_useful={trunk_useful} "
                    f"leaves={len(branch_candidate.get('leaf_observations', []))}",
                    tag="EXPLORE",
                )
            branches_done += 1
            time.sleep(0.05)
        self._emit_log(
            f"step={source_step} explore_end branches={branches_done} steps_used={steps_used} "
            f"candidates={len(self._explore_iteration_candidates)}",
            tag="EXPLORE",
        )

    # -----------------------------
    # Clues
    # -----------------------------
    @staticmethod
    def _extract_keywords(text: str, limit: int = 6) -> list[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\\-]{2,}", (text or "").lower())
        stop = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "using",
            "click",
            "button",
            "open",
            "close",
            "page",
            "screen",
            "app",
            "action",
            "task",
            "branch",
            "observed",
            "likely",
        }
        out = []
        seen = set()
        for token in tokens:
            if token in stop or token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _clean_clue_text(text: str) -> str:
        value = _normalize_space(text)
        if not value:
            return ""
        low = value.lower()
        noisy_patterns = [
            "adb shell",
            "content://",
            "[ ]",
            "- [ ]",
            "yyyy-mm-dd",
            "date only",
            "time only",
        ]
        if any(pattern in low for pattern in noisy_patterns):
            return ""
        if len(value) > 180:
            value = value[:180].rstrip(" ,.;:") + "..."
        return value

    @staticmethod
    def _infer_ui_effect(text: str) -> str:
        low = (text or "").lower()
        mapping = [
            (["more options", "menu", "settings"], "open options/menu"),
            (["search"], "open search"),
            (["record", "resume", "stop", "finish"], "control recording state"),
            (["save", "ok", "confirm", "done"], "confirm/save action"),
            (["new", "create", "add", "folder"], "create new item"),
            (["back", "up", "navigate"], "navigate backward"),
        ]
        for keys, effect in mapping:
            if any(key in low for key in keys):
                return effect
        return "possible next action"

    def _region_from_record(self, rec: dict[str, Any]) -> str:
        bounds = rec.get("bounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
            x1, y1, x2, y2 = [float(v) for v in bounds]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            coord = rec.get("coordinate")
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                cx, cy = float(coord[0]), float(coord[1])
            else:
                width, height = self.env.logical_screen_size
                cx, cy = width / 2.0, height / 2.0
        width, height = self.env.logical_screen_size
        horiz = "left" if cx < width / 3 else ("right" if cx > 2 * width / 3 else "center")
        vert = "top" if cy < height / 3 else ("bottom" if cy > 2 * height / 3 else "middle")
        if vert == "middle" and horiz == "center":
            return "center"
        return f"{vert}-{horiz}"

    def build_prompt_clues_from_candidates(
        self,
        candidates: list[dict[str, Any]],
        current_pixels: np.ndarray,
        max_items: int = 4,
        last_reasoning_action: str | None = None,
    ) -> str:
        self._last_clue_debug = {
            "status": "start",
            "n_candidates": len(candidates or []),
            "best_diff": None,
            "best_action_hit": None,
            "confidence": None,
            "n_leaves": 0,
            "n_selected": 0,
        }
        if not candidates:
            self._last_clue_debug["status"] = "empty_candidates"
            return ""

        current_hash = _phash_pixels(current_pixels)
        last_action_text = (last_reasoning_action or "").lower().strip()
        ranked = []
        for cand in candidates:
            trunk = cand.get("trunk") or {}
            trunk_hash = trunk.get("hash")
            if trunk_hash is None:
                continue
            try:
                diff = _hash_diff(current_hash, int(trunk_hash))
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            trunk_text = " ".join(
                [
                    str(trunk.get("node_text") or ""),
                    str(trunk.get("node_desc") or ""),
                    str(trunk.get("node_match_text") or ""),
                ]
            ).lower()
            action_hit = 0
            if last_action_text and trunk_text:
                if any(k in trunk_text for k in self._extract_keywords(last_action_text, limit=5)):
                    action_hit = 1
            rank_score = float(diff) - (2.0 * action_hit)
            ranked.append((rank_score, diff, action_hit, cand))

        if not ranked:
            self._last_clue_debug["status"] = "no_ranked_candidates"
            return ""

        ranked.sort(key=lambda item: item[0])
        _, best_diff, best_action_hit, best = ranked[0]
        self._last_clue_debug["best_diff"] = int(best_diff)
        self._last_clue_debug["best_action_hit"] = int(best_action_hit)

        if best_diff <= 12:
            confidence = "high"
        elif best_diff <= 24:
            confidence = "medium"
        elif best_diff <= 36:
            confidence = "low"
        else:
            confidence = "very_low"

        self._last_clue_debug["confidence"] = confidence
        trunk = best.get("trunk") or {}
        leaves = best.get("leaf_observations") or []
        self._last_clue_debug["n_leaves"] = len(leaves)
        if confidence == "very_low" and int(best_action_hit or 0) <= 0:
            self._last_clue_debug["status"] = "suppressed_very_low_confidence"
            self._last_clue_debug["n_selected"] = 0
            return ""

        trunk_text = self._clean_clue_text(
            trunk.get("node_text") or trunk.get("node_desc") or trunk.get("node_match_text") or ""
        )
        trunk_text = trunk_text or str(trunk.get("node_match_text") or trunk.get("node_desc") or "")
        trunk_text = self._clue_text_snippet(trunk_text, max_chars=160)
        lines = [
            "[Parallel Exploration Clues]",
            (
                f"- matched_branch_id={best.get('branch_id')}, page_hash_diff={best_diff}, "
                f"previous_action_overlap={best_action_hit}, confidence={confidence}"
            ),
            (
                f"- trunk_action_type={trunk.get('action_type')}, trunk_region={self._region_from_record(trunk)}, "
                f"trunk_coordinate={trunk.get('coordinate')}, trunk_bounds={trunk.get('bounds')}, "
                f"trunk_text={trunk_text}, trunk_resource_id={trunk.get('node_resource_id')}"
            ),
            "- candidate_next_actions:",
        ]

        goal_queries = _extract_task_queries(" ".join(self.history[-3:]))
        scored: list[tuple[float, float, float, dict[str, Any]]] = []
        min_rel = 0.14
        min_prior = 0.08
        for leaf in leaves:
            text = self._clean_clue_text(
                leaf.get("node_text") or leaf.get("node_desc") or leaf.get("node_match_text") or ""
            )
            if not text:
                continue
            rel = float(self._semantic_similarity(goal_queries, text)) if goal_queries else 0.0
            prior = float((leaf.get("score_detail") or {}).get("score", 0.0))
            if rel < min_rel and prior < min_prior:
                continue
            # Prefer semantically relevant leaves, fallback to explore score if semantics are weak.
            rank = rel * 0.75 + prior * 0.25
            scored.append((rank, rel, prior, leaf))
        scored.sort(key=lambda item: item[0], reverse=True)

        added = 0
        seen = set()
        k2_texts = []
        for _, rel, prior, leaf in scored:
            text = self._clean_clue_text(
                leaf.get("node_text") or leaf.get("node_desc") or leaf.get("node_match_text") or ""
            )
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            pos = self._region_from_record(leaf)
            effect = self._infer_ui_effect(text)
            text = self._clue_text_snippet(text, max_chars=140)
            lines.append(
                (
                    f"{added + 1}. action_type={leaf.get('action_type')}, coordinate={leaf.get('coordinate')}, "
                    f"bounds={leaf.get('bounds')}, region={pos}, text={text}, effect={effect}, "
                    f"semantic_relevance={rel:.3f}, explore_score={prior:.3f}"
                )
            )
            k2_texts.append(text)
            added += 1
            if added >= max_items:
                break

        self._last_clue_debug["n_selected"] = int(added)
        self._last_clue_debug["status"] = "ok" if added > 0 else "no_selected_clues"
        if added <= 0:
            return ""

        keywords = self._extract_keywords(" ".join(k2_texts), limit=6)
        if keywords:
            lines.append(f"- candidate_keywords={', '.join(keywords)}")
        lines.append("")
        out = "\n".join(lines)
        if len(out) > 1400:
            out = out[:1400].rstrip() + "\n"
        return out

    def get_last_clue_debug_lines(self) -> list[str]:
        d = self._last_clue_debug or {}
        return [
            f"[ClueDebug] status={d.get('status')}",
            f"[ClueDebug] n_candidates={d.get('n_candidates')} n_leaves={d.get('n_leaves')} selected={d.get('n_selected')}",
            f"[ClueDebug] best_diff={d.get('best_diff')} action_hit={d.get('best_action_hit')} confidence={d.get('confidence')}",
        ]

    # -----------------------------
    # Reasoning prompt helpers
    # -----------------------------
    def _build_hints(self, ui_elements: list[Any]) -> list[ExplorerHint]:
        hints: list[ExplorerHint] = []
        for idx, element in enumerate(ui_elements):
            if not self._is_valid_element(element):
                continue
            if not self._is_interactive(element):
                continue

            score = 0.0
            if idx not in self._recent_indices:
                score += 2.0
            if bool(getattr(element, "is_clickable", False)):
                score += 1.2
            if bool(getattr(element, "is_editable", False)):
                score += 1.2
            if bool(getattr(element, "is_long_clickable", False)):
                score += 0.6
            if bool(getattr(element, "is_scrollable", False)):
                score += 0.4
            if getattr(element, "text", ""):
                score += 1.0
            if getattr(element, "content_description", ""):
                score += 0.8

            hints.append(ExplorerHint(index=idx, score=score, label=_element_hint_compact_label(element)))
        hints.sort(key=lambda hint: hint.score, reverse=True)
        return hints[: self.max_hints]

    def _history_text(self) -> str:
        if not self.history:
            return "None yet."
        tail = self.history[-self.max_history :]
        return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(tail))

    @staticmethod
    def _simplify_history_item(item: str) -> str:
        text = _normalize_space(item)
        if not text:
            return "wait"

        action = "wait"
        action_match = re.search(r"action=([a-zA-Z_]+)", text, flags=re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip().lower()

        coord_match = re.search(r"coordinate=\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text)
        coord_text = ""
        if coord_match:
            coord_text = f"[{coord_match.group(1)}, {coord_match.group(2)}]"

        element_match = re.search(r"element_id=(\d+)", text)
        element_text = f"#{element_match.group(1)}" if element_match else ""

        text_match = re.search(r"text=([^|,]+)", text, flags=re.IGNORECASE)
        arg_text = _normalize_space(text_match.group(1)) if text_match else ""
        if len(arg_text) > 28:
            arg_text = arg_text[:28].rstrip(" ,.;:") + "..."

        if action in {"click", "long_press", "tap"}:
            if coord_text:
                return f"{action} {coord_text}"
            if element_text:
                return f"{action} {element_text}"
            return action
        if action in {"type", "input_text"}:
            if coord_text and arg_text:
                return f"type {coord_text} \"{arg_text}\""
            if arg_text:
                return f"type \"{arg_text}\""
            return "type"
        if action in {"swipe", "scroll"}:
            direction_match = re.search(r"direction=([^|,]+)", text, flags=re.IGNORECASE)
            direction = _normalize_space(direction_match.group(1)) if direction_match else ""
            return f"swipe {direction}".strip()
        if action in {"open", "open_app"}:
            return f"open_app \"{arg_text}\"" if arg_text else "open_app"
        if action == "system_button":
            button_match = re.search(r"button=([^|,]+)", text, flags=re.IGNORECASE)
            button = _normalize_space(button_match.group(1)) if button_match else ""
            return f"system_button {button}".strip()
        if action in {"terminate", "status"}:
            status_match = re.search(r"status=([^|,]+)", text, flags=re.IGNORECASE)
            status = _normalize_space(status_match.group(1)) if status_match else ""
            return f"terminate {status}".strip()
        if action == "answer":
            return "answer"
        return action

    def _history_prompt_text(self, max_items: int = 8) -> str:
        if not self.history:
            return "None yet."
        tail = self.history[-max(1, int(max_items)) :]
        lines = []
        for idx, item in enumerate(tail, start=1):
            lines.append(f"{idx}. {self._simplify_history_item(item)}")
        return "\n".join(lines)

    @staticmethod
    def _hints_text(hints: list[ExplorerHint]) -> str:
        if not hints:
            return "None."
        lines = []
        for rank, hint in enumerate(hints, start=1):
            lines.append(
                f"- rank={rank}, element_id={hint.index}, score={hint.score:.2f}, {hint.label}"
            )
        return "\n".join(lines)

    @staticmethod
    def _safe_center_from_element(element: Any) -> tuple[int, int] | None:
        bbox = getattr(element, "bbox_pixels", None)
        if bbox is None:
            return None
        return int((bbox.x_min + bbox.x_max) / 2.0), int((bbox.y_min + bbox.y_max) / 2.0)

    @staticmethod
    def _index_from_coordinate(
        ui_elements: list[Any],
        x: int | None,
        y: int | None,
    ) -> int | None:
        if x is None or y is None:
            return None
        hit: tuple[int, float] | None = None
        nearest: tuple[int, float] | None = None
        for idx, element in enumerate(ui_elements):
            bbox = getattr(element, "bbox_pixels", None)
            if bbox is None:
                continue
            x_min = int(bbox.x_min)
            y_min = int(bbox.y_min)
            x_max = int(bbox.x_max)
            y_max = int(bbox.y_max)
            cx = int((x_min + x_max) / 2.0)
            cy = int((y_min + y_max) / 2.0)
            dist = float((cx - x) ** 2 + (cy - y) ** 2)
            if nearest is None or dist < nearest[1]:
                nearest = (idx, dist)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                area = float(max(1, (x_max - x_min) * (y_max - y_min)))
                if hit is None or area < hit[1]:
                    hit = (idx, area)
        if hit is not None:
            return hit[0]
        if nearest is not None:
            return nearest[0]
        return None

    def _fallback_explore(
        self,
        hints: list[ExplorerHint],
        ui_elements: list[Any],
    ) -> tuple[json_action.JSONAction, dict[str, Any]]:
        for hint in hints:
            if hint.index not in self._recent_indices:
                center = None
                if 0 <= hint.index < len(ui_elements):
                    center = self._safe_center_from_element(ui_elements[hint.index])
                if center is not None:
                    return (
                        json_action.JSONAction(action_type=json_action.CLICK, x=center[0], y=center[1]),
                        {
                            "name": "mobile_use",
                            "arguments": {
                                "action": "click",
                                "element_id": hint.index,
                                "coordinate": [center[0], center[1]],
                            },
                        },
                    )
                return (
                    json_action.JSONAction(action_type=json_action.CLICK, index=hint.index),
                    {"name": "mobile_use", "arguments": {"action": "click", "element_id": hint.index}},
                )
        if hints:
            idx = hints[0].index
            center = None
            if 0 <= idx < len(ui_elements):
                center = self._safe_center_from_element(ui_elements[idx])
            if center is not None:
                return (
                    json_action.JSONAction(action_type=json_action.CLICK, x=center[0], y=center[1]),
                    {
                        "name": "mobile_use",
                        "arguments": {
                            "action": "click",
                            "element_id": idx,
                            "coordinate": [center[0], center[1]],
                        },
                    },
                )
            return (
                json_action.JSONAction(action_type=json_action.CLICK, index=idx),
                {"name": "mobile_use", "arguments": {"action": "click", "element_id": idx}},
            )
        return (
            json_action.JSONAction(action_type=json_action.WAIT),
            {"name": "mobile_use", "arguments": {"action": "wait"}},
        )

    @staticmethod
    def _tool_call_text(tool_call: dict[str, Any], source: str) -> str:
        args = dict(tool_call.get("arguments") or {})
        action = str(args.get("action") or args.get("action_type") or "unknown")
        idx = _target_index(args)
        parts = [f"[{source}] action={action}"]
        if idx is not None:
            parts.append(f"element_id={idx}")
        if "coordinate" in args:
            parts.append(f"coordinate={args.get('coordinate')}")
        if "x" in args or "y" in args:
            parts.append(f"x={args.get('x')}")
            parts.append(f"y={args.get('y')}")
        if "start_coordinate" in args:
            parts.append(f"start_coordinate={args.get('start_coordinate')}")
        if "end_coordinate" in args:
            parts.append(f"end_coordinate={args.get('end_coordinate')}")
        if "direction" in args:
            parts.append(f"direction={args.get('direction')}")
        if "text" in args:
            parts.append(f"text={args.get('text')}")
        if "button" in args:
            parts.append(f"button={args.get('button')}")
        if "status" in args:
            parts.append(f"status={args.get('status')}")
        if "goal_status" in args:
            parts.append(f"goal_status={args.get('goal_status')}")
        if "app_name" in args:
            parts.append(f"app_name={args.get('app_name')}")
        return ", ".join(parts)

    # -----------------------------
    # Step
    # -----------------------------
    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        self._ensure_task_context(goal)
        step_start_ts = time.time()
        step_start_perf = time.perf_counter()
        # Defensive: ensure no stale explorer thread leaks into next reasoning step.
        self._stop_explorer_thread()
        step_no = len(self.actions) + 1
        self._step_separator(step_no=step_no, phase="start", goal=goal)
        state = self.get_post_transition_state()
        step_start_pixels = np.array(state.pixels, copy=True)
        ui_elements = state.ui_elements
        hints = self._build_hints(ui_elements)
        self._emit_log(
            f"step={step_no} current_page=({self._state_page_hint(state)})",
            tag="STEP",
        )
        self._emit_log_block(
            title=f"step={step_no} action_history",
            content=self._history_text(),
            tag="REASON",
        )
        self._emit_log_block(
            title=f"step={step_no} reasoning_hints",
            content=self._hints_text(hints),
            tag="REASON",
        )
        self._emit_log_block(
            title=f"step={step_no} ui_tree_summary",
            content=self._state_page_hint(state, max_cues=12),
            tag="STEP",
        )

        delta = _pixel_delta(self._last_pixels, state.pixels)
        if self._last_action_text and delta is not None and delta <= self.no_effect_delta_threshold:
            self._no_effect_repeat += 1
            self._execution_feedback = (
                f"Previous action had tiny change (pixel_delta={delta:.3f}). "
                "Avoid same element/action."
            )
        elif delta is not None:
            self._no_effect_repeat = 0
            self._execution_feedback = ""

        clues = ""
        if self._pending_explore_payload:
            pending_candidates = self._pending_explore_payload.get("candidates") or []
            source_step = self._pending_explore_payload.get("source_step")
            clue_text = self.build_prompt_clues_from_candidates(
                candidates=pending_candidates,
                current_pixels=state.pixels,
                max_items=4,
                last_reasoning_action=(self.history[-1] if self.history else ""),
            )
            if clue_text:
                clues = (
                    f"[Clue Source] exploration_step={source_step} -> reasoning_step={len(self.actions) + 1}\n"
                    + clue_text
                )
        if self._execution_feedback:
            fb = f"[Execution Feedback from previous step]\n{self._execution_feedback}\n"
            clues = (clues + "\n" + fb) if clues else fb

        self._emit_log_block(
            title=f"step={step_no} clues",
            content=clues or "None.",
            tag="REASON",
        )
        self._start_explorer_thread(
            goal=goal,
            history_tail=self.history[-3:],
            clues_text=clues,
            source_step=step_no,
        )
        self._wait_for_explore_warmup(step_no=step_no)
        warmup_action_count = self._get_explore_action_count()

        prompt_history = self._history_prompt_text(max_items=8)
        user_text = (
            f"Task: {goal}\n\n"
            f"History:\n{prompt_history}\n\n"
            f"Execution feedback:\n{self._execution_feedback or 'None.'}\n\n"
            f"Parallel explorer clues:\n{clues or 'None.'}\n\n"
            f"Explorer hints:\n{self._hints_text(hints)}"
        )
        prompt_log_text = (
            "[VLM text input | system]\n"
            + SYSTEM_PROMPT
            + "\n\n[VLM text input | user]\n"
            + user_text
        )
        self._emit_log_block(
            title=f"step={step_no} vlm_text_input",
            content=prompt_log_text,
            tag="REASON",
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": _to_data_url(state.pixels)}},
                ],
            },
        ]

        response, _, _ = self.vllm.predict_mm("", [], messages=messages)
        self._emit_log_block(
            title=f"step={step_no} llm_response",
            content=str(response),
            tag="REASON",
        )

        explore_candidates = self._stop_explorer_thread()
        explore_thread_clean = bool(self._explore_thread_stop_clean)
        if not explore_thread_clean:
            self._emit_log(
                f"step={step_no} explorer_thread_not_stopped_cleanly -> guard_mode",
                tag="STEP",
            )
        rollback_info = self._ensure_root_before_reasoning(
            step_no=step_no,
            max_attempts=2,
        )
        self._emit_log(
            f"step={step_no} post_reasoning_rollback={rollback_info}",
            tag="STEP",
        )
        self._pending_explore_payload = {
            "source_step": len(self.actions) + 1,
            "candidates": explore_candidates,
        }

        source = "llm"
        parse_error = None
        execution_error = None
        execution_path = None
        coord_mode_used = self.model_coordinate_mode
        fallback_index = hints[0].index if hints else None

        if not explore_thread_clean:
            source = "thread_guard_wait"
            parse_error = "explorer_thread_not_stopped_cleanly"
            action = json_action.JSONAction(action_type=json_action.WAIT)
            tool_call = {"name": "mobile_use", "arguments": {"action": "wait"}}
            self._emit_log(
                f"step={step_no} thread_guard_failed -> force wait",
                tag="REASON",
            )
        elif not bool(rollback_info.get("verified")):
            source = "rollback_guard_wait"
            parse_error = "rollback_guard_failed_not_at_root"
            action = json_action.JSONAction(action_type=json_action.WAIT)
            tool_call = {"name": "mobile_use", "arguments": {"action": "wait"}}
            self._emit_log(
                f"step={step_no} rollback_guard_failed -> force wait",
                tag="REASON",
            )
        else:
            try:
                tool_call = parse_tool_call(response)
                self._emit_log_block(
                    title=f"step={step_no} parsed_tool_call",
                    content=json.dumps(tool_call, ensure_ascii=False, indent=2),
                    tag="REASON",
                )
                action = _to_json_action(
                    tool_call,
                    ui_elements=ui_elements,
                    fallback_index=fallback_index,
                    logical_screen_size=self.env.logical_screen_size,
                    coordinate_mode=self.model_coordinate_mode,
                )
                tool_args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
                raw_action_name = str(tool_args.get("action") or tool_args.get("action_type") or "").strip().lower()
                if raw_action_name == "answer":
                    answer_text = str(tool_args.get("text") or "").strip()
                    if answer_text:
                        self.env.interaction_cache = answer_text
                self._emit_log_block(
                    title=f"step={step_no} normalized_action",
                    content=repr(action),
                    tag="REASON",
                )
                if action.action_type == json_action.UNKNOWN:
                    raise seeact_utils.ParseActionError("unknown action")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                parse_error = str(exc)
                self._emit_log(
                    f"step={step_no} parse_or_normalize_failed error={parse_error}",
                    tag="REASON",
                )
                if _looks_like_back_intent(response):
                    source = "fallback_parse_back_intent"
                    action = json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)
                    tool_call = {"name": "mobile_use", "arguments": {"action": "navigate_back"}}
                else:
                    source = "fallback_parse"
                    action, tool_call = self._fallback_explore(hints, ui_elements)

        if (
            self._no_effect_repeat >= self.force_explore_after_repeats
            and source not in {"rollback_guard_wait", "thread_guard_wait"}
            and action.action_type not in {json_action.STATUS, json_action.ANSWER}
        ):
            source = "forced_explore"
            action, tool_call = self._fallback_explore(hints, ui_elements)

        try:
            with self._ui_lock:
                execution_path = self._execute_action_with_coordinate_priority(action)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            execution_error = str(exc)
            source = "fallback_exec"
            action, tool_call = self._fallback_explore(hints, ui_elements)
            try:
                with self._ui_lock:
                    execution_path = self._execute_action_with_coordinate_priority(action)
                execution_error = None
            except Exception as exc2:  # pylint: disable=broad-exception-caught
                execution_error = f"{execution_error}; fallback_failed={exc2}"

        idx = _target_index(tool_call.get("arguments", {}))
        if idx is None:
            idx = _safe_int(getattr(action, "index", None))
        if idx is None:
            idx = self._index_from_coordinate(
                ui_elements=ui_elements,
                x=_safe_int(getattr(action, "x", None)),
                y=_safe_int(getattr(action, "y", None)),
            )
        if idx is not None and 0 <= idx < len(ui_elements):
            self._recent_indices.append(idx)

        action_text = self._tool_call_text(tool_call, source=source)
        if parse_error:
            action_text += f" | parse_error={parse_error}"
        if execution_error:
            action_text += f" | execution_error={execution_error}"
        if execution_path:
            action_text += f" | execution_path={execution_path}"
        action_text += f" | coordinate_mode={coord_mode_used}"
        self._emit_log(
            f"step={step_no} final_decision={action_text}",
            tag="STEP",
        )

        self._last_action_text = action_text
        self._last_pixels = np.array(state.pixels, copy=True)
        self.history.append(action_text)

        action_dict = action.as_dict(skip_none=True)
        if action.action_type not in {json_action.STATUS, json_action.ANSWER, json_action.UNKNOWN}:
            self._reasoning_action_history.append(action_dict)

        self.actions.append(
            {
                "response": response,
                "tool_call": tool_call,
                "action": repr(action),
                "action_dict": action_dict,
                "source": source,
                "parse_error": parse_error,
                "execution_error": execution_error,
                "execution_path": execution_path,
                "coordinate_mode": coord_mode_used,
                "clues": clues,
                "clue_debug": self.get_last_clue_debug_lines(),
                "explore_candidates_count": len(explore_candidates),
            }
        )

        done = action.action_type == json_action.STATUS
        with self._ui_lock:
            post_state = self.env.get_state(wait_to_stabilize=False)
        step_latency_sec = float(max(0.0, time.perf_counter() - step_start_perf))
        self._task_step_latencies.append(step_latency_sec)
        avg_step_sec = float(sum(self._task_step_latencies) / max(1, len(self._task_step_latencies)))
        task_elapsed_sec = (
            float(max(0.0, time.time() - float(self._task_start_ts)))
            if self._task_start_ts is not None
            else 0.0
        )
        self._emit_log(
            (
                f"step={step_no} latency_sec={step_latency_sec:.3f} "
                f"avg_step_latency_sec={avg_step_sec:.3f} task_elapsed_sec={task_elapsed_sec:.3f}"
            ),
            tag="INFO",
        )
        self._save_step_trace(
            step_no=step_no,
            start_pixels=step_start_pixels,
            end_pixels=post_state.pixels,
            payload={
                "goal": goal,
                "step": step_no,
                "step_started_at": step_start_ts,
                "step_ended_at": time.time(),
                "step_latency_sec": step_latency_sec,
                "avg_step_latency_sec": avg_step_sec,
                "task_elapsed_sec": task_elapsed_sec,
                "reasoning_input": {
                    "history": self._history_text(),
                    "execution_feedback": self._execution_feedback or "None.",
                    "clues": clues or "None.",
                    "hints": self._hints_text(hints),
                    "vlm_user_text": user_text,
                },
                "explore": {
                    "warmup_action_count": warmup_action_count,
                    "filter_stats": dict(self._last_filter_stats or {}),
                    "candidates_count": len(explore_candidates),
                    "candidates": explore_candidates,
                },
                "llm_response_raw": str(response),
                "rollback_info": rollback_info,
                "decision": {
                    "source": source,
                    "tool_call": tool_call,
                    "action_repr": repr(action),
                    "action_dict": action_dict,
                    "action_text": action_text,
                    "parse_error": parse_error,
                    "execution_error": execution_error,
                    "execution_path": execution_path,
                    "coordinate_mode": coord_mode_used,
                },
                "return": {
                    "done": done,
                    "data": {
                        "response": response,
                        "tool_call": tool_call,
                        "action": repr(action),
                        "action_dict": action_dict,
                        "source": source,
                        "parse_error": parse_error,
                        "execution_error": execution_error,
                        "execution_path": execution_path,
                        "coordinate_mode": coord_mode_used,
                        "no_effect_repeat": self._no_effect_repeat,
                        "clues": clues,
                        "clue_debug": self.get_last_clue_debug_lines(),
                        "explore_candidates_count": len(explore_candidates),
                        "step_latency_sec": step_latency_sec,
                        "task_elapsed_sec": task_elapsed_sec,
                        "avg_step_latency_sec": avg_step_sec,
                    },
                },
            },
        )
        self._emit_log(
            f"step={step_no} post_action_page=({self._state_page_hint(post_state)})",
            tag="STEP",
        )
        self._step_separator(
            step_no=step_no,
            phase="end",
            summary=(
                f"done={done}, source={source}, action={action.action_type}, "
                f"explore_candidates={len(explore_candidates)}, latency_sec={step_latency_sec:.3f}"
            ),
        )
        if done:
            self._finalize_task_context(status="completed")
        return base_agent.AgentInteractionResult(
            done=done,
            data={
                "response": response,
                "tool_call": tool_call,
                "action": repr(action),
                "action_dict": action_dict,
                "source": source,
                "parse_error": parse_error,
                "execution_error": execution_error,
                "execution_path": execution_path,
                "coordinate_mode": coord_mode_used,
                "no_effect_repeat": self._no_effect_repeat,
                "clues": clues,
                "clue_debug": self.get_last_clue_debug_lines(),
                "explore_candidates_count": len(explore_candidates),
                "step_latency_sec": step_latency_sec,
                "task_elapsed_sec": task_elapsed_sec,
                "avg_step_latency_sec": avg_step_sec,
            },
        )

    def save_summary(self, path: str = "androidworld_exec_profile_summary.json") -> None:
        payload = {
            "agent": self.name,
            "steps": len(self.actions),
            "history_tail": self.history[-10:],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# For compatibility with existing `mm_agent.ElementTextAgent` naming.
class ElementTextAgent(ExplorerElementAgent):
    pass
