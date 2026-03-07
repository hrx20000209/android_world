"""Ablation agent: disable knowledge injection into reasoning prompt.

This keeps exploration and execution logic unchanged, but removes injected
exploration clues/hints from the model prompt.
"""

from __future__ import annotations

from typing import Any

from android_world.agents.explorer_agent import ExplorerElementAgent as _BaseExplorerElementAgent
from android_world.agents.explorer_agent import ExplorerHint
from android_world.agents.explorer_agent_utils import _phash_pixels


class ExplorerElementAgent(_BaseExplorerElementAgent):
    """Explorer ablation: no exploration knowledge injection in prompts."""

    def _start_explorer_thread(
        self,
        goal: str,
        history_tail: list[str],
        clues_text: str,
        source_step: int,
        trigger_reason: str = "unknown",
    ) -> None:
        if not self.enable_parallel_exploration:
            self._clear_explore_root_baseline()
            return
        self._stop_explorer_thread()
        if not self._explore_thread_stop_clean:
            self._clear_explore_root_baseline()
            self._emit_log(
                f"step={source_step} explorer_sync_start_skipped reason=previous_thread_not_stopped",
                tag="EXPLORE",
            )
            return
        self._explore_stop_event.clear()
        self._explore_progress_event.clear()
        with self._explore_action_count_lock:
            self._explore_action_count = 0
        self._explore_iteration_candidates = []
        self._explore_trigger_reason = str(trigger_reason or "unknown")
        self._clicked_bounds = set()
        self._branch_action_history = []
        self._replay_action_history = list(self._reasoning_action_history)

        with self._ui_lock:
            root_state = self.env.get_state(wait_to_stabilize=False)
        self._explore_root_pixels = root_state.pixels.copy()
        self._explore_root_hash = _phash_pixels(root_state.pixels)
        self._explore_root_activity = self._foreground_activity_name() or None
        self._explore_thread = None
        self._explore_thread_stop_clean = True
        self._emit_log(
            f"step={source_step} explorer_sync_started trigger={self._explore_trigger_reason} "
            f"root_page=({self._state_page_hint(root_state)})",
            tag="EXPLORE",
        )
        self._explore_worker(goal, history_tail, clues_text, source_step)

    def build_prompt_clues_from_candidates(
        self,
        candidates: list[dict[str, Any]],
        current_pixels,
        max_items: int = 4,
        last_reasoning_action: str | None = None,
    ) -> str:
        _ = candidates
        _ = current_pixels
        _ = max_items
        _ = last_reasoning_action
        self._last_clue_debug = {
            "status": "ablation_no_knowledge",
            "n_candidates": len(candidates or []),
            "best_diff": None,
            "best_action_hit": None,
            "confidence": None,
            "n_leaves": 0,
            "n_selected": 0,
        }
        return ""

    @staticmethod
    def _hints_text(hints: list[ExplorerHint]) -> str:
        _ = hints
        return "None."


class ElementTextAgent(ExplorerElementAgent):
    pass
