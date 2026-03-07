"""Ablation agent: random exploration policy.

This keeps the full ExplorerElementAgent pipeline, but replaces depth candidate
selection with random selection to ablate semantic-guided exploration ranking.
"""

from __future__ import annotations

import random
from typing import Any

from android_world.agents.explorer_agent import CandidateScore
from android_world.agents.explorer_agent import ExplorerElementAgent as _BaseExplorerElementAgent
from android_world.agents.explorer_agent_utils import _phash_pixels


class ExplorerElementAgent(_BaseExplorerElementAgent):
    """Explorer ablation: randomly choose exploration candidates."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._ablation_rng = random.Random()

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

    def _select_depth_candidate(
        self,
        candidates: list[CandidateScore],
        semantic_low: float = 0.20,
        intent_flags: dict[str, bool] | None = None,
        avoid_keys: set[str] | None = None,
        hard_avoid: bool = False,
    ) -> tuple[CandidateScore | None, int]:
        _ = semantic_low
        _ = intent_flags
        if not candidates:
            return None, 0

        avoid = set(avoid_keys or set())
        skipped = 0
        pool: list[CandidateScore] = []
        for cand in candidates:
            if cand.key in avoid and hard_avoid:
                skipped += 1
                skip_map = getattr(self, "_bound_skip_count", None)
                if not isinstance(skip_map, dict):
                    skip_map = {}
                    setattr(self, "_bound_skip_count", skip_map)
                skip_map[cand.key] = int(skip_map.get(cand.key, 0)) + 1
                continue
            pool.append(cand)

        if not pool:
            pool = list(candidates)

        return self._ablation_rng.choice(pool), skipped


class ElementTextAgent(ExplorerElementAgent):
    pass
