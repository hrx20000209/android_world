"""Ablation agent: fixed two-back rollback for exploration.

For exploration-triggered rollback calls, this ablation always sends BACK twice
without any state verification. Reasoning rollback behavior is unchanged.
"""

from __future__ import annotations

from typing import Any

from android_world.agents.explorer_agent import ExplorerElementAgent as _BaseExplorerElementAgent
from android_world.agents.explorer_agent_utils import _phash_pixels
from android_world.env import json_action


class ExplorerElementAgent(_BaseExplorerElementAgent):
    """Explorer ablation: replace explore rollback with fixed two-back."""

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

    def _rollback_to_root(
        self,
        max_depth: int = 2,
        enable_replay: bool = True,
        trigger: str = "",
    ) -> dict[str, Any]:
        if not str(trigger or "").startswith("explore_"):
            return super()._rollback_to_root(
                max_depth=max_depth,
                enable_replay=enable_replay,
                trigger=trigger,
            )

        result: dict[str, Any] = {
            "trigger": trigger or "unspecified",
            "success": False,
            "mode": "fixed_back2",
            "back_presses": 0,
            "replayed_actions": 0,
            "matched_by": "fixed_back2_no_state_check",
            "replay_open_resets": [],
        }

        with self._rollback_lock:
            for _ in range(2):
                if self._explore_stop_event.is_set() and str(result["trigger"]).startswith("explore_"):
                    result["mode"] = "interrupted"
                    return result
                try:
                    with self._ui_lock:
                        self.env.execute_action(
                            json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)
                        )
                    result["back_presses"] += 1
                except Exception:  # pylint: disable=broad-exception-caught
                    # Keep going to preserve fixed-2-back semantics.
                    pass

        result["success"] = True
        self._emit_log(
            f"trigger={result['trigger']} rollback_fixed_back2 back_presses={result['back_presses']}",
            tag="ROLLBACK",
        )
        return result


class ElementTextAgent(ExplorerElementAgent):
    pass
