"""Ablation-2: remove rollback detection (no restoration verification)."""

from __future__ import annotations

import time
from typing import Any

from android_world.agents.explorer_agent_gelab_light import ExplorerElementAgent as _BaseExplorerElementAgent
from android_world.env import json_action


class ExplorerElementAgent(_BaseExplorerElementAgent):
    """Ablation-2: keep rollback actions, but remove restore verification."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _rollback_to_probe_root(
        self,
        root_activity: str,
        root_hash: int,
        replay_actions: list[json_action.JSONAction],
        step_idx: int,
    ) -> dict[str, Any]:
        _ = root_activity
        _ = root_hash
        result: dict[str, Any] = {
            "success": True,
            "mode": "no_detection_backtrack_replay",
            "back_presses": 0,
            "replayed_actions": 0,
            "matched_by": "disabled",
        }
        back_limit = max(1, int(self.light_explore_back_limit))
        for _i in range(back_limit):
            try:
                self.env.execute_action(json_action.JSONAction(action_type=json_action.NAVIGATE_BACK))
                result["back_presses"] += 1
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        for action in replay_actions or []:
            try:
                self.env.execute_action(action)
                result["replayed_actions"] += 1
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        print(
            f"[ROLLBACK {time.strftime('%H:%M:%S')}] "
            f"step: {step_idx + 1} rollback_no_detection "
            f"back_presses={result['back_presses']} replayed={result['replayed_actions']}"
        )
        return result


class ElementTextAgent(ExplorerElementAgent):
    pass
