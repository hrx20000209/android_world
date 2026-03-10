"""Ablation-2: remove rollback detection (back-only, no verification)."""

from __future__ import annotations

import time
from typing import Any

from android_world.agents.explorer_agent_gelab_light import ExplorerElementAgent as _BaseExplorerElementAgent
from android_world.env import json_action


class ExplorerElementAgent(_BaseExplorerElementAgent):
    """Ablation-2: rollback only uses BACK operations, without restore checks."""

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
        _ = replay_actions
        result: dict[str, Any] = {
            "success": False,
            "mode": "no_detection_back_only",
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
        result["success"] = bool(result["back_presses"] > 0)
        print(
            f"[ROLLBACK {time.strftime('%H:%M:%S')}] "
            f"step: {step_idx + 1} rollback_no_detection back_presses={result['back_presses']}"
        )
        return result


class ElementTextAgent(ExplorerElementAgent):
    pass
