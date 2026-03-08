"""Ablation-2: rollback without state-match detection.

Rollback only performs BACK operations and never does pHash/activity validation,
replay recovery, or root-match checks.
"""

from __future__ import annotations

from typing import Any

from android_world.agents.explorer_agent_gelab import ExplorerElementAgent as _BaseExplorerElementAgent
from android_world.env import json_action


class ExplorerElementAgent(_BaseExplorerElementAgent):
    """Ablation-2: remove rollback detection and replay logic."""

    def __init__(self, *args: Any, **kwargs: Any):
        # Keep run.py compatibility (gelab explorer does not use this arg).
        kwargs.pop("image_downsample_scale", None)
        super().__init__(*args, **kwargs)

    def _rollback_to_root(
        self,
        max_depth: int = 2,
        enable_replay: bool = True,
        trigger: str = "",
    ) -> dict[str, Any]:
        _ = enable_replay

        result: dict[str, Any] = {
            "trigger": trigger or "unspecified",
            "success": False,
            "mode": "no_detection_back_only",
            "back_presses": 0,
            "replayed_actions": 0,
            "matched_by": "disabled",
            "replay_open_resets": [],
        }

        with self._rollback_lock:
            presses = max(1, int(max_depth))
            for _ in range(presses):
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
                    # Keep going to preserve "no-detection" semantics.
                    pass

        result["success"] = bool(result["back_presses"] > 0)
        self._emit_log(
            f"trigger={result['trigger']} rollback_no_detection back_presses={result['back_presses']}",
            tag="ROLLBACK",
        )
        return result


class ElementTextAgent(ExplorerElementAgent):
    pass
