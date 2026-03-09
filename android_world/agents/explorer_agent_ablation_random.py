"""Ablation-1: random exploration target selection on light explorer."""

from __future__ import annotations

import random
from typing import Any

from android_world.agents.explorer_agent_gelab_light import ExplorerElementAgent as _BaseExplorerElementAgent


class ExplorerElementAgent(_BaseExplorerElementAgent):
    """Ablation-1: randomly choose which element to probe during exploration."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._ablation_rng = random.Random()

    def _choose_probe_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        return self._ablation_rng.choice(list(candidates))


class ElementTextAgent(ExplorerElementAgent):
    pass
