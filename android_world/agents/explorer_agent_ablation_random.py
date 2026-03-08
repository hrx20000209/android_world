"""Ablation-1: random exploration policy on GELAB explorer baseline.

This keeps the full `explorer_agent_gelab` pipeline, but replaces exploration
candidate selection with uniform random choice.
"""

from __future__ import annotations

import random
from typing import Any

from android_world.agents.explorer_agent_gelab import CandidateScore
from android_world.agents.explorer_agent_gelab import ExplorerElementAgent as _BaseExplorerElementAgent


class ExplorerElementAgent(_BaseExplorerElementAgent):
    """Ablation-1: randomly choose exploration candidates."""

    def __init__(self, *args: Any, **kwargs: Any):
        # Keep run.py compatibility (gelab explorer does not use this arg).
        kwargs.pop("image_downsample_scale", None)
        super().__init__(*args, **kwargs)
        self._ablation_rng = random.Random()

    # Keep exploration gate behavior synchronized with explorer_agent_gelab.
    def _should_start_exploration(
        self,
        step_no: int,
        goal: str,
        current_activity: str | None = None,
    ) -> tuple[bool, str]:
        return super()._should_start_exploration(
            step_no=step_no,
            goal=goal,
            current_activity=current_activity,
        )

    # Keep sequential exploration gate behavior synchronized with explorer_agent_gelab.
    def _should_start_sequential_exploration(
        self,
        step_no: int,
        goal: str,
        current_activity: str | None = None,
        ui_elements: list[Any] | None = None,
    ) -> tuple[bool, str]:
        return super()._should_start_sequential_exploration(
            step_no=step_no,
            goal=goal,
            current_activity=current_activity,
            ui_elements=ui_elements,
        )

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
