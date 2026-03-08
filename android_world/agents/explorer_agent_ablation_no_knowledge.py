"""Ablation-3: no page alignment + no relevance filtering + full injection.

Differences from `explorer_agent_gelab`:
1. Do not use pHash/alignment scoring when converting exploration traces to clues.
2. Do not apply query-keyword relevance filtering in candidate collection.
3. Inject all explored observations into prompt clue/summary text.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from android_world.agents.explorer_agent_gelab import ExplorerElementAgent as _BaseExplorerElementAgent


class ExplorerElementAgent(_BaseExplorerElementAgent):
    """Ablation-3 implementation."""

    def __init__(self, *args: Any, **kwargs: Any):
        # Keep run.py compatibility (gelab explorer does not use this arg).
        kwargs.pop("image_downsample_scale", None)
        super().__init__(*args, **kwargs)

    def _collect_candidates(
        self,
        ui_elements: list[Any],
        safe: bool = True,
        intent_flags: dict[str, bool] | None = None,
        query_keywords: list[str] | None = None,
        avoid_keys: set[str] | None = None,
        hard_avoid: bool = False,
        allow_back_navigation: bool = False,
    ) -> tuple[list[tuple[int, Any]], dict[str, Any]]:
        # Disable relevance filtering by query keywords.
        candidates, stats = super()._collect_candidates(
            ui_elements=ui_elements,
            safe=safe,
            intent_flags=intent_flags,
            query_keywords=[],
            avoid_keys=avoid_keys,
            hard_avoid=hard_avoid,
            allow_back_navigation=allow_back_navigation,
        )
        stats = dict(stats or {})
        if query_keywords:
            stats["filter_level"] = "no_relevance_filter_ablation"
            stats["removed_intent_mismatch"] = 0
        return candidates, stats

    def build_prompt_clues_from_candidates(
        self,
        candidates: list[dict[str, Any]],
        current_pixels: np.ndarray,
        max_items: int = 4,
        last_reasoning_action: str | None = None,
    ) -> str:
        _ = current_pixels
        _ = max_items
        _ = last_reasoning_action
        self._last_clue_debug = {
            "status": "ablation_no_alignment_no_filter",
            "n_candidates": len(candidates or []),
            "best_diff": None,
            "best_action_hit": None,
            "confidence": "disabled",
            "n_leaves": 0,
            "n_selected": 0,
        }
        if not candidates:
            return ""

        observations: list[dict[str, Any]] = []
        for cand in candidates:
            trunk = dict(cand.get("trunk") or {})
            if trunk:
                trunk["_from_trunk"] = True
                observations.append(trunk)
            observations.extend(list(cand.get("leaf_observations") or []))

        self._last_clue_debug["n_leaves"] = len(observations)
        if not observations:
            self._last_clue_debug["status"] = "no_observations"
            return ""

        lines: list[str] = ["[探索发现以下元素（全量注入，无对齐/无筛选）:]"]
        for idx, obs in enumerate(observations, start=1):
            text = self._clean_clue_text(
                obs.get("node_text") or obs.get("node_desc") or obs.get("node_match_text") or ""
            )
            text = text or "entry point"
            action_type = str(obs.get("action_type") or "click").lower()
            pos = self._region_from_record(obs)
            lines.append(f'{idx}. {action_type} "{text}" at {pos}.')

        self._last_clue_debug["n_selected"] = len(observations)
        return "\n".join(lines) + "\n"

    def _summarize_explore_candidates(self, candidates: list[dict[str, Any]], max_items: int = 4) -> str:
        _ = max_items
        if not candidates:
            return "None."

        lines: list[str] = []
        line_idx = 1
        for candidate in candidates:
            trunk = dict(candidate.get("trunk") or {})
            if trunk:
                trunk_text = self._clean_clue_text(
                    trunk.get("node_text") or trunk.get("node_desc") or trunk.get("node_match_text") or ""
                ) or "entry point"
                trunk_action = str(trunk.get("action_type") or "click").lower()
                lines.append(f'{line_idx}. {trunk_action} "{trunk_text}" (trunk).')
                line_idx += 1

            for leaf in list(candidate.get("leaf_observations") or []):
                leaf_text = self._clean_clue_text(
                    leaf.get("node_text") or leaf.get("node_desc") or leaf.get("node_match_text") or ""
                ) or "entry point"
                leaf_action = str(leaf.get("action_type") or "click").lower()
                leaf_pos = self._region_from_record(leaf)
                lines.append(f'{line_idx}. {leaf_action} "{leaf_text}" at {leaf_pos}.')
                line_idx += 1

        return "\n".join(lines) if lines else "None."


class ElementTextAgent(ExplorerElementAgent):
    pass

