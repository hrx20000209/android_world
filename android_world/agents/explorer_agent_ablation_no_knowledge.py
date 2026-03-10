"""Ablation-3: no page alignment + no relevance filter + full knowledge injection."""

from __future__ import annotations

from typing import Any

from android_world.agents.explorer_agent_gelab_light import ExplorerElementAgent as _BaseExplorerElementAgent


class ExplorerElementAgent(_BaseExplorerElementAgent):
    """Ablation-3: inject broad unfiltered exploration knowledge into prompt hint."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs.setdefault("light_explore_launcher_only", False)
        kwargs.setdefault("light_explore_require_keyword", False)
        super().__init__(*args, **kwargs)

    def _collect_probe_candidates(self, state: Any, goal: str) -> list[dict[str, Any]]:
        _ = goal
        ui_elements = list(getattr(state, "ui_elements", None) or [])
        candidates: list[dict[str, Any]] = []
        for idx, element in enumerate(ui_elements):
            if not self._is_interactive(element):
                continue
            center = self._safe_center_from_element(element)
            if center is None:
                continue
            merged = self._element_text(element)
            label = (
                (getattr(element, "text", None) or "")
                or (getattr(element, "content_description", None) or "")
                or "entry"
            )
            label = str(label).strip() or "entry"
            candidates.append(
                {
                    "index": idx,
                    "element": element,
                    "center": center,
                    "label": label,
                    "score": 1.0,
                    "merged": merged,
                }
            )
        return candidates[:12]

    def _build_hint_from_observation(
        self,
        candidate: dict[str, Any],
        changed: bool,
        after_activity: str,
    ) -> str:
        after_short = str(after_activity or "").strip().split("/")[-1]
        labels: list[str] = []
        for item in list(getattr(self, "_last_probe_candidates", [])):
            label = str(item.get("label", "")).strip()
            if label and label not in labels:
                labels.append(label)
            if len(labels) >= 12:
                break
        picked = str(candidate.get("label", "")).strip() or "entry"
        if labels and after_short:
            return (
                "Exploration (no alignment/filter): observed possible entries "
                + ", ".join(labels)
                + f"; current probe clicked \"{picked}\"; post-page={after_short}; changed={bool(changed)}."
            )
        if labels:
            return (
                "Exploration (no alignment/filter): observed possible entries "
                + ", ".join(labels)
                + f"; current probe clicked \"{picked}\"; changed={bool(changed)}."
            )
        return f"Exploration (no alignment/filter): current probe clicked \"{picked}\"."


class ElementTextAgent(ExplorerElementAgent):
    pass
