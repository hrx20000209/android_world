"""State verification and structural action-risk classification."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from typing import Any


class RiskLevel(str, enum.Enum):
  SAFE = "SAFE"
  LOW = "LOW"
  HIGH = "HIGH"
  IRREVERSIBLE = "IRREVERSIBLE"
  UNKNOWN = "UNKNOWN"


@dataclasses.dataclass(frozen=True)
class VerificationResult:
  confidence: float
  visual_recovered: bool
  semantic_recovered: bool
  signal_matches: dict[str, bool]


class StateVerifier:
  """Combines independent signals; pHash alone can never prove recovery."""

  def verify(self, expected: Mapping[str, Any], actual: Mapping[str, Any]) -> VerificationResult:
    exp_activity = str(expected.get("activity", ""))
    act_activity = str(actual.get("activity", ""))
    activity = bool(exp_activity and exp_activity == act_activity)
    package = bool(expected.get("package") and expected.get("package") == actual.get("package"))
    structural = bool(expected.get("structural_signature") and expected.get("structural_signature") == actual.get("structural_signature"))
    visual = bool(expected.get("visual_signature") and expected.get("visual_signature") == actual.get("visual_signature"))
    selected = expected.get("selected_state") is not None and expected.get("selected_state") == actual.get("selected_state")
    signals = {"activity": activity, "package": package, "structural": structural, "visual": visual, "selected_state": selected}
    confidence = 0.25 * activity + 0.15 * package + 0.30 * structural + 0.15 * visual + 0.15 * selected
    return VerificationResult(confidence, visual and activity, activity and package and (structural or selected), signals)


def classify_action_risk(action: Mapping[str, Any], ui_context: Mapping[str, Any] | None = None) -> RiskLevel:
  """Prefer declared effects/roles over brittle literal-text rules."""
  context = dict(ui_context or {})
  effect = str(action.get("effect") or context.get("effect") or "").upper()
  action_type = str(action.get("action_type") or action.get("type") or "").upper()
  role = str(context.get("role") or "").lower()
  persistent = bool(context.get("persistent_side_effect"))
  requires_confirmation = bool(context.get("requires_confirmation"))
  if effect in {"DELETE", "PAYMENT", "PURCHASE", "SEND", "SUBMIT", "ACCOUNT_MODIFICATION", "PERSISTENT_SETTING"}:
    return RiskLevel.IRREVERSIBLE
  if persistent or requires_confirmation:
    return RiskLevel.HIGH
  if action_type in {"INPUT_TEXT", "KEYBOARD_ENTER", "LONG_PRESS", "DOUBLE_TAP"}:
    return RiskLevel.HIGH
  if action_type in {"NAVIGATE_BACK", "NAVIGATE_HOME", "WAIT"}:
    return RiskLevel.SAFE
  if (
      action_type in {"CLICK", "TAP"}
      and role in {"tab", "menu", "navigation", "imagebutton"}
      and bool(context.get("navigation_semantics") or context.get("reversible"))
  ):
    return RiskLevel.LOW
  return RiskLevel.UNKNOWN
