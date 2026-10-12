#!/usr/bin/env python3
"""Upper bound on navigation savings under a PERFECT task contract.

Evaluation only - the policy never sees this. For every step t of a shadow
episode, the "intent" is the screen the model actually reached at step t+k
(hindsight). We ask the real planner, on the exact witness snapshot it had at
step t, whether a witnessed route from the current screen to that future
screen existed, and only when the model's own steps t..t+k-1 were all plain
clicks (a navigation segment). Greedy non-overlapping segments give the
number of model decisions a perfect contract could have replaced.

If this bound is ~0, eliciting a contract from the model cannot pay, whatever
its format; the bottleneck is witness supply, not intent.

  python scripts/hindsight_navigation_bound.py <root>/shadow_frontier
"""

from __future__ import annotations

import collections
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from android_world.parallel_exploration import task_navigation as tn  # noqa: E402


@dataclasses.dataclass(frozen=True)
class OracleIntent:
  target: tuple[str, str]
  package: str
  intent_id: str = "oracle"
  max_hops: int = 4

  def valid(self, task, generation):  # pylint: disable=unused-argument
    return True

  def matches(self, state):
    return (state.get("activity"), state.get("structural_signature")) == self.target


def episode(run: Path) -> dict:
  snaps = {}
  for f in sorted((run / "nav_snapshots").glob("step*.json")):
    d = json.loads(f.read_text())
    snaps[d["step"]] = d
  actions = {}
  for line in (run / "serial_events.jsonl").read_text().splitlines():
    e = json.loads(line)
    if e.get("kind") == "inference":
      actions[e["step"]] = str((e.get("action") or {}).get("action_type") or "")
  steps = sorted(snaps)
  out = {"steps": len(steps), "nav_clicks": sum(1 for s in steps if actions.get(s) == "click")}
  for label, ui_only in (("strict", False), ("ui_only", True)):
    planner = tn.TaskNavigationPlanner(allow_ui_only=ui_only)
    saved, hops, segments, t = 0, 0, [], 0
    rejections = collections.Counter()
    while t < len(steps):
      s = steps[t]
      cur = snaps[s]["current"]
      best = None
      for k in range(1, 5):
        if t + k >= len(steps):
          break
        if any(actions.get(steps[t + i]) != "click" for i in range(k)):
          break
        fut = snaps[steps[t + k]]["current"]
        intent = OracleIntent((fut["activity"], fut["structural_signature"]), cur["package"])
        snap = {"generation": snaps[s]["generation"], "edges": snaps[s]["edges"]}
        route = planner.compile(snap, cur, intent, task="", inference_cost_s=1.0)
        rejections.update(planner.last_rejections)
        if route is not None:
          best = (k, len(route.hops))
      if best:
        saved += best[0]
        hops += best[1]
        segments.append({"step": s, "decisions": best[0], "hops": best[1]})
        t += best[0]
      else:
        t += 1
    out[label] = {"decisions_replaced": saved, "macro_actions": hops,
                  "segments": segments, "rejections": dict(rejections)}
  return out


def main() -> int:
  arm = Path(sys.argv[1])
  rows = {}
  for run in sorted(p for p in arm.iterdir() if (p / "nav_snapshots").exists()):
    rows[run.name] = episode(run)
  total = {k: sum(r[k]["decisions_replaced"] for r in rows.values()) for k in ("strict", "ui_only")}
  summary = {"episodes": len(rows), "model_steps": sum(r["steps"] for r in rows.values()),
             "click_steps": sum(r["nav_clicks"] for r in rows.values()),
             "bound_decisions_replaced": total, "per_task": rows}
  (arm / "hindsight_bound.json").write_text(json.dumps(summary, indent=1))
  print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=1))
  for name, r in rows.items():
    print(f"{name:40s} steps={r['steps']:2d} strict={r['strict']['decisions_replaced']} "
          f"ui_only={r['ui_only']['decisions_replaced']}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
