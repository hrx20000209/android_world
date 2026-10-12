#!/usr/bin/env python3
"""Final three-arm comparison of a paired full run against 4B_2.

  python scripts/final_compare.py <root> [test_arm]

Success is the evaluator's score sum (partial scores count as in the
official tables); steps are episode lengths; success-task steps average
over each arm's own successes, and the paired rows average over tasks both
arms solved, which is the step comparison not confounded by which tasks
succeed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_full_paired import REPO  # noqa: E402


def ref_4b2() -> dict[str, tuple[float, float]]:
  out = {}
  for line in (REPO / "results" / "4B_2.txt").read_text().splitlines()[2:]:
    parts = line.split()
    if len(parts) >= 5 and parts[0][:1].isupper():
      try:
        out[parts[0]] = (float(parts[3]), float(parts[4]))
      except ValueError:
        pass
  return out


def summary(rows: list[tuple[float, float]]) -> dict:
  ok = [s for s, _ in rows]
  succ = [n for s, n in rows if s > 0]
  return {"score": sum(ok), "n": len(rows), "rate": sum(ok) / len(rows),
          "mean_steps": sum(n for _, n in rows) / len(rows),
          "success_steps": sum(succ) / max(1, len(succ))}


def main() -> int:
  root = Path(sys.argv[1])
  test = sys.argv[2] if len(sys.argv) > 2 else "full_rec"
  pairs = {}
  for line in (root / "pairs.jsonl").read_text().splitlines():
    row = json.loads(line)
    pairs[row["task"]] = row  # later rows (reruns) replace earlier ones
  ref = ref_4b2()
  tasks = [t for t in pairs if t in ref and not pairs[t]["base"]["crash"]
           and not pairs[t][test]["crash"]]
  arms = {
      "4B_2": [ref[t] for t in tasks],
      "base": [(pairs[t]["base"]["ok"], pairs[t]["base"]["steps"]) for t in tasks],
      test: [(pairs[t][test]["ok"], pairs[t][test]["steps"]) for t in tasks],
  }
  out = {"tasks": len(tasks), "arms": {k: summary(v) for k, v in arms.items()}}
  for other in ("4B_2", "base"):
    both = [i for i, t in enumerate(tasks) if arms[other][i][0] > 0 and arms[test][i][0] > 0]
    out[f"both_success_vs_{other}"] = {
        "n": len(both),
        other: sum(arms[other][i][1] for i in both) / max(1, len(both)),
        test: sum(arms[test][i][1] for i in both) / max(1, len(both)),
    }
    out[f"wins_losses_vs_{other}"] = {
        "test_only": [tasks[i] for i in range(len(tasks)) if arms[test][i][0] > arms[other][i][0]],
        "other_only": [tasks[i] for i in range(len(tasks)) if arms[test][i][0] < arms[other][i][0]],
    }
  out["inference_calls"] = {
      "base": sum(pairs[t]["base"].get("calls", 0) for t in tasks),
      test: sum(pairs[t][test].get("calls", 0) for t in tasks),
  }
  print(json.dumps(out, ensure_ascii=False, indent=1))
  (root / f"final_compare_{test}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
  return 0


if __name__ == "__main__":
  sys.exit(main())
