"""Run phase 1 of the parallel exploration feasibility harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from android_world.parallel_exploration.processes import run_trial


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True)
  parser.add_argument("--events-jsonl")
  args = parser.parse_args()
  config = json.loads(Path(args.config).read_text(encoding="utf-8"))
  trial = run_trial(config)
  lines = [json.dumps(event.to_dict(), ensure_ascii=False) for event in trial.events]
  if args.events_jsonl:
    output_path = Path(args.events_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  for line in lines:
    print(line)
  print(json.dumps({"result": dict(trial.result)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()
