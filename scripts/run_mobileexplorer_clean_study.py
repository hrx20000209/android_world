#!/usr/bin/env python3
"""Legacy MobileExplorer study driver using app-private snapshots only.

The driver is resumable and intentionally keeps all non-shadow arms on the
same primary emulator. AndroidWorld restores /data/data app snapshots, but it
does not restore external app storage or system state. Sequential arms are not
guaranteed equivalent. The driver is disabled by default pending a verified
full-device reset; override only for explicitly labeled diagnostic runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from android_world import checkpointer  # pylint: disable=wrong-import-position,import-error
from android_world import registry  # pylint: disable=wrong-import-position,import-error


def _append(status_path: Path, event: str, **values: Any) -> None:
  row = {"event": event, "time_s": time.time(), **values}
  status_path.parent.mkdir(parents=True, exist_ok=True)
  with status_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
  print(json.dumps(row, ensure_ascii=False), flush=True)


def _coverage(path: Path) -> int:
  if not path.is_dir():
    return 0
  return len({
      str(row.get("task_template"))
      for row in checkpointer.IncrementalCheckpointer(str(path)).load()
  })


def _snapshot_packages(adb: Path, serial: str) -> list[str]:
  result = subprocess.run(
      [
          str(adb), "-s", serial, "shell",
          "find /data/data/android_world/snapshots -mindepth 1 -maxdepth 1 "
          "-type d 2>/dev/null | sort",
      ],
      check=True, capture_output=True, text=True,
  )
  return [
      Path(line.strip()).name
      for line in result.stdout.splitlines() if line.strip()
  ]


def _run_arm(
    *, root: Path, arm: str, agent: str, seed: int, expected: int,
    memory: Path | None, api_url: str, console_port: int, status: Path,
) -> None:
  checkpoint = root / arm / "checkpoints"
  observed = _coverage(checkpoint)
  if observed == expected:
    _append(status, "arm_reused", arm=arm, completed=observed)
    return
  env = os.environ.copy()
  env.update({
      "ANDROID_WORLD_LLM_API_URL": api_url,
      "ANDROID_WORLD_LLAMACPP_MODEL": "GELAB-ZERO-4B",
      "MOBILEEXPLORER_OUTPUT_PATH": str(root / f"{arm}_agent_logs"),
  })
  if memory is not None:
    env["MOBILEEXPLORER_TRAJECTORY_MEMORY"] = str(memory)
  checkpoint.mkdir(parents=True, exist_ok=True)
  _append(status, "arm_start", arm=arm, completed=observed, expected=expected)
  subprocess.run([
      sys.executable, str(REPO_ROOT / "run.py"),
      "--suite_family=android_world", f"--agent_name={agent}",
      "--n_task_combinations=1", f"--task_random_seed={seed}",
      f"--console_port={console_port}", f"--checkpoint_dir={checkpoint}",
  ], cwd=REPO_ROOT, env=env, check=True)
  observed = _coverage(checkpoint)
  if observed != expected:
    raise RuntimeError(f"{arm}: expected {expected} tasks, observed {observed}")
  _append(status, "arm_complete", arm=arm, completed=observed)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--adb", type=Path, required=True)
  parser.add_argument("--target-seed", type=int, default=4242)
  parser.add_argument("--source-seed", type=int, default=4241)
  parser.add_argument("--primary-console-port", type=int, default=5554)
  parser.add_argument("--shadow-console-port", type=int, default=5556)
  parser.add_argument("--shadow-grpc-port", type=int, default=8556)
  parser.add_argument("--primary-api-url", default="http://127.0.0.1:18083/v1/chat/completions")
  parser.add_argument("--shadow-api-url", default="http://127.0.0.1:18084/v1/chat/completions")
  parser.add_argument(
      "--allow-unsafe-app-only-snapshots", action="store_true",
      help="Diagnostic runs only: app snapshots do not reset external/system state.",
  )
  args = parser.parse_args()

  if not args.allow_unsafe_app_only_snapshots:
    raise RuntimeError(
        "Refusing a causal full study: AndroidWorld app snapshots restore only "
        "/data/data, not external storage or system state. Establish a "
        "verified full-device reset first. For diagnostic-only continuation, "
        "pass --allow-unsafe-app-only-snapshots."
    )

  root = args.root.resolve()
  status = root / "study_status.jsonl"
  expected = len(registry.TaskRegistry().get_registry(family="android_world"))
  primary_serial = f"emulator-{args.primary_console_port}"
  shadow_serial = f"emulator-{args.shadow_console_port}"
  primary_snapshots = _snapshot_packages(args.adb, primary_serial)
  shadow_snapshots = _snapshot_packages(args.adb, shadow_serial)
  if not primary_snapshots or primary_snapshots != shadow_snapshots:
    raise RuntimeError(
        "Primary/shadow app snapshot manifests are missing or inconsistent: "
        f"primary={primary_snapshots}, shadow={shadow_snapshots}"
    )
  manifest = {
      "primary_serial": primary_serial,
      "shadow_serial": shadow_serial,
      "packages": primary_snapshots,
      "package_count": len(primary_snapshots),
  }
  root.mkdir(parents=True, exist_ok=True)
  (root / "snapshot_manifest.json").write_text(
      json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
  )
  _append(status, "snapshot_preflight_complete", **manifest)

  _run_arm(
      root=root, arm="no_memory", agent="mobileexplorer_no_memory",
      seed=args.target_seed, expected=expected, memory=None,
      api_url=args.primary_api_url, console_port=args.primary_console_port,
      status=status,
  )
  _run_arm(
      root=root, arm="memory_source_primary_seed4241",
      agent="mobileexplorer_no_memory", seed=args.source_seed,
      expected=expected, memory=None, api_url=args.primary_api_url,
      console_port=args.primary_console_port, status=status,
  )

  memory = root / "trajectory_memory_frozen.jsonl"
  if not memory.is_file():
    subprocess.run([
        sys.executable,
        str(REPO_ROOT / "scripts/build_offline_trajectory_memory.py"),
        "--checkpoint-dir",
        str(root / "memory_source_primary_seed4241" / "checkpoints"),
        "--exclude-checkpoint-dir", str(root / "no_memory" / "checkpoints"),
        "--output-jsonl", str(memory),
    ], cwd=REPO_ROOT, check=True)
    _append(status, "memory_compiled", path=str(memory))

  _run_arm(
      root=root, arm="offline_trajectory", agent="mobileexplorer_offline",
      seed=args.target_seed, expected=expected, memory=memory,
      api_url=args.primary_api_url, console_port=args.primary_console_port,
      status=status,
  )
  _run_arm(
      root=root, arm="offline_replay", agent="mobileexplorer_replay",
      seed=args.target_seed, expected=expected, memory=memory,
      api_url=args.primary_api_url, console_port=args.primary_console_port,
      status=status,
  )

  mobile_root = root / "mobileexplorer"
  subprocess.run([
      sys.executable, str(REPO_ROOT / "scripts/run_mobileexplorer_full_suite.py"),
      "--memory", str(memory), "--root", str(mobile_root),
      "--task-random-seed", str(args.target_seed),
      "--primary-console-port", str(args.primary_console_port),
      "--shadow-console-port", str(args.shadow_console_port),
      "--shadow-grpc-port", str(args.shadow_grpc_port),
      "--primary-api-url", args.primary_api_url,
      "--shadow-api-url", args.shadow_api_url,
  ], cwd=REPO_ROOT, check=True)
  mobile_coverage = _coverage(mobile_root / "checkpoints")
  if mobile_coverage != expected:
    raise RuntimeError(
        f"mobileexplorer: expected {expected} tasks, observed {mobile_coverage}"
    )
  subprocess.run([
      sys.executable, str(REPO_ROOT / "scripts/analyze_mobileexplorer_full.py"),
      "--root", str(root), "--output-json", str(root / "final.json"),
  ], cwd=REPO_ROOT, check=True)
  _append(status, "study_complete", result=str(root / "final.json"))


if __name__ == "__main__":
  main()
