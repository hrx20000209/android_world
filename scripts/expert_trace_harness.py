#!/usr/bin/env python3
"""Record evaluator-validated expert actions without importing a model client.

Use --protocol or --dry-run without an emulator or AndroidWorld dependencies.
Real execution requires an exclusively assigned emulator; it resets task apps.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


ACTION_TYPES = {
    "answer", "click", "double_tap", "input_text", "keyboard_enter",
    "long_press", "navigate_back", "navigate_home", "open_app", "scroll",
    "status", "swipe", "wait",
}
ACTION_KEYS = {
    "action_type", "index", "x", "y", "text", "direction", "app_name",
    "goal_status", "keycode", "clear_text",
}
COMMANDS = {"observe", "evaluate", "finish", "quit"}
PROTOCOL = """One JSON object per line on stdin or --actions-jsonl:
  {"action": {"action_type": "click", "x": 100, "y": 200}}
  {"action": {"action_type": "input_text", "index": 3,
              "text": "<private required field>", "clear_text": true}}
  {"command": "observe"}   capture current screenshot/UI/activity
  {"command": "evaluate"} run evaluator, without declaring completion
  {"command": "finish"}   declare completion, evaluate, exit
  {"command": "quit"}     exit without validated reference
Actions may also be bare JSONAction objects. answer/status actions finish and
evaluate automatically; answer is executed to populate the evaluator cache.
EOF without finish/status/answer does not validate a reference. No VLM is used.
Use --action-origin manual_expert only for manually selected expert actions;
the default unverified may validate execution but never an expert reference.
All screenshots, goals, params, UI text, actions and library logs stay in
output-dir/private (0700 directory, 0600 files). summary.json contains no goal,
text, coordinates, arbitrary labels or exception message. Output directories
must be new. Task initialization resets task app state: never use a busy device.
--dry-run validates an actions file without importing AndroidWorld, connecting
to an emulator, calling a model, or writing files.
"""


def parse_command(line: str) -> dict:
    """Validate the transport envelope; JSONAction validates again on execution."""
    row = json.loads(line)
    if not isinstance(row, dict):
        raise ValueError("command must be an object")
    if "command" in row:
        if set(row) != {"command"} or row["command"] not in COMMANDS:
            raise ValueError("unsupported command")
        return row
    action = row.get("action", row)
    if "action" in row and set(row) != {"action"}:
        raise ValueError("unsupported envelope fields")
    if not isinstance(action, dict) or set(action) - ACTION_KEYS:
        raise ValueError("unsupported action fields")
    if action.get("action_type") not in ACTION_TYPES:
        raise ValueError("unsupported action type")
    index = action.get("index")
    xy = action.get("x"), action.get("y")
    if index is not None and any(v is not None for v in xy):
        raise ValueError("choose index or coordinates")
    if index is not None and (not isinstance(index, int) or index < 0):
        raise ValueError("index must be a nonnegative integer")
    if any(v is not None and not isinstance(v, (int, float)) for v in xy):
        raise ValueError("coordinates must be numeric")
    if any(v is not None for v in xy) and any(v is None for v in xy):
        raise ValueError("both coordinates are required")
    if action["action_type"] in {"click", "double_tap", "long_press"}:
        if index is None and any(v is None for v in xy):
            raise ValueError("pointing action requires a target")
    if action.get("direction") not in {None, "left", "right", "up", "down"}:
        raise ValueError("unsupported direction")
    if action["action_type"] in {"scroll", "swipe"} and not action.get("direction"):
        raise ValueError("scroll/swipe requires direction")
    if action["action_type"] in {"input_text", "answer"}:
        if not isinstance(action.get("text"), str) or not action["text"]:
            raise ValueError("nonempty text required")
    if action["action_type"] == "open_app" and not action.get("app_name"):
        raise ValueError("app name required")
    return {"action": action}


def private_json_default(value):
    """Task params can contain SQLite dataclasses, dates and binary row IDs."""
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def private_write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2, default=private_json_default)


@contextlib.contextmanager
def capture_library_output(path: Path):
    """Keep even native/library output containing synthetic task values private."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    try:
        with path.open("x", encoding="utf-8") as log:
            os.chmod(path, 0o600)
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield lambda row: os.write(saved[0], (json.dumps(row) + "\n").encode())
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in saved:
            os.close(fd)


def run(args, commands) -> int:
    root = args.output_dir.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    private = root / "private"
    private.mkdir(mode=0o700)
    summary = {
        "schema_version": 1, "task": args.task, "suite_seed": args.seed,
        "actions": 0, "model_calls": 0, "done": False,
        "evaluator_complete": False, "evaluator_score": None,
        "action_origin": args.action_origin, "validated_execution": False,
        "validated_reference": False, "error_type": None,
    }
    env = None
    with capture_library_output(private / "runtime.log") as emit:
        try:
            repo = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(repo))
            from android_world import registry, suite_utils
            from android_world.env import env_launcher, json_action
            from PIL import Image

            task_registry = registry.TaskRegistry().get_registry("android_world")
            if args.task not in task_registry:
                raise ValueError("unknown task template")
            env = env_launcher.load_and_setup_env(
                console_port=args.console_port, grpc_port=args.grpc_port,
                adb_path=args.adb_path, emulator_setup=False,
            )
            env.reset(go_home=True)
            suite = suite_utils.create_suite(
                {args.task: task_registry[args.task]}, tasks=[args.task],
                n_task_combinations=1, seed=args.seed,
                use_identical_params=True, env=env,
            )
            task = suite[args.task][0]
            task.initialize_task(env)
            private_write(private / "task.json", {
                "task": args.task, "suite_seed": args.seed,
                "actual_seed": task.params.get("seed"), "params": task.params,
                "goal": task.goal, "complexity": task.complexity,
                "action_origin": args.action_origin,
                "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "console_port": args.console_port, "grpc_port": args.grpc_port,
                "actions_source": "actions_jsonl" if args.actions_jsonl else "stdin",
                "action_file_sha256": (
                    hashlib.sha256(args.actions_jsonl.read_bytes()).hexdigest()
                    if args.actions_jsonl else None
                ),
            })

            def observe(name):
                state = env.get_state(wait_to_stabilize=True)
                image_path = private / (name + ".png")
                Image.fromarray(state.pixels).save(image_path)
                os.chmod(image_path, 0o600)
                ui = [dataclasses.asdict(x) for x in state.ui_elements]
                forest = state.forest
                # UIAutomator exposes a lazy forest proxy; hasattr(instance)
                # triggers an unsupported gRPC lookup. UI elements above are
                # its complete usable snapshot. Serialize only concrete protos.
                if getattr(type(forest), "SerializeToString", None) is not None:
                    raw_path = private / (name + ".forest.pb")
                    raw_path.write_bytes(forest.SerializeToString())
                    os.chmod(raw_path, 0o600)
                private_write(private / (name + ".json"), {
                    "activity": env.foreground_activity_name,
                    "ui_elements": ui,
                    "logical_screen_size": list(env.logical_screen_size),
                    "screenshot_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "captured_at_unix": time.time(),
                })

            def evaluate():
                summary["evaluator_complete"] = False
                summary["evaluator_score"] = None
                summary["validated_execution"] = False
                summary["validated_reference"] = False
                summary["evaluator_score"] = float(task.is_successful(env))
                summary["evaluator_complete"] = True
                summary["validated_execution"] = (
                    summary["done"] and summary["evaluator_score"] == 1.0
                )
                summary["validated_reference"] = (
                    summary["validated_execution"] and args.action_origin == "manual_expert"
                )
                emit({"event": "evaluation", **summary})

            observe("initial")
            emit({"event": "ready", "task": args.task, "private_dir": str(private)})
            with (private / "actions.jsonl").open("x", encoding="utf-8") as log:
                os.chmod(private / "actions.jsonl", 0o600)
                observation_index = 0
                for row in commands:
                    command = row.get("command")
                    if command == "quit":
                        break
                    if command in {"evaluate", "finish"}:
                        if command == "finish":
                            summary["done"] = True
                        evaluate()
                        if command == "finish":
                            break
                        continue
                    if command == "observe":
                        observe(f"observation_{observation_index:04d}")
                        observation_index += 1
                        emit({"event": "observed", "index": observation_index - 1})
                        continue
                    index = summary["actions"]
                    action = json_action.JSONAction(**row["action"])
                    summary["evaluator_complete"] = False
                    summary["evaluator_score"] = None
                    summary["validated_execution"] = False
                    summary["validated_reference"] = False
                    private_write(private / f"step_{index:04d}_action.json", {
                        "step": index, "action": action.as_dict(),
                    })
                    observe(f"step_{index:04d}_before")
                    started = time.monotonic()
                    env.execute_action(action)
                    observe(f"step_{index:04d}_after")
                    log.write(json.dumps({
                        "step": index, "action": action.as_dict(),
                        "elapsed_s": time.monotonic() - started,
                        "before": f"step_{index:04d}_before",
                        "after": f"step_{index:04d}_after",
                    }, ensure_ascii=False) + "\n")
                    log.flush()
                    summary["actions"] += 1
                    emit({"event": "action_complete", "step": index,
                          "action_type": action.action_type})
                    if action.action_type in {"status", "answer"}:
                        summary["done"] = (
                            action.action_type == "answer" or action.goal_status == "complete"
                        )
                        evaluate()
                        break
            observe("final")
        except (Exception, KeyboardInterrupt) as error:
            summary["error_type"] = type(error).__name__
            summary["validated_execution"] = False
            summary["validated_reference"] = False
            traceback.print_exc()
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as error:
                    summary["error_type"] = type(error).__name__
                    summary["validated_execution"] = False
                    summary["validated_reference"] = False
                    traceback.print_exc()
    private_write(root / "summary.json", summary)
    print(json.dumps(summary))
    return 1 if summary["error_type"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int, default=34)
    parser.add_argument("--console-port", type=int)
    parser.add_argument("--grpc-port", type=int)
    parser.add_argument("--adb-path")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--actions-jsonl", type=Path)
    parser.add_argument("--action-origin", choices=("manual_expert", "unverified"),
                        default="unverified")
    args = parser.parse_args()
    if args.protocol:
        print(PROTOCOL)
        return 0
    if args.dry_run:
        if args.actions_jsonl is None:
            parser.error("--dry-run requires --actions-jsonl")
    elif any(getattr(args, key) is None for key in (
        "task", "console_port", "grpc_port", "adb_path", "output_dir"
    )):
        parser.error("execution requires task, console-port, grpc-port, adb-path, output-dir")
    source = args.actions_jsonl.open(encoding="utf-8") if args.actions_jsonl else sys.stdin
    try:
        if args.actions_jsonl:
            # Validate the complete file before any device interaction.
            commands = [parse_command(line) for line in source if line.strip()]
        else:
            commands = (parse_command(line) for line in source if line.strip())
        if args.dry_run:
            print(json.dumps({"valid": True, "commands": len(commands), "model_calls": 0}))
            return 0
        return run(args, commands)
    except Exception as error:
        # Never echo a malformed command or exception message containing its values.
        print(json.dumps({"valid": False, "error_type": type(error).__name__}))
        return 1
    finally:
        if source is not sys.stdin:
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
