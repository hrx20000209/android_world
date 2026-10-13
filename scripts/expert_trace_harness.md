# Direct expert traces

`expert_trace_harness.py` initializes one AndroidWorld task and executes expert
JSON actions directly. It never instantiates an agent or calls a model. Use a
dedicated, idle emulator: task initialization restores app snapshots and creates
the task state. Do not point it at an emulator assigned to an active shard.

Safe checks, with no device connection or writes:

```bash
python scripts/expert_trace_harness.py --help
python scripts/expert_trace_harness.py --protocol
python scripts/expert_trace_harness.py --dry-run --actions-jsonl /private/actions.jsonl
```

After assigning an idle emulator, run using its actual console and gRPC ports:

```bash
python scripts/expert_trace_harness.py --task FilesDeleteFile --seed 34 \
  --action-origin manual_expert \
  --console-port <idle-console-port> --grpc-port <idle-grpc-port> \
  --adb-path <adb-binary> --output-dir <new-private-experiment-directory>
```

The initial `ready` event points to the private directory. Inspect `task.json`
and `initial.png` privately. Send one JSON object per stdin line. Bare actions
or an `action` envelope are accepted. `index` binds the live UI list, while
`x`/`y` use the live logical screen's pixel coordinates. Do not pass both.

```json
{"action":{"action_type":"open_app","app_name":"files"}}
{"command":"observe"}
{"action":{"action_type":"click","x":100,"y":200}}
{"action":{"action_type":"input_text","index":3,"text":"<private required field>","clear_text":true}}
{"command":"evaluate"}
{"command":"finish"}
```

`observe` adds an observation without advancing the action counter. `evaluate`
checks the current task but does not declare the expert finished. `finish`
declares completion, evaluates and exits. `quit` or stdin EOF exits without a
validated reference. A `status` action with `goal_status=complete`, or an
`answer` action carrying the private answer, also evaluates and exits. ANSWER
is executed through the environment so its interaction cache reaches the IR
evaluator. Text insertion follows the environment API, including its Enter
behavior; `clear_text=true` requests replacement, not append.

`--actions-jsonl` replays a file after validating the entire input. Reuse its
coordinates only against the same task instance and verified UI states. The
script does not promise replay validity or auto-correct expert actions.
The default `--action-origin unverified` never certifies an expert reference.
Pass `manual_expert` only when the actions were manually selected. A VLM-derived
actions file remains a model trajectory even when this harness replays it without
model calls. The private metadata records input source and the actions-file hash.

Private outputs contain the goal/params, actual instance seed, intended actions,
executed action JSONL, initial/final and before/after screenshots, UI elements,
activity, image hashes, logical screen size, and captured library output. The
private directory is 0700 and files are 0600. Keep these out of user-facing
reports. `summary.json` includes only task template, suite seed, action count,
model-call count, evaluator status/score, completion and validation flags, and
exception type and action origin. Errors and unsuccessful traces remain available
privately; `done && evaluator_complete && score == 1` with no capture/cleanup error
qualifies as validated execution; it is a validated expert reference only with
`action_origin=manual_expert`. Every later action invalidates a previous
evaluation, preventing an old success score from certifying changed state.

The script was checked with offline fake environments. No emulator run or
reference collection is implied by those checks. Runtime dependencies are the
repository's AndroidWorld environment and Pillow; help/protocol/dry-run use
only the standard library.
