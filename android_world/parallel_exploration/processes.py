"""Process A/Process B implementation and parent trial orchestration."""

from __future__ import annotations

import dataclasses
import multiprocessing
import queue
import time
import traceback
from typing import Any, Mapping

from android_world.parallel_exploration import inference_backends
from android_world.parallel_exploration.protocol import Event
from android_world.parallel_exploration.protocol import EventKind
from android_world.parallel_exploration.protocol import ProcessRole
from android_world.parallel_exploration.protocol import require_event


def _emit(event_queue: Any, event: Event) -> None:
  event_queue.put(event)


def _inference_process(
    config: Mapping[str, Any], control_queue: Any, status_queue: Any, event_queue: Any
) -> None:
  """Process A: owns inference time and the abort timer."""
  trial_id = str(config["trial_id"])
  try:
    ready_timeout_s = float(config.get("ready_timeout_s", 5.0))
    ready = require_event(
        status_queue.get(timeout=ready_timeout_s), EventKind.EXPLORER_READY
    )
    del ready
    inference_config = dict(config["inference"])
    backend = inference_backends.create_backend(
        inference_config, seed=int(config.get("seed", 0))
    )
    progress_every = max(
        1, int(inference_config.get("decode_progress_every_n_tokens", 5))
    )
    start = Event.now(
        trial_id,
        EventKind.INFERENCE_START,
        ProcessRole.INFERENCE,
        {"backend": inference_config.get("backend", "mock"),
         "model": backend.model_name},
    )
    _emit(event_queue, start)
    control_queue.put(start)

    first_token_emitted = False
    last_update = None
    for update in backend.stream(str(config.get("prompt", ""))):
      last_update = update
      if not first_token_emitted:
        _emit(
            event_queue,
            Event.now(
                trial_id,
                EventKind.FIRST_TOKEN,
                ProcessRole.INFERENCE,
                {"token_count": update.token_count,
                 "partial_text": update.partial_text},
            ),
        )
        first_token_emitted = True
      if update.token_count % progress_every == 0:
        _emit(
            event_queue,
            Event.now(
                trial_id,
                EventKind.DECODE_PROGRESS,
                ProcessRole.INFERENCE,
                {"token_count": update.token_count,
                 "partial_text": update.partial_text},
            ),
        )

    end = Event.now(
        trial_id,
        EventKind.INFERENCE_END,
        ProcessRole.INFERENCE,
        {
            "token_count": last_update.token_count if last_update else 0,
            "text": last_update.partial_text if last_update else "",
            "inference_ms": (time.monotonic_ns() - start.monotonic_ns) / 1e6,
        },
    )
    _emit(event_queue, end)
    abort = Event.now(
        trial_id,
        EventKind.ABORT,
        ProcessRole.INFERENCE,
        {"inference_end_monotonic_ns": end.monotonic_ns},
    )
    control_queue.put(abort)
    _emit(event_queue, abort)

    restore_timeout_s = float(config.get("restore_timeout_s", 10.0))
    terminal = require_event(
        status_queue.get(timeout=restore_timeout_s),
        EventKind.RESTORED,
        EventKind.RESTORE_FAILED,
    )
    received_ns = time.monotonic_ns()
    extension_ms = (received_ns - abort.monotonic_ns) / 1e6
    complete = Event.now(
        trial_id,
        EventKind.TRIAL_COMPLETE,
        ProcessRole.INFERENCE,
        {
            "critical_path_extension_ms": extension_ms,
            "restore_status": terminal.kind.value,
            "restore_worker_elapsed_ms": terminal.payload.get(
                "restore_elapsed_ms"
            ),
            "abort_during_work": terminal.payload.get("abort_during_work"),
            "inference_ms": end.payload["inference_ms"],
        },
    )
    _emit(event_queue, complete)
  except Exception as exc:  # Error must be observable across process boundary.
    _emit(
        event_queue,
        Event.now(
            trial_id,
            EventKind.ERROR,
            ProcessRole.INFERENCE,
            {"code": "INFERENCE_PROCESS_ERROR", "message": str(exc),
             "traceback": traceback.format_exc()},
        ),
    )


def _sleep_abortibly(duration_ms: float, poll_interval_ms: float) -> None:
  deadline = time.monotonic() + max(0.0, duration_ms) / 1000.0
  while True:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
      return
    time.sleep(min(remaining, poll_interval_ms / 1000.0))


def _explorer_process(
    config: Mapping[str, Any], control_queue: Any, status_queue: Any, event_queue: Any
) -> None:
  """Process B phase-1 shell: interruptible work plus restore acknowledgement.

  Real adb probing is intentionally absent until later build stages. This shell
  exists to validate lifecycle ordering and critical-path measurement first.
  """
  trial_id = str(config["trial_id"])
  explorer_config = dict(config.get("explorer", {}))
  poll_interval_ms = float(explorer_config.get("poll_interval_ms", 50.0))
  if poll_interval_ms <= 0 or poll_interval_ms > 200:
    _emit(
        event_queue,
        Event.now(
            trial_id,
            EventKind.ERROR,
            ProcessRole.EXPLORER,
            {"code": "INVALID_ABORT_POLL_INTERVAL",
             "message": "poll_interval_ms must be in (0, 200]"},
        ),
    )
    return
  ready = Event.now(trial_id, EventKind.EXPLORER_READY, ProcessRole.EXPLORER)
  status_queue.put(ready)
  _emit(event_queue, ready)
  inference_started = False
  work_started_ns = 0
  try:
    while True:
      try:
        message = require_event(
            control_queue.get(timeout=poll_interval_ms / 1000.0)
        )
      except queue.Empty:
        if inference_started and not work_started_ns:
          work_started_ns = time.monotonic_ns()
        continue
      if message.kind == EventKind.INFERENCE_START:
        inference_started = True
        work_started_ns = time.monotonic_ns()
        continue
      if message.kind != EventKind.ABORT:
        continue

      abort_received_ns = time.monotonic_ns()
      restore_ms = float(explorer_config.get("mock_restore_ms", 100.0))
      _sleep_abortibly(restore_ms, poll_interval_ms)
      failed = bool(explorer_config.get("mock_restore_failure", False))
      terminal = Event.now(
          trial_id,
          EventKind.RESTORE_FAILED if failed else EventKind.RESTORED,
          ProcessRole.EXPLORER,
          {
              "restore_elapsed_ms":
                  (time.monotonic_ns() - abort_received_ns) / 1e6,
              "abort_during_work": bool(work_started_ns),
              "implementation": "phase1_mock_explorer",
          },
      )
      status_queue.put(terminal)
      _emit(event_queue, terminal)
      return
  except Exception as exc:
    terminal = Event.now(
        trial_id,
        EventKind.RESTORE_FAILED,
        ProcessRole.EXPLORER,
        {"restore_elapsed_ms": 0.0, "abort_during_work": bool(work_started_ns),
         "code": "EXPLORER_PROCESS_ERROR", "message": str(exc)},
    )
    status_queue.put(terminal)
    _emit(event_queue, terminal)


@dataclasses.dataclass(frozen=True)
class TrialRun:
  events: tuple[Event, ...]
  result: Mapping[str, Any]


def run_trial(config: Mapping[str, Any]) -> TrialRun:
  """Run one trial and return the complete event trace."""
  context = multiprocessing.get_context("spawn")
  control_queue = context.Queue()
  status_queue = context.Queue()
  event_queue = context.Queue()
  explorer = context.Process(
      target=_explorer_process,
      name="ui-explorer",
      args=(dict(config), control_queue, status_queue, event_queue),
  )
  inference = context.Process(
      target=_inference_process,
      name="vlm-inference",
      args=(dict(config), control_queue, status_queue, event_queue),
  )
  explorer.start()
  inference.start()
  events: list[Event] = []
  deadline = time.monotonic() + float(config.get("trial_timeout_s", 30.0))
  result: Mapping[str, Any] | None = None
  while time.monotonic() < deadline:
    try:
      event = require_event(event_queue.get(timeout=0.1))
    except queue.Empty:
      if not inference.is_alive() and inference.exitcode is not None:
        break
      continue
    events.append(event)
    if event.kind == EventKind.ERROR:
      break
    if event.kind == EventKind.TRIAL_COMPLETE:
      result = event.payload
      break

  inference.join(timeout=1.0)
  explorer.join(timeout=1.0)
  if inference.is_alive():
    inference.terminate()
    inference.join()
  if explorer.is_alive():
    explorer.terminate()
    explorer.join()
  if result is None:
    errors = [event.payload for event in events if event.kind == EventKind.ERROR]
    raise RuntimeError(f"Trial did not complete; errors={errors}")
  return TrialRun(tuple(events), result)
