"""Bounded adb execution with retry-once and structured failures."""

from __future__ import annotations

import dataclasses
import subprocess
import time
from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class AdbResult:
  command: tuple[str, ...]
  stdout: bytes
  stderr: str
  elapsed_ms: float
  attempts: int


class AdbCallError(RuntimeError):

  def __init__(self, code: str, command: Sequence[str], attempts: int, detail: str):
    super().__init__(f"{code}: {' '.join(command)}: {detail}")
    self.code = code
    self.command = tuple(command)
    self.attempts = attempts
    self.detail = detail


class AdbClient:

  def __init__(self, serial: str | None = None, timeout_s: float = 5.0):
    self.serial = serial
    self.timeout_s = timeout_s

  def run(
      self,
      args: Sequence[str],
      *,
      timeout_s: float | None = None,
      retry_once: bool = True,
  ) -> AdbResult:
    command = ["adb"]
    if self.serial:
      command.extend(["-s", self.serial])
    command.extend(args)
    started = time.monotonic_ns()
    attempts = 2 if retry_once else 1
    last_detail = ""
    for attempt in range(1, attempts + 1):
      try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s or self.timeout_s,
        )
      except subprocess.TimeoutExpired:
        last_detail = f"timeout after {timeout_s or self.timeout_s:.3f}s"
        if attempt == attempts:
          raise AdbCallError("ADB_TIMEOUT", command, attempt, last_detail)
        continue
      stderr = completed.stderr.decode("utf-8", errors="replace")
      if completed.returncode == 0:
        return AdbResult(
            tuple(command), completed.stdout, stderr,
            (time.monotonic_ns() - started) / 1e6, attempt
        )
      combined = (stderr + completed.stdout.decode("utf-8", errors="replace"))
      lowered = combined.casefold()
      if "no devices" in lowered or "device not found" in lowered or "offline" in lowered:
        code = "DEVICE_DISCONNECTED"
      else:
        code = "ADB_NONZERO_EXIT"
      last_detail = f"exit={completed.returncode}: {combined.strip()}"
      if attempt == attempts:
        raise AdbCallError(code, command, attempt, last_detail)
    raise AdbCallError("ADB_UNKNOWN", command, attempts, last_detail)

  def text(self, args: Sequence[str], **kwargs) -> tuple[str, AdbResult]:
    result = self.run(args, **kwargs)
    return result.stdout.decode("utf-8", errors="replace"), result
