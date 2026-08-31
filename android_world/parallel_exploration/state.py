"""Independent activity, accessibility-structure, and screenshot signatures."""

from __future__ import annotations

import abc
import dataclasses
import hashlib
import json
import re
import socket
import struct
import time
import xml.etree.ElementTree as et
from collections.abc import Sequence

import cv2
import numpy as np

from android_world.parallel_exploration.adb import AdbClient
from android_world.parallel_exploration.rankers import UiElement


_BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


@dataclasses.dataclass(frozen=True)
class ActivitySignature:
  component: str
  task_id: int | None
  stack_depth: int | None
  raw_top_line: str


@dataclasses.dataclass(frozen=True)
class StructSignature:
  digest: str
  element_count: int


@dataclasses.dataclass(frozen=True)
class StateSignature:
  activity: ActivitySignature
  struct_sig: StructSignature
  phash: str
  elements: tuple[UiElement, ...]
  timings_ms: dict[str, float]

  @property
  def layout_sig(self) -> str:
    """Tolerant screen identity for graph memory (NOT for rollback checks).

    struct_sig and phash intentionally capture volatile detail - every
    element's text and a pixel hash - which is what makes them good at
    answering "is this the exact state I left?" during rollback
    verification. They are far too strict to answer the different question
    the belief graph needs: "have I been on this kind of screen before?".
    A stopwatch ticking 00:03 -> 00:04, a list losing one row, or a few
    pixels of scroll all produce a brand-new struct_sig/phash, so on
    2026-08-29's 15-task batch a single Activity fragmented into 40
    distinct graph nodes (ExpenseAddMultiple) and progressive memory could
    never accumulate enough repeat visits to reuse anything.

    This signature keeps only the interaction skeleton - which actionable
    controls exist, and roughly where - dropping text content entirely and
    quantizing position coarsely, so the same screen showing different data
    resolves to the same node.
    """
    return layout_signature(self.elements)

  def phash_distance(self, other: "StateSignature") -> int:
    return (int(self.phash, 16) ^ int(other.phash, 16)).bit_count()

  def matches(
      self, other: "StateSignature", phash_max_distance: int = 12
  ) -> dict[str, bool]:
    return {
        "activity": self.activity.component == other.activity.component,
        "struct": self.struct_sig.digest == other.struct_sig.digest,
        "phash": self.phash_distance(other) <= phash_max_distance,
    }


class UiTreeProvider(abc.ABC):

  @abc.abstractmethod
  def dump(self) -> tuple[tuple[UiElement, ...], float, dict[str, float]]:
    raise NotImplementedError


class UiAutomatorDumpProvider(UiTreeProvider):

  def __init__(self, adb: AdbClient):
    self._adb = adb

  def dump(self) -> tuple[tuple[UiElement, ...], float, dict[str, float]]:
    started = time.monotonic_ns()
    self._adb.run(
        ["shell", "uiautomator", "dump", "/sdcard/parallel_explore.xml"],
        timeout_s=8.0,
    )
    xml, _ = self._adb.text(
        ["exec-out", "cat", "/sdcard/parallel_explore.xml"], timeout_s=4.0
    )
    elapsed_ms = (time.monotonic_ns() - started) / 1e6
    return parse_elements(xml), elapsed_ms, {}


class FastSocketUiTreeProvider(UiTreeProvider):
  """Persistent host socket to the on-device AccessibilityService."""

  def __init__(
      self,
      adb: AdbClient,
      local_port: int = 8765,
      compact: bool = False,
      max_nodes: int = 10000,
  ):
    self._adb = adb
    self._local_port = local_port
    self._compact = compact
    self._max_nodes = max_nodes
    self._socket: socket.socket | None = None

  def _connect(self) -> socket.socket:
    if self._socket is not None:
      return self._socket
    self._adb.run(
        ["forward", f"tcp:{self._local_port}",
         "localabstract:androidworld_fast_a11y"],
        timeout_s=3.0,
    )
    connection = socket.create_connection(
        ("127.0.0.1", self._local_port), timeout=3.0
    )
    connection.settimeout(3.0)
    self._socket = connection
    return connection

  @staticmethod
  def _read_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
      chunk = connection.recv(size - len(chunks))
      if not chunk:
        raise ConnectionError("Fast a11y socket closed before response completed")
      chunks.extend(chunk)
    return bytes(chunks)

  def dump(self) -> tuple[tuple[UiElement, ...], float, dict[str, float]]:
    started = time.monotonic_ns()
    last_error: Exception | None = None
    for _ in range(2):
      try:
        connection = self._connect()
        request = f"flat {1 if self._compact else 0} {self._max_nodes}\n"
        connection.sendall(request.encode("ascii"))
        size = struct.unpack(">I", self._read_exact(connection, 4))[0]
        if size > 32 * 1024 * 1024:
          raise ValueError(f"Fast a11y response is unreasonably large: {size}")
        data = json.loads(self._read_exact(connection, size).decode("utf-8"))
        if not data.get("ok"):
          raise RuntimeError(str(data.get("error", "unknown provider error")))
        elapsed_ms = (time.monotonic_ns() - started) / 1e6
        metrics = {
            "a11y_service": float(data.get("serviceMs") or 0.0),
            "a11y_capture": float(data.get("captureMs") or 0.0),
            "a11y_serialize": float(data.get("serializeMs") or 0.0),
        }
        return parse_fast_elements(data), elapsed_ms, metrics
      except Exception as exc:
        last_error = exc
        if self._socket is not None:
          self._socket.close()
          self._socket = None
    raise RuntimeError(f"fast_a11y_socket_unavailable: {last_error}")

  def close(self) -> None:
    if self._socket is not None:
      self._socket.close()
      self._socket = None


def _parse_bounds(value: str) -> tuple[int, int, int, int]:
  match = _BOUNDS_RE.fullmatch(value or "")
  return tuple(map(int, match.groups())) if match else (0, 0, 0, 0)


def parse_elements(xml: str) -> tuple[UiElement, ...]:
  root = et.fromstring(xml)
  elements = []
  for node in root.iter("node"):
    checked_raw = node.attrib.get("checked")
    selected_raw = node.attrib.get("selected")
    elements.append(
        UiElement(
            text=node.attrib.get("text", ""),
            content_desc=node.attrib.get("content-desc", ""),
            resource_id=node.attrib.get("resource-id", ""),
            class_name=node.attrib.get("class", ""),
            bounds=_parse_bounds(node.attrib.get("bounds", "")),
            clickable=node.attrib.get("clickable") == "true",
            scrollable=node.attrib.get("scrollable") == "true",
            checked=(checked_raw == "true") if checked_raw in ("true", "false") else None,
            selected=(selected_raw == "true") if selected_raw in ("true", "false") else None,
        )
    )
  return tuple(elements)


def parse_fast_elements(data: dict) -> tuple[UiElement, ...]:
  elements = []
  for node in data.get("nodes", []):
    bounds = node.get("bounds") or [0, 0, 0, 0]
    elements.append(
        UiElement(
            text=str(node.get("text") or ""),
            content_desc=str(node.get("contentDescription") or ""),
            resource_id=str(node.get("resourceId") or ""),
            class_name=str(node.get("class") or ""),
            bounds=tuple(int(value) for value in bounds),
            clickable=bool(node.get("clickable")),
            scrollable=bool(node.get("scrollable")),
            checked=bool(node.get("checked")) if node.get("checkable") else None,
            selected=bool(node.get("selected")),
            in_navigation_drawer=bool(node.get("inNavigationDrawer")),
        )
    )
  return tuple(elements)


def structural_signature(elements: Sequence[UiElement]) -> StructSignature:
  records = sorted(
      (
          element.class_name or "",
          element.resource_id or "",
          element.text or "",
          bool(element.selected),
          tuple((coordinate // 16) * 16 for coordinate in element.bounds),
      )
      for element in elements
  )
  encoded = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
  return StructSignature(hashlib.sha256(encoded.encode("utf-8")).hexdigest(), len(records))


def layout_signature(elements: Sequence[UiElement], grid: int = 64) -> str:
  """Hash the screen's interaction skeleton; see StateSignature.layout_sig.

  Only actionable controls count, identified by class + resource id +
  coarsely-gridded position. Text is excluded entirely (it is the data on
  the screen, not the identity of the screen) and the grid is deliberately
  coarse so minor reflow does not create a new identity.

  How many times a repeated control appears is also data, not identity. A
  control kind occurring more than once is a list row / grid cell, so it
  contributes its kind but not its position: "the expense list" is the same
  screen whether it holds three rows or four. Without this, a repetitive task
  produced a brand-new node on every iteration - ExpenseAddMultiple's single
  list Activity became five separate nodes (2026-08-30) - so the edge the
  authoritative model had already taken from that screen could never be
  recognised on the next iteration, and progressive memory had nothing to
  reuse. Singleton controls keep their position, which is what still tells
  the list screen apart from a detail or edit screen in the same Activity.
  """
  actionable = [
      element for element in elements
      if element.clickable or element.scrollable or element.checked is not None
  ]
  kind_counts: dict[tuple[str, str], int] = {}
  for element in actionable:
    key = (element.class_name or "", element.resource_id or "")
    kind_counts[key] = kind_counts.get(key, 0) + 1
  records = sorted({
      (
          element.class_name or "",
          element.resource_id or "",
          ()
          if kind_counts[(element.class_name or "", element.resource_id or "")] > 1
          else tuple((coordinate // grid) * grid for coordinate in element.bounds),
      )
      for element in actionable
  })
  encoded = json.dumps(records, ensure_ascii=False, separators=(",", ":"), default=list)
  return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def perceptual_hash(png: bytes) -> str:
  image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
  if image is None:
    raise ValueError("Screenshot is not a valid image")
  resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
  transformed = cv2.dct(np.float32(resized))[:8, :8]
  median = float(np.median(transformed.flatten()[1:]))
  bits = (transformed > median).flatten()
  value = sum(int(bit) << index for index, bit in enumerate(bits))
  return f"{value:016x}"


def parse_activity_dump(raw: str) -> ActivitySignature:
  lines = raw.splitlines()
  top_line = next(
      (line.strip() for line in lines
       if "mResumedActivity" in line or "topResumedActivity" in line),
      "",
  )
  component_match = re.search(r"\s([\w.]+/[\w.$]+)(?=\s|})", top_line)
  task_match = re.search(r"\bt(\d+)\b", top_line)
  component = component_match.group(1) if component_match else ""
  task_id = int(task_match.group(1)) if task_match else None
  stack_depth = None
  if task_id is not None:
    stack_depth = sum(
        1 for line in lines
        if "ActivityRecord{" in line and re.search(rf"\bt{task_id}\b", line)
    )
    stack_depth = stack_depth or None
  return ActivitySignature(component, task_id, stack_depth, top_line)


class ScreenshotProvider(abc.ABC):

  @abc.abstractmethod
  def capture(self) -> tuple[np.ndarray, float]:
    raise NotImplementedError


class AdbScreenshotProvider(ScreenshotProvider):

  def __init__(self, adb: AdbClient):
    self._adb = adb

  def capture(self) -> tuple[np.ndarray, float]:
    result = self._adb.run(["exec-out", "screencap", "-p"], timeout_s=5.0)
    image = cv2.imdecode(
        np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
    )
    if image is None:
      raise ValueError("Screenshot is not a valid PNG")
    return image, result.elapsed_ms


class AndroidWorldGrpcScreenshotProvider(ScreenshotProvider):
  """Official AndroidWorld emulator gRPC screenshot path."""

  def __init__(self, console_port: int = 5554, adb_path: str | None = None):
    from android_world.env import android_world_controller
    from android_world.env import interface

    controller = android_world_controller.get_controller(
        console_port=console_port,
        adb_path=adb_path or android_world_controller.DEFAULT_ADB_PATH,
        a11y_method=android_world_controller.A11yMethod.NONE,
    )
    self._env = interface.AsyncAndroidEnv(controller)

  def capture(self) -> tuple[np.ndarray, float]:
    started = time.monotonic_ns()
    state = self._env.get_state(wait_to_stabilize=False)
    elapsed_ms = (time.monotonic_ns() - started) / 1e6
    return cv2.cvtColor(state.pixels, cv2.COLOR_RGB2GRAY), elapsed_ms

  def close(self) -> None:
    self._env.close()


class StateCapture:

  def __init__(
      self,
      adb: AdbClient,
      tree_provider: UiTreeProvider | None = None,
      screenshot_provider: ScreenshotProvider | None = None,
  ):
    self._adb = adb
    self._tree_provider = tree_provider or UiAutomatorDumpProvider(adb)
    self._screenshot_provider = screenshot_provider or AdbScreenshotProvider(adb)

  def capture(self) -> StateSignature:
    activity_raw, activity_result = self._adb.text(
        ["shell", "dumpsys", "activity", "activities"], timeout_s=5.0
    )
    activity = parse_activity_dump(activity_raw)
    elements, uitree_ms, tree_metrics = self._tree_provider.dump()
    struct_started = time.monotonic_ns()
    struct_sig = structural_signature(elements)
    struct_ms = (time.monotonic_ns() - struct_started) / 1e6
    screenshot, screenshot_ms = self._screenshot_provider.capture()
    phash_started = time.monotonic_ns()
    resized = cv2.resize(screenshot, (32, 32), interpolation=cv2.INTER_AREA)
    transformed = cv2.dct(np.float32(resized))[:8, :8]
    median = float(np.median(transformed.flatten()[1:]))
    bits = (transformed > median).flatten()
    phash = f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"
    phash_ms = (time.monotonic_ns() - phash_started) / 1e6
    return StateSignature(
        activity=activity,
        struct_sig=struct_sig,
        phash=phash,
        elements=elements,
        timings_ms={
            "activity_dump": activity_result.elapsed_ms,
            "uitree_dump": uitree_ms,
            "struct_hash": struct_ms,
            "screenshot": screenshot_ms,
            "phash": phash_ms,
            **tree_metrics,
        },
    )

  def close(self) -> None:
    for provider in (self._tree_provider, self._screenshot_provider):
      close = getattr(provider, "close", None)
      if callable(close):
        close()


def create_optimized_state_capture(
    serial: str = "emulator-5554",
    console_port: int = 5554,
    adb_path: str = "/Users/huangrunxi/Library/Android/sdk/platform-tools/adb",
    a11y_local_port: int = 8765,
) -> StateCapture:
  """Build the low-latency emulator capture path used by the harness."""
  adb = AdbClient(serial)
  return StateCapture(
      adb=adb,
      tree_provider=FastSocketUiTreeProvider(adb, local_port=a11y_local_port),
      screenshot_provider=AndroidWorldGrpcScreenshotProvider(
          console_port=console_port, adb_path=adb_path
      ),
  )
