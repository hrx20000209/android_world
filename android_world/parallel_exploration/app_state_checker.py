"""Independent, read-only persistent-state check for exploration witnesses.

A probe whose UI came back is not a probe without side effects: Back does not
unsend a message or undelete a row. This module snapshots the business state
an app keeps on disk (SQLite rows, document files, preference files) before
and after a window of probes, and certifies the window only when every covered
item is byte-identical afterwards.

What it is and is not:
  * Logical, not file-byte: each table is dumped through ``sqlite3 -readonly``
    in one transaction per database, so WAL frames are included and a page
    reshuffle that leaves rows unchanged is not a change.
  * Coverage is explicit. An app without a spec, a missing database, a
    non-root adb or any dump error yields ``covered=False``; the caller must
    then NOT issue ``state_effect_verified``. Unknown coverage is never read
    as "unchanged".
  * Window-level. Probes in one window are certified together, so a mutation
    undone by a later probe in the same window is invisible. Per-probe
    brackets cost one snapshot per probe and are left to a later pass.
  * Network, cloud sync and other apps' state are not observed. Specs list
    exactly what is observed; anything else is outside the certificate.
  * The policy only reads the verdict. Raw digests are logged for audit and
    never contain the task's answer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import shlex
import subprocess
import time
from collections.abc import Mapping, Sequence
from typing import Any


@dataclasses.dataclass(frozen=True)
class AppStateSpec:
  # (absolute db path, tables); an empty tables tuple means every user table.
  databases: tuple[tuple[str, tuple[str, ...]], ...] = ()
  # Directories whose regular files are hashed recursively.
  file_roots: tuple[str, ...] = ()
  # Preference XML files that only record view history (recent documents,
  # WebView caches). Listed explicitly so the exclusion is auditable.
  prefs_excluded: tuple[str, ...] = ()
  include_prefs: bool = True


_D = "/data/data"
_CONTACTS_DB = f"{_D}/com.android.providers.contacts/databases/contacts2.db"
_SMS_DB = f"{_D}/com.android.providers.telephony/databases/mmssms.db"

SPECS: dict[str, AppStateSpec] = {
    "com.simplemobiletools.calendar.pro": AppStateSpec(
        databases=((f"{_D}/com.simplemobiletools.calendar.pro/databases/events.db",
                    ("events", "event_types", "tasks")),)),
    "com.arduia.expense": AppStateSpec(
        databases=((f"{_D}/com.arduia.expense/databases/accounting.db", ()),)),
    "com.flauschcode.broccoli": AppStateSpec(
        databases=((f"{_D}/com.flauschcode.broccoli/databases/broccoli", ()),)),
    "com.simplemobiletools.contacts.pro": AppStateSpec(
        databases=((f"{_D}/com.simplemobiletools.contacts.pro/databases/local_contacts.db", ()),
                   (_CONTACTS_DB, ("raw_contacts", "data", "groups")))),
    "org.tasks": AppStateSpec(
        databases=((f"{_D}/org.tasks/databases/database", ()),)),
    "net.gsantner.markor": AppStateSpec(
        file_roots=("/storage/emulated/0/Documents",),
        prefs_excluded=("DOCUMENT_MOD_TIMES.xml", "cache.xml", "WebViewChromiumPrefs.xml")),
    "com.simplemobiletools.smsmessenger": AppStateSpec(
        databases=((_SMS_DB, ("sms", "threads")),)),
}


@dataclasses.dataclass(frozen=True)
class StateSnapshot:
  package: str
  covered: bool
  digests: Mapping[str, str]
  cost_s: float
  error: str = ""
  taken_at: float = 0.0


@dataclasses.dataclass(frozen=True)
class StateVerdict:
  verified: bool
  covered: bool
  reason: str
  changed: tuple[str, ...]
  coverage: tuple[str, ...]
  before_digest: str
  after_digest: str
  checker_cost_s: float

  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)


def _combined(digests: Mapping[str, str]) -> str:
  return hashlib.sha256("\n".join(f"{k}={v}" for k, v in sorted(digests.items())).encode()).hexdigest()[:24]


def _sha(text: str) -> str:
  return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()[:24]


class AppStateChecker:
  """Snapshots through ``adb shell`` as root; never writes to the device."""

  def __init__(self, serial: str = "emulator-5554", adb_path: str = "adb",
               specs: Mapping[str, AppStateSpec] | None = None, timeout_s: float = 20.0):
    self.serial = serial
    self.adb_path = adb_path
    self.specs = dict(SPECS if specs is None else specs)
    self.timeout_s = timeout_s
    self._root: bool | None = None

  def _shell(self, command: str) -> str:
    completed = subprocess.run(
        [self.adb_path, "-s", self.serial, "shell", command],
        capture_output=True, timeout=self.timeout_s, check=False)
    # Rows may hold BLOBs; digest the exact bytes instead of failing on them.
    stdout = completed.stdout.decode("utf-8", "surrogateescape")
    if completed.returncode != 0:
      stderr = completed.stderr.decode("utf-8", "replace")
      raise RuntimeError((stderr or stdout).strip()[:200] or "adb shell failed")
    return stdout

  def available(self) -> bool:
    if self._root is None:
      try:
        self._root = self._shell("id -u").strip() == "0"
      except Exception:  # pylint: disable=broad-exception-caught
        self._root = False
    return self._root

  def covers(self, package: str) -> bool:
    return package in self.specs and self.available()

  def _db_digests(self, path: str, tables: Sequence[str]) -> dict[str, str]:
    q = shlex.quote
    if not tables:
      listed = self._shell(
          f"sqlite3 -readonly {q(path)} "
          + q("select name from sqlite_master where type='table' "
              "and name not like 'sqlite_%' and name not like 'android_%' order by name"))
      tables = tuple(t for t in listed.split() if t)
    if not tables:
      raise RuntimeError(f"no tables in {path}")
    marker = "@@TABLE@@"
    # One sqlite3 process and one read transaction: every table is read at
    # the same commit point, including frames still in the WAL.
    script = "BEGIN;" + "".join(
        f"SELECT '{marker}{t}';SELECT * FROM \"{t}\" ORDER BY 1;" for t in tables) + "COMMIT;"
    out = self._shell(f"sqlite3 -readonly {q(path)} {q(script)}")
    chunks: dict[str, list[str]] = {}
    current = None
    for line in out.splitlines():
      if line.startswith(marker):
        current = line[len(marker):]
        chunks[current] = []
      elif current is not None:
        chunks[current].append(line)
    missing = [t for t in tables if t not in chunks]
    if missing:
      raise RuntimeError(f"unreadable tables {missing} in {path}")
    return {f"db:{path.rsplit('/', 1)[-1]}:{t}": _sha("\n".join(rows)) for t, rows in chunks.items()}

  def _file_digests(self, root: str) -> dict[str, str]:
    q = shlex.quote
    # Directories are listed as entries too: creating an empty folder is a
    # persistent change that a regular-file listing cannot see (caught live on
    # MarkorCreateFolder, where the FOLDER click was certified invariant).
    tree = self._shell(f"find {q(root)} 2>/dev/null | sort")
    sums = self._shell(f"find {q(root)} -type f 2>/dev/null | sort | while read f; do md5sum \"$f\"; done")
    return {f"tree:{root}": _sha(tree), f"files:{root}": _sha(sums)}

  def _prefs_digests(self, package: str, excluded: Sequence[str]) -> dict[str, str]:
    directory = f"{_D}/{package}/shared_prefs"
    out = self._shell(f"cd {shlex.quote(directory)} 2>/dev/null && md5sum *.xml 2>/dev/null; true")
    rows = []
    for line in out.splitlines():
      parts = line.split()
      if len(parts) == 2 and parts[1] not in excluded:
        rows.append(line)
    return {f"prefs:{package}": _sha("\n".join(sorted(rows)))}

  def snapshot(self, package: str) -> StateSnapshot:
    started = time.perf_counter()
    spec = self.specs.get(package)
    if spec is None:
      return StateSnapshot(package, False, {}, 0.0, "no_spec_for_package", time.time())
    if not self.available():
      return StateSnapshot(package, False, {}, time.perf_counter() - started, "adb_not_root", time.time())
    digests: dict[str, str] = {}
    try:
      for path, tables in spec.databases:
        digests.update(self._db_digests(path, tables))
      for root in spec.file_roots:
        digests.update(self._file_digests(root))
      if spec.include_prefs:
        digests.update(self._prefs_digests(package, spec.prefs_excluded))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      return StateSnapshot(package, False, {}, time.perf_counter() - started,
                           f"snapshot_failed:{exc}"[:200], time.time())
    return StateSnapshot(package, True, digests, time.perf_counter() - started, "", time.time())


def compare(before: StateSnapshot | None, after: StateSnapshot | None) -> StateVerdict:
  """Certify only identical, fully covered snapshots of the same package."""
  cost = sum(s.cost_s for s in (before, after) if s is not None)
  if before is None or after is None:
    return StateVerdict(False, False, "snapshot_missing", (), (), "", "", cost)
  coverage = tuple(sorted(set(before.digests) | set(after.digests)))
  b, a = _combined(before.digests), _combined(after.digests)
  if before.package != after.package:
    return StateVerdict(False, False, "package_changed", (), coverage, b, a, cost)
  if not before.covered or not after.covered:
    return StateVerdict(False, False, before.error or after.error or "not_covered",
                        (), coverage, b, a, cost)
  changed = tuple(k for k in coverage if before.digests.get(k) != after.digests.get(k))
  if changed:
    return StateVerdict(False, True, "persistent_state_changed", changed, coverage, b, a, cost)
  return StateVerdict(True, True, "invariant", (), coverage, b, a, cost)
