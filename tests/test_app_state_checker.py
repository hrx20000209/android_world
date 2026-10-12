"""The checker certifies only fully covered, identical snapshots."""

from android_world.parallel_exploration import app_state_checker as asc


class FakeChecker(asc.AppStateChecker):

  def __init__(self, outputs, root=True, **kwargs):
    super().__init__(**kwargs)
    self.outputs = outputs
    self.root = root
    self.commands = []

  def _shell(self, command):
    self.commands.append(command)
    if command == "id -u":
      return "0\n" if self.root else "2000\n"
    for key, value in self.outputs.items():
      if key in command:
        if isinstance(value, Exception):
          raise value
        return value
    return ""


SPEC = {"app.x": asc.AppStateSpec(databases=(("/data/data/app.x/databases/a.db", ("t1", "t2")),),
                                  prefs_excluded=("cache.xml",))}


def dump(rows1, rows2):
  return f"@@TABLE@@t1\n{rows1}\n@@TABLE@@t2\n{rows2}\n"


def test_identical_rows_are_invariant_and_reads_are_read_only():
  checker = FakeChecker({"sqlite3": dump("1|a", "2|b"), "md5sum": "x  p.xml\n"}, specs=SPEC)
  verdict = asc.compare(checker.snapshot("app.x"), checker.snapshot("app.x"))
  assert verdict.verified and verdict.reason == "invariant"
  sql = [c for c in checker.commands if "sqlite3" in c]
  assert sql and all("-readonly" in c and "BEGIN;" in c for c in sql)
  assert not any(word in c.upper() for c in sql for word in ("INSERT", "UPDATE", "DELETE", "DROP"))


def test_changed_row_is_named_and_not_certified():
  before = FakeChecker({"sqlite3": dump("1|a", "2|b")}, specs=SPEC).snapshot("app.x")
  after = FakeChecker({"sqlite3": dump("1|a", "2|B")}, specs=SPEC).snapshot("app.x")
  verdict = asc.compare(before, after)
  assert not verdict.verified and verdict.covered
  assert verdict.changed == ("db:a.db:t2",)


def test_view_history_prefs_are_excluded_but_real_prefs_are_not():
  a = FakeChecker({"sqlite3": dump("1", "2"), "md5sum": "1  cache.xml\n5  p.xml\n"}, specs=SPEC)
  b = FakeChecker({"sqlite3": dump("1", "2"), "md5sum": "2  cache.xml\n5  p.xml\n"}, specs=SPEC)
  c = FakeChecker({"sqlite3": dump("1", "2"), "md5sum": "1  cache.xml\n6  p.xml\n"}, specs=SPEC)
  assert asc.compare(a.snapshot("app.x"), b.snapshot("app.x")).verified
  assert not asc.compare(a.snapshot("app.x"), c.snapshot("app.x")).verified


def test_unknown_coverage_is_never_unchanged():
  ok = FakeChecker({"sqlite3": dump("1", "2")}, specs=SPEC)
  assert asc.compare(ok.snapshot("other.app"), ok.snapshot("other.app")).reason == "no_spec_for_package"
  noroot = FakeChecker({"sqlite3": dump("1", "2")}, root=False, specs=SPEC)
  assert asc.compare(noroot.snapshot("app.x"), noroot.snapshot("app.x")).reason == "adb_not_root"
  broken = FakeChecker({"sqlite3": RuntimeError("database is locked")}, specs=SPEC)
  verdict = asc.compare(broken.snapshot("app.x"), broken.snapshot("app.x"))
  assert not verdict.verified and not verdict.covered
  missing_table = FakeChecker({"sqlite3": "@@TABLE@@t1\n1\n"}, specs=SPEC)
  assert not asc.compare(missing_table.snapshot("app.x"), missing_table.snapshot("app.x")).covered


def test_package_change_is_not_certified():
  checker = FakeChecker({"sqlite3": dump("1", "2")}, specs={**SPEC, "app.y": SPEC["app.x"]})
  assert asc.compare(checker.snapshot("app.x"), checker.snapshot("app.y")).reason == "package_changed"


def test_new_empty_directory_is_a_change():
  spec = {"app.m": asc.AppStateSpec(file_roots=("/sdcard/Documents",), include_prefs=False)}
  before = FakeChecker({"-type f": "", "find /sdcard": "/sdcard/Documents\n"}, specs=spec)
  after = FakeChecker({"-type f": "", "find /sdcard": "/sdcard/Documents\n/sdcard/Documents/new\n"},
                      specs=spec)
  verdict = asc.compare(before.snapshot("app.m"), after.snapshot("app.m"))
  assert not verdict.verified and verdict.changed == ("tree:/sdcard/Documents",)
