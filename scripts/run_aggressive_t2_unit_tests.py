#!/usr/bin/env python3
"""Run aggressive t+2 shortcut unit tests and save JSON diagnostics."""

from __future__ import annotations

import argparse
import json
import time
import unittest
from pathlib import Path


class JsonResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[dict[str, object]] = []

    def addSuccess(self, test):  # noqa: N802
        super().addSuccess(test)
        self.events.append({"test": str(test), "status": "passed"})

    def addFailure(self, test, err):  # noqa: N802
        super().addFailure(test, err)
        self.events.append({"test": str(test), "status": "failed", "error": self._exc_info_to_string(err, test)})

    def addError(self, test, err):  # noqa: N802
        super().addError(test, err)
        self.events.append({"test": str(test), "status": "error", "error": self._exc_info_to_string(err, test)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()
    output_dir = Path(args.output_dir or f"results/aggressive_t2_shortcut/shortcut_integration_debug_{time.strftime('%Y%m%dT%H%M%S')}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = unittest.defaultTestLoader.discover("tests", pattern="test_aggressive_t2_shortcut_utils.py", top_level_dir=".")
    runner = unittest.TextTestRunner(verbosity=2, resultclass=JsonResult)
    result: JsonResult = runner.run(suite)  # type: ignore[assignment]
    payload = {
        "ok": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "events": result.events,
    }
    (output_dir / "unit_test_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[unit] wrote {output_dir / 'unit_test_results.json'}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
