#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"

# Keep one SDK/JDK discovery implementation for Linux servers and developer
# machines. build_apk.py selects the newest installed platform/build-tools
# versions instead of assuming a particular Android SDK release.
exec "$PYTHON" "$ROOT/build_apk.py" "$@"
