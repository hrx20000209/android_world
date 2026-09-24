#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
if [[ -z "$NDK" && -n "${ANDROID_HOME:-}" ]]; then
  shopt -s nullglob
  candidates=("${ANDROID_HOME}"/ndk/*)
  if ((${#candidates[@]})); then
    NDK="${candidates[${#candidates[@]}-1]}"
  fi
fi
if [[ -z "$NDK" && -d "$HOME/Library/Android/sdk/ndk" ]]; then
  shopt -s nullglob
  candidates=("$HOME/Library/Android/sdk/ndk"/*)
  if ((${#candidates[@]})); then
    NDK="${candidates[${#candidates[@]}-1]}"
  fi
fi
if [[ -z "$NDK" || ! -d "$NDK" ]]; then
  echo "Set ANDROID_NDK_HOME to an Android NDK installation." >&2
  exit 1
fi

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64|Darwin-x86_64) HOST_TAG="darwin-x86_64" ;;
  Linux-x86_64) HOST_TAG="linux-x86_64" ;;
  Linux-aarch64) HOST_TAG="linux-aarch64" ;;
  *) echo "Unsupported NDK host: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

CLANG_DIR="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin"
CLANG="$CLANG_DIR/aarch64-linux-android21-clang++"
if [[ ! -x "$CLANG" ]]; then
  echo "Missing Android arm64 clang: $CLANG" >&2
  exit 1
fi

"$CLANG" \
  -std=c++17 -O2 -DNDEBUG -static-libstdc++ \
  "$SCRIPT_DIR/explorer_selector.cpp" \
  -o "$SCRIPT_DIR/explorer_selector" \
  -Wl,--gc-sections -ffunction-sections -fdata-sections \
  -Wl,--strip-all

echo "built $SCRIPT_DIR/explorer_selector"
ls -lh "$SCRIPT_DIR/explorer_selector"
