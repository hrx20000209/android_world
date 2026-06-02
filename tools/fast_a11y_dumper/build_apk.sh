#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
BUILD_TOOLS="$SDK/build-tools/36.0.0"
ANDROID_JAR="$SDK/platforms/android-36/android.jar"
OUT="$ROOT/build"
PKG="com.androidworld.fasta11y"

rm -rf "$OUT"
mkdir -p "$OUT/compiled" "$OUT/gen" "$OUT/classes" "$OUT/dex"

"$BUILD_TOOLS/aapt2" compile --dir "$ROOT/res" -o "$OUT/compiled/resources.zip"
"$BUILD_TOOLS/aapt2" link \
  -o "$OUT/fast-a11y-unsigned.apk" \
  -I "$ANDROID_JAR" \
  --manifest "$ROOT/AndroidManifest.xml" \
  --java "$OUT/gen" \
  "$OUT/compiled/resources.zip"

javac -source 8 -target 8 \
  -bootclasspath "$ANDROID_JAR" \
  -classpath "$OUT/gen" \
  -d "$OUT/classes" \
  $(find "$ROOT/src" "$OUT/gen" -name '*.java' | sort)

"$BUILD_TOOLS/d8" --min-api 28 --output "$OUT/dex" $(find "$OUT/classes" -name '*.class' | sort)
(cd "$OUT/dex" && zip -q -u "$OUT/fast-a11y-unsigned.apk" classes.dex)

KEYSTORE="$OUT/debug.keystore"
keytool -genkeypair \
  -keystore "$KEYSTORE" \
  -storepass android \
  -keypass android \
  -alias androiddebugkey \
  -keyalg RSA \
  -keysize 2048 \
  -validity 10000 \
  -dname "CN=Android Debug,O=Android,C=US" >/dev/null

"$BUILD_TOOLS/apksigner" sign \
  --ks "$KEYSTORE" \
  --ks-pass pass:android \
  --key-pass pass:android \
  --out "$OUT/fast-a11y.apk" \
  "$OUT/fast-a11y-unsigned.apk"

echo "$OUT/fast-a11y.apk"
