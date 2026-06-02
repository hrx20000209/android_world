#!/usr/bin/env python3
"""Build the fast accessibility provider APK without requiring Bash."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "build"
PKG = "com.androidworld.fasta11y"


def _run(cmd: list[str | Path], cwd: Path | None = None) -> None:
    printable = " ".join(str(part) for part in cmd)
    print("[build_apk]", printable)
    subprocess.run([str(part) for part in cmd], cwd=str(cwd or ROOT), check=True)


def _sdk_root() -> Path:
    candidates = [
        os.environ.get("ANDROID_HOME"),
        os.environ.get("ANDROID_SDK_ROOT"),
    ]
    adb = shutil.which("adb")
    if adb:
        platform_tools = Path(adb).resolve().parent
        if platform_tools.name.lower() == "platform-tools":
            candidates.append(str(platform_tools.parent))
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(str(Path(local_app_data) / "Android" / "Sdk"))
        candidates.extend(
            [
                r"D:\Android SDK",
                r"C:\Android SDK",
            ]
        )
    else:
        candidates.append(str(Path.home() / "Library" / "Android" / "sdk"))
        candidates.append(str(Path.home() / "Android" / "Sdk"))
    for candidate in candidates:
        if candidate and (Path(candidate) / "platforms").exists() and (Path(candidate) / "build-tools").exists():
            return Path(candidate)
    raise SystemExit(
        "Android SDK not found. Set ANDROID_HOME or ANDROID_SDK_ROOT, or put adb on PATH."
    )


def _version_key(path: Path) -> tuple[int, ...]:
    text = path.name.removeprefix("android-")
    pieces: list[int] = []
    for part in text.replace("-", ".").split("."):
        try:
            pieces.append(int(part))
        except ValueError:
            pieces.append(0)
    return tuple(pieces)


def _latest_dir(parent: Path, pattern: str) -> Path:
    dirs = [path for path in parent.glob(pattern) if path.is_dir()]
    if not dirs:
        raise SystemExit(f"No {pattern} directories under {parent}")
    return sorted(dirs, key=_version_key)[-1]


def _build_tool(build_tools: Path, name: str) -> Path:
    suffixes = [".exe", ".bat", ".cmd", ""] if os.name == "nt" else ["", ".sh"]
    for suffix in suffixes:
        path = build_tools / f"{name}{suffix}"
        if path.exists():
            return path
    raise SystemExit(f"Missing Android build tool: {name} in {build_tools}")


def _java_tool(name: str) -> str:
    java_home = os.environ.get("JAVA_HOME")
    candidates: list[Path] = []
    if java_home:
        candidates.append(Path(java_home) / "bin" / (name + (".exe" if os.name == "nt" else "")))
    if os.name == "nt":
        for env_name in ("ANDROID_STUDIO_JBR", "STUDIO_JDK"):
            env_value = os.environ.get(env_name)
            if env_value:
                candidates.append(Path(env_value) / "bin" / f"{name}.exe")
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.extend(
                [
                    Path(local_app_data) / "Programs" / "Android Studio" / "jbr" / "bin" / f"{name}.exe",
                    Path(local_app_data) / "Programs" / "Gateway" / "jbr" / "bin" / f"{name}.exe",
                ]
            )
        candidates.extend(
            [
                Path(r"D:\Android Studio\jbr\bin") / f"{name}.exe",
                Path(r"C:\Program Files\Android\Android Studio\jbr\bin") / f"{name}.exe",
                Path(r"C:\Program Files\Android Studio\jbr\bin") / f"{name}.exe",
            ]
        )
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    check_arg = "-help" if name == "keytool" else "-version"
    for candidate in candidates:
        if not candidate.exists():
            continue
        result = subprocess.run(
            [str(candidate), check_arg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
            check=False,
        )
        if result.returncode == 0:
            return str(candidate)
    raise SystemExit(f"Missing Java tool: {name}. Install a JDK and set JAVA_HOME.")


def _jdk_home_from_tool(tool: str) -> Path | None:
    path = Path(tool).resolve()
    if path.parent.name.lower() == "bin":
        return path.parent.parent
    return None


def _debug_keystore() -> Path | None:
    home = Path.home() / ".android" / "debug.keystore"
    if home.exists():
        return home
    return None


def _java_sources(*roots: Path) -> list[str]:
    sources: list[str] = []
    for root in roots:
        sources.extend(str(path) for path in sorted(root.rglob("*.java")))
    if not sources:
        raise SystemExit("No Java sources found to compile.")
    return sources


def main() -> int:
    sdk = _sdk_root()
    build_tools = _latest_dir(sdk / "build-tools", "*")
    platform = _latest_dir(sdk / "platforms", "android-*")
    android_jar = platform / "android.jar"
    if not android_jar.exists():
        raise SystemExit(f"Missing android.jar: {android_jar}")

    aapt2 = _build_tool(build_tools, "aapt2")
    d8 = _build_tool(build_tools, "d8")
    apksigner = _build_tool(build_tools, "apksigner")
    javac = _java_tool("javac")
    jdk_home = _jdk_home_from_tool(javac)
    if jdk_home:
        os.environ["JAVA_HOME"] = str(jdk_home)

    if OUT.exists():
        shutil.rmtree(OUT)
    for rel in ("compiled", "gen", "classes", "dex"):
        (OUT / rel).mkdir(parents=True, exist_ok=True)

    unsigned_apk = OUT / "fast-a11y-unsigned.apk"
    keystore = _debug_keystore() or (OUT / "debug.keystore")
    signed_apk = OUT / "fast-a11y.apk"

    _run([aapt2, "compile", "--dir", ROOT / "res", "-o", OUT / "compiled" / "resources.zip"])
    _run(
        [
            aapt2,
            "link",
            "-o",
            unsigned_apk,
            "-I",
            android_jar,
            "--manifest",
            ROOT / "AndroidManifest.xml",
            "--java",
            OUT / "gen",
            OUT / "compiled" / "resources.zip",
        ]
    )
    _run(
        [
            javac,
            "-source",
            "8",
            "-target",
            "8",
            "-bootclasspath",
            android_jar,
            "-classpath",
            OUT / "gen",
            "-d",
            OUT / "classes",
            *_java_sources(ROOT / "src", OUT / "gen"),
        ]
    )
    _run([d8, "--min-api", "28", "--output", OUT / "dex", *sorted(OUT.glob("classes/**/*.class"))])
    with zipfile.ZipFile(unsigned_apk, "a") as apk:
        apk.write(OUT / "dex" / "classes.dex", "classes.dex")
    if not keystore.exists():
        keytool = _java_tool("keytool")
        _run(
            [
                keytool,
                "-genkeypair",
                "-keystore",
                keystore,
                "-storepass",
                "android",
                "-keypass",
                "android",
                "-alias",
                "androiddebugkey",
                "-keyalg",
                "RSA",
                "-keysize",
                "2048",
                "-validity",
                "10000",
                "-dname",
                "CN=Android Debug,O=Android,C=US",
            ]
        )
    _run(
        [
            apksigner,
            "sign",
            "--ks",
            keystore,
            "--ks-pass",
            "pass:android",
            "--key-pass",
            "pass:android",
            "--out",
            signed_apk,
            unsigned_apk,
        ]
    )
    print(signed_apk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
