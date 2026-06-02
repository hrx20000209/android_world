#!/usr/bin/env python3
"""Benchmark Android accessibility tree acquisition paths on the current emulator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADB = "/Users/huangrunxi/Library/Android/sdk/platform-tools/adb"
FAST_AUTHORITY = "com.androidworld.fasta11y.provider"
FAST_SERVICE = "com.androidworld.fasta11y/com.androidworld.fasta11y.FastA11yService"


def run(cmd: list[str], timeout: float = 30.0, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def adb_cmd(adb: str, serial: str, args: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess[bytes]:
    return run([adb, "-s", serial, *args], timeout=timeout)


def ensure_fast_service(adb: str, serial: str) -> None:
    adb_cmd(adb, serial, ["shell", "settings", "put", "secure", "accessibility_enabled", "1"], timeout=10)
    current = adb_cmd(
        adb,
        serial,
        ["shell", "settings", "get", "secure", "enabled_accessibility_services"],
        timeout=10,
    )
    enabled = current.stdout.decode("utf-8", errors="replace").strip()
    services = [] if enabled in {"", "null"} else [part for part in enabled.split(":") if part]
    if FAST_SERVICE not in services:
        services.append(FAST_SERVICE)
    adb_cmd(
        adb,
        serial,
        ["shell", "settings", "put", "secure", "enabled_accessibility_services", ":".join(services)],
        timeout=10,
    )
    adb_cmd(adb, serial, ["shell", "input", "keyevent", "3"], timeout=10)
    time.sleep(0.5)


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[idx]


def parse_xml_count(payload: bytes) -> int:
    try:
        root = ET.fromstring(payload.decode("utf-8", errors="replace"))
    except ET.ParseError:
        return 0
    return sum(1 for _ in root.iter("node"))


def measure_uiautomator(adb: str, serial: str, out_dir: Path, iterations: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    remote = "/sdcard/window_dump.xml"
    for i in range(iterations):
        adb_cmd(adb, serial, ["shell", "rm", "-f", remote], timeout=10)
        start = now_ms()
        dump_start = now_ms()
        try:
            dump = adb_cmd(adb, serial, ["shell", "uiautomator", "dump", remote], timeout=30)
        except subprocess.TimeoutExpired as exc:
            end = now_ms()
            rows.append(
                {
                    "method": "adb_uiautomator",
                    "iteration": i,
                    "ok": False,
                    "full_ms": end - start,
                    "dump_ms": end - dump_start,
                    "transfer_ms": 0.0,
                    "copy_to_workspace_ms": 0.0,
                    "bytes": 0,
                    "node_count": 0,
                    "stderr": f"timeout: {(exc.stdout or b'').decode('utf-8', errors='replace')[-500:]}",
                }
            )
            continue
        dump_end = now_ms()
        cat_start = now_ms()
        try:
            cat = adb_cmd(adb, serial, ["exec-out", "cat", remote], timeout=30)
        except subprocess.TimeoutExpired as exc:
            end = now_ms()
            rows.append(
                {
                    "method": "adb_uiautomator",
                    "iteration": i,
                    "ok": False,
                    "full_ms": end - start,
                    "dump_ms": dump_end - dump_start,
                    "transfer_ms": end - cat_start,
                    "copy_to_workspace_ms": 0.0,
                    "bytes": 0,
                    "node_count": 0,
                    "stderr": f"cat timeout: {(exc.stdout or b'').decode('utf-8', errors='replace')[-500:]}",
                }
            )
            continue
        cat_end = now_ms()
        local_start = now_ms()
        local_path = out_dir / f"uiautomator_{i:02d}.xml"
        local_path.write_bytes(cat.stdout)
        local_end = now_ms()
        dump_err = dump.stderr.decode("utf-8", errors="replace")
        cat_err = cat.stderr.decode("utf-8", errors="replace")
        rows.append(
            {
                "method": "adb_uiautomator",
                "iteration": i,
                "ok": dump.returncode == 0
                and cat.returncode == 0
                and bool(cat.stdout.strip())
                and "ERROR:" not in dump_err,
                "full_ms": local_end - start,
                "dump_ms": dump_end - dump_start,
                "transfer_ms": cat_end - cat_start,
                "copy_to_workspace_ms": local_end - local_start,
                "bytes": len(cat.stdout),
                "node_count": parse_xml_count(cat.stdout),
                "stderr": (dump_err + cat_err)[-500:],
            }
        )
    return rows


def measure_fast_provider(
    adb: str,
    serial: str,
    out_dir: Path,
    iterations: int,
    path: str,
    compact: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query = "?compact=1" if compact else ""
    uri = f"content://{FAST_AUTHORITY}/{path}{query}"
    label = f"fast_provider_{path}" + ("_compact" if compact else "")
    for i in range(iterations):
        start = now_ms()
        proc = adb_cmd(adb, serial, ["shell", "content", "read", "--uri", uri], timeout=30)
        read_end = now_ms()
        local_start = now_ms()
        local_path = out_dir / f"{label}_{i:02d}.json"
        local_path.write_bytes(proc.stdout)
        local_end = now_ms()
        data: dict[str, Any] = {}
        try:
            data = json.loads(proc.stdout.decode("utf-8"))
        except Exception as exc:
            data = {"ok": False, "error": repr(exc)}
        rows.append(
            {
                "method": label,
                "iteration": i,
                "ok": proc.returncode == 0 and bool(data.get("ok")),
                "full_ms": local_end - start,
                "adb_read_ms": read_end - start,
                "copy_to_workspace_ms": local_end - local_start,
                "service_ms": float(data.get("serviceMs") or 0.0),
                "capture_ms": float(data.get("captureMs") or 0.0),
                "serialize_ms": float(data.get("serializeMs") or 0.0),
                "bytes": len(proc.stdout),
                "node_count": int(data.get("nodeCount") or 0),
                "emitted_count": int(data.get("emittedCount") or 0),
                "truncated": bool(data.get("truncated")),
                "stderr": proc.stderr.decode("utf-8", errors="replace")[-500:],
                "error": data.get("error"),
            }
        )
    return rows


def measure_androidworld(adb: str, console_port: int, out_dir: Path, iterations: int, method: str) -> list[dict[str, Any]]:
    env = os.environ.copy()
    if method == "uiautomator":
        env["ANDROID_WORLD_A11Y_METHOD"] = "uiautomator"
    else:
        env.pop("ANDROID_WORLD_A11Y_METHOD", None)

    snippet = f"""
import json, time
from android_world.env import env_launcher
env = env_launcher.load_and_setup_env(console_port={console_port}, emulator_setup=False, adb_path={adb!r})
rows = []
for i in range({iterations}):
    start = time.perf_counter()
    state = env.get_state(wait_to_stabilize=False)
    end = time.perf_counter()
    rows.append({{
        "method": "androidworld_{method}",
        "iteration": i,
        "ok": True,
        "full_ms": (end - start) * 1000.0,
        "node_count": len(state.ui_elements),
        "pixel_shape": list(getattr(state.pixels, "shape", [])),
    }})
print(json.dumps(rows))
"""
    start = now_ms()
    proc = run([sys.executable, "-c", snippet], timeout=max(60.0, iterations * 20.0), env=env)
    elapsed = now_ms() - start
    if proc.returncode != 0:
        return [
            {
                "method": f"androidworld_{method}",
                "iteration": 0,
                "ok": False,
                "full_ms": elapsed,
                "stderr": (proc.stdout + proc.stderr).decode("utf-8", errors="replace")[-4000:],
            }
        ]
    try:
        text = proc.stdout.decode("utf-8", errors="replace").strip().splitlines()[-1]
        rows = json.loads(text)
    except Exception as exc:
        return [
            {
                "method": f"androidworld_{method}",
                "iteration": 0,
                "ok": False,
                "full_ms": elapsed,
                "stderr": repr(exc) + "\n" + proc.stdout.decode("utf-8", errors="replace")[-4000:],
            }
        ]
    (out_dir / f"androidworld_{method}.stdout.txt").write_bytes(proc.stdout)
    (out_dir / f"androidworld_{method}.stderr.txt").write_bytes(proc.stderr)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        group = [row for row in rows if row["method"] == method and row.get("ok")]
        all_group = [row for row in rows if row["method"] == method]
        timed_group = [row for row in all_group if row.get("full_ms") is not None]
        if not group:
            full = [float(row.get("full_ms") or 0.0) for row in timed_group]
            summaries.append(
                {
                    "method": method,
                    "ok": 0,
                    "n": len(all_group),
                    "all_failed": True,
                    "mean_full_ms": mean(full),
                    "stdev_full_ms": stdev(full),
                    "p50_full_ms": percentile(full, 0.5),
                    "p90_full_ms": percentile(full, 0.9),
                    "mean_service_ms": 0.0,
                    "mean_transfer_ms": mean(
                        [float(row.get("transfer_ms") or row.get("adb_read_ms") or 0.0) for row in timed_group]
                    ),
                    "mean_copy_ms": mean([float(row.get("copy_to_workspace_ms") or 0.0) for row in timed_group]),
                    "mean_bytes": mean([float(row.get("bytes") or 0.0) for row in timed_group]),
                    "mean_nodes": mean([float(row.get("node_count") or 0.0) for row in timed_group]),
                    "mean_emitted": mean(
                        [float(row.get("emitted_count") or row.get("node_count") or 0.0) for row in timed_group]
                    ),
                }
            )
            continue
        full = [float(row.get("full_ms") or 0.0) for row in group]
        summaries.append(
            {
                "method": method,
                "ok": len(group),
                "n": len(all_group),
                "mean_full_ms": mean(full),
                "stdev_full_ms": stdev(full),
                "p50_full_ms": percentile(full, 0.5),
                "p90_full_ms": percentile(full, 0.9),
                "mean_service_ms": mean([float(row.get("service_ms") or 0.0) for row in group]),
                "mean_transfer_ms": mean([float(row.get("transfer_ms") or row.get("adb_read_ms") or 0.0) for row in group]),
                "mean_copy_ms": mean([float(row.get("copy_to_workspace_ms") or 0.0) for row in group]),
                "mean_bytes": mean([float(row.get("bytes") or 0.0) for row in group]),
                "mean_nodes": mean([float(row.get("node_count") or 0.0) for row in group]),
                "mean_emitted": mean([float(row.get("emitted_count") or row.get("node_count") or 0.0) for row in group]),
            }
        )
    return summaries


def write_chart(summaries: list[dict[str, Any]], out_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    usable = [row for row in summaries if row.get("ok")]
    if not usable:
        return None
    labels = [row["method"].replace("androidworld_", "aw_").replace("fast_provider_", "fast_") for row in usable]
    full = [row["mean_full_ms"] for row in usable]
    service = [row.get("mean_service_ms") or 0.0 for row in usable]
    transfer = [row.get("mean_transfer_ms") or 0.0 for row in usable]
    copy_ms = [row.get("mean_copy_ms") or 0.0 for row in usable]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.6), 5))
    ax.bar([v - 0.24 for v in x], full, width=0.24, label="end-to-end")
    ax.bar(x, service, width=0.24, label="device service")
    ax.bar([v + 0.24 for v in x], transfer, width=0.24, label="adb/read")
    ax.plot(x, copy_ms, color="#333333", marker="o", linewidth=1.5, label="workspace write")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("milliseconds")
    ax.set_title("A11y tree acquisition latency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    chart = out_dir / "a11y_latency_chart.png"
    fig.savefig(chart, dpi=180)
    plt.close(fig)
    return chart


def write_report(rows: list[dict[str, Any]], summaries: list[dict[str, Any]], out_dir: Path, chart: Path | None) -> Path:
    report = out_dir / "a11y_latency_report_zh.md"
    lines = [
        "# A11y Tree 获取延迟实验报告",
        "",
        f"- 时间：`{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- 样本：每种方法 `{max((row.get('iteration', 0) for row in rows), default=0) + 1}` 次左右，统计只纳入成功样本。",
        f"- 原始数据：`{out_dir / 'a11y_latency_rows.jsonl'}`",
    ]
    if chart is not None:
        lines.append(f"- 图表：`{chart}`")
    lines.extend(
        [
            "",
            "## 汇总",
            "",
            "| 方法 | 成功/总数 | 平均全流程 ms | p50 ms | p90 ms | service ms | adb/read ms | 写入 workspace ms | 平均 bytes | 平均 nodes | 平均 emitted |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        status = f"{row.get('ok', 0)}/{row.get('n', 0)}"
        if row.get("all_failed"):
            status += " failed"
        lines.append(
            f"| {row['method']} | " + status + " | {mean_full_ms:.1f} | {p50_full_ms:.1f} | {p90_full_ms:.1f} | "
            "{mean_service_ms:.1f} | {mean_transfer_ms:.1f} | {mean_copy_ms:.3f} | "
            "{mean_bytes:.0f} | {mean_nodes:.1f} | {mean_emitted:.1f} |".format(**row)
        )
    fastest = min((row for row in summaries if row.get("ok")), key=lambda r: r["mean_full_ms"], default=None)
    lines.extend(["", "## 结论", ""])
    if fastest:
        lines.append(
            f"- 当前最快的成功路径是 `{fastest['method']}`，平均全流程约 `{fastest['mean_full_ms']:.1f} ms`。"
        )
    lines.append(
        "- `service ms` 是设备端 provider/service 内部抓取和序列化时间；`adb/read ms` 是 host 发起 `adb shell content read` 或 dump/cat 的读取时间；`写入 workspace ms` 是把结果落到本地文件的时间。"
    )
    lines.append(
        "- 如果 `fast_provider_flat_compact` 明显更快且 bytes/emitted 更小，说明传输体积是主要瓶颈之一；如果它仍接近 full tree，则瓶颈主要在 adb shell/ContentProvider 启动和 AccessibilityNodeInfo 遍历。"
    )
    lines.append(
        "- App 方案可以常驻：只要无障碍服务保持 enabled，provider 每次 adb 调用都会同步读取当前窗口树；后台不需要前台 Activity。进程被系统杀掉后，无障碍服务通常会由系统重新绑定，但 adb 第一次读取可能返回 `accessibility_service_not_connected` 或出现冷启动抖动。"
    )
    failed = [row for row in rows if not row.get("ok")]
    if failed:
        lines.extend(["", "## 失败样本摘录", ""])
        for row in failed[:5]:
            lines.append(f"- `{row.get('method')}` iteration `{row.get('iteration')}`: `{row.get('error') or row.get('stderr', '')[:300]}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", default=DEFAULT_ADB)
    parser.add_argument("--serial", default="emulator-5554")
    parser.add_argument("--console_port", type=int, default=5554)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "output" / "a11y_latency_benchmark"))
    parser.add_argument("--skip_uiautomator", action="store_true")
    parser.add_argument("--skip_androidworld", action="store_true")
    parser.add_argument("--skip_fast_provider", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if not args.skip_uiautomator:
        rows.extend(measure_uiautomator(args.adb, args.serial, out_dir, args.iterations))
    if not args.skip_androidworld:
        rows.extend(measure_androidworld(args.adb, args.console_port, out_dir, args.iterations, "uiautomator"))
        rows.extend(measure_androidworld(args.adb, args.console_port, out_dir, max(3, args.iterations // 2), "grpc"))
    if not args.skip_fast_provider:
        ensure_fast_service(args.adb, args.serial)
        rows.extend(measure_fast_provider(args.adb, args.serial, out_dir, args.iterations, "tree", compact=False))
        rows.extend(measure_fast_provider(args.adb, args.serial, out_dir, args.iterations, "flat", compact=False))
        rows.extend(measure_fast_provider(args.adb, args.serial, out_dir, args.iterations, "flat", compact=True))

    rows_path = out_dir / "a11y_latency_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summaries = summarize(rows)
    (out_dir / "a11y_latency_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    chart = write_chart(summaries, out_dir)
    report = write_report(rows, summaries, out_dir, chart)
    print(report)


if __name__ == "__main__":
    main()
