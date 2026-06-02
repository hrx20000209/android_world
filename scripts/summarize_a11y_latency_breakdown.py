#!/usr/bin/env python3
"""Summarize AndroidWorld exploration latency traces with fast_provider details."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any


NUMERIC_KEYS = {
    "total_ms",
    "latency_ms",
    "a11y_latency_ms",
    "state_fetch_ms",
    "branch_state_fetch_ms",
    "branch_a11y_latency_ms",
    "a11y_dump_ms",
    "summary_ms",
    "a11y_trace_ms",
    "screenshot_ms",
    "fast_a11y_capture_ms",
    "fast_a11y_serialize_ms",
    "fast_a11y_service_ms",
    "fast_a11y_node_count",
    "fast_a11y_emitted_count",
    "fast_a11y_payload_bytes",
    "rollback_latency_ms",
    "rollback_verify_ms",
    "verify_latency_ms",
}


def _jsonl_rows(path: Path) -> list[Any]:
    rows: list[Any] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return rows


def _walk_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_dicts(item)


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
            "sum": 0.0,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def _find_run_root(variant_root: Path) -> Path:
    if (variant_root / "traces").exists():
        return variant_root
    runs = sorted(
        [p for p in variant_root.glob("run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    return runs[-1] if runs else variant_root


def _collect(run_root: Path) -> tuple[dict[str, list[float]], dict[str, Counter], dict[str, int]]:
    trace_root = run_root / "traces"
    metrics: dict[str, list[float]] = defaultdict(list)
    counters: dict[str, Counter] = defaultdict(Counter)
    file_counts: dict[str, int] = {}

    for path in trace_root.rglob("latency_profile.jsonl"):
        rows = _jsonl_rows(path)
        file_counts["latency_profile.jsonl"] = file_counts.get("latency_profile.jsonl", 0) + len(rows)
        for row in rows:
            if not isinstance(row, dict):
                continue
            event = str(row.get("event") or "unknown")
            context = str(row.get("context") or "all")
            counters["latency_event"][event] += 1
            if row.get("a11y_method"):
                counters["requested_a11y_method"][str(row.get("a11y_method"))] += 1
            if row.get("a11y_actual_method"):
                counters["actual_a11y_method"][str(row.get("a11y_actual_method"))] += 1
            if bool(row.get("a11y_fallback_used")):
                counters["a11y_fallback_used"]["true"] += 1
            for key in NUMERIC_KEYS:
                value = _float(row.get(key))
                if value is None:
                    continue
                metrics[f"latency_profile.{event}.{key}"].append(value)
                if event == "probe_get_state":
                    metrics[f"probe_get_state.{key}"].append(value)
                    metrics[f"probe_get_state.{context}.{key}"].append(value)

    for family in (
        "exploration_branch_trace.jsonl",
        "exploration_page_trace.jsonl",
        "exploration_step_summary.jsonl",
        "rollback_events.jsonl",
    ):
        for path in trace_root.rglob(family):
            rows = _jsonl_rows(path)
            file_counts[family] = file_counts.get(family, 0) + len(rows)
            for row in rows:
                for item in _walk_dicts(row):
                    event = str(item.get("event") or item.get("rollback_mode") or item.get("stop_reason") or "all")
                    for key in NUMERIC_KEYS:
                        value = _float(item.get(key))
                        if value is not None:
                            metrics[f"{family}.{key}"].append(value)
                            metrics[f"{family}.{event}.{key}"].append(value)
                    if item.get("a11y_method"):
                        counters["requested_a11y_method"][str(item.get("a11y_method"))] += 1
                    if item.get("a11y_actual_method"):
                        counters["actual_a11y_method"][str(item.get("a11y_actual_method"))] += 1
                    if bool(item.get("a11y_fallback_used")):
                        counters["a11y_fallback_used"]["true"] += 1

    return metrics, counters, file_counts


def _write_csv(path: Path, metrics: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "count", "mean", "p50", "p95", "min", "max", "sum"],
        )
        writer.writeheader()
        for metric in sorted(metrics):
            row = {"metric": metric}
            row.update(_stats(metrics[metric]))
            writer.writerow(row)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _metric_line(metrics: dict[str, list[float]], key: str, label: str) -> str:
    stat = _stats(metrics.get(key, []))
    return (
        f"| {label} | {stat['count']} | {_fmt(stat['mean'])} | "
        f"{_fmt(stat['p50'])} | {_fmt(stat['p95'])} | {_fmt(stat['max'])} |"
    )


def _write_report(
    path: Path,
    run_root: Path,
    metrics: dict[str, list[float]],
    counters: dict[str, Counter],
    file_counts: dict[str, int],
    csv_path: Path,
) -> None:
    requested = counters.get("requested_a11y_method", Counter())
    actual = counters.get("actual_a11y_method", Counter())
    fallback = counters.get("a11y_fallback_used", Counter()).get("true", 0)
    probe_count = len(metrics.get("probe_get_state.a11y_latency_ms", []))
    fallback_rate = (fallback / probe_count * 100.0) if probe_count else 0.0
    lines = [
        "# A11y / Exploration Latency Breakdown",
        "",
        f"- run_root: `{run_root}`",
        f"- trace_root: `{run_root / 'traces'}`",
        f"- summary_csv: `{csv_path}`",
        "",
        "## A11y 方法与 fallback",
        "",
        f"- requested method counts: `{dict(requested)}`",
        f"- actual method counts: `{dict(actual)}`",
        f"- fast_provider fallback samples: `{fallback}` / `{probe_count}` ({fallback_rate:.2f}%)",
        "",
        "说明：`a11y_method` 是本次请求的方法；`a11y_actual_method` 是真实成功返回的方法。如果 fast_provider 超时并回退到 uiautomator，会计入 fallback。",
        "",
        "## 核心 latency breakdown",
        "",
        "| metric | count | mean ms | p50 ms | p95 ms | max ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        _metric_line(metrics, "probe_get_state.total_ms", "env.get_state total"),
        _metric_line(metrics, "probe_get_state.a11y_latency_ms", "AndroidWorld a11y total"),
        _metric_line(metrics, "probe_get_state.branch.total_ms", "branch state fetch total"),
        _metric_line(metrics, "probe_get_state.branch.a11y_latency_ms", "branch a11y total"),
        _metric_line(metrics, "probe_get_state.fast_a11y_service_ms", "fast_provider service total"),
        _metric_line(metrics, "probe_get_state.fast_a11y_capture_ms", "fast_provider capture"),
        _metric_line(metrics, "probe_get_state.fast_a11y_serialize_ms", "fast_provider serialize"),
        _metric_line(metrics, "probe_get_state.fast_a11y_payload_bytes", "fast_provider payload bytes"),
        _metric_line(metrics, "probe_get_state.fast_a11y_node_count", "fast_provider node count"),
        _metric_line(metrics, "exploration_branch_trace.jsonl.branch_state_fetch_ms", "branch trace state_fetch sum"),
        _metric_line(metrics, "exploration_branch_trace.jsonl.branch_a11y_latency_ms", "branch trace a11y sum"),
        _metric_line(metrics, "exploration_branch_trace.jsonl.a11y_trace_ms", "a11y trace serialization"),
        _metric_line(metrics, "latency_profile.probe_action.latency_ms", "probe action"),
        _metric_line(metrics, "latency_profile.screenshot.latency_ms", "screenshot"),
        _metric_line(metrics, "latency_profile.rollback_verify.latency_ms", "rollback verify"),
        _metric_line(metrics, "latency_profile.rollback_back.latency_ms", "rollback BACK action"),
        _metric_line(metrics, "latency_profile.rollback_home.latency_ms", "rollback HOME action"),
        _metric_line(metrics, "latency_profile.home_replay_action.latency_ms", "home replay action"),
        "",
        "## trace 文件样本数",
        "",
    ]
    for name, count in sorted(file_counts.items()):
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## latency event counts", ""])
    for event, count in counters.get("latency_event", Counter()).most_common():
        lines.append(f"- `{event}`: {count}")
    lines.extend(
        [
            "",
            "## 解读要点",
            "",
            "- 如果 `fast_provider service total` 明显小于 `AndroidWorld a11y total`，瓶颈不在 AccessibilityService 遍历，而在 ADB content read、JSON parse、UIElement 转换或 get_state 包装层。",
            "- 如果 `fallback samples` 很高，本轮不能代表纯 fast_provider；需要先解决 provider 超时或降低并发/branch budget。",
            "- 如果 `branch state fetch total` 接近或大于 `env.get_state total`，探索慢主要来自每个 branch 的状态采样次数，而不是单次 a11y dump。",
            "- 如果 `probe action` 和 `rollback` 很高，优先减少无效 branch 或优化 rollback，不要只优化 a11y。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-root", required=True, help="Variant root, e.g. results/.../LB_MCTS_FINAL")
    parser.add_argument("--run-root", default="", help="Specific run_* dir. Defaults to latest run under variant root.")
    parser.add_argument("--out-dir", default="", help="Output dir. Defaults to run/report/latency_breakdown.")
    args = parser.parse_args()

    variant_root = Path(args.variant_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else _find_run_root(variant_root)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_root / "report" / "latency_breakdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, counters, file_counts = _collect(run_root)
    csv_path = out_dir / "latency_breakdown_summary.csv"
    report_path = out_dir / "latency_breakdown_cn.md"
    _write_csv(csv_path, metrics)
    _write_report(report_path, run_root, metrics, counters, file_counts, csv_path)
    print(f"[latency] report={report_path}")
    print(f"[latency] csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
