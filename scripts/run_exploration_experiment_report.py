#!/usr/bin/env python3
"""Run AndroidWorld exploration experiments and generate a Chinese report.

This script is intentionally self-contained. It can either:

1. launch a full AndroidWorld run with a run-specific exploration trace root, or
2. analyze an existing checkpoint directory and exploration trace root.
"""

from __future__ import annotations

import argparse
import csv
import collections
import datetime as _dt
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT = "explore_agent_gelab"
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "results" / "exploration_full_experiment"
DEFAULT_ADB_PATH = "/Users/huangrunxi/Library/Android/sdk/platform-tools/adb"
ACCESSIBILITY_FORWARDER_SERVICE = (
    "com.google.androidenv.accessibilityforwarder/"
    "com.google.androidenv.accessibilityforwarder.AccessibilityForwarder"
)
FAST_A11Y_SERVICE = "com.androidworld.fasta11y/com.androidworld.fasta11y.FastA11yService"
FAST_A11Y_APK = REPO_ROOT / "tools" / "fast_a11y_dumper" / "build" / "fast-a11y.apk"
FAST_A11Y_BUILD_SCRIPT = REPO_ROOT / "tools" / "fast_a11y_dumper" / "build_apk.sh"


def _run_cmd(
    cmd: list[str],
    env: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def _adb(args: list[str], adb_path: str, serial: str, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    return _run_cmd([adb_path, "-s", serial, *args], timeout=timeout)


def _restart_accessibility_forwarder(adb_path: str, serial: str) -> None:
    commands = [
        ["wait-for-device"],
        ["shell", "input", "keyevent", "3"],
        ["shell", "settings", "put", "secure", "accessibility_enabled", "1"],
        [
            "shell",
            "settings",
            "put",
            "secure",
            "enabled_accessibility_services",
            ACCESSIBILITY_FORWARDER_SERVICE,
        ],
        ["shell", "am", "force-stop", "com.google.androidenv.accessibilityforwarder"],
        ["shell", "input", "keyevent", "3"],
    ]
    for cmd in commands:
        _adb(cmd, adb_path=adb_path, serial=serial, timeout=15.0)
    time.sleep(3.0)


def _preflight_python(a11y_method: str, adb_path: str, console_port: int, timeout: float) -> tuple[bool, str]:
    env = os.environ.copy()
    if a11y_method == "uiautomator":
        env["ANDROID_WORLD_A11Y_METHOD"] = "uiautomator"
    elif a11y_method == "fast_provider":
        env["ANDROID_WORLD_A11Y_METHOD"] = "fast_provider"
    else:
        env.pop("ANDROID_WORLD_A11Y_METHOD", None)
    code = f"""
from android_world.env import env_launcher
env = env_launcher.load_and_setup_env(
    console_port={int(console_port)},
    emulator_setup=False,
    adb_path={adb_path!r},
)
state = env.get_state()
print("A11Y_PREFLIGHT_OK", len(state.ui_elements), state.pixels.shape)
"""
    try:
        result = _run_cmd([sys.executable, "-c", code], env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return False, f"timeout after {timeout}s\n{exc.stdout or ''}"
    return result.returncode == 0 and "A11Y_PREFLIGHT_OK" in result.stdout, result.stdout


def _preflight_uiautomator_dump(adb_path: str, serial: str) -> tuple[bool, str]:
    result = _adb(
        ["shell", "uiautomator", "dump", "/sdcard/window.xml"],
        adb_path=adb_path,
        serial=serial,
        timeout=20.0,
    )
    if result.returncode != 0:
        return False, result.stdout
    check = _adb(
        ["shell", "test", "-s", "/sdcard/window.xml"],
        adb_path=adb_path,
        serial=serial,
        timeout=10.0,
    )
    return check.returncode == 0, result.stdout + check.stdout


def _ensure_fast_a11y_provider(adb_path: str, serial: str) -> str:
    chunks: list[str] = []
    if not FAST_A11Y_APK.exists():
        build = _run_cmd([str(FAST_A11Y_BUILD_SCRIPT)], timeout=120.0)
        chunks.append("build_apk=" + ("ok" if build.returncode == 0 else "failed"))
        chunks.append(build.stdout[-4000:])
        if build.returncode != 0:
            return "\n".join(chunks)
    install = _adb(["install", "-r", str(FAST_A11Y_APK)], adb_path=adb_path, serial=serial, timeout=60.0)
    chunks.append("install=" + ("ok" if install.returncode == 0 else "failed"))
    chunks.append(install.stdout[-4000:])
    _adb(["shell", "settings", "put", "secure", "accessibility_enabled", "1"], adb_path=adb_path, serial=serial)
    current = _adb(
        ["shell", "settings", "get", "secure", "enabled_accessibility_services"],
        adb_path=adb_path,
        serial=serial,
    )
    enabled = current.stdout.strip()
    services = [] if enabled in {"", "null"} else [x for x in enabled.split(":") if x]
    if FAST_A11Y_SERVICE not in services:
        services.append(FAST_A11Y_SERVICE)
    settings = _adb(
        ["shell", "settings", "put", "secure", "enabled_accessibility_services", ":".join(services)],
        adb_path=adb_path,
        serial=serial,
    )
    chunks.append("enable_service=" + ("ok" if settings.returncode == 0 else "failed"))
    _adb(["shell", "input", "keyevent", "3"], adb_path=adb_path, serial=serial)
    time.sleep(1.0)
    smoke = _adb(
        [
            "shell",
            "content",
            "read",
            "--uri",
            "content://com.androidworld.fasta11y.provider/flat?compact=1",
        ],
        adb_path=adb_path,
        serial=serial,
        timeout=20.0,
    )
    chunks.append("content_read_smoke=" + ("ok" if smoke.returncode == 0 and '"ok":true' in smoke.stdout else "failed"))
    chunks.append(smoke.stdout[:1000])
    return "\n".join(chunks)


def _select_a11y_method(args: argparse.Namespace, log_path: Path) -> str:
    requested = str(args.a11y_method).lower()
    if requested not in {"auto", "grpc", "uiautomator", "fast_provider"}:
        raise ValueError("--a11y_method must be one of auto, grpc, uiautomator, fast_provider")

    serial = f"emulator-{int(args.console_port)}"
    preflight_log: list[str] = []
    preflight_log.append(f"requested={requested}")
    preflight_log.append(f"serial={serial}")

    if requested == "fast_provider":
        preflight_log.append(_ensure_fast_a11y_provider(args.adb_path, serial))
        ok, output = _preflight_python(
            "fast_provider",
            adb_path=args.adb_path,
            console_port=int(args.console_port),
            timeout=float(args.a11y_preflight_timeout),
        )
        preflight_log.append("fast_provider_preflight=" + ("ok" if ok else "failed"))
        preflight_log.append(output[-4000:])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n\n".join(preflight_log), encoding="utf-8")
        if ok:
            return "fast_provider"
        raise RuntimeError(f"fast_provider a11y preflight failed. See {log_path}")

    _restart_accessibility_forwarder(args.adb_path, serial)

    if requested in {"auto", "grpc"}:
        ok, output = _preflight_python(
            "grpc",
            adb_path=args.adb_path,
            console_port=int(args.console_port),
            timeout=float(args.a11y_preflight_timeout),
        )
        preflight_log.append("grpc_preflight=" + ("ok" if ok else "failed"))
        preflight_log.append(output[-4000:])
        if ok:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n\n".join(preflight_log), encoding="utf-8")
            return "grpc"
        if requested == "grpc":
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n\n".join(preflight_log), encoding="utf-8")
            raise RuntimeError(
                "gRPC a11y preflight failed. Use --a11y_method=uiautomator or restart the emulator."
            )

    dump_ok, dump_output = _preflight_uiautomator_dump(args.adb_path, serial)
    preflight_log.append("uiautomator_dump=" + ("ok" if dump_ok else "failed"))
    preflight_log.append(dump_output[-2000:])
    ok, output = _preflight_python(
        "uiautomator",
        adb_path=args.adb_path,
        console_port=int(args.console_port),
        timeout=float(args.a11y_preflight_timeout),
    )
    preflight_log.append("uiautomator_preflight=" + ("ok" if ok else "failed"))
    preflight_log.append(output[-4000:])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n\n".join(preflight_log), encoding="utf-8")
    if ok:
        return "uiautomator"
    raise RuntimeError(
        "Both gRPC and uiautomator a11y preflight failed. See "
        f"{log_path} for details."
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                item["_trace_file"] = str(path)
                item["_task_dir"] = str(path.parent)
                item.setdefault("_line", idx)
                rows.append(item)
    return rows


def _load_trace_family(trace_root: Path, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(trace_root.rglob(filename)):
        file_rows = _read_jsonl(path)
        if filename == "action.jsonl":
            for step, row in enumerate(file_rows, start=1):
                row.setdefault("step", step)
        rows.extend(file_rows)
    return rows


def _read_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    if yaml is None:
        try:
            return json.loads(text)
        except Exception:
            return {}
    try:
        cfg = yaml.safe_load(text)
    except Exception:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _load_diagnostic_families(trace_root: Path) -> dict[str, Any]:
    names = [
        "runtime_config.yaml",
        "runtime_config.json",
        "candidate_scores.jsonl",
        "candidate_filter_stats.jsonl",
        "evidence_decisions.jsonl",
        "exploration_step_summary.jsonl",
        "exploration_branch_trace.jsonl",
        "exploration_page_trace.jsonl",
        "exploration_latency.jsonl",
        "state_alignment.jsonl",
        "step_decoupling_status.jsonl",
        "prompt_hint_records.jsonl",
        "prompt_hint_decisions.jsonl",
        "prompt_hints.jsonl",
        "prompt_traces.jsonl",
        "hint_hit_follow.jsonl",
        "rollback_events.jsonl",
        "shortcut_plans.jsonl",
        "shortcut_shadow_eval.jsonl",
    ]
    out: dict[str, Any] = {"files": {}, "rows": {}, "missing": []}
    for name in names:
        paths = sorted(trace_root.rglob(name))
        out["files"][name] = [str(p) for p in paths]
        if not paths:
            out["missing"].append(name)
        if name.endswith(".jsonl"):
            rows: list[dict[str, Any]] = []
            for path in paths:
                rows.extend(_read_jsonl(path))
            out["rows"][name] = rows
        elif name in {"runtime_config.yaml", "runtime_config.json"}:
            configs = []
            for path in paths:
                item = _read_yaml_config(path)
                if item:
                    item["_file"] = str(path)
                    configs.append(item)
            out["rows"][name] = configs
    rollback_events = out["rows"].get("rollback_events.jsonl") or []
    candidate_scores = out["rows"].get("candidate_scores.jsonl") or []
    candidate_filter_stats = out["rows"].get("candidate_filter_stats.jsonl") or []
    evidence_decisions = out["rows"].get("evidence_decisions.jsonl") or []
    prompt_hints = out["rows"].get("prompt_hints.jsonl") or []
    prompt_hint_decisions = out["rows"].get("prompt_hint_decisions.jsonl") or []
    step_decoupling = out["rows"].get("step_decoupling_status.jsonl") or []
    prompt_traces = out["rows"].get("prompt_traces.jsonl") or []
    hint_hit_follow = out["rows"].get("hint_hit_follow.jsonl") or []
    exploration_latency = out["rows"].get("exploration_latency.jsonl") or []
    state_alignment = out["rows"].get("state_alignment.jsonl") or []
    branch_trace_rows = out["rows"].get("exploration_branch_trace.jsonl") or []
    step_summary_rows = out["rows"].get("exploration_step_summary.jsonl") or []
    page_trace_rows = out["rows"].get("exploration_page_trace.jsonl") or []
    def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key) or "")
            if not value:
                value = "<empty>"
            result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))
    injected = [r for r in prompt_hints if bool(r.get("injected"))]
    rejected = [r for r in prompt_hints if not bool(r.get("injected"))]
    evidence_rejected = [r for r in evidence_decisions if not bool(r.get("injected"))]
    candidate_filtered_total = 0
    candidate_raw_total = 0
    quality_examples: list[Any] = []
    for row in exploration_latency:
        candidate_raw_total += int(
            row.get("candidate_count_raw")
            or row.get("raw_candidate_count")
            or 0
        )
        filtered_after = row.get("candidate_count_after_filter") or row.get("filtered_candidate_count")
        raw_count = row.get("candidate_count_raw") or row.get("raw_candidate_count") or 0
        candidate_filtered_total += max(0, int(raw_count) - int(filtered_after or 0))
        for item in row.get("top_filtered_examples") or []:
            if len(quality_examples) < 20:
                quality_examples.append(item)
    if not exploration_latency and candidate_filter_stats:
        for row in candidate_filter_stats:
            candidate_raw_total += int(row.get("raw_candidate_count") or 0)
            candidate_filtered_total += int(row.get("filtered_candidate_count") or 0)
            for item in row.get("top_filtered_examples") or []:
                if len(quality_examples) < 20:
                    quality_examples.append(item)
    out["summary"] = {
        "runtime_config_files": out["files"].get("runtime_config.yaml", []),
        "candidate_scores": len(candidate_scores),
        "candidate_filter_stats": len(candidate_filter_stats),
        "evidence_decisions": len(evidence_decisions),
        "prompt_traces": len(prompt_traces),
        "hint_hit_follow": len(hint_hit_follow),
        "state_alignment": len(state_alignment),
        "step_decoupling_status": len(step_decoupling),
        "decoupled_invalid_steps": sum(1 for r in step_decoupling if not bool(r.get("decoupled_valid", True))),
        "prompt_hint_decisions": len(prompt_hints),
        "prompt_hint_decision_rows": len(prompt_hint_decisions),
        "prompt_hints_injected": sum(1 for r in prompt_hints if bool(r.get("injected"))),
        "prompt_hints_rejected": len(rejected),
        "prompt_hint_rejection_reasons": counts(rejected, "rejected_reason"),
        "evidence_rejection_reasons": counts(evidence_rejected, "rejected_reason"),
        "evidence_type_counts": counts(evidence_decisions, "final_evidence_type"),
        "candidate_operator_counts": counts(candidate_scores, "operator"),
        "candidate_filtered_total": candidate_filtered_total,
        "candidate_raw_total": candidate_raw_total,
        "candidate_quality_examples": quality_examples,
        "exploration_step_summaries": len(step_summary_rows),
        "exploration_branch_traces": len(branch_trace_rows),
        "exploration_page_traces": len(page_trace_rows),
        "state_alignment_aligned": sum(1 for r in state_alignment if bool(r.get("aligned"))),
        "injected_examples": injected[:8],
        "rejected_examples": (evidence_rejected or rejected)[:12],
        "exploration_latency_rows": len(exploration_latency),
        "rollback_events": len(rollback_events),
        "rollback_event_failures": sum(
            1 for r in rollback_events
            if (not bool(r.get("success", True)) or r.get("failure_reasons"))
        ),
        "rollback_level2_triggers": counts(
            [r for r in rollback_events if bool(r.get("level2_triggered"))],
            "level2_trigger_reason",
        ),
        "missing_required_files": list(out["missing"]),
    }
    totals = [float(r.get("exploration_total_ms") or 0.0) for r in exploration_latency]
    branch_a11y = []
    rollback_total = []
    rollback_verify = []
    for row in exploration_latency:
        for branch in row.get("branches") or row.get("branch_latencies") or []:
            if not isinstance(branch, dict):
                continue
            branch_a11y.extend(float(x or 0.0) for x in branch.get("a11y_ms_by_depth") or [])
            if branch.get("depth1_a11y_ms") is not None:
                branch_a11y.append(float(branch.get("depth1_a11y_ms") or 0.0))
            if branch.get("depth2_a11y_ms") is not None:
                branch_a11y.append(float(branch.get("depth2_a11y_ms") or 0.0))
            rollback_total.append(float(branch.get("rollback_total_ms") or 0.0))
            rollback_verify.append(float(branch.get("rollback_verify_ms") or 0.0))
    branch_state = []
    for row in exploration_latency:
        for branch in row.get("branches") or row.get("branch_latencies") or []:
            if not isinstance(branch, dict):
                continue
            branch_state.extend(float(x or 0.0) for x in branch.get("state_fetch_ms_by_depth") or [])
            branch_state.append(float(branch.get("depth1_state_fetch_ms") or 0.0))
            branch_state.append(float(branch.get("depth2_state_fetch_ms") or 0.0))
    out["latency_distribution"] = {
        "exploration_total_ms": _dist(totals),
        "branch_a11y_ms": _dist(branch_a11y),
        "branch_state_fetch_ms": _dist(branch_state),
        "rollback_total_ms": _dist(rollback_total),
        "rollback_verify_ms": _dist(rollback_verify),
    }
    return out


def _dist(values: list[float]) -> dict[str, float | None]:
    finite = sorted(v for v in values if not math.isnan(v) and not math.isinf(v))
    if not finite:
        return {"mean": None, "median": None, "p95": None}
    def pick(q: float) -> float:
        idx = min(len(finite) - 1, max(0, int(round((len(finite) - 1) * q))))
        return float(finite[idx])
    return {
        "mean": float(sum(finite) / len(finite)),
        "median": pick(0.5),
        "p95": pick(0.95),
    }


def _load_episodes(checkpoint_dir: Path | None) -> list[dict[str, Any]]:
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return []
    episodes: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("*.pkl.gz")):
        try:
            with gzip.open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            continue
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item["_checkpoint_file"] = str(path)
                    episodes.append(item)
    return episodes


def _clean(value: Any, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _rate(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return float(num) / float(den)


def _fmt_rate(num: float, den: float) -> str:
    rate = _rate(num, den)
    if rate is None:
        return "n/a"
    return f"{int(num)}/{int(den)} ({rate:.1%})"


def _pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.1%}"


def _resolve_image(path_value: Any) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        return None
    return path


def _img(path_value: Any, alt: str, width: str = "15%") -> str:
    path = _resolve_image(path_value)
    if path is None:
        return ""
    return f'<img src="{path}" alt="{alt}" width="{width}">'


def _point_from_action(action: dict[str, Any]) -> tuple[float, float] | None:
    action_dict = action.get("action_dict") if isinstance(action.get("action_dict"), dict) else {}
    if str(action_dict.get("action_type") or "").lower() == "click":
        x, y = action_dict.get("x"), action_dict.get("y")
        if x is not None and y is not None:
            try:
                return float(x), float(y)
            except (TypeError, ValueError):
                pass
    tool_call = action.get("tool_call") if isinstance(action.get("tool_call"), dict) else {}
    args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
    if str(args.get("action") or "").lower() == "click":
        for key in ("coordinate", "point", "position"):
            value = args.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                try:
                    return float(value[0]), float(value[1])
                except (TypeError, ValueError):
                    return None
    parsed = action.get("parsed_action") if isinstance(action.get("parsed_action"), dict) else {}
    if str(parsed.get("action") or "").lower() == "click":
        value = parsed.get("point")
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None
    return None


def _text_from_action(action: dict[str, Any]) -> str:
    action_dict = action.get("action_dict") if isinstance(action.get("action_dict"), dict) else {}
    if str(action_dict.get("action_type") or "").lower() == "input_text":
        return _clean(action_dict.get("text"), 256)
    tool_call = action.get("tool_call") if isinstance(action.get("tool_call"), dict) else {}
    args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
    if str(args.get("action") or "").lower() in {"type", "input_text"}:
        return _clean(args.get("text") or args.get("value"), 256)
    parsed = action.get("parsed_action") if isinstance(action.get("parsed_action"), dict) else {}
    if str(parsed.get("action") or "").lower() in {"type", "input_text"}:
        return _clean(parsed.get("value") or parsed.get("text"), 256)
    return ""


def _candidate_center_and_bbox(candidate: dict[str, Any]) -> tuple[tuple[float, float] | None, dict[str, float] | None]:
    center = candidate.get("center")
    if not center and isinstance(candidate.get("a11y"), dict):
        center = candidate["a11y"].get("center")
    parsed_center = None
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        try:
            parsed_center = (float(center[0]), float(center[1]))
        except (TypeError, ValueError):
            parsed_center = None
    bbox = candidate.get("bbox")
    if not isinstance(bbox, dict) and isinstance(candidate.get("a11y"), dict):
        bbox = candidate["a11y"].get("bbox")
    parsed_bbox = None
    if isinstance(bbox, dict):
        try:
            parsed_bbox = {
                "x_min": float(bbox["x_min"]),
                "x_max": float(bbox["x_max"]),
                "y_min": float(bbox["y_min"]),
                "y_max": float(bbox["y_max"]),
            }
        except (KeyError, TypeError, ValueError):
            parsed_bbox = None
    return parsed_center, parsed_bbox


def _point_hits_candidate(point: tuple[float, float], candidate: dict[str, Any]) -> tuple[bool, float | None]:
    center, bbox = _candidate_center_and_bbox(candidate)
    x, y = point
    if bbox and bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]:
        return True, 0.0
    if center:
        distance = math.hypot(x - center[0], y - center[1])
        return distance <= 120.0, distance
    return False, None


def _hint_follow(action: dict[str, Any]) -> dict[str, Any]:
    results = action.get("matched_exploration_results")
    if not isinstance(results, list) or not results:
        return {"eligible": False, "followed": False, "reason": "no_matched_results"}
    actionable_results = [
        item for item in results
        if isinstance(item, dict) and str(item.get("hint_kind") or "actionable") != "evidence_only"
    ]
    if not actionable_results:
        return {"eligible": False, "followed": False, "reason": "evidence_only_hint"}
    typed_text = _text_from_action(action)
    if typed_text:
        for item in actionable_results:
            if not isinstance(item, dict):
                continue
            candidate = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
            if str(candidate.get("action_kind") or "").lower() != "type":
                continue
            target = _clean(candidate.get("text"), 256)
            if target and target == typed_text:
                return {
                    "eligible": True,
                    "followed": True,
                    "reason": "type_matched_candidate",
                    "text": typed_text,
                    "label": _clean(candidate.get("label") or item.get("next_label"), 80),
                }
        return {"eligible": True, "followed": False, "reason": "type_missed_candidates", "text": typed_text}
    point = _point_from_action(action)
    if point is None:
        return {"eligible": True, "followed": False, "reason": "no_click_point"}
    best_distance: float | None = None
    best_label = ""
    for item in actionable_results:
        if not isinstance(item, dict):
            continue
        candidate = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
        hit, distance = _point_hits_candidate(point, candidate)
        label = _clean(candidate.get("label") or item.get("next_label"), 80)
        if distance is not None and (best_distance is None or distance < best_distance):
            best_distance = distance
            best_label = label
        if hit:
            return {
                "eligible": True,
                "followed": True,
                "reason": "click_matched_candidate",
                "point": point,
                "label": label,
                "distance": distance,
            }
    return {
        "eligible": True,
        "followed": False,
        "reason": "click_missed_candidates",
        "point": point,
        "label": best_label,
        "distance": best_distance,
    }


def _collect_rollbacks(explorations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in explorations:
        for rb in trace.get("rollbacks") or []:
            if not isinstance(rb, dict):
                continue
            item = dict(rb)
            item["_task_dir"] = trace.get("_task_dir")
            item["_trace_file"] = trace.get("_trace_file")
            item["goal"] = trace.get("goal")
            item["step"] = trace.get("step")
            item["root_screenshot"] = trace.get("root_screenshot")
            rows.append(item)
    return rows


def _summarize(
    episodes: list[dict[str, Any]],
    explorations: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    rollbacks = _collect_rollbacks(explorations)
    complete_episodes = [e for e in episodes if not e.get("exception_info")]
    failed_episodes = [e for e in episodes if e.get("exception_info")]
    successful_episodes = [
        e for e in complete_episodes if float(e.get("is_successful") or 0.0) > 0.0
    ]
    episode_lengths: list[float] = []
    successful_episode_lengths: list[float] = []
    task_episode_rows: list[dict[str, Any]] = []
    for episode in episodes:
        try:
            episode_length = float(episode.get("episode_length"))
        except (TypeError, ValueError):
            episode_length = math.nan
        try:
            success_value = float(episode.get("is_successful") or 0.0)
        except (TypeError, ValueError):
            success_value = 0.0
        task_name = _clean(episode.get("task_template") or episode.get("goal") or Path(str(episode.get("_checkpoint_file") or "")).stem, 80)
        task_episode_rows.append(
            {
                "task": task_name,
                "episode_length": None if math.isnan(episode_length) or math.isinf(episode_length) else episode_length,
                "success": bool(success_value > 0.0),
                "exception": bool(episode.get("exception_info")),
            }
        )
        if not episode.get("exception_info") and not math.isnan(episode_length) and not math.isinf(episode_length):
            episode_lengths.append(episode_length)
            if success_value > 0.0:
                successful_episode_lengths.append(episode_length)

    status_counts = collections.Counter(str(t.get("status") or "unknown") for t in explorations)
    exploration_available = [t for t in explorations if bool(t.get("available_for_next_prompt"))]
    completed_exploration = [t for t in explorations if str(t.get("status")) == "completed"]
    depth2_observations = 0
    total_observations = 0
    branch_state_fetch_ms: list[float] = []
    branch_a11y_latency_ms: list[float] = []
    for trace in explorations:
        for obs in trace.get("observations") or []:
            if isinstance(obs, dict):
                total_observations += 1
                if int(obs.get("depth_reached") or 0) >= 2:
                    depth2_observations += 1
                branch_state_fetch_ms.append(float(obs.get("branch_state_fetch_ms") or obs.get("state_fetch_ms") or 0.0))
                branch_a11y_latency_ms.append(float(obs.get("branch_a11y_latency_ms") or obs.get("a11y_dump_ms") or 0.0))

    rollback_success = sum(1 for rb in rollbacks if bool(rb.get("success")))
    rollback_by_level: dict[str, dict[str, int]] = {}
    for rb in rollbacks:
        level = str(rb.get("level") or "unknown")
        rollback_by_level.setdefault(level, {"total": 0, "success": 0})
        rollback_by_level[level]["total"] += 1
        if bool(rb.get("success")):
            rollback_by_level[level]["success"] += 1

    pending_match_steps = [m for m in matches if int(m.get("pending_trace_count") or 0) > 0]
    matched_steps = [m for m in pending_match_steps if int(m.get("matched_count") or 0) > 0]
    candidate_match_count = sum(int(m.get("candidate_match_count") or 0) for m in matches)
    matched_candidate_count = sum(int(m.get("matched_count") or 0) for m in matches)

    hint_actions = []
    hint_evaluable = []
    hint_evidence_only = []
    hint_followed = []
    hint_missed = []
    for action in actions:
        if action.get("prompt_hint"):
            follow = _hint_follow(action)
            record = dict(action)
            record["_hint_follow"] = follow
            hint_actions.append(record)
            if not follow.get("eligible") and follow.get("reason") == "evidence_only_hint":
                hint_evidence_only.append(record)
            elif follow.get("eligible"):
                hint_evaluable.append(record)
            if follow.get("eligible") and follow.get("followed"):
                hint_followed.append(record)
            elif follow.get("eligible"):
                hint_missed.append(record)

    task_stats: dict[str, dict[str, Any]] = {}
    for row in actions:
        task_dir = str(row.get("_task_dir") or "")
        task = Path(task_dir).name if task_dir else _clean(row.get("goal"), 60)
        stats = task_stats.setdefault(
            task,
            {
                "steps": 0,
                "hint_steps": 0,
                "hint_followed": 0,
                "exploration_attempts": 0,
                "observations": 0,
            },
        )
        stats["steps"] += 1
        if row.get("prompt_hint"):
            stats["hint_steps"] += 1
    for trace in explorations:
        task_dir = str(trace.get("_task_dir") or "")
        task = Path(task_dir).name if task_dir else _clean(trace.get("goal"), 60)
        stats = task_stats.setdefault(
            task,
            {
                "steps": 0,
                "hint_steps": 0,
                "hint_followed": 0,
                "exploration_attempts": 0,
                "observations": 0,
            },
        )
        stats["exploration_attempts"] += 1
        stats["observations"] += len(trace.get("observations") or [])
    for row in hint_followed:
        task_dir = str(row.get("_task_dir") or "")
        task = Path(task_dir).name if task_dir else _clean(row.get("goal"), 60)
        task_stats.setdefault(task, {})["hint_followed"] = task_stats.get(task, {}).get("hint_followed", 0) + 1

    return {
        "episodes": {
            "total": len(episodes),
            "complete": len(complete_episodes),
            "exception_failures": len(failed_episodes),
            "success": len(successful_episodes),
            "success_rate_complete_only": _rate(len(successful_episodes), len(complete_episodes)),
            "success_rate_all_trials": _rate(len(successful_episodes), len(episodes)),
            "avg_episode_length_complete": _safe_mean(episode_lengths),
            "avg_episode_length_success": _safe_mean(successful_episode_lengths),
            "task_episode_rows": task_episode_rows,
        },
        "actions": {
            "total_steps": len(actions),
            "prompt_hint_steps": len(hint_actions),
            "prompt_hint_evaluable_steps": len(hint_evaluable),
            "prompt_hint_evidence_only_steps": len(hint_evidence_only),
            "prompt_hint_followed": len(hint_followed),
            "prompt_hint_missed": len(hint_missed),
            "prompt_hint_follow_rate": _rate(len(hint_followed), len(hint_evaluable)),
            "avg_exploration_candidates_per_step": _safe_mean(
                [float(a.get("exploration_candidate_count") or 0.0) for a in actions]
            ),
            "avg_selected_targets_per_step": _safe_mean(
                [float(a.get("exploration_selected_target_count") or 0.0) for a in actions]
            ),
            "avg_observations_per_step": _safe_mean(
                [float(a.get("exploration_observation_count") or 0.0) for a in actions]
            ),
            "avg_vlm_latency_ms": _safe_mean([float(a.get("vlm_latency_ms") or 0.0) for a in actions]),
            "avg_step_latency_ms": _safe_mean([float(a.get("latency_sec") or 0.0) * 1000.0 for a in actions]),
            "avg_exploration_latency_ms": _safe_mean([float(a.get("exploration_latency_ms") or 0.0) for a in actions]),
            "avg_exploration_wait_after_vlm_ms": _safe_mean(
                [
                    float((a.get("exploration_async") or {}).get("wait_after_vlm_ms") or 0.0)
                    for a in actions
                    if isinstance(a.get("exploration_async"), dict)
                ]
            ),
        },
        "exploration": {
            "total_traces": len(explorations),
            "status_counts": dict(status_counts),
            "completed": len(completed_exploration),
            "available_for_next_prompt": len(exploration_available),
            "available_rate": _rate(len(exploration_available), len(explorations)),
            "total_observations": total_observations,
            "depth2_observations": depth2_observations,
            "depth2_rate": _rate(depth2_observations, total_observations),
            "avg_latency_ms": _safe_mean([float(t.get("latency_ms") or 0.0) for t in explorations]),
            "avg_state_fetch_ms": _safe_mean([float(t.get("exploration_state_fetch_ms") or 0.0) for t in explorations]),
            "avg_a11y_dump_ms": _safe_mean([float(t.get("exploration_a11y_dump_ms") or 0.0) for t in explorations]),
            "avg_summary_ms": _safe_mean([float(t.get("exploration_summary_ms") or 0.0) for t in explorations]),
            "avg_a11y_trace_ms": _safe_mean([float(t.get("exploration_a11y_trace_ms") or 0.0) for t in explorations]),
            "avg_branch_state_fetch_ms": _safe_mean(branch_state_fetch_ms),
            "avg_branch_a11y_latency_ms": _safe_mean(branch_a11y_latency_ms),
        },
        "state_match": {
            "pending_steps": len(pending_match_steps),
            "matched_steps": len(matched_steps),
            "step_match_rate": _rate(len(matched_steps), len(pending_match_steps)),
            "candidate_match_count": candidate_match_count,
            "matched_candidate_count": matched_candidate_count,
            "candidate_match_rate": _rate(matched_candidate_count, candidate_match_count),
        },
        "rollback": {
            "total": len(rollbacks),
            "success": rollback_success,
            "success_rate": _rate(rollback_success, len(rollbacks)),
            "by_level": rollback_by_level,
            "failures": len(rollbacks) - rollback_success,
        },
        "task_stats": task_stats,
        "_hint_actions": hint_actions,
        "_hint_followed": hint_followed,
        "_hint_missed": hint_missed,
        "_rollbacks": rollbacks,
    }


def _safe_mean(values: list[float]) -> float | None:
    finite = [v for v in values if not math.isnan(v) and not math.isinf(v)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _load_baseline_table(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith(("task_num", "task ")):
            continue
        if "=========" in line:
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        task_name = parts[0]
        task_num = _float_or_none(parts[1])
        if task_num is None:
            continue
        rows[task_name] = {
            "task": task_name,
            "task_num": int(task_num),
            "num_complete_trials": _float_or_none(parts[2]),
            "mean_success_rate": _float_or_none(parts[3]),
            "mean_episode_length": _float_or_none(parts[4]),
            "total_runtime_s": _float_or_none(parts[5]),
            "mean_step_latency_s": _float_or_none(parts[6]),
            "num_fail_trials": _float_or_none(parts[7]),
        }
    return rows


def _attach_baseline(summary: dict[str, Any], baseline_rows: dict[str, dict[str, Any]]) -> None:
    compare_rows: list[dict[str, Any]] = []
    for row in summary["episodes"].get("task_episode_rows") or []:
        task = str(row.get("task") or "")
        baseline = baseline_rows.get(task)
        if not baseline:
            continue
        exploration_steps = _float_or_none(row.get("episode_length"))
        exploration_success = 1.0 if bool(row.get("success")) else 0.0
        baseline_steps = _float_or_none(baseline.get("mean_episode_length"))
        baseline_success = _float_or_none(baseline.get("mean_success_rate"))
        compare_rows.append(
            {
                "task": task,
                "baseline_success_rate": baseline_success,
                "baseline_episode_length": baseline_steps,
                "baseline_step_latency_s": _float_or_none(baseline.get("mean_step_latency_s")),
                "exploration_success": exploration_success,
                "exploration_episode_length": exploration_steps,
                "step_delta": (
                    None
                    if baseline_steps is None or exploration_steps is None
                    else exploration_steps - baseline_steps
                ),
                "success_delta": (
                    None
                    if baseline_success is None
                    else exploration_success - baseline_success
                ),
            }
        )
    matched = len(compare_rows)
    baseline_success_values = [
        float(row["baseline_success_rate"])
        for row in compare_rows
        if row.get("baseline_success_rate") is not None
    ]
    exploration_success_values = [float(row["exploration_success"]) for row in compare_rows]
    baseline_step_values = [
        float(row["baseline_episode_length"])
        for row in compare_rows
        if row.get("baseline_episode_length") is not None
    ]
    exploration_step_values = [
        float(row["exploration_episode_length"])
        for row in compare_rows
        if row.get("exploration_episode_length") is not None
    ]
    summary["baseline_compare"] = {
        "baseline_file_loaded": bool(baseline_rows),
        "baseline_task_count": len(baseline_rows),
        "matched_task_count": matched,
        "avg_baseline_success_rate": _safe_mean(baseline_success_values),
        "avg_exploration_success_rate": _safe_mean(exploration_success_values),
        "avg_baseline_episode_length": _safe_mean(baseline_step_values),
        "avg_exploration_episode_length": _safe_mean(exploration_step_values),
        "rows": compare_rows,
    }


def _plot_bar(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    if not labels:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig_w = max(6, min(16, 0.55 * len(labels) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_stacked_rollback(path: Path, summary: dict[str, Any]) -> None:
    by_level = summary["rollback"]["by_level"]
    labels = sorted(by_level)
    if not labels:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    success = [by_level[k]["success"] for k in labels]
    fail = [by_level[k]["total"] - by_level[k]["success"] for k in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, success, label="success", color="#59A14F")
    ax.bar(labels, fail, bottom=success, label="fail", color="#E15759")
    ax.set_title("Rollback success by level")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _make_charts(summary: dict[str, Any], report_dir: Path) -> dict[str, Path]:
    figures = report_dir / "figures"
    charts: dict[str, Path] = {}

    episode_counts = {
        "success": int(summary["episodes"]["success"] or 0),
        "fail": int(summary["episodes"]["complete"] or 0) - int(summary["episodes"]["success"] or 0),
        "exception": int(summary["episodes"]["exception_failures"] or 0),
    }
    path = figures / "episode_success_counts.png"
    _plot_bar(path, "Episode success / fail / exception", list(episode_counts), [episode_counts[k] for k in episode_counts], "count")
    if path.exists():
        charts["episode_success"] = path

    avg_step_labels = ["complete", "success"]
    avg_step_values = [
        float(summary["episodes"].get("avg_episode_length_complete") or 0.0),
        float(summary["episodes"].get("avg_episode_length_success") or 0.0),
    ]
    path = figures / "episode_average_steps.png"
    _plot_bar(path, "Average episode length", avg_step_labels, avg_step_values, "steps")
    if path.exists():
        charts["episode_steps_avg"] = path

    task_episode_rows = list(summary["episodes"].get("task_episode_rows") or [])
    finite_task_rows = [row for row in task_episode_rows if row.get("episode_length") is not None]
    if finite_task_rows:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            labels = [str(row.get("task") or "")[:26] for row in finite_task_rows]
            values = [float(row.get("episode_length") or 0.0) for row in finite_task_rows]
            colors = [
                "#59A14F" if bool(row.get("success")) else "#E15759"
                for row in finite_task_rows
            ]
            fig_w = max(8, min(20, 0.5 * len(labels) + 3))
            fig, ax = plt.subplots(figsize=(fig_w, 4.5))
            ax.bar(labels, values, color=colors)
            ax.set_title("Episode length by task")
            ax.set_ylabel("steps")
            ax.tick_params(axis="x", rotation=40, labelsize=8)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            path = figures / "episode_steps_by_task.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180)
            plt.close(fig)
            if path.exists():
                charts["episode_steps_by_task"] = path
        except Exception:
            pass

    status_counts = summary["exploration"]["status_counts"]
    path = figures / "exploration_status_counts.png"
    _plot_bar(path, "Exploration status counts", list(status_counts), [status_counts[k] for k in status_counts], "count")
    if path.exists():
        charts["exploration_status"] = path

    path = figures / "rollback_success_by_level.png"
    _plot_stacked_rollback(path, summary)
    if path.exists():
        charts["rollback"] = path

    rates = {
        "state-step-match": summary["state_match"]["step_match_rate"],
        "state-cand-match": summary["state_match"]["candidate_match_rate"],
        "hint-follow": summary["actions"]["prompt_hint_follow_rate"],
        "depth>=2": summary["exploration"]["depth2_rate"],
        "next-prompt-ready": summary["exploration"]["available_rate"],
    }
    labels = list(rates)
    values = [float((rates[k] or 0.0) * 100.0) for k in labels]
    path = figures / "key_rates.png"
    _plot_bar(path, "Key exploration rates", labels, values, "percent")
    if path.exists():
        charts["rates"] = path

    task_rows = sorted(
        summary["task_stats"].items(),
        key=lambda kv: int(kv[1].get("exploration_attempts") or 0),
        reverse=True,
    )[:20]
    labels = [k[:28] for k, _ in task_rows]
    values = [int(v.get("exploration_attempts") or 0) for _, v in task_rows]
    path = figures / "top_task_exploration_attempts.png"
    _plot_bar(path, "Top task exploration attempts", labels, values, "attempts")
    if path.exists():
        charts["tasks"] = path

    baseline = summary.get("baseline_compare") if isinstance(summary.get("baseline_compare"), dict) else {}
    if baseline and baseline.get("matched_task_count"):
        path = figures / "baseline_success_comparison.png"
        _plot_bar(
            path,
            "Success rate: exploration vs baseline",
            ["baseline", "exploration"],
            [
                float((baseline.get("avg_baseline_success_rate") or 0.0) * 100.0),
                float((baseline.get("avg_exploration_success_rate") or 0.0) * 100.0),
            ],
            "percent",
        )
        if path.exists():
            charts["baseline_success"] = path

        path = figures / "baseline_steps_comparison.png"
        _plot_bar(
            path,
            "Episode length: exploration vs baseline",
            ["baseline", "exploration"],
            [
                float(baseline.get("avg_baseline_episode_length") or 0.0),
                float(baseline.get("avg_exploration_episode_length") or 0.0),
            ],
            "steps",
        )
        if path.exists():
            charts["baseline_steps"] = path

        rows = [
            row for row in baseline.get("rows") or []
            if row.get("step_delta") is not None
        ]
        if rows:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                rows = rows[:40]
                labels = [str(row.get("task") or "")[:28] for row in rows]
                deltas = [float(row.get("step_delta") or 0.0) for row in rows]
                colors = ["#E15759" if delta > 0 else "#59A14F" for delta in deltas]
                fig_w = max(8, min(22, 0.48 * len(labels) + 3))
                fig, ax = plt.subplots(figsize=(fig_w, 4.8))
                ax.bar(labels, deltas, color=colors)
                ax.axhline(0, color="#333333", linewidth=0.8)
                ax.set_title("Exploration step delta vs baseline")
                ax.set_ylabel("exploration steps - baseline steps")
                ax.tick_params(axis="x", rotation=40, labelsize=8)
                ax.grid(axis="y", alpha=0.25)
                fig.tight_layout()
                path = figures / "baseline_step_delta_by_task.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=180)
                plt.close(fig)
                if path.exists():
                    charts["baseline_step_delta"] = path
            except Exception:
                pass
    return charts


def _build_step_metrics(
    actions: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    match_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for match in matches:
        try:
            step = int(match.get("step") or 0)
        except (TypeError, ValueError):
            step = 0
        match_by_key[(str(match.get("_task_dir") or ""), step)] = match

    rows: list[dict[str, Any]] = []
    for action in actions:
        try:
            step = int(action.get("step") or 0)
        except (TypeError, ValueError):
            step = 0
        task_dir = str(action.get("_task_dir") or "")
        match = match_by_key.get((task_dir, step), {})
        follow = _hint_follow(action) if action.get("prompt_hint") else {
            "eligible": False,
            "followed": False,
            "reason": "no_prompt_hint",
        }
        selected_results = action.get("matched_exploration_results") or []
        suggested_labels = []
        for item in selected_results:
            if not isinstance(item, dict):
                continue
            cand = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
            suggested_labels.append(_clean(cand.get("label") or item.get("next_label"), 80))
        shortcut_event = action.get("shortcut_event") if isinstance(action.get("shortcut_event"), dict) else {}
        planned_root_summary = action.get("planned_root_action_summary")
        if planned_root_summary is None:
            planned_root_summary = shortcut_event.get("planned_root_action")
        shortcut_t2_summary = action.get("shortcut_t2_action_summary")
        if shortcut_t2_summary is None:
            shortcut_t2_summary = shortcut_event.get("shortcut_t2_action")
        rows.append(
            {
                "task_dir": Path(task_dir).name if task_dir else "",
                "goal": _clean(action.get("goal"), 240),
                "step": step,
                "prompt_mode": action.get("prompt_mode") or "",
                "pre_reasoning_pending_traces": int(match.get("pending_trace_count") or 0),
                "pre_reasoning_candidate_matches": int(match.get("candidate_match_count") or 0),
                "pre_reasoning_matched_results": int(match.get("matched_count") or 0),
                "hint_injected": bool(action.get("prompt_hint")),
                "hint_followed": bool(follow.get("followed")),
                "hint_follow_reason": follow.get("reason") or "",
                "suggested_next_elements": "; ".join(x for x in suggested_labels if x),
                "main_action": _short_action(action),
                "post_planning_exploration_status": action.get("exploration_status") or "",
                "exploration_mode": action.get("exploration_mode") or "",
                "exploration_decoupled_from_planned_action": bool(action.get("exploration_decoupled_from_planned_action")),
                "exploration_parallel_with_vlm": bool(action.get("exploration_parallel_with_vlm")),
                "post_planning_exploration_trigger": action.get("exploration_trigger_reason") or "",
                "post_planning_candidate_count": int(action.get("exploration_candidate_count") or 0),
                "post_planning_selected_targets": int(action.get("exploration_selected_target_count") or 0),
                "post_planning_observations": int(action.get("exploration_observation_count") or 0),
                "post_planning_speculative_results": len(action.get("post_planning_exploration_speculative_results") or []),
                "vlm_latency_ms": float(action.get("vlm_latency_ms") or 0.0),
                "exploration_latency_ms": float(action.get("exploration_latency_ms") or 0.0),
                "exploration_state_fetch_ms": float(action.get("exploration_state_fetch_ms") or 0.0),
                "exploration_a11y_dump_ms": float(action.get("exploration_a11y_dump_ms") or 0.0),
                "exploration_summary_ms": float(action.get("exploration_summary_ms") or 0.0),
                "exploration_a11y_trace_ms": float(action.get("exploration_a11y_trace_ms") or 0.0),
                "exploration_wait_after_vlm_ms": float(
                    (action.get("exploration_async") or {}).get("wait_after_vlm_ms") or 0.0
                ) if isinstance(action.get("exploration_async"), dict) else 0.0,
                "rollback_success": action.get("rollback_success"),
                "rollback_levels": ",".join(str(x) for x in (action.get("rollback_levels") or [])),
                "shortcut_mode": action.get("shortcut_mode") or "",
                "shortcut_attempted": bool(action.get("shortcut_attempted")),
                "shortcut_plan_created": bool(action.get("shortcut_plan_created")),
                "shortcut_would_fire": bool(action.get("shortcut_would_fire") or action.get("shortcut_event_would_fire")),
                "shortcut_fired": bool(action.get("shortcut_fired")),
                "skipped_vlm_call": bool(action.get("skipped_vlm_call") or action.get("skipped_vlm_calls")),
                "no_plan_reason": action.get("no_plan_reason") or "",
                "no_fire_reason": action.get("no_fire_reason") or action.get("shortcut_event_no_fire_reason") or "",
                "planned_root_action_summary": json.dumps(planned_root_summary or {}, ensure_ascii=False, default=str),
                "shortcut_t2_action_summary": json.dumps(shortcut_t2_summary or {}, ensure_ascii=False, default=str),
                "shortcut_pattern": (
                    (shortcut_t2_summary or {}).get("shortcut_pattern")
                    if isinstance(shortcut_t2_summary, dict)
                    else ""
                ),
                "is_search_input": bool((shortcut_t2_summary or {}).get("is_safe_search_input")) if isinstance(shortcut_t2_summary, dict) else False,
                "is_exact_result_click": bool((shortcut_t2_summary or {}).get("is_exact_result_click")) if isinstance(shortcut_t2_summary, dict) else False,
                "t1_state_match_method": action.get("t1_state_match_method") or shortcut_event.get("t1_state_match_method") or "",
                "t1_state_match_score": action.get("t1_state_match_score"),
                "t2_target_visible": bool(action.get("t2_target_visible")),
                "shortcut_confidence": action.get("shortcut_confidence") or action.get("shortcut_event_confidence"),
                "latency_sec": float(action.get("latency_sec") or 0.0),
                "summary": _clean(action.get("summary"), 300),
            }
        )
    return rows


def _write_step_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _short_action(row: dict[str, Any]) -> str:
    action_dict = row.get("action_dict") if isinstance(row.get("action_dict"), dict) else {}
    action_type = action_dict.get("action_type") or (row.get("tool_call") or {}).get("arguments", {}).get("action")
    point = _point_from_action(row)
    suffix = f"@({int(point[0])},{int(point[1])})" if point else ""
    return f"{action_type or 'unknown'}{suffix}"


def _a11y_excerpt(items: list[dict[str, Any]], limit: int = 8) -> str:
    compact = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "text": item.get("text"),
                "desc": item.get("content_description"),
                "class": item.get("class_name"),
                "center": item.get("center"),
                "clickable": item.get("is_clickable"),
            }
        )
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _exploration_case_sections(explorations: list[dict[str, Any]], max_cases: int) -> list[str]:
    candidates = [
        t for t in explorations
        if t.get("observations") and str(t.get("status")) in {"completed", "rollback_failed"}
    ]
    candidates.sort(
        key=lambda t: (
            1 if str(t.get("status")) == "rollback_failed" else 0,
            len(t.get("observations") or []),
            float(t.get("latency_ms") or 0.0),
        ),
        reverse=True,
    )
    lines: list[str] = []
    for idx, trace in enumerate(candidates[:max_cases], start=1):
        lines.extend(
            [
                f"### Case {idx}: exploration step {trace.get('step')}",
                "",
                f"- task instruction: `{_clean(trace.get('goal'), 260)}`",
                f"- status: `{trace.get('status')}`; trigger: `{trace.get('trigger_reason')}`",
                f"- candidates: {trace.get('candidate_count')}; selected targets: {len(trace.get('selected_targets') or [])}; latency: {float(trace.get('latency_ms') or 0.0):.1f} ms",
                f"- available for next prompt: `{bool(trace.get('available_for_next_prompt'))}`",
                "",
            ]
        )
        root_img = _img(trace.get("root_screenshot"), "root screenshot")
        if root_img:
            lines.extend(["Root screenshot:", "", root_img, ""])
        selected = trace.get("selected_targets") or []
        if selected:
            lines.extend(
                [
                    "| rank | selected element | score | relevance | center |",
                    "| ---: | --- | ---: | ---: | --- |",
                ]
            )
            for rank, cand in enumerate(selected[:5], start=1):
                if not isinstance(cand, dict):
                    continue
                lines.append(
                    f"| {rank} | `{_clean(cand.get('label'), 80)}` | {float(cand.get('score') or 0):.3f} | "
                    f"{float(cand.get('relevance') or 0):.3f} | `{cand.get('center')}` |"
                )
            lines.append("")
        for obs in (trace.get("observations") or [])[:2]:
            if not isinstance(obs, dict):
                continue
            rb = obs.get("rollback") if isinstance(obs.get("rollback"), dict) else {}
            lines.extend(
                [
                    f"Branch {obs.get('branch_id')}: path=`{' -> '.join(obs.get('labels') or [])}`, score={float(obs.get('score') or 0):.3f}, depth={obs.get('depth_reached')}, rollback={rb.get('level')}/{rb.get('mode')}/{rb.get('success')}",
                    "",
                ]
            )
            for step in (obs.get("steps") or [])[:3]:
                if not isinstance(step, dict):
                    continue
                cand = step.get("candidate") if isinstance(step.get("candidate"), dict) else {}
                action_kind = _clean(cand.get("action_kind") or "click", 20)
                lines.extend(
                    [
                        f"- depth {step.get('depth')}: action=`{action_kind} {_clean(cand.get('label'), 80)}`; score={float(cand.get('score') or 0):.3f}; changed={step.get('changed')}; activity=`{_clean(step.get('after_activity'), 100)}`",
                    ]
                )
                shot = _img(step.get("screenshot"), f"case {idx} depth {step.get('depth')}")
                if shot:
                    lines.extend(["", shot, ""])
            lines.extend(
                [
                    "Observed elements:",
                    "",
                    "```text",
                    _clean("; ".join(obs.get("observed_elements") or []), 500),
                    "```",
                    "",
                ]
            )
        if trace.get("root_a11y"):
            lines.extend(
                [
                    "Root a11y excerpt:",
                    "",
                    "```json",
                    _a11y_excerpt(trace.get("root_a11y") or []),
                    "```",
                    "",
                ]
            )
    if not lines:
        lines.append("没有找到包含 observation 的 exploration trace。")
        lines.append("")
    return lines


def _hint_case_sections(summary: dict[str, Any], max_cases: int) -> list[str]:
    rows = list(summary.get("_hint_followed") or []) + list(summary.get("_hint_missed") or [])
    lines: list[str] = []
    for idx, row in enumerate(rows[:max_cases], start=1):
        follow = row.get("_hint_follow") or {}
        lines.extend(
            [
                f"### Hint Case {idx}",
                "",
                f"- task instruction: `{_clean(row.get('goal'), 260)}`",
                f"- step: {row.get('step')}; action: `{_short_action(row)}`; followed={follow.get('followed')}; reason=`{follow.get('reason')}`",
                f"- matched exploration count: {row.get('matched_exploration_count')}",
                "",
                "Injected hint:",
                "",
                "```text",
                _clean(row.get("prompt_hint"), 1200),
                "```",
                "",
            ]
        )
        results = row.get("matched_exploration_results") or []
        if results:
            lines.extend(
                [
                    "| rank | next explored element | score | matched by | depth2 screenshot |",
                    "| ---: | --- | ---: | --- | --- |",
                ]
            )
            for result in results[:3]:
                if not isinstance(result, dict):
                    continue
                cand = result.get("next_candidate") if isinstance(result.get("next_candidate"), dict) else {}
                screenshot = _img(result.get("depth2_screenshot"), "depth2", "15%")
                lines.append(
                    f"| {result.get('rank', '')} | `{_clean(cand.get('label') or result.get('next_label'), 80)}` | "
                    f"{float(result.get('score') or 0):.3f} | `{result.get('matched_by')}` | {screenshot} |"
                )
            lines.append("")
    if not lines:
        lines.append("没有发现被注入到 prompt 的 exploration hint。")
        lines.append("")
    return lines


def _failure_sections(summary: dict[str, Any], max_cases: int) -> list[str]:
    failures = [rb for rb in summary.get("_rollbacks", []) if not bool(rb.get("success"))]
    lines: list[str] = []
    for idx, rb in enumerate(failures[:max_cases], start=1):
        analysis = rb.get("failure_analysis") if isinstance(rb.get("failure_analysis"), dict) else {}
        lines.extend(
            [
                f"### Rollback Failure {idx}",
                "",
                f"- task instruction: `{_clean(rb.get('goal'), 260)}`",
                f"- step: {rb.get('step')}; level=`{rb.get('level')}`; mode=`{rb.get('mode')}`; matched_by=`{rb.get('matched_by')}`",
                f"- back presses: {rb.get('back_presses')}; replayed actions: {rb.get('replayed_actions')}; latency: {float(rb.get('latency_ms') or 0.0):.1f} ms",
                f"- likely reasons: `{', '.join(analysis.get('likely_reasons') or []) or 'unknown'}`",
                "",
            ]
        )
        shot = _img(rb.get("root_screenshot"), "rollback root")
        if shot:
            lines.extend([shot, ""])
        fail_shot = _img(rb.get("failure_screenshot"), "rollback failure final")
        if fail_shot:
            lines.extend(["Final state after failed rollback:", "", fail_shot, ""])
        lines.extend(
            [
                "Replay actions:",
                "",
                "```json",
                json.dumps(rb.get("replay_action_types") or [], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    if not lines:
        lines.append("没有发现 rollback failure。")
        lines.append("")
    return lines


def _reasoning_case_sections(actions: list[dict[str, Any]], max_cases: int) -> list[str]:
    interesting = [
        row for row in actions
        if row.get("prompt_hint")
        or row.get("post_planning_exploration_speculative_results")
        or str(row.get("exploration_status") or "").startswith("skipped_transaction_unsafe")
        or bool(row.get("rollback_success") is False)
    ]
    if not interesting:
        interesting = list(actions)
    lines: list[str] = []
    for idx, row in enumerate(interesting[:max_cases], start=1):
        parsed = row.get("parsed_action") if isinstance(row.get("parsed_action"), dict) else {}
        lines.extend(
            [
                f"### Reasoning Case {idx}",
                "",
                f"- task instruction: `{_clean(row.get('goal'), 260)}`",
                f"- step: {row.get('step')}; prompt mode: `{row.get('prompt_mode')}`; executed action: `{_short_action(row)}`",
                f"- exploration status after planning: `{row.get('exploration_status')}`; rollback success: `{row.get('rollback_success')}`",
                "",
                "Injected exploration hint before reasoning:",
                "",
                "```text",
                _clean(row.get("prompt_hint") or "<none>", 1400),
                "```",
                "",
                "VLM raw output:",
                "",
                "```text",
                _clean(row.get("response"), 1400),
                "```",
                "",
                "Parsed action:",
                "",
                "```json",
                json.dumps(parsed, ensure_ascii=False, indent=2, default=str)[:1800],
                "```",
                "",
            ]
        )
        speculative = row.get("post_planning_exploration_speculative_results") or []
        if speculative:
            lines.extend(
                [
                    "Post-planning exploration summary prepared for future prompt:",
                    "",
                    "| rank | explored path | depth | prompt line |",
                    "| ---: | --- | ---: | --- |",
                ]
            )
            for item in speculative[:5]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"| {item.get('rank', '')} | `{_clean(item.get('path'), 120)}` | "
                    f"{item.get('depth_reached')} | {_clean(item.get('prompt_line'), 260)} |"
                )
            lines.append("")
        matched = row.get("matched_exploration_results") or []
        if matched:
            lines.extend(
                [
                    "Matched exploration results used in this prompt:",
                    "",
                    "| rank | next candidate | score | matched by |",
                    "| ---: | --- | ---: | --- |",
                ]
            )
            for item in matched[:5]:
                if not isinstance(item, dict):
                    continue
                cand = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
                lines.append(
                    f"| {item.get('rank', '')} | `{_clean(cand.get('label') or item.get('next_label'), 120)}` | "
                    f"{float(item.get('score') or 0.0):.3f} | `{item.get('matched_by')}` |"
                )
            lines.append("")
    if not lines:
        lines.append("没有可展示的 reasoning/action case。")
        lines.append("")
    return lines


def _write_report(
    report_path: Path,
    checkpoint_dir: Path | None,
    trace_root: Path,
    summary: dict[str, Any],
    charts: dict[str, Path],
    explorations: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    step_metrics: list[dict[str, Any]],
    step_metrics_path: Path,
    max_cases: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# AndroidWorld Exploration 完整实验报告",
        "",
        f"- 生成时间: `{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- checkpoint dir: `{checkpoint_dir.resolve() if checkpoint_dir else 'n/a'}`",
            f"- trace root: `{trace_root.resolve()}`",
            f"- per-step metrics: `{step_metrics_path.resolve()}`",
            "",
        "## 总览",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| total episodes | {summary['episodes']['total']} |",
        f"| complete episodes | {summary['episodes']['complete']} |",
        f"| exception failures | {summary['episodes']['exception_failures']} |",
        f"| success rate, complete only | {_pct(summary['episodes']['success_rate_complete_only'])} |",
        f"| success rate, all trials | {_pct(summary['episodes']['success_rate_all_trials'])} |",
        f"| avg episode length, complete | {summary['episodes']['avg_episode_length_complete']} |",
        f"| avg episode length, success | {summary['episodes']['avg_episode_length_success']} |",
            f"| reasoning steps | {summary['actions']['total_steps']} |",
            f"| exploration traces | {summary['exploration']['total_traces']} |",
            f"| prompt hint steps | {summary['actions']['prompt_hint_steps']} |",
            f"| evidence-only hint steps | {summary['actions'].get('prompt_hint_evidence_only_steps', 0)} |",
            f"| evaluable action-hint steps | {summary['actions'].get('prompt_hint_evaluable_steps', summary['actions']['prompt_hint_steps'])} |",
            f"| hint follow rate | {_pct(summary['actions']['prompt_hint_follow_rate'])} |",
            f"| rollback success rate | {_pct(summary['rollback']['success_rate'])} |",
        "",
        "## Diagnostic / Hard Validation",
        "",
        "| artifact | value |",
        "| --- | ---: |",
        f"| evidence_decisions rows | {(summary.get('diagnostics') or {}).get('evidence_decisions')} |",
        f"| prompt_hints rows | {(summary.get('diagnostics') or {}).get('prompt_hint_decisions')} |",
        f"| injected prompt hints | {(summary.get('diagnostics') or {}).get('prompt_hints_injected')} |",
        f"| step decoupling rows | {(summary.get('diagnostics') or {}).get('step_decoupling_status')} |",
        f"| decoupling invalid steps | {(summary.get('diagnostics') or {}).get('decoupled_invalid_steps')} |",
        f"| exploration_latency rows | {(summary.get('diagnostics') or {}).get('exploration_latency_rows')} |",
        f"| rollback_events rows | {(summary.get('diagnostics') or {}).get('rollback_events')} |",
        f"| rollback event failures | {(summary.get('diagnostics') or {}).get('rollback_event_failures')} |",
        f"| missing required files | `{(summary.get('diagnostics') or {}).get('missing_required_files')}` |",
        "",
        "## Prompt Injection Diagnosis",
        "",
        f"- prompt_hints rows: `{(summary.get('diagnostics') or {}).get('prompt_hint_decisions')}`",
        f"- injected hint count: `{(summary.get('diagnostics') or {}).get('prompt_hints_injected')}`",
        f"- rejected hint count: `{(summary.get('diagnostics') or {}).get('prompt_hints_rejected')}`",
        f"- evidence_decisions rows: `{(summary.get('diagnostics') or {}).get('evidence_decisions')}`",
        "",
        "如果 injected hint count 为 `0`，本报告不得把 success rate 变化归因于 prompt augmentation；变化只能来自 exploration/rollback side effects、timing variance、evaluator variance 或随机性。",
        "",
        "### Top rejection reasons",
        "",
        f"- prompt hint rejection reasons: `{(summary.get('diagnostics') or {}).get('prompt_hint_rejection_reasons')}`",
        f"- evidence rejection reasons: `{(summary.get('diagnostics') or {}).get('evidence_rejection_reasons')}`",
        f"- evidence type counts: `{(summary.get('diagnostics') or {}).get('evidence_type_counts')}`",
        "",
        "### Rejected evidence examples",
        "",
        "| step | branch | type | confidence | threshold | reason | candidate |",
        "| ---: | ---: | --- | ---: | ---: | --- | --- |",
        *[
            (
                f"| {row.get('step')} | {row.get('branch_id')} | `{_clean(row.get('final_evidence_type'), 40)}` | "
                f"{row.get('confidence')} | {row.get('threshold')} | `{_clean(row.get('rejected_reason'), 80)}` | "
                f"`{_clean(row.get('candidate_label'), 80)}` |"
            )
            for row in list((summary.get('diagnostics') or {}).get('rejected_examples') or [])[:8]
        ],
        "",
        "## Candidate Quality Filtering",
        "",
        f"- raw candidates: `{(summary.get('diagnostics') or {}).get('candidate_raw_total')}`",
        f"- filtered candidates: `{(summary.get('diagnostics') or {}).get('candidate_filtered_total')}`",
        f"- candidate operator counts: `{(summary.get('diagnostics') or {}).get('candidate_operator_counts')}`",
        f"- top filtered examples: `{(summary.get('diagnostics') or {}).get('candidate_quality_examples')}`",
        "",
        "## Search Strategy",
        "",
        "- strategy name: `Operator-Stratified Best-First Exploration`",
        "- root-level search: stratified breadth-first selection across operator groups.",
        "- branch continuation: gated depth-first continuation to depth 2.",
        "- ranking: best-first score.",
        "- score formula: `2.0*MissingSlotGain + 1.5*TaskEntityMatch + 1.0*OperatorPriority + 1.0*TaskProgress + 0.5*Novelty - 2.0*RiskPenalty - 1.0*RollbackCost - 1.0*WrongScreenRolePenalty - 0.5*RevisitPenalty`.",
        "- operators: `ListInspect`, `SearchPeek`, `FilterPeek`, `NavigationPeek`, `DetailPeek`, `StatsPeek`, `FormSchema`, `RiskBoundary`, `Other`.",
        "",
        "## Latency Breakdown",
        "",
        "| metric | value ms |",
        "| --- | ---: |",
        f"| avg step latency | {summary['actions'].get('avg_step_latency_ms')} |",
        f"| avg VLM latency | {summary['actions'].get('avg_vlm_latency_ms')} |",
        f"| avg exploration latency | {summary['actions'].get('avg_exploration_latency_ms')} |",
        f"| avg wait after VLM for exploration | {summary['actions'].get('avg_exploration_wait_after_vlm_ms')} |",
        f"| avg exploration state fetch | {summary['exploration'].get('avg_state_fetch_ms')} |",
        f"| avg exploration a11y dump | {summary['exploration'].get('avg_a11y_dump_ms')} |",
        f"| avg branch state fetch | {summary['exploration'].get('avg_branch_state_fetch_ms')} |",
        f"| avg branch a11y latency | {summary['exploration'].get('avg_branch_a11y_latency_ms')} |",
        f"| p50 exploration total | {((summary.get('diagnostic_latency_distribution') or {}).get('exploration_total_ms') or {}).get('median')} |",
        f"| p95 exploration total | {((summary.get('diagnostic_latency_distribution') or {}).get('exploration_total_ms') or {}).get('p95')} |",
        f"| p50 branch a11y | {((summary.get('diagnostic_latency_distribution') or {}).get('branch_a11y_ms') or {}).get('median')} |",
        f"| p95 branch a11y | {((summary.get('diagnostic_latency_distribution') or {}).get('branch_a11y_ms') or {}).get('p95')} |",
        f"| p50 branch state fetch | {((summary.get('diagnostic_latency_distribution') or {}).get('branch_state_fetch_ms') or {}).get('median')} |",
        f"| p95 branch state fetch | {((summary.get('diagnostic_latency_distribution') or {}).get('branch_state_fetch_ms') or {}).get('p95')} |",
        f"| p50 rollback verify | {((summary.get('diagnostic_latency_distribution') or {}).get('rollback_verify_ms') or {}).get('median')} |",
        f"| p95 rollback verify | {((summary.get('diagnostic_latency_distribution') or {}).get('rollback_verify_ms') or {}).get('p95')} |",
        "",
        "## 统计图表",
        "",
    ]
    for key in ("exploration_status", "rollback", "rates", "tasks"):
        if key in charts:
            lines.append(f"![{key}]({charts[key].resolve()})")
            lines.append("")
    for key in ("episode_success", "episode_steps_avg", "episode_steps_by_task"):
        if key in charts:
            lines.append(f"![{key}]({charts[key].resolve()})")
            lines.append("")
    for key in ("baseline_success", "baseline_steps", "baseline_step_delta"):
        if key in charts:
            lines.append(f"![{key}]({charts[key].resolve()})")
            lines.append("")

    baseline = summary.get("baseline_compare") if isinstance(summary.get("baseline_compare"), dict) else {}
    if baseline and baseline.get("baseline_file_loaded"):
        lines.extend(
            [
                "## Baseline 对比",
                "",
                f"- baseline task 数: `{baseline.get('baseline_task_count')}`",
                f"- 与本次实验匹配的 task 数: `{baseline.get('matched_task_count')}`",
                f"- baseline 平均成功率: `{_pct(baseline.get('avg_baseline_success_rate'))}`",
                f"- exploration 平均成功率: `{_pct(baseline.get('avg_exploration_success_rate'))}`",
                f"- baseline 平均步数: `{baseline.get('avg_baseline_episode_length')}`",
                f"- exploration 平均步数: `{baseline.get('avg_exploration_episode_length')}`",
                "",
                "| task | baseline success | exploration success | baseline steps | exploration steps | step delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in list(baseline.get("rows") or [])[:40]:
            lines.append(
                f"| `{_clean(row.get('task'), 80)}` | {_pct(row.get('baseline_success_rate'))} | "
                f"{_pct(row.get('exploration_success'))} | {row.get('baseline_episode_length')} | "
                f"{row.get('exploration_episode_length')} | {row.get('step_delta')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Exploration 统计",
            "",
            f"- exploration status: `{summary['exploration']['status_counts']}`",
            f"- 可用于下一步 prompt 的 exploration: {_fmt_rate(summary['exploration']['available_for_next_prompt'], summary['exploration']['total_traces'])}",
            f"- depth>=2 observation: {_fmt_rate(summary['exploration']['depth2_observations'], summary['exploration']['total_observations'])}",
            f"- 平均每步候选元素数: `{summary['actions']['avg_exploration_candidates_per_step']}`",
            f"- 平均每步 selected targets: `{summary['actions']['avg_selected_targets_per_step']}`",
            f"- 平均每步 observations: `{summary['actions']['avg_observations_per_step']}`",
            "",
            "## State Match 与 Hint 命中率",
            "",
            f"- 有 pending exploration 可匹配的 reasoning step: {summary['state_match']['pending_steps']}",
            f"- state match step hit rate: {_fmt_rate(summary['state_match']['matched_steps'], summary['state_match']['pending_steps'])}",
            f"- state match candidate hit rate: {_fmt_rate(summary['state_match']['matched_candidate_count'], summary['state_match']['candidate_match_count'])}",
            f"- hint injected steps: {summary['actions']['prompt_hint_steps']}",
            f"- evidence-only hint steps: {summary['actions'].get('prompt_hint_evidence_only_steps', 0)}",
            f"- evaluable action-hint steps: {summary['actions'].get('prompt_hint_evaluable_steps', summary['actions']['prompt_hint_steps'])}",
            f"- hint followed: {_fmt_rate(summary['actions']['prompt_hint_followed'], summary['actions'].get('prompt_hint_evaluable_steps', summary['actions']['prompt_hint_steps']))}",
            "",
            "这里的 hint followed 只统计可评估的 action-hint：VLM 在注入 state-aligned hint 后，下一步实际 click 落在 exploration 建议的 next candidate bbox 内/距离 candidate center 小于 120 px，或实际 TYPE 文本与 typed speculative candidate 完全一致。evidence-only hint 只提供页面事实，不进入该命中率分母。",
            "",
            "## Per-step Reasoning / Exploration Trace",
            "",
            f"完整逐步记录已保存到 `{step_metrics_path.resolve()}`。下面展示前 30 行，用于检查每次 reasoning 前是否有 pending exploration、匹配到了多少结果、post-planning 阶段又探索了多少分支。",
            "",
            "| task | step | pre pending | pre matched | hint | hint hit | post status | post candidates | selected | obs | action |",
            "| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in step_metrics[:30]:
        lines.append(
            f"| `{_clean(row.get('task_dir'), 48)}` | {row.get('step')} | {row.get('pre_reasoning_pending_traces')} | "
            f"{row.get('pre_reasoning_matched_results')} | {row.get('hint_injected')} | {row.get('hint_followed')} | "
            f"`{row.get('post_planning_exploration_status')}` | {row.get('post_planning_candidate_count')} | "
            f"{row.get('post_planning_selected_targets')} | {row.get('post_planning_observations')} | `{row.get('main_action')}` |"
        )
    lines.extend(
        [
            "",
            "## Rollback 统计",
            "",
            f"- rollback total: {summary['rollback']['total']}",
            f"- rollback success: {_fmt_rate(summary['rollback']['success'], summary['rollback']['total'])}",
            f"- level2 trigger reasons: `{(summary.get('diagnostics') or {}).get('rollback_level2_triggers')}`",
            "",
            "| level | total | success | success rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for level, row in sorted(summary["rollback"]["by_level"].items()):
        lines.append(f"| {level} | {row['total']} | {row['success']} | {_pct(_rate(row['success'], row['total']))} |")

    top_tasks = sorted(
        summary["task_stats"].items(),
        key=lambda kv: int(kv[1].get("exploration_attempts") or 0),
        reverse=True,
    )[:20]
    lines.extend(
        [
            "",
            "## Per-task Exploration Top 20",
            "",
            "| task dir | reasoning steps | exploration attempts | observations | hint steps | hint followed |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for task, row in top_tasks:
        lines.append(
            f"| `{_clean(task, 80)}` | {row.get('steps', 0)} | {row.get('exploration_attempts', 0)} | "
            f"{row.get('observations', 0)} | {row.get('hint_steps', 0)} | {row.get('hint_followed', 0)} |"
        )

    lines.extend(["", "## 具体 Exploration Traces", ""])
    lines.extend(_exploration_case_sections(explorations, max_cases=max_cases))
    lines.extend(["", "## Reasoning / Hint / Action 中间结果", ""])
    lines.extend(_reasoning_case_sections(actions, max_cases=max_cases))
    lines.extend(["", "## 具体 Hint Cases", ""])
    lines.extend(_hint_case_sections(summary, max_cases=max_cases))
    lines.extend(["", "## Rollback Fail Cases", ""])
    lines.extend(_failure_sections(summary, max_cases=max_cases))
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 完整实验中应同时看 `state match rate` 和 `hint follow rate`。前者表示 speculative t+2 信息是否能对齐到 t+1 页面，后者表示 VLM 是否真正采用了该信息。",
            "- `available_for_next_prompt` 很高但 `state match rate` 低，通常说明 exploration 做出来了，但 t+1 页面与 depth1 页面不一致，需要收紧匹配或改变 exploration 目标。",
            "- `state match rate` 高但 `hint follow rate` 低，通常说明 prompt 里信息格式、候选元素选择或任务相关性不足，需要优化 distill 后的 hint 文本。",
            "- rollback failure 必须单独看，尤其是 level2 replay failure；这类失败会污染主进程状态，当前代码已经在 rollback_failed 时 suppress 主 action。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def generate_report(
    checkpoint_dir: Path | None,
    trace_root: Path,
    report_dir: Path,
    max_cases: int,
    baseline_table: Path | None = None,
) -> Path:
    episodes = _load_episodes(checkpoint_dir)
    explorations = _load_trace_family(trace_root, "exploration_trace.jsonl")
    matches = _load_trace_family(trace_root, "exploration_match_trace.jsonl")
    actions = _load_trace_family(trace_root, "action.jsonl")
    summary = _summarize(episodes, explorations, matches, actions)
    diagnostics = _load_diagnostic_families(trace_root)
    summary["diagnostics"] = diagnostics.get("summary", {})
    summary["diagnostic_latency_distribution"] = diagnostics.get("latency_distribution", {})
    summary["runtime_configs"] = (diagnostics.get("rows") or {}).get("runtime_config.json", [])
    _attach_baseline(summary, _load_baseline_table(baseline_table))
    report_dir.mkdir(parents=True, exist_ok=True)
    charts = _make_charts(summary, report_dir)
    step_metrics = _build_step_metrics(actions, matches)
    step_metrics_path = report_dir / "step_metrics.csv"
    _write_step_metrics(step_metrics_path, step_metrics)

    summary_out = dict(summary)
    for key in ("_hint_actions", "_hint_followed", "_hint_missed", "_rollbacks"):
        summary_out.pop(key, None)
    (report_dir / "summary.json").write_text(
        json.dumps(summary_out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report_path = report_dir / "exploration_experiment_report_cn.md"
    _write_report(
        report_path,
        checkpoint_dir,
        trace_root,
        summary,
        charts,
        explorations,
        actions,
        step_metrics,
        step_metrics_path,
        max_cases,
    )
    return report_path


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def _run_androidworld(args: argparse.Namespace, run_dir: Path, trace_root: Path, log_path: Path) -> int:
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if trace_root.exists() and args.clean_trace_root:
        shutil.rmtree(trace_root)
    trace_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "run.py"),
        f"--agent_name={args.agent_name}",
        f"--suite_family={args.suite_family}",
        f"--n_task_combinations={args.n_task_combinations}",
        f"--task_random_seed={args.task_random_seed}",
        f"--image_downsample_scale={args.image_downsample_scale}",
        f"--adb_path={args.adb_path}",
        f"--console_port={args.console_port}",
        f"--checkpoint_dir={checkpoint_dir}",
        f"--output_path={run_dir}",
    ]
    if args.tasks:
        cmd.append(f"--tasks={args.tasks}")
    if args.fixed_task_seed:
        cmd.append("--fixed_task_seed")
    else:
        cmd.append("--nofixed_task_seed")
    if args.perform_emulator_setup:
        cmd.append("--perform_emulator_setup")

    env = os.environ.copy()
    env["ANDROID_WORLD_EXPLORATION_TRACE_ROOT"] = str(trace_root)
    env["ANDROID_WORLD_LIGHT_EXPLORE_ENABLE"] = "1" if args.explore_enable else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_MAX_RUNS"] = str(args.explore_max_runs)
    env["ANDROID_WORLD_LIGHT_EXPLORE_MAX_STEP"] = str(args.explore_max_step)
    env["ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_BUDGET"] = str(args.explore_branch_budget)
    env["ANDROID_WORLD_LIGHT_EXPLORE_BRANCH_DEPTH"] = str(args.explore_branch_depth)
    env["ANDROID_WORLD_LIGHT_EXPLORE_MIN_ATTEMPTS_PER_STEP"] = str(args.explore_min_attempts_per_step)
    env["ANDROID_WORLD_EXPLORATION_TIMING"] = str(args.exploration_timing)
    env["ANDROID_WORLD_DECOUPLED_EXPLORATION"] = "1" if args.decoupled_exploration else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_USE_CURRENT_ACTION"] = "1" if args.explore_use_current_action else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_USE_PLANNING_TEXT"] = "1" if args.explore_use_planning_text else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_SHADOW_ONLY"] = "1" if args.explore_shadow_only else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_RELAXED_DIAGNOSTIC_INJECTION"] = (
        "1" if args.explore_relaxed_diagnostic_injection else "0"
    )
    env["ANDROID_WORLD_LIGHT_EXPLORE_UPPER_BOUND_EVIDENCE"] = (
        "1" if args.explore_upper_bound_evidence else "0"
    )
    env["ANDROID_WORLD_LIGHT_EXPLORE_SUMMARY_ALL_DEBUG"] = (
        "1" if args.explore_summary_all_debug else "0"
    )
    env["ANDROID_WORLD_TRACE_SCREENSHOT_MODE"] = str(args.trace_screenshot_mode)
    env["ANDROID_WORLD_LIGHT_EXPLORE_QUALITY_FILTERS"] = "1" if args.explore_quality_filters else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_DIAGNOSTIC_ARTIFACTS"] = (
        "1" if args.explore_diagnostic_artifacts else "0"
    )
    env["ANDROID_WORLD_LIGHT_EXPLORE_BACK_LIMIT"] = str(args.explore_back_limit)
    env["ANDROID_WORLD_LIGHT_EXPLORE_REPLAY_MAX_ACTIONS"] = str(args.explore_replay_max_actions)
    env["ANDROID_WORLD_LIGHT_EXPLORE_PLANNED_ONLY"] = "1" if args.explore_planned_only else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_DECOUPLE_PLANNED"] = "1" if args.explore_decouple_planned else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_PARALLEL_VLM"] = "1" if args.explore_parallel_vlm else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_PARALLEL_LOOKAHEAD"] = "1" if args.explore_parallel_lookahead else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_FALLBACK_SAFE_CANDIDATES"] = (
        "1" if args.explore_fallback_safe_candidates else "0"
    )
    env["ANDROID_WORLD_LIGHT_EXPLORE_SAFE_CLICK_ONLY"] = "1" if args.explore_safe_click_only else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_SKIP_LAUNCHER"] = "1" if args.explore_skip_launcher else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_FILTER_LAUNCHER_RELEVANCE"] = (
        "1" if args.explore_filter_launcher_relevance else "0"
    )
    env["ANDROID_WORLD_LIGHT_EXPLORE_DIAGNOSTIC_FULL"] = "1" if args.explore_diagnostic_full else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_FORCE_EVERY_STEP"] = "1" if args.explore_force_every_step else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_FAST_MODE"] = "1" if args.explore_fast_mode else "0"
    fast_state = bool(args.explore_fast_mode) if args.explore_fast_state is None else bool(args.explore_fast_state)
    env["ANDROID_WORLD_LIGHT_EXPLORE_FAST_STATE"] = "1" if fast_state else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_SAVE_SCREENSHOTS"] = "1" if args.explore_save_screenshots else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_TRANSACTION_SAFE"] = "1" if args.explore_transaction_safe else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_ACTION_SETTLE_S"] = str(args.explore_action_settle_s)
    env["ANDROID_WORLD_LIGHT_EXPLORE_STRATEGY"] = str(args.explore_strategy)
    env["ANDROID_WORLD_LIGHT_EXPLORE_SKIP_DESTRUCTIVE_GOALS"] = (
        "1" if args.explore_skip_destructive_goals else "0"
    )
    env["ANDROID_WORLD_LIGHT_EXPLORE_DISABLE_PROMPT_INJECTION"] = (
        "1" if args.explore_disable_prompt_injection else "0"
    )
    env["ANDROID_WORLD_LIGHT_EXPLORE_HINT_POLICY"] = str(args.explore_hint_policy)
    env["ANDROID_WORLD_LIGHT_EXPLORE_SEARCH_POLICY"] = str(args.explore_search_policy)
    env["ANDROID_WORLD_LIGHT_EXPLORE_SEARCH_STRATEGY"] = str(args.explore_search_strategy)
    env["ANDROID_WORLD_LIGHT_EXPLORE_ROLLBACK_POLICY"] = str(args.explore_rollback_policy)
    env["ANDROID_WORLD_LIGHT_EXPLORE_FIXED_FRAMEWORK"] = "1" if args.explore_fixed_framework else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_VARIANT"] = str(args.explore_variant)
    env["ANDROID_WORLD_LIGHT_EXPLORE_SAFE_MCTS"] = "1" if args.explore_safe_mcts else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_ANSWER_EXTRACTORS"] = "1" if args.explore_answer_extractors else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_SLOT_COMPLETE"] = "1" if args.explore_slot_complete else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_NO_ACTION_HINT"] = "1" if args.explore_no_action_hint else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_SLOT_POLICY_SWITCHER"] = "1" if args.explore_slot_policy_switcher else "0"
    env["ANDROID_WORLD_LIGHT_EXPLORE_ENABLE_T2_LOOKAHEAD"] = (
        "1" if args.explore_enable_t2_lookahead else "0"
    )
    env["ANDROID_WORLD_LIGHT_EXPLORE_LIGHTWEIGHT_A11Y_TRACE"] = (
        "1" if args.explore_lightweight_a11y_trace else "0"
    )
    env["ANDROID_WORLD_LIGHT_EXPLORE_TRACE_A11Y_LIMIT"] = str(args.explore_trace_a11y_limit)
    env["ANDROID_WORLD_BANDIT_T2_BUDGET"] = str(args.explore_t2_budget)
    env["ANDROID_WORLD_BANDIT_STALLED_T2_BUDGET"] = str(args.explore_stalled_t2_budget)
    env["ANDROID_WORLD_BANDIT_SECONDARY_POOL"] = str(args.explore_secondary_pool)
    env["ANDROID_WORLD_BANDIT_UCB_WEIGHT"] = str(args.explore_ucb_weight)
    env["ANDROID_WORLD_BANDIT_FOLLOW_WEIGHT"] = str(args.explore_follow_weight)
    env["ANDROID_WORLD_BANDIT_NEGATIVE_CONTEXT"] = "1" if args.explore_negative_context else "0"
    env["ANDROID_WORLD_T2_MODE"] = str(args.t2_mode)
    env["ANDROID_WORLD_T2_STATE_MODE"] = str(args.t2_state_mode)
    env["ANDROID_WORLD_T2_ALLOW_SAFE_SEARCH_INPUT"] = "1" if args.t2_allow_safe_search_input else "0"
    env["ANDROID_WORLD_T2_CONFIDENCE_THRESHOLD"] = str(args.t2_confidence_threshold)
    env["ANDROID_WORLD_T2_MIN_TOP1_SCORE"] = str(args.t2_min_top1_score)
    env["ANDROID_WORLD_T2_MIN_SCORE_MARGIN"] = str(args.t2_min_score_margin)
    env["ANDROID_WORLD_LATENCY_PROFILE"] = "1" if args.latency_profile else "0"
    if args._selected_a11y_method == "uiautomator":
        env["ANDROID_WORLD_A11Y_METHOD"] = "uiautomator"
    elif args._selected_a11y_method == "fast_provider":
        env["ANDROID_WORLD_A11Y_METHOD"] = "fast_provider"
    else:
        env.pop("ANDROID_WORLD_A11Y_METHOD", None)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.write(f"ANDROID_WORLD_EXPLORATION_TRACE_ROOT={trace_root}\n\n")
        log_file.write(f"ANDROID_WORLD_A11Y_METHOD={env.get('ANDROID_WORLD_A11Y_METHOD', 'grpc')}\n\n")
        for key in sorted(
            k
            for k in env
            if k.startswith("ANDROID_WORLD_LIGHT_EXPLORE_")
            or k.startswith("ANDROID_WORLD_BANDIT_")
            or k.startswith("ANDROID_WORLD_T2_")
            or k == "ANDROID_WORLD_EXPLORATION_TIMING"
            or k == "ANDROID_WORLD_DECOUPLED_EXPLORATION"
            or k == "ANDROID_WORLD_TRACE_SCREENSHOT_MODE"
            or k == "ANDROID_WORLD_LATENCY_PROFILE"
        ):
            log_file.write(f"{key}={env[key]}\n")
        log_file.write("\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AndroidWorld exploration experiment and generate Chinese Markdown report."
    )
    parser.add_argument("--run", action=argparse.BooleanOptionalAction, default=True, help="Run AndroidWorld before reporting.")
    parser.add_argument("--agent_name", default=DEFAULT_AGENT)
    parser.add_argument("--suite_family", default="android_world")
    parser.add_argument("--tasks", default="", help="Comma-separated task list. Empty means full suite.")
    parser.add_argument("--n_task_combinations", type=int, default=1)
    parser.add_argument("--task_random_seed", type=int, default=30)
    parser.add_argument("--fixed_task_seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_downsample_scale", type=float, default=1.0)
    parser.add_argument("--adb_path", default=DEFAULT_ADB_PATH)
    parser.add_argument("--console_port", type=int, default=5554)
    parser.add_argument(
        "--a11y_method",
        default="auto",
        choices=("auto", "grpc", "uiautomator", "fast_provider"),
        help="auto tries gRPC first and falls back to uiautomator dump; fast_provider uses the lightweight a11y app.",
    )
    parser.add_argument("--a11y_preflight_timeout", type=float, default=60.0)
    parser.add_argument("--perform_emulator_setup", action="store_true")
    parser.add_argument("--explore_enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explore_max_runs", type=int, default=3)
    parser.add_argument("--explore_max_step", type=int, default=8)
    parser.add_argument("--explore_branch_budget", type=int, default=1)
    parser.add_argument("--explore_branch_depth", type=int, default=2)
    parser.add_argument("--explore_min_attempts_per_step", type=int, default=0)
    parser.add_argument(
        "--exploration_timing",
        choices=("pre_reasoning", "pre_reasoning_decoupled", "parallel_shadow", "post_reasoning_decoupled", "invalid_post_planned_action"),
        default="parallel_shadow",
    )
    parser.add_argument("--decoupled_exploration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explore_use_current_action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_use_planning_text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_shadow_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_relaxed_diagnostic_injection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_upper_bound_evidence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_summary_all_debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace_screenshot_mode", default="failure+level2+depth2+injected+sampled")
    parser.add_argument("--explore_quality_filters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explore_diagnostic_artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explore_back_limit", type=int, default=3)
    parser.add_argument("--explore_replay_max_actions", type=int, default=3)
    parser.add_argument("--explore_planned_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_decouple_planned", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explore_parallel_vlm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explore_parallel_lookahead", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--explore_fallback_safe_candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When the planned action is unsafe/unanchored, still explore safe non-planned evidence candidates.",
    )
    parser.add_argument("--explore_safe_click_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explore_skip_launcher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explore_filter_launcher_relevance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explore_skip_destructive_goals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--explore_diagnostic_full",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force exploration every non-terminal step and bypass safety/window/budget gates for trace diagnostics.",
    )
    parser.add_argument(
        "--explore_force_every_step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run exploration after every reasoning step, including terminal/answer steps, and keep trying after rollback failures.",
    )
    parser.add_argument(
        "--explore_fast_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use direct ADB taps/back/home for speculative exploration to avoid pre-action a11y dumps.",
    )
    parser.add_argument(
        "--explore_fast_state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use single state snapshots instead of multi-dump stable-state checks during exploration.",
    )
    parser.add_argument(
        "--explore_save_screenshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save exploration screenshots. Disable only for timing-only runs.",
    )
    parser.add_argument(
        "--explore_transaction_safe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not execute speculative branches that look like irreversible commit/destructive actions.",
    )
    parser.add_argument("--explore_action_settle_s", type=float, default=0.25)
    parser.add_argument("--explore_strategy", choices=("dfs", "bfs"), default="dfs")
    parser.add_argument("--explore_disable_prompt_injection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_hint_policy", choices=("current", "strict"), default="current")
    parser.add_argument("--explore_search_policy", choices=("current", "operator", "task_gate"), default="current")
    parser.add_argument(
        "--explore_search_strategy",
        choices=("greedy", "stratified_bfs", "iddfs", "best_first", "beam", "mcts"),
        default="greedy",
    )
    parser.add_argument("--explore_rollback_policy", choices=("current", "improved"), default="current")
    parser.add_argument("--explore_fixed_framework", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_variant", default="")
    parser.add_argument("--explore_safe_mcts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_answer_extractors", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_slot_complete", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_no_action_hint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_slot_policy_switcher", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_enable_t2_lookahead", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explore_lightweight_a11y_trace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--explore_trace_a11y_limit", type=int, default=80)
    parser.add_argument("--t2_mode", choices=("off", "shadow", "active_safe", "direct_answer_shadow"), default="off")
    parser.add_argument("--t2_state_mode", default="FULL_A11Y_CONTROL")
    parser.add_argument("--t2_allow_safe_search_input", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--t2_confidence_threshold", type=float, default=0.85)
    parser.add_argument("--t2_min_top1_score", type=float, default=7.0)
    parser.add_argument("--t2_min_score_margin", type=float, default=2.0)
    parser.add_argument("--latency_profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--explore_t2_budget", type=int, default=3)
    parser.add_argument("--explore_stalled_t2_budget", type=int, default=4)
    parser.add_argument("--explore_secondary_pool", type=int, default=8)
    parser.add_argument("--explore_ucb_weight", type=float, default=0.18)
    parser.add_argument("--explore_follow_weight", type=float, default=0.35)
    parser.add_argument("--explore_negative_context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--experiment_root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--checkpoint_dir", default="", help="Existing or target checkpoint dir.")
    parser.add_argument("--trace_root", default="", help="Existing or target exploration trace root.")
    parser.add_argument("--report_dir", default="", help="Output report directory.")
    parser.add_argument(
        "--baseline_table",
        default=str(REPO_ROOT / "results" / "4B_2.txt"),
        help="Optional AndroidWorld baseline summary table for report comparison.",
    )
    parser.add_argument("--clean_trace_root", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_cases", type=int, default=8)
    parser.add_argument("--report_on_failure", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    run_dir = experiment_root / f"run_{_timestamp()}"
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else run_dir / "checkpoints"
    trace_root = Path(args.trace_root).expanduser().resolve() if args.trace_root else run_dir / "traces"
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else run_dir / "report"

    exit_code = 0
    if args.run:
        print(f"[experiment] run_dir={run_dir}")
        print(f"[experiment] checkpoint_dir={checkpoint_dir}")
        print(f"[experiment] trace_root={trace_root}")
        print("[experiment] running a11y preflight...")
        args._selected_a11y_method = _select_a11y_method(
            args,
            log_path=run_dir / "a11y_preflight.log",
        )
        print(f"[experiment] selected a11y method: {args._selected_a11y_method}")
        exit_code = _run_androidworld(args, run_dir, trace_root, run_dir / "androidworld_run.log")
        if exit_code != 0 and not args.report_on_failure:
            return exit_code
    else:
        print(f"[experiment] analyze-only checkpoint_dir={checkpoint_dir}")
        print(f"[experiment] analyze-only trace_root={trace_root}")

    report_path = generate_report(
        checkpoint_dir=checkpoint_dir,
        trace_root=trace_root,
        report_dir=report_dir,
        max_cases=args.max_cases,
        baseline_table=Path(args.baseline_table).expanduser().resolve() if args.baseline_table else None,
    )
    print(f"[experiment] report={report_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
