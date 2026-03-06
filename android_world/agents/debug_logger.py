"""Lightweight structured debug logger for explorer agents."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

import numpy as np
from PIL import Image


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _safe_name(text: str, max_len: int = 72) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text or "").strip())
    value = re.sub(r"_+", "_", value).strip("._")
    if not value:
        value = "task"
    return value[: max(8, int(max_len))]


class DebugLogger:
    """Per-run JSONL logger with cheap aggregation for debugging."""

    def __init__(
        self,
        enabled: bool = True,
        log_root: str = "logs",
        save_debug_screenshots: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.log_root = str(log_root or "logs").strip() or "logs"
        self.save_debug_screenshots = bool(save_debug_screenshots)
        self.run_id: str | None = None
        self.run_dir: str | None = None
        self.screens_dir: str | None = None
        self._lock = threading.Lock()
        self._step_count = 0
        self._sum_tau_vlm = 0.0
        self._sum_tau_explore = 0.0
        self._sum_tau_total = 0.0
        self._sum_hint_hit_rate = 0.0
        self._parse_ok = 0
        self._fallback_used = 0
        self._rollback_l1_attempted = 0
        self._rollback_l1_success = 0
        self._rollback_l2_used = 0
        self._rollback_l2_success = 0
        if self.enabled:
            os.makedirs(self.log_root, exist_ok=True)

    def _append_jsonl(self, path: str, payload: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_safe_json(payload), ensure_ascii=False))
            f.write("\n")

    def _write_json(self, path: str, payload: dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_safe_json(payload), f, ensure_ascii=False, indent=2)

    def start_run(self, meta: dict[str, Any]) -> str | None:
        if not self.enabled:
            return None
        task_name = _safe_name(str(meta.get("task_name") or "task"))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{stamp}_{task_name}"
        run_dir = os.path.join(self.log_root, run_id)
        os.makedirs(run_dir, exist_ok=True)
        screens_dir = os.path.join(run_dir, "screens")
        if self.save_debug_screenshots:
            os.makedirs(screens_dir, exist_ok=True)
        with self._lock:
            self.run_id = run_id
            self.run_dir = run_dir
            self.screens_dir = screens_dir if self.save_debug_screenshots else None
            self._step_count = 0
            self._sum_tau_vlm = 0.0
            self._sum_tau_explore = 0.0
            self._sum_tau_total = 0.0
            self._sum_hint_hit_rate = 0.0
            self._parse_ok = 0
            self._fallback_used = 0
            self._rollback_l1_attempted = 0
            self._rollback_l1_success = 0
            self._rollback_l2_used = 0
            self._rollback_l2_success = 0
            self._write_json(
                os.path.join(run_dir, "meta.json"),
                {
                    "run_id": run_id,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    **dict(meta or {}),
                },
            )
        return run_id

    def save_screen(self, name: str, pixels: np.ndarray | None) -> str | None:
        if not self.enabled or not self.save_debug_screenshots:
            return None
        if self.screens_dir is None or pixels is None:
            return None
        file_name = f"{_safe_name(name, max_len=96)}.png"
        path = os.path.join(self.screens_dir, file_name)
        try:
            Image.fromarray(np.array(pixels, copy=True)).save(path)
            return path
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def log_event(self, event_dict: dict[str, Any]) -> None:
        if not self.enabled or not self.run_dir:
            return
        payload = dict(event_dict or {})
        payload.setdefault("ts", time.time())
        with self._lock:
            self._append_jsonl(os.path.join(self.run_dir, "events.jsonl"), payload)

    def log_step(self, step_dict: dict[str, Any]) -> None:
        if not self.enabled or not self.run_dir:
            return
        payload = dict(step_dict or {})
        payload.setdefault("ts_logged", time.time())
        with self._lock:
            self._append_jsonl(os.path.join(self.run_dir, "steps.jsonl"), payload)
            self._step_count += 1
            tau_vlm = float(payload.get("latencies", {}).get("tau_vlm") or 0.0)
            tau_explore = float(payload.get("latencies", {}).get("tau_explore") or 0.0)
            tau_total = float(payload.get("latencies", {}).get("tau_total") or 0.0)
            hint_hit_rate = float(payload.get("hint_hit_rate") or 0.0)
            self._sum_tau_vlm += tau_vlm
            self._sum_tau_explore += tau_explore
            self._sum_tau_total += tau_total
            self._sum_hint_hit_rate += hint_hit_rate
            if bool(payload.get("parse_ok")):
                self._parse_ok += 1
            if bool(payload.get("fallback_used")):
                self._fallback_used += 1
            if bool(payload.get("rollback_level1_attempted")):
                self._rollback_l1_attempted += 1
            if bool(payload.get("rollback_level1_success")):
                self._rollback_l1_success += 1
            if bool(payload.get("rollback_level2_used")):
                self._rollback_l2_used += 1
            if bool(payload.get("rollback_level2_success")):
                self._rollback_l2_success += 1

    def finalize(self, summary_dict: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled or not self.run_dir:
            return dict(summary_dict or {})
        with self._lock:
            steps = max(1, int(self._step_count))
            l1_attempted = max(1, int(self._rollback_l1_attempted))
            l2_used = max(1, int(self._rollback_l2_used))
            summary = {
                "run_id": self.run_id,
                "total_steps": int(self._step_count),
                "rollback_level1_success_rate": float(self._rollback_l1_success) / float(l1_attempted),
                "rollback_level2_rate": float(self._rollback_l2_used) / float(steps),
                "rollback_level2_success_rate": float(self._rollback_l2_success) / float(l2_used),
                "parse_success_rate": float(self._parse_ok) / float(steps),
                "fallback_rate": float(self._fallback_used) / float(steps),
                "avg_hint_hit_rate": float(self._sum_hint_hit_rate) / float(steps),
                "avg_latencies": {
                    "tau_vlm": float(self._sum_tau_vlm) / float(steps),
                    "tau_explore": float(self._sum_tau_explore) / float(steps),
                    "tau_total": float(self._sum_tau_total) / float(steps),
                },
            }
            summary.update(dict(summary_dict or {}))
            self._write_json(os.path.join(self.run_dir, "summary.json"), summary)
            return summary
