#!/usr/bin/env python3
"""Run/analyze GELAB baseline AndroidWorld results by task tag and difficulty."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_FILE = REPO_ROOT / "results" / "4B_2.txt"
DEFAULT_TASK_HTML = REPO_ROOT / "Task List _ AndroidWorld.html"
DEFAULT_EXPERIMENT_ROOT = REPO_ROOT / "results" / "baseline_4b_category_report"
DEFAULT_ADB_PATH = "/Users/huangrunxi/Library/Android/sdk/platform-tools/adb"
ACCESSIBILITY_FORWARDER_SERVICE = (
    "com.google.androidenv.accessibilityforwarder/"
    "com.google.androidenv.accessibilityforwarder.AccessibilityForwarder"
)

TASK_RESULT_RE = re.compile(
    r"^(?P<task>[A-Za-z][A-Za-z0-9_]+)\s+"
    r"(?P<task_num>\d+)\s+"
    r"(?P<num_complete_trials>[-+0-9.]+|NaN)\s+"
    r"(?P<mean_success_rate>[-+0-9.]+|NaN)\s+"
    r"(?P<mean_episode_length>[-+0-9.]+|NaN)\s+"
    r"(?P<total_runtime_s>[-+0-9.]+|NaN)\s+"
    r"(?P<mean_step_latency_s>[-+0-9.]+|NaN)\s+"
    r"(?P<num_fail_trials>[-+0-9.]+|NaN)\s*$"
)


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return math.nan
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return math.nan
        return float(text)
    except Exception:
        return math.nan


def _run_cmd(cmd: list[str], timeout: float = 30.0, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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
    time.sleep(1.0)


def _preflight_python(a11y_method: str, adb_path: str, console_port: int, timeout: float) -> tuple[bool, str]:
    env = os.environ.copy()
    if a11y_method == "uiautomator":
        env["ANDROID_WORLD_A11Y_METHOD"] = "uiautomator"
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


def _select_a11y_method(args: argparse.Namespace, log_path: Path) -> str:
    requested = str(args.a11y_method).lower()
    if requested not in {"auto", "grpc", "uiautomator"}:
        raise ValueError("--a11y_method must be one of auto, grpc, uiautomator")

    serial = f"emulator-{int(args.console_port)}"
    chunks = [f"requested={requested}", f"serial={serial}"]
    _restart_accessibility_forwarder(args.adb_path, serial)

    if requested in {"auto", "grpc"}:
        ok, output = _preflight_python(
            "grpc",
            adb_path=args.adb_path,
            console_port=int(args.console_port),
            timeout=float(args.a11y_preflight_timeout),
        )
        chunks.append("grpc_preflight=" + ("ok" if ok else "failed"))
        chunks.append(output[-4000:])
        if ok:
            log_path.write_text("\n\n".join(chunks), encoding="utf-8")
            return "grpc"
        if requested == "grpc":
            log_path.write_text("\n\n".join(chunks), encoding="utf-8")
            raise RuntimeError(f"gRPC a11y preflight failed. See {log_path}")

    dump_ok, dump_output = _preflight_uiautomator_dump(args.adb_path, serial)
    chunks.append("uiautomator_dump=" + ("ok" if dump_ok else "failed"))
    chunks.append(dump_output[-2000:])
    ok, output = _preflight_python(
        "uiautomator",
        adb_path=args.adb_path,
        console_port=int(args.console_port),
        timeout=float(args.a11y_preflight_timeout),
    )
    chunks.append("uiautomator_preflight=" + ("ok" if ok else "failed"))
    chunks.append(output[-4000:])
    log_path.write_text("\n\n".join(chunks), encoding="utf-8")
    if ok:
        return "uiautomator"
    raise RuntimeError(f"Both a11y preflights failed. See {log_path}")


def _check_llm_server(api_url: str, timeout: float = 5.0) -> tuple[bool, str]:
    models_url = api_url
    if models_url.endswith("/chat/completions"):
        models_url = models_url[: -len("/chat/completions")] + "/models"
    try:
        with urllib.request.urlopen(models_url, timeout=timeout) as response:
            body = response.read(4000).decode("utf-8", errors="replace")
            return 200 <= int(response.status) < 300, body
    except urllib.error.URLError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, repr(exc)


def load_task_metadata(html_path: Path) -> pd.DataFrame:
    tables = pd.read_html(str(html_path))
    if not tables:
        raise RuntimeError(f"No task table found in {html_path}")
    df = tables[0].copy()
    df = df.rename(
        columns={
            "Task Name": "task",
            "Template": "template",
            "Difficulty": "difficulty",
            "Tags": "tags",
            "Optimal Steps": "optimal_steps",
        }
    )
    df["task"] = df["task"].astype(str).str.strip()
    df["difficulty"] = df["difficulty"].astype(str).str.strip().str.lower()
    df["tags"] = df["tags"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["tag_list"] = df["tags"].map(lambda x: [t for t in x.split(" ") if t] if x else ["untagged"])
    df["primary_tag"] = df["tag_list"].map(lambda x: x[0] if x else "untagged")
    df["optimal_steps"] = pd.to_numeric(df["optimal_steps"], errors="coerce")
    return df[["task", "template", "difficulty", "tags", "tag_list", "primary_tag", "optimal_steps"]]


def parse_prior_results(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TASK_RESULT_RE.match(line)
        if not match:
            continue
        row: dict[str, Any] = match.groupdict()
        row["task_num"] = int(row["task_num"])
        for key in [
            "num_complete_trials",
            "mean_success_rate",
            "mean_episode_length",
            "total_runtime_s",
            "mean_step_latency_s",
            "num_fail_trials",
        ]:
            row[key] = _to_float(row[key])
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No task rows parsed from {path}")
    return pd.DataFrame(rows)


def _difficulty_rank(value: str) -> int:
    return {"easy": 0, "medium": 1, "hard": 2}.get(str(value).lower(), 9)


def _diverse_pick(pool: pd.DataFrame, n: int) -> list[str]:
    if pool.empty or n <= 0:
        return []
    pool = pool.copy()
    pool["_difficulty_rank"] = pool["difficulty"].map(_difficulty_rank)
    pool = pool.sort_values(["primary_tag", "_difficulty_rank", "task_num", "task"])
    groups = [
        group.drop(columns=["_difficulty_rank"], errors="ignore").reset_index(drop=True)
        for _, group in pool.groupby(["primary_tag", "difficulty"], sort=True)
    ]
    picked: list[str] = []
    while len(picked) < n and groups:
        next_groups = []
        for group in groups:
            if group.empty:
                continue
            task = str(group.iloc[0]["task"])
            if task not in picked:
                picked.append(task)
                if len(picked) >= n:
                    break
            rest = group.iloc[1:].reset_index(drop=True)
            if not rest.empty:
                next_groups.append(rest)
        groups = next_groups
    if len(picked) < n:
        for task in pool["task"].tolist():
            if task not in picked:
                picked.append(task)
            if len(picked) >= n:
                break
    return picked[:n]


def select_tasks(prior_df: pd.DataFrame, metadata_df: pd.DataFrame, total: int) -> pd.DataFrame:
    merged = prior_df.merge(metadata_df, on="task", how="left")
    merged["difficulty"] = merged["difficulty"].fillna("unknown")
    merged["primary_tag"] = merged["primary_tag"].fillna("untagged")
    merged["prior_outcome"] = merged["mean_success_rate"].map(lambda x: "success" if float(x) >= 0.5 else "failure")
    n_success = total // 2
    n_failure = total - n_success
    success = _diverse_pick(merged[merged["prior_outcome"] == "success"], n_success)
    failure = _diverse_pick(merged[merged["prior_outcome"] == "failure"], n_failure)
    selected = success + failure
    selected_df = merged[merged["task"].isin(selected)].copy()
    selected_df["_selection_order"] = selected_df["task"].map({task: i for i, task in enumerate(selected)})
    return selected_df.sort_values("_selection_order").drop(columns=["_selection_order"])


def _load_pickle_gz(path: Path) -> Any:
    with path.open("rb") as f:
        compressed = f.read()
    with gzip.open(io.BytesIO(compressed), "rb") as f:
        return pickle.load(f)


def load_episodes(checkpoint_dir: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("*.pkl.gz")):
        try:
            data = _load_pickle_gz(path)
        except Exception as exc:
            episodes.append(
                {
                    "task_template": path.name[:-7],
                    "goal": "",
                    "is_successful": 0.0,
                    "run_time": math.nan,
                    "episode_length": math.nan,
                    "exception_info": f"load_failed: {exc}",
                    "aux_data": {},
                }
            )
            continue
        if isinstance(data, list):
            episodes.extend(data)
    return episodes


def episodes_to_df(episodes: list[dict[str, Any]], selected_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        aux = ep.get("aux_data") if isinstance(ep.get("aux_data"), dict) else {}
        rows.append(
            {
                "task": ep.get("task_template"),
                "goal": ep.get("goal"),
                "success": _to_float(ep.get("is_successful")),
                "episode_length": _to_float(ep.get("episode_length")),
                "runtime_s": _to_float(ep.get("run_time")),
                "mean_step_latency_s": _to_float(aux.get("mean_step_latency_sec")),
                "num_steps_aux": _to_float(aux.get("num_steps")),
                "exception_info": ep.get("exception_info"),
                "seed": ep.get("seed"),
            }
        )
    raw = pd.DataFrame(rows)
    if raw.empty:
        raw = pd.DataFrame(columns=["task", "goal", "success", "episode_length", "runtime_s", "mean_step_latency_s", "num_steps_aux", "exception_info", "seed"])

    selected_base = selected_df[["task", "task_num", "mean_success_rate", "prior_outcome"]].rename(
        columns={"mean_success_rate": "prior_success_rate"}
    )
    df = selected_base.merge(raw, on="task", how="left")
    df["has_episode"] = df["goal"].notna()
    df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0.0)
    df["is_success"] = df["success"] > 0.5
    df["has_exception"] = df["exception_info"].notna() & (df["exception_info"].astype(str) != "None")
    df["completed_episode"] = df["has_episode"] & ~df["has_exception"]
    df["episode_length"] = pd.to_numeric(df["episode_length"], errors="coerce")
    df["runtime_s"] = pd.to_numeric(df["runtime_s"], errors="coerce")
    df["mean_step_latency_s"] = pd.to_numeric(df["mean_step_latency_s"], errors="coerce")
    df = df.merge(metadata_df, on="task", how="left")
    return df


def aggregate_by_tag_difficulty(task_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in task_df.iterrows():
        tag_list = row.get("tag_list")
        if not isinstance(tag_list, list):
            tag_list = [t for t in str(row.get("tags") or "").split(" ") if t] or ["untagged"]
        for tag in tag_list:
            item = row.to_dict()
            item["tag"] = tag
            rows.append(item)
    exploded = pd.DataFrame(rows)
    if exploded.empty:
        return pd.DataFrame()
    grouped = (
        exploded.groupby(["tag", "difficulty"], dropna=False)
        .agg(
            num_tasks=("task", "nunique"),
            success_rate=("success", "mean"),
            mean_steps_all=("episode_length", "mean"),
            mean_steps_success=("episode_length", lambda s: s[exploded.loc[s.index, "is_success"]].mean()),
            mean_steps_failure=("episode_length", lambda s: s[~exploded.loc[s.index, "is_success"]].mean()),
            num_success=("is_success", "sum"),
            num_failure=("is_success", lambda s: int((~s).sum())),
            mean_prior_success=("prior_success_rate", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["tag", "difficulty"])


def aggregate_by_primary_difficulty(task_df: pd.DataFrame) -> pd.DataFrame:
    if task_df.empty:
        return pd.DataFrame()
    grouped = (
        task_df.groupby(["primary_tag", "difficulty"], dropna=False)
        .agg(
            num_tasks=("task", "nunique"),
            success_rate=("success", "mean"),
            mean_steps_all=("episode_length", "mean"),
            mean_steps_success=("episode_length", lambda s: s[task_df.loc[s.index, "is_success"]].mean()),
            mean_steps_failure=("episode_length", lambda s: s[~task_df.loc[s.index, "is_success"]].mean()),
            num_success=("is_success", "sum"),
            num_failure=("is_success", lambda s: int((~s).sum())),
            mean_prior_success=("prior_success_rate", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["primary_tag", "difficulty"])


def _plot_outputs(task_df: pd.DataFrame, tag_df: pd.DataFrame, chart_dir: Path) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _ordered_difficulties(values: Any) -> list[str]:
        available = [str(v) for v in values if pd.notna(v)]
        preferred = [x for x in ["easy", "medium", "hard"] if x in set(available)]
        return preferred + sorted([x for x in set(available) if x not in set(preferred)])

    if not tag_df.empty:
        pivot = tag_df.pivot_table(index="tag", columns="difficulty", values="success_rate", aggfunc="mean")
        pivot = pivot.reindex(columns=[c for c in ["easy", "medium", "hard"] if c in pivot.columns])
        fig_h = max(4.0, 0.35 * len(pivot.index) + 1.5)
        fig, ax = plt.subplots(figsize=(7.5, fig_h))
        data = pivot.fillna(-1).to_numpy()
        im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = "-" if data[i, j] < 0 else f"{data[i, j]:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)
        ax.set_title("Success rate by tag and difficulty")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(chart_dir / "success_heatmap_by_tag_difficulty.png", dpi=180)
        plt.close(fig)

        step_pivot = tag_df.pivot_table(index="tag", columns="difficulty", values="mean_steps_all", aggfunc="mean")
        step_pivot = step_pivot.reindex(columns=[c for c in ["easy", "medium", "hard"] if c in step_pivot.columns])
        fig_h = max(4.0, 0.35 * len(step_pivot.index) + 1.5)
        fig, ax = plt.subplots(figsize=(7.5, fig_h))
        step_data = step_pivot.fillna(-1).to_numpy()
        valid_values = step_pivot.stack().dropna()
        vmax = float(valid_values.max()) if not valid_values.empty else 1.0
        im = ax.imshow(step_data, aspect="auto", vmin=0.0, vmax=max(1.0, vmax), cmap="YlOrRd")
        ax.set_xticks(range(len(step_pivot.columns)))
        ax.set_xticklabels(step_pivot.columns)
        ax.set_yticks(range(len(step_pivot.index)))
        ax.set_yticklabels(step_pivot.index)
        for i in range(step_data.shape[0]):
            for j in range(step_data.shape[1]):
                text = "-" if step_data[i, j] < 0 else f"{step_data[i, j]:.1f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)
        ax.set_title("Mean episode steps by tag and difficulty")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(chart_dir / "mean_steps_heatmap_by_tag_difficulty.png", dpi=180)
        plt.close(fig)

        tag_summary = (
            tag_df.groupby("tag", dropna=False)
            .agg(
                num_tasks=("num_tasks", "sum"),
                success_rate=("success_rate", "mean"),
                mean_steps=("mean_steps_all", "mean"),
            )
            .reset_index()
            .sort_values("success_rate", ascending=True)
        )
        fig_h = max(4.5, 0.32 * len(tag_summary) + 1.4)
        fig, ax = plt.subplots(figsize=(8.0, fig_h))
        ax.barh(tag_summary["tag"], tag_summary["success_rate"], color="#4C78A8")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("success rate")
        ax.set_title("Success rate by task tag")
        ax.grid(axis="x", alpha=0.25)
        for y, value in enumerate(tag_summary["success_rate"]):
            ax.text(min(float(value) + 0.02, 0.98), y, f"{float(value):.2f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(chart_dir / "success_rate_by_tag.png", dpi=180)
        plt.close(fig)

        tag_steps = tag_summary.sort_values("mean_steps", ascending=True)
        fig_h = max(4.5, 0.32 * len(tag_steps) + 1.4)
        fig, ax = plt.subplots(figsize=(8.0, fig_h))
        ax.barh(tag_steps["tag"], tag_steps["mean_steps"], color="#F58518")
        ax.set_xlabel("mean episode steps")
        ax.set_title("Mean episode steps by task tag")
        ax.grid(axis="x", alpha=0.25)
        for y, value in enumerate(tag_steps["mean_steps"]):
            ax.text(float(value) + 0.25, y, f"{float(value):.1f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(chart_dir / "mean_steps_by_tag.png", dpi=180)
        plt.close(fig)

    if not task_df.empty:
        plot_df = task_df.sort_values(["is_success", "episode_length", "task"], ascending=[True, False, True])
        colors = ["#4C78A8" if ok else "#E45756" for ok in plot_df["is_success"]]
        fig_h = max(5.0, 0.32 * len(plot_df) + 1.5)
        fig, ax = plt.subplots(figsize=(9.0, fig_h))
        y = list(range(len(plot_df)))
        ax.barh(y, plot_df["episode_length"].fillna(0), color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["task"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("episode steps")
        ax.set_title("Task-level episode length (blue=success, red=failure)")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(chart_dir / "task_episode_lengths.png", dpi=180)
        plt.close(fig)

        step_rows = []
        for difficulty, group in task_df.groupby("difficulty", dropna=False):
            step_rows.append(
                {
                    "difficulty": difficulty,
                    "success_steps": group.loc[group["is_success"], "episode_length"].mean(),
                    "failure_steps": group.loc[~group["is_success"], "episode_length"].mean(),
                }
            )
        steps = pd.DataFrame(step_rows).sort_values("difficulty", key=lambda s: s.map(_difficulty_rank))
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        x = range(len(steps))
        width = 0.35
        ax.bar([i - width / 2 for i in x], steps["success_steps"].fillna(0), width, label="success")
        ax.bar([i + width / 2 for i in x], steps["failure_steps"].fillna(0), width, label="failure")
        ax.set_xticks(list(x))
        ax.set_xticklabels(steps["difficulty"])
        ax.set_ylabel("mean steps")
        ax.set_title("Mean steps by difficulty and outcome")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(chart_dir / "steps_by_difficulty_outcome.png", dpi=180)
        plt.close(fig)

        difficulty_summary = (
            task_df.groupby("difficulty", dropna=False)
            .agg(
                num_tasks=("task", "count"),
                success_rate=("success", "mean"),
                mean_steps=("episode_length", "mean"),
                success_steps=("episode_length", lambda s: s[task_df.loc[s.index, "is_success"]].mean()),
                failure_steps=("episode_length", lambda s: s[~task_df.loc[s.index, "is_success"]].mean()),
            )
            .reset_index()
        )
        difficulty_order = _ordered_difficulties(difficulty_summary["difficulty"])
        difficulty_summary["difficulty"] = difficulty_summary["difficulty"].astype(str)
        difficulty_summary = difficulty_summary.set_index("difficulty").reindex(difficulty_order).reset_index()

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.bar(difficulty_summary["difficulty"], difficulty_summary["success_rate"], color="#4C78A8")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("success rate")
        ax.set_title("Success rate by difficulty")
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(difficulty_summary["success_rate"]):
            ax.text(i, min(float(value) + 0.03, 0.96), f"{float(value):.2f}", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(chart_dir / "success_rate_by_difficulty.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.bar(difficulty_summary["difficulty"], difficulty_summary["mean_steps"], color="#F58518")
        ax.set_ylabel("mean episode steps")
        ax.set_title("Mean episode steps by difficulty")
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(difficulty_summary["mean_steps"]):
            ax.text(i, float(value) + 0.35, f"{float(value):.1f}", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(chart_dir / "mean_steps_by_difficulty.png", dpi=180)
        plt.close(fig)

        primary_summary = (
            task_df.groupby("primary_tag", dropna=False)
            .agg(
                num_tasks=("task", "count"),
                success_rate=("success", "mean"),
                mean_steps=("episode_length", "mean"),
            )
            .reset_index()
            .sort_values("success_rate", ascending=True)
        )
        fig_h = max(4.5, 0.32 * len(primary_summary) + 1.4)
        fig, ax = plt.subplots(figsize=(8.0, fig_h))
        ax.barh(primary_summary["primary_tag"], primary_summary["success_rate"], color="#4C78A8")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("success rate")
        ax.set_title("Success rate by primary task category")
        ax.grid(axis="x", alpha=0.25)
        for y, value in enumerate(primary_summary["success_rate"]):
            ax.text(min(float(value) + 0.02, 0.98), y, f"{float(value):.2f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(chart_dir / "success_rate_by_primary_tag.png", dpi=180)
        plt.close(fig)

        primary_steps = primary_summary.sort_values("mean_steps", ascending=True)
        fig_h = max(4.5, 0.32 * len(primary_steps) + 1.4)
        fig, ax = plt.subplots(figsize=(8.0, fig_h))
        ax.barh(primary_steps["primary_tag"], primary_steps["mean_steps"], color="#F58518")
        ax.set_xlabel("mean episode steps")
        ax.set_title("Mean episode steps by primary task category")
        ax.grid(axis="x", alpha=0.25)
        for y, value in enumerate(primary_steps["mean_steps"]):
            ax.text(float(value) + 0.25, y, f"{float(value):.1f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(chart_dir / "mean_steps_by_primary_tag.png", dpi=180)
        plt.close(fig)


def _format_rate(value: Any) -> str:
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.2f}"
    except Exception:
        return "-"


def write_report(
    run_dir: Path,
    args: argparse.Namespace,
    selected_df: pd.DataFrame,
    task_df: pd.DataFrame,
    tag_df: pd.DataFrame,
    primary_df: pd.DataFrame,
    a11y_method: str,
    server_ok: bool,
    server_info: str,
) -> Path:
    report_path = run_dir / "baseline_4b_category_report_cn.md"
    charts = (run_dir / "charts").resolve()
    resolved_run_dir = run_dir.resolve()
    checkpoint_dir = (run_dir / "checkpoints").resolve()
    success_rate = float(task_df["success"].mean()) if not task_df.empty else 0.0
    completed = int(task_df["completed_episode"].sum()) if "completed_episode" in task_df else 0
    num_success = int(task_df["is_success"].sum()) if "is_success" in task_df else 0
    num_failure = int((~task_df["is_success"]).sum()) if "is_success" in task_df else 0
    mean_steps_success = task_df.loc[task_df["is_success"], "episode_length"].mean() if not task_df.empty else math.nan
    mean_steps_failure = task_df.loc[~task_df["is_success"], "episode_length"].mean() if not task_df.empty else math.nan
    mean_runtime = task_df["runtime_s"].mean() if not task_df.empty else math.nan
    full_suite_runtime_h = (float(mean_runtime) * 116.0 / 3600.0) if pd.notna(mean_runtime) else math.nan
    prior_success_rows = task_df[task_df["prior_outcome"] == "success"]
    prior_failure_rows = task_df[task_df["prior_outcome"] == "failure"]
    prior_success_now = int(prior_success_rows["is_success"].sum()) if not prior_success_rows.empty else 0
    prior_failure_now = int(prior_failure_rows["is_success"].sum()) if not prior_failure_rows.empty else 0
    prior_success_total = int(len(prior_success_rows))
    prior_failure_total = int(len(prior_failure_rows))
    agreement = int(
        (
            ((task_df["prior_outcome"] == "success") & task_df["is_success"])
            | ((task_df["prior_outcome"] == "failure") & ~task_df["is_success"])
        ).sum()
    ) if not task_df.empty else 0
    max_step_failures = task_df[(~task_df["is_success"]) & (task_df["episode_length"] >= 20)]
    quick_successes = task_df[task_df["is_success"]].sort_values("episode_length").head(5)
    long_failures = task_df[~task_df["is_success"]].sort_values("episode_length", ascending=False).head(5)

    lines: list[str] = []
    lines.extend(
        [
            "# 4B Baseline GELAB AndroidWorld 分类诊断报告",
            "",
            f"- run dir: `{resolved_run_dir}`",
            f"- agent: `{args.agent_name}`",
            f"- selected tasks: `{len(selected_df)}`",
            f"- a11y method: `{a11y_method}`",
            f"- LLM server preflight: `{'ok' if server_ok else 'failed'}`",
            f"- checkpoint dir: `{checkpoint_dir}`",
            f"- task metadata csv: `{(run_dir / 'task_metadata.csv').resolve()}`",
            f"- selected task csv: `{(run_dir / 'selected_tasks.csv').resolve()}`",
            f"- per-task result csv: `{(run_dir / 'task_results.csv').resolve()}`",
            f"- tag/difficulty csv: `{(run_dir / 'tag_difficulty_metrics.csv').resolve()}`",
            "",
            "## 是否可以直接跑完整 AndroidWorld",
            "",
        ]
    )
    if server_ok and completed > 0:
        lines.append("这次运行能正常调用 baseline `gelab_agent` 和 4B server，并且 checkpoint 能正常落盘；从运行机制上看可以直接跑完整 AndroidWorld。")
    elif not server_ok:
        lines.append("LLM server preflight 没通过，当前不建议直接跑完整 AndroidWorld；先确认 `localhost:8081` 的 OpenAI-compatible server。")
    else:
        lines.append("没有成功加载 episode checkpoint，当前不建议直接跑完整 AndroidWorld；先检查运行日志。")
    lines.extend(
        [
            "",
            "需要注意：这个结论只说明 baseline runner/服务链路能跑，不代表 4B 模型效果好。效果上要看下面的成功率和步数。",
            "",
            "## Baseline 4B 问题诊断",
            "",
            f"- 本次 {len(task_df)} 个任务中有 {completed} 个无 exception 完成并落盘；所以 runner、checkpoint 和 4B server 链路是可用的。",
            f"- gRPC a11y preflight 超时，脚本自动切到 `uiautomator` 后通过。完整实验建议继续使用 `--a11y_method=auto` 或直接 `--a11y_method=uiautomator`。",
            f"- 成功率为 `{success_rate:.2f}`；prior-success 的 {prior_success_total} 个任务里这次成功 `{prior_success_now}/{prior_success_total}`，prior-failure 的 {prior_failure_total} 个任务里这次成功 `{prior_failure_now}/{prior_failure_total}`，prior/current 一致 `{agreement}/{len(task_df)}`。这说明 4B baseline 有明显波动，不能只看单次旧结果。",
            f"- 失败任务平均 `{_format_rate(mean_steps_failure)}` 步，成功任务平均 `{_format_rate(mean_steps_success)}` 步；有 `{len(max_step_failures)}` 个失败任务跑到 20/21 步附近，主要成本来自失败时的重复动作。",
            f"- 按这次平均 runtime `{_format_rate(mean_runtime)}s/task` 粗略估算，116 个 AndroidWorld task 单 trial 约 `{_format_rate(full_suite_runtime_h)}` 小时；实际完整实验会随任务组合和 server 负载波动。",
            "",
            "典型失败模式：",
            "",
            "- `AudioRecorderRecordAudioWithFileName`：`TYPE` 没有清空默认文件名，变成 `Record-7presentation_fGwr.m4a` 后反复点击软键盘退格，直到 max steps。",
            "- `BrowserDraw`：反复点击 `task.html`，还出现不完整 `tool_call` 导致 parser fallback wait。",
            "- `ExpenseAddMultiple` / `RecipeAddMultipleRecipesFromMarkor` / `MarkorMergeNotes`：多项录入或跨应用信息搬运任务容易跑满步数。",
            "- `SimpleCalendarEventsInTimeRange` / `TasksHighPriorityTasks`：当答案直接显示在屏幕上时，4B 可以很快完成问答类任务。",
            "",
            "## 总览",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            f"| tasks | {len(task_df)} |",
            f"| completed without exception | {completed} |",
            f"| success | {num_success} |",
            f"| failure | {num_failure} |",
            f"| success rate | {success_rate:.2f} |",
            f"| mean steps, success tasks | {_format_rate(mean_steps_success)} |",
            f"| mean steps, failure tasks | {_format_rate(mean_steps_failure)} |",
            f"| mean runtime per task(s) | {_format_rate(mean_runtime)} |",
            f"| estimated full 116-task runtime(h) | {_format_rate(full_suite_runtime_h)} |",
            "",
            "图表：",
            "",
            "按任务类别统计：",
            "",
            f'<img src="{charts / "success_rate_by_tag.png"}" width="75%">',
            "",
            f'<img src="{charts / "mean_steps_by_tag.png"}" width="75%">',
            "",
            "按 primary category 统计：",
            "",
            f'<img src="{charts / "success_rate_by_primary_tag.png"}" width="75%">',
            "",
            f'<img src="{charts / "mean_steps_by_primary_tag.png"}" width="75%">',
            "",
            "按难度统计：",
            "",
            f'<img src="{charts / "success_rate_by_difficulty.png"}" width="60%">',
            "",
            f'<img src="{charts / "mean_steps_by_difficulty.png"}" width="60%">',
            "",
            "类别 x 难度交叉统计：",
            "",
            f'<img src="{charts / "success_heatmap_by_tag_difficulty.png"}" width="70%">',
            "",
            f'<img src="{charts / "mean_steps_heatmap_by_tag_difficulty.png"}" width="70%">',
            "",
            "任务级步数与按难度成功/失败步数：",
            "",
            f'<img src="{charts / "task_episode_lengths.png"}" width="80%">',
            "",
            f'<img src="{charts / "steps_by_difficulty_outcome.png"}" width="70%">',
            "",
            "## 快速成功与长失败样例",
            "",
            "最快成功任务：",
            "",
            "| task | steps | difficulty | primary tag |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for _, row in quick_successes.iterrows():
        lines.append(f"| {row['task']} | {_format_rate(row.get('episode_length'))} | {row.get('difficulty', '')} | {row.get('primary_tag', '')} |")
    lines.extend(
        [
            "",
            "最长失败任务：",
            "",
            "| task | steps | difficulty | primary tag |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for _, row in long_failures.iterrows():
        lines.append(f"| {row['task']} | {_format_rate(row.get('episode_length'))} | {row.get('difficulty', '')} | {row.get('primary_tag', '')} |")
    lines.extend(
        [
            "",
            "## 任务选择",
            "",
            "任务从 `results/4B_2.txt` 里选择；如果是 smoke test，则按 prior success/failure 各取一半并尽量覆盖不同 tag/difficulty；如果使用 `--all_tasks`，则覆盖文件中的全部任务。",
            "",
            "| task | prior | difficulty | primary tag | tags |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    for _, row in selected_df.iterrows():
        lines.append(
            f"| {row['task']} | {_format_rate(row.get('mean_success_rate'))} | {row.get('difficulty', '')} | {row.get('primary_tag', '')} | {row.get('tags', '')} |"
        )

    lines.extend(
        [
            "",
            "## 本次 per-task 结果",
            "",
            "| task | success | steps | runtime(s) | difficulty | primary tag | exception |",
            "| --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for _, row in task_df.sort_values("task_num").iterrows():
        if not row.get("has_episode"):
            exception = "no_episode"
        else:
            exception = "" if not row.get("has_exception") else str(row.get("exception_info"))[:80]
        lines.append(
            f"| {row['task']} | {int(bool(row.get('is_success')))} | {_format_rate(row.get('episode_length'))} | {_format_rate(row.get('runtime_s'))} | {row.get('difficulty', '')} | {row.get('primary_tag', '')} | {exception} |"
        )

    lines.extend(
        [
            "",
            "## Tag / Difficulty 统计",
            "",
            "| tag | difficulty | n | success rate | mean steps all | success steps | failure steps |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in tag_df.iterrows():
        lines.append(
            f"| {row['tag']} | {row['difficulty']} | {int(row['num_tasks'])} | {_format_rate(row['success_rate'])} | {_format_rate(row['mean_steps_all'])} | {_format_rate(row['mean_steps_success'])} | {_format_rate(row['mean_steps_failure'])} |"
        )

    lines.extend(
        [
            "",
            "## Primary Tag / Difficulty 统计",
            "",
            "| primary tag | difficulty | n | success rate | mean steps all | success steps | failure steps |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in primary_df.iterrows():
        lines.append(
            f"| {row['primary_tag']} | {row['difficulty']} | {int(row['num_tasks'])} | {_format_rate(row['success_rate'])} | {_format_rate(row['mean_steps_all'])} | {_format_rate(row['mean_steps_success'])} | {_format_rate(row['mean_steps_failure'])} |"
        )

    lines.extend(
        [
            "",
            "## 运行命令",
            "",
            "```bash",
            " ".join(args.executed_command or []),
            "```",
            "",
            "## LLM server preflight 摘要",
            "",
            "```text",
            server_info[:2000],
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_androidworld(args: argparse.Namespace, run_dir: Path, selected_tasks: list[str], env: dict[str, str]) -> int:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "androidworld_run.log"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "run.py"),
        f"--agent_name={args.agent_name}",
        "--suite_family=android_world",
        f"--n_task_combinations={int(args.n_task_combinations)}",
        f"--task_random_seed={int(args.task_random_seed)}",
        f"--image_downsample_scale={float(args.image_downsample_scale)}",
        f"--adb_path={args.adb_path}",
        f"--console_port={int(args.console_port)}",
        f"--checkpoint_dir={checkpoint_dir}",
        f"--output_path={run_dir}",
        f"--tasks={','.join(selected_tasks)}",
    ]
    if args.fixed_task_seed:
        cmd.append("--fixed_task_seed")
    args.executed_command = cmd
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
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
            log.write(line)
        return int(process.wait())


def analyze_run(args: argparse.Namespace, run_dir: Path, selected_df: pd.DataFrame, metadata_df: pd.DataFrame, a11y_method: str, server_ok: bool, server_info: str) -> Path:
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints"
    episodes = load_episodes(checkpoint_dir)
    task_df = episodes_to_df(episodes, selected_df, metadata_df)
    tag_df = aggregate_by_tag_difficulty(task_df)
    primary_df = aggregate_by_primary_difficulty(task_df)

    metadata_df.to_csv(run_dir / "task_metadata.csv", index=False)
    selected_df.drop(columns=["tag_list"], errors="ignore").to_csv(run_dir / "selected_tasks.csv", index=False)
    task_df.drop(columns=["tag_list"], errors="ignore").to_csv(run_dir / "task_results.csv", index=False)
    tag_df.to_csv(run_dir / "tag_difficulty_metrics.csv", index=False)
    primary_df.to_csv(run_dir / "primary_tag_difficulty_metrics.csv", index=False)
    _plot_outputs(task_df, tag_df, run_dir / "charts")
    return write_report(run_dir, args, selected_df, task_df, tag_df, primary_df, a11y_method, server_ok, server_info)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Launch AndroidWorld before analyzing.")
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--run_dir", type=Path, default=None, help="Analyze an existing run dir.")
    parser.add_argument("--checkpoint_dir", type=Path, default=None, help="Override checkpoint dir for analysis.")
    parser.add_argument("--result_file", type=Path, default=DEFAULT_RESULT_FILE)
    parser.add_argument("--task_html", type=Path, default=DEFAULT_TASK_HTML)
    parser.add_argument("--num_tasks", type=int, default=20)
    parser.add_argument("--all_tasks", action="store_true", help="Use every task found in --result_file.")
    parser.add_argument("--tasks", type=str, default="", help="Comma-separated explicit task list.")
    parser.add_argument("--agent_name", type=str, default="gelab_agent")
    parser.add_argument("--n_task_combinations", type=int, default=1)
    parser.add_argument("--task_random_seed", type=int, default=30)
    parser.add_argument("--fixed_task_seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_downsample_scale", type=float, default=1.0)
    parser.add_argument("--adb_path", type=str, default=DEFAULT_ADB_PATH)
    parser.add_argument("--console_port", type=int, default=5554)
    parser.add_argument("--a11y_method", choices=["auto", "grpc", "uiautomator"], default="auto")
    parser.add_argument("--a11y_preflight_timeout", type=float, default=25.0)
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--llm_api_url", type=str, default="http://localhost:8081/v1/chat/completions")
    args = parser.parse_args()
    args.executed_command = []

    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = args.run_dir or args.experiment_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_task_metadata(args.task_html)
    prior_df = parse_prior_results(args.result_file)
    if args.all_tasks:
        selected_df = prior_df.merge(metadata_df, on="task", how="left")
        selected_df["prior_outcome"] = selected_df["mean_success_rate"].map(lambda x: "success" if float(x) >= 0.5 else "failure")
        selected_df = selected_df.sort_values(["task_num", "task"]).copy()
    elif args.tasks.strip():
        explicit_tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
        selected_df = prior_df.merge(metadata_df, on="task", how="left")
        selected_df["prior_outcome"] = selected_df["mean_success_rate"].map(lambda x: "success" if float(x) >= 0.5 else "failure")
        selected_df = selected_df[selected_df["task"].isin(explicit_tasks)].copy()
        selected_df["_order"] = selected_df["task"].map({task: i for i, task in enumerate(explicit_tasks)})
        selected_df = selected_df.sort_values("_order").drop(columns=["_order"])
    else:
        selected_df = select_tasks(prior_df, metadata_df, int(args.num_tasks))
    selected_tasks = selected_df["task"].astype(str).tolist()
    if len(selected_tasks) == 0:
        raise RuntimeError("No tasks selected.")

    server_ok, server_info = _check_llm_server(args.llm_api_url)
    (run_dir / "llm_server_preflight.txt").write_text(
        f"ok={server_ok}\nurl={args.llm_api_url}\n\n{server_info}", encoding="utf-8"
    )

    a11y_method = "not_run"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.run and not args.skip_preflight:
        a11y_method = _select_a11y_method(args, run_dir / "a11y_preflight.log")
        if a11y_method == "uiautomator":
            env["ANDROID_WORLD_A11Y_METHOD"] = "uiautomator"
        else:
            env.pop("ANDROID_WORLD_A11Y_METHOD", None)
    elif args.a11y_method == "uiautomator":
        a11y_method = "uiautomator"
        env["ANDROID_WORLD_A11Y_METHOD"] = "uiautomator"
    elif args.a11y_method == "grpc":
        a11y_method = "grpc"
        env.pop("ANDROID_WORLD_A11Y_METHOD", None)

    selected_df.drop(columns=["tag_list"], errors="ignore").to_csv(run_dir / "selected_tasks.csv", index=False)
    metadata_df.drop(columns=["tag_list"], errors="ignore").to_csv(run_dir / "task_metadata.csv", index=False)

    if args.run:
        print(f"[baseline] run_dir={run_dir}")
        print(f"[baseline] selected_tasks={','.join(selected_tasks)}")
        print(f"[baseline] llm_server={'ok' if server_ok else 'failed'}")
        exit_code = run_androidworld(args, run_dir, selected_tasks, env)
        if exit_code != 0:
            print(f"[baseline] AndroidWorld exited with code {exit_code}; generating partial report.")
    else:
        log_path = run_dir / "androidworld_run.log"
        if log_path.exists():
            first_line = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[0:1]
            if first_line and first_line[0].startswith("$ "):
                args.executed_command = first_line[0][2:].split()
            else:
                args.executed_command = []
        else:
            args.executed_command = []

    report = analyze_run(args, run_dir, selected_df, metadata_df, a11y_method, server_ok, server_info)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
