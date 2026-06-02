#!/usr/bin/env python3
"""Generate a detailed Chinese report for forced exploration traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _clean(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").replace("\n", " ").split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _img(path: str | Path, width: str = "20%") -> str:
    p = Path(path)
    if not path or not p.exists():
        return "`<missing screenshot>`"
    return f'<img src="{p.resolve()}" width="{width}">'


def _candidate_center(candidate: dict[str, Any]) -> tuple[int, int] | None:
    center = candidate.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        try:
            return int(center[0]), int(center[1])
        except (TypeError, ValueError):
            return None
    return None


def _candidate_center_and_bbox(candidate: dict[str, Any]) -> tuple[tuple[float, float] | None, dict[str, float] | None]:
    center = _candidate_center(candidate)
    parsed_center = (float(center[0]), float(center[1])) if center else None
    a11y = candidate.get("a11y") if isinstance(candidate.get("a11y"), dict) else {}
    bbox = a11y.get("bbox") if isinstance(a11y.get("bbox"), dict) else None
    if bbox:
        try:
            return parsed_center, {
                "x_min": float(bbox["x_min"]),
                "x_max": float(bbox["x_max"]),
                "y_min": float(bbox["y_min"]),
                "y_max": float(bbox["y_max"]),
            }
        except (KeyError, TypeError, ValueError):
            return parsed_center, None
    return parsed_center, None


def _point_from_action(action: dict[str, Any]) -> tuple[float, float] | None:
    action_dict = action.get("action_dict") if isinstance(action.get("action_dict"), dict) else {}
    try:
        x = action_dict.get("x")
        y = action_dict.get("y")
        if x is None or y is None:
            return None
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


def _text_from_action(action: dict[str, Any]) -> str:
    action_dict = action.get("action_dict") if isinstance(action.get("action_dict"), dict) else {}
    return _clean(action_dict.get("text"), 512)


def _point_hits_candidate(point: tuple[float, float], candidate: dict[str, Any]) -> bool:
    center, bbox = _candidate_center_and_bbox(candidate)
    x, y = point
    if bbox and bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]:
        return True
    if center:
        return ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5 <= 120.0
    return False


def _hint_follow(action: dict[str, Any]) -> dict[str, Any]:
    results = action.get("matched_exploration_results")
    if not isinstance(results, list) or not results:
        return {"eligible": False, "followed": False, "reason": "no_matched_results"}
    typed_text = _text_from_action(action)
    if typed_text:
        for item in results:
            if not isinstance(item, dict):
                continue
            cand = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
            if str(cand.get("action_kind") or "").lower() == "type" and _clean(cand.get("text"), 512) == typed_text:
                return {"eligible": True, "followed": True, "reason": "type_matched", "label": cand.get("label")}
        return {"eligible": True, "followed": False, "reason": "type_missed"}
    point = _point_from_action(action)
    if point is None:
        return {"eligible": True, "followed": False, "reason": "no_click_point"}
    for item in results:
        if not isinstance(item, dict):
            continue
        cand = item.get("next_candidate") if isinstance(item.get("next_candidate"), dict) else {}
        if _point_hits_candidate(point, cand):
            return {"eligible": True, "followed": True, "reason": "click_matched", "label": cand.get("label")}
    return {"eligible": True, "followed": False, "reason": "click_missed"}


def _annotate_points(
    image_path: str | Path,
    candidates: list[dict[str, Any]],
    out_path: Path,
    color: tuple[int, int, int] = (255, 0, 0),
) -> str:
    src = Path(image_path)
    if not src.exists():
        return ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 34)
    except Exception:  # pylint: disable=broad-exception-caught
        font = ImageFont.load_default()
    for idx, candidate in enumerate(candidates, start=1):
        center = _candidate_center(candidate)
        if center is None:
            continue
        x, y = center
        radius = 42
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=8)
        draw.text((x + radius + 6, y - radius), str(idx), fill=color, font=font)
    img.save(out_path)
    return str(out_path)


def _action_label(action: dict[str, Any]) -> str:
    action_dict = action.get("action_dict") if isinstance(action.get("action_dict"), dict) else {}
    action_type = _clean(action_dict.get("action_type") or "")
    if action_type in {"click", "long_press"}:
        return f"{action_type}@({action_dict.get('x')},{action_dict.get('y')})"
    if action_type == "input_text":
        return f'TYPE "{_clean(action_dict.get("text"), 80)}"'
    if action_type == "open_app":
        return f"OPEN_APP {action_dict.get('app_name')}"
    if action_type == "swipe":
        return f"SWIPE {action_dict.get('direction')}"
    return action_type or _clean(action_dict)


def _table_row(values: list[Any]) -> str:
    return "| " + " | ".join(_clean(v, 180) for v in values) + " |"


def _a11y_excerpt(items: list[dict[str, Any]], limit: int = 12) -> str:
    compact = []
    for item in items[:limit]:
        compact.append(
            {
                "text": _clean(item.get("text"), 80),
                "desc": _clean(item.get("desc"), 80),
                "class": _clean(item.get("class"), 80),
                "center": item.get("center"),
                "clickable": item.get("clickable"),
            }
        )
    return json.dumps(compact, ensure_ascii=False, indent=2)


def generate(run_dir: Path, task_filter: str = "") -> Path:
    trace_root = run_dir / "traces"
    report_dir = run_dir / "report"
    annotated_dir = report_dir / "forced_exploration_annotated"
    report_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = [p for p in sorted(trace_root.iterdir()) if p.is_dir()]
    if task_filter:
        task_dirs = [p for p in task_dirs if task_filter in p.name]
    if not task_dirs:
        raise RuntimeError(f"No task trace dirs found under {trace_root}")
    task_dir = task_dirs[0]

    actions = _read_jsonl(task_dir / "action.jsonl")
    explorations = _read_jsonl(task_dir / "exploration_trace.jsonl")
    matches = _read_jsonl(task_dir / "exploration_match_trace.jsonl")
    rollbacks = _read_jsonl(task_dir / "rollback_trace.jsonl")
    explorations_by_step = {int(t.get("step") or 0): t for t in explorations}
    matches_by_step = {int(t.get("step") or 0): t for t in matches}
    matched_source_steps: set[int] = set()
    matched_source_candidates = 0
    total_source_candidates = 0
    for match in matches:
        for item in list(match.get("matches") or []):
            if not isinstance(item, dict):
                continue
            total_source_candidates += 1
            if item.get("matched"):
                matched_source_candidates += 1
                try:
                    matched_source_steps.add(int(item.get("source_step") or 0))
                except (TypeError, ValueError):
                    pass
    available_trace_count = len([t for t in explorations if t.get("available_for_next_prompt")])
    next_step_hit_text = (
        f"{len(matched_source_steps)}/{available_trace_count} available traces"
        if available_trace_count
        else "0/0 (no available traces)"
    )
    hint_rows = [a for a in actions if a.get("prompt_hint")]
    hint_follow_rows = [a for a in hint_rows if _hint_follow(a).get("followed")]

    structured: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "task_dir": str(task_dir.resolve()),
        "actions": actions,
        "explorations": explorations,
        "matches": matches,
        "rollbacks": rollbacks,
    }
    structured_path = report_dir / "forced_exploration_structured_trace.json"
    structured_path.write_text(json.dumps(structured, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    completed = [t for t in explorations if t.get("status") == "completed"]
    total_observations = sum(len(t.get("observations") or []) for t in explorations)
    success_rollbacks = sum(1 for r in rollbacks if r.get("success"))
    strategy = _clean(next((t.get("strategy") for t in explorations if t.get("strategy")), ""), 80)
    depth_budget = next((t.get("depth_budget") for t in explorations if t.get("depth_budget") is not None), "")
    depth_totals: dict[str, int] = {}
    for trace in explorations:
        for depth, count in (trace.get("depth_counts") or {}).items():
            depth_totals[str(depth)] = int(depth_totals.get(str(depth), 0)) + int(count or 0)
    goal = _clean((actions[0].get("goal") if actions else explorations[0].get("goal") if explorations else ""), 800)

    lines: list[str] = [
        "# 强制 Exploration 单任务诊断报告",
        "",
        f"- run dir: `{run_dir.resolve()}`",
        f"- task trace dir: `{task_dir.resolve()}`",
        f"- structured trace: `{structured_path.resolve()}`",
        f"- task instruction: `{goal}`",
        "",
        "## 总览",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| reasoning steps | {len(actions)} |",
        f"| exploration traces | {len(explorations)} |",
        f"| completed explorations | {len(completed)} |",
        f"| observations / clicked branches | {total_observations} |",
        f"| rollback total | {len(rollbacks)} |",
        f"| rollback success | {success_rollbacks}/{len(rollbacks)} |",
        f"| strategy | {strategy or 'n/a'} |",
        f"| depth budget | {depth_budget} |",
        f"| depth counts | {depth_totals} |",
        f"| exploration next-step hit | {next_step_hit_text} |",
        f"| candidate state-match hit | {matched_source_candidates}/{total_source_candidates} |",
        f"| prompt hint hit | {len(hint_follow_rows)}/{len(hint_rows)} |",
        "",
        "说明：本报告里的红圈表示 exploration 实际点击或准备点击的 UI 元素；截图统一按 20% 宽度展示。`score` 是当前 agent 用于排序的综合相似度/相关性分数，`relevance`、`planning`、`token` 是其中的任务文本、planning 文本、关键词匹配分量。",
        "",
    ]

    for idx, action in enumerate(actions, start=1):
        trace = explorations_by_step.get(idx, {})
        match = matches_by_step.get(idx, {})
        follow = _hint_follow(action)
        screenshot = task_dir / f"screenshot_{idx - 1}.png"
        root_annotated = ""
        selected = list(trace.get("selected_targets") or [])
        if trace.get("root_screenshot"):
            root_annotated = _annotate_points(
                trace.get("root_screenshot"),
                selected,
                annotated_dir / f"step_{idx:02d}_root_selected.png",
            )
        lines.extend(
            [
                f"## Step {idx}",
                "",
                f"- prompt mode: `{action.get('prompt_mode')}`",
                f"- matched previous exploration: `{action.get('matched_exploration_status')}`; matched count: `{action.get('matched_exploration_count')}`",
                f"- hint hit: `{follow.get('followed')}`; reason: `{follow.get('reason')}`",
                f"- post-planning exploration: `{trace.get('status', 'missing')}`; reason: `{trace.get('trigger_reason', '')}`",
                f"- strategy/depth: `{trace.get('strategy', '')}` / `{trace.get('depth_budget', '')}`; next-step hit later: `{idx in matched_source_steps}`",
                f"- main action: `{_action_label(action)}`",
                f"- summary: {_clean(action.get('summary'), 600)}",
                "",
                "当前 reasoning 输入截图：",
                "",
                _img(screenshot),
                "",
                "VLM reasoning / parsed action：",
                "",
                "```json",
                json.dumps(
                    {
                        "response": _clean(action.get("response"), 1400),
                        "parsed_action": action.get("parsed_action"),
                        "action_dict": action.get("action_dict"),
                        "prompt_hint": action.get("prompt_hint"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                "```",
                "",
            ]
        )

        if match.get("prompt_context"):
            lines.extend(["上一轮 exploration 整理后加入本轮 prompt 的 hint：", "", "```text", _clean(match.get("prompt_context"), 1800), "```", ""])
        elif action.get("prompt_hint"):
            lines.extend(["本轮 prompt hint：", "", "```text", _clean(action.get("prompt_hint"), 1800), "```", ""])

        if not trace:
            lines.extend(["没有找到本 step 的 exploration trace。", ""])
            continue

        lines.extend(
            [
                "Exploration root 标注图：",
                "",
                _img(root_annotated or trace.get("root_screenshot", "")),
                "",
                "Root 页面 a11y tree excerpt：",
                "",
                "```json",
                _a11y_excerpt(list(trace.get("root_a11y") or []), limit=12),
                "```",
                "",
                "Top-5 exploration 元素相似度 / 相关性：",
                "",
                "| rank | label | center | score | relevance | planning | token |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for rank, cand in enumerate(list(trace.get("candidates") or [])[:5], start=1):
            lines.append(
                _table_row(
                    [
                        rank,
                        cand.get("label") or cand.get("merged"),
                        cand.get("center"),
                        f"{float(cand.get('score') or 0.0):.3f}",
                        f"{float(cand.get('relevance') or 0.0):.3f}",
                        f"{float(cand.get('planning_relevance') or 0.0):.3f}",
                        f"{float(cand.get('token_relevance') or 0.0):.3f}",
                    ]
                )
            )
        lines.append("")

        lines.extend(["Branch depth 统计：", "", "| branch | depth reached | depth counts | rollback |", "| ---: | ---: | --- | --- |"])
        for obs in list(trace.get("observations") or []):
            rollback = obs.get("rollback") if isinstance(obs.get("rollback"), dict) else {}
            per_branch: dict[str, int] = {}
            for step in list(obs.get("steps") or []):
                depth_key = str(int(step.get("depth") or 0))
                per_branch[depth_key] = int(per_branch.get(depth_key, 0)) + 1
            lines.append(
                _table_row(
                    [
                        obs.get("branch_id"),
                        obs.get("depth_reached"),
                        per_branch,
                        f"{rollback.get('success')}/{rollback.get('level')}/{rollback.get('mode')}",
                    ]
                )
            )
        lines.append("")

        lines.extend(["本 step exploration 选择点击的元素：", "", "| branch | label | center | score |", "| ---: | --- | --- | ---: |"])
        for branch, cand in enumerate(selected, start=1):
            lines.append(
                _table_row(
                    [
                        branch,
                        cand.get("label") or cand.get("merged"),
                        cand.get("center"),
                        f"{float(cand.get('score') or 0.0):.3f}",
                    ]
                )
            )
        lines.append("")

        for obs in list(trace.get("observations") or []):
            rollback = obs.get("rollback") if isinstance(obs.get("rollback"), dict) else {}
            lines.extend(
                [
                    f"### Step {idx} / Branch {obs.get('branch_id')}",
                    "",
                    f"- path: `{ ' -> '.join(_clean(x, 120) for x in list(obs.get('labels') or [])) }`",
                    f"- depth reached: `{obs.get('depth_reached')}`; changed: `{obs.get('changed')}`; score: `{obs.get('score')}`",
                    f"- rollback: success=`{rollback.get('success')}`, level=`{rollback.get('level')}`, mode=`{rollback.get('mode')}`, matched_by=`{rollback.get('matched_by')}`",
                    "",
                ]
            )
            for step in list(obs.get("steps") or []):
                candidate = step.get("candidate") if isinstance(step.get("candidate"), dict) else {}
                annotated = _annotate_points(
                    step.get("screenshot", ""),
                    [candidate],
                    annotated_dir / f"step_{idx:02d}_branch_{obs.get('branch_id')}_depth_{step.get('depth')}.png",
                )
                lines.extend(
                    [
                        f"- depth {step.get('depth')}: action=`{candidate.get('action_kind', 'click')}` label=`{_clean(candidate.get('label') or candidate.get('merged'), 180)}` changed=`{step.get('changed')}` activity=`{_clean(step.get('after_activity'), 180)}`",
                        "",
                        _img(annotated or step.get("screenshot", "")),
                        "",
                        "A11y excerpt after this exploration action:",
                        "",
                        "```json",
                        _a11y_excerpt(list(step.get("a11y") or []), limit=8),
                        "```",
                        "",
                    ]
                )

        if trace.get("speculative_context"):
            lines.extend(["本 step exploration 整理出的 speculative hint（供下一步匹配后使用）：", "", "```text", _clean(trace.get("speculative_context"), 1800), "```", ""])

    report_path = report_dir / "forced_exploration_diagnostic_cn.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--task_filter", default="")
    args = parser.parse_args()
    report = generate(Path(args.run_dir), task_filter=args.task_filter)
    print(report)


if __name__ == "__main__":
    main()
