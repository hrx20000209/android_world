#!/usr/bin/env python3
"""Generate a detailed Chinese TPES trace report for one AndroidWorld task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            item.setdefault("_line", idx)
            rows.append(item)
    return rows


def _short(text: Any, limit: int = 160) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\n", " ").replace("|", "/").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _label(item: dict[str, Any]) -> str:
    return _short(
        item.get("label")
        or item.get("merged")
        or item.get("text")
        or item.get("content_description")
        or item.get("resource_id")
        or item.get("class_name")
        or "",
        120,
    )


def _action_label(action: dict[str, Any]) -> str:
    action_dict = action.get("action_dict") if isinstance(action.get("action_dict"), dict) else {}
    action_type = action_dict.get("action_type")
    if action_type == "click":
        return f"click@({action_dict.get('x')},{action_dict.get('y')})"
    if action_type == "long_press":
        return f"long_press@({action_dict.get('x')},{action_dict.get('y')})"
    if action_type == "input_text":
        return f"type `{_short(action_dict.get('text'), 80)}`"
    if action_type == "open_app":
        return f"open_app `{_short(action_dict.get('app_name'), 80)}`"
    if action_type == "scroll":
        return f"scroll `{_short(action_dict.get('direction'), 40)}`"
    if action_type == "status":
        return f"status `{_short(action_dict.get('goal_status'), 60)}`"
    return _short(action_type or action.get("tool_call") or action.get("parsed_action"), 160)


def _node_label(node: dict[str, Any]) -> str:
    return _short(
        node.get("text")
        or node.get("content_description")
        or node.get("hint_text")
        or node.get("resource_id")
        or node.get("class_name")
        or "",
        120,
    )


def _a11y_excerpt(nodes: list[dict[str, Any]] | None, limit: int = 20) -> str:
    labels: list[str] = []
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        label = _node_label(node)
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= limit:
            break
    return "; ".join(labels) if labels else "<empty>"


def _img(path: Any, width: str) -> str:
    if not path:
        return ""
    p = Path(str(path))
    src = str(p if p.is_absolute() else p.resolve())
    return f'<img src="{src}" width="{width}">'


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(_short(cell, 220) for cell in row) + " |")
    return out


def _find_task_dir(trace_root: Path, task_dir: str = "") -> Path:
    if task_dir:
        direct = trace_root / task_dir
        if direct.exists():
            return direct
        matches = [p for p in trace_root.iterdir() if p.is_dir() and task_dir in p.name]
        if matches:
            return sorted(matches)[0]
        raise FileNotFoundError(f"Cannot find task dir matching {task_dir!r} under {trace_root}")
    candidates = [
        p
        for p in trace_root.iterdir()
        if p.is_dir() and (p / "action.jsonl").exists() and (p / "exploration_trace.jsonl").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No task trace directory found under {trace_root}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _top_candidate_rows(candidates: list[dict[str, Any]], limit: int) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for idx, cand in enumerate(candidates[:limit], start=1):
        rows.append(
            [
                idx,
                f"`{_label(cand)}`",
                cand.get("action_kind") or "",
                f"{float(cand.get('score') or 0):.3f}",
                f"{float(cand.get('relevance') or 0):.3f}",
                cand.get("center"),
                f"`{_short(cand.get('key'), 80)}`",
            ]
        )
    return rows


def _match_rows(match: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    selected_keys = {
        (item.get("source_step"), item.get("branch_id"), item.get("path"))
        for item in match.get("selected_prompt_results") or []
        if isinstance(item, dict)
    }
    for item in match.get("matches") or []:
        if not isinstance(item, dict):
            continue
        key = (item.get("source_step"), item.get("branch_id"), item.get("path"))
        rows.append(
            [
                item.get("source_step"),
                item.get("branch_id"),
                "yes" if item.get("matched") else "no",
                "yes" if key in selected_keys else "no",
                item.get("matched_by"),
                item.get("depth_reached"),
                item.get("evidence_depth"),
                f"`{_short(item.get('path'), 90)}`",
                f"`{_short(item.get('next_label'), 70)}`",
            ]
        )
    return rows


def _branch_section(obs: dict[str, Any], image_width: str, max_step_a11y: int) -> list[str]:
    labels = " -> ".join(str(x) for x in obs.get("labels") or [])
    rollback = obs.get("rollback") if isinstance(obs.get("rollback"), dict) else {}
    lines = [
        f"- branch `{obs.get('branch_id')}`: depth=`{obs.get('depth_reached')}`, "
        f"score=`{float(obs.get('score') or 0):.3f}`, changed=`{obs.get('changed')}`, "
        f"path=`{_short(labels, 180)}`",
    ]
    if rollback:
        lines.append(
            f"  - rollback: success=`{rollback.get('success')}`, level=`{rollback.get('level')}`, "
            f"mode=`{rollback.get('mode')}`, matched_by=`{rollback.get('matched_by')}`"
        )
    stored_rows: list[list[Any]] = []
    for step in obs.get("steps") or []:
        if not isinstance(step, dict):
            continue
        cand = step.get("candidate") if isinstance(step.get("candidate"), dict) else {}
        a11y = step.get("a11y") if isinstance(step.get("a11y"), list) else []
        stored_rows.append(
            [
                step.get("depth"),
                f"`{_label(cand)}`",
                step.get("changed"),
                step.get("after_activity"),
                step.get("after_hash"),
                len(a11y),
                "; ".join(str(x) for x in (step.get("observed_elements") or [])[:max_step_a11y]),
            ]
        )
    if stored_rows:
        lines.extend(
            _table(
                ["depth", "action/element", "changed", "after_activity", "after_hash", "a11y nodes", "stored text labels"],
                stored_rows,
            )
        )
    for step in obs.get("steps") or []:
        if not isinstance(step, dict):
            continue
        screenshot = step.get("screenshot")
        if screenshot:
            cand = step.get("candidate") if isinstance(step.get("candidate"), dict) else {}
            lines.append(f"  - depth {step.get('depth')} screenshot, element=`{_label(cand)}`")
            lines.append(f"    {_img(screenshot, image_width)}")
    return lines


def generate_report(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir).expanduser().resolve()
    trace_root = Path(args.trace_root).expanduser().resolve() if args.trace_root else run_dir / "traces"
    task_dir = _find_task_dir(trace_root, args.task_dir)
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.output).expanduser().resolve() if args.output else report_dir / "single_task_tpes_trace_report_cn.md"

    actions = _read_jsonl(task_dir / "action.jsonl")
    explorations = _read_jsonl(task_dir / "exploration_trace.jsonl")
    matches = _read_jsonl(task_dir / "exploration_match_trace.jsonl")
    rollbacks = _read_jsonl(task_dir / "rollback_trace.jsonl")

    by_step_action = {idx: row for idx, row in enumerate(actions, start=1)}
    by_step_explore = {int(row.get("step") or idx): row for idx, row in enumerate(explorations, start=1)}
    by_step_match = {int(row.get("step") or idx): row for idx, row in enumerate(matches, start=1)}
    max_step = max([0, *by_step_action.keys(), *by_step_explore.keys(), *by_step_match.keys()])
    if args.max_steps:
        max_step = min(max_step, int(args.max_steps))

    status_counts: dict[str, int] = {}
    depth_counts: dict[int, int] = {}
    rollback_total = 0
    rollback_success = 0
    hint_steps = 0
    matched_steps = 0
    for exp in explorations:
        status = str(exp.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        for obs in exp.get("observations") or []:
            if not isinstance(obs, dict):
                continue
            depth = int(obs.get("depth_reached") or 0)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            rb = obs.get("rollback") if isinstance(obs.get("rollback"), dict) else {}
            if rb:
                rollback_total += 1
                rollback_success += int(bool(rb.get("success")))
    for action in actions:
        hint_steps += int(bool(action.get("prompt_hint")))
    for match in matches:
        matched_steps += int((match.get("matched_count") or 0) > 0)

    goal = ""
    if actions:
        goal = str(actions[0].get("goal") or "")
    elif explorations:
        goal = str(explorations[0].get("goal") or "")

    lines: list[str] = []
    lines.append("# 单任务 TPES Exploration Trace 报告")
    lines.append("")
    lines.append(f"- run dir: `{run_dir}`")
    lines.append(f"- trace dir: `{task_dir}`")
    lines.append(f"- task instruction: `{_short(goal, 260)}`")
    lines.append(f"- reasoning steps: `{len(actions)}`")
    lines.append(f"- exploration traces: `{len(explorations)}`")
    lines.append(f"- match traces: `{len(matches)}`")
    lines.append(f"- rollback traces: `{len(rollbacks)}`")
    lines.append("")

    lines.append("## 总体统计")
    lines.append("")
    lines.extend(
        _table(
            ["metric", "value"],
            [
                ["hint injected steps", f"{hint_steps}/{len(actions)}"],
                ["state matched steps", f"{matched_steps}/{len(matches)}"],
                ["exploration status", status_counts],
                ["depth reached distribution", dict(sorted(depth_counts.items()))],
                ["rollback success", f"{rollback_success}/{rollback_total}"],
            ],
        )
    )
    lines.append("")

    lines.append("## Trace 文件里到底存了什么")
    lines.append("")
    lines.append("- `action.jsonl`: 每个 reasoning step 的原始 VLM response、解析后的 action、实际 tool call、当前页面 `start_page_activity/start_page_hash`、本步注入的 `prompt_hint`、匹配到的 exploration results、以及本步 post-planning exploration 的统计。")
    lines.append("- `exploration_trace.jsonl`: 每个 step 执行完 VLM planning 后启动的 speculative exploration。它保存 root 页面 activity/hash/screenshot/a11y tree、候选元素列表、被选中的 root targets、每个 branch 每个 depth 的 action、after_activity、after_hash、screenshot、a11y nodes、压缩后的 `observed_elements` 文本标签、rollback 结果、以及最终可否加入下一步 prompt。")
    lines.append("- `exploration_match_trace.jsonl`: 下一次 reasoning 前，把真实 t+1 页面和上一轮 pending exploration 的 depth-1 页面做匹配。这里保存 pending 数、每个 candidate 是否匹配、`matched_by` 原因、被选入 prompt 的结果、以及最终 prompt context。")
    lines.append("- `rollback_trace.jsonl`: rollback 的成功/失败、level、mode、back/replay 次数、匹配依据和失败分析。")
    lines.append("")

    lines.append("## 页面匹配使用了什么信息")
    lines.append("")
    lines.append("当前实现不是直接把所有 exploration 结果塞进 prompt，而是只注入与真实 t+1 页面对齐的结果。匹配逻辑使用上一轮 speculative branch 的 depth-1 页面作为目标中间态：")
    lines.append("")
    lines.append("1. 必须先比较 `activity`。activity 相同且 screenshot perceptual hash 差值 `<=4`，认为强匹配，`matched_by=activity+strict_phash<=4:<diff>`。")
    lines.append("2. 如果 hash 没有强匹配，则使用 `activity + phash + a11y_jaccard`：保存的 depth-1 `observed_elements` 文本集合与当前真实页面的文本集合做 Jaccard，相同 activity、hash 差值在阈值内、Jaccard >= 0.35 时匹配。")
    lines.append("3. 如果视觉 hash 变化较大，但 activity 相同且 a11y/text 标签 Jaccard >= 0.55，也认为页面语义匹配。")
    lines.append("4. 匹配后注入的 evidence 来自同一 branch 的更深 depth，通常是 depth>=2 或 depth>=3 的截图/a11y/text labels。也就是说它遵循 `t exploration -> t+1 页面匹配 -> 用 t+2/t+3 evidence 增强 t+1 reasoning`。")
    lines.append("")
    lines.append("用于 Jaccard 的 text 信息不是完整 raw XML，而是从 a11y tree 里抽取和过滤后的 `observed_elements`，例如按钮文字、content-desc、resource id 的可读部分、页面标题等。完整 a11y node 仍然保存在 trace 中用于人工检查。")
    lines.append("")

    lines.append("## 每一步 Reasoning + Exploration")
    lines.append("")
    for step in range(1, max_step + 1):
        action = by_step_action.get(step, {})
        match = by_step_match.get(step, {})
        exp = by_step_explore.get(step, {})
        lines.append(f"### Step {step}")
        lines.append("")
        if action:
            screenshot = task_dir / f"screenshot_model_input_{step - 1}.png"
            if screenshot.exists():
                lines.append(_img(screenshot, args.image_width))
                lines.append("")
            lines.append("**Reasoning / Action**")
            lines.append("")
            lines.extend(
                _table(
                    ["field", "value"],
                    [
                        ["start_page_activity", action.get("start_page_activity")],
                        ["start_page_hash", action.get("start_page_hash")],
                        ["prompt_mode", action.get("prompt_mode")],
                        ["matched_exploration_status", action.get("matched_exploration_status")],
                        ["matched_exploration_count", action.get("matched_exploration_count")],
                        ["action", _action_label(action)],
                        ["summary", action.get("summary") or action.get("parsed_action")],
                        ["latency_sec", f"{float(action.get('latency_sec') or 0):.2f}"],
                    ],
                )
            )
            lines.append("")
            if action.get("response"):
                lines.append("VLM raw response:")
                lines.append("")
                lines.append("```text")
                lines.append(_short(action.get("response"), 1800))
                lines.append("```")
                lines.append("")
            if action.get("prompt_hint"):
                lines.append("注入到本步 prompt 的 exploration hint:")
                lines.append("")
                lines.append("```text")
                lines.append(str(action.get("prompt_hint"))[:2400])
                lines.append("```")
                lines.append("")
        if match:
            lines.append("**Pre-reasoning state match**")
            lines.append("")
            lines.extend(
                _table(
                    ["field", "value"],
                    [
                        ["current_activity", match.get("current_activity")],
                        ["pending_trace_count", match.get("pending_trace_count")],
                        ["candidate_match_count", match.get("candidate_match_count")],
                        ["matched_count", match.get("matched_count")],
                        ["status", match.get("status")],
                    ],
                )
            )
            rows = _match_rows(match)
            if rows:
                lines.append("")
                lines.extend(
                    _table(
                        [
                            "source step",
                            "branch",
                            "matched",
                            "selected",
                            "matched_by",
                            "depth",
                            "evidence depth",
                            "path",
                            "next/evidence label",
                        ],
                        rows,
                    )
                )
            if match.get("prompt_context"):
                lines.append("")
                lines.append("match 后生成的 prompt context:")
                lines.append("")
                lines.append("```text")
                lines.append(str(match.get("prompt_context"))[:2400])
                lines.append("```")
            lines.append("")
        if exp:
            lines.append("**Post-planning exploration**")
            lines.append("")
            lines.extend(
                _table(
                    ["field", "value"],
                    [
                        ["status", exp.get("status")],
                        ["trigger_reason", exp.get("trigger_reason")],
                        ["strategy", exp.get("strategy")],
                        ["depth_budget", exp.get("depth_budget")],
                        ["root_activity", exp.get("root_activity")],
                        ["root_hash", exp.get("root_hash")],
                        ["candidate_count", exp.get("candidate_count")],
                        ["selected_targets", len(exp.get("selected_targets") or [])],
                        ["observations", len(exp.get("observations") or [])],
                        ["available_for_next_prompt", exp.get("available_for_next_prompt")],
                        ["latency_ms", f"{float(exp.get('latency_ms') or 0):.1f}"],
                    ],
                )
            )
            if exp.get("root_screenshot"):
                lines.append("")
                lines.append("Root screenshot:")
                lines.append("")
                lines.append(_img(exp.get("root_screenshot"), args.image_width))
            lines.append("")
            lines.append(f"Root a11y text excerpt: `{_a11y_excerpt(exp.get('root_a11y'), args.max_a11y_labels)}`")
            lines.append("")
            if exp.get("candidates"):
                lines.append("Top candidates saved in trace:")
                lines.append("")
                lines.extend(
                    _table(
                        ["rank", "element", "kind", "score", "relevance", "center", "key"],
                        _top_candidate_rows(exp.get("candidates") or [], args.max_candidates),
                    )
                )
                lines.append("")
            if exp.get("selected_targets"):
                lines.append("Selected root targets:")
                lines.append("")
                lines.extend(
                    _table(
                        ["rank", "element", "kind", "score", "relevance", "center", "key"],
                        _top_candidate_rows(exp.get("selected_targets") or [], args.max_candidates),
                    )
                )
                lines.append("")
            for obs in (exp.get("observations") or [])[: args.max_branches]:
                if isinstance(obs, dict):
                    lines.extend(_branch_section(obs, args.image_width, args.max_observed_labels))
                    lines.append("")
            if exp.get("prompt_context"):
                lines.append("Exploration 自己总结出的 prompt context:")
                lines.append("")
                lines.append("```text")
                lines.append(str(exp.get("prompt_context"))[:2200])
                lines.append("```")
                lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--trace_root", default="")
    parser.add_argument("--task_dir", default="", help="Exact task trace dir name or substring.")
    parser.add_argument("--report_dir", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--image_width", default="20%")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_candidates", type=int, default=8)
    parser.add_argument("--max_branches", type=int, default=8)
    parser.add_argument("--max_a11y_labels", type=int, default=20)
    parser.add_argument("--max_observed_labels", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    path = generate_report(parse_args())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
