#!/usr/bin/env python3
"""Generate a Markdown report from synchronous exploration traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
                item["_trace_file"] = str(path)
                item["_task_dir"] = str(path.parent)
                rows.append(item)
    return rows


def _load_traces(trace_root: Path) -> list[dict[str, Any]]:
    return [
        row
        for path in sorted(trace_root.rglob("exploration_trace.jsonl"))
        for row in _read_jsonl(path)
    ]


def _clean(value: Any, max_len: int = 160) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _rate(num: int, den: int) -> str:
    if den <= 0:
        return "n/a"
    return f"{num}/{den} ({num / den:.1%})"


def _image(path: str, alt: str) -> str:
    if not path:
        return ""
    return f"![{alt}]({Path(path).resolve()})"


def _candidate_table(candidates: list[dict[str, Any]], limit: int = 8) -> str:
    lines = [
        "| rank | label | score | relevance | visits | center |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for rank, cand in enumerate(candidates[:limit], start=1):
        lines.append(
            "| {rank} | {label} | {score:.3f} | {rel:.3f} | {visits} | {center} |".format(
                rank=rank,
                label=_clean(cand.get("label"), 60).replace("|", "\\|"),
                score=float(cand.get("score") or 0.0),
                rel=float(cand.get("relevance") or 0.0),
                visits=int(cand.get("visits") or 0),
                center=cand.get("center"),
            )
        )
    return "\n".join(lines)


def _a11y_excerpt(a11y: list[dict[str, Any]], limit: int = 10) -> str:
    compact = []
    for item in a11y[:limit]:
        compact.append(
            {
                "index": item.get("index"),
                "text": item.get("text"),
                "desc": item.get("content_description"),
                "rid": item.get("resource_id"),
                "class": item.get("class_name"),
                "center": item.get("center"),
                "clickable": item.get("is_clickable"),
                "editable": item.get("is_editable"),
            }
        )
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _rollback_rows(traces: list[dict[str, Any]]) -> tuple[int, int, dict[str, dict[str, int]], list[dict[str, Any]]]:
    total = 0
    success = 0
    levels: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []
    for trace in traces:
        for rb in trace.get("rollbacks") or []:
            if not isinstance(rb, dict):
                continue
            total += 1
            level = str(rb.get("level") or "unknown")
            levels.setdefault(level, {"total": 0, "success": 0})
            levels[level]["total"] += 1
            if bool(rb.get("success")):
                success += 1
                levels[level]["success"] += 1
            else:
                failures.append({"trace": trace, "rollback": rb})
    return total, success, levels, failures


def _success_examples(traces: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    out = []
    for trace in traces:
        if trace.get("prompt_context") and trace.get("status") == "completed":
            out.append(trace)
        if len(out) >= limit:
            break
    return out


def _failure_examples(failures: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    return failures[:limit]


def _trace_section(trace: dict[str, Any], idx: int) -> list[str]:
    lines = [
        f"### Success Case {idx}",
        "",
        f"- task: `{_clean(trace.get('goal'), 260)}`",
        f"- step: {trace.get('step')}",
        f"- trigger: `{trace.get('trigger_reason')}`",
        f"- root activity: `{trace.get('root_activity')}`",
        f"- candidates: {trace.get('candidate_count')}, selected targets: {len(trace.get('selected_targets') or [])}",
        "",
    ]
    image = _image(str(trace.get("root_screenshot") or ""), f"success case {idx} root")
    if image:
        lines.extend([image, ""])
    lines.extend(
        [
            "Candidate ranking:",
            "",
            _candidate_table(list(trace.get("candidates") or [])),
            "",
            "Selected prompt information:",
            "",
            "```text",
            str(trace.get("prompt_context") or ""),
            "```",
            "",
            "Root a11y excerpt:",
            "",
            "```json",
            _a11y_excerpt(list(trace.get("root_a11y") or [])),
            "```",
            "",
        ]
    )
    for obs in list(trace.get("observations") or [])[:2]:
        lines.append(f"Observation branch {obs.get('branch_id')}: labels={obs.get('labels')}")
        for step in list(obs.get("steps") or [])[:2]:
            screenshot = _image(str(step.get("screenshot") or ""), f"branch {obs.get('branch_id')} depth {step.get('depth')}")
            if screenshot:
                lines.extend([screenshot, ""])
            lines.extend(
                [
                    f"- depth: {step.get('depth')}, changed: {step.get('changed')}, activity: `{step.get('after_activity')}`",
                    f"- observed elements: {_clean('; '.join(step.get('observed_elements') or []), 260)}",
                    "",
                ]
            )
    return lines


def _failure_section(item: dict[str, Any], idx: int) -> list[str]:
    trace = item["trace"]
    rb = item["rollback"]
    analysis = rb.get("failure_analysis") if isinstance(rb.get("failure_analysis"), dict) else {}
    lines = [
        f"### Rollback Failure Case {idx}",
        "",
        f"- task: `{_clean(trace.get('goal'), 260)}`",
        f"- step: {trace.get('step')}",
        f"- mode: `{rb.get('mode')}`, level: `{rb.get('level')}`",
        f"- matched_by: `{rb.get('matched_by')}`",
        f"- likely reasons: `{', '.join(analysis.get('likely_reasons') or []) or 'unknown'}`",
        f"- root activity: `{analysis.get('root_activity')}`",
        f"- final activity: `{analysis.get('final_activity')}`",
        f"- hash diff: `{analysis.get('hash_diff')}`",
        "",
    ]
    image = _image(str(trace.get("root_screenshot") or ""), f"failure case {idx} root")
    if image:
        lines.extend([image, ""])
    lines.extend(
        [
            "Replay actions:",
            "",
            "```json",
            json.dumps(rb.get("replay_action_types") or [], ensure_ascii=False, indent=2),
            "```",
            "",
            "Final semantic summary:",
            "",
            "```json",
            json.dumps(analysis.get("final_semantic_summary") or [], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return lines


def generate_report(trace_root: Path, out_path: Path) -> None:
    traces = _load_traces(trace_root)
    rollback_total, rollback_success, levels, failures = _rollback_rows(traces)
    triggered = [t for t in traces if t.get("status") in {"completed", "rollback_failed"}]
    prompt_traces = [t for t in traces if t.get("prompt_context")]

    lines = [
        "# MobileExplorer Synchronous Exploration Report",
        "",
        "## Overview",
        "",
        f"- trace root: `{trace_root.resolve()}`",
        f"- exploration steps: {len(traces)}",
        f"- triggered exploration steps: {len(triggered)}",
        f"- prompt-injected exploration steps: {len(prompt_traces)}",
        f"- rollback success: {_rate(rollback_success, rollback_total)}",
    ]
    for level, row in sorted(levels.items()):
        lines.append(f"- rollback {level}: {_rate(row['success'], row['total'])}")
    lines.extend(
        [
            "",
            "## What The Code Does",
            "",
            "The agent now runs synchronous exploration before VLM reasoning. It ranks clickable a11y nodes with a lightweight hashed text-embedding score plus lexical overlap, penalizes previously probed nodes, probes a bounded number of branches, performs Level-1 backtracking rollback, falls back to Level-2 home-and-replay recovery, and injects only rollback-verified observations into the prompt.",
            "",
            "## Successful Exploration Examples",
            "",
        ]
    )
    successes = _success_examples(traces)
    if successes:
        for idx, trace in enumerate(successes, start=1):
            lines.extend(_trace_section(trace, idx))
    else:
        lines.append("No prompt-injected successful exploration traces were found.")
        lines.append("")
    lines.extend(["## Rollback Failure Examples", ""])
    fail_examples = _failure_examples(failures)
    if fail_examples:
        for idx, item in enumerate(fail_examples, start=1):
            lines.extend(_failure_section(item, idx))
    else:
        lines.append("No rollback failures were found in the collected traces.")
        lines.append("")
    lines.extend(
        [
            "## Design Issues And Next Improvements",
            "",
            "- If Level-2 replay fails, the most likely causes are incomplete replay traces, non-idempotent text/data actions, dialogs/keyboards that change back behavior, or dynamic UI state drift.",
            "- This run exposed low-value probe targets such as `Navigate up`, launcher search, and `Google Lens`; the current agent filters these structural/system controls and requires secondary probes to have non-trivial task relevance.",
            "- The current implementation uses local hashed text embeddings to avoid a runtime model dependency. Replacing this with an on-device MiniLM-style embedding model would better match the paper's semantic ranking.",
            "- Synchronous exploration is easier to reason about and avoids action races, but it does not hide latency inside VLM reasoning. If server latency grows, reintroduce a worker thread with a strict join-before-action barrier.",
            "- Prompt injection is intentionally compact. If useful results are missed, raise `light_explore_prompt_result_limit` or include more observed a11y labels per branch.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exploration report from traces.")
    parser.add_argument(
        "--trace-root",
        default="./output/explorer_agent_gelab_light",
        help="Directory containing task subdirectories with exploration_trace.jsonl.",
    )
    parser.add_argument(
        "--out",
        default="./results/exploration_design_report.md",
        help="Output Markdown path.",
    )
    args = parser.parse_args()
    generate_report(Path(args.trace_root), Path(args.out))


if __name__ == "__main__":
    main()
