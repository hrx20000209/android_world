#!/usr/bin/env python3
"""Phase 0 instrumentation check for decoupled MobileExplorer diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "android_world" / "agents" / "explorer_agent_gelab_light.py"
REPORT = ROOT / "scripts" / "run_exploration_experiment_report.py"
OUT = ROOT / "results" / "decoupled_instrumentation_phase0" / "instrumentation_check_report_cn.md"


def has(text: str, needle: str) -> bool:
    return needle in text


def main() -> int:
    agent = AGENT.read_text(encoding="utf-8", errors="ignore")
    report = REPORT.read_text(encoding="utf-8", errors="ignore")
    checks = []
    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check("decoupled timing env", has(agent, "ANDROID_WORLD_EXPLORATION_TIMING"))
    check("parallel_shadow default", has(agent, '"parallel_shadow"'))
    check("pre reasoning decoupled timing", has(agent, '"pre_reasoning_decoupled"'))
    check("invalid planned-action timing", has(agent, '"invalid_post_planned_action"'))
    check("decoupled exploration env", has(agent, "ANDROID_WORLD_DECOUPLED_EXPLORATION"))
    check("current action usage env", has(agent, "ANDROID_WORLD_LIGHT_EXPLORE_USE_CURRENT_ACTION"))
    check("planning text usage env", has(agent, "ANDROID_WORLD_LIGHT_EXPLORE_USE_PLANNING_TEXT"))
    check("planned action invalid flag", has(agent, '"invalid_run"'))
    check("current planned action disabled when decoupled", has(agent, "current_action = None"))
    check("current VLM text disabled when decoupled", has(agent, 'planning_text = ""'))
    check("same step injection field", has(agent, "evidence_injected_same_step"))
    check("score formula components", all(has(agent, k) for k in [
        "MissingSlotGain", "TaskEntityMatch", "OperatorPriority", "TaskProgress",
        "Novelty", "RiskPenalty", "RollbackCost", "WrongScreenRolePenalty", "RevisitPenalty",
    ]))
    check("operator stratified strategy", has(agent, "Operator-Stratified Best-First Exploration"))
    check("candidate scores artifact", has(agent, "candidate_scores.jsonl"))
    check("candidate quality filters", has(agent, "_candidate_quality_filter_reason"))
    check("min attempts env", has(agent, "ANDROID_WORLD_LIGHT_EXPLORE_MIN_ATTEMPTS_PER_STEP"))
    check("evidence decisions artifact", has(agent, "evidence_decisions.jsonl"))
    check("evidence confidence formula", has(agent, "_evidence_confidence_components"))
    check("relaxed diagnostic injection", has(agent, "ANDROID_WORLD_LIGHT_EXPLORE_RELAXED_DIAGNOSTIC_INJECTION"))
    check("prompt hints artifact", has(agent, "prompt_hints.jsonl"))
    check("exploration latency artifact", has(agent, "exploration_latency.jsonl"))
    check("rollback events artifact", has(agent, "rollback_events.jsonl"))
    check("rollback level2 trigger reason", has(agent, "level2_trigger_reason"))
    check("runtime config artifact", has(agent, "runtime_config.json"))
    check("report loads diagnostics", has(report, "_load_diagnostic_families"))
    check("report loads candidate scores", has(report, "candidate_scores.jsonl"))
    check("report hard validation section", has(report, "Diagnostic / Hard Validation"))
    check("report prompt injection diagnosis", has(report, "Prompt Injection Diagnosis"))
    check("report candidate quality section", has(report, "Candidate Quality Filtering"))
    check("report search strategy section", has(report, "## Search Strategy"))
    check("report p95 latency", has(report, "p95 exploration total"))
    check("CLI min attempts", has(report, "--explore_min_attempts_per_step"))
    check("CLI exploration timing", has(report, "--exploration_timing"))
    check("CLI shadow only", has(report, "--explore_shadow_only"))
    check("CLI relaxed diagnostic injection", has(report, "--explore_relaxed_diagnostic_injection"))

    passed = sum(1 for c in checks if c["ok"])
    failed = [c for c in checks if not c["ok"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 0 instrumentation check",
        "",
        f"- checks passed: `{passed}/{len(checks)}`",
        f"- status: `{'PASS' if not failed else 'FAIL'}`",
        "",
        "## Checks",
        "",
        "| item | status | detail |",
        "|---|---:|---|",
    ]
    for c in checks:
        lines.append(f"| {c['name']} | {'PASS' if c['ok'] else 'FAIL'} | `{c['detail']}` |")
    lines.extend([
        "",
        "## Design statement",
        "",
        "- Decoupled mode uses `ANDROID_WORLD_EXPLORATION_TIMING=parallel_shadow` by default.",
        "- Current-step exploration must not read current VLM planned action when decoupled.",
        "- Exploration result is logged for t+1 prompt matching via prompt/evidence diagnostic artifacts.",
        "- Search strategy is Operator-Stratified Best-First Exploration: root stratified BFS, branch gated DFS to depth 2, ranking by explicit score components.",
        "",
    ])
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(str(OUT))
    print(json.dumps({"passed": passed, "total": len(checks), "failed": failed}, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
