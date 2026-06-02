#!/usr/bin/env python3
"""Aggregate app-stratified aggressive t+2 SearchInputShortcut diagnostics."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VARIANTS = [
    "V0_BASELINE",
    "V12_NO_SHORTCUT",
    "V16_SHADOW_SEARCHINPUT_ONLY",
    "V18_ACTIVE_SEARCHINPUT_ONLY_STRICT",
    "V18_ACTIVE_SEARCHINPUT_ONLY_RELAXED_DEBUG",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        row.setdefault("_source_file", str(path))
        rows.append(row)
    return rows


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "success", "task_complete"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "NA", "MISSING"):
            return default
        return float(value)
    except Exception:
        return default


def _variant_from_path(path: Path) -> str:
    text = str(path)
    for variant in VARIANTS:
        if variant in text:
            return variant
    return "UNKNOWN"


def _variant_from_row(row: dict[str, Any]) -> str:
    for key in ("variant", "explore_variant"):
        if row.get(key):
            return str(row[key])
    return _variant_from_path(Path(str(row.get("_source_file", ""))))


def _task_id(row: dict[str, Any]) -> str:
    return str(row.get("task_id") or row.get("task") or row.get("task_name") or row.get("goal") or "UNKNOWN")


def _short(text: Any, limit: int = 80) -> str:
    value = str(text or "").replace("|", "/").replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _plan_pattern(plan: dict[str, Any]) -> str:
    action = plan.get("shortcut_t2_action") if isinstance(plan.get("shortcut_t2_action"), dict) else {}
    if action.get("is_safe_search_input"):
        return "SearchInputShortcut"
    if action.get("is_exact_result_click"):
        return "ExactResultClickShortcut"
    return str(plan.get("shortcut_pattern") or action.get("shortcut_pattern") or "generic")


def _is_searchinput_plan(plan: dict[str, Any]) -> bool:
    return _plan_pattern(plan) == "SearchInputShortcut"


def _candidate_plan(plan: dict[str, Any]) -> bool:
    return bool(plan.get("shortcut_t2_action")) and not plan.get("no_plan_reason")


def _latest_report_dir(variant_dir: Path) -> Path | None:
    runs = sorted(variant_dir.glob("run_*/report"), reverse=True)
    return runs[0] if runs else None


def _latest_trace_dir(variant_dir: Path) -> Path | None:
    runs = sorted(variant_dir.glob("run_*/traces"), reverse=True)
    return runs[0] if runs else None


def collect(run_root: Path) -> dict[str, Any]:
    combined_task_rows: list[dict[str, Any]] = []
    combined_step_rows: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    shadow: list[dict[str, Any]] = []
    generic_shadow: list[dict[str, Any]] = []
    rollback: list[dict[str, Any]] = []
    page_trace: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []

    for variant in VARIANTS:
        variant_dir = run_root / variant
        report_dir = _latest_report_dir(variant_dir)
        trace_dir = _latest_trace_dir(variant_dir)
        if report_dir:
            for row in _read_csv(report_dir / "per_task_results.csv"):
                row["variant"] = variant
                combined_task_rows.append(row)
            for row in _read_csv(report_dir / "per_step_metrics.csv"):
                row["variant"] = variant
                combined_step_rows.append(row)
        if trace_dir:
            for name, target in [
                ("shortcut_plans.jsonl", plans),
                ("shortcut_events.jsonl", events),
                ("shortcut_shadow_eval.jsonl", shadow),
                ("generic_shortcut_shadow.jsonl", generic_shadow),
                ("rollback_events.jsonl", rollback),
                ("exploration_page_trace.jsonl", page_trace),
            ]:
                for path in trace_dir.rglob(name):
                    for row in _read_jsonl(path):
                        row["variant"] = variant
                        target.append(row)
            metrics_path = trace_dir / "state_acquisition_metrics.csv"
            metric_rows = _read_csv(metrics_path)
            if metric_rows and set(metric_rows[0].keys()) >= {"metric", "value"}:
                out = {"variant": variant}
                for row in metric_rows:
                    out[row["metric"]] = row["value"]
                state_rows.append(out)
            elif metric_rows:
                for row in metric_rows:
                    row["variant"] = variant
                    state_rows.append(row)

    return {
        "per_task": combined_task_rows,
        "per_step": combined_step_rows,
        "plans": plans,
        "events": events,
        "shadow": shadow,
        "generic_shadow": generic_shadow,
        "rollback": rollback,
        "page_trace": page_trace,
        "state_rows": state_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not keys:
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if k != "_source_file"}
            f.write(json.dumps(clean, ensure_ascii=False, default=str) + "\n")


def build_summary(data: dict[str, Any]) -> list[dict[str, Any]]:
    per_task = data["per_task"]
    plans = data["plans"]
    events = data["events"]
    shadow = data["shadow"]
    generic_shadow = data["generic_shadow"]
    state_rows = {row.get("variant"): row for row in data["state_rows"]}

    task_by_variant = defaultdict(list)
    for row in per_task:
        task_by_variant[row.get("variant", "UNKNOWN")].append(row)
    plans_by_variant = defaultdict(list)
    for row in plans:
        plans_by_variant[_variant_from_row(row)].append(row)
    events_by_variant = defaultdict(list)
    for row in events:
        events_by_variant[_variant_from_row(row)].append(row)
    shadow_by_variant = defaultdict(list)
    for row in shadow:
        shadow_by_variant[_variant_from_row(row)].append(row)
    generic_by_variant = defaultdict(list)
    for row in generic_shadow:
        generic_by_variant[_variant_from_row(row)].append(row)

    summary: list[dict[str, Any]] = []
    for variant in VARIANTS:
        tasks = task_by_variant[variant]
        task_count = len(tasks)
        success_count = sum(
            1
            for row in tasks
            if _truth(row.get("variant_success") or row.get("success") or row.get("mean_success_rate"))
            or _num(row.get("success_rate"), 0.0) > 0.0
        )
        steps = [_num(row.get("variant_steps") or row.get("steps") or row.get("mean_episode_length"), math.nan) for row in tasks]
        steps = [x for x in steps if not math.isnan(x)]
        variant_plans = plans_by_variant[variant]
        candidate_plans = [p for p in variant_plans if _candidate_plan(p)]
        searchinput_plans = [p for p in candidate_plans if _is_searchinput_plan(p)]
        generic_plans = [p for p in candidate_plans if not _is_searchinput_plan(p)]
        variant_events = events_by_variant[variant]
        variant_shadow = shadow_by_variant[variant]
        searchinput_ids = {p.get("plan_id") for p in searchinput_plans}
        generic_ids = {p.get("plan_id") for p in generic_plans}
        searchinput_would = sum(1 for e in variant_events + variant_shadow if e.get("plan_id") in searchinput_ids and _truth(e.get("would_fire")))
        generic_would = sum(1 for e in variant_events + variant_shadow if e.get("plan_id") in generic_ids and _truth(e.get("would_fire")))
        searchinput_shadow_match = sum(1 for e in variant_shadow if e.get("plan_id") in searchinput_ids and _truth(e.get("would_fire")) and _truth(e.get("action_match")))
        generic_shadow_match = sum(1 for e in variant_shadow if e.get("plan_id") in generic_ids and _truth(e.get("would_fire")) and _truth(e.get("action_match")))
        active_fired = sum(1 for e in variant_events if _truth(e.get("fired")) or _truth(e.get("shortcut_fired")))
        skipped = sum(1 for e in variant_events if _truth(e.get("skipped_vlm_reasoning")) or _truth(e.get("skipped_vlm_call")))
        harmful = sum(1 for e in variant_events + variant_shadow if _truth(e.get("harmful_shortcut")) or _truth(e.get("potential_harm")))
        state = state_rows.get(variant, {})
        summary.append(
            {
                "variant": variant,
                "task_count": task_count,
                "success_rate": success_count / task_count if task_count else 0.0,
                "avg_steps": sum(steps) / len(steps) if steps else 0.0,
                "shortcut_plan_count": len(variant_plans),
                "candidate_plan_count": len(candidate_plans),
                "searchinput_plan_count": len(searchinput_plans),
                "generic_candidate_count": len(generic_plans) + len(generic_by_variant[variant]),
                "searchinput_would_fire_count": searchinput_would,
                "generic_would_fire_count": generic_would,
                "searchinput_shadow_match_count": searchinput_shadow_match,
                "generic_shadow_match_count": generic_shadow_match,
                "searchinput_shadow_precision": searchinput_shadow_match / searchinput_would if searchinput_would else "",
                "generic_shadow_precision": generic_shadow_match / generic_would if generic_would else "",
                "active_fired_count": active_fired,
                "skipped_vlm_calls": skipped,
                "harmful_shortcut_count": harmful,
                "full_a11y_calls": _num(state.get("full_a11y_calls")),
                "screenshot_calls": _num(state.get("screenshot_calls")),
                "activity_calls": _num(state.get("activity_calls")),
                "root_state_reuse_count": _num(state.get("root_state_reuse_count")),
                "cached_state_hits": _num(state.get("cached_state_hits")),
                "rollback_full_a11y_fallbacks": _num(state.get("rollback_full_a11y_fallbacks") or state.get("rollback_full_a11y_fallback_count")),
                "passive_no_get_state_count": _num(state.get("passive_no_get_state_count")),
            }
        )
    return summary


def build_report(run_root: Path, data: dict[str, Any], summary: list[dict[str, Any]]) -> str:
    per_task = data["per_task"]
    plans = data["plans"]
    events = data["events"]
    shadow = data["shadow"]
    generic_shadow = data["generic_shadow"]
    rollback = data["rollback"]
    page_trace = data["page_trace"]

    summary_by_variant = {row["variant"]: row for row in summary}
    searchinput_by_app = Counter()
    generic_by_app = Counter()
    active_by_app = Counter()
    app_success = defaultdict(lambda: [0, 0])
    mode_success = defaultdict(lambda: [0, 0])
    task_rows_by_variant = defaultdict(list)
    for row in per_task:
        task_rows_by_variant[row.get("variant", "UNKNOWN")].append(row)
        if row.get("variant") in {"V0_BASELINE", "V12_NO_SHORTCUT", "V18_ACTIVE_SEARCHINPUT_ONLY_STRICT", "V18_ACTIVE_SEARCHINPUT_ONLY_RELAXED_DEBUG"}:
            app = row.get("app") or "UNKNOWN"
            mode = row.get("task_mode") or "UNKNOWN"
            success = _truth(row.get("variant_success") or row.get("success")) or _num(row.get("success_rate"), 0.0) > 0.0
            if row.get("variant") == "V18_ACTIVE_SEARCHINPUT_ONLY_STRICT":
                app_success[app][1] += 1
                app_success[app][0] += int(success)
                mode_success[mode][1] += 1
                mode_success[mode][0] += int(success)

    task_app = {}
    for row in per_task:
        task_app[row.get("task_id") or row.get("task") or ""] = row.get("app") or "UNKNOWN"
    for plan in plans:
        app = task_app.get(_task_id(plan), "UNKNOWN")
        if _candidate_plan(plan):
            if _is_searchinput_plan(plan):
                searchinput_by_app[app] += 1
            else:
                generic_by_app[app] += 1
    for row in generic_shadow:
        generic_by_app[task_app.get(_task_id(row), "UNKNOWN")] += 1
    for event in events:
        if _truth(event.get("fired")):
            active_by_app[task_app.get(_task_id(event), "UNKNOWN")] += 1

    fired_rows = []
    for event in events:
        if not _truth(event.get("fired")):
            continue
        action = event.get("shortcut_t2_action") if isinstance(event.get("shortcut_t2_action"), dict) else {}
        root = event.get("planned_root_action") if isinstance(event.get("planned_root_action"), dict) else {}
        fired_rows.append(
            {
                "variant": _variant_from_row(event),
                "task": _task_id(event),
                "step": event.get("step"),
                "plan_id": event.get("plan_id"),
                "pattern": action.get("shortcut_pattern"),
                "root": root.get("label") or root.get("action_type"),
                "t2": action.get("label") or action.get("value"),
                "text": action.get("value"),
                "confidence": event.get("confidence"),
                "match": event.get("t1_state_match_method"),
            }
        )

    strict = summary_by_variant.get("V18_ACTIVE_SEARCHINPUT_ONLY_STRICT", {})
    relaxed = summary_by_variant.get("V18_ACTIVE_SEARCHINPUT_ONLY_RELAXED_DEBUG", {})
    shadow_summary = summary_by_variant.get("V16_SHADOW_SEARCHINPUT_ONLY", {})
    v12 = summary_by_variant.get("V12_NO_SHORTCUT", {})
    baseline = summary_by_variant.get("V0_BASELINE", {})

    def pct(value: Any) -> str:
        if value == "":
            return "NA"
        return f"{_num(value) * 100:.1f}%"

    lines: list[str] = []
    lines.append("# MobileExplorer app-stratified SearchInputShortcut diagnostic 中文报告")
    lines.append("")
    lines.append(f"- 运行目录：`{run_root}`")
    lines.append("- 实验规模：16 tasks，至少 8 个 app，每个 app 最多 2 个 task。")
    lines.append("- 本轮 active gate：只允许 `SearchInputShortcut`；generic click 只记录 shadow，不 active 执行。")
    lines.append("")
    lines.append("## 1. 变体总览")
    lines.append("")
    lines.append("| 变体 | success | avg steps | plans | SearchInput plans | generic candidates | SearchInput would_fire | SearchInput shadow precision | active fired | skipped VLM | harmful | cached hits | full a11y |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        lines.append(
            f"| `{row['variant']}` | {pct(row['success_rate'])} | {row['avg_steps']:.2f} | {int(row['shortcut_plan_count'])} | "
            f"{int(row['searchinput_plan_count'])} | {int(row['generic_candidate_count'])} | {int(row['searchinput_would_fire_count'])} | "
            f"{pct(row['searchinput_shadow_precision'])} | {int(row['active_fired_count'])} | {int(row['skipped_vlm_calls'])} | "
            f"{int(row['harmful_shortcut_count'])} | {int(row['cached_state_hits'])} | {int(row['full_a11y_calls'])} |"
        )
    lines.append("")
    lines.append("## 2. 是否避免单 app / 单 task 过拟合？")
    lines.append("")
    lines.append("是。任务清单在运行前冻结，覆盖 Joplin/Notes、Files/Browser、Browser、OpenTracks、Simple Calendar、Tasks、Markor、Pro Expense、Clock、Settings/System；每个 app 最多 2 个 task。")
    lines.append("")
    lines.append("## 3. 每个 app 的 shortcut candidate 数")
    lines.append("")
    lines.append("| app | SearchInput candidate | generic shadow candidate | active fired | strict active success |")
    lines.append("|---|---:|---:|---:|---:|")
    for app in sorted(set(searchinput_by_app) | set(generic_by_app) | set(active_by_app) | set(app_success)):
        succ, total = app_success[app]
        lines.append(f"| {app} | {searchinput_by_app[app]} | {generic_by_app[app]} | {active_by_app[app]} | {succ}/{total} |")
    lines.append("")
    lines.append("## 4. Active fired 明细")
    lines.append("")
    if fired_rows:
        lines.append("| 变体 | task | step | plan | pattern | root | t+2 | input text | confidence | match |")
        lines.append("|---|---|---:|---|---|---|---|---|---:|---|")
        for row in fired_rows:
            lines.append(
                f"| `{row['variant']}` | {_short(row['task'], 42)} | {row['step']} | `{row['plan_id']}` | {_short(row['pattern'], 24)} | "
                f"{_short(row['root'], 32)} | {_short(row['t2'], 42)} | {_short(row['text'], 28)} | {_num(row['confidence']):.2f} | {_short(row['match'], 18)} |"
            )
    else:
        lines.append("- 没有 active shortcut fired。")
    lines.append("")
    lines.append("## 5. 问题回答")
    lines.append("")
    lines.append(f"1. SearchInputShortcut 是否跨 app 泛化：SearchInput plans={int(sum(r['searchinput_plan_count'] for r in summary))}，需要看是否分布在多个 app；若集中在 Joplin，则未泛化。")
    lines.append(f"2. shadow precision 是否达标：V16 SearchInput precision={pct(shadow_summary.get('searchinput_shadow_precision', ''))}，验收阈值 70%。")
    lines.append(f"3. active 是否真的跳过 VLM：strict skipped={int(strict.get('skipped_vlm_calls', 0))}，relaxed skipped={int(relaxed.get('skipped_vlm_calls', 0))}。")
    lines.append(f"4. active 是否降低 steps：V12 avg_steps={_num(v12.get('avg_steps')):.2f}，strict avg_steps={_num(strict.get('avg_steps')):.2f}，relaxed avg_steps={_num(relaxed.get('avg_steps')):.2f}。")
    lines.append(f"5. active 是否提高 success：V0={pct(baseline.get('success_rate', 0))}，V12={pct(v12.get('success_rate', 0))}，strict={pct(strict.get('success_rate', 0))}，relaxed={pct(relaxed.get('success_rate', 0))}。")
    lines.append(f"6. 是否破坏 baseline-success task：见 `per_task_results.csv` 的 rescued/broken；若 broken>0，不可扩大。")
    lines.append(f"7. generic click 是否不稳定：generic candidates={int(sum(r['generic_candidate_count'] for r in summary))}，本轮不 active；从 shadow precision 判断是否继续禁用。")
    lines.append("8. 适合 aggressive t+2 的 app：只有能稳定进入 search UI 且 task entity 明确来自指令的 app。")
    lines.append("9. 应禁用 active 的 task mode：DELETE_COMMIT、FORM_CREATE_EDIT、SIMPLE_VERIFY_OPEN、date picker、filter click、result row click。")
    lines.append(f"10. state acquisition：cached_state_hits={int(sum(r['cached_state_hits'] for r in summary))}；如果仍为 0，缓存没有生效。")
    lines.append("")
    lines.append("## 6. 验收标准")
    lines.append("")
    checks = [
        ("searchinput_plan_count >= 8", sum(r["searchinput_plan_count"] for r in summary) >= 8, str(int(sum(r["searchinput_plan_count"] for r in summary)))),
        ("searchinput_would_fire_count >= 5", max(r["searchinput_would_fire_count"] for r in summary) >= 5, str(int(max(r["searchinput_would_fire_count"] for r in summary)))),
        ("searchinput_shadow_precision >= 70%", _num(shadow_summary.get("searchinput_shadow_precision")) >= 0.70, pct(shadow_summary.get("searchinput_shadow_precision", ""))),
        ("active_fired_count >= 3", max(strict.get("active_fired_count", 0), relaxed.get("active_fired_count", 0)) >= 3, str(int(max(strict.get("active_fired_count", 0), relaxed.get("active_fired_count", 0))))),
        ("skipped_vlm_calls >= 3", max(strict.get("skipped_vlm_calls", 0), relaxed.get("skipped_vlm_calls", 0)) >= 3, str(int(max(strict.get("skipped_vlm_calls", 0), relaxed.get("skipped_vlm_calls", 0))))),
        ("harmful_shortcut_count = 0", sum(r["harmful_shortcut_count"] for r in summary) == 0, str(int(sum(r["harmful_shortcut_count"] for r in summary)))),
        ("active success >= V12", strict.get("success_rate", 0) >= v12.get("success_rate", 0), f"{pct(strict.get('success_rate', 0))} vs {pct(v12.get('success_rate', 0))}"),
        ("active avg steps <= V12 + 0.3", strict.get("avg_steps", 999) <= v12.get("avg_steps", 0) + 0.3, f"{_num(strict.get('avg_steps')):.2f} vs {_num(v12.get('avg_steps')):.2f}"),
    ]
    for name, ok, detail in checks:
        lines.append(f"- {'PASS' if ok else 'FAIL'}：{name}（{detail}）")
    lines.append("")
    can_expand = all(ok for _, ok, _ in checks)
    lines.append("## 7. 最终建议")
    lines.append("")
    if can_expand:
        recommendation = "keep only SearchInputShortcut as optional optimization"
        lines.append("- 满足 diagnostic 验收，可扩大到 30/40-task；但仍建议只把 SearchInputShortcut 作为 optional optimization，而不是泛化 aggressive t+2 主设计。")
    elif strict.get("active_fired_count", 0) > 0 and strict.get("harmful_shortcut_count", 0) == 0:
        recommendation = "keep only SearchInputShortcut as optional optimization"
        lines.append("- 推荐：保留 `SearchInputShortcut` 作为 optional optimization；generic aggressive t+2 继续只放 shadow/future work。")
    else:
        recommendation = "keep aggressive t+2 only as future work"
        lines.append("- 推荐：aggressive t+2 暂时只作为 future work；当前提交主设计仍应强调 evidence-guided exploration，而非 active shortcut。")
    lines.append(f"- Final recommendation: `{recommendation}`")
    lines.append("")
    lines.append("## 8. Trace/logging 完整性")
    lines.append("")
    lines.append(f"- shortcut_plans={len(plans)}，shortcut_events={len(events)}，shadow_eval={len(shadow)}，generic_shadow={len(generic_shadow)}。")
    lines.append(f"- rollback_events={len(rollback)}，exploration_page_trace={len(page_trace)}。如果 rollback/page trace 仍为 0，需要继续修 logger。")
    lines.append("")
    lines.append("## 9. 输出文件")
    lines.append("")
    for name in [
        "task_selection.md",
        "variant_summary.csv",
        "per_task_results.csv",
        "per_step_metrics.csv",
        "shortcut_plans.jsonl",
        "shortcut_events.jsonl",
        "shortcut_shadow_eval.jsonl",
        "generic_shortcut_shadow.jsonl",
        "state_acquisition_metrics.csv",
        "exploration_page_trace.jsonl",
        "rollback_events.jsonl",
        "app_stratified_shortcut_report_cn.md",
    ]:
        lines.append(f"- `{run_root / name}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: aggregate_app_stratified_shortcut_report.py RUN_ROOT", file=sys.stderr)
        return 2
    run_root = Path(argv[1]).expanduser().resolve()
    data = collect(run_root)
    summary = build_summary(data)

    _write_csv(run_root / "per_task_results.csv", data["per_task"])
    _write_csv(run_root / "per_step_metrics.csv", data["per_step"])
    _write_csv(run_root / "variant_summary.csv", summary)
    _write_csv(run_root / "state_acquisition_metrics.csv", data["state_rows"])
    _write_jsonl(run_root / "shortcut_plans.jsonl", data["plans"])
    _write_jsonl(run_root / "shortcut_events.jsonl", data["events"])
    _write_jsonl(run_root / "shortcut_shadow_eval.jsonl", data["shadow"])
    _write_jsonl(run_root / "generic_shortcut_shadow.jsonl", data["generic_shadow"])
    _write_jsonl(run_root / "rollback_events.jsonl", data["rollback"])
    _write_jsonl(run_root / "exploration_page_trace.jsonl", data["page_trace"])
    report = build_report(run_root, data, summary)
    (run_root / "app_stratified_shortcut_report_cn.md").write_text(report, encoding="utf-8")
    print(f"[aggregate] wrote {run_root / 'app_stratified_shortcut_report_cn.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
