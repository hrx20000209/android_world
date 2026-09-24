#!/usr/bin/env python3
"""Generate a trace-and-screenshot report for E regressions vs 4B_2."""

import argparse
import json
import re
from pathlib import Path


def read_json(path, default=None):
  try:
    return json.loads(path.read_text())
  except (OSError, json.JSONDecodeError):
    return default


def read_jsonl(path):
  rows = []
  try:
    for line in path.read_text().splitlines():
      try:
        rows.append(json.loads(line))
      except json.JSONDecodeError:
        pass
  except OSError:
    pass
  return rows


def baseline_rows(path, task_names):
  found = {}
  for line in path.read_text().splitlines():
    fields = line.split()
    if fields and fields[0] in task_names and len(fields) == 8:
      try:
        found[fields[0]] = {
            "success": float(fields[3]),  # mean_success_rate
            "steps": float(fields[4]),  # mean_episode_length
        }
      except ValueError:
        continue
  return found


def md_cell(value):
  return str(value or "").replace("|", "\\|").replace("\n", " ")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--experiment", type=Path, required=True)
  parser.add_argument("--baseline", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  summary = read_json(args.experiment / "summary.json", {})
  arm = summary["arms"][0]
  tasks = arm["tasks"]
  baseline = baseline_rows(args.baseline, {t["task"] for t in tasks})
  evaluated = [t for t in tasks if t.get("success") is not None]
  unknown_count = len(tasks) - len(evaluated)
  e_success_count = sum(t["success"] >= .5 for t in evaluated)
  paired = [(t, baseline[t["task"]]) for t in evaluated if t["task"] in baseline]
  paired_baseline_success_count = sum(row["success"] >= .5 for _, row in paired)
  paired_e_success_count = sum(task["success"] >= .5 for task, _ in paired)
  paired_both_success = sum(
      task["success"] >= .5 and row["success"] >= .5 for task, row in paired)
  paired_baseline_only = sum(
      task["success"] < .5 and row["success"] >= .5 for task, row in paired)
  paired_e_only = sum(
      task["success"] >= .5 and row["success"] < .5 for task, row in paired)
  baseline_step_mean = (
      sum(row["steps"] for _, row in paired) / len(paired) if paired else 0.0
  )
  e_step_mean = (
      sum(task["steps"] for task, _ in paired) / len(paired) if paired else 0.0
  )
  regressions = [
      t for t in evaluated
      if baseline.get(t["task"], {}).get("success", 0) >= .5 and t["success"] < .5
  ]
  graph = read_json(
      args.experiment / "E_FULL_MOBILEEXPLORER" / "executable_memory.json", {})
  memory_metrics = graph.get("metrics", {}) if isinstance(graph, dict) else {}
  stop_reasons = memory_metrics.get("stop_reasons", {})
  output = args.output or args.experiment / "failure_cases_vs_4B_2.md"
  lines = [
    "# MobileExplorer：相对 4B_2 的失败案例与中间数据",
    "",
    f"实验目录：`{args.experiment}`。筛选规则：4B_2 `mean_success_rate >= 0.5` 且 E 本轮 `success < 0.5`。",
    "本报告展示所有逐任务回归；baseline 本身失败的任务不列作 regression。截图及完整 JSONL 保留在任务目录，图只从 arm 根目录的唯一 executable_memory.json 读取。",
    "",
    f"E：平均 evaluator 分数 {arm['success_rate']:.1%}（{len(evaluated)} evaluated；{unknown_count} infrastructure unknown）；二值成功（score ≥ 0.5）{e_success_count}/{len(evaluated)} = {e_success_count / max(1, len(evaluated)):.1%}。平均步数 {arm['step_mean']:.2f}，中位数 {arm['step_median']:.1f}，P95 {arm['step_p95']:.2f}。",
    f"严格配对的 {len(paired)} 个可评测任务：4B_2 二值成功 {paired_baseline_success_count}/{len(paired)}，E {paired_e_success_count}/{len(paired)}；平均步数 {baseline_step_mean:.2f} → {e_step_mean:.2f}（Δ {e_step_mean - baseline_step_mean:+.2f}）。历史 baseline 未在本轮重跑。",
    f"配对结果：两者成功 {paired_both_success}，仅 4B_2 成功 {paired_baseline_only}，仅 E 成功 {paired_e_only}，两者均未达 0.5 {len(paired) - paired_both_success - paired_baseline_only - paired_e_only}。基线成功→E失败：{len(regressions)} 项。",
    f"E 全量指标：{arm['exploration_probe_count']} probes、{arm['reasoning_count']} reasoning、唯一图 {arm['graph_node_count']} nodes / {arm['graph_edge_count']} edges / {arm['graph_skill_count']} skills、{arm['parse_error_count']} parse errors；图上下文采纳 {arm.get('graph_prompt_adoptions', 0)} 次，route shortcut attempts/hits {arm.get('graph_skip_attempts', 0)}/{arm.get('graph_skip_hits', 0)}。",
    f"5-probe 窗口完成 {memory_metrics.get('probe_rounds_complete', 0)}/{memory_metrics.get('probe_rounds', 0)}（{memory_metrics.get('five_probe_completion_rate', 0.0):.2%}）；累计额外耗时 {memory_metrics.get('extra_time_s', 0.0):.1f}s，恢复失败 {memory_metrics.get('recovery_failures', 0)} 次。提前停止原因：{md_cell(stop_reasons)}。",
    "",
    "## 回归索引",
    "",
    "| Task | 4B_2 success | E steps | probes | graph node/edge |",
    "|---|---:|---:|---:|---:|",
  ]
  for t in regressions:
    lines.append(f"| [{t['task']}](#{t['task'].lower()}) | {baseline[t['task']]['success']:.0%} | {t['steps']} | {t['probe_count']} | {t['graph_node_count']}/{t['graph_edge_count']} |")

  for t in regressions:
    task = t["task"]
    folder = next(args.experiment.glob(f"E_FULL_MOBILEEXPLORER/*_{task}"), None)
    if folder is None:
      continue
    rel = folder.relative_to(output.parent)
    canonical_graph_path = folder.parent / "executable_memory.json"
    graph = read_json(canonical_graph_path, {})
    all_nodes = {n.get("node_id"): n for n in graph.get("states", [])}
    all_edges = graph.get("edges", [])
    probes = read_jsonl(folder / "probe_trace.jsonl")
    traces = read_jsonl(next((folder / "agent_traces").glob("*/action.jsonl"), folder / "missing.jsonl"))
    trace_dir = next((folder / "agent_traces").glob("*"), None)
    app_packages = {
        str(row.get("source_activity") or "").split("/", 1)[0]
        for row in traces if row.get("source_activity")
    }
    for probe in probes:
      for side in ("src", "dst"):
        activity = str((probe.get("graph", {}).get(side) or {}).get("activity") or "")
        if activity:
          app_packages.add(activity.split("/", 1)[0])
    nodes = {
        node_id: node for node_id, node in all_nodes.items()
        if not app_packages or node.get("package") in app_packages
    }
    edges = [edge for edge in all_edges if edge.get("source_node") in nodes]
    lines += ["", f"## {task}", "", f"**E结果**：失败，{t['steps']} reasoning/action steps，{t['probe_count']} probes，{t['reasoning_count']} reasoning calls；graph {t['graph_node_count']} nodes / {t['graph_edge_count']} edges。", ""]
    lines += ["### Reasoning trace 与截图", ""]
    if traces:
      for i, row in enumerate(traces):
        action = row.get("action_dict", {})
        parsed = row.get("parsed_action", {})
        response = re.sub(r"\s+", " ", row.get("response", ""))[:900]
        lines.append(f"#### Call {i + 1} · state `{row.get('state_id', '')}` · {row.get('source_activity', '')}")
        lines.append("")
        context = row.get("executable_memory_prompt_context") or "（本步没有注入图上下文）"
        lines.append(f"Action: `{json.dumps(action, ensure_ascii=False)}`；parse_error: `{md_cell(row.get('parse_error'))}`；graph fusion: `{json.dumps(row.get('executable_memory_fusion', {}), ensure_ascii=False)[:600]}")
        lines.append("")
        lines.append(f"图上下文是否注入：`{bool(row.get('executable_memory_prompt_context_injected'))}`；采纳：`{bool(row.get('executable_memory_prompt_adopted'))}`。")
        lines.append("")
        lines.append(f"```text\n{str(context)[:1600]}\n```")
        lines.append("")
        lines.append(f"<details><summary>原始模型响应</summary>\n\n```text\n{response}\n```\n</details>")
        lines.append("")
        shot = trace_dir / f"screenshot_{i}.png" if trace_dir else None
        if shot and shot.exists():
          lines.append(f'<img src="{shot.relative_to(output.parent)}" alt="{task} reasoning step {i}" style="zoom:25%;">')
          lines.append("")
    else:
      lines.append("未找到 agent action JSONL。")
      lines.append("")
    lines += ["### Exploration：页面、动作、恢复结果", "", "| Reasoning step | Probe | Source page/activity | Action/element | Destination | Recovery | New labels |", "|---:|---:|---|---|---|---|---|"]
    for p in probes:
      elem = p.get("element", {})
      graph_row = p.get("graph", {})
      src = graph_row.get("src", {}).get("activity") or p.get("app_package", "")
      dst = graph_row.get("dst", {}).get("activity") or p.get("discovered", {}).get("reached_activity", "")
      label = elem.get("text") or elem.get("content_desc") or elem.get("resource_id") or p.get("probe_type")
      labels = ", ".join(p.get("discovered", {}).get("new_texts", [])[:8])
      lines.append(f"| {p.get('step_idx','')} | {p.get('probe_idx','')} | {md_cell(src)} | {md_cell(p.get('probe_type'))}: {md_cell(label)} | {md_cell(dst)} | {p.get('recovery_level')} / ok={p.get('recovery_ok')} | {md_cell(labels)} |")
      # `probe_idx` resets for each reasoning window; opportunity screenshots
      # are numbered by evaluator/generation step, so key by step_idx instead.
      shot = trace_dir / f"opportunity_{int(p.get('step_idx', 0)) + 1}.png" if trace_dir else None
      if shot and shot.exists():
        probe_idx = p.get("probe_idx")
        lines += ["", f"Probe screenshot (step {p.get('step_idx')}, probe {probe_idx}):", "", f'<img src="{shot.relative_to(output.parent)}" alt="probe {probe_idx}" style="zoom:25%;">', ""]
    graph_rel = canonical_graph_path.relative_to(output.parent)
    lines += ["", "### 单一权威图（本任务 app 子图）", "", f"源文件：[`executable_memory.json`]({graph_rel})（arm 级唯一持久化图；本节只筛选当前 app package）；节点 {len(nodes)}，边 {len(edges)}。Ex5 回退/安全控制器只在进程内运行，不另存一份 memory graph。", "", "```mermaid", "flowchart LR"]
    for node_id, node in nodes.items():
      short = (node_id or "?")[:8]
      title = (node.get("activity", "?").split("/")[-1] + "\\n" + ", ".join(node.get("semantic_aliases", node.get("landmarks", []))[:3])).replace('"', "'")
      lines.append(f'  {short}["{title}"]')
    for edge in edges:
      src = (edge.get("source_node") or "?")[:8]
      for dst_id in edge.get("target_states", {}):
        if dst_id not in nodes:
          continue
        dst = dst_id[:8]
        label = str(edge.get("action_type", "?")) + ": " + str(
            edge.get("normalized_action_token") or edge.get("function") or "")[:45]
        lines.append(f'  {src} -->|"{label.replace(chr(34), chr(39))}"| {dst}')
    lines += ["```", "", "#### 节点摘要与页面截图", ""]
    screenshot_by_node: dict[str, tuple[Path, float]] = {}
    if trace_dir and traces:
      trace_landmarks = []
      for i, row in enumerate(traces):
        keys = set()
        for item in row.get("source_landmarks", ()):
          parts = str(item).split("|")
          if len(parts) >= 2:
            keys.add(parts[1].casefold())
          keys.update(part.casefold() for part in parts if len(part) > 3)
        trace_landmarks.append((trace_dir / f"screenshot_{i}.png", keys))
      for node_id, node in nodes.items():
        node_keys = set()
        for item in node.get("landmarks", ()):
          parts = str(item).split("|")
          node_keys.update(part.casefold() for part in parts if len(part) > 3)
        for row in node.get("elements", ()):
          selector = row.get("selector") or {}
          node_keys.update(str(selector.get(key) or "").casefold()
                           for key in ("resource_id", "text", "content_desc")
                           if selector.get(key))
        candidates = []
        for order, (shot, keys) in enumerate(trace_landmarks):
          if not shot.exists():
            continue
          union = node_keys | keys
          overlap = len(node_keys & keys) / max(1, len(union))
          candidates.append((overlap, -abs(order - int(node.get("depth", 0))), shot))
        if candidates:
          score, _, shot = max(candidates, key=lambda item: (item[0], item[1]))
          screenshot_by_node[node_id] = (shot, score)
    lines += ["| Node | Activity / depth | visits | semantic aliases |", "|---|---|---:|---|"]
    for node_id, node in nodes.items():
      labels = ", ".join(node.get("semantic_aliases", node.get("landmarks", []))[:8])
      lines.append(f"| `{(node_id or '')[:12]}` | {md_cell(node.get('activity'))} / {node.get('depth', 0)} | {node.get('visit_count', 0)} | {md_cell(labels)} |")
      activity = str(node.get("activity") or "")
      matched = screenshot_by_node.get(node_id)
      if matched:
        shot, similarity = matched
        lines += ["", f"<details><summary>Node `{(node_id or '')[:12]}` screenshot (landmark match {similarity:.2f})：{md_cell(activity)}</summary>", "", f'<img src="{shot.relative_to(output.parent)}" alt="{task} graph state {(node_id or "")[:8]}" style="zoom:25%;">', "", "</details>", ""]
    lines += ["", "#### 图边/路由证据", "", "| Edge | Selector/token | Destination count | support | reversible | confidence | route hit/miss | trap |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for edge in edges:
      destinations = [nodes[key] for key in edge.get("target_states", {}) if key in nodes]
      selector = edge.get("selector") or {}
      desc = selector.get("text") or selector.get("content_desc") or selector.get("resource_id") or edge.get("normalized_action_token")
      total_recovery = int(edge.get("recovery_success_count", 0)) + int(edge.get("recovery_failure_count", 0))
      reversible = int(edge.get("recovery_success_count", 0)) / max(1, total_recovery)
      lines.append(f"| `{str(edge.get('edge_id',''))[:10]}` | {md_cell(desc)} / {md_cell(edge.get('normalized_action_token'))} | {len(destinations)} | {edge.get('support_count', 0)} | {reversible:.2f} | {float(edge.get('alpha',1))/(float(edge.get('alpha',1))+float(edge.get('beta',1))):.2f} | {edge.get('route_hit_count',0)}/{edge.get('route_miss_count',0)} | {edge.get('trap_count',0)} |")
    lines += ["", "#### Probe 原始记录", "", f"[`probe_trace.jsonl`]({rel / 'probe_trace.jsonl'})；完整 reasoning/action 原始记录位于 [`agent_traces/`]({rel / 'agent_traces'})。", ""]

  success_delta = paired_e_success_count - paired_baseline_success_count
  lines += ["", "## 总体诊断（待人工核验）", "", f"所有 failure case 的 trace / screenshots / graph JSON 均可从上方逐任务检查。严格配对集合中 E 与历史 4B_2 的二值成功数差为 {success_delta:+d}；E 有 {unknown_count} 个 infrastructure unknown 已从配对统计中排除。历史对照并非本轮随机配对，不能单独用来归因图记忆效果。图虽检索到路径候选，但本轮 route shortcut attempts/hits 为 {arm.get('graph_skip_attempts', 0)}/{arm.get('graph_skip_hits', 0)}，未贡献可验证的跳步成功。报告中的回归不自动归因为 exploration：请结合每例探测是否发生在目标页面、是否恢复成功、reasoning 输出和图融合拒绝原因判断。", ""]
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text("\n".join(lines))
  print(f"wrote {output} ({len(regressions)} regressions)")


if __name__ == "__main__":
  main()
