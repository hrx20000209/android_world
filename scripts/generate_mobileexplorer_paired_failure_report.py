#!/usr/bin/env python3
"""Create an evidence-rich E vs Ex5 paired-task report."""

import argparse
import html
import json
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
        continue
  except OSError:
    pass
  return rows


def md(value):
  return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def compact(value, limit=700):
  return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:limit]


def task_dir(root, arm, task):
  matches = sorted((root / arm).glob(f"*_{task}"))
  return matches[0] if matches else None


def rel(path, parent):
  return path.relative_to(parent).as_posix()


def image_tag(path, root, alt):
  if not path or not path.exists():
    return ""
  return f'<img src="{html.escape(rel(path, root), quote=True)}" alt="{html.escape(alt, quote=True)}" style="zoom:25%;">'


def trace_dir(folder):
  if folder is None:
    return None
  matches = sorted(folder.glob("agent_traces/*/action.jsonl"))
  return matches[0].parent if matches else None


def render_reasoning(folder, root, arm_label):
  directory = trace_dir(folder)
  if directory is None:
    return ["No reasoning action trace/screenshot was emitted (for example, the task may have failed before its first action).", ""]
  rows = read_jsonl(directory / "action.jsonl")
  out = []
  for index, row in enumerate(rows):
    action = row.get("action_dict", {})
    fusion = row.get("executable_memory_fusion", {})
    out.extend([
      f"#### {arm_label} · reasoning/action {index + 1} · state `{md(row.get('state_id'))}`",
      "",
      f"- Action: `{compact(action, 350)}`",
      f"- Parse error: `{md(row.get('parse_error')) or 'none'}`; fusion: `{compact(fusion, 500)}`",
    ])
    response = str(row.get("response", "")).strip()
    if response:
      out.extend(["", "<details><summary>原始模型响应</summary>", "", "```text", response[:1500], "```", "", "</details>"])
    screenshot = directory / f"screenshot_{index}.png"
    if not screenshot.exists():
      screenshot = directory / f"screenshot_model_input_{index}.png"
    if screenshot.exists():
      out.extend(["", image_tag(screenshot, root, f"{arm_label} reasoning {index + 1}"), ""])
    else:
      out.append("")
  return out


def render_exploration(folder, root, arm_label):
  if folder is None:
    return ["No task directory found.", ""]
  probes = read_jsonl(folder / "probe_trace.jsonl")
  if not probes:
    return ["No probe records for this task.", ""]
  out = [
    "| Reasoning step | Probe | Source page | Probe action/element | Destination page | Recovery | New elements |",
    "|---:|---:|---|---|---|---|---|",
  ]
  for probe in probes:
    element = probe.get("element", {})
    graph = probe.get("graph", {})
    src = graph.get("src", {}).get("activity") or probe.get("app_package", "")
    dst = graph.get("dst", {}).get("activity") or probe.get("discovered", {}).get("reached_activity", "")
    label = element.get("text") or element.get("content_desc") or element.get("resource_id") or element.get("class") or "(unlabeled)"
    detail = f"{probe.get('probe_type', '?')}: {label}; rank={element.get('rank')}, score={element.get('score')}"
    recovery = f"{probe.get('recovery_level')} / ok={probe.get('recovery_ok')}"
    new_labels = ", ".join(probe.get("discovered", {}).get("new_texts", [])[:8])
    out.append(
      f"| {probe.get('step_idx', '')} | {probe.get('probe_idx', '')} | {md(src)} | {md(detail)} | {md(dst)} | {md(recovery)} | {md(new_labels)} |"
    )
    guidance = probe.get("executable_memory_guidance_stats", {})
    why = probe.get("graph_stop_reason") or "candidate-ranked probe"
    out.append(f"|  |  | Selection context | {md(why)}; candidates={guidance.get('candidate_count', 'n/a')}, seen={guidance.get('seen_count', 'n/a')}, relevant={guidance.get('relevant_count', 'n/a')}, trap-suppressed={guidance.get('trap_suppressed_count', 'n/a')} |  |  |  |")
    shot = folder / "agent_traces"
    dirs = sorted(shot.glob("*"))
    probe_index = probe.get("probe_idx")
    screenshot = dirs[0] / f"opportunity_{int(probe_index) + 1}.png" if dirs and isinstance(probe_index, int) else None
    tag = image_tag(screenshot, root, f"{arm_label} probe {probe_index}") if screenshot else ""
    if tag:
      out.extend(["", tag, ""])
  out.append("")
  return out


def render_graph(folder, root):
  if folder is None:
    return ["No graph artifact.", ""]
  path = folder / "progressive_belief_graph.json"
  graph = read_json(path, {})
  nodes = graph.get("nodes", [])
  edges = graph.get("edges", [])
  if not path.exists():
    return ["No graph artifact (expected for Ex5 baseline).", ""]
  out = [
    f"Persisted graph: [`{md(rel(path, root))}`]({rel(path, root)}), {len(nodes)} nodes / {len(edges)} edges.",
    "",
    "```mermaid",
    "flowchart LR",
  ]
  ids = {}
  for index, node in enumerate(nodes):
    node_id = str(node.get("node_id") or node.get("state_id") or f"node{index}")
    alias = f"n{index}"
    ids[node_id] = alias
    activity = str(node.get("activity") or "?").split("/")[-1]
    labels = ", ".join(node.get("salient_ui_labels", [])[:3])
    title = (activity + ("\\n" + labels if labels else "")).replace('"', "'").replace("|", "/")
    out.append(f'  {alias}["{title}"]')
  for edge in edges:
    src = ids.get(edge.get("src_node"), "")
    dst = ids.get(edge.get("dst_node"), "")
    if not src or not dst:
      continue
    action = edge.get("action", {})
    label = f"{action.get('action_type', '?')}: {action.get('element_identity', '')[:36]} [{edge.get('status', '?')}]"
    label = label.replace('"', "'").replace("|", "/")
    out.append(f'  {src} -->|"{label}"| {dst}')
  out.extend(["```", "", "| Node | Activity | Visits | Stable labels |", "|---|---|---:|---|"])
  for index, node in enumerate(nodes):
    labels = ", ".join(node.get("salient_ui_labels", [])[:8])
    out.append(f"| `{index}` | {md(node.get('activity'))} | {node.get('visit_count', 0)} | {md(labels)} |")
  out.append("")
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--experiment-root", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  root = args.experiment_root.resolve()
  summary = read_json(root / "summary.json", {})
  arms = {arm.get("arm"): arm for arm in summary.get("arms", [])}
  e_name, a_name = "E_FULL_MOBILEEXPLORER", "A_EX5_BASELINE"
  if e_name not in arms or a_name not in arms:
    raise SystemExit(f"Expected {e_name} and {a_name} in {root / 'summary.json'}")
  e_arm, a_arm = arms[e_name], arms[a_name]
  e_tasks = {task["task"]: task for task in e_arm.get("tasks", [])}
  a_tasks = {task["task"]: task for task in a_arm.get("tasks", [])}
  shared = [name for name in e_tasks if name in a_tasks]
  common_evaluated = [
      name for name in shared
      if e_tasks[name].get("success") is not None
      and a_tasks[name].get("success") is not None
  ]
  discordant = [name for name in common_evaluated if
                (e_tasks[name].get("success", 0) >= .5) !=
                (a_tasks[name].get("success", 0) >= .5)]
  e_only = [name for name in discordant if e_tasks[name].get("success", 0) >= .5]
  a_only = [name for name in discordant if a_tasks[name].get("success", 0) >= .5]
  output = (args.output or root / "paired_failure_cases.md").resolve()
  output.parent.mkdir(parents=True, exist_ok=True)

  lines = [
    "# MobileExplorer E vs Ex5：配对实验结果与失败案例",
    "",
    f"Experiment: `{root}`. 两臂任务顺序、seed、模型、初始快照及 step budget 一致；seed=34，GELAB-ZERO-4B，snapshot=`mobileexplorer_pair_initial_20260916`，`max_probes=min_probes=5`，a11y=`grpc`。A 为 Ex5 no-memory；E 为完整 MobileExplorer + Semantic Prefix。",
    "",
    "## 总体结果",
    "",
    "| Arm | Success | Steps mean / median / P95 | Mean task wall time | Mean step latency | Probes | New elements / pages | Reasoning | Graph nodes / edges / skills | Prefix candidates / hits / misses | Parse errors |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for arm in (e_arm, a_arm):
    prefix = f"{arm.get('prefix_route_candidate_count', 0)} / {arm.get('prefix_route_hit_count', 0)} / {arm.get('prefix_route_miss_count', 0)}"
    tasks = arm.get("tasks", [])
    mean_wall = sum(task.get("wall_time_s", 0) for task in tasks) / max(1, len(tasks))
    lines.append(
      f"| {arm['arm']} | {arm.get('success_rate', 0):.1%} ({sum(t.get('success') is not None and t.get('success', 0) >= .5 for t in tasks)}/{sum(t.get('success') is not None for t in tasks)} evaluated; {sum(t.get('success') is None for t in tasks)} unknown) | {arm.get('step_mean', 0):.3f} / {arm.get('step_median', 0):.1f} / {arm.get('step_p95', 0):.2f} | {mean_wall:.2f}s | {arm.get('step_latency_mean_s', 0):.2f}s | {arm.get('exploration_probe_count', 0)} | {arm.get('exploration_new_element_count', 0)} / {arm.get('exploration_unique_page_count', 0)} | {arm.get('reasoning_count', 0)} | {arm.get('graph_node_count', 0)} / {arm.get('graph_edge_count', 0)} / {arm.get('graph_skill_count', 0)} | {prefix} | {arm.get('parse_error_count', 0)} |"
    )
  e_win = sum(e_tasks[n].get("success", 0) >= .5 for n in common_evaluated)
  a_win = sum(a_tasks[n].get("success", 0) >= .5 for n in common_evaluated)
  both_success = sum(e_tasks[n].get("success", 0) >= .5 and a_tasks[n].get("success", 0) >= .5 for n in common_evaluated)
  both_fail = sum(e_tasks[n].get("success", 0) < .5 and a_tasks[n].get("success", 0) < .5 for n in common_evaluated)
  unknown_e = sum(e_tasks[n].get("success") is None for n in shared)
  unknown_a = sum(a_tasks[n].get("success") is None for n in shared)
  e_common_steps = [e_tasks[n]["steps"] for n in common_evaluated if e_tasks[n].get("steps") is not None]
  a_common_steps = [a_tasks[n]["steps"] for n in common_evaluated if a_tasks[n].get("steps") is not None]
  e_common_mean = sum(e_common_steps) / max(1, len(e_common_steps))
  a_common_mean = sum(a_common_steps) / max(1, len(a_common_steps))
  e_common_median = sorted(e_common_steps)[len(e_common_steps) // 2] if len(e_common_steps) % 2 else (
      (sorted(e_common_steps)[len(e_common_steps) // 2 - 1] + sorted(e_common_steps)[len(e_common_steps) // 2]) / 2
      if e_common_steps else 0
  )
  a_common_median = sorted(a_common_steps)[len(a_common_steps) // 2] if len(a_common_steps) % 2 else (
      (sorted(a_common_steps)[len(a_common_steps) // 2 - 1] + sorted(a_common_steps)[len(a_common_steps) // 2]) / 2
      if a_common_steps else 0
  )
  last_e = e_arm.get("tasks", [{}])[-1].get("executable", {}) if e_arm.get("tasks") else {}
  lines.extend([
    "",
    f"配对结果（只计两臂都完成 evaluator 的 {len(common_evaluated)} 项）：双方都成功 {both_success} 项、都失败 {both_fail} 项；E-only 成功 {len(e_only)} 项、A-only 成功 {len(a_only)} 项。共同可评估任务成功数 E={e_win}/{len(common_evaluated)}、A={a_win}/{len(common_evaluated)}（差 {e_win - a_win:+d}）。基础任务集中另有 E={unknown_e}、A={unknown_a} 项 evaluator/设备基础设施未知，不作 agent 失败计分。",
    "",
    f"在共同可评估任务上，E/A 平均 steps={e_common_mean:.3f}/{a_common_mean:.3f}（差 {e_common_mean - a_common_mean:+.3f}），中位数={e_common_median:.1f}/{a_common_median:.1f}；全组平均单步延迟 E/A={e_arm.get('step_latency_mean_s', 0):.2f}/{a_arm.get('step_latency_mean_s', 0):.2f}s。",
    f"按逐任务 wall_time_s 求均值，E={sum(t.get('wall_time_s', 0) for t in e_arm.get('tasks', [])) / max(1, len(e_arm.get('tasks', []))):.2f}s/task，A={sum(t.get('wall_time_s', 0) for t in a_arm.get('tasks', [])) / max(1, len(a_arm.get('tasks', []))):.2f}s/task；E 增加约 {sum(t.get('wall_time_s', 0) for t in e_arm.get('tasks', [])) / max(1, len(e_arm.get('tasks', []))) - sum(t.get('wall_time_s', 0) for t in a_arm.get('tasks', [])) / max(1, len(a_arm.get('tasks', []))):+.2f}s/task。",
    "",
    "E 的路由/跳步尚未形成收益：graph prompt adopted=" + str(e_arm.get("graph_prompt_adoptions", 0)) + ", graph overrides=" + str(e_arm.get("graph_overrides", 0)) + ", graph rejections=" + str(e_arm.get("graph_rejections", 0)) + ", skip hits=" + str(e_arm.get("graph_skip_hits", 0)) + "; Prefix route candidates/hits/misses=" + f"{e_arm.get('prefix_route_candidate_count', 0)} / {e_arm.get('prefix_route_hit_count', 0)} / {e_arm.get('prefix_route_miss_count', 0)}" + ".",
    "",
    "Probe 目标是每个 reasoning window 5 次；安全候选不足或无进展时允许提前停止。E 完成 5-probe 的窗口为 " + str(e_arm.get("five_probe_complete_windows", 0)) + "; 总窗口 " + str(last_e.get("probe_rounds", "n/a")) + "，目标 probe=" + str(last_e.get("probes_target", "n/a")) + "，实际 probe=" + str(e_arm.get("exploration_probe_count", 0)) + "。目标完成率很低，常见 early stop 原因见末尾 E 累计日志，不能把预算配置描述成实际完成。",
    "",
    "## 不一致任务索引",
    "",
    "| Direction | Task | E success / steps / probes | A success / steps / probes |",
    "|---|---|---:|---:|",
  ])
  for name in discordant:
    e, a = e_tasks[name], a_tasks[name]
    direction = "E-only" if e.get("success", 0) >= .5 else "A-only regression"
    lines.append(f"| {direction} | [{name}](#{name.lower()}) | {int(e.get('success', 0))} / {e.get('steps', 0)} / {e.get('probe_count', 0)} | {int(a.get('success', 0))} / {a.get('steps', 0)} / {a.get('probe_count', 0)} |")

  for name in discordant:
    e, a = e_tasks[name], a_tasks[name]
    e_folder = task_dir(root, e_name, name)
    a_folder = task_dir(root, a_name, name)
    lines.extend([
      "",
      f"## {name}",
      "",
      f"**配对结果：** E success={int(e.get('success', 0))}, steps={e.get('steps', 0)}, probes={e.get('probe_count', 0)}, reasoning={e.get('reasoning_count', 0)}; A success={int(a.get('success', 0))}, steps={a.get('steps', 0)}, probes={a.get('probe_count', 0)}, reasoning={a.get('reasoning_count', 0)}.",
      "",
      "### E：reasoning trace 与 screenshot",
      "",
    ])
    lines.extend(render_reasoning(e_folder, root, "E"))
    lines.extend(["### E：exploration 页面、候选动作与恢复", ""])
    lines.extend(render_exploration(e_folder, root, "E"))
    lines.extend(["### E：持久化图结构", ""])
    lines.extend(render_graph(e_folder, root))
    lines.extend(["### A Ex5：reasoning trace 与 screenshot", ""])
    lines.extend(render_reasoning(a_folder, root, "A Ex5"))
    lines.extend(["### A Ex5：exploration 页面与恢复", ""])
    lines.extend(render_exploration(a_folder, root, "A Ex5"))
    lines.extend(["### 原始数据", ""])
    for arm_name, folder in ((e_name, e_folder), (a_name, a_folder)):
      if folder:
        lines.append(f"- `{arm_name}`: [`task_metrics.json`]({rel(folder / 'task_metrics.json', root)}), [`probe_trace.jsonl`]({rel(folder / 'probe_trace.jsonl', root)}), [`progressive_belief_graph.json`]({rel(folder / 'progressive_belief_graph.json', root)})")
    lines.append("")

  if last_e:
    reasons = last_e.get("stop_reasons", {})
    lines.extend([
      "## Exploration stop reasons and graph diagnostics",
      "",
      f"E cumulative probe rounds={last_e.get('probe_rounds', 'n/a')}, completed 5-probe rounds={last_e.get('probe_rounds_complete', 'n/a')}, completed probes={last_e.get('probes_completed', 'n/a')} / target={last_e.get('probes_target', 'n/a')}; extra time={last_e.get('extra_time_s', 0):.1f}s; recovery failures={last_e.get('recovery_failures', 0)}.",
      "",
      "| Stop reason | Count |",
      "|---|---:|",
    ])
    for reason, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
      lines.append(f"| `{md(reason)}` | {count} |")
    lines.extend([
      "",
      f"Graph: {last_e.get('node_count', 0)} nodes / {last_e.get('edge_count', 0)} edges / {last_e.get('skill_count', 0)} skills; state merges={last_e.get('state_merges', 0)}; repeated validations={last_e.get('repeated_validations', 0)}; prompt adoptions={last_e.get('prompt_adoptions', 0)}; path candidates={last_e.get('retrieval_path_candidates', 0)}; skip attempts/hits={last_e.get('skip_attempts', 0)}/{last_e.get('skip_hits', 0)}.",
      "",
      "## Interpretation",
      "",
      f"On the {len(common_evaluated)} common evaluable tasks, E has {e_win - a_win:+d} net successes ({len(e_only)} E-only vs {len(a_only)} A-only); this is only {len(discordant)} discordant pairs, so it is not strong evidence of a success-rate change. Common-task mean steps differ by {e_common_mean - a_common_mean:+.3f} (E minus A), with medians {e_common_median:.1f} vs {a_common_median:.1f}; the run does not show a meaningful step reduction. Retrieval produced {e_arm.get('prefix_route_candidate_count', 0)} Prefix route candidates, but there were {e_arm.get('graph_skip_hits', 0)} verified skip hits and {e_arm.get('graph_prompt_adoptions', 0)} graph prompt adoptions. Exploration stopped early on many windows and completed five probes on only a small fraction. Treat this as a diagnostic run: E is slightly ahead on paired success count, but it did not reduce steps or wall time and has not demonstrated a robust advantage over Ex5.",
      "",
    ])
  output.write_text("\n".join(lines), encoding="utf-8")
  print(f"wrote {output} ({len(discordant)} paired discordances: {len(a_only)} E regressions, {len(e_only)} E-only successes)")


if __name__ == "__main__":
  main()
