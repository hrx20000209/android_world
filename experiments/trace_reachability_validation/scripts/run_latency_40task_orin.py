"""Run a paired 20-task AndroidWorld latency comparison on the Orin backend.

Both arms use the same task seed and run an emulator-resident Qwen2-VL-2B
request alongside each remote GELab decision. FULL40 starts with an empty
graph and learns only from earlier tasks in this run.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRV = HERE.parent
REPO = TRV.parents[1]
RUN = TRV / "latency_40task_orin_20261021"
RAW = RUN / "raw"
GRAPHS = RUN / "graph_snapshots"
LOGS = RUN / "logs"
REPORT = RUN / "progress_report.md"
EVENTS = RUN / "runs.jsonl"
TASK_CONFIG = REPO / "configs" / "two_system_scaleup_40tasks.json"

ADB = os.environ.get("ADB", "adb")
SERIAL = os.environ.get("ANDROID_SERIAL", "emulator-5554")
LOCAL_PORT = 18096
DEVICE_PORT = 8085
REMOTE_API = "http://127.0.0.1:18097/v1/chat/completions"
REMOTE_MODELS = "http://127.0.0.1:18097/v1/models"
LOCAL_MODELS = f"http://127.0.0.1:{LOCAL_PORT}/v1/models"
MODEL = "GELab-Zero-4B-preview"
ARMS = ("BASE", "FULL40")
TASK_TIMEOUT_S = 1800
TASK_LIMIT = 20


def get_json(url: str, timeout: float = 5.0) -> dict:
  req = urllib.request.Request(url, headers={"Accept": "application/json"})
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    return json.load(resp)


def model_ids(url: str) -> list[str]:
  return [str(x.get("id", "")) for x in get_json(url).get("data", [])]


def percentile(values: list[float], q: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def fmt(value: float | None, suffix: str = "") -> str:
  return "—" if value is None else f"{value:.2f}{suffix}"


def write_jsonl(path: Path, row: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def read_events() -> dict[tuple[str, str], dict]:
  out: dict[tuple[str, str], dict] = {}
  if not EVENTS.exists():
    return out
  for line in EVENTS.read_text(encoding="utf-8").splitlines():
    try:
      row = json.loads(line)
    except ValueError:
      continue
    if row.get("completed"):
      out[(row["task"], row["arm"])] = row
  return out


def check_services() -> tuple[list[str], list[str]]:
  remote = model_ids(REMOTE_MODELS)
  if not any("GELab-Zero-4B-preview" in x for x in remote):
    raise RuntimeError(f"Expected Orin model unavailable at {REMOTE_MODELS}: {remote}")
  local = model_ids(LOCAL_MODELS)
  if not local:
    raise RuntimeError(f"No llama.cpp model served at {LOCAL_MODELS}")
  return remote, local


def run_one(task: str, arm: str, index: int, task_seed: int, max_steps: int) -> dict:
  arm_dir = RAW / arm
  out = arm_dir / task / "p0"
  if (out / "result.json").exists():
    res = json.loads((out / "result.json").read_text(encoding="utf-8"))
    row = {
        "task": task, "arm": arm, "order_index": index, "seed": task_seed, "max_steps": max_steps,
        "completed": True, "rc": 0, "success": float(res.get("is_successful", 0.0)) == 1.0,
        "steps": res.get("steps"), "model_calls": res.get("model_calls"),
        "n_probes": res.get("n_probes"), "episode_wall_s": res.get("wall_s"),
        "end_to_end_s": res.get("wall_s"), "latency_source": "recovered_result_wall_s",
        "result": str(out / "result.json"), "recovered": True,
    }
    write_jsonl(EVENTS, row)
    return row

  if out.exists():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out.rename(out.with_name(out.name + f"_partial_{stamp}"))
  out.mkdir(parents=True, exist_ok=True)
  log_path = LOGS / arm / f"{task}.log"
  log_path.parent.mkdir(parents=True, exist_ok=True)
  env = dict(os.environ)
  env.update({
      "FULL_ARM": arm,
      "SEQ_APPDIR": str(GRAPHS / "full"),
      "ONDEVICE_LOAD": "1",
      "ONDEVICE_PORT": str(LOCAL_PORT),
      "ANDROID_SERIAL": SERIAL,
  })
  cmd = [
      sys.executable, str(TRV / "scripts" / "full_run.py"),
      "--task", task, "--seed", str(task_seed),
      "--variant", f"latency40_{arm}", "--mode", "agent", "--out", str(out),
      "--api_url", REMOTE_API, "--model", MODEL, "--max_steps", str(max_steps),
  ]

  print(f"START {index:02d}/{TASK_LIMIT} {arm:6s} {task}", flush=True)
  started = time.monotonic()
  rc: int | str
  try:
    with log_path.open("a", encoding="utf-8") as log:
      proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=TRV, env=env,
                            timeout=TASK_TIMEOUT_S, check=False)
    rc = proc.returncode
  except subprocess.TimeoutExpired:
    rc = "timeout"
  elapsed = round(time.monotonic() - started, 2)
  result_path = out / "result.json"
  res = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
  success_val = res.get("is_successful")
  row = {
      "task": task, "arm": arm, "order_index": index, "seed": task_seed, "max_steps": max_steps,
      "completed": result_path.exists(), "rc": rc,
      "success": (float(success_val) == 1.0) if success_val is not None else False,
      "steps": res.get("steps"), "successful_steps": res.get("agent_steps"),
      "model_calls": res.get("model_calls"), "n_probes": res.get("n_probes"),
      "episode_wall_s": res.get("wall_s"), "end_to_end_s": elapsed,
      "latency_source": "parent_process_monotonic", "result": str(result_path),
      "error": res.get("error") or ("no result.json" if not result_path.exists() else None),
  }
  write_jsonl(EVENTS, row)
  print(json.dumps({k: row[k] for k in ("task", "arm", "rc", "success", "steps", "model_calls", "end_to_end_s")},
                   ensure_ascii=False), flush=True)
  return row


def stats(rows: list[dict]) -> dict:
  durations = [float(r["end_to_end_s"]) for r in rows if r.get("end_to_end_s") is not None]
  step_rows = [r for r in rows if isinstance(r.get("steps"), (int, float))]
  success_rows = [r for r in step_rows if r.get("success")]
  calls = [float(r["model_calls"]) for r in rows if isinstance(r.get("model_calls"), (int, float))]
  return {
      "n": len(rows), "successes": sum(bool(r.get("success")) for r in rows),
      "rate": sum(bool(r.get("success")) for r in rows) / len(rows) if rows else None,
      "steps_mean": statistics.mean([float(r["steps"]) for r in step_rows]) if step_rows else None,
      "steps_success_mean": statistics.mean([float(r["steps"]) for r in success_rows]) if success_rows else None,
      "calls_mean": statistics.mean(calls) if calls else None,
      "latency_mean": statistics.mean(durations) if durations else None,
      "latency_median": statistics.median(durations) if durations else None,
      "latency_p90": percentile(durations, 0.9),
      "latency_total": sum(durations) if durations else None,
  }


def report_stage(tasks: list[str], records: dict[tuple[str, str], dict], stage: int,
                 remote_ids: list[str], local_ids: list[str]) -> None:
  paired = [t for t in tasks[:stage] if all((t, a) in records for a in ARMS)]
  if not paired:
    return
  by_arm = {a: [records[(t, a)] for t in paired] for a in ARMS}
  s = {a: stats(by_arm[a]) for a in ARMS}
  base = {r["task"]: r for r in by_arm["BASE"]}
  full = {r["task"]: r for r in by_arm["FULL40"]}
  paired_latency = [float(full[t]["end_to_end_s"]) - float(base[t]["end_to_end_s"]) for t in paired
                    if full[t].get("end_to_end_s") is not None and base[t].get("end_to_end_s") is not None]
  paired_steps = [float(full[t]["steps"]) - float(base[t]["steps"]) for t in paired
                  if isinstance(full[t].get("steps"), (int, float)) and isinstance(base[t].get("steps"), (int, float))]
  same_success = sum(bool(full[t].get("success")) == bool(base[t].get("success")) for t in paired)
  lines = [
      f"## 阶段：{stage} / {len(tasks)} 个配对任务",
      "",
      "| 系统 | 完成数 | 成功 | 成功率 | 平均步数 | 成功任务平均步数 | 平均模型调用 | E2E 均值 | E2E 中位数 | E2E P90 | E2E 总时长 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for arm, label in (("BASE", "Baseline"), ("FULL40", "MobileExplorer / FULL40")):
    q = s[arm]
    lines.append(
        f"| {label} | {q['n']} | {q['successes']} | {fmt(100*q['rate'], '%') if q['rate'] is not None else '—'} "
        f"| {fmt(q['steps_mean'])} | {fmt(q['steps_success_mean'])} | {fmt(q['calls_mean'])} "
        f"| {fmt(q['latency_mean'], ' s')} | {fmt(q['latency_median'], ' s')} | {fmt(q['latency_p90'], ' s')} "
        f"| {fmt(q['latency_total'], ' s')} |"
    )
  lines += [
      "",
      f"- 成功结果一致：{same_success}/{len(paired)}；配对平均 E2E 差（MobileExplorer − Baseline）：{fmt(statistics.mean(paired_latency), ' s') if paired_latency else '—'}。",
      f"- 配对平均步数差（MobileExplorer − Baseline）：{fmt(statistics.mean(paired_steps)) if paired_steps else '—'}。",
      "- E2E 定义：从启动本题 `full_run.py` 子进程到结果进程结束，包含 AndroidWorld 环境初始化、任务重置、agent 执行和成功评估；不含两题之间的调度间隔。",
      "- 模型：Orin GELab-Zero-4B-preview；端侧常驻模型：Android VM llama.cpp Qwen2-VL-2B，Host 通过 adb forward 的 18096 端口并发触发。",
      "",
      f"Remote `/v1/models`: `{', '.join(remote_ids)}`；Android VM `/v1/models`: `{', '.join(local_ids)}`。",
      "",
  ]
  existing = REPORT.read_text(encoding="utf-8") if REPORT.exists() else (
      "# AndroidWorld 20-task Orin E2E 对比\n\n"
      "任务集：`configs/two_system_scaleup_40tasks.json` 的前 20 个固定分层样本（seed 30）。"
      "Baseline 与 FULL40 按任务交错运行；FULL40 图从空目录开始，仅在本轮任务间在线积累。\n\n"
  )
  marker = f"## 阶段：{stage} / {len(tasks)} 个配对任务"
  if marker in existing:
    existing = existing[:existing.index(marker)].rstrip() + "\n\n"
  REPORT.write_text(existing + "\n".join(lines), encoding="utf-8")
  print("\n" + "\n".join(lines), flush=True)


def main() -> int:
  if RUN.exists() and not EVENTS.exists() and any(RUN.iterdir()):
    raise SystemExit(f"Refusing to reuse non-empty run directory without {EVENTS.name}: {RUN}")
  RUN.mkdir(parents=True, exist_ok=True)
  RAW.mkdir(parents=True, exist_ok=True)
  GRAPHS.mkdir(parents=True, exist_ok=True)
  LOGS.mkdir(parents=True, exist_ok=True)

  config = json.loads(TASK_CONFIG.read_text(encoding="utf-8"))
  task_specs = {x["name"]: x for x in config["tasks"]}
  configured_tasks = list(task_specs)
  if len(configured_tasks) != 40:
    raise SystemExit(f"Expected exactly 40 configured tasks, got {len(configured_tasks)}")
  tasks = configured_tasks[:TASK_LIMIT]
  task_seed = int(config.get("seed", 30))

  subprocess.run([ADB, "-s", SERIAL, "forward", f"tcp:{LOCAL_PORT}", f"tcp:{DEVICE_PORT}"],
                 check=True, timeout=15, capture_output=True, text=True)
  try:
    remote_ids, local_ids = check_services()
    meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                                   check=True).stdout.strip(),
        "task_config": str(TASK_CONFIG), "configured_task_count": len(configured_tasks),
        "task_limit": TASK_LIMIT, "tasks": tasks,
        "max_steps": {t: task_specs[t]["max_steps"] for t in tasks}, "seed": task_seed,
        "remote_api": REMOTE_API, "remote_models": remote_ids,
        "on_device_models": local_ids, "adb_serial": SERIAL,
        "on_device_forward": f"tcp:{LOCAL_PORT}->tcp:{DEVICE_PORT}",
        "arms": list(ARMS), "ondevice_load": True,
        "mobileexplorer_graph_starts_empty": True,
    }
    (RUN / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ready: remote={remote_ids}; emulator={local_ids}; paired_tasks={len(tasks)}; seed={task_seed}", flush=True)

    records = read_events()
    paired_count = sum(all((t, a) in records for a in ARMS) for t in tasks)
    for i, task in enumerate(tasks, 1):
      order = ARMS if i % 2 == 1 else ARMS[::-1]
      for arm in order:
        if (task, arm) in records:
          continue
        check_services()
        row = run_one(task, arm, i, task_seed, int(task_specs[task]["max_steps"]))
        if not row["completed"]:
          print(f"ABORT: {arm}/{task} did not produce result.json; inspect {LOGS / arm / (task + '.log')}",
                flush=True)
          return 2
        records[(task, arm)] = row
      new_count = sum(all((t, a) in records for a in ARMS) for t in tasks)
      if new_count > paired_count:
        paired_count = new_count
        if paired_count == len(tasks):
          report_stage(tasks, records, paired_count, remote_ids, local_ids)
      print(f"PAIRED_PROGRESS {new_count}/{len(tasks)}", flush=True)

    if paired_count == len(tasks):
      report_stage(tasks, records, len(tasks), remote_ids, local_ids)
      print(f"DONE {REPORT}", flush=True)
      return 0
    print(f"Stopped with {paired_count}/{len(tasks)} paired tasks; resumable data is in {RUN}", flush=True)
    return 2
  finally:
    subprocess.run([ADB, "-s", SERIAL, "forward", "--remove", f"tcp:{LOCAL_PORT}"],
                   check=False, timeout=15, capture_output=True, text=True)


if __name__ == "__main__":
  raise SystemExit(main())
