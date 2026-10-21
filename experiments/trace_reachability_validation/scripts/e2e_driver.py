"""Full AndroidWorld end-to-end latency run: 116 tasks x 1 pass (seed 30), arms interleaved per task, decisions on the
Jetson AGX Orin vLLM, on-device inference load in the emulator (ONDEVICE_LOAD=1). MobileExplorer arms start from an
empty graph and learn online within this run only. Every 20 paired tasks a progress table (success, inference
steps, successful-task steps, end-to-end latency) is written to reports/e2e_full_orin_zh.md.

usage: e2e_driver.py BASE,FULL41 [tag]
"""
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
ARMS = sys.argv[1].split(",")
TAG = sys.argv[2] if len(sys.argv) > 2 else "e2e"
RAW, GDIR = ROOT / "raw" / TAG, ROOT / "graph_snapshots" / TAG
LOG = ROOT / "reports" / f"{TAG}_runs.jsonl"
REPORT = ROOT / "reports" / f"{TAG}_full_orin_zh.md"
API, MODEL = "http://127.0.0.1:18097/v1/chat/completions", "GELab-Zero-4B-preview"
SEED = 30
LOAD_MODEL = os.environ.get("LOAD_MODEL", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf")


def note(m):
  line = f"- {time.strftime('%m-%d %H:%M:%S')} {m}"
  print(line, flush=True)
  with (ROOT / "reports" / "execution_log.md").open("a") as f:
    f.write(line + "\n")


def services_ok():
  r = subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "8", "http://127.0.0.1:18097/v1/models"],
                     capture_output=True, text=True)
  subprocess.run(["adb", "-s", "emulator-5554", "forward", "tcp:18096", "tcp:8085"], capture_output=True)
  l = subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "8", "http://127.0.0.1:18096/v1/models"],
                     capture_output=True, text=True)
  return "GELab" in r.stdout, ".gguf" in l.stdout


def a11y_ok():
  """Android does not rebind a crashed accessibility service (low memory kills it): re-enable it if needed."""
  out = subprocess.run(["adb", "-s", "emulator-5554", "shell", "dumpsys accessibility | grep 'Crashed services'"],
                       capture_output=True, text=True).stdout
  if "Crashed services:{}" in out:
    return True
  svc = subprocess.run(["adb", "-s", "emulator-5554", "shell", "settings get secure enabled_accessibility_services"],
                       capture_output=True, text=True).stdout.strip()
  subprocess.run(["adb", "-s", "emulator-5554", "shell", "settings put secure enabled_accessibility_services null"])
  time.sleep(2)
  subprocess.run(["adb", "-s", "emulator-5554", "shell", f"settings put secure enabled_accessibility_services {svc}"])
  subprocess.run(["adb", "-s", "emulator-5554", "shell", "settings put secure accessibility_enabled 1"])
  time.sleep(4)
  note(f"a11y services were crashed; re-enabled ({out.strip()[:120]})")
  return False


def empty_capture_share(out):
  import gzip  # noqa: E402
  fs = sorted((out / "a11y").glob("*.json.gz")) if (out / "a11y").exists() else []
  if not fs:
    return 1.0
  empty = 0
  for f in fs:
    try:
      empty += not json.load(gzip.open(f, "rt")).get("payload", {}).get("nodes")
    except Exception:  # pylint: disable=broad-except
      empty += 1
  return empty / len(fs)


def load_rows():
  rows = {}
  if LOG.exists():
    for l in LOG.read_text().splitlines():
      r = json.loads(l)
      if r.get("error"):
        rows.pop((r["task"], r["arm"]), None)   # errored episodes (environment failures) are re-run
        continue
      rows[(r["task"], r["arm"])] = r
  return rows


def report(tasks, rows, n_done):
  paired = [t for t in tasks if all((t, a) in rows for a in ARMS)][:n_done]
  ok = [t for t in paired if not any(rows[(t, a)].get("error") for a in ARMS)]
  lines = [f"## 进度：{len(paired)} 题配对（剔除出错 {len(paired) - len(ok)} 题，统计 {len(ok)} 题） {time.strftime('%m-%d %H:%M')}", "",
           "| 系统 | 成功 | 成功率 | 平均推理次数 | 成功任务推理次数 | E2E 均值 | E2E 中位数 | 成功任务 E2E 均值 | E2E 总时长 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
  for a in ARMS:
    R = [rows[(t, a)] for t in ok]
    succ = [r for r in R if r["success"]]
    e2e = [r["e2e_s"] for r in R]
    lines.append(f"| {a} | {len(succ)} | {100 * len(succ) / max(1, len(R)):.1f}% | {st.mean(r['calls'] or 0 for r in R):.2f} | "
                 f"{st.mean(r['calls'] for r in succ) if succ else 0:.2f} | {st.mean(e2e):.1f} s | {st.median(e2e):.1f} s | "
                 f"{st.mean(r['e2e_s'] for r in succ) if succ else 0:.1f} s | {sum(e2e) / 60:.1f} min |")
  both = [t for t in ok if all(rows[(t, a)]["success"] for a in ARMS)]
  if both:
    lines += ["", f"两边都成功的 {len(both)} 题："]
    for a in ARMS:
      lines.append(f"- {a}: 推理 {st.mean(rows[(t, a)]['calls'] for t in both):.2f} 次，E2E {st.mean(rows[(t, a)]['e2e_s'] for t in both):.1f} s，"
                   f"agent 段 {st.mean(rows[(t, a)]['agent_s'] for t in both):.1f} s")
  lines += ["", "| 系统 | 首 token（prefill）均值 | 生成速度 tok/s | 每次生成 token 数 | 动作字段完成时刻 | 整条输出结束 | 动作后剩余解码 | THINK 比例 | 每步等 summary |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
  for a in ARMS:
    G, W = [], []
    for t in ok:
      d = RAW / a / t / "p0"
      if (d / "gen.jsonl").exists():
        G += [json.loads(l) for l in open(d / "gen.jsonl")]
      if (d / "late_summary.jsonl").exists():
        W += [json.loads(l)["join_wait_s"] for l in open(d / "late_summary.jsonl")]
    g = lambda k: [x[k] for x in G if x.get(k) is not None]  # noqa: E731
    tail = [x["t_end_s"] - x["t_action_s"] for x in G if x.get("t_action_s") is not None]
    mean = lambda v: st.mean(v) if v else float("nan")  # noqa: E731
    lines.append(f"| {a} | {mean(g('ttft_s')):.2f} s | {mean(g('decode_tok_s')):.1f} | {mean(g('completion_tokens')):.0f} | "
                 f"{mean(g('t_action_s')):.2f} s | {mean(g('t_end_s')):.2f} s | {mean(tail):.2f} s | "
                 f"{100 * mean([float(x.get('think', False)) for x in G]):.0f}% | {mean(W) if W else 0:.2f} s |")
  lines.append("")
  with REPORT.open("a") as f:
    f.write("\n".join(lines) + "\n")
  note(f"E2E REPORT {len(paired)} paired")


def main():
  from android_world import registry  # noqa: E402
  tasks = sorted(registry.TaskRegistry().get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY))
  if os.environ.get("E2E_TASKS"):
    tasks = os.environ["E2E_TASKS"].split(",")
  if not REPORT.exists():
    REPORT.write_text(f"# AndroidWorld 全量端到端时延（Orin GELab-Zero-4B + 模拟器内 Qwen2-VL-2B（纯文本）负载）\n\n"
                      f"臂：{', '.join(ARMS)}；seed {SEED}；每题交错运行；MobileExplorer 图从空开始、只在本轮在线积累。\n"
                      f"E2E = 本题 full_run 子进程墙钟（含环境初始化、任务重置、执行、评估）；agent 段 = 第一次截图到最后一步。\n\n")
  rows = load_rows()
  n_paired = sum(all((t, a) in rows for a in ARMS) for t in tasks)
  note(f"E2E DRIVER start {ARMS} tag={TAG} tasks={len(tasks)} already paired={n_paired}")
  for i, task in enumerate(tasks):
    order = ARMS if i % 2 == 0 else ARMS[::-1]
    for arm in order:
      if (task, arm) in rows:
        continue
      out = RAW / arm / task / "p0"
      if out.exists():
        out.rename(out.with_name(out.name + f"_partial_{int(time.time())}"))
      a11y_ok()
      mem = subprocess.run(["adb", "-s", "emulator-5554", "shell", "grep -E 'MemAvailable|SwapFree' /proc/meminfo"],
                           capture_output=True, text=True).stdout.split()
      mem_mb = {mem[i].rstrip(":"): int(mem[i + 1]) // 1024 for i in range(0, len(mem) - 2, 3)} if len(mem) >= 6 else {}
      if mem_mb.get("MemAvailable", 9999) < 300:
        note(f"LOW MEMORY before {arm}/{task}: {mem_mb}")
      while True:
        g, l = services_ok()
        if g and l:
          break
        note(f"service down: orin={g} ondevice={l}")
        if not l:
          # the emulator-resident load model was killed (low memory killer / env reset): start it again
          subprocess.Popen(["adb", "-s", "emulator-5554", "shell",
                            "cd /data/local/tmp/llm && setsid nohup ./llama-server -m " + LOAD_MODEL +
                            " -t 4 -c 4096 -b 64 -ub 64 --cache-ram 0 --host 127.0.0.1 --port 8085 -np 1 > server.log 2>&1 < /dev/null &"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
          note("on-device load server restarted")
          time.sleep(15)
          continue
        time.sleep(60)
      # "X+S": arm X with streamed early action; every arm streams its decisions (same transport, timed)
      env = dict(os.environ, FULL_ARM=arm.split("+")[0], SEQ_APPDIR=str(GDIR), ONDEVICE_LOAD="1", ONDEVICE_PORT="18096",
                 STREAM_IO="1")
      if arm.endswith("+S"):
        env["STREAM_ACT"] = "1"
      cmd = [sys.executable, str(ROOT / "scripts" / "full_run.py"), "--task", task, "--seed", str(SEED),
             "--variant", f"{TAG}_{arm}", "--mode", "agent", "--out", str(out), "--api_url", API, "--model", MODEL]
      t0 = time.time()
      try:
        with (ROOT / "logs" / f"{TAG}.log").open("a") as f:
          rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env, timeout=2400).returncode
      except subprocess.TimeoutExpired:
        rc = "timeout"
      e2e = time.time() - t0
      res = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
      agent_s = None
      try:
        ts = [json.loads(l)["pre"]["t"] for l in open(out / "steps.jsonl") if '"pre"' in l and json.loads(l).get("pre")]
        agent_s = round(max(ts) - min(ts), 1) if ts else 0.0
      except Exception:  # pylint: disable=broad-except
        pass
      row = {"task": task, "arm": arm, "rc": rc, "success": float(res.get("is_successful") or 0) == 1.0,
             "calls": res.get("model_calls"), "steps": res.get("steps"), "probes": res.get("n_probes"),
             "e2e_s": round(e2e, 1), "wall_s": res.get("wall_s"), "agent_s": agent_s,
             "error": (str(res.get("error"))[:200] if res.get("error") else (None if res else "no result"))}
      row["mem_mb_before"] = mem_mb
      share = empty_capture_share(out)
      row["empty_a11y_share"] = round(share, 3)
      if share > 0.2 and not row["error"]:
        row["error"] = f"empty a11y captures {share:.0%}"   # the episode ran blind: re-run it
      with LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
      rows[(task, arm)] = row
      note(f"E2E {arm} {task} rc={rc} succ={res.get('is_successful')} calls={row['calls']} probes={row['probes']} "
           f"e2e={row['e2e_s']}s err={row['error']}")
    n_new = sum(all((t, a) in rows for a in ARMS) for t in tasks)
    if n_new > n_paired:
      n_paired = n_new
      if n_paired % int(os.environ.get("E2E_REPORT_EVERY", "20")) == 0:
        report(tasks, rows, n_paired)
  report(tasks, rows, n_paired)
  note("E2E DRIVER DONE")


if __name__ == "__main__":
  main()
