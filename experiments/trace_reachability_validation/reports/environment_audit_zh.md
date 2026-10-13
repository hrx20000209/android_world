# 环境核查（2026-10-13）

- 上一轮实验：`experiments/runtime_graph_study/`，报告为 `reports/final_report_zh.md`。上一轮的关键数字已从原始数据重新加载核对：143 个 agent episode（Pilot A 使用）；图观测 927 条（427 次探索 probe + agent 转移）。
- 模型：LabServer `GELAB-ZERO-4B`（vLLM 0.13，端口 8095，`--seed 0`）。经 SSH 隧道 `127.0.0.1:18095` 访问，`/v1/models` 正常，运行/等待队列为 0。没有重启服务，没有启动 MAI-UI。
- 推理参数（所有条件一致）：temperature 0，top_p 1，max_tokens 1024，每个请求一个随机 `cache_salt`。M0 重复调用 0/47 改变，确认是确定性的。
- Agent prompt / parser：`android_world/agents/gelab_agent.py` 的 `build_gelab_messages` 和 `parse_gelab_response`，未修改；证据追加在用户文本之后（M2pre 条件放在之前）。
- 设备：本地 `emulator-5554`（Android 13，1080×2400）；在线实验串行执行，不与其他 runner 并发。另一台已连接的真机未使用。
- 磁盘：开始时剩余约 24 GB。
