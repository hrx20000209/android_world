# 环境报告（2026-10-13 实测）

## 设备
| 项 | 值 |
|---|---|
| 主机 | Apple M4，24 GB RAM，macOS（Darwin 25.6） |
| Android | 本地 AVD `AndroidWorldAvd`（`emulator-5554`，`-grpc 8554`），`sdk_gphone64_arm64`，Android 13（TE1A.240213.009），1080×2400，420 dpi |
| 另一台已连接设备 | `10.79.146.209:40607`（真机/远程设备）——**本实验未使用、未操作** |
| a11y 采集 | 仓库自带 `com.androidworld.fasta11y` 服务，`adb forward tcp:8765 localabstract:androidworld_fast_a11y`；flat dump 设备端 3–70 ms |
| 磁盘 | 实验开始时仅剩 ~3.7 GB，所以截图存 JPEG（q82；agent 步骤全分辨率，probe 半分辨率），a11y 存 gzip |

## 模型服务
| 项 | 值 |
|---|---|
| 服务器 | LabServer（143.89.191.15），RTX 4090 D |
| 模型 | **GELAB-ZERO-4B**（vLLM 0.13.0，`--seed 0`，端口 8095，`/home/rxhuang/Projects/models/gelab_zero_4B`） |
| 访问 | 本地 SSH 隧道 `127.0.0.1:18095 → LabServer:8095`（`scripts/tunnel_keepalive.sh` 自动重连） |
| 请求格式 | OpenAI chat completions；system prompt + 文本 + base64 JPEG 截图；`temperature=0`，每个请求带随机 `cache_salt`（关闭前缀缓存的跨请求影响，见仓库已有结论） |
| 输出格式 | `<THINK>…</THINK>` + `action:CLICK point:x,y …` 等 key-value（`gelab_agent.parse_gelab_response`） |

说明：任务说明原定主 agent 为 MAI-UI 2B。实验开始时我在 GPU1 上拉起过一个 MAI-UI-2B vLLM，**按用户指示已停止**，全部实验改用服务器上已有的 GELAB-ZERO-4B。端口 8097 的 GELAB 属于其他用户，未使用。

## Agent
- `android_world/agents/gelab_agent.py`，与 4B_2 基线所用版本（commit 71b87b14）**逐字节一致**（`git diff 71b87b14 HEAD -- gelab_agent.py` 为空）。
- 实验 runner（`scripts/rgs_run.py`）没有改 prompt / parser / 动作翻译，只把 `GELABAgent.step` 拆成"决策（后台线程）"与"执行"两段，以便在推理窗口内并行探索；每步观测前保持原有 1.0 s transition pause，并增加有界 a11y settle（≤3 s，两次连续 dump 一致即停）。
- 步数上限：`min(20, AndroidWorld 复杂度预算)`。

## AndroidWorld
- 本地 registry 共 116 个任务；变体用官方 `suite_utils._instantiate_task(task, seed)`（参数和 app 数据都由 seed 决定，可复现）。
- 成功判定只在 episode 结束后调用 `task.is_successful(env)`，不输入 Explorer 或 agent。

## 关键仓库资产（复用）
- `env_launcher.load_and_setup_env`、`suite_utils`、`gelab_agent`、`infer.LlamaCppWrapper`、fast a11y socket。
- 仓库已有 MobileExplorer 的 rollback / graph 模块较重（为 LabServer Linux worker 设计，adb 路径 `/opt/android`），本实验只借鉴其教训（禁用 checkable、list-row 风险、按结构泛化不可恢复上下文），用独立的最小原型实现 probe–observe–rollback 与观测图，便于逐项消融。
