# AndroidWorld 20-task Orin E2E 对比

任务集：`configs/two_system_scaleup_40tasks.json` 的前 20 个固定分层样本（seed 30）。Baseline 与 FULL40 按任务交错运行；FULL40 图从空目录开始，仅在本轮任务间在线积累。

## 阶段：20 / 20 个配对任务

| 系统 | 完成数 | 成功 | 成功率 | 平均步数 | 成功任务平均步数 | 平均模型调用 | E2E 均值 | E2E 中位数 | E2E P90 | E2E 总时长 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 20 | 14 | 70.00% | 13.40 | 11.00 | 13.40 | 242.08 s | 215.39 s | 430.53 s | 4841.68 s |
| MobileExplorer / FULL40 | 20 | 10 | 50.00% | 12.15 | 7.40 | 11.65 | 211.53 s | 206.33 s | 351.80 s | 4230.60 s |

- 成功结果一致：14/20；配对平均 E2E 差（MobileExplorer − Baseline）：-30.55 s。
- 配对平均步数差（MobileExplorer − Baseline）：-1.25。
- E2E 定义：从启动本题 `full_run.py` 子进程到结果进程结束，包含 AndroidWorld 环境初始化、任务重置、agent 执行和成功评估；不含两题之间的调度间隔。
- 模型：Orin GELab-Zero-4B-preview；端侧常驻模型：Android VM llama.cpp Qwen2-VL-2B，Host 通过 adb forward 的 18096 端口并发触发。

Remote `/v1/models`: `GELab-Zero-4B-preview`；Android VM `/v1/models`: `Qwen2-VL-2B-Instruct-Q4_K_M.gguf`。
