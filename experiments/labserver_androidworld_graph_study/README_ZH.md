# LabServer AndroidWorld 图设计实验

这组配对实验将图机制拆成可归因的组件，目标不是单纯把步数压低，而是在相同任务实例和解码设置下保持 evaluator success rate，同时减少全部任务动作数和成功任务动作数。

## 对应的三个问题

1. **如何高效建图**：`graph_build_only` 只记录执行中观察到的 task-conditioned 状态、控件和转移，不让图干预 exploration 或 VLM prompt；`graph_guided_exploration` 进一步记录探索中新节点、probe 成本、回滚与图增长。图按 arm 隔离，并通过当前任务语义检索，避免跨设计和无关任务复用。
2. **如何用图增强 exploration**：主因果对比是 `graph_guided_exploration` 对 `probe_only`（两者使用相同 live-probe 预算）；图引导应让探测更集中于任务相关的未知节点/边、少走已知错误路径，并降低后续导航动作。
3. **如何用图增强 inference**：`graph_prompt_inference` 对 `graph_guided_exploration` 检查任务相关图摘要的提示注入；`graph_post_fusion` 单独测试图证据融合；三个 skip arms 检查高置信、逐跳在线验证的 route 是否能跳过重复 VLM 推理，以及恢复知识未知/落地身份粒度能否安全增加有效候选。

## 冻结控制与分析

详见同目录 [`protocol.json`](protocol.json)：30 个固定 AndroidWorld 任务、固定 task seed 34、`fixed_task_seed`、客户端 `temperature=0` / `top_p=1`、自启动 vLLM `seed=0`、相同模型与任务步数上限。默认使用 AndroidWorld gRPC accessibility；`SystemWifiTurnOffVerify` 的任务前置条件会关闭模拟器网络，因此该任务在所有 arms（包括 baseline）中统一使用本机 UIAutomator 采集，避免把网络断开误判为 agent 失败。其余任务仍使用 gRPC。每个 task 在所有 arms 下成对运行；arm 顺序用预先固定的随机种子打乱；图只在所属 arm 内跨任务累积。

主要输出包括 evaluator 成功率（含 Wilson 95% 区间；执行过动作或模型请求后的 evaluator 异常按失败计；若异常发生在首个动作前且 vLLM 成功请求计数可观测并确认仍为 0，则保留 attempt 并作为基础设施故障在新 attempt 重试）、成功率配对差 bootstrap 区间、全部 evaluator-complete episode 的总/平均动作数、成功任务步数、共同成功任务的配对步数差，以及 VLM 调用、probe、观察、rollback、skip、route 命中、图增长、可执行图状态/边更新耗时和 probe trace 摄取耗时。状态/边更新计时使用嵌套安全的计时器，避免一次转移中的多次 `observe_page` 重复计数；probe trace 摄取耗时是包含解析/包装工作的更宽时间，不与图更新时间相加。二者都不覆盖磁盘序列化成本。**只有成功率不低于 baseline 且全部任务与成功任务动作数均有下降，才算达到目标；成功率点估计相同不等于统计上已经证明非劣。**

## SSH 断开后的运行与恢复

在 LabServer 已检出此分支的仓库目录运行：

```bash
scripts/launch_labserver_androidworld_graph_study.sh graph-20261002
```

启动器使用 `nohup`/`setsid`，每个任务/arm 将有 append-only 事件、原子状态和独立 AndroidWorld checkpoint。遇到 SSH 断开不影响 supervisor；进程崩溃后可再次用相同 study ID 启动，已完成 checkpoint 会被接回，不覆盖旧 attempt。GPU 只有在显存不超过阈值、利用率近乎空闲时才会启动新的 vLLM；默认 8085 仅在请求计数连续一段时间不变时才作为候选复用，否则 supervisor 等待空闲卡，不会杀停或重配置别人的进程。

私有原始数据留在 `/data/rxhuang/android_world_server/androidworld_graph_studies/<study-id>/`。完整结束后生成 `report.md`、CSV、JSON 和 SVG 图表，并只将白名单聚合文件推送到 `reports/labserver_androidworld_graph_study/<study-id>/`；prompt、goal 实例值、截图、checkpoint、运行日志与服务器绝对路径不会提交。
