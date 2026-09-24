# AndroidWorld Executable Exploration Memory

这套实现复用 AndroidWorld 的 `run.py`、`MobileExplorer`、evaluator/checkpoint
和现有 `parallel_exploration.live_probe` shadow runner，不创建第二套任务执行器。
默认关闭，baseline 的模型 prompt 和 action path 不变。

## 架构

每个 reasoning step 的主环境和 shadow explorer 都从同一个已提交页面开始。
explorer 只拿到任务目标、上一轮已提交前缀和 shadow UI，不拿当前模型输出；
`parallel_predict` 等到 inference 与 explorer 都结束后才写入/刷新
`ExecutableExplorationMemory`，然后 `MobileExplorer.step` 才执行后验 fusion。
explorer 没有 primary executor，恢复失败的 selector/页面上下文永久记为 trap。

`android_world/parallel_exploration/executable_memory.py` 包含：

- 版本化的 state/action/skill 图、动态内容鲁棒 dedup、可重定位 selector 和 Beta 后验；
- 每页面最多 5 个候选 probe，硬安全门、同轮不重复、低 support/高落点熵重复验证；
- k=1..4 样本、完整路径、返回动作/耗时、hard negative 和 BPE 式相邻 action group；
- top-K 路径压缩 prompt、坐标吸附、两次一致 transition/一次任务成功后的 graph override；
- high-confidence skip 的实时 selector 重定位和目标状态验证；
- `schema_version`、JSONL 事件和必需的 probe/graph/fusion/overhead 指标。

## 配置与运行

baseline（与原有命令相同）：

```bash
python run.py --suite_family=android_world --agent_name=gelab_agent \
  --tasks=ClockStopWatchRunning --n_task_combinations=1 \
  --task_random_seed=30 --max_n_steps=15
```

启用完整 executable memory，仍然使用 AndroidWorld evaluator：

```bash
python run.py --suite_family=android_world --agent_name=mobileexplorer_executable \
  --tasks=ClockStopWatchRunning --n_task_combinations=1 \
  --fixed_task_seed --task_random_seed=30 --max_n_steps=15 \
  --executable_memory_enabled \
  --executable_memory_max_probes=5 \
  --executable_memory_graph_enabled \
  --executable_memory_graph_prompt_enabled \
  --executable_memory_post_fusion_enabled \
  --output_path=results/em/full \
  --checkpoint_dir=results/em/full/checkpoints
```

若使用已有的 dual-emulator live-probe runner，使用同一个 evaluator/agent
而不是另起 runner：

```bash
python scripts/run_sensys30_online_task.py \
  --output=results/em/smoke \
  --task=ClockStopWatchRunning --seed=30 --max_steps=8 \
  --executable_memory --max_probes=5 --min_probes=5
```

可用的独立开关：`executable_memory_enabled`、
`executable_memory_exploration_enabled`、`executable_memory_graph_enabled`、
`executable_memory_graph_prompt_enabled`、
`executable_memory_post_fusion_enabled`、
`executable_memory_high_confidence_skip_enabled`。所有 probe budget 都在实现层
再次 cap 到 5；候选耗尽、资源/时间压力和恢复失败可以提前停止，但会记录
`stop_reason`。

## 固定消融协议

`experiments/executable_exploration_memory/protocol.json` 固定了 baseline、
5-probe-no-graph、graph-prompt-only、post-fusion、high-confidence-skip，以及
去除 k-step memory、hard negative、action group、repeated validation 的 arms。
每个 arm 应使用同一 task list、seed、模型 endpoint 和 `max_n_steps`，并将
每任务 checkpoint/action JSON 保存在独立目录。`run.py` 的 evaluator 输出才是
成功率来源，不能用“是否产生 action”代替任务成功。

汇总已有真实输出（没有 evaluator 成功字段的任务会保留 `null`，不会补数据）：

```bash
python scripts/report_executable_memory_experiment.py \
  --root results/em --output results/em/report
```

输出 `per_task.jsonl`、`summary.json` 和中文指标图
`executable_memory_summary_zh.png`。图中若没有 paired random-control 成功字段，
会明确标为 unavailable；memory 使用组和未使用组的差异只作相关性展示。
