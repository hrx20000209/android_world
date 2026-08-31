# MobileExplorer 两系统原型（2026-08-28）

## 当前接入点

- 权威推理/执行：`android_world/agents/gelab_agent.py::GELABAgent.step` 与 `::_execute_action`；M3A 对齐入口为 `android_world/agents/m3a.py::M3A.step`。
- 并行探索：`android_world/parallel_exploration/live_probe.py`，独立进程、收到推理完成信号后停止并恢复。
- 真实运行器：`scripts/run_one_parallel_exploration_task.py`，负责 vLLM metrics、动态预算、belief graph 和 JSONL/CSV。
- 旧搜索原型：`android_world/agents/explorer_agent_gelab_light.py`。它不是新运行时主干，仅保留作算法参考。

## 架构

```text
task/session
  |
  +--> ProgressiveBeliefGraph (task-local, persistent)
  |
  +--> SkipInferenceGate -- verified/fresh/safe? --> execute + verify
  |                              |
  |                              +-- mismatch --> invalidate subtree --> inference
  |
  +--> inference round
         |
         +--> Inference System (authoritative action)
         |
         +--> InformationNeed + ResourceAdaptiveBudget
                    |
                    +--> ExplorationPolicy --> speculative GUI process
                                              |
                                              +--> preemption/barrier
                                              +--> recover + StateVerifier
         |
         +--> alignment --> commit main action --> execution verification --> graph
```

## 已实现

- `ProgressiveBeliefGraph`：node/edge、六种 edge 状态、alignment、execution hit/miss、stale、subtree invalidation、frontier、本地置信度、entropy 和序列化。
- `InformationNeed`：结构化对象、JSON parser、task+UI fallback；缺失不会中止任务。
- `ResourceAdaptiveBudgetController`：内存/进程 RSS/CPU、可插拔 Jetson sampler、utility/reliability/interference 控制，严重压力自动返回零预算。
- `StateVerifier`：activity/package/structure/visual/selected-state 联合验证，区分 visual 与 semantic recovery。
- `SpeculationCoordinator` / `SpeculationBarrier`：GUI ownership、停止信号、恢复验证、主动作 commit barrier。
- `ExplorationPolicy` / `InformationGainPredictor`：可替换接口及 deterministic heuristic 第一版。
- `SkipInferenceGate`：只接受 VERIFIED、低风险、低 entropy、新鲜且高置信边；失败后 invalidates subtree。
- 每轮 JSONL/CSV：推理 action、prefill/decode/inference（request 文件）、预算、probe、depth、rollback、preemption、资源、confidence/reliability。
- M3A summary 可通过 `ANDROID_WORLD_M3A_DISABLE_SUMMARY=1` 关闭，不改变默认 baseline 行为。
- 所有新功能都有 feature flag；`skip_inference` 默认关闭。

## 当前主动探索安全策略

不使用任务关键词黑名单。只有 action/provider 明确给出 `reversible` 或 `navigation_semantics` 元数据时才允许真实 probe。普通 A11y Button、TextView、ImageButton 都是 `UNKNOWN`，仅可进入候选/日志，不执行。原因是实测：Stopwatch Start 会产生 latent side effect；Launcher Google Lens 虽是 ImageButton，Back 也不能恢复 app drawer committed state。

## 尚未完成/不能宣称完成

- 当前 flat A11y provider 没有 reversibility 元数据，所以保守 gate 下多数 round 为零 probe；需要 state-action queue/replay certificate 后才能安全扩大覆盖。
- `ExplorationPolicy` 已模块化，但 live device loop 仍沿用现有 ranker；尚未把多深度 frontier 扩展完全迁入 policy。
- skip gate 和失效逻辑已实现并测试，但真实运行默认 shadow-only；尚无经过两次验证、可主动跳过的 edge。
- Jetson GPU/功耗/温度 sampler 是接口，当前 macOS runner 没有读取远端 tegrastats。
- interference 计算接口已实现，但还缺同一模型/同一 prompt 的交替 solo/concurrent 校准，因此日志当前 estimate 为 0，不能声称 exploration resource-neutral。

## 配置

集中配置见 `android_world/parallel_exploration/config.py`。主要项：`enabled`、`exploration_enabled`、`skip_inference_enabled`、`max_probes`、`max_depth`、`max_exploration_time_s`、内存/CPU压力阈值、interference threshold、reusable confidence、max entropy、freshness 和 recovery timeout。

## 运行命令

GELAB smoke：

```bash
python scripts/run_one_parallel_exploration_task.py \
  --output results/two_system_clock \
  --task ClockStopWatchRunning --max_steps 10 \
  --agent_name gelab_agent --two_system --a11y_method fast_provider
```

M3A baseline 对齐（summary 默认在此 runner 中关闭）：

```bash
python scripts/run_one_parallel_exploration_task.py \
  --output results/two_system_m3a_clock \
  --task ClockStopWatchRunning --max_steps 10 \
  --agent_name m3a_llamacpp --two_system --a11y_method fast_provider
```

输出包括 `request_latency.jsonl`、`inference_windows.jsonl`、`two_system_rounds.jsonl/.csv`、`probe_trace.jsonl`、`filtered_elements.jsonl` 和 `progressive_belief_graph.json`。

## 可学习替换点

当前 path probability、predicted IG、future value、risk metadata 和 confidence update 都是轻量 heuristic/interface。后续可用 trace 中的 predicted IG、realized IG、alignment、execution hit/miss 训练 listwise ranker/calibrator；不需要在 explorer 中再放一个大模型。

