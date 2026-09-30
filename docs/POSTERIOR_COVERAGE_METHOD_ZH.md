# AndroidWorld 在线探索：方法与代码简述

## 方法在做什么

Android GUI agent 通常要先看屏幕、推理，再点控件。推理期间模拟器可能处于空闲状态。这里的想法是：主模型照常生成最终动作的同时，利用这段时间尝试少量安全的 UI 点击，让主模型更可能选择的页面/控件先被观察到；如果最终动作正好已经执行，就避免重复点击，否则尝试恢复到原状态再执行主模型动作。

最近增加的 **posterior coverage** 版本，把“该先探哪个控件”写成一个可测的选择问题：

1. 从当前无障碍 UI 树中筛出可点击、安全、非大容器的候选控件，并把控件中心转换成 agent 使用的坐标。
2. 对每个候选动作，在与完整推理相同的模型 prompt 后拼接该候选的标准动作文本，用 vLLM `prompt_logprobs` 读取坐标 token 的 log probability；对候选分数做 softmax，得到近似动作 posterior。它不是另一个模型的自由生成判断，而是同一模型对限定候选动作的条件打分。
3. 在完整推理尚未返回时执行候选 probe。`posterior_coverage` 根据候选概率质量和在线测得的点击/观察/恢复成本，用 0/1 knapsack 选择预计能在推理窗口内覆盖最多 posterior 质量的一组候选；每次 probe 后根据剩余时间重新规划。
4. 完整推理返回后，如果模型最终动作正是已 probe 的动作，就提交该动作且不重复点击；否则调用恢复逻辑，之后由主 agent 执行最终动作。

实验将其与四种对照比较：`none`（无探索）、`current`（已有关键词/图排序）、`random`（固定 seed 的随机选择）、`probability`（只按 posterior 从高到低选择、不考虑剩余时间），以及 `posterior_coverage`（概率质量与成本联合优化）。其中 `current/random/probability/posterior_coverage` 都计算 posterior；`none` 不计算。因此，后四者之间主要比较选择策略，而它们相对 `none` 的端到端变化也包含 posterior 和 probe 带来的额外成本。

## 和完整 MobileExplorer 的关系

仓库里更广义的 MobileExplorer 设计还包含可执行探索记忆：把观测到的 UI 状态、动作和后继状态整理为可检索的 evidence/transition，后续按当前任务和 UI 状态决定是否提供导航提示、验证动作或进行融合。语义前缀模块可以提供更高层的路线提示。

posterior-coverage 是其中“在线探索时如何选下一次点击”的一个独立、受控实验实现。`scripts/run_posterior_coverage_task.py` 临时包装原始 GELAB agent 的推理和 step 执行来记录数据；它不是 `MobileExplorer` 默认行为的开关，也不能据此声称完整 MobileExplorer 已采用 knapsack 调度。

## 运行边界与安全

当前实验 runner 在 `emulator-5554` 上执行真实点击，不是在隔离的 shadow emulator 中操作。恢复逻辑也可能失败。因此只能用专用、可恢复快照的 AndroidWorld AVD；一个 AVD 不可同时被多个 worker 使用。只有各 worker 拥有独立 AVD、ADB namespace、端口、快照和输出目录时才可并行。恢复失败或状态不明时应先停机检查。

方法指标和任务指标要分开看：probe 覆盖到最终动作、减少了等待时间，不自动意味着 AndroidWorld evaluator 判定成功。要报告 posterior/探测机制，也要报告官方任务成功率、配对任务结果、额外延迟和恢复失败。

对比实验固定完整请求 `temperature=0`、`top_p=1`，posterior 请求 `temperature=0`，vLLM seed `0`，AndroidWorld task seed `34` 和 `--fixed_task_seed`。不要通过改变采样设置或任务集合制造收益。

## 主要代码位置

| 文件 | 作用 |
|---|---|
| [`android_world/parallel_exploration/posterior_coverage.py`](../android_world/parallel_exploration/posterior_coverage.py) | 候选控件过滤、动作坐标/键映射、posterior 请求与 softmax、在线 probe 成本估计、剩余推理预算和 knapsack 选择、probe 后提交或恢复。 |
| [`scripts/run_posterior_coverage_task.py`](../scripts/run_posterior_coverage_task.py) | 单 task、单 method 的 AndroidWorld 实验入口；包装 GELAB 推理，记录每步推理、候选、probe、恢复、资源和耗时。 |
| [`android_world/parallel_exploration/live_probe.py`](../android_world/parallel_exploration/live_probe.py) | 共享的安全候选过滤、真实设备 probe、状态采集和恢复操作。 |
| [`android_world/agents/mobileexplorer.py`](../android_world/agents/mobileexplorer.py) | MobileExplorer agent 主体：任务 step、UI 观察、记忆检索和动作决策/融合。 |
| [`android_world/parallel_exploration/executable_memory.py`](../android_world/parallel_exploration/executable_memory.py) | 可执行探索记忆的数据结构、状态/动作关联、证据记录、检索和安全门控。 |
| [`android_world/parallel_exploration/semantic_prefix_memory.py`](../android_world/parallel_exploration/semantic_prefix_memory.py) | 语义路线前缀的记录、匹配和提示逻辑。 |
| [`tests/test_posterior_coverage.py`](../tests/test_posterior_coverage.py) | knapsack 最优性、概率归一化、动作坐标映射、final action 匹配、commit/recovery 和预算选择的单测。 |
| [`tests/test_executable_memory.py`](../tests/test_executable_memory.py) | executable memory 的状态/动作样本和检索门控相关测试。 |

## 输出如何读

每个 task/method 输出目录的 `steps.jsonl` 记录逐步机制数据，例如候选概率、posterior latency/errors、剩余预算、probe 实际成本、回滚、提交和恢复状态；`filtered_elements.jsonl` 记录安全过滤过程；`run_args.json` 记录实际命令和 latency priors。`--stats` 指向的 JSON 累积 reasoning/probe latency 先验，比较不同 arm 时应为每个 arm 单独保留一份。AndroidWorld 自己的 evaluator/checkpoint 决定 task 成败，不能用 `steps.jsonl` 行数或进程退出码代替。

可从 [`docs/CODEX_LABSERVER_ANDROIDWORLD_OPTIMIZATION_PROMPT_ZH.md`](CODEX_LABSERVER_ANDROIDWORLD_OPTIMIZATION_PROMPT_ZH.md) 复制 LabServer 操作 prompt。里面包含服务器检查、posterior API 预检、隔离 AVD、固定参数、分阶段运行、分析和单因素优化的要求。
