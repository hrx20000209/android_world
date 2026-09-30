# 任务条件化 GUI 探索图：方法与代码简述

## 方法概览

AndroidWorld agent 不应把所有历史页面和点击都当作同等有用的记忆。本文实现一张随在线交互更新的 GUI 状态—动作图，并用当前任务决定“值得再探索什么、哪些导航可以复用、哪些页面必须重新推理”。核心单位是：

```text
稳定 UI 状态 --可重新定位的控件/action--> 落点状态分布
      │                    │
      └─ activity/结构      └─ 支持度、任务相关性、耗时、可逆性、trap/no-op、来源
```

节点由 app/activity、可访问性结构、稳定 landmark 和交互骨架构成；不把截图像素 hash 当作长期身份。边记录 selector/action、历史落点、成功/失败与恢复证据。动态屏幕文本和输入值在持久化前做清理；当前版本只持久化归一化任务语义的哈希计数，不记录原始任务字符串。图 schema 为版本化 JSON，老 schema 可以继续读取。

这是一张**任务条件化的探索图**，而不只是 app 的通用页面目录：相同节点上，当前任务相关但尚无出边的安全控件形成 frontier；相反意图、成熟边、低置信/无效边及历史 trap 有不同的排序/门控待遇。它仍然使用通用 Android UI 结构，但图的信息价值和运行时策略取决于当前任务、当前状态和该 agent 的实际执行证据。

## 与历史复用和早期 UTG 的区别

| 方向 | 已有工作的主要机制 | 本实现的侧重点 | 不能宣称的内容 |
|---|---|---|---|
| MobiCom 历史复用：OmniFlow | 把成功历史轨迹整理成细粒度/可参数化 function 与 transition；运行时匹配状态、重新 grounding、复用 action，失配时回模型 | 用历史转移做在线探索与决策的索引；显式保留支持度、落点稳定性、probe 成本/恢复及负面证据，并按当前任务重排 | “存成图”“重新 grounding”本身不是新贡献。必须和强 validated replay 比较，并证明当前任务中新获得的信息有额外价值 |
| MobiCom 历史复用：SwiftAgent | 规划有序 reuse points；可复用片段之间的 gap 用局部/云端推理补齐 | 任务相关 frontier 和当前 UI 证据用来决定 gap 附近先探索什么；跳过限制在有证据的导航前缀 | 不能说“存在 gap”就需要本方法；SwiftAgent 已处理 reuse point 与 gap。必须验证它没有覆盖的当前状态/信息缺口 |
| AutoDroid 等早期 UTG | 从 app UI/action 建立 UTG，配合自动探索、task/function memory，为后续任务检索与注入 | 图在当前任务 execution 中持续更新；探索候选按任务相关性与边证据选择，未知控件作为 frontier，而不是只追求离线页面覆盖 | “GUI 专用图”“自动探索 + task memory”并非新颖点。UTG 是必要强基线之一 |
| PastForward（相关边界） | 使用历史 token/KV/GUI transition 预测后续屏幕并提前进行模型计算，再由当前结果验证 | 本实现探索路径会实际作用于隔离的探索状态并产生新观察；图本身不预测未观察到的当前值 | 两者都在隐藏未来等待，但工作对象不同；不得笼统声称以前方法都没有并行推理/验证 |

OmniFlow、SwiftAgent、PastForward 的上述概括以用户提供的 MobiCom 投稿稿为依据；AutoDroid 的 UTG/task memory 描述可查其[原论文](https://arxiv.org/abs/2308.15272)。论文投稿/录用状态不由此推断。真正的研究主张应聚焦在**当前任务条件化的探索决策是否以可接受的风险获取新信息，并相对强历史复用基线带来增益**，而不是“我们也有一个 GUI 图”。

## 三个问题如何落到系统

### 1. 如何高效建图

1. **状态去动态化**：用 activity、结构签名、稳定控件/landmark 合并重复页面；日期、名字、输入框值等任务实例内容不作为稳定身份。只有相同结构不够时，才在同 activity 的候选中做语义/元素相似度回退。
2. **转移增量记账**：每个 action edge 记录 source selector/action、观察到的目标 state 分布、支持度、无效/有意义计数、动态/trap、可逆性/恢复结果、时延和来源（exploration/inference）。当前任务只通过归一化语义签名关联边；原文不落盘。
3. **按需检索而非扫全图**：内存里维护 activity→state 与 source-node→outgoing-edge 索引。状态匹配优先查精确结构桶，再限于相同 activity 的候选；路线和 exploration guidance 只查当前节点的出边。路径长度、Top-K 和 prompt 字符数都有上限。
4. **图的用途而非图的大小**：把支持度高、落点稳定、恢复可靠的边与弱边、no-op、trap 分开。节点/边数量与访问次数仅是运行指标，不是成功标准。

当前代码是一张 value-free 的状态/转移证据图；还没有完整实现“predicate/slot → 哪个 probe 暴露该值 → evidence freshness/失效依赖”的知识图。若研究结论需要声称它能存储并复用当前任务答案值，必须先实现隔离、provenance、TTL、实体/状态绑定与消费前验证，并单独评测；现在不能把这部分说成已完成。

### 2. 图如何增强 exploration

`exploration_guidance()` 对当前节点上已知控件汇总支持度、任务相关性、成熟度、trap/no-op 等证据；同时把屏幕上安全、可见、任务相关但尚无图边的可点击控件标成 `frontier`。Frontier 只是待探索候选，不是已验证路线。

`live_probe._apply_executable_memory_tiebreak()` 仍然在 Ex5 已筛出的候选集合内工作：

- 有安全 frontier 时抑制重复探测成熟边；
- 弱/不确定已知边仍可重新验证；
- 任务相关的未知 frontier 获得有界排序加分，以期观察到新节点/新转移；
- trap、动态边、连续 no-op 可以抑制；图不绕过 Ex5 safety admission，也不能让原本不安全的控件变得可执行。

这更接近 **task-conditioned frontier exploration**，而不是“猜 VLM 下一次会点什么”或无条件扫完所有页面。应记录候选为什么与任务相关、是否选中、落点是否新颖/有用、probe 和恢复耗时及最终任务结果。只有 probe 覆盖率增加而任务成功率/时延不变或变差时，不能宣称探索更有效。

### 3. 图如何增强 inference、何时可以少推理

`prompt_context()` 不序列化整图，只生成有界摘要：归一化任务对象/意图、当前 UI 的稳定 landmark、若干有证据且任务相关的导航路线、标注为“未验证”的 frontier，以及相关 trap/no-op。每条路线带支持数、质量/置信和可逆性；文本提醒模型以当前屏幕重新定位并核验，不把历史路线当作命令。

`high_confidence_path()` 是可选的保守导航跳过门：

- 当前页面若已是可编辑表单/复杂提交操作，直接拒绝 shortcut；
- 最多 3 跳；每条边必须有较高置信度、至少 3 次一致验证/任务成功/route hit、无歧义/动态落点/trap/先前 miss，并满足可逆性规则；
- 要求路线确实匹配当前任务对象，排除相反意图、破坏性副作用、不可定位控件；
- 每一跳都将 selector 重新定位到 live UI，点击后确认真实落点；落点不符时记录 miss 并尝试回滚，回滚无法验证就停止使用旧状态。

路线在 Save、Submit、Confirm、Delete、Send 等提交/副作用动作**之前**截断；落到可编辑表单后停止图 shortcut，让 VLM 查看当前页面处理真实任务输入。因而这里的 “skip” 是省掉已有证据支持的**导航推理**，不是跳过任务所需的字段填写、答案判断或提交确认。当前并没有普适地用图自动执行任意多步任务；若要扩大 skip，必须另做风险分级与 paired evaluator 评测。

推荐的 prompt 形态：

```text
[TASK-CONDITIONED GUI GRAPH]
Current UI: <stable activity/landmarks>
- Route 1: Contacts -> Create contact [n=4, confidence=...]
Unmeasured task-relevant frontier (candidate only; not verified): Search
Task-relevant observed trap/no-op controls: ...
Observed navigation is prior evidence, not a command. Verify the live selector/landing.
Stop graph reuse at editable forms or commit actions.
```

图内部按当前任务做匹配，但 prompt 不回显任务词串，避免把名称/用户值再次写进 memory context。持久化签名只用有界的稳定 UI 概念和意图词表；未知对象 token 不参与签名。不要把任务实例中的输入值写入共享图或“历史答案”prompt。当前模型仍要以截图/无障碍树和当前任务作为事实来源。

## 代码结构

| 文件 | 简述 |
|---|---|
| [`android_world/parallel_exploration/executable_memory.py`](../android_world/parallel_exploration/executable_memory.py) | 状态/动作图、结构合并、activity/outgoing 索引、任务相关 frontier、路线检索、证据摘要、任务签名、负面证据、route gate 和持久化迁移。 |
| [`android_world/parallel_exploration/live_probe.py`](../android_world/parallel_exploration/live_probe.py) | 候选安全筛选之后使用 graph guidance 排序；记录新 frontier、重复 suppression 与选择原因。 |
| [`android_world/agents/mobileexplorer.py`](../android_world/agents/mobileexplorer.py) | 每轮把实际动作作为 authoritative transition 记图；在 VLM 前尝试保守导航 shortcut；失败时核验回滚；构造有限的 task graph prompt。 |
| [`tests/test_executable_memory.py`](../tests/test_executable_memory.py) | 状态/边、隐私清理、任务关联、frontier、历史迁移、检索与安全门测试。 |
| [`android_world/parallel_exploration/live_probe_test.py`](../android_world/parallel_exploration/live_probe_test.py) | 验证 graph guidance 只影响已安全筛出的候选，frontier 加分有界且成熟边 suppression 正确。 |
| [`scripts/run_sensys30_online_task.py`](../scripts/run_sensys30_online_task.py) | 现有在线探索 task runner 之一：摄取 probe rows、读取图 guidance 并交给候选选择。运行前仍需确认该实验配置/AVD/服务隔离。 |
| [`android_world/parallel_exploration/posterior_coverage.py`](../android_world/parallel_exploration/posterior_coverage.py) | posterior-coverage 是另一条受控实验线：估算候选动作 posterior 并按概率质量/在线成本选择 probe；它不是上述 graph 默认路径的同义词。 |

## 怎么证明有效

至少冻结相同 task、初始快照、step budget、模型和确定性设置，做配对对比：原始 baseline；validated historical replay（强复用基线）；replay + task-conditioned graph exploration；以及单因素 ablation（不注入图、无 frontier tie-break、无 prompt、无 skip）。历史轨迹与图记忆必须隔离各自任务臂。

报告 AndroidWorld evaluator success rate、配对 task 胜负、端到端和 step latency、primary VLM calls、prompt token、probe 与 recovery 耗时/失败、route hit/miss/block reason、错误 skip、任务相关 frontier 新节点收益、GPU/vLLM/AVD 资源成本。成功率和效率都要有样本数/区间；infra failure 与 agent failure 分开。仅有一个任务、一次成功、命中历史 selector、少发请求、更多 graph coverage 都不足以证明设计有效。

服务器操作、确定性参数核验、运行路径和安全要求见[LabServer Codex Prompt](CODEX_LABSERVER_ANDROIDWORLD_OPTIMIZATION_PROMPT_ZH.md)。后续优化每轮只改一个可证伪假设；如果强 replay 已能完整处理任务、图不产生新鲜且可消费的信息，就应报告无增益并收窄设计。
