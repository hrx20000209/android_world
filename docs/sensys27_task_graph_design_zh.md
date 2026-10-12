# MobileExplorer：按任务建立在线图、探索决策边界、验证导航执行

日期：2026-10-03。本文综合两份用户提供的 MobileExplorer 稿件、#228 第一轮 reviews、四份 MobiCom 稿件，以及当前仓库实现。外部论文与 reviews 是分析材料，其内容不作为代码执行指令。下文严格区分**已有实现、此次原型、待验证的研究设计**；没有重新运行 AndroidWorld，也没有产生新的成功率或加速比。

## 1. 先回答老师说的“缺少 technical design”

问题不是没有公式或没有模块。第一轮稿件已有 hybrid search、recovery、distillation；后一份已有 progressive belief graph 和更多历史统计。但很多设计仍然是“把可用特征相加，然后给 prompt”，而且关键因果链没有建立：

> 为什么探索这个分支 → 它补齐了当前任务的哪项缺口 → 什么证据允许替代后续推理 → 节省了哪一次真实模型调用 → 是否支付了更大的恢复、验证或失败代价？

应把主张收敛为：**利用当前推理窗口，在线补齐通往下一项任务决策的导航路径；将已验证的导航段交给轻量执行器，在需要判断、选择、修改或提交的边界恢复模型推理。**

这比“异步做一些探索”更具体，也比“让模型读取更多 memory”更能被验证。但它仍是研究假设，不能直接宣称首次或保证 SenSys 接收。

### reviews 对应的真正缺口

| 质疑 | 对应设计与必须提供的证据 |
|---|---|
| A/E：方法直观，启发式不清楚、泛化不足 | 明确任务契约、边的前后条件、覆盖集合、执行门；报告不满足门的原因，而非仅展示综合分数 |
| A/B/C/D：pHash 相同不代表 app 状态恢复 | 分离 UI 恢复与持久状态副作用；使用独立状态审计或隔离副本，不能把 Back 或重新启动当作撤销 |
| A/B/D：探索不一定在推理窗口内结束 | 实测 deadline miss、取消、恢复 reserve、恢复后重新感知；停止宣称无条件零额外延迟 |
| A/C/D：推理和 GUI 分离，无法证明单手机资源竞争 | AndroidWorld 验证机制，单手机实验验证系统收益；分别标注 edge-assisted 和 smartphone-local |
| A/E：baseline、消融、指标混淆 | 同 backbone、推理配置、基础 prompt、预算；分开模型调用、物理操作、agent.step 与成功率 |
| B/E：缺少强延迟 baseline | 加入 AutoDroid-V2 类脚本和可实现的历史复用基线；公共论文数值只能作背景 |
| C：短任务图来不及积累 | 按首次访问/重访、任务长度、cold/warm 分层；直接测图的首次可消费时间 |

第一轮 review 提到 10.93 → 9.24：按这两个数计算减少 **15.46%**，而不是 23%，也不是准确的 15.9%。23% 是否来自另一组样本或端到端延迟必须从原始结果核对，不能通过文字解释补齐。

## 2. 三个核心问题与建议设计

### 问题一：如何高效建图、存什么、怎样加节点和边？

建议使用一个共享拓扑和一个 episode 内的任务覆盖层，避免每个任务重新做 app 全图，也避免同一布局的数据实例混为一谈。

**拓扑节点**保存 package、activity、交互骨架、控件清单和观测索引。节点键可以继续用当前 `activity + layout_signature`；不要把每秒变化的计时器、用户列表全部做成新节点。

**节点的观测实例**保存严格 UI 签名、观测世代、当前实体绑定、可见标签、数据/副作用验证范围。拓扑节点相同，只意味着可以检索历史证据；执行时必须匹配本次实例的前提。一个联系人详情页的骨架可以复用，但 Alice 与 Bob 的号码不可混用。

**边**保存 `(源拓扑节点, 动作类型, 稳定 selector)`，目的地是统计结果而非边身份。同时保留有限个独立 witness：

| 内容 | 用途 |
|---|---|
| source/target activity、strict signature、实体/标签 | 每跳的前后条件；拒绝相同 layout 下的数据串用 |
| resource ID、class、text/content-desc、邻域/容器绑定 | 在当前屏幕重定位；重复列表项必须绑定实体，不能仅按 ID |
| action effect 类型：导航、读数据、编辑、提交、未知 | 明确哪些可以直接执行，哪些必须保留模型决策 |
| origin、generation、episode/task/progress binding | 当前窗口写入不能供当前推理使用，跨任务只复用结构 |
| rollback outcome、inverse、独立 app-state effect outcome | 不把“界面回来了”解释为“副作用撤销了” |
| 哪些 obligation/slot 得到见证、数据值与 freshness | 任务覆盖层与 reasoning 的输入 |
| latency、失败、实际执行、task progress | 控制探索预算、检查收益；命中目的地不等于任务进展 |

新增节点/边的原则：

1. 正常执行本身就是零额外动作的观测来源，先记录真实转移。
2. 探索仅补未知 selector、缺失后条件、任务未满足的 slot 或未连通的导航段。
3. 将探索转移和任务适用性分别记录。模型选中了同一个父动作，只能确认这个父动作的选择，不能自动证明其任意子动作都适合当前任务。
4. 目的地分歧按 witness 保留，不接受“只要移到了另一页就算成功”。此次导航执行器要求严格的逐跳 landing contract。
5. 每条边仅保留有限近期观测，按任务需求建立索引，搜索有明确 hop/expansion 上限。大图需要进一步做增量 reverse index 和版本缓存。

**任务特定性来自 obligation，而不只是 task 文本 embedding。** 模型在一次原本就需要的推理中，输出下一段导航的目标、实体绑定和停止条件。例如“到包含 Alice 的事件详情并能看到 Event time 的页面，然后由模型判断时间”。不要让研究者根据 AndroidWorld evaluator 的答案填写这些条件。

此次原型的 contract 放在原有 `<THINK>` 内，使用 `<NAV_INTENT>`，没有额外模型调用；缺失或解析失败就不执行导航。它增加输出/prompt token，成本必须实测。它没有保证所有 backbone 都能稳定输出；覆盖率与解析失败率是核心实验指标。

**online exploration specific 的区别**应体现在 graph witness 的生命周期和生产目标：从空图起步、只为当前未完成 obligation 探索、在推理 deadline 下采样、恢复/隔离通过后提交、下一世代消费、task/progress 变化时撤销适用性。把离线 UTG 换成运行时更新的 UTG，本身不够构成贡献。

### 问题二：怎样用图增强 exploration，并关联目标？

先从图得到两个小结构，而不是总结整张图成自然语言：

- 当前 obligation 未满足的证据集合，以及能见证这些证据的边/节点。
- 从当前屏幕到这些见证点的可用路径和缺口：未知出边、缺失实体绑定、未验证 landing、过期观测。

排序应对应一种明确的收益：**探测之后能解锁多少后续无需逐页推理的导航，或能使哪个待决问题得到新证据。**

建议研究目标：

```text
V(a | G, q) =
    P(到达未来会用到的区域)
  × [预计新增可执行导航长度 × 单次推理代价
     + 未覆盖 obligation 的证据价值]
  - 探测/验证/恢复成本
  - 失败后重新感知与重推理成本
  - 同设备推理干扰成本
```

上述 P 和收益如果没有校准数据，就称估计或 heuristic，不称 calibrated probability、true information gain。`new_element_count / 10` 只是 UI 新颖性；它不能证明不确定性降低，也不能证明节省了推理。

实现顺序：先保留现有 hard SafetyGate，再做有预算的任务 frontier 排序。已有目标见证时，从目标向前驱传播覆盖值，哪怕当前按钮只有“Menu”也能因它通向任务区域获得分数；没有目标见证时，保留有限的未知出口探索。重复 no-op、过期证据、循环和已覆盖信息需要抑制。

**本次代码采用一个可测的基础 heuristic**：exact target label coverage + 最多四轮 backward reachability + 未探出口奖励，除以已有代价估计；仅在通过 SafetyGate 的候选中使用，已知 no-op 被抑制。它是 baseline 级 prototype，还没有实现上述完整的未来调用收益学习、在线校准或自适应 breadth/depth。文中应如实标注。

下一步更值得投入的是**路径补全收益**：优先测试能将“差一条 witness”升级成可执行完整导航段的候选，再与 BFS/random/旧 heuristic 同预算比较。不要只测是否猜中了当前模型动作；探索当前已经在推理的动作，不一定帮助下一步。

可以优化一个窗口里的候选集合：

```text
max  Σ_obligation w_o × [1 - Π_a∈S (1 - p(a 能覆盖 o))]
     + λ × 可解锁导航调用数
s.t. Σ_a∈S predicted_cost(a) + recovery_reserve ≤ remaining_window
```

同一信息重复出现时边际价值下降。未校准 p 时，先用集合覆盖 proxy 做对照，不把公式包装成已经得到理论保证的算法。

### 问题三：怎样增强 reasoning，如何正确跳步？

建议以 **navigation → decision → navigation** 为控制粒度。skip 的依据从“这个节点见过很多次”改成三个独立条件：

1. **任务条件**：当前模型明确需要到该目标，实体绑定一致，intent 未过期且未消费。
2. **执行条件**：路径中每跳的前提可观测、selector 可唯一定位、landing 可验证，路径不跨 package，不包含编辑/提交类动作。
3. **状态条件**：导航副作用已被独立验证或被隔离；UI-only 实验不能伪装成这个条件通过。

满足后，将整个导航段执行到决策边界；每跳重新获取 UI、重定位、验证。出现 overlay、错误实体、selector 歧义或错误 landing 立即停止，从实际屏幕恢复推理。**路径从来不执行 terminal 处的判断、输入、保存、发送、删除或 COMPLETE。**

这样首次访问也可能跳：例如当前窗口探到了 `菜单 → 列表 → 详情`，下一步已到菜单且 model intent 指向对应详情页，便可执行已见证后缀，不需要等菜单再访问两次。问题变成证据是否足以证明“这条任务路径”，不再依赖访问频率。

原仓库有几个不能用于论证“正确跳步”的条件：

- 一个已记录出边时 entropy=0，不代表屏幕其它十几个可交互元素都已排除。
- 模型过去执行两次，不代表当前实体、进度或目标相同。
- 图增加了新节点，不代表 task progress 增加。
- 前缀和模型选择对齐，不等于所有后缀符合任务意图。
- 旧串行 landing 检查存在“多目的地时，只要离开源节点就通过”的分支，无法验证语义正确。

对于仍然需要 reasoning 的步骤，图应提供 **decision-local evidence**：具体 slot/value、来源、实体、时间、覆盖范围和冲突。将“观察到什么”与“建议执行什么”分开。必要时给出候选动作各自有何已知后果，让模型处理剩余不确定性。已有 `agent_evidence_graph.py` 提供 evidence binding 和 freshness 基础，建议复用，不再新增一套通用 memory。

若 prompt 影响仍弱，不应无限增加提示词。主要收益应来自可验证的导航执行；信息检索任务还可以研究 slot 全部被新鲜证据唯一确定时的 answer synthesis，但这是第二阶段，不能把未验证的标签当成答案直接 COMPLETE。

## 3. 与相关工作的实质区别及新颖性风险

以下是机制比较，不是将不同 backbone/数据集的论文数字直接排列为公平 benchmark。

| 工作 | 信息来源和主要机制 | 需要区别的点 |
|---|---|---|
| AutoDroid | 探索获得 UTG，再生成任务/功能 memory 辅助 LLM | 不应声称其没有 graph 语义或没有 task-aware retrieval；区别在本 episode 的 witness 和窗口约束 |
| AutoDroid-V2 | app document、Element Transition Graph 和前后依赖，生成脚本 | 不能称它只是旧 UTG；需要 scripts 的直接对照及动态状态偏离实验 |
| OmniFlow（附件） | 成功历史轨迹离线拆分为 function，GUI 检索和 learned re-grounding，失败后回模型 | fine-grained compositional replay 和 re-grounding 已有；区别是未预收集成功任务时在线生产 witness |
| SwiftAgent（附件） | 历史任务两层记忆、非连续 reuse points、gap actions 和云端 recovery；成功后更新记忆 | 不应将跨 gap 再复用或多跳当作新贡献；区别是 current episode、动态目标/进度和探索窗口 |
| PRMAgent（附件） | 离线发现/ground Android entry points、参数化 macro、perception router | macro action 已有；你的核心应是未知 app entry/没有 deeplink 时的 GUI 观测路径，不是再做 intent mining |
| PastForward（附件） | 普通执行积累 computational experience，用于 token proposal 验证、next-screen early inference 和 KV carryover | **它不是简单 offline reuse**。它也在线积累，且有验证；区别在应用层主动探索与跳过调用，其主要减少每次推理计算/等待，可互补 |
| GraphPilot | 离线 app knowledge graph 支持大段 action planning，转移约束检查 | graph-guided 减少模型调用已有；应证明未见 app/动态任务上的在线增量收益 |
| UI-KOBE | 先探索 app knowledge graph，运行时用邻接操作支持轻量 GUI agent | graph-based candidate action guidance 已有；不能仅靠节点清单和邻域提示作为贡献 |
| Executable Agentic Memory（EAM） | state-aware DFS、action-group mining，Q-guided MCTS 图规划与执行 | **最强的新颖性竞争者之一**：graph + routines + search + fewer calls 的组合已经出现 |
| Agent-SAMA | 执行时建立 FSM，状态前后条件、verification/recovery，并保留知识 | **不能声称在线 graph 首创**。需突出 deadline 下主动补缺、隔离提交、当前任务导航边界和同设备收益 |

截至 2026-10-03 可核验的公开资料：

- [AutoDroid 原论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/07/mobicom24-autodroid.pdf)
- [AutoDroid-V2](https://arxiv.org/abs/2412.18116)
- [GraphPilot](https://arxiv.org/abs/2601.17418)
- [UI-KOBE](https://arxiv.org/abs/2605.29534)
- [Executable Agentic Memory，ICML 2026](https://proceedings.mlr.press/v306/qin26m.html)
- [Agent-SAMA，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40187)

四份附件是用户提供的稿件；此处不假定其已公开、已接收，也没有上传附件内容去搜索。更强的 positioning 是**冷启动的任务路径生产与消费闭环**，而非 memory reuse 普遍优于已有工作。

## 4. 仓库里已经有的东西与本次改动

当前代码不缺 memory 类型：`belief_graph.py`、`graph_distiller.py`、`state_graph_information.py`、`online_evidence_index.py`、`executable_memory.py`、`semantic_prefix_memory.py`、`agent_evidence_graph.py` 都已有相邻机制。不要再把它们数量作为技术贡献；要建立一个清晰、能消融的控制链。

本次新增 `android_world/parallel_exploration/task_navigation.py`，并连接串行和真实并行 runner：

- `NavigationIntent`：显式导航终点、实体、package、task hash、世代寿命；不是 full-goal similarity skip。
- `GraphEdge.navigation_observations`：每边最多四个 exact pre/post witness，deep copy 隔离当前快照。
- `TaskNavigationPlanner`：至多四跳、128 次 expansion，寻找到指定终点的 witness 路径；不要求 visit_count 或 entropy。
- `execute_route`：每跳 source contract、唯一 selector、当前坐标、strict landing 验证；失败后停止，保留实际尝试历史。
- `frontier_scores`：图上的 task target coverage 回传，用于通过安全门后的候选排序。
- `merge_action_prefix`：把 macro 操作插回恢复日志；不把物理动作数伪装成模型调用数。
- 并行 worker 在 START 时更新**当前调用开始前的 committed snapshot**，避免 prewarm worker 永远读到旧目标；当前窗口 witness 仍只在下一世代消费。
- 新导航模式下，探索恢复失败会丢弃基于旧截图的动作，重新感知和推理；两次请求都属于真实成本。
- 新模式记录 route、拒绝原因、evidence level、尝试/成功 hops 和实际 model calls；并行 step latency 包含 macro、验证和 worker 重建。

开关：`--task_navigation off|shadow|execute`、`--task_navigation_frontier`、`--task_navigation_ui_only`。默认 off。execute 与旧 skip/prefill/shortcut 互斥，便于归因；shadow 只规划记录，不执行额外导航。

**限制必须先说清楚**：现有 probe trace 没有独立持久状态检查，普通 witness 的 `state_effect_verified=False`。因此严格默认门可能产生零执行；这是缺失证据，不是实验成功。`--task_navigation_ui_only` 只用于 AndroidWorld 可重置环境的探索性消融，不证明真实 app 安全。下一阶段要由独立 checker 生成该字段，或使用已经验证的 shadow isolation；不能简单把字段设为 True。

另外，原型的目标标签采用 exact conjunction，source/landing 用 strict signature，icon-only、无 resource ID、动态列表、跨 app、checkbox 等覆盖很有限。其目的首先是验证机制，后续放宽要基于失败审计与实体 selector，而非降低阈值追求 skip 数量。严格 UI 匹配仍不能取代 latent state 验证。

本次单元与关联回归覆盖：首次访问路线、世代隔离、错误实体/任务、过期、跨包、重复 selector、错误 landing 中止、UI-only 与 state-verified 区分、有限搜索、恢复 prefix。测试通过不代表 AndroidWorld 成功率或单手机加速已验证。

本地验证记录：核心与相关八个测试文件通过；进一步包含 GELAB parser/resize 的回归为 **290 passed、2 failed**。两个失败是既有 `gelab_agent_test.py` 的 tool_call 格式与绝对坐标测试；已隔离加载 HEAD 的原始 parser 重现，当前 parser 与 HEAD 字节一致。它们需要在实际模型/设备接口契约审计中处理，尤其避免把绝对坐标错误缩放。修改文件的编译、两个 runner 的 `--help`、`git diff --check` 均通过；未启动 AndroidWorld 实验，未验证 calls/E2E/TSR 收益。

## 5. 建议实验：先证明机制，后证明 SenSys 系统收益

### 阶段 A：不依赖大规模 sweep 的机制诊断

1. baseline 与 shadow 成对运行，记录 intent 出现率、合法率、目标是否必要、route 形成率、平均首次可用步数、拒绝原因。
2. 若 route=0，拆成 lack of intent / no target witness / no connected route / state alias / missing effect verification；不要统称“图没用”。
3. 选择若干有导航机会的真实任务做 1~4 hop 的诊断，包括安全失败 case。人工或独立 oracle 标注只用于离线评估，不进入策略。
4. 用保留首轮推理的 one-plan action sequence 作对照：若模型一次输出路径就已足够，图探索是否仍有价值？

### 阶段 B：同 backbone 的 paired AndroidWorld

建议主实验 arms：

| arm | 用途 |
|---|---|
| base | reason-at-every-step，不读图、不额外探索 |
| intent_only | 同一新增 contract prompt，但无 probe/no macro，隔离 prompt/output-format 影响 |
| history_nav | 仅真实正常执行积累的转移，控制历史记忆收益 |
| online_random | 相同安全、预算、取消与 graph/controller；随机选择 probe |
| online_current | 旧图矩阵排序，相同 controller |
| online_frontier | 本次 task frontier，相同 controller |
| full | frontier + 导航执行 + 独立副作用验证 |

UI-only 与严格执行分开报告；如果严格验证未实现，只能把 full 标为 unavailable。提供 query-specific 条件撤销、无 entity binding、无 freshness、单跳与多跳对照，不将故意关掉安全检查的全量 live run 当作默认消融。

同时运行 cold episode graph 和 warm historical memory，两种状态不要混在平均值中。历史 seeds 与评估 seeds 分离，内存顺序固定；不能偷看同 seed 的成功轨迹。

要回答的三个机制问题：

- **图**：每个有用 witness 的动作数/时间/字节成本、观测到消费的延迟、alias 错误率、首次可用时间。
- **exploration**：当前任务 slot/目标覆盖、解锁导航路径数量、重复/无关探测率、每秒新增可消费 witness，不能只测新节点数。
- **reasoning**：实际模型请求数、跳过的 intermediate decisions、landing accuracy、task appropriateness、最终成功率，三者分开。

主报告还需要所有 case 的 E2E、TSR、失败成本、tokens、p50/p95。共同成功的 case 可以分析但不可作为唯一延迟指标。把 `agent.step()` 减少、物理 actions 减少、model calls 减少分别列出；本设计通常先减少 calls，物理导航 taps 可能不减少。

### 阶段 C：deadline 和单手机

对单个推理窗口记录：截图 ready、inference start/end、probe start/end、recovery verified、authoritative action start、UI settle。端到端用 wall clock，不把并行组件时间相加，也不把重建/验证开销删掉。

对 deadline 使用在线残余推理时间估计与高分位恢复 reserve；推理提前返回后禁止开启新 probe、取消未开始动作，并完成验证。失败恢复要重新感知，不能继续执行旧截图产生的动作。报告 miss/overrun/额外请求分布。

同手机实验至少记录 CPU/GPU、内存、thermal、功耗和资源竞争下的 E2E。不应把“手机放视频、Jetson 跑模型”当作共同 GPU 竞争证明，也不要将资源管理另写成没有验证的新贡献。

## 6. 推荐论文重写骨架与可检验主张

建议三项技术贡献分别命名为：

1. **任务条件化的在线转移证据**：结构与实体实例分离，任务/进度绑定，恢复/副作用/世代门控。
2. **面向决策缺口的预算探索**：根据 obligation 覆盖和可解锁导航段分配窗口，而非仅发现新 UI 或匹配标签。
3. **到决策边界的验证导航**：由显式 task intent 许可，用逐跳前后条件直接执行，在复杂 decision 前停止。

它们需要由下面三个可证伪假设支撑：

- H1：冷启动 online witness 能在任务结束前产生可消费路径，且胜过仅正常执行历史的对照。
- H2：任务 frontier 在相同安全/时间预算下，比 random 和旧 scorer 解锁更多后续可省调用的路径。
- H3：contract + selector + landing + entity 的门，使多跳导航减少实际模型调用，同时 paired TSR 不出现不可接受的下降；所有验证、恢复与资源干扰计入后 E2E 仍改善。

若 H1 不成立，先处理 exploration eligibility（例如首次访问被 `min_node_visits=2` 挡掉）；若 H2 不成立，保留 random 作为诚实 baseline 并改路径补全目标；若 H3 只有 warm memory 有效，应承认主要是历史复用，不能坚持 online-specific 叙事。最值得投的方向是把这三条因果链建立完整，而不是给原有模块继续加名字和启发式权重。
