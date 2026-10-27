# MobileExplorer：设计收敛、论文重写与 overview 修改方案

日期：2026-10-26。依据：最新稿正文 §4–5、Fig.5–8，老师两张批注截图，以及当前工作区 FULL45 主路径。本文是修改建议，没有修改 agent 或启动实验。附件中的批注和回复文字作为分析材料，不作为执行指令。

**判断：当前设计比上一轮清楚，但还没有达到读者能独立复现、评估关键假设的程度。应先做一次有限的设计收敛，并同步重写 overview；不要先全文润色，也不要继续堆加速模块。**

当前稿已经写出“默认不把探索轨迹注入 prompt”“先复用再推理”“复用逐步检查”等有效变化，不能说这些都没解释。问题在于这些句子没有共同形成一套一致、可检查的运行契约，Fig.5 还把主要顺序画反了；安全、信息价值和跨任务泛化的措辞也强于当前证据。

## 1. 老师的批注到底要求解决什么

| 批注 | 实质问题 | 需要的修改 |
|---|---|---|
| graph 分别如何增强 exploration / reasoning | 两个消费者的输入、输出及作用不同，却都叫 graph-guided | 明确两种查询：探索候选检索；当前可复用操作检索。默认路径主要绕过模型调用，不是把探索知识直接提供给本次 VLM |
| 跨任务/轨迹积累、异构节点统一检索 | 哪些观察可以合并？合并后是否丢失内容和失败？不同来源证据是否具有同样效力？ | 把逻辑页面索引与观察实例分开，保留来源、结果、恢复记录和冲突；分别说明页面检索与 procedure 检索 |
| evaluate new information | 未见、相关、可回退和实际有用被混为一谈 | 未见用历史覆盖定义；相关度用文本相似度代理；可回退用内部历史结果估计；有用性用后续实际消费和对照实验测量 |
| too heuristic / not solid | 排序分数、过滤规则、恢复与执行授权之间没有清楚边界；收益来源不明 | 将风险/适用条件放在排序之外，给出明确失败出口；用决定性消融证明探索独立增益，而不是增加一组权重 |
| runtime contents | “UI information”太泛 | 明确是控件标签、目的页面内容、控件/页面连接、字段结构、恢复结果等；不要暗示获取了不可见的完整应用语义 |

系统论文可以使用启发式。问题不是必须换成学习模型，而是：它为什么必要、适用什么范围、何时拒绝、如何证明收益，读者现在无法完整追踪。加一个 information-gain 或 Bayesian 公式不会自动解决这些问题。

## 2. 建议保留的主线，以及降级的模块

把主线收敛为：**在模型推理期间获取额外的运行时 UI 转移；把这些观察与实际执行的转移放在可追溯的逻辑记忆中；在后续决策前检查能否复用，从而减少模型调用。**

必须回答的系统矛盾是：并行探索和正常执行共享同一个有状态 UI。即使计算可以并行，探索也会改变模型看到的环境；因此获取知识、恢复现场、使用旧观察之间需要明确的协调边界。

建议正文只突出三个机制：

1. **跨任务观察与转移记忆**：何时合并、保留什么、两个消费者如何查询。
2. **推理期间的探索与恢复协调**：何时允许 probe、如何停止、恢复到何种状态、失败怎样交回模型。
3. **经过当前 UI 检查的执行复用**：task/value 绑定、控件定位、逐步结果检查及提交边界。

Streaming action、adaptive observation、prompt/后端优化放到 runtime support，分别测量其贡献。它们是合理工程优化，但不宜都包装成一个新的 graph-guided 设计。已有图边观测能支持部分等待时间估计，也不意味着整个等待策略具有独立的图算法贡献。

暂不把 exact speculative decoding、KV cache、phase-aware scheduling 加入这篇论文的主设计。此前 server 验证 prompt 可以作为独立加速研究；它们没有解决“在线探索产生的知识是否实际有用”这一核心问题。

## 3. 第一个必须澄清的接口：共享的究竟是什么

### 3.1 页面索引不等于执行等价

建议用三层描述，而不是一个笼统的 content-invariant state：

| 层次 | 内容 | 可以支持什么 | 不足以推出什么 |
|---|---|---|---|
| 逻辑页面索引 `p` | package、activity、结构签名；具体实现说明列表重复项等如何归一化 | 召回结构相似页面，跨内容变化积累导航观察 | 相同业务实体、相同日期、相同 checked 状态或相同持久状态 |
| 观察实例 `o` | 该时刻的 UI 内容/属性、控件、截图引用、任务与时间、来源 | 当前内容匹配、消歧、追踪证据来源 | 这张截图以后仍有效 |
| 转移记录 `e` | 源/目标页面、action/selector、观察实例引用、执行/恢复结果、成本 | 构建导航候选、寻找 procedure 入口、估计恢复历史 | 该动作语义上安全、适合所有任务、目标必然唯一 |

同一结构页的联系人 Alice 与 Bob 可以共用检索索引；执行“删除此联系人”却不能因为结构相同而复用 Alice 的实体上下文。**抽象键用于召回；当前实例与操作前提用于接受或拒绝。**

当前 `joint_graph.py` 已按 `(package, activity, layout_sig)` 合并、保留目的计数和来源。`explore_value.py` 又建立 label/rid/邻接页面索引。Procedure library 与 `GraphReuse` 还有独立存储。因此论文可以说“统一逻辑记忆，包含多个索引”，暂不声称当前代码只有一个同构物理图。

不要为“异构”强行引入一个新的 heterogeneous GNN。首先说清异构来自不同来源/内容实例还是不同实体类型；统一的查询键、类型区分和证据接受条件就可能足够。来源不同的观察可以进入同一导航索引，但探索看到一条边，不等于生成了可执行的完整任务程序。

### 3.2 两种查询必须分开定义

以下接口是建议的论文规格，当前代码的对应逻辑分散在多个函数中，并未实现这套统一 API：

```text
retrieve_reuse(task, live_observation, memory)
  -> candidate action/path/procedure
     + source observations + parameter bindings + checks required
  -> no applicable candidate

retrieve_probes(task, live_observation, memory)
  -> unobserved control candidates
     + relevance evidence + recovery history + cost history
  -> no admissible candidate

record_transition(source, action, destination, origin, outcome, recovery, cost)
```

第一种查询找“现在能复用什么”，第二种找“现有记忆还缺什么、值得尝试什么”。它们使用同一份可追溯观察，但可以采用不同索引和准入条件。

复用应区分三类证据：探索发现的导航边；模型实际执行过的转移；由执行序列抽取的参数化 procedure。后两者也不等于 evaluator 证明了整任务正确：当前 function library 部分来源是模型自报完成，论文不能统称 ground-truth successful procedures。

## 4. 第二个必须收紧的接口：何种情况才允许继续执行

### 4.1 三个判定不能共用一个“state match”

- **检索匹配**：宽松、提高召回，允许内容变化。
- **恢复接受**：检查原观察中的内容/状态约束是否仍满足，不能只看布局。
- **动作仍有效**：除目标控件外，还要确认与动作相关的实体/日期/选中项等上下文。

这是需要改设计和实现的地方，而不只是补文字。当前可构造的函数级反例包括：非编辑标签 `Date: June 1` 与 `Date: June 10` 被前缀规则视为匹配；checkbox 的 checked 改变后，宽松 fallback 仍可接受；Save 按钮未变时，`decision_still_valid` 不检查其他字段的实体/日期上下文。这些是局部合成反例，不是声称已在真实 app 中观察到全部失败。

最低成本的修订应先保守收紧：保留 checkable/checked/selected/editable/enabled 等关键属性；实体、输入值和有语义的日期不能被一般前缀匹配抹掉；无法确认时从当前屏幕重新推理。忽略时钟、toast 等变化需要明确、可审计的范围。更细的 dependency-aware acceptance 是后续设计，不应在未实现时写进论文当成已有机制。

`joint_graph.walk` 当前逐跳验证主要检查 activity，不能写成已经完成了完整目标状态验证。`relaunch_restore` 返回页面结构匹配、另记内容是否一致，也不能直接算语义恢复成功。

### 4.2 “可回退”不等于“无副作用”

Back、重启或导航回原页面，不会自动撤销购物车、已读、设置等持久修改。控件角色、词表、容器和历史结果都可以帮助控制风险，但对首次未知交互的语义副作用没有普适保证。

因此需要收窄范围：把探索描述为有风险过滤和恢复检查的低风险导航探索；写明首次未知交互、图标/本地化与隐式写入的限制。若还要坚持“保证不改变持久状态”，就需要真实的状态隔离/回滚能力或可证明的 app 操作接口，现有导航恢复不能支撑这个主张。

独立评测可以在 server 上运行 AndroidWorld emulator，并使用应用数据库/设置等 oracle。纯截图/轨迹回放只能测试可见状态与接受规则，不能验证不可见持久状态。

### 4.3 并行窗口应使用一个明确的交接点

复用未命中时，模型基于冻结的观察 `o_i` 推理，explorer 操作 live UI。模型生成可解析动作后停止发起新的 probe；正在进行的 probe 要结束并恢复，然后检查该动作是否仍适用于当前 UI。失败就重新观察/推理，不能对被扰动的页面盲执行旧坐标。

应画出这个交接点，并承认恢复可能让动作等待。`action-ready` 事件发生时，模型可能还在生成 summary；不能写成全部模型推理已经结束。当前流式执行机制也要说明下一次构造 prompt 前仍可能等待 summary。

正文现有目标函数把 inference 与 interaction 简单相加，没有反映重叠。更有解释力的是关键路径分解：

```text
一次未命中复用的决策（简化）：
  lookup + max(action_ready_time, exploration_and_recovery_time)
         + action_check + execution + observation + unhidden_output_tail

探索直接增加的动作等待：
  max(0, exploration_and_recovery_time - action_ready_time)
```

推理时间要测并发时的实际值，纳入共享资源干扰。这个分解是计时模型，不是当前已经求解的调度优化问题。不宜再用一个没有对应优化器的“最大成功率、满足时延预算”公式冒充核心算法。

## 5. 第三个问题：怎样减少探索价值设计的 heuristic 感

### 5.1 先把四个不同概念分开

| 概念 | 当前可实现的定义 | 不应使用的替代说法 |
|---|---|---|
| novelty | 当前页面/控件转移是否已在当时可用的记忆中被观察 | 只要目的截图不同就是新知识 |
| relevance | 任务操作词与控件/邻接目的页面文本的相似度代理 | 后续一定有用、具有明确的信息增益 |
| recoverability | 某类操作被系统自身接受为恢复成功的历史统计 | 已校准的无副作用概率 |
| realized utility | 后续实际消费了什么证据、少了哪些调用、E2E 净变化 | 覆盖率、hint-follow 或目的页面看起来相关 |

当前 TF-IDF、L1/L2 固定混合权重、邻接传播折扣与跨控件角色统计都是启发式设计。可以保留一个简单、固定的 ranking baseline；但需要写明数据来源、冻结配置、验证集与测试集边界。图从空开始不意味着 UI IDF、task IDF 或角色先验也没有离线信息。

不建议现在换成另一个更复杂的加权公式。先把“允许不允许”与“允许集合里如何排序”分开，使用最简单的排序；只有实验证明排序限制了收益，再考虑学习/校准。风险条件也不是绝对真值，需要相应误接受评测。

### 5.2 比泛泛任务相关度更有系统含义的可选改进

可在复用器与 explorer 之间增加一个**可枚举的缺失前提接口**：复用器有候选 procedure，但不知道入口怎么到达、当前控件是否对应某个字段、某条已见路径是否仍有效；向 explorer 返回具体 evidence request。这样探索目标是补足可能使候选可执行的前提，而不只是寻找 TF-IDF 高的未知控件。

**这是新设计建议，当前没有完整实现；只有候选 procedure 已存在时才适用，也不能保证 probe 能补足前提。** Task-conditioned procedure retrieval、前提表示和 evidence satisfaction 都需定义，不能用一句“graph guides exploration”带过。若自然工作负载上这种机会很少，就不应为了论文强加这个模块。先记录机会数，再决定是否投入。

### 5.3 一个可以放进正文的概念例子

任务 A 的模型正在推理列表页操作，explorer 打开 `⋮` 菜单，记录“列表页 → 菜单页”的观察，并返回。任务 B 后来明确需要进入这个菜单时，可以在当前控件和入口条件匹配后复用已经观测的打开菜单动作，减少一次模型决策。

这不意味着系统已知道菜单里每个功能的后续转移，更不意味着它可以自动授权 Save/Delete。另一个从模型执行轨迹学到的填写 procedure，只有在当前任务参数可绑定、入口和字段可定位时才可以继续复用。这个例子是机制说明，不是现有日志中的已验证案例。

如果论文的核心例子无法具体到“哪条额外观察→后来哪次消费→原本哪次调用被绕过”，继续完善措辞不会解决贡献不清的问题。

## 6. 决定是否继续改设计的最小实验集

优先使用 server 上的 AndroidWorld emulator 和已有模型服务，不需要先复现手机上的阶段/DVFS/thermal 行为。

| 优先级 | 实验 | 唯一要回答的问题 | 报告口径 |
|---|---|---|---|
| P0 | 执行轨迹记忆+相同复用器 vs 执行与探索轨迹记忆+相同复用器；其余 runtime 完全一致 | 在线探索是否对已经具备历史复用的系统产生独立净收益 | 配对任务流、多顺序/seed、成功率、完整 E2E、全部模型调用、实际动作数 |
| P0 | 记录被消费证据的来源、首次观察/消费时间、入口是否当时已由 agent 观察 | 省下的调用到底来自探索，还是 launch/form/执行轨迹缓存 | 探索独有边/入口的实际使用次数，失败与回退；来源计数是解释性证据，不替代对照 |
| P0 | 对接受规则做受控状态扰动；另用 app-state oracle 核验真实 probe | 相同布局但实体、日期、checked 或隐式状态变化时是否错误接受 | false acceptance、false rejection、不可见状态检查范围；不能只报内部 rollback_ok |
| P1 | 图/任务排序 vs 同候选集合、同风险过滤、同预算的随机排序 | 复杂 ranking 是否必要 | 总体与共同成功集合分开；别用成功子集掩盖失败 |
| P1 | 记录 action-ready、probe end、recovery end、check end、execute/observe end、summary end | 探索是否在关键路径上增加等待、净收益从哪里来 | 各组件时间与 overrun 分布，p50/p95；单步超时不代表全任务一定更慢 |
| P1 | 参数/模板与 app 留出测试、空图/预热任务流、先验数据隔离 | 结果是否依赖 benchmark 特定解析与预先采集的信息 | 英文 goal parser 覆盖、首次任务、摊销曲线、先验成本；app 留出不等同于仅参数换值 |

P0 对照中的“两份记忆”必须从相同起点开始、都只使用过去数据；各自的执行结果自然可能分叉。固定状态回放可用于解释查询/检查行为，但不能替代闭环任务成功率。不要把当前 INJ 臂当成只替换 memory 的对照，它还改变了直接复用路径。

**分支决策：**若 P0 表明探索相对执行记忆有稳定净收益，冻结主设计，主要工作转向论文和补证据。若收益主要来自执行记忆、bootstrap 或 form replay，先改定位或改信息获取/消费机制，再宣传“探索”为主贡献。已有不完整消融、旧版本 NOEXP 或某个成功子集都不足以替代这个决定性比较。

## 7. Fig.5 与 Fig.6 的具体改法

### Fig.5：只讲运行顺序和两个记忆消费者

新草图文件：

- [可编辑 SVG](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_revision_20261026/overview.svg)
- [PNG 预览](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_revision_20261026/overview.png)
- [生成源文件](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_revision_20261026/draw_figures.py)

必须修改：

1. **复用检查移到 VLM 请求之前。** 命中才绕过模型调用；不能先 Reasoning 再画 Graph Match，并声称它节省这次推理。
2. **将两个并行分支画完整。** 一个基于冻结观察推理，一个操作 live UI 并恢复；不是两条无约束的执行流。
3. **显式画恢复/检查交接与重新推理。** 动作字段已就绪不等于可以马上向 explorer 正在操作的 UI 发动作。
4. **两个记忆读接口分别命名。** Reuse lookup 与 Probe selection；正常执行、探索两条写回分别保留。
5. **删除 0.95/0.42/0.17 等装饰性数值。** 当前排序不是校准的概率。若必须展示例子，标注 unobserved/relevance evidence/recovery record 即可。
6. **去掉完整 Task1/Task2 编号网图和大手机图标。** 这些没有解释合并与检索机制；异构证据放 Fig.6。
7. **Streaming 和 adaptive observation 作为 runtime support。** 主图用小注解说明，不再画成与知识获取/复用并列的独立贡献。
8. **统一 step i，移除含义不清的 Summary→graph→next reasoning 箭头。** Summary/history 和图转移记录是不同数据。

草图表达的是建议的概念结构：现有逻辑分散在多个函数与存储中。图上的 restore/check 表示有这一步，**不表示当前实现已经满足 §4 的严格语义条件**。新提出的 evidence-request 接口没有画进当前 overview。

### Fig.6：讲“合并什么，保留什么，执行还检查什么”

- [可编辑 SVG](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_revision_20261026/memory_contract.svg)
- [PNG 预览](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_revision_20261026/memory_contract.png)

用同一表单的两个内容实例展示共享页面索引；旁边保留 agent/explore 两种来源的转移记录、目的与恢复结果；最后显示当前实例的参数绑定和控件检查。不要再重复 Fig.5 底部的两条轨迹拼接图。

这个细节图是建议表达的逻辑契约，不是宣称当前所有属性已被完整保存或每条路径已严格检查。字段 schema、失效条件与实现差距应在正文列清楚。

## 8. 可直接作为 §4 重写起点的英文段落

> MobileExplorer acquires additional runtime UI observations while a model is producing its next action, and reuses previously observed transitions before requesting another model decision. The two activities share a cross-task memory that indexes observations by application and layout while retaining their contents, provenance, and interaction outcomes. Layout matching retrieves candidates; it does not establish that two UI instances are interchangeable for execution.
>
> At each decision boundary, MobileExplorer first retrieves navigation steps or parameterized procedures that may apply to the current task and UI. An applicable candidate is grounded in the live UI and executed with intermediate checks; otherwise, the system requests a model action from the captured observation. During this request, an explorer can probe additional UI transitions and attempt to restore the starting screen. Before releasing the model action, the runtime checks the resulting live UI and either executes the action or obtains a new observation for reasoning. Probing and recovery can extend this handoff beyond action availability, so their cost is included in end-to-end latency.
>
> The memory has two consumers: reuse lookup retrieves previously observed behavior that may avoid model calls, whereas probe selection identifies unobserved transitions and ranks admissible candidates using task relevance and recovery history. In the default execution path, exploration traces are retained for later lookup rather than injected into the ongoing model prompt. Streaming action execution and adaptive UI observation reduce additional waiting in the runtime.

措辞边界：这段描述控制结构；它没有承诺 navigation recovery 可以撤销持久副作用。正文仍须报告上述匹配/检查的具体实现与限制，且不能用这段文字代替缺失的实现。

### Fig.5 caption 起点

> MobileExplorer control flow and memory use. (a) Reuse lookup precedes model inference. When reuse is unavailable, model inference overlaps with UI probing and recovery; the runtime checks the live UI before executing the model action or re-observing. (b) Execution and exploration observations feed a shared logical memory, with separate queries for reuse and probe selection. Memory retrieval supplies candidates, not semantic guarantees.

### Fig.6 caption 起点

> Separate indexing from execution acceptance. UI instances with the same application and structural layout share a retrieval key while retaining instance-specific contents and provenance. Observed transitions preserve their destinations and outcomes. A retrieved action or procedure must still be bound to the current task and checked against the live UI before execution.

## 9. 实际工作顺序

1. **先写清一页契约并改 Fig.5/6。** 使用本文件的表示、两个查询和交接点；逐条标注已实现/需修复/未验证。它是暴露设计缺口的规格，不是全文语言润色。
2. **修复状态接受的明确反例，统一实际开关与论文范围。** 深度二是否开启、何种 commit 留给模型、恢复成功的定义都应只有一个版本。
3. **跑决定性 P0 对照和来源归因。** 这决定是否值得继续把在线探索作为核心贡献；不要先花大量时间扩大设备/模型矩阵。
4. **证据支持主线后冻结设计，重写全文。** 同步摘要、引言、方法、实验、回复，撤掉旧 hint/distillation 解释和未填数据的完成声明。

也就是：**先做小范围设计收敛与关键验证，现在就改 overview/方法结构，最后再做全文重写。** 当前最缺的不是又一个加速点，而是从额外观察到后续调用减少的清楚、可信、可重复的链路。
