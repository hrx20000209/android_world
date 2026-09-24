# MobileExplorer：下一阶段研究判断与实验计划

日期：2026-09-18。依据：meeting notes、当前 MobileExplorer PDF、#228 reviews、七篇用户提供的 MobiCom 投稿稿、AutoDroid 原文，以及本仓库设计文档和关键实现。本文是独立研究建议，不把既有设计文档中的断言当作已验证结论。没有重新运行性能实验；代码发现来自静态检查。同期论文数字仅描述各自稿件，不能跨论文直接比较，也不代表录用状态。

## 1. 最重要的判断

当前应暂停增加 graph 层数、启发式分数和 skip 条件，先验证一个更窄的问题：

> 在可复用导航、参数化脚本和轻量感知都已使用之后，当前任务是否仍存在足够多的“必须交互后才知道”的决策信息；能否提前取得这些信息，并在同一设备上缩短任务完成时间？

Online exploration 不具有普遍必要性。建议研究的是 **task-conditioned online probing**：只针对尚未解决、可能改变后续决策的问题进行短探测。稳定拓扑可以来自 offline exploration，历史经验可以来自部署后积累；它们应成为探测的先验和执行工具。

建议暂定主张：**在残留的昂贵推理窗口内，调度有价值的 GUI 探测；通过依赖关系验证证据能否被后续决策使用，在成功率约束下减少关键路径时间。** 这是待验证假设，不是现有实验已经证明的结论。

## 2. 相关工作：不能简单分成 offline 与 online

应区分四个正交维度：知识何时收集、收集是否主动、复用什么对象、运行时如何验证。After-deployment 是时间位置，不是机制；部署后既可以被动记忆，也可以主动探索。Online 也不等于与推理并发。

| 工作 | 机制与原文位置 | 对本项目的实际压力 |
|---|---|---|
| AutoDroid | §3.2，PDF p6–7：UTG 节点为 UI、边为动作；探索后合成 task/function memory，按任务与当前 UI 检索注入；§3.3 还有查询优化 | “GUI graph + exploration + memory injection”已有充分先例。图是 GUI-specific 并不构成新的区别 |
| Vigil | §3，p4–7：轨迹构建 EFSM、语义条件落到可执行规则；§5，p9：在线发现 gap 后另开收集轮次，经构建/验证后用于下一轮 | 不能说它不适应部署后变化。它优化符号验证与知识准入，不以同一任务推理期间的主动探测加速为目标 |
| OmniFlow | §3.1–3.3，p4–7：历史成功轨迹编译为细粒度、可参数化 function 和 transition graph；实时重新 grounding，失配回 VLM 后继续复用 | 动态值、布局变化、部分轨迹不匹配本身都不足以证明 online probing 必要。必须优于闭环、可参数化 replay |
| SwiftAgent | §4.1–4.4，p5–7：历史 workflow、一次性 reuse plan、非连续 reuse points、gap actions，本地 grounding 与云端恢复；成功后更新 memory | “有 gap”也不够：部分 gap 已由计划和小模型处理。需要证明仍存在依赖未观察信息的 gap |
| PastForward | §3.2–3.4，p4–7：历史输出作为多 token proposal，由当前模型验证；根据历史 next screen 在 action-to-observation 间隙提前推理；复用 KV | 最近的系统定位对照。它提前做计算，本项目拟提前做环境观察；两者都处理投机结果有效性。不能声称已有工作全都严格串行 |
| DroidPrefill | §3，p5–8：候选动作编译、NPU prefill 边界选动作、必要时短 refinement；§4.5，p11–12：同手机完整流程 | 直接压缩可利用的推理窗口。“40 秒推理足够藏探索”不是稳健动机。应测更快推理下的收益边界。其 §4.4 部分 SPAD 收益为 trajectory-preserving replay/projection，需与实机验证范围区分 |
| PRMAgent | §4，p5–8：任务前探测 Android entry points，运行时参数化 macro、landing verification、感知路由；§6，p12 说明边界 | 可以直接消除你希望加速的导航；必须给所有实验臂同样的 macro/轻量感知能力。它报告的 accumulated LLM-request latency 不是完整 E2E |
| Argus | §4，p5 起：跨系统输入与 app callback 传播 subject，ART/DEX 派生对象标识，执行前强制策略 | 值得借鉴的是发现真正的 OS 边界并在那里实施机制。它不是 app snapshot、数据库回滚或远端事务隔离系统 |

这些工作大多已经是 GUI-specific systems；PastForward 还深入模型运行时，Argus 深入 Android framework。把它们统一称为 general agent memory 会削弱论文的可信度。

PastForward 的 §7 已明确讨论 MobileExplorer 的 online UI exploration；需要把新稿贡献收紧到比既有“推理期间探索”更具体、可测的机制。

## 3. 为什么当前信息仍不足以证明 online exploration

历史没有今天的价格、最新的笔记内容，这说明需要当前观察；并不说明需要额外投机探索。闭环 replay 可以很快抵达当前页面，再由轻量 extractor 或 VLM 读取。同一个 extractor 必须提供给 baseline。

要成立，需要验证三个条件的交集：

1. **决策价值**：隐藏信息会改变后续路线、约束判断、动作选择或回答。只是发现新页面不算。
2. **提前价值**：该信息能在 baseline 需要它之前拿到，并且减少后续关键路径，而非把同一次模型调用搬到 shadow。
3. **净收益**：探测、额外感知、恢复/同步、证据验证、资源竞争与失败处理的总影响小于节省的时间。

可用一个近似规划目标组织测量，但不要把它当作精确分解：

`预期净时间收益 = P(及时且有效且被使用) × 可避免的剩余关键路径时间
                   − 并发引起的推理增时 − 暴露到关键路径上的探测/恢复尾部
                   − 增加的验证与 prompt 成本 − 失败后的预期补救成本`

所有项统一用秒。能源、峰值内存和不可接受副作用另外作为约束。不要把全部 probe wall time 都算成关键路径开销，也不能认为与 inference 重叠的部分免费；共享资源会延长 inference。

优先观察三类自然机会：

- 当前数据驱动的条件分支：例如库存、权限、记录内容决定下一步去哪；稳定脚本不能预先确定分支。
- 多个候选分支的消歧：当前页面可见信息不足，只有打开部分详情才能决定哪个满足任务条件。
- 当前会话中新出现/失效的可达性：入口是否仍能到预期页面，需要一次具体检查。

这些只是候选类别，不能手工筛到只剩“有利于本方法”的任务后报告全局收益。尤其第三类变化可能非常罕见；必须报告自然任务分布中的比例。

必须保留负例：完全重复的稳定流程、直接 entry point 已覆盖的导航、当前页即可轻量回答的问题、推理很快的任务、不可安全探测的任务。好系统应在这些场景关闭探索。

## 4. 对当前稿件和已有设计的具体修改

当前 PDF 比 review 描述的版本已经补充了预算、恢复等待计入延迟等说明（§5.3.2，p8），不能继续说这些内容完全缺失。但以下核心问题仍未被解决。

### 4.1 图的 transition validity 与 action usefulness 混在一起

§5.4.3（p9）仍允许通过重复访问、低 decision entropy、唯一有效出边等条件 bypass。它们最多表明过去经常如此导航，不能证明当前目标要求同一动作。页面相同但目标是“删除笔记”或“新增笔记”，可用动作并不相同。

把三个问题分开：

- 动作会不会到达预计页面？
- 观察到的事实此刻是否仍成立？
- 该事实是否足以支持当前任务的下一动作？

达到预计页面不是 task-correctness 的 ground truth，pHash/activity 匹配也不是完整状态恢复证明。

### 4.2 “两层/三层图 + TTL + agent_id”不足以成为贡献

新设计文档的 topology/evidence 分层值得保留；但这些结构也能用于通用 memory。新颖性应来自这个图支持的算法与运行时操作：**从未解决决策反向找探测、明确使用前提、发生变更时定向失效、并用成本选择执行时机。**

不要同时保留多套意义重叠的 belief、evidence、agent 层。先做一个小而清楚的依赖表示，验证它优于相同字段的简单表和 UTG。

### 4.3 当前 shadow oracle 有额外模型调用

`scripts/run_shadow_oracle_probe.py:143` 的 `_extract_answer()` 调用 `vllm.predict_mm`；之后用答案是否出现在可见文本中进行检查，并在约第244行写入 `confidence=1.0`。文本出现只证明可见，不证明它回答了正确的问题，也不代表概率校准。

因此已有“primary VLM calls 少一次”的 pilot 只能说明主路径调用减少，不能证明总推理减少。必须分别统计 primary、explorer、extractor、verifier 的调用、token、时间和能量。若仍使用大 VLM extractor，应如实将其研究为投机并行计算，并验证同设备资源可行性。

### 4.4 文档中的未来状态绑定与实现仍存在落差

`online_exploration_runtime.py` 写入的 `source_state_id` 来自探测起点 context，`EvidenceRecord.is_usable()` 要求它等于消费时的 state；主 agent 的 `LiveEvidence` 同样主要匹配起点 schema/visual id。另存了 shadow observation id，不代表已经有完整的 source-to-consumer 依赖规则。

`mobileexplorer.py` 发布 opportunity 时使用固定 `now + 5.0s`，这不等于测量模型剩余时间的自适应调度。`PhaseAwareProbeScheduler` 当前只允许 DECODE 且使用 CPU PSI/thermal 阈值；这是可以测试的基线，不是所有手机后端都成立的策略。

### 4.5 Shadow 是实验能力，不是免费部署前提

同 seed、相同 prefix 并不能保证远端会话、后台服务、时钟、缓存一致。独立 AVD 存储也不能隔离同一服务器上的写入。副本发出的真实网络请求不会因丢弃副本而撤销。

`evaluation_results/mobileexplorer_full_snapshot_20260916/PROTOCOL.md` 已明确记录 primary/shadow 等价性尚未完整通过、旧轨迹字段不足、部分 policy version 不齐。后续论文不能把这些跑测当成完成审计的全量有效比较。

双模拟器适合测信息价值上限和隔离可行性。若手机上无法提供等价、低成本 shadow，应收窄到已验证的 navigation-only probes，或明确改为边缘辅助部署；不能把真手机部署推迟为不影响主张的工程细节。

### 4.6 旧实验记录里的过强推断需要降级

`docs/mobileexplorer_design_current.md` 后部把当前预填实现的 5–7% 省步称为“物理上限”，依据不足。它只约束该 workload、候选集、表示和策略；不是图记忆方法的理论上限。

同一文档以“触发层和未触发层都少几个成功”推断成功率差全部来自漂移，也不足以建立因果关系。触发通常受处理后的状态路径影响，不同层任务难度不一致；仅跑在同一天不能抵消时间顺序、温度和随机性差异。两边均成功子集可作效率诊断，仍不能代替整体 success/latency 结果。

## 5. Graph 应怎样改：可复用导航与任务证据依赖

借用 AutoDroid 的 UTG 是合理起点；不建议重复发明页面 embedding。建议一个逻辑结构，两个生命周期：

**长期部分：** UI schema、动作模板、可达页面、参数化 selector、哪些字段可能在何处暴露、探测成本分布。可来自 offline、部署后成功/失败轨迹。

**当前任务部分：** 目标/进度、尚未解决的谓词、带 provenance 的观测事实、事实依赖的实体/上下文/版本、最晚消费时间。

关键关系只有五类：

1. `UI --action--> UI`：导航可达性。
2. `probe --reveals--> predicate/slot`：哪次交互能回答哪个问题。
3. `decision --requires--> predicate/slot`：后续决策缺什么。
4. `evidence --valid_if--> dependencies`：事实在哪些条件下有效。
5. `event/action --invalidates--> evidence`：任务执行导致哪些证据失效。

一条示意证据：

```text
problem: candidate_note_matches_requested_meeting
binding: account=A, notebook=B, note_id=N, requested_title=Q
probe: open candidate N -> read title/body -> return
observed: title=Q, attendee_count=70
provenance: episode, UI node/source, timestamp, extraction method
valid_if: same account/entity/query; note version unchanged if observable
invalidate_on: edit/delete note, account switch, conflicting refreshed content
consumer: answer the attendee-count question for note N
cost: measured probe/recovery/extraction distribution
```

这个例子不是现有普适能力。无法观察 note version 时，只能采用有限时效和使用前复核、或拒绝高风险复用，不能声称无条件一致性。

图不是拿来整图塞 prompt。调度器从 `decision requires predicate` 反向找能够 reveal 的可达 probe，选择少量候选。消费者只收到所需事实、实体绑定、来源和适用条件。

构建顺序：任务开始/已有推理输出产生 provisional needs；检索 UTG 候选；执行探测后补 `reveals` 和事实；主路径动作发生时检查依赖并失效；消费前验证仍相关。任务语义解析和 extractor 的开销必须计入；首版可用人工标注 needs 做 oracle，但正式系统必须自动产生，并做 held-out 验证。

这种表示的 GUI 特点是：页面状态有别名、字段在不同页面、观察本身需要动作、动作可能有副作用。Online 特点是：有未提交探测、推理世代、未来消费点、时效与失效。创新若成立，应该是这些条件共同约束的算法，而非新增字段名称。

必须做“扁平事实表 + 完全相同 metadata 和 selector”的对照。如果依赖边没有改善安全复用覆盖率、调度收益或更新成本，就把 graph 降为实现细节。

## 6. System 主线：两种依赖不能同时忽略

### 6.1 UI 状态依赖

当前模型对截图 S 进行推理，explorer 改动 UI 到 S'；此时模型返回针对 S 的动作。这是 GUI 探索与普通后台检索的关键不同。

单实例需要明确状态机：`BOUND -> PROBING -> DRAIN/RECOVER -> VERIFIED -> RELEASE`。探测返回不代表恢复完成；模型早结束时停止新 probe，等可靠恢复或丢弃旧动作重新感知，额外时间必须进入 E2E。若用户输入/异步事件改变 context，应使旧 generation 的结果失效。

只有导航型、已验证可恢复的动作进入这一模式。Back、截图相似、历史回滚成功都不是没有持久副作用的充分条件；“读取邮件”也可能改已读状态。

隔离实例可以允许主路径不等恢复，但应把 shadow teardown/rebuild 从证据 publication 解耦，明确生命周期；后台 cleanup 的资源影响仍需计量。当前同步 controller 的 cancel 标记是合作式取消，不等于任意 app action 可抢占、可撤销。

### 6.2 计算资源依赖

首先画各后端下的 interference matrix：单独 inference、单独 probe、并发，分别测 UI dump、截图、切页、OCR/小模型提取、恢复在 prefill/decode 时的增量影响。

不要预设 decode 有富余资源。Decode 常受内存带宽限制；NPU inference 也共享 DRAM、CPU 驱动、功耗和热预算。另一方面，prefill 禁探测也不应被永久写死：需要实测证明在哪些后端、哪些 probe 类型上成立。

调度决策限定为：`不探索 / 执行一个有界 probe / 完成必要恢复`，避免一开始做复杂 RL。

- 用模型、输入长度、已生成 token、当前负载预测剩余时间分布；不要把均值当 deadline 保证。
- 用并发实测成本估计 probe+restore 的分布；单实例以联合概率约束限制溢出风险。对两个分位数简单相减只是一种保守启发式，不自动得到概率保证。
- 以“证据最晚何时被需要”作为真正 deadline；单实例另受“旧动作执行前必须恢复”的更早约束。
- 运行时每个可停止边界复核；无法中断的动作不承诺即时取消。
- 限制并发 worker、CPU 核/优先级/内存预算；CPU affinity 不能解决共享 DRAM 竞争，应一并报告带宽与热影响。
- 如果 replay 或快模型消除了当前 inference，允许没有探测窗口，不人为插入等待。

Argus 的启发应落实为：primary 与 explorer 的事件归属、状态版本、输入执行权限、过期输出的提交屏障。若需要 OS 级强制，应精确说明 hook 点与支持范围；Python token/lock 不等于 Argus 的 mandatory enforcement，callback identity 也不等于动态业务实体 identity。

不建议同时承诺通用 app snapshot、完备副作用隔离、新图学习算法与 NPU runtime 四个大贡献。先验证收益，再选择能真正落地的执行模式。

## 7. 实验应拆开证明，而不是只做一次全系统对比

### 第一组：信息是否有用

在相同 decision checkpoint 比较无额外证据与正确当前证据。标注人员只能选择可实际到达、可观察的探测结果，不能把 evaluator answer 当运行时输入。可忽略成本得到信息价值上限，但必须标记 oracle。

如果信息仍不改变后续调用/动作/成功率，后续调度没有研究空间。

### 第二组：提前获取是否有用

所有臂使用相同 memory、宏动作、感知、extractor、模型、量化和动作接口：

| 臂 | 作用 |
|---|---|
| A. 强闭环 replay + 按需当前观察/同款 extractor | 主 baseline；不能限制它读取新值或参数化输入 |
| B. A + 相同 probe 在需要时串行执行 | 区分信息价值与并发价值 |
| C. A + 推理期间无条件并行 probe | 测 naive parallel 的竞争与无效探索 |
| D. A + 成本/依赖感知的选择与调度 | 完整方案 |
| E. A + oracle probe 选择 | 选择器上限，和可部署结果分开 |

无 memory、raw trajectory prompting 用作诊断；不能作为唯一主要 baseline。能够获得原实现时加入 OmniFlow/SwiftAgent；自制相似方案标为 style baseline，写明未复现的能力。AutoDroid-v2/script baseline 应按 review 加入。PastForward-style pipeline 与更快推理 backend 用来测组合后的剩余空间，不直接搬论文数字比较。

### 第三组：offline/after-deployment 在何处不足

构造三条独立轴：历史覆盖率、任务相关状态变化率、剩余 inference 时间。

- 覆盖：冷启动、部分历史、充分历史；报告训练/建库成本及按任务数摊销。
- 变化：无变化、仅参数变化、决策相关内容变化、结构变化。仅换随机 seed 不代表未见任务，也不代表强动态性。
- 推理速度：实际可用的快/慢模型与 CPU/GPU/NPU backend，不能靠 sleep 制造主结果。

加入 equal-budget offline exploration、部署后被动更新、两任务之间主动维护图的 baseline。成本对齐既报告相同探索/能量预算，也报告用户等待时间。不要人为禁止 after-deployment 基线用最近完成的历史。

分割区分：同模板新参数（合法的重复使用评测）、未见模板、未见 workflow/app。任务模板重叠不是一律“泄漏”；若宣称 unseen-task 泛化却用同模板历史，才是协议与主张不一致。任何臂都不能使用当前测试结果提前建库。

### 第四组：graph 和 scheduler 是否各自必要

Graph：原始 UI 记录 -> 普通 UTG + 相同 replay -> 扁平事实表 + 相同 metadata -> 加 requires/reveals/invalidates 依赖。控制候选集、探测次数、token budget，防止同时改变多个因素。

Selector：相同安全候选与时间预算下比较 random、novelty、task relevance、预期关键路径收益。关闭安全门不是在真实用户状态上可接受的消融；错误场景只在受控环境测。

Scheduler：串行、始终并行、固定时间预算、仅阶段感知、阶段+负载+价值联合调度。不能只证明比“完全无控制并行”好。

Consumption：raw observations、受限结构化 evidence、evidence+conservative bypass。不要把 inject 与 skip 一起开关后归因给其中一个。

### 所有组共同的测量规则

- 固定完整 task×seed 初态、memory 初态、policy/version；使用成对多 seed、随机/交错臂顺序，控制热状态。memory 持续增长实验单独设置可复现 stream。
- 完整 E2E 从任务提交到终止；运行中 shadow 创建/同步不能移出计时。benchmark 每臂共同的测试 reset 可另记，和算法自身运行成本分开。
- 记录成功率、达到成功的时间分布/成功率-时间曲线、P50/P95、全部失败与 timeout。不能把提前失败当速度快，也不能只报告各自成功子集均值。
- 同时报主路径动作、probe 动作、模型调用总量及各角色分量、prompt 增量、总能量/任务、内存、温度、推理 slowdown、恢复尾部、无用/过期证据、错误 bypass。
- 恢复真值由独立 evaluator、DB/文件/服务端测试记录建立，不用系统自己的 pHash 判断给自己打分。Privileged oracle 可以用于评估，不应偷偷成为 runtime 能力。
- Graph alignment、hint-follow、文本可见率均是机制指标，不是 causal correctness 证据。
- 子组机会标签尽量从固定 baseline checkpoint 预先生成，而非按 treatment 是否最终触发划分。统计以 task/template 为相关单元做分层/cluster bootstrap；少量 app 的 CI 不能证明普适泛化。

## 8. 两周执行顺序与停止条件

以下数量是工作量建议，不是统计充分性的保证；根据初始方差调整，冻结最终协议再跑确认集。

**第1–2天：证据和计时审计。** 选定一套干净 runner，固定 baseline 能力与 memory；把所有 primary/shadow/extractor 调用统一记账；删除未经验证的 confidence 解释；为现有数据标注可用/诊断/无效，不再拼不同配置汇总。产出一页实验协议、一张真实事件时间线。

**第3–5天：测主张上限。** 从强 baseline 的自然任务分布抽取约30–50个决策点，覆盖至少3个 app family，包含有/无机会的点；在相同 checkpoint 做信息 oracle 与按需轻量读取对照。产出 opportunity funnel：残留昂贵决策 -> 隐藏信息相关 -> 可安全探测 -> 能及时完成 -> 实际改变关键路径。每层都给分母。

**第6–7天：同设备微基准。** 在目标手机上测真实模型与 UI probe 的 interference matrix、恢复尾部和能量；至少涵盖快/慢推理工作点。若目标执行模式是 shadow，同周就验证手机上的状态复制与隔离能力，不能只等 emulator 信息 oracle 跑完后才考虑。

**第8–10天：一个最小机制。** 只实现 decision/slot/probe 的依赖、实体绑定、失效、一个 bounded probe 调度器。先不做复杂 agent-specific 学习或大量 multi-hop skip。展示一个完整例子：为什么这个 probe 被选、产生了什么、何时被消费、什么操作让它失效。

**第11–14天：小规模 E2E 决策。** 冻结规则，在多个 app family 与多个 paired seeds 上比较 A–D；同时保留全量自然任务的 no-opportunity 占比。得到完成时间、成功率、资源成本和失败归因，决定是否扩展论文实验。

建议预先和导师确定内部 go/no-go 门槛，例如：在确认集上相对强 baseline 有约10%以上的 E2E 改善且区间支持正收益，成功率满足预设非劣界限（例如绝对2个百分点，需要足够样本），同设备资源可接受，并能解释总体机会覆盖。数字只是项目资源分配门槛，不是 SenSys 的录用标准。

若只有2个 app family 的少数 oracle 点各省一次主调用，继续完善普适性证据，不宣布主线成立。若提升来自额外大模型并行，则明确改变研究主张并计入算力。若只有动态值+手写 extractor 得益，而同款按需 extractor 一样快，应停止 online 主线。若 graph 依赖没有增益，应去掉 graph novelty；若手机隔离不可行，应收窄部署/动作范围或转向参数化 replay 的 runtime verification。

## 9. 论文改写建议

Introduction 从“本地模型很慢，等着可惜”改为三个有数据支撑的事实：强执行复用之后仍有什么 residual decisions；其中多少等待隐藏状态信息；在真实共享硬件与 UI 状态约束下提前观察的收益区间是什么。

Design 以问题组织：哪些信息值得提前读、如何保持证据对后续决策有效、何时读才不拖慢主任务。Graph 服务第二个问题并辅助第一个，scheduler 服务第三个。

贡献建议最多三个：

1. 对 residual decision 与 active observation opportunity 的实测刻画及适用边界。
2. 基于 GUI 操作依赖的证据获取/失效/消费机制。
3. 同设备资源与 UI 状态约束下的调度实现和 E2E 验证。

Reviewer A/D 的 deadline、prompt overhead、状态恢复问题由真实时间线与尾延迟回答；Reviewer B/E 的 practical latency 和 script baseline 由强 baseline 与快推理工作点回答；Reviewer C 的 on-device contention 和持久副作用由实际同设备系统与独立状态审计回答。不能仅增加叙述。

## 10. 来源索引

页码为 PDF 页序，章节以上述原稿为准。

- [当前 MobileExplorer 稿件](</Users/huangrunxi/Downloads/Runxi_Agent_SenSys27 (1).pdf>)：尤其 §5.2–5.4、§6.1。
- [SenSys #228 reviews](/Users/huangrunxi/Desktop/MobiCom/sensys27-reviews-228.txt)：A–E。
- [Vigil](</Users/huangrunxi/Desktop/MobiCom2027 Agent/mobicom27-paper11-Vigil.pdf>)：§3–5。
- [OmniFlow](</Users/huangrunxi/Desktop/MobiCom2027 Agent/mobicom27-paper170-OmniFlow.pdf>)：§3、§6 discussion。
- [PastForward](</Users/huangrunxi/Desktop/MobiCom2027 Agent/mobicom27-paper261-PastForward.pdf>)：§3、§6–7。
- [SwiftAgent](</Users/huangrunxi/Desktop/MobiCom2027 Agent/mobicom27-paper337-SwiftAgent.pdf>)：§4–5。
- [DroidPrefill](</Users/huangrunxi/Desktop/MobiCom2027 Agent/mobicom27-paper582-npu.pdf>)：§3、§4.4–4.5。
- [PRMAgent](</Users/huangrunxi/Desktop/MobiCom2027 Agent/mobicom27-paper1250-PRMAgent.pdf>)：§4、§6。
- [Argus](</Users/huangrunxi/Desktop/MobiCom2027 Agent/mobicom27-paper1676-Argus.pdf>)：§1、§4。
- [AutoDroid 原文](https://arxiv.org/abs/2308.15272)：读取公开 v4，§3.2–3.3。
- 仓库对照：[最新设计](/Users/huangrunxi/Projects/android_world/docs/MOBILEEXPLORER_NEW_DESIGN_ZH.md)、[online 设计](/Users/huangrunxi/Projects/android_world/docs/ONLINE_EXPLORATION_DESIGN_ZH.md)、[历史设计/实验笔记](/Users/huangrunxi/Projects/android_world/docs/mobileexplorer_design_current.md)、[snapshot protocol](/Users/huangrunxi/Projects/android_world/evaluation_results/mobileexplorer_full_snapshot_20260916/PROTOCOL.md)。
