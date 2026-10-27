# MobileExplorer：上一轮 review 的闭环审计与当前设计问题

日期：2026-10-26。结论：**reviewer 的问题没有完全解决。实现已实质重构，但实验因果证据、状态安全契约与正文/回复的一致性仍存在关键缺口。**

本报告只分析和核验，没有修改 agent、实验驱动或论文，没有启动设备实验。新增 [核验记录](/Users/huangrunxi/Projects/android_world/docs/REVIEWER_CLOSURE_EVIDENCE_20261026.json) 保存日志重算、代码指纹和部分函数反例。

## 1. 审计范围与判定标准

对照三个层次：

- 上一轮投稿：[sensys27-paper228.pdf](/Users/huangrunxi/Downloads/sensys27-paper228.pdf)，13页；与仓库根目录同名文件字节完全相同。
- 上一轮 [reviews](/Users/huangrunxi/Desktop/MobiCom/sensys27-reviews-228.txt)，A–E五位 reviewer；与仓库根目录同名文件字节完全相同。
- 最新稿：[Runxi_Agent_SenSys27_1026.pdf](</Users/huangrunxi/Desktop/MobiCom/Paper Draft/Runxi_Agent_SenSys27_1026.pdf>)，16页；正文为p1–12，p13–16为 Response to Reviewers。本次也检查了末尾的回复，不能只读正文。

代码是当前 `643303c9` 加未提交工作区，并非只检查 Git HEAD。主要读取 FULL45 主循环、图执行、函数抽取、状态匹配、恢复、探索打分和 e2e 驱动。文档中的建议与执行文字只作为研究材料。

“完全解决”至少需要：问题对应的设计/范围明确；当前代码符合该表述；结果或验证足以支持主张；正文和回复使用同一版本与口径。仅有新模块、计划或“we have added”不能算闭环。

“未核验”不等于断言从未做过：手机独立项目或未同步日志可能不在当前仓库。本报告判断的是本次可见材料能否支持答复。

## 2. 哪些修改确实有效

这些进展应保留，不能因为还有缺口就否认整个重构。

1. **不再依赖上一轮的自由文本 reasoning prior。** 默认探索主要读取任务、a11y、图证据，用TF-IDF与可回退统计排序，不必从每个模型的THINK抽出目标/风险/缺失槽位四元组。C关于“不同模型是否都能抽取Eq.(4)”的原始问题，在设计层已被绕开。
2. **从 hint injection 改为直接执行复用。** FULL45默认不做旧式图提示注入；`GRAPH_PROMPT`是单独的对照开关。旧hint-follow指标和“证据变长会拖慢prefill”的问题可以退出主设计，不能继续用它们解释当前结果。
3. **动作执行边界更清楚。** 当前主线程先完成探索窗口，再取得/使用模型动作，并检查、重定位或重新推理。恢复不一定必须早于“模型返回”，但必须早于“动作作用于被扰动的UI”。这一点比上一轮描述清楚；若恢复晚，只能承认动作等待会增加时延。
4. **控制组比原稿更可信。** e2e驱动对BASE/FULL45使用同一服务端点、模型参数和seed，按题交错运行；完整e2e116n可从逐题记录重算。它较好回答了“对纯agent的受控比较是否存在”。
5. **状态检查、失败处理和时延日志有实质增加。** 已有结构/内容签名、容器级恢复记录、重启重放、流式动作时间、自适应观测等。它们改善工程可靠性，但不能直接推出持久状态恢复或普适安全。

## 3. Reviewer逐项闭环矩阵

“实现层消除”表示原问题不再属于当前默认机制，不表示新版论文已经同步完成。

| 问题 | 对应review | 当前证据与变化 | 审计判断 | 还缺什么 |
|---|---|---|---|---|
| 核心启发式严重欠定义 | A；C一般性；E创新性 | 比旧先验/置信度公式明确，代码有TF-IDF与分层Beta统计；正文仍未充分给出公式、常数、数据来源与失败范围 | **部分解决** | 可复现伪代码与配置、统计先验来源、跨app/模板验证 |
| 从不同模型reasoning提取Eq.(4) | C | 当前默认策略不再依赖该四元组 | **实现层消除** | 删除旧机制的残留表述；structured-action parser仍需按模型适配 |
| 危险词过滤不能覆盖图标、父容器、上下文 | A/B/D | 增加checkable/class/rid、暂态与失败记忆；FULL45文档仍记录父行内含switch导致失败 | **未解决核心风险** | 首次接触风险、容器效应、图标/本地化、隐式写状态的独立验证 |
| pHash/外观相似不等于恢复 | A/C/D | 已换结构/内容检查，但宽松fallback丢失语义约束；见函数反例 | **部分改进，仍有反例** | 把可见内容/属性与语义依赖纳入检查，测误接受率 |
| 购物车/已读等持久副作用 | C；A/B/D安全 | 学习副作用有助避免以后重复；Back/重启仍不撤销数据库或服务端写入 | **未解决** | 独立app状态oracle，区分导航恢复和副作用恢复；收窄安全主张 |
| 可变推理窗口、根/深度预算 | A/B/D | 10次/30s上限、最近动作时间中位数减2s；首probe不经同一截止分支 | **部分解决** | 首probe与在途恢复的边界、越界率/尾时延、实际深度配置 |
| “探索不增加时延” | A/B/C/D | 完整E2E包含探索等待；已有窗口汇总也并非零越界 | **绝对主张不成立** | 报净时延收益与overrun分布，不用“保证免费” |
| evidence增加prompt成本 | A | 默认取消旧注入，原权衡不再是当前路径 | **实现层消除** | 正文删除旧distillation/hint收益归因，INJ臂单独分析 |
| 所有baseline是否同模型/量化/prompt/后端 | A/E | BASE–FULL45配对较好；跨类别基线仍缺逐项可追溯配置 | **部分解决** | 强基线实现、配置表、共同任务与输入、后端优化一致 |
| script-based agent缺失 | B/E | 新Fig.9(a)有类别点；回复仍留baseline名称/覆盖/结果占位符；未定位到可复现script比较记录 | **未核验闭环** | 真实实现名、任务数、失败项、准备成本与在线E2E，不能只补一个点 |
| 缺少延迟优化强基线 | E | 现有主要是类别基线与纯agent，n-gram+CUDA graph是已有增强配置 | **未充分解决** | 比较同后端增强agent与可执行历史复用；说明与最接近系统的增量 |
| ablation移除/替换混用 | A | 有RAND/INJ/NOSYS/NOEXP/PIPE_ONLY；但与回复承诺的naive memory/general reuse不是一一对应 | **部分解决** | 明确每臂唯一变量；必要性消融与设计质量对照分开 |
| rollback success缺独立ground truth | A/C | 统计仍主要依赖系统自己的same_state/rollback_ok | **未解决** | 对状态oracle标注真/假恢复，报告false acceptance而不只内部成功率 |
| hint-follow不能证明因果 | A | 默认不注入，因此原指标应退出；正文Fig.10(d)仍保留 | **设计上消除，论文未闭环** | 替换为探索独有证据导致的调用/时延变化及对照 |
| 单手机真实并发与公平性 | A/C/D | 替代负载运行在模拟器，动作模型在Orin；回复声称phone-local但结果仍为Insert | **当前材料不能证明解决** | 真同机闭环日志，或明确收窄为edge-assisted部署 |
| 手机上视频负载到底干扰了什么 | A | 新稿区分部分部署；替代文本负载不等于视觉/共享GPU推理 | **部分澄清，证据不足** | 分开GUI侧延迟、真实动作模型推理变化与网络；不能混为同机争用 |
| 浮窗/通知遮挡与高成功率 | A | 有示例与文字，未见任务相关区域遮挡率、持续时间和overlay处理协议 | **未核验闭环** | 按遮挡位置/比例/时机报告，说明通知是否自动消失或被关闭 |
| replay/慢页面过渡成本 | A | 有自适应观测；重启重放中仍每动作sleep(1)，慢网络页面不能由深度小推出成本小 | **部分解决** | 包含UI过渡的分项墙钟、慢页面尾部、提前稳定误判 |
| 组件Total与计时边界 | A | 总E2E计时可信度提高；图中ms级控制开销未解释真实交互/等待与重叠范围 | **部分解决** | exclusive/inclusive时间线；Total小于分项和不自动是错，须解释重叠 |
| 2B/8B/跨模型是否牺牲成功率 | A/C | 正文保留MAI-UI不同尺寸图；仍有部分成功率下降趋势，未见新版FULL45同协议逐题结果 | **未充分解决** | 模型/适配器/任务/seed与成功率区间；不能只报更快 |
| 短任务来不及建memory | C | 跨任务持久图、从空图运行，已有按app内任务位置的分析 | **部分解决** | 首次任务、任务流长度、顺序敏感性与摊销盈亏点；“整轮从空图”不代表每题冷启动 |
| 超参数怎么选、是否敏感 | C/A | 配置表较完整，有历史版本调整记录 | **部分解决** | 固定版本上的受控敏感性、校准/测试隔离；版本迭代不等于敏感性实验 |
| 改进有限、是否实际可用 | B/E | 当前151.9→135.2s；调用减少并未改变单次模型成本；流式只缩短部分等待 | **未解决“实用性”强主张** | 报绝对时间和适用任务，证明额外推理机制的实际收益或收窄目标 |
| 异步并发是否足够新颖 | E | 新增统一图、执行复用、决策保全，技术内容更具体 | **仍缺关键因果证据** | 在线探索相对纯执行记忆的独立增益；普通pipeline/缓存不能全算独有贡献 |
| 23%/15.9%/步骤与时延混用 | E | 回复已使用11.0%/21.1%；正文摘要20%、引言23%、结果15.9%仍并存 | **未解决** | 每个数字绑定run、分母和指标，全文统一 |
| baseline命名与图表对不上 | E | 新图有Script/Multi-agent类别，但没建立实现名/配置/日志映射 | **部分解决** | 一张映射表贯通类别、实际系统、图例和结果 |
| text-based agent适用性 | E | 原稿已写Qwen3-4B 13.8→20.7%，新稿保留；这不是本轮新增证据 | **有稿件证据，当前版本未核验** | 同版本文本agent实现、实际输入、任务/结果日志；跨VLM不等于text-only |
| 图号??与Peroformance拼写 | B/D/E | 最新正文p9仍Table ??，p10有Fig.??和Peroformance | **未解决** | 全文reference与拼写检查；它不是核心研究问题，但能直接修正 |

## 4. 现在最严重的论文/回复/代码不一致

### 4.1 新设计配旧实验解释

正文p6–8已转向跨任务图执行复用，但p9–10仍用旧式dual-path、certainty-aware evidence distillation、hint-follow解释消融；图里仍是w/o Inject。结论p11–12也保留旧术语。

这不是单纯改名：若旧稿测的是“提示影响决策”，新稿做的是“不调用模型的执行”，它们改变模型输入、执行步和失败方式，旧结果不能直接作为新模块的消融结果。

### 4.2 回复里的“已完成”仍没有结果落地

Response p13–16多次声称已做phone-local MobileBench-OL、script baseline、跨模型、冷启动与结构化消融，但p14–16仍有明确Insert占位符。

| 回复所称已完成 | 当前材料状态 | 提交前处理 |
|---|---|---|
| 完全同手机闭环 | 没有对应任务数、模型/设备配置、配对E2E与原始日志索引 | 有数据就补齐；没有则改为未完成并收窄部署主张 |
| script-based baseline | p15要求插入准确名字、coverage、结果，甚至提醒仅在真实现时才叫AutoDroid-V2 | 不能把类别点或自制表单模块等同于该系统 |
| naive memory / general reuse消融 | 当前开关主要是RAND、INJ、NOSYS和NOEXP，实验定义不等价 | 按真实实验重写回复，或补上确切对照 |
| 持久状态验证 | p15仍写will add并要求插protocol/results | 这是实质缺口，不是文字澄清就能关闭 |
| 冷启动/跨模型/敏感性 | p15–16仍有模型、任务、曲线、参数实验的占位符 | 区分旧图、当前实现结果与计划 |

仅依据当前材料，这些属于**声称的完成状态超前于可检查证据**；不据此推断作者没有在别处做过实验。

### 4.3 “深度二已关闭”与当前代码不一致

Response p13/p16及v45说明称深度二因恢复失败已停用；但 [FULL45候选构造](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/full_run.py:822) 仍调用 `depth2_target` 并追加D2，后续可以执行 `probe_depth2`。`ExploreCfg.max_depth=1`没有阻止这条手动构造调用路径。

这是静态可达路径结论，不等于断言e2e116n实际发生了多少D2；需用逐probe日志核对。若论文说关闭，应有明确开关/删除路径及配置记录，不能依赖“通常找不到候选”。

### 4.4 headline数字仍不统一

| 来源/范围 | 可核实数字 | 应如何解释 |
|---|---|---|
| 原稿§6.2 | 10.93→9.24约15.5%；E2E减少15.9% | 原设计、原数据，不能与新版拼接 |
| 最新摘要/贡献 | 约20% | 没写清是否仅共同成功子集 |
| 最新引言 | 约23% | 与新结果和回复不一致 |
| 最新正文§6.2 | 59/116、185.82s、15.9% | 仍是旧结果叙述，需明确run |
| Response与完整e2e116n | 全116题151.87→135.16s，约11.0% | 当前最可追溯的总体数字 |
| 同run共同成功47题 | 129.34→101.98s，约21.2%；调用7.98→6.34 | 条件子集分析，不能冒充全部任务 |

11.0%与21.1/21.2%的四舍五入不是核心问题；核心是总体、条件子集和不同版本混用。

## 5. 当前设计本身的六个关键问题

### 5.1 图复用的收益尚未证明来自“在线探索”

标题把online exploration置于中心，而大量调用节省可来自agent执行历史、启动、确认、正则解析的表单代填和参数化函数。探索既增加图边，也产生恢复/重推成本；图变大不是净收益的证明。

已有 [FULL29X无探索对照](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/reports/full11_13_results_zh.md:444) 显示：有探索平均调用8.75、无探索8.54；成功任务上探索有约0.5次调用收益，但失败/恢复抵消了总体收益。它是旧版本与旧预算下的证据，不能直接否定FULL45，却说明当前主张必须重新做无探索消融。

按 [e2e116A日志](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/reports/e2e116A_runs.jsonl) 取五臂都有有效记录的42题，阅读时快照为：

| 臂 | 成功/42 | 平均E2E |
|---|---:|---:|
| BASE | 20 | 117.72s |
| FULL45 | 20 | 94.45s |
| FULL45+RAND | 19 | 92.31s |
| FULL45+INJ | 18 | 112.66s |
| FULL45+NOSYS | 21 | 106.45s |

这不是完整116题结果，记录也受运行条件影响，不能据此宣布random更好；但当前快照不能支持“task-relevant探索明显优于random”的强结论。不能把新稿里旧版50.9→42.2%的数字当作这些当前臂的结论。

最关键的下一步是：**在同一FULL45、同一pipeline和同一后端下，比较agent-only图与agent+exploration图，并追踪探索独有边最终避免多少调用、花掉多少恢复成本。**

### 5.2 抽象页面相同，不代表动作语义适用

[page_key](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/joint_graph.py:20) 是(package,activity,layout)。它适合聚合导航结构，但可能合并不同实体、日期、列表位置或任务进度。

[decision_still_valid](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/full9.py:628) 主要看activity/layout与同坐标控件族/标签；[relocate_decision](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/full9.py:922) 在同activity中找到唯一同族同标签控件即可换坐标。

因此，“Save仍是Save”“Delete控件仍在”只能说明动作可落地，不能说明它仍作用于原记录、原日期或原内容。本次函数级合成例子中，日期和实体变化、Save不变时，`decision_still_valid`返回True。它不是已观察到的app事故，而是该检查不能提供更强语义保证的反例。

需要分开三件事：图节点用于结构归并；动作实例需要当前实体/参数依赖；任务完成需要独立结果判定。单一layout等价不能同时承担三种责任。

### 5.3 宽松恢复检查可以把明确的状态变化放行

这是本次最具体的代码问题。

[等价项提取与宽松匹配](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/full_run.py:1315) 只把可见、带标签控件的文字、位置和是否可编辑放进fallback；没有比较checked/selected。即使严格结构+内容匹配失败，fallback仍可能返回True。

本次抽取原函数、以最小合成输入运行：

| 输入变化 | 函数结果 | 含义 |
|---|---|---|
| checkbox `checked=False→True`，文字/位置不变 | 严格匹配False；宽松匹配True | 可见状态变化被fallback掩盖 |
| 非输入框 `Date: June 1→Date: June 10` | `_label_match=True` | 字符串前缀规则把不同日期视为允许后缀变化 |
| 输入框相同日期变化 | `_label_match=False` | 输入框受保护，普通展示标签没有同等保护 |

此外，`_equiv_items`忽略无标签元素；位置变化阈值48px固定，不具备跨分辨率不变性。把pHash换成a11y显然有改进，但不能因此宣布C的状态恢复问题已经解决。

### 5.4 导航恢复、副作用恢复与风险学习混在一起

[relaunch_restore](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/full9.py:650) 返回值主要检查activity/layout，把内容是否相同另存在属性中；[调用处](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/full_run.py:960) 在返回成功后先设`rollback_ok=True`，即使内容不同，也主要记录为未来的effectful控件。

这会让一个事件同时被视为“导航恢复成功”和“留下了副作用”。内部rollback_ok率不是应用状态恢复率。事后加入禁探记忆只降低下一次风险，不能撤销第一次已读、加入购物车、自动保存草稿或设置改变。

现有 [FULL45已知问题](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/reports/mobileexplorer_design_v45_zh.md:209) 已记录父行包含switch的失败；[风险检查](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/full_run.py:1210) 当前主要看选中节点自身，没有完整传播后代状态变化能力。

更根本的问题不是危险词不够多，而是缺少“此probe的效应与恢复条件是什么”的明确契约。对未知控件的普适无副作用保证，目前材料不支持。

### 5.5 复用证据含有自我强化，但未独立验证任务正确性

[在线函数库](/Users/huangrunxi/Projects/android_world/experiments/runtime_graph_exploration/scripts/fgraph.py:246) 根据模型自报完成抽取，不等同于evaluator确认成功。这样做避免在线读取benchmark答案，具有合理性；但论文必须称为self-completed而非笼统“成功轨迹”。

[GraphReuse._apply](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/graph_exec.py:106) 把落地ok执行增加valid和evidence；[misuse检查](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/graph_exec.py:137) 主要捕捉模型紧接着Back或另开app。模型没有撤销、或者页面到达正确，都不是该步骤服务于当前任务的独立证明。

因此需要区分：转移可达证据、字段值写入证据、模型认可、任务完成证据。连续重放同一错误模式不能被统计成越来越强的独立正确性证据。

### 5.6 跨任务复用改变了研究对象与泛化范围

原稿主要讲本任务推理窗口内获得信息；现在收益可以在后续任务兑现。它解决了session-local不足的问题，却引入任务顺序、冷启动摊销、历史来源和跨任务污染等新要求。

目前整轮从空图开始、app内分组运行，是在线积累场景，不等于“每个未见任务都有立即收益”。函数/表单又依赖英文任务格式和 [正则槽位解析](/Users/huangrunxi/Projects/android_world/experiments/runtime_graph_exploration/scripts/fgraph.py:50)，并不能由无app硬编码直接推出任意自然语言/本地化GUI泛化。

此外，[goal IDF](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/full9.py:36) 来自历史BASE任务语料，UI IDF与可回退先验也有离线来源。这不自动等于标签泄漏，也不推翻“没有离线轨迹库”的限定；但需要区分在线图冷启动、已有同域统计先验和真正未见app评测。

## 6. 实验协议里还需修正的地方

1. **主结果的统计强度。** 完整e2e116n是同seed的单条任务流。53/116对54/116，只能说该次观察接近；不能证明成功率非劣。47题共同成功子集存在选择偏差，不能作为唯一主指标。
2. **版本对比不能当受控消融。** FULL44→FULL45同时改变安全策略，旧服务→n-gram+CUDA graph也改变后端。它们可以说明工程演进，不能单独识别某机制的因果收益。
3. **INJ不是“只换memory表示”。** `GRAPH_PROMPT`同时关闭多种直接执行路径并改为提示。它比较的是知识消费方式/是否绕过模型，不仅是图结构与naive memory的区别。
4. **RAND也需核对候选集。** 未知候选排序确实随机化，但RV生成处另有`RAND_EXPLORE`条件，可能同时关闭已知边重新观测。若要声称仅替换排序，须固定候选集、预算和深度规则。
5. **NOSYS是组合消融。** 它同时影响完整解码等待、动作提前执行和观测等待，不能将全部差值归给单一“pipeline子组件”。
6. **agent_s边界不准确。** [驱动](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/e2e_driver.py:193) 用各step的pre时间最大值减最小值，可能漏掉最后动作、最终观测及输出尾部。子进程总E2E是更清楚的墙钟，但agent段应有明确start/end事件。
7. **计步不等价。** 一个图宏可含多个动作但只记一个计数步；调用预算20、计数步40是实质改进，但不能把该计数步称为真实UI动作数。网络失败/流式fallback还可能多发请求，建议分开决策调用、实际服务请求与重试。
8. **异常重跑与排队处理。** 保留原错误、重跑关系和图状态来源。不能仅按TTFT阈值选掉慢样本，也需确认重跑是否继承失败attempt写入的图。
9. **模块开销与总能耗。** ms级ranking/检查不包含秒级点击、过渡、重启重放就必须直说；peak power接近不能证明任务能耗接近。

现有 [窗口汇总](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/figures/understand/summary.json) 在138个有probe窗口中列出4个`window_s > model_latency_s`；这是现有日志口径的越界读数，不是完整deadline/恢复尾时延评测，但已经不支持“从不额外等待”的绝对说法。

## 7. 哪些最值得先补，哪些可在server完成

遵循“减少启发式、优先server可复现”的要求，这里不建议再加阶段感知或更多危险词。优先处理契约、反例与因果证据。

| 优先级 | 工作 | 最小可检查交付 | server可完成程度 |
|---|---|---|---|
| P0-1 | 统一正文/回复/结果版本 | 每个结果对应run、配置、脚本hash；去掉或补齐所有Insert；统一11%/条件子集等口径 | 完全可以，先做 |
| P0-2 | 核验语义状态检查 | 对checked、selected、日期、实体、窗口、键盘构造固定反例；分开结构匹配与动作依赖 | 函数级完全可以；不能据此声称真实app全安全 |
| P0-3 | 测探索的独立收益 | 同一pipeline下2×2：是否探索×是否图执行；各臂独立记忆；追踪exploration-only边的使用、节省和成本 | server上的AndroidWorld可做；纯固定trace仅支持来源分析，不能证明闭环因果 |
| P0-4 | 补真正的强基线 | 精确说明script/validated replay实现，匹配任务和后端；计入准备/在线成本与失败恢复 | 有现成server模拟器即可；不能从论文数值拼接 |
| P0-5 | 独立恢复oracle | SQLite/偏好/测试app持久状态前后比较，统计“画面恢复但状态改变” | emulator/test app可在server做；服务端购物车等另行核验 |
| P1-1 | 把新推理方向与当前缺口对应 | 采用新的server任务书，验证图辅助specdec或候选验证是否比普通增强基线更快 | GPU server可先固定请求回放，不需要手机阶段启发式 |
| P1-2 | 冷启动与泛化 | 独立任务流、打乱顺序、未见app/模板、同义改写、冻结先验；报首次任务与后续摊销 | 大部分可在server做 |
| 范围决定 | 单手机实际闭环 | 本地动作模型+GUI同机的任务成功、E2E和资源日志 | 普通server回放不能关闭A/C/D这项；不做手机则收窄部署主张 |

2×2探索实验尤其重要：仅“无memory”“图提示”“无pipeline”不足以回答标题的核心问题。应让所有臂保留相同后端与观测/动作流水，分别测没有探索的执行记忆已经能带来多少收益，以及探索额外增加/减少了多少。

## 8. 当前可以写与不能直接写的结论

**可以写：** 当前实现将模型执行与在线探索观测组织成跨任务图，在实时控件可落地时绕过部分重复决策；完整单轮116题配对显示总体E2E约降低11%，观察到的成功数接近。流式执行与观测等待优化有额外工程作用，但须单独归因。

**暂不能直接写：** 已经证明纯单手机并发优势；在线探索是全部收益的主要来源；所有关键review问题已解决；保证无副作用或绝不增加时延；跨模型成功率不下降；script与naive-memory/general-reuse对照均已完成；总体收益就是共同成功子集的约21%。

若要判断下一轮稿件最需要补什么，我会把顺序排为：**先让证据、正文和实现一致，再证明探索独有的净收益，再解决状态检查的具体反例。** 新的server推理机制可以并行研究，但即使它加速成功，也不会自动补上这些旧review要求的证据。
