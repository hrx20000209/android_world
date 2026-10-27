# MobileExplorer：进一步加速的系统设计与验证路线

日期：2026-10-26（香港时间）。本文是研究建议，新增机制均未被本次工作实测验证。没有启动设备实验，也没有改动 agent、驱动或正在运行的任务。

## 1. 建议先做什么

优先验证一个统一方向：**让在线探索成为由关键路径、资源争用和观测有效性共同约束的运行时服务。** 其三个候选组件是：

1. **资源与截止时间感知的探索调度**：判断此刻是否值得探索、能否及时恢复，以及探索会把本次决策拖慢多少。
2. **带版本的观测与执行检查**：把截图、a11y、动作和回滚结果的对应关系显式记录，减少过时观测造成的补拍、误操作和重新推理。
3. **面向后续推理的计算准备**：探索不仅写入导航边，还在有实际收益时准备可复用的图像预处理/编码结果，并管理其有效期和内存。

前两项最贴近当前代码，第三项更依赖模型结构和实际命中率，收益最不确定。另列“图候选 + 批量验证”作为备选；它与 V-Droid 接近，不建议在未验证差异之前作为主贡献。

**先测收益上限，再实现新模块。** 如果同设备争用很小、补拍耗时很少、可精确复用的图像编码只占很小比例，这条路线就不足以支撑新的系统贡献，应及时停止，不能靠增加模块数量弥补。

## 2. 本次核实的代码与现有证据

执行了 `git fetch origin` 和 `git pull --ff-only origin main`，结果是 Already up to date。当前 HEAD 与 origin/main 均为 `643303c9fb8c010edeb0dec3a63887055e39ca95`。本地存在尚未提交的实验改动，全部保留；本分析针对“该提交 + 当前工作区”，不假定只看 Git HEAD 就能复现实验。

主要阅读入口：

- [主循环与系统机制](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/full_run.py:173)：图执行、探索、决策检查、预算控制。
- [现有探索截止规则](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/full_run.py:880)：最近 15 次动作到达时间的中位数减 2 秒；该判断在 `i_t > 0` 时执行，首个 probe 没有经过同一条截止检查。
- [截图补拍](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/full_run.py:1177)：activity/layout 改变而像素变化很小时补拍；同 layout 的日期、值或开关变化可能不进入此分支。
- [流式输出与历史补全](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/full_run.py:1383)：动作字段完整即执行，`_resolve_pending()` 仍会 join 后台生成线程。
- [实际 prompt](/Users/huangrunxi/Projects/android_world/android_world/agents/gelab_agent.py:1042)：system → task/history → screenshot；历史更新会改变图像前方的上下文。
- [FULL45 设计说明](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/reports/mobileexplorer_design_v45_zh.md)、[完整 e2e116n 报告](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/reports/e2e116n_final_zh.md)。

对 `e2e116n_runs.jsonl` 按 task/arm 取最后一条记录重算：

| 统计范围 | BASE | FULL45 | 能说明什么 |
|---|---:|---:|---|
| 116 题成功数 | 53 | 54 | 单轮结果接近；不能证明成功率不下降 |
| 全部任务平均 E2E | 151.87 s | 135.16 s | 减少约 11.0% |
| 严格共同成功 47 题 E2E | 129.34 s | 101.98 s | 减少约 21.2%；这是条件子集指标 |
| 同一 47 题模型调用 | 7.98 | 6.34 | 减少约 20.5%，表明减少调用已是重要收益来源 |

报告中的机制计数为 early action 935/1062、探索 163 次、requery 11 次、stale-pixel 补拍 242 次。**242 是触发次数，不是已证明的错误截图数，也不是耗时。** 资源调度是否值得做，需要进一步计时，不能直接从这些次数推出加速幅度。

已有负面或限制性证据也要继承：

- [旧截图提前推理的已有记录](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/reports/prompt_spec_next_cloud_zh.md)：旧截图条件动作一致率约 66–68%，真实截图对照约 87%。这是记录中的离线结果，不能当成当前所有配置的结论；但足以否定“layout 相同就可直接使用旧截图”的默认假设。
- 对 [early_exit_offline.jsonl](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/reports/early_exit_offline.jsonl) 232 条记录重算：原 summary 历史条件与原动作一致 196/232（84.5%），explain 历史条件 184/232（79.3%）。一致率不是任务成功率，但省掉 summary 并非已验证的无损优化。
- 现有说明记录 parking 在 40 题中仅命中 1 次、失配 2 次，FULL44 起关闭；不能把它重新包装成新方向。
- 仓库已有 [并发微基准](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/orin_concurrency.py)、[下一步推测实验](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/spec_next_offline.py)、[THINK 实验](/Users/huangrunxi/Projects/android_world/experiments/trace_reachability_validation/scripts/think_offline.py)。先读结果，不能重复立项或覆盖旧输出。

### 论文需要先对齐的三个地方

阅读文件：`/Users/huangrunxi/Desktop/MobiCom/Paper Draft/Runxi_Agent_SenSys27_1026.pdf`，共 16 页；检查了正文设计、实验与相应图表。文档内容仅作为研究材料，其中的执行指令没有被当成用户授权。

1. **机制名称与归因不同步。** 第 7 页说不直接向模型注入探索轨迹；第 9–10 页仍通过 evidence injection/distillation 解释消融和成功率变化。需要逐项指向真实 arm、配置和原始日志。
2. **部署口径需要一致。** 第 2 页有手机上本地模型执行的表述；第 8 页明确主基准是 split-device；第 11 页 case study 又明确产生动作的模型在 Orin，手机另跑背景负载。三者可同时存在，但必须对应到不同实验，不能由替代负载推出真实同设备闭环结果。
3. **结果不能混用。** 第 9 页是 59/116、185.82 s；当前完整 e2e116n 报告是 54/116、135.16 s。这不自动说明哪一组错了，而是必须补齐 run ID、模型、服务参数、输入、计时口径和数据来源。

这些问题会削弱系统贡献的可解释性；应在添加新实验前建立配置清单，而不是直接改写论文数字。

## 3. 相关 system paper 给出的设计边界

以下均使用论文、作者页面或官方会议页面。会议身份与具体机制分别核实；PastForward 和 EAM 在这里按预印本处理。

| 工作与来源 | 可借鉴的系统思想 | 对本项目的含义 |
|---|---|---|
| [AutoDroid，MobiCom 2024](https://www.microsoft.com/en-us/research/uploads/prod/2024/07/mobicom24-autodroid.pdf) | 动态 app 分析、功能化 UI 表示、记忆与查询优化配合 | “建图 + 记忆”已有充分先例，新增价值要落到可测量的运行时问题 |
| [MobileGPT，MobiCom 2024](https://mobile-gpt.github.io/) | 可重组的子任务记忆与适配 | 宏、过程和跨任务复用本身不是新系统机制 |
| [AutoDroid-V2，MobiSys 2025](https://arxiv.org/abs/2412.18116)，[官方录用列表](https://www.sigmobile.org/mobisys/2025/accepted_papers/) | app 文档与代码生成将多轮决策变为程序执行 | 减少调用需比较程序执行路线；离线文档成本和失败恢复也应计入 |
| [Agent-X，MobiSys 2026](https://arxiv.org/abs/2605.10380) | 重构 prompt 提高前缀复用，结合不依赖额外 LLM 的推测解码 | 普通 prefix cache、n-gram 或拼接静态 prompt 更适合作为增强基线 |
| [VLMCache，MobiSys 2026](https://doi.org/10.1145/3745756.3809243)，[官方说明](https://www.sigmobile.org/mobisys/2026/accepted_papers/) | 稳定/变化视觉块分离与重新融合，使视觉计算可复用 | “只重算变化区域”已有直接先例；普通 ViT patch 与 decoder KV 不能按像素差随意拼接 |
| [TimelyLLM，MobiSys 2026](https://www.anuragkhandelwal.com/papers/timelyllm/) | 按 agent 执行需求分段、调度生成，考虑输出何时被需要 | 借鉴关键路径与 slack；其多 agent、物理 I/O 场景不同，不能直接搬用其收益 |
| [MMEdge，SenSys 2026](https://arxiv.org/abs/2510.25327)，[官方录用列表](https://sensys.acm.org/2026/accepted_papers.html) | 将采集与编码拆成可流水的单元 | GUI 中下一屏由动作触发，要显式表达闭环依赖；持续传感流水不能直接等同于 GUI 流水 |
| [PELM，SenSys 2026](https://doi.org/10.1145/3774906.3802783) | 联合调节推测工作量、验证深度和频率，考虑温度/功耗 | 可借鉴在线控制方法；不是给 GUI agent 加一个通用 DVFS governor 就形成新贡献 |
| [Libra，SenSys 2026](https://doi.org/10.1145/3774906.3800479) | 面向持续上下文的分块压缩、交换和生命周期管理 | 图若持有推理状态，就必须同时处理内存成本与恢复成本 |
| [mzCache，MobiCom 2026](https://arxiv.org/abs/2609.01338)，[官方日程](https://sigmobile.org/mobicom/2026/accepted_papers_schedule.html) | 手机多任务下管理被驱逐的模型与上下文，缩短恢复路径 | 多 app 场景要测内存压力；缓存命中率不能脱离内存预算讨论 |
| [V-Droid，MobiCom 2026](https://v-droid-agent.github.io/) | 离散候选动作与 prefill-only 验证减少生成代价 | “生成改验证”已有直接先例；其展示的低延迟使用 2×4090，不能视为手机时延 |
| [PastForward，2026-09 预印本](https://arxiv.org/html/2609.32166v1) | 已执行经验支持 token 提案、预测下一屏提前推理及跨步 KV 复用 | 下一屏提前推理与计算经验复用不是空白；其接受检查要求实际 app 内容匹配，不能把它描述为只匹配布局 |
| [EAM，2026 预印本](https://arxiv.org/abs/2605.12294) | 可执行记忆图、动作组与图搜索 | 参数化图过程需与可执行记忆路线区分，不能只靠换术语形成差异 |

因此，比较可信的候选研究缺口是：**当在线探索、前台 UI 和本地推理共享资源时，如何使探索产生的未来收益大于当前争用、恢复、失效和内存代价。** 这是从文献与代码得到的研究判断，不是已证明的新颖性或性能结论。

## 4. 先用关键路径模型约束设计

一次模型请求从时刻 0 开始，定义：

- `A`：动作字段完整到达时间；`R`：探索及恢复完成时间。
- `V`：恢复后的检查耗时；`U`：动作执行到得到可用下一观测的耗时。
- `F`：完整输出（含 summary）结束时间。

当前流水的简化下界是：`next_request >= max(max(A, R) + V + U, F)`。

它揭示三个问题：

1. 探索即使满足 `R < A`，仍可能通过资源争用增大 `A` 和 `F`，并不自动免费。
2. summary 的可节省量是未被 UI 执行与观测隐藏的残余，不是整段 summary decode 时间。
3. 重复截图、重复检查与网络排队可能混入 TTFT。**客户端 TTFT 不等于纯 prefill**，需要服务端阶段计时。

对一个候选 probe，使用统一的时间收益估计：

`G = P(在有效期内被使用) × 预计避免的关键路径时间 - 推理争用增量 - 探索越界增量 - 预计恢复/重推代价 - 缓存驱逐的预计代价`。

所有代价先统一成秒；能耗与内存用约束单独管理。不能把 TF-IDF 相关性直接当概率，也不能把“以后访问该页”当成“一定省下一次推理”。图执行已经跳过的模型调用不能再记成计算缓存的收益。

## 5. 设计 A：资源与截止时间感知的探索调度（第一优先）

**关键变化：探索窗口是系统决策的结果，而不是固定给定的免费时间。**

将工作拆成可计时的单元：读取 UI、候选排序、probe、观测、恢复、检查、图像预处理。标记它们的 CPU/GPU/内存需求。前台请求也拆成视觉处理、prefill、decode、动作后 summary；不同后端以实测阶段为准。

运行时每次决定：开始/延后哪个 probe，是否只做图整理或 CPU 预处理，什么时候必须恢复。优先实现小型规则控制器，不先上 RL：

1. 用同设备实测估计任务与阶段的相互拖慢量，不预设 prefill 或 decode 一定更适合探索。
2. 用当前请求进度预测动作还需多久；没有进度接口时使用按阶段、输出类型分组的保守经验分布。
3. 对每个 probe，包括第一个，检查 `P(探测 + 恢复 + 检查 <= 剩余动作时间 - 裕量) >= 1-ε`。初步可用恢复耗时的高分位数与动作到达时间的低分位数；概率需在留出数据上校准，不宣称硬实时保证。
4. 只接纳预计 `G > 0` 的 probe。对低复用页面、内存压力或持续温升，减少主动工作。
5. 动作已到达时停止发新 probe，优先完成在途恢复。取消推理必须确认服务端真的停止计算，不能只关闭客户端连接。

**与现有 FULL45 的区别**：不仅将固定 2 秒改成分位数，还同时控制“争用拖慢”“恢复尾部”和“未来价值”；若只修改常数，仍然是参数调整。

**与 TimelyLLM/PELM 的区别**：调度对象是一个 GUI 闭环中的探索及恢复事务，且会改变未来图覆盖与推理次数。这个差异需要消融证明，不能仅凭对象不同就宣称创新。

**可能否定它的结果**：split-device 上几乎没有争用；真实同设备上现有探索量也不足以影响推理；仅减少探索次数就能达到同样收益。上述结果意味着应缩小主张，而不是强行保留复杂调度器。

## 6. 设计 B：带版本的观测与执行检查（第二优先）

**关键变化：把“观测何时可用、哪些结果依赖它”变成显式接口。** 不是再加一个像素阈值。

每次观测携带 `obs_id、action_seq、window_id、layout_version、content_version、截图时间、a11y 获取区间、可见上下文`。导航图仍可按内容无关的布局组织；执行检查与计算缓存使用更严格的键。

可先在 Android helper 中使用窗口/内容变化事件唤醒观测，替代固定 0.15 秒轮询。Android API 30 的 `takeScreenshot` 返回截图时间，API 34 可按窗口截图；需要声明截图能力。来源：[AccessibilityService](https://developer.android.com/reference/android/accessibilityservice/AccessibilityService)、[ScreenshotResult](https://developer.android.com/reference/android/accessibilityservice/AccessibilityService.ScreenshotResult)、[AccessibilityEvent](https://developer.android.com/reference/android/view/accessibility/AccessibilityEvent)。

工作流程：

1. 记录动作发出序号，监听相关窗口事件；事件合并后读取候选 a11y/截图，避免每个事件都触发完整截图。
2. 截图前后检查版本与关键内容，剔除明显的过渡帧、旧帧和图树冲突。
3. 对模型动作和图动作记录可检查的依赖，例如目标元素、实体标签、选中日期、输入值、列表上下文；依赖变化则失效，只有位置变化才允许重新定位。
4. 对完整 VLM 结果，默认依赖完整输入观测。**不能从最后点击的控件反推出模型只看了这个控件**；无法可靠得到 read set 时采用保守检查并回退。
5. 隐藏 app 状态不一定在截图/a11y 中。对无法验证的上下文不放宽复用，不声称恢复检查等价于完整 app 状态回滚。

事件静默、截图时间晚于动作、layout 相同，都不是原子快照的证明。此方案只能通过检查与超时回退减少不一致；视觉自绘界面、动画和未上报的内容变化必须单独测。

**预期作用**：为调度和缓存提供失效边界，并尝试减少重复观测与过时决策。若只是减少少量补拍，适合作为支撑组件；若能解释并消除明显的重复推理和长尾，才可能成为重要系统贡献。

## 7. 设计 C：让探索生成可复用的计算产物（高风险、先测上限）

现有图记录“可到达哪里、怎么回去”。候选新接口进一步记录“以后到达此处时，哪些推理准备已完成、何时会失效、占多少内存”。用途必须针对**图无法直接跳过、仍然需要模型的步骤**。

按依赖分层，而不是把页面截图与 KV 混为一谈：

| 缓存对象 | 可接受的初始复用条件 | 不允许的推断 |
|---|---|---|
| 图像解码、缩放、patch 输入 | 相同实际模型输入、相同预处理配置与尺寸 | 原始截图看着像，就视为相同输入 |
| 独立视觉编码器/投影结果 | 架构确认其与文本上下文独立，且模型权重、精度、位置/形状信息一致 | layout 相同就复用视觉特征 |
| decoder prefix KV | token/embedding 前缀、位置与上游上下文均一致，后端支持正确恢复 | 同一张图或同一页面就复用整段 KV |
| 部分视觉块、近似视觉/KV 状态 | 单独实现并测准确性，比较 VLMCache/PastForward | 将近似复用称为无损或自然成立 |

最小原型先做精确的整图预处理/视觉输出复用。图按 `有效期内访问概率 × 可避免时间 / 内存字节` 管理有限预算，并把恢复失败、参数写入、页面更新关联到缓存失效。

探索时首次进入未知分支，可以趁机准备其计算产物；真正再次观测到**当前**输入、验证命中后才消费缓存。动态页不直接拿旧截图喂给模型。执行证明只说明一条导航边能走，不等于缓存内容仍有效。

**与已有工作的差异假设**：面向主动探索发现的分支，联合决定是否获取、预计算、保留与失效，而不是单独提出新的视觉缓存算法。必须通过“普通缓存”“无探索缓存”“同预算预计算”对照检验这个差异是否有价值。

**核心风险**：ViT 很快，prefill 大头在 decoder；精确命中少；未来步骤已被图绕过；预计算与当前推理争带宽；缓存挤出模型/KV。任一情况都可能让收益小于成本。仅靠降低 TTFT、提高 hit rate 不能宣布端到端加速。

## 8. 备选 D：图候选 + 共享前缀的批量验证

对“不能直接复用、但图给出少量候选”的导航步骤，用一次候选验证代替长篇自回归动作生成。候选可由图边、当前可见控件和任务槽位共同提供，验证不确定时退回原生成器。

这与 V-Droid 的核心思想重叠较大。潜在增量只能是在线探索形成候选、控制候选覆盖、共享图像/文本前缀以及有依据的回退，不能声称首次把 GUI 决策改成验证。

先测 top-k 候选是否包含正确动作，再测 verifier；普通 GELab 的 teacher-forcing likelihood 不自动是可靠 verifier。要控制候选长度、模板、顺序和分数校准；批量评分不必然只付一次 prefill，需要后端真正共享前缀。额外训练成本、显存、失败后的双重推理必须记录。只在小样本有信号时继续。

## 9. 给 Luna 的实验清单与停止条件

以下数值是建议预先登记的工程筛选线，不是实测结论或统计显著性的替代。

| 编号/优先级 | 要验证的问题 | 实验与对照 | 必须交付的结果 | 继续/停止依据 |
|---|---|---|---|---|
| E0 / 必做 | 现有结果对应哪份代码与部署？ | 冻结代码 hash、模型/后端/服务参数、输入模式、图来源、任务顺序；将 PDF 各表映射到 run | 配置清单；已完成/进行中/不可复现分别标记 | 不可追溯的结果不用于新机制归因 |
| E1 / P0 | 剩余关键路径到底在哪里？ | 先分析完整 e2e116n；新增阶段计时用约 100–200 个固定实际请求，前台独占后端；完整记录 encode、排队、prefill、decode-to-action、恢复、观测、summary join | 每任务时间线、阶段占比、p50/p95；客户端与服务端计时对齐 | 若目标阶段总占比过小，先停止该分支 |
| E2 / P0 | 各机制的理想收益上限够不够？ | 离线构造零争用/零恢复越界/理想观测/精确视觉复用的分别上界；只计仍需模型的步骤；再将推理时间缩放为 0.5×/0.25× | 各方向可节省秒数与 E2E 比例、假设列表、快模型下探索窗口变化 | 建议：上界不足 5% 的方向不作主贡献；上界不是实测加速 |
| E3 / P1 | 调度器能减少争用而保留有用探索？ | 同设备固定请求/固定 probe 脚本：无探索、现有规则、仅改截止、仅资源感知、两者结合；跨轻重负载与冷/热状态；不同阶段各约 30 对交错重复 | 动作到达拖慢、恢复越界分布、probe 采用率、能耗、温度；在线预测校准 | 若简单减探测量同样有效，采用简单方案；不能仅以 probe 数增多为胜 |
| E4 / P1 | 观测机制解决真实不一致还是多等了一会？ | 事件/轮询 × 旧检查/版本检查的 2×2；注入绘制延迟、同 layout 改值、列表更新、弹窗、键盘；每类约 30 次 | 观测可用时延、旧帧误接受/新帧误拒绝、截图次数、错误动作和 requery | 检查延迟也计入；若只改善阈值个例，缩为工程修正 |
| E5 / P1 | 探索计算缓存有净收益吗？ | 精确无缓存、普通视觉缓存、探索预计算+LRU、探索预计算+图效用；同一内存预算；100–200 真实请求起步，覆盖静态/动态页 | encode/decoder 分项耗时、精确命中、失效、节省/浪费算时、内存与 E2E | 若仅 ViT 复用的可实现净收益不足 5%，不继续大改后端 |
| E6 / P2 可选 | 图候选验证是否值得替代生成？ | 留出 app 或任务模板约 300–500 步；k=1/2/4/8，比较原生成、图直接选择、普通模型评分、经训练 verifier；含回退 | 候选覆盖-精度曲线、错误类型、回退成本、共享前缀实测成本 | 候选覆盖或高置信精度不足就停止；动作一致率不能代替闭环成功率 |
| E7 / P1 通过后 | 三组件是否带来独立增益和协同？ | FULL45+同一增强后端作为对照，A/B/C 用 2×2×2 或先单项再完整；10题排错→40题pilot→全116题多seed | 全任务与条件子集分开；成功率/置信区间、调用、agent E2E/总E2E、deadline success、能耗；A/B/C归因 | 只有相对当前强基线仍有稳定额外收益才写新贡献 |
| E8 / 最终 | 真手机持续使用能否成立？ | 真正产生动作的模型与 GUI 同机；代表任务连续15–30分钟，重复≥3次；对比split-device、替代负载仅作诊断 | 持续吞吐/时延、温度、频率、峰值内存、J/任务、后台app影响 | 如果只有分离设备有效，就按分离部署报告，不推广为全端侧结论 |

### 实验口径不能省略

- 固定模型、量化、prompt、后端优化、温度策略、步数和墙钟预算；只改变被测模块。n-gram、CUDA graph、APC 等服务设置在各臂一致，先确认实际生效。
- 保留每题最多 20 次模型调用、40 个现有计数步的规则，同时另记真实动作数、图动作数和 probe 数；不能把多动作宏变成不受约束的预算优势。
- 区分三种部署：远端决策+本地GUI；远端决策+本地替代负载；真正本地决策+本地GUI。当前 `_LOAD_TEXT` 是文本负载，不能直接代表真实 VLM 的视觉编码、GPU占用或阶段分布。
- 对同一实际请求交错重复，避免把 THINK 长度差、网络波动、热状态或其他客户端排队当作机制收益。争用实验保留其争用数据；不能仅按 TTFT 超阈值选择性剔除慢 episode。
- 每个 arm 的图/缓存独立，从空开始，跨任务在线积累；任务顺序相同并另测随机顺序。阈值按留出 app/模板定，不能查看后续任务结果再决定是否探索。
- 报告冷启动、在线变暖、稳定阶段。按 app 聚集的顺序可能放大复用机会，应披露并测敏感性。
- 成功必须以 evaluator 判定；partial credit 单独列，不能把 0.5 当完整成功。失败/提前终止也保留；另报固定 deadline 下成功率和失败赋统一超时惩罚的时延，防止“更早失败所以更快”。
- 共同成功子集有选择偏差。全任务为主，条件子集用于解释；bootstrap 按任务或独立任务流聚类，不能把相关步骤当成独立样本。在线记忆共享时用多条独立运行流检验；配对成功结果可用 McNemar 分析。
- 10/40 题只是筛选。若预设成功率非劣界为 2 个百分点，需报告配对区间；样本不够就写“证据不足”，不能把差异不显著等同于不降成功率。
- 缓存访问、失效检查、拷贝、编码、预计算浪费、恢复、重新推理、初始化和评估都要有计时；重叠工作按时间线归因，不能重复扣减。
- 能耗报告任务总能量与 J/成功任务；电池估计注明采样方式，峰值功率接近不等于能耗接近。

## 10. 如何决定论文下一步

先做 E0–E2，再用 E3/E4/E5 的微基准选择真正有量级的模块。A 最先实现，但如果资源争用上限不足，就不要预设它一定成为主贡献。B 更像必要的正确性接口；C 必须证明主动探索比普通缓存多带来收益。

若结果支持，可以形成如下待验证主线：**在共享资源的端侧 GUI 闭环中，运行时根据当前关键路径与未来复用价值选择探索事务，并通过带版本的观测管理其执行和计算产物，使在线探索有可量化的净收益。**

要支撑这句话，至少需要：相对现有 FULL45 和同后端强基线的额外加速、真实同设备因果证据、模块消融、失效/恢复长尾以及能耗/内存曲线。如果这些证据不成立，就诚实地把现有成果定位为执行复用框架，系统优化作为补充。

## 附：阅读时的工作区指纹

以下 SHA-256 仅用于识别当时版本；后续聊天应重新检查，不能覆盖用户后来产生的改动。

| 文件 | SHA-256 |
|---|---|
| `experiments/trace_reachability_validation/scripts/full_run.py` | `04a46f6be794ec0d74fa2a6d711347fc86a75a343bbc72d9670ba1b0fbff588b` |
| `experiments/trace_reachability_validation/scripts/e2e_driver.py` | `60a2b51ee5fcd9841bfe70864165e89fb0e3e2cf5a058c1c9da859a94c8e966a` |
| `experiments/trace_reachability_validation/reports/e2e116n_final_zh.md` | `541c3fa15092e3d49b6b0da67c23ae611be608f8174a8ff59e54156504cd3696` |
| `experiments/trace_reachability_validation/reports/early_exit_offline.jsonl` | `4e201247fcee4031cb13b4ecea9470d3805d0fa1af7eca00a5501876bd8a0d13` |
