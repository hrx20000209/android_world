# MobileExplorer：overview 与三个模块图的分工和画法

日期：2026-10-27。对应最新稿 Fig.5–8：overview、Mix-Trajectory Graph、Task-relevance novelty exploration、graph-guided pipelined execution。以下给出结构草图和叙事方法，不增加新实验结论，不修改 agent。

## 1. 为什么现有图容易让人看不懂

对照最新稿 p5–7，主要问题是**看不到模块处理前后的具体变化**：

- Fig.5：Reasoning 之后才出现 Graph Match，无法解释省去了哪次推理；底部图又重复 Fig.6；缺少 live UI 恢复与动作交接。
- Fig.6：两类轨迹流入大编号网图，但没有展示合并键、内容实例、记录字段与查询输出。“图变大”没有解释“知识如何变得可复用”。
- Fig.7：控件→相关度条→Add Contact，缺少去重/风险/恢复历史的筛选过程，也没有画出相关度使用的证据。Add Contact 还与当前副作用过滤规则冲突。
- Fig.8：顶部推理、下部 reuse、底部时间轴混在一起。无法判断块的左右位置表示先后还是依赖，更无法看出恢复和完整模型输出对动作释放的约束。

原图的截图、模型图标、手机与节点数量都不少，但读者仍无法回答：**输入是什么、怎样改变、结果是什么、哪个结果以后被使用。**

## 2. 四张图各负责一个问题

| 图 | 读者看完必须能回答 | 应画的形式 | 不承担的细节 |
|---|---|---|---|
| Overview / Fig.5 | 一次决策如何走完，三个模块如何互相使用结果？ | 控制流程＋两类记忆读写 | 状态签名实现、TF-IDF公式、完整恢复阶梯 |
| M1 / Fig.6 | 不同轨迹、不同内容如何被组织成可查询的记忆？ | 观察实例→对齐键→小图与记录→查询结果 | 下一次动作何时执行、CPU/GPU调度 |
| M2 / Fig.7 | 为什么探这个控件，具体得到什么，怎样返回？ | 一个真实候选集的筛选/排序＋一次probe循环 | 全局复杂任务网图、所有探索变体 |
| M3 / Fig.8 | 哪些模型调用被绕过，未命中时哪些等待被重叠？ | 上半复用判断；下半带泳道的时间线 | 再次解释图如何合并、重复候选打分 |

图名可以仍使用论文已有正式名称，图内方框优先写操作动词：`Retrieve candidates`、`Exclude observed controls`、`Restore UI`、`Check live action`。`Graph-guided` 描述依赖关系，本身没有说明方框做了什么。

## 3. 用一个例子贯穿，不要四图四个 app

本轮草图使用一个概念 UI，方便讲清逻辑：

- 页面 L：记录列表，存在 Search / Categories / Settings / Sync(toggle)。
- 页面 R：搜索页。
- 页面 C：分类页，包含 Travel / Work。
- 任务 A：查看记录 Alice 的分类。模型走 Search 路径；探索尝试 Categories，观察 C 后回到 L。
- 任务 B：之后需要打开分类列表，且 L 中的记录内容已经变为 Bob。

贯穿的额外观察是 **`L — Categories → C`**。M1 保存它；M2 展示它如何产生；M3 展示后续满足复用条件时如何消费它。Overview 只放它所在的位置，不展开每个字段。

**例子是机制草图，不是从日志抽取的实验案例。** 投稿前应选择一条真实 episode 替换。尤其不能据此声称任意探索边都可以直接执行：实际消费者的 goal 匹配、执行支持、控件定位与提交边界必须成立。若实现中该边只用于寻找 procedure 入口，最终图就画“经该边到入口→绑定并执行 procedure”，不能冒充任意任务导航。

`Search` 标为已观察，表示当时可用记忆中已有该转移，不使用当前尚未完成的模型调用结果。排序面板只使用 probe 前已经可见的控件文字；probe 之后才出现 Travel / Work，避免把未来内容当成决策依据。

## 4. Overview：画一次闭环，清楚标出省推理的位置

[预览](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_four_figures_20261027/01_overview.png) · [可编辑SVG](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_four_figures_20261027/01_overview.svg)

### 布局

主面板从左到右：

```text
Task + live observation → M3 reuse check → applicable? → ground / execute / observe
                                            │ no
                         ┌──────────────────┴──────────────────┐
                         │                                     │
                    Model inference                     M2 probe + restore
                    frozen screenshot                   live UI interaction
                         └───────────────┬─────────────────────┘
                                  M3 action check
                         accept → execute; reject → re-observe
```

底部单独画 M1，使用两个有含义的读接口和两个写来源：

- `M1 → M2`：coverage、related destination observations、recovery history。
- `M1 → M3`：candidate transitions/procedures、selectors、source evidence。
- `agent/replay → M1`：实际执行转移及结果。
- `explorer → M1`：probe目的观察与恢复结果，包括失败。

### 必须画出来的三件事

1. **复用发生在 VLM 请求之前。** 命中分支明确标注 `skip model request`。
2. **模型读冻结观察，explorer改实时UI。** 两个分支不能都简单写“UI”，否则读者会以为各自操作独立副本。
3. **两个分支在动作交接处汇合。** 恢复晚就等待；检查失败就重新观察。不要让模型动作直接越过交接点进入UI。

Overview 中不要放裸露的 .95/.42/.17、Task1/Task2完整子图、大恢复阶梯或所有operator。M1/M2/M3编号只表示模块身份；另用①②③编号流程会与模块编号冲突，可改成无编号动词。

**推荐caption：**

> MobileExplorer checks reusable behavior before requesting a model decision. On a miss, the model reasons from a captured observation while the explorer probes and restores the live UI; action execution follows a live-UI check. Execution and exploration observations update M1, which provides separate inputs to probe selection and reuse.

## 5. M1：画观察对齐与查询，不再画大编号网图

[预览](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_four_figures_20261027/02_memory.png) · [可编辑SVG](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_four_figures_20261027/02_memory.svg)

### 左：输入观察

放两条极短轨迹，每条2–3个页面缩略图即可：

- agent：`L — Search → R`。
- explore：`L — Categories → C — restore → L`。

缩略图截取能解释转移的区域，保留页面名和控件标签。恢复箭头在轨迹里显示，是否写作导航图中的边取决于实际存储；不要把“回到起点”自动当成相同控件的逆操作。

### 中：共享索引与保留实例

画一个明确的对齐操作：

```text
page key = (app, activity, layout)
control key = family / selector
```

再放一个小 inset：L(Alice)、L(Bob)映射到同一页面索引 L，但各自内容、控件属性与来源观察仍保留。旁边写一句：`shared retrieval key ≠ interchangeable execution state`。

这正面回答老师关于 node heterogeneity 的问题。不要用一个“Merge”箭头跳过此处。

### 右：三节点图与一条展开记录

只用 L、R、C 三个节点，写页面含义，不用1–11编号。边上必须写 Search / Categories，并用颜色或虚实区分agent/explore来源。

把 Categories 边放大成一条记录：

```text
source=L; control=Categories; destination=C
origin=explore; observation=task A; recovery=accepted
```

另标出失败/冲突记录不会被覆盖。Procedure index 单独画为“来自执行路径的 entry + selectors + parameter slots”，不要暗示点击一个未知分支就生成了完整任务程序。

### 底部：输出可查到什么

只画两个出口：给M2的已知性/相关观察/恢复历史；给M3的导航候选/procedure及其证据。**如果没有查询结果，读者仍然只看见你在存一张图。**

**推荐caption：**

> M1 aligns execution and exploration observations using shared page and control keys while retaining instance contents and provenance. The example adds an exploration-observed category-page transition alongside the agent’s search transition. Separate indexes expose coverage and recovery evidence to M2 and reuse candidates to M3; a shared retrieval key does not establish execution equivalence.

## 6. M2：画出“为什么不是另一个控件”

[预览](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_four_figures_20261027/03_exploration.png) · [可编辑SVG](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_four_figures_20261027/03_exploration.svg)

这张图用一个具体候选集，不要再画抽象得分条。

### 上半：四列

1. 当前控件：Search / Categories / Settings / Sync。
2. 排除或准入：Search已观察；Sync被风险规则拒绝；Categories/Settings进入排序。画出记忆输入，不把去重也说成文本相关性。
3. 证据：任务去掉参数Alice后包含category；候选自有文字分别是categories/settings。若使用同label/rid的历史目的观察，必须画出来源页面以及该观察在probe前已存在；没有就只画自身文字。
4. 选择和新观察：点击Categories，**此时才看到**目的页C中的Travel/Work。

必要时使用小表格：`candidate / covered? / admissible? / relevance evidence / selection`。重点是展示排除理由和证据，不是让读者读一个大公式。TF-IDF、权重与阈值放正文/算法框。

### 下半：一次完整的probe循环

从C返回起点L→比较起始观察→记录目的内容与恢复结果。成功和失败都要有出口；失败后重新观察，不向变化后的UI发旧坐标。

在边上标 `during model inference`；底部只留一句停止条件：动作字段已就绪或剩余恢复窗口不足时停止发起新probe。在途probe不能被一句“stop”抹掉，时间行为由M3图说明。

不要把Add/Save等作为正向探索示例；它们正好用于展示被过滤的候选。不要把图中固定权重写成confidence probability，也不要把相关度称为已证明的information gain。

**推荐caption：**

> M2 excludes covered or inadmissible controls, then ranks remaining candidates using task terms and available UI evidence. In the example, Categories matches the task’s operation/object terms and reveals a new category page. The explorer attempts recovery, checks the resulting live UI, and records the observation and recovery outcome for later lookup.

## 7. M3：把复用逻辑与真实时间线分成两个面板

[预览](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_four_figures_20261027/04_execution.png) · [可编辑SVG](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_four_figures_20261027/04_execution.svg)

### 面板(a)：后续任务如何消费那条边

任务B＋当前L→检索 `L — Categories → C`→检查任务/参数、当前控件、执行支持与提交边界→执行并观测C。把 `no model request for this transition` 写在成功分支上。未命中或不满足条件，落到面板(b)的模型路径。

如果真实实现消费的是procedure，就在此展示 `entry → fill current parameters → stop before commit`，并用Alice→Bob说明参数变化。不要为了一张图同时罗列launch、confirm、form、macro、函数库等所有operator；正文用表格列变体。

### 面板(b)：时间线必须有独立泳道

横轴统一为time，纵向分三条：

- Model output：action fields→summary tail。
- Live UI：probe→restore→execute model action→observe。
- Runtime：action-ready与恢复结束后，检查/重定位或重新推理。

**probe、restore和正常动作共用一条Live UI泳道。** 这能让读者直接看到只有一个UI，探索不能一边未恢复、一边执行旧动作。Summary tail与动作/观测可重叠；下一次prompt仍需等必要输出与新观察。

用一条竖虚线标action-ready，用箭头/标记指出recovery end。动作交接不早于两者都满足的时刻。图中的时间宽度只是示意，不作为速度比例或实验结果。

### 小面板(c)：恢复晚的情况

画一个超过action-ready的恢复条，超出的部分明确标wait。这样读者理解优化是在减少关键路径上的净等待，而不是所有探索永远免费。

**推荐caption：**

> M3 first checks whether retrieved behavior applies to the current task and UI, avoiding a model request when reuse succeeds. Otherwise, action generation overlaps with UI probing and recovery. The runtime checks the live action before execution; remaining output can overlap execution and observation. Recovery that finishes after action availability adds waiting to the critical path. Timing is schematic.

## 8. 四图统一的表达规则

- 固定模块身份：M1记忆、M2探索、M3复用与交接。模型是被调用的推理组件，不再并列叫另一个核心模块。
- 同一个页面始终叫L/R/C；同一个控件始终叫Categories。保持英文labels和正文一致，正式稿可将页面字母放在真实截图角落。
- 每条关键箭头标一个数据或事件：`candidate + selector`、`observation + recovery`、`action fields ready`。不要只写“guide”“enhance”“knowledge”。
- 控制图按逻辑顺序排；时间线按真实时间排。不要拿一根time箭头装饰普通模块框图。
- 仅重复模块名与贯穿例子，不重复整套内部逻辑。图注写处理前后发生了什么，而不是罗列颜色和框名。
- 单栏图必须减少面板与字数；当前草图按跨双栏布局设计。最终7英寸宽度下检查最小字大约7–8pt，去掉讲解用副标题与长脚注后再导出。

## 9. 草图的实现边界

SVG显示的是建议的逻辑表达。当前多处检查仍需按上一份设计审计收紧；不能把“check”方框自动解释为语义安全保证。页面匹配不能恢复不可见持久状态，Back不等于撤销业务副作用。M1的物理数据结构也不必是单一同构图。

本轮没有把新提出的缺失前提/evidence-request接口画进既有系统，没有引入阶段感知或资源状态调度，也没有用旧设计prompt里的尚未实现机制填图。

源文件：[draw_storyboards.py](/Users/huangrunxi/Projects/android_world/docs/figures/mobileexplorer_four_figures_20261027/draw_storyboards.py)。它复用前一版绘图helper；SVG本身可以直接编辑、导入矢量工具。PNG均由SVG渲染并逐张检查。
