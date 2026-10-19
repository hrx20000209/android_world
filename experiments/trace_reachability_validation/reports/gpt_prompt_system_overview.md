你是一名移动系统 / GUI agent 方向的论文写作助手。请为我的论文（投 SenSys，系统名 MobileExplorer）写 **System Overview** 一节（英文，学术写作风格），并给出 Figure 2（system overview 图）的图注。我会附上当前的 overview 图。

请严格依据下面"系统事实"写作，不要添加事实之外的机制或结论。图与事实不一致时，以事实为准，并在正文之后单独列出"图需要修改的地方"。

---

## 1. 写作要求

- 篇幅：正文约 700–900 词（双栏论文约 1 栏），外加一段约 80–120 词的图注。
- 结构：
  1. 一段总述：问题、核心思想、两个循环 + 一张图；
  2. 按组件分段，依次为：推理循环与图引导执行、并行探索（含资源调度器）、互补证据图（CEG）；
  3. 一段单步工作流（walk-through），按时间顺序讲清一个推理步内发生了什么，并对应图中编号。
- 每个组件都要写清楚三件事：**输入是什么、按什么原则做决定、输出写到哪里**。导师之前的批注是"技术细节不清楚、利用探索结果的原则太 naive、看不出记忆构造对 GUI 运行时有什么特殊之处"，这一节必须正面回答这三点。
- 只写机制和设计原理，不放实验数字；必要时用一句"(§X evaluates ...)"指向实验章节。
- 术语保持统一，见第 4 节术语表。
- 不要使用夸张措辞（如 "revolutionary", "significantly outperforms"）。

## 2. 系统事实（以此为准）

**问题与动机**
- 4B 级 VLM agent（GELAB-ZERO-4B）在 AndroidWorld 上逐步操作手机，每一步调用一次 VLM 推理。
- 界面在运行时会变化（首次启动引导页、弹窗、列表内容、标签页状态）。历史轨迹复用假设界面与历史一致，遇到运行时新状态会失效。
- MobileExplorer 在 agent 推理的同时探索界面，把 agent 执行与探索观测合并成一张运行时图，并用这张图**直接执行能确定的步骤以替代 VLM 推理**，同时为探索本身提供选择依据和恢复路径。

**(A) 推理循环与图引导执行（推理之前）**
- 每一步在调用 VLM **之前**，先检查一组"图执行算子"。每个算子 = 入口条件 + 参数绑定；条件满足就直接在设备上执行，不调用 VLM，计 1 步：
  - `open_app`：在桌面上，按任务中的 app 名直接打开，且仅限图中见过的包（不计步）；
  - `confirm`：模型自己选择提交后弹出的纯确认对话框，直接点确认；
  - `fill_record`：参数化函数的首步能在当前屏落地时，用当前任务记录的字段值执行整段函数；
  - `form`：当前屏有空输入框，其标签与任务记录的字段名匹配时，填入对应值；
  - `graph_reuse`：本局中在同一页面上模型两次选择同一安全导航控件、落地一致且离开了本页时，后续直接执行；页内重复（如翻月、数字键）不复用，模型撤销即刹车。
- **两条原则：**
  1. **落地（grounding）**：要操作的控件或字段必须在当前实时屏幕上找得到；
  2. **停在提交前（stop before commit）**：Save / Send / Delete 等提交动作永远交给 VLM。
- 条件不满足时调用 VLM。图执行过的动作会以 "[graph] Typed ..." 的形式写入 agent 的动作历史，让模型知道哪些步骤已完成。
- **不向 prompt 注入图信息**：实验中各种注入 / 高亮形式都无效，因为图提示的正确率低于模型自身，而模型又倾向于服从提示。这是选择"执行替代决策"而非"提示增强"的依据，请在正文用一句话交代这一设计选择。

**(B) 与推理并行的探索（Parallel Exploration under Reasoning Latency）**
- 每次 VLM 推理期间（中位数约 1.4 秒），在同一台设备上并行做只读探测，探测数量由资源调度器的自适应探索预算决定（见下）。每次探测都是"点击一个候选控件 → 观测 → 恢复原状"，前一个恢复完成后才开始下一个；每个控件容器每个窗口最多探 1 次。
- 当前屏处于暂态（弹窗、对话框、输入法、选择模式、抽屉）时不探测：这些状态不在返回栈上，探测会毁掉它们。
- **候选过滤（全部依据图）：**
  1. **去重**：(app, activity, 控件族) 的转移已在图中（不论来自 agent 还是探索、哪个任务）则不探；
  2. **副作用排除**：标签首词为操作动词的控件（Add to / Delete / Save / Send / Start / Play 等）、开关类控件，以及图中学到的副作用控件 / 容器，都不探；
  3. **可回退性预测**：P(rev) 是 Beta 后验，只探 P(rev) ≥ 0.7 的控件。先验是跨 app 统计的"控件角色 × 屏幕位置"可回退率（例如带选中态的兄弟组——标签页——只有约 37%）；证据是本 app 同一容器在图中的回退成功 / 失败次数。
- **价值打分（统一的图探索价值）：** 相关性 = 加权 TF-IDF 余弦（任务操作词向量 · 证据文本向量）。
  - 任务侧：去掉 app 名、日期、数字和参数值后剩下的操作词和对象词；
  - 界面侧：词权重是全局界面 IDF 与图中当前 app 页面局部 IDF 的几何平均。
  - 按图中证据强弱分层取证据文本：
    - L1 已知边的目的页内容，已知页面的价值沿图回传：V = max(rel, 0.7 × 子页最大 V)；
    - L2 图中同标签 / 同 resource-id 控件所通向页面的平均价值；
    - L3 控件自身的标签、resource-id 和说明文字。
  - 例：一个无文字的 "+" 图标，只看自身文本时得分接近 0；但图中它的目的页是含 Name / Amount / Category / Note 的表单，得分就很高。
- **四类探测：**
  - **E1**：预测模型下一步会点的控件并提前打开；模型若确实选了它，就复用预观测的下一屏；
  - **FX**：按价值排序的未知控件；
  - **D2**：当前页没有未知控件时，沿一条已验证可被 Back 撤销、跨 activity 的已知边深入一层；
  - **RV**：对已知边在当前页面内容下重新观测，刷新随任务变化的动态内容。
- **资源调度器（Resource Scheduler）**：由资源监视器（Resource Monitor）观测设备资源状态，并据此给出自适应探索预算（Adaptive Exploration Budget），决定每个推理窗口内探测多少。请按图中的设计描述这一组件，保持在机制层面；具体监测哪些资源信号、预算如何随之调整，如果下文没有给出，不要自行编造，列入输出的第 4 部分由我补充。
- **恢复阶梯（rollback）**：
  1. 原地变化 → 再点一次；
  2. 切换了标签 → 选回原标签；
  3. Back（最多 2 次）；
  4. 沿图中探索边恢复（≤2 跳）/ 联合图安全路径恢复（≤3 跳）；
  5. 重启加重放：强制停止 app，重新启动，按本局自启动以来记录的动作路径重放（仅当路径不含提交、≤12 步）；
  6. 内容校验：恢复后内容变了，记为副作用。

**(C) 互补证据图（Complementary Evidence Graph, CEG）**
- **两种来源合成一张图：**
  - inference trajectory：agent 实际执行的转移；
  - exploration trajectory：探测观测到的转移。
  - 两者统一到同一页面键和控件族上，可混合使用。
- **屏幕层（screen layer）：**
  - 页面节点 = (app, activity, layout signature)，**不含内容**：列表多一行、时间变化不产生新节点。系统另外维护多粒度签名（exact / struct / layout / content），分别用于不同用途（"是否恢复原状"用 struct + content，落地校验用 activity）；
  - 边 = (页面, 控件族) → 目的页面，每条边记录：来源（agent / explore，以及最早发现者）、可回退性证据、目的屏截图；
  - 容器级可回退性：控件角色由最近的具名父容器（标签栏、数字键盘、工具栏）表示；一个控件 Back 失败，整个容器记为不可回退，跨任务持久；
  - 副作用记忆：恢复后内容变化或改动了输入框的控件，记为有副作用。
- **跨任务复用的真实含义（务必准确）：**
  1. `fill_record` 所用的参数化函数（入口屏 → 导航 / 按字段点选 / 按字段输入 → 停在提交前，从成功轨迹中抽取，字段槽位按任务指令中的值识别）可以跨任务模板复用（同 app、同动词、字段名匹配）；
  2. 图中边（来自任何任务）用于探索价值估计、去重、可回退性学习和恢复路径。
  - **不要写成"跨任务直接重放多步导航路径"**：这一点目前没有实现成执行机制，是未来工作。

## 3. 图的对应关系与需要修改的地方

写正文时按下面的对应关系描述图中组件；正文后请单独列出"图需要修改的地方"，至少包括以下几条：

| 图中组件 | 正文中对应 | 需要修改 |
|---|---|---|
| 顶部 Reasoning → Trajectory Reuse → Action Execution | (A) | 复用发生在推理**之前**，应画成"图执行算子能落地 → 跳过推理；否则推理" |
| 右上 Prompt → Reasoning for Step t+1 | (A) | 只是动作历史（含图执行步骤），不是注入图信息 |
| Task-relevance Novelty Exploration with Rollback | (B) | 示例中的 "Add" 按副作用规则不会被探，应改为 Search 等导航控件并把 Add 标为被过滤；"Novelty" 应改为 graph-informed value × reversibility；补充 E1 / FX / D2 / RV 与恢复阶梯 |
| Resource Scheduler / Resource Monitor / Adaptive Exploration Budget | (B) | 保持图中设计 |
| CEG 中的 Cross-task Reuse（节点 2 → 5 → 6） | (C) | 不能表示跨任务重放导航路径；应表示为"其他任务的边用于价值 / 去重 / 恢复" |

## 4. 术语表（正文中保持一致）

- runtime graph / Complementary Evidence Graph (CEG)
- inference trajectory, exploration trajectory
- screen layer
- content-invariant page node, multi-granularity state signatures
- transition edge, provenance, reversibility evidence, destination capture
- container-level reversibility, side-effect memory
- parameterized function, field slot, stop before commit
- Resource Scheduler, Resource Monitor, Adaptive Exploration Budget
- graph execution operators: launch folding (`open_app`), commit confirmation (`confirm`), record filling (`fill_record`), form filling (`form`), in-episode reuse (`graph_reuse`)
- grounding on the live screen
- unified graph exploration value, evidence tiers (destination / neighbour / self), value backup
- probe types: E1 (predicted next), FX (frontier), D2 (depth-two), RV (re-observation)
- restore ladder, relaunch-and-replay

## 5. 输出格式

1. **System Overview 正文**（英文，可直接放入 LaTeX；用 \textbf{} 标出组件名，用 Figure~\ref{fig:overview} 引用图）；
2. **Figure 2 图注**（英文）；
3. **图需要修改的地方**（中文，逐条，指出位置和改法）；
4. **你认为仍不清楚、需要我补充的事实**（中文；不要自行编造）。
