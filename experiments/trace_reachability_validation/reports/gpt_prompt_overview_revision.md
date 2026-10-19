你是一名移动系统 / GUI agent 方向的论文写作助手。下面是我论文（SenSys，系统名 \name = MobileExplorer）现有的 System Overview 段落。请**在原文基础上做小幅修改**，主要是调整叙述顺序，不要大幅改写。

---

## 1. 修改要求

1. **保留原文的措辞、术语和主张**：progressive belief graph、predictive graph exploration、certainty-aware evidence distillation、dynamically adapted exploration budget 等名称都保持不变。能移动句子就移动，能删改几个词就不要重写整句。
2. **按下面四部分重新组织**（导师建议的顺序），每部分一段，可以加一句承上启下的过渡句：
   1. **系统在运行时如何导航 UI**：传统的感知—推理—操作串行流程，以及 \name 如何在模型推理窗口内并行探索选定的 UI 分支（候选选择 → 投机执行 → 轻量恢复与校验）；
   2. **系统如何在运行时探索中有效地组织和存储知识**：progressive belief graph 记录什么（UI 状态、投机转移、模型决策、执行结果、恢复反馈、动作是否改变了持久 app 状态），以及如何随交互逐步校准；
   3. **系统如何利用这些知识服务后续决策**，分三点写：
      - (a) 引导后续探索：已探分支引导之后的探索；任务实体已知后，探索转向尚未处理的实体；
      - (b) 跳过推理：安全且充分验证的后续动作直接复用，执行后立即校验；对多实体重复任务做过程级复用，停在第一个改变 app 状态的动作之前；
      - (c) 增强推理：certainty-aware evidence distillation；
   4. **如何在设备上有效协调计算任务**：根据设备运行时资源可用性以及与并发模型推理的干扰，在线调整探索预算。
3. 第一段开头的总述（"In this paper, we propose ..." 以及与传统 GUI agent 的对比、Fig.~\ref{fig:system_overview} 的引用）保留在最前面；最后一句 "Overall, ..." 的总结保留在最后。
4. 修改量控制在原文的 20% 以内。总长度与原文相当，可 ±15%。
5. 保持 LaTeX 格式（\name、\textit{}、Fig.~\ref{}），不要新增未定义的宏。

## 2. 需要你标出、但不要自行改动的地方

以下几处，原文的表述与当前系统实现或 overview 图可能不一致。**请保持原文不变**，只在输出的第 3 部分逐条列出这些句子，供我决定：

1. **注入（certainty-aware evidence distillation）：** 原文写"将局部图蒸馏成紧凑事实并注入下一次模型输入"。当前最终实现不向 prompt 注入图信息：实验中各种注入 / 高亮形式对成功率和步数都没有显著作用，收益来自直接执行替代推理。
2. **"unresolved information extracted from previous reasoning"：** 当前实现的探索候选打分不解析模型的推理文本，用的是：
   - 任务操作词与图中证据文本（已知边目的页 > 图中近邻 > 控件自身文本）的 TF-IDF 相关性；
   - 图中已知性去重；
   - 可回退性预测；
   - 副作用排除。
3. **跳过推理的覆盖面：** 原文只写了"验证过的后续动作复用"和"多实体任务的过程级复用"。当前实现中跳过推理还包括：
   - 从任务指令中的 app 名直接启动 app；
   - 模型自己提交后自动点确认对话框；
   - 从成功轨迹抽取的参数化函数跨任务复用（按任务字段值填写记录 / 表单）。
4. **名称与图不一致：**
   - 正文叫 progressive belief graph，overview 图里叫 Complementary Evidence Graph (CEG)；
   - 正文叫 predictive graph exploration，图里叫 Task-relevance Novelty Exploration with Rollback。
5. **恢复细节：** 原文只写"lightweight state recovery and verification"。当前实现是一个恢复阶梯：再点一次 / 选回原标签 → Back → 沿图中已知边恢复 → 强制停止 app 并按本局记录的路径重启重放（仅当路径不含提交）。是否需要在 overview 中点到，由我决定。

## 3. 输出格式

1. **修改后的段落**（LaTeX，可直接替换原文）；
2. **修改说明**：逐条列出移动或改动了哪些句子，以及原因；
3. **待作者确认的不一致**：按第 2 节逐条列出原句，不做修改；
4. 如有其他你认为逻辑不通或前后矛盾的地方，单独列出（中文）。

---

## 4. 原文

```latex
In this paper, we propose \name, a new on-device mobile GUI agent framework that exploits the long reasoning latency of models (LLMs/VLMs) to perform lightweight, parallel exploration and proactively acquire information for future decisions.
As illustrated in Fig.~\ref{fig:system_overview}, unlike conventional GUI agents that follow a sequential perception--reasoning--operation pipeline and remain idle during model inference, \name proactively explores selected UI branches within the model reasoning window and organizes the resulting transitions into a \textit{progressive belief graph}.
The graph incrementally captures UI states, speculative transitions, model decisions, execution outcomes, and recovery feedback, together with whether each executed action changed persistent app state, allowing exploration knowledge to be progressively calibrated and reused across subsequent interaction steps.

Specifically, to identify informative interactions under a dynamically adapted exploration budget, \name employs \textit{predictive graph exploration}, where the exploration budget is adjusted online according to the mobile device's runtime resource availability and interference with concurrent model inference.
At each UI state, it jointly considers the current interface, accumulated graph history, unresolved information extracted from previous reasoning, recovery reliability, and exploration cost to estimate which candidate interactions are most likely to reveal information useful for future decisions.
Selected candidates are executed speculatively and followed by lightweight state recovery and verification.
The resulting observations are then incorporated back into the progressive belief graph, together with later model alignment and execution feedback, so that previously explored branches can progressively guide future exploration rather than being treated as isolated observations. Once the task entities are known, exploration is further directed toward the entities that the task has not yet handled.

Finally, \name incorporates \textit{certainty-aware evidence distillation} to determine how exploration knowledge should support subsequent reasoning.
When the graph contains task-relevant but insufficiently certain evidence, \name deterministically distills the local graph into a small set of compact, state-aligned facts and injects them into the next model input without invoking an additional LLM/VLM.
When the graph instead identifies a safe and sufficiently verified continuation with low decision uncertainty, \name directly reuses the corresponding action to bypass redundant model inference, followed by immediate execution verification; weak, stale, or irrelevant evidence is discarded.
For tasks that repeat one procedure over several entities (e.g., deleting several recipes), the same bypass is applied at the procedure level: \name reuses the navigation steps that the model took for one entity on the next entity, and stops before the first action that changes app state, which is left to the model.
Overall, by using the progressive belief graph to both guide future exploration and support or bypass subsequent reasoning, \name reduces unnecessary trial-and-error interactions and redundant model invocations, ultimately lowering the end-to-end latency of on-device mobile GUI agents.
```
