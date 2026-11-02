# Fig.5 System Overview 修改稿（2026-11-02）

文件：`overview.pdf`（矢量，LaTeX 直接用）· `overview.svg`（可编辑）· `overview.png`（4× 预览）· `draw_overview.py`（生成源）· `render.js`（SVG→PDF/PNG）

## 1. 老师两条批注分别对应什么问题

| 批注 | 现图的具体问题 | 新图的处理 |
|---|---|---|
| “not clear, change to show how to enhance model reasoning and exploration. Do not show meaningless graph and UI everywhere” | 五个面板都画了手机截图缩略图和圆圈节点，但没有一个箭头说明传递的是什么数据；Reuse 面板里 “State i+1 → Action Prediction → Graph → ✓ → Trajectory Reuse” 看不出省掉了哪次推理；底部 Task1/Task2/Old/New Task 网图与 Fig.6 重复；Optimizer 里 “Model” 标在手机图标下、“UI” 标在 VLM 图标下（看起来标反了）；“Step i / Step t” 混用 | 全图只保留**一个**输入截图（控件编号 1–4 贯穿全图）和**一个**有语义的小图（节点=页面，边=控件编号）。三层结构：上层 Reuse 写明“verified reuse skips the VLM call”（增强推理）；下层 reasoning window 写明“graph-guided probes while the VLM reasons”（增强探索）；中间 Graph 的每条读/写箭头都标注数据内容 |
| “leverage what info to measure novelty?” | 只有裸分数 0.95 / 0.42 / 0.17 和 “General Elements / Clickable Elements”，看不出分数来源 | Novelty 面板显式列出三类信息：**Novelty**（该 (state, control) 转移是否已在图中）、**Relevance**（任务词与控件文本/已知目的页的 TF-IDF）、**Recovery**（历史恢复结果与副作用属性）；再用 4 个控件的小表展示 probe / skip / filter 各因哪一项决定。Graph→Exploration 的箭头标注 `coverage · destinations · recovery history` |

## 2. 新图怎么读（建议的 caption）

> **Figure 5: MobileExplorer at decision step *i*.** (Top) Before any model call, Mix-Trajectory Reuse retrieves a recorded edge or procedure from the graph, grounds it on the live UI, and verifies the resulting state; a hit advances the task without invoking the VLM. (Bottom) On a miss, the VLM reasons on the captured screen while the explorer probes the live UI. A control is probed only if its transition is absent from the graph, it is ranked by TF-IDF relevance between task terms and its text and known destinations, and it is admitted by its recovery history; the adaptive runtime optimizer fits the probes into the reasoning window. (Middle) Executed and explored transitions, with their recovery outcomes, accumulate in the cross-task graph that serves both paths. Controls 1–4 and bar lengths are illustrative.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/overview.pdf}
  \caption{...}
  \label{fig:overview}
\end{figure*}
```

## 3. “用什么信息衡量 novelty”：图与实现的对应（FULL45，`full_run.py` / `explore_value.py`）

| 图中项 | 实现 | 作用 |
|---|---|---|
| Novelty | `full9.page_family_index(graph)`：(package, activity, control family) 已被 agent 或任一任务的探索记录过就排除；另排除 `_learned_eff`、`_bad_groups` | **过滤条件**（深度 1 的 β=0，不是打分项） |
| Relevance | `GraphValue.value`：TF-IDF cosine(任务的操作/对象词, 证据文本)。证据按优先级：已知边的目的页（conf 0.8）→ 图中同 label/rid 控件在别处的目的页（conf 0.5）→ 控件自身 label + rid + 邻近 caption。任务侧去掉 app 名、引号字面量、数字和解析出的参数值；UI 侧 IDF = 全局（以 app 为文档）× 图内本 app 页面 × 本 episode 页面的几何平均 | 排序分数 |
| Recovery | `p_reversible_h`：由图中同容器/同类控件的历史恢复结果估计 P_rev，需 ≥ 0.8（新建入口 0.7）；`_risky_slot` 排除风险词（setting/search/edit/delete…）、toggle/checkable、上一步刚操作的控件 | 过滤 + 惩罚 κ(1−P_rev) |

建议在 §5.3 补一个显式定义（把正文和图里的 *q<sub>j</sub>* 对上，直接回答老师的问题）：

```latex
q_j = \mathrm{rel}(\mathcal{G}, d_j) - \kappa\,(1-\hat{p}^{\mathrm{rec}}_j),\qquad
j \in \mathcal{C}_i = \{\, j : (v_i, a_j) \notin \mathcal{E}_i,\ \hat{p}^{\mathrm{rec}}_j \ge \theta,\ a_j \notin \mathcal{A}^{\mathrm{risk}} \,\}
```

其中 rel 是上表的 TF-IDF 相关度，d<sub>j</sub> 是控件 *j* 的文本证据（自身文本，以及图中已观察到的目的页面）。

## 4. 需要和正文/其他图对齐的地方

1. **命名**：正文是 “Mix-Trajectory”，旧图是 “Mixed-trajectory”；§5.3 标题 “Graph-Guided Exploration” 与组件名 “Novelty-Seeking UI Exploration” 不一致。新图统一用 Mix-Trajectory / Novelty-Seeking Exploration。
2. **步编号**：§5.1 用 *i*，§5.5 的 Eq.(3) 用 *t*，旧图两者都有。新图统一为 *i*，建议正文也统一。
3. **Fig.7 与实现不一致**：Fig.7 中 Search、Setting 通过了 risk filtering，但 FULL44/45 的 `_RISKY_WORDS` 含 `setting`、`search`，实际从不探测。建议 Fig.7 也把它们标为过滤，或在正文说明。
4. **Optimizer**：新图按正文 Eq.(3) 画。“explore more ↔ infer faster” 滑块同时对应正文里“给探索更多资源 / 偏向推理”和 `rto.plan` 的 explore_more / infer_faster 两种模式。若 §5.5 改写为 `rto.py` 的 graph-verified fast decision，可在图中加一条 Graph → VLM 的 “verify fast action” 读接口。这是“探索增强推理”更直接的证据：e2e116R 中 166 次被接受的快速决策里，有 31 次用到了探索得到的边。
5. 例子（Notes app，控件 1–4、Task A）是机制示意，不是日志中的真实 episode；条形长度不是测量值。投稿前可换成真实截图，只要保持“一个截图 + 编号”的用法。

## 5. 修改与重新生成

- 小改动：直接编辑 `overview.svg`（Inkscape / Illustrator；PowerPoint 插入 SVG 后“转换为形状”即可逐个编辑）。图中只用了基本形状和文字，没有 CSS 和 marker，导入后不会丢样式。字体为 Liberation Sans，与 Arial 等宽，在 Mac/Windows 上会落到 Arial。
- 结构性改动：改 `draw_overview.py` 后运行

```bash
python3 draw_overview.py && node render.js   # 需要 playwright（Chromium）
```

字号：标题 13.5、正文 10、最小 9（SVG 单位）。按 \textwidth = 7 in 排版时约为 6.8 / 5.0 / 4.5 pt，与原图相当或略大。
