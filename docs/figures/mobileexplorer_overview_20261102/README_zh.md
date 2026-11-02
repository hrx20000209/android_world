# Fig.5 System Overview 修改稿（2026-11-02）

| 版本 | 文件 | 说明 |
|---|---|---|
| **v3（推荐）** | `overview_v3.pdf` / `.svg` / `.png`，`draw_overview_v3.py` | 在原 overview 上修改：布局、配色、图标沿用原图，重做 exploration 和 reuse 两个面板 |
| v2 | `overview_v2.*`，`draw_overview_v2.py` | 概念级时间线版（与 MMEdge 风格过近，仅作参考） |
| v1 | `overview.*`，`draw_overview.py` | 细节版；其中的候选表和时间线可拆到 Fig.6–9 |

渲染：`python3 draw_overview_v3.py && node render.js overview_v3.svg`（需要 playwright）。

## 1. v3 相对原图改了什么

**保留**：五个面板的位置（Input │ Adaptive Runtime Optimizer │ Model Reasoning + Novelty-Seeking Parallel Exploration │ Mix-Trajectory Reuse，下方 Cross-task Graph）；粉/淡紫/绿底色、芯片与资源条、橙色 chevron、推理横幅、红色截止线、元素标签和分数扇形。

**Novelty-Seeking Parallel Exploration**（主要改动）
- State *i* 上的标签按“图中是否见过”着色：Search、Setting 为灰色（见过），Add 为橙色并带星标（从未见过）。
- 每个候选旁标出 seen count：*n* = 0 / 1 / 3。图面板有一条橙色箭头 *seen count n* 指向本面板，说明 *n* 来自图。
- 分数 0.95 / 0.42 / 0.17 保留，但现在可以解释：**Score = Novelty × Task Relevance，Novelty = 1 / (1 + *n*)**。
- 分数最高的 Add 被 *probe*，然后 *Back*。图里对应的未探索边画成橙色虚线 “?”（unseen frontier）。橙色在全图统一表示“新”。

**Mix-Trajectory Reuse**（对应老师第一条批注）
- 改为：State *i*+1 + Task 查询 → retrieve 一条 **Mixed Trajectory**（实线来自推理，橙色虚线来自探索）→ 在 live UI 上 verified → 直接执行（open / click / type）→ **skip reasoning for 3 steps**。
- 去掉原面板里重复的两张截图和图节点条。

**其他**
- Optimizer：原图中 “Model” 写在手机图标下、“UI” 写在 VLM 图标下，已改正。
- 推理横幅：Step *t* 统一为 Step *i*。
- 名称：Mixed-trajectory 统一为 Mix-Trajectory。
- 底部图：去掉轨迹截图和 Old/New Task 示意，改为节点轨迹 → merge states → 一张合并图；标出 *reused* 路径和 *unseen* 边，分别与上方的 reuse 和 exploration 对应。
- 截图是线框示意，可以直接换回你原来的真实截图，位置不变。

## 2. Novelty 怎么度量（建议写进 §5.3）

对 State *i* 上的候选元素 *e*：

```latex
N_i(e) =
\begin{cases}
0, & (v_i, e) \in \mathcal{E}_i \quad \text{(this transition is already recorded: skip)}\\[2pt]
\dfrac{1}{1 + n(e)}, & \text{otherwise}
\end{cases}
\qquad
q_i(e) = N_i(e)\cdot R(e, \mathcal{G}), \quad e \in \mathcal{C}_i^{\mathrm{safe}}
```

- *n*(*e*)：图中与 *e* 同 label / resource-id 的元素在其他状态、app、任务里已被执行或探索过的次数。通用元素（Search、Settings）*n* 大、novelty 低；本 app 特有的元素 novelty 高。
- *R*：任务描述与元素文本（及已知目的页）的 TF-IDF 相关度。
- *C*<sup>safe</sup>：去掉高风险、历史上恢复失败的元素。
- 图中的数字按此计算：1.0×0.95、0.5×0.84、0.25×0.68，即 0.95 / 0.42 / 0.17（示意值）。

**与当前实现的差异**（FULL45，`full_run.py` / `explore_value.py`）：
- 第一行（本状态已记录则跳过）已实现：`page_family_index`。
- `n(e)` 的计数也已实现：`GraphValue.novelty = 1/(1+seen[label])`。但它以 **加法** `value + β·novelty` 进入分数，且深度 1 时 β = 0，只在深度 2 时 β = 0.3。
- 通用元素被降权，目前主要靠 TF-IDF 中的 IDF。
- 若论文采用上面的乘法形式，需要相应改代码（深度 1 也启用 novelty，改成乘法）。另一种做法是把公式写成实现的形式：*q* = *R* + β·*N*，并说明 IDF 的作用。

## 3. 建议的 caption

> **Figure 5: System overview of MobileExplorer.** While the model reasons on State *i*, MobileExplorer explores the live UI in parallel. Novelty-seeking exploration scores each clickable element by its novelty, 1/(1+*n*), where *n* is how often the element has been seen in the cross-task Mix-Trajectory Graph, weighted by its task relevance, and probes the highest-scoring element before rolling back. Mix-trajectory reuse retrieves recorded trajectories that mix reasoning and exploration transitions, verifies them on the live UI, and executes them without model reasoning. The adaptive runtime optimizer maximizes exploration within the reasoning latency.

## 4. 需要和正文/其他图对齐的地方

1. §5.3 标题 “Graph-Guided Exploration” 与组件名 Novelty-Seeking (UI) Exploration 不一致。
2. §5.4 正文写的是“用当前状态和任务目标查询”，原图 reuse 面板却用 *Action Prediction (90, 1800)* 查询。v3 已按正文改为 State *i*+1 + Task。
3. Fig.7 中 Search、Setting 通过了 risk filtering，但 FULL44/45 的 `_RISKY_WORDS` 含 `setting`、`search`，实际从不探测。Fig.5 只表达 novelty 排序；Fig.7 改版时建议和代码一致。
4. 若 §5.5 改写为 `rto.py` 的 graph-verified fast decision，可在 reuse 面板旁加一条 “verify fast action” 的说明（e2e116R：被接受的 166 次快速决策中有 31 次用到了探索得到的边）。

## 5. 编辑

- 直接编辑 SVG（Inkscape / Illustrator；PowerPoint 插入后“转换为形状”）。只用了基本形状、文字和线性渐变，没有 CSS 和 marker。字体为 Liberation Sans（与 Arial 等宽）。
- 尺寸 1000×410 单位，与原图比例相同（约 2.44:1）。按 \textwidth = 7 in 排版约 7 × 2.9 in。
