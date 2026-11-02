# Fig.5 System Overview 修改稿（2026-11-02）

| 版本 | 文件 | 用途 |
|---|---|---|
| **v2（推荐，放 Fig.5）** | `overview_v2.pdf` / `.svg` / `.png`，`draw_overview_v2.py` | 概念级 overview，按 MMEdge Fig.4 的风格画 |
| v1（细节版） | `overview.pdf` / `.svg` / `.png`，`draw_overview.py` | 不放 overview；里面的候选表、读写接口、时间线可拆到 Fig.6–9 |

`render.js` 把任一 SVG 转成矢量 PDF 和 4× PNG：`node render.js overview_v2.svg`。

## 1. v2 的画法（参照 MMEdge Fig.4）

MMEdge 的 overview 只有一个核心画面：两条传感流沿时间轴的 pipeline。三个模块直接画在它作用的那段 pipeline 上，标签只有 1–3 个词，唯一的“公式”是 *Max Accuracy s.t. Latency Constraint*，机制细节都留给正文和模块图。v2 采用同样的组织方式：

- **核心画面：Parallel Reasoning and Exploration**。上面是 VLM 泳道，下面是 Live UI 泳道。Step *i* 中，模型推理期间 UI 上同时在做探索；Step *i*+1 中复用命中，VLM 泳道显示为 *Reasoning skipped*。
- **Novelty-Seeking Exploration** 框住 Step *i* 的探索段（探测 → 新页面 → 返回），底部三个标签 *Unseen / Task-relevant / Recoverable*，直接回答“用什么信息衡量 novelty”。
- **Mix-Trajectory Reuse** 框住 Step *i*+1：菱形 *Reuse?*，Yes 走 recorded action，No 回到 reason & explore（对应 MMEdge 的 *Skip?*）。
- **Cross-task Mix-Trajectory Graph** 只出现一次：实线为推理执行的转移，虚线为探索得到的转移。两处高亮标注 *explored at step i* 和 *reused at step i+1*，和上面的时间线对应。四条短箭头标注 *coverage / record / record / retrieve*。
- **Adaptive Runtime Optimizer** 放在左下（同 MMEdge 的 Adaptive Multimodal Configuration），沿用原图的 Reasoning 条、Exploration 箭头和红色截止线，标语为 *Max Exploration s.t. Reasoning Latency*，并修正了原图 Model/UI 标反的问题。
- 截图只出现在输入框和 Live UI 泳道上，每张都代表时间线上的一个状态，不再到处放缩略图。

## 2. 老师两条批注 → v2 的对应

| 批注 | v2 怎么回答 |
|---|---|
| show how to enhance model reasoning and exploration; no meaningless graph and UI everywhere | 增强探索：探索放在推理窗口里并行进行（同一个 Step *i* 列），并由图的 coverage 引导。增强推理：复用命中时 Step *i*+1 的推理被跳过。图只画一次，并用高亮把“这一步探索到的”和“下一步复用的”连到时间线上 |
| leverage what info to measure novelty? | 三个标签：*Unseen*（图中尚无该转移，信息来自图的 coverage 箭头）、*Task-relevant*（任务描述与控件文本）、*Recoverable*（历史恢复结果） |

## 3. 建议的 caption

> **Figure 5: System overview of MobileExplorer.** While the VLM reasons on the current screen, MobileExplorer explores unseen, task-relevant, and recoverable UI controls on the live UI and records both executed and explored transitions in a cross-task Mix-Trajectory Graph. Mix-trajectory reuse retrieves recorded actions from the graph to skip model reasoning at later steps, and an adaptive runtime optimizer maximizes exploration within the reasoning latency.

## 4. 从 overview 移出的细节应放在哪里

| 细节 | 建议位置 |
|---|---|
| TF-IDF 证据（控件文本、已知目的页、同 label/rid 目的页）、恢复阈值、候选筛选表 | §5.3 正文 + Fig.7（v1 的四控件表可直接改成 Fig.7） |
| 节点键 (app, activity, layout signature)、边字段、procedure 参数槽 | §5.2 + Fig.6 |
| Query → Retrieve → Ground → Verify | §5.4 + Fig.8 |
| Eq.(3)、restore / check / 流式执行的时间线 | §5.5 + Fig.9 |

§5.3 建议补一个显式定义，同时对应老师的问题和图中的三个标签：

```latex
q_j = \mathrm{rel}(\mathcal{G}, d_j) - \kappa\,(1-\hat{p}^{\mathrm{rec}}_j),\qquad
j \in \mathcal{C}_i = \{\, j : (v_i, a_j) \notin \mathcal{E}_i,\ \hat{p}^{\mathrm{rec}}_j \ge \theta,\ a_j \notin \mathcal{A}^{\mathrm{risk}} \,\}
```

与 FULL45 实现的对应（`full_run.py` / `explore_value.py`）：

| 标签 | 实现 |
|---|---|
| Unseen | `page_family_index(graph)`：(package, activity, control family) 已被 agent 或任一任务的探索记录过就排除（是过滤条件，深度 1 的 β=0） |
| Task-relevant | `GraphValue.value`：任务操作/对象词与证据文本的 TF-IDF cosine。证据依次为已知边的目的页（conf 0.8）、同 label/rid 控件的目的页（0.5）、控件自身文本 |
| Recoverable | `p_reversible_h`：由图中同容器/同类控件的恢复历史估计 P_rev（≥ 0.8，新建入口 0.7），加上 `_risky_slot` 风险过滤和 κ(1−P_rev) 惩罚 |

## 5. 需要和正文/其他图对齐的地方

1. **命名**：统一为 Mix-Trajectory（旧图写的是 Mixed-trajectory）；§5.3 的标题 “Graph-Guided Exploration” 与组件名 Novelty-Seeking Exploration 不一致。
2. **步编号**：§5.1 用 *i*，Eq.(3) 用 *t*；新图统一为 *i*。
3. **Fig.7 与实现不一致**：Fig.7 中 Search、Setting 通过了 risk filtering，但 FULL44/45 的 `_RISKY_WORDS` 包含 `setting`、`search`，实际从不探测。
4. **可选**：如果 §5.5 改写成 `rto.py` 的 graph-verified fast decision，可以在 v2 加一条 Graph → VLM 的短箭头（如 *verify fast action*）。这是“探索增强推理”最直接的证据：e2e116R 中 166 次被接受的快速决策里，有 31 次用到了探索得到的边。
5. 截图是示意用的线框图；投稿前可以换成真实截图，但只换输入框和 Live UI 泳道上的那几张。

## 6. 修改与重新生成

- 小改：直接编辑 SVG（Inkscape / Illustrator；PowerPoint 插入 SVG 后“转换为形状”）。只用了基本形状、文字和两个线性渐变，没有 CSS 和 marker。字体为 Liberation Sans（与 Arial 等宽）。
- 大改：改 `draw_overview_v2.py`，然后运行 `python3 draw_overview_v2.py && node render.js overview_v2.svg`（需要 playwright）。
- 尺寸：1000×376 单位，按 \textwidth = 7 in 排版约 7 × 2.6 in，与 MMEdge Fig.4 相当。字号：大标题 15、模块标题 12.5–13、标签 9.5–11（约 4.8–7.6 pt）。
